# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Two live collections and updates with one resident trainer, without RL."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from verl_speco.integration.ascend_state_config import hccl_environments
from verl_speco.probe_deepseek_v4_training_cycle import (
    prompt_from_trajectory,
    run_stage,
)
from verl_speco.trainer.deepseek_v4_resident_probe import atomic_json


def validate_worker_rounds(rounds, world_size, *, production_lifecycle=False):
    if len(rounds) != 2:
        raise RuntimeError("Exactly two resident rounds are required")
    previous = {}
    for revision, report in enumerate(rounds, 1):
        workers = report["workers"]
        if (
            report.get("status") != "passed"
            or {row["rank"] for row in workers} != set(range(world_size))
            or len(workers) != world_size
        ):
            raise RuntimeError("Incomplete resident worker acknowledgements")
        for row in workers:
            if (
                production_lifecycle
                and row.get("state_lifecycle") != "production_hooks"
            ):
                raise RuntimeError("Worker did not use production lifecycle hooks")
            if (
                row["round"] != revision
                or row["global_step"] != revision
                or row["state"] != "parked"
                or not row["optimizer_steps"]
                or any(step != revision for step in row["optimizer_steps"])
            ):
                raise RuntimeError("Resident optimizer progress differs")
            if revision == 2:
                old = previous[row["rank"]]
                if (
                    not row["wake_verified"]
                    or row["pid"] != old["pid"]
                    or row["identities"] != old["identities"]
                ):
                    raise RuntimeError(
                        "Trainer/optimizer was replaced instead of reused"
                    )
            previous[row["rank"]] = row


def plan(args):
    root = Path(args.output)
    trainer_base = getattr(args, "trainer_hccl_port_base", 61000)
    serving_base = getattr(args, "serving_hccl_port_base", 62000)
    trainer_hccl, serving_hccl = hccl_environments(trainer_base, serving_base)
    shared = [
        "--model",
        args.model,
        "--trajectories",
        args.trajectories,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--response-tokens",
        str(args.response_tokens),
        "--trainer-hccl-port-base",
        str(trainer_base),
        "--serving-hccl-port-base",
        str(serving_base),
    ]
    serve = []
    for revision in range(3):
        snapshots = [
            str(root / f"snapshots/revision_{r}") for r in range(1, revision + 1)
        ]
        command = [
            sys.executable,
            "-m",
            "verl_speco.probe_deepseek_v4_online_rounds",
            *shared,
            "--stage",
            "serve",
            "--serving-revision",
            str(revision),
            "--output",
            str(root / f"serving_{revision}"),
        ]
        for snapshot in snapshots:
            command.extend(["--snapshot", snapshot])
        serve.append(
            {
                "name": f"serving_{revision}",
                "command": command,
                "environment": serving_hccl,
            }
        )
    trainer = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.tensor_parallel_size * args.data_parallel_size}",
        "-m",
        "verl_speco.trainer.deepseek_v4_resident_probe",
        "--model",
        args.model,
        "--output",
        str(root),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]
    if getattr(args, "production_lifecycle", False):
        trainer.append("--production-lifecycle")
    return {
        "repo": str(Path(__file__).resolve().parents[1]),
        "environment": {},
        "serving_stages": serve,
        "trainer_command": trainer,
        "trainer_environment": {
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            **trainer_hccl,
        },
    }


def wait_json(path, process, timeout=1800):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if process.poll() is not None:
            raise RuntimeError(
                f"Resident trainer exited with code {process.returncode}; inspect trainer.log"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"Resident trainer timed out waiting for {path}")
        time.sleep(0.2)
    return json.loads(Path(path).read_text())


async def serving_round(actor, args, prompt, contract):
    from verl_speco.probe_deepseek_v4_server_update import run_probe

    if args.serving_revision < 2:
        raise RuntimeError("Rollout hidden-state capture has been removed")
    await actor.set_global_steps.remote(0)
    updates = []
    for revision, snapshot in enumerate(args.snapshot, 1):
        metadata = json.loads(
            (Path(snapshot) / "speco_runtime_snapshot.json").read_text()
        )
        if (
            metadata["contract"] != contract.identity()
            or metadata["drafter_revision"] != revision
            or metadata["target_version"] != 0
            or metadata["training_tp"] != args.tensor_parallel_size
            or metadata["training_dp"] != args.data_parallel_size
            or metadata["recovery_checkpoint"]
        ):
            raise ValueError("Runtime snapshot version/contract/topology differs")
        updates.append(
            await run_probe(
                actor,
                SimpleNamespace(
                    checkpoint=snapshot,
                    tensor_parallel_size=args.tensor_parallel_size,
                    data_parallel_size=args.data_parallel_size,
                    ipc_timeout=600,
                    revision=revision,
                    skip_probe_collection=True,
                ),
                prompt,
            )
        )
    return {
        "status": "passed",
        "target_version": 0,
        "drafter_revision": args.serving_revision,
        "updates": updates,
    }


def serve(args, prompt, contract):
    import ray

    from verl_speco.integration.deepseek_v4_server_probe import (
        create_server_actor_class,
    )

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    count = args.tensor_parallel_size * args.data_parallel_size
    _, hccl_env = hccl_environments(
        args.trainer_hccl_port_base, args.serving_hccl_port_base
    )
    actor = None
    try:
        ray.init(
            address="local",
            num_cpus=4,
            include_dashboard=False,
            object_store_memory=512 * 1024**2,
        )
        if ray.cluster_resources().get("NPU", 0) < count:
            raise RuntimeError("Ray did not detect required NPUs")
        options = {
            "model": args.model,
            "tp_size": args.tensor_parallel_size,
            "dp_size": args.data_parallel_size,
            "max_model_len": args.max_model_len,
            "devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            "hccl_environment": hccl_env,
        }
        if args.serving_revision < 2:
            options["feature_store"] = str(root / "features")
        actor = (
            create_server_actor_class()
            .options(
                num_cpus=2, resources={"NPU": count}, runtime_env={"env_vars": hccl_env}
            )
            .remote(options)
        )
        ray.get(actor.launch_server.remote(), timeout=1800)
        from verl_speco.probe_deepseek_v4_draft_update import require_phase

        workers = ray.get(actor.probe_rpc.remote("speco_standalone_hccl_environment"))
        require_phase(
            workers, "configured", args.tensor_parallel_size, args.data_parallel_size
        )
        if any(row["environment"] != hccl_env for row in workers):
            raise RuntimeError(
                "Serving workers did not inherit the isolated HCCL ports"
            )
        report = asyncio.run(serving_round(actor, args, prompt, contract))
        report["hccl_environment"] = hccl_env
        report["worker_hccl_environments"] = workers
        atomic_json(root / "serving_report.json", report)
    finally:
        try:
            if actor is not None:
                try:
                    ray.get(actor.probe_shutdown.remote(), timeout=30)
                except Exception as cleanup:  # noqa: BLE001 -- still terminate our actor
                    print(f"Serving shutdown failed: {cleanup!r}", flush=True)
                finally:
                    ray.kill(actor, no_restart=True)
        finally:
            ray.shutdown()


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    invocation = plan(args)
    atomic_json(root / "plan.json", invocation)
    process = None
    rounds = []
    try:
        run_stage(invocation["serving_stages"][0], invocation, root)
        with (root / "trainer.log").open("w") as log:
            process = subprocess.Popen(
                invocation["trainer_command"],
                cwd=invocation["repo"],
                env={
                    **os.environ,
                    **invocation["trainer_environment"],
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_json(root / "trainer_ready.json", process)
                for revision in (1, 2):
                    atomic_json(
                        root / f"command_{revision:02d}.json",
                        {
                            "operation": "train",
                            "round": revision,
                            "store": str(
                                root
                                / f"serving_{revision - 1}/features/target_step_00000000"
                            ),
                            "snapshot": str(root / f"snapshots/revision_{revision}"),
                        },
                    )
                    print(
                        f"Waiting for resident training round {revision}; log: {root / 'trainer.log'}",
                        flush=True,
                    )
                    rounds.append(
                        wait_json(root / f"result_{revision:02d}.json", process)
                    )
                    run_stage(invocation["serving_stages"][revision], invocation, root)
                atomic_json(root / "command_03.json", {"operation": "stop"})
                if process.wait(timeout=180) != 0:
                    raise RuntimeError("Resident trainer shutdown failed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        validate_worker_rounds(
            rounds,
            args.tensor_parallel_size * args.data_parallel_size,
            production_lifecycle=bool(getattr(args, "production_lifecycle", False)),
        )
        final = json.loads((root / "serving_2/serving_report.json").read_text())
        if final["status"] != "passed" or [
            r["drafter_revision"] for r in final["updates"]
        ] != [1, 2]:
            raise RuntimeError("Final serving revision sequence differs")
        report = {
            "status": "passed",
            "target_version": 0,
            "drafter_revisions": [0, 1, 2],
            "optimizer_steps": [1, 2],
            "resident_trainer_process_validated": True,
            "production_lifecycle_hooks_validated": bool(
                getattr(args, "production_lifecycle", False)
            ),
            "engine_optimizer_scheduler_reused": True,
            "address_preserving_park_wake_validated": True,
            "rollout_server_restarted_between_training_rounds": True,
            "rl_integration_validated": False,
            "full_checkpoint_saved": False,
            "runtime_snapshots": [
                str(root / f"snapshots/revision_{r}") for r in (1, 2)
            ],
        }
        atomic_json(root / "online_rounds_report.json", report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        atomic_json(
            root / "failure.json",
            {
                "status": "failed",
                "completed_training_rounds": len(rounds),
                "error": repr(error),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "trajectories", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--trainer-hccl-port-base", type=int, default=61000)
    parser.add_argument("--serving-hccl-port-base", type=int, default=62000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--production-lifecycle", action="store_true")
    parser.add_argument(
        "--stage", choices=["run", "serve"], default="run", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--serving-revision", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--snapshot", action="append", default=[], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    parser.error(
        "This rollout-capture probe was retired. Use actor old-logprob capture "
        "for online training or replay_deepseek_v4_features for standalone data."
    )
    try:
        hccl_environments(args.trainer_hccl_port_base, args.serving_hccl_port_base)
    except ValueError as error:
        parser.error(str(error))
    from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

    args.model = str(Path(args.model).resolve(strict=True))
    args.trajectories = str(Path(args.trajectories).resolve(strict=True))
    args.output = str(Path(args.output).resolve())
    if (
        args.tensor_parallel_size < 1
        or args.data_parallel_size < 1
        or args.response_tokens < 16
    ):
        parser.error("Positive TP/DP and at least 16 response tokens are required")
    if (
        args.serving_revision not in (0, 1, 2)
        or len(args.snapshot) != args.serving_revision
    ):
        parser.error("Serving revision must match the ordered runtime snapshot chain")
    contract = DeepSeekV4Contract.from_model(args.model)
    prompt = prompt_from_trajectory(
        args.trajectories,
        vocab_size=contract.vocab_size,
        max_model_len=args.max_model_len,
        response_tokens=args.response_tokens,
    )
    if Path(args.output).exists():
        parser.error("Use a fresh output directory")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return
    devices = [
        x.strip()
        for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if x.strip()
    ]
    if len(set(devices)) != args.tensor_parallel_size * args.data_parallel_size or len(
        devices
    ) != len(set(devices)):
        parser.error("Set exactly TP*DP distinct visible NPUs")
    if args.stage == "serve":
        serve(args, prompt, contract)
    else:
        run(args)


if __name__ == "__main__":
    main()
