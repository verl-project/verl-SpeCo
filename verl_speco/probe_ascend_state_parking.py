# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Small NPU allocator/ZeRO lifecycle probe; no target weights, features or RL."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ALLOCATOR_CONF = "expandable_segments:False"
ENVIRONMENT_KEYS = (
    "PYTORCH_NPU_ALLOC_CONF",
    "ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "HCCL_HOST_SOCKET_PORT_RANGE",
)


def launch_plan(args, environment=None):
    """Set the allocator policy before torchrun and fresh worker interpreters."""
    environment = dict(os.environ if environment is None else environment)
    environment["PYTORCH_NPU_ALLOC_CONF"] = ALLOCATOR_CONF
    environment.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61099")
    environment.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "61100-61199")
    count = args.tensor_parallel_size * args.data_parallel_size
    environment.setdefault(
        "ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, range(count)))
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={count}",
        "-m",
        "verl_speco.probe_ascend_state_parking",
        "--worker",
        "--output",
        str(Path(args.output).resolve()),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--rounds",
        str(args.rounds),
    ]
    if getattr(args, "state_storage", "memory") == "file":
        command.extend(["--state-storage", "file", "--chunk-mib", str(args.chunk_mib)])
    if getattr(args, "check_environment_only", False):
        command.append("--check-environment-only")
    return command, environment


def prepare_worker_environment(args, environment=None, modules=None):
    """Support an outer torchrun only while the worker is still pre-torch.

    torchrun launches fresh worker interpreters. Missing configuration can be
    installed here safely before importing torch; never change an already
    imported allocator or override an explicit conflicting configuration.
    """
    environment = os.environ if environment is None else environment
    modules = sys.modules if modules is None else modules
    actual = environment.get("PYTORCH_NPU_ALLOC_CONF")
    source = "inherited"
    if actual is None:
        if any(name in modules for name in ("torch", "torch_npu")):
            raise RuntimeError(
                "PYTORCH_NPU_ALLOC_CONF=None after torch/torch_npu import; "
                "restart through the direct Python launcher"
            )
        environment["PYTORCH_NPU_ALLOC_CONF"] = ALLOCATOR_CONF
        source = "worker_bootstrap"
    elif actual != ALLOCATOR_CONF:
        raise RuntimeError(
            f"PYTORCH_NPU_ALLOC_CONF={actual!r}; expected {ALLOCATOR_CONF!r}. "
            "Restart through the direct Python launcher"
        )
    environment.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61099")
    environment.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "61100-61199")
    count = args.tensor_parallel_size * args.data_parallel_size
    environment.setdefault(
        "ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, range(count)))
    )
    return source


def allocator_environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def launch(args):
    command, environment = launch_plan(args)
    print(
        json.dumps(
            {
                "phase": "launch_plan",
                "command": command,
                "environment": {key: environment.get(key) for key in ENVIRONMENT_KEYS},
            }
        ),
        flush=True,
    )
    if not args.dry_run:
        # Never import torch in this launcher before supplying the child env.
        subprocess.run(command, env=environment, check=True)


def probe_config():
    """Keep the packaged ZeRO policy, using small buckets and a supplied AdamW."""
    from omegaconf import OmegaConf

    path = Path(__file__).parent / "config/deepseek_v4_dspark_backend.yaml"
    config = OmegaConf.to_container(OmegaConf.load(path).deepspeed, resolve=True)
    config.pop("optimizer", None)
    config.pop("scheduler", None)
    config.update(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1)
    config["zero_optimization"].update(
        allgather_bucket_size=262144, reduce_bucket_size=262144
    )
    return config


def worker(args):
    import torch
    import torch.distributed as dist

    from verl_speco.backends.vllm_ascend_parallel import DeepSpeedParallelTopology
    from verl_speco.backends.vllm_ascend_runtime import initialize_distributed
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )
    from verl_speco.trainer.ascend_state_parking import state_probe
    from verl_speco.trainer.deepseek_v4_resident_probe import atomic_json
    from verl_speco.trainer.vllm_ascend_lifecycle import AscendTrainingLifecycle

    install_verl_npu_vllm_import_compat()
    import deepspeed

    device, owned = initialize_distributed()
    control = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    topology = DeepSpeedParallelTopology(args.tensor_parallel_size)
    root = Path(args.output)
    failure = [None]
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=False)
        except Exception as error:  # noqa: BLE001 -- broadcast directory failures
            failure[0] = repr(error)
    dist.broadcast_object_list(failure, src=0, group=control)
    if failure[0]:
        raise RuntimeError(f"Probe output directory failed: {failure[0]}")
    factory = None
    if args.state_storage == "file":
        from verl_speco.trainer.ascend_file_state import FileBackedAscendStateParking

        factory = lambda: FileBackedAscendStateParking(
            root / "state_parking", chunk_bytes=args.chunk_mib * 1024**2
        )
    parking = AscendTrainingLifecycle(control_group=control, parking_factory=factory)
    records = []
    session = None

    def settle(phase, function):
        result, error = None, None
        try:
            result = function()
        except Exception as exception:  # noqa: BLE001 -- settle all ranks before advancing
            error = repr(exception)
            atomic_json(
                root / f"failure_rank_{rank}.json",
                {
                    "phase": phase,
                    "error": error,
                    "traceback": traceback.format_exc(),
                    "allocation_inventory": parking.coverage_inventory,
                },
            )
        errors = [None] * world
        dist.all_gather_object(errors, error, group=control)
        if any(errors):
            raise RuntimeError(f"Allocator probe {phase} failed: {errors}")
        return result

    try:

        def construct():
            with parking.allocation_scope():
                torch.manual_seed(42)
                model = torch.nn.Linear(32, 32, bias=False).to(device, torch.bfloat16)
                teacher = torch.nn.Linear(32, 32, bias=False).to(device, torch.bfloat16)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
                runtime, _, _, _ = deepspeed.initialize(
                    args=argparse.Namespace(local_rank=int(os.environ["LOCAL_RANK"])),
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    config=probe_config(),
                    mpu=topology,
                )
                return SimpleNamespace(model_engine=runtime, teacher=teacher)

        session = settle("construct", construct)
        identities = None
        for step in range(1, args.rounds + 1):

            def train():
                runtime = session.model_engine
                before = runtime.module.weight.detach().cpu().clone()
                with parking.allocation_scope():
                    inputs = torch.ones(1, 32, device=device, dtype=torch.bfloat16)
                    loss = runtime(inputs).float().square().mean()
                    runtime.backward(loss)
                    runtime.step()
                    loss_value = float(loss.detach().cpu())
                    del loss, inputs
                torch.npu.synchronize()
                after = runtime.module.weight.detach().cpu()
                if torch.equal(before, after):
                    raise RuntimeError("Optimizer did not change model weights")
                # Deliberately exercise the communication-stream statistic case.
                norm = runtime.optimizer._global_grad_norm
                if torch.is_tensor(norm):
                    norm = float(norm.detach().cpu())
                with torch.npu.stream(torch.npu.Stream()):
                    statistic = torch.tensor(float(norm), device=device)
                torch.npu.synchronize()
                runtime.optimizer._global_grad_norm = statistic
                runtime._global_grad_norm = statistic
                return loss_value

            loss = settle(f"train_{step}", train)
            settle(f"coverage_{step}", lambda: parking.check_coverage(session))
            settle(f"park_{step}", lambda: parking.park(session))

            def pressure_and_wake(expected_step=step):
                noise = torch.ones(256, 256, device=device)
                if noise.sum().item() != 65536:
                    raise RuntimeError(
                        "Default allocator failed while training pool was parked"
                    )
                del noise
                torch.npu.synchronize()
                torch.npu.empty_cache()
                parking.wake(session)
                state = state_probe(session)
                if (
                    state["global_step"] != expected_step
                    or any(value != expected_step for value in state["optimizer_steps"])
                    or not state["optimizer_steps"]
                ):
                    raise RuntimeError("Optimizer progress differs across park/wake")
                return state

            state = settle(f"wake_{step}", pressure_and_wake)
            current = {
                key: state[key]
                for key in (
                    "engine_id",
                    "optimizer_id",
                    "inner_optimizer_id",
                    "scheduler_id",
                )
            }
            if identities is not None and identities != current:
                raise RuntimeError("Engine, optimizer or scheduler was reconstructed")
            identities = current
            records.append(
                {
                    "step": step,
                    "loss": loss,
                    "global_step": state["global_step"],
                    "optimizer_steps": state["optimizer_steps"],
                    "scheduler_state": state["scheduler_state"],
                    "allocation_inventory": parking.coverage_inventory,
                    "wake_verified": True,
                    "host_snapshot_bytes": parking.last_status["host_snapshot_bytes"],
                    "scratch_disk_bytes": parking.last_status["scratch_disk_bytes"],
                }
            )
        results = [None] * world
        dist.all_gather_object(
            results,
            {
                "rank": rank,
                "pid": os.getpid(),
                "allocator_environment": allocator_environment(),
                "identities": identities,
                "rounds": records,
            },
            group=control,
        )
        if rank == 0:
            report = {
                "status": "passed",
                "backend": "vllm_ascend",
                "tensor_parallel_size": topology.tensor_parallel_size,
                "data_parallel_size": topology.data_parallel_size,
                "rounds": args.rounds,
                "state_storage": args.state_storage,
                "allocator_environment": allocator_environment(),
                "workers": results,
                "scope": "small-model ZeRO state ownership and allocator lifecycle",
                "deepseek_numerics_validated": False,
                "rollout_integration_validated": False,
            }
            atomic_json(root / "report.json", report)
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "workers"}
                ),
                flush=True,
            )
    finally:
        # Restore mappings before tensor destruction, including failed phases.
        try:
            if session is not None and parking.state == "parked":
                parking.wake(session)
        finally:
            if owned:
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--state-storage", choices=("memory", "file"), default="memory")
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-environment-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.tensor_parallel_size, args.data_parallel_size, args.chunk_mib) < 1
        or args.rounds < 2
    ):
        parser.error("Positive TP/DP and at least two rounds are required")
    # Keep direct torchrun invocations compatible, while the normal entrypoint
    # owns the allocator environment instead of relying on shell line continuations.
    worker_process = args.worker or all(
        name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if not worker_process:
        launch(args)
        return
    if args.dry_run:
        parser.error("--dry-run belongs to the launcher, not a torchrun worker")
    try:
        source = prepare_worker_environment(args)
    except RuntimeError as error:
        parser.error(str(error))
    expected = args.tensor_parallel_size * args.data_parallel_size
    if os.environ.get("WORLD_SIZE") != str(expected):
        parser.error(f"Expected WORLD_SIZE={expected} for the requested TP/DP")
    if os.environ.get("RANK") == "0" or args.check_environment_only:
        print(
            json.dumps(
                {
                    "phase": "worker_environment",
                    "rank": os.environ.get("RANK"),
                    "world_size": os.environ.get("WORLD_SIZE"),
                    "configuration_source": source,
                    **allocator_environment(),
                }
            ),
            flush=True,
        )
    if not args.check_environment_only:
        worker(args)


if __name__ == "__main__":
    main()
