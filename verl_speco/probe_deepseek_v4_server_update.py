# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Verify the real standalone verl server's draft commit boundary without RL."""

import argparse
import asyncio
import json
import os
import urllib.request
import uuid
from pathlib import Path

from verl_speco.integration.serving_drafter_commit import publish_serving_revision
from verl_speco.probe_deepseek_v4_draft_update import require_phase


async def run_probe(server, args, ids):
    from verl_speco.integration.deepseek_v4_ipc_probe import send_checkpoint_channels

    await server.set_global_steps.remote(0)
    revision = int(getattr(args, "revision", 1))
    params = {"temperature": 0, "max_tokens": 8, "_verl_global_steps": -999}
    if getattr(args, "skip_probe_collection", False):
        params["_verl_skip_drafter_collection"] = True
    before = await server.generate.remote(ids, dict(params), "server-before")
    prepared = await server.probe_rpc.remote(
        "speco_standalone_draft_update",
        {"operation": "prepare", "checkpoint": args.checkpoint},
    )
    require_phase(
        prepared, "prepared", args.tensor_parallel_size, args.data_parallel_size
    )
    queued = []

    async def while_paused():
        state = await server.speco_get_serving_versions.remote()
        if state["admission_open"] or state["active_requests"]:
            raise RuntimeError("Serving gate is not closed and idle")
        task = asyncio.ensure_future(
            server.generate.remote(ids, dict(params), "server-queued")
        )
        queued.append(task)
        for _ in range(100):
            state = await server.speco_get_serving_versions.remote()
            if state["waiting_requests"] > 0:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Queued request did not reach server admission")
        if task.done():
            raise RuntimeError("Ordinary serving escaped the closed admission boundary")

    async def publish():
        channels = await server.probe_rpc.remote(
            "speco_standalone_ipc_update",
            {"operation": "arm", "prefix": "speco-srv-" + uuid.uuid4().hex},
        )
        require_phase(
            channels, "armed", args.tensor_parallel_size, args.data_parallel_size
        )
        if len({row["channel"] for row in channels}) != len(channels):
            raise RuntimeError("Duplicate IPC channels")
        stats = os.statvfs("/dev/shm")
        if (
            sum(row["bucket_mb"] << 20 for row in channels)
            > stats.f_bavail * stats.f_frsize
        ):
            raise RuntimeError("Insufficient shared memory for serving IPC")
        receive = asyncio.ensure_future(
            server.probe_rpc.remote(
                "speco_standalone_ipc_update",
                {"operation": "receive", "timeout": args.ipc_timeout},
            )
        )
        try:
            sent = await asyncio.to_thread(
                send_checkpoint_channels,
                args.checkpoint,
                channels,
                args.tensor_parallel_size,
                args.ipc_timeout,
            )
        finally:
            committed = await receive
        require_phase(
            committed,
            "ipc_committed",
            args.tensor_parallel_size,
            args.data_parallel_size,
        )
        return {"committed": committed, "sent": sent}

    try:
        boundary = await publish_serving_revision(
            [server],
            target_version=0,
            revision=revision,
            publish=publish,
            while_paused=while_paused,
        )
        after = await queued[0]
        if before.token_ids != after.token_ids:
            raise RuntimeError("Target greedy output changed after draft publication")
        for output, expected_revision in [(before, revision - 1), (after, revision)]:
            fields = output.extra_fields
            if fields.get("global_steps") != 0 or any(
                fields.get(key) != expected_revision
                for key in [
                    "_speco_drafter_revision_min",
                    "_speco_drafter_revision_max",
                ]
            ):
                raise RuntimeError(f"Serving output version labels differ: {fields}")
        finalized = await server.probe_rpc.remote(
            "speco_standalone_draft_update", {"operation": "finalize"}
        )
        require_phase(
            finalized, "finalized", args.tensor_parallel_size, args.data_parallel_size
        )
        return {
            "status": "passed",
            "ipc_validated": True,
            "server_version_commit_validated": True,
            "target_version": 0,
            "drafter_revision": revision,
            "admission_blocked_during_update": True,
            "target_tokens_unchanged": True,
            "tokens": after.token_ids,
            "boundary": boundary,
            "rl_integration_validated": False,
        }
    finally:
        for task in queued:
            if not task.done():
                task.cancel()
        if queued:
            await asyncio.gather(*queued, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "checkpoint", "trajectories", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--ipc-timeout", type=int, default=600)
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.data_parallel_size, args.ipc_timeout) <= 0:
        parser.error("Positive TP/DP/timeout required")
    import ray

    from verl_speco.integration.deepseek_v4_server_probe import (
        create_server_actor_class,
    )
    from verl_speco.integration.vllm_ascend_implementation import (
        compatible_implementation_fingerprint,
    )
    from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

    contract = DeepSeekV4Contract.from_model(args.model)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    metadata = json.loads((checkpoint / "speco_ds4.json").read_text())
    if metadata["identity"][
        "contract"
    ] != contract.identity() or not compatible_implementation_fingerprint(
        metadata["identity"]["source_fingerprint"]
    ):
        raise ValueError("Checkpoint contract/implementation differs")
    if metadata["topology"]["tensor_parallel_size"] != args.tensor_parallel_size:
        raise ValueError("Checkpoint/serving TP differs")
    args.checkpoint = str(checkpoint)
    with Path(args.trajectories).open() as source:
        row = next(json.loads(line) for line in source if line.strip())
    ids = row["input_ids"]
    if (
        not ids
        or len(ids) + 8 > args.max_model_len
        or any(type(i) is not int or not 0 <= i < contract.vocab_size for i in ids)
    ):
        raise ValueError("Invalid prepared probe trajectory")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    count = args.tensor_parallel_size * args.data_parallel_size
    if len([value for value in devices.split(",") if value.strip()]) != count:
        raise ValueError("Set exactly TP*DP visible NPU devices")
    if ray.is_initialized():
        raise RuntimeError("Run this probe in a fresh Python process")
    actor = None
    try:
        ray.init(
            address="local",
            num_cpus=4,
            include_dashboard=False,
            object_store_memory=512 * 1024**2,
        )
        if ray.cluster_resources().get("NPU", 0) < count:
            raise RuntimeError("Ray did not detect the required NPU resources")
        options = {
            "model": args.model,
            "tp_size": args.tensor_parallel_size,
            "dp_size": args.data_parallel_size,
            "max_model_len": args.max_model_len,
            "devices": devices,
        }
        actor = (
            create_server_actor_class()
            .options(num_cpus=2, resources={"NPU": count})
            .remote(options)
        )
        ray.get(actor.launch_server.remote(), timeout=1800)
        host, port = ray.get(actor.get_server_address.remote())
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=10
        ) as health:
            if health.status != 200:
                raise RuntimeError("HTTP server health check failed")
        report = asyncio.run(run_probe(actor, args, ids))
        report["http_health_validated"] = True
        (output / "server_update_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except Exception as error:
        rollback = None
        if actor is not None:
            try:
                rollback = ray.get(
                    actor.probe_rpc.remote(
                        "speco_standalone_ipc_update", {"operation": "rollback"}
                    ),
                    timeout=120,
                )
            except Exception as recovery:  # noqa: BLE001 -- preserve original failure
                rollback = {"error": repr(recovery)}
        (output / "failure.json").write_text(
            json.dumps({"error": repr(error), "rollback": rollback}, indent=2)
        )
        raise
    finally:
        if actor is not None:
            try:
                ray.get(actor.probe_shutdown.remote(), timeout=30)
            except Exception as cleanup:  # noqa: BLE001 -- terminate only the owned actor
                print(f"Probe shutdown failed: {cleanup!r}", flush=True)
            ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
