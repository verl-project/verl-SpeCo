# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone IPC probe reusing SpeCo's production draft receiver and verl SHM."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from verl_speco.integration.deepseek_v4_online import runtime_weight_name
from verl_speco.integration.deepseek_v4_update_probe import _canonical, tensor_digest
from verl_speco.integration.standalone_rollout_capture import (
    StandaloneCaptureWorkerExtension,
)
from verl_speco.integration.vllm_runtime import SpecoVLLMColocateWorkerExtension
from verl_speco.transport.bucketed_weight_transfer import configure_socket_timeout


def checkpoint_runtime_weights(checkpoint, aliases, required, tp_size):
    from verl_speco.models.deepseek_v4.export import (
        _rank_state_files,
        _topology_rank_state_files,
    )

    directory = Path(checkpoint)
    files = _topology_rank_state_files(directory, _rank_state_files(directory), 0)
    if len(files) != tp_size:
        raise ValueError("IPC publish training/serving TP differs")
    for rank, path in enumerate(files):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        seen = {}
        for source, value in state.items():
            name = aliases.get(_canonical(source))
            if name not in required:
                continue
            if name in seen:
                if not torch.equal(value, seen[name]):
                    raise ValueError(f"Conflicting runtime aliases: {name}")
                continue
            seen[name] = value
            yield runtime_weight_name(rank, tp_size, name), value
        if set(seen) != set(required):
            raise ValueError("IPC source omitted draft parameters")


def send_checkpoint_channels(checkpoint, channels, tp_size, timeout=600):
    """Each channel uses the installed verl sender; tensors stay mmap-backed."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        BucketedWeightSender,
    )

    from verl_speco.integration.vllm_runtime import (
        patch_verl_bucketed_weight_transfer_shm_reuse,
    )

    patch_verl_bucketed_weight_transfer_shm_reuse()

    def send(channel):
        sender = BucketedWeightSender(
            zmq_handle=channel["channel"],
            bucket_size_mb=channel["bucket_mb"],
            use_shm=True,
        )
        configure_socket_timeout(sender, timeout)
        weights = checkpoint_runtime_weights(
            checkpoint, channel["aliases"], channel["required"], tp_size
        )
        asyncio.run(sender.async_send_weights(weights))
        return {"channel": channel["channel"], "status": "sent"}

    with ThreadPoolExecutor(max_workers=len(channels)) as pool:
        return list(pool.map(send, channels))


class StandaloneIPCDraftWorkerExtension(
    StandaloneCaptureWorkerExtension, SpecoVLLMColocateWorkerExtension
):
    """Opt-in diagnostic extension; default capture worker remains unchanged."""

    def speco_standalone_hccl_environment(self):
        from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank

        return {
            "status": "configured",
            "tp_rank": get_tensor_model_parallel_rank(),
            "dp_rank": get_dp_group().rank_in_group,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "HCCL_NPU_SOCKET_PORT_RANGE",
                    "HCCL_HOST_SOCKET_PORT_RANGE",
                )
            },
        }

    def _get_speco_draft_zmq_handle(self):
        channel = getattr(self, "_speco_probe_ipc_channel", None)
        if channel:
            return channel
        return SpecoVLLMColocateWorkerExtension._get_speco_draft_zmq_handle(self)

    def speco_standalone_ipc_update(self, operation, prefix=None, timeout=600):
        from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = get_dp_group().rank_in_group
        session = getattr(self, "_speco_draft_update_probe", None)
        try:
            if session is None and operation == "rollback":
                return {"status": "rolled_back", "tp_rank": tp_rank, "dp_rank": dp_rank}
            if session is None:
                raise RuntimeError("IPC probe requires prepared reference weights")
            if operation == "arm":
                if session.committed or hasattr(session, "previous_revision"):
                    raise RuntimeError("IPC session is already armed or committed")
                if not prefix or "/" in prefix or len(prefix) > 50:
                    raise ValueError("Invalid IPC probe channel prefix")
                self._speco_probe_ipc_channel = (
                    f"ipc:///tmp/{prefix}-dp{dp_rank}-tp{tp_rank}.sock"
                )
                session.backup = {
                    name: value.detach().cpu().clone()
                    for name, value in session.loader.params.items()
                    if name in session.loader.required
                }
                session.previous_revision = int(
                    getattr(self, "_speco_draft_runtime_revision", 0)
                )
                session.previous_source = getattr(
                    self, "_speco_draft_weight_source", None
                )
                aliases = {
                    _canonical(alias): name
                    for alias, name in session.aliases.items()
                    if name in session.loader.required
                }
                bucket_mb = max(
                    1, (session.max_tensor_bytes + (1 << 20) - 1) // (1 << 20)
                )
                session.loader.pending.clear()
                session.loader.experts.clear()
                return {
                    "status": "armed",
                    "tp_rank": tp_rank,
                    "dp_rank": dp_rank,
                    "channel": self._speco_probe_ipc_channel,
                    "bucket_mb": bucket_mb,
                    "aliases": aliases,
                    "required": sorted(session.loader.required),
                    "previous_revision": session.previous_revision,
                }
            if operation == "receive":
                result = self.update_draft_weights_from_ipc(
                    use_shm=True, timeout=timeout
                )
                if any(
                    tensor_digest(session.loader.params[name]) != digest
                    for name, digest in session.expected.items()
                ):
                    raise RuntimeError(
                        "IPC runtime parameters differ from checkpoint reference"
                    )
                revision = int(self._speco_draft_runtime_revision)
                if revision != session.previous_revision + 1:
                    raise RuntimeError(
                        "IPC runtime revision did not advance exactly once"
                    )
                session.committed = True
                return {
                    "status": "ipc_committed",
                    "tp_rank": tp_rank,
                    "dp_rank": dp_rank,
                    "runtime_revision": revision,
                    "num_tensors": result["loaded_params"],
                }
            if operation == "rollback":
                session.rollback()
                if hasattr(session, "previous_revision"):
                    self._speco_draft_runtime_revision = session.previous_revision
                    self._speco_draft_weight_source = session.previous_source
                self._speco_draft_update_probe = None
                return {"status": "rolled_back", "tp_rank": tp_rank, "dp_rank": dp_rank}
            raise ValueError("Unknown IPC probe operation")
        except Exception as error:  # noqa: BLE001 -- coordinate errors across workers
            return {
                "status": "failed",
                "tp_rank": tp_rank,
                "dp_rank": dp_rank,
                "error": repr(error),
            }


def transfer_ipc_probe(engine, checkpoint, tp_size, dp_size, timeout=600):
    import os
    import uuid

    from verl_speco.probe_deepseek_v4_draft_update import require_phase

    rpc = getattr(engine, "draft_update_rpc", engine.collective_rpc)
    channels = rpc(
        "speco_standalone_ipc_update",
        kwargs={"operation": "arm", "prefix": "speco-ipc-" + uuid.uuid4().hex},
    )
    require_phase(channels, "armed", tp_size, dp_size)
    if len({row["channel"] for row in channels}) != len(channels):
        raise RuntimeError("IPC worker channels are not unique")
    stats = os.statvfs("/dev/shm")
    needed = sum(row["bucket_mb"] << 20 for row in channels)
    if needed > stats.f_bavail * stats.f_frsize:
        raise RuntimeError(
            f"Insufficient /dev/shm for IPC buckets: need {needed} bytes"
        )
    # Only one engine RPC runs at a time. Always drain it before rollback or
    # generation; the DP frontend request queues are intentionally serial.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            rpc,
            "speco_standalone_ipc_update",
            kwargs={"operation": "receive", "timeout": timeout},
        )
        try:
            sent = send_checkpoint_channels(checkpoint, channels, tp_size, timeout)
        finally:
            received = future.result()
    require_phase(received, "ipc_committed", tp_size, dp_size)
    return received, [
        {"status": row["status"], "channel": row["channel"]} for row in sent
    ]
