# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Actual installed verl SHM/ZMQ transfer in a fresh CPU receiver process."""

import asyncio
import importlib.metadata
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

try:
    importlib.metadata.version("verl")
    HAS_VERL = True
except importlib.metadata.PackageNotFoundError:
    HAS_VERL = False
pytestmark = pytest.mark.skipif(
    not HAS_VERL, reason="Requires installed verl transport"
)


class CPUDevice:
    def synchronize(self):
        pass

    def empty_cache(self):
        pass


def _cpu_transport():
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transfer

    from verl_speco.integration.vllm_runtime import (
        patch_verl_bucketed_weight_transfer_shm_reuse,
    )

    patch_verl_bucketed_weight_transfer_shm_reuse(transfer)
    transfer.get_torch_device = lambda: CPUDevice()
    transfer.is_support_ipc = lambda: False
    return transfer


def _receive_cpu(channel, report, fail, ready):
    import json

    from vllm import distributed, platforms

    from verl_speco.integration.deepseek_v4_online import EmbeddedDraftLoader
    from verl_speco.integration.drafter_config_env import set_drafter_config_env
    from verl_speco.integration.vllm_runtime import SpecoVLLMColocateWorkerExtension

    _cpu_transport()
    # CPU-only hardware stand-ins; the transport and production receiver are real.
    platforms.current_platform = SimpleNamespace(device_type="cpu")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 2
    set_drafter_config_env(
        json.dumps(
            {
                "training": {
                    "backend": "vllm_ascend",
                    "collect_hidden_states_from_rollout": True,
                    "rollout_capture": {"model_family": "deepseek_v4"},
                }
            }
        )
    )
    model = torch.nn.Module()
    model.norm = torch.nn.Parameter(torch.zeros(256))
    model.experts = torch.nn.Module()
    model.experts.register_buffer("expert_map", torch.tensor([1, -1, -1, 0]))
    model.experts.w13_weight = torch.nn.Parameter(
        torch.zeros(2, 256, 1024, dtype=torch.bfloat16)
    )
    model.experts.w2_weight = torch.nn.Parameter(
        torch.zeros(2, 512, 256, dtype=torch.bfloat16)
    )
    model.model = SimpleNamespace(named_parameters=list, named_buffers=list)
    worker = object.__new__(SpecoVLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.model_runner = SimpleNamespace(
        drafter=SimpleNamespace(
            model=model, speculative_config=SimpleNamespace(method="dspark")
        )
    )
    worker._get_speco_draft_zmq_handle = lambda: channel
    original = EmbeddedDraftLoader.receive
    buckets = []

    def receive(loader, weights):
        buckets.append(len(weights))
        if fail:
            raise ValueError("injected callback failure")
        return original(loader, weights)

    EmbeddedDraftLoader.receive = receive
    ready.set()
    try:
        result = worker.update_draft_weights_from_ipc(use_shm=True, timeout=5)
        # Gate/up halves from both TP lanes must survive SHM bucket reuse.
        assert torch.all(model.experts.w13_weight[:, :, :256] == 1)
        assert torch.all(model.experts.w13_weight[:, :, 256:512] == 2)
        assert torch.all(model.experts.w13_weight[:, :, 512:768] == 11)
        assert torch.all(model.experts.w13_weight[:, :, 768:] == 12)
        assert torch.all(model.experts.w2_weight[:, :256, :] == 21)
        assert torch.all(model.experts.w2_weight[:, 256:, :] == 22)
        assert worker._speco_draft_runtime_revision == 1
        result = {
            "status": "passed",
            "count": result["loaded_params"],
            "buckets": len(buckets),
            "runtime_revision": worker._speco_draft_runtime_revision,
        }
    except Exception as error:  # noqa: BLE001 -- propagate CPU subprocess result
        assert torch.count_nonzero(model.norm) == 0
        assert torch.count_nonzero(model.experts.w13_weight) == 0
        assert worker._speco_draft_runtime_revision == 0
        result = {"status": "rejected", "error": repr(error)}
    Path(report).write_text(json.dumps(result))


@pytest.mark.parametrize("fail", [False, True])
def test_actual_shm_bucket_reuse_and_callback_failure(tmp_path, fail):
    import json

    import zmq

    from verl_speco.integration.deepseek_v4_online import runtime_weight_name
    from verl_speco.transport.bucketed_weight_transfer import configure_socket_timeout

    transfer = _cpu_transport()
    channel = "ipc://" + str(tmp_path / "weights.sock")
    report = tmp_path / "receiver.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_receive_cpu, args=(channel, str(report), fail, ready)
    )
    process.start()

    def weights():
        for rank in range(2):
            yield (
                runtime_weight_name(rank, 2, "experts.w13_weight"),
                torch.cat(
                    [
                        torch.full((4, 256, 256), rank + 1, dtype=torch.bfloat16),
                        torch.full((4, 256, 256), rank + 11, dtype=torch.bfloat16),
                    ],
                    dim=2,
                ),
            )
            yield (
                runtime_weight_name(rank, 2, "experts.w2_weight"),
                torch.full((4, 256, 256), rank + 21, dtype=torch.bfloat16),
            )
            yield runtime_weight_name(rank, 2, "norm"), torch.ones(256)

    sender = transfer.BucketedWeightSender(
        zmq_handle=channel, bucket_size_mb=1, use_shm=True
    )
    configure_socket_timeout(sender, 10)
    try:
        assert ready.wait(60), "Receiver did not initialize"
        if fail:
            with pytest.raises(zmq.Again):
                asyncio.run(sender.async_send_weights(weights()))
        else:
            asyncio.run(sender.async_send_weights(weights()))
        process.join(30)
        assert process.exitcode == 0
        result = json.loads(report.read_text())
        assert result["status"] == ("rejected" if fail else "passed")
        if not fail:
            assert result["buckets"] > 1
        else:
            assert "injected callback failure" in result["error"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        cleanup = getattr(transfer, "_speco_cleanup_persistent_weight_shm", None)
        if cleanup:
            cleanup()
