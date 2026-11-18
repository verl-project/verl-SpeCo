# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v4_update_probe import checkpoint

from verl_speco.integration import deepseek_v4_ipc_probe as probe
from verl_speco.transport.bucketed_weight_transfer import required_bucket_megabytes


def test_shared_memory_bucket_fits_largest_runtime_tensor():
    assert (
        required_bucket_megabytes({"large": SimpleNamespace(nbytes=1 << 30)}, 512)
        == 1024
    )
    assert required_bucket_megabytes({"small": torch.ones(3)}, 512) == 512
    assert required_bucket_megabytes(iter([]), 512) == 512
    with pytest.raises(ValueError):
        required_bucket_megabytes({}, 0)


def test_checkpoint_sender_preserves_tp_names_and_bf16(tmp_path):
    checkpoint(tmp_path)
    aliases = {
        name: name for name in ["experts.w13_weight", "experts.w2_weight", "norm"]
    }
    actual = dict(probe.checkpoint_runtime_weights(tmp_path, aliases, set(aliases), 2))
    assert len(actual) == 6
    assert actual["__speco_tp_0_of_2__/experts.w13_weight"].dtype == torch.bfloat16
    assert actual["__speco_tp_1_of_2__/norm"].dtype == torch.float32
    with pytest.raises(ValueError, match="omitted"):
        list(probe.checkpoint_runtime_weights(tmp_path, aliases, {"missing"}, 2))


@pytest.mark.parametrize("fail", [False, True])
def test_coordinator_drains_receiver_rpc_before_returning(monkeypatch, fail):
    started, done, drained = threading.Event(), threading.Event(), threading.Event()
    channels = [
        {
            "status": "armed",
            "tp_rank": t,
            "dp_rank": 0,
            "channel": f"ipc:///tmp/test-{t}",
            "bucket_mb": 1,
            "aliases": {},
            "required": [],
        }
        for t in range(2)
    ]

    class Engine:
        def collective_rpc(self, *args, **kwargs):
            raise AssertionError("Use all-DP RPC")

        def draft_update_rpc(self, method, kwargs):
            if kwargs["operation"] == "arm":
                return channels
            started.set()
            assert done.wait(5)
            drained.set()
            return [
                {"status": "ipc_committed", "tp_rank": t, "dp_rank": 0}
                for t in range(2)
            ]

    def send(*args):
        assert started.wait(5)
        done.set()
        if fail:
            raise RuntimeError("sender failure")
        return [{"status": "sent", "channel": row["channel"]} for row in channels]

    monkeypatch.setattr(probe, "send_checkpoint_channels", send)
    monkeypatch.setattr(
        "os.statvfs", lambda _: SimpleNamespace(f_bavail=100000000, f_frsize=4096)
    )
    if fail:
        with pytest.raises(RuntimeError, match="sender failure"):
            probe.transfer_ipc_probe(Engine(), "unused", 2, 1)
    else:
        result, sent = probe.transfer_ipc_probe(Engine(), "unused", 2, 1)
        assert len(result) == len(sent) == 2
    assert drained.is_set()


@pytest.mark.parametrize("corrupt", [False, True])
def test_ipc_revision_and_explicit_rollback(monkeypatch, tmp_path, corrupt):
    import sys

    from test_deepseek_v4_expert_weights import Draft

    from verl_speco.integration.deepseek_v4_update_probe import worker_update_probe

    checkpoint(tmp_path)
    model = Draft([0, -1, -1, 1])
    worker = object.__new__(probe.StandaloneIPCDraftWorkerExtension)
    worker.model_runner = SimpleNamespace(drafter=SimpleNamespace(model=model))
    worker._speco_draft_runtime_revision = 4
    worker._speco_draft_weight_source = "checkpoint"
    result = worker_update_probe(worker, "prepare", str(tmp_path), tp_rank=0, tp_size=2)
    assert result["status"] == "prepared"
    session = worker._speco_draft_update_probe
    expected = {name: value.clone() for name, value in session.loader.pending.items()}
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            get_dp_group=lambda: SimpleNamespace(rank_in_group=0),
            get_tensor_model_parallel_rank=lambda: 0,
        ),
    )

    def apply(**kwargs):
        assert kwargs == {"use_shm": True, "timeout": 600}
        with torch.inference_mode():
            for name, value in expected.items():
                session.loader.params[name].copy_(value)
            if corrupt:
                model.norm.add_(1)
        worker._speco_draft_runtime_revision += 1
        return {"loaded_params": len(expected)}

    monkeypatch.setattr(worker, "update_draft_weights_from_ipc", apply)
    armed = worker.speco_standalone_ipc_update("arm", prefix="unit")
    assert armed["status"] == "armed"
    assert not session.loader.pending
    committed = worker.speco_standalone_ipc_update("receive")
    assert committed["status"] == ("failed" if corrupt else "ipc_committed")
    assert worker._speco_draft_runtime_revision == 5
    assert worker.speco_standalone_ipc_update("rollback")["status"] == "rolled_back"
    assert worker._speco_draft_runtime_revision == 4
    assert worker._speco_draft_weight_source == "checkpoint"
    assert all(torch.count_nonzero(value) == 0 for value in model.parameters())
