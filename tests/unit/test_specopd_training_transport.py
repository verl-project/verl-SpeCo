# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import asyncio
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest
import torch

from verl_speco.integration import specopd_training_transport as transport
from verl_speco.joint_scoring import (
    JointScoringRequest,
    DraftBranch,
    JointScores,
    BranchScores,
)


def sample(length=8192):
    request = JointScoringRequest(
        "r", tuple(range(length)), 2, (DraftBranch("b", 9, (1, 2, 3)),)
    )
    score = JointScores(
        "r",
        torch.zeros(length - 2),
        (BranchScores("b", None, torch.zeros(3), torch.ones(3, 4)),),
        {1: torch.arange(length * 4, dtype=torch.bfloat16).reshape(length, 4)},
    )
    return request, score


def test_numpy_wire_crops_storage_and_preserves_bf16_exactly():
    request, score = sample()
    payload, size = transport.encode_training_payload([(request, score)])
    decoded = transport.decode_training_payload(payload)
    assert size == 10 * 4 * 2 + 3 * 4 * 4 + 3 * 4
    assert decoded[0][0].trunk_ids == request.trunk_ids
    torch.testing.assert_close(
        decoded[0][1].trunk_features[1], score.trunk_features[1][:10], rtol=0, atol=0
    )
    assert decoded[0][1].response_log_probs.numel() == 0

    def check(value):
        assert not torch.is_tensor(value)
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                check(v)

    check(payload)
    assert isinstance(
        next(iter(payload["tensors"].values()))["0/f/1"]["data"], np.ndarray
    )


def test_payload_identity_rejects_reordered_or_corrupt_request_metadata():
    payload, _ = transport.encode_training_payload([sample(32)])
    request, features, branches = payload["samples"][0]
    payload["samples"][0] = (
        JointScoringRequest("wrong", request.trunk_ids, 2, request.branches),
        features,
        branches,
    )
    with pytest.raises(ValueError, match="identity"):
        transport.decode_training_payload(payload)


def test_hidden_only_wire_omits_logz_and_produces_identical_train_batch():
    from dataclasses import replace

    from verl_speco.backends.specopd_forward import build_specopd_training_batch
    from verl_speco.specopd_config import SpecOPDConfig

    request, score = sample(32)
    original, original_bytes = transport.encode_training_payload([(request, score)])
    score_without_logz = replace(score, branches=(replace(score.branches[0], target_logz=None),))
    lean, lean_bytes = transport.encode_training_payload([(request, score_without_logz)])
    assert original_bytes - lean_bytes == 3 * torch.float32.itemsize
    assert original["samples"][0][2][0][2] == "0/b/0/z"
    assert lean["samples"][0][2][0][2] is None
    [(request, decoded_full)] = transport.decode_training_payload(original)
    [(_, decoded_lean)] = transport.decode_training_payload(lean)
    assert decoded_full.branches[0].target_logz is not None
    assert decoded_lean.branches[0].target_logz is None
    assert original["identity"] != lean["identity"]

    full_batch = build_specopd_training_batch(request, decoded_full, "cpu", SpecOPDConfig(), (1,))
    lean_batch = build_specopd_training_batch(request, decoded_lean, "cpu", SpecOPDConfig(), (1,))
    assert full_batch.keys() == lean_batch.keys()
    for key, value in full_batch.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(lean_batch[key], value, atol=0, rtol=0)
        else:
            assert lean_batch[key] == value


def worker(rank=0):
    return NS(
        rank=rank,
        trainer=NS(_get_sp_world_size=lambda: 2, _get_dp_world_size=lambda: 8),
    )


def test_cpu_stage_waits_for_payload_and_is_one_use():
    async def run():
        payload, _ = transport.encode_training_payload([sample(32)])
        w = worker()
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            transport.stage_training_group(w, (future,), 5, 0, (1,))
        )
        await asyncio.sleep(0)
        assert not task.done() and not hasattr(w, "_specopd_prepared_group")
        future.set_result(payload)
        ready = await task
        assert ready["rank"] == 0 and ready["sp"] * ready["dp"] == 16
        data = transport.take_training_group(w, 5, 0, (1,))
        assert data[0][0].sample_id == "r"
        with pytest.raises(RuntimeError, match="no matching"):
            transport.take_training_group(w, 5, 0, (1,))

    asyncio.run(run())


def test_cancel_during_transfer_never_leaves_cached_data():
    async def run():
        w = worker()
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            transport.stage_training_group(w, (future,), 5, 0, (1,))
        )
        await asyncio.sleep(0)
        transport.discard_training_group(w, 5, 0, (1,))
        future.set_result(transport.encode_training_payload([sample(32)])[0])
        with pytest.raises(RuntimeError, match="cancelled"):
            await task
        assert getattr(w, "_specopd_prepared_group", None) is None

    asyncio.run(run())


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "digest"])
def test_driver_never_dispatches_training_before_all_16_readiness_acks(
    monkeypatch, fault
):
    events = []
    monkeypatch.setitem(
        sys.modules, "ray", NS(put=lambda p: events.append("put-once") or p)
    )

    class Group:
        def prepare_specopd_group(self, refs, step, revision, features):
            assert len(refs) == 1
            self.samples = transport.decode_training_payload(refs[0])
            events.append("all-data-loaded")
            status = [
                dict(
                    rank=i,
                    step=step,
                    revision=revision,
                    features=features,
                    identity=refs[0]["identity"],
                    schedule="same-plan",
                    samples=1,
                    sp=2,
                    dp=8,
                )
                for i in range(16)
            ]
            if fault == "missing":
                status.pop()
            if fault == "duplicate":
                status[-1]["rank"] = 0
            if fault == "digest":
                status[-1]["identity"] = "wrong"
            return status

        def train_specopd_group(self, data, step, revision, features):
            assert data is None and events == ["put-once", "all-data-loaded"]
            events.append("device-collectives")
            return [dict(trained=True)] * 16

        def discard_specopd_group(self, *args):
            events.append("release")
            return []

    g = Group()
    t = NS(_require_speco_worker_group=lambda: g, _ray_get_if_needed=lambda x: x)
    if fault:
        with pytest.raises(RuntimeError):
            transport.dispatch_prepared_group(t, [sample(32)], 5, 0, (1,), {})
        assert events == ["put-once", "all-data-loaded", "release"]
    else:
        normal = JointScoringRequest("normal", (1, 2, 3), 1)
        assert (
            len(transport.dispatch_prepared_group(t, [(normal, None), sample(32)], 5, 0, (1,), {}))
            == 16
        )
        assert events == [
            "put-once",
            "all-data-loaded",
            "device-collectives",
            "release",
        ]
