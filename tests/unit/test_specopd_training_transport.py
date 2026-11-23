# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import asyncio
import sys
from types import SimpleNamespace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verl_speco.integration import specopd_collectives as module
from verl_speco.integration import specopd_training_transport as transport
from verl_speco.joint_scoring import (
    BranchScores,
    DraftBranch,
    JointScores,
    JointScoringRequest,
)


def sample(length=8192):
    request = JointScoringRequest(
        "r", tuple(range(length)), 2, (DraftBranch("b", 9, (1, 2, 3)),)
    )
    score = JointScores(
        "r",
        torch.zeros(length - 2),
        (BranchScores("b", torch.ones(3, 8), torch.zeros(3), torch.ones(3, 4)),),
        {1: torch.arange(length * 4, dtype=torch.bfloat16).reshape(length, 4)},
    )
    return request, score


def test_numpy_wire_crops_storage_and_preserves_bf16_exactly():
    request, score = sample()
    payload, size = transport.encode_training_payload([(request, score)])
    decoded = transport.decode_training_payload(payload)
    assert size == 10 * 4 * 2 + 3 * 4 * 4
    [decoded_branch] = decoded[0][1].branches
    assert decoded_branch.target_logits is decoded_branch.target_logz is None
    torch.testing.assert_close(decoded_branch.target_hidden, score.branches[0].target_hidden)
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


def _collective_inputs():
    trainer = SimpleNamespace(
        runtime_device="cpu",
        _get_sp_group=lambda: None,
        _get_dp_group=lambda: "dp",
        _get_sp_world_size=lambda: 1,
        _get_dp_world_size=lambda: 8,
    )
    request = SimpleNamespace(
        sample_id="r",
        trunk_ids=[1, 2],
        response_start=1,
        branches=[SimpleNamespace(branch_id="b", anchor=0, token_ids=[3, 4])],
    )
    options = dict(
        step=5,
        selected=[(request, None)],
        feature_indices=(1,),
        group_sizes=[1],
        spec=SimpleNamespace(
            optimizer_updates_per_collection=1, min_scored_positions_per_update=1
        ),
    )
    return trainer, options


def test_preflight_uses_same_dp_group_as_loss_reduction(monkeypatch):
    trainer, options = _collective_inputs()
    calls = []
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        module.dist, "all_reduce", lambda value, **kw: calls.append((value.clone(), kw))
    )
    module.warm_specopd_training_collectives(trainer, step=options["step"])
    assert len(calls) == 1
    assert calls[0][1] == {"group": "dp", "op": module.dist.ReduceOp.MIN}
    assert calls[0][0].shape == (1,)


def test_cpu_schedule_signature_covers_optimizer_policy():
    trainer, options = _collective_inputs()
    options.pop("selected")
    options["payload_identity"] = "shared-data"
    before = module.training_schedule_digest(trainer, **options)
    options["spec"].optimizer_updates_per_collection += 1
    assert module.training_schedule_digest(trainer, **options) != before


def test_single_rank_needs_no_distributed_runtime():
    trainer, options = _collective_inputs()
    trainer._get_dp_world_size = lambda: 1
    module.warm_specopd_training_collectives(trainer, step=options["step"])


def test_multi_rank_requires_explicit_group():
    trainer, options = _collective_inputs()
    trainer._get_dp_group = lambda: None
    with pytest.raises(RuntimeError, match="missing its process group"):
        module.warm_specopd_training_collectives(trainer, step=options["step"])
