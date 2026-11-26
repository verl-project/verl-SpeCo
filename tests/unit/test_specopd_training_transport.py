# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import asyncio
from dataclasses import replace
import sys
from types import SimpleNamespace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verl_speco.integration import specopd_collectives as module
from verl_speco.integration import specopd_training_transport as transport
from verl_speco.specopd_config import SpecOPDConfig
from verl_speco.backends.specopd_forward import build_specopd_training_batch
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
    return build_specopd_training_batch(request, score, "cpu", SpecOPDConfig(), (1,))


@pytest.mark.parametrize("lengths,dp,expected", [
    ([4096], 1, [1]), ([2048, 2048], 1, [2]), ([2049, 2048], 1, [1, 1]),
    ([4097], 1, None), ([10, 4097], 1, None), ([10, 4097], 2, None),
])
def test_training_plan_enforces_single_trajectory_token_budget(lengths, dp, expected):
    from verl_speco.backends.specopd_forward import plan_specopd_training_groups

    samples = [{"input_ids": torch.empty(1, length)} for length in lengths]
    if expected is None:
        with pytest.raises(ValueError, match="context length 4097.*training_microbatch_token_budget=4096"):
            plan_specopd_training_groups(samples, dp, 10, 4096)
    else:
        assert plan_specopd_training_groups(samples, dp, 10, 4096) == expected


def test_numpy_wire_crops_storage_and_preserves_bf16_exactly():
    expected = sample()
    payload, size = transport.encode_training_payload([expected], SpecOPDConfig())
    decoded = transport.decode_training_payload(payload)
    batch = decoded[0]
    assert size == sum(t.numel() * t.element_size() for t in expected.values() if torch.is_tensor(t))
    assert batch["_specopd_request_id"] == "r"
    assert batch.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
        else:
            assert batch[key] == value

    from verl_speco.backends.specopd_forward import prepare_specopd_training_batches

    [dummy], _ = prepare_specopd_training_batches(
        decoded, "cpu", real_slots=[False],
    )
    assert not dummy["proposal_mask"].any()
    assert batch["proposal_mask"].all()  # A dummy rank cannot mutate shared input.

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
        next(iter(payload["tensors"].values()))["0/hidden_states"]["data"], np.ndarray
    )


def test_payload_identity_rejects_reordered_or_corrupt_request_metadata():
    payload, _ = transport.encode_training_payload([sample(32)], SpecOPDConfig(enable=True))
    payload["samples"][0]["_specopd_request_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        transport.decode_training_payload(payload)


def worker(rank=0):
    return NS(
        rank=rank,
        trainer=NS(_get_sp_world_size=lambda: 2, _get_dp_world_size=lambda: 8),
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_cpu_stage_is_cancellable_and_one_use(cancel):
    async def run():
        payload, _ = transport.encode_training_payload([sample(32)], SpecOPDConfig(enable=True))
        w = worker()
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(transport.stage_training_group(w, (future,), 5, 0, (1,)))
        await asyncio.sleep(0)
        assert not task.done() and not hasattr(w, "_specopd_prepared_group")
        if cancel:
            transport.discard_training_group(w, 5, 0, (1,))
        future.set_result(payload)
        if cancel:
            with pytest.raises(RuntimeError, match="cancelled"):
                await task
            assert getattr(w, "_specopd_prepared_group", None) is None
        else:
            ready = await task
            assert ready["rank"] == 0 and ready["sp"] * ready["dp"] == 16
            assert transport.take_training_group(w, 5, 0, (1,)).samples[0]["_specopd_request_id"] == "r"
            with pytest.raises(RuntimeError, match="no matching"):
                transport.take_training_group(w, 5, 0, (1,))

    asyncio.run(run())


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "digest", "groups"])
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
                    groups=(1,),
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
            if fault == "groups":
                status[-1]["groups"] = (2,)
            return status

        def train_specopd_group(self, step, revision, features):
            assert events == ["put-once", "all-data-loaded"]
            events.append("device-collectives")
            return [dict(trained=True)] * 16

        def discard_specopd_group(self, *args):
            events.append("release")
            return []

    g = Group()
    t = NS(_require_speco_worker_group=lambda: g, _ray_get_if_needed=lambda x: x)
    if fault:
        with pytest.raises(RuntimeError):
            transport.dispatch_prepared_group(t, [sample(32)], 5, 0, (1,), {}, SpecOPDConfig())
        assert events == ["put-once", "all-data-loaded", "release"]
    else:
        assert (
            len(transport.dispatch_prepared_group(t, [sample(32)], 5, 0, (1,), {}, SpecOPDConfig()))
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
    options = dict(
        step=5,
        feature_indices=(1,),
        group_sizes=[1],
        spec=SpecOPDConfig(),
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
    samples = [sample(32)]
    spec = SpecOPDConfig(enable=True)
    before, _ = transport.encode_training_payload(samples, spec)
    after, _ = transport.encode_training_payload(samples, replace(spec, optimizer_updates_per_collection=2))
    assert before["identity"] != after["identity"]
    before["spec"] = after["spec"]
    with pytest.raises(ValueError, match="identity"):
        transport.decode_training_payload(before)
