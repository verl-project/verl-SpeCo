# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Pre-forward admission checks for SpecOPD device collectives."""

from types import SimpleNamespace

import pytest

from verl_speco.integration import specopd_collectives as module


def inputs():
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
    trainer, options = inputs()
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
    trainer, options = inputs()
    options.pop("selected")
    options["payload_identity"] = "shared-data"
    before = module.training_schedule_digest(trainer, **options)
    options["spec"].optimizer_updates_per_collection += 1
    assert module.training_schedule_digest(trainer, **options) != before


def test_single_rank_needs_no_distributed_runtime():
    trainer, options = inputs()
    trainer._get_dp_world_size = lambda: 1
    module.warm_specopd_training_collectives(trainer, step=options["step"])


def test_multi_rank_requires_explicit_group():
    trainer, options = inputs()
    trainer._get_dp_group = lambda: None
    with pytest.raises(RuntimeError, match="missing its process group"):
        module.warm_specopd_training_collectives(trainer, step=options["step"])
