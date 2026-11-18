# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise live router reads and failure atomicity, without an NPU model."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl_speco.integration.deepseek_v4_online import (
    EmbeddedDraftLoader,
    runtime_weight_name,
)
from verl_speco.integration.deepseek_v4_update_probe import DraftUpdateSession
from verl_speco.ops.ascend_weight_cache import (
    FP32WeightCaches,
    refresh_fp32_weight_caches,
)


def draft(dtype=torch.bfloat16, *, alias=False):
    model = nn.Module()
    model.gate = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.gate.weight.copy_(torch.eye(2))
    model.gate.precast_fp32_weight = True
    cache = model.gate.weight.detach().float()
    model.gate.weight_fp32 = cache if alias else cache.clone()
    return model


def staged(model, fill=2):
    loader = EmbeddedDraftLoader(model)
    loader.receive(
        [
            (
                runtime_weight_name(0, 1, "gate.weight"),
                torch.full_like(model.gate.weight, fill),
            )
        ]
    )
    return loader


@pytest.mark.parametrize(
    "dtype,alias",
    [(torch.bfloat16, False), (torch.float32, False), (torch.float32, True)],
)
def test_commit_refreshes_native_router_read_without_changing_storage(dtype, alias):
    model = draft(dtype, alias=alias)
    cache = model.gate.weight_fp32
    pointers = model.gate.weight.data_ptr(), cache.data_ptr()
    for fill in (2, 3):
        loader = staged(model, fill)
        assert loader.finish() == 1 and loader.committed
        assert model.gate.weight_fp32 is cache
        assert (model.gate.weight.data_ptr(), cache.data_ptr()) == pointers
        torch.testing.assert_close(
            nn.functional.linear(torch.tensor([[1.0, 3.0]]), cache),
            torch.full((1, 2), 4.0 * fill),
        )


def test_cache_refresh_failure_rolls_back_parameters_and_all_caches(monkeypatch):
    model = draft()
    model.other = draft().gate
    loader = EmbeddedDraftLoader(model)
    before_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    before_caches = loader.caches.snapshot()
    loader.receive(
        [
            (runtime_weight_name(0, 1, n), torch.full_like(p, 2))
            for n, p in model.named_parameters()
        ]
    )
    original = FP32WeightCaches.refresh

    def fail_after_refresh(caches):
        original(caches)
        raise RuntimeError("injected cache failure")

    monkeypatch.setattr(FP32WeightCaches, "refresh", fail_after_refresh)
    with pytest.raises(RuntimeError, match="injected cache failure"):
        loader.finish()
    assert loader.failed and not loader.committed and not loader.pending
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before_params[n], rtol=0, atol=0)
    for value, before in zip(loader.caches.snapshot(), before_caches, strict=True):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


def test_incomplete_update_never_mutates_parameter_or_cache():
    model = draft()
    model.other = draft().gate
    before = model.gate.weight.detach().clone()
    loader = staged(model)
    with pytest.raises(RuntimeError, match="Incomplete"):
        loader.finish()
    assert torch.equal(model.gate.weight, before)
    assert torch.equal(model.gate.weight_fp32, before.float())


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype", "replacement"])
def test_reject_invalid_cache_before_parameter_mutation(invalid):
    model = draft()
    loader = staged(model)
    before = model.gate.weight.detach().clone()
    if invalid == "missing":
        del model.gate.weight_fp32
    elif invalid == "shape":
        model.gate.weight_fp32 = torch.zeros(1)
    elif invalid == "dtype":
        model.gate.weight_fp32 = model.gate.weight.detach().clone()
    else:
        model.gate.weight_fp32 = model.gate.weight_fp32.clone()
    with pytest.raises(RuntimeError, match="cache storage changed"):
        loader.finish()
    assert torch.equal(model.gate.weight, before)


def test_refresh_after_restore_and_probe_rollback():
    model = draft()
    session = DraftUpdateSession.__new__(DraftUpdateSession)
    session.model = model
    session.loader = staged(model)
    session.backup = {"gate.weight": model.gate.weight.detach().clone()}
    session.loader.finish()
    session.rollback()
    assert torch.equal(model.gate.weight_fp32, torch.eye(2))
    with torch.no_grad():
        model.gate.weight.fill_(3)
    refresh_fp32_weight_caches(model)
    assert torch.equal(model.gate.weight_fp32, torch.full((2, 2), 3.0))


def test_missing_required_cache_is_rejected_but_plain_linear_is_supported():
    model = draft()
    del model.gate.weight_fp32
    with pytest.raises(ValueError, match="Invalid Ascend FP32"):
        EmbeddedDraftLoader(model)
    del model.gate.precast_fp32_weight
    assert staged(model).finish() == 1


def test_parking_inventory_and_value_probe_include_unregistered_cache():
    from verl_speco.trainer.ascend_state_parking import state_probe, trainer_tensors

    model = draft()
    engine = SimpleNamespace(
        model_engine=SimpleNamespace(
            module=model, optimizer=None, lr_scheduler=None, global_steps=0
        ),
        teacher=nn.Module(),
    )
    assert "gate.weight_fp32" not in dict(model.named_buffers())
    inventory = dict(trainer_tensors(engine))
    assert inventory["model/cache/gate.weight_fp32"] is model.gate.weight_fp32
    before = state_probe(engine)
    model.gate.weight_fp32.add_(1)
    assert state_probe(engine) != before
