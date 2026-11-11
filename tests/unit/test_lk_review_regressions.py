# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise runtime helpers without requiring the distributed model stack."""
from __future__ import annotations

import ast
import asyncio
import logging
import math
import time
from types import MethodType, SimpleNamespace
from pathlib import Path
from typing import Any, Optional, cast

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, **bindings):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    namespace = dict(torch=torch, F=torch.nn.functional, logging=logging, Any=Any, Optional=Optional, cast=cast,
                     logger=logging.getLogger(__name__), time=time,
                     _is_ray_object_ref=lambda value: False, **bindings)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


async def _identity_async(value):
    return value


def test_feature_chunks_reject_partial_logz_but_allow_legacy():
    resolve = load_function("verl_speco/workers/speco_worker.py", "_resolve_feature_chunks",
                            asyncio=asyncio, _resolve_ray_object_ref=_identity_async)
    chunks = [dict(ref=dict(hidden_states=torch.ones(2, 3), target_logz=torch.ones(2)),
                   chunk_length=2, chunk_row_indices=[0, 1]),
              dict(ref=torch.ones(2, 3), chunk_length=2, chunk_row_indices=[2, 3])]
    with pytest.raises(ValueError, match="target_logz.*chunk"):
        asyncio.run(resolve(chunks))
    chunks[0]["ref"] = torch.ones(2, 3)
    hidden, logz = asyncio.run(resolve(chunks))
    assert hidden.shape == (1, 4, 3)
    assert logz is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_target_logz_bounds_fp32_workspace_and_preserves_logits(monkeypatch, dtype):
    from verl_speco.integration import oldlogprob_runtime as runtime

    logits = torch.randn(2, 35, 17000, dtype=dtype)
    original = logits.clone()
    expected = torch.logsumexp(logits.float() / 0.7, dim=-1, keepdim=True)
    real = torch.logsumexp
    sizes = []
    def tracked(value, *args, **kwargs):
        sizes.append(value.numel())
        return real(value, *args, **kwargs)
    monkeypatch.setattr(torch, "logsumexp", tracked)
    actual = runtime._compute_oldlogprob_target_logz(
        {"logits": logits}, {runtime.OLD_LOGPROB_TARGET_LOGZ_TEMPERATURE_KEY: 0.7})
    torch.testing.assert_close(actual, expected)
    assert torch.equal(logits, original)
    assert max(sizes) <= 32 * 16384


def test_dflash_empty_target_and_flat_gather(monkeypatch):
    # Load the production helper if it exists, then exercise the backend entry point.
    shared = {}
    path = ROOT / "verl_speco/backends/target_hidden.py"
    if path.exists():
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), shared)
    gather = load_function("verl_speco/backends/dflash_trainer_backend.py",
                           "_gather_dflash_aligned_target_hidden",
                           **{k: v for k, v in shared.items() if k == "gather_aligned_target_hidden"})
    labels = torch.tensor([[[1, 3], [2, 5]], [[2, 4], [1, 3]]])
    keep = torch.tensor([[True, False], [True, True]])
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    indices = torch.where(keep.unsqueeze(-1), labels - 1, 0)
    expected = torch.stack([hidden[b][indices[b]] for b in range(2)])
    def forbid(*args, **kwargs):
        raise AssertionError("expanded gather must not be used")
    monkeypatch.setattr(torch, "gather", forbid)
    torch.testing.assert_close(gather(hidden, labels, keep), expected)
    assert gather(hidden[:, :0], labels, keep) is None
    assert gather(None, labels, keep) is None


@pytest.mark.parametrize("standalone_metrics", [None, False, True])
@pytest.mark.parametrize("lk_loss_type", ["alpha", "adaptive_hybrid"])
def test_eagle3_lk_metrics_ignore_log_cadence(standalone_metrics, lk_loss_type):
    from verl_speco.backends.lk_loss import adaptive_hybrid_acceptance_loss, negative_log_acceptance_loss

    path = "verl_speco/backends/eagle3_trainer_backend.py"
    compute = load_function(path, "compute_loss", math=math,
                            adaptive_hybrid_acceptance_loss=adaptive_hybrid_acceptance_loss,
                            negative_log_acceptance_loss=negative_log_acceptance_loss,
                            _masked_soft_cross_entropy=load_function(path, "_masked_soft_cross_entropy"))
    class Config(dict):
        __getattr__ = dict.__getitem__
    backend = SimpleNamespace(
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training=Config(
            use_logits=False, eagle3_ce_loss_alpha=0.0,
            eagle3_lk_loss_alpha=1.0, eagle3_lk_loss_type=lk_loss_type,
            eagle3_lk_telemetry_interval_steps=0)))),
        target_model=lambda x: torch.zeros(1, 2, 3),
        _compute_target_p_padded=lambda **kw: (torch.full((1, 3, 3), 1/3), torch.ones(1, 3)),
        _compute_target_acceptance_p_padded=lambda **kw: torch.full((1, 3, 3), 1/3),
    )
    if standalone_metrics is not None:
        backend.enable_standalone_training_metrics = standalone_metrics
    logits = torch.nn.Parameter(torch.zeros(1, 2, 3))
    class Model:
        t2d = torch.ones(3, dtype=torch.bool)
        def __call__(self, **kw):
            return dict(logits=[logits], position_masks=[torch.ones(1, 2)])
    batch = dict(input_ids=torch.zeros(1, 2, dtype=torch.long), hidden_states=torch.zeros(1, 2, 3),
                 last_hidden_states=torch.zeros(1, 2, 3), target_logz=torch.zeros(1, 2),
                 attention_mask=torch.ones(1, 2), loss_mask=torch.ones(1, 2),
                 position_ids=torch.arange(2).unsqueeze(0))
    result = compute(backend, Model(), batch, 0)
    result["total_local_ploss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    diagnostics = result["diagnostics"]
    if lk_loss_type == "adaptive_hybrid":
        assert diagnostics["lk_diagnostic_token_count"].item() == 2
        assert diagnostics["lk_acceptance_sum"].item() == pytest.approx(2)
    assert diagnostics["lk_weighted_token_count"].item() == 2
    trainer_path = "verl_speco/trainer/base_trainer.py"
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="eagle3"),
                              optimizer=None, optimizer_steps_total=0,
                              _reduce_training_metric=lambda value: value,
                              _block_drafter_config_value=lambda key, default: default)
    for name in ("reset_training_metrics", "_block_drafter_metric_prefix",
                 "_pack_training_diagnostics", "_accumulate_training_diagnostics",
                 "_summarize_step_metrics", "_record_dflash_training_metrics",
                 "get_training_metrics"):
        setattr(trainer, name, MethodType(load_function(trainer_path, name), trainer))
    trainer.reset_training_metrics()
    trainer._record_dflash_training_metrics({"diagnostics": diagnostics})
    metrics = trainer.get_training_metrics()
    if lk_loss_type == "adaptive_hybrid":
        assert metrics["eagle3/lk/acceptance"] == pytest.approx(1)
    assert metrics["eagle3/lk_loss"] == pytest.approx(0, abs=1e-6)


def test_training_sample_uid_survives_copies_and_object_replacement():
    from uuid import uuid4

    uid = load_function("verl_speco/trainer/base_trainer.py", "_training_sample_uid", uuid4=uuid4)
    item = {}
    first = uid(item)
    assert uid(item) == first
    copied_item = dict(item)
    del item
    assert uid(copied_item) == first
    # Different samples with identical contents must still count separately.
    assert uid({}) != first


def test_shared_target_alignment_handles_noncontiguous_features_and_gradients():
    from verl_speco.backends.target_hidden import gather_aligned_target_hidden

    base = torch.randn(2, 3, 5, requires_grad=True)
    hidden = base.transpose(1, 2)
    labels = torch.tensor([[[0, 7]], [[3, 4]]])
    result = gather_aligned_target_hidden(hidden, labels, torch.ones(2, 1, dtype=torch.bool))
    expected = torch.stack([hidden[0, [0, 4]], hidden[1, [2, 3]]]).unsqueeze(1)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert base.grad.sum().item() == 12
    with pytest.raises(ValueError, match="batch size"):
        gather_aligned_target_hidden(hidden, labels[:1], torch.ones(1, 1, dtype=torch.bool))


def test_feature_chunks_preserve_target_logz_row_order():
    resolve = load_function("verl_speco/workers/speco_worker.py", "_resolve_feature_chunks",
                            asyncio=asyncio, _resolve_ray_object_ref=_identity_async)
    hidden, logz = asyncio.run(resolve([
        dict(ref=dict(hidden_states=torch.tensor([[2., 3.], [4., 5.]]),
                      target_logz=torch.tensor([6., 7.])),
             chunk_length=2, chunk_row_indices=[1, 0])]))
    torch.testing.assert_close(hidden, torch.tensor([[[4., 5.], [2., 3.]]]))
    torch.testing.assert_close(logz, torch.tensor([[7., 6.]]))
