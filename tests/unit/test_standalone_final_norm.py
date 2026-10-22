# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Model-owned norm and origin gates; load production AST without optional backends."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
import torch
import verl_speco.trainer.target_feature_replay as replay
from verl_speco.backends.target_hidden import gather_aligned_target_hidden


def _load_class(path, name, methods=None, extra=None):
    source = Path(__file__).parents[2] / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    ns = {"torch": torch, "Any": Any, "Optional": Optional, "cast": cast,
          "logging": logging, "logger": logging.getLogger(__name__)}
    ns.update(extra or {})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), ns)
    return ns[name]


def _config(mode="offline", payload="vllm_reference"):
    return SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={
            "mode": mode, "feature_store": {"type": "tq"}, "use_logits": False,
            "transfer_queue": {"expected_feature": {"payload_kind": payload}},
        })), model={"trust_remote_code": False},
    )


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_standalone_training_error_propagates_when_requested(raise_on_error):
    import asyncio

    cls = _load_class("verl_speco/trainer/base_trainer.py", "DrafterBaseTrainer",
                      {"training_step_from_batch"})
    trainer = cls()
    error = ValueError("injected optimizer error")

    async def fail(batch, step):
        raise error

    trainer._training_step_on_batch = fail
    if raise_on_error:
        with pytest.raises(ValueError) as caught:
            asyncio.run(trainer.training_step_from_batch({}, 1, raise_on_error=True))
        assert caught.value is error
    else:
        assert not asyncio.run(trainer.training_step_from_batch({}, 1))
    assert trainer.last_standalone_training_error is error


@pytest.mark.parametrize("batch_pad,legacy_pad,expected", [
    (0, None, 0), (3, None, 3), (0, 7, 0), (None, 7, 7), (None, None, 0),
])
def test_prepared_batch_runs_actual_optimizer_step_without_old_prepare_state(batch_pad, legacy_pad, expected):
    import asyncio
    import time
    from contextlib import nullcontext

    cls = _load_class("verl_speco/trainer/base_trainer.py", "DrafterBaseTrainer",
                      {"_training_step_on_batch", "_clip_training_grad_norm"},
                      {"time": time, "device_name": "cpu"})
    trainer = cls()
    trainer.model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = None
    trainer.optimizer_steps_total = trainer.training_steps = 0
    trainer._ulysses_group_context = nullcontext
    trainer._record_dflash_training_metrics = lambda _: None
    trainer.record_training_timing = lambda *args: None
    trainer._reduce_loss_metrics = lambda v, p, n: (v.detach(), p.detach(), n, 1)
    trainer._is_checkpoint_leader = lambda: False
    if legacy_pad is not None:
        trainer._current_pad_size = legacy_pad
    received = []

    def compute_loss(model, batch, pad_size):
        received.append(pad_size)
        assert "_speco_pad_size" not in batch
        loss = model(batch["hidden_states"]).float().square().sum()
        return {"total_local_vloss": loss, "total_local_ploss": loss * 0,
                "local_num_tokens": torch.tensor(2.0), "v_weight": 1.0, "p_weight": 0.0}

    trainer.backend = SimpleNamespace(model_type="dspark", compute_loss=compute_loss)
    batch = {"hidden_states": torch.ones(2, 2)}
    if batch_pad is not None:
        batch["_speco_pad_size"] = batch_pad
    before = trainer.model.weight.detach().clone()
    assert asyncio.run(trainer._training_step_on_batch(batch, 0))
    assert received == [expected]
    assert trainer.optimizer_steps_total == trainer.training_steps == 1
    assert not torch.equal(before, trainer.model.weight)


def _eagle_head(device="cpu"):
    cls = _load_class("verl_speco/backends/eagle3_trainer_backend.py", "_Eagle3VllmTargetHead")
    head = SimpleNamespace(fc=torch.nn.Linear(2, 3, bias=False))
    norm = torch.nn.LayerNorm(2)
    return cls(head, norm).to(device)


def test_eagle_model_norm_is_frozen_and_only_runs_for_raw_source():
    model = _eagle_head()
    seen = []
    model.verifier_norm.register_forward_hook(lambda *args: seen.append(torch.is_grad_enabled()))
    raw = torch.tensor([[[1., 3.], [4., 8.]]], requires_grad=True)
    original = raw.detach().clone()
    scores, bad = model(raw, hidden_is_raw=True, hidden_source="standalone_vllm_reference")
    expected = model.fc(torch.nn.functional.layer_norm(original, (2,)))
    torch.testing.assert_close(scores, expected)
    assert seen == [False] and not bad.any()
    assert all(not p.requires_grad for p in model.verifier_norm.parameters())
    assert not scores.requires_grad
    torch.testing.assert_close(raw, original)
    # Already-normalized/co-train input uses the existing plain head behavior.
    normalized = torch.nn.functional.layer_norm(original, (2,))
    torch.testing.assert_close(model(normalized), model.fc(normalized))
    assert len(seen) == 1
    with pytest.raises(ValueError, match="standalone vLLM"):
        model(raw, hidden_is_raw=True, hidden_source="cotrain")
    assert len(seen) == 1


@pytest.mark.parametrize("mode,payload,expected_loads", [
    ("online", "vllm_reference", 0), ("offline", None, 0),
    ("offline", "encoded_feature", 0), ("offline", "vllm_reference", 1),
])
@pytest.mark.parametrize("store_type", ["tq", "ray_reference"])
def test_eagle_model_setup_does_not_load_norm_on_other_paths(monkeypatch, mode, payload, expected_loads, store_type):
    wrapper = type(_eagle_head())
    cls = _load_class("verl_speco/backends/eagle3_trainer_backend.py", "Eagle3TrainerBackend",
                      {"_maybe_build_vllm_target_head"}, {"_Eagle3VllmTargetHead": wrapper})
    backend = cls()
    backend.config = _config(mode, payload)
    backend.config.rollout.drafter.training["feature_store"]["type"] = store_type
    backend.model_type = "eagle3"
    calls = []
    def load(*args, **kwargs):
        calls.append(args)
        return torch.nn.LayerNorm(2)
    monkeypatch.setattr(replay, "load_vllm_final_norm", load)
    head = SimpleNamespace(fc=torch.nn.Linear(2, 3))
    result = backend._maybe_build_vllm_target_head(head, "target")
    assert len(calls) == expected_loads
    assert (result is head) == (expected_loads == 0)


@pytest.mark.parametrize("mode,source,payload,raw,allowed", [
    ("offline", "standalone_vllm_reference", "vllm_reference", True, True),
    ("online", "standalone_vllm_reference", "vllm_reference", True, False),
    ("offline", "cotrain", "vllm_reference", True, False),
    ("offline", "standalone_vllm_reference", None, True, False),
    ("online", None, None, False, True),
])
@pytest.mark.parametrize("store_type", ["tq", "ray_reference"])
def test_dspark_backend_restricts_raw_norm_to_file_path(mode, source, payload, raw, allowed, store_type):
    cls = _load_class("verl_speco/backends/dspark_trainer_backend.py", "DSparkTrainerBackend", {"compute_loss"})
    backend = cls()
    backend.config = _config(mode, payload)
    backend.config.rollout.drafter.training["feature_store"]["type"] = store_type
    backend.target_lm_head = SimpleNamespace(fc=torch.nn.Linear(2, 3))
    class ReachedModel(Exception):
        pass
    class Model:
        draft_model = SimpleNamespace(num_context_layers=2)
        def __call__(self, **kwargs):
            assert kwargs.get("target_hidden_is_raw", False) is raw
            assert "target_final_norm" not in kwargs
            raise ReachedModel
    batch = {"input_ids": torch.ones(1, 3, dtype=torch.long), "hidden_states": torch.zeros(1, 3, 4),
             "loss_mask": torch.ones(1, 3)}
    if source is not None:
        batch["hidden_source"] = source
    if raw:
        batch["last_hidden_state_norm"] = "raw_target_residual"
    with pytest.raises(ReachedModel if allowed else ValueError):
        backend.compute_loss(Model(), batch, 0)


@pytest.mark.parametrize("store_type", ["tq", "ray_reference"])
@pytest.mark.parametrize("invalid_aux", [False, True])
def test_cpu_stage_only_moves_data_and_preserves_raw_target(store_type, invalid_aux):
    cls = _load_class("verl_speco/trainer/base_trainer.py", "DrafterBaseTrainer", {
        "_stage_standalone_cpu_batch", "_sanitize_training_batch", "_sanitize_sequence_tensor",
        "_get_hidden_state_clip_value", "_mask_loss_for_bad_inputs",
    })
    trainer = cls()
    trainer.config = _config()
    trainer.config.rollout.drafter.training["feature_store"]["type"] = store_type
    trainer.runtime_device = torch.device("cpu")
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.rank = 0
    raw = torch.full((1, 3, 2), 1.0e5)
    batch = {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3),
             "loss_mask": torch.ones(1, 3), "hidden_states": torch.zeros(1, 3, 4),
             "target_last_hidden_states": raw, "last_hidden_state_norm": "raw_target_residual"}
    if invalid_aux:
        batch["hidden_states"][0, 1, 0] = float("nan")
    result = trainer._stage_standalone_cpu_batch(batch)
    assert torch.equal(result["target_last_hidden_states"], raw)
    assert result["last_hidden_state_norm"] == "raw_target_residual"
    assert not hasattr(trainer, "_standalone_final_norm")
    if invalid_aux:
        assert torch.isfinite(result["hidden_states"]).all()
        assert result["loss_mask"][0, 1] == 0
        assert result["attention_mask"][0, 1] == 0


def test_eagle_target_norm_masks_nonfinite_raw_rows_after_cpu_scan_is_skipped():
    model = _eagle_head()
    raw = torch.tensor([[[float("nan"), 1.0], [2.0, 3.0]]])
    scores, bad = model(raw, hidden_is_raw=True, hidden_source="standalone_vllm_reference")
    assert bad.tolist() == [[True, False]]
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("mode,source,payload,raw,allowed", [
    ("offline", "standalone_vllm_reference", "vllm_reference", True, True),
    ("online", "standalone_vllm_reference", "vllm_reference", True, False),
    ("offline", "cotrain", "vllm_reference", True, False),
    ("offline", "standalone_vllm_reference", None, True, False),
    ("online", None, None, False, True),
])
@pytest.mark.parametrize("store_type", ["tq", "ray_reference"])
def test_eagle_backend_restricts_raw_input(mode, source, payload, raw, allowed, store_type):
    source_file = Path(__file__).parents[2] / "verl_speco/backends/eagle3_trainer_backend.py"
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Eagle3TrainerBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "attention_mask" for t in n.targets))
    method.body = method.body[:end] + [ast.parse("return raw_target_hidden").body[0]]
    ast.fix_missing_locations(method)
    model = _eagle_head()
    ns = {"_Eagle3VllmTargetHead": type(model)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_file), "exec"), ns)
    backend = SimpleNamespace(config=_config(mode, payload), target_model=model)
    backend.config.rollout.drafter.training["feature_store"]["type"] = store_type
    batch = {"input_ids": torch.ones(1, 3), "hidden_states": torch.zeros(1, 3, 4)}
    if raw:
        batch["last_hidden_state_norm"] = "raw_target_residual"
    if source is not None:
        batch["hidden_source"] = source
    if allowed:
        assert ns[method.name](backend, None, batch, 0) is raw
    else:
        with pytest.raises(ValueError):
            ns[method.name](backend, None, batch, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU norm parity requires CUDA")
def test_eagle_gpu_norm_matches_cpu():
    cpu = _eagle_head()
    gpu = _eagle_head("cuda")
    gpu.load_state_dict(cpu.state_dict())
    raw = torch.tensor([[[1., 3.], [4., 8.]]])
    expected, _ = cpu(raw, hidden_is_raw=True, hidden_source="standalone_vllm_reference")
    actual, _ = gpu(raw.cuda(), hidden_is_raw=True, hidden_source="standalone_vllm_reference")
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("apply_norm", [False, True])
def test_dspark_forward_selects_target_positions_before_norm(apply_norm):
    source = Path(__file__).parents[2] / "verl_speco/backends/dspark_trainer_backend.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkTrainingModel")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    gather = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_gather_aligned_target_hidden")
    # Run the production target preparation section, stopping before draft losses.
    end = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "weight_mask" for t in n.targets))
    forward.body = forward.body[:end] + [ast.parse("return aligned_target_hidden, eval_mask").body[0]]
    namespace = {
        "torch": torch, "Optional": Optional,
        "gather_aligned_target_hidden": gather_aligned_target_hidden, 
        "_resolve_sliding_windows": lambda config: None,
        "build_dflash_attention_masks": lambda **kwargs: (None, None),
    }
    ast.fix_missing_locations(forward)
    exec(compile(ast.Module(body=[gather, forward], type_ignores=[]), str(source), "exec"), namespace)

    class Draft:
        config = {}

        def extract_context_feature(self, hidden):
            return hidden[0]

        def __call__(self, **kwargs):
            return torch.zeros(1, 1, 2, 2)

    trainer = SimpleNamespace(
        _debug_forward_count=0, draft_model=Draft(), block_size=2,
        _sample_anchor_positions=lambda *a, **kw: (torch.tensor([[0]]), torch.tensor([[True]])),
        _create_noise_embed=lambda *a: torch.zeros(1, 1, 2, 2),
        _create_position_ids=lambda *a: (torch.arange(3), torch.arange(2)),
        _build_label_tensors=lambda **kw: (
            torch.zeros(1, 1, 2, dtype=torch.long), torch.zeros(1, 1, 2, dtype=torch.long),
            torch.ones(1, 1, 2, dtype=torch.bool), torch.tensor([[[1, 3]]]),
        ),
    )
    trainer._gather_aligned_target_hidden = lambda **kw: namespace[gather.name](trainer, **kw)
    norm = torch.nn.LayerNorm(2)
    seen = []
    norm.register_forward_hook(lambda module, args, output: seen.append(tuple(args[0].shape)))
    trainer.verifier_norm = norm.requires_grad_(False)
    trainer.target_hidden_clip_value = 1.0e4
    raw = torch.tensor([[[1., 3.], [100., 200.], [5., 9.]]])
    target, valid = namespace[forward.name](
        trainer, torch.tensor([[1, 2, 3]]), [raw], torch.ones(1, 3), torch.ones(4, 2),
        target_last_hidden_states=raw, target_hidden_is_raw=apply_norm,
        hidden_source="standalone_vllm_reference",
    )
    expected = raw[:, [0, 2]].unsqueeze(1)
    if apply_norm:
        expected = torch.nn.functional.layer_norm(expected, (2,))
        assert seen == [(1, 1, 2, 2)]
    else:
        assert seen == []
    torch.testing.assert_close(target, expected)
    assert valid.all()


