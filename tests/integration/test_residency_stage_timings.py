# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure residency boundaries without changing optimizer state or ordering."""

import ast
import asyncio
import copy
import json
import logging
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _trainer(events, *, fail_offload=False, skip=False, device="cuda"):
    path = Path(__file__).parents[2] / "verl_speco" / "trainer" / "base_trainer.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DrafterBaseTrainer"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_residency_stage",
            "activate_training_model",
            "cleanup_training",
            "_record_gpu_transfer",
        }
    ]
    cls.bases, cls.decorator_list = [], []
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)

    offload_cache_flags = []

    def offload(model, empty_cache=True):
        offload_cache_flags.append(empty_cache)
        events.append("offload_model")
        if fail_offload:
            raise RuntimeError("offload failure")

    namespace = {
        "logging": logging,
        "json": json,
        "time": time,
        "asyncio": asyncio,
        "torch": torch,
        "contextmanager": contextmanager,
        "logger": logging.getLogger(__name__),
        "device_name": device,
        "get_device_id": lambda: 0,
        "offload_fsdp_model_to_cpu": offload,
        "load_fsdp_model_to_gpu": lambda model: events.append("load_model"),
        "load_fsdp_optimizer": lambda **kwargs: events.append("load_optimizer"),
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - actual repository methods
    trainer = namespace["DrafterBaseTrainer"]()
    trainer._test_offload_cache_flags = offload_cache_flags
    trainer.rank = 0
    trainer._log_training_memory = lambda stage: None
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        trainer.model.weight.copy_(torch.arange(4).reshape(2, 2) / 7)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer._pending_checkpoint_future = None
    trainer._pending_full_checkpoint_future = None
    trainer._full_checkpoint_executor = SimpleNamespace(
        shutdown=lambda **kwargs: events.append("shutdown")
    )
    trainer.collected_data = SimpleNamespace(
        clear=lambda: events.append("clear_samples")
    )
    trainer.data_buffer = SimpleNamespace(clear=lambda: events.append("clear_buffer"))
    trainer._mark_buffer_changed = lambda: events.append("buffer_changed")
    trainer._offload_optimizer_state_to_cpu = lambda: events.append("offload_optimizer")
    trainer._use_blocking_npu_optimizer_offload = lambda: False
    trainer._move_target_lm_head = lambda device: events.append("head_" + str(device))
    trainer._park_idle_drafter_hccl = lambda: events.append("park_hccl")
    trainer.device_module = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
    )
    trainer.runtime_device = torch.device("cpu")
    trainer._apply_pending_target_lm_head_weight = lambda: events.append("apply_head")
    trainer._optimizer_state_on_runtime_device = lambda: False
    trainer.is_offload_param = False
    trainer.is_offload_optimizer = False
    trainer.skip_heavy_cleanup_after_drafter_training = skip
    trainer._training_active = True
    trainer.optimizer_steps_total = 40
    return trainer


@pytest.mark.parametrize("clear_data", [False, True])
def test_cleanup_activation_preserve_order_and_next_adam_update(clear_data):
    events = []
    trainer = _trainer(events)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 9
    trainer.model(inputs).square().mean().backward()
    trainer.optimizer.step()
    reference = copy.deepcopy(trainer.model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    reference_optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    before = copy.deepcopy(trainer.optimizer.state_dict())
    asyncio.run(trainer.cleanup_training(clear_data=clear_data))
    expected = ["offload_model", "offload_optimizer", "head_cpu"]
    if clear_data:
        expected += ["clear_samples", "clear_buffer", "buffer_changed"]
    expected += ["shutdown", "synchronize", "empty_cache", "park_hccl"]
    assert events == expected
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    torch.testing.assert_close(trainer.optimizer.state_dict(), before, rtol=0, atol=0)
    for name in reference.state_dict():
        torch.testing.assert_close(
            trainer.model.state_dict()[name],
            reference.state_dict()[name],
            rtol=0,
            atol=0,
        )
    events.clear()
    assert asyncio.run(trainer.activate_training_model())
    assert events == [
        "apply_head",
        "load_model",
        "load_optimizer",
        "head_cpu",
        "apply_head",
    ]
    assert trainer.optimizer_steps_total == 40
    assert all(value >= 0 for value in trainer._last_cleanup_stage_timings.values())
    assert all(value >= 0 for value in trainer._last_activation_stage_timings.values())
    for model, optimizer in (
        (trainer.model, trainer.optimizer),
        (reference, reference_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    torch.testing.assert_close(
        trainer.model.state_dict(), reference.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        trainer.optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0
    )


def test_checkpoint_waits_are_retained_and_timing_resets():
    trainer = _trainer([])
    for name in ("_pending_checkpoint_future", "_pending_full_checkpoint_future"):
        future = Future()
        future.set_result(None)
        setattr(trainer, name, future)
    trainer._last_cleanup_stage_timings = {"stale": 100}
    asyncio.run(trainer.cleanup_training(clear_data=False))
    assert trainer._pending_checkpoint_future is None
    assert trainer._pending_full_checkpoint_future is None
    assert "checkpoint_wait" in trainer._last_cleanup_stage_timings
    assert "full_checkpoint_wait" in trainer._last_cleanup_stage_timings
    assert "stale" not in trainer._last_cleanup_stage_timings


def test_existing_best_effort_offload_failure_semantics_are_retained():
    events = []
    trainer = _trainer(events, fail_offload=True)
    asyncio.run(trainer.cleanup_training(clear_data=False))
    assert "offload_optimizer" in events and "synchronize" in events
    assert "offload_model" in trainer._last_cleanup_stage_timings


def test_skip_heavy_cleanup_keeps_existing_branch():
    events = []
    trainer = _trainer(events, skip=True)
    asyncio.run(trainer.cleanup_training(clear_data=False))
    assert events == []
    assert set(trainer._last_cleanup_stage_timings) == {"zero_grad"}


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])
def test_only_cuda_defers_model_helper_cache_reclaim(device):
    events = []
    trainer = _trainer(events, device=device)
    asyncio.run(trainer.cleanup_training(clear_data=False))
    assert trainer._test_offload_cache_flags == [device != "cuda"]
    if device != "cpu":
        assert events.count("synchronize") == events.count("empty_cache") == 1
        assert events.index("head_cpu") < events.index("synchronize")
        assert events.index("synchronize") < events.index("empty_cache")
    else:
        assert "synchronize" not in events and "empty_cache" not in events


@pytest.mark.parametrize(
    "device,enabled,strategy,used",
    [
        ("cuda", True, "fsdp2", True),
        ("cuda", True, "fsdp2", False),
        ("cuda", False, "fsdp2", False),
        ("cuda", True, "fsdp", False),
        ("npu", True, "fsdp2", False),
    ],
)
def test_activation_routes_batched_load_and_legacy_fallback(
    monkeypatch, device, enabled, strategy, used
):
    from verl_speco.trainer import gpu_transfer

    events = []
    trainer = _trainer(events, device=device)
    trainer._training_active = False
    trainer._training_initialized = True
    trainer._gpu_model_offload_reuse = enabled
    trainer._drafter_strategy = lambda: strategy
    trainer.training_device_mesh = object()
    metrics = []
    trainer._record_gpu_transfer = lambda scope, stats: metrics.append((scope, stats))

    def load(model, target, **kwargs):
        assert model is trainer.model
        events.append("batched_load")
        return used, {
            "copied_bytes": 16 if used else 0,
            "fallback_reason": "none" if used else "test_decline",
        }

    monkeypatch.setattr(gpu_transfer, "load_model_from_host", load)
    assert asyncio.run(trainer.activate_training_model())
    eligible = device == "cuda" and enabled and strategy == "fsdp2"
    assert ("batched_load" in events) == eligible
    assert ("load_model" in events) == (not eligible or not used)
    if eligible:
        assert metrics[0][0] == "model_load"
        assert metrics[0][1]["used_batched_load"] == int(used)


def test_activation_stops_after_batched_copy_failure(monkeypatch):
    from verl_speco.trainer import gpu_transfer

    events = []
    trainer = _trainer(events)
    trainer._training_active = False
    trainer._training_initialized = True
    trainer._gpu_model_offload_reuse = True
    trainer._drafter_strategy = lambda: "fsdp2"
    trainer.training_device_mesh = object()

    def fail(*args, **kwargs):
        raise RuntimeError("injected batched copy failure")

    monkeypatch.setattr(gpu_transfer, "load_model_from_host", fail)
    assert not asyncio.run(trainer.activate_training_model())
    assert not trainer._training_active
    assert "load_model" not in events and "load_optimizer" not in events
    assert "injected batched copy failure" in str(trainer.last_activation_error)


def test_bubble_activation_keeps_legacy_restore_and_stage_names(monkeypatch):
    from verl_speco.trainer import gpu_transfer

    events = []
    trainer = _trainer(events)
    trainer._bubble_time_enabled = True
    trainer._training_active = False
    trainer._gpu_model_offload_reuse = True
    trainer._drafter_strategy = lambda: "fsdp2"
    trainer.training_device_mesh = object()
    monkeypatch.setattr(
        gpu_transfer,
        "load_model_from_host",
        lambda *a, **k: pytest.fail("Bubble must retain its existing restore path"),
    )
    assert asyncio.run(trainer.activate_training_model())
    assert "load_model" in events
    assert {"model_restore", "host_head_apply", "head_move", "head_apply"} <= set(
        trainer._last_activation_stage_timings
    )
    assert "load_model" not in trainer._last_activation_stage_timings
