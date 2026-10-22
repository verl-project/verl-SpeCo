# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise validation guards without importing the optional Ray/Torch runtime."""

import ast
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def trainer_class():
    path = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    source = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecoRayPPOTrainer"
    )
    methods = {
        "_validate",
        "_speco_try_launch_rollout_idle_training",
        "_speco_start_rollout_idle_event_loop",
        "_speco_service_rollout_idle_events",
        "_speco_drain_rollout_idle_events",
    }

    class Parent:
        def _validate(self):
            assert bool(
                getattr(self, "_speco_validation_in_progress", False)
            ) == getattr(self, "expected_guard", True)
            self.calls.append("parent_validate")
            if self.fail_validation:
                raise ValueError("evaluation failed")
            return "validated"

    selected = ast.ClassDef(
        name="Trainer",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ],
        decorator_list=[],
    )
    namespace = {
        "Parent": Parent,
        "Any": Any,
        "time": __import__("time"),
        "drain_rollout_idle_events": lambda name: [{"event_type": "worker_idle"}],
        "DRAFTER_SAMPLE_READY_EVENT": "drafter_sample_ready",
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["Trainer"]


def test_validation_blocks_thread_launch_and_discards_events(trainer_class):
    trainer = trainer_class()
    trainer._speco_validation_in_progress = True
    trainer._speco_rollout_idle_worker_enabled = lambda: True
    trainer._speco_rollout_idle_event_bus_name = lambda: "test"
    assert trainer._speco_start_rollout_idle_event_loop() == (None, None)
    assert trainer._speco_try_launch_rollout_idle_training() == {
        "bubble/training_blocked_by_validation": 1
    }
    metrics = trainer._speco_service_rollout_idle_events()
    assert metrics["bubble/validation_events_discarded"] == 1


def test_late_validation_samples_are_not_committed_after_resume(trainer_class):
    trainer = trainer_class()
    trainer._speco_runtime_idle_callback_verified = False
    trainer._speco_runtime_idle_events_this_generation = 0
    trainer._speco_rollout_idle_event_bus_name = lambda: "test"
    trainer._speco_ignore_runtime_events_through_ts = 10.0
    committed = []
    trainer._speco_commit_runtime_drafter_samples = (
        lambda samples: committed.extend(samples) or {}
    )
    method = trainer._speco_drain_rollout_idle_events
    method.__func__.__globals__["drain_rollout_idle_events"] = lambda name: [
        {"event_type": "drafter_sample_ready", "event_ts": 9.0},
        {"event_type": "drafter_sample_ready", "event_ts": 11.0},
    ]
    method()
    assert committed == [{"event_type": "drafter_sample_ready", "event_ts": 11.0}]


def test_sync_validation_keeps_original_wait_only_path(trainer_class):
    trainer = trainer_class()
    trainer.calls = []
    trainer.fail_validation = False
    trainer.expected_guard = False
    trainer._speco_rollout_idle_worker_enabled = lambda: False
    trainer._speco_wait_pending_drafter_publish = lambda: trainer.calls.append(
        "publish_wait"
    )
    assert trainer._validate() == "validated"
    assert trainer.calls == ["publish_wait", "parent_validate"]


@pytest.mark.parametrize("failure", [False, True])
def test_validation_quiesces_and_restores_guard_on_error(trainer_class, failure):
    trainer = trainer_class()
    trainer.calls = []
    trainer.fail_validation = failure
    trainer._speco_rollout_idle_worker_enabled = lambda: True
    trainer._speco_bubble_training_lock = nullcontext
    trainer._speco_service_rollout_idle_events = lambda **kwargs: trainer.calls.append(
        ("events", getattr(trainer, "_speco_validation_in_progress", False))
    )
    trainer._speco_reclaim_rollout_idle_workers_before_generation = (
        lambda: trainer.calls.append("reclaim") or {}
    )
    trainer._speco_record_rollout_idle_metrics = lambda metrics: None
    trainer._speco_wait_target_lm_head_prefetch = lambda: trainer.calls.append(
        "head_wait"
    )
    trainer._speco_wait_pending_drafter_publish = lambda: trainer.calls.append(
        "publish_wait"
    )
    if failure:
        with pytest.raises(ValueError, match="evaluation failed"):
            trainer._validate()
    else:
        assert trainer._validate() == "validated"
    assert trainer.calls == [
        ("events", False),
        "reclaim",
        "head_wait",
        "publish_wait",
        "parent_validate",
        ("events", True),
    ]
    assert not trainer._speco_validation_in_progress
    assert trainer._speco_ignore_runtime_events_through_ts > 0
