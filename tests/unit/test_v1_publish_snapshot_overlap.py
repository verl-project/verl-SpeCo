# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU contracts for V1 snapshot overlap without importing device runtimes."""

import ast
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _methods(path, class_name, names):
    path = Path(__file__).resolve().parents[2] / path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == names
    for method in methods:
        method.decorator_list = []
    namespace = {"time": time, "logger": logging.getLogger(__name__)}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Subject"]


def _driver():
    cls = _methods(
        "verl_speco/trainer/v1/bubble_mixin.py",
        "V1BubbleMixin",
        {
            "_speco_v1_prefetch_publish_snapshot",
            "_speco_v1_bubble_step_end",
        },
    )
    getter = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_get_published_drafter_weights",
        },
    )
    cls._speco_get_published_drafter_weights = (
        getter._speco_get_published_drafter_weights
    )
    driver = cls()
    driver.global_steps = 4
    driver.total_training_steps = 100
    config = SimpleNamespace(idle_worker_early_publish_staging=True, publish_async=True)
    scheduler = SimpleNamespace(
        _training_quota_debt_steps=0,
        training_work_metrics=lambda: {},
        plan_publish=lambda **kwargs: SimpleNamespace(publish=True),
    )
    plan = SimpleNamespace(
        source_global_step=4, publish_after_success=True, launch=True, plan_id="quota-4"
    )
    session = SimpleNamespace(
        pending=[(plan, SimpleNamespace(trained=True))], close=lambda _: {}
    )
    driver._speco_drafter_schedule_config = lambda: config
    driver._speco_bubble_session = lambda: session
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_colocate_admission_closed = True
    driver._speco_wait_target_lm_head_prefetch = lambda: None
    driver._speco_v1_finish_receipts = lambda: {}
    driver._speco_v1_separate_bubble = lambda: False
    driver._ray_get_if_needed = lambda refs: refs
    driver._first_non_null = lambda values: next(
        (v for v in values if v is not None), None
    )
    events = []
    payload = {"weights_ref": object()}
    driver._require_speco_worker_group = lambda: SimpleNamespace(
        maybe_publish_async=lambda **kw: events.append(("cpu_export", kw))
        or [None, payload],
    )
    driver.speco_maybe_publish = lambda **kw: pytest.fail("duplicate snapshot RPC")

    @contextmanager
    def boundary():
        yield

    driver._speco_v1_quiescent_boundary = boundary
    return driver, config, scheduler, session, events, payload


def test_cpu_export_overlaps_target_sync_and_snapshot_is_consumed_once():
    driver, _, _, _, events, payload = _driver()

    def get_refs(refs):
        assert "target_sync" in events, "snapshot readiness must not block target sync"
        return refs

    driver._ray_get_if_needed = get_refs

    def target_sync():
        assert driver._speco_publish_snapshot_prefetch["source_step"] == 4
        events.append("target_sync")
        # Repeated submission must reuse the refs even before they are consumed.
        assert driver._speco_v1_prefetch_publish_snapshot() == {}
        return "target-result"

    def publish():
        events.append("publish_after_target")
        assert driver._speco_get_published_drafter_weights() is payload
        assert driver._speco_publish_snapshot_prefetch is None
        return {}

    driver._speco_v1_publish_bubble = publish
    assert driver._speco_v1_bubble_step_end(target_sync) == "target-result"
    assert events == [
        ("cpu_export", {"source_step": 4}),
        "target_sync",
        "publish_after_target",
    ]
    assert (
        driver._pending_sync_metrics["bubble/publish_snapshot_prefetch_submitted"] == 1
    )


def test_failed_target_sync_does_not_apply_or_discard_prefetched_weights():
    driver, _, _, _, events, _ = _driver()
    driver._speco_v1_publish_bubble = lambda: pytest.fail("publish after failed sync")

    def fail_sync():
        raise RuntimeError("target sync failed")

    with pytest.raises(RuntimeError, match="target sync failed"):
        driver._speco_v1_bubble_step_end(fail_sync)
    assert driver._speco_publish_snapshot_prefetch["source_step"] == 4
    assert driver._speco_colocate_admission_closed
    assert events == [("cpu_export", {"source_step": 4})]


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "sync_publish",
        "debt",
        "admission",
        "inflight",
        "validation",
        "checkpoint",
        "multiple",
        "untrained",
        "no_publish",
        "old_version",
        "not_launched",
        "publish_interval",
    ],
)
def test_ineligible_prefetch_does_not_touch_workers(guard):
    driver, config, scheduler, session, events, _ = _driver()
    if guard == "disabled":
        config.idle_worker_early_publish_staging = False
    elif guard == "sync_publish":
        config.publish_async = False
    elif guard == "debt":
        scheduler._training_quota_debt_steps = 1
    elif guard == "admission":
        driver._speco_colocate_admission_closed = False
    elif guard == "inflight":
        driver._pending_drafter_publish_refs = [object()]
    elif guard in {"validation", "checkpoint"}:
        setattr(driver, f"_speco_{guard}_in_progress", True)
    elif guard == "multiple":
        session.pending *= 2
    elif guard == "untrained":
        session.pending[0][1].trained = False
    elif guard == "no_publish":
        session.pending[0][0].publish_after_success = False
    elif guard == "not_launched":
        session.pending[0][0].launch = False
    elif guard == "publish_interval":
        scheduler.plan_publish = lambda **kwargs: SimpleNamespace(publish=False)
    else:
        session.pending[0][0].source_global_step = 3
    assert driver._speco_v1_prefetch_publish_snapshot() == {}
    assert events == []


def test_prefetch_cache_miss_falls_back_only_after_target_sync():
    driver, _, _, _, events, payload = _driver()
    driver._require_speco_worker_group = lambda: SimpleNamespace(
        maybe_publish_async=lambda **kw: events.append("cpu_cache_miss") or [None],
    )
    driver._speco_v1_prefetch_publish_snapshot()
    assert events == ["cpu_cache_miss"]
    events.append("target_sync")
    driver.speco_maybe_publish = lambda: events.append("safe_point_export") or [payload]
    assert driver._speco_get_published_drafter_weights() is payload
    assert events == ["cpu_cache_miss", "target_sync", "safe_point_export"]


def test_version_mismatch_and_export_failure_are_not_silently_retried():
    driver, _, _, _, events, _ = _driver()
    driver._speco_publish_snapshot_prefetch = {"source_step": 3, "refs": [object()]}
    with pytest.raises(RuntimeError, match="another publication"):
        driver._speco_v1_prefetch_publish_snapshot()
    with pytest.raises(RuntimeError, match="source version"):
        driver._speco_get_published_drafter_weights()
    assert events == []
    driver._speco_publish_snapshot_prefetch = None

    def fail(**kwargs):
        raise RuntimeError("export failed")

    driver._require_speco_worker_group = lambda: SimpleNamespace(
        maybe_publish_async=fail
    )
    with pytest.raises(RuntimeError, match="export failed"):
        driver._speco_v1_prefetch_publish_snapshot()
    assert driver._speco_publish_snapshot_prefetch is None


def test_prefetch_rejects_another_plan_even_with_the_same_source_step():
    driver, _, _, _, events, _ = _driver()
    driver._speco_publish_snapshot_prefetch = {
        "source_step": 4,
        "plan_id": "another-quota",
        "refs": [object()],
    }
    with pytest.raises(RuntimeError, match="another publication"):
        driver._speco_v1_prefetch_publish_snapshot()
    assert events == []


def test_real_publish_interval_does_not_consume_worker_snapshot():
    from verl_speco.trainer.scheduler import (
        DrafterExecutionStrategy,
        DrafterScheduleConfig,
        DrafterScheduler,
    )

    driver, _, _, session, events, _ = _driver()
    driver.global_steps = 5
    plan = session.pending[0][0]
    plan.source_global_step = 5
    plan.execution_strategy = DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        idle_worker_early_publish_staging=True,
        publish_async=True,
        training_quota_enable=False,
        publish_interval_steps=4,
    )
    driver._speco_drafter_schedule_config = lambda: config
    driver._speco_get_drafter_scheduler = lambda: scheduler
    assert driver._speco_v1_prefetch_publish_snapshot() == {}
    assert events == []


def test_same_step_snapshot_preparation_reuses_cache_without_export():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "clear_pending_publish_state_dict",
            "prepare_model_state_dict_for_publish",
            "pop_model_state_dict_for_publish",
        },
    )
    trainer = cls()
    trainer.clear_pending_publish_state_dict()
    trainer.rank = 0
    calls = []
    snapshot = {"weight": object()}
    trainer.get_model_state_dict = lambda: calls.append("export") or snapshot
    assert trainer.prepare_model_state_dict_for_publish(4)
    assert trainer.prepare_model_state_dict_for_publish(4)
    assert calls == ["export"]
    ready, actual = trainer.pop_model_state_dict_for_publish(4)
    assert ready and actual is snapshot
    assert trainer.pop_model_state_dict_for_publish(4) == (False, None)
