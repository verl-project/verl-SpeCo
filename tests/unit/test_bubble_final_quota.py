# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Final quota completion must precede final checkpoint and validation."""

import time
from dataclasses import replace
from types import SimpleNamespace
from types import MethodType
from contextlib import contextmanager
import dataclasses

import pytest

from test_bubble_review_regressions import _methods
from test_rollout_idle_worker_scheduler import (
    _scheduler_with_statuses,
    _auto_idle_config,
    _context,
    _status,
)
from verl_speco.trainer.scheduler.lifecycle import BeforeActorUpdateContext
from verl_speco.trainer.scheduler import (
    TrainingPlan,
    DrafterExecutionStrategy,
    CollectionPlan,
    CollectionOutcome,
)


def _driver(*, step=50, bubble=True, debt=0):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_is_final_training_step", "_speco_finalize_quota_publication"},
        time=time,
    )
    driver = cls()
    driver.global_steps, driver.total_training_steps = step, 50
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver._speco_drafter_schedule_config = lambda: SimpleNamespace(
        training_quota_enable=True
    )
    driver._pending_drafter_publish_refs = None
    driver._speco_deferred_publish_count = 0
    scheduler = SimpleNamespace(
        _training_quota_debt_steps=debt,
        training_work_metrics=lambda: {"bubble/training_quota_debt_steps": debt},
    )
    driver._speco_get_drafter_scheduler = lambda: scheduler
    return driver


@pytest.mark.parametrize("step,bubble", [(20, True), (50, False)])
def test_periodic_and_sync_boundaries_do_not_drain(step, bubble):
    driver = _driver(step=step, bubble=bubble, debt=20)
    assert (
        driver._speco_finalize_quota_publication(lambda **kw: pytest.fail("drain"))
        == {}
    )


def test_unfinished_final_quota_fails_instead_of_silent_success():
    driver = _driver(debt=20)
    with pytest.raises(RuntimeError, match="remaining_optimizer_steps=20"):
        driver._speco_finalize_quota_publication(lambda **kw: pytest.fail("drain"))


def test_final_waits_for_apply_ack_before_checkpoint():
    driver = _driver()
    driver._speco_deferred_publish_count = 1
    calls = []

    def drain(**kw):
        assert kw["safe_point"] == "final_training_step"
        calls.append("stage")
        driver._speco_deferred_publish_count = 0
        driver._pending_drafter_publish_refs = ["pending-apply"]
        return {}

    def wait():
        calls.append("apply_ack")
        driver._pending_drafter_publish_refs = None

    driver._speco_wait_pending_drafter_publish = wait
    metrics = driver._speco_finalize_quota_publication(drain)
    calls.extend(["checkpoint", "validation"])
    assert calls == ["stage", "apply_ack", "checkpoint", "validation"]
    assert metrics["bubble/final_quota_completed"] == 1


def test_final_propagates_publish_failure():
    driver = _driver()
    driver._pending_drafter_publish_refs = ["bad-ack"]

    def wait():
        raise RuntimeError("invalid apply ACK")

    driver._speco_wait_pending_drafter_publish = wait
    with pytest.raises(RuntimeError, match="invalid apply ACK"):
        driver._speco_finalize_quota_publication(lambda **kw: {})


def test_final_no_progress_does_not_loop():
    driver = _driver()
    driver._speco_deferred_publish_count = 1
    with pytest.raises(RuntimeError, match="no progress"):
        driver._speco_finalize_quota_publication(lambda **kw: {})


@pytest.mark.parametrize("force", [False, True])
def test_last_quota_completes_before_normal_lag_deadline(force):
    scheduler = _scheduler_with_statuses(("0", "1"))
    group = ("0", "1")
    scheduler._metadata_idle_training_groups = (group,)
    scheduler._idle_worker_writer_group = group
    scheduler._idle_worker_writer_state_version = 44
    scheduler._idle_optimizer_initialized_groups.add(group)
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_oldest_cycle_step = 48
    scheduler._training_quota_last_cycle_step = 48
    scheduler._training_quota_data_version = 48
    scheduler._worker_executor = replace(
        scheduler._worker_executor,
        inspect_data=lambda *args, **kw: [
            {
                **replace(
                    _status(worker),
                    data_version=48,
                    target_version=48,
                    oldest_sample_step=48,
                    newest_sample_step=48,
                ).__dict__,
                "available": True,
            }
            for worker in group
        ],
    )
    config = replace(
        _auto_idle_config(2),
        training_quota_enable=True,
        training_interval_steps=4,
        training_quota_max_completion_lag_steps=3,
        training_quota_max_sync_topup_steps=2,
    )
    context = replace(_context(), global_step=50, collected_samples_this_step=0)
    plan = scheduler._plan_training_quota_topup(context, config, force_completion=force)
    if force:
        assert plan.launch and plan.max_batches == 20
        assert plan.target_worker_ids == group
        assert plan.required_target_version == plan.data_version == 48
        assert plan.reason == "quota_forced_completion_ready"
    else:
        assert plan is None


def test_force_only_from_explicit_final_context():
    scheduler = _scheduler_with_statuses(("0",))
    config = replace(_auto_idle_config(1), training_quota_enable=True)
    calls = []
    scheduler._plan_training_quota_topup = lambda *args, **kw: calls.append(kw) or None
    scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=_context(),
            config=config,
            force_quota_completion=True,
        )
    )
    assert calls == [{"force_completion": True}]


def test_final_actor_hook_trains_and_publishes_before_returning_to_save(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_online_fit_hooks",
            "_speco_is_final_training_step",
            "_speco_finalize_quota_publication",
            "_speco_merge_training_metrics",
        },
        time=time,
        MethodType=MethodType,
        dataclasses=dataclasses,
        CollectionPlan=CollectionPlan,
        CollectionOutcome=CollectionOutcome,
        DrafterExecutionStrategy=DrafterExecutionStrategy,
        _speco_optional_float=lambda value: float(value) if value is not None else None,
        _speco_metric_float=lambda value: float(value) if value is not None else None,
    )
    driver = cls()
    driver.global_steps, driver.total_training_steps = 50, 50
    driver._pending_drafter_publish_refs = None
    driver._pending_target_lm_head_sync = None
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_drafter_schedule_config = lambda: SimpleNamespace(
        training_quota_enable=True, training_quota_allow_critical_path_fallback=True
    )
    calls = []
    scheduler = SimpleNamespace(
        _training_quota_debt_steps=20,
        needs_idle_optimizer_bootstrap=lambda config: False,
        plan_publish=lambda **kw: SimpleNamespace(publish=True),
    )
    scheduler.training_work_metrics = lambda: {
        "bubble/training_quota_debt_steps": scheduler._training_quota_debt_steps
    }
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_rollout_generation_target = lambda: SimpleNamespace(
        generate_sequences=lambda: None
    )
    driver.checkpoint_manager = SimpleNamespace(update_weights=lambda *args: None)
    driver._compute_old_log_prob = lambda batch: batch
    driver._speco_oldlogprob_collection_requested = lambda: False
    driver._speco_oldlogprob_entropy_hook_enabled = lambda: False
    result = SimpleNamespace(meta_info={"metrics": {}})
    driver._update_actor = lambda *args: calls.append("actor") or result
    driver._speco_pop_rollout_idle_metrics = lambda: {}
    driver._speco_poll_pending_drafter_training = lambda: (None, None)
    driver._speco_poll_target_lm_head_prefetch = lambda: {}
    plan = TrainingPlan(
        launch=True,
        reason="quota_forced_completion_ready",
        interval_matched=False,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=50,
        max_batches=20,
        publish_after_success=True,
        required_target_version=48,
        data_version=48,
        target_worker_ids=("0", "1"),
    )
    planned = []

    def before(**kw):
        assert kw["force_quota_completion"]
        planned.append(True)
        return SimpleNamespace(
            training_plan=replace(plan, launch=False) if len(planned) == 1 else plan,
            metrics={},
        )

    driver._speco_on_before_actor_update = before
    driver._speco_log_drafter_training_plan = lambda *args: None
    driver._speco_should_prefetch_target_lm_head = lambda: False
    driver._speco_park_drafter_training_memory = lambda *args: {}
    driver._speco_wait_target_lm_head_prefetch = (
        lambda: calls.append("head_ready") or {}
    )
    driver._speco_gate_bubble_quota_topup = lambda requested: (requested, {})

    def train(requested):
        assert requested.required_target_version == 48 and requested.max_batches == 20
        calls.append("topup20")
        scheduler._training_quota_debt_steps = 0
        return True, {"drafter/trained": 1, "drafter/train_successful_steps_max": 20}

    driver._speco_train_drafter = train

    def publish(*args, **kw):
        calls.append("stage")
        driver._pending_drafter_publish_refs = ["apply"]
        return {"drafter/published": 1}

    driver._speco_publish_drafter_weights = publish

    def wait():
        if driver._pending_drafter_publish_refs:
            calls.append("apply_ack")
            driver._pending_drafter_publish_refs = None

    driver._speco_wait_pending_drafter_publish = wait
    driver._speco_maybe_save_freeze_branch_checkpoint = lambda: None
    driver._speco_flush_request_accept_len_payloads = lambda: None
    driver._speco_update_output_metrics = (
        lambda output, metrics: output.meta_info["metrics"].update(metrics) or output
    )
    with contextmanager(driver._speco_online_fit_hooks)():
        output = driver._update_actor(object())
        calls.extend(["checkpoint", "validation"])
        assert calls == [
            "actor",
            "head_ready",
            "topup20",
            "stage",
            "apply_ack",
            "checkpoint",
            "validation",
        ]
        assert output.meta_info["metrics"]["bubble/final_quota_completed"] == 1
