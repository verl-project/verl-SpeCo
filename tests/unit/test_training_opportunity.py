# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest

from verl_speco.trainer.scheduler import (
    AdaptiveScheduleConfig,
    BeforeActorUpdateContext,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    TrainingDataStatus,
)
from verl_speco.trainer.scheduler.training_opportunity import training_opportunity


def config():
    return DrafterScheduleConfig(
        training_interval_steps=5,
        collect_interval_steps=5,
        adaptive_schedule=AdaptiveScheduleConfig(enable=True, warmup_max_steps=3),
    )


def context(step=1, **kwargs):
    return DrafterScheduleContext(
        global_step=step,
        training_mode="online",
        collected_samples_this_step=1,
        oldlogprob_collection_requested=True,
        **kwargs,
    )


def status(step=1):
    return TrainingDataStatus(
        current_step=step,
        current_step_samples=1,
        buffer_samples=1,
        trainable_samples=1,
        trainable_batches=1,
        batch_size_per_gpu=1,
        partial_batch_available=False,
        oldest_sample_step=step,
        newest_sample_step=step,
        same_step_data_required=False,
    )


@pytest.mark.parametrize(
    "step,due,raw",
    [
        (1, True, False),
        (3, False, False),
        (4, False, False),
        (5, True, True),
        (0, False, False),
    ],
)
def test_opportunity_separates_configured_interval_from_startup(step, due, raw):
    decision = training_opportunity(step, config())
    assert decision.due == due
    assert decision.collection_due == due
    assert DrafterScheduler.training_interval_matched(step, config()) == raw


def test_early_exit_is_next_step_and_disabled_startup_preserves_fixed_interval():
    assert training_opportunity(2, config(), startup_ended_after=2).due
    assert not training_opportunity(3, config(), startup_ended_after=2).due
    cfg = replace(config(), adaptive_schedule=AdaptiveScheduleConfig())
    assert not training_opportunity(1, cfg).due
    assert training_opportunity(5, cfg).due


def test_opportunity_evaluation_does_not_initialize_reset_or_consume_controller():
    scheduler = DrafterScheduler()
    for step in (1, 5):
        scheduler.training_opportunity(step, config())
        assert scheduler.adaptive_controller is None
    controller = scheduler.configure_adaptive(config())
    controller.state.warmup_ended_after = 1
    snapshot = asdict(controller.state)
    assert not scheduler.training_opportunity(2, config()).due
    assert scheduler.training_opportunity(5, config()).due
    scheduler.training_opportunity(1, DrafterScheduleConfig())
    assert scheduler.adaptive_controller is controller
    assert asdict(controller.state) == snapshot


@pytest.mark.parametrize("has_status", [False, True])
@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"training_mode": "collect_only"}, "collect_only"),
        ({"pending_training_count": 1}, "pending_training"),
    ],
)
def test_execution_veto_prevents_all_training_rpcs_but_preserves_collection(
    has_status, changes, reason
):
    worker = Mock()
    scheduler = DrafterScheduler(worker_executor=worker)
    collection = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=1,
            source=DrafterCollectionSource.OLD_LOGPROB,
            require_training_interval=True,
        ),
        config(),
    )
    assert collection.collect
    ctx = replace(context(data_status=status() if has_status else None), **changes)
    event = scheduler.on_before_actor_update(BeforeActorUpdateContext(ctx, config()))
    assert not event.training_plan.launch
    assert event.training_plan.reason == reason
    assert event.training_plan.interval_matched
    assert worker.mock_calls == []


def test_missing_current_step_samples_avoids_worker_inspection():
    worker = Mock()
    scheduler = DrafterScheduler(worker_executor=worker)
    plan = scheduler.prepare_training_plan(
        replace(context(), collected_samples_this_step=0), config()
    )
    assert not plan.launch and plan.reason == "no_current_step_oldlogprob_samples"
    assert plan.interval_matched
    assert worker.mock_calls == []


@pytest.mark.parametrize(
    "step,mode,pending,samples,launch,reason",
    [
        (1, "online", 0, 1, False, "interval_not_reached"),
        (1, "online", 0, 0, False, "interval_not_reached"),
        (1, "collect_only", 0, 1, False, "collect_only"),
        (1, "online", 1, 1, False, "pending_training"),
        (5, "online", 0, 1, True, "training_ready"),
        (5, "online", 0, 0, False, "no_current_step_oldlogprob_samples"),
        (5, "collect_only", 0, 1, False, "collect_only"),
        (5, "online", 1, 1, False, "pending_training"),
    ],
)
def test_disabled_startup_preserves_legacy_decisions(
    step, mode, pending, samples, launch, reason
):
    cfg = DrafterScheduleConfig(training_interval_steps=5)
    ctx = replace(
        context(step, data_status=status(step)),
        training_mode=mode,
        pending_training_count=pending,
        collected_samples_this_step=samples,
    )
    plan = DrafterScheduler().prepare_training_plan(ctx, cfg)
    assert (plan.launch, plan.reason) == (launch, reason)


def test_startup_frequency_is_every_step_before_absolute_cap():
    cfg = replace(
        config(),
        training_interval_steps=100,
        collect_interval_steps=100,
        adaptive_schedule=AdaptiveScheduleConfig(enable=True, warmup_max_steps=5),
    )
    assert [step for step in range(8) if training_opportunity(step, cfg).due] == [
        1,
        2,
        3,
        4,
    ]
    disabled = replace(
        cfg, adaptive_schedule=AdaptiveScheduleConfig(enable=True, warmup_max_steps=0)
    )
    assert not training_opportunity(1, disabled).due


def test_scheduler_uses_existing_trigger_policy_with_adaptive_frequency():
    from verl_speco.trainer.scheduler.schedule_types import TriggerDecision

    scheduler = DrafterScheduler()
    scheduler.trigger_policy = Mock()
    scheduler.trigger_policy.should_train.return_value = TriggerDecision(
        False, "pending_training"
    )
    ctx = context(data_status=status())
    plan = scheduler.plan_training(ctx, config())
    assert not plan.launch
    assert plan.reason == "pending_training"
    scheduler.trigger_policy.should_train.assert_called_once_with(
        ctx, config(), interval_matched=True
    )
