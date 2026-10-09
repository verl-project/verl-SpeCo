# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import json
from dataclasses import asdict, replace
from itertools import pairwise
from unittest.mock import Mock

import pytest

from verl_speco.trainer.scheduler import (
    AcceptanceFeedback,
    AdaptiveScheduleConfig,
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterRuntimeState,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    PublishOutcome,
    TrainingDataStatus,
)
from verl_speco.trainer.scheduler.adaptive_schedule import AdaptiveScheduleController
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome


def config(**kwargs):
    return AdaptiveScheduleConfig(enable=True, **kwargs)


def feedback(step, length, samples=100):
    return AcceptanceFeedback(step, samples, length)


def observe_and_decide(controller, step, length, samples=100):
    controller.observe(feedback(step, length, samples), current_step=step)
    controller.decide_budget(step)


def update(controller, step):
    controller.record_training(step, 20)
    controller.record_publish(step)


def schedule_config(**kwargs):
    return DrafterScheduleConfig(
        collect_interval_steps=5,
        training_interval_steps=5,
        adaptive_schedule=config(**kwargs),
    )


def context(step, length=3.0, **kwargs):
    return DrafterScheduleContext(
        global_step=step,
        training_mode="online",
        collected_samples_this_step=1,
        oldlogprob_collection_requested=True,
        acceptance_feedback=feedback(step, length),
        **kwargs,
    )


def data_status(step):
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


def test_sudden_length_drop_recovers_budget_quickly():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    controller.state.budget = 5
    observe_and_decide(controller, 1, 8.0)
    observe_and_decide(controller, 2, 1.0)
    observe_and_decide(controller, 3, 1.0)
    assert controller.state.budget == 25
    observe_and_decide(controller, 4, 1.0)
    assert controller.state.budget == 25
    for step in range(5, 8):
        observe_and_decide(controller, step, 1.0)
    assert controller.state.budget >= 20
    assert controller.state.budget <= 25


def test_delta_scales_with_relative_trend_not_fixed_increment():
    def adjustment(length):
        controller = AdaptiveScheduleController(
            config(warmup_max_steps=0, ema_fast_alpha=1.0, ema_slow_alpha=0.01)
        )
        controller.state.budget = 5
        observe_and_decide(controller, 1, 8.0)
        observe_and_decide(controller, 2, length)
        return controller.state.budget

    assert adjustment(2.0) == 25
    assert 5 < adjustment(7.0) < adjustment(2.0)


def test_rising_length_reduces_budget_to_zero_without_new_publication():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    observe_and_decide(controller, 1, 1.0)
    for step in range(2, 30):
        observe_and_decide(controller, step, 8.0)
    assert controller.state.budget == 0
    assert controller.last_reason in {"hold", "acceptance_rise"}


def test_zero_budget_skips_cycle_and_next_cycle_can_recover():
    cfg = schedule_config(warmup_max_steps=0, ema_fast_alpha=1.0, ema_slow_alpha=0.01)
    scheduler = DrafterScheduler(worker_executor=Mock())
    controller = scheduler.configure_adaptive(cfg)
    controller.state.budget = 5
    controller.observe(feedback(0, 1.0), current_step=0, training_interval_steps=5)
    for step in range(1, 5):
        scheduler.prepare_training_plan(
            context(step, 8.0, data_status=data_status(step)), cfg
        )
    skipped = scheduler.prepare_training_plan(
        context(5, 8.0, data_status=data_status(5)), cfg
    )
    assert not skipped.launch and skipped.max_batches == 0
    assert skipped.reason == "no_training_budget"
    assert scheduler.prepare_training_execution(skipped) == {
        "drafter/target_lm_head_synced": 0
    }
    scheduler._worker_executor.prepare_training.assert_not_called()
    assert not scheduler.prepare_training_plan(
        context(6, 1.0, data_status=data_status(6)), cfg
    ).launch
    for step in range(7, 10):
        scheduler.prepare_training_plan(
            context(step, 1.0, data_status=data_status(step)), cfg
        )
    resumed = scheduler.prepare_training_plan(
        context(10, 1.0, data_status=data_status(10)), cfg
    )
    assert resumed.launch and resumed.max_batches > 0
    assert cfg.training_interval_steps == 5
    assert controller.state.last_publish_step == -1


def test_warmup_has_independent_fixed_budget_despite_length_changes():
    cfg = schedule_config(warmup_train_steps=20, max_train_steps=10)
    scheduler = DrafterScheduler()
    for step, length in enumerate((8.0, 1.0, 7.0, 2.0), 1):
        plan = scheduler.prepare_training_plan(
            context(step, length, data_status=data_status(step)), cfg
        )
        assert plan.max_batches == 20
        assert scheduler.adaptive_controller.training_steps(step) == 20
    assert scheduler.adaptive_controller.state.budget == 10


@pytest.mark.parametrize(
    "samples,length",
    [(31, 3.0), (0, 3.0), (float("nan"), 3.0), (100, float("inf")), (100, 0.5)],
)
def test_invalid_or_insufficient_feedback_does_not_update_ema_or_budget(
    samples, length
):
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    observe_and_decide(controller, 1, 4.0)
    before = (
        controller.state.fast,
        controller.state.slow,
        controller.state.budget,
        controller.state.observations,
    )
    observe_and_decide(controller, 2, length, samples)
    assert (
        controller.state.fast,
        controller.state.slow,
        controller.state.budget,
        controller.state.observations,
    ) == before


def test_duplicate_stale_and_future_feedback_does_not_change_state():
    controller = AdaptiveScheduleController(config())
    observe_and_decide(controller, 2, 3.0)
    snapshot = asdict(controller.state)
    for step in (1, 2, 3):
        controller.observe(feedback(step, 1.0), current_step=2)
    assert asdict(controller.state) == snapshot


def test_training_must_be_published_before_feedback_is_consumed():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    controller.record_training(1, 20)
    observe_and_decide(controller, 2, 3.0)
    assert controller.state.fast is None
    controller.record_publish(1)
    observe_and_decide(controller, 3, 3.0)
    assert controller.state.fast == 3.0


def test_warmup_plateau_exits_after_three_full_windows():
    controller = AdaptiveScheduleController(config())
    for step in range(1, 7):
        observe_and_decide(controller, step, 3.0)
        assert controller.state.warmup_ended_after is None
    observe_and_decide(controller, 7, 3.0)
    assert controller.state.warmup_ended_after == 7
    assert not controller.warmup_active(8)


def test_warmup_continuous_improvement_and_absolute_cap():
    controller = AdaptiveScheduleController(config())
    for step in range(1, 20):
        observe_and_decide(controller, step, 1 + step * 0.2)
        assert controller.warmup_active(step)
        assert controller.state.warmup_ended_after is None
    controller.observe(None, current_step=20)
    assert not controller.warmup_active(20)


def test_warmup_gain_at_threshold_resets_plateau_count():
    controller = AdaptiveScheduleController(
        config(warmup_window_size=2, warmup_patience=2, warmup_min_improvement=0.5)
    )
    for step, length in enumerate((2.0, 2.0, 2.5, 3.0), 1):
        observe_and_decide(controller, step, length)
    assert controller.state.warmup_no_improvement_count == 0
    assert controller.state.warmup_ended_after is None


def test_early_exit_keeps_collection_and_training_consistent():
    scheduler = DrafterScheduler()
    cfg = schedule_config(warmup_window_size=2, warmup_patience=1)
    controller = scheduler.configure_adaptive(cfg)
    observe_and_decide(controller, 1, 3.0)
    collection = DrafterCollectionContext(
        global_step=2, source=DrafterCollectionSource.OLD_LOGPROB
    )
    assert scheduler.plan_collection(collection, cfg).collect
    assert (
        scheduler.prepare_training_plan(
            context(2, data_status=data_status(2)), cfg
        ).max_batches
        == 20
    )
    assert scheduler.plan_collection(collection, cfg).collect
    assert not scheduler.plan_collection(
        replace(collection, global_step=3), cfg
    ).collect


def test_disabled_policy_preserves_legacy_budget():
    cfg = DrafterScheduleConfig(training_interval_steps=5, train_batches_per_trigger=37)
    plan = DrafterScheduler().prepare_training_plan(
        context(5, data_status=data_status(5)), cfg
    )
    assert plan.launch and plan.max_batches == 37


def test_explicit_training_step_zero_remains_disabled():
    cfg = replace(schedule_config(), train_batches_per_trigger=0)
    plan = DrafterScheduler().prepare_training_plan(
        context(1, data_status=data_status(1)), cfg
    )
    assert not plan.launch and plan.max_batches == 0


def test_checkpoint_restores_length_ema_zero_budget_and_history(tmp_path):
    controller = AdaptiveScheduleController(config())
    for step in range(1, 6):
        observe_and_decide(controller, step, 3.0)
    controller.state.budget = 0
    path = tmp_path / "adaptive_schedule.json"
    controller.save(path)
    resumed = AdaptiveScheduleController(config())
    assert resumed.restore(path)
    for step in (6, 7, 20):
        observe_and_decide(controller, step, 2.0)
        observe_and_decide(resumed, step, 2.0)
        assert controller.state == resumed.state


def test_changed_config_or_old_schema_recalibrates(tmp_path):
    path = tmp_path / "adaptive_schedule.json"
    controller = AdaptiveScheduleController(config())
    assert not controller.restore(path)
    controller.save(path)
    assert not AdaptiveScheduleController(config(warmup_train_steps=15)).restore(path)
    payload = json.loads(path.read_text())
    payload["schema"] = 1
    path.write_text(json.dumps(payload))
    assert not controller.restore(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_window_size": 1},
        {"warmup_patience": 0},
        {"warmup_train_steps": 0},
        {"min_train_steps": -1},
        {"min_train_steps": 26},
        {"max_step_change": 0},
        {"min_feedback_samples": 0},
        {"ema_slow_alpha": float("nan")},
        {"warmup_min_improvement": -1},
        {"trend_tolerance": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        AdaptiveScheduleConfig(**kwargs)


def test_execution_and_async_publish_lifecycle_update_controller_only_after_success():
    scheduler = DrafterScheduler(publish_executor=Mock())
    cfg = schedule_config()
    plan = scheduler.prepare_training_plan(context(1, data_status=data_status(1)), cfg)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                {"trained": True, "successful_steps": 2, "attempted_steps": 2}
            ],
            elapsed_sec=0.1,
        )
    )
    runtime = DrafterRuntimeState()
    runtime.submit(plan, started_at=0)
    runtime.mark_running()
    scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime))
    assert scheduler.adaptive_controller.state.successful_updates == 1
    scheduler.execute_publish_plan = Mock(return_value=PublishOutcome(True, True))
    scheduler.on_safe_point(
        AfterWeightUpdateContext(1, True, replace(cfg, publish_async=True), plan)
    )
    assert scheduler.adaptive_controller.state.last_publish_step == -1
    scheduler.wait_pending_publish()
    assert scheduler.adaptive_controller.state.last_publish_step == 1


@pytest.mark.parametrize(
    "magnitude,delta", [(0.03, 3), (0.05, 5), (0.15, 15), (0.2, 20), (0.8, 20)]
)
@pytest.mark.parametrize("direction", [-1, 1])
def test_relative_trend_uses_twenty_percent_normalization(magnitude, delta, direction):
    controller = AdaptiveScheduleController(
        config(warmup_max_steps=0, max_train_steps=100)
    )
    controller.state.budget = 20
    observe_and_decide(controller, 0, 3.0)
    # Isolate the adjustment rule from EMA floating-point rounding.
    controller.relative_trend = Mock(return_value=direction * magnitude)
    observe_and_decide(controller, 1, 3.0)
    assert controller.state.budget == 20 - direction * delta


@pytest.mark.parametrize("trend", [-0.02, -0.01, 0.0, 0.01, 0.02])
def test_relative_trend_inside_tolerance_never_changes_budget(trend):
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    controller.relative_trend = Mock(return_value=trend)
    for step in range(1, 12):
        observe_and_decide(controller, step, 3.0)
    assert controller.state.budget == 20


def test_constant_acceptance_length_keeps_budget_unchanged():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    for step in range(1, 30):
        observe_and_decide(controller, step, 3.0)
    assert controller.relative_trend() == 0
    assert controller.state.budget == 20


def test_gradually_improving_length_reduces_budget_in_multiple_adjustments():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    budgets = [controller.state.budget]
    for step in range(1, 40):
        observe_and_decide(controller, step, 2.0 * 1.01**step)
        budgets.append(controller.state.budget)
    assert all(after <= before for before, after in pairwise(budgets))
    assert any(0 < budget < 20 for budget in budgets)
    assert budgets[-1] == 0


@pytest.mark.parametrize("early", [False, True])
def test_warmup_transition_resets_both_emas_to_window_mean(early):
    controller = AdaptiveScheduleController(config(warmup_max_steps=20 if early else 6))
    for step, length in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), 1):
        controller.observe(feedback(step, length), current_step=step)
    assert controller.relative_trend() > 0
    if early:
        controller.state.warmup_ended_after = 5
    controller.observe(feedback(6, 8.0), current_step=6)
    assert controller.state.fast == controller.state.slow == 3.0
    assert controller.relative_trend() == 0
    controller.decide_budget(6)
    assert controller.state.budget == 20
    controller.observe(feedback(7, 4.0), current_step=7)
    assert controller.state.fast == pytest.approx(3.3)
    assert controller.state.slow == pytest.approx(3.05)


def test_real_plateau_exit_resets_ema_once_and_checkpoint_preserves_phase(tmp_path):
    controller = AdaptiveScheduleController(
        config(warmup_window_size=2, warmup_patience=1)
    )
    for step, length in enumerate((1.0, 4.0, 4.0), 1):
        controller.observe(feedback(step, length), current_step=step)
    assert controller.state.warmup_ended_after == 3
    assert controller.relative_trend() > 0
    controller.observe(None, current_step=4)
    assert controller.state.fast == controller.state.slow == 4.0
    path = tmp_path / "phase.json"
    controller.save(path)
    resumed = AdaptiveScheduleController(
        config(warmup_window_size=2, warmup_patience=1)
    )
    assert resumed.restore(path)
    resumed.observe(feedback(5, 2.0), current_step=5)
    assert resumed.state.fast == pytest.approx(3.4)
    assert resumed.state.slow == pytest.approx(3.9)


def test_observe_updates_ema_without_changing_budget_or_reason():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    for step, length in enumerate((1.0, 4.0, 8.0), 1):
        controller.observe(feedback(step, length), current_step=step)
        assert controller.state.budget == 20
        assert controller.last_reason == "initial_budget"
    assert controller.state.fast > controller.state.slow
    controller.observe(feedback(4, 8.0, 1), current_step=4)
    assert controller.last_reason == "initial_budget"


def test_only_training_opportunity_changes_budget_and_reason_once():
    cfg = schedule_config(warmup_max_steps=0)
    scheduler = DrafterScheduler()
    scheduler.configure_adaptive(cfg).observe(
        feedback(0, 1.0), current_step=0, training_interval_steps=5
    )
    for step in range(1, 5):
        plan = scheduler.prepare_training_plan(
            context(step, float(step), data_status=data_status(step)), cfg
        )
        controller = scheduler.adaptive_controller
        assert not plan.launch
        assert controller.state.budget == 20
        assert controller.last_reason == "initial_budget"
    assert controller.state.observations == 5
    plan = scheduler.prepare_training_plan(
        context(5, 5.0, data_status=data_status(5)), cfg
    )
    assert plan.max_batches < 20
    assert controller.last_reason == "acceptance_rise"
    before = asdict(controller.state)
    scheduler.prepare_training_plan(context(5, 5.0, data_status=data_status(5)), cfg)
    assert asdict(controller.state) == before
    budget, reason = controller.state.budget, controller.last_reason
    scheduler.prepare_training_plan(context(6, 1.0, data_status=data_status(6)), cfg)
    assert (controller.state.budget, controller.last_reason) == (budget, reason)


def test_metrics_remain_read_only_at_phase_boundary():
    controller = AdaptiveScheduleController(config(warmup_max_steps=3))
    controller.observe(feedback(1, 1.0), current_step=1)
    controller.observe(feedback(2, 4.0), current_step=2)
    before = asdict(controller.state)
    controller.metrics(3)
    assert asdict(controller.state) == before


@pytest.mark.parametrize(
    "warmup,trained", [(True, True), (False, True), (True, False), (False, False)]
)
def test_budget_metric_and_info_log_only_after_successful_training(
    warmup, trained, caplog
):
    import logging

    cfg = schedule_config(warmup_max_steps=20 if warmup else 0)
    scheduler = DrafterScheduler(
        worker_executor=Mock(prepare_training=Mock(return_value={}))
    )
    step = 1 if warmup else 5
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(context(step, data_status=data_status(step)), cfg)
    )
    assert "drafter/adaptive_budget_steps" not in event.metrics
    plan = event.training_plan
    runtime = DrafterRuntimeState()
    runtime.submit(plan, started_at=0)
    runtime.mark_running()
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                {
                    "trained": trained,
                    "successful_steps": 20 if trained else 0,
                    "attempted_steps": 20,
                }
            ],
            elapsed_sec=0.1,
        )
    )
    with caplog.at_level(logging.INFO):
        result = scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime))
    logs = [r.message for r in caplog.records if "[adaptive_schedule]" in r.message]
    if trained:
        assert result.metrics["drafter/adaptive_budget_steps"] == 20
        assert len(logs) == 1
        if warmup:
            assert (
                logs[0]
                == "[adaptive_schedule] step=1 warmup=true budget=20 reason=warmup_budget"
            )
        else:
            assert (
                logs[0]
                == "[adaptive_schedule] step=5 warmup=false interval_trend=n/a budget=20->20 reason=hold"
            )
    else:
        assert "drafter/adaptive_budget_steps" not in result.metrics
        assert logs == []


def test_non_opportunity_has_no_budget_metric_or_training_log(caplog):
    scheduler = DrafterScheduler(
        worker_executor=Mock(prepare_training=Mock(return_value={}))
    )
    cfg = schedule_config(warmup_max_steps=0)
    event = scheduler.on_before_actor_update(BeforeActorUpdateContext(context(1), cfg))
    assert not event.training_plan.launch
    assert "drafter/adaptive_budget_steps" not in event.metrics
    result = scheduler.on_after_actor_update(
        AfterActorUpdateContext(event.training_plan, DrafterRuntimeState())
    )
    assert "drafter/adaptive_budget_steps" not in result.metrics
    assert not any("[adaptive_schedule]" in r.message for r in caplog.records)


def test_training_log_reports_budget_before_and_after_decision(caplog):
    cfg = schedule_config(warmup_max_steps=0, ema_fast_alpha=1.0, ema_slow_alpha=0.01)
    scheduler = DrafterScheduler(
        worker_executor=Mock(prepare_training=Mock(return_value={}))
    )
    controller = scheduler.configure_adaptive(cfg)
    controller.state.budget = 5
    controller.observe(feedback(0, 8.0), current_step=0, training_interval_steps=5)
    for step in range(1, 5):
        scheduler.prepare_training_plan(
            context(step, 2.0, data_status=data_status(step)), cfg
        )
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(context(5, 2.0, data_status=data_status(5)), cfg)
    )
    plan = event.training_plan
    assert plan.max_batches == 25
    runtime = DrafterRuntimeState()
    runtime.submit(plan, started_at=0)
    runtime.mark_running()
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                {"trained": True, "successful_steps": 25, "attempted_steps": 25}
            ],
            elapsed_sec=0.1,
        )
    )
    scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime))
    logs = [r.message for r in caplog.records if "[adaptive_schedule]" in r.message]
    assert len(logs) == 1
    assert "budget=5->25 reason=acceptance_drop" in logs[0]


@pytest.mark.parametrize("enabled", [True, False])
def test_zero_or_disabled_adaptive_has_no_budget_metric_or_info(enabled, caplog):
    scheduler = DrafterScheduler(
        worker_executor=Mock(prepare_training=Mock(return_value={}))
    )
    cfg = schedule_config(warmup_max_steps=0)
    if enabled:
        scheduler.configure_adaptive(cfg).state.budget = 0
    else:
        cfg = replace(cfg, adaptive_schedule=AdaptiveScheduleConfig())
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(context(5, data_status=data_status(5)), cfg)
    )
    plan = event.training_plan
    runtime = DrafterRuntimeState()
    if plan.launch:
        runtime.submit(plan, started_at=0)
        runtime.mark_running()
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                {"trained": True, "successful_steps": 20, "attempted_steps": 20}
            ],
            elapsed_sec=0.1,
        )
    )
    result = scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime))
    assert "drafter/adaptive_budget_steps" not in event.metrics
    assert "drafter/adaptive_budget_steps" not in result.metrics
    assert not any("[adaptive_schedule]" in r.message for r in caplog.records)
    if enabled:
        scheduler.execute_training_plan.assert_not_called()


@pytest.mark.parametrize("interval", [5, 10])
def test_training_interval_controls_mean_and_decision_direction(interval):
    cfg = replace(
        schedule_config(warmup_max_steps=0),
        training_interval_steps=interval,
    )
    scheduler = DrafterScheduler()
    controller = scheduler.configure_adaptive(cfg)
    controller.observe(
        feedback(0, 3.0), current_step=0, training_interval_steps=interval
    )
    trends = [0.04, 0.05, 0.03, 0.06, 0.01] * (interval // 5)
    for step, trend in enumerate(trends, 1):
        controller.relative_trend = Mock(return_value=trend)
        plan = scheduler.prepare_training_plan(
            context(step, data_status=data_status(step)), cfg
        )
        if step < interval:
            assert controller.state.budget == 20
            assert len(controller.state.interval_trend_history) == step
    # Last-step trend is inside tolerance; interval mean still reduces budget.
    assert controller.last_interval_trend == pytest.approx(0.038)
    assert plan.max_batches == 16
    assert controller.last_reason == "acceptance_rise"
    assert controller.state.interval_trend_history == []
    assert controller.metrics(interval)[
        "drafter/adaptive_interval_trend"
    ] == pytest.approx(0.038)
    before = asdict(controller.state)
    controller.decide_budget(interval, interval)
    assert asdict(controller.state) == before


@pytest.mark.parametrize("invalid_kind", ["missing", "invalid", "stale", "unpublished"])
def test_incomplete_interval_holds_and_does_not_leak_into_next_window(invalid_kind):
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    controller.observe(feedback(0, 3.0), current_step=0, training_interval_steps=5)
    controller.relative_trend = Mock(return_value=0.05)
    for step in range(1, 6):
        sample = feedback(step, 3.0)
        if step == 3:
            if invalid_kind == "missing":
                sample = None
            elif invalid_kind == "invalid":
                sample = feedback(step, 3.0, 1)
            elif invalid_kind == "stale":
                sample = feedback(2, 3.0)
            else:
                controller.record_training(2, 20)
        if step == 4 and invalid_kind == "unpublished":
            controller.record_publish(2)
        controller.observe(sample, current_step=step, training_interval_steps=5)
    assert [row[0] for row in controller.state.interval_trend_history] == [1, 2, 4, 5]
    controller.decide_budget(5, 5)
    assert controller.state.budget == 20
    assert controller.last_reason == "invalid_or_insufficient_feedback"
    assert controller.last_interval_trend is None
    assert controller.state.interval_trend_history == []
    for step in range(6, 11):
        controller.observe(
            feedback(step, 3.0), current_step=step, training_interval_steps=5
        )
    controller.decide_budget(10, 5)
    assert controller.last_interval_trend == pytest.approx(0.05)
    assert controller.state.budget == 15


def test_duplicate_feedback_not_counted_and_skipped_interval_expires():
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    controller.observe(feedback(0, 3.0), current_step=0, training_interval_steps=5)
    for step in range(1, 7):
        controller.observe(
            feedback(step, 4.0), current_step=step, training_interval_steps=5
        )
        controller.observe(
            feedback(step, 4.0), current_step=step, training_interval_steps=5
        )
    assert [row[0] for row in controller.state.interval_trend_history] == [6]


def test_checkpoint_preserves_partial_interval(tmp_path):
    controller = AdaptiveScheduleController(config(warmup_max_steps=0))
    for step in range(4):
        controller.observe(
            feedback(step, 2.0 + step), current_step=step, training_interval_steps=5
        )
    path = tmp_path / "interval.json"
    controller.save(path)
    resumed = AdaptiveScheduleController(controller.config)
    assert resumed.restore(path)
    for step in (4, 5):
        for item in (controller, resumed):
            item.observe(
                feedback(step, 2.0 + step), current_step=step, training_interval_steps=5
            )
    controller.decide_budget(5, 5)
    resumed.decide_budget(5, 5)
    assert resumed.state == controller.state
    assert resumed.last_interval_trend == controller.last_interval_trend


@pytest.mark.parametrize("early", [True, False])
def test_first_adaptive_interval_excludes_all_warmup_trends(early):
    controller = AdaptiveScheduleController(
        config(
            warmup_max_steps=20 if early else 4, warmup_window_size=2, warmup_patience=1
        )
    )
    for step, length in enumerate((1.0, 4.0, 4.0), 1):
        controller.observe(
            feedback(step, length), current_step=step, training_interval_steps=5
        )
        assert controller.state.interval_trend_history == []
    assert controller.relative_trend() > 0
    controller.observe(feedback(4, 4.0), current_step=4, training_interval_steps=5)
    assert controller.relative_trend() == 0
    assert controller.state.interval_trend_history == []
    controller.relative_trend = Mock(return_value=-0.05)
    controller.observe(feedback(5, 3.0), current_step=5, training_interval_steps=5)
    controller.decide_budget(5, 5)
    assert controller.state.budget == 20  # Partial interval cannot use warmup samples.
    assert controller.last_interval_trend is None
    for step in range(6, 11):
        controller.observe(
            feedback(step, 3.0), current_step=step, training_interval_steps=5
        )
    controller.decide_budget(10, 5)
    assert controller.last_interval_trend == pytest.approx(-0.05)
    assert controller.state.budget == 25  # First full interval acts immediately.
