# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded hot-writer readmission after an underestimated rollout tail."""

from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_bubble_review_regressions import _methods
from test_rollout_idle_worker_scheduler import (
    _auto_idle_config,
    _context,
    _scheduler_with_statuses,
    _status,
)
from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    IdleWindowConfidence,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    TrainingPlan,
    TrainingOutcome,
)


def _hot_writer(monkeypatch):
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: 100.0
    )
    scheduler = _scheduler_with_statuses(("0", "1"))
    group = ("0", "1")
    scheduler._metadata_idle_training_groups = (group,)
    scheduler._idle_worker_writer_group = group
    scheduler._idle_worker_writer_state_version = 8
    scheduler._idle_worker_hot_prewarmed_groups.add(group)
    scheduler._idle_optimizer_initialized_groups.add(group)
    scheduler._training_quota_debt_steps = 20
    for worker in group:
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker,
                replica_rank=0,
                event_ts=99.0,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
            )
        )
    # Replay the long/short alternation seen around log44 step18. Baseline
    # recent median is short, but a live one-second tail rules those out.
    history = [3.2, 4.5, 6.5, 0.087, 0.117, 0.883]
    scheduler._replica_idle_window_samples_sec[0] = deque(history, maxlen=32)
    scheduler._replica_idle_windows_by_source[0] = {
        "generation_completed": deque(history, maxlen=32)
    }
    config = replace(
        _auto_idle_config(2),
        training_quota_enable=True,
        idle_worker_keep_hot_between_plans=True,
        idle_worker_historical_window_readmission=True,
        idle_worker_initial_batch_estimate_sec=0.15,
        idle_worker_deadline_guard_sec=0.05,
    )
    return scheduler, config, group


def test_readmission_defaults_off_and_parses_explicit_opt_in():
    assert not DrafterScheduleConfig().idle_worker_historical_window_readmission
    config = DrafterScheduleConfig.from_mapping(
        {"scheduler": {"idle_worker": {"historical_window_readmission": True}}}
    )
    assert config.idle_worker_historical_window_readmission


def _live_episode_plan(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: clock[0]
    )
    scheduler._training_quota_data_version = 8
    scheduler._replica_idle_window_samples_sec[0] = deque([3.0, 4.0, 5.0])
    scheduler._replica_idle_windows_by_source[0]["generation_completed"] = deque(
        [3.0, 4.0, 5.0]
    )
    context = replace(
        _context(),
        data_status=replace(_status("0", batches=20), data_version=8, target_version=8),
    )
    plan = scheduler.prepare_idle_worker_training_plan(context, config)
    assert plan.launch and plan.idle_generation_epochs == (("0", 0), ("1", 0))
    assert "idle_generation_epochs" not in plan.to_worker_payload()
    return scheduler, config, group, context, plan, clock


def _next_idle_episode(
    scheduler, group, *, busy_workers=None, busy_ts=103.9, idle_ts=104.0, deadline=106.0
):
    for worker in group if busy_workers is None else busy_workers:
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker,
                replica_rank=0,
                event_ts=busy_ts,
            )
        )
    for worker in group:
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker,
                replica_rank=0,
                event_ts=idle_ts,
                must_be_ready_at=deadline,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
            )
        )


@pytest.mark.parametrize("reclaimed", [False, True])
def test_new_complete_idle_episode_detaches_old_deadline(monkeypatch, reclaimed):
    scheduler, config, group, context, previous, clock = _live_episode_plan(monkeypatch)
    previous = replace(previous, idle_readmission_blocked=reclaimed)
    clock[0] = 104.0
    assert previous.deadline_ts < clock[0]
    _next_idle_episode(scheduler, group)
    assert scheduler.idle_continuation_for_episode(previous, config) is None
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    assert plan.launch and not plan.idle_continuation
    assert 104.0 < plan.deadline_ts <= 106.0
    assert plan.idle_generation_epochs == (("0", 1), ("1", 1))
    assert plan.data_version == previous.data_version == 8
    assert plan.required_target_version == previous.required_target_version == 8


@pytest.mark.parametrize(
    "guard",
    [
        "duplicate_idle",
        "stale_start",
        "partial_group",
        "busy",
        "memory",
        "speculative",
        "data_version",
        "head_version",
        "legacy",
    ],
)
def test_new_episode_does_not_bypass_existing_protections(monkeypatch, guard):
    scheduler, config, group, context, previous, clock = _live_episode_plan(monkeypatch)
    previous = replace(previous, idle_readmission_blocked=True)
    clock[0] = 104.0
    _next_idle_episode(
        scheduler,
        group,
        busy_workers=()
        if guard == "duplicate_idle"
        else (group[:1] if guard == "partial_group" else None),
        busy_ts=98.0 if guard == "stale_start" else 103.9,
    )
    if guard == "busy":
        scheduler._idle_workers[group[1]].status = "generating"
    elif guard == "memory":
        scheduler._idle_workers[group[1]].memory_released = False
    elif guard == "speculative":
        scheduler._idle_workers[
            group[1]
        ].idle_confidence = IdleWindowConfidence.SPECULATIVE
    elif guard == "data_version":
        scheduler._training_quota_data_version = 9
    elif guard == "head_version":
        previous = replace(previous, required_target_version=7)
    elif guard == "legacy":
        previous = replace(previous, idle_generation_epochs=())
    assert scheduler.idle_continuation_for_episode(previous, config) is previous
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    assert not plan.launch


def test_new_episode_still_obeys_new_runtime_deadline(monkeypatch):
    scheduler, config, group, context, previous, clock = _live_episode_plan(monkeypatch)
    clock[0] = 104.0
    _next_idle_episode(scheduler, group, deadline=103.99)
    assert scheduler.idle_continuation_for_episode(previous, config) is None
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    assert not plan.launch
    assert not scheduler._idle_readmission_attempts


def test_driver_clears_old_reclaim_gate_even_when_new_admission_is_too_small(
    monkeypatch,
):
    import time
    from verl_speco.trainer.scheduler import DrafterRuntimeStatus

    scheduler, config, group, _, previous, clock = _live_episode_plan(monkeypatch)
    clock[0] = 104.0
    _next_idle_episode(scheduler, group)
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_try_launch_rollout_idle_training"},
        DrafterRuntimeStatus=DrafterRuntimeStatus,
        time=time,
    )
    driver = cls()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._pending_drafter_publish_refs = None
    driver._speco_idle_continuation_plan = replace(
        previous, idle_readmission_blocked=True
    )
    driver._speco_drafter_schedule_config = lambda: config
    driver._speco_bubble_training_lock = nullcontext
    driver._speco_get_drafter_runtime_state = lambda: SimpleNamespace(
        status=DrafterRuntimeStatus.IDLE
    )
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_poll_target_lm_head_prefetch = lambda: {}
    driver._speco_log_drafter_training_plan = lambda *args: None
    driver._speco_migrate_idle_writer_if_useful = lambda config: {}
    driver._speco_ready_target_lm_head_workers = {8: group}
    driver._speco_ready_target_lm_head_versions = {8}
    driver._speco_target_lm_head_ready_for_workers = lambda *args: True
    calls = []

    def schedule(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            training_plan=replace(previous, launch=False), metrics={}
        )

    driver._speco_on_before_actor_update = schedule
    driver._speco_try_launch_rollout_idle_training()
    assert driver._speco_idle_continuation_plan is None
    assert calls == [{"allow_quota_topup": False, "idle_continuation_plan": None}]


def test_reordered_idle_callback_cannot_make_old_busy_event_a_new_epoch(monkeypatch):
    scheduler, config, group, _, previous, clock = _live_episode_plan(monkeypatch)
    clock[0] = 104.0
    _next_idle_episode(scheduler, group, busy_workers=())
    for worker in group:
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker,
                replica_rank=0,
                event_ts=98.0,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
            )
        )
    _next_idle_episode(scheduler, group)
    assert scheduler.idle_continuation_for_episode(previous, config) is previous
    assert all(scheduler._idle_workers[w].generation_epoch == 0 for w in group)


def test_short_forecast_recovers_only_after_live_idle_survives_it(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    assert scheduler._effective_historical_idle_window_sec(worker_ids=group) < 1.0
    assert (
        scheduler._historical_idle_readmission_window(config, group, now=99.05) is None
    )
    assert scheduler._historical_idle_readmission_window(
        config, group, now=100.0
    ) == pytest.approx(2.2)
    assert scheduler.should_retry_idle_admission(config, now=100.0)
    assert not scheduler.should_retry_idle_admission(config, now=100.1)
    assert scheduler.should_retry_idle_admission(config, now=100.3)


def test_log44_step18_short_forecast_can_be_reconsidered(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    # Replica0 completed tails at steps9..17, in order, from log44. The last
    # three contain a 4.97s tail, but their median caps the forecast at 0.883s.
    tails = [
        4.5136914253,
        0.1059157848,
        0.1124265194,
        6.5082399845,
        0.0874090195,
        0.1168115139,
        0.8827855587,
        4.9743146896,
        0.1002292633,
    ]
    scheduler._replica_idle_window_samples_sec[0] = deque(tails, maxlen=32)
    scheduler._replica_idle_windows_by_source[0]["generation_completed"] = deque(
        tails, maxlen=32
    )
    assert scheduler._effective_historical_idle_window_sec(
        worker_ids=group
    ) == pytest.approx(0.8827855587)
    scheduler._replica_idle_started_at[0] = 99.05
    for worker in group:
        scheduler._idle_workers[worker].event_ts = 99.05
    # This decision uses only past tails and the live 0.95s age. It must not
    # use step18's future observed 2.092s window to manufacture a deadline.
    assert scheduler.should_retry_idle_admission(config, now=100.0)
    plan = scheduler.prepare_idle_worker_training_plan(
        replace(_context(), data_status=_status("0", batches=20)), config
    )
    assert plan.launch and 1 < plan.max_batches < scheduler._training_quota_debt_steps
    assert plan.idle_window_source == "historical_survival"


def test_readmission_is_one_attempt_with_budgeted_optimizer_steps(monkeypatch, caplog):
    scheduler, config, group = _hot_writer(monkeypatch)
    context = replace(_context(), data_status=_status("0", batches=20))
    plan = scheduler.prepare_idle_worker_training_plan(context, config)
    assert plan.launch
    assert plan.target_worker_ids == group
    assert plan.idle_window_source == "historical_survival"
    assert 1 < plan.max_batches < scheduler._training_quota_debt_steps
    assert plan.deadline_ts <= 102.2
    assert not plan.publish_after_success
    assert not scheduler.should_retry_idle_admission(config, now=100.3)
    assert not scheduler.select_idle_training_resources(config, now=100.3).available
    assert "step_limit=window_budget" in caplog.text
    assert "generation_epochs=" in caplog.text
    assert "max_steps=1" not in caplog.text


@pytest.mark.parametrize(
    "episode", ["duplicate_idle", "stale_start", "partial", "full"]
)
def test_readmission_token_requires_a_new_generation_on_every_writer_rank(
    monkeypatch, episode
):
    scheduler, config, group = _hot_writer(monkeypatch)
    scheduler._training_quota_data_version = 8
    context = replace(
        _context(),
        data_status=replace(_status("0", batches=20), data_version=8, target_version=8),
    )
    previous = scheduler.prepare_idle_worker_training_plan(context, config)
    assert previous.launch and previous.idle_window_source == "historical_survival"
    old_deadline = previous.deadline_ts
    consumed_epochs = dict(scheduler._idle_readmission_attempts[group])
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: 105.0
    )
    _next_idle_episode(
        scheduler,
        group,
        busy_workers=(
            ()
            if episode == "duplicate_idle"
            else group[:1]
            if episode == "partial"
            else group
        ),
        busy_ts=98.0 if episode == "stale_start" else 103.9,
        deadline=None,
    )
    # Later idle timestamps alone must not renew an expired conditional bound.
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    assert plan.launch == (episode == "full")
    if episode == "full":
        assert 1 < plan.max_batches < scheduler._training_quota_debt_steps
        assert plan.deadline_ts > 105.0 > old_deadline
        assert all(
            scheduler._idle_readmission_attempts[group][worker]
            > consumed_epochs[worker]
            for worker in group
        )
    else:
        assert previous.deadline_ts == old_deadline < 105.0
        assert scheduler._idle_readmission_attempts[group] == consumed_epochs
        assert not scheduler.should_retry_idle_admission(config, now=105.3)


def test_retry_consumes_inspection_time_instead_of_resetting_window(monkeypatch):
    scheduler, config, _ = _hot_writer(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: clock[0]
    )

    def slow_inspection(**kwargs):
        clock[0] = 103.0
        return _status("0", batches=20)

    scheduler.inspect_training_data = slow_inspection
    plan = scheduler.prepare_idle_worker_training_plan(_context(), config)
    assert not plan.launch
    assert plan.max_batches == 0
    assert not scheduler.should_retry_idle_admission(config, now=103.1)


def test_log50_conditional_window_preserves_three_step_budget(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    config = replace(
        config,
        gradient_accumulation_steps=2,
        idle_worker_initial_batch_estimate_sec=0.1763777,
    )
    # Only completed tails before step10, never its future 5.97s observation.
    tails = [
        0.146453,
        2.216355,
        0.097827,
        1.860091,
        8.649169,
        0.130143,
        0.108398,
        0.095439,
        0.174703,
    ]
    scheduler._replica_idle_window_samples_sec[0] = deque(tails, maxlen=32)
    scheduler._replica_idle_windows_by_source[0]["generation_completed"] = deque(
        tails, maxlen=32
    )
    scheduler._replica_idle_started_at[0] = 99.577
    for worker in group:
        scheduler._idle_workers[worker].event_ts = 99.577
    plan = scheduler.prepare_idle_worker_training_plan(
        replace(_context(), data_status=_status("0", batches=20)), config
    )
    assert plan.launch and plan.max_batches == 3
    assert plan.idle_window_source == "historical_survival"
    assert plan.gradient_accumulation_steps == 2
    assert plan.deadline_ts <= 101.387092
    assert plan.max_batches * 0.3527554 <= plan.deadline_ts - 100.0
    assert not plan.publish_after_success
    assert not scheduler.select_idle_training_resources(config, now=100.3).available


@pytest.mark.parametrize("debt", [1, 2, 3, 20])
def test_conditional_plan_never_finishes_quota_or_stages_snapshot(monkeypatch, debt):
    scheduler, config, _ = _hot_writer(monkeypatch)
    scheduler._training_quota_debt_steps = debt
    plan = scheduler.prepare_idle_worker_training_plan(
        replace(_context(), data_status=_status("0", batches=20)), config
    )
    assert plan.launch == (debt > 1)
    assert plan.max_batches <= max(debt - 1, 0)
    assert not plan.publish_after_success


@pytest.mark.parametrize(
    "blocked,source,version,head_version",
    [
        (False, "historical_observed", 8, 8),
        (True, "historical_observed", 8, 8),
        (False, "runtime_deadline", 8, 8),
        (False, "historical_observed", 7, 7),
        (False, "historical_observed", 8, None),
        (False, "historical_observed", 8, 7),
    ],
)
@pytest.mark.parametrize("use_logits", [False, True])
def test_expired_short_plan_reenters_only_same_unreclaimed_quota(
    monkeypatch, blocked, source, version, head_version, use_logits
):
    scheduler, config, group = _hot_writer(monkeypatch)
    config = replace(config, use_logits=use_logits)
    scheduler._training_quota_data_version = 8
    previous = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=False,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=9,
        max_batches=2,
        publish_after_success=False,
        deadline_ts=99.9,
        target_worker_ids=group,
        idle_window_source=source,
        data_version=version,
        required_target_version=None if use_logits else head_version,
        idle_readmission_blocked=blocked,
    )
    context = replace(
        _context(),
        data_status=replace(_status("0", batches=20), data_version=8, target_version=8),
    )
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    allowed = (
        not blocked
        and source == "historical_observed"
        and version == 8
        and (use_logits or head_version == 8)
    )
    assert plan.launch == allowed
    if allowed:
        assert 1 < plan.max_batches < scheduler._training_quota_debt_steps
        assert not plan.idle_continuation
        assert plan.idle_window_source == "historical_survival"
        assert plan.deadline_ts > 100.0
        assert plan.data_version == previous.data_version
        assert plan.required_target_version == (None if use_logits else 8)
    else:
        assert not scheduler._idle_readmission_attempts


@pytest.mark.parametrize(
    "source,expected",
    [("historical_observed", 20), ("historical_survival", 20), ("runtime_deadline", 1)],
)
def test_predicted_expiry_does_not_poison_next_window_cap(source, expected):
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._idle_group_batch_caps[("0",)] = 20
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=False,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=9,
        max_batches=2,
        publish_after_success=False,
        target_worker_ids=("0",),
        idle_window_source=source,
        idle_window_sec=1.1,
    )
    outcome = TrainingOutcome(
        trained=True,
        reason="deadline_reached",
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        metrics={"timing_s/drafter_worker_elapsed": 0.9},
    )
    scheduler._record_idle_dynamic_batch_cap(plan, outcome)
    assert scheduler._idle_group_batch_caps[("0",)] == expected


@pytest.mark.parametrize("elapsed", [None, 2.0, float("nan")])
def test_historical_expiry_still_reduces_cap_on_overrun_or_missing_timing(elapsed):
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._idle_group_batch_caps[("0",)] = 20
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=False,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=9,
        max_batches=2,
        publish_after_success=False,
        target_worker_ids=("0",),
        idle_window_source="historical_observed",
        idle_window_sec=1.1,
    )
    outcome = TrainingOutcome(
        trained=True,
        reason="deadline_reached",
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=2.0,
        metrics={} if elapsed is None else {"timing_s/drafter_worker_elapsed": elapsed},
    )
    scheduler._record_idle_dynamic_batch_cap(plan, outcome)
    assert scheduler._idle_group_batch_caps[("0",)] == 1


def test_new_idle_episode_can_retry_but_only_with_remaining_support(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    resources = scheduler.select_idle_training_resources(config, now=100.0)
    assert resources.available
    assert (
        scheduler._historical_idle_readmission_window(config, group, now=100.1) is None
    )
    # Only a real intervening generation earns a new attempt, not a later
    # idle notification in the same episode.
    _next_idle_episode(scheduler, group, busy_ts=108.9, idle_ts=109.0, deadline=None)
    assert scheduler._historical_idle_readmission_window(
        config, group, now=110.0
    ) == pytest.approx(2.2)
    assert (
        scheduler._historical_idle_readmission_window(config, group, now=120.0) is None
    )


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "sync",
        "no_quota",
        "keep_cold",
        "final_step",
        "cold_model",
        "cold_optimizer",
        "no_writer_state",
        "reclaimed",
        "busy",
        "memory",
        "speculative",
        "deadline",
        "too_few_tails",
        "other_replica_only",
        "old_tails",
    ],
)
def test_readmission_preserves_safety_guards(monkeypatch, guard):
    scheduler, config, group = _hot_writer(monkeypatch)
    if guard == "disabled":
        config = replace(config, idle_worker_historical_window_readmission=False)
    elif guard == "sync":
        config = replace(config, execution_strategy=DrafterExecutionStrategy.SYNC)
    elif guard == "no_quota":
        config = replace(config, training_quota_enable=False)
    elif guard == "keep_cold":
        config = replace(config, idle_worker_keep_hot_between_plans=False)
    elif guard == "final_step":
        scheduler._training_quota_debt_steps = 1
    elif guard == "cold_model":
        scheduler._idle_worker_hot_prewarmed_groups.clear()
    elif guard == "cold_optimizer":
        scheduler._idle_optimizer_initialized_groups.clear()
    elif guard == "no_writer_state":
        scheduler._idle_worker_writer_state_version = None
    elif guard == "reclaimed":
        scheduler._idle_worker_prebatch_reclaim_streak = 1
    elif guard == "busy":
        scheduler._idle_workers["1"].status = "generating"
    elif guard == "memory":
        scheduler._idle_workers["1"].memory_released = False
    elif guard == "speculative":
        scheduler._idle_workers["1"].idle_confidence = IdleWindowConfidence.SPECULATIVE
    elif guard == "deadline":
        # Even an expired explicit deadline must not be replaced by history.
        scheduler._idle_workers["1"].must_be_ready_at = 99.5
    elif guard == "too_few_tails":
        scheduler._replica_idle_windows_by_source[0]["generation_completed"] = deque(
            [3.2, 4.5]
        )
    elif guard == "other_replica_only":
        scheduler._replica_idle_windows_by_source[1] = (
            scheduler._replica_idle_windows_by_source.pop(0)
        )
    elif guard == "old_tails":
        scheduler._replica_idle_windows_by_source[0]["generation_completed"].extend(
            [0.1] * 16
        )
    assert (
        scheduler._historical_idle_readmission_window(config, group, now=100.0) is None
    )


def test_readmission_never_restarts_expired_deadline(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    for worker in group:
        scheduler._idle_workers[worker].must_be_ready_at = 99.5
    resources = scheduler.select_idle_training_resources(config, now=100.0)
    assert not resources.available
    assert resources.minimum_idle_window_sec == 0.0


@pytest.mark.parametrize(
    "open_generation,allow_launch", [(True, True), (False, True), (True, False)]
)
def test_quiet_poll_rechecks_only_while_generation_is_open(
    open_generation, allow_launch
):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_service_rollout_idle_events"},
    )
    driver = cls()
    driver._speco_rollout_idle_generation_open = open_generation
    driver._speco_historical_window_readmission_enabled = True
    driver._speco_bubble_training_lock = nullcontext
    driver._speco_drain_rollout_idle_events = lambda: {}
    driver._speco_release_completed_rollout_training_lease = lambda: {}
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_drafter_schedule_config = lambda: object()
    checks, launches = [], []
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        should_retry_idle_admission=lambda config: checks.append(True) or True
    )
    driver._speco_try_launch_rollout_idle_training = lambda: launches.append(True) or {}
    driver._speco_record_rollout_idle_metrics = lambda metrics: None
    driver._speco_service_rollout_idle_events(allow_launch=allow_launch)
    assert bool(checks) == (open_generation and allow_launch)
    assert bool(launches) == (open_generation and allow_launch)


def test_disabled_quiet_poll_adds_no_scheduler_queries():
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_service_rollout_idle_events"},
    )
    driver = cls()
    driver._speco_rollout_idle_generation_open = True
    driver._speco_bubble_training_lock = nullcontext
    driver._speco_drain_rollout_idle_events = lambda: {}
    driver._speco_release_completed_rollout_training_lease = lambda: {}
    driver._speco_record_rollout_idle_metrics = lambda metrics: None
    driver._speco_get_drafter_scheduler = lambda: pytest.fail(
        "disabled retry queried scheduler"
    )
    assert driver._speco_service_rollout_idle_events() == {}


@pytest.mark.parametrize(
    "guard",
    [
        "no_debt",
        "final_step",
        "no_state",
        "no_group",
        "cold",
        "cold_optimizer",
        "reclaim",
        "throttled",
    ],
)
def test_quiet_retry_skips_config_factory_without_ready_writer(monkeypatch, guard):
    scheduler, _, _ = _hot_writer(monkeypatch)
    if guard == "no_debt":
        scheduler._training_quota_debt_steps = 0
    elif guard == "final_step":
        scheduler._training_quota_debt_steps = 1
    elif guard == "no_state":
        scheduler._idle_worker_writer_state_version = None
    elif guard == "no_group":
        scheduler._idle_worker_writer_group = None
    elif guard == "cold":
        scheduler._idle_worker_hot_prewarmed_groups.clear()
    elif guard == "cold_optimizer":
        scheduler._idle_optimizer_initialized_groups.clear()
    elif guard == "reclaim":
        scheduler._idle_worker_prebatch_reclaim_streak = 1
    elif guard == "throttled":
        scheduler._idle_readmission_last_check_ts = 99.9
    assert not scheduler.should_retry_idle_admission(
        lambda: pytest.fail("impossible quiet retry parsed config"), now=100.0
    )


def test_quiet_retry_resolves_lazy_config_only_at_retry_interval(monkeypatch):
    scheduler, config, _ = _hot_writer(monkeypatch)
    calls = []

    def resolve():
        calls.append(True)
        return config

    assert scheduler.should_retry_idle_admission(resolve, now=100.0)
    assert not scheduler.should_retry_idle_admission(resolve, now=100.1)
    assert scheduler.should_retry_idle_admission(resolve, now=100.3)
    assert len(calls) == 2


def test_lazy_config_time_is_charged_to_live_window(monkeypatch):
    scheduler, config, _ = _hot_writer(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: clock[0]
    )

    def resolve():
        clock[0] = 110.0
        return config

    # All recorded tails have expired by the time configuration resolves.
    assert not scheduler.should_retry_idle_admission(resolve)
    assert scheduler._idle_readmission_last_check_ts == 110.0


def test_log8_step11_short_live_tail_does_not_miss_readmission(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    config = replace(config, idle_worker_initial_batch_estimate_sec=0.2941769229946658)
    # Only prior completed writer tails: never use step11's future 2.025s tail.
    tails = [
        0.13772559,
        2.96871924,
        0.09742641,
        4.73100591,
        0.11418986,
        0.12435579,
        0.12220764,
        0.12342143,
        0.11599874,
        0.63566518,
    ]
    scheduler._replica_idle_window_samples_sec[0] = deque(tails, maxlen=32)
    scheduler._replica_idle_windows_by_source[0]["generation_completed"] = deque(
        tails, maxlen=32
    )
    scheduler._replica_idle_started_at[0] = 100.0
    for worker in group:
        scheduler._idle_workers[worker].event_ts = 100.0
    scheduler._idle_readmission_last_check_ts = 99.95
    # At 50ms the forecast has not elapsed. At 150ms the same historical
    # safety gate already proves a 0.485s residual, enough for a bounded step.
    assert not scheduler.should_retry_idle_admission(config, now=100.05)
    assert scheduler.should_retry_idle_admission(config, now=100.15)
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: 100.15
    )
    scheduler._training_quota_data_version = 8
    context = replace(
        _context(),
        data_status=replace(_status("0", batches=20), data_version=8, target_version=8),
    )
    plan = scheduler.prepare_idle_worker_training_plan(context, config)
    assert plan.launch and plan.max_batches == 1
    assert plan.idle_window_source == "historical_survival"
    assert plan.deadline_ts <= 100.63566518
    assert not scheduler.should_retry_idle_admission(config, now=100.20)


@pytest.mark.parametrize("blocked", ["busy", "not_released", "deadline", "boundary"])
def test_quiet_retry_does_not_parse_config_outside_live_idle(monkeypatch, blocked):
    scheduler, _, group = _hot_writer(monkeypatch)
    state = scheduler._idle_workers[group[0]]
    if blocked == "busy":
        state.status = "generating"
    elif blocked == "not_released":
        state.memory_released = False
    elif blocked == "deadline":
        state.must_be_ready_at = 102.0
    else:
        state.confirmed_at_generation_boundary = True
    assert not scheduler.should_retry_idle_admission(
        lambda: pytest.fail("no usable live idle; configuration must stay lazy"),
        now=100.0,
    )


@pytest.mark.parametrize("has_event", [False, True])
def test_no_quota_quiet_gate_preserves_event_and_completion_servicing(
    monkeypatch, has_event
):
    scheduler, _, _ = _hot_writer(monkeypatch)
    scheduler._training_quota_debt_steps = 0
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_service_rollout_idle_events"},
    )
    driver = cls()
    serviced = []
    driver._speco_rollout_idle_generation_open = True
    driver._speco_historical_window_readmission_enabled = True
    driver._speco_bubble_training_lock = nullcontext

    def drain():
        serviced.append("events")
        return {"bubble/events": 1} if has_event else {}

    driver._speco_drain_rollout_idle_events = drain
    driver._speco_release_completed_rollout_training_lease = (
        lambda: serviced.append("completion") or {}
    )
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_drafter_schedule_config = lambda: pytest.fail(
        "empty quota parsed config"
    )
    driver._speco_try_launch_rollout_idle_training = (
        lambda: serviced.append("launch") or {}
    )
    driver._speco_record_rollout_idle_metrics = lambda metrics: None
    driver._speco_service_rollout_idle_events()
    assert serviced == ["events", "completion"] + (["launch"] if has_event else [])
