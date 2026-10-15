# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verl_speco.trainer.scheduler import (
    CallbackDrafterWorkerExecutor,
    DrafterExecutionStrategy,
    IdleWindowConfidence,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    TrainingDataStatus,
    TrainingOutcome,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.lifecycle import BeforeActorUpdateContext

try:
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer
except ModuleNotFoundError:
    SpecoRayPPOTrainer = None


class _FakeGenerationOutput:
    def __init__(self, samples=None):
        self.non_tensor_batch = {"drafter_sample": samples or []}
        self.meta_info = {"metrics": {}}


def _status(worker_id: str, *, batches: int = 5) -> TrainingDataStatus:
    return TrainingDataStatus(
        current_step=10,
        current_step_samples=batches,
        buffer_samples=batches,
        trainable_samples=batches,
        trainable_batches=batches,
        batch_size_per_gpu=1,
        partial_batch_available=False,
        oldest_sample_step=10,
        newest_sample_step=10,
        same_step_data_required=False,
        target_version=10,
        data_version=10,
        buffer_version=1,
        worker_id=worker_id,
        worker_incarnation=f"worker-{worker_id}",
    )


def _context() -> DrafterScheduleContext:
    return DrafterScheduleContext(
        global_step=10,
        training_mode="online",
        collected_samples_this_step=2,
        oldlogprob_collection_requested=False,
    )


def test_first_full_collective_quota_bootstrap_does_not_prepare_new_sync_head():
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0", "1"),)
    scheduler._metadata_full_collective_idle_groups = (("0", "1"),)

    def inspect_data(*args):
        return [
            {
                **replace(
                    _status(worker),
                    current_step=2,
                    oldest_sample_step=2,
                    newest_sample_step=2,
                    target_version=2,
                    data_version=2,
                ).__dict__,
                "available": True,
            }
            for worker in ("0", "1")
        ]

    scheduler._worker_executor = replace(
        scheduler._worker_executor,
        prepare=lambda plan: pytest.fail(
            "Full-collective Bubble bootstrap must reuse the already cached head"
        ),
        inspect_data=inspect_data,
    )
    config = replace(
        _idle_config(),
        training_interval_steps=2,
        train_batches_per_trigger=20,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_allow_critical_path_fallback=True,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), target_version=2, data_version=2),
        global_step=2,
        config=config,
    )
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=2),
            config=config,
            allow_quota_topup=True,
        )
    )
    plan = event.training_plan
    assert plan.launch
    assert plan.reason == "quota_forced_completion_ready"
    assert plan.execution_strategy is DrafterExecutionStrategy.SYNC
    assert plan.source_global_step == plan.required_target_version == 2
    assert plan.max_batches == scheduler._training_quota_debt_steps == 20


@pytest.mark.parametrize(
    "enabled,boundary,cost,spike,expected",
    [
        (False, True, 0, False, False),
        (True, False, 0, False, False),
        (True, True, 0, True, False),
        (True, True, 100, False, False),
        (True, True, 1, False, True),
    ],
)
def test_writer_switch_requires_persistent_gain_ack_boundary_and_cost_margin(
    enabled, boundary, cost, spike, expected
):
    scheduler = DrafterScheduler()
    source, target = ("0", "1"), ("2", "3")
    scheduler._metadata_idle_training_groups = (source, target)
    scheduler._idle_worker_writer_group = source
    scheduler._writer_boundary_migration_allowed = boundary
    scheduler._latest_published_drafter_version = 10
    scheduler._rollout_worker_drafter_versions = {
        worker: 10 for worker in source + target
    }
    scheduler._writer_idle_history.extend(
        [
            {source: 0.1, target: 5.0 if not spike or index == 3 else 0.1}
            for index in range(4)
        ]
    )
    config = replace(_idle_config(), idle_worker_writer_migration_enable=enabled)
    candidate, metrics = scheduler.writer_switch_decision(config, cost_sec=cost)
    assert (candidate == target) is expected
    assert metrics["bubble/writer_switch_missed_idle_s"] > 0
    scheduler._rollout_worker_drafter_versions["3"] = 9
    assert scheduler.writer_switch_decision(config, cost_sec=0)[0] is None


@pytest.mark.parametrize(
    "now,writer", [(100.0, ("0",)), (100.4, ("0",)), (102.0, ("0",)), (100.0, ("1",))]
)
def test_idle_continuation_never_renews_deadline_or_switches_writer(
    monkeypatch, now, writer
):
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    scheduler._idle_optimizer_initialized_groups.add(("0",))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0",))
    scheduler._idle_worker_partial_tail_samples_sec.append(0.3)
    scheduler._idle_worker_final_tail_samples_sec.append(0.3)
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: now
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=120.0,
            event_ts=99.0,
        )
    )
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=False,
        idle_worker_keep_hot_between_plans=True,
        idle_worker_initial_batch_estimate_sec=0.1,
        idle_worker_deadline_guard_sec=0.05,
    )
    context = replace(_context(), data_status=_status("0", batches=10))
    previous = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=2,
        publish_after_success=False,
        deadline_ts=101.2,
        target_worker_ids=writer,
    )
    plan = scheduler.prepare_idle_worker_training_plan(
        context, config, continuation_plan=previous
    )
    if now >= 101.2 or writer != ("0",):
        assert not plan.launch
    else:
        assert plan.launch
        assert plan.deadline_ts <= previous.deadline_ts
        assert plan.target_worker_ids == previous.target_worker_ids


def _idle_config() -> DrafterScheduleConfig:
    return DrafterScheduleConfig(
        training_interval_steps=5,
        train_batches_per_trigger=10,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_training_groups=(("0", "1"),),
        training_quota_trigger_mode="interval",
        training_quota_allow_critical_path_fallback=True,
        idle_worker_min_idle_window_sec=0.25,
        idle_worker_initial_batch_estimate_sec=0.9,
        idle_worker_deadline_guard_sec=0.2,
    )


def _auto_idle_config(group_size: int) -> DrafterScheduleConfig:
    return replace(
        _idle_config(),
        idle_worker_training_groups=(),
        idle_worker_group_mode="auto",
        idle_worker_group_size=group_size,
    )


def _scheduler_with_statuses(
    worker_ids: tuple[str, ...],
    *,
    empty_worker_ids: tuple[str, ...] = (),
) -> DrafterScheduler:
    def inspect_data(
        sample_last_n_steps: int,
        require_full_batch: bool,
        requested_worker_ids: tuple[str, ...] | None = None,
        target_version: int | None = None,
    ):
        selected = requested_worker_ids or worker_ids
        return [
            {
                **replace(
                    _status(
                        worker_id,
                        batches=0 if worker_id in empty_worker_ids else 5,
                    ),
                    data_version=(target_version or _status(worker_id).data_version),
                    target_version=(
                        target_version or _status(worker_id).target_version
                    ),
                ).__dict__,
                "available": True,
                "rank": int(worker_id) if str(worker_id).isdigit() else index,
            }
            for index, worker_id in enumerate(selected)
        ]

    return DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=inspect_data,
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )


def test_idle_worker_plan_requires_complete_training_group() -> None:
    scheduler = DrafterScheduler()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=103.0,
            event_ts=100.0,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), _idle_config())

    assert not plan.launch
    assert plan.reason == "incomplete_training_group"
    assert plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert plan.metrics()["bubble/skipped_incomplete_group"] == 1


def test_speculative_idle_window_waits_for_generation_confirmation() -> None:
    now = time.time()
    config = replace(
        _idle_config(),
        idle_worker_training_groups=(("0",),),
        idle_worker_speculative_window_multiplier=1.5,
    )
    confirmed = DrafterScheduler()
    confirmed.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 1.3,
            event_ts=now,
        )
    )
    speculative = DrafterScheduler()
    speculative.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.SPECULATIVE,
            must_be_ready_at=now + 1.3,
            event_ts=now,
        )
    )

    assert confirmed.select_idle_training_resources(config, now=now).available
    speculative_resources = speculative.select_idle_training_resources(config, now=now)
    assert not speculative_resources.available
    assert speculative_resources.reason == "speculative_idle_unconfirmed"

    metrics = speculative.record_generation_completed(
        now + 0.1,
        confirm_speculative_idle=True,
        must_be_ready_at=now + 5.0,
    )
    assert metrics["bubble/speculative_idle_confirmed"] == 1
    confirmed_resources = speculative.select_idle_training_resources(
        config, now=now + 0.1
    )
    assert confirmed_resources.available
    assert confirmed_resources.idle_confidence is IdleWindowConfidence.CONFIRMED


def test_replica_local_plan_accumulates_to_full_collective_batch() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    config = replace(_idle_config(), training_interval_steps=1)
    context = DrafterScheduleContext(
        global_step=10,
        training_mode="online",
        collected_samples_this_step=6,
        oldlogprob_collection_requested=False,
        data_status=replace(_status("0", batches=6), trainable_valid_tokens=600),
    )
    resources = type(
        "Resources",
        (),
        {
            "training_group_id": "idle-group-0",
            "worker_ids": ("0", "1"),
            "minimum_idle_window_sec": 20.0,
            "idle_confidence": IdleWindowConfidence.CONFIRMED,
        },
    )()

    plan = scheduler.plan_training(context, config, resources=resources)

    assert plan.launch
    assert plan.gradient_accumulation_steps == 2
    assert plan.max_batches == config.train_batches_per_trigger
    assert plan.planned_optimizer_steps == config.train_batches_per_trigger
    assert plan.planned_valid_tokens == 2000


def test_partial_quota_idle_plan_retains_replay_and_defers_publish() -> None:
    scheduler = DrafterScheduler()
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    scheduler._training_quota_debt_steps = 20
    config = replace(
        _idle_config(),
        training_interval_steps=1,
        training_quota_enable=True,
        training_quota_target_steps=20,
        train_batches_per_trigger=10,
    )
    context = DrafterScheduleContext(
        global_step=5,
        training_mode="online",
        collected_samples_this_step=4,
        oldlogprob_collection_requested=False,
        data_status=replace(_status("0", batches=4), trainable_valid_tokens=400),
    )
    resources = type(
        "Resources",
        (),
        {
            "training_group_id": "idle-group-0",
            "worker_ids": ("0", "1"),
            "minimum_idle_window_sec": 10.0,
            "idle_confidence": IdleWindowConfidence.CONFIRMED,
        },
    )()

    plan = scheduler.plan_training(
        context, config, resources=resources, require_interval=False
    )

    assert plan.launch
    assert plan.max_batches == 10
    assert not plan.publish_after_success
    assert plan.retain_replay_session


def test_completed_quota_blocks_training_until_publish_ack() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_collection_step = 4
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
    )

    plan = scheduler.prepare_training_plan(
        replace(_context(), global_step=4),
        config,
    )

    assert not plan.launch
    assert plan.reason == "training_quota_publish_pending"


def test_final_quota_idle_plan_is_capped_and_publishes_in_bubble() -> None:
    scheduler = DrafterScheduler()
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    scheduler._training_quota_debt_steps = 3
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        train_batches_per_trigger=10,
    )
    context = DrafterScheduleContext(
        global_step=6,
        training_mode="online",
        collected_samples_this_step=4,
        oldlogprob_collection_requested=False,
        data_status=replace(_status("0", batches=4), trainable_valid_tokens=400),
    )
    resources = type(
        "Resources",
        (),
        {
            "training_group_id": "idle-group-0",
            "worker_ids": ("0", "1"),
            "minimum_idle_window_sec": 10.0,
            "idle_confidence": IdleWindowConfidence.CONFIRMED,
        },
    )()

    plan = scheduler.plan_training(
        context, config, resources=resources, require_interval=False
    )

    assert plan.launch
    assert plan.max_batches == 3
    assert plan.publish_after_success
    assert not plan.retain_replay_session


def test_auto_idle_worker_groups_do_not_train_half_collective_group() -> None:
    scheduler = DrafterScheduler()
    deadline_ts = time.time() + 2.8
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="1",
            replica_rank=1,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), _auto_idle_config(2))

    assert not plan.launch
    assert plan.reason == "incomplete_training_group"


def test_auto_idle_worker_groups_select_complete_group_from_four_workers() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1", "2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=int(worker_id),
            )
        )
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )

    plan = scheduler.prepare_training_plan(_context(), _auto_idle_config(2))

    assert plan.launch
    assert plan.target_worker_ids == ("2", "3")
    assert plan.training_group_id == "idle-group-1"


def test_explicit_idle_worker_groups_override_auto_group_size() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1", "2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=int(worker_id),
            )
        )
    for worker_id in ("0", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(
            _auto_idle_config(2),
            idle_worker_training_groups=(("0", "3"),),
        ),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("0", "3")
    assert plan.training_group_id == "idle-group-0"


def test_idle_worker_auto_group_config_from_nested_mapping() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "scheduler": {
                "execution": {"strategy": "rollout_idle_worker"},
                "idle_worker": {"group_mode": "auto", "group_size": 2},
            }
        }
    )

    assert config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert config.idle_worker_group_mode == "auto"
    assert config.idle_worker_group_size == 2
    assert config.idle_worker_training_groups == ()


def test_training_quota_config_reuses_training_step_by_default() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "step": 10,
            "scheduler": {
                "execution": {"strategy": "rollout_idle_worker"},
                "training_quota": {
                    "enable": True,
                    "target_steps": None,
                    "max_debt_age_steps": 3,
                    "max_completion_lag_steps": 5,
                    "max_accumulated_debt": 20,
                    "max_sync_topup_steps": 2,
                },
            },
        }
    )

    assert config.training_quota_enable
    assert config.training_quota_target_steps is None
    assert config.train_batches_per_trigger == 10
    assert config.training_quota_max_debt_age_steps == 3
    assert config.training_quota_max_completion_lag_steps == 5
    assert config.training_quota_max_accumulated_debt == 20
    assert config.training_quota_max_sync_topup_steps == 2


def test_training_quota_defaults_prioritize_bubble_completion() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "step": 10,
            "scheduler": {
                "execution": {"strategy": "rollout_idle_worker"},
                "training_quota": {"enable": True},
            },
        }
    )

    assert config.training_quota_target_steps == 20
    assert config.training_quota_max_debt_age_steps == 1
    assert config.training_quota_max_completion_lag_steps == 3
    assert config.training_quota_max_accumulated_debt == 20
    assert config.training_quota_max_sync_topup_steps == 2


def test_training_quota_topup_leaves_final_step_for_bubble_publish() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    config = replace(
        _idle_config(),
        training_interval_steps=4,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_debt_age_steps=0,
        training_quota_max_accumulated_debt=20,
        training_quota_max_sync_topup_steps=20,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=4),
        global_step=5,
        config=config,
    )
    scheduler._training_quota_debt_steps = 6

    before_boundary = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=7),
            config=config,
        )
    )
    boundary = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=8),
            config=config,
        )
    )

    assert before_boundary.training_plan is not None
    assert not before_boundary.training_plan.launch
    assert boundary.training_plan is not None
    assert boundary.training_plan.reason == "quota_topup_training_ready"
    assert boundary.training_plan.max_batches == 5
    assert not boundary.training_plan.publish_after_success
    assert boundary.training_plan.retain_replay_session


def test_hot_writer_keeps_soft_debt_for_idle_but_hard_deadline_still_completes() -> (
    None
):
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_debt_age_steps=1,
        training_quota_max_completion_lag_steps=3,
        training_quota_max_sync_topup_steps=20,
        idle_worker_keep_hot_between_plans=True,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=4),
        global_step=4,
        config=config,
    )
    scheduler._training_quota_debt_steps = 8
    scheduler._replica_idle_worker_groups = {0: ("0", "1")}
    scheduler._replica_idle_window_samples_sec[0] = deque([5.0])
    soft = scheduler._plan_training_quota_topup(
        replace(_context(), global_step=6), config
    )
    assert soft is None and scheduler._training_quota_debt_steps == 8
    hard = scheduler._plan_training_quota_topup(
        replace(_context(), global_step=8), config
    )
    assert hard is not None and hard.max_batches == 8
    assert hard.reason == "quota_forced_completion_ready" and hard.publish_after_success


def test_regular_idle_launch_preempts_training_quota_topup() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    config = replace(
        _idle_config(),
        training_interval_steps=4,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_debt_age_steps=0,
        training_quota_max_accumulated_debt=20,
        training_quota_max_sync_topup_steps=20,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=4),
        global_step=5,
        config=config,
    )
    scheduler._training_quota_debt_steps = 6
    scheduler.prepare_training_plan = lambda *args, **kwargs: TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=8,
        max_batches=10,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
    )

    boundary = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=8),
            config=config,
        )
    )

    assert boundary.training_plan is not None
    assert boundary.training_plan.reason == "training_ready"
    assert boundary.training_plan.max_batches == 10


def test_training_quota_bootstrap_waits_for_safe_point_then_uses_full_collective() -> (
    None
):
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._idle_worker_writer_group = ("0", "1")
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_debt_age_steps=3,
        training_quota_max_completion_lag_steps=10,
        training_quota_max_accumulated_debt=20,
        training_quota_max_sync_topup_steps=2,
    )
    first_context = replace(_context(), global_step=10)
    scheduler._register_training_quota_cycle(
        data_status=_status("0"), global_step=10, config=config
    )

    first_event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=first_context,
            config=config,
            allow_quota_topup=False,
        )
    )

    assert first_event.training_plan is not None
    assert not first_event.training_plan.launch
    assert scheduler._training_quota_debt_steps == 10

    due_event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(first_context, global_step=13),
            config=config,
            allow_quota_topup=False,
        )
    )
    plan = due_event.training_plan

    assert plan is not None
    assert not plan.launch
    assert scheduler._training_quota_debt_steps == 10
    assert due_event.metrics["bubble/training_quota_topup_requested"] == 0

    forced_event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(first_context, global_step=20),
            config=config,
        )
    )
    forced_plan = forced_event.training_plan

    assert forced_plan is not None
    assert forced_plan.launch
    assert forced_plan.reason == "quota_forced_completion_ready"
    assert forced_plan.execution_strategy is DrafterExecutionStrategy.SYNC
    assert forced_plan.target_worker_ids == ()
    assert set((forced_plan.worker_snapshots or {}).keys()) == {
        "0",
        "1",
        "2",
        "3",
    }
    assert forced_plan.max_batches == 10
    assert forced_plan.deadline_ts is None
    assert forced_plan.publish_after_success
    assert forced_plan.hot_bootstrap_worker_ids == ("0", "1", "2", "3")
    assert forced_event.metrics["bubble/training_quota_force_complete"] == 1

    scheduler._record_training_outcome(
        forced_plan,
        TrainingOutcome(
            trained=True,
            successful_steps=10,
            worker_results=[],
            raw_results=[],
            elapsed_sec=1.0,
            reason="completed",
            metrics={},
        ),
    )
    assert scheduler._training_quota_debt_steps == 0
    assert scheduler.idle_writer_group() is None
    assert scheduler._idle_worker_writer_state_version is None
    assert ("0", "1", "2", "3") in scheduler._idle_worker_hot_prewarmed_groups


def test_full_collective_bootstrap_does_not_pin_next_idle_writer() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._record_training_outcome(
        TrainingPlan(
            launch=True,
            reason="quota_forced_completion_ready",
            interval_matched=True,
            execution_strategy=DrafterExecutionStrategy.SYNC,
            source_global_step=4,
            max_batches=10,
            publish_after_success=True,
            hot_bootstrap_worker_ids=("0", "1"),
        ),
        TrainingOutcome(
            trained=True,
            successful_steps=10,
            worker_results=[],
            raw_results=[],
            elapsed_sec=1.0,
            reason="completed",
            metrics={},
        ),
    )
    now = time.time()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="1",
            replica_rank=1,
            memory_released=True,
            event_ts=now,
        )
    )
    scheduler._replica_idle_window_samples_sec[1] = deque((4.0,), maxlen=32)

    plan = scheduler.prepare_idle_worker_training_plan(
        replace(_context(), data_status=_status("1", batches=4)),
        replace(_auto_idle_config(1), training_interval_steps=1),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("1",)
    assert scheduler.idle_writer_group() == ("1",)


def test_full_collective_hot_bootstrap_prefers_idle_capacity() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._global_publish_leader_worker_id = "0"
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._replica_idle_window_samples_sec = {
        0: deque((0.05,), maxlen=32),
        1: deque((2.5,), maxlen=32),
    }

    assert scheduler._best_idle_bootstrap_group() == ("1",)


def test_initial_writer_does_not_follow_one_stale_long_tail():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._replica_idle_worker_groups = {0: ("0", "1"), 1: ("2", "3")}
    # GPU log33: the second replica's first-round outlier must not outweigh
    # the more recent observation when choosing the initial optimizer owner.
    scheduler._replica_idle_window_samples_sec = {
        0: deque([0.04649, 4.44544]),
        1: deque([5.10592, 0.03724]),
    }
    assert scheduler._best_idle_bootstrap_group() == ("0", "1")
    # Election scoring does not alter the safety/admission window estimator.
    assert scheduler._effective_historical_idle_window_sec(
        replica_rank=1
    ) == pytest.approx(5.10592)
    scheduler._idle_worker_writer_group = ("2", "3")
    scheduler._idle_worker_writer_state_version = 2
    scheduler._best_idle_bootstrap_group()
    assert scheduler.idle_writer_group() == ("2", "3")


def test_initial_writer_discounts_a_single_unconfirmed_capacity_sample():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._replica_idle_window_samples_sec = {
        0: deque([4.0]),
        1: deque([2.0, 2.0, 2.0]),
    }
    assert scheduler._best_idle_bootstrap_group() == ("1",)


def test_initial_writer_prefers_completed_rollout_history_over_request_gaps():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._replica_idle_window_samples_sec = {
        0: deque([99.0, 99.0, 99.0]),
        1: deque([0.01, 0.01, 0.01]),
    }
    scheduler._replica_idle_windows_by_source = {
        0: {"generation_completed": deque([0.05, 0.05, 0.05])},
        1: {"generation_completed": deque([3.0, 3.0, 3.0])},
    }
    assert scheduler._best_idle_bootstrap_group() == ("1",)


def test_collection_writer_pins_data_without_initializing_optimizer():
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._replica_idle_worker_groups = {0: ("0", "1"), 1: ("2", "3")}
    scheduler._replica_idle_window_samples_sec = {
        0: deque([0.4, 0.4, 0.4]),
        1: deque([3.0, 3.0, 3.0]),
    }
    assert scheduler.prepare_collection_writer_group() == ("2", "3")
    assert scheduler._idle_worker_writer_state_version is None
    assert not scheduler._idle_optimizer_initialized_groups
    assert not scheduler._idle_worker_hot_prewarmed_groups
    now = time.time()
    # If bootstrap is waiting for its head, an idle group with no copy of the
    # collected pool must not steal ownership before the first optimizer step.
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=0,
                memory_released=True,
                must_be_ready_at=now + 20.0,
            )
        )
    resources = scheduler.select_idle_training_resources(_auto_idle_config(2), now=now)
    assert not resources.available
    assert scheduler.idle_writer_group() == ("2", "3")
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                must_be_ready_at=now + 10.0,
            )
        )
    resources = scheduler.select_idle_training_resources(_auto_idle_config(2), now=now)
    assert resources.available
    assert resources.worker_ids == ("2", "3")


def test_collection_writer_does_not_bypass_blocked_migration():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_migration_blocked = True
    assert scheduler.prepare_collection_writer_group() is None
    assert scheduler._idle_worker_writer_group == ("0",)
    assert scheduler._idle_worker_bootstrap_collection_group is None


@pytest.mark.parametrize("route_before_bootstrap", [False, True])
def test_first_optimizer_quota_starts_at_ready_safe_point_on_longer_group(
    route_before_bootstrap,
) -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._replica_idle_worker_groups = {0: ("0", "1"), 1: ("2", "3")}
    scheduler._replica_idle_window_samples_sec = {
        0: deque([0.4]),
        1: deque([2.3]),
    }
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_completion_lag_steps=3,
    )
    if route_before_bootstrap:
        assert scheduler.prepare_collection_writer_group() == ("2", "3")
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=2), global_step=2, config=config
    )
    assert scheduler.needs_idle_optimizer_bootstrap(config)
    plan = scheduler._plan_training_quota_topup(
        replace(_context(), global_step=2), config
    )
    assert plan is not None and plan.launch
    assert plan.target_worker_ids == ("2", "3")
    assert plan.max_batches == 20 and plan.publish_after_success
    assert plan.reason == "quota_forced_completion_ready"
    assert scheduler.idle_writer_group() == (
        ("2", "3") if route_before_bootstrap else None
    )  # Planning is not a state mutation.
    scheduler._record_training_outcome(
        plan,
        TrainingOutcome(
            trained=True,
            successful_steps=20,
            worker_results=[],
            raw_results=[],
            elapsed_sec=10,
            reason="trained",
            metrics={"bubble/training_residency_retained": 1},
        ),
    )
    assert scheduler.idle_writer_group() == ("2", "3")
    assert not scheduler.needs_idle_optimizer_bootstrap(config)
    # A longer alternative window cannot silently replace trained optimizer state.
    scheduler._replica_idle_window_samples_sec[0].append(20.0)
    assert scheduler.idle_writer_group() == ("2", "3")


def test_safe_point_head_preparation_breaks_cold_budget_feedback() -> None:
    scheduler = DrafterScheduler()
    group = ("0", "1")
    scheduler._idle_worker_hot_prewarmed_groups.add(group)
    scheduler._idle_worker_head_prepare_samples[group] = deque([3.04])
    scheduler._idle_worker_last_target_version[group] = 2
    scheduler._training_quota_data_version = 6
    config = _idle_config()
    assert scheduler._effective_idle_startup_reserve_sec(
        config, group
    ) == pytest.approx(3.04)
    scheduler.record_hot_target_head_prepared(group, 6, 0.5)
    assert scheduler._effective_idle_startup_reserve_sec(config, group) == 0
    scheduler._training_quota_data_version = 8
    assert scheduler._effective_idle_startup_reserve_sec(
        config, group
    ) == pytest.approx(0.5)


def test_full_collective_hot_bootstrap_uses_idle_capacity_without_leader_metadata() -> (
    None
):
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._replica_idle_window_samples_sec = {
        0: deque((0.05,), maxlen=32),
        1: deque((2.5,), maxlen=32),
    }

    assert scheduler._best_idle_bootstrap_group() == ("1",)


def test_topup_probe_does_not_elect_writer_before_fallback_is_due() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0",))
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_oldest_cycle_step = 2
    scheduler._training_quota_last_cycle_step = 2
    scheduler._training_quota_data_version = 2
    scheduler._training_quota_collection_step = 2
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_debt_age_steps=1,
        training_quota_max_completion_lag_steps=3,
    )

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=2),
            config=config,
            allow_quota_topup=False,
        )
    )

    assert event.training_plan is not None
    assert not event.training_plan.launch
    assert scheduler.idle_writer_group() is None
    assert scheduler.target_lm_head_sync_worker_ids() == ("0", "1")


def test_production_quota_never_falls_back_to_critical_path() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_oldest_cycle_step = 10
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_collection_step = 10
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_completion_lag_steps=0,
        training_quota_allow_critical_path_fallback=False,
    )

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=20),
            config=config,
        )
    )

    assert event.training_plan is not None
    assert not event.training_plan.launch
    assert scheduler._training_quota_debt_steps == 20
    assert event.metrics["bubble/training_quota_topup_requested"] == 0
    assert event.metrics["bubble/training_quota_critical_path_fallback_blocked"] == 1


def test_completed_quota_loss_regression_requests_next_full_refresh() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 1
    scheduler._training_quota_collection_step = 10
    scheduler._quality_loss_baseline = 1.0
    scheduler._quality_loss_increase_ratio = 0.10
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[{"train/avg_loss": 1.2}],
        elapsed_sec=1.0,
        reason="completed",
        metrics={},
    )

    scheduler._record_training_outcome(plan, outcome)
    scheduler.record_training_quota_publish_completed(global_step=10)

    assert scheduler._training_quota_debt_steps == 0
    assert scheduler._quality_loss_refresh_requested
    assert scheduler._quality_latest_loss == pytest.approx(1.2)


def test_zero_progress_full_collective_falls_back_to_writer_when_data_is_local() -> (
    None
):
    scheduler = _scheduler_with_statuses(
        ("0", "1", "2", "3"),
        empty_worker_ids=("2", "3"),
    )
    scheduler._idle_worker_writer_group = ("0", "1")
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_completion_lag_steps=0,
    )

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=10),
            config=config,
        )
    )
    plan = event.training_plan

    assert plan is not None
    assert plan.launch
    assert plan.reason == "quota_forced_completion_ready"
    assert plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert plan.target_worker_ids == ("0", "1")
    assert plan.max_batches == 10


@pytest.mark.parametrize("empty_workers", [("0", "1"), ("1",)])
def test_overdue_quota_without_matching_data_or_head_fails_explicitly(empty_workers):
    scheduler = _scheduler_with_statuses(("0", "1"), empty_worker_ids=empty_workers)
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    scheduler._training_quota_debt_steps = 16
    scheduler._training_quota_oldest_cycle_step = 4
    scheduler._training_quota_data_version = 4
    scheduler._training_quota_collection_step = 4
    config = replace(
        _idle_config(),
        training_interval_steps=2,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_completion_lag_steps=3,
    )
    # Before the hard freshness bound, idle training still gets its chance.
    assert (
        scheduler._plan_training_quota_topup(replace(_context(), global_step=6), config)
        is None
    )
    with pytest.raises(RuntimeError, match="quota_completion_blocked") as error:
        scheduler._plan_training_quota_topup(replace(_context(), global_step=7), config)
    assert "required_target_version=4" in str(error.value)
    assert "debt_steps=16" in str(error.value)
    assert "worker_snapshots=" in str(error.value)
    assert scheduler._training_quota_debt_steps == 16
    assert scheduler.training_quota_blocks_collection(8, config)
    # Intentional production isolation and in-flight training must not fail.
    assert (
        scheduler._plan_training_quota_topup(
            replace(_context(), global_step=8),
            replace(config, training_quota_allow_critical_path_fallback=False),
        )
        is None
    )
    assert (
        scheduler._plan_training_quota_topup(
            replace(_context(), global_step=8, pending_training_count=1), config
        )
        is None
    )
    scheduler.prepare_training_plan = lambda *args, **kwargs: TrainingPlan(
        launch=False,
        reason="no_idle_worker",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=8,
        max_batches=0,
        publish_after_success=False,
    )
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=8),
            config=config,
            drafter_frozen=True,
        )
    )
    assert not event.training_plan.launch


def test_training_quota_hard_deadline_finishes_and_publishes() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._training_quota_debt_steps = 6
    scheduler._training_quota_oldest_cycle_step = 8
    scheduler._training_quota_last_cycle_step = 8
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_collection_step = 10
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_debt_age_steps=1,
        training_quota_max_completion_lag_steps=3,
        # Disabling soft top-ups must not disable the hard freshness bound.
        training_quota_max_sync_topup_steps=0,
        publish_interval_steps=4,
    )
    # Even if a normal deadline-limited Bubble plan is launchable, the hard
    # freshness bound must replace it with an exact, blocking completion plan.
    scheduler.prepare_training_plan = lambda *args, **kwargs: TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=11,
        max_batches=6,
        deadline_ts=time.time() + 10.0,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
    )

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=replace(_context(), global_step=11),
            config=config,
        )
    )
    plan = event.training_plan

    assert plan is not None
    assert plan.launch
    assert plan.reason == "quota_forced_completion_ready"
    assert plan.max_batches == 6
    assert plan.publish_after_success
    assert not plan.retain_replay_session
    assert event.metrics["bubble/training_quota_force_complete"] == 1
    assert event.metrics["bubble/training_quota_force_complete_steps"] == 6

    scheduler._record_training_outcome(
        plan,
        TrainingOutcome(
            trained=True,
            successful_steps=6,
            worker_results=[],
            raw_results=[],
            elapsed_sec=1.0,
            reason="completed",
            metrics={},
        ),
    )
    publish = scheduler.plan_publish(
        global_step=11,
        drafter_trained=True,
        config=config,
        training_plan=plan,
    )

    assert scheduler._training_quota_debt_steps == 0
    assert scheduler._training_quota_data_version is None
    assert scheduler._training_quota_collection_step == 10
    assert publish.publish
    assert publish.reason == "training_quota_complete"

    scheduler.record_training_quota_publish_completed()

    assert scheduler._training_quota_collection_step is None


@pytest.mark.parametrize(
    "reason",
    ["training_ready", "quota_topup_training_ready", "quota_forced_completion_ready"],
)
def test_topup_is_not_reported_as_idle_training(reason) -> None:
    plan = TrainingPlan(
        launch=True,
        reason=reason,
        interval_matched=True,
        source_global_step=10,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        max_batches=2,
        publish_after_success=False,
    )
    runtime = DrafterRuntimeState()
    runtime.submit(plan, started_at=time.time())
    runtime.mark_running()
    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(
            raw_results=[{"trained": True, "successful_steps": 2}], elapsed_sec=1.0
        ),
        runtime_state=runtime,
        plan=plan,
    )
    idle = reason == "training_ready"
    assert outcome.metrics["drafter/idle_trained"] == int(idle)
    assert outcome.metrics["bubble/idle_optimizer_steps"] == (2 if idle else 0)
    assert outcome.metrics["bubble/critical_path_topup_optimizer_steps"] == (
        0 if idle else 2
    )


def test_writer_small_window_is_not_hidden_by_other_group_migration(capsys) -> None:
    scheduler = DrafterScheduler()
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 10
    config = replace(_idle_config(), idle_worker_training_groups=(("0",), ("1",)))
    for worker, window in [("0", 0.5), ("1", 30.0)]:
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker,
                replica_rank=int(worker),
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=100.0,
                must_be_ready_at=100.0 + window,
            )
        )
    resources = scheduler.select_idle_training_resources(config, now=100.0)
    assert not resources.available
    assert resources.reason == "window_too_small"
    assert resources.worker_ids == ("0",)
    assert resources.minimum_idle_window_sec == pytest.approx(0.5)
    assert "budget_deficit_s=" in capsys.readouterr().out


def test_local_topup_calibrates_overhead_and_separate_totals() -> None:
    scheduler = DrafterScheduler()
    config = _idle_config()
    idle = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        source_global_step=10,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        max_batches=1,
        publish_after_success=False,
        target_worker_ids=("0", "1"),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=5.0,
        reason="trained",
        metrics={
            "timing_s/drafter_worker_preflight": 3.0,
            "timing_s/drafter_worker_cleanup": 2.0,
            "timing_s/drafter_worker_memory_park": 1.0,
            "bubble/training_residency_retained": 0,
        },
    )
    scheduler._record_training_outcome(idle, outcome)
    topup = replace(idle, reason="quota_topup_training_ready")
    scheduler._record_training_outcome(topup, outcome)
    assert outcome.metrics["bubble/idle_optimizer_steps_total"] == 1
    assert outcome.metrics["bubble/critical_path_topup_optimizer_steps_total"] == 1
    assert scheduler._effective_idle_startup_reserve_sec(config, ("0", "1")) == 3.0
    assert scheduler._effective_idle_tail_reserve_sec(config) == 2.0


def test_quota_publish_preserves_optimizer_owner_across_cycles() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 10
    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_collection_step = 10
    now = time.time()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 1.0,
            event_ts=now,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="1",
            replica_rank=1,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 5.0,
            event_ts=now,
        )
    )
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        idle_worker_initial_batch_estimate_sec=0.1,
        idle_worker_deadline_guard_sec=0.0,
    )

    pinned = scheduler.select_idle_training_resources(config, now=now)

    assert pinned.available
    assert pinned.worker_ids == ("0",)

    scheduler.record_training_quota_publish_completed(global_step=10)

    assert scheduler.idle_writer_group() == ("0",)
    assert scheduler._idle_worker_writer_state_version == 10
    assert not scheduler._idle_worker_writer_migration_blocked
    assert scheduler._training_quota_collection_step is None

    next_group = scheduler.select_idle_training_resources(config, now=now)

    assert next_group.available
    assert next_group.worker_ids == ("0",)


def test_released_writer_does_not_switch_to_stale_drafter_group(caplog) -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 10
    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_collection_step = 10
    now = time.time()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 1.0,
            event_ts=now,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="1",
            replica_rank=1,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 5.0,
            event_ts=now,
        )
    )
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        idle_worker_initial_batch_estimate_sec=0.1,
        idle_worker_deadline_guard_sec=0.0,
    )

    with caplog.at_level(logging.INFO):
        scheduler.record_rollout_drafter_publish_completed(
            10,
            acknowledgements=[
                {"published": True, "published_version": 10, "worker_rank": 0}
            ],
        )
        scheduler.record_training_quota_publish_completed(global_step=10)

        next_group = scheduler.select_idle_training_resources(config, now=now)

    assert next_group.available
    assert next_group.worker_ids == ("0",)
    assert "rollout_drafter_versions_updated" in caplog.text
    assert "writer_state_migration_required" in caplog.text


def test_weight_ack_alone_does_not_allow_optimizer_owner_switch() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 10
    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_collection_step = 10
    now = time.time()
    for worker_id, replica_rank, ready_delta in (("0", 0, 1.0), ("1", 1, 5.0)):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                must_be_ready_at=now + ready_delta,
                event_ts=now,
            )
        )
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        idle_worker_initial_batch_estimate_sec=0.1,
        idle_worker_deadline_guard_sec=0.0,
    )

    scheduler.record_rollout_drafter_publish_completed(
        10,
        acknowledgements=[
            {"published": True, "published_version": 10, "worker_rank": 0},
            {"published": True, "published_version": 10, "worker_rank": 1},
        ],
    )
    scheduler.record_training_quota_publish_completed(global_step=10)

    next_group = scheduler.select_idle_training_resources(config, now=now)

    assert next_group.available
    assert next_group.worker_ids == ("0",)


def test_partial_forced_completion_does_not_repay_or_publish() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 6
    scheduler._training_quota_oldest_cycle_step = 8
    scheduler._training_quota_data_version = 10
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        publish_interval_steps=1,
    )
    plan = TrainingPlan(
        launch=True,
        reason="quota_forced_completion_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=11,
        max_batches=6,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        plan_id="forced-partial",
    )

    with pytest.raises(RuntimeError, match="requested=6 completed=5"):
        scheduler._record_training_outcome(
            plan,
            TrainingOutcome(
                trained=True,
                successful_steps=5,
                worker_results=[],
                raw_results=[],
                elapsed_sec=1.0,
                reason="reclaimed",
                metrics={},
            ),
        )

    publish = scheduler.plan_publish(
        global_step=11,
        drafter_trained=True,
        config=config,
        training_plan=plan,
    )
    assert scheduler._training_quota_debt_steps == 6
    assert not publish.publish
    assert publish.reason == "training_quota_incomplete"


def test_training_quota_does_not_add_debt_for_each_data_version() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_accumulated_debt=20,
    )

    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=11),
        global_step=11,
        config=config,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=12),
        global_step=12,
        config=config,
    )

    assert scheduler._training_quota_last_cycle_step == 15
    assert scheduler._training_quota_debt_steps == 10


def test_training_quota_pins_one_cycle_until_it_is_repaid() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        training_interval_steps=5,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_max_accumulated_debt=20,
    )

    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=1),
        global_step=1,
        config=config,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=6),
        global_step=6,
        config=config,
    )

    # A newer collection must not charge another quota against the older
    # cycle's replay data. It is registered after the pinned cycle completes.
    assert scheduler._training_quota_debt_steps == 20
    assert scheduler._training_quota_data_version == 1
    assert scheduler._training_quota_due(6, config)
    assert not scheduler._training_quota_force_completion_due(6, config)
    assert scheduler._training_quota_force_completion_due(8, config)


def test_training_quota_age_starts_after_interval_boundary() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        training_interval_steps=5,
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_debt_age_steps=3,
        training_quota_max_accumulated_debt=100,
    )

    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=1),
        global_step=1,
        config=config,
    )

    assert scheduler._training_quota_oldest_cycle_step == 5
    assert scheduler._training_quota_age_steps(4) == 0
    assert not scheduler._training_quota_due(4, config)
    assert not scheduler._training_quota_due(7, config)
    assert scheduler._training_quota_due(8, config)


def test_zero_completion_lag_still_waits_for_cycle_boundary() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        training_interval_steps=5,
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_completion_lag_steps=0,
    )
    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=1),
        global_step=1,
        config=config,
    )

    assert not scheduler._training_quota_force_completion_due(4, config)
    assert scheduler._training_quota_force_completion_due(5, config)


def test_training_quota_debt_cap_still_preserves_first_bubble_interval() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        training_interval_steps=5,
        training_quota_enable=True,
        training_quota_target_steps=10,
        training_quota_max_debt_age_steps=3,
        training_quota_max_accumulated_debt=10,
    )

    scheduler._register_training_quota_cycle(
        data_status=replace(_status("0"), data_version=1),
        global_step=1,
        config=config,
    )

    assert not scheduler._training_quota_due(4, config)
    assert not scheduler._training_quota_due(5, config)
    assert not scheduler._training_quota_due(6, config)
    assert scheduler._training_quota_due(8, config)


def test_training_quota_background_poll_never_launches_topup() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_oldest_cycle_step = 1
    scheduler._training_quota_last_cycle_step = 10
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_max_debt_age_steps=1,
        training_quota_max_sync_topup_steps=2,
    )

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=_context(),
            config=config,
            allow_quota_topup=False,
        )
    )

    assert event.training_plan is not None
    assert not event.training_plan.launch
    assert event.training_plan.reason != "quota_topup_training_ready"


def test_successful_bubble_steps_repay_training_quota_debt() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_oldest_cycle_step = 7
    plan = TrainingPlan(
        launch=True,
        reason="quota_topup_training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=2,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        plan_id="quota-plan",
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=2,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        reason="completed",
        metrics={},
    )

    scheduler._record_training_outcome(plan, outcome)

    assert scheduler._training_quota_debt_steps == 8
    assert scheduler._training_quota_oldest_cycle_step == 7


def test_partial_training_quota_topup_fails_closed_without_repaying_debt() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_oldest_cycle_step = 7
    plan = TrainingPlan(
        launch=True,
        reason="quota_topup_training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=10,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        plan_id="partial-quota-plan",
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        reason="completed",
        metrics={},
    )

    with pytest.raises(RuntimeError, match="requested=10 completed=1"):
        scheduler._record_training_outcome(plan, outcome)

    assert scheduler._training_quota_debt_steps == 10
    assert scheduler._training_quota_oldest_cycle_step == 7


@pytest.mark.parametrize(
    "reason",
    ["quota_topup_training_ready", "quota_forced_completion_ready"],
)
def test_training_quota_topup_uses_blocking_execution_strategy(reason: str) -> None:
    class _RecordingStrategy:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, plan, *, executor, runtime_state):
            self.calls += 1
            return ExecutionOutcome(raw_results=[], elapsed_sec=0.0)

    scheduler = DrafterScheduler(worker_executor=object())
    blocking = _RecordingStrategy()
    asynchronous = _RecordingStrategy()
    scheduler.sync_execution_strategy = blocking
    scheduler.rollout_idle_execution_strategy = asynchronous
    plan = TrainingPlan(
        launch=True,
        reason=reason,
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=2,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
    )

    scheduler.execute_training_plan(plan, runtime_state=DrafterRuntimeState())

    assert blocking.calls == 1
    assert asynchronous.calls == 0


def test_publish_waits_until_bubble_training_quota_is_complete() -> None:
    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 3
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        publish_interval_steps=1,
    )
    training_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=8,
        max_batches=10,
        publish_after_success=True,
    )

    blocked = scheduler.plan_publish(
        global_step=8,
        drafter_trained=True,
        config=config,
        training_plan=training_plan,
    )
    scheduler._training_quota_debt_steps = 0
    ready = scheduler.plan_publish(
        global_step=8,
        drafter_trained=True,
        config=config,
        training_plan=training_plan,
    )

    assert not blocked.publish
    assert blocked.reason == "training_quota_incomplete"
    assert ready.publish


def test_metadata_idle_worker_groups_wait_for_all_collective_replicas() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1, 2, 3],
            },
            {
                "in_drafter_train_group": True,
                "replica_rank": 1,
                "training_group_ranks": [2, 3],
                "full_collective_ranks": [0, 1, 2, 3],
            },
        ]
    )
    config = replace(_auto_idle_config(2), idle_worker_group_size=None)
    deadline_ts = time.time() + 2.8
    for replica_rank in (0, 1):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=str(replica_rank),
                replica_rank=replica_rank,
            )
        )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    half_plan = scheduler.prepare_training_plan(_context(), config)

    assert not half_plan.launch
    assert half_plan.reason == "incomplete_training_group"

    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="1",
            replica_rank=1,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.target_worker_ids == ("0", "1", "2", "3")
    assert plan.training_group_id == "idle-group-0"


def test_metadata_replica_groups_merge_multiple_ranks_for_same_replica() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0],
                "full_collective_ranks": [0, 1],
            },
            {
                "rank": 1,
                "worker_id": "1",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )
    deadline_ts = time.time() + 2.8
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(_auto_idle_config(2), idle_worker_group_size=None),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("0", "1")


def test_metadata_replica_worker_mapping_is_available_for_fallback() -> None:
    scheduler = DrafterScheduler()
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0],
                "full_collective_ranks": [0, 1],
            },
            {
                "rank": 1,
                "worker_id": "1",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )

    assert scheduler.rollout_idle_replica_ranks() == (0,)
    assert scheduler.rollout_idle_worker_ids_for_replica(0) == ("0", "1")
    assert scheduler.rollout_idle_worker_ids_for_replica(
        1, fallback_worker_id="worker-1"
    ) == ("worker-1",)


def test_replica_local_metadata_uses_training_group_not_sync_collective() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0],
                "full_collective_ranks": [0, 1],
                "sync_collective_ranks": [0, 1],
                "idle_collective_scope": "replica_local",
            },
            {
                "rank": 1,
                "worker_id": "1",
                "in_drafter_train_group": True,
                "replica_rank": 1,
                "training_group_ranks": [1],
                "full_collective_ranks": [0, 1],
                "sync_collective_ranks": [0, 1],
                "idle_collective_scope": "replica_local",
            },
        ]
    )
    deadline_ts = time.time() + 10.0

    metrics = scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )
    plan = scheduler.prepare_training_plan(
        _context(),
        replace(_auto_idle_config(2), idle_worker_group_size=None),
    )

    assert scheduler._idle_training_groups(_auto_idle_config(2)) == (("0",), ("1",))
    assert metrics["bubble/idle_training_groups"] == 1
    assert plan.launch
    assert plan.target_worker_ids == ("0",)


def test_auto_idle_worker_without_metadata_or_group_size_fails_closed() -> None:
    scheduler = DrafterScheduler()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )

    plan = scheduler.prepare_training_plan(
        _context(), replace(_auto_idle_config(1), idle_worker_group_size=None)
    )

    assert not plan.launch
    assert plan.reason == "missing_training_group_metadata"


def test_idle_worker_auto_budget_bootstraps_one_batch_without_manual_estimate() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    config = replace(
        _auto_idle_config(2),
        idle_worker_min_idle_window_sec=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.max_batches == config.train_batches_per_trigger
    assert plan.reason == "training_ready"
    assert plan.idle_batch_estimate_sec == pytest.approx(0.25)


def test_non_quota_idle_writer_without_deadline_bootstraps_one_batch() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=None,
            )
        )
    config = replace(
        _auto_idle_config(2),
        idle_worker_min_idle_window_sec=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.max_batches == 1
    assert plan.reason == "training_ready"


@pytest.mark.parametrize("bootstrapped", [False, True])
def test_quota_cold_idle_writer_without_deadline_requires_bootstrap(
    bootstrapped,
) -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_bootstrapped = bootstrapped
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    scheduler._training_quota_debt_steps = 10
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=None,
            )
        )
    config = replace(
        _auto_idle_config(2),
        training_quota_enable=True,
        idle_worker_min_idle_window_sec=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    # Legacy/model prewarm flags cannot prove optimizer initialization.
    assert not plan.launch
    assert plan.reason == "cold_writer_wait_for_optimizer_bootstrap"


def test_resource_lease_does_not_override_short_historical_request_gaps() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    for replica_rank, worker_id in enumerate(("0", "1")):
        for start in (100.0, 101.0, 102.0):
            scheduler.on_worker_event(
                RolloutWorkerEvent(
                    RolloutWorkerEventType.WORKER_IDLE,
                    worker_id=worker_id,
                    replica_rank=replica_rank,
                    memory_released=True,
                    idle_confidence=IdleWindowConfidence.CONFIRMED,
                    event_ts=start,
                )
            )
            scheduler.on_worker_event(
                RolloutWorkerEvent(
                    RolloutWorkerEventType.GENERATION_STARTED,
                    worker_id=worker_id,
                    replica_rank=replica_rank,
                    event_ts=start + 0.01,
                )
            )

    now = time.time()
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    plan = scheduler.prepare_training_plan(_context(), _auto_idle_config(2))

    assert scheduler._effective_historical_idle_window_sec() == pytest.approx(0.01)
    assert not plan.launch
    assert plan.reason == "window_too_small"
    assert plan.idle_window_sec == pytest.approx(0.01, abs=0.01)


def test_replica_local_history_selects_earlier_idle_group() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    scheduler._replica_idle_window_samples_sec = {
        0: deque([0.03], maxlen=32),
        1: deque([3.05], maxlen=32),
    }
    now = time.time()
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    plan = scheduler.prepare_idle_worker_training_plan(_context(), _auto_idle_config(1))

    assert plan.launch
    assert plan.target_worker_ids == ("1",)
    assert plan.idle_window_source == "historical_observed"
    assert plan.idle_window_sec == pytest.approx(3.05, abs=0.01)
    assert scheduler._effective_historical_idle_window_sec(
        worker_ids=("0",)
    ) == pytest.approx(0.03)
    assert scheduler._effective_historical_idle_window_sec(
        worker_ids=("1",)
    ) == pytest.approx(3.05)


def test_active_quota_does_not_override_tiny_historical_request_gap() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._replica_idle_window_samples_sec = {
        0: deque([0.03], maxlen=32),
        1: deque([0.03], maxlen=32),
    }
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_collection_step = 2
    now = time.time()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            event_ts=now,
        )
    )
    config = replace(
        _auto_idle_config(1),
        idle_worker_group_size=None,
        training_quota_enable=True,
        training_quota_target_steps=10,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert not plan.launch
    # The other configured group is also incomplete, so the aggregate skip
    # reason keeps the existing group-completeness precedence.
    assert plan.reason == "incomplete_training_group"


def test_active_quota_waits_for_runtime_deadline_before_first_optimizer_step() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("1",)
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_collection_step = 10
    now = time.time()
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(
            _idle_config(),
            idle_worker_training_groups=(),
            training_quota_enable=True,
            training_quota_target_steps=10,
        ),
    )

    assert not plan.launch
    assert plan.reason == "cold_writer_wait_for_optimizer_bootstrap"
    assert scheduler.idle_writer_group() == ("1",)


def test_active_quota_keeps_writer_after_first_optimizer_step() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._metadata_idle_training_groups = (("0",), ("1",))
    scheduler._idle_worker_writer_group = ("1",)
    scheduler._idle_worker_writer_state_version = 9
    scheduler._training_quota_debt_steps = 10
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_collection_step = 10
    now = time.time()
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(
            _idle_config(),
            idle_worker_training_groups=(),
            training_quota_enable=True,
            training_quota_target_steps=10,
        ),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("1",)
    assert scheduler.idle_writer_group() == ("1",)


def test_generation_boundary_confirms_already_confirmed_runtime_idle() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    config = replace(
        _auto_idle_config(1),
        idle_worker_initial_batch_estimate_sec=0.5,
        idle_worker_deadline_guard_sec=0.0,
        idle_worker_min_idle_window_sec=0.1,
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            event_ts=100.0,
        )
    )
    scheduler._replica_idle_window_samples_sec[0] = deque((2.0,), maxlen=32)

    metrics = scheduler.record_generation_completed(
        102.0,
        confirm_speculative_idle=True,
        must_be_ready_at=104.0,
    )
    resources = scheduler.select_idle_training_resources(config, now=103.0)

    assert metrics["bubble/speculative_idle_confirmed"] == 0
    assert metrics["bubble/generation_boundary_idle_confirmed"] == 1
    assert resources.available
    assert resources.worker_ids == ("0",)
    assert resources.idle_window_source == "runtime_deadline"
    assert resources.minimum_idle_window_sec == pytest.approx(1.0)


def test_recent_real_tails_bound_stale_replica_history_without_borrowing_peer_idle():
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._replica_idle_worker_groups = {0: ("0", "1"), 1: ("2", "3")}
    tails = {
        0: [0.0531, 4.8212, 12.1551, 0.0517, 0.0791, 0.0536],
        1: [2.2769, 0.0506, 0.0379, 5.0655, 6.2654, 7.2700],
    }
    for rank, values in tails.items():
        scheduler._replica_idle_window_samples_sec[rank] = deque(values, maxlen=32)
        scheduler._replica_idle_windows_by_source[rank] = {
            "generation_completed": deque(values, maxlen=32)
        }
    writer = scheduler._effective_historical_idle_window_sec(worker_ids=("0", "1"))
    peer = scheduler._effective_historical_idle_window_sec(worker_ids=("2", "3"))
    assert writer == pytest.approx(0.0536)
    assert peer > 5.0
    for value in (6.0, 7.0, 8.0):
        scheduler._replica_idle_window_samples_sec[0].append(value)
        scheduler._replica_idle_windows_by_source[0]["generation_completed"].append(
            value
        )
    assert scheduler._effective_historical_idle_window_sec(replica_rank=0) >= 6.0


def test_rollout_tail_complete_requires_every_known_rank_to_release_serving():
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._replica_idle_worker_groups = {0: ("0",), 1: ("1",)}
    assert not scheduler.rollout_generation_tail_complete()
    for rank in (0, 1):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=str(rank),
                replica_rank=rank,
                memory_released=True,
                event_ts=100.0,
            )
        )
        assert scheduler.rollout_generation_tail_complete() == (rank == 1)
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="1",
            replica_rank=1,
            event_ts=101.0,
        )
    )
    assert not scheduler.rollout_generation_tail_complete()


def test_idle_worker_admission_uses_observed_replica_idle_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                event_ts=100.0,
            )
        )
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=replica_rank,
                event_ts=102.0,
            )
        )

    now = time.time()
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                must_be_ready_at=now + 30.0,
                event_ts=now,
            )
        )
    config = replace(
        _idle_config(),
        idle_worker_initial_batch_estimate_sec=0.5,
        idle_worker_deadline_guard_sec=0.1,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.idle_window_sec == pytest.approx(2.0, abs=0.1)
    assert plan.max_batches == plan.idle_window_batches == 3


def test_generation_completion_records_real_idle_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                must_be_ready_at=130.0,
                event_ts=100.0 + replica_rank,
            )
        )

    metrics = scheduler.record_generation_completed(event_ts=103.0)

    assert metrics["bubble/observed_idle_windows"] == 2
    assert metrics["bubble/observed_idle_window_min_s"] == pytest.approx(2.0)
    assert metrics["bubble/observed_idle_window_max_s"] == pytest.approx(3.0)
    assert metrics["bubble/observed_idle_window_mean_s"] == pytest.approx(2.5)
    assert metrics["bubble/replica_0_observed_idle_window_s"] == pytest.approx(3.0)
    assert metrics["bubble/replica_1_observed_idle_window_s"] == pytest.approx(2.0)
    assert metrics["bubble/replica_0_historical_idle_window_s"] == pytest.approx(3.0)
    assert metrics["bubble/replica_1_historical_idle_window_s"] == pytest.approx(2.0)
    assert len(scheduler._replica_idle_window_samples_sec) == 2

    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=replica_rank,
                event_ts=150.0,
            )
        )

    assert len(scheduler._replica_idle_window_samples_sec) == 2


def test_generation_started_closed_window_is_exposed_in_idle_metrics() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._replica_idle_worker_groups = {0: ("0",)}
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            event_ts=100.0,
        )
    )

    metrics = scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
            event_ts=101.25,
        )
    )

    assert metrics["bubble/replica_0_observed_idle_window_s"] == pytest.approx(1.25)
    assert metrics["bubble/replica_0_historical_idle_window_s"] == pytest.approx(1.25)
    assert metrics["bubble/replica_0_idle_window_samples"] == 1
    assert metrics["bubble/replica_0_idle_window_close_source"] == 1


def test_small_idle_window_does_not_launch_training() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 0.35
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    config = _idle_config()
    context = DrafterScheduleContext(
        global_step=4,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
    )

    plan = scheduler.prepare_training_plan(context, config)

    assert not plan.launch
    assert plan.reason == "window_too_small"
    assert plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER


def test_idle_guard_does_not_charge_publish_drain_and_park_twice() -> None:
    scheduler = DrafterScheduler()
    config = replace(_idle_config(), idle_worker_deadline_guard_sec=0.1)

    scheduler.record_reclaim_elapsed(0.7)

    scheduler.record_memory_park_elapsed(0.2)
    assert scheduler._effective_idle_deadline_guard_sec(config) == pytest.approx(0.1)
    assert scheduler._effective_idle_tail_reserve_sec(config) == pytest.approx(0.2)


def test_retained_snapshot_cannot_register_another_quota() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(), training_quota_enable=True, training_quota_target_steps=20
    )
    status = replace(_status("0"), data_version=2)
    scheduler._register_training_quota_cycle(
        data_status=status, global_step=2, config=config
    )
    scheduler._training_quota_debt_steps = 0
    scheduler.record_training_quota_publish_completed(global_step=5)
    scheduler._register_training_quota_cycle(
        data_status=status, global_step=6, config=config
    )
    assert scheduler._training_quota_debt_steps == 0
    scheduler._register_training_quota_cycle(
        data_status=replace(status, data_version=6), global_step=6, config=config
    )
    assert scheduler._training_quota_debt_steps == 20


def test_private_writer_topup_never_rejoins_stale_replica_groups() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_writer_state_version = 11
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_data_version = 10
    scheduler._training_quota_oldest_cycle_step = 12
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_allow_critical_path_fallback=True,
        training_quota_max_completion_lag_steps=3,
    )
    plan = scheduler._plan_training_quota_topup(
        replace(_context(), global_step=15), config
    )
    assert plan is not None and plan.launch
    assert plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert plan.target_worker_ids == ("0", "1")
    assert plan.gradient_accumulation_steps == 2 * config.gradient_accumulation_steps


def test_cpu_bootstrap_calibration_is_used_for_group_admission() -> None:
    scheduler = DrafterScheduler()
    scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda *args: [],
            prepare=lambda plan: {},
            activate=lambda: [
                {
                    "activated": True,
                    "reason": "activated",
                    "training_group_ranks": [0, 1],
                    "activation_elapsed_sec": 45.0,
                    "restore_elapsed_sec": 0.4,
                    "park_elapsed_sec": 0.3,
                    "snapshot_elapsed_sec": 2.0,
                }
            ],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )
    scheduler.activate_training_workers()
    assert scheduler._effective_idle_startup_reserve_sec(
        _idle_config(), ("0", "1")
    ) == pytest.approx(0.4)
    assert scheduler._effective_idle_tail_reserve_sec(_idle_config()) == pytest.approx(
        0.3
    )
    assert scheduler._effective_idle_tail_reserve_sec(
        _idle_config(), final=True
    ) == pytest.approx(2.3)


@pytest.mark.parametrize(
    "participants, nonparticipants", [(1, 1), (2, 2), (4, 4), (4, 12)]
)
def test_optimizer_warmup_does_not_poison_idle_batch_estimate(
    participants, nonparticipants
) -> None:
    scheduler = DrafterScheduler()
    plan = TrainingPlan(
        launch=True,
        reason="quota_topup_training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=4,
        max_batches=3,
        publish_after_success=False,
        gradient_accumulation_steps=2,
        target_worker_ids=tuple(str(rank) for rank in range(participants)),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=3,
        worker_results=[],
        elapsed_sec=9,
        reason="trained",
        metrics={"timing_s/drafter_worker_training_loop": 9.0},
        raw_results=[
            {
                "worker_id": str(rank),
                "successful_steps": 3,
                "optimizer_step_samples_sec": [8.0, 0.35, 0.4],
                "optimizer_warmup_steps": 1,
            }
            for rank in range(participants)
        ]
        + [
            {
                "worker_id": str(rank + participants),
                "successful_steps": 0,
                "optimizer_step_samples_sec": [],
                "reason": "not_in_training_group",
            }
            for rank in range(nonparticipants)
        ],
    )
    scheduler._record_training_outcome(plan, outcome)
    assert list(scheduler._idle_worker_batch_samples_sec) == pytest.approx([0.175, 0.2])
    assert (
        scheduler._effective_idle_batch_estimate_sec(
            replace(_idle_config(), idle_worker_initial_batch_estimate_sec=None)
        )
        <= 0.2
    )


def test_writer_migration_is_explicitly_opt_in() -> None:
    assert not DrafterScheduleConfig.from_mapping(
        {}
    ).idle_worker_writer_migration_enable
    config = DrafterScheduleConfig.from_mapping(
        {"scheduler": {"idle_worker": {"writer_migration_enable": True}}}
    )
    assert config.idle_worker_writer_migration_enable


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_disabled_writer_migration_does_not_inspect_or_transfer_states() -> None:
    trainer = _trainer_with_idle_config()
    trainer._speco_get_drafter_scheduler = lambda: pytest.fail(
        "disabled migration must do no work"
    )
    assert trainer._speco_migrate_idle_writer_if_useful(_idle_config()) == {}


def test_missing_group_does_not_invent_fourfold_optimizer_step_cost() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    scheduler._idle_worker_writer_group = ("0", "1")
    assert scheduler._idle_gradient_accumulation_steps(_idle_config(), ()) == 2


def test_worker_event_returns_idle_metrics() -> None:
    scheduler = DrafterScheduler()

    metrics = scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=time.time() + 5.0,
        )
    )

    assert metrics["bubble/idle_workers"] == 1


def test_idle_worker_prebatch_reclaim_splits_setup_and_tail_reserves() -> None:
    scheduler = DrafterScheduler()
    config = _idle_config()
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[],
        # Driver-side RPC wall time includes waiting for the next rollout to
        # reclaim.  It must not be treated as worker setup overhead.
        elapsed_sec=24.0,
        reason="submitted_async",
        metrics={
            "bubble/train_reclaimed_before_first_batch": 1,
            "timing_s/drafter_worker_preflight": 22.0,
            "timing_s/drafter_worker_preflight_to_stop": 1.5,
            "timing_s/drafter_worker_cleanup": 3.0,
        },
    )

    scheduler.record_idle_training_outcome(outcome)

    # A pre-batch reclaim is a failed admission, not a representative Bubble
    # training cost.  It must not poison future startup/tail reserves.
    assert scheduler._effective_idle_startup_reserve_sec(config) == pytest.approx(0.0)
    assert scheduler._effective_idle_tail_reserve_sec(config) == pytest.approx(0.0)

    final_outcome = replace(
        outcome,
        trained=True,
        successful_steps=1,
        metrics={
            "bubble/train_first_batch_started": 1,
            "timing_s/drafter_worker_cleanup": 1.0,
            "timing_s/drafter_publish_snapshot": 4.0,
        },
    )
    scheduler.record_idle_training_outcome(final_outcome, final=True)

    assert scheduler._effective_idle_tail_reserve_sec(config) == pytest.approx(0.0)
    assert scheduler._effective_idle_tail_reserve_sec(
        config, final=True
    ) == pytest.approx(5.0)


def test_keep_hot_tail_does_not_budget_disabled_offload() -> None:
    scheduler = DrafterScheduler()
    scheduler.record_memory_park_elapsed(6.0)
    scheduler._idle_worker_partial_tail_samples_sec.append(0.1)
    scheduler._idle_worker_final_tail_samples_sec.append(4.1)
    cold = _idle_config()
    hot = replace(cold, idle_worker_keep_hot_between_plans=True)
    assert scheduler._effective_idle_tail_reserve_sec(cold) == pytest.approx(6.1)
    assert scheduler._effective_idle_tail_reserve_sec(hot) == pytest.approx(0.1)
    assert scheduler._effective_idle_tail_reserve_sec(hot, final=True) == pytest.approx(
        4.1
    )


def test_nonpark_cleanup_is_calibrated_without_cross_rank_subtraction() -> None:
    scheduler = DrafterScheduler()
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=20.0,
        reason="trained",
        metrics={
            "timing_s/drafter_worker_cleanup": 8.0,
            "timing_s/drafter_worker_memory_park": 7.0,
            "timing_s/drafter_worker_cleanup_nonpark": 3.0,
            "timing_s/drafter_publish_snapshot": 12.0,
        },
    )
    scheduler.record_idle_training_outcome(outcome, final=False)
    # Snapshot is not part of a partial plan; park is added exactly once.
    assert scheduler._effective_idle_tail_reserve_sec(_idle_config()) == 10.0
    assert list(scheduler._idle_worker_partial_tail_samples_sec) == [3.0]
    scheduler.record_idle_training_outcome(outcome, final=True)
    assert (
        scheduler._effective_idle_tail_reserve_sec(_idle_config(), final=True) == 22.0
    )
    assert list(scheduler._idle_worker_final_tail_samples_sec) == [15.0]


def test_prebatch_reclaim_penalty_is_capped_by_batch_estimate() -> None:
    scheduler = DrafterScheduler()
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=3,
        max_batches=4,
        publish_after_success=False,
        idle_batch_estimate_sec=0.5,
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[],
        elapsed_sec=24.0,
        reason="submitted_async",
        metrics={
            "bubble/train_reclaimed_before_first_batch": 1,
            "timing_s/drafter_worker_preflight": 10.7,
            "timing_s/drafter_worker_preflight_to_stop": 0.1,
            "timing_s/drafter_worker_elapsed": 3.2,
        },
    )

    scheduler._record_prebatch_reclaim_penalty(plan, outcome)

    assert scheduler._effective_idle_reclaim_penalty_sec() == pytest.approx(1.5)


def test_idle_worker_prewarm_removes_bootstrap_startup_reserve() -> None:
    scheduler = DrafterScheduler()
    config = replace(
        _idle_config(),
        idle_worker_initial_batch_estimate_sec=None,
    )

    assert scheduler._effective_idle_startup_reserve_sec(config) > 0.0
    scheduler._idle_worker_startup_samples_sec.append(7.0)
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))

    assert scheduler._effective_idle_startup_reserve_sec(config, ("0", "1")) == 0.0
    assert scheduler._effective_idle_startup_reserve_sec(
        config, ("2", "3")
    ) == pytest.approx(7.0)


def test_idle_worker_prewarm_targets_one_candidate_without_electing_writer() -> None:
    class _PrewarmExecutor:
        def __init__(self) -> None:
            self.worker_ids = []

        def prewarm_training_workers(self, worker_ids=None):
            self.worker_ids.append(worker_ids)
            return [
                {"activated": True, "worker_id": worker_id, "reason": "prewarmed"}
                for worker_id in worker_ids
            ]

    scheduler = DrafterScheduler(worker_executor=_PrewarmExecutor())
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))

    scheduler.prewarm_idle_training_workers()

    assert scheduler._worker_executor.worker_ids == [("0", "1")]
    assert ("0", "1") in scheduler._idle_worker_hot_prewarmed_groups
    assert ("2", "3") not in scheduler._idle_worker_hot_prewarmed_groups
    assert scheduler.idle_writer_group() is None


def test_idle_resource_discovery_does_not_elect_writer_without_a_plan() -> None:
    scheduler = _scheduler_with_statuses(("2", "3"))
    config = _auto_idle_config(2)
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))

    now = time.time()
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                must_be_ready_at=now + 10.0,
            )
        )

    resources = scheduler.select_idle_training_resources(config, now=now)

    assert resources.available
    assert resources.reason == "training_group_ready"
    assert resources.worker_ids == ("2", "3")
    assert scheduler.idle_writer_group() is None


def test_first_trainable_idle_plan_elects_writer_not_prewarm_order() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    now = time.time()
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                must_be_ready_at=now + 10.0,
                event_ts=now,
            )
        )

    plan = scheduler.prepare_idle_worker_training_plan(
        _context(),
        replace(_auto_idle_config(2), training_interval_steps=1),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("2", "3")
    assert scheduler.idle_writer_group() == ("2", "3")


def test_idle_resource_discovery_does_not_move_untrained_writer() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    config = _auto_idle_config(2)
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")

    now = time.time()
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    resources = scheduler.select_idle_training_resources(config, now=now)

    assert resources.available
    assert resources.worker_ids == ("2", "3")
    assert scheduler.idle_writer_group() == ("0", "1")


def test_idle_worker_does_not_migrate_after_writer_state_exists() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    config = _auto_idle_config(2)
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_writer_state_version = 8

    now = time.time()
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                idle_confidence=IdleWindowConfidence.CONFIRMED,
                event_ts=now,
            )
        )

    resources = scheduler.select_idle_training_resources(config, now=now)

    assert not resources.available
    assert scheduler.idle_writer_group() == ("0", "1")


def test_idle_worker_selects_later_group_when_earlier_window_is_too_small() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    config = replace(
        _auto_idle_config(2),
        idle_worker_min_idle_window_sec=1.0,
        idle_worker_initial_batch_estimate_sec=0.5,
        idle_worker_deadline_guard_sec=0.0,
    )
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))

    now = time.time()
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=0,
                memory_released=True,
                must_be_ready_at=now + 0.2,
            )
        )
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                must_be_ready_at=now + 5.0,
            )
        )

    resources = scheduler.select_idle_training_resources(config, now=now)

    assert resources.available
    assert resources.worker_ids == ("2", "3")
    assert resources.training_group_id == "idle-group-1"


def test_idle_worker_keeps_single_writer_after_stable_hot_group_successes() -> None:
    scheduler = _scheduler_with_statuses(("2", "3"))
    config = _auto_idle_config(2)
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_writer_state_version = 3

    now = time.time()
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=1,
                memory_released=True,
                must_be_ready_at=now + 10.0,
            )
        )

    resources = scheduler.select_idle_training_resources(config, now=now)

    assert not resources.available
    assert resources.reason == "incomplete_training_group"
    assert resources.worker_ids == ()
    assert scheduler.idle_writer_group() == ("0", "1")


def test_target_lm_head_sync_workers_cover_all_groups_before_writer_lease() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))

    assert scheduler.target_lm_head_sync_worker_ids() == ("0", "1", "2", "3")


def test_target_lm_head_sync_workers_use_writer_once_lease_exists() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")

    assert scheduler.target_lm_head_sync_worker_ids() == ("0", "1")


def test_target_lm_head_sync_workers_use_single_writer_after_state_exists() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_writer_state_version = 3

    assert scheduler.target_lm_head_sync_worker_ids() == ("0", "1")


def test_disabling_replica_local_group_clears_hot_prewarm_state() -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))

    scheduler._disable_replica_local_idle_group(("0", "1"), reason="replica_local_oom")

    assert ("0", "1") not in scheduler._idle_worker_hot_prewarmed_groups
    assert scheduler._metadata_idle_training_groups == (("2", "3"),)


def test_idle_worker_full_collective_fallback_requires_explicit_config() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1],
                "sync_collective_ranks": [0, 1, 2, 3],
            },
            {
                "rank": 2,
                "worker_id": "2",
                "in_drafter_train_group": True,
                "replica_rank": 1,
                "training_group_ranks": [2, 3],
                "full_collective_ranks": [2, 3],
                "sync_collective_ranks": [0, 1, 2, 3],
            },
        ]
    )
    scheduler._disable_replica_local_idle_group(("0", "1"), reason="replica_local_oom")
    scheduler._disable_replica_local_idle_group(("2", "3"), reason="replica_local_oom")

    assert scheduler._idle_training_groups(_auto_idle_config(2)) == ()


def test_idle_worker_full_collective_fallback_cannot_combine_disabled_local_groups() -> (
    None
):
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1],
                "sync_collective_ranks": [0, 1, 2, 3],
            },
            {
                "rank": 2,
                "worker_id": "2",
                "in_drafter_train_group": True,
                "replica_rank": 1,
                "training_group_ranks": [2, 3],
                "full_collective_ranks": [2, 3],
                "sync_collective_ranks": [0, 1, 2, 3],
            },
        ]
    )
    scheduler._disable_replica_local_idle_group(("0", "1"), reason="replica_local_oom")
    scheduler._disable_replica_local_idle_group(("2", "3"), reason="replica_local_oom")
    config = replace(
        _auto_idle_config(2),
        idle_worker_full_collective_fallback=True,
    )

    now = time.time()
    for worker_id in ("0", "1", "2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=0 if worker_id in {"0", "1"} else 1,
                memory_released=True,
                must_be_ready_at=now + 10.0,
            )
        )

    with pytest.raises(ValueError, match="independent replica-local optimizers"):
        scheduler.select_idle_training_resources(config, now=now)


def test_idle_worker_prewarm_tries_next_group_after_failure() -> None:
    class _PrewarmExecutor:
        def __init__(self) -> None:
            self.calls = []

        def prewarm_training_workers(self, worker_ids=None):
            self.calls.append(worker_ids)
            if worker_ids == ("0", "1"):
                return [
                    {
                        "activated": False,
                        "worker_id": worker_id,
                        "reason": "activation_failed",
                        "replica_local_unavailable": True,
                        "replica_local_oom": True,
                        "training_group_ranks": ("0", "1"),
                    }
                    for worker_id in worker_ids
                ]
            return [
                {"activated": True, "worker_id": worker_id, "reason": "prewarmed"}
                for worker_id in worker_ids
            ]

    executor = _PrewarmExecutor()
    scheduler = DrafterScheduler(worker_executor=executor)
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))

    scheduler.prewarm_idle_training_workers()

    assert executor.calls == [("0", "1"), ("2", "3")]
    assert ("0", "1") in scheduler._disabled_replica_local_idle_groups
    assert ("2", "3") in scheduler._idle_worker_hot_prewarmed_groups


def test_replica_local_activation_failure_disables_idle_group() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1],
                "sync_collective_ranks": [0, 1, 2, 3],
                "idle_collective_scope": "replica_local",
            }
        ]
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="activation-failed-plan",
        idle_usable_window_sec=10.0,
        idle_batch_estimate_sec=1.0,
        worker_snapshots={
            "0": {"worker_incarnation": "worker-0"},
            "1": {"worker_incarnation": "worker-1"},
        },
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=time.time())
    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(
            raw_results=[
                {
                    "ready": False,
                    "participating": True,
                    "activated": False,
                    "worker_id": "0",
                    "worker_incarnation": "worker-0",
                    "reason": "activation_failed",
                    "replica_local_unavailable": True,
                    "replica_local_oom": True,
                },
                {
                    "ready": False,
                    "participating": True,
                    "activated": False,
                    "worker_id": "1",
                    "worker_incarnation": "worker-1",
                    "reason": "activation_failed",
                    "replica_local_unavailable": True,
                    "replica_local_oom": True,
                },
            ],
            elapsed_sec=0.5,
            launched=False,
            reason="worker_preflight_failed",
        ),
        runtime_state=runtime_state,
        plan=plan,
    )

    scheduler._record_training_outcome(plan, outcome)
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=0,
                memory_released=True,
                must_be_ready_at=time.time() + 30.0,
            )
        )
    next_plan = scheduler.prepare_training_plan(
        _context(),
        config,
    )

    assert outcome.metrics["bubble/replica_local_unavailable"] == 1
    assert outcome.metrics["bubble/replica_local_oom"] == 1
    assert not next_plan.launch
    assert next_plan.reason == "replica_local_unavailable"


def test_idle_worker_prebatch_reclaim_penalty_does_not_block_usable_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    failed_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="reclaimed-plan",
        idle_usable_window_sec=10.0,
        idle_batch_estimate_sec=1.0,
    )
    failed_outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[],
        elapsed_sec=3.0,
        reason="submitted_async",
        metrics={
            "bubble/train_reclaimed_before_first_batch": 1,
            "timing_s/drafter_worker_preflight": 1.2,
            "timing_s/drafter_worker_preflight_to_stop": 0.1,
            "timing_s/drafter_worker_cleanup": 1.7,
        },
    )

    scheduler._record_training_outcome(failed_plan, failed_outcome)

    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=time.time() + 5.0,
            )
        )
    next_plan = scheduler.prepare_training_plan(
        _context(),
        config,
    )

    assert next_plan.launch
    assert next_plan.reason == "training_ready"
    assert next_plan.idle_reclaim_penalty_sec == pytest.approx(2.3)
    assert next_plan.metrics()["bubble/idle_reclaim_penalty_active"] == 1
    assert next_plan.metrics()["bubble/idle_reclaim_penalty_s"] == pytest.approx(2.3)


def test_idle_worker_prebatch_reclaim_penalty_allows_large_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_reclaim_penalty_sec = 9.5
    scheduler._idle_worker_reclaim_penalty_last_step = 10

    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=time.time() + 30.0,
            )
        )
    plan = scheduler.prepare_training_plan(
        replace(_context(), global_step=11),
        config,
    )

    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.idle_reclaim_penalty_sec == pytest.approx(4.75)


def test_idle_worker_prebatch_reclaim_streak_keeps_multi_batch_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_prebatch_reclaim_streak = 1
    scheduler._idle_worker_reclaim_penalty_sec = 2.0
    scheduler._idle_worker_reclaim_penalty_last_step = 10

    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=time.time() + 80.0,
            )
        )
    plan = scheduler.prepare_training_plan(
        replace(_context(), global_step=10),
        config,
    )

    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.max_batches == config.train_batches_per_trigger


def test_idle_worker_reclaim_penalty_decays_on_next_step() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_reclaim_penalty_sec = 8.0
    scheduler._idle_worker_reclaim_penalty_last_step = 10

    scheduler._decay_idle_reclaim_penalty(11)
    assert scheduler._effective_idle_reclaim_penalty_sec() == pytest.approx(4.0)

    scheduler._decay_idle_reclaim_penalty(12)
    assert scheduler._effective_idle_reclaim_penalty_sec() == pytest.approx(2.0)


def test_idle_worker_prebatch_reclaim_penalty_shrinks_from_stale_high_value() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_reclaim_penalty_sec = 35.56564683914185
    scheduler._idle_worker_reclaim_penalty_last_step = 10
    failed_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=11,
        max_batches=3,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="reclaimed-plan",
        idle_usable_window_sec=27.77753145694733,
        idle_batch_estimate_sec=4.228558301925659,
        idle_startup_reserve_sec=2.7478981018066406,
    )
    failed_outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[],
        elapsed_sec=25.0,
        reason="submitted_async",
        metrics={
            "bubble/train_reclaimed_before_first_batch": 1,
            "timing_s/drafter_worker_elapsed": 3.56601881980896,
            "timing_s/drafter_worker_preflight": 2.7478981018066406,
            "timing_s/drafter_worker_preflight_to_stop": 0.9483940601348877,
        },
    )

    scheduler._record_training_outcome(failed_plan, failed_outcome)

    assert scheduler._effective_idle_reclaim_penalty_sec() == pytest.approx(
        4.228558301925659 * 3.0
    )


def test_idle_worker_success_resets_prebatch_reclaim_penalty() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_prebatch_reclaim_streak = 2
    scheduler._idle_worker_reclaim_penalty_sec = 9.5
    scheduler._idle_worker_reclaim_penalty_last_step = 10
    success_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="success-plan",
    )
    success_outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=4.0,
        reason="submitted_async",
        metrics={
            "bubble/train_reclaimed_before_first_batch": 0,
            "timing_s/drafter_worker_training_loop": 1.0,
        },
    )

    scheduler._record_training_outcome(success_plan, success_outcome)

    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=time.time() + 30.0,
            )
        )
    plan = scheduler.prepare_training_plan(
        replace(_context(), global_step=11),
        config,
    )

    assert plan.launch
    assert plan.reason == "training_ready"


def test_idle_worker_training_does_not_wait_for_training_interval() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    config = replace(
        _auto_idle_config(2),
        training_interval_steps=5,
        idle_worker_min_idle_window_sec=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )
    context = DrafterScheduleContext(
        global_step=6,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
    )

    plan = scheduler.prepare_training_plan(context, config)

    assert not plan.interval_matched
    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.max_batches == config.train_batches_per_trigger


def test_idle_worker_plan_uses_buffered_data_target_version() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    buffered_status = replace(
        _status("group"),
        current_step=11,
        newest_sample_step=10,
        data_version=10,
        target_version=10,
        worker_snapshots={
            worker_id: {
                "buffer_version": 1,
                "data_version": 10,
                "worker_incarnation": f"worker-{worker_id}",
                "trainable_samples": 5,
            }
            for worker_id in ("0", "1")
        },
    )
    context = DrafterScheduleContext(
        global_step=11,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
        data_status=buffered_status,
    )

    plan = scheduler.prepare_training_plan(context, _idle_config())

    assert plan.launch
    assert plan.source_global_step == 11
    assert plan.data_version == 10
    assert plan.required_target_version == 10


def test_idle_worker_publish_is_async_after_weight_update() -> None:
    scheduler = DrafterScheduler()
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
    )

    publish_plan = scheduler.plan_publish(
        global_step=10,
        drafter_trained=True,
        config=_idle_config(),
        training_plan=plan,
    )

    assert publish_plan.publish
    assert publish_plan.asynchronous


def test_forced_completion_metrics_are_blocking_not_async_work() -> None:
    state = DrafterRuntimeState()
    plan = TrainingPlan(
        launch=True,
        reason="quota_forced_completion_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=11,
        max_batches=1,
        publish_after_success=True,
        data_version=10,
        required_target_version=10,
        plan_id="forced-plan",
        target_worker_ids=("0",),
        worker_snapshots={
            "0": {
                "buffer_version": 1,
                "data_version": 10,
                "worker_incarnation": "worker-0",
                "trainable_samples": 4,
            }
        },
    )
    state.submit(plan, started_at=time.time())
    state.mark_running()

    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(
            raw_results=[
                {
                    "trained": True,
                    "triggered": True,
                    "source_global_step": 11,
                    "execution_strategy": "rollout_idle_worker",
                    "attempted_steps": 1,
                    "successful_steps": 1,
                    "optimizer_step": 20,
                    "buffer_size_before": 4,
                    "buffer_size_after": 0,
                    "elapsed_sec": 0.5,
                    "reason": "trained",
                    "publish_snapshot_cached": True,
                    "worker_id": "0",
                    "worker_incarnation": "worker-0",
                    "plan_id": "forced-plan",
                    "data_version": 10,
                    "target_version": 10,
                    "is_publish_leader": True,
                }
            ],
            elapsed_sec=0.5,
        ),
        runtime_state=state,
        plan=plan,
    )

    assert outcome.metrics["bubble/training_quota_topup_completed"] == 1
    assert outcome.metrics["bubble/training_quota_force_complete_completed"] == 1
    assert "timing_s/drafter_async_training_work" not in outcome.metrics


def test_scheduler_duplicate_tuning_subtrees_do_not_override_training_fields() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
            "publish_async": False,
            "step": 5,
            "collection_sample_rate": 0.5,
            "max_collect_samples_per_step_per_replica": 6,
            "max_collect_tokens_per_step_per_replica": 7,
            "min_trainable_batches": 8,
            "require_full_batch": False,
            "sample_last_n_steps": 10,
            "scheduler": {
                "collection": {
                    "interval_steps": 20,
                    "sample_rate": 0.1,
                    "max_samples_per_step_per_replica": 60,
                    "max_tokens_per_step_per_replica": 70,
                },
                "trigger": {
                    "interval_steps": 30,
                    "min_trainable_batches": 80,
                },
                "budget": {
                    "max_batches": 50,
                    "require_full_batch": True,
                    "sample_last_n_steps": 100,
                },
                "publish": {
                    "interval_optimizer_steps": 40,
                    "async_update": True,
                },
            },
        }
    )

    assert config.collect_interval_steps == 2
    assert config.training_interval_steps == 3
    assert config.publish_interval_steps == 4
    assert config.publish_async is False
    assert config.train_batches_per_trigger == 5
    assert config.collection_sample_rate == 0.5
    assert config.max_collect_samples_per_replica == 6
    assert config.max_collect_tokens_per_replica == 7
    assert config.min_trainable_batches == 8
    assert config.require_full_batch is False
    assert config.sample_last_n_steps == 10


def test_idle_worker_window_is_admission_not_hard_batch_cap() -> None:
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [
                {
                    **_status("0").__dict__,
                    "available": True,
                    "rank": 0,
                },
                {
                    **_status("1").__dict__,
                    "available": True,
                    "rank": 1,
                },
            ],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
                event_ts=100.0,
            )
        )

    plan = scheduler.prepare_training_plan(
        _context(), replace(_idle_config(), publish_interval_steps=1000)
    )

    assert plan.launch
    assert plan.max_batches == _idle_config().train_batches_per_trigger
    assert plan.target_worker_ids == ("0", "1")
    assert plan.training_group_id == "idle-group-0"
    assert plan.to_worker_payload()["execution_strategy"] == "rollout_idle_worker"


def test_idle_worker_dynamic_cap_grows_after_reaching_cap() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="success-plan",
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[{"reason": "trained", "stop_reason": "max_batches_reached"}],
        elapsed_sec=1.0,
        reason="submitted_async",
        metrics={"bubble/train_reclaimed_before_first_batch": 0},
    )

    scheduler._record_training_outcome(plan, outcome)

    assert scheduler._effective_idle_dynamic_batch_cap(config) == 2


def test_idle_worker_data_version_rejection_does_not_reduce_dynamic_cap() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_dynamic_batch_cap = 4
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=4,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="version-race-plan",
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[{"ready": False, "reason": "data_version_changed"}],
        elapsed_sec=0.1,
        reason="data_version_changed",
        metrics={},
    )

    scheduler._record_training_outcome(plan, outcome)

    assert scheduler._effective_idle_dynamic_batch_cap(config) == 4


def test_idle_worker_prebatch_reclaim_does_not_reduce_dynamic_cap() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_dynamic_batch_cap = 4
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=4,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="prebatch-reclaim-plan",
        idle_confidence=IdleWindowConfidence.CONFIRMED,
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[
            {"worker_id": "2", "reason": "not_in_training_group"},
            {"worker_id": "0", "stop_reason": "reclaim_requested"},
        ],
        elapsed_sec=0.1,
        reason="completed",
        metrics={"bubble/train_reclaimed_before_first_batch": 1},
    )

    scheduler._record_idle_dynamic_batch_cap(plan, outcome)

    assert scheduler._effective_idle_dynamic_batch_cap(config) == 4
    assert scheduler._idle_stop_reason(outcome, plan) == "reclaim_requested"


def test_idle_worker_expired_preflight_does_not_reduce_dynamic_cap() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_dynamic_batch_cap = 4
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=4,
        publish_after_success=True,
        target_worker_ids=("0", "1"),
        training_group_id="idle-group-0",
        plan_id="expired-plan",
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=0,
        worker_results=[],
        raw_results=[{"worker_id": "0", "reason": "plan_expired_before_preflight"}],
        elapsed_sec=0.1,
        reason="worker_preflight_failed",
        metrics={},
    )

    scheduler._record_idle_dynamic_batch_cap(plan, outcome)

    assert scheduler._effective_idle_dynamic_batch_cap(config) == 4


def test_idle_worker_dynamic_cap_reduces_on_gen_slowdown() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    config = _auto_idle_config(2)
    scheduler._idle_worker_dynamic_batch_cap = 4

    scheduler.record_step_metrics(
        {"timing_per_token_ms/gen": 0.2},
        config,
    )
    metrics = scheduler.record_step_metrics(
        {
            "timing_per_token_ms/gen": 0.24,
            "drafter/idle_trained": 1,
        },
        config,
    )

    assert metrics["bubble/gen_slowdown_cap_reduced"] == 1
    assert scheduler._effective_idle_dynamic_batch_cap(config) == 2


def test_repeated_gen_slowdown_temporarily_suspends_new_bubble_launches() -> None:
    scheduler = DrafterScheduler()
    now = time.time()
    config = replace(
        _idle_config(),
        idle_worker_training_groups=(("0",),),
        idle_worker_dynamic_batch_cap=True,
        idle_worker_initial_dynamic_batches=1,
        idle_worker_gen_slowdown_patience=2,
        idle_worker_gen_slowdown_cooldown_steps=2,
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            idle_confidence=IdleWindowConfidence.CONFIRMED,
            must_be_ready_at=now + 10.0,
            event_ts=now,
        )
    )
    scheduler.record_step_metrics(
        {"timing_per_token_ms/gen": 0.2},
        config,
        global_step=1,
    )
    for step in (2, 3):
        scheduler.record_step_metrics(
            {
                "timing_per_token_ms/gen": 0.24,
                "drafter/idle_trained": 1,
            },
            config,
            global_step=step,
        )

    suspended = scheduler.select_idle_training_resources(config, now=now)

    assert not suspended.available
    assert suspended.reason == "generation_slowdown_cooldown"
    scheduler.record_step_metrics(
        {"timing_per_token_ms/gen": 0.2},
        config,
        global_step=5,
    )
    assert scheduler.select_idle_training_resources(config, now=now).available


def test_idle_worker_async_submit_and_poll_completion() -> None:
    events = []
    state = DrafterRuntimeState()
    plan = replace(
        DrafterScheduler().plan_training(
            DrafterScheduleContext(
                global_step=10,
                training_mode="online",
                collected_samples_this_step=2,
                oldlogprob_collection_requested=False,
                data_status=TrainingDataStatus(
                    **{
                        **_status("0").__dict__,
                        "worker_snapshots": {
                            "0": {
                                "buffer_version": 1,
                                "data_version": 10,
                                "worker_incarnation": "worker-0",
                                "trainable_samples": 5,
                            }
                        },
                    }
                ),
            ),
            _idle_config(),
            resources=type(
                "Resources",
                (),
                {
                    "worker_ids": ("0",),
                    "training_group_id": "idle-group-0",
                    "minimum_idle_window_sec": 3.0,
                },
            )(),
        ),
        max_batches=1,
    )
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append(("submit", payload)) or "ref",
            resolve=lambda value: (
                [
                    {
                        "participating": True,
                        "ready": True,
                        "worker_id": "0",
                        "worker_incarnation": "worker-0",
                        "data_version": 10,
                        "target_version": 10,
                    }
                ]
                if value == "preflight-ref"
                else [
                    {
                        "trained": True,
                        "triggered": True,
                        "source_global_step": 10,
                        "execution_strategy": "rollout_idle_worker",
                        "attempted_steps": 1,
                        "successful_steps": 1,
                        "optimizer_step": 1,
                        "buffer_size_before": 5,
                        "buffer_size_after": 4,
                        "elapsed_sec": 0.8,
                        "reason": "trained",
                        "publish_snapshot_cached": True,
                        "worker_id": "0",
                        "worker_incarnation": "worker-0",
                        "plan_id": plan.plan_id,
                        "data_version": 10,
                        "target_version": 10,
                        "is_publish_leader": True,
                    }
                ]
            ),
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: "preflight-ref",
            abort_preflight=lambda plan_id: [],
            poll=lambda submission: (
                events.append(("poll", submission)) or (True, submission)
            ),
        )
    )

    execution = scheduler.execute_training_plan(plan, runtime_state=state)

    assert execution.reason == "submitted_async"
    assert state.status is DrafterRuntimeStatus.RUNNING
    assert events[0] == ("submit", plan.to_worker_payload())

    outcome = scheduler.poll_pending_training(runtime_state=state)

    assert outcome is not None
    assert outcome.trained
    assert outcome.successful_steps == 1
    assert scheduler._idle_worker_batch_estimate_sec == pytest.approx(0.8)
    assert state.status is DrafterRuntimeStatus.IDLE


def _trainer_with_idle_config() -> SpecoRayPPOTrainer:
    assert SpecoRayPPOTrainer is not None
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = {
        "actor_rollout_ref": {
            "rollout": {
                "data_parallel_size": 2,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "training": {
                        "scheduler": {
                            "execution": {"strategy": "rollout_idle_worker"},
                            "idle_worker": {
                                "training_groups": [["worker-0", "worker-1"]],
                                "initial_batch_estimate_sec": 0.5,
                                "deadline_guard_sec": 0.1,
                            },
                        }
                    },
                },
            }
        }
    }
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._drafter_runtime_state = DrafterRuntimeState()
    return trainer


def _trainer_with_default_idle_budget() -> SpecoRayPPOTrainer:
    assert SpecoRayPPOTrainer is not None
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = {
        "actor_rollout_ref": {
            "rollout": {
                "data_parallel_size": 1,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "training": {
                        "scheduler": {
                            "execution": {"strategy": "rollout_idle_worker"},
                            "idle_worker": {
                                "training_groups": [["worker-0"]],
                            },
                        }
                    },
                },
            }
        }
    }
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._drafter_runtime_state = DrafterRuntimeState()
    return trainer


class _ActivationRecorder:
    def __init__(self) -> None:
        self.calls = 0
        self.prewarm_calls = 0

    def activate_training_workers(self):
        self.calls += 1
        return []

    def prewarm_idle_training_workers(self):
        self.prewarm_calls += 1
        return []


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
@pytest.mark.parametrize("keep_hot", [False, True])
def test_boundary_parks_completed_writer_without_releasing_quota(keep_hot) -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"][
        "idle_worker"
    ]["keep_hot_between_plans"] = keep_hot
    scheduler = trainer._drafter_scheduler
    scheduler._idle_worker_writer_group = ("worker-0", "worker-1")
    scheduler._idle_worker_writer_state_version = 10
    scheduler._training_quota_debt_steps = 11
    scheduler._idle_worker_hot_prewarmed_groups.add(scheduler._idle_worker_writer_group)
    calls = []
    trainer._require_speco_worker_group = lambda: SimpleNamespace(
        park_drafter_training_model=lambda: calls.append("park") or [{"parked": True}]
    )
    trainer._ray_get_if_needed = lambda result: result

    trainer._speco_reclaim_rollout_idle_workers_before_post_rollout()

    assert calls == ([] if keep_hot else ["park"])
    assert scheduler._training_quota_debt_steps == 11
    assert scheduler._idle_worker_writer_state_version == 10
    assert scheduler._idle_worker_writer_group == ("worker-0", "worker-1")
    assert bool(scheduler._idle_worker_hot_prewarmed_groups) == keep_hot


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_before_fit_bootstraps_rollout_idle_models_on_cpu() -> None:
    trainer = _trainer_with_idle_config()
    recorder = _ActivationRecorder()
    trainer._drafter_scheduler = recorder

    trainer._speco_activate_drafter_training_model_before_fit()

    assert recorder.calls == 1
    assert recorder.prewarm_calls == 0
    assert recorder._idle_worker_bootstrapped


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_before_fit_activation_still_runs_for_sync_training() -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"][
        "execution"
    ]["strategy"] = "sync"
    recorder = _ActivationRecorder()
    trainer._drafter_scheduler = recorder

    trainer._speco_activate_drafter_training_model_before_fit()

    assert recorder.calls == 1
    assert recorder.prewarm_calls == 0


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_generation_completion_does_not_create_synthetic_idle_window() -> None:
    trainer = _trainer_with_idle_config()
    output = _FakeGenerationOutput(
        [{"replica_rank": 0, "id": "a"}, {"replica_rank": 1, "id": "b"}]
    )

    start_metrics = trainer._speco_emit_rollout_generation_started()
    complete_metrics = trainer._speco_emit_rollout_generation_completed(output)

    assert start_metrics["bubble/idle_workers"] == 0
    assert complete_metrics == {}
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 0
    assert output.non_tensor_batch["drafter_sample"][0]["id"] == "a"


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_generation_completion_does_not_pollute_runtime_idle_window_history() -> None:
    trainer = _trainer_with_idle_config()
    scheduler = trainer._drafter_scheduler
    output = _FakeGenerationOutput([])
    for replica_rank, worker_id in enumerate(("worker-0", "worker-1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                event_ts=100.0 + replica_rank,
            )
        )
    trainer._speco_runtime_idle_events_this_generation = 2

    metrics = trainer._speco_emit_rollout_generation_completed(output)

    assert metrics == {}
    assert scheduler._replica_idle_window_samples_sec == {}
    assert scheduler._replica_idle_started_at == {0: 100.0, 1: 101.0}


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_runtime_idle_events_are_accumulated_across_event_loop_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verl_speco.trainer import speco_ray_trainer

    trainer = _trainer_with_idle_config()
    trainer._speco_runtime_idle_callback_verified = False
    trainer._speco_runtime_idle_events_this_generation = 0
    trainer._speco_runtime_sample_events_this_generation = 0
    event_batches = [
        [
            {
                "event_type": RolloutWorkerEventType.WORKER_IDLE.value,
                "worker_id": "worker-0",
                "replica_rank": 0,
                "memory_released": True,
            }
        ],
        [],
    ]
    monkeypatch.setattr(
        speco_ray_trainer,
        "drain_rollout_idle_events",
        lambda _name: event_batches.pop(0) if event_batches else [],
    )

    first_metrics = trainer._speco_drain_rollout_idle_events()
    second_metrics = trainer._speco_drain_rollout_idle_events()

    assert first_metrics["bubble/runtime_worker_events_drained"] == 1
    assert second_metrics == {}
    assert trainer._speco_runtime_idle_events_this_generation == 1


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_fallback_idle_events_from_generation_output() -> None:
    trainer = _trainer_with_idle_config()
    output = _FakeGenerationOutput(
        [{"replica_rank": 0, "id": "a"}, {"replica_rank": 1, "id": "b"}]
    )

    trainer._speco_emit_rollout_generation_started()
    metrics = trainer._speco_emit_rollout_idle_from_generation_output(
        output,
        reason="test_no_runtime_events",
    )

    assert metrics["bubble/fallback_idle_events"] == 2
    assert metrics["bubble/idle_workers"] == 2
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 2
    resources = trainer._drafter_scheduler.select_idle_training_resources(
        trainer._speco_drafter_schedule_config()
    )
    assert resources.available is False
    assert resources.reason == "incomplete_training_group"
    assert output.non_tensor_batch["drafter_sample"][0]["id"] == "a"


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_strict_idle_mode_rejects_synthetic_idle_window() -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"][
        "idle_worker"
    ]["require_runtime_idle_events"] = True
    output = _FakeGenerationOutput([{"replica_rank": 0, "id": "a"}])

    metrics = trainer._speco_emit_rollout_idle_from_generation_output(
        output,
        reason="test_no_runtime_events",
    )

    assert metrics == {"bubble/fallback_idle_events_disabled": 1}
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 0


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_fallback_idle_uses_bootstrap_window_floor() -> None:
    trainer = _trainer_with_default_idle_budget()

    deadline_ts = trainer._speco_rollout_idle_fallback_deadline_ts()

    assert deadline_ts - time.time() >= 9.0


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_reclaims_active_idle_workers_before_next_generation() -> None:
    trainer = _trainer_with_idle_config()
    trainer._speco_park_drafter_training_memory = lambda boundary: {}
    events = []
    trainer._drafter_scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
            reclaim=lambda worker_ids: events.append(worker_ids),
        )
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("worker-0", "worker-1"),
    )
    trainer._drafter_runtime_state.submit(plan, started_at=time.time())
    trainer._drafter_runtime_state.mark_running()
    drain_calls = []
    trainer._speco_wait_pending_drafter_training = lambda: (
        drain_calls.append(True) or (plan, None)
    )

    metrics = trainer._speco_reclaim_rollout_idle_workers_before_generation()

    assert metrics["bubble/reclaim_requested"] == 1
    assert metrics["bubble/reclaim_drained"] == 0
    assert metrics["timing_s/drafter_reclaim_wait"] >= 0.0
    assert metrics["timing_s/drafter_critical_path_before_generation"] >= 0.0
    assert events == [("worker-0", "worker-1")]
    assert drain_calls == [True]
    assert len(trainer._drafter_scheduler._idle_worker_reclaim_samples_sec) == 1


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_can_skip_reclaim_drain_when_configured() -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"][
        "idle_worker"
    ]["drain_before_next_rollout"] = False
    events = []
    trainer._drafter_scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
            reclaim=lambda worker_ids: events.append(worker_ids),
        )
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("worker-0", "worker-1"),
    )
    trainer._drafter_runtime_state.submit(plan, started_at=time.time())
    trainer._drafter_runtime_state.mark_running()
    drain_calls = []
    trainer._speco_wait_pending_drafter_training = lambda: (
        drain_calls.append(True) or (plan, None)
    )

    metrics = trainer._speco_reclaim_rollout_idle_workers(boundary="test")

    assert metrics == {"bubble/reclaim_requested": 1}
    assert events == [("worker-0", "worker-1")]
    assert drain_calls == []


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_post_generation_event_drain_never_launches_new_training() -> None:
    trainer = _trainer_with_idle_config()
    trainer._speco_drain_rollout_idle_events = lambda: {
        "bubble/runtime_worker_events_drained": 1
    }
    trainer._speco_try_launch_rollout_idle_training = lambda: pytest.fail(
        "post-generation event drain must not launch drafter training"
    )
    trainer._speco_record_rollout_idle_metrics = lambda metrics: None

    metrics = trainer._speco_service_rollout_idle_events(allow_launch=False)

    assert metrics["bubble/post_generation_launch_suppressed"] == 1


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_post_rollout_reclaim_hands_completion_to_safe_publish_callback() -> None:
    trainer = _trainer_with_idle_config()
    trainer._speco_park_drafter_training_memory = lambda boundary: {}
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"][
        "idle_worker"
    ]["drain_before_next_rollout"] = False
    events = []
    trainer._drafter_scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
            reclaim=lambda worker_ids: events.append(("reclaim", worker_ids)),
        )
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("worker-0", "worker-1"),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        raw_results=[],
        elapsed_sec=0.1,
        reason="completed",
        metrics={},
    )
    trainer._drafter_runtime_state.submit(plan, started_at=time.time())
    trainer._drafter_runtime_state.mark_running()
    trainer._speco_wait_pending_drafter_training = lambda: (plan, outcome)

    metrics = trainer._speco_reclaim_rollout_idle_workers_before_post_rollout(
        on_completed=lambda completed_plan, completed_outcome: events.append(
            ("completed", completed_plan, completed_outcome)
        )
        or {"bubble/deferred_publish_enqueued": 1}
    )

    assert events[0] == ("reclaim", ("worker-0", "worker-1"))
    assert events[1] == ("completed", plan, outcome)
    assert metrics["bubble/reclaim_drained"] == 1
    assert metrics["bubble/deferred_publish_enqueued"] == 1
    assert metrics["timing_s/drafter_critical_path_before_post_rollout"] >= 0.0


def test_writer_failover_is_blocked_after_optimizer_state_exists() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    scheduler._idle_worker_writer_group = ("0", "1")
    scheduler._idle_worker_writer_state_version = 10

    scheduler._disable_replica_local_idle_group(("0", "1"), reason="replica_local_oom")
    plan = scheduler.prepare_idle_worker_training_plan(
        _context(), replace(_auto_idle_config(2), training_interval_steps=1)
    )

    assert scheduler.idle_writer_group() is None
    assert scheduler._idle_worker_writer_migration_blocked
    assert not plan.launch
    assert plan.reason == "writer_state_migration_required"


def test_quota_debt_does_not_expand_an_insufficient_historical_window() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 1
    scheduler._training_quota_debt_steps = 10
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            event_ts=time.time() - 0.01,
        )
    )
    scheduler._replica_idle_window_samples_sec[0] = deque([0.03], maxlen=32)
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        train_batches_per_trigger=10,
        gradient_accumulation_steps=2,
    )
    context = replace(
        _context(),
        global_step=3,
        data_status=replace(
            _status("0", batches=4),
            trainable_batches=4,
            trainable_valid_tokens=4096,
            target_version=3,
        ),
    )

    plan = scheduler.prepare_idle_worker_training_plan(context, config)

    assert not plan.launch
    assert plan.reason == "window_too_small"
    assert plan.deadline_ts is None


def test_historical_window_strictly_limits_planned_batches() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    scheduler._idle_worker_writer_group = ("0",)
    scheduler._idle_worker_writer_state_version = 1
    scheduler._training_quota_debt_steps = 10
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            event_ts=time.time(),
        )
    )
    scheduler._replica_idle_window_samples_sec[0] = deque([2.05], maxlen=32)
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        train_batches_per_trigger=10,
        gradient_accumulation_steps=1,
    )
    context = replace(
        _context(),
        global_step=3,
        data_status=replace(
            _status("0", batches=4),
            trainable_batches=4,
            trainable_valid_tokens=4096,
            target_version=3,
        ),
    )

    plan = scheduler.prepare_idle_worker_training_plan(context, config)

    assert plan.launch
    assert plan.idle_window_source == "historical_observed"
    assert plan.idle_window_batches == 2
    assert plan.max_batches == plan.idle_window_batches
    assert plan.deadline_ts is not None
    assert time.time() < plan.deadline_ts < time.time() + 2.05


@pytest.mark.parametrize("debt", [1, 3, 20])
@pytest.mark.parametrize("source", ["historical_observed", "runtime_deadline"])
def test_final_snapshot_must_fit_before_completing_quota(debt, source) -> None:
    from verl_speco.trainer.scheduler.schedule_types import AvailableTrainingResources

    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    scheduler._idle_worker_bootstrapped = True
    scheduler._training_quota_debt_steps = debt
    scheduler._training_quota_data_version = 10
    scheduler._idle_worker_partial_tail_samples_sec.append(0.1)
    scheduler._idle_worker_final_tail_samples_sec.append(10.0)
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        train_batches_per_trigger=20,
        idle_worker_initial_batch_estimate_sec=0.2,
        idle_worker_deadline_guard_sec=0.1,
    )
    resources = AvailableTrainingResources(
        True,
        "training_group_ready",
        worker_ids=("0", "1"),
        minimum_idle_window_sec=5.0,
        idle_window_source=source,
    )
    plan = scheduler.plan_training(
        replace(_context(), data_status=_status("group", batches=20)),
        config,
        resources=resources,
        require_interval=False,
    )
    assert plan.max_batches == debt - 1
    assert not plan.publish_after_success
    assert plan.launch == (debt > 1)
    if debt == 1:
        assert plan.reason == "final_tail_window_too_small"
    assert scheduler._training_quota_debt_steps == debt


@pytest.mark.parametrize("debt", [1, 3, 20])
def test_hot_writer_finishes_quota_before_safe_point_snapshot(debt) -> None:
    from verl_speco.trainer.scheduler.schedule_types import AvailableTrainingResources

    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    scheduler._idle_worker_bootstrapped = True
    scheduler._training_quota_debt_steps = debt
    scheduler._training_quota_data_version = 10
    scheduler._idle_worker_final_tail_samples_sec.append(20.0)
    config = replace(
        _idle_config(),
        training_quota_enable=True,
        training_quota_target_steps=20,
        train_batches_per_trigger=20,
        idle_worker_keep_hot_between_plans=True,
        idle_worker_initial_batch_estimate_sec=0.1,
        idle_worker_deadline_guard_sec=0.1,
    )
    plan = scheduler.plan_training(
        replace(_context(), data_status=_status("group", batches=40)),
        config,
        resources=AvailableTrainingResources(
            True,
            "training_group_ready",
            worker_ids=("0", "1"),
            minimum_idle_window_sec=5.0,
            idle_window_source="historical_observed",
        ),
        require_interval=False,
    )
    assert plan.launch and plan.max_batches == debt
    assert plan.publish_after_success and plan.publish_snapshot_deferred
    assert plan.to_worker_payload()["publish_snapshot_deferred"]
    assert plan.to_worker_payload()["idle_snapshot_estimate_sec"] == 20.0
    assert plan.to_worker_payload()["reason"] == plan.reason
    assert scheduler._training_quota_debt_steps == debt


def test_first_step_does_not_poison_steady_budget_and_head_prepare_is_not_free() -> (
    None
):
    scheduler = DrafterScheduler()
    config = replace(_idle_config(), idle_worker_initial_batch_estimate_sec=None)
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=3,
        publish_after_success=False,
        target_worker_ids=("0", "1"),
        required_target_version=10,
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=3,
        worker_results=[],
        elapsed_sec=1.5,
        reason="trained",
        metrics={"timing_s/drafter_worker_target_head_prepare": 0.2},
        raw_results=[
            {
                "worker_id": "0",
                "successful_steps": 3,
                "optimizer_step_samples_sec": [0.9, 0.3, 0.3],
                "optimizer_warmup_steps": 0,
            }
        ],
    )
    scheduler._record_training_outcome(plan, outcome)
    assert scheduler._effective_idle_batch_estimate_sec(config) == pytest.approx(0.3)
    assert ("0", "1") in scheduler._idle_optimizer_initialized_groups
    assert scheduler._effective_idle_startup_reserve_sec(
        config, ("0", "1")
    ) == pytest.approx(0.6)
    scheduler._training_quota_data_version = 11
    assert scheduler._effective_idle_startup_reserve_sec(
        config, ("0", "1")
    ) == pytest.approx(0.8)
    one_step = replace(
        outcome,
        successful_steps=1,
        raw_results=[
            {
                "worker_id": "0",
                "successful_steps": 1,
                "optimizer_step_samples_sec": [0.9],
                "optimizer_warmup_steps": 0,
            }
        ],
    )
    scheduler._record_training_outcome(replace(plan, max_batches=1), one_step)
    assert scheduler._effective_idle_batch_estimate_sec(config) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "reason", ["quota_topup_training_ready", "quota_forced_completion_ready"]
)
def test_sync_topup_first_step_does_not_inflate_idle_restart_budget(reason) -> None:
    scheduler = DrafterScheduler()
    scheduler._metadata_full_collective_idle_groups = (("0", "1", "2", "3"),)
    config = replace(_idle_config(), idle_worker_initial_batch_estimate_sec=None)
    plan = TrainingPlan(
        launch=True,
        reason=reason,
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=3,
        publish_after_success=False,
        target_worker_ids=("0", "1"),
        gradient_accumulation_steps=2,
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=3,
        worker_results=[],
        elapsed_sec=2.8,
        reason="trained",
        metrics={},
        raw_results=[
            {
                "worker_id": "0",
                "successful_steps": 3,
                "optimizer_step_samples_sec": [2.2, 0.3, 0.3],
                "optimizer_warmup_steps": 0,
            }
        ],
    )
    scheduler._record_training_outcome(plan, outcome)
    assert not scheduler._idle_worker_first_step_samples_sec
    assert scheduler._effective_idle_batch_estimate_sec(config) == pytest.approx(0.15)
    assert scheduler._effective_idle_first_step_extra_sec(config, ("0", "1")) == 0
    # Real idle observations still calibrate the first-step reserve.
    scheduler._record_training_outcome(replace(plan, reason="training_ready"), outcome)
    assert list(scheduler._idle_worker_first_step_samples_sec) == [1.1]
    assert scheduler._effective_idle_first_step_extra_sec(
        config, ("0", "1")
    ) == pytest.approx(1.9)


def test_window_histories_are_separated_by_close_source() -> None:
    scheduler = DrafterScheduler()
    for opened, closed, source in [
        (100.0, 102.0, "generation_started"),
        (200.0, 210.0, "generation_completed"),
    ]:
        scheduler._replica_idle_started_at[0] = opened
        scheduler._record_observed_replica_idle_window(0, closed, source=source)
    metrics = scheduler.idle_worker_metrics()
    assert metrics["bubble/replica_0_generation_start_closed_p75_s"] == 2.0
    assert metrics["bubble/replica_0_generation_end_closed_p75_s"] == 10.0
    assert metrics["bubble/replica_0_generation_start_closed_samples"] == 1
    assert metrics["bubble/replica_0_generation_end_closed_samples"] == 1


def test_idle_timeline_reports_callback_release_and_delivery(monkeypatch) -> None:
    scheduler = DrafterScheduler()
    scheduler._replica_idle_worker_groups = {0: ("0",)}
    monkeypatch.setattr(
        "verl_speco.trainer.scheduler.drafter_scheduler.time.time", lambda: 105.0
    )
    metrics = scheduler.on_worker_event(
        {
            "event_type": "worker_idle",
            "worker_id": "0",
            "replica_rank": 0,
            "memory_released": True,
            "event_ts": 100.4,
            "request_completed_ts": 100.0,
            "event_emit_ts": 100.2,
            "lease_released_ts": 100.4,
            "bus_received_ts": 100.3,
        }
    )
    assert metrics["bubble/replica_0_request_to_lease_release_s"] == pytest.approx(0.4)
    assert metrics["bubble/replica_0_emit_to_driver_s"] == pytest.approx(4.8)
    assert scheduler._replica_idle_started_at[0] == 100.4


def test_first_step_extra_is_reserved_once_not_for_every_batch() -> None:
    from verl_speco.trainer.scheduler.schedule_types import AvailableTrainingResources

    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler._idle_worker_hot_prewarmed_groups.add(("0", "1"))
    scheduler._idle_worker_batch_estimate_sec = 0.3
    scheduler._idle_worker_batch_samples_sec.append(0.3)
    scheduler._idle_worker_first_step_samples_sec.append(0.9)
    config = replace(_idle_config(), idle_worker_initial_batch_estimate_sec=None)
    plan = scheduler.plan_training(
        replace(_context(), data_status=_status("group", batches=10)),
        config,
        resources=AvailableTrainingResources(
            True,
            "training_group_ready",
            worker_ids=("0", "1"),
            minimum_idle_window_sec=1.9,
            idle_window_source="historical_observed",
        ),
        require_interval=False,
    )
    assert plan.launch and plan.max_batches == 3
    assert plan.idle_first_step_extra_sec == pytest.approx(0.6)
    assert plan.to_worker_payload()["idle_first_step_extra_sec"] == pytest.approx(0.6)


def test_early_reclaim_does_not_collapse_next_window_capacity() -> None:
    scheduler = DrafterScheduler()
    scheduler._idle_worker_dynamic_batch_cap = 20
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=5,
        max_batches=9,
        publish_after_success=False,
        target_worker_ids=("0", "1"),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        elapsed_sec=0.6,
        reason="trained",
        metrics={},
        raw_results=[{"worker_id": "0", "stop_reason": "reclaim_requested"}],
    )
    scheduler._record_idle_dynamic_batch_cap(plan, outcome)
    assert scheduler._effective_idle_dynamic_batch_cap(_idle_config(), ("0", "1")) == 10


def test_dynamic_caps_are_group_local_and_recover_from_window_limited_success() -> None:
    scheduler = DrafterScheduler()
    scheduler._idle_group_batch_caps[("0", "1")] = 2
    scheduler._idle_group_batch_caps[("2", "3")] = 8
    config = replace(_idle_config(), train_batches_per_trigger=20)
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=5,
        max_batches=1,
        publish_after_success=False,
        target_worker_ids=("0", "1"),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        elapsed_sec=0.6,
        reason="trained",
        metrics={},
        raw_results=[{"worker_id": "0", "stop_reason": "max_batches_reached"}],
    )
    scheduler._record_idle_dynamic_batch_cap(plan, outcome)
    assert scheduler._effective_idle_dynamic_batch_cap(config, ("0", "1")) == 4
    assert scheduler._effective_idle_dynamic_batch_cap(config, ("2", "3")) == 8
    assert scheduler._effective_idle_dynamic_batch_cap(config, ("4", "5")) == 20
    scheduler._record_idle_dynamic_batch_cap(
        plan,
        replace(
            outcome, raw_results=[{"worker_id": "0", "stop_reason": "deadline_reached"}]
        ),
    )
    assert scheduler._effective_idle_dynamic_batch_cap(config, ("0", "1")) == 1
    scheduler._record_idle_dynamic_batch_cap(
        replace(plan, reason="quota_forced_completion_ready"), outcome
    )
    assert scheduler._effective_idle_dynamic_batch_cap(config, ("0", "1")) == 1


def test_first_step_budget_is_recent_group_and_restart_mode_local() -> None:
    from verl_speco.trainer.scheduler import AvailableTrainingResources

    scheduler = DrafterScheduler()
    config = replace(_idle_config(), idle_worker_initial_batch_estimate_sec=0.1)
    group = ("0", "1")
    scheduler._idle_worker_hot_prewarmed_groups.add(group)
    scheduler._idle_first_step_by_mode[(group, "cold")] = deque([4.0], maxlen=4)
    scheduler._idle_first_step_by_mode[(group, "hot")] = deque([0.9], maxlen=4)
    scheduler._idle_first_step_by_mode[(group, "continuation")] = deque(
        [0.15], maxlen=4
    )
    assert scheduler._effective_idle_first_step_extra_sec(
        config, group
    ) == pytest.approx(0.8)
    assert scheduler._effective_idle_first_step_extra_sec(
        config, group, continuation=True
    ) == pytest.approx(0.05)
    assert scheduler._effective_idle_first_step_extra_sec(config, ("2", "3")) == 0
    scheduler._idle_first_step_by_mode[(group, "hot")].extend([0.2] * 4)
    assert scheduler._effective_idle_first_step_extra_sec(
        config, group
    ) == pytest.approx(0.1)
    plan = scheduler.plan_training(
        replace(_context(), data_status=_status("group", batches=10)),
        config,
        resources=AvailableTrainingResources(
            True,
            "training_group_ready",
            worker_ids=group,
            minimum_idle_window_sec=0.8,
            idle_window_source="historical_observed",
        ),
        require_interval=False,
        idle_continuation=True,
    )
    assert plan.launch and plan.idle_continuation
    assert plan.idle_first_step_extra_sec == pytest.approx(0.05)
    assert plan.idle_startup_reserve_sec == pytest.approx(0.05)
    assert plan.to_worker_payload()["idle_continuation"] is True
    # Same-window continuation is not hot after training resources were parked.
    scheduler._idle_worker_hot_prewarmed_groups.remove(group)
    assert scheduler._effective_idle_first_step_extra_sec(
        config, group, continuation=True
    ) == pytest.approx(3.9)


def test_worker_parking_is_not_double_charged_to_tail() -> None:
    scheduler = DrafterScheduler()
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=2.5,
        reason="trained",
        metrics={
            "bubble/train_first_batch_started": 1,
            "timing_s/drafter_worker_cleanup": 2.1,
            "timing_s/drafter_worker_memory_park": 2.0,
        },
    )
    scheduler.record_idle_training_outcome(outcome)
    assert scheduler._effective_idle_tail_reserve_sec(_idle_config()) == pytest.approx(
        2.1
    )


def test_parked_writer_retains_state_without_zero_startup_budget() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=False,
        target_worker_ids=("0",),
    )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        reason="trained",
        metrics={"bubble/training_residency_retained": 0},
    )
    scheduler._record_training_outcome(plan, outcome)
    assert scheduler._idle_worker_writer_state_version == 10
    assert ("0",) not in scheduler._idle_worker_hot_prewarmed_groups


def test_quota_completing_plan_uses_final_tail_reserve() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._idle_optimizer_initialized_groups.add(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    scheduler._training_quota_debt_steps = 3
    scheduler._idle_worker_partial_tail_samples_sec.append(0.5)
    scheduler._idle_worker_final_tail_samples_sec.append(2.0)
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=time.time() + 10.0,
            event_ts=time.time(),
        )
    )
    config = replace(
        _auto_idle_config(1),
        training_quota_enable=True,
        train_batches_per_trigger=3,
        gradient_accumulation_steps=1,
    )
    context = replace(
        _context(),
        global_step=3,
        data_status=replace(
            _status("0", batches=4),
            trainable_batches=4,
            target_version=3,
        ),
    )

    plan = scheduler.prepare_idle_worker_training_plan(context, config)

    assert plan.launch
    assert plan.max_batches == 3
    assert plan.publish_after_success
    assert plan.idle_tail_reserve_sec == pytest.approx(2.0)
    assert plan.idle_park_after_training


def test_runtime_deadline_idle_plan_keeps_hard_deadline() -> None:
    scheduler = _scheduler_with_statuses(("0",))
    scheduler._metadata_idle_training_groups = (("0",),)
    deadline = time.time() + 20.0
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline,
            event_ts=time.time(),
        )
    )
    config = replace(
        _auto_idle_config(1),
        train_batches_per_trigger=10,
        gradient_accumulation_steps=1,
    )
    context = replace(
        _context(),
        global_step=3,
        data_status=replace(
            _status("0", batches=4),
            trainable_batches=4,
            trainable_valid_tokens=4096,
            target_version=3,
        ),
    )

    plan = scheduler.prepare_idle_worker_training_plan(context, config)

    assert plan.launch
    assert plan.idle_window_source == "runtime_deadline"
    assert plan.max_batches == 10
    assert plan.deadline_ts is not None
