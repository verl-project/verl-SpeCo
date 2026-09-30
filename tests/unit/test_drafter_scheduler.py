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
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl_speco.trainer.scheduler import (
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    DrafterTrainingDataSource,
    ProducerAction,
    TrainingDataStatus,
    TrainingPlan,
    step_matches_interval,
)


def _context(
    *,
    step=5,
    mode="online",
    samples=1,
    oldlogprob_requested=False,
    trainable_batches=None,
) -> DrafterScheduleContext:
    if trainable_batches is None:
        trainable_batches = 1 if samples > 0 else 0
    return DrafterScheduleContext(
        global_step=step,
        training_mode=mode,
        collected_samples_this_step=samples,
        oldlogprob_collection_requested=oldlogprob_requested,
        data_status=TrainingDataStatus(
            current_step=step,
            current_step_samples=samples,
            buffer_samples=samples,
            trainable_samples=trainable_batches,
            trainable_batches=trainable_batches,
            batch_size_per_gpu=1,
            partial_batch_available=False,
            oldest_sample_step=step if trainable_batches else None,
            newest_sample_step=step if trainable_batches else None,
            same_step_data_required=False,
        ),
    )


def test_step_interval_matches_released_semantics() -> None:
    assert step_matches_interval(6, 3)
    assert not step_matches_interval(0, 3)
    assert not step_matches_interval(None, 3)
    assert not step_matches_interval(6, 0)
    assert not step_matches_interval("bad-step", 3)
    assert not step_matches_interval(6, "bad-interval")


def test_legacy_config_maps_released_sync_values() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 5,
            "publish_interval_steps": 10,
            "use_data_buffer": True,
            "step": 7,
        }
    )
    assert config == DrafterScheduleConfig(
        collect_interval_steps=2,
        training_interval_steps=5,
        publish_interval_steps=10,
        use_data_buffer=True,
        train_batches_per_trigger=7,
        collection_sample_rate=1.0,
        max_collect_samples_per_replica=16,
        max_collect_tokens_per_replica=None,
        hidden_window_mode="front",
        hidden_window_tokens_per_sample=512,
        hidden_window_min_rows=512,
    )


def test_sglang_collection_plan_contains_static_budget_and_metrics() -> None:
    plan = DrafterScheduler().plan_collection(
        DrafterCollectionContext(
            global_step=6,
            source=DrafterCollectionSource.SGLANG,
        ),
        DrafterScheduleConfig(
            collect_interval_steps=2,
            collection_sample_rate=0.5,
            max_collect_samples_per_replica=4,
            max_collect_tokens_per_replica=2048,
            hidden_window_mode="random",
            hidden_window_tokens_per_sample=256,
            hidden_window_min_rows=64,
        ),
    )
    assert plan.collect
    assert plan.reason == "collection_enabled"
    assert plan.sample_rate == 0.5
    assert plan.max_samples_per_replica == 4
    assert plan.max_tokens_per_replica == 2048
    assert plan.metrics()["drafter/collection_plan_source"] == 1
    assert plan.metrics()["drafter/collection_plan_reason"] == 7
    assert plan.producer_action is ProducerAction.RUN
    assert plan.max_new_samples is None


def test_oldlogprob_collection_plan_preserves_training_interval_requirement() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        collect_interval_steps=2,
        training_interval_steps=4,
    )
    plan = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=6,
            source=DrafterCollectionSource.OLD_LOGPROB,
            require_training_interval=True,
        ),
        config,
    )
    assert not plan.collect
    assert plan.collect_interval_matched
    assert not plan.training_interval_matched
    assert plan.reason == "training_interval_not_reached"


def test_bubble_collection_is_single_flight_until_quota_completes() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        collect_interval_steps=2,
        training_interval_steps=4,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        training_quota_enable=True,
        training_quota_target_steps=20,
    )
    first = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=2,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )

    scheduler.record_collection_outcome(
        first,
        SimpleNamespace(collected=True),
        config,
    )
    assert scheduler._training_quota_debt_steps == 20
    assert scheduler._training_quota_data_version == 2

    same_step = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=2,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )
    next_interval = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=4,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )

    assert same_step.collect
    assert not next_interval.collect
    assert next_interval.reason == "training_quota_incomplete"
    assert next_interval.metrics()["drafter/collection_plan_reason"] == 10

    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_oldest_cycle_step = None
    scheduler._training_quota_data_version = None
    awaiting_publish = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=2,
            source=DrafterCollectionSource.OLD_LOGPROB,
        ),
        config,
    )
    assert not awaiting_publish.collect
    assert awaiting_publish.reason == "training_quota_incomplete"

    scheduler.record_training_quota_publish_completed()
    next_cycle = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=4,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )
    scheduler.record_collection_outcome(
        next_cycle,
        SimpleNamespace(collected=True),
        config,
    )

    assert next_cycle.collect
    assert scheduler._training_quota_last_cycle_step == 8
    assert scheduler._training_quota_debt_steps == 20


def test_adaptive_quota_keeps_full_target_but_skips_healthy_refresh() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        collect_interval_steps=2,
        training_interval_steps=4,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_trigger_mode="adaptive",
        training_quota_acceptance_drop_ratio=0.03,
        training_quota_min_refresh_interval_steps=2,
        training_quota_max_refresh_interval_steps=10,
    )
    first = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=2,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )
    scheduler.record_collection_outcome(
        first,
        SimpleNamespace(collected=True),
        config,
    )

    assert first.collect
    assert scheduler._training_quota_debt_steps == 20

    scheduler._training_quota_debt_steps = 0
    scheduler._training_quota_data_version = None
    scheduler.record_training_quota_publish_completed(global_step=4)
    scheduler.record_step_metrics(
        {"drafter/spec_decode/mean_acceptance_length": 3.5},
        config,
        global_step=5,
    )
    healthy = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=6,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )

    assert not healthy.collect
    assert healthy.reason == "quality_refresh_not_due"

    scheduler.record_step_metrics(
        {"drafter/spec_decode/mean_acceptance_length": 3.3},
        config,
        global_step=7,
    )
    degraded = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=8,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )

    assert degraded.collect
    assert degraded.reason == "collection_enabled"


def test_adaptive_quota_forces_refresh_at_maximum_age() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        collect_interval_steps=2,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        training_quota_enable=True,
        training_quota_trigger_mode="adaptive",
        training_quota_min_refresh_interval_steps=2,
        training_quota_max_refresh_interval_steps=6,
    )
    scheduler._quality_last_publish_step = 4
    scheduler._quality_acceptance_baseline = 3.5
    scheduler._quality_latest_acceptance = 3.5

    before_max_age = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=8,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )
    at_max_age = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=10,
            source=DrafterCollectionSource.SGLANG,
        ),
        config,
    )

    assert not before_max_age.collect
    assert before_max_age.reason == "quality_refresh_not_due"
    assert at_max_age.collect


@pytest.mark.parametrize(
    ("context", "config", "reason"),
    [
        (
            DrafterCollectionContext(
                global_step=2,
                source=DrafterCollectionSource.SGLANG,
                drafter_enabled=False,
            ),
            DrafterScheduleConfig(collect_interval_steps=2),
            "drafter_disabled",
        ),
        (
            DrafterCollectionContext(
                global_step=2,
                source=DrafterCollectionSource.SGLANG,
                source_enabled=False,
            ),
            DrafterScheduleConfig(collect_interval_steps=2),
            "source_disabled",
        ),
        (
            DrafterCollectionContext(
                global_step=2,
                source=DrafterCollectionSource.SGLANG,
                validation=True,
            ),
            DrafterScheduleConfig(collect_interval_steps=2),
            "validation",
        ),
        (
            DrafterCollectionContext(
                global_step=3,
                source=DrafterCollectionSource.SGLANG,
            ),
            DrafterScheduleConfig(collect_interval_steps=2),
            "interval_not_reached",
        ),
        (
            DrafterCollectionContext(
                global_step=2,
                source=DrafterCollectionSource.SGLANG,
            ),
            DrafterScheduleConfig(
                collect_interval_steps=2,
                collection_sample_rate=0,
            ),
            "sample_rate_zero",
        ),
    ],
)
def test_collection_plan_skip_reasons(context, config, reason) -> None:
    plan = DrafterScheduler().plan_collection(context, config)
    assert not plan.collect
    assert plan.reason == reason


def test_sync_plan_launches_for_current_step_samples() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(training_interval_steps=5)

    plan = scheduler.plan_training(_context(), config)

    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.interval_matched
    assert plan.execution_strategy is DrafterExecutionStrategy.SYNC
    assert plan.source_global_step == 5
    assert plan.publish_after_success
    assert plan.data_source is DrafterTrainingDataSource.LOCAL_BUFFER
    assert plan.required_samples is None
    assert plan.to_worker_payload()["execution_strategy"] == "sync"
    assert plan.metrics() == {
        "drafter/scheduler_used": 1,
        "drafter/schedule_launch": 1,
        "drafter/schedule_interval_matched": 1,
        "drafter/schedule_strategy": 0,
        "drafter/schedule_reason": 10,
        "drafter/schedule_max_batches": 100,
        "drafter/schedule_publish_after_success": 1,
        "drafter/schedule_min_batches": 1,
        "drafter/schedule_require_full_batch": 0,
        "drafter/schedule_sample_last_n_steps": 2,
        "drafter/schedule_source_global_step": 5,
    }


@pytest.mark.parametrize(
    ("context", "config", "reason"),
    [
        (_context(mode="collect_only"), DrafterScheduleConfig(), "collect_only"),
        (
            _context(step=4),
            DrafterScheduleConfig(training_interval_steps=5),
            "interval_not_reached",
        ),
        (
            _context(samples=0, oldlogprob_requested=True),
            DrafterScheduleConfig(training_interval_steps=5, use_data_buffer=True),
            "no_trainable_batch",
        ),
        (
            _context(samples=0),
            DrafterScheduleConfig(training_interval_steps=5),
            "no_trainable_batch",
        ),
    ],
)
def test_sync_plan_preserves_skip_conditions(context, config, reason) -> None:
    plan = DrafterScheduler().plan_training(context, config)
    assert not plan.launch
    assert plan.reason == reason


def test_sync_plan_preserves_data_buffer_fallback() -> None:
    plan = DrafterScheduler().plan_training(
        _context(samples=0, trainable_batches=9),
        DrafterScheduleConfig(
            training_interval_steps=5,
            use_data_buffer=True,
            train_batches_per_trigger=9,
        ),
    )
    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.max_batches == 9
    assert plan.publish_after_success


def test_sync_plan_uses_configured_steps_when_pool_has_fewer_batches() -> None:
    plan = DrafterScheduler().plan_training(
        _context(trainable_batches=4),
        DrafterScheduleConfig(
            training_interval_steps=1,
            train_batches_per_trigger=10,
        ),
    )

    assert plan.launch
    assert plan.max_batches == 10


def test_sync_plan_carries_publish_decision_to_worker() -> None:
    plan = DrafterScheduler().plan_training(
        _context(step=6),
        DrafterScheduleConfig(
            training_interval_steps=3,
            publish_interval_steps=4,
        ),
    )
    assert plan.launch
    assert not plan.publish_after_success


def test_publish_plan_preserves_released_interval_behavior() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(publish_interval_steps=4)

    assert not scheduler.plan_publish(
        global_step=6, drafter_trained=True, config=config
    ).publish
    assert scheduler.plan_publish(
        global_step=8, drafter_trained=True, config=config
    ).publish
    assert not scheduler.plan_publish(
        global_step=8, drafter_trained=False, config=config
    ).publish


def test_invalid_publish_interval_still_raises() -> None:
    with pytest.raises(ValueError):
        DrafterScheduler().plan_publish(
            global_step=5,
            drafter_trained=True,
            config=DrafterScheduleConfig(publish_interval_steps="bad"),
        )


def test_publish_plan_honors_training_plan_publish_decision() -> None:
    training_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=8,
        max_batches=1,
        publish_after_success=False,
    )
    plan = DrafterScheduler().plan_publish(
        global_step=8,
        drafter_trained=True,
        config=DrafterScheduleConfig(publish_interval_steps=1),
        training_plan=training_plan,
    )

    assert not plan.publish
    assert plan.reason == "training_plan_publish_disabled"


def test_idle_worker_publish_is_asynchronous() -> None:
    training_plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=8,
        max_batches=1,
        publish_after_success=True,
    )

    plan = DrafterScheduler().plan_publish(
        global_step=8,
        drafter_trained=True,
        config=DrafterScheduleConfig(
            publish_interval_steps=1,
            publish_async=False,
        ),
        training_plan=training_plan,
    )

    assert plan.publish
    assert plan.asynchronous


def test_budget_smaller_than_minimum_does_not_launch() -> None:
    plan = DrafterScheduler().plan_training(
        _context(trainable_batches=5),
        DrafterScheduleConfig(
            training_interval_steps=1,
            train_batches_per_trigger=2,
            min_trainable_batches=3,
        ),
    )

    assert not plan.launch
    assert plan.reason == "insufficient_training_budget"
    assert plan.max_batches == 2
    assert plan.min_batches == 3


def test_inconsistent_target_versions_do_not_launch() -> None:
    context = _context(trainable_batches=2)
    context = DrafterScheduleContext(
        global_step=context.global_step,
        training_mode=context.training_mode,
        collected_samples_this_step=context.collected_samples_this_step,
        oldlogprob_collection_requested=context.oldlogprob_collection_requested,
        data_status=TrainingDataStatus(
            **{
                **context.data_status.__dict__,
                "target_version_consistent": False,
            }
        ),
    )

    plan = DrafterScheduler().plan_training(
        context, DrafterScheduleConfig(training_interval_steps=1)
    )

    assert not plan.launch
    assert plan.reason == "inconsistent_target_version"


def test_inconsistent_target_versions_do_not_block_logits_training() -> None:
    context = _context(trainable_batches=2)
    context = DrafterScheduleContext(
        global_step=context.global_step,
        training_mode=context.training_mode,
        collected_samples_this_step=context.collected_samples_this_step,
        oldlogprob_collection_requested=context.oldlogprob_collection_requested,
        data_status=TrainingDataStatus(
            **{
                **context.data_status.__dict__,
                "target_version_consistent": False,
            }
        ),
    )

    plan = DrafterScheduler().plan_training(
        context,
        DrafterScheduleConfig(training_interval_steps=1, use_logits=True),
    )

    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.required_target_version is None
