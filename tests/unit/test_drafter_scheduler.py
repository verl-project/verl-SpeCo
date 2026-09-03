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

import pytest

from verl_speco.trainer.scheduler import (
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
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
    assert plan.min_sample_step == 5
    assert plan.max_sample_step == 5
    assert plan.data_filter_reason == "current_step_only"
    assert plan.publish_after_success
    assert plan.to_worker_payload()["execution_strategy"] == "sync"
    assert plan.to_worker_payload()["min_sample_step"] == 5
    assert plan.to_worker_payload()["max_sample_step"] == 5
    assert plan.metrics() == {
        "drafter/scheduler_used": 1,
        "drafter/schedule_launch": 1,
        "drafter/schedule_interval_matched": 1,
        "drafter/schedule_strategy": 0,
        "drafter/schedule_reason": 10,
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
            "no_current_step_oldlogprob_samples",
        ),
        (
            _context(samples=0),
            DrafterScheduleConfig(training_interval_steps=5),
            "no_current_step_samples",
        ),
    ],
)
def test_sync_plan_preserves_skip_conditions(context, config, reason) -> None:
    plan = DrafterScheduler().plan_training(context, config)
    assert not plan.launch
    assert plan.reason == reason


def test_sync_plan_preserves_data_buffer_fallback() -> None:
    plan = DrafterScheduler().plan_training(
        _context(step=5, samples=0, trainable_batches=9),
        DrafterScheduleConfig(
            training_interval_steps=5,
            use_data_buffer=True,
            train_batches_per_trigger=9,
            sample_last_n_steps=2,
        ),
    )
    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.max_batches == 9
    assert plan.min_sample_step == 3
    assert plan.max_sample_step == 5
    assert plan.data_filter_reason == "recent_buffer_window"
    assert plan.publish_after_success


def test_sync_plan_forces_current_step_when_worker_requires_same_step_data() -> None:
    context = _context(step=7, samples=1, trainable_batches=2)
    context = DrafterScheduleContext(
        global_step=context.global_step,
        training_mode=context.training_mode,
        collected_samples_this_step=context.collected_samples_this_step,
        oldlogprob_collection_requested=context.oldlogprob_collection_requested,
        data_status=TrainingDataStatus(
            **{
                **context.data_status.__dict__,
                "same_step_data_required": True,
            }
        ),
    )

    plan = DrafterScheduler().plan_training(
        context,
        DrafterScheduleConfig(
            training_interval_steps=1,
            use_data_buffer=True,
            sample_last_n_steps=4,
        ),
    )

    assert plan.launch
    assert plan.min_sample_step == 7
    assert plan.max_sample_step == 7
    assert plan.data_filter_reason == "same_step_required"


def test_oldlogprob_collection_does_not_fallback_to_old_buffer_data() -> None:
    plan = DrafterScheduler().plan_training(
        _context(samples=0, oldlogprob_requested=True, trainable_batches=4),
        DrafterScheduleConfig(
            training_interval_steps=5,
            use_data_buffer=True,
            train_batches_per_trigger=4,
        ),
    )

    assert not plan.launch
    assert plan.reason == "no_current_step_oldlogprob_samples"


def test_training_without_data_buffer_requires_current_step_samples() -> None:
    plan = DrafterScheduler().plan_training(
        _context(samples=0, trainable_batches=4),
        DrafterScheduleConfig(
            training_interval_steps=5,
            use_data_buffer=False,
            train_batches_per_trigger=4,
        ),
    )

    assert not plan.launch
    assert plan.reason == "no_current_step_samples"


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
