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

from verl_speco.trainer.scheduler import (
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    TrainingDataStatus,
)
from verl_speco.trainer.scheduler.training_budget import (
    AdaptiveTrainingBudgetPolicy,
    adaptive_drafter_training_steps,
    adaptive_warmup_window_end,
)


# The DFlash online run: collect and train every 5th main step, 20 optimizer
# steps per session, 100 main steps in total.  The adaptive schedule has to
# work on top of these values without editing them.
_DFLASH_TRAINING_CONFIG = {
    "collect_interval_steps": 5,
    "training_interval_steps": 5,
    "step": 20,
    "adaptive_schedule": {
        "enable": True,
        "train_steps_by_acceptance": True,
        "train_every_step_in_warmup": True,
        "warmup_ratio": 0.1,
        "warmup_max_steps": 20,
        "max_train_steps": 25,
        "min_train_steps": 5,
        "low_acceptance": 2.5,
        "high_acceptance": 4.0,
    },
}
# 0.1 * 100 = 10, so the warmup window is [0, 10) and covers main steps 1..9.
_TOTAL_TRAINING_STEPS = 100


def _adaptive_config(**overrides) -> DrafterScheduleConfig:
    training_config = dict(_DFLASH_TRAINING_CONFIG)
    adaptive = dict(training_config["adaptive_schedule"])
    for key, value in overrides.items():
        if key == "adaptive_schedule":
            adaptive.update(value)
        else:
            training_config[key] = value
    training_config["adaptive_schedule"] = adaptive
    return DrafterScheduleConfig.from_mapping(training_config)


def _collection_context(
    step,
    *,
    total_training_steps=_TOTAL_TRAINING_STEPS,
    source=DrafterCollectionSource.OLD_LOGPROB,
) -> DrafterCollectionContext:
    return DrafterCollectionContext(
        global_step=step,
        source=source,
        drafter_enabled=True,
        source_enabled=True,
        require_training_interval=source is DrafterCollectionSource.OLD_LOGPROB,
        total_training_steps=total_training_steps,
    )


def _schedule_context(
    step,
    *,
    samples=16,
    acceptance=3.2,
    total_training_steps=_TOTAL_TRAINING_STEPS,
) -> DrafterScheduleContext:
    return DrafterScheduleContext(
        global_step=step,
        training_mode="online",
        collected_samples_this_step=samples,
        oldlogprob_collection_requested=True,
        data_status=TrainingDataStatus(
            current_step=step,
            current_step_samples=samples,
            buffer_samples=samples,
            trainable_samples=samples,
            trainable_batches=samples // 4,
            batch_size_per_gpu=4,
            partial_batch_available=False,
            oldest_sample_step=step,
            newest_sample_step=step,
            same_step_data_required=False,
            target_version=step,
        ),
        pending_training_count=0,
        total_training_steps=total_training_steps,
        prev_acceptance_length=acceptance,
    )


def test_warmup_collects_on_a_non_interval_step() -> None:
    scheduler = DrafterScheduler()
    config = _adaptive_config()

    plan = scheduler.plan_collection(_collection_context(1), config)

    assert plan.collect
    assert plan.reason == "warmup_collection_enabled"
    assert plan.collect_interval_matched
    assert plan.training_interval_matched
    assert plan.metrics()["drafter/collection_plan_reason"] == 8


def test_warmup_trains_on_a_non_interval_step_with_the_adaptive_budget() -> None:
    scheduler = DrafterScheduler()
    config = _adaptive_config()

    plan = scheduler.plan_training(_schedule_context(1), config)

    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.warmup_active
    assert plan.interval_matched
    assert plan.max_batches > config.train_batches_per_trigger
    assert plan.min_sample_step == 1
    assert plan.max_sample_step == 1
    assert plan.data_filter_reason == "current_step_only"


def test_warmup_window_covers_the_configured_horizon_fraction() -> None:
    scheduler = DrafterScheduler()
    config = _adaptive_config()

    assert scheduler.adaptive_warmup_active(_schedule_context(9), config)
    assert not scheduler.adaptive_warmup_active(_schedule_context(10), config)

    # The cap shortens the window when the ratio would cover more steps.
    capped = _adaptive_config(
        adaptive_schedule={"warmup_ratio": 0.5, "warmup_max_steps": 3}
    )
    assert scheduler.adaptive_warmup_active(_schedule_context(2), capped)
    assert not scheduler.adaptive_warmup_active(_schedule_context(3), capped)


def test_warmup_stays_inactive_without_a_known_horizon() -> None:
    scheduler = DrafterScheduler()
    config = _adaptive_config()
    context = _collection_context(1, total_training_steps=None)

    plan = scheduler.plan_collection(context, config)

    assert not plan.collect
    assert plan.reason == "interval_not_reached"
    assert not plan.collect_interval_matched
    assert not scheduler.plan_training(
        _schedule_context(1, total_training_steps=None), config
    ).launch


def test_warmup_switch_relaxes_intervals_but_keeps_the_adaptive_budget() -> None:
    """The two adaptive features are independent of each other."""

    scheduler = DrafterScheduler()
    config = _adaptive_config(adaptive_schedule={"train_every_step_in_warmup": False})

    collection = scheduler.plan_collection(_collection_context(1), config)
    assert not collection.collect
    assert collection.reason == "interval_not_reached"

    interval_step = scheduler.plan_training(_schedule_context(5), config)
    assert interval_step.launch
    assert not interval_step.warmup_active
    assert interval_step.max_batches > config.train_batches_per_trigger


def test_acceptance_switch_keeps_the_warmup_but_uses_the_configured_steps() -> None:
    """The two adaptive features are independent of each other."""

    scheduler = DrafterScheduler()
    config = _adaptive_config(adaptive_schedule={"train_steps_by_acceptance": False})

    plan = scheduler.plan_training(_schedule_context(1), config)

    assert plan.launch
    assert plan.warmup_active
    assert plan.interval_matched
    assert plan.max_batches == config.train_batches_per_trigger == 20


def test_disabled_adaptive_schedule_keeps_the_released_decisions() -> None:
    scheduler = DrafterScheduler()
    config = _adaptive_config(adaptive_schedule={"enable": False})

    assert not scheduler.plan_collection(_collection_context(1), config).collect
    assert not scheduler.plan_training(_schedule_context(1), config).launch

    interval_step = scheduler.plan_training(_schedule_context(5), config)
    assert interval_step.launch
    assert not interval_step.warmup_active
    assert interval_step.max_batches == config.train_batches_per_trigger == 20


def test_prepare_training_plan_carries_the_horizon_into_the_launch_decision() -> None:
    """The worker-status refresh must not drop ``total_training_steps``.

    ``prepare_training_plan`` re-plans with the fetched training-data status.
    Rebuilding that context by hand drops the horizon and the acceptance
    length, which silently disables both adaptive features on exactly the
    branch that launches training.
    """

    scheduler = DrafterScheduler()
    config = _adaptive_config()

    warmup_plan = scheduler.prepare_training_plan(_schedule_context(1), config)
    assert warmup_plan.launch
    assert warmup_plan.warmup_active
    assert warmup_plan.max_batches == 25

    interval_plan = scheduler.prepare_training_plan(_schedule_context(10), config)
    assert interval_plan.launch
    assert interval_plan.interval_matched
    assert not interval_plan.warmup_active
    assert interval_plan.max_batches == 24
    assert interval_plan.max_batches > config.train_batches_per_trigger


def test_adaptive_budget_decays_with_progress_and_follows_acceptance() -> None:
    # Base count decays linearly from max_train_steps to min_train_steps.
    assert adaptive_drafter_training_steps(
        global_step=1, total_training_steps=100
    ) == 25
    assert adaptive_drafter_training_steps(
        global_step=99, total_training_steps=100
    ) == 5
    # Halfway through the horizon the base count is exactly 15.
    assert (
        adaptive_drafter_training_steps(
            global_step=50, total_training_steps=100, prev_acceptance_length=None
        )
        == 15
    )
    # Poor acceptance trains more, good acceptance trains less.
    assert (
        adaptive_drafter_training_steps(
            global_step=50, total_training_steps=100, prev_acceptance_length=2.0
        )
        == 19
    )
    assert (
        adaptive_drafter_training_steps(
            global_step=50, total_training_steps=100, prev_acceptance_length=5.0
        )
        == 12
    )
    # The result stays inside [min_train_steps, max_train_steps].
    assert (
        adaptive_drafter_training_steps(
            global_step=1, total_training_steps=100, prev_acceptance_length=2.0
        )
        == 25
    )
    assert (
        adaptive_drafter_training_steps(
            global_step=99, total_training_steps=100, prev_acceptance_length=5.0
        )
        == 5
    )


def test_adaptive_budget_falls_back_to_the_configured_step_count() -> None:
    policy = AdaptiveTrainingBudgetPolicy()
    config = _adaptive_config()

    budget = policy.make_budget(
        _schedule_context(1, total_training_steps=None), config
    )
    assert budget.max_batches == config.train_batches_per_trigger
    assert budget.reason == "sync_budget_ready"

    adaptive_budget = policy.make_budget(_schedule_context(1), config)
    assert adaptive_budget.reason == "adaptive_budget_ready"
    assert adaptive_budget.max_batches == 25


def test_acceptance_switch_falls_back_to_the_configured_step_count() -> None:
    policy = AdaptiveTrainingBudgetPolicy()
    config = _adaptive_config(adaptive_schedule={"train_steps_by_acceptance": False})

    budget = policy.make_budget(_schedule_context(1), config)

    assert budget.max_batches == config.train_batches_per_trigger
    assert budget.reason == "sync_budget_ready"


def test_warmup_window_end_degrades_on_degenerate_input() -> None:
    assert (
        adaptive_warmup_window_end(
            total_training_steps=None, warmup_ratio=0.1, warmup_max_steps=20
        )
        == 0
    )
    assert (
        adaptive_warmup_window_end(
            total_training_steps=100, warmup_ratio=0.0, warmup_max_steps=20
        )
        == 0
    )
    assert (
        adaptive_warmup_window_end(
            total_training_steps=100, warmup_ratio=0.1, warmup_max_steps=0
        )
        == 0
    )
    assert (
        adaptive_warmup_window_end(
            total_training_steps=100, warmup_ratio=0.1, warmup_max_steps=20
        )
        == 10
    )
    assert (
        adaptive_warmup_window_end(
            total_training_steps=1000, warmup_ratio=0.5, warmup_max_steps=20
        )
        == 20
    )
