# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training budget policies for drafter scheduling."""

from __future__ import annotations

from typing import Protocol

from verl_speco.trainer.scheduler.schedule_types import (
    DrafterScheduleConfig,
    DrafterScheduleContext,
    TrainingBudget,
)

_DEFAULT_ACCEPTANCE_FACTOR_MAX = 1.25
_DEFAULT_ACCEPTANCE_FACTOR_MIN = 0.8


def _as_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (str, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def adaptive_warmup_window_end(
    *,
    total_training_steps: int | None,
    warmup_ratio: float = 0.1,
    warmup_max_steps: int = 20,
) -> int:
    """Return the main-training step at which the warmup window ends.

    The window covers the first ``min(warmup_ratio * total, warmup_max_steps)``
    main steps. Inside it the training trigger fires on every main step.
    A missing horizon, or a non-positive ratio or cap, disables the window
    entirely (end = 0), restoring interval-gated triggering.
    """

    try:
        total = int(total_training_steps) if total_training_steps is not None else 0
        ratio = float(warmup_ratio)
        cap = int(warmup_max_steps)
    except (TypeError, ValueError):
        return 0
    if total <= 0 or ratio <= 0.0 or cap <= 0:
        return 0
    return int(min(round(ratio * total), cap))


def adaptive_drafter_training_steps(
    *,
    global_step: object,
    total_training_steps: int,
    prev_acceptance_length: float | None = None,
    max_train_steps: int = 25,
    min_train_steps: int = 5,
    low_acceptance: float = 2.5,
    high_acceptance: float = 4.0,
) -> int:
    """Return the adaptive per-session drafter optimizer-step count.

    Ported from the validated v0.1.0 ``_get_adaptive_drafter_training_steps``:
    a base count that decays linearly from ``max_train_steps`` to
    ``min_train_steps`` over the main-training horizon, scaled by an
    acceptance-length factor.  Poor acceptance (``<= low_acceptance``) trains
    more; good acceptance (``>= high_acceptance``) trains less; in between the
    factor interpolates linearly.  Degenerate input (missing or non-numeric
    acceptance, inverted bounds) degrades to the neutral 1.0 factor and a
    normalized [min, max] range instead of raising.
    """

    if max_train_steps < min_train_steps:
        max_train_steps, min_train_steps = min_train_steps, max_train_steps

    step = _as_float_or_none(global_step)
    progress = step / max(float(total_training_steps), 1.0) if step is not None else 0.0
    progress = max(0.0, min(1.0, progress))

    base_steps = max_train_steps - (max_train_steps - min_train_steps) * progress

    acceptance = _as_float_or_none(prev_acceptance_length)
    if acceptance is None or high_acceptance <= low_acceptance:
        acceptance_factor = 1.0
    elif acceptance <= low_acceptance:
        # Acceptance length is poor -> train more.
        acceptance_factor = _DEFAULT_ACCEPTANCE_FACTOR_MAX
    elif acceptance >= high_acceptance:
        # Acceptance length is good -> train less.
        acceptance_factor = _DEFAULT_ACCEPTANCE_FACTOR_MIN
    else:
        # Linear interpolation between the poor and good acceptance factors.
        acceptance_factor = _DEFAULT_ACCEPTANCE_FACTOR_MAX - (
            (acceptance - low_acceptance)
            / (high_acceptance - low_acceptance)
            * (_DEFAULT_ACCEPTANCE_FACTOR_MAX - _DEFAULT_ACCEPTANCE_FACTOR_MIN)
        )

    train_steps = base_steps * acceptance_factor
    return int(round(max(min_train_steps, min(max_train_steps, train_steps))))


class TrainingBudgetPolicy(Protocol):
    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget: ...


class SyncTrainingBudgetPolicy:
    """Run the configured number of synchronous optimizer steps.

    Data availability is a launch precondition handled by the trigger.  Once
    launched, online training samples from the eligible worker-local pool on
    every optimizer step, so the number of distinct batches in that pool must
    not silently reduce the configured ``step`` count.
    """

    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget:
        max_batches = max(config.train_batches_per_trigger, 0)
        return TrainingBudget(
            max_batches=max_batches,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            reason="sync_budget_ready" if max_batches > 0 else "no_training_budget",
        )


class AdaptiveTrainingBudgetPolicy:
    """Run the acceptance- and progress-decayed optimizer steps.

    When ``adaptive_schedule.enable`` is false,
    ``adaptive_schedule.train_steps_by_acceptance`` is false, or the
    main-training horizon (``total_training_steps``) is unknown, behavior
    falls back to :class:`SyncTrainingBudgetPolicy` so the configured
    ``step`` count runs unchanged.  Otherwise every launched session runs
    ``adaptive_drafter_training_steps`` optimizer steps, which start near
    ``adaptive_max_train_steps`` early in training and decay toward
    ``adaptive_min_train_steps`` as the main model approaches the end of
    training, scaled up to 1.25x for poor acceptance and down to 0.8x for
    good acceptance.  Session launch frequency is owned by the trigger, not
    by this policy.
    """

    def __init__(self) -> None:
        self._sync_policy = SyncTrainingBudgetPolicy()

    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget:
        total_steps = _as_int_or_none(context.total_training_steps)
        if (
            not config.adaptive_schedule_enabled
            or not config.adaptive_train_steps_by_acceptance
            or total_steps is None
            or total_steps <= 0
        ):
            return self._sync_policy.make_budget(context, config)
        max_batches = adaptive_drafter_training_steps(
            global_step=context.global_step,
            total_training_steps=total_steps,
            prev_acceptance_length=context.prev_acceptance_length,
            max_train_steps=config.adaptive_max_train_steps,
            min_train_steps=config.adaptive_min_train_steps,
            low_acceptance=config.adaptive_low_acceptance,
            high_acceptance=config.adaptive_high_acceptance,
        )
        return TrainingBudget(
            max_batches=max_batches,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            reason="adaptive_budget_ready" if max_batches > 0 else "no_training_budget",
        )
