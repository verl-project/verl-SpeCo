# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Pure frequency decisions, independent of permission to execute training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .adaptive_schedule import AdaptiveScheduleConfig
    from .schedule_types import DrafterScheduleConfig


def step_matches_interval(
    global_step: Any, interval_steps: Any, *, default_interval: int = 1
) -> bool:
    """Match the released speco_step_matches_interval semantics."""
    try:
        interval = int(default_interval if interval_steps is None else interval_steps)
    except (TypeError, ValueError):
        return False
    if interval <= 0 or global_step is None:
        return False
    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return False
    return step > 0 and step % interval == 0


def startup_active(
    step: int, config: AdaptiveScheduleConfig, ended_after: int | None
) -> bool:
    return (
        config.enable
        and 1 <= step < config.warmup_max_steps
        and (ended_after is None or step <= ended_after)
    )


@dataclass(frozen=True)
class TrainingOpportunity:
    due: bool
    collection_due: bool


def training_opportunity(
    step: object,
    config: DrafterScheduleConfig,
    *,
    startup_ended_after: int | None = None,
) -> TrainingOpportunity:
    """Read-only frequency decision shared by collection and training."""
    matched = step_matches_interval(step, config.training_interval_steps)
    collect = step_matches_interval(step, config.collect_interval_steps)
    try:
        startup = startup_active(
            int(cast(Any, step)), config.adaptive_schedule, startup_ended_after
        )
    except (TypeError, ValueError, OverflowError):
        startup = False
    return TrainingOpportunity(
        due=matched or startup,
        collection_due=collect or startup,
    )
