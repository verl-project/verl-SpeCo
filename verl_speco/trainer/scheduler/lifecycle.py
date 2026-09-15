# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Typed lifecycle Facade contracts for drafter scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from verl_speco.trainer.scheduler.drafter_runtime_state import DrafterRuntimeState
from verl_speco.trainer.scheduler.publish_strategy import PublishOutcome
from verl_speco.trainer.scheduler.schedule_types import (
    DrafterScheduleConfig,
    DrafterScheduleContext,
    PublishPlan,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.training_outcome import TrainingOutcome


@dataclass(frozen=True)
class BeforeActorUpdateContext:
    schedule_context: DrafterScheduleContext
    config: DrafterScheduleConfig


@dataclass(frozen=True)
class AfterActorUpdateContext:
    training_plan: TrainingPlan
    runtime_state: DrafterRuntimeState


@dataclass(frozen=True)
class AfterWeightUpdateContext:
    global_step: object
    drafter_trained: bool
    config: DrafterScheduleConfig
    training_plan: TrainingPlan | None = None


@dataclass(frozen=True)
class SchedulerEventOutcome:
    training_plan: TrainingPlan | None = None
    training_execution: TrainingOutcome | None = None
    publish_plan: PublishPlan | None = None
    publish_outcome: PublishOutcome | None = None
    metrics: dict[str, float | int] | None = None
