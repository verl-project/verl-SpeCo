# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training trigger policies for drafter scheduling."""

from __future__ import annotations

from typing import Protocol

from verl_speco.trainer.scheduler.schedule_types import (
    DrafterScheduleConfig,
    DrafterScheduleContext,
    TriggerDecision,
)


class TrainingTriggerPolicy(Protocol):
    def should_train(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        interval_matched: bool,
    ) -> TriggerDecision: ...


class IntervalAndBufferTrigger:
    """Require interval readiness and enough worker-reported trainable data."""

    def should_train(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        interval_matched: bool,
    ) -> TriggerDecision:
        if context.training_mode == "collect_only":
            return TriggerDecision(False, "collect_only")
        if context.pending_training_count > 0:
            return TriggerDecision(False, "pending_training")
        if not interval_matched:
            return TriggerDecision(False, "interval_not_reached")
        if context.collected_samples_this_step <= 0:
            if context.oldlogprob_collection_requested:
                return TriggerDecision(False, "no_current_step_oldlogprob_samples")
            if not config.use_data_buffer:
                return TriggerDecision(False, "no_current_step_samples")
        status = context.data_status
        if status is None:
            return TriggerDecision(False, "missing_data_status")
        if not config.use_logits and not status.target_version_consistent:
            return TriggerDecision(False, "inconsistent_target_version")
        if not status.data_version_consistent:
            return TriggerDecision(False, "inconsistent_data_version")
        if status.trainable_batches < max(config.min_trainable_batches, 1):
            return TriggerDecision(False, "no_trainable_batch")
        return TriggerDecision(True, "training_ready")
