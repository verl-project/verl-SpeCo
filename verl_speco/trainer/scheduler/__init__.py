# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Public facade for the drafter scheduling subsystem.

Code outside this package should import scheduling contracts from here instead
of depending on trigger, budget, or execution implementation modules.
"""

from .drafter_runtime_state import DrafterRuntimeState, DrafterRuntimeStatus
from .drafter_scheduler import DrafterScheduler, step_matches_interval
from .schedule_types import (
    CollectionPlan,
    CollectionPayload,
    CollectionWorkerResult,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    PublishPlan,
    TrainingBudget,
    TrainingDataStatus,
    TrainingPlan,
    TrainingResult,
    TriggerDecision,
)
from .worker_executor import CallbackDrafterWorkerExecutor, DrafterWorkerExecutor
from .publish_executor import CallbackDrafterPublishExecutor, DrafterPublishExecutor
from .publish_strategy import PublishOutcome
from .data_status_policy import ConservativeTrainingDataStatusPolicy
from .collection_executor import (
    CallbackDrafterCollectionExecutor,
    DrafterCollectionExecutor,
)
from .lifecycle import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    SchedulerEventOutcome,
)
from .collection_strategy import CollectionOutcome, DrafterCollectionStrategy
from .collection_adapter import DrafterCollectionAdapter
from .training_outcome import TrainingOutcome

__all__ = [
    "AfterActorUpdateContext",
    "AfterWeightUpdateContext",
    "BeforeActorUpdateContext",
    "CollectionPlan",
    "CollectionPayload",
    "CollectionWorkerResult",
    "CollectionOutcome",
    "ConservativeTrainingDataStatusPolicy",
    "CallbackDrafterWorkerExecutor",
    "CallbackDrafterCollectionExecutor",
    "CallbackDrafterPublishExecutor",
    "DrafterCollectionContext",
    "DrafterCollectionSource",
    "DrafterExecutionStrategy",
    "DrafterRuntimeState",
    "DrafterRuntimeStatus",
    "DrafterScheduleConfig",
    "DrafterScheduleContext",
    "DrafterScheduler",
    "DrafterWorkerExecutor",
    "DrafterCollectionExecutor",
    "DrafterCollectionAdapter",
    "DrafterCollectionStrategy",
    "DrafterPublishExecutor",
    "PublishPlan",
    "SchedulerEventOutcome",
    "PublishOutcome",
    "TrainingBudget",
    "TrainingDataStatus",
    "TrainingPlan",
    "TrainingOutcome",
    "TrainingResult",
    "TriggerDecision",
    "step_matches_interval",
]
