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
"""Legacy-equivalent synchronous drafter scheduling.

This module is the single decision source for synchronous collection, training,
batch limits, and publication. Trainers and runtimes request plans; workers only
execute the serialized plan. The blocking training behavior and call ordering
remain unchanged.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPlan,
    CollectionPayload,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    PublishPlan,
    TrainingPlan,
    _as_int,
)
from verl_speco.trainer.scheduler.execution_strategy import SyncExecutionStrategy
from verl_speco.trainer.scheduler.training_budget import SyncTrainingBudgetPolicy
from verl_speco.trainer.scheduler.training_trigger import IntervalAndBufferTrigger
from verl_speco.trainer.scheduler.worker_executor import DrafterWorkerExecutor
from verl_speco.trainer.scheduler.publish_executor import DrafterPublishExecutor
from verl_speco.trainer.scheduler.publish_strategy import PublishExecutionStrategy
from verl_speco.trainer.scheduler.data_status_policy import (
    ConservativeTrainingDataStatusPolicy,
)
from verl_speco.trainer.scheduler.lifecycle import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    SchedulerEventOutcome,
)
from verl_speco.trainer.scheduler.collection_executor import (
    DrafterCollectionExecutor,
)
from verl_speco.trainer.scheduler.collection_strategy import SyncCollectionStrategy
from verl_speco.trainer.scheduler.collection_adapter import (
    DrafterCollectionAdapter,
    OldLogProbCollectionAdapter,
    SGLangCollectionAdapter,
)
from verl_speco.trainer.scheduler.training_outcome import TrainingOutcome


def step_matches_interval(
    global_step: Any,
    interval_steps: Any,
    *,
    default_interval: int = 1,
) -> bool:
    """Match the released ``speco_step_matches_interval`` semantics exactly."""

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


class DrafterScheduler:
    """Make synchronous collect, train, and publish decisions.

    Every decision receives a
    fresh legacy-compatible config snapshot so runtime config mutation retains
    the same behavior as the released trainer implementation.
    """

    def __init__(
        self,
        worker_executor: DrafterWorkerExecutor | None = None,
        publish_executor: DrafterPublishExecutor | None = None,
        collection_executor: DrafterCollectionExecutor | None = None,
    ) -> None:
        self.trigger_policy = IntervalAndBufferTrigger()
        self.sync_budget_policy = SyncTrainingBudgetPolicy()
        self.sync_execution_strategy = SyncExecutionStrategy()
        self._worker_executor = worker_executor
        self.data_status_policy = ConservativeTrainingDataStatusPolicy()
        self.publish_execution_strategy = PublishExecutionStrategy()
        self._publish_executor = publish_executor
        self.collection_strategy = SyncCollectionStrategy()
        self._collection_executor = collection_executor
        self._collection_adapters: dict[
            DrafterCollectionSource, DrafterCollectionAdapter
        ] = {
            DrafterCollectionSource.SGLANG: SGLangCollectionAdapter(),
            DrafterCollectionSource.OLD_LOGPROB: OldLogProbCollectionAdapter(),
        }

    def bind_worker_executor(self, worker_executor: DrafterWorkerExecutor) -> None:
        """Bind the worker execution port used by all execution strategies."""

        self._worker_executor = worker_executor

    def bind_publish_executor(self, publish_executor: DrafterPublishExecutor) -> None:
        self._publish_executor = publish_executor

    def bind_collection_executor(
        self, collection_executor: DrafterCollectionExecutor
    ) -> None:
        self._collection_executor = collection_executor

    def register_collection_adapter(self, adapter: DrafterCollectionAdapter) -> None:
        """Register or replace the payload adapter for a collection source."""
        self._collection_adapters[adapter.source] = adapter

    def prepare_collection_payload(
        self,
        *,
        source: DrafterCollectionSource,
        samples: list[dict],
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners=None,
    ) -> CollectionPayload:
        """Build the common payload without exposing bucketing to the Trainer."""
        adapter = self._collection_adapters.get(source)
        if adapter is None:
            raise ValueError(f"No collection adapter registered for {source.value}")
        return adapter.prepare_payload(
            samples,
            owner_count=owner_count,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=raw_samples,
            collection_id=collection_id,
            owners=owners,
        )

    def inspect_training_data(
        self, *, global_step: object, config: DrafterScheduleConfig
    ):
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        statuses = self._worker_executor.get_training_data_status(
            sample_last_n_steps=config.sample_last_n_steps,
            require_full_batch=config.require_full_batch,
        )
        return self.data_status_policy.aggregate(statuses, global_step=global_step)

    def prepare_training_plan(
        self, context: DrafterScheduleContext, config: DrafterScheduleConfig
    ) -> TrainingPlan:
        """Build a plan while avoiding worker RPCs for cheap skip conditions."""

        interval_matched = self.training_interval_matched(context.global_step, config)
        if (
            context.training_mode == "collect_only"
            or context.pending_training_count > 0
            or not interval_matched
            or (
                context.collected_samples_this_step <= 0
                and (
                    context.oldlogprob_collection_requested
                    or not config.use_data_buffer
                )
            )
        ):
            return self.plan_training(context, config)
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step, config=config
        )
        return self.plan_training(
            DrafterScheduleContext(
                global_step=context.global_step,
                training_mode=context.training_mode,
                collected_samples_this_step=context.collected_samples_this_step,
                oldlogprob_collection_requested=context.oldlogprob_collection_requested,
                data_status=data_status,
                pending_training_count=context.pending_training_count,
            ),
            config,
        )

    def prepare_training_execution(self, plan: TrainingPlan) -> dict[str, Any]:
        if not plan.launch:
            return {"drafter/target_lm_head_synced": 0}
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        return self._worker_executor.prepare_training(plan)

    def activate_training_workers(self) -> list[Any]:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        results = self._worker_executor.activate_training_workers()
        active = [
            result
            for result in results
            if isinstance(result, dict)
            and result.get("reason") not in {"disabled", "not_in_training_group"}
        ]
        failures = [result for result in active if not result.get("activated", False)]
        if failures:
            raise RuntimeError(
                f"SPECO drafter trainer activation failed: {failures[:3]}"
            )
        return results

    def wait_pending_publish(self) -> int:
        if self._publish_executor is None:
            return 0
        return self._publish_executor.wait_pending()

    @staticmethod
    def should_collect(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.collect_interval_steps)

    def plan_collection(
        self,
        context: DrafterCollectionContext,
        config: DrafterScheduleConfig,
    ) -> CollectionPlan:
        collect_interval_matched = self.should_collect(context.global_step, config)
        training_interval_matched = self.training_interval_matched(
            context.global_step, config
        )
        common: Any = {
            "collection_id": uuid4().hex,
            "source": context.source,
            "source_global_step": context.global_step,
            "collect_interval_matched": collect_interval_matched,
            "training_interval_matched": training_interval_matched,
            "sample_rate": config.collection_sample_rate,
            "max_samples_per_replica": config.max_collect_samples_per_replica,
            "max_tokens_per_replica": config.max_collect_tokens_per_replica,
            "hidden_window_mode": config.hidden_window_mode,
            "hidden_window_tokens_per_sample": config.hidden_window_tokens_per_sample,
            "hidden_window_min_rows": config.hidden_window_min_rows,
        }
        if not context.drafter_enabled:
            return CollectionPlan(collect=False, reason="drafter_disabled", **common)
        if not context.source_enabled:
            return CollectionPlan(collect=False, reason="source_disabled", **common)
        if context.validation:
            return CollectionPlan(collect=False, reason="validation", **common)
        if not collect_interval_matched:
            return CollectionPlan(
                collect=False, reason="interval_not_reached", **common
            )
        if context.require_training_interval and not training_interval_matched:
            return CollectionPlan(
                collect=False,
                reason="training_interval_not_reached",
                **common,
            )
        if config.collection_sample_rate <= 0:
            return CollectionPlan(collect=False, reason="sample_rate_zero", **common)
        return CollectionPlan(collect=True, reason="collection_enabled", **common)

    def execute_collection_plan(self, plan: CollectionPlan, payload: CollectionPayload):
        if self._collection_executor is None:
            raise RuntimeError("Drafter collection executor has not been bound")
        return self.collection_strategy.execute(
            plan,
            payload,
            executor=self._collection_executor,
        )

    def on_collection_ready(self, plan: CollectionPlan, payload: CollectionPayload):
        """Lifecycle Facade event for a source adapter's prepared payload."""
        return self.execute_collection_plan(plan, payload)

    def on_before_actor_update(
        self, context: BeforeActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Plan and prepare drafter training before the PPO actor update."""
        plan = self.prepare_training_plan(
            context.schedule_context,
            context.config,
        )
        metrics: dict[str, Any] = dict(plan.metrics())
        metrics.update(self.prepare_training_execution(plan))
        return SchedulerEventOutcome(
            training_plan=plan,
            metrics=metrics,
        )

    def on_after_actor_update(
        self, context: AfterActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Execute the prepared plan after the PPO actor update."""
        plan = context.training_plan
        if not plan.launch:
            return SchedulerEventOutcome(training_plan=plan, metrics={})
        execution = self.execute_training_plan(
            plan,
            runtime_state=context.runtime_state,
        )
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=context.runtime_state,
            plan=plan,
        )
        return SchedulerEventOutcome(
            training_plan=plan,
            training_execution=outcome,
            metrics=outcome.metrics,
        )

    def on_safe_point(self, context: AfterWeightUpdateContext) -> SchedulerEventOutcome:
        """Plan and execute publication at a rollout-safe lifecycle point."""
        plan = self.plan_publish(
            global_step=context.global_step,
            drafter_trained=context.drafter_trained,
            config=context.config,
            training_plan=context.training_plan,
        )
        outcome = self.execute_publish_plan(plan)
        return SchedulerEventOutcome(
            training_plan=context.training_plan,
            publish_plan=plan,
            publish_outcome=outcome,
            metrics=outcome.metrics(),
        )

    def on_after_weight_update(
        self, context: AfterWeightUpdateContext
    ) -> SchedulerEventOutcome:
        """Named lifecycle alias for the post-weight-update safe point."""
        return self.on_safe_point(context)

    @staticmethod
    def training_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.training_interval_steps)

    def plan_training(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        trigger = self.trigger_policy.should_train(
            context,
            config,
            interval_matched=interval_matched,
        )
        budget = self.sync_budget_policy.make_budget(context, config)
        min_sample_step, max_sample_step, data_filter_reason = (
            self._training_data_filter_window(
                context, config, budget.sample_last_n_steps
            )
        )
        common: Any = {
            "interval_matched": interval_matched,
            "execution_strategy": DrafterExecutionStrategy.SYNC,
            "source_global_step": context.global_step,
            "max_batches": budget.max_batches,
            "min_batches": budget.min_batches,
            "deadline_ts": budget.deadline_ts,
            "require_full_batch": budget.require_full_batch,
            "sample_last_n_steps": budget.sample_last_n_steps,
            "data_version": (
                context.data_status.data_version if context.data_status else None
            ),
            "required_target_version": (
                None if config.use_logits else _as_int(context.global_step)
            ),
            "min_sample_step": min_sample_step,
            "max_sample_step": max_sample_step,
            "data_filter_reason": data_filter_reason,
            "plan_id": uuid4().hex,
            "worker_snapshots": (
                context.data_status.worker_snapshots if context.data_status else None
            ),
        }
        if not trigger.should_train:
            return TrainingPlan(
                launch=False,
                reason=trigger.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches <= 0:
            return TrainingPlan(
                launch=False,
                reason=budget.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches < budget.min_batches:
            return TrainingPlan(
                launch=False,
                reason="insufficient_training_budget",
                publish_after_success=False,
                **common,
            )
        return TrainingPlan(
            launch=True,
            reason=trigger.reason,
            publish_after_success=self._publish_interval_matched(
                context.global_step, config
            ),
            **common,
        )

    @staticmethod
    def _training_data_filter_window(
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        sample_last_n_steps: int,
    ) -> tuple[int | None, int | None, str]:
        """Return the sample-step window workers must apply when training.

        The scheduler owns the filtering decision, while workers/base trainers
        only execute this window against their local buffer or Feature Store
        replay source.
        """

        try:
            current_step = _as_int(context.global_step)
        except Exception:  # noqa: BLE001
            return None, None, "invalid_global_step"

        max_sample_step = current_step
        if (
            context.data_status is not None
            and context.data_status.same_step_data_required
        ):
            return current_step, max_sample_step, "same_step_required"
        if not config.use_data_buffer:
            return current_step, max_sample_step, "current_step_only"

        min_sample_step = max(0, current_step - max(int(sample_last_n_steps), 0))
        return min_sample_step, max_sample_step, "recent_buffer_window"

    def execute_training_plan(self, plan: TrainingPlan, *, runtime_state):
        """Execute through the strategy selected by the generated plan."""

        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")

        if plan.execution_strategy is DrafterExecutionStrategy.SYNC:
            return self.sync_execution_strategy.execute(
                plan,
                executor=self._worker_executor,
                runtime_state=runtime_state,
            )
        raise NotImplementedError(
            f"Unsupported drafter execution strategy: {plan.execution_strategy.value}"
        )

    @staticmethod
    def _publish_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        interval = _as_int(config.publish_interval_steps or 0)
        return interval <= 0 or _as_int(global_step) % interval == 0

    @staticmethod
    def plan_publish(
        *,
        global_step: object,
        drafter_trained: bool,
        config: DrafterScheduleConfig,
        training_plan: TrainingPlan | None = None,
    ) -> PublishPlan:
        if not drafter_trained or (
            training_plan is not None and not training_plan.publish_after_success
        ):
            return PublishPlan(
                publish=False,
                reason=(
                    "drafter_not_trained"
                    if not drafter_trained
                    else "training_plan_publish_disabled"
                ),
                interval_matched=False,
                source_global_step=global_step,
                asynchronous=config.publish_async,
            )
        # Preserve the released path exactly: invalid publish configuration is
        # an error instead of being silently converted into a skipped publish.
        interval_matched = DrafterScheduler._publish_interval_matched(
            global_step, config
        )
        return PublishPlan(
            publish=interval_matched,
            reason=(
                "publish_interval_reached"
                if interval_matched
                else "publish_interval_not_reached"
            ),
            interval_matched=interval_matched,
            source_global_step=global_step,
            asynchronous=config.publish_async,
        )

    def execute_publish_plan(self, plan: PublishPlan):
        if self._publish_executor is None:
            raise RuntimeError("Drafter publish executor has not been bound")
        return self.publish_execution_strategy.execute(
            plan, executor=self._publish_executor
        )
