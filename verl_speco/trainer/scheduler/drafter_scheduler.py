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

import logging

from dataclasses import replace
from typing import Any, Sequence
from uuid import uuid4

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPlan,
    CollectionPayload,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterTrainingDataSource,
    ProducerAction,
    PublishPlan,
    QueueScheduleContext,
    TrainingPlan,
    _as_int,
)
from verl_speco.trainer.scheduler.execution_strategy import SyncExecutionStrategy
from verl_speco.trainer.scheduler.training_budget import (
    SyncTrainingBudgetPolicy,
    AdaptiveTrainingBudgetPolicy,
)
from verl_speco.trainer.scheduler.adaptive_schedule import AdaptiveScheduleController
from verl_speco.trainer.scheduler.training_trigger import IntervalAndBufferTrigger
from verl_speco.trainer.scheduler.training_opportunity import (
    TrainingOpportunity,
    training_opportunity,
    step_matches_interval,
)
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
from verl_speco.trainer.scheduler.drafter_runtime_state import DrafterRuntimeState
from verl_speco.trainer.scheduler.standalone_executor import (
    StandaloneCollectionExecutionStrategy,
    StandaloneCollectionExecutor,
    StandaloneCollectionOutcome,
    StandaloneTrainingExecutionStrategy,
    StandaloneTrainingExecutor,
    StandaloneTrainingOutcome,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        standalone_collection_executor: StandaloneCollectionExecutor | None = None,
        standalone_training_executor: StandaloneTrainingExecutor | None = None,
    ) -> None:
        self.trigger_policy = IntervalAndBufferTrigger()
        self.sync_budget_policy = SyncTrainingBudgetPolicy()
        self.adaptive_controller: AdaptiveScheduleController | None = None
        self._adaptive_pending_publish_step: int | None = None
        self.sync_execution_strategy = SyncExecutionStrategy()
        self._worker_executor = worker_executor
        self.data_status_policy = ConservativeTrainingDataStatusPolicy()
        self.publish_execution_strategy = PublishExecutionStrategy()
        self._publish_executor = publish_executor
        self.collection_strategy = SyncCollectionStrategy()
        self._collection_executor = collection_executor
        self.standalone_collection_strategy = StandaloneCollectionExecutionStrategy()
        self.standalone_training_strategy = StandaloneTrainingExecutionStrategy()
        self._standalone_collection_executor = standalone_collection_executor
        self._standalone_training_executor = standalone_training_executor
        self._collection_adapters: dict[
            DrafterCollectionSource, DrafterCollectionAdapter
        ] = {
            DrafterCollectionSource.SGLANG: SGLangCollectionAdapter(),
            DrafterCollectionSource.OLD_LOGPROB: OldLogProbCollectionAdapter(),
        }

    def bind_worker_executor(self, worker_executor: DrafterWorkerExecutor) -> None:
        """Bind the worker execution port used by all execution strategies."""

        self._worker_executor = worker_executor

    def bind_standalone_collection_executor(
        self, executor: StandaloneCollectionExecutor
    ) -> None:
        self._standalone_collection_executor = executor

    def bind_standalone_training_executor(
        self, executor: StandaloneTrainingExecutor
    ) -> None:
        self._standalone_training_executor = executor

    @staticmethod
    def plan_queue_collection(
        context: QueueScheduleContext,
    ) -> CollectionPlan:
        """Build a queue Producer plan with high/low-watermark backpressure."""

        status = context.queue_status
        config = context.config
        if context.producer_done:
            action = ProducerAction.STOP
            reason = "producer_done"
        elif status.ready_samples >= config.high_watermark_samples:
            action = ProducerAction.PAUSE
            reason = "high_watermark_reached"
        elif status.ready_samples <= config.low_watermark_samples:
            action = ProducerAction.RUN
            reason = "low_watermark_reached"
        elif context.producer_paused:
            action = ProducerAction.PAUSE
            reason = "watermark_hysteresis_paused"
        else:
            action = ProducerAction.RUN
            reason = "watermark_hysteresis_running"
        return CollectionPlan(
            collect=action is ProducerAction.RUN,
            reason=reason,
            source=DrafterCollectionSource.TRANSFER_QUEUE,
            source_global_step=0,
            collect_interval_matched=True,
            training_interval_matched=True,
            sample_rate=1.0,
            max_samples_per_replica=None,
            max_tokens_per_replica=None,
            hidden_window_mode="front",
            hidden_window_tokens_per_sample=None,
            hidden_window_min_rows=0,
            producer_action=action,
            max_new_samples=None,
        )

    @staticmethod
    def plan_queue_training(
        context: QueueScheduleContext,
        *,
        selected_keys: tuple[str, ...] = (),
    ) -> TrainingPlan:
        """Plan at most one complete standalone global batch."""

        status = context.queue_status
        config = context.config
        common: dict[str, Any] = {
            "interval_matched": True,
            "execution_strategy": DrafterExecutionStrategy.STANDALONE_ASYNC,
            "source_global_step": 0,
            "max_batches": 1,
            "publish_after_success": False,
            "min_batches": 1,
            "require_full_batch": True,
            "data_filter_reason": "transfer_queue",
            "data_source": DrafterTrainingDataSource.TRANSFER_QUEUE,
            "required_samples": config.global_batch_size,
        }
        if context.consumer_training:
            return TrainingPlan(
                launch=False,
                reason="consumer_training",
                **common,
            )
        if status.ready_samples < config.global_batch_size:
            return TrainingPlan(
                launch=False,
                reason="insufficient_ready_samples",
                **common,
            )
        return TrainingPlan(
            launch=True,
            reason="training_ready",
            selected_keys=selected_keys,
            **common,
        )

    def execute_standalone_collection_plan(
        self,
        plan: CollectionPlan,
        *,
        producer_paused: bool,
        producer_done: bool,
    ) -> StandaloneCollectionOutcome:
        if self._standalone_collection_executor is None:
            raise RuntimeError("Standalone collection executor has not been bound")
        return self.standalone_collection_strategy.execute(
            plan,
            executor=self._standalone_collection_executor,
            producer_paused=producer_paused,
            producer_done=producer_done,
        )

    def execute_standalone_training_plan(
        self,
        plan: TrainingPlan,
        *,
        runtime_state: DrafterRuntimeState,
        selected_entries: Sequence[Any],
    ) -> StandaloneTrainingOutcome:
        if self._standalone_training_executor is None:
            raise RuntimeError("Standalone training executor has not been bound")
        return self.standalone_training_strategy.execute(
            plan,
            executor=self._standalone_training_executor,
            runtime_state=runtime_state,
            selected_entries=selected_entries,
        )

    def complete_standalone_training(
        self,
        *,
        runtime_state: DrafterRuntimeState,
        completed_keys: Sequence[str],
        successful: bool,
    ) -> StandaloneTrainingOutcome:
        return self.standalone_training_strategy.complete(
            runtime_state=runtime_state,
            completed_keys=completed_keys,
            successful=successful,
        )

    def stop_standalone_consumer(self) -> Any:
        if self._standalone_training_executor is None:
            raise RuntimeError("Standalone training executor has not been bound")
        return self._standalone_training_executor.stop_consumer()

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

        controller = self.configure_adaptive(config)
        if controller is not None:
            controller.observe(
                context.acceptance_feedback,
                current_step=_as_int(context.global_step),
                training_interval_steps=_as_int(config.training_interval_steps),
            )
        interval_matched = self.training_opportunity(context.global_step, config).due
        if (
            controller is not None
            and interval_matched
            and context.training_mode != "collect_only"
            and context.pending_training_count <= 0
        ):
            controller.decide_budget(
                _as_int(context.global_step), _as_int(config.training_interval_steps)
            )
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
        return self.plan_training(replace(context, data_status=data_status), config)

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
        waited = self._publish_executor.wait_pending()
        if self._adaptive_pending_publish_step is not None:
            if self.adaptive_controller is not None:
                self.adaptive_controller.record_publish(
                    self._adaptive_pending_publish_step
                )
            self._adaptive_pending_publish_step = None
        return waited

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
        self.configure_adaptive(config)
        opportunity = self.training_opportunity(context.global_step, config)
        collect_interval_matched = opportunity.collection_due
        training_interval_matched = opportunity.due
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
        if (
            self.adaptive_controller is not None
            and context.config.adaptive_schedule.enable
        ):
            metrics.update(
                self.adaptive_controller.metrics(
                    _as_int(context.schedule_context.global_step)
                )
            )
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
        if self.adaptive_controller is not None and outcome.trained:
            controller = self.adaptive_controller
            step = _as_int(plan.source_global_step)
            controller.record_training(step, outcome.successful_steps)
            outcome.metrics["drafter/adaptive_budget_steps"] = plan.max_batches
            if controller.warmup_active(step):
                logger.info(
                    "[adaptive_schedule] step=%s warmup=true budget=%s reason=warmup_budget",
                    step,
                    plan.max_batches,
                )
            else:
                logger.info(
                    "[adaptive_schedule] step=%s warmup=false interval_trend=%s budget=%s->%s reason=%s",
                    step,
                    f"{controller.last_interval_trend:.3f}"
                    if controller.last_interval_trend is not None
                    else "n/a",
                    controller.last_budget_before,
                    plan.max_batches,
                    controller.last_reason,
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
        if outcome.published and self.adaptive_controller is not None:
            if plan.asynchronous:
                self._adaptive_pending_publish_step = _as_int(plan.source_global_step)
            else:
                self.adaptive_controller.record_publish(
                    _as_int(plan.source_global_step)
                )
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

    def configure_adaptive(
        self, config: DrafterScheduleConfig
    ) -> AdaptiveScheduleController | None:
        if not config.adaptive_schedule.enable:
            self.adaptive_controller = None
            self._adaptive_pending_publish_step = None
            return None
        if (
            self.adaptive_controller is None
            or self.adaptive_controller.config != config.adaptive_schedule
        ):
            self.adaptive_controller = AdaptiveScheduleController(
                config.adaptive_schedule
            )
        return self.adaptive_controller

    def training_opportunity(
        self, step: object, config: DrafterScheduleConfig
    ) -> TrainingOpportunity:
        """Read-only evaluation; configuration happens at planning entry points."""
        controller = self.adaptive_controller
        ended_after = (
            controller.state.warmup_ended_after
            if controller is not None and controller.config == config.adaptive_schedule
            else None
        )
        return training_opportunity(step, config, startup_ended_after=ended_after)

    def plan_training(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingPlan:
        controller = self.configure_adaptive(config)
        interval_matched = self.training_opportunity(context.global_step, config).due
        trigger = self.trigger_policy.should_train(
            context,
            config,
            interval_matched=interval_matched,
        )
        budget = (
            AdaptiveTrainingBudgetPolicy(controller).make_budget(context, config)
            if controller is not None
            else self.sync_budget_policy.make_budget(context, config)
        )
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
