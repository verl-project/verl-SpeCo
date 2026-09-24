# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Execution boundaries for event-driven standalone drafter training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPlan,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterTrainingDataSource,
    ProducerAction,
    TrainingPlan,
)


class StandaloneCollectionExecutor(Protocol):
    """Translate standalone collection actions into Producer commands."""

    def pause_producer(self) -> Any: ...

    def resume_producer(self) -> Any: ...

    def stop_producer(self) -> Any: ...


@dataclass(frozen=True)
class CallbackStandaloneCollectionExecutor:
    pause: Callable[[], Any]
    resume: Callable[[], Any]
    stop: Callable[[], Any]
    resolve: Callable[[Any], Any]

    def pause_producer(self) -> Any:
        return self.resolve(self.pause())

    def resume_producer(self) -> Any:
        return self.resolve(self.resume())

    def stop_producer(self) -> Any:
        return self.resolve(self.stop())


class StandaloneTrainingExecutor(Protocol):
    """Translate a standalone TrainingPlan into a Consumer command."""

    def submit_training(
        self, plan: TrainingPlan, *, selected_entries: Sequence[Any]
    ) -> Any: ...

    def stop_consumer(self) -> Any: ...


@dataclass(frozen=True)
class CallbackStandaloneTrainingExecutor:
    submit: Callable[[TrainingPlan, Sequence[Any]], Any]
    stop: Callable[[], Any]

    def submit_training(
        self, plan: TrainingPlan, *, selected_entries: Sequence[Any]
    ) -> Any:
        return self.submit(plan, selected_entries)

    def stop_consumer(self) -> Any:
        return self.stop()


@dataclass(frozen=True)
class StandaloneCollectionOutcome:
    action: ProducerAction
    changed: bool
    producer_paused: bool
    producer_stopped: bool


@dataclass(frozen=True)
class StandaloneTrainingOutcome:
    submitted: bool
    selected_keys: tuple[str, ...] = ()
    assigned_samples: int = 0
    completed: bool = False
    reason: str = ""


class StandaloneCollectionExecutionStrategy:
    def execute(
        self,
        plan: CollectionPlan,
        *,
        executor: StandaloneCollectionExecutor,
        producer_paused: bool,
        producer_done: bool,
    ) -> StandaloneCollectionOutcome:
        if plan.source is not DrafterCollectionSource.TRANSFER_QUEUE:
            raise ValueError(
                "Standalone collection execution requires a TransferQueue plan"
            )
        action = plan.producer_action
        changed = False
        stopped = producer_done
        paused = producer_paused
        if action is ProducerAction.PAUSE and not producer_paused and not producer_done:
            executor.pause_producer()
            changed = True
            paused = True
        elif action is ProducerAction.RUN and producer_paused and not producer_done:
            executor.resume_producer()
            changed = True
            paused = False
        elif action is ProducerAction.STOP and not producer_done:
            executor.stop_producer()
            changed = True
            stopped = True
        return StandaloneCollectionOutcome(
            action=action,
            changed=changed,
            producer_paused=paused,
            producer_stopped=stopped,
        )


class StandaloneTrainingExecutionStrategy:
    def execute(
        self,
        plan: TrainingPlan,
        *,
        executor: StandaloneTrainingExecutor,
        runtime_state: DrafterRuntimeState,
        selected_entries: Sequence[Any],
    ) -> StandaloneTrainingOutcome:
        if plan.data_source is not DrafterTrainingDataSource.TRANSFER_QUEUE:
            raise ValueError(
                "Standalone training execution requires a TransferQueue plan"
            )
        if plan.execution_strategy is not DrafterExecutionStrategy.STANDALONE_ASYNC:
            raise ValueError(
                "Standalone training execution requires STANDALONE_ASYNC strategy"
            )
        if not plan.launch:
            return StandaloneTrainingOutcome(submitted=False, reason=plan.reason)
        selected_keys = tuple(str(entry.key) for entry in selected_entries)
        if selected_keys != tuple(plan.selected_keys):
            raise RuntimeError(
                "Standalone TrainingPlan selected_keys do not match the confirmed "
                "TransferQueue entries"
            )
        if plan.required_samples is not None and len(selected_keys) != int(
            plan.required_samples
        ):
            raise RuntimeError(
                "Standalone TrainingPlan does not contain one complete global batch"
            )
        if runtime_state.status in {
            DrafterRuntimeStatus.COMPLETED,
            DrafterRuntimeStatus.FAILED,
        }:
            runtime_state.reset()
        started_at = time.perf_counter()
        runtime_state.submit(plan, started_at=started_at)
        try:
            executor.submit_training(plan, selected_entries=selected_entries)
            runtime_state.mark_running()
        except Exception as error:
            runtime_state.mark_failed(error)
            raise
        return StandaloneTrainingOutcome(
            submitted=True,
            selected_keys=selected_keys,
            assigned_samples=len(selected_keys),
            reason="submitted",
        )

    @staticmethod
    def complete(
        *,
        runtime_state: DrafterRuntimeState,
        completed_keys: Sequence[str],
        successful: bool,
    ) -> StandaloneTrainingOutcome:
        plan = runtime_state.active_plan
        if runtime_state.status is not DrafterRuntimeStatus.RUNNING or plan is None:
            raise RuntimeError(
                "Standalone Consumer completed without an active TrainingPlan"
            )
        expected_keys = tuple(plan.selected_keys)
        actual_keys = tuple(str(key) for key in completed_keys)
        if actual_keys != expected_keys:
            error = RuntimeError(
                "Standalone Consumer completed keys do not match the active "
                "TrainingPlan"
            )
            runtime_state.mark_failed(error)
            raise error
        if not successful:
            error = RuntimeError("Standalone Consumer reported a failed training step")
            runtime_state.mark_failed(error)
            raise error
        elapsed = max(time.perf_counter() - float(runtime_state.started_at or 0.0), 0.0)
        runtime_state.mark_completed(completed_batches=1, elapsed_sec=elapsed)
        return StandaloneTrainingOutcome(
            submitted=True,
            selected_keys=actual_keys,
            assigned_samples=len(actual_keys),
            completed=True,
            reason="completed",
        )


__all__ = [
    "CallbackStandaloneCollectionExecutor",
    "CallbackStandaloneTrainingExecutor",
    "StandaloneCollectionExecutionStrategy",
    "StandaloneCollectionExecutor",
    "StandaloneCollectionOutcome",
    "StandaloneTrainingExecutionStrategy",
    "StandaloneTrainingExecutor",
    "StandaloneTrainingOutcome",
]
