# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Normalized multi-worker outcome for one drafter training event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.schedule_types import (
    TrainingPlan,
    TrainingResult,
    _as_float,
    _as_int,
)


def _metric_float(value: object) -> float | None:
    try:
        return _as_float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TrainingOutcome:
    trained: bool
    successful_steps: int
    worker_results: list[TrainingResult]
    raw_results: list[Any]
    elapsed_sec: float
    reason: str
    metrics: dict[str, float | int]

    @classmethod
    def from_execution(
        cls,
        execution: ExecutionOutcome,
        *,
        runtime_state: DrafterRuntimeState,
        plan: TrainingPlan,
    ) -> "TrainingOutcome":
        normalized_results: list[dict[str, object]] = []
        for result in execution.raw_results:
            if isinstance(result, dict):
                normalized_results.append(result)
            else:
                trained = bool(result)
                normalized_results.append(
                    {
                        "trained": trained,
                        "triggered": trained,
                        "attempted_steps": int(trained),
                        "successful_steps": int(trained),
                        "elapsed_sec": 0.0,
                        "reason": "legacy_bool_result",
                    }
                )

        trained = any(
            bool(result.get("trained", False)) for result in normalized_results
        )
        successful_steps = max(
            (
                _as_int(result.get("successful_steps", 0))
                for result in normalized_results
            ),
            default=0,
        )
        worker_results = [
            TrainingResult.from_mapping(result) for result in normalized_results
        ]
        participating_results = [
            TrainingResult.from_mapping(result)
            for result in normalized_results
            if bool(result.get("triggered", False))
        ]
        expected_worker_ids = set((plan.worker_snapshots or {}).keys())
        actual_worker_ids = {result.worker_id for result in participating_results}
        strict_consistency = bool(expected_worker_ids)
        worker_ids_consistent = actual_worker_ids == expected_worker_ids and len(
            participating_results
        ) == len(expected_worker_ids)
        incarnations_consistent = all(
            result.worker_incarnation for result in participating_results
        )
        plan_ids_consistent = all(
            result.plan_id == plan.plan_id for result in participating_results
        )
        source_steps_consistent = all(
            result.source_global_step == _as_int(plan.source_global_step)
            for result in participating_results
        )
        data_versions_consistent = all(
            result.data_version == plan.data_version for result in participating_results
        )
        target_versions_consistent = all(
            plan.required_target_version is None
            or result.target_version == plan.required_target_version
            for result in participating_results
        )
        trained_consistent = (
            len({result.trained for result in participating_results}) == 1
        )
        successful_steps_consistent = (
            len({result.successful_steps for result in participating_results}) == 1
        )
        optimizer_steps_consistent = (
            len({result.optimizer_step for result in participating_results}) == 1
        )
        publish_leaders = [
            result for result in participating_results if result.is_publish_leader
        ]
        publish_snapshot_consistent = not plan.publish_after_success or (
            len(publish_leaders) == 1 and publish_leaders[0].snapshot_ready
        )
        result_consistent = not strict_consistency or (
            worker_ids_consistent
            and incarnations_consistent
            and plan_ids_consistent
            and source_steps_consistent
            and data_versions_consistent
            and target_versions_consistent
            and trained_consistent
            and successful_steps_consistent
            and optimizer_steps_consistent
            and publish_snapshot_consistent
        )
        if not result_consistent:
            trained = False
        metrics: dict[str, float | int] = {
            "drafter/trained": int(trained),
            "drafter/train_successful_steps_max": successful_steps,
            "drafter/train_no_trainable_batch": int(
                any(
                    result.get("reason") == "no_trainable_batch"
                    for result in normalized_results
                )
            ),
            "drafter/train_activation_failed": int(
                any(
                    result.get("reason") == "activation_failed"
                    for result in normalized_results
                )
            ),
            "drafter/train_attempted_batches_max": max(
                (result.attempted_batches for result in worker_results), default=0
            ),
            "drafter/train_buffer_size_before_min": min(
                (result.buffer_size_before for result in worker_results), default=0
            ),
            "drafter/train_buffer_size_after_min": min(
                (result.buffer_size_after for result in worker_results), default=0
            ),
            "drafter/train_optimizer_step_max": max(
                (result.optimizer_step for result in worker_results), default=0
            ),
            "drafter/train_worker_results_consistent": int(result_consistent),
            "drafter/train_worker_ids_consistent": int(worker_ids_consistent),
            "drafter/train_worker_incarnations_consistent": int(
                incarnations_consistent
            ),
            "drafter/train_plan_ids_consistent": int(plan_ids_consistent),
            "drafter/train_source_steps_consistent": int(source_steps_consistent),
            "drafter/train_data_versions_consistent": int(data_versions_consistent),
            "drafter/train_target_versions_consistent": int(target_versions_consistent),
            "drafter/train_successful_steps_consistent": int(
                successful_steps_consistent
            ),
            "drafter/train_optimizer_steps_consistent": int(optimizer_steps_consistent),
            "drafter/train_publish_leader_count": len(publish_leaders),
            "drafter/train_publish_leader_snapshot_ready": int(
                len(publish_leaders) == 1 and publish_leaders[0].snapshot_ready
            ),
        }
        for key in (
            "timing_s/drafter_prepare_batch",
            "timing_s/drafter_forward_loss",
            "timing_s/drafter_reduce_loss",
            "timing_s/drafter_backward",
            "timing_s/drafter_optimizer",
            "timing_s/drafter_publish_snapshot",
            "activation_elapsed_sec",
            "training_loop_elapsed_sec",
            "cleanup_elapsed_sec",
            "elapsed_sec",
        ):
            values = [
                value
                for result in normalized_results
                if (value := _metric_float(result.get(key))) is not None
            ]
            if values:
                metric_key = {
                    "activation_elapsed_sec": "timing_s/drafter_worker_activation",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                }.get(key, key)
                metrics[metric_key] = max(values)

        metrics["timing_s/drafter_train_rpc"] = execution.elapsed_sec
        outcome_reason = (
            execution.reason if result_consistent else "worker_result_inconsistent"
        )
        if runtime_state.status is DrafterRuntimeStatus.RUNNING and result_consistent:
            runtime_state.mark_completed(
                completed_batches=successful_steps,
                elapsed_sec=execution.elapsed_sec,
            )
        elif runtime_state.status in {
            DrafterRuntimeStatus.SUBMITTED,
            DrafterRuntimeStatus.RUNNING,
        }:
            runtime_state.mark_failed(outcome_reason)
        else:
            raise RuntimeError(
                "Drafter training execution returned with unexpected runtime state "
                f"{runtime_state.status.name}"
            )
        metrics.update(runtime_state.metrics())
        runtime_state.reset()
        return cls(
            trained=trained,
            successful_steps=successful_steps,
            worker_results=worker_results,
            raw_results=execution.raw_results,
            elapsed_sec=execution.elapsed_sec,
            reason=outcome_reason,
            metrics=metrics,
        )
