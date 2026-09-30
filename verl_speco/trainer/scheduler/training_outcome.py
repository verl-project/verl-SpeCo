# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Normalized multi-worker outcome for one drafter training event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.schedule_types import (
    DrafterExecutionStrategy,
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
        # A deadline/reclaim may stop an otherwise valid Bubble plan after it
        # has completed only part of ``max_batches``.  Such optimizer steps
        # must still repay the training quota even though the worker correctly
        # did not cache a publish snapshot.  Require the snapshot only once the
        # plan has completed all requested steps and is therefore publishable.
        publish_snapshot_required = bool(
            plan.publish_after_success
            and trained
            and successful_steps >= int(plan.max_batches)
        )
        publish_snapshot_consistent = not publish_snapshot_required or (
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
            "drafter/trained_any": int(trained),
            "drafter/idle_trained": int(
                plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
                and trained
            ),
            "drafter/train_successful_steps_max": successful_steps,
            "drafter/train_successful_valid_tokens_max": max(
                (result.successful_valid_tokens for result in worker_results),
                default=0,
            ),
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
            "bubble/replica_local_unavailable": int(
                any(
                    bool(result.get("replica_local_unavailable", False))
                    for result in normalized_results
                )
            ),
            "bubble/replica_local_oom": int(
                any(
                    bool(result.get("replica_local_oom", False))
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
            "drafter/train_trained_consistent": int(trained_consistent),
            "drafter/train_successful_steps_consistent": int(
                successful_steps_consistent
            ),
            "drafter/train_optimizer_steps_consistent": int(optimizer_steps_consistent),
            "drafter/train_publish_snapshot_required": int(publish_snapshot_required),
            "drafter/train_publish_snapshot_consistent": int(
                publish_snapshot_consistent
            ),
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
            "preflight_elapsed_sec",
            "preflight_to_first_batch_sec",
            "preflight_to_stop_sec",
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
                    "preflight_elapsed_sec": "timing_s/drafter_worker_preflight",
                    "preflight_to_first_batch_sec": "timing_s/drafter_worker_preflight_to_first_batch",
                    "preflight_to_stop_sec": "timing_s/drafter_worker_preflight_to_stop",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                }.get(key, key)
                metrics[metric_key] = max(values)

        metrics["bubble/train_reclaimed_before_first_batch"] = int(
            any(
                (
                    result.get("reason") == "reclaim_requested"
                    or result.get("stop_reason") == "reclaim_requested"
                )
                and int(cast(Any, result.get("attempted_steps", 0) or 0)) == 0
                for result in normalized_results
            )
        )
        stop_reasons = {
            str(result.get("stop_reason") or result.get("reason") or "")
            for result in normalized_results
        }
        metrics["bubble/train_stop_reclaim_requested"] = int(
            "reclaim_requested" in stop_reasons
        )
        metrics["bubble/train_stop_deadline_reached"] = int(
            "deadline_reached" in stop_reasons
        )
        metrics["bubble/train_stop_max_batches_reached"] = int(
            "max_batches_reached" in stop_reasons
        )
        metrics["bubble/train_stop_no_trainable_batch"] = int(
            "no_trainable_batch" in stop_reasons
        )
        metrics["bubble/train_first_batch_started"] = int(
            any(
                bool(result.get("first_batch_started", False))
                for result in normalized_results
            )
        )
        metrics["bubble/training_residency_retained"] = int(
            any(
                bool(result.get("training_residency_retained", False))
                for result in normalized_results
            )
        )
        metrics["timing_s/drafter_train_rpc"] = execution.elapsed_sec
        if (
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and plan.reason
            not in {
                "quota_topup_training_ready",
                "quota_forced_completion_ready",
            }
        ):
            # Separate Bubble worker work from the time the PPO path waited
            # to reclaim that worker group for a subsequent rollout.
            metrics["timing_s/drafter_async_training_work"] = float(
                metrics.get("timing_s/drafter_worker_elapsed", 0.0)
            )
        if plan.reason in {
            "quota_topup_training_ready",
            "quota_forced_completion_ready",
        }:
            topup_completed = bool(
                trained and successful_steps == int(plan.max_batches)
            )
            metrics.update(
                {
                    "bubble/training_quota_topup_completed": int(topup_completed),
                    "bubble/training_quota_topup_successful_steps": successful_steps,
                    "bubble/training_quota_topup_shortfall_steps": max(
                        int(plan.max_batches) - successful_steps,
                        0,
                    ),
                    "bubble/training_quota_topup_elapsed_s": execution.elapsed_sec,
                    "bubble/training_quota_force_complete_completed": int(
                        topup_completed
                        and plan.reason == "quota_forced_completion_ready"
                    ),
                }
            )
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
