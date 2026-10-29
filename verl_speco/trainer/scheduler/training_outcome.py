# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Normalized multi-worker outcome for one drafter training event."""

from __future__ import annotations

from dataclasses import dataclass
import logging
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

logger = logging.getLogger(__name__)


def _metric_float(value: object) -> float | None:
    try:
        return _as_float(value)
    except (TypeError, ValueError):
        return None


def _forward_residency_metrics(results: list[dict[str, object]]) -> dict[str, float]:
    """Maximum host span per stage; individual maxima need not share a rank."""
    import math

    metrics: dict[str, float] = {}
    for result in results:
        for key, raw_value in result.items():
            if not isinstance(key, str) or not key.startswith(
                (
                    "timing_s/drafter_activation_",
                    "timing_s/drafter_cleanup_",
                    "timing_s/drafter_gpu_",
                    "drafter/gpu_",
                )
            ):
                continue
            value = _metric_float(raw_value)
            if value is not None and math.isfinite(value) and value >= 0:
                metrics[key] = max(metrics.get(key, 0.0), value)
    return metrics


# Per-block drafter training scalars that SpecoWorker.get_training_metrics()
# reports in each worker result (e.g. dspark/confidence_loss). The summary dict
# built by TrainingOutcome.from_execution is a fixed whitelist of drafter/*
# keys, so these "<block_prefix>/<suffix>" scalars are forwarded separately to
# keep them observable in co-train logs and TensorBoard.
_BLOCK_METRIC_PASSTHROUGH_SUFFIXES = frozenset(
    {
        "confidence_loss",
        "confidence_accept_rate",
        "confidence_pred_mean",
        "confidence_weighted_token_count",
        "ce_loss",
        "l1_loss",
        "accuracy",
    }
)


def _forward_block_metrics(
    results: list[dict[str, object]],
) -> dict[str, float]:
    """Mean of whitelisted ``<prefix>/<suffix>`` scalars across worker results.

    Keys are kept verbatim (including their block prefix) so DSpark's
    ``dspark/confidence_*`` and DFlash's ``dflash/...`` metrics retain their
    namespace. Non-numeric values are skipped.
    """
    collected: dict[str, list[float]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        for key, raw_value in result.items():
            if not isinstance(key, str) or "/" not in key:
                continue
            if key.rsplit("/", 1)[-1] not in _BLOCK_METRIC_PASSTHROUGH_SUFFIXES:
                continue
            value = _metric_float(raw_value)
            if value is not None:
                collected.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in collected.items()}


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
    ) -> TrainingOutcome:
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
            and not plan.publish_snapshot_deferred
            and trained
            and successful_steps >= int(plan.max_batches)
        )
        publish_snapshot_consistent = not publish_snapshot_required or (
            len(publish_leaders) == 1 and publish_leaders[0].snapshot_ready
        )
        if (
            plan.publish_snapshot_deferred
            and trained
            and successful_steps >= int(plan.max_batches)
        ):
            publish_snapshot_consistent = all(
                bool(result.get("publish_snapshot_deferred", False))
                for result in normalized_results
                if result.get("triggered", False)
            ) and bool(participating_results)
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
            logger.error(
                "[BubbleTime] training_result_rejected: plan_id=%s execution_reason=%s "
                "launched=%s expected_workers=%s workers=%s",
                plan.plan_id,
                execution.reason,
                execution.launched,
                sorted(expected_worker_ids),
                [
                    {
                        key: result.get(key)
                        for key in (
                            "worker_id",
                            "participating",
                            "triggered",
                            "ready",
                            "reason",
                            "plan_id",
                            "source_global_step",
                            "worker_global_step",
                            "data_version",
                            "target_version",
                            "successful_steps",
                            "optimizer_step",
                        )
                    }
                    for result in normalized_results
                ],
            )
        quota_topup = plan.reason in {
            "quota_topup_training_ready",
            "quota_forced_completion_ready",
        }
        idle_training = bool(
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and not quota_topup
        )
        metrics: dict[str, float | int] = {
            **{
                f"bubble/{name}": int(
                    any(
                        result.get(name, False)
                        for result in normalized_results
                        if result.get("triggered", False)
                    )
                )
                for name in ("reclaim_received_via_bus", "reclaim_control_error")
            },
            **{
                f"timing_s/drafter_{name}": max(
                    (
                        (_metric_float(result.get(f"{name}_sec")) or 0.0)
                        for result in normalized_results
                        if result.get("triggered", False)
                    ),
                    default=0.0,
                )
                for name in (
                    "reclaim_receive_to_stop",
                    "reclaim_receive_to_cleanup",
                    "idle_control_yield_elapsed",
                )
            },
            "timing_s/drafter_idle_control": max(
                (
                    (_metric_float(result.get("idle_control_elapsed_sec")) or 0.0)
                    for result in normalized_results
                    if result.get("triggered", False)
                ),
                default=0.0,
            ),
            "bubble/activation_started_resident": int(
                bool(participating_results)
                and all(
                    result.get("activation_started_resident", False)
                    for result in normalized_results
                    if result.get("triggered", False)
                )
            ),
            "drafter/trained": int(trained),
            "drafter/trained_any": int(trained),
            "drafter/idle_trained": int(idle_training and trained),
            "bubble/idle_optimizer_steps": successful_steps
            if idle_training and trained
            else 0,
            "bubble/critical_path_topup_optimizer_steps": successful_steps
            if quota_topup and trained
            else 0,
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
        # Surface per-block drafter training scalars (e.g. dspark/confidence_*)
        # computed by SpecoWorker.get_training_metrics() instead of dropping them.
        for block_key, block_value in _forward_block_metrics(
            normalized_results
        ).items():
            metrics.setdefault(block_key, block_value)

        for key in (
            "timing_s/drafter_prepare_batch",
            "timing_s/drafter_forward_loss",
            "timing_s/drafter_reduce_loss",
            "timing_s/drafter_backward",
            "timing_s/drafter_optimizer",
            "timing_s/drafter_publish_snapshot",
            "activation_elapsed_sec",
            "target_head_prepare_elapsed_sec",
            "model_restore_elapsed_sec",
            "optimizer_restore_elapsed_sec",
            "preflight_elapsed_sec",
            "preflight_to_first_batch_sec",
            "preflight_to_stop_sec",
            "training_loop_elapsed_sec",
            "cleanup_elapsed_sec",
            "cleanup_nonpark_elapsed_sec",
            "memory_park_elapsed_sec",
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
                    "target_head_prepare_elapsed_sec": "timing_s/drafter_worker_target_head_prepare",
                    "model_restore_elapsed_sec": "timing_s/drafter_worker_model_restore",
                    "optimizer_restore_elapsed_sec": "timing_s/drafter_worker_optimizer_restore",
                    "preflight_elapsed_sec": "timing_s/drafter_worker_preflight",
                    "preflight_to_first_batch_sec": "timing_s/drafter_worker_preflight_to_first_batch",
                    "preflight_to_stop_sec": "timing_s/drafter_worker_preflight_to_stop",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "cleanup_nonpark_elapsed_sec": "timing_s/drafter_worker_cleanup_nonpark",
                    "memory_park_elapsed_sec": "timing_s/drafter_worker_memory_park",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                }.get(key, key)
                metrics[metric_key] = max(values)

        # Preserve drafter-quality metrics emitted by the backend across the
        # scheduler boundary. These values are already reduced inside each
        # trainer worker, so averaging participating workers matches the
        # historical synchronous-training reporting contract.
        block_drafter_quality_metric_keys = {
            f"{prefix}/{metric_name}"
            for prefix in ("dflash", "dspark", "eagle3")
            for metric_name in (
                "accuracy",
                "ce_loss",
                "l1_loss",
                "lk_loss",
                "top1_acc",
                "top5_acc",
                "lk/acceptance",
                "lk/kl_weight",
                "lk/forward_kl",
                "lk/tv",
            )
        }
        block_drafter_position_metric_prefixes = tuple(
            f"{prefix}/{metric_name}/"
            for prefix in ("dflash", "dspark", "eagle3")
            for metric_name in ("accuracy_per_position", "loss_per_position")
        )
        for result in normalized_results:
            for key in result:
                if key in block_drafter_quality_metric_keys or any(
                    key.startswith(prefix) and key.removeprefix(prefix).isdigit()
                    for prefix in block_drafter_position_metric_prefixes
                ):
                    block_drafter_quality_metric_keys.add(key)
        for key in sorted(block_drafter_quality_metric_keys):
            values = [
                value
                for result in normalized_results
                if (value := _metric_float(result.get(key))) is not None
            ]
            if values:
                metrics[key] = sum(values) / len(values)

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
        metrics.update(_forward_residency_metrics(normalized_results))
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
            execution.reason
            if (result_consistent or not execution.launched)
            else "worker_result_inconsistent"
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
