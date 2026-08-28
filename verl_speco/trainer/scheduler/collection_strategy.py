# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Execution strategies for drafter sample collection."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import ClassVar, Protocol

from verl_speco.trainer.scheduler.collection_executor import (
    DrafterCollectionExecutor,
)
from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPayload,
    CollectionPlan,
    CollectionWorkerResult,
    _as_int,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionOutcome:
    attempted: bool
    collected: bool
    reason: str
    source_global_step: object
    collected_samples: int = 0
    raw_samples: int = 0
    elapsed_sec: float = 0.0
    worker_results: list[CollectionWorkerResult] | None = None
    finalized: bool = False

    _REASON_CODES: ClassVar[dict[str, int]] = {
        "collection_completed": 1,
        "empty_collection_payload": 2,
        "collection_result_incomplete": 3,
        "collection_worker_rejected_samples": 4,
        "collection_version_mismatch": 5,
        "collection_stage_failed": 6,
        "collection_id_mismatch": 7,
        "collection_finalize_failed": 8,
    }

    def metrics(self) -> dict[str, float | int]:
        worker_results = self.worker_results or []
        return {
            "drafter/collection_attempted": int(self.attempted),
            "drafter/collection_completed": int(self.collected),
            "drafter/collection_outcome_reason": self._REASON_CODES.get(self.reason, 0),
            "drafter/collected_samples": self.collected_samples,
            "drafter/collection_rejected_samples": sum(
                result.rejected_samples for result in worker_results
            ),
            "drafter/collection_worker_results": len(worker_results),
            "drafter/collection_finalized": int(self.finalized),
            "drafter/collection_expired_stages": sum(
                result.expired_stages for result in worker_results
            ),
            "drafter/raw_drafter_samples": self.raw_samples,
            "timing_s/drafter_collection_rpc": self.elapsed_sec,
        }


class DrafterCollectionStrategy(Protocol):
    def execute(
        self,
        plan: CollectionPlan,
        payload: CollectionPayload,
        *,
        executor: DrafterCollectionExecutor,
    ) -> CollectionOutcome: ...


class SyncCollectionStrategy:
    """Set the collection step and synchronously populate Worker buffers."""

    def execute(
        self,
        plan: CollectionPlan,
        payload: CollectionPayload,
        *,
        executor: DrafterCollectionExecutor,
    ) -> CollectionOutcome:
        if payload.source is not plan.source:
            raise ValueError(
                f"Collection payload source {payload.source.value} does not match "
                f"plan source {plan.source.value}"
            )
        if not plan.collection_id or payload.collection_id != plan.collection_id:
            raise ValueError("Collection payload collection_id does not match its plan")
        if not plan.collect:
            return CollectionOutcome(
                attempted=False,
                collected=False,
                reason=plan.reason,
                source_global_step=plan.source_global_step,
                raw_samples=payload.raw_samples,
            )
        if not payload.has_samples:
            return CollectionOutcome(
                attempted=False,
                collected=False,
                reason="empty_collection_payload",
                source_global_step=plan.source_global_step,
                raw_samples=payload.raw_samples,
            )

        started_at = time.perf_counter()
        executor.set_global_step(plan.source_global_step)
        try:
            staged = executor.stage(payload)
        except Exception:
            executor.abort(payload)
            raise

        expected_nonempty_by_owner = {
            str(owner): len(bucket)
            for owner, bucket in enumerate(payload.buckets)
            if bucket
        }
        staged_by_owner = {result.worker_id: result.staged_samples for result in staged}
        stage_ids = [result.worker_id for result in staged]
        routing_complete = all(
            staged_by_owner.get(owner) == expected
            for owner, expected in expected_nonempty_by_owner.items()
        ) and all(
            result.worker_id in expected_nonempty_by_owner or result.staged_samples == 0
            for result in staged
        )
        stage_valid = (
            len(stage_ids) == len(set(stage_ids))
            and all(
                result.collection_id == plan.collection_id
                and result.worker_incarnation
                and result.source_global_step == _as_int(plan.source_global_step)
                and result.buffer_version_after == result.buffer_version_before
                and result.reason == "collection_staged"
                for result in staged
            )
            and routing_complete
        )
        if not stage_valid:
            executor.abort(payload)
            return CollectionOutcome(
                attempted=True,
                collected=False,
                reason="collection_stage_failed",
                source_global_step=plan.source_global_step,
                raw_samples=payload.raw_samples,
                elapsed_sec=time.perf_counter() - started_at,
                worker_results=staged,
            )

        try:
            results = executor.commit(payload)
        except Exception:
            executor.rollback(payload)
            raise
        accepted_samples = sum(result.accepted_samples for result in results)
        rejected_samples = sum(result.rejected_samples for result in results)
        result_steps = {result.source_global_step for result in results}
        data_versions = {result.data_version for result in results}
        worker_ids = [result.worker_id for result in results]
        expected_nonempty_by_owner = {
            str(owner): len(bucket)
            for owner, bucket in enumerate(payload.buckets)
            if bucket
        }
        accepted_by_owner = {
            result.worker_id: result.accepted_samples for result in results
        }
        identities_valid = all(
            result.worker_id and result.worker_incarnation for result in results
        ) and len(worker_ids) == len(set(worker_ids))
        routing_complete = all(
            accepted_by_owner.get(owner) == expected
            for owner, expected in expected_nonempty_by_owner.items()
        ) and all(
            result.worker_id in expected_nonempty_by_owner
            or result.accepted_samples == 0
            for result in results
        )
        collection_ids_valid = all(
            result.collection_id == plan.collection_id for result in results
        )
        versions_valid = (
            result_steps == {_as_int(plan.source_global_step)}
            and data_versions == {_as_int(plan.source_global_step)}
            and all(
                result.buffer_version_after >= result.buffer_version_before
                for result in results
            )
        )
        complete = (
            bool(results)
            and identities_valid
            and collection_ids_valid
            and routing_complete
            and accepted_samples == payload.collected_samples
            and all(result.collected for result in results)
        )
        if not collection_ids_valid:
            reason = "collection_id_mismatch"
        elif rejected_samples > 0:
            reason = "collection_worker_rejected_samples"
        elif not complete:
            reason = "collection_result_incomplete"
        elif not versions_valid:
            reason = "collection_version_mismatch"
        else:
            reason = "collection_completed"
        finalized = False
        if reason == "collection_completed":
            try:
                executor.finalize(payload)
                finalized = True
            except Exception:
                logger.exception(
                    "Failed to finalize committed drafter collection %s",
                    plan.collection_id,
                )
                executor.rollback(payload)
                reason = "collection_finalize_failed"
        else:
            executor.rollback(payload)
        return CollectionOutcome(
            attempted=True,
            collected=reason == "collection_completed",
            reason=reason,
            source_global_step=plan.source_global_step,
            collected_samples=(
                accepted_samples if reason == "collection_completed" else 0
            ),
            raw_samples=payload.raw_samples,
            elapsed_sec=time.perf_counter() - started_at,
            worker_results=results,
            finalized=finalized,
        )
