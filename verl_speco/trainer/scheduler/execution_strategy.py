# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Execution strategies for drafter training plans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.schedule_types import TrainingPlan
from verl_speco.trainer.scheduler.worker_executor import DrafterWorkerExecutor


@dataclass(frozen=True)
class ExecutionOutcome:
    raw_results: list[Any]
    elapsed_sec: float
    launched: bool = True
    reason: str = "completed"


class DrafterTrainingExecutionStrategy(Protocol):
    def execute(
        self,
        plan: TrainingPlan,
        *,
        executor: DrafterWorkerExecutor,
        runtime_state: DrafterRuntimeState,
    ) -> ExecutionOutcome: ...


class SyncExecutionStrategy:
    """Submit a training plan and synchronously wait for all worker results."""

    def execute(
        self,
        plan: TrainingPlan,
        *,
        executor: DrafterWorkerExecutor,
        runtime_state: DrafterRuntimeState,
    ) -> ExecutionOutcome:
        if not plan.launch:
            raise ValueError("Cannot execute an inactive drafter training plan")
        if runtime_state.status in {
            DrafterRuntimeStatus.COMPLETED,
            DrafterRuntimeStatus.FAILED,
        }:
            runtime_state.reset()

        started_at = time.perf_counter()
        runtime_state.submit(plan, started_at=started_at)
        try:
            try:
                readiness = executor.preflight_training(plan)
            except Exception:
                executor.abort_training_preflight(plan)
                raise
            participants = [
                result
                for result in readiness
                if isinstance(result, dict) and result.get("participating", False)
            ]
            expected_worker_ids = set((plan.worker_snapshots or {}).keys())
            ready_worker_ids = {
                str(result.get("worker_id", result.get("rank", "")))
                for result in participants
                if bool(result.get("ready", False))
            }
            all_expected_workers_ready = bool(expected_worker_ids) and (
                ready_worker_ids == expected_worker_ids
                and len(participants) == len(expected_worker_ids)
                and all(bool(result.get("ready", False)) for result in participants)
                and all(
                    result.get("data_version") == plan.data_version
                    for result in participants
                )
                and all(
                    plan.required_target_version is None
                    or result.get("target_version") == plan.required_target_version
                    for result in participants
                )
            )
            if not all_expected_workers_ready:
                executor.abort_training_preflight(plan)
                return ExecutionOutcome(
                    raw_results=readiness,
                    elapsed_sec=time.perf_counter() - started_at,
                    launched=False,
                    reason="worker_preflight_failed",
                )
            runtime_state.mark_running()
            submission = executor.submit_training(plan)
            results = executor.resolve_training(submission)
        except Exception as error:
            runtime_state.mark_failed(error)
            raise
        elapsed_sec = time.perf_counter() - started_at
        return ExecutionOutcome(raw_results=results, elapsed_sec=elapsed_sec)
