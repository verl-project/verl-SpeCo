# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Worker execution boundary for drafter scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from verl_speco.trainer.scheduler.schedule_types import TrainingDataStatus, TrainingPlan


class DrafterWorkerExecutor(Protocol):
    """Submit plans to drafter workers and resolve their results.

    The scheduler depends on this interface instead of Ray or a concrete worker
    group.  A future Bubble executor can therefore change resource selection
    and waiting behavior without changing training trigger or budget policies.
    """

    def submit_training(self, plan: TrainingPlan) -> Any: ...

    def resolve_training(self, submission: Any) -> list[Any]: ...

    def poll_training(self, submission: Any) -> tuple[bool, Any]: ...

    def request_reclaim(self, worker_ids: tuple[str, ...]) -> Any: ...

    def resolve(self, submission: Any) -> Any: ...

    def get_training_data_status(
        self,
        *,
        sample_last_n_steps: int,
        require_full_batch: bool,
        worker_ids: tuple[str, ...] | None = None,
        target_version: int | None = None,
    ) -> list[TrainingDataStatus]: ...

    def prepare_training(self, plan: TrainingPlan) -> dict[str, Any]: ...

    def activate_training_workers(self) -> list[Any]: ...

    def preflight_training(self, plan: TrainingPlan) -> list[Any]: ...

    def abort_training_preflight(self, plan: TrainingPlan) -> list[Any]: ...

    def prewarm_training_workers(
        self, worker_ids: tuple[str, ...] | None = None
    ) -> list[Any]: ...


@dataclass(frozen=True)
class CallbackDrafterWorkerExecutor:
    """Adapter for the existing worker-group RPC and result resolver."""

    submit: Callable[[dict[str, object]], Any]
    resolve: Callable[[Any], Any]
    inspect_data: Callable[..., Any]
    prepare: Callable[[TrainingPlan], dict[str, Any]]
    activate: Callable[[], Any]
    preflight: Callable[[dict[str, object]], Any]
    abort_preflight: Callable[[str], Any]
    poll: Callable[[Any], Any] | None = None
    reclaim: Callable[[tuple[str, ...]], Any] | None = None
    prewarm: Callable[[tuple[str, ...] | None], Any] | None = None

    def submit_training(self, plan: TrainingPlan) -> Any:
        return self.submit(plan.to_worker_payload())

    def resolve_training(self, submission: Any) -> list[Any]:
        results = self.resolve(submission) or []
        if isinstance(results, list):
            return results
        return [results]

    def poll_training(self, submission: Any) -> tuple[bool, Any]:
        if self.poll is None:
            return (False, submission)
        result = self.poll(submission)
        if isinstance(result, tuple) and len(result) == 2:
            return (bool(result[0]), result[1])
        if isinstance(result, dict):
            return (
                bool(result.get("ready", False)),
                result.get("submission", submission),
            )
        return (bool(result), submission)

    def request_reclaim(self, worker_ids: tuple[str, ...]) -> Any:
        if self.reclaim is None:
            return None
        return self.reclaim(worker_ids)

    def get_training_data_status(
        self,
        *,
        sample_last_n_steps: int,
        require_full_batch: bool,
        worker_ids: tuple[str, ...] | None = None,
        target_version: int | None = None,
    ) -> list[TrainingDataStatus]:
        try:
            inspection = self.inspect_data(
                sample_last_n_steps,
                require_full_batch,
                worker_ids,
                target_version,
            )
        except TypeError:
            try:
                inspection = self.inspect_data(
                    sample_last_n_steps, require_full_batch, worker_ids
                )
            except TypeError:
                inspection = self.inspect_data(sample_last_n_steps, require_full_batch)
        results = self.resolve(inspection) or []
        if not isinstance(results, list):
            results = [results]
        statuses = [
            TrainingDataStatus.from_mapping(result)
            for result in results
            if isinstance(result, dict) and bool(result.get("available", False))
        ]
        if worker_ids is None:
            return statuses
        expected = {str(worker_id) for worker_id in worker_ids}
        return [status for status in statuses if str(status.worker_id) in expected]

    def prepare_training(self, plan: TrainingPlan) -> dict[str, Any]:
        return self.prepare(plan)

    def activate_training_workers(self) -> list[Any]:
        results = self.resolve(self.activate()) or []
        return results if isinstance(results, list) else [results]

    def prewarm_training_workers(
        self, worker_ids: tuple[str, ...] | None = None
    ) -> list[Any]:
        if self.prewarm is None:
            return self.activate_training_workers()
        results = self.resolve(self.prewarm(worker_ids)) or []
        return results if isinstance(results, list) else [results]

    def preflight_training(self, plan: TrainingPlan) -> list[Any]:
        results = self.resolve(self.preflight(plan.to_worker_payload())) or []
        return results if isinstance(results, list) else [results]

    def abort_training_preflight(self, plan: TrainingPlan) -> list[Any]:
        results = self.resolve(self.abort_preflight(plan.plan_id)) or []
        return results if isinstance(results, list) else [results]
