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
"""Runtime lifecycle state for drafter training execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from verl_speco.trainer.scheduler.schedule_types import TrainingPlan


class DrafterRuntimeStatus(IntEnum):
    """Stable numeric states suitable for tracker metrics."""

    IDLE = 0
    SUBMITTED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


@dataclass
class DrafterRuntimeState:
    """Mutable execution state kept separate from pure scheduling decisions."""

    status: DrafterRuntimeStatus = DrafterRuntimeStatus.IDLE
    active_plan: TrainingPlan | None = None
    training_ref: Any = None
    started_at: float | None = None
    completed_batches: int = 0
    last_batch_duration_sec: float | None = None
    last_error: str | None = None

    def submit(
        self,
        plan: TrainingPlan,
        *,
        started_at: float,
        training_ref: Any = None,
    ) -> None:
        if self.status is not DrafterRuntimeStatus.IDLE:
            raise RuntimeError(
                f"Cannot submit drafter training while runtime is {self.status.name}"
            )
        if not plan.launch:
            raise ValueError("Cannot submit an inactive drafter training plan")
        self.status = DrafterRuntimeStatus.SUBMITTED
        self.active_plan = plan
        self.training_ref = training_ref
        self.started_at = started_at
        self.completed_batches = 0
        self.last_error = None

    def mark_running(self) -> None:
        self._require(DrafterRuntimeStatus.SUBMITTED)
        self.status = DrafterRuntimeStatus.RUNNING

    def mark_completed(self, *, completed_batches: int, elapsed_sec: float) -> None:
        self._require(DrafterRuntimeStatus.RUNNING)
        self.status = DrafterRuntimeStatus.COMPLETED
        self.completed_batches = max(int(completed_batches), 0)
        if self.completed_batches > 0:
            self.last_batch_duration_sec = (
                max(float(elapsed_sec), 0.0) / self.completed_batches
            )

    def mark_failed(self, error: BaseException | str) -> None:
        if self.status not in {
            DrafterRuntimeStatus.SUBMITTED,
            DrafterRuntimeStatus.RUNNING,
        }:
            raise RuntimeError(
                f"Cannot fail drafter training while runtime is {self.status.name}"
            )
        self.status = DrafterRuntimeStatus.FAILED
        self.last_error = str(error)

    def metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "drafter/runtime_state": int(self.status),
            "drafter/runtime_completed_batches": self.completed_batches,
            "drafter/runtime_inflight": int(
                self.status
                in {DrafterRuntimeStatus.SUBMITTED, DrafterRuntimeStatus.RUNNING}
            ),
        }
        if self.last_batch_duration_sec is not None:
            metrics["timing_s/drafter_runtime_batch_average"] = (
                self.last_batch_duration_sec
            )
        return metrics

    def reset(self) -> None:
        if self.status in {
            DrafterRuntimeStatus.SUBMITTED,
            DrafterRuntimeStatus.RUNNING,
        }:
            raise RuntimeError(
                f"Cannot reset drafter runtime while it is {self.status.name}"
            )
        self.status = DrafterRuntimeStatus.IDLE
        self.active_plan = None
        self.training_ref = None
        self.started_at = None
        self.completed_batches = 0
        self.last_error = None

    def _require(self, expected: DrafterRuntimeStatus) -> None:
        if self.status is not expected:
            raise RuntimeError(
                f"Expected drafter runtime {expected.name}, got {self.status.name}"
            )
