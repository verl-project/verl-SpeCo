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

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
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


@dataclass
class PendingStandalonePlan:
    """Metadata and execution state of one dispatched standalone global batch."""

    plan_id: int
    keys: tuple[str, ...]
    runtime: DrafterRuntimeState
    entries: tuple[Any, ...] = ()
    receipt_token: str | None = None
    materialized: bool = False
    optimizer_applied: bool = False
    acknowledged_ranks: set[int] = field(default_factory=set)
    payload_released: bool = False


class StandaloneQueueState:
    """Confirmed keys, reservations and FIFO plans; contains no transport calls."""

    def __init__(self, capacity: int, batch_size: int):
        if capacity < 1 or batch_size < 1:
            raise ValueError("Standalone plan capacity and batch size must be positive")
        self.capacity = capacity
        self.batch_size = batch_size
        self.ready: deque[Any] = deque()
        self.pending: deque[PendingStandalonePlan] = deque()
        self.reserved: set[str] = set()
        self.next_id = 0
        self.completed_id = -1
        self.last_completed_keys: tuple[str, ...] = ()
        self._completion_history: dict[int, tuple[bytes, str | None]] = {}
        self._completion_history_limit = 4096

    @property
    def has_capacity(self) -> bool:
        return len(self.pending) < self.capacity

    def refresh(self, entries: Sequence[Any]) -> None:
        # Successful clear removes completed keys before refresh. The snapshot
        # replaces this cache, never accumulates duplicate entries across lists.
        unique = {
            entry.key: entry for entry in entries if entry.key not in self.reserved
        }
        self.ready = deque(
            sorted(
                unique.values(),
                key=lambda entry: (int(entry.tag["sequence_no"]), entry.key),
            )
        )

    def reserve(self, *, size: int | None = None) -> tuple[PendingStandalonePlan, list[Any]]:
        count = self.batch_size if size is None else size
        if type(count) is not int or count < 1 or count > self.batch_size:
            raise ValueError("Standalone reservation size must fit one global batch")
        if not self.has_capacity or len(self.ready) < count:
            raise RuntimeError(
                "Cannot reserve an incomplete batch or exceed plan capacity"
            )
        entries = [self.ready.popleft() for _ in range(count)]
        keys = tuple(entry.key for entry in entries)
        pending = PendingStandalonePlan(
            self.next_id, keys, DrafterRuntimeState(), tuple(entries)
        )
        self.next_id += 1
        self.pending.append(pending)
        self.reserved.update(keys)
        return pending, entries

    def mark_rank_materialized(self, plan_id: int, receipt_token: str, rank: int, world_size: int) -> bool:
        if (type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str)
                or not receipt_token or type(rank) is not int or type(world_size) is not int
                or not 0 <= rank < world_size):
            raise RuntimeError("Invalid rank materialization acknowledgement")
        if plan_id <= self.completed_id:
            previous = self._completion_history.get(plan_id)
            if previous is None or previous[1] != receipt_token:
                raise RuntimeError("Expired or conflicting rank materialization acknowledgement")
            return False
        pending = next((item for item in self.pending if item.plan_id == plan_id), None)
        if pending is None or pending.receipt_token != receipt_token:
            raise RuntimeError("Rank materialization acknowledgement does not match a pending plan")
        if rank in pending.acknowledged_ranks:
            return False
        if pending.materialized:
            raise RuntimeError("Rank materialization acknowledgement arrived after aggregate acknowledgement")
        pending.acknowledged_ranks.add(rank)
        return True

    def mark_materialized(self, plan_id: int, receipt_token: str, *, required_ranks: int = 0) -> bool:
        """Accept an aggregate all-rank materialization acknowledgement once."""
        if type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str) or not receipt_token:
            raise RuntimeError("Invalid reference materialization acknowledgement")
        if plan_id <= self.completed_id:
            previous = self._completion_history.get(plan_id)
            if previous is None or previous[1] != receipt_token:
                raise RuntimeError("Expired or conflicting reference materialization acknowledgement")
            return False
        if not self.pending:
            raise RuntimeError("Reference materialization acknowledgement without a pending plan")
        pending = next((item for item in self.pending if item.plan_id == plan_id), None)
        if pending is None or pending.receipt_token != receipt_token:
            raise RuntimeError("Reference materialization acknowledgement does not match a pending plan")
        if pending.materialized:
            return False
        if required_ranks and pending.acknowledged_ranks != set(range(required_ranks)):
            raise RuntimeError("Reference materialization requires every rank acknowledgement")
        pending.materialized = True
        return True

    def mark_payload_released(self, pending: PendingStandalonePlan) -> None:
        if not any(item is pending for item in self.pending) or not pending.materialized:
            raise RuntimeError("Payload release requires a pending materialized plan")
        pending.payload_released = True

    def mark_applied(self, plan_id: int, receipt_token: str) -> bool:
        """Record an all-rank optimizer step before releasing references."""
        if type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str) or not receipt_token:
            raise RuntimeError("Invalid reference optimizer acknowledgement")
        if plan_id <= self.completed_id:
            previous = self._completion_history.get(plan_id)
            if previous is None or previous[1] != receipt_token:
                raise RuntimeError("Expired or conflicting reference optimizer acknowledgement")
            return False
        pending = next((item for item in self.pending if item.plan_id == plan_id), None)
        if pending is None or pending.receipt_token != receipt_token:
            raise RuntimeError("Reference optimizer acknowledgement does not match a pending plan")
        if not pending.materialized:
            raise RuntimeError("Reference optimizer acknowledgement before materialization acknowledgement")
        if pending.optimizer_applied:
            return False
        pending.optimizer_applied = True
        return True

    def completion(
        self, plan_id: int | None, keys: Sequence[str], *, receipt_token: str | None = None
    ) -> PendingStandalonePlan | None:
        actual = tuple(keys)
        if plan_id is not None and (type(plan_id) is not int or plan_id < 0):
            raise RuntimeError("Invalid TQ completion plan ID")
        if plan_id is not None and plan_id <= self.completed_id:
            previous = self._completion_history.get(plan_id)
            if previous is None:
                raise RuntimeError("Expired TQ completion cannot be verified")
            if previous != (self._completion_fingerprint(actual), receipt_token):
                raise RuntimeError("Conflicting duplicate TQ completion keys or receipt token")
            return None
        if not self.pending:
            raise RuntimeError("TQ completion without an outstanding plan")
        pending = self.pending[0]
        if (
            plan_id is not None and plan_id != pending.plan_id
        ) or actual != pending.keys or receipt_token != pending.receipt_token:
            raise RuntimeError("TQ completion does not match the FIFO plan")
        return pending

    def release(self, pending: PendingStandalonePlan) -> None:
        if not self.pending or self.pending[0] is not pending:
            raise RuntimeError("TQ plans must complete in FIFO order")
        self.pending.popleft()
        self.reserved.difference_update(pending.keys)
        self.completed_id = pending.plan_id
        self.last_completed_keys = pending.keys
        self._completion_history[pending.plan_id] = (self._completion_fingerprint(pending.keys), pending.receipt_token)
        if len(self._completion_history) > self._completion_history_limit:
            del self._completion_history[next(iter(self._completion_history))]

    @staticmethod
    def _completion_fingerprint(keys: Sequence[str]) -> bytes:
        digest = hashlib.sha256()
        for key in keys:
            if not isinstance(key, str):
                raise RuntimeError("Invalid TQ completion key")
            encoded = key.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.digest()
