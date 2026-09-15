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
"""Pure scheduling contracts for online drafter training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, cast


def _as_int(value: object) -> int:
    """Convert a value received from configuration or an RPC payload."""

    return int(cast(Any, value))


def _as_float(value: object) -> float:
    """Convert a value received from configuration or an RPC payload."""

    return float(cast(Any, value))


class DrafterExecutionStrategy(str, Enum):
    """Supported drafter-training execution strategies.

    PR 1 intentionally executes only ``SYNC``. ``ROLLOUT_IDLE_WORKER`` is
    reserved in the contract so the later bubble-time implementation can reuse
    the same plan type without changing the released synchronous path.
    """

    SYNC = "sync"
    ROLLOUT_IDLE_WORKER = "rollout_idle_worker"


class DrafterCollectionSource(str, Enum):
    """Target-model feature source used by drafter collection."""

    SGLANG = "sglang"
    OLD_LOGPROB = "oldlogprob"


@dataclass(frozen=True)
class DrafterScheduleConfig:
    """Legacy-compatible scheduling values read from ``training`` config."""

    collect_interval_steps: object = 1
    training_interval_steps: object = 1
    publish_interval_steps: object = 0
    publish_async: bool = False
    use_logits: bool = False
    use_data_buffer: bool = False
    train_batches_per_trigger: int = 100
    collection_sample_rate: float = 1.0
    max_collect_samples_per_replica: int | None = 16
    max_collect_tokens_per_replica: int | None = None
    hidden_window_mode: str = "front"
    hidden_window_tokens_per_sample: int | None = 512
    hidden_window_min_rows: int = 512
    min_trainable_batches: int = 1
    require_full_batch: bool = False
    sample_last_n_steps: int = 2

    @classmethod
    def from_mapping(cls, config) -> "DrafterScheduleConfig":
        config = config or {}
        get = config.get if hasattr(config, "get") else lambda key, default: default
        train_batches = int(get("step", 100))
        return cls(
            collect_interval_steps=get("collect_interval_steps", 1),
            training_interval_steps=get("training_interval_steps", 1),
            publish_interval_steps=get("publish_interval_steps", 0),
            publish_async=bool(get("publish_async", False)),
            use_logits=bool(get("use_logits", False)),
            use_data_buffer=bool(get("use_data_buffer", False)),
            train_batches_per_trigger=train_batches,
            collection_sample_rate=float(get("collection_sample_rate", 1.0) or 0.0),
            max_collect_samples_per_replica=_optional_int(
                get("max_collect_samples_per_step_per_replica", 16)
            ),
            max_collect_tokens_per_replica=_optional_int(
                get("max_collect_tokens_per_step_per_replica", None)
            ),
            hidden_window_mode=str(get("hidden_state_window_mode", "front") or "front"),
            hidden_window_tokens_per_sample=_optional_int(
                get("hidden_state_window_tokens_per_sample", 512)
            ),
            hidden_window_min_rows=int(get("hidden_state_window_min_rows", 512)),
            min_trainable_batches=int(get("min_trainable_batches", 1)),
            require_full_batch=bool(get("require_full_batch", False)),
            sample_last_n_steps=int(get("sample_last_n_steps", 2)),
        )


def _optional_int(value: object) -> int | None:
    return None if value is None else _as_int(value)


@dataclass(frozen=True)
class DrafterCollectionContext:
    """Read-only facts used to decide target-feature collection."""

    global_step: object
    source: DrafterCollectionSource
    drafter_enabled: bool = True
    source_enabled: bool = True
    validation: bool = False
    require_training_interval: bool = False


@dataclass(frozen=True)
class CollectionPlan:
    """Scheduler decision and static budget for one collection step."""

    collect: bool
    reason: str
    source: DrafterCollectionSource
    source_global_step: object
    collect_interval_matched: bool
    training_interval_matched: bool
    sample_rate: float
    max_samples_per_replica: int | None
    max_tokens_per_replica: int | None
    hidden_window_mode: str
    hidden_window_tokens_per_sample: int | None
    hidden_window_min_rows: int
    collection_id: str = ""

    _REASON_CODES: ClassVar[dict[str, int]] = {
        "drafter_disabled": 1,
        "source_disabled": 2,
        "validation": 3,
        "interval_not_reached": 4,
        "training_interval_not_reached": 5,
        "sample_rate_zero": 6,
        "collection_enabled": 7,
    }

    def metrics(self) -> dict[str, float | int]:
        source_code = {
            DrafterCollectionSource.SGLANG: 1,
            DrafterCollectionSource.OLD_LOGPROB: 2,
        }[self.source]
        metrics: dict[str, float | int] = {
            "drafter/collection_plan_used": 1,
            "drafter/collection_plan_collect": int(self.collect),
            "drafter/collection_plan_reason": self._REASON_CODES.get(self.reason, 0),
            "drafter/collection_plan_source": source_code,
            "drafter/collection_plan_interval_matched": int(
                self.collect_interval_matched
            ),
            "drafter/collection_plan_training_interval_matched": int(
                self.training_interval_matched
            ),
        }
        return metrics


@dataclass(frozen=True)
class CollectionPayload:
    """Prepared per-worker samples consumed by a collection executor."""

    source: DrafterCollectionSource
    buckets: list[list[dict]]
    collected_samples: int
    raw_samples: int = 0
    collection_id: str = ""

    @property
    def has_samples(self) -> bool:
        return self.collected_samples > 0 and any(self.buckets)


@dataclass(frozen=True)
class CollectionWorkerResult:
    """Actual collection result reported by one Worker process."""

    worker_id: str
    worker_incarnation: str
    source_global_step: int | None
    accepted_samples: int
    rejected_samples: int
    buffer_version_before: int
    buffer_version_after: int
    data_version: int | None
    collected: bool
    reason: str
    collection_id: str = ""
    staged_samples: int = 0
    expired_stages: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "CollectionWorkerResult":
        return cls(
            worker_id=str(value.get("worker_id", "")),
            worker_incarnation=str(value.get("worker_incarnation", "")),
            source_global_step=_optional_int(value.get("source_global_step")),
            accepted_samples=_as_int(value.get("accepted_samples", 0)),
            rejected_samples=_as_int(value.get("rejected_samples", 0)),
            buffer_version_before=_as_int(value.get("buffer_version_before", 0)),
            buffer_version_after=_as_int(value.get("buffer_version_after", 0)),
            data_version=_optional_int(value.get("data_version")),
            collected=bool(value.get("collected", False)),
            reason=str(value.get("reason", "")),
            collection_id=str(value.get("collection_id", "")),
            staged_samples=_as_int(value.get("staged_samples", 0)),
            expired_stages=_as_int(value.get("expired_stages", 0)),
        )


@dataclass(frozen=True)
class DrafterScheduleContext:
    """Read-only facts used to make the released synchronous decision."""

    global_step: object
    training_mode: str
    collected_samples_this_step: int
    oldlogprob_collection_requested: bool
    data_status: TrainingDataStatus | None = None
    pending_training_count: int = 0


@dataclass(frozen=True)
class TrainingDataStatus:
    """Aggregated worker-side snapshot of currently trainable drafter data."""

    current_step: int
    current_step_samples: int
    buffer_samples: int
    trainable_samples: int
    trainable_batches: int
    batch_size_per_gpu: int
    partial_batch_available: bool
    oldest_sample_step: int | None
    newest_sample_step: int | None
    same_step_data_required: bool
    target_version: int | None = None
    target_version_consistent: bool = True
    data_version: int | None = None
    data_version_consistent: bool = True
    buffer_version: int = 0
    worker_incarnation: str = ""
    worker_id: str = ""
    worker_snapshots: dict[str, dict[str, object]] | None = None
    min_sample_step: int | None = None
    max_sample_step: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "TrainingDataStatus":
        return cls(
            current_step=_as_int(value.get("current_step", 0)),
            current_step_samples=_as_int(value.get("current_step_samples", 0)),
            buffer_samples=_as_int(value.get("buffer_samples", 0)),
            trainable_samples=_as_int(value.get("trainable_samples", 0)),
            trainable_batches=_as_int(value.get("trainable_batches", 0)),
            batch_size_per_gpu=_as_int(value.get("batch_size_per_gpu", 1)),
            partial_batch_available=bool(value.get("partial_batch_available", False)),
            oldest_sample_step=_optional_int(value.get("oldest_sample_step")),
            newest_sample_step=_optional_int(value.get("newest_sample_step")),
            same_step_data_required=bool(value.get("same_step_data_required", False)),
            target_version=_optional_int(value.get("target_version")),
            target_version_consistent=bool(
                value.get("target_version_consistent", True)
            ),
            data_version=_optional_int(
                value.get("data_version", value.get("newest_sample_step"))
            ),
            data_version_consistent=bool(value.get("data_version_consistent", True)),
            buffer_version=_as_int(value.get("buffer_version", 0)),
            worker_incarnation=str(value.get("worker_incarnation", "")),
            worker_id=str(value.get("worker_id", value.get("rank", ""))),
            min_sample_step=_optional_int(value.get("min_sample_step")),
            max_sample_step=_optional_int(value.get("max_sample_step")),
        )

    def metrics(self) -> dict[str, int]:
        return {
            "drafter/data_current_step_samples": self.current_step_samples,
            "drafter/data_buffer_samples": self.buffer_samples,
            "drafter/data_trainable_samples": self.trainable_samples,
            "drafter/data_trainable_batches": self.trainable_batches,
            "drafter/data_partial_batch_available": int(self.partial_batch_available),
            "drafter/data_same_step_required": int(self.same_step_data_required),
            "drafter/data_target_version_consistent": int(
                self.target_version_consistent
            ),
            "drafter/data_version_consistent": int(self.data_version_consistent),
        }


@dataclass(frozen=True)
class TriggerDecision:
    should_train: bool
    reason: str


@dataclass(frozen=True)
class TrainingBudget:
    max_batches: int
    min_batches: int
    deadline_ts: float | None
    require_full_batch: bool
    sample_last_n_steps: int
    reason: str


@dataclass(frozen=True)
class TrainingPlan:
    """The synchronous training decision produced by ``DrafterScheduler``."""

    launch: bool
    reason: str
    interval_matched: bool
    execution_strategy: DrafterExecutionStrategy
    source_global_step: object
    max_batches: int
    publish_after_success: bool
    min_batches: int = 1
    deadline_ts: float | None = None
    require_full_batch: bool = False
    sample_last_n_steps: int = 2
    data_version: int | None = None
    required_target_version: int | None = None
    min_sample_step: int | None = None
    max_sample_step: int | None = None
    data_filter_reason: str = ""
    plan_id: str = ""
    worker_snapshots: dict[str, dict[str, object]] | None = None

    _REASON_CODES: ClassVar[dict[str, int]] = {
        "collect_only": 1,
        "interval_not_reached": 2,
        "current_step_samples": 3,
        "no_current_step_oldlogprob_samples": 4,
        "data_buffer_enabled": 5,
        "no_current_step_samples": 6,
        "pending_training": 7,
        "missing_data_status": 8,
        "no_trainable_batch": 9,
        "training_ready": 10,
        "no_training_budget": 11,
        "insufficient_training_budget": 12,
        "inconsistent_target_version": 13,
        "inconsistent_data_version": 14,
        "worker_preflight_failed": 15,
    }

    def to_worker_payload(self) -> dict[str, object]:
        """Serialize the scheduler decision for the drafter worker RPC."""

        return {
            "launch": self.launch,
            "execution_strategy": self.execution_strategy.value,
            "source_global_step": self.source_global_step,
            "max_batches": self.max_batches,
            "publish_after_success": self.publish_after_success,
            "min_batches": self.min_batches,
            "deadline_ts": self.deadline_ts,
            "require_full_batch": self.require_full_batch,
            "sample_last_n_steps": self.sample_last_n_steps,
            "data_version": self.data_version,
            "required_target_version": self.required_target_version,
            "min_sample_step": self.min_sample_step,
            "max_sample_step": self.max_sample_step,
            "data_filter_reason": self.data_filter_reason,
            "plan_id": self.plan_id,
            "worker_snapshots": self.worker_snapshots or {},
        }

    def metrics(self) -> dict[str, int]:
        """Return numeric observability fields accepted by metric backends."""

        strategy_code = {
            DrafterExecutionStrategy.SYNC: 0,
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER: 1,
        }[self.execution_strategy]
        metrics = {
            "drafter/scheduler_used": 1,
            "drafter/schedule_launch": int(self.launch),
            "drafter/schedule_interval_matched": int(self.interval_matched),
            "drafter/schedule_strategy": strategy_code,
            "drafter/schedule_reason": self._REASON_CODES.get(self.reason, 0),
        }
        return metrics


@dataclass(frozen=True)
class PublishPlan:
    """Whether the already-released synchronous path should publish weights."""

    publish: bool
    reason: str
    interval_matched: bool
    source_global_step: object
    asynchronous: bool = False


@dataclass(frozen=True)
class TrainingResult:
    """Normalized outcome of executing one drafter training plan."""

    source_global_step: int
    execution_strategy: DrafterExecutionStrategy
    attempted_batches: int
    successful_steps: int
    optimizer_step: int
    buffer_size_before: int
    buffer_size_after: int
    elapsed_sec: float
    reason: str
    snapshot_ready: bool
    trained: bool = False
    worker_id: str = ""
    worker_incarnation: str = ""
    plan_id: str = ""
    data_version: int | None = None
    target_version: int | None = None
    is_publish_leader: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "TrainingResult":
        return cls(
            source_global_step=_as_int(value.get("source_global_step", 0)),
            execution_strategy=DrafterExecutionStrategy(
                value.get("execution_strategy", DrafterExecutionStrategy.SYNC.value)
            ),
            attempted_batches=_as_int(value.get("attempted_steps", 0)),
            successful_steps=_as_int(value.get("successful_steps", 0)),
            optimizer_step=_as_int(value.get("optimizer_step", 0)),
            buffer_size_before=_as_int(value.get("buffer_size_before", 0)),
            buffer_size_after=_as_int(value.get("buffer_size_after", 0)),
            elapsed_sec=_as_float(value.get("elapsed_sec", 0.0)),
            reason=str(value.get("reason", "")),
            snapshot_ready=bool(value.get("publish_snapshot_cached", False)),
            trained=bool(value.get("trained", False)),
            worker_id=str(value.get("worker_id", value.get("rank", ""))),
            worker_incarnation=str(value.get("worker_incarnation", "")),
            plan_id=str(value.get("plan_id", "")),
            data_version=_optional_int(value.get("data_version")),
            target_version=_optional_int(value.get("target_version")),
            is_publish_leader=bool(value.get("is_publish_leader", False)),
        )
