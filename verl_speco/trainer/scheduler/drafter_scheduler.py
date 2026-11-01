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
"""Legacy-equivalent synchronous drafter scheduling.

This module is the single decision source for synchronous collection, training,
batch limits, and publication. Trainers and runtimes request plans; workers only
execute the serialized plan. The blocking training behavior and call ordering
remain unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Sequence, cast
from uuid import uuid4

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPlan,
    CollectionPayload,
    AvailableTrainingResources,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterTrainingDataSource,
    IdleWindowConfidence,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    ProducerAction,
    QueueScheduleContext,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    PublishPlan,
    TrainingBudget,
    TrainingPlan,
    _as_float,
    _as_int,
)
from verl_speco.trainer.scheduler.execution_strategy import (
    RolloutIdleWorkerExecutionStrategy,
    SyncExecutionStrategy,
)
from verl_speco.trainer.scheduler.training_budget import SyncTrainingBudgetPolicy
from verl_speco.trainer.scheduler.training_trigger import IntervalAndBufferTrigger
from verl_speco.trainer.scheduler.worker_executor import DrafterWorkerExecutor
from verl_speco.trainer.scheduler.publish_executor import DrafterPublishExecutor
from verl_speco.trainer.scheduler.publish_strategy import PublishExecutionStrategy
from verl_speco.trainer.scheduler.data_status_policy import (
    ConservativeTrainingDataStatusPolicy,
)
from verl_speco.trainer.scheduler.lifecycle import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    SchedulerEventOutcome,
)
from verl_speco.trainer.scheduler.collection_executor import (
    DrafterCollectionExecutor,
)
from verl_speco.trainer.scheduler.collection_strategy import SyncCollectionStrategy
from verl_speco.trainer.scheduler.collection_adapter import (
    DrafterCollectionAdapter,
    OldLogProbCollectionAdapter,
    SGLangCollectionAdapter,
)
from verl_speco.trainer.scheduler.training_outcome import TrainingOutcome
from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.standalone_executor import (
    StandaloneCollectionExecutionStrategy,
    StandaloneCollectionExecutor,
    StandaloneCollectionOutcome,
    StandaloneTrainingExecutionStrategy,
    StandaloneTrainingExecutor,
    StandaloneTrainingOutcome,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_IDLE_BATCH_ESTIMATE_SEC = 0.25
_BOOTSTRAP_IDLE_DEADLINE_GUARD_SEC = 0.05
_BOOTSTRAP_IDLE_STARTUP_RESERVE_SEC = 2.0
_BOOTSTRAP_IDLE_TAIL_RESERVE_SEC = 2.0


def _conservative_percentile(values: Iterable[float], quantile: float = 0.9) -> float:
    samples = sorted(max(float(value), 0.0) for value in values)
    if not samples:
        return 0.0
    index = max(0, min(int(math.ceil(quantile * len(samples))) - 1, len(samples) - 1))
    return samples[index]


@dataclass
class _IdleWorkerState:
    worker_id: str
    replica_rank: int
    status: str = "ready"
    memory_released: bool = False
    must_be_ready_at: float | None = None
    event_ts: float = 0.0
    idle_confidence: IdleWindowConfidence = IdleWindowConfidence.SPECULATIVE
    confirmed_at_generation_boundary: bool = False
    generation_epoch: int = 0
    epoch_event_watermark: float = 0.0


def _rollout_worker_event_type(value: object) -> RolloutWorkerEventType:
    if isinstance(value, RolloutWorkerEventType):
        return value
    text = str(value)
    try:
        return RolloutWorkerEventType(text)
    except ValueError:
        return RolloutWorkerEventType[text.upper()]


def _idle_window_confidence(value: object) -> IdleWindowConfidence:
    if isinstance(value, IdleWindowConfidence):
        return value
    text = str(value or IdleWindowConfidence.SPECULATIVE.value).strip().lower()
    try:
        return IdleWindowConfidence(text)
    except ValueError:
        return IdleWindowConfidence.SPECULATIVE


def _natural_worker_sort_key(worker_id: object) -> tuple[str, int, str]:
    """Sort worker ids by trailing numeric rank when possible."""

    text = str(worker_id)
    prefix, separator, suffix = text.rpartition("-")
    if separator and suffix.isdigit():
        return (prefix, int(suffix), text)
    if text.isdigit():
        return ("", int(text), text)
    return (text, -1, text)


def _normalize_worker_id_group(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    try:
        worker_ids = [str(worker_id) for worker_id in cast(Iterable[object], value)]
    except TypeError:
        return (str(value),)
    return tuple(dict.fromkeys(sorted(worker_ids, key=_natural_worker_sort_key)))


def _flatten_metadata_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        records: list[dict[str, Any]] = []
        for item in value:
            records.extend(_flatten_metadata_records(item))
        return records
    return []


def _idle_state_summary(
    states: dict[str, _IdleWorkerState],
    *,
    now: float | None = None,
) -> list[dict[str, object]]:
    now = time.time() if now is None else now
    summary: list[dict[str, object]] = []
    for worker_id in sorted(states, key=_natural_worker_sort_key):
        state = states[worker_id]
        window = (
            None
            if state.must_be_ready_at is None
            else max(float(state.must_be_ready_at) - now, 0.0)
        )
        summary.append(
            {
                "worker_id": worker_id,
                "replica_rank": state.replica_rank,
                "status": state.status,
                "memory_released": state.memory_released,
                "idle_confidence": state.idle_confidence.value,
                "window_s": None if window is None else round(window, 3),
                "event_age_s": round(max(now - float(state.event_ts), 0.0), 3),
            }
        )
    return summary


def step_matches_interval(
    global_step: Any,
    interval_steps: Any,
    *,
    default_interval: int = 1,
) -> bool:
    """Match the released ``speco_step_matches_interval`` semantics exactly."""

    try:
        interval = int(default_interval if interval_steps is None else interval_steps)
    except (TypeError, ValueError):
        return False
    if interval <= 0 or global_step is None:
        return False
    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return False
    return step > 0 and step % interval == 0


class DrafterScheduler:
    """Make synchronous collect, train, and publish decisions.

    Every decision receives a
    fresh legacy-compatible config snapshot so runtime config mutation retains
    the same behavior as the released trainer implementation.
    """

    def __init__(
        self,
        worker_executor: DrafterWorkerExecutor | None = None,
        publish_executor: DrafterPublishExecutor | None = None,
        collection_executor: DrafterCollectionExecutor | None = None,
        standalone_collection_executor: StandaloneCollectionExecutor | None = None,
        standalone_training_executor: StandaloneTrainingExecutor | None = None,
    ) -> None:
        self.trigger_policy = IntervalAndBufferTrigger()
        self.sync_budget_policy = SyncTrainingBudgetPolicy()
        self.sync_execution_strategy = SyncExecutionStrategy()
        self._worker_executor = worker_executor
        self.data_status_policy = ConservativeTrainingDataStatusPolicy()
        self.publish_execution_strategy = PublishExecutionStrategy()
        self._publish_executor = publish_executor
        self.collection_strategy = SyncCollectionStrategy()
        self._collection_executor = collection_executor
        self._collection_sequence = 0
        self.standalone_collection_strategy = StandaloneCollectionExecutionStrategy()
        self.standalone_training_strategy = StandaloneTrainingExecutionStrategy()
        self._standalone_collection_executor = standalone_collection_executor
        self._standalone_training_executor = standalone_training_executor
        self._collection_adapters: dict[
            DrafterCollectionSource, DrafterCollectionAdapter
        ] = {
            DrafterCollectionSource.SGLANG: SGLangCollectionAdapter(),
            DrafterCollectionSource.OLD_LOGPROB: OldLogProbCollectionAdapter(),
        }
        self.rollout_idle_execution_strategy = RolloutIdleWorkerExecutionStrategy()
        self._idle_workers: dict[str, _IdleWorkerState] = {}
        self._metadata_idle_training_groups: tuple[tuple[str, ...], ...] = ()
        self._metadata_full_collective_idle_groups: tuple[tuple[str, ...], ...] = ()
        self._global_publish_leader_worker_id: str | None = None
        self._replica_idle_worker_groups: dict[int, tuple[str, ...]] = {}
        self._idle_worker_batch_estimate_sec: float | None = None
        self._idle_optimizer_steps_total = 0
        self._critical_path_topup_optimizer_steps_total = 0
        self._idle_worker_batch_samples_sec: deque[float] = deque(maxlen=32)
        # Serialized writer top-ups are a bootstrap throughput reference, not
        # observations of a hot rollout-idle restart or partial cleanup.
        self._quota_topup_batch_samples_sec: deque[float] = deque(maxlen=32)
        self._quota_topup_final_tail_samples_sec: deque[float] = deque(maxlen=32)
        self._quota_topup_park_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_worker_first_step_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_first_step_by_mode: dict[
            tuple[tuple[str, ...], str], deque[float]
        ] = {}
        self._idle_worker_head_prepare_samples: dict[tuple[str, ...], deque[float]] = {}
        self._idle_worker_hot_head_prepare_samples: dict[
            tuple[str, ...], deque[float]
        ] = {}
        self._idle_worker_last_target_version: dict[tuple[str, ...], int] = {}
        self._idle_worker_reclaim_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_worker_startup_samples_sec: deque[float] = deque(maxlen=32)
        # Intermediate plans only release their plan-local reservation and keep
        # the writer hot.  Final plans additionally materialize the publish
        # snapshot.  Mixing both paths makes one expensive publish permanently
        # shrink every later partial Bubble window.
        self._idle_worker_partial_tail_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_tail_by_mode: dict[
            tuple[tuple[str, ...], str, bool], deque[float]
        ] = {}
        self._idle_steady_by_group: dict[tuple[str, ...], deque[float]] = {}
        self._idle_worker_final_tail_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_worker_hot_prewarmed_groups: set[tuple[str, ...]] = set()
        self._idle_worker_bootstrapped = False
        # Loading/publishing a model does not initialize optimizer state.
        self._idle_optimizer_initialized_groups: set[tuple[str, ...]] = set()
        self._idle_worker_restore_samples: dict[tuple[str, ...], deque[float]] = {}
        self._idle_worker_park_samples_sec: deque[float] = deque(maxlen=32)
        self._training_quota_last_registered_data_version: int | None = None
        self._idle_worker_writer_group: tuple[str, ...] | None = None
        self._idle_worker_bootstrap_collection_group: tuple[str, ...] | None = None
        self._idle_worker_writer_state_version: int | None = None
        self._idle_worker_writer_migration_blocked: bool = False
        self._writer_boundary_migration_allowed = False
        self._writer_idle_history: deque[dict[tuple[str, ...], float]] = deque(
            maxlen=64
        )
        self._latest_published_drafter_version: int | None = None
        self._rollout_worker_drafter_versions: dict[str, int] = {}
        self._idle_worker_group_success_counts: dict[tuple[str, ...], int] = {}
        self._replica_idle_started_at: dict[int, float] = {}
        self._idle_readmission_attempts: dict[tuple[str, ...], dict[str, int]] = {}
        self._idle_readmission_last_check_ts: float = 0.0
        # Keep rollout-tail history replica-local.  Pooling samples globally
        # lets the replica that finishes last (and therefore has an almost
        # zero tail) suppress useful Bubble windows on every earlier replica.
        self._replica_idle_window_samples_sec: dict[int, deque[float]] = {}
        self._replica_idle_windows_by_source: dict[int, dict[str, deque[float]]] = {}
        self._replica_idle_timeline: dict[int, dict[str, float | None]] = {}
        self._replica_idle_event_delivery_lag_sec: dict[int, float] = {}
        self._replica_last_observed_idle_window_sec: dict[int, float] = {}
        self._replica_idle_window_close_source: dict[int, str] = {}
        self._idle_worker_prebatch_reclaim_streak: int = 0
        self._idle_worker_reclaim_penalty_sec: float = 0.0
        self._idle_worker_reclaim_penalty_last_step: int | None = None
        self._idle_worker_dynamic_batch_cap: int | None = None
        self._idle_group_batch_caps: dict[tuple[str, ...], int] = {}
        self._idle_worker_last_feedback_step: int | None = None
        self._runtime_replica_local_topology = False
        self._idle_worker_gen_per_token_baseline_ms: float | None = None
        self._idle_worker_gen_per_token_samples: deque[float] = deque(maxlen=16)
        self._idle_worker_gen_slowdown_streak: int = 0
        self._idle_worker_suspended_until_step: int | None = None
        self._last_observed_global_step: int | None = None
        self._disabled_replica_local_idle_groups: set[tuple[str, ...]] = set()
        self._replica_local_idle_unavailable_reason: str = ""
        self._training_quota_last_cycle_step: int | None = None
        self._training_quota_debt_steps: int = 0
        self._training_quota_oldest_cycle_step: int | None = None
        self._training_quota_data_version: int | None = None
        # A successful collection owns the next quality quota even before its
        # target LM-head snapshot becomes trainable. Keep that collection
        # single-flight so later intervals do not produce unusable data.
        self._training_quota_collection_step: int | None = None
        # Production quality gating never shrinks a quota. It only decides
        # when the next full, single-version quota may be collected.
        self._quality_last_publish_step: int | None = None
        self._quality_acceptance_baseline: float | None = None
        self._quality_pending_acceptance_baseline: bool = False
        self._quality_latest_acceptance: float | None = None
        self._quality_loss_baseline: float | None = None
        self._quality_latest_loss: float | None = None
        self._quality_loss_refresh_requested: bool = False
        self._quality_loss_increase_ratio: float = 0.10

    def _effective_idle_batch_estimate_sec(
        self,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...] | None = None,
    ) -> float:
        if config.idle_worker_initial_batch_estimate_sec is not None:
            return max(float(config.idle_worker_initial_batch_estimate_sec), 1.0e-9)
        group = _normalize_worker_id_group(
            worker_ids
        ) or self._current_idle_writer_group(assign_default=False)
        recent = self._idle_steady_by_group.get(group or ())
        if (
            config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and config.idle_worker_calibrated_budgets
            and recent is not None
            and len(recent) >= 3
        ):
            # These are non-first optimizer steps from this writer, normalized
            # by GAS. Do not let an old cross-plan EMA floor pin them forever.
            return max(_conservative_percentile(recent), 1.0e-9)
        if self._idle_worker_batch_estimate_sec is not None:
            conservative_batch_sec = _conservative_percentile(
                self._idle_worker_batch_samples_sec
            )
            return max(
                float(self._idle_worker_batch_estimate_sec),
                conservative_batch_sec,
                1.0e-9,
            )
        if self._quota_topup_batch_samples_sec:
            return max(
                _conservative_percentile(self._quota_topup_batch_samples_sec), 1.0e-9
            )
        return _BOOTSTRAP_IDLE_BATCH_ESTIMATE_SEC

    def _idle_batch_estimate_is_bootstrap(
        self,
        config: DrafterScheduleConfig,
    ) -> bool:
        return (
            config.idle_worker_initial_batch_estimate_sec is None
            and self._idle_worker_batch_estimate_sec is None
            and not self._quota_topup_batch_samples_sec
        )

    def _effective_idle_deadline_guard_sec(
        self,
        config: DrafterScheduleConfig,
    ) -> float:
        if config.idle_worker_deadline_guard_sec is not None:
            configured_guard = max(float(config.idle_worker_deadline_guard_sec), 0.0)
        else:
            estimate = self._effective_idle_batch_estimate_sec(config)
            configured_guard = max(_BOOTSTRAP_IDLE_DEADLINE_GUARD_SEC, estimate * 0.1)
        # Reclaim drain includes in-flight training and sometimes publication.
        # Offload belongs to the tail budget, not an additional deadline guard.
        return configured_guard

    def _effective_idle_first_step_extra_sec(
        self, config, worker_ids=(), *, continuation=False
    ) -> float:
        group = _normalize_worker_id_group(worker_ids)
        mode = (
            "continuation"
            if continuation and group in self._idle_worker_hot_prewarmed_groups
            else ("hot" if group in self._idle_worker_hot_prewarmed_groups else "cold")
        )
        history = self._idle_first_step_by_mode.get((group, mode))
        if history is None:
            # A new restart mode cannot inherit another group's cold outlier.
            # Continuation uses the hot restart estimate until measured itself.
            history = (
                self._idle_first_step_by_mode.get((group, "hot"))
                if mode == "continuation"
                else None
            )
        budget_history: Iterable[float] = (
            history
            if history is not None
            else (
                ()
                if self._idle_first_step_by_mode
                else self._idle_worker_first_step_samples_sec
            )
        )
        return max(
            _conservative_percentile(budget_history)
            - self._effective_idle_batch_estimate_sec(config, group),
            0.0,
        ) * self._idle_gradient_accumulation_steps(
            config, _normalize_worker_id_group(worker_ids)
        )

    def _effective_idle_startup_reserve_sec(
        self,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...] | None = None,
        *,
        continuation: bool = False,
    ) -> float:
        """Reserve activation and preflight work before the first batch."""

        group = _normalize_worker_id_group(worker_ids)
        first_extra = self._effective_idle_first_step_extra_sec(
            config, group, continuation=continuation
        )
        if group and group in self._idle_worker_hot_prewarmed_groups:
            head_prepare = 0.0
            target = self._training_quota_data_version
            if (
                target is not None
                and self._idle_worker_last_target_version.get(group) != target
            ):
                head_history: Iterable[float] = (
                    self._idle_worker_hot_head_prepare_samples.get(group)
                    or self._idle_worker_head_prepare_samples.get(group)
                    or ()
                )
                head_prepare = _conservative_percentile(head_history)
            return head_prepare + first_extra
        restore = _conservative_percentile(
            self._idle_worker_restore_samples.get(group, ())
        )
        if restore > 0.0:
            return restore + first_extra
        historical = _conservative_percentile(self._idle_worker_startup_samples_sec)
        if historical > 0.0:
            return historical + first_extra
        if self._idle_batch_estimate_is_bootstrap(config):
            return _BOOTSTRAP_IDLE_STARTUP_RESERVE_SEC + first_extra
        return first_extra

    def prewarm_idle_training_workers(self) -> list[Any]:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        candidate_groups = tuple(
            _normalize_worker_id_group(group)
            for group in self._metadata_idle_training_groups
            if _normalize_worker_id_group(group)
            not in self._disabled_replica_local_idle_groups
        )
        if not candidate_groups:
            logger.warning(
                "[BubbleTime] idle_prewarm_skipped: reason=no_training_group_metadata"
            )
            return []
        all_results: list[Any] = []
        for target_group in candidate_groups:
            results = self._worker_executor.prewarm_training_workers(target_group)
            all_results.extend(results)
            active = [
                result
                for result in results
                if isinstance(result, dict)
                and result.get("reason") not in {"disabled", "not_in_training_group"}
            ]
            successful = [
                result for result in active if bool(result.get("activated", False))
            ]
            failed = [
                result for result in active if not bool(result.get("activated", False))
            ]
            for result in failed:
                if bool(result.get("replica_local_unavailable", False)):
                    self._disable_replica_local_idle_group(
                        tuple(
                            str(worker_id)
                            for worker_id in result.get("training_group_ranks", ())
                        ),
                        reason=(
                            "replica_local_oom"
                            if bool(result.get("replica_local_oom", False))
                            else "replica_local_prewarm_failed"
                        ),
                    )
            if active and successful and not failed:
                self._idle_worker_hot_prewarmed_groups.add(target_group)
                # Prewarming proves only that this group can host the trainer.
                # Metadata order must not elect the mutable-state writer: it
                # is unrelated to which rollout replica exposes a useful idle
                # window once training data becomes available.
            logger.warning(
                "[BubbleTime] idle_prewarm_completed: group=%s active=%s "
                "successful=%s failed=%s writer_group=%s hot_groups=%s",
                target_group,
                len(active),
                len(successful),
                len(failed),
                self._idle_worker_writer_group,
                tuple(sorted(self._idle_worker_hot_prewarmed_groups)),
            )
            # Prewarming every replica-local trainer keeps one complete model
            # and optimizer resident per rollout group before an owner exists.
            # That consumes generation memory and can turn a Bubble
            # optimization into an OOM or a throughput regression. Warm one
            # candidate without granting it the writer lease; the first real
            # trainable idle plan is still free to elect any other group.
            if target_group in self._idle_worker_hot_prewarmed_groups:
                break
        return all_results

    def _current_idle_writer_group(
        self,
        *,
        assign_default: bool = True,
    ) -> tuple[str, ...] | None:
        """Return the only replica-local group allowed to mutate drafter state."""

        if self._idle_worker_writer_migration_blocked:
            return None
        writer_group = _normalize_worker_id_group(self._idle_worker_writer_group)
        if (
            writer_group
            and writer_group not in self._disabled_replica_local_idle_groups
            and (
                not self._metadata_idle_training_groups
                or writer_group in self._metadata_idle_training_groups
            )
        ):
            return writer_group
        if not assign_default:
            return None
        for group in sorted(self._idle_worker_hot_prewarmed_groups):
            normalized = _normalize_worker_id_group(group)
            if (
                normalized
                and normalized not in self._disabled_replica_local_idle_groups
            ):
                self._idle_worker_writer_group = normalized
                return normalized
        for group in self._metadata_idle_training_groups:
            normalized = _normalize_worker_id_group(group)
            if (
                normalized
                and normalized not in self._disabled_replica_local_idle_groups
            ):
                self._idle_worker_writer_group = normalized
                return normalized
        return None

    def idle_writer_group(self) -> tuple[str, ...] | None:
        """Return the current Bubble writer group without exposing mutable state."""

        return self._current_idle_writer_group(assign_default=False)

    def prepare_collection_writer_group(self) -> tuple[str, ...] | None:
        """Pin the initial data destination, not initialized optimizer state.

        Call only after a nonempty old-logprob sample set has been selected.
        Routing the complete pool once avoids discarding other replicas' data
        when the first quota is bootstrapped on a single writer group.
        """

        writer_group = self._current_idle_writer_group(assign_default=False)
        if writer_group is not None:
            return writer_group
        if (
            self._idle_worker_writer_migration_blocked
            or self._idle_worker_writer_state_version is not None
        ):
            return None
        writer_group = self._best_idle_bootstrap_group()
        if writer_group:
            self._idle_worker_writer_group = writer_group
            self._idle_worker_bootstrap_collection_group = writer_group
            logger.warning(
                "[BubbleTime] collection_writer_selected: group=%s "
                "reason=first_nonempty_sync_aligned_collection",
                writer_group,
            )
        return writer_group

    def idle_checkpoint_group(self) -> tuple[str, ...] | None:
        """Return one complete group that owns the checkpoint operation."""

        return self._current_idle_writer_group(assign_default=True)

    def idle_writer_migration_blocked(self) -> bool:
        """Whether the current writer owns state that cannot be failed over."""

        return self._idle_worker_writer_migration_blocked

    def _known_rollout_training_worker_ids(self) -> tuple[str, ...]:
        worker_ids = {
            worker_id
            for group in (
                self._metadata_idle_training_groups
                or self._metadata_full_collective_idle_groups
            )
            for worker_id in _normalize_worker_id_group(group)
        }
        if not worker_ids:
            worker_ids.update(self._idle_workers)
        return tuple(sorted(worker_ids, key=_natural_worker_sort_key))

    @staticmethod
    def _flatten_publish_acknowledgements(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple, set)):
            flattened: list[dict[str, Any]] = []
            for item in value:
                flattened.extend(
                    DrafterScheduler._flatten_publish_acknowledgements(item)
                )
            return flattened
        return []

    def record_rollout_drafter_publish_completed(
        self,
        source_version: object,
        *,
        acknowledgements: Any = None,
        worker_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Record rollout workers that acknowledged the live drafter version.

        Bubble writer switching is safe only after the candidate rollout group
        has already applied the latest published drafter weights.  This method
        intentionally records only completed live publishes, not staged-only
        acknowledgements.
        """

        try:
            published_version = _as_int(source_version)
        except (TypeError, ValueError):
            return
        self._latest_published_drafter_version = published_version
        ack_workers: set[str] = set()
        stale_or_failed_acks = 0
        for ack in self._flatten_publish_acknowledgements(acknowledgements):
            if not bool(ack.get("published", False)):
                stale_or_failed_acks += 1
                continue
            ack_version = ack.get("published_version", source_version)
            try:
                if _as_int(ack_version) != published_version:
                    stale_or_failed_acks += 1
                    continue
            except (TypeError, ValueError):
                stale_or_failed_acks += 1
                continue
            worker_id = ack.get("worker_id", ack.get("worker_rank", ack.get("rank")))
            if worker_id is None:
                stale_or_failed_acks += 1
                continue
            ack_workers.add(str(worker_id))
        if worker_ids:
            ack_workers.update(str(worker_id) for worker_id in worker_ids)
        elif acknowledgements is None:
            # Synchronous publish already waited for the rollout update RPC but
            # older executors do not expose per-worker ack payloads. Mark the
            # known training workers so writer selection remains compatible.
            ack_workers.update(self._known_rollout_training_worker_ids())
        for worker_id in ack_workers:
            self._rollout_worker_drafter_versions[worker_id] = published_version
        logger.info(
            "[BubbleTime] rollout_drafter_versions_updated: "
            "published_version=%s workers=%s stale_or_failed_acks=%s versions=%s",
            published_version,
            tuple(sorted(ack_workers, key=_natural_worker_sort_key)),
            stale_or_failed_acks,
            dict(sorted(self._rollout_worker_drafter_versions.items())),
        )

    def _drafter_version_ready_for_writer_group(
        self, group: tuple[str, ...]
    ) -> tuple[bool, dict[str, int | None]]:
        latest = self._latest_published_drafter_version
        versions = {
            worker_id: self._rollout_worker_drafter_versions.get(worker_id)
            for worker_id in _normalize_worker_id_group(group)
        }
        if latest is None:
            return True, versions
        return all(
            version is not None and int(version) >= latest
            for version in versions.values()
        ), versions

    def target_lm_head_sync_worker_ids(
        self,
        *,
        full_collective_fallback: bool = False,
    ) -> tuple[str, ...] | None:
        """Workers that should cache the next target LM-head snapshot.

        Returning ``None`` preserves the legacy/sync broadcast behavior.  In
        Bubble replica-local mode, keep the heavy target head cache limited to
        the single writer group once a writer lease exists.  Before the first
        lease, cache all legal candidates so the earliest genuinely idle group
        can become the writer without fetching the live actor head from inside
        the rollout bubble.
        """

        groups: list[tuple[str, ...]] = []
        writer_group = self._current_idle_writer_group(assign_default=False)
        if writer_group is not None:
            groups.append(writer_group)
        elif self._metadata_idle_training_groups:
            groups.extend(
                normalized
                for group in self._metadata_idle_training_groups
                if (normalized := _normalize_worker_id_group(group))
                and normalized not in self._disabled_replica_local_idle_groups
            )
        if not groups:
            for group in self._metadata_idle_training_groups:
                normalized = _normalize_worker_id_group(group)
                if (
                    normalized
                    and normalized not in self._disabled_replica_local_idle_groups
                ):
                    groups.append(normalized)
                    break
        if (
            not groups
            and full_collective_fallback
            and self._disabled_replica_local_idle_groups
        ):
            for group in self._metadata_full_collective_idle_groups:
                normalized = _normalize_worker_id_group(group)
                if normalized:
                    groups.append(normalized)
                    break
        worker_ids = tuple(
            dict.fromkeys(
                worker_id
                for group in groups
                for worker_id in _normalize_worker_id_group(group)
            )
        )
        return worker_ids or None

    def _effective_idle_tail_reserve_sec(
        self,
        config: DrafterScheduleConfig,
        *,
        final: bool = False,
        worker_ids: tuple[str, ...] | None = None,
    ) -> float:
        """Reserve the plan tail without mixing partial and final publish cost."""

        samples = (
            self._idle_worker_final_tail_samples_sec
            if final
            else self._idle_worker_partial_tail_samples_sec
        )
        historical = _conservative_percentile(samples)
        group = _normalize_worker_id_group(
            worker_ids
        ) or self._current_idle_writer_group(assign_default=False)
        mode = "hot" if config.idle_worker_keep_hot_between_plans else "cold"
        recent = self._idle_tail_by_mode.get((group or (), mode, final))
        if (
            config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and config.idle_worker_calibrated_budgets
            and recent is not None
            and len(recent) >= 3
        ):
            historical = _conservative_percentile(recent)
        park = (
            0.0
            if config.idle_worker_keep_hot_between_plans
            else _conservative_percentile(
                self._idle_worker_park_samples_sec or self._quota_topup_park_samples_sec
            )
        )
        if historical > 0.0:
            return historical + park
        if final and self._quota_topup_final_tail_samples_sec:
            # Until a true idle finalization is measured, keep its synchronous
            # counterpart as a conservative final-only safety reference.
            return (
                _conservative_percentile(self._quota_topup_final_tail_samples_sec)
                + park
            )
        if final and self._idle_batch_estimate_is_bootstrap(config):
            return _BOOTSTRAP_IDLE_TAIL_RESERVE_SEC + park
        return park

    def record_memory_park_elapsed(self, elapsed_sec: float) -> None:
        if elapsed_sec > 0.0:
            self._idle_worker_park_samples_sec.append(float(elapsed_sec))

    def record_reclaim_elapsed(self, elapsed_sec: float) -> None:
        elapsed_sec = max(float(elapsed_sec), 0.0)
        if elapsed_sec <= 0.0:
            return
        self._idle_worker_reclaim_samples_sec.append(elapsed_sec)
        logger.warning(
            "[BubbleTime] updated reclaim guard: observed_s=%.3f guard_s=%.3f samples=%s",
            elapsed_sec,
            _conservative_percentile(self._idle_worker_reclaim_samples_sec),
            len(self._idle_worker_reclaim_samples_sec),
        )

    def record_idle_training_outcome(
        self,
        outcome: TrainingOutcome,
        *,
        final: bool = False,
        worker_ids: tuple[str, ...] = (),
    ) -> None:
        """Learn conservative admission overhead from useful training plans.

        A pre-batch reclaim is a scheduling miss, not a representative Bubble
        training cost.  Recording its cold activation or cleanup here makes
        later real idle windows look too small and can permanently prevent
        Bubble training from launching.
        """

        first_batch_started = bool(
            outcome.metrics.get("bubble/train_first_batch_started", 0)
        )
        if not first_batch_started and int(outcome.successful_steps) <= 0:
            logger.debug(
                "[BubbleTime] idle_overhead_update_skipped: reason=%s "
                "successful_steps=%s first_batch_started=%s",
                outcome.reason,
                outcome.successful_steps,
                first_batch_started,
            )
            return

        setup_sec = max(
            float(outcome.metrics.get("timing_s/drafter_worker_preflight", 0.0) or 0.0),
            0.0,
        )
        park_sec = max(
            float(
                outcome.metrics.get("timing_s/drafter_worker_memory_park", 0.0) or 0.0
            ),
            0.0,
        )
        self.record_memory_park_elapsed(park_sec)
        group = _normalize_worker_id_group(worker_ids)
        head_prepare = float(
            outcome.metrics.get("timing_s/drafter_worker_target_head_prepare", 0.0)
            or 0.0
        )
        if group and head_prepare > 0.0:
            self._idle_worker_head_prepare_samples.setdefault(
                group, deque(maxlen=32)
            ).append(head_prepare)
            if outcome.metrics.get("bubble/activation_started_resident", 0):
                self._idle_worker_hot_head_prepare_samples.setdefault(
                    group, deque(maxlen=32)
                ).append(head_prepare)
        # Aggregate each rank's non-park cleanup directly. Subtracting the
        # maximum park from the maximum cleanup can mix different ranks.
        cleanup_sec = max(
            float(
                outcome.metrics.get(
                    "timing_s/drafter_worker_cleanup_nonpark",
                    max(
                        float(
                            outcome.metrics.get("timing_s/drafter_worker_cleanup", 0.0)
                            or 0.0
                        )
                        - park_sec,
                        0.0,
                    ),
                )
                or 0.0
            ),
            0.0,
        )
        snapshot_sec = max(
            float(outcome.metrics.get("timing_s/drafter_publish_snapshot", 0.0) or 0.0),
            0.0,
        )
        tail_sec = cleanup_sec + (snapshot_sec if final else 0.0)
        if setup_sec > 0.0:
            self._idle_worker_startup_samples_sec.append(setup_sec)
            group = _normalize_worker_id_group(worker_ids)
            if group:
                self._idle_worker_restore_samples.setdefault(
                    group, deque(maxlen=32)
                ).append(setup_sec)
        tail_samples = (
            self._idle_worker_final_tail_samples_sec
            if final
            else self._idle_worker_partial_tail_samples_sec
        )
        if tail_sec > 0.0:
            tail_samples.append(tail_sec)
            if group and "bubble/training_residency_retained" in outcome.metrics:
                mode = (
                    "hot"
                    if outcome.metrics["bubble/training_residency_retained"]
                    else "cold"
                )
                self._idle_tail_by_mode.setdefault(
                    (group, mode, final), deque(maxlen=4)
                ).append(tail_sec)
        if setup_sec <= 0.0 and tail_sec <= 0.0:
            return
        logger.warning(
            "[BubbleTime] idle_overhead_updated: reason=%s setup_s=%.3f "
            "tail_s=%.3f tail_kind=%s activation_s=%.3f park_s=%.3f "
            "snapshot_s=%.3f nonpark_cleanup_s=%.3f setup_reserve_s=%.3f "
            "partial_tail_reserve_s=%.3f final_tail_reserve_s=%.3f",
            outcome.reason,
            setup_sec,
            tail_sec,
            "final" if final else "partial",
            float(
                outcome.metrics.get("timing_s/drafter_worker_activation", 0.0) or 0.0
            ),
            park_sec,
            snapshot_sec,
            cleanup_sec,
            _conservative_percentile(self._idle_worker_startup_samples_sec),
            _conservative_percentile(self._idle_worker_partial_tail_samples_sec),
            _conservative_percentile(self._idle_worker_final_tail_samples_sec),
        )

    def _disable_replica_local_idle_group(
        self,
        worker_ids: tuple[str, ...],
        *,
        reason: str,
    ) -> None:
        group = _normalize_worker_id_group(worker_ids)
        if not group:
            return
        first_seen = group not in self._disabled_replica_local_idle_groups
        self._disabled_replica_local_idle_groups.add(group)
        self._idle_worker_hot_prewarmed_groups.discard(group)
        if self._idle_worker_writer_group == group:
            if self._idle_worker_writer_state_version is None:
                # No optimizer update has happened yet, so another initialized
                # group still has an equivalent training state.
                self._idle_worker_writer_group = None
            else:
                # The group owns model, optimizer, scheduler and RNG state.
                # Never fail over to a stale group without state transfer.
                self._idle_worker_writer_migration_blocked = True
                reason = "writer_state_migration_required"
        self._replica_local_idle_unavailable_reason = reason
        self._metadata_idle_training_groups = tuple(
            existing
            for existing in self._metadata_idle_training_groups
            if _normalize_worker_id_group(existing) != group
        )
        logger.error(
            "[BubbleTime] replica_local_group_disabled: group=%s reason=%s "
            "disabled_groups=%s remaining_groups=%s first_seen=%s "
            "writer_group=%s writer_state_version=%s migration_blocked=%s",
            group,
            reason,
            tuple(sorted(self._disabled_replica_local_idle_groups)),
            self._metadata_idle_training_groups,
            first_seen,
            self._idle_worker_writer_group,
            self._idle_worker_writer_state_version,
            self._idle_worker_writer_migration_blocked,
        )

    def _record_replica_local_unavailable(
        self,
        plan: TrainingPlan,
        outcome: TrainingOutcome,
    ) -> None:
        if not bool(outcome.metrics.get("bubble/replica_local_unavailable", 0)):
            return
        reason = (
            "replica_local_oom"
            if bool(outcome.metrics.get("bubble/replica_local_oom", 0))
            else "replica_local_activation_failed"
        )
        self._disable_replica_local_idle_group(
            plan.target_worker_ids,
            reason=reason,
        )

    def _replica_ranks_for_worker_group(
        self,
        worker_ids: tuple[str, ...] | None,
    ) -> tuple[int, ...]:
        """Resolve the rollout replicas that own one training group."""

        group = _normalize_worker_id_group(worker_ids)
        if not group:
            return ()
        ranks = {
            int(state.replica_rank)
            for worker_id in group
            if (state := self._idle_workers.get(worker_id)) is not None
        }
        if ranks:
            return tuple(sorted(ranks))
        group_members = set(group)
        return tuple(
            sorted(
                int(replica_rank)
                for replica_rank, replica_workers in self._replica_idle_worker_groups.items()
                if group_members.intersection(replica_workers)
            )
        )

    def _effective_historical_idle_window_sec(
        self,
        *,
        replica_rank: int | None = None,
        worker_ids: tuple[str, ...] | None = None,
    ) -> float | None:
        """Return a conservative idle tail for only the requested replica/group."""

        if replica_rank is not None:
            replica_ranks: tuple[int, ...] = (int(replica_rank),)
        elif worker_ids:
            replica_ranks = self._replica_ranks_for_worker_group(worker_ids)
        else:
            replica_ranks = tuple(sorted(self._replica_idle_window_samples_sec))
        conservative_windows = []
        for rank in replica_ranks:
            history = self._replica_idle_window_samples_sec.get(rank)
            if not history:
                continue
            window = _conservative_percentile(history, quantile=0.75)
            tails = self._replica_idle_windows_by_source.get(rank, {}).get(
                "generation_completed", ()
            )
            if len(tails) >= 3:
                # A formerly early replica may now finish last repeatedly.
                # Bound its old p75 with its recent real generation tails;
                # never borrow another replica's longer idle or reclaim wait.
                window = min(
                    window,
                    _conservative_percentile(list(tails)[-3:], quantile=0.5),
                )
            conservative_windows.append(window)
        if not conservative_windows:
            return None
        # A multi-replica collective is bounded by its shortest member, while
        # unrelated replica-local groups never contaminate one another.
        return min(conservative_windows)

    def _historical_idle_readmission_window(
        self,
        config: DrafterScheduleConfig,
        group: tuple[str, ...],
        *,
        now: float,
    ) -> float | None:
        """Conservative conditional tail, not a new resource/deadline guarantee.

        A short recent median must not permanently suppress a live long tail.
        Wait until that forecast has actually elapsed, then require at least
        three completed tails of this replica that survived the same age. Use
        their shortest residual, not a maximum or a borrowed replica's tail.
        The caller consumes at most one admission attempt per idle episode.
        """
        if (
            not config.idle_worker_historical_window_readmission
            or config.execution_strategy
            is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or not config.training_quota_enable
            or not config.idle_worker_keep_hot_between_plans
            or self._training_quota_debt_steps <= 1
            or group != self._current_idle_writer_group(assign_default=False)
            or self._idle_worker_writer_state_version is None
            or group not in self._idle_worker_hot_prewarmed_groups
            or group not in self._idle_optimizer_initialized_groups
            or self._idle_worker_prebatch_reclaim_streak > 0
        ):
            return None
        ranks = self._replica_ranks_for_worker_group(group)
        if len(ranks) != 1:
            return None
        started = self._replica_idle_started_at.get(ranks[0])
        if started is None:
            return None
        states = [self._idle_workers.get(worker) for worker in group]
        if not states or any(
            state is None
            or state.status != "idle"
            or not state.memory_released
            or state.idle_confidence is not IdleWindowConfidence.CONFIRMED
            or state.must_be_ready_at is not None
            or state.confirmed_at_generation_boundary
            for state in states
        ):
            return None
        previous_epochs = self._idle_readmission_attempts.get(group)
        if previous_epochs is not None and any(
            self._idle_workers[worker].generation_epoch
            <= previous_epochs.get(worker, self._idle_workers[worker].generation_epoch)
            for worker in group
        ):
            # A fresh idle notification is not proof of a new rollout episode.
            # Every writer rank must have passed through a newer generation
            # before this group may consume another conditional attempt.
            return None
        forecast = self._effective_historical_idle_window_sec(worker_ids=group)
        age = max(now - started, 0.0)
        if (
            forecast is None
            or not math.isfinite(forecast)
            or forecast <= 0
            or age <= 0
            or age < forecast
        ):
            return None
        history = self._replica_idle_windows_by_source.get(ranks[0], {}).get(
            "generation_completed", ()
        )
        survivors = [
            float(tail) - age
            for tail in list(history)[-16:]
            if math.isfinite(float(tail)) and float(tail) > age
        ]
        if len(survivors) < 3:
            return None
        return min(survivors)

    def should_retry_idle_admission(
        self,
        config: DrafterScheduleConfig | Callable[[], DrafterScheduleConfig],
        *,
        now: float | None = None,
    ) -> bool:
        """Reject impossible/throttled quiet retries before resolving config.

        A factory avoids reconstructing an OmegaConf-backed schedule on every
        50ms empty poll. These cached prerequisites already exist in the full
        readmission gate; live-window, deadline and epoch checks remain there.
        """
        checked_at = time.time() if now is None else now
        if (
            self._training_quota_debt_steps <= 1
            or self._idle_worker_writer_state_version is None
            or self._idle_worker_prebatch_reclaim_streak > 0
        ):
            return False
        group = self._current_idle_writer_group(assign_default=False)
        if (
            group is None
            or group not in self._idle_worker_hot_prewarmed_groups
            or group not in self._idle_optimizer_initialized_groups
        ):
            return False
        # An empty poll during generation is not an admission attempt. In
        # particular it must not throttle the first usable live idle poll.
        states = [self._idle_workers.get(worker) for worker in group]
        if any(
            state is None
            or state.status != "idle"
            or not state.memory_released
            or state.idle_confidence is not IdleWindowConfidence.CONFIRMED
            or state.must_be_ready_at is not None
            or state.confirmed_at_generation_boundary
            for state in states
        ):
            return False
        ranks = self._replica_ranks_for_worker_group(group)
        if len(ranks) != 1:
            return False
        started = self._replica_idle_started_at.get(ranks[0])
        forecast = self._effective_historical_idle_window_sec(worker_ids=group)
        if (
            started is None
            or forecast is None
            or not math.isfinite(forecast)
            or forecast <= 0
        ):
            return False
        age = max(checked_at - started, 0.0)
        if age < forecast:
            return False
        # A short forecast can leave less than 250ms in which the unchanged
        # survival gate proves one optimizer step fits (log8, step11). Check
        # at the event-poll cadence only near that boundary, then back off.
        retry_interval = 0.05 if age - forecast < 0.25 else 0.25
        if checked_at - self._idle_readmission_last_check_ts < retry_interval:
            return False
        config = config() if callable(config) else config
        if not config.idle_worker_historical_window_readmission:
            return False
        # Charge factory time before evaluating the real residual window.
        now = time.time() if now is None else now
        self._idle_readmission_last_check_ts = now
        window = self._historical_idle_readmission_window(config, group, now=now)
        return window is not None and window >= self._minimum_idle_training_window_sec(
            config, worker_ids=group
        )

    def _best_idle_bootstrap_group(self) -> tuple[str, ...] | None:
        """Choose the replica most likely to consume the next idle window."""

        candidates = []
        for raw_group in self._metadata_idle_training_groups:
            group = _normalize_worker_id_group(raw_group)
            if group and group not in self._disabled_replica_local_idle_groups:
                candidates.append(group)
        if not candidates:
            return self._current_idle_writer_group(assign_default=True)
        writer_group = self._current_idle_writer_group(assign_default=False)

        def candidate_key(group: tuple[str, ...]) -> tuple[float, int, int]:
            # Election is not admission: a high historical percentile is too
            # optimistic for choosing a long-lived optimizer owner. Discount
            # sparse observations and give the latest real tails more weight.
            scores = []
            counts = []
            for rank in self._replica_ranks_for_worker_group(group):
                tails = self._replica_idle_windows_by_source.get(rank, {}).get(
                    "generation_completed", ()
                )
                history = tails or self._replica_idle_window_samples_sec.get(rank, ())
                recent = list(history)[-4:]
                if not recent:
                    return (-1.0, 0, int(group == writer_group))
                score = recent[0]
                for value in recent[1:]:
                    score = 0.75 * value + 0.25 * score
                confidence = min(len(recent) / 3.0, 1.0)
                scores.append(score * confidence)
                counts.append(min(len(recent), 3))
            return (
                min(scores, default=-1.0),
                min(counts, default=0),
                int(group == writer_group),
            )

        # Prefer measured rollout capacity.  A prior writer wins exact ties;
        # metadata order remains the deterministic fallback without history.
        _, best_group = max(
            enumerate(candidates),
            key=lambda item: (candidate_key(item[1]), -item[0]),
        )
        return best_group

    def _all_idle_training_worker_ids(self) -> tuple[str, ...]:
        groups = (
            self._metadata_idle_training_groups
            or self._metadata_full_collective_idle_groups
        )
        collected: list[str] = []
        for raw_group in groups:
            group = _normalize_worker_id_group(raw_group)
            if not group or group in self._disabled_replica_local_idle_groups:
                continue
            collected.extend(group)
        worker_ids = tuple(dict.fromkeys(collected))
        return tuple(sorted(worker_ids, key=_natural_worker_sort_key))

    def _hot_groups_for_worker_ids(
        self,
        worker_ids: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        worker_set = set(_normalize_worker_id_group(worker_ids))
        if not worker_set:
            return ()
        hot_groups: list[tuple[str, ...]] = []
        for raw_group in self._metadata_idle_training_groups:
            group = _normalize_worker_id_group(raw_group)
            if (
                group
                and set(group).issubset(worker_set)
                and group not in self._disabled_replica_local_idle_groups
            ):
                hot_groups.append(group)
        return tuple(hot_groups)

    def _effective_idle_min_window_sec(
        self,
        config: DrafterScheduleConfig,
    ) -> float:
        if config.idle_worker_min_idle_window_sec is not None:
            return max(float(config.idle_worker_min_idle_window_sec), 0.0)
        return self._effective_idle_deadline_guard_sec(config)

    def _minimum_idle_training_window_sec(
        self,
        config: DrafterScheduleConfig,
        *,
        min_batches: int = 1,
        worker_ids: tuple[str, ...] | None = None,
        continuation: bool = False,
    ) -> float:
        """Minimum window that can start and finish a useful idle batch."""

        accumulation_steps = self._idle_gradient_accumulation_steps(
            config, _normalize_worker_id_group(worker_ids)
        )
        optimizer_step_estimate_sec = (
            self._effective_idle_batch_estimate_sec(config, worker_ids)
            * accumulation_steps
        )
        return max(
            self._effective_idle_min_window_sec(config),
            self._effective_idle_deadline_guard_sec(config)
            + self._effective_idle_startup_reserve_sec(
                config, worker_ids, continuation=continuation
            )
            + self._effective_idle_tail_reserve_sec(config, worker_ids=worker_ids)
            + optimizer_step_estimate_sec * max(int(min_batches), 1),
        )

    def _effective_idle_dynamic_batch_cap(
        self,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...] | None = None,
    ) -> int:
        hard_cap = max(int(config.train_batches_per_trigger), 1)
        if not config.idle_worker_dynamic_batch_cap:
            return hard_cap
        if self._idle_worker_dynamic_batch_cap is None:
            initial_cap = config.idle_worker_initial_dynamic_batches
            self._idle_worker_dynamic_batch_cap = (
                hard_cap if initial_cap is None else max(int(initial_cap), 1)
            )
        group = (
            _normalize_worker_id_group(worker_ids)
            if worker_ids is not None
            else self._current_idle_writer_group(assign_default=False)
        )
        cap = self._idle_group_batch_caps.get(group) if group else None
        if cap is None:
            cap = (
                self._idle_worker_dynamic_batch_cap
                if not self._idle_group_batch_caps or not group
                else (config.idle_worker_initial_dynamic_batches or hard_cap)
            )
        return max(min(int(cap), hard_cap), 1)

    @staticmethod
    def _idle_stop_reason(
        outcome: TrainingOutcome,
        plan: TrainingPlan | None = None,
    ) -> str:
        target_worker_ids = {
            str(worker_id) for worker_id in (plan.target_worker_ids if plan else ())
        }
        for result in outcome.raw_results:
            if isinstance(result, dict):
                worker_id = str(result.get("worker_id", result.get("rank", "")))
                if target_worker_ids and worker_id not in target_worker_ids:
                    continue
                reason = str(result.get("stop_reason") or result.get("reason") or "")
                if reason and reason not in {"disabled", "not_in_training_group"}:
                    return reason
        return str(outcome.reason or "")

    def _record_idle_dynamic_batch_cap(
        self,
        plan: TrainingPlan,
        outcome: TrainingOutcome,
    ) -> None:
        if plan.execution_strategy is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            return
        if plan.reason in {
            "quota_topup_training_ready",
            "quota_forced_completion_ready",
        }:
            return
        group = _normalize_worker_id_group(plan.target_worker_ids)
        previous = max(
            int(
                self._idle_group_batch_caps.get(
                    group,
                    self._idle_worker_dynamic_batch_cap
                    if not self._idle_group_batch_caps
                    else None,
                )
                or plan.max_batches
                or 1
            ),
            1,
        )
        next_cap = previous
        stop_reason = self._idle_stop_reason(outcome, plan)
        # Preflight/data-version failures say nothing about how many batches
        # fit in an idle window.  Do not poison the runtime capacity estimate
        # with a data-plane race or a temporarily unavailable snapshot.
        capacity_neutral_reasons = {
            "buffer_version_changed",
            "data_version_changed",
            "data_reservation_failed",
            "insufficient_worker_data",
            "missing_worker_snapshot",
            "preflight_not_ready",
            "target_version_mismatch",
            "target_version_unavailable",
            "worker_restarted",
            "plan_expired_before_preflight",
            "plan_expired_during_preflight",
        }
        reported_reasons = {str(outcome.reason or ""), stop_reason}
        reported_reasons.update(
            str(result.get("reason") or result.get("stop_reason") or "")
            for result in outcome.raw_results
            if isinstance(result, dict)
        )
        if reported_reasons & capacity_neutral_reasons:
            return
        if bool(outcome.metrics.get("bubble/train_reclaimed_before_first_batch", 0)):
            # A request-level speculative idle event can be reclaimed before
            # batch one without saying anything about steady-state capacity.
            # Keep the cap unchanged; confirmed post-start spill below remains
            # the only reclaim signal that can reduce it.
            return
        elif not outcome.trained or outcome.successful_steps <= 0:
            return
        elif stop_reason == "reclaim_requested":
            # An early rollout return invalidates this opportunity, not the
            # writer's capacity in the next window. Deadline/budget and the
            # generation-slowdown guard still bound subsequent work.
            return
        elif stop_reason == "deadline_reached":
            if plan.idle_window_source in {
                "historical_observed",
                "historical_survival",
            }:
                # Only a measured, in-budget completion is capacity-neutral.
                # Missing timing or a real overrun retains the old reduction.
                try:
                    elapsed = float(outcome.metrics["timing_s/drafter_worker_elapsed"])
                    window = (
                        float(plan.idle_window_sec)
                        if plan.idle_window_sec is not None
                        else 0.0
                    )
                except (KeyError, TypeError, ValueError):
                    elapsed, window = math.inf, 0.0
                if math.isfinite(elapsed) and 0 <= elapsed <= window:
                    outcome.metrics[
                        "bubble/predicted_window_exhausted_without_overrun"
                    ] = 1
                    return
            next_cap = max(1, min(previous, max(outcome.successful_steps, 1)))
        elif stop_reason == "max_batches_reached" or outcome.successful_steps >= max(
            int(plan.max_batches), 1
        ):
            next_cap = max(previous * 2, max(int(plan.max_batches), 1) * 2)
        self._idle_group_batch_caps[group] = max(int(next_cap), 1)
        self._idle_worker_dynamic_batch_cap = max(int(next_cap), 1)
        if self._idle_worker_dynamic_batch_cap != previous:
            print(
                "[BubbleTime] idle_dynamic_batch_cap_updated: "
                f"plan_id={plan.plan_id} source_step={plan.source_global_step} "
                f"previous={previous} current={self._idle_worker_dynamic_batch_cap} "
                f"successful_steps={outcome.successful_steps} "
                f"stop_reason={stop_reason}",
                flush=True,
            )

    def record_step_metrics(
        self,
        metrics: dict[str, Any],
        config: DrafterScheduleConfig,
        *,
        global_step: object | None = None,
    ) -> dict[str, float | int]:
        """Record quality feedback and fail closed on generation interference."""

        if (
            config.execution_strategy
            is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        ):
            return {}
        observed_step = global_step
        if observed_step is None:
            observed_step = metrics.get(
                "training/global_step", metrics.get("global_step")
            )
        try:
            if observed_step is not None:
                self._last_observed_global_step = max(
                    _as_int(observed_step),
                    self._last_observed_global_step or 0,
                )
        except (TypeError, ValueError):
            pass

        acceptance = None
        for key in (
            "drafter/spec_decode/mean_acceptance_length",
            "dflash2/mean_acceptance_length",
            "spec_decode/mean_acceptance_length",
        ):
            try:
                if metrics.get(key) is not None:
                    acceptance = _as_float(metrics[key])
                    break
            except (TypeError, ValueError):
                continue
        if acceptance is not None and acceptance > 0.0:
            self._quality_latest_acceptance = acceptance
            if (
                self._quality_pending_acceptance_baseline
                and self._last_observed_global_step is not None
                and (
                    self._quality_last_publish_step is None
                    or self._last_observed_global_step > self._quality_last_publish_step
                )
            ) or (
                not self._quality_pending_acceptance_baseline
                and self._quality_acceptance_baseline is None
            ):
                self._quality_acceptance_baseline = acceptance
                if self._quality_pending_acceptance_baseline:
                    self._quality_pending_acceptance_baseline = False
            elif (
                self._quality_acceptance_baseline is not None
                and acceptance > self._quality_acceptance_baseline
            ):
                # Track improvements slowly, but never adapt the baseline
                # downward and hide a real quality regression.
                self._quality_acceptance_baseline = (
                    self._quality_acceptance_baseline * 0.8 + acceptance * 0.2
                )
        gen_ms = None
        try:
            if metrics.get("timing_per_token_ms/gen") is not None:
                gen_ms = _as_float(metrics.get("timing_per_token_ms/gen"))
        except (TypeError, ValueError):
            gen_ms = None
        if gen_ms is None:
            try:
                gen_s = _as_float(metrics.get("timing_s/gen"))
                tokens = _as_float(metrics.get("perf/total_num_tokens"))
                if tokens > 0:
                    gen_ms = gen_s * 1000.0 / tokens
            except (TypeError, ValueError):
                gen_ms = None
        current_cap = self._effective_idle_dynamic_batch_cap(config)
        result: dict[str, float | int] = {
            "bubble/idle_dynamic_batch_cap": current_cap,
        }
        result.update(self._quality_gate_metrics(config))
        if gen_ms is None or gen_ms <= 0:
            return result
        try:
            feedback_step = (
                _as_int(observed_step) if observed_step is not None else None
            )
        except (TypeError, ValueError):
            feedback_step = None
        if feedback_step is not None:
            if (
                self._idle_worker_last_feedback_step is not None
                and feedback_step <= self._idle_worker_last_feedback_step
            ):
                # Training and deferred publication may merge the same step's
                # metrics more than once. One rollout is one slowdown sample.
                return result
            self._idle_worker_last_feedback_step = feedback_step
        idle_active = (
            any(
                bool(metrics.get(key, 0))
                for key in (
                    "drafter/idle_trained",
                    "scheduler/train_launched",
                    "drafter/runtime_inflight",
                )
            )
            or float(metrics.get("timing_s/drafter_async_training_work", 0.0) or 0.0)
            > 0.0
        )
        baseline = self._idle_worker_gen_per_token_baseline_ms
        if not idle_active:
            self._idle_worker_gen_slowdown_streak = 0
            self._idle_worker_gen_per_token_samples.append(gen_ms)
            if baseline is None:
                self._idle_worker_gen_per_token_baseline_ms = gen_ms
            else:
                self._idle_worker_gen_per_token_baseline_ms = (
                    baseline * 0.8 + gen_ms * 0.2
                )
            result["bubble/gen_slowdown_cap_reduced"] = 0
            result["bubble/gen_per_token_baseline_ms"] = (
                self._idle_worker_gen_per_token_baseline_ms
            )
            return result
        if baseline is None:
            result["bubble/gen_slowdown_cap_reduced"] = 0
            return result
        ratio = gen_ms / max(baseline, 1.0e-9)
        result["bubble/gen_slowdown_ratio"] = ratio
        threshold = 1.0 + float(config.idle_worker_gen_slowdown_threshold)
        if config.idle_worker_dynamic_batch_cap and ratio > threshold:
            self._idle_worker_gen_slowdown_streak += 1
            if current_cap > 1:
                previous = current_cap
                self._idle_worker_dynamic_batch_cap = max(
                    1, int(math.ceil(current_cap / 2))
                )
                group = self._current_idle_writer_group(assign_default=False)
                if group:
                    self._idle_group_batch_caps[group] = (
                        self._idle_worker_dynamic_batch_cap
                    )
                result["bubble/gen_slowdown_cap_reduced"] = 1
                result["bubble/idle_dynamic_batch_cap"] = (
                    self._idle_worker_dynamic_batch_cap
                )
                print(
                    "[BubbleTime] idle_dynamic_batch_cap_reduced: "
                    f"reason=gen_slowdown previous={previous} "
                    f"current={self._idle_worker_dynamic_batch_cap} "
                    f"gen_per_token_ms={gen_ms:.6f} baseline_ms={baseline:.6f} "
                    f"ratio={ratio:.3f} threshold={threshold:.3f}",
                    flush=True,
                )
            else:
                result["bubble/gen_slowdown_cap_reduced"] = 0
            if (
                self._idle_worker_gen_slowdown_streak
                >= int(config.idle_worker_gen_slowdown_patience)
                and self._last_observed_global_step is not None
            ):
                self._idle_worker_suspended_until_step = (
                    self._last_observed_global_step
                    + int(config.idle_worker_gen_slowdown_cooldown_steps)
                )
                result["bubble/idle_training_suspended"] = 1
                result["bubble/idle_training_suspended_until_step"] = int(
                    self._idle_worker_suspended_until_step
                )
                print(
                    "[BubbleTime] idle_training_suspended: "
                    "reason=repeated_gen_slowdown "
                    f"step={self._last_observed_global_step} "
                    f"until_step={self._idle_worker_suspended_until_step} "
                    f"ratio={ratio:.3f} threshold={threshold:.3f}",
                    flush=True,
                )
        else:
            self._idle_worker_gen_slowdown_streak = 0
            result["bubble/gen_slowdown_cap_reduced"] = 0
        return result

    def _quality_gate_metrics(
        self,
        config: DrafterScheduleConfig,
    ) -> dict[str, float | int]:
        baseline = self._quality_acceptance_baseline
        latest = self._quality_latest_acceptance
        drop_ratio = 0.0
        if baseline is not None and latest is not None and baseline > 0.0:
            drop_ratio = max((baseline - latest) / baseline, 0.0)
        return {
            "bubble/quality_trigger_adaptive": int(
                config.training_quota_trigger_mode == "adaptive"
            ),
            "bubble/quality_acceptance_baseline": float(baseline or 0.0),
            "bubble/quality_acceptance_latest": float(latest or 0.0),
            "bubble/quality_acceptance_drop_ratio": float(drop_ratio),
            "bubble/quality_loss_baseline": float(self._quality_loss_baseline or 0.0),
            "bubble/quality_loss_latest": float(self._quality_latest_loss or 0.0),
            "bubble/quality_loss_refresh_requested": int(
                self._quality_loss_refresh_requested
            ),
        }

    def _adaptive_quality_refresh_due(
        self,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        if (
            not config.training_quota_enable
            or config.execution_strategy
            is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or config.training_quota_trigger_mode != "adaptive"
        ):
            return True
        # Bootstrap exactly once so adaptive mode cannot become a no-training
        # mode before it has a published drafter and a quality baseline.
        if self._quality_last_publish_step is None:
            return True
        step = _as_int(global_step)
        age = max(step - self._quality_last_publish_step, 0)
        if age < int(config.training_quota_min_refresh_interval_steps):
            return False
        max_age = int(config.training_quota_max_refresh_interval_steps)
        if max_age > 0 and age >= max_age:
            return True
        baseline = self._quality_acceptance_baseline
        latest = self._quality_latest_acceptance
        acceptance_due = bool(
            baseline is not None
            and latest is not None
            and latest
            <= baseline * (1.0 - float(config.training_quota_acceptance_drop_ratio))
        )
        return acceptance_due or self._quality_loss_refresh_requested

    def _idle_gradient_accumulation_steps(
        self,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...],
    ) -> int:
        """Match the sync path's effective data-parallel batch in Bubble mode."""

        group = _normalize_worker_id_group(
            worker_ids
        ) or self._current_idle_writer_group(assign_default=False)
        if not group:
            return max(int(config.gradient_accumulation_steps), 1)
        group_size = len(group)
        sync_size = max(
            (len(group) for group in self._metadata_full_collective_idle_groups),
            default=group_size,
        )
        if sync_size % group_size:
            raise ValueError(
                "Bubble writer group cannot match the Sync effective batch: non-divisible group sizes"
            )
        replica_ratio = max(sync_size // group_size, 1)
        return max(int(config.gradient_accumulation_steps), 1) * replica_ratio

    def bind_worker_executor(self, worker_executor: DrafterWorkerExecutor) -> None:
        """Bind the worker execution port used by all execution strategies."""

        self._worker_executor = worker_executor

    def bind_standalone_collection_executor(
        self, executor: StandaloneCollectionExecutor
    ) -> None:
        self._standalone_collection_executor = executor

    def bind_standalone_training_executor(
        self, executor: StandaloneTrainingExecutor
    ) -> None:
        self._standalone_training_executor = executor

    @staticmethod
    def plan_queue_collection(
        context: QueueScheduleContext,
    ) -> CollectionPlan:
        """Build a queue Producer plan with high/low-watermark backpressure."""

        status = context.queue_status
        config = context.config
        if context.producer_done:
            action = ProducerAction.STOP
            reason = "producer_done"
        elif status.ready_samples >= config.high_watermark_samples:
            action = ProducerAction.PAUSE
            reason = "high_watermark_reached"
        elif status.ready_samples <= config.low_watermark_samples:
            action = ProducerAction.RUN
            reason = "low_watermark_reached"
        elif context.producer_paused:
            action = ProducerAction.PAUSE
            reason = "watermark_hysteresis_paused"
        else:
            action = ProducerAction.RUN
            reason = "watermark_hysteresis_running"
        return CollectionPlan(
            collect=action is ProducerAction.RUN,
            reason=reason,
            source=DrafterCollectionSource.TRANSFER_QUEUE,
            source_global_step=0,
            collect_interval_matched=True,
            training_interval_matched=True,
            sample_rate=1.0,
            max_samples_per_replica=None,
            max_tokens_per_replica=None,
            hidden_window_mode="front",
            hidden_window_tokens_per_sample=None,
            hidden_window_min_rows=0,
            producer_action=action,
            max_new_samples=None,
        )

    @staticmethod
    def plan_queue_training(
        context: QueueScheduleContext,
        *,
        selected_keys: tuple[str, ...] = (),
    ) -> TrainingPlan:
        """Plan at most one complete standalone global batch."""

        status = context.queue_status
        config = context.config
        common: dict[str, Any] = {
            "interval_matched": True,
            "execution_strategy": DrafterExecutionStrategy.STANDALONE_ASYNC,
            "source_global_step": 0,
            "max_batches": 1,
            "publish_after_success": False,
            "min_batches": 1,
            "require_full_batch": True,
            "data_filter_reason": "transfer_queue",
            "data_source": DrafterTrainingDataSource.TRANSFER_QUEUE,
            "required_samples": config.global_batch_size,
        }
        if not context.consumer_ready:
            return TrainingPlan(launch=False, reason="consumer_initializing", **common)
        if (
            context.remaining_steps is not None
            and context.outstanding_batches >= context.remaining_steps
        ):
            return TrainingPlan(
                launch=False, reason="training_budget_reserved", **common
            )
        if context.consumer_training and context.max_outstanding_batches == 1:
            return TrainingPlan(
                launch=False,
                reason="consumer_training",
                **common,
            )
        if context.outstanding_batches >= context.max_outstanding_batches:
            return TrainingPlan(
                launch=False, reason="outstanding_capacity_reached", **common
            )
        if status.ready_samples < config.global_batch_size:
            return TrainingPlan(
                launch=False,
                reason="insufficient_ready_samples",
                **common,
            )
        return TrainingPlan(
            launch=True,
            reason="training_ready",
            selected_keys=selected_keys,
            **common,
        )

    @staticmethod
    def queue_budget_exhausted(context: QueueScheduleContext) -> bool:
        """Stop only after the restored training budget and all plans finish."""
        return (
            context.consumer_ready
            and context.remaining_steps is not None
            and context.remaining_steps <= 0
            and context.outstanding_batches == 0
        )

    def execute_standalone_collection_plan(
        self,
        plan: CollectionPlan,
        *,
        producer_paused: bool,
        producer_done: bool,
    ) -> StandaloneCollectionOutcome:
        if self._standalone_collection_executor is None:
            raise RuntimeError("Standalone collection executor has not been bound")
        return self.standalone_collection_strategy.execute(
            plan,
            executor=self._standalone_collection_executor,
            producer_paused=producer_paused,
            producer_done=producer_done,
        )

    def execute_standalone_training_plan(
        self,
        plan: TrainingPlan,
        *,
        runtime_state: DrafterRuntimeState,
        selected_entries: Sequence[Any],
    ) -> StandaloneTrainingOutcome:
        if self._standalone_training_executor is None:
            raise RuntimeError("Standalone training executor has not been bound")
        return self.standalone_training_strategy.execute(
            plan,
            executor=self._standalone_training_executor,
            runtime_state=runtime_state,
            selected_entries=selected_entries,
        )

    def complete_standalone_training(
        self,
        *,
        runtime_state: DrafterRuntimeState,
        completed_keys: Sequence[str],
        successful: bool,
    ) -> StandaloneTrainingOutcome:
        return self.standalone_training_strategy.complete(
            runtime_state=runtime_state,
            completed_keys=completed_keys,
            successful=successful,
        )

    def stop_standalone_consumer(self) -> Any:
        if self._standalone_training_executor is None:
            raise RuntimeError("Standalone training executor has not been bound")
        return self._standalone_training_executor.stop_consumer()

    def bind_publish_executor(self, publish_executor: DrafterPublishExecutor) -> None:
        self._publish_executor = publish_executor

    def bind_collection_executor(
        self, collection_executor: DrafterCollectionExecutor
    ) -> None:
        self._collection_executor = collection_executor

    def register_collection_adapter(self, adapter: DrafterCollectionAdapter) -> None:
        """Register or replace the payload adapter for a collection source."""
        self._collection_adapters[adapter.source] = adapter

    def prepare_collection_payload(
        self,
        *,
        source: DrafterCollectionSource,
        samples: list[dict],
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners=None,
    ) -> CollectionPayload:
        """Build the common payload without exposing bucketing to the Trainer."""
        adapter = self._collection_adapters.get(source)
        if adapter is None:
            raise ValueError(f"No collection adapter registered for {source.value}")
        return adapter.prepare_payload(
            samples,
            owner_count=owner_count,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=raw_samples,
            collection_id=collection_id,
            owners=owners,
        )

    def inspect_training_data(
        self,
        *,
        global_step: object,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...] | None = None,
        target_version: int | None = None,
    ):
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        probe_started = time.perf_counter()
        from verl_speco.integration.generation_diagnostics import background_span

        with background_span("data_probe"):
            statuses = self._worker_executor.get_training_data_status(
                sample_last_n_steps=config.sample_last_n_steps,
                require_full_batch=config.require_full_batch,
                worker_ids=worker_ids,
                target_version=target_version,
            )
        status = self.data_status_policy.aggregate(statuses, global_step=global_step)
        if config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            logger.debug(
                "[BubbleTime] training_data_probe: step=%s workers=%s "
                "target_version=%s elapsed_s=%.4f",
                global_step,
                worker_ids,
                target_version,
                time.perf_counter() - probe_started,
            )
        return status

    def on_worker_event(
        self,
        event: RolloutWorkerEvent | dict[str, object],
    ) -> dict[str, float | int]:
        """Record rollout replica state for Bubble Time idle-worker planning."""

        if isinstance(event, dict):
            if str(event.get("event_type", "")).lower() == "worker_idle":
                replica = _as_int(event.get("replica_rank", 0))
                timeline = {
                    key: (_as_float(event[key]) if event.get(key) is not None else None)
                    for key in (
                        "request_completed_ts",
                        "event_emit_ts",
                        "lease_released_ts",
                        "bus_received_ts",
                    )
                }
                timeline["driver_received_ts"] = time.time()
                self._replica_idle_timeline[replica] = timeline
                logger.debug(
                    "[BubbleTime] idle_event_timeline: replica=%s release_source=%s timestamps=%s",
                    replica,
                    event.get("release_source"),
                    timeline,
                )
            event = RolloutWorkerEvent(
                event_type=_rollout_worker_event_type(event.get("event_type", "")),
                worker_id=str(event.get("worker_id", "")),
                replica_rank=_as_int(event.get("replica_rank", 0)),
                memory_released=bool(event.get("memory_released", False)),
                release_source=str(event.get("release_source", "") or ""),
                idle_confidence=_idle_window_confidence(
                    event.get("idle_confidence", IdleWindowConfidence.CONFIRMED.value)
                ),
                must_be_ready_at=(
                    None
                    if event.get("must_be_ready_at") is None
                    else float(cast(Any, event.get("must_be_ready_at", 0.0)))
                ),
                event_ts=(
                    None
                    if event.get("event_ts") is None
                    else float(cast(Any, event.get("event_ts", 0.0)))
                ),
            )
        elif not isinstance(event.event_type, RolloutWorkerEventType):
            event = RolloutWorkerEvent(
                event_type=_rollout_worker_event_type(event.event_type),
                worker_id=event.worker_id,
                replica_rank=event.replica_rank,
                memory_released=event.memory_released,
                release_source=event.release_source,
                idle_confidence=_idle_window_confidence(event.idle_confidence),
                must_be_ready_at=event.must_be_ready_at,
                event_ts=event.event_ts,
            )
        event_ts = event.event_ts if event.event_ts is not None else time.time()
        delivery_lag_sec = max(time.time() - event_ts, 0.0)
        self._replica_idle_event_delivery_lag_sec[event.replica_rank] = delivery_lag_sec
        if event.event_type is RolloutWorkerEventType.WORKER_IDLE:
            if event.memory_released:
                self._replica_idle_started_at[event.replica_rank] = event_ts
            else:
                self._replica_idle_started_at.pop(event.replica_rank, None)
        elif event.event_type is RolloutWorkerEventType.GENERATION_STARTED:
            self._record_observed_replica_idle_window(
                event.replica_rank,
                event_ts,
                source="generation_started",
            )
        worker_ids = self._replica_idle_worker_groups.get(event.replica_rank)
        if not worker_ids:
            worker_ids = (event.worker_id,)
        for worker_id in worker_ids:
            self._record_worker_event_state(event, worker_id, event_ts)
        logger.info(
            "[BubbleTime] worker_event type=%s worker_id=%s replica_rank=%s "
            "expanded_worker_ids=%s memory_released=%s release_source=%s must_be_ready_at=%s "
            "event_ts=%s idle_state=%s",
            event.event_type.value,
            event.worker_id,
            event.replica_rank,
            worker_ids,
            event.memory_released,
            event.release_source,
            event.must_be_ready_at,
            event_ts,
            _idle_state_summary(self._idle_workers, now=event_ts),
        )
        return self.idle_worker_metrics()

    def _effective_idle_reclaim_penalty_sec(self) -> float:
        return max(float(self._idle_worker_reclaim_penalty_sec), 0.0)

    def _decay_idle_reclaim_penalty(self, global_step: object) -> None:
        if self._idle_worker_reclaim_penalty_sec <= 0.0:
            return
        try:
            step = _as_int(global_step)
        except (TypeError, ValueError):
            return
        last_step = self._idle_worker_reclaim_penalty_last_step
        if last_step is None:
            self._idle_worker_reclaim_penalty_last_step = step
            return
        # Decay on every later step.  The penalty is only a short-lived
        # admission guard for windows that were reclaimed before batch 1; if it
        # stays sticky, it can incorrectly consume otherwise usable rollout
        # idle windows for many steps.
        decay_steps = max(step - last_step, 0)
        if decay_steps <= 0:
            return
        previous = self._idle_worker_reclaim_penalty_sec
        self._idle_worker_reclaim_penalty_sec *= 0.5 ** min(decay_steps, 8)
        self._idle_worker_reclaim_penalty_last_step = step
        if self._idle_worker_reclaim_penalty_sec < 0.25:
            self._idle_worker_reclaim_penalty_sec = 0.0
            self._idle_worker_prebatch_reclaim_streak = 0
        print(
            "[BubbleTime] idle_reclaim_penalty_decayed: "
            f"step={step} previous_s={previous:.3f} "
            f"current_s={self._idle_worker_reclaim_penalty_sec:.3f} "
            f"decay_steps={decay_steps}",
            flush=True,
        )

    def _record_prebatch_reclaim_penalty(
        self,
        plan: TrainingPlan,
        outcome: TrainingOutcome,
    ) -> None:
        """Adjust the dynamic admission penalty after zero-batch reclaims.

        A zero-batch reclaim means the scheduler managed to reserve the worker
        group, but the next rollout reclaimed it before the first batch could
        start.  Treat that failed estimate as an admission-risk signal instead
        of a fixed step cooldown: future windows that are clearly larger can
        still train, while similarly marginal windows are skipped.
        """

        reclaimed_before_first_batch = bool(
            outcome.metrics.get("bubble/train_reclaimed_before_first_batch", 0)
        )
        if outcome.trained and outcome.successful_steps > 0:
            self._writer_boundary_migration_allowed = False
            if self._idle_worker_prebatch_reclaim_streak:
                print(
                    "[BubbleTime] idle_reclaim_penalty_reset: "
                    f"plan_id={plan.plan_id} source_step={plan.source_global_step} "
                    f"successful_steps={outcome.successful_steps} "
                    f"previous_s={self._idle_worker_reclaim_penalty_sec:.3f}",
                    flush=True,
                )
            self._idle_worker_prebatch_reclaim_streak = 0
            self._idle_worker_reclaim_penalty_sec = 0.0
            self._idle_worker_reclaim_penalty_last_step = None
            return
        if not reclaimed_before_first_batch:
            return
        self._idle_worker_prebatch_reclaim_streak += 1
        try:
            source_step = _as_int(plan.source_global_step)
        except (TypeError, ValueError):
            source_step = 0
        worker_elapsed_sec = max(
            float(outcome.metrics.get("timing_s/drafter_worker_elapsed", 0.0) or 0.0),
            0.0,
        )
        failed_usable_window_sec = max(float(plan.idle_usable_window_sec or 0.0), 0.0)
        batch_estimate_sec = max(float(plan.idle_batch_estimate_sec or 0.0), 0.0)
        preflight_sec = max(
            float(outcome.metrics.get("timing_s/drafter_worker_preflight", 0.0) or 0.0),
            0.0,
        )
        preflight_to_stop_sec = max(
            float(
                outcome.metrics.get(
                    "timing_s/drafter_worker_preflight_to_stop",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
        startup_reserve_sec = max(float(plan.idle_startup_reserve_sec or 0.0), 0.0)
        observed_empty_launch_sec = max(
            worker_elapsed_sec,
            preflight_sec + preflight_to_stop_sec,
            startup_reserve_sec,
        )
        # Do not turn the whole failed window into a penalty.  A pre-batch
        # reclaim tells us that the admission margin was too optimistic, not
        # that a future window must be larger than the whole previous window.
        # Penalize by the observed empty-launch overhead plus one estimated
        # batch, cap it to a few batches, and allow the value to shrink after
        # repeated lower-cost failures.
        penalty_cap = max(batch_estimate_sec * 3.0, batch_estimate_sec)
        observed_penalty = min(
            observed_empty_launch_sec + batch_estimate_sec,
            penalty_cap,
        )
        previous_penalty = self._idle_worker_reclaim_penalty_sec
        if previous_penalty > 0.0:
            next_penalty = max(observed_penalty, previous_penalty * 0.5)
        else:
            next_penalty = observed_penalty
        next_penalty = min(next_penalty, penalty_cap)
        self._idle_worker_reclaim_penalty_sec = next_penalty
        self._idle_worker_reclaim_penalty_last_step = source_step
        print(
            "[BubbleTime] idle_reclaim_penalty_updated: "
            f"plan_id={plan.plan_id} source_step={plan.source_global_step} "
            f"streak={self._idle_worker_prebatch_reclaim_streak} "
            "reason=reclaim_before_first_batch "
            f"previous_penalty_s={previous_penalty:.3f} "
            f"penalty_s={self._idle_worker_reclaim_penalty_sec:.3f} "
            f"failed_usable_window_s={failed_usable_window_sec:.3f} "
            f"batch_estimate_s={batch_estimate_sec:.3f} "
            f"observed_empty_launch_s={observed_empty_launch_sec:.3f} "
            f"penalty_cap_s={penalty_cap:.3f} "
            f"worker_elapsed_s={worker_elapsed_sec:.3f} "
            f"preflight_s={preflight_sec:.3f} "
            f"preflight_to_stop_s={preflight_to_stop_sec:.3f}",
            flush=True,
        )

    def _record_observed_replica_idle_window(
        self,
        replica_rank: int,
        event_ts: float,
        *,
        source: str,
    ) -> float | None:
        idle_started_at = self._replica_idle_started_at.pop(replica_rank, None)
        if idle_started_at is None or event_ts <= idle_started_at:
            return None
        observed_window = event_ts - idle_started_at
        samples = self._replica_idle_window_samples_sec.setdefault(
            int(replica_rank), deque(maxlen=32)
        )
        samples.append(observed_window)
        source_samples = self._replica_idle_windows_by_source.setdefault(
            int(replica_rank), {}
        ).setdefault(source, deque(maxlen=32))
        source_samples.append(observed_window)
        self._replica_last_observed_idle_window_sec[int(replica_rank)] = observed_window
        self._replica_idle_window_close_source[int(replica_rank)] = str(source)
        logger.warning(
            "[BubbleTime] idle_window_timeline: replica=%s usable_idle_start_ts=%.6f "
            "window_closed_ts=%.6f observed_window_s=%.4f close_source=%s "
            "conservative_window_s=%.3f samples=%s event_delivery_lag_s=%.3f by_source_p75=%s",
            replica_rank,
            idle_started_at,
            event_ts,
            observed_window,
            source,
            self._effective_historical_idle_window_sec(replica_rank=replica_rank),
            len(samples),
            self._replica_idle_event_delivery_lag_sec.get(replica_rank, 0.0),
            {
                kind: _conservative_percentile(values, quantile=0.75)
                for kind, values in self._replica_idle_windows_by_source[
                    int(replica_rank)
                ].items()
            },
        )
        return observed_window

    def record_generation_completed(
        self,
        event_ts: float | None = None,
        *,
        confirm_speculative_idle: bool = False,
        must_be_ready_at: float | None = None,
    ) -> dict[str, float | int]:
        """Close runtime idle windows at the real rollout completion boundary."""

        event_ts = time.time() if event_ts is None else float(event_ts)
        observed = [
            (replica_rank, window)
            for replica_rank in tuple(self._replica_idle_started_at)
            if (
                window := self._record_observed_replica_idle_window(
                    replica_rank,
                    event_ts,
                    source="generation_completed",
                )
            )
            is not None
        ]
        promoted = 0
        observed_by_replica = dict(observed)
        history_row = {}
        groups = self._metadata_idle_training_groups or tuple(
            self._replica_idle_worker_groups.values()
        )
        for raw_group in groups:
            group = _normalize_worker_id_group(raw_group)
            replicas = self._replica_ranks_for_worker_group(group)
            if replicas and all(rank in observed_by_replica for rank in replicas):
                history_row[group] = min(observed_by_replica[rank] for rank in replicas)
        self._writer_idle_history.append(history_row)
        boundary_confirmed = 0
        if confirm_speculative_idle:
            for state in self._idle_workers.values():
                if state.status == "idle" and state.memory_released:
                    if state.idle_confidence is IdleWindowConfidence.SPECULATIVE:
                        promoted += 1
                    # Runtime idle callbacks are request-local: they may arrive
                    # before the outer generate call returns, and the scheduler
                    # often evaluates them only after post-rollout bookkeeping.
                    # Once the generation boundary is reached, every still-idle
                    # memory-released worker is an authoritative rollout-idle
                    # lease.  Mark all of them as boundary-confirmed so later
                    # admission uses the lease/fallback deadline instead of
                    # subtracting event age from the historical window and
                    # incorrectly shrinking real Bubble time to zero.
                    state.idle_confidence = IdleWindowConfidence.CONFIRMED
                    state.event_ts = event_ts
                    state.must_be_ready_at = must_be_ready_at
                    state.confirmed_at_generation_boundary = True
                    boundary_confirmed += 1
            if promoted:
                print(
                    "[BubbleTime] speculative_idle_confirmed: "
                    f"workers={promoted} generation_complete_ts={event_ts:.6f} "
                    f"must_be_ready_at={must_be_ready_at}",
                    flush=True,
                )
            if boundary_confirmed:
                print(
                    "[BubbleTime] generation_boundary_idle_confirmed: "
                    f"workers={boundary_confirmed} "
                    f"generation_complete_ts={event_ts:.6f} "
                    f"must_be_ready_at={must_be_ready_at}",
                    flush=True,
                )
        metrics: dict[str, float | int] = {
            "bubble/speculative_idle_confirmed": promoted,
            "bubble/generation_boundary_idle_confirmed": boundary_confirmed,
        }
        if observed:
            observed_windows = [window for _, window in observed]
            metrics.update(
                {
                    "bubble/observed_idle_windows": len(observed),
                    "bubble/observed_idle_window_min_s": min(observed_windows),
                    "bubble/observed_idle_window_max_s": max(observed_windows),
                    "bubble/observed_idle_window_mean_s": (
                        sum(observed_windows) / len(observed_windows)
                    ),
                    "bubble/historical_idle_window_s": (
                        self._effective_historical_idle_window_sec() or 0.0
                    ),
                }
            )
            for replica_rank, window in observed:
                metrics[f"bubble/replica_{replica_rank}_observed_idle_window_s"] = (
                    window
                )
                metrics[f"bubble/replica_{replica_rank}_historical_idle_window_s"] = (
                    self._effective_historical_idle_window_sec(
                        replica_rank=replica_rank
                    )
                    or 0.0
                )
                metrics[f"bubble/replica_{replica_rank}_idle_event_delivery_lag_s"] = (
                    self._replica_idle_event_delivery_lag_sec.get(replica_rank, 0.0)
                )
        # Also surface windows that were closed by a later GENERATION_STARTED
        # event before the outer generation boundary. Those request-gap
        # samples still drive admission and must not be invisible in step logs.
        metrics.update(self.idle_worker_metrics())
        return metrics

    def writer_switch_decision(self, config, *, cost_sec=0.0):
        """Compare paired, completed rollout observations, never a single event."""
        source = self.idle_writer_group()
        rows = list(self._writer_idle_history)[
            -config.idle_worker_writer_migration_history_steps :
        ]
        metrics = {
            "bubble/writer_switch_enabled": int(
                config.idle_worker_writer_migration_enable
            ),
            "bubble/writer_switch_history_samples": len(rows),
            "bubble/writer_switch_estimated_cost_s": float(cost_sec),
            "bubble/writer_switch_eligible": 0,
        }
        if not source or len(rows) < config.idle_worker_writer_migration_history_steps:
            return None, metrics
        candidates = []
        for raw_group in self._metadata_idle_training_groups:
            target = _normalize_worker_id_group(raw_group)
            if (
                target == source
                or len(target) != len(source)
                or target in self._disabled_replica_local_idle_groups
            ):
                continue
            if not all(source in row and target in row for row in rows):
                continue
            gains = [row[target] - row[source] for row in rows]
            mean_gain = sum(gains) / len(gains)
            wins = sum(
                gain >= config.idle_worker_writer_migration_min_idle_gain_sec
                for gain in gains
            )
            candidates.append((mean_gain, wins, target, gains))
        if not candidates:
            return None, metrics
        gain, wins, target, gains = max(candidates, key=lambda item: (item[0], item[1]))
        source_mean = sum(row[source] for row in rows) / len(rows)
        metrics.update(
            {
                "bubble/writer_switch_current_idle_mean_s": source_mean,
                "bubble/writer_switch_candidate_idle_mean_s": source_mean + gain,
                "bubble/writer_switch_missed_idle_s": sum(
                    max(value, 0.0) for value in gains
                ),
                "bubble/writer_switch_persistent_wins": wins,
                # Amortize over a bounded next-history-sized horizon, not forever.
                "bubble/writer_switch_estimated_gain_s": max(gain, 0.0) * len(rows),
            }
        )
        ready, versions = self._drafter_version_ready_for_writer_group(target)
        ready = ready and all(
            value == self._latest_published_drafter_version
            for value in versions.values()
        )
        metrics["bubble/writer_switch_candidate_ack_ready"] = int(ready)
        metrics["bubble/writer_switch_boundary_ready"] = int(
            self._writer_boundary_migration_allowed
        )
        eligible = (
            config.idle_worker_writer_migration_enable
            and self._writer_boundary_migration_allowed
            and self._latest_published_drafter_version is not None
            and ready
            and wins >= math.ceil(len(rows) * 0.75)
            and gain >= config.idle_worker_writer_migration_min_idle_gain_sec
            and gain * len(rows)
            > cost_sec * config.idle_worker_writer_migration_cost_margin
        )
        metrics["bubble/writer_switch_eligible"] = int(eligible)
        return (target if eligible else None), metrics

    def _record_worker_event_state(
        self,
        event: RolloutWorkerEvent,
        worker_id: str,
        event_ts: float,
    ) -> None:
        worker_id = str(worker_id)
        state = self._idle_workers.get(worker_id)
        if state is None:
            state = _IdleWorkerState(
                worker_id=worker_id,
                replica_rank=event.replica_rank,
            )
            self._idle_workers[worker_id] = state
        state.replica_rank = event.replica_rank
        if (
            event.event_type is RolloutWorkerEventType.GENERATION_STARTED
            and state.status != "generating"
            and event_ts > max(state.event_ts, state.epoch_event_watermark)
        ):
            state.generation_epoch += 1
        state.epoch_event_watermark = max(state.epoch_event_watermark, event_ts)
        state.event_ts = event_ts
        if event.event_type is RolloutWorkerEventType.GENERATION_STARTED:
            state.status = "generating"
            state.memory_released = False
            state.must_be_ready_at = None
            state.confirmed_at_generation_boundary = False
        elif event.event_type is RolloutWorkerEventType.WORKER_IDLE:
            state.status = "idle"
            state.memory_released = event.memory_released
            state.must_be_ready_at = event.must_be_ready_at
            state.idle_confidence = _idle_window_confidence(event.idle_confidence)
            state.confirmed_at_generation_boundary = False
        elif event.event_type is RolloutWorkerEventType.WORKER_RECLAIM_REQUESTED:
            state.status = "reclaiming"
        elif event.event_type is RolloutWorkerEventType.WORKER_READY:
            state.status = "ready"
            state.memory_released = False
            state.must_be_ready_at = None
            state.idle_confidence = IdleWindowConfidence.SPECULATIVE
            state.confirmed_at_generation_boundary = False

    def register_idle_training_resource_metadata(
        self,
        metadata: Any,
    ) -> dict[str, float | int]:
        """Register true drafter training groups discovered from workers.

        Metadata is intentionally authoritative over ``group_size``.  In sync
        mode workers may report the whole connected mesh as
        ``full_collective_ranks``; in Bubble Time workers can instead report a
        replica-local collective group so the scheduler can launch as soon as a
        complete rollout replica-local group is idle.
        """

        records = _flatten_metadata_records(metadata)
        self._runtime_replica_local_topology = (
            self._runtime_replica_local_topology
            or any(
                bool(record.get("in_drafter_train_group", False))
                and bool(record.get("full_collective_ranks"))
                and len(record["full_collective_ranks"])
                < len(record.get("sync_collective_ranks", ()))
                for record in records
            )
        )
        replica_group_members: dict[int, set[str]] = {}
        full_groups: list[tuple[str, ...]] = []
        full_collective_groups: list[tuple[str, ...]] = []
        seen_groups: set[tuple[str, ...]] = set()
        seen_full_collective_groups: set[tuple[str, ...]] = set()
        global_publish_leaders: list[str] = []
        for record in records:
            if not bool(record.get("in_drafter_train_group", False)):
                continue
            replica_rank = record.get("replica_rank")
            training_ranks = _normalize_worker_id_group(
                record.get("training_group_ranks", ())
            )
            if bool(record.get("is_global_publish_leader", False)):
                leader_worker_id = str(record.get("worker_id", record.get("rank", "")))
                if leader_worker_id:
                    global_publish_leaders.append(leader_worker_id)
            if replica_rank is not None and training_ranks:
                replica_group_members.setdefault(int(replica_rank), set()).update(
                    training_ranks
                )
            fallback_group = _normalize_worker_id_group(
                record.get("sync_collective_ranks", ())
            )
            if not fallback_group:
                fallback_group = _normalize_worker_id_group(
                    record.get("full_collective_ranks", ())
                )
            if not fallback_group:
                fallback_group = training_ranks
            if fallback_group and fallback_group not in seen_full_collective_groups:
                full_collective_groups.append(fallback_group)
                seen_full_collective_groups.add(fallback_group)
            if training_ranks in self._disabled_replica_local_idle_groups:
                continue
            idle_collective_scope = (
                str(record.get("idle_collective_scope", "") or "").strip().lower()
            )
            if idle_collective_scope == "replica_local":
                full_group = training_ranks
            else:
                full_group = _normalize_worker_id_group(
                    record.get("full_collective_ranks", ())
                )
                if not full_group:
                    full_group = training_ranks
            if full_group and full_group not in seen_groups:
                full_groups.append(full_group)
                seen_groups.add(full_group)
        replica_groups = {
            replica_rank: tuple(sorted(worker_ids, key=_natural_worker_sort_key))
            for replica_rank, worker_ids in sorted(replica_group_members.items())
        }
        self._replica_idle_worker_groups = replica_groups
        self._metadata_idle_training_groups = tuple(full_groups)
        self._metadata_full_collective_idle_groups = tuple(full_collective_groups)
        self._global_publish_leader_worker_id = (
            global_publish_leaders[0] if global_publish_leaders else None
        )
        metadata_summary = [
            {
                "rank": record.get("rank"),
                "worker_id": record.get("worker_id"),
                "replica_rank": record.get("replica_rank"),
                "in_group": bool(record.get("in_drafter_train_group", False)),
                "training_group_ranks": record.get("training_group_ranks", ()),
                "full_collective_ranks": record.get("full_collective_ranks", ()),
                "sync_collective_ranks": record.get("sync_collective_ranks", ()),
                "idle_collective_scope": record.get("idle_collective_scope", ""),
                "is_global_publish_leader": bool(
                    record.get("is_global_publish_leader", False)
                ),
                "reason": record.get("reason", ""),
            }
            for record in records
        ]
        logger.warning(
            "[BubbleTime] resource_metadata groups=%s replica_groups=%s records=%s "
            "full_collective_fallback_groups=%s record_summary=%s",
            self._metadata_idle_training_groups,
            self._replica_idle_worker_groups,
            len(records),
            self._metadata_full_collective_idle_groups,
            metadata_summary,
        )
        return {
            "bubble/registered_training_groups": len(full_groups),
            "bubble/registered_training_workers": len(
                {worker_id for group in full_groups for worker_id in group}
            ),
            "bubble/registered_replica_groups": len(replica_groups),
        }

    def training_work_metrics(self) -> dict[str, int]:
        """Monotonic counters avoid per-plan metric overwrite within a step."""
        return {
            "bubble/idle_optimizer_steps_total": self._idle_optimizer_steps_total,
            "bubble/critical_path_topup_optimizer_steps_total": self._critical_path_topup_optimizer_steps_total,
        }

    def idle_worker_metrics(self) -> dict[str, float | int]:
        complete_groups = 0
        for group in self._metadata_idle_training_groups:
            worker_ids = _normalize_worker_id_group(group)
            if worker_ids and all(
                (state := self._idle_workers.get(worker_id)) is not None
                and state.status == "idle"
                and state.memory_released
                for worker_id in worker_ids
            ):
                complete_groups += 1
        metrics: dict[str, float | int] = {
            **self.training_work_metrics(),
            "bubble/idle_workers": sum(
                int(state.status == "idle") for state in self._idle_workers.values()
            ),
            "bubble/idle_training_groups": complete_groups,
        }
        source_codes = {
            "generation_started": 1,
            "generation_completed": 2,
        }
        replica_ranks = set(self._replica_idle_worker_groups)
        replica_ranks.update(self._replica_idle_window_samples_sec)
        replica_ranks.update(self._replica_last_observed_idle_window_sec)
        for replica_rank in sorted(replica_ranks):
            prefix = f"bubble/replica_{replica_rank}"
            histories = self._replica_idle_windows_by_source.get(replica_rank, {})
            for source, label in (
                ("generation_started", "generation_start_closed"),
                ("generation_completed", "generation_end_closed"),
            ):
                history = histories.get(source, ())
                metrics[f"{prefix}_{label}_p75_s"] = _conservative_percentile(
                    history, quantile=0.75
                )
                metrics[f"{prefix}_{label}_samples"] = len(history)
            timeline = self._replica_idle_timeline.get(replica_rank, {})
            received = timeline.get("driver_received_ts")
            emitted = timeline.get("event_emit_ts")
            completed = timeline.get("request_completed_ts")
            released = timeline.get("lease_released_ts")
            if received is not None and emitted is not None:
                metrics[f"{prefix}_emit_to_driver_s"] = max(received - emitted, 0.0)
            if completed is not None and released is not None:
                metrics[f"{prefix}_request_to_lease_release_s"] = max(
                    released - completed, 0.0
                )
            metrics[f"{prefix}_observed_idle_window_s"] = float(
                self._replica_last_observed_idle_window_sec.get(replica_rank, 0.0)
            )
            metrics[f"{prefix}_historical_idle_window_s"] = float(
                self._effective_historical_idle_window_sec(replica_rank=replica_rank)
                or 0.0
            )
            metrics[f"{prefix}_idle_event_delivery_lag_s"] = float(
                self._replica_idle_event_delivery_lag_sec.get(replica_rank, 0.0)
            )
            metrics[f"{prefix}_idle_window_samples"] = len(
                self._replica_idle_window_samples_sec.get(replica_rank, ())
            )
            metrics[f"{prefix}_idle_window_close_source"] = source_codes.get(
                self._replica_idle_window_close_source.get(replica_rank, ""),
                0,
            )
        return metrics

    def rollout_idle_replica_ranks(self) -> tuple[int, ...]:
        """Return rollout replicas with registered drafter training resources."""

        return tuple(sorted(self._replica_idle_worker_groups))

    def rollout_idle_replica_groups(self) -> dict[int, tuple[str, ...]]:
        """Return an immutable copy of the authoritative replica/group mapping."""

        return {
            int(replica_rank): tuple(worker_ids)
            for replica_rank, worker_ids in self._replica_idle_worker_groups.items()
        }

    def rollout_idle_worker_ids_for_replica(
        self,
        replica_rank: int,
        *,
        fallback_worker_id: str | None = None,
    ) -> tuple[str, ...]:
        """Resolve drafter worker ids made idle by one rollout replica."""

        worker_ids = self._replica_idle_worker_groups.get(int(replica_rank), ())
        if worker_ids:
            return worker_ids
        if fallback_worker_id is not None:
            return (str(fallback_worker_id),)
        return (str(replica_rank),)

    def _idle_training_groups(
        self,
        config: DrafterScheduleConfig,
    ) -> tuple[tuple[str, ...], ...]:
        """Resolve legal collective groups for rollout-idle training.

        Explicit ``training_groups`` remains the most precise option.  In the
        common case, ``group_mode: auto`` builds stable groups from all known
        rollout workers, including workers that are currently still generating.
        That makes a half-idle collective group report ``incomplete`` instead
        of accidentally training only the idle subset.
        """

        if self._idle_worker_writer_migration_blocked:
            return ()
        if (
            config.idle_worker_full_collective_fallback
            and self._runtime_replica_local_topology
        ):
            raise ValueError(
                "full_collective_fallback cannot combine independent replica-local "
                "optimizers; it requires an actual full-collective training topology"
            )
        if config.idle_worker_training_groups:
            requested = tuple(
                _normalize_worker_id_group(group)
                for group in config.idle_worker_training_groups
                if _normalize_worker_id_group(group)
                not in self._disabled_replica_local_idle_groups
            )
            if self._metadata_idle_training_groups:
                legal = set(self._metadata_idle_training_groups)
                invalid = [group for group in requested if group not in legal]
                if invalid:
                    raise ValueError(
                        "idle_worker.training_groups must match complete runtime "
                        f"training collectives: invalid={invalid}, legal={sorted(legal)}"
                    )
            return requested
        if self._metadata_idle_training_groups:
            return tuple(
                group
                for group in self._metadata_idle_training_groups
                if _normalize_worker_id_group(group)
                not in self._disabled_replica_local_idle_groups
            )
        if (
            self._disabled_replica_local_idle_groups
            and config.idle_worker_full_collective_fallback
            and self._metadata_full_collective_idle_groups
        ):
            logger.warning(
                "[BubbleTime] idle_full_collective_fallback_enabled: "
                "disabled_replica_local_groups=%s fallback_groups=%s",
                tuple(sorted(self._disabled_replica_local_idle_groups)),
                self._metadata_full_collective_idle_groups,
            )
            return self._metadata_full_collective_idle_groups
        if self._disabled_replica_local_idle_groups:
            return ()
        if config.idle_worker_group_mode != "auto":
            return ()
        known_workers = tuple(sorted(self._idle_workers, key=_natural_worker_sort_key))
        if not known_workers:
            return ()
        if config.idle_worker_group_size is None:
            return ()
        group_size = config.idle_worker_group_size
        if group_size <= 1:
            return tuple((worker_id,) for worker_id in known_workers)
        groups: list[tuple[str, ...]] = []
        for start in range(0, len(known_workers), group_size):
            group = known_workers[start : start + group_size]
            if len(group) == group_size:
                normalized = _normalize_worker_id_group(group)
                if normalized not in self._disabled_replica_local_idle_groups:
                    groups.append(tuple(group))
        return tuple(groups)

    def rollout_generation_tail_complete(self) -> bool:
        """All known replicas released serving; there is no proven overlap left."""
        workers = {
            worker
            for group in self._replica_idle_worker_groups.values()
            for worker in group
        }
        return bool(workers) and all(
            (state := self._idle_workers.get(worker)) is not None
            and state.status == "idle"
            and state.memory_released
            for worker in workers
        )

    def select_idle_training_resources(
        self,
        config: DrafterScheduleConfig,
        *,
        now: float | None = None,
        continuation: bool = False,
    ) -> AvailableTrainingResources:
        now = time.time() if now is None else now
        if (
            self._idle_worker_suspended_until_step is not None
            and self._last_observed_global_step is not None
            and self._last_observed_global_step < self._idle_worker_suspended_until_step
        ):
            return AvailableTrainingResources(False, "generation_slowdown_cooldown")
        if (
            self._idle_worker_suspended_until_step is not None
            and self._last_observed_global_step is not None
            and self._last_observed_global_step
            >= self._idle_worker_suspended_until_step
        ):
            self._idle_worker_suspended_until_step = None
            self._idle_worker_gen_slowdown_streak = 0
        idle_states = {
            worker_id: state
            for worker_id, state in self._idle_workers.items()
            if state.status == "idle"
            and (
                not config.idle_worker_require_memory_released or state.memory_released
            )
        }
        groups = self._idle_training_groups(config)
        if not groups:
            if (
                config.idle_worker_group_mode == "auto"
                and not config.idle_worker_training_groups
                and not self._metadata_idle_training_groups
                and config.idle_worker_group_size is None
                and self._idle_workers
            ):
                logger.warning(
                    "[BubbleTime] idle_resource_skip reason=missing_training_group_metadata "
                    "group_mode=%s group_size=%s known_state=%s",
                    config.idle_worker_group_mode,
                    config.idle_worker_group_size,
                    _idle_state_summary(self._idle_workers, now=now),
                )
                return AvailableTrainingResources(
                    False, "missing_training_group_metadata"
                )
            logger.info(
                "[BubbleTime] idle_resource_skip reason=no_idle_worker "
                "groups=%s known_state=%s require_memory_released=%s",
                groups,
                _idle_state_summary(self._idle_workers, now=now),
                config.idle_worker_require_memory_released,
            )
            return AvailableTrainingResources(False, "no_idle_worker")
        incomplete_seen = False
        not_prewarmed_seen = False
        window_too_small_seen = False
        speculative_unconfirmed_seen = False
        writer_state_migration_seen = False
        stale_drafter_version_seen = False
        best_small_window: AvailableTrainingResources | None = None
        ready_candidates: list[
            tuple[
                int,
                tuple[str, ...],
                float,
                IdleWindowConfidence,
                tuple[float, ...],
                str,
            ]
        ] = []
        writer_group = self._current_idle_writer_group(assign_default=False)
        writer_has_private_state = (
            writer_group is not None
            and self._idle_worker_writer_state_version is not None
        )
        for index, group in enumerate(groups):
            group = _normalize_worker_id_group(group)
            if (
                not writer_has_private_state
                and writer_group is not None
                and self._idle_worker_bootstrap_collection_group == writer_group
                and group != writer_group
            ):
                # Before the first optimizer step, ownership is already fixed
                # by collection routing. Other replicas have no copy of this
                # quota's complete pool and cannot safely replace its writer.
                incomplete_seen = True
                logger.info(
                    "[BubbleTime] idle_group_not_collection_writer group=%s "
                    "writer_group=%s reason=bootstrap_data_owned_by_writer",
                    group,
                    writer_group,
                )
                continue
            if writer_has_private_state and group != writer_group:
                not_prewarmed_seen = True
                writer_state_migration_seen = True
                logger.info(
                    "[BubbleTime] idle_group_not_writer group_id=idle-group-%s "
                    "group=%s writer_group=%s writer_state_version=%s "
                    "hot_groups=%s reason=writer_state_migration_required",
                    index,
                    group,
                    writer_group,
                    self._idle_worker_writer_state_version,
                    tuple(sorted(self._idle_worker_hot_prewarmed_groups)),
                )
                continue
            if not writer_has_private_state:
                version_ready, group_versions = (
                    self._drafter_version_ready_for_writer_group(group)
                )
                if not version_ready:
                    stale_drafter_version_seen = True
                    logger.info(
                        "[BubbleTime] idle_group_stale_drafter_version "
                        "group_id=idle-group-%s group=%s latest_published=%s "
                        "worker_versions=%s reason=stale_drafter_version",
                        index,
                        group,
                        self._latest_published_drafter_version,
                        group_versions,
                    )
                    continue
            missing = [worker_id for worker_id in group if worker_id not in idle_states]
            if missing:
                incomplete_seen = True
                logger.info(
                    "[BubbleTime] idle_group_incomplete group_id=idle-group-%s "
                    "group=%s missing=%s idle_workers=%s known_state=%s",
                    index,
                    group,
                    missing,
                    tuple(sorted(idle_states, key=_natural_worker_sort_key)),
                    _idle_state_summary(self._idle_workers, now=now),
                )
                continue
            windows = [
                max(float(state.must_be_ready_at) - now, 0.0)
                for worker_id in group
                if (state := idle_states[worker_id]).must_be_ready_at is not None
            ]
            event_ages = tuple(
                round(max(now - idle_states[worker_id].event_ts, 0.0), 3)
                for worker_id in group
            )
            historical_window = self._effective_historical_idle_window_sec(
                worker_ids=group
            )
            group_confidence = (
                IdleWindowConfidence.CONFIRMED
                if all(
                    idle_states[worker_id].idle_confidence
                    is IdleWindowConfidence.CONFIRMED
                    for worker_id in group
                )
                else IdleWindowConfidence.SPECULATIVE
            )
            if group_confidence is IdleWindowConfidence.SPECULATIVE:
                speculative_unconfirmed_seen = True
                logger.info(
                    "[BubbleTime] idle_resource_skip "
                    "reason=speculative_idle_unconfirmed group_id=idle-group-%s "
                    "group=%s event_age_s=%s source=request_local_idle",
                    index,
                    group,
                    event_ages,
                )
                continue
            has_runtime_deadline = bool(windows)
            generation_boundary_deadline = all(
                idle_states[worker_id].confirmed_at_generation_boundary
                for worker_id in group
            )
            source = "runtime_deadline" if has_runtime_deadline else "bootstrap_minimum"
            minimum_window = min(windows, default=math.inf)
            historical_remaining = None
            # The outer generation boundary owns an authoritative post-rollout
            # lease. Do not let short request-local gaps cap that deadline.
            # Request-local windows retain the historical safety bound even
            # when an atomic resource lease is available.  The lease prevents
            # rollout/training overlap; it is not evidence that the idle
            # window will last long enough and must never replace its estimate.
            if historical_window is not None and not generation_boundary_deadline:
                historical_remaining = min(
                    max(
                        historical_window
                        - max(now - idle_states[worker_id].event_ts, 0.0),
                        0.0,
                    )
                    for worker_id in group
                )
                if math.isinf(minimum_window) or historical_remaining < minimum_window:
                    source = "historical_observed"
                minimum_window = min(minimum_window, historical_remaining)
            if math.isinf(minimum_window):
                minimum_window = self._minimum_idle_training_window_sec(
                    config,
                    worker_ids=group,
                    continuation=continuation,
                )
            min_idle_window_sec = self._minimum_idle_training_window_sec(
                config,
                worker_ids=group,
                continuation=continuation,
            )
            if minimum_window < min_idle_window_sec and not continuation:
                retry_window = self._historical_idle_readmission_window(
                    config, group, now=now
                )
                if retry_window is not None and retry_window >= min_idle_window_sec:
                    minimum_window = retry_window
                    source = "historical_survival"
            if minimum_window < min_idle_window_sec:
                window_too_small_seen = True
                logger.debug(
                    "[BubbleTime] idle_resource_window_too_small: reason=window_too_small "
                    "group_id=idle-group-%s group=%s minimum_window_s=%.3f "
                    "min_required_s=%.3f budget_deficit_s=%.3f historical_window_s=%s source=%s "
                    "replica_ranks=%s replica_history_s=%s event_age_s=%s "
                    "startup_reserve_s=%.3f tail_reserve_s=%.3f final_tail_reserve_s=%.3f writer_group=%s "
                    "batch_estimate_s=%.3f guard_s=%.3f reclaim_penalty_s=%.3f "
                    "prebatch_reclaim_streak=%s now=%.3f idle_state=%s",
                    index,
                    group,
                    minimum_window,
                    min_idle_window_sec,
                    max(min_idle_window_sec - minimum_window, 0.0),
                    historical_window,
                    source,
                    self._replica_ranks_for_worker_group(group),
                    {
                        replica_rank: self._effective_historical_idle_window_sec(
                            replica_rank=replica_rank
                        )
                        for replica_rank in self._replica_ranks_for_worker_group(group)
                    },
                    event_ages,
                    self._effective_idle_startup_reserve_sec(config, group),
                    self._effective_idle_tail_reserve_sec(config),
                    self._effective_idle_tail_reserve_sec(config, final=True),
                    writer_group,
                    self._effective_idle_batch_estimate_sec(config),
                    self._effective_idle_deadline_guard_sec(config),
                    self._effective_idle_reclaim_penalty_sec(),
                    self._idle_worker_prebatch_reclaim_streak,
                    now,
                    _idle_state_summary(self._idle_workers, now=now),
                )
                small_window = AvailableTrainingResources(
                    False,
                    "window_too_small",
                    training_group_id=f"idle-group-{index}",
                    worker_ids=group,
                    minimum_idle_window_sec=minimum_window,
                    idle_confidence=group_confidence,
                    idle_window_source=source,
                )
                if (
                    best_small_window is None
                    or small_window.minimum_idle_window_sec
                    > best_small_window.minimum_idle_window_sec
                ):
                    best_small_window = small_window
                continue
            logger.info(
                "[BubbleTime] idle_resource_ready group_id=idle-group-%s group=%s "
                "minimum_window_s=%.3f min_required_s=%.3f "
                "historical_window_s=%s source=%s "
                "replica_ranks=%s replica_history_s=%s "
                "prebatch_reclaim_streak=%s now=%.3f",
                index,
                group,
                minimum_window,
                min_idle_window_sec,
                historical_window,
                source,
                self._replica_ranks_for_worker_group(group),
                {
                    replica_rank: self._effective_historical_idle_window_sec(
                        replica_rank=replica_rank
                    )
                    for replica_rank in self._replica_ranks_for_worker_group(group)
                },
                self._idle_worker_prebatch_reclaim_streak,
                now,
            )
            ready_candidates.append(
                (
                    index,
                    group,
                    minimum_window,
                    group_confidence,
                    event_ages,
                    source,
                )
            )
        if ready_candidates:
            (
                chosen_index,
                chosen_group,
                chosen_window,
                chosen_confidence,
                chosen_ages,
                chosen_source,
            ) = max(
                ready_candidates,
                key=lambda item: (item[2], min(item[4], default=0.0), -item[0]),
            )
            if chosen_source == "historical_survival":
                rank = self._replica_ranks_for_worker_group(chosen_group)[0]
                self._idle_readmission_attempts[chosen_group] = {
                    worker: self._idle_workers[worker].generation_epoch
                    for worker in chosen_group
                }
                logger.warning(
                    "[BubbleTime] idle_historical_readmission: group=%s replica=%s "
                    "idle_start_ts=%.6f age_s=%.3f remaining_estimate_s=%.3f "
                    "step_limit=window_budget generation_epochs=%s",
                    chosen_group,
                    rank,
                    self._replica_idle_started_at[rank],
                    now - self._replica_idle_started_at[rank],
                    chosen_window,
                    self._idle_readmission_attempts[chosen_group],
                )
            logger.warning(
                "[BubbleTime] idle_resource_selected: group_id=idle-group-%s "
                "group=%s window_s=%.3f confidence=%s event_age_s=%s "
                "writer_group=%s candidates=%s source=%s",
                chosen_index,
                chosen_group,
                chosen_window,
                chosen_confidence.value,
                chosen_ages,
                writer_group,
                [
                    {
                        "group_id": f"idle-group-{candidate[0]}",
                        "group": candidate[1],
                        "window_s": round(candidate[2], 3),
                        "event_age_s": candidate[4],
                    }
                    for candidate in ready_candidates
                ],
                chosen_source,
            )
            return AvailableTrainingResources(
                True,
                "training_group_ready",
                training_group_id=f"idle-group-{chosen_index}",
                worker_ids=chosen_group,
                minimum_idle_window_sec=chosen_window,
                idle_confidence=chosen_confidence,
                idle_window_source=chosen_source,
                idle_window_measured_at=(
                    now if chosen_source != "bootstrap_minimum" else None
                ),
            )
        reason = (
            "incomplete_training_group"
            if incomplete_seen
            else "speculative_idle_unconfirmed"
            if speculative_unconfirmed_seen
            else "window_too_small"
            if window_too_small_seen
            else "writer_state_migration_required"
            if writer_state_migration_seen
            else "stale_drafter_version"
            if stale_drafter_version_seen
            else "idle_group_not_prewarmed"
            if not_prewarmed_seen
            else "no_idle_worker"
        )
        logger.info(
            "[BubbleTime] idle_resource_skip reason=%s groups=%s idle_workers=%s "
            "known_state=%s hot_groups=%s best_small_window=%s",
            reason,
            groups,
            tuple(sorted(idle_states, key=_natural_worker_sort_key)),
            _idle_state_summary(self._idle_workers, now=now),
            tuple(sorted(self._idle_worker_hot_prewarmed_groups)),
            best_small_window,
        )
        if reason == "window_too_small" and best_small_window is not None:
            return best_small_window
        return AvailableTrainingResources(False, reason)

    def prepare_training_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        idle_continuation_plan: TrainingPlan | None = None,
    ) -> TrainingPlan:
        """Build a plan while avoiding worker RPCs for cheap skip conditions."""

        interval_matched = self.training_interval_matched(context.global_step, config)
        if config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            return self.prepare_idle_worker_training_plan(
                context,
                config,
                continuation_plan=idle_continuation_plan,
            )
        if (
            context.training_mode == "collect_only"
            or context.pending_training_count > 0
            or not interval_matched
        ):
            return self.plan_training(context, config)
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step, config=config
        )
        return self.plan_training(
            DrafterScheduleContext(
                global_step=context.global_step,
                training_mode=context.training_mode,
                collected_samples_this_step=context.collected_samples_this_step,
                oldlogprob_collection_requested=context.oldlogprob_collection_requested,
                data_status=data_status,
                pending_training_count=context.pending_training_count,
            ),
            config,
        )

    def idle_continuation_for_episode(
        self,
        plan: TrainingPlan,
        config: DrafterScheduleConfig,
    ) -> TrainingPlan | None:
        """Keep the old bound unless the entire writer has a new idle epoch.

        A duplicate idle callback or a changed prediction is not a new lease.
        Unknown/legacy epoch snapshots fail closed and retain the old deadline.
        """
        epochs = dict(plan.idle_generation_epochs)
        group = tuple(plan.target_worker_ids)
        if (
            not epochs
            or set(epochs) != set(group)
            or group != self._current_idle_writer_group(assign_default=False)
            or plan.data_version != self._training_quota_data_version
            or (
                not config.use_logits
                and plan.required_target_version != self._training_quota_data_version
            )
        ):
            return plan
        for worker in group:
            state = self._idle_workers.get(worker)
            if (
                state is None
                or state.generation_epoch <= epochs[worker]
                or state.status != "idle"
                or not state.memory_released
                or state.idle_confidence is not IdleWindowConfidence.CONFIRMED
            ):
                return plan
        return None

    def prepare_idle_worker_training_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        continuation_plan: TrainingPlan | None = None,
    ) -> TrainingPlan:
        self._decay_idle_reclaim_penalty(context.global_step)
        if self._quota_settled_without_new_collection(context, config):
            return self._skip_idle_worker_plan(
                context,
                config,
                AvailableTrainingResources(False, "training_quota_complete"),
            )
        if (
            config.training_quota_enable
            and self._training_quota_collection_step is not None
            and self._training_quota_debt_steps <= 0
        ):
            return self._skip_idle_worker_plan(
                context,
                config,
                AvailableTrainingResources(
                    available=False,
                    reason="training_quota_publish_pending",
                ),
            )
        if continuation_plan is not None:
            continuation_plan = self.idle_continuation_for_episode(
                continuation_plan, config
            )
        readmission = False
        if (
            continuation_plan is not None
            and not continuation_plan.idle_readmission_blocked
            and continuation_plan.idle_window_source
            in {"historical_observed", "historical_survival"}
            and continuation_plan.deadline_ts is not None
            and continuation_plan.deadline_ts <= time.time()
            and (
                config.use_logits
                or continuation_plan.required_target_version
                == self._training_quota_data_version
            )
            and continuation_plan.data_version == self._training_quota_data_version
        ):
            group = tuple(continuation_plan.target_worker_ids)
            retry_window = self._historical_idle_readmission_window(
                config, group, now=time.time()
            )
            readmission = (
                retry_window is not None
                and retry_window
                >= self._minimum_idle_training_window_sec(config, worker_ids=group)
            )
        resources = self.select_idle_training_resources(
            config, continuation=continuation_plan is not None and not readmission
        )
        if readmission:
            assert continuation_plan is not None
            if resources.idle_window_source == "historical_survival" and tuple(
                resources.worker_ids
            ) == tuple(continuation_plan.target_worker_ids):
                # A new bounded attempt, not continuation of the expired plan.
                # Selection already consumed this idle episode's retry token.
                continuation_plan = None
            else:
                resources = replace(
                    resources, available=False, reason="window_too_small"
                )
        if continuation_plan is not None and resources.available:
            # A completed small plan does not buy a new historical window.
            # Stay on the same writer/data snapshot and retain the original
            # absolute deadline, including time spent resolving its outcome.
            deadline = continuation_plan.deadline_ts
            if deadline is None or tuple(resources.worker_ids) != tuple(
                continuation_plan.target_worker_ids
            ):
                resources = replace(
                    resources, available=False, reason="window_too_small"
                )
            else:
                now = time.time()
                remaining = max(float(deadline) - now, 0.0)
                # The previous deadline already excluded guard/cleanup. Add
                # those back only to the resource budget representation; the
                # new planner subtracts them again, never extending the bound.
                reserved_tail = self._effective_idle_deadline_guard_sec(
                    config
                ) + self._effective_idle_tail_reserve_sec(config)
                resources = replace(
                    resources,
                    minimum_idle_window_sec=min(
                        resources.minimum_idle_window_sec,
                        remaining + reserved_tail,
                    ),
                    idle_window_measured_at=now,
                )
                if remaining <= 0:
                    resources = replace(
                        resources, available=False, reason="window_too_small"
                    )
        if not resources.available:
            if self._idle_worker_writer_migration_blocked:
                resources = AvailableTrainingResources(
                    available=False,
                    reason="writer_state_migration_required",
                )
            if (
                not self._idle_worker_writer_migration_blocked
                and self._disabled_replica_local_idle_groups
                and not self._metadata_idle_training_groups
            ):
                resources = AvailableTrainingResources(
                    available=False,
                    reason="replica_local_unavailable",
                )
                print(
                    "[BubbleTime] skip idle launch: reason=replica_local_unavailable "
                    f"disabled_groups={tuple(sorted(self._disabled_replica_local_idle_groups))} "
                    f"detail={self._replica_local_idle_unavailable_reason}",
                    flush=True,
                )
            return self._skip_idle_worker_plan(context, config, resources)
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step,
            config=config,
            worker_ids=resources.worker_ids,
            target_version=(
                self._training_quota_data_version
                if self._training_quota_debt_steps > 0
                else None
            ),
        )
        self._register_training_quota_cycle(
            data_status=data_status,
            global_step=context.global_step,
            config=config,
        )
        logger.info(
            "[BubbleTime] idle_data_status step=%s group=%s workers=%s "
            "trainable_batches=%s trainable_samples=%s buffer_samples=%s "
            "data_version=%s target_version=%s require_full_batch=%s "
            "sample_last_n_steps=%s",
            context.global_step,
            resources.training_group_id,
            resources.worker_ids,
            data_status.trainable_batches,
            data_status.trainable_samples,
            data_status.buffer_samples,
            data_status.data_version,
            data_status.target_version,
            config.require_full_batch,
            config.sample_last_n_steps,
        )
        plan = self.plan_training(
            DrafterScheduleContext(
                global_step=context.global_step,
                training_mode=context.training_mode,
                collected_samples_this_step=context.collected_samples_this_step,
                oldlogprob_collection_requested=context.oldlogprob_collection_requested,
                data_status=data_status,
                pending_training_count=context.pending_training_count,
            ),
            config,
            resources=resources,
            require_interval=False,
            idle_continuation=continuation_plan is not None,
        )
        if plan.launch and self._idle_worker_writer_state_version is None:
            # Idle observations before collection must not pin all future
            # Bubble work. Elect the writer only for the first plan that can
            # actually consume a training batch.
            chosen_group = _normalize_worker_id_group(plan.target_worker_ids)
            if chosen_group and chosen_group != self._idle_worker_writer_group:
                previous_writer = self._idle_worker_writer_group
                self._idle_worker_writer_group = chosen_group
                logger.warning(
                    "[BubbleTime] idle_writer_elected: from=%s to=%s "
                    "plan_id=%s step=%s reason=first_trainable_idle_window",
                    previous_writer,
                    chosen_group,
                    plan.plan_id,
                    plan.source_global_step,
                )
        return plan

    @staticmethod
    def _training_quota_target_steps(config: DrafterScheduleConfig) -> int:
        configured = config.training_quota_target_steps
        if configured is None:
            configured = config.train_batches_per_trigger
        return max(int(configured), 0)

    def _quota_settled_without_new_collection(
        self, context: DrafterScheduleContext, config: DrafterScheduleConfig
    ) -> bool:
        """Skip empty polling only after a known quota has fully published.

        Durable collection immediately registers the next quota. Preserve the
        legacy boundary-discovery path before that first registration, and for
        callers supplying fresh collection counts or an explicit data status.
        """
        return bool(
            config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and config.training_quota_enable
            and self._training_quota_last_registered_data_version is not None
            and self._training_quota_collection_step is None
            and self._training_quota_debt_steps <= 0
            and context.collected_samples_this_step <= 0
            and context.data_status is None
        )

    def training_quota_blocks_collection(
        self,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        """Whether a prior Bubble collection still owns the training quota.

        Multiple request-local commits from the same rollout step remain
        valid. Only later rollout steps are back-pressured.
        """

        if (
            not config.training_quota_enable
            or config.execution_strategy
            is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or self._training_quota_target_steps(config) <= 0
        ):
            return False
        collection_step = self._training_quota_collection_step
        if collection_step is None:
            return self._training_quota_debt_steps > 0
        # Once training has repaid the quota, keep every collection source
        # blocked until the corresponding publication is acknowledged.
        # Same-step request-local commits are valid only while debt remains.
        if self._training_quota_debt_steps <= 0:
            return True
        try:
            current_step = _as_int(global_step)
        except (TypeError, ValueError):
            return True
        return current_step != collection_step

    def record_collection_outcome(
        self,
        plan: CollectionPlan,
        outcome,
        config: DrafterScheduleConfig,
    ) -> None:
        """Claim one single-flight Bubble quota after a durable collection."""

        if (
            not outcome.collected
            or not config.training_quota_enable
            or config.execution_strategy
            is not DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or self._training_quota_target_steps(config) <= 0
        ):
            return
        collection_step = _as_int(plan.source_global_step)
        if self._training_quota_collection_step is None:
            self._training_quota_collection_step = collection_step
        elif self._training_quota_collection_step != collection_step:
            raise RuntimeError(
                "A new drafter collection completed while the previous "
                "training quota was still active: "
                f"active_step={self._training_quota_collection_step} "
                f"new_step={collection_step}"
            )
        versions = {
            int(result.data_version)
            for result in (getattr(outcome, "worker_results", None) or [])
            if result.data_version is not None
        }
        if len(versions) > 1:
            raise RuntimeError(
                "Drafter collection committed inconsistent data versions: "
                f"versions={tuple(sorted(versions))}"
            )
        data_version = next(iter(versions), collection_step)
        self._register_training_quota_version(
            data_version=data_version,
            global_step=plan.source_global_step,
            config=config,
            advance_completed_cycle=True,
        )

    def _register_training_quota_cycle(
        self,
        *,
        data_status,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> None:
        """Add one bounded quality quota per configured training interval."""

        if not config.training_quota_enable or data_status is None:
            return
        if (
            data_status.trainable_batches < max(config.min_trainable_batches, 1)
            or not data_status.data_version_consistent
            or (not config.use_logits and not data_status.target_version_consistent)
            or data_status.data_version is None
        ):
            return
        self._register_training_quota_version(
            data_version=int(data_status.data_version),
            global_step=global_step,
            config=config,
        )

    def _register_training_quota_version(
        self,
        *,
        data_version: int,
        global_step: object,
        config: DrafterScheduleConfig,
        advance_completed_cycle: bool = False,
    ) -> None:
        # Boundary inspections may rediscover a retained replay buffer. They
        # must not turn that same snapshot into another optimizer-step quota.
        previous_version = self._training_quota_last_registered_data_version
        if previous_version is not None and data_version <= previous_version:
            return
        try:
            interval = int(cast(Any, config.training_interval_steps))
        except (TypeError, ValueError):
            interval = 0
        current_step = _as_int(global_step)
        # Use the upcoming synchronous trigger as the quota cycle.  This lets
        # Bubble work performed anywhere inside the interval repay the same
        # quality target, without creating fresh debt for every collection.
        cycle_step = (
            ((max(current_step, 1) - 1) // interval + 1) * interval
            if interval > 0
            else data_version
        )
        if (
            self._training_quota_last_cycle_step is not None
            and cycle_step <= self._training_quota_last_cycle_step
        ):
            if (
                advance_completed_cycle
                and self._training_quota_debt_steps <= 0
                and interval > 0
            ):
                cycle_step = self._training_quota_last_cycle_step + interval
            else:
                return
        target_steps = self._training_quota_target_steps(config)
        if target_steps <= 0:
            self._training_quota_last_cycle_step = cycle_step
            self._training_quota_collection_step = None
            self._writer_boundary_migration_allowed = True
            return
        previous_debt = self._training_quota_debt_steps
        if previous_debt > 0:
            # Do not charge a new cycle against the old cycle's pinned replay
            # data. Once this cycle completes, the newest collected version
            # becomes the next cycle and receives its own target quota.
            return
        debt_trigger = max(int(config.training_quota_max_accumulated_debt), 0)
        # ``max_accumulated_debt`` is an urgency threshold, not permission to
        # discard optimizer-step obligations.  The due check uses it to force
        # a top-up, while the ledger always retains the full accumulated debt.
        self._training_quota_debt_steps = previous_debt + target_steps
        self._training_quota_last_registered_data_version = data_version
        self._quality_loss_increase_ratio = float(
            config.training_quota_loss_increase_ratio
        )
        self._training_quota_last_cycle_step = cycle_step
        if previous_debt <= 0 and self._training_quota_debt_steps > 0:
            # The cycle deadline, rather than its first data-arrival step, is
            # the start of debt aging. This leaves the whole interval available
            # for opportunistic Bubble work before serial top-up is considered.
            self._training_quota_oldest_cycle_step = cycle_step
            self._training_quota_data_version = data_version
        print(
            "[BubbleTime] training_quota_registered: "
            f"step={global_step} quota_cycle_step={cycle_step} "
            f"data_version={data_version} "
            f"target_steps={target_steps} debt_before={previous_debt} "
            f"debt_after={self._training_quota_debt_steps} "
            f"max_accumulated_debt={debt_trigger}",
            flush=True,
        )

    def protected_target_lm_head_versions(self) -> tuple[int, ...]:
        """Keep the quota's supervision head until training and publish ACK."""
        version = self._training_quota_data_version
        if version is None and self._training_quota_collection_step is not None:
            # Debt reaches zero before the staged weights are applied. Keep
            # protecting the completed cycle until publish releases backpressure.
            version = self._training_quota_last_registered_data_version
        return () if version is None else (int(version),)

    def _training_quota_age_steps(self, global_step: object) -> int:
        if self._training_quota_oldest_cycle_step is None:
            return 0
        return max(
            _as_int(global_step) - self._training_quota_oldest_cycle_step,
            0,
        )

    def _training_quota_due(
        self,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        if not config.training_quota_enable or self._training_quota_debt_steps <= 0:
            return False
        interval_boundary_reached = (
            self._training_quota_oldest_cycle_step is None
            or _as_int(global_step) >= self._training_quota_oldest_cycle_step
        )
        age_due = interval_boundary_reached and self._training_quota_age_steps(
            global_step
        ) >= max(int(config.training_quota_max_debt_age_steps), 0)
        debt_limit = max(int(config.training_quota_max_accumulated_debt), 0)
        debt_due = (
            debt_limit > 0
            and self._training_quota_debt_steps > debt_limit
            and interval_boundary_reached
            and self._training_quota_age_steps(global_step) > 0
        )
        return age_due or debt_due

    def _training_quota_force_completion_due(
        self,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        """Whether drafter freshness now outweighs further Bubble waiting."""

        if not config.training_quota_enable or self._training_quota_debt_steps <= 0:
            return False
        if self._training_quota_oldest_cycle_step is None:
            return True
        return _as_int(
            global_step
        ) >= self._training_quota_oldest_cycle_step and self._training_quota_age_steps(
            global_step
        ) >= max(int(config.training_quota_max_completion_lag_steps), 0)

    def _plan_training_quota_topup(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        force_completion: bool = False,
    ) -> TrainingPlan | None:
        """Plan a blocking top-up without mixing optimizer ownership.

        If Bubble has already advanced the cycle, only its writer group may
        finish that optimizer state.  If it made no progress at all, avoid
        fragmented writer-only top-ups and use the normal full collective once
        the hard freshness deadline expires.
        """

        if (
            not config.training_quota_enable
            or context.pending_training_count > 0
            or not config.training_quota_allow_critical_path_fallback
            or self._quota_settled_without_new_collection(context, config)
        ):
            return None
        writer_group = self._current_idle_writer_group(assign_default=False)
        if self._training_quota_debt_steps <= 0:
            # Preserve the boundary-inspection fallback for callers that did
            # not observe a collection outcome, but never elect a writer just
            # to perform that probe. With no existing owner, inspect all
            # workers and leave the first useful rollout bubble free to choose.
            boundary_data_status = self.inspect_training_data(
                global_step=context.global_step,
                config=config,
                worker_ids=writer_group,
                target_version=self._training_quota_data_version,
            )
            self._register_training_quota_cycle(
                data_status=boundary_data_status,
                global_step=context.global_step,
                config=config,
            )
            if self._training_quota_debt_steps <= 0:
                return None
        bootstrap_due = self.needs_idle_optimizer_bootstrap(config)
        force_completion = (
            force_completion
            or bootstrap_due
            or self._training_quota_force_completion_due(context.global_step, config)
        )
        if not force_completion and not self._training_quota_due(
            context.global_step, config
        ):
            return None
        target_steps = self._training_quota_target_steps(config)
        completed_bubble_steps = max(
            target_steps - int(self._training_quota_debt_steps),
            0,
        )
        sync_workers = self._all_idle_training_worker_ids()
        replica_local_topology = self._runtime_replica_local_topology or any(
            len(group) < len(sync_workers)
            for group in self._metadata_idle_training_groups
        )
        full_collective_fallback = (
            force_completion
            and completed_bubble_steps == 0
            and self._idle_worker_writer_state_version is None
            and not replica_local_topology
        )
        # Soft top-ups before any genuine Bubble progress only add critical-path
        # activation/snapshot overhead and leave a split 2+2+16 schedule. Keep
        # the full quota available for rollout idle windows until the hard
        # freshness bound, then match Sync with one full-collective call.
        if completed_bubble_steps == 0 and not force_completion:
            return None

        if full_collective_fallback:
            bootstrap_group = (
                self._current_idle_writer_group(assign_default=False)
                if self._idle_worker_writer_state_version is not None
                else self._best_idle_bootstrap_group()
            )
            if bootstrap_group:
                writer_group = bootstrap_group
                print(
                    "[BubbleTime] full_collective_hot_bootstrap_selected: "
                    f"group={bootstrap_group} "
                    "replica_ranks="
                    f"{self._replica_ranks_for_worker_group(bootstrap_group)} "
                    "historical_window_s="
                    f"{self._effective_historical_idle_window_sec(worker_ids=bootstrap_group)} "
                    "selection_reason="
                    f"{'global_publish_leader' if self._global_publish_leader_worker_id in bootstrap_group else 'idle_capacity'}",
                    flush=True,
                )

        # Do not elect a writer while merely probing whether a top-up is due.
        # Before the first real Bubble optimizer step, assigning the first hot
        # group here pins target-head prefetch and all later idle admission to
        # metadata order rather than the replica that actually becomes idle.
        writer_group = writer_group or self._best_idle_bootstrap_group()
        if not writer_group or self._idle_worker_writer_migration_blocked:
            return None
        if (
            not force_completion
            and config.idle_worker_keep_hot_between_plans
            and writer_group in self._idle_worker_hot_prewarmed_groups
        ):
            historical_window = self._effective_historical_idle_window_sec(
                worker_ids=writer_group
            )
            if (
                historical_window is not None
                and historical_window
                >= self._minimum_idle_training_window_sec(
                    config,
                    worker_ids=writer_group,
                )
            ):
                # Keep the exact quota/snapshot available to later idle leases.
                # The existing hard freshness deadline still forces completion
                # even if future windows do not materialize.
                logger.warning(
                    "[BubbleTime] quota_soft_topup_deferred: step=%s writer=%s debt=%s "
                    "historical_window_s=%.3f reason=hot_writer_idle_before_hard_deadline",
                    context.global_step,
                    writer_group,
                    self._training_quota_debt_steps,
                    historical_window,
                )
                return None
        writer_data_status = self.inspect_training_data(
            global_step=context.global_step,
            config=config,
            worker_ids=writer_group,
            target_version=self._training_quota_data_version,
        )

        data_status = (
            self.inspect_training_data(
                global_step=context.global_step,
                config=config,
                worker_ids=None,
                target_version=self._training_quota_data_version,
            )
            if full_collective_fallback
            else writer_data_status
        )
        # Normal blocking assistance leaves the final optimizer step to a real
        # Bubble window. Once the freshness deadline expires, exact completion
        # and publication take precedence over further critical-path hiding.
        topup_steps = (
            int(self._training_quota_debt_steps)
            if force_completion
            else min(
                max(self._training_quota_debt_steps - 1, 0),
                int(config.training_quota_max_sync_topup_steps),
            )
        )
        if topup_steps <= 0:
            return None
        topup_context = replace(context, data_status=data_status)
        plan = self.plan_training(
            topup_context,
            replace(
                config,
                execution_strategy=DrafterExecutionStrategy.SYNC,
                training_interval_steps=1,
                train_batches_per_trigger=topup_steps,
                gradient_accumulation_steps=(
                    config.gradient_accumulation_steps
                    if full_collective_fallback
                    else self._idle_gradient_accumulation_steps(config, writer_group)
                ),
                min_trainable_batches=min(
                    max(int(config.min_trainable_batches), 1), topup_steps
                ),
            ),
            require_interval=False,
        )
        if full_collective_fallback and not plan.launch:
            # Single-writer collection may intentionally leave the other
            # replica owners without this replay version.  Never strand the
            # quota in that case: retain the safe writer-local completion
            # path.  When all ranks do have the version, the faster normal
            # full collective remains selected above.
            print(
                "[BubbleTime] training_quota_full_collective_unavailable: "
                f"step={context.global_step} reason={plan.reason} "
                f"fallback_workers={writer_group}",
                flush=True,
            )
            full_collective_fallback = False
            data_status = writer_data_status
            topup_context = replace(context, data_status=data_status)
            plan = self.plan_training(
                topup_context,
                replace(
                    config,
                    execution_strategy=DrafterExecutionStrategy.SYNC,
                    training_interval_steps=1,
                    train_batches_per_trigger=topup_steps,
                    min_trainable_batches=min(
                        max(int(config.min_trainable_batches), 1), topup_steps
                    ),
                ),
                require_interval=False,
            )
        if not plan.launch:
            if (
                config.execution_strategy
                is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
                and self._training_quota_force_completion_due(
                    context.global_step, config
                )
            ):
                raise RuntimeError(
                    "[BubbleTime] quota_completion_blocked: "
                    f"step={context.global_step} "
                    f"required_target_version={self._training_quota_data_version} "
                    f"debt_steps={self._training_quota_debt_steps} "
                    f"debt_age_steps={self._training_quota_age_steps(context.global_step)} "
                    f"writer_group={writer_group} reason={plan.reason} "
                    f"buffer_samples={getattr(data_status, 'buffer_samples', None)} "
                    f"trainable_samples={getattr(data_status, 'trainable_samples', None)} "
                    f"observed_target_version={getattr(data_status, 'target_version', None)} "
                    f"worker_snapshots={getattr(data_status, 'worker_snapshots', None)}. "
                    "No matching data/head can complete the overdue quota. "
                    "Restore the original data/head snapshot or restart from a valid checkpoint; "
                    "do not substitute the current actor head or discard the training debt."
                )
            return None
        # The temporary Sync config above only selects a blocking budget.
        # Top-up still belongs to Bubble's original quota and frozen head.
        plan = replace(
            plan,
            required_target_version=(
                None if config.use_logits else self._training_quota_data_version
            ),
        )
        if bootstrap_due:
            logger.warning(
                "[BubbleTime] quota_optimizer_bootstrap_planned: step=%s writer=%s quota_steps=%s historical_window_s=%s",
                context.global_step,
                writer_group,
                topup_steps,
                self._effective_historical_idle_window_sec(worker_ids=writer_group),
            )
        print(
            "[BubbleTime] training_quota_topup_planned: "
            f"step={context.global_step} workers="
            f"{'full_collective' if full_collective_fallback else writer_group} "
            f"debt_steps={self._training_quota_debt_steps} "
            f"debt_age_steps={self._training_quota_age_steps(context.global_step)} "
            f"topup_steps={topup_steps} force_completion={force_completion} "
            f"completed_bubble_steps={completed_bubble_steps} "
            f"full_collective_fallback={full_collective_fallback} "
            f"data_version={plan.data_version}",
            flush=True,
        )
        if full_collective_fallback:
            hot_worker_ids = (
                tuple(
                    sorted(
                        (plan.worker_snapshots or {}).keys(),
                        key=_natural_worker_sort_key,
                    )
                )
                or self._all_idle_training_worker_ids()
            )
            return replace(
                plan,
                reason="quota_forced_completion_ready",
                execution_strategy=DrafterExecutionStrategy.SYNC,
                target_worker_ids=(),
                training_group_id="quota-full-collective",
                deadline_ts=None,
                publish_after_success=True,
                retain_replay_session=False,
                hot_bootstrap_worker_ids=hot_worker_ids,
            )
        # Preserve the rollout-idle worker payload so only the writer group
        # participates, but execute this specially marked plan synchronously.
        return replace(
            plan,
            reason=(
                "quota_forced_completion_ready"
                if force_completion
                else "quota_topup_training_ready"
            ),
            execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
            target_worker_ids=writer_group,
            training_group_id="quota-topup",
            gradient_accumulation_steps=self._idle_gradient_accumulation_steps(
                config, writer_group
            ),
            deadline_ts=None,
            publish_after_success=force_completion,
            retain_replay_session=not force_completion,
        )

    def needs_idle_optimizer_bootstrap(self, config: DrafterScheduleConfig) -> bool:
        """Model prewarm alone must not delay the first real optimizer quota."""
        return bool(
            self._training_quota_debt_steps > 0
            and self._training_quota_debt_steps
            >= self._training_quota_target_steps(config)
            and self._training_quota_data_version is not None
            and self._idle_worker_writer_state_version is None
            and not self._idle_optimizer_initialized_groups
        )

    def record_hot_target_head_prepared(
        self, worker_ids: tuple[str, ...], target_version: int, elapsed_sec: float
    ) -> None:
        group = _normalize_worker_id_group(worker_ids)
        self._idle_worker_last_target_version[group] = int(target_version)
        if elapsed_sec > 0:
            self._idle_worker_hot_head_prepare_samples.setdefault(
                group, deque(maxlen=32)
            ).append(float(elapsed_sec))

    def _skip_idle_worker_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        resources: AvailableTrainingResources,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        reclaim_penalty_sec = self._effective_idle_reclaim_penalty_sec()
        base_usable_window = max(
            resources.minimum_idle_window_sec
            - self._effective_idle_deadline_guard_sec(config)
            - self._effective_idle_startup_reserve_sec(
                config,
                resources.worker_ids,
            )
            - self._effective_idle_tail_reserve_sec(config),
            0.0,
        )
        usable_window = max(
            base_usable_window - reclaim_penalty_sec,
            0.0,
        )
        gradient_accumulation_steps = self._idle_gradient_accumulation_steps(
            config, resources.worker_ids
        )
        batch_estimate = (
            self._effective_idle_batch_estimate_sec(config)
            * gradient_accumulation_steps
        )
        return TrainingPlan(
            launch=False,
            reason=resources.reason,
            interval_matched=interval_matched,
            execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
            source_global_step=context.global_step,
            max_batches=0,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            publish_after_success=False,
            required_target_version=(
                None
                if config.use_logits
                else (
                    context.data_status.target_version
                    if context.data_status is not None
                    else _as_int(context.global_step)
                )
            ),
            plan_id=uuid4().hex,
            target_worker_ids=resources.worker_ids,
            training_group_id=resources.training_group_id,
            idle_window_sec=resources.minimum_idle_window_sec,
            idle_usable_window_sec=usable_window,
            idle_window_batches=int(math.floor(usable_window / batch_estimate)),
            idle_batch_estimate_sec=batch_estimate,
            idle_startup_reserve_sec=self._effective_idle_startup_reserve_sec(
                config,
                resources.worker_ids,
            ),
            idle_tail_reserve_sec=self._effective_idle_tail_reserve_sec(config),
            idle_reclaim_penalty_sec=reclaim_penalty_sec,
            idle_confidence=getattr(
                resources, "idle_confidence", IdleWindowConfidence.CONFIRMED
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def prepare_training_execution(self, plan: TrainingPlan) -> dict[str, Any]:
        if not plan.launch:
            return {"drafter/target_lm_head_synced": 0}
        if (
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or plan.reason
            in {"quota_topup_training_ready", "quota_forced_completion_ready"}
        ):
            # The exact head for buffered step N data is cached before the step
            # N actor update.  Fetching the live actor head inside step N+1
            # rollout would both block the bubble and select the wrong version.
            # This also applies to a Bubble full-collective top-up whose
            # execution strategy is Sync, not just replica-local idle plans.
            return {"drafter/target_lm_head_synced": 0}
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        return self._worker_executor.prepare_training(plan)

    def activate_training_workers(self) -> list[Any]:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        results = self._worker_executor.activate_training_workers()
        active = [
            result
            for result in results
            if isinstance(result, dict)
            and result.get("reason") not in {"disabled", "not_in_training_group"}
        ]
        failures = [result for result in active if not result.get("activated", False)]
        if failures:
            raise RuntimeError(
                f"SPECO drafter trainer activation failed: {failures[:3]}"
            )
        for result in active:
            group = _normalize_worker_id_group(result.get("training_group_ranks", ()))
            restore = float(result.get("restore_elapsed_sec", 0.0) or 0.0)
            if group and restore > 0.0:
                self._idle_worker_restore_samples.setdefault(
                    group, deque(maxlen=32)
                ).append(restore)
            park = float(result.get("park_elapsed_sec", 0.0) or 0.0)
            self.record_memory_park_elapsed(park)
            snapshot = float(result.get("snapshot_elapsed_sec", 0.0) or 0.0)
            if snapshot > 0.0:
                self._idle_worker_final_tail_samples_sec.append(snapshot)
            if group:
                logger.warning(
                    "[BubbleTime] CPU restore calibrated: group=%s restore_s=%.4f park_s=%.4f snapshot_s=%.4f",
                    group,
                    restore,
                    park,
                    snapshot,
                )
        return results

    def wait_pending_publish(self) -> int:
        if self._publish_executor is None:
            return 0
        return self._publish_executor.wait_pending()

    @staticmethod
    def should_collect(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.collect_interval_steps)

    def _next_collection_id(
        self, source: DrafterCollectionSource, global_step: object
    ) -> str:
        try:
            source_global_step = int(str(global_step))
        except (TypeError, ValueError):
            source_global_step = -1
        self._collection_sequence += 1
        return (
            f"collection-{source_global_step:020d}-"
            f"{self._collection_sequence:020d}-{source.value}"
        )

    def plan_collection(
        self,
        context: DrafterCollectionContext,
        config: DrafterScheduleConfig,
    ) -> CollectionPlan:
        collect_interval_matched = self.should_collect(context.global_step, config)
        training_interval_matched = self.training_interval_matched(
            context.global_step, config
        )
        common: Any = {
            "collection_id": self._next_collection_id(
                context.source, context.global_step
            ),
            "source": context.source,
            "source_global_step": context.global_step,
            "collect_interval_matched": collect_interval_matched,
            "training_interval_matched": training_interval_matched,
            "sample_rate": config.collection_sample_rate,
            "max_samples_per_replica": config.max_collect_samples_per_replica,
            "max_tokens_per_replica": config.max_collect_tokens_per_replica,
            "hidden_window_mode": config.hidden_window_mode,
            "hidden_window_tokens_per_sample": config.hidden_window_tokens_per_sample,
            "hidden_window_min_rows": config.hidden_window_min_rows,
        }
        if not context.drafter_enabled:
            return CollectionPlan(collect=False, reason="drafter_disabled", **common)
        if not context.source_enabled:
            return CollectionPlan(collect=False, reason="source_disabled", **common)
        if context.drafter_frozen:
            return CollectionPlan(collect=False, reason="drafter_frozen", **common)
        if context.validation:
            return CollectionPlan(collect=False, reason="validation", **common)
        if not collect_interval_matched:
            return CollectionPlan(
                collect=False, reason="interval_not_reached", **common
            )
        if context.require_training_interval and not training_interval_matched:
            return CollectionPlan(
                collect=False,
                reason="training_interval_not_reached",
                **common,
            )
        if self.training_quota_blocks_collection(context.global_step, config):
            return CollectionPlan(
                collect=False,
                reason="training_quota_incomplete",
                **common,
            )
        if not self._adaptive_quality_refresh_due(context.global_step, config):
            return CollectionPlan(
                collect=False,
                reason="quality_refresh_not_due",
                **common,
            )
        if config.collection_sample_rate <= 0:
            return CollectionPlan(collect=False, reason="sample_rate_zero", **common)
        return CollectionPlan(collect=True, reason="collection_enabled", **common)

    def execute_collection_plan(self, plan: CollectionPlan, payload: CollectionPayload):
        if self._collection_executor is None:
            raise RuntimeError("Drafter collection executor has not been bound")
        return self.collection_strategy.execute(
            plan,
            payload,
            executor=self._collection_executor,
        )

    def on_collection_ready(self, plan: CollectionPlan, payload: CollectionPayload):
        """Lifecycle Facade event for a source adapter's prepared payload."""
        return self.execute_collection_plan(plan, payload)

    def on_before_actor_update(
        self, context: BeforeActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Plan and prepare drafter training before the PPO actor update."""
        plan = self.prepare_training_plan(
            context.schedule_context,
            context.config,
            **(
                {"idle_continuation_plan": context.idle_continuation_plan}
                if context.idle_continuation_plan is not None
                else {}
            ),
        )
        force_completion_due = False
        if (
            context.allow_quota_topup
            and not context.drafter_frozen
            and context.config.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        ):
            force_completion_due = (
                context.force_quota_completion
                or self._training_quota_force_completion_due(
                    context.schedule_context.global_step,
                    context.config,
                )
            )
            # Before the hard freshness bound, a real rollout-idle launch wins
            # and serial top-up remains bounded. At the hard bound, force
            # completion even when another deadline-limited Bubble plan could
            # launch; otherwise repeated partial plans can postpone publication.
            if force_completion_due or not plan.launch:
                quota_plan = self._plan_training_quota_topup(
                    context.schedule_context,
                    context.config,
                    **(
                        {"force_completion": True}
                        if context.force_quota_completion
                        else {}
                    ),
                )
                if quota_plan is not None:
                    plan = quota_plan
        # Freeze gate: force launch=False BEFORE prepare_training_execution so
        # a frozen step never runs the target lm-head sync / set-global-step
        # prepare RPCs that exist only to feed drafter training. The post-actor
        # frozen replacement in the trainer remains as defense in depth.
        if getattr(context, "drafter_frozen", False) and plan.launch:
            plan = replace(plan, launch=False, reason="drafter_convergence_frozen")
        metrics: dict[str, Any] = dict(plan.metrics())
        metrics.update(
            {
                "scheduler/train_requested": int(plan.launch),
                "scheduler/planned_batches": int(plan.max_batches),
                "bubble/training_quota_debt_steps": int(
                    self._training_quota_debt_steps
                ),
                "bubble/training_quota_debt_age_steps": int(
                    self._training_quota_age_steps(context.schedule_context.global_step)
                ),
                "bubble/training_quota_completion_lag_steps": int(
                    self._training_quota_age_steps(context.schedule_context.global_step)
                ),
                "bubble/training_quota_data_version": int(
                    self._training_quota_data_version
                    if self._training_quota_data_version is not None
                    else -1
                ),
                "bubble/training_quota_collection_step": int(
                    self._training_quota_collection_step
                    if self._training_quota_collection_step is not None
                    else -1
                ),
                "bubble/training_quota_topup_requested": int(
                    plan.reason
                    in {
                        "quota_topup_training_ready",
                        "quota_forced_completion_ready",
                    }
                ),
                "bubble/training_quota_topup_batches": (
                    int(plan.max_batches)
                    if plan.reason
                    in {
                        "quota_topup_training_ready",
                        "quota_forced_completion_ready",
                    }
                    else 0
                ),
                "bubble/training_quota_force_complete": int(
                    plan.reason == "quota_forced_completion_ready"
                ),
                "bubble/training_quota_force_complete_steps": (
                    int(plan.max_batches)
                    if plan.reason == "quota_forced_completion_ready"
                    else 0
                ),
                "bubble/training_quota_critical_path_fallback_blocked": int(
                    force_completion_due
                    and not context.config.training_quota_allow_critical_path_fallback
                    and not plan.launch
                    and self._training_quota_debt_steps > 0
                ),
                "bubble/publish_waiting_for_quota": int(
                    self._training_quota_debt_steps > 0
                ),
            }
        )
        metrics.update(self.prepare_training_execution(plan))
        return SchedulerEventOutcome(
            training_plan=plan,
            metrics=metrics,
        )

    def on_after_actor_update(
        self, context: AfterActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Execute the prepared plan after the PPO actor update."""
        plan = context.training_plan
        if not plan.launch:
            return SchedulerEventOutcome(training_plan=plan, metrics={})
        execution = self.execute_training_plan(
            plan,
            runtime_state=context.runtime_state,
        )
        if execution.reason == "submitted_async":
            metrics: dict[str, float | int] = {
                "scheduler/train_launched": 1,
                "scheduler/pending_training_count": 1,
            }
            metrics.update(context.runtime_state.metrics())
            return SchedulerEventOutcome(
                training_plan=plan,
                metrics=metrics,
            )
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=context.runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return SchedulerEventOutcome(
            training_plan=plan,
            training_execution=outcome,
            metrics=outcome.metrics,
        )

    def on_safe_point(self, context: AfterWeightUpdateContext) -> SchedulerEventOutcome:
        """Plan and execute publication at a rollout-safe lifecycle point."""
        plan = self.plan_publish(
            global_step=context.global_step,
            drafter_trained=context.drafter_trained,
            config=context.config,
            training_plan=context.training_plan,
        )
        outcome = self.execute_publish_plan(plan)
        return SchedulerEventOutcome(
            training_plan=context.training_plan,
            publish_plan=plan,
            publish_outcome=outcome,
            metrics=outcome.metrics(),
        )

    def on_after_weight_update(
        self, context: AfterWeightUpdateContext
    ) -> SchedulerEventOutcome:
        """Named lifecycle alias for the post-weight-update safe point."""
        return self.on_safe_point(context)

    @staticmethod
    def training_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.training_interval_steps)

    def plan_training(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        resources: AvailableTrainingResources | None = None,
        *,
        require_interval: bool = True,
        idle_continuation: bool = False,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        execution_strategy = (
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            if resources is not None
            else DrafterExecutionStrategy.SYNC
        )
        trigger = self.trigger_policy.should_train(
            replace(
                context,
                allow_cached_training=(
                    context.allow_cached_training
                    or resources is not None
                    or (not require_interval and config.training_quota_enable)
                ),
            ),
            config,
            interval_matched=interval_matched if require_interval else True,
        )
        budget = self.sync_budget_policy.make_budget(context, config)
        gradient_accumulation_steps = 1
        planned_optimizer_steps = budget.max_batches
        idle_tail_reserve_sec = (
            self._effective_idle_tail_reserve_sec(
                config, worker_ids=resources.worker_ids
            )
            if resources is not None
            else None
        )
        planned_valid_tokens = (
            context.data_status.trainable_valid_tokens
            if context.data_status is not None
            else 0
        )
        idle_publish_deferred = False
        publish_snapshot_deferred = False
        final_tail_blocked = False
        if resources is not None and budget.max_batches > 0:
            budget_started_ts = time.time()
            measured_at = getattr(resources, "idle_window_measured_at", None)
            if measured_at is not None:
                # Inspection RPCs spend the selected window too. Never restart
                # its clock after data readiness has been checked.
                resources = replace(
                    resources,
                    minimum_idle_window_sec=max(
                        resources.minimum_idle_window_sec
                        - max(budget_started_ts - measured_at, 0.0),
                        0.0,
                    ),
                )
            sync_budget_batches = budget.max_batches
            gradient_accumulation_steps = self._idle_gradient_accumulation_steps(
                config, resources.worker_ids
            )
            deadline_guard_sec = self._effective_idle_deadline_guard_sec(config)
            startup_reserve_sec = self._effective_idle_startup_reserve_sec(
                config,
                resources.worker_ids,
                continuation=idle_continuation,
            )
            tail_reserve_sec = float(idle_tail_reserve_sec or 0.0)
            reclaim_penalty_sec = self._effective_idle_reclaim_penalty_sec()
            micro_batch_estimate = self._effective_idle_batch_estimate_sec(
                config, resources.worker_ids
            )
            optimizer_step_estimate = micro_batch_estimate * gradient_accumulation_steps
            hard_train_batch_cap = max(int(config.train_batches_per_trigger), 1)
            train_batch_cap = self._effective_idle_dynamic_batch_cap(
                config, resources.worker_ids
            )
            base_usable_window = max(
                resources.minimum_idle_window_sec
                - deadline_guard_sec
                - startup_reserve_sec
                - tail_reserve_sec,
                0.0,
            )
            usable_window = max(
                base_usable_window - reclaim_penalty_sec,
                0.0,
            )
            # The worker checks ``deadline_ts`` only after activation and
            # preflight.  Do not subtract setup twice from that boundary:
            # setup constrains admission, while tail + guard constrain the
            # final batch start so cleanup can still finish safely.
            worker_deadline_window = max(
                resources.minimum_idle_window_sec
                - deadline_guard_sec
                - tail_reserve_sec,
                0.0,
            )
            has_hard_runtime_deadline = (
                getattr(resources, "idle_window_source", "runtime_deadline")
                == "runtime_deadline"
            )
            window_batches = int(math.floor(usable_window / optimizer_step_estimate))
            trainable_micro_batches = (
                context.data_status.trainable_batches if context.data_status else 0
            )
            trainable_batches = trainable_micro_batches // gradient_accumulation_steps
            # Bubble owns a version-homogeneous, plan-local reservation.  Once
            # it has one complete optimizer batch, it may replay that snapshot
            # just as Sync re-samples its DataBuffer.  Distinct samples are an
            # admission requirement, not an upper bound on plan length.
            replay_seed_available = trainable_batches > 0

            # Only an authoritative runtime deadline may rely on cooperative
            # reclaim after admission. Historical/bootstrap windows are
            # estimates, so their planned optimizer work must fit completely
            # inside the estimated window.
            def _window_limited_batch_cap(available_window_batches: int) -> int:
                cap = min(train_batch_cap, sync_budget_batches)
                if not has_hard_runtime_deadline:
                    cap = min(cap, max(available_window_batches, 0))
                return cap if available_window_batches > 0 else 0

            max_batches = (
                _window_limited_batch_cap(window_batches)
                if replay_seed_available
                else 0
            )
            if getattr(resources, "idle_window_source", "") == "historical_survival":
                # Reuse this one lease for the work that fits its conditional
                # window, rather than throwing away the remaining budget after
                # one optimizer step. Never renew its absolute deadline or
                # finish the quota (and stage a snapshot) on this estimate.
                # The worker still checks deadline/reclaim before every step.
                max_batches = min(
                    max_batches, max(self._training_quota_debt_steps - 1, 0)
                )
            cold_writer_reason: str | None = None
            cold_writer_min_steps = 2
            cold_writer_first_real_plan = bool(
                replay_seed_available
                and max_batches > 0
                and config.training_quota_enable
                and self._training_quota_debt_steps > 0
                and resources.worker_ids not in self._idle_optimizer_initialized_groups
                and self._idle_worker_writer_state_version is None
                and not self._idle_worker_writer_migration_blocked
            )
            if (
                cold_writer_first_real_plan
                and config.training_quota_allow_critical_path_fallback
            ):
                # An unmeasured first optimizer update can take tens of seconds.
                # Bootstrap through the existing bounded-freshness top-up, not
                # a model-only prewarm falsely treated as a hot optimizer.
                max_batches = 0
                cold_writer_reason = "cold_writer_wait_for_optimizer_bootstrap"
            elif cold_writer_first_real_plan and not has_hard_runtime_deadline:
                max_batches = 0
                cold_writer_reason = "cold_writer_wait_for_runtime_deadline"
            elif cold_writer_first_real_plan and max_batches < cold_writer_min_steps:
                max_batches = 0
                cold_writer_reason = "cold_writer_window_too_small"

            # A completing plan pays snapshot/publish staging cost; an
            # intermediate plan does not. Re-evaluate the capacity with the
            # final-tail history only when this plan can actually close the
            # current quota (or publish in non-quota mode).
            projected_debt = max(
                self._training_quota_debt_steps - int(max_batches),
                0,
            )
            uses_final_tail = bool(
                max_batches > 0
                and (
                    (
                        config.training_quota_enable
                        and self._training_quota_debt_steps > 0
                        and projected_debt == 0
                    )
                    or (
                        not config.training_quota_enable
                        and self._publish_interval_matched(context.global_step, config)
                    )
                )
            )
            if uses_final_tail:
                final_tail_reserve_sec = self._effective_idle_tail_reserve_sec(
                    config,
                    final=True,
                    worker_ids=resources.worker_ids,
                )
                final_base_usable_window = max(
                    resources.minimum_idle_window_sec
                    - deadline_guard_sec
                    - startup_reserve_sec
                    - final_tail_reserve_sec,
                    0.0,
                )
                final_usable_window = max(
                    final_base_usable_window - reclaim_penalty_sec,
                    0.0,
                )
                final_window_batches = int(
                    math.floor(final_usable_window / optimizer_step_estimate)
                )
                final_max_batches = (
                    _window_limited_batch_cap(final_window_batches)
                    if replay_seed_available
                    else 0
                )
                # Completing and exporting must fit the final budget even
                # when a hard deadline permits opportunistic partial work.
                final_max_batches = min(final_max_batches, final_window_batches)
                final_projected_debt = max(
                    self._training_quota_debt_steps - int(final_max_batches),
                    0,
                )
                still_final = bool(
                    (
                        config.training_quota_enable
                        and self._training_quota_debt_steps > 0
                        and final_projected_debt == 0
                    )
                    or (
                        not config.training_quota_enable
                        and final_max_batches > 0
                        and self._publish_interval_matched(context.global_step, config)
                    )
                )
                if still_final:
                    tail_reserve_sec = final_tail_reserve_sec
                    idle_tail_reserve_sec = final_tail_reserve_sec
                    base_usable_window = final_base_usable_window
                    usable_window = final_usable_window
                    window_batches = final_window_batches
                    max_batches = final_max_batches
                    if cold_writer_first_real_plan:
                        if not has_hard_runtime_deadline:
                            max_batches = 0
                            cold_writer_reason = "cold_writer_wait_for_runtime_deadline"
                        elif max_batches < cold_writer_min_steps:
                            max_batches = 0
                            cold_writer_reason = "cold_writer_window_too_small"
                    worker_deadline_window = max(
                        resources.minimum_idle_window_sec
                        - deadline_guard_sec
                        - final_tail_reserve_sec,
                        0.0,
                    )
                else:
                    final_tail_blocked = True
                    idle_publish_deferred = True
                    if (
                        config.training_quota_enable
                        and config.idle_worker_keep_hot_between_plans
                        and window_batches >= self._training_quota_debt_steps
                        and max_batches >= self._training_quota_debt_steps
                    ):
                        # Complete optimizer work in this real idle window.
                        # The canonical writer is frozen until publish ACK,
                        # so its immutable snapshot can be exported later at
                        # the existing safe publish point, not on this lease.
                        max_batches = self._training_quota_debt_steps
                        publish_snapshot_deferred = True
                        idle_publish_deferred = False
                        final_tail_blocked = False
                    elif config.training_quota_enable:
                        # Keep useful partial work, but never cross the quota
                        # completion boundary without budgeting its snapshot.
                        max_batches = min(
                            max_batches, max(self._training_quota_debt_steps - 1, 0)
                        )
            if max_batches > 0:
                idle_budget_reason = "idle_worker_budget_ready"
            elif final_tail_blocked:
                idle_budget_reason = "final_tail_window_too_small"
            elif cold_writer_reason is not None:
                idle_budget_reason = cold_writer_reason
            elif window_batches <= 0:
                idle_budget_reason = "window_too_small"
            elif not replay_seed_available:
                idle_budget_reason = "no_trainable_batch"
            elif train_batch_cap <= 0 or budget.max_batches <= 0:
                idle_budget_reason = "no_training_budget"
            else:
                idle_budget_reason = "no_training_budget"
            budget = TrainingBudget(
                max_batches=max_batches,
                min_batches=budget.min_batches,
                deadline_ts=(
                    budget_started_ts + worker_deadline_window
                    if has_hard_runtime_deadline or max_batches > 0
                    else None
                ),
                require_full_batch=budget.require_full_batch,
                sample_last_n_steps=budget.sample_last_n_steps,
                reason=idle_budget_reason,
            )
            planned_optimizer_steps = max_batches
            if context.data_status is not None and trainable_micro_batches > 0:
                planned_valid_tokens = int(
                    context.data_status.trainable_valid_tokens
                    * max_batches
                    / max(trainable_batches, 1)
                )
            logger.warning(
                "[BubbleTime] idle_budget: step=%s group=%s workers=%s "
                "minimum_window_s=%.3f base_usable_window_s=%.3f "
                "usable_window_s=%.3f reclaim_penalty_s=%.3f guard_s=%.3f "
                "startup_reserve_s=%.3f first_step_extra_s=%.3f tail_reserve_s=%.3f worker_deadline_window_s=%.3f "
                "micro_batch_estimate_s=%.3f optimizer_step_estimate_s=%.3f "
                "admission_window_batches=%s trainable_batches=%s replay_seed_available=%s "
                "trainable_micro_batches=%s gradient_accumulation_steps=%s "
                "dynamic_train_batch_cap=%s hard_train_batch_cap=%s sync_budget_batches=%s "
                "planned_batches=%s window_mode=%s reason=%s window_source=%s "
                "estimate_source=%s prebatch_reclaim_streak=%s "
                "trainable_valid_tokens=%s planned_valid_tokens=%s "
                "calibrated_budgets=%s writer_steady_samples=%s writer_partial_tail_samples=%s",
                context.global_step,
                resources.training_group_id,
                resources.worker_ids,
                resources.minimum_idle_window_sec,
                base_usable_window,
                usable_window,
                reclaim_penalty_sec,
                deadline_guard_sec,
                startup_reserve_sec,
                self._effective_idle_first_step_extra_sec(
                    config, resources.worker_ids, continuation=idle_continuation
                ),
                tail_reserve_sec,
                worker_deadline_window,
                micro_batch_estimate,
                optimizer_step_estimate,
                window_batches,
                trainable_batches,
                replay_seed_available,
                trainable_micro_batches,
                gradient_accumulation_steps,
                train_batch_cap,
                hard_train_batch_cap,
                sync_budget_batches,
                max_batches,
                "admission",
                budget.reason,
                getattr(resources, "idle_window_source", "runtime_deadline"),
                (
                    "bootstrap"
                    if self._idle_batch_estimate_is_bootstrap(config)
                    else (
                        "config"
                        if config.idle_worker_initial_batch_estimate_sec is not None
                        else "history"
                    )
                ),
                self._idle_worker_prebatch_reclaim_streak,
                context.data_status.trainable_valid_tokens
                if context.data_status
                else 0,
                planned_valid_tokens,
                int(config.idle_worker_calibrated_budgets),
                len(self._idle_steady_by_group.get(resources.worker_ids, ())),
                len(
                    self._idle_tail_by_mode.get(
                        (
                            resources.worker_ids,
                            "hot"
                            if config.idle_worker_keep_hot_between_plans
                            else "cold",
                            False,
                        ),
                        (),
                    )
                ),
            )
            limiting_factor = (
                "window"
                if window_batches <= 0
                else ("data" if not replay_seed_available else "config")
            )
            logger.debug(
                "[BubbleTime] idle_budget_limits: step=%s group=%s "
                "window_optimizer_steps=%s data_optimizer_steps=%s "
                "config_optimizer_steps=%s dynamic_cap_optimizer_steps=%s "
                "planned_optimizer_steps=%s planned_valid_tokens=%s limiting_factor=%s",
                context.global_step,
                resources.training_group_id,
                window_batches,
                trainable_batches,
                min(hard_train_batch_cap, sync_budget_batches),
                train_batch_cap,
                planned_optimizer_steps,
                planned_valid_tokens,
                limiting_factor,
            )
        if (
            resources is not None
            and config.training_quota_enable
            and self._training_quota_debt_steps > 0
            and budget.max_batches > self._training_quota_debt_steps
        ):
            budget = replace(
                budget,
                max_batches=int(self._training_quota_debt_steps),
            )
            planned_optimizer_steps = budget.max_batches
        common: Any = {
            "interval_matched": interval_matched,
            "execution_strategy": execution_strategy,
            "source_global_step": context.global_step,
            "max_batches": budget.max_batches,
            "min_batches": budget.min_batches,
            "deadline_ts": budget.deadline_ts,
            "require_full_batch": budget.require_full_batch,
            "sample_last_n_steps": budget.sample_last_n_steps,
            "data_version": (
                context.data_status.data_version if context.data_status else None
            ),
            "required_target_version": (
                # Sync exports the current step's head after this status query;
                # only Bubble reuses the quota's historical target version.
                None
                if config.use_logits
                else (
                    context.data_status.target_version
                    if config.execution_strategy
                    is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
                    and context.data_status is not None
                    else _as_int(context.global_step)
                )
            ),
            "plan_id": uuid4().hex,
            "worker_snapshots": (
                context.data_status.worker_snapshots if context.data_status else None
            ),
            "min_sample_step": context.data_status.min_sample_step
            if context.data_status
            else None,
            "max_sample_step": context.data_status.max_sample_step
            if context.data_status
            else None,
            "target_worker_ids": resources.worker_ids if resources else (),
            "training_group_id": resources.training_group_id if resources else "",
            "idle_window_sec": (
                resources.minimum_idle_window_sec if resources is not None else None
            ),
            "idle_usable_window_sec": (
                max(
                    max(
                        resources.minimum_idle_window_sec
                        - self._effective_idle_deadline_guard_sec(config)
                        - self._effective_idle_startup_reserve_sec(
                            config,
                            resources.worker_ids,
                            continuation=idle_continuation,
                        )
                        - float(idle_tail_reserve_sec or 0.0),
                        0.0,
                    )
                    - self._effective_idle_reclaim_penalty_sec(),
                    0.0,
                )
                if resources is not None
                else None
            ),
            "idle_reclaim_penalty_sec": (
                self._effective_idle_reclaim_penalty_sec()
                if resources is not None
                else None
            ),
            "idle_window_source": (
                getattr(resources, "idle_window_source", "runtime_deadline")
                if resources is not None
                else ""
            ),
            "idle_first_step_extra_sec": (
                self._effective_idle_first_step_extra_sec(
                    config, resources.worker_ids, continuation=idle_continuation
                )
                if resources is not None
                else 0.0
            ),
            "idle_continuation": idle_continuation,
            "idle_generation_epochs": (
                tuple(
                    (worker, self._idle_workers[worker].generation_epoch)
                    for worker in resources.worker_ids
                    if worker in self._idle_workers
                )
                if resources is not None
                else ()
            ),
            "idle_window_batches": (
                int(
                    math.floor(
                        max(
                            max(
                                resources.minimum_idle_window_sec
                                - self._effective_idle_deadline_guard_sec(config)
                                - self._effective_idle_startup_reserve_sec(
                                    config,
                                    resources.worker_ids,
                                    continuation=idle_continuation,
                                )
                                - float(idle_tail_reserve_sec or 0.0),
                                0.0,
                            )
                            - self._effective_idle_reclaim_penalty_sec(),
                            0.0,
                        )
                        / (
                            self._effective_idle_batch_estimate_sec(
                                config, resources.worker_ids
                            )
                            * max(gradient_accumulation_steps, 1)
                        )
                    )
                )
                if resources is not None
                else None
            ),
            "idle_batch_estimate_sec": (
                self._effective_idle_batch_estimate_sec(config, resources.worker_ids)
                * max(gradient_accumulation_steps, 1)
                if resources is not None
                else None
            ),
            "idle_startup_reserve_sec": (
                self._effective_idle_startup_reserve_sec(
                    config,
                    resources.worker_ids,
                    continuation=idle_continuation,
                )
                if resources is not None
                else None
            ),
            "idle_tail_reserve_sec": (
                float(idle_tail_reserve_sec or 0.0) if resources is not None else None
            ),
            "idle_trainable_batches": (
                (
                    context.data_status.trainable_batches
                    // max(gradient_accumulation_steps, 1)
                )
                if resources is not None and context.data_status is not None
                else None
            ),
            "idle_confidence": (
                getattr(resources, "idle_confidence", IdleWindowConfidence.CONFIRMED)
                if resources is not None
                else IdleWindowConfidence.CONFIRMED
            ),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "planned_optimizer_steps": planned_optimizer_steps,
            "planned_valid_tokens": planned_valid_tokens,
            "idle_park_after_training": bool(
                resources is not None
                and getattr(resources, "idle_window_source", "runtime_deadline")
                == "runtime_deadline"
                and budget.deadline_ts is not None
                and not config.idle_worker_keep_hot_between_plans
            ),
            "publish_snapshot_deferred": publish_snapshot_deferred,
            "idle_snapshot_estimate_sec": (
                self._effective_idle_tail_reserve_sec(config, final=True)
                if publish_snapshot_deferred
                else 0.0
            ),
        }
        if not trigger.should_train:
            return TrainingPlan(
                launch=False,
                reason=trigger.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches <= 0:
            return TrainingPlan(
                launch=False,
                reason=budget.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches < budget.min_batches:
            return TrainingPlan(
                launch=False,
                reason="insufficient_training_budget",
                publish_after_success=False,
                **common,
            )
        projected_quota_debt = max(
            self._training_quota_debt_steps - int(budget.max_batches),
            0,
        )
        quota_cycle_incomplete = bool(
            config.training_quota_enable
            and config.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and projected_quota_debt > 0
        )
        return TrainingPlan(
            launch=True,
            reason=trigger.reason,
            # Intermediate quota work is intentionally private. Publishing it
            # would expose a partially trained drafter and add snapshot/weight
            # traffic before the boundary top-up publishes the completed cycle.
            publish_after_success=(
                not quota_cycle_incomplete
                and not idle_publish_deferred
                and (
                    self._publish_interval_matched(context.global_step, config)
                    or (
                        config.training_quota_enable
                        and self._training_quota_debt_steps > 0
                    )
                )
            ),
            retain_replay_session=quota_cycle_incomplete,
            **common,
        )

    def execute_training_plan(self, plan: TrainingPlan, *, runtime_state):
        """Execute through the strategy selected by the generated plan."""

        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")

        if plan.execution_strategy is DrafterExecutionStrategy.SYNC or plan.reason in {
            "quota_topup_training_ready",
            "quota_forced_completion_ready",
        }:
            return self.sync_execution_strategy.execute(
                plan,
                executor=self._worker_executor,
                runtime_state=runtime_state,
            )
        if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            return self.rollout_idle_execution_strategy.execute(
                plan,
                executor=self._worker_executor,
                runtime_state=runtime_state,
            )
        raise NotImplementedError(
            f"Unsupported drafter execution strategy: {plan.execution_strategy.value}"
        )

    def poll_pending_training(self, *, runtime_state) -> TrainingOutcome | None:
        plan = runtime_state.active_plan
        if plan is None:
            return None
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        execution = self.rollout_idle_execution_strategy.poll(
            executor=self._worker_executor,
            runtime_state=runtime_state,
        )
        if execution is None:
            return None
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return outcome

    def pending_training_ready(self, *, runtime_state) -> bool:
        """Check whether asynchronous worker work ended without consuming it."""

        if runtime_state.status is not DrafterRuntimeStatus.RUNNING:
            return False
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        ready, _ = self._worker_executor.poll_training(runtime_state.training_ref)
        return bool(ready)

    def wait_pending_training(self, *, runtime_state) -> TrainingOutcome | None:
        plan = runtime_state.active_plan
        if plan is None:
            return None
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        execution = self.rollout_idle_execution_strategy.wait(
            executor=self._worker_executor,
            runtime_state=runtime_state,
        )
        if execution is None:
            return None
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return outcome

    def _record_training_outcome(
        self,
        plan: TrainingPlan,
        outcome: TrainingOutcome,
    ) -> None:
        if (
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and outcome.reason == "worker_result_inconsistent"
            and outcome.successful_steps > 0
        ):
            # Some shards may already have advanced model/optimizer state.
            # A retryable zero-step result can continue with divergent shards.
            raise RuntimeError(
                f"Inconsistent Bubble training state: plan_id={plan.plan_id}; restart from a consistent checkpoint"
            )
        is_quota_topup = plan.reason in {
            "quota_topup_training_ready",
            "quota_forced_completion_ready",
        }
        quota_debt_before = self._training_quota_debt_steps
        if outcome.trained:
            if (
                is_quota_topup
                or plan.execution_strategy
                is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            ):
                self._writer_boundary_migration_allowed = False
            if is_quota_topup:
                self._critical_path_topup_optimizer_steps_total += int(
                    outcome.successful_steps
                )
            elif (
                plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            ):
                self._idle_optimizer_steps_total += int(outcome.successful_steps)
        outcome.metrics.update(
            {
                "bubble/idle_optimizer_steps_total": self._idle_optimizer_steps_total,
                "bubble/critical_path_topup_optimizer_steps_total": self._critical_path_topup_optimizer_steps_total,
            }
        )
        if is_quota_topup and (
            not outcome.trained
            or int(outcome.successful_steps) != int(plan.max_batches)
        ):
            raise RuntimeError(
                "Bubble training quota top-up did not complete its requested "
                "optimizer steps: "
                f"requested={plan.max_batches} "
                f"completed={outcome.successful_steps} "
                f"reason={outcome.reason} plan_id={plan.plan_id}"
            )
        if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            completed_final_plan = bool(
                plan.publish_after_success
                and outcome.trained
                and int(outcome.successful_steps) >= int(plan.max_batches)
            )
            if is_quota_topup:
                # A blocking top-up runs after PPO, not in a rollout lease.
                # Preserve its wall timings without feeding its setup/cleanup
                # into hot idle admission (notably the partial tail reserve).
                outcome.metrics["bubble/topup_overhead_excluded_from_idle_budget"] = 1
                for suffix in (
                    "preflight",
                    "activation",
                    "cleanup",
                    "cleanup_nonpark",
                    "memory_park",
                    "elapsed",
                    "training_loop",
                ):
                    key = f"timing_s/drafter_worker_{suffix}"
                    if key in outcome.metrics:
                        outcome.metrics[f"timing_s/drafter_topup_{suffix}"] = (
                            outcome.metrics[key]
                        )
                park_sec = max(
                    float(
                        outcome.metrics.get("timing_s/drafter_worker_memory_park", 0.0)
                        or 0.0
                    ),
                    0.0,
                )
                if park_sec > 0.0:
                    # Cold plans still need an offload safety reference when
                    # no actual idle park has been measured yet. Hot plans
                    # never pay this reserve.
                    self._quota_topup_park_samples_sec.append(park_sec)
                if completed_final_plan:
                    self._quota_topup_final_tail_samples_sec.append(
                        max(
                            float(
                                outcome.metrics.get(
                                    "timing_s/drafter_worker_cleanup_nonpark",
                                    outcome.metrics.get(
                                        "timing_s/drafter_worker_cleanup", 0.0
                                    ),
                                )
                                or 0.0
                            ),
                            0.0,
                        )
                        + max(
                            float(
                                outcome.metrics.get(
                                    "timing_s/drafter_publish_snapshot", 0.0
                                )
                                or 0.0
                            ),
                            0.0,
                        )
                    )
            else:
                self.record_idle_training_outcome(
                    outcome,
                    final=completed_final_plan,
                    worker_ids=plan.target_worker_ids,
                )
            self._record_replica_local_unavailable(plan, outcome)
            if not is_quota_topup:
                self._record_prebatch_reclaim_penalty(plan, outcome)
                self._record_idle_dynamic_batch_cap(plan, outcome)

        quota_accounted_training = bool(
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            or is_quota_topup
        )
        if (
            quota_accounted_training
            and outcome.trained
            and outcome.successful_steps > 0
            and self._training_quota_debt_steps > 0
        ):
            debt_before = self._training_quota_debt_steps
            self._training_quota_debt_steps = max(
                debt_before - int(outcome.successful_steps), 0
            )
            if self._training_quota_debt_steps == 0:
                self._record_completed_quota_loss(outcome)
                self._training_quota_oldest_cycle_step = None
                self._training_quota_data_version = None
            print(
                "[BubbleTime] training_quota_repaid: "
                f"step={plan.source_global_step} plan_id={plan.plan_id} "
                f"reason={plan.reason} successful_steps={outcome.successful_steps} "
                f"debt_before={debt_before} "
                f"debt_after={self._training_quota_debt_steps}",
                flush=True,
            )

        if quota_accounted_training:
            outcome.metrics.update(
                {
                    "bubble/training_quota_debt_before": quota_debt_before,
                    "bubble/training_quota_debt_after": self._training_quota_debt_steps,
                    "bubble/training_quota_debt_steps": self._training_quota_debt_steps,
                }
            )

        if outcome.trained and outcome.successful_steps > 0:
            bootstrap_worker_ids = _normalize_worker_id_group(
                plan.hot_bootstrap_worker_ids
            )
            bootstrap_groups = self._hot_groups_for_worker_ids(bootstrap_worker_ids)
            if not bootstrap_groups and bootstrap_worker_ids:
                bootstrap_groups = (bootstrap_worker_ids,)
            if bootstrap_groups:
                self._idle_optimizer_initialized_groups.update(bootstrap_groups)
                # Every worker just completed the same full-collective quota.
                # Keep the participating replica-local trainers resident, but
                # do not elect a private writer yet.  The next real Bubble
                # plan can then choose the replica that actually exposes idle.
                self._idle_worker_hot_prewarmed_groups.update(bootstrap_groups)
                self._idle_worker_writer_group = None
                self._idle_worker_writer_state_version = None
                logger.warning(
                    "[BubbleTime] full_collective_hot_bootstrap_completed: "
                    "groups=%s source_step=%s successful_steps=%s",
                    bootstrap_groups,
                    plan.source_global_step,
                    outcome.successful_steps,
                )
            if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
                group = _normalize_worker_id_group(plan.target_worker_ids)
                if group:
                    self._idle_optimizer_initialized_groups.add(group)
                    previous_group_successes = (
                        self._idle_worker_group_success_counts.get(group, 0)
                    )
                    self._idle_worker_group_success_counts[group] = (
                        previous_group_successes + 1
                    )
                    writer_group = self._current_idle_writer_group(assign_default=False)
                    if writer_group is None:
                        self._idle_worker_writer_group = group
                        writer_group = group
                    is_writer = group == writer_group
                    if is_writer:
                        target_version = (
                            plan.required_target_version
                            if plan.required_target_version is not None
                            else plan.data_version
                        )
                        if target_version is not None:
                            self._idle_worker_last_target_version[group] = int(
                                target_version
                            )
                        if outcome.metrics.get("bubble/training_residency_retained", 1):
                            self._idle_worker_hot_prewarmed_groups.add(group)
                        else:
                            self._idle_worker_hot_prewarmed_groups.discard(group)
                        self._idle_worker_writer_state_version = _as_int(
                            plan.source_global_step
                        )
                    logger.debug(
                        "[BubbleTime] idle_group_success_recorded: group=%s "
                        "successful_steps=%s group_successes=%s is_writer=%s "
                        "writer_group=%s hot_groups=%s",
                        group,
                        outcome.successful_steps,
                        self._idle_worker_group_success_counts[group],
                        is_writer,
                        writer_group,
                        tuple(sorted(self._idle_worker_hot_prewarmed_groups)),
                    )
            training_loop_sec = max(
                float(
                    outcome.metrics.get("timing_s/drafter_worker_training_loop", 0.0)
                    or 0.0
                ),
                0.0,
            )
            worker_batch_estimates = [
                (
                    training_loop_sec
                    if training_loop_sec > 0.0
                    # Preserve compatibility with legacy workers that do not
                    # report a split training-loop timing.
                    else max(float(result.elapsed_sec), 0.0)
                )
                / max(result.successful_steps, 1)
                for result in outcome.worker_results
                if result.successful_steps > 0
            ]
            optimizer_step_sec = max(
                worker_batch_estimates,
                default=0.0,
            )
            micro_batch_sec = optimizer_step_sec / max(
                int(plan.gradient_accumulation_steps), 1
            )
            if (
                (
                    micro_batch_sec > 0
                    or any(
                        isinstance(result, dict)
                        and result.get("optimizer_step_samples_sec")
                        for result in outcome.raw_results
                    )
                )
                and plan.execution_strategy
                is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            ):
                measured_steps = []
                first_steps = []
                for result in outcome.raw_results:
                    if (
                        not isinstance(result, dict)
                        or int(result.get("successful_steps", 0) or 0) <= 0
                    ):
                        continue
                    worker_id = result.get("worker_id")
                    if (
                        plan.target_worker_ids
                        and worker_id is not None
                        and str(worker_id) not in plan.target_worker_ids
                    ):
                        continue
                    samples = result.get("optimizer_step_samples_sec")
                    if isinstance(samples, list):
                        warmup = int(result.get("optimizer_warmup_steps", 0) or 0)
                        if (
                            samples
                            and warmup == 0
                            and plan.reason
                            not in {
                                "quota_topup_training_ready",
                                "quota_forced_completion_ready",
                            }
                        ):
                            # A serialized top-up may have a different first
                            # step (e.g. after actor work). It can calibrate
                            # steady throughput, not hot idle restart overhead.
                            first_steps.append(
                                float(samples[0])
                                / max(plan.gradient_accumulation_steps, 1)
                            )
                        measured_steps.append(
                            [
                                float(value)
                                for value in samples[max(warmup, 1) :]
                                if float(value) > 0
                            ]
                        )
                if first_steps:
                    self._idle_worker_first_step_samples_sec.append(max(first_steps))
                    group = _normalize_worker_id_group(plan.target_worker_ids)
                    resident = all(
                        bool(
                            result.get(
                                "activation_started_resident",
                                group in self._idle_worker_hot_prewarmed_groups,
                            )
                        )
                        for result in outcome.raw_results
                        if isinstance(result, dict)
                        and int(result.get("successful_steps", 0) or 0) > 0
                        and (
                            not group
                            or str(result.get("worker_id", result.get("rank", "")))
                            in group
                        )
                    )
                    mode = (
                        "continuation"
                        if plan.idle_continuation and resident
                        else ("hot" if resident else "cold")
                    )
                    self._idle_first_step_by_mode.setdefault(
                        (group, mode), deque(maxlen=4)
                    ).append(max(first_steps))
                if measured_steps:
                    shared_count = min(len(samples) for samples in measured_steps)
                    if shared_count == 0:
                        # A single cold optimizer step is not a steady-state
                        # throughput measurement; keep the prior admission estimate.
                        return
                    normalized = [
                        max(samples[index] for samples in measured_steps)
                        / max(plan.gradient_accumulation_steps, 1)
                        for index in range(shared_count)
                    ]
                    # Use a short rolling history so an initialization outlier
                    # cannot suppress every future idle window.
                    if not is_quota_topup:
                        group = _normalize_worker_id_group(plan.target_worker_ids)
                        if group:
                            self._idle_steady_by_group.setdefault(
                                group, deque(maxlen=4)
                            ).extend(normalized)
                        self._idle_worker_batch_samples_sec = deque(
                            [*self._idle_worker_batch_samples_sec, *normalized][-8:],
                            maxlen=32,
                        )
                    micro_batch_sec = _conservative_percentile(normalized, quantile=0.5)
                else:
                    normalized = [micro_batch_sec]
                    if not is_quota_topup:
                        self._idle_worker_batch_samples_sec.append(micro_batch_sec)
                if is_quota_topup:
                    self._quota_topup_batch_samples_sec = deque(
                        [*self._quota_topup_batch_samples_sec, *normalized][-8:],
                        maxlen=32,
                    )
                    return
                if self._idle_worker_batch_estimate_sec is None:
                    self._idle_worker_batch_estimate_sec = micro_batch_sec
                else:
                    self._idle_worker_batch_estimate_sec = (
                        self._idle_worker_batch_estimate_sec * 0.8
                        + micro_batch_sec * 0.2
                    )
                logger.info(
                    "[BubbleTime] updated idle batch estimate: step=%s "
                    "observed_micro_batch_s=%.3f estimated_micro_batch_s=%.3f "
                    "gradient_accumulation_steps=%s optimizer_step_s=%.3f "
                    "successful_steps=%s elapsed_s=%.3f",
                    plan.source_global_step,
                    micro_batch_sec,
                    self._idle_worker_batch_estimate_sec,
                    plan.gradient_accumulation_steps,
                    optimizer_step_sec,
                    outcome.successful_steps,
                    outcome.elapsed_sec,
                )

    def _record_completed_quota_loss(
        self,
        outcome: TrainingOutcome,
    ) -> None:
        """Use end-of-quota loss as a secondary refresh signal.

        Online accept length remains the primary freshness measurement. Loss
        is sampled only while training, so it can request an earlier *next*
        quota but can never shorten the current quota.
        """

        candidates: list[float] = []
        for result in outcome.raw_results:
            if not isinstance(result, dict):
                continue
            direct = result.get("train/avg_loss")
            try:
                if direct is not None:
                    candidates.append(_as_float(direct))
                    continue
            except (TypeError, ValueError):
                pass
            position_losses: list[float] = []
            for key, value in result.items():
                if "/loss_per_position/" not in str(key):
                    continue
                try:
                    position_losses.append(_as_float(value))
                except (TypeError, ValueError):
                    continue
            if position_losses:
                candidates.append(sum(position_losses) / len(position_losses))
        if not candidates:
            return
        latest = sum(candidates) / len(candidates)
        if not math.isfinite(latest) or latest < 0.0:
            return
        self._quality_latest_loss = latest
        if self._quality_loss_baseline is None:
            self._quality_loss_baseline = latest
            self._quality_loss_refresh_requested = False
            return
        increase_ratio = self._quality_loss_increase_ratio
        self._quality_loss_refresh_requested = bool(
            latest > self._quality_loss_baseline * (1.0 + increase_ratio)
        )
        if not self._quality_loss_refresh_requested:
            self._quality_loss_baseline = (
                self._quality_loss_baseline * 0.8 + latest * 0.2
            )

    def record_training_quota_publish_completed(
        self,
        global_step: object | None = None,
    ) -> None:
        """Release collection backpressure only after publication succeeds."""

        if self._training_quota_debt_steps <= 0:
            self._writer_boundary_migration_allowed = True
            had_active_quota = self._training_quota_collection_step is not None
            completed_step = global_step
            if completed_step is None:
                completed_step = self._training_quota_collection_step
            if had_active_quota and completed_step is not None:
                self._quality_last_publish_step = _as_int(completed_step)
                self._quality_pending_acceptance_baseline = True
            self._training_quota_collection_step = None
            released_writer_group = self._idle_worker_writer_group
            released_writer_version = self._idle_worker_writer_state_version
            if released_writer_group is not None or released_writer_version is not None:
                # Publishing inference weights does not copy Adam moments or
                # scheduler state. Preserve the authoritative training owner
                # across quotas until a complete state migration is implemented.
                logger.warning(
                    "[BubbleTime] idle_writer_retained_after_quota_publish: "
                    "group=%s state_version=%s completed_step=%s",
                    released_writer_group,
                    released_writer_version,
                    completed_step,
                )

    def request_reclaim(self, worker_ids: tuple[str, ...]) -> Any:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        for worker_id in worker_ids:
            if worker_id in self._idle_workers:
                self._idle_workers[worker_id].status = "reclaiming"
        request = self._worker_executor.request_reclaim(
            tuple(str(worker_id) for worker_id in worker_ids)
        )
        return self._worker_executor.resolve(request)

    @staticmethod
    def _publish_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        interval = _as_int(config.publish_interval_steps or 0)
        return interval <= 0 or _as_int(global_step) % interval == 0

    def plan_publish(
        self,
        *,
        global_step: object,
        drafter_trained: bool,
        config: DrafterScheduleConfig,
        training_plan: TrainingPlan | None = None,
    ) -> PublishPlan:
        # A Bubble Time publish is issued only after the upstream actor weight
        # update has resumed vLLM.  Keep the transfer off that critical path;
        # the generation hook waits only if it is still pending at its next
        # safe point.  The legacy synchronous path continues to honor the
        # explicit publish_async setting.
        asynchronous = config.publish_async or (
            training_plan is not None
            and training_plan.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        )
        if not drafter_trained or (
            training_plan is not None and not training_plan.publish_after_success
        ):
            return PublishPlan(
                publish=False,
                reason=(
                    "drafter_not_trained"
                    if not drafter_trained
                    else "training_plan_publish_disabled"
                ),
                interval_matched=False,
                source_global_step=global_step,
                asynchronous=asynchronous,
            )
        if (
            training_plan is not None
            and training_plan.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and config.training_quota_enable
            and self._training_quota_debt_steps > 0
        ):
            return PublishPlan(
                publish=False,
                reason="training_quota_incomplete",
                interval_matched=False,
                source_global_step=global_step,
                asynchronous=asynchronous,
            )
        # Preserve the released path exactly: invalid publish configuration is
        # an error instead of being silently converted into a skipped publish.
        quota_completed_plan = bool(
            training_plan is not None
            and training_plan.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and config.training_quota_enable
            and training_plan.publish_after_success
            and self._training_quota_debt_steps == 0
        )
        interval_matched = quota_completed_plan or self._publish_interval_matched(
            global_step, config
        )
        return PublishPlan(
            publish=interval_matched,
            reason=(
                "training_quota_complete"
                if quota_completed_plan
                else "publish_interval_reached"
                if interval_matched
                else "publish_interval_not_reached"
            ),
            interval_matched=interval_matched,
            source_global_step=global_step,
            asynchronous=asynchronous,
        )

    def execute_publish_plan(self, plan: PublishPlan):
        if self._publish_executor is None:
            raise RuntimeError("Drafter publish executor has not been bound")
        return self.publish_execution_strategy.execute(
            plan, executor=self._publish_executor
        )
