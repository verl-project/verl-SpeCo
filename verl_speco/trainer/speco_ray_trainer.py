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
"""SPECO adapter for the legacy RayPPOTrainer in verl 0.8 and 0.9."""

import hashlib
import json
import logging
import os
import time
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import MethodType
from typing import Any, cast

import ray
import torch
from omegaconf import open_dict
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl_speco.integration.agent_loop_runtime import (
    SPECO_AGENT_LOOP_MANAGER_CLASS,
    install_agent_loop_runtime_patch,
)
from verl_speco.integration.rollout_publish import resolve_drafter_publish_payload
from verl_speco.integration.rollout_idle_events import (
    DRAFTER_SAMPLE_READY_EVENT,
    SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV,
    configure_rollout_resource_groups,
    drain_rollout_idle_events,
    ensure_rollout_idle_event_bus,
    release_rollout_training_resources,
    reserve_rollout_training_resources,
)
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY,
    OLD_LOGPROB_AUX_LAYER_IDS_KEY,
    OLD_LOGPROB_COLLECT_MASK_KEY,
    OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
    OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY,
    OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
    OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
    OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
    OLD_LOGPROB_HIDDEN_REF_META_KEY,
    OLD_LOGPROB_HIDDEN_REFS_KEY,
    OLD_LOGPROB_HIDDEN_STATES_KEY,
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_TIMING_KEY,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    assert_sglang_aux_last_layer_norm_safe,
    resolve_drafter_hidden_states_layout,
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.integration.sglang_adapter import (
    DRAFTER_SAMPLE_KEY,
    normalize_drafter_samples,
    pop_drafter_samples,
)
from verl_speco.integration.sglang_runtime import (
    clear_sglang_runtime_config,
    configure_sglang_runtime_from_config,
    install_upstream_sglang_runtime_bridge,
    should_install_sglang_base_compat_runtime,
)
from verl_speco.integration.vllm_runtime import (
    SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
    configure_vllm_runtime_from_config,
)
from verl_speco.trainer.bubble_profiler import inject_bubble_metrics
from verl_speco.trainer.scheduler import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    CallbackDrafterCollectionExecutor,
    CallbackDrafterPublishExecutor,
    CallbackDrafterWorkerExecutor,
    CollectionPlan,
    CollectionPayload,
    CollectionOutcome,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    IdleWindowConfidence,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    TrainingPlan,
)
from verl_speco.workers import SpecoWorker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC = (
    "drafter/spec_decode/mean_acceptance_length"
)
_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY = "_speco_vllm_spec_decode_drafts"
_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY = "_speco_vllm_spec_decode_accepted_tokens"
_SPECO_DRAFTER_TIMING_DEDUCTED_KEY = "_speco_drafter_timing_deducted_from_update_actor"
_SPECO_BOOTSTRAP_FALLBACK_IDLE_WINDOW_SEC = 10.0
_DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"
# Keep the coordinator's readiness bookkeeping aligned with the worker-side
# target-head cache (``_target_lm_head_snapshot_limit = 2``).  This is not a
# tensor cache; it prevents stale readiness markers from authorizing a version
# that the worker has already evicted.
_SPECO_TARGET_LM_HEAD_READY_VERSION_LIMIT = 2

_DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS = {
    None,
    "",
    "null",
    "None",
    "/path/to/drafter/checkpoint",
}
_POLICY_MODEL_NON_TENSOR_KEYS = {"multi_modal_inputs", "pad_token_id"}


def _select_policy_model_batch(batch: DataProto) -> DataProto:
    """Keep rollout/drafter side-channel data out of policy-model forward paths."""
    non_tensor_batch_keys = [
        key for key in _POLICY_MODEL_NON_TENSOR_KEYS if key in batch.non_tensor_batch
    ]
    return batch.select(non_tensor_batch_keys=non_tensor_batch_keys)


def _get_nested(config, path, default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _speco_is_ray_object_ref(value: Any) -> bool:
    object_ref_type = getattr(ray, "ObjectRef", ())
    return bool(object_ref_type) and isinstance(value, object_ref_type)


def _speco_alpha_counter(value: int) -> str:
    """Encode a positive counter with letters so Ray log dedup keeps each sample."""

    value = max(int(value), 1)
    chars = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))


def _speco_ref_meta_rows(meta: Any) -> int:
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("rows", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _speco_ref_meta_nbytes(meta: Any) -> int:
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("nbytes", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _speco_ref_meta_row_count(meta: Any, default: int = 0) -> int:
    if not isinstance(meta, dict):
        return int(default)
    row_indices = meta.get("chunk_row_indices")
    if torch.is_tensor(row_indices):
        row_indices = cast(torch.Tensor, row_indices)
        return int(row_indices.numel())
    if isinstance(row_indices, (list, tuple)):
        return len(row_indices)
    try:
        return int(meta.get("chunk_length", meta.get("rows", default)) or 0)
    except (TypeError, ValueError):
        return int(default)


def _speco_metric_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _speco_move_drafter_timing_next_to_update_actor(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    drafter_elapsed = _speco_metric_float(data.get("timing_s/drafter"))
    mean_acceptance_length = _speco_metric_float(
        data.get(SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC)
    )
    update_actor_elapsed = _speco_metric_float(data.get("timing_s/update_actor"))
    already_deducted = bool(data.get(_SPECO_DRAFTER_TIMING_DEDUCTED_KEY))
    if (
        drafter_elapsed is None
        and mean_acceptance_length is None
        and not already_deducted
    ):
        return data

    adjusted_update_actor = None
    adjusted_update_actor_per_token = None
    if (
        drafter_elapsed is not None
        and update_actor_elapsed is not None
        and not already_deducted
    ):
        adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
        update_actor_per_token = _speco_metric_float(
            data.get("timing_per_token_ms/update_actor")
        )
        if update_actor_per_token is not None:
            adjusted_update_actor_per_token = (
                update_actor_per_token * adjusted_update_actor / update_actor_elapsed
                if update_actor_elapsed > 0
                else 0.0
            )

    rewritten = {}
    inserted_drafter_metrics = False
    for key, value in data.items():
        if key in {
            "timing_s/drafter",
            SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC,
            _SPECO_DRAFTER_TIMING_DEDUCTED_KEY,
        }:
            continue
        if key == "timing_s/update_actor":
            rewritten[key] = (
                adjusted_update_actor if adjusted_update_actor is not None else value
            )
            if drafter_elapsed is not None:
                rewritten["timing_s/drafter"] = drafter_elapsed
            if mean_acceptance_length is not None:
                rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = (
                    mean_acceptance_length
                )
            inserted_drafter_metrics = True
        elif (
            key == "timing_per_token_ms/update_actor"
            and adjusted_update_actor_per_token is not None
        ):
            rewritten[key] = adjusted_update_actor_per_token
        else:
            rewritten[key] = value
    if not inserted_drafter_metrics:
        if drafter_elapsed is not None:
            rewritten["timing_s/drafter"] = drafter_elapsed
        if mean_acceptance_length is not None:
            rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = (
                mean_acceptance_length
            )
    return rewritten


def _speco_float_values(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]

    normalized = []
    for value in values:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _speco_vllm_spec_decode_stats_from_batch(batch: Any) -> dict[str, float]:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if not isinstance(non_tensor_batch, dict):
        return {}

    def values(name: str) -> list[float]:
        return _speco_float_values(
            non_tensor_batch.get(f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_{name}")
        )

    drafts = values("drafts")
    accepted_tokens = values("accepted_tokens")
    total_drafts = float(sum(drafts))
    total_accepted_tokens = float(sum(accepted_tokens))
    if total_drafts <= 0.0 and total_accepted_tokens <= 0.0:
        return {}

    return {
        _SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: total_drafts,
        _SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: total_accepted_tokens,
    }


def _speco_vllm_spec_decode_metrics_from_stats(
    stats: dict[str, float],
) -> dict[str, float]:
    drafts = float(stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0)
    if drafts <= 0.0:
        return {}
    accepted_tokens = float(
        stats.get(_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY, 0.0) or 0.0
    )
    return {
        SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC: 1.0 + accepted_tokens / drafts
    }


def _speco_truthy_meta_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _speco_generation_meta_info(value: Any) -> dict[str, Any] | None:
    meta_info = getattr(value, "meta_info", None)
    if isinstance(meta_info, dict):
        return meta_info
    if isinstance(value, dict):
        meta_info = value.get("meta_info")
        if isinstance(meta_info, dict):
            return meta_info
    return None


def _speco_is_validation_generation_value(value: Any) -> bool:
    meta_info = _speco_generation_meta_info(value)
    if not isinstance(meta_info, dict):
        return False
    for key in ("validate", "validation", "is_validate", "is_validation", "test"):
        if key in meta_info and _speco_truthy_meta_value(meta_info.get(key)):
            return True
    phase = (
        str(
            meta_info.get("phase")
            or meta_info.get("split")
            or meta_info.get("mode")
            or meta_info.get("stage")
            or ""
        )
        .strip()
        .lower()
    )
    return phase in {"validate", "validation", "val", "test", "eval", "evaluation"}


def _speco_is_validation_generation(
    args: tuple[Any, ...], kwargs: dict[str, Any], output: Any = None
) -> bool:
    candidates = [output, *args]
    for key in ("batch", "prompts", "data", "input_batch"):
        if key in kwargs:
            candidates.append(kwargs[key])
    return any(
        _speco_is_validation_generation_value(candidate) for candidate in candidates
    )


def _speco_set_generation_meta_flag(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key: str,
    value: bool,
):
    """Set one request-scoped generation flag and return a restore callback."""

    candidates = [*args]
    for candidate_key in ("batch", "prompts", "data", "input_batch"):
        if candidate_key in kwargs:
            candidates.append(kwargs[candidate_key])
    missing = object()
    for candidate in candidates:
        meta_info = _speco_generation_meta_info(candidate)
        if not isinstance(meta_info, dict):
            continue
        previous = meta_info.get(key, missing)
        meta_info[key] = bool(value)

        def restore(meta_info=meta_info, previous=previous) -> None:
            if previous is missing:
                meta_info.pop(key, None)
            else:
                meta_info[key] = previous

        return restore
    return lambda: None


def _speco_merge_vllm_spec_decode_stats(
    existing: dict[str, float] | None,
    current: dict[str, float],
) -> dict[str, float]:
    if not current:
        return existing or {}
    totals = {
        _SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 0.0,
        _SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 0.0,
    }
    for key in totals:
        totals[key] = float((existing or {}).get(key, 0.0) or 0.0) + float(
            current.get(key, 0.0) or 0.0
        )
    return totals


class SpecoRayPPOTrainer(RayPPOTrainer):
    """External trainer adapter for SPECO.

    Normal PPO still delegates to upstream ``RayPPOTrainer.fit``. SPECO online
    drafter training installs scoped hooks around that loop, delegating normal PPO
    behavior while keeping SPECO collection/training/publishing in
    ``verl_speco`` instead of requiring external ``verl`` source edits.
    """

    def __init__(self, *args, **kwargs):
        self.speco_worker_cls = kwargs.pop("speco_worker_cls", None)
        super().__init__(*args, **kwargs)
        self.drafter_wg = None
        self._drafter_scheduler = DrafterScheduler()
        self._drafter_runtime_state = DrafterRuntimeState()
        self._pending_drafter_publish_refs = None
        self._pending_drafter_publish_contexts: list[dict[str, Any]] = []
        self._speco_published_drafter_version: int | None = None
        self._pending_drafter_checkpoint_refs = []
        self._pending_target_lm_head_sync = None
        self._speco_ready_target_lm_head_versions: set[int] = set()
        self._speco_ready_target_lm_head_workers: dict[int, tuple[str, ...]] = {}
        self._speco_last_known_trainable_batches: int | None = None
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_oldlogprob_candidate_samples = 0
        self._speco_last_oldlogprob_planned_samples = 0
        self._speco_last_oldlogprob_collected_samples = 0
        self._speco_last_oldlogprob_collected_rows = 0
        self._speco_last_oldlogprob_payload_mib = 0.0
        self._speco_last_oldlogprob_select_elapsed_sec = 0.0
        self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
        self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
        self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
        self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
        self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
        self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
        self._speco_last_oldlogprob_total_elapsed_sec = 0.0
        self._speco_last_collect_interval_matched = 0
        self._speco_last_collection_outcome = None
        self._speco_bubble_lock = threading.RLock()
        self._speco_last_rollout_idle_metrics: dict[str, Any] = {}
        self._speco_direct_sample_ids: set[str] = set()
        self._speco_direct_collected_samples = 0
        self._speco_runtime_reclaim_requested_plan_id: str | None = None
        self._speco_rollout_training_lease: tuple[str, tuple[str, ...]] | None = None
        self._speco_runtime_idle_callback_verified = False
        # A rollout event loop may drain events several times while one
        # generation is active.  Keep generation-scoped totals separately
        # from the last drain's metrics so a real idle callback never gets
        # mistaken for a missing callback at generation completion.
        self._speco_runtime_idle_events_this_generation = 0
        self._speco_runtime_sample_events_this_generation = 0
        self._speco_deferred_publish_count = 0

    def attach_speco_worker_group(self, worker_group):
        self.drafter_wg = worker_group
        self._speco_get_drafter_scheduler().bind_worker_executor(
            CallbackDrafterWorkerExecutor(
                submit=self.speco_train_drafter,
                resolve=self._ray_get_if_needed,
                inspect_data=self.speco_get_drafter_training_data_status,
                prepare=self._speco_prepare_drafter_training_rpc,
                activate=self.speco_activate_drafter_training_model,
                prewarm=self.speco_prewarm_drafter_training_model,
                preflight=self.speco_preflight_drafter_training,
                abort_preflight=self.speco_abort_drafter_training_preflight,
                poll=self._speco_poll_drafter_training,
                reclaim=self.speco_request_drafter_training_reclaim,
            )
        )
        self._speco_register_drafter_training_resource_metadata()
        self._speco_get_drafter_scheduler().bind_collection_executor(
            CallbackDrafterCollectionExecutor(
                set_step=self.speco_set_global_step,
                stage_submit=self.speco_stage_rollout_features,
                commit_submit=self.speco_commit_rollout_features,
                abort_submit=self.speco_abort_rollout_features,
                rollback_submit=self.speco_rollback_rollout_features,
                finalize_submit=self.speco_finalize_rollout_features,
                resolve=self._ray_get_if_needed,
            )
        )
        self._speco_bind_publish_executor()

    def _speco_bind_publish_executor(self) -> None:
        self._speco_get_drafter_scheduler().bind_publish_executor(
            CallbackDrafterPublishExecutor(
                wait=self._speco_wait_pending_drafter_publish_rpc,
                fetch=self._speco_get_published_drafter_weights,
                update=self._speco_update_rollout_drafter_weights,
                normalize_payload=resolve_drafter_publish_payload,
            )
        )

    def _require_speco_worker_group(self):
        if self.drafter_wg is None:
            raise RuntimeError("SpecoWorker group has not been initialized yet.")
        return self.drafter_wg

    def speco_get_drafter_training_resource_metadata(self):
        return (
            self._require_speco_worker_group().get_drafter_training_resource_metadata()
        )

    def _speco_register_drafter_training_resource_metadata(self) -> dict[str, Any]:
        if self.drafter_wg is None:
            return {}
        try:
            metadata = self._ray_get_if_needed(
                self.speco_get_drafter_training_resource_metadata()
            )
            metrics = self._speco_get_drafter_scheduler().register_idle_training_resource_metadata(
                metadata
            )
            scheduler = self._speco_get_drafter_scheduler()
            lease_groups_configured = configure_rollout_resource_groups(
                self._speco_rollout_idle_event_bus_name(),
                scheduler.rollout_idle_replica_groups(),
            )
            metrics["bubble/resource_lease_groups_configured"] = int(
                lease_groups_configured
            )
            logger.warning(
                "[BubbleTime] registered training resource metadata: groups=%s "
                "workers=%s replica_groups=%s",
                metrics.get("bubble/registered_training_groups", 0),
                metrics.get("bubble/registered_training_workers", 0),
                metrics.get("bubble/registered_replica_groups", 0),
            )
            return metrics
        except Exception:  # noqa: BLE001
            logger.warning(
                "[BubbleTime] failed to discover drafter training resource metadata; "
                "falling back to idle_worker.group_size/training_groups if configured.",
                exc_info=True,
            )
            return {"bubble/training_resource_metadata_error": 1}

    def speco_set_global_step(self, global_step: int):
        return self._require_speco_worker_group().set_global_step(global_step)

    def speco_stage_rollout_features(self, requests: list[list[dict]]):
        return self._require_speco_worker_group().stage_rollout_features(requests)

    def speco_commit_rollout_features(self, requests: list[list[dict]]):
        return self._require_speco_worker_group().commit_rollout_features(requests)

    def speco_abort_rollout_features(self, requests: list[list[dict]]):
        return self._require_speco_worker_group().abort_rollout_features(requests)

    def speco_rollback_rollout_features(self, requests: list[list[dict]]):
        return self._require_speco_worker_group().rollback_rollout_features(requests)

    def speco_finalize_rollout_features(self, requests: list[list[dict]]):
        return self._require_speco_worker_group().finalize_rollout_features(requests)

    def speco_sync_target_lm_head_weight(self, payload: Any, global_step: Any = None):
        return self._require_speco_worker_group().sync_target_lm_head_weight(
            payload, global_step=global_step
        )

    def speco_get_drafter_target_lm_head_row_indices(self):
        return (
            self._require_speco_worker_group().get_drafter_target_lm_head_row_indices()
        )

    def speco_train_drafter(self, training_plan: dict[str, object]):
        return self._require_speco_worker_group().train_drafter(training_plan)

    def _speco_poll_drafter_training(self, submission):
        refs = submission if isinstance(submission, list) else [submission]
        if not refs:
            return (True, submission)
        if not all(_speco_is_ray_object_ref(ref) for ref in refs):
            return (True, submission)
        ready_refs, pending_refs = ray.wait(
            refs,
            num_returns=len(refs),
            timeout=0,
        )
        if pending_refs:
            return (False, submission)
        return (True, ready_refs)

    def speco_request_drafter_training_reclaim(self, worker_ids: tuple[str, ...]):
        return self._require_speco_worker_group().request_drafter_training_reclaim(
            list(worker_ids)
        )

    def speco_preflight_drafter_training(self, training_plan: dict[str, object]):
        return self._require_speco_worker_group().preflight_drafter_training(
            training_plan
        )

    def speco_abort_drafter_training_preflight(self, plan_id: str):
        return self._require_speco_worker_group().abort_drafter_training_preflight(
            plan_id
        )

    def speco_get_drafter_training_data_status(
        self,
        sample_last_n_steps: int,
        require_full_batch: bool,
        worker_ids: tuple[str, ...] | None = None,
        target_version: int | None = None,
    ):
        return self._require_speco_worker_group().get_drafter_training_data_status(
            sample_last_n_steps,
            require_full_batch,
            target_version,
        )

    def speco_activate_drafter_training_model(self):
        return self._require_speco_worker_group().activate_drafter_training_model()

    def speco_prewarm_drafter_training_model(
        self, worker_ids: tuple[str, ...] | None = None
    ):
        return self._require_speco_worker_group().prewarm_drafter_training_model(
            list(worker_ids) if worker_ids is not None else None
        )

    def speco_maybe_publish(self):
        return self._require_speco_worker_group().maybe_publish()

    def speco_commit_staged_drafter_weights(self, global_step: object):
        return self._speco_actor_rollout_method("commit_staged_draft_weights")(
            global_steps=global_step
        )

    def speco_save_checkpoint(
        self,
        global_step: int,
        wait: bool = True,
        worker_ids: tuple[str, ...] | None = None,
    ):
        return self._require_speco_worker_group().save_checkpoint(
            global_step,
            wait=wait,
            worker_ids=worker_ids,
        )

    def speco_wait_checkpoint(self):
        return self._require_speco_worker_group().wait_checkpoint()

    def init_workers(self):
        drafter_rollout_enabled = self.is_drafter_rollout_enabled(self.config)
        online_drafter_enabled = self.is_drafter_training_enabled(self.config)
        print(
            "[BubbleTime] init_workers: "
            f"rollout_enabled={drafter_rollout_enabled} "
            f"training_enabled={online_drafter_enabled} "
            f"rollout_name={_get_nested(self.config, ('actor_rollout_ref', 'rollout', 'name'), None)} "
            f"strategy={self._speco_drafter_schedule_config().execution_strategy.value}",
            flush=True,
        )
        if online_drafter_enabled:
            self._speco_prepare_drafter_checkpoint_for_worker_init()
        if drafter_rollout_enabled:
            if online_drafter_enabled:
                self._speco_configure_rollout_idle_event_bus()
            configure_sglang_runtime_from_config(self.config)
            configure_vllm_runtime_from_config(self.config)
            if online_drafter_enabled:
                install_agent_loop_runtime_patch()
            if (
                _get_nested(self.config, ("actor_rollout_ref", "rollout", "name"), None)
                == "sglang"
            ):
                install_upstream_sglang_runtime_bridge()
        else:
            clear_sglang_runtime_config()
            if should_install_sglang_base_compat_runtime(self.config):
                install_upstream_sglang_runtime_bridge(base_compat_only=True)
        with self._hide_speco_drafter_config_from_upstream_rollout():
            with self._use_speco_agent_loop_manager(online_drafter_enabled):
                super().init_workers()
        if online_drafter_enabled:
            self._init_speco_drafter_workers()
            # Fail closed on the divergent SGLang last-layer-norm combination at
            # init, before any (expensive) rollout generation runs.
            self._speco_validate_sglang_aux_last_layer_norm()

    @contextmanager
    def _use_speco_agent_loop_manager(self, enabled: bool):
        if not enabled:
            yield
            return
        manager_class = SPECO_AGENT_LOOP_MANAGER_CLASS

        rollout_config = _get_nested(
            self.config, ("actor_rollout_ref", "rollout"), None
        )
        if rollout_config is None:
            yield
            return

        missing = object()
        original_agent = (
            rollout_config.get("agent", missing)
            if hasattr(rollout_config, "get")
            else missing
        )
        agent_config = original_agent if original_agent is not missing else {}
        previous_manager_class = (
            agent_config.get("agent_loop_manager_class", missing)
            if hasattr(agent_config, "get")
            else missing
        )
        with open_dict(rollout_config):
            if "agent" not in rollout_config or rollout_config["agent"] is None:
                rollout_config["agent"] = {}
            rollout_config["agent"]["agent_loop_manager_class"] = manager_class
        try:
            yield
        finally:
            with open_dict(rollout_config):
                if original_agent is missing:
                    del rollout_config["agent"]
                elif previous_manager_class is missing:
                    rollout_config["agent"] = original_agent
                    rollout_config["agent"].pop("agent_loop_manager_class", None)
                else:
                    rollout_config["agent"] = original_agent
                    rollout_config["agent"]["agent_loop_manager_class"] = (
                        previous_manager_class
                    )

    @contextmanager
    def _hide_speco_drafter_config_from_upstream_rollout(self):
        rollout_config = _get_nested(
            self.config, ("actor_rollout_ref", "rollout"), None
        )
        missing = object()
        drafter_config = missing
        if rollout_config is not None and "drafter" in rollout_config:
            drafter_config = rollout_config["drafter"]
            with open_dict(rollout_config):
                del rollout_config["drafter"]
        try:
            yield
        finally:
            if drafter_config is not missing:
                with open_dict(rollout_config):
                    rollout_config["drafter"] = drafter_config

    def _init_speco_drafter_workers(self):
        if self.drafter_wg is not None:
            return

        speco_worker_cls = self.speco_worker_cls or ray.remote(SpecoWorker)
        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self.role_worker_mapping
            else Role.ActorRollout
        )
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        drafter_cls = RayClassWithInitArgs(
            cls=speco_worker_cls,
            config=self.config.actor_rollout_ref,
            role="drafter",
            device_name=self.device_name,
        )

        worker_group = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=drafter_cls,
            name_prefix="speco_drafter",
            device_name=self.device_name,
        )
        worker_group.init_model()
        self.attach_speco_worker_group(worker_group)

    def _ray_get_if_needed(self, value):
        if value is None:
            return None
        try:
            import ray
        except Exception:  # noqa: BLE001
            return value

        object_ref_type = getattr(ray, "ObjectRef", ())
        if object_ref_type and isinstance(value, object_ref_type):
            return ray.get(value)
        if isinstance(value, (list, tuple)) and value and object_ref_type:
            if all(isinstance(item, object_ref_type) for item in value):
                return ray.get(list(value))
        return value

    @staticmethod
    def _first_non_null(value):
        if isinstance(value, (list, tuple)):
            non_null = [item for item in value if item is not None]
            if len(non_null) > 1:
                raise RuntimeError(
                    f"Expected at most one non-null SPECO result, got {len(non_null)}"
                )
            return non_null[0] if non_null else None
        return value

    def _speco_online_enabled(self) -> bool:
        return self.is_drafter_training_enabled(self.config)

    def _speco_drafter_training_config(self):
        return _get_nested(
            self.config, ("actor_rollout_ref", "rollout", "drafter", "training"), {}
        )

    def _speco_drafter_config(self):
        return _get_nested(
            self.config, ("actor_rollout_ref", "rollout", "drafter"), None
        )

    @staticmethod
    def _speco_set_config_value(config, key: str, value: Any):
        try:
            with open_dict(config):
                config[key] = value
        except Exception:  # noqa: BLE001
            if hasattr(config, "__setitem__"):
                config[key] = value
            else:
                setattr(config, key, value)

    def _speco_ensure_drafter_checkpoint_path(self) -> str | None:
        drafter_cfg = self._speco_drafter_config()
        if drafter_cfg is None:
            return None

        checkpoint_path = (
            drafter_cfg.get("checkpoint_path", None)
            if hasattr(drafter_cfg, "get")
            else getattr(drafter_cfg, "checkpoint_path", None)
        )
        if checkpoint_path not in _DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS:
            return checkpoint_path

        default_local_dir = _get_nested(
            self.config, ("trainer", "default_local_dir"), None
        )
        if default_local_dir in (None, ""):
            return None

        checkpoint_path = os.path.join(str(default_local_dir), "drafter")
        self._speco_set_config_value(drafter_cfg, "checkpoint_path", checkpoint_path)
        return checkpoint_path

    def _speco_drafter_checkpoint_save_config_enabled(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        if hasattr(training_cfg, "get"):
            return bool(training_cfg.get("save_full_drafter_checkpoint", True))
        return True

    def _speco_resume_global_step_hint(self) -> int | None:
        trainer_cfg = _get_nested(self.config, ("trainer",), None)
        resume_mode = str(
            _get_nested(trainer_cfg, ("resume_mode",), "disable") or "disable"
        )
        if resume_mode == "disable":
            return None

        global_step_folder = None
        if resume_mode == "resume_path":
            global_step_folder = _get_nested(trainer_cfg, ("resume_from_path",), None)
        elif resume_mode == "auto":
            checkpoint_folder = _get_nested(trainer_cfg, ("default_local_dir",), None)
            if checkpoint_folder:
                checkpoint_folder = os.path.abspath(os.fspath(checkpoint_folder))
                try:
                    from verl.utils.checkpoint.checkpoint_manager import (
                        find_latest_ckpt_path,
                    )

                    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Unable to resolve latest actor checkpoint for drafter resume: %s",
                        exc,
                    )

        if not global_step_folder:
            return None
        folder_name = os.path.basename(os.path.normpath(os.fspath(global_step_folder)))
        if not folder_name.startswith("global_step_"):
            return None
        try:
            return int(folder_name.removeprefix("global_step_"))
        except ValueError:
            return None

    def _speco_prepare_drafter_checkpoint_for_worker_init(self):
        drafter_cfg = self._speco_drafter_config()
        if drafter_cfg is None:
            return

        checkpoint_save_enabled = self._speco_drafter_checkpoint_save_config_enabled()
        if checkpoint_save_enabled:
            self._speco_ensure_drafter_checkpoint_path()

        training_cfg = self._speco_drafter_training_config()
        resume_setting = training_cfg.get("resume_trainer_state_from_checkpoint", None)
        if resume_setting is None:
            resume_setting = training_cfg.get(
                "resume_lr_scheduler_from_checkpoint", True
            )
        if not bool(resume_setting):
            return

        resume_step = self._speco_resume_global_step_hint()
        if resume_step is None:
            return

        from verl_speco.trainer.checkpoint import (
            get_drafter_checkpoint_step,
            resolve_drafter_checkpoint_path,
        )

        model_path = _get_nested(drafter_cfg, ("model_path",), None)
        checkpoint_path = _get_nested(drafter_cfg, ("checkpoint_path",), None)
        resolved_path = resolve_drafter_checkpoint_path(
            model_path, checkpoint_path, resume_step
        )
        if resolved_path is None:
            return
        if os.path.normpath(resolved_path) == os.path.normpath(
            os.fspath(model_path or "")
        ):
            if get_drafter_checkpoint_step(resolved_path) != resume_step:
                message = (
                    f"[drafter resume] no complete draft_step_{resume_step} checkpoint under "
                    f"{checkpoint_path}; model_path={model_path}"
                )
                if checkpoint_save_enabled:
                    raise RuntimeError(message)
                logger.warning("%s; starting drafter state from model_path", message)
            return
        self._speco_set_config_value(drafter_cfg, "model_path", resolved_path)
        logger.info(
            "[drafter resume] resolved global_step=%s checkpoint=%s",
            resume_step,
            resolved_path,
        )

    def _speco_should_save_drafter_checkpoint(self) -> bool:
        if not self.is_drafter_training_enabled(self.config):
            return False
        if self._speco_drafter_training_mode() == "collect_only":
            return False
        if self.drafter_wg is None:
            return False
        if not self._speco_drafter_checkpoint_save_config_enabled():
            return False
        return True

    @staticmethod
    def _speco_flatten_checkpoint_results(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                flattened.extend(
                    SpecoRayPPOTrainer._speco_flatten_checkpoint_results(item)
                )
            return flattened
        return []

    @classmethod
    def _speco_validate_drafter_checkpoint_results(
        cls, value: Any, *, require_saved: bool
    ) -> None:
        results = cls._speco_flatten_checkpoint_results(value)
        allowed_skips = {
            "not_checkpoint_group",
            "not_checkpoint_replica",
            "not_in_training_group",
        }
        failures = [
            result
            for result in results
            if not bool(result.get("saved", False))
            and result.get("reason") not in allowed_skips
        ]
        if failures:
            raise RuntimeError(f"Drafter checkpoint failed: {failures}")
        if require_saved and not any(
            bool(result.get("saved", False)) for result in results
        ):
            raise RuntimeError(f"Drafter checkpoint produced no saved state: {results}")

    def _speco_save_drafter_checkpoint(self, *, wait: bool = True):
        if not self._speco_should_save_drafter_checkpoint():
            return None
        if self._speco_ensure_drafter_checkpoint_path() is None:
            return None
        checkpoint_worker_ids = None
        if self._speco_rollout_idle_worker_enabled():
            checkpoint_worker_ids = (
                self._speco_get_drafter_scheduler().idle_checkpoint_group()
            )
            if not checkpoint_worker_ids:
                raise RuntimeError(
                    "Bubble drafter checkpoint has no complete writer group"
                )
        checkpoint_refs = self.speco_save_checkpoint(
            self.global_steps,
            wait=wait,
            worker_ids=checkpoint_worker_ids,
        )
        if wait:
            results = self._ray_get_if_needed(checkpoint_refs)
            self._speco_validate_drafter_checkpoint_results(results, require_saved=True)
            return results
        if not hasattr(self, "_pending_drafter_checkpoint_refs"):
            self._pending_drafter_checkpoint_refs = []
        self._pending_drafter_checkpoint_refs.append(checkpoint_refs)
        return checkpoint_refs

    def _speco_wait_pending_drafter_checkpoint(self) -> int:
        pending_refs = getattr(self, "_pending_drafter_checkpoint_refs", None)
        if not pending_refs:
            return 0
        self._pending_drafter_checkpoint_refs = []
        for refs in pending_refs:
            results = self._ray_get_if_needed(refs)
            self._speco_validate_drafter_checkpoint_results(results, require_saved=True)
        wait_results = self._ray_get_if_needed(self.speco_wait_checkpoint())
        incomplete = [
            result
            for result in self._speco_flatten_checkpoint_results(wait_results)
            if result.get("completed") is False
        ]
        if incomplete:
            raise RuntimeError(f"Drafter checkpoint wait failed: {incomplete}")
        return len(pending_refs)

    def _speco_plan_drafter_collection(
        self,
        source: DrafterCollectionSource,
        *,
        validation: bool = False,
    ) -> CollectionPlan:
        self._speco_last_collection_outcome = None
        training_cfg = self._speco_drafter_training_config()
        source_enabled = bool(
            training_cfg.get(
                "collect_hidden_states_from_sgl"
                if source is DrafterCollectionSource.SGLANG
                else "collect_hidden_states_from_old_logprob",
                False,
            )
        )
        plan = self._speco_get_drafter_scheduler().plan_collection(
            DrafterCollectionContext(
                global_step=self.global_steps,
                source=source,
                drafter_enabled=self._speco_online_enabled(),
                source_enabled=source_enabled,
                validation=validation,
                require_training_interval=(
                    source is DrafterCollectionSource.OLD_LOGPROB
                    and not self._speco_rollout_idle_worker_enabled()
                ),
            ),
            self._speco_drafter_schedule_config(),
        )
        config = self._speco_drafter_schedule_config()
        if (
            self._speco_rollout_idle_worker_enabled()
            and plan.collect
            and self._speco_get_drafter_scheduler().idle_writer_migration_blocked()
        ):
            plan = replace(
                plan,
                collect=False,
                reason="writer_state_migration_required",
            )
            print(
                "[BubbleTime] collection_skipped: "
                f"step={self.global_steps} source={source.value} "
                "reason=writer_state_migration_required",
                flush=True,
            )
        target_batches = config.idle_worker_collection_target_batches
        known_batches = getattr(self, "_speco_last_known_trainable_batches", None)
        if (
            source is DrafterCollectionSource.SGLANG
            and self._speco_rollout_idle_worker_enabled()
            and plan.collect
            and target_batches is not None
            and known_batches is not None
            and known_batches >= max(int(target_batches), 1)
        ):
            plan = replace(plan, collect=False, reason="buffer_target_reached")
            print(
                "[BubbleTime] collection_skipped: "
                f"step={self.global_steps} source={source.value} "
                f"reason=buffer_target_reached trainable_batches={known_batches} "
                f"target_batches={target_batches}",
                flush=True,
            )
        self._speco_last_collection_plan = plan
        return plan

    @staticmethod
    def _speco_log_drafter_collection_plan(plan: CollectionPlan) -> None:
        logger.info(
            "[DrafterScheduler] collection step=%s source=%s collect=%s reason=%s "
            "collect_interval_matched=%s training_interval_matched=%s "
            "sample_rate=%s max_samples_per_replica=%s max_tokens_per_replica=%s "
            "window_mode=%s window_tokens=%s window_min_rows=%s",
            plan.source_global_step,
            plan.source.value,
            plan.collect,
            plan.reason,
            plan.collect_interval_matched,
            plan.training_interval_matched,
            plan.sample_rate,
            plan.max_samples_per_replica,
            plan.max_tokens_per_replica,
            plan.hidden_window_mode,
            plan.hidden_window_tokens_per_sample,
            plan.hidden_window_min_rows,
        )

    def _speco_get_drafter_scheduler(self) -> DrafterScheduler:
        scheduler = getattr(self, "_drafter_scheduler", None)
        if scheduler is None:
            scheduler = DrafterScheduler()
            self._drafter_scheduler = scheduler
        return scheduler

    def _speco_get_drafter_runtime_state(self) -> DrafterRuntimeState:
        runtime_state = getattr(self, "_drafter_runtime_state", None)
        if runtime_state is None:
            runtime_state = DrafterRuntimeState()
            self._drafter_runtime_state = runtime_state
        return runtime_state

    def _speco_drafter_schedule_config(self) -> DrafterScheduleConfig:
        return DrafterScheduleConfig.from_mapping(self._speco_drafter_training_config())

    def _speco_rollout_idle_worker_enabled(self) -> bool:
        return (
            self._speco_online_enabled()
            and self._speco_drafter_schedule_config().execution_strategy.value
            == "rollout_idle_worker"
        )

    def _speco_should_prefetch_target_lm_head(self) -> bool:
        """Return whether the current collection needs an async head snapshot.

        This decision follows the configured execution mode rather than the
        transient plan selected for the current scheduler event, so a Bubble
        run prepares the LM-head version required by the next idle window.
        """
        return (
            self._speco_rollout_idle_worker_enabled()
            and int(getattr(self, "_speco_last_collected_samples", 0) or 0) > 0
        )

    def _speco_rollout_idle_event_bus_name(self) -> str | None:
        training_cfg = self._speco_drafter_training_config()
        name = _get_nested(
            training_cfg,
            ("scheduler", "idle_worker", "event_bus_name"),
            None,
        )
        if name:
            return str(name)
        return os.getenv(SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV)

    def _speco_set_rollout_idle_event_bus_name(self, name: str) -> None:
        training_cfg = self._speco_drafter_training_config()
        try:
            from omegaconf import OmegaConf

            context = (
                open_dict(training_cfg)
                if OmegaConf.is_config(training_cfg)
                else nullcontext()
            )
        except Exception:  # noqa: BLE001
            context = nullcontext()
        with context:
            scheduler_cfg = _get_nested(training_cfg, ("scheduler",), None)
            if scheduler_cfg is None:
                training_cfg["scheduler"] = {}
                scheduler_cfg = training_cfg["scheduler"]
            idle_cfg = _get_nested(scheduler_cfg, ("idle_worker",), None)
            if idle_cfg is None:
                scheduler_cfg["idle_worker"] = {}
                idle_cfg = scheduler_cfg["idle_worker"]
            idle_cfg["event_bus_name"] = name
        os.environ[SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV] = name

    def _speco_configure_rollout_idle_event_bus(self) -> str | None:
        if not self._speco_rollout_idle_worker_enabled():
            os.environ.pop(SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV, None)
            return None
        name = self._speco_rollout_idle_event_bus_name()
        if not name:
            name = f"speco-rollout-idle-{os.getpid()}-{uuid.uuid4().hex}"
            self._speco_set_rollout_idle_event_bus_name(name)
        else:
            os.environ[SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV] = name
        ensure_rollout_idle_event_bus(name)
        return name

    def _speco_rollout_idle_poll_interval_sec(self) -> float:
        training_cfg = self._speco_drafter_training_config()
        value = _get_nested(
            training_cfg,
            ("scheduler", "idle_worker", "event_poll_interval_sec"),
            0.05,
        )
        try:
            return max(float(value), 0.01)
        except (TypeError, ValueError):
            return 0.05

    def _speco_bubble_training_lock(self):
        lock = getattr(self, "_speco_bubble_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._speco_bubble_lock = lock
        return lock

    def _speco_drain_rollout_idle_events(self) -> dict[str, Any]:
        name = self._speco_rollout_idle_event_bus_name()
        metrics: dict[str, Any] = {}
        events = drain_rollout_idle_events(name)
        sample_events: list[dict[str, Any]] = []
        lifecycle_events: list[dict[str, Any]] = []
        for event in events:
            target = (
                sample_events
                if str(event.get("event_type", "")).lower()
                == DRAFTER_SAMPLE_READY_EVENT
                else lifecycle_events
            )
            target.append(event)

        # Commit samples first even when SAMPLE_READY and WORKER_IDLE arrive in
        # one drain.  The scheduler can then only observe the newly versioned
        # buffer after the collection transaction has completed.
        sample_metrics = self._speco_commit_runtime_drafter_samples(sample_events)
        for key, value in sample_metrics.items():
            metrics[key] = metrics.get(key, 0) + value

        for event in lifecycle_events:
            scheduler = self._speco_get_drafter_scheduler()
            event_metrics = scheduler.on_worker_event(event)
            metrics.update(event_metrics)
            event_type = str(event.get("event_type", "")).lower()
            if event_type == RolloutWorkerEventType.GENERATION_STARTED.value:
                runtime_state = self._speco_get_drafter_runtime_state()
                active_plan = runtime_state.active_plan
                replica_workers = scheduler.rollout_idle_worker_ids_for_replica(
                    int(event.get("replica_rank", 0) or 0),
                    fallback_worker_id=str(event.get("worker_id", "")),
                )
                if (
                    runtime_state.status is DrafterRuntimeStatus.RUNNING
                    and active_plan is not None
                    and set(replica_workers).intersection(active_plan.target_worker_ids)
                    and getattr(self, "_speco_runtime_reclaim_requested_plan_id", None)
                    != active_plan.plan_id
                ):
                    scheduler.request_reclaim(active_plan.target_worker_ids)
                    self._speco_runtime_reclaim_requested_plan_id = active_plan.plan_id
                    metrics["bubble/runtime_request_reclaim_requested"] = 1
                    print(
                        "[BubbleTime] runtime_request_reclaim_requested: "
                        f"plan_id={active_plan.plan_id} "
                        f"request_replica={event.get('replica_rank')} "
                        f"workers={active_plan.target_worker_ids}",
                        flush=True,
                    )
            if (
                event_type == RolloutWorkerEventType.WORKER_IDLE.value
                and bool(event.get("memory_released", False))
                and int(event_metrics.get("bubble/idle_training_groups", 0)) > 0
            ):
                # Only treat the runtime callback as established after it has
                # produced at least one complete collective group.  A single
                # replica callback is insufficient proof for group training.
                self._speco_runtime_idle_callback_verified = True
            if event_type == RolloutWorkerEventType.WORKER_IDLE.value:
                self._speco_runtime_idle_events_this_generation = (
                    int(getattr(self, "_speco_runtime_idle_events_this_generation", 0))
                    + 1
                )
        if sample_events:
            self._speco_runtime_sample_events_this_generation = int(
                getattr(self, "_speco_runtime_sample_events_this_generation", 0)
            ) + len(sample_events)
        if events:
            metrics["bubble/runtime_worker_events_drained"] = len(lifecycle_events)
            metrics["bubble/runtime_sample_events_drained"] = len(sample_events)
            logger.warning(
                "[BubbleTime] drained rollout events: lifecycle=%s samples=%s "
                "idle_workers=%s callback_verified=%s generation_idle_events=%s",
                len(lifecycle_events),
                len(sample_events),
                metrics.get("bubble/idle_workers", 0),
                self._speco_runtime_idle_callback_verified,
                self._speco_runtime_idle_events_this_generation,
            )
        return metrics

    def _speco_commit_runtime_drafter_samples(
        self, events: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Commit all ready SGLang samples from one event-loop poll together."""

        if not events:
            return {}
        metrics: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        sample_ids: list[str] = []
        try:
            current_step = int(self.global_steps)
        except (TypeError, ValueError):
            current_step = -1
        for event in events:
            sample_id = str(event.get("sample_id", "") or "")
            if not sample_id or sample_id in self._speco_direct_sample_ids:
                metrics["bubble/direct_sample_duplicate"] = (
                    metrics.get("bubble/direct_sample_duplicate", 0) + 1
                )
                continue
            try:
                event_step = int(cast(Any, event.get("global_step")))
            except (TypeError, ValueError):
                event_step = -1
            if event_step != current_step:
                logger.warning(
                    "[BubbleTime] direct_sample_rejected: sample_id=%s "
                    "reason=step_mismatch event_step=%s current_step=%s",
                    sample_id,
                    event_step,
                    current_step,
                )
                metrics["bubble/direct_sample_stale"] = (
                    metrics.get("bubble/direct_sample_stale", 0) + 1
                )
                continue
            sample = self._ray_get_if_needed(event.get("sample_ref"))
            if not isinstance(sample, dict):
                logger.warning(
                    "[BubbleTime] direct_sample_rejected: sample_id=%s reason=invalid_ref",
                    sample_id,
                )
                metrics["bubble/direct_sample_rejected"] = (
                    metrics.get("bubble/direct_sample_rejected", 0) + 1
                )
                continue
            sample.setdefault("source_replica_rank", sample.get("replica_rank"))
            sample.setdefault("_speco_global_sample_id", sample_id)
            samples.append(sample)
            sample_ids.append(sample_id)

        if not samples:
            return metrics
        with self._speco_bubble_training_lock():
            collection_plan = self._speco_plan_drafter_collection(
                DrafterCollectionSource.SGLANG
            )
            if not collection_plan.collect:
                metrics["bubble/direct_sample_not_planned"] = len(samples)
                return metrics
            routed_owners = (
                [owner_ranks[index % len(owner_ranks)] for index in range(len(samples))]
                if (owner_ranks := self._speco_bubble_collection_owner_ranks())
                else None
            )
            payload = self._speco_get_drafter_scheduler().prepare_collection_payload(
                source=DrafterCollectionSource.SGLANG,
                samples=samples,
                owners=routed_owners,
                owner_count=self._speco_sglang_collection_owner_count(
                    samples,
                    routed_owners,
                ),
                dispatch_bucket_count=self._speco_dispatch_bucket_count(),
                raw_samples=len(samples),
                collection_id=collection_plan.collection_id,
            )
            outcome = self._speco_execute_collection(collection_plan, payload)
            if not outcome.collected:
                logger.warning(
                    "[BubbleTime] direct_sample_batch_rejected: count=%s reason=%s",
                    len(samples),
                    outcome.reason,
                )
                metrics["bubble/direct_sample_rejected"] = metrics.get(
                    "bubble/direct_sample_rejected", 0
                ) + len(samples)
                return metrics
            self._speco_direct_sample_ids.update(sample_ids)
            self._speco_direct_collected_samples += outcome.collected_samples
            self._speco_last_collected_samples = self._speco_direct_collected_samples
            logger.warning(
                "[BubbleTime] direct_sample_batch_committed: count=%s replicas=%s "
                "source_step=%s collected=%s",
                len(samples),
                tuple(sorted({str(sample.get("replica_rank")) for sample in samples})),
                current_step,
                outcome.collected_samples,
            )
            metrics["bubble/direct_sample_committed"] = outcome.collected_samples
            return metrics

    def _speco_record_rollout_idle_metrics(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            return
        existing = getattr(self, "_speco_last_rollout_idle_metrics", None)
        if not isinstance(existing, dict):
            existing = {}
            self._speco_last_rollout_idle_metrics = existing
        existing.update(metrics)

    def _speco_pop_rollout_idle_metrics(self) -> dict[str, Any]:
        metrics = dict(getattr(self, "_speco_last_rollout_idle_metrics", {}) or {})
        self._speco_last_rollout_idle_metrics = {}
        return metrics

    @staticmethod
    def _speco_plan_requires_rollout_resource_lease(plan: TrainingPlan) -> bool:
        return bool(
            plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            and plan.target_worker_ids
            and plan.reason
            not in {
                "quota_topup_training_ready",
                "quota_forced_completion_ready",
            }
        )

    def _speco_reserve_rollout_training_lease(self, plan: TrainingPlan) -> bool:
        if not self._speco_plan_requires_rollout_resource_lease(plan):
            return True
        # Release is tied to the actual Ray training future, not a wall-clock
        # estimate. Expiring a slow NPU plan could otherwise permit generation
        # to overlap its final batch and recreate the OOM/corruption hazard.
        expires_at = 0.0
        acquired = reserve_rollout_training_resources(
            self._speco_rollout_idle_event_bus_name(),
            worker_ids=tuple(str(worker_id) for worker_id in plan.target_worker_ids),
            plan_id=plan.plan_id,
            expires_at=expires_at,
        )
        if acquired:
            self._speco_rollout_training_lease = (
                plan.plan_id,
                tuple(str(worker_id) for worker_id in plan.target_worker_ids),
            )
            self._speco_runtime_reclaim_requested_plan_id = None
            print(
                "[BubbleTime] training_resource_lease_acquired: "
                f"plan_id={plan.plan_id} workers={plan.target_worker_ids} "
                "expiry=training_future",
                flush=True,
            )
        return acquired

    def _speco_release_rollout_training_lease(
        self,
        plan: TrainingPlan | None = None,
    ) -> bool:
        lease = getattr(self, "_speco_rollout_training_lease", None)
        if not lease:
            return False
        plan_id, worker_ids = lease
        if plan is not None and plan.plan_id != plan_id:
            return False
        released = release_rollout_training_resources(
            self._speco_rollout_idle_event_bus_name(),
            worker_ids=tuple(worker_ids),
            plan_id=str(plan_id),
        )
        if released:
            self._speco_rollout_training_lease = None
            print(
                "[BubbleTime] training_resource_lease_released: "
                f"plan_id={plan_id} workers={worker_ids}",
                flush=True,
            )
        return released

    def _speco_release_completed_rollout_training_lease(self) -> dict[str, Any]:
        lease = getattr(self, "_speco_rollout_training_lease", None)
        if not lease:
            return {}
        runtime_state = self._speco_get_drafter_runtime_state()
        active_plan = runtime_state.active_plan
        if active_plan is None:
            return {}
        if not self._speco_get_drafter_scheduler().pending_training_ready(
            runtime_state=runtime_state
        ):
            return {}
        released = self._speco_release_rollout_training_lease(active_plan)
        return {"bubble/training_resource_lease_released": int(released)}

    def _speco_try_launch_rollout_idle_training(self) -> dict[str, Any]:
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        if bool(getattr(self, "_speco_checkpoint_in_progress", False)):
            return {"bubble/training_blocked_by_checkpoint": 1}
        config = self._speco_drafter_schedule_config()
        pending_publish_count = int(bool(self._pending_drafter_publish_refs)) + int(
            getattr(self, "_speco_deferred_publish_count", 0) or 0
        )
        if pending_publish_count >= config.idle_worker_max_pending_publish:
            return {
                "bubble/training_blocked_by_publish_backpressure": 1,
                "bubble/publish_pending_count": pending_publish_count,
            }
        runtime_state = self._speco_get_drafter_runtime_state()
        if runtime_state.status in {
            DrafterRuntimeStatus.SUBMITTED,
            DrafterRuntimeStatus.RUNNING,
        }:
            active_plan = runtime_state.active_plan
            logger.info(
                "[BubbleTime] skip idle launch: pending_training status=%s "
                "plan_id=%s workers=%s",
                runtime_state.status.name,
                getattr(active_plan, "plan_id", ""),
                getattr(active_plan, "target_worker_ids", ()),
            )
            return {"scheduler/pending_training_count": 1}
        with self._speco_bubble_training_lock():
            runtime_state = self._speco_get_drafter_runtime_state()
            if runtime_state.status in {
                DrafterRuntimeStatus.SUBMITTED,
                DrafterRuntimeStatus.RUNNING,
            }:
                active_plan = runtime_state.active_plan
                logger.info(
                    "[BubbleTime] skip idle launch: pending_training status=%s "
                    "plan_id=%s workers=%s",
                    runtime_state.status.name,
                    getattr(active_plan, "plan_id", ""),
                    getattr(active_plan, "target_worker_ids", ()),
                )
                return {"scheduler/pending_training_count": 1}
            head_metrics = self._speco_poll_target_lm_head_prefetch()
            event = self._speco_on_before_actor_update(allow_quota_topup=False)
            plan = event.training_plan
            metrics = dict(event.metrics or {})
            metrics.update(head_metrics)
            if plan is not None and plan.idle_trainable_batches is not None:
                self._speco_last_known_trainable_batches = int(
                    plan.idle_trainable_batches
                )
            required_target_version = (
                None if plan is None else plan.required_target_version
            )
            required_target_workers = (
                tuple(str(worker_id) for worker_id in plan.target_worker_ids)
                if plan is not None
                else ()
            )
            ready_target_workers = (
                self._speco_ready_target_lm_head_workers.get(
                    int(required_target_version)
                )
                if required_target_version is not None
                else None
            )
            target_lm_head_ready_for_workers = (
                required_target_version is None
                or self._speco_target_lm_head_ready_for_workers(
                    required_target_version,
                    required_target_workers,
                )
            )
            if (
                plan is not None
                and plan.launch
                and required_target_version is not None
                and not target_lm_head_ready_for_workers
            ):
                print(
                    "[BubbleTime] idle_launch_blocked: "
                    f"plan_id={plan.plan_id} reason=target_lm_head_not_ready "
                    f"required_target_version={required_target_version} "
                    f"required_workers={required_target_workers} "
                    f"ready_workers={ready_target_workers} "
                    "ready_versions="
                    f"{tuple(sorted(self._speco_ready_target_lm_head_versions))}",
                    flush=True,
                )
                plan = replace(
                    plan,
                    launch=False,
                    reason="target_lm_head_not_ready",
                    max_batches=0,
                    publish_after_success=False,
                )
                metrics.update(
                    {
                        "scheduler/train_requested": 0,
                        "scheduler/planned_batches": 0,
                        "bubble/target_lm_head_not_ready": 1,
                    }
                )
            elif (
                plan is not None and plan.launch and required_target_version is not None
            ):
                logger.debug(
                    "[BubbleTime] target_lm_head_ready_gate: plan_id=%s "
                    "required_target_version=%s required_workers=%s ready_workers=%s "
                    "ready_versions=%s",
                    plan.plan_id,
                    required_target_version,
                    required_target_workers,
                    ready_target_workers,
                    tuple(sorted(self._speco_ready_target_lm_head_versions)),
                )
            print(
                "[BubbleTime] idle_launch_decision: "
                f"launch={getattr(plan, 'launch', False)} "
                f"reason={getattr(plan, 'reason', None)} "
                f"max_batches={getattr(plan, 'max_batches', 0)} "
                f"group_id={getattr(plan, 'training_group_id', '')} "
                f"workers={getattr(plan, 'target_worker_ids', ())} "
                f"idle_window_s={getattr(plan, 'idle_window_sec', None)} "
                f"usable_window_s={getattr(plan, 'idle_usable_window_sec', None)} "
                f"window_batches={getattr(plan, 'idle_window_batches', None)} "
                f"trainable_batches={getattr(plan, 'idle_trainable_batches', None)} "
                f"batch_estimate_s={getattr(plan, 'idle_batch_estimate_sec', None)} "
                f"startup_reserve_s={getattr(plan, 'idle_startup_reserve_sec', None)} "
                f"tail_reserve_s={getattr(plan, 'idle_tail_reserve_sec', None)} "
                f"reclaim_penalty_s={getattr(plan, 'idle_reclaim_penalty_sec', None)} "
                f"deadline_ts={getattr(plan, 'deadline_ts', None)} "
                f"window_source={getattr(plan, 'idle_window_source', '')} "
                "worker_deadline_in_s="
                f"{max(float(getattr(plan, 'deadline_ts', 0.0) or 0.0) - time.time(), 0.0):.3f} "
                f"idle_workers={metrics.get('bubble/idle_workers', 0)} "
                f"idle_groups={metrics.get('bubble/idle_training_groups', 0)}",
                flush=True,
            )
            if plan is None or not plan.launch:
                if plan is not None:
                    self._speco_log_drafter_training_plan(plan, metrics)
                return metrics
            self._speco_log_drafter_training_plan(plan, metrics)
            if not self._speco_reserve_rollout_training_lease(plan):
                metrics.update(
                    {
                        "scheduler/train_requested": 0,
                        "scheduler/planned_batches": 0,
                        "bubble/training_resource_lease_raced": 1,
                    }
                )
                print(
                    "[BubbleTime] idle_launch_blocked: "
                    f"plan_id={plan.plan_id} reason=rollout_resource_lease_raced "
                    f"workers={plan.target_worker_ids}",
                    flush=True,
                )
                return metrics
            try:
                _, train_metrics = self._speco_train_drafter(plan)
            except Exception:
                self._speco_release_rollout_training_lease(plan)
                raise
            runtime_state = self._speco_get_drafter_runtime_state()
            if runtime_state.status is not DrafterRuntimeStatus.RUNNING:
                self._speco_release_rollout_training_lease(plan)
            metrics.update(train_metrics)
            return metrics

    def _speco_service_rollout_idle_events(
        self,
        *,
        allow_launch: bool = True,
    ) -> dict[str, Any]:
        metrics = self._speco_drain_rollout_idle_events()
        metrics.update(self._speco_release_completed_rollout_training_lease())
        if metrics and allow_launch:
            metrics.update(self._speco_try_launch_rollout_idle_training())
        elif metrics:
            metrics["bubble/post_generation_launch_suppressed"] = 1
        self._speco_record_rollout_idle_metrics(metrics)
        return metrics

    def _speco_start_rollout_idle_event_loop(self):
        if not self._speco_rollout_idle_worker_enabled():
            return None, None
        bus_name = self._speco_rollout_idle_event_bus_name()
        if not bus_name:
            print(
                "[BubbleTime] idle_event_loop: not_started reason=missing_event_bus",
                flush=True,
            )
            logger.warning(
                "[BubbleTime] rollout idle event loop not started: missing event bus"
            )
            return None, None
        poll_interval = self._speco_rollout_idle_poll_interval_sec()
        print(
            "[BubbleTime] idle_event_loop: starting "
            f"bus={bus_name} poll_interval_s={poll_interval:.3f}",
            flush=True,
        )
        logger.warning(
            "[BubbleTime] starting rollout idle event loop: bus=%s poll_interval_s=%.3f",
            bus_name,
            poll_interval,
        )
        stop_event = threading.Event()

        def run_loop() -> None:
            while not stop_event.wait(poll_interval):
                try:
                    self._speco_service_rollout_idle_events()
                except Exception:  # noqa: BLE001
                    logger.exception("[BubbleTime] rollout idle event loop failed")

        thread = threading.Thread(
            target=run_loop,
            name="speco-rollout-idle-events",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    @staticmethod
    def _speco_stop_rollout_idle_event_loop(stop_event, thread) -> None:
        if stop_event is None or thread is None:
            return
        stop_event.set()
        thread.join(timeout=1.0)

    def _speco_rollout_idle_worker_ids(self) -> tuple[str, ...]:
        config = self._speco_drafter_schedule_config()
        if config.idle_worker_training_groups:
            return tuple(
                dict.fromkeys(
                    worker_id
                    for group in config.idle_worker_training_groups
                    for worker_id in group
                )
            )
        rollout_cfg = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
        rollout_dp = int(_get_nested(rollout_cfg, ("data_parallel_size",), 1) or 1)
        return tuple(str(replica_rank) for replica_rank in range(max(rollout_dp, 1)))

    @staticmethod
    def _speco_rollout_replica_rank(worker_id: str, fallback: int) -> int:
        try:
            return int(worker_id)
        except (TypeError, ValueError):
            suffix = str(worker_id).rsplit("-", 1)[-1]
            try:
                return int(suffix)
            except (TypeError, ValueError):
                return int(fallback)

    def _speco_emit_rollout_generation_started(self) -> dict[str, Any]:
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        scheduler = self._speco_get_drafter_scheduler()
        metrics: dict[str, Any] = {}
        for index, worker_id in enumerate(self._speco_rollout_idle_worker_ids()):
            metrics.update(
                scheduler.on_worker_event(
                    RolloutWorkerEvent(
                        RolloutWorkerEventType.GENERATION_STARTED,
                        worker_id=worker_id,
                        replica_rank=self._speco_rollout_replica_rank(worker_id, index),
                    )
                )
            )
        return metrics

    def _speco_emit_rollout_generation_completed(
        self,
        gen_batch_output: Any,
    ) -> dict[str, Any]:
        del gen_batch_output
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        generation_complete_ts = time.time()
        runtime_idle_events = int(
            getattr(self, "_speco_runtime_idle_events_this_generation", 0)
        )
        if runtime_idle_events > 0:
            print(
                "[BubbleTime] idle_window_closed: "
                "reason=generation_complete "
                f"runtime_idle_events_this_generation={runtime_idle_events} "
                f"generation_complete_ts={generation_complete_ts:.6f}",
                flush=True,
            )
        # Request-local vLLM counters only provide speculative idle events.
        # The outer generation return is the first authoritative boundary at
        # which no more requests from this generation can be assigned. Close
        # the measured tail here (rather than at next GENERATION_STARTED) and
        # promote still-idle workers to confirmed for safe post-generation
        # Bubble work.
        metrics = self._speco_get_drafter_scheduler().record_generation_completed(
            generation_complete_ts,
            confirm_speculative_idle=(runtime_idle_events > 0),
            must_be_ready_at=(
                self._speco_rollout_idle_fallback_deadline_ts()
                if runtime_idle_events > 0
                else None
            ),
        )
        runtime_state = self._speco_get_drafter_runtime_state()
        active_plan = runtime_state.active_plan
        if active_plan is None or not active_plan.target_worker_ids:
            logger.info("[BubbleTime] generation completed: no active idle training")
            return metrics
        self._speco_get_drafter_scheduler().request_reclaim(
            active_plan.target_worker_ids
        )
        logger.warning(
            "[BubbleTime] generation completed: reclaim requested plan_id=%s workers=%s",
            active_plan.plan_id,
            active_plan.target_worker_ids,
        )
        metrics["bubble/reclaim_requested"] = 1
        return metrics

    @staticmethod
    def _speco_generation_output_replica_ranks(
        gen_batch_output: Any,
    ) -> tuple[int, ...]:
        non_tensor_batch = getattr(gen_batch_output, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return ()
        samples = normalize_drafter_samples(non_tensor_batch.get(DRAFTER_SAMPLE_KEY))
        ranks: set[int] = set()
        for sample in samples:
            replica_rank = sample.get("replica_rank")
            if replica_rank is None:
                continue
            try:
                ranks.add(int(replica_rank))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(ranks))

    def _speco_fallback_idle_replica_ranks(
        self,
        gen_batch_output: Any,
    ) -> tuple[int, ...]:
        replica_ranks = self._speco_generation_output_replica_ranks(gen_batch_output)
        if replica_ranks:
            return replica_ranks
        scheduler = self._speco_get_drafter_scheduler()
        metadata_ranks = scheduler.rollout_idle_replica_ranks()
        if metadata_ranks:
            return metadata_ranks
        return tuple(
            self._speco_rollout_replica_rank(worker_id, index)
            for index, worker_id in enumerate(self._speco_rollout_idle_worker_ids())
        )

    def _speco_rollout_idle_fallback_deadline_ts(self) -> float:
        scheduler = self._speco_get_drafter_scheduler()
        config = self._speco_drafter_schedule_config()
        batch_estimate = scheduler._effective_idle_batch_estimate_sec(config)
        guard = scheduler._effective_idle_deadline_guard_sec(config)
        min_window = scheduler._minimum_idle_training_window_sec(config)
        max_batches = 1
        if not scheduler._idle_batch_estimate_is_bootstrap(config):
            max_batches = max(int(config.train_batches_per_trigger), 1)
        startup = scheduler._effective_idle_startup_reserve_sec(config)
        tail = scheduler._effective_idle_tail_reserve_sec(config)
        fallback_window = max(
            startup + batch_estimate * max(max_batches, 1) + tail + guard,
            min_window,
        )
        if scheduler._idle_batch_estimate_is_bootstrap(config):
            fallback_window = max(
                fallback_window,
                _SPECO_BOOTSTRAP_FALLBACK_IDLE_WINDOW_SEC,
            )
        return time.time() + fallback_window

    def _speco_emit_rollout_idle_from_generation_output(
        self,
        gen_batch_output: Any,
        *,
        reason: str,
    ) -> dict[str, Any]:
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        config = self._speco_drafter_schedule_config()
        if config.idle_worker_require_runtime_idle_events:
            # A synthetic deadline is only a timing estimate.  In strict mode
            # do not start collective drafter training until the runtime has
            # positively reported every required rollout worker as idle.
            logger.info(
                "[BubbleTime] fallback idle disabled: reason=%s; "
                "require_runtime_idle_events=true",
                reason,
            )
            return {"bubble/fallback_idle_events_disabled": 1}
        replica_ranks = self._speco_fallback_idle_replica_ranks(gen_batch_output)
        if not replica_ranks:
            print(
                f"[BubbleTime] fallback_idle: skipped reason={reason} no_replica_ranks",
                flush=True,
            )
            logger.warning(
                "[BubbleTime] fallback idle events skipped: reason=%s "
                "no drafter_sample replica ranks in generation output",
                reason,
            )
            return {"bubble/fallback_idle_events_skipped": 1}
        worker_ids = self._speco_rollout_idle_worker_ids()
        deadline_ts = self._speco_rollout_idle_fallback_deadline_ts()
        emitted_worker_ids: list[str] = []
        scheduler = self._speco_get_drafter_scheduler()
        for replica_rank in replica_ranks:
            fallback_worker_id = (
                worker_ids[replica_rank]
                if 0 <= replica_rank < len(worker_ids)
                else None
            )
            for worker_id in scheduler.rollout_idle_worker_ids_for_replica(
                replica_rank,
                fallback_worker_id=fallback_worker_id,
            ):
                emitted_worker_ids.append(worker_id)
                scheduler.on_worker_event(
                    RolloutWorkerEvent(
                        RolloutWorkerEventType.WORKER_IDLE,
                        worker_id=worker_id,
                        replica_rank=replica_rank,
                        # A trainer-side fallback can prove that the outer
                        # generation call returned, but cannot prove that the
                        # runtime released request/KV resources.  Never forge
                        # that stronger signal.
                        memory_released=False,
                        release_source="synthetic_generation_complete",
                        idle_confidence=IdleWindowConfidence.CONFIRMED,
                        must_be_ready_at=deadline_ts,
                    )
                )
        logger.warning(
            "[BubbleTime] fallback idle events emitted: reason=%s "
            "replica_ranks=%s worker_ids=%s deadline_in_s=%.3f",
            reason,
            replica_ranks,
            tuple(emitted_worker_ids),
            max(deadline_ts - time.time(), 0.0),
        )
        print(
            "[BubbleTime] fallback_idle: emitted "
            f"reason={reason} replica_ranks={replica_ranks} "
            f"worker_ids={tuple(emitted_worker_ids)}",
            flush=True,
        )
        metrics = scheduler.idle_worker_metrics()
        metrics.update(
            {
                "bubble/fallback_idle_events": len(emitted_worker_ids),
                "bubble/fallback_idle_replica_count": len(replica_ranks),
            }
        )
        return metrics

    def _speco_reclaim_rollout_idle_workers(
        self,
        *,
        boundary: str,
        on_completed=None,
        force_drain: bool = False,
    ) -> dict[str, Any]:
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        runtime_state = self._speco_get_drafter_runtime_state()
        active_plan = runtime_state.active_plan
        if active_plan is None or not active_plan.target_worker_ids:
            return {}
        config = self._speco_drafter_schedule_config()
        scheduler = self._speco_get_drafter_scheduler()
        should_drain = force_drain or config.idle_worker_drain_before_next_rollout
        drain_started = time.perf_counter() if should_drain else None
        scheduler.request_reclaim(active_plan.target_worker_ids)
        metrics: dict[str, Any] = {"bubble/reclaim_requested": 1}
        if not should_drain:
            logger.info(
                "[BubbleTime] reclaim requested without drain: boundary=%s "
                "plan_id=%s workers=%s",
                boundary,
                active_plan.plan_id,
                active_plan.target_worker_ids,
            )
            return metrics

        # Reclaim is cooperative: a worker finishes its in-flight batch and
        # cleans up before a full-group PPO or rollout phase starts. This keeps
        # Bubble work inside the confirmed rollout-idle window.
        assert drain_started is not None
        completed_plan, completed_outcome = self._speco_wait_pending_drafter_training()
        metrics["bubble/reclaim_drained"] = int(completed_outcome is not None)
        if completed_outcome is not None:
            self._speco_last_known_trainable_batches = None
            metrics.update(completed_outcome.metrics)
            if completed_outcome.trained:
                if on_completed is not None:
                    completion_metrics = on_completed(
                        completed_plan,
                        completed_outcome,
                    )
                    if completion_metrics:
                        metrics.update(completion_metrics)
                else:
                    # Before the next generation, the preceding actor update
                    # has completed and publication is rollout-safe.
                    metrics.update(
                        self._speco_publish_drafter_weights(
                            completed_outcome.trained,
                            completed_plan,
                        )
                    )
                    metrics.update(self._speco_publish_boundary())
        # The worker is reusable only after both cooperative training cleanup
        # and a required safe-point publication have completed.  Learn that
        # whole critical-path cost for the next admission deadline.
        drain_elapsed = time.perf_counter() - drain_started
        scheduler.record_reclaim_elapsed(drain_elapsed)
        metrics["timing_s/drafter_reclaim_wait"] = drain_elapsed
        metrics[f"timing_s/drafter_critical_path_{boundary}"] = drain_elapsed
        logger.warning(
            "[BubbleTime] reclaim drain: boundary=%s plan_id=%s workers=%s "
            "completed=%s elapsed_s=%.4f",
            boundary,
            active_plan.plan_id,
            active_plan.target_worker_ids,
            completed_outcome is not None,
            drain_elapsed,
        )
        return metrics

    def _speco_reclaim_rollout_idle_workers_before_generation(self) -> dict[str, Any]:
        return self._speco_reclaim_rollout_idle_workers(
            boundary="before_generation",
        )

    def _speco_reclaim_rollout_idle_workers_before_post_rollout(
        self,
        *,
        on_completed=None,
    ) -> dict[str, Any]:
        return self._speco_reclaim_rollout_idle_workers(
            boundary="before_post_rollout",
            on_completed=on_completed,
            force_drain=True,
        )

    def _speco_drafter_training_mode(self) -> str:
        training_cfg = self._speco_drafter_training_config()
        return str(training_cfg.get("mode", "online") or "online").strip().lower()

    def _speco_drafter_schedule_context(self) -> DrafterScheduleContext:
        return DrafterScheduleContext(
            global_step=self.global_steps,
            training_mode=self._speco_drafter_training_mode(),
            collected_samples_this_step=int(
                getattr(self, "_speco_last_collected_samples", 0) or 0
            ),
            oldlogprob_collection_requested=(
                self._speco_oldlogprob_collection_requested()
            ),
            data_status=None,
            pending_training_count=int(
                self._speco_get_drafter_runtime_state().status
                in {DrafterRuntimeStatus.SUBMITTED, DrafterRuntimeStatus.RUNNING}
            ),
        )

    def _speco_on_before_actor_update(
        self,
        *,
        allow_quota_topup: bool = True,
    ):
        return self._speco_get_drafter_scheduler().on_before_actor_update(
            BeforeActorUpdateContext(
                schedule_context=self._speco_drafter_schedule_context(),
                config=self._speco_drafter_schedule_config(),
                allow_quota_topup=allow_quota_topup,
            )
        )

    @staticmethod
    def _speco_log_drafter_training_plan(
        plan: TrainingPlan,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        metrics = metrics or {}
        log = logger.warning if plan.launch else logger.info
        log(
            "[DrafterScheduler] step=%s strategy=%s launch=%s reason=%s "
            "interval_matched=%s max_batches=%s publish_after_success=%s "
            "training_group=%s target_worker_ids=%s deadline_ts=%s "
            "window_s=%s usable_window_s=%s window_batches=%s "
            "batch_estimate_s=%s trainable_batches_for_group=%s",
            plan.source_global_step,
            plan.execution_strategy.value,
            plan.launch,
            plan.reason,
            plan.interval_matched,
            plan.max_batches,
            plan.publish_after_success,
            plan.training_group_id,
            plan.target_worker_ids,
            plan.deadline_ts,
            plan.idle_window_sec,
            plan.idle_usable_window_sec,
            plan.idle_window_batches,
            plan.idle_batch_estimate_sec,
            plan.idle_trainable_batches,
        )

    def _speco_set_drafter_global_step(self):
        return self._ray_get_if_needed(self.speco_set_global_step(self.global_steps))

    def _speco_prepare_drafter_training_rpc(
        self, training_plan: TrainingPlan
    ) -> dict[str, Any]:
        self._speco_set_drafter_global_step()
        metrics, pending = self._speco_start_target_lm_head_weight_sync(training_plan)
        self._pending_target_lm_head_sync = pending
        return metrics

    def _speco_execute_collection(
        self,
        plan: CollectionPlan,
        payload: CollectionPayload,
    ) -> CollectionOutcome:
        scheduler = self._speco_get_drafter_scheduler()
        outcome = scheduler.on_collection_ready(plan, payload)
        scheduler.record_collection_outcome(
            plan,
            outcome,
            self._speco_drafter_schedule_config(),
        )
        self._speco_last_collection_outcome = outcome
        print(
            "[BubbleTime] collection_completed: "
            f"step={plan.source_global_step} source={plan.source.value} "
            f"reason={outcome.reason} raw_samples={payload.raw_samples} "
            f"collected_samples={outcome.collected_samples} "
            f"elapsed_s={outcome.elapsed_sec:.4f}",
            flush=True,
        )
        if plan.source is DrafterCollectionSource.OLD_LOGPROB:
            self._speco_last_oldlogprob_collect_rpc_elapsed_sec = outcome.elapsed_sec
        return outcome

    def _speco_oldlogprob_collection_requested(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return bool(training_cfg.get("collect_hidden_states_from_old_logprob", False))

    def _speco_oldlogprob_collection_enabled(self) -> bool:
        if (
            not self._speco_online_enabled()
            or not self._speco_oldlogprob_collection_requested()
        ):
            return False
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("collect_hidden_states_from_sgl", False)):
            raise ValueError(
                "SPECO old-logprob hidden collection requires "
                "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false"
            )
        if bool(training_cfg.get("use_logits", False)):
            raise ValueError(
                "SPECO old-logprob hidden collection currently supports use_logits=false only"
            )
        strategy = str(
            _get_nested(self.config, ("actor_rollout_ref", "actor", "strategy"), "")
            or ""
        ).lower()
        if strategy not in {"fsdp", "fsdp2", "veomni"}:
            raise ValueError(
                "SPECO old-logprob hidden collection supports "
                "actor.strategy=fsdp/fsdp2/veomni, "
                f"got {strategy!r}"
            )
        capture_impl = str(
            training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook")
            or "forward_hook"
        )
        if capture_impl not in {"forward_hook", "output_hidden_states"}:
            raise ValueError(
                f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}"
            )
        return True

    def _speco_oldlogprob_entropy_config_value(self):
        training_cfg = self._speco_drafter_training_config()
        value = training_cfg.get("old_logprob_calculate_entropy", None)
        if value is None:
            value = _get_nested(
                self.config, ("actor_rollout_ref", "actor", "calculate_entropy"), None
            )
        return value

    @staticmethod
    def _speco_bool_config(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _speco_oldlogprob_entropy_hook_enabled(self) -> bool:
        value = self._speco_oldlogprob_entropy_config_value()
        if value is None and not self.is_drafter_rollout_enabled(self.config):
            return False
        return not self._speco_oldlogprob_calculate_entropy()

    def _speco_oldlogprob_calculate_entropy(self) -> bool:
        value = self._speco_oldlogprob_entropy_config_value()
        if value is None:
            value = False
        return self._speco_bool_config(value)

    def _speco_oldlogprob_hidden_capture_impl(self) -> str:
        training_cfg = self._speco_drafter_training_config()
        return str(
            training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook")
            or "forward_hook"
        )

    def _speco_oldlogprob_hidden_layout(self) -> str:
        drafter_cfg = self._speco_drafter_config()
        algorithm = _get_nested(drafter_cfg, ("speculative_algorithm",), "")
        return resolve_drafter_hidden_states_layout(
            algorithm, self._speco_drafter_training_config()
        )

    def _speco_oldlogprob_aux_layer_id_space(self) -> str:
        algorithm = (
            str(
                _get_nested(
                    self._speco_drafter_config(),
                    ("speculative_algorithm",),
                    "EAGLE3",
                )
                or "EAGLE3"
            )
            .strip()
            .upper()
        )
        # EAGLE3 config IDs are the same output IDs passed to vLLM serve.
        # DFlash-family and EAGLE1/2 IDs remain decoder-layer indices.
        return "output" if algorithm == "EAGLE3" else "decoder"

    @staticmethod
    def _speco_oldlogprob_window_train_rows(training_cfg) -> int:
        window_rows = training_cfg.get("hidden_state_window_tokens_per_sample")
        if window_rows is None:
            window_rows = training_cfg.get("hidden_state_window_min_rows", 64)
        return int(window_rows or 0)

    @staticmethod
    def _speco_oldlogprob_window_mode(training_cfg) -> str:
        mode = (
            str(training_cfg.get("hidden_state_window_mode", "front") or "front")
            .strip()
            .lower()
        )
        if mode not in {"front", "random"}:
            return "front"
        return mode

    @staticmethod
    def _speco_load_model_config(model_path: Any) -> dict[str, Any] | None:
        if not model_path:
            return None
        config_path = os.path.join(str(model_path), "config.json")
        try:
            with open(config_path, encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return None
        return config if isinstance(config, dict) else None

    @staticmethod
    def _speco_num_hidden_layers_from_config(config) -> int | None:
        candidates = (
            ("num_hidden_layers",),
            ("text_config", "num_hidden_layers"),
            ("model", "num_hidden_layers"),
            ("n_layer",),
            ("num_layers",),
        )
        for path in candidates:
            value = _get_nested(config, path, None)
            if value is not None:
                return int(value)
        return None

    def _speco_target_num_hidden_layers(self) -> int | None:
        target_model_cfg = _get_nested(
            self.config, ("actor_rollout_ref", "model"), None
        )
        num_layers = self._speco_num_hidden_layers_from_config(target_model_cfg)
        if num_layers is not None:
            return num_layers
        target_model_path = _get_nested(target_model_cfg, ("path",), None)
        target_config = self._speco_load_model_config(target_model_path)
        return self._speco_num_hidden_layers_from_config(target_config)

    def _speco_validate_sglang_aux_last_layer_norm(self) -> None:
        """Fail closed if SGLang collection would capture the last aux layer pre-norm.

        SGLang's aux/context capture skips the target's final norm, so a last-layer
        (or ``-1``) ``target_layer_id`` diverges from the offline / old-logprob
        (post-norm / embedding) semantics; see ``assert_sglang_aux_last_layer_norm_safe``.
        Best-effort: skips silently when the layer ids or target depth cannot be resolved.
        """
        training_cfg = self._speco_drafter_training_config()
        if not bool(training_cfg.get("collect_hidden_states_from_sgl", False)):
            return
        drafter_cfg = self._speco_drafter_config()
        model_configs = []
        for path_key in ("model_path", "checkpoint_path"):
            model_config = self._speco_load_model_config(
                _get_nested(drafter_cfg, (path_key,), None)
            )
            if model_config is not None:
                model_configs.append(model_config)
        num_hidden_layers = self._speco_target_num_hidden_layers()
        try:
            layer_ids = resolve_oldlogprob_aux_layer_ids(
                drafter_cfg,
                target_num_hidden_layers=num_hidden_layers,
                model_configs=model_configs,
            )
        except Exception:  # noqa: BLE001 -- best-effort guard, never masks the real resolve path
            return
        assert_sglang_aux_last_layer_norm_safe(
            layer_ids,
            num_hidden_layers,
            collect_from_sgl=True,
            allow_prenorm_last=bool(
                training_cfg.get("allow_sglang_prenorm_last_layer", False)
            ),
        )

    def _speco_oldlogprob_aux_layer_ids(self) -> list[int]:
        drafter_cfg = self._speco_drafter_config()
        model_configs = []
        for path_key in ("model_path", "checkpoint_path"):
            model_config = self._speco_load_model_config(
                _get_nested(drafter_cfg, (path_key,), None)
            )
            if model_config is not None:
                model_configs.append(model_config)

        num_hidden_layers = self._speco_target_num_hidden_layers()
        layer_ids = resolve_oldlogprob_aux_layer_ids(
            drafter_cfg,
            target_num_hidden_layers=num_hidden_layers,
            model_configs=model_configs,
        )
        if layer_ids is None:
            raise RuntimeError(
                "SPECO old-logprob hidden collection requires explicit DFlash target_layer_ids, "
                "EAGLE3 eagle_aux_hidden_state_layer_ids/target_hidden_layer_ids in drafter config or checkpoint, "
                "or a readable target model config at actor_rollout_ref.model.path/config.json with "
                "num_hidden_layers. Refusing to guess aux hidden layers."
            )
        return layer_ids

    @staticmethod
    def _speco_hash_fraction(key: str) -> float:
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    @staticmethod
    def _speco_hash_int(key: str, inclusive_max: int) -> int:
        if inclusive_max <= 0:
            return 0
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % (
            inclusive_max + 1
        )

    def _speco_build_oldlogprob_collect_plan(
        self, batch: DataProto
    ) -> dict[str, Any] | None:
        if not self._speco_oldlogprob_collection_enabled():
            return None
        collection_plan = self._speco_plan_drafter_collection(
            DrafterCollectionSource.OLD_LOGPROB
        )
        self._speco_log_drafter_collection_plan(collection_plan)
        if not collection_plan.collect:
            return None
        training_cfg = self._speco_drafter_training_config()
        sample_rate = collection_plan.sample_rate
        window_mode = self._speco_oldlogprob_window_mode(training_cfg)

        batch_tensors = batch.batch
        required_keys = ("prompts", "responses", "attention_mask")
        if any(key not in batch_tensors for key in required_keys):
            return None
        prompts = batch_tensors["prompts"]
        attention_mask = batch_tensors["attention_mask"]
        response_mask = batch_tensors.get("response_mask", None)
        batch_size = int(prompts.size(0))
        prompt_width = int(prompts.size(1))

        train_rows = self._speco_oldlogprob_window_train_rows(training_cfg)
        if train_rows <= 0:
            return None
        hidden_rows = train_rows + 1
        collect_mask = torch.zeros(batch_size, dtype=torch.bool)
        hidden_positions = torch.zeros(batch_size, hidden_rows, dtype=torch.long)
        hidden_position_mask = torch.zeros(batch_size, hidden_rows, dtype=torch.bool)
        owner_rank = torch.zeros(batch_size, dtype=torch.long)

        owner_count = self._speco_owner_bucket_count()
        if owner_count is None:
            owner_count = 1
        owner_count = max(int(owner_count), 1)
        bubble_owner_ranks = self._speco_bubble_collection_owner_ranks()
        eligible_owner_ranks = (
            tuple(owner for owner in bubble_owner_ranks if 0 <= owner < owner_count)
            if bubble_owner_ranks
            else tuple(range(owner_count))
        )
        if not eligible_owner_ranks:
            eligible_owner_ranks = tuple(range(owner_count))
        max_per_owner = collection_plan.max_samples_per_replica
        max_per_owner = max_per_owner if max_per_owner is not None else batch_size
        max_per_owner = max(max_per_owner, 0)
        if self._speco_rollout_idle_worker_enabled():
            # Single-writer Bubble routing concentrates samples that Sync
            # would have kept on multiple owners.  Do not let the legacy
            # per-owner default silently reduce its plan-local dataset.
            writer_minimum = max(
                int(
                    self._speco_drafter_schedule_config().idle_worker_min_collect_samples_per_writer
                ),
                0,
            )
            if max_per_owner < writer_minimum:
                print(
                    "[BubbleTime] collection_writer_capacity_raised: "
                    f"configured_per_owner={max_per_owner} writer_minimum={writer_minimum}",
                    flush=True,
                )
                max_per_owner = writer_minimum
        max_tokens_per_owner = collection_plan.max_tokens_per_replica
        if max_tokens_per_owner is not None:
            max_tokens_per_owner = max(max_tokens_per_owner, 0)
            if self._speco_rollout_idle_worker_enabled():
                # Old-logprob capture needs one extra hidden row for shifted
                # supervision.  A common ``N * window_rows`` token budget would
                # therefore admit only N-1 samples after Bubble concentrates
                # the Sync owners onto its single writer.  Correct that exact
                # off-by-one shape without overriding deliberately smaller
                # token budgets.
                configured_window_budget = max_per_owner * train_rows
                required_window_budget = max_per_owner * hidden_rows
                if (
                    max_tokens_per_owner == configured_window_budget
                    and max_tokens_per_owner < required_window_budget
                ):
                    print(
                        "[BubbleTime] collection_writer_token_capacity_raised: "
                        f"configured_per_owner={max_tokens_per_owner} "
                        f"required_per_owner={required_window_budget} "
                        f"samples={max_per_owner} hidden_rows={hidden_rows}",
                        flush=True,
                    )
                    max_tokens_per_owner = required_window_budget
        owner_counts = [0 for _ in range(owner_count)]
        owner_token_counts = [0 for _ in range(owner_count)]
        seed_by_step = bool(training_cfg.get("hidden_state_random_seed_by_step", True))
        step_key = self.global_steps if seed_by_step else "request"

        prompt_lens: list[int] = []
        response_lens: list[int] = []
        candidate_count = 0
        selected_count = 0
        for batch_idx in range(batch_size):
            prompt_len = int(
                attention_mask[batch_idx, :prompt_width].detach().sum().item()
            )
            if response_mask is not None:
                response_len = int(response_mask[batch_idx].detach().sum().item())
            else:
                response_len = int(
                    attention_mask[batch_idx, prompt_width:].detach().sum().item()
                )
            prompt_lens.append(prompt_len)
            response_lens.append(response_len)
            if prompt_len <= 0 or response_len < hidden_rows:
                continue
            candidate_count += 1
            sample_key = f"{step_key}:{batch_idx}:{prompt_len}:{response_len}"
            if (
                sample_rate < 1.0
                and self._speco_hash_fraction(sample_key) >= sample_rate
            ):
                continue
            owner = eligible_owner_ranks[selected_count % len(eligible_owner_ranks)]
            if owner_counts[owner] >= max_per_owner:
                continue
            if (
                max_tokens_per_owner is not None
                and owner_token_counts[owner] + hidden_rows > max_tokens_per_owner
            ):
                continue
            max_start_offset = max(response_len - hidden_rows, 0)
            if window_mode == "random":
                random_offset = self._speco_hash_int(
                    f"{sample_key}:window", max_start_offset
                )
            else:
                random_offset = 0
            start = max(prompt_len - 1, 0) + random_offset
            positions = torch.arange(start, start + hidden_rows, dtype=torch.long)
            collect_mask[batch_idx] = True
            hidden_positions[batch_idx, :] = positions
            hidden_position_mask[batch_idx, :] = True
            owner_rank[batch_idx] = owner
            owner_counts[owner] += 1
            owner_token_counts[owner] += hidden_rows
            selected_count += 1

        self._speco_last_raw_drafter_samples = candidate_count
        self._speco_last_oldlogprob_candidate_samples = candidate_count
        self._speco_last_oldlogprob_planned_samples = selected_count
        if selected_count <= 0:
            return None
        return {
            "collection_plan": collection_plan,
            "collect_mask": collect_mask,
            "hidden_positions": hidden_positions,
            "hidden_position_mask": hidden_position_mask,
            "owner_rank": owner_rank,
            "prompt_lens": prompt_lens,
            "response_lens": response_lens,
            "hidden_rows": hidden_rows,
            "owner_count": owner_count,
            "eligible_owner_ranks": eligible_owner_ranks,
            "selected_count": selected_count,
            "candidate_count": candidate_count,
            "owner_token_counts": owner_token_counts,
            "window_mode": window_mode,
        }

    @staticmethod
    def _speco_tensor_rows(tensor: torch.Tensor | None) -> list[torch.Tensor]:
        if tensor is None:
            return []
        if torch.is_tensor(tensor) and tensor.is_nested:
            return list(tensor.unbind())
        if torch.is_tensor(tensor):
            return [row for row in tensor]
        return []

    @staticmethod
    def _speco_sequence_item(value: Any, index: int):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[index] if 0 <= index < len(value) else None
        return None

    @staticmethod
    def _speco_flatten_non_tensor_rows(value: Any):
        if not isinstance(value, (list, tuple)):
            return value
        if not value or not all(isinstance(item, (list, tuple)) for item in value):
            return value
        flattened = []
        for item in value:
            flattened.extend(item)
        return flattened

    @staticmethod
    def _speco_sum_timing_rows(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            return None
        if torch.is_tensor(tensor) and tensor.is_nested:
            rows = [
                row.reshape(-1).float() for row in tensor.unbind() if row.numel() > 0
            ]
            if not rows:
                return None
            width = min(int(row.numel()) for row in rows)
            return torch.stack([row[:width] for row in rows], dim=0).sum(dim=0).cpu()
        if tensor.numel() == 0:
            return None
        if tensor.dim() == 1:
            return tensor.float().cpu()
        return tensor.reshape(-1, tensor.shape[-1]).float().sum(dim=0).cpu()

    def _speco_collect_oldlogprob_features(
        self,
        batch: DataProto,
        collect_plan: dict[str, Any] | None,
        output: Any,
    ) -> int:
        if not collect_plan:
            return 0
        hidden_states = tu.get(output, OLD_LOGPROB_HIDDEN_STATES_KEY)
        hidden_refs = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_REFS_KEY)
        )
        hidden_ref_meta = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_REF_META_KEY)
        )
        chunk_refs = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY)
        )
        chunk_meta = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_META_KEY)
        )
        if hidden_states is None and hidden_refs is None and chunk_refs is None:
            return 0
        hidden_rows = self._speco_tensor_rows(hidden_states)
        if not hidden_rows and not hidden_refs and not chunk_refs:
            return 0
        timing = self._speco_sum_timing_rows(tu.get(output, OLD_LOGPROB_TIMING_KEY))
        if timing is not None and int(timing.numel()) >= 2:
            self._speco_last_oldlogprob_select_elapsed_sec = (
                float(timing[0].item()) / 1_000_000.0
            )
            self._speco_last_oldlogprob_sp_merge_elapsed_sec = (
                float(timing[1].item()) / 1_000_000.0
            )
            if int(timing.numel()) >= 5:
                self._speco_last_oldlogprob_concat_elapsed_sec = (
                    float(timing[2].item()) / 1_000_000.0
                )
                self._speco_last_oldlogprob_cpu_copy_elapsed_sec = (
                    float(timing[3].item()) / 1_000_000.0
                )
                self._speco_last_oldlogprob_ray_put_elapsed_sec = (
                    float(timing[4].item()) / 1_000_000.0
                )

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask_tensor = batch.batch.get("response_mask", None)
        collect_mask = collect_plan["collect_mask"]
        hidden_positions = collect_plan["hidden_positions"]
        owner_rank = collect_plan["owner_rank"]
        prompt_lens = collect_plan["prompt_lens"]
        response_lens = collect_plan["response_lens"]
        samples: list[dict[str, Any]] = []
        owners: list[int] = []
        collected_rows = 0
        payload_bytes = 0
        sample_ref_chunks: dict[int, list[dict[str, Any]]] = {}
        if isinstance(chunk_refs, (list, tuple)) and isinstance(
            chunk_meta, (list, tuple)
        ):
            for chunk_index, (chunk_ref, chunk_info) in enumerate(
                zip(chunk_refs, chunk_meta, strict=False)
            ):
                if chunk_ref is None or not isinstance(chunk_info, dict):
                    continue
                sample_indices = chunk_info.get("sample_indices") or []
                starts = chunk_info.get("starts") or []
                lengths = chunk_info.get("lengths") or []
                row_indices_payload = chunk_info.get("row_indices") or []
                for item_idx, batch_idx in enumerate(sample_indices):
                    try:
                        batch_idx = int(batch_idx)
                    except (TypeError, ValueError):
                        continue
                    if batch_idx < 0:
                        continue
                    start = int(starts[item_idx]) if item_idx < len(starts) else 0
                    length = int(lengths[item_idx]) if item_idx < len(lengths) else 0
                    row_indices = (
                        row_indices_payload[item_idx]
                        if item_idx < len(row_indices_payload)
                        else None
                    )
                    sample_ref_chunks.setdefault(batch_idx, []).append(
                        {
                            "ref": chunk_ref,
                            "chunk_index": int(chunk_index),
                            "chunk_start": start,
                            "chunk_length": length,
                            "chunk_row_indices": row_indices,
                            "dtype": chunk_info.get("dtype"),
                            "shape": chunk_info.get("shape"),
                        }
                    )

        item_count = max(
            int(collect_mask.numel()),
            len(hidden_rows),
            len(hidden_refs) if isinstance(hidden_refs, (list, tuple)) else 0,
            max(sample_ref_chunks.keys(), default=-1) + 1,
        )
        for batch_idx in range(item_count):
            if batch_idx >= int(collect_mask.numel()) or not bool(
                collect_mask[batch_idx].item()
            ):
                continue
            prompt_len = int(prompt_lens[batch_idx])
            response_len = int(response_lens[batch_idx])
            valid_positions = hidden_positions[batch_idx].reshape(-1)
            valid_rows = int(valid_positions.numel())
            if valid_rows <= 0:
                continue
            hidden_ref = self._speco_sequence_item(hidden_refs, batch_idx)
            ref_meta = self._speco_sequence_item(hidden_ref_meta, batch_idx)
            ref_chunks = sample_ref_chunks.get(batch_idx)
            hidden = hidden_rows[batch_idx] if batch_idx < len(hidden_rows) else None
            if ref_chunks:
                collected_rows += sum(
                    _speco_ref_meta_row_count(chunk, 0) for chunk in ref_chunks
                )
                payload_bytes += sum(
                    int(chunk.get("chunk_length", 0) or 0)
                    * int((chunk.get("shape") or [0, 0])[-1] or 0)
                    * 2
                    for chunk in ref_chunks
                )
            elif hidden_ref is None:
                if hidden is None:
                    continue
                hidden = hidden[:valid_rows].contiguous()
                if hidden.numel() == 0:
                    continue
                collected_rows += int(hidden.size(0))
                payload_bytes += int(hidden.numel()) * int(hidden.element_size())
            else:
                collected_rows += _speco_ref_meta_rows(ref_meta) or valid_rows
                payload_bytes += _speco_ref_meta_nbytes(ref_meta)
            owner = int(owner_rank[batch_idx].item())
            prompt_mask = attention_mask[batch_idx, : prompts.size(1)].bool()
            if response_mask_tensor is not None:
                response_mask = response_mask_tensor[batch_idx].bool()
            else:
                response_mask = attention_mask[
                    batch_idx, prompts.size(1) : prompts.size(1) + responses.size(1)
                ].bool()
            prompt_ids = prompts[batch_idx][prompt_mask].detach().cpu()
            response_ids = responses[batch_idx][response_mask].detach().cpu()
            prompt_ids = prompt_ids[:prompt_len]
            response_ids = response_ids[:response_len]
            sample_input_ids = torch.cat([prompt_ids, response_ids], dim=0)
            sample = {
                "input_ids": sample_input_ids.unsqueeze(0),
                "prompts": prompt_ids.unsqueeze(0),
                "responses": response_ids.unsqueeze(0),
                "hidden_positions": valid_positions.detach().cpu().unsqueeze(0),
                "hidden_states_layout": self._speco_oldlogprob_hidden_layout(),
                "hidden_position_start": int(valid_positions[0].item()),
                "hidden_position_end": int(valid_positions[-1].item()) + 1,
                "global_step": self.global_steps,
                "replica_rank": owner,
                "source_replica_rank": owner,
                "_speco_global_sample_id": (
                    f"oldlogprob:{self.global_steps}:{batch_idx}:{prompt_len}:"
                    f"{response_len}:{int(valid_positions[0].item())}"
                ),
            }
            if ref_chunks:
                sample["hidden_states_ref_chunks"] = ref_chunks
            elif hidden_ref is None:
                hidden = cast(torch.Tensor, hidden)
                sample["hidden_states"] = hidden.detach().cpu().unsqueeze(0)
            else:
                sample["hidden_states_ref"] = hidden_ref
                sample["hidden_states_ref_meta"] = ref_meta
            samples.append(sample)
            owners.append(owner)

        collected = len(samples)
        if collected <= 0:
            return 0
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        payload = self._speco_get_drafter_scheduler().prepare_collection_payload(
            source=DrafterCollectionSource.OLD_LOGPROB,
            samples=samples,
            owners=owners,
            owner_count=int(collect_plan["owner_count"]),
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=int(collect_plan.get("candidate_count", collected)),
            collection_id=collect_plan["collection_plan"].collection_id,
        )
        outcome = self._speco_execute_collection(
            collect_plan["collection_plan"],
            payload,
        )
        self._speco_last_collected_samples = outcome.collected_samples
        self._speco_last_oldlogprob_collected_samples = outcome.collected_samples
        self._speco_last_oldlogprob_collected_rows = collected_rows
        self._speco_last_oldlogprob_payload_mib = payload_bytes / float(1024 * 1024)
        return outcome.collected_samples

    def _speco_num_rollout_replicas(self, samples: list[dict]) -> int:
        sample_max = (
            max((int(sample.get("replica_rank", 0)) for sample in samples), default=0)
            + 1
        )
        rollout_cfg = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
        rollout_dp = int(_get_nested(rollout_cfg, ("data_parallel_size",), 1) or 1)
        return max(sample_max, rollout_dp, 1)

    def _speco_sglang_collection_owner_count(
        self,
        samples: list[dict],
        owners: list[int] | tuple[int, ...] | None = None,
    ) -> int:
        """Return the owner bucket count for SGLang sample collection.

        Sync SGLang collection buckets by ``replica_rank`` and should keep the
        legacy replica-derived count.  Bubble Time may route a sparse direct
        event batch (for example, one sample from replica 0) to the current
        writer owner (for example, owner 1).  In that routed case, the payload
        must expose enough owner buckets for the explicit owner IDs.
        """

        owner_count = self._speco_num_rollout_replicas(samples)
        if owners:
            owner_count = max(owner_count, max(int(owner) for owner in owners) + 1)
            routed_owner_count = self._speco_owner_bucket_count()
            if routed_owner_count is not None:
                owner_count = max(owner_count, int(routed_owner_count))
        return max(int(owner_count), 1)

    def _speco_collect_generation_samples(self, gen_batch_output: Any) -> int:
        self._speco_last_raw_drafter_samples = 0
        direct_collected = int(getattr(self, "_speco_direct_collected_samples", 0) or 0)
        self._speco_last_collected_samples = direct_collected
        collection_plan = self._speco_plan_drafter_collection(
            DrafterCollectionSource.SGLANG
        )
        self._speco_log_drafter_collection_plan(collection_plan)
        self._speco_last_collect_interval_matched = int(
            collection_plan.collect_interval_matched
        )
        if not self._speco_online_enabled():
            return 0
        all_samples = pop_drafter_samples(gen_batch_output)
        self._speco_last_raw_drafter_samples = len(all_samples)
        committed_ids = cast(set[str], getattr(self, "_speco_direct_sample_ids", set()))
        samples = [
            sample
            for sample in all_samples
            if str(sample.get("_speco_sample_id", "") or "") not in committed_ids
        ]
        if direct_collected:
            logger.warning(
                "[BubbleTime] direct_collection_summary: raw=%s committed=%s "
                "fallback_samples=%s",
                len(all_samples),
                direct_collected,
                len(samples),
            )
        if not samples:
            return direct_collected
        if not collection_plan.collect:
            return 0

        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        bubble_owner_ranks = self._speco_bubble_collection_owner_ranks()
        routed_owners = None
        if bubble_owner_ranks:
            routed_owners = [
                bubble_owner_ranks[index % len(bubble_owner_ranks)]
                for index in range(len(samples))
            ]
            for index, sample in enumerate(samples):
                sample.setdefault("source_replica_rank", sample.get("replica_rank"))
                sample.setdefault(
                    "_speco_global_sample_id",
                    str(sample.get("_speco_sample_id", "") or "")
                    or (
                        f"sglang:{self.global_steps}:"
                        f"{sample.get('replica_rank', 'unknown')}:{index}"
                    ),
                )
        payload = self._speco_get_drafter_scheduler().prepare_collection_payload(
            source=DrafterCollectionSource.SGLANG,
            samples=samples,
            owners=routed_owners,
            owner_count=self._speco_sglang_collection_owner_count(
                samples,
                routed_owners,
            ),
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=len(samples),
            collection_id=collection_plan.collection_id,
        )

        outcome = self._speco_execute_collection(
            collection_plan,
            payload,
        )
        total_collected = direct_collected + outcome.collected_samples
        self._speco_last_collected_samples = total_collected
        return total_collected

    def _speco_owner_route_mapping(self):
        worker_group = self.drafter_wg
        if worker_group is None:
            return None
        mapping = None
        dispatch_info = getattr(worker_group, "_dispatch_info", None)
        if isinstance(dispatch_info, dict):
            mapping = dispatch_info.get("drafter_owner_route")
        if mapping is None and hasattr(worker_group, "_query_dispatch_info"):
            mapping = worker_group._query_dispatch_info("drafter_owner_route")
            if isinstance(dispatch_info, dict):
                dispatch_info["drafter_owner_route"] = mapping
        return mapping

    def _speco_owner_route_collect_mask(self):
        worker_group = self.drafter_wg
        if worker_group is None:
            return None
        collect_mask = None
        collect_info = getattr(worker_group, "_collect_info", None)
        if isinstance(collect_info, dict):
            collect_mask = collect_info.get("drafter_owner_route")
        if collect_mask is None and hasattr(worker_group, "_query_collect_info"):
            collect_mask = worker_group._query_collect_info("drafter_owner_route")
            if isinstance(collect_info, dict):
                collect_info["drafter_owner_route"] = collect_mask
        return collect_mask

    def _speco_dispatch_bucket_count(self) -> int | None:
        mapping = self._speco_owner_route_mapping()
        if not mapping:
            return None
        return max(int(dp_rank) for dp_rank in mapping) + 1

    def _speco_owner_bucket_count(self) -> int | None:
        mapping = self._speco_owner_route_mapping()
        if not mapping:
            return None
        collect_mask = self._speco_owner_route_collect_mask()
        if collect_mask and len(collect_mask) == len(mapping):
            owner_ranks = {
                int(dp_rank)
                for dp_rank, is_collect in zip(mapping, collect_mask, strict=False)
                if bool(is_collect)
            }
            if owner_ranks:
                return max(owner_ranks) + 1

        mapping_ranks = {int(dp_rank) for dp_rank in mapping}
        dispatch_bucket_count = max(mapping_ranks) + 1
        return max(dispatch_bucket_count - 1, 1)

    def _speco_bubble_collection_owner_ranks(self) -> tuple[int, ...] | None:
        """Route every Bubble sample to the logical single-writer group.

        The owner-route dispatch already transports each bucket once and fans it
        out to the ranks that share that owner.  Selecting writer owners here
        therefore centralizes availability without copying tensors through the
        scheduler or changing the synchronous collection path.
        """

        if not self._speco_rollout_idle_worker_enabled():
            return None
        writer_group = self._speco_get_drafter_scheduler().target_lm_head_sync_worker_ids(
            full_collective_fallback=bool(
                self._speco_drafter_schedule_config().idle_worker_full_collective_fallback
            )
        )
        mapping = self._speco_owner_route_mapping()
        if not writer_group or not mapping:
            return None
        owner_ranks: list[int] = []
        for worker_id in writer_group:
            try:
                worker_rank = int(worker_id)
            except (TypeError, ValueError):
                continue
            if 0 <= worker_rank < len(mapping):
                owner_ranks.append(int(mapping[worker_rank]))
        normalized = tuple(dict.fromkeys(owner_ranks))
        return normalized or None

    def _speco_get_drafter_target_lm_head_row_selection(self):
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("use_logits", False)):
            return None
        drafter_cfg = self._speco_drafter_config()
        algorithm = str(
            _get_nested(drafter_cfg, ("speculative_algorithm",), "") or ""
        ).upper()
        if (
            algorithm == "DSPARK"
            and float(training_cfg.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0
        ):
            return None
        if not bool(training_cfg.get("target_lm_head_row_restricted_sync", True)):
            return None

        row_infos = (
            self._ray_get_if_needed(self.speco_get_drafter_target_lm_head_row_indices())
            or []
        )
        non_null_infos = [
            info
            for info in row_infos
            if isinstance(info, dict) and info.get("row_indices") is not None
        ]
        if not non_null_infos:
            return None
        source_vocab_sizes = {
            int(info.get("source_vocab_size"))
            for info in non_null_infos
            if info.get("source_vocab_size") is not None
        }
        if len(source_vocab_sizes) > 1:
            raise RuntimeError(
                "Inconsistent SPECO target lm_head source vocab sizes across replicas: "
                f"{sorted(source_vocab_sizes)}"
            )
        source_vocab_size = next(iter(source_vocab_sizes), None)
        row_tensors = []
        for info in non_null_infos:
            row_indices = info.get("row_indices")
            if torch.is_tensor(row_indices):
                rows = row_indices.detach().cpu().long().reshape(-1)
            elif isinstance(row_indices, (list, tuple)):
                rows = torch.tensor([int(idx) for idx in row_indices], dtype=torch.long)
            else:
                continue
            if rows.numel() > 0:
                row_tensors.append(rows)
        if not row_tensors:
            return None
        union_rows = (
            torch.unique(torch.cat(row_tensors), sorted=True)
            .to(dtype=torch.long)
            .contiguous()
        )
        selected_rows = int(union_rows.numel())
        if source_vocab_size is not None and selected_rows >= int(source_vocab_size):
            return None
        return {
            "row_indices": union_rows,
            "source_vocab_size": source_vocab_size,
            "selected_rows": selected_rows,
        }

    def _speco_actor_rollout_method(self, name: str):
        method = getattr(self.actor_rollout_wg, name, None)
        if not callable(method):
            raise RuntimeError(
                f"SPECO online drafter training requires actor_rollout_wg.{name}(). "
                "Attach a rollout worker implementing DraftWeightPublishMixin."
            )
        return method

    def _speco_build_drafter_target_lm_head_sync_args(
        self,
        payload: dict[str, torch.Tensor],
    ) -> tuple[Any, Any, int]:
        worker_group = self.drafter_wg
        if worker_group is None:
            return payload, self.global_steps, 1

        target_sync_mapping = None
        dispatch_info = getattr(worker_group, "_dispatch_info", None)
        if isinstance(dispatch_info, dict):
            target_sync_mapping = dispatch_info.get(_DRAFTER_TARGET_SYNC_MESH)
        if target_sync_mapping is None and hasattr(
            worker_group, "_query_dispatch_info"
        ):
            target_sync_mapping = worker_group._query_dispatch_info(
                _DRAFTER_TARGET_SYNC_MESH
            )
            if isinstance(dispatch_info, dict):
                dispatch_info[_DRAFTER_TARGET_SYNC_MESH] = target_sync_mapping
        if not target_sync_mapping:
            return payload, self.global_steps, 1

        target_sync_bucket_count = (
            max(int(dp_rank) for dp_rank in target_sync_mapping) + 1
        )
        payload_buckets = [payload for _ in range(target_sync_bucket_count)]
        global_step_buckets = [
            self.global_steps for _ in range(target_sync_bucket_count)
        ]
        return payload_buckets, global_step_buckets, target_sync_bucket_count

    def _speco_start_target_lm_head_weight_sync(
        self,
        training_plan: TrainingPlan | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        sync_started = time.perf_counter()
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("use_logits", False)):
            return {"drafter/target_lm_head_synced": 0}, None
        if training_plan is not None and not training_plan.launch:
            return {"drafter/target_lm_head_synced": 0}, None

        if (
            training_plan is None
            and self._speco_rollout_idle_worker_enabled()
            and self._speco_get_drafter_scheduler().idle_writer_migration_blocked()
        ):
            print(
                "[BubbleTime] target_lm_head_prefetch_skipped: "
                "reason=writer_state_migration_required",
                flush=True,
            )
            return {
                "drafter/target_lm_head_synced": 0,
                "bubble/writer_state_migration_required": 1,
            }, None

        if (
            training_plan is None
            and self._speco_rollout_idle_worker_enabled()
            and self._pending_target_lm_head_sync is not None
        ):
            pending = self._pending_target_lm_head_sync
            print(
                "[BubbleTime] target_lm_head_prefetch_skipped: "
                f"reason=previous_prefetch_pending "
                f"target_version={pending.get('target_version')}",
                flush=True,
            )
            return {"drafter/target_lm_head_synced": 0}, None

        target_worker_ids: tuple[str, ...] | None = None
        if self._speco_rollout_idle_worker_enabled():
            if training_plan is not None:
                target_worker_ids = tuple(
                    str(worker_id) for worker_id in training_plan.target_worker_ids
                )
            else:
                target_worker_ids = self._speco_get_drafter_scheduler().target_lm_head_sync_worker_ids(
                    full_collective_fallback=bool(
                        self._speco_drafter_schedule_config().idle_worker_full_collective_fallback
                    )
                )

        if training_plan is None and self._speco_rollout_idle_worker_enabled():
            target_version = int(self.global_steps)
            ready_workers = self._speco_ready_target_lm_head_workers.get(target_version)
            already_ready = (
                target_version in self._speco_ready_target_lm_head_versions
                and (
                    (
                        target_worker_ids is not None
                        and self._speco_target_lm_head_ready_for_workers(
                            target_version,
                            target_worker_ids,
                        )
                    )
                    or (target_worker_ids is None and ready_workers is None)
                )
            )
            if already_ready:
                print(
                    "[BubbleTime] target_lm_head_prefetch_skipped: "
                    f"reason=version_already_ready target_version={target_version} "
                    f"target_workers={target_worker_ids}",
                    flush=True,
                )
                return {"drafter/target_lm_head_synced": 0}, None

        row_selection = self._speco_get_drafter_target_lm_head_row_selection()
        row_indices = (
            row_selection.get("row_indices") if row_selection is not None else None
        )
        selected_rows = (
            int(row_selection.get("selected_rows", 0) or 0)
            if row_selection is not None
            else 0
        )
        source_vocab_size = (
            int(row_selection.get("source_vocab_size", 0) or 0)
            if row_selection is not None
            else 0
        )
        get_actor_lm_head_weight = self._speco_actor_rollout_method(
            "get_actor_lm_head_weight"
        )
        actor_backend = (
            str(
                _get_nested(
                    self.config,
                    ("actor_rollout_ref", "actor", "strategy"),
                    "",
                )
                or ""
            )
            .strip()
            .lower()
        )
        actor_veomni_param_offload = bool(
            _get_nested(
                self.config,
                ("actor_rollout_ref", "actor", "veomni", "param_offload"),
                False,
            )
        )
        keep_actor_model_on_device = bool(
            actor_backend == "veomni"
            and str(self.device_name).lower() == "npu"
            and actor_veomni_param_offload
        )
        fetch_started = time.perf_counter()
        payload_refs = get_actor_lm_head_weight(
            row_indices,
            keep_model_on_device=keep_actor_model_on_device,
        )
        fetch_submit_elapsed = time.perf_counter() - fetch_started
        bubble_prefetch = (
            training_plan is None
            and self._speco_drafter_schedule_config().execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        )
        if bubble_prefetch:
            target_version = int(self.global_steps)
            pending = {
                "stage": "fetch",
                "fetch_refs": payload_refs,
                "sync_started": sync_started,
                "fetch_started": fetch_started,
                "target_version": target_version,
                "global_step": target_version,
                "selected_rows": selected_rows,
                "source_vocab_size": source_vocab_size,
                "target_worker_ids": target_worker_ids,
            }
            return {
                "drafter/target_lm_head_synced": 0,
                "drafter/target_lm_head_selected_rows": selected_rows,
                "drafter/target_lm_head_source_vocab_size": source_vocab_size,
                "drafter/target_lm_head_target_workers": (
                    len(target_worker_ids) if target_worker_ids is not None else 0
                ),
                "timing_s/drafter_target_lm_head_fetch_submit": (fetch_submit_elapsed),
                "timing_s/drafter_target_lm_head_prefetch_submit_critical_path": (
                    time.perf_counter() - sync_started
                ),
            }, pending
        # An ObjectRef is not a snapshot of actor state.  Resolve the actor-side
        # export before PPO mutates the model so the cached version is truly the
        # pre-update lm_head.  Only the subsequent CPU payload dispatch to the
        # drafter workers is allowed to overlap ``original_update_actor``.
        payloads = self._ray_get_if_needed(payload_refs) or []
        fetch_elapsed = time.perf_counter() - fetch_started
        payload = self._first_non_null(payloads)
        if payload is None:
            metrics = {
                "drafter/target_lm_head_synced": 0,
                "drafter/target_lm_head_selected_rows": selected_rows,
                "drafter/target_lm_head_source_vocab_size": source_vocab_size,
                "timing_s/drafter_sync_target_lm_head": time.perf_counter()
                - sync_started,
                "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            }
            if bubble_prefetch:
                metrics.update(
                    {
                        "timing_s/drafter_target_lm_head_fetch_submit": (
                            fetch_submit_elapsed
                        ),
                        "timing_s/drafter_target_lm_head_fetch_critical_path": (
                            fetch_elapsed
                        ),
                        "timing_s/drafter_target_lm_head_prefetch_submit_critical_path": (
                            time.perf_counter() - sync_started
                        ),
                    }
                )
            return metrics, None

        metrics, pending = self._speco_dispatch_target_lm_head_payload(
            payload,
            sync_started=sync_started,
            fetch_elapsed=fetch_elapsed,
            selected_rows=selected_rows,
            source_vocab_size=source_vocab_size,
            target_worker_ids=target_worker_ids,
        )
        if bubble_prefetch:
            metrics.update(
                {
                    "timing_s/drafter_target_lm_head_fetch_submit": (
                        fetch_submit_elapsed
                    ),
                    "timing_s/drafter_target_lm_head_fetch_critical_path": (
                        fetch_elapsed
                    ),
                    "timing_s/drafter_target_lm_head_prefetch_submit_critical_path": (
                        time.perf_counter() - sync_started
                    ),
                }
            )
            if pending is not None:
                pending["target_version"] = int(self.global_steps)
                pending["stage"] = "dispatch"
        if (
            pending is not None
            and bool(pending.get("defer_device_apply", False))
            and pending.get("refs") is not None
        ):
            return metrics, pending
        if pending is not None:
            metrics.update(self._speco_finish_target_lm_head_weight_sync(pending))
        return metrics, None

    @staticmethod
    def _speco_ray_refs_ready(refs: Any) -> bool:
        """Check Ray readiness without putting a ``ray.get`` on the PPO path."""

        object_ref_type = getattr(ray, "ObjectRef", ())
        if not object_ref_type:
            return True

        flattened: list[Any] = []

        def collect(value: Any) -> None:
            if isinstance(value, object_ref_type):
                flattened.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(refs)
        if not flattened:
            return True
        _, pending = ray.wait(flattened, num_returns=len(flattened), timeout=0)
        return not pending

    def _speco_poll_target_lm_head_prefetch(self) -> dict[str, Any]:
        """Advance the Bubble-only target-head state machine without waiting."""

        pending = self._pending_target_lm_head_sync
        if pending is None:
            return {
                "bubble/target_lm_head_prefetch_pending": 0,
                "bubble/target_lm_head_ready_versions": len(
                    self._speco_ready_target_lm_head_versions
                ),
            }
        stage = str(pending.get("stage", "dispatch"))
        refs = pending.get("fetch_refs") if stage == "fetch" else pending.get("refs")
        target_version = pending.get("target_version", pending.get("global_step"))
        if not self._speco_ray_refs_ready(refs):
            if pending.get("last_logged_stage") != stage:
                pending["last_logged_stage"] = stage
                print(
                    "[BubbleTime] target_lm_head_prefetch_pending: "
                    f"target_version={target_version} stage={stage}",
                    flush=True,
                )
            return {
                "bubble/target_lm_head_prefetch_pending": 1,
                "bubble/target_lm_head_ready_versions": len(
                    self._speco_ready_target_lm_head_versions
                ),
            }

        if stage == "fetch":
            payload = self._first_non_null(self._ray_get_if_needed(refs) or [])
            if payload is None:
                self._pending_target_lm_head_sync = None
                print(
                    "[BubbleTime] target_lm_head_prefetch_failed: "
                    f"target_version={target_version} reason=empty_actor_payload",
                    flush=True,
                )
                return {"bubble/target_lm_head_prefetch_failed": 1}
            metrics, dispatch_pending = self._speco_dispatch_target_lm_head_payload(
                payload,
                sync_started=float(pending["sync_started"]),
                fetch_elapsed=time.perf_counter() - float(pending["fetch_started"]),
                selected_rows=int(pending["selected_rows"]),
                source_vocab_size=int(pending["source_vocab_size"]),
                target_worker_ids=pending.get("target_worker_ids"),
            )
            assert dispatch_pending is not None
            dispatch_pending["target_version"] = target_version
            dispatch_pending["stage"] = "dispatch"
            self._pending_target_lm_head_sync = dispatch_pending
            print(
                "[BubbleTime] target_lm_head_prefetch_dispatched: "
                f"target_version={target_version} "
                f"target_workers={dispatch_pending.get('target_worker_ids')}",
                flush=True,
            )
            metrics["bubble/target_lm_head_prefetch_pending"] = 1
            return metrics

        # ``ray.wait`` above proved the apply RPC is complete, so this get is
        # non-blocking and only surfaces a worker-side exception.
        self._ray_get_if_needed(refs)
        self._pending_target_lm_head_sync = None
        evicted_versions = self._speco_mark_target_lm_head_version_ready(
            target_version,
            worker_ids=pending.get("target_worker_ids"),
        )
        print(
            "[BubbleTime] target_lm_head_prefetch_ready: "
            f"target_version={target_version} "
            f"ready_versions={tuple(sorted(self._speco_ready_target_lm_head_versions))} "
            "ready_workers="
            f"{self._speco_ready_target_lm_head_workers.get(int(target_version), None)} "
            f"evicted_versions={evicted_versions}",
            flush=True,
        )
        return {
            "drafter/target_lm_head_synced": 1,
            "bubble/target_lm_head_prefetch_pending": 0,
            "bubble/target_lm_head_ready_versions": len(
                self._speco_ready_target_lm_head_versions
            ),
        }

    def _speco_wait_target_lm_head_prefetch(self) -> dict[str, Any]:
        """Finish both stages of a pending target-head prefetch."""

        metrics: dict[str, Any] = {}
        while self._pending_target_lm_head_sync is not None:
            pending = self._pending_target_lm_head_sync
            stage = str(pending.get("stage", "dispatch"))
            refs = (
                pending.get("fetch_refs") if stage == "fetch" else pending.get("refs")
            )
            self._ray_get_if_needed(refs)
            metrics.update(self._speco_poll_target_lm_head_prefetch())
        return metrics

    def _speco_quiesce_bubble_for_checkpoint(self) -> None:
        """Drain Bubble collectives before entering distributed checkpointing."""

        if not self._speco_rollout_idle_worker_enabled():
            return
        started = time.perf_counter()
        with self._speco_bubble_training_lock():
            scheduler = self._speco_get_drafter_scheduler()
            writer_group = scheduler.idle_checkpoint_group()
            if not writer_group:
                raise RuntimeError("Bubble checkpoint has no complete writer group")
            runtime_state = self._speco_get_drafter_runtime_state()
            active_plan = runtime_state.active_plan
            if active_plan is not None and active_plan.target_worker_ids:
                scheduler.request_reclaim(active_plan.target_worker_ids)
                completed_plan, outcome = self._speco_wait_pending_drafter_training()
                if outcome is not None and outcome.trained:
                    self._speco_publish_drafter_weights(outcome.trained, completed_plan)
                    self._speco_publish_boundary()
            self._speco_wait_target_lm_head_prefetch()
            self._speco_wait_pending_drafter_publish()
            self._speco_wait_pending_drafter_publish_rpc()
        print(
            "[BubbleTime] checkpoint_quiesced: "
            f"step={self.global_steps} writer_group={writer_group} "
            f"elapsed_s={time.perf_counter() - started:.4f}",
            flush=True,
        )

    def _speco_mark_target_lm_head_version_ready(
        self,
        target_version: object,
        *,
        worker_ids: tuple[str, ...] | None = None,
    ) -> tuple[int, ...]:
        """Record only target-head versions that can still exist on workers."""

        try:
            version = int(cast(Any, target_version))
        except (TypeError, ValueError):
            return ()
        self._speco_ready_target_lm_head_versions.add(version)
        if worker_ids is not None:
            self._speco_ready_target_lm_head_workers[version] = tuple(
                str(worker_id) for worker_id in worker_ids
            )
        else:
            self._speco_ready_target_lm_head_workers.pop(version, None)
        evicted_versions: list[int] = []
        while (
            len(self._speco_ready_target_lm_head_versions)
            > _SPECO_TARGET_LM_HEAD_READY_VERSION_LIMIT
        ):
            version = min(self._speco_ready_target_lm_head_versions)
            self._speco_ready_target_lm_head_versions.remove(version)
            self._speco_ready_target_lm_head_workers.pop(version, None)
            evicted_versions.append(version)
        if evicted_versions:
            logger.warning(
                "[BubbleTime] target_lm_head_ready_versions_pruned: "
                "evicted=%s retained=%s limit=%s",
                tuple(evicted_versions),
                tuple(sorted(self._speco_ready_target_lm_head_versions)),
                _SPECO_TARGET_LM_HEAD_READY_VERSION_LIMIT,
            )
        return tuple(evicted_versions)

    def _speco_target_lm_head_ready_for_workers(
        self,
        target_version: object,
        worker_ids: tuple[str, ...],
    ) -> bool:
        try:
            version = int(cast(Any, target_version))
        except (TypeError, ValueError):
            return False
        if version not in self._speco_ready_target_lm_head_versions:
            return False
        ready_workers = self._speco_ready_target_lm_head_workers.get(version)
        if ready_workers is None:
            return True
        return set(str(worker_id) for worker_id in worker_ids).issubset(
            set(ready_workers)
        )

    def _speco_dispatch_target_lm_head_payload(
        self,
        payload: Any,
        *,
        sync_started: float,
        fetch_elapsed: float,
        selected_rows: int,
        source_vocab_size: int,
        target_worker_ids: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Stage a captured actor head on drafter workers without waiting."""

        export_strategy = (
            str(payload.get("export_strategy", "unknown"))
            if isinstance(payload, dict)
            else "unknown"
        )
        # Reconstructing supervision from last hidden states requires a fresh
        # target head for every drafter backend. Stage the payload on CPU while
        # the actor updates, then apply it when the drafter activates.
        defer_device_apply = isinstance(payload, dict)
        if defer_device_apply:
            payload = dict(payload)
            payload["defer_device_apply"] = True
            if target_worker_ids:
                payload["target_worker_ids"] = tuple(
                    str(worker_id) for worker_id in target_worker_ids
                )
        payload_arg, global_step_arg, _ = (
            self._speco_build_drafter_target_lm_head_sync_args(payload)
        )
        dispatch_started = time.perf_counter()
        pending_refs = self.speco_sync_target_lm_head_weight(
            payload_arg, global_step=global_step_arg
        )
        dispatch_elapsed = time.perf_counter() - dispatch_started
        metrics = {
            "drafter/target_lm_head_apply_deferred": int(defer_device_apply),
            "drafter/target_lm_head_selected_rows": selected_rows,
            "drafter/target_lm_head_source_vocab_size": source_vocab_size,
            "drafter/target_lm_head_target_workers": len(target_worker_ids or ()),
            "drafter/target_lm_head_direct_sparse_export": int(
                export_strategy in {"direct_sparse", "veomni_lm_head_sparse"}
            ),
            "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            "timing_s/drafter_sync_target_lm_head_dispatch": dispatch_elapsed,
        }
        pending = {
            "refs": pending_refs,
            "global_step": self.global_steps,
            "dispatch_finished": dispatch_started + dispatch_elapsed,
            "dispatch_elapsed": dispatch_elapsed,
            "pre_dispatch_elapsed": dispatch_started - sync_started,
            "defer_device_apply": defer_device_apply,
            "target_worker_ids": target_worker_ids,
        }
        return metrics, pending

    def _speco_finish_target_lm_head_weight_sync(
        self, pending: dict[str, Any]
    ) -> dict[str, Any]:
        if "fetch_refs" in pending:
            fetch_wait_started = time.perf_counter()
            payloads = self._ray_get_if_needed(pending["fetch_refs"]) or []
            fetch_finished = time.perf_counter()
            fetch_elapsed = fetch_finished - float(pending["fetch_started"])
            payload = self._first_non_null(payloads)
            if payload is None:
                return {
                    "drafter/target_lm_head_synced": 0,
                    "drafter/target_lm_head_selected_rows": int(
                        pending["selected_rows"]
                    ),
                    "drafter/target_lm_head_source_vocab_size": int(
                        pending["source_vocab_size"]
                    ),
                    "timing_s/drafter_sync_target_lm_head": (
                        fetch_finished - float(pending["sync_started"])
                    ),
                    "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
                    "timing_s/drafter_target_lm_head_fetch_async_work": fetch_elapsed,
                    "timing_s/drafter_target_lm_head_fetch_critical_path": (
                        fetch_finished - fetch_wait_started
                    ),
                }
            metrics, dispatch_pending = self._speco_dispatch_target_lm_head_payload(
                payload,
                sync_started=float(pending["sync_started"]),
                fetch_elapsed=fetch_elapsed,
                selected_rows=int(pending["selected_rows"]),
                source_vocab_size=int(pending["source_vocab_size"]),
                target_worker_ids=pending.get("target_worker_ids"),
            )
            metrics["timing_s/drafter_target_lm_head_fetch_async_work"] = fetch_elapsed
            metrics["timing_s/drafter_target_lm_head_fetch_critical_path"] = (
                fetch_finished - fetch_wait_started
            )
            if dispatch_pending is not None:
                metrics.update(
                    self._speco_finish_target_lm_head_weight_sync(dispatch_pending)
                )
            return metrics

        wait_started = time.perf_counter()
        self._ray_get_if_needed(pending.get("refs"))
        finished = time.perf_counter()
        wait_elapsed = finished - wait_started
        dispatch_elapsed = float(pending.get("dispatch_elapsed", 0.0) or 0.0)
        pre_dispatch_elapsed = float(pending.get("pre_dispatch_elapsed", 0.0) or 0.0)
        dispatch_finished = float(
            pending.get("dispatch_finished", wait_started) or wait_started
        )
        overlap_window_elapsed = max(
            wait_started - dispatch_finished,
            0.0,
        )
        critical_path_elapsed = pre_dispatch_elapsed + dispatch_elapsed + wait_elapsed
        logger.warning(
            "[DrafterTarget] cached target lm_head: target_version=%s wait_s=%.4f overlap_window_s=%.4f",
            pending.get("global_step"),
            wait_elapsed,
            overlap_window_elapsed,
        )
        return {
            "drafter/target_lm_head_synced": 1,
            "timing_s/drafter_sync_target_lm_head": critical_path_elapsed,
            "timing_s/drafter_sync_target_lm_head_apply": (
                dispatch_elapsed + wait_elapsed
            ),
            "timing_s/drafter_sync_target_lm_head_wait": wait_elapsed,
            "timing_s/drafter_sync_target_lm_head_overlap_window": (
                overlap_window_elapsed
            ),
        }

    def _speco_sync_target_lm_head_weight(
        self, training_plan: TrainingPlan | None = None
    ) -> dict[str, Any]:
        metrics, pending = self._speco_start_target_lm_head_weight_sync(training_plan)
        if pending is not None:
            metrics.update(self._speco_finish_target_lm_head_weight_sync(pending))
        return metrics

    def _speco_train_drafter(
        self, training_plan: TrainingPlan
    ) -> tuple[bool, dict[str, Any]]:
        runtime_state = self._speco_get_drafter_runtime_state()
        try:
            logger.info(
                "[DrafterRuntime] submitting training: step=%s strategy=%s "
                "plan_id=%s workers=%s max_batches=%s deadline_ts=%s",
                training_plan.source_global_step,
                training_plan.execution_strategy.value,
                training_plan.plan_id,
                training_plan.target_worker_ids,
                training_plan.max_batches,
                training_plan.deadline_ts,
            )
            event = self._speco_get_drafter_scheduler().on_after_actor_update(
                AfterActorUpdateContext(
                    training_plan=training_plan,
                    runtime_state=runtime_state,
                )
            )
            outcome = event.training_execution
            if outcome is None:
                return False, dict(event.metrics or {})
            logger.info(
                "[DrafterRuntime] training completed: step=%s strategy=%s "
                "reason=%s trained=%s successful_steps=%s elapsed_s=%.4f",
                training_plan.source_global_step,
                training_plan.execution_strategy.value,
                outcome.reason,
                outcome.trained,
                outcome.successful_steps,
                outcome.elapsed_sec,
            )
        except Exception:
            logger.exception(
                "[DrafterRuntime] synchronous training failed at step=%s",
                training_plan.source_global_step,
            )
            raise
        return outcome.trained, dict(outcome.metrics)

    def _speco_poll_pending_drafter_training(
        self,
    ) -> tuple[TrainingPlan | None, Any | None]:
        runtime_state = self._speco_get_drafter_runtime_state()
        training_plan = runtime_state.active_plan
        outcome = self._speco_get_drafter_scheduler().poll_pending_training(
            runtime_state=runtime_state
        )
        if outcome is not None and training_plan is not None:
            self._speco_release_rollout_training_lease(training_plan)
            self._speco_last_known_trainable_batches = None
            logger.info(
                "[DrafterRuntime] async training completed: step=%s strategy=%s "
                "reason=%s trained=%s successful_steps=%s elapsed_s=%.4f",
                training_plan.source_global_step,
                training_plan.execution_strategy.value,
                outcome.reason,
                outcome.trained,
                outcome.successful_steps,
                outcome.elapsed_sec,
            )
        return training_plan, outcome

    def _speco_wait_pending_drafter_training(
        self,
    ) -> tuple[TrainingPlan | None, Any | None]:
        runtime_state = self._speco_get_drafter_runtime_state()
        training_plan = runtime_state.active_plan
        outcome = self._speco_get_drafter_scheduler().wait_pending_training(
            runtime_state=runtime_state
        )
        if outcome is not None and training_plan is not None:
            self._speco_release_rollout_training_lease(training_plan)
            self._speco_last_known_trainable_batches = None
            logger.warning(
                "[BubbleTime] reclaimed idle training before actor update: "
                "plan_id=%s trained=%s successful_steps=%s reason=%s elapsed_s=%.4f",
                training_plan.plan_id,
                outcome.trained,
                outcome.successful_steps,
                outcome.reason,
                outcome.elapsed_sec,
            )
        return training_plan, outcome

    def _speco_activate_drafter_training_model_before_fit(self) -> None:
        if not self.is_drafter_training_enabled(self.config):
            return
        if (
            self._speco_drafter_schedule_config().execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        ):
            # Do not place a complete model and optimizer on an arbitrary
            # rollout group before a real idle lease exists. If another group
            # wins the first trainable window, both copies would remain hot
            # and steal generation memory. The admission budget already
            # reserves cold-start time; activate the elected writer inside its
            # first genuine rollout bubble, then retain only that state.
            logger.warning(
                "[BubbleTime] deferred before-fit idle activation: "
                "reason=wait_for_first_trainable_idle_lease"
            )
            print(
                "[BubbleTime] idle_prewarm_deferred: "
                "reason=wait_for_first_trainable_idle_lease",
                flush=True,
            )
            return
        self._speco_get_drafter_scheduler().activate_training_workers()

    def _speco_wait_pending_drafter_publish_rpc(self) -> int:
        if not self._pending_drafter_publish_refs:
            return 0
        pending_refs = self._pending_drafter_publish_refs
        contexts = list(getattr(self, "_pending_drafter_publish_contexts", []) or [])
        wait_started = time.perf_counter()
        stage_elapsed_sec = 0.0
        commit_elapsed_sec = 0.0
        try:
            acknowledgements = self._ray_get_if_needed(pending_refs)
            stage_elapsed_sec = time.perf_counter() - wait_started
            ack_items = (
                acknowledgements
                if isinstance(acknowledgements, (list, tuple))
                else [acknowledgements]
            )
            completed_publish_acks = ack_items
            explicit_acks = [ack for ack in ack_items if isinstance(ack, dict)]
            failed_acks = [
                ack
                for ack in explicit_acks
                if not bool(ack.get("published", False))
                and not bool(ack.get("staged", False))
            ]
            if failed_acks:
                raise RuntimeError(
                    f"Drafter publish staging was not acknowledged: {failed_acks}"
                )
            staged_only = bool(explicit_acks) and all(
                bool(ack.get("staged", False)) and not bool(ack.get("published", False))
                for ack in explicit_acks
            )
            if staged_only:
                source_versions = {
                    int(cast(Any, context.get("source_global_step")))
                    for context in contexts
                    if context.get("source_global_step") is not None
                }
                if len(source_versions) != 1:
                    raise RuntimeError(
                        "Staged drafter publish requires one unambiguous source version: "
                        f"{sorted(source_versions)}"
                    )
                source_version = next(iter(source_versions))
                for context in contexts:
                    logger.warning(
                        "[BubbleTime] publish_stage_ack: plan_id=%s "
                        "source_step=%s stage_wait_s=%.4f",
                        context.get("plan_id"),
                        context.get("source_global_step"),
                        stage_elapsed_sec,
                    )
                    print(
                        "[BubbleTime] publish_stage_ack: "
                        f"plan_id={context.get('plan_id')} "
                        f"source_step={context.get('source_global_step')} "
                        f"stage_wait_s={stage_elapsed_sec:.4f}",
                        flush=True,
                    )
                commit_started = time.perf_counter()
                commit_result = self._ray_get_if_needed(
                    self.speco_commit_staged_drafter_weights(source_version)
                )
                commit_elapsed_sec = time.perf_counter() - commit_started
                commit_items = (
                    commit_result
                    if isinstance(commit_result, (list, tuple))
                    else [commit_result]
                )
                failed_commits = [
                    ack
                    for ack in commit_items
                    if isinstance(ack, dict) and not bool(ack.get("published", False))
                ]
                if failed_commits:
                    raise RuntimeError(
                        f"Staged drafter publish commit failed: {failed_commits}"
                    )
                completed_publish_acks = commit_items
            source_versions = {
                int(context.get("source_global_step"))
                for context in contexts
                if context.get("source_global_step") is not None
            }
            if len(source_versions) == 1:
                self._speco_get_drafter_scheduler().record_rollout_drafter_publish_completed(
                    next(iter(source_versions)),
                    acknowledgements=completed_publish_acks,
                )
        except Exception:
            logger.exception(
                "[BubbleTime] publish_failed: plan_ids=%s",
                tuple(context.get("plan_id") for context in contexts),
            )
            raise
        finally:
            self._pending_drafter_publish_refs = None
            self._pending_drafter_publish_contexts = []
        elapsed_sec = time.perf_counter() - wait_started
        for context in contexts:
            source_step = context.get("source_global_step")
            if source_step is not None:
                self._speco_published_drafter_version = int(source_step)
            logger.warning(
                "[BubbleTime] publish_succeeded: plan_id=%s source_step=%s "
                "workers=%s stage_wait_s=%.4f commit_s=%.4f elapsed_s=%.4f",
                context.get("plan_id"),
                context.get("source_global_step"),
                context.get("worker_ids"),
                stage_elapsed_sec,
                commit_elapsed_sec,
                elapsed_sec,
            )
            print(
                "[BubbleTime] publish_succeeded: "
                f"plan_id={context.get('plan_id')} "
                f"source_step={context.get('source_global_step')} "
                f"workers={context.get('worker_ids')} "
                f"stage_wait_s={stage_elapsed_sec:.4f} "
                f"commit_s={commit_elapsed_sec:.4f} "
                f"elapsed_s={elapsed_sec:.4f} mode=async_two_phase",
                flush=True,
            )
            print(
                "[BubbleTime] publish_version: "
                f"plan_id={context.get('plan_id')} "
                f"published_version={self._speco_published_drafter_version} "
                f"latest_target_version={self.global_steps} "
                "mode=bubble",
                flush=True,
            )
        self._speco_get_drafter_scheduler().record_training_quota_publish_completed(
            self.global_steps
        )
        return len(pending_refs) if isinstance(pending_refs, (list, tuple)) else 1

    def _speco_poll_pending_drafter_publish(self) -> dict[str, Any]:
        if not self._pending_drafter_publish_refs:
            return {"bubble/publish_pending": 0}
        if not self._speco_ray_refs_ready(self._pending_drafter_publish_refs):
            contexts = list(self._pending_drafter_publish_contexts or [])
            oldest_step = min(
                (
                    int(cast(Any, context.get("source_global_step")))
                    for context in contexts
                    if context.get("source_global_step") is not None
                ),
                default=int(self.global_steps),
            )
            lag_steps = max(int(self.global_steps) - oldest_step, 0)
            return {
                "bubble/publish_pending": 1,
                "bubble/publish_lag_steps": lag_steps,
            }
        completed = self._speco_wait_pending_drafter_publish_rpc()
        return {
            "bubble/publish_pending": 0,
            "bubble/publish_acknowledged": completed,
            "bubble/published_version": int(self._speco_published_drafter_version or 0),
        }

    def _speco_publish_boundary(self) -> dict[str, Any]:
        """Poll staging and block only after the configured lag bound."""

        metrics = self._speco_poll_pending_drafter_publish()
        if not metrics.get("bubble/publish_pending"):
            return metrics
        config = self._speco_drafter_schedule_config()
        lag_steps = int(metrics.get("bubble/publish_lag_steps", 0) or 0)
        if lag_steps < config.idle_worker_max_publish_lag_steps:
            print(
                "[BubbleTime] publish_deferred_without_blocking: "
                f"lag_steps={lag_steps} "
                f"max_lag_steps={config.idle_worker_max_publish_lag_steps}",
                flush=True,
            )
            return metrics
        wait_started = time.perf_counter()
        completed = self._speco_wait_pending_drafter_publish()
        metrics.update(
            {
                "bubble/publish_pending": 0,
                "bubble/publish_acknowledged": completed,
                "bubble/publish_forced_by_lag": 1,
                "timing_s/drafter_publish_lag_wait": time.perf_counter() - wait_started,
            }
        )
        return metrics

    def _speco_wait_pending_drafter_publish(self) -> int:
        scheduler = self._speco_get_drafter_scheduler()
        if getattr(scheduler, "_publish_executor", None) is None:
            self._speco_bind_publish_executor()
        return scheduler.wait_pending_publish()

    def _speco_get_published_drafter_weights(self):
        published = self._ray_get_if_needed(self.speco_maybe_publish()) or []
        return self._first_non_null(published)

    def _speco_update_rollout_drafter_weights(
        self, payload: Any, global_step: object, asynchronous: bool
    ) -> None:
        two_phase = bool(
            asynchronous and getattr(self, "_speco_two_phase_publish_active", False)
        )
        method_name = (
            "stage_draft_weights_async"
            if two_phase
            else (
                "update_draft_weights_async" if asynchronous else "update_draft_weights"
            )
        )
        update_result = self._speco_actor_rollout_method(method_name)(
            payload, global_steps=global_step
        )
        if asynchronous:
            self._pending_drafter_publish_refs = update_result
        else:
            self._ray_get_if_needed(update_result)

    def _speco_publish_drafter_weights(
        self,
        drafter_trained: bool,
        training_plan: TrainingPlan | None = None,
        *,
        after_weight_update: bool = False,
    ) -> dict[str, Any]:
        scheduler = self._speco_get_drafter_scheduler()
        if getattr(scheduler, "_publish_executor", None) is None:
            self._speco_bind_publish_executor()
        context = AfterWeightUpdateContext(
            global_step=self.global_steps,
            drafter_trained=drafter_trained,
            config=self._speco_drafter_schedule_config(),
            training_plan=training_plan,
        )
        self._speco_two_phase_publish_active = bool(
            training_plan is not None
            and training_plan.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        )
        try:
            event = (
                scheduler.on_after_weight_update(context)
                if after_weight_update
                else scheduler.on_safe_point(context)
            )
        finally:
            self._speco_two_phase_publish_active = False
        publish_plan = event.publish_plan
        publish_outcome = event.publish_outcome
        if (
            publish_plan is not None
            and publish_outcome is not None
            and publish_plan.asynchronous
            and publish_outcome.published
        ):
            pending_contexts = getattr(self, "_pending_drafter_publish_contexts", None)
            if not isinstance(pending_contexts, list):
                pending_contexts = []
                self._pending_drafter_publish_contexts = pending_contexts
            pending_contexts.append(
                {
                    "plan_id": getattr(training_plan, "plan_id", None),
                    "source_global_step": publish_plan.source_global_step,
                    "worker_ids": getattr(training_plan, "target_worker_ids", ()),
                    "submitted_at_step": int(self.global_steps),
                }
            )
            logger.warning(
                "[BubbleTime] publish_submitted: plan_id=%s source_step=%s workers=%s",
                getattr(training_plan, "plan_id", None),
                publish_plan.source_global_step,
                getattr(training_plan, "target_worker_ids", ()),
            )
            print(
                "[BubbleTime] publish_submitted: "
                f"plan_id={getattr(training_plan, 'plan_id', None)} "
                f"source_step={publish_plan.source_global_step} "
                f"workers={getattr(training_plan, 'target_worker_ids', ())}",
                flush=True,
            )
        elif (
            publish_plan is not None
            and publish_outcome is not None
            and publish_outcome.published
        ):
            # Synchronous publication completes in ``update_rollout_weights``
            # and therefore has no pending-Ray-ref wait path to emit the
            # success log.  Keep this at the shared publish boundary so both
            # deferred and direct safe-point callers are observable.
            logger.warning(
                "[BubbleTime] publish_succeeded: plan_id=%s source_step=%s "
                "workers=%s mode=sync",
                getattr(training_plan, "plan_id", None),
                publish_plan.source_global_step,
                getattr(training_plan, "target_worker_ids", ()),
            )
            print(
                "[BubbleTime] publish_succeeded: "
                f"plan_id={getattr(training_plan, 'plan_id', None)} "
                f"source_step={publish_plan.source_global_step} "
                f"workers={getattr(training_plan, 'target_worker_ids', ())} mode=sync",
                flush=True,
            )
            scheduler.record_rollout_drafter_publish_completed(
                publish_plan.source_global_step
            )
            scheduler.record_training_quota_publish_completed(self.global_steps)
        elif publish_plan is not None and publish_outcome is not None:
            logger.warning(
                "[BubbleTime] publish_skipped: plan_id=%s source_step=%s "
                "reason=%s attempted=%s",
                getattr(training_plan, "plan_id", None),
                publish_plan.source_global_step,
                publish_plan.reason,
                publish_outcome.attempted,
            )
            print(
                "[BubbleTime] publish_skipped: "
                f"plan_id={getattr(training_plan, 'plan_id', None)} "
                f"source_step={publish_plan.source_global_step} "
                f"reason={publish_plan.reason} "
                f"attempted={publish_outcome.attempted}",
                flush=True,
            )
        return dict(event.metrics or {})

    def _speco_update_output_metrics(self, output: Any, metrics: dict[str, Any]):
        if not metrics:
            return output
        meta_info = getattr(output, "meta_info", None)
        if isinstance(meta_info, dict):
            output_metrics = meta_info.setdefault("metrics", {})
            output_metrics.update(metrics)
            drafter_elapsed = _speco_metric_float(
                output_metrics.get("timing_s/drafter")
            )
            update_actor_elapsed = _speco_metric_float(
                output_metrics.get("timing_s/update_actor")
            )
            if drafter_elapsed is not None and update_actor_elapsed is not None:
                adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
                update_actor_per_token = _speco_metric_float(
                    output_metrics.get("timing_per_token_ms/update_actor")
                )
                if update_actor_per_token is not None:
                    output_metrics["timing_per_token_ms/update_actor"] = (
                        update_actor_per_token
                        * adjusted_update_actor
                        / update_actor_elapsed
                        if update_actor_elapsed > 0
                        else 0.0
                    )
                output_metrics["timing_s/update_actor"] = adjusted_update_actor
                output_metrics[_SPECO_DRAFTER_TIMING_DEDUCTED_KEY] = True
            try:
                scheduler = self._speco_get_drafter_scheduler()
                config = self._speco_drafter_schedule_config()
                feedback_metrics = scheduler.record_step_metrics(
                    output_metrics,
                    config,
                    global_step=self.global_steps,
                )
            except Exception:  # noqa: BLE001
                feedback_metrics = {}
            if feedback_metrics:
                output_metrics.update(feedback_metrics)
        return output

    def _speco_rollout_generation_target(self):
        for attr_name in ("async_rollout_manager", "actor_rollout_wg"):
            target = getattr(self, attr_name, None)
            if target is not None and callable(
                getattr(target, "generate_sequences", None)
            ):
                return target
        raise RuntimeError(
            "SPECO online drafter training requires a rollout generation object "
            "with generate_sequences(), but neither async_rollout_manager nor "
            "actor_rollout_wg exposes it."
        )

    def _speco_store_rollout_metrics(self, output: Any) -> None:
        current_step = getattr(self, "global_steps", None)
        if getattr(self, "_speco_last_rollout_metrics_step", None) != current_step:
            self._speco_last_rollout_metrics = {}
            self._speco_last_rollout_metrics_step = current_step
        self._speco_last_rollout_metrics = _speco_merge_vllm_spec_decode_stats(
            getattr(self, "_speco_last_rollout_metrics", None),
            _speco_vllm_spec_decode_stats_from_batch(output),
        )

    def _speco_current_step_rollout_metrics(self) -> dict[str, float]:
        if getattr(self, "_speco_last_rollout_metrics_step", None) != getattr(
            self, "global_steps", None
        ):
            return {}
        return _speco_vllm_spec_decode_metrics_from_stats(
            getattr(self, "_speco_last_rollout_metrics", None) or {}
        )

    @contextmanager
    def _speco_rollout_metrics_fit_hook(self):
        rollout_generation_target = self._speco_rollout_generation_target()
        original_generate_sequences = rollout_generation_target.generate_sequences

        def generate_sequences_with_speco_metrics(manager_self, *args, **kwargs):
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            if not _speco_is_validation_generation(args, kwargs, gen_batch_output):
                self._speco_store_rollout_metrics(gen_batch_output)
            return gen_batch_output

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco_metrics,
            rollout_generation_target,
        )
        try:
            yield
        finally:
            rollout_generation_target.generate_sequences = original_generate_sequences

    def _speco_bubble_profiler_enabled(self) -> bool:
        return bool(
            _get_nested(
                self.config,
                ("actor_rollout_ref", "rollout", "drafter", "profile_bubble"),
                False,
            )
        )

    def _speco_augment_log_data(
        self, data: Any, latest_rollout_metrics: dict[str, float]
    ) -> Any:
        if (
            isinstance(data, dict)
            and isinstance(latest_rollout_metrics, dict)
            and data.get("training/global_step") == self.global_steps
        ):
            data = dict(data)
            data.update(latest_rollout_metrics)
        data = _speco_move_drafter_timing_next_to_update_actor(data)
        if self._speco_bubble_profiler_enabled():
            data = inject_bubble_metrics(data)
        return data

    @contextmanager
    def _speco_tracking_metrics_hook(self):
        try:
            from verl.utils.tracking import Tracking
        except ImportError:
            yield
            return

        original_log = getattr(Tracking, "log", None)
        if not callable(original_log) or getattr(
            original_log, "_speco_drafter_timing_hook", False
        ):
            yield
            return

        def log_with_speco_metrics(tracking_self, *args, **kwargs):
            latest_rollout_metrics = self._speco_current_step_rollout_metrics()
            if "data" in kwargs:
                kwargs = dict(kwargs)
                kwargs["data"] = self._speco_augment_log_data(
                    kwargs["data"], latest_rollout_metrics
                )
                return original_log(tracking_self, *args, **kwargs)
            if args:
                args = (
                    self._speco_augment_log_data(args[0], latest_rollout_metrics),
                    *args[1:],
                )
            return original_log(tracking_self, *args, **kwargs)

        log_with_speco_metrics._speco_drafter_timing_hook = True
        Tracking.log = log_with_speco_metrics
        try:
            yield
        finally:
            Tracking.log = original_log

    def _speco_compute_old_log_prob_without_forced_entropy(self, batch: DataProto):
        batch = _select_policy_model_batch(batch)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self._speco_oldlogprob_calculate_entropy()
        tu.assign_non_tensor(
            batch_td, calculate_entropy=calculate_entropy, compute_loss=False
        )

        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]

        log_probs = no_padding_2_padding(log_probs, batch_td)
        if entropy is None:
            entropy = torch.zeros_like(log_probs, dtype=torch.float32)
        else:
            entropy = no_padding_2_padding(entropy, batch_td)
        if routed_experts is not None:
            old_log_prob = tu.get_tensordict(
                {
                    "old_log_probs": log_probs.float(),
                    "entropys": entropy.float(),
                    "routed_experts": routed_experts,
                }
            )
        else:
            old_log_prob = tu.get_tensordict(
                {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
            )
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    @contextmanager
    def _speco_oldlogprob_entropy_fit_hook(self):
        original_compute_old_log_prob = self._compute_old_log_prob

        def compute_old_log_prob_without_forced_entropy(trainer_self, batch: DataProto):
            return self._speco_compute_old_log_prob_without_forced_entropy(batch)

        self._compute_old_log_prob = MethodType(
            compute_old_log_prob_without_forced_entropy, self
        )
        try:
            yield
        finally:
            self._compute_old_log_prob = original_compute_old_log_prob

    @contextmanager
    def _speco_online_fit_hooks(self):
        rollout_generation_target = self._speco_rollout_generation_target()
        original_generate_sequences = rollout_generation_target.generate_sequences
        original_compute_old_log_prob = self._compute_old_log_prob
        original_update_actor = self._update_actor
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        original_checkpoint_update_weights = (
            getattr(checkpoint_manager, "update_weights", None)
            if checkpoint_manager is not None
            else None
        )
        defer_publish_until_update_weights = callable(
            original_checkpoint_update_weights
        )
        pending_drafter_publishes: list[dict[str, Any]] = []

        def drain_deferred_drafter_publishes(*, safe_point: str) -> dict[str, Any]:
            """Publish every completed Bubble plan at a rollout-safe boundary.

            ``checkpoint_manager.update_weights`` is an opportunistic early
            boundary, but some verl runtime variants update rollout weights
            through a different object.  The next generation invocation is a
            guaranteed fallback boundary: the preceding PPO iteration has
            completed its actor/rollout-weight update, while the next rollout
            has not started yet.
            """

            metrics: dict[str, Any] = {}
            while pending_drafter_publishes:
                if (
                    self._speco_rollout_idle_worker_enabled()
                    and self._pending_drafter_publish_refs
                ):
                    metrics["bubble/publish_queue_blocked_by_staging"] = len(
                        pending_drafter_publishes
                    )
                    break
                pending_publish = pending_drafter_publishes.pop(0)
                self._speco_deferred_publish_count = len(pending_drafter_publishes)
                training_plan = pending_publish["training_plan"]
                print(
                    "[BubbleTime] deferred_publish_drained: "
                    f"plan_id={getattr(training_plan, 'plan_id', None)} "
                    f"safe_point={safe_point}",
                    flush=True,
                )
                publish_started = time.perf_counter()
                try:
                    publish_metrics = self._speco_publish_drafter_weights(
                        pending_publish["drafter_trained"],
                        training_plan,
                        after_weight_update=True,
                    )
                except Exception:
                    publish_elapsed = time.perf_counter() - publish_started
                    logger.exception(
                        "[BubbleTime] publish_failed: plan_id=%s safe_point=%s "
                        "elapsed_s=%.4f",
                        getattr(training_plan, "plan_id", None),
                        safe_point,
                        publish_elapsed,
                    )
                    raise
                publish_elapsed = time.perf_counter() - publish_started
                published = bool(publish_metrics.get("drafter/published", 0))
                print(
                    "[BubbleTime] publish_safe_point_completed: "
                    f"plan_id={getattr(training_plan, 'plan_id', None)} "
                    f"safe_point={safe_point} published={published} "
                    f"elapsed_s={publish_elapsed:.4f}",
                    flush=True,
                )
                self._speco_update_output_metrics(
                    pending_publish["actor_output"], publish_metrics
                )
                for key, value in publish_metrics.items():
                    if key in {"drafter/publish_attempted", "drafter/published"}:
                        metrics[key] = max(int(metrics.get(key, 0)), int(value))
                    elif key.startswith("timing_s/"):
                        metrics[key] = float(metrics.get(key, 0.0)) + float(value)
                    else:
                        metrics[key] = value
                metrics["timing_s/drafter_publish_critical_path"] = (
                    float(metrics.get("timing_s/drafter_publish_critical_path", 0.0))
                    + publish_elapsed
                )
                metrics["timing_s/drafter_publish_safe_point"] = (
                    float(metrics.get("timing_s/drafter_publish_safe_point", 0.0))
                    + publish_elapsed
                )
            return metrics

        def defer_drafter_publish(
            *,
            drafter_trained: bool,
            training_plan: TrainingPlan | None,
            actor_output: Any,
        ) -> None:
            if not drafter_trained:
                return
            if training_plan is not None and not training_plan.publish_after_success:
                print(
                    "[BubbleTime] deferred_publish_skipped: "
                    f"plan_id={getattr(training_plan, 'plan_id', None)} "
                    "reason=quota_cycle_incomplete",
                    flush=True,
                )
                return
            publish_preview = self._speco_get_drafter_scheduler().plan_publish(
                global_step=self.global_steps,
                drafter_trained=True,
                config=self._speco_drafter_schedule_config(),
                training_plan=training_plan,
            )
            if not publish_preview.publish:
                print(
                    "[BubbleTime] deferred_publish_skipped: "
                    f"plan_id={getattr(training_plan, 'plan_id', None)} "
                    f"reason={publish_preview.reason}",
                    flush=True,
                )
                return
            pending_drafter_publishes.append(
                {
                    "drafter_trained": drafter_trained,
                    "training_plan": training_plan,
                    "actor_output": actor_output,
                }
            )
            self._speco_deferred_publish_count = len(pending_drafter_publishes)
            print(
                "[BubbleTime] deferred_publish_enqueued: "
                f"plan_id={getattr(training_plan, 'plan_id', None)} "
                "reason=await_upstream_weight_update",
                flush=True,
            )

        def generate_sequences_with_speco(manager_self, *args, **kwargs):
            generation_metrics = self._speco_publish_boundary()
            input_is_validation = _speco_is_validation_generation(args, kwargs)
            config = self._speco_drafter_schedule_config()
            scheduler = self._speco_get_drafter_scheduler()

            # Publication is part of a Bubble quota cycle.  Finish it before
            # deciding whether the next collection boundary is blocked;
            # otherwise an asynchronously staged (or locally deferred)
            # publish makes the just-completed cycle look active and drops a
            # whole collection interval (observed at steps 8/16 on NPU).
            if not self._pending_drafter_publish_refs:
                generation_metrics.update(
                    drain_deferred_drafter_publishes(
                        safe_point="before_next_generation"
                    )
                )
            collection_boundary = not input_is_validation and scheduler.should_collect(
                self.global_steps, config
            )
            if (
                collection_boundary
                and scheduler.training_quota_blocks_collection(
                    self.global_steps,
                    config,
                )
                and (self._pending_drafter_publish_refs or pending_drafter_publishes)
            ):
                publish_wait_started = time.perf_counter()
                # At most one publish RPC may be staged at once.  A completed
                # wait lets the next locally deferred plan enter staging.
                while self._pending_drafter_publish_refs or pending_drafter_publishes:
                    if self._pending_drafter_publish_refs:
                        completed = self._speco_wait_pending_drafter_publish()
                        generation_metrics["bubble/publish_acknowledged"] = int(
                            generation_metrics.get("bubble/publish_acknowledged", 0)
                        ) + int(completed)
                    if pending_drafter_publishes:
                        generation_metrics.update(
                            drain_deferred_drafter_publishes(
                                safe_point="collection_boundary"
                            )
                        )
                generation_metrics["bubble/publish_forced_by_collection"] = 1
                generation_metrics["timing_s/drafter_publish_collection_wait"] = (
                    time.perf_counter() - publish_wait_started
                )

            skip_collection_for_quota = (
                not input_is_validation
                and scheduler.training_quota_blocks_collection(
                    self.global_steps,
                    config,
                )
            )
            restore_collection_flag = _speco_set_generation_meta_flag(
                args,
                kwargs,
                "skip_drafter_collection",
                skip_collection_for_quota,
            )
            if skip_collection_for_quota:
                generation_metrics["bubble/collection_skipped_active_quota"] = 1
            logger.debug(
                "[BubbleTime] generation_hook: validation=%s online_enabled=%s "
                "idle_enabled=%s event_bus=%s",
                input_is_validation,
                self._speco_online_enabled(),
                self._speco_rollout_idle_worker_enabled(),
                self._speco_rollout_idle_event_bus_name() or "",
            )
            if not input_is_validation:
                self._speco_direct_sample_ids = set()
                self._speco_direct_collected_samples = 0
                self._speco_runtime_idle_events_this_generation = 0
                self._speco_runtime_sample_events_this_generation = 0
                generation_metrics.update(
                    self._speco_reclaim_rollout_idle_workers_before_generation()
                )
                # Version every sample collected in this rollout before any
                # replica can report an idle Bubble Time window.
                self._speco_set_drafter_global_step()
                generation_metrics.update(self._speco_emit_rollout_generation_started())
            stop_event, event_thread = self._speco_start_rollout_idle_event_loop()
            try:
                gen_batch_output = original_generate_sequences(*args, **kwargs)
            finally:
                self._speco_stop_rollout_idle_event_loop(stop_event, event_thread)
                # Events observed after generation has returned no longer own
                # an idle window. Record them, but never launch new training.
                generation_metrics.update(
                    self._speco_service_rollout_idle_events(allow_launch=False)
                )
                restore_collection_flag()
            is_validation_generation = _speco_is_validation_generation(
                args, kwargs, gen_batch_output
            )
            runtime_idle_events = int(
                getattr(self, "_speco_runtime_idle_events_this_generation", 0)
            )
            runtime_sample_events = int(
                getattr(self, "_speco_runtime_sample_events_this_generation", 0)
            )
            logger.debug(
                "[BubbleTime] generation_complete: validation=%s metrics_keys=%s "
                "runtime_events_drained=%s runtime_idle_events_this_generation=%s "
                "runtime_sample_events_this_generation=%s",
                is_validation_generation,
                tuple(sorted(generation_metrics)),
                generation_metrics.get("bubble/runtime_worker_events_drained", 0),
                runtime_idle_events,
                runtime_sample_events,
            )
            if not is_validation_generation:

                def handle_post_rollout_completion(completed_plan, outcome):
                    if defer_publish_until_update_weights:
                        defer_drafter_publish(
                            drafter_trained=outcome.trained,
                            training_plan=completed_plan,
                            actor_output=None,
                        )
                        return {"bubble/post_rollout_training_drained": 1}
                    return self._speco_publish_drafter_weights(
                        outcome.trained,
                        completed_plan,
                    )

                # Do not let Bubble work leak into old_log_prob, ref, or
                # update_actor. A worker completes only its in-flight batch,
                # then cleanup/offload finishes before this hook returns.
                generation_metrics.update(
                    self._speco_reclaim_rollout_idle_workers_before_post_rollout(
                        on_completed=handle_post_rollout_completion,
                    )
                )
                generation_metrics.update(
                    self._speco_emit_rollout_generation_completed(gen_batch_output)
                )
                if (
                    self._speco_rollout_idle_worker_enabled()
                    and runtime_idle_events == 0
                ):
                    # Without a runtime release callback there is no proven
                    # overlap window. Fail closed instead of moving drafter
                    # work onto the PPO critical path.
                    generation_metrics["bubble/no_confirmed_idle_window"] = 1
                elif self._speco_rollout_idle_worker_enabled():
                    # Old-log-prob collection feeds the next Bubble quota, so
                    # a confirmed post-rollout idle window should immediately
                    # train from the previous ready snapshot.  Do this before
                    # post-generation storage/collection work so the scheduler
                    # sees the fresh runtime deadline instead of an expired
                    # window_s=0.
                    generation_metrics.update(
                        self._speco_try_launch_rollout_idle_training()
                    )
                self._speco_store_rollout_metrics(gen_batch_output)
                collected = self._speco_collect_generation_samples(gen_batch_output)
                if collected:
                    meta_info = getattr(gen_batch_output, "meta_info", None)
                    if isinstance(meta_info, dict):
                        meta_info.setdefault("metrics", {})[
                            "drafter/collected_samples"
                        ] = collected
                meta_info = getattr(gen_batch_output, "meta_info", None)
                if isinstance(meta_info, dict) and generation_metrics:
                    meta_info.setdefault("metrics", {}).update(generation_metrics)
            return gen_batch_output

        def compute_old_log_prob_with_speco(trainer_self, batch: DataProto):
            if not self._speco_oldlogprob_collection_enabled():
                if self._speco_oldlogprob_entropy_hook_enabled():
                    return self._speco_compute_old_log_prob_without_forced_entropy(
                        batch
                    )
                return original_compute_old_log_prob(batch)

            oldlogprob_started = time.perf_counter()
            self._speco_last_oldlogprob_candidate_samples = 0
            self._speco_last_oldlogprob_planned_samples = 0
            self._speco_last_oldlogprob_collected_samples = 0
            self._speco_last_oldlogprob_collected_rows = 0
            self._speco_last_oldlogprob_payload_mib = 0.0
            self._speco_last_oldlogprob_select_elapsed_sec = 0.0
            self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
            self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
            self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
            self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
            self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
            self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
            self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
            self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
            self._speco_last_oldlogprob_total_elapsed_sec = 0.0
            collection_plan = self._speco_plan_drafter_collection(
                DrafterCollectionSource.OLD_LOGPROB
            )
            self._speco_log_drafter_collection_plan(collection_plan)
            self._speco_last_collect_interval_matched = int(
                collection_plan.collect_interval_matched
            )
            prepare_started = time.perf_counter()
            original_batch = batch

            def compute_old_log_prob_without_collection():
                self._speco_last_oldlogprob_prepare_elapsed_sec = (
                    time.perf_counter() - prepare_started
                )
                compute_started = time.perf_counter()
                if self._speco_oldlogprob_entropy_hook_enabled():
                    old_log_prob, old_log_prob_mfu = (
                        self._speco_compute_old_log_prob_without_forced_entropy(
                            original_batch
                        )
                    )
                else:
                    old_log_prob, old_log_prob_mfu = original_compute_old_log_prob(
                        original_batch
                    )
                self._speco_last_oldlogprob_compute_elapsed_sec = (
                    time.perf_counter() - compute_started
                )
                self._speco_last_oldlogprob_total_elapsed_sec = (
                    time.perf_counter() - oldlogprob_started
                )
                return old_log_prob, old_log_prob_mfu

            if not collection_plan.collect:
                return compute_old_log_prob_without_collection()

            batch = _select_policy_model_batch(batch)
            collect_plan = self._speco_build_oldlogprob_collect_plan(batch)
            if collect_plan is None:
                return compute_old_log_prob_without_collection()
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self._speco_oldlogprob_calculate_entropy()
            tu.assign_non_tensor(
                batch_td, calculate_entropy=calculate_entropy, compute_loss=False
            )
            batch_td[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
            batch_td[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan[
                "hidden_positions"
            ]
            batch_td[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan[
                "hidden_position_mask"
            ]
            batch_td[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_AUX_LAYER_IDS_KEY,
                self._speco_oldlogprob_aux_layer_ids(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY,
                self._speco_oldlogprob_aux_layer_id_space(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
                self._speco_oldlogprob_hidden_capture_impl(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
                self._speco_oldlogprob_hidden_layout(),
            )
            tu.assign_non_tensor_data(batch_td, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True)

            self._speco_last_oldlogprob_prepare_elapsed_sec = (
                time.perf_counter() - prepare_started
            )
            compute_started = time.perf_counter()
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            self._speco_last_oldlogprob_compute_elapsed_sec = (
                time.perf_counter() - compute_started
            )
            collect_started = time.perf_counter()
            collected = self._speco_collect_oldlogprob_features(
                batch,
                collect_plan,
                output,
            )
            self._speco_last_oldlogprob_collect_elapsed_sec = (
                time.perf_counter() - collect_started
            )
            if collected > 0 and self._speco_rollout_idle_worker_enabled():
                # Bubble Time deliberately consumes buffered data from the
                # preceding rollout.  Launching here would make the current
                # rollout wait for old-logprob collection and spend the same
                # bubble that should be reserved for training.  update_actor
                # below caches the matching actor head before its update; the
                # next rollout's generation-complete hook then selects this
                # immutable data/head pair and trains it asynchronously.
                print(
                    "[BubbleTime] pipeline_data_staged: "
                    f"samples={collected} source_step={self.global_steps} "
                    "launch_deferred_to_next_generation=True",
                    flush=True,
                )

            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]

            log_probs = no_padding_2_padding(log_probs, batch_td)
            if entropy is None:
                entropy = torch.zeros_like(log_probs, dtype=torch.float32)
            else:
                entropy = no_padding_2_padding(entropy, batch_td)
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {
                        "old_log_probs": log_probs.float(),
                        "entropys": entropy.float(),
                        "routed_experts": routed_experts,
                    }
                )
            else:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
                )
            old_log_prob = DataProto.from_tensordict(old_log_prob)
            self._speco_last_oldlogprob_total_elapsed_sec = (
                time.perf_counter() - oldlogprob_started
            )
            return old_log_prob, old_log_prob_mfu

        def update_actor_with_speco(trainer_self, *args, **kwargs):
            update_actor_started = time.perf_counter()
            metrics = {
                "drafter/raw_drafter_samples": int(
                    getattr(self, "_speco_last_raw_drafter_samples", 0)
                ),
                "drafter/collected_samples": int(
                    getattr(self, "_speco_last_collected_samples", 0)
                ),
                "drafter/collect_interval_matched": int(
                    getattr(self, "_speco_last_collect_interval_matched", 0)
                ),
            }
            metrics.update(self._speco_pop_rollout_idle_metrics())
            completed_training_metrics: dict[str, Any] = {}
            completion_wait_started = time.perf_counter()
            if self._speco_rollout_idle_worker_enabled():
                # Post-rollout reclaim has already stopped Bubble work before
                # PPO. Poll defensively in case a backend completed during the
                # boundary transition; never start overlap from this hook.
                completed_plan, completed_outcome = (
                    self._speco_poll_pending_drafter_training()
                )
                metrics["bubble/training_completion_polled"] = 1
                metrics["timing_s/drafter_completion_wait"] = 0.0
            else:
                completed_plan, completed_outcome = (
                    self._speco_wait_pending_drafter_training()
                )
                metrics["timing_s/drafter_completion_wait"] = (
                    time.perf_counter() - completion_wait_started
                )
            if completed_outcome is not None:
                # Keep the completion result separate until after the current
                # step's scheduling metrics are assembled.  Otherwise a
                # subsequent "no_trainable_batch" plan overwrites the result
                # of a successfully completed asynchronous Bubble plan.
                completed_training_metrics.update(completed_outcome.metrics)
                if defer_publish_until_update_weights:
                    defer_drafter_publish(
                        drafter_trained=completed_outcome.trained,
                        training_plan=completed_plan,
                        actor_output=None,
                    )
                else:
                    metrics.update(
                        self._speco_publish_drafter_weights(
                            completed_outcome.trained,
                            completed_plan,
                        )
                    )
            collection_plan = getattr(self, "_speco_last_collection_plan", None)
            if isinstance(collection_plan, CollectionPlan):
                metrics.update(collection_plan.metrics())
            collection_outcome = getattr(self, "_speco_last_collection_outcome", None)
            if isinstance(collection_outcome, CollectionOutcome):
                metrics.update(collection_outcome.metrics())
            if self._speco_rollout_idle_worker_enabled():
                # Advance an already-finished previous prefetch before deciding
                # whether the current collected version needs a new one. This
                # is timeout=0/non-blocking and avoids dropping a version merely
                # because completion had not yet been observed by the driver.
                metrics.update(self._speco_poll_target_lm_head_prefetch())
            quota_topup_safe = (
                completed_outcome is None
                and not pending_drafter_publishes
                and not self._pending_drafter_publish_refs
                and self._pending_target_lm_head_sync is None
            )
            before_actor_event = self._speco_on_before_actor_update(
                allow_quota_topup=quota_topup_safe
            )
            training_plan = before_actor_event.training_plan
            if training_plan is None:
                raise RuntimeError(
                    "Drafter before-actor-update event returned no training plan"
                )
            metrics.update(before_actor_event.metrics or {})
            self._speco_log_drafter_training_plan(training_plan, metrics)
            metrics["drafter/train_interval_matched"] = int(
                training_plan.interval_matched
            )
            if self._speco_should_prefetch_target_lm_head():
                # Cache actor head N after collecting step N features and before
                # actor update N.  The step N+1 bubble selects this immutable
                # version instead of fetching the live, updated actor head.
                cache_metrics, pending_target_lm_head_sync = (
                    self._speco_start_target_lm_head_weight_sync(None)
                )
                metrics.update(cache_metrics)
                if pending_target_lm_head_sync is not None:
                    self._pending_target_lm_head_sync = pending_target_lm_head_sync
                    print(
                        "[BubbleTime] target_lm_head_prefetch_submitted: "
                        "target_version="
                        f"{pending_target_lm_head_sync.get('target_version')} "
                        "target_workers="
                        f"{pending_target_lm_head_sync.get('target_worker_ids')} "
                        "selected_rows="
                        f"{cache_metrics.get('drafter/target_lm_head_selected_rows', 0)} "
                        "submit_s="
                        f"{cache_metrics.get('timing_s/drafter_target_lm_head_fetch_submit', 0.0):.3f}",
                        flush=True,
                    )
            actor_started = time.perf_counter()
            actor_output = original_update_actor(*args, **kwargs)
            actor_elapsed = time.perf_counter() - actor_started
            if self._speco_rollout_idle_worker_enabled():
                # Never force a Ray wait here.  A later idle event advances
                # this state machine and only launches once the exact cached
                # target version is ready on every drafter worker.
                metrics.update(self._speco_poll_target_lm_head_prefetch())
            if (
                training_plan.launch
                and training_plan.reason == "quota_topup_training_ready"
                and self._pending_target_lm_head_sync is not None
            ):
                # A quota top-up is deliberately subordinate to the PPO
                # critical path.  Never let its synchronous preflight queue
                # behind an unfinished target-head transfer; preserve the
                # debt and retry at a later safe point instead.
                pending_target_version = self._pending_target_lm_head_sync.get(
                    "target_version"
                )
                print(
                    "[BubbleTime] training_quota_topup_deferred: "
                    "reason=target_lm_head_prefetch_pending "
                    f"plan_id={training_plan.plan_id} "
                    f"target_version={pending_target_version}",
                    flush=True,
                )
                metrics["bubble/training_quota_topup_deferred"] = 1
                training_plan = replace(
                    training_plan,
                    launch=False,
                    reason="quota_topup_lm_head_prefetch_pending",
                    max_batches=0,
                    publish_after_success=False,
                )
                metrics.update(training_plan.metrics())
            if training_plan.launch:
                drafter_trained, train_metrics = self._speco_train_drafter(
                    training_plan
                )
            else:
                drafter_trained, train_metrics = (
                    False,
                    {
                        "drafter/trained": 0,
                        "drafter/train_successful_steps_max": 0,
                        "drafter/train_no_trainable_batch": int(
                            training_plan.reason == "no_trainable_batch"
                        ),
                        "drafter/train_activation_failed": 0,
                    },
                )
                train_metrics.update(self._speco_get_drafter_runtime_state().metrics())
            metrics.update(train_metrics)
            metrics.update(completed_training_metrics)
            if defer_publish_until_update_weights and drafter_trained:
                defer_drafter_publish(
                    drafter_trained=drafter_trained,
                    training_plan=training_plan,
                    actor_output=actor_output,
                )
            else:
                metrics.update(
                    self._speco_publish_drafter_weights(drafter_trained, training_plan)
                )
            metrics["timing_s/drafter"] = max(
                0.0, time.perf_counter() - update_actor_started - actor_elapsed
            )
            known_drafter_timing = 0.0
            for key in (
                "timing_s/drafter_sync_target_lm_head",
                "timing_s/drafter_train_rpc",
                "timing_s/drafter_publish_wait_pending",
                "timing_s/drafter_publish_fetch_snapshot",
                "timing_s/drafter_publish_update_weights",
                "timing_s/drafter_target_lm_head_prefetch_submit_critical_path",
            ):
                value = _speco_metric_float(metrics.get(key))
                if value is not None:
                    known_drafter_timing += value
            metrics["timing_s/drafter_outer_unaccounted"] = max(
                0.0,
                metrics["timing_s/drafter"] - known_drafter_timing,
            )
            metrics["timing_s/drafter_critical_path_update_actor"] = metrics[
                "timing_s/drafter"
            ]
            metrics["timing_s/drafter_control_overhead"] = metrics[
                "timing_s/drafter_outer_unaccounted"
            ]
            metrics["drafter/trained_any"] = int(
                bool(metrics.get("drafter/trained", 0))
                or bool(metrics.get("drafter/idle_trained", 0))
            )
            if (
                training_plan.execution_strategy
                is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            ):
                metrics.setdefault(
                    "drafter/idle_trained",
                    int(bool(train_metrics.get("drafter/trained", 0))),
                )
            return self._speco_update_output_metrics(actor_output, metrics)

        def update_weights_with_speco(manager_self, *args, **kwargs):
            result = original_checkpoint_update_weights(*args, **kwargs)
            publish_metrics = drain_deferred_drafter_publishes(
                safe_point="checkpoint_manager_update_weights"
            )
            # Checkpoint-manager update may not have an actor output object to
            # attach to.  Preserve the result for the next actor metrics flush
            # so publish success and its safe-point cost remain observable.
            self._speco_record_rollout_idle_metrics(publish_metrics)
            return self._speco_update_output_metrics(result, publish_metrics)

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco,
            rollout_generation_target,
        )
        if self._speco_oldlogprob_collection_requested():
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco, self
            )
        elif self._speco_oldlogprob_entropy_hook_enabled():
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco,
                self,
            )
        self._update_actor = MethodType(update_actor_with_speco, self)
        if defer_publish_until_update_weights:
            checkpoint_manager.update_weights = MethodType(
                update_weights_with_speco, checkpoint_manager
            )
        try:
            yield
        finally:
            rollout_generation_target.generate_sequences = original_generate_sequences
            self._compute_old_log_prob = original_compute_old_log_prob
            self._update_actor = original_update_actor
            if defer_publish_until_update_weights:
                checkpoint_manager.update_weights = original_checkpoint_update_weights
            drain_deferred_drafter_publishes(safe_point="fit_teardown")
            self._speco_wait_pending_drafter_publish()

    @staticmethod
    def is_drafter_rollout_enabled(config) -> bool:
        return bool(
            _get_nested(
                config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False
            )
        )

    @staticmethod
    def is_drafter_training_enabled(config) -> bool:
        drafter_enabled = bool(
            _get_nested(
                config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False
            )
        )
        training_enabled = bool(
            _get_nested(
                config,
                ("actor_rollout_ref", "rollout", "drafter", "enable_drafter_training"),
                False,
            )
        )
        return drafter_enabled and training_enabled

    def fit(self):
        try:
            if self.is_drafter_training_enabled(self.config):
                self._speco_activate_drafter_training_model_before_fit()
                with (
                    self._speco_tracking_metrics_hook(),
                    self._speco_online_fit_hooks(),
                ):
                    return super().fit()
            if self.is_drafter_rollout_enabled(self.config):
                with (
                    self._speco_tracking_metrics_hook(),
                    self._speco_rollout_metrics_fit_hook(),
                ):
                    if self._speco_oldlogprob_entropy_hook_enabled():
                        with self._speco_oldlogprob_entropy_fit_hook():
                            return super().fit()
                    return super().fit()
            if self._speco_oldlogprob_entropy_hook_enabled():
                with self._speco_oldlogprob_entropy_fit_hook():
                    return super().fit()

            return super().fit()
        finally:
            self._speco_wait_pending_drafter_checkpoint()

    def _save_checkpoint(self):
        # A checkpoint boundary must not retain an async publish payload or let
        # draft loading overlap actor/drafter serialization. This is redundant
        # with the normal next-generation barrier by design: save/test order is
        # controlled by upstream VERL and can change independently.
        self._speco_checkpoint_in_progress = True
        try:
            # Bubble mode may still own rollout resources or have an LM-head /
            # publish operation in flight. Quiesce those operations first;
            # the explicit publish wait below also preserves the upstream
            # barrier for synchronous and rollout-only configurations.
            self._speco_quiesce_bubble_for_checkpoint()
            self._speco_wait_pending_drafter_publish()
            self._speco_save_drafter_checkpoint(wait=True)
            return super()._save_checkpoint()
        finally:
            self._speco_checkpoint_in_progress = False

    def _validate(self, *args, **kwargs):
        # Validation commonly drives KV usage to the configured limit. Ensure
        # online weight loading and its temporary buffers have completed before
        # validation admits requests into the rollout engine.
        self._speco_wait_pending_drafter_publish()
        return super()._validate(*args, **kwargs)
