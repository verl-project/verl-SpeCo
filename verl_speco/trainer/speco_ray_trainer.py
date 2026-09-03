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
from contextlib import contextmanager
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
from verl_speco.integration.oldlogprob_runtime import (
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
    OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY,
    OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY,
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_TIMING_KEY,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    assert_sglang_aux_last_layer_norm_safe,
    resolve_drafter_hidden_states_layout,
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.integration.sglang_adapter import pop_drafter_samples
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
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
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
_DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"

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


def _speco_cap_online_dspark_validation_batch_size(config) -> int | None:
    """Apply the SpeCo validation cap before VERL builds its dataloader.

    VERL 0.9 uses the complete validation dataset as one batch when
    ``data.val_batch_size`` is unset.  With native MRV2 DSpark that batch is
    further expanded by ``val_kwargs.n`` and can drive the vLLM KV cache to its
    limit while the auxiliary-hidden-state projection is materialized.
    Preserve 0.8/MRV1 behavior and explicit data-level values; only the MRV2
    launcher opts into this cap through ``VLLM_USE_V2_MODEL_RUNNER``.
    """

    rollout_cfg = _get_nested(config, ("actor_rollout_ref", "rollout"), None)
    drafter_cfg = _get_nested(rollout_cfg, ("drafter",), None)
    training_cfg = _get_nested(drafter_cfg, ("training",), None)
    if (
        str(_get_nested(rollout_cfg, ("name",), "")).lower() != "vllm"
        or not bool(_get_nested(drafter_cfg, ("enable",), False))
        or not bool(_get_nested(drafter_cfg, ("enable_drafter_training",), False))
        or str(_get_nested(drafter_cfg, ("speculative_algorithm",), "")).upper()
        != "DSPARK"
        or str(_get_nested(training_cfg, ("mode",), "online")).lower() != "online"
        or os.getenv("VLLM_USE_V2_MODEL_RUNNER", "").lower() not in {"1", "true", "yes"}
    ):
        return None

    data_cfg = _get_nested(config, ("data",), None)
    if data_cfg is None:
        return None
    explicit_batch_size = _get_nested(data_cfg, ("val_batch_size",), None)
    if explicit_batch_size is not None:
        return int(explicit_batch_size)

    configured_batch_size = _get_nested(training_cfg, ("validation_batch_size",), None)
    if configured_batch_size is None:
        return None
    if isinstance(configured_batch_size, bool):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.validation_batch_size "
            "must be a positive integer or null"
        )
    try:
        validation_batch_size = int(configured_batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.validation_batch_size "
            "must be a positive integer or null"
        ) from exc
    if validation_batch_size <= 0:
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.validation_batch_size "
            "must be a positive integer or null"
        )

    with open_dict(data_cfg):
        data_cfg["val_batch_size"] = validation_batch_size
    logger.warning(
        "SPECO bounded native MRV2 DSpark validation to %d prompts per batch before "
        "val_kwargs.n expansion",
        validation_batch_size,
    )
    return validation_batch_size


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
        config = kwargs.get("config", args[0] if args else None)
        if config is not None:
            _speco_cap_online_dspark_validation_batch_size(config)
        super().__init__(*args, **kwargs)
        self.drafter_wg = None
        self._drafter_scheduler = DrafterScheduler()
        self._drafter_runtime_state = DrafterRuntimeState()
        self._pending_drafter_publish_refs = None
        self._pending_drafter_checkpoint_refs = []
        self._pending_target_lm_head_sync = None
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

    def attach_speco_worker_group(self, worker_group):
        self.drafter_wg = worker_group
        self._speco_get_drafter_scheduler().bind_worker_executor(
            CallbackDrafterWorkerExecutor(
                submit=self.speco_train_drafter,
                resolve=self._ray_get_if_needed,
                inspect_data=self.speco_get_drafter_training_data_status,
                prepare=self._speco_prepare_drafter_training_rpc,
                activate=self.speco_activate_drafter_training_model,
                preflight=self.speco_preflight_drafter_training,
                abort_preflight=self.speco_abort_drafter_training_preflight,
            )
        )
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
    ):
        return self._require_speco_worker_group().get_drafter_training_data_status(
            sample_last_n_steps,
            require_full_batch,
        )

    def speco_activate_drafter_training_model(self):
        return self._require_speco_worker_group().activate_drafter_training_model()

    def speco_maybe_publish(self):
        return self._require_speco_worker_group().maybe_publish()

    def speco_save_checkpoint(
        self,
        global_step: int,
        wait: bool = True,
    ):
        return self._require_speco_worker_group().save_checkpoint(
            global_step,
            wait=wait,
        )

    def speco_wait_checkpoint(self):
        return self._require_speco_worker_group().wait_checkpoint()

    def init_workers(self):
        drafter_rollout_enabled = self.is_drafter_rollout_enabled(self.config)
        online_drafter_enabled = self.is_drafter_training_enabled(self.config)
        if online_drafter_enabled:
            self._speco_prepare_drafter_checkpoint_for_worker_init()
        if drafter_rollout_enabled:
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
        allowed_skips = {"not_checkpoint_replica", "not_in_training_group"}
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
        checkpoint_refs = self.speco_save_checkpoint(
            self.global_steps,
            wait=wait,
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
                ),
            ),
            self._speco_drafter_schedule_config(),
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

    def _speco_on_before_actor_update(self):
        return self._speco_get_drafter_scheduler().on_before_actor_update(
            BeforeActorUpdateContext(
                schedule_context=self._speco_drafter_schedule_context(),
                config=self._speco_drafter_schedule_config(),
            )
        )

    @staticmethod
    def _speco_log_drafter_training_plan(plan: TrainingPlan) -> None:
        logger.info(
            "[DrafterScheduler] step=%s strategy=%s launch=%s reason=%s "
            "interval_matched=%s max_batches=%s publish_after_success=%s",
            plan.source_global_step,
            plan.execution_strategy.value,
            plan.launch,
            plan.reason,
            plan.interval_matched,
            plan.max_batches,
            plan.publish_after_success,
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
        outcome = self._speco_get_drafter_scheduler().on_collection_ready(
            plan,
            payload,
        )
        self._speco_last_collection_outcome = outcome
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
        if strategy not in {"fsdp", "fsdp2", "megatron", "veomni"}:
            raise ValueError(
                "SPECO old-logprob hidden collection supports "
                "actor.strategy=fsdp/fsdp2/megatron/veomni, "
                f"got {strategy!r}"
            )
        if strategy == "megatron":
            tp_size = int(
                _get_nested(
                    self.config,
                    (
                        "actor_rollout_ref",
                        "actor",
                        "megatron",
                        "tensor_model_parallel_size",
                    ),
                    1,
                )
                or 1
            )
            pp_size = int(
                _get_nested(
                    self.config,
                    (
                        "actor_rollout_ref",
                        "actor",
                        "megatron",
                        "pipeline_model_parallel_size",
                    ),
                    1,
                )
                or 1
            )
            logger.warning(
                "SPECO old-logprob hidden collection with Megatron backend: "
                f"TP={tp_size}, PP={pp_size}. "
                "TP>1 uses Megatron native sequence parallelism for hidden-state gathering. "
                "PP>1 uses cross-stage dist communication for capture transfer."
            )
        capture_impl = str(
            training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook")
            or "forward_hook"
        )
        if capture_impl not in {"forward_hook", "output_hidden_states"}:
            raise ValueError(
                f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}"
            )
        if strategy == "megatron" and capture_impl != "forward_hook":
            raise ValueError(
                "SPECO old-logprob hidden collection with Megatron backend supports "
                f"forward_hook capture only, got {capture_impl!r}"
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
        max_per_owner = collection_plan.max_samples_per_replica
        max_per_owner = max_per_owner if max_per_owner is not None else batch_size
        max_per_owner = max(max_per_owner, 0)
        max_tokens_per_owner = collection_plan.max_tokens_per_replica
        if max_tokens_per_owner is not None:
            max_tokens_per_owner = max(max_tokens_per_owner, 0)
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
            owner = selected_count % owner_count
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
        # PP>1 single-put path: the last stage ray.put()s the concatenated
        # hidden tensor once and returns the ObjectRef.  Materialize it here
        # (once) and release the ref immediately so the big tensor does not pin
        # the Ray object store; the rest of this function treats it as an
        # inline tensor.
        whole_ref = tu.get(output, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY)
        if hidden_states is None and whole_ref is not None:
            import ray as _ray

            hidden_states = _ray.get(whole_ref)
            # Drop every reference to the ObjectRef (local var + the copy held
            # inside the output TensorDict) so Ray can free the big tensor from
            # the object store as soon as this materialization completes,
            # instead of waiting for the whole step output to be GC'd.
            del whole_ref
            try:
                tu.assign_non_tensor_data(
                    output, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY, None
                )
                tu.assign_non_tensor_data(
                    output, OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY, None
                )
            except Exception:
                pass
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

    def _speco_collect_generation_samples(self, gen_batch_output: Any) -> int:
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        collection_plan = self._speco_plan_drafter_collection(
            DrafterCollectionSource.SGLANG
        )
        self._speco_log_drafter_collection_plan(collection_plan)
        self._speco_last_collect_interval_matched = int(
            collection_plan.collect_interval_matched
        )
        if not self._speco_online_enabled():
            return 0
        samples = pop_drafter_samples(gen_batch_output)
        self._speco_last_raw_drafter_samples = len(samples)
        if not samples:
            return 0
        if not collection_plan.collect:
            return 0

        num_replicas = self._speco_num_rollout_replicas(samples)
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        payload = self._speco_get_drafter_scheduler().prepare_collection_payload(
            source=DrafterCollectionSource.SGLANG,
            samples=samples,
            owner_count=num_replicas,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=len(samples),
            collection_id=collection_plan.collection_id,
        )

        outcome = self._speco_execute_collection(
            collection_plan,
            payload,
        )
        self._speco_last_collected_samples = outcome.collected_samples
        return outcome.collected_samples

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
        payloads = (
            self._ray_get_if_needed(
                get_actor_lm_head_weight(
                    row_indices,
                    keep_model_on_device=keep_actor_model_on_device,
                )
            )
            or []
        )
        fetch_elapsed = time.perf_counter() - fetch_started
        payload = self._first_non_null(payloads)
        if payload is None:
            return (
                {
                    "drafter/target_lm_head_synced": 0,
                    "drafter/target_lm_head_selected_rows": selected_rows,
                    "drafter/target_lm_head_source_vocab_size": source_vocab_size,
                    "timing_s/drafter_sync_target_lm_head": time.perf_counter()
                    - sync_started,
                    "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
                },
                None,
            )

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
            "drafter/target_lm_head_direct_sparse_export": int(
                export_strategy in {"direct_sparse", "veomni_lm_head_sparse"}
            ),
            "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            "timing_s/drafter_sync_target_lm_head_dispatch": dispatch_elapsed,
        }
        pending = {
            "refs": pending_refs,
            "dispatch_finished": dispatch_started + dispatch_elapsed,
            "dispatch_elapsed": dispatch_elapsed,
            "pre_dispatch_elapsed": dispatch_started - sync_started,
        }
        if defer_device_apply and pending_refs is not None:
            return metrics, pending

        metrics.update(self._speco_finish_target_lm_head_weight_sync(pending))
        return metrics, None

    def _speco_finish_target_lm_head_weight_sync(
        self, pending: dict[str, Any]
    ) -> dict[str, Any]:
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
            event = self._speco_get_drafter_scheduler().on_after_actor_update(
                AfterActorUpdateContext(
                    training_plan=training_plan,
                    runtime_state=runtime_state,
                )
            )
            outcome = event.training_execution
            if outcome is None:
                raise RuntimeError(
                    "Drafter after-actor-update event returned no training outcome"
                )
        except Exception:
            logger.exception(
                "[DrafterRuntime] synchronous training failed at step=%s",
                training_plan.source_global_step,
            )
            raise
        return outcome.trained, dict(outcome.metrics)

    def _speco_activate_drafter_training_model_before_fit(self) -> None:
        if not self.is_drafter_training_enabled(self.config):
            return
        self._speco_get_drafter_scheduler().activate_training_workers()

    def _speco_wait_pending_drafter_publish_rpc(self) -> int:
        if not self._pending_drafter_publish_refs:
            return 0
        pending_refs = self._pending_drafter_publish_refs
        self._pending_drafter_publish_refs = None
        self._ray_get_if_needed(pending_refs)
        return len(pending_refs) if isinstance(pending_refs, (list, tuple)) else 1

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
        method_name = (
            "update_draft_weights_async" if asynchronous else "update_draft_weights"
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
        event = (
            scheduler.on_after_weight_update(context)
            if after_weight_update
            else scheduler.on_safe_point(context)
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
        pending_drafter_publish = {
            "ready": False,
            "drafter_trained": False,
            "actor_output": None,
            "training_plan": None,
        }

        def generate_sequences_with_speco(manager_self, *args, **kwargs):
            self._speco_wait_pending_drafter_publish()
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            is_validation_generation = _speco_is_validation_generation(
                args, kwargs, gen_batch_output
            )
            if not is_validation_generation:
                self._speco_store_rollout_metrics(gen_batch_output)
                collected = self._speco_collect_generation_samples(gen_batch_output)
                if collected:
                    meta_info = getattr(gen_batch_output, "meta_info", None)
                    if isinstance(meta_info, dict):
                        meta_info.setdefault("metrics", {})[
                            "drafter/collected_samples"
                        ] = collected
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
                OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
                self._speco_oldlogprob_hidden_capture_impl(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
                self._speco_oldlogprob_hidden_layout(),
            )
            tu.assign_non_tensor_data(batch_td, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True)
            # Pass the user's sequence_parallel setting through the batch,
            # because MindSpeed repatch may override tf_config.sequence_parallel
            # back to True even when the user sets it to False.
            _actor_megatron_cfg = _get_nested(
                self.config, ("actor_rollout_ref", "actor", "megatron"), {}
            )
            _user_seq_parallel = _actor_megatron_cfg.get("sequence_parallel", True)
            tu.assign_non_tensor_data(
                batch_td,
                "speco_oldlogprob_sp_disabled",
                not bool(_user_seq_parallel),
            )

            self._speco_last_oldlogprob_prepare_elapsed_sec = (
                time.perf_counter() - prepare_started
            )
            compute_started = time.perf_counter()
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            self._speco_last_oldlogprob_compute_elapsed_sec = (
                time.perf_counter() - compute_started
            )
            collect_started = time.perf_counter()
            self._speco_collect_oldlogprob_features(batch, collect_plan, output)
            self._speco_last_oldlogprob_collect_elapsed_sec = (
                time.perf_counter() - collect_started
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
            pending_target_lm_head_sync = None
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
            collection_plan = getattr(self, "_speco_last_collection_plan", None)
            if isinstance(collection_plan, CollectionPlan):
                metrics.update(collection_plan.metrics())
            collection_outcome = getattr(self, "_speco_last_collection_outcome", None)
            if isinstance(collection_outcome, CollectionOutcome):
                metrics.update(collection_outcome.metrics())
            before_actor_event = self._speco_on_before_actor_update()
            training_plan = before_actor_event.training_plan
            if training_plan is None:
                raise RuntimeError(
                    "Drafter before-actor-update event returned no training plan"
                )
            self._speco_log_drafter_training_plan(training_plan)
            metrics.update(before_actor_event.metrics or {})
            metrics["drafter/train_interval_matched"] = int(
                training_plan.interval_matched
            )
            actor_started = time.perf_counter()
            actor_output = original_update_actor(*args, **kwargs)
            actor_elapsed = time.perf_counter() - actor_started
            pending_target_lm_head_sync = self._pending_target_lm_head_sync
            self._pending_target_lm_head_sync = None
            if pending_target_lm_head_sync is not None:
                metrics.update(
                    self._speco_finish_target_lm_head_weight_sync(
                        pending_target_lm_head_sync
                    )
                )
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
            if defer_publish_until_update_weights and drafter_trained:
                pending_drafter_publish["ready"] = True
                pending_drafter_publish["drafter_trained"] = drafter_trained
                pending_drafter_publish["actor_output"] = actor_output
                pending_drafter_publish["training_plan"] = training_plan
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
            ):
                value = _speco_metric_float(metrics.get(key))
                if value is not None:
                    known_drafter_timing += value
            metrics["timing_s/drafter_outer_unaccounted"] = max(
                0.0,
                metrics["timing_s/drafter"] - known_drafter_timing,
            )
            return self._speco_update_output_metrics(actor_output, metrics)

        def update_weights_with_speco(manager_self, *args, **kwargs):
            result = original_checkpoint_update_weights(*args, **kwargs)
            if pending_drafter_publish["ready"]:
                publish_metrics = self._speco_publish_drafter_weights(
                    pending_drafter_publish["drafter_trained"],
                    pending_drafter_publish["training_plan"],
                    after_weight_update=True,
                )
                self._speco_update_output_metrics(
                    pending_drafter_publish["actor_output"], publish_metrics
                )
                pending_drafter_publish["ready"] = False
                pending_drafter_publish["drafter_trained"] = False
                pending_drafter_publish["actor_output"] = None
                pending_drafter_publish["training_plan"] = None
            return result

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco,
            rollout_generation_target,
        )
        if (
            self._speco_oldlogprob_collection_requested()
            or self._speco_oldlogprob_entropy_hook_enabled()
        ):
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco, self
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
        self._speco_wait_pending_drafter_publish()
        self._speco_save_drafter_checkpoint(wait=True)
        return super()._save_checkpoint()

    def _validate(self, *args, **kwargs):
        # Validation commonly drives KV usage to the configured limit. Ensure
        # online weight loading and its temporary buffers have completed before
        # validation admits requests into the rollout engine.
        self._speco_wait_pending_drafter_publish()
        return super()._validate(*args, **kwargs)
