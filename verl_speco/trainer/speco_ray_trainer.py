"""SPECO adapter for verl v0.8.0 RayPPOTrainer."""

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from types import MethodType
from typing import Any

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
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_TIMING_KEY,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    assert_sglang_aux_last_layer_norm_safe,
    resolve_drafter_hidden_states_layout,
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.integration.sglang_adapter import (
    bucket_drafter_samples_by_replica,
    pop_drafter_samples,
    speco_step_matches_interval,
)
from verl_speco.integration.sglang_runtime import (
    clear_sglang_runtime_config,
    configure_sglang_runtime_from_config,
    install_upstream_sglang_runtime_bridge,
    should_install_sglang_base_compat_runtime,
)
from verl_speco.integration.vllm_runtime import SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX, configure_vllm_runtime_from_config
from verl_speco.workers import SpecoWorker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC = "drafter/spec_decode/mean_acceptance_length"
_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY = "_speco_vllm_spec_decode_drafts"
_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY = "_speco_vllm_spec_decode_accepted_tokens"
_SPECO_DRAFTER_TIMING_DEDUCTED_KEY = "_speco_drafter_timing_deducted_from_update_actor"
_DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"

_DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS = {None, "", "null", "None", "/path/to/drafter/checkpoint"}
_POLICY_MODEL_NON_TENSOR_KEYS = {"multi_modal_inputs", "pad_token_id"}


def _select_policy_model_batch(batch: DataProto) -> DataProto:
    """Keep rollout/drafter side-channel data out of policy-model forward paths."""
    non_tensor_batch_keys = [key for key in _POLICY_MODEL_NON_TENSOR_KEYS if key in batch.non_tensor_batch]
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
    mean_acceptance_length = _speco_metric_float(data.get(SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC))
    update_actor_elapsed = _speco_metric_float(data.get("timing_s/update_actor"))
    already_deducted = bool(data.get(_SPECO_DRAFTER_TIMING_DEDUCTED_KEY))
    if drafter_elapsed is None and mean_acceptance_length is None and not already_deducted:
        return data

    adjusted_update_actor = None
    adjusted_update_actor_per_token = None
    if drafter_elapsed is not None and update_actor_elapsed is not None and not already_deducted:
        adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
        update_actor_per_token = _speco_metric_float(data.get("timing_per_token_ms/update_actor"))
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
            rewritten[key] = adjusted_update_actor if adjusted_update_actor is not None else value
            if drafter_elapsed is not None:
                rewritten["timing_s/drafter"] = drafter_elapsed
            if mean_acceptance_length is not None:
                rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = mean_acceptance_length
            inserted_drafter_metrics = True
        elif key == "timing_per_token_ms/update_actor" and adjusted_update_actor_per_token is not None:
            rewritten[key] = adjusted_update_actor_per_token
        else:
            rewritten[key] = value
    if not inserted_drafter_metrics:
        if drafter_elapsed is not None:
            rewritten["timing_s/drafter"] = drafter_elapsed
        if mean_acceptance_length is not None:
            rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = mean_acceptance_length
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
        return _speco_float_values(non_tensor_batch.get(f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_{name}"))

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


def _speco_vllm_spec_decode_metrics_from_stats(stats: dict[str, float]) -> dict[str, float]:
    drafts = float(stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0)
    if drafts <= 0.0:
        return {}
    accepted_tokens = float(stats.get(_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY, 0.0) or 0.0)
    return {SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC: 1.0 + accepted_tokens / drafts}


def _speco_vllm_spec_decode_metrics_from_batch(batch: Any) -> dict[str, float]:
    return _speco_vllm_spec_decode_metrics_from_stats(_speco_vllm_spec_decode_stats_from_batch(batch))


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
    phase = str(
        meta_info.get("phase")
        or meta_info.get("split")
        or meta_info.get("mode")
        or meta_info.get("stage")
        or ""
    ).strip().lower()
    return phase in {"validate", "validation", "val", "test", "eval", "evaluation"}


def _speco_is_validation_generation(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any = None) -> bool:
    candidates = [output, *args]
    for key in ("batch", "prompts", "data", "input_batch"):
        if key in kwargs:
            candidates.append(kwargs[key])
    return any(_speco_is_validation_generation_value(candidate) for candidate in candidates)


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
        totals[key] = float((existing or {}).get(key, 0.0) or 0.0) + float(current.get(key, 0.0) or 0.0)
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
        self._pending_drafter_publish_refs = None
        self._pending_drafter_checkpoint_refs = []
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

    def attach_speco_worker_group(self, worker_group):
        self.drafter_wg = worker_group

    def _require_speco_worker_group(self):
        if self.drafter_wg is None:
            raise RuntimeError("SpecoWorker group has not been initialized yet.")
        return self.drafter_wg

    def speco_set_global_step(self, global_step: int):
        return self._require_speco_worker_group().set_global_step(global_step)

    def speco_collect_rollout_features(self, samples: list[list[dict]]):
        return self._require_speco_worker_group().collect_rollout_features(samples)

    def speco_sync_target_lm_head_weight(self, payload: Any, global_step: Any = None):
        return self._require_speco_worker_group().sync_target_lm_head_weight(payload, global_step=global_step)

    def speco_get_drafter_target_lm_head_row_indices(self):
        return self._require_speco_worker_group().get_drafter_target_lm_head_row_indices()

    def speco_train_drafter(self):
        return self._require_speco_worker_group().train_drafter()

    def speco_activate_drafter_training_model(self):
        return self._require_speco_worker_group().activate_drafter_training_model()

    def speco_maybe_publish(self):
        return self._require_speco_worker_group().maybe_publish()

    def speco_save_checkpoint(self, global_step: int, wait: bool = True):
        return self._require_speco_worker_group().save_checkpoint(global_step, wait=wait)

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
            if _get_nested(self.config, ("actor_rollout_ref", "rollout", "name"), None) == "sglang":
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

        rollout_config = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
        if rollout_config is None:
            yield
            return

        missing = object()
        original_agent = rollout_config.get("agent", missing) if hasattr(rollout_config, "get") else missing
        agent_config = original_agent if original_agent is not missing else {}
        previous_manager_class = (
            agent_config.get("agent_loop_manager_class", missing) if hasattr(agent_config, "get") else missing
        )
        with open_dict(rollout_config):
            if "agent" not in rollout_config or rollout_config["agent"] is None:
                rollout_config["agent"] = {}
            rollout_config["agent"]["agent_loop_manager_class"] = SPECO_AGENT_LOOP_MANAGER_CLASS
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
                    rollout_config["agent"]["agent_loop_manager_class"] = previous_manager_class

    @contextmanager
    def _hide_speco_drafter_config_from_upstream_rollout(self):
        rollout_config = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
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
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        drafter_cls = RayClassWithInitArgs(
            cls=speco_worker_cls,
            config=self.config.actor_rollout_ref,
            role="drafter",
            device_name=self.device_name,
        )

        self.drafter_wg = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=drafter_cls,
            name_prefix="speco_drafter",
            device_name=self.device_name,
        )
        self.drafter_wg.init_model()

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
                raise RuntimeError(f"Expected at most one non-null SPECO result, got {len(non_null)}")
            return non_null[0] if non_null else None
        return value

    def _speco_online_enabled(self) -> bool:
        return self.is_drafter_training_enabled(self.config)

    def _speco_drafter_training_config(self):
        return _get_nested(self.config, ("actor_rollout_ref", "rollout", "drafter", "training"), {})

    def _speco_drafter_config(self):
        return _get_nested(self.config, ("actor_rollout_ref", "rollout", "drafter"), None)

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

        default_local_dir = _get_nested(self.config, ("trainer", "default_local_dir"), None)
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
        resume_mode = str(_get_nested(trainer_cfg, ("resume_mode",), "disable") or "disable")
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
                    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

                    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Unable to resolve latest actor checkpoint for drafter resume: %s", exc)

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
            resume_setting = training_cfg.get("resume_lr_scheduler_from_checkpoint", True)
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
        resolved_path = resolve_drafter_checkpoint_path(model_path, checkpoint_path, resume_step)
        if resolved_path is None:
            return
        if os.path.normpath(resolved_path) == os.path.normpath(os.fspath(model_path or "")):
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
                flattened.extend(SpecoRayPPOTrainer._speco_flatten_checkpoint_results(item))
            return flattened
        return []

    @classmethod
    def _speco_validate_drafter_checkpoint_results(cls, value: Any, *, require_saved: bool) -> None:
        results = cls._speco_flatten_checkpoint_results(value)
        allowed_skips = {"not_checkpoint_replica", "not_in_training_group"}
        failures = [
            result
            for result in results
            if not bool(result.get("saved", False)) and result.get("reason") not in allowed_skips
        ]
        if failures:
            raise RuntimeError(f"Drafter checkpoint failed: {failures}")
        if require_saved and not any(bool(result.get("saved", False)) for result in results):
            raise RuntimeError(f"Drafter checkpoint produced no saved state: {results}")

    def _speco_save_drafter_checkpoint(self, *, wait: bool = True):
        if not self._speco_should_save_drafter_checkpoint():
            return None
        if self._speco_ensure_drafter_checkpoint_path() is None:
            return None
        checkpoint_refs = self.speco_save_checkpoint(self.global_steps, wait=wait)
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

    def _speco_should_collect_drafter_this_step(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return speco_step_matches_interval(self.global_steps, training_cfg.get("collect_interval_steps", 1))

    def _speco_should_train_drafter_this_step(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return speco_step_matches_interval(self.global_steps, training_cfg.get("training_interval_steps", 1))

    def _speco_drafter_training_mode(self) -> str:
        training_cfg = self._speco_drafter_training_config()
        return str(training_cfg.get("mode", "online") or "online").strip().lower()

    def _speco_has_collected_drafter_samples_this_step(self) -> bool:
        return int(getattr(self, "_speco_last_collected_samples", 0) or 0) > 0

    def _speco_should_attempt_drafter_train_this_step(self) -> bool:
        if self._speco_drafter_training_mode() == "collect_only":
            return False
        if not self._speco_should_train_drafter_this_step():
            return False
        if self._speco_has_collected_drafter_samples_this_step():
            return True
        training_cfg = self._speco_drafter_training_config()
        if self._speco_oldlogprob_collection_requested():
            return False
        return bool(training_cfg.get("use_data_buffer", False))

    def _speco_set_drafter_global_step(self, *, log_timing: bool = True):
        return self._ray_get_if_needed(self.speco_set_global_step(self.global_steps))

    def _speco_collect_rollout_features_rpc(self, source: str, buckets: list[list[dict]]):
        if not buckets or not any(bucket for bucket in buckets):
            return None
        rpc_started = time.perf_counter()
        result = self.speco_collect_rollout_features(buckets)
        if source == "oldlogprob":
            self._speco_last_oldlogprob_collect_rpc_elapsed_sec = time.perf_counter() - rpc_started
        return result

    def _speco_oldlogprob_collection_requested(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return bool(training_cfg.get("collect_hidden_states_from_old_logprob", False))

    def _speco_oldlogprob_collection_enabled(self) -> bool:
        if not self._speco_online_enabled() or not self._speco_oldlogprob_collection_requested():
            return False
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("collect_hidden_states_from_sgl", False)):
            raise ValueError(
                "SPECO old-logprob hidden collection requires "
                "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false"
            )
        if bool(training_cfg.get("use_logits", False)):
            raise ValueError("SPECO old-logprob hidden collection currently supports use_logits=false only")
        strategy = str(_get_nested(self.config, ("actor_rollout_ref", "actor", "strategy"), "") or "").lower()
        if strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(
                "SPECO old-logprob hidden collection currently supports actor.strategy=fsdp/fsdp2 only, "
                f"got {strategy!r}"
            )
        capture_impl = str(training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook") or "forward_hook")
        if capture_impl not in {"forward_hook", "output_hidden_states"}:
            raise ValueError(f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}")
        return True

    def _speco_oldlogprob_entropy_config_value(self):
        training_cfg = self._speco_drafter_training_config()
        value = training_cfg.get("old_logprob_calculate_entropy", None)
        if value is None:
            value = _get_nested(self.config, ("actor_rollout_ref", "actor", "calculate_entropy"), None)
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
        return str(training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook") or "forward_hook")

    def _speco_oldlogprob_hidden_layout(self) -> str:
        drafter_cfg = self._speco_drafter_config()
        algorithm = _get_nested(drafter_cfg, ("speculative_algorithm",), "")
        return resolve_drafter_hidden_states_layout(algorithm, self._speco_drafter_training_config())

    @staticmethod
    def _speco_oldlogprob_window_train_rows(training_cfg) -> int:
        window_rows = training_cfg.get("hidden_state_window_tokens_per_sample")
        if window_rows is None:
            window_rows = training_cfg.get("hidden_state_window_min_rows", 64)
        return int(window_rows or 0)

    @staticmethod
    def _speco_oldlogprob_window_mode(training_cfg) -> str:
        mode = str(training_cfg.get("hidden_state_window_mode", "front") or "front").strip().lower()
        if mode not in {"front", "random"}:
            return "front"
        return mode

    @staticmethod
    def _speco_oldlogprob_aux_layer_ids_from_config(config) -> Any | None:
        candidates = (
            ("model", "eagle_config", "target_hidden_layer_ids"),
            ("model", "eagle_config", "eagle_aux_hidden_state_layer_ids"),
            ("eagle_config", "target_hidden_layer_ids"),
            ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
            ("target_hidden_layer_ids",),
            ("eagle_aux_hidden_state_layer_ids",),
            ("target_layer_ids",),
        )
        for path in candidates:
            layer_ids = _get_nested(config, path, None)
            if layer_ids is not None:
                return layer_ids
        return None

    @staticmethod
    def _speco_normalize_oldlogprob_aux_layer_ids(layer_ids) -> list[int] | None:
        if layer_ids is None:
            return None
        if isinstance(layer_ids, int):
            return [int(layer_ids)]
        if isinstance(layer_ids, str):
            raw = layer_ids.strip()
            if not raw:
                return None
            if raw.startswith("["):
                layer_ids = json.loads(raw)
            else:
                layer_ids = [part.strip() for part in raw.split(",") if part.strip()]
        return [int(layer_id) for layer_id in list(layer_ids)]

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
        target_model_cfg = _get_nested(self.config, ("actor_rollout_ref", "model"), None)
        num_layers = self._speco_num_hidden_layers_from_config(target_model_cfg)
        if num_layers is not None:
            return num_layers
        target_model_path = _get_nested(target_model_cfg, ("path",), None)
        target_config = self._speco_load_model_config(target_model_path)
        return self._speco_num_hidden_layers_from_config(target_config)

    @staticmethod
    def _speco_default_eagle3_aux_layer_ids(num_hidden_layers: int) -> list[int]:
        num_hidden_layers = int(num_hidden_layers)
        if num_hidden_layers <= 0:
            raise RuntimeError(f"SPECO cannot derive EAGLE3 aux hidden layers from num_hidden_layers={num_hidden_layers}")
        return [2, num_hidden_layers // 2, num_hidden_layers - 3]

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
            model_config = self._speco_load_model_config(_get_nested(drafter_cfg, (path_key,), None))
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
            allow_prenorm_last=bool(training_cfg.get("allow_sglang_prenorm_last_layer", False)),
        )

    def _speco_oldlogprob_aux_layer_ids(self) -> list[int]:
        drafter_cfg = self._speco_drafter_config()
        model_configs = []
        for path_key in ("model_path", "checkpoint_path"):
            model_config = self._speco_load_model_config(_get_nested(drafter_cfg, (path_key,), None))
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
        return int.from_bytes(digest, byteorder="big", signed=False) % (inclusive_max + 1)

    def _speco_build_oldlogprob_collect_plan(self, batch: DataProto) -> dict[str, Any] | None:
        if not self._speco_oldlogprob_collection_enabled():
            return None
        if not self._speco_should_collect_drafter_this_step():
            return None
        if not self._speco_should_train_drafter_this_step():
            return None
        training_cfg = self._speco_drafter_training_config()
        sample_rate = float(training_cfg.get("collection_sample_rate", 1.0) or 0.0)
        if sample_rate <= 0:
            return None
        window_mode = self._speco_oldlogprob_window_mode(training_cfg)

        batch_tensors = batch.batch
        required_keys = ("prompts", "responses", "attention_mask")
        if any(key not in batch_tensors for key in required_keys):
            return None
        prompts = batch_tensors["prompts"]
        responses = batch_tensors["responses"]
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
        max_per_owner = training_cfg.get("max_collect_samples_per_step_per_replica", 16)
        max_per_owner = int(max_per_owner) if max_per_owner is not None else batch_size
        max_per_owner = max(max_per_owner, 0)
        max_tokens_per_owner = training_cfg.get("max_collect_tokens_per_step_per_replica", None)
        max_tokens_per_owner = int(max_tokens_per_owner) if max_tokens_per_owner is not None else None
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
            prompt_len = int(attention_mask[batch_idx, :prompt_width].detach().sum().item())
            if response_mask is not None:
                response_len = int(response_mask[batch_idx].detach().sum().item())
            else:
                response_len = int(attention_mask[batch_idx, prompt_width:].detach().sum().item())
            prompt_lens.append(prompt_len)
            response_lens.append(response_len)
            if prompt_len <= 0 or response_len < hidden_rows:
                continue
            candidate_count += 1
            sample_key = f"{step_key}:{batch_idx}:{prompt_len}:{response_len}"
            if sample_rate < 1.0 and self._speco_hash_fraction(sample_key) >= sample_rate:
                continue
            owner = selected_count % owner_count
            if owner_counts[owner] >= max_per_owner:
                continue
            if max_tokens_per_owner is not None and owner_token_counts[owner] + hidden_rows > max_tokens_per_owner:
                continue
            max_start_offset = max(response_len - hidden_rows, 0)
            if window_mode == "random":
                random_offset = self._speco_hash_int(f"{sample_key}:window", max_start_offset)
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
            rows = [row.reshape(-1).float() for row in tensor.unbind() if row.numel() > 0]
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
        hidden_refs = self._speco_flatten_non_tensor_rows(tu.get(output, OLD_LOGPROB_HIDDEN_REFS_KEY))
        hidden_ref_meta = self._speco_flatten_non_tensor_rows(tu.get(output, OLD_LOGPROB_HIDDEN_REF_META_KEY))
        chunk_refs = self._speco_flatten_non_tensor_rows(tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY))
        chunk_meta = self._speco_flatten_non_tensor_rows(tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_META_KEY))
        if hidden_states is None and hidden_refs is None and chunk_refs is None:
            return 0
        hidden_rows = self._speco_tensor_rows(hidden_states)
        if not hidden_rows and not hidden_refs and not chunk_refs:
            return 0
        timing = self._speco_sum_timing_rows(tu.get(output, OLD_LOGPROB_TIMING_KEY))
        if timing is not None and int(timing.numel()) >= 2:
            self._speco_last_oldlogprob_select_elapsed_sec = float(timing[0].item()) / 1_000_000.0
            self._speco_last_oldlogprob_sp_merge_elapsed_sec = float(timing[1].item()) / 1_000_000.0
            if int(timing.numel()) >= 5:
                self._speco_last_oldlogprob_concat_elapsed_sec = float(timing[2].item()) / 1_000_000.0
                self._speco_last_oldlogprob_cpu_copy_elapsed_sec = float(timing[3].item()) / 1_000_000.0
                self._speco_last_oldlogprob_ray_put_elapsed_sec = float(timing[4].item()) / 1_000_000.0

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask_tensor = batch.batch.get("response_mask", None)
        collect_mask = collect_plan["collect_mask"]
        hidden_positions = collect_plan["hidden_positions"]
        owner_rank = collect_plan["owner_rank"]
        prompt_lens = collect_plan["prompt_lens"]
        response_lens = collect_plan["response_lens"]
        buckets = [[] for _ in range(int(collect_plan["owner_count"]))]
        collected_rows = 0
        payload_bytes = 0
        sample_ref_chunks: dict[int, list[dict[str, Any]]] = {}
        if isinstance(chunk_refs, (list, tuple)) and isinstance(chunk_meta, (list, tuple)):
            for chunk_index, (chunk_ref, chunk_info) in enumerate(zip(chunk_refs, chunk_meta, strict=False)):
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
                    row_indices = row_indices_payload[item_idx] if item_idx < len(row_indices_payload) else None
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
            if batch_idx >= int(collect_mask.numel()) or not bool(collect_mask[batch_idx].item()):
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
                collected_rows += sum(_speco_ref_meta_row_count(chunk, 0) for chunk in ref_chunks)
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
                response_mask = attention_mask[batch_idx, prompts.size(1) : prompts.size(1) + responses.size(1)].bool()
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
                sample["hidden_states"] = hidden.detach().cpu().unsqueeze(0)
            else:
                sample["hidden_states_ref"] = hidden_ref
                sample["hidden_states_ref_meta"] = ref_meta
            buckets[owner].append(sample)

        collected = sum(len(bucket) for bucket in buckets)
        if collected <= 0:
            return 0
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        if dispatch_bucket_count is not None and dispatch_bucket_count > len(buckets):
            buckets.extend([[] for _ in range(dispatch_bucket_count - len(buckets))])
        self._speco_set_drafter_global_step(log_timing=False)
        self._speco_collect_rollout_features_rpc("oldlogprob", buckets)
        self._speco_last_collected_samples = collected
        self._speco_last_oldlogprob_collected_samples = collected
        self._speco_last_oldlogprob_collected_rows = collected_rows
        self._speco_last_oldlogprob_payload_mib = payload_bytes / float(1024 * 1024)
        return collected

    def _speco_num_rollout_replicas(self, samples: list[dict]) -> int:
        sample_max = max((int(sample.get("replica_rank", 0)) for sample in samples), default=0) + 1
        rollout_cfg = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
        rollout_dp = int(_get_nested(rollout_cfg, ("data_parallel_size",), 1) or 1)
        return max(sample_max, rollout_dp, 1)

    def _speco_collect_generation_samples(self, gen_batch_output: Any) -> int:
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_collect_interval_matched = int(self._speco_should_collect_drafter_this_step())
        if not self._speco_online_enabled():
            return 0
        samples = pop_drafter_samples(gen_batch_output)
        self._speco_last_raw_drafter_samples = len(samples)
        if not samples:
            return 0
        if not self._speco_should_collect_drafter_this_step():
            return 0

        num_replicas = self._speco_num_rollout_replicas(samples)
        buckets = bucket_drafter_samples_by_replica(samples, num_replicas=num_replicas)

        # Some verl worker-group dispatch meshes may expose more owner buckets
        # than rollout replicas. Extend with empty buckets when that mapping is
        # discoverable, while keeping sample validation based on replicas.
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        if dispatch_bucket_count is not None and dispatch_bucket_count > len(buckets):
            buckets.extend([[] for _ in range(dispatch_bucket_count - len(buckets))])

        self._speco_set_drafter_global_step()
        self._speco_collect_rollout_features_rpc("rollout", buckets)
        self._speco_last_collected_samples = len(samples)
        return len(samples)

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
        algorithm = str(_get_nested(drafter_cfg, ("speculative_algorithm",), "") or "").upper()
        if algorithm == "DSPARK" and float(training_cfg.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0:
            return None
        if not bool(training_cfg.get("target_lm_head_row_restricted_sync", True)):
            return None

        row_infos = self._ray_get_if_needed(self.speco_get_drafter_target_lm_head_row_indices()) or []
        non_null_infos = [info for info in row_infos if isinstance(info, dict) and info.get("row_indices") is not None]
        if not non_null_infos:
            return None
        source_vocab_sizes = {int(info.get("source_vocab_size")) for info in non_null_infos if info.get("source_vocab_size") is not None}
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
        union_rows = torch.unique(torch.cat(row_tensors), sorted=True).to(dtype=torch.long).contiguous()
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
        if target_sync_mapping is None and hasattr(worker_group, "_query_dispatch_info"):
            target_sync_mapping = worker_group._query_dispatch_info(_DRAFTER_TARGET_SYNC_MESH)
            if isinstance(dispatch_info, dict):
                dispatch_info[_DRAFTER_TARGET_SYNC_MESH] = target_sync_mapping
        if not target_sync_mapping:
            return payload, self.global_steps, 1

        target_sync_bucket_count = max(int(dp_rank) for dp_rank in target_sync_mapping) + 1
        payload_buckets = [payload for _ in range(target_sync_bucket_count)]
        global_step_buckets = [self.global_steps for _ in range(target_sync_bucket_count)]
        return payload_buckets, global_step_buckets, target_sync_bucket_count

    def _speco_sync_target_lm_head_weight(self) -> dict[str, Any]:
        sync_started = time.perf_counter()
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("use_logits", False)):
            return {"drafter/target_lm_head_synced": 0}
        if not self._speco_should_train_drafter_this_step():
            return {"drafter/target_lm_head_synced": 0}

        row_selection = self._speco_get_drafter_target_lm_head_row_selection()
        row_indices = row_selection.get("row_indices") if row_selection is not None else None
        selected_rows = int(row_selection.get("selected_rows", 0) or 0) if row_selection is not None else 0
        source_vocab_size = (
            int(row_selection.get("source_vocab_size", 0) or 0) if row_selection is not None else 0
        )
        get_actor_lm_head_weight = self._speco_actor_rollout_method("get_actor_lm_head_weight")
        fetch_started = time.perf_counter()
        payloads = self._ray_get_if_needed(get_actor_lm_head_weight(row_indices)) or []
        fetch_elapsed = time.perf_counter() - fetch_started
        payload = self._first_non_null(payloads)
        if payload is None:
            return {
                "drafter/target_lm_head_synced": 0,
                "drafter/target_lm_head_selected_rows": selected_rows,
                "drafter/target_lm_head_source_vocab_size": source_vocab_size,
                "timing_s/drafter_sync_target_lm_head": time.perf_counter() - sync_started,
                "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            }

        payload_arg, global_step_arg, _ = self._speco_build_drafter_target_lm_head_sync_args(payload)
        apply_started = time.perf_counter()
        self._ray_get_if_needed(self.speco_sync_target_lm_head_weight(payload_arg, global_step=global_step_arg))
        apply_elapsed = time.perf_counter() - apply_started
        export_strategy = str(payload.get("export_strategy", "unknown")) if isinstance(payload, dict) else "unknown"
        return {
            "drafter/target_lm_head_synced": 1,
            "drafter/target_lm_head_selected_rows": selected_rows,
            "drafter/target_lm_head_source_vocab_size": source_vocab_size,
            "drafter/target_lm_head_direct_sparse_export": int(export_strategy == "direct_sparse"),
            "timing_s/drafter_sync_target_lm_head": time.perf_counter() - sync_started,
            "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            "timing_s/drafter_sync_target_lm_head_apply": apply_elapsed,
        }

    def _speco_train_drafter(self) -> tuple[bool, dict[str, Any]]:
        train_rpc_started = time.perf_counter()
        train_results = self._ray_get_if_needed(self.speco_train_drafter()) or []
        train_rpc_elapsed = time.perf_counter() - train_rpc_started
        if not isinstance(train_results, list):
            train_results = [train_results]

        normalized_results = []
        for result in train_results:
            if isinstance(result, dict):
                normalized_results.append(result)
            else:
                normalized_results.append(
                    {
                        "trained": bool(result),
                        "triggered": bool(result),
                        "attempted_steps": int(bool(result)),
                        "successful_steps": int(bool(result)),
                        "elapsed_sec": 0.0,
                        "reason": "legacy_bool_result",
                    }
                )

        trained = any(bool(result.get("trained", False)) for result in normalized_results)
        successful_steps_max = max((int(result.get("successful_steps", 0)) for result in normalized_results), default=0)

        metrics = {
            "drafter/trained": int(trained),
            "drafter/train_successful_steps_max": successful_steps_max,
            "drafter/train_no_trainable_batch": int(
                any(result.get("reason") == "no_trainable_batch" for result in normalized_results)
            ),
            "drafter/train_activation_failed": int(
                any(result.get("reason") == "activation_failed" for result in normalized_results)
            ),
        }
        for key in (
            "timing_s/drafter_prepare_batch",
            "timing_s/drafter_forward_loss",
            "timing_s/drafter_reduce_loss",
            "timing_s/drafter_backward",
            "timing_s/drafter_optimizer",
            "timing_s/drafter_publish_snapshot",
            "activation_elapsed_sec",
            "training_loop_elapsed_sec",
            "cleanup_elapsed_sec",
            "elapsed_sec",
        ):
            values = []
            for result in normalized_results:
                value = _speco_metric_float(result.get(key))
                if value is not None:
                    values.append(value)
            if values:
                metric_key = {
                    "activation_elapsed_sec": "timing_s/drafter_worker_activation",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                }.get(key, key)
                metrics[metric_key] = max(values)
        metrics["timing_s/drafter_train_rpc"] = train_rpc_elapsed
        return trained, metrics

    def _speco_activate_drafter_training_model_before_fit(self) -> None:
        if not self.is_drafter_training_enabled(self.config):
            return
        activation_results = self._ray_get_if_needed(self.speco_activate_drafter_training_model()) or []
        if not isinstance(activation_results, list):
            activation_results = [activation_results]

        active_results = [
            result
            for result in activation_results
            if isinstance(result, dict) and result.get("reason") not in {"disabled", "not_in_training_group"}
        ]
        failed_results = [result for result in active_results if not bool(result.get("activated", False))]
        if failed_results:
            raise RuntimeError(
                "SPECO drafter trainer activation failed before RL fit: "
                f"{failed_results[:3]}"
            )

    def _speco_should_publish_drafter_weights(self, drafter_trained: bool) -> bool:
        if not drafter_trained:
            return False
        training_cfg = self._speco_drafter_training_config()
        publish_interval = int(training_cfg.get("publish_interval_steps", 0) or 0)
        return publish_interval <= 0 or self.global_steps % publish_interval == 0

    def _speco_wait_pending_drafter_publish(self) -> int:
        if not self._pending_drafter_publish_refs:
            return 0
        pending_refs = self._pending_drafter_publish_refs
        self._pending_drafter_publish_refs = None
        self._ray_get_if_needed(pending_refs)
        return len(pending_refs) if isinstance(pending_refs, (list, tuple)) else 1

    def _speco_get_published_drafter_weights(self):
        published = self._ray_get_if_needed(self.speco_maybe_publish()) or []
        return self._first_non_null(published)

    def _speco_publish_drafter_weights(self, drafter_trained: bool) -> dict[str, Any]:
        if not self._speco_should_publish_drafter_weights(drafter_trained):
            return {"drafter/publish_attempted": 0, "drafter/published": 0}

        wait_started = time.perf_counter()
        self._speco_wait_pending_drafter_publish()
        wait_elapsed = time.perf_counter() - wait_started
        fetch_started = time.perf_counter()
        published = self._speco_get_published_drafter_weights()
        fetch_elapsed = time.perf_counter() - fetch_started
        metrics = {
            "drafter/publish_attempted": 1,
            "timing_s/drafter_publish_wait_pending": wait_elapsed,
            "timing_s/drafter_publish_fetch_snapshot": fetch_elapsed,
        }
        if published is None:
            metrics["drafter/published"] = 0
            return metrics

        payload = resolve_drafter_publish_payload(published)
        training_cfg = self._speco_drafter_training_config()
        publish_async = bool(training_cfg.get("publish_async", False))
        method_name = "update_draft_weights_async" if publish_async else "update_draft_weights"
        update_draft_weights = self._speco_actor_rollout_method(method_name)
        update_started = time.perf_counter()
        update_result = update_draft_weights(payload, global_steps=self.global_steps)
        if publish_async:
            self._pending_drafter_publish_refs = update_result
        else:
            self._ray_get_if_needed(update_result)
        update_elapsed = time.perf_counter() - update_started

        metrics.update(
            {
                "drafter/published": 1,
                "timing_s/drafter_publish_update_weights": update_elapsed,
            }
        )
        return metrics

    def _speco_update_output_metrics(self, output: Any, metrics: dict[str, Any]):
        if not metrics:
            return output
        meta_info = getattr(output, "meta_info", None)
        if isinstance(meta_info, dict):
            output_metrics = meta_info.setdefault("metrics", {})
            output_metrics.update(metrics)
            drafter_elapsed = _speco_metric_float(output_metrics.get("timing_s/drafter"))
            update_actor_elapsed = _speco_metric_float(output_metrics.get("timing_s/update_actor"))
            if drafter_elapsed is not None and update_actor_elapsed is not None:
                adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
                update_actor_per_token = _speco_metric_float(output_metrics.get("timing_per_token_ms/update_actor"))
                if update_actor_per_token is not None:
                    output_metrics["timing_per_token_ms/update_actor"] = (
                        update_actor_per_token * adjusted_update_actor / update_actor_elapsed
                        if update_actor_elapsed > 0
                        else 0.0
                    )
                output_metrics["timing_s/update_actor"] = adjusted_update_actor
                output_metrics[_SPECO_DRAFTER_TIMING_DEDUCTED_KEY] = True
        return output

    def _speco_rollout_generation_target(self):
        for attr_name in ("async_rollout_manager", "actor_rollout_wg"):
            target = getattr(self, attr_name, None)
            if target is not None and callable(getattr(target, "generate_sequences", None)):
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
        if getattr(self, "_speco_last_rollout_metrics_step", None) != getattr(self, "global_steps", None):
            return {}
        return _speco_vllm_spec_decode_metrics_from_stats(getattr(self, "_speco_last_rollout_metrics", None) or {})

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

    @contextmanager
    def _speco_tracking_metrics_hook(self):
        try:
            from verl.utils.tracking import Tracking
        except ImportError:
            yield
            return

        original_log = getattr(Tracking, "log", None)
        if not callable(original_log) or getattr(original_log, "_speco_drafter_timing_hook", False):
            yield
            return

        def log_with_speco_metrics(tracking_self, *args, **kwargs):
            latest_rollout_metrics = self._speco_current_step_rollout_metrics()
            if "data" in kwargs:
                kwargs = dict(kwargs)
                data = kwargs["data"]
                if (
                    isinstance(data, dict)
                    and isinstance(latest_rollout_metrics, dict)
                    and data.get("training/global_step") == self.global_steps
                ):
                    data = dict(data)
                    data.update(latest_rollout_metrics)
                kwargs["data"] = _speco_move_drafter_timing_next_to_update_actor(data)
                return original_log(tracking_self, *args, **kwargs)
            if args:
                data = args[0]
                if (
                    isinstance(data, dict)
                    and isinstance(latest_rollout_metrics, dict)
                    and data.get("training/global_step") == self.global_steps
                ):
                    data = dict(data)
                    data.update(latest_rollout_metrics)
                args = (_speco_move_drafter_timing_next_to_update_actor(data), *args[1:])
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
        tu.assign_non_tensor(batch_td, calculate_entropy=calculate_entropy, compute_loss=False)

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
                {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
            )
        else:
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    @contextmanager
    def _speco_oldlogprob_entropy_fit_hook(self):
        original_compute_old_log_prob = self._compute_old_log_prob

        def compute_old_log_prob_without_forced_entropy(trainer_self, batch: DataProto):
            return self._speco_compute_old_log_prob_without_forced_entropy(batch)

        self._compute_old_log_prob = MethodType(compute_old_log_prob_without_forced_entropy, self)
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
            getattr(checkpoint_manager, "update_weights", None) if checkpoint_manager is not None else None
        )
        defer_publish_until_update_weights = callable(original_checkpoint_update_weights)
        pending_drafter_publish = {
            "ready": False,
            "drafter_trained": False,
            "actor_output": None,
        }

        def generate_sequences_with_speco(manager_self, *args, **kwargs):
            self._speco_wait_pending_drafter_publish()
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            is_validation_generation = _speco_is_validation_generation(args, kwargs, gen_batch_output)
            if not is_validation_generation:
                self._speco_store_rollout_metrics(gen_batch_output)
                collected = self._speco_collect_generation_samples(gen_batch_output)
                if collected:
                    meta_info = getattr(gen_batch_output, "meta_info", None)
                    if isinstance(meta_info, dict):
                        meta_info.setdefault("metrics", {})["drafter/collected_samples"] = collected
            return gen_batch_output

        def compute_old_log_prob_with_speco(trainer_self, batch: DataProto):
            if not self._speco_oldlogprob_collection_enabled():
                if self._speco_oldlogprob_entropy_hook_enabled():
                    return self._speco_compute_old_log_prob_without_forced_entropy(batch)
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
            collect_interval_matched = self._speco_should_collect_drafter_this_step()
            train_interval_matched = self._speco_should_train_drafter_this_step()
            self._speco_last_collect_interval_matched = int(collect_interval_matched)
            prepare_started = time.perf_counter()
            original_batch = batch

            def compute_old_log_prob_without_collection():
                self._speco_last_oldlogprob_prepare_elapsed_sec = time.perf_counter() - prepare_started
                compute_started = time.perf_counter()
                if self._speco_oldlogprob_entropy_hook_enabled():
                    old_log_prob, old_log_prob_mfu = self._speco_compute_old_log_prob_without_forced_entropy(
                        original_batch
                    )
                else:
                    old_log_prob, old_log_prob_mfu = original_compute_old_log_prob(original_batch)
                self._speco_last_oldlogprob_compute_elapsed_sec = time.perf_counter() - compute_started
                self._speco_last_oldlogprob_total_elapsed_sec = time.perf_counter() - oldlogprob_started
                return old_log_prob, old_log_prob_mfu

            if not collect_interval_matched or not train_interval_matched:
                return compute_old_log_prob_without_collection()

            batch = _select_policy_model_batch(batch)
            collect_plan = self._speco_build_oldlogprob_collect_plan(batch)
            if collect_plan is None:
                return compute_old_log_prob_without_collection()
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self._speco_oldlogprob_calculate_entropy()
            tu.assign_non_tensor(batch_td, calculate_entropy=calculate_entropy, compute_loss=False)
            batch_td[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
            batch_td[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan["hidden_positions"]
            batch_td[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan["hidden_position_mask"]
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

            self._speco_last_oldlogprob_prepare_elapsed_sec = time.perf_counter() - prepare_started
            compute_started = time.perf_counter()
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            self._speco_last_oldlogprob_compute_elapsed_sec = time.perf_counter() - compute_started
            collect_started = time.perf_counter()
            self._speco_collect_oldlogprob_features(batch, collect_plan, output)
            self._speco_last_oldlogprob_collect_elapsed_sec = time.perf_counter() - collect_started

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
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
            self._speco_last_oldlogprob_total_elapsed_sec = time.perf_counter() - oldlogprob_started
            return old_log_prob, old_log_prob_mfu

        def update_actor_with_speco(trainer_self, *args, **kwargs):
            update_actor_started = time.perf_counter()
            metrics = {
                "drafter/raw_drafter_samples": int(getattr(self, "_speco_last_raw_drafter_samples", 0)),
                "drafter/collected_samples": int(getattr(self, "_speco_last_collected_samples", 0)),
                "drafter/collect_interval_matched": int(
                    getattr(self, "_speco_last_collect_interval_matched", 0)
                ),
                "drafter/train_interval_matched": int(self._speco_should_train_drafter_this_step()),
            }
            should_train_drafter = self._speco_should_attempt_drafter_train_this_step()
            if should_train_drafter:
                self._speco_set_drafter_global_step()
                metrics.update(self._speco_sync_target_lm_head_weight())
            else:
                metrics["drafter/target_lm_head_synced"] = 0
            actor_started = time.perf_counter()
            actor_output = original_update_actor(*args, **kwargs)
            actor_elapsed = time.perf_counter() - actor_started
            if should_train_drafter:
                drafter_trained, train_metrics = self._speco_train_drafter()
            else:
                drafter_trained, train_metrics = (
                    False,
                    {
                        "drafter/trained": 0,
                        "drafter/train_successful_steps_max": 0,
                        "drafter/train_no_trainable_batch": int(self._speco_should_train_drafter_this_step()),
                        "drafter/train_activation_failed": 0,
                    },
                )
            metrics.update(train_metrics)
            if defer_publish_until_update_weights and drafter_trained:
                pending_drafter_publish["ready"] = True
                pending_drafter_publish["drafter_trained"] = drafter_trained
                pending_drafter_publish["actor_output"] = actor_output
            else:
                metrics.update(self._speco_publish_drafter_weights(drafter_trained))
            metrics["timing_s/drafter"] = max(0.0, time.perf_counter() - update_actor_started - actor_elapsed)
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
                publish_metrics = self._speco_publish_drafter_weights(pending_drafter_publish["drafter_trained"])
                self._speco_update_output_metrics(pending_drafter_publish["actor_output"], publish_metrics)
                pending_drafter_publish["ready"] = False
                pending_drafter_publish["drafter_trained"] = False
                pending_drafter_publish["actor_output"] = None
            return result

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco,
            rollout_generation_target,
        )
        if self._speco_oldlogprob_collection_requested():
            self._compute_old_log_prob = MethodType(compute_old_log_prob_with_speco, self)
        elif self._speco_oldlogprob_entropy_hook_enabled():
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco,
                self,
            )
        self._update_actor = MethodType(update_actor_with_speco, self)
        if defer_publish_until_update_weights:
            checkpoint_manager.update_weights = MethodType(update_weights_with_speco, checkpoint_manager)
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
        return bool(_get_nested(config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False))

    @staticmethod
    def is_drafter_training_enabled(config) -> bool:
        drafter_enabled = bool(_get_nested(config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False))
        training_enabled = bool(
            _get_nested(config, ("actor_rollout_ref", "rollout", "drafter", "enable_drafter_training"), False)
        )
        return drafter_enabled and training_enabled

    def fit(self):
        try:
            if self.is_drafter_training_enabled(self.config):
                self._speco_activate_drafter_training_model_before_fit()
                with self._speco_tracking_metrics_hook(), self._speco_online_fit_hooks():
                    return super().fit()
            if self.is_drafter_rollout_enabled(self.config):
                with self._speco_tracking_metrics_hook(), self._speco_rollout_metrics_fit_hook():
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
        self._speco_save_drafter_checkpoint(wait=True)
        return super()._save_checkpoint()
