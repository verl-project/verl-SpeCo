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

import dataclasses
import hashlib
import json
import logging
import math
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
    OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY,
    OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY,
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
from verl_speco.integration.sglang_adapter import (
    pop_drafter_samples,
    speco_step_matches_interval,
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
from verl_speco.trainer.hard_sample_selection import partition_hard_candidates
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
    step_matches_interval,
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
_SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY = "_speco_vllm_request_verify_rounds"
_SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY = "_speco_vllm_request_accepted_tokens"
_SPECO_VLLM_REQUEST_DRAFT_TOKENS_KEY = "_speco_vllm_request_draft_tokens"
_SPECO_VLLM_REQUEST_INVALID_TOKENS_KEY = "_speco_vllm_request_invalid_spec_tokens"
_SPECO_VLLM_REQUEST_COMPLETION_INDEX_KEY = "_speco_vllm_request_completion_index"
_SPECO_VLLM_REQUEST_ELAPSED_SEC_KEY = "_speco_vllm_request_elapsed_sec"
_SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY = "_verl_request_mean_accept_len"
_SPECO_VLLM_REQUEST_IS_HARD_KEY = "_verl_is_hard"
_SPECO_VLLM_REQUEST_HARD_SCORE_KEY = "_verl_hard_score"
_SPECO_VLLM_REQUEST_ID_KEY = "_speco_vllm_request_id"
_SPECO_DRAFTER_TIMING_DEDUCTED_KEY = "_speco_drafter_timing_deducted_from_update_actor"
# Meta-info markers carried by fixed paired-probe batches (RFC sec. 6).
_SPECO_FREEZE_PROBE_META_KEY = "_speco_freeze_probe"
_SPECO_FREEZE_PROBE_MAX_TOKENS_META_KEY = "_speco_freeze_probe_max_tokens"
_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH_ENV = (
    "VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH"
)
_SPECO_REQUEST_ACCEPT_LEN_HIST_PRINT_ENV = "VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_PRINT"
_SPECO_REQUEST_ACCEPT_LEN_HIST_BIN_WIDTH_ENV = (
    "VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_BIN_WIDTH"
)
_DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"

_DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS = {
    None,
    "",
    "null",
    "None",
    "/path/to/drafter/checkpoint",
}
_POLICY_MODEL_NON_TENSOR_KEYS = {
    "multi_modal_inputs",
    "pad_token_id",
    "_speco_vllm_request_id",
    "_speco_vllm_request_verify_rounds",
    "_speco_vllm_request_accepted_tokens",
    "_speco_vllm_request_draft_tokens",
    "_speco_vllm_request_invalid_spec_tokens",
    "_speco_vllm_request_completion_index",
    "_speco_vllm_request_elapsed_sec",
    "_verl_request_mean_accept_len",
}


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


def _speco_optional_float(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` (filters NaN/Inf unlike metric_float)."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _speco_sequence_values(value: Any, size: int) -> list[Any]:
    """Normalize a per-request field into a length-``size`` list (None-padded)."""
    if value is None:
        return [None for _ in range(size)]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [value[index] if index < len(value) else None for index in range(size)]


def _speco_request_cell_float(value: Any) -> float | None:
    """Coerce one non_tensor_batch cell to float, unwrapping nested lists.

    Agent-loop object-array cells hold the server's per-completion lists
    (e.g. ``[2.0]`` for n=1); drill into singleton lists before delegating
    to the regular finite-float coercion.
    """
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return _speco_optional_float(value)


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


def _speco_diag_enabled(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "off", "no", "n", ""}


def _speco_append_jsonl_batch(path: str | None, payloads: list[dict[str, Any]]) -> None:
    if not path:
        return
    if not payloads:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(
                "".join(
                    json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
                    for payload in payloads
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to append SPECO diagnostic log %s: %s", path, exc)


def _speco_accept_len_quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _speco_accept_len_hist_bin_width() -> float:
    value = _speco_optional_float(
        os.getenv(_SPECO_REQUEST_ACCEPT_LEN_HIST_BIN_WIDTH_ENV, "0.1")
    )
    return value if value is not None and value > 0 else 0.1


def _speco_accept_len_histogram(
    values: list[float], *, bin_width: float
) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    for value in values:
        index = math.floor(value / bin_width)
        counts[index] = counts.get(index, 0) + 1
    return [
        {
            "left": round(index * bin_width, 6),
            "right": round((index + 1) * bin_width, 6),
            "center": round((index + 0.5) * bin_width, 6),
            "count": count,
        }
        for index, count in sorted(counts.items())
    ]


def _speco_request_accept_len_step_payload(
    *,
    step: int,
    source: str,
    records: list[dict[str, Any]],
    accept_len_var: float,
) -> dict[str, Any] | None:
    values = [
        float(record["mean_accept_len"])
        for record in records
        if _speco_optional_float(record.get("mean_accept_len")) is not None
    ]
    if not values:
        return None
    sorted_values = sorted(values)
    bin_width = _speco_accept_len_hist_bin_width()
    mean_accept_len = sum(values) / len(values)
    p50 = cast(float, _speco_accept_len_quantile(sorted_values, 0.50))
    p90 = cast(float, _speco_accept_len_quantile(sorted_values, 0.90))
    p95 = cast(float, _speco_accept_len_quantile(sorted_values, 0.95))
    return {
        "step": int(step),
        "source": source,
        "count": len(values),
        "accept_lens": [round(value, 6) for value in values],
        "summary": {
            "min": round(sorted_values[0], 6),
            "max": round(sorted_values[-1], 6),
            "mean": round(mean_accept_len, 6),
            "p50": round(p50, 6),
            "p90": round(p90, 6),
            "p95": round(p95, 6),
            "var": round(float(accept_len_var), 6),
        },
        "histogram": {
            "bin_width": round(bin_width, 6),
            "x": "mean_accept_len",
            "y": "request_count",
            "bins": _speco_accept_len_histogram(values, bin_width=bin_width),
        },
    }


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
        # ``_speco_drafter_frozen`` is the single gate that suppresses drafter
        # training; it is driven by the marginal-utility freeze policy in
        # active mode (``_speco_observe_rollout_evidence``).
        self._speco_drafter_frozen = False
        self._speco_last_request_accept_len_records: list[dict[str, Any]] = []
        # Marginal-utility freeze policy (method=marginal_utility_v1). Shadow
        # mode only logs would_freeze + metrics; active mode drives the
        # ``_speco_drafter_frozen`` gate that stops both drafter training and
        # hidden-state collection (accept-length / PPO old_log_prob stay on).
        self._speco_freeze_policy = None  # built lazily in fit()
        self._speco_freeze_policy_active = False
        self._speco_last_freeze_decision = None
        self._speco_freeze_state_sidecar = "speco_drafter_freeze_state.json"
        # Freeze-transition fork checkpoint
        # (freeze_transition_branch_checkpoint_design.md). The request is
        # recorded in the generate hook (intent only, no I/O) and consumed at
        # the post-actor-update extension point, never inside the rollout
        # callback. ``once`` semantics: at most one fork per run.
        self._speco_freeze_branch_save_pending = None
        self._speco_freeze_branch_saved = False
        # Coalesces an event save and the base loop's periodic save landing on
        # the same step: the full save chain runs at most once per global step.
        self._speco_last_checkpoint_saved_step = None
        self._speco_pending_request_accept_len_payloads = []
        self._speco_last_request_accept_len_var = 0.0
        self._speco_last_oldlogprob_skipped_missing_hidden = 0

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
                # Active freeze stops hidden-state feature collection (the
                # drafter no longer trains); accept-length and PPO old_log_prob
                # stay ungated. Shadow never sets _speco_drafter_frozen.
                drafter_frozen=getattr(self, "_speco_drafter_frozen", False),
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

    def _speco_should_train_drafter_this_step(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return speco_step_matches_interval(
            self.global_steps, training_cfg.get("training_interval_steps", 1)
        )

    def _speco_has_collected_drafter_samples_this_step(self) -> bool:
        return int(getattr(self, "_speco_last_collected_samples", 0) or 0) > 0

    # ------------------------------------------------------------------ #
    # Per-request accept-length records (feed the freeze policy evidence)
    # ------------------------------------------------------------------ #
    def _speco_batch_size_from_request_stats(self, batch: Any) -> int:
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return 0
        for key in (
            _SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY,
            _SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY,
            _SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY,
            _SPECO_VLLM_REQUEST_ID_KEY,
        ):
            values = non_tensor_batch.get(key)
            if values is None or isinstance(values, (str, bytes)):
                continue
            try:
                return len(values)
            except TypeError:
                tolist = getattr(values, "tolist", None)
                if callable(tolist):
                    try:
                        return len(tolist())
                    except TypeError:
                        continue
        return 0

    def _speco_freeze_request_accept_lengths(
        self, batch: Any, batch_size: int
    ) -> list[float | None]:
        """Per-request mean accept length from the rollout non_tensor_batch.

        Agent-loop batches wrap each trajectory's extra fields in object-array
        cells, and the vLLM server emits per-completion values as lists (n>=1),
        so cells arrive shaped like ``[[2.0], [1.5], ...]``. Unwrap singleton
        (or arbitrarily nested) lists before float coercion.
        """
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return [None for _ in range(batch_size)]

        explicit = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY),
            batch_size,
        )
        result = [_speco_request_cell_float(value) for value in explicit]
        if any(value is not None for value in result):
            return result

        rounds = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY), batch_size
        )
        accepted = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY), batch_size
        )
        for index, (round_value, accepted_value) in enumerate(
            zip(rounds, accepted, strict=False)
        ):
            verify_rounds = _speco_request_cell_float(round_value)
            accepted_tokens = _speco_request_cell_float(accepted_value)
            if verify_rounds is None or verify_rounds <= 0 or accepted_tokens is None:
                continue
            result[index] = 1.0 + accepted_tokens / verify_rounds
        return result

    def _speco_populate_request_accept_len_records(self, batch: Any) -> None:
        """Store this step's per-request accept lengths for the freeze gate.

        Mirrors the source repo's record pipeline but without the hard-sample
        collection dependency: builds ``_speco_last_request_accept_len_records``
        directly from the rollout batch so the freeze policy's evidence feed
        (``_speco_observe_rollout_evidence``) has per-request data each step.
        """
        batch_size = self._speco_batch_size_from_request_stats(batch)
        if batch_size <= 0:
            self._speco_last_request_accept_len_records = []
            return
        accept_lens = self._speco_freeze_request_accept_lengths(batch, batch_size)
        records = []
        for batch_idx, mean_accept_len in enumerate(accept_lens):
            if mean_accept_len is None:
                continue
            records.append(
                {
                    "batch_idx": batch_idx,
                    "mean_accept_len": round(float(mean_accept_len), 6),
                }
            )
        self._speco_last_request_accept_len_records = records

    # ------------------------------------------------------------------ #
    # Marginal-utility freeze policy (method=marginal_utility_v1)
    # ------------------------------------------------------------------ #
    def _speco_init_freeze_policy(self):
        """Build the version-clock freeze policy from config, or ``None``.

        The policy object is mode-agnostic: shadow vs active is enforced here
        in the trainer (shadow never sets ``_speco_drafter_frozen``).
        """
        training_cfg = self._speco_drafter_training_config()
        cfg = training_cfg.get("drafter_convergence_freeze", {}) or {}
        if not cfg.get("enabled", False):
            return None
        economics = cfg.get("economics", {}) or {}
        # shortcut: reject async ROI gating until completion waits are attributed correctly.
        if training_cfg.get("publish_async", False) and economics.get("enabled", False):
            raise ValueError(
                "publish_async=true with drafter_convergence_freeze.economics.enabled=true "
                "is not supported while async publish completion-cost accounting is pending; "
                "set publish_async=false or economics.enabled=false"
            )
        method = str(cfg.get("method", "marginal_utility_v1")).strip().lower()
        if method != "marginal_utility_v1":
            raise ValueError(
                f"drafter_convergence_freeze.method={method!r} is not supported: "
                "the legacy single-metric tracker was removed "
                "(use method=marginal_utility_v1)"
            )
        from omegaconf import OmegaConf

        from verl_speco.trainer.drafter_freeze_policy import DrafterFreezePolicy

        raw = OmegaConf.to_container(cfg, resolve=True)
        policy = DrafterFreezePolicy(config=raw)
        self._speco_freeze_policy_active = (
            str(raw.get("mode", "shadow")).strip().lower() == "active"
        )
        self._speco_freeze_load_state(policy)
        return policy

    def _speco_freeze_branch_config(self) -> dict[str, Any]:
        cfg = (
            self._speco_drafter_training_config().get("drafter_convergence_freeze", {})
            or {}
        )
        branch = cfg.get("branch_checkpoint", {}) or {}
        # self.config holds raw OmegaConf nodes: DictConfig is NOT a dict
        # subclass (it extends MutableMapping), so normalize to a plain dict
        # before the isinstance gate below. Fail closed: if normalization
        # (including interpolation resolution) fails, let it raise at startup
        # rather than silently returning {} and disabling the whole feature.
        from omegaconf import OmegaConf

        if OmegaConf.is_config(branch):
            branch = OmegaConf.to_container(branch, resolve=True)
        if not isinstance(branch, dict):
            raise TypeError(
                "drafter_convergence_freeze.branch_checkpoint must resolve to a "
                f"mapping, got {type(branch).__name__}"
            )
        return branch

    def _speco_validate_freeze_branch_config(self) -> None:
        """Fail fast on unsupported branch_checkpoint combinations.

        The first version supports exactly one mode: a single automatic fork
        checkpoint on the first active ``-> FROZEN`` transition, failing
        closed. See freeze_transition_branch_checkpoint_design.md sec. 5.1.
        """
        branch = self._speco_freeze_branch_config()
        if not bool(branch.get("enabled", False)):
            return
        cfg = (
            self._speco_drafter_training_config().get("drafter_convergence_freeze", {})
            or {}
        )
        problems: list[str] = []
        if not cfg.get("enabled", False):
            problems.append("drafter_convergence_freeze.enabled must be true")
        if (
            str(cfg.get("method", "marginal_utility_v1")).strip().lower()
            != "marginal_utility_v1"
        ):
            problems.append(
                "branch_checkpoint requires method=marginal_utility_v1 "
                "(a state-transition FreezeDecision)"
            )
        if str(cfg.get("mode", "shadow")).strip().lower() != "active":
            problems.append("branch_checkpoint requires mode=active")
        if bool(branch.get("once", True)) is not True:
            problems.append("only branch_checkpoint.once=true is supported")
        if bool(branch.get("fail_on_error", True)) is not True:
            problems.append("only branch_checkpoint.fail_on_error=true is supported")
        if problems:
            raise ValueError(
                "invalid freeze branch_checkpoint config: " + "; ".join(problems)
            )

    def _speco_default_local_dir(self) -> str | None:
        """Resolve the training output root.

        Base verl ``RayPPOTrainer`` does NOT set ``self.default_local_dir``;
        it reads ``self.config.trainer.default_local_dir`` at save points.
        Support both (the attr takes precedence when present).
        """
        local_dir = getattr(self, "default_local_dir", None)
        if local_dir:
            return str(local_dir)
        local_dir = _get_nested(self.config, ("trainer", "default_local_dir"), None)
        return str(local_dir) if local_dir else None

    def _speco_freeze_state_resume_path(self) -> tuple[str | None, str | None]:
        """Sidecar matching the checkpoint verl actually restores.

        Returns ``(sidecar_path, resume_folder)``; both are ``None`` when no
        policy state must be loaded.

        * ``disable`` (new training): never load. A stale sidecar left in the
          output directory from an earlier run must not be inherited, or a
          fresh run could start FROZEN and skip drafter training.
        * ``auto``: resolve the same checkpoint the base trainer restores --
          the tracker-selected latest ``global_step_S`` under
          ``default_local_dir`` -- and read the sidecar committed inside it.
          No resolvable checkpoint means the base trainer starts from scratch,
          so the policy must start fresh too; a stale sidecar is NOT a valid
          state source.
        * ``resume_path``: only the sidecar carried inside the explicitly
          chosen ``resume_from_path`` folder. A sidecar in the output
          directory never overrides the user's choice.
        """
        trainer_cfg = _get_nested(self.config, ("trainer",), None)
        resume_mode = str(
            _get_nested(trainer_cfg, ("resume_mode",), "disable") or "disable"
        )
        if resume_mode == "disable":
            return None, None
        if resume_mode == "resume_path":
            folder = _get_nested(trainer_cfg, ("resume_from_path",), None)
            resume_folder = str(folder) if folder else None
        elif resume_mode == "auto":
            root = self._speco_default_local_dir()
            if not root:
                return None, None
            from verl.utils.checkpoint.checkpoint_manager import (
                find_latest_ckpt_path,
            )

            resume_folder = find_latest_ckpt_path(os.path.abspath(os.fspath(root)))
            if not resume_folder:
                return None, None
        else:
            return None, None
        if not resume_folder:
            return None, None
        return (
            os.path.join(str(resume_folder), self._speco_freeze_state_sidecar),
            str(resume_folder),
        )

    def _speco_freeze_load_state(self, policy) -> None:
        path, resume_folder = self._speco_freeze_state_resume_path()
        if not path or resume_folder is None:
            return
        from verl_speco.trainer.checkpoint import freeze_branch_manifest_path

        has_fork_manifest = os.path.exists(freeze_branch_manifest_path(resume_folder))
        if not os.path.exists(path):
            if has_fork_manifest:
                # A committed fork point is only complete with its policy
                # state: fail closed rather than silently train the branch.
                raise RuntimeError(
                    f"committed freeze branch checkpoint {resume_folder} is "
                    f"missing its policy state sidecar ({path}); the fork "
                    "point is incomplete, refusing to resume"
                )
            # Checkpoints written before per-checkpoint freeze state (or by
            # runs with the feature disabled) carry no sidecar: say so
            # explicitly instead of silently rebinding stale state.
            logger.warning(
                "[drafter freeze] no policy state sidecar in resumed "
                "checkpoint %s; freeze policy starts fresh (compatibility "
                "downgrade, not a full state restore)",
                resume_folder,
            )
            return
        with open(path, "r", encoding="utf-8") as handle:
            policy.load_state_dict(json.load(handle))
        print(
            f"[drafter freeze] restored policy state from {path} "
            f"(version={policy.drafter_version}, state={policy.state.value})",
            flush=True,
        )
        if has_fork_manifest:
            # Fork-specific validation ONLY when a fork manifest exists; a
            # plain checkpoint never triggers fork semantics.
            self._speco_freeze_assert_matches_fork_manifest(policy, resume_folder)

    def _speco_freeze_assert_matches_fork_manifest(
        self, policy, resume_folder: str
    ) -> None:
        """Fail fast when a forked branch did not restore the frozen state.

        After resuming from a committed freeze branch checkpoint, the policy
        MUST still be FROZEN at the manifest's drafter_version. A mismatch
        means a fingerprinted freeze parameter drifted (e.g. an omitted
        override silently reset the policy to CALIBRATING); continuing would
        silently train the "frozen" branch, so abort startup.
        """
        from verl_speco.trainer.checkpoint import read_freeze_branch_manifest

        try:
            manifest = read_freeze_branch_manifest(resume_folder)
        except Exception as exc:  # noqa: BLE001 - no committed fork point
            raise RuntimeError(
                f"resume folder {resume_folder} is not a committed freeze "
                f"branch checkpoint: {exc}"
            ) from exc
        expected_version = int(manifest["drafter_version"])
        problems: list[str] = []
        if policy.state.value != "FROZEN":
            problems.append(f"state={policy.state.value!r} (expected FROZEN)")
        if int(policy.drafter_version) != expected_version:
            problems.append(
                f"drafter_version={int(policy.drafter_version)} "
                f"(expected {expected_version})"
            )
        if problems:
            raise RuntimeError(
                "freeze branch resume failed to restore the frozen policy "
                f"state from {resume_folder}: " + "; ".join(problems)
            )

    def _speco_freeze_save_state(self) -> None:
        """Write policy state before the base save commits the latest tracker.

        An interrupted model save may leave an uncommitted folder; auto-resume
        selects only the checkpoint named by the tracker.
        """
        policy = getattr(self, "_speco_freeze_policy", None)
        local_dir = self._speco_default_local_dir()
        if policy is None or not local_dir:
            return
        folder = os.path.join(
            local_dir,
            f"global_step_{int(getattr(self, 'global_steps', 0) or 0)}",
        )
        os.makedirs(folder, exist_ok=True)
        from verl_speco.trainer.checkpoint import (
            atomic_write_json,
            freeze_policy_sidecar_path,
        )

        atomic_write_json(freeze_policy_sidecar_path(folder), policy.state_dict())

    def _speco_freeze_emit_transition(self, decision) -> None:
        if decision is None or not decision.transitioned:
            return
        payload = {
            "event": "drafter_freeze_transition",
            "step": int(getattr(self, "global_steps", 0) or 0),
            "state": decision.state.value,
            "reason": decision.reason,
            "drafter_version": int(decision.drafter_version),
            "update_opportunity_id": int(decision.update_opportunity_id),
            "mode": "active" if self._speco_freeze_policy_active else "shadow",
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _speco_record_freeze_branch_request(self, decision) -> None:
        """Record (intent only, no I/O) the first active transition into FROZEN.

        Consumed later by :meth:`_speco_maybe_save_freeze_branch_checkpoint`
        at the post-actor-update safe point. Must never do I/O itself: it runs
        inside the generate hook (design sec. 4.2).
        """
        if self._speco_freeze_branch_saved:
            return
        if self._speco_freeze_branch_save_pending is not None:
            return
        branch_cfg = self._speco_freeze_branch_config()
        if not bool(branch_cfg.get("enabled", False)):
            return
        if not self._speco_freeze_policy_active or decision is None:
            return

        from verl_speco.trainer.drafter_freeze_policy import FreezeState

        if not decision.transitioned or decision.state != FreezeState.FROZEN:
            return

        request = {
            "trigger_step": int(getattr(self, "global_steps", 0) or 0),
            "reason": str(decision.reason),
            "drafter_version": int(decision.drafter_version),
            "update_opportunity_id": int(decision.update_opportunity_id),
        }
        self._speco_freeze_branch_save_pending = request
        print(
            json.dumps(
                {
                    "event": "freeze_branch_checkpoint_requested",
                    "step": request["trigger_step"],
                    "reason": request["reason"],
                    "drafter_version": request["drafter_version"],
                    "update_opportunity_id": request["update_opportunity_id"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _speco_maybe_save_freeze_branch_checkpoint(self) -> None:
        """Consume a pending fork request using the full checkpoint chain.

        Called from the patched ``_update_actor`` after the actor update (and
        after the frozen step skipped its drafter train/publish), at the same
        ordering point the base loop uses for periodic saves
        (post-update_actor, pre-``checkpoint_manager.update_weights``,
        pre-next-rollout). Never call this from the generate or tracking-log
        hooks. Fails closed: on any error the failure event is emitted and the
        exception propagates; pending is retained and success is not marked.
        """
        request = getattr(self, "_speco_freeze_branch_save_pending", None)
        if request is None or getattr(self, "_speco_freeze_branch_saved", False):
            return
        branch_cfg = self._speco_freeze_branch_config()
        if not bool(branch_cfg.get("enabled", False)):
            # Feature was disabled after the request was recorded: drop intent.
            self._speco_freeze_branch_save_pending = None
            return

        from verl_speco.trainer.checkpoint import (
            FREEZE_BRANCH_FORMAT_VERSION,
            atomic_write_json,
            freeze_policy_sidecar_path,
            validate_freeze_branch_checkpoint,
            write_freeze_branch_manifest,
        )

        step = int(getattr(self, "global_steps", 0) or 0)
        local_dir = self._speco_default_local_dir()
        target_folder = (
            os.path.join(local_dir, f"global_step_{step}") if local_dir else None
        )
        try:
            if step != int(request["trigger_step"]):
                raise RuntimeError(
                    "freeze branch checkpoint requested at step "
                    f"{request['trigger_step']} but consumed at step {step}"
                )
            if not local_dir or target_folder is None:
                raise RuntimeError(
                    "freeze branch checkpoint requires trainer.default_local_dir"
                )
            # Full chain: wait async publish, save drafter draft_step_S, then
            # actor/critic/dataloader; on success the policy sidecar is
            # committed inside global_step_S as well.
            self._save_checkpoint()

            # Persist the freeze policy state INSIDE the fork folder so the
            # branch checkpoint is self-contained; written before the
            # manifest, which is the last (atomic) commit. This is redundant
            # with the sidecar committed by _save_checkpoint by design: the
            # fork contract requires the sidecar unconditionally (also when
            # the save was deduplicated), and validation below enforces it.
            policy = getattr(self, "_speco_freeze_policy", None)
            if policy is not None:
                atomic_write_json(
                    freeze_policy_sidecar_path(target_folder),
                    policy.state_dict(),
                )

            manifest = {
                "format_version": FREEZE_BRANCH_FORMAT_VERSION,
                "complete": True,
                "trigger_step": step,
                "checkpoint_step": step,
                "drafter_version": int(request["drafter_version"]),
                "update_opportunity_id": int(request["update_opportunity_id"]),
                "freeze_reason": request["reason"],
                "freeze_mode_at_save": (
                    "active" if self._speco_freeze_policy_active else "shadow"
                ),
                "compare_from_step": step + 1,
            }
            write_freeze_branch_manifest(target_folder, manifest)
            drafter_root = self._speco_ensure_drafter_checkpoint_path()
            validate_freeze_branch_checkpoint(
                target_folder,
                drafter_root=drafter_root,
                step=step,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "freeze_branch_checkpoint_failed",
                        "step": step,
                        "path": target_folder,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if bool(branch_cfg.get("fail_on_error", True)):
                raise
            return

        self._speco_freeze_branch_saved = True
        self._speco_freeze_branch_save_pending = None
        print(
            json.dumps(
                {
                    "event": "freeze_branch_checkpoint_saved",
                    "step": step,
                    "path": target_folder,
                    "drafter_version": int(request["drafter_version"]),
                    "freeze_reason": request["reason"],
                    "compare_from_step": step + 1,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    @staticmethod
    def _speco_rollout_response_tokens(output: Any) -> int | None:
        """Read the actual response-token count before Tracking.log exists."""
        batch = getattr(output, "batch", None)
        if batch is None:
            return None
        response_mask = batch.get("response_mask", None)
        if torch.is_tensor(response_mask):
            return int(response_mask.detach().sum().item())
        responses = batch.get("responses", None)
        attention_mask = batch.get("attention_mask", None)
        prompts = batch.get("prompts", None)
        if not (
            torch.is_tensor(responses)
            and torch.is_tensor(attention_mask)
            and torch.is_tensor(prompts)
        ):
            return None
        start = int(prompts.shape[-1])
        width = int(responses.shape[-1])
        return int(attention_mask[..., start : start + width].detach().sum().item())

    def _speco_observe_rollout_evidence(
        self,
        output: Any,
        serving_version: int,
        *,
        generation_seconds: float | None = None,
    ):
        """Feed one rollout step's per-request evidence to the freeze policy.

        Runs inside the generate hook (post-generation, pre update_actor), so
        evidence precedes the inline publish event; under deferred publish the
        previous step's publish has already been observed by the time the
        tagged ``serving_version`` arrives (see policy event-order handling).
        """
        policy = getattr(self, "_speco_freeze_policy", None)
        if policy is None:
            return None
        records = getattr(self, "_speco_last_request_accept_len_records", None) or []
        accept_lens = [
            float(record["mean_accept_len"])
            for record in records
            if isinstance(record, dict) and record.get("mean_accept_len") is not None
        ]

        meta_info = getattr(output, "meta_info", None)
        output_metrics = (
            meta_info.get("metrics", {}) if isinstance(meta_info, dict) else {}
        )
        response_length_mean = _speco_optional_float(
            output_metrics.get("response_length/mean")
        )
        gen_time = _speco_optional_float(generation_seconds)
        if gen_time is None:
            gen_time = _speco_optional_float(output_metrics.get("timing_s/gen"))
        response_tokens = self._speco_rollout_response_tokens(output)
        if response_tokens is None:
            response_tokens = (
                int(len(accept_lens) * response_length_mean)
                if response_length_mean is not None
                else 0
            )

        low_fraction = None
        if accept_lens:
            from verl_speco.trainer.accept_len_convergence import bimodal_metrics

            low_fraction = float(bimodal_metrics(accept_lens)["low_fraction"])

        from verl_speco.trainer.drafter_freeze_policy import FreezeEvidence

        evidence = FreezeEvidence(
            global_step=int(getattr(self, "global_steps", 0) or 0),
            drafter_version=int(serving_version),
            request_accept_lens=accept_lens,
            response_tokens=response_tokens,
            generation_seconds=float(gen_time or 0.0),
            opportunity_this_step=bool(self._speco_should_train_drafter_this_step()),
            low_fraction=low_fraction,
        )
        decision = policy.observe(evidence)
        self._speco_last_freeze_decision = decision
        if self._speco_freeze_policy_active:
            # Same gate the legacy tracker drives; read by the train attempt
            # gate and by update_actor's frozen-plan replacement this step.
            self._speco_drafter_frozen = not bool(decision.should_train)
        self._speco_freeze_emit_transition(decision)
        # Intent only (no I/O): consumed at the post-actor-update safe point.
        self._speco_record_freeze_branch_request(decision)
        return decision

    @staticmethod
    def _speco_freeze_publish_cost_seconds(
        metrics: dict[str, Any],
    ) -> float | None:
        cost = 0.0
        for key in (
            "timing_s/drafter_publish_wait_pending",
            "timing_s/drafter_publish_fetch_snapshot",
            "timing_s/drafter_publish_update_weights",
        ):
            value = _speco_optional_float(metrics.get(key))
            if value is not None:
                cost += value
        return cost if cost > 0.0 else None

    def _speco_freeze_observe_publish(
        self,
        publish_metrics: dict[str, Any],
        *,
        update_cost_seconds: float | None = None,
        probe=None,
    ) -> None:
        """Record a successful train+publish as one real version-clock update.

        Failed attempts (``drafter/published`` != 1) never reach the policy, so
        a failed train/publish can never be counted as a zero-gain update.
        ``probe`` is the fixed paired :class:`ProbeComparison` around this exact
        publish (RFC sec. 6); absent on probe-skipped/failed publishes, where
        the economics gate fails open.
        """
        policy = getattr(self, "_speco_freeze_policy", None)
        if policy is None or not isinstance(publish_metrics, dict):
            return
        if int(publish_metrics.get("drafter/published", 0) or 0) != 1:
            return
        cost = update_cost_seconds
        if cost is None:
            cost = self._speco_freeze_publish_cost_seconds(publish_metrics)

        from verl_speco.trainer.drafter_freeze_policy import FreezeEvidence

        evidence = FreezeEvidence(
            global_step=int(getattr(self, "global_steps", 0) or 0),
            drafter_version=int(policy.drafter_version) + 1,
            publish_completed=True,
            update_succeeded=True,
            publish_succeeded=True,
            update_cost_seconds=cost,
            probe=probe,
        )
        decision = policy.observe(evidence)
        self._speco_last_freeze_decision = decision

    # ------------------------------------------------------------------ #
    # Fixed paired probe execution (RFC sec. 6)
    # ------------------------------------------------------------------ #
    def _speco_freeze_probe_config(self) -> dict[str, Any]:
        cfg = (
            self._speco_drafter_training_config().get("drafter_convergence_freeze", {})
            or {}
        )
        probe = cfg.get("probe", {}) or {}
        # Raw Hydra configs are DictConfig nodes (NOT dict subclasses); without
        # normalization the isinstance gate below silently returns {} and the
        # executor falls back to default prompts/token counts while the policy
        # still schedules probes. Normalize like _speco_freeze_branch_config.
        from omegaconf import OmegaConf

        if OmegaConf.is_config(probe):
            probe = OmegaConf.to_container(probe, resolve=True)
        if not isinstance(probe, dict):
            raise TypeError(
                "drafter_convergence_freeze.probe must resolve to a mapping, "
                f"got {type(probe).__name__}"
            )
        return probe

    def _speco_freeze_probe_pool_data(self) -> list[dict[str, Any]]:
        """Fixed ordered prompt pool as collated chunks (built once).

        Same prompts in the same order for every probe/arm: pairing is by row
        position. Defaults to the first ``probe.prompt_count`` TRAINING rows
        (selection is forced sequential even when the training data config
        shuffles). Rows are collated in ``probe.batch_size`` chunks and each
        arm generates one chunk at a time. Set ``probe.prompts_file`` to a
        parquet to use a dedicated probe pool.
        """
        cached = getattr(self, "_speco_freeze_probe_pool_cache", None)
        if cached is not None:
            return cached
        import copy

        import numpy as np
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.trainer.ppo.utils import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        probe_cfg = self._speco_freeze_probe_config()
        prompt_count = int(probe_cfg.get("prompt_count", 256))
        batch_size = int(probe_cfg.get("batch_size", 64))
        prompts_file = probe_cfg.get("prompts_file", None)
        files = prompts_file if prompts_file else self.config.data.train_files
        # Never let the training config's shuffle turn max_samples into a
        # random subset: the probe pool must be identical across runs/arms.
        data_cfg = copy.deepcopy(self.config.data)
        data_cfg.shuffle = False
        dataset = create_rl_dataset(
            files,
            data_cfg,
            self.tokenizer,
            self.processor,
            max_samples=prompt_count,
        )
        loader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=default_collate_fn,
        )
        chunks: list[dict[str, Any]] = []
        remaining = prompt_count
        row_offset = 0
        for batch in loader:
            if remaining <= 0:
                break
            chunk = dict(batch)
            first_value = next(iter(chunk.values()))
            n_rows = min(len(first_value), remaining)
            if n_rows != len(first_value):
                chunk = {key: value[:n_rows] for key, value in chunk.items()}
            # Stable uids double as deterministic pair ids across arms/runs.
            chunk["uid"] = np.array(
                [
                    f"speco-freeze-probe-{i:04d}"
                    for i in range(row_offset, row_offset + n_rows)
                ],
                dtype=object,
            )
            chunks.append(chunk)
            row_offset += n_rows
            remaining -= n_rows
        if not chunks:
            raise RuntimeError("freeze probe pool is empty")
        print(
            f"[drafter freeze] fixed probe pool ready: {row_offset} prompts "
            f"in {len(chunks)} chunk(s) of <= {batch_size} "
            f"from {prompts_file or 'train_files'}",
            flush=True,
        )
        self._speco_freeze_probe_pool_cache = chunks
        return chunks

    def _speco_build_freeze_probe_gen_batch(self, data: dict[str, Any]):
        from verl.protocol import pad_dataproto_to_divisor

        batch = DataProto.from_single_dict(data)
        gen_batch = self._get_gen_batch(batch)
        probe_cfg = self._speco_freeze_probe_config()
        max_new_tokens = int(probe_cfg.get("max_new_tokens", 2048))
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
            "global_steps": int(getattr(self, "global_steps", 0) or 0),
            _SPECO_FREEZE_PROBE_META_KEY: True,
            _SPECO_FREEZE_PROBE_MAX_TOKENS_META_KEY: max_new_tokens,
        }
        size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
        return pad_dataproto_to_divisor(gen_batch, size_divisor)

    def _speco_freeze_probe_rows(self, output: Any) -> list[dict[str, Any]]:
        """Per-request accept length / generated tokens / wall seconds."""
        non_tensor_batch = getattr(output, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            raise RuntimeError("freeze probe output has no non_tensor_batch")
        batch_size = self._speco_batch_size_from_request_stats(output)
        if batch_size <= 0:
            raise RuntimeError("freeze probe output carries no per-request stats")

        accept_lens = self._speco_freeze_request_accept_lengths(output, batch_size)
        rounds = [
            _speco_request_cell_float(value)
            for value in _speco_sequence_values(
                non_tensor_batch.get(_SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY),
                batch_size,
            )
        ]
        accepted = [
            _speco_request_cell_float(value)
            for value in _speco_sequence_values(
                non_tensor_batch.get(_SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY),
                batch_size,
            )
        ]
        elapsed = [
            _speco_request_cell_float(value)
            for value in _speco_sequence_values(
                non_tensor_batch.get(_SPECO_VLLM_REQUEST_ELAPSED_SEC_KEY),
                batch_size,
            )
        ]
        uids = [
            str(value) if value is not None else f"speco-freeze-probe-{i:04d}"
            for i, value in enumerate(
                _speco_sequence_values(non_tensor_batch.get("uid"), batch_size)
            )
        ]
        rows: list[dict[str, Any]] = []
        for i in range(batch_size):
            verify_rounds = rounds[i]
            accepted_tokens = accepted[i]
            tokens = (
                verify_rounds + accepted_tokens
                if verify_rounds is not None
                and verify_rounds > 0
                and accepted_tokens is not None
                else None
            )
            seconds = elapsed[i]
            rows.append(
                {
                    "uid": uids[i],
                    "accept": accept_lens[i],
                    "tokens": tokens,
                    "seconds": seconds if seconds is not None and seconds > 0 else None,
                }
            )
        return rows

    def _speco_run_freeze_probe_arm(self) -> tuple[list[dict[str, Any]], float]:
        from verl.protocol import unpad_dataproto

        rows: list[dict[str, Any]] = []
        wall_seconds = 0.0
        for chunk in self._speco_freeze_probe_pool_data():
            gen_batch_padded, pad_size = self._speco_build_freeze_probe_gen_batch(chunk)
            started = time.perf_counter()
            output_padded = self._speco_rollout_generation_target().generate_sequences(
                gen_batch_padded
            )
            wall_seconds += time.perf_counter() - started
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            rows.extend(self._speco_freeze_probe_rows(output))
        # Re-stamp global pair ids: pairing downstream is by row position, so
        # labels must be unique even if an engine path dropped per-request uids.
        for i, row in enumerate(rows):
            row["uid"] = f"speco-freeze-probe-{i:04d}"
        return rows, wall_seconds

    def _speco_build_freeze_probe_comparison(
        self,
        before_arm: tuple[list[dict[str, Any]], float],
        after_arm: tuple[list[dict[str, Any]], float],
        *,
        version_before: int,
    ):
        from verl_speco.trainer.drafter_freeze_policy import ProbeComparison

        before_rows, before_wall = before_arm
        after_rows, after_wall = after_arm
        n_pairs = min(len(before_rows), len(after_rows))
        if n_pairs <= 0:
            raise RuntimeError("freeze probe arms produced no aligned rows")
        if len(before_rows) != len(after_rows):
            logger.warning(
                "[drafter freeze] probe arm size mismatch: before=%d after=%d; "
                "using first %d rows",
                len(before_rows),
                len(after_rows),
                n_pairs,
            )
        ids: list[str] = []
        accept_before: list[float | None] = []
        accept_after: list[float | None] = []
        tokens_before, seconds_before = [], []
        tokens_after, seconds_after = [], []
        for i in range(n_pairs):
            row_b, row_a = before_rows[i], after_rows[i]
            ids.append(row_b["uid"])
            accept_before.append(row_b["accept"])
            accept_after.append(row_a["accept"])
            tokens_before.append(row_b["tokens"])
            seconds_before.append(row_b["seconds"])
            tokens_after.append(row_a["tokens"])
            seconds_after.append(row_a["seconds"])
        return ProbeComparison(
            request_ids=ids,
            accept_before=accept_before,
            accept_after=accept_after,
            tokens_before=tokens_before,
            seconds_before=seconds_before,
            tokens_after=tokens_after,
            seconds_after=seconds_after,
            version_before=int(version_before),
            version_after=int(version_before) + 1,
            global_step=int(getattr(self, "global_steps", 0) or 0),
            wall_seconds=float(before_wall + after_wall),
        )

    def _speco_freeze_log_probe(self, probe, probe_error: str | None) -> None:
        policy = getattr(self, "_speco_freeze_policy", None)
        if policy is None:
            return
        stats = getattr(policy, "_latest_probe_stats", None)
        payload: dict[str, Any] = {
            "event": "drafter_freeze_probe",
            "step": int(getattr(self, "global_steps", 0) or 0),
            "mode": "active" if self._speco_freeze_policy_active else "shadow",
        }
        if probe_error:
            payload["error"] = probe_error
        if probe is not None:
            payload["wall_seconds"] = round(float(probe.wall_seconds or 0.0), 3)
        if isinstance(stats, dict):
            for key in (
                "version_after",
                "request_count",
                "accept_pairs",
                "timing_pairs",
                "coverage",
                "gain_all",
                "gain_hard",
                "delta_sec_per_token",
                "delta_ucb95",
                "sec_per_token_before",
                "sec_per_token_after",
            ):
                value = stats.get(key)
                if isinstance(value, float):
                    payload[key] = round(value, 6)
                elif value is not None:
                    payload[key] = value
            roi = policy._probe_roi(stats)
            if roi is not None:
                payload["roi_next"] = round(roi["roi"], 6)
                payload["roi_next_ucb95"] = round(roi["roi_ucb95"], 6)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _speco_freeze_metrics(self) -> dict[str, float]:
        """Flat metrics for Tracking.log; shadow decisions are report-only."""
        policy = getattr(self, "_speco_freeze_policy", None)
        decision = getattr(self, "_speco_last_freeze_decision", None)
        if policy is None or decision is None:
            return {}
        metrics = {
            str(key): float(value)
            for key, value in decision.metrics.items()
            if value is not None
        }
        metrics["drafter/freeze_shadow_mode"] = (
            0.0 if self._speco_freeze_policy_active else 1.0
        )
        metrics["drafter/freeze_would_freeze"] = float(decision.would_freeze)
        metrics["drafter/freeze_active_enforced"] = float(
            self._speco_freeze_policy_active
            and getattr(self, "_speco_drafter_frozen", False)
        )
        metrics["drafter/dropped_late_rollouts"] = float(
            getattr(policy, "dropped_late_rollouts", 0)
        )
        return metrics

    def _speco_drafter_schedule_config(self) -> DrafterScheduleConfig:
        return DrafterScheduleConfig.from_mapping(self._speco_drafter_training_config())

    def _speco_should_log_request_accept_len_variance(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        interval_steps = training_cfg.get(
            "request_accept_len_variance_interval_steps", 1
        )
        return step_matches_interval(self.global_steps, interval_steps)

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
                drafter_frozen=bool(getattr(self, "_speco_drafter_frozen", False)),
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

    def _speco_request_accept_lengths(
        self, batch: DataProto, batch_size: int
    ) -> list[float | None]:
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return [None for _ in range(batch_size)]

        explicit = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY),
            batch_size,
        )
        result = [_speco_optional_float(value) for value in explicit]
        if any(value is not None for value in result):
            return result

        rounds = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY), batch_size
        )
        accepted = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY), batch_size
        )
        for index, (round_value, accepted_value) in enumerate(
            zip(rounds, accepted, strict=False)
        ):
            verify_rounds = _speco_optional_float(round_value)
            accepted_tokens = _speco_optional_float(accepted_value)
            if verify_rounds is None or verify_rounds <= 0 or accepted_tokens is None:
                continue
            result[index] = 1.0 + accepted_tokens / verify_rounds
        return result

    def _speco_request_ids(
        self,
        batch: DataProto,
        batch_size: int,
        *,
        require: bool = False,
        required_mask: list[bool] | None = None,
    ) -> list[str]:
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            if require:
                raise RuntimeError(
                    "SPECO request acceptance stats require a non-tensor request id field"
                )
            return [str(index) for index in range(batch_size)]
        if require and _SPECO_VLLM_REQUEST_ID_KEY not in non_tensor_batch:
            raise RuntimeError(
                f"SPECO request acceptance stats missing {_SPECO_VLLM_REQUEST_ID_KEY}"
            )
        request_ids = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_ID_KEY), batch_size
        )
        if require and any(
            value is None and (required_mask is None or required_mask[index])
            for index, value in enumerate(request_ids)
        ):
            raise RuntimeError(
                "SPECO request acceptance stats contain a missing request id"
            )
        present_ids = [str(value) for value in request_ids if value is not None]
        if require and len(set(present_ids)) != len(present_ids):
            raise RuntimeError(
                "SPECO request acceptance stats contain duplicate request ids"
            )
        return [
            str(value) if value is not None else str(index)
            for index, value in enumerate(request_ids)
        ]

    def _speco_request_completion_indices(
        self, batch: DataProto, batch_size: int
    ) -> list[int | None]:
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return [None for _ in range(batch_size)]
        values = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_COMPLETION_INDEX_KEY),
            batch_size,
        )
        indices = []
        for value in values:
            parsed = _speco_optional_float(value)
            indices.append(int(parsed) if parsed is not None else None)
        return indices

    def _speco_request_elapsed_secs(
        self, batch: DataProto, batch_size: int
    ) -> list[float | None]:
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            return [None for _ in range(batch_size)]
        values = _speco_sequence_values(
            non_tensor_batch.get(_SPECO_VLLM_REQUEST_ELAPSED_SEC_KEY),
            batch_size,
        )
        return [_speco_optional_float(value) for value in values]

    def _speco_log_request_accept_len_variance_from_batch(
        self, batch: DataProto, *, source: str
    ) -> bool:
        batch_size = self._speco_batch_size_from_request_stats(batch)
        if batch_size <= 0:
            return False

        request_accept_lens = self._speco_request_accept_lengths(batch, batch_size)
        if not any(value is not None for value in request_accept_lens):
            return False
        request_ids = self._speco_request_ids(
            batch,
            batch_size,
            require=True,
            required_mask=[value is not None for value in request_accept_lens],
        )
        request_completion_indices = self._speco_request_completion_indices(
            batch, batch_size
        )
        request_elapsed_secs = self._speco_request_elapsed_secs(batch, batch_size)
        candidates = [
            {
                "batch_idx": batch_idx,
                "request_id": request_ids[batch_idx],
                "mean_accept_len": request_accept_lens[batch_idx],
                "completion_index": request_completion_indices[batch_idx],
                "elapsed_sec": request_elapsed_secs[batch_idx],
            }
            for batch_idx in range(batch_size)
        ]
        marker = torch.zeros(batch_size, dtype=torch.bool)
        records, accept_len_var = self._speco_log_request_accept_lens(
            candidates=candidates,
            is_hard=marker,
            collect_mask=marker,
        )
        payload = _speco_request_accept_len_step_payload(
            step=self.global_steps,
            source=source,
            records=records,
            accept_len_var=accept_len_var,
        )
        if payload:
            self._speco_pending_request_accept_len_payloads.append(payload)
        return bool(records)

    def _speco_log_request_accept_lens(
        self,
        *,
        candidates: list[dict[str, Any]],
        is_hard: torch.Tensor,
        collect_mask: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], float]:
        records = []
        for candidate in candidates:
            mean_accept_len = candidate.get("mean_accept_len")
            if mean_accept_len is None:
                continue
            batch_idx = int(candidate["batch_idx"])
            elapsed_sec = _speco_optional_float(candidate.get("elapsed_sec"))
            records.append(
                {
                    "completion_index": candidate.get("completion_index"),
                    "batch_idx": batch_idx,
                    "request_id": str(candidate.get("request_id")),
                    "mean_accept_len": round(float(mean_accept_len), 6),
                    "elapsed_sec": (
                        round(elapsed_sec, 6) if elapsed_sec is not None else None
                    ),
                    "is_hard": bool(is_hard[batch_idx].item()),
                    "collected": bool(collect_mask[batch_idx].item()),
                }
            )
        records.sort(
            key=lambda item: (
                item["completion_index"] is None,
                item["completion_index"]
                if item["completion_index"] is not None
                else item["batch_idx"],
                item["batch_idx"],
            )
        )
        accept_lens = [
            parsed
            for record in records
            if (parsed := _speco_optional_float(record.get("mean_accept_len")))
            is not None
        ]
        if accept_lens:
            mean_accept_len = sum(accept_lens) / len(accept_lens)
            accept_len_var = sum(
                (value - mean_accept_len) ** 2 for value in accept_lens
            ) / len(accept_lens)
        else:
            accept_len_var = 0.0
        self._speco_last_request_accept_len_records = records
        self._speco_last_request_accept_len_var = accept_len_var
        return records, accept_len_var

    def _speco_flush_request_accept_len_payloads(self) -> None:
        payloads = self._speco_pending_request_accept_len_payloads
        if not payloads:
            return
        self._speco_pending_request_accept_len_payloads = []
        path = os.getenv(
            _SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH_ENV,
            "/tmp/speco_vllm_request_accept_len_hist.jsonl",
        )
        _speco_append_jsonl_batch(path, payloads)
        if _speco_diag_enabled(_SPECO_REQUEST_ACCEPT_LEN_HIST_PRINT_ENV, "0"):
            for payload in payloads:
                print(
                    "[speco request accept len hist] %s"
                    % json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                    flush=True,
                )
        if self._speco_should_log_request_accept_len_variance():
            for payload in payloads:
                print(
                    "[speco request accept len] step=%s requests=%s "
                    "request_accept_len_var=%.3f source=%s"
                    % (
                        payload["step"],
                        payload["count"],
                        payload["summary"]["var"],
                        payload["source"],
                    ),
                    flush=True,
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
        owner_rank = torch.full((batch_size,), -1, dtype=torch.long)

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
        request_accept_lens = self._speco_request_accept_lengths(batch, batch_size)
        require_request_ids = any(value is not None for value in request_accept_lens)
        request_ids = self._speco_request_ids(
            batch,
            batch_size,
            require=require_request_ids,
            required_mask=[value is not None for value in request_accept_lens],
        )
        request_completion_indices = self._speco_request_completion_indices(
            batch, batch_size
        )
        request_elapsed_secs = self._speco_request_elapsed_secs(batch, batch_size)
        all_request_accept_len_candidates = [
            {
                "batch_idx": batch_idx,
                "request_id": request_ids[batch_idx],
                "mean_accept_len": request_accept_lens[batch_idx],
                "completion_index": request_completion_indices[batch_idx],
                "elapsed_sec": request_elapsed_secs[batch_idx],
            }
            for batch_idx in range(batch_size)
        ]

        hard_sample_ratio = float(
            training_cfg.get("dspark_hard_sample_ratio", 0.0) or 0.0
        )
        hard_candidate_ratio = float(
            training_cfg.get("dspark_hard_candidate_ratio", 0.0) or 0.0
        )
        hard_enabled = hard_sample_ratio > 0.0 and hard_candidate_ratio > 0.0
        # The transport now retains every DP shard. The window is a logical
        # collection budget, never a physical prefix of the actor batch.
        usable_window_size = batch_size
        capacity_per_owner = max_per_owner
        if max_tokens_per_owner is not None:
            capacity_per_owner = min(
                capacity_per_owner, max_tokens_per_owner // hidden_rows
            )
        algorithm = (
            str(
                _get_nested(
                    self._speco_drafter_config(), ("speculative_algorithm",), ""
                )
                or ""
            )
            .strip()
            .upper()
        )
        if algorithm == "DSPARK" or hard_enabled:
            collection_limit = int(
                training_cfg.get("dspark_collect_samples_per_step", 80)
            )
            if collection_limit <= 0:
                raise ValueError("dspark_collect_samples_per_step must be positive")
        else:
            collection_limit = capacity_per_owner * owner_count
        for name, ratio in (
            ("candidate", hard_candidate_ratio),
            ("sample", hard_sample_ratio),
        ):
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"Invalid hard {name} ratio: {ratio}")
        shortage_policy = training_cfg.get(
            "dspark_hard_shortfall_policy", "fill_normal"
        )
        if shortage_policy not in ("fill_normal", "strict"):
            raise ValueError(f"Invalid hard shortfall policy: {shortage_policy}")

        candidate_count = 0
        selected_count = 0
        candidates: list[dict[str, Any]] = []
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
            candidates.append(
                {
                    "batch_idx": batch_idx,
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "sample_key": sample_key,
                    "request_id": request_ids[batch_idx],
                    "mean_accept_len": request_accept_lens[batch_idx],
                    "completion_index": request_completion_indices[batch_idx],
                    "elapsed_sec": request_elapsed_secs[batch_idx],
                    "hash": self._speco_hash_fraction(
                        f"{step_key}:{request_ids[batch_idx]}:{batch_idx}:hard"
                    ),
                }
            )

        scored_candidates = [
            candidate
            for candidate in candidates
            if candidate["mean_accept_len"] is not None
        ]

        is_hard = torch.zeros(batch_size, dtype=torch.bool)
        hard_score = torch.zeros(batch_size, dtype=torch.float32)
        mean_accept_len_tensor = torch.full(
            (batch_size,), float("nan"), dtype=torch.float32
        )
        for index, mean_accept_len in enumerate(request_accept_lens):
            if mean_accept_len is not None:
                mean_accept_len_tensor[index] = float(mean_accept_len)
                hard_score[index] = float(-mean_accept_len)

        def owner_has_capacity(owner: int) -> bool:
            if owner_counts[owner] >= max_per_owner:
                return False
            return (
                max_tokens_per_owner is None
                or owner_token_counts[owner] + hidden_rows <= max_tokens_per_owner
            )

        def add_candidate(candidate: dict[str, Any], owner: int) -> bool:
            nonlocal selected_count
            if selected_count >= total_capacity or not owner_has_capacity(owner):
                return False
            batch_idx = int(candidate["batch_idx"])
            max_start_offset = max(int(candidate["response_len"]) - hidden_rows, 0)
            if window_mode == "random":
                random_offset = self._speco_hash_int(
                    f"{candidate['sample_key']}:window", max_start_offset
                )
            else:
                random_offset = 0
            start = max(int(candidate["prompt_len"]) - 1, 0) + random_offset
            positions = torch.arange(start, start + hidden_rows, dtype=torch.long)
            collect_mask[batch_idx] = True
            hidden_positions[batch_idx, :] = positions
            hidden_position_mask[batch_idx, :] = True
            owner_rank[batch_idx] = owner
            owner_counts[owner] += 1
            owner_token_counts[owner] += hidden_rows
            selected_count += 1
            return True

        def add_round_robin(
            items: list[dict[str, Any]], *, start_owner: int = 0
        ) -> int:
            added = 0
            next_owner = start_owner
            for candidate in items:
                if bool(collect_mask[int(candidate["batch_idx"])].item()):
                    continue
                for offset in range(owner_count):
                    owner = (next_owner + offset) % owner_count
                    if add_candidate(candidate, owner):
                        added += 1
                        next_owner = (owner + 1) % owner_count
                        break
            return added

        total_capacity = min(
            collection_limit, len(candidates), capacity_per_owner * owner_count
        )
        hard_added = 0
        target_hard = 0
        labeled_hard_count = 0
        hard_shortfall = 0
        normal_shortfall = 0
        if not hard_enabled:
            normal_added = add_round_robin(candidates)
        else:
            hard_candidates, labeled_hard_count = partition_hard_candidates(
                scored_candidates,
                candidate_ratio=hard_candidate_ratio,
            )
            for candidate in hard_candidates:
                is_hard[candidate["batch_idx"]] = True
            hard_pool = sorted(hard_candidates, key=lambda row: row["hash"])
            normal_pool = sorted(
                (row for row in candidates if not bool(is_hard[row["batch_idx"]])),
                key=lambda row: self._speco_hash_fraction(
                    f"{row['sample_key']}:normal"
                ),
            )
            target_hard = round(total_capacity * hard_sample_ratio)
            actual_hard = min(target_hard, len(hard_pool))
            hard_shortfall = target_hard - actual_hard
            normal_quota = total_capacity - actual_hard
            normal_shortfall = max(normal_quota - len(normal_pool), 0)
            if shortage_policy == "strict" and (hard_shortfall or normal_shortfall):
                raise RuntimeError(
                    f"SPECO unique collection quota unavailable: hard_shortfall={hard_shortfall} "
                    f"normal_shortfall={normal_shortfall}"
                )
            # If normals are also exhausted, retain a smaller UNIQUE set and
            # report the shortfall; do not silently exceed the hard quota.
            first_owner = int(self.global_steps) % owner_count
            hard_added = add_round_robin(
                hard_pool[:actual_hard], start_owner=first_owner
            )
            normal_added = add_round_robin(
                normal_pool[:normal_quota],
                start_owner=(first_owner + hard_added) % owner_count,
            )

        selected_indices = collect_mask.nonzero(as_tuple=False).flatten().tolist()
        hard_counts = [
            int((collect_mask & is_hard & (owner_rank == owner)).sum())
            for owner in range(owner_count)
        ]
        normal_counts = [
            owner_counts[owner] - hard_counts[owner] for owner in range(owner_count)
        ]
        selection_metrics = {
            "eligible_rows": len(candidates),
            "scored_rows": len(scored_candidates),
            "candidate_hard_count": labeled_hard_count,
            "target_hard": target_hard,
            "selected_hard_total": hard_added,
            "selected_hard_unique": hard_added,
            "selected_normal_total": normal_added,
            "hard_shortfall": hard_shortfall,
            "normal_shortfall": normal_shortfall,
            "duplicate_hard_count": 0,
            "selected_total": selected_count,
            "selected_unique": len(selected_indices),
            "collection_shortfall": collection_limit - selected_count,
            "actual_hard_ratio": hard_added / selected_count if selected_count else 0.0,
        }
        for owner in range(owner_count):
            selection_metrics[f"hard_count_owner_{owner}"] = hard_counts[owner]
            selection_metrics[f"normal_count_owner_{owner}"] = normal_counts[owner]
        self._speco_last_hard_selection_metrics = {
            f"drafter/selection/{key}": value
            for key, value in selection_metrics.items()
        }
        selection_hash = hashlib.sha256(
            ",".join(map(str, selected_indices)).encode()
        ).hexdigest()
        print(
            f"[speco hard selection] step={self.global_steps} "
            f"metrics={selection_metrics} selected_source_indices_hash={selection_hash}",
            flush=True,
        )

        self._speco_log_request_accept_lens(
            candidates=all_request_accept_len_candidates,
            is_hard=is_hard,
            collect_mask=collect_mask,
        )

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
            "hard_enabled": hard_enabled,
            "usable_window_size": usable_window_size,
            "selection_metrics": selection_metrics,
            "selected_source_indices_hash": selection_hash,
            "request_ids": request_ids,
            "request_mean_accept_len": mean_accept_len_tensor,
            "request_is_hard": is_hard,
            "request_hard_score": hard_score,
            "request_completion_indices": request_completion_indices,
            "request_elapsed_secs": request_elapsed_secs,
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

    def _speco_materialize_oldlogprob_whole_refs(
        self, output: Any
    ) -> list[torch.Tensor | None]:
        """Materialize row-aligned PP captures once per DP-shard ObjectRef."""
        whole_refs = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY)
        )
        if isinstance(whole_refs, (list, tuple)):
            whole_metas = self._speco_flatten_non_tensor_rows(
                tu.get(output, OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY)
            )
            if not isinstance(whole_metas, (list, tuple)) or len(whole_metas) != len(
                whole_refs
            ):
                raise RuntimeError(
                    "SPECO whole hidden refs and metadata are not row-aligned"
                )
            unique_refs = list(
                dict.fromkeys(ref for ref in whole_refs if ref is not None)
            )
            states_by_ref = dict(zip(unique_refs, ray.get(unique_refs), strict=True))
            hidden_rows: list[torch.Tensor | None] = []
            for ref, meta in zip(whole_refs, whole_metas, strict=True):
                if ref is None:
                    hidden_rows.append(None)
                    continue
                if not isinstance(meta, dict) or "row_index" not in meta:
                    raise RuntimeError(
                        "SPECO whole hidden metadata is missing a row index"
                    )
                row_index = int(meta["row_index"])
                hidden = states_by_ref[ref]
                if (
                    not torch.is_tensor(hidden)
                    or hidden.dim() < 1
                    or not 0 <= row_index < hidden.size(0)
                ):
                    raise RuntimeError("SPECO whole hidden row index is out of bounds")
                hidden_rows.append(hidden[row_index])
        else:
            hidden_rows = self._speco_tensor_rows(ray.get(whole_refs))
        tu.assign_non_tensor_data(output, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY, None)
        tu.assign_non_tensor_data(output, OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY, None)
        return hidden_rows

    def _speco_collect_oldlogprob_features(
        self,
        batch: DataProto,
        collect_plan: dict[str, Any] | None,
        output: Any,
    ) -> int:
        if not collect_plan:
            return 0
        hard_sampling_enabled = bool(collect_plan.get("hard_enabled", False))
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
        hidden_rows = []
        if hidden_states is None and whole_ref is not None:
            hidden_rows = self._speco_materialize_oldlogprob_whole_refs(output)
            del whole_ref
        if (
            hidden_states is None
            and not hidden_rows
            and hidden_refs is None
            and chunk_refs is None
        ):
            raise RuntimeError("SPECO planned collection returned no hidden payload")
        if not hidden_rows:
            hidden_rows = self._speco_tensor_rows(hidden_states)
        if not hidden_rows and not hidden_refs and not chunk_refs:
            raise RuntimeError("SPECO planned collection returned empty hidden payload")
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
        request_mean_accept_len = collect_plan.get("request_mean_accept_len")
        request_is_hard = collect_plan.get("request_is_hard")
        request_hard_score = collect_plan.get("request_hard_score")
        request_completion_indices = collect_plan.get("request_completion_indices")
        request_elapsed_secs = collect_plan.get("request_elapsed_secs")
        request_ids = collect_plan.get("request_ids")
        request_mean_accept_len_tensor = (
            cast(torch.Tensor, request_mean_accept_len)
            if torch.is_tensor(request_mean_accept_len)
            else None
        )
        request_is_hard_tensor = (
            cast(torch.Tensor, request_is_hard)
            if torch.is_tensor(request_is_hard)
            else None
        )
        request_hard_score_tensor = (
            cast(torch.Tensor, request_hard_score)
            if torch.is_tensor(request_hard_score)
            else None
        )
        collected_rows = 0
        payload_bytes = 0
        planned_samples = int(collect_plan["selected_count"])
        skipped_empty_positions = 0
        skipped_missing_hidden = 0
        skipped_empty_hidden = 0
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
                skipped_empty_positions += 1
                continue
            hidden_ref = self._speco_sequence_item(hidden_refs, batch_idx)
            ref_meta = self._speco_sequence_item(hidden_ref_meta, batch_idx)
            ref_chunks = sample_ref_chunks.get(batch_idx)
            if isinstance(ref_meta, dict):
                source_index = ref_meta.get("source_batch_index")
                if source_index is not None and int(source_index) != batch_idx:
                    raise RuntimeError(
                        f"SPECO hidden row mismatch: {source_index} != {batch_idx}"
                    )
                if ref_meta.get("owner") is not None and int(ref_meta["owner"]) != int(
                    owner_rank[batch_idx]
                ):
                    raise RuntimeError("SPECO hidden owner mismatch")
                if (
                    not ref_chunks
                    and hidden_ref is not None
                    and "chunk_start" in ref_meta
                ):
                    ref_chunks = [{"ref": hidden_ref, **ref_meta}]
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
                    skipped_missing_hidden += 1
                    continue
                hidden = hidden[:valid_rows].contiguous()
                if hidden.numel() == 0:
                    skipped_empty_hidden += 1
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
            if hard_sampling_enabled:
                if isinstance(request_ids, (list, tuple)) and batch_idx < len(
                    request_ids
                ):
                    sample[_SPECO_VLLM_REQUEST_ID_KEY] = str(request_ids[batch_idx])
                if request_mean_accept_len_tensor is not None and batch_idx < int(
                    request_mean_accept_len_tensor.numel()
                ):
                    mean_accept_len = float(
                        request_mean_accept_len_tensor[batch_idx].item()
                    )
                    if math.isfinite(mean_accept_len):
                        sample[_SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY] = (
                            mean_accept_len
                        )
                if request_is_hard_tensor is not None and batch_idx < int(
                    request_is_hard_tensor.numel()
                ):
                    sample[_SPECO_VLLM_REQUEST_IS_HARD_KEY] = bool(
                        request_is_hard_tensor[batch_idx].item()
                    )
                if request_hard_score_tensor is not None and batch_idx < int(
                    request_hard_score_tensor.numel()
                ):
                    sample[_SPECO_VLLM_REQUEST_HARD_SCORE_KEY] = float(
                        request_hard_score_tensor[batch_idx].item()
                    )
                if isinstance(
                    request_completion_indices, (list, tuple)
                ) and batch_idx < len(request_completion_indices):
                    completion_index = request_completion_indices[batch_idx]
                    if completion_index is not None:
                        sample[_SPECO_VLLM_REQUEST_COMPLETION_INDEX_KEY] = int(
                            completion_index
                        )
                if isinstance(request_elapsed_secs, (list, tuple)) and batch_idx < len(
                    request_elapsed_secs
                ):
                    elapsed_sec = request_elapsed_secs[batch_idx]
                    if elapsed_sec is not None:
                        sample[_SPECO_VLLM_REQUEST_ELAPSED_SEC_KEY] = float(elapsed_sec)
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
        self._speco_last_oldlogprob_skipped_missing_hidden = skipped_missing_hidden
        missing_hidden_samples = max(planned_samples - collected, 0)
        logger.info(
            "[speco oldlogprob payload] step=%s planned=%s payload=%s "
            "skip_empty_positions=%s skip_missing_hidden=%s skip_empty_hidden=%s",
            self.global_steps,
            planned_samples,
            collected,
            skipped_empty_positions,
            skipped_missing_hidden,
            skipped_empty_hidden,
        )
        if not hasattr(self, "_speco_last_hard_selection_metrics"):
            self._speco_last_hard_selection_metrics = {}
        self._speco_last_hard_selection_metrics.update(
            {
                "drafter/selection/payload_total": collected,
                "drafter/selection/payload_missing": missing_hidden_samples,
            }
        )
        if missing_hidden_samples:
            raise RuntimeError(
                f"SPECO hidden payload incomplete: planned={planned_samples} collected={collected}"
            )
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
            and resolve_drafter_hidden_states_layout(algorithm, training_cfg)
            == "dflash_aux_plus_last"
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
        if getattr(self, "_speco_ci_result", None) is not None:
            self._speco_ci_result.record_workers(outcome.raw_results)
            self._speco_ci_result.record_metrics(outcome.metrics)
        return outcome.trained, dict(outcome.metrics)

    def _speco_activate_drafter_training_model_before_fit(self) -> None:
        if not self.is_drafter_training_enabled(self.config):
            return
        results = self._speco_get_drafter_scheduler().activate_training_workers()
        if getattr(self, "_speco_ci_result", None) is not None:
            self._speco_ci_result.record_workers(results)

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
        metrics = dict(event.metrics or {})
        if getattr(self, "_speco_ci_result", None) is not None:
            self._speco_ci_result.record_metrics(metrics)
        return metrics

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
                self._speco_populate_request_accept_len_records(gen_batch_output)
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
        if (
            getattr(self, "_speco_ci_result", None) is not None
            and isinstance(data, dict)
            and "training/global_step" in data
        ):
            self._speco_ci_result.record_metrics(
                {"training/global_step": data["training/global_step"]}
            )
        data = _speco_move_drafter_timing_next_to_update_actor(data)
        if isinstance(data, dict):
            data.update(self._speco_freeze_metrics())
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
            "train_cost_seconds": None,
        }

        def generate_sequences_with_speco(manager_self, *args, **kwargs):
            self._speco_wait_pending_drafter_publish()
            generation_started = time.perf_counter()
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            generation_elapsed = time.perf_counter() - generation_started
            is_validation_generation = _speco_is_validation_generation(
                args, kwargs, gen_batch_output
            )
            if not is_validation_generation:
                self._speco_store_rollout_metrics(gen_batch_output)
                self._speco_populate_request_accept_len_records(gen_batch_output)
                # Version that actually served THIS generation. Read after the
                # call: an inline publish lands later (in update_actor), while a
                # deferred publish lands inside this call before generation.
                freeze_policy = getattr(self, "_speco_freeze_policy", None)
                serving_version = (
                    int(freeze_policy.drafter_version)
                    if freeze_policy is not None
                    else 0
                )
                self._speco_observe_rollout_evidence(
                    gen_batch_output,
                    serving_version,
                    generation_seconds=generation_elapsed,
                )
                try:
                    self._speco_log_request_accept_len_variance_from_batch(
                        gen_batch_output, source="rollout"
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "[speco request accept len] skipped rollout diagnostics: %s",
                        exc,
                    )
                # Active freeze stops hidden-state collection here and in the
                # scheduler plan; shadow mode keeps collecting. Accept-length
                # evidence above is ungated and keeps feeding drift detection.
                collection_allowed = not getattr(self, "_speco_drafter_frozen", False)
                if collection_allowed:
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
            self._speco_last_hard_selection_metrics = {}
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
            batch_td[OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY] = torch.arange(
                len(batch), dtype=torch.long
            )
            batch_td[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
            batch_td[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan[
                "hidden_positions"
            ]
            batch_td[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan[
                "hidden_position_mask"
            ]
            batch_td[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
            if collect_plan.get("hard_enabled"):
                batch_td[_SPECO_VLLM_REQUEST_IS_HARD_KEY] = collect_plan[
                    "request_is_hard"
                ]
                batch_td[_SPECO_VLLM_REQUEST_HARD_SCORE_KEY] = collect_plan[
                    "request_hard_score"
                ]
                batch_td[_SPECO_VLLM_REQUEST_MEAN_ACCEPT_LEN_KEY] = collect_plan[
                    "request_mean_accept_len"
                ]
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
            metrics.update(getattr(self, "_speco_last_hard_selection_metrics", {}))
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
            if getattr(self, "_speco_drafter_frozen", False):
                training_plan = dataclasses.replace(
                    training_plan, launch=False, reason="drafter_convergence_frozen"
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
            # Successful publish = one real version-clock update (inline path);
            # cost is total drafter train+publish time spent at this opportunity.
            # Fixed paired probes only run around the deferred (update_weights)
            # flush, where a constant-target before-arm is possible; here the
            # gate simply fails open and this is logged once.
            inline_policy = getattr(self, "_speco_freeze_policy", None)
            if (
                inline_policy is not None
                and drafter_trained
                and inline_policy.probe_due(int(inline_policy.drafter_version) + 1)
                and not getattr(self, "_speco_freeze_probe_inline_warned", False)
            ):
                self._speco_freeze_probe_inline_warned = True
                logger.warning(
                    "[drafter freeze] probe is due but the inline publish path "
                    "cannot run a fixed paired probe (no constant-target "
                    "before-arm); economics gate fails open this publish"
                )
            self._speco_freeze_observe_publish(
                metrics,
                update_cost_seconds=_speco_optional_float(
                    metrics.get("timing_s/drafter")
                ),
            )
            if pending_drafter_publish["ready"]:
                pending_drafter_publish["train_cost_seconds"] = _speco_optional_float(
                    metrics.get("timing_s/drafter")
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
            # Event checkpoint at the same safety point the base loop uses for
            # periodic saves (actor update done, drafter frozen/trained for
            # this step, next rollout not started). A coinciding periodic save
            # in the base loop coalesces to a no-op via _save_checkpoint.
            self._speco_maybe_save_freeze_branch_checkpoint()
            self._speco_flush_request_accept_len_payloads()
            return self._speco_update_output_metrics(actor_output, metrics)

        def update_weights_with_speco(manager_self, *args, **kwargs):
            result = original_checkpoint_update_weights(*args, **kwargs)
            if pending_drafter_publish["ready"]:
                # Fixed paired probe (RFC sec. 6): the target checkpoint is
                # constant from here on. Run the before-arm while the engine
                # still serves drafter v, flush the pending publish (v+1), then
                # run the after-arm -- same prompts, greedy, same engine.
                policy = getattr(self, "_speco_freeze_policy", None)
                run_probe = policy is not None and policy.probe_due(
                    int(policy.drafter_version) + 1
                )
                before_arm = None
                probe_error = None
                if run_probe:
                    try:
                        before_arm = self._speco_run_freeze_probe_arm()
                    except Exception:  # noqa: BLE001 - probe never blocks training
                        run_probe = False
                        probe_error = "before_arm_failed"
                        logger.exception(
                            "[drafter freeze] probe before-arm failed; "
                            "publishing without paired probe (fail-open)"
                        )
                publish_metrics = self._speco_publish_drafter_weights(
                    pending_drafter_publish["drafter_trained"],
                    pending_drafter_publish["training_plan"],
                    after_weight_update=True,
                )
                probe = None
                if run_probe and before_arm is not None:
                    try:
                        after_arm = self._speco_run_freeze_probe_arm()
                        version_before = int(policy.drafter_version)
                        probe = self._speco_build_freeze_probe_comparison(
                            before_arm,
                            after_arm,
                            version_before=version_before,
                        )
                    except Exception:  # noqa: BLE001 - probe never blocks training
                        probe_error = "after_arm_failed"
                        logger.exception(
                            "[drafter freeze] probe after-arm failed; "
                            "publishing without paired probe (fail-open)"
                        )
                if probe is not None:
                    publish_metrics["timing_s/drafter_probe"] = round(
                        float(probe.wall_seconds or 0.0), 4
                    )
                train_cost = _speco_optional_float(
                    pending_drafter_publish.get("train_cost_seconds")
                )
                publish_cost = self._speco_freeze_publish_cost_seconds(publish_metrics)
                total_cost = (
                    (train_cost or 0.0) + (publish_cost or 0.0)
                    if train_cost is not None or publish_cost is not None
                    else None
                )
                # Deferred path: the publish completes at the next weight
                # update; carry the previous step's training cost so C_update
                # remains the full train+publish critical path.
                self._speco_freeze_observe_publish(
                    publish_metrics,
                    update_cost_seconds=total_cost,
                    probe=probe,
                )
                if run_probe:
                    self._speco_freeze_log_probe(probe, probe_error)
                self._speco_update_output_metrics(
                    pending_drafter_publish["actor_output"], publish_metrics
                )
                pending_drafter_publish["ready"] = False
                pending_drafter_publish["drafter_trained"] = False
                pending_drafter_publish["actor_output"] = None
                pending_drafter_publish["training_plan"] = None
                pending_drafter_publish["train_cost_seconds"] = None
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
        result_path = os.getenv("SPECO_CI_RESULT_PATH")
        if not result_path:
            return self._speco_fit_impl()
        from verl_speco.integration.ci_result import CIResult

        self._speco_ci_result = CIResult(self.is_drafter_training_enabled(self.config))
        try:
            result = self._speco_fit_impl()
            self._speco_ci_result.data["completed"] = True
            return result
        except BaseException as error:
            self._speco_ci_result.data["errors"].append(
                {"type": type(error).__name__, "message": str(error)}
            )
            raise
        finally:
            self._speco_ci_result.write(result_path)

    def _speco_fit_impl(self):
        try:
            if self.is_drafter_training_enabled(self.config):
                self._speco_activate_drafter_training_model_before_fit()
                self._speco_freeze_policy = self._speco_init_freeze_policy()
                self._speco_validate_freeze_branch_config()
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
        # A freeze-branch event save runs inside the patched _update_actor,
        # immediately before the base loop reaches its own periodic save point.
        # When both land on the same step the full chain must run exactly
        # once; the second call (base loop, or a repeated event request)
        # becomes a no-op.
        if (
            getattr(self, "_speco_last_checkpoint_saved_step", None)
            == self.global_steps
        ):
            logger.info(
                "[speco] checkpoint for global_step=%s already saved this "
                "step; skipping duplicate save",
                self.global_steps,
            )
            return None
        # A checkpoint boundary must not retain an async publish payload or let
        # draft loading overlap actor/drafter serialization. This is redundant
        # with the normal next-generation barrier by design: save/test order is
        # controlled by upstream VERL and can change independently.
        self._speco_wait_pending_drafter_publish()
        self._speco_save_drafter_checkpoint(wait=True)
        # The base save advances the latest-checkpoint tracker before returning.
        # Persist policy first and propagate failures so a committed checkpoint
        # cannot be missing its policy state.
        self._speco_freeze_save_state()
        result = super()._save_checkpoint()
        self._speco_last_checkpoint_saved_step = self.global_steps
        return result

    def _validate(self, *args, **kwargs):
        # Validation commonly drives KV usage to the configured limit. Ensure
        # online weight loading and its temporary buffers have completed before
        # validation admits requests into the rollout engine.
        self._speco_wait_pending_drafter_publish()
        return super()._validate(*args, **kwargs)
