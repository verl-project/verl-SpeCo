# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable

import torch
import sglang.srt.entrypoints.engine
from sglang.srt.utils import MultiprocessingSerializer

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False
    tl = None
    triton = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TARGET_WEIGHT_LOADER_ENV = "VERL_SGLANG_TARGET_WEIGHT_LOADER"
_DRAFT_WEIGHT_LOADER_ENV = "VERL_SGLANG_DRAFT_WEIGHT_LOADER"
_EAGLE_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_VERIFY_MODE"
_EAGLE_V1_TARGET_SAMPLING_ENV = "VERL_SGLANG_NPU_EAGLE_V1_TARGET_SAMPLING"
_EAGLE_V1_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_V1_VERIFY_MODE"
_EAGLE_FORCE_FP32_SAMPLING_ENV = "VERL_SGLANG_NPU_EAGLE_FORCE_FP32_SAMPLING"
_EAGLE_LINEAR_TRITON_ENV = "VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON"
_EAGLE_LINEAR_TRITON_DEBUG_ENV = "VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON_DEBUG"
_EAGLE_TOP_K_RENORM_FAST_PATH_ENV = "VERL_SGLANG_NPU_EAGLE_TOP_K_RENORM_FAST_PATH"
_DRAFTER_RETURN_LAST_HIDDEN_ENV = "VERL_SGLANG_DRAFTER_RETURN_LAST_HIDDEN"
_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_ENV = "VERL_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK"
_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_MAX_LOGS_ENV = (
    "VERL_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_MAX_LOGS"
)
_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_TOPK_ENV = (
    "VERL_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_TOPK"
)
_DRAFTER_RAW_TOP_LOGPROBS_ENV = "VERL_DRAFTER_RAW_TOP_LOGPROBS"
_DRAFTER_RAW_TOP_LOGPROBS_TOPK_ENV = "VERL_DRAFTER_RAW_TOP_LOGPROBS_TOPK"
_SGLANG_RETURN_ORIGINAL_LOGPROB_ENV = "SGLANG_RETURN_ORIGINAL_LOGPROB"
_DISABLE_SGLANG_PATCH_ENV = "VERL_DISABLE_SGLANG_PATCH"
_SGLANG_PATCHES_ENV = "VERL_SGLANG_PATCHES"
_SGLANG_BASE_COMPAT_PATCHES_ENV = "VERL_SGLANG_BASE_COMPAT_PATCHES"

_target_weight_loader: str | None = os.environ.get(_TARGET_WEIGHT_LOADER_ENV)
_draft_weight_loader: str | None = os.environ.get(_DRAFT_WEIGHT_LOADER_ENV)
_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = (
    sglang.srt.entrypoints.engine.run_scheduler_process
)
_ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = None
_SGLANG_EAGLE_UPDATE_PATCHED = False
_SGLANG_NPU_EAGLE_SAMPLING_PATCHED = False
_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = False
_SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = False
_SGLANG_RAW_TOP_LOGPROBS_REQUEST_GATE_PATCHED = False
_SGLANG_QWEN3_ROPE_COMPAT_PATCHED = False
_SGLANG_SCHEDULER_PROCESS_PATCHED = False
_SGLANG_EAGLE_LEGACY_ALIGNMENT_PATCHED = False
_SGLANG_DRAFTER_DECOUPLED_FORWARD_LOG_COUNT = 0
_SGLANG_DRAFTER_DECOUPLED_REPLAY_LOG_COUNT = 0
_SGLANG_LAST_HIDDEN_LOGPROB_CHECK_LOG_COUNT = 0
_SGLANG_LAST_HIDDEN_LOGPROB_CHECK_SKIP_LOG_COUNT = 0
_SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT = 0
_SCHEDULER_PROCESS_PATCH_ATTR = "_verl_patched_scheduler_process"
_SGLANG_TOP_K_ALL = 1 << 30
_SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME = "qwen3_rope_compat"
_SGLANG_PATCH_NAMES = {
    "eagle_update_weights",
    "npu_eagle_target_sampling",
    "hidden_states_tensor_output",
    _SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME,
}
_VERL_DRAFTER_HIDDEN_WINDOW_PARAM = "_verl_drafter_hidden_state_window"
_VERL_HIDDEN_STATE_FRONT_TOKENS_PARAM = "_verl_hidden_state_front_tokens_per_sample"
_VERL_HIDDEN_STATE_MAX_ROWS_PARAM = "_verl_hidden_state_max_rows"
_VERL_HIDDEN_STATE_PROMPT_LEN_PARAM = "_verl_prompt_len"
_VERL_HIDDEN_STATE_WINDOW_START_PARAM = "_verl_hidden_state_window_start"
_VERL_HIDDEN_STATE_WINDOW_END_PARAM = "_verl_hidden_state_window_end"
_VERL_HIDDEN_STATE_METADATA_MARKER = "__verl_hidden_state_metadata__"
_VERL_HIDDEN_STATES_STREAM_FINAL_ATTR = "_verl_hidden_states_stream_final"
_VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM = "_verl_drafter_return_last_hidden"
_VERL_DRAFTER_RAW_TOP_LOGPROBS_PARAM = "_verl_drafter_raw_top_logprobs"
_VERL_DRAFTER_RAW_TOP_LOGPROBS_REQUESTED_ATTR = (
    "_verl_drafter_raw_top_logprobs_requested"
)
_VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
_VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR = "_verl_drafter_last_hidden_positions"
_VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR = (
    "_verl_drafter_last_hidden_filtered_by_accept_indices"
)
_VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR = "_verl_drafter_last_hidden_materialized"
_VERL_DFLASH_RETURN_AUX_HIDDEN_PARAM = "_verl_dflash_return_aux_hidden"
_VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR = (
    "_verl_drafter_lh_check_recomputed_top_ids"
)
_VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR = (
    "_verl_drafter_lh_check_recomputed_top_logprobs"
)
_VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR = (
    "_verl_drafter_lh_check_recomputed_at_sglang_top"
)
_VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR = "_verl_drafter_lh_check_sglang_top_ids"
_VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR = (
    "_verl_drafter_lh_check_sglang_top_logprobs"
)
_VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR = "_verl_drafter_lh_check_raw_topk_ids"
_VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR = (
    "_verl_drafter_lh_check_raw_topk_logprobs"
)
_VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR = (
    "_verl_drafter_lh_check_raw_topk_positions"
)
_VERL_DRAFTER_LH_CHECK_SUMMARY_ATTR = "_verl_drafter_lh_check_summary"
_VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR = (
    "_verl_drafter_last_hidden_filter_summary"
)
_VERL_RAW_TARGET_LOGPROBS_METADATA_KEY = "raw_target_logprobs"
_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY = "raw_target_logprobs_positions"
_VERL_TARGET_LOGPROBS_SOURCE_METADATA_KEY = "target_logprobs_source"
_VERL_RAW_TOPK_LOGPROB_CHECK_METADATA_KEY = "raw_topk_logprob_check"
_VERL_TARGET_LOGPROBS_SOURCE_RAW_HIDDEN_METADATA = "raw_hidden_metadata"
_SGLANG_PATCH_SELECTION_FROM_ENV = object()


def enable_sglang_original_logprob_return() -> None:
    """Make SGLang return pre-temperature/top-p/top-k logprobs when logprobs are requested."""
    os.environ[_SGLANG_RETURN_ORIGINAL_LOGPROB_ENV] = "1"
    try:
        sampler_module = importlib.import_module("sglang.srt.layers.sampler")
        setattr(sampler_module, _SGLANG_RETURN_ORIGINAL_LOGPROB_ENV, True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not sync SGLang original-logprob sampler flag yet: %s", exc)


def _sglang_version_tuple() -> tuple[int, int, int] | None:
    raw_version = getattr(sglang, "__version__", None)
    if raw_version is None:
        return None
    version_parts = re.findall(r"\d+", str(raw_version).split("+", 1)[0])
    if not version_parts:
        return None
    values = [int(part) for part in version_parts[:3]]
    values.extend([0] * (3 - len(values)))
    return tuple(values[:3])


def _sglang_version_in_range(
    *,
    min_inclusive: tuple[int, int, int] | None = None,
    max_exclusive: tuple[int, int, int] | None = None,
) -> bool:
    version_tuple = _sglang_version_tuple()
    if version_tuple is None:
        return False
    if min_inclusive is not None and version_tuple < min_inclusive:
        return False
    if max_exclusive is not None and version_tuple >= max_exclusive:
        return False
    return True


def _sglang_needs_eagle_legacy_alignment_patch() -> bool:
    return _sglang_version_in_range(min_inclusive=(0, 5, 10), max_exclusive=(0, 5, 12))


def _sglang_needs_qwen3_rope_compat_patch() -> bool:
    return _sglang_version_in_range(min_inclusive=(0, 5, 10), max_exclusive=(0, 5, 12))


def _patch_sglang_eagle_legacy_draft_forward_source(source: str) -> str | None:
    """Match SGLang >=0.5.12 draft position timing on legacy 0.5.10/0.5.11."""

    lines = textwrap.dedent(source).splitlines(keepends=True)
    position_line = None
    search_start = 0
    for idx, line in enumerate(lines[:-1]):
        if "forward_batch.out_cache_loc = out_cache_loc[i]" not in line:
            continue
        next_line = lines[idx + 1]
        if "forward_batch.positions.add_(1)" not in next_line:
            continue
        position_line = lines.pop(idx + 1)
        search_start = idx + 1
        break
    if position_line is None:
        return None

    for idx in range(search_start, len(lines)):
        if lines[idx].strip() != "hidden_states = logits_output.hidden_states":
            continue
        lines.insert(idx + 1, position_line)
        return "".join(lines)
    return None


def _make_sglang_eagle_legacy_draft_forward_alignment_patch(original_method):
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None
    patched_source = _patch_sglang_eagle_legacy_draft_forward_source(source)
    if patched_source is None:
        return None

    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        original_method.__globals__,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_eagle_legacy_alignment = True
    return patched_method


def patch_sglang_eagle_legacy_alignment_compat() -> None:
    """Normalize legacy SGLang 0.5.10/0.5.11 EAGLE alignment to 0.5.12 semantics."""

    global _SGLANG_EAGLE_LEGACY_ALIGNMENT_PATCHED
    if (
        _SGLANG_EAGLE_LEGACY_ALIGNMENT_PATCHED
        or not _sglang_needs_eagle_legacy_alignment_patch()
    ):
        return

    targets = (
        ("sglang.srt.speculative.eagle_worker", "EAGLEWorker"),
        ("sglang.srt.speculative.multi_layer_eagle_worker", "MultiLayerEagleWorker"),
        ("sglang.srt.speculative.eagle_worker_v2", "EagleDraftWorker"),
        (
            "sglang.srt.speculative.multi_layer_eagle_worker_v2",
            "MultiLayerEagleDraftWorker",
        ),
    )
    patched_targets = []
    for module_name, class_name in targets:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            original_method = getattr(worker_cls, "draft_forward", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skip SGLang legacy EAGLE alignment patch for %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            continue
        if original_method is None:
            continue
        if getattr(original_method, "_verl_patched_eagle_legacy_alignment", False):
            patched_targets.append(f"{module_name}.{class_name}.draft_forward")
            continue
        patched_method = _make_sglang_eagle_legacy_draft_forward_alignment_patch(
            original_method
        )
        if patched_method is None:
            continue
        setattr(worker_cls, "draft_forward", patched_method)
        patched_targets.append(f"{module_name}.{class_name}.draft_forward")

    if patched_targets:
        _SGLANG_EAGLE_LEGACY_ALIGNMENT_PATCHED = True
        logger.warning(
            "SGLang legacy EAGLE alignment patch active for %s (version=%s).",
            ", ".join(patched_targets),
            getattr(sglang, "__version__", None),
        )


def configure_sglang_eagle_weight_update_patch(
    target_weight_loader: str | None,
    draft_weight_loader: str | None,
) -> None:
    global _target_weight_loader, _draft_weight_loader

    if target_weight_loader is not None:
        _target_weight_loader = target_weight_loader
        os.environ[_TARGET_WEIGHT_LOADER_ENV] = target_weight_loader
    if draft_weight_loader is not None:
        _draft_weight_loader = draft_weight_loader
        os.environ[_DRAFT_WEIGHT_LOADER_ENV] = draft_weight_loader


def _get_route_markers() -> tuple[str | None, str | None]:
    return (
        _target_weight_loader or os.environ.get(_TARGET_WEIGHT_LOADER_ENV),
        _draft_weight_loader or os.environ.get(_DRAFT_WEIGHT_LOADER_ENV),
    )


def _speco_drafter_runtime_enabled() -> bool:
    raw = os.environ.get("VERL_SPECO_SGLANG_DRAFTER_CONFIG")
    if not raw:
        return False
    try:
        import json

        loaded = json.loads(raw)
    except Exception:  # noqa: BLE001
        return True
    return bool(isinstance(loaded, dict) and loaded.get("enable"))


def _get_sglang_worker_tp_rank(worker) -> int:
    for obj in (
        worker,
        getattr(worker, "model_runner", None),
        getattr(worker, "draft_model_runner", None),
        getattr(worker, "target_worker", None),
        getattr(worker, "target_model_worker", None),
    ):
        if obj is None:
            continue
        for attr_name in ("tp_rank", "rank"):
            attr = getattr(obj, attr_name, None)
            if attr is not None:
                return int(attr)
    return 0


def _get_sglang_draft_runner(worker):
    for attr_name in ("draft_model_runner", "model_runner"):
        runner = getattr(worker, attr_name, None)
        if runner is not None:
            return runner
    for worker_attr in ("draft_worker", "_draft_worker"):
        draft_worker = getattr(worker, worker_attr, None)
        if draft_worker is None:
            continue
        for runner_attr in ("draft_runner", "model_runner"):
            runner = getattr(draft_worker, runner_attr, None)
            if runner is not None:
                return runner
    return None


def _get_sglang_target_runner(worker):
    for worker_attr in ("target_worker", "target_model_worker"):
        target_worker = getattr(worker, worker_attr, None)
        if target_worker is None:
            continue
        runner = getattr(target_worker, "model_runner", None)
        if runner is not None:
            return runner
    return None


_SGLANG_QWEN3_ROPE_PARAMETERS_PATTERN = re.compile(
    r'(?m)^(?P<indent>[ \t]+)rope_theta = config\.rope_parameters\["rope_theta"\]\s*\n'
    r"(?P=indent)rope_scaling = config\.rope_parameters\s*$"
)
_SGLANG_QWEN3_ZERO_ARG_SUPER_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]+)super\(\)\.__init__\(\)\s*$"
)


def _render_sglang_qwen3_rope_compat(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}if (\n"
        f'{indent}    hasattr(config, "rope_parameters")\n'
        f"{indent}    and config.rope_parameters\n"
        f'{indent}    and "rope_theta" in config.rope_parameters\n'
        f"{indent}):\n"
        f'{indent}    rope_theta = config.rope_parameters["rope_theta"]\n'
        f"{indent}    rope_scaling = config.rope_parameters\n"
        f"{indent}else:\n"
        f'{indent}    rope_theta = getattr(config, "rope_theta", 1000000)\n'
        f'{indent}    rope_scaling = getattr(config, "rope_scaling", None)'
    )


def _patch_sglang_qwen3_decoder_layer_init_source(source: str) -> str | None:
    patched_source, replacement_count = _SGLANG_QWEN3_ROPE_PARAMETERS_PATTERN.subn(
        _render_sglang_qwen3_rope_compat,
        source,
        count=1,
    )
    if replacement_count <= 0 or patched_source == source:
        return None
    patched_source = _SGLANG_QWEN3_ZERO_ARG_SUPER_PATTERN.sub(
        r"\g<indent>super(Qwen3DecoderLayer, self).__init__()",
        patched_source,
        count=1,
    )
    return patched_source


def _make_sglang_qwen3_decoder_layer_init_patch(original_init):
    try:
        source = inspect.getsource(original_init)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source = _patch_sglang_qwen3_decoder_layer_init_source(source)
    if patched_source is None:
        return None

    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        original_init.__globals__,
        namespace,
    )
    patched_init = namespace[original_init.__name__]
    patched_init = wraps(original_init)(patched_init)
    patched_init._verl_patched_qwen3_rope_compat = True
    return patched_init


def patch_sglang_qwen3_rope_compat() -> None:
    """Patch SGLang 0.5.10 Qwen3 to accept transformers configs without rope_parameters."""
    global _SGLANG_QWEN3_ROPE_COMPAT_PATCHED
    if _SGLANG_QWEN3_ROPE_COMPAT_PATCHED:
        return

    try:
        qwen3_module = importlib.import_module("sglang.srt.models.qwen3")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang Qwen3 rope compatibility patch: %s", exc)
        return

    qwen3_decoder_layer = getattr(qwen3_module, "Qwen3DecoderLayer", None)
    original_init = getattr(qwen3_decoder_layer, "__init__", None)
    if original_init is None:
        return
    if getattr(original_init, "_verl_patched_qwen3_rope_compat", False):
        _SGLANG_QWEN3_ROPE_COMPAT_PATCHED = True
        return

    patched_init = _make_sglang_qwen3_decoder_layer_init_patch(original_init)
    if patched_init is None:
        return

    qwen3_decoder_layer.__init__ = patched_init
    _SGLANG_QWEN3_ROPE_COMPAT_PATCHED = True
    logger.warning("SGLang Qwen3 rope-parameters compatibility patch active.")


def _make_verl_eagle_update_weights_patch(original_update_weights):
    @wraps(original_update_weights)
    def patched_update_weights_from_tensor(self, recv_req):
        target_weight_loader, draft_weight_loader = _get_route_markers()
        load_format = getattr(recv_req, "load_format", None)
        # In upstream verl target weight sync normally arrives without SPECO
        # route flags. Once SPECO runtime is enabled, treat that unmarked
        # update as target-only so EAGLE draft parameters are not updated with
        # target model tensors. Do not depend solely on custom loader markers:
        # they may be unavailable in older or locally modified SGLang builds.
        unmarked_target_sync = load_format is None and (
            target_weight_loader is not None or _speco_drafter_runtime_enabled()
        )
        target_only = (
            target_weight_loader is not None
            and load_format == target_weight_loader
            or unmarked_target_sync
        )
        draft_only = (
            draft_weight_loader is not None and load_format == draft_weight_loader
        )
        disable_draft_model = (
            bool(getattr(recv_req, "disable_draft_model", False)) or target_only
        )
        disable_target_model = (
            bool(getattr(recv_req, "disable_target_model", False)) or draft_only
        )

        if not (disable_draft_model or disable_target_model):
            return original_update_weights(self, recv_req)

        if disable_draft_model and disable_target_model:
            return False, "Both target and draft model updates are disabled."

        serialized_named_tensors = getattr(recv_req, "serialized_named_tensors", None)
        if not serialized_named_tensors:
            return True, "No tensor is provided for routed EAGLE weight update."

        tp_rank = _get_sglang_worker_tp_rank(self)
        if tp_rank >= len(serialized_named_tensors):
            return (
                False,
                "Invalid routed EAGLE update tensor shard index: "
                f"tp_rank={tp_rank}, num_shards={len(serialized_named_tensors)}.",
            )

        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            serialized_named_tensors[tp_rank]
        )

        # The verl-only custom loaders are route markers. After EAGLEWorker
        # selects the correct side, let that model runner use its normal loader.
        routed_load_format = None if target_only or draft_only else load_format

        if not disable_draft_model:
            draft_runner = _get_sglang_draft_runner(self)
            if draft_runner is None:
                return False, "SGLang EAGLE draft model runner is missing."
            success, message = draft_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=routed_load_format,
            )
            if not success:
                return success, message

        if not disable_target_model:
            target_runner = _get_sglang_target_runner(self)
            if target_runner is None:
                return False, "SGLang EAGLE target model runner is missing."
            success, message = target_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=routed_load_format,
            )
            if not success:
                return success, message

        return True, "Routed EAGLE weight update succeeded."

    patched_update_weights_from_tensor._verl_patched_eagle_update_weights = True
    return patched_update_weights_from_tensor


def patch_sglang_eagle_update_weights_from_tensor() -> None:
    """Patch SGLang EAGLE update so target-only and draft-only sync skip the wrong side early."""
    global _SGLANG_EAGLE_UPDATE_PATCHED
    if _SGLANG_EAGLE_UPDATE_PATCHED:
        return

    patched_classes = []
    for module_name in (
        "sglang.srt.speculative.eagle_worker",
        "sglang.srt.speculative.eagle_worker_v2",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for class_name, cls in vars(module).items():
            if not isinstance(cls, type) or not class_name.lower().startswith("eagle"):
                continue

            original_update_weights = getattr(cls, "update_weights_from_tensor", None)
            if original_update_weights is None or getattr(
                original_update_weights, "_verl_patched_eagle_update_weights", False
            ):
                continue

            cls.update_weights_from_tensor = _make_verl_eagle_update_weights_patch(
                original_update_weights
            )
            patched_classes.append(f"{module_name}.{class_name}")

    if patched_classes:
        _SGLANG_EAGLE_UPDATE_PATCHED = True
        logger.info(
            "Patched SGLang EAGLE routed weight update for %s",
            ", ".join(patched_classes),
        )


def _is_sglang_npu_backend() -> bool:
    for module_name in ("sglang.srt.utils.common", "sglang.srt.utils"):
        try:
            is_npu = getattr(importlib.import_module(module_name), "is_npu", None)
        except Exception:  # noqa: BLE001
            continue
        if callable(is_npu) and is_npu():
            return True

    return hasattr(torch, "npu") and torch.npu.is_available()


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n", ""}:
        return False
    return default


def _sglang_npu_eagle_force_fp32_sampling_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_FORCE_FP32_SAMPLING_ENV, default=False)


def _sglang_npu_eagle_linear_triton_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_LINEAR_TRITON_ENV, default=True)


def _sglang_npu_eagle_linear_triton_debug_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_LINEAR_TRITON_DEBUG_ENV, default=False)


def _sglang_drafter_return_last_hidden_enabled() -> bool:
    return _env_flag_enabled(_DRAFTER_RETURN_LAST_HIDDEN_ENV, default=False)


def _sglang_last_hidden_logprob_check_enabled() -> bool:
    return _env_flag_enabled(_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_ENV, default=False)


def _sglang_raw_top_logprobs_enabled(forward_batch=None) -> bool:
    if not _env_flag_enabled(_DRAFTER_RAW_TOP_LOGPROBS_ENV, default=False):
        return False
    if forward_batch is None:
        return True
    if _sglang_forward_mode_is_draft_extend(forward_batch):
        return False
    return _sglang_forward_batch_requests_raw_top_logprobs(forward_batch)


def _sglang_raw_top_logprobs_topk() -> int | None:
    raw_value = os.getenv(_DRAFTER_RAW_TOP_LOGPROBS_TOPK_ENV)
    if raw_value is None:
        return None
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return None


def _sglang_last_hidden_logprob_check_max_logs() -> int:
    raw_value = os.getenv(_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_MAX_LOGS_ENV)
    if raw_value is None:
        return 8
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 8


def _sglang_last_hidden_logprob_check_topk() -> int:
    raw_value = os.getenv(_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK_TOPK_ENV)
    if raw_value is None:
        return 5
    try:
        return max(1, min(int(raw_value), 64))
    except (TypeError, ValueError):
        return 5


def _log_sglang_last_hidden_logprob_check_skip(reason: str, **details: Any) -> None:
    global _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_SKIP_LOG_COUNT
    if not _sglang_last_hidden_logprob_check_enabled():
        return
    if (
        _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_SKIP_LOG_COUNT
        >= _sglang_last_hidden_logprob_check_max_logs()
    ):
        return
    logger.warning(
        "[sglang last_hidden logprob check skip] reason=%s details=%s", reason, details
    )
    _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_SKIP_LOG_COUNT += 1


def _debug_sglang_npu_eagle_linear_triton(reason: str, **details: Any) -> None:
    if not _sglang_npu_eagle_linear_triton_debug_enabled():
        return
    logger.warning(
        "SGLang NPU EAGLE linear Triton skip: %s details=%s", reason, details
    )


def _debug_sglang_npu_eagle_linear_triton_exception(
    reason: str, exc: Exception, **details: Any
) -> None:
    if not _sglang_npu_eagle_linear_triton_debug_enabled():
        return
    details = {
        **details,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "repr": repr(exc),
    }
    logger.warning(
        "SGLang NPU EAGLE linear Triton exception: %s details=%s",
        reason,
        details,
        exc_info=True,
    )


def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
    }


def _triton_property_int(properties: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in properties:
            continue
        try:
            return int(properties[key])
        except (TypeError, ValueError):
            continue
    return None


def _triton_ascend_backend_module_available() -> bool:
    try:
        return importlib.util.find_spec("triton.backends.ascend") is not None
    except Exception:  # noqa: BLE001
        return False


def _triton_ascend_available() -> bool:
    if not _HAS_TRITON:
        _debug_sglang_npu_eagle_linear_triton("triton_import_unavailable")
        return False
    if not _is_sglang_npu_backend():
        _debug_sglang_npu_eagle_linear_triton("sglang_backend_not_npu")
        return False

    try:
        active_driver = triton.runtime.driver.active
    except Exception as exc:  # noqa: BLE001
        _debug_sglang_npu_eagle_linear_triton_exception(
            "triton_active_driver_unavailable",
            exc,
        )
        return False

    target_backend = None
    try:
        target = active_driver.get_current_target()
        target_backend = getattr(target, "backend", None)
    except Exception as exc:  # noqa: BLE001
        _debug_sglang_npu_eagle_linear_triton_exception(
            "triton_target_unavailable",
            exc,
        )
    if target_backend is not None and target_backend != "npu":
        _debug_sglang_npu_eagle_linear_triton(
            "triton_target_not_npu",
            target_backend=target_backend,
        )
        return False

    try:
        if hasattr(active_driver, "get_current_device"):
            device = active_driver.get_current_device()
        else:
            device = torch.npu.current_device()
        properties = active_driver.utils.get_device_properties(device)
    except Exception as exc:  # noqa: BLE001
        has_ascend_backend = _triton_ascend_backend_module_available()
        _debug_sglang_npu_eagle_linear_triton_exception(
            "triton_device_properties_unavailable",
            exc,
            target_backend=target_backend,
            has_ascend_backend=has_ascend_backend,
        )
        logger.debug("Triton Ascend device properties are unavailable: %s", exc)
        return False

    if not isinstance(properties, dict):
        _debug_sglang_npu_eagle_linear_triton(
            "triton_device_properties_invalid",
            properties_type=type(properties).__name__,
        )
        return False

    aicore = _triton_property_int(
        properties,
        "num_aicore",
        "num_ai_core",
        "num_aic",
        "aicore_num",
        "ai_core_num",
    )
    vectorcore = _triton_property_int(
        properties,
        "num_vectorcore",
        "num_vector_core",
        "vectorcore_num",
        "vector_core_num",
    )
    if aicore is not None or vectorcore is not None:
        available = (aicore or 0) > 0 or (vectorcore or 0) > 0
        if not available:
            _debug_sglang_npu_eagle_linear_triton(
                "triton_core_count_unavailable",
                aicore=aicore,
                vectorcore=vectorcore,
                properties=properties,
            )
        return available

    # Some Triton Ascend builds expose a non-empty properties dict without the
    # exact core-count keys above. In NPU mode, prefer trying the kernel and
    # falling back through the existing exception guard if launch fails.
    return bool(properties)


def _sglang_npu_eagle_top_k_renorm_fast_path_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_TOP_K_RENORM_FAST_PATH_ENV, default=False)


def _sglang_verl_patches_disabled() -> bool:
    return _env_flag_enabled(_DISABLE_SGLANG_PATCH_ENV, default=False)


def _normalize_sglang_patch_selection(raw_value: str | None) -> set[str] | None:
    if raw_value is None or not raw_value.strip():
        return None

    patch_names = [
        item.strip().lower()
        for item in re.split(r"[\s,]+", raw_value.strip())
        if item.strip()
    ]
    if not patch_names:
        return None

    invalid_names = sorted(
        patch_name
        for patch_name in patch_names
        if patch_name not in {"all", "none"} and patch_name not in _SGLANG_PATCH_NAMES
    )
    if invalid_names:
        raise ValueError(
            f"Unknown SGLang patch '{invalid_names[0]}' in {_SGLANG_PATCHES_ENV}. "
            f"Supported values are: all, none, {', '.join(sorted(_SGLANG_PATCH_NAMES))}."
        )

    special_names = {"all", "none"}.intersection(patch_names)
    if special_names:
        special_name = "all" if "all" in special_names else "none"
        if len(patch_names) > 1:
            raise ValueError(
                f"{_SGLANG_PATCHES_ENV}={special_name} cannot be combined with other patch names. "
                f"Use either a single special value or a list of canonical patch names."
            )
        patch_name = patch_names[0]
        if patch_name == "all":
            return None
        if patch_name == "none":
            return set()

    return set(patch_names)


def _selected_sglang_patches() -> set[str] | None:
    return _normalize_sglang_patch_selection(os.getenv(_SGLANG_PATCHES_ENV))


def _normalize_sglang_patch_names(
    patches: Iterable[str] | str | None,
) -> set[str] | None:
    if patches is None:
        return None
    if isinstance(patches, str):
        return _normalize_sglang_patch_selection(patches)

    patch_names = ",".join(str(patch_name) for patch_name in patches)
    if not patch_names.strip():
        return set()
    return _normalize_sglang_patch_selection(patch_names)


def _set_sglang_patch_selection_env(patches: set[str] | None) -> None:
    if patches is None:
        os.environ.pop(_SGLANG_PATCHES_ENV, None)
        return
    os.environ[_SGLANG_PATCHES_ENV] = ",".join(sorted(patches)) if patches else "none"


def _selected_sglang_base_compat_patches() -> set[str]:
    raw_value = os.getenv(_SGLANG_BASE_COMPAT_PATCHES_ENV)
    if raw_value is None or not raw_value.strip():
        return set()

    patch_names = {
        item.strip().lower()
        for item in re.split(r"[\s,]+", raw_value.strip())
        if item.strip()
    }
    if "all" in patch_names:
        return {_SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME}
    return {
        patch_name
        for patch_name in patch_names
        if patch_name == _SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME
    }


def _sglang_patch_enabled(
    patch_name: str,
    selected: set[str] | None | object = _SGLANG_PATCH_SELECTION_FROM_ENV,
) -> bool:
    if selected is _SGLANG_PATCH_SELECTION_FROM_ENV:
        selected = _selected_sglang_patches()
    return selected is None or patch_name in selected


def _normalize_sglang_npu_eagle_verify_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode in {"0", "false", "off", "greedy"}:
        return "greedy"
    if normalized_mode in {"1", "true", "on", "target", "target_only"}:
        return "target_only"
    return None


def _sglang_npu_eagle_verify_mode(version_env: str | None = None) -> str:
    mode = (
        _normalize_sglang_npu_eagle_verify_mode(os.getenv(version_env))
        if version_env
        else None
    )
    if mode is not None:
        return mode
    mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(_EAGLE_VERIFY_MODE_ENV))
    if mode is not None:
        return mode
    if version_env == _EAGLE_V1_VERIFY_MODE_ENV:
        legacy_mode = _normalize_sglang_npu_eagle_verify_mode(
            os.getenv(_EAGLE_V1_TARGET_SAMPLING_ENV)
        )
        if legacy_mode is not None:
            return legacy_mode
    return "target_only"


def _sglang_npu_eagle_v1_verify_mode() -> str:
    return _sglang_npu_eagle_verify_mode(_EAGLE_V1_VERIFY_MODE_ENV)


def _as_sglang_npu_eagle_sampling_float(tensor: torch.Tensor) -> torch.Tensor:
    if _sglang_npu_eagle_force_fp32_sampling_enabled() and tensor.is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor


def _renorm_probs_by_top_k_top_p(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    top_ks = top_ks.to(device=probs_for_sampling.device, dtype=torch.long).view(-1)
    top_ps = top_ps.to(
        device=probs_for_sampling.device, dtype=probs_for_sampling.dtype
    ).view(-1)

    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where(
        (top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks
    )
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()) and bool(
        torch.all(top_ps >= 1.0).item()
    ):
        return probs_for_sampling

    sorted_probs, sorted_indices = torch.sort(
        probs_for_sampling, dim=-1, descending=True
    )
    ranks = torch.arange(vocab_size, device=probs_for_sampling.device).view(1, -1)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs = sorted_probs.masked_fill(
        (cumulative_probs - sorted_probs) > top_ps.view(-1, 1), 0.0
    )

    normalizer = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = torch.where(
        normalizer > 0,
        sorted_probs / normalizer.clamp_min(torch.finfo(probs_for_sampling.dtype).tiny),
        0.0,
    )

    renormed_probs = torch.zeros_like(probs_for_sampling)
    renormed_probs.scatter_(dim=1, index=sorted_indices, src=sorted_probs)
    return renormed_probs


def _top_k_renorm_prob_torch_fast(
    probs: torch.Tensor, top_ks: torch.Tensor
) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    if probs_for_sampling.numel() == 0:
        return probs_for_sampling

    top_ks = top_ks.to(device=probs_for_sampling.device, dtype=torch.long).view(-1)
    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where(
        (top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks
    )
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()):
        return probs_for_sampling

    fast_row_indices = torch.nonzero(top_ks < vocab_size, as_tuple=False).view(-1)
    if fast_row_indices.numel() == 0:
        return probs_for_sampling

    fast_top_ks = top_ks.index_select(0, fast_row_indices)
    max_top_k = int(fast_top_ks.max().item())
    if max_top_k <= 0:
        return probs_for_sampling

    fast_probs = probs_for_sampling.index_select(0, fast_row_indices)
    topk_probs, topk_indices = torch.topk(fast_probs, max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=probs_for_sampling.device).view(1, -1)
    topk_probs = topk_probs.masked_fill(ranks >= fast_top_ks.view(-1, 1), 0.0)

    normalizer = topk_probs.sum(dim=-1, keepdim=True)
    topk_probs = torch.where(
        normalizer > 0,
        topk_probs / normalizer.clamp_min(torch.finfo(probs_for_sampling.dtype).tiny),
        0.0,
    )

    fast_renormed_probs = torch.zeros_like(fast_probs)
    fast_renormed_probs.scatter_(dim=1, index=topk_indices, src=topk_probs)
    if fast_row_indices.numel() == probs_for_sampling.shape[0]:
        return fast_renormed_probs

    renormed_probs = probs_for_sampling.clone()
    renormed_probs.index_copy_(0, fast_row_indices, fast_renormed_probs)
    return renormed_probs


def _top_k_renorm_prob_torch(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    if _sglang_npu_eagle_top_k_renorm_fast_path_enabled():
        try:
            return _top_k_renorm_prob_torch_fast(probs, top_ks)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "SGLang NPU top-k renorm fast path failed; falling back to sort path: %s",
                exc,
            )
    top_ps = torch.ones(
        (probs.shape[0],),
        dtype=torch.float32
        if _sglang_npu_eagle_force_fp32_sampling_enabled()
        else probs.dtype,
        device=probs.device,
    )
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_ks = torch.full(
        (probs.shape[0],), probs.shape[-1], dtype=torch.long, device=probs.device
    )
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _sample_from_probs_with_coin(
    probs: torch.Tensor, coin: torch.Tensor
) -> torch.Tensor:
    squeeze_output = probs.dim() == 1
    if squeeze_output:
        probs = probs.unsqueeze(0)

    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    totals = probs_for_sampling.sum(dim=-1, keepdim=True)
    probs_for_sampling = torch.where(
        totals > 0,
        probs_for_sampling,
        torch.ones_like(probs_for_sampling),
    )
    totals = probs_for_sampling.sum(dim=-1, keepdim=True)
    threshold = (
        coin.to(
            dtype=probs_for_sampling.dtype,
            device=probs_for_sampling.device,
        ).view(-1, 1)
        * totals
    )
    cumulative = torch.cumsum(probs_for_sampling, dim=-1)
    samples = torch.argmax((cumulative > threshold).to(torch.int32), dim=-1).to(
        torch.int32
    )
    return samples[0] if squeeze_output else samples


_tree_speculative_sampling_target_only_linear_triton_kernel = None
if triton is not None:

    @triton.jit(do_not_specialize=["threshold_single", "threshold_acc"])
    def _tree_speculative_sampling_target_only_linear_triton_kernel(
        predicts_ptr,
        accept_index_ptr,
        accept_token_num_ptr,
        candidates_ptr,
        retrive_index_ptr,
        retrive_next_token_ptr,
        uniform_samples_ptr,
        uniform_samples_for_final_sampling_ptr,
        target_probs_ptr,
        vocab_size,
        threshold_single,
        threshold_acc,
        NUM_DRAFT_TOKENS: tl.constexpr,
        NUM_SPECULATIVE_TOKENS: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
        SPEC_BLOCK: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        row_base = req_idx * NUM_DRAFT_TOKENS
        accept_index_base = req_idx * NUM_SPECULATIVE_TOKENS

        spec_offsets = tl.arange(0, SPEC_BLOCK)
        tl.store(
            accept_index_ptr + accept_index_base + spec_offsets,
            -1,
            mask=spec_offsets < NUM_SPECULATIVE_TOKENS,
        )
        tl.store(accept_token_num_ptr + req_idx, 0)

        threshold_acc = tl.maximum(threshold_acc, 1.0e-9)
        cur_prob_idx = tl.full((), 0, tl.int64)
        accepted_count = tl.full((), 0, tl.int32)
        active = tl.full((), True, tl.int1)
        residual_token_id = tl.full((), -1, tl.int64)
        residual_token_prob = tl.full((), 0.0, tl.float32)

        last_accepted_retrive_idx = tl.load(retrive_index_ptr + row_base)
        tl.store(accept_index_ptr + accept_index_base, last_accepted_retrive_idx)
        coin = tl.load(uniform_samples_ptr + row_base).to(tl.float32)

        for _ in range(1, NUM_SPECULATIVE_TOKENS):
            next_idx = tl.load(retrive_next_token_ptr + row_base + cur_prob_idx)
            valid = active & (next_idx >= 0)
            safe_next_idx = tl.maximum(next_idx, 0)
            draft_token_id = tl.load(candidates_ptr + row_base + safe_next_idx).to(
                tl.int64
            )
            target_prob_single = tl.load(
                target_probs_ptr
                + (row_base + cur_prob_idx) * vocab_size
                + draft_token_id,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            accepted = valid & (
                (coin <= (target_prob_single / threshold_acc))
                | (target_prob_single >= threshold_single)
            )

            if accepted:
                accepted_retrive_idx = tl.load(
                    retrive_index_ptr + row_base + safe_next_idx
                )
                tl.store(predicts_ptr + last_accepted_retrive_idx, draft_token_id)
                accepted_count += 1
                tl.store(
                    accept_index_ptr + accept_index_base + accepted_count,
                    accepted_retrive_idx,
                    mask=accepted_count < NUM_SPECULATIVE_TOKENS,
                )
                cur_prob_idx = safe_next_idx
                last_accepted_retrive_idx = accepted_retrive_idx
                coin = tl.load(uniform_samples_ptr + row_base + cur_prob_idx).to(
                    tl.float32
                )
                residual_token_id = tl.full((), -1, tl.int64)
                residual_token_prob = tl.full((), 0.0, tl.float32)
            else:
                if valid:
                    residual_token_id = draft_token_id
                    residual_token_prob = target_prob_single
                active = tl.full((), False, tl.int1)

        tl.store(accept_token_num_ptr + req_idx, accepted_count)

        final_row_base = (row_base + cur_prob_idx) * vocab_size
        need_residual = (accepted_count != (NUM_SPECULATIVE_TOKENS - 1)) & (
            residual_token_id >= 0
        )
        num_vocab_blocks = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        total = tl.full((), 0.0, tl.float32)
        vocab_offsets = tl.arange(0, SUB_BLOCK)
        for block_idx in range(num_vocab_blocks):
            token_offsets = block_idx * SUB_BLOCK + vocab_offsets
            mask = token_offsets < vocab_size
            probs = tl.load(
                target_probs_ptr + final_row_base + token_offsets, mask=mask, other=0.0
            ).to(tl.float32)
            if need_residual:
                probs = tl.where(
                    token_offsets == residual_token_id,
                    tl.maximum(probs - residual_token_prob, 0.0),
                    probs,
                )
            total += tl.sum(probs, axis=0)

        final_coin = tl.load(uniform_samples_for_final_sampling_ptr + req_idx).to(
            tl.float32
        )
        last_vocab_token_id = (vocab_size - 1).to(tl.int64)
        final_token_id = last_vocab_token_id
        if total <= 0.0:
            final_token_id = tl.minimum(
                (final_coin * vocab_size).to(tl.int64), last_vocab_token_id
            )
        else:
            threshold = final_coin * total
            cumulative = tl.full((), 0.0, tl.float32)
            found = tl.full((), False, tl.int1)
            for block_idx in range(num_vocab_blocks):
                token_offsets = block_idx * SUB_BLOCK + vocab_offsets
                mask = token_offsets < vocab_size
                probs = tl.load(
                    target_probs_ptr + final_row_base + token_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                if need_residual:
                    probs = tl.where(
                        token_offsets == residual_token_id,
                        tl.maximum(probs - residual_token_prob, 0.0),
                        probs,
                    )
                cumulative_probs = tl.cumsum(probs, 0) + cumulative
                hits = (cumulative_probs > threshold) & mask
                hit_values = hits.to(tl.int32)
                has_hit = tl.max(hit_values, axis=0) > 0
                if has_hit & (~found):
                    final_token_id = (
                        block_idx * SUB_BLOCK + tl.argmax(hit_values, axis=0)
                    ).to(tl.int64)
                found = found | has_hit
                cumulative += tl.sum(probs, axis=0)

        tl.store(predicts_ptr + last_accepted_retrive_idx, final_token_id)


def _triton_next_power_of_2(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _try_tree_speculative_sampling_target_only_linear_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> bool:
    linear_triton_enabled = _sglang_npu_eagle_linear_triton_enabled()
    if not linear_triton_enabled:
        _debug_sglang_npu_eagle_linear_triton("linear_triton_disabled")
        return False

    triton_ascend_available = _triton_ascend_available()
    if not triton_ascend_available:
        _debug_sglang_npu_eagle_linear_triton(
            "triton_ascend_unavailable",
        )
        return False
    if _tree_speculative_sampling_target_only_linear_triton_kernel is None:
        _debug_sglang_npu_eagle_linear_triton("kernel_not_compiled")
        return False
    if target_probs.device.type != "npu":
        _debug_sglang_npu_eagle_linear_triton(
            "target_probs_not_npu",
            target_probs=_tensor_debug_summary(target_probs),
        )
        return False
    if target_probs.ndim != 3 or candidates.ndim != 2:
        _debug_sglang_npu_eagle_linear_triton(
            "unexpected_rank",
            target_probs=_tensor_debug_summary(target_probs),
            candidates=_tensor_debug_summary(candidates),
        )
        return False

    input_tensors = {
        "predicts": predicts,
        "accept_index": accept_index,
        "accept_token_num": accept_token_num,
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "uniform_samples": uniform_samples,
        "uniform_samples_for_final_sampling": uniform_samples_for_final_sampling,
        "target_probs": target_probs,
    }
    non_contiguous_inputs = [
        name for name, tensor in input_tensors.items() if not tensor.is_contiguous()
    ]
    if non_contiguous_inputs:
        _debug_sglang_npu_eagle_linear_triton(
            "non_contiguous_inputs",
            tensors=non_contiguous_inputs,
        )
        return False

    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    vocab_size = target_probs.shape[-1]
    if (
        batch_size == 0
        or num_draft_tokens == 0
        or num_speculative_tokens == 0
        or target_probs.shape[:2] != candidates.shape
        or uniform_samples.shape != candidates.shape
    ):
        _debug_sglang_npu_eagle_linear_triton(
            "unexpected_shape",
            batch_size=int(batch_size),
            num_draft_tokens=int(num_draft_tokens),
            num_speculative_tokens=int(num_speculative_tokens),
            target_probs=_tensor_debug_summary(target_probs),
            candidates=_tensor_debug_summary(candidates),
            uniform_samples=_tensor_debug_summary(uniform_samples),
        )
        return False

    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    if not (
        target_probs_for_sampling.is_contiguous()
        and uniform_samples_for_sampling.is_contiguous()
        and final_uniform_samples_for_sampling.is_contiguous()
    ):
        _debug_sglang_npu_eagle_linear_triton(
            "non_contiguous_sampling_inputs",
            target_probs_for_sampling=_tensor_debug_summary(target_probs_for_sampling),
            uniform_samples_for_sampling=_tensor_debug_summary(
                uniform_samples_for_sampling
            ),
            final_uniform_samples_for_sampling=_tensor_debug_summary(
                final_uniform_samples_for_sampling
            ),
        )
        return False

    sub_block = 4096
    kernel_meta = {
        "batch_size": int(batch_size),
        "num_draft_tokens": int(num_draft_tokens),
        "num_speculative_tokens": int(num_speculative_tokens),
        "vocab_size": int(vocab_size),
        "sub_block": int(sub_block),
        "num_vocab_blocks": (int(vocab_size) + sub_block - 1) // sub_block,
        "spec_block": _triton_next_power_of_2(num_speculative_tokens),
        "multibuffer": False,
        "threshold_single": float(threshold_single),
        "threshold_acc": float(threshold_acc),
    }
    try:
        _tree_speculative_sampling_target_only_linear_triton_kernel[(batch_size,)](
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            uniform_samples_for_sampling,
            final_uniform_samples_for_sampling,
            target_probs_for_sampling,
            kernel_meta["vocab_size"],
            float(threshold_single),
            max(float(threshold_acc), 1.0e-9),
            NUM_DRAFT_TOKENS=kernel_meta["num_draft_tokens"],
            NUM_SPECULATIVE_TOKENS=kernel_meta["num_speculative_tokens"],
            SUB_BLOCK=kernel_meta["sub_block"],
            SPEC_BLOCK=kernel_meta["spec_block"],
            multibuffer=kernel_meta["multibuffer"],
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _debug_sglang_npu_eagle_linear_triton_exception(
            "kernel_launch_failed",
            exc,
            kernel="_tree_speculative_sampling_target_only_linear_triton_kernel",
            meta=kernel_meta,
            predicts=_tensor_debug_summary(predicts),
            accept_index=_tensor_debug_summary(accept_index),
            accept_token_num=_tensor_debug_summary(accept_token_num),
            candidates=_tensor_debug_summary(candidates),
            retrive_index=_tensor_debug_summary(retrive_index),
            retrive_next_token=_tensor_debug_summary(retrive_next_token),
            uniform_samples_for_sampling=_tensor_debug_summary(
                uniform_samples_for_sampling
            ),
            final_uniform_samples_for_sampling=_tensor_debug_summary(
                final_uniform_samples_for_sampling
            ),
            target_probs_for_sampling=_tensor_debug_summary(target_probs_for_sampling),
        )
        logger.debug(
            "SGLang NPU EAGLE linear target-only Triton kernel failed: %s", exc
        )
        return False


def _tree_speculative_sampling_target_only_linear_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    del draft_probs, deterministic, retrive_next_sibling

    batch_size, _ = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    device = target_probs_for_sampling.device

    threshold_acc = max(float(threshold_acc), 1e-9)
    threshold_single = float(threshold_single)

    accept_index.fill_(-1)
    accept_token_num.zero_()

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    inactive_next_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    reset_retrive_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    reset_residual_prob = torch.zeros(
        (batch_size,),
        dtype=target_probs_for_sampling.dtype,
        device=device,
    )

    cur_prob_idx = torch.zeros((batch_size,), dtype=torch.long, device=device)
    accepted_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)

    last_accepted_retrive_idx = retrive_index[:, 0].to(torch.long)
    accept_index[:, 0].copy_(last_accepted_retrive_idx.to(dtype=accept_index.dtype))
    coin = uniform_samples_for_sampling[:, 0]
    residual_token_id = reset_retrive_idx.clone()
    residual_token_prob = reset_residual_prob.clone()

    for _ in range(1, num_speculative_tokens):
        next_idx = torch.where(
            active,
            retrive_next_token[batch_indices, cur_prob_idx],
            inactive_next_idx,
        )
        valid = active & (next_idx >= 0)
        safe_next_idx = next_idx.clamp_min(0)
        draft_token_id = candidates[batch_indices, safe_next_idx].to(torch.long)
        target_prob_single = target_probs_for_sampling[
            batch_indices,
            cur_prob_idx,
            draft_token_id,
        ]
        target_prob_single = torch.where(
            valid, target_prob_single, torch.zeros_like(target_prob_single)
        )
        accepted = valid & (
            (coin <= (target_prob_single / threshold_acc))
            | (target_prob_single >= threshold_single)
        )
        rejected = valid & ~accepted

        residual_token_id = torch.where(rejected, draft_token_id, residual_token_id)
        residual_token_prob = torch.where(
            rejected, target_prob_single, residual_token_prob
        )

        accepted_retrive_idx = retrive_index[batch_indices, safe_next_idx].to(
            torch.long
        )
        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(
            accepted, draft_token_id.to(dtype=predicts.dtype), old_predicts
        )
        predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=predict_updates)

        next_accepted_count = accepted_count + accepted.to(dtype=torch.long)
        accept_index_position = next_accepted_count.clamp_max(
            num_speculative_tokens - 1
        ).view(-1, 1)
        old_accept_index = accept_index.gather(
            dim=1, index=accept_index_position
        ).squeeze(1)
        accept_index_updates = torch.where(
            accepted,
            accepted_retrive_idx.to(dtype=accept_index.dtype),
            old_accept_index,
        )
        accept_index.scatter_(
            dim=1, index=accept_index_position, src=accept_index_updates.view(-1, 1)
        )

        cur_prob_idx = torch.where(accepted, safe_next_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(
            accepted, accepted_retrive_idx, last_accepted_retrive_idx
        )
        accepted_count = next_accepted_count
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_token_id = torch.where(accepted, reset_retrive_idx, residual_token_id)
        residual_token_prob = torch.where(
            accepted, reset_residual_prob, residual_token_prob
        )

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    need_residual = accepted_count != (num_speculative_tokens - 1)
    residual_mask = need_residual & (residual_token_id >= 0)
    if bool(residual_mask.any().item()):
        final_probs = final_target_probs.clone()
        residual_rows = torch.nonzero(residual_mask, as_tuple=False).view(-1)
        residual_cols = residual_token_id[residual_rows]
        final_probs[residual_rows, residual_cols] = torch.clamp(
            final_probs[residual_rows, residual_cols]
            - residual_token_prob[residual_rows],
            min=0.0,
        )
    else:
        final_probs = final_target_probs
    final_probs = torch.where(
        final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs
    )
    final_token_ids = _sample_from_probs_with_coin(
        final_probs, final_uniform_samples_for_sampling
    )
    predicts.scatter_(
        dim=0,
        index=last_accepted_retrive_idx,
        src=final_token_ids.to(dtype=predicts.dtype),
    )


def _tree_speculative_sampling_target_only_vectorized_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> None:
    del draft_probs

    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    device = target_probs_for_sampling.device
    vocab_size = target_probs_for_sampling.shape[-1]

    threshold_acc = max(float(threshold_acc), 1e-9)
    threshold_single = float(threshold_single)

    accept_index.fill_(-1)
    accept_token_num.zero_()

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    cur_prob_idx = torch.zeros((batch_size,), dtype=torch.long, device=device)
    accepted_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)

    last_accepted_retrive_idx = retrive_index[:, 0].to(torch.long)
    accept_index[:, 0].copy_(last_accepted_retrive_idx.to(dtype=accept_index.dtype))
    coin = uniform_samples_for_sampling[:, 0]
    residual_draft_probs = torch.zeros(
        (batch_size, vocab_size),
        dtype=target_probs_for_sampling.dtype,
        device=device,
    )

    for _ in range(1, num_speculative_tokens):
        sibling_idx = torch.where(
            active,
            retrive_next_token[batch_indices, cur_prob_idx],
            torch.full_like(cur_prob_idx, -1),
        )
        found_idx = torch.full_like(cur_prob_idx, -1)
        prob_acc = torch.zeros(
            (batch_size,),
            dtype=target_probs_for_sampling.dtype,
            device=device,
        )

        for _ in range(num_draft_tokens):
            valid = active & (found_idx < 0) & (sibling_idx >= 0)
            safe_sibling_idx = sibling_idx.clamp_min(0)
            draft_token_id = candidates[batch_indices, safe_sibling_idx].to(torch.long)
            target_prob_single = target_probs_for_sampling[
                batch_indices,
                cur_prob_idx,
                draft_token_id,
            ]
            target_prob_single = torch.where(
                valid, target_prob_single, torch.zeros_like(target_prob_single)
            )
            next_prob_acc = prob_acc + target_prob_single

            old_residual = residual_draft_probs.gather(
                dim=1,
                index=draft_token_id.view(-1, 1),
            ).squeeze(1)
            residual_update = torch.where(valid, target_prob_single, old_residual)
            residual_draft_probs.scatter_(
                dim=1,
                index=draft_token_id.view(-1, 1),
                src=residual_update.view(-1, 1),
            )

            accepted = valid & (
                (coin <= (next_prob_acc / threshold_acc))
                | (target_prob_single >= threshold_single)
            )
            found_idx = torch.where(accepted, sibling_idx, found_idx)
            prob_acc = torch.where(valid, next_prob_acc, prob_acc)
            sibling_idx = torch.where(
                valid & ~accepted,
                retrive_next_sibling[batch_indices, safe_sibling_idx],
                sibling_idx,
            )

        accepted = active & (found_idx >= 0)
        safe_found_idx = found_idx.clamp_min(0)
        accepted_token_id = candidates[batch_indices, safe_found_idx].to(
            dtype=predicts.dtype
        )
        accepted_retrive_idx = retrive_index[batch_indices, safe_found_idx].to(
            torch.long
        )

        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(accepted, accepted_token_id, old_predicts)
        predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=predict_updates)

        next_accepted_count = accepted_count + accepted.to(dtype=torch.long)
        accept_index_position = next_accepted_count.clamp_max(
            num_speculative_tokens - 1
        ).view(-1, 1)
        old_accept_index = accept_index.gather(
            dim=1, index=accept_index_position
        ).squeeze(1)
        accept_index_updates = torch.where(
            accepted,
            accepted_retrive_idx.to(dtype=accept_index.dtype),
            old_accept_index,
        )
        accept_index.scatter_(
            dim=1, index=accept_index_position, src=accept_index_updates.view(-1, 1)
        )

        cur_prob_idx = torch.where(accepted, safe_found_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(
            accepted, accepted_retrive_idx, last_accepted_retrive_idx
        )
        accepted_count = next_accepted_count
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_draft_probs.mul_(
            (~accepted).to(dtype=residual_draft_probs.dtype).view(-1, 1)
        )

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    residual_probs = torch.clamp(final_target_probs - residual_draft_probs, min=0.0)
    need_residual = accepted_count != (num_speculative_tokens - 1)
    final_probs = torch.where(
        need_residual.view(-1, 1), residual_probs, final_target_probs
    )
    final_probs = torch.where(
        final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs
    )
    final_token_ids = _sample_from_probs_with_coin(
        final_probs, final_uniform_samples_for_sampling
    )
    predicts.scatter_(
        dim=0,
        index=last_accepted_retrive_idx,
        src=final_token_ids.to(dtype=predicts.dtype),
    )


def _tree_speculative_sampling_target_only_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    # Linear EAGLE trees (for example spec_topk=1) can skip sibling scanning.
    has_sibling = bool(torch.any(retrive_next_sibling >= 0).item())
    if not has_sibling:
        if _try_tree_speculative_sampling_target_only_linear_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
        ):
            return
        _tree_speculative_sampling_target_only_linear_torch(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=deterministic,
        )
        return

    _debug_sglang_npu_eagle_linear_triton(
        "tree_has_sibling",
        retrive_next_sibling=_tensor_debug_summary(retrive_next_sibling),
    )
    _tree_speculative_sampling_target_only_vectorized_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )


def _patch_sglang_npu_eagle_triton_bool_compat() -> None:
    try:
        spec_utils = importlib.import_module("sglang.srt.speculative.spec_utils")
        kernel = getattr(spec_utils, "assign_draft_cache_locs", None)
        if kernel is None:
            logger.warning(
                "Skip SGLang NPU EAGLE Triton bool compatibility patch: assign_draft_cache_locs missing."
            )
            return
        if getattr(kernel, "_verl_patched_triton_bool_compat", False):
            return
        kernel_fn = getattr(kernel, "fn", kernel)
        source = textwrap.dedent(inspect.getsource(kernel_fn))
        patched_source, replacement_count = re.subn(
            r"(?m)^([ \t]*)if\s+page_size\s*!=\s*1\s+and\s+topk\s*!=\s*1\s+and\s+duplicate_cache_len\s*>\s*0\s*:",
            r"\1if (page_size != 1 and topk != 1) and duplicate_cache_len > 0:",
            source,
            count=1,
        )
        if replacement_count <= 0:
            logger.warning(
                "Skip SGLang NPU EAGLE Triton bool compatibility patch: "
                "chained condition not found in %s.",
                getattr(kernel_fn, "__code__", None).co_filename
                if getattr(kernel_fn, "__code__", None)
                else kernel_fn,
            )
            return
        namespace = {}
        exec(  # noqa: S102
            "from __future__ import annotations\n" + patched_source,
            kernel_fn.__globals__,
            namespace,
        )
        patched_kernel = namespace[kernel_fn.__name__]
        patched_kernel._verl_patched_triton_bool_compat = True
        spec_utils.assign_draft_cache_locs = patched_kernel
        for module_name in (
            "sglang.srt.speculative.eagle_worker",
            "sglang.srt.speculative.eagle_worker_v2",
            "sglang.srt.speculative.multi_layer_eagle_worker",
            "sglang.srt.speculative.multi_layer_eagle_worker_v2",
            "sglang.srt.speculative.frozen_kv_mtp_worker",
        ):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "assign_draft_cache_locs"):
                module.assign_draft_cache_locs = patched_kernel
        logger.warning("SGLang NPU EAGLE Triton bool compatibility patch active.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skip SGLang NPU EAGLE Triton bool compatibility patch: %s", exc)


def patch_sglang_npu_eagle_target_sampling() -> None:
    """Patch SGLang NPU EAGLE v1 verification to use target-only sampling."""
    global _SGLANG_NPU_EAGLE_SAMPLING_PATCHED

    _patch_sglang_npu_eagle_triton_bool_compat()
    if _SGLANG_NPU_EAGLE_SAMPLING_PATCHED or not _is_sglang_npu_backend():
        return

    patched_targets = []

    v1_verify_mode = _sglang_npu_eagle_v1_verify_mode()
    if v1_verify_mode != "greedy":
        try:
            eagle_info = importlib.import_module("sglang.srt.speculative.eagle_info")
            eagle_info.top_k_renorm_prob = _top_k_renorm_prob_torch
            eagle_info.top_p_renorm_prob = _top_p_renorm_prob_torch
            eagle_info.tree_speculative_sampling_target_only = (
                _tree_speculative_sampling_target_only_torch
            )
            eagle_info.TREE_SPEC_KERNEL_AVAILABLE = True
            patched_targets.append("sglang.srt.speculative.eagle_info(target_only)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang EAGLE v1 target sampling patch: %s", exc)
    else:
        logger.info(
            "Skip SGLang EAGLE v1 target sampling patch. Set %s=target_only to enable it.",
            _EAGLE_V1_VERIFY_MODE_ENV,
        )

    if patched_targets:
        _SGLANG_NPU_EAGLE_SAMPLING_PATCHED = True
        logger.warning(
            "Patched SGLang NPU EAGLE sampling for %s", ", ".join(patched_targets)
        )


def _is_torch_tensor(value: Any) -> bool:
    is_tensor = getattr(torch, "is_tensor", None)
    return bool(callable(is_tensor) and is_tensor(value))


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _custom_flag_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "y"}
    return bool(value)


def _sglang_req_custom_params(req) -> dict[str, Any]:
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def _sglang_req_requests_last_hidden_for_drafter(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(
            _VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM, False
        )
    )


def _sglang_req_requests_raw_top_logprobs(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(_VERL_DRAFTER_RAW_TOP_LOGPROBS_PARAM, False)
    )


def _sglang_forward_batch_requests_raw_top_logprobs(forward_batch) -> bool:
    requested = getattr(
        forward_batch, _VERL_DRAFTER_RAW_TOP_LOGPROBS_REQUESTED_ATTR, None
    )
    if requested is not None:
        return bool(requested)
    for req in getattr(forward_batch, "reqs", []) or []:
        if getattr(
            req, "return_hidden_states", False
        ) and _sglang_req_requests_raw_top_logprobs(req):
            return True
    return False


def _mark_sglang_forward_batch_raw_top_logprobs_requested(
    forward_batch, source_batch
) -> None:
    if forward_batch is None:
        return
    setattr(
        forward_batch,
        _VERL_DRAFTER_RAW_TOP_LOGPROBS_REQUESTED_ATTR,
        _sglang_forward_batch_requests_raw_top_logprobs(source_batch),
    )


def _sglang_forward_batch_requests_last_hidden_for_drafter(forward_batch) -> bool:
    if _sglang_drafter_return_last_hidden_enabled():
        return True
    for req in getattr(forward_batch, "reqs", []) or []:
        if getattr(
            req, "return_hidden_states", False
        ) and _sglang_req_requests_last_hidden_for_drafter(req):
            return True
    return False


def _sglang_should_filter_eagle_verify_last_hidden(batch) -> bool:
    """Build accepted-token metadata without replacing SGLang's native hidden tensor."""
    return _sglang_forward_batch_requests_last_hidden_for_drafter(
        batch
    ) or _sglang_raw_top_logprobs_enabled(batch)


def _sglang_req_requests_dflash_aux_hidden(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(_VERL_DFLASH_RETURN_AUX_HIDDEN_PARAM, False)
    )


def _sglang_forward_batch_requests_dflash_aux_hidden(forward_batch) -> bool:
    for req in getattr(forward_batch, "reqs", []) or []:
        if getattr(
            req, "return_hidden_states", False
        ) and _sglang_req_requests_dflash_aux_hidden(req):
            return True
    return False


def _sglang_dflash_should_return_verify_hidden(batch) -> bool:
    return _sglang_forward_batch_requests_dflash_aux_hidden(batch)


def _sglang_dflash_restore_verify_hidden(
    batch, logits_output, next_target_hidden
) -> None:
    if not _sglang_dflash_should_return_verify_hidden(batch):
        return
    dflash_hidden_states = _normalize_sglang_dflash_aux_hidden_states(
        next_target_hidden
    )
    if dflash_hidden_states is None:
        logger.warning(
            "SGLang DFlash verify hidden states requested but unavailable: "
            "next_target_hidden_type=%s output_hidden_type=%s",
            type(next_target_hidden).__name__,
            type(getattr(logits_output, "hidden_states", None)).__name__,
        )
        return
    logits_output.hidden_states = dflash_hidden_states
    setattr(logits_output, "_verl_dflash_aux_hidden_states", True)


def _normalize_sglang_dflash_aux_hidden_states(aux_hidden_states):
    if aux_hidden_states is None:
        return None

    if _is_torch_tensor(aux_hidden_states):
        if aux_hidden_states.dim() <= 2:
            return aux_hidden_states
        if aux_hidden_states.dim() == 3:
            first_dim = int(aux_hidden_states.shape[0])
            second_dim = int(aux_hidden_states.shape[1])
            # SGLang aux hidden is commonly [num_layers, rows, hidden].
            # DFlash training expects [rows, num_layers * hidden].
            if first_dim <= 64 and second_dim > first_dim:
                return aux_hidden_states.permute(1, 0, 2).reshape(second_dim, -1)
            return aux_hidden_states.reshape(first_dim, -1)
        return aux_hidden_states.reshape(aux_hidden_states.shape[0], -1)

    if isinstance(aux_hidden_states, (list, tuple)):
        tensors = [tensor for tensor in aux_hidden_states if _is_torch_tensor(tensor)]
        if not tensors:
            return None
        if len(tensors) == 1:
            return _normalize_sglang_dflash_aux_hidden_states(tensors[0])
        base_shape = tuple(tensors[0].shape[:-1])
        if all(tuple(tensor.shape[:-1]) == base_shape for tensor in tensors):
            return torch.cat(tensors, dim=-1)
        try:
            stacked = torch.stack(tensors, dim=0)
        except Exception:  # noqa: BLE001
            return None
        return _normalize_sglang_dflash_aux_hidden_states(stacked)

    return None


def _sglang_concat_last_hidden_for_drafter(
    req, logits_output, hidden_chunk, last_hidden_chunk
):
    if not _sglang_req_requests_last_hidden_for_drafter(req):
        return hidden_chunk
    if bool(getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR, False)):
        return hidden_chunk
    if last_hidden_chunk is None:
        raise RuntimeError(
            "SGLang did not return final target hidden states for drafter training."
        )
    if not (_is_torch_tensor(hidden_chunk) and _is_torch_tensor(last_hidden_chunk)):
        raise RuntimeError(
            "SGLang drafter last-hidden output requires tensor hidden chunks: "
            f"hidden_type={type(hidden_chunk)}, last_hidden_type={type(last_hidden_chunk)}."
        )
    if tuple(hidden_chunk.shape[:-1]) != tuple(last_hidden_chunk.shape[:-1]):
        raise RuntimeError(
            "SGLang drafter last-hidden shape does not align with EAGLE hidden states: "
            f"hidden_shape={tuple(hidden_chunk.shape)}, last_hidden_shape={tuple(last_hidden_chunk.shape)}."
        )
    if last_hidden_chunk.device != hidden_chunk.device:
        last_hidden_chunk = last_hidden_chunk.to(hidden_chunk.device)
    return torch.cat((hidden_chunk, last_hidden_chunk), dim=-1)


def _slice_sglang_drafter_last_hidden_output(logits_output, index):
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if last_hidden_states is None:
        return None
    return last_hidden_states[index]


def _materialize_sglang_drafter_last_hidden_output(logits_output, stage: str) -> bool:
    if logits_output is None:
        return False
    if bool(getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR, False)):
        return True
    base_hidden_states = getattr(logits_output, "hidden_states", None)
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if not _is_torch_tensor(last_hidden_states):
        return False
    if base_hidden_states is None:
        logits_output.hidden_states = last_hidden_states
        materialized_shape = tuple(last_hidden_states.shape)
    elif _is_torch_tensor(base_hidden_states):
        if tuple(base_hidden_states.shape[:-1]) != tuple(last_hidden_states.shape[:-1]):
            logger.warning(
                "Skip materializing SGLang drafter final hidden: stage=%s "
                "base_shape=%s last_hidden_shape=%s",
                stage,
                tuple(base_hidden_states.shape),
                tuple(last_hidden_states.shape),
            )
            return False
        if last_hidden_states.device != base_hidden_states.device:
            last_hidden_states = last_hidden_states.to(base_hidden_states.device)
        logits_output.hidden_states = torch.cat(
            (base_hidden_states, last_hidden_states), dim=-1
        )
        materialized_shape = tuple(logits_output.hidden_states.shape)
    else:
        logger.warning(
            "Skip materializing SGLang drafter final hidden: stage=%s base_type=%s last_hidden_shape=%s",
            stage,
            type(base_hidden_states).__name__,
            tuple(last_hidden_states.shape),
        )
        return False

    setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR, True)
    select_summary = getattr(
        logits_output, "_verl_drafter_last_hidden_select_summary", None
    )
    if isinstance(select_summary, dict):
        setattr(
            logits_output,
            "_verl_drafter_last_hidden_select_summary",
            {
                **select_summary,
                "materialized_stage": stage,
                "materialized_shape": materialized_shape,
            },
        )
    return True


def _sglang_forward_mode_is_target_verify(logits_metadata) -> bool:
    forward_mode = getattr(logits_metadata, "forward_mode", None)
    is_target_verify = getattr(forward_mode, "is_target_verify", None)
    if callable(is_target_verify):
        try:
            return bool(is_target_verify())
        except Exception:  # noqa: BLE001
            return False
    return False


def _sglang_forward_mode_is_draft_extend(logits_metadata) -> bool:
    forward_mode = getattr(logits_metadata, "forward_mode", None)
    is_draft_extend = getattr(forward_mode, "is_draft_extend", None)
    if callable(is_draft_extend):
        try:
            return bool(is_draft_extend(include_v2=True))
        except TypeError:
            try:
                return bool(is_draft_extend())
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001
            return False
    return False


def _attach_sglang_lm_head_fingerprint(logits_output, lm_head) -> None:
    if not _sglang_last_hidden_logprob_check_enabled():
        return
    lm_head_weight = getattr(lm_head, "weight", None)
    if not _is_torch_tensor(lm_head_weight):
        return
    try:
        weight = lm_head_weight.detach()
        block = weight[
            : min(8, int(weight.shape[0])), : min(8, int(weight.shape[1]))
        ].float()
        row0_norm = weight[0].float().norm() if int(weight.shape[0]) > 0 else None
        row_last_norm = weight[-1].float().norm() if int(weight.shape[0]) > 0 else None
        setattr(
            logits_output,
            "_verl_drafter_lh_check_lm_head_fingerprint",
            {
                "shape": tuple(weight.shape),
                "dtype": str(weight.dtype),
                "device": str(weight.device),
                "block_sum": float(block.sum().detach().cpu().item()),
                "row0_norm": None
                if row0_norm is None
                else float(row0_norm.detach().cpu().item()),
                "row_last_index": int(weight.shape[0]) - 1,
                "row_last_norm": None
                if row_last_norm is None
                else float(row_last_norm.detach().cpu().item()),
            },
        )
    except Exception as exc:  # noqa: BLE001
        setattr(
            logits_output,
            "_verl_drafter_lh_check_lm_head_fingerprint",
            {"error": str(exc)},
        )


def _sglang_top_logprobs_num_from_metadata(logits_metadata) -> int:
    top_logprobs_nums = getattr(logits_metadata, "top_logprobs_nums", None)
    if top_logprobs_nums is None:
        return 0
    if isinstance(top_logprobs_nums, int):
        return max(int(top_logprobs_nums), 0)
    try:
        return max((int(x or 0) for x in top_logprobs_nums), default=0)
    except Exception:  # noqa: BLE001
        return 0


def _sglang_raw_top_logprobs_rows_for_window(
    logits_metadata,
    positions,
    rows: int,
    *,
    rows_per_req: list[int] | None = None,
):
    if rows <= 0 or not _is_torch_tensor(positions) or positions.dim() <= 0:
        return None
    positions = positions.reshape(-1)
    if int(positions.numel()) != rows:
        return None
    positions = positions.to(dtype=torch.long)

    reqs = list(getattr(logits_metadata, "reqs", []) or [])
    if not reqs:
        return None

    mask = torch.zeros((rows,), dtype=torch.bool, device=positions.device)
    if rows_per_req is not None and len(rows_per_req) > 0:
        offset = 0
        for req, req_rows in zip(reqs, rows_per_req):
            try:
                req_rows = max(int(req_rows), 0)
            except (TypeError, ValueError):
                req_rows = 0
            end = min(offset + req_rows, rows)
            if end <= offset:
                offset = end
                continue
            if getattr(
                req, "return_hidden_states", False
            ) and _sglang_req_requests_raw_top_logprobs(req):
                window_config = _sglang_hidden_window_config(req)
                if window_config is None:
                    mask[offset:end] = True
                else:
                    req_positions = positions[offset:end]
                    mask[offset:end] = (
                        req_positions >= int(window_config["window_start"])
                    ) & (req_positions < int(window_config["window_end"]))
            offset = end
            if offset >= rows:
                break
        return torch.nonzero(mask, as_tuple=False).flatten()

    for req in reqs:
        if not (
            getattr(req, "return_hidden_states", False)
            and _sglang_req_requests_raw_top_logprobs(req)
        ):
            continue
        window_config = _sglang_hidden_window_config(req)
        if window_config is None:
            return None
        mask |= (positions >= int(window_config["window_start"])) & (
            positions < int(window_config["window_end"])
        )
    return torch.nonzero(mask, as_tuple=False).flatten()


def _normalize_sglang_raw_topk_positions(positions, rows: int, *, device=None):
    if positions is None or rows <= 0:
        return None
    try:
        if _is_torch_tensor(positions):
            normalized = (
                positions.detach().to(device=device, dtype=torch.long).reshape(-1)
            )
        else:
            normalized = torch.tensor(
                list(positions), device=device, dtype=torch.long
            ).reshape(-1)
        if int(normalized.numel()) != int(rows):
            return None
        return normalized.contiguous()
    except Exception:  # noqa: BLE001
        return None


def _set_sglang_raw_topk_positions(logits_output, positions, rows: int, *, device=None):
    normalized = _normalize_sglang_raw_topk_positions(positions, rows, device=device)
    if _is_torch_tensor(normalized):
        # SGLang's positions accompanying next_token_logits already use the
        # supervised-token coordinate. Do not shift them again. Only the
        # hidden-chunk fallback in _sglang_hidden_debug_metadata converts
        # source hidden positions p to supervised-token positions p + 1.
        setattr(
            logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR, normalized
        )
    else:
        try:
            if hasattr(logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR):
                delattr(logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR)
        except Exception:  # noqa: BLE001
            setattr(logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR, None)
    return normalized


def _attach_sglang_raw_top_logprobs(
    logits_output,
    logits_metadata,
    *,
    topk: int | None = None,
    positions=None,
    source_row_indices=None,
    rows_per_req: list[int] | None = None,
) -> bool:
    next_token_logits = getattr(logits_output, "next_token_logits", None)
    if not _is_torch_tensor(next_token_logits) or next_token_logits.dim() < 2:
        return False

    if topk is None:
        topk = _sglang_raw_top_logprobs_topk()
    if topk is None or int(topk) <= 0:
        topk = _sglang_top_logprobs_num_from_metadata(logits_metadata)
    if topk is None or int(topk) <= 0:
        topk = _sglang_last_hidden_logprob_check_topk()

    try:
        with torch.no_grad():
            source_rows = int(next_token_logits.shape[0])
            raw_positions = None
            # Raw top-k rows are training targets when use_logits=True. Keep
            # their source positions as normal metadata so rollout can align
            # teacher rows without relying on compact row order.
            track_raw_positions = True
            if track_raw_positions:
                if positions is None:
                    positions = getattr(logits_metadata, "positions", None)
                raw_positions = _normalize_sglang_raw_topk_positions(
                    positions,
                    source_rows,
                    device=next_token_logits.device,
                )
            if _is_torch_tensor(source_row_indices):
                source_row_indices = source_row_indices.to(
                    device=next_token_logits.device,
                    dtype=torch.long,
                ).reshape(-1)
                if int(source_row_indices.numel()) <= 0:
                    return False
                next_token_logits = next_token_logits[source_row_indices]
                if _is_torch_tensor(raw_positions):
                    raw_positions = raw_positions[source_row_indices]
                elif track_raw_positions:
                    raw_positions = _normalize_sglang_raw_topk_positions(
                        positions,
                        int(source_row_indices.numel()),
                        device=next_token_logits.device,
                    )
            rows = int(next_token_logits.shape[0])
            vocab = int(next_token_logits.shape[-1])
            raw_topk = min(max(int(topk), 1), vocab)
            if rows <= 0 or raw_topk <= 0:
                return False

            row_indices = _sglang_raw_top_logprobs_rows_for_window(
                logits_metadata,
                raw_positions,
                rows,
                rows_per_req=rows_per_req,
            )
            if _is_torch_tensor(row_indices) and int(row_indices.numel()) <= 0:
                return False

            compute_logits = next_token_logits.detach().float()
            computed_subset = _is_torch_tensor(row_indices)
            if computed_subset:
                row_indices = row_indices.to(
                    device=next_token_logits.device, dtype=torch.long
                )
                compute_logits = compute_logits[row_indices]

            sglang_logprobs = torch.nn.functional.log_softmax(compute_logits, dim=-1)
            computed_logprobs, computed_ids = torch.topk(
                sglang_logprobs, k=raw_topk, dim=-1
            )
            if computed_subset:
                raw_topk_logprobs = torch.full(
                    (rows, raw_topk),
                    float("-inf"),
                    dtype=computed_logprobs.dtype,
                    device=computed_logprobs.device,
                )
                raw_topk_ids = torch.full(
                    (rows, raw_topk),
                    -1,
                    dtype=computed_ids.dtype,
                    device=computed_ids.device,
                )
                raw_topk_logprobs[row_indices] = computed_logprobs
                raw_topk_ids[row_indices] = computed_ids
            else:
                raw_topk_logprobs = computed_logprobs
                raw_topk_ids = computed_ids
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
                raw_topk_ids.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
                raw_topk_logprobs.detach(),
            )
            if track_raw_positions:
                _set_sglang_raw_topk_positions(
                    logits_output,
                    raw_positions,
                    rows,
                    device=raw_topk_logprobs.device,
                )
            logits_shapes = getattr(
                logits_output, "_verl_drafter_lh_check_logits_shapes", None
            )
            if not isinstance(logits_shapes, dict):
                logits_shapes = {}
            logits_shapes.update(
                {
                    "raw_logits_shape": tuple(next_token_logits.shape),
                    "raw_topk": raw_topk,
                    "raw_topk_source": "next_token_logits",
                    "raw_topk_computed_rows": int(compute_logits.shape[0]),
                    "raw_topk_windowed": bool(computed_subset),
                }
            )
            setattr(
                logits_output, "_verl_drafter_lh_check_logits_shapes", logits_shapes
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to attach SGLang raw top-logprobs: %s", exc)
        return False


def _attach_sglang_last_hidden_logprob_check(
    logits_processor, logits_output, hidden_states, lm_head, logits_metadata
) -> None:
    if not _sglang_last_hidden_logprob_check_enabled():
        return
    if not _sglang_forward_mode_is_target_verify(logits_metadata):
        _log_sglang_last_hidden_logprob_check_skip(
            "not_target_verify",
            forward_mode=repr(getattr(logits_metadata, "forward_mode", None)),
        )
        return
    next_token_logits = getattr(logits_output, "next_token_logits", None)
    if not (_is_torch_tensor(next_token_logits) and _is_torch_tensor(hidden_states)):
        _log_sglang_last_hidden_logprob_check_skip(
            "missing_logits_or_hidden",
            next_token_logits_type=type(next_token_logits).__name__,
            hidden_states_type=type(hidden_states).__name__,
        )
        return

    try:
        with torch.no_grad():
            sglang_logits_snapshot = next_token_logits.detach().clone()
            recomputed_logits = logits_processor._get_logits(
                hidden_states, lm_head, logits_metadata
            )
            rows = min(
                int(recomputed_logits.shape[0]), int(sglang_logits_snapshot.shape[0])
            )
            vocab = min(
                int(recomputed_logits.shape[-1]), int(sglang_logits_snapshot.shape[-1])
            )
            if rows <= 0 or vocab <= 0:
                return
            setattr(
                logits_output,
                "_verl_drafter_lh_check_logits_shapes",
                {
                    "recomputed_logits_shape": tuple(recomputed_logits.shape),
                    "sglang_logits_shape": tuple(sglang_logits_snapshot.shape),
                    "compared_rows": rows,
                    "compared_vocab": vocab,
                },
            )
            recomputed_logits = recomputed_logits[:rows, :vocab].float()
            sglang_logits = sglang_logits_snapshot[:rows, :vocab].float()
            recomputed_logprobs = torch.nn.functional.log_softmax(
                recomputed_logits, dim=-1
            )
            sglang_logprobs = torch.nn.functional.log_softmax(sglang_logits, dim=-1)
            raw_topk = min(_sglang_last_hidden_logprob_check_topk(), vocab)
            raw_topk_logprobs, raw_topk_ids = torch.topk(
                sglang_logprobs, k=raw_topk, dim=-1
            )
            sglang_top_logprobs = raw_topk_logprobs[:, 0]
            sglang_top_ids = raw_topk_ids[:, 0]
            recomputed_top_logprobs, recomputed_top_ids = recomputed_logprobs.max(
                dim=-1
            )
            row_ids = torch.arange(rows, device=recomputed_logprobs.device)
            recomputed_at_sglang_top = recomputed_logprobs[row_ids, sglang_top_ids]
            _attach_sglang_lm_head_fingerprint(logits_output, lm_head)
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR,
                recomputed_top_ids.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR,
                recomputed_top_logprobs.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR,
                recomputed_at_sglang_top.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR,
                sglang_top_ids.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR,
                sglang_top_logprobs.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
                raw_topk_ids.detach(),
            )
            setattr(
                logits_output,
                _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
                raw_topk_logprobs.detach(),
            )
            _set_sglang_raw_topk_positions(
                logits_output,
                getattr(logits_metadata, "positions", None),
                rows,
                device=raw_topk_logprobs.device,
            )
            _log_sglang_last_hidden_logprob_check(logits_output, stage="attach")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to attach SGLang last-hidden logprob check tensors: %s", exc
        )


def _attach_sglang_last_hidden_logprob_check_from_graph_runner(
    graph_runner, logits_output, logits_metadata
) -> None:
    if not _sglang_last_hidden_logprob_check_enabled():
        return
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if not _is_torch_tensor(last_hidden_states):
        return
    model_runner = getattr(graph_runner, "model_runner", None)
    model = getattr(model_runner, "model", None)
    logits_processor = getattr(model, "logits_processor", None)
    lm_head = getattr(model, "lm_head", None)
    if logits_processor is None or lm_head is None:
        logger.warning(
            "Skip SGLang graph last-hidden logprob check: missing logits_processor/lm_head "
            "model_type=%s",
            type(model).__name__,
        )
        return
    _attach_sglang_last_hidden_logprob_check(
        logits_processor,
        logits_output,
        last_hidden_states,
        lm_head,
        logits_metadata,
    )


def _sglang_logits_metadata_view(logits_metadata):
    if hasattr(logits_metadata, "extend_return_logprob"):
        return logits_metadata
    try:
        from sglang.srt.layers.logits_processor import LogitsMetadata

        return LogitsMetadata.from_forward_batch(logits_metadata)
    except Exception:  # noqa: BLE001
        return logits_metadata


def _select_sglang_last_hidden_for_drafter(
    logits_processor,
    logits_output,
    input_ids,
    hidden_states,
    hidden_states_before_norm,
    aux_hidden_states,
    logits_metadata,
):
    next_token_logits = getattr(logits_output, "next_token_logits", None)
    raw_shape = tuple(hidden_states.shape) if _is_torch_tensor(hidden_states) else None
    output_hidden_states = getattr(logits_output, "hidden_states", None)
    output_hidden_shape = (
        tuple(output_hidden_states.shape)
        if _is_torch_tensor(output_hidden_states)
        else None
    )
    selected = hidden_states
    pruned_shape = None
    sample_indices_shape = None
    store_shape = None
    source = "raw_hidden_states"
    logits_metadata_view = _sglang_logits_metadata_view(logits_metadata)
    try:
        get_pruned_states = getattr(logits_processor, "_get_pruned_states", None)
        get_hidden_states_to_store = getattr(
            logits_processor, "_get_hidden_states_to_store", None
        )
        if (
            callable(get_pruned_states)
            and callable(get_hidden_states_to_store)
            and _is_torch_tensor(hidden_states)
        ):
            try:
                pruned_result = get_pruned_states(
                    hidden_states,
                    None,
                    None,
                    logits_metadata_view,
                )
                source = "sglang_hidden_states_to_store_final"
            except TypeError:
                try:
                    pruned_result = get_pruned_states(
                        input_ids, hidden_states, None, logits_metadata_view
                    )
                    source = "sglang_hidden_states_to_store_final_input_ids"
                except TypeError:
                    pruned_result = get_pruned_states(
                        hidden_states, logits_metadata_view
                    )
                    source = "sglang_hidden_states_to_store_final_legacy"

            if isinstance(pruned_result, tuple):
                pruned_states = pruned_result[0] if len(pruned_result) > 0 else None
                sample_indices = pruned_result[3] if len(pruned_result) > 3 else None
            else:
                pruned_states = pruned_result
                sample_indices = None
            if _is_torch_tensor(pruned_states):
                pruned_shape = tuple(pruned_states.shape)
                if _is_torch_tensor(sample_indices):
                    sample_indices_shape = tuple(sample_indices.shape)
                stored = get_hidden_states_to_store(
                    hidden_states,
                    None,
                    None,
                    pruned_states,
                    None,
                    None,
                    sample_indices,
                    logits_metadata_view,
                )
                if _is_torch_tensor(stored):
                    selected = stored
                    store_shape = tuple(stored.shape)
                else:
                    source = f"{source}_no_store"
    except Exception as exc:  # noqa: BLE001
        setattr(logits_output, "_verl_drafter_last_hidden_select_error", str(exc))

    selected_shape = tuple(selected.shape) if _is_torch_tensor(selected) else None
    logits_shape = (
        tuple(next_token_logits.shape) if _is_torch_tensor(next_token_logits) else None
    )
    setattr(
        logits_output,
        "_verl_drafter_last_hidden_select_summary",
        {
            "source": source,
            "raw_shape": raw_shape,
            "pruned_shape": pruned_shape,
            "store_shape": store_shape,
            "selected_shape": selected_shape,
            "output_hidden_shape": output_hidden_shape,
            "sample_indices_shape": sample_indices_shape,
            "next_token_logits_shape": logits_shape,
            "logits_metadata_type": type(logits_metadata).__name__,
            "logits_metadata_view_type": type(logits_metadata_view).__name__,
            "error": getattr(
                logits_output, "_verl_drafter_last_hidden_select_error", None
            ),
        },
    )
    return selected


def _filter_sglang_last_hidden_logprob_check_tensors(logits_output, index) -> None:
    for attr in (
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR,
    ):
        value = getattr(logits_output, attr, None)
        if not _is_torch_tensor(value):
            continue
        try:
            index_len = int(index.numel()) if _is_torch_tensor(index) else len(index)
        except Exception:  # noqa: BLE001
            index_len = None
        if (
            index_len is not None
            and value.dim() > 0
            and int(value.shape[0]) < index_len
        ):
            continue
        try:
            setattr(logits_output, attr, value[index])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to filter SGLang last-hidden logprob check tensor %s: %s",
                attr,
                exc,
            )


def _build_sglang_last_hidden_logprob_check_summary(
    logits_output, stage: str
) -> dict | None:
    recomputed_top_ids = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR, None
    )
    recomputed_top_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR, None
    )
    recomputed_at_sglang_top = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR, None
    )
    sglang_top_ids = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR, None
    )
    sglang_top_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR, None
    )
    raw_topk_ids = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR, None
    )
    raw_topk_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None
    )
    lm_head_fingerprint = getattr(
        logits_output, "_verl_drafter_lh_check_lm_head_fingerprint", None
    )
    logits_shapes = getattr(logits_output, "_verl_drafter_lh_check_logits_shapes", None)
    if not all(
        _is_torch_tensor(t)
        for t in (
            recomputed_top_ids,
            recomputed_top_logprobs,
            recomputed_at_sglang_top,
            sglang_top_ids,
            sglang_top_logprobs,
        )
    ):
        return None

    try:
        with torch.no_grad():
            rows = min(
                int(recomputed_top_ids.numel()),
                int(recomputed_top_logprobs.numel()),
                int(recomputed_at_sglang_top.numel()),
                int(sglang_top_ids.numel()),
                int(sglang_top_logprobs.numel()),
            )
            if rows <= 0:
                return None
            recomputed_top_ids = recomputed_top_ids[:rows].reshape(-1)
            recomputed_top_logprobs = recomputed_top_logprobs[:rows].reshape(-1).float()
            recomputed_at_sglang_top = (
                recomputed_at_sglang_top[:rows].reshape(-1).float()
            )
            sglang_top_ids = sglang_top_ids[:rows].reshape(-1)
            sglang_top_logprobs = sglang_top_logprobs[:rows].reshape(-1).float()
            finite = torch.isfinite(recomputed_at_sglang_top) & torch.isfinite(
                sglang_top_logprobs
            )
            if not finite.any():
                return None
            match = (recomputed_top_ids == sglang_top_ids) & finite
            diff = (
                recomputed_at_sglang_top[finite] - sglang_top_logprobs[finite]
            ).abs()
            p95 = torch.quantile(diff, 0.95) if diff.numel() > 1 else diff.max()
            summary = {
                "stage": stage,
                "target_logprobs_source": _VERL_TARGET_LOGPROBS_SOURCE_RAW_HIDDEN_METADATA,
                "rows": rows,
                "valid_rows": int(finite.detach().sum().cpu().item()),
                "top1_match": round(
                    float(match[finite].float().mean().detach().cpu().item()), 6
                ),
                "logprob_abs_diff_mean": round(
                    float(diff.mean().detach().cpu().item()), 6
                ),
                "logprob_abs_diff_p95": round(float(p95.detach().cpu().item()), 6),
                "logprob_abs_diff_max": round(
                    float(diff.max().detach().cpu().item()), 6
                ),
                "recomputed_top1_logprob_mean": round(
                    float(recomputed_top_logprobs[finite].mean().detach().cpu().item()),
                    6,
                ),
                "sglang_top1_logprob_mean": round(
                    float(sglang_top_logprobs[finite].mean().detach().cpu().item()), 6
                ),
                "lm_head": lm_head_fingerprint,
            }
            if _is_torch_tensor(raw_topk_ids) and _is_torch_tensor(raw_topk_logprobs):
                raw_rows = min(
                    int(raw_topk_ids.shape[0]), int(raw_topk_logprobs.shape[0]), rows
                )
                raw_topk = min(
                    int(raw_topk_ids.shape[-1]), int(raw_topk_logprobs.shape[-1])
                )
                if raw_rows > 0 and raw_topk > 0:
                    summary.update(
                        {
                            "raw_topk_rows": raw_rows,
                            "raw_topk": raw_topk,
                            "raw_topk_shape": tuple(raw_topk_logprobs.shape),
                            "raw_top1_ids_head": [
                                int(x)
                                for x in raw_topk_ids[: min(raw_rows, 8), 0]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                        }
                    )
            if isinstance(logits_shapes, dict):
                summary.update(logits_shapes)
            setattr(logits_output, _VERL_DRAFTER_LH_CHECK_SUMMARY_ATTR, summary)
            return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to build SGLang last-hidden logprob check summary: %s", exc
        )
        return None


def _log_sglang_last_hidden_logprob_check(
    logits_output, stage: str = "filtered"
) -> None:
    global _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_LOG_COUNT
    if not _sglang_last_hidden_logprob_check_enabled():
        return
    summary = _build_sglang_last_hidden_logprob_check_summary(logits_output, stage)
    if summary is None:
        return
    if (
        _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_LOG_COUNT
        >= _sglang_last_hidden_logprob_check_max_logs()
    ):
        return

    try:
        logger.warning(
            "[sglang last_hidden logprob check] stage=%s rows=%s valid_rows=%s top1_match=%.6f "
            "logprob_abs_diff_mean=%.6g logprob_abs_diff_p95=%.6g "
            "logprob_abs_diff_max=%.6g recomputed_top1_logprob_mean=%.6g "
            "sglang_top1_logprob_mean=%.6g lm_head=%s",
            summary["stage"],
            summary["rows"],
            summary["valid_rows"],
            summary["top1_match"],
            summary["logprob_abs_diff_mean"],
            summary["logprob_abs_diff_p95"],
            summary["logprob_abs_diff_max"],
            summary["recomputed_top1_logprob_mean"],
            summary["sglang_top1_logprob_mean"],
            summary.get("lm_head"),
        )
        _SGLANG_LAST_HIDDEN_LOGPROB_CHECK_LOG_COUNT += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log SGLang last-hidden logprob check: %s", exc)


def _sglang_tensor_head_tail(
    value, limit: int = 8
) -> tuple[list[int] | None, list[int] | None]:
    if value is None:
        return None, None
    try:
        tensor = (
            value.detach().flatten().to("cpu")
            if _is_torch_tensor(value)
            else torch.tensor(list(value)).flatten()
        )
        if int(tensor.numel()) <= 0:
            return [], []
        head = [int(x) for x in tensor[:limit].tolist()]
        tail = (
            [int(x) for x in tensor[-limit:].tolist()]
            if int(tensor.numel()) > limit
            else head
        )
        return head, tail
    except Exception as exc:  # noqa: BLE001
        error = [f"error:{exc}"]
        return error, error


def _sglang_first_contiguous_position_len(
    positions,
) -> tuple[int | None, bool | None, int | None]:
    if not _is_torch_tensor(positions):
        return None, None, None
    flat = positions.detach().flatten()
    rows = int(flat.numel())
    if rows <= 1:
        return rows, True, None
    contiguous = flat[1:] == flat[:-1] + 1
    breaks = torch.nonzero(~contiguous, as_tuple=False).flatten()
    if int(breaks.numel()) <= 0:
        return rows, True, None
    first_break = int(breaks[0].detach().cpu().item()) + 1
    return first_break, False, first_break


def _sglang_nested_attr(obj, names: tuple[str, ...]):
    for name in names:
        if obj is None:
            return None
        try:
            if isinstance(obj, dict):
                obj = obj.get(name)
            else:
                obj = getattr(obj, name, None)
        except Exception:  # noqa: BLE001
            return None
    return obj


def _sglang_candidate_attrs(obj) -> list[str]:
    if obj is None:
        return []
    try:
        if isinstance(obj, dict):
            names = [str(key) for key in obj.keys()]
        else:
            names = [name for name in dir(obj) if not name.startswith("_")]
        return sorted(
            name
            for name in names
            if any(part in name for part in ("accept", "position", "spec", "verify"))
        )[:32]
    except Exception:  # noqa: BLE001
        return []


def _iter_sglang_verify_objects(*candidates):
    seen = set()
    stack = list(candidates)
    nested_names = (
        "verify_output",
        "output",
        "result",
        "spec_info",
        "speculative_info",
        "eagle_verify_output",
    )
    while stack:
        obj = stack.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj
        if isinstance(obj, (list, tuple)):
            stack[:0] = list(obj[:4])
            continue
        for name in nested_names:
            nested = _sglang_nested_attr(obj, (name,))
            if nested is not None:
                stack.append(nested)


def _find_sglang_accepted_indices(*candidates):
    for obj in _iter_sglang_verify_objects(*candidates):
        for name in (
            "accept_indices",
            "accepted_indices",
            "accept_index",
            "accepted_index",
        ):
            value = _sglang_nested_attr(obj, (name,))
            if value is not None:
                return value
        reconstructed = _reconstruct_sglang_accepted_indices_from_lens(obj)
        if reconstructed is not None:
            return reconstructed
    return None


def _sglang_int_list(value) -> list[int] | None:
    if value is None:
        return None
    try:
        if _is_torch_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in list(value)]
    except Exception:  # noqa: BLE001
        return None


def _sglang_accept_rows_from_verify_result(obj) -> list[int] | None:
    """Return accepted verify rows per request, including the target/bonus token."""

    accept_lens = _sglang_int_list(_sglang_nested_attr(obj, ("accept_lens",)))
    if accept_lens is not None:
        return [max(int(value), 1) for value in accept_lens]

    for name in (
        "num_correct_drafts_per_req_cpu",
        "num_accepted_drafts_per_req_cpu",
        "accept_length_per_req_cpu",
    ):
        draft_lens = _sglang_int_list(_sglang_nested_attr(obj, (name,)))
        if draft_lens is not None:
            return [max(int(value) + 1, 1) for value in draft_lens]

    return None


def _reconstruct_sglang_accepted_indices_from_lens(obj):
    draft_tokens = _sglang_nested_attr(obj, ("speculative_num_draft_tokens",))
    try:
        draft_tokens = int(draft_tokens)
    except (TypeError, ValueError):
        return None
    if draft_tokens <= 0:
        return None

    accept_lens = _sglang_accept_rows_from_verify_result(obj)
    if not accept_lens:
        return None

    accepted = []
    for req_idx, accept_len in enumerate(accept_lens):
        keep = min(max(int(accept_len), 0), draft_tokens)
        if keep <= 0:
            continue
        start = req_idx * draft_tokens
        accepted.extend(range(start, start + keep))
    if not accepted:
        return None
    return torch.tensor(accepted, dtype=torch.long)


def _find_sglang_verify_positions(*candidates):
    for obj in _iter_sglang_verify_objects(*candidates):
        for name in (
            "positions",
            "hidden_positions",
            "accept_positions",
            "accepted_positions",
        ):
            value = _sglang_nested_attr(obj, (name,))
            if value is not None:
                return value
    return None


def _set_sglang_last_hidden_filter_summary(logits_output, summary: dict) -> None:
    existing = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
    )
    if not isinstance(existing, dict):
        existing = {}
    setattr(
        logits_output,
        _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR,
        {**existing, **summary},
    )


def _mark_sglang_last_hidden_identity_filter(
    logits_output,
    *,
    reason: str,
    positions=None,
    accepted_rows: int | None = None,
) -> bool:
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if not _is_torch_tensor(last_hidden_states):
        return False
    rows = _sglang_hidden_state_rows(last_hidden_states)
    if rows <= 0:
        return False
    hidden_states = getattr(logits_output, "hidden_states", None)
    hidden_rows = _sglang_hidden_state_rows(hidden_states)
    if hidden_rows > 0 and hidden_rows != rows:
        return False
    if (
        accepted_rows is not None
        and int(accepted_rows) > 0
        and int(accepted_rows) != rows
    ):
        return False

    filtered_positions = None
    if positions is not None:
        try:
            if _is_torch_tensor(positions):
                filtered_positions = positions.detach().to(dtype=torch.long).reshape(-1)
            else:
                filtered_positions = torch.tensor(
                    list(positions), dtype=torch.long
                ).reshape(-1)
            if int(filtered_positions.numel()) >= rows:
                filtered_positions = filtered_positions[:rows]
                setattr(
                    logits_output,
                    _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR,
                    filtered_positions,
                )
            else:
                filtered_positions = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to normalize SGLang identity-filter positions: %s", exc
            )
            filtered_positions = None
    pos_head, pos_tail = _sglang_tensor_head_tail(filtered_positions)
    contiguous_rows, positions_contiguous, first_break = (
        _sglang_first_contiguous_position_len(filtered_positions)
    )
    setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, True)
    _set_sglang_last_hidden_filter_summary(
        logits_output,
        {
            "stage": "identity_filtered",
            "reason": reason,
            "hidden_rows_before": rows,
            "hidden_rows_after": rows,
            "index_len": rows,
            "accepted_rows": accepted_rows,
            "hidden_shape": tuple(last_hidden_states.shape),
            "positions_len": int(filtered_positions.numel())
            if _is_torch_tensor(filtered_positions)
            else None,
            "positions_contiguous": positions_contiguous,
            "first_contiguous_rows": contiguous_rows,
            "first_break": first_break,
            "positions_truncated": False,
            "positions_head": pos_head,
            "positions_tail": pos_tail,
        },
    )
    return True


def _mark_sglang_last_hidden_missing_filter(logits_output, *candidates) -> None:
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if not _is_torch_tensor(last_hidden_states):
        return
    hidden_states = getattr(logits_output, "hidden_states", None)
    _set_sglang_last_hidden_filter_summary(
        logits_output,
        {
            "stage": "missing_accept_indices",
            "hidden_rows_before": _sglang_hidden_state_rows(last_hidden_states),
            "hidden_rows_after": None,
            "hidden_shape": tuple(last_hidden_states.shape),
            "base_hidden_rows": _sglang_hidden_state_rows(hidden_states),
            "candidate_types": [
                type(obj).__name__ for obj in candidates if obj is not None
            ][:8],
            "candidate_attrs": [
                {"type": type(obj).__name__, "attrs": _sglang_candidate_attrs(obj)}
                for obj in candidates
                if obj is not None
            ][:4],
        },
    )


def _truncate_sglang_last_hidden_logprob_check_tensors(
    logits_output, rows: int
) -> None:
    for attr in (
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR,
    ):
        value = getattr(logits_output, attr, None)
        if _is_torch_tensor(value) and value.dim() > 0 and int(value.shape[0]) > rows:
            setattr(logits_output, attr, value[:rows])


def _maybe_attach_sglang_raw_top_logprobs_after_filter(
    logits_output,
    *,
    batch=None,
    index_tensor=None,
    filtered_positions=None,
    accepted_rows_per_req: list[int] | None = None,
) -> None:
    if batch is None or not _sglang_raw_top_logprobs_enabled(batch):
        return
    if _sglang_forward_mode_is_target_verify(batch):
        return
    if _is_torch_tensor(
        getattr(logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None)
    ):
        return

    next_token_logits = getattr(logits_output, "next_token_logits", None)
    if not (_is_torch_tensor(next_token_logits) and next_token_logits.dim() >= 2):
        return

    source_row_indices = None
    rows = int(next_token_logits.shape[0])
    if _is_torch_tensor(index_tensor) and int(index_tensor.numel()) > 0:
        index_tensor = index_tensor.to(dtype=torch.long).reshape(-1)
        index_len = int(index_tensor.numel())
        try:
            index_max = int(index_tensor.max().detach().cpu().item())
        except Exception:  # noqa: BLE001
            index_max = rows
        identity_filtered = False
        if rows == index_len:
            try:
                identity_filtered = bool(
                    torch.equal(
                        index_tensor.detach().cpu(),
                        torch.arange(index_len, dtype=torch.long),
                    )
                )
            except Exception:  # noqa: BLE001
                identity_filtered = False
        if rows != index_len or not identity_filtered:
            if index_max < rows:
                source_row_indices = index_tensor

    if _is_torch_tensor(source_row_indices) and _is_torch_tensor(filtered_positions):
        attach_rows = int(filtered_positions.reshape(-1).numel())
        if attach_rows > 0 and int(source_row_indices.numel()) > attach_rows:
            source_row_indices = source_row_indices[:attach_rows]

    _attach_sglang_raw_top_logprobs(
        logits_output,
        batch,
        positions=filtered_positions,
        source_row_indices=source_row_indices,
        rows_per_req=accepted_rows_per_req,
    )


def _filter_sglang_drafter_last_hidden_output(
    logits_output,
    index,
    positions=None,
    batch=None,
    verify_result=None,
    accepted_rows_per_req: list[int] | None = None,
    filter_base_hidden_states: bool = True,
) -> None:
    global _SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT
    try:
        index_tensor = index if _is_torch_tensor(index) else torch.tensor(list(index))
        index_tensor = index_tensor.to(dtype=torch.long).reshape(-1)
        index_len = int(index_tensor.numel())
        index_cpu = index_tensor.detach().to("cpu")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to normalize SGLang drafter accepted indices: %s", exc)
        return
    accept_head, accept_tail = _sglang_tensor_head_tail(index_cpu)
    if accepted_rows_per_req is None and verify_result is not None:
        try:
            accepted_rows_per_req = _sglang_decode_accept_rows_per_req(verify_result)
        except Exception:  # noqa: BLE001
            accepted_rows_per_req = None

    base_hidden_states = getattr(logits_output, "hidden_states", None)
    base_hidden_filtered = None
    if (
        filter_base_hidden_states
        and _is_torch_tensor(base_hidden_states)
        and not bool(
            getattr(
                logits_output,
                "_verl_drafter_base_hidden_filtered_by_accept_indices",
                False,
            )
        )
    ):
        try:
            base_index = index_tensor.to(
                device=base_hidden_states.device, dtype=torch.long
            )
            if base_hidden_states.dim() == 3:
                base_hidden_states = base_hidden_states.reshape(
                    -1, base_hidden_states.shape[-1]
                )
            if index_len > 0 and int(base_hidden_states.shape[0]) > int(
                base_index.max().item()
            ):
                base_hidden_filtered = base_hidden_states[base_index]
                setattr(
                    logits_output,
                    "_verl_drafter_base_hidden_filtered_by_accept_indices_tensor",
                    base_hidden_filtered,
                )
                setattr(
                    logits_output,
                    "_verl_drafter_base_hidden_filtered_by_accept_indices",
                    True,
                )
            elif int(base_hidden_states.shape[0]) == index_len:
                base_hidden_filtered = base_hidden_states
                setattr(
                    logits_output,
                    "_verl_drafter_base_hidden_filtered_by_accept_indices_tensor",
                    base_hidden_filtered,
                )
                setattr(
                    logits_output,
                    "_verl_drafter_base_hidden_filtered_by_accept_indices",
                    True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to filter SGLang drafter base hidden states by accepted indices: %s",
                exc,
            )

    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if last_hidden_states is None:
        if bool(getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, False)):
            _maybe_attach_sglang_raw_top_logprobs_after_filter(
                logits_output,
                batch=batch,
                filtered_positions=getattr(
                    logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
                ),
                accepted_rows_per_req=accepted_rows_per_req,
            )
            return
        filtered_positions = None
        if positions is not None:
            try:
                if _is_torch_tensor(positions):
                    positions_tensor = positions.to(dtype=torch.long).reshape(-1)
                else:
                    positions_tensor = torch.tensor(
                        list(positions), dtype=torch.long
                    ).reshape(-1)
                if index_len > 0:
                    position_index = index_tensor.to(
                        device=positions_tensor.device, dtype=torch.long
                    )
                    if int(positions_tensor.numel()) > int(position_index.max().item()):
                        filtered_positions = positions_tensor[position_index].detach()
                        setattr(
                            logits_output,
                            _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR,
                            filtered_positions,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to filter SGLang drafter positions without last_hidden: %s",
                    exc,
                )
        _filter_sglang_last_hidden_logprob_check_tensors(logits_output, index_tensor)
        setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, True)
        pos_head, pos_tail = _sglang_tensor_head_tail(filtered_positions)
        contiguous_rows, positions_contiguous, first_break = (
            _sglang_first_contiguous_position_len(filtered_positions)
        )
        raw_topk_logprobs = getattr(
            logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None
        )
        _set_sglang_last_hidden_filter_summary(
            logits_output,
            {
                "stage": "filtered_no_last_hidden",
                "hidden_rows_before": None,
                "hidden_rows_after": None,
                "base_hidden_rows_after": _sglang_hidden_state_rows(
                    getattr(logits_output, "hidden_states", None)
                ),
                "raw_topk_rows_after": (
                    int(raw_topk_logprobs.shape[0])
                    if _is_torch_tensor(raw_topk_logprobs)
                    and raw_topk_logprobs.dim() > 0
                    else None
                ),
                "index_len": index_len,
                "accept_indices_head": accept_head,
                "accept_indices_tail": accept_tail,
                "positions_len": int(filtered_positions.numel())
                if _is_torch_tensor(filtered_positions)
                else None,
                "positions_contiguous": positions_contiguous,
                "first_contiguous_rows": contiguous_rows,
                "first_break": first_break,
                "positions_truncated": False,
                "positions_head": pos_head,
                "positions_tail": pos_tail,
            },
        )
        _maybe_attach_sglang_raw_top_logprobs_after_filter(
            logits_output,
            batch=batch,
            index_tensor=index_tensor,
            filtered_positions=filtered_positions,
            accepted_rows_per_req=accepted_rows_per_req,
        )
        return
    if bool(getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, False)):
        filtered_positions = getattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
        )
        pos_head, pos_tail = _sglang_tensor_head_tail(filtered_positions)
        hidden_shape = (
            tuple(last_hidden_states.shape)
            if _is_torch_tensor(last_hidden_states)
            else None
        )
        existing_summary = getattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
        )
        if not isinstance(existing_summary, dict):
            existing_summary = {}
        setattr(
            logits_output,
            _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR,
            {
                **existing_summary,
                "stage": "already_filtered",
                "hidden_rows_after": _sglang_hidden_state_rows(last_hidden_states),
                "index_len": index_len,
                "hidden_shape": hidden_shape,
                "accept_indices_head": accept_head,
                "accept_indices_tail": accept_tail,
                "positions_len": int(filtered_positions.numel())
                if _is_torch_tensor(filtered_positions)
                else None,
                "positions_head": pos_head,
                "positions_tail": pos_tail,
            },
        )
        _log_sglang_last_hidden_logprob_check(logits_output, stage="already_filtered")
        _maybe_attach_sglang_raw_top_logprobs_after_filter(
            logits_output,
            batch=batch,
            index_tensor=None,
            filtered_positions=filtered_positions,
            accepted_rows_per_req=accepted_rows_per_req,
        )
        return
    if positions is not None:
        setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, positions)

    # Keep this helper idempotent through the explicit filtered flag above.
    # Equal row counts do not mean the accepted-index filter has already been
    # applied: EAGLE can accept every row while still requiring a tree-order
    # reindex, and SGLang only filters logits_output.hidden_states itself.
    if _is_torch_tensor(last_hidden_states) and last_hidden_states.dim() > 0:
        hidden_rows = int(last_hidden_states.shape[0])
        filter_summary = {
            "stage": "before_filter",
            "hidden_rows_before": hidden_rows,
            "index_len": index_len,
            "hidden_shape": tuple(last_hidden_states.shape),
            "accept_indices_head": accept_head,
            "accept_indices_tail": accept_tail,
        }
        if (
            _sglang_last_hidden_logprob_check_enabled()
            and _SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT
            < _sglang_last_hidden_logprob_check_max_logs()
        ):
            try:
                identity_prefix = bool(
                    int(index_cpu.numel()) > 0
                    and torch.equal(
                        index_cpu[: min(int(index_cpu.numel()), 16)],
                        torch.arange(min(int(index_cpu.numel()), 16)),
                    )
                )
                index_min = (
                    int(index_cpu.min().item()) if int(index_cpu.numel()) > 0 else None
                )
                index_max = (
                    int(index_cpu.max().item()) if int(index_cpu.numel()) > 0 else None
                )
            except Exception as exc:  # noqa: BLE001
                accept_head = [f"error:{exc}"]
                accept_tail = accept_head
                identity_prefix = None
                index_min = None
                index_max = None
            filter_summary.update(
                {
                    "index_min": index_min,
                    "index_max": index_max,
                    "identity_prefix": identity_prefix,
                    "index_head": accept_head,
                    "index_tail": accept_tail,
                }
            )
            logger.warning(
                "[sglang last_hidden filter check] before hidden_rows=%s index_len=%s index_min=%s "
                "index_max=%s identity_prefix=%s accept_indices_head=%s accept_indices_tail=%s hidden_shape=%s",
                hidden_rows,
                index_len,
                index_min,
                index_max,
                identity_prefix,
                accept_head,
                accept_tail,
                tuple(last_hidden_states.shape),
            )
            _SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT += 1
        else:
            try:
                filter_summary.update(
                    {
                        "index_min": int(index_cpu.min().item())
                        if int(index_cpu.numel()) > 0
                        else None,
                        "index_max": int(index_cpu.max().item())
                        if int(index_cpu.numel()) > 0
                        else None,
                        "index_head": accept_head,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        setattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, filter_summary
        )
        if hidden_rows < index_len:
            logger.warning(
                "Skip filtering SGLang drafter last-hidden output: hidden_rows=%s < index_len=%s",
                hidden_rows,
                index_len,
            )
            return

    try:
        last_hidden_index = index_tensor.to(
            device=last_hidden_states.device, dtype=torch.long
        )
        filtered_last_hidden_states = last_hidden_states[last_hidden_index]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to filter SGLang drafter last-hidden output by accepted indices: %s",
            exc,
        )
        return
    setattr(
        logits_output,
        _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR,
        filtered_last_hidden_states,
    )
    last_hidden_positions = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
    )
    filtered_positions = None
    if _is_torch_tensor(last_hidden_positions):
        try:
            position_index = index_tensor.to(
                device=last_hidden_positions.device, dtype=torch.long
            )
            filtered_positions = last_hidden_positions[position_index].detach()
            setattr(
                logits_output,
                _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR,
                filtered_positions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to filter SGLang drafter last-hidden positions by accepted indices: %s",
                exc,
            )
    _filter_sglang_last_hidden_logprob_check_tensors(logits_output, index_tensor)
    contiguous_rows, positions_contiguous, first_break = (
        _sglang_first_contiguous_position_len(filtered_positions)
    )
    if contiguous_rows is not None and contiguous_rows < int(
        filtered_last_hidden_states.shape[0]
    ):
        logger.debug(
            "[sglang last_hidden filter check] positions are not globally contiguous; keep all rows=%s "
            "first_break=%s positions_head=%s positions_tail=%s",
            int(filtered_last_hidden_states.shape[0]),
            first_break,
            *_sglang_tensor_head_tail(filtered_positions),
        )
    setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, True)
    filter_summary = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
    )
    if isinstance(filter_summary, dict):
        pos_head, pos_tail = _sglang_tensor_head_tail(filtered_positions)
        filter_summary = {
            **filter_summary,
            "stage": "filtered",
            "hidden_rows_after": int(filtered_last_hidden_states.shape[0]),
            "positions_len": int(filtered_positions.numel())
            if _is_torch_tensor(filtered_positions)
            else None,
            "positions_contiguous": positions_contiguous,
            "first_contiguous_rows": contiguous_rows,
            "first_break": first_break,
            "positions_truncated": False,
            "positions_head": pos_head,
            "positions_tail": pos_tail,
            "filtered_shape": tuple(filtered_last_hidden_states.shape),
        }
        setattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, filter_summary
        )
        if (
            _sglang_last_hidden_logprob_check_enabled()
            and _SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT
            < _sglang_last_hidden_logprob_check_max_logs()
        ):
            logger.warning(
                "[sglang last_hidden filter check] after hidden_rows_before=%s hidden_rows_after=%s "
                "accept_len=%s accept_indices_head=%s accept_indices_tail=%s positions_len=%s "
                "positions_contiguous=%s positions_head=%s positions_tail=%s",
                filter_summary.get("hidden_rows_before"),
                filter_summary.get("hidden_rows_after"),
                index_len,
                accept_head,
                accept_tail,
                filter_summary.get("positions_len"),
                positions_contiguous,
                pos_head,
                pos_tail,
            )
            _SGLANG_LAST_HIDDEN_FILTER_DEBUG_LOG_COUNT += 1
    _maybe_attach_sglang_raw_top_logprobs_after_filter(
        logits_output,
        batch=batch,
        index_tensor=index_tensor,
        filtered_positions=filtered_positions,
        accepted_rows_per_req=accepted_rows_per_req,
    )
    _log_sglang_last_hidden_logprob_check(logits_output, stage="filtered")


def _filter_sglang_drafter_last_hidden_from_verify_result(
    logits_output, result, positions=None, batch=None
) -> None:
    accepted_indices = _find_sglang_accepted_indices(result)
    if positions is None:
        positions = _find_sglang_verify_positions(result)
    if accepted_indices is None:
        accepted_rows = None
        accepted_rows_per_req = None
        try:
            accepted_rows_per_req = _sglang_decode_accept_rows_per_req(result)
            if accepted_rows_per_req is not None:
                accepted_rows = sum(int(rows) for rows in accepted_rows_per_req)
        except Exception:  # noqa: BLE001
            accepted_rows = None
        if _mark_sglang_last_hidden_identity_filter(
            logits_output,
            reason="missing_accept_indices_rows_already_aligned",
            positions=positions,
            accepted_rows=accepted_rows,
        ):
            filtered_positions = getattr(
                logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
            )
            _maybe_attach_sglang_raw_top_logprobs_after_filter(
                logits_output,
                batch=batch,
                filtered_positions=filtered_positions,
                accepted_rows_per_req=accepted_rows_per_req,
            )
            return
        _mark_sglang_last_hidden_missing_filter(logits_output, result)
        return
    _filter_sglang_drafter_last_hidden_output(
        logits_output,
        accepted_indices,
        positions=positions,
        batch=batch,
        verify_result=result,
    )


def _sglang_hidden_chunk_rows(chunk) -> int:
    if _is_torch_tensor(chunk):
        if chunk.dim() <= 1:
            return 1
        return int(chunk.shape[0])
    try:
        return len(chunk)
    except TypeError:
        return 1


def _slice_sglang_hidden_chunk(chunk, start: int, end: int):
    if start <= 0 and end >= _sglang_hidden_chunk_rows(chunk):
        return chunk
    return chunk[start:end]


def _to_cpu_sglang_hidden_chunk(chunk):
    if _is_torch_tensor(chunk):
        return chunk.detach().to("cpu", copy=True)
    return chunk


def _sglang_req_prompt_len(req) -> int:
    custom_prompt_len = _int_or_none(
        _sglang_req_custom_params(req).get(_VERL_HIDDEN_STATE_PROMPT_LEN_PARAM)
    )
    if custom_prompt_len is not None:
        return max(custom_prompt_len, 0)
    try:
        return len(getattr(req, "origin_input_ids", []) or [])
    except TypeError:
        return 0


def _sglang_req_hidden_prefix_cache_rows(req) -> int:
    value = _int_or_none(getattr(req, "_verl_hidden_prefix_cache_rows", None))
    if value is not None:
        return max(value, 0)
    custom_value = _int_or_none(_sglang_req_custom_params(req).get("prefix_cache_rows"))
    if custom_value is not None:
        return max(custom_value, 0)
    return 0


def _sglang_hidden_window_config(req) -> dict[str, int] | None:
    custom_params = _sglang_req_custom_params(req)
    if not _custom_flag_enabled(
        custom_params.get(_VERL_DRAFTER_HIDDEN_WINDOW_PARAM, False)
    ):
        return None

    prompt_len = _sglang_req_prompt_len(req)
    prefix_cache_rows = _sglang_req_hidden_prefix_cache_rows(req)
    explicit_window_start = _int_or_none(
        custom_params.get(_VERL_HIDDEN_STATE_WINDOW_START_PARAM)
    )
    explicit_window_end = _int_or_none(
        custom_params.get(_VERL_HIDDEN_STATE_WINDOW_END_PARAM)
    )
    if explicit_window_start is not None and explicit_window_end is not None:
        window_start = max(int(explicit_window_start), 0)
        window_end = max(int(explicit_window_end), window_start)
        if window_end <= window_start:
            return None
        return {
            "prompt_len": prompt_len,
            "prefix_cache_rows": prefix_cache_rows,
            "window_start": window_start,
            "window_end": window_end,
        }

    front_tokens = _positive_int_or_none(
        custom_params.get(_VERL_HIDDEN_STATE_FRONT_TOKENS_PARAM)
    )
    if front_tokens is None:
        front_tokens = _positive_int_or_none(
            custom_params.get(_VERL_HIDDEN_STATE_MAX_ROWS_PARAM)
        )
    if front_tokens is None:
        front_tokens = _positive_int_or_none(
            getattr(req, "_verl_hidden_state_max_rows", None)
        )
    if front_tokens is None:
        return None

    window_start = max(prefix_cache_rows, max(prompt_len - 1, 0))
    return {
        "prompt_len": prompt_len,
        "prefix_cache_rows": prefix_cache_rows,
        "window_start": window_start,
        "window_end": window_start + front_tokens,
    }


def _append_sglang_hidden_chunk_payload(
    req, chunk, metadata: dict[str, int] | None = None
) -> int:
    appended = _to_cpu_sglang_hidden_chunk(chunk)
    appended_rows = _sglang_hidden_chunk_rows(appended)
    if metadata is None:
        req.hidden_states.append(appended)
    else:
        req.hidden_states.append(
            {
                _VERL_HIDDEN_STATE_METADATA_MARKER: True,
                "hidden_states": appended,
                **metadata,
            }
        )
    return appended_rows


def _slice_sglang_debug_rows(value, row_slice):
    if row_slice is None or not _is_torch_tensor(value) or value.dim() <= 0:
        return value
    try:
        if isinstance(row_slice, int):
            return value[row_slice : row_slice + 1]
        return value[row_slice]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to slice SGLang raw top-k debug rows: %s", exc)
        return value


def _sglang_raw_target_logprobs_tensor(logits_output, row_slice=None):
    raw_topk_ids = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR, None
    )
    raw_topk_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None
    )
    if not (_is_torch_tensor(raw_topk_ids) and _is_torch_tensor(raw_topk_logprobs)):
        return None
    if raw_topk_ids.dim() != 2 or raw_topk_logprobs.dim() != 2:
        return None
    raw_topk_ids = _slice_sglang_debug_rows(raw_topk_ids, row_slice)
    raw_topk_logprobs = _slice_sglang_debug_rows(raw_topk_logprobs, row_slice)
    rows = min(int(raw_topk_ids.shape[0]), int(raw_topk_logprobs.shape[0]))
    topk = min(int(raw_topk_ids.shape[1]), int(raw_topk_logprobs.shape[1]))
    if rows <= 0 or topk <= 0:
        return None
    logprobs = raw_topk_logprobs[:rows, :topk].detach().to(dtype=torch.float32)
    token_ids = (
        raw_topk_ids[:rows, :topk]
        .detach()
        .to(device=logprobs.device, dtype=torch.float32)
    )
    return torch.stack((logprobs, token_ids), dim=-1).contiguous()


def _sglang_raw_target_logprobs_positions_tensor(logits_output, row_slice=None):
    raw_positions = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR, None
    )
    if not _is_torch_tensor(raw_positions):
        return None
    raw_positions = _slice_sglang_debug_rows(raw_positions, row_slice)
    if not _is_torch_tensor(raw_positions):
        return None
    return raw_positions.detach().to("cpu", dtype=torch.long).reshape(-1).contiguous()


def _sglang_debug_row_slice_for_hidden_chunk(
    logits_output, start: int, end: int, chunk_rows: int
):
    row_slice = slice(start, end)
    raw_topk_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None
    )
    if not (_is_torch_tensor(raw_topk_logprobs) and raw_topk_logprobs.dim() > 0):
        return row_slice
    try:
        raw_rows = int(raw_topk_logprobs.shape[0])
        chunk_rows = int(chunk_rows)
        start = int(start)
        end = int(end)
    except (TypeError, ValueError):
        return row_slice
    if raw_rows <= 0 or chunk_rows <= 0:
        return row_slice
    if 0 <= start <= end <= raw_rows:
        return row_slice
    if raw_rows == chunk_rows:
        return slice(0, chunk_rows)
    return row_slice


def _raw_topk_metadata_summary(raw_target_logprobs, raw_positions=None) -> dict | None:
    if not _is_torch_tensor(raw_target_logprobs) or raw_target_logprobs.dim() != 3:
        return None
    rows = int(raw_target_logprobs.shape[0])
    topk = int(raw_target_logprobs.shape[1])
    if rows <= 0 or topk <= 0:
        return None
    summary = {
        "target_logprobs_source": _VERL_TARGET_LOGPROBS_SOURCE_RAW_HIDDEN_METADATA,
        "shape": tuple(raw_target_logprobs.shape),
        "rows": rows,
        "topk": topk,
        "top1_ids_head": [
            int(x)
            for x in raw_target_logprobs[: min(rows, 8), 0, 1].detach().cpu().tolist()
        ],
    }
    if _is_torch_tensor(raw_positions):
        raw_positions = raw_positions.detach().to("cpu", dtype=torch.long).reshape(-1)
        if int(raw_positions.numel()) == rows:
            summary.update(
                {
                    "positions_len": rows,
                    "positions_valid": int((raw_positions >= 0).sum().item()),
                    "positions_head": [
                        int(x) for x in raw_positions[: min(rows, 8)].tolist()
                    ],
                    "positions_tail": [
                        int(x) for x in raw_positions[-min(rows, 8) :].tolist()
                    ],
                    "positions_contiguous": bool(
                        rows <= 1
                        or torch.equal(raw_positions[1:], raw_positions[:-1] + 1)
                    ),
                }
            )
    return summary


def _slice_sglang_row_aligned_metadata(
    metadata: dict | None, start: int, end: int, keep_mask=None
) -> dict | None:
    if not isinstance(metadata, dict):
        return metadata
    result = dict(metadata)
    raw_target_logprobs = result.get(_VERL_RAW_TARGET_LOGPROBS_METADATA_KEY)
    if _is_torch_tensor(raw_target_logprobs) and raw_target_logprobs.dim() > 0:
        try:
            raw_rows = int(raw_target_logprobs.shape[0])
            raw_positions = result.get(_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY)
            if _is_torch_tensor(raw_positions):
                raw_positions = (
                    raw_positions.detach().to("cpu", dtype=torch.long).reshape(-1)
                )
                if int(raw_positions.numel()) != raw_rows:
                    raw_positions = None
            slice_start = min(max(int(start), 0), raw_rows)
            slice_end = min(max(int(end), slice_start), raw_rows)
            if slice_end > slice_start:
                sliced = raw_target_logprobs[slice_start:slice_end]
                sliced_positions = (
                    raw_positions[slice_start:slice_end]
                    if _is_torch_tensor(raw_positions)
                    else None
                )
                if _is_torch_tensor(keep_mask):
                    mask = (
                        keep_mask.detach()
                        .to(device=sliced.device, dtype=torch.bool)
                        .reshape(-1)
                    )
                    if int(mask.numel()) != max(int(end) - int(start), 0):
                        mask = None
                    else:
                        mask_start = slice_start - int(start)
                        mask = mask[mask_start : mask_start + int(sliced.shape[0])]
                    if _is_torch_tensor(mask) and int(mask.numel()) == int(
                        sliced.shape[0]
                    ):
                        sliced = sliced[mask]
                        if _is_torch_tensor(sliced_positions):
                            sliced_positions = sliced_positions[mask.detach().to("cpu")]
                result[_VERL_RAW_TARGET_LOGPROBS_METADATA_KEY] = sliced
                if _is_torch_tensor(sliced_positions):
                    result[_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY] = (
                        sliced_positions
                    )
                else:
                    result.pop(_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY, None)
                raw_summary = _raw_topk_metadata_summary(sliced, sliced_positions)
                if raw_summary is not None:
                    result[_VERL_RAW_TOPK_LOGPROB_CHECK_METADATA_KEY] = raw_summary
            else:
                result.pop(_VERL_RAW_TARGET_LOGPROBS_METADATA_KEY, None)
                result.pop(_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY, None)
                result.pop(_VERL_RAW_TOPK_LOGPROB_CHECK_METADATA_KEY, None)
                if (
                    result.get(_VERL_TARGET_LOGPROBS_SOURCE_METADATA_KEY)
                    == _VERL_TARGET_LOGPROBS_SOURCE_RAW_HIDDEN_METADATA
                ):
                    result.pop(_VERL_TARGET_LOGPROBS_SOURCE_METADATA_KEY, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to slice raw target logprobs metadata: %s", exc)
    return result


def _sglang_hidden_debug_metadata(
    logits_output, row_slice=None, raw_positions=None
) -> dict:
    metadata = {}
    fingerprint = getattr(
        logits_output, "_verl_drafter_lh_check_lm_head_fingerprint", None
    )
    if isinstance(fingerprint, dict):
        metadata["lm_head_fingerprint"] = fingerprint
    lh_check_summary = getattr(logits_output, _VERL_DRAFTER_LH_CHECK_SUMMARY_ATTR, None)
    if isinstance(lh_check_summary, dict):
        metadata["last_hidden_logprob_check"] = lh_check_summary
    raw_target_logprobs = _sglang_raw_target_logprobs_tensor(
        logits_output, row_slice=row_slice
    )
    if _is_torch_tensor(raw_target_logprobs):
        raw_target_logprobs = raw_target_logprobs.detach().to("cpu")
        metadata[_VERL_TARGET_LOGPROBS_SOURCE_METADATA_KEY] = (
            _VERL_TARGET_LOGPROBS_SOURCE_RAW_HIDDEN_METADATA
        )
        metadata[_VERL_RAW_TARGET_LOGPROBS_METADATA_KEY] = raw_target_logprobs
        normalized_raw_positions = _sglang_raw_target_logprobs_positions_tensor(
            logits_output, row_slice=row_slice
        )
        if not (
            _is_torch_tensor(normalized_raw_positions)
            and int(normalized_raw_positions.numel())
            == int(raw_target_logprobs.shape[0])
        ):
            normalized_raw_positions = _normalize_sglang_raw_topk_positions(
                raw_positions,
                int(raw_target_logprobs.shape[0]),
                device="cpu",
            )
            if _is_torch_tensor(normalized_raw_positions):
                normalized_raw_positions = (normalized_raw_positions + 1).contiguous()
        if _is_torch_tensor(normalized_raw_positions):
            metadata[_VERL_RAW_TARGET_LOGPROBS_POSITIONS_METADATA_KEY] = (
                normalized_raw_positions
            )
        raw_summary = _raw_topk_metadata_summary(
            raw_target_logprobs, normalized_raw_positions
        )
        if raw_summary is not None:
            if isinstance(lh_check_summary, dict):
                raw_summary["attach_top1_match"] = lh_check_summary.get("top1_match")
            metadata[_VERL_RAW_TOPK_LOGPROB_CHECK_METADATA_KEY] = raw_summary
    filter_summary = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
    )
    if isinstance(filter_summary, dict):
        metadata["last_hidden_filter"] = filter_summary
    select_summary = getattr(
        logits_output, "_verl_drafter_last_hidden_select_summary", None
    )
    if isinstance(select_summary, dict):
        metadata["last_hidden_select"] = select_summary
    return metadata


def _sglang_hidden_debug_metadata_for_req(
    req,
    logits_output,
    start: int,
    end: int,
    chunk_rows: int,
    *,
    positions=None,
) -> dict | None:
    if not getattr(req, "return_hidden_states", False):
        return None
    debug_row_slice = _sglang_debug_row_slice_for_hidden_chunk(
        logits_output, start, end, chunk_rows
    )
    raw_positions = positions if positions is not None and int(chunk_rows) > 0 else None
    return _sglang_hidden_debug_metadata(
        logits_output, row_slice=debug_row_slice, raw_positions=raw_positions
    )


def _mark_sglang_hidden_states_stream_final(req) -> None:
    setattr(req, _VERL_HIDDEN_STATES_STREAM_FINAL_ATTR, True)


def _refresh_sglang_batch_return_hidden_states(batch) -> None:
    if batch is not None:
        setattr(
            batch, "return_hidden_states", _sglang_batch_requests_hidden_states(batch)
        )


def _finish_sglang_hidden_state_capture(req, batch=None) -> None:
    req.return_hidden_states = False
    _refresh_sglang_batch_return_hidden_states(batch)


def _sglang_req_should_stream_hidden_states(req) -> bool:
    if getattr(req, _VERL_HIDDEN_STATES_STREAM_FINAL_ATTR, False):
        try:
            return bool(req.finished())
        except Exception:  # noqa: BLE001
            return False
    return bool(getattr(req, "return_hidden_states", False))


def _append_sglang_hidden_state_chunk_with_budget(
    req,
    chunk,
    *,
    position_start: int | None = None,
    positions=None,
    prefix_cache_rows: int | None = None,
    extra_metadata: dict | None = None,
    batch=None,
) -> None:
    if not getattr(req, "return_hidden_states", False):
        return

    if prefix_cache_rows is not None:
        setattr(req, "_verl_hidden_prefix_cache_rows", max(int(prefix_cache_rows), 0))

    chunk_rows = _sglang_hidden_chunk_rows(chunk)
    if chunk_rows <= 0:
        return

    if positions is not None:
        if _is_torch_tensor(positions):
            positions = positions.detach().to("cpu", dtype=torch.long).reshape(-1)
        else:
            positions = torch.tensor(list(positions), dtype=torch.long).reshape(-1)
        if int(positions.numel()) < chunk_rows:
            logger.warning(
                "Ignore short SGLang hidden positions metadata: positions=%s chunk_rows=%s",
                int(positions.numel()),
                chunk_rows,
            )
            positions = None
        elif int(positions.numel()) > chunk_rows:
            positions = positions[:chunk_rows]

    if position_start is None:
        position_start = _int_or_none(getattr(req, "_verl_hidden_next_position", None))
        if position_start is None:
            if positions is not None and int(positions.numel()) > 0:
                position_start = int(positions[0].item())
            else:
                position_start = _sglang_req_hidden_prefix_cache_rows(req)
    position_start = max(int(position_start), 0)
    position_end = position_start + chunk_rows
    setattr(req, "_verl_hidden_next_position", position_end)
    if positions is None:
        positions = torch.arange(position_start, position_end, dtype=torch.long)

    window_config = _sglang_hidden_window_config(req)
    if window_config is not None:
        _mark_sglang_hidden_states_stream_final(req)
        if getattr(req, "_verl_hidden_state_window_done", False):
            return
        window_start = window_config["window_start"]
        window_end = window_config["window_end"]
        if positions is not None:
            keep_mask = (positions >= window_start) & (positions < window_end)
            if not bool(keep_mask.any()):
                if int(positions[-1].item()) >= window_end - 1:
                    setattr(req, "_verl_hidden_state_window_done", True)
                    _finish_sglang_hidden_state_capture(req, batch)
                return
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
            local_start = int(keep_idx[0].item())
            local_end = int(keep_idx[-1].item()) + 1
            clipped_chunk = _slice_sglang_hidden_chunk(chunk, local_start, local_end)
            clipped_positions = positions[local_start:local_end]
            clipped_metadata = _slice_sglang_row_aligned_metadata(
                extra_metadata, local_start, local_end
            )
            if not bool(keep_mask[local_start:local_end].all()):
                local_keep_mask = keep_mask[local_start:local_end]
                clipped_chunk = clipped_chunk[local_keep_mask]
                clipped_positions = clipped_positions[local_keep_mask]
                clipped_metadata = _slice_sglang_row_aligned_metadata(
                    extra_metadata,
                    local_start,
                    local_end,
                    keep_mask=local_keep_mask,
                )
            metadata = {
                "position_start": int(clipped_positions[0].item()),
                "position_end": int(clipped_positions[-1].item()) + 1,
                "positions": clipped_positions,
                "prefix_cache_rows": window_config["prefix_cache_rows"],
                "window_start": window_start,
                "window_end": window_end,
                **(clipped_metadata or {}),
            }
            _append_sglang_hidden_chunk_payload(req, clipped_chunk, metadata)
            if int(positions[-1].item()) >= window_end - 1:
                setattr(req, "_verl_hidden_state_window_done", True)
                _finish_sglang_hidden_state_capture(req, batch)
            return

        clipped_start = max(position_start, window_start)
        clipped_end = min(position_end, window_end)
        if clipped_start >= clipped_end:
            if position_end >= window_end:
                setattr(req, "_verl_hidden_state_window_done", True)
                _finish_sglang_hidden_state_capture(req, batch)
            return

        local_start = clipped_start - position_start
        local_end = clipped_end - position_start
        clipped_chunk = _slice_sglang_hidden_chunk(chunk, local_start, local_end)
        clipped_metadata = _slice_sglang_row_aligned_metadata(
            extra_metadata, local_start, local_end
        )
        _append_sglang_hidden_chunk_payload(
            req,
            clipped_chunk,
            {
                "position_start": clipped_start,
                "position_end": clipped_end,
                "prefix_cache_rows": window_config["prefix_cache_rows"],
                "window_start": window_start,
                "window_end": window_end,
                **(clipped_metadata or {}),
            },
        )
        if clipped_end >= window_end:
            setattr(req, "_verl_hidden_state_window_done", True)
            _finish_sglang_hidden_state_capture(req, batch)
        return

    if getattr(req, "_verl_hidden_state_budget_done", False):
        return

    max_rows = getattr(req, "_verl_hidden_state_max_rows", None)
    if max_rows is not None:
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            max_rows = None

    collected_rows = int(getattr(req, "_verl_hidden_state_rows", 0) or 0)
    if max_rows is not None and max_rows > 0:
        _mark_sglang_hidden_states_stream_final(req)
        remaining_rows = max_rows - collected_rows
        if remaining_rows <= 0:
            setattr(req, "_verl_hidden_state_budget_done", True)
            _finish_sglang_hidden_state_capture(req, batch)
            return
        if (
            _is_torch_tensor(chunk)
            and chunk.dim() > 0
            and int(chunk.shape[0]) > remaining_rows
        ):
            chunk = chunk[:remaining_rows]
            if positions is not None:
                positions = positions[:remaining_rows]
        elif not _is_torch_tensor(chunk):
            try:
                if len(chunk) > remaining_rows:
                    chunk = chunk[:remaining_rows]
                    if positions is not None:
                        positions = positions[:remaining_rows]
            except TypeError:
                pass

    metadata = _slice_sglang_row_aligned_metadata(
        extra_metadata,
        0,
        _sglang_hidden_chunk_rows(chunk),
    )
    if positions is not None and int(positions.numel()) > 0:
        metadata = {
            "position_start": int(positions[0].item()),
            "position_end": int(positions[-1].item()) + 1,
            "positions": positions,
            **(metadata or {}),
        }
    appended_rows = _append_sglang_hidden_chunk_payload(req, chunk, metadata)
    collected_rows += appended_rows
    setattr(req, "_verl_hidden_state_rows", collected_rows)
    if max_rows is not None and max_rows > 0 and collected_rows >= max_rows:
        setattr(req, "_verl_hidden_state_budget_done", True)
        _finish_sglang_hidden_state_capture(req, batch)


def _append_sglang_prefill_hidden_states(
    req, logits_output, hidden_state_offset: int, extend_input_len: int, batch=None
) -> int:
    try:
        rows = max(int(extend_input_len), 0)
    except (TypeError, ValueError):
        rows = len(getattr(req, "origin_input_ids", []) or [])
    if rows <= 0:
        return hidden_state_offset

    end = hidden_state_offset + rows
    if not getattr(req, "return_hidden_states", False):
        return end

    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        if not getattr(req, "_verl_logged_missing_prefill_hidden_states", False):
            logger.warning(
                "SGLang did not return prefill hidden states for drafter collection: "
                "request_dflash_aux=%s logits_output_type=%s",
                _sglang_req_requests_dflash_aux_hidden(req),
                type(logits_output).__name__,
            )
            setattr(req, "_verl_logged_missing_prefill_hidden_states", True)
        return hidden_state_offset

    prompt_len = len(getattr(req, "origin_input_ids", []) or [])
    prefix_cache_rows = max(prompt_len - rows, 0)
    chunk = hidden_states[hidden_state_offset:end]
    chunk = _sglang_concat_last_hidden_for_drafter(
        req,
        logits_output,
        chunk,
        _slice_sglang_drafter_last_hidden_output(
            logits_output, slice(hidden_state_offset, end)
        ),
    )
    chunk_rows = _sglang_hidden_chunk_rows(chunk)
    chunk_positions = torch.arange(
        prefix_cache_rows, prefix_cache_rows + chunk_rows, dtype=torch.long
    )
    _append_sglang_hidden_state_chunk_with_budget(
        req,
        chunk,
        position_start=prefix_cache_rows,
        positions=chunk_positions,
        prefix_cache_rows=prefix_cache_rows,
        extra_metadata=_sglang_hidden_debug_metadata_for_req(
            req,
            logits_output,
            hidden_state_offset,
            end,
            chunk_rows,
            positions=chunk_positions,
        ),
        batch=batch,
    )
    return end


def _sglang_decode_accept_rows_per_req(result) -> list[int] | None:
    """Return per-request accepted row counts for SGLang speculative decode output.

    Older EAGLE spec-v1 outputs use accept_length_per_req_cpu. Newer spec-v1
    outputs use num_correct_drafts_per_req_cpu or num_accepted_drafts_per_req_cpu.
    These draft counts do not include the target/bonus token. Spec-v2
    accept_lens already includes it.
    """

    draft_tokens = _positive_int_or_none(
        getattr(result, "speculative_num_draft_tokens", None)
    )

    def _clamp_to_verify_rows(rows: list[int]) -> list[int]:
        if draft_tokens is None:
            return rows
        return [min(row, draft_tokens) for row in rows]

    rows = _sglang_accept_rows_from_verify_result(result)
    return None if rows is None else _clamp_to_verify_rows(rows)


def _append_sglang_decode_hidden_states(
    req, logits_output, result, req_index: int, hidden_state_offset: int, batch=None
) -> int:
    if batch is not None and not _sglang_batch_requests_hidden_states(batch):
        accept_rows_per_req = _sglang_decode_accept_rows_per_req(result)
        if accept_rows_per_req is not None and req_index < len(accept_rows_per_req):
            try:
                return hidden_state_offset + max(int(accept_rows_per_req[req_index]), 0)
            except (TypeError, ValueError):
                return hidden_state_offset
        return hidden_state_offset

    _filter_sglang_drafter_last_hidden_from_verify_result(
        logits_output,
        result,
        positions=getattr(result, "positions", None),
        batch=batch,
    )
    hidden_states = getattr(
        logits_output,
        "_verl_drafter_base_hidden_filtered_by_accept_indices_tensor",
        None,
    )
    if not _is_torch_tensor(hidden_states):
        hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        if getattr(req, "return_hidden_states", False) and not getattr(
            req, "_verl_logged_missing_decode_hidden_states", False
        ):
            logger.warning(
                "SGLang did not return decode hidden states for drafter collection: "
                "request_dflash_aux=%s logits_output_type=%s",
                _sglang_req_requests_dflash_aux_hidden(req),
                type(logits_output).__name__,
            )
            setattr(req, "_verl_logged_missing_decode_hidden_states", True)
        return hidden_state_offset
    if (
        _sglang_req_requests_last_hidden_for_drafter(req)
        and _is_torch_tensor(
            getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
        )
        and not bool(
            getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, False)
        )
    ):
        filter_summary = getattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
        )
        raise RuntimeError(
            "SGLang EAGLE verify did not expose accepted indices for final target hidden states. "
            f"filter_summary={filter_summary}. This would train on unfiltered tree-order last_hidden rows."
        )
    hidden_positions = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
    )

    accept_rows_per_req = _sglang_decode_accept_rows_per_req(result)
    if (
        accept_rows_per_req is not None
        and req_index < len(accept_rows_per_req)
        and _is_torch_tensor(hidden_states)
    ):
        rows = accept_rows_per_req[req_index]

        if getattr(logits_output, "_verl_dflash_aux_hidden_states", False):
            end = hidden_state_offset + rows
            total_hidden = int(hidden_states.shape[0])
            if hidden_states.dim() >= 2 and end <= total_hidden:
                if not getattr(req, "return_hidden_states", False):
                    return end
                chunk = hidden_states[hidden_state_offset:end]
                chunk_rows = _sglang_hidden_chunk_rows(chunk)
                chunk_positions = (
                    hidden_positions[hidden_state_offset:end]
                    if _is_torch_tensor(hidden_positions)
                    else None
                )
                _append_sglang_hidden_state_chunk_with_budget(
                    req,
                    chunk,
                    position_start=(
                        None
                        if not _is_torch_tensor(hidden_positions)
                        else int(
                            hidden_positions[hidden_state_offset].detach().cpu().item()
                        )
                    ),
                    positions=chunk_positions,
                    extra_metadata=_sglang_hidden_debug_metadata_for_req(
                        req,
                        logits_output,
                        hidden_state_offset,
                        end,
                        chunk_rows,
                        positions=chunk_positions,
                    ),
                    batch=batch,
                )
                return end
            if getattr(req, "return_hidden_states", False):
                raise RuntimeError(
                    "SGLang DFlash verify hidden states are incomplete for accepted tokens: "
                    f"shape={tuple(hidden_states.shape)}, req_index={req_index}, "
                    f"offset={hidden_state_offset}, required_rows={rows}."
                )

        total_hidden = int(hidden_states.shape[0])
        expected_rows = sum(accept_rows_per_req)
        available_rows = max(total_hidden - hidden_state_offset, 0)
        append_rows = min(rows, available_rows)
        end = hidden_state_offset + append_rows
        filter_summary = getattr(
            logits_output, _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR, None
        )
        positions_truncated = isinstance(filter_summary, dict) and bool(
            filter_summary.get("positions_truncated", False)
        )
        if hidden_states.dim() >= 2 and append_rows > 0:
            if not getattr(req, "return_hidden_states", False):
                return end
            chunk = hidden_states[hidden_state_offset:end]
            chunk = _sglang_concat_last_hidden_for_drafter(
                req,
                logits_output,
                chunk,
                _slice_sglang_drafter_last_hidden_output(
                    logits_output, slice(hidden_state_offset, end)
                ),
            )
            chunk_rows = _sglang_hidden_chunk_rows(chunk)
            chunk_positions = (
                hidden_positions[hidden_state_offset:end]
                if _is_torch_tensor(hidden_positions)
                else None
            )
            _append_sglang_hidden_state_chunk_with_budget(
                req,
                chunk,
                position_start=(
                    None
                    if not _is_torch_tensor(hidden_positions)
                    else int(
                        hidden_positions[hidden_state_offset].detach().cpu().item()
                    )
                ),
                positions=chunk_positions,
                extra_metadata=_sglang_hidden_debug_metadata_for_req(
                    req,
                    logits_output,
                    hidden_state_offset,
                    end,
                    chunk_rows,
                    positions=chunk_positions,
                ),
                batch=batch,
            )
            return end

        if getattr(req, "return_hidden_states", False) and not positions_truncated:
            raise RuntimeError(
                "SGLang EAGLE verify hidden states are incomplete for accepted tokens: "
                f"shape={tuple(hidden_states.shape)}, req_index={req_index}, "
                f"offset={hidden_state_offset}, required_rows={rows}, expected_total_rows={expected_rows}."
            )
        return hidden_state_offset

    if not getattr(req, "return_hidden_states", False):
        return hidden_state_offset
    if _is_torch_tensor(hidden_states):
        chunk = hidden_states[req_index]
        chunk = _sglang_concat_last_hidden_for_drafter(
            req,
            logits_output,
            chunk,
            _slice_sglang_drafter_last_hidden_output(logits_output, req_index),
        )
        chunk_rows = _sglang_hidden_chunk_rows(chunk)
        chunk_positions = (
            hidden_positions[req_index] if _is_torch_tensor(hidden_positions) else None
        )
        _append_sglang_hidden_state_chunk_with_budget(
            req,
            chunk,
            position_start=(
                None
                if not _is_torch_tensor(hidden_positions)
                else int(hidden_positions[req_index].detach().cpu().item())
            ),
            positions=chunk_positions,
            extra_metadata=_sglang_hidden_debug_metadata_for_req(
                req,
                logits_output,
                req_index,
                req_index + 1,
                chunk_rows,
                positions=chunk_positions,
            ),
            batch=batch,
        )
    else:
        chunk = hidden_states[req_index]
        chunk_rows = _sglang_hidden_chunk_rows(chunk)
        _append_sglang_hidden_state_chunk_with_budget(
            req,
            chunk,
            position_start=None,
            extra_metadata=_sglang_hidden_debug_metadata_for_req(
                req,
                logits_output,
                req_index,
                req_index + 1,
                chunk_rows,
            ),
            batch=batch,
        )
    return hidden_state_offset


def _sglang_batch_requests_hidden_states(batch) -> bool:
    return any(
        bool(getattr(req, "return_hidden_states", False))
        for req in getattr(batch, "reqs", []) or []
    )


def _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info) -> None:
    if not _sglang_batch_requests_hidden_states(batch):
        return
    try:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Cannot import SGLang CaptureHiddenMode for EAGLE hidden-state patch: %s",
            exc,
        )
        return
    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL


def _sglang_hidden_state_rows(hidden_states) -> int:
    if not torch.is_tensor(hidden_states):
        try:
            return len(hidden_states)
        except TypeError:
            return 0
    if hidden_states.dim() == 0:
        return 1
    if hidden_states.dim() >= 3:
        return int(hidden_states.shape[0]) * int(hidden_states.shape[1])
    return int(hidden_states.shape[0])


def _sglang_eagle_verify_expected_hidden_rows(batch, spec_info) -> int:
    batch_size = len(getattr(batch, "reqs", []) or [])
    draft_token_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
    return batch_size * draft_token_num


def _sglang_eagle_verify_hidden_states_incomplete(
    batch, spec_info, logits_output
) -> bool:
    if not _sglang_batch_requests_hidden_states(batch):
        return False
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    if expected_rows <= 0:
        return False
    hidden_states = getattr(logits_output, "hidden_states", None)
    return (
        hidden_states is None
        or _sglang_hidden_state_rows(hidden_states) < expected_rows
    )


def _sglang_batch_requests_last_hidden_for_drafter(batch) -> bool:
    return any(
        _sglang_req_requests_last_hidden_for_drafter(req)
        for req in getattr(batch, "reqs", []) or []
    )


def _sglang_eagle_verify_last_hidden_incomplete(
    batch, spec_info, logits_output
) -> bool:
    if not _sglang_batch_requests_last_hidden_for_drafter(batch):
        return False
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    if bool(getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR, False)):
        hidden_states = getattr(logits_output, "hidden_states", None)
        return hidden_states is None or (
            expected_rows > 0
            and _sglang_hidden_state_rows(hidden_states) < expected_rows
        )
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    if last_hidden_states is None:
        return True
    return (
        expected_rows > 0
        and _sglang_hidden_state_rows(last_hidden_states) < expected_rows
    )


def _rerun_sglang_eagle_verify_without_graph(worker, model_worker_batch):
    target_worker = getattr(worker, "target_worker", None)
    model_runner = getattr(target_worker, "model_runner", None)
    graph_runner = getattr(model_runner, "graph_runner", None)
    try:
        if model_runner is not None:
            model_runner.graph_runner = None
        return target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
    finally:
        if model_runner is not None:
            model_runner.graph_runner = graph_runner


def _validate_sglang_eagle_verify_hidden_states(
    batch, spec_info, logits_output
) -> None:
    if not _sglang_eagle_verify_hidden_states_incomplete(
        batch, spec_info, logits_output
    ):
        return
    hidden_states = getattr(logits_output, "hidden_states", None)
    shape = tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    actual_rows = _sglang_hidden_state_rows(hidden_states)
    raise RuntimeError(
        "SGLang EAGLE verify did not return full hidden states for drafter training: "
        f"actual_rows={actual_rows}, expected_rows={expected_rows}, shape={shape}. "
        "This would train on partial/incorrect hidden alignment."
    )


def _validate_sglang_eagle_verify_last_hidden(batch, spec_info, logits_output) -> None:
    if not _sglang_eagle_verify_last_hidden_incomplete(batch, spec_info, logits_output):
        return
    last_hidden_states = getattr(
        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None
    )
    shape = (
        tuple(last_hidden_states.shape) if torch.is_tensor(last_hidden_states) else None
    )
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    actual_rows = _sglang_hidden_state_rows(last_hidden_states)
    select_summary = getattr(
        logits_output, "_verl_drafter_last_hidden_select_summary", None
    )
    raise RuntimeError(
        "SGLang EAGLE verify did not return final target hidden states for drafter training: "
        f"actual_rows={actual_rows}, expected_rows={expected_rows}, shape={shape}, "
        f"select_summary={select_summary}. "
        "This would train EAGLE3 against logits computed from a different hidden stream."
    )


def _wrap_sglang_eagle_verify_last_hidden_filter(method):
    if getattr(method, "_verl_patched_drafter_last_hidden_filter", False):
        return method

    @wraps(method)
    def patched_verify_last_hidden_filter(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        try:
            batch = (
                kwargs.get("batch")
                if "batch" in kwargs
                else (args[0] if args else None)
            )
            logits_output = (
                result[0]
                if isinstance(result, (list, tuple))
                else getattr(result, "logits_output", None)
            )
            if logits_output is None:
                return result
            should_filter_last_hidden = _sglang_should_filter_eagle_verify_last_hidden(
                batch
            )
            if not should_filter_last_hidden:
                return result
            verify_output = (
                result[1]
                if isinstance(result, (list, tuple)) and len(result) > 1
                else result
            )
            spec_info = getattr(self, "spec_info", None)
            accepted_indices = _find_sglang_accepted_indices(
                verify_output, result, spec_info, self
            )
            positions = _find_sglang_verify_positions(
                spec_info, verify_output, result, self
            )
            if accepted_indices is not None:
                _filter_sglang_drafter_last_hidden_output(
                    logits_output,
                    accepted_indices,
                    positions=positions,
                    batch=batch,
                    verify_result=verify_output,
                    filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),
                )
            else:
                _filter_sglang_drafter_last_hidden_from_verify_result(
                    logits_output,
                    verify_output,
                    positions=positions,
                    batch=batch,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to apply SGLang drafter last-hidden verify filter: %s", exc
            )
        return result

    patched_verify_last_hidden_filter._verl_patched_drafter_last_hidden_filter = True
    return patched_verify_last_hidden_filter


def _wrap_sglang_eagle_verify_input_last_hidden_filter(method):
    if getattr(method, "_verl_patched_drafter_last_hidden_filter", False):
        return method

    @wraps(method)
    def patched_verify_input_last_hidden_filter(
        self, batch, logits_output, *args, **kwargs
    ):
        result = method(self, batch, logits_output, *args, **kwargs)
        try:
            accepted_indices = _find_sglang_accepted_indices(result, self, batch)
            positions = _find_sglang_verify_positions(self, result, batch)
            should_filter_last_hidden = _sglang_should_filter_eagle_verify_last_hidden(
                batch
            )
            if not should_filter_last_hidden:
                return result
            if accepted_indices is not None:
                _filter_sglang_drafter_last_hidden_output(
                    logits_output,
                    accepted_indices,
                    positions=positions,
                    batch=batch,
                    verify_result=result,
                    filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),
                )
            else:
                _filter_sglang_drafter_last_hidden_from_verify_result(
                    logits_output,
                    result,
                    positions=positions,
                    batch=batch,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to apply SGLang drafter last-hidden verify-input filter: %s",
                exc,
            )
        return result

    patched_verify_input_last_hidden_filter._verl_patched_drafter_last_hidden_filter = (
        True
    )
    return patched_verify_input_last_hidden_filter


def _wrap_sglang_eagle_forward_generation_last_hidden_materialize(method):
    if getattr(method, "_verl_patched_drafter_last_hidden_materialize", False):
        return method

    @wraps(method)
    def patched_forward_generation_last_hidden_materialize(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        try:
            _materialize_sglang_drafter_last_hidden_output(
                getattr(result, "logits_output", None),
                stage="eagle_forward_generation_result",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to materialize SGLang drafter final hidden at EAGLE output boundary: %s",
                exc,
            )
        return result

    patched_forward_generation_last_hidden_materialize._verl_patched_drafter_last_hidden_materialize = True
    return patched_forward_generation_last_hidden_materialize


def _patch_sglang_eagle_verify_full_hidden_source(source: str) -> str | None:
    source = textwrap.dedent(source)
    patched_source = source

    old_prepare = "        spec_info.prepare_for_verify(batch, self.page_size)\n"
    new_prepare = (
        "        spec_info.prepare_for_verify(batch, self.page_size)\n"
        "        _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info)\n"
    )
    if old_prepare in patched_source:
        patched_source = patched_source.replace(old_prepare, new_prepare, 1)
    else:
        return None

    old_forward = """        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
"""
    new_forward = """        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        if _sglang_eagle_verify_hidden_states_incomplete(
            batch, spec_info, logits_output
        ) or _sglang_eagle_verify_last_hidden_incomplete(batch, spec_info, logits_output):
            logger.warning(
                "SGLang EAGLE verify returned incomplete hidden states; rerunning without graph for full hidden output."
            )
            batch_result = _rerun_sglang_eagle_verify_without_graph(self, model_worker_batch)
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
        _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output)
        _validate_sglang_eagle_verify_last_hidden(batch, spec_info, logits_output)
"""
    if old_forward in patched_source:
        patched_source = patched_source.replace(old_forward, new_forward, 1)
    else:
        return None

    hidden_filter_replacements = (
        (
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accept_indices
            ]
""",
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accept_indices
            ]
            _filter_sglang_drafter_last_hidden_output(
                logits_output,
                res.accept_indices,
                positions=getattr(spec_info, "positions", None),
                batch=batch,
                verify_result=res,
                filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),
            )
""",
        ),
        (
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accepted_indices
            ]
""",
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accepted_indices
            ]
            _filter_sglang_drafter_last_hidden_output(
                logits_output,
                res.accepted_indices,
                positions=getattr(spec_info, "positions", None),
                batch=batch,
                verify_result=res,
                filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),
            )
""",
        ),
        (
            "        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]\n",
            (
                "        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]\n"
                "        _filter_sglang_drafter_last_hidden_output(\n"
                "            logits_output,\n"
                "            res.accept_indices,\n"
                '            positions=getattr(spec_info, "positions", None),\n'
                "            batch=batch,\n"
                "            verify_result=res,\n"
                "            filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),\n"
                "        )\n"
            ),
        ),
        (
            "        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]\n",
            (
                "        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]\n"
                "        _filter_sglang_drafter_last_hidden_output(\n"
                "            logits_output,\n"
                "            res.accepted_indices,\n"
                '            positions=getattr(spec_info, "positions", None),\n'
                "            batch=batch,\n"
                "            verify_result=res,\n"
                "            filter_base_hidden_states=not _sglang_needs_eagle_legacy_alignment_patch(),\n"
                "        )\n"
            ),
        ),
    )
    for old_hidden_filter, new_hidden_filter in hidden_filter_replacements:
        if old_hidden_filter in patched_source:
            patched_source = patched_source.replace(
                old_hidden_filter, new_hidden_filter, 1
            )
            break

    return patched_source


def _make_sglang_eagle_verify_full_hidden_patch(original_method):
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

    patched_source = _patch_sglang_eagle_verify_full_hidden_source(source)
    if patched_source is None:
        return None

    globals_dict = original_method.__globals__
    globals_dict["logger"] = logger
    globals_dict["_ensure_sglang_eagle_verify_full_hidden_mode"] = (
        _ensure_sglang_eagle_verify_full_hidden_mode
    )
    globals_dict["_sglang_eagle_verify_hidden_states_incomplete"] = (
        _sglang_eagle_verify_hidden_states_incomplete
    )
    globals_dict["_sglang_eagle_verify_last_hidden_incomplete"] = (
        _sglang_eagle_verify_last_hidden_incomplete
    )
    globals_dict["_rerun_sglang_eagle_verify_without_graph"] = (
        _rerun_sglang_eagle_verify_without_graph
    )
    globals_dict["_validate_sglang_eagle_verify_hidden_states"] = (
        _validate_sglang_eagle_verify_hidden_states
    )
    globals_dict["_validate_sglang_eagle_verify_last_hidden"] = (
        _validate_sglang_eagle_verify_last_hidden
    )
    globals_dict["_filter_sglang_drafter_last_hidden_output"] = (
        _filter_sglang_drafter_last_hidden_output
    )
    globals_dict["_sglang_needs_eagle_legacy_alignment_patch"] = (
        _sglang_needs_eagle_legacy_alignment_patch
    )
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_eagle_verify_full_hidden_states = True
    return _wrap_sglang_eagle_verify_last_hidden_filter(patched_method)


def patch_sglang_eagle_verify_hidden_states_full() -> None:
    """Force SGLang EAGLE v1 verify to return full per-token hidden states."""
    global _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED
    if _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED:
        return

    verify_input_patched = False
    try:
        eagle_info_module = importlib.import_module("sglang.srt.speculative.eagle_info")
        verify_input_cls = getattr(eagle_info_module, "EagleVerifyInput")
        original_verify_input = getattr(verify_input_cls, "verify", None)
        if original_verify_input is not None:
            wrapped_verify_input = _wrap_sglang_eagle_verify_input_last_hidden_filter(
                original_verify_input
            )
            if wrapped_verify_input is not original_verify_input:
                setattr(verify_input_cls, "verify", wrapped_verify_input)
            verify_input_patched = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang EagleVerifyInput last-hidden filter patch: %s", exc)

    targets = (
        ("sglang.srt.speculative.eagle_worker", "EAGLEWorker"),
        ("sglang.srt.speculative.multi_layer_eagle_worker", "MultiLayerEagleWorker"),
    )
    patched_targets = []
    for module_name, class_name in targets:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            original_method = getattr(worker_cls, "verify", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skip SGLang EAGLE full hidden-state patch for %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            continue
        if original_method is None:
            continue
        original_forward_generation = getattr(
            worker_cls, "forward_batch_generation", None
        )
        if original_forward_generation is not None and not getattr(
            original_forward_generation,
            "_verl_patched_drafter_last_hidden_materialize",
            False,
        ):
            setattr(
                worker_cls,
                "forward_batch_generation",
                _wrap_sglang_eagle_forward_generation_last_hidden_materialize(
                    original_forward_generation
                ),
            )
            patched_targets.append(
                f"{module_name}.{class_name}.forward_batch_generation[last-hidden-materialize]"
            )
        if getattr(
            original_method, "_verl_patched_eagle_verify_full_hidden_states", False
        ):
            wrapped_method = _wrap_sglang_eagle_verify_last_hidden_filter(
                original_method
            )
            if wrapped_method is not original_method:
                setattr(worker_cls, "verify", wrapped_method)
            patched_targets.append(f"{module_name}.{class_name}.verify")
            continue
        patched_method = _make_sglang_eagle_verify_full_hidden_patch(original_method)
        if patched_method is None:
            wrapped_method = _wrap_sglang_eagle_verify_last_hidden_filter(
                original_method
            )
            if wrapped_method is original_method:
                logger.debug(
                    "Skip SGLang EAGLE full hidden-state patch for %s.%s",
                    module_name,
                    class_name,
                )
                continue
            setattr(worker_cls, "verify", wrapped_method)
            patched_targets.append(
                f"{module_name}.{class_name}.verify[last-hidden-filter]"
            )
            logger.warning(
                "SGLang EAGLE full hidden-state source patch skipped for %s.%s; "
                "installed last-hidden accepted-index filter only.",
                module_name,
                class_name,
            )
            continue
        setattr(worker_cls, "verify", patched_method)
        patched_targets.append(f"{module_name}.{class_name}.verify")

    if patched_targets or verify_input_patched:
        _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = True
        if verify_input_patched:
            patched_targets.append(
                "sglang.srt.speculative.eagle_info.EagleVerifyInput.verify[last-hidden-filter]"
            )
        logger.warning(
            "Patched SGLang EAGLE verify full hidden states for %s",
            ", ".join(patched_targets),
        )


_SGLANG_HIDDEN_STATES_LIST_OUTPUT_PATTERN = re.compile(
    r"\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\)",
)
_SGLANG_DECODE_REQUEST_LOOP_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)for i, (?:"
    r"\(req, next_token_id\) in enumerate\(\s*zip\(batch\.reqs,\s*next_token_ids\)\s*\)"
    r"|req in enumerate\(batch\.reqs\)"
    r"):\r?\n",
)
_SGLANG_DECODE_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s*(?:\(\s*)?req\.return_hidden_states\s+"
    r"and\s+logits_output\.hidden_states\s+is\s+not\s+None\s*(?:\))?\s*:\r?\n"
    r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"^(?P=indent)[ \t]+\)\r?\n",
)
_SGLANG_PREFILL_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s*(?:\(\s*)?req\.return_hidden_states\s+"
    r"and\s+logits_output\.hidden_states\s+is\s+not\s+None\s*(?:\))?\s*:\r?\n"
    r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"(?:\.detach\(\)\.to\(\"cpu\",\s*copy=True\)|\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\))\r?\n"
    r"(?P=indent)[ \t]+\)\r?\n",
)
_SGLANG_STREAM_HIDDEN_STATES_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s+req\.return_hidden_states\s*:\r?\n"
    r"(?P=indent)[ \t]+if\s+output_hidden_states\s+is\s+None\s*:\r?\n"
    r"(?P=indent)[ \t]+[ \t]+output_hidden_states\s*=\s*\[\]\r?\n"
    r"(?P=indent)[ \t]+output_hidden_states\.append\(req\.hidden_states\)\r?\n",
)


def _replace_sglang_hidden_states_list_output(source: str) -> tuple[str, int]:
    return _SGLANG_HIDDEN_STATES_LIST_OUTPUT_PATTERN.subn(
        '.detach().to("cpu", copy=True)',
        source,
    )


def _render_sglang_decode_hidden_states_append(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}hidden_state_offset = _append_sglang_decode_hidden_states(\n"
        f"{indent}    req,\n"
        f"{indent}    logits_output,\n"
        f"{indent}    result,\n"
        f"{indent}    i,\n"
        f"{indent}    hidden_state_offset,\n"
        f"{indent}    batch,\n"
        f"{indent})\n"
    )


def _render_sglang_prefill_hidden_states_append(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}hidden_state_offset = _append_sglang_prefill_hidden_states(\n"
        f"{indent}    req,\n"
        f"{indent}    logits_output,\n"
        f"{indent}    hidden_state_offset,\n"
        f"{indent}    (\n"
        f"{indent}        extend_input_len_per_req[i]\n"
        f"{indent}        if extend_input_len_per_req is not None\n"
        f"{indent}        else len(req.origin_input_ids)\n"
        f"{indent}    ),\n"
        f"{indent}    batch,\n"
        f"{indent})\n"
    )


def _render_sglang_stream_hidden_states(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}if _sglang_req_should_stream_hidden_states(req):\n"
        f"{indent}    if output_hidden_states is None:\n"
        f"{indent}        output_hidden_states = [[] for _ in range(len(rids) - 1)]\n"
        f"{indent}    output_hidden_states.append(req.hidden_states)\n"
        f"{indent}elif output_hidden_states is not None:\n"
        f"{indent}    output_hidden_states.append([])\n"
    )


def _insert_sglang_decode_hidden_state_offset(source: str) -> str | None:
    if re.search(
        r"(?ms)^def\s+process_batch_result_decode\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:\r?\n"
        r"[ \t]+hidden_state_offset = 0\s*(?:#.*)?\r?\n",
        source,
    ):
        return source

    patched_source, function_count = re.subn(
        r"(?ms)^(?P<header>def\s+process_batch_result_decode\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:\r?\n)",
        lambda match: f"{match.group('header')}    hidden_state_offset = 0\n",
        source,
        count=1,
    )
    if function_count <= 0:
        return None
    return patched_source


def _patch_sglang_decode_hidden_states_source(source: str) -> str | None:
    patched_source, hidden_block_count = (
        _SGLANG_DECODE_HIDDEN_STATES_APPEND_PATTERN.subn(
            _render_sglang_decode_hidden_states_append,
            source,
        )
    )
    if hidden_block_count <= 0:
        return None
    return _insert_sglang_decode_hidden_state_offset(patched_source)


def _patch_sglang_prefill_hidden_states_source(source: str) -> str | None:
    patched_source, hidden_block_count = (
        _SGLANG_PREFILL_HIDDEN_STATES_APPEND_PATTERN.subn(
            _render_sglang_prefill_hidden_states_append,
            source,
            count=1,
        )
    )
    if hidden_block_count <= 0:
        return None
    return patched_source


def _patch_sglang_stream_hidden_states_source(source: str) -> str | None:
    patched_source, hidden_stream_count = _SGLANG_STREAM_HIDDEN_STATES_PATTERN.subn(
        _render_sglang_stream_hidden_states,
        source,
        count=1,
    )
    if hidden_stream_count <= 0:
        return None
    return patched_source


def _make_sglang_hidden_states_tensor_output_patch(original_method):
    """Patch SGLang output processors to keep hidden-state chunks as CPU tensors.

    SGLang 0.5.9 and 0.5.10 both append hidden states with
    `.cpu().clone().tolist()`. The `.tolist()` conversion serializes every
    hidden value through Python objects and dominates rollout latency when
    drafter collection is enabled. Keeping CPU tensors preserves the existing
    ownership/lifetime behavior while avoiding Python list materialization.
    """
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source, conversion_count = _replace_sglang_hidden_states_list_output(source)
    if original_method.__name__ == "process_batch_result_prefill":
        patched_prefill_source = _patch_sglang_prefill_hidden_states_source(
            patched_source
        )
        if patched_prefill_source is None:
            logger.warning(
                "Skip SGLang prefill hidden-state window patch for %s: hidden append block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_prefill_source
    elif original_method.__name__ == "process_batch_result_decode":
        patched_decode_source = _patch_sglang_decode_hidden_states_source(
            patched_source
        )
        if patched_decode_source is None:
            logger.warning(
                "Skip SGLang decode hidden-state full-output patch for %s: hidden append block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_decode_source
    elif original_method.__name__ == "stream_output_generation":
        patched_stream_source = _patch_sglang_stream_hidden_states_source(
            patched_source
        )
        if patched_stream_source is None:
            logger.warning(
                "Skip SGLang stream hidden-state final-output patch for %s: hidden stream block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_stream_source
    elif conversion_count <= 0:
        return None

    if patched_source == source:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_append_sglang_prefill_hidden_states"] = (
        _append_sglang_prefill_hidden_states
    )
    globals_dict["_append_sglang_decode_hidden_states"] = (
        _append_sglang_decode_hidden_states
    )
    globals_dict["_sglang_req_should_stream_hidden_states"] = (
        _sglang_req_should_stream_hidden_states
    )
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_hidden_states_tensor_output = True
    return patched_method


def _make_sglang_drafter_output_forward_patch(
    original_method,
    *,
    enable_raw_top_logprobs: bool,
    enable_last_hidden: bool,
):
    @wraps(original_method)
    def patched_logits_processor_forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        return_last_hidden = enable_last_hidden and (
            _sglang_drafter_return_last_hidden_enabled()
            or _sglang_forward_batch_requests_last_hidden_for_drafter(logits_metadata)
        )
        return_dflash_aux_hidden = _sglang_forward_batch_requests_dflash_aux_hidden(
            logits_metadata
        )

        output = original_method(
            self,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm,
        )
        if return_dflash_aux_hidden:
            dflash_hidden_states = _normalize_sglang_dflash_aux_hidden_states(
                aux_hidden_states
            )
            if dflash_hidden_states is None:
                logger.warning(
                    "SGLang DFlash aux hidden states requested but unavailable: "
                    "aux_type=%s output_hidden_type=%s",
                    type(aux_hidden_states).__name__,
                    type(getattr(output, "hidden_states", None)).__name__,
                )
            else:
                output.hidden_states = dflash_hidden_states
                setattr(output, "_verl_dflash_aux_hidden_states", True)
        raw_top_logprobs_requested = (
            enable_raw_top_logprobs
            and _sglang_raw_top_logprobs_enabled(logits_metadata)
        )
        if raw_top_logprobs_requested and return_last_hidden:
            global _SGLANG_DRAFTER_DECOUPLED_FORWARD_LOG_COUNT
            if _SGLANG_DRAFTER_DECOUPLED_FORWARD_LOG_COUNT < 8:
                _SGLANG_DRAFTER_DECOUPLED_FORWARD_LOG_COUNT += 1
                logger.warning(
                    "SGLang drafter output requested raw top-logprobs and last hidden; "
                    "the payload paths are handled independently."
                )
        if raw_top_logprobs_requested:
            _attach_sglang_raw_top_logprobs(output, logits_metadata)
        if return_last_hidden and hidden_states is not None:
            if getattr(output, "hidden_states", None) is not None:
                _attach_sglang_lm_head_fingerprint(output, lm_head)
                _attach_sglang_last_hidden_logprob_check(
                    self, output, hidden_states, lm_head, logits_metadata
                )
            last_hidden_for_drafter = _select_sglang_last_hidden_for_drafter(
                self,
                output,
                input_ids,
                hidden_states,
                hidden_states_before_norm,
                aux_hidden_states,
                logits_metadata,
            )
            setattr(
                output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, last_hidden_for_drafter
            )
            setattr(output, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, False)
        return output

    patched_logits_processor_forward._verl_patched_drafter_output = True
    patched_logits_processor_forward._verl_patched_drafter_raw_top_logprobs = (
        enable_raw_top_logprobs
    )
    patched_logits_processor_forward._verl_patched_drafter_last_hidden_output = (
        enable_last_hidden
    )
    return patched_logits_processor_forward


def _copy_sglang_drafter_last_hidden_output(src, dst, index) -> None:
    if dst is None:
        return
    last_hidden_states = getattr(src, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
    if last_hidden_states is not None:
        setattr(dst, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, last_hidden_states[index])
        last_hidden_positions = getattr(
            src, _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR, None
        )
        if _is_torch_tensor(last_hidden_positions):
            setattr(
                dst,
                _VERL_DRAFTER_LAST_HIDDEN_POSITIONS_ATTR,
                last_hidden_positions[index],
            )
        setattr(
            dst,
            _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR,
            bool(getattr(src, _VERL_DRAFTER_LAST_HIDDEN_FILTERED_ATTR, False)),
        )
    for attr_name in (
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_TOP_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RECOMPUTED_AT_SGLANG_TOP_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_SGLANG_TOP_LOGPROBS_ATTR,
    ):
        value = getattr(src, attr_name, None)
        if _is_torch_tensor(value):
            try:
                setattr(dst, attr_name, value[index])
            except Exception:  # noqa: BLE001
                setattr(dst, attr_name, value)
    for attr_name in (
        "_verl_drafter_last_hidden_select_summary",
        "_verl_drafter_lh_check_lm_head_fingerprint",
        _VERL_DRAFTER_LH_CHECK_SUMMARY_ATTR,
        _VERL_DRAFTER_LAST_HIDDEN_FILTER_SUMMARY_ATTR,
        _VERL_DRAFTER_LAST_HIDDEN_MATERIALIZED_ATTR,
    ):
        if hasattr(src, attr_name):
            setattr(dst, attr_name, getattr(src, attr_name))


def _copy_sglang_raw_top_logprobs_output(src, dst, index) -> None:
    """Copy raw teacher metadata without touching last-hidden fields."""
    if dst is None:
        return
    for attr_name in (
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR,
    ):
        value = getattr(src, attr_name, None)
        if _is_torch_tensor(value):
            try:
                setattr(dst, attr_name, value[index])
            except Exception:  # noqa: BLE001
                setattr(dst, attr_name, value)


def _clear_sglang_raw_top_logprobs(logits_output) -> None:
    if logits_output is None:
        return
    for attr_name in (
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR,
        _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR,
    ):
        try:
            if hasattr(logits_output, attr_name):
                delattr(logits_output, attr_name)
        except Exception:  # noqa: BLE001
            setattr(logits_output, attr_name, None)


def _raw_top_logprobs_metadata_complete(logits_output) -> bool:
    next_token_logits = getattr(logits_output, "next_token_logits", None)
    raw_topk_ids = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_IDS_ATTR, None
    )
    raw_topk_logprobs = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_LOGPROBS_ATTR, None
    )
    raw_positions = getattr(
        logits_output, _VERL_DRAFTER_LH_CHECK_RAW_TOPK_POSITIONS_ATTR, None
    )
    if not all(
        _is_torch_tensor(value)
        for value in (next_token_logits, raw_topk_ids, raw_topk_logprobs, raw_positions)
    ):
        return False
    rows = int(next_token_logits.shape[0])
    return (
        next_token_logits.dim() >= 2
        and raw_topk_ids.dim() == 2
        and raw_topk_logprobs.dim() == 2
        and int(raw_topk_ids.shape[0]) == rows
        and int(raw_topk_logprobs.shape[0]) == rows
        and int(raw_positions.numel()) == rows
    )


def _ensure_sglang_raw_top_logprobs_for_replay(
    logits_output, forward_batch, source_output=None
) -> bool:
    if logits_output is None or not _sglang_raw_top_logprobs_enabled(forward_batch):
        return False
    if _raw_top_logprobs_metadata_complete(logits_output):
        return True
    if source_output is not None and source_output is not logits_output:
        try:
            rows = int(logits_output.next_token_logits.shape[0])
            _copy_sglang_raw_top_logprobs_output(
                source_output, logits_output, slice(0, rows)
            )
        except Exception:  # noqa: BLE001
            pass
        if _raw_top_logprobs_metadata_complete(logits_output):
            return True
    return _attach_sglang_raw_top_logprobs(logits_output, forward_batch)


def _sglang_graph_replay_output_buffer(runner, original_method, forward_batch=None):
    keys = []
    variant_label = None
    resolve_variant = getattr(runner, "_resolve_lora_variant", None)
    if callable(resolve_variant) and forward_batch is not None:
        try:
            variant_label = resolve_variant(forward_batch)
        except Exception:  # noqa: BLE001
            variant_label = None
    stream_idx = None
    if getattr(runner, "enable_pdmux", False):
        get_current_stream_idx = original_method.__globals__.get(
            "get_current_stream_idx"
        )
        if callable(get_current_stream_idx):
            try:
                stream_idx = get_current_stream_idx()
            except Exception:  # noqa: BLE001
                stream_idx = None
    make_graph_key = getattr(runner, "_make_graph_key", None)
    if callable(make_graph_key):
        try:
            keys.append(make_graph_key(runner.bs, stream_idx, variant_label))
        except TypeError:
            try:
                keys.append(make_graph_key(runner.bs, stream_idx))
            except TypeError:
                keys.append(make_graph_key(runner.bs))
        except Exception:  # noqa: BLE001
            pass
    if stream_idx is not None:
        base_key = f"{stream_idx}_{runner.bs}"
        if variant_label is not None:
            keys.append(f"{variant_label}_{base_key}")
        keys.append(base_key)
    if variant_label is not None:
        keys.append(f"{variant_label}_{runner.bs}")
    keys.append(runner.bs)
    for key in keys:
        try:
            return runner.output_buffers[key]
        except Exception:  # noqa: BLE001
            continue
    return None


def _sglang_graph_replay_raw_num_tokens(runner) -> int:
    for attr_name in ("raw_num_token", "raw_num_tokens"):
        value = getattr(runner, attr_name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


def _make_sglang_drafter_graph_replay_patch(
    original_method,
    *,
    enable_raw_top_logprobs: bool,
    enable_last_hidden: bool,
):
    @wraps(original_method)
    def patched_graph_replay(self, *args, **kwargs):
        result = original_method(self, *args, **kwargs)
        forward_batch = args[0] if args else kwargs.get("forward_batch")
        output = _sglang_graph_replay_output_buffer(
            self, original_method, forward_batch
        )
        copy_last_hidden = (
            enable_last_hidden
            and output is not None
            and _sglang_forward_batch_requests_last_hidden_for_drafter(forward_batch)
        )
        if copy_last_hidden:
            _copy_sglang_drafter_last_hidden_output(
                output,
                result,
                slice(0, _sglang_graph_replay_raw_num_tokens(self)),
            )
        if enable_raw_top_logprobs:
            _ensure_sglang_raw_top_logprobs_for_replay(result, forward_batch, output)
        if copy_last_hidden:
            _attach_sglang_last_hidden_logprob_check_from_graph_runner(
                self, result, forward_batch
            )
        return result

    patched_graph_replay._verl_patched_drafter_output = True
    patched_graph_replay._verl_patched_drafter_raw_top_logprobs = (
        enable_raw_top_logprobs
    )
    patched_graph_replay._verl_patched_drafter_last_hidden_output = enable_last_hidden
    return patched_graph_replay


def _make_sglang_drafter_inline_graph_replay_patch(
    original_method,
    *,
    enable_raw_top_logprobs: bool,
    enable_last_hidden: bool,
):
    source = textwrap.dedent(inspect.getsource(original_method))
    marker = "return LogitsProcessorOutput("
    marker_index = source.find(marker)
    block_end = None
    while marker_index >= 0:
        depth = 0
        for offset, char in enumerate(source[marker_index:]):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    block_end = marker_index + offset + 1
                    break
        if block_end is None:
            break
        old_block = source[marker_index:block_end]
        if (
            "output.next_token_logits" in old_block
            and "output.hidden_states" in old_block
        ):
            indent = source[source.rfind("\n", 0, marker_index) + 1 : marker_index]
            new_block = (
                old_block.replace(
                    marker,
                    "_verl_replay_logits_output = LogitsProcessorOutput(",
                    1,
                )
                + "\n"
            )
            if enable_last_hidden:
                new_block += (
                    f"{indent}if _sglang_forward_batch_requests_last_hidden_for_drafter(forward_batch):\n"
                    f"{indent}    _copy_sglang_drafter_last_hidden_output(\n"
                    f"{indent}        output, _verl_replay_logits_output,\n"
                    f"{indent}        slice(0, self.raw_num_tokens),\n"
                    f"{indent}    )\n"
                )
            if enable_raw_top_logprobs:
                new_block += (
                    f"{indent}_ensure_sglang_raw_top_logprobs_for_replay(\n"
                    f"{indent}    _verl_replay_logits_output, forward_batch, output,\n"
                    f"{indent})\n"
                )
            new_block += f"{indent}return _verl_replay_logits_output"
            source = source[:marker_index] + new_block + source[block_end:]
            break
        marker_index = source.find(marker, block_end)
        block_end = None
    if block_end is None:
        raise RuntimeError(
            f"Could not patch inline graph replay {original_method.__qualname__}."
        )

    globals_dict = original_method.__globals__
    globals_dict["_copy_sglang_drafter_last_hidden_output"] = (
        _copy_sglang_drafter_last_hidden_output
    )
    globals_dict["_ensure_sglang_raw_top_logprobs_for_replay"] = (
        _ensure_sglang_raw_top_logprobs_for_replay
    )
    globals_dict["_sglang_forward_batch_requests_last_hidden_for_drafter"] = (
        _sglang_forward_batch_requests_last_hidden_for_drafter
    )
    namespace = {}
    exec("from __future__ import annotations\n" + source, globals_dict, namespace)  # noqa: S102
    patched_method = wraps(original_method)(namespace[original_method.__name__])
    patched_method._verl_patched_drafter_output = True
    patched_method._verl_patched_drafter_raw_top_logprobs = enable_raw_top_logprobs
    patched_method._verl_patched_drafter_last_hidden_output = enable_last_hidden
    return patched_method


def _make_sglang_forward_batch_init_new_raw_top_logprobs_patch(original_method):
    @wraps(original_method)
    def patched_init_new(cls, *args, **kwargs):
        forward_batch = original_method(*args, **kwargs)
        source_batch = (
            kwargs.get("batch") if "batch" in kwargs else (args[0] if args else None)
        )
        _mark_sglang_forward_batch_raw_top_logprobs_requested(
            forward_batch, source_batch
        )
        return forward_batch

    patched_init_new._verl_patched_raw_top_logprobs_request_gate = True
    return patched_init_new


def patch_sglang_raw_top_logprobs_request_gate() -> None:
    global _SGLANG_RAW_TOP_LOGPROBS_REQUEST_GATE_PATCHED
    if _SGLANG_RAW_TOP_LOGPROBS_REQUEST_GATE_PATCHED:
        return

    try:
        forward_batch_module = importlib.import_module(
            "sglang.srt.model_executor.forward_batch_info"
        )
        forward_batch_cls = getattr(forward_batch_module, "ForwardBatch")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang raw top-logprobs request gate patch: %s", exc)
        return

    original_init_new = getattr(forward_batch_cls, "init_new", None)
    if original_init_new is None:
        logger.debug(
            "Skip SGLang raw top-logprobs request gate patch: ForwardBatch.init_new missing"
        )
        return
    if getattr(original_init_new, "_verl_patched_raw_top_logprobs_request_gate", False):
        _SGLANG_RAW_TOP_LOGPROBS_REQUEST_GATE_PATCHED = True
        return

    setattr(
        forward_batch_cls,
        "init_new",
        classmethod(
            _make_sglang_forward_batch_init_new_raw_top_logprobs_patch(
                original_init_new
            )
        ),
    )
    _SGLANG_RAW_TOP_LOGPROBS_REQUEST_GATE_PATCHED = True
    logger.warning("SGLang raw top-logprobs request gate patch active")


def _patch_sglang_drafter_output(
    *,
    enable_raw_top_logprobs: bool,
    enable_last_hidden: bool,
) -> None:
    """Install independently selectable raw-topk and final-hidden output paths."""
    global _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED
    if _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED:
        return

    if enable_raw_top_logprobs:
        patch_sglang_raw_top_logprobs_request_gate()

    try:
        logits_module = importlib.import_module("sglang.srt.layers.logits_processor")
        logits_processor_cls = getattr(logits_module, "LogitsProcessor")
    except Exception as exc:  # noqa: BLE001
        if enable_last_hidden:
            raise RuntimeError(
                "Failed to import SGLang logits processor for drafter output patch."
            ) from exc
        logger.debug("Skip SGLang drafter output patch: %s", exc)
        return

    active_parts = []

    original_forward = getattr(logits_processor_cls, "forward", None)
    if original_forward is None:
        if enable_last_hidden:
            raise RuntimeError(
                "SGLang LogitsProcessor.forward is missing; "
                "cannot patch drafter last-hidden output."
            )
    elif getattr(original_forward, "_verl_patched_drafter_output", False):
        active_parts.append("LogitsProcessor.forward")
    else:
        setattr(
            logits_processor_cls,
            "forward",
            _make_sglang_drafter_output_forward_patch(
                original_forward,
                enable_raw_top_logprobs=enable_raw_top_logprobs,
                enable_last_hidden=enable_last_hidden,
            ),
        )
        active_parts.append("LogitsProcessor.forward")

    graph_targets = (
        ("sglang.srt.model_executor.cuda_graph_runner", "CudaGraphRunner"),
        (
            "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner",
            "NPUGraphRunner",
        ),
    )
    for module_name, class_name in graph_targets:
        try:
            graph_module = importlib.import_module(module_name)
            graph_runner_cls = getattr(graph_module, class_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skip SGLang graph replay last-hidden patch for %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            continue
        original_replay = getattr(graph_runner_cls, "replay", None)
        if original_replay is None or getattr(
            original_replay, "_verl_patched_drafter_output", False
        ):
            if original_replay is not None:
                active_parts.append(f"{class_name}.replay")
            continue
        setattr(
            graph_runner_cls,
            "replay",
            _make_sglang_drafter_graph_replay_patch(
                original_replay,
                enable_raw_top_logprobs=enable_raw_top_logprobs,
                enable_last_hidden=enable_last_hidden,
            ),
        )
        active_parts.append(f"{class_name}.replay")

    for module_name, class_name in (
        (
            "sglang.srt.model_executor.piecewise_cuda_graph_runner",
            "PiecewiseCudaGraphRunner",
        ),
        (
            "sglang.srt.model_executor.breakable_cuda_graph_runner",
            "BreakableCudaGraphRunner",
        ),
    ):
        try:
            graph_module = importlib.import_module(module_name)
            graph_runner_cls = getattr(graph_module, class_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skip SGLang inline graph replay patch for %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            continue
        original_replay = getattr(graph_runner_cls, "replay", None)
        if original_replay is None or getattr(
            original_replay, "_verl_patched_drafter_output", False
        ):
            continue
        try:
            patched_replay = _make_sglang_drafter_inline_graph_replay_patch(
                original_replay,
                enable_raw_top_logprobs=enable_raw_top_logprobs,
                enable_last_hidden=enable_last_hidden,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skip SGLang inline graph replay patch for %s.%s: %s",
                module_name,
                class_name,
                exc,
            )
            continue
        setattr(graph_runner_cls, "replay", patched_replay)
        active_parts.append(f"{class_name}.replay")

    if active_parts:
        _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = True
        logger.warning(
            "SGLang drafter output patch active for %s; raw_top_logprobs=%s last_hidden=%s",
            ", ".join(active_parts),
            enable_raw_top_logprobs,
            enable_last_hidden,
        )


def patch_sglang_raw_top_logprobs_output() -> None:
    _patch_sglang_drafter_output(
        enable_raw_top_logprobs=True,
        enable_last_hidden=_sglang_drafter_return_last_hidden_enabled(),
    )


def patch_sglang_drafter_last_hidden_output() -> None:
    _patch_sglang_drafter_output(
        enable_raw_top_logprobs=_env_flag_enabled(
            _DRAFTER_RAW_TOP_LOGPROBS_ENV, default=False
        ),
        enable_last_hidden=True,
    )


def _get_sglang_dflash_verify_arg(args, kwargs, index: int, name: str):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return None


def _make_sglang_dflash_verify_hidden_states_patch(original_method):
    @wraps(original_method)
    def patched_verify(self, *args, **kwargs):
        result = original_method(self, *args, **kwargs)
        batch = _get_sglang_dflash_verify_arg(args, kwargs, 0, "batch")
        logits_output = _get_sglang_dflash_verify_arg(args, kwargs, 1, "logits_output")
        if batch is None or logits_output is None:
            return result

        try:
            next_target_hidden = result[2]
        except (IndexError, TypeError):
            if _sglang_dflash_should_return_verify_hidden(batch):
                logger.warning(
                    "SGLang DFlash verify hidden states requested but verify returned unexpected result: %s",
                    type(result).__name__,
                )
            return result

        _sglang_dflash_restore_verify_hidden(batch, logits_output, next_target_hidden)
        return result

    patched_verify._verl_patched_dflash_verify_hidden_states = True
    return patched_verify


def patch_sglang_dflash_verify_hidden_states() -> None:
    """Restore DFlash verify hidden states for VERL drafter sample collection."""
    global _SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED
    if _SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED:
        return

    try:
        module = importlib.import_module("sglang.srt.speculative.dflash_info")
        verify_cls = getattr(module, "DFlashVerifyInput")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang DFlash verify hidden-state patch: %s", exc)
        return

    original_method = getattr(verify_cls, "verify", None)
    if original_method is None:
        logger.debug(
            "Skip SGLang DFlash verify hidden-state patch: DFlashVerifyInput.verify missing"
        )
        return
    if getattr(original_method, "_verl_patched_dflash_verify_hidden_states", False):
        _SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED = True
        return

    patched_method = _make_sglang_dflash_verify_hidden_states_patch(original_method)

    setattr(verify_cls, "verify", patched_method)
    _SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED = True
    logger.warning("SGLang DFlash verify hidden-state patch active")


def patch_sglang_hidden_states_tensor_output() -> None:
    """Return SGLang hidden-state chunks as CPU tensors instead of Python lists."""
    global _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED
    patch_sglang_eagle_legacy_alignment_compat()
    patch_sglang_eagle_verify_hidden_states_full()
    patch_sglang_dflash_verify_hidden_states()
    raw_top_logprobs_enabled = _env_flag_enabled(
        _DRAFTER_RAW_TOP_LOGPROBS_ENV, default=False
    )
    last_hidden_enabled = _sglang_drafter_return_last_hidden_enabled()
    if raw_top_logprobs_enabled:
        patch_sglang_raw_top_logprobs_output()
    elif last_hidden_enabled:
        patch_sglang_drafter_last_hidden_output()
    else:
        _patch_sglang_drafter_output(
            enable_raw_top_logprobs=False,
            enable_last_hidden=False,
        )
    if _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED:
        return

    try:
        module = importlib.import_module(
            "sglang.srt.managers.scheduler_output_processor_mixin"
        )
        processor_cls = getattr(module, "SchedulerOutputProcessorMixin")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang hidden-state tensor output patch: %s", exc)
        return

    patched_methods = []
    active_methods = []
    for method_name in (
        "process_batch_result_prefill",
        "process_batch_result_decode",
        "stream_output_generation",
    ):
        original_method = getattr(processor_cls, method_name, None)
        if original_method is None or getattr(
            original_method,
            "_verl_patched_hidden_states_tensor_output",
            False,
        ):
            if original_method is not None:
                active_methods.append(method_name)
            continue

        patched_method = _make_sglang_hidden_states_tensor_output_patch(original_method)
        if patched_method is None:
            logger.debug(
                "Skip SGLang hidden-state tensor output patch for %s", method_name
            )
            continue

        setattr(processor_cls, method_name, patched_method)
        patched_methods.append(method_name)
        active_methods.append(method_name)

    required_methods = {
        "process_batch_result_prefill",
        "process_batch_result_decode",
        "stream_output_generation",
    }
    missing_methods = sorted(required_methods.difference(active_methods))
    if missing_methods:
        raise RuntimeError(
            "Failed to install required SGLang hidden-state tensor output patch methods: "
            f"{', '.join(missing_methods)}. EAGLE drafter training requires full hidden states."
        )

    if active_methods:
        _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = True
        logger.warning(
            "SGLang hidden-state tensor output patch active for %s%s",
            ", ".join(active_methods),
            f" (newly patched: {', '.join(patched_methods)})"
            if patched_methods
            else "",
        )


def _apply_selected_sglang_patches(
    patches: Iterable[str] | str | None | object = _SGLANG_PATCH_SELECTION_FROM_ENV,
) -> bool:
    selected_patches = (
        _selected_sglang_patches()
        if patches is _SGLANG_PATCH_SELECTION_FROM_ENV
        else _normalize_sglang_patch_names(patches)
    )

    patchers = (
        (_SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME, patch_sglang_qwen3_rope_compat),
        ("eagle_update_weights", patch_sglang_eagle_update_weights_from_tensor),
        ("npu_eagle_target_sampling", patch_sglang_npu_eagle_target_sampling),
        ("hidden_states_tensor_output", patch_sglang_hidden_states_tensor_output),
    )

    applied_any = False
    skipped = []
    for patch_name, patcher in patchers:
        if _sglang_patch_enabled(patch_name, selected_patches):
            patcher()
            applied_any = True
        else:
            skipped.append(patch_name)

    if skipped:
        logger.info(
            "Skip verl SGLang patches not selected by %s: %s",
            _SGLANG_PATCHES_ENV,
            ", ".join(skipped),
        )
    return applied_any


def _apply_sglang_child_process_patches() -> None:
    if _sglang_verl_patches_disabled():
        logger.warning(
            "Skip all verl SGLang patches because %s=1.", _DISABLE_SGLANG_PATCH_ENV
        )
        return

    if os.getenv(_SGLANG_RETURN_ORIGINAL_LOGPROB_ENV) == "1":
        enable_sglang_original_logprob_return()

    base_compat_patches = _selected_sglang_base_compat_patches()
    if base_compat_patches:
        logger.warning(
            "Applying verl SGLang base compatibility patches in scheduler subprocess: %s",
            ", ".join(sorted(base_compat_patches)),
        )
        if _SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME in base_compat_patches:
            patch_sglang_qwen3_rope_compat()
        return

    selected_patches = _selected_sglang_patches()
    patch_summary = (
        "all"
        if selected_patches is None
        else ", ".join(sorted(selected_patches)) or "none"
    )
    logger.warning(
        "Applying verl SGLang patches in scheduler subprocess: %s", patch_summary
    )
    _apply_selected_sglang_patches(selected_patches)


def _run_scheduler_process_with_verl_patches(*args, **kwargs):
    _apply_sglang_child_process_patches()
    return _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = True
setattr(_run_scheduler_process_with_verl_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


def _run_direct_scheduler_process_with_verl_patches(*args, **kwargs):
    global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS

    _apply_sglang_child_process_patches()
    if _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS is None:
        scheduler_module = importlib.import_module("sglang.srt.managers.scheduler")
        _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = (
            scheduler_module.run_scheduler_process
        )
    return _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_direct_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = (
    True
)
setattr(
    _run_direct_scheduler_process_with_verl_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True
)


def patch_sglang_scheduler_process_entrypoints() -> None:
    """Install child-process patches for both SGLang 0.5.9 and 0.5.10 launch paths."""
    global _SGLANG_SCHEDULER_PROCESS_PATCHED
    if _SGLANG_SCHEDULER_PROCESS_PATCHED:
        return

    patched_entrypoints = []
    modules = [sglang.srt.entrypoints.engine]
    try:
        modules.append(importlib.import_module("sglang.srt.managers.scheduler"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip direct scheduler entrypoint patch: %s", exc)

    for module in modules:
        original_run_scheduler_process = getattr(module, "run_scheduler_process", None)
        if original_run_scheduler_process is None or getattr(
            original_run_scheduler_process,
            _SCHEDULER_PROCESS_PATCH_ATTR,
            False,
        ):
            continue
        if module is sglang.srt.entrypoints.engine:
            module.run_scheduler_process = _run_scheduler_process_with_verl_patches
        else:
            global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS
            _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = (
                original_run_scheduler_process
            )
            module.run_scheduler_process = (
                _run_direct_scheduler_process_with_verl_patches
            )
        patched_entrypoints.append(module.__name__)

    engine_cls = getattr(sglang.srt.entrypoints.engine, "Engine", None)
    if engine_cls is not None:
        run_scheduler_process_func = getattr(
            engine_cls, "run_scheduler_process_func", None
        )
        if not getattr(
            run_scheduler_process_func, _SCHEDULER_PROCESS_PATCH_ATTR, False
        ):
            engine_cls.run_scheduler_process_func = staticmethod(
                _run_scheduler_process_with_verl_patches
            )
            patched_entrypoints.append(
                "sglang.srt.entrypoints.engine.Engine.run_scheduler_process_func"
            )

    if patched_entrypoints:
        _SGLANG_SCHEDULER_PROCESS_PATCHED = True
        logger.warning(
            "Patched SGLang scheduler entrypoints for %s",
            ", ".join(patched_entrypoints),
        )


def install_sglang_verl_patches(
    set_envs_and_config: Callable | None = None,
    target_weight_loader: str | None = None,
    draft_weight_loader: str | None = None,
    patches: Iterable[str] | str | None | object = _SGLANG_PATCH_SELECTION_FROM_ENV,
) -> None:
    global _target_weight_loader, _draft_weight_loader

    if _sglang_verl_patches_disabled():
        logger.warning(
            "Skip installing verl SGLang patches because %s=1.",
            _DISABLE_SGLANG_PATCH_ENV,
        )
        return

    selected_patches = (
        _selected_sglang_patches()
        if patches is _SGLANG_PATCH_SELECTION_FROM_ENV
        else _normalize_sglang_patch_names(patches)
    )
    if patches is not _SGLANG_PATCH_SELECTION_FROM_ENV:
        os.environ.pop(_SGLANG_BASE_COMPAT_PATCHES_ENV, None)
        _set_sglang_patch_selection_env(selected_patches)
        if not _sglang_patch_enabled("eagle_update_weights", selected_patches):
            _target_weight_loader = None
            _draft_weight_loader = None
            os.environ.pop(_TARGET_WEIGHT_LOADER_ENV, None)
            os.environ.pop(_DRAFT_WEIGHT_LOADER_ENV, None)

    if _sglang_patch_enabled("eagle_update_weights", selected_patches):
        configure_sglang_eagle_weight_update_patch(
            target_weight_loader, draft_weight_loader
        )
    applied_any = _apply_selected_sglang_patches(selected_patches)
    if applied_any:
        patch_sglang_scheduler_process_entrypoints()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config


def install_sglang_qwen3_rope_compat_patch(
    set_envs_and_config: Callable | None = None,
) -> None:
    if _sglang_verl_patches_disabled():
        logger.warning(
            "Skip installing SGLang Qwen3 rope compat patch because %s=1.",
            _DISABLE_SGLANG_PATCH_ENV,
        )
        return

    os.environ[_SGLANG_BASE_COMPAT_PATCHES_ENV] = _SGLANG_QWEN3_ROPE_COMPAT_PATCH_NAME
    patch_sglang_qwen3_rope_compat()
    patch_sglang_scheduler_process_entrypoints()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config
