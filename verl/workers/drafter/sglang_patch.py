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
import textwrap
from functools import wraps
from typing import Any, Callable

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
_TOP_LOGPROBS_VALUES_DTYPE_ENV = "VERL_SGLANG_TOP_LOGPROBS_VALUES_DTYPE"
_DISABLE_SGLANG_PATCH_ENV = "VERL_DISABLE_SGLANG_PATCH"
_SGLANG_PATCHES_ENV = "VERL_SGLANG_PATCHES"

_target_weight_loader: str | None = os.environ.get(_TARGET_WEIGHT_LOADER_ENV)
_draft_weight_loader: str | None = os.environ.get(_DRAFT_WEIGHT_LOADER_ENV)
_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = sglang.srt.entrypoints.engine.run_scheduler_process
_ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = None
_SGLANG_EAGLE_UPDATE_PATCHED = False
_SGLANG_QWEN3_VL_EAGLE3_AUX_HIDDEN_PATCHED = False
_SGLANG_NPU_EAGLE_SAMPLING_PATCHED = False
_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = False
_SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = False
_SGLANG_TOP_LOGPROBS_TENSOR_OUTPUT_PATCHED = False
_SGLANG_SCHEDULER_PROCESS_PATCHED = False
_SCHEDULER_PROCESS_PATCH_ATTR = "_verl_patched_scheduler_process"
_SGLANG_TOP_K_ALL = 1 << 30
_SGLANG_PATCH_NAMES = {
    "eagle_update_weights",
    "npu_eagle_target_sampling",
    "hidden_states_tensor_output",
    "top_logprobs_tensor_output",
}
_VERL_DRAFTER_HIDDEN_WINDOW_PARAM = "_verl_drafter_hidden_state_window"
_VERL_HIDDEN_STATE_FRONT_TOKENS_PARAM = "_verl_hidden_state_front_tokens_per_sample"
_VERL_HIDDEN_STATE_MAX_ROWS_PARAM = "_verl_hidden_state_max_rows"
_VERL_HIDDEN_STATE_PROMPT_LEN_PARAM = "_verl_prompt_len"
_VERL_HIDDEN_STATE_METADATA_MARKER = "__verl_hidden_state_metadata__"
_VERL_HIDDEN_STATES_STREAM_FINAL_ATTR = "_verl_hidden_states_stream_final"
_VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM = "_verl_drafter_return_last_hidden"
_VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
_VERL_DFLASH_RETURN_AUX_HIDDEN_PARAM = "_verl_dflash_return_aux_hidden"
_VERL_TOP_LOGPROBS_TENSOR_PARAM = "_verl_top_logprobs_tensor_output"
_VERL_OUTPUT_TOP_LOGPROBS_TENSOR_KEY = "_verl_output_top_logprobs_tensor"
_VERL_TOP_LOGPROBS_TENSOR_CHUNK_MARKER = "__verl_top_logprobs_tensor_chunk__"
_VERL_TOP_LOGPROBS_OUTPUT_ROW_START_PARAM = "_verl_top_logprobs_output_row_start"
_VERL_TOP_LOGPROBS_OUTPUT_ROW_END_PARAM = "_verl_top_logprobs_output_row_end"


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


def _make_verl_eagle_update_weights_patch(original_update_weights):
    @wraps(original_update_weights)
    def patched_update_weights_from_tensor(self, recv_req):
        target_weight_loader, draft_weight_loader = _get_route_markers()
        load_format = getattr(recv_req, "load_format", None)
        target_only = target_weight_loader is not None and load_format == target_weight_loader
        draft_only = draft_weight_loader is not None and load_format == draft_weight_loader
        disable_draft_model = bool(getattr(recv_req, "disable_draft_model", False)) or target_only
        disable_target_model = bool(getattr(recv_req, "disable_target_model", False)) or draft_only

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
        named_tensors = MultiprocessingSerializer.deserialize(serialized_named_tensors[tp_rank])

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

            cls.update_weights_from_tensor = _make_verl_eagle_update_weights_patch(original_update_weights)
            patched_classes.append(f"{module_name}.{class_name}")

    if patched_classes:
        _SGLANG_EAGLE_UPDATE_PATCHED = True
        logger.info("Patched SGLang EAGLE routed weight update for %s", ", ".join(patched_classes))


def _sglang_qwen3_vl_forward_supports_aux_hidden(forward_method) -> bool:
    try:
        source = inspect.getsource(forward_method)
    except (OSError, TypeError):
        return False
    call_start = source.find("return self.logits_processor(")
    if call_start < 0:
        return False
    open_idx = source.find("(", call_start)
    close_idx = _find_matching_paren(source, open_idx)
    if close_idx is None:
        return False
    call_source = source[call_start : close_idx + 1]
    return "hidden_states, aux_hidden_states" in source and "aux_hidden_states" in call_source


def _find_matching_paren(source: str, open_idx: int) -> int | None:
    depth = 0
    in_string = False
    string_quote = ""
    triple_quoted = False
    escaped = False
    idx = open_idx
    while idx < len(source):
        ch = source[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif triple_quoted and source.startswith(string_quote * 3, idx):
                in_string = False
                idx += 2
            elif not triple_quoted and ch == string_quote:
                in_string = False
        else:
            if ch in {"'", '"'}:
                string_quote = ch
                triple_quoted = source.startswith(ch * 3, idx)
                in_string = True
                if triple_quoted:
                    idx += 2
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return idx
        idx += 1
    return None


def _append_aux_hidden_arg_to_logits_processor_call(source: str) -> str | None:
    call_marker = "return self.logits_processor("
    call_start = source.find(call_marker)
    if call_start < 0:
        return None

    open_idx = source.find("(", call_start)
    close_idx = _find_matching_paren(source, open_idx)
    if close_idx is None:
        return None

    call_source = source[call_start : close_idx + 1]
    if "aux_hidden_states" in call_source:
        return source

    if "\n" not in call_source:
        return source[:close_idx] + ", aux_hidden_states" + source[close_idx:]

    close_line_start = source.rfind("\n", 0, close_idx) + 1
    close_indent = source[close_line_start:close_idx]
    if close_indent.strip():
        return None
    arg_indent = close_indent + "    "
    return source[:close_idx] + f"{arg_indent}aux_hidden_states,\n{close_indent}" + source[close_idx:]


def _make_sglang_qwen3_vl_eagle3_forward_patch(original_forward):
    try:
        source = inspect.getsource(original_forward)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    if "hidden_states, aux_hidden_states" not in source:
        last_rank_match = re.search(r"^(\s*)if self\.pp_group\.is_last_rank:", source, re.MULTILINE)
        if last_rank_match is None:
            return None
        indent = last_rank_match.group(1)
        aux_block = (
            f'{indent}aux_hidden_states = None\n'
            f'{indent}if getattr(self, "capture_aux_hidden_states", False):\n'
            f"{indent}    hidden_states, aux_hidden_states = hidden_states\n\n"
        )
        source = source[: last_rank_match.start()] + aux_block + source[last_rank_match.start() :]

    patched_source = _append_aux_hidden_arg_to_logits_processor_call(source)
    if patched_source is None or patched_source == source:
        return None
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        original_forward.__globals__,
        namespace,
    )
    patched_forward = wraps(original_forward)(namespace[original_forward.__name__])
    patched_forward._verl_patched_qwen3_vl_eagle3_aux_hidden = True
    return patched_forward


def _sglang_qwen3_vl_get_embed_and_head(self):
    return self.model.embed_tokens.weight, self.lm_head.weight


def _sglang_qwen3_vl_set_eagle3_layers_to_capture(self, layer_ids=None):
    self.capture_aux_hidden_states = True
    self.model.capture_aux_hidden_states = True
    if layer_ids is None:
        num_layers = int(getattr(self.config, "num_hidden_layers"))
        self.model.layers_to_capture = [
            2,
            num_layers // 2,
            num_layers - 3,
        ]
    else:
        self.model.layers_to_capture = [int(val) + 1 for val in layer_ids]


def patch_sglang_qwen3_vl_eagle3_aux_hidden_capture() -> None:
    """Ensure Qwen3-VL EAGLE3 aux-hidden capture support is available."""
    global _SGLANG_QWEN3_VL_EAGLE3_AUX_HIDDEN_PATCHED
    if _SGLANG_QWEN3_VL_EAGLE3_AUX_HIDDEN_PATCHED:
        return

    try:
        qwen3_vl_module = importlib.import_module("sglang.srt.models.qwen3_vl")
        qwen3_vl_cls = getattr(qwen3_vl_module, "Qwen3VLForConditionalGeneration")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang Qwen3-VL EAGLE3 aux-hidden patch: %s", exc)
        return

    original_forward = getattr(qwen3_vl_cls, "forward", None)
    if original_forward is None:
        logger.debug("Skip SGLang Qwen3-VL EAGLE3 aux-hidden patch: forward missing")
        return

    forward_supports_aux_hidden = (
        getattr(original_forward, "_verl_patched_qwen3_vl_eagle3_aux_hidden", False)
        or _sglang_qwen3_vl_forward_supports_aux_hidden(original_forward)
    )
    already_supported = (
        forward_supports_aux_hidden
        and hasattr(qwen3_vl_cls, "get_embed_and_head")
        and hasattr(qwen3_vl_cls, "set_eagle3_layers_to_capture")
    )
    if already_supported:
        _SGLANG_QWEN3_VL_EAGLE3_AUX_HIDDEN_PATCHED = True
        return

    if not forward_supports_aux_hidden:
        patched_forward = _make_sglang_qwen3_vl_eagle3_forward_patch(original_forward)
        if patched_forward is None:
            raise RuntimeError(
                "Failed to backport SGLang Qwen3-VL EAGLE3 aux-hidden forward path. "
                "The Qwen3-VL forward source layout is not compatible with this patch."
            )
        qwen3_vl_cls.forward = patched_forward

    if not hasattr(qwen3_vl_cls, "get_embed_and_head"):
        qwen3_vl_cls.get_embed_and_head = _sglang_qwen3_vl_get_embed_and_head
    if not hasattr(qwen3_vl_cls, "set_eagle3_layers_to_capture"):
        qwen3_vl_cls.set_eagle3_layers_to_capture = (
            _sglang_qwen3_vl_set_eagle3_layers_to_capture
        )

    _SGLANG_QWEN3_VL_EAGLE3_AUX_HIDDEN_PATCHED = True
    logger.warning("SGLang Qwen3-VL EAGLE3 aux-hidden capture patch active")


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


def _sglang_top_logprobs_values_dtype() -> torch.dtype:
    dtype_name = os.getenv(_TOP_LOGPROBS_VALUES_DTYPE_ENV, "float32").strip().lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _debug_sglang_npu_eagle_linear_triton(reason: str, **details: Any) -> None:
    if not _sglang_npu_eagle_linear_triton_debug_enabled():
        return
    logger.warning("SGLang NPU EAGLE linear Triton skip: %s details=%s", reason, details)


def _debug_sglang_npu_eagle_linear_triton_exception(reason: str, exc: Exception, **details: Any) -> None:
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


def _selected_sglang_patches() -> set[str] | None:
    raw_value = os.getenv(_SGLANG_PATCHES_ENV)
    if raw_value is None or not raw_value.strip():
        return None

    patch_names = [item.strip().lower() for item in re.split(r"[\s,]+", raw_value.strip()) if item.strip()]
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


def _sglang_patch_enabled(patch_name: str) -> bool:
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
    mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(version_env)) if version_env else None
    if mode is not None:
        return mode
    mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(_EAGLE_VERIFY_MODE_ENV))
    if mode is not None:
        return mode
    if version_env == _EAGLE_V1_VERIFY_MODE_ENV:
        legacy_mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(_EAGLE_V1_TARGET_SAMPLING_ENV))
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
    top_ps = top_ps.to(device=probs_for_sampling.device, dtype=probs_for_sampling.dtype).view(-1)

    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where((top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks)
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()) and bool(torch.all(top_ps >= 1.0).item()):
        return probs_for_sampling

    sorted_probs, sorted_indices = torch.sort(probs_for_sampling, dim=-1, descending=True)
    ranks = torch.arange(vocab_size, device=probs_for_sampling.device).view(1, -1)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs = sorted_probs.masked_fill((cumulative_probs - sorted_probs) > top_ps.view(-1, 1), 0.0)

    normalizer = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = torch.where(
        normalizer > 0,
        sorted_probs / normalizer.clamp_min(torch.finfo(probs_for_sampling.dtype).tiny),
        0.0,
    )

    renormed_probs = torch.zeros_like(probs_for_sampling)
    renormed_probs.scatter_(dim=1, index=sorted_indices, src=sorted_probs)
    return renormed_probs


def _top_k_renorm_prob_torch_fast(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    if probs_for_sampling.numel() == 0:
        return probs_for_sampling

    top_ks = top_ks.to(device=probs_for_sampling.device, dtype=torch.long).view(-1)
    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where((top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks)
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
            logger.debug("SGLang NPU top-k renorm fast path failed; falling back to sort path: %s", exc)
    top_ps = torch.ones(
        (probs.shape[0],),
        dtype=torch.float32 if _sglang_npu_eagle_force_fp32_sampling_enabled() else probs.dtype,
        device=probs.device,
    )
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_ks = torch.full((probs.shape[0],), probs.shape[-1], dtype=torch.long, device=probs.device)
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _sample_from_probs_with_coin(probs: torch.Tensor, coin: torch.Tensor) -> torch.Tensor:
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
    threshold = coin.to(
        dtype=probs_for_sampling.dtype,
        device=probs_for_sampling.device,
    ).view(-1, 1) * totals
    cumulative = torch.cumsum(probs_for_sampling, dim=-1)
    samples = torch.argmax((cumulative > threshold).to(torch.int32), dim=-1).to(torch.int32)
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
            draft_token_id = tl.load(candidates_ptr + row_base + safe_next_idx).to(tl.int64)
            target_prob_single = tl.load(
                target_probs_ptr + (row_base + cur_prob_idx) * vocab_size + draft_token_id,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            accepted = valid & (
                (coin <= (target_prob_single / threshold_acc)) | (target_prob_single >= threshold_single)
            )

            if accepted:
                accepted_retrive_idx = tl.load(retrive_index_ptr + row_base + safe_next_idx)
                tl.store(predicts_ptr + last_accepted_retrive_idx, draft_token_id)
                accepted_count += 1
                tl.store(
                    accept_index_ptr + accept_index_base + accepted_count,
                    accepted_retrive_idx,
                    mask=accepted_count < NUM_SPECULATIVE_TOKENS,
                )
                cur_prob_idx = safe_next_idx
                last_accepted_retrive_idx = accepted_retrive_idx
                coin = tl.load(uniform_samples_ptr + row_base + cur_prob_idx).to(tl.float32)
                residual_token_id = tl.full((), -1, tl.int64)
                residual_token_prob = tl.full((), 0.0, tl.float32)
            else:
                if valid:
                    residual_token_id = draft_token_id
                    residual_token_prob = target_prob_single
                active = tl.full((), False, tl.int1)

        tl.store(accept_token_num_ptr + req_idx, accepted_count)

        final_row_base = (row_base + cur_prob_idx) * vocab_size
        need_residual = (accepted_count != (NUM_SPECULATIVE_TOKENS - 1)) & (residual_token_id >= 0)
        num_vocab_blocks = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        total = tl.full((), 0.0, tl.float32)
        vocab_offsets = tl.arange(0, SUB_BLOCK)
        for block_idx in range(num_vocab_blocks):
            token_offsets = block_idx * SUB_BLOCK + vocab_offsets
            mask = token_offsets < vocab_size
            probs = tl.load(target_probs_ptr + final_row_base + token_offsets, mask=mask, other=0.0).to(tl.float32)
            if need_residual:
                probs = tl.where(
                    token_offsets == residual_token_id,
                    tl.maximum(probs - residual_token_prob, 0.0),
                    probs,
                )
            total += tl.sum(probs, axis=0)

        final_coin = tl.load(uniform_samples_for_final_sampling_ptr + req_idx).to(tl.float32)
        last_vocab_token_id = (vocab_size - 1).to(tl.int64)
        final_token_id = last_vocab_token_id
        if total <= 0.0:
            final_token_id = tl.minimum((final_coin * vocab_size).to(tl.int64), last_vocab_token_id)
        else:
            threshold = final_coin * total
            cumulative = tl.full((), 0.0, tl.float32)
            found = tl.full((), False, tl.int1)
            for block_idx in range(num_vocab_blocks):
                token_offsets = block_idx * SUB_BLOCK + vocab_offsets
                mask = token_offsets < vocab_size
                probs = tl.load(target_probs_ptr + final_row_base + token_offsets, mask=mask, other=0.0).to(tl.float32)
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
                    final_token_id = (block_idx * SUB_BLOCK + tl.argmax(hit_values, axis=0)).to(tl.int64)
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
            uniform_samples_for_sampling=_tensor_debug_summary(uniform_samples_for_sampling),
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
            uniform_samples_for_sampling=_tensor_debug_summary(uniform_samples_for_sampling),
            final_uniform_samples_for_sampling=_tensor_debug_summary(
                final_uniform_samples_for_sampling
            ),
            target_probs_for_sampling=_tensor_debug_summary(target_probs_for_sampling),
        )
        logger.debug("SGLang NPU EAGLE linear target-only Triton kernel failed: %s", exc)
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
        target_prob_single = torch.where(valid, target_prob_single, torch.zeros_like(target_prob_single))
        accepted = valid & (
            (coin <= (target_prob_single / threshold_acc)) | (target_prob_single >= threshold_single)
        )
        rejected = valid & ~accepted

        residual_token_id = torch.where(rejected, draft_token_id, residual_token_id)
        residual_token_prob = torch.where(rejected, target_prob_single, residual_token_prob)

        accepted_retrive_idx = retrive_index[batch_indices, safe_next_idx].to(torch.long)
        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(accepted, draft_token_id.to(dtype=predicts.dtype), old_predicts)
        predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=predict_updates)

        next_accepted_count = accepted_count + accepted.to(dtype=torch.long)
        accept_index_position = next_accepted_count.clamp_max(num_speculative_tokens - 1).view(-1, 1)
        old_accept_index = accept_index.gather(dim=1, index=accept_index_position).squeeze(1)
        accept_index_updates = torch.where(
            accepted,
            accepted_retrive_idx.to(dtype=accept_index.dtype),
            old_accept_index,
        )
        accept_index.scatter_(dim=1, index=accept_index_position, src=accept_index_updates.view(-1, 1))

        cur_prob_idx = torch.where(accepted, safe_next_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(accepted, accepted_retrive_idx, last_accepted_retrive_idx)
        accepted_count = next_accepted_count
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_token_id = torch.where(accepted, reset_retrive_idx, residual_token_id)
        residual_token_prob = torch.where(accepted, reset_residual_prob, residual_token_prob)

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    need_residual = accepted_count != (num_speculative_tokens - 1)
    residual_mask = need_residual & (residual_token_id >= 0)
    if bool(residual_mask.any().item()):
        final_probs = final_target_probs.clone()
        residual_rows = torch.nonzero(residual_mask, as_tuple=False).view(-1)
        residual_cols = residual_token_id[residual_rows]
        final_probs[residual_rows, residual_cols] = torch.clamp(
            final_probs[residual_rows, residual_cols] - residual_token_prob[residual_rows],
            min=0.0,
        )
    else:
        final_probs = final_target_probs
    final_probs = torch.where(final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs)
    final_token_ids = _sample_from_probs_with_coin(final_probs, final_uniform_samples_for_sampling)
    predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=final_token_ids.to(dtype=predicts.dtype))


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
            target_prob_single = torch.where(valid, target_prob_single, torch.zeros_like(target_prob_single))
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
                (coin <= (next_prob_acc / threshold_acc)) | (target_prob_single >= threshold_single)
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
        accepted_token_id = candidates[batch_indices, safe_found_idx].to(dtype=predicts.dtype)
        accepted_retrive_idx = retrive_index[batch_indices, safe_found_idx].to(torch.long)

        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(accepted, accepted_token_id, old_predicts)
        predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=predict_updates)

        next_accepted_count = accepted_count + accepted.to(dtype=torch.long)
        accept_index_position = next_accepted_count.clamp_max(num_speculative_tokens - 1).view(-1, 1)
        old_accept_index = accept_index.gather(dim=1, index=accept_index_position).squeeze(1)
        accept_index_updates = torch.where(
            accepted,
            accepted_retrive_idx.to(dtype=accept_index.dtype),
            old_accept_index,
        )
        accept_index.scatter_(dim=1, index=accept_index_position, src=accept_index_updates.view(-1, 1))

        cur_prob_idx = torch.where(accepted, safe_found_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(accepted, accepted_retrive_idx, last_accepted_retrive_idx)
        accepted_count = next_accepted_count
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_draft_probs.mul_((~accepted).to(dtype=residual_draft_probs.dtype).view(-1, 1))

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    residual_probs = torch.clamp(final_target_probs - residual_draft_probs, min=0.0)
    need_residual = accepted_count != (num_speculative_tokens - 1)
    final_probs = torch.where(need_residual.view(-1, 1), residual_probs, final_target_probs)
    final_probs = torch.where(final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs)
    final_token_ids = _sample_from_probs_with_coin(final_probs, final_uniform_samples_for_sampling)
    predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=final_token_ids.to(dtype=predicts.dtype))


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


def patch_sglang_npu_eagle_target_sampling() -> None:
    """Patch SGLang NPU EAGLE v1 verification to use target-only sampling."""
    global _SGLANG_NPU_EAGLE_SAMPLING_PATCHED
    if _SGLANG_NPU_EAGLE_SAMPLING_PATCHED or not _is_sglang_npu_backend():
        return

    patched_targets = []

    v1_verify_mode = _sglang_npu_eagle_v1_verify_mode()
    if v1_verify_mode != "greedy":
        try:
            eagle_info = importlib.import_module("sglang.srt.speculative.eagle_info")
            eagle_info.top_k_renorm_prob = _top_k_renorm_prob_torch
            eagle_info.top_p_renorm_prob = _top_p_renorm_prob_torch
            eagle_info.tree_speculative_sampling_target_only = _tree_speculative_sampling_target_only_torch
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
        logger.warning("Patched SGLang NPU EAGLE sampling for %s", ", ".join(patched_targets))


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


def _sglang_req_requests_top_logprobs_tensor(req) -> bool:
    return _custom_flag_enabled(_sglang_req_custom_params(req).get(_VERL_TOP_LOGPROBS_TENSOR_PARAM, False))


def _sglang_top_logprobs_output_row_bounds(custom_params: dict[str, Any]) -> tuple[int, int | None]:
    start = _int_or_none(custom_params.get(_VERL_TOP_LOGPROBS_OUTPUT_ROW_START_PARAM))
    end = _int_or_none(custom_params.get(_VERL_TOP_LOGPROBS_OUTPUT_ROW_END_PARAM))
    start = max(int(start), 0) if start is not None else 0
    if end is not None:
        end = max(int(end), start)
    return start, end


def _sglang_req_top_logprobs_output_row_bounds(req) -> tuple[int, int | None]:
    return _sglang_top_logprobs_output_row_bounds(_sglang_req_custom_params(req))


def _sglang_state_top_logprobs_output_row_bounds(state) -> tuple[int, int | None]:
    return _sglang_top_logprobs_output_row_bounds(_sglang_state_custom_params(state))


def _sglang_req_should_keep_top_logprobs_output_row(req, output_row: int) -> bool:
    start, end = _sglang_req_top_logprobs_output_row_bounds(req)
    output_row = int(output_row)
    if output_row < start:
        return False
    return end is None or output_row < end


def _sglang_top_logprobs_nums_for_spec_output_rows(reqs, top_logprobs_nums, num_tokens_per_req):
    nums = []
    for req, topk, num_tokens in zip(reqs, top_logprobs_nums, num_tokens_per_req):
        topk = int(topk or 0)
        row_base = len(getattr(req, "output_token_logprobs_val", []) or [])
        for row_offset in range(int(num_tokens)):
            if topk > 0 and _sglang_req_should_keep_top_logprobs_output_row(req, row_base + row_offset):
                nums.append(topk)
            else:
                nums.append(0)
    return nums


def _sglang_req_requests_last_hidden_for_drafter(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(_VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM, False)
    )


def _sglang_forward_batch_requests_last_hidden_for_drafter(forward_batch) -> bool:
    if _sglang_drafter_return_last_hidden_enabled():
        return True
    for req in getattr(forward_batch, "reqs", []) or []:
        if (
            getattr(req, "return_hidden_states", False)
            and _sglang_req_requests_last_hidden_for_drafter(req)
        ):
            return True
    return False


def _sglang_req_requests_dflash_aux_hidden(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(_VERL_DFLASH_RETURN_AUX_HIDDEN_PARAM, False)
    )


def _sglang_forward_batch_requests_dflash_aux_hidden(forward_batch) -> bool:
    for req in getattr(forward_batch, "reqs", []) or []:
        if (
            getattr(req, "return_hidden_states", False)
            and _sglang_req_requests_dflash_aux_hidden(req)
        ):
            return True
    return False


def _sglang_dflash_should_return_verify_hidden(batch) -> bool:
    return _sglang_forward_batch_requests_dflash_aux_hidden(batch)


def _sglang_dflash_restore_verify_hidden(batch, logits_output, next_target_hidden) -> None:
    if not _sglang_dflash_should_return_verify_hidden(batch):
        return
    dflash_hidden_states = _normalize_sglang_dflash_aux_hidden_states(next_target_hidden)
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


def _sglang_concat_last_hidden_for_drafter(req, logits_output, hidden_chunk, last_hidden_chunk):
    if not _sglang_req_requests_last_hidden_for_drafter(req):
        return hidden_chunk
    if last_hidden_chunk is None:
        raise RuntimeError("SGLang did not return final target hidden states for drafter training.")
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
    last_hidden_states = getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
    if last_hidden_states is None:
        return None
    return last_hidden_states[index]


def _filter_sglang_drafter_last_hidden_output(logits_output, index) -> None:
    last_hidden_states = getattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
    if last_hidden_states is None:
        return

    # The verify path may already have applied this filter through a source
    # patch. Keep this helper idempotent so a wrapper can safely call it again
    # across SGLang versions whose source layout differs.
    try:
        index_len = int(index.numel()) if _is_torch_tensor(index) else len(index)
    except Exception:  # noqa: BLE001
        index_len = None
    if _is_torch_tensor(last_hidden_states) and index_len is not None and last_hidden_states.dim() > 0:
        hidden_rows = int(last_hidden_states.shape[0])
        if hidden_rows == index_len:
            return
        if hidden_rows < index_len:
            logger.warning(
                "Skip filtering SGLang drafter last-hidden output: hidden_rows=%s < index_len=%s",
                hidden_rows,
                index_len,
            )
            return

    try:
        filtered_last_hidden_states = last_hidden_states[index]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to filter SGLang drafter last-hidden output by accepted indices: %s", exc)
        return
    setattr(logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, filtered_last_hidden_states)


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
    custom_prompt_len = _int_or_none(_sglang_req_custom_params(req).get(_VERL_HIDDEN_STATE_PROMPT_LEN_PARAM))
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
    if not _custom_flag_enabled(custom_params.get(_VERL_DRAFTER_HIDDEN_WINDOW_PARAM, False)):
        return None

    front_tokens = _positive_int_or_none(custom_params.get(_VERL_HIDDEN_STATE_FRONT_TOKENS_PARAM))
    if front_tokens is None:
        front_tokens = _positive_int_or_none(custom_params.get(_VERL_HIDDEN_STATE_MAX_ROWS_PARAM))
    if front_tokens is None:
        front_tokens = _positive_int_or_none(getattr(req, "_verl_hidden_state_max_rows", None))
    if front_tokens is None:
        return None

    prompt_len = _sglang_req_prompt_len(req)
    prefix_cache_rows = _sglang_req_hidden_prefix_cache_rows(req)
    window_start = max(prefix_cache_rows, max(prompt_len - 1, 0))
    return {
        "prompt_len": prompt_len,
        "prefix_cache_rows": prefix_cache_rows,
        "window_start": window_start,
        "window_end": window_start + front_tokens,
    }


def _append_sglang_hidden_chunk_payload(req, chunk, metadata: dict[str, int] | None = None) -> int:
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


def _mark_sglang_hidden_states_stream_final(req) -> None:
    setattr(req, _VERL_HIDDEN_STATES_STREAM_FINAL_ATTR, True)


def _refresh_sglang_batch_return_hidden_states(batch) -> None:
    if batch is not None:
        setattr(batch, "return_hidden_states", _sglang_batch_requests_hidden_states(batch))


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
    prefix_cache_rows: int | None = None,
    batch=None,
) -> None:
    if not getattr(req, "return_hidden_states", False):
        return

    if prefix_cache_rows is not None:
        setattr(req, "_verl_hidden_prefix_cache_rows", max(int(prefix_cache_rows), 0))

    chunk_rows = _sglang_hidden_chunk_rows(chunk)
    if chunk_rows <= 0:
        return

    if position_start is None:
        position_start = _int_or_none(getattr(req, "_verl_hidden_next_position", None))
        if position_start is None:
            position_start = _sglang_req_hidden_prefix_cache_rows(req)
    position_start = max(int(position_start), 0)
    position_end = position_start + chunk_rows
    setattr(req, "_verl_hidden_next_position", position_end)

    window_config = _sglang_hidden_window_config(req)
    if window_config is not None:
        _mark_sglang_hidden_states_stream_final(req)
        if getattr(req, "_verl_hidden_state_window_done", False):
            return
        window_start = window_config["window_start"]
        window_end = window_config["window_end"]
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
        _append_sglang_hidden_chunk_payload(
            req,
            clipped_chunk,
            {
                "position_start": clipped_start,
                "position_end": clipped_end,
                "prefix_cache_rows": window_config["prefix_cache_rows"],
                "window_start": window_start,
                "window_end": window_end,
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
        if _is_torch_tensor(chunk) and chunk.dim() > 0 and int(chunk.shape[0]) > remaining_rows:
            chunk = chunk[:remaining_rows]
        elif not _is_torch_tensor(chunk):
            try:
                if len(chunk) > remaining_rows:
                    chunk = chunk[:remaining_rows]
            except TypeError:
                pass

    appended_rows = _append_sglang_hidden_chunk_payload(req, chunk)
    collected_rows += appended_rows
    setattr(req, "_verl_hidden_state_rows", collected_rows)
    if max_rows is not None and max_rows > 0 and collected_rows >= max_rows:
        setattr(req, "_verl_hidden_state_budget_done", True)
        _finish_sglang_hidden_state_capture(req, batch)


def _append_sglang_prefill_hidden_states(req, logits_output, hidden_state_offset: int, extend_input_len: int, batch=None) -> int:
    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        if getattr(req, "return_hidden_states", False) and not getattr(req, "_verl_logged_missing_prefill_hidden_states", False):
            logger.warning(
                "SGLang did not return prefill hidden states for drafter collection: "
                "request_dflash_aux=%s logits_output_type=%s",
                _sglang_req_requests_dflash_aux_hidden(req),
                type(logits_output).__name__,
            )
            setattr(req, "_verl_logged_missing_prefill_hidden_states", True)
        return hidden_state_offset

    try:
        rows = max(int(extend_input_len), 0)
    except (TypeError, ValueError):
        rows = len(getattr(req, "origin_input_ids", []) or [])
    if rows <= 0:
        return hidden_state_offset

    prompt_len = len(getattr(req, "origin_input_ids", []) or [])
    prefix_cache_rows = max(prompt_len - rows, 0)
    end = hidden_state_offset + rows
    chunk = hidden_states[hidden_state_offset:end]
    chunk = _sglang_concat_last_hidden_for_drafter(
        req,
        logits_output,
        chunk,
        _slice_sglang_drafter_last_hidden_output(logits_output, slice(hidden_state_offset, end)),
    )
    _append_sglang_hidden_state_chunk_with_budget(
        req,
        chunk,
        position_start=prefix_cache_rows,
        prefix_cache_rows=prefix_cache_rows,
        batch=batch,
    )
    return end


def _append_sglang_decode_hidden_states(req, logits_output, result, req_index: int, hidden_state_offset: int, batch=None) -> int:
    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        if getattr(req, "return_hidden_states", False) and not getattr(req, "_verl_logged_missing_decode_hidden_states", False):
            logger.warning(
                "SGLang did not return decode hidden states for drafter collection: "
                "request_dflash_aux=%s logits_output_type=%s",
                _sglang_req_requests_dflash_aux_hidden(req),
                type(logits_output).__name__,
            )
            setattr(req, "_verl_logged_missing_decode_hidden_states", True)
        return hidden_state_offset

    accept_lengths = getattr(result, "accept_length_per_req_cpu", None)
    if accept_lengths is None:
        accept_lengths = getattr(result, "accept_lens", None)
    if accept_lengths is None:
        accept_lengths = getattr(result, "num_correct_drafts_per_req_cpu", None)
    if accept_lengths is not None and req_index < len(accept_lengths) and _is_torch_tensor(hidden_states):
        rows = max(int(accept_lengths[req_index]) + 1, 1)

        if getattr(logits_output, "_verl_dflash_aux_hidden_states", False):
            end = hidden_state_offset + rows
            total_hidden = int(hidden_states.shape[0])
            if hidden_states.dim() >= 2 and end <= total_hidden:
                position_start = max(
                    len(getattr(req, "origin_input_ids", []) or [])
                    + len(getattr(req, "output_ids", []) or [])
                    - rows,
                    0,
                )
                _append_sglang_hidden_state_chunk_with_budget(
                    req,
                    hidden_states[hidden_state_offset:end],
                    position_start=position_start,
                    batch=batch,
                )
                return end
            if getattr(req, "return_hidden_states", False):
                raise RuntimeError(
                    "SGLang DFlash verify hidden states are incomplete for accepted tokens: "
                    f"shape={tuple(hidden_states.shape)}, req_index={req_index}, "
                    f"offset={hidden_state_offset}, required_rows={rows}."
                )

        if hidden_states.dim() == 3 and req_index < int(hidden_states.shape[0]):
            rows = min(rows, int(hidden_states.shape[1]))
            if rows <= 0:
                return hidden_state_offset
            position_start = max(
                len(getattr(req, "origin_input_ids", []) or [])
                + len(getattr(req, "output_ids", []) or [])
                - rows,
                0,
            )
            chunk = hidden_states[req_index, :rows]
            chunk = _sglang_concat_last_hidden_for_drafter(
                req,
                logits_output,
                chunk,
                _slice_sglang_drafter_last_hidden_output(logits_output, (req_index, slice(0, rows))),
            )
            _append_sglang_hidden_state_chunk_with_budget(
                req,
                chunk,
                position_start=position_start,
                batch=batch,
            )
            return hidden_state_offset + rows

        num_requests = len(accept_lengths)
        total_hidden = int(hidden_states.shape[0])
        per_req_alloc = total_hidden // num_requests if num_requests > 0 else total_hidden
        rows = min(rows, max(per_req_alloc, 1))
        position_start = max(
            len(getattr(req, "origin_input_ids", []) or [])
            + len(getattr(req, "output_ids", []) or [])
            - rows,
            0,
        )
        expected_rows = num_requests * per_req_alloc
        end = hidden_state_offset + rows
        has_expected_rows = total_hidden >= expected_rows and end <= total_hidden
        if hidden_states.dim() >= 2 and has_expected_rows:
            chunk = hidden_states[hidden_state_offset:end]
            chunk = _sglang_concat_last_hidden_for_drafter(
                req,
                logits_output,
                chunk,
                _slice_sglang_drafter_last_hidden_output(logits_output, slice(hidden_state_offset, end)),
            )
            _append_sglang_hidden_state_chunk_with_budget(
                req,
                chunk,
                position_start=position_start,
                batch=batch,
            )
            return end

        if getattr(req, "return_hidden_states", False):
            raise RuntimeError(
                "SGLang EAGLE verify hidden states are incomplete for accepted tokens: "
                f"shape={tuple(hidden_states.shape)}, req_index={req_index}, "
                f"offset={hidden_state_offset}, required_rows={rows}, expected_total_rows={expected_rows}."
            )

    if not getattr(req, "return_hidden_states", False):
        return hidden_state_offset
    position_start = max(
        len(getattr(req, "origin_input_ids", []) or []) + len(getattr(req, "output_ids", []) or []) - 2,
        0,
    )
    if _is_torch_tensor(hidden_states):
        chunk = hidden_states[req_index]
        chunk = _sglang_concat_last_hidden_for_drafter(
            req,
            logits_output,
            chunk,
            _slice_sglang_drafter_last_hidden_output(logits_output, req_index),
        )
        _append_sglang_hidden_state_chunk_with_budget(
            req,
            chunk,
            position_start=position_start,
            batch=batch,
        )
    else:
        _append_sglang_hidden_state_chunk_with_budget(
            req,
            hidden_states[req_index],
            position_start=position_start,
            batch=batch,
        )
    return hidden_state_offset


def _sglang_batch_requests_hidden_states(batch) -> bool:
    return any(bool(getattr(req, "return_hidden_states", False)) for req in getattr(batch, "reqs", []) or [])


def _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info) -> None:
    if not _sglang_batch_requests_hidden_states(batch):
        return
    try:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cannot import SGLang CaptureHiddenMode for EAGLE hidden-state patch: %s", exc)
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


def _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output) -> bool:
    if not _sglang_batch_requests_hidden_states(batch):
        return False
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    if expected_rows <= 0:
        return False
    hidden_states = getattr(logits_output, "hidden_states", None)
    return hidden_states is None or _sglang_hidden_state_rows(hidden_states) < expected_rows


def _rerun_sglang_eagle_verify_without_graph(worker, model_worker_batch):
    target_worker = getattr(worker, "target_worker", None)
    model_runner = getattr(target_worker, "model_runner", None)
    graph_runner = getattr(model_runner, "graph_runner", None)
    try:
        if model_runner is not None:
            model_runner.graph_runner = None
        return target_worker.forward_batch_generation(model_worker_batch, is_verify=True)
    finally:
        if model_runner is not None:
            model_runner.graph_runner = graph_runner


def _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output) -> None:
    if not _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output):
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


def _wrap_sglang_eagle_verify_last_hidden_filter(method):
    if getattr(method, "_verl_patched_drafter_last_hidden_filter", False):
        return method

    @wraps(method)
    def patched_verify_last_hidden_filter(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        try:
            logits_output, verify_output = result[0], result[1]
            accepted_indices = getattr(verify_output, "accept_indices", None)
            if accepted_indices is None:
                accepted_indices = getattr(verify_output, "accepted_indices", None)
            if accepted_indices is not None:
                _filter_sglang_drafter_last_hidden_output(logits_output, accepted_indices)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to apply SGLang drafter last-hidden verify filter: %s", exc)
        return result

    patched_verify_last_hidden_filter._verl_patched_drafter_last_hidden_filter = True
    return patched_verify_last_hidden_filter


def _make_sglang_eagle_verify_full_hidden_patch(original_method):
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

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
        if _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output):
            logger.warning(
                "SGLang EAGLE verify returned incomplete hidden states; rerunning without graph for full hidden output."
            )
            batch_result = _rerun_sglang_eagle_verify_without_graph(self, model_worker_batch)
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
        _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output)
"""
    if old_forward in patched_source:
        patched_source = patched_source.replace(old_forward, new_forward, 1)
    else:
        return None

    hidden_filter_replacements = (
        (
            "        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]\n",
            (
                "        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]\n"
                "        _filter_sglang_drafter_last_hidden_output(logits_output, res.accepted_indices)\n"
            ),
        ),
        (
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accept_indices
            ]
""",
            """            logits_output.hidden_states = logits_output.hidden_states[
                res.accept_indices
            ]
            _filter_sglang_drafter_last_hidden_output(logits_output, res.accept_indices)
""",
        ),
        (
            "        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]\n",
            (
                "        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]\n"
                "        _filter_sglang_drafter_last_hidden_output(logits_output, res.accept_indices)\n"
            ),
        ),
    )
    for old_hidden_filter, new_hidden_filter in hidden_filter_replacements:
        if old_hidden_filter in patched_source:
            patched_source = patched_source.replace(old_hidden_filter, new_hidden_filter, 1)
            break

    globals_dict = original_method.__globals__
    globals_dict["logger"] = logger
    globals_dict["_ensure_sglang_eagle_verify_full_hidden_mode"] = _ensure_sglang_eagle_verify_full_hidden_mode
    globals_dict["_sglang_eagle_verify_hidden_states_incomplete"] = _sglang_eagle_verify_hidden_states_incomplete
    globals_dict["_rerun_sglang_eagle_verify_without_graph"] = _rerun_sglang_eagle_verify_without_graph
    globals_dict["_validate_sglang_eagle_verify_hidden_states"] = _validate_sglang_eagle_verify_hidden_states
    globals_dict["_filter_sglang_drafter_last_hidden_output"] = _filter_sglang_drafter_last_hidden_output
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
            logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s: %s", module_name, class_name, exc)
            continue
        if original_method is None:
            continue
        if getattr(original_method, "_verl_patched_eagle_verify_full_hidden_states", False):
            wrapped_method = _wrap_sglang_eagle_verify_last_hidden_filter(original_method)
            if wrapped_method is not original_method:
                setattr(worker_cls, "verify", wrapped_method)
            patched_targets.append(f"{module_name}.{class_name}.verify")
            continue
        patched_method = _make_sglang_eagle_verify_full_hidden_patch(original_method)
        if patched_method is None:
            wrapped_method = _wrap_sglang_eagle_verify_last_hidden_filter(original_method)
            if wrapped_method is original_method:
                logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s", module_name, class_name)
                continue
            setattr(worker_cls, "verify", wrapped_method)
            patched_targets.append(f"{module_name}.{class_name}.verify[last-hidden-filter]")
            logger.warning(
                "SGLang EAGLE full hidden-state source patch skipped for %s.%s; "
                "installed last-hidden accepted-index filter only.",
                module_name,
                class_name,
            )
            continue
        setattr(worker_cls, "verify", patched_method)
        patched_targets.append(f"{module_name}.{class_name}.verify")

    if patched_targets:
        _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = True
        logger.info("Patched SGLang EAGLE verify full hidden states for %s", ", ".join(patched_targets))


def _sglang_logprob_stage_name(stage: Any) -> str:
    return str(getattr(stage, "name", stage)).upper()


def _make_sglang_top_logprobs_raw_tensor_output(original_fn):
    @wraps(original_fn)
    def patched_get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage,
        extend_logprob_pruned_lens_cpu=None,
        no_copy_to_cpu=False,
    ):
        if not top_logprobs_nums:
            return [], []

        active_positions = [i for i, k in enumerate(top_logprobs_nums) if int(k) > 0]
        max_k = max((int(top_logprobs_nums[i]) for i in active_positions), default=0)
        if max_k <= 0:
            empty_values = torch.empty((len(top_logprobs_nums), 0), dtype=_sglang_top_logprobs_values_dtype())
            empty_indices = torch.empty((len(top_logprobs_nums), 0), dtype=torch.int32)
            return empty_values, empty_indices

        if _sglang_logprob_stage_name(stage) == "DECODE" and len(active_positions) < int(logprobs.shape[0]):
            active_index = torch.tensor(active_positions, device=logprobs.device, dtype=torch.long)
            active_values, active_indices = logprobs.index_select(0, active_index).topk(max_k, dim=-1)
            values = torch.full(
                (len(top_logprobs_nums), max_k),
                float("-inf"),
                device=active_values.device,
                dtype=active_values.dtype,
            )
            indices = torch.full(
                (len(top_logprobs_nums), max_k),
                -1,
                device=active_indices.device,
                dtype=active_indices.dtype,
            )
            values.index_copy_(0, active_index, active_values)
            indices.index_copy_(0, active_index, active_indices)
        else:
            values, indices = logprobs.topk(max_k, dim=-1)

        values = values.detach().to(
            device="cpu",
            dtype=_sglang_top_logprobs_values_dtype(),
            copy=True,
        )
        indices = indices.detach().to(device="cpu", dtype=torch.int32, copy=True)

        if _sglang_logprob_stage_name(stage) == "DECODE":
            return values, indices

        if extend_logprob_pruned_lens_cpu is None:
            fallback_kwargs = {"extend_logprob_pruned_lens_cpu": extend_logprob_pruned_lens_cpu}
            try:
                if "no_copy_to_cpu" in inspect.signature(original_fn).parameters:
                    fallback_kwargs["no_copy_to_cpu"] = no_copy_to_cpu
            except (TypeError, ValueError):
                pass
            return original_fn(
                logprobs,
                top_logprobs_nums,
                stage,
                **fallback_kwargs,
            )

        top_logprobs_val = []
        top_logprobs_idx = []
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0 or k <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            end = pt + pruned_len
            top_logprobs_val.append(values[pt:end, :k])
            top_logprobs_idx.append(indices[pt:end, :k])
            pt = end

        return top_logprobs_val, top_logprobs_idx

    patched_get_top_logprobs_raw._verl_patched_top_logprobs_tensor_output = True
    return patched_get_top_logprobs_raw


def _make_sglang_detokenize_logprob_tokens_tensor_aware(original_method):
    @wraps(original_method)
    def patched_detokenize_logprob_tokens(self, token_logprobs_val, token_logprobs_idx, decode_to_text):
        if _is_torch_tensor(token_logprobs_val):
            token_logprobs_val = token_logprobs_val.detach().cpu().tolist()
        if _is_torch_tensor(token_logprobs_idx):
            token_logprobs_idx = token_logprobs_idx.detach().cpu().tolist()
        return original_method(self, token_logprobs_val, token_logprobs_idx, decode_to_text)

    patched_detokenize_logprob_tokens._verl_patched_top_logprobs_tensor_output = True
    return patched_detokenize_logprob_tokens


def _make_sglang_detokenize_top_logprobs_tokens_tensor_aware(original_method):
    @wraps(original_method)
    def patched_detokenize_top_logprobs_tokens(self, token_logprobs_val, token_logprobs_idx, decode_to_text):
        ret = []
        for i in range(len(token_logprobs_val)):
            values = token_logprobs_val[i]
            indices = token_logprobs_idx[i]
            if values is None:
                ret.append(None)
                continue
            if _is_torch_tensor(values):
                if values.numel() <= 0:
                    ret.append(None)
                    continue
            elif not values:
                ret.append(None)
                continue
            ret.append(self.detokenize_logprob_tokens(values, indices, decode_to_text))
        return ret

    patched_detokenize_top_logprobs_tokens._verl_patched_top_logprobs_tensor_output = True
    return patched_detokenize_top_logprobs_tokens


def _sglang_state_custom_params(state) -> dict[str, Any]:
    obj = getattr(state, "obj", None)
    sampling_params = getattr(obj, "sampling_params", None)
    if isinstance(sampling_params, dict):
        custom_params = sampling_params.get("custom_params")
    else:
        custom_params = getattr(sampling_params, "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def _sglang_state_requests_top_logprobs_tensor(state) -> bool:
    return _custom_flag_enabled(_sglang_state_custom_params(state).get(_VERL_TOP_LOGPROBS_TENSOR_PARAM, False))


def _sglang_1d_tensor_from_top_logprob_row(row, *, topk: int, dtype: torch.dtype, fill_value: float):
    if _is_torch_tensor(row):
        tensor = row.detach().cpu().to(dtype=dtype).reshape(-1)
    elif row is None:
        tensor = torch.empty((0,), dtype=dtype)
    else:
        tensor = torch.tensor(list(row), dtype=dtype).reshape(-1)

    if tensor.numel() >= topk:
        return tensor[:topk]

    padded = torch.full((topk,), fill_value, dtype=dtype)
    if tensor.numel() > 0:
        padded[: tensor.numel()] = tensor
    return padded


def _sglang_pad_2d_tensor(tensor: torch.Tensor, *, topk: int, fill_value: float | int) -> torch.Tensor:
    rows = int(tensor.size(0))
    cols = min(int(tensor.size(1)), int(topk))
    if int(tensor.size(1)) == int(topk):
        return tensor.contiguous()

    padded = torch.full((rows, topk), fill_value, dtype=tensor.dtype)
    if cols > 0:
        padded[:, :cols] = tensor[:, :cols]
    return padded.contiguous()


def _sglang_normalize_top_logprobs_split_payload(payload, topk: int) -> dict[str, torch.Tensor] | None:
    if topk <= 0 or payload is None:
        return None

    value_dtype = _sglang_top_logprobs_values_dtype()
    values = None
    indices = None
    legacy_tensor = None
    if isinstance(payload, dict):
        values = payload.get("values")
        indices = payload.get("indices")
        legacy_tensor = payload.get("tensor")
    elif _is_torch_tensor(payload):
        legacy_tensor = payload

    if values is not None and indices is not None:
        if not _is_torch_tensor(values):
            values = torch.tensor(values, dtype=value_dtype)
        else:
            values = values.detach().cpu().to(dtype=value_dtype)
        if not _is_torch_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.int32)
        else:
            indices = indices.detach().cpu().to(dtype=torch.int32)
        if values.dim() != 2 or indices.dim() != 2:
            return None
        rows = min(int(values.size(0)), int(indices.size(0)))
        cols = min(int(values.size(1)), int(indices.size(1)), int(topk))
        if rows <= 0 or cols <= 0:
            return None
        values = _sglang_pad_2d_tensor(values[:rows, :cols], topk=topk, fill_value=float("-inf"))
        indices = _sglang_pad_2d_tensor(indices[:rows, :cols], topk=topk, fill_value=-1)
        ret = {"values": values, "indices": indices}
        if isinstance(payload, dict):
            for key in ("output_row_start", "output_row_end"):
                if key in payload:
                    ret[key] = payload[key]
        return ret

    if not _is_torch_tensor(legacy_tensor):
        return None
    tensor = legacy_tensor.detach().cpu()
    if tensor.dim() != 3 or tensor.size(-1) < 2:
        return None
    rows = int(tensor.size(0))
    cols = min(int(tensor.size(1)), int(topk))
    if rows <= 0 or cols <= 0:
        return None
    values = tensor[:, :cols, 0].to(dtype=value_dtype)
    indices = tensor[:, :cols, 1].to(dtype=torch.int32)
    values = _sglang_pad_2d_tensor(values, topk=topk, fill_value=float("-inf"))
    indices = _sglang_pad_2d_tensor(indices, topk=topk, fill_value=-1)
    ret = {"values": values, "indices": indices}
    if isinstance(payload, dict):
        for key in ("output_row_start", "output_row_end"):
            if key in payload:
                ret[key] = payload[key]
    return ret


def _sglang_top_logprobs_chunk_tensor(chunk, topk: int) -> dict[str, torch.Tensor] | None:
    if isinstance(chunk, dict) and chunk.get(_VERL_TOP_LOGPROBS_TENSOR_CHUNK_MARKER):
        return _sglang_normalize_top_logprobs_split_payload(chunk, topk)
    return None


def _pack_sglang_output_top_logprobs_tensor(
    values_rows,
    indices_rows,
    topk: int,
    *,
    output_row_start: int = 0,
    output_row_end: int | None = None,
) -> dict[str, torch.Tensor] | None:
    if topk <= 0 or values_rows is None or indices_rows is None:
        return None

    total_rows = len(values_rows)
    output_row_start = max(int(output_row_start), 0)
    output_row_end = total_rows if output_row_end is None else min(max(int(output_row_end), output_row_start), total_rows)

    chunk_payloads = []
    plain_values_rows = []
    plain_indices_rows = []
    for row_idx, values in enumerate(values_rows):
        chunk_payload = _sglang_top_logprobs_chunk_tensor(values, topk)
        if chunk_payload is not None:
            chunk_payloads.append(chunk_payload)
            continue
        if row_idx < output_row_start or row_idx >= output_row_end:
            continue
        values = values_rows[row_idx]
        indices = indices_rows[row_idx] if row_idx < len(indices_rows) else None
        plain_values_rows.append(values)
        plain_indices_rows.append(indices)

    value_rows = []
    index_rows = []
    for values, indices in zip(plain_values_rows, plain_indices_rows):
        value_row = _sglang_1d_tensor_from_top_logprob_row(
            values,
            topk=topk,
            dtype=_sglang_top_logprobs_values_dtype(),
            fill_value=float("-inf"),
        )
        index_row = _sglang_1d_tensor_from_top_logprob_row(
            indices,
            topk=topk,
            dtype=torch.int32,
            fill_value=-1,
        )
        value_rows.append(value_row)
        index_rows.append(index_row)

    payloads = list(chunk_payloads)
    if value_rows:
        payloads.append(
            {
                "values": torch.stack(value_rows, dim=0).contiguous(),
                "indices": torch.stack(index_rows, dim=0).contiguous(),
            }
        )
    if not payloads:
        return None
    if len(payloads) == 1:
        return {
            "values": payloads[0]["values"].contiguous(),
            "indices": payloads[0]["indices"].contiguous(),
            "output_row_start": output_row_start,
            "output_row_end": output_row_start + int(payloads[0]["values"].size(0)),
        }
    values = torch.cat([payload["values"] for payload in payloads], dim=0).contiguous()
    indices = torch.cat([payload["indices"] for payload in payloads], dim=0).contiguous()
    return {
        "values": values,
        "indices": indices,
        "output_row_start": output_row_start,
        "output_row_end": output_row_start + int(values.size(0)),
    }


def _sglang_pack_output_top_logprobs_stream_slice(req, start: int, end: int | None = None):
    values_rows = getattr(req, "output_top_logprobs_val", [])
    indices_rows = getattr(req, "output_top_logprobs_idx", [])
    row_start, row_end = _sglang_req_top_logprobs_output_row_bounds(req)
    row_start = max(int(start), row_start)
    if end is not None:
        end = int(end)
        row_end = min(row_end, end) if row_end is not None else end
    payload = _pack_sglang_output_top_logprobs_tensor(
        values_rows,
        indices_rows,
        int(getattr(req, "top_logprobs_num", 0) or 0),
        output_row_start=row_start,
        output_row_end=row_end,
    )
    if payload is None:
        return []
    return [
        {
            _VERL_TOP_LOGPROBS_TENSOR_CHUNK_MARKER: True,
            "values": payload["values"],
            "indices": payload["indices"],
            "output_row_start": payload.get("output_row_start"),
            "output_row_end": payload.get("output_row_end"),
        }
    ]


def _make_sglang_add_logprob_to_meta_info_tensor_output(original_method):
    @wraps(original_method)
    def patched_add_logprob_to_meta_info(
        self,
        meta_info: dict,
        state,
        top_logprobs_num: int,
        token_ids_logprob,
        return_text_in_logprobs: bool,
    ):
        if not _sglang_state_requests_top_logprobs_tensor(state):
            return original_method(
                self,
                meta_info,
                state,
                top_logprobs_num,
                token_ids_logprob,
                return_text_in_logprobs,
            )

        if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):
            state.input_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.input_token_logprobs_val[len(state.input_token_logprobs) :],
                    state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
            state.output_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                    state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_logprobs"] = state.input_token_logprobs
        meta_info["output_token_logprobs"] = state.output_token_logprobs
        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)

        topk = int(top_logprobs_num or 0)
        if topk > 0:
            output_row_start, output_row_end = _sglang_state_top_logprobs_output_row_bounds(state)
            output_top_logprobs = _pack_sglang_output_top_logprobs_tensor(
                state.output_top_logprobs_val,
                state.output_top_logprobs_idx,
                topk,
                output_row_start=output_row_start,
                output_row_end=output_row_end,
            )
            if output_top_logprobs is not None:
                meta_info[_VERL_OUTPUT_TOP_LOGPROBS_TENSOR_KEY] = output_top_logprobs

            # Keep public fields present but empty for this verl-only internal path.
            # The full top-k payload is in the tensor side-channel above.
            meta_info["input_top_logprobs"] = []
            meta_info["output_top_logprobs"] = []

        if token_ids_logprob is not None:
            if len(state.input_token_ids_logprobs_val) > len(state.input_token_ids_logprobs):
                state.input_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_token_ids_logprobs_val[len(state.input_token_ids_logprobs) :],
                        state.input_token_ids_logprobs_idx[len(state.input_token_ids_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_token_ids_logprobs_val) > len(state.output_token_ids_logprobs):
                state.output_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_token_ids_logprobs_val[len(state.output_token_ids_logprobs) :],
                        state.output_token_ids_logprobs_idx[len(state.output_token_ids_logprobs) :],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs
            meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs

    patched_add_logprob_to_meta_info._verl_patched_top_logprobs_tensor_output = True
    return patched_add_logprob_to_meta_info


def patch_sglang_top_logprobs_tensor_output() -> None:
    """Return SGLang output top-logprobs through a verl-only tensor side-channel."""
    global _SGLANG_TOP_LOGPROBS_TENSOR_OUTPUT_PATCHED
    if _SGLANG_TOP_LOGPROBS_TENSOR_OUTPUT_PATCHED:
        return

    try:
        logprob_module = importlib.import_module("sglang.srt.layers.utils.logprob")
        sampler_module = importlib.import_module("sglang.srt.layers.sampler")
        logits_processor_module = importlib.import_module("sglang.srt.layers.logits_processor")
        tokenizer_manager_module = importlib.import_module("sglang.srt.managers.tokenizer_manager")
        output_processor_module = importlib.import_module("sglang.srt.managers.scheduler_output_processor_mixin")
        tokenizer_manager_cls = getattr(tokenizer_manager_module, "TokenizerManager")
        output_processor_cls = getattr(output_processor_module, "SchedulerOutputProcessorMixin")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang top-logprobs tensor output patch: %s", exc)
        return

    raw_fn = getattr(logprob_module, "get_top_logprobs_raw", None)
    if raw_fn is None:
        logger.debug("Skip SGLang top-logprobs tensor output patch: get_top_logprobs_raw missing")
        return

    if not getattr(raw_fn, "_verl_patched_top_logprobs_tensor_output", False):
        patched_raw_fn = _make_sglang_top_logprobs_raw_tensor_output(raw_fn)
        logprob_module.get_top_logprobs_raw = patched_raw_fn

        def patched_get_top_logprobs(logprobs, top_logprobs_nums):
            return patched_raw_fn(logprobs, top_logprobs_nums, stage=logprob_module.LogprobStage.DECODE)

        def patched_get_top_logprobs_prefill(all_logprobs, logits_metadata):
            return patched_raw_fn(
                all_logprobs,
                logits_metadata.top_logprobs_nums,
                stage=logprob_module.LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
            )

        patched_get_top_logprobs._verl_patched_top_logprobs_tensor_output = True
        patched_get_top_logprobs_prefill._verl_patched_top_logprobs_tensor_output = True
        logprob_module.get_top_logprobs = patched_get_top_logprobs
        logprob_module.get_top_logprobs_prefill = patched_get_top_logprobs_prefill
        sampler_module.get_top_logprobs = patched_get_top_logprobs
        logits_processor_module.get_top_logprobs_prefill = patched_get_top_logprobs_prefill

    spec_logprob_fn = getattr(logprob_module, "add_output_logprobs_for_spec_v1", None)
    if spec_logprob_fn is not None and not getattr(
        spec_logprob_fn,
        "_verl_patched_top_logprobs_tensor_output",
        False,
    ):
        patched_spec_logprob_fn = _make_sglang_spec_top_logprobs_window_patch(spec_logprob_fn)
        if patched_spec_logprob_fn is not None:
            logprob_module.add_output_logprobs_for_spec_v1 = patched_spec_logprob_fn
            for module_name in (
                "sglang.srt.speculative.eagle_worker",
                "sglang.srt.speculative.multi_layer_eagle_worker",
                "sglang.srt.speculative.ngram_worker",
            ):
                try:
                    spec_module = importlib.import_module(module_name)
                except Exception:  # noqa: BLE001
                    continue
                if getattr(spec_module, "add_output_logprobs_for_spec_v1", None) is spec_logprob_fn:
                    spec_module.add_output_logprobs_for_spec_v1 = patched_spec_logprob_fn
        else:
            logger.debug(
                "Skip SGLang spec top-logprobs row-window patch: add_output_logprobs_for_spec_v1 source block not found."
            )

    detokenize_logprob = getattr(tokenizer_manager_cls, "detokenize_logprob_tokens", None)
    if detokenize_logprob is not None and not getattr(
        detokenize_logprob,
        "_verl_patched_top_logprobs_tensor_output",
        False,
    ):
        tokenizer_manager_cls.detokenize_logprob_tokens = _make_sglang_detokenize_logprob_tokens_tensor_aware(
            detokenize_logprob
        )

    detokenize_top = getattr(tokenizer_manager_cls, "detokenize_top_logprobs_tokens", None)
    if detokenize_top is not None and not getattr(
        detokenize_top,
        "_verl_patched_top_logprobs_tensor_output",
        False,
    ):
        tokenizer_manager_cls.detokenize_top_logprobs_tokens = _make_sglang_detokenize_top_logprobs_tokens_tensor_aware(
            detokenize_top
        )

    add_logprob = getattr(tokenizer_manager_cls, "add_logprob_to_meta_info", None)
    if add_logprob is None:
        logger.debug("Skip SGLang top-logprobs tensor output patch: add_logprob_to_meta_info missing")
        return
    if not getattr(add_logprob, "_verl_patched_top_logprobs_tensor_output", False):
        tokenizer_manager_cls.add_logprob_to_meta_info = _make_sglang_add_logprob_to_meta_info_tensor_output(
            add_logprob
        )

    stream_output = getattr(output_processor_cls, "stream_output_generation", None)
    if stream_output is not None and not (
        getattr(stream_output, "_verl_patched_hidden_states_tensor_output", False)
        or getattr(stream_output, "_verl_patched_top_logprobs_stream_tensor_output", False)
    ):
        patched_stream_output = _make_sglang_top_logprobs_stream_output_patch(stream_output)
        if patched_stream_output is not None:
            output_processor_cls.stream_output_generation = patched_stream_output
        else:
            logger.debug(
                "Skip SGLang top-logprobs stream tensor output patch: output_top_logprobs slice block not found."
            )

    _SGLANG_TOP_LOGPROBS_TENSOR_OUTPUT_PATCHED = True
    logger.warning("SGLang top-logprobs tensor output patch active")


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
_SGLANG_STREAM_TOP_LOGPROBS_SLICE_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)output_top_logprobs_val\.append\(\r?\n"
    r"(?P=indent)[ \t]+req\.output_top_logprobs_val\[\r?\n"
    r"(?P=indent)[ \t]+[ \t]+send_output_token_logprobs_offset:(?P<end>[A-Za-z_][A-Za-z0-9_]*)?\r?\n"
    r"(?P=indent)[ \t]+\]\r?\n"
    r"(?P=indent)\)\r?\n"
    r"(?P=indent)output_top_logprobs_idx\.append\(\r?\n"
    r"(?P=indent)[ \t]+req\.output_top_logprobs_idx\[\r?\n"
    r"(?P=indent)[ \t]+[ \t]+send_output_token_logprobs_offset:(?P=end)?\r?\n"
    r"(?P=indent)[ \t]+\]\r?\n"
    r"(?P=indent)\)\r?\n",
)
_SGLANG_SPEC_TOP_LOGPROBS_REPEAT_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)top_logprobs_nums_repeat_interleaved\s*=\s*\[\r?\n"
    r"(?P=indent)[ \t]+num\r?\n"
    r"(?P=indent)[ \t]+for\s+num,\s*num_tokens\s+in\s+zip\(top_logprobs_nums,\s*num_tokens_per_req\)\r?\n"
    r"(?P=indent)[ \t]+for\s+_\s+in\s+range\(num_tokens\)\r?\n"
    r"(?P=indent)\]\r?\n",
)
_SGLANG_SPEC_TOP_LOGPROBS_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s+req\.top_logprobs_num\s*>\s*0\s*:\r?\n"
    r"(?P=indent)[ \t]+assert\s*\(\r?\n"
    r"(?P=indent)[ \t]+[ \t]+should_top_logprobs\r?\n"
    r"(?P=indent)[ \t]+\),\s*\"Inconsistent state: should_top_logprobs is False\"\r?\n"
    r"(?P=indent)[ \t]+req\.output_top_logprobs_val\.append\(token_top_logprobs_val\[pt\]\)\r?\n"
    r"(?P=indent)[ \t]+req\.output_top_logprobs_idx\.append\(token_top_logprobs_idx\[pt\]\)\r?\n",
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


def _render_sglang_stream_top_logprobs_slice(match: re.Match) -> str:
    indent = match.group("indent")
    end_expr = match.group("end")
    end_arg = f"{indent}            {end_expr},\n" if end_expr else ""
    slice_end = end_expr or ""
    return (
        f"{indent}if _sglang_req_requests_top_logprobs_tensor(req) and req.top_logprobs_num > 0:\n"
        f"{indent}    output_top_logprobs_val.append(\n"
        f"{indent}        _sglang_pack_output_top_logprobs_stream_slice(\n"
        f"{indent}            req,\n"
        f"{indent}            send_output_token_logprobs_offset,\n"
        f"{end_arg}"
        f"{indent}        )\n"
        f"{indent}    )\n"
        f"{indent}    output_top_logprobs_idx.append([])\n"
        f"{indent}else:\n"
        f"{indent}    output_top_logprobs_val.append(\n"
        f"{indent}        req.output_top_logprobs_val[\n"
        f"{indent}            send_output_token_logprobs_offset:{slice_end}\n"
        f"{indent}        ]\n"
        f"{indent}    )\n"
        f"{indent}    output_top_logprobs_idx.append(\n"
        f"{indent}        req.output_top_logprobs_idx[\n"
        f"{indent}            send_output_token_logprobs_offset:{slice_end}\n"
        f"{indent}        ]\n"
        f"{indent}    )\n"
    )


def _render_sglang_spec_top_logprobs_repeat(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}top_logprobs_nums_repeat_interleaved = _sglang_top_logprobs_nums_for_spec_output_rows(\n"
        f"{indent}    batch.reqs,\n"
        f"{indent}    top_logprobs_nums,\n"
        f"{indent}    num_tokens_per_req,\n"
        f"{indent})\n"
    )


def _render_sglang_spec_top_logprobs_append(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}if req.top_logprobs_num > 0:\n"
        f"{indent}    if _sglang_req_should_keep_top_logprobs_output_row(\n"
        f"{indent}        req,\n"
        f"{indent}        len(req.output_token_logprobs_val) - 1,\n"
        f"{indent}    ):\n"
        f"{indent}        assert (\n"
        f"{indent}            should_top_logprobs\n"
        f"{indent}        ), \"Inconsistent state: should_top_logprobs is False\"\n"
        f"{indent}        req.output_top_logprobs_val.append(token_top_logprobs_val[pt])\n"
        f"{indent}        req.output_top_logprobs_idx.append(token_top_logprobs_idx[pt])\n"
        f"{indent}    else:\n"
        f"{indent}        req.output_top_logprobs_val.append([])\n"
        f"{indent}        req.output_top_logprobs_idx.append([])\n"
    )


def _patch_sglang_spec_top_logprobs_window_source(source: str) -> str | None:
    patched_source, repeat_count = _SGLANG_SPEC_TOP_LOGPROBS_REPEAT_PATTERN.subn(
        _render_sglang_spec_top_logprobs_repeat,
        source,
        count=1,
    )
    if repeat_count <= 0:
        return None
    patched_source, append_count = _SGLANG_SPEC_TOP_LOGPROBS_APPEND_PATTERN.subn(
        _render_sglang_spec_top_logprobs_append,
        patched_source,
        count=1,
    )
    if append_count <= 0:
        return None
    return patched_source


def _insert_sglang_decode_hidden_state_offset(source: str) -> str | None:
    if re.search(r"(?m)^[ \t]+hidden_state_offset = 0\s*$", source):
        return source

    patched_source, loop_count = _SGLANG_DECODE_REQUEST_LOOP_PATTERN.subn(
        lambda match: f"{match.group('indent')}hidden_state_offset = 0\n\n{match.group(0)}",
        source,
        count=1,
    )
    if loop_count <= 0:
        return None
    return patched_source


def _patch_sglang_decode_hidden_states_source(source: str) -> str | None:
    patched_source, hidden_block_count = _SGLANG_DECODE_HIDDEN_STATES_APPEND_PATTERN.subn(
        _render_sglang_decode_hidden_states_append,
        source,
    )
    if hidden_block_count <= 0:
        return None
    return _insert_sglang_decode_hidden_state_offset(patched_source)


def _patch_sglang_prefill_hidden_states_source(source: str) -> str | None:
    patched_source, hidden_block_count = _SGLANG_PREFILL_HIDDEN_STATES_APPEND_PATTERN.subn(
        _render_sglang_prefill_hidden_states_append,
        source,
        count=1,
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


def _patch_sglang_stream_top_logprobs_source(source: str) -> str | None:
    patched_source, top_logprobs_count = _SGLANG_STREAM_TOP_LOGPROBS_SLICE_PATTERN.subn(
        _render_sglang_stream_top_logprobs_slice,
        source,
        count=1,
    )
    if top_logprobs_count <= 0:
        return None
    return patched_source


def _make_sglang_top_logprobs_stream_output_patch(original_method):
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source = _patch_sglang_stream_top_logprobs_source(source)
    if patched_source is None or patched_source == source:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_sglang_req_requests_top_logprobs_tensor"] = _sglang_req_requests_top_logprobs_tensor
    globals_dict["_sglang_pack_output_top_logprobs_stream_slice"] = _sglang_pack_output_top_logprobs_stream_slice
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_top_logprobs_stream_tensor_output = True
    return patched_method


def _make_sglang_spec_top_logprobs_window_patch(original_fn):
    try:
        source = inspect.getsource(original_fn)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source = _patch_sglang_spec_top_logprobs_window_source(source)
    if patched_source is None or patched_source == source:
        return None

    globals_dict = original_fn.__globals__
    globals_dict["_sglang_top_logprobs_nums_for_spec_output_rows"] = (
        _sglang_top_logprobs_nums_for_spec_output_rows
    )
    globals_dict["_sglang_req_should_keep_top_logprobs_output_row"] = (
        _sglang_req_should_keep_top_logprobs_output_row
    )
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_fn = namespace[original_fn.__name__]
    patched_fn = wraps(original_fn)(patched_fn)
    patched_fn._verl_patched_top_logprobs_tensor_output = True
    return patched_fn


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
    patched_top_logprobs_stream = False
    if original_method.__name__ == "process_batch_result_prefill":
        patched_prefill_source = _patch_sglang_prefill_hidden_states_source(patched_source)
        if patched_prefill_source is None:
            logger.warning(
                "Skip SGLang prefill hidden-state window patch for %s: hidden append block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_prefill_source
    elif original_method.__name__ == "process_batch_result_decode":
        patched_decode_source = _patch_sglang_decode_hidden_states_source(patched_source)
        if patched_decode_source is None:
            logger.warning(
                "Skip SGLang decode hidden-state full-output patch for %s: hidden append block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_decode_source
    elif original_method.__name__ == "stream_output_generation":
        patched_stream_source = _patch_sglang_stream_hidden_states_source(patched_source)
        if patched_stream_source is None:
            logger.warning(
                "Skip SGLang stream hidden-state final-output patch for %s: hidden stream block not found.",
                original_method.__name__,
            )
            return None
        patched_source = patched_stream_source
        if _sglang_patch_enabled("top_logprobs_tensor_output"):
            patched_top_logprobs_source = _patch_sglang_stream_top_logprobs_source(patched_source)
            if patched_top_logprobs_source is not None:
                patched_source = patched_top_logprobs_source
                patched_top_logprobs_stream = True
            else:
                logger.debug(
                    "Skip SGLang stream top-logprobs tensor chunk patch for %s: output_top_logprobs slice block not found.",
                    original_method.__name__,
                )
    elif conversion_count <= 0:
        return None

    if patched_source == source:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_append_sglang_prefill_hidden_states"] = _append_sglang_prefill_hidden_states
    globals_dict["_append_sglang_decode_hidden_states"] = _append_sglang_decode_hidden_states
    globals_dict["_sglang_req_should_stream_hidden_states"] = _sglang_req_should_stream_hidden_states
    globals_dict["_sglang_req_requests_top_logprobs_tensor"] = _sglang_req_requests_top_logprobs_tensor
    globals_dict["_sglang_pack_output_top_logprobs_stream_slice"] = _sglang_pack_output_top_logprobs_stream_slice
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_hidden_states_tensor_output = True
    if patched_top_logprobs_stream:
        patched_method._verl_patched_top_logprobs_stream_tensor_output = True
    return patched_method


def _make_sglang_drafter_last_hidden_forward_patch(original_method):
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
        return_last_hidden = False
        if _sglang_drafter_return_last_hidden_enabled():
            return_last_hidden = True
        else:
            return_last_hidden = _sglang_forward_batch_requests_last_hidden_for_drafter(logits_metadata)
        return_dflash_aux_hidden = _sglang_forward_batch_requests_dflash_aux_hidden(logits_metadata)

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
            dflash_hidden_states = _normalize_sglang_dflash_aux_hidden_states(aux_hidden_states)
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
        if return_last_hidden and getattr(output, "hidden_states", None) is not None and hidden_states is not None:
            setattr(output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, hidden_states)
        return output

    patched_logits_processor_forward._verl_patched_drafter_last_hidden_output = True
    return patched_logits_processor_forward


def _copy_sglang_drafter_last_hidden_output(src, dst, index) -> None:
    last_hidden_states = getattr(src, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
    if last_hidden_states is not None and dst is not None:
        setattr(dst, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, last_hidden_states[index])


def _sglang_graph_replay_output_buffer(runner, original_method):
    keys = []
    if getattr(runner, "enable_pdmux", False):
        get_current_stream_idx = original_method.__globals__.get("get_current_stream_idx")
        if callable(get_current_stream_idx):
            keys.append(f"{get_current_stream_idx()}_{runner.bs}")
    keys.append(runner.bs)
    for key in keys:
        try:
            return runner.output_buffers[key]
        except Exception:  # noqa: BLE001
            continue
    return None


def _make_sglang_drafter_last_hidden_graph_replay_patch(original_method):
    @wraps(original_method)
    def patched_graph_replay(self, *args, **kwargs):
        result = original_method(self, *args, **kwargs)
        output = _sglang_graph_replay_output_buffer(self, original_method)
        if output is not None:
            _copy_sglang_drafter_last_hidden_output(output, result, slice(0, self.raw_num_token))
        return result

    patched_graph_replay._verl_patched_drafter_last_hidden_output = True
    return patched_graph_replay


def patch_sglang_drafter_last_hidden_output() -> None:
    """Carry final target hidden to verl output without changing SGLang EAGLE's 3H hidden."""
    global _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED
    if _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED:
        return

    try:
        logits_module = importlib.import_module("sglang.srt.layers.logits_processor")
        logits_processor_cls = getattr(logits_module, "LogitsProcessor")
    except Exception as exc:  # noqa: BLE001
        if _sglang_drafter_return_last_hidden_enabled():
            raise RuntimeError(
                "Failed to import SGLang logits processor for drafter last-hidden output patch."
            ) from exc
        logger.debug("Skip SGLang drafter last-hidden output patch: %s", exc)
        return

    active_parts = []

    original_forward = getattr(logits_processor_cls, "forward", None)
    if original_forward is None:
        if _sglang_drafter_return_last_hidden_enabled():
            raise RuntimeError(
                "SGLang LogitsProcessor.forward is missing; "
                "cannot patch drafter last-hidden output."
            )
    elif getattr(original_forward, "_verl_patched_drafter_last_hidden_output", False):
        active_parts.append("LogitsProcessor.forward")
    else:
        setattr(
            logits_processor_cls,
            "forward",
            _make_sglang_drafter_last_hidden_forward_patch(original_forward),
        )
        active_parts.append("LogitsProcessor.forward")

    graph_targets = (
        ("sglang.srt.model_executor.cuda_graph_runner", "CudaGraphRunner"),
        ("sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner", "NPUGraphRunner"),
    )
    for module_name, class_name in graph_targets:
        try:
            graph_module = importlib.import_module(module_name)
            graph_runner_cls = getattr(graph_module, class_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang graph replay last-hidden patch for %s.%s: %s", module_name, class_name, exc)
            continue
        original_replay = getattr(graph_runner_cls, "replay", None)
        if original_replay is None or getattr(original_replay, "_verl_patched_drafter_last_hidden_output", False):
            if original_replay is not None:
                active_parts.append(f"{class_name}.replay")
            continue
        setattr(
            graph_runner_cls,
            "replay",
            _make_sglang_drafter_last_hidden_graph_replay_patch(original_replay),
        )
        active_parts.append(f"{class_name}.replay")

    if active_parts:
        _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = True
        logger.warning(
            "SGLang drafter last-hidden output patch active for %s; return_last_hidden=%s",
            ", ".join(active_parts),
            _sglang_drafter_return_last_hidden_enabled(),
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
        logger.debug("Skip SGLang DFlash verify hidden-state patch: DFlashVerifyInput.verify missing")
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
    patch_sglang_eagle_verify_hidden_states_full()
    patch_sglang_dflash_verify_hidden_states()
    patch_sglang_drafter_last_hidden_output()
    if _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED:
        return

    try:
        module = importlib.import_module("sglang.srt.managers.scheduler_output_processor_mixin")
        processor_cls = getattr(module, "SchedulerOutputProcessorMixin")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang hidden-state tensor output patch: %s", exc)
        return

    patched_methods = []
    active_methods = []
    for method_name in ("process_batch_result_prefill", "process_batch_result_decode", "stream_output_generation"):
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
            logger.debug("Skip SGLang hidden-state tensor output patch for %s", method_name)
            continue

        setattr(processor_cls, method_name, patched_method)
        patched_methods.append(method_name)
        active_methods.append(method_name)

    required_methods = {"process_batch_result_prefill", "process_batch_result_decode", "stream_output_generation"}
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
            f" (newly patched: {', '.join(patched_methods)})" if patched_methods else "",
        )


def _apply_selected_sglang_patches() -> bool:
    selected_patches = _selected_sglang_patches()
    if selected_patches != set():
        patch_sglang_qwen3_vl_eagle3_aux_hidden_capture()

    patchers = (
        ("eagle_update_weights", patch_sglang_eagle_update_weights_from_tensor),
        ("npu_eagle_target_sampling", patch_sglang_npu_eagle_target_sampling),
        ("hidden_states_tensor_output", patch_sglang_hidden_states_tensor_output),
        ("top_logprobs_tensor_output", patch_sglang_top_logprobs_tensor_output),
    )

    applied_any = False
    skipped = []
    for patch_name, patcher in patchers:
        if _sglang_patch_enabled(patch_name):
            patcher()
            applied_any = True
        else:
            skipped.append(patch_name)

    if skipped:
        logger.info("Skip verl SGLang patches not selected by %s: %s", _SGLANG_PATCHES_ENV, ", ".join(skipped))
    return applied_any


def _apply_sglang_child_process_patches() -> None:
    if _sglang_verl_patches_disabled():
        logger.warning("Skip all verl SGLang patches because %s=1.", _DISABLE_SGLANG_PATCH_ENV)
        return

    logger.warning("Applying verl SGLang patches in scheduler subprocess.")
    _apply_selected_sglang_patches()


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
        _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = scheduler_module.run_scheduler_process
    return _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_direct_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = True
setattr(_run_direct_scheduler_process_with_verl_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


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
            _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = original_run_scheduler_process
            module.run_scheduler_process = _run_direct_scheduler_process_with_verl_patches
        patched_entrypoints.append(module.__name__)

    engine_cls = getattr(sglang.srt.entrypoints.engine, "Engine", None)
    if engine_cls is not None:
        run_scheduler_process_func = getattr(engine_cls, "run_scheduler_process_func", None)
        if not getattr(run_scheduler_process_func, _SCHEDULER_PROCESS_PATCH_ATTR, False):
            engine_cls.run_scheduler_process_func = staticmethod(_run_scheduler_process_with_verl_patches)
            patched_entrypoints.append("sglang.srt.entrypoints.engine.Engine.run_scheduler_process_func")

    if patched_entrypoints:
        _SGLANG_SCHEDULER_PROCESS_PATCHED = True
        logger.warning("Patched SGLang scheduler entrypoints for %s", ", ".join(patched_entrypoints))


def install_sglang_verl_patches(
    set_envs_and_config: Callable | None = None,
    target_weight_loader: str | None = None,
    draft_weight_loader: str | None = None,
) -> None:
    if _sglang_verl_patches_disabled():
        logger.warning("Skip installing verl SGLang patches because %s=1.", _DISABLE_SGLANG_PATCH_ENV)
        return

    if _sglang_patch_enabled("eagle_update_weights"):
        configure_sglang_eagle_weight_update_patch(target_weight_loader, draft_weight_loader)
    applied_any = _apply_selected_sglang_patches()
    if applied_any:
        patch_sglang_scheduler_process_entrypoints()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config
