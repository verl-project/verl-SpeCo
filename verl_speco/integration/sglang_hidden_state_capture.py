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
"""Source rewrites that route SGLang's per-request hidden-state capture through SPECO.

SPECO collects drafter training data from SGLang by rewriting the scheduler's batch
result processors, so that every prefill and decode step appends its hidden-state rows
through ``_append_sglang_prefill_hidden_states`` / ``_append_sglang_decode_hidden_states``
(per-request row offsets, accepted verify rows, positions and capture windows) instead of
SGLang's native append. Two layouts of those processors are known:

* ``sglang.srt.managers.scheduler_output_processor_mixin.SchedulerOutputProcessorMixin``
  (SGLang up to 0.5.12), with the hidden-state appends inline in
  ``process_batch_result_prefill`` / ``process_batch_result_decode`` and the stream output
  built in ``stream_output_generation``;
* ``sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor``
  (the SGLang build in the ROCm image, which reports 0.5.16), where prefill calls
  ``self._append_prefill_hidden_states(...)`` and the stream output moved to
  ``output_streamer._GenerationStreamAccumulator`` (rewritten by
  :mod:`verl_speco.integration.sglang_hidden_state_stream`).

If neither layout can be patched, SGLang's native capture stays active. It appends one row
per request and verify step, indexed by the request's batch position, from a hidden-state
tensor that DFlash compacts over all requests' committed rows. The rows then belong to
other requests or positions and carry no position metadata. Training on them is silently
wrong, so :func:`install_sglang_output_processor_capture_patches` raises instead.

This module does not import SGLang so the rewrites can be unit-tested on plain sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from verl_speco.integration.sglang_patch_controls import (
    DISABLE_SGLANG_PATCH_ENV,
    sglang_verl_patches_disabled,
)

PREFILL_METHOD = "process_batch_result_prefill"
DECODE_METHOD = "process_batch_result_decode"
STREAM_METHOD = "stream_output_generation"
PATCHED_METHOD_ATTR = "_verl_patched_hidden_states_tensor_output"


@dataclass(frozen=True)
class SGLangOutputProcessorLayout:
    name: str
    module: str
    class_name: str
    methods: tuple[str, ...]
    # The stream output is built by ``_GenerationStreamAccumulator.accept``, which must
    # be patched separately for chunk payloads to reach the tokenizer manager.
    stream_via_accumulator: bool


SGLANG_OUTPUT_PROCESSOR_LAYOUTS: tuple[SGLangOutputProcessorLayout, ...] = (
    SGLangOutputProcessorLayout(
        name="scheduler_output_processor_mixin",
        module="sglang.srt.managers.scheduler_output_processor_mixin",
        class_name="SchedulerOutputProcessorMixin",
        methods=(PREFILL_METHOD, DECODE_METHOD, STREAM_METHOD),
        stream_via_accumulator=False,
    ),
    SGLangOutputProcessorLayout(
        name="scheduler_components.batch_result_processor",
        module="sglang.srt.managers.scheduler_components.batch_result_processor",
        class_name="SchedulerBatchResultProcessor",
        methods=(PREFILL_METHOD, DECODE_METHOD),
        stream_via_accumulator=True,
    ),
)


class SGLangHiddenStateCapturePatchError(RuntimeError):
    """SPECO cannot route SGLang's hidden-state capture through its own collector."""


_HIDDEN_STATES_LIST_OUTPUT_PATTERN = re.compile(
    r"\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\)",
)
_HIDDEN_STATES_GUARD = (
    r"if\s*(?:\(\s*)?req\.return_hidden_states\s+"
    r"and\s+logits_output\.hidden_states\s+is\s+not\s+None\s*(?:\))?\s*:\r?\n"
)
_DECODE_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)"
    + _HIDDEN_STATES_GUARD
    + r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"^(?P=indent)[ \t]+\)\r?\n",
)
# Mixin layout: the prefill slice is appended inline.
_PREFILL_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)"
    + _HIDDEN_STATES_GUARD
    + r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"(?:\.detach\(\)\.to\(\"cpu\",\s*copy=True\)|\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\))\r?\n"
    r"(?P=indent)[ \t]+\)\r?\n",
)
# Batch-result-processor layout: the prefill slice goes through a helper method, which
# slices ``len(req.origin_input_ids)`` rows even when part of the prompt was a prefix-cache
# hit and only the extend rows are present.
_PREFILL_HIDDEN_STATES_HELPER_CALL_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)"
    + _HIDDEN_STATES_GUARD
    + r"(?P=indent)[ \t]+hidden_state_offset\s*=\s*self\._append_prefill_hidden_states\(\r?\n"
    r".*?"
    r"^(?P=indent)[ \t]+\)\r?\n",
)
_STREAM_HIDDEN_STATES_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s+req\.return_hidden_states\s*:\r?\n"
    r"(?P=indent)[ \t]+if\s+output_hidden_states\s+is\s+None\s*:\r?\n"
    r"(?P=indent)[ \t]+[ \t]+output_hidden_states\s*=\s*\[\]\r?\n"
    r"(?P=indent)[ \t]+output_hidden_states\.append\(req\.hidden_states\)\r?\n",
)
_DECODE_FUNCTION_HEADER_PATTERN = re.compile(
    r"(?ms)^(?P<header>def\s+process_batch_result_decode\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:\r?\n)",
)
_DECODE_OFFSET_ALREADY_SET_PATTERN = re.compile(
    r"(?ms)^def\s+process_batch_result_decode\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:\r?\n"
    r"[ \t]+hidden_state_offset = 0\s*(?:#.*)?\r?\n",
)


def replace_sglang_hidden_states_list_output(source: str) -> tuple[str, int]:
    return _HIDDEN_STATES_LIST_OUTPUT_PATTERN.subn(
        '.detach().to("cpu", copy=True)',
        source,
    )


def _render_decode_hidden_states_append(match: re.Match[str]) -> str:
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


def _render_prefill_hidden_states_append(match: re.Match[str]) -> str:
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


def _render_stream_hidden_states(match: re.Match[str]) -> str:
    indent = match.group("indent")
    return (
        f"{indent}if _sglang_req_should_stream_hidden_states(req):\n"
        f"{indent}    if output_hidden_states is None:\n"
        f"{indent}        output_hidden_states = [[] for _ in range(len(rids) - 1)]\n"
        f"{indent}    output_hidden_states.append(req.hidden_states)\n"
        f"{indent}elif output_hidden_states is not None:\n"
        f"{indent}    output_hidden_states.append([])\n"
    )


def _insert_decode_hidden_state_offset(source: str) -> str | None:
    if _DECODE_OFFSET_ALREADY_SET_PATTERN.search(source):
        return source
    patched_source, function_count = _DECODE_FUNCTION_HEADER_PATTERN.subn(
        lambda match: f"{match.group('header')!s}    hidden_state_offset = 0\n",
        source,
        count=1,
    )
    if function_count <= 0:
        return None
    return patched_source


def patch_sglang_decode_hidden_states_source(source: str) -> str | None:
    patched_source, count = _DECODE_HIDDEN_STATES_APPEND_PATTERN.subn(
        _render_decode_hidden_states_append,
        source,
    )
    if count <= 0:
        return None
    return _insert_decode_hidden_state_offset(patched_source)


def patch_sglang_prefill_hidden_states_source(source: str) -> str | None:
    for pattern in (
        _PREFILL_HIDDEN_STATES_APPEND_PATTERN,
        _PREFILL_HIDDEN_STATES_HELPER_CALL_PATTERN,
    ):
        patched_source, count = pattern.subn(
            _render_prefill_hidden_states_append,
            source,
            count=1,
        )
        if count > 0:
            return patched_source
    return None


def patch_sglang_stream_hidden_states_source(source: str) -> str | None:
    patched_source, count = _STREAM_HIDDEN_STATES_PATTERN.subn(
        _render_stream_hidden_states,
        source,
        count=1,
    )
    if count <= 0:
        return None
    return patched_source


def patch_sglang_output_processor_method_source(
    method_name: str, source: str
) -> str | None:
    """Rewrite one dedented output-processor method source.

    Returns ``None`` when the method's hidden-state block is not found. The patched code
    calls ``_append_sglang_prefill_hidden_states``, ``_append_sglang_decode_hidden_states``
    and ``_sglang_req_should_stream_hidden_states``, which the caller must provide as
    globals.
    """
    patched_source, conversion_count = replace_sglang_hidden_states_list_output(source)
    if method_name == PREFILL_METHOD:
        result = patch_sglang_prefill_hidden_states_source(patched_source)
    elif method_name == DECODE_METHOD:
        result = patch_sglang_decode_hidden_states_source(patched_source)
    elif method_name == STREAM_METHOD:
        result = patch_sglang_stream_hidden_states_source(patched_source)
    else:
        result = patched_source if conversion_count > 0 else None
    if result is None or result == source:
        return None
    return result


def _capture_error(message: str) -> SGLangHiddenStateCapturePatchError:
    return SGLangHiddenStateCapturePatchError(
        f"{message} SGLang's native hidden-state capture would then feed misaligned rows "
        "to drafter training. Use an SGLang build with a supported layout, set "
        "drafter training.collect_hidden_states_from_sgl=False, or set "
        f"{DISABLE_SGLANG_PATCH_ENV}=1 to turn off all verl SGLang patches."
    )


def install_sglang_output_processor_capture_patches(
    *,
    import_module: Callable[[str], Any],
    make_patched_method: Callable[[Any], Any | None],
    patch_stream_accumulator: Callable[[], bool],
    layouts: tuple[SGLangOutputProcessorLayout, ...] = SGLANG_OUTPUT_PROCESSOR_LAYOUTS,
) -> dict[str, list[str]]:
    """Patch every importable output-processor layout, or raise.

    ``make_patched_method`` turns an original method into its patched version (or
    ``None`` when its hidden-state block is not found). ``patch_stream_accumulator``
    patches ``_GenerationStreamAccumulator.accept`` and returns whether it is active.
    Returns ``{layout name: active methods}``; an empty dict means the patches are
    disabled through ``VERL_DISABLE_SGLANG_PATCH``.
    """
    if sglang_verl_patches_disabled():
        return {}

    unavailable: list[str] = []
    active_by_layout: dict[str, list[str]] = {}
    for layout in layouts:
        try:
            processor_cls = getattr(import_module(layout.module), layout.class_name)
        except Exception as exc:  # noqa: BLE001
            unavailable.append(f"{layout.module}.{layout.class_name} ({exc!s})")
            continue

        active_methods: list[str] = []
        for method_name in layout.methods:
            original_method = getattr(processor_cls, method_name, None)
            if original_method is None:
                continue
            if getattr(original_method, PATCHED_METHOD_ATTR, False):
                active_methods.append(method_name)
                continue
            patched_method = make_patched_method(original_method)
            if patched_method is None:
                continue
            setattr(processor_cls, method_name, patched_method)
            active_methods.append(method_name)

        missing_methods = [m for m in layout.methods if m not in active_methods]
        if missing_methods:
            raise _capture_error(
                f"Failed to patch SGLang hidden-state capture in "
                f"{layout.module}.{layout.class_name}: {', '.join(missing_methods)} "
                "missing or not in a known shape."
            )
        if layout.stream_via_accumulator and not patch_stream_accumulator():
            raise _capture_error(
                "Failed to patch SGLang's stream accumulator hidden-state delivery for "
                f"{layout.module}.{layout.class_name}."
            )
        active_by_layout[layout.name] = active_methods

    if not active_by_layout:
        raise _capture_error(
            "No known SGLang output-processor layout found for hidden-state capture; "
            f"tried {'; '.join(unavailable)}."
        )
    return active_by_layout
