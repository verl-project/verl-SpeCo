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
"""SPECO's rewrite of SGLang's per-request hidden-state capture.

The processor classes below copy the hidden-state blocks (verbatim, with the surrounding
method trimmed to the variables those blocks use) from:

* SGLang 0.5.12 ``sglang/srt/managers/scheduler_output_processor_mixin.py``
  (``SchedulerOutputProcessorMixin``), the layout of the CUDA images;
* the SGLang build in the ROCm image (commit 0801cc05, reports 0.5.16)
  ``sglang/srt/managers/scheduler_components/batch_result_processor.py``
  (``SchedulerBatchResultProcessor``);
* public SGLang 0.5.16, whose decode block has a different shape.

Each test rewrites the real method source, runs it on a small batch with recording
collectors, and checks that every request's rows go through SPECO with its own offsets.
"""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from verl_speco.integration import sglang_hidden_state_capture as capture
from verl_speco.integration.sglang_hidden_state_capture import (
    PATCHED_METHOD_ATTR,
    SGLangHiddenStateCapturePatchError,
    install_sglang_output_processor_capture_patches,
    patch_sglang_output_processor_method_source,
    sglang_prefill_extend_rows,
)
from verl_speco.integration.sglang_patch_controls import DISABLE_SGLANG_PATCH_ENV


class _V0512Mixin:
    def process_batch_result_prefill(self, batch, result):
        logits_output, extend_input_len_per_req = (
            result.logits_output,
            result.extend_input_len_per_req,
        )
        hidden_state_offset = 0
        for i, req in enumerate(batch.reqs):
            if req.inflight_middle_chunks <= 0:
                if (
                    req.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    req.hidden_states.append(
                        logits_output.hidden_states[
                            hidden_state_offset : (
                                hidden_state_offset := hidden_state_offset
                                + len(req.origin_input_ids)
                            )
                        ]
                        .cpu()
                        .clone()
                        .tolist()
                    )

    def process_batch_result_decode(self, batch, result):
        logits_output = result.logits_output
        is_spec_v1 = not batch.spec_algorithm.is_none() and not batch.is_spec_v2
        for i, req in enumerate(batch.reqs):
            if is_spec_v1:
                if req.return_hidden_states and logits_output.hidden_states is not None:
                    req.hidden_states.append(
                        logits_output.hidden_states[i].cpu().clone().tolist()
                    )
                continue

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

    def stream_output_generation(self, reqs):
        rids = []
        output_hidden_states = None
        for req in reqs:
            rids.append(req.rid)
            if req.return_hidden_states:
                if output_hidden_states is None:
                    output_hidden_states = []
                output_hidden_states.append(req.hidden_states)
        return output_hidden_states


class _ImageBatchResultProcessor:
    def process_batch_result_prefill(self, batch, result):
        (
            logits_output,
            extend_input_len_per_req,
        ) = (
            result.logits_output,
            result.extend_input_len_per_req,
        )
        hidden_state_offset = 0
        for i, req in enumerate(batch.reqs):
            if req.inflight_middle_chunks <= 0:
                if (
                    req.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    hidden_state_offset = self._append_prefill_hidden_states(
                        req=req,
                        logits_output=logits_output,
                        hidden_state_offset=hidden_state_offset,
                    )

    def _append_prefill_hidden_states(self, *, req, logits_output, hidden_state_offset):
        req.hidden_states.append(
            logits_output.hidden_states[
                hidden_state_offset : (
                    hidden_state_offset := hidden_state_offset
                    + len(req.origin_input_ids)
                )
            ]
        )
        return hidden_state_offset

    def process_batch_result_decode(self, batch, result):
        logits_output = result.logits_output
        is_spec_v1 = not batch.spec_algorithm.is_none() and not batch.is_spec_v2
        for i, req in enumerate(batch.reqs):
            if is_spec_v1:
                if req.return_hidden_states and logits_output.hidden_states is not None:
                    req.hidden_states.append(
                        logits_output.hidden_states[i].cpu().clone().tolist()
                    )
                if req.grammar is not None:
                    req.grammar.finished = req.finished()
                continue

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )


class _V0516BatchResultProcessor:
    def process_batch_result_decode(self, batch, result):
        logits_output = result.logits_output
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, result.next_token_ids)):
            if req.return_hidden_states and logits_output.hidden_states is not None:
                # hidden_states is [bs * stride, hidden_dim], one row per emitted
                # token; stride = speculative_num_draft_tokens for spec, 1 for non-spec.
                stride = result.speculative_num_draft_tokens or 1
                accept_len = len(next_token_id)
                start = i * stride
                req.hidden_states.extend(
                    logits_output.hidden_states[start : start + accept_len]
                    .cpu()
                    .tolist()
                )


class _Recorder:
    """Stands in for SPECO's collectors; rows per request come from the batch result."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def prefill(
        self, req, logits_output, offset, extend_input_len, batch, *, req_index=None
    ):
        rows = sglang_prefill_extend_rows(req, extend_input_len, batch, req_index)
        self.calls.append(("prefill", req.rid, offset, rows))
        return offset + rows

    def decode(self, req, logits_output, result, i, offset, batch):
        self.calls.append(("decode", req.rid, i, offset))
        return offset + result.commit_lens[i]

    def globals(self) -> dict[str, Any]:
        return {
            "_append_sglang_prefill_hidden_states": self.prefill,
            "_append_sglang_decode_hidden_states": self.decode,
            "_sglang_req_should_stream_hidden_states": lambda req: bool(
                req.return_hidden_states and req.stream_hidden_states
            ),
        }


def _patched(cls: type, method_name: str, recorder: _Recorder):
    source = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    patched_source = patch_sglang_output_processor_method_source(method_name, source)
    assert patched_source is not None, f"{cls.__name__}.{method_name} not patched"
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        recorder.globals(),
        namespace,
    )
    return namespace[method_name]


class _Unreachable:
    """Native capture path; touching it means the rewrite missed a block."""

    def __getitem__(self, item):
        raise AssertionError("native SGLang hidden-state capture ran")


def _req(
    rid: str,
    *,
    prompt_len: int,
    collect: bool = True,
    stream: bool = True,
    extend_input_len: int | None = None,
):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(prompt_len)),
        extend_input_len=extend_input_len,
        return_hidden_states=collect,
        stream_hidden_states=stream,
        hidden_states=[],
        inflight_middle_chunks=0,
        grammar=None,
    )


def _spec(*, is_none: bool):
    return SimpleNamespace(is_none=lambda: is_none)


PROCESSORS = [_V0512Mixin, _ImageBatchResultProcessor]


@pytest.mark.parametrize("processor_cls", PROCESSORS)
def test_prefill_uses_extend_rows_not_full_prompt(processor_cls) -> None:
    # Prefix-cache hits: only the extend rows are in the batch, not the full prompt.
    # Requests that do not collect still own rows, so the collector sees every request.
    recorder = _Recorder()
    prefill = _patched(processor_cls, "process_batch_result_prefill", recorder)
    reqs = [
        _req("a", prompt_len=10),
        _req("skip", prompt_len=4, collect=False),
        _req("b", prompt_len=12),
    ]
    batch = SimpleNamespace(reqs=reqs)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=_Unreachable()),
        extend_input_len_per_req=[3, 4, 5],
    )

    prefill(processor_cls(), batch, result)

    assert recorder.calls == [
        ("prefill", "a", 0, 3),
        ("prefill", "skip", 3, 4),
        ("prefill", "b", 7, 5),
    ]


@pytest.mark.parametrize("processor_cls", PROCESSORS)
@pytest.mark.parametrize("overlap_copy", [False, True])
def test_prefill_without_logprob_lengths_uses_batch_extend_lens(
    processor_cls, overlap_copy
) -> None:
    # SGLang sets extend_input_len_per_req only for logprob batches, and DFlash rejects
    # logprobs. The batch's extend_lens (dropped by the overlap-schedule copy, where the
    # request's extend_input_len is still set) give the real rows per request.
    recorder = _Recorder()
    prefill = _patched(processor_cls, "process_batch_result_prefill", recorder)
    reqs = [
        _req("a", prompt_len=94, extend_input_len=40),
        _req("b", prompt_len=88, extend_input_len=88),
        _req("c", prompt_len=88, extend_input_len=24),
    ]
    batch = SimpleNamespace(
        reqs=reqs, extend_lens=None if overlap_copy else [40, 88, 24]
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=_Unreachable()),
        extend_input_len_per_req=None,
    )

    prefill(processor_cls(), batch, result)

    assert recorder.calls == [
        ("prefill", "a", 0, 40),
        ("prefill", "b", 40, 88),
        ("prefill", "c", 128, 24),
    ]


def test_prefill_extend_rows_order() -> None:
    req = _req("a", prompt_len=94, extend_input_len=30)
    other = _req("b", prompt_len=10, extend_input_len=10)
    batch = SimpleNamespace(reqs=[req, other], extend_lens=[40, 10])

    assert sglang_prefill_extend_rows(req, 7, batch, 0) == 7
    assert sglang_prefill_extend_rows(req, None, batch, 0) == 40
    # Batch lists that do not line up with this request are not trusted.
    assert sglang_prefill_extend_rows(req, None, batch, 1) == 30
    assert (
        sglang_prefill_extend_rows(req, None, SimpleNamespace(reqs=[req]), 0) == 30
    )
    assert sglang_prefill_extend_rows(_req("c", prompt_len=12), None, None) == 12


@pytest.mark.parametrize("processor_cls", PROCESSORS)
@pytest.mark.parametrize("spec_v1", [True, False])
def test_decode_routes_every_request_with_running_offset(
    processor_cls, spec_v1
) -> None:
    # DFlash verify compacts all requests' committed rows into one tensor; each request
    # must read its own slice, not ``hidden_states[i]``.
    recorder = _Recorder()
    decode = _patched(processor_cls, "process_batch_result_decode", recorder)
    reqs = [_req("a", prompt_len=4), _req("b", prompt_len=4), _req("c", prompt_len=4)]
    batch = SimpleNamespace(
        reqs=reqs, spec_algorithm=_spec(is_none=False), is_spec_v2=not spec_v1
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=_Unreachable()),
        commit_lens=[2, 4, 1],
    )

    decode(processor_cls(), batch, result)

    assert recorder.calls == [
        ("decode", "a", 0, 0),
        ("decode", "b", 1, 2),
        ("decode", "c", 2, 6),
    ]


def test_v0512_stream_keeps_hidden_states_aligned_with_rids() -> None:
    recorder = _Recorder()
    stream = _patched(_V0512Mixin, "stream_output_generation", recorder)
    reqs = [
        _req("plain", prompt_len=1, collect=False),
        _req("a", prompt_len=1),
        _req("window", prompt_len=1, stream=False),
    ]
    reqs[1].hidden_states = ["chunk"]

    assert stream(_V0512Mixin(), reqs) == [[], ["chunk"], []]


def test_v0516_public_decode_shape_is_not_patched() -> None:
    source = textwrap.dedent(
        inspect.getsource(_V0516BatchResultProcessor.process_batch_result_decode)
    )
    assert (
        patch_sglang_output_processor_method_source(
            "process_batch_result_decode", source
        )
        is None
    )


def _module(cls: type, name: str) -> SimpleNamespace:
    return SimpleNamespace(**{name: cls})


def _fresh(cls: type) -> type:
    return type(cls.__name__, (cls,), {})


def _make_patched_method(original_method):
    source = textwrap.dedent(inspect.getsource(original_method))
    patched_source = patch_sglang_output_processor_method_source(
        original_method.__name__, source
    )
    if patched_source is None:
        return None
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        _Recorder().globals(),
        namespace,
    )
    method = namespace[original_method.__name__]
    setattr(method, PATCHED_METHOD_ATTR, True)
    return method


_OLD = capture.SGLANG_OUTPUT_PROCESSOR_LAYOUTS[0]
_NEW = capture.SGLANG_OUTPUT_PROCESSOR_LAYOUTS[1]


def _importer(available: dict[str, Any]):
    def import_module(name: str):
        if name not in available:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return available[name]

    return import_module


@pytest.fixture(autouse=True)
def _patches_enabled(monkeypatch) -> None:
    monkeypatch.delenv(DISABLE_SGLANG_PATCH_ENV, raising=False)


def test_install_patches_old_layout_without_accumulator() -> None:
    cls = _fresh(_V0512Mixin)
    active = install_sglang_output_processor_capture_patches(
        import_module=_importer({_OLD.module: _module(cls, _OLD.class_name)}),
        make_patched_method=_make_patched_method,
        patch_stream_accumulator=lambda: False,
    )

    assert active == {_OLD.name: list(_OLD.methods)}
    for method_name in _OLD.methods:
        assert getattr(getattr(cls, method_name), PATCHED_METHOD_ATTR, False)


def test_install_patches_image_layout_and_requires_accumulator() -> None:
    cls = _fresh(_ImageBatchResultProcessor)
    import_module = _importer({_NEW.module: _module(cls, _NEW.class_name)})
    with pytest.raises(SGLangHiddenStateCapturePatchError, match="stream accumulator"):
        install_sglang_output_processor_capture_patches(
            import_module=import_module,
            make_patched_method=_make_patched_method,
            patch_stream_accumulator=lambda: False,
        )

    active = install_sglang_output_processor_capture_patches(
        import_module=import_module,
        make_patched_method=_make_patched_method,
        patch_stream_accumulator=lambda: True,
    )
    assert active == {_NEW.name: list(_NEW.methods)}
    for method_name in _NEW.methods:
        assert getattr(getattr(cls, method_name), PATCHED_METHOD_ATTR, False)


def test_install_fails_loudly_when_no_layout_is_importable() -> None:
    with pytest.raises(SGLangHiddenStateCapturePatchError) as exc_info:
        install_sglang_output_processor_capture_patches(
            import_module=_importer({}),
            make_patched_method=_make_patched_method,
            patch_stream_accumulator=lambda: True,
        )
    message = str(exc_info.value)
    assert _OLD.module in message and _NEW.module in message
    assert DISABLE_SGLANG_PATCH_ENV in message
    assert "collect_hidden_states_from_sgl=False" in message


def test_install_fails_loudly_on_unknown_method_shape() -> None:
    cls = _fresh(_ImageBatchResultProcessor)
    cls.process_batch_result_decode = (
        _V0516BatchResultProcessor.process_batch_result_decode
    )
    with pytest.raises(
        SGLangHiddenStateCapturePatchError, match="process_batch_result_decode"
    ):
        install_sglang_output_processor_capture_patches(
            import_module=_importer({_NEW.module: _module(cls, _NEW.class_name)}),
            make_patched_method=_make_patched_method,
            patch_stream_accumulator=lambda: True,
        )


def test_disable_env_skips_install(monkeypatch) -> None:
    monkeypatch.setenv(DISABLE_SGLANG_PATCH_ENV, "1")

    def import_module(name: str):
        raise AssertionError("must not import SGLang when patches are disabled")

    assert (
        install_sglang_output_processor_capture_patches(
            import_module=import_module,
            make_patched_method=_make_patched_method,
            patch_stream_accumulator=lambda: True,
        )
        == {}
    )
