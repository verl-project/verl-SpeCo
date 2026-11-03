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
"""Per-request hidden-state output patch on real SGLang output-processor sources.

The processors under test are vendored verbatim in ``sglang_sources/`` (see the
provenance header of each file):

* ``scheduler_output_processor_mixin_v0_5_9`` / ``_v0_5_10`` / ``_v0_5_12``:
  the scheduler mixin layout, up to v0.5.12. Prefill slices
  ``len(req.origin_input_ids)`` rows inline; decode appends ``hidden_states[i]``;
  ``stream_output_generation`` streams ``req.hidden_states``.
* ``batch_result_processor_0801cc05``: sgl-project/sglang main at 0801cc05, the
  build in the ROCm image (it reports 0.5.16). Decode appends
  ``hidden_states[i]`` per request, while DFlash verify packs the committed rows
  of all requests as ``[sum(commit_lens), D]``.
* ``batch_result_processor_v0_5_14``: the v0.5.14 tag. Speculative decode
  extends ``hidden_states[i * stride : i * stride + accept_len]``;
  non-speculative decode still appends ``hidden_states[i]``.
* ``batch_result_processor_v0_5_16``: the v0.5.16 tag. All decode uses the
  strided extend.

* ``output_streamer_v0_5_16``: the v0.5.16 ``SchedulerOutputStreamer``. From
  v0.5.13 it sends ``req.hidden_states`` only while ``req.return_hidden_states``
  is set, sliced to ``req.finished_len`` entries.

* ``dflash_worker_v2_v0_5_16``: the v0.5.16 DFlash spec-v2 worker (the only
  DFlash worker from v0.5.14 on). It drops the target hidden states after
  writing them into the draft KV cache; the tests run its real
  ``forward_batch_generation`` with the patch and feed the result to the real
  v0.5.16 processor.

Every layout slices ``len(req.origin_input_ids)`` prefill rows per request,
which ignores prefix-cache hits.

The per-request tests run the real, patched ``process_batch_result_prefill`` and
``process_batch_result_decode`` on hidden states whose every row encodes its
(request, position). Each request's collected chunks then go through the
runtime's chunk decoding and the block-drafter position check, and the rows and
positions are compared to the exact reference.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# sglang_patch imports torch at module level; the CPU CI image has none.
torch = pytest.importorskip("torch")

_NEW_MODULE = "sglang.srt.managers.scheduler_components.batch_result_processor"
_OLD_MODULE = "sglang.srt.managers.scheduler_output_processor_mixin"
_DFLASH_INFO_MODULE = "sglang.srt.speculative.dflash_info"
_STREAMER_MODULE = "sglang.srt.managers.scheduler_components.output_streamer"
_STREAMER_SOURCE = "output_streamer_v0_5_16.py.txt"

_SOURCES_DIR = Path(__file__).parent / "sglang_sources"
_HEADER_END = "# --- end of verl-SpeCo header ---\n"
_IMAGE_SOURCE = "batch_result_processor_0801cc05.py.txt"
_V0_5_14_SOURCE = "batch_result_processor_v0_5_14.py.txt"
_V0_5_16_SOURCE = "batch_result_processor_v0_5_16.py.txt"
_MIXIN_V0_5_9_SOURCE = "scheduler_output_processor_mixin_v0_5_9.py.txt"
_MIXIN_V0_5_10_SOURCE = "scheduler_output_processor_mixin_v0_5_10.py.txt"
_MIXIN_SOURCE = "scheduler_output_processor_mixin_v0_5_12.py.txt"
_MIXIN_SOURCES = (_MIXIN_V0_5_9_SOURCE, _MIXIN_V0_5_10_SOURCE, _MIXIN_SOURCE)
_NEW_CLASS = "SchedulerBatchResultProcessor"
_MIXIN_CLASS = "SchedulerOutputProcessorMixin"
_ALL_SOURCES = (
    _IMAGE_SOURCE,
    _V0_5_14_SOURCE,
    _V0_5_16_SOURCE,
    _STREAMER_SOURCE,
    *_MIXIN_SOURCES,
)

_STUBBED = (
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.utils",
    "sglang.srt.utils.patch_torch",
    "sglang.srt.managers",
    "sglang.srt.managers.scheduler_components",
    "sglang.srt.speculative",
)

_DFLASH_COLLECT_CONFIG = {
    "enable": True,
    "speculative_algorithm": "DFLASH",
    "training": {"collect_hidden_states_from_sgl": True},
}

# DFlash verify block size (speculative_num_draft_tokens).
_STRIDE = 4
_HIDDEN_DIM = 3


def _vendored(file_name: str) -> tuple[str, str]:
    header, body = (_SOURCES_DIR / file_name).read_text().split(_HEADER_END, 1)
    return header, body


def _stub_sglang_imports(body: str) -> str:
    """Bind every top-level ``from sglang... import`` name to a MagicMock.

    The method sources stay byte-identical, so the rewrite sees the real code.
    """
    tree = ast.parse(body)
    lines = body.splitlines(keepends=True)
    replacements: dict[int, tuple[int, str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom):
            if not (node.module or "").startswith("sglang"):
                continue
        elif not any(alias.name.startswith("sglang") for alias in node.names):
            continue
        names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        stub = "".join(f"{name} = _SGLangStub(name={name!r})\n" for name in names)
        replacements[node.lineno - 1] = (node.end_lineno, stub)
    out: list[str] = []
    index = 0
    while index < len(lines):
        if index in replacements:
            end, stub = replacements[index]
            out.append("from unittest.mock import MagicMock as _SGLangStub\n" + stub)
            index = end
            continue
        out.append(lines[index])
        index += 1
    return "".join(out)


def _load_module(monkeypatch, tmp_path, module_name: str, source: str, file_stem: str):
    path = tmp_path / f"{file_stem}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while the class is built.
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_vendored(monkeypatch, tmp_path, module_name: str, file_name: str):
    _, body = _vendored(file_name)
    return _load_module(
        monkeypatch,
        tmp_path,
        module_name,
        _stub_sglang_imports(body),
        file_name.split(".", 1)[0],
    )


def _install_source(monkeypatch, tmp_path, module_name: str, source: str):
    return _load_module(
        monkeypatch,
        tmp_path,
        module_name,
        textwrap.dedent(source),
        module_name.rpartition(".")[2],
    )


@pytest.fixture
def stub_sglang(monkeypatch):
    stubs = {name: types.ModuleType(name) for name in _STUBBED}
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, module)
    stubs["sglang.srt.entrypoints.engine"].run_scheduler_process = lambda *a, **k: None
    stubs["sglang.srt.utils"].MultiprocessingSerializer = types.SimpleNamespace(
        deserialize=lambda payload: payload
    )
    stubs["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    monkeypatch.setitem(sys.modules, _OLD_MODULE, None)
    monkeypatch.setitem(sys.modules, _NEW_MODULE, None)
    monkeypatch.setitem(sys.modules, _DFLASH_INFO_MODULE, None)
    monkeypatch.setitem(sys.modules, _STREAMER_MODULE, None)
    monkeypatch.delenv("VERL_SPECO_DRAFTER_CONFIG", raising=False)
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    return stubs


def _fresh_sglang_patch(monkeypatch):
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )
    return importlib.import_module("verl_speco.integration.sglang_patch")


def _import_sglang_patch(monkeypatch):
    module = _fresh_sglang_patch(monkeypatch)
    monkeypatch.setattr(module, "_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED", False)
    # The installer also applies the verify/logits-processor patches first;
    # they target other SGLang modules and are not under test here.
    for name in (
        "patch_sglang_eagle_legacy_alignment_compat",
        "patch_sglang_eagle_verify_hidden_states_full",
        "patch_sglang_dflash_verify_hidden_states",
        "patch_sglang_raw_top_logprobs_output",
        "patch_sglang_drafter_last_hidden_output",
        "_patch_sglang_drafter_output",
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    return module


# --- vendored sources -------------------------------------------------------


@pytest.mark.parametrize("file_name", _ALL_SOURCES)
def test_vendored_source_is_verbatim(file_name) -> None:
    header, body = _vendored(file_name)
    match = re.search(r"SHA-256 of the upstream file: ([0-9a-f]{64})", header)
    assert match is not None
    assert hashlib.sha256(body.encode()).hexdigest() == match.group(1)
    assert re.search(r"Commit: [0-9a-f]{40} ", header)
    assert "Apache License, Version 2.0" in header


def _method_source(file_name: str, class_name: str, method_name: str) -> str:
    _, body = _vendored(file_name)
    tree = ast.parse(body)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    segment = ast.get_source_segment(body, item, padded=True)
                    return textwrap.dedent(segment)
    raise AssertionError(f"{class_name}.{method_name} not in {file_name}")


@pytest.mark.parametrize(
    ("file_name", "class_name", "method_name", "helper_counts"),
    [
        (
            _IMAGE_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_prefill",
            {"_append_sglang_prefill_hidden_states(": 1},
        ),
        (
            _IMAGE_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_decode",
            {"_append_sglang_decode_hidden_states(": 2},
        ),
        (
            _V0_5_14_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_prefill",
            {"_append_sglang_prefill_hidden_states(": 1},
        ),
        (
            _V0_5_14_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_decode",
            {"_append_sglang_strided_decode_hidden_states(": 2},
        ),
        (
            _V0_5_16_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_prefill",
            {"_append_sglang_prefill_hidden_states(": 1},
        ),
        (
            _V0_5_16_SOURCE,
            "SchedulerBatchResultProcessor",
            "process_batch_result_decode",
            {"_append_sglang_strided_decode_hidden_states(": 1},
        ),
        *(
            (
                file_name,
                "SchedulerOutputProcessorMixin",
                "process_batch_result_prefill",
                {"_append_sglang_prefill_hidden_states(": 1},
            )
            for file_name in _MIXIN_SOURCES
        ),
        (
            _MIXIN_V0_5_9_SOURCE,
            "SchedulerOutputProcessorMixin",
            "process_batch_result_decode",
            {"_append_sglang_decode_hidden_states(": 1},
        ),
        (
            _MIXIN_V0_5_10_SOURCE,
            "SchedulerOutputProcessorMixin",
            "process_batch_result_decode",
            {"_append_sglang_decode_hidden_states(": 1},
        ),
        (
            _MIXIN_SOURCE,
            "SchedulerOutputProcessorMixin",
            "process_batch_result_decode",
            {"_append_sglang_decode_hidden_states(": 2},
        ),
    ],
)
def test_rewrite_replaces_every_native_hidden_state_site(
    monkeypatch, stub_sglang, file_name, class_name, method_name, helper_counts
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    source = _method_source(file_name, class_name, method_name)
    if method_name == "process_batch_result_prefill":
        patched = sglang_patch._patch_sglang_prefill_hidden_states_source(source)
    else:
        patched = sglang_patch._patch_sglang_decode_hidden_states_source(source)

    assert patched is not None
    for helper, count in helper_counts.items():
        assert patched.count(helper) == count, helper
    # No native per-request hidden-state append or extend is left behind.
    assert "req.hidden_states.append(" not in patched
    assert "req.hidden_states.extend(" not in patched
    compile(patched, file_name, "exec")


@pytest.mark.parametrize("file_name", _MIXIN_SOURCES)
def test_rewrite_streams_hidden_states_of_every_request(
    monkeypatch, stub_sglang, file_name
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    source = _method_source(
        file_name, "SchedulerOutputProcessorMixin", "stream_output_generation"
    )
    patched = sglang_patch._patch_sglang_stream_hidden_states_source(source)

    assert patched is not None
    # req.hidden_states (the per-request chunks, with their positions) is
    # streamed unchanged; requests without hidden states get an empty entry so
    # the list stays aligned with rids.
    assert patched.count("output_hidden_states.append(req.hidden_states)") == 1
    assert patched.count("output_hidden_states.append([])") == 1
    compile(patched, file_name, "exec")


@pytest.mark.parametrize(
    ("file_name", "module_name", "class_name", "methods"),
    [
        (
            _IMAGE_SOURCE,
            _NEW_MODULE,
            "SchedulerBatchResultProcessor",
            ("process_batch_result_prefill", "process_batch_result_decode"),
        ),
        (
            _V0_5_14_SOURCE,
            _NEW_MODULE,
            "SchedulerBatchResultProcessor",
            ("process_batch_result_prefill", "process_batch_result_decode"),
        ),
        (
            _V0_5_16_SOURCE,
            _NEW_MODULE,
            "SchedulerBatchResultProcessor",
            ("process_batch_result_prefill", "process_batch_result_decode"),
        ),
        *(
            (
                file_name,
                _OLD_MODULE,
                "SchedulerOutputProcessorMixin",
                (
                    "process_batch_result_prefill",
                    "process_batch_result_decode",
                    "stream_output_generation",
                ),
            )
            for file_name in _MIXIN_SOURCES
        ),
    ],
)
def test_installer_patches_real_layout(
    monkeypatch, tmp_path, stub_sglang, file_name, module_name, class_name, methods
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    module = _install_vendored(monkeypatch, tmp_path, module_name, file_name)
    streamer_module = None
    if module_name == _NEW_MODULE:
        # 0801cc05 and v0.5.14 share the streamer method the patch wraps.
        streamer_module = _install_vendored(
            monkeypatch, tmp_path, _STREAMER_MODULE, _STREAMER_SOURCE
        )

    sglang_patch.patch_sglang_hidden_states_tensor_output()

    cls = getattr(module, class_name)
    for name in methods:
        assert getattr(cls, name)._verl_patched_hidden_states_tensor_output
    assert sglang_patch._SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED
    if streamer_module is not None:
        streamer = streamer_module.SchedulerOutputStreamer._stream_output_generation
        assert streamer._verl_patched_stream_final_hidden_states


def test_new_layout_without_streamer_fails_closed_when_collecting(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    _install_vendored(monkeypatch, tmp_path, _NEW_MODULE, _V0_5_16_SOURCE)

    with pytest.raises(RuntimeError, match="output streamer hidden-state patch"):
        sglang_patch.patch_sglang_hidden_states_tensor_output()


# --- deterministic per-request alignment ------------------------------------


def _ref_row(request_id: int, position: int) -> list[float]:
    # Every row encodes the (request, position) it belongs to. Values stay
    # below 256 so they are exact after the runtime's bfloat16 conversion.
    assert 0 <= position < 64
    return [float(request_id * 64 + position)] * _HIDDEN_DIM


_FILLER_ROW = [-1.0] * _HIDDEN_DIM


class _Req:
    def __init__(self, request_id: int, prompt_len: int, cached: int) -> None:
        self.request_id = request_id
        self.rid = f"r{request_id}"
        self.origin_input_ids = list(range(prompt_len))
        self.extend_input_len = prompt_len - cached
        self.return_hidden_states = True
        self.hidden_states: list = []
        self.output_ids: list = []
        self.is_retracted = False
        self.inflight_middle_chunks = 0
        self.time_stats = MagicMock()
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        # Scheduler mixin (v0.5.12 and older) fields.
        self.is_chunked = 0
        self.multimodal_inputs = None
        self.session = None

    def finished(self) -> bool:
        return False

    def update_finish_state(self, *args) -> None:
        return None

    def check_finished(self, *args) -> None:
        return None


class _SpecAlgorithm:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def is_none(self) -> bool:
        return not self.enabled


class _Batch:
    def __init__(self, reqs, *, extend_lens=None, spec: bool = True) -> None:
        self.reqs = reqs
        self.extend_lens = extend_lens
        self.decoding_reqs = None
        self.return_logprob = False
        self.is_spec_v2 = False
        self.spec_algorithm = _SpecAlgorithm(spec)
        self.prefill_stats = None
        self.dp_cooperation_info = None

    def batch_size(self) -> int:
        return len(self.reqs)


class _Result:
    def __init__(self, logits_output, **kwargs) -> None:
        self.logits_output = logits_output
        self.next_token_ids = None
        self.copy_done = None
        self.routed_experts_output = None
        self.indexer_topk_output = None
        self.can_run_cuda_graph = False
        self.extend_input_len_per_req = None
        self.extend_logprob_start_len_per_req = None
        self.num_correct_drafts = 0
        self.num_block_accept_tokens = 0
        self.num_cap_tokens = 0
        self.speculative_num_draft_tokens = _STRIDE
        for key, value in kwargs.items():
            setattr(self, key, value)


def _processor_self(next_token_ids=None):
    processor = MagicMock()
    processor.is_generation = True
    processor.enable_overlap = False
    processor.enable_overlap_mlx = False
    processor.server_args.enable_hisparse = False
    processor.server_args.enable_metrics = False
    processor._normalize_decode_outputs.return_value = (next_token_ids, None)
    return processor


def _collected(req):
    """Decode req.hidden_states the way the rollout server does."""
    from verl_speco.integration.sglang_runtime import (
        _block_drafter_hidden_alignment_failure,
        _hidden_state_chunk_to_tensor_and_metadata,
        _iter_hidden_state_chunks,
    )

    rows, positions = [], []
    for chunk in _iter_hidden_state_chunks(req.hidden_states):
        tensor, metadata = _hidden_state_chunk_to_tensor_and_metadata(chunk)
        assert tensor is not None
        assert torch.is_tensor(metadata.get("positions")), metadata
        rows.append(tensor)
        positions.append(metadata["positions"])
    rows_t = torch.cat(rows, dim=0)
    positions_t = torch.cat(positions, dim=0).to(torch.long)
    assert _block_drafter_hidden_alignment_failure(positions_t, rows_t.shape[0]) is None
    return rows_t, positions_t


def _run_prefill(cls, reqs, *, extend_lens_on_batch: bool) -> None:
    packed = [
        _ref_row(req.request_id, pos)
        for req in reqs
        for pos in range(len(req.origin_input_ids) - req.extend_input_len, len(req.origin_input_ids))
    ]
    batch = _Batch(
        reqs,
        extend_lens=[req.extend_input_len for req in reqs] if extend_lens_on_batch else None,
    )
    result = _Result(
        types.SimpleNamespace(hidden_states=torch.tensor(packed), customized_info=None),
        next_token_ids=torch.zeros(len(reqs), dtype=torch.long),
    )
    cls.process_batch_result_prefill(_processor_self(), batch, result)


def _run_decode_step(cls, reqs, commit_lens, next_positions, *, mode: str) -> None:
    spec = mode != "non_spec"
    if mode in ("strided", "non_spec"):
        # [bs * stride, D]: request i owns rows i * stride onward. Rows past its
        # commit length are rejected drafts.
        stride = _STRIDE if spec else 1
        rows = [_FILLER_ROW] * (stride * len(reqs))
        for index, (req, commit) in enumerate(zip(reqs, commit_lens, strict=True)):
            for offset in range(commit):
                rows[index * stride + offset] = _ref_row(
                    req.request_id, next_positions[req.request_id] + offset
                )
        logits_output = types.SimpleNamespace(
            hidden_states=torch.tensor(rows), customized_info=None
        )
        extra = {
            "accept_lens": list(commit_lens),
            "speculative_num_draft_tokens": _STRIDE if spec else None,
        }
    else:
        # Spec-v1 verify output with only the committed rows, packed as
        # [sum(commit_lens), D]: SPECO's DFlash verify patch ("packed") or
        # SGLang's EAGLE verify ("packed_eagle").
        rows = [
            _ref_row(req.request_id, next_positions[req.request_id] + offset)
            for req, commit in zip(reqs, commit_lens, strict=True)
            for offset in range(commit)
        ]
        logits_output = types.SimpleNamespace(
            hidden_states=torch.tensor(rows), customized_info=None
        )
        if mode == "packed":
            logits_output._verl_dflash_aux_hidden_states = True
        drafts = [n - 1 for n in commit_lens]
        extra = {
            # v0.5.12 and later name the per-request draft count
            # num_correct_drafts_per_req_cpu; older releases use
            # accept_length_per_req_cpu.
            "num_correct_drafts_per_req_cpu": drafts,
            "accept_length_per_req_cpu": drafts,
            "num_accepted_tokens": sum(drafts),
            "next_token_ids": torch.zeros(sum(commit_lens), dtype=torch.long),
        }
    for req, commit in zip(reqs, commit_lens, strict=True):
        next_positions[req.request_id] += commit
    next_token_ids = [[0] * commit for commit in commit_lens]
    cls.process_batch_result_decode(
        _processor_self(next_token_ids),
        _Batch(reqs, spec=spec),
        _Result(logits_output, **extra),
    )


@pytest.mark.parametrize(
    ("file_name", "module_name", "class_name", "mode"),
    [
        (_MIXIN_V0_5_9_SOURCE, _OLD_MODULE, _MIXIN_CLASS, "packed_eagle"),
        (_MIXIN_V0_5_10_SOURCE, _OLD_MODULE, _MIXIN_CLASS, "packed_eagle"),
        (_MIXIN_SOURCE, _OLD_MODULE, _MIXIN_CLASS, "packed_eagle"),
        (_MIXIN_SOURCE, _OLD_MODULE, _MIXIN_CLASS, "packed"),
        # The image build; DFlash runs spec-v1 with SPECO's verify patch.
        (_IMAGE_SOURCE, _NEW_MODULE, _NEW_CLASS, "packed"),
        (_V0_5_14_SOURCE, _NEW_MODULE, _NEW_CLASS, "strided"),
        (_V0_5_14_SOURCE, _NEW_MODULE, _NEW_CLASS, "non_spec"),
        (_V0_5_16_SOURCE, _NEW_MODULE, _NEW_CLASS, "strided"),
        (_V0_5_16_SOURCE, _NEW_MODULE, _NEW_CLASS, "non_spec"),
    ],
)
@pytest.mark.parametrize("extend_lens_on_batch", [True, False])
def test_real_processor_collects_exact_rows_per_request(
    monkeypatch,
    tmp_path,
    stub_sglang,
    file_name,
    module_name,
    class_name,
    mode,
    extend_lens_on_batch,
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_vendored(monkeypatch, tmp_path, module_name, file_name)
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    cls = getattr(module, class_name)

    # Request 1 hits 5 of its 9 prompt tokens in the prefix cache, so the packed
    # prefill tensor holds only its last 4 rows. Without logprobs (DFlash),
    # extend_input_len_per_req is None, and slicing the prompt length would
    # shift request 2's rows.
    reqs = [_Req(0, 6, 0), _Req(1, 9, 5), _Req(2, 3, 0)]
    _run_prefill(cls, reqs, extend_lens_on_batch=extend_lens_on_batch)

    next_positions = {req.request_id: len(req.origin_input_ids) for req in reqs}
    steps = (
        ([1, 1, 1],) * 3
        if mode == "non_spec"
        else ([3, 1, 4], [1, 4, 2], [2, 2, 1])
    )
    for commit_lens in steps:
        _run_decode_step(cls, reqs, commit_lens, next_positions, mode=mode)

    for req in reqs:
        cached = len(req.origin_input_ids) - req.extend_input_len
        expected_positions = list(range(cached, next_positions[req.request_id]))
        expected_rows = torch.tensor(
            [_ref_row(req.request_id, pos) for pos in expected_positions]
        )
        rows, positions = _collected(req)
        assert positions.tolist() == expected_positions, req.rid
        assert torch.equal(rows, expected_rows), req.rid


def _finish_for_streaming(req, finished_len: int) -> None:
    """Give a ``_Req`` the fields the v0.5.16 streamer reads, finished."""
    req.finished = lambda: True
    req.finished_output = False
    req.finished_len = finished_len
    req.finished_reason = None
    req.stream = False
    req.http_worker_ipc = None
    req.decoded_text = ""
    req.init_incremental_detokenize = lambda: ([], 0)
    req.send_decode_id_offset = 0
    req.send_token_offset = 0
    req.send_output_token_logprobs_offset = 0
    req.output_ids_through_stop = req.output_ids[:finished_len]
    req.sampling_params = MagicMock()
    req.reasoning_tokens = 0
    req.cached_tokens = 0
    req.cached_tokens_device = 0
    req.cached_tokens_host = 0
    req.cached_tokens_storage = 0
    req.mm_image_tokens = req.mm_audio_tokens = req.mm_video_tokens = 0
    req.retraction_count = 0
    req.return_routed_experts = False
    req.return_indexer_topk = False
    req.customized_info = None
    for name in (
        "spec_verify_ct",
        "spec_num_correct_drafts",
        "spec_num_block_accept_tokens",
        "spec_num_cap_tokens",
        "spec_correct_drafts_histogram",
        "spec_cap_lens_histogram",
    ):
        setattr(req, name, 0)


def test_streamer_returns_hidden_states_of_request_whose_budget_filled(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    """Real v0.5.16 processor and streamer, one request capped by a row budget.

    Reaching the budget stops the capture by clearing ``return_hidden_states``,
    which the v0.5.16 streamer reads when it sends the final output.
    """
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_vendored(monkeypatch, tmp_path, _NEW_MODULE, _V0_5_16_SOURCE)
    streamer_module = _install_vendored(
        monkeypatch, tmp_path, _STREAMER_MODULE, _STREAMER_SOURCE
    )
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    cls = module.SchedulerBatchResultProcessor

    capped, uncapped = _Req(0, 6, 0), _Req(1, 4, 0)
    capped._verl_hidden_state_max_rows = 8
    reqs = [capped, uncapped]
    _run_prefill(cls, reqs, extend_lens_on_batch=True)
    next_positions = {req.request_id: len(req.origin_input_ids) for req in reqs}
    for commit_lens in ([3, 1], [2, 4]):
        _run_decode_step(cls, reqs, commit_lens, next_positions, mode="strided")
    # The budget is full after the first decode step.
    assert capped.return_hidden_states is False
    assert uncapped.return_hidden_states is True

    # A stop token on the second output token: one entry per token would keep
    # only the first two chunks.
    for req in reqs:
        _finish_for_streaming(req, finished_len=2)
    capped_chunks = list(capped.hidden_states)
    assert len(uncapped.hidden_states) == 3

    def stream(method):
        sent = MagicMock()
        streamer = streamer_module.SchedulerOutputStreamer(
            send_to_detokenizer=sent,
            tree_cache=MagicMock(),
            ps=MagicMock(),
            server_args=MagicMock(stream_interval=1),
            is_generation=True,
            spec_algorithm=_SpecAlgorithm(True),
            disaggregation_mode=MagicMock(),
            enable_hicache_storage=lambda: False,
        )
        for req in reqs:
            req.finished_output = False
        method(streamer, reqs, False)
        return streamer_module.BatchTokenIDOutput.call_args.kwargs[
            "output_hidden_states"
        ]

    patched = streamer_module.SchedulerOutputStreamer._stream_output_generation
    # Upstream v0.5.16 drops the capped request and slices the other.
    upstream = stream(patched.__wrapped__)
    assert upstream[0] is None
    assert len(upstream[1]) == 2

    sent_hidden = stream(patched)
    assert type(sent_hidden[0]) is list and type(sent_hidden[1]) is list
    for req, chunks in zip(reqs, sent_hidden, strict=True):
        rows, positions = _collected(types.SimpleNamespace(hidden_states=chunks))
        stop = 8 if req is capped else next_positions[req.request_id]
        expected_positions = list(range(0, stop))
        assert positions.tolist() == expected_positions, req.rid
        assert torch.equal(
            rows,
            torch.tensor([_ref_row(req.request_id, pos) for pos in expected_positions]),
        ), req.rid
    # The request keeps its own state.
    assert capped.return_hidden_states is False
    assert type(capped.hidden_states) is list
    assert capped.hidden_states == capped_chunks


# --- installer behavior -----------------------------------------------------


def test_missing_layout_fails_closed_when_collecting(
    monkeypatch, stub_sglang
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))

    with pytest.raises(RuntimeError, match="no known SGLang output processor"):
        sglang_patch.patch_sglang_hidden_states_tensor_output()
    assert not sglang_patch._SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED


def test_missing_layout_is_skipped_without_collection(
    monkeypatch, stub_sglang
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    monkeypatch.setenv(
        "VERL_SPECO_DRAFTER_CONFIG",
        json.dumps({"enable": True, "training": {}}),
    )

    sglang_patch.patch_sglang_hidden_states_tensor_output()
    assert not sglang_patch._SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED


def test_unmatched_new_layout_fails_closed(monkeypatch, tmp_path, stub_sglang) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    _install_source(
        monkeypatch,
        tmp_path,
        _NEW_MODULE,
        """
        class SchedulerBatchResultProcessor:
            def process_batch_result_prefill(self, batch, result):
                return None

            def process_batch_result_decode(self, batch, result):
                return None
        """,
    )

    with pytest.raises(RuntimeError, match="process_batch_result_decode"):
        sglang_patch.patch_sglang_hidden_states_tensor_output()


@pytest.mark.parametrize(
    ("config", "raises"),
    [
        (_DFLASH_COLLECT_CONFIG, True),
        ({**_DFLASH_COLLECT_CONFIG, "speculative_algorithm": "EAGLE3"}, False),
        ({**_DFLASH_COLLECT_CONFIG, "training": {}}, False),
    ],
)
def test_dflash_collection_fails_closed_without_spec_v1_or_v2_worker(
    monkeypatch, stub_sglang, config, raises
) -> None:
    # No spec-v1 DFlashVerifyInput.verify (v0.5.14 onwards) and no patchable
    # spec-v2 worker either.
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.dflash_worker_v2", None)
    dflash_info = types.ModuleType(_DFLASH_INFO_MODULE)
    dflash_info.DFlashVerifyInput = type("DFlashVerifyInput", (), {})
    monkeypatch.setitem(sys.modules, _DFLASH_INFO_MODULE, dflash_info)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(config))
    sglang_patch = _fresh_sglang_patch(monkeypatch)

    if raises:
        with pytest.raises(RuntimeError, match="DFlashVerifyInput.verify is missing and the spec-v2"):
            sglang_patch.patch_sglang_dflash_verify_hidden_states()
    else:
        sglang_patch.patch_sglang_dflash_verify_hidden_states()
    assert not sglang_patch._SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED


@pytest.mark.parametrize(
    ("positions", "rows", "reason"),
    [
        (None, 4, "hidden_positions_missing_or_mismatched"),
        ([0, 1, 2], 4, "hidden_positions_missing_or_mismatched"),
        ([0, 1, 3, 4], 4, "hidden_positions_not_contiguous"),
        ([5, 6, 7, 8], 4, None),
        ([7], 1, None),
    ],
)
def test_block_drafter_sample_requires_contiguous_positions(
    positions, rows, reason
) -> None:
    from verl_speco.integration.sglang_runtime import (
        _block_drafter_hidden_alignment_failure,
    )

    tensor = None if positions is None else torch.tensor(positions)
    assert _block_drafter_hidden_alignment_failure(tensor, rows) == reason


# --- DFlash spec-v2 worker (v0.5.14 onwards) --------------------------------

_DFLASH_V2_MODULE = "sglang.srt.speculative.dflash_worker_v2"
_DFLASH_V2_SOURCE = "dflash_worker_v2_v0_5_16.py.txt"
_DFLASH_V2_CLASS = "DFlashWorkerV2"


def _dflash_aux_req(request_id: int, prompt_len: int, cached: int) -> _Req:
    req = _Req(request_id, prompt_len, cached)
    req.sampling_params = types.SimpleNamespace(
        custom_params={"_verl_dflash_return_aux_hidden": True}
    )
    return req


def test_vendored_dflash_v2_source_is_verbatim() -> None:
    test_vendored_source_is_verbatim(_DFLASH_V2_SOURCE)


def test_dflash_v2_worker_drops_hidden_states_after_draft_kv_append() -> None:
    """Pin the upstream contract the spec-v2 patch relies on.

    ``forward_batch_generation`` hands the target hidden states to
    ``_append_target_hidden_to_draft_kv_by_loc(target_hidden=...)`` once for
    prefill and once for decode, then sets ``logits_output.hidden_states = None``.
    The decode result reports ``accept_lens=commit_lens`` and
    ``speculative_num_draft_tokens=int(self.block_size)``, which is the stride
    the batch result processor slices the restored tensor with.
    """
    source = _method_source(_DFLASH_V2_SOURCE, _DFLASH_V2_CLASS, "forward_batch_generation")
    tree = ast.parse(source)
    appends = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_append_target_hidden_to_draft_kv_by_loc"
    ]
    assert len(appends) == 2
    assert all(
        any(keyword.arg == "target_hidden" for keyword in call.keywords)
        for call in appends
    )
    assert source.count("logits_output.hidden_states = None") == 2
    assert "accept_lens=commit_lens" in source
    assert "speculative_num_draft_tokens=int(self.block_size)" in source

    _, body = _vendored(_DFLASH_V2_SOURCE)
    assert "def _append_target_hidden_to_draft_kv_by_loc(" in body
    assert body.count("._append_target_hidden_to_draft_kv_by_loc(") == 2


def _install_dflash_v2(monkeypatch, tmp_path):
    _, body = _vendored(_DFLASH_V2_SOURCE)
    source = _stub_sglang_imports(body)
    # DFlashWorkerV2 subclasses BaseSpecWorker; a stub instance cannot be a base.
    source = source.replace(
        "BaseSpecWorker = _SGLangStub(name='BaseSpecWorker')\n",
        "BaseSpecWorker = object\n",
    )
    assert "BaseSpecWorker = object\n" in source
    module = _load_module(
        monkeypatch, tmp_path, _DFLASH_V2_MODULE, source, "dflash_worker_v2_v0_5_16"
    )

    class _DraftInput:
        bonus_tokens = None
        reserved_seq_lens_cpu = None

    class _VerifyInput:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def prepare_for_verify(self, batch, target_worker):
            return MagicMock(), None

    module.DFlashDraftInputV2 = _DraftInput
    module.DFlashVerifyInput = _VerifyInput
    module.GenerationBatchResult = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.compute_position = lambda *args, **kwargs: (None, None)
    # The draft KV write is not under test; record what the worker passes.
    kv_writes = []
    monkeypatch.setattr(
        module.DFlashWorkerV2,
        "_append_target_hidden_to_draft_kv_by_loc",
        lambda self, **kwargs: kv_writes.append(kwargs),
    )
    for name in (
        "_validate_phase1_sampling_support",
        "_ensure_draft_block_buffers",
    ):
        monkeypatch.setattr(module.DFlashWorkerV2, name, lambda self, *a, **k: None)
    monkeypatch.setattr(
        module.DFlashWorkerV2,
        "_make_next_draft_input_prefill",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        module.DFlashWorkerV2,
        "_make_next_draft_input_decode",
        lambda self, **kwargs: None,
    )
    # CPU tensors cannot record a CUDA stream.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None, raising=False)
    return module, _DraftInput, kv_writes


class _Mode:
    def __init__(self, extend: bool) -> None:
        self.extend = extend

    def is_extend(self) -> bool:
        return self.extend

    def is_idle(self) -> bool:
        return False


def _dflash_v2_worker(module, *, bs: int, prefill_hidden=None, verify_hidden=None):
    worker = module.DFlashWorkerV2.__new__(module.DFlashWorkerV2)
    block = _STRIDE
    worker.device = "cpu"
    worker.block_size = block
    worker._mask_token_id = 0
    worker._use_triton_prepare_block = False
    worker._use_triton_accept_bonus = False
    worker._need_mamba_verify_commit = False
    worker.use_compact_draft_cache = False
    worker._draft_sampler = None
    worker._draft_block_spec_info = None
    worker._block_pos_offsets = torch.arange(block)
    worker._draft_block_ids_buf = torch.zeros((bs, block), dtype=torch.long)
    worker._draft_block_positions_buf = torch.zeros((bs, block), dtype=torch.long)
    worker._draft_block_tokens_buf = torch.zeros((bs, block), dtype=torch.long)
    worker._draft_verify_out_cache_loc_buf = torch.zeros((bs, block), dtype=torch.long)
    worker._draft_block_end_buf = torch.zeros(bs, dtype=torch.long)
    worker._draft_seq_lens_cpu_buf = torch.zeros(bs, dtype=torch.int32)
    worker._greedy_sample_from_vocab_parallel_head = lambda hidden_states, lm_head: (
        torch.zeros(hidden_states.shape[0], dtype=torch.long)
    )

    target_model = MagicMock()
    target_model.get_input_embeddings.return_value = lambda ids: torch.zeros(
        (*ids.shape, 2)
    )
    worker.target_worker = MagicMock()
    worker.target_worker.model_runner.model = target_model
    worker.model_runner = MagicMock()
    worker.draft_model_runner = MagicMock()
    worker.draft_model_runner.forward.return_value = types.SimpleNamespace(
        logits_output=types.SimpleNamespace(hidden_states=torch.zeros((bs * block, 2))),
        can_run_graph=False,
    )

    def target_forward(*args, **kwargs):
        if "capture_hidden_mode" in kwargs:
            return types.SimpleNamespace(
                logits_output=types.SimpleNamespace(
                    hidden_states=prefill_hidden, customized_info=None
                ),
                next_token_ids=torch.zeros(bs, dtype=torch.long),
            )
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                hidden_states=verify_hidden,
                next_token_logits=torch.zeros((bs * block, 8)),
                customized_info=None,
            ),
            can_run_cuda_graph=False,
        )

    worker.target_worker.forward_batch_generation.side_effect = target_forward
    module.assign_extend_cache_locs_func = lambda **kwargs: torch.zeros(
        bs * block, dtype=torch.long
    )
    return worker


def _dflash_v2_batch(reqs, draft_input_cls, *, extend: bool, prefix_lens=None):
    batch = _Batch(reqs, extend_lens=[req.extend_input_len for req in reqs])
    batch.is_spec_v2 = True
    batch.forward_mode = _Mode(extend)
    batch.is_extend_in_batch = extend
    batch.return_logprob = False
    batch.prefix_lens = [len(req.origin_input_ids) - req.extend_input_len for req in reqs]
    batch.out_cache_loc = torch.zeros(sum(batch.extend_lens), dtype=torch.long)
    batch.sampling_info = None
    batch.req_pool_indices = torch.arange(len(reqs))
    batch.seq_lens_cpu = None
    batch.seq_lens_sum = 0
    if extend:
        batch.seq_lens = torch.tensor([len(req.origin_input_ids) for req in reqs])
    else:
        batch.seq_lens = torch.tensor(prefix_lens)
        draft_input = draft_input_cls()
        draft_input.bonus_tokens = torch.zeros(len(reqs), dtype=torch.long)
        batch.spec_info = draft_input
    return batch


def test_dflash_v2_worker_returns_hidden_states_per_request(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    """Real v0.5.16 DFlash spec-v2 worker feeding the real v0.5.16 processor."""
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    sglang_patch = _import_sglang_patch(monkeypatch)
    processor_module = _install_vendored(
        monkeypatch, tmp_path, _NEW_MODULE, _V0_5_16_SOURCE
    )
    _install_vendored(monkeypatch, tmp_path, _STREAMER_MODULE, _STREAMER_SOURCE)
    worker_module, draft_input_cls, kv_writes = _install_dflash_v2(monkeypatch, tmp_path)
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    assert sglang_patch.patch_sglang_dflash_v2_worker_hidden_states()
    processor_cls = processor_module.SchedulerBatchResultProcessor

    reqs = [_dflash_aux_req(0, 6, 0), _dflash_aux_req(1, 9, 5), _dflash_aux_req(2, 3, 0)]
    prefill_rows = torch.tensor(
        [
            _ref_row(req.request_id, pos)
            for req in reqs
            for pos in range(
                len(req.origin_input_ids) - req.extend_input_len,
                len(req.origin_input_ids),
            )
        ]
    )
    worker = _dflash_v2_worker(worker_module, bs=len(reqs), prefill_hidden=prefill_rows)
    batch = _dflash_v2_batch(reqs, draft_input_cls, extend=True)
    result = worker.forward_batch_generation(batch)
    # The worker wrote the same tensor into the draft KV cache.
    assert kv_writes[-1]["target_hidden"] is prefill_rows
    assert torch.equal(result.logits_output.hidden_states, prefill_rows)
    result.extend_input_len_per_req = None
    result.extend_logprob_start_len_per_req = None
    result.copy_done = None
    result.routed_experts_output = None
    result.indexer_topk_output = None
    result.can_run_cuda_graph = False
    processor_cls.process_batch_result_prefill(_processor_self(), batch, result)

    next_positions = {req.request_id: len(req.origin_input_ids) for req in reqs}
    for commit_lens in ([3, 1, 4], [1, 4, 2], [2, 2, 1]):
        # [bs * block_size, D]: request i owns rows i * block_size onward, at
        # positions seq_len .. seq_len + block_size - 1. Rows past the commit
        # length are rejected drafts.
        rows = [_FILLER_ROW] * (_STRIDE * len(reqs))
        for index, (req, commit) in enumerate(zip(reqs, commit_lens, strict=True)):
            for offset in range(commit):
                rows[index * _STRIDE + offset] = _ref_row(
                    req.request_id, next_positions[req.request_id] + offset
                )
        verify_hidden = torch.tensor(rows)
        worker = _dflash_v2_worker(
            worker_module, bs=len(reqs), verify_hidden=verify_hidden
        )
        accept = torch.tensor([commit - 1 for commit in commit_lens])
        worker_module.compute_dflash_correct_drafts_and_bonus = (
            lambda candidates, target_predict, accept=accept: (
                accept,
                torch.zeros(len(accept), dtype=torch.long),
            )
        )
        batch = _dflash_v2_batch(
            reqs,
            draft_input_cls,
            extend=False,
            prefix_lens=[next_positions[req.request_id] for req in reqs],
        )
        result = worker.forward_batch_generation(batch)
        assert kv_writes[-1]["commit_lens"].tolist() == commit_lens
        assert torch.equal(result.logits_output.hidden_states, verify_hidden)
        assert result.speculative_num_draft_tokens == _STRIDE
        for req, commit in zip(reqs, commit_lens, strict=True):
            next_positions[req.request_id] += commit
        processor_cls.process_batch_result_decode(
            _processor_self([[0] * commit for commit in commit_lens]),
            batch,
            _Result(
                result.logits_output,
                accept_lens=result.accept_lens.tolist(),
                speculative_num_draft_tokens=result.speculative_num_draft_tokens,
            ),
        )

    for req in reqs:
        cached = len(req.origin_input_ids) - req.extend_input_len
        expected_positions = list(range(cached, next_positions[req.request_id]))
        expected_rows = torch.tensor(
            [_ref_row(req.request_id, pos) for pos in expected_positions]
        )
        rows, positions = _collected(req)
        assert positions.tolist() == expected_positions, req.rid
        assert torch.equal(rows, expected_rows), req.rid


def test_dflash_v2_worker_keeps_upstream_behaviour_without_request(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    sglang_patch = _import_sglang_patch(monkeypatch)
    worker_module, draft_input_cls, kv_writes = _install_dflash_v2(monkeypatch, tmp_path)
    assert sglang_patch.patch_sglang_dflash_v2_worker_hidden_states()
    # Patching twice keeps a single wrapper.
    forward = worker_module.DFlashWorkerV2.forward_batch_generation
    assert sglang_patch.patch_sglang_dflash_v2_worker_hidden_states()
    assert worker_module.DFlashWorkerV2.forward_batch_generation is forward

    reqs = [_Req(0, 4, 0), _dflash_aux_req(1, 3, 0)]
    reqs[1].return_hidden_states = False  # asked for aux, but not for hidden states
    prefill_rows = torch.ones((7, _HIDDEN_DIM))
    worker = _dflash_v2_worker(worker_module, bs=2, prefill_hidden=prefill_rows)
    result = worker.forward_batch_generation(
        _dflash_v2_batch(reqs, draft_input_cls, extend=True)
    )
    assert kv_writes[-1]["target_hidden"] is prefill_rows
    assert result.logits_output.hidden_states is None
    assert getattr(worker, "_verl_dflash_v2_target_hidden") is None


def test_dflash_verify_patch_uses_v2_worker_without_spec_v1_verify(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    dflash_info = types.ModuleType(_DFLASH_INFO_MODULE)
    dflash_info.DFlashVerifyInput = type("DFlashVerifyInput", (), {})
    monkeypatch.setitem(sys.modules, _DFLASH_INFO_MODULE, dflash_info)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    sglang_patch = _fresh_sglang_patch(monkeypatch)
    worker_module, _, _ = _install_dflash_v2(monkeypatch, tmp_path)

    sglang_patch.patch_sglang_dflash_verify_hidden_states()

    assert sglang_patch._SGLANG_DFLASH_VERIFY_HIDDEN_STATES_PATCHED
    assert worker_module.DFlashWorkerV2.forward_batch_generation._verl_patched_dflash_v2_hidden_states
