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

* ``scheduler_output_processor_mixin_v0_5_12``: the mixin layout, up to v0.5.12.
* ``batch_result_processor_0801cc05``: sgl-project/sglang main at 0801cc05, the
  build in the ROCm image (it reports 0.5.16). Decode appends
  ``hidden_states[i]`` per request, while DFlash verify packs the committed rows
  of all requests as ``[sum(commit_lens), D]``.
* ``batch_result_processor_v0_5_14``: the v0.5.14 tag. Speculative decode
  extends ``hidden_states[i * stride : i * stride + accept_len]``;
  non-speculative decode still appends ``hidden_states[i]``.
* ``batch_result_processor_v0_5_16``: the v0.5.16 tag. All decode uses the
  strided extend.

Both batch-result processors slice ``len(req.origin_input_ids)`` prefill rows per
request, which ignores prefix-cache hits.

The per-request tests run the real, patched ``process_batch_result_prefill`` and
``process_batch_result_decode`` on hidden states whose every row encodes its
(request, position), and compare each request's collected rows and positions to
the exact reference.
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

_SOURCES_DIR = Path(__file__).parent / "sglang_sources"
_HEADER_END = "# --- end of verl-SpeCo header ---\n"
_IMAGE_SOURCE = "batch_result_processor_0801cc05.py.txt"
_V0_5_14_SOURCE = "batch_result_processor_v0_5_14.py.txt"
_V0_5_16_SOURCE = "batch_result_processor_v0_5_16.py.txt"
_MIXIN_SOURCE = "scheduler_output_processor_mixin_v0_5_12.py.txt"
_ALL_SOURCES = (_IMAGE_SOURCE, _V0_5_14_SOURCE, _V0_5_16_SOURCE, _MIXIN_SOURCE)

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
        (
            _MIXIN_SOURCE,
            "SchedulerOutputProcessorMixin",
            "process_batch_result_prefill",
            {"_append_sglang_prefill_hidden_states(": 1},
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
        (
            _MIXIN_SOURCE,
            _OLD_MODULE,
            "SchedulerOutputProcessorMixin",
            (
                "process_batch_result_prefill",
                "process_batch_result_decode",
                "stream_output_generation",
            ),
        ),
    ],
)
def test_installer_patches_real_layout(
    monkeypatch, tmp_path, stub_sglang, file_name, module_name, class_name, methods
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(_DFLASH_COLLECT_CONFIG))
    module = _install_vendored(monkeypatch, tmp_path, module_name, file_name)

    sglang_patch.patch_sglang_hidden_states_tensor_output()

    cls = getattr(module, class_name)
    for name in methods:
        assert getattr(cls, name)._verl_patched_hidden_states_tensor_output
    assert sglang_patch._SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED


# --- deterministic per-request alignment ------------------------------------


def _ref_row(request_id: int, position: int) -> list[float]:
    # Every row encodes the (request, position) it belongs to.
    return [float(request_id * 1000 + position)] * _HIDDEN_DIM


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

    def finished(self) -> bool:
        return False

    def update_finish_state(self, *args) -> None:
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
    rows, positions = [], []
    for item in req.hidden_states:
        assert isinstance(item, dict), type(item)
        rows.append(item["hidden_states"])
        positions.append(item["positions"])
    return torch.cat(rows, dim=0), torch.cat(positions, dim=0)


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
        # SPECO's DFlash verify patch packs only committed rows:
        # [sum(commit_lens), D].
        rows = [
            _ref_row(req.request_id, next_positions[req.request_id] + offset)
            for req, commit in zip(reqs, commit_lens, strict=True)
            for offset in range(commit)
        ]
        logits_output = types.SimpleNamespace(
            hidden_states=torch.tensor(rows),
            customized_info=None,
            _verl_dflash_aux_hidden_states=True,
        )
        extra = {"num_correct_drafts_per_req_cpu": [n - 1 for n in commit_lens]}
    for req, commit in zip(reqs, commit_lens, strict=True):
        next_positions[req.request_id] += commit
    next_token_ids = [[0] * commit for commit in commit_lens]
    cls.process_batch_result_decode(
        _processor_self(next_token_ids),
        _Batch(reqs, spec=spec),
        _Result(logits_output, **extra),
    )


@pytest.mark.parametrize(
    ("file_name", "mode"),
    [
        # The image build; DFlash runs spec-v1 with SPECO's verify patch.
        (_IMAGE_SOURCE, "packed"),
        (_V0_5_14_SOURCE, "strided"),
        (_V0_5_14_SOURCE, "non_spec"),
        (_V0_5_16_SOURCE, "strided"),
        (_V0_5_16_SOURCE, "non_spec"),
    ],
)
@pytest.mark.parametrize("extend_lens_on_batch", [True, False])
def test_real_processor_collects_exact_rows_per_request(
    monkeypatch, tmp_path, stub_sglang, file_name, mode, extend_lens_on_batch
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_vendored(monkeypatch, tmp_path, _NEW_MODULE, file_name)
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    cls = module.SchedulerBatchResultProcessor

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
def test_dflash_collection_fails_closed_without_spec_v1_verify(
    monkeypatch, stub_sglang, config, raises
) -> None:
    # SGLang v0.5.16 only has the spec-v2 DFlash worker: DFlashVerifyInput has
    # no verify(), and the worker sets logits_output.hidden_states = None.
    dflash_info = types.ModuleType(_DFLASH_INFO_MODULE)
    dflash_info.DFlashVerifyInput = type("DFlashVerifyInput", (), {})
    monkeypatch.setitem(sys.modules, _DFLASH_INFO_MODULE, dflash_info)
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(config))
    sglang_patch = _fresh_sglang_patch(monkeypatch)

    if raises:
        with pytest.raises(RuntimeError, match="DFlashVerifyInput.verify is missing"):
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
