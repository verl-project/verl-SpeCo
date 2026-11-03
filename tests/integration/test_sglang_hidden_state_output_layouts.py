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
"""Per-request hidden-state output patch across SGLang output-processor layouts.

SGLang 0.5.16 moved the prefill/decode result processing from
``scheduler_output_processor_mixin.SchedulerOutputProcessorMixin`` into
``scheduler_components.batch_result_processor.SchedulerBatchResultProcessor``.
If the patch silently misses the new class, stock SGLang appends
``hidden_states[i]`` per request on decode, which for a DFlash verify output
packed as ``[sum(commit_lens), D]`` is a single row of the wrong request, and
the prefill slice ignores prefix-cache hits.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import textwrap
import types

import pytest

# sglang_patch imports torch at module level; the CPU CI image has none.
torch = pytest.importorskip("torch")

_NEW_MODULE = "sglang.srt.managers.scheduler_components.batch_result_processor"
_OLD_MODULE = "sglang.srt.managers.scheduler_output_processor_mixin"

_STUBBED = (
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.utils",
    "sglang.srt.utils.patch_torch",
    "sglang.srt.managers",
    "sglang.srt.managers.scheduler_components",
)

# Mirrors the hidden-state sites of SGLang 0.5.16
# scheduler_components/batch_result_processor.py.
_NEW_LAYOUT_SOURCE = '''
class SchedulerBatchResultProcessor:
    def process_batch_result_prefill(self, batch, result):
        logits_output, extend_input_len_per_req = (
            result.logits_output,
            result.extend_input_len_per_req,
        )
        hidden_state_offset = 0
        for i, req in enumerate(batch.reqs):
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
            .cpu()
            .clone()
            .tolist()
        )
        return hidden_state_offset

    def process_batch_result_decode(self, batch, result):
        logits_output = result.logits_output
        is_spec_v1 = not batch.is_spec_v2
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
'''

# Mirrors the hidden-state sites of the older SchedulerOutputProcessorMixin.
_OLD_LAYOUT_SOURCE = '''
class SchedulerOutputProcessorMixin:
    def process_batch_result_prefill(self, batch, result):
        logits_output = result.logits_output
        extend_input_len_per_req = result.extend_input_len_per_req
        hidden_state_offset = 0
        for i, req in enumerate(batch.reqs):
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
        for i, req in enumerate(batch.reqs):
            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

    def stream_output_generation(self, reqs):
        output_hidden_states = None
        rids = []
        for req in reqs:
            rids.append(req.rid)
            if req.return_hidden_states:
                if output_hidden_states is None:
                    output_hidden_states = []
                output_hidden_states.append(req.hidden_states)
        return output_hidden_states
'''

_DFLASH_COLLECT_CONFIG = {
    "enable": True,
    "training": {"collect_hidden_states_from_sgl": True},
}


def _load_source_module(tmp_path, module_name, source):
    path = tmp_path / (module_name.rpartition(".")[2] + ".py")
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    monkeypatch.delenv("VERL_SPECO_DRAFTER_CONFIG", raising=False)
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    return stubs


def _import_sglang_patch(monkeypatch):
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )
    module = importlib.import_module("verl_speco.integration.sglang_patch")
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


def _install_layout(monkeypatch, tmp_path, module_name, source):
    module = _load_source_module(tmp_path, module_name, source)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _req(rid, prompt_len):
    return types.SimpleNamespace(
        rid=rid,
        return_hidden_states=True,
        hidden_states=[],
        origin_input_ids=list(range(prompt_len)),
    )


def _rows(req):
    """Concatenate collected rows and their positions for one request."""
    chunks, positions = [], []
    for item in req.hidden_states:
        assert isinstance(item, dict), item
        chunks.append(item["hidden_states"])
        positions.append(item["positions"])
    return torch.cat(chunks, dim=0), torch.cat(positions, dim=0)


def _hidden(values):
    # Row r holds value r in every column, so a row identifies its source.
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)


def test_new_layout_is_patched(monkeypatch, tmp_path, stub_sglang) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_layout(monkeypatch, tmp_path, _NEW_MODULE, _NEW_LAYOUT_SOURCE)

    sglang_patch.patch_sglang_hidden_states_tensor_output()

    cls = module.SchedulerBatchResultProcessor
    for name in ("process_batch_result_prefill", "process_batch_result_decode"):
        assert getattr(cls, name)._verl_patched_hidden_states_tensor_output
    assert sglang_patch._SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED


@pytest.mark.parametrize("is_spec_v2", [False, True])
def test_new_layout_splits_dflash_verify_rows_per_request(
    monkeypatch, tmp_path, stub_sglang, is_spec_v2
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_layout(monkeypatch, tmp_path, _NEW_MODULE, _NEW_LAYOUT_SOURCE)
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    processor = module.SchedulerBatchResultProcessor()

    req_a, req_b = _req("a", 3), _req("b", 2)
    batch = types.SimpleNamespace(reqs=[req_a, req_b], is_spec_v2=is_spec_v2)
    processor.process_batch_result_prefill(
        batch,
        types.SimpleNamespace(
            logits_output=types.SimpleNamespace(hidden_states=_hidden(range(5))),
            extend_input_len_per_req=[3, 2],
        ),
    )

    # DFlash verify packs the committed rows of all requests: req a commits 3
    # rows (2 correct drafts + bonus), req b commits 1 row.
    commit_lens = [3, 1]
    decode_output = types.SimpleNamespace(
        hidden_states=_hidden([100, 101, 102, 200]),
        _verl_dflash_aux_hidden_states=True,
    )
    if is_spec_v2:
        decode_result = types.SimpleNamespace(
            logits_output=decode_output, accept_lens=commit_lens
        )
    else:
        decode_result = types.SimpleNamespace(
            logits_output=decode_output,
            num_correct_drafts_per_req_cpu=[n - 1 for n in commit_lens],
        )
    processor.process_batch_result_decode(batch, decode_result)

    rows_a, positions_a = _rows(req_a)
    rows_b, positions_b = _rows(req_b)
    assert rows_a[:, 0].tolist() == [0, 1, 2, 100, 101, 102]
    assert positions_a.tolist() == [0, 1, 2, 3, 4, 5]
    assert rows_b[:, 0].tolist() == [3, 4, 200]
    assert positions_b.tolist() == [0, 1, 2]


def test_new_layout_prefill_respects_prefix_cache_hit(
    monkeypatch, tmp_path, stub_sglang
) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_layout(monkeypatch, tmp_path, _NEW_MODULE, _NEW_LAYOUT_SOURCE)
    sglang_patch.patch_sglang_hidden_states_tensor_output()
    processor = module.SchedulerBatchResultProcessor()

    # req a hits 6 of its 10 prompt tokens in the prefix cache, so only 4 rows
    # are computed; req b has no cache hit.
    req_a, req_b = _req("a", 10), _req("b", 2)
    batch = types.SimpleNamespace(reqs=[req_a, req_b], is_spec_v2=False)
    processor.process_batch_result_prefill(
        batch,
        types.SimpleNamespace(
            logits_output=types.SimpleNamespace(hidden_states=_hidden(range(6))),
            extend_input_len_per_req=[4, 2],
        ),
    )

    rows_a, positions_a = _rows(req_a)
    rows_b, positions_b = _rows(req_b)
    assert rows_a[:, 0].tolist() == [0, 1, 2, 3]
    assert positions_a.tolist() == [6, 7, 8, 9]
    assert rows_b[:, 0].tolist() == [4, 5]
    assert positions_b.tolist() == [0, 1]


def test_old_layout_is_still_patched(monkeypatch, tmp_path, stub_sglang) -> None:
    sglang_patch = _import_sglang_patch(monkeypatch)
    module = _install_layout(monkeypatch, tmp_path, _OLD_MODULE, _OLD_LAYOUT_SOURCE)

    sglang_patch.patch_sglang_hidden_states_tensor_output()

    cls = module.SchedulerOutputProcessorMixin
    for name in (
        "process_batch_result_prefill",
        "process_batch_result_decode",
        "stream_output_generation",
    ):
        assert getattr(cls, name)._verl_patched_hidden_states_tensor_output

    req = _req("a", 4)
    batch = types.SimpleNamespace(reqs=[req])
    cls().process_batch_result_prefill(
        batch,
        types.SimpleNamespace(
            logits_output=types.SimpleNamespace(hidden_states=_hidden(range(2))),
            extend_input_len_per_req=[2],
        ),
    )
    rows, positions = _rows(req)
    assert rows[:, 0].tolist() == [0, 1]
    assert positions.tolist() == [2, 3]


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
    _install_layout(
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
