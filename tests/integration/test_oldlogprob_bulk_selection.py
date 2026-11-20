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
"""Check exact selection order, SP ownership and absence of CUDA scalar reads."""

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.integration import oldlogprob_runtime as runtime


@pytest.fixture
def selection_env(monkeypatch):
    names = (
        "verl",
        "verl.utils",
        "verl.utils.tensordict_utils",
        "verl.utils.dataset",
        "verl.utils.dataset.dataset_utils",
        "verl.utils.ulysses",
    )
    for name in names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    tu = sys.modules["verl.utils.tensordict_utils"]
    monkeypatch.setattr(
        tu,
        "get_non_tensor_data",
        lambda data, key, default=None: data.get(key, default),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["verl.utils.dataset.dataset_utils"],
        "DatasetPadMode",
        SimpleNamespace(NO_PADDING="no_padding"),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["verl.utils.ulysses"],
        "get_ulysses_sequence_parallel_group",
        lambda: "sp",
        raising=False,
    )
    monkeypatch.setattr(runtime, "_BATCH_POSTPROCESS_PATCHED", True)
    monkeypatch.setattr(runtime, "_POSTPROCESS_PATCHED", True)


class CudaControl:
    def __init__(self, value, copies):
        self.value, self.copies = value, copies
        self.device = SimpleNamespace(type="cuda")
        self.shape = value.shape

    def bool(self):
        return CudaControl(self.value.bool(), self.copies)

    def long(self):
        return CudaControl(self.value.long(), self.copies)

    def detach(self):
        return self

    def to(self, *, device, dtype=None):
        assert device == "cpu"
        self.copies.append(self.value.numel())
        return self.value.to(dtype=dtype) if dtype is not None else self.value

    def __getitem__(self, index):
        raise AssertionError("CUDA control values must be read in bulk")


def data(empty=False):
    offsets = torch.tensor([0, 4, 4, 9, 12, 14])
    positions = torch.tensor(
        [
            [3, 0, 3, -1, 4, 1],
            [0, -1, 1, 2, 3, 0],
            [4, 1, 0, 8, -1, 3],
            [2, 1, 0, 3, 1, 0],
            [1, 0, 1, -1, 2, 0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    collect = (
        torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool)
        if not empty
        else torch.zeros(5, dtype=torch.bool)
    )
    return offsets, positions, mask, collect


def reference(offsets, positions, mask, collect, size, rank):
    start, end = (
        (0, 14) if size == 1 else (rank * (16 // size), (rank + 1) * (16 // size))
    )
    flat = offsets[:-1, None] + positions
    valid = (
        mask
        & collect[:, None]
        & (positions >= 0)
        & (positions < (offsets[1:] - offsets[:-1])[:, None])
        & (flat >= start)
        & (flat < end)
    )
    batch, row = valid.nonzero(as_tuple=True)
    owners = {}
    for b, r in zip(batch.tolist(), row.tolist(), strict=True):
        owners.setdefault(b, []).append(r)
    return {
        "selected_batch_indices": (collect & mask.any(dim=1))
        .nonzero()
        .reshape(-1)
        .tolist(),
        "local_positions": (flat[batch, row] - start).tolist(),
        "local_batch_indices": batch.tolist(),
        "local_row_indices": row.tolist(),
        "local_3d_batch_indices": batch.tolist(),
        "local_3d_seq_positions": positions[batch, row].tolist(),
        "selected_owner_rows": owners,
    }


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("object_ref", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("bulk", [False, True])
def test_bulk_selection_matches_vectorized_reference_without_scalar_reads(
    selection_env, monkeypatch, size, object_ref, empty, bulk
):
    offsets, positions, mask, collect = data(empty)
    copies = []
    wrap = (lambda value: CudaControl(value, copies)) if bulk else (lambda value: value)
    batch = {
        "input_ids": SimpleNamespace(
            offsets=lambda: offsets, values=lambda: torch.zeros(14)
        ),
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: wrap(collect),
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: wrap(positions),
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: wrap(mask),
        runtime.OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY: object_ref,
    }
    for rank in range(size):
        copies.clear()
        monkeypatch.setattr(
            torch.distributed, "get_rank", lambda group=None, rank=rank: rank
        )
        context = runtime._build_selection_context(
            SimpleNamespace(
                use_ulysses_sp=size > 1, ulysses_sequence_parallel_size=size
            ),
            {"pad_size": 2},
            batch,
        )
        for name, expected in reference(
            offsets, positions, mask, collect, size, rank
        ).items():
            assert context[name] == expected, name
        assert len(copies) == (
            (1 if empty else 3) if bulk else 0
        )  # Offsets are already on the CPU.
        assert context["timing_us"]["bulk_host_contexts"] == int(bulk)
        assert context["sparse_sp_merge"] == (object_ref and size > 1)
        assert context["output_batch_size"] == 5
        hidden = torch.arange(16 * 3).reshape(16, 3)
        start, end = (
            (0, 14) if size == 1 else (rank * (16 // size), (rank + 1) * (16 // size))
        )
        actual = runtime._select_rows_from_local_hidden(context, hidden[start:end])
        expected_rows = hidden[
            torch.tensor(context["local_positions"], dtype=torch.long) + start
        ]
        if context["sparse_sp_merge"]:
            torch.testing.assert_close(actual["rows"], expected_rows, rtol=0, atol=0)
        else:
            for index, (b, r) in enumerate(
                zip(
                    context["local_batch_indices"],
                    context["local_row_indices"],
                    strict=True,
                )
            ):
                torch.testing.assert_close(
                    actual[b, r], expected_rows[index], rtol=0, atol=0
                )


@pytest.mark.parametrize("width", [2, 5, 9])
def test_timing_decoder_accepts_legacy_and_extended_widths(width):
    values = torch.tensor(
        [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 1000, 4][
            :width
        ],
        dtype=torch.float32,
    )
    result = runtime.decode_oldlogprob_timing(values)
    assert result["select"] == 1 and result["sp_merge"] == 2
    assert ("ray_put" in result) == (width >= 5)
    assert ("context_build" in result) == (width == 9)
    if width == 9:
        assert result["context_build"] == 6 and result["control_copy"] == 7
        assert result["control_bytes"] == 1000 and result["bulk_host_contexts"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_controls_match_cpu_reference(selection_env, monkeypatch):
    offsets, positions, mask, collect = data()
    batch = {
        "input_ids": SimpleNamespace(
            offsets=lambda: offsets, values=lambda: torch.zeros(14, device="cuda")
        ),
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: collect.cuda(),
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: positions.cuda(),
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: mask.cuda(),
    }
    original_item = torch.Tensor.item

    def forbid_cuda_scalar(tensor, *args):
        if tensor.device.type == "cuda":
            raise AssertionError("per-value CUDA read")
        return original_item(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", forbid_cuda_scalar)
    context = runtime._build_selection_context(
        SimpleNamespace(use_ulysses_sp=False), {}, batch
    )
    for name, expected in reference(offsets, positions, mask, collect, 1, 0).items():
        assert context[name] == expected
    wire = runtime._timing_tensor_from_context(context, torch.device("cuda"))
    assert runtime.decode_oldlogprob_timing(wire.sum(dim=0))["bulk_host_contexts"] == 1


@pytest.mark.parametrize("field", ["collect", "positions", "mask", "offsets"])
def test_mixed_control_devices_do_not_leave_scalar_cuda_reads(selection_env, field):
    offsets, positions, mask, collect = data()
    copies = []
    values = {
        "collect": collect,
        "positions": positions,
        "mask": mask,
        "offsets": offsets,
    }
    values[field] = CudaControl(values[field], copies)
    batch = {
        "input_ids": SimpleNamespace(
            offsets=lambda: values["offsets"], values=lambda: torch.zeros(14)
        ),
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: values["collect"],
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: values["positions"],
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: values["mask"],
    }
    context = runtime._build_selection_context(
        SimpleNamespace(use_ulysses_sp=False), {}, batch
    )
    for name, expected in reference(offsets, positions, mask, collect, 1, 0).items():
        assert context[name] == expected
    assert len(copies) == 1


def test_cuda_packed_inputs_with_cpu_controls_also_avoid_scalar_loops(
    selection_env, monkeypatch
):
    offsets, positions, mask, collect = data()
    batch = {
        "input_ids": SimpleNamespace(
            offsets=lambda: offsets,
            values=lambda: SimpleNamespace(
                device=SimpleNamespace(type="cuda"), numel=lambda: 14
            ),
        ),
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: collect,
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: positions,
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: mask,
    }
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", lambda *args: pytest.fail("scalar loop"))
        actual = runtime._build_selection_context(
            SimpleNamespace(use_ulysses_sp=False), {}, batch
        )
    for name, expected in reference(offsets, positions, mask, collect, 1, 0).items():
        assert actual[name] == expected
    assert actual["timing_us"]["bulk_host_contexts"] == 1


@pytest.mark.parametrize("cuda_capture", [False, True])
def test_timing_extension_keeps_legacy_prefix_and_non_cuda_shape(cuda_capture):
    context = {
        "batch_size": 2,
        "cuda_capture": cuda_capture,
        "timing_us": {
            "select": 1,
            "sp_merge": 2,
            "concat": 3,
            "context_build": 6,
            "control_copy": 7,
            "control_bytes": 1000,
            "bulk_host_contexts": 1,
        },
    }
    timing = runtime._timing_tensor_from_context(context, "cpu")
    assert timing.shape == (2, 9 if cuda_capture else 5)
    torch.testing.assert_close(
        timing[0, :5],
        torch.tensor([1, 2, 3, 0, 0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert timing[1].count_nonzero() == 0
    if cuda_capture:
        assert (
            runtime.decode_oldlogprob_timing(timing.sum(dim=0))["control_bytes"] == 1000
        )
