# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Compare exported rows and ownership against source data, including duplicates."""

import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from verl_speco.integration import oldlogprob_runtime as runtime


@pytest.fixture
def store(monkeypatch):
    stored = {}
    monkeypatch.setattr(runtime, "_BATCH_POSTPROCESS_PATCHED", True)
    monkeypatch.setattr(runtime, "_POSTPROCESS_PATCHED", True)

    def put(value):
        key = str(len(stored))
        stored[key] = value
        return key

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(put=put))
    return stored


def batch():
    return {
        runtime.OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY: True,
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: torch.tensor([True, False, True]),
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: torch.ones(
            3, 3, dtype=torch.bool
        ),
        runtime.OLD_LOGPROB_OWNER_RANK_KEY: torch.tensor([1, -1, 0]),
        runtime.OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY: torch.tensor([205, 11, 93]),
    }


def materialize(output, stored, index):
    meta = output[runtime.OLD_LOGPROB_HIDDEN_REF_META_KEY][index]
    ref = output[runtime.OLD_LOGPROB_HIDDEN_REFS_KEY][index]
    from verl_speco.integration.tensor_wire import decode_cpu_tensors

    payload = decode_cpu_tensors(stored[ref])
    tensor = (
        payload.get("hidden", payload.get("hidden_states"))
        if isinstance(payload, dict)
        else payload
    )
    return tensor[meta["chunk_start"] : meta["chunk_start"] + meta["chunk_length"]]


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0, 2], [], [4]])
def test_host_row_selection_preserves_order_and_duplicates(indices):
    source = torch.arange(30).reshape(5, 6)
    result = runtime._select_host_rows(source, indices)
    assert torch.equal(result, source[indices])
    if indices and indices == list(range(indices[0], indices[0] + len(indices))):
        assert (
            result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("with_logz", [False, True])
@pytest.mark.parametrize("gpu", [False, True])
def test_sparse_export_preserves_sorted_rows_owners_and_sources(
    store, monkeypatch, with_logz, gpu
):
    source = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    indices = torch.tensor([2, 0, 2, 0, 2, 0, 0])
    positions = torch.tensor([2, 1, 0, 0, 1, 1, -1])
    monkeypatch.setenv(
        "VERL_SPECO_DRAFTER_CONFIG",
        json.dumps({"training": {"gpu_transfer_optimizations": True}}),
    )
    hidden = {
        "rows": DeviceTensor(source) if gpu else source,
        "batch_indices": indices,
        "row_indices": positions,
    }
    target_logz = torch.arange(7, dtype=torch.float32).reshape(7, 1) + 0.125
    output = {runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: hidden}
    if with_logz:
        output[runtime.OLD_LOGPROB_TARGET_LOGZ_KEY] = {
            "rows": target_logz,
            "batch_indices": indices,
            "row_indices": positions,
        }
    # No scalar extraction is required from the control tensors.
    monkeypatch.setattr(torch.Tensor, "item", lambda *_a: pytest.fail("scalar read"))
    result = runtime._put_oldlogprob_hidden_refs(output, batch())
    assert torch.equal(materialize(result, store, 0), source[[3, 1, 5]])
    assert torch.equal(materialize(result, store, 2), source[[2, 4, 0]])
    metas = result[runtime.OLD_LOGPROB_HIDDEN_REF_META_KEY]
    assert metas[0]["owner"] == 1 and metas[0]["source_batch_index"] == 205
    assert metas[2]["owner"] == 0 and metas[2]["source_batch_index"] == 93
    assert metas[1] is None
    from verl_speco.integration.tensor_wire import decode_cpu_tensors

    for value in store.values():
        decoded = decode_cpu_tensors(value)
        payload = (
            decoded.get("hidden", decoded.get("hidden_states"))
            if isinstance(decoded, dict)
            else decoded
        )
        assert payload.storage_offset() == 0
        assert (
            payload.untyped_storage().nbytes()
            == payload.numel() * payload.element_size()
        )
    snapshot = materialize(result, store, 0).clone()
    source.fill_(99)
    assert torch.equal(materialize(result, store, 0), snapshot)

    if with_logz:
        for index, expected_sources in [(0, [3, 1, 5]), (2, [2, 4, 0])]:
            ref = result[runtime.OLD_LOGPROB_HIDDEN_REFS_KEY][index]
            decoded = decode_cpu_tensors(store[ref])
            meta = metas[index]
            start, length = meta["chunk_start"], meta["chunk_length"]
            actual = decoded["target_logz"][start : start + length]
            assert actual.dtype == torch.float32
            assert torch.equal(actual, target_logz[expected_sources])


class DeviceTensor(torch.Tensor):
    copies = 0
    device_kind = "cuda"

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, False)

    @property
    def device(self):
        return torch.device(self.device_kind)

    def to(self, *args, **kwargs):
        type(self).copies += 1
        return self.as_subclass(torch.Tensor).to(*args, **kwargs)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_compact_cuda_export_batches_only_full_windows(store, partial, dtype):
    controls = batch()
    if partial:
        controls[runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY][2, 2] = False
    source = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    hidden = DeviceTensor(source)
    DeviceTensor.copies = 0
    result = runtime._put_oldlogprob_hidden_refs(
        {
            runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: hidden,
            runtime.OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY: [0, 2],
        },
        controls,
    )
    assert DeviceTensor.copies == (2 if partial else 1)
    assert torch.equal(materialize(result, store, 0), source[0])
    assert torch.equal(materialize(result, store, 2), source[1, : 2 if partial else 3])
    before = materialize(result, store, 0).clone()
    source.fill_(0)
    assert torch.equal(materialize(result, store, 0), before)


def test_npu_keeps_selective_copy_path(store):
    class NpuTensor(DeviceTensor):
        copies = 0

        @property
        def device(self):
            return SimpleNamespace(type="npu")

    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = runtime._put_oldlogprob_hidden_refs(
        {
            runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: NpuTensor(source),
            runtime.OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY: [0, 2],
        },
        batch(),
    )
    assert NpuTensor.copies == 2
    assert torch.equal(materialize(output, store, 2), source[1])


@pytest.mark.parametrize("cuda", [False, True])
def test_export_diagnostics_are_once_per_mode_not_per_token(
    store, monkeypatch, caplog, cuda
):
    import logging

    from verl_speco.integration import phase_timing

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setattr(phase_timing, "configure_phase_logger", lambda: None)
    monkeypatch.setattr(phase_timing, "logger", logging.getLogger("packing_test"))
    monkeypatch.setattr(
        runtime._put_oldlogprob_hidden_refs, "_logged_pack_modes", set(), raising=False
    )
    caplog.set_level(logging.WARNING, logger="packing_test")
    for _ in range(3):
        runtime._put_oldlogprob_hidden_refs(
            {
                runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: (
                    DeviceTensor(torch.zeros(3, 3, 4)) if cuda else torch.zeros(3, 3, 4)
                ),
            },
            batch(),
        )
    records = [
        record for record in caplog.records if "[SpecoHostPack]" in record.message
    ]
    assert len(records) == 1
    assert "snapshot=independent" in records[0].message


@pytest.mark.parametrize("with_logz", [False, True])
@pytest.mark.parametrize("enabled", [False, True, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("legacy_env", [False, True])
def test_cuda_collection_respects_master_switch_and_preserves_values(
    store,
    monkeypatch,
    enabled,
    dtype,
    legacy_env,
    with_logz,
):
    canonical = "VERL_SPECO_DRAFTER_CONFIG"
    legacy = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"
    monkeypatch.delenv(canonical, raising=False)
    monkeypatch.delenv(legacy, raising=False)
    training = {} if enabled is None else {"gpu_transfer_optimizations": enabled}
    monkeypatch.setenv(
        legacy if legacy_env else canonical, json.dumps({"training": training})
    )
    source = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    target_logz = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1) + 0.125
    output = {
        runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: DeviceTensor(source),
        runtime.OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY: [0, 2],
    }
    if with_logz:
        output[runtime.OLD_LOGPROB_TARGET_LOGZ_KEY] = target_logz
    result = runtime._put_oldlogprob_hidden_refs(
        output,
        batch(),
    )
    assert all(
        (isinstance(payload, dict) and "_speco_cpu_tensor_wire_v1" in payload)
        == (enabled is not False)
        for payload in store.values()
    )
    for index, expected in [(0, source[0]), (2, source[1])]:
        actual = materialize(result, store, index)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    if with_logz:
        from verl_speco.integration.tensor_wire import decode_cpu_tensors

        for index, source_index in [(0, 0), (2, 1)]:
            ref = result[runtime.OLD_LOGPROB_HIDDEN_REFS_KEY][index]
            decoded = decode_cpu_tensors(store[ref])
            meta = result[runtime.OLD_LOGPROB_HIDDEN_REF_META_KEY][index]
            start, length = meta["chunk_start"], meta["chunk_length"]
            actual = decoded["target_logz"][start : start + length]
            assert actual.dtype == torch.float32
            assert torch.equal(
                actual.reshape(-1), target_logz[source_index].reshape(-1)
            )


def test_transport_config_cache_observes_runtime_switch_changes(monkeypatch):
    name = "VERL_SPECO_DRAFTER_CONFIG"
    for enabled in (True, False, True):
        monkeypatch.setenv(
            name, json.dumps({"training": {"gpu_transfer_optimizations": enabled}})
        )
        raw = runtime.get_drafter_config_env()
        assert runtime._gpu_collection_transport_enabled(raw) is enabled
