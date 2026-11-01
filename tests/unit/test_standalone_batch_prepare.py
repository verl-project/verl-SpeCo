# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import ast
import logging
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl_speco.trainer.standalone_batch_prepare import prepare_standalone_cpu_batch
from verl_speco.trainer.target_feature_replay import (
    FeatureContract, HiddenStateAlignmentError, feature_from_vllm_payload,
)
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag,
)


def _contract(algorithm="DSPARK", *, reuse_final=False):
    return FeatureContract(
        algorithm=algorithm, target_layer_ids=[0, 1],
        vllm_aux_hidden_state_layer_ids=[1, 4] if reuse_final else [1, 2],
        target_num_hidden_layers=4,
        hidden_states_layout=(
            "eagle3_aux_plus_last" if algorithm == "EAGLE3" else
            "dflash_aux" if algorithm == "DFLASH" else "dflash_aux_plus_last"
        ),
        dtype=torch.bfloat16, target_model_id="target", target_model_revision="rev",
        tokenizer_fingerprint="tokens", require_full_alignment=True,
    )


def _sample(tmp_path, contract, *, length=5, index=0, positions=None, hidden_offset=0):
    path = tmp_path / f"sample-{index}.safetensors"
    layers = 2 if contract.hidden_states_layout == "dflash_aux" or contract.vllm_aux_hidden_state_layer_ids == [1, 4] else 3
    hidden = torch.arange(length * layers * 2).reshape(length, layers, 2).float()
    save_file({"token_ids": torch.arange(length), "hidden_states": hidden[hidden_offset:]}, path)
    values = asdict(contract)
    values["dtype"] = str(contract.dtype)
    positions = list(range(length)) if positions is None else positions
    tag = make_reference_tag(
        SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index), path=str(path),
        request={
            "input_ids": list(range(length)), "loss_mask": [0] + [1] * (length - 1),
            "position_ids": list(range(length)), "feature_positions": positions,
            "draft_position_ids": [p + 1 for p in positions],
            "vllm_prompt_token_ids": list(range(length)), "source_metadata": {},
        }, feature_contract=values,
    )
    return tag, hidden


def test_dspark_padded_batch_retains_raw_final_and_files(tmp_path):
    contract = _contract()
    a, ah = _sample(tmp_path, contract)
    b, bh = _sample(tmp_path, contract, length=3, index=1)
    state = torch.random.get_rng_state().clone()
    batch = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False)
    assert torch.equal(state, torch.random.get_rng_state())
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]]
    torch.testing.assert_close(batch["hidden_states"][0], ah[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], ah[:, 2].bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][1, :3], bh[:, 2].bfloat16())
    assert torch.equal(batch["label_mask"], batch["loss_mask"])
    assert batch["last_hidden_state_norm"] == "raw_target_residual"
    assert all(t.device.type == "cpu" for t in batch.values() if isinstance(t, torch.Tensor))
    assert Path(a["hidden_states_path"]).exists() and Path(b["hidden_states_path"]).exists()


def test_eagle3_shift_and_final_layer_reuse(tmp_path):
    contract = _contract("EAGLE3", reuse_final=True)
    tag, hidden = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert batch["input_ids"].tolist() == [[1, 2, 3]]
    assert batch["loss_mask"].tolist() == [[1, 1, 1]]
    assert batch["position_ids"].tolist() == [[1, 2, 3]]
    torch.testing.assert_close(batch["last_hidden_states"][0], hidden[1:4, 1].bfloat16())
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:3].flatten(1).bfloat16())


def test_dflash_window_and_suffix_alignment(tmp_path):
    contract = _contract("DFLASH")
    tag, hidden = _sample(tmp_path, contract, length=8, positions=[3, 4, 5, 6, 7], hidden_offset=3)
    tag["request"]["loss_mask"] = [0, 0, 0, 0, 0, 0, 1, 1]
    batch = prepare_standalone_cpu_batch([tag], contract, max_window=3, pin_memory=False)
    assert batch["input_ids"].tolist() == [[5, 6, 7]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1]]
    torch.testing.assert_close(batch["hidden_states"][0], hidden[5:8].flatten(1).bfloat16())
    assert "last_hidden_state_norm" not in batch


def test_dflash_aux_plus_last_reference_batch_keeps_raw_final(tmp_path):
    contract = replace(_contract("DFLASH"), hidden_states_layout="dflash_aux_plus_last")
    tag, hidden = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], hidden[:, 2].bfloat16())
    assert batch["last_hidden_state_norm"] == "raw_target_residual"
    assert batch["hidden_source"] == "standalone_vllm_reference"


def _old_pack(chunks, block_size, limit):
    # Execute the unchanged production packer without importing distributed/model
    # dependencies. This checks parity against actual legacy code, not a copy.
    source = Path(__file__).parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DrafterBaseTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pack_block_drafter_chunks")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "logger": logging.getLogger(__name__)}
    exec(compile(module, str(source), "exec", flags=__import__('__future__').annotations.compiler_flag), namespace)
    cfg = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={"packing": {"max_packed_len": limit}})))
    trainer = SimpleNamespace(
        config=cfg, _block_drafter_config_value=lambda *args: block_size,
        _block_drafter_metric_prefix=lambda: "dspark",
    )
    keys = ["input_ids", "loss_mask", "hidden_states", "position_ids", "target_last_hidden_states"]
    return namespace[method.name](trainer, *[[c[k] for c in chunks] for k in keys])


@pytest.mark.parametrize("limit", [0, 8])
def test_document_packing_matches_existing_packer(tmp_path, limit):
    contract = _contract()
    a, _ = _sample(tmp_path, contract, length=5)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    chunks = []
    for tag in [a, b]:
        single = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
        chunks.append({k: v.squeeze(0) for k, v in single.items() if isinstance(v, torch.Tensor)})
    old = _old_pack(chunks, 4, limit)
    batch = prepare_standalone_cpu_batch(
        [a, b], contract, packing=True, block_size=4, max_packed_len=limit, pin_memory=False,
    )
    for key, expected in zip(
        ["input_ids", "loss_mask", "hidden_states", "position_ids", "attention_mask", "document_ids",
         "target_last_hidden_states"], old[:7],
    ):
        torch.testing.assert_close(batch[key], expected)
    assert batch["document_ids"][0, 5:8].tolist() == [-1, -1, -1]


def test_pin_happens_after_collation(tmp_path, monkeypatch):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    shapes = []

    def pin(tensor):
        assert tensor.device.type == "cpu"
        shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    batch = prepare_standalone_cpu_batch([tag], contract)
    assert len(shapes) == sum(isinstance(v, torch.Tensor) for v in batch.values())
    assert (1, 5, 4) in shapes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real pinned allocation requires CUDA runtime")
def test_real_pinned_batch(tmp_path):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract)
    assert all(t.is_pinned() and t.device.type == "cpu" for t in batch.values() if torch.is_tensor(t))


def test_invalid_contract_rejected_before_file_read(tmp_path):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    Path(tag["hidden_states_path"]).unlink()
    with pytest.raises(ValueError, match="contract mismatch"):
        prepare_standalone_cpu_batch([tag], replace(contract, target_model_id="other"), pin_memory=False)


def test_file_prefetch_leaves_nonfinite_handling_to_training(tmp_path, monkeypatch):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    hidden[0, 0, 0] = float("nan")
    tag["hidden_states_path"] = str(tmp_path / "nonfinite.safetensors")
    save_file({"token_ids": torch.arange(5), "hidden_states": hidden}, tag["hidden_states_path"])
    monkeypatch.setattr(torch, "isfinite", lambda *args, **kwargs: pytest.fail("CPU full hidden scan"))
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert torch.isnan(batch["hidden_states"][0, 0, 0])
    assert Path(tag["hidden_states_path"]).exists()


def test_legacy_conversion_still_rejects_nonfinite_hidden(tmp_path):
    contract = _contract()
    _, hidden = _sample(tmp_path, contract)
    hidden[0, 0, 0] = float("nan")
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    with pytest.raises(HiddenStateAlignmentError, match="NaN/Inf"):
        feature_from_vllm_payload({"token_ids": torch.arange(5), "hidden_states": hidden},
                                 request, contract, normalize_final_hidden=False)


def test_cpu_workers_prepare_assigned_samples_in_parallel_and_keep_order(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract()
    a, _ = _sample(tmp_path, contract, length=5)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    expected = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False)
    barrier = Barrier(2)
    original = FileHiddenStatesStore.load
    calls = []

    def load(store, path):
        assert not torch.is_grad_enabled()
        calls.append(path)
        barrier.wait(timeout=3)  # Sequential execution cannot pass this barrier.
        return original(store, path)

    monkeypatch.setattr(FileHiddenStatesStore, "load", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False, executor=executor)
    assert len(calls) == 2
    for key, tensor in expected.items():
        if torch.is_tensor(tensor):
            torch.testing.assert_close(actual[key], tensor)


def test_raw_conversion_preserves_old_normalized_default(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    payload = {"token_ids": torch.arange(5), "hidden_states": hidden}
    norm = torch.nn.LayerNorm(2).bfloat16()
    old = feature_from_vllm_payload(payload, request, contract, final_norm=norm)
    raw = feature_from_vllm_payload(payload, request, contract, normalize_final_hidden=False)
    assert old.metadata["last_hidden_state_norm"] == "target_final_norm"
    assert raw.metadata["last_hidden_state_norm"] == "raw_target_residual"
    torch.testing.assert_close(old.hidden_states[:, :4], raw.hidden_states[:, :4])
    torch.testing.assert_close(old.hidden_states[:, 4:], norm(raw.hidden_states[:, 4:]))
    with pytest.raises(ValueError, match="require the target final norm"):
        feature_from_vllm_payload(payload, request, contract)
    # Preserve both auxiliary values and raw targets for training sanitation.
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], hidden[:, 2].bfloat16())


def test_token_mismatch_and_insufficient_rows_preserve_files(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    save_file({"token_ids": torch.arange(5) + 1, "hidden_states": hidden}, tag["hidden_states_path"])
    with pytest.raises(HiddenStateAlignmentError, match="token_ids"):
        prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    save_file({"token_ids": torch.arange(5), "hidden_states": hidden[2:]}, tag["hidden_states_path"])
    with pytest.raises(HiddenStateAlignmentError, match="complete feature window"):
        prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert Path(tag["hidden_states_path"]).exists()


@pytest.mark.parametrize("algorithm,packing", [
    ("EAGLE3", False), ("DSPARK", False), ("DSPARK", True),
    ("DFLASH", False), ("DFLASH", True),
])
def test_contiguous_file_batch_avoids_hidden_gather_and_cat(tmp_path, monkeypatch, algorithm, packing):
    contract = _contract(algorithm)
    a, _ = _sample(tmp_path, contract)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    original = torch.Tensor.index_select

    def select(tensor, dimension, indices):
        assert tensor.dim() != 3, "Contiguous hidden rows must use a view"
        return original(tensor, dimension, indices)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("Intermediate cat in file preparation"))
    batch = prepare_standalone_cpu_batch(
        [a, b], contract, packing=packing, block_size=4, pin_memory=False,
    )
    assert batch["hidden_states"].dtype == torch.bfloat16
    assert batch["hidden_states"].shape[-1] == 4
    assert batch["hidden_states"].is_contiguous()


@pytest.mark.parametrize("algorithm", ["EAGLE3", "DSPARK", "DFLASH"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_split_features_preserve_legacy_values_and_sparse_order(tmp_path, algorithm, dtype):
    contract = replace(_contract(algorithm), dtype=dtype)
    tag, hidden = _sample(tmp_path, contract, positions=[0, 2, 4])
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.tensor([0, 2, 4]),
        draft_position_ids=torch.tensor([1, 3, 5]), source_metadata={},
    )
    payload = {"token_ids": torch.arange(5), "hidden_states": hidden}
    legacy = feature_from_vllm_payload(payload, request, contract, normalize_final_hidden=False)
    split = feature_from_vllm_payload(payload, request, contract,
                                     normalize_final_hidden=False, split_hidden_states=True)
    torch.testing.assert_close(split.input_ids, legacy.input_ids)
    torch.testing.assert_close(split.loss_mask, legacy.loss_mask)
    torch.testing.assert_close(split.hidden_states.flatten(1).bfloat16(), legacy.hidden_states[:, :4].bfloat16())
    if algorithm != "DFLASH":
        torch.testing.assert_close(split.last_hidden_states.bfloat16(), legacy.hidden_states[:, 4:].bfloat16())
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    expected = legacy.hidden_states[:, :4].bfloat16()
    if algorithm == "EAGLE3":
        expected = expected[:1]
    torch.testing.assert_close(batch["hidden_states"][0], expected)


def test_split_contiguous_features_share_source_storage(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    split = feature_from_vllm_payload(
        {"token_ids": torch.arange(5), "hidden_states": hidden}, request, contract,
        normalize_final_hidden=False, split_hidden_states=True,
    )
    pointer = hidden.untyped_storage().data_ptr()
    assert split.hidden_states.untyped_storage().data_ptr() == pointer
    assert split.last_hidden_states.untyped_storage().data_ptr() == pointer


@pytest.mark.parametrize("algorithm,packing", [("DSPARK", False), ("DSPARK", True), ("DFLASH", False), ("EAGLE3", False)])
def test_mooncake_reference_reuses_cpu_prepare_and_cleanup(tmp_path, monkeypatch, algorithm, packing):
    from verl_speco.producer import hidden_states_store as stores
    from verl_speco.trainer.feature_store import build_feature_store_from_config
    from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples
    from verl_speco.trainer.tq_feature_store import ReadyEntry
    from verl_speco.transport.drafter_sample_protocol import SampleMetadata, make_sample_key

    contract = _contract(algorithm)
    tag, hidden = _sample(tmp_path, contract)
    expected = prepare_standalone_cpu_batch([tag], contract, packing=packing, block_size=2, pin_memory=False)
    tag.pop("hidden_states_path")
    tag.update(backend="mooncake", handle="hs-1", transfer_params={"handle": "hs-1", "version": 1})

    class Store:
        backend = "mooncake"
        def __init__(self):
            self.loads = 0
            self.deletes = 0
            self.closed = False
        def load(self, handle, *, timeout=120.0):
            assert handle == "hs-1"
            self.loads += 1
            return {"hidden_states": hidden, "token_ids": torch.arange(5)}, 0
        def release(self, handle):
            assert handle == "hs-1"
            self.deletes += 1
            if self.deletes == 1:
                raise RuntimeError("transient delete failure")
        def close(self):
            self.closed = True

    store = Store()
    monkeypatch.setattr(stores, "build_hidden_states_store", lambda config: store)
    config = {"run_id": "run", "ray": {"address": "local"}, "expected_feature": {}}
    adapter = build_feature_store_from_config(
        {"type": "ray_reference"}, read_only=True, transfer_queue_cfg=config,
        hidden_states_store_cfg={"backend": "mooncake", "mooncake": {"master_server_address": "host:50051"}},
    )
    assert adapter.config["hidden_states_store"]["backend"] == "mooncake"
    adapter.connect()
    entry = ReadyEntry(make_sample_key(SampleMetadata.from_dict(tag)), tag)
    actual = adapter.prepare_batch([entry], packing=packing, block_size=2, pin_memory=False)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value
    assert store.loads == 1 and store.deletes == 0
    assert cleanup_standalone_samples(adapter, [entry], attempts=2) == []
    assert store.loads == 1 and store.deletes == 2
    adapter.close()
    assert store.closed


def test_mooncake_batch_reads_share_one_timeout_budget(tmp_path, monkeypatch):
    import verl_speco.trainer.standalone_batch_prepare as prepare
    contract = _contract()
    tags = [_sample(tmp_path, contract, index=i)[0] for i in range(2)]
    hidden = torch.zeros(5, 3, 2)
    for i, tag in enumerate(tags):
        tag.pop("hidden_states_path")
        tag.update(backend="mooncake", handle=f"hs-{i}")
    now = [100.0]
    timeouts = []

    class Store:
        backend = "mooncake"
        def load(self, handle, *, timeout):
            timeouts.append(timeout)
            now[0] += 4
            return {"hidden_states": hidden, "token_ids": torch.arange(5)}, 0

    monkeypatch.setattr(prepare, "time", SimpleNamespace(monotonic=lambda: now[0]))
    prepare.prepare_standalone_cpu_batch(tags, contract, hidden_states_store=Store(), read_timeout=10, pin_memory=False)
    assert timeouts == [10, 6]
    now[0] = 100
    timeouts.clear()
    with pytest.raises(TimeoutError, match="batch read deadline"):
        prepare.prepare_standalone_cpu_batch(tags, contract, hidden_states_store=Store(), read_timeout=3, pin_memory=False)
    assert timeouts == [3]
