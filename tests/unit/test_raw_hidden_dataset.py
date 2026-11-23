# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Raw dataset validation, exact tensor preservation and commit boundaries."""

import errno
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from verl_speco.producer.hidden_states_store import (
    RawHiddenDatasetWriter,
    load_raw_hidden_manifest,
    prepare_raw_hidden_sample,
)
from verl_speco.trainer.target_feature_replay import FeatureContract


def sample_parts():
    payload = {
        "token_ids": torch.tensor([10, 20]),
        "hidden_states": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
    }
    request = SimpleNamespace(
        sample_id="sample",
        sequence_no=0,
        prompt_token_ids=[10, 20],
        input_ids=payload["token_ids"],
        loss_mask=torch.tensor([0.0, 1.0]),
        position_ids=torch.tensor([0, 1]),
        feature_positions=torch.tensor([0, 1]),
        draft_position_ids=torch.tensor([1, 2]),
        source_metadata={},
    )
    contract = FeatureContract(
        algorithm="DSPARK",
        target_layer_ids=[0],
        vllm_aux_hidden_state_layer_ids=[1],
        hidden_states_layout="dflash_aux_plus_last",
        dtype=torch.bfloat16,
        target_model_id="target",
        target_model_revision="rev",
        tokenizer_fingerprint="tok",
        require_full_alignment=True,
        target_num_hidden_layers=3,
    )
    return payload, request, contract


def test_original_tensor_identity_and_file_values_are_preserved(tmp_path):
    payload, request, contract = sample_parts()
    sample = prepare_raw_hidden_sample(
        SimpleNamespace(payload=payload), request, contract
    )
    assert sample.payload["hidden_states"] is payload["hidden_states"]
    writer = RawHiddenDatasetWriter(tmp_path)
    name = writer.write_many([sample])[0]
    saved = load_file(str(tmp_path / name))
    assert saved["hidden_states"].shape == (2, 2, 4)
    assert saved["hidden_states"].dtype == torch.float32
    assert torch.equal(saved["hidden_states"], payload["hidden_states"])
    entry = load_raw_hidden_manifest(tmp_path)[0]
    assert entry["request"]["feature_positions"] == [0, 1]
    assert entry["feature_contract"]["dtype"] == "torch.bfloat16"
    assert (
        json.loads((tmp_path / "metadata.json").read_text())["last_hidden_state_norm"]
        == "raw_target_residual"
    )


@pytest.mark.parametrize("fault", ["tokens", "layers", "rows", "nan"])
def test_invalid_connector_payload_is_rejected(fault):
    payload, request, contract = sample_parts()
    if fault == "tokens":
        payload["token_ids"] = torch.tensor([11, 20])
    elif fault == "layers":
        payload["hidden_states"] = payload["hidden_states"][:, :1]
    elif fault == "rows":
        payload["hidden_states"] = payload["hidden_states"][:1]
    else:
        payload["hidden_states"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        prepare_raw_hidden_sample(SimpleNamespace(payload=payload), request, contract)


def test_normalized_dataset_cannot_be_appended_as_raw(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"format": "vllm_safetensors", "schema_version": 1})
    )
    with pytest.raises(ValueError, match="normalized"):
        RawHiddenDatasetWriter(tmp_path)


def test_orphan_file_is_not_a_commit_and_is_never_overwritten(tmp_path):
    payload, request, contract = sample_parts()
    RawHiddenDatasetWriter(tmp_path)
    orphan = tmp_path / "hs_000000.safetensors"
    orphan.write_bytes(b"orphan")
    assert load_raw_hidden_manifest(tmp_path) == []
    writer = RawHiddenDatasetWriter(tmp_path)
    name = writer.write_many(
        [prepare_raw_hidden_sample(SimpleNamespace(payload=payload), request, contract)]
    )[0]
    assert name == "hs_000001.safetensors"
    assert orphan.read_bytes() == b"orphan"
    (tmp_path / name).unlink()
    with pytest.raises(FileNotFoundError):
        load_raw_hidden_manifest(tmp_path)


@pytest.mark.parametrize(
    "layers,include_final,valid",
    [(2, True, True), (1, False, True), (2, False, False), (1, True, False)],
)
def test_capture_plan_matches_service_layers_without_training_settings(
    layers, include_final, valid
):
    payload, request, previous = sample_parts()
    payload["hidden_states"] = payload["hidden_states"][:, :layers]
    captured = {
        "contract_type": "raw_capture",
        "vllm_aux_hidden_state_layer_ids": [1],
        "include_final_hidden": include_final,
        "target_num_hidden_layers": 3,
        "target_model_id": previous.target_model_id,
    }
    if valid:
        sample = prepare_raw_hidden_sample(
            SimpleNamespace(payload=payload), request, captured
        )
        assert sample.feature_contract["raw_hidden_dtype"] == "torch.float32"
        assert "algorithm" not in sample.feature_contract
    else:
        with pytest.raises(ValueError, match="capture plan"):
            prepare_raw_hidden_sample(
                SimpleNamespace(payload=payload), request, captured
            )


def test_adopted_file_is_unchanged_and_only_committed_by_writer(tmp_path, monkeypatch):
    import safetensors.torch

    payload, request, contract = sample_parts()
    source = tmp_path / "connector.safetensors"
    save_file(payload, str(source))
    original = source.read_bytes()
    writer = RawHiddenDatasetWriter(tmp_path / "dataset")

    def unexpected(*args, **kwargs):
        pytest.fail("File adoption must not load or serialize the hidden payload")

    monkeypatch.setattr(safetensors.torch, "load_file", unexpected)
    monkeypatch.setattr(safetensors.torch, "save_file", unexpected)
    monkeypatch.setattr(torch, "isfinite", unexpected)
    timings = {}
    entry = writer.stage_file(source, request, contract, timings=timings)
    assert not source.exists()
    assert (writer.path / entry["path"]).read_bytes() == original
    assert load_raw_hidden_manifest(writer.path) == []
    writer.commit(entry)
    assert load_raw_hidden_manifest(writer.path) == [entry]
    assert timings["file_bytes"] == len(original)
    assert timings["cross_fs"] == 0
    assert "file_inspect" in timings and "file_move" in timings
    assert "file_bytes" not in entry


@pytest.mark.parametrize("validate", [False, True])
def test_file_full_numerical_validation_is_optional(tmp_path, validate):
    payload, request, contract = sample_parts()
    payload["hidden_states"][0, 0, 0] = float("nan")
    source = tmp_path / "connector.safetensors"
    save_file(payload, str(source))
    writer = RawHiddenDatasetWriter(tmp_path / "dataset")
    if validate:
        with pytest.raises(ValueError, match="finite"):
            writer.stage_file(source, request, contract, validate_outputs=True)
        assert source.exists()
    else:
        writer.commit(writer.stage_file(source, request, contract))
        assert len(load_raw_hidden_manifest(writer.path)) == 1


@pytest.mark.parametrize("fault", ["tokens", "layers", "rows"])
def test_fast_file_path_still_checks_alignment(tmp_path, fault):
    payload, request, contract = sample_parts()
    if fault == "tokens":
        payload["token_ids"] = torch.tensor([11, 20])
    elif fault == "layers":
        payload["hidden_states"] = payload["hidden_states"][:, :1]
    else:
        payload["hidden_states"] = payload["hidden_states"][:1]
    source = tmp_path / "connector.safetensors"
    save_file({key: value.contiguous() for key, value in payload.items()}, str(source))
    writer = RawHiddenDatasetWriter(tmp_path / "dataset")
    with pytest.raises(ValueError):
        writer.stage_file(source, request, contract)
    assert source.exists()
    assert load_raw_hidden_manifest(writer.path) == []


@pytest.mark.parametrize("fail_copy", [False, True])
def test_cross_filesystem_copy_is_atomic_and_failed_copy_is_not_committed(
    tmp_path, monkeypatch, fail_copy
):
    from verl_speco.producer import hidden_states_store as stores

    payload, request, contract = sample_parts()
    source = tmp_path / "connector.safetensors"
    save_file(payload, str(source))
    original = source.read_bytes()
    writer = RawHiddenDatasetWriter(tmp_path / "dataset")
    replace, copyfile = stores.os.replace, stores.shutil.copyfile

    def cross_device(src, dst):
        if str(src) == str(source):
            raise OSError(errno.EXDEV, "cross filesystem")
        return replace(src, dst)

    def copy(src, dst):
        assert not list(writer.path.glob("*.safetensors"))
        assert load_raw_hidden_manifest(writer.path) == []
        if fail_copy:
            dst.write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "disk full")
        return copyfile(src, dst)

    monkeypatch.setattr(stores.os, "replace", cross_device)
    monkeypatch.setattr(stores.shutil, "copyfile", copy)
    if fail_copy:
        with pytest.raises(OSError, match="disk full"):
            writer.stage_file(source, request, contract)
        assert source.read_bytes() == original
        assert not list(writer.path.glob("*.safetensors"))
    else:
        timings = {}
        entry = writer.stage_file(source, request, contract, timings=timings)
        assert not source.exists()
        assert (writer.path / entry["path"]).read_bytes() == original
        writer.commit(entry)
        assert timings["cross_fs"] == 1
        assert timings["file_move"] >= timings["file_copy"] >= 0
    assert not list(writer.path.glob("*.tmp"))


def test_parallel_file_staging_uses_unique_names_and_preserves_orphans(tmp_path):
    payload, request, contract = sample_parts()
    root = tmp_path / "dataset"
    RawHiddenDatasetWriter(root)
    (root / "hs_000000.safetensors").write_bytes(b"orphan")
    writer = RawHiddenDatasetWriter(root)
    sources = [tmp_path / f"connector-{index}.safetensors" for index in range(12)]
    for source in sources:
        save_file(payload, str(source))
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(
            pool.map(lambda path: writer.stage_file(path, request, contract), sources)
        )
    assert len({entry["path"] for entry in entries}) == len(sources)
    assert load_raw_hidden_manifest(root) == []
    for entry in entries:
        writer.commit(entry)
    assert len(load_raw_hidden_manifest(root)) == len(sources)
    assert (root / "hs_000000.safetensors").read_bytes() == b"orphan"


def test_safetensors_save_failure_is_fatal_io_error_without_commit(
    tmp_path, monkeypatch
):
    import safetensors.torch
    from safetensors import SafetensorError

    payload, request, contract = sample_parts()
    sample = prepare_raw_hidden_sample(
        SimpleNamespace(payload=payload), request, contract
    )
    writer = RawHiddenDatasetWriter(tmp_path / "dataset")

    def fail(*args, **kwargs):
        raise SafetensorError("disk full")

    monkeypatch.setattr(safetensors.torch, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        writer.stage_sample(sample)
    assert load_raw_hidden_manifest(writer.path) == []
    assert not list(writer.path.glob("*.tmp"))
