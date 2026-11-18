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
"""Actual CPU tensor round-trips through unchanged SpeCo storage and TQ codecs."""

import copy
import json

import pytest
import torch

from verl_speco.trainer.deepseek_v4_contract import (
    EXTENSION,
    DeepSeekV4Contract,
    from_engine_tensors,
    to_engine_tensors,
)
from verl_speco.trainer.deepseek_v4_dataset import import_capture
from verl_speco.trainer.feature_store import TorchShardFeatureStore
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ExpectedFeatureConfig,
    SampleMetadata,
    decode_sample,
    encode_sample,
    make_ready_tag,
    make_sample_key,
)


@pytest.fixture
def capture(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-0731-bf16"
    model.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "hc_mult": 2,
        "vocab_size": 32,
        "num_hidden_layers": 5,
        "dspark_target_layer_ids": [0, 2, 4],
        "dspark_block_size": 2,
    }
    (model / "config.json").write_text(json.dumps(config))
    contract = DeepSeekV4Contract.from_model(model)
    values = {
        "input_ids": torch.arange(8, dtype=torch.int64),
        "positions": torch.arange(100, 108, dtype=torch.int64),
        "hidden_states": torch.arange(96, dtype=torch.float32).reshape(8, 12),
        "target_head_hidden_states": torch.arange(64, dtype=torch.float32).reshape(
            8, 8
        ),
    }
    loss_mask = [0, 0, 1, 1, 1, 1, 1, 1]
    root = tmp_path / "capture"
    request = root / "request_000000"
    request.mkdir(parents=True)
    for name, tensor in values.items():
        torch.save(tensor, request / f"step_0_{name}.cpu.pt")
    masks = tmp_path / "chats.jsonl"
    masks.write_text(
        json.dumps({"sample_idx": "request_000000", "loss_mask": loss_mask}) + "\n"
    )
    return model, contract, values, loss_mask, root, masks


def test_import_and_tq_roundtrip_preserve_pre_hc_and_absolute_positions(
    capture, tmp_path
):
    model, contract, values, loss_mask, root, masks = capture
    output = tmp_path / "store"
    report = import_capture(
        model=str(model), capture=str(root), masks=str(masks), output=str(output)
    )
    assert report["samples"] == 1 and report["tokens"] == 8
    store = TorchShardFeatureStore(output, read_only=True)
    sample = store.read(next(store.iter_keys()))
    expected_loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
    assert torch.equal(sample.loss_mask, expected_loss_mask)
    meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "smoke", "request-0", 0)
    restored = decode_sample(
        make_sample_key(meta),
        make_ready_tag(meta),
        encode_sample(sample, meta),
        ExpectedFeatureConfig(run_id="smoke"),
    )
    assert torch.equal(restored.loss_mask, expected_loss_mask)
    for name, tensor in to_engine_tensors(restored, contract).items():
        expected = (
            torch.tensor(loss_mask, dtype=torch.float32)
            if name == "loss_mask"
            else values[name]
        )
        assert torch.equal(tensor, expected), name
    store.close()


@pytest.mark.parametrize(
    "error", ["mask", "position", "hidden", "teacher", "finite", "token", "layout"]
)
def test_bad_features_fail_closed(capture, error):
    _, contract, values, loss_mask, _, _ = capture
    sample = from_engine_tensors(values, loss_mask, contract, sample_id="sample")
    if error == "mask":
        sample.loss_mask[0] = 0.5
    elif error == "position":
        sample.position_ids[3] += 1
    elif error == "hidden":
        sample.hidden_states = sample.hidden_states[:, :-1]
    elif error == "teacher":
        sample.metadata[EXTENSION]["target_head_hidden_states"] = sample.hidden_states
    elif error == "finite":
        sample.metadata[EXTENSION]["target_head_hidden_states"][0, 0] = float("nan")
    elif error == "token":
        sample.input_ids[0] = 32
    elif error == "layout":
        sample.metadata[EXTENSION]["contract"]["target_offset"] = 1
    with pytest.raises(ValueError):
        contract.validate(sample)


def test_missing_teacher_and_malformed_mask_are_rejected(capture):
    _, contract, values, loss_mask, _, _ = capture
    missing = copy.copy(values)
    missing.pop("target_head_hidden_states")
    with pytest.raises(KeyError):
        from_engine_tensors(missing, loss_mask, contract, sample_id="x")
    with pytest.raises(ValueError, match="loss_mask"):
        from_engine_tensors(values, loss_mask[:-1], contract, sample_id="x")


def test_legacy_mask_field_is_rejected(capture, tmp_path):
    model, _, _, loss_mask, root, masks = capture
    masks.write_text(
        json.dumps(
            {"sample_idx": "request_000000", "prompt_generation_mask": loss_mask}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="require SpeCo loss_mask"):
        import_capture(
            model=str(model),
            capture=str(root),
            masks=str(masks),
            output=str(tmp_path / "store"),
        )


def test_import_is_non_destructive_and_never_commits_partial_store(capture, tmp_path):
    model, _, _, _, root, masks = capture
    output = tmp_path / "store"
    output.mkdir()
    sentinel = output / "existing"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        import_capture(
            model=str(model), capture=str(root), masks=str(masks), output=str(output)
        )
    assert sentinel.read_text() == "keep"
    (root / "request_000000" / "step_0_target_head_hidden_states.cpu.pt").unlink()
    with pytest.raises(FileNotFoundError):
        import_capture(
            model=str(model),
            capture=str(root),
            masks=str(masks),
            output=str(tmp_path / "new"),
        )
    assert not (tmp_path / "new").exists()


def test_ordinal_mapping_is_explicit_and_duplicates_are_rejected(capture, tmp_path):
    model, _, _, loss_mask, root, masks = capture
    row = json.dumps({"sample_idx": "custom", "loss_mask": loss_mask})
    masks.write_text(row + "\n")
    with pytest.raises(ValueError, match="ordinal"):
        import_capture(
            model=str(model),
            capture=str(root),
            masks=str(masks),
            output=str(tmp_path / "no"),
        )
    assert (
        import_capture(
            model=str(model),
            capture=str(root),
            masks=str(masks),
            output=str(tmp_path / "yes"),
            allow_ordinal=True,
        )["samples"]
        == 1
    )
    masks.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        import_capture(
            model=str(model),
            capture=str(root),
            masks=str(masks),
            output=str(tmp_path / "duplicate"),
        )


def test_qwen_config_is_not_accepted_as_ds4(capture):
    model, _, _, _, _, _ = capture
    path = model / "config.json"
    config = json.loads(path.read_text())
    config["model_type"] = "qwen3"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="deepseek_v4"):
        DeepSeekV4Contract.from_model(model)
