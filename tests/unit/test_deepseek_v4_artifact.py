# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from verl_speco.models.deepseek_v4.artifact import (
    iter_draft_weights,
    save_draft_artifact,
)
from verl_speco.trainer.checkpoint import is_pretrained_drafter_checkpoint


@pytest.mark.parametrize("serialization", ["safetensors", "pytorch"])
def test_standard_shards_roundtrip_and_speco_metadata(tmp_path, serialization):
    weights = {
        "mtp.0.attn.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "mtp.1.hc_head_scale": torch.tensor([0.25], dtype=torch.float32),
    }
    output = tmp_path / "draft_step_1"
    report = save_draft_artifact(
        weights,
        output=output,
        config={"model_type": "deepseek_v4"},
        step=1,
        serialization=serialization,
        max_shard_bytes=10,
    )
    assert report["weights_verified"]
    assert is_pretrained_drafter_checkpoint(output)
    actual = dict(iter_draft_weights(output))
    for name, value in weights.items():
        assert actual[name].dtype == value.dtype
        assert torch.equal(actual[name], value)
    assert json.loads((output / "config.json").read_text())["speco_draft_only"]
    with pytest.raises(FileExistsError):
        save_draft_artifact(weights, output=output, config={}, step=1)


def test_rejects_bad_tensor_without_publishing(tmp_path):
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="Invalid"):
        save_draft_artifact(
            {"mtp.0.weight": torch.tensor([float("nan")])},
            output=output,
            config={},
            step=1,
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_reader_rejects_incomplete_and_unsafe_shard(tmp_path):
    save_draft_artifact(
        {"mtp.0.weight": torch.ones(2)}, output=tmp_path / "draft", config={}, step=1
    )
    root = tmp_path / "draft"
    index = root / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["mtp.0.weight"] = "../outside.safetensors"
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        list(iter_draft_weights(root))
    metadata = root / "metadata.json"
    data = json.loads(metadata.read_text())
    data["complete"] = False
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        list(iter_draft_weights(root))


def test_export_entrypoint_reuses_tp_merge_without_patching_target(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from verl_speco import export_deepseek_v4_draft as entry
    from verl_speco.models.deepseek_v4 import export as merge
    from verl_speco.models.deepseek_v4 import export_validation

    checkpoint, model = tmp_path / "checkpoint", tmp_path / "model"
    checkpoint.mkdir()
    model.mkdir()
    identity = {"hidden_size": 4}
    (checkpoint / "speco_ds4.json").write_text(
        json.dumps(
            {
                "identity": {"contract": identity, "source_fingerprint": "tested"},
                "topology": {"data_parallel_size": 2},
                "progress": {"global_step": 1},
            }
        )
    )
    (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (model / "model.safetensors").write_bytes(b"headers mocked in this test")
    monkeypatch.setattr(
        entry.DeepSeekV4Contract,
        "from_model",
        lambda _: SimpleNamespace(identity=lambda: identity),
    )
    monkeypatch.setattr(
        entry, "compatible_implementation_fingerprint", lambda value: value == "tested"
    )
    monkeypatch.setattr(merge, "_target_tensor_shapes", lambda _: {})
    calls = []
    monkeypatch.setattr(
        export_validation,
        "validate_tp_export_replicas",
        lambda path, **kw: calls.append((path, "validate", kw["dp_rank"])),
    )

    def load(path, layout, *, data_parallel_rank, target_shapes):
        calls.append((path, layout, data_parallel_rank))
        return {"mtp.0.weight": torch.arange(4)}

    monkeypatch.setattr(merge, "_load_export_weights", load)
    output = tmp_path / "draft_step_1"
    report = entry.export_checkpoint(
        checkpoint=str(checkpoint),
        model=str(model),
        output=str(output),
        dp_rank=1,
        format="draft-pytorch",
    )
    assert report["weights_verified"]
    assert calls == [
        (checkpoint.resolve(), "validate", 1),
        (checkpoint.resolve(), "dspark", 1),
    ]
    assert torch.equal(
        dict(iter_draft_weights(output))["mtp.0.weight"], torch.arange(4)
    )
    assert (model / "model.safetensors").read_bytes() == b"headers mocked in this test"
