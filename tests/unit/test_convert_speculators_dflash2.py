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
from __future__ import annotations

import json
import os

import pytest

from verl_speco.convert_speculators_dflash2 import (
    convert_speculators_dflash2_checkpoint,
    convert_speculators_dflash2_config,
    main,
)


def _speculators_config(**overrides):
    config = {
        "architectures": ["DFlash2DraftModel"],
        "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "draft_vocab_size": 151936,
        "dtype": "bfloat16",
        "mask_token_id": 151669,
        "selector_rank": 256,
        "selector_top_k": 16,
        "speculators_config": {"algorithm": "dflash2"},
        "speculators_model_type": "dflash2",
        "transformer_layer_config": {
            "attention_bias": False,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "layer_types": ["sliding_attention"] * 5,
            "max_position_embeddings": 40960,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "sliding_window": 2048,
            "use_sliding_window": True,
            "vocab_size": 151936,
        },
    }
    config.update(overrides)
    return config


_TARGET_CONFIG = {
    "num_hidden_layers": 36,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vocab_size": 151936,
}


def test_convert_emits_the_zlab_layout() -> None:
    converted = convert_speculators_dflash2_config(
        _speculators_config(), _TARGET_CONFIG
    )

    assert converted["architectures"] == ["DFlash2DraftModel"]
    assert converted["model_type"] == "qwen3"
    assert converted["is_causal"] is False
    assert converted["num_target_layers"] == 36
    assert converted["hidden_size"] == 2560
    assert converted["num_hidden_layers"] == 5
    assert converted["rope_theta"] == 1000000
    assert converted["rope_parameters"] == {
        "rope_theta": 1000000,
        "rope_type": "default",
    }
    assert converted["eos_token_id"] == 151645
    assert converted["bos_token_id"] == 151643
    assert converted["dflash_config"] == {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 151669,
        "selector_rank": 256,
        "selector_top_k": 16,
        # hidden_states indices minus one: decoder-layer indices.
        "target_layer_ids": [0, 8, 16, 24, 32],
    }
    # Speculators keys must not leak into the runtime contract.
    assert "speculators_config" not in converted
    assert "transformer_layer_config" not in converted
    assert "aux_hidden_state_layer_ids" not in converted


def test_convert_defaults_to_full_attention_but_can_keep_the_window() -> None:
    full = convert_speculators_dflash2_config(_speculators_config(), _TARGET_CONFIG)
    assert full["use_sliding_window"] is False
    assert full["sliding_window"] is None
    assert full["layer_types"] == ["full_attention"] * 5

    windowed = convert_speculators_dflash2_config(
        _speculators_config(), _TARGET_CONFIG, keep_sliding_window=True
    )
    assert windowed["use_sliding_window"] is True
    assert windowed["sliding_window"] == 2048
    assert windowed["layer_types"] == ["sliding_attention"] * 5


def test_convert_prefers_explicit_target_layer_ids() -> None:
    converted = convert_speculators_dflash2_config(
        _speculators_config(target_layer_ids=[2, 4, 6, 8, 10]), _TARGET_CONFIG
    )
    assert converted["dflash_config"]["target_layer_ids"] == [2, 4, 6, 8, 10]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"speculators_model_type": "eagle3", "speculators_config": {}},
            "not a speculators DFlash2",
        ),
        ({"transformer_layer_config": None}, "transformer_layer_config"),
        ({"aux_hidden_state_layer_ids": [1, 9, 17, 25, 40]}, "exceed the target"),
        ({"aux_hidden_state_layer_ids": [0, 9, 17, 25, 33]}, "must be >= 1"),
        ({"draft_vocab_size": 32000}, "share the target vocabulary"),
        ({"mask_token_id": None}, "lacks mask_token_id"),
        ({"selector_rank": None}, "lacks \\['selector_rank'\\]"),
    ],
)
def test_convert_rejects_malformed_configs(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        convert_speculators_dflash2_config(
            _speculators_config(**overrides), _TARGET_CONFIG
        )


def test_convert_rejects_missing_layer_ids() -> None:
    config = _speculators_config()
    del config["aux_hidden_state_layer_ids"]
    with pytest.raises(ValueError, match="neither target_layer_ids"):
        convert_speculators_dflash2_config(config, _TARGET_CONFIG)


def _write_dirs(tmp_path):
    source = tmp_path / "speculators"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "config.json").write_text(
        json.dumps(_speculators_config()), encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (target / "config.json").write_text(json.dumps(_TARGET_CONFIG), encoding="utf-8")
    return source, target


def test_convert_checkpoint_writes_config_and_copies_weights(tmp_path) -> None:
    source, target = _write_dirs(tmp_path)
    output = tmp_path / "converted"

    converted = convert_speculators_dflash2_checkpoint(
        str(source), str(target), str(output)
    )

    written = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert written == converted
    assert (output / "model.safetensors").read_bytes() == b"weights"
    assert not os.path.islink(output / "model.safetensors")


def test_convert_checkpoint_can_symlink_weights_and_overwrites(tmp_path) -> None:
    source, target = _write_dirs(tmp_path)
    output = tmp_path / "converted"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"stale")

    convert_speculators_dflash2_checkpoint(
        str(source), str(target), str(output), link_weights=True
    )

    assert os.path.islink(output / "model.safetensors")
    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_convert_checkpoint_requires_weights(tmp_path) -> None:
    source, target = _write_dirs(tmp_path)
    (source / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        convert_speculators_dflash2_checkpoint(
            str(source), str(target), str(tmp_path / "out")
        )


def test_main_converts_and_reports(tmp_path, capsys) -> None:
    source, target = _write_dirs(tmp_path)
    output = tmp_path / "converted"

    main(["--input", str(source), "--target", str(target), "--output", str(output)])

    assert (output / "config.json").exists()
    assert "DFlash2DraftModel" in capsys.readouterr().out
