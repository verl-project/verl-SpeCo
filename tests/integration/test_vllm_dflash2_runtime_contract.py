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
"""vLLM co-training contract for the DFlash2 drafter (served through DFlash)."""

from __future__ import annotations

import json

import pytest

from verl_speco.integration import vllm_runtime
from verl_speco.integration.vllm_runtime import (
    _assert_vllm_supports_dflash2,
    _dflash2_engine_param_name,
    _draft_param_name_candidates,
    _normalize_dflash2_runtime_aliases,
    _validate_vllm_dflash2_block_size,
    _validate_vllm_dflash_drafter_config,
    build_vllm_speculative_config_from_drafter,
)

_DFLASH2_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "dflash_config": {
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "mask_token_id": 151669,
        "target_layer_ids": [0, 8, 16, 24, 32],
    },
}


def _write_drafter(tmp_path, config, name="dflash2-drafter"):
    model_path = tmp_path / name
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_path


def _drafter(model_path, **overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DFLASH2",
        "model_path": str(model_path),
        "rollout": {"spec_steps": 1, "spec_verify_tokens": 7},
        "training": {"dflash2_block_size": 8},
        "vllm": {},
    }
    config.update(overrides)
    return config


@pytest.fixture
def vllm_has_dflash2(monkeypatch):
    monkeypatch.setattr(vllm_runtime, "_vllm_supports_dflash2", lambda: True)


def test_dflash2_speculative_config_is_the_dflash_contract(
    tmp_path, vllm_has_dflash2
) -> None:
    model_path = _write_drafter(tmp_path, _DFLASH2_CONFIG)

    config = build_vllm_speculative_config_from_drafter(_drafter(model_path))

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": str(model_path),
        "num_speculative_tokens": 7,
    }


def test_dflash2_speculative_config_needs_spec_verify_tokens(
    tmp_path, vllm_has_dflash2
) -> None:
    model_path = _write_drafter(tmp_path, _DFLASH2_CONFIG)

    with pytest.raises(ValueError, match="spec_verify_tokens"):
        build_vllm_speculative_config_from_drafter(
            _drafter(model_path, rollout={"spec_steps": 3})
        )


def test_dflash2_speculative_config_refuses_a_vllm_without_dflash2(
    tmp_path, monkeypatch
) -> None:
    model_path = _write_drafter(tmp_path, _DFLASH2_CONFIG)
    monkeypatch.setattr(vllm_runtime, "_vllm_supports_dflash2", lambda: False)

    with pytest.raises(ValueError, match="vLLM 0.28.0"):
        build_vllm_speculative_config_from_drafter(_drafter(model_path))


def test_dflash2_capability_probe_is_skipped_without_vllm(monkeypatch) -> None:
    monkeypatch.setattr(vllm_runtime, "_vllm_supports_dflash2", lambda: None)
    _assert_vllm_supports_dflash2()


def test_dflash2_capability_probe_reads_the_installed_vllm(monkeypatch) -> None:
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "vllm":
            return object()
        if name == vllm_runtime._VLLM_DFLASH2_MODULE:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert vllm_runtime._vllm_supports_dflash2() is False

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert vllm_runtime._vllm_supports_dflash2() is None


def test_dflash2_validator_rejects_a_plain_dflash_checkpoint(tmp_path) -> None:
    """A DFlash checkpoint would load and silently serve without the DFlash2 modules."""
    model_path = _write_drafter(
        tmp_path,
        {"architectures": ["DFlashDraftModel"], **_DFLASH2_CONFIG["dflash_config"]},
    )

    with pytest.raises(ValueError, match="DFlash2 drafter checkpoint"):
        _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH2")


def test_dflash2_validator_requires_the_runtime_hyperparameters(tmp_path) -> None:
    config = json.loads(json.dumps(_DFLASH2_CONFIG))
    del config["dflash_config"]["selector_top_k"]
    del config["dflash_config"]["conv_group_size"]
    model_path = _write_drafter(tmp_path, config)

    with pytest.raises(
        ValueError, match=r"missing \['conv_group_size', 'selector_top_k'\]"
    ):
        _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH2")


def test_dflash2_validator_accepts_top_level_hyperparameters(tmp_path) -> None:
    """Configs saved by the trainer keep the knobs flat; the alias patch nests them."""
    config = {
        "architectures": ["Qwen3DFlash2Model"],
        **_DFLASH2_CONFIG["dflash_config"],
    }
    model_path = _write_drafter(tmp_path, config)

    _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH2")


def test_dflash2_validator_skips_a_missing_checkpoint(tmp_path) -> None:
    _validate_vllm_dflash_drafter_config(tmp_path / "absent", algorithm="DFLASH2")
    _validate_vllm_dflash_drafter_config(None, algorithm="DFLASH2")


def test_dflash2_block_size_must_match_the_engine_block(
    tmp_path, vllm_has_dflash2
) -> None:
    model_path = _write_drafter(tmp_path, _DFLASH2_CONFIG)

    with pytest.raises(ValueError, match="spec_verify_tokens=16 but block_size=8"):
        build_vllm_speculative_config_from_drafter(
            _drafter(model_path, rollout={"spec_verify_tokens": 16})
        )


def test_dflash2_block_size_falls_back_to_the_checkpoint() -> None:
    config = json.loads(json.dumps(_DFLASH2_CONFIG))
    _validate_vllm_dflash2_block_size(config, {"training": {}}, 7)
    with pytest.raises(ValueError, match="block_size=8"):
        _validate_vllm_dflash2_block_size(config, {"training": {}}, 3)
    # Training config wins over the checkpoint.
    _validate_vllm_dflash2_block_size(
        config, {"training": {"dflash2_block_size": 4}}, 3
    )
    # Nothing to compare against: accept.
    _validate_vllm_dflash2_block_size(None, {"training": {}}, 3)
    _validate_vllm_dflash2_block_size({"architectures": ["DFlash2DraftModel"]}, {}, 3)


def test_dflash2_codebooks_publish_under_the_engine_parameter_names() -> None:
    """Trainer nn.Embedding ``.weight`` -> vLLM bare ``nn.Parameter``."""
    candidates = _draft_param_name_candidates(
        "draft_model.candidate_selector.predecessor_codebook.weight"
    )
    assert "model.candidate_selector.predecessor_codebook" in candidates
    assert "candidate_selector.predecessor_codebook" in candidates
    # The spelled-out name is still tried first for engines that keep the module.
    assert candidates.index(
        "candidate_selector.predecessor_codebook.weight"
    ) < candidates.index("candidate_selector.predecessor_codebook")

    successor = _draft_param_name_candidates(
        "module.draft_model.candidate_selector.successor_codebook.weight"
    )
    assert "model.candidate_selector.successor_codebook" in successor

    # Every other DFlash2 parameter already matches the engine spelling.
    projection = _draft_param_name_candidates(
        "draft_model.candidate_selector.hidden_projection.weight"
    )
    assert "model.candidate_selector.hidden_projection.weight" in projection
    assert not any(name.endswith("hidden_projection") for name in projection)
    conv = _draft_param_name_candidates(
        "draft_model.layers.0.attention_conv.base_kernel"
    )
    assert "model.layers.0.attention_conv.base_kernel" in conv


def test_dflash2_runtime_aliases_nest_flat_hyperparameters() -> None:
    config = {
        "architectures": ["DFlash2DraftModel"],
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "dflash_config": {"target_layer_ids": [0, 8]},
    }

    assert _normalize_dflash2_runtime_aliases(config) is True
    assert config["dflash_config"] == {
        "target_layer_ids": [0, 8],
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    }
    # Idempotent once nested.
    assert _normalize_dflash2_runtime_aliases(config) is False


def test_dflash2_runtime_aliases_create_the_nested_block_and_ignore_others() -> None:
    config = {"architectures": ["Qwen3DFlash2Model"], "selector_top_k": 4}
    assert _normalize_dflash2_runtime_aliases(config) is True
    assert config["dflash_config"] == {"selector_top_k": 4}

    dflash = {"architectures": ["DFlashDraftModel"], "selector_top_k": 4}
    assert _normalize_dflash2_runtime_aliases(dflash) is False
    assert "dflash_config" not in dflash

    assert (
        _normalize_dflash2_runtime_aliases({"architectures": "DFlash2DraftModel"})
        is False
    )


def test_dflash2_runtime_aliases_refuse_conflicts() -> None:
    config = {
        "architectures": ["DFlash2DraftModel"],
        "selector_top_k": 16,
        "dflash_config": {"selector_top_k": 8},
    }
    with pytest.raises(ValueError, match="selector_top_k conflicts"):
        _normalize_dflash2_runtime_aliases(config)

    with pytest.raises(TypeError, match="must be a mapping"):
        _normalize_dflash2_runtime_aliases(
            {"architectures": ["DFlash2DraftModel"], "dflash_config": "bad"}
        )


def test_dflash2_engine_param_name_renames_only_the_codebooks() -> None:
    """The IPC receiver hands vLLM's load_weights the engine spelling."""
    assert (
        _dflash2_engine_param_name("candidate_selector.predecessor_codebook.weight")
        == "candidate_selector.predecessor_codebook"
    )
    assert (
        _dflash2_engine_param_name("candidate_selector.successor_codebook.weight")
        == "candidate_selector.successor_codebook"
    )
    for untouched in (
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "layers.0.attention_conv.base_kernel",
        "layers.0.mlp_conv.kernel_projection.weight",
        "fc.weight",
    ):
        assert _dflash2_engine_param_name(untouched) == untouched
