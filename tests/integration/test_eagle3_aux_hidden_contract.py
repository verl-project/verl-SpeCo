from __future__ import annotations

import json

import pytest


def _minimal_eagle3_config(**overrides) -> dict:
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 128,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "target_hidden_size": 4,
        "vocab_size": 32,
    }
    config.update(overrides)
    return config


def test_qwen3_eagle3_config_alias_preserves_five_target_layers(tmp_path) -> None:
    pytest.importorskip("transformers")
    from verl_speco.models.auto import AutoDraftModelConfig

    config_path = tmp_path / "config.json"
    config = _minimal_eagle3_config(
        architectures=["Qwen3Eagle3Model"],
        target_layer_ids=[1, 9, 17, 25, 33],
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AutoDraftModelConfig.from_file(str(config_path))

    assert loaded.architectures == ["LlamaForCausalLMEagle3"]
    assert loaded.target_layer_ids == [1, 9, 17, 25, 33]
    assert loaded.eagle_aux_hidden_state_layer_ids == [1, 9, 17, 25, 33]
    assert loaded.eagle_config["eagle_aux_hidden_state_layer_ids"] == [1, 9, 17, 25, 33]
    assert loaded.num_aux_hidden_states == 5


def test_eagle3_model_uses_dynamic_aux_hidden_count() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3

    config = AutoDraftModelConfig._config_mapping["LlamaForCausalLMEagle3"].from_dict(
        _minimal_eagle3_config(
            target_hidden_size=4,
            target_layer_ids=[1, 9, 17, 25, 33],
            num_aux_hidden_states=5,
        )
    )
    model = LlamaForCausalLMEagle3(config)

    assert model.num_aux_hidden_states == 5
    assert model.fc.in_features == 20
    projected = model.project_hidden_states(torch.randn(2, 3, 20))
    assert projected.shape == (2, 3, config.hidden_size)

    with pytest.raises(ValueError, match="num_aux_hidden_states=5"):
        model.project_hidden_states(torch.randn(2, 3, 12))
