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
"""CPU tests for the DSV4-Flash DSpark standalone (independent) trainer backend."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

dsv4_backend = pytest.importorskip(
    "verl_speco.backends.dsv4_dspark_trainer_backend"
)
dsv4_models = pytest.importorskip("verl_speco.models.dsv4_dspark")
dsv4_weights = pytest.importorskip("verl_speco.models.dsv4_dspark.weights")
dspark_backend = pytest.importorskip("verl_speco.backends.dspark_trainer_backend")
factory = pytest.importorskip("verl_speco.backends.factory")
oldlogprob = pytest.importorskip("verl_speco.integration.oldlogprob_layer_ids")

DSV4DSparkTrainerBackend = dsv4_backend.DSV4DSparkTrainerBackend
DSV4DSparkConfig = dsv4_models.DSV4DSparkConfig
DSV4DSparkDraftModel = dsv4_models.DSV4DSparkDraftModel
DSparkTrainingModel = dspark_backend.DSparkTrainingModel


def _tiny_dsv4_config(**overrides) -> DSV4DSparkConfig:
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        target_hidden_size=64,
        target_num_hidden_layers=8,
        num_context_layers=2,
        target_layer_ids=[5, 6],
        mask_token_id=127,
        block_size=4,
        num_anchors=2,
        markov_rank=8,
        markov_head_type="vanilla",
        dsv4_num_heads=4,
        dsv4_head_dim=16,
        dsv4_rope_head_dim=8,
        dsv4_q_lora_rank=16,
        dsv4_o_lora_rank=16,
        dsv4_o_groups=2,
        dsv4_window_size=128,
        dsv4_rope_theta=10000.0,
        dsv4_rope_factor=16.0,
        dsv4_original_seq_len=65536,
        dsv4_beta_fast=32.0,
        dsv4_beta_slow=1.0,
        dsv4_n_routed_experts=4,
        dsv4_n_shared_experts=1,
        dsv4_n_activated_experts=2,
        dsv4_moe_inter_dim=32,
        dsv4_score_func="sqrtsoftplus",
        dsv4_route_scale=1.5,
        dsv4_swiglu_limit=10.0,
        dsv4_hc_mult=2,
        dsv4_hc_sinkhorn_iters=20,
        dsv4_hc_eps=1e-6,
    )
    base.update(overrides)
    return DSV4DSparkConfig(**base)


def _tiny_training_model(**overrides) -> DSparkTrainingModel:
    config = _tiny_dsv4_config(**overrides.pop("config_overrides", {}))
    kwargs = dict(
        block_size=4,
        num_anchors=2,
        l1_loss_alpha=0.0,
        distribution_loss_impl="eager",
    )
    kwargs.update(overrides)
    return DSparkTrainingModel(
        draft_model=DSV4DSparkDraftModel(config), **kwargs
    )


# --------------------------------------------------------------------------- #
# Registration / config wiring
# --------------------------------------------------------------------------- #


def test_factory_registers_dsv4_dspark():
    assert "DSV4_DSPARK" in factory.SUPPORTED_DRAFTER_ALGORITHMS


def test_factory_builds_dsv4_backend():
    config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(speculative_algorithm="DSV4_DSPARK")
        )
    )
    backend = factory.build_trainer_backend(config, SimpleNamespace())
    assert isinstance(backend, DSV4DSparkTrainerBackend)
    assert backend.model_type == "dsv4_dspark"


def test_dsv4_config_default_architecture():
    config = _tiny_dsv4_config()
    assert config.architectures == ["DSV4DSparkDraftModel"]
    assert config.model_type == "dsv4_dspark"
    assert config.dsv4_num_heads == 4
    assert config.dsv4_hc_mult == 2


def test_dsv4_config_pretrained_roundtrip(tmp_path):
    source = _tiny_dsv4_config().to_dict()
    source.update(
        {
            "architectures": ["DSV4DSparkDraftModel"],
            "model_type": "dsv4_dspark",
            "dsv4_num_heads": 4,
            "block_size": 4,
            "num_anchors": 2,
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(source), encoding="utf-8")

    loaded = DSV4DSparkConfig.from_dsv4_dspark_pretrained(str(tmp_path))
    assert loaded.model_type == "dsv4_dspark"
    assert loaded.dsv4_num_heads == 4
    assert loaded.block_size == 4


# --------------------------------------------------------------------------- #
# Backend config building / normalization
# --------------------------------------------------------------------------- #


def _backend(training: dict) -> DSV4DSparkTrainerBackend:
    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    backend.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training=training)),
        model=SimpleNamespace(path="/nonexistent"),
    )
    return backend


def _target_hf_config():
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )


def _fallback_training_cfg() -> dict:
    return {
        "dsv4_dspark_hidden_size": 64,
        "dsv4_dspark_num_target_layers": 2,
        "dsv4_dspark_num_hidden_layers": 2,
        "dsv4_dspark_block_size": 4,
        "dsv4_dspark_num_anchors": 2,
        "dsv4_dspark_markov_rank": 8,
        "dsv4_dspark_mask_token_id": 127,
        "dsv4_dspark_backbone_num_heads": 4,
        "dsv4_dspark_backbone_head_dim": 16,
        "dsv4_dspark_backbone_rope_head_dim": 8,
        "dsv4_dspark_backbone_q_lora_rank": 16,
        "dsv4_dspark_backbone_o_lora_rank": 16,
        "dsv4_dspark_backbone_o_groups": 2,
        "dsv4_dspark_backbone_n_routed_experts": 4,
        "dsv4_dspark_backbone_n_activated_experts": 2,
        "dsv4_dspark_backbone_moe_inter_dim": 32,
        "dsv4_dspark_backbone_hc_mult": 2,
    }


def test_dsv4_fallback_config_uses_dsv4_backbone_defaults():
    backend = _backend(_fallback_training_cfg())
    config = backend._build_fallback_config(_target_hf_config())

    assert isinstance(config, DSV4DSparkConfig)
    assert config.num_context_layers == 2
    assert config.num_hidden_layers == 2
    assert config.block_size == 4
    assert config.mask_token_id == 127
    assert config.architectures == ["DSV4DSparkDraftModel"]


def test_dsv4_normalize_config_overrides_num_context_layers():
    backend = _backend(_fallback_training_cfg())
    config = _tiny_dsv4_config()
    config.num_context_layers = 5

    normalized = backend._normalize_dflash_config(
        config, _target_hf_config(), None, ""
    )

    assert normalized.num_context_layers == 2
    assert normalized.target_layer_ids == [5, 6]


def test_dsv4_confidence_topology_reads_dsv4_keys():
    backend = _backend(
        {
            "dsv4_dspark_confidence_head_with_markov": True,
            "dsv4_dspark_confidence_head_alpha": 1.0,
            "dsv4_dspark_confidence_loss_alpha": 1.0,
            "mode": "offline",
        }
    )
    config = _tiny_dsv4_config(enable_confidence_head=False)

    normalized = backend._normalize_dflash_config(
        config, _target_hf_config(), None, ""
    )

    assert normalized.enable_confidence_head is True
    assert normalized.confidence_head_with_markov is True
    assert DSV4DSparkDraftModel(normalized).confidence_head is not None


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def _preprocess_model_config():
    return SimpleNamespace(
        pad_token_id=0,
        hidden_size=64,
        target_hidden_size=64,
        num_context_layers=2,
        num_target_layers=2,
    )


def test_dsv4_preprocess_splits_aux_plus_last():
    backend = _backend({"dsv4_dspark_max_window": 4})
    rows, h_dim, ctx = 8, 64, 2
    item = {
        "input_ids": torch.arange(rows),
        "hidden_states": torch.arange(
            rows * h_dim * (ctx + 1), dtype=torch.float32
        ).reshape(rows, h_dim * (ctx + 1)),
        "loss_mask": torch.tensor([0, 0, *([1] * (rows - 2))], dtype=torch.float32),
        "hidden_states_layout": "dflash_aux_plus_last",
    }

    result = backend.preprocess_individual_items(
        [item], torch.device("cpu"), _preprocess_model_config()
    )

    assert result["ids"][0].numel() == 4
    assert result["h_states"][0].shape == (4, ctx * h_dim)
    assert result["target_last_h_states"][0].shape == (4, h_dim)
    # The final hidden block must be sliced from the tail of the first cropped row.
    torch.testing.assert_close(
        result["target_last_h_states"][0][0],
        item["hidden_states"][0, ctx * h_dim : (ctx + 1) * h_dim],
        check_dtype=False,
    )


def test_dsv4_preprocess_rejects_wrong_hidden_dim():
    backend = _backend({"dsv4_dspark_max_window": 0})
    item = {
        "input_ids": torch.arange(4),
        "hidden_states": torch.randn(4, 128),
        "loss_mask": torch.ones(4),
        "hidden_states_layout": "dflash_aux_plus_last",
    }

    with pytest.raises(ValueError, match="exactly 192 hidden dims"):
        backend.preprocess_individual_items(
            [item], torch.device("cpu"), _preprocess_model_config()
        )


# --------------------------------------------------------------------------- #
# Model / training model forward-backward
# --------------------------------------------------------------------------- #


def test_dsv4_draft_model_structure():
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    state_keys = set(model.state_dict())
    assert {"fc.weight", "hidden_norm.weight", "norm.weight"}.issubset(state_keys)
    assert "hc_head.hc_fn" in state_keys
    assert any(key.startswith("layers.0.ffn.router.") for key in state_keys)
    assert model.markov_head is not None
    assert len(model.layers) == 2


def test_dsv4_training_forward_backward_cpu():
    model = _tiny_training_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 6, 64), torch.randn(1, 6, 64)]
    lm_head = torch.randn(128, 64)

    loss, accuracy, loss_pp, acc_pp, count_pp, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head,
    )

    assert torch.isfinite(loss)
    assert diagnostics["ce_weighted_token_count"].item() > 0
    loss.backward()
    finite = [
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.requires_grad
    ]
    assert finite and all(finite)


def test_dsv4_markov_head_biases_logits():
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    base_logits = torch.zeros(1, 1, 4, 128)
    prev_ids = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long)
    hidden = torch.zeros(1, 1, 4, 64)
    with torch.no_grad():
        model.markov_head.markov_w1.weight.fill_(0.5)
        model.markov_head.markov_w2.weight.fill_(0.25)

    corrected = model.apply_markov_logits(
        base_logits, prev_token_ids=prev_ids, draft_hidden=hidden
    )

    assert corrected.abs().sum().item() > 0
    assert not torch.equal(corrected, base_logits)


def test_dsv4_l1_loss_requires_target_last_hidden_states():
    model = _tiny_training_model(l1_loss_alpha=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 64), torch.randn(1, 5, 64)]

    with pytest.raises(ValueError, match="target_last_hidden_states"):
        model(
            input_ids=input_ids,
            hidden_states_list=hidden_states,
            loss_mask=loss_mask,
            lm_head_weight=torch.randn(128, 64),
            target_last_hidden_states=None,
        )


def test_dsv4_l1_loss_uses_target_last_hidden_states():
    model = _tiny_training_model(l1_loss_alpha=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 64), torch.randn(1, 5, 64)]

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=torch.randn(128, 64),
        target_last_hidden_states=torch.randn(1, 5, 64),
    )

    assert torch.isfinite(loss)
    assert diagnostics["l1_weighted_token_count"].item() > 0


# --------------------------------------------------------------------------- #
# Weight loading / key mapping / EP gate / hidden layout
# --------------------------------------------------------------------------- #


def test_dsv4_released_key_mapping():
    assert dsv4_weights.map_released_key("mtp.0.main_proj.weight") == "fc.weight"
    assert (
        dsv4_weights.map_released_key("mtp.0.main_norm.weight")
        == "hidden_norm.weight"
    )
    assert (
        dsv4_weights.map_released_key("mtp.1.ffn.experts.3.w2.weight")
        == "layers.1.ffn.experts.w2_idx_3"
    )
    assert (
        dsv4_weights.map_released_key("mtp.2.hc_head_fn")
        == "hc_head.hc_fn"
    )
    assert dsv4_weights.map_released_key("layers.0.foo.weight") is None
    assert dsv4_weights.map_released_key("mtp.0.ffn.gate.weight") == (
        "layers.0.ffn.router.weight"
    )


def test_dsv4_ep_disabled_by_default():
    from omegaconf import OmegaConf

    from verl_speco.backends.dsv4_ep_utils import is_ep_enabled

    config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DSV4_DSPARK",
                    "training": {"dsv4_dspark_enable_ep": False},
                }
            }
        }
    )
    assert is_ep_enabled(config) is False
    config.rollout.drafter.training.dsv4_dspark_enable_ep = True
    assert is_ep_enabled(config) is True
    config.rollout.drafter.speculative_algorithm = "DSPARK"
    assert is_ep_enabled(config) is False


def test_dsv4_hidden_layout_resolver_uses_dsv4_keys():
    training = {"dsv4_dspark_tv_loss_alpha": 1.8}
    assert (
        oldlogprob.resolve_drafter_hidden_states_layout("DSV4_DSPARK", training)
        == "dflash_aux_plus_last"
    )
    # Zero loss coefficients keep the aux-only layout so the feature producer
    # does not need to collect the final hidden state.
    assert (
        oldlogprob.resolve_drafter_hidden_states_layout(
            "DSV4_DSPARK", {"dsv4_dspark_tv_loss_alpha": 0.0}
        )
        == "dflash_aux"
    )


def test_dsv4_forward_rejects_flex_block_mask():
    # On CUDA the DFlash mask builder returns a flex BlockMask and no dense mask;
    # the DSV4 sink attention must fail closed instead of silently dropping it.
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())

    with pytest.raises(NotImplementedError, match="dense_attention_mask"):
        model(
            draft_input_ids=torch.zeros(1, 4, dtype=torch.long),
            context_feature=torch.randn(1, 8, 64),
            draft_position_ids=torch.zeros(1, 4, dtype=torch.long),
            context_position_ids=torch.zeros(1, 8, dtype=torch.long),
            block_mask=object(),
            dense_attention_mask=None,
        )


def test_dsv4_config_maps_noise_token_to_mask_token():
    config = DSV4DSparkConfig.from_dsv4_dspark_dict(
        {
            "architectures": ["DSV4DSparkDraftModel"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 130000,
            "dspark_noise_token_id": 128799,
        }
    )
    assert config.mask_token_id == 128799


def test_dsv4_config_lifts_aux_hidden_state_layer_ids():
    config = DSV4DSparkConfig.from_dsv4_dspark_dict(
        {
            "architectures": ["DSV4DSparkDraftModel"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 130000,
            "aux_hidden_state_layer_ids": [41, 42, 43],
        }
    )
    assert config.target_layer_ids == [41, 42, 43]


def test_dsv4_fallback_config_requires_mask_token_id():
    training = _fallback_training_cfg()
    training.pop("dsv4_dspark_mask_token_id")
    backend = _backend(training)

    with pytest.raises(ValueError, match="dsv4_dspark_mask_token_id"):
        backend._build_fallback_config(_target_hf_config())


def test_dsv4_normalize_applies_training_mask_token_id():
    training = dict(_fallback_training_cfg())
    training["dsv4_dspark_mask_token_id"] = 12345
    backend = _backend(training)
    config = _tiny_dsv4_config(mask_token_id=127)

    normalized = backend._normalize_dflash_config(
        config, _target_hf_config(), None, ""
    )

    assert normalized.mask_token_id == 12345


def test_dsv4_ep_active_requires_multi_rank_process_group():
    # A single-rank run cannot slice experts: EP is treated as inactive there
    # (full expert set, torch reference path) instead of failing at build time.
    from verl_speco.backends.dsv4_ep_utils import is_ep_active

    config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                speculative_algorithm="DSV4_DSPARK",
                training=SimpleNamespace(dsv4_dspark_enable_ep=True),
            )
        )
    )
    # No process group in this test process -> EP inactive despite the config.
    assert is_ep_active(config) is False


def test_dsv4_ep_enabled_supports_namespace_configs():
    from verl_speco.backends.dsv4_ep_utils import is_ep_enabled

    config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                speculative_algorithm="DSV4_DSPARK",
                training=SimpleNamespace(dsv4_dspark_enable_ep=True),
            )
        )
    )
    assert is_ep_enabled(config) is True
    config.rollout.drafter.speculative_algorithm = "DFLASH"
    assert is_ep_enabled(config) is False


def test_dsv4_init_router_bias_breaks_symmetry():
    from verl_speco.backends.dsv4_ep_utils import _init_router_bias

    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    router = model.layers[0].ffn.router
    assert torch.count_nonzero(router.bias).item() == 0

    _init_router_bias(model, world_size=2, experts_per_rank=2, rank=1)

    # The shared (exported) bias must stay untouched; the per-rank preference is
    # a transient attribute applied at routing time only.
    assert torch.count_nonzero(router.bias).item() == 0
    assert router._bias_initialized is True
    assert router._ep_rank_bias is not None
    assert router._ep_rank_bias.shape == (router.bias.numel(),)
    # Rank 1 owns global experts [2, 3] and prefers them (+boost), the rest -boost.
    assert router._ep_rank_bias[2] > 0 and router._ep_rank_bias[3] > 0
    assert router._ep_rank_bias[0] < 0 and router._ep_rank_bias[1] < 0


def test_dsv4_load_embedding_handles_multiple_index_and_nested_key(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    embed = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64)
    key = "model.language_model.embed_tokens.weight"
    safetensors_torch.save_file({key: embed}, str(tmp_path / "a.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "a.safetensors"}}), encoding="utf-8"
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight_scale": "a.safetensors"}}),
        encoding="utf-8",
    )

    model.load_embedding(str(tmp_path))

    torch.testing.assert_close(
        model.embed_tokens.weight, embed.to(model.embed_tokens.weight.dtype)
    )


def test_dsv4_training_cfg_view_maps_dspark_keys():
    view = dsv4_backend._DSV4TrainingCfgView(
        {
            "dsv4_dspark_confidence_head_alpha": 0.7,
            "mode": "offline",
        }
    )
    assert view.get("dspark_confidence_head_alpha") == 0.7
    assert view.get("mode") == "offline"
    # Missing dsv4 keys fall back to the shared spelling.
    assert view.get("dspark_missing_key", "default") == "default"



def test_dsv4_standalone_export_writes_dsv4_dspark_config(tmp_path):
    from verl_speco.trainer.draft_training_loop import (
        _rewrite_standalone_block_runtime_config,
    )

    config = _tiny_dsv4_config(num_context_layers=3, target_layer_ids=[40, 41, 42])
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    source_dir = tmp_path / "source_dsv4"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dsv4_dspark"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    model_path=str(source_dir),
                    training={"dsv4_dspark_target_layer_ids": [40, 41, 42]},
                )
            )
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["speco_training_model_type"] == "dsv4_dspark"
    # Decoder ids [40, 41, 42] -> vLLM output ids [41, 42, 43].
    assert runtime_config["target_layer_ids"] == [40, 41, 42]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [41, 42, 43]
    # The DSV4 backbone knobs are exported under the ``dsv4_dspark_config`` child.
    assert runtime_config["dsv4_dspark_config"]["dsv4_num_heads"] == 4
    assert runtime_config["dsv4_dspark_config"]["target_layer_ids"] == [40, 41, 42]


def test_dsv4_confidence_head_detected_in_mtp_namespaced_state():
    backend = _backend({"mode": "offline"})
    config = _tiny_dsv4_config(enable_confidence_head=False)
    # Released keys are ``mtp.*``; the resolver must still see the confidence head.
    normalized_state = {
        "mtp.0.confidence_head.proj.weight": torch.zeros(1, 64 + 8),
        "mtp.0.confidence_head.proj.bias": torch.zeros(1),
    }

    normalized = backend._normalize_dflash_config(
        config, _target_hf_config(), normalized_state, "checkpoint"
    )

    assert normalized.enable_confidence_head is True
    model = DSV4DSparkDraftModel(normalized)
    assert model.confidence_head is not None
    assert model.confidence_head.proj.in_features == 64 + 8



def test_dsv4_routed_expert_count_rejects_training_checkpoint_mismatch():
    from verl_speco.backends.dsv4_ep_utils import resolve_routed_expert_count

    config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={"dsv4_dspark_backbone_n_routed_experts": 256}
            )
        )
    )
    drafter_config = SimpleNamespace(dsv4_n_routed_experts=128)

    with pytest.raises(ValueError, match="routed-expert mismatch"):
        resolve_routed_expert_count(config, drafter_config)


def test_dsv4_routed_expert_count_prefers_checkpoint_when_config_absent():
    from verl_speco.backends.dsv4_ep_utils import resolve_routed_expert_count

    config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={}))
    )
    drafter_config = SimpleNamespace(dsv4_n_routed_experts=128)

    assert resolve_routed_expert_count(config, drafter_config) == 128


def test_dsv4_ensure_ep_divisible():
    from verl_speco.backends.dsv4_ep_utils import ensure_ep_divisible

    assert ensure_ep_divisible(256, 8) == 32
    with pytest.raises(ValueError, match="divisible"):
        ensure_ep_divisible(256, 6)


def test_dsv4_draft_depth_comes_from_training_config_not_checkpoint():
    # The released checkpoint config's num_hidden_layers is the TARGET depth
    # (e.g. 43); the draft depth must come from the training config.
    backend = _backend({"dsv4_dspark_num_hidden_layers": 3})
    assert backend._resolve_draft_depth(backend.config.rollout.drafter.training) == 3
    backend4 = _backend({"dsv4_dspark_num_hidden_layers": 4})
    assert backend4._resolve_draft_depth(backend4.config.rollout.drafter.training) == 4


def test_dsv4_from_pretrained_consumes_target_style_draft_config(tmp_path):
    # Mirrors the released DeepSeek-V4-Flash DSpark config: a TARGET config
    # (deepseek_v4, 43 layers) plus a dspark_*/hc_* draft section.
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "vocab_size": 129280,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    config = DSV4DSparkConfig.from_dsv4_dspark_pretrained(str(tmp_path))

    # Draft-section keys are consumed.
    assert config.target_layer_ids == [40, 41, 42]
    assert config.mask_token_id == 128799
    assert config.block_size == 5
    assert config.markov_rank == 256
    assert config.dsv4_hc_mult == 4
    assert config.dsv4_hc_sinkhorn_iters == 20
    assert config.dsv4_hc_eps == pytest.approx(1e-6)
    # The target depth is NOT used as the draft depth (training config does).
    assert config.num_hidden_layers == 43


def test_dsv4_preprocess_rejects_aux_only_when_target_hidden_required():
    backend = _backend({"dsv4_dspark_max_window": 0, "dsv4_dspark_tv_loss_alpha": 1.8})
    item = {
        "input_ids": torch.arange(4),
        "hidden_states": torch.randn(4, 128),
        "loss_mask": torch.ones(4),
        "hidden_states_layout": "dflash_aux",
    }

    with pytest.raises(ValueError, match="target-hidden losses"):
        backend.preprocess_individual_items(
            [item], torch.device("cpu"), _preprocess_model_config()
        )


def test_dsv4_normalize_accepts_biasless_confidence_head():
    backend = _backend({"mode": "offline"})
    config = _tiny_dsv4_config(enable_confidence_head=False)
    # Released DSV4 confidence head has only the weight (no bias key).
    state = {"mtp.2.confidence_head.proj.weight": torch.zeros(1, 64 + 8)}

    normalized = backend._normalize_dflash_config(
        config, _target_hf_config(), state, "checkpoint"
    )

    assert normalized.enable_confidence_head is True
    model = DSV4DSparkDraftModel(normalized)
    assert model.confidence_head is not None


def test_dsv4_model_freqs_use_yarn_scaling():
    from verl_speco.models.dsv4_dspark.backbone.rotary import precompute_freqs_cis

    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    bb = model.backbone_cfg
    yarn = torch.view_as_real(
        precompute_freqs_cis(
            model._rope_dim,
            bb.original_seq_len or 1,
            bb.original_seq_len,
            bb.rope_theta,
            bb.rope_factor,
            bb.beta_fast,
            bb.beta_slow,
        )
    ).contiguous()
    plain = torch.view_as_real(
        precompute_freqs_cis(
            model._rope_dim,
            bb.original_seq_len or 1,
            0,
            bb.rope_theta,
            bb.rope_factor,
            bb.beta_fast,
            bb.beta_slow,
        )
    ).contiguous()

    assert torch.allclose(model.freqs_cis, yarn)
    assert not torch.allclose(model.freqs_cis, plain)


def test_dsv4_rope_is_batch_invariant():
    # Regression: gathering the flattened positions and truncating to the first
    # ``seq_len`` rows applied batch row 0's rotation to every row.
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    model.eval()
    bsz, ctx, n = 2, 10, 8
    torch.manual_seed(0)
    ctx_feat = torch.randn(bsz, ctx, 64)
    draft_ids = torch.randint(0, 100, (bsz, n))
    # Distinct per-row draft positions (mirrors per-sample anchor sampling).
    draft_pos = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [20, 21, 22, 23, 24, 25, 26, 27]])
    ctx_pos = torch.arange(ctx).unsqueeze(0).expand(bsz, -1).contiguous()
    mask = torch.ones(bsz, 1, n, ctx + n, dtype=torch.bool)

    with torch.no_grad():
        batched = model(
            draft_ids, ctx_feat, draft_pos, ctx_pos, dense_attention_mask=mask
        )
        solo = torch.cat(
            [
                model(
                    draft_ids[i : i + 1],
                    ctx_feat[i : i + 1],
                    draft_pos[i : i + 1],
                    ctx_pos[i : i + 1],
                    dense_attention_mask=mask[i : i + 1],
                )
                for i in range(bsz)
            ],
            dim=0,
        )
    torch.testing.assert_close(batched, solo, atol=1e-5, rtol=1e-4)


def test_dsv4_router_init_is_non_degenerate():
    # Regression: ``torch.empty`` router weights could be all-zero/garbage, making
    # every token collapse onto the first ``topk`` experts.
    torch.manual_seed(0)
    model = DSV4DSparkDraftModel(_tiny_dsv4_config())
    router = model.layers[0].ffn.router
    assert router.weight.std().item() > 0
    _, indices = router(torch.randn(256, 64))
    assert set(indices.flatten().tolist()) == set(
        range(router.n_routed_experts)
    )


def test_dsv4_build_model_wires_ce_target_source():
    # Regression: the DSV4 build_model dropped ``ce_target_source`` so a
    # configured target_argmax CE was silently ignored (defaulted to labels).
    training = dict(_fallback_training_cfg())
    training["ce_target_source"] = "target_argmax"
    training["dsv4_dspark_confidence_target_mode"] = "probabilistic"
    backend = _backend(training)
    backend.config.rollout.drafter.model_path = ""
    backend._get_target_hf_config = _target_hf_config
    backend._build_target_lm_head = lambda *args, **kwargs: None
    backend._skip_weight_loading = True

    model, _ = backend.build_model()

    assert model.ce_target_source == "target_argmax"
    assert model.confidence_target_mode == "probabilistic"
