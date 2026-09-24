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

import pytest

torch = pytest.importorskip("torch")

dspark_backend = pytest.importorskip("verl_speco.backends.dspark_trainer_backend")
dspark_models = pytest.importorskip("verl_speco.models.dspark")
dflash_backend = pytest.importorskip("verl_speco.backends.dflash_trainer_backend")
dflash_models = pytest.importorskip("verl_speco.models.dflash")

DSparkTrainingModel = dspark_backend.DSparkTrainingModel
DSparkConfig = dspark_models.DSparkConfig
DSparkDraftModel = dspark_models.DSparkDraftModel
create_dense_attention_mask = dflash_backend._create_dflash_dense_attention_mask
DFlashTrainingModel = dflash_backend.DFlashTrainingModel
DFlashConfig = dflash_models.DFlashConfig
DFlashDraftModel = dflash_models.DFlashDraftModel


def test_dspark_fallback_config_uses_native_qwen_mrv2_architecture() -> None:
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )

    assert config.architectures == ["Qwen3DSparkModel"]


def test_dspark_checkpoint_preserves_source_config_and_vllm_weight_names(
    tmp_path,
) -> None:
    initial_config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_config = initial_config.to_dict()
    source_config.update(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3DSparkModel"],
            "num_anchors": 17,
            "source_only_field": {"preserved": True},
        }
    )
    source_config.pop("enable_confidence_head", None)
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    config = DSparkConfig.from_dspark_pretrained(str(source_dir))
    config.num_anchors = 99

    model = DSparkDraftModel(config)
    state_keys = set(model.state_dict())
    assert {"fc.weight", "hidden_norm.weight", "norm.weight"}.issubset(state_keys)
    assert not {
        "context_proj.weight",
        "context_norm.weight",
        "final_norm.weight",
    }.intersection(state_keys)

    assert config.model_type == "dspark"
    model.save_pretrained(output_dir, safe_serialization=False)
    saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    saved_state = torch.load(
        output_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    for key, value in source_config.items():
        assert saved_config[key] == value
    assert saved_config["enable_confidence_head"] is False
    assert {"fc.weight", "hidden_norm.weight", "norm.weight"}.issubset(saved_state)

    reloaded = DSparkConfig.from_dspark_pretrained(str(output_dir))
    assert reloaded.model_type == "dspark"
    assert reloaded.to_dict()["model_type"] == source_config["model_type"]
    assert reloaded.to_dict()["architectures"] == source_config["architectures"]


def _small_dspark_training_model(
    block_size: int = 4,
    l1_loss_alpha: float = 0.0,
    lk_loss_alpha: float = 0.0,
    l1_chunk_size: int = 0,
    loss_mode: str = "full_vocab",
    distribution_loss_impl: str = "auto",
    lk_temperature: float = 1.0,
    lk_loss_type: str = "alpha",
    lk_hybrid_eta: float = 3.0,
    loss_decay_gamma: float = 7.0,
    per_position_loss_weight: str = "fixed_exp_decay",
    dpace_alpha: float = 0.5,
):
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=block_size,
        num_anchors=2,
        markov_rank=4,
        markov_head_type="vanilla",
    )
    draft_model = DSparkDraftModel(config)
    return DSparkTrainingModel(
        draft_model=draft_model,
        block_size=block_size,
        num_anchors=2,
        loss_decay_gamma=loss_decay_gamma,
        loss_mode=loss_mode,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        l1_loss_alpha=l1_loss_alpha,
        lk_loss_alpha=lk_loss_alpha,
        lk_temperature=lk_temperature,
        lk_loss_type=lk_loss_type,
        lk_hybrid_eta=lk_hybrid_eta,
        l1_chunk_size=l1_chunk_size,
        distribution_loss_impl=distribution_loss_impl,
    )


def test_dspark_default_loss_weights_match_deepspec():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(config))

    assert config.ce_loss_alpha == pytest.approx(0.1)
    assert config.l1_loss_alpha == pytest.approx(0.9)
    assert model.ce_loss_alpha == pytest.approx(0.1)
    assert model.l1_loss_alpha == pytest.approx(0.9)
    assert model.lk_loss_alpha == 0.0
    assert model.l1_chunk_size == 0
    assert model.distribution_loss_impl == "auto"


def test_dspark_rejects_unknown_distribution_loss_implementation():
    with pytest.raises(ValueError, match="auto, fused, eager"):
        _small_dspark_training_model(distribution_loss_impl="unknown")


def test_dspark_forced_fused_loss_fails_closed_on_cpu():
    model = _small_dspark_training_model(distribution_loss_impl="fused")

    with pytest.raises(RuntimeError, match="fused distribution loss"):
        model._use_fused_distribution_loss(torch.device("cpu"))


def test_dspark_fused_l1_honors_chunk_size(monkeypatch):
    model = _small_dspark_training_model(l1_loss_alpha=1.0, l1_chunk_size=2)
    rows = 5
    vocab = model.draft_model.config.vocab_size
    hidden_size = model.draft_model.config.hidden_size
    active_hidden = torch.randn(rows, hidden_size)
    active_target_hidden = torch.randn(rows, hidden_size)
    active_prev_tokens = torch.arange(rows, dtype=torch.long)
    active_weights = torch.linspace(0.5, 1.0, rows)
    lm_head_weight = torch.randn(vocab, hidden_size)
    active_draft_logits = torch.randn(rows, vocab, requires_grad=True)
    chunk_rows = []

    def eager_total_variation(draft_logits, target_logits):
        chunk_rows.append(int(draft_logits.size(0)))
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        target_probs = torch.softmax(target_logits.float(), dim=-1)
        return 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)

    monkeypatch.setattr(
        dspark_backend, "fused_total_variation", eager_total_variation
    )
    l1_sum, l1_den = model._compute_fused_l1_loss_for_active(
        active_hidden=active_hidden,
        active_prev_tokens=active_prev_tokens,
        active_target_hidden=active_target_hidden,
        active_weights=active_weights,
        lm_head_weight=lm_head_weight,
        active_draft_logits=active_draft_logits,
    )

    assert chunk_rows == [2, 2, 1]
    assert l1_sum.ndim == 0
    assert l1_den == pytest.approx(float(active_weights.sum()))
    with torch.no_grad():
        target_logits = torch.nn.functional.linear(
            active_target_hidden, lm_head_weight
        )
        expected_l1 = (
            (
                torch.softmax(active_draft_logits.float(), dim=-1)
                - torch.softmax(target_logits.float(), dim=-1)
            )
            .abs()
            .sum(dim=-1)
            .mul(active_weights)
            .sum()
        )
    torch.testing.assert_close(l1_sum.detach(), expected_l1)
    l1_sum.backward()
    assert active_draft_logits.grad is not None

    chunk_rows.clear()
    recomputed_sum, recomputed_den = model._compute_fused_l1_loss_for_active(
        active_hidden=active_hidden,
        active_prev_tokens=active_prev_tokens,
        active_target_hidden=active_target_hidden,
        active_weights=active_weights,
        lm_head_weight=lm_head_weight,
        active_draft_logits=None,
    )
    assert chunk_rows == [2, 2, 1]
    assert recomputed_sum.ndim == 0
    assert recomputed_den == pytest.approx(float(active_weights.sum()))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lk_loss_type": "unknown"}, "lk_loss_type"),
        ({"lk_hybrid_eta": -1.0}, "lk_hybrid_eta"),
    ],
)
def test_block_drafter_rejects_invalid_lk_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_dspark_training_model(**kwargs)


def test_dspark_untrained_confidence_head_is_kept_but_excluded_from_optimizer():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        markov_rank=4,
        enable_confidence_head=True,
    )
    model = DSparkTrainingModel(
        draft_model=DSparkDraftModel(config),
        confidence_head_alpha=0.0,
    )

    confidence_head = model.draft_model.confidence_head
    assert confidence_head is not None
    assert "draft_model.confidence_head.proj.weight" in model.state_dict()
    assert all(
        not parameter.requires_grad for parameter in confidence_head.parameters()
    )
    optimizer_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert optimizer_parameter_ids.isdisjoint(
        id(parameter) for parameter in confidence_head.parameters()
    )


def test_dspark_label_and_prev_token_alignment():
    model = _small_dspark_training_model(block_size=4)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    anchor_positions = torch.tensor([[2]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])

    target_ids, prev_token_ids, eval_mask, label_indices = model._build_label_tensors(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )

    assert label_indices.tolist() == [[[3, 4, 5, 6]]]
    assert target_ids.tolist() == [[[13, 14, 15, 16]]]
    assert prev_token_ids.tolist() == [[[12, 13, 14, 15]]]
    assert eval_mask.tolist() == [[[True, True, True, True]]]


def test_dspark_target_hidden_gather_matches_reference_for_multiple_batches():
    model = _small_dspark_training_model(block_size=3)
    target_hidden = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    label_indices = torch.tensor(
        [
            [[1, 3, 6], [2, 4, 5]],
            [[6, 5, 4], [1, 2, 3]],
        ],
        dtype=torch.long,
    )
    block_keep_mask = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool
    )

    actual = model._gather_aligned_target_hidden(
        target_last_hidden_states=target_hidden,
        label_indices=label_indices,
        block_keep_mask=block_keep_mask,
    )

    target_pred_indices = (label_indices - 1).clamp(min=0, max=5)
    target_pred_indices = torch.where(
        block_keep_mask.unsqueeze(-1),
        target_pred_indices,
        torch.zeros_like(target_pred_indices),
    )
    expected = torch.gather(
        target_hidden.unsqueeze(1).expand(-1, 2, -1, -1),
        2,
        target_pred_indices.unsqueeze(-1).expand(-1, -1, -1, 4),
    )

    assert actual.shape == (2, 2, 3, 4)
    assert torch.equal(actual, expected)
    # Masked blocks still select token zero from their own batch.  This catches
    # a missing or incorrect batch offset in the flattened implementation.
    assert torch.equal(actual[1, 0, 0], target_hidden[1, 0])


def test_dspark_target_hidden_gather_rejects_batch_size_mismatch():
    model = _small_dspark_training_model(block_size=3)

    with pytest.raises(ValueError, match="batch size must match"):
        model._gather_aligned_target_hidden(
            target_last_hidden_states=torch.zeros(2, 6, 4),
            label_indices=torch.ones(1, 2, 3, dtype=torch.long),
            block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_dspark_dense_attention_mask_matches_deepspec_block_contract():
    anchor_positions = torch.tensor([[2, 4]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, True]])

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=2,
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 4, 10)
    assert torch.nonzero(mask[0, 0, 0], as_tuple=False).flatten().tolist() == [
        0,
        1,
        6,
        7,
    ]
    assert torch.nonzero(mask[0, 0, 1], as_tuple=False).flatten().tolist() == [
        0,
        1,
        6,
        7,
    ]
    assert torch.nonzero(mask[0, 0, 2], as_tuple=False).flatten().tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
    ]
    assert torch.nonzero(mask[0, 0, 3], as_tuple=False).flatten().tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
    ]


def test_dspark_dense_attention_mask_keeps_dummy_rows_finite_safe():
    anchor_positions = torch.tensor([[2, 0]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, False]])

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=2,
    )

    assert torch.nonzero(mask[0, 0, 2], as_tuple=False).flatten().tolist() == [8]
    assert torch.nonzero(mask[0, 0, 3], as_tuple=False).flatten().tolist() == [9]


def test_dspark_first_position_is_masked_when_first_target_invalid():
    model = _small_dspark_training_model(block_size=4)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    loss_mask[0, 3] = 0.0
    anchor_positions = torch.tensor([[2]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])

    _, _, eval_mask, _ = model._build_label_tensors(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )

    assert eval_mask.tolist() == [[[False, False, False, False]]]


def test_dspark_markov_rank_zero_keeps_base_logits():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=2,
        markov_rank=0,
    )
    draft_model = DSparkDraftModel(config)
    base_logits = torch.randn(1, 2, 4, 32)
    prev_ids = torch.zeros(1, 2, 4, dtype=torch.long)
    hidden = torch.randn(1, 2, 4, 8)

    corrected = draft_model.apply_markov_logits(
        base_logits,
        prev_token_ids=prev_ids,
        draft_hidden=hidden,
    )

    assert torch.equal(corrected, base_logits)


def test_dspark_markov_bias_changes_logits():
    model = _small_dspark_training_model(block_size=4).draft_model
    assert model.markov_head is not None
    base_logits = torch.zeros(1, 1, 4, 32)
    prev_ids = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long)
    hidden = torch.zeros(1, 1, 4, 8)

    with torch.no_grad():
        model.markov_head.markov_w1.weight.fill_(0.5)
        model.markov_head.markov_w2.weight.fill_(0.25)

    corrected = model.apply_markov_logits(
        base_logits,
        prev_token_ids=prev_ids,
        draft_hidden=hidden,
    )

    assert corrected.abs().sum().item() > 0
    assert not torch.equal(corrected, base_logits)


def test_dspark_l1_loss_uses_target_last_hidden_states():
    model = _small_dspark_training_model(block_size=2, l1_loss_alpha=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )

    assert torch.isfinite(loss)
    assert diagnostics["ce_weighted_token_count"].item() > 0
    assert diagnostics["ce_loss_sum"].item() >= 0
    assert diagnostics["l1_weighted_token_count"].item() > 0
    assert diagnostics["l1_loss_sum"].item() >= 0


@pytest.mark.parametrize(
    ("loss_mode", "expects_reused_log_probs"),
    [("full_vocab", True), ("restricted_ce", False)],
)
def test_dspark_l1_reuses_only_full_vocab_ce_log_probs(
    monkeypatch, loss_mode, expects_reused_log_probs
):
    model = _small_dspark_training_model(
        block_size=2,
        l1_loss_alpha=0.5,
        loss_mode=loss_mode,
    )
    captured_log_probs = []
    original_compute_distribution_losses = model._compute_distribution_losses_for_active

    def capture_compute_distribution_losses(**kwargs):
        captured_log_probs.append(kwargs.get("active_draft_log_probs"))
        return original_compute_distribution_losses(**kwargs)

    monkeypatch.setattr(
        model,
        "_compute_distribution_losses_for_active",
        capture_compute_distribution_losses,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_ = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert len(captured_log_probs) == 1
    assert (captured_log_probs[0] is not None) is expects_reused_log_probs
    if captured_log_probs[0] is not None:
        probability_mass = captured_log_probs[0].exp().sum(dim=-1)
        assert torch.allclose(
            probability_mass, torch.ones_like(probability_mass), atol=1e-5, rtol=1e-5
        )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_from_dspark_dict_normalizes_transformer_layer_config() -> None:
    config = DSparkConfig.from_dspark_dict(
        {
            "architectures": ["Qwen3DSparkModel"],
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "head_dim": 16,
            },
            "block_size": 8,
            "num_anchors": 512,
            "markov_rank": 256,
        }
    )
    assert config.hidden_size == 64
    assert config.intermediate_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.vocab_size == 128
    assert config.block_size == 8


def test_dspark_fallback_prefers_dspark_intermediate_size() -> None:
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    backend = dspark_backend.DSparkTrainerBackend.__new__(
        dspark_backend.DSparkTrainerBackend
    )
    backend.config = OmegaConf.create(
        {
            "actor": {"fsdp_config": {}},
            "rollout": {
                "drafter": {"training": {"dspark_intermediate_size": 6144}}
            },
        }
    )
    target = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        head_dim=None,
        rope_theta=10000.0,
    )

    selected = backend._build_fallback_config(target)
    assert selected.intermediate_size == 6144

    backend.config.rollout.drafter.training.dspark_intermediate_size = None
    defaulted = backend._build_fallback_config(target)
    # MoE targets have no dense intermediate_size, so hidden_size * 4 is used.
    assert defaulted.intermediate_size == 2048 * 4


def test_from_dspark_dict_lifts_released_aux_layer_ids_into_serving_config(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from verl_speco.trainer.draft_training_loop import (
        _rewrite_standalone_block_runtime_config,
    )

    # Mirrors the released RedHatAI/Qwen3.6-35B-A3B-speculator.dspark config
    # (40-layer target): the architecture is nested and the context layers live
    # under ``aux_hidden_state_layer_ids``.
    released_config = {
        "architectures": ["Qwen3DSparkModel"],
        "speculators_model_type": "dspark",
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 5,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 151936,
            "head_dim": 128,
        },
        "aux_hidden_state_layer_ids": [2, 10, 20, 30, 37],
        "block_size": 8,
        "num_anchors": 512,
        "markov_rank": 256,
    }

    config = DSparkConfig.from_dspark_dict(released_config)
    # ``aux_hidden_state_layer_ids`` uses the same EAGLE ``output_hidden_states``
    # indexing as SpeCo's training-side ``target_layer_ids``, so it must be kept
    # verbatim rather than shifted or replaced by the spaced fallback.
    assert config.target_layer_ids == [2, 10, 20, 30, 37]

    checkpoint_dir = tmp_path / "draft_step_10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3DSparkModel"]}),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(source_dir))
            )
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    # vLLM reads ``eagle_aux_hidden_state_layer_ids`` directly; the z-lab
    # ``target_layer_ids`` aliases are one less and vLLM adds the +1 back.
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 20, 30, 37]
    assert runtime_config["target_layer_ids"] == [1, 9, 19, 29, 36]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [1, 9, 19, 29, 36]


def test_dspark_lk_loss_matches_negative_log_probability_overlap():
    model = _small_dspark_training_model(block_size=2, lk_loss_alpha=1.0)
    draft_probs = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float32)
    target_probs = torch.tensor([[0.45, 0.4, 0.15]], dtype=torch.float32)
    lm_head_weight = torch.eye(3, dtype=torch.float32)

    _, lk_sum, loss_den, _ = model._compute_distribution_losses_for_active(
        active_hidden=torch.zeros(1, 3),
        active_prev_tokens=torch.zeros(1, dtype=torch.long),
        active_target_hidden=target_probs.log(),
        active_weights=torch.ones(1),
        lm_head_weight=lm_head_weight,
        active_draft_log_probs=draft_probs.log(),
    )

    expected_overlap = torch.minimum(draft_probs, target_probs).sum(dim=-1)
    assert torch.allclose(lk_sum, -expected_overlap.log().sum())
    assert loss_den.item() == pytest.approx(1.0)


def test_dspark_lk_temperature_scales_both_distributions():
    temperature = 0.6
    model = _small_dspark_training_model(
        block_size=2, lk_loss_alpha=1.0, lk_temperature=temperature
    )
    draft_logits = torch.tensor([[1.2, -0.3, 0.4]], dtype=torch.float32)
    target_logits = torch.tensor([[0.7, 0.5, -0.2]], dtype=torch.float32)

    _, lk_sum, _, _ = model._compute_distribution_losses_for_active(
        active_hidden=torch.zeros(1, 3),
        active_prev_tokens=torch.zeros(1, dtype=torch.long),
        active_target_hidden=target_logits,
        active_weights=torch.ones(1),
        lm_head_weight=torch.eye(3),
        active_draft_log_probs=torch.log_softmax(draft_logits, dim=-1),
    )

    expected_overlap = torch.minimum(
        torch.softmax(draft_logits / temperature, dim=-1),
        torch.softmax(target_logits / temperature, dim=-1),
    ).sum(dim=-1)
    torch.testing.assert_close(lk_sum, -expected_overlap.log().sum())


def test_dspark_adaptive_hybrid_uses_global_acceptance_across_chunks():
    eta = 3.0
    model = _small_dspark_training_model(
        block_size=2,
        lk_loss_alpha=1.0,
        lk_loss_type="adaptive_hybrid",
        lk_hybrid_eta=eta,
        l1_chunk_size=1,
    )
    draft_probs = torch.tensor(
        [[0.6, 0.1, 0.3], [0.2, 0.5, 0.3]], dtype=torch.float32
    )
    target_probs = torch.tensor(
        [[0.45, 0.4, 0.15], [0.1, 0.2, 0.7]], dtype=torch.float32
    )
    weights = torch.tensor([1.0, 2.0])

    _, lk_sum, loss_den, diagnostics = (
        model._compute_distribution_losses_for_active(
            active_hidden=torch.zeros(2, 3),
            active_prev_tokens=torch.zeros(2, dtype=torch.long),
            active_target_hidden=target_probs.log(),
            active_weights=weights,
            lm_head_weight=torch.eye(3),
            active_draft_log_probs=draft_probs.log(),
        )
    )

    acceptance = torch.minimum(draft_probs, target_probs).sum(dim=-1)
    forward_kl = (target_probs * (target_probs.log() - draft_probs.log())).sum(
        dim=-1
    )
    tv = 1.0 - acceptance
    kl_weight = torch.exp(-eta * acceptance.mean())
    expected_loss = (
        kl_weight * (forward_kl * weights).sum()
        + (1.0 - kl_weight) * (tv * weights).sum()
    )
    torch.testing.assert_close(lk_sum, expected_loss)
    assert loss_den.item() == pytest.approx(weights.sum().item())
    torch.testing.assert_close(diagnostics["acceptance_sum"], acceptance.sum())
    torch.testing.assert_close(
        diagnostics["kl_weight_sum"], kl_weight * draft_probs.size(0)
    )
    torch.testing.assert_close(diagnostics["forward_kl_sum"], forward_kl.sum())
    torch.testing.assert_close(diagnostics["tv_sum"], tv.sum())


def test_dspark_pure_lk_loss_is_finite_and_backpropagates():
    model = _small_dspark_training_model(
        block_size=2,
        l1_loss_alpha=0.0,
        lk_loss_alpha=1.0,
    )
    model.ce_loss_alpha = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["lk_weighted_token_count"].item() > 0
    assert diagnostics["lk_loss_sum"].item() >= 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_dflash_pure_lk_loss_is_finite_and_backpropagates():
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    model = DFlashTrainingModel(
        draft_model=DFlashDraftModel(config),
        block_size=2,
        num_anchors=2,
        ce_loss_alpha=0.0,
        lk_loss_alpha=1.0,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["lk_weighted_token_count"].item() > 0
    assert diagnostics["lk_loss_sum"].item() >= 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("model_type", ["dflash", "dspark"])
def test_block_drafter_adaptive_hybrid_records_lk_diagnostics(model_type):
    if model_type == "dspark":
        model = _small_dspark_training_model(
            block_size=2,
            l1_loss_alpha=0.0,
            lk_loss_alpha=1.0,
            lk_loss_type="adaptive_hybrid",
        )
        model.ce_loss_alpha = 0.0
    else:
        config = DFlashConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            num_target_layers=4,
            num_context_layers=2,
            target_hidden_size=8,
            target_num_hidden_layers=4,
            target_layer_ids=[1, 3],
            mask_token_id=31,
        )
        model = DFlashTrainingModel(
            draft_model=DFlashDraftModel(config),
            block_size=2,
            num_anchors=2,
            ce_loss_alpha=0.0,
            lk_loss_alpha=1.0,
            lk_loss_type="adaptive_hybrid",
        )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )
    loss.backward()

    count = diagnostics["lk_diagnostic_token_count"]
    assert count.item() > 0
    assert 0.0 <= (diagnostics["lk_acceptance_sum"] / count).item() <= 1.0
    assert 0.0 < (diagnostics["lk_kl_weight_sum"] / count).item() <= 1.0
    assert (diagnostics["lk_forward_kl_sum"] / count).item() >= 0.0
    assert 0.0 <= (diagnostics["lk_tv_sum"] / count).item() <= 1.0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_dflash_lk_batch_includes_target_last_hidden_states():
    from omegaconf import OmegaConf

    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    seq_len, hidden_size = 5, 8
    input_ids = torch.arange(seq_len, dtype=torch.long)
    hidden_states = torch.randn(seq_len, hidden_size * 2)
    target_last_hidden_states = torch.randn(seq_len, hidden_size)
    loss_mask = torch.ones(seq_len, dtype=torch.float32)

    class _FakeDFlashBackend:
        model_type = "dflash"

        def preprocess_individual_items(self, items, device, model_config):
            return {
                "ids": [input_ids],
                "h_states": [hidden_states],
                "masks": [loss_mask],
                "target_last_h_states": [target_last_hidden_states],
            }

    trainer = object.__new__(DrafterBaseTrainer)
    trainer.backend = _FakeDFlashBackend()
    trainer.batch_size = 1
    trainer.current_rl_step = 0
    trainer.training_steps = 0
    trainer.use_data_buffer = False
    trainer.collected_data = [{"step": 0, "hidden_states": hidden_states}]
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"dflash_lk_loss_alpha": 1.0}}}}
    )
    trainer.use_ulysses_sp = False
    trainer.rank = 0
    trainer.model_config = None
    trainer.model = torch.nn.Linear(2, 2)

    batch = trainer._prepare_training_batch()

    assert batch is not None
    assert torch.equal(batch["target_last_hidden_states"][0], target_last_hidden_states)


def test_dflash_dpace_position_weights_are_used():
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    model = DFlashTrainingModel(
        draft_model=DFlashDraftModel(config),
        block_size=4,
        num_anchors=1,
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
    )
    active_mask = torch.tensor([False, True, True, True])
    base_loss_mask = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])
    per_token_ce = torch.tensor([0.2, 0.5, 0.7])

    weights = model._dpace_weight_mask(
        per_token_ce=per_token_ce,
        finite_loss=torch.ones(3, dtype=torch.bool),
        active_mask=active_mask,
        base_loss_mask=base_loss_mask,
    )

    smooth = 0.5 * torch.exp(-per_token_ce) + 0.5
    prefix = torch.cumprod(smooth, dim=-1)
    expected = torch.tensor([[[0.0, prefix.sum(), prefix[1:].sum(), prefix[2]]]])
    torch.testing.assert_close(weights, expected)



def test_dspark_dpace_weights_only_ce_and_keeps_fixed_distribution_weights(
    monkeypatch,
):
    model = _small_dspark_training_model(
        block_size=2,
        l1_loss_alpha=1.0,
        loss_decay_gamma=2.0,
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
    )
    captured_distribution_weights = []
    original_compute_distribution_losses = model._compute_distribution_losses_for_active

    def fixed_dpace_weights(*, base_loss_mask, **_kwargs):
        return base_loss_mask * 9.0

    def capture_distribution_weights(**kwargs):
        captured_distribution_weights.append(kwargs["active_weights"].detach().clone())
        return original_compute_distribution_losses(**kwargs)

    monkeypatch.setattr(model, "_dpace_weight_mask", fixed_dpace_weights)
    monkeypatch.setattr(
        model,
        "_compute_distribution_losses_for_active",
        capture_distribution_weights,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    _loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )

    assert diagnostics["ce_weighted_token_count"].item() == pytest.approx(36.0)
    assert len(captured_distribution_weights) == 1
    expected_fixed_weights = torch.exp(torch.tensor([0.0, -0.5, 0.0, -0.5]))
    torch.testing.assert_close(
        captured_distribution_weights[0], expected_fixed_weights
    )
