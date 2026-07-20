from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

dspark_backend = pytest.importorskip("verl_speco.backends.dspark_trainer_backend")
dspark_models = pytest.importorskip("verl_speco.models.dspark")
dflash_backend = pytest.importorskip("verl_speco.backends.dflash_trainer_backend")

DSparkTrainingModel = dspark_backend.DSparkTrainingModel
DSparkConfig = dspark_models.DSparkConfig
DSparkDraftModel = dspark_models.DSparkDraftModel
create_dense_attention_mask = dflash_backend._create_dflash_dense_attention_mask


def test_dspark_checkpoint_preserves_source_config_and_vllm_weight_names(tmp_path) -> None:
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
    saved_state = torch.load(output_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
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
    l1_chunk_size: int = 0,
    loss_mode: str = "full_vocab",
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
        loss_mode=loss_mode,
        l1_loss_alpha=l1_loss_alpha,
        l1_chunk_size=l1_chunk_size,
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
    assert model.l1_chunk_size == 0


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
    assert all(not parameter.requires_grad for parameter in confidence_head.parameters())
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
    assert torch.nonzero(mask[0, 0, 0], as_tuple=False).flatten().tolist() == [0, 1, 6, 7]
    assert torch.nonzero(mask[0, 0, 1], as_tuple=False).flatten().tolist() == [0, 1, 6, 7]
    assert torch.nonzero(mask[0, 0, 2], as_tuple=False).flatten().tolist() == [0, 1, 2, 3, 8, 9]
    assert torch.nonzero(mask[0, 0, 3], as_tuple=False).flatten().tolist() == [0, 1, 2, 3, 8, 9]


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
def test_dspark_l1_reuses_only_full_vocab_ce_log_probs(monkeypatch, loss_mode, expects_reused_log_probs):
    model = _small_dspark_training_model(
        block_size=2,
        l1_loss_alpha=0.5,
        loss_mode=loss_mode,
    )
    captured_log_probs = []
    original_compute_l1 = model._compute_l1_loss_for_active

    def capture_compute_l1(**kwargs):
        captured_log_probs.append(kwargs.get("active_draft_log_probs"))
        return original_compute_l1(**kwargs)

    monkeypatch.setattr(model, "_compute_l1_loss_for_active", capture_compute_l1)
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
        assert torch.allclose(probability_mass, torch.ones_like(probability_mass), atol=1e-5, rtol=1e-5)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
