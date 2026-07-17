from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

dspark_backend = pytest.importorskip("verl_speco.backends.dspark_trainer_backend")
dspark_models = pytest.importorskip("verl_speco.models.dspark")

DSparkTrainingModel = dspark_backend.DSparkTrainingModel
DSparkConfig = dspark_models.DSparkConfig
DSparkDraftModel = dspark_models.DSparkDraftModel


def _small_dspark_training_model(block_size: int = 4):
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
    return DSparkTrainingModel(draft_model=draft_model, block_size=block_size, num_anchors=2)


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
    print('dspark label and prev token alignment test')
    assert label_indices.tolist() == [[[3, 4, 5, 6]]]
    assert target_ids.tolist() == [[[13, 14, 15, 16]]]
    assert prev_token_ids.tolist() == [[[12, 13, 14, 15]]]
    assert eval_mask.tolist() == [[[True, True, True, True]]]


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
