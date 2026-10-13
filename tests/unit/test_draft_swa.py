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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainingModel,
    _create_dflash_dense_attention_mask,
    _create_dflash_mask_mod,
    _resolve_sliding_windows,
    _sliding_window_config,
    build_dflash_attention_masks,
    _document_boundary_validity,
)
from verl_speco.backends.dspark_trainer_backend import (
    DSparkTrainerBackend,
    DSparkTrainingModel,
)
from verl_speco.models.dflash import DFlashConfig, DFlashDraftModel
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.trainer.base_trainer import DrafterBaseTrainer


def _mask_inputs():
    anchor_positions = torch.tensor([[5, 8]])
    block_keep_mask = torch.tensor([[True, True]])
    return anchor_positions, block_keep_mask


def test_dense_mask_caps_context_to_sliding_window() -> None:
    anchor_positions, block_keep_mask = _mask_inputs()
    windowed = _create_dflash_dense_attention_mask(
        anchor_positions, block_keep_mask, ctx_len=8, block_size=2, sliding_window=3
    )
    full = _create_dflash_dense_attention_mask(
        anchor_positions, block_keep_mask, ctx_len=8, block_size=2, sliding_window=None
    )

    # Block 0 anchor is 5: with window 3 only context keys 2..4 stay visible.
    assert bool(full[0, 0, 0, 0]) is True
    assert bool(windowed[0, 0, 0, 0]) is False
    assert bool(windowed[0, 0, 0, 2])
    assert bool(windowed[0, 0, 0, 4])
    assert not bool(windowed[0, 0, 0, 5])
    # Draft keys stay block-local.
    assert bool(windowed[0, 0, 0, 8])
    assert not bool(windowed[0, 0, 0, 10])


def test_dense_mask_isolates_packed_documents() -> None:
    # Two length-4 documents packed into one row; anchor 4 starts document 1.
    document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    anchor_positions = torch.tensor([[0, 4]])
    block_keep_mask = torch.tensor([[True, True]])
    kwargs = {
        "anchor_positions": anchor_positions,
        "block_keep_mask": block_keep_mask,
        "ctx_len": 8,
        "block_size": 2,
    }

    isolated = _create_dflash_dense_attention_mask(**kwargs, document_ids=document_ids)
    unisolated = _create_dflash_dense_attention_mask(**kwargs, document_ids=None)

    # Query block 1 (doc 1) may see doc-0 key 0 only without isolation.
    assert bool(unisolated[0, 0, 2, 0]) is True
    assert bool(isolated[0, 0, 2, 0]) is False


def test_dense_mask_excludes_negative_document_ids() -> None:
    # A -1 marks an excluded position even when it precedes the anchor.
    document_ids = torch.tensor([[0, 0, -1, 0, 1, 1, 1, 1]])
    anchor_positions = torch.tensor([[3, 5]])
    block_keep_mask = torch.tensor([[True, True]])
    mask = _create_dflash_dense_attention_mask(
        anchor_positions,
        block_keep_mask,
        ctx_len=8,
        block_size=2,
        document_ids=document_ids,
    )

    # Query block 0 (anchor 3) sees doc-0 keys 0 and 1 but not the -1 hole at 2.
    assert bool(mask[0, 0, 0, 0]) is True
    assert bool(mask[0, 0, 0, 1]) is True
    assert bool(mask[0, 0, 0, 2]) is False


def test_flex_mask_mod_isolates_documents_at_draft_keys() -> None:
    # Regression: the document lookup must not index past ctx_len for draft keys.
    document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    anchor_positions = torch.tensor([[0, 6]])
    block_keep_mask = torch.tensor([[True, True]])
    mask_mod = _create_dflash_mask_mod(
        anchor_positions,
        block_keep_mask,
        ctx_len=8,
        block_size=2,
        document_ids=document_ids,
    )

    # Query block 1 (anchor 6) cannot see doc-0 context but sees doc-1 context.
    assert bool(mask_mod(0, 0, 2, 0)) is False
    assert bool(mask_mod(0, 0, 2, 5)) is True
    # Draft keys (kv_idx >= ctx_len) must not raise and stay block-local.
    assert bool(mask_mod(0, 0, 2, 10)) is True
    assert bool(mask_mod(0, 0, 2, 8)) is False


def test_document_boundary_validity_rejects_cross_document_blocks() -> None:
    # doc 0 = positions 0..3, position 4 is padding, doc 1 = positions 5..7.
    document_ids = torch.tensor([[0, 0, 0, 0, -1, 1, 1, 1]])
    seq_len, block_size = 8, 2
    indices = torch.arange(seq_len - block_size + 1).unsqueeze(0).expand(1, -1)
    valid = torch.ones_like(indices, dtype=torch.bool)

    # DFlash same-position labels end at anchor + block_size - 1.
    kept = _document_boundary_validity(
        valid,
        indices=indices,
        document_ids=document_ids,
        block_size=block_size,
        seq_len=seq_len,
    )

    assert bool(kept[0, 0]) is True  # labels 0..1 stay in doc 0
    assert bool(kept[0, 2]) is True  # labels 2..3 stay in doc 0
    assert bool(kept[0, 3]) is False  # labels 3..4 reach the -1 padding
    assert bool(kept[0, 4]) is False  # anchor is padding
    assert bool(kept[0, 5]) is True  # labels 5..6 stay in doc 1
    assert bool(kept[0, 6]) is True  # labels 6..7 stay in doc 1

    # DSpark/DOMINO next-token shifted labels end at anchor + block_size.
    shifted = _document_boundary_validity(
        valid,
        indices=indices,
        document_ids=document_ids,
        block_size=block_size,
        seq_len=seq_len,
        label_shift=1,
    )

    assert bool(shifted[0, 0]) is True  # labels 1..2 stay in doc 0
    assert bool(shifted[0, 2]) is False  # labels 3..4 reach the -1 padding
    assert bool(shifted[0, 3]) is False  # labels 4..5 cross into doc 1
    assert bool(shifted[0, 4]) is False  # anchor is padding
    assert bool(shifted[0, 5]) is True  # labels 6..7 stay in doc 1


def test_flex_mask_mod_honors_sliding_window() -> None:
    anchor_positions, block_keep_mask = _mask_inputs()
    windowed = _create_dflash_mask_mod(
        anchor_positions, block_keep_mask, ctx_len=8, block_size=2, sliding_window=3
    )
    full = _create_dflash_mask_mod(
        anchor_positions, block_keep_mask, ctx_len=8, block_size=2, sliding_window=None
    )

    assert bool(full(0, 0, 0, 0)) is True
    assert bool(windowed(0, 0, 0, 0)) is False
    assert bool(windowed(0, 0, 0, 2)) is True
    assert bool(windowed(0, 0, 0, 8)) is True


def test_resolve_sliding_windows_uses_layer_types() -> None:
    all_sliding = SimpleNamespace(
        num_hidden_layers=3,
        use_sliding_window=True,
        sliding_window=2048,
        layer_types=None,
    )
    assert _resolve_sliding_windows(all_sliding) == [2048, 2048, 2048]

    mixed = SimpleNamespace(
        num_hidden_layers=5,
        use_sliding_window=True,
        sliding_window=2048,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    assert _resolve_sliding_windows(mixed) == [2048, 2048, None, 2048, None]

    disabled = SimpleNamespace(
        num_hidden_layers=2,
        use_sliding_window=False,
        sliding_window=2048,
        layer_types=None,
    )
    assert _resolve_sliding_windows(disabled) == [None, None]


def test_build_masks_returns_per_layer_lists_for_mixed_windows() -> None:
    anchor_positions, block_keep_mask = _mask_inputs()
    kwargs = {
        "anchor_positions": anchor_positions,
        "block_keep_mask": block_keep_mask,
        "ctx_len": 8,
        "block_size": 2,
        "device": torch.device("cpu"),
    }

    _, dense = build_dflash_attention_masks(windows=[3, 3, 3], **kwargs)
    assert not isinstance(dense, list)

    _, dense_layers = build_dflash_attention_masks(windows=[3, None, 3], **kwargs)
    assert isinstance(dense_layers, list) and len(dense_layers) == 3
    # The full-attention layer sees the anchor's whole prefix, the windowed one does not.
    assert bool(dense_layers[1][0, 0, 0, 0]) is True
    assert bool(dense_layers[0][0, 0, 0, 0]) is False


def test_sliding_window_config_defaults_to_all_sliding() -> None:
    assert _sliding_window_config(None, 5) == {}
    kwargs = _sliding_window_config(2048, 5)
    assert kwargs["sliding_window"] == 2048
    assert kwargs["use_sliding_window"] is True
    assert kwargs["layer_types"] == ["sliding_attention"] * 5


def test_dflash_config_validates_layer_types_length() -> None:
    with pytest.raises(ValueError, match="layer_types"):
        DFlashConfig(num_hidden_layers=5, layer_types=["full_attention"] * 3)


def test_dspark_training_forward_with_mixed_sliding_layers() -> None:
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        markov_rank=4,
        markov_head_type="vanilla",
        sliding_window=3,
        use_sliding_window=True,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = DSparkTrainingModel(
        draft_model=DSparkDraftModel(config),
        block_size=2,
        num_anchors=2,
        l1_loss_alpha=0.0,
    )

    result = model(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        hidden_states_list=[torch.randn(1, 5, 8), torch.randn(1, 5, 8)],
        loss_mask=torch.ones(1, 5),
        lm_head_weight=torch.randn(32, 8),
    )

    assert torch.isfinite(result[0])


def test_dflash_training_forward_accepts_document_ids() -> None:
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        num_target_layers=4,
        num_context_layers=1,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
        mask_token_id=15,
    )
    model = DFlashTrainingModel(
        draft_model=DFlashDraftModel(config),
        block_size=2,
        num_anchors=2,
    )

    # doc 0 = positions 0..2, positions 3..4 are packing padding.
    document_ids = torch.tensor([[0, 0, 0, -1, -1]])
    result = model(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        hidden_states_list=[torch.randn(1, 5, 8)],
        loss_mask=torch.ones(1, 5),
        lm_head_weight=torch.randn(16, 8),
        document_ids=document_ids,
    )

    assert torch.isfinite(result[0])


def test_dflash_trailing_label_token_is_supervised() -> None:
    """DFlash must supervise the final label token that has no context row."""
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        num_target_layers=4,
        num_context_layers=1,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
        mask_token_id=15,
    )
    model = DFlashTrainingModel(
        draft_model=DFlashDraftModel(config),
        block_size=2,
        num_anchors=2,
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def _capture(**kwargs):
        captured["targets"] = kwargs["active_targets"].detach().clone()
        return None, {}

    model._auxiliary_loss = _capture  # type: ignore[assignment]

    context_rows, label_rows = 2, 3
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)
    label_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    model(
        input_ids=input_ids,
        hidden_states_list=[torch.randn(1, context_rows, 8)],
        loss_mask=torch.ones(1, context_rows),
        lm_head_weight=torch.randn(config.vocab_size, 8),
        label_ids=label_ids,
        label_mask=torch.ones(1, label_rows),
    )

    # 12 is the trailing label (index 2) predicted from context row 1; it is not
    # present in input_ids, so it can only be supervised via the label sequence.
    assert "targets" in captured
    assert 12 in captured["targets"].tolist()


class _RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, **kwargs):
        self.seen.append(kwargs.get("dense_attention_mask"))
        return kwargs["draft_hidden"]


def test_draft_model_routes_per_layer_masks() -> None:
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        num_target_layers=4,
        num_context_layers=1,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
    )
    model = DFlashDraftModel(config)
    layers = nn.ModuleList([_RecordingLayer(), _RecordingLayer()])
    model.layers = layers

    sentinel_a = torch.zeros(1, 1, 4, 12, dtype=torch.bool)
    sentinel_b = torch.ones(1, 1, 4, 12, dtype=torch.bool)
    model(
        draft_input_ids=torch.zeros(1, 4, dtype=torch.long),
        context_feature=torch.zeros(1, 8, 8),
        draft_position_ids=torch.arange(4).view(1, 4),
        context_position_ids=torch.arange(8).view(1, 8),
        dense_attention_mask=[sentinel_a, sentinel_b],
    )

    assert layers[0].seen == [sentinel_a]
    assert layers[1].seen == [sentinel_b]


def _qwen3_6_target_stub() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=40,
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=2,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        rope_parameters={"rope_theta": 10000000, "rope_type": "default"},
    )


def test_dspark_fallback_matches_redhat_sliding_config() -> None:
    redhat_transformer = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "sliding_window": 2048,
        "layer_types": ["sliding_attention"] * 5,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
    }
    training = OmegaConf.create(
        {
            "dspark_num_target_layers": 5,
            "dspark_num_hidden_layers": 5,
            "dspark_intermediate_size": 6144,
            "dspark_sliding_window": 2048,
            "dspark_target_layer_ids": [2, 10, 20, 30, 37],
            "dspark_mask_token_id": 248077,
        }
    )
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training=training))
    )
    config = DSparkTrainerBackend._build_fallback_config(
        backend, _qwen3_6_target_stub()
    )

    for name, expected in redhat_transformer.items():
        assert getattr(config, name) == expected, name
    assert config.use_sliding_window is True
    assert config.rope_theta == 10000000
    assert config.mask_token_id == 248077
    assert config.target_layer_ids == [2, 10, 20, 30, 37]
    assert _resolve_sliding_windows(config) == [2048] * 5


def _packing_trainer(
    block_size: int = 4, max_packed_len: int = 0
) -> DrafterBaseTrainer:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "dspark_block_size": block_size,
                        "dflash_block_size": block_size,
                        "packing": {"enable": True, "max_packed_len": max_packed_len},
                    }
                }
            }
        }
    )
    return trainer


def test_pack_block_drafter_chunks_marks_padding_with_sentinel() -> None:
    trainer = _packing_trainer(block_size=4)
    packed = trainer._pack_block_drafter_chunks(
        input_id_chunks=[
            torch.tensor([10, 11, 12]),
            torch.tensor([20, 21, 22, 23, 24]),
        ],
        loss_mask_chunks=[torch.ones(3), torch.ones(5)],
        hidden_state_chunks=[torch.zeros(3, 2), torch.zeros(5, 2)],
        position_id_chunks=[torch.arange(3), torch.arange(5)],
        target_last_hidden_state_chunks=[],
    )

    assert packed is not None
    (
        input_ids,
        loss_mask,
        _base_h,
        _position_ids,
        attn_mask,
        document_ids,
        target,
        num_packed,
    ) = packed
    # Sample 0 (len 3) pads to 4, sample 1 (len 5) pads to 8.
    assert input_ids.shape == (1, 12)
    assert attn_mask.shape == (1, 12)
    assert loss_mask[0].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert document_ids[0].tolist() == [0, 0, 0, -1, 1, 1, 1, 1, 1, -1, -1, -1]
    assert target is None
    assert num_packed == 2


def test_pack_block_drafter_chunks_skips_oversized_without_dropping_rest() -> None:
    trainer = _packing_trainer(block_size=4, max_packed_len=12)
    packed = trainer._pack_block_drafter_chunks(
        input_id_chunks=[
            torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),  # pads to 8, fits.
            torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]),  # 8 more would exceed.
            torch.tensor([30, 31, 32]),  # pads to 4, fits in the remaining budget.
        ],
        loss_mask_chunks=[torch.ones(8), torch.ones(8), torch.ones(3)],
        hidden_state_chunks=[torch.zeros(8, 2), torch.zeros(8, 2), torch.zeros(3, 2)],
        position_id_chunks=[torch.arange(8), torch.arange(8), torch.arange(3)],
        target_last_hidden_state_chunks=[],
    )

    assert packed is not None
    input_ids, _loss_mask, _base_h, _position_ids, _attn, document_ids, _, num_packed = (
        packed
    )
    assert num_packed == 2
    assert input_ids.shape == (1, 12)
    # The oversized middle sample is skipped, not the whole remainder.
    assert document_ids[0].tolist() == [0] * 8 + [1, 1, 1, -1]
    assert input_ids[0].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 0]


def test_pack_block_drafter_chunks_rejects_partial_target_hidden() -> None:
    trainer = _packing_trainer(block_size=4)
    packed = trainer._pack_block_drafter_chunks(
        input_id_chunks=[torch.tensor([10, 11, 12])],
        loss_mask_chunks=[torch.ones(3)],
        hidden_state_chunks=[torch.zeros(3, 2)],
        position_id_chunks=[torch.arange(3)],
        target_last_hidden_state_chunks=[torch.zeros(3, 2), torch.zeros(3, 2)],
    )

    assert packed is None


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("dflash", True),
        ("dflash2", True),
        ("dspark", True),
        ("domino", True),
        ("eagle3", False),
    ],
)
def test_packing_gate_only_for_block_drafters(model_type, expected) -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type=model_type)
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"packing": {"enable": True}}}}}
    )
    assert trainer._packing_enabled() is expected

    trainer.config.rollout.drafter.training.packing.enable = False
    assert trainer._packing_enabled() is False


def test_domino_training_forward_accepts_document_ids() -> None:
    from verl_speco.backends.domino_trainer_backend import DominoTrainingModel
    from verl_speco.models.domino import DominoConfig, DominoDraftModel

    config = DominoConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=8,
        emb_dim=6,
        gru_hidden_dim=10,
        pure_draft_prefix_len=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    model = DominoTrainingModel(
        draft_model=DominoDraftModel(config),
        block_size=config.block_size,
        num_anchors=config.num_anchors,
        pure_draft_prefix_len=config.pure_draft_prefix_len,
    )

    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    hidden_states_list = [
        torch.randn(bsz, seq_len, config.target_hidden_size)
        for _ in config.target_layer_ids
    ]
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)
    # Two length-8 documents packed into one row.
    document_ids = torch.tensor([[0] * 8 + [1] * 8] * bsz)

    result = model(
        input_ids,
        hidden_states_list,
        loss_mask,
        lm_head_weight,
        document_ids=document_ids,
    )

    assert torch.isfinite(result[0])


def test_dflash2_training_forward_accepts_document_ids() -> None:
    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainingModel
    from verl_speco.models.dflash2 import DFlash2Config, DFlash2DraftModel

    config = DFlash2Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=8,
        conv_kernel_size=2,
        conv_group_size=4,
        selector_rank=6,
        selector_top_k=5,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    model = DFlash2TrainingModel(
        draft_model=DFlash2DraftModel(config),
        block_size=config.block_size,
        num_anchors=config.num_anchors,
    )

    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    hidden_states_list = [
        torch.randn(bsz, seq_len, config.target_hidden_size)
        for _ in config.target_layer_ids
    ]
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)
    document_ids = torch.tensor([[0] * 8 + [1] * 8] * bsz)

    result = model(
        input_ids,
        hidden_states_list,
        loss_mask,
        lm_head_weight,
        document_ids=document_ids,
    )

    assert torch.isfinite(result[0])


def test_pack_block_drafter_batch_exposes_document_ids() -> None:
    for model_type in ("dflash", "dflash2", "dspark", "domino"):
        trainer = _packing_trainer(block_size=4)
        trainer.backend = SimpleNamespace(model_type=model_type)
        assert trainer._packing_enabled() is True

        fields, num_packed = trainer._pack_block_drafter_batch(
            input_id_chunks=[
                torch.tensor([10, 11, 12]),
                torch.tensor([20, 21, 22, 23, 24]),
            ],
            loss_mask_chunks=[torch.ones(3), torch.ones(5)],
            hidden_state_chunks=[torch.zeros(3, 2), torch.zeros(5, 2)],
            position_id_chunks=[torch.arange(3), torch.arange(5)],
            target_last_hidden_state_chunks=[],
        )

        assert fields is not None
        assert fields["input_ids"].shape == (1, 12)
        assert fields["document_ids"][0].tolist() == [
            0,
            0,
            0,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
        ]
        assert "target_last_hidden_states" not in fields
        assert num_packed == 2


def test_prepare_training_batch_carries_document_ids_to_final_batch() -> None:
    seq_len, hidden = 8, 4
    ids = torch.arange(seq_len, dtype=torch.long)
    hidden_states = torch.randn(seq_len, hidden)
    loss_mask = torch.ones(seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.long)

    class _FakeBackend:
        model_type = "dflash"

        def preprocess_individual_items(self, items, dev, model_config):
            return {
                "ids": [ids],
                "h_states": [hidden_states],
                "masks": [loss_mask],
                "position_ids": [position_ids],
            }

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = _FakeBackend()
    trainer.batch_size = 1
    trainer.current_rl_step = 0
    trainer.training_steps = 0
    trainer.use_data_buffer = False
    trainer.collected_data = [{"step": 0, "hidden_states": hidden_states}]
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "dflash_block_size": 4,
                        "packing": {"enable": True, "max_packed_len": 0},
                    }
                }
            }
        }
    )
    trainer.use_ulysses_sp = False
    trainer.rank = 0
    trainer.model_config = None
    trainer.model = nn.Linear(2, 2)

    batch = trainer._prepare_training_batch()

    assert batch is not None
    assert batch["document_ids"][0].tolist() == [0] * seq_len
