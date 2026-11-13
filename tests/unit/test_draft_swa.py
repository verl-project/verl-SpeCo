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


@pytest.mark.parametrize("windows", [[2, 5], [2, None], [None]])
def test_owned_anchor_windows_preserve_attention_and_labels(windows):
    from verl_speco.trainer.standalone_batch_prepare import plan_owned_anchor_windows
    docs = [0] * 9 + [-1] * 2 + [1] * 10 + [-1]
    anchors = [3, 6, 13, 18, 0]
    keep = [True, True, True, True, False]
    state = torch.random.get_rng_state().clone()
    groups = plan_owned_anchor_windows(docs, anchors, keep, block_size=2, label_shift=1,
                                       windows=windows, world_size=3)
    assert [[p["anchor_slot"] for p in group] for group in groups] == [[0, 3], [1], [2]]
    assert torch.equal(state, torch.random.get_rng_state())
    for window in windows:
        full = _create_dflash_dense_attention_mask(
            torch.tensor([anchors]), torch.tensor([keep]), len(docs), 2,
            sliding_window=window, document_ids=torch.tensor([docs]))
        for part in (part for group in groups for part in group):
            slot, start, end = part["anchor_slot"], part["start"], part["end"]
            cropped = _create_dflash_dense_attention_mask(
                torch.tensor([[part["local_anchor"]]]), torch.tensor([[True]]), end - start, 2,
                sliding_window=window, document_ids=torch.tensor([docs[start:end]]))
            original = full[0, 0, slot * 2, :len(docs)]
            assert not original[:start].any() and not original[end:].any()
            assert torch.equal(original[start:end], cropped[0, 0, 0, :end - start])
            assert torch.equal(full[0, 0, slot * 2, len(docs) + slot * 2:len(docs) + (slot + 1) * 2],
                               cropped[0, 0, 0, end - start:])
            assert docs[start:end] == [part["document_id"]] * (end - start)
            assert list(range(part["anchor"] + 1, part["anchor"] + 3)) == list(range(end - 2, end))


@pytest.mark.parametrize("docs,anchors,keep", [
    ([0, 0, 1, 1], [1], [True]), ([0, -1, 0], [0], [True]),
    ([0, 0], [0, 0], [True, True]), ([0, -1], [1], [True]),
])
def test_owned_anchor_windows_reject_invalid_document_or_anchor(docs, anchors, keep):
    from verl_speco.trainer.standalone_batch_prepare import plan_owned_anchor_windows
    with pytest.raises(ValueError):
        plan_owned_anchor_windows(docs, anchors, keep, block_size=2, label_shift=1,
                                  windows=[2], world_size=2)


@pytest.mark.parametrize("windows", [[3, 3], [3, None]])
def test_owned_anchor_window_matches_model_output_and_gradient(windows):
    from copy import deepcopy
    from verl_speco.trainer.standalone_batch_prepare import plan_owned_anchor_windows

    torch.manual_seed(73)
    config = DFlashConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                          vocab_size=16, num_target_layers=4, num_context_layers=1,
                          target_hidden_size=8, target_num_hidden_layers=4, target_layer_ids=[1],
                          sliding_window=3, use_sliding_window=True,
                          layer_types=["sliding_attention" if w else "full_attention" for w in windows])
    model = DFlashDraftModel(config).eval()
    cropped_model = deepcopy(model)
    docs = [0] * 6 + [-1] + [1] * 9
    anchor = 12
    part = plan_owned_anchor_windows(docs, [anchor], [True], block_size=2, label_shift=1,
                                      windows=windows, world_size=1)[0][0]
    start, end = part["start"], part["end"]
    context = torch.randn(1, len(docs), 8, requires_grad=True)
    local_context = context.detach()[:, start:end].clone().requires_grad_()
    noise = torch.randn(1, 2, 8)
    def forward(module, features, docs_slice, anchor_pos):
        length = len(docs_slice)
        masks = [_create_dflash_dense_attention_mask(
            torch.tensor([[anchor_pos]]), torch.tensor([[True]]), length, 2,
            sliding_window=window, document_ids=torch.tensor([docs_slice])) for window in windows]
        return module(draft_input_ids=None, context_feature=features,
                      draft_position_ids=torch.tensor([[anchor_pos, anchor_pos + 1]]),
                      context_position_ids=torch.arange(length).unsqueeze(0),
                      dense_attention_mask=masks, noise_embedding=noise)
    full = forward(model, context, docs, anchor)
    local = forward(cropped_model, local_context, docs[start:end], part["local_anchor"])
    torch.testing.assert_close(full, local, rtol=2e-5, atol=2e-6)
    weights = torch.randn_like(full)
    (full * weights).sum().backward()
    (local * weights).sum().backward()
    torch.testing.assert_close(context.grad[:, start:end], local_context.grad, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(context.grad[:, :start]) == 0 and torch.count_nonzero(context.grad[:, end:]) == 0
    for original, cropped in zip(model.parameters(), cropped_model.parameters(), strict=True):
        assert (original.grad is None) == (cropped.grad is None)
        if original.grad is not None:
            torch.testing.assert_close(original.grad, cropped.grad, rtol=2e-5, atol=2e-6)


def _anchor_test_model():
    config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                          vocab_size=16, num_target_layers=4, num_context_layers=1,
                          target_hidden_size=8, target_num_hidden_layers=4, target_layer_ids=[1],
                          mask_token_id=15, markov_rank=4, markov_head_type="vanilla")
    return DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                               num_anchors=3, l1_loss_alpha=0.0).eval()


def test_preselected_anchors_match_original_sampling_and_gradient():
    from copy import deepcopy
    torch.manual_seed(31)
    model = _anchor_test_model()
    replay = deepcopy(model)
    docs = torch.tensor([[0] * 6 + [-1] + [1] * 6])
    mask = (docs >= 0).float()
    args = dict(input_ids=torch.arange(13).reshape(1, 13), hidden_states_list=[torch.randn(1, 13, 8)],
                loss_mask=mask, lm_head_weight=torch.randn(16, 8), document_ids=docs)
    torch.manual_seed(41)
    anchors, keep = replay._sample_anchor_positions(13, mask, torch.device("cpu"), document_ids=docs)
    after_sample = torch.random.get_rng_state().clone()
    replay_out = replay(**args, preselected_anchor_positions=anchors, preselected_anchor_keep_mask=keep)
    assert torch.equal(after_sample, torch.random.get_rng_state())
    torch.manual_seed(41)
    original_out = model(**args)
    for original, selected in zip(original_out[:5], replay_out[:5], strict=True):
        torch.testing.assert_close(original, selected)
    original_out[0].backward()
    replay_out[0].backward()
    for original, selected in zip(model.parameters(), replay.parameters(), strict=True):
        assert (original.grad is None) == (selected.grad is None)
        if original.grad is not None:
            torch.testing.assert_close(original.grad, selected.grad)


@pytest.mark.parametrize("fault", ["boundary", "mask", "duplicate", "outside", "dtype", "missing"])
def test_preselected_anchors_reject_invalid_inputs(fault):
    model = _anchor_test_model()
    docs = torch.tensor([[0, 0, 0, -1, 1, 1, 1]])
    mask = torch.tensor([[1., 1., 1., 0., 1., 1., 1.]])
    anchors, keep = torch.tensor([[0]]), torch.tensor([[True]])
    if fault == "boundary": anchors = torch.tensor([[1]])
    elif fault == "mask": mask[0, 1] = 0
    elif fault == "duplicate": anchors, keep = torch.tensor([[0, 0]]), torch.tensor([[True, True]])
    elif fault == "outside": anchors = torch.tensor([[7]])
    elif fault == "dtype": anchors = anchors.float()
    else: keep = None
    with pytest.raises(ValueError, match="[Aa]nchor"):
        model._validate_preselected_anchors(anchors, keep, seq_len=7, label_mask=mask,
                                            document_ids=docs, next_token_loss_mask=None)


def test_preselected_anchors_follow_aligned_next_token_mask():
    model = _anchor_test_model()
    docs = torch.tensor([[0, 0, 0, -1, 1, 1, 1]])
    mask = (docs >= 0).float()
    next_mask = mask.clone()
    anchors, keep = model._sample_anchor_positions(7, mask, torch.device("cpu"),
                                                    document_ids=docs, next_token_loss_mask=next_mask)
    positions, active = model._validate_preselected_anchors(
        anchors, keep, seq_len=7, label_mask=mask, document_ids=docs, next_token_loss_mask=next_mask)
    assert torch.equal(positions, anchors) and torch.equal(active, keep)
    if keep.any():
        next_mask[0, int(anchors[keep][0])] = 0
        with pytest.raises(ValueError, match="boundaries"):
            model._validate_preselected_anchors(anchors, keep, seq_len=7, label_mask=mask,
                                                document_ids=docs, next_token_loss_mask=next_mask)


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


def _collection_trainer(model_type: str) -> DrafterBaseTrainer:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type=model_type)
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"use_logits": False}}}}
    )
    trainer.copy_stream = None
    trainer.rank = 0
    trainer.pad_token_id = 0
    trainer.model_config = None
    trainer.current_rl_step = 0
    trainer.use_data_buffer = False
    trainer.collected_data = []
    trainer.buffer_version = 0
    return trainer


def _collect_two_response_tokens(model_type: str) -> dict:
    trainer = _collection_trainer(model_type)
    trainer.collect_online_data(
        {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "prompts": torch.tensor([[10]]),
            "responses": torch.tensor([[11, 12]]),
        },
        torch.randn(1, 2, 4),
    )
    assert len(trainer.collected_data) == 1
    return trainer.collected_data[0]


def test_dflash_collection_keeps_trailing_label_for_sampler() -> None:
    """One prompt token + two response tokens + two hidden rows must keep the
    trailing label so the DFlash anchor sampler still finds a valid anchor."""
    item = _collect_two_response_tokens("dflash")
    assert item["input_ids"].tolist() == [10, 11, 12]
    assert item["hidden_states"].size(0) == 2

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
        draft_model=DFlashDraftModel(config), block_size=2, num_anchors=2
    )
    label_mask = (item["loss_mask"] > 0).float().unsqueeze(0)
    _, keep = model._sample_anchor_positions(
        item["input_ids"].numel(), label_mask, torch.device("cpu")
    )
    assert int(keep.sum()) >= 1


def test_dspark_collection_keeps_context_aligned_with_hidden_rows() -> None:
    item = _collect_two_response_tokens("dspark")
    assert item["input_ids"].tolist() == [10, 11]
    assert item["next_token_ids"].tolist() == [11, 12]


def test_dflash_raw_target_hidden_requires_standalone_reference_source() -> None:
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
    norm = nn.LayerNorm(8)
    model = DFlashTrainingModel(
        draft_model=DFlashDraftModel(config),
        block_size=2,
        num_anchors=2,
        lk_loss_alpha=1.0,
        verifier_norm=norm,
        target_hidden_clip_value=1.0e4,
    )
    model.eval()
    kwargs = dict(
        input_ids=torch.tensor([[10, 11, 12]]),
        hidden_states_list=[torch.randn(1, 3, 8)],
        loss_mask=torch.ones(1, 3),
        lm_head_weight=torch.randn(config.vocab_size, 8),
        target_last_hidden_states=torch.randn(1, 3, 8),
        target_hidden_is_raw=True,
    )
    with pytest.raises(ValueError, match="standalone vLLM"):
        model(**kwargs, hidden_source="co_train")
    outputs = model(**kwargs, hidden_source="standalone_vllm_reference")
    assert torch.isfinite(outputs[0])


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
        next_token_ids,
        next_token_loss_mask,
        num_packed,
    ) = packed
    # Sample 0 (len 3) pads to 4, sample 1 (len 5) pads to 8.
    assert input_ids.shape == (1, 12)
    assert attn_mask.shape == (1, 12)
    assert loss_mask[0].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert document_ids[0].tolist() == [0, 0, 0, -1, 1, 1, 1, 1, 1, -1, -1, -1]
    assert target is None
    assert next_token_ids is None
    assert next_token_loss_mask is None
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
    (
        input_ids,
        _loss_mask,
        _base_h,
        _position_ids,
        _attn,
        document_ids,
        _,
        _,
        _,
        num_packed,
    ) = packed
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


def test_pack_block_drafter_batch_keeps_aligned_next_token_labels() -> None:
    trainer = _packing_trainer(block_size=4)
    trainer.backend = SimpleNamespace(model_type="dspark")

    fields, num_packed = trainer._pack_block_drafter_batch(
        input_id_chunks=[torch.tensor([10, 11, 12])],
        loss_mask_chunks=[torch.ones(3)],
        hidden_state_chunks=[torch.zeros(3, 2)],
        position_id_chunks=[torch.arange(3)],
        target_last_hidden_state_chunks=[],
        next_token_id_chunks=[torch.tensor([11, 12, 13])],
        next_token_loss_mask_chunks=[torch.ones(3)],
    )

    assert num_packed == 1
    assert fields["input_ids"][0].tolist() == [10, 11, 12, 0]
    assert fields["next_token_ids"][0].tolist() == [11, 12, 13, 0]
    assert fields["next_token_loss_mask"][0].tolist() == [1, 1, 1, 0]


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
    trainer._active_training_reservation_id = None
    trainer._bubble_time_enabled = False
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


def test_prepare_dspark_batch_keeps_trailing_label_out_of_context() -> None:
    ids = torch.tensor([10, 11])
    next_ids = torch.tensor([11, 12])
    hidden_states = torch.randn(2, 4)
    loss_mask = torch.ones(2)

    class _FakeBackend:
        model_type = "dspark"

        def preprocess_individual_items(self, items, dev, model_config):
            return {
                "ids": [ids],
                "h_states": [hidden_states],
                "masks": [loss_mask],
                "position_ids": [torch.arange(2)],
                "target_last_h_states": [None],
                "next_token_ids": [next_ids],
                "next_token_masks": [loss_mask],
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
                        "dspark_block_size": 2,
                        "dspark_l1_loss_alpha": 0.0,
                        "dspark_lk_loss_alpha": 0.0,
                        "packing": {"enable": False},
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
    assert batch["input_ids"].tolist() == [[10, 11]]
    assert batch["hidden_states"].shape[1] == 2
    assert batch["next_token_ids"].tolist() == [[11, 12]]


def _eagle3_lk_replay_trainer(*, synced_step=3, **item_overrides):
    ids = torch.arange(5, dtype=torch.long)
    hidden_states = torch.randn(5, 4)
    last_hidden_states = torch.randn(5, 4)
    loss_mask = torch.ones(5)

    class _FakeBackend:
        model_type = "eagle3"

        def preprocess_individual_items(self, items, dev, model_config):
            return {
                "ids": [ids],
                "h_states": [hidden_states],
                "masks": [loss_mask],
                "position_ids": [torch.arange(5)],
                "last_h_states": [last_hidden_states],
                "target_logz": [items[0].get("target_logz")],
            }

    item = {
        "step": 3,
        "global_step": 3,
        "hidden_states": hidden_states,
        "target_logz": torch.ones(5),
        "target_logz_temperature": 1.0,
        **item_overrides,
    }
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = _FakeBackend()
    trainer.batch_size = 1
    trainer.current_rl_step = 3
    trainer.training_steps = 0
    trainer.use_data_buffer = False
    trainer._bubble_time_enabled = False
    trainer.collected_data = [item]
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "use_logits": False,
                        "eagle3_lk_loss_alpha": 1.0,
                        "lk_temperature": 1.0,
                    }
                }
            }
        }
    )
    trainer.use_ulysses_sp = False
    trainer.rank = 0
    trainer.model_config = None
    trainer.model = nn.Linear(2, 2)
    trainer._target_lm_head_weight_step = synced_step
    return trainer


def test_eagle3_lk_replay_rejects_missing_synced_target_snapshot() -> None:
    trainer = _eagle3_lk_replay_trainer(synced_step=None)

    with pytest.raises(ValueError, match="synchronized target LM-head snapshot"):
        trainer._prepare_training_batch()


def test_eagle3_lk_replay_rejects_temperature_mismatch() -> None:
    trainer = _eagle3_lk_replay_trainer(target_logz_temperature=0.5)

    with pytest.raises(ValueError, match="temperature mismatch"):
        trainer._prepare_training_batch()


def test_eagle3_lk_replay_rejects_missing_target_logz() -> None:
    trainer = _eagle3_lk_replay_trainer(target_logz=None)

    with pytest.raises(ValueError, match="missing target_logz"):
        trainer._prepare_training_batch()


def test_prepare_training_batch_packing_keeps_trailing_label_row() -> None:
    seq_len, hidden = 8, 4
    ids = torch.arange(seq_len + 1, dtype=torch.long)
    hidden_states = torch.randn(seq_len, hidden)
    loss_mask = torch.ones(seq_len + 1)

    class _FakeBackend:
        model_type = "dflash"

        def preprocess_individual_items(self, items, dev, model_config):
            return {
                "ids": [ids],
                "h_states": [hidden_states],
                "masks": [loss_mask],
                "position_ids": [torch.arange(seq_len, dtype=torch.long)],
            }

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = _FakeBackend()
    trainer._active_training_reservation_id = None
    trainer._bubble_time_enabled = False
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
    assert batch["input_ids"][0].tolist() == list(range(seq_len + 1)) + [0, 0, 0]
    assert batch["document_ids"][0].tolist() == [0] * (seq_len + 1) + [-1] * 3
    assert batch["hidden_states"][0, seq_len:].abs().sum().item() == 0


@pytest.mark.parametrize("sliding_window", [None, 3])
def test_flex_and_dense_masks_match_for_unused_anchor_rows(sliding_window):
    anchors = torch.tensor([[3, 5, 0]])
    keep = torch.tensor([[True, True, False]])
    documents = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    kwargs = dict(anchor_positions=anchors, block_keep_mask=keep, ctx_len=8,
                  block_size=2, sliding_window=sliding_window, document_ids=documents)
    dense = _create_dflash_dense_attention_mask(**kwargs)
    flex = _create_dflash_mask_mod(**kwargs)
    for query in range(6):
        for key in range(14):
            assert bool(flex(0, 0, query, key)) == bool(dense[0, 0, query, key])
    assert dense[0, 0, 4].sum() == 1 and dense[0, 0, 4, 12]
    assert dense[0, 0, 5].sum() == 1 and dense[0, 0, 5, 13]


@pytest.mark.parametrize("algorithm,markov,confidence", [
    ("dflash", None, False), ("dspark", "vanilla", False),
    ("dspark", "gated", True),
])
def test_zero_supervision_row_has_finite_zero_gradient(algorithm, markov, confidence):
    common = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                  vocab_size=16, num_target_layers=4, num_context_layers=1,
                  target_hidden_size=8, target_num_hidden_layers=4,
                  target_layer_ids=[1], mask_token_id=15)
    if algorithm == "dspark":
        config = DSparkConfig(**common, markov_rank=4, markov_head_type=markov,
                              enable_confidence_head=confidence, confidence_head_with_markov=confidence)
        model = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                    num_anchors=2, l1_loss_alpha=0.0,
                                    confidence_head_alpha=float(confidence))
    else:
        config = DFlashConfig(**common)
        model = DFlashTrainingModel(draft_model=DFlashDraftModel(config), block_size=2, num_anchors=2)
    result = model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                   hidden_states_list=[torch.randn(1, 5, 8)], loss_mask=torch.zeros(1, 5),
                   lm_head_weight=torch.randn(16, 8))
    loss = result[0]
    assert torch.isfinite(loss) and loss.item() == pytest.approx(0)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all()
                             and torch.count_nonzero(gradient) == 0 for gradient in gradients)


def _zero_supervision_ddp_worker(rank, port, algorithm, confidence, strategy, results):
    from copy import deepcopy
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(17)
        common = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                      vocab_size=16, num_target_layers=4, num_context_layers=1,
                      target_hidden_size=8, target_num_hidden_layers=4,
                      target_layer_ids=[1], mask_token_id=15)
        if algorithm == "dspark":
            cfg = DSparkConfig(**common, markov_rank=4, markov_head_type="gated",
                               enable_confidence_head=confidence, confidence_head_with_markov=confidence)
            raw = DSparkTrainingModel(draft_model=DSparkDraftModel(cfg), block_size=2,
                                      num_anchors=2, l1_loss_alpha=0.0,
                                      confidence_head_alpha=float(confidence))
        else:
            cfg = DFlashConfig(**common)
            raw = DFlashTrainingModel(draft_model=DFlashDraftModel(cfg), block_size=2, num_anchors=2)
        reference = deepcopy(raw)
        if strategy == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(raw, broadcast_buffers=False)
            model_to_compare = model.module
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard
            mesh = init_device_mesh("cpu", (2,))
            for layer in raw.draft_model.layers:
                fully_shard(layer, mesh=mesh)
            fully_shard(raw, mesh=mesh)
            model = model_to_compare = raw
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                      hidden_states_list=[torch.randn(1, 5, 8)],
                      lm_head_weight=torch.randn(16, 8))
        if confidence:
            inputs["target_last_hidden_states"] = torch.randn(1, 5, 8)
        rng = torch.get_rng_state()
        if rank == 0:
            ref_result = reference(**inputs, loss_mask=torch.ones(1, 5))
            ref_result[0].backward()
            torch.set_rng_state(rng)
        result = model(**inputs, loss_mask=torch.ones(1, 5) if rank == 0 else torch.zeros(1, 5))
        tokens = (result[5]["ce_weighted_token_count"] if algorithm == "dspark"
                  else result[4].sum()).detach().clone()
        global_tokens = tokens.clone()
        dist.all_reduce(global_tokens)
        assert global_tokens.item() > 0
        (result[0] * tokens * 2.0 / global_tokens).backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad)
        diffs = []
        for parameter, ref_parameter in zip(model_to_compare.parameters(), reference.parameters(), strict=True):
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad.full_tensor() if strategy == "fsdp2" else parameter.grad
            if rank == 0:
                diffs.append((gradient - ref_parameter.grad).abs().max().item())
        if rank == 0:
            results.put(max(diffs, default=0.0))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("algorithm,confidence", [("dspark", False), ("dspark", True), ("dflash", False)])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp2"])
def test_zero_supervision_rank_matches_real_model_reference(algorithm, confidence, strategy):
    import multiprocessing
    import socket
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_zero_supervision_ddp_worker,
                                 args=(rank, port, algorithm, confidence, strategy, results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
        assert all(process.exitcode == 0 for process in processes)
        assert results.get(timeout=5) < 1e-5
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


def test_dspark_rnn_empty_supervision_preserves_zero_loss():
    config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                          vocab_size=16, num_target_layers=4, num_context_layers=1,
                          target_hidden_size=8, target_num_hidden_layers=4,
                          target_layer_ids=[1], mask_token_id=15,
                          markov_rank=4, markov_head_type="rnn")
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                num_anchors=2, l1_loss_alpha=0.0)
    result = model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                   hidden_states_list=[torch.randn(1, 5, 8)], loss_mask=torch.zeros(1, 5),
                   lm_head_weight=torch.randn(16, 8))
    assert torch.isfinite(result[0]) and result[0].item() == 0
