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
"""End-to-end smoke for the LiLiCorr training hook (CPU, tiny dims).

Exercises the real DFlash forward: anchor sampling, block layout, the lattice
objective, metrics plumbing, and backward into the backbone.
"""

from __future__ import annotations

import torch

from verl_speco.backends.lilicorr_trainer_backend import LiLiCorrTrainingModel
from verl_speco.models.lilicorr import LiLiCorrConfig, LiLiCorrDraftModel

HIDDEN = 32
VOCAB = 8  # == candidate_topk: keep every gt in the lattice so the smoke has supervision
BLOCK = 4
CONTEXT_LAYERS = 2
SEQ = 24


def _tiny_config() -> LiLiCorrConfig:
    return LiLiCorrConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        num_target_layers=8,
        num_context_layers=CONTEXT_LAYERS,
        target_hidden_size=HIDDEN,
        target_num_hidden_layers=8,
        target_layer_ids=[1, 6],
        mask_token_id=VOCAB - 1,
        block_size=BLOCK,
        num_anchors=2,
        loss_decay_gamma=0.0,
        lilicorr_hidden_size=None,
        lilicorr_factor_dim=8,
        lilicorr_num_layers=1,
        lilicorr_num_heads=2,
        lilicorr_candidate_topk=8,
    )


def test_forward_backward_produces_lattice_metrics():
    torch.manual_seed(0)
    cfg = _tiny_config()
    draft_model = LiLiCorrDraftModel(cfg)
    lm_head_weight = torch.randn(VOCAB, HIDDEN, requires_grad=False) * 0.02

    model = LiLiCorrTrainingModel(
        draft_model=draft_model,
        block_size=BLOCK,
        num_anchors=2,
        loss_decay_gamma=0.0,
        loss_mode="full_vocab",
        lilicorr_w_ce=0.25,
        lilicorr_w_pen=0.25,
        loss_chunk_anchors=1,
    )

    input_ids = torch.randint(0, VOCAB, (1, SEQ))
    loss_mask = torch.ones(1, SEQ)
    hidden_list = [torch.randn(1, SEQ, HIDDEN) for _ in range(CONTEXT_LAYERS)]
    target_last = torch.randn(1, SEQ, HIDDEN)
    document_ids = torch.zeros(1, SEQ, dtype=torch.long)

    loss, _, _, _, _, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_list,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last,
        document_ids=document_ids,
    )

    assert torch.isfinite(loss)
    for key in (
        "lilicorr_ce_sum",
        "lilicorr_penalty_sum",
        "lilicorr_eal_sum",
        "lilicorr_top1_eal_sum",
        "lilicorr_oracle_eal_sum",
    ):
        assert key in diagnostics, key
        assert torch.isfinite(torch.as_tensor(diagnostics[key]))

    loss.backward()
    # The lattice terms must reach the head and the backbone.
    head_grad = draft_model.lilicorr.out_head.weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0


def test_config_validation_rejects_bad_topk():
    import pytest

    with pytest.raises(ValueError, match="power of two"):
        LiLiCorrConfig(
            hidden_size=HIDDEN,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=VOCAB,
            block_size=BLOCK,
            lilicorr_candidate_topk=3,
            lilicorr_hidden_size=None,
            lilicorr_num_heads=2,
        )
