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
"""L3: LiLiCorr head numeric parity against the speculators training head.

The serving contract is pinned by speculators' own
``tests/integration/models/test_lilicorr_vllm_parity.py``; this test pins the
same contract for the verl-SpeCo port, without needing vLLM installed.
"""

from __future__ import annotations

import pytest
import torch

from verl_speco.models.lilicorr import (
    LiLiCorrConfig,
    LiLiCorrHead,
    compute_lattice_terms,
)

spec_head_mod = pytest.importorskip(
    "speculators.models.lilicorr.model_definitions",
    reason="speculators (lilicorr branch) not importable",
)

MODEL_HIDDEN = 16
BLOCK = 8
TOPK = 4


def _ours(head_width: int, seed: int = 7) -> LiLiCorrHead:
    torch.manual_seed(seed)
    head = LiLiCorrHead(
        model_hidden_size=MODEL_HIDDEN,
        block_size=BLOCK,
        candidate_topk=TOPK,
        hidden_size=head_width,
        factor_dim=12,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        logit_scale=8.0,
        vector_eps=1e-6,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        for param in head.parameters():
            param.normal_(std=0.1)
    return head


def _theirs(head_width: int, seed: int = 7):
    torch.manual_seed(seed)
    head = spec_head_mod.LiLiCorrHead(
        model_hidden_size=MODEL_HIDDEN,
        block_size=BLOCK,
        candidate_topk=TOPK,
        hidden_size=head_width,
        factor_dim=12,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        logit_scale=8.0,
        vector_eps=1e-6,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        for param in head.parameters():
            param.normal_(std=0.1)
    return head


def _inputs(seed: int = 123):
    gen = torch.Generator().manual_seed(seed)
    num_slots = BLOCK - 1

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    log_probs = randn(2, num_slots, 23).log_softmax(-1).topk(TOPK).values
    return (
        randn(2, num_slots, TOPK, MODEL_HIDDEN),
        log_probs,
        randn(2, num_slots, MODEL_HIDDEN),
        randn(2, MODEL_HIDDEN),
        torch.tensor([True, False]),
    )


@pytest.mark.parametrize("head_width", [8, 16], ids=["projected", "identity-token"])
def test_state_dict_is_identical(head_width):
    ours = _ours(head_width)
    theirs = _theirs(head_width)
    ours_state = ours.state_dict()
    theirs_state = theirs.state_dict()
    assert set(ours_state) == set(theirs_state), (
        set(ours_state) ^ set(theirs_state)
    )
    theirs.load_state_dict(ours_state, strict=True)


@pytest.mark.parametrize("head_width", [8, 16], ids=["projected", "identity-token"])
def test_scores_match(head_width):
    ours = _ours(head_width)
    theirs = _theirs(head_width)
    theirs.load_state_dict(ours.state_dict(), strict=True)
    inputs = _inputs()
    with torch.no_grad():
        a_start, a_pair = ours(*inputs)
        b_start, b_pair = theirs(*inputs)
    torch.testing.assert_close(a_start, b_start, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a_pair, b_pair, atol=1e-6, rtol=1e-6)


def test_lattice_terms_grad_and_metrics():
    torch.manual_seed(0)
    vocab = 37
    block = 4
    head = LiLiCorrHead(
        model_hidden_size=MODEL_HIDDEN,
        block_size=block,
        candidate_topk=TOPK,
        hidden_size=8,
        factor_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
        logit_scale=8.0,
        vector_eps=1e-6,
        rms_norm_eps=1e-6,
    )
    embed = torch.nn.Embedding(vocab, MODEL_HIDDEN)
    lm_head = torch.nn.Linear(MODEL_HIDDEN, vocab, bias=False)
    bsz, nb = 2, 3
    draft_hidden = torch.randn(bsz, nb, block, MODEL_HIDDEN, requires_grad=True)
    target_hidden = torch.randn(bsz, nb, block, MODEL_HIDDEN)
    anchor_hidden = torch.randn(bsz, nb, MODEL_HIDDEN)
    loss, metrics = compute_lattice_terms(
        head=head,
        embed_tokens=embed,
        lm_head_weight=lm_head.weight,
        block_size=block,
        draft_hidden=draft_hidden,
        target_hidden=target_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=torch.ones(bsz, nb, dtype=torch.bool),
        valid_block=torch.ones(bsz, nb, dtype=torch.bool),
        slot_mask=torch.ones(bsz, nb, block, dtype=torch.bool),
        chunk_blocks=2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert draft_hidden.grad is not None
    assert draft_hidden.grad.abs().sum() > 0
    for key in (
        "lilicorr_ce_sum",
        "lilicorr_penalty_sum",
        "lilicorr_eal_sum",
        "lilicorr_top1_eal_sum",
        "lilicorr_oracle_eal_sum",
        "lilicorr_eal_total",
    ):
        assert key in metrics


def test_config_nested_lift(tmp_path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 101,
                "architectures": ["LiLiCorrDraftModel"],
                "draft_vocab_size": 101,
                "dflash_config": {
                    "block_size": 4,
                    "target_layer_ids": [1, 33],
                    "mask_token_id": 100,
                    "lilicorr_candidate_topk": 8,
                    "lilicorr_hidden_size": 32,
                    "lilicorr_factor_dim": 16,
                    "lilicorr_num_layers": 2,
                    "lilicorr_num_heads": 4,
                    "lilicorr_mlp_ratio": 2.0,
                    "lilicorr_logit_scale": 8.0,
                    "lilicorr_vector_eps": 1e-4,
                    "sample_from_anchor": False,
                },
            }
        )
    )
    cfg = LiLiCorrConfig.from_lilicorr_pretrained(str(tmp_path))
    assert cfg.block_size == 4
    assert cfg.lilicorr_candidate_topk == 8
    assert cfg.lilicorr_hidden_size == 32
    assert cfg.lilicorr_factor_dim == 16
    assert cfg.target_layer_ids == [1, 33]
    assert cfg.model_type == "lilicorr"
