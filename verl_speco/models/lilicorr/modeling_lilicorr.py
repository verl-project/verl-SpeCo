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
"""LiLiCorr draft model: DFlash backbone + candidate-lattice head.

The head is a faithful port of speculators'
``src/speculators/models/lilicorr/model_definitions.py`` (commit d8c73a88,
Apache-2.0, itself ported from NVIDIA ModelOpt). Submodule names follow vLLM's
``vllm/model_executor/models/lilicorr.py`` so a checkpoint saved from here loads
there directly: the serving loader collects keys under ``model.lilicorr.*`` and
hands them to an ``AutoWeightsLoader`` that never applies the backbone's
``up_proj -> gate_up_proj`` stacking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from verl_speco.models.dflash import DFlashDraftModel, DFlashRMSNorm
from verl_speco.models.lilicorr.configuration_lilicorr import LiLiCorrConfig

__all__ = [
    "LiLiCorrAttention",
    "LiLiCorrDraftModel",
    "LiLiCorrHead",
    "LiLiCorrLayer",
    "LiLiCorrMLP",
    "compute_lattice_terms",
]

# [log_p, p, log_p - max log_p, rank / (k - 1), is_top1]
_NUM_CANDIDATE_FEATURES = 5


class LiLiCorrMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.up_proj = nn.Linear(input_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class LiLiCorrAttention(nn.Module):
    """Unmasked attention over one block's lattice with an additive bias.

    Parameters use ``nn.MultiheadAttention``'s layout (``in_proj_weight`` /
    ``in_proj_bias`` / ``out_proj``), which is what vLLM loads.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        q, k, v = (
            F.linear(x, self.in_proj_weight, self.in_proj_bias)
            .view(bsz, seq_len, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out_proj(out.transpose(1, 2).reshape(bsz, seq_len, hidden))


class LiLiCorrLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.attn_norm = DFlashRMSNorm(hidden_size, eps=eps)
        self.attn = LiLiCorrAttention(hidden_size, num_heads)
        self.mlp_norm = DFlashRMSNorm(hidden_size, eps=eps)
        self.mlp = LiLiCorrMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.mlp(self.mlp_norm(x))


class LiLiCorrHead(nn.Module):
    """Scores the ``slots x k`` candidate lattice of every block in one pass.

    Each candidate gets unit-norm ``out`` and ``in`` vectors; a transition from
    candidate ``i`` at slot ``s`` to candidate ``j`` at slot ``s+1`` scores
    ``logit_scale * cos(out[s, i], in[s+1, j])``. Slot 0 is scored against an
    ``anchor_out`` vector built from the target context row before the block.
    """

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        candidate_topk: int,
        hidden_size: int | None,
        factor_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        logit_scale: float,
        vector_eps: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        hidden_size = hidden_size or model_hidden_size
        self.block_size = block_size
        self.num_slots = block_size - 1
        self.candidate_topk = candidate_topk
        self.logit_scale = logit_scale
        self.vector_eps = vector_eps

        self.token_proj = (
            nn.Identity()
            if hidden_size == model_hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_norm = nn.LayerNorm(_NUM_CANDIDATE_FEATURES)
        self.feature_mlp = LiLiCorrMLP(
            _NUM_CANDIDATE_FEATURES, hidden_size, hidden_size
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_slots, 1, hidden_size)
        )
        self.rank_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.zeros(num_heads))
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            LiLiCorrLayer(hidden_size, num_heads, mlp_ratio, rms_norm_eps)
            for _ in range(num_layers)
        )
        self.output_norm = DFlashRMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = DFlashRMSNorm(hidden_size, eps=rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.out_head = nn.Linear(hidden_size, factor_dim)
        self.in_head = nn.Linear(hidden_size, factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, factor_dim)
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02) -> None:
        # Linears get N(0, std) with zero bias. The lattice QKV keeps its xavier
        # init, and the embeddings and attention biases start at zero.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                nn.init.zeros_(module.bias)

    def _attention_bias(self, device: torch.device) -> torch.Tensor:
        slot_ids = torch.arange(self.num_slots, device=device).repeat_interleave(
            self.candidate_topk
        )
        rel = slot_ids[:, None] - slot_ids[None, :]
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        bias = bias + (rel == 0) * self.same_slot_bias[:, None, None]
        return bias.unsqueeze(0)  # [1, heads, S*K, S*K], broadcast over blocks

    def forward(
        self,
        token_embeddings: torch.Tensor,  # [N, S, K, H_model]
        candidate_log_probs: torch.Tensor,  # [N, S, K], descending rank order
        pass_hidden: torch.Tensor,  # [N, S, H_model]
        anchor_hidden: torch.Tensor,  # [N, H_model]
        anchor_valid: torch.Tensor,  # [N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_start [N, K], log_pair [N, S-1, K, K])`` in fp32."""
        num_blocks, num_slots, topk = candidate_log_probs.shape

        log_probs = candidate_log_probs.float()
        rank = torch.arange(topk, device=log_probs.device) / max(topk - 1, 1)
        features = torch.stack(
            [
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.amax(-1, keepdim=True),
                rank.expand_as(log_probs),
                (rank == 0).float().expand_as(log_probs),
            ],
            dim=-1,
        )

        x = self.token_proj(token_embeddings) + self.pass_hidden_proj(
            pass_hidden
        ).unsqueeze(-2)
        x = x + self.feature_mlp(self.feature_norm(features.to(x.dtype)))
        x = x + self.slot_embedding[0] + self.rank_embedding[0]
        x = x.reshape(num_blocks, num_slots * topk, -1)

        bias = self._attention_bias(x.device)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.output_norm(x).view(num_blocks, num_slots, topk, -1)

        # A zeroed input still projects to context_proj's bias; zero the output too.
        anchor = self.context_proj(anchor_hidden) * anchor_valid[:, None].to(x.dtype)
        anchor = self.anchor_norm(anchor)
        a = anchor[:, None, None, :].expand_as(x)
        factors = F.silu(self.factor_input_proj(torch.cat([x, a, x * a], dim=-1)))

        out_vec = F.normalize(self.out_head(factors), dim=-1, eps=self.vector_eps)
        in_vec = F.normalize(self.in_head(factors), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(
            self.anchor_out_head(anchor), dim=-1, eps=self.vector_eps
        )

        start = (anchor_out[:, None, :] * in_vec[:, 0]).sum(-1)
        pair = out_vec[:, :-1] @ in_vec[:, 1:].transpose(-1, -2)
        return self.logit_scale * start.float(), self.logit_scale * pair.float()


class LiLiCorrDraftModel(DFlashDraftModel):
    """DFlash backbone plus the LiLiCorr candidate-lattice head."""

    config_class = LiLiCorrConfig

    def __init__(self, config: LiLiCorrConfig):
        super().__init__(config=config)
        if getattr(config, "draft_vocab_size", None) not in (None, config.vocab_size):
            raise ValueError(
                "LiLiCorr requires the full verifier vocabulary (vLLM cannot serve "
                "a reduced draft vocab); omit draft_vocab_size."
            )
        if config.sample_from_anchor:
            raise ValueError("LiLiCorr requires sample_from_anchor=False.")
        topk = config.lilicorr_candidate_topk
        if topk < 1 or topk > 16 or topk & (topk - 1):
            raise ValueError("lilicorr_candidate_topk must be a power of two <= 16.")
        self.lilicorr = LiLiCorrHead(
            model_hidden_size=config.hidden_size,
            block_size=config.block_size,
            candidate_topk=topk,
            hidden_size=config.lilicorr_hidden_size,
            factor_dim=config.lilicorr_factor_dim,
            num_layers=config.lilicorr_num_layers,
            num_heads=config.lilicorr_num_heads,
            mlp_ratio=config.lilicorr_mlp_ratio,
            logit_scale=config.lilicorr_logit_scale,
            vector_eps=config.lilicorr_vector_eps,
            rms_norm_eps=config.rms_norm_eps,
        )


@torch.no_grad()
def _accepted_length_counts(
    correct: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (accepted_length_sum, valid_block_count), vLLM's +1 bonus convention."""
    accepted = torch.logical_and(correct, valid).to(torch.float32)
    per_block_len = accepted.cumprod(dim=-1).sum(dim=-1) + 1.0
    block_valid = valid.any(dim=-1).to(torch.float32)
    return (per_block_len * block_valid).sum(), block_valid.sum()


def compute_lattice_terms(
    *,
    head: LiLiCorrHead,
    embed_tokens: nn.Module,
    lm_head_weight: torch.Tensor,
    block_size: int,
    draft_hidden: torch.Tensor,  # [bsz, nb, block, H] draft trunk output
    target_hidden: torch.Tensor,  # [bsz, nb, block, H] target final hidden (normed)
    anchor_hidden: torch.Tensor,  # [bsz, nb, H] context row before each anchor
    anchor_valid: torch.Tensor,  # [bsz, nb] bool
    valid_block: torch.Tensor,  # [bsz, nb] bool
    slot_mask: torch.Tensor,  # [bsz, nb, block] bool
    w_ce: float = 0.25,
    w_pen: float = 0.25,
    chunk_blocks: int = 32,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-forced LiLiCorr lattice terms over a whole batch of blocks.

    Mirrors speculators' ``LiLiCorrDraftModel._lattice_terms``: candidates come
    from the draft logits' top-K, ground truth from the target logits' argmax,
    candidate log-probs are recomputed online so the backbone gets gradient, and
    only slots whose ground truth is in the lattice with every earlier slot hit
    are supervised. Chunked over blocks because ``[blocks*block, vocab]`` logits
    are materialized.
    """
    topk = head.candidate_topk
    assert block_size == head.block_size
    bsz, nb, block, _ = draft_hidden.shape
    device = draft_hidden.device

    ce_num = torch.zeros((), dtype=torch.float32, device=device)
    pen_num = torch.zeros((), dtype=torch.float32, device=device)
    sup_count = torch.zeros((), dtype=torch.float32, device=device)
    eal_sum = torch.zeros((), dtype=torch.float32, device=device)
    eal_total = torch.zeros((), dtype=torch.float32, device=device)
    top1_sum = torch.zeros((), dtype=torch.float32, device=device)
    top1_total = torch.zeros((), dtype=torch.float32, device=device)
    orac_sum = torch.zeros((), dtype=torch.float32, device=device)
    orac_total = torch.zeros((), dtype=torch.float32, device=device)

    for b0 in range(0, nb, max(int(chunk_blocks), 1)):
        b1 = min(b0 + max(int(chunk_blocks), 1), nb)
        cb = b1 - b0
        dh = draft_hidden[:, b0:b1]  # [bsz, cb, block, H]
        th = target_hidden[:, b0:b1]
        logits = F.linear(dh, lm_head_weight)  # [bsz, cb, block, V]
        target_logits = F.linear(th, lm_head_weight)

        n = bsz * cb
        logits_s = logits.reshape(n, block, -1)[:, 1:]  # [n, S, V]
        tgt_s = target_logits.reshape(n, block, -1)[:, 1:]
        hidden_s = dh.reshape(n, block, -1)[:, 1:]  # [n, S, H]
        smask = slot_mask[:, b0:b1].reshape(n, block)[:, 1:].bool()

        with torch.no_grad():
            candidate_ids = logits_s.topk(topk, dim=-1).indices  # [n, S, K]
            gt_ids = tgt_s.argmax(-1)  # [n, S]
            gt_hits = candidate_ids == gt_ids.unsqueeze(-1)
            gt_in_lattice = gt_hits.any(-1)
            gt_index = gt_hits.float().argmax(-1)  # [n, S]

        candidate_log_probs = logits_s.gather(
            -1, candidate_ids
        ).float() - torch.logsumexp(logits_s.float(), dim=-1, keepdim=True)

        vblock = valid_block[:, b0:b1].reshape(n)
        log_start, log_pair = head(
            embed_tokens(candidate_ids).detach(),
            candidate_log_probs,
            hidden_s,
            anchor_hidden[:, b0:b1].reshape(n, -1),
            anchor_valid[:, b0:b1].reshape(n),
        )

        prev_gt = gt_index[:, :-1, None, None].expand(-1, -1, 1, topk)
        nodes = torch.cat(
            [log_start.unsqueeze(1), log_pair.gather(2, prev_gt).squeeze(2)], dim=1
        )  # [n, S, K]
        log_p = F.log_softmax(nodes, dim=-1)

        supervised = (gt_in_lattice.long().cumprod(-1).bool() & vblock[:, None]).float()
        ce = -log_p.gather(-1, gt_index.unsqueeze(-1)).squeeze(-1)
        ce_num = ce_num + (ce * supervised).sum()

        cand_target_logits = tgt_s.gather(-1, candidate_ids).float()
        gt_target_logit = cand_target_logits.gather(-1, gt_index.unsqueeze(-1))
        rejection = (gt_target_logit - cand_target_logits).clamp_min(0.0)
        pen_num = pen_num + ((log_p.exp() * rejection).sum(-1) * supervised).sum()
        sup_count = sup_count + supervised.sum()

        with torch.no_grad():
            current = log_start.argmax(-1)
            picks = [current]
            for slot in range(block - 2):
                row = log_pair[:, slot].gather(
                    1, current[:, None, None].expand(-1, 1, topk)
                )
                current = row.squeeze(1).argmax(-1)
                picks.append(current)
            selected_ids = candidate_ids.gather(
                -1, torch.stack(picks, dim=1).unsqueeze(-1)
            ).squeeze(-1)
            eval_mask = smask & vblock[:, None]
            for correct, kind in (
                (selected_ids == gt_ids, "eal"),
                (candidate_ids[..., 0] == gt_ids, "top1"),
                (gt_in_lattice, "oracle"),
            ):
                s, t = _accepted_length_counts(correct, eval_mask)
                if kind == "eal":
                    eal_sum, eal_total = eal_sum + s, eal_total + t
                elif kind == "top1":
                    top1_sum, top1_total = top1_sum + s, top1_total + t
                else:
                    orac_sum, orac_total = orac_sum + s, orac_total + t

    denom = sup_count.clamp_min(1.0)
    ce_loss = ce_num / denom
    pen_loss = pen_num / denom
    lattice_loss = w_ce * ce_loss + w_pen * pen_loss

    metrics = {
        "lilicorr_ce_sum": ce_loss.detach(),
        "lilicorr_ce_total": torch.ones((), dtype=torch.float32, device=device),
        "lilicorr_penalty_sum": pen_loss.detach(),
        "lilicorr_penalty_total": torch.ones((), dtype=torch.float32, device=device),
        "lilicorr_supervised_token_count": sup_count.detach(),
        "lilicorr_eal_sum": eal_sum,
        "lilicorr_eal_total": eal_total,
        "lilicorr_top1_eal_sum": top1_sum,
        "lilicorr_top1_eal_total": top1_total,
        "lilicorr_oracle_eal_sum": orac_sum,
        "lilicorr_oracle_eal_total": orac_total,
    }
    return lattice_loss, metrics
