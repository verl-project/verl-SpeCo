# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Latent attention with a learnable attention sink.

``sink_block_attention`` is the numerical core (single-shared-KV block
attention with a sink row), dispatched through the kernel registry. It is the
draft-side replacement for the target model's compressed/sparse attention.
"""

from __future__ import annotations

import torch
from torch import nn

from .kernels import get_kernel, torch_kernel
from .norm import RMSNorm, UnweightedRMSNorm
from .rotary import apply_rotary_emb

_SINK_OP = "sink_block_attention"


@torch_kernel(_SINK_OP)
def _sink_block_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference sink-block attention.

    Shapes: ``q [N, Sq, H, D]``, ``k/v [N, Sk, D]`` (a single shared KV head),
    ``sink [H]``, ``attn_bias [N, Sq, Sk]``. The sink participates as one extra
    logit per query row so no position is ever fully masked out.
    """
    if k.dim() != 3:
        raise ValueError(
            "shared-KV sink attention expects k/v [N, Sk, D] (single head); "
            f"got {tuple(k.shape)}"
        )
    s = torch.einsum("nqhd,nkd->nqhk", q.float(), k.float()) * scale
    if attn_bias is not None:
        s = s + attn_bias.float().unsqueeze(2)
    sink_h = sink.float().reshape(1, 1, -1, 1)
    row_max = torch.maximum(s.max(dim=-1, keepdim=True).values, sink_h)
    e = torch.exp(s - row_max)
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink_h - row_max)
    p = e / denom
    return torch.einsum("nqhk,nkd->nqhd", p, v.float()).to(q.dtype)


def sink_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Dispatch sink-block attention through the kernel registry."""
    return get_kernel(_SINK_OP, backend)(q, k, v, sink, scale, attn_bias)


class LatentAttention(nn.Module):
    """MLA-style attention: low-rank q/o projections, a single shared KV head,
    partial RoPE and a learnable attention sink.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.scale = cfg.head_dim**-0.5

        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.num_heads * cfg.head_dim, bias=False)
        self.q_head_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        self.wkv = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.wo_a = nn.Linear(
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
            cfg.o_groups * cfg.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False
        )
        self.attn_sink = nn.Parameter(torch.zeros(cfg.num_heads))

    def project_q(self, block_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        rd = self.rope_head_dim
        q = self.q_norm(self.wq_a(block_x))
        q = self.wq_b(q).unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_head_norm(q)
        rope = apply_rotary_emb(q[..., -rd:], freqs_cis)
        return torch.cat([q[..., :-rd], rope], dim=-1)

    def project_kv(self, kv_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(kv_x))
        rope = apply_rotary_emb(kv[..., -rd:], freqs_cis)
        return torch.cat([kv[..., :-rd], rope], dim=-1)

    def forward(
        self,
        block_x: torch.Tensor,
        context_x: torch.Tensor,
        block_freqs: torch.Tensor,
        context_freqs: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.project_q(block_x, block_freqs)
        kv_ctx = self.project_kv(context_x, context_freqs)
        kv_blk = self.project_kv(block_x, block_freqs)
        kv = torch.cat([kv_ctx, kv_blk], dim=1)
        o = sink_block_attention(q, kv, kv, self.attn_sink, self.scale, attn_bias)
        return self.combine_output(o, block_freqs)

    def combine_output(
        self, o: torch.Tensor, q_freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        rd = self.rope_head_dim
        derot = apply_rotary_emb(o[..., -rd:], q_freqs_cis, inverse=True)
        o = torch.cat([o[..., :-rd], derot], dim=-1)
        n, sq = o.shape[0], o.shape[1]
        o = o.reshape(n, sq, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.cfg.o_lora_rank, -1)
        o = torch.einsum("nsgd,grd->nsgr", o, wo_a)
        return self.wo_b(o.flatten(2))
