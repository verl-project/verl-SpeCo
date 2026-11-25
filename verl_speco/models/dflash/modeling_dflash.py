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
import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

from ..eagle.base import resolve_embedding_key
from .configuration_dflash import (
    _UNSCALED_ROPE_TYPES,
    DEFAULT_ROPE_THETA,
    DFlashConfig,
    resolve_rope_theta,
)
from .flex_attention import compile_friendly_flex_attention


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        rope_parameters: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.attention_scaling = 1.0
        rope_type = "default"
        if isinstance(rope_parameters, dict):
            rope_type = str(rope_parameters.get("rope_type") or "default")
        if rope_type in _UNSCALED_ROPE_TYPES:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            )
        else:
            # Delegate to transformers so the draft rotary matches the upstream
            # YaRN/linear/llama3 frequencies and the attention-factor scaling.
            from types import SimpleNamespace

            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            if rope_type not in ROPE_INIT_FUNCTIONS:
                raise ValueError(
                    f"Unsupported rope_type={rope_type!r} for the DFlash draft rotary"
                )
            rope_kwargs = dict(rope_parameters)
            rope_kwargs.setdefault("rope_theta", base)
            rope_kwargs["rope_type"] = rope_type
            rope_config = SimpleNamespace(
                rope_parameters=rope_kwargs,
                hidden_size=self.dim,
                head_dim=self.dim,
                num_attention_heads=1,
                max_position_embeddings=max_position_embeddings,
            )
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                rope_config, device="cpu", seq_len=max_position_embeddings
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            max_position_embeddings + 20, self.inv_freq.device, torch.float32
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * self.attention_scaling)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * self.attention_scaling)[None, None, :, :].to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _apply_rotary_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """NeoX-style rotary where the two halves of the last dim are paired."""
    return (x * cos) + (rotate_half(x) * sin)


def _apply_rotary_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """DeepSeek-style rotary where adjacent dims ``(x0, x1), (x2, x3)`` are paired.

    ``cos``/``sin`` are ``cat(freqs, freqs)``; the first half holds the per-pair
    angle. Computed directly on the even/odd slices, matching upstream.
    """
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # Interleave the rotated even/odd pairs back into adjacent channels; a plain
    # ``cat`` would place all even outputs before all odd ones.
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_apply_mscale(rope_parameters: Optional[dict], base_scale: float) -> float:
    """Replicate the upstream DeepSeek-V3 YaRN attention scaling."""
    if not isinstance(rope_parameters, dict):
        return base_scale
    if str(rope_parameters.get("rope_type") or "default") in _UNSCALED_ROPE_TYPES:
        return base_scale
    factor = float(rope_parameters.get("factor", 1.0))
    mscale_all_dim = float(rope_parameters.get("mscale_all_dim", 0.0))
    if mscale_all_dim:
        base_scale = base_scale * _yarn_get_mscale(factor, mscale_all_dim) ** 2
    return base_scale


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> list[int]:
    """Select uniformly spaced target layers following the TorchSpec DFlash convention."""
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_target_layers - 1)))
        for i in range(num_target_layers)
    ]


class DFlashAttention(nn.Module):
    """Dual-source KV attention.

    Query comes from draft hidden states. Key/value are concatenated from
    target-context features and draft hidden states, so a draft block can see
    target context before its anchor and its own block tokens under the training
    mask generated by the backend.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DFlashRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            base=resolve_rope_theta(config),
        )

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]

        q = self.q_proj(draft_hidden).view(
            bsz, draft_len, self.num_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(context_hidden)
        v_ctx = self.v_proj(context_hidden)
        k_draft = self.k_proj(draft_hidden)
        v_draft = self.v_proj(draft_hidden)
        k = torch.cat([k_ctx, k_draft], dim=1)
        v = torch.cat([v_ctx, v_draft], dim=1)

        total_len = ctx_len + draft_len
        k = k.view(bsz, total_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        cos, sin = self.rotary_emb(q, seq_len=total_len)
        cos = cos.to(q.device).squeeze(1).squeeze(0)
        sin = sin.to(q.device).squeeze(1).squeeze(0)

        cos_q = cos[draft_position_ids].unsqueeze(1)
        sin_q = sin[draft_position_ids].unsqueeze(1)
        q = (q * cos_q) + (rotate_half(q) * sin_q)

        cos_k = cos[full_position_ids].unsqueeze(1)
        sin_k = sin[full_position_ids].unsqueeze(1)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

        if block_mask is not None:
            if dense_attention_mask is not None:
                raise ValueError(
                    "DFlash attention received both block_mask and dense_attention_mask"
                )
            attn_output = compile_friendly_flex_attention(
                q,
                k.contiguous(),
                v.contiguous(),
                block_mask=block_mask,
                enable_gqa=True,
            )
        else:
            k = repeat_kv(k, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=dense_attention_mask,
                is_causal=False,
                dropout_p=0.0,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, draft_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class DFlashMLAAttention(nn.Module):
    """Dual-source KV attention with DeepSeek-style multi-head latent attention.

    GLM-5 / DeepSeek targets compress Q and KV into low-rank latents. The draft
    expands the KV latent back to every query head, applies rotary only to the
    rope channels, and keeps the same context/draft KV concatenation and block
    mask protocol as :class:`DFlashAttention`.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.rope_interleave = bool(getattr(config, "rope_interleave", True))
        rope_parameters = getattr(config, "rope_parameters", None)
        self.rope_parameters = (
            dict(rope_parameters) if isinstance(rope_parameters, dict) else None
        )
        if self.v_head_dim > self.qk_head_dim:
            raise ValueError(
                "v_head_dim must not exceed qk_head_dim, got "
                f"{self.v_head_dim} > {self.qk_head_dim}"
            )

        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = DFlashRMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = DFlashRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        base = getattr(config, "rope_theta", None)
        if base is None and self.rope_parameters is not None:
            base = self.rope_parameters.get("rope_theta")
        base = float(base) if base is not None else DEFAULT_ROPE_THETA
        self.rotary_emb = DFlashRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            base=base,
            rope_parameters=self.rope_parameters,
        )
        self.scaling = _yarn_apply_mscale(self.rope_parameters, self.qk_head_dim**-0.5)

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if self.rope_interleave:
            return _apply_rotary_interleaved(x, cos, sin)
        return _apply_rotary_neox(x, cos, sin)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]
        total_len = ctx_len + draft_len

        if self.q_lora_rank:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(draft_hidden)))
        else:
            q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.qk_head_dim)
        q_nope, q_rot = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_rot = q_rot.transpose(1, 2)

        kv = self.kv_a_proj_with_mqa(torch.cat([context_hidden, draft_hidden], dim=1))
        kv_lora, k_rot = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_lora))
        kv = kv.view(
            bsz,
            total_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)
        k_rot = k_rot.view(bsz, total_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)

        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        cos, sin = self.rotary_emb(q_rot, seq_len=total_len)
        cos = cos.to(q_rot.device).squeeze(1).squeeze(0)
        sin = sin.to(q_rot.device).squeeze(1).squeeze(0)

        cos_q = cos[draft_position_ids].unsqueeze(1)
        sin_q = sin[draft_position_ids].unsqueeze(1)
        q_rot = self._apply_rotary(q_rot, cos_q, sin_q)

        cos_k = cos[full_position_ids].unsqueeze(1)
        sin_k = sin[full_position_ids].unsqueeze(1)
        k_rot = self._apply_rotary(k_rot, cos_k, sin_k)

        q = torch.cat([q_nope, q_rot], dim=-1)
        k = torch.cat([k_nope, k_rot], dim=-1)
        if self.v_head_dim < self.qk_head_dim:
            v = F.pad(v, (0, self.qk_head_dim - self.v_head_dim))

        if block_mask is not None:
            if dense_attention_mask is not None:
                raise ValueError(
                    "DFlash MLA attention received both block_mask and "
                    "dense_attention_mask"
                )
            attn_output = compile_friendly_flex_attention(
                q,
                k.contiguous(),
                v.contiguous(),
                block_mask=block_mask,
                enable_gqa=False,
                scale=self.scaling,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=dense_attention_mask,
                is_causal=False,
                dropout_p=0.0,
                scale=self.scaling,
            )

        attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, draft_len, self.num_heads * self.v_head_dim
        )
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    attention_class = DFlashAttention

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.self_attn = self.attention_class(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DFlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Optional dynamic-convolution hooks, populated only by DFlash2. Every
        # other DFlash-family drafter leaves them None, so their forward is
        # unchanged.
        self.attention_conv = None
        self.mlp_conv = None

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)
        attention_kernel = None
        if self.attention_conv is not None:
            draft_hidden, attention_kernel = self.attention_conv.prepare(draft_hidden)
        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            dense_attention_mask=dense_attention_mask,
        )
        if attention_kernel is not None:
            draft_hidden = self.attention_conv.finish(draft_hidden, attention_kernel)
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        mlp_kernel = None
        if self.mlp_conv is not None:
            draft_hidden, mlp_kernel = self.mlp_conv.prepare(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        if mlp_kernel is not None:
            draft_hidden = self.mlp_conv.finish(draft_hidden, mlp_kernel)
        return residual + draft_hidden


class DFlashMLADecoderLayer(DFlashDecoderLayer):
    """DFlash decoder layer whose attention expands a latent KV (MLA)."""

    attention_class = DFlashMLAAttention


class DFlashDraftModel(PreTrainedModel):
    """DFlash draft model with target-context feature injection."""

    config_class = DFlashConfig
    _no_split_modules = ["DFlashDecoderLayer", "DFlashMLADecoderLayer"]
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: list[str] = []

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        target_num_hidden = int(
            getattr(
                config,
                "target_num_hidden_layers",
                getattr(config, "num_target_layers", 36),
            )
        )
        self.num_target_layers = int(
            getattr(config, "num_target_layers", target_num_hidden)
        )
        self.target_hidden_size = getattr(
            config, "target_hidden_size", config.hidden_size
        )
        self.mask_token_id = getattr(config, "mask_token_id", config.vocab_size - 1)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        self.num_context_layers = getattr(config, "num_context_layers", None)
        if self.target_layer_ids is not None:
            self.target_layer_ids = [
                int(layer_id) for layer_id in self.target_layer_ids
            ]
            if self.num_context_layers is None:
                self.num_context_layers = len(self.target_layer_ids)
        else:
            if self.num_context_layers is None:
                self.num_context_layers = (
                    self.num_target_layers
                    if self.num_target_layers != target_num_hidden
                    else 5
                )
            self.target_layer_ids = build_target_layer_ids(
                int(self.num_context_layers), target_num_hidden
            )
        self.num_context_layers = int(self.num_context_layers)
        if len(self.target_layer_ids) != self.num_context_layers:
            raise ValueError(
                f"DFlash expected {self.num_context_layers} target layer ids, got {len(self.target_layer_ids)}"
            )
        self.fc = nn.Linear(
            self.num_context_layers * self.target_hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.hidden_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = self._build_decoder_layers(config)
        self.norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_decoder_layers(self, config: PretrainedConfig) -> nn.ModuleList:
        """Build the decoder stack; subclasses may swap in another attention."""
        return nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(self.num_layers)]
        )

    def extract_context_feature(
        self, all_hidden_states: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(all_hidden_states) != self.num_context_layers:
            raise ValueError(
                f"DFlash expected {self.num_context_layers} hidden-state tensors, got {len(all_hidden_states)}"
            )
        concatenated = torch.cat(all_hidden_states, dim=-1).to(self.fc.weight.dtype)
        return self.hidden_norm(self.fc(concatenated))

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        block_masks = block_mask if isinstance(block_mask, list) else None
        dense_masks = (
            dense_attention_mask if isinstance(dense_attention_mask, list) else None
        )
        for index, layer in enumerate(self.layers):
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_masks[index]
                if block_masks is not None
                else block_mask,
                dense_attention_mask=(
                    dense_masks[index]
                    if dense_masks is not None
                    else dense_attention_mask
                ),
            )
        return self.norm(draft_hidden)

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path)
        # Multimodal / MoE targets nest the text embedding under a different key.
        embedding_key = resolve_embedding_key(model_path, embedding_key)

        import glob as glob_mod

        index_json_path = glob_mod.glob(os.path.join(model_path, "*.index.json"))
        if len(index_json_path) == 0:
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                with safe_open(safetensors_path, framework="pt") as f:
                    self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
                return
            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                state_dict = torch.load(
                    pytorch_model_path, map_location="cpu", weights_only=True
                )
                self.embed_tokens.weight.copy_(state_dict[embedding_key])
                return
            raise FileNotFoundError(
                f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
            )
        if len(index_json_path) > 1:
            raise FileNotFoundError(f"Multiple index.json files found in {model_path}")

        with open(index_json_path[0], "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][embedding_key]
        if ckpt_file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
        else:
            state_dict = torch.load(
                os.path.join(model_path, ckpt_file),
                map_location="cpu",
                weights_only=True,
            )
            self.embed_tokens.weight.copy_(state_dict[embedding_key])
