"""P-EAGLE (parallel-drafting) draft model.

Logic ported from NeMo AutoModel's P-EAGLE (``peagle_draft.py`` mixins on the
EAGLE-3 draft, itself a port of speculators PR #480). Unlike EAGLE-3's sequential
``cache_hidden`` TTT recurrence, P-EAGLE flattens all COD depths into one sequence
and attends in a single ``flex_attention`` pass with the COD block mask: the depth
is baked into ``position_ids = anchor_pos + depth``, and cross-depth visibility is
enforced entirely by the mask. Layer 0 fuses ``[embed, hidden]`` (2H); deeper
layers refine plain H. A single learnable ``mask_hidden`` placeholder substitutes
for the target aux feature at every masked (depth>=1) slot.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from verl_speco.models.eagle.base import DraftModel
from verl_speco.models.peagle.configuration_peagle import PeagleConfig
from verl_speco.models.peagle.peagle_mask import create_peagle_mask_mod

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_flex_attention_compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True)


def _flex_supported(q: torch.Tensor) -> bool:
    # Inductor's flex lowering needs CUDA and head_dim >= 16; eager fallback keeps
    # CPU / small-head unit tests runnable.
    return q.is_cuda and q.shape[-1] >= 16


def _run_flex_attention(q, k, v, *, block_mask, scale):
    flex = _flex_attention_compiled if _flex_supported(q) else flex_attention
    return flex(q, k, v, block_mask=block_mask, scale=scale)


class PeagleAttention(nn.Module):
    """GQA self attention for the P-EAGLE draft (flex-attention, COD block mask)."""

    def __init__(self, config: PeagleConfig, fuse_input: bool):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        in_features = config.hidden_size * 2 if fuse_input else config.hidden_size

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def _repeat_kv(self, k, v):
        if self.num_key_value_groups == 1:
            return k, v
        return (
            k.repeat_interleave(self.num_key_value_groups, dim=1),
            v.repeat_interleave(self.num_key_value_groups, dim=1),
        )

    def forward_peagle(self, combined_states: torch.Tensor, position_ids: torch.Tensor, block_mask) -> torch.Tensor:
        batch_size, seq_len, _ = combined_states.shape
        q = self.q_proj(combined_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(combined_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(combined_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(combined_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self._repeat_kv(k, v)
        attn_output = _run_flex_attention(q, k, v, block_mask=block_mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class PeagleMLP(nn.Module):
    def __init__(self, config: PeagleConfig):
        super().__init__()
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PeagleFusedLayer(nn.Module):
    """Layer 0: fuses ``[input_layernorm(embed), hidden_norm(hidden)]`` (2H)."""

    def __init__(self, config: PeagleConfig):
        super().__init__()
        self.self_attn = PeagleAttention(config, fuse_input=True)
        self.mlp = PeagleMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_peagle(self, input_embeds, hidden_states, position_ids, block_mask) -> torch.Tensor:
        residual = hidden_states
        combined = torch.cat((self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)), dim=-1)
        hidden_states = residual + self.self_attn.forward_peagle(combined, position_ids, block_mask)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class PeagleVanillaLayer(nn.Module):
    """Deeper layers: standard pre-norm Llama block over plain H hidden states."""

    def __init__(self, config: PeagleConfig):
        super().__init__()
        self.self_attn = PeagleAttention(config, fuse_input=False)
        self.mlp = PeagleMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_peagle(self, hidden_states, position_ids, block_mask) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn.forward_peagle(hidden_states, position_ids, block_mask)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class LlamaForCausalLMPeagle(DraftModel):
    """Parallel-drafting EAGLE draft that predicts all COD depths in one forward."""

    config_class = PeagleConfig
    _no_split_modules = ["PeagleFusedLayer", "PeagleVanillaLayer"]

    def __init__(self, config: PeagleConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.num_aux_hidden_states = int(getattr(config, "num_aux_hidden_states", 3))
        self.draft_vocab_size = int(getattr(config, "draft_vocab_size", config.vocab_size))

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.fc = nn.Linear(self.target_hidden_size * self.num_aux_hidden_states, config.hidden_size, bias=False)
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [LlamaRMSNorm(self.target_hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_aux_hidden_states)]
            )
        else:
            self.fc_norm = None

        num_layers = max(1, int(getattr(config, "num_draft_layers", 4)))
        layers = [PeagleFusedLayer(config)]
        layers += [PeagleVanillaLayer(config) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # Learnable placeholder for masked (depth>=1) aux slots, at the pre-fc
        # concatenated-aux width so it flows through project_hidden_states like a
        # real aux vector. Unit-variance init matches speculators.
        self.mask_hidden = nn.Parameter(torch.empty(1, 1, self.fc.in_features))

        self.post_init()

        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        t2d[: self.draft_vocab_size] = True
        d2t = torch.arange(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        nn.init.normal_(self.mask_hidden, mean=0.0, std=1.0)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat([norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)], dim=-1)
        return self.fc(hidden_states)

    def masked_projected_hidden(self) -> torch.Tensor:
        return self.project_hidden_states(self.mask_hidden.view(1, -1))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def selected_token_ids(self) -> torch.Tensor:
        return torch.nonzero(self.t2d, as_tuple=False).flatten()

    def build_peagle_block_mask(self, anchor_pos, depth, lengths, total_seq_len):
        mask_mod = create_peagle_mask_mod(anchor_pos=anchor_pos, depth=depth, lengths=lengths, total_seq_len=total_seq_len)
        return create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=anchor_pos.shape[0], KV_LEN=anchor_pos.shape[0], device=anchor_pos.device
        )

    def forward_peagle(self, sampled_input_ids, sampled_projected_hidden, position_ids, block_mask) -> torch.Tensor:
        draft_input_embeds = self.embed_tokens(sampled_input_ids).to(sampled_projected_hidden.dtype)
        hidden_states = self.layers[0].forward_peagle(
            input_embeds=draft_input_embeds,
            hidden_states=sampled_projected_hidden,
            position_ids=position_ids,
            block_mask=block_mask,
        )
        for layer in self.layers[1:]:
            hidden_states = layer.forward_peagle(hidden_states, position_ids, block_mask)
        # The final norm is applied in compute_logits (lm_head(norm(h))), matching
        # the EAGLE-3 draft; forward_peagle returns the pre-norm hidden states.
        return hidden_states


__all__ = ["PeagleConfig", "LlamaForCausalLMPeagle"]
