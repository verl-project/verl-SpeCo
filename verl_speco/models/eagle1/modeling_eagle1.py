"""Llama-style dense draft model for EAGLE-1 / EAGLE-2 training.

Logic ported from NeMo AutoModel's EAGLE-1/2 ``LlamaEagleDraftModel``
(``draft_llama_v12.py``): a single ``fc`` layer fuses the token embedding with
the target's last-layer hidden state once, followed by one (configurable)
standard Llama decoder layer and a final norm. The draft predicts the target's
*next-step* hidden state; token logits are produced by the frozen target
``lm_head`` (weight tying), so the draft itself carries no ``lm_head``.

This differs from the EAGLE-3 draft (``llama_eagle.py``), which fuses multiple
aux hidden states and re-concatenates the embedding at every decoder layer.
"""

import logging
import os

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from verl_speco.models.eagle.base import DraftModel
from verl_speco.models.eagle1.configuration_eagle1 import Eagle1Config

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _build_causal_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build an additive causal + padding mask for eager attention."""
    batch_size, seq_len = attention_mask.shape
    min_value = torch.finfo(dtype).min
    causal = torch.full(
        (seq_len, seq_len), min_value, device=attention_mask.device, dtype=dtype
    )
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value
    return causal + expanded


class EagleLlamaAttention(nn.Module):
    """Standard Llama-style GQA self attention for the EAGLE-1/2 draft."""

    def __init__(self, config: Eagle1Config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=attention_bias
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def _repeat_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return tensor
        return tensor.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.o_proj(attn_output)


class EagleLlamaMLP(nn.Module):
    """Standard SwiGLU MLP used by the EAGLE-1/2 draft."""

    def __init__(self, config: Eagle1Config):
        super().__init__()
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class EagleLlamaDecoderLayer(nn.Module):
    """Single standard (H-dim) decoder layer for the EAGLE-1/2 draft."""

    def __init__(self, config: Eagle1Config):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        self.self_attn = EagleLlamaAttention(config)
        self.mlp = EagleLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaForCausalLMEagle1(DraftModel):
    """Dense EAGLE-1 / EAGLE-2 draft that predicts the target's next hidden state.

    The token embedding and the target last-layer feature are fused once by
    ``fc`` (``hidden_size + target_hidden_size -> hidden_size``), then passed
    through ``draft_num_hidden_layers`` standard decoder layers and a final norm.
    ``forward`` returns the predicted hidden states; token logits are computed by
    the frozen target ``lm_head`` in the trainer backend.
    """

    config_class = Eagle1Config
    _no_split_modules = ["EagleLlamaDecoderLayer"]

    def __init__(self, config: Eagle1Config) -> None:
        super().__init__(config)
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.target_hidden_size = getattr(
            config, "target_hidden_size", config.hidden_size
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.fc = nn.Linear(
            config.hidden_size + self.target_hidden_size, config.hidden_size, bias=False
        )
        num_layers = max(
            1, int(getattr(config, "draft_num_hidden_layers", config.num_hidden_layers))
        )
        self.layers = nn.ModuleList(
            [EagleLlamaDecoderLayer(config) for _ in range(num_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Predict the target next-step hidden states.

        Args:
            input_ids: ``(batch, seq_len)`` draft input token ids (the shifted
                next tokens, per the EAGLE teacher-forcing alignment).
            hidden_states: ``(batch, seq_len, target_hidden_size)`` target
                last-layer feature fed into the ``fc`` fusion.
            attention_mask: ``(batch, seq_len)`` 1/0 padding mask.
            position_ids: ``(batch, seq_len)`` position ids; defaults to
                ``arange`` when not provided.
        """
        inputs_embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)
        fused = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        batch_size, seq_len, _ = fused.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.long, device=fused.device
            )
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=fused.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        causal_mask = _build_causal_mask(attention_mask, fused.dtype)

        for layer in self.layers:
            fused = layer(fused, attention_mask=causal_mask, position_ids=position_ids)
        return self.norm(fused)
