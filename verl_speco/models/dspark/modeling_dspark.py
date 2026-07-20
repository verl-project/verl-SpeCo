from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from verl_speco.models.dflash import DFlashDraftModel

from .configuration_dspark import DSparkConfig


class DSparkForwardOutput:
    def __init__(
        self,
        *,
        draft_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        block_keep_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        confidence_pred: Optional[torch.Tensor] = None,
    ):
        self.draft_hidden = draft_hidden
        self.base_logits = base_logits
        self.draft_logits = draft_logits
        self.target_ids = target_ids
        self.prev_token_ids = prev_token_ids
        self.eval_mask = eval_mask
        self.block_keep_mask = block_keep_mask
        self.anchor_positions = anchor_positions
        self.confidence_pred = confidence_pred


class DSparkVanillaMarkovHead(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"markov_rank must be positive for vanilla Markov head, got {markov_rank}"
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del hidden_states
        return self.markov_w2(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        return base_logits + self.compute_step_bias(token_ids)


class DSparkGatedMarkovHead(DSparkVanillaMarkovHead):
    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("gated DSpark Markov head requires hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([hidden_states, prev_embeddings], dim=-1))
        )
        return self.markov_w2(gate.to(prev_embeddings.dtype) * prev_embeddings)


class DSparkRNNMarkovHead(DSparkVanillaMarkovHead):
    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("rnn DSpark Markov head requires hidden_states")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for pos in range(block_size):
            prev_embeddings = self.get_prev_embeddings(token_ids[..., pos])
            joint = torch.cat(
                [state, prev_embeddings, hidden_states[..., pos, :]], dim=-1
            )
            gate_raw, candidate_raw, output_raw = self.joint_proj(joint).chunk(
                3, dim=-1
            )
            gate = torch.sigmoid(gate_raw)
            candidate = torch.tanh(candidate_raw)
            state = gate * state + (1.0 - gate) * candidate
            output_logits.append(
                base_logits[..., pos, :] + self.markov_w2(torch.tanh(output_raw))
            )
        return torch.stack(output_logits, dim=-2)


def build_dspark_markov_head(config: DSparkConfig) -> nn.Module | None:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        return None
    markov_head_type = str(
        getattr(config, "markov_head_type", "vanilla") or "vanilla"
    ).lower()
    if markov_head_type == "vanilla":
        return DSparkVanillaMarkovHead(
            vocab_size=config.vocab_size, markov_rank=markov_rank
        )
    if markov_head_type == "gated":
        return DSparkGatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return DSparkRNNMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise ValueError(f"Unsupported DSpark markov_head_type: {markov_head_type!r}")


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class DSparkDraftModel(DFlashDraftModel):
    config_class = DSparkConfig

    def __init__(self, config: DSparkConfig):
        super().__init__(config)
        self.markov_head = build_dspark_markov_head(config)
        self.enable_confidence_head = bool(
            getattr(config, "enable_confidence_head", False)
        )
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )
        if (
            self.enable_confidence_head
            and self.confidence_head_with_markov
            and self.markov_head is None
        ):
            raise ValueError("confidence_head_with_markov requires markov_rank > 0")
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += int(getattr(config, "markov_rank", 0))
            self.confidence_head = DSparkConfidenceHead(input_dim)

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.markov_head is None:
            return base_logits
        return self.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=draft_hidden,
        )

    def predict_confidence(
        self,
        draft_hidden: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            if self.markov_head is None:
                return None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=draft_hidden.dtype
            )
            return self.confidence_head(
                torch.cat([draft_hidden, prev_embeddings], dim=-1)
            ).float()
        return self.confidence_head(draft_hidden).float()


__all__ = [
    "DSparkConfig",
    "DSparkDraftModel",
    "DSparkForwardOutput",
    "DSparkVanillaMarkovHead",
    "DSparkGatedMarkovHead",
    "DSparkRNNMarkovHead",
    "DSparkConfidenceHead",
    "build_dspark_markov_head",
]
