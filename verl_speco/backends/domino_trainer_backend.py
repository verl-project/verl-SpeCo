"""Domino drafter training backend.

Logic follows NeMo AutoModel's Domino training wrapper (``dflash/domino_core.py``):
the DFlash parallel block backbone drafts a whole block in one non-causal forward,
and a lightweight *causal* correction head fixes each position's blindness to the
block's earlier (drafted) tokens. A single-layer GRU encodes a causal state from
the block's previous tokens, and a low-rank projection of ``[backbone hidden | GRU
state]`` produces a full-vocabulary logit delta that is added to the parallel base
logits. Training jointly supervises the refined (final) and backbone-only (base)
logits with a base-anchor curriculum ``loss = (1-lambda)*final + lambda*base``,
``lambda`` decaying to 0.

Domino reuses the DFlash block-drafter plumbing (anchor sampling, noise block,
block attention mask, shifted labels), so ``DominoTrainerBackend`` subclasses
``DFlashTrainerBackend`` exactly like DSpark does; only ``build_model`` and the
training forward differ. The shifted-label alignment (target ``x[a+1:a+1+block]``,
prev ``[x[a], labels[:-1]]``, every position supervised) is the DSpark alignment,
which equals AutoModel's ``shift_label=True`` Domino path.

Domino is not an engine-level speculative algorithm: engines expose it as a
``projector_type`` sub-mode of DFlash, so the serve method stays ``dflash`` and the
correction head is enabled from the checkpoint's ``dflash_config.projector_type``.
``DOMINO`` is therefore never a valid engine algorithm string; see ``vllm_runtime``
and ``sglang_runtime`` for the serve-time guardrails that point at ``DFLASH``.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainerBackend,
    DFlashTrainingModel,
    _create_dflash_mask_mod,
)
from verl_speco.models.dflash.flex_attention import compile_friendly_create_block_mask
from verl_speco.models.domino import DominoConfig, DominoDraftModel
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def get_lambda_base(step: int, decay_steps: int, lambda_start: float) -> float:
    """Base-anchor curriculum weight: linearly decays lambda_start -> 0 over decay_steps."""
    decay_steps = max(1, int(decay_steps))
    progress = min(max(step, 0) / decay_steps, 1.0)
    return max(0.0, min(1.0, lambda_start * (1.0 - progress)))


class DominoTrainingModel(DFlashTrainingModel):
    """Training wrapper around DominoDraftModel (DFlash backbone + causal head)."""

    def __init__(
        self,
        draft_model: DominoDraftModel,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        pure_draft_prefix_len: int = 1,
        lambda_base_start: float = 1.0,
        lambda_base_decay_steps: int = 2000,
    ):
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            front_position_weight=1.0,
            front_position_count=0,
            loss_mode="full_vocab",
            sampled_ce_negatives=0,
        )
        if getattr(draft_model, "projector_type", None) != "domino":
            raise ValueError(
                "DominoTrainingModel requires a draft model built with projector_type='domino' "
                "(so prefix_gru / embed_proj exist)."
            )
        self.pure_draft_prefix_len = int(pure_draft_prefix_len)
        self.lambda_base_start = float(lambda_base_start)
        self.lambda_base_decay_steps = int(lambda_base_decay_steps)
        self._forward_count = 0

    @property
    def _suffix_start(self) -> int:
        # Shifted labels => position 0 is a real next-token prediction; pure_draft_prefix_len
        # leading positions stay backbone-only (uncorrected).
        return int(self.pure_draft_prefix_len)

    def _current_lambda_base(self) -> float:
        return get_lambda_base(
            self._forward_count, self.lambda_base_decay_steps, self.lambda_base_start
        )

    # --- shifted-label anchor sampling / label building (DSpark alignment) ---
    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ):
        bsz = loss_mask.shape[0]
        num_candidates = max(seq_len - 1, 0)
        if num_candidates <= 0:
            anchors = torch.zeros(
                bsz, self.num_anchors, dtype=torch.long, device=device
            )
            keep_mask = torch.zeros(
                bsz, self.num_anchors, dtype=torch.bool, device=device
            )
            return anchors, keep_mask

        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        indices = (
            self._cached_arange("domino_anchor_indices", num_candidates, device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        masked_indices = torch.where(valid, indices, seq_len + 1)
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)
        take_n = min(self.num_anchors, num_candidates)
        _, top_idx = torch.topk(
            random_vals, k=take_n, dim=1, largest=False, sorted=False
        )
        selected = torch.gather(masked_indices, 1, top_idx).sort(dim=1).values
        if take_n < self.num_anchors:
            selected = torch.cat(
                [
                    selected,
                    torch.zeros(
                        bsz, self.num_anchors - take_n, dtype=torch.long, device=device
                    ),
                ],
                dim=1,
            )
        keep_mask = self._cached_arange(
            "domino_anchor_keep", self.num_anchors, device
        ).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=self.num_anchors)
        return torch.where(keep_mask, selected, 0), keep_mask

    def _build_label_tensors(
        self, *, input_ids, loss_mask, anchor_positions, block_keep_mask
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        n_blocks = anchor_positions.shape[1]
        label_offsets = (
            self._cached_arange(
                "domino_label_offsets", self.block_size, device, view_shape=(1, 1, -1)
            )
            + 1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=max(seq_len - 1, 0))
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_mask = (
            valid_label_mask & (target_loss_mask > 0.5) & block_keep_mask.unsqueeze(-1)
        )
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()

        anchor_token_ids = torch.gather(
            input_ids, 1, anchor_positions.clamp(0, max(seq_len - 1, 0))
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        return target_ids, prev_token_ids, eval_mask, label_indices

    def forward(self, input_ids, hidden_states_list, loss_mask, lm_head_weight):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        self._forward_count += 1
        lambda_base = self._current_lambda_base()

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )
        draft_len = n_blocks * self.block_size

        block_mask = None
        if device.type == "cuda":
            block_mask = compile_friendly_create_block_mask(
                mask_mod=_create_dflash_mask_mod(
                    anchor_positions, block_keep_mask, seq_len, self.block_size
                ),
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=seq_len + draft_len,
                device=device,
            )

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        ).view(bsz, n_blocks, self.block_size, -1)

        target_ids, prev_token_ids, eval_mask, _ = self._build_label_tensors(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )

        weight_mask = eval_mask.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = self._cached_arange(
                "domino_decay_positions", self.block_size, device, view_shape=(1, 1, -1)
            )
            weight_mask = weight_mask * torch.exp(
                -positions.float() / float(self.loss_decay_gamma)
            )

        # Causal GRU state over the block's previous tokens (full block, then gather active).
        block_emb = self.draft_model.embed_tokens(prev_token_ids)  # [bsz, n, block, H]
        gru_out, _ = self.draft_model.prefix_gru(
            block_emb.reshape(bsz * n_blocks, self.block_size, -1)
        )
        gru_out = gru_out.reshape(bsz, n_blocks, self.block_size, -1)

        pos_in_block = (
            self._cached_arange(
                "domino_pos_in_block", self.block_size, device, view_shape=(1, 1, -1)
            )
            .expand(bsz, n_blocks, -1)
            .reshape(-1)
        )
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        active_mask = flat_weights > 0
        active_hidden = draft_hidden.reshape(-1, draft_hidden.size(-1))[active_mask]
        active_gru = gru_out.reshape(-1, gru_out.size(-1))[active_mask]
        active_targets = flat_targets[active_mask]
        active_weights = flat_weights[active_mask]
        active_pos = pos_in_block[active_mask]
        suffix_mask = active_pos >= self._suffix_start

        loss_per_token = torch.zeros_like(flat_weights)
        sanitized_rows = torch.zeros((), dtype=torch.float32, device=device)
        active_final_pred = None
        active_base_pred = None
        active_final_logits = None
        active_top5 = None
        if active_targets.numel() == 0:
            loss = flat_weights.sum() * 0.0
            final_loss = base_loss = loss.detach()
        else:
            # Base logits (backbone-only) over every active row. The Domino head only
            # perturbs suffix positions, so we compute the correction and the final
            # CE on the suffix rows and reuse the base CE elsewhere -- this avoids
            # materializing a second full [num_active, vocab] logits tensor.
            base_logits = F.linear(
                active_hidden, lm_head_weight
            ).float()  # [num_active, vocab]
            base_ce = F.cross_entropy(base_logits, active_targets, reduction="none")
            final_ce = base_ce.clone()
            if bool(suffix_mask.any()):
                suffix_delta = self.draft_model.embed_proj(
                    torch.cat(
                        [active_hidden[suffix_mask], active_gru[suffix_mask]], dim=-1
                    )
                )
                active_final_logits = base_logits[suffix_mask] + suffix_delta.float()
                final_ce[suffix_mask] = F.cross_entropy(
                    active_final_logits, active_targets[suffix_mask], reduction="none"
                )

            finite = torch.isfinite(final_ce) & torch.isfinite(base_ce)
            sanitized_rows = (~finite).sum().to(dtype=torch.float32)
            final_ce = torch.where(finite, final_ce, torch.zeros_like(final_ce))
            base_ce = torch.where(finite, base_ce, torch.zeros_like(base_ce))
            active_loss_weights = active_weights * finite.to(dtype=active_weights.dtype)
            den = active_loss_weights.sum().clamp(min=1e-6)
            final_loss = (final_ce * active_loss_weights).sum() / den
            base_loss = (base_ce * active_loss_weights).sum() / den
            loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
            loss_per_token[active_mask] = final_ce
            with torch.no_grad():
                topk = min(5, base_logits.shape[-1])
                active_base_pred = base_logits.argmax(dim=-1)
                active_final_pred = active_base_pred.clone()
                active_top5 = base_logits.topk(topk, dim=-1).indices
                if active_final_logits is not None:
                    active_final_pred[suffix_mask] = active_final_logits.argmax(dim=-1)
                    active_top5[suffix_mask] = active_final_logits.topk(
                        topk, dim=-1
                    ).indices

        with torch.no_grad():
            flat_eval_mask = eval_mask.reshape(-1)
            binary_eval_mask = flat_eval_mask & (flat_weights > 0)
            correct = torch.zeros_like(flat_weights, dtype=torch.bool)
            base_correct = torch.zeros_like(flat_weights, dtype=torch.bool)
            top1_correct = torch.zeros((), dtype=torch.float32, device=device)
            top5_correct = torch.zeros((), dtype=torch.float32, device=device)
            quality_token_count = torch.zeros((), dtype=torch.float32, device=device)
            if active_final_pred is not None and active_targets.numel() > 0:
                active_correct = active_final_pred.eq(active_targets)
                correct[active_mask] = active_correct
                base_correct[active_mask] = active_base_pred.eq(active_targets)
                top1_correct = active_correct.float().sum()
                top5_correct = (
                    active_top5.eq(active_targets.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .sum()
                )
                quality_token_count = active_targets.new_tensor(
                    float(active_targets.numel()), dtype=torch.float32
                )

            binary_weights = binary_eval_mask.view(
                bsz, n_blocks, self.block_size
            ).float()
            loss_3d = loss_per_token.view(bsz, n_blocks, self.block_size)
            correct_3d = correct.view(bsz, n_blocks, self.block_size).float()
            count_per_position = binary_weights.sum(dim=(0, 1)).to(torch.float32)
            loss_sum_per_position = (loss_3d * binary_weights).sum(dim=(0, 1))
            correct_per_position = correct_3d.sum(dim=(0, 1))
            loss_per_position = loss_sum_per_position / count_per_position.clamp(
                min=1.0
            )
            acc_per_position = correct_per_position / count_per_position.clamp(min=1.0)
            valid_token_count = active_weights.sum().to(dtype=torch.float32)
            weighted_token_count = flat_weights.sum().to(dtype=torch.float32)
            accuracy = correct.float().sum() / binary_eval_mask.float().sum().clamp(
                min=1.0
            )
            base_accuracy = (
                base_correct.float().sum()
                / binary_eval_mask.float().sum().clamp(min=1.0)
            )

        diagnostics = {
            "correct_count": correct.float().sum().detach(),
            "eval_token_count": binary_eval_mask.float().sum().detach(),
            "top1_correct_count": top1_correct.detach(),
            "top5_correct_count": top5_correct.detach(),
            "quality_token_count": quality_token_count.detach(),
            "valid_token_count": valid_token_count.detach(),
            "weighted_token_count": weighted_token_count.detach(),
            "sanitized_rows": sanitized_rows.detach(),
            "masked_rows": (~binary_eval_mask & flat_eval_mask).float().sum().detach(),
            "sampled_vocab_size": torch.tensor(
                float(lm_head_weight.shape[0]), dtype=torch.float32, device=device
            ),
            "loss_mode_id": torch.tensor(0.0, dtype=torch.float32, device=device),
            "loss_sum_per_position": loss_sum_per_position.detach(),
            "correct_per_position": correct_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "local_ploss_sum": (loss_per_token * binary_eval_mask.float())
            .sum()
            .detach(),
            # Domino-specific diagnostics.
            "domino_final_loss": final_loss.detach()
            if torch.is_tensor(final_loss)
            else torch.tensor(0.0, device=device),
            "domino_base_loss": base_loss.detach()
            if torch.is_tensor(base_loss)
            else torch.tensor(0.0, device=device),
            "domino_base_accuracy": base_accuracy.detach(),
            "domino_lambda_base": torch.tensor(
                float(lambda_base), dtype=torch.float32, device=device
            ),
        }
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            diagnostics,
        )


class DominoTrainerBackend(DFlashTrainerBackend):
    @property
    def model_type(self):
        return "domino"

    def _training_value(
        self, training_cfg, domino_key: str, dflash_key: str, default: Any
    ):
        value = training_cfg.get(domino_key, None)
        if value is not None:
            return value
        return training_cfg.get(dflash_key, default)

    def _normalize_dflash_config(
        self, drafter_config, target_hf_config, normalized_state, spec_model_path
    ):
        training_cfg = self.config.rollout.drafter.training
        if training_cfg.get("domino_num_target_layers", None) is not None:
            if getattr(drafter_config, "num_context_layers", None) is None:
                drafter_config.num_context_layers = int(
                    training_cfg["domino_num_target_layers"]
                )
        return super()._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = self._training_value(
            training_cfg, "domino_hidden_size", "dflash_hidden_size", None
        )
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(
            self._training_value(
                training_cfg, "domino_num_target_layers", "dflash_num_target_layers", 5
            )
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 36)
        )
        mask_token_id_cfg = self._training_value(
            training_cfg, "domino_mask_token_id", "dflash_mask_token_id", None
        )
        mask_token_id = int(
            mask_token_id_cfg
            if mask_token_id_cfg is not None
            else target_text_config.vocab_size - 1
        )
        target_layer_ids = self._training_value(
            training_cfg, "domino_target_layer_ids", "dflash_target_layer_ids", None
        )
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        return DominoConfig(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=int(
                self._training_value(
                    training_cfg,
                    "domino_num_hidden_layers",
                    "dflash_num_hidden_layers",
                    5,
                )
            ),
            num_attention_heads=int(getattr(target_text_config, "num_attention_heads")),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    getattr(target_text_config, "num_attention_heads"),
                )
            ),
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(
                getattr(target_text_config, "max_position_embeddings", 32768)
            ),
            rope_theta=float(getattr(target_text_config, "rope_theta", 10000.0)),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("domino_block_size", 16)),
            num_anchors=int(training_cfg.get("domino_num_anchors", 512)),
            loss_decay_gamma=float(training_cfg.get("domino_loss_decay_gamma", 7.0)),
            emb_dim=int(training_cfg.get("domino_emb_dim", 256)),
            gru_hidden_dim=int(training_cfg.get("domino_gru_hidden_dim", 1024)),
            pure_draft_prefix_len=int(
                training_cfg.get("domino_pure_draft_prefix_len", 1)
            ),
            shift_label=bool(training_cfg.get("domino_shift_label", True)),
            lambda_base_start=float(training_cfg.get("domino_lambda_base_start", 1.0)),
            lambda_base_decay_steps=int(
                training_cfg.get("domino_lambda_base_decay_steps", 2000)
            ),
            architectures=["DominoDraftModel"],
        )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = (
            os.path.join(spec_model_path, "config.json") if spec_model_path else None
        )
        target_hf_config = self._get_target_hf_config()
        normalized_state = None

        if config_path and os.path.exists(config_path):
            drafter_config = DominoConfig.from_domino_pretrained(spec_model_path)
            if spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger, spec_model_path, action="Loading Domino drafter weights"
                )
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_dict(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DominoConfig):
            raise TypeError(
                f"Domino config is not a DominoConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        draft_model = DominoDraftModel(deepcopy(drafter_config))
        if (
            spec_model_path
            and os.path.exists(spec_model_path)
            and os.path.exists(config_path)
        ):
            self._load_draft_checkpoint(
                draft_model, spec_model_path, normalized_state=normalized_state
            )
        draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        self.target_lm_head = self._build_target_lm_head(
            target_model_path, target_hf_config
        )
        training_cfg = self.config.rollout.drafter.training
        return DominoTrainingModel(
            draft_model=draft_model,
            block_size=int(
                training_cfg.get(
                    "domino_block_size", getattr(drafter_config, "block_size", 16)
                )
            ),
            num_anchors=int(
                training_cfg.get(
                    "domino_num_anchors", getattr(drafter_config, "num_anchors", 512)
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "domino_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            pure_draft_prefix_len=int(
                training_cfg.get(
                    "domino_pure_draft_prefix_len",
                    getattr(drafter_config, "pure_draft_prefix_len", 1),
                )
            ),
            lambda_base_start=float(
                training_cfg.get(
                    "domino_lambda_base_start",
                    getattr(drafter_config, "lambda_base_start", 1.0),
                )
            ),
            lambda_base_decay_steps=int(
                training_cfg.get(
                    "domino_lambda_base_decay_steps",
                    getattr(drafter_config, "lambda_base_decay_steps", 2000),
                )
            ),
        ), drafter_config
