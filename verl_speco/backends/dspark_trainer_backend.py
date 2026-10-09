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

import logging
import os
from copy import deepcopy
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainerBackend,
    DFlashTrainingModel,
    _block_acceptance_counts,
    _check_block_drafter_rows,
    _resolve_sliding_windows,
    _sliding_window_config,
    build_dflash_attention_masks,
    _document_boundary_validity,
)
from verl_speco.models.dflash import resolve_rope_theta
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.ops.dspark_fused_loss import (
    fused_label_cross_entropy,
    fused_loss_capability,
    fused_total_variation,
)
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _nested_get(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value from a mapping or attribute chain."""

    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


class DSparkTrainingModel(DFlashTrainingModel):
    """Training wrapper around DSparkDraftModel.

    It reuses the DFlash block drafter backbone and changes the target alignment
    to DSpark semantics:

    - anchor token `x[a]` seeds position 0,
    - labels are `x[a + 1 : a + 1 + block_size]`,
    - previous tokens are `[x[a], labels[:-1]]`,
    - every supervised position, including position 0, contributes to CE.
    """

    draft_model: DSparkDraftModel

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        loss_mode: str = "full_vocab",
        sampled_ce_negatives: int = 0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 0.0,
        l1_chunk_size: int = 0,
        distribution_loss_impl: str = "auto",
        debug_log: bool = False,
        debug_log_first_n: int = 2,
        debug_log_interval: int = 100,
    ):
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            front_position_weight=1.0,
            front_position_count=0,
            loss_mode=loss_mode,
            sampled_ce_negatives=sampled_ce_negatives,
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.l1_chunk_size = int(l1_chunk_size or 0)
        self.distribution_loss_impl = str(distribution_loss_impl).lower()
        if self.distribution_loss_impl not in {"auto", "fused", "eager"}:
            raise ValueError(
                "distribution_loss_impl must be one of: auto, fused, eager"
            )
        self._logged_distribution_loss_backend: Optional[str] = None
        self._distribution_loss_by_device: dict[str, bool] = {}
        confidence_head = getattr(self.draft_model, "confidence_head", None)
        self.confidence_loss_enabled = self.confidence_head_alpha > 0
        if self.confidence_loss_enabled:
            if confidence_head is None:
                raise ValueError(
                    "DSpark confidence loss is enabled but the draft model has no "
                    "confidence head; set dspark_confidence_head_alpha>0 or load a "
                    "checkpoint that carries one."
                )
            confidence_head.requires_grad_(True)
            logger.info(
                "[dspark-trainer] confidence head is trainable (alpha=%.4f)",
                self.confidence_head_alpha,
            )
        elif confidence_head is not None:
            confidence_head.requires_grad_(False)
            logger.info(
                "[dspark-trainer] confidence head is loaded but frozen because confidence loss is disabled"
            )
        self.debug_log = bool(debug_log)
        self.debug_log_first_n = max(int(debug_log_first_n), 0)
        self.debug_log_interval = max(int(debug_log_interval), 1)
        self._debug_forward_count = 0

    def _use_fused_distribution_loss(self, device: torch.device) -> bool:
        if self.distribution_loss_impl == "eager":
            return False
        device_key = str(device)
        cached = self._distribution_loss_by_device.get(device_key)
        if cached is not None:
            return cached
        capability = fused_loss_capability(device)
        if not capability.available and self.distribution_loss_impl == "fused":
            raise RuntimeError(
                "DSpark fused distribution loss was requested but is unavailable: "
                f"{capability.reason}"
            )
        use_fused = capability.available
        resolved = capability.backend if use_fused else f"eager ({capability.reason})"
        if resolved != self._logged_distribution_loss_backend:
            logger.info(
                "[dspark-trainer] distribution loss implementation: %s", resolved
            )
            self._logged_distribution_loss_backend = resolved
        self._distribution_loss_by_device[device_key] = use_fused
        return use_fused

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        document_ids: Optional[torch.Tensor] = None,
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
        indices = (
            self._cached_arange("dspark_anchor_indices", num_candidates, device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        if document_ids is not None:
            # Keep the whole anchor window inside one document so a block never
            # spans a packing boundary.
            valid = _document_boundary_validity(
                valid,
                indices=indices,
                document_ids=document_ids,
                block_size=self.block_size,
                seq_len=seq_len,
                label_shift=1,
            )
        valid_counts = valid.sum(dim=1)
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
            "dspark_anchor_keep", self.num_anchors, device
        ).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=self.num_anchors)
        return torch.where(keep_mask, selected, 0), keep_mask

    def _build_label_tensors(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        n_blocks = anchor_positions.shape[1]
        label_offsets = (
            self._cached_arange(
                "dspark_label_offsets", self.block_size, device, view_shape=(1, 1, -1)
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

    def _markov_bias_for_active(
        self,
        *,
        active_hidden: torch.Tensor,
        active_prev_tokens: torch.Tensor,
        restricted_vocab: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        markov_head = getattr(self.draft_model, "markov_head", None)
        if markov_head is None:
            return None
        if markov_head.__class__.__name__ == "DSparkRNNMarkovHead":
            raise NotImplementedError(
                "DSpark rnn markov_head_type requires full-block logits and is not supported yet"
            )

        if restricted_vocab is None:
            return markov_head.compute_step_bias(active_prev_tokens, active_hidden)

        prev_embeddings = markov_head.get_prev_embeddings(active_prev_tokens)
        if hasattr(markov_head, "gate_proj"):
            gate = torch.sigmoid(
                markov_head.gate_proj(
                    torch.cat([active_hidden, prev_embeddings], dim=-1)
                )
            )
            prev_embeddings = gate.to(prev_embeddings.dtype) * prev_embeddings
        markov_w2 = markov_head.markov_w2.weight.index_select(
            0, restricted_vocab.to(markov_head.markov_w2.weight.device)
        )
        return F.linear(prev_embeddings, markov_w2.to(prev_embeddings.dtype))

    def _gather_aligned_target_hidden(
        self,
        *,
        target_last_hidden_states: Optional[torch.Tensor],
        label_indices: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_last_hidden_states is None:
            return None
        if target_last_hidden_states.dim() != 3:
            raise ValueError(
                "DSpark target_last_hidden_states must have shape [batch, seq, hidden], "
                f"got {tuple(target_last_hidden_states.shape)}"
            )
        seq_len = int(target_last_hidden_states.size(1))
        if seq_len <= 0:
            return None
        target_pred_indices = (label_indices - 1).clamp(min=0, max=seq_len - 1)
        target_pred_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            target_pred_indices,
            torch.zeros_like(target_pred_indices),
        )
        batch_size, _, hidden_size = target_last_hidden_states.shape
        if target_pred_indices.size(0) != batch_size:
            raise ValueError(
                "DSpark label_indices batch size must match target_last_hidden_states: "
                f"{target_pred_indices.size(0)} != {batch_size}"
            )

        # Convert each per-batch token index into a row index of a flattened
        # [batch * seq, hidden] tensor.  The previous gather input was an
        # expanded [batch, anchors, seq, hidden] view; Ascend may materialize
        # that view and allocate tens of GiB before gather executes.
        batch_offsets = (
            torch.arange(
                batch_size,
                device=target_pred_indices.device,
                dtype=target_pred_indices.dtype,
            ).view(batch_size, 1, 1)
            * seq_len
        )
        flat_indices = (target_pred_indices + batch_offsets).reshape(-1)
        flat_target_hidden = target_last_hidden_states.reshape(
            batch_size * seq_len, hidden_size
        )
        return flat_target_hidden.index_select(0, flat_indices).view(
            *target_pred_indices.shape, hidden_size
        )

    def _compute_distribution_losses_for_active(
        self,
        *,
        active_hidden: torch.Tensor,
        active_prev_tokens: torch.Tensor,
        active_target_hidden: torch.Tensor,
        active_weights: torch.Tensor,
        lm_head_weight: torch.Tensor,
        use_fused: bool = False,
        active_draft_logits: Optional[torch.Tensor] = None,
        active_draft_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute the DSpark distribution losses for the active (valid) rows.

        Returns ``(l1_sum, l1_den, confidence_sum, confidence_den,
        acceptance_sum, confidence_pred_sum)``.

        The confidence head learns the analytical rejection-sampling acceptance
        rate ``alpha = sum_v min(p_v, q_v) = 1 - d_TV`` between the target
        (verifier) and draft distributions, mirroring ``speculators``. The
        acceptance target is built from the frozen full-vocab distributions and
        is detached; the confidence term itself flows into the head and, through
        the shared draft hidden/Markov features, the draft backbone. Both losses
        share the same per-chunk draft/target distributions, so the draft
        full-vocab forward is not repeated when both are enabled.
        """
        if active_hidden.numel() == 0:
            zero = active_weights.new_zeros(())
            return zero, zero, zero, zero, zero, zero

        l1_enabled = self.l1_loss_alpha > 0
        confidence_enabled = self.confidence_loss_enabled

        l1_sum = active_weights.new_zeros((), dtype=torch.float32)
        l1_den = active_weights.new_zeros((), dtype=torch.float32)
        if l1_enabled:
            l1_den = active_weights.float().sum()
        conf_sum = active_weights.new_zeros((), dtype=torch.float32)
        conf_den = active_weights.new_zeros((), dtype=torch.float32)
        accept_sum = active_weights.new_zeros((), dtype=torch.float32)
        pred_sum = active_weights.new_zeros((), dtype=torch.float32)

        active_count = int(active_hidden.size(0))
        expected_shape = (active_count, int(lm_head_weight.size(0)))
        for name, tensor in (
            ("draft log probabilities", active_draft_log_probs),
            ("draft logits", active_draft_logits),
        ):
            if tensor is not None and tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"DSpark precomputed {name} must have shape "
                    f"{expected_shape}, got {tuple(tensor.shape)}"
                )

        chunk_size = self.l1_chunk_size if self.l1_chunk_size > 0 else active_count
        for start in range(0, active_count, chunk_size):
            end = min(start + chunk_size, active_count)
            hidden_chunk = active_hidden[start:end]
            prev_chunk = active_prev_tokens[start:end]
            target_hidden_chunk = active_target_hidden[start:end]
            weights_chunk = active_weights[start:end].float()

            draft_logits: torch.Tensor | None = None
            draft_probs: torch.Tensor | None
            if not use_fused and active_draft_log_probs is not None:
                draft_probs = active_draft_log_probs[start:end].exp()
            elif use_fused and active_draft_logits is not None:
                draft_logits = active_draft_logits[start:end]
                draft_probs = None
            else:
                draft_logits = F.linear(hidden_chunk, lm_head_weight)
                markov_bias = self._markov_bias_for_active(
                    active_hidden=hidden_chunk,
                    active_prev_tokens=prev_chunk,
                    restricted_vocab=None,
                )
                if markov_bias is not None:
                    draft_logits = draft_logits + markov_bias
                draft_probs = (
                    None if use_fused else torch.softmax(draft_logits.float(), dim=-1)
                )

            with torch.no_grad():
                target_logits = F.linear(target_hidden_chunk, lm_head_weight)
                if not use_fused:
                    target_probs = torch.softmax(target_logits.float(), dim=-1)

            if use_fused:
                if draft_logits is None:
                    raise RuntimeError(
                        "DSpark fused distribution loss requires draft logits"
                    )
                if l1_enabled:
                    tv = fused_total_variation(draft_logits, target_logits)
                else:
                    # Confidence-only: the varying part is discarded, so skip
                    # building an autograd graph for the fused TV.
                    with torch.no_grad():
                        tv = fused_total_variation(draft_logits, target_logits)
                l1_dist = 2.0 * tv
                acceptance_target = (1.0 - tv).clamp(0.0, 1.0).detach()
            else:
                if draft_probs is None:
                    if draft_logits is None:
                        raise RuntimeError(
                            "DSpark distribution loss requires draft logits or "
                            "probabilities"
                        )
                    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
                if l1_enabled:
                    l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
                else:
                    # Confidence-only: the varying part is discarded, so skip
                    # building an autograd graph for the eager L1 distance.
                    with torch.no_grad():
                        l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
                acceptance_target = (1.0 - 0.5 * l1_dist).clamp(0.0, 1.0).detach()

            if l1_enabled:
                l1_sum = l1_sum + (l1_dist * weights_chunk).sum()

            if confidence_enabled:
                confidence_logits = self.draft_model.predict_confidence(
                    hidden_chunk, prev_token_ids=prev_chunk
                )
                if confidence_logits is None:
                    raise ValueError(
                        "DSpark confidence loss requires draft_model.confidence_head"
                    )
                if confidence_logits.shape != l1_dist.shape:
                    raise ValueError(
                        "DSpark confidence logits must match the active-token shape: "
                        f"expected={tuple(l1_dist.shape)} got={tuple(confidence_logits.shape)}"
                    )
                bce = F.binary_cross_entropy_with_logits(
                    confidence_logits.float(), acceptance_target, reduction="none"
                )
                finite_confidence = torch.isfinite(bce)
                confidence_weights = weights_chunk * finite_confidence.to(
                    dtype=weights_chunk.dtype
                )
                conf_sum = (
                    conf_sum
                    + (
                        torch.where(finite_confidence, bce, torch.zeros_like(bce))
                        * confidence_weights
                    ).sum()
                )
                conf_den = conf_den + confidence_weights.sum()
                # Weight the acceptance and prediction diagnostics by the same
                # finite-token mask as the loss so ``accept_sum / conf_den`` and
                # ``pred_sum / conf_den`` stay valid weighted means.
                accept_sum = accept_sum + (acceptance_target * confidence_weights).sum()
                pred_sum = (
                    pred_sum
                    + (
                        confidence_logits.detach().float().sigmoid()
                        * confidence_weights
                    ).sum()
                )
        return l1_sum, l1_den, conf_sum, conf_den, accept_sum, pred_sum

    def _should_debug_log(self) -> bool:
        if not self.debug_log:
            return False
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank != 0:
            return False
        count = self._debug_forward_count
        return count <= self.debug_log_first_n or count % self.debug_log_interval == 0

    def _log_debug_preview(
        self,
        *,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        target_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        active_logits: Optional[torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        if not self._should_debug_log():
            return
        valid_anchor = torch.nonzero(block_keep_mask, as_tuple=False)
        preview: dict[str, Any] = {}
        if valid_anchor.numel() > 0:
            batch_idx = int(valid_anchor[0, 0].detach().cpu().item())
            block_idx = int(valid_anchor[0, 1].detach().cpu().item())
            anchor = int(anchor_positions[batch_idx, block_idx].detach().cpu().item())
            preview = {
                "anchor": anchor,
                "anchor_token": int(input_ids[batch_idx, anchor].detach().cpu().item()),
                "labels": target_ids[batch_idx, block_idx]
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
                "prev": prev_token_ids[batch_idx, block_idx]
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
                "eval": eval_mask[batch_idx, block_idx]
                .to(torch.int32)
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
            }
        logger.info(
            "[dspark-trainer] batch input_shape=%s loss_tokens=%s anchors=%s sampled=%s preview=%s logits_shape=%s loss=%.6f",
            tuple(input_ids.shape),
            int(eval_mask.detach().sum().cpu().item()),
            int(block_keep_mask.numel()),
            int(block_keep_mask.detach().sum().cpu().item()),
            preview,
            tuple(active_logits.shape) if active_logits is not None else None,
            float(loss.detach().float().cpu().item()),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: list[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        document_ids: Optional[torch.Tensor] = None,
        label_ids: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        self._debug_forward_count += 1
        if label_ids is None:
            label_ids = input_ids
        if label_mask is None:
            label_mask = loss_mask
        label_len = label_ids.shape[1]
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            label_len, label_mask, device, document_ids=document_ids
        )
        # Anchors must point at a real context row (the hidden states may be one
        # row shorter than the label sequence when a trailing label token exists).
        block_keep_mask = block_keep_mask & (anchor_positions < seq_len)
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )
        block_mask, dense_attention_mask = build_dflash_attention_masks(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            ctx_len=seq_len,
            block_size=self.block_size,
            device=device,
            windows=_resolve_sliding_windows(self.draft_model.config),
            document_ids=document_ids,
        )

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            dense_attention_mask=dense_attention_mask,
            noise_embedding=noise_embedding,
        ).view(bsz, n_blocks, self.block_size, -1)

        target_ids, prev_token_ids, eval_mask, label_indices = (
            self._build_label_tensors(
                input_ids=label_ids,
                loss_mask=label_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
            )
        )
        aligned_target_hidden = self._gather_aligned_target_hidden(
            target_last_hidden_states=target_last_hidden_states,
            label_indices=label_indices,
            block_keep_mask=block_keep_mask,
        )

        weight_mask = eval_mask.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = self._cached_arange(
                "dspark_decay_positions", self.block_size, device, view_shape=(1, 1, -1)
            )
            weight_mask = weight_mask * torch.exp(
                -positions.float() / float(self.loss_decay_gamma)
            )

        flat_targets = target_ids.reshape(-1)
        flat_prev_tokens = prev_token_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        active_mask = flat_weights > 0
        flat_hidden = draft_hidden.reshape(-1, draft_hidden.size(-1))
        active_hidden = flat_hidden[active_mask]
        active_targets = flat_targets[active_mask]
        active_prev_tokens = flat_prev_tokens[active_mask]
        active_weights = flat_weights[active_mask]
        active_target_hidden = (
            aligned_target_hidden.reshape(-1, aligned_target_hidden.size(-1))[
                active_mask
            ]
            if aligned_target_hidden is not None
            else None
        )
        loss_per_token = torch.zeros_like(flat_weights)

        sanitized_rows = torch.zeros((), dtype=torch.float32, device=device)
        local_ploss_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_ce_den = torch.zeros((), dtype=torch.float32, device=device)
        local_l1_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_l1_den = torch.zeros((), dtype=torch.float32, device=device)
        local_conf_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_conf_den = torch.zeros((), dtype=torch.float32, device=device)
        confidence_accept_sum = torch.zeros((), dtype=torch.float32, device=device)
        confidence_pred_sum = torch.zeros((), dtype=torch.float32, device=device)
        active_logits = None
        active_log_probs = None
        restricted_vocab = None
        use_fused_loss = self._use_fused_distribution_loss(device)
        if active_targets.numel() == 0:
            loss = flat_weights.sum() * 0.0
        else:
            if self.loss_mode in {"restricted_ce", "sampled_ce"}:
                restricted_vocab = self._build_restricted_vocab(
                    input_ids, active_targets, int(lm_head_weight.shape[0])
                )
                restricted_weight = lm_head_weight.index_select(0, restricted_vocab)
                active_logits = F.linear(active_hidden, restricted_weight)
                markov_bias = self._markov_bias_for_active(
                    active_hidden=active_hidden,
                    active_prev_tokens=active_prev_tokens,
                    restricted_vocab=restricted_vocab,
                )
                if markov_bias is not None:
                    active_logits = active_logits + markov_bias
                active_ce_targets = torch.searchsorted(restricted_vocab, active_targets)
                active_loss = (
                    fused_label_cross_entropy(active_logits, active_ce_targets)
                    if use_fused_loss
                    else F.cross_entropy(
                        active_logits, active_ce_targets, reduction="none"
                    )
                )
            else:
                active_logits = F.linear(active_hidden, lm_head_weight)
                markov_bias = self._markov_bias_for_active(
                    active_hidden=active_hidden,
                    active_prev_tokens=active_prev_tokens,
                    restricted_vocab=None,
                )
                if markov_bias is not None:
                    active_logits = active_logits + markov_bias
                if use_fused_loss:
                    active_loss = fused_label_cross_entropy(
                        active_logits, active_targets
                    )
                else:
                    active_log_probs = F.log_softmax(active_logits.float(), dim=-1)
                    active_loss = F.nll_loss(
                        active_log_probs, active_targets, reduction="none"
                    )

            finite_loss = torch.isfinite(active_loss)
            sanitized_rows = (~finite_loss).sum().to(dtype=torch.float32)
            active_loss = torch.where(
                finite_loss, active_loss, torch.zeros_like(active_loss)
            )
            active_loss_weights = active_weights * finite_loss.to(
                dtype=active_weights.dtype
            )
            loss_per_token[active_mask] = active_loss
            local_ce_den = active_loss_weights.sum()
            valid_token_count = local_ce_den.clamp(min=1e-6)
            local_ploss_sum = (active_loss * active_loss_weights).sum()
            ce_loss = local_ploss_sum / valid_token_count
            needs_target_distribution = (
                self.l1_loss_alpha > 0 or self.confidence_loss_enabled
            )
            if needs_target_distribution:
                if active_target_hidden is None:
                    raise ValueError(
                        "DSpark L1/confidence losses require target_last_hidden_states. "
                        "Enable old-logprob dflash_aux_plus_last collection or disable both "
                        "dspark_l1_loss_alpha and dspark_confidence_loss_alpha."
                    )
                finite_target_hidden = torch.isfinite(active_target_hidden).all(dim=-1)
                distribution_mask = finite_loss & finite_target_hidden
                if distribution_mask.any():
                    all_rows = bool(distribution_mask.all())
                    reusable_draft_logits = None
                    reusable_draft_log_probs = None
                    # Full-vocab CE already normalizes the complete LM head and Markov
                    # bias. Restricted/sampled CE must rebuild full-vocab probabilities
                    # for the L1 and confidence targets inside the loss helper.
                    if restricted_vocab is None:
                        if use_fused_loss:
                            reusable_draft_logits = (
                                active_logits
                                if all_rows
                                else active_logits[distribution_mask]
                            )
                        else:
                            if active_log_probs is None:
                                raise ValueError(
                                    "DSpark distribution losses require active_log_probs"
                                )
                            reusable_draft_log_probs = (
                                active_log_probs
                                if all_rows
                                else active_log_probs[distribution_mask]
                            )
                    (
                        local_l1_sum,
                        local_l1_den,
                        local_conf_sum,
                        local_conf_den,
                        confidence_accept_sum,
                        confidence_pred_sum,
                    ) = self._compute_distribution_losses_for_active(
                        active_hidden=(
                            active_hidden
                            if all_rows
                            else active_hidden[distribution_mask]
                        ),
                        active_prev_tokens=(
                            active_prev_tokens
                            if all_rows
                            else active_prev_tokens[distribution_mask]
                        ),
                        active_target_hidden=(
                            active_target_hidden
                            if all_rows
                            else active_target_hidden[distribution_mask]
                        ),
                        active_weights=(
                            active_loss_weights
                            if all_rows
                            else active_loss_weights[distribution_mask]
                        ),
                        lm_head_weight=lm_head_weight,
                        use_fused=use_fused_loss,
                        active_draft_logits=reusable_draft_logits,
                        active_draft_log_probs=reusable_draft_log_probs,
                    )
                l1_loss = local_l1_sum / local_l1_den.clamp(min=1e-6)
                confidence_loss = local_conf_sum / local_conf_den.clamp(min=1e-6)
            else:
                l1_loss = local_ploss_sum.new_zeros(())
                confidence_loss = local_ploss_sum.new_zeros(())
            loss = (
                (ce_loss * self.ce_loss_alpha)
                + (l1_loss * self.l1_loss_alpha)
                + (confidence_loss * self.confidence_head_alpha)
            )

        with torch.no_grad():
            flat_eval_mask = eval_mask.reshape(-1)
            binary_eval_mask = flat_eval_mask & (flat_weights > 0)
            correct = torch.zeros_like(flat_weights, dtype=torch.bool)
            top1_correct = torch.zeros((), dtype=torch.float32, device=device)
            top5_correct = torch.zeros((), dtype=torch.float32, device=device)
            quality_token_count = torch.zeros((), dtype=torch.float32, device=device)
            sampled_vocab_size = torch.tensor(
                float(restricted_vocab.numel())
                if restricted_vocab is not None
                else float(lm_head_weight.shape[0]),
                dtype=torch.float32,
                device=device,
            )
            if active_logits is not None and active_targets.numel() > 0:
                if restricted_vocab is not None:
                    pred_local = active_logits.argmax(dim=-1)
                    pred_tokens = restricted_vocab[pred_local]
                    topk = min(5, int(restricted_vocab.numel()))
                    top_tokens = restricted_vocab[
                        active_logits.topk(topk, dim=-1).indices
                    ]
                else:
                    pred_tokens = active_logits.argmax(dim=-1)
                    topk = min(5, active_logits.shape[-1])
                    top_tokens = active_logits.topk(topk, dim=-1).indices
                active_correct = pred_tokens.eq(active_targets)
                correct[active_mask] = active_correct
                top1_correct = active_correct.float().sum()
                top5_correct = (
                    top_tokens.eq(active_targets.unsqueeze(-1))
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
            # Prefix acceptance per block; the per-position accuracies above are
            # marginals and cannot be combined into it. Labels here are shifted
            # by one, so unlike DFlash there is no anchor column: every position
            # is a real prediction and none may be sliced off.
            accepted_length_sum, scored_block_count = _block_acceptance_counts(
                correct.view(bsz, n_blocks, self.block_size),
                binary_weights > 0,
            )
            valid_token_count = active_weights.sum().to(dtype=torch.float32)
            weighted_token_count = flat_weights.sum().to(dtype=torch.float32)
            accuracy = correct.float().sum() / binary_eval_mask.float().sum().clamp(
                min=1.0
            )

        self._log_debug_preview(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_ids=target_ids,
            prev_token_ids=prev_token_ids,
            eval_mask=eval_mask,
            active_logits=active_logits,
            loss=loss,
        )

        diagnostics = {
            "correct_count": correct.float().sum().detach(),
            "eval_token_count": binary_eval_mask.float().sum().detach(),
            "top1_correct_count": top1_correct.detach(),
            "top5_correct_count": top5_correct.detach(),
            "quality_token_count": quality_token_count.detach(),
            "valid_token_count": valid_token_count.detach(),
            "weighted_token_count": weighted_token_count.detach(),
            "ce_loss_sum": local_ploss_sum.detach(),
            "ce_weighted_token_count": local_ce_den.detach(),
            "l1_loss_sum": local_l1_sum.detach(),
            "l1_weighted_token_count": local_l1_den.detach(),
            "confidence_loss_sum": local_conf_sum.detach(),
            "confidence_weighted_token_count": local_conf_den.detach(),
            "confidence_accept_rate_sum": confidence_accept_sum.detach(),
            "confidence_pred_mean_sum": confidence_pred_sum.detach(),
            "sanitized_rows": sanitized_rows.detach(),
            "masked_rows": (~binary_eval_mask & flat_eval_mask).float().sum().detach(),
            "sampled_vocab_size": sampled_vocab_size.detach(),
            "loss_mode_id": torch.tensor(
                {"full_vocab": 0.0, "restricted_ce": 1.0, "sampled_ce": 2.0}.get(
                    self.loss_mode, 0.0
                ),
                dtype=torch.float32,
                device=device,
            ),
            "distribution_loss_impl_id": torch.tensor(
                1.0 if use_fused_loss else 0.0,
                dtype=torch.float32,
                device=device,
            ),
            "loss_sum_per_position": loss_sum_per_position.detach(),
            "correct_per_position": correct_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "accepted_length_sum": accepted_length_sum.detach(),
            "scored_block_count": scored_block_count.detach(),
            "local_ploss_sum": local_ploss_sum.detach(),
        }
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            diagnostics,
        )


class DSparkTrainerBackend(DFlashTrainerBackend):
    @property
    def model_type(self):
        return "dspark"

    def _training_value(
        self, training_cfg, dspark_key: str, dflash_key: str, default: Any
    ):
        value = training_cfg.get(dspark_key, None)
        if value is not None:
            return value
        return training_cfg.get(dflash_key, default)

    def _build_target_lm_head(self, target_model_path: str, target_hf_config=None):
        target_lm_head = super()._build_target_lm_head(
            target_model_path, target_hf_config
        )
        actor_config = getattr(self.config, "actor", None)
        actor_strategy = (
            ""
            if actor_config is None
            else (
                actor_config.get("strategy", "")
                if hasattr(actor_config, "get")
                else getattr(actor_config, "strategy", "")
            )
        )
        target_weight = getattr(getattr(target_lm_head, "fc", None), "weight", None)
        if str(actor_strategy).lower() == "veomni" and torch.is_tensor(target_weight):
            target_weight_tensor = cast(torch.Tensor, target_weight)
            if (
                target_weight_tensor.device.type == "npu"
                and target_weight_tensor.dtype != torch.bfloat16
            ):
                target_lm_head = target_lm_head.to(dtype=torch.bfloat16)
                logger.debug(
                    "[dspark-trainer] keep the frozen NPU VeOmni target lm_head in bfloat16"
                )
        return target_lm_head

    def _normalize_dflash_config(
        self, drafter_config, target_hf_config, normalized_state, spec_model_path
    ):
        training_cfg = self.config.rollout.drafter.training
        if training_cfg.get("dspark_num_target_layers", None) is not None:
            if getattr(drafter_config, "num_context_layers", None) is None:
                drafter_config.num_context_layers = int(
                    training_cfg["dspark_num_target_layers"]
                )
        drafter_config = super()._normalize_dflash_config(
            drafter_config,
            target_hf_config,
            normalized_state,
            spec_model_path,
        )
        return self._normalize_dspark_confidence_config(
            drafter_config, normalized_state, spec_model_path, training_cfg
        )

    def _normalize_dspark_confidence_config(
        self, drafter_config, normalized_state, spec_model_path, training_cfg
    ):
        """Resolve the confidence-head topology from config, checkpoint and training.

        Whether the head exists, and whether its input includes the Markov
        previous-token embedding, follows what is actually present. Training
        knobs may only align with the checkpoint topology; a mismatch fails
        closed instead of silently loading an incompatible head through
        ``strict=False``. A markov-less head with ``markov_rank>0`` is only
        valid for offline training because the served vLLM head is Markov-fused.
        """
        training_head_alpha = float(
            training_cfg.get("dspark_confidence_head_alpha", 0.0) or 0.0
        )
        training_loss_alpha = float(
            training_cfg.get("dspark_confidence_loss_alpha", 0.0) or 0.0
        )
        if training_head_alpha < 0 or training_loss_alpha < 0:
            raise ValueError(
                "DSpark confidence-head and loss alphas must be non-negative"
            )

        state = normalized_state or {}
        confidence_weight = state.get("confidence_head.proj.weight")
        confidence_bias = state.get("confidence_head.proj.bias")
        if (confidence_weight is None) != (confidence_bias is None):
            raise ValueError(
                "DSpark checkpoint must contain both confidence_head.proj.weight "
                f"and confidence_head.proj.bias: model_path={spec_model_path}"
            )

        training_requests_head = training_head_alpha > 0 or training_loss_alpha > 0
        config_enables_head = bool(
            getattr(drafter_config, "enable_confidence_head", False)
        )
        checkpoint_has_head = confidence_weight is not None
        enable_confidence_head = (
            training_requests_head or config_enables_head or checkpoint_has_head
        )
        confidence_with_markov = bool(
            getattr(drafter_config, "confidence_head_with_markov", True)
        )
        if training_requests_head:
            confidence_with_markov = bool(
                training_cfg.get("dspark_confidence_head_with_markov", True)
            )

        if checkpoint_has_head:
            if confidence_weight.ndim != 2 or int(confidence_weight.shape[0]) != 1:
                raise ValueError(
                    "DSpark confidence_head.proj.weight must have shape [1, input_dim], "
                    f"got {tuple(confidence_weight.shape)} in {spec_model_path}"
                )
            if confidence_bias.ndim != 1 or tuple(confidence_bias.shape) != (1,):
                raise ValueError(
                    "DSpark confidence_head.proj.bias must have shape [1], "
                    f"got {tuple(confidence_bias.shape)} in {spec_model_path}"
                )
            hidden_size = int(drafter_config.hidden_size)
            markov_rank = int(getattr(drafter_config, "markov_rank", 0))
            confidence_input_dim = int(confidence_weight.shape[1])
            if confidence_input_dim == hidden_size:
                checkpoint_with_markov = False
            elif confidence_input_dim == hidden_size + markov_rank and markov_rank > 0:
                checkpoint_with_markov = True
            else:
                raise ValueError(
                    "DSpark confidence-head input dim is incompatible with the draft "
                    f"hidden/Markov dimensions: checkpoint={confidence_input_dim} "
                    f"hidden_size={hidden_size} markov_rank={markov_rank} "
                    f"model_path={spec_model_path}"
                )
            source_config = getattr(drafter_config, "_source_checkpoint_config", None)
            topology_is_explicit = training_requests_head or (
                isinstance(source_config, dict)
                and "confidence_head_with_markov" in source_config
            )
            if (
                topology_is_explicit
                and confidence_with_markov != checkpoint_with_markov
            ):
                raise ValueError(
                    "DSpark confidence-head topology disagrees with its checkpoint: "
                    f"configured_with_markov={confidence_with_markov} "
                    f"checkpoint_with_markov={checkpoint_with_markov} "
                    f"model_path={spec_model_path}"
                )
            confidence_with_markov = checkpoint_with_markov

        if enable_confidence_head:
            markov_rank = int(getattr(drafter_config, "markov_rank", 0))
            if confidence_with_markov and markov_rank <= 0:
                raise ValueError(
                    "DSpark confidence_head_with_markov requires dspark_markov_rank > 0"
                )
            training_mode = (
                str(training_cfg.get("mode", "online") or "online").strip().lower()
            )
            if (
                training_mode not in {"offline", "collect_only"}
                and training_requests_head
                and not checkpoint_has_head
                and not config_enables_head
            ):
                # A fresh head is created from a checkpoint that does not declare
                # one. The running engine was built without it, so it will skip
                # the published weights until it is restarted from a checkpoint
                # whose config.json declares the head.
                logger.warning(
                    "[dspark-trainer] training a fresh confidence head from a "
                    "checkpoint that does not declare one; a rollout engine "
                    "started from this checkpoint will not register the head and "
                    "will skip its published weights. Restart the engine from a "
                    "checkpoint whose config.json declares "
                    "enable_confidence_head=true to serve the trained head."
                )
            dynamic_spec_config = _nested_get(
                self.config,
                (
                    "rollout",
                    "engine_kwargs",
                    "vllm",
                    "additional_config",
                    "dynamic_spec_config",
                ),
                None,
            )
            dynamic_method = (
                dynamic_spec_config.get("method")
                if hasattr(dynamic_spec_config, "get")
                else getattr(dynamic_spec_config, "method", None)
            )
            dynamic_serving = (
                dynamic_method is not None
                and str(dynamic_method).strip().lower() == "dspark"
            )
            if dynamic_serving and not confidence_with_markov and markov_rank > 0:
                # The vLLM-Ascend confidence-thresholded dynamic verify budget
                # assumes a Markov-fused confidence head. The engine itself
                # follows the checkpoint topology, so this only restricts the
                # dynamic path; static serving and offline export may be
                # markov-less.
                raise ValueError(
                    "DSpark dynamic_spec_config.method=dspark requires "
                    "dspark_confidence_head_with_markov=True because the "
                    "vLLM-Ascend dynamic verify-budget consumer assumes a "
                    "Markov-fused head. Set dspark_confidence_head_with_markov=True, "
                    "or drop rollout.engine_kwargs.vllm.additional_config."
                    "dynamic_spec_config (static serving and offline training may "
                    "be markov-less)."
                )
            drafter_config.enable_confidence_head = True
            drafter_config.confidence_head_with_markov = confidence_with_markov
            drafter_config.confidence_head_alpha = max(
                float(getattr(drafter_config, "confidence_head_alpha", 0.0)),
                training_head_alpha,
                training_loss_alpha,
            )
            logger.info(
                "[dspark-trainer] confidence head enabled "
                "(with_markov=%s head_alpha=%.4f loss_alpha=%.4f checkpoint_has_head=%s)",
                confidence_with_markov,
                training_head_alpha,
                training_loss_alpha,
                checkpoint_has_head,
            )
        return drafter_config

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = self._training_value(
            training_cfg, "dspark_hidden_size", "dflash_hidden_size", None
        )
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(
            self._training_value(
                training_cfg, "dspark_num_target_layers", "dflash_num_target_layers", 5
            )
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 36)
        )
        mask_token_id_cfg = self._training_value(
            training_cfg, "dspark_mask_token_id", "dflash_mask_token_id", None
        )
        mask_token_id = int(
            mask_token_id_cfg
            if mask_token_id_cfg is not None
            else target_text_config.vocab_size - 1
        )
        target_layer_ids = self._training_value(
            training_cfg, "dspark_target_layer_ids", "dflash_target_layer_ids", None
        )
        target_head_dim = getattr(target_text_config, "head_dim", None)
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        confidence_head_alpha = max(
            float(training_cfg.get("dspark_confidence_head_alpha", 0.0) or 0.0),
            float(training_cfg.get("dspark_confidence_loss_alpha", 0.0) or 0.0),
        )
        confidence_loss_alpha = float(
            training_cfg.get("dspark_confidence_loss_alpha", 0.0)
        )
        intermediate_size_cfg = self._training_value(
            training_cfg, "dspark_intermediate_size", "dflash_intermediate_size", None
        )
        intermediate_size = int(
            intermediate_size_cfg
            if intermediate_size_cfg is not None
            else getattr(target_text_config, "intermediate_size", hidden_size * 4)
        )
        num_hidden_layers = int(
            self._training_value(
                training_cfg,
                "dspark_num_hidden_layers",
                "dflash_num_hidden_layers",
                1,
            )
        )
        sliding_window = self._training_value(
            training_cfg, "dspark_sliding_window", "dflash_sliding_window", None
        )
        return DSparkConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(getattr(target_text_config, "num_attention_heads")),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    getattr(target_text_config, "num_attention_heads"),
                )
            ),
            head_dim=int(target_head_dim) if target_head_dim is not None else None,
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(
                getattr(target_text_config, "max_position_embeddings", 32768)
            ),
            rope_theta=resolve_rope_theta(target_text_config),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("dspark_block_size", 7)),
            num_anchors=int(training_cfg.get("dspark_num_anchors", 512)),
            markov_rank=int(training_cfg.get("dspark_markov_rank", 256)),
            markov_head_type=str(
                training_cfg.get("dspark_markov_head_type", "vanilla")
            ),
            confidence_head_alpha=confidence_head_alpha,
            enable_confidence_head=bool(
                confidence_head_alpha > 0.0 or confidence_loss_alpha > 0.0
            ),
            confidence_head_with_markov=bool(
                training_cfg.get("dspark_confidence_head_with_markov", True)
            ),
            ce_loss_alpha=float(training_cfg.get("dspark_ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(training_cfg.get("dspark_l1_loss_alpha", 0.9)),
            **_sliding_window_config(sliding_window, num_hidden_layers),
        )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")
        target_hf_config = self._get_target_hf_config()
        normalized_state = None

        if config_path and os.path.exists(config_path):
            drafter_config = DSparkConfig.from_dspark_pretrained(spec_model_path)
            if spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger, spec_model_path, action="Loading DSpark drafter weights"
                )
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_dict(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DSparkConfig):
            raise TypeError(
                f"DSpark config is not a DSparkConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        if (
            spec_model_path
            and os.path.exists(spec_model_path)
            and os.path.exists(config_path)
        ):
            draft_model = DSparkDraftModel(deepcopy(drafter_config))
            self._load_draft_checkpoint(
                draft_model, spec_model_path, normalized_state=normalized_state
            )
        else:
            draft_model = DSparkDraftModel(deepcopy(drafter_config))
        draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        self.target_lm_head = self._build_target_lm_head(
            target_model_path, target_hf_config
        )
        training_cfg = self.config.rollout.drafter.training
        return DSparkTrainingModel(
            draft_model=draft_model,
            block_size=int(
                training_cfg.get(
                    "dspark_block_size", getattr(drafter_config, "block_size", 7)
                )
            ),
            num_anchors=int(
                training_cfg.get(
                    "dspark_num_anchors", getattr(drafter_config, "num_anchors", 512)
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "dspark_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            loss_mode=str(training_cfg.get("dspark_loss_mode", "full_vocab")),
            sampled_ce_negatives=int(
                training_cfg.get("dspark_sampled_ce_negatives", 0)
            ),
            ce_loss_alpha=float(
                training_cfg.get(
                    "dspark_ce_loss_alpha",
                    getattr(drafter_config, "ce_loss_alpha", 0.1),
                )
            ),
            l1_loss_alpha=float(
                training_cfg.get(
                    "dspark_l1_loss_alpha",
                    getattr(drafter_config, "l1_loss_alpha", 0.9),
                )
            ),
            confidence_head_alpha=float(
                training_cfg.get("dspark_confidence_loss_alpha", 0.0)
            ),
            l1_chunk_size=int(training_cfg.get("dspark_l1_chunk_size", 0)),
            distribution_loss_impl=str(
                training_cfg.get("dspark_distribution_loss_impl", "auto")
            ),
            debug_log=bool(training_cfg.get("dspark_debug_log", False)),
            debug_log_first_n=int(training_cfg.get("dspark_debug_log_first_n", 2)),
            debug_log_interval=int(training_cfg.get("dspark_debug_log_interval", 100)),
        ), drafter_config

    def preprocess_individual_items(self, items, device, model_config):
        res = {"ids": [], "h_states": [], "masks": [], "target_last_h_states": []}
        raw_max_window = self.config.rollout.drafter.training.get("dspark_max_window")
        max_window = (
            None
            if raw_max_window is None or str(raw_max_window).strip() == ""
            else int(raw_max_window)
        )
        if max_window is not None and max_window < 0:
            raise ValueError("dspark_max_window must be non-negative")
        if max_window == 0:
            max_window = None
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = int(
            getattr(model_config, "target_hidden_size", model_config.hidden_size)
        )
        num_context_layers = int(
            getattr(
                model_config,
                "num_context_layers",
                getattr(model_config, "num_target_layers", 5),
            )
        )
        expected_hidden_dim = h_dim * num_context_layers

        for item in items:
            layout = item.get("hidden_states_layout")
            if layout not in (None, "dflash_aux", "dflash_aux_plus_last"):
                raise ValueError(
                    f"DSpark expected hidden_states_layout='dflash_aux' or 'dflash_aux_plus_last', got {layout!r}. "
                    "This usually means EAGLE3 aux+last hidden states were routed into DSpark training."
                )
            ids = item["input_ids"].to(device, non_blocking=True)
            raw_h = item["hidden_states"]
            full_h = (
                torch.cat(raw_h, dim=-1) if isinstance(raw_h, (list, tuple)) else raw_h
            )
            full_h = full_h.to(device, dtype=torch.bfloat16)
            if full_h.size(-1) < expected_hidden_dim:
                raise ValueError(
                    f"DSpark expected at least {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), got {full_h.size(-1)}"
                )
            target_last_h = None
            if layout == "dflash_aux_plus_last":
                expected_with_last = expected_hidden_dim + h_dim
                if full_h.size(-1) != expected_with_last:
                    raise ValueError(
                        "DSpark hidden_states_layout='dflash_aux_plus_last' expected exactly "
                        f"{expected_with_last} hidden dims ({num_context_layers} context layers plus final hidden), "
                        f"got {full_h.size(-1)}"
                    )
                target_last_h = full_h[..., expected_hidden_dim:expected_with_last]
                full_h = full_h[..., :expected_hidden_dim]
            elif layout == "dflash_aux" and full_h.size(-1) != expected_hidden_dim:
                raise ValueError(
                    f"DSpark hidden_states_layout='dflash_aux' expected exactly {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), got {full_h.size(-1)}"
                )

            if item.get("loss_mask") is not None:
                item_loss_mask = item["loss_mask"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
            elif "prompts" in item and "responses" in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                prompt_len = item["prompts"].size(0)
                responses = item["responses"]
                item_loss_mask[prompt_len : prompt_len + responses.size(0)] = (
                    responses != pad_id
                ).float()[: max(0, ids.size(0) - prompt_len)]
            else:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                item_loss_mask[:] = 1.0

            _check_block_drafter_rows(
                int(ids.size(0)),
                int(full_h.size(0)),
                int(item_loss_mask.size(0)),
                "DSpark",
            )
            if max_window is None:
                start, end = 0, ids.size(0)
            else:
                nonzero = torch.nonzero(item_loss_mask)
                if nonzero.numel() > 0:
                    r_start = nonzero[0, 0]
                    start = torch.clamp(
                        r_start - (max_window // 2),
                        min=0,
                        max=max(0, ids.size(0) - max_window),
                    ).item()
                    end = min(start + max_window, ids.size(0))
                else:
                    start, end = max(0, ids.size(0) - max_window), ids.size(0)

            # ``ids`` / ``item_loss_mask`` carry the (optional) trailing label
            # token; the hidden context is one row shorter when it is present.
            hidden_end = max(start, min(end, int(full_h.size(0))))
            res["ids"].append(ids[start:end])
            res["h_states"].append(full_h[start:hidden_end, :expected_hidden_dim])
            res["masks"].append(item_loss_mask[start:end])
            if target_last_h is not None:
                res["target_last_h_states"].append(target_last_h[start:hidden_end])
            else:
                res["target_last_h_states"].append(None)
        return res

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError(
                "DSpark drafter training does not support Ulysses sequence parallel yet"
            )
        if self.target_lm_head is None:
            raise ValueError("DSpark target_lm_head is not initialized")

        draft_model = model.module if hasattr(model, "module") else model
        hidden_states = batch["hidden_states"]
        num_context_layers = draft_model.draft_model.num_context_layers
        per_layer_dim = hidden_states.shape[-1] // num_context_layers
        hidden_states_list = list(hidden_states.split(per_layer_dim, dim=-1))

        loss, accuracy, loss_pp, acc_pp, count_pp, diagnostics = model(
            input_ids=batch["input_ids"],
            hidden_states_list=hidden_states_list,
            loss_mask=batch["loss_mask"],
            lm_head_weight=self.target_lm_head.fc.weight,
            target_last_hidden_states=batch.get("target_last_hidden_states"),
            document_ids=batch.get("document_ids"),
            label_ids=batch.get("label_ids"),
            label_mask=batch.get("label_mask"),
        )
        local_num_tokens = diagnostics.get("ce_weighted_token_count")
        if not torch.is_tensor(local_num_tokens):
            local_num_tokens = count_pp.sum()
        local_num_tokens = local_num_tokens.to(loss.device, dtype=loss.dtype)
        return {
            "total_local_vloss": torch.tensor(0.0, device=batch["input_ids"].device),
            "total_local_ploss": loss * local_num_tokens,
            "local_num_tokens": local_num_tokens,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_pp.detach(),
            "acc_per_position": acc_pp.detach(),
            "count_per_position": count_pp.detach(),
            "diagnostics": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in diagnostics.items()
            },
        }
