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
"""LiLiCorr trainer backend (arXiv:2608.20530).

LiLiCorr binds the DFlash backbone; only the auxiliary objective changes. The
lattice head is a *block-shaped* objective (candidates per slot, teacher-forced
predecessor, greedy walk for metrics), so it plugs into the block-layout hook
``DFlashTrainingModel._block_lattice_loss`` instead of the flattened
``_auxiliary_loss``. Everything else — anchor sampling, FSDP/DDP, Muon, LR, TQ,
checkpoints — is inherited unchanged.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy

import torch

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainerBackend,
    DFlashTrainingModel,
    _sliding_window_config,
)
from verl_speco.models.dflash import build_target_layer_ids, resolve_rope_theta
from verl_speco.models.lilicorr import (
    LiLiCorrConfig,
    LiLiCorrDraftModel,
    compute_lattice_terms,
)
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step

logger = logging.getLogger(__name__)


class LiLiCorrTrainingModel(DFlashTrainingModel):
    """DFlash training wrapper plus the LiLiCorr lattice objective.

    ``loss = dflash_loss + lilicorr_w_ce * CE + lilicorr_w_pen * penalty`` — the
    paper's ``base`` objective (0.25 each), with no margin hinge: the training
    reference (speculators' ``models/lilicorr/core.py``) does not implement one.
    """

    def __init__(
        self,
        *args,
        lilicorr_w_ce: float = 0.25,
        lilicorr_w_pen: float = 0.25,
        loss_chunk_anchors: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.loss_mode != "full_vocab":
            raise ValueError(
                "LiLiCorr requires loss_mode='full_vocab': vLLM cannot serve a "
                "reduced draft vocabulary, so the candidate lattice must cover "
                f"real ids. Got {self.loss_mode!r}."
            )
        self.lilicorr_w_ce = float(lilicorr_w_ce)
        self.lilicorr_w_pen = float(lilicorr_w_pen)
        self.lilicorr_loss_chunk_anchors = (
            None if loss_chunk_anchors in (None, 0) else int(loss_chunk_anchors)
        )

    # Name-shadowing the base's ``None`` class attribute makes the block hook run.
    def _block_lattice_loss(
        self,
        *,
        draft_hidden,
        lm_head_weight,
        target_ids,
        aligned_target_hidden,
        base_loss_mask,
        original_loss_mask,
        valid_label_mask,
        block_keep_mask,
        anchor_positions,
        context_feature,
        document_ids,
        seq_len,
        target_hidden_is_raw,
    ):
        if aligned_target_hidden is None:
            raise ValueError(
                "LiLiCorr needs the target's final hidden states: enable "
                "'dflash_aux_plus_last' hidden-state collection (dflash_aux alone "
                "stops at the aux layers, so the target argmax / distractor cost "
                "cannot be formed)."
            )
        head = getattr(self.draft_model, "lilicorr", None)
        if head is None:
            raise ValueError("LiLiCorr head is missing from the draft model")

        block = self.block_size
        bsz = draft_hidden.shape[0]
        n_blocks = anchor_positions.shape[1]
        draft_blocks = draft_hidden.view(bsz, n_blocks, block, -1)
        target_blocks = aligned_target_hidden.view(bsz, n_blocks, block, -1)

        # ``base_loss_mask`` already zeroes the anchor column, so the lattice's
        # "all slots supervised" test has to read the gathered label mask
        # instead; the anchor only bounds the block, it is not a slot.
        slot_mask = (original_loss_mask.view(bsz, n_blocks, block) > 0) & (
            valid_label_mask.view(bsz, n_blocks, block)
        )

        # Context row immediately before the anchor token, projected exactly like
        # the context the draft attends to. Positions outside the context window
        # are invalid anchors (the block layout already drops them).
        prev_pos = (anchor_positions - 1).clamp(min=0, max=seq_len - 1)
        anchor_valid = (anchor_positions > 0) & (anchor_positions < seq_len)
        if document_ids is not None:
            doc_len = document_ids.shape[1]
            doc_anchor = torch.gather(
                document_ids, 1, anchor_positions.clamp(max=doc_len - 1)
            )
            doc_prev = torch.gather(document_ids, 1, prev_pos)
            anchor_valid = anchor_valid & (doc_anchor == doc_prev)
            last_pos = (anchor_positions + block - 1).clamp(max=doc_len - 1)
            doc_last = torch.gather(document_ids, 1, last_pos)
            doc_ok = doc_anchor == doc_last
        else:
            doc_ok = torch.ones_like(anchor_positions, dtype=torch.bool)

        valid_block = (
            slot_mask[:, :, 1:].all(-1) & block_keep_mask.bool() & anchor_valid & doc_ok
        )

        # [1, seq, H] -- bsz-1 context rows -> [bsz, nb, H]. A zeroed row still
        # projects to a bias, so an invalid anchor is masked head-side too.
        context_rows = context_feature[0]
        anchor_hidden = context_rows.index_select(0, prev_pos.reshape(-1)).reshape(
            bsz, n_blocks, -1
        )

        chunk = self.lilicorr_loss_chunk_anchors or n_blocks
        loss, metrics = compute_lattice_terms(
            head=head,
            embed_tokens=self.draft_model.embed_tokens,
            lm_head_weight=lm_head_weight,
            block_size=block,
            draft_hidden=draft_blocks,
            target_hidden=target_blocks,
            anchor_hidden=anchor_hidden,
            anchor_valid=anchor_valid,
            valid_block=valid_block,
            slot_mask=slot_mask,
            w_ce=self.lilicorr_w_ce,
            w_pen=self.lilicorr_w_pen,
            chunk_blocks=chunk,
        )
        return loss, metrics


class LiLiCorrTrainerBackend(DFlashTrainerBackend):
    @property
    def model_type(self):
        return "lilicorr"

    def _training_value(
        self, training_cfg, lilicorr_key: str, dflash_key: str, default
    ):
        value = training_cfg.get(lilicorr_key, None)
        if value is not None:
            return value
        return training_cfg.get(dflash_key, default)

    def preprocess_individual_items(self, items, device, model_config):
        # The base DFlash preprocessor crops to ``dflash_max_window``; LiLiCorr
        # exposes its own window key so the two families stay independently
        # tunable. Map it just for the duration of this call.
        window = self.config.rollout.drafter.training.get("lilicorr_max_window", None)
        if window is not None:
            self.config.rollout.drafter.training["dflash_max_window"] = int(window)
        return super().preprocess_individual_items(items, device, model_config)

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size = int(
            self._training_value(
                training_cfg,
                "lilicorr_backbone_hidden_size",
                "dflash_hidden_size",
                None,
            )
            or target_text_config.hidden_size
        )
        num_context_layers = int(
            self._training_value(
                training_cfg,
                "lilicorr_num_target_layers",
                "dflash_num_target_layers",
                5,
            )
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 36)
        )
        mask_token_id = int(
            self._training_value(
                training_cfg,
                "lilicorr_mask_token_id",
                "dflash_mask_token_id",
                target_text_config.vocab_size - 1,
            )
        )
        target_layer_ids = self._training_value(
            training_cfg, "lilicorr_target_layer_ids", "dflash_target_layer_ids", None
        )
        if target_layer_ids is None:
            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        num_hidden_layers = int(
            self._training_value(
                training_cfg,
                "lilicorr_num_hidden_layers",
                "dflash_num_hidden_layers",
                5,
            )
        )
        target_head_dim = getattr(target_text_config, "head_dim", None)
        return LiLiCorrConfig(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(target_text_config.num_attention_heads),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    target_text_config.num_attention_heads,
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
            target_layer_ids=list(target_layer_ids),
            mask_token_id=mask_token_id,
            # The draft backbone has its own layer count; the target's
            # layer_types list cannot be copied verbatim (length must match).
            **_sliding_window_config(
                training_cfg.get("lilicorr_sliding_window", None), num_hidden_layers
            ),
            block_size=int(training_cfg.get("lilicorr_block_size", 16)),
            num_anchors=int(training_cfg.get("lilicorr_num_anchors", 512)),
            loss_decay_gamma=float(training_cfg.get("lilicorr_loss_decay_gamma", 7.0)),
            lilicorr_candidate_topk=int(training_cfg.get("lilicorr_candidate_topk", 8)),
            lilicorr_hidden_size=training_cfg.get("lilicorr_head_hidden_size", 1024),
            lilicorr_factor_dim=int(training_cfg.get("lilicorr_factor_dim", 1024)),
            lilicorr_num_layers=int(training_cfg.get("lilicorr_num_layers", 2)),
            lilicorr_num_heads=int(training_cfg.get("lilicorr_num_heads", 8)),
            lilicorr_mlp_ratio=float(training_cfg.get("lilicorr_mlp_ratio", 2.0)),
            lilicorr_logit_scale=float(training_cfg.get("lilicorr_logit_scale", 8.0)),
            lilicorr_vector_eps=float(training_cfg.get("lilicorr_vector_eps", 1e-4)),
            sample_from_anchor=False,
            causal=True,
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
            drafter_config = LiLiCorrConfig.from_lilicorr_pretrained(spec_model_path)
            log_drafter_checkpoint_step(
                logger, spec_model_path, action="Loading LiLiCorr drafter weights"
            )
            normalized_state = self._normalize_draft_state_dict(
                self._load_draft_state_dict(spec_model_path)
            )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, LiLiCorrConfig):
            raise TypeError(
                f"LiLiCorr config is not a LiLiCorrConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        draft_model = LiLiCorrDraftModel(deepcopy(drafter_config))
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
        # The standalone vLLM reference producer sends the target's *raw*
        # residual stream for the final-layer teacher; normalizing it needs the
        # verifier's final norm, exactly like the DFlash backend.
        verifier_norm = None
        tq_expected = training_cfg.get("transfer_queue", {}).get("expected_feature", {})
        if (
            training_cfg.get("mode") == "offline"
            and training_cfg.get("feature_store", {}).get("type")
            in {"tq", "ray_reference"}
            and not training_cfg.get("use_logits", False)
            and tq_expected.get("payload_kind") == "vllm_reference"
            and tq_expected.get("hidden_states_layout") == "dflash_aux_plus_last"
        ):
            from verl_speco.trainer.target_feature_replay import load_vllm_final_norm

            verifier_norm = load_vllm_final_norm(
                target_model_path,
                dtype=torch.bfloat16,
                trust_remote_code=bool(
                    self.config.model.get("trust_remote_code", False)
                ),
            )
        clip = training_cfg.get("hidden_state_clip_value", 1.0e4)
        return LiLiCorrTrainingModel(
            draft_model=draft_model,
            verifier_norm=verifier_norm,
            target_hidden_clip_value=float(clip)
            if clip is not None and float(clip) > 0
            else None,
            block_size=int(training_cfg.get("lilicorr_block_size", 16)),
            num_anchors=int(
                training_cfg.get(
                    "lilicorr_num_anchors",
                    getattr(drafter_config, "num_anchors", 512),
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "lilicorr_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            loss_mode="full_vocab",
            lilicorr_w_ce=float(training_cfg.get("lilicorr_w_ce", 0.25)),
            lilicorr_w_pen=float(training_cfg.get("lilicorr_w_pen", 0.25)),
            loss_chunk_anchors=int(
                training_cfg.get("lilicorr_loss_chunk_anchors", 128)
            ),
        ), drafter_config


__all__ = ["LiLiCorrTrainerBackend", "LiLiCorrTrainingModel"]
