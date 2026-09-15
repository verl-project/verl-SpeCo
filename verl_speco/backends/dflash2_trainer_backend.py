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

import torch
import torch.nn.functional as F

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainerBackend,
    DFlashTrainingModel,
)
from verl_speco.models.dflash import resolve_rope_theta
from verl_speco.models.dflash2 import DFlash2Config, DFlash2DraftModel
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step

logger = logging.getLogger(__name__)

# Substrings that mark a parameter as belonging to one of the two modules DFlash2
# adds on top of the DFlash backbone.
_DFLASH2_MODULE_KEY_MARKERS = ("attention_conv.", "mlp_conv.", "candidate_selector.")


def _is_dflash2_module_key(key: str) -> bool:
    return any(marker in key for marker in _DFLASH2_MODULE_KEY_MARKERS)


class DFlash2TrainingModel(DFlashTrainingModel):
    """DFlash training wrapper plus the DFlash2 candidate-selector objective.

    The dynamic convolutions live inside the draft model, so they are already
    exercised by the inherited CE path. Only the selector needs its own loss: it
    learns to re-rank the drafter's own top-k candidates using the previous
    token, which at training time is teacher-forced to the ground truth.
    """

    _no_split_modules = ["DFlashDecoderLayer"]

    def __init__(self, *args, selector_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector_loss_weight = float(selector_loss_weight)
        if self.loss_mode != "full_vocab":
            raise ValueError(
                "DFlash2's candidate selector scores real vocabulary ids, so it "
                f"requires loss_mode='full_vocab'; got {self.loss_mode!r}. A "
                "restricted/sampled vocabulary would make the selector's "
                "candidate ids meaningless."
            )

    def _auxiliary_loss(
        self,
        *,
        input_ids,
        safe_label_indices,
        active_mask,
        active_hidden,
        active_logits,
        active_targets,
        active_weights,
    ):
        selector = self.draft_model.candidate_selector
        if self.selector_loss_weight <= 0:
            return None, {}

        device = active_hidden.device
        top_k = min(selector.top_k, active_logits.shape[-1])
        # Detach the backbone inputs. The selector is an auxiliary head that
        # re-ranks whatever the drafter already produced, so its objective must
        # not reshape the drafter itself: torch.topk passes gradient through the
        # selected logits, and hidden_projection would otherwise push gradient
        # back through the backbone as well. Leaving them attached also breaks
        # any DFlash/DFlash2 comparison, because the two backbones would then be
        # trained under different effective objectives rather than differing
        # only by architecture.
        unary, candidates = torch.topk(
            active_logits.detach(), top_k, dim=-1, sorted=False
        )
        selector_hidden = active_hidden.detach()

        # Teacher-forced predecessor: the token immediately before the slot this
        # row predicts. Block-relative position 0 is the anchor and never carries
        # loss weight, so index - 1 stays inside the sequence for every active row.
        predecessor_indices = (safe_label_indices - 1).clamp(min=0)
        bsz, n_blocks, block_size = safe_label_indices.shape
        flat_predecessor_indices = predecessor_indices.reshape(bsz, -1)
        predecessor_ids = torch.gather(input_ids, 1, flat_predecessor_indices)
        predecessor_ids = predecessor_ids.reshape(-1)[active_mask]

        scores = selector.pair_scores(
            selector_hidden, unary.float(), candidates, predecessor_ids
        )

        # The selector can only learn on rows whose ground truth survived the
        # drafter's own top-k; elsewhere there is no correct choice to make.
        # Those rows are zero-weighted rather than sliced out, so the autograd
        # graph covers the same parameters on every rank. Slicing here would
        # make the selector parameters receive gradients only on the ranks that
        # happened to have coverage this step, which desyncs the FSDP2/DDP
        # gradient reduction.
        target_hits = candidates == active_targets.unsqueeze(-1)
        learnable = target_hits.any(dim=-1)
        # argmax over an all-False row yields 0; the row is masked out below.
        row_targets = torch.argmax(target_hits.int(), dim=-1)
        per_row = F.cross_entropy(scores, row_targets, reduction="none")
        finite = torch.isfinite(per_row)
        per_row = torch.where(finite, per_row, torch.zeros_like(per_row))
        row_weights = (
            active_weights.float()
            * learnable.to(torch.float32)
            * finite.to(torch.float32)
        )
        selector_loss = (per_row * row_weights).sum() / row_weights.sum().clamp(
            min=1e-6
        )

        with torch.no_grad():
            scored = learnable & finite
            # Control for the selector's own accuracy. S_t(a, b) starts from the
            # drafter's logit U_t(b), so once the backbone ranks the truth first
            # on its own the selector scores perfectly without the bilinear term
            # having learned anything. Measuring the unary-only ranking on the
            # same rows is what isolates the selector's actual contribution:
            # the lift is selector_correct_count - selector_base_correct_count.
            metrics = {
                "selector_loss": selector_loss.detach().float(),
                "selector_correct_count": (
                    (torch.argmax(scores, dim=-1) == row_targets) & scored
                )
                .float()
                .sum(),
                "selector_base_correct_count": (
                    (torch.argmax(unary, dim=-1) == row_targets) & scored
                )
                .float()
                .sum(),
                "selector_token_count": scored.float().sum(),
                "selector_coverage_count": learnable.float().sum(),
                "selector_active_count": torch.tensor(
                    float(active_targets.numel()), dtype=torch.float32, device=device
                ),
            }
        return self.selector_loss_weight * selector_loss, metrics


class DFlash2TrainerBackend(DFlashTrainerBackend):
    # Upstream z-lab DFlash2 checkpoints store the two selector codebooks as bare
    # ``nn.Parameter`` tensors, while this overlay holds them in ``nn.Embedding``
    # modules, whose state dict spells the same tensor with a trailing
    # ``.weight``. Every other DFlash2 parameter name matches upstream exactly.
    # Publish-side inverse: vllm_runtime._dflash2_engine_param_name — keep the
    # two in sync when adding aliases here.
    _CHECKPOINT_KEY_ALIASES = {
        "candidate_selector.predecessor_codebook": (
            "candidate_selector.predecessor_codebook.weight"
        ),
        "candidate_selector.successor_codebook": (
            "candidate_selector.successor_codebook.weight"
        ),
    }

    @property
    def model_type(self):
        return "dflash2"

    def _training_value(self, training_cfg, dflash2_key: str, dflash_key: str, default):
        value = training_cfg.get(dflash2_key, None)
        if value is not None:
            return value
        return training_cfg.get(dflash_key, default)

    def _validate_normalized_state(
        self, draft_model, normalized_state, model_path: str
    ) -> None:
        """Fail loud when a checkpoint carries DFlash2 modules under other names.

        The base loader drops keys the model does not have, and its required-key
        gate covers the DFlash backbone only. An upstream rename would therefore
        leave the convolutions or the selector at their cold-start values, which
        reads as a merely weak drafter rather than as a failed load. A checkpoint
        carrying none of these keys is still accepted: warm-starting DFlash2 from
        a plain DFlash backbone is a legitimate flow, and the DFlash2 modules
        cold-start as an identity passthrough by design.
        """
        expected = {
            key for key in draft_model.state_dict() if _is_dflash2_module_key(key)
        }
        present = expected.intersection(normalized_state)
        stray = {
            key
            for key in normalized_state
            if _is_dflash2_module_key(key) and key not in expected
        }
        if not stray and (not present or present == expected):
            return
        raise ValueError(
            "DFlash2 checkpoint carries only part of the DFlash2 modules under the "
            "expected parameter names, so the rest would silently stay at their "
            f"cold-start values: missing={sorted(expected - present)} "
            f"unrecognized={sorted(stray)} model_path={model_path}. Add the "
            "renamed keys to DFlash2TrainerBackend._CHECKPOINT_KEY_ALIASES."
        )

    def _resolved_block_size(self, drafter_config) -> int:
        """Single source of truth for the block size.

        The convolutions are built from ``drafter_config.block_size`` while the
        training wrapper takes its own ``dflash2_block_size``. If those disagree
        the conv's notion of a block spans several anchor blocks, and its causal
        tap silently reads across an anchor boundary without tripping the
        block-multiple guard. Resolve once and apply to both.
        """
        training_cfg = self.config.rollout.drafter.training
        return int(
            training_cfg.get(
                "dflash2_block_size", getattr(drafter_config, "block_size", 8)
            )
        )

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = self._training_value(
            training_cfg, "dflash2_hidden_size", "dflash_hidden_size", None
        )
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(
            self._training_value(
                training_cfg, "dflash2_num_target_layers", "dflash_num_target_layers", 5
            )
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 36)
        )
        mask_token_id_cfg = self._training_value(
            training_cfg, "dflash2_mask_token_id", "dflash_mask_token_id", None
        )
        mask_token_id = int(
            mask_token_id_cfg
            if mask_token_id_cfg is not None
            else target_text_config.vocab_size - 1
        )
        target_layer_ids = self._training_value(
            training_cfg, "dflash2_target_layer_ids", "dflash_target_layer_ids", None
        )
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        return DFlash2Config(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=int(
                self._training_value(
                    training_cfg,
                    "dflash2_num_hidden_layers",
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
            rope_theta=resolve_rope_theta(target_text_config),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("dflash2_block_size", 8)),
            num_anchors=int(training_cfg.get("dflash2_num_anchors", 512)),
            loss_decay_gamma=float(training_cfg.get("dflash2_loss_decay_gamma", 7.0)),
            conv_kernel_size=int(training_cfg.get("dflash2_conv_kernel_size", 2)),
            conv_group_size=int(training_cfg.get("dflash2_conv_group_size", 16)),
            selector_rank=int(training_cfg.get("dflash2_selector_rank", 256)),
            selector_top_k=int(training_cfg.get("dflash2_selector_top_k", 16)),
            selector_loss_weight=float(
                training_cfg.get("dflash2_selector_loss_weight", 1.0)
            ),
            architectures=["DFlash2DraftModel"],
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
            drafter_config = DFlash2Config.from_dflash2_pretrained(spec_model_path)
            if spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger, spec_model_path, action="Loading DFlash2 drafter weights"
                )
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_dict(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DFlash2Config):
            raise TypeError(
                f"DFlash2 config is not a DFlash2Config: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )
        # Pin the config's block size to the one the trainer will use, so the
        # convolutions and the block layout cannot drift apart.
        block_size = self._resolved_block_size(drafter_config)
        drafter_config.block_size = block_size

        draft_model = DFlash2DraftModel(deepcopy(drafter_config))
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
        return DFlash2TrainingModel(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=int(
                training_cfg.get(
                    "dflash2_num_anchors", getattr(drafter_config, "num_anchors", 512)
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "dflash2_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            selector_loss_weight=float(
                training_cfg.get(
                    "dflash2_selector_loss_weight",
                    getattr(drafter_config, "selector_loss_weight", 1.0),
                )
            ),
        ), drafter_config


__all__ = ["DFlash2TrainerBackend", "DFlash2TrainingModel"]
