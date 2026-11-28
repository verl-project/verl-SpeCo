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
"""LiLiCorr configuration (arXiv:2608.20530).

LiLiCorr binds the DFlash backbone and adds a candidate-lattice head that
rescores the backbone's per-slot top-k candidates. The serving contract lives in
vLLM's ``vllm/model_executor/models/lilicorr.py``; the training-side reference
is speculators' ``models/lilicorr/{config,model_definitions}.py`` (commit
d8c73a88).

Geometry is config-driven: the released Qwen3.8-27B LiLiCorr checkpoint and the
Qwen3-8B paper recipe differ in block size, aux layer ids, decay gamma and vocab,
so nothing here may be hardcoded to one target.
"""

from __future__ import annotations

import json
import os

from verl_speco.models.dflash import DFlashConfig

# Keys the LiLiCorr checkpoints nest under ``dflash_config`` instead of putting at
# the config top level (same handling DFlash2 uses). ``from_lilicorr_pretrained``
# lifts them so the rest of the overlay can read them as plain attributes.
_NESTED_LILICORR_KEYS = (
    "block_size",
    "mask_token_id",
    "target_layer_ids",
    "sample_from_anchor",
    "causal",
    "lilicorr_candidate_topk",
    "lilicorr_hidden_size",
    "lilicorr_factor_dim",
    "lilicorr_num_layers",
    "lilicorr_num_heads",
    "lilicorr_mlp_ratio",
    "lilicorr_logit_scale",
    "lilicorr_vector_eps",
)


class LiLiCorrConfig(DFlashConfig):
    """DFlash config plus the LiLiCorr candidate-lattice head geometry.

    Defaults are the head geometry of the paper's main model: K=8, width 1024,
    2 layers, 8 heads, cosine factors scaled by a fixed 8.0. All DFlash fields
    are inherited unchanged.
    """

    model_type = "lilicorr"

    def __init__(
        self,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        lilicorr_candidate_topk: int = 8,
        lilicorr_hidden_size: int | None = 1024,
        lilicorr_factor_dim: int = 1024,
        lilicorr_num_layers: int = 2,
        lilicorr_num_heads: int = 8,
        lilicorr_mlp_ratio: float = 2.0,
        lilicorr_logit_scale: float = 8.0,
        lilicorr_vector_eps: float = 1e-4,
        sample_from_anchor: bool = False,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = float(loss_decay_gamma)
        self.lilicorr_candidate_topk = int(lilicorr_candidate_topk)
        self.lilicorr_hidden_size = (
            None if lilicorr_hidden_size is None else int(lilicorr_hidden_size)
        )
        self.lilicorr_factor_dim = int(lilicorr_factor_dim)
        self.lilicorr_num_layers = int(lilicorr_num_layers)
        self.lilicorr_num_heads = int(lilicorr_num_heads)
        self.lilicorr_mlp_ratio = float(lilicorr_mlp_ratio)
        self.lilicorr_logit_scale = float(lilicorr_logit_scale)
        self.lilicorr_vector_eps = float(lilicorr_vector_eps)
        self.sample_from_anchor = bool(sample_from_anchor)
        self.causal = bool(causal)
        self._validate()

    def _validate(self) -> None:
        topk = self.lilicorr_candidate_topk
        if topk < 1 or topk > 16 or topk & (topk - 1):
            raise ValueError("lilicorr_candidate_topk must be a power of two <= 16.")
        if self.block_size < 2:
            raise ValueError(
                f"LiLiCorr needs block_size >= 2 (block-1 speculative slots), "
                f"got {self.block_size}"
            )
        head_width = self.lilicorr_hidden_size or self.hidden_size
        if head_width % self.lilicorr_num_heads != 0:
            raise ValueError(
                f"lilicorr_hidden_size={head_width} must be divisible by "
                f"lilicorr_num_heads={self.lilicorr_num_heads}."
            )
        if self.sample_from_anchor:
            raise ValueError("LiLiCorr requires sample_from_anchor=False.")
        if getattr(self, "draft_vocab_size", None) not in (None, self.vocab_size):
            raise ValueError(
                "LiLiCorr requires the full verifier vocabulary (vLLM cannot serve "
                "a reduced draft vocab); omit draft_vocab_size."
            )

    @classmethod
    def from_lilicorr_pretrained(cls, model_path: str) -> LiLiCorrConfig:
        """Load a LiLiCorr checkpoint's config.

        Accepts both the released layout (LiLiCorr knobs nested under
        ``dflash_config``) and the layout this overlay writes (top level wins).
        """
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        nested = config.get("dflash_config") or {}
        for key in _NESTED_LILICORR_KEYS:
            if key not in config and key in nested:
                config[key] = nested[key]

        config["model_type"] = cls.model_type
        config["architectures"] = ["LiLiCorrDraftModel"]
        # The serving head reads its geometry from the flat config fields; keep
        # the nested block in sync so a checkpoint round-trips to vLLM.
        merged = dict(config)
        merged["dflash_config"] = {
            **nested,
            **{key: config[key] for key in _NESTED_LILICORR_KEYS if key in config},
        }
        return cls.from_dict(merged)


__all__ = ["LiLiCorrConfig"]
