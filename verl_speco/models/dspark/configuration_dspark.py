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

import json
import os
from copy import deepcopy
from typing import Optional

from verl_speco.models.dflash import DFlashConfig

# Speculators snapshots nest the transformer architecture under
# ``transformer_layer_config``; mirror ``convert_speculators_dflash2`` and copy
# these keys into the flat config used to build the draft model.
_SPECULATOR_TRANSFORMER_KEYS = (
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "rope_parameters",
    "vocab_size",
    # MLA (GLM-5 / DeepSeek-style) draft topology.
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rope_interleave",
    "draft_attn",
)

# Sliding-window attention knobs live in the same nested config. Lift them too
# so a checkpoint configured with SWA does not silently fall back to full
# attention (mirrors ``convert_speculators_dflash2._SLIDING_WINDOW_KEYS``).
_SPECULATOR_SLIDING_WINDOW_KEYS = (
    "layer_types",
    "sliding_window",
    "use_sliding_window",
)


class DSparkConfig(DFlashConfig):
    """Configuration for the DSpark draft model.

    DSpark uses the same target-hidden-state context backbone as DFlash and adds
    a Markov head plus an optional confidence head.
    """

    model_type = "dspark"

    def __init__(
        self,
        *args,
        block_size: int = 7,
        num_anchors: int = 512,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: Optional[bool] = None,
        confidence_head_alpha: float = 0.0,
        confidence_head_with_markov: bool = True,
        confidence_detach_backbone: bool = False,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        loss_decay_gamma: float = 7.0,
        draft_attn: str = "gqa",
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        rope_interleave: bool = True,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", None)
        self._source_checkpoint_config = None
        super().__init__(*args, **kwargs)
        self.draft_attn = str(draft_attn)
        if self.draft_attn not in ("gqa", "mla"):
            raise ValueError(
                f"draft_attn must be 'gqa' or 'mla', got {self.draft_attn!r}"
            )
        self.q_lora_rank = int(q_lora_rank) if q_lora_rank is not None else None
        self.kv_lora_rank = int(kv_lora_rank) if kv_lora_rank is not None else None
        self.qk_nope_head_dim = (
            int(qk_nope_head_dim) if qk_nope_head_dim is not None else None
        )
        self.qk_rope_head_dim = (
            int(qk_rope_head_dim) if qk_rope_head_dim is not None else None
        )
        self.v_head_dim = int(v_head_dim) if v_head_dim is not None else None
        self.rope_interleave = bool(rope_interleave)
        if self.draft_attn == "mla":
            self._validate_mla_topology()
        # Native vLLM MRV2 dispatches Qwen DSpark by this architecture name.
        # ``DSparkDraftModel`` is registered to the unrelated DeepSeek-V4
        # implementation in current vLLM and is not a safe generic fallback.
        # The GLM-5 MLA draft is the one exception: it is registered under
        # ``glm5_dspark`` on the serving side and keyed by ``DSparkDraftModel``.
        default_architectures = (
            ["DSparkDraftModel"] if self.draft_attn == "mla" else ["Qwen3DSparkModel"]
        )
        self.architectures = architectures or default_architectures
        self.block_size = int(block_size)
        self.num_anchors = int(num_anchors)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = str(markov_head_type)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.enable_confidence_head = (
            bool(enable_confidence_head)
            if enable_confidence_head is not None
            else self.confidence_head_alpha > 0.0
        )
        self.confidence_head_with_markov = bool(confidence_head_with_markov)
        self.confidence_detach_backbone = bool(confidence_detach_backbone)
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.loss_decay_gamma = float(loss_decay_gamma)

    def _validate_mla_topology(self) -> None:
        """Validate the DeepSeek-style MLA fields and normalize derived values."""
        required = {
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "draft_attn='mla' requires "
                + ", ".join(f"{name}" for name in missing)
                + " in the config"
            )
        if self.kv_lora_rank < 1:
            raise ValueError(f"kv_lora_rank must be positive, got {self.kv_lora_rank}")
        if self.qk_nope_head_dim < 1 or self.qk_rope_head_dim < 1:
            raise ValueError(
                "qk_nope_head_dim and qk_rope_head_dim must be positive, got "
                f"{self.qk_nope_head_dim} and {self.qk_rope_head_dim}"
            )
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError(
                "qk_rope_head_dim must be even for interleaved rotary, got "
                f"{self.qk_rope_head_dim}"
            )
        if self.v_head_dim < 1:
            raise ValueError(f"v_head_dim must be positive, got {self.v_head_dim}")
        if self.use_sliding_window or (
            self.layer_types is not None and "sliding_attention" in self.layer_types
        ):
            raise ValueError(
                "draft_attn='mla' does not support sliding-window attention"
            )
        # MLA expands the latent KV back to every query head and rotary only
        # covers the rope channels, so ``head_dim`` follows the upstream
        # ``qk_rope_head_dim`` convention.
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.qk_rope_head_dim

    def to_dict(self) -> dict:
        """Preserve source fields while exporting the active confidence topology."""
        config = super().to_dict()
        confidence_config = {
            "enable_confidence_head": self.enable_confidence_head,
            "confidence_head_alpha": self.confidence_head_alpha,
            "confidence_head_with_markov": self.confidence_head_with_markov,
        }
        source_config = config.pop("_source_checkpoint_config", None)
        if source_config is not None:
            config.update(deepcopy(source_config))
        # A source checkpoint may predate the confidence head. Training can add the
        # head later, so these live values must win over the preserved source JSON.
        config.update(confidence_config)
        # The MLA draft is exported under the architecture name the GLM-5
        # serving stack dispatches on, even when the source snapshot used the
        # generic DFlash/DSpark spellings.
        if self.draft_attn == "mla":
            config["architectures"] = ["DSparkDraftModel"]
        return config

    def to_diff_dict(self) -> dict:
        if self._source_checkpoint_config is not None:
            return self.to_dict()
        return super().to_diff_dict()

    @classmethod
    def from_dspark_dict(cls, config: dict) -> "DSparkConfig":
        source_config = deepcopy(config)
        internal_config = deepcopy(config)
        source_model_type = internal_config.get("model_type")
        internal_config["model_type"] = cls.model_type
        # Speculators snapshots nest the transformer architecture under
        # ``transformer_layer_config``; normalize it before building the config
        # so the released draft checkpoint matches the constructed model.
        transformer = internal_config.pop("transformer_layer_config", None)
        transformer_model_type = None
        if isinstance(transformer, dict):
            transformer_model_type = transformer.get("model_type")
            for key in _SPECULATOR_TRANSFORMER_KEYS + _SPECULATOR_SLIDING_WINDOW_KEYS:
                value = transformer.get(key)
                if value is not None:
                    internal_config[key] = value
            rope_parameters = transformer.get("rope_parameters")
            if (
                internal_config.get("rope_theta") is None
                and isinstance(rope_parameters, dict)
                and rope_parameters.get("rope_theta") is not None
            ):
                internal_config["rope_theta"] = rope_parameters["rope_theta"]
        if internal_config.get("draft_attn") is None:
            is_mla = (
                source_model_type == "glm5_dspark"
                or transformer_model_type == "glm5_dspark"
                or (
                    internal_config.get("kv_lora_rank") is not None
                    and internal_config.get("qk_rope_head_dim") is not None
                )
            )
            internal_config["draft_attn"] = "mla" if is_mla else "gqa"
        if internal_config["draft_attn"] == "mla":
            internal_config["architectures"] = ["DSparkDraftModel"]
        # Released speculators checkpoints record the auxiliary context layers
        # under ``aux_hidden_state_layer_ids`` (EAGLE ``output_hidden_states``
        # indices). That is the same indexing SpeCo uses for the training-side
        # ``target_layer_ids``, so lift the list verbatim; otherwise
        # ``_normalize_dflash_config`` silently substitutes an evenly spaced set
        # that does not match the pretrained ``fc`` projection.
        aux_layer_ids = internal_config.get("aux_hidden_state_layer_ids")
        if internal_config.get("target_layer_ids") is None and aux_layer_ids:
            internal_config["target_layer_ids"] = [
                int(layer_id) for layer_id in aux_layer_ids
            ]
        if "enable_confidence_head" not in internal_config:
            internal_config["enable_confidence_head"] = (
                float(internal_config.get("confidence_head_alpha", 0.0)) > 0.0
            )
        loaded = cls.from_dict(internal_config)
        loaded._source_checkpoint_config = source_config
        return loaded

    @classmethod
    def from_dspark_pretrained(cls, model_path: str) -> "DSparkConfig":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls.from_dspark_dict(config)
