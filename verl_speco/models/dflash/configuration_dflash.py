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
import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

DEFAULT_ROPE_THETA = 10000.0

# rope_type values that need no scaling beyond the base.
_UNSCALED_ROPE_TYPES = frozenset({"default", ""})

# rope_type values already warned about, so a per-layer rotary build does not
# repeat the same line once per decoder layer.
_WARNED_ROPE_TYPES: set[str] = set()


def _lookup(source: Any, key: str):
    """Read ``key`` from either a mapping or a config object."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _warn_once_if_scaling_is_ignored(source: Any, rope_parameters: Any) -> None:
    """Warn when the config asks for RoPE scaling the DFlash draft cannot apply.

    ``DFlashRotaryEmbedding`` takes only a base, so a target using yarn, linear
    or llama3 scaling gets a draft whose rotary phase diverges from it past the
    original context length. That is invisible for the same reason a defaulted
    base is, so say it once per distinct ``rope_type``.
    """
    rope_type = _lookup(rope_parameters, "rope_type")
    if rope_type is None:
        rope_scaling = _lookup(source, "rope_scaling")
        rope_type = _lookup(rope_scaling, "rope_type") or _lookup(rope_scaling, "type")
    if rope_type is None:
        return
    rope_type = str(rope_type)
    if rope_type in _UNSCALED_ROPE_TYPES or rope_type in _WARNED_ROPE_TYPES:
        return
    _WARNED_ROPE_TYPES.add(rope_type)
    logger.warning(
        "DFlash draft RoPE ignores rope_type=%r: the draft rotary applies the base only, "
        "so its phase diverges from a target using scaled RoPE beyond the original "
        "context length.",
        rope_type,
    )


def resolve_rope_theta(source: Any, default: float = DEFAULT_ROPE_THETA) -> float:
    """Resolve the RoPE base from a config that may nest it under ``rope_parameters``.

    transformers 5 moved the RoPE base out of a top-level ``rope_theta`` and into
    a ``rope_parameters`` dict. Released DFlash-family drafter checkpoints and
    modern target configs carry only the nested spelling, so reading
    ``rope_theta`` alone silently falls back to ``default``, which is three orders
    of magnitude off for a model trained at 1e7. A top-level value still wins, so
    a config this overlay wrote itself round-trips unchanged.
    ``verl_speco.integration.sglang_patch`` bridges the same split on the serving
    side.

    Args:
        source: A config object or a raw config mapping.
        default: RoPE base to use when neither spelling carries one.

    Returns:
        float: The resolved RoPE base.
    """
    rope_parameters = _lookup(source, "rope_parameters")
    _warn_once_if_scaling_is_ignored(source, rope_parameters)
    top_level = _lookup(source, "rope_theta")
    if top_level is not None:
        return float(top_level)
    nested = _lookup(rope_parameters, "rope_theta")
    if nested is not None:
        return float(nested)
    return float(default)


class DFlashConfig(PretrainedConfig):
    """Configuration for the DFlash draft model.

    DFlash consumes selected target hidden-state layers. ``num_target_layers``
    follows the upstream DFlash meaning: the total number of layers in the
    target model. ``num_context_layers`` is the number of selected target hidden
    states concatenated before the context projection.
    """

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: Optional[float] = None,
        num_target_layers: int = 36,
        num_context_layers: Optional[int] = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[list[int]] = None,
        mask_token_id: int = 151669,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        # ``modeling_dflash`` reads ``rope_theta`` directly, so keep it a plain
        # attribute, but accept the transformers-5 ``rope_parameters`` spelling
        # the released checkpoints use.
        self.rope_theta = resolve_rope_theta(
            {"rope_theta": rope_theta, "rope_parameters": kwargs.get("rope_parameters")}
        )
        self.num_target_layers = num_target_layers
        self.num_context_layers = num_context_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.mask_token_id = mask_token_id

    @classmethod
    def from_dflash_pretrained(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        architectures = config.get("architectures") or []
        if "DFlashForCausalLM" in architectures or config.get("model_type") == "qwen3":
            config["model_type"] = cls.model_type
            config["architectures"] = ["DFlashForCausalLM"]
        return cls.from_dict(config)
