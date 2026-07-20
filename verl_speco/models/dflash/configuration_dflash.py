import json
import os
from typing import Optional

from transformers import PretrainedConfig


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
        rope_theta: float = 10000.0,
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
        self.rope_theta = rope_theta
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
