# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4-Flash DSpark configuration.

Extends :class:`DSparkConfig` with the DeepSeek-V4-Flash backbone
hyperparameters (latent attention, MoE and mHC), mirroring the released config.
"""

from __future__ import annotations

from verl_speco.models.dspark import DSparkConfig


class DSV4DSparkConfig(DSparkConfig):
    model_type = "dsv4_dspark"

    def __init__(
        self,
        *args,
        dsv4_num_heads: int = 64,
        dsv4_head_dim: int = 512,
        dsv4_rope_head_dim: int = 64,
        dsv4_q_lora_rank: int = 1024,
        dsv4_o_lora_rank: int = 1024,
        dsv4_o_groups: int = 8,
        dsv4_window_size: int = 128,
        dsv4_rope_theta: float = 10000.0,
        dsv4_rope_factor: float = 16.0,
        dsv4_original_seq_len: int = 65536,
        dsv4_beta_fast: float = 32.0,
        dsv4_beta_slow: float = 1.0,
        dsv4_n_routed_experts: int = 256,
        dsv4_n_shared_experts: int = 1,
        dsv4_n_activated_experts: int = 6,
        dsv4_moe_inter_dim: int = 2048,
        dsv4_score_func: str = "sqrtsoftplus",
        dsv4_route_scale: float = 1.5,
        dsv4_swiglu_limit: float = 10.0,
        dsv4_hc_mult: int = 4,
        dsv4_hc_sinkhorn_iters: int = 20,
        dsv4_hc_eps: float = 1e-6,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", None) or ["DSV4DSparkDraftModel"]
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        super().__init__(*args, architectures=architectures, **kwargs)
        self.dsv4_num_heads = int(dsv4_num_heads)
        self.dsv4_head_dim = int(dsv4_head_dim)
        self.dsv4_rope_head_dim = int(dsv4_rope_head_dim)
        self.dsv4_q_lora_rank = int(dsv4_q_lora_rank)
        self.dsv4_o_lora_rank = int(dsv4_o_lora_rank)
        self.dsv4_o_groups = int(dsv4_o_groups)
        self.dsv4_window_size = int(dsv4_window_size)
        self.dsv4_rope_theta = float(dsv4_rope_theta)
        self.dsv4_rope_factor = float(dsv4_rope_factor)
        self.dsv4_original_seq_len = int(dsv4_original_seq_len)
        self.dsv4_beta_fast = float(dsv4_beta_fast)
        self.dsv4_beta_slow = float(dsv4_beta_slow)
        self.dsv4_n_routed_experts = int(dsv4_n_routed_experts)
        self.dsv4_n_shared_experts = int(dsv4_n_shared_experts)
        self.dsv4_n_activated_experts = int(dsv4_n_activated_experts)
        self.dsv4_moe_inter_dim = int(dsv4_moe_inter_dim)
        self.dsv4_score_func = str(dsv4_score_func)
        self.dsv4_route_scale = float(dsv4_route_scale)
        self.dsv4_swiglu_limit = float(dsv4_swiglu_limit)
        self.dsv4_hc_mult = int(dsv4_hc_mult)
        self.dsv4_hc_sinkhorn_iters = int(dsv4_hc_sinkhorn_iters)
        self.dsv4_hc_eps = float(dsv4_hc_eps)

    def to_dict(self) -> dict:
        config = super().to_dict()
        for key in (
            "dsv4_num_heads",
            "dsv4_head_dim",
            "dsv4_rope_head_dim",
            "dsv4_q_lora_rank",
            "dsv4_o_lora_rank",
            "dsv4_o_groups",
            "dsv4_window_size",
            "dsv4_rope_theta",
            "dsv4_rope_factor",
            "dsv4_original_seq_len",
            "dsv4_beta_fast",
            "dsv4_beta_slow",
            "dsv4_n_routed_experts",
            "dsv4_n_shared_experts",
            "dsv4_n_activated_experts",
            "dsv4_moe_inter_dim",
            "dsv4_score_func",
            "dsv4_route_scale",
            "dsv4_swiglu_limit",
            "dsv4_hc_mult",
            "dsv4_hc_sinkhorn_iters",
            "dsv4_hc_eps",
        ):
            config[key] = getattr(self, key)
        return config

    @classmethod
    def from_dsv4_dspark_dict(cls, config: dict) -> DSV4DSparkConfig:
        from copy import deepcopy

        source_config = deepcopy(config)
        internal_config = deepcopy(config)
        internal_config["model_type"] = cls.model_type
        # The released checkpoint records the noise token as
        # ``dspark_noise_token_id``; map it onto ``mask_token_id`` so the draft
        # model uses the correct noise embedding (128799 for DeepSeek-V4-Flash).
        if internal_config.get("mask_token_id") is None:
            noise_token_id = internal_config.get("dspark_noise_token_id")
            if noise_token_id is None:
                noise_token_id = internal_config.get("dsv4_dspark_mask_token_id")
            if noise_token_id is not None:
                internal_config["mask_token_id"] = int(noise_token_id)
        # The released checkpoint is the target config plus a ``dspark_*``/``hc_*``
        # draft section. Consume those draft keys (target_layer_ids must be
        # explicit so a variant selection is not silently replaced by a default).
        draft_key_sources = {
            "target_layer_ids": (
                "dspark_target_layer_ids",
                "aux_hidden_state_layer_ids",
            ),
            "block_size": ("dspark_block_size",),
            "markov_rank": ("dspark_markov_rank",),
            "dsv4_hc_mult": ("hc_mult",),
            "dsv4_hc_sinkhorn_iters": ("hc_sinkhorn_iters",),
            "dsv4_hc_eps": ("hc_eps",),
        }
        for target_key, source_keys in draft_key_sources.items():
            if internal_config.get(target_key) is not None:
                continue
            for source_key in source_keys:
                value = internal_config.get(source_key)
                if value is not None:
                    internal_config[target_key] = value
                    break
        if "enable_confidence_head" not in internal_config:
            internal_config["enable_confidence_head"] = (
                float(internal_config.get("confidence_head_alpha", 0.0)) > 0.0
            )
        # The released config is the *target* config, which may declare sliding
        # window attention (``sliding_window``/``layer_types``). The DSV4 draft
        # backbone is dense sink attention and only supports a single dense mask,
        # so force sliding-window off instead of inheriting a mask-list layout
        # that ``_mask_to_bias`` would reject at forward time.
        internal_config["use_sliding_window"] = False
        loaded = cls.from_dict(internal_config)
        loaded._source_checkpoint_config = source_config
        return loaded

    @classmethod
    def from_dsv4_dspark_pretrained(cls, model_path: str) -> DSV4DSparkConfig:
        import json
        import os

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls.from_dsv4_dspark_dict(config)
