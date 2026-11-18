# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Adapted from AIRDrafter training_common/modeling/ds4_mtp.py.
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from verl_speco.models.deepseek_v4.modeling_dspark import (
    _install_dspark_dsa_query_training_replay,
    _install_dspark_training_safe_rope,
)
from verl_speco.models.deepseek_v4.runtime import (
    DeepseekV4ServingTrainingWrapper,
)


class DeepseekV4MTP(nn.Module):
    """Trainable DeepSeek V4 MTP backed by the serving runtime."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        layout = getattr(config, "ds4_layout", "vllm_ascend")
        if layout != "vllm_ascend":
            raise ValueError(
                "DeepSeek V4 training supports only ds4_layout='vllm_ascend'"
            )
        implementation = getattr(config, "ds4_mtp_impl", "vllm_import_train")
        if implementation != "vllm_import_train":
            raise ValueError(
                "DeepSeek V4 training supports only ds4_mtp_impl='vllm_import_train'"
            )
        self.config = config
        self.ds4_layout = layout
        self.ds4_mtp_impl = implementation
        self.model = DeepseekV4ServingTrainingWrapper(config)
        # MTP and DSpark use the same native DSA prolog. Its inference-only
        # in-place operators cut the Q projection graph. Reuse the established
        # adapters: forward returns native values, backward follows replay.
        self.using_training_safe_rope = _install_dspark_training_safe_rope()
        self.using_training_dsa_query_replay = (
            _install_dspark_dsa_query_training_replay()
        )
        if not self.using_training_dsa_query_replay:
            raise RuntimeError(
                "The installed vLLM-Ascend DSA runtime exposes no supported "
                "MLA prolog for MTP query-gradient replay"
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def runtime_device(self) -> torch.device:
        return self.model.runtime_device()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.compute_logits(
            hidden_states,
            lm_head,
            spec_step_idx=spec_step_idx,
        )

    def compute_local_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.compute_local_logits(
            hidden_states,
            spec_step_idx=spec_step_idx,
        )

    @torch.no_grad()
    def load_weights(self, path: str, mode: str = "full") -> dict[str, Any]:
        return self.model.load_weights(path, mode=mode)

    def freeze_params(self, cfg: Any) -> None:
        explicitly_frozen = set(cfg.train.params_to_freeze)
        shared_tokens = ("embed_tokens", "shared_head", "external_lm_head")
        for name, param in self.named_parameters():
            should_train = (
                param.is_floating_point()
                and not any(token in name for token in shared_tokens)
                and not name.endswith("e_score_correction_bias")
                and name not in explicitly_frozen
            )
            param.requires_grad_(should_train)

        unexpected_frozen = [
            name
            for name, param in self.named_parameters()
            if param.is_floating_point()
            and not any(token in name for token in shared_tokens)
            and not name.endswith("e_score_correction_bias")
            and name not in explicitly_frozen
            and not param.requires_grad
        ]
        if unexpected_frozen:
            raise RuntimeError(
                "Non-shared MTP parameters remain frozen: "
                + ", ".join(unexpected_frozen)
            )
        if self.model.dequantize_mtp_for_training:
            remaining_integer_weights = [
                name
                for name, param in self.named_parameters()
                if not param.is_floating_point()
                and any(token in name for token in ("weight", "w13", "w2"))
            ]
            if remaining_integer_weights:
                raise RuntimeError(
                    "Quantized MTP weights remain after BF16 training conversion: "
                    + ", ".join(remaining_integer_weights)
                )
