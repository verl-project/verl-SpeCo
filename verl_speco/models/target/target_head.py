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
"""Minimal target lm-head loader for SPECO drafter training."""

import torch
from torch import nn

from verl_speco.checkpoint_tensor import _load_checkpoint_tensor


class TargetHead(nn.Module):
    """Frozen linear target head loaded from a Hugging Face checkpoint."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(
                f"TargetHead weight must be rank-2, got shape={tuple(weight.shape)}"
            )
        vocab_size, hidden_size = weight.shape
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(weight.detach().to(dtype=self.fc.weight.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        lm_head_key: str = "lm_head.weight",
        tied_embedding_key: str = "model.embed_tokens.weight",
    ) -> "TargetHead":
        try:
            weight = _load_checkpoint_tensor(model_path, lm_head_key)
        except KeyError:
            weight = _load_checkpoint_tensor(model_path, tied_embedding_key)
        return cls(weight)
