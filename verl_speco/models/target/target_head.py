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

import glob
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn


def _load_checkpoint_tensor(model_path: str, key: str) -> torch.Tensor:
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    index_paths = glob.glob(os.path.join(model_path, "*.index.json"))
    if len(index_paths) > 1:
        raise FileNotFoundError(f"Multiple index.json files found in {model_path}")

    if index_paths:
        with open(index_paths[0], encoding="utf-8") as f:
            index_json = json.load(f)
        weight_map = index_json.get("weight_map", {})
        if key not in weight_map:
            raise KeyError(
                f"Tensor {key!r} is not present in checkpoint index for {model_path}"
            )
        ckpt_file = os.path.join(model_path, weight_map[key])
        if ckpt_file.endswith(".safetensors"):
            with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        return torch.load(ckpt_file, map_location="cpu", weights_only=True)[key]

    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)[key]

    raise FileNotFoundError(
        f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
    )


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
