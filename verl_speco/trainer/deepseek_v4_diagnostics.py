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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/gradient_audit.py.
import torch


def dspark_parameter_audit_group(name: str) -> str:
    """Return a stable layer/family key for DSpark gradient diagnostics."""

    layer = "wrapper"
    layer_marker = ".model.layers."
    if layer_marker in name:
        layer = name.split(layer_marker, 1)[1].split(".", 1)[0]

    if ".main_proj." in name:
        family = "main_projection"
    elif "confidence_head" in name:
        family = "confidence_head"
    elif ".markov_head." in name:
        family = "markov_head"
    elif ".hc_" in name:
        family = "hyperconnection"
    elif ".self_attn.attn_sink" in name:
        family = "attention_sink"
    elif ".self_attn.wq" in name or ".self_attn.q_norm" in name:
        family = "attention_query"
    elif ".self_attn.wkv" in name or ".self_attn.kv_norm" in name:
        family = "attention_kv"
    elif ".self_attn.wo" in name:
        family = "attention_output"
    elif ".mlp.gate." in name:
        family = "moe_router"
    elif ".mlp.shared_experts." in name:
        family = "moe_shared"
    elif ".mlp.experts.routed_experts." in name:
        family = "moe_routed"
    elif "layernorm" in name or name.endswith(".norm.weight"):
        family = "normalization"
    elif ".self_attn." in name:
        family = "attention_other"
    elif ".mlp." in name:
        family = "mlp_other"
    else:
        family = "other"
    return f"layer_{layer}/{family}"


def capture_dspark_gradient_probe(
    gradient: torch.Tensor,
    *,
    chunk_elements: int = 16 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture finite/max/index scalars without a full-size abs temporary."""

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    flat = gradient.detach().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("Cannot audit an empty DSpark gradient")
    max_abs = torch.zeros((), dtype=torch.float32, device=flat.device)
    max_index = torch.zeros((), dtype=torch.long, device=flat.device)
    finite = torch.ones((), dtype=torch.bool, device=flat.device)
    for start in range(0, flat.numel(), chunk_elements):
        end = min(start + chunk_elements, flat.numel())
        chunk = flat[start:end]
        chunk_abs = chunk.abs()
        chunk_max, chunk_index = chunk_abs.max(dim=0)
        chunk_max = chunk_max.float()
        replace = chunk_max > max_abs
        max_abs = torch.maximum(max_abs, chunk_max)
        max_index = torch.where(
            replace,
            chunk_index.to(torch.long) + start,
            max_index,
        )
        finite = torch.logical_and(finite, torch.isfinite(chunk).all())
    return finite, max_abs, max_index
