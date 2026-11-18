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

# Adapted from loading_trained_mtp/loading_ds4.py.
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

LAYER_PREFIX = "model.layers.0."


FICLONE = 0x40049409


DEDICATED_MTP_EMBEDDING_KEY = "mtp.0.emb.tok_emb.weight"


SHARED_MODEL_EMBEDDING_KEYS = ("embed.weight",)


VLLM_ASCEND_REQUIRED_TRAINABLE_SOURCES = {
    f"{LAYER_PREFIX}hc_head_fn",
    f"{LAYER_PREFIX}hc_head_base",
    f"{LAYER_PREFIX}hc_head_scale",
    f"{LAYER_PREFIX}e_proj.weight",
    f"{LAYER_PREFIX}h_proj.weight",
    f"{LAYER_PREFIX}enorm.weight",
    f"{LAYER_PREFIX}hnorm.weight",
    f"{LAYER_PREFIX}mtp_block.hc_attn_fn",
    f"{LAYER_PREFIX}mtp_block.hc_attn_base",
    f"{LAYER_PREFIX}mtp_block.hc_attn_scale",
    f"{LAYER_PREFIX}mtp_block.hc_ffn_fn",
    f"{LAYER_PREFIX}mtp_block.hc_ffn_base",
    f"{LAYER_PREFIX}mtp_block.hc_ffn_scale",
    f"{LAYER_PREFIX}mtp_block.self_attn.attn_sink",
    f"{LAYER_PREFIX}mtp_block.self_attn.wq_a.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.q_norm.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.wq_b.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.wkv.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.kv_norm.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.wo_a.weight",
    f"{LAYER_PREFIX}mtp_block.self_attn.wo_b.weight",
    f"{LAYER_PREFIX}mtp_block.mlp.gate.weight",
    f"{LAYER_PREFIX}mtp_block.mlp.shared_experts.gate_up_proj.weight",
    f"{LAYER_PREFIX}mtp_block.mlp.shared_experts.down_proj.weight",
    f"{LAYER_PREFIX}mtp_block.mlp.experts.w13_weight",
    f"{LAYER_PREFIX}mtp_block.mlp.experts.w2_weight",
    f"{LAYER_PREFIX}mtp_block.input_layernorm.weight",
    f"{LAYER_PREFIX}mtp_block.post_attention_layernorm.weight",
}


def _canonical_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize wrapper and vLLM runtime container names for export."""
    canonical = {}
    for name, value in state_dict.items():
        key = name
        while key.startswith("model.model."):
            key = key[len("model.") :]
        # vLLM 0.26 changed FusedMoE from a parameter-owning module into a
        # MoERunner whose RoutedExperts child owns the weights. Keep the
        # export contract independent of that runtime-only container level.
        key = key.replace(
            ".mtp_block.mlp.experts.routed_experts.",
            ".mtp_block.mlp.experts.",
        )
        key = key.replace(
            ".mlp.experts.routed_experts.",
            ".mlp.experts.",
        )
        canonical[key] = value
    return canonical


def _first_tensor(
    state_dict: dict[str, torch.Tensor],
    *names: str,
) -> torch.Tensor | None:
    for name in names:
        value = state_dict.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _export_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Keep the trained dtype; HC and attention-sink parameters are FP32."""
    return tensor.detach().contiguous()


def _build_vllm_ascend_export_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    missing_sources = sorted(VLLM_ASCEND_REQUIRED_TRAINABLE_SOURCES - state_dict.keys())
    if missing_sources:
        raise KeyError(
            "Training checkpoint is missing DS4 trainable tensors: "
            + ", ".join(missing_sources)
        )
    mapping = {
        "model.layers.0.hc_head_fn": "mtp.0.hc_head_fn",
        "model.layers.0.hc_head_base": "mtp.0.hc_head_base",
        "model.layers.0.hc_head_scale": "mtp.0.hc_head_scale",
        "model.embed_tokens.weight": "mtp.0.emb.tok_emb.weight",
        "model.layers.0.enorm.weight": "mtp.0.enorm.weight",
        "model.layers.0.hnorm.weight": "mtp.0.hnorm.weight",
        "model.layers.0.e_proj.weight": "mtp.0.e_proj.weight",
        "model.layers.0.h_proj.weight": "mtp.0.h_proj.weight",
        "model.layers.0.shared_head.norm.weight": "mtp.0.norm.weight",
        "model.layers.0.mtp_block.hc_attn_fn": "mtp.0.hc_attn_fn",
        "model.layers.0.mtp_block.hc_attn_base": "mtp.0.hc_attn_base",
        "model.layers.0.mtp_block.hc_attn_scale": "mtp.0.hc_attn_scale",
        "model.layers.0.mtp_block.hc_ffn_fn": "mtp.0.hc_ffn_fn",
        "model.layers.0.mtp_block.hc_ffn_base": "mtp.0.hc_ffn_base",
        "model.layers.0.mtp_block.hc_ffn_scale": "mtp.0.hc_ffn_scale",
        "model.layers.0.mtp_block.self_attn.attn_sink": "mtp.0.attn.attn_sink",
        "model.layers.0.mtp_block.self_attn.wq_a.weight": "mtp.0.attn.wq_a.weight",
        "model.layers.0.mtp_block.self_attn.q_norm.weight": "mtp.0.attn.q_norm.weight",
        "model.layers.0.mtp_block.self_attn.wq_b.weight": "mtp.0.attn.wq_b.weight",
        "model.layers.0.mtp_block.self_attn.wkv.weight": "mtp.0.attn.wkv.weight",
        "model.layers.0.mtp_block.self_attn.kv_norm.weight": "mtp.0.attn.kv_norm.weight",
        "model.layers.0.mtp_block.self_attn.wo_a.weight": "mtp.0.attn.wo_a.weight",
        "model.layers.0.mtp_block.self_attn.wo_b.weight": "mtp.0.attn.wo_b.weight",
        "model.layers.0.mtp_block.mlp.gate.weight": "mtp.0.ffn.gate.weight",
        "model.layers.0.mtp_block.mlp.gate.e_score_correction_bias": "mtp.0.ffn.gate.bias",
        "model.layers.0.mtp_block.mlp.shared_experts.down_proj.weight": "mtp.0.ffn.shared_experts.w2.weight",
        "model.layers.0.mtp_block.input_layernorm.weight": "mtp.0.attn_norm.weight",
        "model.layers.0.mtp_block.post_attention_layernorm.weight": "mtp.0.ffn_norm.weight",
    }

    new_weights = {
        target: _export_tensor(state_dict[source])
        for source, target in mapping.items()
        if source in state_dict
    }

    wo_a = state_dict[f"{LAYER_PREFIX}mtp_block.self_attn.wo_a.weight"]
    if wo_a.ndim == 3:
        # Runtime grouped input-major [groups, in, out_per_group] ->
        # checkpoint output-major [groups * out_per_group, in].
        new_weights["mtp.0.attn.wo_a.weight"] = _export_tensor(
            wo_a.permute(0, 2, 1).reshape(-1, wo_a.shape[1])
        )
    elif wo_a.ndim != 2:
        raise ValueError(f"Unexpected wo_a shape: {tuple(wo_a.shape)}")

    shared_gate_up = _first_tensor(
        state_dict,
        f"{LAYER_PREFIX}mtp_block.mlp.shared_experts.gate_up_proj.weight",
    )
    if shared_gate_up is not None:
        if shared_gate_up.shape[0] % 2 != 0:
            raise ValueError(
                "Shared gate_up_proj output dimension must be even, got "
                f"{tuple(shared_gate_up.shape)}"
            )
        shared_w1, shared_w3 = shared_gate_up.chunk(2, dim=0)
        new_weights["mtp.0.ffn.shared_experts.w1.weight"] = _export_tensor(shared_w1)
        new_weights["mtp.0.ffn.shared_experts.w3.weight"] = _export_tensor(shared_w3)

    w13 = _first_tensor(
        state_dict,
        f"{LAYER_PREFIX}mtp_block.mlp.experts.w13_weight",
    )
    w2 = _first_tensor(
        state_dict,
        f"{LAYER_PREFIX}mtp_block.mlp.experts.w2_weight",
    )
    if (w13 is None) != (w2 is None):
        raise KeyError("Routed MoE export requires both w13_weight and w2_weight")
    if w13 is not None and w2 is not None:
        if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
            raise ValueError(
                f"Unexpected routed MoE shapes: w13={tuple(w13.shape)}, "
                f"w2={tuple(w2.shape)}"
            )
        router_weight = state_dict[f"{LAYER_PREFIX}mtp_block.mlp.gate.weight"]
        if router_weight.ndim != 2 or router_weight.shape[0] != w13.shape[0]:
            raise ValueError(
                "Routed expert count does not match router output dimension: "
                f"router={tuple(router_weight.shape)}, w13={tuple(w13.shape)}"
            )

        # vLLM-Ascend postprocessing stores MoE matrices input-major:
        # w13=[expert, hidden, 2*intermediate], w2=[expert, intermediate, hidden].
        if w13.shape[1] == w2.shape[2] and w13.shape[2] == 2 * w2.shape[1]:
            w1_input_major, w3_input_major = w13.chunk(2, dim=2)
            w1 = w1_input_major.transpose(1, 2)
            w3 = w3_input_major.transpose(1, 2)
            w2_output_major = w2.transpose(1, 2)
        elif w13.shape[1] == 2 * w2.shape[2] and w13.shape[2] == w2.shape[1]:
            w1, w3 = w13.chunk(2, dim=1)
            w2_output_major = w2
        else:
            raise ValueError(
                "Cannot determine routed MoE matrix layout from "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
            )

        for expert_index in tqdm(range(w13.shape[0]), desc="Preparing experts"):
            new_weights[f"mtp.0.ffn.experts.{expert_index}.w1.weight"] = _export_tensor(
                w1[expert_index]
            )
            new_weights[f"mtp.0.ffn.experts.{expert_index}.w3.weight"] = _export_tensor(
                w3[expert_index]
            )
            new_weights[f"mtp.0.ffn.experts.{expert_index}.w2.weight"] = _export_tensor(
                w2_output_major[expert_index]
            )

    return new_weights


def _dspark_layer_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.hc_attn_fn$")
    layer_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := layer_pattern.fullmatch(key)) is not None
        }
    )
    if not layer_indices:
        raise KeyError("Training checkpoint does not contain DSpark decoder layers")
    expected = list(range(layer_indices[0], layer_indices[-1] + 1))
    if layer_indices != expected:
        raise ValueError(
            f"DSpark decoder layer indices must be contiguous, got {layer_indices}"
        )
    return layer_indices


def _build_dspark_export_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    layer_indices = _dspark_layer_indices(state_dict)
    first_layer_idx = layer_indices[0]
    last_layer_idx = layer_indices[-1]
    new_weights: dict[str, torch.Tensor] = {}

    layer_mapping = {
        "hc_attn_fn": "hc_attn_fn",
        "hc_attn_base": "hc_attn_base",
        "hc_attn_scale": "hc_attn_scale",
        "hc_ffn_fn": "hc_ffn_fn",
        "hc_ffn_base": "hc_ffn_base",
        "hc_ffn_scale": "hc_ffn_scale",
        "self_attn.attn_sink": "attn.attn_sink",
        "self_attn.wq_a.weight": "attn.wq_a.weight",
        "self_attn.q_norm.weight": "attn.q_norm.weight",
        "self_attn.wq_b.weight": "attn.wq_b.weight",
        "self_attn.wkv.weight": "attn.wkv.weight",
        "self_attn.kv_norm.weight": "attn.kv_norm.weight",
        "self_attn.wo_b.weight": "attn.wo_b.weight",
        "mlp.gate.weight": "ffn.gate.weight",
        "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
        "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    required_suffixes = set(layer_mapping) | {
        "self_attn.wo_a.weight",
        "mlp.shared_experts.gate_up_proj.weight",
        "mlp.experts.w13_weight",
        "mlp.experts.w2_weight",
    }

    for stage, layer_idx in enumerate(layer_indices):
        source_prefix = f"model.layers.{layer_idx}."
        target_prefix = f"mtp.{stage}."
        missing = sorted(
            suffix
            for suffix in required_suffixes
            if f"{source_prefix}{suffix}" not in state_dict
        )
        if missing:
            raise KeyError(
                f"DSpark layer {layer_idx} is missing tensors: " + ", ".join(missing)
            )

        for source_suffix, target_suffix in layer_mapping.items():
            new_weights[f"{target_prefix}{target_suffix}"] = _export_tensor(
                state_dict[f"{source_prefix}{source_suffix}"]
            )

        wo_a = state_dict[f"{source_prefix}self_attn.wo_a.weight"]
        if wo_a.ndim == 3:
            wo_a = wo_a.permute(0, 2, 1).reshape(-1, wo_a.shape[1])
        elif wo_a.ndim != 2:
            raise ValueError(
                f"Unexpected DSpark wo_a shape at layer {layer_idx}: "
                f"{tuple(wo_a.shape)}"
            )
        new_weights[f"{target_prefix}attn.wo_a.weight"] = _export_tensor(wo_a)

        shared_gate_up = state_dict[
            f"{source_prefix}mlp.shared_experts.gate_up_proj.weight"
        ]
        if shared_gate_up.ndim != 2 or shared_gate_up.shape[0] % 2 != 0:
            raise ValueError(
                f"Unexpected DSpark shared gate/up shape at layer {layer_idx}: "
                f"{tuple(shared_gate_up.shape)}"
            )
        shared_w1, shared_w3 = shared_gate_up.chunk(2, dim=0)
        new_weights[f"{target_prefix}ffn.shared_experts.w1.weight"] = _export_tensor(
            shared_w1
        )
        new_weights[f"{target_prefix}ffn.shared_experts.w3.weight"] = _export_tensor(
            shared_w3
        )

        w13 = state_dict[f"{source_prefix}mlp.experts.w13_weight"]
        w2 = state_dict[f"{source_prefix}mlp.experts.w2_weight"]
        if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
            raise ValueError(
                f"Unexpected DSpark routed MoE shapes at layer {layer_idx}: "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
            )
        if w13.shape[1] == w2.shape[2] and w13.shape[2] == 2 * w2.shape[1]:
            w1_input_major, w3_input_major = w13.chunk(2, dim=2)
            w1 = w1_input_major.transpose(1, 2)
            w3 = w3_input_major.transpose(1, 2)
            w2_output_major = w2.transpose(1, 2)
        elif w13.shape[1] == 2 * w2.shape[2] and w13.shape[2] == w2.shape[1]:
            w1, w3 = w13.chunk(2, dim=1)
            w2_output_major = w2
        else:
            raise ValueError(
                f"Cannot determine DSpark routed MoE layout at layer {layer_idx}: "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
            )
        router = state_dict[f"{source_prefix}mlp.gate.weight"]
        if router.ndim != 2 or router.shape[0] != w13.shape[0]:
            raise ValueError(
                f"DSpark expert count mismatch at layer {layer_idx}: "
                f"router={tuple(router.shape)}, w13={tuple(w13.shape)}"
            )
        for expert_idx in tqdm(
            range(w13.shape[0]), desc=f"Preparing DSpark stage {stage} experts"
        ):
            expert_prefix = f"{target_prefix}ffn.experts.{expert_idx}."
            new_weights[f"{expert_prefix}w1.weight"] = _export_tensor(w1[expert_idx])
            new_weights[f"{expert_prefix}w3.weight"] = _export_tensor(w3[expert_idx])
            new_weights[f"{expert_prefix}w2.weight"] = _export_tensor(
                w2_output_major[expert_idx]
            )

    first_prefix = f"model.layers.{first_layer_idx}."
    for suffix in ("main_proj.weight", "main_norm.weight"):
        source = f"{first_prefix}{suffix}"
        if source not in state_dict:
            raise KeyError(f"DSpark checkpoint is missing first-stage tensor: {source}")
        new_weights[f"mtp.0.{suffix}"] = _export_tensor(state_dict[source])

    last_prefix = f"model.layers.{last_layer_idx}."
    last_stage = len(layer_indices) - 1
    for suffix in (
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ):
        source = f"{last_prefix}{suffix}"
        if source not in state_dict:
            raise KeyError(f"DSpark checkpoint is missing final-stage tensor: {source}")
        new_weights[f"mtp.{last_stage}.{suffix}"] = _export_tensor(state_dict[source])

    for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        tensor = _first_tensor(
            state_dict,
            f"model.{suffix}",
            f"{last_prefix}{suffix}",
        )
        if tensor is None:
            raise KeyError(
                f"DSpark checkpoint is missing final HC head tensor: {suffix}"
            )
        new_weights[f"mtp.{last_stage}.{suffix}"] = _export_tensor(tensor)

    # The training wrapper owns this head. Preserve it for compatible runtimes,
    # while patch_model_weights skips it when the target checkpoint lacks it.
    for suffix in ("weight", "bias"):
        source = f"confidence_head.{suffix}"
        if source in state_dict:
            new_weights[f"mtp.{last_stage}.confidence_head.proj.{suffix}"] = (
                _export_tensor(state_dict[source])
            )

    return new_weights


def build_export_weights(
    state_dict: dict[str, torch.Tensor],
    *,
    layout: str,
) -> dict[str, torch.Tensor]:
    state_dict = _canonical_state_dict(state_dict)
    if layout == "dspark":
        return _build_dspark_export_weights(state_dict)
    if layout == "vllm_ascend":
        return _build_vllm_ascend_export_weights(state_dict)
    raise ValueError(f"Unknown DS4 export layout: {layout!r}")


def _target_tensor_shapes(safetensors_files: list[Path]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for st_path in safetensors_files:
        with safe_open(st_path, framework="pt", device="cpu") as source:
            for key in source.keys():  # noqa: SIM118 -- safetensors/runtime key API is explicit
                if key in shapes:
                    raise ValueError(
                        f"Duplicate tensor key across safetensors files: {key}"
                    )
                shapes[key] = tuple(source.get_slice(key).get_shape())
    return shapes


def _adapt_export_weights_to_target_layout(
    export_weights: dict[str, torch.Tensor],
    target_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], str | None]:
    """Account for native DS4 checkpoints that share the target embedding."""
    if (
        DEDICATED_MTP_EMBEDDING_KEY not in export_weights
        or DEDICATED_MTP_EMBEDDING_KEY in target_shapes
    ):
        return export_weights, None

    expected_shape = tuple(export_weights[DEDICATED_MTP_EMBEDDING_KEY].shape)
    shared_embedding_key = next(
        (key for key in SHARED_MODEL_EMBEDDING_KEYS if key in target_shapes),
        None,
    )
    if shared_embedding_key is None:
        return export_weights, None
    if target_shapes[shared_embedding_key] != expected_shape:
        raise ValueError(
            "Shared target embedding shape does not match the trained embedding: "
            f"{shared_embedding_key}={target_shapes[shared_embedding_key]}, "
            f"trained={expected_shape}"
        )

    adapted_weights = dict(export_weights)
    del adapted_weights[DEDICATED_MTP_EMBEDDING_KEY]
    return adapted_weights, shared_embedding_key


def _drop_optional_dspark_confidence_weights(
    export_weights: dict[str, torch.Tensor],
    target_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Patch confidence tensors only when the target runtime exposes them."""

    skipped = sorted(
        key
        for key in export_weights
        if ".confidence_head.proj." in key and key not in target_shapes
    )
    if not skipped:
        return export_weights, []
    return (
        {key: value for key, value in export_weights.items() if key not in skipped},
        skipped,
    )


class _ModelFileCopier:
    def __init__(self, mode: str):
        self.mode = mode
        self.reflinked = 0
        self.hardlinked = 0
        self.copied = 0

    def __call__(self, source: str, target: str) -> str:
        if self.mode == "hardlink" and source.endswith(".safetensors"):
            try:
                os.link(source, target)
            except OSError as exc:
                raise RuntimeError(
                    "Hard-link copy failed. Source and target must be on the "
                    "same filesystem; use --copy-mode full otherwise."
                ) from exc
            self.hardlinked += 1
            return target

        if self.mode in {"auto", "reflink"} and stat.S_ISREG(os.stat(source).st_mode):
            try:
                import fcntl

                with (
                    open(source, "rb") as source_file,
                    open(target, "wb") as target_file,
                ):
                    fcntl.ioctl(target_file.fileno(), FICLONE, source_file.fileno())
                shutil.copystat(source, target)
                self.reflinked += 1
                return target
            except (ImportError, OSError):
                if os.path.exists(target):
                    os.unlink(target)
                if self.mode == "reflink":
                    raise RuntimeError(
                        "The source filesystem does not support reflink copies. "
                        "Use --copy-mode auto or --copy-mode full."
                    )

        shutil.copy2(source, target)
        self.copied += 1
        return target


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Training checkpoint is not a state dict: {path}")
    return state_dict


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _state_tensor_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )


def _validate_replicated_tensors(
    reference_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    keys: list[str],
    *,
    candidate_path: Path,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> None:
    """Reject TP rank states whose logically replicated tensors diverged."""

    if markov_w1_max_differing_elements < 1:
        raise ValueError("markov_w1_max_differing_elements must be positive")

    for key in keys:
        reference = reference_state.get(key)
        candidate = candidate_state.get(key)
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            raise KeyError(
                f"Missing replicated DSpark tensor {key} in {candidate_path}"
            )
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError(
                f"Inconsistent replicated DSpark tensor metadata for {key} in "
                f"{candidate_path}: reference={tuple(reference.shape)}/{reference.dtype}, "
                f"candidate={tuple(candidate.shape)}/{candidate.dtype}"
            )
        is_markov = key.endswith(
            (
                ".markov_head.markov_w1.weight",
                ".markov_head.markov_w2.weight",
            )
        )
        if is_markov and (
            not bool(torch.isfinite(reference).all())
            or not bool(torch.isfinite(candidate).all())
        ):
            raise ValueError(f"Non-finite Markov replica: {key} in {candidate_path}")
        if torch.equal(reference, candidate):
            continue
        difference = (reference.float() - candidate.float()).abs()
        max_abs_delta = float(difference.max().item())
        mean_abs_delta = float(difference.mean().item())
        differing_elements = int(torch.count_nonzero(difference).item())
        detail = (
            f"tensor={key}, candidate={candidate_path}, max_abs_delta={max_abs_delta}, "
            f"mean_abs_delta={mean_abs_delta}, "
            f"differing_elements={differing_elements}/{reference.numel()}"
        )
        if (
            allow_markov_w1_one_ulp
            and key.endswith(".markov_head.markov_w1.weight")
            and reference.dtype == torch.bfloat16
            and 0 < differing_elements <= markov_w1_max_differing_elements
        ):
            offsets = torch.nonzero(difference.reshape(-1), as_tuple=True)[0]
            examples = []
            for offset in offsets.tolist():
                a = reference.reshape(-1)[offset]
                b = candidate.reshape(-1)[offset]
                bits_a, bits_b = int(a.view(torch.int16)), int(b.view(torch.int16))
                ordered_a = -32768 - bits_a if bits_a < 0 else bits_a
                ordered_b = -32768 - bits_b if bits_b < 0 else bits_b
                examples.append(
                    {
                        "flat_index": offset,
                        "reference": float(a),
                        "candidate_value": float(b),
                        "bf16_ulp_distance": abs(ordered_a - ordered_b),
                    }
                )
            if all(example["bf16_ulp_distance"] == 1 for example in examples):
                print(
                    "WARNING: accepting explicitly bounded BF16 W1 "
                    "one-ULP differences; keeping TP rank 0 (no averaging): "
                    f"{detail}, limit={markov_w1_max_differing_elements}, "
                    f"examples={examples}",
                    flush=True,
                )
                continue
        if (
            is_markov
            and markov_replica_atol > 0
            and math.isfinite(max_abs_delta)
            and max_abs_delta <= markov_replica_atol
        ):
            print(
                f"WARNING: accepting Markov replica drift within explicit "
                f"atol={markov_replica_atol}; keeping TP rank 0 (no averaging): {detail}",
                flush=True,
            )
            continue
        raise ValueError(
            "DSpark TP replicated tensor values differ across rank checkpoints: "
            + detail
        )


def _rank_state_files(directory: Path) -> list[Path]:
    rank_pattern = re.compile(r"model_states_rank_(\d+)\.pt$")
    ranked_files = []
    for path in directory.iterdir():
        match = rank_pattern.fullmatch(path.name)
        if match:
            ranked_files.append((int(match.group(1)), path))
    return [path for _, path in sorted(ranked_files)]


def _topology_rank_state_files(
    directory: Path,
    state_files: list[Path],
    data_parallel_rank: int,
) -> list[Path]:
    topology_path = directory / "parallel_topology.json"
    if not topology_path.exists():
        if data_parallel_rank != 0:
            raise ValueError(
                "--data-parallel-rank requires parallel_topology.json in the checkpoint"
            )
        return state_files

    with topology_path.open("r", encoding="utf-8") as source:
        topology = json.load(source)
    tp_size = int(topology["tensor_parallel_size"])
    dp_size = int(topology["data_parallel_size"])
    world_size = int(topology["world_size"])
    if topology.get("rank_order") != "dp_major_tp_minor":
        raise ValueError(
            f"Unsupported checkpoint rank order: {topology.get('rank_order')!r}"
        )
    if len(state_files) != world_size:
        raise ValueError(
            f"Checkpoint topology expects {world_size} rank states, found {len(state_files)}"
        )
    if not 0 <= data_parallel_rank < dp_size:
        raise ValueError(
            f"data_parallel_rank={data_parallel_rank} is outside dp_size={dp_size}"
        )
    start = data_parallel_rank * tp_size
    selected = state_files[start : start + tp_size]
    print(
        f"Selected DP replica {data_parallel_rank}/{dp_size}: "
        f"{len(selected)} TP rank states"
    )
    return selected


def _add_eh_projection_tp_shards(
    simple_tp_shards: dict[str, int],
    state: dict[str, torch.Tensor],
    *,
    tp_size: int,
    source: Path,
) -> None:
    for projection_name in ("e_proj.weight", "h_proj.weight"):
        projection_key = f"{LAYER_PREFIX}{projection_name}"
        projection = state.get(projection_key)
        if not isinstance(projection, torch.Tensor):
            raise KeyError(f"Missing projection tensor in {source}: {projection_key}")
        if projection.ndim != 2:
            raise ValueError(
                f"Invalid projection tensor {projection_key}: "
                f"shape={tuple(projection.shape)}"
            )
        if projection.shape[0] == projection.shape[1]:
            continue
        if projection.shape[0] * tp_size != projection.shape[1]:
            raise ValueError(
                f"Cannot identify TP output shard for {projection_key}: "
                f"shape={tuple(projection.shape)}, tp={tp_size}"
            )
        simple_tp_shards[projection_key] = 0


def _add_attention_projection_tp_shards(
    simple_tp_shards: dict[str, int],
    state: dict[str, torch.Tensor],
    *,
    tp_size: int,
    source: Path,
) -> None:
    projection_specs = (
        ("wq_a.weight", "q_norm.weight"),
        ("wkv.weight", "kv_norm.weight"),
    )
    attention_prefix = f"{LAYER_PREFIX}mtp_block.self_attn."
    for projection_name, norm_name in projection_specs:
        projection_key = f"{attention_prefix}{projection_name}"
        norm_key = f"{attention_prefix}{norm_name}"
        projection = state.get(projection_key)
        norm = state.get(norm_key)
        if not isinstance(projection, torch.Tensor):
            raise KeyError(
                f"Missing attention projection tensor in {source}: {projection_key}"
            )
        if not isinstance(norm, torch.Tensor):
            raise KeyError(f"Missing attention norm tensor in {source}: {norm_key}")
        if projection.ndim != 2 or norm.ndim != 1:
            raise ValueError(
                "Invalid attention projection/norm tensors: "
                f"{projection_key}={tuple(projection.shape)}, "
                f"{norm_key}={tuple(norm.shape)}"
            )
        expected_output_size = int(norm.numel())
        local_output_size = int(projection.shape[0])
        if local_output_size == expected_output_size:
            continue
        if local_output_size * tp_size != expected_output_size:
            raise ValueError(
                f"Cannot identify TP output shard for {projection_key}: "
                f"projection={tuple(projection.shape)}, "
                f"norm={tuple(norm.shape)}, tp={tp_size}"
            )
        simple_tp_shards[projection_key] = 0


def _dspark_projection_layout(
    state: dict[str, torch.Tensor],
    layer_indices: list[int],
    tp_size: int,
    target_shapes: dict[str, tuple[int, ...]] | None,
) -> tuple[dict[str, int], list[str]]:
    sharded: dict[str, int] = {}
    replicated: list[str] = []
    projections = [
        (layer_indices[0], "main_proj.weight", "mtp.0.main_proj.weight"),
        *[
            (
                layer_indices[-1],
                f"markov_head.{name}.weight",
                f"mtp.{len(layer_indices) - 1}.markov_head.{name}.weight",
            )
            for name in ("markov_w1", "markov_w2")
        ],
    ]
    for layer, suffix, target_key in projections:
        key = f"model.layers.{layer}.{suffix}"
        value = state[key]
        shape = tuple(value.shape)
        expected = None if target_shapes is None else target_shapes.get(target_key)
        if target_shapes is not None and expected is None:
            raise KeyError(f"Target checkpoint is missing {target_key}")
        if expected is None:
            if suffix == "main_proj.weight":
                norm = state[f"model.layers.{layer}.main_norm.weight"]
                expected = (int(norm.numel()), shape[1])
            else:
                # Legacy callers without a serving checkpoint used vocab shards.
                expected = (shape[0] * tp_size, *shape[1:])
        if shape == tuple(expected):
            replicated.append(key)
        elif value.ndim == 2 and (shape[0] * tp_size, shape[1]) == tuple(expected):
            sharded[key] = 0
        else:
            raise ValueError(
                f"Cannot identify DSpark TP layout for {key}: "
                f"local={shape}, target={expected}, tp={tp_size}"
            )
    return sharded, replicated


def _load_dspark_export_weights(
    trained_state: Path,
    data_parallel_rank: int,
    target_shapes: dict[str, tuple[int, ...]] | None = None,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> dict[str, torch.Tensor]:
    state_files = (
        [trained_state] if trained_state.is_file() else _rank_state_files(trained_state)
    )
    if not state_files:
        raise FileNotFoundError(
            f"No model_states_rank_<rank>.pt files found in {trained_state}"
        )
    if trained_state.is_dir():
        topology_path = trained_state / "parallel_topology.json"
        if len(state_files) > 1 and not topology_path.exists():
            raise ValueError(
                "DSpark TP export requires parallel_topology.json so DP replicas "
                "cannot be mistaken for TP shards"
            )
        state_files = _topology_rank_state_files(
            trained_state,
            state_files,
            data_parallel_rank,
        )

    def load_rank_state(shard_index: int) -> dict[str, torch.Tensor]:
        path = state_files[shard_index]
        file_size = path.stat().st_size
        started = time.monotonic()
        print(
            f"Loading DSpark TP rank state {shard_index + 1}/{len(state_files)}: "
            f"{path} ({_format_bytes(file_size)})",
            flush=True,
        )
        state = _canonical_state_dict(_load_state(path))
        elapsed = time.monotonic() - started
        print(
            f"Loaded DSpark TP rank state {shard_index + 1}/{len(state_files)} "
            f"in {elapsed:.1f}s: tensors={len(state)}, "
            f"tensor_bytes={_format_bytes(_state_tensor_bytes(state))}",
            flush=True,
        )
        return state

    first_state = load_rank_state(0)
    layer_indices = _dspark_layer_indices(first_state)
    if len(state_files) == 1:
        return _build_dspark_export_weights(first_state)

    replicated_routing_state_keys = [
        f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        for layer_idx in layer_indices
    ]

    simple_specs: dict[str, int] = {}
    shared_gate_up_keys: list[str] = []
    routed_keys: list[tuple[str, str, int, int]] = []
    for layer_idx in layer_indices:
        prefix = f"model.layers.{layer_idx}."
        simple_specs.update(
            {
                f"{prefix}self_attn.attn_sink": 0,
                f"{prefix}self_attn.wq_b.weight": 0,
                f"{prefix}self_attn.wo_a.weight": 0,
                f"{prefix}self_attn.wo_b.weight": 1,
                f"{prefix}mlp.shared_experts.down_proj.weight": 1,
            }
        )
        shared_gate_up_keys.append(f"{prefix}mlp.shared_experts.gate_up_proj.weight")
        w13_key = f"{prefix}mlp.experts.w13_weight"
        w2_key = f"{prefix}mlp.experts.w2_weight"
        w13 = first_state.get(w13_key)
        w2 = first_state.get(w2_key)
        if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
            raise KeyError(f"Missing DSpark routed MoE tensors in {state_files[0]}")
        if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
            raise ValueError(
                f"Unexpected DSpark routed MoE shard shapes at layer {layer_idx}: "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
            )
        if w13.shape[1] == w2.shape[2] and w13.shape[2] == 2 * w2.shape[1]:
            routed_keys.append((w13_key, w2_key, 2, 1))
        elif w13.shape[1] == 2 * w2.shape[2] and w13.shape[2] == w2.shape[1]:
            routed_keys.append((w13_key, w2_key, 1, 2))
        else:
            raise ValueError(
                f"Cannot identify DSpark routed MoE shard orientation at layer "
                f"{layer_idx}: w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
            )

    projection_shards, replicated_projections = _dspark_projection_layout(
        first_state,
        layer_indices,
        len(state_files),
        target_shapes,
    )
    simple_specs.update(projection_shards)
    replicated_routing_state_keys.extend(replicated_projections)

    print(
        "Allocating CPU tensors for merged DSpark TP weights: "
        f"tp_shards={len(state_files)}, decoder_layers={len(layer_indices)}",
        flush=True,
    )
    merged_simple: dict[str, torch.Tensor] = {}
    for key, shard_dim in simple_specs.items():
        shard = first_state.get(key)
        if not isinstance(shard, torch.Tensor):
            raise KeyError(
                f"Missing DSpark TP-sharded tensor in {state_files[0]}: {key}"
            )
        if shard.ndim <= shard_dim:
            raise ValueError(
                f"Invalid DSpark TP shard dimension for {key}: "
                f"shape={tuple(shard.shape)}, dim={shard_dim}"
            )
        merged_shape = list(shard.shape)
        merged_shape[shard_dim] *= len(state_files)
        merged_simple[key] = torch.empty(merged_shape, dtype=shard.dtype)

    merged_shared: dict[str, torch.Tensor] = {}
    for key in shared_gate_up_keys:
        shard = first_state.get(key)
        if not isinstance(shard, torch.Tensor) or shard.ndim != 2 or shard.shape[0] % 2:
            raise ValueError(
                f"Invalid DSpark shared gate/up shard for {key}: "
                f"{None if shard is None else tuple(shard.shape)}"
            )
        merged_shared[key] = torch.empty(
            (shard.shape[0] * len(state_files), shard.shape[1]),
            dtype=shard.dtype,
        )

    merged_routed: dict[str, torch.Tensor] = {}
    for w13_key, w2_key, w13_dim, w2_dim in routed_keys:
        for key, shard_dim in ((w13_key, w13_dim), (w2_key, w2_dim)):
            shard = first_state[key]
            merged_shape = list(shard.shape)
            merged_shape[shard_dim] *= len(state_files)
            merged_routed[key] = torch.empty(merged_shape, dtype=shard.dtype)

    merged_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensors in (merged_simple, merged_shared, merged_routed)
        for tensor in tensors.values()
    )
    print(
        f"Allocated {_format_bytes(merged_bytes)} for merged DSpark TP weights",
        flush=True,
    )

    def copy_shard(
        target: torch.Tensor,
        shard: torch.Tensor,
        shard_index: int,
        shard_dim: int,
    ) -> None:
        start = shard_index * shard.shape[shard_dim]
        target.narrow(shard_dim, start, shard.shape[shard_dim]).copy_(shard)

    def copy_state_shards(
        state: dict[str, torch.Tensor],
        shard_index: int,
        path: Path,
    ) -> None:
        for key, shard_dim in simple_specs.items():
            shard = state.get(key)
            first_shard = first_state.get(key)
            if (
                not isinstance(shard, torch.Tensor)
                or not isinstance(first_shard, torch.Tensor)
                or shard.shape != first_shard.shape
            ):
                raise ValueError(
                    f"Inconsistent DSpark TP shard for {key} in {path}: "
                    f"{None if shard is None else tuple(shard.shape)}"
                )
            copy_shard(merged_simple[key], shard, shard_index, shard_dim)

        for key in shared_gate_up_keys:
            shard = state.get(key)
            first_shard = first_state.get(key)
            if (
                not isinstance(shard, torch.Tensor)
                or not isinstance(first_shard, torch.Tensor)
                or shard.shape != first_shard.shape
            ):
                raise ValueError(
                    f"Inconsistent DSpark shared gate/up shard for {key} in {path}: "
                    f"{None if shard is None else tuple(shard.shape)}"
                )
            local_w1, local_w3 = shard.chunk(2, dim=0)
            local_size = local_w1.shape[0]
            full_size = local_size * len(state_files)
            merged_shared[key].narrow(0, shard_index * local_size, local_size).copy_(
                local_w1
            )
            merged_shared[key].narrow(
                0, full_size + shard_index * local_size, local_size
            ).copy_(local_w3)

        for w13_key, w2_key, w13_dim, w2_dim in routed_keys:
            shard_w13 = state.get(w13_key)
            shard_w2 = state.get(w2_key)
            first_w13 = first_state.get(w13_key)
            first_w2 = first_state.get(w2_key)
            if (
                not isinstance(shard_w13, torch.Tensor)
                or not isinstance(shard_w2, torch.Tensor)
                or not isinstance(first_w13, torch.Tensor)
                or not isinstance(first_w2, torch.Tensor)
                or shard_w13.shape != first_w13.shape
                or shard_w2.shape != first_w2.shape
            ):
                raise ValueError(f"Inconsistent DSpark routed MoE shards in {path}")
            local_w1, local_w3 = shard_w13.chunk(2, dim=w13_dim)
            local_size = local_w1.shape[w13_dim]
            full_size = local_size * len(state_files)
            merged_routed[w13_key].narrow(
                w13_dim, shard_index * local_size, local_size
            ).copy_(local_w1)
            merged_routed[w13_key].narrow(
                w13_dim, full_size + shard_index * local_size, local_size
            ).copy_(local_w3)
            copy_shard(merged_routed[w2_key], shard_w2, shard_index, w2_dim)

    merge_started = time.monotonic()
    copy_state_shards(first_state, 0, state_files[0])
    print(
        f"Merged DSpark TP rank state 1/{len(state_files)}",
        flush=True,
    )
    for shard_index, path in enumerate(state_files[1:], start=1):
        state = load_rank_state(shard_index)
        if _dspark_layer_indices(state) != layer_indices:
            raise ValueError(f"Inconsistent DSpark layer indices in {path}")
        _validate_replicated_tensors(
            first_state,
            state,
            replicated_routing_state_keys,
            candidate_path=path,
            markov_replica_atol=markov_replica_atol,
            allow_markov_w1_one_ulp=allow_markov_w1_one_ulp,
            markov_w1_max_differing_elements=markov_w1_max_differing_elements,
        )
        copy_state_shards(state, shard_index, path)
        del state
        print(
            f"Merged DSpark TP rank state {shard_index + 1}/{len(state_files)}",
            flush=True,
        )

    first_state.update(merged_simple)
    first_state.update(merged_shared)
    first_state.update(merged_routed)
    print(
        f"Merged {len(state_files)} DSpark TP shards across "
        f"{len(layer_indices)} decoder layers in "
        f"{time.monotonic() - merge_started:.1f}s"
    )
    return _build_dspark_export_weights(first_state)


def _load_export_weights(
    trained_state: Path,
    layout: str,
    data_parallel_rank: int = 0,
    target_shapes: dict[str, tuple[int, ...]] | None = None,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> dict[str, torch.Tensor]:
    if layout == "dspark":
        return _load_dspark_export_weights(
            trained_state,
            data_parallel_rank,
            target_shapes,
            markov_replica_atol,
            allow_markov_w1_one_ulp=allow_markov_w1_one_ulp,
            markov_w1_max_differing_elements=markov_w1_max_differing_elements,
        )
    if layout == "vllm_ascend":
        has_explicit_topology = (
            trained_state.is_dir()
            and (trained_state / "parallel_topology.json").exists()
        )
        state_files = (
            [trained_state]
            if trained_state.is_file()
            else _rank_state_files(trained_state)
        )
        if not state_files:
            raise FileNotFoundError(
                f"No model_states_rank_<rank>.pt files found in {trained_state}"
            )
        if trained_state.is_dir():
            state_files = _topology_rank_state_files(
                trained_state,
                state_files,
                data_parallel_rank,
            )
        first_state = _canonical_state_dict(_load_state(state_files[0]))
        if len(state_files) > 1:
            w13_key = f"{LAYER_PREFIX}mtp_block.mlp.experts.w13_weight"
            w2_key = f"{LAYER_PREFIX}mtp_block.mlp.experts.w2_weight"
            first_w13 = first_state.get(w13_key)
            first_w2 = first_state.get(w2_key)
            if not isinstance(first_w13, torch.Tensor) or not isinstance(
                first_w2, torch.Tensor
            ):
                return _build_vllm_ascend_export_weights(first_state)
            if (
                first_w13.ndim != 3
                or first_w2.ndim != 3
                or first_w13.shape[0] != first_w2.shape[0]
            ):
                raise ValueError(
                    "Unexpected vllm_ascend routed MoE shard shapes: "
                    f"w13={tuple(first_w13.shape)}, w2={tuple(first_w2.shape)}"
                )

            input_major = (
                first_w13.shape[1] == first_w2.shape[2]
                and first_w13.shape[2] == 2 * first_w2.shape[1]
            )
            output_major = (
                first_w13.shape[1] == 2 * first_w2.shape[2]
                and first_w13.shape[2] == first_w2.shape[1]
            )
            if input_major:
                w13_shard_dim, w2_shard_dim = 2, 1
            elif output_major:
                w13_shard_dim, w2_shard_dim = 1, 2
            else:
                raise ValueError(
                    "Cannot identify vllm_ascend routed MoE shard orientation: "
                    f"w13={tuple(first_w13.shape)}, w2={tuple(first_w2.shape)}"
                )

            merged_w13_shape = list(first_w13.shape)
            merged_w2_shape = list(first_w2.shape)
            merged_w13_shape[w13_shard_dim] *= len(state_files)
            merged_w2_shape[w2_shard_dim] *= len(state_files)
            merged_w13 = torch.empty(merged_w13_shape, dtype=first_w13.dtype)
            merged_w2 = torch.empty(merged_w2_shape, dtype=first_w2.dtype)

            simple_tp_shards = (
                {
                    "model.embed_tokens.weight": 0,
                    f"{LAYER_PREFIX}mtp_block.self_attn.attn_sink": 0,
                    f"{LAYER_PREFIX}mtp_block.self_attn.wq_b.weight": 0,
                    f"{LAYER_PREFIX}mtp_block.self_attn.wo_a.weight": 0,
                    f"{LAYER_PREFIX}mtp_block.self_attn.wo_b.weight": 1,
                    f"{LAYER_PREFIX}mtp_block.mlp.shared_experts.down_proj.weight": 1,
                }
                if has_explicit_topology
                else {}
            )
            if has_explicit_topology:
                _add_eh_projection_tp_shards(
                    simple_tp_shards,
                    first_state,
                    tp_size=len(state_files),
                    source=state_files[0],
                )
                _add_attention_projection_tp_shards(
                    simple_tp_shards,
                    first_state,
                    tp_size=len(state_files),
                    source=state_files[0],
                )
            merged_simple = {}
            for key, shard_dim in simple_tp_shards.items():
                shard = first_state.get(key)
                if not isinstance(shard, torch.Tensor):
                    raise KeyError(
                        f"Missing TP-sharded tensor in {state_files[0]}: {key}"
                    )
                if shard.ndim <= shard_dim:
                    raise ValueError(
                        f"Invalid TP shard dimension for {key}: shape={tuple(shard.shape)}, "
                        f"dim={shard_dim}"
                    )
                merged_shape = list(shard.shape)
                merged_shape[shard_dim] *= len(state_files)
                merged_simple[key] = torch.empty(merged_shape, dtype=shard.dtype)

            shared_gate_up_key = (
                f"{LAYER_PREFIX}mtp_block.mlp.shared_experts.gate_up_proj.weight"
            )
            first_shared_gate_up = first_state.get(shared_gate_up_key)
            merged_shared_gate_up = None
            if has_explicit_topology:
                if not isinstance(first_shared_gate_up, torch.Tensor) or (
                    first_shared_gate_up.ndim != 2
                    or first_shared_gate_up.shape[0] % 2 != 0
                ):
                    raise ValueError(
                        "Invalid TP-sharded shared gate/up tensor: "
                        f"{None if first_shared_gate_up is None else tuple(first_shared_gate_up.shape)}"
                    )
                merged_shared_gate_up = torch.empty(
                    (
                        first_shared_gate_up.shape[0] * len(state_files),
                        first_shared_gate_up.shape[1],
                    ),
                    dtype=first_shared_gate_up.dtype,
                )

            def copy_shard(target, shard, shard_index, shard_dim):
                start = shard_index * shard.shape[shard_dim]
                target.narrow(shard_dim, start, shard.shape[shard_dim]).copy_(shard)

            def copy_w13_shard(shard, shard_index):
                local_w1, local_w3 = shard.chunk(2, dim=w13_shard_dim)
                local_size = local_w1.shape[w13_shard_dim]
                full_size = local_size * len(state_files)
                merged_w13.narrow(
                    w13_shard_dim, shard_index * local_size, local_size
                ).copy_(local_w1)
                merged_w13.narrow(
                    w13_shard_dim,
                    full_size + shard_index * local_size,
                    local_size,
                ).copy_(local_w3)

            def copy_shared_gate_up_shard(shard, shard_index):
                if merged_shared_gate_up is None:
                    return
                local_w1, local_w3 = shard.chunk(2, dim=0)
                local_size = local_w1.shape[0]
                full_size = local_size * len(state_files)
                merged_shared_gate_up.narrow(
                    0, shard_index * local_size, local_size
                ).copy_(local_w1)
                merged_shared_gate_up.narrow(
                    0, full_size + shard_index * local_size, local_size
                ).copy_(local_w3)

            def copy_additional_tp_shards(state, shard_index, path):
                for key, shard_dim in simple_tp_shards.items():
                    shard = state.get(key)
                    first_shard = first_state.get(key)
                    if not isinstance(shard, torch.Tensor) or (
                        isinstance(first_shard, torch.Tensor)
                        and shard.shape != first_shard.shape
                    ):
                        raise ValueError(
                            f"Inconsistent TP shard for {key} in {path}: "
                            f"{None if shard is None else tuple(shard.shape)}"
                        )
                    copy_shard(merged_simple[key], shard, shard_index, shard_dim)

                if merged_shared_gate_up is None:
                    return
                shard = state.get(shared_gate_up_key)
                if not isinstance(shard, torch.Tensor) or (
                    shard.shape != first_shared_gate_up.shape
                ):
                    raise ValueError(
                        f"Inconsistent shared gate/up TP shard in {path}: "
                        f"{None if shard is None else tuple(shard.shape)}"
                    )
                copy_shared_gate_up_shard(shard, shard_index)

            copy_w13_shard(first_w13, 0)
            copy_shard(merged_w2, first_w2, 0, w2_shard_dim)
            copy_additional_tp_shards(first_state, 0, state_files[0])
            for shard_index, path in enumerate(state_files[1:], start=1):
                state = _canonical_state_dict(_load_state(path))
                shard_w13 = state.get(w13_key)
                shard_w2 = state.get(w2_key)
                if not isinstance(shard_w13, torch.Tensor) or not isinstance(
                    shard_w2, torch.Tensor
                ):
                    raise KeyError(f"Missing routed MoE shard tensors in {path}")
                if (
                    shard_w13.shape != first_w13.shape
                    or shard_w2.shape != first_w2.shape
                ):
                    raise ValueError(
                        f"Inconsistent routed MoE shard shapes in {path}: "
                        f"w13={tuple(shard_w13.shape)}, w2={tuple(shard_w2.shape)}"
                    )
                copy_w13_shard(shard_w13, shard_index)
                copy_shard(merged_w2, shard_w2, shard_index, w2_shard_dim)
                copy_additional_tp_shards(state, shard_index, path)
                del state
            first_state[w13_key] = merged_w13
            first_state[w2_key] = merged_w2
            first_state.update(merged_simple)
            if merged_shared_gate_up is not None:
                first_state[shared_gate_up_key] = merged_shared_gate_up
            print(
                f"Merged {len(state_files)} vllm_ascend TP shards: "
                f"w13={tuple(merged_w13.shape)}, w2={tuple(merged_w2.shape)}"
            )
        return _build_vllm_ascend_export_weights(first_state)

    raise ValueError("Unknown DS4 export layout: " + repr(layout))


def patch_model_weights(
    trained_state: str,
    safetensors_dir: str,
    *,
    layout: str,
    data_parallel_rank: int = 0,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> None:
    safetensors_files = sorted(Path(safetensors_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {safetensors_dir}")
    target_shapes = _target_tensor_shapes(safetensors_files)
    new_weights = _load_export_weights(
        Path(trained_state),
        layout,
        data_parallel_rank=data_parallel_rank,
        target_shapes=target_shapes,
        markov_replica_atol=markov_replica_atol,
        allow_markov_w1_one_ulp=allow_markov_w1_one_ulp,
        markov_w1_max_differing_elements=markov_w1_max_differing_elements,
    )
    print(f"Prepared {len(new_weights)} trained tensors for export")

    new_weights, shared_embedding_key = _adapt_export_weights_to_target_layout(
        new_weights,
        target_shapes,
    )
    new_weights, skipped_confidence = _drop_optional_dspark_confidence_weights(
        new_weights,
        target_shapes,
    )
    if shared_embedding_key is not None:
        print(
            "Target checkpoint shares the MTP token embedding with "
            f"{shared_embedding_key}; skipping the absent dedicated "
            f"{DEDICATED_MTP_EMBEDDING_KEY} tensor"
        )
    if skipped_confidence:
        print(
            "Target checkpoint has no DSpark confidence head; skipping "
            + ", ".join(skipped_confidence)
        )
    missing_targets = sorted(new_weights.keys() - target_shapes.keys())
    if missing_targets:
        raise KeyError(
            "Target checkpoint does not contain trained tensors: "
            + ", ".join(missing_targets)
        )
    shape_mismatches = [
        f"{key}: trained={tuple(value.shape)}, target={target_shapes[key]}"
        for key, value in new_weights.items()
        if tuple(value.shape) != target_shapes[key]
    ]
    if shape_mismatches:
        displayed = shape_mismatches[:20]
        if len(shape_mismatches) > len(displayed):
            displayed.append(
                f"... {len(shape_mismatches) - len(displayed)} additional mismatches omitted"
            )
        expert_mismatches = [
            key
            for key, value in new_weights.items()
            if ".ffn.experts." in key
            and key in target_shapes
            and tuple(value.shape) != target_shapes[key]
        ]
        hint = ""
        if expert_mismatches:
            hint = (
                "\nRouted experts appear tensor-parallel sharded. For "
                "vllm_ascend or dspark, pass the epoch directory containing one "
                "model_states_rank_<rank>.pt file per TP shard instead of a "
                "single rank0 file."
            )
        raise ValueError(
            "Target checkpoint tensor shapes do not match the trained DS4 state:\n"
            + "\n".join(displayed)
            + hint
        )

    quant_keys_to_drop = {
        suffix_key
        for key in new_weights
        for suffix_key in (f"{key}_scale", f"{key}_offset")
    }
    replaced = set()
    for st_path in tqdm(safetensors_files, desc="Patching safetensors"):
        with safe_open(st_path, framework="pt", device="cpu") as source:
            file_keys = list(source.keys())
            replacements = set(file_keys) & new_weights.keys()
            file_keys_to_drop = set(file_keys) & quant_keys_to_drop
            if not replacements and not file_keys_to_drop:
                continue
            tensors = {
                key: new_weights[key]
                if key in replacements
                else source.get_tensor(key).contiguous()
                for key in file_keys
                if key not in file_keys_to_drop
            }
        temporary_path = st_path.with_suffix(st_path.suffix + ".tmp")
        try:
            save_file(tensors, str(temporary_path))
            os.replace(temporary_path, st_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        replaced.update(replacements)

    json_path = Path(safetensors_dir) / "quant_model_description.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as source:
            quant_desc = json.load(source)
        for key in new_weights:
            if key in quant_desc:
                quant_desc[key] = "FLOAT"
            quant_desc.pop(f"{key}_scale", None)
            quant_desc.pop(f"{key}_offset", None)
        with json_path.open("w", encoding="utf-8") as target:
            json.dump(quant_desc, target, indent=2)

    print(f"Replaced {len(replaced)} tensors")


def verify_patched_model_weights(
    trained_state: str,
    patched_model_path: str | Path,
    *,
    layout: str,
    data_parallel_rank: int = 0,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> None:
    """Verify that every exported tensor was written exactly as prepared."""
    safetensors_files = sorted(Path(patched_model_path).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {patched_model_path}")

    target_shapes = _target_tensor_shapes(safetensors_files)
    expected_weights = _load_export_weights(
        Path(trained_state),
        layout,
        data_parallel_rank=data_parallel_rank,
        target_shapes=target_shapes,
        markov_replica_atol=markov_replica_atol,
        allow_markov_w1_one_ulp=allow_markov_w1_one_ulp,
        markov_w1_max_differing_elements=markov_w1_max_differing_elements,
    )
    expected_weights, shared_embedding_key = _adapt_export_weights_to_target_layout(
        expected_weights,
        target_shapes,
    )
    expected_weights, skipped_confidence = _drop_optional_dspark_confidence_weights(
        expected_weights,
        target_shapes,
    )
    if shared_embedding_key is not None:
        print(
            "Target checkpoint shares the MTP token embedding with "
            f"{shared_embedding_key}; verifying the remaining exported tensors"
        )
    if skipped_confidence:
        print(
            "Target checkpoint has no DSpark confidence head; verifying the "
            "remaining exported tensors"
        )

    found = set()
    stale_quant_keys = []
    expected_quant_keys = {
        suffix
        for key in expected_weights
        for suffix in (f"{key}_scale", f"{key}_offset")
    }
    for st_path in tqdm(safetensors_files, desc="Verifying safetensors"):
        with safe_open(st_path, framework="pt", device="cpu") as source:
            file_keys = set(source.keys())
            stale_quant_keys.extend(sorted(file_keys & expected_quant_keys))
            for key in sorted(file_keys & expected_weights.keys()):
                if key in found:
                    raise ValueError(
                        f"Patched tensor appears in multiple safetensors files: {key}"
                    )
                actual = source.get_tensor(key)
                expected = expected_weights[key]
                if actual.dtype != expected.dtype:
                    raise ValueError(
                        f"Patched tensor dtype mismatch for {key}: "
                        f"actual={actual.dtype}, expected={expected.dtype}"
                    )
                if tuple(actual.shape) != tuple(expected.shape):
                    raise ValueError(
                        f"Patched tensor shape mismatch for {key}: "
                        f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
                    )
                if not torch.equal(actual, expected):
                    max_diff = (actual.float() - expected.float()).abs().max().item()
                    raise ValueError(
                        f"Patched tensor value mismatch for {key}: max_diff={max_diff}"
                    )
                found.add(key)

    missing = sorted(expected_weights.keys() - found)
    if missing:
        raise KeyError(
            "Patched model is missing trained tensors: " + ", ".join(missing)
        )
    if stale_quant_keys:
        raise ValueError(
            "Patched FLOAT tensors still have obsolete quantization state: "
            + ", ".join(stale_quant_keys)
        )

    quant_description_path = Path(patched_model_path) / "quant_model_description.json"
    if quant_description_path.exists():
        with quant_description_path.open("r", encoding="utf-8") as source:
            quant_description = json.load(source)
        incorrect_descriptions = [
            key
            for key in expected_weights
            if key in quant_description and quant_description[key] != "FLOAT"
        ]
        if incorrect_descriptions:
            raise ValueError(
                "Patched tensors are not marked FLOAT in quant_model_description.json: "
                + ", ".join(sorted(incorrect_descriptions))
            )

    print(f"Patch verification PASS: {len(found)} tensors match training state exactly")


def export_patched_model(
    trained_state: str,
    source_model_path: Path,
    output_model_path: Path,
    *,
    layout: str,
    copy_mode: str = "auto",
    data_parallel_rank: int = 0,
    markov_replica_atol: float = 0.0,
    allow_markov_w1_one_ulp: bool = False,
    markov_w1_max_differing_elements: int = 1,
) -> None:
    source_model_path = source_model_path.resolve()
    output_model_path = output_model_path.resolve()
    if source_model_path == output_model_path:
        raise ValueError("Source and output model paths must be different")
    if output_model_path.exists():
        raise FileExistsError(f"Output model path already exists: {output_model_path}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_model_path.parent / (
        f".{output_model_path.name}.patch-{uuid.uuid4().hex}"
    )
    file_copier = _ModelFileCopier(copy_mode)
    try:
        print(f"Copying source model to temporary output: {staging_path}")
        shutil.copytree(
            source_model_path,
            staging_path,
            copy_function=file_copier,
        )
        print(
            "Model copy complete: "
            f"reflinked={file_copier.reflinked}, "
            f"hardlinked={file_copier.hardlinked}, "
            f"fully_copied={file_copier.copied}"
        )
        patch_model_weights(
            trained_state,
            str(staging_path),
            layout=layout,
            data_parallel_rank=data_parallel_rank,
            markov_replica_atol=markov_replica_atol,
            allow_markov_w1_one_ulp=allow_markov_w1_one_ulp,
            markov_w1_max_differing_elements=markov_w1_max_differing_elements,
        )
        os.replace(staging_path, output_model_path)
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)
    print(f"Patched model written to: {output_model_path}")
