# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Validate every replicated DSpark source before publishing a merged artifact."""

from pathlib import Path

import torch

from verl_speco.models.deepseek_v4 import export as native


def replicated_export_keys(state, layer_indices, projection_shards):
    """Exclude actual TP shards and duplicate native module registrations."""
    sharded = set(projection_shards)
    for layer in layer_indices:
        prefix = f"model.layers.{layer}."
        sharded.update(
            prefix + suffix
            for suffix in (
                "self_attn.attn_sink",
                "self_attn.wq_b.weight",
                "self_attn.wo_a.weight",
                "self_attn.wo_b.weight",
                "mlp.shared_experts.gate_up_proj.weight",
                "mlp.shared_experts.down_proj.weight",
            )
        )
    prefixes = tuple(f"model.layers.{layer}." for layer in layer_indices)
    return sorted(
        name
        for name, tensor in state.items()
        if torch.is_tensor(tensor)
        and name not in sharded
        and (
            (
                name.startswith(prefixes)
                and ".self_attn.dsa_attn." not in name
                and ".mlp.experts." not in name
            )
            or name in {"model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale"}
            or name.startswith("confidence_head.")
        )
    )


def validate_tp_export_replicas(checkpoint, *, dp_rank, target_shapes):
    """Reject divergent replicas instead of silently selecting TP rank zero.

    Memory-map one rank at a time. This check never modifies a checkpoint,
    averages weights, or changes optimizer recovery identity.
    """
    root = Path(checkpoint)
    rank_files = native._rank_state_files(root)
    if len(rank_files) > 1 and not (root / "parallel_topology.json").exists():
        raise ValueError("Multiple rank states require parallel_topology.json")
    paths = native._topology_rank_state_files(root, rank_files, dp_rank)
    if not paths:
        raise FileNotFoundError(f"No native rank states in {root}")

    def load(path):
        return native._canonical_state_dict(
            torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        )

    reference = load(paths[0])
    layers = native._dspark_layer_indices(reference)
    shards, _ = native._dspark_projection_layout(
        reference, layers, len(paths), target_shapes
    )
    keys = replicated_export_keys(reference, layers, shards)
    for key in keys:
        if not bool(torch.isfinite(reference[key]).all()):
            raise ValueError(f"Non-finite replicated export tensor: {key}")
    for path in paths[1:]:
        candidate = load(path)
        if native._dspark_layer_indices(candidate) != layers:
            raise ValueError(f"Inconsistent DSpark layers in {path}")
        native._validate_replicated_tensors(
            reference, candidate, keys, candidate_path=path
        )
        del candidate
    return {"tp_ranks": len(paths), "replicated_tensors": len(keys)}
