# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Strict MTP export and initialization using existing native TP conversion."""

import json
from pathlib import Path

import torch

from verl_speco.integration.vllm_ascend_mtp_config import mtp_implementation_fingerprint
from verl_speco.models.deepseek_v4 import export as native
from verl_speco.models.deepseek_v4.artifact import (
    iter_draft_weights,
    save_draft_artifact,
)
from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract


def validate_mtp_replicas(checkpoint, dp_rank):
    root = Path(checkpoint)
    if not (root / "parallel_topology.json").is_file():
        raise ValueError("MTP export requires explicit parallel_topology.json")
    paths = native._topology_rank_state_files(
        root, native._rank_state_files(root), dp_rank
    )
    reference = native._canonical_state_dict(
        torch.load(paths[0], map_location="cpu", weights_only=True, mmap=True)
    )
    prefix = native.LAYER_PREFIX
    sharded = {
        prefix + name
        for name in (
            "mtp_block.self_attn.attn_sink",
            "mtp_block.self_attn.wq_b.weight",
            "mtp_block.self_attn.wo_a.weight",
            "mtp_block.self_attn.wo_b.weight",
            "mtp_block.mlp.shared_experts.gate_up_proj.weight",
            "mtp_block.mlp.shared_experts.down_proj.weight",
            "mtp_block.mlp.experts.w13_weight",
            "mtp_block.mlp.experts.w2_weight",
            "shared_head.head.weight",
        )
    }
    projection_shards = {}
    if len(paths) > 1:
        native._add_eh_projection_tp_shards(
            projection_shards, reference, tp_size=len(paths), source=paths[0]
        )
        native._add_attention_projection_tp_shards(
            projection_shards, reference, tp_size=len(paths), source=paths[0]
        )
    sharded.update(projection_shards)
    # MoERunner exposes the same shared-expert modules again through an
    # internal container in 0.27. Per-rank CPU serialization copies aliases,
    # so verify their values against the public TP shard within each rank.
    aliases = {
        name: name.replace(
            ".mlp.experts._shared_experts._layer.", ".mlp.shared_experts."
        )
        for name in reference
        if ".mlp.experts._shared_experts._layer." in name
    }

    def validate_aliases(state, path):
        for alias, canonical in aliases.items():
            if canonical not in sharded or canonical not in state or alias not in state:
                raise ValueError(f"Unknown MTP shared-expert alias: {alias} in {path}")
            if not torch.equal(state[alias], state[canonical]):
                raise ValueError(f"MTP shared-expert alias differs: {alias} in {path}")

    validate_aliases(reference, paths[0])
    sharded.update(aliases)
    keys = sorted(
        name
        for name in reference
        if name.startswith(prefix)
        and name not in sharded
        and ".self_attn.dsa_attn." not in name
    )
    if not keys or not native.VLLM_ASCEND_REQUIRED_TRAINABLE_SOURCES.issubset(
        reference
    ):
        raise ValueError("MTP checkpoint is missing required trainable tensors")
    for name in keys:
        if not bool(torch.isfinite(reference[name]).all()):
            raise ValueError(f"Non-finite MTP replica: {name}")
    for path in paths[1:]:
        state = native._canonical_state_dict(
            torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        )
        validate_aliases(state, path)
        native._validate_replicated_tensors(reference, state, keys, candidate_path=path)
    return {"tp_ranks": len(paths), "replicated_tensors": len(keys)}


def export_mtp_checkpoint(
    *,
    checkpoint,
    model,
    output,
    dp_rank=0,
    source_root=None,
    format="draft-pytorch",
    max_shard_bytes=2 * 1024**3,
):
    del source_root
    root = Path(checkpoint).resolve(strict=True)
    metadata = json.loads((root / "speco_ds4.json").read_text())
    contract = DeepSeekV4MTPContract.from_model(model)
    if (
        metadata["identity"]["contract"] != contract.identity()
        or metadata["identity"]["source_fingerprint"]
        != mtp_implementation_fingerprint()
    ):
        raise ValueError("MTP export checkpoint model/implementation differs")
    if not 0 <= dp_rank < int(metadata["topology"]["data_parallel_size"]):
        raise ValueError("MTP export DP rank is outside the saved topology")
    replica_report = validate_mtp_replicas(root, dp_rank)
    if format not in {"draft-pytorch", "draft-safetensors"}:
        raise ValueError("MTP export supports draft-pytorch or draft-safetensors")
    weights = native._load_export_weights(
        root, "vllm_ascend", data_parallel_rank=dp_rank
    )
    result = save_draft_artifact(
        weights,
        output=output,
        config=json.loads((Path(model) / "config.json").read_text()),
        step=metadata["progress"]["global_step"],
        serialization="pytorch" if format == "draft-pytorch" else "safetensors",
        max_shard_bytes=max_shard_bytes,
        weight_layout="deepseek_v4_mtp_v1",
    )
    report = {
        **result,
        "algorithm": "MTP",
        "replicas": replica_report,
        "contract": contract.identity(),
    }
    (Path(output) / "speco_export_report.json").write_text(json.dumps(report, indent=2))
    return report


def load_mtp_artifact(model, directory, *, base_checkpoint):
    from verl_speco.models.deepseek_v4.artifact_loading import _finish_loading
    from verl_speco.models.target.deepseek_v4_head import _find_safetensor_weights

    path, base = Path(directory), Path(base_checkpoint)
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata.get("weight_layout") != "deepseek_v4_mtp_v1"
        or metadata.get("complete") is not True
    ):
        raise ValueError("Expected a complete MTP artifact, not a DSpark draft")
    actual_config = json.loads((path / "config.json").read_text())
    actual_config = {
        k: v
        for k, v in actual_config.items()
        if k not in {"speco_draft_only", "speco_weight_layout"}
    }
    if actual_config != json.loads((base / "config.json").read_text()):
        raise ValueError("MTP artifact and explicit base configurations differ")
    wrapper = model.model
    if wrapper._weights_postprocessed or getattr(
        wrapper, "_artifact_load_failed", False
    ):
        raise RuntimeError(
            "MTP artifact initialization requires a fresh native wrapper"
        )
    shared, sources = _find_safetensor_weights(
        str(base),
        {
            "model.embed_tokens.weight": (
                "embed.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ),
            "model.layers.0.shared_head.head.weight": (
                "head.weight",
                "lm_head.weight",
                "model.lm_head.weight",
            ),
        },
    )
    aliases = dict(wrapper.model.named_parameters(remove_duplicate=False))
    try:
        with wrapper._set_current_vllm_config(wrapper.vllm_config):
            loaded = set(wrapper.model.load_weights(iter_draft_weights(path)))
            loaded.update(
                wrapper._load_shared_main_weights(shared, protected_names=loaded)
            )
            required = {id(p) for p in aliases.values()}
            covered = {id(aliases[name]) for name in loaded if name in aliases}
            if required != covered:
                missing = sorted(
                    name for name, p in aliases.items() if id(p) not in covered
                )
                raise ValueError(
                    f"MTP artifact did not initialize every native parameter: {missing}"
                )
            _finish_loading(wrapper)
        wrapper._keep_hc_parameters_float32()
        wrapper._ensure_vllm_ascend_tp_equivalent(stage="mtp_artifact_load", force=True)
    except BaseException:
        wrapper._artifact_load_failed = True
        raise
    return {"loaded": len(loaded), "source": str(path), "base_fallback": sources}
