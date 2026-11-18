# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Cold-load standard DSpark artifacts through the native Ascend weight loader.

The artifact contains draft weights, not a full target. Frozen shared token
weights and any unexported native confidence head come from the explicit base
checkpoint, matching the target-sharing policy used during training.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from verl_speco.models.deepseek_v4.artifact import iter_draft_weights


def artifact_sources(directory, base_checkpoint):
    root, base = Path(directory), Path(base_checkpoint)
    metadata = json.loads((root / "metadata.json").read_text())
    if (
        metadata.get("complete") is not True
        or metadata.get("weight_layout") != "deepseek_v4_dspark_mtp"
    ):
        raise ValueError("Incomplete or unsupported DSpark artifact")
    config = json.loads((root / "config.json").read_text())
    base_config = json.loads((base / "config.json").read_text())
    normalized = {
        k: v
        for k, v in config.items()
        if k not in {"speco_draft_only", "speco_weight_layout"}
    }
    if normalized != base_config:
        raise ValueError("Artifact and explicit base model configurations differ")
    serialization = metadata.get("serialization")
    if serialization not in {"pytorch", "safetensors"}:
        raise ValueError("Unsupported DSpark artifact serialization")
    index_name = (
        "pytorch_model.bin.index.json"
        if serialization == "pytorch"
        else "model.safetensors.index.json"
    )
    keys = set(json.loads((root / index_name).read_text())["weight_map"])
    if not keys or any(not k.startswith("mtp.") for k in keys):
        raise ValueError("Expected canonical mtp.* artifact weights")
    last = int(config.get("n_mtp_layers", config.get("dspark_num_mtp_layers", 3))) - 1
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
    else:
        mapping = {}
        for path in sorted(base.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for name in source.keys():  # noqa: SIM118 -- safetensors is not a mapping
                    if name in mapping:
                        raise ValueError(f"Duplicate base model weight: {name}")
                    mapping[name] = path.name
    unsupported = sorted(
        name for name in keys if ".confidence_head." in name and name not in mapping
    )
    if unsupported:
        raise ValueError(
            f"Artifact confidence schema is unsupported by the base runtime: {unsupported}"
        )
    fallback = {}
    for artifact_key, destination, candidates in [
        (
            "mtp.0.embed.weight",
            "embed.weight",
            ["embed.weight", "model.embed_tokens.weight", "embed_tokens.weight"],
        ),
        (
            f"mtp.{last}.head.weight",
            "head.weight",
            ["head.weight", "lm_head.weight", "model.lm_head.weight"],
        ),
    ]:
        if artifact_key not in keys:
            source_key = next((name for name in candidates if name in mapping), None)
            if source_key is None:
                raise ValueError(f"Missing shared base weight for {artifact_key}")
            fallback[destination] = source_key
    for name in mapping:
        if name.startswith(f"mtp.{last}.confidence_head.") and name not in keys:
            fallback[name] = name
    return config, mapping, fallback


def _weight_stream(directory, base, mapping, fallback):
    for name, tensor in iter_draft_weights(directory):
        if tensor.dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError(f"Only BF16/FP32 artifact weights are supported: {name}")
        yield name, tensor
    for destination, source_key in fallback.items():
        filename = mapping[source_key]
        if Path(filename).name != filename or (base / filename).is_symlink():
            raise ValueError("Unsafe base weight shard path")
        with safe_open(str(base / filename), framework="pt", device="cpu") as source:
            tensor = source.get_tensor(source_key)
            if (
                tensor.dtype not in {torch.bfloat16, torch.float32}
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid shared base weight: {source_key}")
            yield destination, tensor


def _finish_loading(wrapper):
    # Reuse the same processing order as the existing full-checkpoint loader.
    from verl_speco.models.deepseek_v4.modeling_dspark import (
        _install_fused_eh_projection_collective,
        _install_tp_sharded_attention_projections,
        _install_tp_sharded_eh_projections,
    )

    wrapper._dequantize_weights_for_training()
    wrapper._validate_loaded_training_weight_format()
    wrapper._process_weights_after_loading()
    wrapper.using_tp_sharded_eh_projections = _install_tp_sharded_eh_projections(
        wrapper.model
    )
    wrapper.using_fused_eh_projection_collective = (
        _install_fused_eh_projection_collective(wrapper.model)
    )
    wrapper.using_tp_sharded_attention_projections = (
        _install_tp_sharded_attention_projections(wrapper.model)
    )


def load_standard_draft_artifact(wrapper, directory, *, base_checkpoint):
    """Initialize a fresh wrapper; this is not a live hot-update API."""
    if wrapper._weights_postprocessed or getattr(
        wrapper, "_artifact_load_failed", False
    ):
        raise RuntimeError("Standard artifact loading requires a fresh wrapper")
    config, mapping, fallback = artifact_sources(directory, base_checkpoint)
    if (
        type(wrapper.runtime_contract).from_config(SimpleNamespace(**config))
        != wrapper.runtime_contract
    ):
        raise ValueError("Artifact does not match the runtime contract")
    aliases = dict(wrapper.model.named_parameters(remove_duplicate=False))
    required = {id(p) for p in aliases.values()}
    try:
        with wrapper._set_current_vllm_config(wrapper.vllm_config):
            loaded = wrapper.model.load_weights(
                _weight_stream(directory, Path(base_checkpoint), mapping, fallback)
            )
            covered = {id(aliases[name]) for name in loaded if name in aliases}
            if covered != required:
                missing = sorted(
                    name for name, p in aliases.items() if id(p) not in covered
                )
                raise ValueError(
                    f"Artifact did not initialize all native parameters: {missing}"
                )
            _finish_loading(wrapper)
        wrapper._keep_hc_parameters_float32()
        wrapper.freeze_shared_token_weights()
        wrapper._ensure_vllm_ascend_tp_equivalent(
            stage="standard_artifact_load", force=True
        )
        wrapper.token_weight_sources = {
            "checkpoint_path": str(Path(directory).resolve()),
            "base_checkpoint_path": str(Path(base_checkpoint).resolve()),
            "base_fallback": fallback,
            "live_target_alias": False,
        }
    except BaseException:
        wrapper._artifact_load_failed = True
        raise
    return {
        "loaded": len(covered),
        "source": str(Path(directory).resolve()),
        "base_fallback": fallback,
    }
