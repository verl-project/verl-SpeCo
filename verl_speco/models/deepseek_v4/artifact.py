# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standard tensor files for DSpark draft weights (not a full target model)."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def iter_draft_weights(directory):
    """Read one standard shard at a time; reject incomplete/malformed artifacts."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("complete") is not True or metadata.get("weight_layout") not in {
        "deepseek_v4_dspark_mtp",
        "deepseek_v4_mtp_v1",
    }:
        raise ValueError("Incomplete or unsupported DSpark artifact")
    serialization = metadata["serialization"]
    if serialization not in {"safetensors", "pytorch"}:
        raise ValueError("Unsupported DSpark serialization")
    index_name = (
        "model.safetensors.index.json"
        if serialization == "safetensors"
        else "pytorch_model.bin.index.json"
    )
    index = json.loads((root / index_name).read_text())
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("Empty DSpark weight index")
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Unsafe DSpark shard path")
        path = root / filename
        if path.is_symlink():
            raise ValueError("DSpark shard must not be a symlink")
        state = (
            load_file(str(path))
            if serialization == "safetensors"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        if set(state) != {
            key for key, value in weight_map.items() if value == filename
        }:
            raise ValueError("DSpark shard differs from weight index")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid DSpark weight: {name}")
            yield name, tensor


def save_draft_artifact(
    weights,
    *,
    output,
    config,
    step,
    serialization="safetensors",
    max_shard_bytes=2 * 1024**3,
    weight_layout="deepseek_v4_dspark_mtp",
):
    """Atomically publish tensor files and SpeCo completion metadata after readback."""
    if (
        serialization not in {"safetensors", "pytorch"}
        or max_shard_bytes <= 0
        or weight_layout not in {"deepseek_v4_dspark_mtp", "deepseek_v4_mtp_v1"}
    ):
        raise ValueError("Invalid draft serialization or shard size")
    if not weights or any(not name.startswith("mtp.") for name in weights):
        raise ValueError("Expected canonical DSpark mtp.* tensors")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent)
    )
    try:
        groups, current, size = [], [], 0
        total = 0
        for name, tensor in sorted(weights.items()):
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid DSpark weight: {name}")
            nbytes = tensor.numel() * tensor.element_size()
            if current and size + nbytes > max_shard_bytes:
                groups.append(current)
                current, size = [], 0
            current.append(name)
            size += nbytes
            total += nbytes
        groups.append(current)
        weight_map = {}
        for number, names in enumerate(groups, 1):
            filename = (
                f"model-{number:05d}-of-{len(groups):05d}.safetensors"
                if serialization == "safetensors"
                else f"pytorch_model-{number:05d}-of-{len(groups):05d}.bin"
            )
            shard = {
                name: weights[name].detach().to("cpu").contiguous().clone()
                for name in names
            }
            if serialization == "safetensors":
                save_file(shard, str(temporary / filename), metadata={"format": "pt"})
            else:
                torch.save(shard, temporary / filename)
            weight_map.update({name: filename for name in names})
            del shard
        index_name = (
            "model.safetensors.index.json"
            if serialization == "safetensors"
            else "pytorch_model.bin.index.json"
        )
        (temporary / index_name).write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
            )
        )
        draft_config = dict(config)
        draft_config["speco_draft_only"] = True
        draft_config["speco_weight_layout"] = weight_layout
        (temporary / "config.json").write_text(json.dumps(draft_config, indent=2))
        metadata = {
            "step": int(step),
            "format": "pretrained_drafter_checkpoint",
            "serialization": serialization,
            "complete": True,
            "weight_layout": weight_layout,
            "training_backend": "vllm_ascend",
            "optimizer": None,
            "trainer_state": {},
            "inference_validated": False,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2))
        seen = set()
        for name, tensor in iter_draft_weights(temporary):
            expected = weights[name].detach().to("cpu")
            if (
                tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
                or not torch.equal(tensor, expected)
            ):
                raise ValueError(f"Draft artifact readback differs: {name}")
            seen.add(name)
        if seen != set(weights):
            raise ValueError("Draft artifact readback is incomplete")
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
        return {
            "status": "passed",
            "stage": "export",
            "output": str(output.resolve()),
            "weights_verified": True,
            "inference_validated": False,
            "num_tensors": len(seen),
            "serialization": serialization,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
