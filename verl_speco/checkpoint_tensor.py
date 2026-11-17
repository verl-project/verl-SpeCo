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
"""Read a target checkpoint tensor without importing drafter model implementations."""

import glob
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

CHECKPOINT_FINGERPRINT_PREFIX = "target-checkpoint-v1-sha256-"


def resolve_checkpoint_path(model_path: str) -> Path:
    """Use an existing directory or resolve a Hub model to a single snapshot."""
    path = Path(model_path)
    return path if path.is_dir() else Path(snapshot_download(repo_id=model_path))


def checkpoint_fingerprint(path: Path) -> str:
    """Hash the config and all checkpoint weight files without loading tensors."""
    files = {"config.json"}
    indexes = sorted(path.glob("*.index.json"))
    if indexes:
        for index in indexes:
            with index.open(encoding="utf-8") as stream:
                weight_map = json.load(stream).get("weight_map", {})
            if weight_map:
                files.add(index.name)
                for name in set(weight_map.values()):
                    relative = Path(name)
                    if (
                        relative.is_absolute()
                        or relative.drive
                        or ".." in relative.parts
                    ):
                        raise ValueError(f"Invalid checkpoint shard path: {name!r}")
                    files.add(name)
    else:
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (path / name).is_file():
                files.add(name)
                break
    if len(files) == 1:
        raise FileNotFoundError(f"No checkpoint weights found in {path}")

    digest = hashlib.sha256()
    for name in sorted(files):
        file_digest = hashlib.sha256()
        with (path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(file_digest.digest())
    return CHECKPOINT_FINGERPRINT_PREFIX + digest.hexdigest()


# Multimodal / conditional-generation targets (e.g. Qwen3.5/3.6 MoE) nest the
# text backbone under ``model.language_model`` (or similar) while the module
# tree built from the config may name it ``model``. Try the common aliases.
_KEY_PREFIX_ALIASES = (
    ("model.", "model.language_model."),
    ("model.", "language_model.model."),
    ("model.", "model.text_model."),
)


def _candidate_keys(key: str) -> list[str]:
    candidates = [key]
    for prefix, alias in _KEY_PREFIX_ALIASES:
        if key.startswith(prefix):
            candidates.append(alias + key[len(prefix) :])
    return candidates


def _load_checkpoint_tensor(model_path: str, key: str) -> torch.Tensor:
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    index_paths = sorted(glob.glob(os.path.join(model_path, "*.index.json")))
    if index_paths:
        # Quantized DeepSeek-V4 packages ship multiple indices (a drafter/model
        # index plus a quantized-weights index); search each for the key instead
        # of rejecting the checkpoint.
        for index_path in index_paths:
            with open(index_path, encoding="utf-8") as f:
                index_json = json.load(f)
            weight_map = index_json.get("weight_map", {})
            for candidate in _candidate_keys(key):
                if candidate in weight_map:
                    resolved_key = candidate
                    break
            else:
                continue
            ckpt_file = os.path.join(model_path, weight_map[resolved_key])
            if ckpt_file.endswith(".safetensors"):
                with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                    return f.get_tensor(resolved_key)
            return torch.load(ckpt_file, map_location="cpu", weights_only=True)[
                resolved_key
            ]
        raise KeyError(
            f"Tensor {key!r} is not present in any checkpoint index for {model_path}"
        )

    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for candidate in _candidate_keys(key):
                if candidate in available:
                    return f.get_tensor(candidate)
            # Preserve the native safetensors error (and message) when neither
            # the requested key nor an alias is present.
            return f.get_tensor(key)

    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        for candidate in _candidate_keys(key):
            if candidate in state_dict:
                return state_dict[candidate]
        raise KeyError(f"Tensor {key!r} is not present in {pytorch_path}")

    raise FileNotFoundError(
        f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
    )
