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
"""Standalone-only data progress stored beside a drafter checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch

RESUME_METADATA_NAME = "standalone_resume.json"
CONSUMED_SEQUENCE_NAME = "consumed_sequence_nos.pt"
RESUME_SCHEMA_VERSION = 1
# Full content hashing is cheap enough for ordinary JSONL shards; very large
# shards are sampled deterministically so a multi-node resume does not depend
# on node-local mtimes or on reading terabytes at every checkpoint.
_HASH_FULL_LIMIT_BYTES = 64 * 1024 * 1024
_HASH_CHUNK_BYTES = 64 * 1024
_HASH_SAMPLE_COUNT = 6


def _file_content_fingerprint(file_path: Path) -> dict[str, Any]:
    size = int(file_path.stat().st_size)
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with file_path.open("rb") as stream:
        if size <= _HASH_FULL_LIMIT_BYTES:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        else:
            offsets = {0, max(size - _HASH_CHUNK_BYTES, 0)}
            for index in range(1, _HASH_SAMPLE_COUNT):
                offsets.add((size * index) // _HASH_SAMPLE_COUNT)
            for offset in sorted(offsets):
                stream.seek(offset)
                digest.update(stream.read(_HASH_CHUNK_BYTES))
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _fingerprint_identity(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match on resume; the mount path is informational only."""

    return {key: value for key, value in fingerprint.items() if key != "path"}


def build_input_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    input_path = Path(path).resolve()
    if input_path.is_dir():
        metadata_path = input_path / "metadata.json"
        manifest_path = input_path / "manifest.jsonl"
        if metadata_path.is_file() and manifest_path.is_file():
            fingerprints = {}
            for name, file_path in (
                ("metadata.json", metadata_path),
                ("manifest.jsonl", manifest_path),
            ):
                digest = hashlib.sha256()
                with file_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                fingerprints[name] = digest.hexdigest()
            return {
                "path": str(input_path),
                "format": "standalone_raw_hidden",
                "sampler_version": 1,
                "files": fingerprints,
            }
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if jsonl_files:
            return {
                "path": str(input_path),
                "format": "jsonl_token_replay",
                "files": {
                    file_path.name: _file_content_fingerprint(file_path)
                    for file_path in jsonl_files
                },
            }
        directory_files = sorted(
            file_path for file_path in input_path.rglob("*") if file_path.is_file()
        )
        return {
            "path": str(input_path),
            "format": "directory",
            "files": {
                file_path.relative_to(input_path).as_posix(): _file_content_fingerprint(
                    file_path
                )
                for file_path in directory_files
            },
        }
    return {
        "path": str(input_path),
        **_file_content_fingerprint(input_path),
    }


def save_standalone_resume(
    checkpoint_path: str | os.PathLike[str],
    consumed_sequence_nos: Iterable[int] | torch.Tensor,
    *,
    optimizer_step: int,
    input_path: str | os.PathLike[str],
    producer_shuffle: dict[str, Any] | None = None,
    data_progress: dict[str, Any] | None = None,
    loader_config: dict[str, Any] | None = None,
    input_fingerprint: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(consumed_sequence_nos, torch.Tensor):
        values = consumed_sequence_nos.detach().to(device="cpu", dtype=torch.int64)
    else:
        values = torch.tensor(
            sorted({int(value) for value in consumed_sequence_nos}),
            dtype=torch.int64,
        )
    values = torch.unique(values.flatten(), sorted=True)
    if values.numel() and int(values[0]) < 0:
        raise ValueError("consumed sequence numbers must be non-negative")

    tensor_path = checkpoint_dir / CONSUMED_SEQUENCE_NAME
    tensor_temporary = tensor_path.with_suffix(tensor_path.suffix + ".incomplete")
    torch.save(values, tensor_temporary)
    os.replace(tensor_temporary, tensor_path)

    metadata = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "optimizer_step": int(optimizer_step),
        "consumed_count": int(values.numel()),
        "consumed_sequence_file": CONSUMED_SEQUENCE_NAME,
        "input_fingerprint": (
            input_fingerprint
            if input_fingerprint is not None
            else build_input_fingerprint(input_path)
        ),
    }
    if producer_shuffle is not None:
        # Consumed sequence numbers are position-based, so their meaning
        # depends on the producer shuffle settings that produced them. Saved
        # here so a resumed Producer can refuse a mismatched configuration
        # instead of silently reordering the stream.
        metadata["producer_shuffle"] = {
            "shuffle": bool(producer_shuffle.get("shuffle", False)),
            "shuffle_seed": int(producer_shuffle.get("shuffle_seed", 42)),
        }
    if data_progress is not None:
        # Deterministic file stores do not need (or maintain) a per-sample
        # consumed set; their loader order is a pure function of
        # ``(seed, epoch, store)``. Persist the batch cursor instead so a
        # resumed run continues at the exact epoch/batch position.
        epoch = int(data_progress.get("epoch", 0) or 0)
        batch_index = int(data_progress.get("batch_index", 0) or 0)
        if epoch < 0 or batch_index < 0:
            raise ValueError(f"data_progress must be non-negative: {data_progress!r}")
        progress_metadata: dict[str, Any] = {
            "epoch": epoch,
            "batch_index": batch_index,
        }
        read_errors = data_progress.get("read_errors")
        if read_errors is not None:
            progress_metadata["read_errors"] = int(read_errors)
        read_errors_exact = data_progress.get("read_errors_exact")
        if read_errors_exact is not None:
            progress_metadata["read_errors_exact"] = bool(read_errors_exact)
        metadata["data_progress"] = progress_metadata
    if loader_config is not None:
        # The cursor is a batch ordinal over the filtered key order, so every
        # setting that changes that order (batch size, sharding, shuffle/seed,
        # filters, ...) must match on resume.
        metadata["data_loader"] = {
            str(key): value for key, value in loader_config.items()
        }
    metadata_path = checkpoint_dir / RESUME_METADATA_NAME
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".incomplete")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)


def load_standalone_resume(
    checkpoint_path: str | os.PathLike[str] | None,
    *,
    input_path: str | os.PathLike[str] | None = None,
    current_fingerprint: dict[str, Any] | None = None,
) -> tuple[set[int], dict[str, Any] | None]:
    if not checkpoint_path:
        return set(), None
    checkpoint_dir = Path(checkpoint_path)
    metadata_path = checkpoint_dir / RESUME_METADATA_NAME
    if not metadata_path.is_file():
        return set(), None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != RESUME_SCHEMA_VERSION:
        raise ValueError(f"Unsupported standalone resume metadata: {metadata_path}")
    tensor_name = metadata.get("consumed_sequence_file")
    if tensor_name != CONSUMED_SEQUENCE_NAME:
        raise ValueError(f"Invalid consumed sequence file in {metadata_path}")
    tensor_path = checkpoint_dir / tensor_name
    try:
        values = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except TypeError:
        values = torch.load(tensor_path, map_location="cpu")
    if not isinstance(values, torch.Tensor) or values.dtype != torch.int64:
        raise ValueError(f"Invalid consumed sequence tensor: {tensor_path}")
    values = values.flatten()
    if values.numel() and int(values[0]) < 0:
        raise ValueError(f"Negative consumed sequence number in {tensor_path}")
    consumed = {int(value) for value in values.tolist()}
    if len(consumed) != int(metadata.get("consumed_count", -1)):
        raise ValueError(f"Consumed sequence count mismatch in {checkpoint_dir}")
    if input_path is not None or current_fingerprint is not None:
        saved_fingerprint = metadata.get("input_fingerprint")
        if current_fingerprint is None:
            if input_path is None:
                raise ValueError("load_standalone_resume requires input_path")
            current_fingerprint = build_input_fingerprint(input_path)
        if not isinstance(saved_fingerprint, dict) or _fingerprint_identity(
            saved_fingerprint
        ) != _fingerprint_identity(current_fingerprint):
            raise ValueError(
                "Standalone resume input file changed since the checkpoint was saved: "
                f"saved={saved_fingerprint!r}, current={current_fingerprint!r}"
            )
    return consumed, metadata


def validate_standalone_loader_config(
    saved: dict[str, Any] | None, current: dict[str, Any]
) -> None:
    """Fail closed when the saved cursor no longer describes this loader.

    ``data_progress`` is a batch ordinal over the filtered key order. Batch
    size, sharding, shuffle/seed, step windows and read-side filters all change
    that order, so resuming with different settings would silently repeat or
    skip samples instead of continuing at the saved position.
    """

    if saved is None:
        return
    mismatched = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    }
    if mismatched:
        raise ValueError(
            "Standalone resume loader configuration changed since the checkpoint "
            f"was saved: {mismatched}"
        )


__all__ = [
    "CONSUMED_SEQUENCE_NAME",
    "RESUME_METADATA_NAME",
    "build_input_fingerprint",
    "load_standalone_resume",
    "save_standalone_resume",
    "validate_standalone_loader_config",
]
