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

import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch


RESUME_METADATA_NAME = "standalone_resume.json"
CONSUMED_SEQUENCE_NAME = "consumed_sequence_nos.pt"
RESUME_SCHEMA_VERSION = 1


def build_input_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    input_path = Path(path).resolve()
    stat = input_path.stat()
    return {
        "path": str(input_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def save_standalone_resume(
    checkpoint_path: str | os.PathLike[str],
    consumed_sequence_nos: Iterable[int] | torch.Tensor,
    *,
    optimizer_step: int,
    input_path: str | os.PathLike[str],
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
        "input_fingerprint": build_input_fingerprint(input_path),
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
    if input_path is not None:
        saved_fingerprint = metadata.get("input_fingerprint")
        current_fingerprint = build_input_fingerprint(input_path)
        if saved_fingerprint != current_fingerprint:
            raise ValueError(
                "Standalone resume input file changed since the checkpoint was saved: "
                f"saved={saved_fingerprint!r}, current={current_fingerprint!r}"
            )
    return consumed, metadata


__all__ = [
    "CONSUMED_SEQUENCE_NAME",
    "RESUME_METADATA_NAME",
    "build_input_fingerprint",
    "load_standalone_resume",
    "save_standalone_resume",
]
