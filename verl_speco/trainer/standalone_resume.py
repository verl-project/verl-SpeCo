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

import gzip
import json
import hashlib
import os
import re
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


RESUME_METADATA_NAME = "standalone_resume.json"
CONSUMED_SEQUENCE_NAME = "consumed_sequence_nos.pt"
CONSUMED_GZIP_NAME = "consumed_sequence_nos.i64.gz"
RESUME_SCHEMA_VERSION = 1
_COMPRESSED_SCHEMA_VERSION = 2
_IMMUTABLE_SCHEMA_VERSION = 3
_COMPRESSED_CODEC = "gzip-delta-int64-le-v1"
_MAX_COMPRESSED_ENTRIES = 100_000_000
_MAX_ENCODED_BYTES = 1024**3
_CHUNK_ENTRIES = 131_072


def producer_data_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture Producer settings that alter selected tokens or sample admission."""
    defaults = {
        "max_sequence_length": 8192, "max_feature_length": 512,
        "generation_max_tokens": 511, "min_supervised_tokens": 1,
        "on_error": "skip", "on_missing_response": "skip",
        "on_truncated_supervision": "keep", "parser_strict_roles": False,
    }
    settings = {key: config.get(key, default) for key, default in defaults.items()}
    for key in ("max_sequence_length", "max_feature_length", "generation_max_tokens", "min_supervised_tokens"):
        settings[key] = int(settings[key] or 0)
    for key in ("on_error", "on_missing_response", "on_truncated_supervision"):
        settings[key] = str(settings[key] or defaults[key])
    settings["on_missing_response"] = settings["on_missing_response"].lower()
    settings["parser_strict_roles"] = bool(settings["parser_strict_roles"])
    stable_source_ids = config.get("stable_source_ids", False)
    if type(stable_source_ids) is not bool:
        raise ValueError("stable_source_ids must be a boolean")
    if stable_source_ids:
        settings["stable_source_ids"] = True
    if config.get("source_route_wave_samples", 0):
        settings["source_routing"] = {
            "wave_samples": config["source_route_wave_samples"],
            "max_samples": config.get("source_route_max_samples", 64),
            "max_tokens": config.get("source_route_max_tokens", 1048576),
            "max_sequence_tokens": config.get("source_route_max_sequence_tokens", 65536),
            "endpoints": list(config.get("vllm_endpoints") or []),
        }
    settings["render_boundary"] = bool((config.get("render_boundary") or {}).get("enabled", False))
    return {"version": 1, "settings": settings}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_cache_content_sha256(directory: str | os.PathLike[str]) -> str:
    """Hash all regular files by relative path and contents, independent of cache location."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("Content-verified model cache must be a local directory")
    files, directories = [], []
    def fail(error):
        raise error
    for current, children, names in os.walk(root, followlinks=False, onerror=fail):
        current_path = Path(current)
        directories.append((current_path, current_path.stat().st_mtime_ns))
        children.sort()
        for child in children:
            if (current_path / child).is_symlink():
                raise ValueError("Model cache cannot contain directory symlinks")
        for name in sorted(names):
            path = current_path / name
            if not path.is_file():
                raise ValueError("Model cache contains a non-regular file")
            files.append(path)
            if len(files) > 100_000:
                raise ValueError("Model cache exceeds the file count limit")
    if (not (root / "config.json").is_file()
            or not any(path.suffix in {".safetensors", ".bin", ".pt", ".pth"} for path in files)):
        raise ValueError("Model cache must contain config.json and model files")
    digest = hashlib.sha256(b"speco-model-cache-v1\0")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        link_before = path.lstat()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(before.st_size.to_bytes(8, "big"))
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        link_after, current = path.lstat(), path.stat()
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        # Windows may report different ctime semantics for fstat and stat.
        if (signature(before) != signature(after) or signature(before)[:4] != signature(current)[:4]
                or signature(link_before) != signature(link_after)):
            raise ValueError("Model cache changed while being fingerprinted")
    if any(path.stat().st_mtime_ns != mtime for path, mtime in directories):
        raise ValueError("Model cache directory changed while being fingerprinted")
    return digest.hexdigest()


def _write_compressed_sequences(values: torch.Tensor, path: Path) -> str:
    """Stream sorted int64 deltas without materializing the full decoded blob."""
    import numpy as np

    if values.numel() > _MAX_COMPRESSED_ENTRIES:
        raise ValueError("Compressed resume progress exceeds the entry limit")
    digest = hashlib.sha256()
    previous = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
            for start in range(0, values.numel(), _CHUNK_ENTRIES):
                chunk = values[start : start + _CHUNK_ENTRIES].numpy().astype("<i8", copy=False)
                deltas = np.empty(chunk.size, dtype="<i8")
                deltas[0] = int(chunk[0]) - previous
                np.subtract(chunk[1:], chunk[:-1], out=deltas[1:])
                previous = int(chunk[-1])
                payload = deltas.tobytes()
                digest.update(payload)
                stream.write(payload)
        raw.flush()
        os.fsync(raw.fileno())
    if path.stat().st_size > _MAX_ENCODED_BYTES:
        raise ValueError("Compressed resume progress exceeds the encoded byte limit")
    return digest.hexdigest()


def _read_compressed_sequences(path: Path, metadata: dict[str, Any]) -> set[int]:
    """Validate checksums, decoded length and strictly increasing int64 values."""
    import numpy as np

    count = metadata.get("consumed_count")
    decoded_bytes = metadata.get("consumed_sequence_decoded_bytes")
    decoded_sha = metadata.get("consumed_sequence_decoded_sha256")
    if (type(count) is not int or count < 0 or count > _MAX_COMPRESSED_ENTRIES
            or type(decoded_bytes) is not int or decoded_bytes != count * 8
            or not isinstance(decoded_sha, str) or len(decoded_sha) != 64
            or any(character not in "0123456789abcdef" for character in decoded_sha)):
        raise ValueError("Invalid compressed resume progress descriptor")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Compressed resume progress must be a regular file")
    encoded_limit = min(_MAX_ENCODED_BYTES, max(4096, decoded_bytes * 2 + 4096))
    if path.stat().st_size > encoded_limit:
        raise ValueError("Compressed resume progress exceeds the encoded byte limit")
    digest = hashlib.sha256()
    consumed: set[int] = set()
    previous = 0
    seen = 0
    try:
        with gzip.open(path, "rb") as stream:
            while seen < count:
                chunk_count = min(_CHUNK_ENTRIES, count - seen)
                payload = stream.read(chunk_count * 8)
                if len(payload) != chunk_count * 8:
                    raise ValueError("Compressed resume progress is truncated")
                digest.update(payload)
                deltas = np.frombuffer(payload, dtype="<i8")
                if np.any(deltas < 0) or np.any(deltas[1 if seen == 0 else 0:] == 0):
                    raise ValueError("Invalid compressed resume sequence deltas")
                cumulative = np.cumsum(deltas, dtype=np.int64)
                if (np.any(cumulative < 0) or np.any(np.diff(cumulative) < 0)
                        or int(cumulative[-1]) > (2**63 - 1) - previous):
                    raise ValueError("Compressed resume sequence overflow")
                values = cumulative + previous
                consumed.update(values.tolist())
                previous = int(values[-1])
                seen += chunk_count
            if stream.read(1):
                raise ValueError("Compressed resume progress exceeds decoded size")
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("Invalid compressed resume progress") from exc
    if digest.hexdigest() != decoded_sha or len(consumed) != count:
        raise ValueError("Compressed resume progress checksum or count mismatch")
    return consumed


def build_input_fingerprint(
    path: str | os.PathLike[str], *, include_content_hash: bool = False,
) -> dict[str, Any]:
    input_path = Path(path).resolve()
    if input_path.is_dir():
        from verl_speco.producer.sequence_replay import SequenceReplayReader

        reader = SequenceReplayReader(input_path)
        try:
            return {"path": str(input_path), "manifest_sha256": reader.manifest_sha256}
        finally:
            reader.close()
    stat = input_path.stat()
    fingerprint = {
        "path": str(input_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_content_hash:
        if not input_path.is_file():
            raise ValueError(f"Standalone resume input must be a regular file: {input_path}")
        fingerprint["sha256"] = _file_sha256(input_path)
        after = input_path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        ):
            raise ValueError("Standalone resume input changed while being fingerprinted")
    return fingerprint


def input_content_identity(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Return a path-independent identity backed by a complete content digest."""
    if not isinstance(fingerprint, Mapping):
        raise ValueError("Standalone input fingerprint is missing")
    digest = fingerprint.get("manifest_sha256")
    if digest is not None:
        identity = {"kind": "replay_manifest", "sha256": digest}
    else:
        digest = fingerprint.get("sha256")
        size = fingerprint.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValueError("Standalone input fingerprint has invalid size")
        identity = {"kind": "file", "sha256": digest, "size_bytes": size}
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Standalone input fingerprint requires a SHA-256 digest")
    return identity


def validate_input_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("Standalone input identity is missing")
    kind = identity.get("kind")
    if kind == "file" and set(identity) == {"kind", "sha256", "size_bytes"}:
        reconstructed = input_content_identity({"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]})
    elif kind == "replay_manifest" and set(identity) == {"kind", "sha256"}:
        reconstructed = input_content_identity({"manifest_sha256": identity["sha256"]})
    else:
        raise ValueError("Standalone input identity is malformed")
    if reconstructed != identity:
        raise ValueError("Standalone input identity is malformed")
    return reconstructed


def validate_reference_lease_snapshot(snapshot):
    if not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int or snapshot["version"] != 1:
        raise ValueError("Invalid reference lease snapshot")
    digest = snapshot.get("sha256")
    payload = {key: value for key, value in snapshot.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 16 * 1024**2 or not isinstance(digest, str) or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("Reference lease snapshot checksum or size differs")
    incarnation = snapshot.get("incarnation")
    if not isinstance(incarnation, str) or not 1 <= len(incarnation) <= 256:
        raise ValueError("Invalid reference lease snapshot incarnation")
    if snapshot.get("drained") is True:
        if set(snapshot) != {"version", "incarnation", "drained", "sha256"}:
            raise ValueError("Invalid drained reference lease snapshot")
        return snapshot
    if (set(snapshot) != {"version", "run_id", "incarnation", "leases", "payloads", "acknowledgements", "sha256"}
            or not isinstance(snapshot["run_id"], str) or not snapshot["run_id"]
            or any(not isinstance(snapshot[key], list) for key in ("leases", "payloads", "acknowledgements"))):
        raise ValueError("Invalid reference lease snapshot structure")
    owners, phases, ack_counts, lease_keys = {}, {}, {}, {}
    for row in snapshot["leases"]:
        if (not isinstance(row, list) or len(row) != 3 or type(row[0]) is not int or row[0] < 0
                or row[0] in owners or not isinstance(row[1], str)
                or row[1] not in {"DISPATCH_INTENT", "LEASED", "MATERIALIZED", "RELEASE_INTENT", "RELEASED"}
                or not isinstance(row[2], str)):
            raise ValueError("Invalid snapshot lease record")
        identity = json.loads(row[2])
        if (not isinstance(identity, dict) or type(identity.get("owner_count")) is not int
                or not 0 <= identity["owner_count"] <= 65536 or not isinstance(identity.get("keys"), list)
                or not identity["keys"] or any(not isinstance(key, str) or not key for key in identity["keys"])
                or len(set(identity["keys"])) != len(identity["keys"])
                or not isinstance(identity.get("receipt"), str) or not identity["receipt"]):
            raise ValueError("Invalid snapshot lease identity")
        owners[row[0]] = identity["owner_count"]
        phases[row[0]], ack_counts[row[0]] = row[1], 0
        lease_keys[row[0]] = set(identity["keys"])
    acknowledgements = set()
    for row in snapshot["acknowledgements"]:
        if (not isinstance(row, list) or len(row) != 2 or any(type(value) is not int for value in row)
                or row[0] not in owners or phases[row[0]] == "DISPATCH_INTENT"
                or not 0 <= row[1] < owners[row[0]] or tuple(row) in acknowledgements):
            raise ValueError("Invalid snapshot acknowledgement")
        acknowledgements.add(tuple(row))
        ack_counts[row[0]] += 1
    if any(phases[plan] in {"MATERIALIZED", "RELEASE_INTENT", "RELEASED"} and ack_counts[plan] != owners[plan] for plan in owners):
        raise ValueError("Snapshot lease is missing owner acknowledgements")
    payloads = set()
    for row in snapshot["payloads"]:
        if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or not row[0]
                or row[0] in payloads or not isinstance(row[1], str)):
            raise ValueError("Invalid snapshot payload identity")
        entry = json.loads(row[1])
        if not isinstance(entry, dict) or entry.get("key") != row[0]:
            raise ValueError("Snapshot payload key differs")
        payloads.add(row[0])
    assigned = set()
    for plan, keys in lease_keys.items():
        if phases[plan] != "RELEASED":
            if not keys <= payloads or assigned & keys:
                raise ValueError("Snapshot lease payload dependencies are missing or duplicated")
            assigned.update(keys)
    return snapshot


def _load_reference_lease_snapshot(checkpoint_dir, digest):
    path = checkpoint_dir / "reference_leases.json"
    limit = 64 * 1024**2
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Reference lease snapshot is missing or exceeds metadata budget")
    # Bound the read itself: a producer or shared filesystem may change the file
    # after stat(), so the size check alone cannot enforce the metadata budget.
    with path.open("rb") as stream:
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("Reference lease snapshot exceeds metadata budget")
    snapshot = validate_reference_lease_snapshot(json.loads(encoded.decode("utf-8")))
    if snapshot["sha256"] != digest:
        raise ValueError("Reference lease snapshot differs from resume metadata")
    return snapshot


def save_standalone_resume(
    checkpoint_path: str | os.PathLike[str],
    consumed_sequence_nos: Iterable[int] | torch.Tensor,
    *,
    optimizer_step: int,
    input_path: str | os.PathLike[str],
    producer_shuffle: dict[str, Any] | None = None,
    training_plan: dict[str, Any] | None = None,
    storage_format: str = "tensor",
    expected_input_identity: Mapping[str, Any] | None = None,
    lease_snapshot_sha256: str | None = None,
) -> None:
    if not isinstance(storage_format, str) or storage_format not in {"tensor", "gzip"}:
        raise ValueError("Standalone resume storage_format must be tensor or gzip")
    if expected_input_identity is not None:
        expected_input_identity = validate_input_identity(expected_input_identity)
        initial = build_input_fingerprint(input_path, include_content_hash=True)
        if input_content_identity(initial) != expected_input_identity:
            raise ValueError("Standalone input changed since training began; checkpoint cannot be published")
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if lease_snapshot_sha256 is not None:
        _load_reference_lease_snapshot(checkpoint_dir, lease_snapshot_sha256)
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

    suffix = "i64.gz" if storage_format == "gzip" else "pt"
    filename = f"consumed_sequence_nos.{uuid.uuid4().hex}.{suffix}"
    data_path = checkpoint_dir / filename
    temporary = data_path.with_suffix(data_path.suffix + ".incomplete")
    try:
        if storage_format == "gzip":
            decoded_sha = _write_compressed_sequences(values, temporary)
        else:
            torch.save(values, temporary)
        encoded_sha = _file_sha256(temporary)
        os.replace(temporary, data_path)
    finally:
        temporary.unlink(missing_ok=True)

    publish_attempted = False
    try:
        fingerprint = build_input_fingerprint(input_path, include_content_hash=True)
        if expected_input_identity is not None and input_content_identity(fingerprint) != expected_input_identity:
            raise ValueError("Standalone input changed since training began; checkpoint cannot be published")
        metadata = {
            "schema_version": _IMMUTABLE_SCHEMA_VERSION,
            "storage_format": storage_format,
            "optimizer_step": int(optimizer_step),
            "consumed_count": int(values.numel()),
            "consumed_sequence_file": filename,
            "consumed_sequence_sha256": encoded_sha,
            "input_fingerprint": fingerprint,
        }
        if expected_input_identity is not None:
            metadata["input_content_identity"] = expected_input_identity
        if lease_snapshot_sha256 is not None:
            metadata["reference_lease_snapshot_sha256"] = lease_snapshot_sha256
        if storage_format == "gzip":
            metadata.update({
                "consumed_sequence_codec": _COMPRESSED_CODEC,
                "consumed_sequence_decoded_bytes": int(values.numel()) * 8,
                "consumed_sequence_decoded_sha256": decoded_sha,
            })
        if training_plan is not None:
            metadata["training_plan"] = dict(training_plan)
        if producer_shuffle is not None:
            # Consumed sequence numbers are position-based, so their meaning
            # depends on the producer shuffle settings that produced them. Saved
            # here so a resumed Producer can refuse a mismatched configuration
            # instead of silently reordering the stream.
            metadata["producer_shuffle"] = {
                "shuffle": bool(producer_shuffle.get("shuffle", False)),
                "shuffle_seed": int(producer_shuffle.get("shuffle_seed", 42)),
            }
            if producer_shuffle.get("deterministic_publication", False):
                metadata["producer_shuffle"]["deterministic_publication"] = True
            if producer_shuffle.get("source_proportional_field") is not None:
                metadata["producer_shuffle"]["source_proportional_field"] = producer_shuffle["source_proportional_field"]
            if producer_shuffle.get("data_recipe") is not None:
                metadata["producer_shuffle"]["data_recipe"] = producer_shuffle["data_recipe"]
        metadata_path = checkpoint_dir / RESUME_METADATA_NAME
        metadata_temporary = checkpoint_dir / f".{RESUME_METADATA_NAME}.{uuid.uuid4().hex}.incomplete"
        try:
            metadata_temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            publish_attempted = True
            os.replace(metadata_temporary, metadata_path)
        finally:
            metadata_temporary.unlink(missing_ok=True)
    except Exception:
        # A failed replace can have an ambiguous outcome on remote filesystems.
        if not publish_attempted:
            data_path.unlink(missing_ok=True)
        raise



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
    if "reference_lease_snapshot_sha256" in metadata:
        _load_reference_lease_snapshot(checkpoint_dir, metadata["reference_lease_snapshot_sha256"])
    version = metadata.get("schema_version")
    if type(version) is not int or version not in {RESUME_SCHEMA_VERSION, _COMPRESSED_SCHEMA_VERSION, _IMMUTABLE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported standalone resume metadata: {metadata_path}")
    saved_fingerprint = metadata.get("input_fingerprint")
    saved_identity = None
    if "input_content_identity" in metadata:
        saved_identity = validate_input_identity(metadata["input_content_identity"])
        if input_content_identity(saved_fingerprint) != saved_identity:
            raise ValueError("Standalone saved input identity conflicts with its fingerprint")
    tensor_name = metadata.get("consumed_sequence_file")
    compressed = version == _COMPRESSED_SCHEMA_VERSION
    if version == _IMMUTABLE_SCHEMA_VERSION:
        storage_format = metadata.get("storage_format")
        if not isinstance(storage_format, str) or storage_format not in {"tensor", "gzip"}:
            raise ValueError(f"Invalid resume storage format in {metadata_path}")
        compressed = storage_format == "gzip"
        suffix = r"i64\.gz" if compressed else "pt"
        valid_name = isinstance(tensor_name, str) and re.fullmatch(rf"consumed_sequence_nos\.[0-9a-f]{{32}}\.{suffix}", tensor_name)
    else:
        expected_name = CONSUMED_GZIP_NAME if compressed else CONSUMED_SEQUENCE_NAME
        valid_name = tensor_name == expected_name
    if not valid_name:
        raise ValueError(f"Invalid consumed sequence file in {metadata_path}")
    if compressed and metadata.get("consumed_sequence_codec") != _COMPRESSED_CODEC:
        raise ValueError(f"Unsupported compressed resume codec in {metadata_path}")
    tensor_path = checkpoint_dir / tensor_name
    if compressed:
        count = metadata.get("consumed_count")
        decoded_bytes = metadata.get("consumed_sequence_decoded_bytes")
        if (type(count) is not int or count < 0 or count > _MAX_COMPRESSED_ENTRIES
                or type(decoded_bytes) is not int or decoded_bytes != count * 8):
            raise ValueError("Invalid compressed resume progress descriptor")
        if tensor_path.is_symlink() or not tensor_path.is_file():
            raise ValueError("Compressed resume progress must be a regular file")
        if tensor_path.stat().st_size > min(_MAX_ENCODED_BYTES, max(4096, decoded_bytes * 2 + 4096)):
            raise ValueError("Compressed resume progress exceeds the encoded byte limit")
    expected_sha256 = metadata.get("consumed_sequence_sha256")
    if expected_sha256 is not None or version in {_COMPRESSED_SCHEMA_VERSION, _IMMUTABLE_SCHEMA_VERSION}:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Invalid consumed sequence checksum in {metadata_path}")
        actual_sha256 = _file_sha256(tensor_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Consumed sequence checksum mismatch in {checkpoint_dir}")
    if compressed:
        consumed = _read_compressed_sequences(tensor_path, metadata)
    else:
        try:
            values = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except TypeError:
            values = torch.load(tensor_path, map_location="cpu")
        if not isinstance(values, torch.Tensor) or values.dtype != torch.int64:
            raise ValueError(f"Invalid consumed sequence tensor: {tensor_path}")
        values = values.flatten()
        if values.numel() and bool((values < 0).any()):
            raise ValueError(f"Negative consumed sequence number in {tensor_path}")
        consumed = {int(value) for value in values.tolist()}
        if len(consumed) != int(metadata.get("consumed_count", -1)):
            raise ValueError(f"Consumed sequence count mismatch in {checkpoint_dir}")
    if input_path is not None:
        current_fingerprint = build_input_fingerprint(
            input_path,
            include_content_hash=saved_identity is not None or (isinstance(saved_fingerprint, dict) and "sha256" in saved_fingerprint),
        )
        differs = (input_content_identity(current_fingerprint) != saved_identity if saved_identity is not None
                   else saved_fingerprint != current_fingerprint)
        if differs:
            raise ValueError(
                "Standalone resume input file changed since the checkpoint was saved: "
                f"saved={saved_fingerprint!r}, current={current_fingerprint!r}"
            )
    return consumed, metadata


__all__ = [
    "CONSUMED_SEQUENCE_NAME",
    "CONSUMED_GZIP_NAME",
    "RESUME_METADATA_NAME",
    "build_input_fingerprint",
    "load_standalone_resume",
    "save_standalone_resume",
]
