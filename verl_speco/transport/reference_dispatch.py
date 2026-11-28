# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Encoding and bounded reassembly for Ray feature-assignment commands."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def bounded_metadata_size(value, limit):
    """Count canonical JSON bytes incrementally before issuing a control RPC."""
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    size = 0
    for part in encoder.iterencode(value):
        size += len(part)  # ensure_ascii produces only single-byte characters
        if size > limit:
            raise ValueError(f"Reference control metadata exceeds {limit} bytes")
    return size


def reference_command_frames(command, *, max_bytes, frame_bytes):
    """Serialize large commands into ordered, independently retryable frames."""
    size = bounded_metadata_size(command, max_bytes)
    if size <= frame_bytes:
        return iter((command,))
    encoded = json.dumps(
        command,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) != size:
        raise RuntimeError("Reference command encoding size changed")
    digest = hashlib.sha256(encoded).hexdigest()
    count = (size + frame_bytes - 1) // frame_bytes
    plan_id, receipt_token = command["plan_id"], command["receipt_token"]
    return (
        {
            "kind": "batch_fragment",
            "plan_id": plan_id,
            "receipt_token": receipt_token,
            "index": index,
            "count": count,
            "nbytes": size,
            "sha256": digest,
            "data": encoded[index * frame_bytes : (index + 1) * frame_bytes],
        }
        for index in range(count)
    )


def command_fingerprint(command: dict[str, Any]) -> bytes:
    payload = json.dumps(
        command, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


class ReferenceCommandAssembler:
    """Validate and reassemble bounded, duplicate-safe command fragments."""

    def __init__(self, max_bytes: int, frame_bytes: int):
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 1024**3:
            raise ValueError("Invalid reference command byte limit")
        if type(frame_bytes) is not int or not 1024 <= frame_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid reference command frame limit")
        self.max_bytes, self.frame_bytes = max_bytes, frame_bytes
        self.active = None
        self.recent: dict[
            int, tuple[tuple[int, str, int, int, str], tuple[bytes, ...]]
        ] = {}
        self.recent_hashes = 0

    def add(self, frame):
        fields = {
            "kind",
            "plan_id",
            "receipt_token",
            "index",
            "count",
            "nbytes",
            "sha256",
            "data",
        }
        if (
            not isinstance(frame, dict)
            or set(frame) != fields
            or frame["kind"] != "batch_fragment"
        ):
            raise ValueError("Malformed reference command frame")
        plan, token = frame["plan_id"], frame["receipt_token"]
        index, count, size, digest, data = (
            frame[key] for key in ("index", "count", "nbytes", "sha256", "data")
        )
        if (
            type(plan) is not int
            or plan < 0
            or not isinstance(token, str)
            or not 1 <= len(token) <= 128
            or type(size) is not int
            or not self.frame_bytes < size <= self.max_bytes
            or type(count) is not int
            or count != (size + self.frame_bytes - 1) // self.frame_bytes
            or type(index) is not int
            or not 0 <= index < count
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(data) is not bytes
            or len(data) != min(self.frame_bytes, size - index * self.frame_bytes)
        ):
            raise ValueError("Invalid reference command frame bounds or identity")
        identity = (plan, token, count, size, digest)
        chunk_hash = hashlib.sha256(data).digest()
        prior = self.recent.get(plan)
        if prior is not None:
            if prior[0] != identity or prior[1][index] != chunk_hash:
                raise ValueError("Conflicting completed reference command frame")
            return None
        if self.active is None:
            if index != 0:
                raise ValueError("Reference command starts with an out-of-order frame")
            self.active = {"identity": identity, "data": bytearray(), "hashes": []}
        active = self.active
        if active["identity"] != identity:
            raise ValueError("Reference command frames have conflicting identities")
        next_index = len(active["hashes"])
        if index < next_index:
            if active["hashes"][index] != chunk_hash:
                raise ValueError("Conflicting duplicate reference command frame")
            return None
        if index != next_index:
            raise ValueError("Reference command frame is out of order")
        active["data"].extend(data)
        active["hashes"].append(chunk_hash)
        if len(active["hashes"]) < count:
            return None
        payload = active["data"]
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Reference command frame checksum mismatch")
        try:
            command = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                "Reference command frame payload is not valid JSON"
            ) from exc
        if (
            not isinstance(command, dict)
            or command.get("kind") != "batch"
            or command.get("plan_id") != plan
            or command.get("receipt_token") != token
        ):
            raise ValueError("Reference command frame payload identity differs")
        hashes = tuple(active["hashes"])
        self.active = None
        if len(hashes) <= 8192:
            self.recent[plan] = (identity, hashes)
            self.recent_hashes += len(hashes)
            while len(self.recent) > 64 or self.recent_hashes > 8192:
                oldest = next(iter(self.recent))
                _, old_hashes = self.recent.pop(oldest)
                self.recent_hashes -= len(old_hashes)
        return command
