# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Incremental immutable window publication using the shared hidden-store API."""

from collections import Counter, OrderedDict

from _thread import LockType
import hashlib
import json
import logging
import os
import struct
import threading
import weakref
import time
from pathlib import Path

import numpy as np
import torch
from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
from verl_speco.producer.feature_export import hidden_states_are_finite

_LOCK = threading.Lock()
_WINDOW_LOCKS: weakref.WeakValueDictionary[Path, LockType] = (
    weakref.WeakValueDictionary()
)
# Bounded process-local diagnostic totals, independent of sample lifetime.
_STEP_TOTALS: OrderedDict[tuple, Counter] = OrderedDict()
logger = logging.getLogger(__name__)


def _sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_window(directory, payload, plan):
    """Aggregate fragments in a file-backed window; flush once before publication."""
    directory = Path(directory)
    with _LOCK:
        window_lock = _WINDOW_LOCKS.setdefault(directory, threading.Lock())
    lock_started = time.perf_counter()
    with window_lock:
        write_started = time.perf_counter()
        lock_wait = write_started - lock_started
        if (directory / "skipped").exists():
            return
        # vLLM input IDs can arrive as int32; the window wire format is I64.
        if payload["token_ids"].dtype == torch.int32:
            payload = {**payload, "token_ids": payload["token_ids"].long()}
        if (directory / "window.ready").exists():
            existing, _ = load_window(directory)
            metadata = existing["metadata"]
            positions = payload["positions"]
            prefix = f"Conflicting window replay after immutable publication: sample={plan['sample_id']}"
            expected = {
                **{
                    key: plan.get(key)
                    for key in (
                        "sample_id",
                        "target_version",
                        "request_id",
                        "run_id",
                        "start",
                        "end",
                    )
                },
                "layer_ids": payload["layer_ids"],
                "layout": payload["layout"],
                "norm": payload["last_hidden_norm"],
            }
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise RuntimeError(
                        f"{prefix} field={field} stored={metadata.get(field)!r} incoming={value!r}"
                    )
            if positions.ndim != 1 or positions.dtype != torch.int64:
                raise RuntimeError(
                    f"{prefix} invalid positions shape={tuple(positions.shape)} dtype={positions.dtype}"
                )
            offsets = positions - metadata["start"]
            outside = (offsets < 0) | (offsets >= metadata["rows"])
            if outside.any():
                raise RuntimeError(
                    f"{prefix} out-of-range position={int(positions[outside][0])}"
                )
            for field in ("token_ids", "hidden_states"):
                stored = existing[field][offsets]
                incoming = payload[field]
                if stored.shape != incoming.shape or stored.dtype != incoming.dtype:
                    raise RuntimeError(
                        f"{prefix} field={field} expected={tuple(stored.shape)}/{stored.dtype} incoming={tuple(incoming.shape)}/{incoming.dtype}"
                    )
                if field == "hidden_states":
                    stored = stored.view(torch.int16)
                    incoming = incoming.contiguous().view(torch.int16)
                if not torch.equal(stored, incoming):
                    row = int((stored != incoming).nonzero()[0, 0])
                    raise RuntimeError(
                        f"{prefix} field={field} position={int(positions[row])}"
                    )
            return
        rows = int(plan["end"]) - int(plan["start"])
        hidden = payload["hidden_states"].contiguous()
        if hidden.ndim != 2 or hidden.shape[1] <= 0 or rows <= 0:
            raise RuntimeError("Invalid window hidden shape or range")
        positions = payload["positions"]
        tokens = payload["token_ids"]
        if (
            positions.ndim != 1
            or tokens.ndim != 1
            or positions.numel() != hidden.shape[0]
            or tokens.numel() != hidden.shape[0]
            or positions.dtype != torch.int64
            or tokens.dtype != torch.int64
        ):
            raise RuntimeError(
                f"Invalid window token/position shape or dtype: "
                f"tokens={tuple(tokens.shape)} {tokens.dtype}, "
                f"positions={tuple(positions.shape)} {positions.dtype}, "
                f"hidden={tuple(hidden.shape)}"
            )
        if not hidden_states_are_finite(hidden):
            raise RuntimeError("Window contains non-finite hidden states")
        if bool(
            ((positions < int(plan["start"])) | (positions >= int(plan["end"]))).any()
        ):
            raise RuntimeError("Window position outside reservation")
        width = hidden.shape[-1]
        descriptor = directory / "window.meta"
        metadata = {
            "rows": rows,
            "width": width,
            "dtype": str(hidden.dtype),
            "start": plan["start"],
            "end": plan["end"],
            "sample_id": plan["sample_id"],
            "run_id": plan.get("run_id"),
            "target_version": plan.get("target_version"),
            "request_id": plan.get("request_id"),
            "layer_ids": payload["layer_ids"],
            "layout": payload["layout"],
            "norm": payload["last_hidden_norm"],
            "transport_snapshot_verified": bool(
                payload.get("transport_snapshot_verified", False)
            ),
        }
        if hidden.dtype != torch.bfloat16:
            raise RuntimeError("Window store requires BF16 capture")
        if not descriptor.exists():
            from verl_speco.integration.rollout_reference import _reserve

            budget = plan.get("reference_budget") or _reserve(plan)
            if budget is None:
                (directory / "skipped").touch()
                logger.warning(
                    "[rollout window] event=skip reason=byte_capacity sample=%s",
                    plan["sample_id"],
                )
                return
            metadata["budget"] = budget
            if rows * width * 2 + rows * 9 + 4096 > budget["bytes"]:
                from verl_speco.integration.rollout_reference import _release

                _release({"budget": budget})
                raise RuntimeError("Window exceeds reserved byte budget")
            descriptor.write_text(json.dumps(metadata))
            sizes = [rows * width * 2, rows * 8, rows]
            header = {}
            offset = 0
            for name, dtype, shape, size in zip(
                ("hidden_states", "token_ids", "coverage"),
                ("BF16", "I64", "U8"),
                ([rows, width], [rows], [rows]),
                sizes,
            ):
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * ((-len(encoded)) % 8)
            with (directory / "window.safetensors").open("wb") as stream:
                stream.write(struct.pack("<Q", len(encoded)))
                stream.write(encoded)
                stream.truncate(8 + len(encoded) + offset)
        else:
            existing = json.loads(descriptor.read_text())
            metadata["budget"] = existing.get("budget")
            if existing != metadata:
                raise RuntimeError("Window layer/layout changed during request")
        path = directory / "window.safetensors"
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
        base = 8 + header_size
        mapping = np.memmap(path, dtype=np.uint8, mode="r+", offset=base)
        hidden_bytes = rows * width * 2
        token_end = hidden_bytes + rows * 8
        values = mapping[:hidden_bytes].view(np.uint16).reshape(rows, width)
        token_values = mapping[hidden_bytes:token_end].view(np.int64)
        coverage = mapping[token_end:]
        bits = hidden.view(torch.uint16).numpy()
        indices = positions.numpy() - int(plan["start"])
        token_ids = tokens.numpy()
        # Ordinary decode fragments cover a contiguous range. Avoid indexed
        # copies and sorting on that hot path; sparse/replayed rows remain valid.
        contiguous = len(indices) and (
            len(indices) == 1 or np.all(np.diff(indices) == 1)
        )
        if contiguous:
            target = slice(indices[0], indices[-1] + 1)
        else:
            target = indices
            unique, first, inverse = np.unique(
                indices, return_index=True, return_inverse=True
            )
            if len(unique) != len(indices) and (
                not np.array_equal(token_ids, token_ids[first][inverse])
                or not np.array_equal(bits, bits[first][inverse])
            ):
                raise RuntimeError("Conflicting window replay")
        present = coverage[target].astype(bool)
        if present.any() and (
            not np.array_equal(token_values[indices[present]], token_ids[present])
            or not np.array_equal(values[indices[present]], bits[present])
        ):
            raise RuntimeError("Conflicting window replay")
        values[target] = bits
        token_values[target] = token_ids
        coverage[target] = 1
        complete = bool(np.all(coverage))
        write_elapsed = time.perf_counter() - write_started
        flush_elapsed = 0.0
        if complete:
            started = time.perf_counter()
            mapping.flush()
            flush_elapsed = time.perf_counter() - started
        del coverage, token_values, values, mapping
        step_key = (plan.get("run_id"), plan.get("global_step"))
        with _LOCK:
            totals = _STEP_TOTALS.setdefault(step_key, Counter())
            _STEP_TOTALS.move_to_end(step_key)
            if len(_STEP_TOTALS) > 8:
                _STEP_TOTALS.popitem(last=False)
            timing = payload.get("capture_timing", {})
            totals.update(
                export_tasks_equivalent=1 / timing.get("group_size", 1),
                snapshot_host_s=timing.get("snapshot_host_s", 0.0),
                d2h_submit_host_s=timing.get("d2h_submit_host_s", 0.0),
                snapshot_copy_submit_host_s=timing.get(
                    "snapshot_copy_submit_host_s", 0.0
                ),
                pending_marker_host_s=timing.get("pending_marker_host_s", 0.0),
                gpu_samples=int("d2h_gpu_s" in timing),
                sampled_snapshot_gpu_s=timing.get("snapshot_gpu_s", 0.0),
                sampled_d2h_gpu_s=timing.get("d2h_gpu_s", 0.0),
                fragments=1,
                hidden_bytes=hidden.numel() * hidden.element_size(),
                writer_lock_s=lock_wait,
                writer_work_s=write_elapsed,
                flush_s=flush_elapsed,
                executor_queue_s=payload.get("capture_timing", {}).get("queue_s", 0.0),
            )
            snapshot = dict(totals) if complete else None
        if complete:
            digest_started = time.perf_counter()
            metadata["digests"] = {"window.safetensors": _sha(path)}
            logger.warning(
                "[rollout window] step=%s process_totals=%s window_digest_s=%.6f",
                step_key,
                snapshot,
                time.perf_counter() - digest_started,
            )
            temporary = directory / "window.tmp"
            temporary.write_text(json.dumps(metadata))
            logger.warning(
                "[rollout window] event=ready sample=%s rows=%s bytes=%s",
                plan["sample_id"],
                rows,
                rows * width * 2,
            )
            os.replace(temporary, directory / "window.ready")


def load_window(reference, *, timeout=120):
    """Rollout metadata adapter around the existing producer file store."""
    directory = Path(reference)
    metadata = json.loads((directory / "window.ready").read_text())
    if (
        any(
            type(metadata.get(key)) is not int
            for key in ("rows", "width", "start", "end")
        )
        or metadata["width"] <= 0
        or metadata["start"] < 0
        or metadata.get("dtype") != str(torch.bfloat16)
    ):
        raise RuntimeError("Invalid window tensor contract metadata")
    path = directory / "window.safetensors"
    if metadata["digests"] != {"window.safetensors": _sha(path)}:
        raise RuntimeError("Window checksum mismatch")
    payload, size = FileHiddenStatesStore(wait_for_writer=False).load(
        str(path), timeout=timeout
    )
    rows = metadata["end"] - metadata["start"]
    if (
        rows <= 0
        or metadata["rows"] != rows
        or payload["hidden_states"].shape != (rows, metadata["width"])
        or payload["hidden_states"].dtype != torch.bfloat16
        or payload["token_ids"].shape != (rows,)
        or payload["token_ids"].dtype != torch.int64
        or payload["coverage"].shape != (rows,)
        or payload["coverage"].dtype != torch.uint8
    ):
        raise RuntimeError("Invalid window tensor contract")
    if not bool((payload.pop("coverage") == 1).all()):
        raise RuntimeError("Incomplete window coverage")
    # The training sample owns its memory before source ACK/removal.
    return {
        "hidden_states": payload["hidden_states"].clone(),
        "token_ids": payload["token_ids"].clone(),
        "positions": torch.arange(metadata["start"], metadata["end"]),
        "metadata": metadata,
    }, size
