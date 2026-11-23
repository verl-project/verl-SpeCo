# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Background export of computed features, with optional capacity limits.

This component never calls a model. The caller owns device synchronization and
supplies a completion callback plus the objects whose lifetime it protects.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import logging
import io
import zipfile
import os
import threading
import time
import uuid
import hashlib
import json

logger = logging.getLogger(__name__)


def hidden_states_are_finite(hidden):
    """Scan CPU features without large temporary masks or global thread changes."""
    import numpy as np
    import torch

    if hidden.device.type != "cpu" or not hidden.is_contiguous():
        return bool(torch.isfinite(hidden).all())
    value = hidden.detach().reshape(-1)
    if value.dtype == torch.bfloat16:
        # BF16 exponent bits are all ones for both NaN and either infinity.
        bits = value.view(torch.int16).numpy()
        return all(
            not np.any((bits[i : i + 65536] & 0x7F80) == 0x7F80)
            for i in range(0, bits.size, 65536)
        )
    if value.dtype in (torch.float16, torch.float32, torch.float64):
        array = value.numpy()
        return all(
            np.isfinite(array[i : i + 65536]).all() for i in range(0, array.size, 65536)
        )
    return bool(torch.isfinite(hidden).all())


def feature_digest(payload):
    """Bitwise digest of feature identity and tensors, including BF16 bits."""
    import torch

    digest = hashlib.sha256()
    for key in ("sample_id", "layer_ids", "layout", "last_hidden_norm"):
        digest.update(json.dumps([key, payload.get(key)], sort_keys=True).encode())
    for key in ("hidden_states", "token_ids", "positions"):
        tensor = payload[key].detach().contiguous().cpu()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def verify_feature_digest(payload):
    if (
        "transport_sha256" in payload
        and feature_digest(payload) != payload["transport_sha256"]
    ):
        raise RuntimeError("Captured feature transport checksum mismatch")
    return payload


class FeatureExportQueue:
    def __init__(
        self, *, max_bytes=256 << 20, max_pending=16, workers=2, pack_chunks=False
    ):
        self.pack_chunks = pack_chunks
        self._pack_lock = threading.Lock()
        self._pack_index = {}
        self._window_markers = {}
        self._window_marker_lock = threading.Lock()
        self.max_bytes = max_bytes
        self.max_pending = max_pending
        self._bytes = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="speco-feature-export"
        )

    def reserve(self, size):
        with self._lock:
            if (
                size < 0
                or (self.max_bytes is not None and self._bytes + size > self.max_bytes)
                or (self.max_pending is not None and self._pending >= self.max_pending)
            ):
                return False
            self._bytes += size
            self._pending += 1
            return True

    def release(self, size):
        with self._lock:
            self._bytes -= size
            self._pending -= 1

    def submit_reserved(
        self, directory, payload, *, size, wait=lambda: None, owners=()
    ):
        """Transfer a reservation and all buffer ownership to the worker.

        Caller releases the reservation only if submission raises. The pending
        marker exists before control returns to the engine/frontend.
        """
        marker = Path(directory) / f"{uuid.uuid4().hex}.pending"
        marker.touch()
        try:
            self._executor.submit(
                self._write, marker, payload, size, wait, owners, time.perf_counter()
            )
        except BaseException:
            marker.unlink(missing_ok=True)
            raise

    def _retain_window_marker(self, directory):
        with self._window_marker_lock:
            entry = self._window_markers.get(directory)
            if entry is None:
                marker = directory / f"{uuid.uuid4().hex}.pending"
                marker.touch()
                entry = (marker, 0)
            marker, count = entry
            self._window_markers[directory] = marker, count + 1
            return marker

    def _release_window_marker(self, marker):
        directory = marker.parent
        with self._window_marker_lock:
            _, count = self._window_markers[directory]
            if count == 1:
                marker.unlink(missing_ok=True)
                del self._window_markers[directory]
            else:
                self._window_markers[directory] = marker, count - 1

    @contextmanager
    def window_batch(self):
        """Submit small groups per runner call, retaining per-window cancellation markers."""
        jobs = []

        def dispatch():
            if not jobs:
                return
            group = tuple(jobs)
            jobs.clear()
            try:
                self._executor.submit(self._write_group, group)
            except RuntimeError:
                # Jobs already own reservations and asynchronous copies.
                self._write_group(group)

        def append(directory, payload, *, size, wait, owners):
            started = time.perf_counter()
            marker = self._retain_window_marker(Path(directory))
            payload["capture_timing"]["pending_marker_host_s"] = (
                time.perf_counter() - started
            )
            jobs.append((marker, payload, size, wait, owners, time.perf_counter()))
            if len(jobs) == 4:
                dispatch()

        try:
            yield append
        finally:
            dispatch()

    def _write_group(self, jobs):
        for job in jobs:
            job[1]["capture_timing"]["group_size"] = len(jobs)
            try:
                self._write(*job, shared_marker=True)
            except Exception:
                # Cleanup failures in one window must not strand sibling reservations.
                logger.exception("Window export cleanup failed for %s", job[0])

    def _write(
        self, marker, payload, size, wait, owners, submitted, *, shared_marker=False
    ):
        temporary = marker.with_suffix(".tmp")
        recycle = payload.pop("_capture_recycle", None)
        window_plan = payload.pop("_window_plan", None)
        copy_complete = False
        try:
            import torch

            started = time.perf_counter()
            wait()
            copy_complete = True
            ready = time.perf_counter()
            # Acceptance filtering runs in wait(). A verification can leave no
            # committed rows; avoid serializing an empty tensor archive. The
            # finally block still removes the marker and releases ownership.
            positions = payload.get("positions")
            if positions is not None and positions.numel() == 0:
                return
            if payload.get("transport_snapshot_verified"):
                payload["transport_sha256"] = feature_digest(payload)
            if "sample_id" in payload:
                payload.setdefault("capture_timing", {}).update(
                    queue_s=started - submitted, wait_filter_s=ready - started
                )
            if not (marker.parent / "live").exists():
                return
            if window_plan is not None:
                from verl_speco.producer.rollout_window_store import write_window

                write_window(marker.parent, payload, window_plan)
                return
            if self.pack_chunks:
                self._write_pack(marker, payload)
            else:
                torch.save(payload, temporary)
            if (marker.parent / "live").exists():
                if not self.pack_chunks:
                    os.replace(temporary, marker.with_suffix(".pt"))
                from verl_speco.integration.vllm_rollout_capture import capture_trace

                capture_trace(
                    "Feature export complete sample=%s reserved_bytes=%s queue_seconds=%.6f "
                    "wait_and_filter_seconds=%.6f write_seconds=%.6f",
                    payload.get("sample_id"),
                    size,
                    started - submitted,
                    ready - started,
                    time.perf_counter() - ready,
                )
        except Exception as exc:
            logger.exception(
                "Feature export failed sample=%s", payload.get("sample_id")
            )
            try:
                if (marker.parent / "live").exists():
                    error_path = marker.parent / "error"
                    if window_plan is None:
                        error_path.touch()
                    else:
                        record = {
                            "step": window_plan.get("global_step"),
                            "sample_id": window_plan.get("sample_id"),
                            "type": type(exc).__name__,
                            "reason": str(exc),
                        }
                        # Install the complete first failure without overwriting a sibling's.
                        detail = marker.parent / f"error-{uuid.uuid4().hex}.tmp"
                        try:
                            detail.write_text(json.dumps(record), encoding="utf-8")
                            os.link(detail, error_path)
                        except FileExistsError:
                            pass
                        except OSError:
                            error_path.touch(exist_ok=True)
                            raise
                        finally:
                            detail.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not persist feature export error")
        finally:
            # Keep device snapshots and pinned destinations alive through wait
            # and serialization, including cancellation and error paths.
            try:
                del owners
                if shared_marker:
                    self._release_window_marker(marker)
                else:
                    temporary.unlink(missing_ok=True)
                    marker.unlink(missing_ok=True)
                if not (marker.parent / "live").exists():
                    from verl_speco.integration.vllm_rollout_capture import (
                        reclaim_cancelled_export,
                    )

                    reclaim_cancelled_export(marker.parent, plan=window_plan)
            finally:
                try:
                    if recycle is not None and copy_complete:
                        recycle()
                finally:
                    self.release(size)

    def _write_pack(self, marker, payload):
        import torch

        # Serialization stays in the background worker; no new model forward.
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        # ZIP_STORED avoids compression work. The pending marker remains until
        # the archive is closed, so consumers never observe partial records.
        with self._pack_lock:
            if not (marker.parent / "live").exists():
                return
            cached = self._pack_index.get(marker.parent)
            if cached is not None:
                path, count, total = cached
                if path.exists() and count < 16 and total + len(data) <= (1 << 20):
                    with zipfile.ZipFile(
                        path, "a", compression=zipfile.ZIP_STORED
                    ) as archive:
                        archive.writestr(marker.stem + ".pt", data)
                    self._pack_index[marker.parent] = (
                        path,
                        count + 1,
                        total + len(data),
                    )
                    return
            # A known full pack can rotate directly. Re-scanning the growing
            # directory and reopening its last ZIP is only needed on cache miss.
            if cached is not None and cached[0].exists():
                path = marker.parent / f"{int(cached[0].stem) + 1:08d}.ptpack"
                with zipfile.ZipFile(
                    path, "a", compression=zipfile.ZIP_STORED
                ) as archive:
                    archive.writestr(marker.stem + ".pt", data)
                self._pack_index[marker.parent] = (path, 1, len(data))
                return
            packs = sorted(marker.parent.glob("*.ptpack"))
            path = packs[-1] if packs else None
            if path is not None:
                with zipfile.ZipFile(path) as archive:
                    full = len(archive.infolist()) >= 16 or sum(
                        entry.file_size for entry in archive.infolist()
                    ) + len(data) > (1 << 20)
                if full:
                    path = None
            if path is None:
                path = marker.parent / f"{len(packs):08d}.ptpack"
            with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(marker.stem + ".pt", data)
                entries = archive.infolist()
                self._pack_index[marker.parent] = (
                    path,
                    len(entries),
                    sum(e.file_size for e in entries),
                )
            # Cache metadata only; no tensor/serialized payload is retained.
            if len(self._pack_index) > 256:
                self._pack_index.pop(next(iter(self._pack_index)))

    def close(self):
        self._executor.shutdown(wait=True)


def wait_for_exports(directory, *, timeout=120.0):
    deadline = time.monotonic() + timeout
    while any(Path(directory).glob("*.pending")):
        if (Path(directory) / "error").exists() or (
            Path(directory) / "skipped"
        ).exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for captured feature export")
        time.sleep(0.005)


def rollout_feature_fields(prompt, response, hidden, positions):
    """Shared SGLang/vLLM sample fields; no cloning or model execution.

    Inputs are unbatched tensors already owned by the caller. CPU conversion is
    a no-op for exporter results and preserves the existing SGLang contract.
    """
    import torch

    if positions is not None:
        if positions.ndim != 1 or positions.numel() != hidden.shape[0]:
            raise ValueError("Hidden positions must match feature rows")
    return {
        "input_ids": torch.cat([prompt, response], dim=0).unsqueeze(0),
        "prompts": prompt.unsqueeze(0),
        "responses": response.unsqueeze(0),
        "hidden_states": hidden.unsqueeze(0).cpu(),
        "hidden_positions": positions.unsqueeze(0).cpu()
        if positions is not None
        else None,
    }


def load_export_chunks(directory):
    """Read legacy chunks and opt-in packed exports after pending writes finish."""
    import torch

    for path in sorted(Path(directory).glob("*.pt")):
        yield verify_feature_digest(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    for path in sorted(Path(directory).glob("*.ptpack")):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                with archive.open(entry) as stream:
                    yield verify_feature_digest(
                        torch.load(
                            io.BytesIO(stream.read()),
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
