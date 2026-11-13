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
"""Consumer-side fetch and cleanup for vLLM hidden-state transfers.

vLLM has two interchangeable ways to hand extracted hidden states to a
consumer:

* the legacy ``ExampleHiddenStatesConnector`` returns a ``hidden_states_path``
  to a safetensors file on a shared filesystem; and
* the out-of-tree ``hs_connectors.MooncakeHiddenStatesConnector`` returns a
  ``handle`` key into a shared Mooncake store.

This module hides that difference behind a small ``HiddenStatesStore`` object so
the Producer and the offline replay path can load and release either
representation without branching on the connector.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import inspect
import json
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

FILE_BACKEND = "file"
MOONCAKE_BACKEND = "mooncake"
NATIVE_BACKEND = "native"
SUPPORTED_BACKENDS = (FILE_BACKEND, MOONCAKE_BACKEND)

_initialized_pid: int | None = None
_init_lock = threading.Lock()


def _local_ip() -> str:
    """Return the host's primary routable IP.

    Mooncake's Ascend transport parses ``local_hostname`` as an IP address, so a
    bare hostname fails with ``is not a valid ip address``.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def ensure_accelerator_context() -> None:
    """Create the accelerator context Mooncake's Ascend transport needs.

    Some backends cannot allocate their local segment without an active device
    context, which CPU-only/worker processes lack. Runs once per process; a
    ``fork`` in a child re-runs it because the PID changes.
    """
    global _initialized_pid  # noqa: PLW0603 - process-local init guard
    pid = os.getpid()
    if _initialized_pid == pid:
        return

    with _init_lock:
        if _initialized_pid == pid:
            return

        accelerator = None
        with contextlib.suppress(Exception):
            accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or not torch.accelerator.is_available():
            _initialized_pid = pid
            return

        with contextlib.suppress(Exception):
            torch.accelerator.set_device_index(torch.accelerator.current_device_index())
        with contextlib.suppress(Exception):
            torch.zeros(1, device=accelerator)

        _initialized_pid = pid


@dataclass(frozen=True)
class MooncakeStoreSettings:
    """Connection settings for the Mooncake-backed handle store."""

    local_hostname: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 4 * 1024**3
    local_buffer_size: int = 2 * 1024**3
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 8

    def __post_init__(self) -> None:
        if not self.local_hostname:
            object.__setattr__(self, "local_hostname", _local_ip())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> MooncakeStoreSettings:
        values = dict(mapping or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(values) - known
        if unknown:
            logger.warning("Unknown Mooncake store settings ignored: %s", unknown)
        return cls(**{key: values[key] for key in values if key in known})

    def as_store_config(self) -> Any:
        from hs_connectors.mooncake_store import (  # noqa: PLC0415
            MooncakeStoreConfig,
        )

        return MooncakeStoreConfig(
            local_hostname=self.local_hostname,
            metadata_server=self.metadata_server,
            master_server_address=self.master_server_address,
            global_segment_size=int(self.global_segment_size),
            local_buffer_size=int(self.local_buffer_size),
            protocol=self.protocol,
            device_name=self.device_name,
            num_writer_threads=int(self.num_writer_threads),
        )


@dataclass(frozen=True)
class HiddenStatesStoreConfig:
    """Selects the consumer-side backend for hidden-state transfers."""

    backend: str = FILE_BACKEND
    mooncake: MooncakeStoreSettings = field(default_factory=MooncakeStoreSettings)
    file_root: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported hidden-state store backend {self.backend!r}; "
                f"expected one of {SUPPORTED_BACKENDS}"
            )
        if self.file_root is not None and (self.backend != FILE_BACKEND or not isinstance(self.file_root, str)
                                           or not self.file_root or not Path(self.file_root).is_absolute()):
            raise ValueError("Shared file root requires an absolute path and file backend")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> HiddenStatesStoreConfig:
        values = dict(mapping or {})
        backend = str(values.get("backend", FILE_BACKEND) or FILE_BACKEND).lower()
        return cls(
            backend=backend,
            file_root=values.get("file_root"),
            mooncake=MooncakeStoreSettings.from_mapping(values.get("mooncake")),
        )


class HiddenStatesStore:
    """Loads and releases hidden-state payloads from one backend."""

    backend: str = FILE_BACKEND

    def load(
        self, reference: str | Mapping[str, Any], *, timeout: float = 120.0
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return ``(payload, byte_size)`` for a response reference.

        The Mooncake backend takes the full transfer manifest (a mapping);
        the file backend takes a path string.
        """
        raise NotImplementedError

    def release(self, reference: str) -> None:
        """Drop a payload once it has been consumed.

        Implementations may raise; callers for which cleanup is strictly
        best-effort must guard the call themselves.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend resources held by the store."""


def file_reference_tensor_bytes(reference: str) -> int:
    """Inspect a stable safetensors header without loading tensor payloads."""
    from safetensors import safe_open

    path = Path(reference)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors reference")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > 16 * 1024 * 1024 or header_bytes + 8 > before.st_size:
            raise ValueError("Invalid safetensors reference header size")
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            if not tensors.keys():
                raise ValueError("Empty safetensors reference")
        after, current = os.fstat(stream.fileno()), path.stat()
        signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if signature(before) != signature(after) or signature(before) != signature(current):
            raise ValueError("Safetensors reference changed during byte accounting")
        payload_bytes = before.st_size - header_bytes - 8
        if payload_bytes <= 0:
            raise ValueError("Empty safetensors reference payload")
        return payload_bytes


class FileHiddenStatesStore(HiddenStatesStore):
    """Legacy safetensors file backend (``hidden_states_path``)."""

    backend = FILE_BACKEND

    def __init__(self, root: str | None = None):
        self.root = None if root is None else Path(root)
        if self.root is not None and (not self.root.is_absolute() or self.root == Path(self.root.anchor)
                                      or self.root.is_symlink() or not self.root.is_dir()):
            raise ValueError("Shared file root must be an existing absolute non-symlink directory")
        if self.root is not None:
            info = self.root.stat()
            self._root_identity = (self.root.resolve(), info.st_dev, info.st_ino)

    def validate_reference(self, reference: str) -> Path:
        path = Path(reference)
        if self.root is not None:
            root = self.root.resolve()
            info = self.root.stat()
            if self.root.is_symlink() or (root, info.st_dev, info.st_ino) != self._root_identity:
                raise ValueError("Configured shared file root changed")
            if (not path.is_absolute() or path.is_symlink() or path.resolve() == root
                    or not path.resolve().is_relative_to(root)):
                raise ValueError("Hidden reference is outside the configured shared file root")
        return path

    def scan_bad_auxiliary_rows(self, reference, *, all_positions, selected_positions, prompt_ids,
                                auxiliary_layers, required_layers, feature_dtype, clip_value,
                                chunk_rows=512, timeout=30.0):
        """Check rows with bounded hidden-state reads and baseline BF16 conversion."""
        from safetensors import safe_open
        from verl_speco.trainer.target_feature_replay import HiddenStateAlignmentError

        if (not isinstance(all_positions, (list, tuple)) or not all_positions
                or not isinstance(selected_positions, (list, tuple)) or not selected_positions
                or any(type(position) is not int or position < 0 for position in all_positions)
                or any(type(position) is not int or position < 0 for position in selected_positions)
                or type(auxiliary_layers) is not int or auxiliary_layers < 1
                or type(required_layers) is not int or required_layers < auxiliary_layers
                or type(chunk_rows) is not int or not 1 <= chunk_rows <= 512
                or not isinstance(feature_dtype, torch.dtype)
                or (clip_value is not None and (isinstance(clip_value, bool)
                    or not isinstance(clip_value, (int, float)) or not math.isfinite(clip_value) or clip_value <= 0))
                or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid hidden-row scan configuration")
        deadline = time.monotonic() + timeout
        path = self.validate_reference(reference)
        _wait_for_lock(Path(f"{path}.lock"), timeout=timeout)
        with safe_open(str(path), framework="pt", device="cpu") as data:
            if "token_ids" not in data.keys() or "hidden_states" not in data.keys():
                raise ValueError("vLLM hidden-states payload must contain token_ids and hidden_states")
            if data.get_tensor("token_ids").long().tolist() != list(prompt_ids):
                raise HiddenStateAlignmentError("vLLM hidden-states token_ids do not match replay input")
            hidden = data.get_slice("hidden_states")
            shape = hidden.get_shape()
            if len(shape) != 3 or shape[1] < required_layers or shape[2] <= 0:
                raise HiddenStateAlignmentError("vLLM hidden-state shape does not satisfy the feature contract")
            offset = max(len(prompt_ids) - shape[0], 0)
            if min(all_positions) < offset or max(all_positions) >= offset + shape[0]:
                raise HiddenStateAlignmentError("vLLM hidden-state rows do not cover the complete feature window")
            if min(selected_positions) < offset or max(selected_positions) >= offset + shape[0]:
                raise HiddenStateAlignmentError("Selected hidden-state rows are outside the original payload")
            bad = []
            for start in range(0, len(selected_positions), chunk_rows):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Reference batch read deadline exceeded")
                rows = [position - offset for position in selected_positions[start:start + chunk_rows]]
                first, last = min(rows), max(rows) + 1
                if last - first <= 2 * len(rows):
                    block = hidden[first:last][:, :auxiliary_layers, :]
                    indices = torch.tensor([row - first for row in rows], dtype=torch.long)
                    values = block.index_select(0, indices)
                else:
                    runs = []
                    for row in rows:
                        if runs and row == runs[-1][1]:
                            runs[-1][1] += 1
                        else:
                            runs.append([row, row + 1])
                    chunks = [hidden[first:last][:, :auxiliary_layers, :] for first, last in runs]
                    values = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
                if feature_dtype != torch.bfloat16:
                    values = values.to(dtype=feature_dtype)
                values = values.to(dtype=torch.bfloat16)
                invalid = ~torch.isfinite(values).flatten(start_dim=1).all(dim=1)
                if clip_value is not None:
                    sanitized = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    invalid |= sanitized.float().abs().flatten(start_dim=1).amax(dim=1) > clip_value
                bad.extend(invalid.tolist())
            return bad

    def hidden_shape(self, reference: str, *, timeout: float = 30.0) -> tuple[int, ...]:
        """Inspect tensor dimensions without materializing its payload."""
        from safetensors import safe_open

        path = self.validate_reference(reference)
        _wait_for_lock(Path(f"{path}.lock"), timeout=timeout)
        with safe_open(str(path), framework="pt", device="cpu") as data:
            if "hidden_states" not in data.keys():
                raise ValueError("vLLM hidden-states payload must contain hidden_states")
            shape = tuple(data.get_slice("hidden_states").get_shape())
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError("vLLM hidden_states must have non-empty [seq, layers, hidden] shape")
        return shape

    def load(
        self, reference: str | Mapping[str, Any], *, timeout: float = 30.0
    ) -> tuple[dict[str, torch.Tensor], int]:
        if not isinstance(reference, str):
            raise TypeError("The file hidden-state store loads path strings")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("vLLM hidden-state replay requires safetensors") from exc
        path = self.validate_reference(reference)
        _wait_for_lock(Path(f"{path}.lock"), timeout=timeout)
        if not path.is_file():
            raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
        payload = dict(load_file(str(path), device="cpu"))
        return payload, int(path.stat().st_size)

    def load_window(
        self, reference: str, *, all_positions: list[int], selected_positions: list[int],
        prompt_length: int, timeout: float = 30.0, compact: bool = False,
    ) -> tuple[dict[str, torch.Tensor], int | list[int]]:
        """Load only selected hidden rows after checking full original coverage."""
        from safetensors import safe_open
        from verl_speco.trainer.target_feature_replay import HiddenStateAlignmentError

        path = self.validate_reference(reference)
        _wait_for_lock(Path(f"{path}.lock"), timeout=timeout)
        if not path.is_file():
            raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as data:
            if "token_ids" not in data.keys() or "hidden_states" not in data.keys():
                raise ValueError("vLLM hidden-states payload must contain token_ids and hidden_states")
            hidden = data.get_slice("hidden_states")
            shape = hidden.get_shape()
            if len(shape) != 3:
                raise ValueError(f"vLLM hidden_states must have shape [seq, layers, hidden], got {tuple(shape)}")
            original_offset = max(prompt_length - shape[0], 0)
            if any(position < original_offset or position >= original_offset + shape[0]
                   for position in all_positions):
                raise HiddenStateAlignmentError("vLLM hidden-state rows do not cover the complete feature window")
            if not selected_positions:
                raise ValueError("Reference window cannot be empty")
            if any(position < original_offset or position >= original_offset + shape[0]
                   for position in selected_positions):
                raise HiddenStateAlignmentError("Selected hidden-state rows are outside the original payload")
            first = min(selected_positions) - original_offset
            last = max(selected_positions) - original_offset + 1
            if compact and last - first > 2 * len(selected_positions):
                runs = []
                for position in selected_positions:
                    row = position - original_offset
                    if runs and row == runs[-1][1]:
                        runs[-1][1] += 1
                    else:
                        runs.append([row, row + 1])
                # Bound metadata and read operations for extremely fragmented input.
                if len(runs) <= 1024:
                    chunks = [hidden[start:end] for start, end in runs]
                    rows = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
                    return {"token_ids": data.get_tensor("token_ids"), "hidden_states": rows}, list(selected_positions)
            return {"token_ids": data.get_tensor("token_ids"), "hidden_states": hidden[first:last]}, original_offset + first

    def release(self, reference: str) -> None:
        path = self.validate_reference(reference)
        path.unlink(missing_ok=True)
        Path(f"{path}.lock").unlink(missing_ok=True)


class MooncakeHiddenStatesStore(HiddenStatesStore):
    """Handle-based Mooncake store backend (``handle``).

    The load reference is the full transfer manifest vLLM returns in
    ``kv_transfer_params`` (handle plus packed tensor layout); the release
    reference is the bare handle string.
    """

    backend = MOONCAKE_BACKEND

    def __init__(self, settings: MooncakeStoreSettings):
        self._settings = settings
        self._store: Any | None = None
        self._lock = threading.Lock()

    def _ensure_store(self) -> Any:
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            try:
                from hs_connectors.mooncake_store import (  # noqa: PLC0415
                    MooncakeHiddenStatesStore as _Store,
                )
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise RuntimeError(
                    "The Mooncake hidden-state backend requires the hs_connectors "
                    "package (pip install hs-connectors)"
                ) from exc
            ensure_accelerator_context()
            store = _Store(self._settings.as_store_config()).setup()
            self._store = store
            return store

    def load(
        self, reference: str | Mapping[str, Any], *, timeout: float = 120.0
    ) -> tuple[dict[str, torch.Tensor], int]:
        if not isinstance(reference, Mapping):
            raise TypeError(
                "The Mooncake hidden-state store requires the full transfer "
                "manifest from kv_transfer_params (a mapping); the bare handle "
                "alone lacks the tensor layout get_sample decodes payloads with"
            )
        store = self._ensure_store()
        payload = dict(store.get_sample(dict(reference), timeout=timeout))
        byte_size = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in payload.values()
            if isinstance(tensor, torch.Tensor)
        )
        return payload, byte_size

    def release(self, reference: str) -> None:
        store = self._ensure_store()
        # Deletion errors deliberately propagate so callers with a retry policy
        # (the Producer cleanup loop) can retry transient store failures. Callers
        # for which cleanup is strictly best-effort wrap this in their own guard.
        store.delete_sample(reference)

    def close(self) -> None:
        # hs_connectors exposes no explicit teardown; dropping the reference
        # releases the client and unmounts its segment. The producer drains
        # pending samples before this runs, so no consumer is left reading.
        self._store = None


class NativeHiddenStatesStore(HiddenStatesStore):
    """Adapt an explicitly owned native reader and source-release callback.

    The callback must enforce the scheduler's reader acknowledgements before
    releasing the source. A stuck native operation needs a fenced process;
    the elapsed-time check alone cannot cancel a native call.
    """

    backend = NATIVE_BACKEND

    def metrics(self) -> dict[str, float]:
        metrics = getattr(self._transport, "metrics", None)
        return metrics() if callable(metrics) else {}

    def __init__(self, transport: Any, release_callback: Any):
        if not callable(getattr(transport, "read", None)) or not callable(release_callback):
            raise ValueError("Native store requires a reader and an explicit source-release callback")
        self._transport = transport
        self._release_callback = release_callback

    @staticmethod
    def _descriptor(reference: str) -> dict[str, Any]:
        if not isinstance(reference, str) or not reference:
            raise ValueError("Invalid native hidden reference")
        try:
            if len(reference.encode("utf-8")) > 64 * 1024:
                raise ValueError("Native hidden reference exceeds its byte limit")
            value = json.loads(reference)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Malformed native hidden reference") from exc
        if not isinstance(value, dict):
            raise ValueError("Invalid native hidden reference descriptor")
        return value

    def load(self, reference: str, *, timeout: float = 120.0) -> tuple[dict[str, torch.Tensor], int]:
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Native reference timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        payload = self._transport.read(self._descriptor(reference), output_device="cpu")
        if time.monotonic() >= deadline:
            raise TimeoutError("Native reference read exceeded its deadline")
        if (not isinstance(payload, dict) or not payload
                or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in payload.values())):
            raise ValueError("Native reference must produce CPU tensor fields")
        return payload, sum(value.numel() * value.element_size() for value in payload.values())

    def release(self, reference: str) -> None:
        result = self._release_callback(self._descriptor(reference))
        if result is not None:
            if inspect.iscoroutine(result):
                result.close()
            raise RuntimeError("Native source release callback must complete synchronously and return None")

    def close(self) -> None:
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()


def build_managed_native_reader(config: Mapping[str, Any], *, run_id: str, rank: int,
                                world_size: int, device: Any, engine_factory: Any = None) -> NativeHiddenStatesStore:
    """Build a rank-local reader whose source lifetime is owned by the scheduler."""
    settings = dict(config)
    expected = {"endpoints", "metadata_server", "generation", "max_bytes", "chunk_bytes"}
    if set(settings) != expected:
        raise ValueError("Native reader settings require endpoints, metadata_server, generation, max_bytes and chunk_bytes")
    endpoints = settings["endpoints"]
    if (type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size
            or not isinstance(endpoints, (list, tuple)) or len(endpoints) != world_size
            or any(not isinstance(value, str) or not value for value in endpoints)
            or len(set(endpoints)) != len(endpoints)):
        raise ValueError("Native reader requires a distinct endpoint for every rank")
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    transport = NativeFeatureTransport(run_id=run_id, generation=settings["generation"], endpoint=endpoints[rank],
                                       metadata_server=settings["metadata_server"], device=device,
                                       max_bytes=settings["max_bytes"], max_slots=1,
                                       chunk_bytes=settings["chunk_bytes"], engine_factory=engine_factory)
    # Only the producer can free the source after its dispatch-wide ACKs. Local
    # cleanup must not independently release a buffer another rank may read.
    return NativeHiddenStatesStore(transport, lambda _descriptor: None)


def build_hidden_states_store(
    config: HiddenStatesStoreConfig | Mapping[str, Any] | None = None,
) -> HiddenStatesStore:
    """Build the consumer store for ``config`` (defaults to the file backend)."""
    if config is None:
        resolved = HiddenStatesStoreConfig()
    elif isinstance(config, HiddenStatesStoreConfig):
        resolved = config
    else:
        resolved = HiddenStatesStoreConfig.from_mapping(config)
    if resolved.backend == MOONCAKE_BACKEND:
        return MooncakeHiddenStatesStore(resolved.mooncake)
    return FileHiddenStatesStore(resolved.file_root)


def _wait_for_lock(lock_path: Path, timeout: float = 30.0) -> None:
    if not lock_path.exists():
        return
    try:
        fcntl: Any = importlib.import_module("fcntl")
    except ImportError:
        # vLLM's file connector is Linux-only. Keep the old existence-based
        # fallback for dependency-light tests on other platforms.
        deadline = time.monotonic() + timeout
        while lock_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for vLLM hidden-state lock: {lock_path}"
                )
            time.sleep(0.01)
        return

    deadline = time.monotonic() + timeout
    try:
        lock_file = lock_path.open("rb")
    except FileNotFoundError:
        # The writer removed the lock between the exists() check and open().
        return
    with lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for vLLM hidden-state lock: {lock_path}"
                    ) from exc
                time.sleep(0.01)


__all__ = [
    "FILE_BACKEND",
    "MOONCAKE_BACKEND",
    "NATIVE_BACKEND",
    "SUPPORTED_BACKENDS",
    "FileHiddenStatesStore",
    "HiddenStatesStore",
    "HiddenStatesStoreConfig",
    "MooncakeHiddenStatesStore",
    "MooncakeStoreSettings",
    "NativeHiddenStatesStore",
    "build_managed_native_reader",
    "build_hidden_states_store",
]
