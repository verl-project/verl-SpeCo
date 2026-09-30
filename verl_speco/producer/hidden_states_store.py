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
import logging
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

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported hidden-state store backend {self.backend!r}; "
                f"expected one of {SUPPORTED_BACKENDS}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> HiddenStatesStoreConfig:
        values = dict(mapping or {})
        backend = str(values.get("backend", FILE_BACKEND) or FILE_BACKEND).lower()
        return cls(
            backend=backend,
            mooncake=MooncakeStoreSettings.from_mapping(values.get("mooncake")),
        )


class HiddenStatesStore:
    """Loads and releases hidden-state payloads from one backend."""

    backend: str = FILE_BACKEND

    def load(self, reference: str) -> tuple[dict[str, torch.Tensor], int]:
        """Return ``(payload, byte_size)`` for a response reference."""
        raise NotImplementedError

    def release(self, reference: str) -> None:
        """Drop a payload once it has been consumed (best effort)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend resources held by the store."""


class FileHiddenStatesStore(HiddenStatesStore):
    """Legacy safetensors file backend (``hidden_states_path``)."""

    backend = FILE_BACKEND

    def load(self, reference: str) -> tuple[dict[str, torch.Tensor], int]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("vLLM hidden-state replay requires safetensors") from exc
        path = Path(reference)
        _wait_for_lock(Path(f"{path}.lock"))
        if not path.is_file():
            raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
        payload = dict(load_file(str(path), device="cpu"))
        return payload, int(path.stat().st_size)

    def release(self, reference: str) -> None:
        path = Path(reference)
        path.unlink(missing_ok=True)
        Path(f"{path}.lock").unlink(missing_ok=True)


class MooncakeHiddenStatesStore(HiddenStatesStore):
    """Handle-based Mooncake store backend (``handle``)."""

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

    def load(self, reference: str) -> tuple[dict[str, torch.Tensor], int]:
        store = self._ensure_store()
        payload = dict(store.get_sample(reference))
        byte_size = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in payload.values()
            if isinstance(tensor, torch.Tensor)
        )
        return payload, byte_size

    def release(self, reference: str) -> None:
        if self._store is None:
            return
        try:
            self._store.delete_sample(reference)
        except Exception:  # noqa: BLE001 - cleanup must not break the producer
            logger.warning(
                "Failed to delete Mooncake hidden-state handle %s", reference
            )

    def close(self) -> None:
        # hs_connectors exposes no explicit teardown; dropping the reference
        # releases the client and unmounts its segment. The producer drains
        # pending samples before this runs, so no consumer is left reading.
        self._store = None


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
    return FileHiddenStatesStore()


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
    "SUPPORTED_BACKENDS",
    "FileHiddenStatesStore",
    "HiddenStatesStore",
    "HiddenStatesStoreConfig",
    "MooncakeHiddenStatesStore",
    "MooncakeStoreSettings",
    "build_hidden_states_store",
]
