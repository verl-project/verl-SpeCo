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
import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
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
    wait_for_writer: bool = True

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


class FileHiddenStatesStore(HiddenStatesStore):
    """Legacy safetensors file backend (``hidden_states_path``)."""

    backend = FILE_BACKEND

    def __init__(self, *, wait_for_writer=True):
        self.wait_for_writer = wait_for_writer

    def load(
        self, reference: str | Mapping[str, Any], *, timeout: float = 30.0
    ) -> tuple[dict[str, torch.Tensor], int]:
        if not isinstance(reference, str):
            raise TypeError("The file hidden-state store loads path strings")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("vLLM hidden-state replay requires safetensors") from exc
        path = Path(reference)
        if self.wait_for_writer:
            _wait_for_lock(Path(f"{path}.lock"), timeout=timeout)
        if not path.is_file():
            raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
        payload = dict(load_file(str(path), device="cpu"))
        return payload, int(path.stat().st_size)

    def release(self, reference: str) -> None:
        path = Path(reference)
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


RAW_HIDDEN_FORMAT = "standalone_raw_hidden"
RAW_HIDDEN_VERSION = 1


@dataclass(frozen=True)
class RawHiddenSample:
    payload: dict[str, torch.Tensor]
    sample_id: str
    sequence_no: int
    request: dict[str, Any]
    feature_contract: dict[str, Any]


def prepare_raw_hidden_sample(raw, request, contract, *, validate_outputs=True):
    """Validate alignment without transforming, normalizing or cropping tensors."""
    payload = raw.payload
    tokens, hidden = payload.get("token_ids"), payload.get("hidden_states")
    if not isinstance(tokens, torch.Tensor) or not isinstance(hidden, torch.Tensor):
        raise TypeError(
            "Raw hidden payload requires token_ids and hidden_states tensors"
        )
    if tokens.ndim != 1 or hidden.ndim != 3:
        raise ValueError(
            "Raw hidden payload requires [T] token_ids and [T,K,H] hidden_states"
        )
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in payload.values()
    ):
        raise ValueError("Raw hidden payload must contain CPU tensors only")
    if not hidden.is_floating_point() or (
        validate_outputs and not torch.isfinite(hidden).all()
    ):
        raise ValueError("Raw hidden states must be finite floating-point tensors")
    return _raw_hidden_metadata(
        tokens, hidden.shape, str(hidden.dtype), request, contract, payload
    )


def _raw_hidden_metadata(tokens, shape, dtype, request, contract, payload):
    if tokens.ndim != 1 or len(shape) != 3:
        raise ValueError(
            "Raw hidden payload requires [T] token_ids and [T,K,H] hidden_states"
        )
    if tokens.tolist() != list(request.prompt_token_ids):
        raise ValueError("Raw hidden token_ids do not match the requested prefill")
    rows, layers, width = shape
    offset = tokens.numel() - rows
    positions = request.feature_positions
    if (
        rows <= 0
        or width <= 0
        or offset < 0
        or positions.numel() == 0
        or int(positions.min()) < offset
        or int(positions.max()) >= tokens.numel()
    ):
        raise ValueError("Raw hidden rows do not cover the requested feature window")
    if isinstance(contract, Mapping):
        values = dict(contract)
        if values.get("contract_type") != "raw_capture":
            raise ValueError("Unknown raw hidden capture descriptor")
        aux_ids = values["vllm_aux_hidden_state_layer_ids"]
        final_layer = values["target_num_hidden_layers"]
        required = len(aux_ids) + int(
            values["include_final_hidden"] and final_layer not in aux_ids
        )
        if layers != required:
            raise ValueError(
                f"Captured hidden layer count is {layers}, configured capture plan requires {required}"
            )
        values["raw_hidden_dtype"] = dtype
    else:
        # Previously generated raw datasets carried a bound training contract.
        aux_ids = contract.vllm_aux_hidden_state_layer_ids or contract.target_layer_ids
        include_final = contract.hidden_states_layout.endswith("_plus_last")
        required = len(aux_ids) + int(
            include_final and contract.target_num_hidden_layers not in aux_ids
        )
        if layers < required:
            raise ValueError(
                f"Raw hidden layer count is {layers}, expected at least {required}"
            )
        values = asdict(contract)
        values["dtype"] = str(contract.dtype)
    request_values = {
        name: getattr(request, name).tolist()
        for name in (
            "input_ids",
            "loss_mask",
            "position_ids",
            "feature_positions",
            "draft_position_ids",
        )
    }
    request_values.update(
        vllm_prompt_token_ids=list(request.prompt_token_ids),
        source_metadata=request.source_metadata,
    )
    return RawHiddenSample(
        dict(payload), request.sample_id, request.sequence_no, request_values, values
    )


def bind_raw_hidden_contract(stored, expected):
    """Bind capture metadata to the selected training algorithm at training time."""
    if stored.get("contract_type") != "raw_capture":
        for name, wanted in expected.items():
            if stored.get(name) != wanted:
                raise ValueError(
                    f"Saved hidden feature contract mismatch for {name}: {stored.get(name)!r} != {wanted!r}"
                )
        return dict(stored)
    for name in (
        "target_model_id",
        "target_model_revision",
        "tokenizer_fingerprint",
        "vllm_aux_hidden_state_layer_ids",
    ):
        if stored.get(name) != expected[name]:
            raise ValueError(f"Saved hidden capture metadata mismatch for {name}")
    depth = int(stored["target_num_hidden_layers"])
    has_final = (
        bool(stored.get("include_final_hidden"))
        or depth in stored["vllm_aux_hidden_state_layer_ids"]
    )
    if expected["hidden_states_layout"].endswith("_plus_last") and not has_final:
        raise ValueError(
            "Training losses require final hidden states absent from the saved capture"
        )
    return {
        **expected,
        "target_num_hidden_layers": depth,
        "target_config_fingerprint": stored.get("target_config_fingerprint"),
        "source": stored.get("source", "standalone_data_generate"),
    }


def repair_jsonl_tail(path):
    """Recover an interrupted append; reject corruption in committed lines."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            try:
                if line.strip():
                    json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                stream.truncate(offset)
                logger.warning("Removed interrupted JSONL tail from %s", path)
                return
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                return


def load_raw_hidden_manifest(path):
    """Require an explicit format marker; never interpret normalized features as raw."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("format") != RAW_HIDDEN_FORMAT
        or metadata.get("schema_version") != RAW_HIDDEN_VERSION
    ):
        raise ValueError(
            "Expected standalone_raw_hidden dataset; normalized feature stores are incompatible"
        )
    manifest = path / "manifest.jsonl"
    if not manifest.exists():
        return []
    with manifest.open(encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream if line.strip()]
    for entry in entries:
        if (
            not isinstance(entry.get("sample_id"), str)
            or not entry["sample_id"]
            or not isinstance(entry.get("request"), dict)
            or not isinstance(entry.get("feature_contract"), dict)
            or not entry["feature_contract"]
        ):
            raise ValueError(
                "Raw hidden manifest requires sample_id, request and feature_contract"
            )
        if (
            entry.get("format") != RAW_HIDDEN_FORMAT
            or entry.get("schema_version") != RAW_HIDDEN_VERSION
        ):
            raise ValueError("Raw hidden manifest format/version mismatch")
        relative = Path(entry["path"])
        if relative.name != entry["path"] or relative.suffix != ".safetensors":
            raise ValueError(
                "Raw hidden manifest must reference a local safetensors basename"
            )
        if not (path / relative).is_file():
            raise FileNotFoundError(path / relative)
    return entries


def _replace_hidden_file(source, destination):
    # Windows file scanners can briefly retain a just-closed connector file.
    # Linux permission errors must propagate immediately, as do persistent errors.
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))


class RawHiddenDatasetWriter:
    """One writer: atomic tensor file first, manifest commit second."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            load_raw_hidden_manifest(self.path)
        else:
            if any(self.path.iterdir()):
                raise ValueError(
                    "Refusing to append raw hidden data to an unmarked directory"
                )
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": RAW_HIDDEN_FORMAT,
                        "schema_version": RAW_HIDDEN_VERSION,
                        "last_hidden_state_norm": "raw_target_residual",
                        "created_by": "verl_speco.standalone_data_generate",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        # Include files left before an interrupted manifest commit; never overwrite.
        self.next_index = (
            max(
                (
                    int(p.stem[3:])
                    for p in self.path.glob("hs_*.safetensors")
                    if p.stem[3:].isdigit()
                ),
                default=-1,
            )
            + 1
        )
        self._index_lock = threading.Lock()
        self._manifest = None
        self._closed = False

    def _entry(self, sample):
        with self._index_lock:
            name = f"hs_{self.next_index:06d}.safetensors"
            self.next_index += 1
        entry = {
            "format": RAW_HIDDEN_FORMAT,
            "schema_version": RAW_HIDDEN_VERSION,
            "path": name,
            "sample_id": sample.sample_id,
            "sequence_no": sample.sequence_no,
            "num_samples": 1,
            "num_tokens": sum(sample.request["loss_mask"]),
            "sequence_length": len(sample.request["feature_positions"]),
            "request": sample.request,
            "feature_contract": sample.feature_contract,
        }
        # Reject unsupported metadata before moving or saving a payload.
        json.dumps(entry, ensure_ascii=False, allow_nan=False)
        return entry

    def stage_file(
        self, reference, request, contract, *, validate_outputs=False, timings=None
    ):
        """Adopt a connector file without materializing its hidden tensor."""
        from safetensors import safe_open

        if timings is None:
            timings = {}
        source = Path(reference)
        started = time.perf_counter()
        _wait_for_lock(Path(f"{source}.lock"))
        timings["file_lock"] = time.perf_counter() - started
        started = time.perf_counter()
        timings["file_bytes"] = source.stat().st_size
        with safe_open(str(source), framework="pt", device="cpu") as saved:
            tokens = saved.get_tensor("token_ids").clone()
            hidden = saved.get_slice("hidden_states")
            dtype = {
                "F16": "torch.float16",
                "BF16": "torch.bfloat16",
                "F32": "torch.float32",
                "F64": "torch.float64",
            }.get(hidden.get_dtype())
            if dtype is None:
                raise ValueError("Raw hidden states require a supported floating dtype")
            sample = _raw_hidden_metadata(
                tokens, hidden.get_shape(), dtype, request, contract, {}
            )
        # Release safetensors mappings before renaming (also required on Windows).
        del saved, tokens, hidden
        timings["file_inspect"] = time.perf_counter() - started
        started = time.perf_counter()
        if validate_outputs:
            with safe_open(str(source), framework="pt", device="cpu") as saved:
                if not torch.isfinite(saved.get_tensor("hidden_states")).all():
                    raise ValueError(
                        "Raw hidden states must be finite floating-point tensors"
                    )
            del saved
        timings["file_validate"] = time.perf_counter() - started
        entry = self._entry(sample)
        with tempfile.NamedTemporaryFile(
            dir=self.path, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            started = time.perf_counter()
            timings["cross_fs"] = 0
            try:
                _replace_hidden_file(source, temporary)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # Cross-filesystem copies remain invisible until the final rename.
                timings["cross_fs"] = 1
                copying = time.perf_counter()
                shutil.copyfile(source, temporary)
                timings["file_copy"] = time.perf_counter() - copying
            _replace_hidden_file(temporary, self.path / entry["path"])
            source.unlink(missing_ok=True)
            timings["file_move"] = time.perf_counter() - started
        finally:
            temporary.unlink(missing_ok=True)
        return entry

    def stage_sample(self, sample):
        """Persist an in-memory backend payload; leave its manifest uncommitted."""
        from safetensors import SafetensorError
        from safetensors.torch import save_file

        if not isinstance(sample, RawHiddenSample):
            raise TypeError("Raw hidden writer requires RawHiddenSample")
        entry = self._entry(sample)
        with tempfile.NamedTemporaryFile(
            dir=self.path, suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
        try:
            try:
                save_file(
                    {key: value.contiguous() for key, value in sample.payload.items()},
                    temporary,
                )
            except SafetensorError as exc:
                raise OSError(f"Failed to save raw hidden payload: {exc}") from exc
            os.replace(temporary, self.path / entry["path"])
        finally:
            Path(temporary).unlink(missing_ok=True)
        return entry

    def commit(self, entry):
        """Single-writer metadata commit, after the payload is in its final place."""
        if self._closed:
            raise ValueError("Raw hidden writer is closed")
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
        if not (self.path / entry["path"]).is_file():
            raise FileNotFoundError(self.path / entry["path"])
        if self._manifest is None:
            self._manifest = (self.path / "manifest.jsonl").open("a", encoding="utf-8")
        self._manifest.write(line)
        self._manifest.flush()
        return entry["path"]

    def write_many(self, samples):
        return [self.commit(self.stage_sample(sample)) for sample in samples]

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._manifest is not None:
            self._manifest.close()


__all__ = [
    "FILE_BACKEND",
    "MOONCAKE_BACKEND",
    "RAW_HIDDEN_FORMAT",
    "RAW_HIDDEN_VERSION",
    "SUPPORTED_BACKENDS",
    "FileHiddenStatesStore",
    "HiddenStatesStore",
    "HiddenStatesStoreConfig",
    "MooncakeHiddenStatesStore",
    "MooncakeStoreSettings",
    "RawHiddenDatasetWriter",
    "RawHiddenSample",
    "build_hidden_states_store",
    "load_raw_hidden_manifest",
    "bind_raw_hidden_contract",
    "prepare_raw_hidden_sample",
]
