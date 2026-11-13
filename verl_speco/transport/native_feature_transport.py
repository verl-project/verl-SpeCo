# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded registered tensor transfer with explicitly acknowledged leases.

The optional native engine is imported only when an instance is constructed.
The scheduler remains responsible for identifying every reader and fencing a
failed reader before authorizing source reuse.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from queue import Empty

import torch

_ALIGN = 2 * 1024 * 1024
_FIELD_ALIGN = 256
_DESCRIPTOR_LIMIT = 64 * 1024
_CHECKSUM_CHUNK = 8 * 1024 * 1024
_CONTROL_LIMIT = 128 * 1024
_DTYPES = {str(dtype): dtype for dtype in (torch.bfloat16, torch.float16, torch.float32,
                                            torch.int64, torch.int32, torch.uint8, torch.bool)}
# Keep registered device allocations alive even if their caller drops the
# transport. Explicit successful unregistration removes the last reference;
# uncertain native operations intentionally retain it until process exit.
_REGISTERED_OWNERS = set()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _aligned(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _content_hash(data):
    """Hash a byte tensor using bounded host-side copies."""
    digest = hashlib.sha256()
    for start in range(0, data.numel(), _CHECKSUM_CHUNK):
        chunk = data[start:start + _CHECKSUM_CHUNK].detach().to(device="cpu", copy=True)
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def _readers(value):
    if (not isinstance(value, (list, tuple)) or not value or len(value) > 4096
            or any(type(rank) is not int or rank < 0 for rank in value) or len(value) != len(set(value))):
        raise ValueError("Invalid native lease readers")
    return frozenset(value)


def _dispatch(plan_id, receipt_token):
    if plan_id is None and receipt_token is None:
        return None
    if (type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str)
            or not 1 <= len(receipt_token) <= 256):
        raise ValueError("Invalid native dispatch identity")
    return plan_id, receipt_token


_CONTROL_ARGUMENTS = {
    "validate_published": frozenset(),
    "bind_readers": frozenset({"readers", "plan_id", "receipt_token"}),
    "acknowledge": frozenset({"reader", "plan_id", "receipt_token"}),
    "release": frozenset({"plan_id", "receipt_token"}),
    "discard_unbound": frozenset(),
    "seal_if_empty": frozenset(),
    "stop_publication": frozenset(),
}


def _control_queue(queue, operation, value=None, *, timeout):
    actor = getattr(queue, "actor", None)
    if actor is None:
        return queue.put(value, timeout=timeout) if operation == "put" else queue.get(timeout=timeout)
    import ray
    ref = actor.put.remote(value, timeout) if operation == "put" else actor.get.remote(timeout)
    try:
        return ray.get(ref, timeout=timeout + 1)
    except ray.exceptions.GetTimeoutError:
        ray.cancel(ref, force=False)
        raise TimeoutError("Native source control queue RPC timed out") from None


def _control_message(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid native source control message") from exc
    if len(encoded) > _CONTROL_LIMIT:
        raise ValueError("Native source control message exceeds its byte limit")
    return json.loads(encoded)


def _validate_control_channel(request_queue, response_queue, channel_id, timeout):
    if (not isinstance(channel_id, str) or len(channel_id) != 32
            or any(c not in "0123456789abcdef" for c in channel_id)
            or isinstance(timeout, bool) or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout) or timeout <= 0
            or any(not callable(getattr(queue, operation, None)) for queue in (request_queue, response_queue)
                   for operation in ("get", "put"))):
        raise ValueError("Invalid native source control channel")


class NativeSourceControlService:
    """Execute bounded lease-control messages in the process owning the source."""

    def __init__(self, source, request_queue, response_queue, *, channel_id, timeout=30.0):
        _validate_control_channel(request_queue, response_queue, channel_id, timeout)
        source._require_open()
        self.source, self.requests, self.responses = source, request_queue, response_queue
        self.channel_id, self.timeout = channel_id, float(timeout)
        self._stopping, self._failed = threading.Event(), None
        self._thread = threading.Thread(target=self._serve, name="native-source-control", daemon=True)
        self._thread.start()

    def _fail(self, error):
        self._failed = error
        try:
            self.source.stop_publication()
        except BaseException:
            # An already quarantined source also rejects future publication.
            pass

    def _serve(self):
        try:
            self._serve_commands()
        finally:
            # Closing the control channel must also close source admission.
            # Keep this on the service thread so close() retains its deadline.
            try:
                self.source.stop_publication()
            except BaseException as exc:
                if self._failed is None:
                    self._failed = exc

    def _serve_commands(self):
        while not self._stopping.is_set():
            try:
                request = _control_queue(self.requests, "get", timeout=min(self.timeout, 0.25))
            except Empty:
                continue
            except BaseException as exc:
                self._fail(exc)
                return
            try:
                request = _control_message(request)
                method, arguments = request.get("method"), request.get("arguments")
                if (set(request) != {"channel_id", "request_id", "method", "descriptor", "arguments"}
                        or request["channel_id"] != self.channel_id
                        or not isinstance(request["request_id"], str) or len(request["request_id"]) != 32
                        or method not in _CONTROL_ARGUMENTS or not isinstance(arguments, dict)
                        or set(arguments) != _CONTROL_ARGUMENTS[method]):
                    raise ValueError("Native source control identity or operation differs")
                if method in {"seal_if_empty", "stop_publication"}:
                    if request["descriptor"] is not None:
                        raise ValueError("Native source seal must not carry a descriptor")
                    getattr(self.source, method)()
                else:
                    getattr(self.source, method)(request["descriptor"], **arguments)
                response = {"channel_id": self.channel_id, "request_id": request["request_id"], "ok": True}
            except BaseException as exc:
                self._fail(exc)
                if (isinstance(request, dict) and request.get("channel_id") == self.channel_id
                        and isinstance(request.get("request_id"), str) and len(request["request_id"]) == 32):
                    try:
                        _control_queue(self.responses, "put", {"channel_id": self.channel_id,
                            "request_id": request["request_id"], "ok": False}, timeout=self.timeout)
                    except BaseException:
                        pass
                return  # Do not accept another command after an ambiguous failure.
            try:
                _control_queue(self.responses, "put", response, timeout=self.timeout)
            except BaseException as exc:
                self._fail(exc)
                return

    def close(self):
        self._stopping.set()
        self._thread.join(timeout=self.timeout + 2)
        if self._thread.is_alive():
            raise RuntimeError("Native source control is still running; source fencing is required")
        if self._failed is not None:
            raise RuntimeError("Native source control failed; retain source buffers") from self._failed


class _NativeControlOperations:
    """Shared lease operations for direct and directory-routed clients."""

    def validate_published(self, descriptor):
        self._call("validate_published", descriptor)

    def bind_readers(self, descriptor, *, readers, plan_id, receipt_token):
        self._call("bind_readers", descriptor, readers=readers, plan_id=plan_id, receipt_token=receipt_token)

    def acknowledge(self, descriptor, *, reader, plan_id, receipt_token):
        self._call("acknowledge", descriptor, reader=reader, plan_id=plan_id, receipt_token=receipt_token)

    def release(self, descriptor, *, plan_id, receipt_token):
        self._call("release", descriptor, plan_id=plan_id, receipt_token=receipt_token)

    def discard_unbound(self, descriptor):
        self._call("discard_unbound", descriptor)


class NativeSourceControlClient(_NativeControlOperations):
    """Serialize lease operations over a dedicated queue pair; never transfer tensors."""

    def __init__(self, request_queue, response_queue, *, channel_id, timeout=30.0):
        _validate_control_channel(request_queue, response_queue, channel_id, timeout)
        self.requests, self.responses, self.channel_id, self.timeout = request_queue, response_queue, channel_id, float(timeout)
        self._lock, self._failed = threading.Lock(), False

    def _call(self, method, descriptor, **arguments):
        with self._lock:
            if self._failed:
                raise RuntimeError("Native source control client is quarantined")
            request_id = uuid.uuid4().hex
            request = _control_message({"channel_id": self.channel_id, "request_id": request_id,
                                        "method": method, "descriptor": descriptor, "arguments": arguments})
            deadline = time.monotonic() + self.timeout
            try:
                _control_queue(self.requests, "put", request, timeout=self.timeout)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Native source control deadline exceeded")
                try:
                    response = _control_message(_control_queue(self.responses, "get", timeout=remaining))
                except Empty as exc:
                    raise TimeoutError("Native source control deadline exceeded") from exc
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native source control deadline exceeded")
                if response == {"channel_id": self.channel_id, "request_id": request_id, "ok": False}:
                    raise RuntimeError("Native source control operation failed; retain source buffers")
                if response != {"channel_id": self.channel_id, "request_id": request_id, "ok": True}:
                    raise ValueError("Native source control acknowledgement identity differs")
            except BaseException:
                self._failed = True
                raise

    def seal_if_empty(self):
        self._call("seal_if_empty", None)

    def stop_publication(self):
        self._call("stop_publication", None)


class NativeSourceControlRouter(_NativeControlOperations):
    """Route each descriptor to its source-owned control channel."""

    def __init__(self, routes):
        if (not isinstance(routes, dict) or not 1 <= len(routes) <= 4096
                or any(not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 256 for endpoint in routes)
                or any(not isinstance(options, dict) for options in routes.values())):
            raise ValueError("Invalid native source control routes")
        self._routes = {endpoint: NativeSourceControlClient(**options) for endpoint, options in routes.items()}

    def _call(self, method, descriptor, **arguments):
        endpoint = descriptor.get("segment") if isinstance(descriptor, dict) else None
        if not isinstance(endpoint, str) or endpoint not in self._routes:
            raise ValueError("Native descriptor has no configured source control route")
        return getattr(self._routes[endpoint], method)(descriptor, **arguments)


class NativeSourceDirectory:
    """Bounded run-scoped handle registry for source control and descriptors."""

    def __init__(self, *, run_id, generation, max_sources, max_records, max_descriptor_bytes=256 * 1024**2,
                 source_requirements=None):
        if (not isinstance(run_id, str) or not 1 <= len(run_id) <= 256
                or type(generation) is not int or generation < 0
                or type(max_sources) is not int or not 1 <= max_sources <= 4096
                or type(max_records) is not int or not 1 <= max_records <= 10_000_000
                or type(max_descriptor_bytes) is not int or not 1 <= max_descriptor_bytes <= 16 * 1024**3):
            raise ValueError("Invalid native source directory scope or bounds")
        self.run_id, self.generation = run_id, generation
        self.max_sources, self.max_records = max_sources, max_records
        self.max_descriptor_bytes, self._descriptor_bytes = max_descriptor_bytes, 0
        if source_requirements is not None:
            required = {"max_bytes", "max_slots", "chunk_bytes", "metadata_server", "reader_endpoints"}
            if (not isinstance(source_requirements, dict) or set(source_requirements) != required
                    or any(type(source_requirements[key]) is not int or source_requirements[key] <= 0
                           for key in ("max_bytes", "max_slots", "chunk_bytes"))
                    or not isinstance(source_requirements["metadata_server"], str)
                    or not source_requirements["metadata_server"]
                    or not isinstance(source_requirements["reader_endpoints"], list)
                    or any(not isinstance(item, str) or not item for item in source_requirements["reader_endpoints"])):
                raise ValueError("Invalid native source requirements")
            source_requirements = _control_message(source_requirements)
        self._source_requirements = source_requirements
        self._sources, self._records, self._active_handles = {}, {}, set()
        self._condition, self._closed, self._frozen = asyncio.Condition(), False, False

    async def register_source(self, endpoint, source_id, options, source_settings=None):
        if (not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 256
                or not isinstance(source_id, str) or len(source_id) != 32
                or any(c not in "0123456789abcdef" for c in source_id)
                or not isinstance(options, dict) or set(options) != {"request_queue", "response_queue", "channel_id", "timeout"}):
            raise ValueError("Invalid native source registration")
        # Validate the channel without opening a connection or sending data.
        NativeSourceControlClient(**options)
        if self._source_requirements is not None:
            required = self._source_requirements
            if (not isinstance(source_settings, dict)
                    or set(source_settings) != {"max_bytes", "max_slots", "chunk_bytes", "metadata_server"}
                    or any(type(source_settings[key]) is not int or source_settings[key] < required[key]
                           for key in ("max_bytes", "max_slots"))
                    or type(source_settings["chunk_bytes"]) is not int
                    or source_settings["chunk_bytes"] != required["chunk_bytes"]
                    or source_settings["metadata_server"] != required["metadata_server"]
                    or endpoint in required["reader_endpoints"]):
                raise ValueError("Native source configuration does not meet directory requirements")
        async with self._condition:
            if self._closed or self._frozen:
                raise RuntimeError("Native source directory is closed or frozen")
            previous = self._sources.get(endpoint)
            if previous is not None:
                if previous[0] != source_id or previous[1]["channel_id"] != options["channel_id"]:
                    raise ValueError("Native source endpoint belongs to another incarnation")
                return
            if len(self._sources) >= self.max_sources:
                raise ValueError("Native source directory source limit exceeded")
            self._sources[endpoint] = (source_id, dict(options))
            self._condition.notify_all()

    async def source_route(self, endpoint):
        async with self._condition:
            if self._closed or endpoint not in self._sources:
                raise ValueError("Native source control route is unavailable")
            return dict(self._sources[endpoint][1])

    async def scope(self):
        async with self._condition:
            if self._closed:
                raise RuntimeError("Native source directory is closed")
            if self._frozen:
                raise RuntimeError("Native source directory is frozen")
            return {"run_id": self.run_id, "generation": self.generation}

    async def publish(self, handle, descriptor):
        if not isinstance(handle, str) or not 1 <= len(handle) <= 256:
            raise ValueError("Invalid native source handle")
        descriptor = _control_message(descriptor)
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid native source descriptor")
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if (len(encoded) > _DESCRIPTOR_LIMIT or type(descriptor.get("version")) is not int or descriptor["version"] != 2
                or descriptor.get("run_id") != self.run_id or type(descriptor.get("generation")) is not int
                or descriptor["generation"] != self.generation
                or descriptor.get("lease_id") != handle or not isinstance(descriptor.get("segment"), str)
                or descriptor.get("sha256") != _fingerprint({key: value for key, value in descriptor.items() if key != "sha256"})):
            raise ValueError("Native source descriptor identity differs")
        async with self._condition:
            if self._closed or self._frozen or descriptor["segment"] not in self._sources:
                raise ValueError("Native source endpoint is not registered")
            previous = self._records.get(handle)
            if previous is not None:
                if previous[0] != descriptor["sha256"] or previous[1] is None:
                    raise ValueError("Native source handle was reused or retired")
                return
            if len(self._records) >= self.max_records:
                raise ValueError("Native source directory record limit exceeded")
            if self._descriptor_bytes + len(encoded) > self.max_descriptor_bytes:
                raise ValueError("Native source directory descriptor byte limit exceeded")
            self._active_handles.add(handle)
            self._records[handle] = (descriptor["sha256"], descriptor)
            self._descriptor_bytes += len(encoded)
            self._condition.notify_all()

    async def resolve(self, handle, timeout):
        if (not isinstance(handle, str) or not 1 <= len(handle) <= 256
                or isinstance(timeout, bool) or not isinstance(timeout, (float, int))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid native source handle or timeout")
        async with self._condition:
            await asyncio.wait_for(self._condition.wait_for(
                lambda: self._closed or self._frozen or handle in self._records), timeout)
            record = self._records.get(handle)
            if self._closed or record is None or record[1] is None:
                raise ValueError("Native source handle is closed or retired")
            return _control_message(record[1])

    async def retire(self, handle, fingerprint):
        async with self._condition:
            record = self._records.get(handle)
            if self._closed or record is None or record[0] != fingerprint:
                raise ValueError("Native source retirement identity differs")
            self._records[handle] = (record[0], None)
            if record[1] is not None:
                self._descriptor_bytes -= len(json.dumps(record[1], sort_keys=True, separators=(",", ":"),
                                                         allow_nan=False).encode())
                self._active_handles.remove(handle)

    async def status(self, limit=0):
        """Return bounded metadata; source device/process state is not inferred."""
        if type(limit) is not int or not 0 <= limit <= 64:
            raise ValueError("Native directory status limit must be between 0 and 64")
        async with self._condition:
            active = []
            for handle in self._active_handles:
                if len(active) >= limit:
                    break
                fingerprint, descriptor = self._records[handle]
                active.append({"handle": handle, "segment": descriptor["segment"], "sha256": fingerprint})
            return {"run_id": self.run_id, "generation": self.generation, "closed": self._closed,
                    "frozen": self._frozen, "sources": len(self._sources), "records": len(self._records),
                    "active": len(self._active_handles), "max_sources": self.max_sources,
                    "max_records": self.max_records, "max_descriptor_bytes": self.max_descriptor_bytes,
                    "active_descriptor_bytes": self._descriptor_bytes,
                    "active_sample": active}

    async def close(self):
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def freeze(self, require_empty=True):
        """Prevent new sources and publications before checking source buffers."""
        if type(require_empty) is not bool:
            raise ValueError("require_empty must be a boolean")
        async with self._condition:
            if self._closed:
                raise RuntimeError("Native source directory is closed")
            self._frozen = True
            self._condition.notify_all()
            if require_empty and self._active_handles:
                raise RuntimeError("Native source directory has active references")
            return list(self._sources)


class NativeSourceDirectoryRouter(_NativeControlOperations):
    """Resolve source channels and producer handles through a run-scoped Ray directory."""

    def __init__(self, directory, *, timeout=30.0):
        if (directory is None or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid native source directory client")
        self.directory, self.timeout = directory, float(timeout)
        self._routes, self._lock = {}, threading.Lock()
        self._stop_lock, self._stop_result = threading.Lock(), None

    def _directory_call(self, method, *args):
        import ray
        return ray.get(getattr(self.directory, method).remote(*args), timeout=self.timeout + 1)

    def resolve(self, handle):
        return self._directory_call("resolve", handle, self.timeout)

    def _client(self, descriptor):
        endpoint = descriptor.get("segment") if isinstance(descriptor, dict) else None
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("Native descriptor has no source endpoint")
        with self._lock:
            existing = self._routes.get(endpoint)
        if existing is not None:
            return existing
        # Never hold the shared cache lock across an RPC: one unavailable route
        # must not block attempts to stop publication on every other source.
        options = self._directory_call("source_route", endpoint)
        candidate = NativeSourceControlClient(**options)
        with self._lock:
            return self._routes.setdefault(endpoint, candidate)

    def _call(self, method, descriptor, **arguments):
        getattr(self._client(descriptor), method)(descriptor, **arguments)
        if method in {"release", "discard_unbound"}:
            self._directory_call("retire", descriptor["lease_id"], descriptor["sha256"])

    def freeze_and_seal(self):
        for endpoint in self._directory_call("freeze"):
            self._client({"segment": endpoint}).seal_if_empty()

    def stop_admission(self):
        """Freeze publication while retaining active descriptors and source memory.

        This is an admission barrier; running readers still require fencing.
        """
        with self._stop_lock:
            owner = self._stop_result is None
            if owner:
                self._stop_result = Future()
            result = self._stop_result
        if not owner:
            return result.result(timeout=self.timeout + 1)
        try:
            self._stop_admission_once()
        except BaseException as exc:
            result.set_exception(exc)
            raise
        else:
            result.set_result(None)

    def _stop_admission_once(self):
        deadline = time.monotonic() + self.timeout + 1
        endpoints = self._directory_call("freeze", False)
        if not endpoints:
            return
        executor = ThreadPoolExecutor(max_workers=min(8, len(endpoints)), thread_name_prefix="speco-source-stop")
        futures = []
        try:
            def stop(endpoint):
                self._client({"segment": endpoint}).stop_publication()
            futures = [executor.submit(stop, endpoint) for endpoint in endpoints]
            done, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
            failures = [future.exception() for future in done if future.exception() is not None]
            if pending or failures:
                error = failures[0] if failures else TimeoutError("Native source stop deadline exceeded")
                raise RuntimeError(f"Native source publication stop unconfirmed for {len(pending) + len(failures)} sources") from error
        finally:
            # A running control RPC cannot be cancelled; retain all source buffers.
            executor.shutdown(wait=False, cancel_futures=True)


class NativeProducerReferenceStore:
    """Resolve vLLM handles and retire only references never sent to readers."""

    backend = "native"

    def __init__(self, router):
        self.router = router

    def resolve_reference(self, reference):
        descriptor = self.router.resolve(reference)
        return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def release(self, reference):
        try:
            descriptor = json.loads(reference)
        except (TypeError, ValueError):
            descriptor = None
        if not isinstance(descriptor, dict) or not {"version", "lease_id", "segment", "sha256"} <= descriptor.keys():
            descriptor = self.router.resolve(reference)
        self.router.discard_unbound(descriptor)

    def close(self):
        pass


class NativeRetiredReferenceStore:
    """Driver adapter: source release is performed by the Producer before cleanup."""

    backend = "native"

    def release(self, reference):
        from verl_speco.producer.hidden_states_store import NativeHiddenStatesStore
        NativeHiddenStatesStore._descriptor(reference)

    def close(self):
        pass


class NativeFeatureTransport:
    """Own registered ranges until all bound readers acknowledge completion."""

    def __init__(self, *, run_id, generation, endpoint, metadata_server, device,
                 max_bytes, max_slots, chunk_bytes, engine_factory=None):
        if (not isinstance(run_id, str) or not 1 <= len(run_id) <= 256
                or type(generation) is not int or generation < 0
                or not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 256
                or not isinstance(metadata_server, str) or not metadata_server
                or type(max_bytes) is not int or max_bytes < 2 * _ALIGN - 1
                or type(max_slots) is not int or not 1 <= max_slots <= 4096
                or type(chunk_bytes) is not int or chunk_bytes < _ALIGN or chunk_bytes % _ALIGN):
            raise ValueError("Invalid native transport scope or limits")
        native_engine = engine_factory is None
        if native_engine:
            import torch_npu  # noqa: F401 - register the device type lazily
            from mooncake.engine import TransferEngine
            engine_factory = TransferEngine
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "npu"} or (native_engine and self.device.type != "npu"):
            raise ValueError("Native transport requires a CPU test device or NPU")
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
        self.run_id, self.generation, self.endpoint = run_id, generation, endpoint
        self._pid = os.getpid()
        self.max_bytes, self.max_slots, self.chunk_bytes = max_bytes, max_slots, chunk_bytes
        self._lock, self._slots, self._registered, self._closed = threading.RLock(), {}, {}, False
        self._registered_bytes = 0
        self._completed = OrderedDict()
        self._completed_readers = 0
        self._failed, self._sealed = False, False
        self._read_metrics = {}
        self._metrics_lock = threading.Lock()
        self._engine = engine_factory()
        self._require_zero("initialize", self._engine.initialize(endpoint, metadata_server, "ascend", ""))

    @staticmethod
    def _require_zero(operation, result):
        if result != 0:
            raise RuntimeError(f"Native transport {operation} failed: {result}")

    def _require_open(self):
        if os.getpid() != self._pid:
            raise RuntimeError("Native transport cannot be used from a different process")
        if self._closed:
            raise RuntimeError("Native transport is closed")
        if self._failed:
            raise RuntimeError("Native transport is quarantined after an uncertain native operation")

    def _device_scope(self):
        if os.getpid() != self._pid:
            raise RuntimeError("Native transport cannot be used from a different process")
        return torch.npu.device(self.device) if self.device.type == "npu" else nullcontext()

    def _synchronize(self):
        if self.device.type == "npu":
            try:
                torch.npu.synchronize(self.device)
            except BaseException:
                self._failed = True
                raise

    def _register(self, size):
        size = _aligned(size, _ALIGN)
        allocation_bytes = size + _ALIGN - 1
        if (len(self._registered) >= self.max_slots
                or self._registered_bytes + allocation_bytes > self.max_bytes):
            raise ValueError("Native registered slot or byte budget exceeded")
        allocation = torch.empty(allocation_bytes, dtype=torch.uint8, device=self.device)
        start = (-allocation.data_ptr()) % _ALIGN
        buffer = allocation[start:start + size]
        self._registered[buffer.data_ptr()] = buffer
        self._registered_bytes += allocation.untyped_storage().nbytes()
        _REGISTERED_OWNERS.add(self)
        try:
            self._require_zero("register_memory", self._engine.register_memory(buffer.data_ptr(), size))
        except BaseException:
            # An error may be ambiguous; never free a possibly registered range.
            self._failed = True
            raise
        return buffer

    def _unregister(self, buffer):
        address = buffer.data_ptr()
        try:
            self._require_zero("unregister_memory", self._engine.unregister_memory(address))
        except BaseException:
            self._failed = True
            raise
        del self._registered[address]
        self._registered_bytes -= buffer.untyped_storage().nbytes()
        if not self._registered:
            _REGISTERED_OWNERS.discard(self)

    def stop_publication(self):
        """Forbid new source allocations while preserving every active lease."""
        with self._lock:
            if self._closed and os.getpid() == self._pid:
                return
            self._require_open()
            self._sealed = True

    def seal_if_empty(self):
        """Forbid new publications and confirm no registered source memory remains."""
        with self._lock:
            self._require_open()
            self._sealed = True
            if self._registered:
                raise RuntimeError("Native source still owns registered memory")

    def publish(self, fields, *, lease_id, readers=None):
        with self._lock, self._device_scope():
            self._require_open()
            if self._sealed:
                raise RuntimeError("Native source is sealed")
            if (not isinstance(lease_id, str) or not 1 <= len(lease_id) <= 256
                    or len(self._slots) >= self.max_slots
                    or not isinstance(fields, dict) or not fields or len(fields) > 64
                    or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in fields)):
                raise ValueError("Invalid native lease, readers, fields or slot limit")
            expected = None if readers is None else _readers(readers)
            if lease_id in self._slots:
                raise ValueError("Native lease is already active")
            if lease_id in self._completed:
                raise ValueError("Native lease identifier was already completed")
            layout, offset = [], _FIELD_ALIGN
            for name in sorted(fields):
                tensor = fields[name]
                if (not isinstance(tensor, torch.Tensor) or (tensor.device.type != "cpu" and tensor.device != self.device)
                        or not tensor.is_contiguous() or not tensor.numel() or tensor.ndim > 16
                        or str(tensor.dtype) not in _DTYPES):
                    raise ValueError("Native fields require contiguous nonempty CPU or local device tensors")
                offset = _aligned(offset, _FIELD_ALIGN)
                nbytes = tensor.numel() * tensor.element_size()
                layout.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                               "offset": offset, "nbytes": nbytes})
                offset += nbytes
            buffer = self._register(offset)
            try:
                for item in layout:
                    buffer[item["offset"]:item["offset"] + item["nbytes"]].copy_(fields[item["name"]].reshape(-1).view(torch.uint8))
                self._synchronize()
                for item in layout:
                    item["sha256"] = _content_hash(buffer[item["offset"]:item["offset"] + item["nbytes"]])
                descriptor = {"version": 2, "run_id": self.run_id, "generation": self.generation,
                              "lease_id": lease_id, "nonce": uuid.uuid4().hex, "segment": self.endpoint,
                              "address": buffer.data_ptr(), "nbytes": offset, "fields": layout}
                descriptor["sha256"] = _fingerprint(descriptor)
                buffer[:64].copy_(torch.tensor(list(descriptor["sha256"].encode()), dtype=torch.uint8, device=self.device))
                self._synchronize()
            except BaseException:
                self._synchronize()
                if not self._failed:
                    self._unregister(buffer)
                raise
            self._slots[lease_id] = {"buffer": buffer, "descriptor": descriptor,
                                     "expected": expected, "pending": None if expected is None else set(expected),
                                     "dispatch": None, "acknowledged": set()}
            return json.loads(json.dumps(descriptor))

    def _validate(self, descriptor):
        keys = {"version", "run_id", "generation", "lease_id", "nonce", "segment", "address", "nbytes", "fields", "sha256"}
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid native feature descriptor or scope")
        try:
            serialized = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if len(serialized) > _DESCRIPTOR_LIMIT:
                raise ValueError("Native feature descriptor exceeds its byte limit")
            descriptor = json.loads(serialized)
        except (TypeError, ValueError, OverflowError, RecursionError, RuntimeError) as exc:
            raise ValueError("Invalid native feature descriptor or scope") from exc
        if (set(descriptor) != keys
                or type(descriptor["version"]) is not int or descriptor["version"] != 2
                or descriptor["run_id"] != self.run_id or type(descriptor["generation"]) is not int
                or descriptor["generation"] != self.generation
                or not isinstance(descriptor["lease_id"], str) or not 1 <= len(descriptor["lease_id"]) <= 256
                or not isinstance(descriptor["nonce"], str) or len(descriptor["nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in descriptor["nonce"])
                or not isinstance(descriptor["segment"], str) or not 1 <= len(descriptor["segment"]) <= 256
                or type(descriptor["address"]) is not int or descriptor["address"] <= 0
                or type(descriptor["nbytes"]) is not int or not _FIELD_ALIGN < descriptor["nbytes"] <= self.max_bytes
                or descriptor["address"] + descriptor["nbytes"] > 2**64
                or not isinstance(descriptor["fields"], list) or not 1 <= len(descriptor["fields"]) <= 64
                or not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64):
            raise ValueError("Invalid native feature descriptor or scope")
        try:
            valid_hash = descriptor["sha256"] == _fingerprint({k: v for k, v in descriptor.items() if k != "sha256"})
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Invalid native feature descriptor or scope") from exc
        if not valid_hash:
            raise ValueError("Invalid native feature descriptor or scope")
        end, names = _FIELD_ALIGN, set()
        for field in descriptor["fields"]:
            if (not isinstance(field, dict) or set(field) != {"name", "shape", "dtype", "offset", "nbytes", "sha256"}
                    or not isinstance(field["name"], str) or not 1 <= len(field["name"]) <= 128
                    or field["name"] in names or not isinstance(field["dtype"], str) or field["dtype"] not in _DTYPES
                    or not isinstance(field["shape"], list) or len(field["shape"]) > 16
                    or any(type(dim) is not int or dim <= 0 for dim in field["shape"])
                    or type(field["offset"]) is not int or field["offset"] % _FIELD_ALIGN
                    or type(field["nbytes"]) is not int or field["offset"] < end
                    or field["nbytes"] != math.prod(field["shape"]) * torch.empty((), dtype=_DTYPES[field["dtype"]]).element_size()
                    or not isinstance(field["sha256"], str) or len(field["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in field["sha256"])
                    or field["offset"] + field["nbytes"] > descriptor["nbytes"]):
                raise ValueError("Invalid native feature field layout")
            end = field["offset"] + field["nbytes"]
            names.add(field["name"])
        if end != descriptor["nbytes"]:
            raise ValueError("Native feature descriptor has trailing data")
        return descriptor

    def read(self, descriptor, *, output_device=None):
        """Synchronously copy into independent tensors and verify the lease twice."""
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            destination = self.device if output_device is None else torch.device(output_device)
            if destination.type != "cpu" and destination != self.device:
                raise ValueError("Native transport output must use CPU or the local transport device")
            started = time.monotonic()
            buffer = self._register(descriptor["nbytes"])
            registered = time.monotonic()
            transfer_failed = False
            try:
                for offset in range(0, descriptor["nbytes"], self.chunk_bytes):
                    count = min(self.chunk_bytes, descriptor["nbytes"] - offset)
                    try:
                        self._require_zero("transfer_sync_read", self._engine.transfer_sync_read(
                            descriptor["segment"], buffer.data_ptr() + offset, descriptor["address"] + offset, count))
                    except BaseException:
                        transfer_failed = self._failed = True
                        raise
                self._synchronize()
                transferred = time.monotonic()
                if bytes(buffer[:64].cpu().tolist()) != descriptor["sha256"].encode():
                    raise RuntimeError("Native source lease changed during transfer")
                for field in descriptor["fields"]:
                    data = buffer[field["offset"]:field["offset"] + field["nbytes"]]
                    if _content_hash(data) != field["sha256"]:
                        raise RuntimeError("Native feature payload checksum differs")
                checked = time.monotonic()
                result = {field["name"]: buffer[field["offset"]:field["offset"] + field["nbytes"]]
                          .view(_DTYPES[field["dtype"]]).reshape(field["shape"])
                          .to(device=destination, copy=True).contiguous()
                          for field in descriptor["fields"]}
                self._synchronize()
                materialized = time.monotonic()
                try:
                    self._require_zero("transfer_sync_read", self._engine.transfer_sync_read(
                        descriptor["segment"], buffer.data_ptr(), descriptor["address"], 64))
                except BaseException:
                    transfer_failed = self._failed = True
                    raise
                self._synchronize()
                if bytes(buffer[:64].cpu().tolist()) != descriptor["sha256"].encode():
                    raise RuntimeError("Native source lease changed during transfer")
                verified = time.monotonic()
            finally:
                self._synchronize()
                if not transfer_failed and not self._failed:
                    self._unregister(buffer)
            finished = time.monotonic()
            values = {
                "reads_total": 1, "descriptor_bytes_total": descriptor["nbytes"],
                "transfer_bytes_total": descriptor["nbytes"] + 64,
                "register_seconds_total": registered - started,
                "transfer_seconds_total": transferred - registered,
                "checksum_seconds_total": checked - transferred,
                "materialize_seconds_total": materialized - checked,
                "lease_verify_seconds_total": verified - materialized,
                "cleanup_seconds_total": finished - verified,
                "read_seconds_total": finished - started,
            }
            with self._metrics_lock:
                for name, value in values.items():
                    self._read_metrics[name] = self._read_metrics.get(name, 0.0) + value
            return result

    def metrics(self):
        """Successful reads only; all intervals use this process's monotonic clock."""
        with self._metrics_lock:
            return {f"native/{key}": float(value) for key, value in self._read_metrics.items()}

    def validate_published(self, descriptor):
        """Verify that a descriptor names an active buffer owned by this process."""
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            slot = self._slots.get(descriptor["lease_id"])
            if slot is None or slot["descriptor"] != descriptor:
                raise ValueError("Native descriptor is not an active source lease")

    def bind_readers(self, descriptor, *, readers, plan_id, receipt_token):
        """Bind one published source to the scheduler's immutable dispatch."""
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            expected, dispatch = _readers(readers), _dispatch(plan_id, receipt_token)
            if dispatch is None:
                raise ValueError("Native source binding requires dispatch identity")
            slot = self._slots.get(descriptor["lease_id"])
            if slot is None:
                completed = self._completed.get(descriptor["lease_id"])
                if completed is not None and completed == (descriptor["sha256"], expected, dispatch):
                    return
                raise ValueError("Native source binding identity differs")
            if slot["descriptor"] != descriptor:
                raise ValueError("Native source binding identity differs")
            if slot["expected"] is not None:
                if slot["expected"] == expected and slot["dispatch"] == dispatch:
                    return
                # A producer may know the intended ranks before the scheduler
                # assigns a plan. Bind that plan only before any legacy ACK so
                # a receipt can never be moved between dispatch identities.
                if (slot["expected"] == expected and slot["dispatch"] is None
                        and not slot["acknowledged"]):
                    slot["dispatch"] = dispatch
                    return
                raise ValueError("Native source is already bound to another dispatch")
            slot["expected"], slot["pending"], slot["dispatch"] = expected, set(expected), dispatch

    def acknowledge(self, descriptor, *, reader, plan_id=None, receipt_token=None):
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            dispatch = _dispatch(plan_id, receipt_token)
            slot = self._slots.get(descriptor["lease_id"])
            if slot is None:
                completed = self._completed.get(descriptor["lease_id"])
                if (completed is not None and completed[0] == descriptor["sha256"] and completed[2] == dispatch
                        and type(reader) is int and reader in completed[1]):
                    return
                raise ValueError("Native lease acknowledgement identity differs")
            if slot["descriptor"] != descriptor or type(reader) is not int or slot["dispatch"] != dispatch:
                raise ValueError("Native lease acknowledgement identity differs")
            if slot["pending"] is None:
                raise RuntimeError("Native source lease has not been bound to readers")
            if reader in slot["acknowledged"]:
                return
            if reader not in slot["pending"]:
                raise ValueError("Native lease acknowledgement reader differs")
            slot["pending"].remove(reader)
            slot["acknowledged"].add(reader)

    def release(self, descriptor, *, plan_id=None, receipt_token=None):
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            dispatch = _dispatch(plan_id, receipt_token)
            slot = self._slots.get(descriptor["lease_id"])
            if slot is None:
                completed = self._completed.get(descriptor["lease_id"])
                if completed is not None and completed[0] == descriptor["sha256"] and completed[2] == dispatch:
                    return
                raise ValueError("Native source lease identity differs")
            if slot["descriptor"] != descriptor or slot["dispatch"] != dispatch:
                raise ValueError("Native source lease identity differs")
            if slot["pending"] is None:
                raise RuntimeError("Native source lease has not been bound to readers")
            if slot["pending"]:
                raise RuntimeError("Native source lease still has unacknowledged readers")
            self._retire_slot(descriptor, slot, dispatch)

    def discard_unbound(self, descriptor):
        """Discard an unpublished-to-readers source after the scheduler fences dispatch."""
        with self._lock, self._device_scope():
            self._require_open()
            descriptor = self._validate(descriptor)
            slot = self._slots.get(descriptor["lease_id"])
            if slot is None:
                completed = self._completed.get(descriptor["lease_id"])
                if completed == (descriptor["sha256"], frozenset(), None):
                    return
                raise ValueError("Native source discard identity differs")
            if slot["descriptor"] != descriptor or slot["dispatch"] is not None or slot["acknowledged"]:
                raise ValueError("Native source has already been dispatched or read")
            self._retire_slot(descriptor, slot, None)

    def _retire_slot(self, descriptor, slot, dispatch):
        try:
            slot["buffer"][:64].zero_()
            self._synchronize()
        except BaseException:
            # A failed device operation may still have been submitted.
            # Keep the registration and prohibit reuse until fenced.
            self._failed = True
            raise
        self._unregister(slot["buffer"])
        readers = frozenset(slot["acknowledged"])
        self._completed[descriptor["lease_id"]] = (descriptor["sha256"], readers, dispatch)
        self._completed_readers += len(readers)
        while len(self._completed) > 4096 or self._completed_readers > 8192:
            _, (_, old_readers, _) = self._completed.popitem(last=False)
            self._completed_readers -= len(old_readers)
        del self._slots[descriptor["lease_id"]]

    def close(self):
        with self._lock, self._device_scope():
            if os.getpid() != self._pid:
                raise RuntimeError("Native transport cannot be closed from a different process")
            if self._closed:
                return
            if self._failed:
                raise RuntimeError("Cannot close a quarantined native transport without process fencing")
            if self._slots:
                raise RuntimeError("Cannot close native transport with active source leases")
            self._synchronize()
            for buffer in list(self._registered.values()):
                self._unregister(buffer)
            self._closed = True
            self._engine = None
