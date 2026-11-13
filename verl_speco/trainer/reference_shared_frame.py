# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded immutable local tensor frames; Linux resources are loaded lazily."""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import socket
import stat
import struct
import time

import torch
from safetensors import safe_open
from safetensors.torch import load, save

_CONTROL_LIMIT = 64 * 1024


def _checksum(manifest, payload):
    digest = _manifest_digest(manifest)
    digest.update(payload)
    return digest.hexdigest()


def _manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode())


def _map_tensors(path):
    with safe_open(path, framework="pt", device="cpu") as opened:
        return {name: opened.get_tensor(name) for name in opened.keys()}


def _binding(value):
    if (not isinstance(value, dict) or set(value) != {"run_id", "generation", "batch_id", "rank", "receipt_token"}
            or not isinstance(value["run_id"], str) or not value["run_id"] or len(value["run_id"]) > 256
            or not isinstance(value["receipt_token"], str) or not value["receipt_token"]
            or len(value["receipt_token"]) > 256
            or any(type(value[key]) is not int or value[key] < 0 for key in ("generation", "batch_id", "rank"))):
        raise ValueError("Invalid local frame identity")
    return dict(value)


def _metadata(value, depth=0):
    """Accept only bounded-depth JSON data; never deserialize arbitrary objects."""
    if depth > 8:
        raise ValueError("Local frame metadata nesting is too deep")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_metadata(item, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata(item, depth + 1) for key, item in value.items()}
    raise ValueError("Local frame metadata must contain finite JSON values")


def encode_local_frame(batch, *, binding, max_bytes):
    """Serialize a prepared CPU batch without executable pickle data."""
    identity = _binding(binding)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("Local frame byte limit must be positive")
    if not isinstance(batch, dict) or any(not isinstance(name, str) or not name for name in batch):
        raise ValueError("Local frame requires named batch values")
    tensors = {name: value for name, value in batch.items() if isinstance(value, torch.Tensor)}
    metadata = {name: _metadata(value) for name, value in batch.items() if not isinstance(value, torch.Tensor)}
    if not tensors or any(value.device.type != "cpu" or not value.is_contiguous() for value in tensors.values()):
        raise ValueError("Local frame requires named contiguous CPU tensors")
    if len(json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()) > _CONTROL_LIMIT:
        raise ValueError("Local frame metadata exceeds its byte limit")
    # Bound tensor bytes before serialization allocates its output copy.
    if sum(value.numel() * value.element_size() for value in tensors.values()) > max_bytes:
        raise ValueError("Local frame exceeds its byte limit")
    payload = save(tensors)
    if len(payload) > max_bytes:
        raise ValueError("Local frame exceeds its byte limit")
    descriptor = {"binding": identity, "nbytes": len(payload), "metadata": metadata,
                  "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for key, value in tensors.items()}}
    descriptor["sha256"] = _checksum(descriptor, payload)
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > _CONTROL_LIMIT:
        raise ValueError("Local frame descriptor exceeds its byte limit")
    return payload, encoded


def _parse_manifest(descriptor, *, binding, max_bytes, size):
    if not isinstance(descriptor, bytes) or len(descriptor) > _CONTROL_LIMIT:
        raise ValueError("Invalid local frame descriptor")
    try:
        manifest = json.loads(descriptor)
        if (not isinstance(manifest, dict) or set(manifest) != {"binding", "nbytes", "sha256", "tensors", "metadata"}
                or _binding(manifest["binding"]) != _binding(binding)
                or type(max_bytes) is not int or type(manifest["nbytes"]) is not int
                or not 0 < size <= max_bytes or manifest["nbytes"] != size):
            raise ValueError("Local frame identity or length differs")
        metadata = _metadata(manifest["metadata"])
        if not isinstance(metadata, dict):
            raise ValueError("Local frame metadata must be a mapping")
        return manifest, metadata
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Malformed local frame descriptor") from exc


def _validate_tensors(manifest, metadata, tensors):
    actual = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
              for key, value in tensors.items()}
    if not tensors or actual != manifest["tensors"] or set(metadata) & set(tensors):
        raise ValueError("Local frame tensor metadata differs")
    return {**tensors, **metadata}


def decode_local_frame(payload, descriptor, *, binding, max_bytes):
    if not isinstance(payload, bytes):
        raise ValueError("Invalid local frame payload")
    manifest, metadata = _parse_manifest(descriptor, binding=binding, max_bytes=max_bytes, size=len(payload))
    if manifest["sha256"] != _checksum({k: v for k, v in manifest.items() if k != "sha256"}, payload):
        raise ValueError("Local frame checksum differs")
    return _validate_tensors(manifest, metadata, load(payload))


def _read_mapped_frame(fd, descriptor, *, binding, max_bytes, deadline=None):
    """Verify the entire sealed file, then map its tensors without a payload copy."""
    size = os.fstat(fd).st_size
    manifest, metadata = _parse_manifest(descriptor, binding=binding, max_bytes=max_bytes, size=size)
    digest = _manifest_digest({key: value for key, value in manifest.items() if key != "sha256"})
    offset = 0
    while offset < size:
        if deadline is not None:
            _remaining_time(deadline)
        chunk = os.pread(fd, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError("Truncated local frame")
        digest.update(chunk)
        offset += len(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("Local frame checksum differs")
    if deadline is not None:
        _remaining_time(deadline)
    batch = _validate_tensors(manifest, metadata, _map_tensors(f"/proc/self/fd/{fd}"))
    if deadline is not None:
        _remaining_time(deadline)
    return batch


def create_local_frame(payload, *, max_bytes):
    """Create a sealed anonymous FD; the caller owns and must close it."""
    if (not isinstance(payload, bytes) or type(max_bytes) is not int
            or not 0 < len(payload) <= max_bytes):
        raise ValueError("Invalid local frame payload or byte limit")
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("Local shared frames require Linux memfd")
    import fcntl

    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    fd = os.memfd_create("speco-reference-frame", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        if os.fstat(fd).st_size != len(payload):
            raise RuntimeError("Incomplete local frame write")
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_local_socket(connection):
    if not hasattr(socket, "SCM_RIGHTS") or not hasattr(socket, "SOCK_SEQPACKET"):
        raise RuntimeError("Local shared frames require Unix descriptor passing")
    if connection.family != socket.AF_UNIX or connection.type != socket.SOCK_SEQPACKET:
        raise ValueError("Local frames require a message-oriented Unix socket")
    if not hasattr(socket, "SO_PEERCRED") or not hasattr(os, "geteuid"):
        raise RuntimeError("Local shared frames require Linux peer credentials")
    _, peer_uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if peer_uid != os.geteuid():
        raise ValueError("Local frame peer has a different user identity")


def send_local_frame(connection, fd, descriptor):
    _verify_local_socket(connection)
    if not isinstance(descriptor, bytes) or not 0 < len(descriptor) <= _CONTROL_LIMIT:
        raise ValueError("Invalid local frame descriptor")
    rights = array.array("i", [fd])
    if connection.sendmsg([descriptor], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]) != len(descriptor):
        raise RuntimeError("Incomplete local frame send")


def receive_local_frame(connection, *, binding, max_bytes, deadline=None):
    """Verify one sealed FD and close every received handle, including on error."""
    _verify_local_socket(connection)
    import fcntl

    data, ancillary, flags, _ = connection.recvmsg(
        _CONTROL_LIMIT, socket.CMSG_SPACE(4 * 8), getattr(socket, "MSG_CMSG_CLOEXEC", 0)
    )
    received = []
    try:
        for level, kind, values in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                handles = array.array("i")
                handles.frombytes(values[:len(values) - len(values) % handles.itemsize])
                received.extend(handles)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(received) != 1:
            raise ValueError("Local frame must carry exactly one complete descriptor")
        fd = received[0]
        info = os.fstat(fd)
        size = info.st_size
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        if (type(max_bytes) is not int or not 0 < size <= max_bytes
                or not stat.S_ISREG(info.st_mode) or info.st_nlink != 0
                or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals != seals):
            raise ValueError("Local frame is not sealed or exceeds its byte limit")
        return _read_mapped_frame(fd, data, binding=binding, max_bytes=max_bytes, deadline=deadline)
    finally:
        for fd in received:
            os.close(fd)


def _node_address(binding, owner_rank):
    identity = _binding(binding)
    identity["rank"] = owner_rank
    return "\0speco-node-frame-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _remaining_time(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Node reference batch exceeded its deadline")
    return remaining


def serve_node_batches(groups, *, binding, prepare, max_bytes, timeout):
    """One owner prepares each local assignment and sends only its rank's frame."""
    identity = _binding(binding)
    owner = identity["rank"]
    if (not isinstance(groups, dict) or owner not in groups or not groups
            or any(type(rank) is not int or rank < 0 or not entries for rank, entries in groups.items())
            or not callable(prepare) or type(max_bytes) is not int or max_bytes <= 0
            or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Invalid node reference batch plan")
    if not hasattr(os, "memfd_create") or not hasattr(socket, "SOCK_SEQPACKET"):
        raise RuntimeError("Node reference batches require Linux shared frames")
    started = time.perf_counter()
    remaining = set(groups) - {owner}
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
        listener.bind(_node_address(identity, owner))
        listener.listen(max(1, len(remaining)))
        while remaining:
            listener.settimeout(_remaining_time(deadline))
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(_remaining_time(deadline))
                _verify_local_socket(connection)
                request = connection.recv(_CONTROL_LIMIT + 1)
                if len(request) > _CONTROL_LIMIT:
                    raise ValueError("Node reference request is too large")
                requested = _binding(json.loads(request))
                rank = requested["rank"]
                if rank not in remaining or requested != {**identity, "rank": rank}:
                    raise ValueError("Node reference request does not match a pending rank")
                batch, _ = prepare(groups[rank])
                _remaining_time(deadline)
                payload, descriptor = encode_local_frame(batch, binding=requested, max_bytes=max_bytes)
                _remaining_time(deadline)
                fd = create_local_frame(payload, max_bytes=max_bytes)
                try:
                    connection.settimeout(_remaining_time(deadline))
                    send_local_frame(connection, fd, descriptor)
                    _remaining_time(deadline)
                finally:
                    os.close(fd)
                del batch, payload
                remaining.remove(rank)
        _remaining_time(deadline)
        batch, _ = prepare(groups[owner])
        _remaining_time(deadline)
        return batch, time.perf_counter() - started


def receive_node_batch(*, binding, owner_rank, max_bytes, timeout):
    identity = _binding(binding)
    if (type(owner_rank) is not int or owner_rank < 0 or owner_rank == identity["rank"]
            or type(max_bytes) is not int or max_bytes <= 0
            or not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Invalid node reference owner")
    if not hasattr(os, "memfd_create") or not hasattr(socket, "SOCK_SEQPACKET"):
        raise RuntimeError("Node reference batches require Linux shared frames")
    deadline = time.monotonic() + timeout
    address = _node_address(identity, owner_rank)
    while True:
        remaining = _remaining_time(deadline)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            connection.settimeout(remaining)
            connection.connect(address)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            connection.close()
            time.sleep(min(0.05, _remaining_time(deadline)))
        except BaseException:
            connection.close()
            raise
    with connection:
        connection.settimeout(_remaining_time(deadline))
        request = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        if connection.send(request) != len(request):
            raise RuntimeError("Incomplete node reference request")
        started = time.perf_counter()
        connection.settimeout(_remaining_time(deadline))
        batch = receive_local_frame(connection, binding=identity, max_bytes=max_bytes, deadline=deadline)
        _remaining_time(deadline)
        return batch, time.perf_counter() - started
