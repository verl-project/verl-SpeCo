# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory, lossless scratch storage for address-preserving Ascend parking."""

import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import torch

from verl_speco.trainer.ascend_state_parking import AscendStateParking, state_probe


def _write_all(stream, data):
    while data:
        written = stream.write(data)
        if not written:
            raise OSError("Short write in Ascend state scratch file")
        data = data[written:]


def _read_exact(stream, data):
    remaining = data
    while remaining:
        count = stream.readinto(remaining)
        if not count:
            raise EOFError("Truncated Ascend state scratch file")
        remaining = remaining[count:]


def _flush_and_release(stream, offset, length):
    stream.flush()
    os.fsync(stream.fileno())
    advise = getattr(os, "posix_fadvise", None)
    if advise is not None:
        advise(stream.fileno(), offset, length, os.POSIX_FADV_DONTNEED)


class FileBackedAscendStateParking(AscendStateParking):
    """Use anonymous files, a bounded pinned staging buffer and exact byte copies.

    Storage is scratch, not a recovery checkpoint. File descriptors are released
    by the OS on process exit. Every byte of every owned allocation is retained,
    including allocator cache; no state is inferred disposable from tensor names.
    """

    def __init__(
        self,
        storage_root,
        *,
        chunk_bytes=128 * 1024**2,
        stage_factory=None,
        device_to_host=None,
        host_to_device=None,
        unmap_handle=None,
        map_handle=None,
        flush_and_release=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if chunk_bytes <= 0:
            raise ValueError("State parking chunk size must be positive")
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.chunk_bytes = int(chunk_bytes)
        self.stage_factory = stage_factory or (
            lambda size: torch.empty(
                size, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        )
        self.device_to_host = device_to_host or self._device_to_host
        self.host_to_device = host_to_device or self._host_to_device
        self.unmap_handle = unmap_handle or self._unmap
        self.map_handle = map_handle or self._map
        self.flush_and_release = flush_and_release or _flush_and_release
        self.entries = {}
        self.snapshot_bytes = 0
        self.prepared = False
        atexit.register(self.close)

    @staticmethod
    def _copy(destination, source, count, direction):
        from acl.rt import memcpy

        result = memcpy(destination, count, source, count, direction)
        if result != 0:
            raise RuntimeError(
                f"Ascend state memcpy failed: direction={direction}, code={result}"
            )

    @classmethod
    def _device_to_host(cls, pointer, stage, count):
        cls._copy(stage.data_ptr(), pointer, count, 2)

    @classmethod
    def _host_to_device(cls, pointer, stage, count):
        cls._copy(pointer, stage.data_ptr(), count, 1)

    @staticmethod
    def _unmap(handle):
        from vllm_ascend.device_allocator.camem import unmap_and_release

        unmap_and_release(handle)

    @staticmethod
    def _map(handle):
        from vllm_ascend.device_allocator.camem import create_and_map

        create_and_map(handle)

    def host_snapshot_bytes(self, managed_bytes):
        return min(self.chunk_bytes, managed_bytes)

    def storage_budget(self, managed_bytes):
        return {
            "storage": str(self.storage_root),
            "required": managed_bytes,
            "available": shutil.disk_usage(self.storage_root).free,
        }

    def close(self):
        for entry in self.entries.values():
            entry["file"].close()
        self.entries.clear()

    def prepare_park(self, engine):
        if self.state != "active":
            raise RuntimeError("Resident state is not active")
        try:
            self.synchronize()
            size = self.check_coverage(engine)
            self.probe = state_probe(engine)
            if self.entries:
                raise RuntimeError("Previous scratch snapshot has not been released")
            stage = self.stage_factory(self.host_snapshot_bytes(size))
            for pointer, data in self.allocator.pointer_to_data.items():
                count = data.handle[1]
                stream = tempfile.TemporaryFile(dir=self.storage_root, buffering=0)  # noqa: SIM115 -- kept open across park/wake
                entry = {"file": stream, "bytes": count, "sha256": None}
                self.entries[pointer] = entry
                digest = hashlib.sha256()
                for offset in range(0, count, self.chunk_bytes):
                    length = min(self.chunk_bytes, count - offset)
                    self.device_to_host(pointer + offset, stage, length)
                    view = memoryview(stage[:length].numpy())
                    digest.update(view)
                    _write_all(stream, view)
                    self.flush_and_release(stream, offset, length)
                entry["sha256"] = digest.hexdigest()
            self.snapshot_bytes = size
            self.prepared = True
            return size
        except BaseException:
            self.close()
            self.state = "failed"
            raise

    def abort_prepared(self):
        self.close()
        self.prepared = False

    def commit_park(self):
        if self.state != "active" or not self.prepared:
            raise RuntimeError("State files must be complete before unmapping")
        try:
            for data in self.allocator.pointer_to_data.values():
                self.unmap_handle(data.handle)
            self.prepared = False
            self.state = "parked"
            return self.snapshot_bytes
        except BaseException:
            self.state = "failed"
            raise

    def park(self, engine):
        self.prepare_park(engine)
        return self.commit_park()

    def wake(self, engine):
        if self.state != "parked":
            raise RuntimeError("Resident state is not parked")
        try:
            stage = self.stage_factory(self.host_snapshot_bytes(self.snapshot_bytes))
            if set(self.entries) != set(self.allocator.pointer_to_data):
                raise RuntimeError("Scratch allocation identity changed while parked")
            for pointer, data in self.allocator.pointer_to_data.items():
                entry = self.entries[pointer]
                stream, count = entry["file"], data.handle[1]
                if (
                    count != entry["bytes"]
                    or os.fstat(stream.fileno()).st_size != count
                ):
                    raise RuntimeError("Scratch allocation length differs")
                stream.seek(0)
                self.map_handle(data.handle)
                digest = hashlib.sha256()
                for offset in range(0, count, self.chunk_bytes):
                    length = min(self.chunk_bytes, count - offset)
                    view = memoryview(stage[:length].numpy())
                    _read_exact(stream, view)
                    digest.update(view)
                    self.host_to_device(pointer + offset, stage, length)
                    advise = getattr(os, "posix_fadvise", None)
                    if advise is not None:
                        advise(stream.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError("Scratch state checksum differs; rejecting wake")
            self.synchronize()
            if state_probe(engine) != self.probe:
                raise RuntimeError("Resident state changed across file parking")
            self.close()
            self.state = "active"
            return self.probe
        except BaseException:
            self.state = "failed"
            raise
