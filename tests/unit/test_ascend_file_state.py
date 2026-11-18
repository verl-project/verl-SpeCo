# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real CPU byte transfers exercise lossless scratch IO without an NPU."""

import ctypes
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.ascend_file_state import FileBackedAscendStateParking
from verl_speco.trainer.ascend_state_parking import state_probe, trainer_tensors
from verl_speco.trainer.vllm_ascend_lifecycle import AscendTrainingLifecycle


def sample_engine():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    runtime = SimpleNamespace(
        module=model, optimizer=optimizer, lr_scheduler=scheduler, global_steps=1
    )
    return SimpleNamespace(model_engine=runtime, teacher=torch.nn.Linear(4, 3))


class Allocator:
    def __init__(self, engine):
        self.pointer_to_data = {}
        self.allocator_and_pools = {}
        for _, value in trainer_tensors(engine):
            storage = value.untyped_storage()
            self.pointer_to_data[storage.data_ptr()] = SimpleNamespace(
                tag="speco_resident_training",
                handle=(0, storage.nbytes(), storage.data_ptr(), 0),
            )

    def get_current_usage(self):
        return 0

    @contextmanager
    def use_memory_pool(self, tag):
        self.allocator_and_pools[tag] = (object(), None)
        yield

    def use_allocation_tag(self, tag):
        return nullcontext()


def parking(engine, tmp_path, *, flush=None, chunk_bytes=11):
    stages, unmaps = [], []

    def stage(size):
        stages.append(size)
        return torch.empty(size, dtype=torch.uint8)

    def unmap(handle):
        unmaps.append(handle[2])
        ctypes.memset(handle[2], 0, handle[1])

    instance = FileBackedAscendStateParking(
        tmp_path,
        allocator=Allocator(engine),
        pool_context=lambda _: nullcontext(),
        synchronize=lambda: None,
        cache_backend=SimpleNamespace(),
        chunk_bytes=chunk_bytes,
        stage_factory=stage,
        device_to_host=lambda pointer, buffer, n: ctypes.memmove(
            buffer.data_ptr(), pointer, n
        ),
        host_to_device=lambda pointer, buffer, n: ctypes.memmove(
            pointer, buffer.data_ptr(), n
        ),
        unmap_handle=unmap,
        map_handle=lambda handle: None,
        flush_and_release=flush,
    )
    with instance.allocation_scope():
        pass
    return instance, stages, unmaps


def test_exact_bytes_addresses_adam_and_scheduler_survive_bounded_file_roundtrip(
    tmp_path,
):
    engine = sample_engine()
    instance, stages, unmaps = parking(engine, tmp_path)
    original = state_probe(engine)
    expected = {
        ptr: ctypes.string_at(ptr, data.handle[1])
        for ptr, data in instance.allocator.pointer_to_data.items()
    }
    count = instance.park(engine)
    assert count == sum(map(len, expected.values()))
    assert instance.state == "parked" and len(unmaps) == len(expected)
    descriptors = [entry["file"] for entry in instance.entries.values()]
    assert instance.host_snapshot_bytes(count) == 11
    assert max(stages) == 11
    instance.wake(engine)
    assert state_probe(engine) == original
    assert all(
        ctypes.string_at(ptr, len(value)) == value for ptr, value in expected.items()
    )
    assert all(stream.closed for stream in descriptors)
    assert instance.state == "active" and not instance.entries
    assert list(tmp_path.iterdir()) == []


def test_io_failure_does_not_unmap_or_change_original_training_state(tmp_path):
    engine = sample_engine()
    before = state_probe(engine)

    def failed_sync(*args):
        raise OSError("scratch disk failure")

    instance, _, unmaps = parking(engine, tmp_path, flush=failed_sync)
    with pytest.raises(OSError, match="scratch disk failure"):
        instance.park(engine)
    assert not unmaps and not instance.entries
    assert state_probe(engine) == before


def test_corruption_rejects_wake_before_any_training_can_resume(tmp_path):
    engine = sample_engine()
    instance, _, _ = parking(engine, tmp_path)
    instance.park(engine)
    entry = next(iter(instance.entries.values()))
    stream = entry["file"]
    stream.seek(0)
    first = stream.read(1)
    stream.seek(0)
    stream.write(bytes([first[0] ^ 1]))
    stream.flush()
    with pytest.raises(RuntimeError, match="checksum"):
        instance.wake(engine)
    assert instance.state == "failed"
    instance.close()


def test_truncation_is_rejected(tmp_path):
    engine = sample_engine()
    instance, _, _ = parking(engine, tmp_path)
    instance.park(engine)
    next(iter(instance.entries.values()))["file"].truncate(0)
    with pytest.raises(RuntimeError, match="length differs"):
        instance.wake(engine)
    instance.close()


def test_lifecycle_accounts_for_stage_ram_and_collective_disk_capacity(tmp_path):
    engine = sample_engine()
    instance, _, _ = parking(engine, tmp_path)
    lifecycle = AscendTrainingLifecycle(
        parking=instance,
        gather=lambda value: [value],
        host_reserve_bytes=0,
        host_available=lambda: 32,
        device_free=lambda: 10**9,
    )
    assert instance.check_coverage(engine) > 32
    lifecycle.park(engine)
    assert lifecycle.last_status["host_snapshot_bytes"] == 11
    assert lifecycle.last_status["scratch_disk_bytes"] > 32
    lifecycle.wake(engine)


def test_peer_file_failure_is_settled_before_any_local_unmap(tmp_path):
    engine = sample_engine()
    instance, _, unmaps = parking(engine, tmp_path)
    calls = 0

    def gather(value):
        nonlocal calls
        calls += 1
        # quiesce, coverage, budget metadata, budget gather, then prepare result
        if calls == 5:
            return [value, "remote IO failure"]
        return [value]

    lifecycle = AscendTrainingLifecycle(
        parking=instance,
        gather=gather,
        host_reserve_bytes=0,
        host_available=lambda: 10**9,
        device_free=lambda: 10**9,
    )
    with pytest.raises(RuntimeError, match="snapshot_write"):
        lifecycle.park(engine)
    assert not unmaps and not instance.entries
    assert lifecycle.failed
