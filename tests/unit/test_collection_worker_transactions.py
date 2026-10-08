# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.data_buffer import DataBuffer  # noqa: E402
from verl_speco.workers import speco_worker  # noqa: E402
from verl_speco.workers.speco_worker import SpecoWorker  # noqa: E402


def _worker() -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    data_buffer = DataBuffer(max_size=8)
    data_buffer.buffer.append({"value": "before"})
    data_buffer._current_step = 4
    worker.trainer = SimpleNamespace(
        buffer_version=3,
        collected_data=deque([{"value": "before"}], maxlen=8),
        data_buffer=data_buffer,
    )
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"collection_stage_ttl_sec": 1.0})
        )
    )
    worker._staged_rollout_features = {}
    worker._collection_commit_journals = {}
    return worker


def test_collection_buffer_snapshot_restores_data_and_version() -> None:
    worker = _worker()
    snapshot = worker._snapshot_collection_buffer()
    worker.trainer.collected_data.append({"value": "committed"})
    worker.trainer.data_buffer.buffer.append({"value": "committed"})
    worker.trainer.buffer_version = 5

    worker._restore_collection_buffer(snapshot)

    assert list(worker.trainer.collected_data) == [{"value": "before"}]
    assert list(worker.trainer.data_buffer.buffer) == [{"value": "before"}]
    assert worker.trainer.data_buffer.get_current_step() == 4
    assert worker.trainer.buffer_version == 3


def test_expired_collection_stages_are_removed() -> None:
    worker = _worker()
    worker._staged_rollout_features = {
        "expired": {"samples": [], "staged_at": time.monotonic() - 2.0},
        "active": {"samples": [], "staged_at": time.monotonic()},
    }

    removed = worker._cleanup_expired_collection_stages()

    assert removed == 1
    assert set(worker._staged_rollout_features) == {"active"}


def test_data_buffer_reservation_is_versioned_and_consume_once() -> None:
    buffer = DataBuffer(max_size=8)
    buffer.update_rl_step(4)
    step_three = {"value": "old", "target_version": 3}
    step_four_a = {"value": "new-a", "target_version": 4}
    step_four_b = {"value": "new-b", "target_version": 4}
    for sample in (step_three, step_four_a, step_four_b):
        buffer.add_batch(sample)

    reserved = buffer.reserve("plan-4", target_version=4, max_samples=2)

    assert reserved == [step_four_a, step_four_b]
    assert buffer.get_available_data(target_version=4) == []
    assert (
        buffer.get_available_data(target_version=4, reservation_id="plan-4") == reserved
    )
    assert buffer.consume("plan-4", [step_four_a]) == 1
    assert buffer.release_reservation("plan-4") == 1
    assert buffer.get_available_data(target_version=4) == [step_four_b]


def test_collection_ref_resolution_awaits_without_ray_get(monkeypatch) -> None:
    class AwaitableRef:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            async def _resolve():
                return self.value

            return _resolve().__await__()

    monkeypatch.setattr(speco_worker.ray, "ObjectRef", AwaitableRef)
    monkeypatch.setattr(
        speco_worker.ray,
        "get",
        lambda _: pytest.fail("collection ref resolution must not call ray.get"),
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    resolved = asyncio.run(
        speco_worker._resolve_hidden_state_chunks(
            [
                {
                    "ref": AwaitableRef(hidden),
                    "chunk_start": 0,
                    "chunk_length": 2,
                    "chunk_row_indices": [0, 1],
                }
            ],
            expected_rows=2,
        )
    )

    assert torch.equal(resolved, hidden.unsqueeze(0))
