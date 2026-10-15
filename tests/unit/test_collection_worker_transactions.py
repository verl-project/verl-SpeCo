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
    worker.replica_rank = 0
    worker.worker_incarnation = "worker-0"
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


class _FeatureWriter:
    def __init__(self) -> None:
        self.written = []
        self.flushed = []

    def write_many(self, samples) -> None:
        self.written.extend(samples)

    def flush_on_step(self, global_step, interval_steps) -> None:
        self.flushed.append((global_step, interval_steps))


def _collect_only_worker() -> tuple[SpecoWorker, _FeatureWriter]:
    worker = _worker()
    worker.config.rollout.drafter.training.update(
        {"mode": "collect_only", "feature_store": {"flush_interval_steps": 1}}
    )
    writer = _FeatureWriter()
    worker.feature_writer = writer
    worker.last_global_step = 4
    worker._get_feature_writer = lambda: writer
    return worker, writer


def test_collect_only_rollback_discards_unpublished_feature_samples() -> None:
    worker, writer = _collect_only_worker()
    worker._collection_commit_journals["collection-4"] = {
        **worker._snapshot_collection_buffer(),
        "feature_store_samples": ["sample-a"],
    }

    result = worker.rollback_rollout_features([{"collection_id": "collection-4"}])

    assert result["worker_id"] == "0"
    assert result["worker_incarnation"] == "worker-0"
    assert writer.written == []
    assert writer.flushed == []
    assert "collection-4" not in worker._collection_commit_journals


def test_collect_only_finalize_publishes_staged_feature_samples() -> None:
    worker, writer = _collect_only_worker()
    worker._collection_commit_journals["collection-4"] = {
        **worker._snapshot_collection_buffer(),
        "feature_store_samples": ["sample-a"],
    }

    result = worker.finalize_rollout_features([{"collection_id": "collection-4"}])

    assert result["worker_id"] == "0"
    assert result["worker_incarnation"] == "worker-0"

    assert writer.written == ["sample-a"]
    assert writer.flushed == [(4, 1)]
    assert "collection-4" not in worker._collection_commit_journals

    worker.finalize_rollout_features([{"collection_id": "collection-4"}])
    assert writer.written == ["sample-a"]
    assert writer.flushed == [(4, 1)]


@pytest.mark.parametrize("finalize", [False, True])
def test_collect_only_commit_defers_writes_until_finalize(finalize) -> None:
    from omegaconf import OmegaConf

    worker, writer = _collect_only_worker()
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "EAGLE3",
                    "training": {
                        "mode": "collect_only",
                        "feature_store": {"flush_interval_steps": 1},
                    },
                }
            }
        }
    )
    worker._staged_rollout_features["collection-4"] = {
        "samples": [
            {
                "input_ids": torch.arange(4).unsqueeze(0),
                "prompts": torch.tensor([[0, 1]]),
                "responses": torch.tensor([[2, 3]]),
                "hidden_states": torch.zeros(1, 4, 8),
            }
        ]
    }
    requests = [{"collection_id": "collection-4"}]

    result = asyncio.run(worker.commit_rollout_features(requests))

    assert result["accepted_samples"] == 1
    assert writer.written == []
    assert writer.flushed == []
    assert (
        len(worker._collection_commit_journals["collection-4"]["feature_store_samples"])
        == 1
    )

    if finalize:
        worker.finalize_rollout_features(requests)
        assert len(writer.written) == 1
        assert writer.flushed == [(4, 1)]
    else:
        worker.rollback_rollout_features(requests)
        assert writer.written == []
        assert writer.flushed == []
    assert "collection-4" not in worker._collection_commit_journals


def test_collect_only_failed_finalize_preserves_journal(monkeypatch) -> None:
    worker, writer = _collect_only_worker()
    worker._collection_commit_journals["collection-4"] = {
        "feature_store_samples": ["sample-a"]
    }

    def fail_write(_samples):
        raise OSError("feature store unavailable")

    monkeypatch.setattr(writer, "write_many", fail_write)

    with pytest.raises(OSError, match="feature store unavailable"):
        worker.finalize_rollout_features([{"collection_id": "collection-4"}])

    assert "collection-4" in worker._collection_commit_journals
    assert writer.flushed == []
