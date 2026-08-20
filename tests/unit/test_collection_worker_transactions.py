# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from verl_speco.trainer.data_buffer import DataBuffer
from verl_speco.workers.speco_worker import SpecoWorker


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
            drafter=SimpleNamespace(
                training={"collection_stage_ttl_sec": 1.0}
            )
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
