# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from verl_speco.trainer.scheduler import (
    ConservativeTrainingDataStatusPolicy,
    TrainingDataStatus,
)


def _status(**overrides) -> TrainingDataStatus:
    values = {
        "current_step": 4,
        "current_step_samples": 8,
        "buffer_samples": 12,
        "trainable_samples": 8,
        "trainable_batches": 4,
        "batch_size_per_gpu": 2,
        "partial_batch_available": True,
        "oldest_sample_step": 2,
        "newest_sample_step": 4,
        "same_step_data_required": False,
        "target_version": 4,
        "worker_id": "0",
        "worker_incarnation": "worker-0",
        "buffer_version": 7,
    }
    values.update(overrides)
    return TrainingDataStatus(**values)


def test_status_policy_aggregates_distributed_capacity_conservatively() -> None:
    result = ConservativeTrainingDataStatusPolicy().aggregate(
        [
            _status(trainable_batches=4, oldest_sample_step=2),
            _status(
                worker_id="1",
                worker_incarnation="worker-1",
                buffer_version=9,
                trainable_batches=2,
                batch_size_per_gpu=4,
                partial_batch_available=False,
                oldest_sample_step=3,
                newest_sample_step=5,
            ),
        ],
        global_step=5,
    )

    assert result is not None
    assert result.current_step == 5
    assert result.trainable_batches == 2
    assert result.batch_size_per_gpu == 4
    assert not result.partial_batch_available
    assert result.oldest_sample_step == 2
    assert result.newest_sample_step == 5
    assert not result.data_version_consistent
    assert result.worker_snapshots == {
        "0": {
            "buffer_version": 7,
            "data_version": 4,
            "worker_incarnation": "worker-0",
            "trainable_samples": 8,
        },
        "1": {
            "buffer_version": 9,
            "data_version": 5,
            "worker_incarnation": "worker-1",
            "trainable_samples": 8,
        },
    }


def test_status_policy_marks_mismatched_target_versions_unknown() -> None:
    result = ConservativeTrainingDataStatusPolicy().aggregate(
        [_status(target_version=3), _status(target_version=4)], global_step=4
    )

    assert result is not None
    assert result.target_version is None


def test_status_policy_returns_none_without_available_workers() -> None:
    assert (
        ConservativeTrainingDataStatusPolicy().aggregate([], global_step=4) is None
    )
