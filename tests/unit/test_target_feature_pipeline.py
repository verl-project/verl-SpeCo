# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import time

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.feature_store import DraftFeatureSample  # noqa: E402
from verl_speco.trainer.target_feature_pipeline import (  # noqa: E402
    TargetFeatureProducer,
)


def _feature(value: int) -> DraftFeatureSample:
    return DraftFeatureSample(
        input_ids=torch.tensor([value]),
        loss_mask=torch.ones(1),
        hidden_states=torch.zeros(1, 4),
        position_ids=torch.ones(1, dtype=torch.long),
    )


class _Replayer:
    backend = "vllm_file"

    def materialize(self, samples):
        time.sleep(0.01)
        return [_feature(int(samples[0]))]


def test_target_feature_producer_preserves_batch_order_and_prefetches():
    producer = TargetFeatureProducer(
        [[0, 1], [2, 3]],
        _Replayer(),
        rank=0,
        concurrency=2,
        producer_prefetch_depth=2,
        prefetch_depth=2,
        queue_timeout=2,
    )
    try:
        batches = list(producer)
    finally:
        producer.close()

    assert [[int(x.input_ids[0]) for x in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
    ]
    assert producer.metrics()["producer/samples_total"] == 4


class _FailingReplayer:
    backend = "vllm_file"

    def materialize(self, samples):
        raise ValueError("broken replay")


def test_target_feature_producer_propagates_background_failure():
    producer = TargetFeatureProducer(
        [[0]],
        _FailingReplayer(),
        rank=0,
        concurrency=1,
        producer_prefetch_depth=1,
        prefetch_depth=1,
        queue_timeout=2,
    )
    try:
        with pytest.raises(RuntimeError, match="producer failed") as exc_info:
            next(producer)
        assert isinstance(exc_info.value.__cause__, ValueError)
    finally:
        producer.close()
