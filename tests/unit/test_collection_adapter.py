# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import pytest

from verl_speco.trainer.scheduler import DrafterCollectionSource, DrafterScheduler


def test_sglang_adapter_buckets_by_replica_and_pads_dispatch_mesh() -> None:
    samples = [
        {"replica_rank": 1, "value": "b"},
        {"replica_rank": 0, "value": "a"},
    ]

    payload = DrafterScheduler().prepare_collection_payload(
        source=DrafterCollectionSource.SGLANG,
        samples=samples,
        owner_count=2,
        dispatch_bucket_count=3,
        raw_samples=4,
    )

    assert payload.buckets == [[samples[1]], [samples[0]], []]
    assert payload.collected_samples == 2
    assert payload.raw_samples == 4


def test_oldlogprob_adapter_uses_explicit_owners_and_preserves_samples() -> None:
    samples = [{"value": "a"}, {"value": "b"}, {"value": "c"}]

    payload = DrafterScheduler().prepare_collection_payload(
        source=DrafterCollectionSource.OLD_LOGPROB,
        samples=samples,
        owners=[1, 0, 1],
        owner_count=2,
        dispatch_bucket_count=2,
        raw_samples=5,
    )

    assert payload.buckets == [[samples[1]], [samples[0], samples[2]]]
    assert payload.collected_samples == 3
    assert payload.raw_samples == 5


def test_collection_adapter_rejects_out_of_range_owner() -> None:
    with pytest.raises(ValueError, match="outside"):
        DrafterScheduler().prepare_collection_payload(
            source=DrafterCollectionSource.SGLANG,
            samples=[{"replica_rank": 2}],
            owner_count=2,
            dispatch_bucket_count=None,
            raw_samples=1,
        )


def test_sglang_adapter_requires_replica_rank() -> None:
    with pytest.raises(ValueError, match="missing replica_rank"):
        DrafterScheduler().prepare_collection_payload(
            source=DrafterCollectionSource.SGLANG,
            samples=[{"value": "a"}],
            owner_count=1,
            dispatch_bucket_count=None,
            raw_samples=1,
        )


def test_collection_adapter_keeps_owner_buckets_when_dispatch_is_smaller() -> None:
    payload = DrafterScheduler().prepare_collection_payload(
        source=DrafterCollectionSource.SGLANG,
        samples=[{"replica_rank": 1}],
        owner_count=2,
        dispatch_bucket_count=1,
        raw_samples=1,
    )

    assert len(payload.buckets) == 2
    assert payload.buckets[1][0]["replica_rank"] == 1
