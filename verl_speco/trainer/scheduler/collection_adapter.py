# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Source adapters for building per-worker drafter collection payloads."""

from __future__ import annotations

from typing import Protocol, Sequence

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPayload,
    DrafterCollectionSource,
)


class DrafterCollectionAdapter(Protocol):
    """Convert source-specific samples into the common collection payload."""

    source: DrafterCollectionSource

    def prepare_payload(
        self,
        samples: list[dict],
        *,
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners: Sequence[int] | None = None,
    ) -> CollectionPayload: ...


def _build_payload(
    *,
    source: DrafterCollectionSource,
    samples: list[dict],
    owners: Sequence[int],
    owner_count: int,
    dispatch_bucket_count: int | None,
    raw_samples: int,
    collection_id: str,
) -> CollectionPayload:
    if owner_count <= 0:
        raise ValueError(f"Collection owner_count must be positive, got {owner_count}")
    if len(samples) != len(owners):
        raise ValueError(
            "Collection samples and owners must have the same length: "
            f"{len(samples)} != {len(owners)}"
        )

    buckets: list[list[dict]] = [[] for _ in range(owner_count)]
    for sample, owner in zip(samples, owners, strict=True):
        owner = int(owner)
        if owner < 0 or owner >= owner_count:
            raise ValueError(f"Collection owner {owner} is outside [0, {owner_count})")
        buckets[owner].append(sample)

    if dispatch_bucket_count is not None:
        buckets.extend([[] for _ in range(max(dispatch_bucket_count - owner_count, 0))])

    return CollectionPayload(
        source=source,
        buckets=buckets,
        collected_samples=len(samples),
        raw_samples=raw_samples,
        collection_id=collection_id,
    )


class SGLangCollectionAdapter:
    source = DrafterCollectionSource.SGLANG

    def prepare_payload(
        self,
        samples: list[dict],
        *,
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners: Sequence[int] | None = None,
    ) -> CollectionPayload:
        if owners is not None:
            raise ValueError("SGLang collection owners are read from replica_rank")
        replica_owners = []
        for sample in samples:
            replica_rank = sample.get("replica_rank")
            if replica_rank is None:
                raise ValueError(
                    "drafter_sample is missing replica_rank for owner routing"
                )
            replica_owners.append(int(replica_rank))
        return _build_payload(
            source=self.source,
            samples=samples,
            owners=replica_owners,
            owner_count=owner_count,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=raw_samples,
            collection_id=collection_id,
        )


class OldLogProbCollectionAdapter:
    source = DrafterCollectionSource.OLD_LOGPROB

    def prepare_payload(
        self,
        samples: list[dict],
        *,
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners: Sequence[int] | None = None,
    ) -> CollectionPayload:
        if owners is None:
            raise ValueError("Old-log-prob collection requires explicit owners")
        return _build_payload(
            source=self.source,
            samples=samples,
            owners=owners,
            owner_count=owner_count,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=raw_samples,
            collection_id=collection_id,
        )
