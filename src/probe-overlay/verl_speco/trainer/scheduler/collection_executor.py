# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Infrastructure boundary for collecting drafter training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPayload,
    CollectionWorkerResult,
)


class DrafterCollectionExecutor(Protocol):
    def set_global_step(self, global_step: object) -> None: ...

    def stage(self, payload: CollectionPayload) -> list[CollectionWorkerResult]: ...

    def commit(self, payload: CollectionPayload) -> list[CollectionWorkerResult]: ...

    def abort(self, payload: CollectionPayload) -> None: ...

    def rollback(self, payload: CollectionPayload) -> None: ...

    def finalize(self, payload: CollectionPayload) -> list[CollectionWorkerResult]: ...


@dataclass(frozen=True)
class CallbackDrafterCollectionExecutor:
    """Adapt existing WorkerGroup callbacks to the collection interface."""

    set_step: Callable[[object], Any]
    stage_submit: Callable[[list[list[dict]]], Any]
    commit_submit: Callable[[list[list[dict]]], Any]
    abort_submit: Callable[[list[list[dict]]], Any]
    rollback_submit: Callable[[list[list[dict]]], Any]
    finalize_submit: Callable[[list[list[dict]]], Any]
    resolve: Callable[[Any], Any]

    def set_global_step(self, global_step: object) -> None:
        self.resolve(self.set_step(global_step))

    @staticmethod
    def _control_buckets(
        payload: CollectionPayload, *, include_samples: bool
    ) -> list[list[dict]]:
        return [
            [
                {
                    "collection_id": payload.collection_id,
                    "samples": bucket if include_samples else None,
                }
            ]
            for bucket in payload.buckets
        ]

    @staticmethod
    def _normalize(results: Any) -> list[CollectionWorkerResult]:
        results = results if isinstance(results, list) else [results]
        return [
            result
            if isinstance(result, CollectionWorkerResult)
            else CollectionWorkerResult.from_mapping(result)
            for result in results
            if isinstance(result, (CollectionWorkerResult, dict))
        ]

    def stage(self, payload: CollectionPayload) -> list[CollectionWorkerResult]:
        requests = self._control_buckets(payload, include_samples=True)
        return self._normalize(self.resolve(self.stage_submit(requests)) or [])

    def commit(self, payload: CollectionPayload) -> list[CollectionWorkerResult]:
        requests = self._control_buckets(payload, include_samples=False)
        return self._normalize(self.resolve(self.commit_submit(requests)) or [])

    def abort(self, payload: CollectionPayload) -> None:
        requests = self._control_buckets(payload, include_samples=False)
        self.resolve(self.abort_submit(requests))

    def rollback(self, payload: CollectionPayload) -> None:
        requests = self._control_buckets(payload, include_samples=False)
        self.resolve(self.rollback_submit(requests))

    def finalize(self, payload: CollectionPayload) -> list[CollectionWorkerResult]:
        requests = self._control_buckets(payload, include_samples=False)
        return self._normalize(self.resolve(self.finalize_submit(requests)) or [])
