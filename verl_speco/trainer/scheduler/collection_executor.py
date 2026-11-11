# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Infrastructure boundary for collecting drafter training samples."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from verl_speco.integration.phase_timing import phase_span, phase_timing_enabled
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
    phase_timing: bool = False

    def _call(self, phase, submit, request, **labels):
        enabled = self.phase_timing or phase_timing_enabled()
        if not enabled:
            return self.resolve(submit(request))
        # One existing phase record includes both submission and awaited work.
        # Mutate the nested labels while inside the span; phase_span serializes
        # them on exit, including partial timing when an exception propagates.
        stages = {}
        started = time.perf_counter()
        with phase_span(
            f"collection_{phase}", enabled=enabled, stages=stages, **labels
        ):
            try:
                refs = submit(request)
            finally:
                stages["submit_s"] = time.perf_counter() - started
            submitted_at = time.perf_counter()
            try:
                result = self.resolve(refs)
            finally:
                stages["wait_s"] = time.perf_counter() - submitted_at
        return result

    def set_global_step(self, global_step: object) -> None:
        self._call("set_step", self.set_step, global_step, source_step=str(global_step))

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
        return self._normalize(
            self._call(
                "stage",
                self.stage_submit,
                requests,
                collection_id=payload.collection_id,
            )
            or []
        )

    def commit(self, payload: CollectionPayload) -> list[CollectionWorkerResult]:
        requests = self._control_buckets(payload, include_samples=False)
        return self._normalize(
            self._call(
                "commit",
                self.commit_submit,
                requests,
                collection_id=payload.collection_id,
            )
            or []
        )

    def abort(self, payload: CollectionPayload) -> None:
        requests = self._control_buckets(payload, include_samples=False)
        self.resolve(self.abort_submit(requests))

    def rollback(self, payload: CollectionPayload) -> None:
        requests = self._control_buckets(payload, include_samples=False)
        self.resolve(self.rollback_submit(requests))

    def finalize(self, payload: CollectionPayload) -> list[CollectionWorkerResult]:
        requests = self._control_buckets(payload, include_samples=False)
        return self._normalize(
            self._call(
                "finalize",
                self.finalize_submit,
                requests,
                collection_id=payload.collection_id,
            )
            or []
        )
