# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded producer/prefetch pipeline for standalone target features."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from verl_speco.trainer.feature_store import DraftFeatureSample

logger = logging.getLogger(__name__)


@dataclass
class _PipelineFailure:
    error: BaseException


_END = object()


class TargetFeatureProducer:
    """Materialize future batches while the current FSDP step is running.

    The coordinator owns the source iterator and puts complete batches into a
    bounded ready queue.  Sample requests inside each batch are concurrent.
    Consequently a training rank never observes a partially materialized batch.
    """

    def __init__(
        self,
        source: Iterable[list[Any]],
        replayer: Any,
        *,
        rank: int,
        concurrency: int,
        producer_prefetch_depth: int,
        prefetch_depth: int,
        queue_timeout: float,
    ):
        self.source = iter(source)
        self.replayer = replayer
        self.rank = int(rank)
        self.concurrency = max(int(concurrency), 1)
        self.producer_prefetch_depth = max(int(producer_prefetch_depth), 1)
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.queue_timeout = max(float(queue_timeout), 1.0)
        self._ready: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_depth)
        self._stop = threading.Event()
        self._request_executor = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix=f"speco-request-r{self.rank}",
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"speco-target-producer-r{self.rank}",
            daemon=True,
        )
        self.produced_batches = 0
        self.produced_samples = 0
        self.producer_seconds = 0.0
        self.queue_wait_seconds = 0.0
        self.consumer_wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.failed_batches = 0
        self._thread.start()
        logger.info(
            "[target producer rank=%s] started request_concurrency=%s "
            "producer_prefetch_depth=%s prefetch_depth=%s",
            self.rank,
            self.concurrency,
            self.producer_prefetch_depth,
            self.prefetch_depth,
        )

    def _run(self) -> None:
        try:
            pending: deque[tuple[float, list[Future[Any]]]] = deque()

            def submit_next() -> bool:
                if self._stop.is_set():
                    return False
                try:
                    samples = next(self.source)
                except StopIteration:
                    return False
                started = time.perf_counter()
                futures = [
                    self._request_executor.submit(self.replayer.materialize, [sample])
                    for sample in samples
                ]
                pending.append((started, futures))
                return True

            for _ in range(self.producer_prefetch_depth):
                if not submit_next():
                    break

            while pending and not self._stop.is_set():
                started, request_futures = pending.popleft()
                produced = [future.result() for future in request_futures]
                self.producer_seconds += time.perf_counter() - started
                submit_next()
                transfer_started = time.perf_counter()
                batch = [item for group in produced for item in group]
                self.transfer_seconds += time.perf_counter() - transfer_started
                self.produced_batches += 1
                self.produced_samples += len(batch)
                self._put(batch)
            self._put(_END)
        except BaseException as exc:  # noqa: BLE001
            self.failed_batches += 1
            self._put(_PipelineFailure(exc))

    def _put(self, value: Any) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._ready.put(value, timeout=0.2)
                self.queue_wait_seconds += time.perf_counter() - started
                return
            except queue.Full:
                continue

    def __iter__(self) -> Iterator[list[DraftFeatureSample]]:
        return self

    def __next__(self) -> list[DraftFeatureSample]:
        started = time.perf_counter()
        try:
            value = self._ready.get(timeout=self.queue_timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                "Timed out waiting for target-feature producer; inspect the vLLM "
                "logs for a stalled request or missing hidden-state file"
            ) from exc
        self.consumer_wait_seconds += time.perf_counter() - started
        if value is _END:
            raise StopIteration
        if isinstance(value, _PipelineFailure):
            raise RuntimeError("Target-feature producer failed") from value.error
        return value

    def metrics(self) -> dict[str, float]:
        return {
            "producer/batches_total": float(self.produced_batches),
            "producer/samples_total": float(self.produced_samples),
            "producer/materialize_time_total": float(self.producer_seconds),
            "producer/transfer_time_total": float(self.transfer_seconds),
            "producer/queue_block_time_total": float(self.queue_wait_seconds),
            "producer/consumer_wait_time_total": float(self.consumer_wait_seconds),
            "producer/ready_queue_size": float(self._ready.qsize()),
            "producer/ready_queue_capacity": float(self.prefetch_depth),
            "producer/failed_batches_total": float(self.failed_batches),
        }

    def close(self) -> None:
        self._stop.set()
        self._request_executor.shutdown(wait=True, cancel_futures=True)
        self._thread.join(timeout=5.0)
