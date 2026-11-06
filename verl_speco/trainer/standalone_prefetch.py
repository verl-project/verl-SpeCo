# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded standalone prefetch, rank coordination and FIFO memory pinning."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.trainer.standalone_data import StandaloneLocalBatch


class StandalonePrefetchIterator:
    """One ordered coordinator per rank, bounded payloads, separate CPU collectives.

    Each fetch occupies a slot before next(source). The main thread taking a
    batch frees a slot. EOS/errors live outside the payload buffer so shutdown
    does not depend on a consumer draining a full queue.
    """

    def __init__(
        self,
        loader: Any,
        depth: int,
        *,
        close_timeout: float = 180.0,
        pin_memory: bool = False,
        prepare_workers: int = 4,
        pin_device: torch.device | None = None,
    ):
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        if prepare_workers < 0:
            raise ValueError("prepare_workers must be non-negative")
        if pin_memory and getattr(loader, "prepare_batch", None) is None:
            raise ValueError("Independent pinning requires file batch preparation")
        self.loader = loader
        self.depth = depth
        self.close_timeout = close_timeout
        self.stop = loader.stop_event = threading.Event()
        self._condition = threading.Condition()
        self._batches: deque[StandaloneLocalBatch] = deque()
        self._pin_batches: deque[StandaloneLocalBatch] = deque()
        self._reserved = 0
        self._source_done = False
        self._pin_device = pin_device
        self._original_prepare: Any | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._pin_thread: threading.Thread | None = None
        if pin_memory:
            assert loader.prepare_batch is not None
            self._original_prepare = loader.prepare_batch
            if prepare_workers:
                self._executor = ThreadPoolExecutor(
                    max_workers=prepare_workers,
                    thread_name_prefix=f"file-prepare-{loader.rank}",
                )
            loader.prepare_batch = partial(
                loader.prepare_batch,
                pin_memory=False,
                executor=self._executor,
            )
            self._pin_thread = threading.Thread(
                target=self._pin_loop,
                name=f"file-pin-{loader.rank}",
                daemon=True,
            )
        self._done = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"{'file' if getattr(loader, 'prepare_batch', None) is not None else 'tq'}-prefetch-{loader.rank}",
            daemon=True,
        )
        if self._pin_thread is not None:
            self._pin_thread.start()
        self._thread.start()

    def _sync(self, values: list[int]) -> list[int]:
        if self.loader.world_size > 1:
            if self.loader.process_group is None:
                raise RuntimeError(
                    "Distributed prefetch requires its own Gloo process group"
                )
            state = torch.tensor(values, dtype=torch.int32)
            dist.all_reduce(
                state, op=dist.ReduceOp.MAX, group=self.loader.process_group
            )
            return state.tolist()
        return values

    def _run(self) -> None:
        source = iter(self.loader)
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stop.is_set() or self._reserved < self.depth
                    )
                    if not self.stop.is_set():
                        self._reserved += 1
                if self._sync([int(self.stop.is_set())])[0]:
                    return
                batch = None
                error = None
                exhausted = False
                try:
                    batch = next(source)
                except StopIteration:
                    exhausted = True
                except BaseException as exc:  # noqa: BLE001 - deliver thread failures to every rank
                    error = exc
                failed, ended = self._sync([int(error is not None), int(exhausted)])
                if failed:
                    raise RuntimeError(
                        f"Consumer prefetch failed on rank={self.loader.rank}"
                    ) from error
                if ended:
                    return
                if batch is None:
                    raise RuntimeError(
                        "Consumer prefetch returned no batch without EOS"
                    )
                with self._condition:
                    if self._pin_thread is None:
                        self._batches.append(batch)
                    else:
                        batch.delivery_times["_pin_queued_at"] = time.monotonic()
                        self._pin_batches.append(batch)
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the main thread
            with self._condition:
                self._error = exc
        finally:
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            finally:
                with self._condition:
                    self._source_done = True
                    if self._pin_thread is None:
                        self._done = True
                    self._condition.notify_all()

    def _pin_loop(self) -> None:
        """Pin FIFO batches while the coordinator prepares the next assignment.

        This thread never performs distributed collectives. Its errors surface
        through next(), then the existing main-thread rank agreement aborts all
        ranks before a mismatched training step can start.
        """
        try:
            # Match torch DataLoader's pin thread, avoiding parallel CPU copies.
            torch.set_num_threads(1)
            if self._pin_device is not None and self._pin_device.type != "cpu":
                getattr(torch, self._pin_device.type).set_device(self._pin_device)
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self.stop.is_set() or self._pin_batches or self._source_done
                        )
                    )
                    if self.stop.is_set() or not self._pin_batches:
                        return
                    batch = self._pin_batches.popleft()
                started = time.perf_counter()
                delivery_times = dict(batch.delivery_times)
                queued_at = delivery_times.pop("_pin_queued_at", None)
                if queued_at is not None:
                    delivery_times["loader_pin_queue_time"] = (
                        time.monotonic() - queued_at
                    )
                if batch.prepared_batch is None:
                    raise RuntimeError("File pinning received no prepared batch")
                pinned = {
                    key: value.pin_memory() if torch.is_tensor(value) else value
                    for key, value in batch.prepared_batch.items()
                }
                delivery_times["loader_pin_time"] = time.perf_counter() - started
                delivery_times["_ready_at"] = time.monotonic()
                batch = replace(
                    batch,
                    delivery_times=delivery_times,
                    prepared_batch=pinned,
                    batch_prepare_seconds=batch.batch_prepare_seconds
                    + time.perf_counter()
                    - started,
                )
                with self._condition:
                    self._batches.append(batch)
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surface pin failures in the main thread
            with self._condition:
                self._error = exc
                self.stop.set()
                self._condition.notify_all()
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> StandaloneLocalBatch:
        with self._condition:
            self._condition.wait_for(lambda: bool(self._batches) or self._done)
            if self._batches:
                batch = self._batches.popleft()
                if self._pin_thread is not None:
                    ready_at = batch.delivery_times.pop("_ready_at", None)
                    if ready_at is not None:
                        batch.delivery_times["loader_ready_queue_time"] = (
                            time.monotonic() - ready_at
                        )
                self._reserved -= 1
                self._condition.notify_all()
                return batch
            if self._error is not None:
                raise self._error
            raise StopIteration

    def close(self) -> None:
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=self.close_timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                "Consumer prefetch thread did not stop before its shutdown deadline"
            )
        if self._pin_thread is not None:
            self._pin_thread.join(timeout=self.close_timeout)
            if self._pin_thread.is_alive():
                raise RuntimeError(
                    "Consumer pin thread did not stop before its shutdown deadline"
                )
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._original_prepare is not None:
            self.loader.prepare_batch = self._original_prepare
        self._batches.clear()
        self._pin_batches.clear()
        self._reserved = 0
