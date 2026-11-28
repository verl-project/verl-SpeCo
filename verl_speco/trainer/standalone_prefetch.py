# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded standalone prefetch, rank coordination and FIFO memory pinning."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.config import validate_timeout
from verl_speco.trainer.standalone_data import StandaloneLocalBatch


def _prepared_host_bytes(batch):
    """Bound original tensor storages plus the copies made by the pin thread."""
    storages, pin_bytes = {}, 0
    for value in batch.values():
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
            pin_bytes += max(storage.nbytes(), value.numel() * value.element_size())
    return sum(storages.values()) + pin_bytes


def _reference_device_transfer_bytes(batch):
    """Conservatively account every tensor copied by the device staging path."""
    return sum(
        max(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        for value in batch.values()
        if torch.is_tensor(value)
    )


def _host_budget_depth(depth, max_host_bytes, max_batch_host_bytes):
    if (
        type(max_host_bytes) is not int
        or max_host_bytes < 0
        or type(max_batch_host_bytes) is not int
        or max_batch_host_bytes < 0
        or bool(max_host_bytes) != bool(max_batch_host_bytes)
    ):
        raise ValueError(
            "Prefetch host byte limits must both be non-negative integers and enabled together"
        )
    if max_host_bytes:
        if max_host_bytes < max_batch_host_bytes:
            raise ValueError("Prefetch host bytes must cover one maximum-sized batch")
        depth = min(depth, max_host_bytes // max_batch_host_bytes)
    return depth


class _TrackedPrefetchExecutor(ThreadPoolExecutor):
    """Keep in-flight reads visible after the coordinator has stopped."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracked, self._tracked_lock = set(), threading.Lock()

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        with self._tracked_lock:
            self._tracked.add(future)
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future):
        with self._tracked_lock:
            self._tracked.discard(future)

    def pending(self):
        with self._tracked_lock:
            return {future for future in self._tracked if not future.done()}


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
        max_host_bytes: int = 0,
        max_batch_host_bytes: int = 0,
    ):
        depth = _host_budget_depth(depth, max_host_bytes, max_batch_host_bytes)
        if max_host_bytes and (
            not pin_memory or getattr(loader, "cleanup_after_load", False)
        ):
            raise ValueError(
                "Prefetch host limits require deferred-cleanup reference batches"
            )
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        if prepare_workers < 0:
            raise ValueError("prepare_workers must be non-negative")
        validate_timeout(close_timeout, "Prefetch close timeout")
        if pin_memory and getattr(loader, "prepare_batch", None) is None:
            raise ValueError("Independent pinning requires file batch preparation")
        self.loader = loader
        self.depth = depth
        self._max_batch_host_bytes = max_batch_host_bytes
        self.close_timeout = close_timeout
        self.stop = loader.stop_event = threading.Event()
        self._condition = threading.Condition()
        self._batches: deque[StandaloneLocalBatch] = deque()
        self._pin_batches: deque[StandaloneLocalBatch] = deque()
        self._reserved = 0
        self._source_done = False
        self._pin_device = pin_device
        self._original_prepare: Any | None = None
        self._executor: _TrackedPrefetchExecutor | None = None
        self._pin_thread: threading.Thread | None = None
        if pin_memory:
            assert loader.prepare_batch is not None
            self._original_prepare = loader.prepare_batch
            if prepare_workers:
                self._executor = _TrackedPrefetchExecutor(
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

    def _open_source(self, factory):
        source, error = None, None
        try:
            source = factory()
        except BaseException as exc:
            error = exc
        if self._sync([int(error is not None)])[0]:
            raise RuntimeError(
                "Consumer prefetch source initialization failed on at least one rank"
            ) from error
        return source

    def _run(self) -> None:
        source = None
        try:
            source = self._open_source(lambda: iter(self.loader))
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
                    if self._max_batch_host_bytes:
                        if batch.prepared_batch is None:
                            raise RuntimeError(
                                "Reference prefetch returned no prepared batch"
                            )
                        if (
                            _prepared_host_bytes(batch.prepared_batch)
                            > self._max_batch_host_bytes
                        ):
                            raise ValueError(
                                "Prepared reference batch exceeds consumer_prefetch_batch_bytes"
                            )
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
                        lambda: self.stop.is_set()
                        or self._pin_batches
                        or self._source_done
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

    def _thread_work_running(self) -> bool:
        return (
            self._thread.is_alive()
            or (self._pin_thread is not None and self._pin_thread.is_alive())
            or (self._executor is not None and bool(self._executor.pending()))
        )

    @property
    def has_running_work(self) -> bool:
        return self._thread_work_running()

    def _begin_close(self, timeout: float | None) -> float:
        timeout = self.close_timeout if timeout is None else timeout
        validate_timeout(timeout, "Prefetch close timeout")
        deadline = time.monotonic() + timeout
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        return deadline

    def close(self, *, timeout: float | None = None) -> None:
        deadline = self._begin_close(timeout)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._pin_thread is not None:
            self._pin_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            wait(
                self._executor.pending(), timeout=max(0.0, deadline - time.monotonic())
            )
        if self._thread_work_running():
            raise RuntimeError(
                "TQ prefetch work did not stop before its shutdown deadline"
            )
        if self._original_prepare is not None:
            self.loader.prepare_batch = self._original_prepare
        self._batches.clear()
        self._pin_batches.clear()
        self._reserved = 0
