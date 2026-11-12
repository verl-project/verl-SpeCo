# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Scheduler-owned assignments, whole-batch CPU processes, ordered pinning.

Only the dispatcher uses the prefetch Gloo group. Workers do not use Ray,
TQ or scheduler state. The caller controls when hidden references are released.
Mooncake may initialize a device context required by its transport.
Results use torch multiprocessing reducers, then the existing pin thread.
"""

from __future__ import annotations

import os
import logging
import math
import threading
import time
from collections import deque
from dataclasses import replace
from functools import partial
from multiprocessing import TimeoutError as PoolTimeout
from typing import Any

import torch
import torch.multiprocessing as mp

from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.tq_sample_source import (
    TQPrefetchIterator, _host_budget_depth, _prepared_host_bytes, _reference_device_transfer_bytes,
)

_STORE: Any = None
_OPTIONS: dict[str, Any] = {}
_ATTEMPTS = 1
_CLEANUP_AFTER_LOAD = True
_MAX_BATCH_HOST_BYTES = 0
logger = logging.getLogger(__name__)


def _initialize_worker(config, options, attempts, cleanup_after_load=True, max_batch_host_bytes=0):
    global _STORE, _OPTIONS, _ATTEMPTS, _CLEANUP_AFTER_LOAD, _MAX_BATCH_HOST_BYTES
    # This is a separate process: do not change the training process's threads.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)
    from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore

    _STORE = ReferenceFeatureStore(config)
    _STORE.connect()
    _OPTIONS, _ATTEMPTS = dict(options), attempts
    _CLEANUP_AFTER_LOAD = cleanup_after_load
    _MAX_BATCH_HOST_BYTES = max_batch_host_bytes


@torch.no_grad()
def _prepare_batch(entries):
    started = time.perf_counter()
    batch = retry_io(
        partial(_STORE.prepare_batch, entries, pin_memory=False, **_OPTIONS),
        name="Reference batch preparation",
        attempts=_ATTEMPTS,
    )
    if _MAX_BATCH_HOST_BYTES and _prepared_host_bytes(batch) > _MAX_BATCH_HOST_BYTES:
        raise ValueError("Prepared reference batch exceeds consumer_prefetch_batch_bytes")
    # Do not delete inside the preparation retry: a retry must still be able to
    # read all samples. A failed release must not be acknowledged as complete.
    from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples

    if _CLEANUP_AFTER_LOAD:
        remaining = cleanup_standalone_samples(_STORE, entries)
        if remaining:
            raise RuntimeError(f"Failed to release {len(remaining)} prepared references")
    batch["_speco_worker_finished_at"] = time.monotonic()
    return batch, time.perf_counter() - started


def _node_frame_batch(groups, *, binding, owner_rank, max_bytes, timeout):
    from verl_speco.trainer.reference_shared_frame import serve_node_batches, receive_node_batch
    if groups is not None:
        if _CLEANUP_AFTER_LOAD:
            raise RuntimeError("Node reference owner cannot release payloads during preparation")
        if not isinstance(binding, dict) or binding.get("rank") != owner_rank:
            raise ValueError("Node reference batch was not assigned to its owner")
        return serve_node_batches(groups, binding=binding, prepare=_prepare_batch,
                                  max_bytes=max_bytes, timeout=timeout)
    batch, seconds = receive_node_batch(binding=binding, owner_rank=owner_rank, max_bytes=max_bytes, timeout=timeout)
    if _MAX_BATCH_HOST_BYTES and _prepared_host_bytes(batch) > _MAX_BATCH_HOST_BYTES:
        raise ValueError("Received node reference batch exceeds consumer_prefetch_batch_bytes")
    return batch, seconds


class ReferenceBatchPrefetchIterator(TQPrefetchIterator):
    """Dispatch up to depth batches without waiting for previous preparation.

    The collector drains results FIFO, independently of command reception.
    Capacity includes running/queued tasks, pinning and ready batches. Worker
    failures reach the existing main-thread rank readiness agreement.
    """

    def __init__(
        self,
        loader,
        depth,
        *,
        workers=2,
        prepare_options=None,
        close_timeout=180.0,
        pin_device=None,
        defer_cleanup=False,
        max_host_bytes=0,
        max_batch_host_bytes=0,
        node_ranks_per_host=0,
    ):
        depth = _host_budget_depth(depth, max_host_bytes, max_batch_host_bytes)
        if workers < 1 or depth < 1:
            raise ValueError(
                "Reference batch workers and prefetch depth must be positive"
            )
        if loader.prepare_batch is None or getattr(
            loader.store, "uses_transfer_queue", True
        ):
            raise ValueError("Process batch prefetch requires ray_reference input")
        if node_ranks_per_host and (depth != 1 or workers != 1 or not defer_cleanup
                or not max_batch_host_bytes or max_host_bytes < 3 * max_batch_host_bytes
                or loader.node_ranks_per_host != node_ranks_per_host):
            raise ValueError("Node reference owner requires one prefetch batch and worker, delayed cleanup and a 3-batch host budget")
        self.node_ranks_per_host = node_ranks_per_host
        self.node_frame_bytes = max_batch_host_bytes
        self._pool_shutdown_thread = None
        self._pool_shutdown_error = None
        self._pending: deque[Any] = deque()
        self.defer_cleanup = defer_cleanup
        self._assignments_done = False
        # Set defaults before spawn imports torch/numpy in the child. Existing
        # user overrides are preserved; only child torch intra-op is forced to 1.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        self._pool = mp.get_context("spawn").Pool(
            processes=min(workers, depth),
            initializer=_initialize_worker,
            initargs=(
                loader.store.config,
                dict(prepare_options or {}),
                loader.io_retry_max_attempts,
                not defer_cleanup,
                max_batch_host_bytes,
            ),
        )
        try:
            super().__init__(
                loader,
                depth,
                close_timeout=close_timeout,
                pin_memory=True,
                prepare_workers=0,
                pin_device=pin_device,
            )
            self._collector = threading.Thread(
                target=self._collect,
                name=f"file-batch-results-{loader.rank}",
                daemon=True,
            )
            self._collector.start()
            logger.info(
                "Reference batch process prefetch rank=%s workers=%s depth=%s",
                loader.rank,
                min(workers, depth),
                depth,
            )
        except BaseException:
            self._pool.terminate()
            self._pool.join()
            raise

    def _run(self):
        source = None
        try:
            source = self._open_source(self.loader.iter_assignments)
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stop.is_set() or self._reserved < self.depth
                    )
                    if not self.stop.is_set():
                        self._reserved += 1
                if self._sync([int(self.stop.is_set())])[0]:
                    return
                assignment, result, error, ended = None, None, None, False
                try:
                    assignment = next(source)
                    if assignment is None:
                        raise RuntimeError("Missing file batch assignment")
                    assignment.delivery_times["_worker_submitted_at"] = time.monotonic()
                    if self.node_ranks_per_host:
                        owner_rank = self.loader.rank // self.node_ranks_per_host * self.node_ranks_per_host
                        binding = {"run_id": self.loader.store.run_id, "generation": 0,
                                   "batch_id": assignment.plan_id, "rank": self.loader.rank,
                                   "receipt_token": assignment.receipt_token}
                        result = self._pool.apply_async(
                            _node_frame_batch, (assignment.node_file_entries,),
                            {"binding": binding, "owner_rank": owner_rank,
                             "max_bytes": self.node_frame_bytes, "timeout": self.close_timeout},
                        )
                    else:
                        result = self._pool.apply_async(_prepare_batch, (assignment.file_entries,))
                except StopIteration:
                    ended = True
                except BaseException as exc:
                    error = exc
                failed, exhausted = self._sync([int(error is not None), int(ended)])
                if failed:
                    raise RuntimeError(
                        "Failed to receive file batch assignment"
                    ) from error
                if exhausted:
                    return
                if assignment is None:
                    raise RuntimeError("Missing file batch assignment")
                with self._condition:
                    self._pending.append((assignment, result))
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self.stop.set()
                self._condition.notify_all()
        finally:
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            finally:
                with self._condition:
                    self._assignments_done = True
                    self._condition.notify_all()

    def _collect(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stop.is_set()
                        or self._pending
                        or self._assignments_done
                    )
                    if self.stop.is_set() or not self._pending:
                        return
                    assignment, result = self._pending.popleft()
                deadline = time.monotonic() + self.close_timeout
                while not self.stop.is_set():
                    try:
                        prepared, seconds = result.get(timeout=0.1)
                        break
                    except PoolTimeout:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Reference batch worker exceeded preparation deadline"
                            ) from None
                else:
                    return
                received_at = time.monotonic()
                delivery_times = dict(assignment.delivery_times)
                submitted_at = delivery_times.pop("_worker_submitted_at")
                delivery_times["loader_worker_result_wait_time"] = max(
                    0.0, received_at - submitted_at - seconds
                )
                finished_at = prepared.pop("_speco_worker_finished_at", None)
                if finished_at is not None:
                    delivery_times["loader_ipc_collect_time"] = (
                        received_at - finished_at
                    )
                delivery_times["_pin_queued_at"] = time.monotonic()
                with self._condition:
                    self._pin_batches.append(
                        replace(
                            assignment,
                            prepared_batch=prepared,
                            file_entries=None,
                            node_file_entries=None,
                            batch_prepare_seconds=seconds,
                            cleanup_after_load=not self.defer_cleanup,
                            delivery_times=delivery_times,
                        )
                    )
                    self._condition.notify_all()
                # AsyncResult and the collector otherwise retain the previous
                # shared-memory payload after the pin thread has advanced.
                del prepared, result
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self.stop.set()
                self._condition.notify_all()
        finally:
            with self._condition:
                self._source_done = True
                self._condition.notify_all()

    @property
    def has_running_work(self):
        return (super().has_running_work or self._collector.is_alive()
                or (self._pool_shutdown_thread is not None and self._pool_shutdown_thread.is_alive())
                or any(process.is_alive() for process in self._pool._pool))

    def close(self, *, timeout=None):
        timeout = self.close_timeout if timeout is None else timeout
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Prefetch close timeout must be a positive finite number")
        deadline = time.monotonic() + timeout
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        self._collector.join(timeout=max(0.0, deadline - time.monotonic()))
        error = None
        try:
            super().close(timeout=max(1e-9, deadline - time.monotonic()))
        except Exception as exc:
            error = exc
        if self._pool_shutdown_thread is None:
            def shutdown_pool():
                try:
                    self._pool.terminate()
                    self._pool.join()
                except BaseException as exc:
                    self._pool_shutdown_error = exc
            self._pool_shutdown_thread = threading.Thread(target=shutdown_pool, name="reference-pool-shutdown", daemon=True)
            self._pool_shutdown_thread.start()
        self._pool_shutdown_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._pool_shutdown_error is not None:
            raise RuntimeError("Reference worker pool shutdown failed") from self._pool_shutdown_error
        if self.has_running_work:
            raise RuntimeError("Reference batch prefetch did not stop before its shutdown deadline") from error
        self._pending.clear()
        if error is not None:
            raise error


class ReferenceDevicePrefetchIterator:
    """Copy at most one future reference batch on an independent device stream."""

    def __init__(self, source, *, device, backend, max_batch_bytes, close_timeout=180.0):
        if (type(max_batch_bytes) is not int or max_batch_bytes <= 0
                or isinstance(close_timeout, bool) or not isinstance(close_timeout, (int, float))
                or not math.isfinite(close_timeout) or close_timeout <= 0):
            raise ValueError("Device prefetch requires positive batch bytes and finite close timeout")
        self.source, self.device, self.backend = source, device, backend
        self.max_batch_bytes, self.close_timeout = max_batch_bytes, close_timeout
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._closed = False
        self._stop = False
        self._done = False
        self._ready = None
        self._error = None
        self._thread = threading.Thread(target=self._run, name="reference-device-prefetch", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.backend.set_device(self.device.index if self.device.index is not None else 0)
            stream = self.backend.Stream()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._ready is None)
                    if self._stop:
                        return
                try:
                    batch = next(self.source)
                except StopIteration:
                    return
                with self._condition:
                    if self._stop:
                        return
                if batch.prepared_batch is None:
                    raise RuntimeError("Device prefetch requires a prepared reference batch")
                if _reference_device_transfer_bytes(batch.prepared_batch) > self.max_batch_bytes:
                    raise ValueError("Reference batch exceeds consumer_device_batch_bytes")
                try:
                    with self.backend.stream(stream):
                        transferred = {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                                       for key, value in batch.prepared_batch.items()}
                finally:
                    # Also wait after a partial-copy failure: pinned input must
                    # remain alive until all already-issued copies finish.
                    stream.synchronize()
                batch = replace(batch, prepared_batch=transferred)
                with self._condition:
                    if self._stop:
                        return
                    self._ready = batch
                    self._condition.notify_all()
                del batch, transferred
        except BaseException as exc:
            with self._condition:
                self._error = exc
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        with self._condition:
            self._condition.wait_for(lambda: self._ready is not None or self._done or self._stop)
            if self._stop:
                raise StopIteration
            if self._ready is not None:
                batch, self._ready = self._ready, None
                if self.device.type != "cpu":
                    stream = self.backend.current_stream()
                    for value in batch.prepared_batch.values():
                        if torch.is_tensor(value):
                            value.record_stream(stream)
                self._condition.notify_all()
                return batch
            if self._error is not None:
                raise self._error
            raise StopIteration

    @property
    def has_running_work(self):
        return self._thread.is_alive() or bool(getattr(self.source, "has_running_work", False))

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            deadline = time.monotonic() + self.close_timeout
            with self._condition:
                self._stop = True
                self._condition.notify_all()
            error = None
            try:
                if isinstance(self.source, TQPrefetchIterator):
                    self.source.close(timeout=max(1e-9, deadline - time.monotonic()))
                else:
                    self.source.close()
            except Exception as exc:
                error = exc
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise RuntimeError("Reference device prefetch did not stop") from error
            with self._condition:
                self._ready = None
            if error is not None:
                raise error
            self._closed = True
