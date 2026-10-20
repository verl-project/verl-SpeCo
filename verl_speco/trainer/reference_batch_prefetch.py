# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Scheduler-owned assignments, whole-batch CPU processes, ordered pinning.

Only the dispatcher uses the prefetch Gloo group. Workers do not use Ray,
TQ or scheduler state. Hidden references are released after CPU preparation.
Mooncake may initialize a device context required by its transport.
Results use torch multiprocessing reducers, then the existing pin thread.
"""

from __future__ import annotations

import os
import logging
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
from verl_speco.trainer.tq_sample_source import TQPrefetchIterator

_STORE: Any = None
_OPTIONS: dict[str, Any] = {}
_ATTEMPTS = 1
logger = logging.getLogger(__name__)


def _initialize_worker(config, options, attempts):
    global _STORE, _OPTIONS, _ATTEMPTS
    # This is a separate process: do not change the training process's threads.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)
    from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore

    _STORE = ReferenceFeatureStore(config)
    _STORE.connect()
    _OPTIONS, _ATTEMPTS = dict(options), attempts


@torch.no_grad()
def _prepare_batch(entries):
    started = time.perf_counter()
    batch = retry_io(
        partial(_STORE.prepare_batch, entries, pin_memory=False, **_OPTIONS),
        name="Reference batch preparation",
        attempts=_ATTEMPTS,
    )
    # Do not delete inside the preparation retry: a retry must still be able to
    # read all samples. Deletion retries are independent and nonfatal.
    from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples

    cleanup_standalone_samples(_STORE, entries)
    return batch, time.perf_counter() - started


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
    ):
        if workers < 1 or depth < 1:
            raise ValueError(
                "Reference batch workers and prefetch depth must be positive"
            )
        if loader.prepare_batch is None or getattr(
            loader.store, "uses_transfer_queue", True
        ):
            raise ValueError("Process batch prefetch requires ray_reference input")
        self._pending: deque[Any] = deque()
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
        source = self.loader.iter_assignments()
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
                assignment, result, error, ended = None, None, None, False
                try:
                    assignment = next(source)
                    if assignment is None:
                        raise RuntimeError("Missing file batch assignment")
                    result = self._pool.apply_async(
                        _prepare_batch, (assignment.file_entries,)
                    )
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
                with self._condition:
                    self._pin_batches.append(
                        replace(
                            assignment,
                            prepared_batch=prepared,
                            file_entries=None,
                            batch_prepare_seconds=seconds,
                            cleanup_after_load=True,
                        )
                    )
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self.stop.set()
                self._condition.notify_all()
        finally:
            with self._condition:
                self._source_done = True
                self._condition.notify_all()

    def close(self):
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        self._collector.join(timeout=self.close_timeout)
        try:
            super().close()
        finally:
            # Cancel untrained tasks, including a blocked read or crashed worker.
            # Files of already-loaded batches may have been deleted; other
            # source files are left for regeneration or a later run.
            self._pool.terminate()
            self._pool.join()
            self._pending.clear()
        if self._collector.is_alive():
            raise RuntimeError("Reference batch result collector did not stop")
