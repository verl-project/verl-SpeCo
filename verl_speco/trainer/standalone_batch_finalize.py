# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded FIFO finalization of successfully trained reference batches.

Only the training thread submits/checks tasks. The worker performs CPU I/O,
never distributed collectives. Failed tasks remain observable until shutdown.
"""

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import math
import threading
import time


class AsyncBatchFinalizer:
    def __init__(self, capacity: int, *, timeout: float | None = None):
        if capacity < 1:
            raise ValueError("Finalization capacity must be positive")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Finalization timeout must be a positive finite number")
        self.capacity = capacity
        self.timeout = timeout
        self._pending: deque[Future] = deque()
        # Only the single executor thread reads or writes this failure fence.
        self._failure: BaseException | None = None
        self._stopped = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="file-finalize"
        )

    def _run(self, operation: Callable[[], None]) -> None:
        if self._stopped.is_set():
            raise RuntimeError("Finalization stopped after a timeout")
        if self._failure is not None:
            raise RuntimeError("Earlier batch finalization failed; later operations were fenced") from self._failure
        try:
            operation()
        except BaseException as error:
            self._failure = error
            raise

    def check(self) -> None:
        while self._pending and self._pending[0].done():
            self._pending[0].result()
            self._pending.popleft()

    @property
    def has_running_work(self) -> bool:
        return any(future.running() for future in self._pending)

    def submit(self, operation: Callable[[], None]) -> None:
        if self._stopped.is_set():
            raise RuntimeError("Finalization stopped after a timeout")
        self.check()
        if len(self._pending) >= self.capacity:
            try:
                self._pending[0].result(timeout=self.timeout)
            except TimeoutError:
                self._stopped.set()
                raise
            self._pending.popleft()
        self._pending.append(self._executor.submit(self._run, operation))

    def drain(self, *, timeout: float | None = None) -> None:
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Finalization timeout must be a positive finite number")
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                self._pending[0].result(timeout=remaining)
            except TimeoutError:
                self._stopped.set()
                raise
            self._pending.popleft()

    def close(self, *, timeout: float | None = None) -> None:
        try:
            self.drain(timeout=timeout)
        except BaseException:
            # A running I/O call cannot be cancelled; cancel only work that has
            # not started and let the caller preserve unresolved references.
            self._stopped.set()
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            self._executor.shutdown(wait=True)
