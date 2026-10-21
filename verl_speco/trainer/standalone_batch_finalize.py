# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Bounded FIFO finalization of successfully trained file batches.

Only the training thread submits/checks tasks. The worker performs CPU I/O,
never distributed collectives. Failed tasks remain observable until shutdown.
"""

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class AsyncBatchFinalizer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Finalization capacity must be positive")
        self.capacity = capacity
        self._pending: deque[Future] = deque()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="file-finalize"
        )

    def check(self) -> None:
        while self._pending and self._pending[0].done():
            self._pending[0].result()
            self._pending.popleft()

    def submit(self, operation: Callable[[], None]) -> None:
        self.check()
        if len(self._pending) >= self.capacity:
            self._pending[0].result()
            self._pending.popleft()
        self._pending.append(self._executor.submit(operation))

    def drain(self) -> None:
        while self._pending:
            self._pending[0].result()
            self._pending.popleft()

    def close(self) -> None:
        try:
            self.drain()
        finally:
            # Already-successful steps must finish cleanup even on training failure.
            self._executor.shutdown(wait=True)
