# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded retries for transient transport failures, including wrapped TQ errors."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def is_transient_io_error(error: BaseException) -> bool:
    # TQ wraps controller/storage exceptions in RuntimeError; inspect causes,
    # rather than retrying arbitrary RuntimeError (e.g. malformed sample data).
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        if type(error).__module__.startswith(("ray.", "zmq")) and type(
            error
        ).__name__ in {
            "GetTimeoutError",
            "RaySystemError",
            "Again",
        }:
            return True
        cause = error.__cause__ or (
            None if error.__suppress_context__ else error.__context__
        )
        if cause is None:
            return False
        error = cause
    return False


def retry_io(
    operation: Callable[[], T],
    *,
    name: str,
    attempts: int = 3,
    stop: threading.Event | None = None,
    retry_if: Callable[[BaseException], bool] = is_transient_io_error,
    initial_backoff: float = 0.2,
    max_backoff: float = 2.0,
    jitter: bool = True,
) -> T:
    if attempts < 1:
        raise ValueError("io_retry_max_attempts must be positive")
    if initial_backoff < 0 or max_backoff < initial_backoff:
        raise ValueError("Retry backoff must be non-negative with max >= initial")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt + 1 == attempts or not retry_if(error):
                raise
            delay = min(initial_backoff * 2**attempt, max_backoff)
            if jitter:
                delay *= random.uniform(0.9, 1.1)
            logger.warning(
                "%s retry %s/%s in %.3fs: %r", name, attempt + 1, attempts, delay, error
            )
            if stop is not None:
                if stop.wait(delay):
                    raise RuntimeError(f"{name} cancelled during retry") from error
            else:
                time.sleep(delay)
    raise AssertionError("unreachable")
