# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in server lifecycle traces without accelerator reads or scheduling changes."""

import inspect
import json
import os
import time
from contextlib import contextmanager
from functools import wraps

from verl_speco.integration.phase_timing import (
    configure_phase_logger,
    logger,
    phase_timing_enabled,
)


@contextmanager
def lifecycle_span(server, method):
    if not phase_timing_enabled():
        yield
        return
    configure_phase_logger()
    state = server.__dict__.setdefault("_speco_lifecycle_trace", {})
    counts = state.setdefault("calls", {})
    counts[method] = counts.get(method, 0) + 1
    started = time.perf_counter()
    labels = {
        "method": method,
        "pid": os.getpid(),
        "replica_rank": getattr(server, "replica_rank", None),
        "call_index": counts[method],
        "first_call": counts[method] == 1,
    }
    logger.warning(
        "[SpecoLifecycle] %s",
        json.dumps(
            {
                **labels,
                "event": "begin",
                "wall_time_s": time.time(),
            },
            sort_keys=True,
        ),
    )
    failed = True
    try:
        yield
        failed = False
    finally:
        ended = time.perf_counter()
        detail = {
            **labels,
            "event": "end",
            "wall_time_s": time.time(),
            "elapsed_s": ended - started,
            "failed": failed,
        }
        # Diagnostic intervals only: these fields never control request admission.
        if not failed:
            if method in {"abort_all_requests", "pause_generation"}:
                state.setdefault("paused_at", ended)
            elif method == "resume_generation":
                paused = state.pop("paused_at", None)
                if paused is not None:
                    detail["pause_to_resume_s"] = ended - paused
            elif method == "sleep":
                state.setdefault("slept_at", ended)
            elif method == "wake_up":
                slept = state.pop("slept_at", None)
                if slept is not None:
                    detail["sleep_to_wake_s"] = ended - slept
        logger.warning("[SpecoLifecycle] %s", json.dumps(detail, sort_keys=True))


def _wrap(method, name):
    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def measured(self, *args, **kwargs):
            with lifecycle_span(self, name):
                return await method(self, *args, **kwargs)
    else:

        @wraps(method)
        def measured(self, *args, **kwargs):
            with lifecycle_span(self, name):
                return method(self, *args, **kwargs)

    measured._speco_lifecycle_timed = True
    return measured


def install_server_lifecycle_timing(server_cls):
    """Install before Ray captures the actor method table; preserve signatures."""
    if not phase_timing_enabled():
        return
    for name in (
        "abort_all_requests",
        "pause_generation",
        "resume_generation",
        "sleep",
        "wake_up",
        "clear_kv_cache",
    ):
        method = getattr(server_cls, name, None)
        descriptor = inspect.getattr_static(server_cls, name, None)
        if (
            not callable(method)
            or isinstance(descriptor, (staticmethod, classmethod))
            or inspect.isasyncgenfunction(method)
            or inspect.isgeneratorfunction(method)
            or getattr(method, "_speco_lifecycle_timed", False)
        ):
            continue
        setattr(server_cls, name, _wrap(method, name))
