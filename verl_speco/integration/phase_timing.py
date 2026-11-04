# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in host-side phase tracing; never synchronize accelerators or Ray refs."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import itertools
import json
import logging
import os
import time

logger = logging.getLogger(__name__)
_ids = itertools.count()
_REF_TRACE_KEY = "_speco_ref_phase_trace"
_ref_trace: ContextVar[tuple[dict, float] | None] = ContextVar(
    "speco_ref_phase_trace", default=None
)


@contextmanager
def ref_dispatch_trace(batch, *, source_step):
    """Temporarily carry a small diagnostic token through the existing RPC.

    Cross-process wall-clock differences include scheduling and transport;
    they are not isolated queue latency and require aligned host clocks.
    """
    meta = getattr(batch, "meta_info", None)
    if not phase_timing_enabled() or not isinstance(meta, dict):
        yield {}
        return
    trace = {
        "trace_id": f"{os.getpid()}:{source_step}:{next(_ids)}",
        "source_step": int(source_step),
        "driver_dispatch_unix_s": time.time(),
    }
    missing = object()
    previous = meta.get(_REF_TRACE_KEY, missing)
    meta[_REF_TRACE_KEY] = trace
    try:
        yield trace
    finally:
        if previous is missing:
            meta.pop(_REF_TRACE_KEY, None)
        else:
            meta[_REF_TRACE_KEY] = previous


def _read_ref_phase_trace(batch):
    """Read batch metadata or TensorDict's NonTensorData without transfers.

    Upstream batch conversion moves meta_info entries to top-level non-tensor
    keys; TensorDict does not retain the original meta_info attribute.
    This mirrors get_non_tensor_data's get/unwrap, without importing torch.
    """
    meta = getattr(batch, "meta_info", None)
    trace = meta.get(_REF_TRACE_KEY) if isinstance(meta, dict) else None
    if trace is None:
        getter = getattr(batch, "get", None)
        if callable(getter):
            trace = getter(_REF_TRACE_KEY, None)
    trace = getattr(trace, "data", trace)
    return trace if isinstance(trace, dict) else {}


def install_ref_worker_phase_timing(worker):
    """Wrap the existing bound RPC implementation, retaining dispatch metadata."""
    original = getattr(worker, "compute_ref_log_prob", None)
    if (
        not phase_timing_enabled()
        or not callable(original)
        or getattr(original, "_speco_ref_phase_timed", False)
    ):
        return

    @wraps(original)
    def measured(*args, **kwargs):
        if not phase_timing_enabled():
            return original(*args, **kwargs)
        batch = args[0] if args else kwargs.get("data", kwargs.get("batch"))
        trace = _read_ref_phase_trace(batch)
        labels = {"rank": getattr(worker, "rank", None), "role": "ref"}
        if isinstance(trace, dict):
            labels.update(
                {
                    key: trace[key]
                    for key in ("trace_id", "source_step", "driver_dispatch_unix_s")
                    if key in trace
                }
            )
        dispatched = labels.get("driver_dispatch_unix_s")
        if isinstance(dispatched, (int, float)):
            labels["dispatch_to_worker_observed_s"] = time.time() - dispatched
        token = _ref_trace.set((labels, time.perf_counter()))
        try:
            with phase_span("ref_worker", **labels):
                return original(*args, **kwargs)
        finally:
            _ref_trace.reset(token)

    setattr(measured, "_speco_ref_phase_timed", True)
    worker.compute_ref_log_prob = measured


def phase_timing_enabled():
    return os.getenv("VERL_SPECO_PHASE_TIMING", "").lower() in {"1", "true", "yes"}


def configure_phase_logger():
    """Opt-in stderr route independent of driver/Hydra logger filtering.

    Reconfigure only this diagnostic logger. Do not enable the root logger or
    change training log levels. A single owned handler prevents duplicates.
    """
    if not phase_timing_enabled():
        return
    logger.disabled = False
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not any(
        getattr(handler, "_speco_phase_handler", False) for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        setattr(handler, "_speco_phase_handler", True)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.addHandler(handler)


def log_driver_phase_setup(*, enabled, trainer_logger, source_file, ref_available):
    if not phase_timing_enabled():
        return
    configure_phase_logger()
    logger.warning(
        "[SpecoPhaseSetup] driver_enabled=%s pid=%s source_file=%s "
        "env=%r ref_hook_available=%s trainer_logger_disabled=%s "
        "trainer_logger_level=%s trainer_logger_propagate=%s route=diagnostic_stderr",
        enabled,
        os.getpid(),
        source_file,
        os.getenv("VERL_SPECO_PHASE_TIMING"),
        ref_available,
        trainer_logger.disabled,
        trainer_logger.getEffectiveLevel(),
        trainer_logger.propagate,
    )


@contextmanager
def phase_span(phase, *, enabled=True, metrics=None, **labels):
    """Wall timestamps allow cross-process correlation only on aligned clocks.

    Durations are monotonic, inclusive host wall times, not GPU kernel times.
    Nested spans must not be summed. Exceptions propagate unchanged.
    """
    if not enabled or not phase_timing_enabled():
        yield
        return
    call_id = next(_ids)
    start_unix_s = time.time()
    started = time.perf_counter()
    failed = True
    try:
        yield
        failed = False
    finally:
        elapsed = time.perf_counter() - started
        if metrics is not None:
            metrics[f"timing_s/phase_{phase}"] = elapsed
        configure_phase_logger()
        logger.warning(
            "[SpecoPhase] %s",
            json.dumps(
                dict(
                    labels,
                    phase=phase,
                    pid=os.getpid(),
                    call_id=call_id,
                    start_unix_s=start_unix_s,
                    end_unix_s=time.time(),
                    elapsed_s=elapsed,
                    failed=failed,
                ),
                sort_keys=True,
            ),
        )


class _TimedMode:
    def __init__(self, context, phase, labels):
        self.context, self.phase, self.labels = context, phase, labels

    def __enter__(self):
        with phase_span(f"{self.phase}_enter", **self.labels):
            return self.context.__enter__()

    def __exit__(self, *exc):
        with phase_span(f"{self.phase}_exit", **self.labels):
            return self.context.__exit__(*exc)


def install_engine_phase_timing(engine, *, role, rank):
    """Instrument existing engine calls without adding work or changing results.

    Mode enter/exit include restore/offload for Bubble and opt-in CUDA Sync;
    engine.to logs its model/optimizer/grad flags without splitting the transfer.
    """
    if not phase_timing_enabled() or getattr(engine, "_speco_phase_timed", False):
        return

    def wrap(name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            labels = {"role": role, "rank": rank}
            trace = _ref_trace.get() if role == "ref" else None
            if trace is not None:
                labels.update(trace[0])
                labels["worker_entry_to_call_s"] = time.perf_counter() - trace[1]
            if name == "to":
                labels["device"] = str(kwargs.get("device", args[0] if args else ""))
                for index, key in enumerate(("model", "optimizer", "grad"), 1):
                    labels[key] = bool(
                        kwargs.get(key, args[index] if len(args) > index else True)
                    )
            if name in {"train_mode", "eval_mode"}:
                return _TimedMode(original(*args, **kwargs), name, labels)
            with phase_span(f"engine_{name}", **labels):
                return original(*args, **kwargs)

        return measured

    for name in ("train_mode", "eval_mode", "to", "train_batch", "infer_batch"):
        original = getattr(engine, name, None)
        if callable(original):
            setattr(engine, name, wrap(name, original))
    engine._speco_phase_timed = True
