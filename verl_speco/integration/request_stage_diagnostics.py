# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in request wall timings; no tensor reads, extra RPCs or device sync."""

from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
import sys
import time
from time import perf_counter as _clock

FLAG = "VERL_SPECO_DIAGNOSTIC_REQUEST_STAGES"
_request = ContextVar("speco_request_stage", default=None)
_prompt = ContextVar("speco_prompt_stage", default=None)
_scope = ContextVar("speco_request_stage_scope", default=None)


def enabled():
    return os.getenv(FLAG, "").lower() in {"1", "true", "yes"}


def emit(kind, **fields):
    try:
        print(
            "[SpecoRequestStages] "
            + json.dumps(
                dict(kind=kind, pid=os.getpid(), wall_unix_s=time.time(), **fields),
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 -- diagnostics cannot break generation
        pass


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def _stage(original, name):
    @wraps(original)
    async def wrapped(*args, **kwargs):
        # Resolve process-local state after Ray deserializes the wrapper.
        from verl_speco.integration import request_stage_diagnostics as diagnostics

        trace = diagnostics._request.get() if enabled() else None
        if trace is None:
            return await _await(original(*args, **kwargs))
        start = _clock()
        metric = name
        if name == "llm_rpc" and diagnostics._scope.get() in {"teacher", "reward"}:
            metric = diagnostics._scope.get() + "_llm_rpc"
        scope_token = diagnostics._scope.set(name) if name in {"teacher", "reward"} else None
        try:
            return await _await(original(*args, **kwargs))
        finally:
            if scope_token is not None:
                diagnostics._scope.reset(scope_token)
            trace[metric + "_s"] = (
                trace.get(metric + "_s", 0.0) + _clock() - start
            )
            trace[metric + "_done_unix_s"] = time.time()

    wrapped._speco_request_stage_wrapped = True
    return wrapped


def _install_stage(owner, method, name):
    original = getattr(owner, method, None)
    if callable(original) and not getattr(
        original, "_speco_request_stage_wrapped", False
    ):
        setattr(owner, method, _stage(original, name))


def install_request_stage_diagnostics(worker_cls, client_cls=None):
    """Handle TQ fire-and-forget prompts and inherited/by-value worker methods."""
    if not enabled():
        return
    if client_cls is not None:
        _install_stage(client_cls, "generate", "llm_rpc")
    client_module = sys.modules.get("verl.workers.rollout.llm_server")
    partial_client = getattr(client_module, "FullyAsyncLLMServerClient", None)
    if partial_client is not None:
        _install_stage(partial_client, "generate", "llm_rpc")
    for method, name in (
        ("_compute_score", "reward"),
        ("_compute_teacher_logprobs", "teacher"),
        ("_agent_loop_postprocess", "postprocess"),
    ):
        _install_stage(worker_cls, method, name)
    run = getattr(worker_cls, "_run_agent_loop", None)
    if callable(run) and not getattr(run, "_speco_request_stage_wrapped", False):

        @wraps(run)
        async def request(self, sampling_params, trajectory, *args, **kwargs):
            # Resolve process-local state after Ray deserializes the wrapper.
            from verl_speco.integration import request_stage_diagnostics as diagnostics

            if not enabled() or diagnostics._request.get() is not None:
                return await _await(
                    run(self, sampling_params, trajectory, *args, **kwargs)
                )
            trajectory = trajectory if isinstance(trajectory, dict) else {}
            uid = str(kwargs.get("uid", ""))
            trace = {
                "uid_hash": hashlib.sha256(uid.encode()).hexdigest()[:16],
                "session": str(kwargs.get("session_id", "")),
                "source_step": str(trajectory.get("step", "unknown")),
                "validate": bool(trajectory.get("validate", False)),
                "start_unix_s": time.time(),
                "failed": True,
            }
            start = _clock()
            token = diagnostics._request.set(trace)
            try:
                result = await _await(
                    run(self, sampling_params, trajectory, *args, **kwargs)
                )
                trace["failed"] = False
                return result
            finally:
                diagnostics._request.reset(token)
                trace["total_s"] = _clock() - start
                trace["postprocess_other_s"] = max(
                    0.0,
                    trace.get("postprocess_s", 0.0)
                    - sum(
                        trace.get(key + "_s", 0.0)
                        for key in ("reward", "teacher", "tq_put")
                    ),
                )
                bucket = diagnostics._prompt.get()
                if bucket is None:
                    emit("request", **trace)
                else:
                    bucket["count"] += 1
                    bucket["total_request_s"] += trace["total_s"]
                    bucket["slowest"] = sorted(
                        bucket["slowest"] + [trace],
                        key=lambda row: row["total_s"],
                        reverse=True,
                    )[:3]

        request._speco_request_stage_wrapped = True
        worker_cls._run_agent_loop = request
    run_prompt = getattr(worker_cls, "_run_prompt", None)
    if callable(run_prompt) and not getattr(
        run_prompt, "_speco_request_stage_wrapped", False
    ):

        @wraps(run_prompt)
        async def prompt(self, *args, **kwargs):
            # Resolve process-local state after Ray deserializes the wrapper.
            from verl_speco.integration import request_stage_diagnostics as diagnostics

            if not enabled() or diagnostics._prompt.get() is not None:
                return await _await(run_prompt(self, *args, **kwargs))
            bucket = {"count": 0, "total_request_s": 0.0, "slowest": []}
            start = _clock()
            token = diagnostics._prompt.set(bucket)
            failed = True
            try:
                result = await _await(run_prompt(self, *args, **kwargs))
                failed = False
                return result
            finally:
                diagnostics._prompt.reset(token)
                emit(
                    "prompt_summary",
                    total_s=_clock() - start,
                    failed=failed,
                    **bucket,
                )

        prompt._speco_request_stage_wrapped = True
        worker_cls._run_prompt = prompt
    tq = sys.modules.get("transfer_queue")
    if tq is not None:
        _install_stage(tq, "async_kv_batch_put", "tq_put")
        status = getattr(tq, "async_kv_put", None)
        if callable(status) and not getattr(
            status, "_speco_request_stage_wrapped", False
        ):

            @wraps(status)
            async def notified(*args, **kwargs):
                # Resolve process-local state after Ray deserializes the wrapper.
                from verl_speco.integration import request_stage_diagnostics as diagnostics

                bucket = diagnostics._prompt.get() if enabled() else None
                if bucket is None:
                    return await _await(status(*args, **kwargs))
                start = _clock()
                try:
                    return await _await(status(*args, **kwargs))
                finally:
                    bucket["status_notify_s"] = (
                        bucket.get("status_notify_s", 0.0) + _clock() - start
                    )

            notified._speco_request_stage_wrapped = True
            tq.async_kv_put = notified


def install_replay_sample_timing(replay):
    """Log the complete V1 gen wait, separately from per-request stages."""
    if not enabled() or replay is None:
        return
    original = getattr(replay, "sample", None)
    if not callable(original) or getattr(
        original, "_speco_request_stage_wrapped", False
    ):
        return

    @wraps(original)
    def sampled(*args, **kwargs):
        if not enabled():
            return original(*args, **kwargs)
        start, wall = _clock(), time.time()
        failed = True
        try:
            result = original(*args, **kwargs)
            failed = False
            return result
        finally:
            emit(
                "replay_sample",
                source_step=str(kwargs.get("global_steps", "unknown")),
                start_unix_s=wall,
                total_s=_clock() - start,
                failed=failed,
            )

    sampled._speco_request_stage_wrapped = True
    replay.sample = sampled
