# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in host timings, with no additional RPCs or device synchronization."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
import time
import hashlib
import marshal
import sys

from verl_speco.integration.phase_timing import (
    configure_phase_logger,
    logger,
    phase_timing_enabled,
)

_background = ContextVar("speco_background_generation_diagnostics", default=None)


def emit_runtime_identity(owner, role, *functions):
    """Identify loaded code once, without a git subprocess or device access."""
    if not phase_timing_enabled() or getattr(
        owner, "_speco_generation_identity_logged", False
    ):
        return
    try:
        owner._speco_generation_identity_logged = True
        loaded = []
        for function in functions:
            code = getattr(getattr(function, "__func__", function), "__code__", None)
            if code is not None:
                loaded.append(
                    {
                        "name": code.co_name,
                        "file": code.co_filename,
                        "loaded_code_sha256": hashlib.sha256(
                            marshal.dumps(code)
                        ).hexdigest()[:16],
                    }
                )
        engine = getattr(owner, "engine", None)
        emit_summary(
            "runtime_identity",
            role=role,
            diagnostic_schema=2,
            python=sys.version.split()[0],
            vllm_version=getattr(sys.modules.get("vllm"), "__version__", None),
            engine_class=type(engine).__module__ + "." + type(engine).__name__
            if engine is not None
            else None,
            engine_log_stats=getattr(engine, "log_stats", None),
            capture_installed=bool(
                getattr(engine, "_speco_request_spec_decode_capture_installed", False)
            ),
            loaded=loaded,
        )
    except Exception:  # noqa: BLE001 - optional diagnostics cannot break serving
        logger.warning("[SpecoGeneration] runtime_identity_unavailable role=%s", role)


def empty_idle_poll_ablation_allowed(owner, training_config):
    """Diagnostic-only A/B before the first old-logprob collection.

    There can be no samples arriving during this rollout. Leases, admission,
    and the normal post-rollout event drain remain mandatory at the caller.
    Never apply this to SGLang collection or any existing training state.
    """
    if not phase_timing_enabled() or os.getenv(
        "VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", ""
    ).lower() not in {"1", "true", "yes"}:
        return False
    scheduler = getattr(owner, "_drafter_scheduler", None)
    runtime = getattr(owner, "_drafter_runtime_state", None)
    if scheduler is None or runtime is None:
        return False
    if (
        getattr(getattr(runtime, "status", None), "name", None) != "IDLE"
        or getattr(runtime, "active_plan", None) is not None
        or getattr(runtime, "training_ref", None) is not None
        or getattr(scheduler, "_training_quota_debt_steps", None) != 0
        or getattr(scheduler, "_training_quota_data_version", None) is not None
        or getattr(scheduler, "_training_quota_collection_step", None) is not None
        or getattr(scheduler, "_idle_worker_writer_state_version", None) is not None
        or getattr(scheduler, "_idle_optimizer_initialized_groups", ())
        or getattr(scheduler, "_latest_published_drafter_version", None) is not None
        or getattr(owner, "_speco_last_published_drafter_step", 0)
        or getattr(owner, "_speco_published_drafter_version", None) is not None
        or any(
            getattr(owner, name, None)
            for name in (
                "_pending_target_lm_head_sync",
                "_pending_drafter_publish_refs",
                "_pending_drafter_publish_contexts",
                "_pending_drafter_checkpoint_refs",
                "_speco_rollout_training_lease",
                "_speco_idle_completed_publications",
                "_speco_last_collected_samples",
                "_speco_direct_collected_samples",
            )
        )
    ):
        return False
    config = training_config()
    quota = (config.get("scheduler") or {}).get("training_quota") or {}
    try:
        first_collection = min(
            int(config.get("collect_interval_steps", 1)),
            int(config.get("training_interval_steps", 1)),
        )
        before_first_collection = 0 < int(owner.global_steps) <= first_collection
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        before_first_collection
        and quota.get("enable", False) is True
        and config.get("collect_hidden_states_from_old_logprob", False) is True
        and not config.get("collect_hidden_states_from_sgl", False)
    )


def record_sidecar_io(owner, directory, elapsed, *, succeeded):
    """Cumulative worker I/O cost, at most one record every 30 seconds.

    No extra files/RPCs and no change to acceptance-counter publication.
    These are process-local synchronous I/O spans, not GPU execution time.
    """
    try:
        totals = getattr(owner, "_speco_sidecar_io_diagnostic", None)
        if totals is None:
            totals = {"calls": 0, "sum_s": 0.0, "max_s": 0.0, "failures": 0}
            owner._speco_sidecar_io_diagnostic = totals
        totals["calls"] += 1
        totals["sum_s"] += elapsed
        totals["max_s"] = max(totals["max_s"], elapsed)
        totals["failures"] += int(not succeeded)
        now = time.monotonic()
        last = getattr(owner, "_speco_sidecar_io_last_report", None)
        if last is None or now - last >= 30.0:
            owner._speco_sidecar_io_last_report = now
            emit_summary("sidecar_io", directory=os.fspath(directory), **totals)
    except Exception:  # noqa: BLE001 - optional reporting cannot affect counters
        pass


@contextmanager
def background_diagnostics(collector):
    token = _background.set(collector)
    try:
        yield
    finally:
        _background.reset(token)


@contextmanager
def background_span(name):
    collector = _background.get()
    if collector is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        collector[f"{name}_calls"] = collector.get(f"{name}_calls", 0) + 1
        collector[f"{name}_sum_s"] = collector.get(f"{name}_sum_s", 0.0) + elapsed
        collector[f"{name}_max_s"] = max(collector.get(f"{name}_max_s", 0.0), elapsed)


def emit_summary(kind, **fields):
    try:
        configure_phase_logger()
        try:
            payload = json.dumps(
                dict(kind=kind, pid=os.getpid(), wall_time_s=time.time(), **fields),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            logger.warning(
                "[SpecoGeneration] summary_serialization_failed kind=%s", kind
            )
            return
        logger.warning(
            "[SpecoGeneration] %s",
            payload,
        )
    except Exception:  # noqa: BLE001, S110 - logging this failure could raise again
        # Custom handlers/filters can raise even when logging.raiseExceptions
        # is false. Never retry through that same failing diagnostic route.
        pass


def record_admission(plan):
    """Count planning decisions, not successful launches or optimizer steps."""
    collector = _background.get()
    if collector is None:
        return
    reason = str(getattr(plan, "reason", "no_plan"))
    counts = collector.setdefault("admission_reasons", {})
    counts[reason] = counts.get(reason, 0) + 1
    collector["last_admission"] = {
        "reason": reason,
        "launch_planned": bool(getattr(plan, "launch", False)),
        "workers": list(getattr(plan, "target_worker_ids", ()) or ()),
        "planned_steps": getattr(plan, "max_batches", 0),
        "window_s": getattr(plan, "idle_window_sec", None),
        "deadline_ts": getattr(plan, "deadline_ts", None),
    }
    for name in (
        "idle_usable_window_sec",
        "idle_window_batches",
        "idle_batch_estimate_sec",
        "idle_startup_reserve_sec",
        "idle_first_step_extra_sec",
        "idle_tail_reserve_sec",
        "idle_reclaim_penalty_sec",
        "idle_window_source",
    ):
        collector["last_admission"][name] = getattr(plan, name, None)


def record_gate_metrics(metrics):
    collector = _background.get()
    if collector is None:
        return
    for key, value in metrics.items():
        if (
            (
                key.startswith("bubble/training_blocked_")
                or key
                in {
                    "scheduler/pending_training_count",
                    "bubble/rollout_tail_launch_suppressed",
                    "bubble/post_generation_launch_suppressed",
                    "bubble/rollout_resource_lease_raced",
                    "bubble/training_resource_lease_raced",
                }
            )
            and isinstance(value, (int, float))
            and value > 0
        ):
            counts = collector.setdefault("gate_counts", {})
            counts[key] = counts.get(key, 0) + 1


def control_snapshot(owner):
    """Read cached coordinator state only: never select, poll or resolve refs."""
    scheduler = getattr(owner, "_drafter_scheduler", None)
    runtime = getattr(owner, "_drafter_runtime_state", None)
    plan = getattr(runtime, "active_plan", None)
    pending = getattr(owner, "_pending_target_lm_head_sync", None) or {}
    return {
        "writer_workers": list(
            getattr(scheduler, "_idle_worker_writer_group", ()) or ()
        ),
        "active_plan_id": getattr(plan, "plan_id", None),
        "runtime_status": getattr(getattr(runtime, "status", None), "name", None),
        "active_workers": list(getattr(plan, "target_worker_ids", ()) or ()),
        "quota_data_version": getattr(scheduler, "_training_quota_data_version", None),
        "quota_collection_step": getattr(
            scheduler, "_training_quota_collection_step", None
        ),
        "quota_debt_steps": getattr(scheduler, "_training_quota_debt_steps", 0),
        "head_pending_stage": pending.get("stage"),
        "head_pending_version": pending.get(
            "target_version", pending.get("global_step")
        ),
        "publish_pending": bool(getattr(owner, "_pending_drafter_publish_refs", None)),
        "publish_deferred_count": getattr(owner, "_speco_deferred_publish_count", 0),
        "published_version": getattr(
            scheduler, "_latest_published_drafter_version", None
        ),
        "idle_optimizer_steps_total": getattr(
            scheduler, "_idle_optimizer_steps_total", 0
        ),
        "topup_optimizer_steps_total": getattr(
            scheduler, "_critical_path_topup_optimizer_steps_total", 0
        ),
    }


def idle_snapshot(owner):
    """Observed tails are opportunities, not a proof that training would fit."""
    scheduler = getattr(owner, "_drafter_scheduler", None)
    mapping = getattr(scheduler, "_replica_idle_worker_groups", {})
    windows = getattr(scheduler, "_replica_last_observed_idle_window_sec", {})
    histories = getattr(scheduler, "_replica_idle_window_samples_sec", {})
    states = getattr(scheduler, "_idle_workers", {})
    writer = tuple(getattr(scheduler, "_idle_worker_writer_group", ()) or ())
    generation_history = getattr(scheduler, "_writer_idle_history", ())
    current_tails = generation_history[-1] if generation_history else {}
    replicas = []
    for rank, workers in mapping.items():
        replicas.append(
            {
                "replica_rank": rank,
                "workers": list(workers),
                "writer": tuple(workers) == writer,
                "last_observed_tail_s": windows.get(rank),
                "current_generation_group_tail_s": current_tails.get(tuple(workers)),
                "last_close_source": getattr(
                    scheduler, "_replica_idle_window_close_source", {}
                ).get(rank),
                "event_delivery_lag_s": getattr(
                    scheduler, "_replica_idle_event_delivery_lag_sec", {}
                ).get(rank),
                "recent_tail_s": list(histories.get(rank, ()))[-4:],
                "worker_status": [
                    getattr(states.get(worker), "status", "unknown")
                    for worker in workers
                ],
                "memory_released": [
                    getattr(states.get(worker), "memory_released", False)
                    for worker in workers
                ],
            }
        )
    return {"control": control_snapshot(owner), "replicas": replicas}


def allocator_snapshot(device_module):
    """Process-local allocator counters, not device-wide free memory or a sync."""
    result = {}
    for name in ("memory_allocated", "memory_reserved"):
        getter = getattr(device_module, name, None)
        if callable(getter):
            try:
                result[f"{name}_bytes"] = int(getter())
            except Exception:  # noqa: BLE001 - optional device diagnostics
                pass
    return result


@contextmanager
def pipeline_boundary(owner, kind):
    if (
        not phase_timing_enabled()
        or not getattr(owner, "_speco_rollout_idle_worker_enabled", lambda: False)()
    ):
        yield None
        return
    started = time.perf_counter()
    timings = {}
    before = control_snapshot(owner)
    failed = True
    try:
        yield timings
        failed = False
    finally:
        emit_summary(
            kind,
            source_step=getattr(owner, "global_steps", None),
            elapsed_s=time.perf_counter() - started,
            stages=timings,
            failed=failed,
            control_before=before,
            control_after=control_snapshot(owner),
        )


@contextmanager
def boundary_stage(timings, name):
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[f"{name}_s"] = time.perf_counter() - started


def request_metadata(args, kwargs):
    params = kwargs.get("sampling_params", args[1] if len(args) > 1 else None)
    step = params.get("_verl_global_steps") if isinstance(params, dict) else None
    request_id = kwargs.get("request_id", args[2] if len(args) > 2 else None)
    tokens = kwargs.get("prompt_ids", args[0] if args else None)
    # Never materialize tensors or log prompt/response contents.
    prompt_tokens = len(tokens) if isinstance(tokens, (list, tuple)) else None
    return {
        "source_step_hint": step if isinstance(step, int) else None,
        "request_id": str(request_id)[:96] if request_id is not None else None,
        "prompt_tokens": prompt_tokens,
    }


def request_engine_timing(metrics):
    """Read optional native host timestamps; never enable vLLM stats or sync.

    Missing metrics stay absent, not zero. Durations cover one request and
    include overlapping work; they are not GPU-kernel execution time.
    """

    def number(name):
        value = getattr(metrics, name, None)
        return (
            float(value)
            if isinstance(value, (int, float)) and math.isfinite(value)
            else None
        )

    result = {}
    for name, start, end in (
        ("queue_s", "arrival_time", "first_scheduled_time"),
        ("time_to_first_token_s", "arrival_time", "first_token_time"),
        ("decode_span_s", "first_token_time", "last_token_time"),
        ("request_span_s", "arrival_time", "finished_time"),
    ):
        before, after = number(start), number(end)
        if before is not None and after is not None and 0 <= before <= after:
            result[name] = after - before
    queued = number("time_in_queue")
    if queued is not None and queued >= 0:
        result["queue_s"] = queued
    # Recent V1 RequestStateStats uses core monotonic timestamps. Do not
    # subtract its wall-clock arrival_time from any *_ts value.
    for name, start, end in (
        ("queue_s", "queued_ts", "scheduled_ts"),
        ("prefill_span_s", "scheduled_ts", "first_token_ts"),
        ("decode_span_s", "first_token_ts", "last_token_ts"),
    ):
        before, after = number(start), number(end)
        if before is not None and after is not None and 0 < before <= after:
            result.setdefault(name, after - before)
    latency = number("first_token_latency")
    if latency is not None and latency > 0:
        result.setdefault("time_to_first_token_s", latency)
    return result


def observe_engine_output(observation, output):
    """Bounded status for a registered request; no token contents retained."""
    try:
        observation["outputs"] += 1
        observation["output_type"] = type(output).__name__
        observation["has_finished"] = hasattr(output, "finished")
        if not getattr(output, "finished", False):
            observation["state"] = "final_output_not_seen"
            return
        observation["final_outputs"] += 1
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            observation["state"] = "metrics_missing"
            return
        observation["metrics_type"] = type(metrics).__name__
        observation["timestamp_fields"] = [
            name
            for name in (
                "arrival_time",
                "first_scheduled_time",
                "first_token_time",
                "last_token_time",
                "finished_time",
                "time_in_queue",
                "queued_ts",
                "scheduled_ts",
                "first_token_ts",
                "last_token_ts",
                "first_token_latency",
            )
            if getattr(metrics, name, None) is not None
        ]
        timing = request_engine_timing(metrics)
        observation["state"] = (
            "available" if timing else "timestamps_missing_or_invalid"
        )
        if timing:
            observation["timing"] = timing
    except Exception:  # noqa: BLE001 - preserve the original request result/error
        observation["state"] = "diagnostic_error"


def record_request(owner, trace, *, failed):
    """O(1) counters; log only when the current serving episode drains.

    An episode is not necessarily a whole rollout. Later requests for the same
    step may produce another summary, so step hints and episode IDs are explicit.
    Request sums overlap across concurrent calls; they are not critical-path time.
    """
    summary = getattr(owner, "_speco_generation_request_summary", None)
    if summary is None:
        episode = getattr(owner, "_speco_generation_diagnostic_episode", 0) + 1
        owner._speco_generation_diagnostic_episode = episode
        summary = {"episode": episode, "requests": 0, "failed": 0}
        owner._speco_generation_request_summary = summary
    summary["requests"] += 1
    summary["failed"] += int(failed)
    timing_state = trace.get("engine_timing_status", {}).get("state")
    if timing_state is not None:
        counts = summary.setdefault("engine_timing_status_counts", {})
        counts[timing_state] = counts.get(timing_state, 0) + 1
    if trace.get("lease_acquire_called"):
        summary["lease_acquire_calls"] = summary.get("lease_acquire_calls", 0) + 1
        summary["lease_unavailable_calls"] = summary.get(
            "lease_unavailable_calls", 0
        ) + int(not trace.get("lease_acquired"))
    for name in (
        "admission",
        "initialize",
        "lease_lock",
        "lease_acquire",
        "generate",
        "release",
        "wrapper",
    ):
        elapsed = trace.get(f"{name}_s", 0.0)
        summary[f"{name}_sum_s"] = summary.get(f"{name}_sum_s", 0.0) + elapsed
        summary[f"{name}_max_s"] = max(summary.get(f"{name}_max_s", 0.0), elapsed)
    step = trace.get("source_step_hint")
    if step is not None:
        summary["source_step_min"] = min(summary.get("source_step_min", step), step)
        summary["source_step_max"] = max(summary.get("source_step_max", step), step)
    if trace["total_s"] >= summary.get("slowest_total_s", -1.0):
        summary["slowest_total_s"] = trace["total_s"]
        summary["slowest"] = dict(trace, failed=failed)


def flush_requests(owner):
    summary = getattr(owner, "_speco_generation_request_summary", None)
    if summary is not None:
        owner._speco_generation_request_summary = None
        emit_summary(
            "replica_episode",
            replica_rank=getattr(owner, "replica_rank", None),
            **summary,
        )
