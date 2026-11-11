# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in snapshots and request-level counters; no token logging or device fences."""

import os
import hashlib
import sys
from functools import lru_cache
import time
from contextlib import contextmanager
from pathlib import Path


def enabled():
    from verl_speco.utils.env import env_flag

    return env_flag("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF")


@lru_cache(maxsize=1)
def source_fingerprints():
    result = {}
    for name in (
        "verl_speco.integration.vllm_runtime",
        "verl_speco.workers.speco_worker",
        "verl_speco.trainer.base_trainer",
        "verl_speco.integration.rollout_performance",
    ):
        filename = getattr(sys.modules.get(name), "__file__", None)
        if filename:
            try:
                result[name] = {
                    "path": filename,
                    "disk_sha256": hashlib.sha256(
                        Path(filename).read_bytes()
                    ).hexdigest(),
                }
            except OSError:
                result[name] = {"path": filename, "disk_sha256": "unavailable"}
    return result


def runtime_snapshot(owner, role, stage, *, requested=False):
    """Never initialize CUDA, gather tensors, read tensor values or synchronize."""
    if not requested and not enabled():
        return {}
    started = time.perf_counter()
    result = {
        "role": role,
        "stage": stage,
        "rank": getattr(owner, "rank", None),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "unavailable": [],
    }
    try:
        status = Path("/proc/self/status").read_text()
        result["process"] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.split(":", 1)[0] in {"VmRSS", "VmLck", "Threads"}
        }
    except OSError:
        result["unavailable"].append("proc_status")
    if hasattr(os, "sched_getaffinity"):
        result["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    import torch

    result["sources"] = source_fingerprints()
    result["torch_version"] = str(torch.__version__)
    result["torch_threads"] = torch.get_num_threads()
    from verl_speco.trainer.performance_diagnostics import accelerator_runtime_metadata

    result.update(accelerator_runtime_metadata())
    trainer = getattr(owner, "trainer", None)
    if trainer is not None:
        result["training"] = {
            "resident": getattr(trainer, "_training_device_resident", None),
            "active": getattr(trainer, "_training_active", None),
            "optimizer_steps": getattr(trainer, "optimizer_steps_total", None),
            "group_ranks": list(getattr(owner, "training_group_ranks", ()) or ()),
        }
    config = getattr(owner, "vllm_config", None)
    runner = getattr(owner, "model_runner", None)
    for label, obj, fields in (
        (
            "cache_config",
            getattr(config, "cache_config", None),
            (
                "block_size",
                "num_gpu_blocks",
                "gpu_memory_utilization",
                "kv_cache_memory_bytes",
                "cache_dtype",
            ),
        ),
        (
            "scheduler_config",
            getattr(config, "scheduler_config", None),
            ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill"),
        ),
        ("kv_cache_config", getattr(runner, "kv_cache_config", None), ("num_blocks",)),
    ):
        result[label] = {}
        for name in fields:
            value = getattr(obj, name, None)
            if isinstance(value, (int, float, str, bool)):
                result[label][name] = value
            elif role == "vllm_worker":
                result[label][name] = None
                result["unavailable"].append(f"{label}.{name}")
    result["snapshot_elapsed_s"] = time.perf_counter() - started
    from verl_speco.integration.generation_diagnostics import emit_summary

    emit_summary("rollout_perf_snapshot", **result)
    return result


def safe_runtime_snapshot(owner, role, stage, *, requested=False):
    try:
        return runtime_snapshot(owner, role, stage, requested=requested)
    except Exception as error:
        from verl_speco.integration.generation_diagnostics import emit_summary

        emit_summary(
            "rollout_perf_snapshot_failed",
            role=role,
            stage=stage,
            error=type(error).__name__,
        )
        return {"unavailable": [type(error).__name__]}


@contextmanager
def generation_execution(owner):
    """Frontend execution concurrency, NOT engine scheduler running/waiting counts."""
    from verl_speco.utils.diagnostics import diagnostic_state

    if (
        not enabled()
        or getattr(diagnostic_state(owner), "_speco_perf_execution_episode", 0) >= 3
    ):
        yield
        return
    now = time.perf_counter()
    state = getattr(diagnostic_state(owner), "_speco_perf_execution", None)
    if state is None:
        state = {
            "active": 0,
            "peak": 0,
            "requests": 0,
            "start": now,
            "last_event": now,
            "last_submit": now,
            "active_area_s": 0.0,
            "submit_gap_max_s": 0.0,
            "execution_sum_s": 0.0,
            "failed": 0,
        }
        diagnostic_state(owner)._speco_perf_execution = state
    state["active_area_s"] += state["active"] * (now - state["last_event"])
    state["submit_gap_max_s"] = max(
        state["submit_gap_max_s"], now - state["last_submit"]
    )
    state["last_submit"] = state["last_event"] = now
    state["requests"] += 1
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    failed = True
    try:
        yield
        failed = False
    finally:
        end = time.perf_counter()
        state["active_area_s"] += state["active"] * (end - state["last_event"])
        state["execution_sum_s"] += end - now
        state["failed"] += int(failed)
        state["active"] -= 1
        state["last_event"] = end
        if state["active"] == 0:
            diagnostic_state(owner)._speco_perf_execution = None
            episode = (
                getattr(diagnostic_state(owner), "_speco_perf_execution_episode", 0) + 1
            )
            diagnostic_state(owner)._speco_perf_execution_episode = episode
            from verl_speco.integration.generation_diagnostics import emit_summary

            elapsed = end - state["start"]
            emit_summary(
                "rollout_perf_execution",
                replica_rank=getattr(owner, "replica_rank", None),
                episode=episode,
                requests=state["requests"],
                failed=state["failed"],
                frontend_peak_active=state["peak"],
                frontend_mean_active=state["active_area_s"] / max(elapsed, 1e-9),
                submit_span_s=state["last_submit"] - state["start"],
                submit_gap_max_s=state["submit_gap_max_s"],
                execution_wall_s=elapsed,
                execution_sum_s=state["execution_sum_s"],
                engine_scheduler_counts="unavailable_in_frontend",
            )
