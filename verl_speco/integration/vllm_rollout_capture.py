# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in V1 rollout feature capture with asynchronous export (vLLM V1).

The sampler hook runs before the proposer can reuse target output buffers.
Only intersecting window rows are copied to CPU and filtered before export.
Owned GPU snapshots protect graph outputs while D2H/export overlap execution.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import os
import shutil
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

CAPTURE_KEY = "speco_rollout_capture"
CAPTURE_FLAG = "collect_hidden_states_from_vllm"
logger = logging.getLogger(__name__)
_EXPORT_QUEUE: Any = None
_GPU_TIMING_SEQUENCE = itertools.count()


def capture_trace(message, *args):
    """Expose request decisions without enabling verbose engine-wide logs."""
    if os.getenv("SPECO_CAPTURE_DIAGNOSTICS") == "1":
        rendered = message % args if args else message
        print(
            f"[speco capture pid={os.getpid()} time={time.time():.6f}] {rendered}",
            flush=True,
        )
    else:
        logger.debug(message, *args)


def capture_detail(message, *args):
    if os.getenv("SPECO_CAPTURE_DIAGNOSTICS_DETAIL") == "1":
        capture_trace(message, *args)


def export_queue():
    global _EXPORT_QUEUE
    if _EXPORT_QUEUE is None:
        from verl_speco.producer.feature_export import FeatureExportQueue

        _EXPORT_QUEUE = FeatureExportQueue(
            # Request/sample/token admission remains owned by the scheduler.
            # Do not impose a second, smaller chunk limit on authorized samples.
            max_bytes=int(os.getenv("SPECO_CAPTURE_EXPORT_MAX_BYTES", str(512 << 20))),
            max_pending=int(os.getenv("SPECO_CAPTURE_EXPORT_MAX_PENDING", "256")),
            pack_chunks=os.getenv("SPECO_CAPTURE_PACK_CHUNKS") == "1",
        )
    return _EXPORT_QUEUE


def enabled(config: Any) -> bool:
    return bool((config.get("training") or {}).get(CAPTURE_FLAG, False))


def validate_config(drafter: Any, rollout: Any, engine: Any) -> None:
    if not enabled(drafter):
        return
    training = drafter.get("training") or {}
    spec = engine.get("speculative_config") or {}
    if spec.get("method") == "extract_hidden_states":
        raise ValueError(
            "Direct rollout capture cannot use method=extract_hidden_states (#62411)"
        )
    connector = engine.get("kv_transfer_config") or {}
    if connector.get("kv_connector") == "ExampleHiddenStatesConnector":
        raise ValueError(
            "Direct rollout capture cannot use ExampleHiddenStatesConnector"
        )
    if not drafter.get("enable") or not drafter.get("enable_drafter_training"):
        raise ValueError("vLLM rollout capture requires drafter training enabled")
    if any(
        training.get(k)
        for k in (
            "collect_hidden_states_from_sgl",
            "collect_hidden_states_from_old_logprob",
            "use_logits",
        )
    ):
        raise ValueError(
            "vLLM rollout capture requires exclusive hidden collection and use_logits=false"
        )
    for key in (
        "pipeline_model_parallel_size",
        "prefill_context_parallel_size",
    ):
        if int(rollout.get(key, 1) or 1) != 1:
            raise ValueError(f"vLLM rollout capture requires {key}=1")
    if training.get("vllm_capture_reference", False) and not os.getenv(
        "SPECO_CAPTURE_EXPORT_ROOT"
    ):
        raise ValueError(
            "Reference capture requires explicit shared SPECO_CAPTURE_EXPORT_ROOT"
        )
    if engine.get("async_scheduling") is not False:
        raise ValueError(
            "vLLM rollout capture requires engine_kwargs.vllm.async_scheduling=false"
        )
    if rollout.get("enable_prefix_caching", True) is not False:
        raise ValueError(
            "vLLM rollout capture requires rollout.enable_prefix_caching=false"
        )
    if str(training.get("hidden_state_window_mode", "front")) not in {
        "front",
        "random",
    }:
        raise ValueError("vLLM rollout capture supports front/random windows")
    if int(training.get("hidden_state_window_tokens_per_sample", 512)) <= 0:
        raise ValueError("vLLM capture window must be positive and bounded")
    if (
        not 1
        <= int(training.get("hidden_state_window_min_rows", 1))
        <= int(training.get("hidden_state_window_tokens_per_sample", 512))
    ):
        raise ValueError(
            "vLLM capture min_rows must be positive and fit within the window"
        )
    if training.get("hidden_state_max_tokens_per_sample") is not None:
        logger.warning(
            "vLLM capture ignores hidden_state_max_tokens_per_sample; use hidden_state_window_tokens_per_sample"
        )
    if training.get("preserve_prompt_hidden_states"):
        raise ValueError(
            "vLLM rollout capture currently collects response prediction rows only"
        )


def confirmed_row_count(scheduled: int, drafts: int, count: int) -> int:
    if drafts:
        return max(0, min(scheduled, scheduled - drafts + max(count - 1, 0)))
    return scheduled


def filter_accepted_snapshot(
    hidden, token_ids, *, lo, start, scheduled, drafts, accepted
):
    """Optional device-side prefix compaction, outside the captured forward.

    Boolean indexing may synchronize for the dynamic result shape. Keep this
    explicitly opt-in until real CUDA-graph benchmarks justify the tradeoff.
    """
    import torch

    keep = (
        (scheduled - drafts + (accepted - 1).clamp(min=0)).clamp(min=0, max=scheduled)
        if drafts
        else scheduled
    )
    mask = torch.arange(lo, lo + hidden.shape[0], device=hidden.device) < start + keep
    return hidden[mask], token_ids[mask]


def make_plan(
    training: Any,
    *,
    prompt_len: int,
    max_tokens: int,
    request_id: str,
    global_step: int,
    replica_rank: int,
) -> dict[str, Any]:
    rows = min(
        int(training.get("hidden_state_window_tokens_per_sample", 512)), max_tokens
    )
    offset = 0
    if training.get("hidden_state_window_mode", "front") == "random":
        limit = min(
            int(training.get("hidden_state_random_max_offset") or max_tokens),
            max_tokens,
        )
        rows = min(rows, limit)
        seed = f"{request_id}:{replica_rank}:{global_step if training.get('hidden_state_random_seed_by_step', True) else ''}"
        offset = int.from_bytes(
            hashlib.blake2b(seed.encode(), digest_size=8).digest(), "big"
        ) % (max(limit - rows, 0) + 1)
    return {
        "window_store": bool(training.get("vllm_capture_reference", False)),
        "run_id": os.environ.get("SPECO_CAPTURE_RUN_ID"),
        "start": prompt_len - 1 + offset,
        "end": prompt_len - 1 + offset + rows,
        "min_rows": int(training.get("hidden_state_window_min_rows", 1)),
        "global_step": global_step,
        "replica_rank": replica_rank,
        "export_host": socket.gethostname(),
        "export_root": str(capture_root().resolve()),
        "shared_export": bool(os.getenv("SPECO_CAPTURE_EXPORT_ROOT")),
        "sample_id": uuid.uuid4().hex,
    }


def capture_root() -> Path:
    configured = os.getenv("SPECO_CAPTURE_EXPORT_ROOT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise ValueError(
                "SPECO_CAPTURE_EXPORT_ROOT must be an absolute shared directory"
            )
        return path
    return Path(tempfile.gettempdir()) / "verl-speco-rollout-capture"


def plan_directory(plan: dict[str, Any]) -> Path:
    sample_id = plan["sample_id"]
    if (
        not isinstance(sample_id, str)
        or len(sample_id) != 32
        or any(c not in "0123456789abcdef" for c in sample_id)
    ):
        raise ValueError("Invalid capture sample ID")
    return capture_root() / sample_id


def capture_sampled_rows(
    runner: Any, state: Any, sampler_output: Any, *, selected=None
) -> None:
    config = getattr(runner, "vllm_config", None)
    parallel = getattr(config, "parallel_config", None)
    tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
    if parallel is not None:
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
        ):
            if int(getattr(parallel, name, 1)) != 1:
                raise ValueError(f"vLLM capture requires {name}=1")
        if getattr(parallel, "use_ubatching", False):
            raise ValueError("vLLM capture does not support microbatching")
    if tp_size > 1:
        from vllm.distributed import get_tp_group

        # TP decoder outputs are replicated full-width activations. Export once
        # per TP group; no capture-specific collective may enter the sampler.
        if get_tp_group().rank_in_group != 0:
            return
    batch = runner.input_batch
    if os.getenv("SPECO_CAPTURE_DIAGNOSTICS") == "1":
        capture_detail(
            "runner_scan requests=%s scheduled=%s",
            len(batch.req_ids),
            state.scheduler_output.num_scheduled_tokens,
        )
    if selected is None:
        selected = []
        for index, request_id in enumerate(batch.req_ids):
            params = runner.requests[request_id].sampling_params
            plan = (getattr(params, "extra_args", None) or {}).get(CAPTURE_KEY)
            if not plan:
                continue
            start = int(batch.num_computed_tokens_cpu[index])
            scheduled = int(
                state.scheduler_output.num_scheduled_tokens.get(request_id, 0)
            )
            capture_detail(
                "runner_request request=%s sample=%s computed=%s scheduled=%s window=[%s,%s)",
                request_id,
                plan["sample_id"],
                start,
                scheduled,
                plan["start"],
                plan["end"],
            )
            if (
                scheduled > 0
                and start < int(plan["end"])
                and start + scheduled > int(plan["start"])
            ):
                selected.append((index, request_id, plan))
    if not selected:
        capture_detail("runner_skip reason=no_intersecting_capture_window")
        return
    # All DP replicas export their own requests. TP rank zero is local to each
    # DP replica; using global rank zero would lose every other replica's data.
    # Cross-host writers require an explicitly configured shared filesystem.
    root = str(capture_root().resolve())
    host = socket.gethostname()
    for _, _, plan in selected:
        if plan.get("export_root", root) != root:
            raise RuntimeError("Capture frontend/worker export roots do not match")
        remote = plan.get("export_host", host) != host
        if remote and not (
            plan.get("shared_export") and os.getenv("SPECO_CAPTURE_EXPORT_ROOT")
        ):
            raise RuntimeError(
                "Cross-host capture requires SPECO_CAPTURE_EXPORT_ROOT on a shared filesystem"
            )
        if remote and not capture_root().is_dir():
            raise RuntimeError(
                "Capture shared root is not visible on the export worker"
            )
    dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1))
    dp_size = int(getattr(parallel, "data_parallel_size", 1))
    if tp_size > 1 or dcp_size > 1 or dp_size > 1:
        width = int(runner.model_config.hf_config.hidden_size)
        for hidden in [state.hidden_states, *(state.aux_hidden_states or [])]:
            if hidden.ndim != 2 or hidden.shape[-1] != width:
                raise RuntimeError(
                    "Parallel capture requires replicated full-width hidden states"
                )
            if hidden.shape[0] < int(runner.query_start_loc.np[len(batch.req_ids)]):
                raise RuntimeError(
                    "Parallel capture does not support sequence-sharded hidden states"
                )
    if runner.use_async_scheduling:
        raise RuntimeError("vLLM capture does not support async scheduling")
    if not state.aux_hidden_states:
        raise RuntimeError("Target model did not return auxiliary hidden states")
    layer_ids = runner._get_eagle3_aux_layers_from_config()
    if layer_ids is None:
        layer_ids = runner.get_model().get_eagle3_default_aux_hidden_state_layers()
    layer_ids = list(layer_ids)
    if len(layer_ids) != len(state.aux_hidden_states):
        raise RuntimeError("Auxiliary layer metadata does not match model outputs")
    depth = getattr(runner.model_config.hf_config, "num_hidden_layers", None)
    if depth is None or any(
        type(layer) is not int or not 0 <= layer < depth for layer in layer_ids
    ):
        raise ValueError(
            "Capture aux layers must exclude the pre-norm final-layer output"
        )
    queue = export_queue()
    from contextlib import nullcontext

    group = (
        queue.window_batch()
        if any(plan.get("window_store") for _, _, plan in selected)
        else nullcontext(None)
    )
    with group as pending:
        _export_selected_rows(
            runner, state, sampler_output, selected, layer_ids, queue, pending
        )


def _export_selected_rows(
    runner, state, sampler_output, selected, layer_ids, queue, pending
):
    import torch

    batch = runner.input_batch
    timing_stride = int(os.getenv("SPECO_CAPTURE_GPU_TIMING_STRIDE", "0"))
    for index, request_id, plan in selected:
        directory = plan_directory(plan)
        # The frontend owns this directory. Cancellation removes the live marker.
        if (
            not (directory / "live").exists()
            or (directory / "error").exists()
            or (directory / "skipped").exists()
            # A recomputed request may revisit this range long after its
            # immutable training window has already been published.
            or (plan.get("window_store") and (directory / "window.ready").exists())
        ):
            continue
        begin = int(runner.query_start_loc.np[index])
        scheduled = int(state.scheduler_output.num_scheduled_tokens[request_id])
        drafts = len(
            state.scheduler_output.scheduled_spec_decode_tokens.get(request_id, [])
        )
        start = int(batch.num_computed_tokens_cpu[index])
        lo, hi = (
            max(start, int(plan["start"])),
            min(start + scheduled, int(plan["end"])),
        )
        if lo >= hi:
            continue
        selection = slice(begin + lo - start, begin + hi - start)
        capture_trace(
            "runner_selected request=%s sample=%s rows=[%s,%s) layers=%s layout=%s",
            request_id,
            plan["sample_id"],
            lo,
            hi,
            layer_ids,
            plan["layout"],
        )
        parts = [h[selection] for h in state.aux_hidden_states]
        if plan["layout"].endswith("plus_last"):
            parts.append(state.hidden_states[selection])
        # Reserve before allocating snapshots/pinned destinations. Admission is
        # nonblocking; an overloaded sample is explicitly discarded as a whole.
        # Include the GPU snapshot, pinned destination and writer-side compact
        # accepted-row payload, which can coexist during serialization.
        size = sum(h.numel() * h.element_size() for h in parts) * (
            5 if os.getenv("SPECO_CAPTURE_PACK_CHUNKS") == "1" else 3
        )
        size += (hi - lo) * (runner.input_ids.gpu.element_size() + 8) * 3
        size += 16  # owned device/host acceptance-count scalar
        if not queue.reserve(size):
            (directory / "skipped").touch()
            logger.warning(
                "Skipped capture sample %s: export capacity exhausted",
                plan["sample_id"],
            )
            continue
        done = None
        recycle_buffers = None
        stream = None
        gpu_start = gpu_copy_start = None
        try:
            snapshot_submit_started = time.perf_counter()
            if parts[0].is_cuda and (
                os.getenv("SPECO_CAPTURE_DIAGNOSTICS") == "1"
                or (
                    plan.get("window_store")
                    and timing_stride > 0
                    and next(_GPU_TIMING_SEQUENCE) % timing_stride == 0
                )
            ):
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_start.record(torch.cuda.current_stream(parts[0].device))
            # cat and clone run on the forward stream, so next graph execution
            # cannot overwrite their source before these snapshots are complete.
            snapshot = torch.cat(parts, dim=-1).detach()
            token_snapshot = runner.input_ids.gpu[selection].detach().clone()
            snapshot_host_s = time.perf_counter() - snapshot_submit_started
            sampled = sampler_output.sampled_token_ids[index]
            counts = getattr(sampler_output, "speco_num_sampled", None)
            count_snapshot = (
                counts[index].detach().clone()
                if counts is not None
                else ((sampled >= 0) & (sampled < batch.vocab_size)).sum()
            )
            if os.getenv("SPECO_CAPTURE_GPU_FILTER") == "1":
                snapshot, token_snapshot = filter_accepted_snapshot(
                    snapshot,
                    token_snapshot,
                    lo=lo,
                    start=start,
                    scheduled=scheduled,
                    drafts=drafts,
                    accepted=count_snapshot,
                )
            captured_rows = snapshot.shape[0]
            d2h_submit_started = time.perf_counter()
            if snapshot.is_cuda:
                from verl_speco.producer.capture_buffer_pool import CaptureBufferPool

                pool = getattr(runner, "_speco_capture_buffer_pool", None)
                if pool is None:
                    pool = runner._speco_capture_buffer_pool = CaptureBufferPool()
                ready = pool.acquire_event(snapshot.device, gpu_start is not None)
                ready.record(torch.cuda.current_stream(snapshot.device))
                stream = getattr(runner, "_speco_capture_copy_stream", None)
                if stream is None:
                    stream = runner._speco_capture_copy_stream = torch.cuda.Stream(
                        device=snapshot.device
                    )
                hidden = pool.acquire(snapshot)
                ids = pool.acquire(token_snapshot)
                count = pool.acquire(count_snapshot)
                owned_buffers = (hidden, ids, count)
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    if gpu_start is not None:
                        gpu_copy_start = torch.cuda.Event(enable_timing=True)
                        gpu_copy_start.record(stream)
                    hidden.copy_(snapshot, non_blocking=True)
                    ids.copy_(token_snapshot, non_blocking=True)
                    count.copy_(count_snapshot, non_blocking=True)
                    done = pool.acquire_event(snapshot.device, gpu_start is not None)
                    done.record(stream)
                snapshot.record_stream(stream)
                token_snapshot.record_stream(stream)
                count_snapshot.record_stream(stream)

                def recycle_buffers(
                    pool=pool,
                    buffers=owned_buffers,
                    device=snapshot.device,
                    events=(ready, done),
                    timing=gpu_start is not None,
                ):
                    pool.release(buffers)
                    pool.release_events(device, list(events), timing)
            else:
                hidden, ids = snapshot, token_snapshot
                count = count_snapshot
            payload = {
                "_capture_recycle": recycle_buffers,
                "_window_plan": plan if plan.get("window_store") else None,
                "capture_timing": {
                    "snapshot_copy_submit_host_s": time.perf_counter()
                    - snapshot_submit_started,
                    "snapshot_host_s": snapshot_host_s,
                    "d2h_submit_host_s": time.perf_counter() - d2h_submit_started
                    if snapshot.is_cuda
                    else 0.0,
                    "export_bytes": sum(
                        t.numel() * t.element_size() for t in (hidden, ids, count)
                    ),
                },
                "hidden_states": hidden,
                "token_ids": ids,
                "layer_ids": layer_ids,
                "layout": plan["layout"],
                "last_hidden_norm": "post_norm",
                "positions": torch.arange(lo, lo + captured_rows),
                "sample_id": plan["sample_id"],
            }

            def finish_copy(
                event=done,
                data=payload,
                accepted=count,
                offset=start,
                first_position=lo,
                fixed_window=bool(plan.get("window_store")),
                total=scheduled,
                draft_count=drafts,
                snapshot_start=gpu_start,
                snapshot_end=ready if snapshot.is_cuda else None,
                copy_start=gpu_copy_start,
                original_hidden=snapshot,
                original_ids=token_snapshot,
                original_count=count_snapshot,
                verify_transport=os.getenv("SPECO_CAPTURE_VERIFY_TRANSPORT") == "1",
            ):
                if event is not None:
                    event.synchronize()
                if verify_transport:
                    for value, original in (
                        (data["hidden_states"], original_hidden),
                        (data["token_ids"], original_ids),
                        (accepted, original_count),
                    ):
                        if not torch.equal(
                            value.contiguous().reshape(-1).view(torch.uint8),
                            original.detach()
                            .cpu()
                            .contiguous()
                            .reshape(-1)
                            .view(torch.uint8),
                        ):
                            raise RuntimeError(
                                "Runner snapshot and CPU export differ bitwise"
                            )
                    data["transport_snapshot_verified"] = True
                if snapshot_start is not None:
                    data["capture_timing"].update(
                        snapshot_gpu_s=snapshot_start.elapsed_time(snapshot_end) / 1000,
                        d2h_gpu_s=copy_start.elapsed_time(event) / 1000,
                    )
                # Read the scalar only after D2H completes, outside the runner.
                keep = confirmed_row_count(total, draft_count, int(accepted))
                if fixed_window:
                    # The CPU window writer finishes before the pinned buffers recycle.
                    # Positions form a contiguous prefix, so a slice needs no gather.
                    length = max(0, offset + keep - first_position)
                    for field in ("hidden_states", "token_ids", "positions"):
                        data[field] = data[field][:length]
                    return
                valid = data["positions"] < offset + keep
                for field in ("hidden_states", "token_ids", "positions"):
                    data[field] = data[field][valid]

            submit = (
                pending
                if plan.get("window_store") and pending is not None
                else queue.submit_reserved
            )
            submit(
                directory,
                payload,
                size=size,
                wait=finish_copy,
                owners=(snapshot, token_snapshot, count_snapshot, hidden, ids, count),
            )
            if os.getenv("SPECO_CAPTURE_DIAGNOSTICS") == "1" or logger.isEnabledFor(
                logging.DEBUG
            ):
                d2h_bytes = (
                    sum(t.numel() * t.element_size() for t in (hidden, ids, count))
                    if done is not None
                    else 0
                )
                capture_trace(
                    "vLLM capture chunk sample=%s window=[%s,%s) d2h_bytes=%s",
                    plan["sample_id"],
                    lo,
                    hi,
                    d2h_bytes,
                )
        except BaseException:
            # Copies may have been launched before a submission error. Wait
            # before dropping pinned buffers and releasing their reservation.
            try:
                if done is not None:
                    done.synchronize()
                elif stream is not None:
                    stream.synchronize()
            finally:
                if recycle_buffers is not None:
                    recycle_buffers()
                queue.release(size)
            if (directory / "live").exists():
                raise
            continue


def install_runner_capture() -> None:
    """Install before model loading in each vLLM worker process."""
    from importlib.metadata import version
    from packaging.version import Version

    installed = Version(version("vllm"))
    if installed < Version("0.22.0"):
        raise RuntimeError(
            "Rollout capture requires vLLM >=0.22.0 with a compatible V1 runner"
        )
    if installed.base_version != "0.22.0":
        logger.warning(
            "Rollout capture on vLLM %s: checking V1 runner capabilities; "
            "GPU numerical validation is required for this version",
            installed,
        )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from verl_speco.integration.vllm_rollout_capture_v2 import install_v2_capture

    install_v2_capture()
    if getattr(GPUModelRunner, "_speco_capture_installed", False):
        return
    missing = [
        name
        for name in ("sample_tokens", "_sample")
        if not callable(getattr(GPUModelRunner, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"Incompatible vLLM {installed} V1 runner: missing capture hooks {missing}"
        )
    original_sample_tokens = GPUModelRunner.sample_tokens
    original_sample = GPUModelRunner._sample
    original_init = GPUModelRunner.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        capture_trace(
            "runner_init class=%s version=%s", type(self).__qualname__, installed
        )
        config = self.vllm_config
        if (
            getattr(config.speculative_config, "method", None)
            == "extract_hidden_states"
        ):
            raise ValueError("Direct capture cannot use extractor mode")
        parallel = config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.use_ubatching:
            raise ValueError("vLLM capture requires PP1 without microbatching")
        if self.use_async_scheduling or config.cache_config.enable_prefix_caching:
            raise ValueError(
                "vLLM capture requires sync scheduling and no prefix cache"
            )
        if self.device.type != "cuda" or self.supports_mm_inputs:
            raise ValueError("vLLM capture requires CUDA and a text-only target")
        if not self.use_aux_hidden_state_outputs:
            raise ValueError("vLLM capture requires an auxiliary-hidden-state drafter")

    @functools.wraps(original_sample_tokens)
    def sample_tokens(self, *args, **kwargs):
        if not hasattr(self, "execute_model_state"):
            raise RuntimeError(
                "Incompatible vLLM V1 runner: missing execute_model_state"
            )
        state = self.execute_model_state
        if state is not None:
            required = ("scheduler_output", "hidden_states", "aux_hidden_states")
            missing = [name for name in required if not hasattr(state, name)]
            if missing:
                raise RuntimeError(
                    f"Incompatible vLLM {installed} capture execution state: missing {missing}"
                )
        self._speco_capture_state = state
        self._speco_capture_sample_called = False
        try:
            output = original_sample_tokens(self, *args, **kwargs)
            if state is not None and not self._speco_capture_sample_called:
                active = any(
                    (getattr(request.sampling_params, "extra_args", None) or {}).get(
                        CAPTURE_KEY
                    )
                    for request in self.requests.values()
                )
                if active:
                    raise RuntimeError(
                        f"Incompatible vLLM {installed}: sample_tokens bypassed the capture _sample hook"
                    )
            return output
        finally:
            self._speco_capture_state = None

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        self._speco_capture_sample_called = True
        result = original_sample(self, *args, **kwargs)
        state = getattr(self, "_speco_capture_state", None)
        if state is not None:
            try:
                capture_sampled_rows(self, state, result)
            except Exception:
                # Feature export must not take down unrelated engine requests.
                # Mark selected requests invalid; the frontend fails closed.
                logger.exception("vLLM rollout feature capture failed")
                for request_id in self.input_batch.req_ids:
                    params = self.requests[request_id].sampling_params
                    plan = (getattr(params, "extra_args", None) or {}).get(CAPTURE_KEY)
                    if plan:
                        try:
                            (plan_directory(plan) / "error").touch()
                        except OSError:
                            logger.warning("Could not persist capture failure marker")
        return result

    GPUModelRunner.sample_tokens = sample_tokens
    GPUModelRunner._sample = sample
    GPUModelRunner.__init__ = init
    GPUModelRunner._speco_capture_installed = True
    capture_trace("runner_hooks_installed version=%s file=%s", installed, __file__)


def load_sample(
    plan: dict[str, Any], prompt: list[int], response: list[int]
) -> dict[str, Any] | None:
    """Validate position coverage and tokens before entering the shared trainer."""
    import torch

    directory = plan_directory(plan)
    capture_trace(
        "load_begin sample=%s directory=%s prompt_tokens=%s response_tokens=%s",
        plan["sample_id"],
        directory,
        len(prompt),
        len(response),
    )
    from verl_speco.producer.feature_export import (
        wait_for_exports,
        rollout_feature_fields,
    )

    if (directory / "error").exists():
        raise RuntimeError("vLLM worker reported a hidden-state export failure")
    end = min(int(plan["end"]), len(prompt) + len(response) - 1)
    start = int(plan["start"])
    if end - start < int(plan["min_rows"]) or end <= start:
        capture_trace(
            "load_skip sample=%s reason=short_response rows=%s min_rows=%s",
            plan["sample_id"],
            end - start,
            plan["min_rows"],
        )
        return None
    wait_started = time.perf_counter()
    wait_for_exports(directory)
    export_tail_wait_s = time.perf_counter() - wait_started
    capture_trace(
        "vLLM capture export wait sample=%s seconds=%.6f",
        plan["sample_id"],
        time.perf_counter() - wait_started,
    )
    if (directory / "skipped").exists():
        return None
    if (directory / "error").exists():
        raise RuntimeError("vLLM worker reported a hidden-state export failure")
    from verl_speco.producer.feature_export import load_export_chunks

    read_started = time.perf_counter()
    if plan.get("window_store"):
        from verl_speco.producer.rollout_window_store import load_window

        window, _ = load_window(directory)
        metadata = window["metadata"]
        chunks = [
            {
                **window,
                "sample_id": metadata["sample_id"],
                "layout": metadata["layout"],
                "last_hidden_norm": metadata["norm"],
                "layer_ids": metadata["layer_ids"],
            }
        ]
        if metadata.get("transport_snapshot_verified"):
            from verl_speco.producer.feature_export import feature_digest

            chunks[0]["transport_snapshot_verified"] = True
            chunks[0]["transport_sha256"] = feature_digest(chunks[0])
    else:
        chunks = list(load_export_chunks(directory))
    if os.getenv("SPECO_CAPTURE_VERIFY_TRANSPORT") == "1":
        if not chunks or any(
            not c.get("transport_snapshot_verified") or not c.get("transport_sha256")
            for c in chunks
        ):
            raise RuntimeError(
                "Transport verification requested but chunk verification is missing"
            )
        print(
            f"SPECO_TRANSPORT_VERIFIED sample={plan['sample_id']} chunks={len(chunks)} bitwise=True",
            flush=True,
        )
    read_s = time.perf_counter() - read_started
    assemble_started = time.perf_counter()
    capture_trace("load_chunks sample=%s chunks=%s", plan["sample_id"], len(chunks))
    rows: dict[int, tuple[Any, int]] = {}
    layer_ids = None
    for chunk in chunks:
        if chunk["sample_id"] != plan["sample_id"]:
            raise RuntimeError("Capture sample identity mismatch")
        if (
            chunk["layout"] != plan["layout"]
            or chunk["last_hidden_norm"] != "post_norm"
        ):
            raise RuntimeError("Capture hidden layout/norm mismatch")
        if layer_ids is not None and layer_ids != chunk["layer_ids"]:
            raise RuntimeError("Capture auxiliary layers changed during request")
        layer_ids = chunk["layer_ids"]
        for p, h, token in zip(
            chunk["positions"].tolist(),
            chunk["hidden_states"],
            chunk["token_ids"].tolist(),
            strict=True,
        ):
            if p in rows:
                # Fail closed rather than silently choosing stale replay data.
                if rows[p][1] != token or not torch.equal(rows[p][0], h):
                    raise RuntimeError("Conflicting recomputed capture position")
            rows[p] = (h, token)
    tokens = prompt + response
    positions = list(range(start, end))
    if any(p not in rows or rows[p][1] != tokens[p] for p in positions):
        capture_trace(
            "load_alignment_failure sample=%s expected_rows=%s captured_rows=%s missing=%s mismatched=%s",
            plan["sample_id"],
            len(positions),
            len(rows),
            sum(p not in rows for p in positions),
            sum(p in rows and rows[p][1] != tokens[p] for p in positions),
        )
        raise RuntimeError(
            "vLLM rollout capture missing rows or token alignment mismatch"
        )
    sample = {
        **rollout_feature_fields(
            torch.tensor(prompt),
            torch.tensor(response),
            torch.stack([rows[p][0] for p in positions]),
            torch.tensor(positions),
        ),
        "hidden_position_start": start,
        "hidden_position_end": end,
        "hidden_states_layout": plan["layout"],
        "hidden_source": "vllm_rollout",
        "hidden_layer_ids": layer_ids,
        "last_hidden_state_norm": "post_norm",
        "global_step": plan["global_step"],
        "replica_rank": plan["replica_rank"],
        "_speco_sample_id": plan["sample_id"],
    }
    totals: dict[str, float] = {}
    for chunk in chunks:
        for key, value in chunk.get("capture_timing", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    capture_trace(
        "[vLLM capture timing] sample=%s step=%s chunks=%s rows=%s "
        "export_tail_wait_s=%.6f read_s=%.6f assemble_validate_s=%.6f "
        "chunk_work_sum=%s timing_mode=host_wall_no_new_device_sync",
        plan["sample_id"],
        plan["global_step"],
        len(chunks),
        len(positions),
        export_tail_wait_s,
        read_s,
        time.perf_counter() - assemble_started,
        totals,
    )
    return sample


def reclaim_cancelled_export(directory, *, plan=None):
    """Reclaim only after cancellation and the final writer relinquishes ownership."""
    directory = Path(directory)
    if (directory / "live").exists() or any(directory.glob("*.pending")):
        return
    budget = (plan or {}).get("reference_budget")
    metadata = directory / "window.meta"
    if metadata.exists():
        import json

        budget = json.loads(metadata.read_text()).get("budget", budget)
    shutil.rmtree(directory, ignore_errors=True)
    if budget is not None and not directory.exists():
        from verl_speco.integration.rollout_reference import _release

        _release({"budget": budget})


def cleanup(plan: dict[str, Any]) -> None:
    directory = plan_directory(plan)
    capture_trace("cleanup sample=%s directory=%s", plan["sample_id"], directory)
    (directory / "live").unlink(missing_ok=True)
    reclaim_cancelled_export(directory, plan=plan)


async def generate_with_capture(
    server: Any,
    generate: Any,
    args: tuple,
    kwargs: dict,
    drafter: dict,
    *,
    skip: bool = False,
    collection_step: int | None = None,
) -> Any:
    """Frontend adapter; expensive imports and file allocation are opt-in."""
    if not enabled(drafter):
        return await generate(*args, **kwargs)
    if skip:
        capture_trace(
            "vLLM capture skipped reason=request_skip step=%s replica=%s",
            server.global_steps,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    import inspect
    from verl_speco.trainer.scheduler import (
        DrafterCollectionContext,
        DrafterCollectionSource,
        DrafterScheduleConfig,
        DrafterScheduler,
    )
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
    )

    training = drafter["training"]
    serving_step = int(server.global_steps)
    step = serving_step if collection_step is None else int(collection_step)
    scheduler = DrafterScheduler()
    decision = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=step,
            source=DrafterCollectionSource.VLLM,
            drafter_enabled=True,
            source_enabled=True,
            validation=False,
        ),
        DrafterScheduleConfig.from_mapping(training),
    )
    if not decision.collect:
        capture_trace(
            "vLLM capture skipped reason=%s step=%s replica=%s",
            decision.reason,
            step,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    bound = inspect.signature(generate).bind(*args, **kwargs)
    values = bound.arguments
    prompt = values.get("prompt_ids")
    params = values.get("sampling_params")
    request_id = str(values.get("request_id", ""))
    if not isinstance(prompt, list) or not prompt or not isinstance(params, dict):
        raise RuntimeError(
            "Unsupported verl vLLM generate signature for rollout capture"
        )
    if int(params.get("n", 1)) != 1 or any(
        values.get(k) is not None for k in ("image_data", "video_data", "audio_data")
    ):
        raise ValueError("vLLM capture requires text-only n=1 requests")
    maximum = int(
        params.get(
            "max_tokens", params.get("max_new_tokens", server.config.response_length)
        )
    )
    plan = make_plan(
        training,
        prompt_len=len(prompt),
        max_tokens=maximum,
        request_id=request_id,
        global_step=step,
        replica_rank=int(server.replica_rank),
    )
    if plan["end"] - plan["start"] < plan["min_rows"]:
        capture_trace(
            "vLLM capture skipped reason=window_too_short request=%s step=%s planned_rows=%s min_rows=%s",
            request_id,
            step,
            plan["end"] - plan["start"],
            plan["min_rows"],
        )
        return await generate(*args, **kwargs)
    # Replica-local admission happens before any await, so concurrent requests
    # cannot exceed the configured per-step sample/token budgets.
    budget = getattr(server, "_speco_capture_budget", None)
    if budget is None or budget[0] != step:
        budget = server._speco_capture_budget = [step, 0, 0]
    sample_limit = int(training.get("max_collect_samples_per_step_per_replica", 16))
    token_limit = int(training.get("max_collect_tokens_per_step_per_replica", 16384))
    if (
        budget[1] == 0
        and getattr(server, "_speco_capture_budget_logged_step", None) != step
    ):
        logger.warning(
            "[vLLM capture budget] pid=%s step=%s serving_step=%s replica=%s "
            "max_samples=%s max_tokens=%s window_rows=%s sample_rate=%s",
            os.getpid(),
            step,
            serving_step,
            server.replica_rank,
            sample_limit,
            token_limit,
            plan["end"] - plan["start"],
            training.get("collection_sample_rate", 1.0),
        )
        server._speco_capture_budget_logged_step = step
    rows = plan["end"] - plan["start"]
    fraction = (
        int.from_bytes(
            hashlib.blake2b(request_id.encode(), digest_size=8).digest(), "big"
        )
        / 2**64
    )
    reason = None
    if sample_limit > 0 and budget[1] >= sample_limit:
        reason = "sample_budget"
    elif token_limit > 0 and budget[2] + rows > token_limit:
        reason = "token_budget"
    elif fraction >= float(training.get("collection_sample_rate", 1.0)):
        reason = "sample_rate"
    if reason:
        if getattr(server, "_speco_capture_budget_skip_logged_step", None) != step:
            logger.warning(
                "[vLLM capture budget_skip] step=%s replica=%s reason=%s "
                "authorized_samples=%s reserved_rows=%s max_samples=%s max_tokens=%s",
                step,
                server.replica_rank,
                reason,
                budget[1],
                budget[2],
                sample_limit,
                token_limit,
            )
            server._speco_capture_budget_skip_logged_step = step
        capture_trace(
            "vLLM capture skipped reason=%s request=%s step=%s replica=%s",
            reason,
            request_id,
            step,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    budget[1] += 1
    budget[2] += rows
    plan["target_version"] = serving_step
    plan["request_id"] = request_id
    plan["layout"] = resolve_drafter_hidden_states_layout(
        drafter["speculative_algorithm"], training
    )
    directory = plan_directory(plan)
    extra = dict(params.get("extra_args") or {})
    if CAPTURE_KEY in extra:
        raise ValueError("speco_rollout_capture is reserved for the rollout adapter")
    extra[CAPTURE_KEY] = plan
    values["sampling_params"] = {**params, "extra_args": extra}
    directory.mkdir(parents=True, exist_ok=False)
    reference_owned = False
    try:
        if plan.get("window_store"):
            from verl_speco.integration.rollout_reference import _reserve

            width = getattr(server, "_speco_window_widths", {}).get(plan["layout"])
            window_bytes = rows * (2 * width + 9) + 4096 if width else None
            reservation = _reserve(plan, window_bytes=window_bytes)
            if reservation is None:
                # Keep the already authorized sample on the original inline
                # exporter rather than creating two independent reservations.
                plan["window_store"] = False
                plan["reference_capacity_fallback"] = True
                logger.warning(
                    "[rollout reference] event=fallback reason=byte_capacity sample=%s",
                    plan["sample_id"],
                )
            else:
                plan["reference_budget"] = reservation
        (directory / "live").touch()
        capture_trace(
            "vLLM capture admitted request=%s sample=%s step=%s replica=%s window=[%s,%s) min_rows=%s layout=%s",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
            plan["start"],
            plan["end"],
            plan["min_rows"],
            plan["layout"],
        )
        output = await generate(*bound.args, **bound.kwargs)
        if getattr(output, "stop_reason", None) == "aborted":
            capture_trace(
                "vLLM capture skipped reason=aborted request=%s sample=%s",
                request_id,
                plan["sample_id"],
            )
            return output
        if int(server.global_steps) != serving_step:
            raise RuntimeError("Target serving version changed during capture")
        import asyncio

        if training.get("vllm_capture_reference", False):
            from verl_speco.integration.rollout_reference import submit

            reference = submit(
                plan, prompt, list(output.token_ids), request_id, serving_step
            )
            if reference is not None:
                import json

                ready = directory / "window.ready"
                if ready.is_file():
                    width = json.loads(ready.read_text())["width"]
                    server.__dict__.setdefault("_speco_window_widths", {})[
                        plan["layout"]
                    ] = width
                output.extra_fields["drafter_sample"] = reference
                reference_owned = True
                return output
        load_started = time.perf_counter()
        sample = await asyncio.to_thread(
            load_sample, plan, prompt, list(output.token_ids)
        )
        output.extra_fields["_speco_capture_skipped_capacity"] = int(
            (directory / "skipped").exists()
        )
        output.extra_fields["_speco_capture_rows"] = (
            int(sample["hidden_states"].shape[1]) if sample is not None else 0
        )
        capture_trace(
            "vLLM capture result request=%s sample=%s step=%s replica=%s reason=%s rows=%s response_tokens=%s load_seconds=%.6f",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
            "collected"
            if sample is not None
            else (
                "capacity"
                if output.extra_fields["_speco_capture_skipped_capacity"]
                else "short_response"
            ),
            output.extra_fields["_speco_capture_rows"],
            len(output.token_ids),
            time.perf_counter() - load_started,
        )
        if sample is not None:
            if training.get("vllm_capture_reference", False):
                sample["_speco_reference_capacity_fallback"] = bool(
                    plan.get("reference_capacity_fallback")
                )
                server.__dict__.setdefault("_speco_window_widths", {})[
                    plan["layout"]
                ] = int(sample["hidden_states"].shape[-1])
            sample["request_id"] = request_id
            sample["target_version"] = serving_step
            output.extra_fields["drafter_sample"] = sample
        return output
    except Exception:
        logger.exception(
            "vLLM capture failed request=%s sample=%s step=%s replica=%s",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
        )
        raise
    finally:
        if not reference_owned:
            cleanup(plan)


def finalize_rollout_sample(output, training):
    """Attach final request evidence before publishing to either consumer path."""
    import math

    fields = getattr(output, "extra_fields", None)
    if not isinstance(fields, dict):
        return
    sample = fields.get("drafter_sample")
    if isinstance(sample, dict) and sample.get("kind") == "speco_rollout_reference_v1":
        sample["evidence"] = {
            key: value
            for key, value in fields.items()
            if key.startswith(("_speco_", "_verl_"))
            and key
            not in {
                "_speco_sample_id",
                "_speco_global_sample_id",
                "_speco_direct_event_emitted",
            }
        }
        accept = fields.get("_verl_request_mean_accept_len")
        if isinstance(accept, (int, float)) and math.isfinite(accept):
            sample["evidence"]["_verl_accept_len"] = float(accept)
            sample["evidence"].setdefault("_verl_hard_score", -float(accept))
        return
    if not isinstance(sample, dict) or sample.get("hidden_source") != "vllm_rollout":
        return
    for key in (
        "_speco_vllm_request_id",
        "_speco_vllm_request_verify_rounds",
        "_speco_vllm_request_accepted_tokens",
        "_speco_drafter_revision_min",
        "_speco_drafter_revision_max",
        "_verl_is_hard",
        "_verl_hard_score",
    ):
        if fields.get(key) is not None:
            sample[key] = fields[key]
    accept = fields.get("_verl_request_mean_accept_len")
    if isinstance(accept, (int, float)) and math.isfinite(accept):
        sample["_verl_accept_len"] = float(accept)
        sample.setdefault("_verl_hard_score", -float(accept))
    if sample.get("_speco_direct_event_emitted"):
        return
    from verl_speco.integration.rollout_idle_events import emit_rollout_drafter_sample

    name = ((training.get("scheduler") or {}).get("idle_worker") or {}).get(
        "event_bus_name"
    )
    emitted = emit_rollout_drafter_sample(
        name,
        sample,
        sample_id=sample["_speco_sample_id"],
        replica_rank=sample["replica_rank"],
        global_step=sample["global_step"],
    )
    sample["_speco_direct_event_emitted"] = emitted
    capture_trace(
        "sample_attached sample=%s event_emitted=%s scored=%s rows=%s",
        sample["_speco_sample_id"],
        emitted,
        "_verl_accept_len" in sample,
        sample["hidden_states"].shape[1],
    )
