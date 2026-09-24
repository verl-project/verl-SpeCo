# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone vLLM target-feature Producer writing directly to TransferQueue."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Mapping

import torch

from verl_speco.config import config_int
from verl_speco.integration import transferqueue_bridge as default_transport
from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.producer.input_reader import (
    GenerationRequest,
    SampleFilteredError,
    TokenizedRequest,
    build_render_fn,
    iter_input_records,
    prepare_generation_request,
    prepare_generated_prefill_request,
    tokenize_record,
    tokenize_record_with_render_boundary,
)
from verl_speco.producer.vllm_feature_client import (
    RawVllmFeature,
    VllmEndpoint,
    VllmFeatureClientPool,
    delete_temporary_result,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.standalone_resume import load_standalone_resume
from verl_speco.trainer.target_feature_replay import (
    FeatureContract,
    HiddenStateAlignmentError,
    feature_from_vllm_payload,
    load_vllm_final_norm,
)
from verl_speco.transport.drafter_sample_protocol import (
    DRAFTER_TQ_PARTITION,
    PROTOCOL_SCHEMA_VERSION,
    SampleMetadata,
    encode_sample,
    is_ready_sample_tag,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
)


logger = logging.getLogger(__name__)
_INPUT_DONE = object()
_PUBLISH_DONE = object()
_FEATURE_CONVERSION_WORKERS = 8
_HEARTBEAT_INTERVAL_SECONDS = 60.0
_PERF_WINDOW_SAMPLES = 100


@dataclass
class ProducerStats:
    input_count: int = 0
    published_count: int = 0
    failed_count: int = 0
    dropped_count: int = 0
    filtered_count: int = 0
    pending_bytes: int = 0


@dataclass(frozen=True)
class PreparedFeature:
    request: TokenizedRequest
    raw: RawVllmFeature
    sample: DraftFeatureSample
    metadata: SampleMetadata
    timing: dict[str, float]


def _encode_publish_payload(
    result: PreparedFeature,
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any], float]:
    encode_started = time.monotonic()
    key = make_sample_key(result.metadata)
    fields = encode_sample(result.sample, result.metadata)
    tag = make_ready_tag(result.metadata)
    encode_seconds = time.monotonic() - encode_started
    return key, fields, tag, encode_seconds


def _put_sample_sync(
    transport: Any,
    key: str,
    fields: dict[str, torch.Tensor],
    tag: dict[str, Any],
) -> None:
    transport.put_sample(key, fields, tag=tag)


def _cleanup_result_sync(result: PreparedFeature) -> float:
    cleanup_started = time.monotonic()
    delete_temporary_result(result.raw)
    return time.monotonic() - cleanup_started


async def publish_one(
    result: PreparedFeature,
    transport: Any,
    *,
    executor: ThreadPoolExecutor | None = None,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 0.5,
) -> tuple[str, float, float, float, int, float]:
    """Publish one sample with bounded retries without blocking the event loop."""

    loop = asyncio.get_running_loop()
    key, fields, tag, encode_seconds = await loop.run_in_executor(
        executor, partial(_encode_publish_payload, result)
    )
    transport_seconds = 0.0
    retry_wait_seconds = 0.0
    for attempt in range(1, max_attempts + 1):
        transport_started = time.monotonic()
        try:
            await loop.run_in_executor(
                executor,
                partial(_put_sample_sync, transport, key, fields, tag),
            )
            transport_seconds += time.monotonic() - transport_started
            cleanup_seconds = await loop.run_in_executor(
                executor, partial(_cleanup_result_sync, result)
            )
            return (
                key,
                encode_seconds,
                transport_seconds,
                cleanup_seconds,
                attempt,
                retry_wait_seconds,
            )
        except Exception as exc:
            transport_seconds += time.monotonic() - transport_started
            if attempt >= max_attempts:
                raise
            delay = retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Standalone TQ Producer publish retry sample_id=%s sequence_no=%s "
                "attempt=%s/%s retry_in=%.3fs error=%r",
                result.request.sample_id,
                result.request.sequence_no,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            retry_wait_started = time.monotonic()
            await asyncio.sleep(delay)
            retry_wait_seconds += time.monotonic() - retry_wait_started
    raise AssertionError("unreachable")


def validate_producer_config(config: Any) -> None:
    producer_cfg, training_cfg, tq_cfg = _config_sections(config)
    required = (
        "input_path",
        "tokenizer_path",
        "tokenizer_fingerprint",
        "target_model_id",
        "target_model_revision",
        "vllm_model",
    )
    missing = [name for name in required if not producer_cfg.get(name)]
    if missing:
        raise ValueError(f"standalone_tq_producer missing required fields: {missing}")
    endpoints = producer_cfg.get("vllm_endpoints")
    if not isinstance(endpoints, list) or not endpoints or not all(endpoints):
        raise ValueError(
            "standalone_tq_producer.vllm_endpoints must be a non-empty list"
        )
    target_layer_ids = producer_cfg.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError(
            "standalone_tq_producer.target_layer_ids must be a non-empty list"
        )
    algorithm = str(training_cfg.get("speculative_algorithm", "") or "").strip()
    if not algorithm:
        raise ValueError("drafter.speculative_algorithm must not be empty")
    if bool(training_cfg.get("use_logits", False)):
        raise ValueError("Standalone TQ Producer does not support use_logits=true")
    if int(tq_cfg.get("schema_version", 0)) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"transfer_queue.schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )
    if tq_cfg.get("package_version") != "0.1.10":
        raise ValueError("transfer_queue.package_version must be '0.1.10'")
    if tq_cfg.get("partition_id") != DRAFTER_TQ_PARTITION:
        raise ValueError(
            f"transfer_queue.partition_id must be {DRAFTER_TQ_PARTITION!r}"
        )
    if not tq_cfg.get("run_id"):
        raise ValueError("transfer_queue.run_id must not be empty")
    ray_cfg = tq_cfg.get("ray") or {}
    if not isinstance(ray_cfg, Mapping) or not ray_cfg.get("address"):
        raise ValueError(
            "transfer_queue.ray.address must point to a running Ray cluster"
        )
    positive_fields = (
        "max_inflight_requests",
        "per_endpoint_concurrency",
        "input_queue_size",
        "publish_queue_size",
        "max_pending_samples",
        "generation_max_tokens",
    )
    invalid = [name for name in positive_fields if int(producer_cfg.get(name, 0)) <= 0]
    if invalid:
        raise ValueError(f"standalone_tq_producer fields must be positive: {invalid}")
    if int(producer_cfg.get("publish_workers", 4)) <= 0:
        raise ValueError("standalone_tq_producer.publish_workers must be positive")
    if int(producer_cfg.get("publish_max_attempts", 3)) <= 0:
        raise ValueError("standalone_tq_producer.publish_max_attempts must be positive")
    if float(producer_cfg.get("publish_retry_backoff_seconds", 0.5)) < 0:
        raise ValueError(
            "standalone_tq_producer.publish_retry_backoff_seconds must be non-negative"
        )
    if int(producer_cfg.get("vllm_success_log_interval", 100)) < 0:
        raise ValueError("vllm_success_log_interval must be non-negative")


def _should_log_sample_progress(count: int) -> bool:
    return count <= 3 or count % 50 == 0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _timing_value(timing: Mapping[str, float], name: str) -> float:
    return max(float(timing.get(name, 0.0)), 0.0)


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _sample_bytes(sample: DraftFeatureSample) -> int:
    return sum(
        _tensor_bytes(getattr(sample, name))
        for name in (
            "input_ids",
            "loss_mask",
            "hidden_states",
            "last_hidden_states",
            "target",
            "target_logprobs",
            "position_ids",
        )
    )


def _format_perf_sample(row: Mapping[str, Any]) -> str:
    timing = row["timing"]
    return (
        f"sample_id={row['sample_id']} sequence_no={row['sequence_no']} "
        f"endpoint={row['endpoint']} prompt_tokens={row['prompt_tokens']} "
        f"generated_tokens={row['generated_tokens']} hidden_state_mb={row['mb']:.2f} "
        f"e2e={_timing_value(timing, 'e2e'):.3f}s "
        f"input_prepare={_timing_value(timing, 'input_prepare'):.3f}s "
        f"input_queue={_timing_value(timing, 'input_queue'):.3f}s "
        f"scheduler_wait={_timing_value(timing, 'scheduler_wait'):.3f}s "
        f"tq_capacity={_timing_value(timing, 'tq_capacity'):.3f}s "
        f"generate={_timing_value(timing, 'generate'):.3f}s "
        f"prefill={_timing_value(timing, 'prefill'):.3f}s "
        f"feature_slot={_timing_value(timing, 'feature_slot'):.3f}s "
        f"conversion={_timing_value(timing, 'conversion'):.3f}s "
        f"publish_queue={_timing_value(timing, 'publish_queue'):.3f}s "
        f"tq_encode={_timing_value(timing, 'tq_encode'):.3f}s "
        f"tq_transport={_timing_value(timing, 'tq_transport'):.3f}s "
        f"tq_attempts={int(_timing_value(timing, 'tq_attempts'))} "
        f"tq_retry_wait={_timing_value(timing, 'tq_retry_wait'):.3f}s "
        f"temp_cleanup={_timing_value(timing, 'temp_cleanup'):.3f}s "
        f"active={_timing_value(timing, 'active'):.3f}s "
        f"backpressure={_timing_value(timing, 'backpressure'):.3f}s"
    )


async def run_producer(
    config: Any,
    *,
    transport: Any = default_transport,
    tokenizer: Any | None = None,
    client_pool: Any | None = None,
    before_request: Any | None = None,
    on_published: Any | None = None,
    get_runtime_state: Any | None = None,
) -> ProducerStats:
    """Run the bounded input -> vLLM -> TQ pipeline and publish EOS on success."""

    validate_producer_config(config)
    producer_cfg, drafter_cfg, tq_cfg = _config_sections(config)
    run_id = str(tq_cfg["run_id"])
    stats = ProducerStats()
    max_consecutive_feature_drops = config_int(
        producer_cfg, "max_consecutive_feature_drops", 20
    )
    if max_consecutive_feature_drops < 0:
        raise ValueError("max_consecutive_feature_drops must be >= 0")
    connected = False
    completed = False
    pool = client_pool
    feature_executor: ThreadPoolExecutor | None = None
    publish_executor: ThreadPoolExecutor | None = None
    try:
        logger.info(
            "Standalone TQ Producer starting run_id=%s input=%s endpoints=%s",
            run_id,
            producer_cfg["input_path"],
            producer_cfg["vllm_endpoints"],
        )
        if not transport.configure_transfer_queue(tq_cfg):
            raise RuntimeError("Standalone TQ Producer requires TransferQueue==0.1.10")
        ray_cfg = tq_cfg["ray"]
        logger.info(
            "Standalone TQ Producer connecting Ray address=%s namespace=%s",
            ray_cfg["address"],
            ray_cfg.get("namespace"),
        )
        transport.connect_ray_cluster(
            str(ray_cfg["address"]),
            str(ray_cfg["namespace"]) if ray_cfg.get("namespace") else None,
        )
        logger.info("Standalone TQ Producer connected Ray; initializing TQ client")
        transport.connect_transfer_queue_client()
        connected = True
        logger.info("Standalone TQ Producer connected TQ; waiting for owner_ready")
        await _wait_for_owner_ready(
            transport,
            run_id,
            timeout=float(producer_cfg["owner_ready_timeout_seconds"]),
            poll_interval=float(producer_cfg["pending_poll_interval_seconds"]),
        )
        logger.info("Standalone TQ Producer observed owner_ready run_id=%s", run_id)

        consumed_sequence_nos, resume_metadata = load_standalone_resume(
            producer_cfg.get("resume_checkpoint_path"),
            input_path=str(producer_cfg["input_path"]),
        )
        logger.info(
            "Standalone TQ Producer resume progress checkpoint=%s consumed=%s step=%s",
            producer_cfg.get("resume_checkpoint_path"),
            len(consumed_sequence_nos),
            None if resume_metadata is None else resume_metadata.get("optimizer_step"),
        )

        if tokenizer is None:
            logger.info(
                "Standalone TQ Producer loading tokenizer path=%s",
                producer_cfg["tokenizer_path"],
            )
            tokenizer = await asyncio.to_thread(_load_tokenizer, producer_cfg)
            logger.info("Standalone TQ Producer tokenizer loaded")
        if pool is None:
            endpoint_concurrency = int(producer_cfg["per_endpoint_concurrency"])
            pool = VllmFeatureClientPool(
                [
                    VllmEndpoint(str(url).rstrip("/"), endpoint_concurrency)
                    for url in producer_cfg["vllm_endpoints"]
                ],
                model=str(producer_cfg["vllm_model"]),
                max_inflight_requests=int(producer_cfg["max_inflight_requests"]),
                request_timeout=float(producer_cfg["request_timeout"]),
                success_log_interval=int(
                    producer_cfg.get("vllm_success_log_interval", 100)
                ),
            )
        await pool.start()
        logger.info("Standalone TQ Producer vLLM client pool started")

        algorithm = str(drafter_cfg["speculative_algorithm"]).strip().upper()
        feature_contract = FeatureContract(
            algorithm=algorithm,
            target_layer_ids=[int(value) for value in producer_cfg["target_layer_ids"]],
            hidden_states_layout=resolve_drafter_hidden_states_layout(
                algorithm, drafter_cfg
            ),
            dtype=_parse_dtype(producer_cfg["hidden_dtype"]),
            target_model_id=str(producer_cfg["target_model_id"]),
            target_model_revision=str(producer_cfg["target_model_revision"]),
            tokenizer_fingerprint=str(producer_cfg["tokenizer_fingerprint"]),
            use_logits=False,
            require_full_alignment=True,
        )
        final_norm = None
        if feature_contract.hidden_states_layout.endswith("_plus_last"):
            final_norm = await asyncio.to_thread(
                load_vllm_final_norm,
                feature_contract.target_model_id,
                dtype=feature_contract.dtype,
                trust_remote_code=bool(producer_cfg.get("trust_remote_code", False)),
            )
            target_num_hidden_layers = getattr(
                final_norm, "_speco_target_num_hidden_layers", None
            )
            if target_num_hidden_layers is not None:
                feature_contract = replace(
                    feature_contract,
                    target_num_hidden_layers=int(target_num_hidden_layers),
                )
        feature_executor = ThreadPoolExecutor(
            max_workers=_FEATURE_CONVERSION_WORKERS,
            thread_name_prefix="speco-feature",
        )
        publish_workers = int(producer_cfg.get("publish_workers", 4))
        publish_executor = ThreadPoolExecutor(
            max_workers=publish_workers,
            thread_name_prefix="speco-publish",
        )
        logger.info(
            "Standalone TQ Producer pipeline workers requests=%s features=%s "
            "publishers=%s",
            int(producer_cfg["max_inflight_requests"]),
            _FEATURE_CONVERSION_WORKERS,
            publish_workers,
        )
        feature_slots = asyncio.Semaphore(_FEATURE_CONVERSION_WORKERS)
        event_loop = asyncio.get_running_loop()
        worker_count = int(producer_cfg["max_inflight_requests"])
        input_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=int(producer_cfg["input_queue_size"])
        )
        publish_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=int(producer_cfg["publish_queue_size"])
        )
        stages: dict[str, tuple[str, float, str]] = {}
        sample_timings: dict[int, dict[str, float]] = {}
        perf_rows: list[dict[str, Any]] = []
        perf_window_started = time.monotonic()
        peak_publish_queue = 0
        publish_inflight = 0
        peak_publish_inflight = 0
        peak_pending_bytes = 0
        last_published_at = time.monotonic()
        producer_started_at = last_published_at
        max_samples = int(producer_cfg.get("max_samples", 0) or 0)

        def mark_stage(worker: str, stage: str, sample_id: str = "") -> None:
            stages[worker] = (stage, time.monotonic(), sample_id)

        async def log_heartbeat() -> None:
            last_inputs = 0
            last_published = 0
            last_heartbeat_at = producer_started_at
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
                now = time.monotonic()
                elapsed = max(now - last_heartbeat_at, 1e-9)
                oldest = sorted(
                    (
                        (now - started, worker, stage, sample_id)
                        for worker, (stage, started, sample_id) in stages.items()
                    ),
                    reverse=True,
                )[:3]
                logger.info(
                    "Standalone TQ Producer heartbeat state=%s inputs=%s "
                    "published=%s dropped=%s input_rate=%.2f/s publish_rate=%.2f/s "
                    "input_queue=%s/%s publish_queue=%s/%s pending_bytes=%s "
                    "seconds_since_publish=%.0f stages=%s oldest=%s",
                    get_runtime_state() if get_runtime_state is not None else "running",
                    stats.input_count,
                    stats.published_count,
                    stats.dropped_count,
                    (stats.input_count - last_inputs) / elapsed,
                    (stats.published_count - last_published) / elapsed,
                    input_queue.qsize(),
                    input_queue.maxsize,
                    publish_queue.qsize(),
                    publish_queue.maxsize,
                    stats.pending_bytes,
                    now - last_published_at,
                    dict(Counter(stage for stage, _, _ in stages.values())),
                    [
                        (worker, stage, sample_id, round(age))
                        for age, worker, stage, sample_id in oldest
                    ],
                )
                last_inputs = stats.input_count
                last_published = stats.published_count
                last_heartbeat_at = now

        def iter_requests():
            epoch = 0
            source_sequence_no = 0
            render_boundary_cfg = producer_cfg.get("render_boundary", {}) or {}
            render_fn = None
            if bool(render_boundary_cfg.get("enabled", False)):
                endpoints = list(producer_cfg.get("vllm_endpoints") or [])
                if not endpoints:
                    raise ValueError(
                        "render_boundary.enabled requires a vllm_endpoints entry"
                    )
                render_fn = build_render_fn(
                    str(endpoints[0]),
                    timeout=float(render_boundary_cfg.get("timeout", 10.0) or 10.0),
                )
                logger.info(
                    "Standalone TQ Producer using render-boundary loss masks "
                    "endpoint=%s",
                    endpoints[0],
                )
            while True:
                scanned_count = 0
                considered_count = 0
                produced_count = 0
                for source_record in iter_input_records(
                    str(producer_cfg["input_path"]),
                    on_error=str(producer_cfg.get("on_error", "skip") or "skip"),
                    max_consecutive_errors=config_int(
                        producer_cfg, "max_consecutive_errors", 20
                    ),
                    parser_strict_roles=bool(
                        producer_cfg.get("parser_strict_roles", False)
                    ),
                ):
                    sequence_no = source_sequence_no
                    source_sequence_no += 1
                    scanned_count += 1
                    if sequence_no in consumed_sequence_nos:
                        # Already consumed by a previous run; not a filtered row.
                        continue
                    considered_count += 1
                    # iter_input_records restarts sequence_no at zero on every
                    # pass. TQ keys require a run-global sequence number so a
                    # repeated sample never overwrites an earlier pending copy.
                    record = replace(
                        source_record,
                        sequence_no=sequence_no,
                    )
                    try:
                        if record.response is None:
                            request = prepare_generation_request(
                                record, tokenizer, producer_cfg
                            )
                        elif render_fn is not None:
                            try:
                                request = tokenize_record_with_render_boundary(
                                    record, tokenizer, producer_cfg, render_fn
                                )
                            except SampleFilteredError:
                                raise
                            except Exception as exc:  # noqa: BLE001 - fall back locally
                                logger.warning(
                                    "Render-boundary tokenization failed for %s (%s); "
                                    "falling back to the local tokenizer",
                                    record.sample_id,
                                    exc,
                                )
                                request = tokenize_record(
                                    record, tokenizer, producer_cfg
                                )
                        else:
                            request = tokenize_record(record, tokenizer, producer_cfg)
                    except SampleFilteredError as exc:
                        stats.filtered_count += 1
                        logger.warning(
                            "Standalone TQ Producer filtered sample "
                            "sequence_no=%s sample_id=%s filtered=%s reason=%s",
                            sequence_no,
                            record.sample_id,
                            stats.filtered_count,
                            exc,
                        )
                        continue
                    produced_count += 1
                    yield request
                if scanned_count == 0:
                    raise ValueError("Standalone TQ Producer input contains no samples")
                if produced_count == 0 and considered_count > 0 and max_samples > 0:
                    # Only rows that were not already consumed count: an epoch
                    # that is entirely resume-skipped must advance to the next
                    # one instead of aborting the run.
                    raise ValueError(
                        "Standalone TQ Producer filtered every newly considered "
                        f"sample (scanned={scanned_count}, "
                        f"resumed={scanned_count - considered_count}) but still "
                        f"needs max_samples={max_samples}; refusing to rescan "
                        "the input forever"
                    )
                if max_samples <= 0:
                    return
                epoch += 1
                logger.info(
                    "Standalone TQ Producer restarting input epoch=%s attempts=%s "
                    "target_published=%s",
                    epoch,
                    stats.input_count,
                    max_samples,
                )

        requests = iter(iter_requests())
        # Tokenization (and the synchronous vLLM /render calls it can make when
        # render-boundary is enabled) runs inside this generator. Advance it on a
        # worker thread so a slow render never blocks the event loop, which also
        # drives the other workers, the publisher and the heartbeat.
        request_generation_lock = asyncio.Lock()

        def next_request() -> Any:
            input_started = time.monotonic()
            try:
                request = next(requests)
            except StopIteration:
                # ``StopIteration`` cannot be raised across the thread/future
                # boundary used by ``asyncio.to_thread``; signal exhaustion
                # with a sentinel instead.
                return _INPUT_DONE
            sample_timings[int(request.sequence_no)] = {
                "e2e_started": input_started,
                "input_prepare": time.monotonic() - input_started,
            }
            stats.input_count += 1
            if _should_log_sample_progress(stats.input_count):
                logger.info(
                    "Standalone TQ Producer queued input count=%s sample_id=%s",
                    stats.input_count,
                    request.sample_id,
                )
            return request

        async def next_request_async() -> Any:
            async with request_generation_lock:
                return await asyncio.to_thread(next_request)

        async def read_inputs() -> None:
            initial_count = 0
            while max_samples <= 0 or initial_count < max_samples:
                request = await next_request_async()
                if request is _INPUT_DONE:
                    break
                mark_stage("input", "input_queue_put", request.sample_id)
                sample_timings[int(request.sequence_no)]["input_queue_started"] = (
                    time.monotonic()
                )
                await input_queue.put(request)
                initial_count += 1
            for _ in range(worker_count):
                await input_queue.put(_INPUT_DONE)
            logger.info(
                "Standalone TQ Producer input exhausted total=%s", stats.input_count
            )
            stages.pop("input", None)

        async def request_worker() -> None:
            nonlocal peak_pending_bytes, peak_publish_queue
            current = asyncio.current_task()
            worker = current.get_name() if current is not None else "request-unknown"
            replacement_request = None
            consecutive_replacements = 0
            while True:
                if replacement_request is None:
                    mark_stage(worker, "input_queue_get")
                    request = await input_queue.get()
                else:
                    request = replacement_request
                    replacement_request = None
                if request is _INPUT_DONE:
                    stages.pop(worker, None)
                    return
                timing = sample_timings[int(request.sequence_no)]
                now = time.monotonic()
                input_queue_started = timing.pop("input_queue_started", now)
                timing["input_queue"] = now - input_queue_started
                if before_request is not None:
                    scheduler_wait_started = time.monotonic()
                    if (
                        get_runtime_state is not None
                        and get_runtime_state() == "paused"
                    ):
                        mark_stage(worker, "scheduler_paused", request.sample_id)
                    await before_request()
                    timing["scheduler_wait"] = time.monotonic() - scheduler_wait_started
                mark_stage(worker, "tq_capacity", request.sample_id)
                capacity_started = time.monotonic()
                # The scheduled Ray path applies high/low-watermark control in
                # the Driver. Keep legacy polling only for the subprocess path.
                if before_request is None:
                    await _wait_for_pending_capacity(
                        transport,
                        run_id,
                        max_pending_samples=int(producer_cfg["max_pending_samples"]),
                        poll_interval=float(
                            producer_cfg["pending_poll_interval_seconds"]
                        ),
                    )
                timing["tq_capacity"] = time.monotonic() - capacity_started
                if _should_log_sample_progress(int(request.sequence_no) + 1):
                    logger.info(
                        "Standalone TQ Producer requesting vLLM sequence_no=%s "
                        "sample_id=%s mode=%s",
                        request.sequence_no,
                        request.sample_id,
                        "generate_then_prefill"
                        if isinstance(request, GenerationRequest)
                        else "prefill",
                    )
                if isinstance(request, GenerationRequest):
                    mark_stage(worker, "vllm_generate", request.sample_id)
                    generate_started = time.monotonic()
                    generated = await pool.generate(request)
                    timing["generate"] = time.monotonic() - generate_started
                    try:
                        try:
                            request = prepare_generated_prefill_request(
                                request,
                                generated.generated_token_ids,
                                producer_cfg,
                            )
                        except SampleFilteredError as exc:
                            stats.filtered_count += 1
                            consecutive_replacements += 1
                            logger.warning(
                                "Standalone TQ Producer filtered generated sample "
                                "sequence_no=%s sample_id=%s filtered=%s "
                                "consecutive=%s/%s reason=%s",
                                request.sequence_no,
                                request.sample_id,
                                stats.filtered_count,
                                consecutive_replacements,
                                max_consecutive_feature_drops,
                                exc,
                            )
                            if (
                                max_samples > 0
                                and consecutive_replacements
                                > max_consecutive_feature_drops
                            ):
                                raise RuntimeError(
                                    "Standalone TQ Producer exceeded "
                                    "max_consecutive_feature_drops="
                                    f"{max_consecutive_feature_drops} without a "
                                    "successful feature conversion; aborting "
                                    "instead of requesting replacement samples "
                                    "forever"
                                ) from exc
                            if max_samples > 0:
                                replacement_request = await next_request_async()
                            sample_timings.pop(int(request.sequence_no), None)
                            continue
                    finally:
                        # The generation request may still produce a prompt-only
                        # connector file. It is not the training payload; the
                        # following full-sequence prefill produces that payload.
                        await asyncio.to_thread(delete_temporary_result, generated)
                    mark_stage(worker, "vllm_prefill", request.sample_id)
                    prefill_started = time.monotonic()
                    raw = await pool.prefill(request)
                    timing["prefill"] = time.monotonic() - prefill_started
                else:
                    mark_stage(worker, "vllm_prefill", request.sample_id)
                    prefill_started = time.monotonic()
                    raw = await pool.prefill(request)
                    timing["prefill"] = time.monotonic() - prefill_started
                stats.pending_bytes += int(raw.byte_size)
                peak_pending_bytes = max(peak_pending_bytes, stats.pending_bytes)
                try:
                    mark_stage(worker, "feature_conversion", request.sample_id)
                    feature_slot_started = time.monotonic()
                    async with feature_slots:
                        timing["feature_slot"] = time.monotonic() - feature_slot_started
                        conversion_started = time.monotonic()
                        sample = await event_loop.run_in_executor(
                            feature_executor,
                            partial(
                                feature_from_vllm_payload,
                                raw,
                                request,
                                feature_contract,
                                final_norm=final_norm,
                            ),
                        )
                        timing["conversion"] = time.monotonic() - conversion_started
                except HiddenStateAlignmentError as exc:
                    stats.dropped_count += 1
                    consecutive_replacements += 1
                    stats.pending_bytes = max(
                        stats.pending_bytes - int(raw.byte_size), 0
                    )
                    await asyncio.to_thread(delete_temporary_result, raw)
                    logger.warning(
                        "Standalone TQ Producer dropped misaligned sample "
                        "sequence_no=%s sample_id=%s dropped=%s consecutive=%s/%s "
                        "reason=%s",
                        request.sequence_no,
                        request.sample_id,
                        stats.dropped_count,
                        consecutive_replacements,
                        max_consecutive_feature_drops,
                        exc,
                    )
                    if (
                        max_samples > 0
                        and consecutive_replacements > max_consecutive_feature_drops
                    ):
                        raise RuntimeError(
                            "Standalone TQ Producer exceeded "
                            "max_consecutive_feature_drops="
                            f"{max_consecutive_feature_drops} without a successful "
                            "feature conversion; aborting instead of requesting "
                            "replacement samples forever"
                        ) from exc
                    if max_samples > 0:
                        replacement_request = await next_request_async()
                        logger.info(
                            "Standalone TQ Producer replacing dropped sample "
                            "with sequence_no=%s sample_id=%s",
                            replacement_request.sequence_no,
                            replacement_request.sample_id,
                        )
                    sample_timings.pop(int(request.sequence_no), None)
                    continue
                # A successful feature conversion resets the consecutive-replacement
                # circuit breaker (feature drops and filtered generations).
                consecutive_replacements = 0
                mark_stage(worker, "publish_queue_put", request.sample_id)
                timing["publish_queue_started"] = time.monotonic()
                await publish_queue.put(
                    PreparedFeature(
                        request=request,
                        raw=raw,
                        sample=sample,
                        metadata=_sample_metadata(
                            request, sample, feature_contract, run_id, tq_cfg
                        ),
                        timing=timing,
                    )
                )
                peak_publish_queue = max(peak_publish_queue, publish_queue.qsize())

        def log_perf_window() -> None:
            nonlocal perf_window_started, peak_publish_queue
            nonlocal peak_publish_inflight, peak_pending_bytes
            if len(perf_rows) < _PERF_WINDOW_SAMPLES:
                return
            now = time.monotonic()
            window = max(now - perf_window_started, 1e-9)
            timing_names = (
                "e2e",
                "input_prepare",
                "input_queue",
                "scheduler_wait",
                "tq_capacity",
                "generate",
                "prefill",
                "feature_slot",
                "conversion",
                "publish_queue",
                "tq_encode",
                "tq_transport",
                "tq_retry_wait",
                "temp_cleanup",
                "active",
                "backpressure",
            )
            averages = {
                name: sum(_timing_value(row["timing"], name) for row in perf_rows)
                / len(perf_rows)
                for name in timing_names
            }
            e2e = [_timing_value(row["timing"], "e2e") for row in perf_rows]
            transport_times = [
                _timing_value(row["timing"], "tq_transport") for row in perf_rows
            ]
            total_mb = sum(float(row["mb"]) for row in perf_rows)
            logger.info(
                "Standalone TQ Producer perf samples=%s window=%.3fs rate=%.2f/s "
                "e2e_avg=%.3fs e2e_p95=%.3fs e2e_max=%.3fs "
                "input_prepare_avg=%.3fs input_queue_avg=%.3fs "
                "scheduler_wait_avg=%.3fs tq_capacity_avg=%.3fs "
                "generate_avg=%.3fs prefill_avg=%.3fs "
                "feature_slot_avg=%.3fs conversion_avg=%.3fs "
                "publish_queue_avg=%.3fs tq_encode_avg=%.3fs "
                "tq_transport_avg=%.3fs tq_transport_p95=%.3fs "
                "tq_transport_max=%.3fs temp_cleanup_avg=%.3fs "
                "tq_retry_wait_avg=%.3fs tq_retried_samples=%s "
                "active_avg=%.3fs backpressure_avg=%.3fs hidden_state_mb=%.2f "
                "hidden_state_mb_per_second=%.2f tq_transport_mb_per_second=%.2f "
                "prefill_tokens=%s prefill_tokens_per_second=%.2f "
                "publish_queue_peak=%s "
                "publish_inflight_peak=%s pending_mb_peak=%.2f",
                len(perf_rows),
                window,
                len(perf_rows) / window,
                sum(e2e) / len(e2e),
                _percentile(e2e, 0.95),
                max(e2e),
                averages["input_prepare"],
                averages["input_queue"],
                averages["scheduler_wait"],
                averages["tq_capacity"],
                averages["generate"],
                averages["prefill"],
                averages["feature_slot"],
                averages["conversion"],
                averages["publish_queue"],
                averages["tq_encode"],
                averages["tq_transport"],
                _percentile(transport_times, 0.95),
                max(transport_times),
                averages["temp_cleanup"],
                averages["tq_retry_wait"],
                sum(
                    1
                    for row in perf_rows
                    if _timing_value(row["timing"], "tq_attempts") > 1
                ),
                averages["active"],
                averages["backpressure"],
                total_mb,
                total_mb / window,
                total_mb / max(sum(transport_times), 1e-9),
                sum(int(row["prompt_tokens"]) for row in perf_rows),
                sum(int(row["prompt_tokens"]) for row in perf_rows) / window,
                peak_publish_queue,
                peak_publish_inflight,
                peak_pending_bytes / (1024 * 1024),
            )
            ordered_by_sequence = sorted(
                perf_rows, key=lambda row: int(row["sequence_no"])
            )
            middle_index = max((len(ordered_by_sequence) // 2) - 1, 0)
            middle = ordered_by_sequence[middle_index : middle_index + 2]
            slowest_e2e = sorted(
                perf_rows,
                key=lambda row: _timing_value(row["timing"], "e2e"),
                reverse=True,
            )[:2]
            slowest_transport = sorted(
                perf_rows,
                key=lambda row: _timing_value(row["timing"], "tq_transport"),
                reverse=True,
            )[:2]
            logger.info(
                "Standalone TQ Producer perf middle samples: %s",
                " | ".join(_format_perf_sample(row) for row in middle),
            )
            logger.info(
                "Standalone TQ Producer perf slowest e2e samples: %s",
                " | ".join(_format_perf_sample(row) for row in slowest_e2e),
            )
            logger.info(
                "Standalone TQ Producer perf slowest transport samples: %s",
                " | ".join(_format_perf_sample(row) for row in slowest_transport),
            )
            perf_rows.clear()
            perf_window_started = now
            peak_publish_queue = publish_queue.qsize()
            peak_publish_inflight = publish_inflight
            peak_pending_bytes = stats.pending_bytes

        async def publish_results() -> None:
            nonlocal last_published_at, publish_inflight, peak_publish_inflight
            current = asyncio.current_task()
            worker = current.get_name() if current is not None else "publisher-unknown"
            while True:
                mark_stage(worker, "publish_queue_get")
                result = await publish_queue.get()
                if result is _PUBLISH_DONE:
                    stages.pop(worker, None)
                    return
                result.timing["publish_queue"] = time.monotonic() - result.timing.pop(
                    "publish_queue_started"
                )
                mark_stage(worker, "tq_put", result.request.sample_id)
                put_started = time.monotonic()
                publish_inflight += 1
                peak_publish_inflight = max(peak_publish_inflight, publish_inflight)
                try:
                    (
                        _,
                        encode_seconds,
                        transport_seconds,
                        cleanup_seconds,
                        attempts,
                        retry_wait_seconds,
                    ) = await publish_one(
                        result,
                        transport,
                        executor=publish_executor,
                        max_attempts=int(producer_cfg.get("publish_max_attempts", 3)),
                        retry_backoff_seconds=float(
                            producer_cfg.get("publish_retry_backoff_seconds", 0.5)
                        ),
                    )
                finally:
                    publish_inflight -= 1
                put_elapsed = time.monotonic() - put_started
                result.timing["tq_encode"] = encode_seconds
                result.timing["tq_transport"] = transport_seconds
                result.timing["tq_attempts"] = float(attempts)
                result.timing["tq_retry_wait"] = retry_wait_seconds
                result.timing["temp_cleanup"] = cleanup_seconds
                result.timing["e2e"] = time.monotonic() - result.timing.pop(
                    "e2e_started"
                )
                result.timing["active"] = sum(
                    _timing_value(result.timing, name)
                    for name in (
                        "input_prepare",
                        "generate",
                        "prefill",
                        "conversion",
                        "tq_encode",
                        "tq_transport",
                        "temp_cleanup",
                    )
                )
                result.timing["backpressure"] = sum(
                    _timing_value(result.timing, name)
                    for name in (
                        "input_queue",
                        "scheduler_wait",
                        "tq_capacity",
                        "feature_slot",
                        "publish_queue",
                        "tq_retry_wait",
                    )
                )
                if put_elapsed >= 30:
                    logger.warning(
                        "Standalone TQ Producer slow TQ put sample_id=%s elapsed=%.1fs",
                        result.request.sample_id,
                        put_elapsed,
                    )
                stats.published_count += 1
                if on_published is not None:
                    await on_published(result.request.sequence_no)
                last_published_at = time.monotonic()
                if _should_log_sample_progress(stats.published_count):
                    logger.info(
                        "Standalone TQ Producer published count=%s sequence_no=%s "
                        "sample_id=%s",
                        stats.published_count,
                        result.request.sequence_no,
                        result.request.sample_id,
                    )
                stats.pending_bytes = max(
                    stats.pending_bytes - int(result.raw.byte_size), 0
                )
                sample_timings.pop(int(result.request.sequence_no), None)
                perf_rows.append(
                    {
                        "sample_id": result.request.sample_id,
                        "sequence_no": result.request.sequence_no,
                        "endpoint": result.raw.endpoint_url,
                        "prompt_tokens": len(result.request.prompt_token_ids),
                        "generated_tokens": len(result.raw.generated_token_ids),
                        "mb": _sample_bytes(result.sample) / (1024 * 1024),
                        "timing": result.timing,
                    }
                )
                log_perf_window()

        request_tasks = [
            asyncio.create_task(request_worker(), name=f"request-{index}")
            for index in range(worker_count)
        ]

        async def finish_requests() -> None:
            await asyncio.gather(*request_tasks)
            for _ in range(publish_workers):
                await publish_queue.put(_PUBLISH_DONE)

        tasks = [asyncio.create_task(read_inputs(), name="input")]
        tasks.append(asyncio.create_task(finish_requests(), name="request-coordinator"))
        tasks.extend(
            asyncio.create_task(publish_results(), name=f"publisher-{index}")
            for index in range(publish_workers)
        )
        heartbeat = asyncio.create_task(log_heartbeat(), name="producer-heartbeat")
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            failure_task = next(
                (task for task in done if task.exception() is not None), None
            )
            failure_error = (
                failure_task.exception() if failure_task is not None else None
            )
            if failure_task is not None and failure_error is not None:
                stats.failed_count += 1
                logger.error(
                    "Standalone TQ Producer task failed task=%s stages=%s "
                    "input_queue=%s publish_queue=%s",
                    failure_task.get_name(),
                    stages,
                    input_queue.qsize(),
                    publish_queue.qsize(),
                    exc_info=(
                        type(failure_error),
                        failure_error,
                        failure_error.__traceback__,
                    ),
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure_error
            await asyncio.gather(*pending)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        if stats.published_count == 0:
            logger.error(
                "Standalone TQ Producer published no samples inputs=%s filtered=%s "
                "dropped=%s; emitting EOS with total_samples=0",
                stats.input_count,
                stats.filtered_count,
                stats.dropped_count,
            )
        eos_key, eos_fields, eos_tag = make_eos_record(run_id, stats.published_count)
        await asyncio.to_thread(transport.put_sample, eos_key, eos_fields, tag=eos_tag)
        completed = True
        logger.info(
            "Standalone TQ Producer completed inputs=%s published=%s dropped=%s "
            "failed=%s elapsed=%.3fs average_rate=%.2f/s",
            stats.input_count,
            stats.published_count,
            stats.dropped_count,
            stats.failed_count,
            time.monotonic() - producer_started_at,
            stats.published_count / max(time.monotonic() - producer_started_at, 1e-9),
        )
        return stats
    finally:
        try:
            try:
                if pool is not None:
                    await pool.close()
            finally:
                if feature_executor is not None:
                    feature_executor.shutdown(wait=True)
                if publish_executor is not None:
                    publish_executor.shutdown(wait=True)
        finally:
            if connected:
                if completed:
                    # Closing a remote store client unmounts the producer's
                    # segment; wait until the consumer has fetched and cleared
                    # every sample before releasing it.
                    await _drain_pending_samples(
                        transport,
                        run_id,
                        timeout=float(
                            os.environ.get(
                                "SPECO_TQ_PRODUCER_DRAIN_TIMEOUT_SECONDS", "1800"
                            )
                            or 0
                        ),
                        poll_interval=float(
                            producer_cfg["pending_poll_interval_seconds"]
                        ),
                    )
                transport.close_transfer_queue_client()


async def _wait_for_owner_ready(
    transport: Any,
    run_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        records = await asyncio.to_thread(transport.list_samples)
        if any(
            tag.get("record_type") == "control"
            and tag.get("status") == "owner_ready"
            and tag.get("run_id") == run_id
            for tag in records.values()
        ):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for TQ owner_ready for run_id={run_id!r}"
            )
        await asyncio.sleep(poll_interval)


async def _wait_for_pending_capacity(
    transport: Any,
    run_id: str,
    *,
    max_pending_samples: int,
    poll_interval: float,
) -> None:
    while True:
        records = await asyncio.to_thread(transport.list_samples)
        ready_count = sum(
            1
            for tag in records.values()
            if is_ready_sample_tag(
                tag,
                run_id=run_id,
                schema_version=PROTOCOL_SCHEMA_VERSION,
            )
        )
        if ready_count < max_pending_samples:
            return
        await asyncio.sleep(poll_interval)


async def _drain_pending_samples(
    transport: Any,
    run_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    """Wait until the consumer has fetched and cleared every published sample.

    Closing a remote store client unmounts the producer segment, so samples a
    slow consumer has not fetched yet would be dropped. The consumer deletes
    consumed records, therefore an empty ready set means everything was acked.
    A fixed linger cannot guarantee this; ``timeout`` only bounds a stalled
    consumer and logs a warning instead of failing the (already completed) run.
    """

    if timeout <= 0:
        return
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        records = await asyncio.to_thread(transport.list_samples)
        pending = sum(
            1
            for tag in records.values()
            if is_ready_sample_tag(
                tag,
                run_id=run_id,
                schema_version=PROTOCOL_SCHEMA_VERSION,
            )
        )
        if pending == 0:
            logger.info(
                "Standalone TQ Producer drained; the consumer cleared all samples"
            )
            return
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "Standalone TQ Producer drain timed out with %s unconsumed "
                "samples; closing the client may drop them",
                pending,
            )
            return
        await asyncio.sleep(poll_interval)


def _sample_metadata(
    request: TokenizedRequest,
    sample: DraftFeatureSample,
    contract: FeatureContract,
    run_id: str,
    tq_cfg: Mapping[str, Any],
) -> SampleMetadata:
    del sample, contract
    return SampleMetadata(
        schema_version=int(tq_cfg["schema_version"]),
        run_id=run_id,
        sample_id=request.sample_id,
        sequence_no=request.sequence_no,
    )


def _config_sections(
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plain = _plain_config(config)
    try:
        producer_cfg = plain["speco"]["standalone_tq_producer"]
        drafter = plain["actor_rollout_ref"]["rollout"]["drafter"]
        training_cfg = drafter["training"]
        tq_cfg = training_cfg["transfer_queue"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Producer configuration missing section {exc}") from exc
    if not all(
        isinstance(value, dict) for value in (producer_cfg, training_cfg, tq_cfg)
    ):
        raise TypeError("Producer configuration sections must resolve to mappings")
    return (
        producer_cfg,
        {**training_cfg, "speculative_algorithm": drafter.get("speculative_algorithm")},
        tq_cfg,
    )


def _plain_config(config: Any) -> dict[str, Any]:
    value = config
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            value = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    if not isinstance(value, Mapping):
        raise TypeError("Producer configuration must be a mapping")
    return dict(value)


def _load_tokenizer(config: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Standalone TQ Producer requires transformers") from exc
    return AutoTokenizer.from_pretrained(
        str(config["tokenizer_path"]),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )


def _parse_dtype(value: Any) -> torch.dtype:
    name = str(value).strip().lower().removeprefix("torch.")
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype = getattr(torch, aliases.get(name, name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported standalone_tq_producer.hidden_dtype={value!r}")
    return dtype


def _hydra_main(config: Any) -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_producer(config))


def main() -> None:
    try:
        import hydra
    except ImportError as exc:
        raise RuntimeError("Standalone TQ Producer requires hydra-core") from exc
    hydra.main(config_path="config", config_name="speco_base", version_base=None)(
        _hydra_main
    )()


if __name__ == "__main__":
    main()


__all__ = [
    "PreparedFeature",
    "ProducerStats",
    "main",
    "publish_one",
    "run_producer",
    "validate_producer_config",
]
