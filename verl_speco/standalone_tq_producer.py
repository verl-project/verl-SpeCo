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
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Mapping

import torch

from verl_speco.integration import transferqueue_bridge as default_transport
from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.producer.input_reader import (
    GenerationRequest,
    TokenizedRequest,
    iter_input_records,
    prepare_generation_request,
    prepare_generated_prefill_request,
    tokenize_record,
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


@dataclass
class ProducerStats:
    input_count: int = 0
    published_count: int = 0
    failed_count: int = 0
    dropped_count: int = 0
    pending_bytes: int = 0


@dataclass(frozen=True)
class PreparedFeature:
    request: TokenizedRequest
    raw: RawVllmFeature
    sample: DraftFeatureSample
    metadata: SampleMetadata


async def publish_one(result: PreparedFeature, transport: Any) -> str:
    """Publish one sample and delete its temporary file only after TQ succeeds."""

    key = make_sample_key(result.metadata)
    fields = encode_sample(result.sample, result.metadata)
    tag = make_ready_tag(result.metadata)
    await asyncio.to_thread(transport.put_sample, key, fields, tag=tag)
    delete_temporary_result(result.raw)
    return key


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


def _should_log_sample_progress(count: int) -> bool:
    return count <= 3 or count % 50 == 0


async def run_producer(
    config: Any,
    *,
    transport: Any = default_transport,
    tokenizer: Any | None = None,
    client_pool: Any | None = None,
) -> ProducerStats:
    """Run the bounded input -> vLLM -> TQ pipeline and publish EOS on success."""

    validate_producer_config(config)
    producer_cfg, drafter_cfg, tq_cfg = _config_sections(config)
    run_id = str(tq_cfg["run_id"])
    stats = ProducerStats()
    connected = False
    pool = client_pool
    feature_executor: ThreadPoolExecutor | None = None
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
        feature_executor = ThreadPoolExecutor(
            max_workers=_FEATURE_CONVERSION_WORKERS,
            thread_name_prefix="speco-feature",
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
        last_published_at = time.monotonic()
        max_samples = int(producer_cfg.get("max_samples", 0) or 0)

        def mark_stage(worker: str, stage: str, sample_id: str = "") -> None:
            stages[worker] = (stage, time.monotonic(), sample_id)

        async def log_heartbeat() -> None:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
                now = time.monotonic()
                oldest = sorted(
                    (
                        (now - started, worker, stage, sample_id)
                        for worker, (stage, started, sample_id) in stages.items()
                    ),
                    reverse=True,
                )[:3]
                logger.warning(
                    "Standalone TQ Producer heartbeat inputs=%s published=%s "
                    "dropped=%s input_queue=%s/%s publish_queue=%s/%s "
                    "seconds_since_publish=%.0f stages=%s oldest=%s",
                    stats.input_count,
                    stats.published_count,
                    stats.dropped_count,
                    input_queue.qsize(),
                    input_queue.maxsize,
                    publish_queue.qsize(),
                    publish_queue.maxsize,
                    now - last_published_at,
                    dict(Counter(stage for stage, _, _ in stages.values())),
                    [
                        (worker, stage, sample_id, round(age))
                        for age, worker, stage, sample_id in oldest
                    ],
                )

        def iter_requests():
            shuffle = bool(producer_cfg.get("shuffle", False))
            shuffle_seed = int(producer_cfg.get("shuffle_seed", 42))
            buffered_records: list | None = None
            if shuffle:
                buffered_records = list(
                    iter_input_records(str(producer_cfg["input_path"]))
                )
                logger.info(
                    "Standalone TQ Producer buffered %s records for shuffle_seed=%s",
                    len(buffered_records),
                    shuffle_seed,
                )
            epoch = 0
            source_sequence_no = 0
            while True:
                scanned_count = 0
                if shuffle:
                    # Deterministic per-epoch permutation (seed + epoch), so a
                    # resumed run replays the same order and the consumed
                    # sequence set stays aligned across restarts.
                    record_order = list(range(len(buffered_records)))
                    random.Random(shuffle_seed + epoch).shuffle(record_order)
                    record_iter = (buffered_records[i] for i in record_order)
                else:
                    record_iter = iter_input_records(str(producer_cfg["input_path"]))
                for source_record in record_iter:
                    sequence_no = source_sequence_no
                    source_sequence_no += 1
                    scanned_count += 1
                    if sequence_no in consumed_sequence_nos:
                        continue
                    # iter_input_records restarts sequence_no at zero on every
                    # pass. TQ keys require a run-global sequence number so a
                    # repeated sample never overwrites an earlier pending copy.
                    record = replace(
                        source_record,
                        sequence_no=sequence_no,
                    )
                    request = (
                        prepare_generation_request(record, tokenizer, producer_cfg)
                        if record.response is None
                        else tokenize_record(record, tokenizer, producer_cfg)
                    )
                    yield request
                if scanned_count == 0:
                    raise ValueError("Standalone TQ Producer input contains no samples")
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

        def next_request() -> Any:
            request = next(requests)
            stats.input_count += 1
            if _should_log_sample_progress(stats.input_count):
                logger.info(
                    "Standalone TQ Producer queued input count=%s sample_id=%s",
                    stats.input_count,
                    request.sample_id,
                )
            return request

        async def read_inputs() -> None:
            initial_count = 0
            while max_samples <= 0 or initial_count < max_samples:
                try:
                    request = next_request()
                except StopIteration:
                    break
                mark_stage("input", "input_queue_put", request.sample_id)
                await input_queue.put(request)
                initial_count += 1
            for _ in range(worker_count):
                await input_queue.put(_INPUT_DONE)
            logger.info(
                "Standalone TQ Producer input exhausted total=%s", stats.input_count
            )
            stages.pop("input", None)

        async def request_worker() -> None:
            current = asyncio.current_task()
            worker = current.get_name() if current is not None else "request-unknown"
            replacement_request = None
            while True:
                if replacement_request is None:
                    mark_stage(worker, "input_queue_get")
                    request = await input_queue.get()
                else:
                    request = replacement_request
                    replacement_request = None
                if request is _INPUT_DONE:
                    mark_stage(worker, "publish_queue_done")
                    await publish_queue.put(_PUBLISH_DONE)
                    stages.pop(worker, None)
                    return
                mark_stage(worker, "tq_capacity", request.sample_id)
                await _wait_for_pending_capacity(
                    transport,
                    run_id,
                    max_pending_samples=int(producer_cfg["max_pending_samples"]),
                    poll_interval=float(producer_cfg["pending_poll_interval_seconds"]),
                )
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
                    generated = await pool.generate(request)
                    try:
                        request = prepare_generated_prefill_request(
                            request,
                            generated.generated_token_ids,
                            producer_cfg,
                        )
                    finally:
                        # The generation request may still produce a prompt-only
                        # connector file. It is not the training payload; the
                        # following full-sequence prefill produces that payload.
                        await asyncio.to_thread(delete_temporary_result, generated)
                    mark_stage(worker, "vllm_prefill", request.sample_id)
                    raw = await pool.prefill(request)
                else:
                    mark_stage(worker, "vllm_prefill", request.sample_id)
                    raw = await pool.prefill(request)
                stats.pending_bytes += int(raw.byte_size)
                try:
                    mark_stage(worker, "feature_conversion", request.sample_id)
                    async with feature_slots:
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
                except HiddenStateAlignmentError as exc:
                    stats.dropped_count += 1
                    stats.pending_bytes = max(
                        stats.pending_bytes - int(raw.byte_size), 0
                    )
                    await asyncio.to_thread(delete_temporary_result, raw)
                    logger.warning(
                        "Standalone TQ Producer dropped misaligned sample "
                        "sequence_no=%s sample_id=%s dropped=%s reason=%s",
                        request.sequence_no,
                        request.sample_id,
                        stats.dropped_count,
                        exc,
                    )
                    if max_samples > 0:
                        replacement_request = next_request()
                        logger.info(
                            "Standalone TQ Producer replacing dropped sample "
                            "with sequence_no=%s sample_id=%s",
                            replacement_request.sequence_no,
                            replacement_request.sample_id,
                        )
                    continue
                mark_stage(worker, "publish_queue_put", request.sample_id)
                await publish_queue.put(
                    PreparedFeature(
                        request=request,
                        raw=raw,
                        sample=sample,
                        metadata=_sample_metadata(
                            request, sample, feature_contract, run_id, tq_cfg
                        ),
                    )
                )

        async def publish_results() -> None:
            nonlocal last_published_at
            finished_workers = 0
            while finished_workers < worker_count:
                mark_stage("publisher", "publish_queue_get")
                result = await publish_queue.get()
                if result is _PUBLISH_DONE:
                    finished_workers += 1
                    continue
                mark_stage("publisher", "tq_put", result.request.sample_id)
                put_started = time.monotonic()
                await publish_one(result, transport)
                put_elapsed = time.monotonic() - put_started
                if put_elapsed >= 30:
                    logger.warning(
                        "Standalone TQ Producer slow TQ put sample_id=%s elapsed=%.1fs",
                        result.request.sample_id,
                        put_elapsed,
                    )
                stats.published_count += 1
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
            stages.pop("publisher", None)

        tasks = [asyncio.create_task(read_inputs(), name="input")]
        tasks.extend(
            asyncio.create_task(request_worker(), name=f"request-{index}")
            for index in range(worker_count)
        )
        tasks.append(asyncio.create_task(publish_results(), name="publisher"))
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

        eos_key, eos_fields, eos_tag = make_eos_record(run_id, stats.published_count)
        await asyncio.to_thread(transport.put_sample, eos_key, eos_fields, tag=eos_tag)
        logger.info(
            "Standalone TQ Producer completed inputs=%s published=%s dropped=%s",
            stats.input_count,
            stats.published_count,
            stats.dropped_count,
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
        finally:
            if connected:
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
