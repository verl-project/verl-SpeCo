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
from dataclasses import dataclass, replace
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
        worker_count = int(producer_cfg["max_inflight_requests"])
        input_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=int(producer_cfg["input_queue_size"])
        )
        publish_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=int(producer_cfg["publish_queue_size"])
        )

        async def read_inputs() -> None:
            max_samples = int(producer_cfg.get("max_samples", 0) or 0)
            epoch = 0
            source_sequence_no = 0
            while True:
                epoch_count = 0
                scanned_count = 0
                for source_record in iter_input_records(
                    str(producer_cfg["input_path"])
                ):
                    if max_samples > 0 and stats.input_count >= max_samples:
                        break
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
                    await input_queue.put(request)
                    stats.input_count += 1
                    epoch_count += 1
                    if _should_log_sample_progress(stats.input_count):
                        logger.info(
                            "Standalone TQ Producer queued input count=%s epoch=%s "
                            "sample_id=%s has_response=%s",
                            stats.input_count,
                            epoch,
                            record.sample_id,
                            record.response is not None,
                        )
                if scanned_count == 0 and stats.input_count == 0:
                    raise ValueError("Standalone TQ Producer input contains no samples")
                if max_samples <= 0 or stats.input_count >= max_samples:
                    break
                epoch += 1
                logger.info(
                    "Standalone TQ Producer restarting input epoch=%s "
                    "queued=%s target=%s",
                    epoch,
                    stats.input_count,
                    max_samples,
                )
            for _ in range(worker_count):
                await input_queue.put(_INPUT_DONE)
            logger.info(
                "Standalone TQ Producer input exhausted total=%s", stats.input_count
            )

        async def request_worker() -> None:
            while True:
                request = await input_queue.get()
                if request is _INPUT_DONE:
                    await publish_queue.put(_PUBLISH_DONE)
                    return
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
                    raw = await pool.prefill(request)
                else:
                    raw = await pool.prefill(request)
                stats.pending_bytes += int(raw.byte_size)
                try:
                    sample = feature_from_vllm_payload(raw, request, feature_contract)
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
                    continue
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
            finished_workers = 0
            while finished_workers < worker_count:
                result = await publish_queue.get()
                if result is _PUBLISH_DONE:
                    finished_workers += 1
                    continue
                await publish_one(result, transport)
                stats.published_count += 1
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

        tasks = [asyncio.create_task(read_inputs())]
        tasks.extend(asyncio.create_task(request_worker()) for _ in range(worker_count))
        tasks.append(asyncio.create_task(publish_results()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        failure = next(
            (task.exception() for task in done if task.exception() is not None), None
        )
        if failure is not None:
            stats.failed_count += 1
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise failure
        await asyncio.gather(*pending)

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
            if pool is not None:
                await pool.close()
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
