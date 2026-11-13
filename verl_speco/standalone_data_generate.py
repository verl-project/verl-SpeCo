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
"""Generate response JSONL or persistent hidden features using external vLLM servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import hydra
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class _GenerationResult:
    value: dict
    timings: dict
    diagnostics: object | None = None


def _log_generation_timing(rows, elapsed, stats, inputs, results, diagnostics):
    elapsed = max(elapsed, 1e-9)
    total_bytes = sum(row.get("file_bytes", 0) for row in rows)
    logger.info(
        "Standalone generation timing samples=%s written=%s failed=%s "
        "window=%.3fs samples_per_s=%.3f input_tokens_per_s=%.1f "
        "payload_MiB_per_s=%.1f cross_fs=%s input_queue=%s result_queue=%s",
        len(rows),
        stats["written"],
        stats["failed"],
        elapsed,
        sum(row.get("ok", 0) for row in rows) / elapsed,
        sum(row.get("input_tokens", 0) for row in rows) / elapsed,
        total_bytes / 1024**2 / elapsed,
        sum(row.get("cross_fs", 0) for row in rows),
        inputs.qsize(),
        results.qsize(),
    )
    names = (
        "input_read",
        "input_wait",
        "tokenize",
        "request",
        "global_wait",
        "endpoint_wait",
        "http",
        "retry_backoff",
        "file_slot_wait",
        "file_thread_wait",
        "file_total",
        "file_lock",
        "file_inspect",
        "file_validate",
        "file_move",
        "file_copy",
        "handle_load",
        "payload_save",
        "cleanup_thread_wait",
        "cleanup_io",
        "cleanup",
        "result_wait",
        "commit_thread_wait",
        "commit_io",
        "commit",
        "process",
        "e2e",
    )
    fields = []
    for name in names:
        values = [row[name] for row in rows if name in row]
        if values:
            fields.append(
                f"{name}={1000 * sum(values) / len(values):.2f}/{1000 * max(values):.2f}"
            )
    logger.info("Standalone generation stages_ms(avg/max) %s", " ".join(fields))
    if diagnostics is not None:
        snapshot = diagnostics.snapshot()
        logger.info(
            "Standalone generation HTTP inflight_avg=%.2f inflight_peak=%s "
            "inflight_now=%s calls=%s",
            snapshot["http_inflight_avg"],
            snapshot["http_inflight_peak"],
            snapshot["http_inflight_now"],
            snapshot["http_calls"],
        )


async def _await_output_io(pending):
    """Finish an owning I/O thread before cancellation can release its resources."""
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Pipeline shutdown can cancel a worker through more than one gather.
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                pass
        pending.result()
        raise


def response_messages(record):
    """Discard only the last answer, retaining all earlier conversation turns."""
    if isinstance(record.prompt, str):
        return [{"role": "user", "content": record.prompt}]
    return [dict(message) for message in record.prompt]


def response_row(record, text):
    messages = [*response_messages(record), {"role": "assistant", "content": text}]
    roles = {"user": "human", "assistant": "assistant", "system": "system"}
    return {
        "sample_id": record.sample_id,
        "conversations": [
            {
                "from": roles.get(message["role"], message["role"]),
                "value": message["content"],
            }
            for message in messages
        ],
    }


def normalize_response(choice):
    """Preserve visible reasoning as one complete leading think block.

    This handles non-streaming Chat Completions text only. Ambiguous or
    incomplete outputs fail rather than silently discarding training text.
    """
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and finish_reason != "stop":
        raise ValueError(
            f"Incomplete/non-text generation: finish_reason={finish_reason!r}"
        )
    message = choice["message"]
    if (
        message.get("tool_calls")
        or message.get("function_call")
        or message.get("refusal")
    ):
        raise ValueError("Tool calls/refusals cannot be saved as a plain-text answer")
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("Generation must contain a text content field")

    def split_thinking(text):
        if "</think>" not in text:
            if "<think>" in text:
                raise ValueError("Unclosed think block in generation")
            return None, text
        if text.count("</think>") != 1 or text.count("<think>") > 1:
            raise ValueError("Multiple think blocks in generation")
        reasoning, answer = text.split("</think>", 1)
        if "<think>" in reasoning:
            if not reasoning.lstrip().startswith("<think>"):
                raise ValueError("Think block must precede the answer")
            reasoning = reasoning.lstrip()[len("<think>") :]
        if "<think>" in answer:
            raise ValueError("Unexpected think block after the answer")
        return reasoning.strip(), answer.lstrip()

    inline_reasoning, answer = split_thinking(content)
    reasoning_parts = []
    if inline_reasoning is not None:
        reasoning_parts.append(inline_reasoning)
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise TypeError(f"Unsupported non-text {key} field")
        wrapped, remainder = split_thinking(value)
        if wrapped is not None:
            if remainder.strip():
                raise ValueError(f"Unexpected answer inside {key} field")
            value = wrapped
        reasoning_parts.append(value.strip())
    if message.get("reasoning_details") and not reasoning_parts:
        raise ValueError("Unsupported reasoning_details without text reasoning")
    # Some gateways send both aliases or duplicate reasoning inside content.
    nonempty = {part for part in reasoning_parts if part}
    if len(nonempty) > 1:
        raise ValueError("Conflicting inline/separate reasoning fields")
    if not answer.strip():
        raise ValueError("Generation has no final answer")
    if reasoning_parts:
        reasoning = next(iter(nonempty), "")
        return f"<think>\n{reasoning}\n</think>\n\n{answer}"
    return answer


class ResponseClient:
    """Bounded endpoint dispatch with retries across manually configured services."""

    def __init__(self, config):
        from verl_speco.producer.metrics import ProducerDiagnostics

        self.config = config
        self.diagnostics = ProducerDiagnostics()
        self.clients = []
        self.inflight = []

    async def start(self):
        import httpx

        for endpoint in self.config["endpoints"]:
            self.clients.append(
                httpx.AsyncClient(
                    base_url=str(endpoint).rstrip("/") + "/",
                    headers={
                        "Authorization": "Bearer " + self.config.get("api_key", "EMPTY")
                    },
                    timeout=float(self.config["request_timeout"]),
                )
            )
            self.inflight.append(0)
        self.slots = [
            asyncio.Semaphore(int(self.config["per_endpoint_concurrency"]))
            for _ in self.clients
        ]

    async def close(self):
        await asyncio.gather(*(client.aclose() for client in self.clients))

    async def generate(self, record, sampling):
        previous = None
        timings = {"endpoint_wait": 0.0, "http": 0.0, "retry_backoff": 0.0}
        for attempt in range(int(self.config["max_retries"]) + 1):
            candidates = [i for i in range(len(self.clients)) if i != previous] or [0]
            index = min(candidates, key=lambda i: (self.inflight[i], i))
            previous = index
            self.inflight[index] += 1
            try:
                waiting = perf_counter()
                async with self.slots[index]:
                    timings["endpoint_wait"] += perf_counter() - waiting
                    started = perf_counter()
                    self.diagnostics.http_start()
                    try:
                        result = await self.clients[index].post(
                            "chat/completions",
                            json={
                                "model": self.config["model"],
                                "messages": response_messages(record),
                                **{
                                    key: value
                                    for key, value in sampling.items()
                                    if value is not None
                                },
                            },
                        )
                    finally:
                        timings["http"] += perf_counter() - started
                        self.diagnostics.http_finish()
                    result.raise_for_status()
                    text = normalize_response(result.json()["choices"][0])
                    return _GenerationResult(
                        response_row(record, text), timings, self.diagnostics
                    )
            except Exception:
                if attempt >= int(self.config["max_retries"]):
                    raise
                # Hold this endpoint's load reservation during cooldown so other
                # workers choose another server while it recovers.
                started = perf_counter()
                await asyncio.sleep(float(self.config["endpoint_cooldown"]))
                timings["retry_backoff"] += perf_counter() - started
            finally:
                self.inflight[index] -= 1


async def _hidden_components(config, stack):
    hidden, vllm = config["hidden"], config["vllm"]
    file_concurrency = int(hidden.get("file_concurrency", 4))
    if file_concurrency <= 0:
        raise ValueError("hidden.file_concurrency must be positive")

    from transformers import AutoConfig, AutoTokenizer

    from verl_speco.producer.hidden_states_store import (
        RawHiddenDatasetWriter,
        build_hidden_states_store,
        load_raw_hidden_manifest,
        prepare_raw_hidden_sample,
    )
    from verl_speco.producer.input_reader import (
        SampleFilteredError,
        build_render_fn,
        tokenize_record,
        tokenize_record_with_render_boundary,
    )
    from verl_speco.producer.vllm_feature_client import (
        VllmEndpoint,
        VllmFeatureClientPool,
        delete_temporary_result,
        load_hidden_state_result,
    )
    from verl_speco.standalone_tq_training_launcher import _saved_hidden_identity

    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        hidden["tokenizer_path"] or hidden["target_model_path"],
        trust_remote_code=hidden["trust_remote_code"],
    )
    model_config = await asyncio.to_thread(
        AutoConfig.from_pretrained,
        hidden["target_model_path"],
        trust_remote_code=bool(hidden["trust_remote_code"]),
    )
    target_depth = int(
        getattr(model_config, "text_config", model_config).num_hidden_layers
    )
    aux_ids = [int(value) for value in hidden["vllm_aux_hidden_state_layer_ids"]]
    if (
        not aux_ids
        or len(set(aux_ids)) != len(aux_ids)
        or any(value < 1 or value > target_depth for value in aux_ids)
    ):
        raise ValueError(
            "Configure distinct auxiliary output IDs within the target depth"
        )
    store = RawHiddenDatasetWriter(config["output_path"])
    stack.callback(store.close)
    existing = load_raw_hidden_manifest(store.path)
    previous_contract = existing[0]["feature_contract"] if existing else {}
    contract = {
        "contract_type": "raw_capture",
        "vllm_aux_hidden_state_layer_ids": aux_ids,
        "include_final_hidden": bool(hidden["include_final_hidden"]),
        "target_num_hidden_layers": target_depth,
        "target_model_id": hidden["target_model_path"],
        "target_model_revision": _saved_hidden_identity(
            "target",
            hidden["target_model_path"],
            previous_contract.get("target_model_revision"),
        ),
        "tokenizer_fingerprint": _saved_hidden_identity(
            "tokenizer",
            hidden["tokenizer_path"] or hidden["target_model_path"],
            previous_contract.get("tokenizer_fingerprint"),
        ),
        "source": "standalone_data_generate",
    }
    if previous_contract:
        for name in ("target_model_revision", "tokenizer_fingerprint"):
            if previous_contract.get(name) != contract[name]:
                raise ValueError(f"Saved hidden capture identity changed: {name}")
    pool = VllmFeatureClientPool(
        [
            VllmEndpoint(url.rstrip("/"), int(vllm["per_endpoint_concurrency"]))
            for url in vllm["endpoints"]
        ],
        model=vllm["model"],
        max_inflight_requests=int(vllm["concurrency"]),
        request_timeout=float(vllm["request_timeout"]),
        hidden_states_store=build_hidden_states_store(hidden["hidden_states_store"]),
    )
    stack.push_async_callback(pool.close)
    await pool.start()
    # Capture the full tokenized conversation. Training windows, block-size
    # filtering and conversion dtype belong to the training Producer.
    token_config = {
        "max_sequence_length": hidden["max_sequence_length"],
        "preserve_prompt_context": True,
        "max_feature_length": 0,
        "min_supervised_tokens": 1,
    }
    render_cfg = hidden.get("render_boundary", {})
    render_fn = (
        build_render_fn(vllm["endpoints"][0], timeout=render_cfg["timeout"])
        if render_cfg.get("enabled", False)
        else None
    )

    def tokenize(record):
        if render_fn is not None:
            try:
                return tokenize_record_with_render_boundary(
                    record, tokenizer, token_config, render_fn
                )
            except SampleFilteredError:
                raise
            except Exception as exc:  # noqa: BLE001 - match Producer local-tokenizer fallback
                logger.warning(
                    "Render-boundary tokenization failed for %s (%s); "
                    "falling back to the local tokenizer",
                    record.sample_id,
                    exc,
                )
        return tokenize_record(record, tokenizer, token_config)

    file_slots = asyncio.Semaphore(file_concurrency)
    file_executor = ThreadPoolExecutor(
        max_workers=file_concurrency, thread_name_prefix="hidden-file"
    )
    stack.callback(file_executor.shutdown, wait=True)
    loop = asyncio.get_running_loop()

    logged_locations = set()

    def stage(response, request, timings, submitted):
        started = perf_counter()
        timings["file_thread_wait"] = started - submitted
        validate = bool(hidden.get("validate_outputs", False))
        if not response.is_handle:
            entry = store.stage_file(
                response.reference,
                request,
                contract,
                validate_outputs=validate,
                timings=timings,
            )
        else:
            before = perf_counter()
            raw = load_hidden_state_result(response, response.store)
            timings["handle_load"] = perf_counter() - before
            timings["file_bytes"] = raw.byte_size
            before = perf_counter()
            sample = prepare_raw_hidden_sample(
                raw, request, contract, validate_outputs=validate
            )
            timings["file_inspect"] = perf_counter() - before
            before = perf_counter()
            entry = store.stage_sample(sample)
            timings["payload_save"] = perf_counter() - before
        timings["file_total"] = perf_counter() - started
        return entry

    async def process(record):
        timings = {}
        started = perf_counter()
        request = await asyncio.to_thread(tokenize, record)
        timings["tokenize"] = perf_counter() - started
        timings["input_tokens"] = len(request.prompt_token_ids)
        # Only HTTP occupies the request slots; file I/O has its own budget.
        started = perf_counter()
        reference = await pool.prefill_reference(request)
        timings["request"] = perf_counter() - started
        timings.update(reference.timings)
        try:
            if not reference.is_handle:
                location = (
                    reference.endpoint_url,
                    str(Path(reference.reference).parent),
                )
                if location not in logged_locations:
                    logged_locations.add(location)
                    logger.info(
                        "Hidden file transfer endpoint=%s source_dir=%s output_dir=%s",
                        *location,
                        store.path,
                    )
            started = perf_counter()
            async with file_slots:
                timings["file_slot_wait"] = perf_counter() - started
                pending = loop.run_in_executor(
                    file_executor, stage, reference, request, timings, perf_counter()
                )
                entry = await _await_output_io(pending)
        finally:
            started = perf_counter()

            def cleanup():
                timings["cleanup_thread_wait"] = perf_counter() - started
                before = perf_counter()
                delete_temporary_result(reference)
                timings["cleanup_io"] = perf_counter() - before

            await asyncio.to_thread(cleanup)
            timings["cleanup"] = perf_counter() - started
        return _GenerationResult(entry, timings, getattr(pool, "diagnostics", None))

    process.diagnostics = getattr(pool, "diagnostics", None)
    return process, store


def _resume_state(config, output):
    """Validate dataset/config identity and derive progress from committed outputs."""
    state_path = Path(str(output) + ".generation.json")
    identity = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "resume",
            "on_error",
            "max_samples",
            "log_interval",
            "log_interval_seconds",
        }
    }
    # Runtime I/O/validation settings do not change data; preserve existing resumes.
    identity["hidden"] = {
        key: value
        for key, value in identity["hidden"].items()
        if key not in {"validate_outputs", "file_concurrency"}
    }
    # Keep fingerprints compatible with configs that exposed unused null identities.
    identity["hidden"].setdefault("target_model_revision", None)
    identity["hidden"].setdefault("tokenizer_fingerprint", None)
    if config["task"] == "responses":
        identity["response_format_version"] = 1
    if config["task"] == "hidden":
        identity["hidden_format"] = "standalone_raw_hidden_capture_v1"
    with Path(config["input_path"]).open("rb") as source:
        identity["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if state_path.exists():
        if not config["resume"]:
            raise FileExistsError(f"Generation output already initialized: {output}")
        if json.loads(state_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Resume input/config differs from the original generation")
    else:
        if output.exists() and (output.is_file() or any(output.iterdir())):
            raise FileExistsError(f"Refusing to append to unmanaged output: {output}")
        state_path.write_text(
            json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
        )
    from verl_speco.producer.hidden_states_store import repair_jsonl_tail

    repair_jsonl_tail(
        output / "manifest.jsonl" if config["task"] == "hidden" else output
    )
    if config["task"] == "hidden":
        from verl_speco.producer.hidden_states_store import load_raw_hidden_manifest

        return (
            {entry["sample_id"] for entry in load_raw_hidden_manifest(output)}
            if output.exists() and any(output.iterdir())
            else set()
        )
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                completed.add(json.loads(line)["sample_id"])
    return completed


async def generate(config):
    """One pass with bounded input/results queues and one persistent writer."""
    from verl_speco.producer.input_reader import iter_input_records

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    config = dict(config)
    if config["task"] not in {"responses", "hidden"}:
        raise ValueError("task must be responses or hidden")
    vllm = config["vllm"]
    if (
        not vllm["endpoints"]
        or not vllm["model"]
        or int(vllm["concurrency"]) <= 0
        or int(vllm["per_endpoint_concurrency"]) <= 0
        or int(vllm["max_retries"]) < 0
        or float(vllm["endpoint_cooldown"]) < 0
    ):
        raise ValueError(
            "Configure model, endpoints and positive concurrency; retries/cooldown >= 0"
        )
    if config["on_error"] not in {"skip", "raise"}:
        raise ValueError("on_error must be skip or raise")
    if int(config["input_queue_size"]) <= 0 or int(config["publish_queue_size"]) <= 0:
        raise ValueError("input_queue_size and publish_queue_size must be positive")
    log_interval = int(config.get("log_interval", 20))
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    log_seconds = float(config.get("log_interval_seconds", 10))
    if log_seconds <= 0:
        raise ValueError("log_interval_seconds must be positive")
    output = Path(config["output_path"])
    if output.resolve() == Path(config["input_path"]).resolve():
        raise ValueError("input_path and output_path must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = await asyncio.to_thread(_resume_state, config, output)
    concurrency = int(vllm["concurrency"])
    # As in speculators, extra workers overlap post-request file handling with HTTP.
    worker_count = concurrency * (2 if config["task"] == "hidden" else 1)
    inputs = asyncio.Queue(int(config["input_queue_size"]))
    results = asyncio.Queue(int(config["publish_queue_size"]))
    stats = {"written": 0, "failed": 0, "resumed": len(completed)}
    async with AsyncExitStack() as stack:
        errors = stack.enter_context(
            Path(str(output) + ".errors.jsonl").open("a", encoding="utf-8")  # noqa: ASYNC230, SIM115
        )
        commit_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="generation-commit"
        )
        stack.callback(commit_executor.shutdown, wait=True)
        loop = asyncio.get_running_loop()
        if config["task"] == "responses":
            client = ResponseClient(vllm)
            stack.push_async_callback(client.close)
            await client.start()

            async def process(record):
                started = perf_counter()
                result = await client.generate(record, config["responses"])
                if isinstance(result, _GenerationResult):
                    result.timings["request"] = perf_counter() - started
                return result

            stream = stack.enter_context(output.open("a", encoding="utf-8"))  # noqa: ASYNC230

            def write(value):
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                stream.flush()
        else:
            process, store = await _hidden_components(config, stack)

            def write(value):
                store.commit(value)

        window_started = perf_counter()
        timing_rows = []
        last_diagnostics = (
            client.diagnostics
            if config["task"] == "responses"
            else getattr(process, "diagnostics", None)
        )
        active_workers = 0
        logger.info(
            "Standalone generation pipeline task=%s workers=%s request_limit=%s "
            "per_endpoint=%s endpoints=%s file_limit=%s validate_outputs=%s log_interval=%s",
            config["task"],
            worker_count,
            concurrency,
            vllm["per_endpoint_concurrency"],
            len(vllm["endpoints"]),
            int(config["hidden"].get("file_concurrency", 4))
            if config["task"] == "hidden"
            else 0,
            config["hidden"].get("validate_outputs", False),
            log_interval,
        )

        def log_window():
            nonlocal window_started
            if timing_rows:
                now = perf_counter()
                _log_generation_timing(
                    timing_rows,
                    now - window_started,
                    stats,
                    inputs,
                    results,
                    last_diagnostics,
                )
                timing_rows.clear()
                window_started = now

        async def heartbeat():
            while True:
                await asyncio.sleep(log_seconds)
                log_window()
                occupancy = (
                    last_diagnostics.snapshot(reset=False)["http_inflight_now"]
                    if last_diagnostics is not None
                    else "unknown"
                )
                logger.info(
                    "Standalone generation status written=%s failed=%s "
                    "active_workers=%s http_inflight=%s input_queue=%s result_queue=%s",
                    stats["written"],
                    stats["failed"],
                    active_workers,
                    occupancy,
                    inputs.qsize(),
                    results.qsize(),
                )

        async def read():
            count = 0
            seen = set()
            records = iter(
                iter_input_records(
                    config["input_path"],
                    on_error=config["on_error"],
                    parser_strict_roles=bool(config["parser_strict_roles"]),
                    max_consecutive_errors=int(config["max_consecutive_errors"]),
                )
            )
            while True:
                # Parsing/index reads stay off the event loop; a sentinel avoids
                # propagating StopIteration through an asyncio Future.
                started = perf_counter()
                record = await asyncio.to_thread(next, records, None)
                read_time = perf_counter() - started
                if record is None:
                    break
                if record.sample_id in seen:
                    raise ValueError(f"Duplicate sample id: {record.sample_id}")
                seen.add(record.sample_id)
                if record.sample_id in completed:
                    continue
                if int(config["max_samples"]) > 0 and count >= int(
                    config["max_samples"]
                ):
                    break
                await inputs.put((record, read_time, perf_counter()))
                count += 1
            for _ in range(worker_count):
                await inputs.put(None)

        async def worker():
            nonlocal active_workers
            while (item := await inputs.get()) is not None:
                record, read_time, queued = item
                started = perf_counter()
                active_workers += 1
                try:
                    value = await process(record)
                except Exception as exc:
                    if config["on_error"] == "raise" or (
                        config["task"] == "hidden" and isinstance(exc, OSError)
                    ):
                        raise
                    await results.put(
                        (
                            False,
                            _GenerationResult(
                                {"sample_id": record.sample_id, "error": str(exc)},
                                {"process": perf_counter() - started},
                            ),
                            perf_counter(),
                            queued,
                        )
                    )
                else:
                    if not isinstance(value, _GenerationResult):
                        value = _GenerationResult(value, {})
                    value.timings.update(
                        input_read=read_time,
                        input_wait=started - queued,
                        process=perf_counter() - started,
                    )
                    await results.put((True, value, perf_counter(), queued))
                finally:
                    active_workers -= 1

        async def write_results():
            nonlocal last_diagnostics

            def commit(ok, value, timings, submitted):
                timings["commit_thread_wait"] = perf_counter() - submitted
                before = perf_counter()
                if ok:
                    write(value)
                else:
                    errors.write(json.dumps(value, ensure_ascii=False) + "\n")
                    errors.flush()
                timings["commit_io"] = perf_counter() - before

            while (item := await results.get()) is not None:
                ok, result, queued, input_queued = item
                value, timings = result.value, result.timings
                timings["result_wait"] = perf_counter() - queued
                started = perf_counter()

                # Metadata/error writes must neither queue behind large copies
                # nor block the event loop when the output filesystem is slow.
                pending_write = loop.run_in_executor(
                    commit_executor, commit, ok, value, timings, started
                )
                await _await_output_io(pending_write)
                if ok:
                    stats["written"] += 1
                else:
                    stats["failed"] += 1
                    if stats["failed"] <= 5 or stats["failed"] % 100 == 0:
                        logger.warning(
                            "Standalone generation skipped sample_id=%s error=%s",
                            value["sample_id"],
                            value["error"],
                        )
                timings.update(
                    commit=perf_counter() - started,
                    e2e=perf_counter() - input_queued,
                    ok=int(ok),
                )
                timing_rows.append(timings)
                last_diagnostics = result.diagnostics or last_diagnostics
                count = stats["written"] + stats["failed"]
                if count == 1 or count % log_interval == 0:
                    log_window()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]

        async def finish():
            await asyncio.gather(*workers)
            await results.put(None)

        tasks = [
            *workers,
            asyncio.create_task(read()),
            asyncio.create_task(write_results()),
            asyncio.create_task(finish()),
        ]
        monitor = asyncio.create_task(heartbeat())
        try:
            await asyncio.gather(*tasks)
        finally:
            monitor.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, monitor, return_exceptions=True)
            log_window()
    logger.info("Standalone generation finished: %s", stats)
    return stats


@hydra.main(
    config_path="config", config_name="standalone_data_generation", version_base=None
)
def main(config):
    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate(config.standalone_data_generation))


if __name__ == "__main__":
    main()
