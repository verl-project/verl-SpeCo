# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Standalone generation dispatch, output compatibility and commit-based resume."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from omegaconf import OmegaConf

from verl_speco import standalone_data_generate as generation
from verl_speco.producer.input_reader import iter_input_records


def config(tmp_path):
    cfg = OmegaConf.to_container(
        OmegaConf.load(
            Path(__file__).parents[2]
            / "verl_speco/config/standalone_data_generation.yaml"
        )
    )["standalone_data_generation"]
    cfg["input_path"] = str(tmp_path / "input.jsonl")
    cfg["output_path"] = str(tmp_path / "responses.jsonl")
    cfg["vllm"].update(
        model="test",
        endpoints=["http://first/v1", "http://second/v1"],
        concurrency=4,
        per_endpoint_concurrency=2,
        endpoint_cooldown=0,
    )
    Path(cfg["input_path"]).write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"row-{i}",
                    "conversations": [
                        {"from": "human", "value": f"question-{i}"},
                        {"from": "assistant", "value": "old answer"},
                    ],
                }
            )
            + "\n"
            for i in range(8)
        ),
        encoding="utf-8",
    )
    return cfg


def test_response_dispatch_resume_and_producer_roundtrip(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    calls, inflight, peak = [], {"first": 0, "second": 0}, {"first": 0, "second": 0}

    first_requests = asyncio.Event()

    async def handler(request):
        host = request.url.host
        inflight[host] += 1
        peak[host] = max(peak[host], inflight[host])
        payload = json.loads(request.content)
        calls.append((host, payload))
        if len(calls) >= 2:
            first_requests.set()
        await asyncio.wait_for(first_requests.wait(), timeout=5)
        inflight[host] -= 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "new answer"}}]}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(generation.generate(cfg))
    assert result["written"] == 8
    assert all(0 < value <= 2 for value in peak.values())
    assert sum(peak.values()) > 1
    assert {host for host, _ in calls} == {"first", "second"}
    assert all(payload["max_tokens"] == 512 for _, payload in calls)
    assert all(len(payload["messages"]) == 1 for _, payload in calls)
    records = list(iter_input_records(cfg["output_path"], on_error="raise"))
    assert {r.sample_id for r in records} == {f"row-{i}" for i in range(8)}
    assert all(record.response == "new answer" for record in records)
    assert "token_ids" not in Path(cfg["output_path"]).read_text()
    assert asyncio.run(generation.generate(cfg))["written"] == 0
    assert len(calls) == 8
    cfg["responses"]["max_tokens"] = 12
    with pytest.raises(ValueError, match="differs"):
        asyncio.run(generation.generate(cfg))


def test_retry_changes_endpoint(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["max_samples"] = 1
    hosts = []

    async def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "first":
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    assert asyncio.run(generation.generate(cfg))["written"] == 1
    assert hosts == ["first", "second"]


def test_request_failures_are_recorded_and_retried_on_resume(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.update(on_error="skip", max_samples=1)

    async def fail(*args):
        raise ValueError("bad server")

    monkeypatch.setattr(generation.ResponseClient, "generate", fail)
    assert asyncio.run(generation.generate(cfg))["failed"] == 1
    assert Path(cfg["output_path"]).read_text() == ""
    assert "bad server" in Path(cfg["output_path"] + ".errors.jsonl").read_text()

    async def succeed(self, record, sampling):
        return generation.response_row(record, "ok")

    monkeypatch.setattr(generation.ResponseClient, "generate", succeed)
    assert asyncio.run(generation.generate(cfg))["written"] == 1


def test_hidden_single_writer_and_manifest_resume(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file

    from verl_speco.producer.hidden_states_store import (
        RawHiddenDatasetWriter,
        RawHiddenSample,
        load_raw_hidden_manifest,
    )

    cfg = config(tmp_path)
    cfg.update(task="hidden", output_path=str(tmp_path / "hidden"))

    async def components(config, stack):
        store = RawHiddenDatasetWriter(config["output_path"])
        stack.callback(store.close)

        async def process(record):
            return store.stage_sample(
                RawHiddenSample(
                    payload={
                        "token_ids": torch.tensor([1, 2]),
                        "hidden_states": torch.ones(2, 2, 4),
                    },
                    sample_id=record.sample_id,
                    sequence_no=record.sequence_no,
                    request={"loss_mask": [0, 1], "feature_positions": [0, 1]},
                    feature_contract={"algorithm": "DSPARK"},
                )
            )

        return process, store

    monkeypatch.setattr(generation, "_hidden_components", components)
    assert asyncio.run(generation.generate(cfg))["written"] == 8
    entries = load_raw_hidden_manifest(cfg["output_path"])
    assert len(entries) == 8
    assert load_file(str(Path(cfg["output_path"]) / entries[0]["path"]))[
        "hidden_states"
    ].shape == (2, 2, 4)
    assert asyncio.run(generation.generate(cfg))["written"] == 0


def test_fatal_error_cancels_bounded_pipeline(tmp_path, monkeypatch):
    cfg = config(tmp_path)

    async def fail(*args):
        raise ValueError("abort")

    monkeypatch.setattr(generation.ResponseClient, "generate", fail)

    async def start_without_http_clients(self):
        # This test measures queue cancellation, not HTTP/SSL client startup.
        pass

    monkeypatch.setattr(generation.ResponseClient, "start", start_without_http_clients)

    async def run():
        with pytest.raises(ValueError, match="abort"):
            await asyncio.wait_for(generation.generate(cfg), timeout=5)

    asyncio.run(run())


@pytest.mark.parametrize("response", ["a", "answer" * 100])
@pytest.mark.parametrize("backend", ["file", "mooncake"])
def test_hidden_preserves_raw_values_and_replays_through_existing_cpu_preparation(
    tmp_path, monkeypatch, response, backend
):
    import sys
    from types import SimpleNamespace

    import torch

    from verl_speco.producer import vllm_feature_client as client_module
    from verl_speco.producer.hidden_states_store import load_raw_hidden_manifest

    cfg = config(tmp_path)
    cfg.update(task="hidden", output_path=str(tmp_path / "features"), max_samples=1)
    cfg["hidden"].update(
        vllm_aux_hidden_state_layer_ids=[1],
        target_model_path="fake-model",
    )
    Path(cfg["input_path"]).write_text(
        json.dumps({"sample_id": "known-answer", "prompt": "q", "response": response})
        + "\n"
    )

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(num_hidden_layers=3)
            ),
        ),
    )
    requests, deleted, originals = [], [], []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

        async def prefill_reference(self, request):
            requests.append(request)
            payload = {
                "token_ids": torch.tensor(request.prompt_token_ids),
                "hidden_states": torch.arange(
                    len(request.prompt_token_ids) * 8, dtype=torch.float32
                ).reshape(-1, 2, 4),
            }
            originals.append(payload)
            if backend == "mooncake":
                return client_module.VllmResponse(
                    hidden_states_path=None,
                    endpoint_url="fake",
                    handle="sample-handle",
                    store=SimpleNamespace(
                        backend="mooncake", load=lambda reference: (payload, 100)
                    ),
                )
            from safetensors.torch import save_file

            source = tmp_path / "connector.safetensors"
            save_file(payload, str(source))
            return client_module.VllmResponse(
                hidden_states_path=str(source),
                endpoint_url="fake",
            )

    monkeypatch.setattr(client_module, "VllmFeatureClientPool", Pool)
    monkeypatch.setattr(
        client_module, "delete_temporary_result", lambda raw: deleted.append(raw)
    )
    assert asyncio.run(generation.generate(cfg))["written"] == 1
    assert requests and len(deleted) == 1
    from safetensors.torch import load_file

    from verl_speco.trainer.standalone_batch_prepare import prepare_standalone_cpu_batch
    from verl_speco.trainer.target_feature_replay import FeatureContract
    from verl_speco.transport.drafter_sample_protocol import (
        PROTOCOL_SCHEMA_VERSION,
        SampleMetadata,
        make_reference_tag,
    )

    entry = load_raw_hidden_manifest(cfg["output_path"])[0]
    path = Path(cfg["output_path"]) / entry["path"]
    saved = load_file(str(path))
    assert set(saved) == {"token_ids", "hidden_states"}
    assert torch.equal(saved["hidden_states"], originals[0]["hidden_states"])
    assert saved["hidden_states"].dtype == originals[0]["hidden_states"].dtype
    assert entry["sample_id"] == "known-answer"
    assert entry["feature_contract"]["target_num_hidden_layers"] == 3
    assert entry["request"]["loss_mask"] == requests[0].loss_mask.tolist()
    assert entry["request"]["feature_positions"] == list(range(len("q" + response)))
    assert saved["hidden_states"].shape[0] == len("q" + response)
    from verl_speco.producer.hidden_states_store import bind_raw_hidden_contract

    captured = entry["feature_contract"]
    assert captured["contract_type"] == "raw_capture"
    assert (
        not {"algorithm", "dtype", "hidden_states_layout", "target_layer_ids"}
        & captured.keys()
    )
    values = bind_raw_hidden_contract(
        captured,
        {
            "algorithm": "DSPARK",
            "dtype": "torch.bfloat16",
            "hidden_states_layout": "dflash_aux_plus_last",
            "target_layer_ids": [0],
            "vllm_aux_hidden_state_layer_ids": [1],
            "target_model_id": captured["target_model_id"],
            "target_model_revision": captured["target_model_revision"],
            "tokenizer_fingerprint": captured["tokenizer_fingerprint"],
            "require_full_alignment": True,
            "use_logits": False,
        },
    )
    wire_contract = dict(values)
    values["dtype"] = getattr(torch, values["dtype"].removeprefix("torch."))
    tag = make_reference_tag(
        SampleMetadata(PROTOCOL_SCHEMA_VERSION, "replay", "known-answer", 0),
        path=str(path),
        request=entry["request"],
        feature_contract=wire_contract,
    )
    batch = prepare_standalone_cpu_batch(
        [tag], FeatureContract(**values), pin_memory=False
    )
    assert batch["last_hidden_state_norm"] == "raw_target_residual"
    assert torch.equal(
        batch["target_last_hidden_states"][0].float(),
        saved["hidden_states"][:, -1].bfloat16().float(),
    )
    assert path.exists()


@pytest.mark.parametrize("algorithm,response", [("DSPARK", "a"), ("DFLASH", "short")])
def test_hidden_rejects_the_same_short_supervision_as_live_producer(
    algorithm, response
):
    from verl_speco.producer.input_reader import (
        InputRecord,
        SampleFilteredError,
        tokenize_record,
    )
    from verl_speco.standalone_tq_producer import standalone_token_config

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

    token_cfg = standalone_token_config(
        {"max_feature_length": 512},
        {
            "speculative_algorithm": algorithm,
            "dflash_block_size": 16,
            "preserve_prompt_hidden_states": True,
        },
    )
    record = InputRecord(0, "short", "q", response, {})
    with pytest.raises(SampleFilteredError):
        tokenize_record(record, Tokenizer(), token_cfg)


@pytest.mark.parametrize(
    "message",
    [
        {"content": "analysis</think>answer"},
        {"content": "<think>analysis</think>answer"},
        {"reasoning_content": "analysis", "content": "answer"},
        {"reasoning": "analysis", "content": "answer"},
        {"reasoning_content": "analysis", "reasoning": "analysis", "content": "answer"},
        {"reasoning": "analysis", "content": "<think>analysis</think>answer"},
        {"reasoning_content": "<think>analysis</think>", "content": "answer"},
    ],
)
def test_reasoning_formats_survive_http_jsonl_and_producer_read(
    tmp_path, monkeypatch, message
):
    cfg = config(tmp_path)
    cfg["max_samples"] = 1

    async def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    assert asyncio.run(generation.generate(cfg))["written"] == 1
    record = next(iter_input_records(cfg["output_path"], on_error="raise"))
    assert record.response == "<think>\nanalysis\n</think>\n\nanswer"
    assert "reasoning_content" not in Path(cfg["output_path"]).read_text()


@pytest.mark.parametrize(
    "choice",
    [
        {"message": {"content": "answer"}, "finish_reason": "length"},
        {"message": {"content": "answer"}, "finish_reason": "tool_calls"},
        {"message": {"content": "answer", "tool_calls": [{"id": "call"}]}},
        {"message": {"content": "answer", "refusal": "refused"}},
        {"message": {"content": "<think>unfinished"}},
        {"message": {"content": "analysis</think>"}},
        {"message": {"content": " "}},
        {"message": {"content": None, "reasoning": "analysis"}},
        {"message": {"content": [{"type": "text", "text": "answer"}]}},
        {"message": {"content": "answer", "reasoning": ["analysis"]}},
        {
            "message": {
                "content": "answer",
                "reasoning_content": "one",
                "reasoning": "two",
            }
        },
        {"message": {"content": "<think>one</think>answer", "reasoning": "two"}},
        {"message": {"content": "<think>one</think><think>two</think>answer"}},
        {"message": {"content": "answer", "reasoning_details": [{"type": "unknown"}]}},
    ],
)
def test_ambiguous_or_incomplete_reasoning_is_rejected(choice):
    with pytest.raises((ValueError, TypeError)):
        generation.normalize_response(choice)


def test_plain_answer_is_unchanged_and_empty_thinking_is_supported():
    assert (
        generation.normalize_response({"message": {"content": "answer\n"}})
        == "answer\n"
    )
    assert generation.normalize_response(
        {"message": {"content": "<think></think>answer"}}
    ) == ("<think>\n\n</think>\n\nanswer")


def test_invalid_reasoning_goes_to_error_log_without_success_commit(
    tmp_path, monkeypatch
):
    cfg = config(tmp_path)
    cfg.update(on_error="skip", max_samples=1)
    cfg["vllm"]["max_retries"] = 0

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "<think>unfinished"},
                    }
                ]
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    stats = asyncio.run(generation.generate(cfg))
    assert stats["written"] == 0 and stats["failed"] == 1
    assert Path(cfg["output_path"]).read_text() == ""
    assert "finish_reason" in Path(cfg["output_path"] + ".errors.jsonl").read_text()


def test_capture_config_has_only_extraction_settings():
    cfg = OmegaConf.to_container(
        OmegaConf.load(
            Path(__file__).parents[2]
            / "verl_speco/config/standalone_data_generation.yaml"
        )
    )["standalone_data_generation"]["hidden"]
    assert cfg["include_final_hidden"] is True
    assert cfg["validate_outputs"] is False
    assert (
        not {
            "algorithm",
            "dtype",
            "dflash_block_size",
            "max_feature_length",
            "preserve_prompt_hidden_states",
            "min_supervised_tokens",
            "on_truncated_supervision",
        }
        & cfg.keys()
    )
    assert not any("loss_alpha" in key for key in cfg)


@pytest.fixture
def hidden_config(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(num_hidden_layers=3)
            ),
        ),
    )
    cfg = config(tmp_path)
    cfg.update(task="hidden", output_path=str(tmp_path / "hidden"), max_samples=2)
    cfg["hidden"].update(target_model_path="fake", vllm_aux_hidden_state_layer_ids=[1])
    cfg["vllm"].update(
        endpoints=["http://first/v1"], concurrency=1, per_endpoint_concurrency=1
    )
    Path(cfg["input_path"]).write_text(
        "".join(
            json.dumps({"sample_id": f"row-{i}", "prompt": "q", "response": "answer"})
            + "\n"
            for i in range(2)
        )
    )
    return cfg


def test_slow_file_handling_does_not_hold_http_request_slot(
    tmp_path, monkeypatch, hidden_config
):
    import threading

    import torch
    from safetensors.torch import save_file

    from verl_speco.producer.hidden_states_store import (
        RawHiddenDatasetWriter,
        load_raw_hidden_manifest,
    )

    stage_started, next_http = threading.Event(), threading.Event()
    stage_file = RawHiddenDatasetWriter.stage_file
    calls = []

    def slow_stage(self, reference, request, contract, **kwargs):
        if request.sequence_no == 0:
            stage_started.set()
            assert next_http.wait(5), "Next request was blocked by file handling"
        return stage_file(self, reference, request, contract, **kwargs)

    async def handler(request):
        tokens = json.loads(request.content)["prompt"]
        calls.append(tokens)
        source = tmp_path / f"connector-{len(calls)}.safetensors"
        save_file(
            {
                "token_ids": torch.tensor(tokens),
                "hidden_states": torch.ones(len(tokens), 2, 4),
            },
            str(source),
        )
        if len(calls) == 2:
            assert await asyncio.to_thread(stage_started.wait, 5)
            next_http.set()
        return httpx.Response(
            200,
            json={
                "choices": [{"prompt_token_ids": tokens}],
                "kv_transfer_params": {"hidden_states_path": str(source)},
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(RawHiddenDatasetWriter, "stage_file", slow_stage)
    hidden_config["hidden"]["file_concurrency"] = 1

    async def run():
        from concurrent.futures import ThreadPoolExecutor

        # Even one control thread must remain available while the mover blocks.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )
        return await generation.generate(hidden_config)

    result = asyncio.run(run())
    assert result["written"] == 2 and len(calls) == 2
    assert len(load_raw_hidden_manifest(hidden_config["output_path"])) == 2
    assert not list(tmp_path.glob("connector-*.safetensors"))


def test_hidden_disk_failures_abort_even_with_skip(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.update(task="hidden", output_path=str(tmp_path / "hidden"), on_error="skip")

    async def components(config, stack):
        async def process(record):
            raise OSError("disk full")

        return process, None

    monkeypatch.setattr(generation, "_hidden_components", components)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(generation.generate(cfg))


def test_cancelled_file_worker_finishes_move_before_connector_cleanup(
    tmp_path, monkeypatch, hidden_config
):
    import threading
    from contextlib import AsyncExitStack

    import torch
    from safetensors.torch import save_file

    from verl_speco.producer import vllm_feature_client as client
    from verl_speco.producer.hidden_states_store import (
        RawHiddenDatasetWriter,
        load_raw_hidden_manifest,
    )

    source = tmp_path / "connector.safetensors"
    started, finish = threading.Event(), threading.Event()
    cleaned = []
    stage_file = RawHiddenDatasetWriter.stage_file

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

        async def prefill_reference(self, request):
            tokens = request.prompt_token_ids
            save_file(
                {
                    "token_ids": torch.tensor(tokens),
                    "hidden_states": torch.ones(len(tokens), 2, 4),
                },
                str(source),
            )
            return client.VllmResponse(str(source), "fake")

    def slow_stage(self, *args, **kwargs):
        started.set()
        assert finish.wait(5)
        return stage_file(self, *args, **kwargs)

    def cleanup(reference):
        assert not source.exists(), "Source was released while the mover was using it"
        cleaned.append(reference)

    monkeypatch.setattr(client, "VllmFeatureClientPool", Pool)
    monkeypatch.setattr(client, "delete_temporary_result", cleanup)
    monkeypatch.setattr(RawHiddenDatasetWriter, "stage_file", slow_stage)

    async def run():
        async with AsyncExitStack() as stack:
            process, store = await generation._hidden_components(hidden_config, stack)
            record = next(
                iter_input_records(hidden_config["input_path"], on_error="raise")
            )
            task = asyncio.create_task(process(record))
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not cleaned
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(cleaned) == 1
            assert load_raw_hidden_manifest(store.path) == []
            assert len(list(store.path.glob("*.safetensors"))) == 1

    asyncio.run(run())


def test_validation_toggle_preserves_existing_generation_resume(tmp_path):
    cfg = config(tmp_path)
    cfg["hidden"].pop("validate_outputs")
    cfg["hidden"].pop("file_concurrency")
    cfg.pop("log_interval")
    cfg.pop("log_interval_seconds")
    output = Path(cfg["output_path"])
    generation._resume_state(cfg, output)
    cfg["hidden"]["validate_outputs"] = True
    cfg["hidden"]["file_concurrency"] = 2
    cfg.update(log_interval=1, log_interval_seconds=1)
    assert generation._resume_state(cfg, output) == set()


def test_timing_and_heartbeat_are_visible_without_changing_jsonl(
    tmp_path, monkeypatch, caplog
):
    import logging

    cfg = config(tmp_path)
    cfg.update(max_samples=2, log_interval=1, log_interval_seconds=0.005)

    async def handler(request):
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handler)),
    )
    with caplog.at_level(logging.INFO, logger=generation.__name__):
        assert asyncio.run(generation.generate(cfg))["written"] == 2
    assert "Standalone generation status written=0" in caplog.text
    assert "Standalone generation stages_ms(avg/max)" in caplog.text
    assert "endpoint_wait=" in caplog.text and "http=" in caplog.text
    assert "request=" in caplog.text and "commit=" in caplog.text
    assert "Standalone generation HTTP inflight_avg=" in caplog.text
    rows = [
        json.loads(line) for line in Path(cfg["output_path"]).read_text().splitlines()
    ]
    assert all(set(row) == {"sample_id", "conversations"} for row in rows)


def test_timing_throughput_uses_wall_time_not_sum_of_parallel_stage_times(caplog):
    import logging

    row = {
        "ok": 1,
        "input_tokens": 2048,
        "file_bytes": 2 * 1024**2,
        "cross_fs": 1,
        "http": 0.5,
        "file_copy": 0.25,
    }
    with caplog.at_level(logging.INFO, logger=generation.__name__):
        generation._log_generation_timing(
            [row],
            2,
            {"written": 1, "failed": 0},
            asyncio.Queue(),
            asyncio.Queue(),
            None,
        )
    assert "samples_per_s=0.500" in caplog.text
    assert "input_tokens_per_s=1024.0" in caplog.text
    assert "payload_MiB_per_s=1.0" in caplog.text
    assert "cross_fs=1" in caplog.text
    assert "http=500.00/500.00" in caplog.text


def test_slow_error_log_does_not_block_event_loop(tmp_path, monkeypatch, caplog):
    import logging
    import threading

    cfg = config(tmp_path)
    cfg.update(max_samples=1, on_error="skip", log_interval_seconds=0.005)
    writing, release = threading.Event(), threading.Event()
    original_open = Path.open

    class SlowErrors:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, value):
            writing.set()
            assert release.wait(5), "Error-log write blocked the event loop"
            return self.stream.write(value)

        def flush(self):
            return self.stream.flush()

    def open_file(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return (
            SlowErrors(stream)
            if str(path).endswith(".errors.jsonl") and "a" in mode
            else stream
        )

    async def fail(*args):
        raise ValueError("bad sample")

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(generation.ResponseClient, "generate", fail)

    async def run():
        task = asyncio.create_task(generation.generate(cfg))
        try:
            assert await asyncio.to_thread(writing.wait, 5)
            await asyncio.sleep(0.03)
            assert "Standalone generation status written=0 failed=0" in caplog.text
        finally:
            release.set()
        return await task

    with caplog.at_level(logging.INFO, logger=generation.__name__):
        assert asyncio.run(run())["failed"] == 1
    assert "bad sample" in Path(cfg["output_path"] + ".errors.jsonl").read_text()
    assert "skipped sample_id=" in caplog.text


@pytest.mark.parametrize("file_concurrency", [0, -1])
def test_invalid_file_concurrency_is_rejected(hidden_config, file_concurrency):
    hidden_config["hidden"]["file_concurrency"] = file_concurrency
    with pytest.raises(ValueError, match="file_concurrency"):
        asyncio.run(generation.generate(hidden_config))
