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

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from verl_speco.producer.vllm_feature_client import RawVllmFeature
from verl_speco.standalone_tq_producer import (
    _drain_pending_samples,
    run_producer,
    validate_producer_config,
)
from verl_speco.trainer.standalone_resume import save_standalone_resume
from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION
from verl_speco.transport.drafter_sample_protocol import decode_sample


@pytest.fixture(autouse=True)
def _disable_consumer_drain(monkeypatch):
    # Unit tests use a fake transport whose consumer never clears samples.
    # Skip the post-run drain wait; the drain logic has dedicated tests below.
    monkeypatch.setenv("SPECO_TQ_PRODUCER_DRAIN_TIMEOUT_SECONDS", "0")


@pytest.fixture(autouse=True)
def target_final_norm(monkeypatch):
    # These pipeline tests use a fake /target checkpoint. Loader accuracy is
    # covered separately with real tiny HF checkpoints.
    norm = torch.nn.RMSNorm(2, eps=1e-6).requires_grad_(False)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 3.0]))
    monkeypatch.setattr(
        "verl_speco.standalone_tq_producer.load_vllm_final_norm",
        lambda *args, **kwargs: norm,
    )
    return norm


def _config(input_path: Path) -> dict[str, Any]:
    return {
        "speco": {
            "standalone_tq_producer": {
                "input_path": str(input_path),
                "tokenizer_path": "/target",
                "tokenizer_fingerprint": "sha256:tokenizer",
                "target_model_id": "/target",
                "target_model_revision": "rev-a",
                "target_layer_ids": [2, 8],
                "vllm_aux_hidden_state_layer_ids": [3, 9],
                "hidden_dtype": "float32",
                "trust_remote_code": False,
                "vllm_endpoints": ["http://vllm:8000/v1"],
                "vllm_model": "/target",
                "request_timeout": 10,
                "max_inflight_requests": 2,
                "per_endpoint_concurrency": 1,
                "input_queue_size": 2,
                "publish_queue_size": 2,
                "max_pending_samples": 8,
                "pending_poll_interval_seconds": 0.01,
                "owner_ready_timeout_seconds": 1,
                "max_sequence_length": 16,
                "max_feature_length": 8,
                "generation_max_tokens": 4,
            }
        },
        "actor_rollout_ref": {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DSPARK",
                    "training": {
                        "use_logits": False,
                        "dspark_l1_loss_alpha": 0.9,
                        "transfer_queue": {
                            "enable": True,
                            "package_version": "0.1.10",
                            "ray": {
                                "address": "ray-head:6379",
                                "namespace": "speco-drafter",
                            },
                            "partition_id": "speco_drafter_features",
                            "run_id": "run-a",
                            "schema_version": PROTOCOL_SCHEMA_VERSION,
                        },
                    },
                }
            }
        },
    }


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        values = {
            "Q1: ": [1, 2],
            "Q1: A1": [1, 2, 3, 4],
            "Q2: ": [5, 6],
            "Q2: A2": [5, 6, 7, 8],
        }
        return {"input_ids": values[text]}


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "Q3"}]
        assert tokenize is True
        assert add_generation_prompt is True
        return [9, 10]


class _Transport:
    def __init__(self, *, fail_sample_put: bool = False):
        self.fail_sample_put = fail_sample_put
        self.records: dict[str, dict[str, Any]] = {
            "control:v2:run-a:owner-ready": {
                "record_type": "control",
                "status": "owner_ready",
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "run_id": "run-a",
            }
        }
        self.payloads: dict[str, dict[str, torch.Tensor]] = {}
        self.closed = False

    def configure_transfer_queue(self, config: dict[str, Any]) -> bool:
        return bool(config["enable"])

    def connect_ray_cluster(self, address: str, namespace: str | None) -> None:
        assert (address, namespace) == ("ray-head:6379", "speco-drafter")

    def connect_transfer_queue_client(self) -> None:
        pass

    def list_samples(self) -> dict[str, dict[str, Any]]:
        return dict(self.records)

    def put_sample(
        self,
        key: str,
        fields: dict[str, torch.Tensor],
        *,
        tag: dict[str, Any],
    ) -> None:
        if self.fail_sample_put and tag.get("record_type") == "sample":
            raise RuntimeError("put failed")
        self.records[key] = dict(tag)
        self.payloads[key] = fields

    def close_transfer_queue_client(self) -> None:
        self.closed = True


class _ConcurrentTransport(_Transport):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self.active_puts = 0
        self.max_active_puts = 0

    def put_sample(self, key, fields, *, tag):
        if tag.get("record_type") != "sample":
            return super().put_sample(key, fields, tag=tag)
        with self._lock:
            self.active_puts += 1
            self.max_active_puts = max(self.max_active_puts, self.active_puts)
        try:
            time.sleep(0.05)
            return super().put_sample(key, fields, tag=tag)
        finally:
            with self._lock:
                self.active_puts -= 1


class _TransientFailureTransport(_Transport):
    def __init__(self, failures: int):
        super().__init__()
        self.failures_remaining = failures
        self.sample_put_attempts = 0

    def put_sample(self, key, fields, *, tag):
        if tag.get("record_type") == "sample":
            self.sample_put_attempts += 1
            if self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise RuntimeError("transient put failure")
        return super().put_sample(key, fields, tag=tag)


class _Pool:
    def __init__(self, root: Path, *, close_error: BaseException | None = None):
        self.root = root
        self.close_error = close_error
        self.paths: list[Path] = []
        self.started = False
        self.closed = False
        self.generate_calls = 0
        self.prefill_calls = 0

    async def start(self) -> None:
        self.started = True

    async def prefill(self, request: Any) -> RawVllmFeature:
        self.prefill_calls += 1
        path = self.root / f"{request.sample_id}-{request.sequence_no}.safetensors"
        path.write_bytes(b"temporary")
        self.paths.append(path)
        token_ids = torch.tensor(request.prompt_token_ids, dtype=torch.int64)
        hidden = torch.arange(token_ids.numel() * 3 * 2, dtype=torch.float32).reshape(
            token_ids.numel(), 3, 2
        )
        return RawVllmFeature(
            payload={"token_ids": token_ids, "hidden_states": hidden},
            temporary_path=str(path),
            endpoint_url="http://vllm:8000/v1",
            byte_size=path.stat().st_size,
        )

    async def generate(self, request: Any) -> RawVllmFeature:
        self.generate_calls += 1
        path = self.root / f"{request.sample_id}-{request.sequence_no}.safetensors"
        path.write_bytes(b"temporary")
        self.paths.append(path)
        # ExampleHiddenStatesConnector excludes the final generated token because
        # it was never consumed by a model forward pass.
        token_ids = torch.tensor([*request.prompt_token_ids, 11], dtype=torch.int64)
        hidden = torch.arange(token_ids.numel() * 3 * 2, dtype=torch.float32).reshape(
            token_ids.numel(), 3, 2
        )
        return RawVllmFeature(
            payload={"token_ids": token_ids, "hidden_states": hidden},
            temporary_path=str(path),
            endpoint_url="http://vllm:8000/v1",
            byte_size=path.stat().st_size,
            generated_token_ids=(11, 12),
        )

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _OneMisalignedPool(_Pool):
    def __init__(self, root: Path):
        super().__init__(root)
        self.misaligned_remaining = 1

    async def prefill(self, request: Any) -> RawVllmFeature:
        raw = await super().prefill(request)
        if self.misaligned_remaining:
            self.misaligned_remaining -= 1
            raw.payload["hidden_states"] = raw.payload["hidden_states"][-1:]
        return raw


class _AlwaysMisalignedPool(_Pool):
    async def prefill(self, request: Any) -> RawVllmFeature:
        raw = await super().prefill(request)
        raw.payload["hidden_states"] = raw.payload["hidden_states"][-1:]
        return raw


class _NanHiddenStatePool(_Pool):
    async def prefill(self, request: Any) -> RawVllmFeature:
        raw = await super().prefill(request)
        raw.payload["hidden_states"] = torch.full_like(
            raw.payload["hidden_states"], float("nan")
        )
        return raw


def _write_input(path: Path) -> None:
    records = [
        {"sample_id": "sample-1", "prompt": "Q1: ", "response": "A1"},
        {"sample_id": "sample-2", "prompt": "Q2: ", "response": "A2"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_run_producer_publishes_samples_then_eos(
    tmp_path: Path, target_final_norm
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _Pool(tmp_path)

    stats = asyncio.run(
        run_producer(
            _config(input_path),
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    sample_keys = [
        key
        for key, tag in transport.records.items()
        if tag.get("record_type") == "sample"
    ]
    eos_tags = [tag for tag in transport.records.values() if tag.get("status") == "eos"]
    assert stats.input_count == stats.published_count == 2
    assert stats.failed_count == stats.dropped_count == stats.pending_bytes == 0
    assert len(sample_keys) == 2
    assert eos_tags == [
        {
            "record_type": "control",
            "status": "eos",
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "run_id": "run-a",
            "total_samples": 2,
        }
    ]
    assert all(not path.exists() for path in pool.paths)
    assert pool.started and pool.closed and transport.closed
    first_fields = transport.payloads[sorted(sample_keys)[0]]
    assert tuple(first_fields["sample__hidden_states"].shape) == (3, 6)
    # The existing response feature window starts at prompt_length - 1 = 1.
    raw = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)[1:]
    torch.testing.assert_close(
        first_fields["sample__hidden_states"][:, :4], raw[:, :2].flatten(1)
    )
    torch.testing.assert_close(
        first_fields["sample__hidden_states"][:, 4:], target_final_norm(raw[:, 2])
    )
    first_key = sorted(sample_keys)[0]
    sample = decode_sample(
        first_key, transport.records[first_key], first_fields, {"run_id": "run-a"}
    )
    torch.testing.assert_close(
        sample.hidden_states[:, 4:], target_final_norm(raw[:, 2])
    )
    assert sample.metadata["last_hidden_state_norm"] == "target_final_norm"


def test_run_producer_publishes_samples_concurrently(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["publish_workers"] = 2
    transport = _ConcurrentTransport()

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )

    assert stats.published_count == 2
    assert transport.max_active_puts == 2
    assert any(tag.get("status") == "eos" for tag in transport.records.values())


def test_run_producer_logs_perf_window_at_info(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import verl_speco.standalone_tq_producer as producer_module

    monkeypatch.setattr(producer_module, "_PERF_WINDOW_SAMPLES", 2)
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)

    with caplog.at_level("INFO", logger="verl_speco.standalone_tq_producer"):
        asyncio.run(
            run_producer(
                _config(input_path),
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_Pool(tmp_path),
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Producer perf samples=2" in message for message in messages)
    assert any("perf middle samples:" in message for message in messages)
    assert any("perf slowest e2e samples:" in message for message in messages)
    assert any("perf slowest transport samples:" in message for message in messages)


def test_run_producer_retries_transient_publish_failure(
    tmp_path: Path, caplog
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer = config["speco"]["standalone_tq_producer"]
    producer["max_samples"] = 1
    producer["publish_workers"] = 1
    producer["publish_max_attempts"] = 3
    producer["publish_retry_backoff_seconds"] = 0
    transport = _TransientFailureTransport(failures=2)
    pool = _Pool(tmp_path)

    with caplog.at_level("WARNING", logger="verl_speco.standalone_tq_producer"):
        stats = asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert stats.published_count == 1
    assert transport.sample_put_attempts == 3
    assert sum("publish retry" in record.getMessage() for record in caplog.records) == 2
    assert all(not path.exists() for path in pool.paths)
    assert any(tag.get("status") == "eos" for tag in transport.records.values())


def test_direct_producer_derives_vllm_ids_from_target_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer = config["speco"]["standalone_tq_producer"]
    producer.pop("vllm_aux_hidden_state_layer_ids")
    producer["max_samples"] = 1

    stats = asyncio.run(
        run_producer(
            config,
            transport=_Transport(),
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )

    assert stats.published_count == 1


def test_direct_producer_derives_target_ids_from_vllm_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer = config["speco"]["standalone_tq_producer"]
    producer.pop("target_layer_ids")
    producer["max_samples"] = 1
    transport = _Transport()

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )

    assert stats.published_count == 1
    sample_key, sample_tag = next(
        (key, tag)
        for key, tag in transport.records.items()
        if tag.get("record_type") == "sample"
    )
    sample = decode_sample(
        sample_key,
        sample_tag,
        transport.payloads[sample_key],
        {"run_id": "run-a"},
    )
    assert sample.metadata["target_layer_ids"] == [2, 8]


def test_direct_producer_rejects_mismatched_layer_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"][
        "vllm_aux_hidden_state_layer_ids"
    ] = [3]

    with pytest.raises(ValueError, match="algorithm convention"):
        validate_producer_config(config)


def test_run_producer_does_not_republish_when_temporary_cleanup_fails(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import verl_speco.standalone_tq_producer as producer_module

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer = config["speco"]["standalone_tq_producer"]
    producer["max_samples"] = 1
    producer["publish_workers"] = 1
    producer["publish_retry_backoff_seconds"] = 0
    transport = _TransientFailureTransport(failures=0)
    pool = _Pool(tmp_path)
    cleanup_calls = 0

    def fail_cleanup(raw):
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise PermissionError(raw.temporary_path)

    monkeypatch.setattr(producer_module, "delete_temporary_result", fail_cleanup)
    with caplog.at_level("WARNING", logger="verl_speco.standalone_tq_producer"):
        stats = asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert stats.published_count == 1
    assert transport.sample_put_attempts == 1
    assert cleanup_calls == 3
    assert any(
        "cleanup failed; ignoring" in record.getMessage() for record in caplog.records
    )
    assert any(tag.get("status") == "eos" for tag in transport.records.values())


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
def test_aux_only_producer_does_not_load_or_apply_final_norm(
    tmp_path, monkeypatch, algorithm
):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("aux-only features must not load a target final norm")

    monkeypatch.setattr(
        "verl_speco.standalone_tq_producer.load_vllm_final_norm", unexpected_load
    )
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    drafter = config["actor_rollout_ref"]["rollout"]["drafter"]
    drafter["speculative_algorithm"] = algorithm
    drafter["training"]["dspark_l1_loss_alpha"] = 0.0
    transport = _Transport()
    asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )
    sample_key = next(
        key
        for key, tag in transport.records.items()
        if tag.get("record_type") == "sample"
    )
    sample = decode_sample(
        sample_key,
        transport.records[sample_key],
        transport.payloads[sample_key],
        {"run_id": "run-a"},
    )
    raw_aux = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)[1:, :2].flatten(1)
    torch.testing.assert_close(sample.hidden_states, raw_aux)


def test_run_producer_restarts_input_until_max_samples(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 5
    transport = _Transport()
    pool = _Pool(tmp_path)

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    sample_tags = [
        tag for tag in transport.records.values() if tag.get("record_type") == "sample"
    ]
    eos_tags = [tag for tag in transport.records.values() if tag.get("status") == "eos"]
    assert stats.input_count == stats.published_count == 5
    assert sorted(tag["sequence_no"] for tag in sample_tags) == [0, 1, 2, 3, 4]
    assert pool.prefill_calls == 5
    assert eos_tags[0]["total_samples"] == 5


def test_run_producer_skips_consumed_sequences_before_vllm(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [0],
        optimizer_step=1,
        input_path=input_path,
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    producer_cfg["max_samples"] = 2
    transport = _Transport()
    pool = _Pool(tmp_path)

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    sequence_nos = sorted(
        int(tag["sequence_no"])
        for tag in transport.records.values()
        if tag.get("record_type") == "sample"
    )
    assert stats.input_count == 2
    assert pool.prefill_calls == 2
    assert sequence_nos == [1, 2]


def test_run_producer_resumes_after_a_fully_consumed_epoch(tmp_path: Path) -> None:
    """A resume whose first pass is entirely consumed must advance epochs.

    Regression: the zero-production guard used to fire before ``epoch += 1``,
    so a checkpoint that had consumed a whole input epoch could never resume.
    """
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    checkpoint_path = tmp_path / "draft_step_2"
    save_standalone_resume(
        checkpoint_path,
        [0, 1],
        optimizer_step=2,
        input_path=input_path,
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    producer_cfg["max_samples"] = 2
    transport = _Transport()
    pool = _Pool(tmp_path)

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    sequence_nos = sorted(
        int(tag["sequence_no"])
        for tag in transport.records.values()
        if tag.get("record_type") == "sample"
    )
    assert stats.published_count == 2
    assert sequence_nos == [2, 3]
    assert pool.closed and transport.closed


def test_run_producer_generates_response_for_verl_chat_prompt(tmp_path: Path) -> None:
    input_path = tmp_path / "dapo.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Q3"}],
                "reward_model": {"ground_truth": "42"},
                "extra_info": {"index": "dapo-row"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    transport = _Transport()
    pool = _Pool(tmp_path)

    stats = asyncio.run(
        run_producer(
            _config(input_path),
            transport=transport,
            tokenizer=_ChatTokenizer(),
            client_pool=pool,
        )
    )

    sample_keys = [
        key
        for key, tag in transport.records.items()
        if tag.get("record_type") == "sample"
    ]
    assert stats.input_count == stats.published_count == 1
    assert pool.generate_calls == 1
    assert pool.prefill_calls == 1
    assert len(sample_keys) == 1
    fields = transport.payloads[sample_keys[0]]
    assert fields["sample__input_ids"].tolist() == [10, 11]
    assert fields["sample__loss_mask"].tolist() == [0.0, 1.0]


def test_run_producer_bounds_consecutive_generated_filters(tmp_path: Path) -> None:
    """A target that always generates untrainable samples must abort, not loop."""

    input_path = tmp_path / "dapo.jsonl"
    input_path.write_text(
        json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n",
        encoding="utf-8",
    )
    transport = _Transport()
    pool = _Pool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    config["speco"]["standalone_tq_producer"]["max_inflight_requests"] = 1
    # No generated completion can reach this many supervised tokens, so every
    # generated sample is filtered after generation and replaced.
    config["speco"]["standalone_tq_producer"]["min_supervised_tokens"] = 99
    config["speco"]["standalone_tq_producer"]["max_consecutive_feature_drops"] = 2

    with pytest.raises(RuntimeError, match="max_consecutive_feature_drops=2"):
        asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_ChatTokenizer(),
                client_pool=pool,
            )
        )

    assert pool.closed and transport.closed


def test_run_producer_renders_off_the_event_loop(
    tmp_path: Path, monkeypatch
) -> None:
    """The synchronous /render calls must not run on the event-loop thread."""

    import verl_speco.standalone_tq_producer as producer_module

    input_path = tmp_path / "chat.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "chat-1",
                "prompt": [{"role": "user", "content": "Q1"}],
                "response": "A1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    render_threads: list[int] = []

    def fake_render(messages, *, add_generation_prompt, max_length=None):
        render_threads.append(threading.get_ident())
        if add_generation_prompt:
            return [1, 2, 3]
        if len(messages) > 1:
            return [1, 2, 3, 4, 5]
        return [1, 2, 3]

    def fake_build_render_fn(endpoint, *, timeout):
        return fake_render

    monkeypatch.setattr(producer_module, "build_render_fn", fake_build_render_fn)

    transport = _Transport()
    pool = _Pool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["render_boundary"] = {"enabled": True}
    main_thread = threading.main_thread().ident

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    assert stats.published_count == 1
    assert render_threads
    assert all(ident != main_thread for ident in render_threads)


def test_run_producer_replaces_misaligned_sample_before_eos(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _OneMisalignedPool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    sample_tags = [
        tag
        for tag in transport.records.values()
        if tag.get("record_type") == "sample"
    ]
    eos = next(tag for tag in transport.records.values() if tag.get("status") == "eos")
    assert stats.input_count == 3
    assert stats.published_count == 2
    assert stats.dropped_count == 1
    assert len(sample_tags) == 2
    assert eos["total_samples"] == 2
    assert all(not path.exists() for path in pool.paths)
    assert pool.closed and transport.closed


@pytest.mark.parametrize(
    "pool_factory", [_AlwaysMisalignedPool, _NanHiddenStatePool]
)
def test_run_producer_bounds_consecutive_feature_drops(
    tmp_path: Path, pool_factory: type[_Pool]
) -> None:
    """A persistently broken endpoint must abort, not replace samples forever."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = pool_factory(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    config["speco"]["standalone_tq_producer"]["max_inflight_requests"] = 1
    config["speco"]["standalone_tq_producer"]["max_consecutive_feature_drops"] = 2

    with pytest.raises(RuntimeError, match="max_consecutive_feature_drops=2"):
        asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert pool.closed and transport.closed


def test_run_producer_feature_drop_limit_zero_fails_on_first_drop(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    config["speco"]["standalone_tq_producer"]["max_inflight_requests"] = 1
    config["speco"]["standalone_tq_producer"]["max_consecutive_feature_drops"] = 0

    with pytest.raises(RuntimeError, match="max_consecutive_feature_drops=0"):
        asyncio.run(
            run_producer(
                config,
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_OneMisalignedPool(tmp_path),
            )
        )


def test_run_producer_transient_feature_drop_resets_breaker(
    tmp_path: Path,
) -> None:
    """A drop below the bound must not abort once a later conversion succeeds."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _OneMisalignedPool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    config["speco"]["standalone_tq_producer"]["max_inflight_requests"] = 1
    config["speco"]["standalone_tq_producer"]["max_consecutive_feature_drops"] = 1

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    assert stats.dropped_count == 1
    assert stats.published_count == 2


def test_run_producer_logs_error_when_nothing_is_published(
    tmp_path: Path, caplog
) -> None:
    """A single pass that drops every sample must be visible, not silent."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _AlwaysMisalignedPool(tmp_path)
    config = _config(input_path)
    # max_samples<=0 means one pass; a single worker avoids the fake pool's
    # per-sample-id temporary-file race.
    config["speco"]["standalone_tq_producer"]["max_inflight_requests"] = 1

    with caplog.at_level("ERROR", logger="verl_speco.standalone_tq_producer"):
        stats = asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert stats.published_count == 0
    assert stats.dropped_count == 2
    assert "published no samples" in caplog.text


def test_run_producer_put_failure_keeps_temporary_file_and_omits_eos(
    tmp_path: Path,
    caplog,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport(fail_sample_put=True)
    pool = _Pool(tmp_path)

    with pytest.raises(RuntimeError, match="put failed"):
        asyncio.run(
            run_producer(
                _config(input_path),
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert any(path.exists() for path in pool.paths)
    assert not any(tag.get("status") == "eos" for tag in transport.records.values())
    assert pool.closed and transport.closed
    assert "Producer task failed task=publisher" in caplog.text
    assert "'tq_put'" in caplog.text


def test_validate_producer_rejects_consumer_partition_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path / "input.jsonl")
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"][
        "partition_id"
    ] = "other"

    with pytest.raises(ValueError, match="partition_id"):
        validate_producer_config(config)


def test_pool_close_failure_does_not_skip_transport_close(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _Pool(tmp_path, close_error=RuntimeError("pool close failed"))

    with pytest.raises(RuntimeError, match="pool close failed"):
        asyncio.run(
            run_producer(
                _config(input_path),
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )

    assert pool.closed and transport.closed


def test_run_producer_errors_when_every_row_is_filtered(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _Pool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    # No feature window can reach this many supervised tokens, so every row is
    # filtered (SampleFilteredError) and no request is produced.
    config["speco"]["standalone_tq_producer"]["min_supervised_tokens"] = 99

    with pytest.raises(ValueError, match="filtered every newly considered sample"):
        asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )


def test_config_int_preserves_explicit_zero() -> None:
    from verl_speco.config import config_int

    assert config_int({"max_consecutive_errors": 0}, "max_consecutive_errors", 20) == 0
    assert config_int({}, "max_consecutive_errors", 20) == 20
    assert (
        config_int({"max_consecutive_errors": None}, "max_consecutive_errors", 20) == 20
    )
    assert (
        config_int({"max_consecutive_errors": "7"}, "max_consecutive_errors", 20) == 7
    )


def test_run_producer_preserves_explicit_zero_max_consecutive_errors(
    tmp_path: Path,
) -> None:
    """An explicit ``max_consecutive_errors=0`` must fail on the first bad row.

    Regression: ``config.get(key, 20) or 20`` silently turned ``0`` into ``20``,
    so the circuit breaker never fired for the documented "fail fast" setting.
    """
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"sample_id": "sample-1", "prompt": "Q1: ", "response": "A1"})
        + "\n"
        + "{not valid json\n",
        encoding="utf-8",
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["on_error"] = "skip"
    producer_cfg["max_consecutive_errors"] = 0
    transport = _Transport()
    pool = _Pool(tmp_path)

    with pytest.raises(RuntimeError, match="max_consecutive_errors=0"):
        asyncio.run(
            run_producer(
                config,
                transport=transport,
                tokenizer=_Tokenizer(),
                client_pool=pool,
            )
        )


def _ready_tag(sequence_no: int = 0) -> dict[str, Any]:
    return {
        "record_type": "sample",
        "status": "ready",
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "run_id": "run-a",
        "sample_id": f"sample-{sequence_no}",
        "sequence_no": sequence_no,
    }


def test_drain_pending_samples_returns_after_consumer_clears() -> None:
    class _DrainTransport:
        def __init__(self) -> None:
            self.calls = 0

        def list_samples(self) -> dict[str, dict[str, Any]]:
            self.calls += 1
            # Emulate a consumer that acks the batch on its third poll.
            if self.calls >= 3:
                return {}
            return {"sample:0": _ready_tag()}

    transport = _DrainTransport()
    asyncio.run(
        _drain_pending_samples(
            transport, "run-a", timeout=5.0, poll_interval=0.0
        )
    )
    assert transport.calls >= 3


def test_drain_pending_samples_times_out_without_failing() -> None:
    class _StuckTransport:
        def __init__(self) -> None:
            self.calls = 0

        def list_samples(self) -> dict[str, dict[str, Any]]:
            self.calls += 1
            return {"sample:0": _ready_tag()}

    transport = _StuckTransport()
    asyncio.run(
        _drain_pending_samples(
            transport, "run-a", timeout=0.05, poll_interval=0.0
        )
    )
    assert transport.calls >= 1


def test_drain_pending_samples_disabled_with_zero_timeout() -> None:
    class _UnexpectedTransport:
        def list_samples(self) -> dict[str, dict[str, Any]]:
            raise AssertionError("drain must not poll when disabled")

    asyncio.run(
        _drain_pending_samples(
            _UnexpectedTransport(), "run-a", timeout=0.0, poll_interval=0.0
        )
    )
