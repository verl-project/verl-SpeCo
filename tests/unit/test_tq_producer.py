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

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verl_speco.producer.vllm_feature_client import RawVllmFeature
from verl_speco.standalone_tq_producer import (
    _apply_direct_producer_defaults,
    _drain_pending_samples,
    run_producer,
    validate_producer_config,
)
from verl_speco.trainer.standalone_resume import save_standalone_resume
from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION
from verl_speco.transport.drafter_sample_protocol import decode_sample


@pytest.mark.parametrize("preserve", [None, False, True])
@pytest.mark.parametrize("prefix", ["", "++"])
def test_direct_producer_prompt_context_default_and_override(preserve, prefix):
    from omegaconf import OmegaConf

    key = "actor_rollout_ref.rollout.drafter.training.preserve_prompt_hidden_states"
    base = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "verl_speco/config/speco_base.yaml"
    )
    assert OmegaConf.select(base, key) is False
    config = OmegaConf.merge(base)
    overrides = ["speco.standalone_tq_producer.max_feature_length=3"]
    if preserve is not None:
        OmegaConf.update(config, key, preserve)
        overrides.append(f"{prefix}{key}={str(preserve).lower()}")
    _apply_direct_producer_defaults(config, overrides)
    expected = True if preserve is None else preserve
    assert OmegaConf.select(config, key) is expected
    assert OmegaConf.select(base, key) is False


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
                        "dflash_block_size": 2,
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


@pytest.mark.parametrize("algorithm,minimum", [("DSPARK", 2), ("DFLASH", 4)])
def test_producer_rejects_response_budget_without_training_anchor(algorithm, minimum):
    config = _config(Path("unused.jsonl"))
    drafter = config["actor_rollout_ref"]["rollout"]["drafter"]
    drafter["speculative_algorithm"] = algorithm
    drafter["training"]["dflash_block_size"] = 4
    producer = config["speco"]["standalone_tq_producer"]
    producer["max_feature_length"] = minimum - 1
    with pytest.raises(ValueError, match=f"at least {minimum}"):
        validate_producer_config(config)
    for budget in (0, minimum):
        producer["max_feature_length"] = budget
        validate_producer_config(config)


@pytest.mark.parametrize("key,value", [
    ("max_jsonl_line_bytes", -1), ("max_jsonl_line_bytes", True),
    ("max_indexed_records", -1), ("max_indexed_records", 1.5),
    ("verify_indexed_input", "true"),
])
def test_producer_rejects_invalid_jsonl_limits(key, value):
    config = _config(Path("unused.jsonl"))
    config["speco"]["standalone_tq_producer"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_producer_config(config)


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
        token_ids = torch.tensor([*request.prompt_token_ids, 11, 12], dtype=torch.int64)
        hidden = torch.arange(token_ids.numel() * 3 * 2, dtype=torch.float32).reshape(
            token_ids.numel(), 3, 2
        )
        return RawVllmFeature(
            payload={"token_ids": token_ids, "hidden_states": hidden},
            temporary_path=str(path),
            endpoint_url="http://vllm:8000/v1",
            byte_size=path.stat().st_size,
            generated_token_ids=(11, 12, 13),
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


class _OneFailingPrefillPool(_Pool):
    def __init__(self, root: Path):
        super().__init__(root)
        self.fail_remaining = 1

    async def prefill(self, request: Any) -> RawVllmFeature:
        if self.fail_remaining:
            self.fail_remaining -= 1
            raise RuntimeError("vLLM request failed after 4 attempts")
        return await super().prefill(request)


class _AlwaysFailingPrefillPool(_Pool):
    async def prefill(self, request: Any) -> RawVllmFeature:
        raise RuntimeError("vLLM request failed after 4 attempts")


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


@pytest.mark.parametrize("preserve", [None, False, True])
def test_run_producer_publishes_samples_then_eos(
    tmp_path: Path, target_final_norm, preserve
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _Pool(tmp_path)
    config = _config(input_path)
    if preserve is not None:
        config["actor_rollout_ref"]["rollout"]["drafter"]["training"][
            "preserve_prompt_hidden_states"
        ] = preserve

    stats = asyncio.run(
        run_producer(
            config,
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
    start = 1 if preserve is False else 0
    assert tuple(first_fields["sample__hidden_states"].shape) == (4 - start, 6)
    raw = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)[start:]
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


def test_native_reference_producer_dispatch_and_read_with_injected_source(tmp_path, monkeypatch):
    import ctypes
    import json
    from queue import Queue
    from types import SimpleNamespace
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport

    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
        def transfer_sync_read(self, segment, local, remote, size):
            ctypes.memmove(local, remote, size)
            return 0

    def make_transport():
        return NativeFeatureTransport(run_id="run-a", generation=1, endpoint="localhost:1234",
                                      metadata_server="metadata", device="cpu", max_bytes=8 * 1024**2,
                                      max_slots=2, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    source, receiver = make_transport(), make_transport()
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_published_references"] = 2
    config["speco"]["standalone_tq_producer"]["hidden_states_store"] = {"backend": "native"}
    with pytest.raises(ValueError, match="injected native"):
        validate_producer_config(config)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {
        "payload_kind": "vllm_reference"
    }
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    events = Queue()
    actor = StandaloneProducerActor(config, events, reference_batch_size=2, native_transport=source)
    native_store = SimpleNamespace(backend="native")

    class Pool(_Pool):
        async def prefill_reference(self, request):
            descriptor = source.publish({"ids": torch.tensor(request.prompt_token_ids)},
                                        lease_id=str(request.sequence_no))
            return VllmResponse(None, "http://vllm:8000/v1", store=native_store,
                                handle=json.dumps(descriptor, sort_keys=True, separators=(",", ":")))

    def create_pool(*args, **kwargs):
        assert kwargs["hidden_states_store"] is native_store
        return Pool(tmp_path)
    monkeypatch.setattr(producer, "VllmFeatureClientPool", create_pool)
    async def run():
        result = await run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), hidden_states_store=native_store,
                                    on_references=actor._on_references, reference_batch_size=2)
        entries = events.get_nowait()["entries"]
        assert result.published_count == len(entries) == 2
        keys = [entry["key"] for entry in entries]
        await actor._on_metrics(2, {})
        assert actor._metrics_pending[-1]["metrics"]["producer/reference_native_span_bytes"] == sum(
            json.loads(entry["tag"]["handle"])["nbytes"] for entry in entries)
        await actor.register_dispatch(actor._incarnation, 0, "receipt", keys, 1)
        for entry in entries:
            descriptor = json.loads(entry["tag"]["handle"])
            assert entry["tag"]["backend"] == "native"
            assert receiver.read(descriptor, output_device="cpu")["ids"].numel() > 0
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
        await actor.release_references(actor._incarnation, 0, "receipt", keys)
        assert not source._slots and not actor._active_references
        await actor._on_metrics(2, {})
        assert actor._metrics_pending[-1]["metrics"]["producer/reference_native_span_bytes"] == 0
    asyncio.run(run())
    source.close()
    receiver.close()


@pytest.mark.parametrize("batch_size", [None, 2, 3])
@pytest.mark.parametrize("backend", ["file", "mooncake"])
def test_reference_producer_never_reads_or_normalizes_hidden(tmp_path, monkeypatch, batch_size, backend):
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["hidden_states_store"] = {"backend": backend}
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {
        "payload_kind": "vllm_reference"
    }
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    monkeypatch.setattr(producer, "load_vllm_final_norm", lambda *a, **kw: pytest.fail("Producer loaded norm"))
    monkeypatch.setattr(producer, "feature_from_vllm_payload", lambda *a, **kw: pytest.fail("Producer converted hidden"))
    monkeypatch.setattr(producer, "encode_sample", lambda *a, **kw: pytest.fail("Encoded tensor fields"))

    class ReferencePool(_Pool):
        async def prefill(self, request):
            pytest.fail("Eager hidden read was requested")

        async def prefill_reference(self, request):
            path = self.root / f"{request.sequence_no}.safetensors"
            # Invalid tensor data is intentional: Producer must never open it.
            path.write_bytes(b"not a tensor file")
            self.paths.append(path)
            if backend == "mooncake":
                return VllmResponse(None, "http://vllm:8000/v1", handle=f"hs-{request.sequence_no}",
                                    transfer_params={"handle": f"hs-{request.sequence_no}", "version": 1})
            return VllmResponse(str(path), "http://vllm:8000/v1")

    class MetadataTransport(_Transport):
        def put_metadata(self, key, *, tag):
            self.records[key] = tag

    batches = []
    measured_rows = []
    original_metrics = producer.producer_window_metrics

    def capture_metrics(rows, elapsed):
        measured_rows.extend(rows)
        return original_metrics(rows, elapsed)

    monkeypatch.setattr(producer, "producer_window_metrics", capture_metrics)

    async def submit(entries):
        batches.append(entries)
        await asyncio.sleep(0.05)
        return {"transport_seconds": 0.01, "retry_wait_seconds": 0.005, "attempts": 2}

    class NoTQ:
        def __getattr__(self, name):
            pytest.fail(f"Ray file publication accessed TQ: {name}")

    pool, transport = ReferencePool(tmp_path), MetadataTransport()
    options = {} if batch_size is None else {
        "on_references": submit, "reference_batch_size": batch_size,
    }
    result = asyncio.run(run_producer(
        config, transport=transport if batch_size is None else NoTQ(),
        tokenizer=_Tokenizer(), client_pool=pool, **options,
    ))
    if batch_size is not None:
        assert [len(batch) for batch in batches] == [2]
        assert all(item["tag"]["backend"] == backend for item in batches[0])
        if backend == "mooncake":
            assert all(item["tag"]["transfer_params"]["handle"] == item["tag"]["handle"] for item in batches[0])
        assert all(path.exists() for path in pool.paths)
        assert all(item["tag"]["payload_kind"] == "vllm_reference" for item in batches[0])
        assert result.published_count == 2
        assert len(measured_rows) == 2
        for row in measured_rows:
            timing = row["timing"]
            assert timing["publish_transport"] == 0.01  # Full RPC latency, not divided by two.
            assert timing["publish_attempts"] == 2
            assert timing["metadata_submit"] > 0
            parts = [timing[name] for name in ("metadata_build", "publish_enqueue", "publish_residence", "publish_batch_wait")]
            assert all(value >= 0 for value in parts)
            assert sum(parts) == pytest.approx(timing["publish_queue"], abs=0.005)
            assert timing["metadata_ready_delay"] >= 0
            assert timing["active"] + timing["backpressure"] <= timing["e2e"] + 1e-5
            assert not any(key.startswith("tq_") for key in timing)
        return
    assert result.published_count == 2
    records = [tag for tag in transport.records.values() if tag.get("record_type") == "sample"]
    assert len(records) == 2 and all(tag["payload_kind"] == "vllm_reference" for tag in records)
    assert all(path.exists() for path in pool.paths)
    assert all(key not in transport.payloads for key, tag in transport.records.items() if tag in records)


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


@pytest.mark.parametrize("ending", [False, True])
def test_reference_batch_flushes_on_timeout_or_end(ending):
    from verl_speco.standalone_tq_producer import _PUBLISH_DONE, _collect_reference_batch

    async def run():
        queue = asyncio.Queue()
        if ending:
            await queue.put(_PUBLISH_DONE)
        batch, done = await _collect_reference_batch(queue, "first", 4, timeout=0.01)
        assert batch == ["first"] and done is ending

    asyncio.run(run())


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


@pytest.mark.parametrize("count", [37, 100, 137, 200])
def test_producer_emits_complete_and_final_partial_metric_windows(tmp_path, count):
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = count
    windows = []

    async def on_metrics(total, metrics):
        windows.append((total, metrics))

    stats = asyncio.run(
        run_producer(
            config,
            transport=_Transport(),
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
            before_request=lambda: asyncio.sleep(0),
            on_metrics=on_metrics,
        )
    )
    expected = list(range(100, count + 1, 100))
    if count % 100:
        expected.append(count)
    assert stats.published_count == count
    assert [total for total, _ in windows] == expected
    for _, metrics in windows:
        assert metrics["producer/samples_per_second"] > 0
        assert "producer/generation_time" not in metrics
        assert metrics["producer/other_time"] >= 0


def test_producer_cancellation_flushes_partial_window(tmp_path):
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg.update(max_samples=100, publish_workers=1)
    windows = []

    async def exercise():
        ready = asyncio.Event()
        published = 0

        async def on_published(sequence_no):
            nonlocal published
            published += 1
            if published == 37:
                ready.set()
                await asyncio.Event().wait()

        async def on_metrics(total, metrics):
            windows.append((total, metrics))

        task = asyncio.create_task(
            run_producer(
                config,
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_Pool(tmp_path),
                before_request=lambda: asyncio.sleep(0),
                on_published=on_published,
                on_metrics=on_metrics,
            )
        )
        await asyncio.wait_for(ready.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert [total for total, _ in windows] == [37]


def test_run_producer_retries_transient_publish_failure(tmp_path: Path, caplog) -> None:
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
    config["speco"]["standalone_tq_producer"]["vllm_aux_hidden_state_layer_ids"] = [3]

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
@pytest.mark.parametrize("preserve", [None, False, True])
def test_aux_only_producer_does_not_load_or_apply_final_norm(
    tmp_path, monkeypatch, algorithm, preserve
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
    if preserve is not None:
        drafter["training"]["preserve_prompt_hidden_states"] = preserve
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
    start = 1 if preserve is False else 0
    raw_aux = (
        torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)[start:, :2].flatten(1)
    )
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


def test_run_producer_shuffle_permutes_per_epoch_deterministically(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["max_samples"] = 4  # two epochs over the two-record file
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 42
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

    sample_tags = sorted(
        (
            tag
            for tag in transport.records.values()
            if tag.get("record_type") == "sample"
        ),
        key=lambda tag: int(tag["sequence_no"]),
    )
    assert stats.input_count == 4
    assert [int(tag["sequence_no"]) for tag in sample_tags] == [0, 1, 2, 3]
    # sequence_no is run-global and follows the permuted queue order, so the
    # sample_id sequence must equal the per-epoch permutations (speculators
    # NumPy default_rng(seed + epoch).permutation recipe).
    records = ["sample-1", "sample-2"]
    epoch0 = np.random.default_rng(42).permutation(2)
    epoch1 = np.random.default_rng(43).permutation(2)
    expected = [records[i] for i in [*epoch0, *epoch1]]
    assert [tag["sample_id"] for tag in sample_tags] == expected


@pytest.mark.parametrize("resume", [False, True])
def test_producer_stable_implicit_ids_across_shuffled_epochs(tmp_path, resume):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(''.join(json.dumps({"prompt": f"Q{i % 2 + 1}: ", "response": f"A{i % 2 + 1}"}) + "\n"
                                  for i in range(3)), encoding="utf-8")
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg.update(max_samples=6, shuffle=True, shuffle_seed=42, stable_source_ids=True)
    if resume:
        from verl_speco.trainer.standalone_resume import producer_data_recipe
        checkpoint = tmp_path / "checkpoint"
        save_standalone_resume(checkpoint, [0, 1, 2], optimizer_step=1, input_path=input_path,
                               producer_shuffle={"shuffle": True, "shuffle_seed": 42,
                                                 "data_recipe": producer_data_recipe(producer_cfg)})
        producer_cfg.update(max_samples=3, resume_checkpoint_path=str(checkpoint))
    transport = _Transport()
    asyncio.run(run_producer(config, transport=transport, tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))
    tags = sorted((tag for tag in transport.records.values() if tag.get("record_type") == "sample"),
                  key=lambda tag: tag["sequence_no"])
    expected = [f"train-{int(index):06d}" for epoch in range(2)
                for index in np.random.default_rng(42 + epoch).permutation(3)]
    assert [tag["sequence_no"] for tag in tags] == (list(range(3, 6)) if resume else list(range(6)))
    assert [tag["sample_id"] for tag in tags] == (expected[3:] if resume else expected)


@pytest.mark.parametrize("storage_format", ["tensor", "gzip"])
def test_run_producer_shuffle_resume_replays_epoch_order(tmp_path: Path, storage_format: str) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    # Epoch 0 already fully consumed: sequence_nos 0 and 1.
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [0, 1],
        optimizer_step=1,
        input_path=input_path,
        producer_shuffle={"shuffle": True, "shuffle_seed": 42},
        storage_format=storage_format,
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    producer_cfg["max_samples"] = 2
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 42
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

    sample_tags = sorted(
        (
            tag
            for tag in transport.records.values()
            if tag.get("record_type") == "sample"
        ),
        key=lambda tag: int(tag["sequence_no"]),
    )
    assert stats.input_count == 2
    # Only the replayed epoch-1 permutation is queued, with fresh run-global
    # sequence numbers continuing after the consumed prefix.
    assert [int(tag["sequence_no"]) for tag in sample_tags] == [2, 3]
    records = ["sample-1", "sample-2"]
    epoch1 = np.random.default_rng(43).permutation(2)
    assert [tag["sample_id"] for tag in sample_tags] == [records[i] for i in epoch1]


def test_producer_trains_from_committed_replay_without_regeneration(tmp_path, target_final_norm, monkeypatch):
    from verl_speco.producer.sequence_replay import SequenceReplayWriter

    replay_path = tmp_path / "replay"
    writer = SequenceReplayWriter(
        replay_path, identity={"model": "/target", "model_revision": "rev-a",
                               "tokenizer_fingerprint": "sha256:tokenizer"}, segment_rows=1,
    )
    for index in range(2):
        writer.append({"sequence_no": index, "sample_id": f"replay-{index}",
                       "input_ids": [1, 2, 11, 12, 13], "loss_mask": [0, 0, 1, 1, 1],
                       "source_metadata": {}})
    writer.close()
    config = _config(replay_path)
    config["speco"]["standalone_tq_producer"]["shuffle"] = True
    config["speco"]["standalone_tq_producer"]["max_samples"] = 4
    transport, pool = _Transport(), _Pool(tmp_path)
    monkeypatch.setattr("verl_speco.standalone_tq_producer._load_tokenizer",
                        lambda *_: pytest.fail("replay should not tokenize again"))
    stats = asyncio.run(run_producer(config, transport=transport, client_pool=pool))
    tags = sorted((tag for tag in transport.records.values() if tag.get("record_type") == "sample"),
                  key=lambda tag: tag["sequence_no"])
    assert stats.published_count == 4 and pool.generate_calls == 0
    assert [tag["sequence_no"] for tag in tags] == [0, 1, 2, 3]
    assert {tag["sample_id"] for tag in tags[:2]} == {"replay-0", "replay-1"}
    assert {tag["sample_id"] for tag in tags[2:]} == {"replay-0", "replay-1"}


def test_producer_rejects_replay_from_different_target(tmp_path, target_final_norm):
    from verl_speco.producer.sequence_replay import SequenceReplayWriter

    replay_path = tmp_path / "replay"
    writer = SequenceReplayWriter(replay_path, identity={"model": "other"}, segment_rows=1)
    writer.append({"sequence_no": 0, "sample_id": "x", "input_ids": [1, 2, 3, 4, 5],
                   "loss_mask": [0, 0, 1, 1, 1], "source_metadata": {}})
    writer.close()
    with pytest.raises(ValueError, match="target or tokenizer identity differs"):
        asyncio.run(run_producer(_config(replay_path), transport=_Transport(), client_pool=_Pool(tmp_path)))


def test_generate_replay_then_resume_training_without_regeneration(tmp_path, target_final_norm):
    from verl_speco.producer.sequence_replay import SequenceReplayWriter, generate_sequences

    source = tmp_path / "prompts.jsonl"
    source.write_text('''{"sample_id":"s1","prompt":"Q1: "}\n{"sample_id":"s2","prompt":"Q2: "}\n''', encoding="utf-8")
    directory = tmp_path / "replay"
    identity = {"model": "/target", "model_revision": "rev-a", "tokenizer_fingerprint": "sha256:tokenizer"}
    writer = SequenceReplayWriter(directory, identity=identity, segment_rows=1)
    calls = []
    async def generate(request):
        calls.append(request.sample_id)
        return [11, 12, 13]
    settings = {"generation_max_tokens": 4, "max_sequence_length": 16}
    assert asyncio.run(generate_sequences(source, writer=writer, tokenizer=_Tokenizer(), generate=generate, config=settings)) == 2
    writer.close()
    writer = SequenceReplayWriter(directory, identity=identity)
    assert asyncio.run(generate_sequences(source, writer=writer, tokenizer=_Tokenizer(), generate=generate, config=settings)) == 0
    writer.close()
    assert calls == ["s1", "s2"]
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0, 1], optimizer_step=1, input_path=directory,
                           producer_shuffle={"shuffle": True, "shuffle_seed": 42})
    config = _config(directory)
    producer = config["speco"]["standalone_tq_producer"]
    producer.update(resume_checkpoint_path=str(checkpoint), shuffle=True, shuffle_seed=42, max_samples=2)
    transport, pool = _Transport(), _Pool(tmp_path)
    stats = asyncio.run(run_producer(config, transport=transport, client_pool=pool))
    tags = sorted((tag for tag in transport.records.values() if tag.get("record_type") == "sample"),
                  key=lambda tag: tag["sequence_no"])
    assert stats.published_count == 2 and pool.generate_calls == 0
    assert [tag["sequence_no"] for tag in tags] == [2, 3]
    assert {tag["sample_id"] for tag in tags} == {"s1", "s2"}


def test_sequence_only_cli_generates_and_reopens_committed_corpus(tmp_path, monkeypatch):
    import httpx
    from transformers import AutoTokenizer
    from verl_speco.producer.sequence_replay import SequenceReplayReader, main

    source = tmp_path / "prompts.jsonl"
    source.write_text('{"sample_id":"s1","prompt":"Q1: "}\n', encoding="utf-8")
    directory = tmp_path / "replay"
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"prompt_token_ids": [1, 2], "token_ids": [11, 12, 13]}]}
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, path, *, json, timeout):
            calls.append((path, json, timeout))
            return Response()
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    args = ["--input", str(source), "--output", str(directory), "--model", "/target",
            "--tokenizer", "/target", "--endpoint", "http://localhost:8000/v1",
            "--temperature", "0.7", "--max-tokens", "4"]
    assert main(args) == 0
    row = next(SequenceReplayReader(directory).iter_rows())
    assert row["input_ids"] == [1, 2, 11, 12, 13] and row["loss_mask"] == [0, 0, 1, 1, 1]
    assert calls[0][1]["temperature"] == 0.7 and calls[0][1]["return_token_ids"] is True
    assert main(args) == 0 and len(calls) == 1


def test_sequence_generation_rejects_invalid_response_without_publishing(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayWriter, generate_sequences

    source = tmp_path / "prompt.jsonl"
    source.write_text('{"sample_id":"s1","prompt":"Q1: "}\n', encoding="utf-8")
    directory = tmp_path / "replay"
    writer = SequenceReplayWriter(directory, identity={"model": "v1"}, segment_rows=1)
    async def invalid(request):
        return [1] * (request.max_tokens + 1)
    with pytest.raises(ValueError, match="invalid token IDs"):
        asyncio.run(generate_sequences(source, writer=writer, tokenizer=_Tokenizer(), generate=invalid,
                                       config={"generation_max_tokens": 4, "max_sequence_length": 16}))
    writer.close(commit=False)
    assert not (directory / "manifest.json").exists()


def test_sequence_generation_preserves_existing_response_prefill_boundary(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayReader, SequenceReplayWriter, generate_sequences

    source = tmp_path / "input.jsonl"
    source.write_text('{"sample_id":"s1","prompt":"Q1: ","response":"A1"}\n', encoding="utf-8")
    writer = SequenceReplayWriter(tmp_path / "replay", identity={"model": "v1"}, segment_rows=1)
    async def unexpected(_):
        pytest.fail("a row with a response must not call generation")
    assert asyncio.run(generate_sequences(source, writer=writer, tokenizer=_Tokenizer(),
                                           generate=unexpected, config={"max_feature_length": 0})) == 1
    writer.close()
    row = next(SequenceReplayReader(tmp_path / "replay").iter_rows())
    assert row["input_ids"] == [1, 2, 3, 4] and row["prefill_excludes_last"] is False


def test_run_producer_shuffle_resume_with_filtered_rows_stays_aligned(
    tmp_path: Path,
) -> None:
    """Shuffle + prompt-only rows + partial resume stay sequence-aligned.

    Filtered rows consume a sequence_no like any other row (#110 semantics).
    A resumed run must replay the same permutation, re-filter the same
    prompt-only row, and republish only the unconsumed samples under their
    original sequence numbers.
    """
    input_path = tmp_path / "input.jsonl"
    records = [
        {"sample_id": "sample-1", "prompt": "Q1: ", "response": "A1"},
        {"sample_id": "sample-2", "prompt": "Q2: ", "response": "A2"},
        # Prompt-only row: filtered under the default on_missing_response=skip.
        {"prompt": [{"role": "user", "content": "Q3"}]},
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    base_config = _config(input_path)
    producer_cfg = base_config["speco"]["standalone_tq_producer"]
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 42

    # First run: one full shuffled pass; the prompt-only row is filtered and
    # consumes the remaining sequence number of {0, 1, 2}.
    transport = _Transport()
    stats_first = asyncio.run(
        run_producer(
            base_config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )
    assert stats_first.input_count == 2  # yielded requests only
    assert stats_first.filtered_count == 1
    assert stats_first.published_count == 2
    seq_by_sample = {
        tag["sample_id"]: int(tag["sequence_no"])
        for tag in transport.records.values()
        if tag.get("record_type") == "sample"
    }
    assert len(seq_by_sample) == 2
    assert set(seq_by_sample.values()) < {0, 1, 2}

    # Resume having consumed only the lower-numbered sample of epoch 0.
    consumed = min(seq_by_sample.values())
    remaining_sample = max(seq_by_sample, key=lambda s: seq_by_sample[s])
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [consumed],
        optimizer_step=1,
        input_path=input_path,
        producer_shuffle={"shuffle": True, "shuffle_seed": 42},
    )
    resume_config = _config(input_path)
    resume_cfg = resume_config["speco"]["standalone_tq_producer"]
    resume_cfg["shuffle"] = True
    resume_cfg["shuffle_seed"] = 42
    resume_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    resume_transport = _Transport()

    stats_resumed = asyncio.run(
        run_producer(
            resume_config,
            transport=resume_transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )
    resumed_tags = [
        tag
        for tag in resume_transport.records.values()
        if tag.get("record_type") == "sample"
    ]
    assert stats_resumed.input_count == 1
    assert stats_resumed.filtered_count == 1  # same row re-filtered
    assert stats_resumed.published_count == 1  # only the unconsumed sample
    assert len(resumed_tags) == 1
    assert resumed_tags[0]["sample_id"] == remaining_sample
    # The surviving sample keeps its original sequence_no: same permutation,
    # same assignment, the consumed one skipped in place.
    assert int(resumed_tags[0]["sequence_no"]) == seq_by_sample[remaining_sample]


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("consumed_epochs", [1, 3])
@pytest.mark.parametrize("reference", [False, True])
def test_resume_advances_epoch_when_only_unconsumed_rows_are_filtered(tmp_path, monkeypatch, shuffle, consumed_epochs, reference):
    import numpy as np

    source = tmp_path / "input.jsonl"
    records = [
        {"sample_id": "valid", "prompt": "Q1: ", "response": "A1"},
        {"sample_id": "filtered", "prompt": "Q2: "},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    def epoch_order(epoch):
        return list(np.random.default_rng(42 + epoch).permutation(2)) if shuffle else [0, 1]
    consumed = [epoch * 2 + epoch_order(epoch).index(0) for epoch in range(consumed_epochs)]
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, consumed, optimizer_step=consumed_epochs, input_path=source,
                           producer_shuffle={"shuffle": shuffle, "shuffle_seed": 42,
                                             "deterministic_publication": reference})
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        shuffle=shuffle, shuffle_seed=42, resume_checkpoint_path=str(checkpoint), max_samples=1,
        deterministic_publication=reference)
    if reference:
        import verl_speco.standalone_tq_producer as producer
        from verl_speco.producer.vllm_feature_client import VllmResponse
        monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
        config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
        class Pool(_Pool):
            async def prefill_reference(self, request):
                path = self.root / f"reference-{request.sequence_no}"
                path.write_bytes(b"reference")
                return VllmResponse(str(path), "http://vllm:8000/v1")
        samples = []
        async def publish(entries):
            samples.extend(entry["tag"] for entry in entries)
        operation = run_producer(config, tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
                                 on_references=publish, reference_batch_size=1)
    else:
        transport = _Transport()
        operation = run_producer(config, transport=transport, tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path))
    result = asyncio.run(operation)
    if not reference:
        samples = [tag for tag in transport.records.values() if tag.get("record_type") == "sample"]
    next_order = epoch_order(consumed_epochs)
    assert result.published_count == 1
    assert [(tag["sample_id"], tag["sequence_no"]) for tag in samples] == [
        ("valid", consumed_epochs * 2 + next_order.index(0))]


def test_run_producer_shuffle_resume_rejects_changed_shuffle_config(
    tmp_path: Path,
) -> None:
    """Consumed sequence numbers are permutation positions: refuse a resume
    whose checkpoint was saved under different shuffle settings."""
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [0],
        optimizer_step=1,
        input_path=input_path,
        producer_shuffle={"shuffle": True, "shuffle_seed": 42},
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 7

    with pytest.raises(ValueError, match="shuffle config changed"):
        asyncio.run(
            run_producer(
                config,
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_Pool(tmp_path),
            )
        )


@pytest.mark.parametrize("field,value", [
    ("max_sequence_length", 1024), ("max_feature_length", 128),
    ("generation_max_tokens", 64), ("min_supervised_tokens", 9),
    ("on_missing_response", "generate"), ("on_truncated_supervision", "drop"),
    ("parser_strict_roles", True), ("on_error", "raise"),
    ("render_boundary", {"enabled": True}),
])
def test_resume_rejects_changed_producer_data_recipe(tmp_path, field, value):
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    options = config["speco"]["standalone_tq_producer"]
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(checkpoint_path, [0], optimizer_step=1, input_path=input_path,
        producer_shuffle={"shuffle": False, "shuffle_seed": 42,
                          "data_recipe": producer_data_recipe(options)})
    options["resume_checkpoint_path"] = str(checkpoint_path)
    options[field] = value
    with pytest.raises(ValueError, match="Producer data recipe changed"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))


def test_run_producer_shuffle_rejects_resume_from_legacy_progress(
    tmp_path: Path,
) -> None:
    """Legacy checkpoints (no shuffle metadata) with consumed progress cannot
    be resumed shuffled.

    Regression for the two-record case [A, B]: a legacy sequential checkpoint
    with consumed={0} means A was consumed. Resuming with shuffle=true
    replays a different permutation where sequence 0 now lands on B, silently
    dropping B and republishing A.
    """
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
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 42

    with pytest.raises(ValueError, match="predates shuffle metadata"):
        asyncio.run(
            run_producer(
                config,
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_Pool(tmp_path),
            )
        )


def test_run_producer_shuffle_allows_resume_from_empty_legacy_progress(
    tmp_path: Path,
) -> None:
    """A legacy checkpoint with no consumed progress is a fresh run and may
    start shuffled freely (nothing can misalign)."""
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=0,
        input_path=input_path,
    )
    config = _config(input_path)
    producer_cfg = config["speco"]["standalone_tq_producer"]
    producer_cfg["resume_checkpoint_path"] = str(checkpoint_path)
    producer_cfg["shuffle"] = True
    producer_cfg["shuffle_seed"] = 42
    transport = _Transport()

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=_Pool(tmp_path),
        )
    )
    assert stats.published_count == 2


def test_validate_producer_config_rejects_parquet_shuffle(tmp_path: Path) -> None:
    """Parquet+shuffle must fail at startup: no bounded shuffled Parquet
    reader exists, so the only implementation would materialize the whole
    dataset before the first request."""
    input_path = tmp_path / "train.parquet"
    input_path.write_bytes(b"PAR1-fixture")
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["shuffle"] = True

    with pytest.raises(ValueError, match="Parquet"):
        validate_producer_config(config)


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


@pytest.mark.parametrize("preserve", [None, False, True])
def test_run_producer_generates_response_for_verl_chat_prompt(
    tmp_path: Path, preserve
) -> None:
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
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["on_missing_response"] = "generate"
    if preserve is not None:
        config["actor_rollout_ref"]["rollout"]["drafter"]["training"][
            "preserve_prompt_hidden_states"
        ] = preserve

    stats = asyncio.run(
        run_producer(
            config,
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
    expected_ids = [10, 11, 12] if preserve is False else [9, 10, 11, 12]
    expected_mask = [0.0, 1.0, 1.0] if preserve is False else [0.0, 0.0, 1.0, 1.0]
    assert fields["sample__input_ids"].tolist() == expected_ids
    assert fields["sample__loss_mask"].tolist() == expected_mask


def test_run_producer_skips_rows_without_response_by_default(tmp_path: Path) -> None:
    """Prompt-only rows are filtered (not generated) unless opted in."""

    input_path = tmp_path / "prompt_only.jsonl"
    input_path.write_text(
        json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n",
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

    assert stats.filtered_count == 1
    assert stats.published_count == 0
    assert pool.generate_calls == 0
    assert pool.closed and transport.closed


def test_run_producer_rejects_invalid_on_missing_response(tmp_path: Path) -> None:
    """A bad on_missing_response must fail at startup, not mid-run."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["on_missing_response"] = "bogus"

    with pytest.raises(ValueError, match="on_missing_response"):
        asyncio.run(
            run_producer(
                config,
                transport=_Transport(),
                tokenizer=_Tokenizer(),
                client_pool=_Pool(tmp_path),
            )
        )


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
    config["speco"]["standalone_tq_producer"]["on_missing_response"] = "generate"
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


def test_run_producer_renders_off_the_event_loop(tmp_path: Path, monkeypatch) -> None:
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


def test_run_producer_misaligned_drop_survives_cleanup_failure(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A store deletion failure on the drop path must not abort the producer."""

    import verl_speco.standalone_tq_producer as producer_module

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _OneMisalignedPool(tmp_path)
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2

    def fail_cleanup(raw):
        raise RuntimeError("store delete failed")

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

    assert stats.dropped_count == 1
    assert stats.published_count == 2
    assert any(
        "best-effort temporary cleanup failed" in record.getMessage()
        for record in caplog.records
    )


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
        tag for tag in transport.records.values() if tag.get("record_type") == "sample"
    ]
    eos = next(tag for tag in transport.records.values() if tag.get("status") == "eos")
    assert stats.input_count == 3
    assert stats.published_count == 2
    assert stats.dropped_count == 1
    assert len(sample_tags) == 2
    assert eos["total_samples"] == 2
    assert all(not path.exists() for path in pool.paths)
    assert pool.closed and transport.closed


def test_run_producer_replaces_sample_after_terminal_request_failure(
    tmp_path: Path,
    caplog,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _OneFailingPrefillPool(tmp_path)
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
        tag for tag in transport.records.values() if tag.get("record_type") == "sample"
    ]
    eos = next(tag for tag in transport.records.values() if tag.get("status") == "eos")
    assert stats.input_count == 3
    assert stats.published_count == 2
    assert stats.request_failed_count == 1
    assert stats.failed_count == 0
    assert len(sample_tags) == 2
    assert eos["total_samples"] == 2
    assert all(not path.exists() for path in pool.paths)
    assert pool.closed and transport.closed
    assert "dropped sample after vLLM prefill failure" in caplog.text


def test_run_producer_bounds_consecutive_terminal_request_failures(
    tmp_path: Path,
) -> None:
    """A persistently failing endpoint must abort, not replace samples forever."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _AlwaysFailingPrefillPool(tmp_path)
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


def test_run_producer_transient_request_failure_resets_breaker(
    tmp_path: Path,
) -> None:
    """One terminal request failure followed by success must not trip the breaker."""

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _OneFailingPrefillPool(tmp_path)
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

    assert stats.request_failed_count == 1
    assert stats.published_count == 2
    assert pool.closed and transport.closed


def test_run_producer_filters_over_length_sample_without_aborting(
    tmp_path: Path,
    caplog,
) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    transport = _Transport()
    pool = _Pool(tmp_path)
    config = _config(input_path)
    # Any sample whose vLLM prefill would exceed this cap must be skipped
    # instead of aborting the producer.
    config["speco"]["standalone_tq_producer"]["max_sequence_length"] = 2

    stats = asyncio.run(
        run_producer(
            config,
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    assert stats.filtered_count == 2
    assert stats.published_count == 0
    assert stats.failed_count == 0
    assert pool.prefill_calls == 0
    assert pool.closed and transport.closed
    assert "exceeding max_sequence_length=2" in caplog.text


@pytest.mark.parametrize("pool_factory", [_AlwaysMisalignedPool, _NanHiddenStatePool])
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
    assert "'publish_submit'" in caplog.text


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


def test_hidden_state_store_closes_after_consumer_drain(
    tmp_path: Path, monkeypatch
) -> None:
    """The hidden-state pool must close only after the TQ consumer drains.

    Mooncake allocates objects across all of a process's registered segments
    (allocation_strategy=random), so closing the producer's hidden-state store
    before the drain can unmount a segment holding unconsumed TQ fields and
    surface as batch_get_into error -704.
    """

    monkeypatch.setenv("SPECO_TQ_PRODUCER_DRAIN_TIMEOUT_SECONDS", "30")
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    order: list[str] = []

    class _DrainingTransport(_Transport):
        def list_samples(self):
            order.append("list")
            # Simulate the consumer fetching and clearing every sample.
            self.records = {
                key: tag
                for key, tag in self.records.items()
                if tag.get("record_type") != "sample"
            }
            return dict(self.records)

    class _OrderedPool(_Pool):
        async def close(self) -> None:
            order.append("pool_close")
            await super().close()

    transport = _DrainingTransport()
    pool = _OrderedPool(tmp_path)

    stats = asyncio.run(
        run_producer(
            _config(input_path),
            transport=transport,
            tokenizer=_Tokenizer(),
            client_pool=pool,
        )
    )

    assert stats.published_count > 0
    assert "pool_close" in order
    # No TQ metadata read may happen after the hidden-state store closes.
    assert "list" not in order[order.index("pool_close") + 1 :]


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
        _drain_pending_samples(transport, "run-a", timeout=5.0, poll_interval=0.0)
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
        _drain_pending_samples(transport, "run-a", timeout=0.05, poll_interval=0.0)
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


@pytest.mark.parametrize("accept_first", [False, True])
def test_cancel_reference_producer_cleans_only_unsubmitted_results(tmp_path, monkeypatch, accept_first):
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from types import SimpleNamespace

    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    input_path.write_text(input_path.read_text(encoding="utf-8") * 10, encoding="utf-8")
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["hidden_states_store"] = {"backend": "mooncake"}
    config["speco"]["standalone_tq_producer"]["publish_queue_size"] = 1
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *a: 32)
    generated, submitted, attempted, deleted = set(), set(), set(), []
    ready = asyncio.Event()
    store = SimpleNamespace(release=deleted.append)

    class Pool(_Pool):
        async def prefill_reference(self, request):
            handle = f"hs-{request.sequence_no}"
            generated.add(handle)
            if len(generated) >= 4 + int(accept_first):
                ready.set()
            return VllmResponse(None, "http://vllm:8000/v1", handle=handle, store=store)

    async def submit(entries):
        attempted.update(item["tag"]["handle"] for item in entries)
        if accept_first and not submitted:
            submitted.update(item["tag"]["handle"] for item in entries)
            return {}
        await asyncio.Event().wait()

    async def exercise():
        task = asyncio.create_task(run_producer(
            config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
            on_references=submit, reference_batch_size=1,
        ))
        # An in-flight RPC has uncertain ownership even when its await is cancelled.
        # Only references never handed to submit are safe to release here.
        try:
            await asyncio.wait_for(ready.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    assert set(deleted) == generated - attempted
    assert attempted - submitted  # the cancelled RPC may already have delivered
    assert len(deleted) == len(set(deleted))
    assert bool(submitted) == accept_first


def test_eagle3_filters_short_reference_and_fills_batch_budget(tmp_path, monkeypatch):
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.trainer.tq_feature_store import ReadyEntry, ReferenceFeatureStore
    from safetensors.torch import save_file

    input_path = tmp_path / "input.jsonl"
    records = [{"sample_id": name, "prompt": name + ":", "response": "answer"}
               for name in ("short", "valid-a", "valid-b")]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    config = _config(input_path)
    config["speco"]["standalone_tq_producer"]["max_samples"] = 2
    config["speco"]["standalone_tq_producer"]["vllm_aux_hidden_state_layer_ids"] = [2, 8]
    drafter = config["actor_rollout_ref"]["rollout"]["drafter"]
    drafter["speculative_algorithm"] = "EAGLE3"
    drafter["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *a: 32)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            length = 2 if text.endswith(":") else (3 if text.startswith("short:") else 6)
            return {"input_ids": list(range(length))}

    requested, batches = [], []
    class Pool(_Pool):
        async def prefill_reference(self, request):
            requested.append(request.sample_id)
            raw = await super().prefill(request)
            save_file(raw.payload, raw.temporary_path)
            return VllmResponse(raw.temporary_path, raw.endpoint_url)

    async def submit(entries):
        batches.append(entries)

    stats = asyncio.run(run_producer(
        config, tokenizer=Tokenizer(), client_pool=Pool(tmp_path),
        on_references=submit, reference_batch_size=2,
    ))
    assert stats.filtered_count == 1
    assert stats.published_count == 2
    assert set(requested) == {"valid-a", "valid-b"}
    assert [len(batch) for batch in batches] == [2]
    store = ReferenceFeatureStore({"run_id": "run-a", "ray": {"address": "unused"}})
    store.connect()
    entries = [ReadyEntry(item["key"], item["tag"]) for item in batches[0]]
    assert all(len(entry.tag["request"]["feature_positions"]) == 5 for entry in entries)
    batch = store.prepare_batch(entries, pin_memory=False)
    assert batch["input_ids"].shape == (1, 6)


def test_ordered_reference_publication_preserves_input_order_with_bounded_window(tmp_path, monkeypatch):
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    source = tmp_path / "input.jsonl"
    _write_input(source)
    config = _config(source)
    options = config["speco"]["standalone_tq_producer"]
    options.update(deterministic_publication=True, max_ordered_inflight=2, max_samples=4)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    completed, published = [], []
    class Pool(_Pool):
        async def prefill_reference(self, request):
            await asyncio.sleep(0.08 if request.sequence_no % 2 == 0 else 0.001)
            completed.append(request.sequence_no)
            path = self.root / f"ordered-{request.sequence_no}.safetensors"
            path.write_bytes(b"reference")
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def submit(entries):
        published.extend(int(item["tag"]["sequence_no"]) for item in entries)
    async def run():
        return await asyncio.wait_for(run_producer(
            config, tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
            on_references=submit, reference_batch_size=2,
        ), timeout=10)
    result = asyncio.run(run())
    assert result.published_count == 4
    assert completed == [1, 0, 3, 2]
    assert published == [0, 1, 2, 3]
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0, 1], optimizer_step=1, input_path=source,
                           producer_shuffle={"shuffle": False, "shuffle_seed": 42,
                                             "deterministic_publication": True})
    options.update(resume_checkpoint_path=str(checkpoint), max_samples=2)
    completed.clear()
    published.clear()
    assert asyncio.run(run()).published_count == 2
    assert published == [2, 3]


@pytest.mark.parametrize("published_limit", [0, 2])
def test_reference_execution_slots_bound_prefill_until_completed_release(tmp_path, monkeypatch, published_limit):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "input.jsonl"
    _write_input(source)
    config = _config(source)
    options = config["speco"]["standalone_tq_producer"]
    options.update(deterministic_publication=True, max_ordered_inflight=2, max_samples=4)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 2, "max_published_references": published_limit}}},
        queue, reference_batch_size=2)
    requested = []
    class Pool(_Pool):
        async def prefill_reference(self, request):
            requested.append(request.sequence_no)
            await asyncio.sleep(0.05 if request.sequence_no % 2 == 0 else 0.001)
            path = self.root / f"slot-{request.sequence_no}.safetensors"
            path.write_bytes(b"reference")
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def run():
        task = asyncio.create_task(run_producer(config, tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
            on_references=actor._on_references, reference_batch_size=2,
            acquire_reference_slot=actor._acquire_reference_slot,
            release_unpublished_slot=actor._release_unpublished_slot,
            transfer_unpublished_slot=actor._transfer_unpublished_slot))
        first = await asyncio.wait_for(asyncio.to_thread(queue.get, True, 5), 6)
        assert [entry["tag"]["sequence_no"] for entry in first["entries"]] == [0, 1]
        await asyncio.sleep(0.05)
        assert sorted(requested) == [0, 1] and not task.done()
        await actor.register_dispatch(actor._incarnation, 0, "first", [entry["key"] for entry in first["entries"]])
        await actor.release_references(actor._incarnation, 0, "first", [entry["key"] for entry in first["entries"]])
        second = await asyncio.wait_for(asyncio.to_thread(queue.get, True, 5), 6)
        assert [entry["tag"]["sequence_no"] for entry in second["entries"]] == [2, 3]
        await actor.register_dispatch(actor._incarnation, 1, "second", [entry["key"] for entry in second["entries"]])
        await actor.release_references(actor._incarnation, 1, "second", [entry["key"] for entry in second["entries"]])
        result = await asyncio.wait_for(task, 5)
        assert result.published_count == 4
        assert not actor._execution_pending and not actor._active_references
    asyncio.run(run())


def test_reference_file_byte_reservation_covers_prefill_until_completion(tmp_path, monkeypatch):
    from queue import Queue
    from safetensors.torch import save_file
    import torch
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "input.jsonl"
    _write_input(source)
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        deterministic_publication=True, max_ordered_inflight=2, max_samples=4)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16,
        "max_reserved_file_tensor_bytes": 32}}}, queue, reference_batch_size=2)
    requested = []
    class Pool(_Pool):
        async def prefill_reference(self, request):
            requested.append(request.sequence_no)
            path = self.root / f"reserved-{request.sequence_no}.safetensors"
            save_file({"x": torch.arange(4, dtype=torch.float32)}, str(path))
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def run():
        task = asyncio.create_task(run_producer(config, tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
            on_references=actor._on_references, reference_batch_size=2,
            acquire_reference_slot=actor._acquire_reference_slot,
            release_unpublished_slot=actor._release_unpublished_slot,
            transfer_unpublished_slot=actor._transfer_unpublished_slot))
        first = await asyncio.wait_for(asyncio.to_thread(queue.get, True, 5), 6)
        assert sorted(requested) == [0, 1] and actor._published_file_bytes == 32
        assert not task.done()
        keys = [entry["key"] for entry in first["entries"]]
        await actor.register_dispatch(actor._incarnation, 0, "first", keys)
        await actor.release_references(actor._incarnation, 0, "first", keys)
        second = await asyncio.wait_for(asyncio.to_thread(queue.get, True, 5), 6)
        keys = [entry["key"] for entry in second["entries"]]
        await actor.register_dispatch(actor._incarnation, 1, "second", keys)
        await actor.release_references(actor._incarnation, 1, "second", keys)
        assert (await asyncio.wait_for(task, 5)).published_count == 4
        assert not actor._execution_pending and actor._published_file_bytes == 0
    asyncio.run(run())


def test_generation_cleanup_failure_does_not_reuse_execution_slot(tmp_path, monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "prompts.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(max_inflight_requests=1, max_samples=1,
        on_missing_response="generate", min_supervised_tokens=2)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    def fail_cleanup(_):
        raise OSError("generation cleanup failed")
    monkeypatch.setattr(producer, "delete_temporary_result", fail_cleanup)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 1}}}, Queue(), reference_batch_size=1)
    prefilled = []
    class Pool(_Pool):
        async def generate_reference(self, request):
            path = self.root / "generation"
            path.write_bytes(b"temporary")
            return VllmResponse(str(path), "http://vllm:8000/v1", generated_token_ids=(11, 12, 13, 14))
        async def prefill_reference(self, request):
            prefilled.append(request.sequence_no)
            raise AssertionError("prefill must not run")
    async def run():
        with pytest.raises(OSError, match="generation cleanup failed"):
            await asyncio.wait_for(run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
                on_references=actor._on_references, reference_batch_size=1,
                acquire_reference_slot=actor._acquire_reference_slot,
                release_unpublished_slot=actor._release_unpublished_slot,
                transfer_unpublished_slot=actor._transfer_unpublished_slot), 10)
        assert not prefilled and actor._execution_pending == {0}
    asyncio.run(run())


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("execution_slots", [False, True])
@pytest.mark.parametrize("ordered_window", [1, 2])
def test_filtered_generation_transfers_slot_without_blocking_queued_input(tmp_path, monkeypatch, cleanup_fails, ordered, execution_slots, ordered_window):
    if cleanup_fails and not execution_slots and not ordered:
        pytest.skip("Legacy unordered cleanup remains best effort")
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "prompts.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        max_inflight_requests=1, max_samples=2, on_missing_response="generate", min_supervised_tokens=2,
        deterministic_publication=ordered, max_ordered_inflight=ordered_window)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    if cleanup_fails:
        def fail_cleanup(result):
            raise OSError("injected cleanup failure")
        monkeypatch.setattr(producer, "delete_temporary_result", fail_cleanup)
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 2 if execution_slots else 0}}}, queue, reference_batch_size=2)
    transfers = []
    class Pool(_Pool):
        async def generate_reference(self, request):
            path = self.root / f"generate-{request.sequence_no}"
            path.write_bytes(b"generated")
            tokens = (11,) if request.sequence_no == 0 else (11, 12, 13, 14)
            return VllmResponse(str(path), "http://vllm:8000/v1", generated_token_ids=tokens)
        async def prefill_reference(self, request):
            path = self.root / f"prefill-{request.sequence_no}"
            path.write_bytes(b"reference")
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def transfer(old, new):
        assert not (tmp_path / f"generate-{old.sequence_no}").exists()
        transfers.append((old.sequence_no, new.sequence_no))
        await actor._transfer_unpublished_slot(old, new)
    async def run():
        slot_options = ({"acquire_reference_slot": actor._acquire_reference_slot,
                         "release_unpublished_slot": actor._release_unpublished_slot,
                         "transfer_unpublished_slot": transfer} if execution_slots else {})
        operation = run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
            on_references=actor._on_references, reference_batch_size=2, **slot_options)
        if cleanup_fails:
            with pytest.raises(OSError, match="injected cleanup failure"):
                await asyncio.wait_for(operation, 10)
            assert not transfers
            if execution_slots:
                assert 0 in actor._execution_pending
            return
        result = await asyncio.wait_for(operation, 10)
        assert result.filtered_count == 1 and result.published_count == 2
        transferred_to = 1 if ordered and ordered_window == 1 else 2
        assert transfers == ([(0, transferred_to)] if execution_slots else [])
        entries = []
        while not queue.empty():
            entries.extend(queue.get_nowait()["entries"])
        assert {entry["tag"]["sequence_no"] for entry in entries} == {1, 2}
        if ordered:
            assert [entry["tag"]["sequence_no"] for entry in entries] == [1, 2]
    asyncio.run(run())


def test_ordered_filter_keeps_order_when_later_request_finishes_first(tmp_path, monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "prompt.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        max_inflight_requests=2, max_samples=2, on_missing_response="generate", min_supervised_tokens=2,
        deterministic_publication=True, max_ordered_inflight=2)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 2}}}, queue, reference_batch_size=2)
    completed = []
    class Pool(_Pool):
        async def generate_reference(self, request):
            if request.sequence_no == 0:
                await asyncio.sleep(0.1)
            path = self.root / f"generated-{request.sequence_no}"
            path.write_bytes(b"temporary")
            completed.append(request.sequence_no)
            tokens = (11,) if request.sequence_no == 1 else (11, 12, 13, 14)
            return VllmResponse(str(path), "http://vllm:8000/v1", generated_token_ids=tokens)
        async def prefill_reference(self, request):
            path = self.root / f"prefill-{request.sequence_no}"
            path.write_bytes(b"reference")
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def run():
        result = await asyncio.wait_for(run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
            on_references=actor._on_references, reference_batch_size=2,
            acquire_reference_slot=actor._acquire_reference_slot,
            release_unpublished_slot=actor._release_unpublished_slot,
            transfer_unpublished_slot=actor._transfer_unpublished_slot), 10)
        entries = queue.get_nowait()["entries"]
        assert result.filtered_count == 1 and result.published_count == 2
        assert completed.index(1) < completed.index(0)
        assert [entry["tag"]["sequence_no"] for entry in entries] == [0, 2]
        checkpoint = tmp_path / "checkpoint"
        save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source,
            producer_shuffle={"shuffle": False, "shuffle_seed": 42, "deterministic_publication": True})
        config["speco"]["standalone_tq_producer"].update(resume_checkpoint_path=str(checkpoint), max_samples=1)
        resumed_queue = Queue()
        resumed = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_reference_execution_slots": 2}}}, resumed_queue, reference_batch_size=1)
        again = await asyncio.wait_for(run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
            on_references=resumed._on_references, reference_batch_size=1,
            acquire_reference_slot=resumed._acquire_reference_slot,
            release_unpublished_slot=resumed._release_unpublished_slot,
            transfer_unpublished_slot=resumed._transfer_unpublished_slot), 10)
        assert again.published_count == 1 and again.filtered_count == 1
        assert [entry["tag"]["sequence_no"] for entry in resumed_queue.get_nowait()["entries"]] == [2]
    asyncio.run(run())


def test_ordered_filtered_input_releases_slot_at_end_of_stream(tmp_path, monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "prompt.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        max_inflight_requests=1, max_samples=0, on_missing_response="generate", min_supervised_tokens=2,
        deterministic_publication=True, max_ordered_inflight=1)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 1}}}, Queue(), reference_batch_size=1)
    class Pool(_Pool):
        async def generate_reference(self, request):
            path = self.root / "filtered-generated"
            path.write_bytes(b"temporary")
            return VllmResponse(str(path), "http://vllm:8000/v1", generated_token_ids=(11,))
    async def run():
        result = await asyncio.wait_for(run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
            on_references=actor._on_references, reference_batch_size=1,
            acquire_reference_slot=actor._acquire_reference_slot,
            release_unpublished_slot=actor._release_unpublished_slot,
            transfer_unpublished_slot=actor._transfer_unpublished_slot), 10)
        assert result.filtered_count == 1 and result.published_count == 0
        assert not actor._execution_pending and not actor._active_references
        assert not (tmp_path / "filtered-generated").exists()
    asyncio.run(run())


def test_ordered_generated_filter_preserves_replacement_circuit_breaker(tmp_path, monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "prompt.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Q3"}]}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        max_inflight_requests=1, max_samples=2, on_missing_response="generate", min_supervised_tokens=2,
        deterministic_publication=True, max_ordered_inflight=2, max_consecutive_feature_drops=2)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 2}}}, Queue(), reference_batch_size=2)
    class Pool(_Pool):
        async def generate_reference(self, request):
            path = self.root / f"filtered-{request.sequence_no}"
            path.write_bytes(b"temporary")
            return VllmResponse(str(path), "http://vllm:8000/v1", generated_token_ids=(11,))
    async def run():
        with pytest.raises(RuntimeError, match="max_consecutive_feature_drops=2"):
            await asyncio.wait_for(run_producer(config, tokenizer=_ChatTokenizer(), client_pool=Pool(tmp_path),
                on_references=actor._on_references, reference_batch_size=2,
                acquire_reference_slot=actor._acquire_reference_slot,
                release_unpublished_slot=actor._release_unpublished_slot,
                transfer_unpublished_slot=actor._transfer_unpublished_slot), 10)
    asyncio.run(run())
    assert not list(tmp_path.glob("filtered-*"))


def test_producer_rejects_input_change_after_start_handshake(tmp_path):
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    source = tmp_path / "input.jsonl"
    _write_input(source)
    identity = input_content_identity(build_input_fingerprint(source, include_content_hash=True))
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    async def submit(entries):
        pytest.fail("changed input must not be published")
    pool = _Pool(tmp_path)
    with pytest.raises(ValueError, match="input changed before reading"):
        asyncio.run(run_producer(_config(source), tokenizer=_Tokenizer(), client_pool=pool,
                                 on_references=submit, expected_input_identity=identity))
    assert pool.closed


def test_reference_execution_failure_retains_slot_and_stops_replacement(tmp_path, monkeypatch):
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    source = tmp_path / "input.jsonl"
    _write_input(source)
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(max_inflight_requests=1, max_samples=2)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 2}}}, reference_batch_size=2)
    class Failed(_Pool):
        async def prefill_reference(self, request):
            raise TimeoutError("uncertain request")
    async def submit(entries):
        pytest.fail("failed requests must not be published")
    async def run():
        with pytest.raises(RuntimeError, match="execution slot retained"):
            await asyncio.wait_for(run_producer(config, tokenizer=_Tokenizer(), client_pool=Failed(tmp_path),
                on_references=submit, acquire_reference_slot=actor._acquire_reference_slot,
                release_unpublished_slot=actor._release_unpublished_slot,
                transfer_unpublished_slot=actor._transfer_unpublished_slot), 5)
        assert actor._execution_pending
    asyncio.run(run())


def test_ordered_reference_publication_aborts_instead_of_replacing_failed_input(tmp_path, monkeypatch):
    import verl_speco.standalone_tq_producer as producer
    source = tmp_path / "input.jsonl"
    _write_input(source)
    config = _config(source)
    options = config["speco"]["standalone_tq_producer"]
    options.update(deterministic_publication=True, max_ordered_inflight=2, max_samples=2)
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    class Pool(_Pool):
        async def prefill_reference(self, request):
            from verl_speco.producer.vllm_feature_client import VllmResponse
            if request.sequence_no == 0:
                await asyncio.sleep(0.08)
                raise RuntimeError("injected failure")
            path = self.root / "unpublished.safetensors"
            path.write_bytes(b"reference")
            return VllmResponse(str(path), "http://vllm:8000/v1")
    async def submit(entries):
        pytest.fail("failed requests must not be published")
    async def run():
        await asyncio.wait_for(run_producer(config, tokenizer=_Tokenizer(), client_pool=Pool(tmp_path),
                                            on_references=submit), timeout=10)
    with pytest.raises(RuntimeError, match="cannot replace a failed request"):
        asyncio.run(run())
    assert not (tmp_path / "unpublished.safetensors").exists()


def test_ordered_publication_survives_batch_timeout_with_out_of_order_entries():
    from types import SimpleNamespace
    from verl_speco.standalone_tq_producer import _OrderedReferenceQueue, _collect_reference_batch
    def item(sequence):
        return SimpleNamespace(request=SimpleNamespace(sequence_no=sequence))
    async def run():
        queue = asyncio.Queue()
        ordered = _OrderedReferenceQueue(queue, {0: 0, 1: 1, 2: 2})
        await queue.put(item(0))
        await queue.put(item(2))
        first = await ordered.get()
        batch, done = await _collect_reference_batch(ordered, first, 2, timeout=0.01)
        assert [x.request.sequence_no for x in batch] == [0] and not done
        await queue.put(item(1))
        assert (await ordered.get()).request.sequence_no == 1
        assert (await ordered.get()).request.sequence_no == 2
    asyncio.run(run())


@pytest.mark.parametrize("shuffle", [False, True])
def test_producer_applies_jsonl_line_limit(tmp_path, shuffle):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"prompt": "Q1: ", "response": "A1"}) + "\n", encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(shuffle=shuffle, max_jsonl_line_bytes=8)
    with pytest.raises(ValueError, match="max_jsonl_line_bytes"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))


@pytest.mark.parametrize("generate", [False, True])
def test_producer_attaches_fixed_source_routes(tmp_path, generate):
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    source = tmp_path / "input.jsonl"
    row = {"prompt": "Q1: "} if generate else {"prompt": "Q1: ", "response": "A1"}
    source.write_text((json.dumps(row) + "\n") * 3, encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        source_route_wave_samples=2, on_missing_response="generate",
        vllm_endpoints=["http://a/v1", "http://b/v1"])
    with pytest.raises(ValueError, match="requires deterministic_publication"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(),
                                 client_pool=_Pool(tmp_path), on_references=lambda entries: None))
    captured = []
    generated = []
    class Pool(_Pool):
        async def generate(self, request):
            generated.append(request)
            return await super().generate(request)
        async def prefill(self, request):
            captured.append(request)
            return await super().prefill(request)
    asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=Pool(tmp_path)))
    captured.sort(key=lambda request: request.sequence_no)
    assert [request.source_route for request in captured] == [0, 1, 0]
    assert captured[0].source_plan_sha256 == captured[1].source_plan_sha256
    assert captured[2].source_plan_sha256 != captured[0].source_plan_sha256
    if generate:
        assert {(r.sequence_no, r.source_route, r.source_plan_sha256) for r in generated} == {
            (r.sequence_no, r.source_route, r.source_plan_sha256) for r in captured}
    expected = [(r.sequence_no, r.source_route, r.source_plan_sha256) for r in captured[1:]]
    checkpoint = tmp_path / "checkpoint"
    options = config["speco"]["standalone_tq_producer"]
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source,
        producer_shuffle={"shuffle": False, "shuffle_seed": 42, "data_recipe": producer_data_recipe(options)})
    options["resume_checkpoint_path"] = str(checkpoint)
    captured.clear()
    asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=Pool(tmp_path)))
    assert sorted((r.sequence_no, r.source_route, r.source_plan_sha256) for r in captured) == expected
    options["vllm_endpoints"].reverse()
    with pytest.raises(ValueError, match="Producer data recipe changed"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=Pool(tmp_path)))


def test_producer_applies_jsonl_index_limit(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text((json.dumps({"prompt": "Q1: ", "response": "A1"}) + "\n") * 2, encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(shuffle=True, max_indexed_records=1)
    with pytest.raises(ValueError, match="max_indexed_records"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))


@pytest.mark.parametrize("shuffle", [False, True])
def test_source_proportional_resume_preserves_occurrence_identity(tmp_path, shuffle):
    from verl_speco.producer.input_reader import ShuffledJsonlReader

    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"prompt": "Q1: ", "response": "A1", "source": label}) + "\n"
                              for i, label in enumerate(["a", "a", "a", "b"])), encoding="utf-8")
    config = _config(source)
    options = config["speco"]["standalone_tq_producer"]
    options.update(source_proportional_field="source", shuffle=shuffle, shuffle_seed=42, max_samples=4)
    reader = ShuffledJsonlReader(source, source_field="source")
    order = list(reader.source_proportional_order(seed=42 if shuffle else None))
    transport = _Transport()
    asyncio.run(run_producer(config, transport=transport, tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))
    tags = sorted((tag for tag in transport.records.values() if tag.get("record_type") == "sample"),
                  key=lambda tag: int(tag["sequence_no"]))
    assert [(tag["sequence_no"], tag["sample_id"]) for tag in tags] == [
        (i, f"train-{index:06d}") for i, index in enumerate(order)]

    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0, 2], optimizer_step=1, input_path=source,
                           producer_shuffle={"shuffle": shuffle, "shuffle_seed": 42,
                                             "source_proportional_field": "source"})
    options.update(resume_checkpoint_path=str(checkpoint), max_samples=2)
    resumed = _Transport()
    asyncio.run(run_producer(config, transport=resumed, tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))
    resumed_tags = sorted((tag for tag in resumed.records.values() if tag.get("record_type") == "sample"),
                          key=lambda tag: int(tag["sequence_no"]))
    assert [(tag["sequence_no"], tag["sample_id"]) for tag in resumed_tags] == [
        (i, f"train-{order[i]:06d}") for i in (1, 3)]
    options["source_proportional_field"] = None
    with pytest.raises(ValueError, match="source-proportional ordering changed"):
        asyncio.run(run_producer(config, transport=_Transport(), tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))


def test_source_proportional_order_across_epochs(tmp_path):
    from verl_speco.producer.input_reader import ShuffledJsonlReader

    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"prompt": "Q1: ", "response": "A1", "source": label}) + "\n"
                              for label in ["a", "a", "a", "b"]), encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"].update(
        source_proportional_field="source", shuffle=True, shuffle_seed=42, max_samples=8)
    transport = _Transport()
    asyncio.run(run_producer(config, transport=transport, tokenizer=_Tokenizer(), client_pool=_Pool(tmp_path)))
    tags = sorted((tag for tag in transport.records.values() if tag.get("record_type") == "sample"),
                  key=lambda tag: int(tag["sequence_no"]))
    reader = ShuffledJsonlReader(source, source_field="source")
    order = [*reader.source_proportional_order(seed=42), *reader.source_proportional_order(seed=43)]
    assert [(tag["sequence_no"], tag["sample_id"]) for tag in tags] == [
        (position, f"train-{index:06d}") for position, index in enumerate(order)]


def test_fragment_failure_preserves_all_ambiguous_reference_owners(tmp_path, monkeypatch):
    import verl_speco.standalone_ray_runtime as runtime
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.producer.vllm_feature_client import VllmResponse

    source = tmp_path / "input.jsonl"
    rows = [{"sample_id": f"sample-{i}", "prompt": "Q1: ", "response": "A1", "filler": "x" * 600}
            for i in range(2)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = _config(source)
    config["speco"]["standalone_tq_producer"]["reference_event_max_bytes"] = 2048
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["transfer_queue"]["expected_feature"] = {"payload_kind": "vllm_reference"}
    monkeypatch.setattr(producer, "_load_target_depth", lambda *args: 32)
    monkeypatch.setattr(runtime, "retry_io", lambda operation, **kwargs: operation())
    class Pool(_Pool):
        async def prefill_reference(self, request):
            path = self.root / f"{request.sequence_no}.safetensors"
            path.write_bytes(b"data")
            self.paths.append(path)
            return VllmResponse(str(path), "http://vllm:8000/v1")
    class Queue:
        def __init__(self): self.items = []
        def put(self, event):
            self.items.append(event)
            if event["batch_id"] == 1:
                raise ConnectionError("ambiguous delivery")
    queue, pool = Queue(), Pool(tmp_path)
    actor = runtime.StandaloneProducerActor(config, queue)
    with pytest.raises(ConnectionError, match="ambiguous delivery"):
        asyncio.run(run_producer(config, tokenizer=_Tokenizer(), client_pool=pool,
                                 on_references=actor._on_references, reference_batch_size=2))
    assert [event["batch_id"] for event in queue.items] == [0, 1]
    assert len(pool.paths) == 2 and all(path.exists() for path in pool.paths)
