# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise the standalone driver against the real confirmed-row collector."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl_speco.collect_deepseek_v4_features import collect_features
from verl_speco.integration.rollout_feature_capture import RolloutFeatureCollector
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract
from verl_speco.trainer.feature_store import TorchShardFeatureStore
from verl_speco.transport.rollout_feature_payload import encode_feature_payload


class Engine:
    def __init__(self, *, fail=False, extra_payload=False):
        self.model_config = SimpleNamespace(max_model_len=64)
        self.collector = RolloutFeatureCollector()
        self.running = False
        self.calls = []
        self.fail = fail
        self.extra_payload = extra_payload
        self.forwards = 0

    def has_unfinished_requests(self):
        return self.running

    def collective_rpc(self, method, *, kwargs):
        assert method == "speco_standalone_capture"
        op = kwargs["operation"]
        self.calls.append(op)
        request_id = kwargs.get("request_id")
        if op == "begin":
            self.collector.begin(request_id, kwargs["prompt_ids"])
        elif op == "finish":
            payload = encode_feature_payload(
                self.collector.finish(request_id, kwargs["final_ids"])
            )
            return [payload, payload if self.extra_payload else None]
        elif op == "cancel":
            self.collector.cancel(request_id)
        elif op == "close":
            self.collector.close()
        return [None, None]

    def add_request(self, request_id, prompt, params):
        assert self.calls[-1] == "begin"
        assert request_id in self.collector.requests
        self.request_id, self.prompt = request_id, prompt["prompt_token_ids"]
        self.running = True

    def step(self):
        self.forwards += 1
        if self.fail:
            raise RuntimeError("device failure")
        generated = [10, 11, 12, 13]
        ids = torch.tensor([*self.prompt, *generated[:-1]])
        self.collector.submit(
            self.request_id + "-internal",
            input_ids=ids,
            positions=torch.arange(len(ids)),
            hidden_states=torch.ones(len(ids), 8),
            target_head_hidden_states=torch.full((len(ids), 8), 2.0),
            sampled_token_ids=torch.tensor([13]),
            num_draft_tokens=0,
        )
        self.running = False
        return [
            SimpleNamespace(
                finished=True,
                prompt_token_ids=self.prompt,
                outputs=[SimpleNamespace(token_ids=generated)],
            )
        ]

    def abort_request(self, request_ids):
        assert self.request_id in request_ids
        self.calls.append("abort")
        self.running = False


@pytest.fixture
def contract(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 5,
                "dspark_target_layer_ids": [0, 2],
                "dspark_block_size": 2,
            }
        )
    )
    return DeepSeekV4Contract.from_model(model)


def run(engine, contract, output, prompts=None):
    return collect_features(
        SimpleNamespace(llm_engine=engine),
        prompts or [{"prompt_token_ids": [1, 2, 3]}],
        contract=contract,
        output=output,
        max_tokens=4,
        sampling_params=object(),
    )


def test_actual_collector_rows_feed_existing_store_without_replay(tmp_path, contract):
    engine = Engine()
    report = run(
        engine,
        contract,
        tmp_path / "features",
        prompts=[
            {"prompt_token_ids": [1, 2, 3]},
            {"prompt_token_ids": [4, 5]},
        ],
    )
    assert engine.forwards == 2
    assert engine.calls == ["begin", "finish", "cancel"] * 2 + ["close"]
    assert not engine.collector.requests and engine.collector.pending_bytes == 0
    assert report["rl_integration_validated"] is False
    store = TorchShardFeatureStore(tmp_path / "features", read_only=True)
    try:
        samples = [store.read(key) for key in store.iter_keys()]
        assert len(samples) == 2
        for sample in samples:
            contract.validate(sample)
        assert samples[0].input_ids.tolist() == [1, 2, 3, 10, 11, 12]
        assert samples[0].loss_mask.tolist() == [0, 0, 0, 1, 1, 1]
        assert store.get_metadata()["source"] == "standalone_rollout"
    finally:
        store.close()
    assert (tmp_path / "features/capture_report.json").exists()


@pytest.mark.parametrize("extra_payload", [False, True])
def test_failure_aborts_or_rejects_duplicate_leader_payload_and_cleans_up(
    tmp_path, contract, extra_payload
):
    engine = Engine(fail=not extra_payload, extra_payload=extra_payload)
    with pytest.raises(
        (RuntimeError, ValueError), match="device failure|one TP-leader"
    ):
        run(engine, contract, tmp_path / "features")
    assert engine.calls[-1] == "close"
    assert not engine.running and not engine.collector.requests
    assert engine.collector.pending_bytes == 0
    assert not (tmp_path / "features/capture_report.json").exists()


def test_existing_output_is_not_modified(tmp_path, contract):
    output = tmp_path / "features"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    engine = Engine()
    try:
        with pytest.raises(FileExistsError):
            run(engine, contract, output)
        assert (output / "keep").read_text() == "unchanged"
        assert engine.calls == []
    finally:
        engine.collector.close()


def test_help_does_not_need_vllm_or_verl():
    result = subprocess.run(
        [sys.executable, "-m", "verl_speco.collect_deepseek_v4_features", "--help"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--tensor-parallel-size" in result.stdout


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_worker_extension_keeps_capture_on_tp_leader(monkeypatch, tp_rank):
    from verl_speco.integration import standalone_rollout_capture as extension

    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: tp_rank
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    captures = []

    def factory(runner, settings):
        capture = SimpleNamespace(collector=RolloutFeatureCollector())
        capture.close = capture.collector.close
        captures.append(capture)
        return capture

    monkeypatch.setattr(extension, "AscendRolloutCapture", factory)
    worker = SimpleNamespace(model_runner=object())
    invoke = extension.StandaloneCaptureWorkerExtension.speco_standalone_capture
    invoke(worker, "begin", request_id="r", prompt_ids=[1, 2], settings={})
    assert len(captures) == int(tp_rank == 0)
    if tp_rank == 0:
        assert "r" in captures[0].collector.requests
    invoke(worker, "cancel", request_id="r")
    invoke(worker, "close")
    invoke(worker, "close")
    if tp_rank == 0:
        assert not captures[0].collector.requests
        assert worker._speco_standalone_capture is None


@pytest.mark.parametrize("chunk_budget", [None, 1024])
def test_shared_runtime_builder_preserves_generation_defaults_and_supports_replay_chunks(
    monkeypatch, chunk_budget
):
    from verl_speco import collect_deepseek_v4_features as capture

    runtime = ModuleType("vllm")
    runtime.LLM = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    monkeypatch.setattr(
        capture.importlib.metadata,
        "version",
        lambda name: {
            "vllm": "0.26.0+empty",
            "vllm-ascend": "0.26.0rc1",
        }[name],
    )
    settings = capture.build_capture_llm(
        "/model",
        SimpleNamespace(block_size=2),
        tensor_parallel_size=8,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=chunk_budget,
    )
    assert settings["enable_chunked_prefill"] == (chunk_budget is not None)
    assert settings["max_num_batched_tokens"] == (chunk_budget or 4096)
    assert settings["enforce_eager"] and not settings["enable_prefix_caching"]
    assert not settings["enable_expert_parallel"]
    assert settings["speculative_config"]["method"] == "dspark"
