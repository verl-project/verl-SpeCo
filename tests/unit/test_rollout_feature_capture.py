# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU request, verifier and storage contracts; no serving NPU claims."""

import json
from types import SimpleNamespace

import pytest
import torch

from verl_speco.integration.rollout_feature_capture import RolloutFeatureCollector
from verl_speco.integration.vllm_rollout_capture import (
    AscendRolloutCapture,
)
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract


def submit(
    collector, ids, positions, *, sampled=None, drafts=0, request_id="r", head=False
):
    values = torch.tensor(ids)
    hidden = torch.stack([values.float(), values.float() + 100], dim=-1)
    collector.submit(
        request_id,
        input_ids=values,
        positions=torch.tensor(positions),
        hidden_states=hidden,
        target_head_hidden_states=hidden + 200 if head else None,
        sampled_token_ids=torch.tensor(sampled or [-1]),
        num_draft_tokens=drafts,
    )
    # The runtime immediately reuses its buffers after returning from the hook.
    hidden.zero_()


def test_confirmed_rows_include_chunked_prefill_replacement_and_bonus():
    collector = RolloutFeatureCollector()
    try:
        collector.begin("r", [10, 11, 12])
        submit(collector, [10, 11], [0, 1], head=True)
        submit(collector, [12], [2], sampled=[20], head=True)
        # 21 accepted, 99 rejected and replaced by 22; its hidden state is dropped.
        submit(
            collector,
            [20, 21, 99],
            [3, 4, 5],
            sampled=[21, 22, -1],
            drafts=2,
            head=True,
        )
        submit(collector, [22, 23], [5, 6], sampled=[23, 24], drafts=1, head=True)
        result = collector.finish("r", [10, 11, 12, 20, 21, 22, 23, 24])
        assert result["input_ids"].tolist() == [10, 11, 12, 20, 21, 22, 23]
        assert result["hidden_states"][:, 0].tolist() == [10, 11, 12, 20, 21, 22, 23]
        assert result["target_head_hidden_states"][:, 0].tolist() == [
            210,
            211,
            212,
            220,
            221,
            222,
            223,
        ]
        assert collector.pending_bytes == 0 and not collector.requests
    finally:
        collector.close()


def test_stop_truncates_extra_verified_rows_without_inventing_terminal_state():
    collector = RolloutFeatureCollector()
    try:
        collector.begin("r", [10, 11])
        submit(collector, [10, 11], [0, 1])
        submit(collector, [12, 13, 14], [2, 3, 4], sampled=[13, 14, 15], drafts=2)
        result = collector.finish("r", [10, 11, 12, 13])
        assert result["input_ids"].tolist() == [10, 11, 12]
    finally:
        collector.close()


@pytest.mark.parametrize(
    "failure", ["gap", "tokens", "invalid_verifier", "capacity", "mixed_completion"]
)
def test_bad_captures_are_rejected_and_released(failure):
    collector = RolloutFeatureCollector(max_bytes=1 if failure == "capacity" else 10000)
    try:
        collector.begin("r", [10, 11])
        submit(collector, [10, 11], [0, 2] if failure == "gap" else [0, 1])
        if failure == "invalid_verifier":
            submit(collector, [12, 99], [2, 3], sampled=[55, 56], drafts=1)
        elif failure == "mixed_completion":
            submit(collector, [12], [2], request_id="r-1")
            submit(collector, [12], [2], request_id="r-2")
        with pytest.raises(ValueError):
            collector.finish("r", [10, 99 if failure == "tokens" else 11, 12])
        assert not collector.requests and collector.pending_bytes == 0
    finally:
        collector.close()


class FusedLayer(torch.nn.Module):
    def forward(self, hidden, residual):
        return hidden * 2, residual


class Norm(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 50


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FusedLayer(), FusedLayer()])
        self.model.norm = Norm()


class Runner:
    def __init__(self):
        self.model = Target()
        self.input_batch = SimpleNamespace(req_ids=["r"])
        self.input_ids = SimpleNamespace(gpu=torch.tensor([10, 11]))
        self.execute_model_state = SimpleNamespace(
            scheduler_output=SimpleNamespace(
                total_num_scheduled_tokens=2, num_scheduled_tokens={"r": 2}
            ),
            positions=torch.arange(2),
            spec_decode_metadata=None,
        )

    def get_model(self):
        return self.model

    def _sample(self):
        return SimpleNamespace(sampled_token_ids=torch.tensor([[12]]))

    def sample_tokens(self):
        # Reviewed native runner may clear its state before calling _sample.
        self.execute_model_state = None
        return self._sample()


@pytest.mark.parametrize(
    "versions", [("0.25.1", "0.25.1rc1"), ("0.26.0+empty", "0.26.0rc1")]
)
def test_runner_capture_preserves_sampling_and_uses_fused_residual(
    monkeypatch, versions
):
    monkeypatch.setattr(
        "verl_speco.integration.vllm_rollout_capture.importlib.metadata.version",
        lambda name: versions[0] if name == "vllm" else versions[1],
    )
    runner = Runner()
    settings = {
        "model_family": "qwen3",
        "target_layer_ids": [0],
        "layer_id_space": "decoder",
        "hidden_states_layout": "dflash_aux_plus_last",
    }
    capture = AscendRolloutCapture(runner, settings)
    try:
        capture.collector.begin("r", [10, 11])
        hidden, residual = runner.model.model.layers[0](
            torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])
        )
        normalized = runner.model.model.norm(hidden + residual)
        normalized.zero_()  # A serving buffer can be reused before sampling.
        output = runner.sample_tokens()
        assert output.sampled_token_ids.tolist() == [[12]]
        result = capture.collector.finish("r", [10, 11, 12])
        assert result["hidden_states"].tolist() == [[5, 55], [8, 58]]
    finally:
        capture.collector.close()


def test_unreviewed_sampler_version_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "verl_speco.integration.vllm_rollout_capture.importlib.metadata.version",
        lambda name: "0.26.0",
    )
    with pytest.raises(RuntimeError, match="requires vLLM"):
        AscendRolloutCapture(Runner(), {})


@pytest.fixture
def ds4(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 2,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 3,
                "dspark_target_layer_ids": [0, 1],
                "dspark_block_size": 2,
            }
        )
    )
    return model, DeepSeekV4Contract.from_model(model)


def test_deepseek_model_capture_keeps_native_aux_and_pre_hc_distinct():
    from verl_speco.integration.vllm_rollout_capture import TargetModelCapture

    class Backbone(torch.nn.Module):
        def __call__(self, inputs):
            # vLLM's compilation decorator bypasses nn.Module._call_impl.
            return self.forward(inputs)

        def forward(self, inputs):
            return inputs + 999, (inputs, inputs + 10)

    model = torch.nn.Module()
    model.model = Backbone()
    pre_hc = torch.arange(8).float().reshape(2, 4) + 100
    model.get_mtp_target_hidden_states = lambda: pre_hc
    capture = TargetModelCapture(
        SimpleNamespace(get_model=lambda: model),
        {
            "model_family": "deepseek_v4",
            "target_layer_ids": [0, 1],
            "hidden_size": 2,
            "hidden_states_layout": "deepseek_v4",
        },
    )
    try:
        inputs = torch.arange(4).float().reshape(2, 2)
        native_aux = model.model(inputs)[1]
        auxiliary, teacher = capture.tensors(2, auxiliary_states=native_aux)
        expected_head = pre_hc.clone()
        inputs.zero_()
        pre_hc.zero_()
        assert torch.equal(auxiliary, torch.tensor([[0, 1, 10, 11], [2, 3, 12, 13]]))
        assert torch.equal(teacher, expected_head)
        assert not capture.handles
        # A stale hook cache cannot substitute for this forward's native state.
        capture.values["aux"] = native_aux
        with pytest.raises(RuntimeError, match="execute state"):
            capture.tensors(2)
    finally:
        capture.close()


def test_deepseek_sampler_captures_native_state_before_buffer_reuse(monkeypatch):
    monkeypatch.setattr(
        "verl_speco.integration.vllm_rollout_capture.importlib.metadata.version",
        lambda name: "0.26.0+empty" if name == "vllm" else "0.26.0rc1",
    )

    class DSparkRunner(Runner):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.model = torch.nn.Identity()
            self.model.model.aux_hidden_state_layers = (1, 2)
            self.head = torch.arange(8).float().reshape(2, 4) + 100
            self.aux = [torch.ones(2, 2), torch.full((2, 2), 2.0)]
            self.model.get_mtp_target_hidden_states = lambda: self.head
            self.execute_model_state.aux_hidden_states = self.aux

        def _sample(self):
            self.head.zero_()
            for value in self.aux:
                value.zero_()
            return super()._sample()

    runner = DSparkRunner()
    capture = AscendRolloutCapture(
        runner,
        {
            "model_family": "deepseek_v4",
            "target_layer_ids": [0, 1],
            "hidden_size": 2,
            "hidden_states_layout": "deepseek_v4",
        },
    )
    try:
        capture.collector.begin("r", [10, 11])
        runner.sample_tokens()
        result = capture.collector.finish("r", [10, 11, 12])
        assert result["hidden_states"].tolist() == [[1, 1, 2, 2], [1, 1, 2, 2]]
        assert result["target_head_hidden_states"].tolist() == [
            [100, 101, 102, 103],
            [104, 105, 106, 107],
        ]
    finally:
        capture.close()


@pytest.mark.parametrize("kind", ["empty", "unrelated", "unscheduled"])
def test_sampler_skips_dp_idle_or_unregistered_rows_without_reading_features(
    monkeypatch, kind
):
    monkeypatch.setattr(
        "verl_speco.integration.vllm_rollout_capture.importlib.metadata.version",
        lambda name: "0.26.0+empty" if name == "vllm" else "0.26.0rc1",
    )
    runner = Runner()
    if kind == "empty":
        runner.execute_model_state.scheduler_output.total_num_scheduled_tokens = 0
        runner.execute_model_state.scheduler_output.num_scheduled_tokens = {}
    else:
        runner.input_batch.req_ids = (
            ["other"] if kind == "unrelated" else ["r", "other"]
        )
        runner.execute_model_state.scheduler_output.num_scheduled_tokens = {"other": 2}
    capture = AscendRolloutCapture(
        runner,
        {
            "model_family": "qwen3",
            "target_layer_ids": [0],
            "layer_id_space": "decoder",
            "hidden_states_layout": "dflash_aux_plus_last",
        },
    )
    try:
        capture.collector.begin("r", [10, 11])
        runner.sample_tokens()
        session = capture.collector.requests["r"]
        assert session.failure is None and not session.futures
        assert capture.collector.pending_bytes == 0
    finally:
        capture.close()


def test_deepseek_auxiliary_layer_ids_must_match_output_index_space():
    from verl_speco.integration.vllm_rollout_capture import TargetModelCapture

    model = torch.nn.Module()
    model.model = torch.nn.Identity()
    model.model.aux_hidden_state_layers = (
        0,
        1,
    )  # decoder IDs were not shifted to output IDs
    model.get_mtp_target_hidden_states = lambda: torch.ones(2, 4)
    with pytest.raises(ValueError, match="auxiliary layers"):
        TargetModelCapture(
            SimpleNamespace(get_model=lambda: model),
            {"model_family": "deepseek_v4", "target_layer_ids": [0, 1]},
        )


def test_removed_rollout_flag_is_rejected_before_model_loading():
    from verl_speco.integration.vllm_rollout_capture import configure_capture

    drafter = {
        "training": {
            "backend": "vllm_ascend",
            "collect_hidden_states_from_rollout": True,
        }
    }
    with pytest.raises(ValueError, match="removed"):
        configure_capture(None, drafter, {}, {})


def test_generate_has_no_hidden_state_rpc_or_feature_fields(monkeypatch):
    import asyncio

    from verl_speco.integration import vllm_runtime as runtime

    def unexpected():
        raise AssertionError(
            "Generation must not resolve hidden-state collection config"
        )

    monkeypatch.setattr(runtime, "_load_env_drafter_config", unexpected)
    calls = []

    async def ensure():
        pass

    async def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(token_ids=[3, 4], extra_fields={})

    server = SimpleNamespace(
        global_steps=7,
        _speco_install_request_spec_decode_capture=lambda: None,
        _speco_ensure_initial_draft_weights=ensure,
        _speco_add_vllm_spec_decode_extra_fields=lambda *a, **kw: None,
        _speco_add_vllm_request_accept_extra_fields=lambda *a, **kw: None,
    )
    output = asyncio.run(
        runtime._speco_generate_with_request_stats_unblocked(
            server, generate, ([1, 2], {}, "request"), {}
        )
    )
    assert len(calls) == 1 and output.token_ids == [3, 4]
    assert "drafter_sample" not in output.extra_fields
    assert not any(
        key.startswith("_speco_rollout_feature") for key in output.extra_fields
    )
