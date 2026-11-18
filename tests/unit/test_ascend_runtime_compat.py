# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reviewed runtime pairs and the named Ascend 0.26 execution-state ABI."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from verl_speco.integration.ascend_runtime_compat import (
    capture_state,
    require_ascend_runtime,
    validate_capture_runner,
)


@pytest.mark.parametrize(
    "vllm,ascend",
    [
        ("0.25.1", "0.25.1rc1"),
        ("0.26.0", "0.26.0rc1"),
        ("0.26.0+empty", "0.26.0rc1"),
        ("0.26.0+ascend", "0.26.0rc1+custom"),
        ("0.27.1", "0.27.1rc1"),
        ("0.27.1+empty", "0.27.1rc1+custom"),
    ],
)
def test_reviewed_pairs_accept_local_build_tags(vllm, ascend):
    assert require_ascend_runtime(vllm, ascend) == (
        vllm.split("+")[0],
        ascend.split("+")[0],
    )


@pytest.mark.parametrize(
    "vllm,ascend",
    [
        ("0.26.0", "0.25.1rc1"),
        ("0.25.1", "0.26.0rc1"),
        ("0.26.0", "0.26.0"),
        ("0.26.0rc1", "0.26.0rc1"),
        ("0.27.0", "0.27.0rc1"),
        ("0.27.1", "0.26.0rc1"),
        ("0.26.0", "0.27.1rc1"),
        ("0.27.1", "0.27.1"),
        ("0.27.1", "0.27.1rc2"),
        (None, "0.26.0rc1"),
    ],
)
def test_unreviewed_or_mixed_pairs_fail_closed(vllm, ascend):
    with pytest.raises(RuntimeError, match="Ascend execution requires"):
        require_ascend_runtime(vllm, ascend)


def test_training_and_rollout_use_the_same_pair_validation(monkeypatch):
    from verl_speco.backends import vllm_ascend_runtime

    versions = {"vllm": "0.26.0+empty", "vllm-ascend": "0.26.0rc1"}
    monkeypatch.setattr(
        vllm_ascend_runtime.importlib.metadata,
        "version",
        lambda name: versions.get(name, "installed"),
    )
    assert vllm_ascend_runtime.runtime_versions(require=True)["vllm"] == "0.26.0+empty"
    versions["vllm-ascend"] = "0.25.1rc1"
    with pytest.raises(RuntimeError, match="requires"):
        vllm_ascend_runtime.runtime_versions(require=True)


@pytest.mark.parametrize("version", ["0.25.1rc1", "0.26.0rc1", "0.27.1rc1"])
@pytest.mark.parametrize("tp_size", [1, 8])
def test_metadata_warmup_is_version_scoped_and_preserves_rng(
    monkeypatch, version, tp_size
):
    from verl_speco.backends import vllm_ascend_runtime as runtime

    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: version)
    calls = []
    topology = SimpleNamespace(
        tensor_parallel_size=tp_size, tensor_parallel_group=object()
    )

    def gather(output, value, *, group):
        assert group is topology.tensor_parallel_group
        assert output.numel() == tp_size and value.numel() == 1
        assert value.dtype == torch.bfloat16 and value.eq(0).all()
        output.fill_(0)
        calls.append("gather")

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda _: calls.append("sync")),
        raising=False,
    )
    rng = torch.get_rng_state().clone()
    active = tp_size > 1 and version == "0.27.1rc1"
    assert runtime.warmup_metadata_collectives(topology, "cpu") == active
    assert calls == (["gather", "sync"] if active else [])
    assert torch.equal(rng, torch.get_rng_state())


class State026(NamedTuple):
    scheduler_output: object
    logits: object
    spec_decode_metadata: object
    spec_decode_common_attn_metadata: object
    hidden_states: object
    sample_hidden_states: object
    aux_hidden_states: object
    attn_metadata: object
    positions: object
    ec_connector_output: object
    cudagraph_stats: object
    batch_desc: object


def test_026_named_state_and_noop_sampling():
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r": 2}
    )
    state = State026(
        schedule,
        None,
        None,
        None,
        None,
        None,
        None,
        {},
        torch.arange(2),
        None,
        None,
        "batch",
    )
    runner = SimpleNamespace(execute_model_state=state)
    assert capture_state(runner) is state
    runner.execute_model_state = None
    assert capture_state(runner) is None
    runner.execute_model_state = tuple(state)
    with pytest.raises(TypeError, match="ExecuteModelState"):
        capture_state(runner)


def test_v2_or_incomplete_runner_is_rejected():
    runner = SimpleNamespace(vllm_config=SimpleNamespace(use_v2_model_runner=True))
    with pytest.raises(RuntimeError, match="V1 model runner"):
        validate_capture_runner(runner)
    runner.vllm_config.use_v2_model_runner = False
    with pytest.raises(TypeError, match="V1 sampler boundary"):
        validate_capture_runner(runner)


def test_026_routed_expert_namespace_and_fp32_hc_publish_in_place():
    from verl_speco.integration.deepseek_v4_online import (
        EmbeddedDraftLoader,
        runtime_weight_name,
    )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.hc_head_fn = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    model.model.experts = torch.nn.Module()
    model.model.experts.routed_experts = torch.nn.Module()
    model.model.experts.routed_experts.w13_weight = torch.nn.Parameter(
        torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    )
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    weights = [
        (runtime_weight_name(0, 1, name), torch.ones_like(value))
        for name, value in model.named_parameters()
    ]
    session = EmbeddedDraftLoader(model)
    session.receive(weights)
    assert session.finish() == 2
    for name, value in model.named_parameters():
        assert value.data_ptr() == pointers[name]
        assert torch.equal(value, torch.ones_like(value))
    assert model.model.hc_head_fn.dtype == torch.float32
    assert model.model.experts.routed_experts.w13_weight.dtype == torch.bfloat16
