# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Native merge scope, full-gradient storage and all replay execution modes."""

import copy
from contextvars import Context
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from verl_speco.ops import deepseek_v4_runtime as runtime
from verl_speco.ops.ascend_moe_combine import (
    combine_routes,
    expert_matmul,
    native_route_combine,
    native_route_combine_enabled,
)


def test_scope_is_local_nested_and_restored_on_exception():
    assert not native_route_combine_enabled()
    with pytest.raises(RuntimeError, match="test failure"), native_route_combine():
        assert native_route_combine_enabled()
        assert not Context().run(native_route_combine_enabled)
        with native_route_combine():
            assert native_route_combine_enabled()
        raise RuntimeError("test failure")
    assert not native_route_combine_enabled()


def test_native_cpu_formula_retains_product_precision_and_slot_order():
    values = torch.tensor([[1.0], [-256.0], [256.0]], dtype=torch.bfloat16)
    route_ids = torch.tensor([1, 2, 0])
    result = combine_routes(values, route_ids, torch.ones(1, 3, dtype=torch.bfloat16))
    assert result.item() == 1.0


def test_npu_path_requires_native_operator_even_when_legacy_flag_is_disabled(
    monkeypatch,
):
    class NpuValues(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    called = []

    def unavailable(values, ids, weights, *, force):
        called.append(force)

    monkeypatch.setenv("MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE", "0")
    monkeypatch.setattr(runtime, "_combine_grouped_routes_token_unpermute", unavailable)
    with pytest.raises(RuntimeError, match="requires native"):
        combine_routes(
            torch.ones(1, 2).as_subclass(NpuValues), torch.tensor([0]), torch.ones(1, 1)
        )
    assert called == [True]


def test_single_expert_gmm_honors_saved_backward_choice_outside_forward_scope(
    monkeypatch,
):
    class NpuValues(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    called = []

    def gmm(x, weights, counts):
        called.append((weights.shape, counts.tolist()))
        return x.as_subclass(torch.Tensor) @ weights[0]

    monkeypatch.setattr(runtime, "_npu_grouped_moe_matmul", gmm)
    assert not native_route_combine_enabled()
    values = torch.ones(2, 3).as_subclass(NpuValues)
    weights = torch.ones(3, 3, 4)
    result = expert_matmul(values, weights, 1, torch.tensor([2]), native=True)
    assert called == [(torch.Size([1, 3, 4]), [2])]
    assert torch.equal(result, torch.full((2, 4), 3.0))


def reference(x, ids, weights, w13, w2):
    rows = []
    for token in range(ids.shape[0]):
        parts = []
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = (x[token] @ w13[expert]).chunk(2)
            parts.append((F.silu(gate) * up) @ w2[expert])
        rows.append(torch.stack(parts))
    rows = torch.stack(rows).float()
    result = rows[:, 0] * weights[:, 0, None].float()
    for slot in range(1, ids.shape[1]):
        result = result + rows[:, slot] * weights[:, slot, None].float()
    return result.to(x.dtype)


@pytest.fixture
def cpu_gmm(monkeypatch):
    # Substitute only NPU GMM primitives; exercise the real replay, merge and
    # explicit backward against an independent CPU autograd reference.
    def grouped_matmul(x, weight, counts):
        offset, outputs = 0, []
        for expert, count in enumerate(counts.tolist()):
            outputs.append(x[offset : offset + count] @ weight[expert])
            offset += count
        return torch.cat(outputs)

    def grouped_weight_grad(x, grad, counts):
        offset, outputs = 0, []
        for count in counts.tolist():
            outputs.append(x[offset : offset + count].T @ grad[offset : offset + count])
            offset += count
        return torch.stack(outputs)

    monkeypatch.setattr(runtime, "_npu_grouped_moe_matmul", grouped_matmul)
    monkeypatch.setattr(runtime, "_npu_grouped_moe_weight_grad", grouped_weight_grad)


def configure_mode(monkeypatch, mode):
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_FULL_BACKWARD", "1")
    monkeypatch.setenv(
        "MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING",
        str(int(mode in {"streaming", "budgeted"})),
    )
    monkeypatch.setenv(
        "MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP",
        "2" if mode == "budgeted" else "1",
    )
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET", "16")


@pytest.mark.parametrize("mode", ["expert", "grouped", "streaming", "budgeted"])
@pytest.mark.usefixtures("cpu_gmm")
def test_all_replay_modes_keep_full_parameter_storage_and_match_reference_gradients(
    monkeypatch, mode
):
    configure_mode(monkeypatch, mode)
    torch.manual_seed(573)
    x = nn.Parameter(torch.randn(3, 4) * 0.2)
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
    weights = nn.Parameter(torch.rand(3, 2))
    w13 = nn.Parameter(torch.randn(3, 4, 6) * 0.2)
    w2 = nn.Parameter(torch.randn(3, 3, 4) * 0.2)
    params = [x, weights, w13, w2]
    pointers = [p.data_ptr() for p in params]
    cloned = copy.deepcopy(params)
    replay_choices = []
    original_matmul = runtime.expert_matmul

    def track(*args, **kwargs):
        replay_choices.append(kwargs.get("native"))
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(runtime, "expert_matmul", track)
    core = (
        runtime._ConsolidatedRoutedMoE
        if mode == "expert"
        else runtime._GroupedConsolidatedRoutedMoE
    )
    calls_before = runtime.get_vllm_ascend_training_diagnostics().get(
        "native_route_combine_forward_calls", 0
    )
    with native_route_combine():
        actual = core.apply(x, ids, weights, w13, w2, 0.0)
    assert (
        runtime.get_vllm_ascend_training_diagnostics()[
            "native_route_combine_forward_calls"
        ]
        == calls_before + 1
    )
    expected = reference(cloned[0], ids, *cloned[1:])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    if mode in {"expert", "streaming"}:
        assert True in replay_choices
    for index, (param, ref) in enumerate(zip(params, cloned, strict=True)):
        assert param.data_ptr() == pointers[index]
        assert param.grad.shape == param.shape
        torch.testing.assert_close(param.grad, ref.grad, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("mode", ["expert", "grouped", "streaming", "budgeted"])
@pytest.mark.usefixtures("cpu_gmm")
def test_legacy_direct_core_keeps_its_previous_bf16_merge_outside_scope(
    monkeypatch, mode
):
    configure_mode(monkeypatch, mode)
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 2]])
    weights = torch.ones(1, 3, dtype=torch.bfloat16)
    w13 = torch.ones(3, 1, 2, dtype=torch.bfloat16)
    w2 = torch.tensor([350.0, 1.3671875, -350.0], dtype=torch.bfloat16).reshape(3, 1, 1)
    core = (
        runtime._ConsolidatedRoutedMoE
        if mode == "expert"
        else runtime._GroupedConsolidatedRoutedMoE
    )
    before = core.apply(x, ids, weights, w13, w2, 0.0)
    with native_route_combine():
        native = core.apply(x, ids, weights, w13, w2, 0.0)
    after = core.apply(x, ids, weights, w13, w2, 0.0)
    assert torch.equal(before, after)
    assert before.item() == 0.0 and native.item() == 1.0
