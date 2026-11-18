# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Legacy/new model resolution and split-MoE value/gradient contracts on CPU."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from verl_speco.backends import vllm_ascend_compat as compat
from verl_speco.ops.ascend_moe_training import (
    _NativeMoEFinalReduce,
    patch_split_moe_model,
    training_moe_forward,
)


@pytest.mark.parametrize("modern", [False, True])
def test_model_layouts_keep_legacy_symbols(monkeypatch, modern):
    decoder, draft, mtp = (
        type("Decoder", (), {}),
        type("DSpark", (), {}),
        type("MTP", (), {}),
    )
    modules = {}
    if modern:
        modules[compat.MODEL_MODULES["model"][0]] = SimpleNamespace(
            DeepseekV4DecoderLayer=decoder
        )
        modules[compat.MODEL_MODULES["dspark"][0]] = SimpleNamespace(
            DSparkDeepseekV4ForCausalLM=draft
        )
        modules[compat.MODEL_MODULES["mtp"][0]] = SimpleNamespace(DeepSeekV4MTP=mtp)
    else:
        modules[compat.MODEL_MODULES["model"][1]] = SimpleNamespace(
            DeepseekV2DecoderLayer=decoder
        )
        modules[compat.MODEL_MODULES["dspark"][1]] = SimpleNamespace(
            DSparkDeepseekV4ForCausalLM=draft
        )
        modules[compat.MODEL_MODULES["mtp"][1]] = SimpleNamespace(DeepSeekV4MTP=mtp)

    def load(name):
        if name not in modules:
            raise ModuleNotFoundError(name, name=name)
        return modules[name]

    monkeypatch.setattr(compat, "import_module", load)
    assert compat.decoder_layer_type() is decoder
    assert compat.model_symbol("dspark", "DSparkDeepseekV4ForCausalLM") is draft
    assert compat.model_symbol("mtp", "DeepSeekV4MTP") is mtp


def test_real_dependency_failure_does_not_trigger_legacy_fallback(monkeypatch):
    calls = []

    def load(name):
        calls.append(name)
        raise ModuleNotFoundError("missing accelerator dependency", name="torch_npu")

    monkeypatch.setattr(compat, "import_module", load)
    with pytest.raises(ModuleNotFoundError, match="accelerator"):
        compat.model_symbol("dspark", "DSparkDeepseekV4ForCausalLM")
    assert calls == [compat.MODEL_MODULES["dspark"][0]]


@pytest.mark.parametrize("modern", [False, True])
def test_moe_apply_apis_are_not_mixed(monkeypatch, modern):
    class Old:
        def apply(self, layer, x, use_grouped_topk, top_k, router_logits, renormalize):
            pass

    class New:
        def apply(
            self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        ):
            pass

    cls = New if modern else Old
    monkeypatch.setattr(
        compat,
        "import_module",
        lambda _: SimpleNamespace(AscendUnquantizedFusedMoEMethod=cls),
    )
    assert compat.split_moe_api() is modern
    assert compat.unquantized_moe_type() is cls


class Method:
    pass


class Projection(nn.Linear):
    tp_size = 1
    reduce_results = False

    def forward(self, x):
        return super().forward(x), None


class Router:
    def __init__(self, scoring, renormalize, scale):
        self.scoring_func, self.renormalize, self.routed_scaling_factor = (
            scoring,
            renormalize,
            scale,
        )
        self.bias = torch.tensor([0.4, -0.2, 0.1])

    def scores(self, logits):
        if self.scoring_func == "softmax":
            return logits.softmax(-1)
        if self.scoring_func == "sigmoid":
            return logits.sigmoid()
        return F.softplus(logits).sqrt()

    def _select_experts(self, hidden_states, router_logits, input_ids=None):
        scores = self.scores(router_logits)
        ids = (scores + self.bias).topk(2, dim=-1).indices
        weights = scores.gather(1, ids)
        if self.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights * self.routed_scaling_factor, ids


def model(scoring, renormalize, router_scale=1.5, with_shared=True):
    m = nn.Module()
    m.gate = nn.Linear(4, 3, bias=False)
    m.experts = nn.Module()
    m.experts.routed_scaling_factor = 1.0
    e = m.experts.routed_experts = nn.Module()
    e.quant_method = Method()
    e.moe_config = SimpleNamespace(
        ep_size=1, use_ep=False, tp_size=1, dp_size=1, sp_size=1
    )
    e.router = Router(scoring, renormalize, router_scale)
    e.w13_weight = nn.Parameter(torch.randn(3, 4, 10) * 0.2)
    e.w2_weight = nn.Parameter(torch.randn(3, 5, 4) * 0.2)
    if with_shared:
        m.shared_experts = nn.Module()
        m.shared_experts.gate_up_proj = Projection(4, 6, bias=False)
        m.shared_experts.down_proj = Projection(3, 4, bias=False)
    else:
        m.shared_experts = None
    m.ep_size, m.tp_size, m.is_sequence_parallel = 1, 1, False
    m.swiglu_limit, m.routed_scaling_factor = 0.7, 1.5
    return m


def reference(m, x):
    e = m.experts.routed_experts
    logits = F.linear(x.float(), m.gate.weight.float())
    weights, ids = e.router._select_experts(x, logits)
    weights = weights.to(x.dtype)
    result = torch.zeros_like(x)

    def activation(value, limit):
        g, u = value.chunk(2, -1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        return F.silu(g) * u

    for token in range(x.shape[0]):
        for slot in range(2):
            expert = ids[token, slot].item()
            limit = (
                m.swiglu_limit if getattr(e, "swigluoai_uninterleave", False) else 0.0
            )
            value = (
                activation(x[token] @ e.w13_weight[expert], limit) @ e.w2_weight[expert]
            )
            result[token] = result[token] + weights[token, slot] * value
    if m.experts.routed_scaling_factor != 1.0:
        result = result * m.experts.routed_scaling_factor
    if m.shared_experts is not None:
        shared = m.shared_experts
        projected = F.linear(x, shared.gate_up_proj.weight)
        result = result + F.linear(
            activation(projected, m.swiglu_limit), shared.down_proj.weight
        )
    return result


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("router_scale", [1.0, 1.5])
def test_split_moe_matches_independent_reference_values_and_all_gradients(
    monkeypatch, scoring, renormalize, router_scale
):
    torch.manual_seed(41)
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY", "0")
    actual = model(scoring, renormalize, router_scale)
    expected = copy.deepcopy(actual)
    x = torch.randn(5, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    before = {name: p.data_ptr() for name, p in actual.named_parameters()}
    output = training_moe_forward(actual, x)
    ref_output = reference(expected, ref_x)
    torch.testing.assert_close(output, ref_output, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    ref_output.backward(upstream)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
    for name, param in actual.named_parameters():
        target = dict(expected.named_parameters())[name]
        assert param.data_ptr() == before[name]
        assert param.grad is not None and param.grad.shape == param.shape
        torch.testing.assert_close(param.grad, target.grad, atol=3e-6, rtol=3e-5)
    assert actual.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "router_scale,output_scale", [(1.5, 1.0), (1.0, 1.5), (1.5, 1.5)]
)
def test_split_moe_keeps_native_bf16_scale_and_cast_order(
    monkeypatch, router_scale, output_scale
):
    from verl_speco.ops import deepseek_v4_runtime as runtime

    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY", "0")
    torch.manual_seed(71)
    actual = model("sqrtsoftplus", True, router_scale, with_shared=False).to(
        torch.bfloat16
    )
    actual.experts.routed_scaling_factor = output_scale
    x = torch.randn(32, 4, dtype=torch.bfloat16, requires_grad=True)
    logits = F.linear(x.float(), actual.gate.weight.float())
    native, _ = actual.experts.routed_experts.router._select_experts(x, logits)
    observed = []

    class Combine:
        @staticmethod
        def apply(x, ids, weights, *args):
            observed.append(weights.detach().clone())
            return x * weights.sum(-1, keepdim=True)

    monkeypatch.setattr(runtime, "_ConsolidatedRoutedMoE", Combine)
    output = training_moe_forward(actual, x)
    expected_weights = native.to(x.dtype)
    assert torch.equal(observed[0], expected_weights)
    expected = x * expected_weights.sum(-1, keepdim=True)
    if output_scale != 1.0:
        expected = expected * output_scale
    assert torch.equal(output, expected)
    if router_scale != 1.0:
        assert not torch.equal(expected_weights, (native / router_scale).to(x.dtype))
    output.float().sum().backward()
    assert actual.gate.weight.grad is not None
    assert torch.isfinite(actual.gate.weight.grad).all()
    assert actual.gate.weight.grad.abs().sum() > 0


def test_native_moe_reduce_keeps_native_forward_and_identity_backward():
    x = torch.randn(4, 8, requires_grad=True)
    native = torch.randn_like(x)

    class Runner:
        def _maybe_reduce_final_output(self, partial, trunc_size, *, output_is_reduced):
            assert not torch.is_grad_enabled()
            assert partial is x and trunc_size == 8 and output_is_reduced is False
            return native

    output = _NativeMoEFinalReduce.apply(x, Runner())
    assert torch.equal(output, native)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    assert torch.equal(x.grad, upstream)


def test_split_moe_defaults_to_native_final_reduce(monkeypatch):
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    monkeypatch.delenv("MTP_DS4_DETERMINISTIC_TP_MOE_FORWARD", raising=False)
    actual = model("sqrtsoftplus", True, with_shared=False)
    actual.tp_size = actual.experts.routed_experts.moe_config.tp_size = 2
    calls = []

    def reduce(partial, trunc_size, *, output_is_reduced):
        calls.append((trunc_size, output_is_reduced))
        return partial * 2

    actual.experts._maybe_reduce_final_output = reduce
    training_moe_forward(actual, torch.randn(5, 4, requires_grad=True)).sum().backward()
    assert calls == [(4, False)]


@pytest.mark.parametrize("clipped", [False, True])
def test_split_moe_routed_activation_matches_native_branch_above_clip_threshold(
    monkeypatch, clipped
):
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY", "0")
    actual = model("sqrtsoftplus", True, with_shared=False)
    actual.experts.routed_experts.swigluoai_uninterleave = clipped
    with torch.no_grad():
        actual.experts.routed_experts.w13_weight.fill_(2.0)
        actual.experts.routed_experts.w2_weight.fill_(1.0)
    expected = copy.deepcopy(actual)
    x = torch.ones(5, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    output = training_moe_forward(actual, x)
    ref_output = reference(expected, ref_x)
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, param in actual.named_parameters():
        torch.testing.assert_close(
            param.grad, dict(expected.named_parameters())[name].grad
        )


def test_moe_probe_reports_differences_without_accepting_a_tolerance():
    from verl_speco.probe_deepseek_v4_moe import tensor_difference

    report = tensor_difference(torch.tensor([1.0, 3.0]), torch.ones(2))
    assert not report["exact"] and report["finite"]
    assert report["different_elements"] == 1
    assert report["max_abs"] == 2.0
    assert report["relative_l2"] == pytest.approx(2**0.5)
    invalid = tensor_difference(torch.tensor([float("nan")]), torch.ones(1))
    assert not invalid["finite"] and invalid["max_abs"] is None


@pytest.mark.parametrize("module_method", [False, True])
def test_moe_probe_fixture_supports_parameter_traversal_and_gradient_report(
    module_method,
):
    from verl_speco.probe_deepseek_v4_moe import make_fixture

    class QuantMethod(nn.Module if module_method else object):
        def __init__(self):
            raise AssertionError(
                "Probe must not call the native config-dependent constructor"
            )

    native = SimpleNamespace(
        activation="silu",
        quant_method_type=QuantMethod,
        router_type=lambda **kwargs: SimpleNamespace(**kwargs),
        dispatcher_type=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    fixture, x, _ = make_fixture(native, stress=False, device="cpu")
    expected = {
        "gate.weight",
        "experts.routed_experts.w13_weight",
        "experts.routed_experts.w2_weight",
    }
    params = dict(fixture.named_parameters())
    assert set(params) == set(fixture.state_dict()) == expected
    # Exercise the exact traversal that previously failed after NPU backward.
    loss = x.float().square().mean() + sum(
        p.float().square().mean() for p in params.values()
    )
    loss.backward()
    assert x.grad is not None
    for _, p in fixture.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_moe_combine_controls_distinguish_precision_and_order_with_native_row_map():
    from verl_speco.probe_deepseek_v4_moe import combine_arithmetic_candidates

    # Token-slot rows [256, 1, -256] are stored in a different expert-row order.
    rows = torch.tensor([[-256.0], [256.0], [1.0]], dtype=torch.bfloat16)
    inverse = torch.tensor([1, 2, 0], dtype=torch.int32)
    weights = torch.ones(1, 3, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[2, 0, 1]])
    controls = combine_arithmetic_candidates(rows, inverse, weights, expert_ids)
    assert controls["product_fp32_sum_fp32_slot_order"].item() == 1.0
    assert controls["product_bf16_sum_bf16_slot_order"].item() == 0.0
    assert controls["product_bf16_sum_bf16_expert_order"].item() == 1.0


def test_moe_combine_controls_distinguish_product_rounding_and_reject_bad_map():
    from verl_speco.probe_deepseek_v4_moe import combine_arithmetic_candidates

    values = torch.tensor([[1.0078125], [1.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[1.0078125, -1.015625]], dtype=torch.bfloat16)
    index = torch.tensor([0, 1], dtype=torch.int32)
    ids = torch.tensor([[0, 1]])
    controls = combine_arithmetic_candidates(values, index, weights, ids)
    assert controls["product_fp32_sum_fp32_slot_order"].item() == 2**-14
    assert controls["product_bf16_sum_fp32_slot_order"].item() == 0.0
    with pytest.raises(ValueError, match="exactly once"):
        combine_arithmetic_candidates(
            values, torch.zeros(2, dtype=torch.long), weights, ids
        )


def test_existing_native_combine_helper_builds_token_slot_inverse_map(monkeypatch):
    import sys

    from verl_speco.ops.deepseek_v4_runtime import (
        _combine_grouped_routes_token_unpermute,
    )

    class NpuValue(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    captured = {}

    def unpermute(*, permuted_tokens, sorted_indices, probs):
        captured["inverse"] = sorted_indices.clone()
        rows = permuted_tokens.as_subclass(torch.Tensor).index_select(
            0, sorted_indices.long()
        )
        rows = rows.reshape(*probs.shape, -1).float()
        output = rows[:, 0] * probs[:, 0, None].float()
        for slot in range(1, probs.shape[1]):
            output = output + rows[:, slot] * probs[:, slot, None].float()
        return output.to(permuted_tokens.dtype)

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_moe_token_unpermute=unpermute)
    )
    monkeypatch.setenv("MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE", "1")
    sorted_routes = torch.tensor([2, 0, 1])
    rows = torch.tensor([[-256.0], [256.0], [1.0]], dtype=torch.bfloat16).as_subclass(
        NpuValue
    )
    result = _combine_grouped_routes_token_unpermute(
        rows, sorted_routes, torch.ones(1, 3, dtype=torch.bfloat16)
    )
    assert torch.equal(captured["inverse"], torch.tensor([1, 2, 0], dtype=torch.int32))
    assert result.item() == 1.0


def test_tp_probe_uses_distinct_expert_shards_but_identical_router_and_inputs():
    from verl_speco.probe_deepseek_v4_moe import make_fixture
    from verl_speco.probe_deepseek_v4_moe_tp import geometry

    assert geometry(4096, 2048, 8, 2)["local_intermediate_size"] == 256
    with pytest.raises(ValueError, match="divisible"):
        geometry(64, 63, 8, 2)
    native = SimpleNamespace(
        activation="silu",
        quant_method_type=nn.Module,
        router_type=lambda **kw: SimpleNamespace(**kw),
        dispatcher_type=lambda **kw: SimpleNamespace(**kw),
    )
    a, x, _ = make_fixture(
        native,
        stress=False,
        device="cpu",
        hidden_size=8,
        local_intermediate_size=4,
        expert_seed=100,
    )
    b, y, _ = make_fixture(
        native,
        stress=False,
        device="cpu",
        hidden_size=8,
        local_intermediate_size=4,
        expert_seed=101,
    )
    assert torch.equal(x, y) and torch.equal(a.gate.weight, b.gate.weight)
    assert not torch.equal(
        a.experts.routed_experts.w13_weight, b.experts.routed_experts.w13_weight
    )
    assert a.experts.routed_experts.w13_weight.shape == (256, 8, 8)


@pytest.mark.parametrize(
    "field,value", [("ep_size", 2), ("is_sequence_parallel", True)]
)
def test_unsupported_parallel_modes_fail_closed(monkeypatch, field, value):
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    m = model("sqrtsoftplus", True)
    if field == "ep_size":
        m.experts.routed_experts.moe_config.ep_size = value
    else:
        setattr(m, field, value)
    with pytest.raises(RuntimeError, match="without EP/SP"):
        training_moe_forward(m, torch.randn(5, 4))


def test_patch_preserves_native_inference_and_is_idempotent(monkeypatch):
    class MoE(nn.Module):
        def forward(self, hidden_states, input_ids=None):
            return hidden_states + 9

    from verl_speco.ops import ascend_moe_training as api

    calls = []
    monkeypatch.setattr(
        api, "training_moe_forward", lambda m, x, ids: calls.append(ids) or x * 2
    )
    assert patch_split_moe_model(MoE)
    patched = MoE.forward
    assert patch_split_moe_model(MoE) and MoE.forward is patched
    m, x = MoE(), torch.ones(2, 4)
    with torch.no_grad():
        assert torch.equal(m(x), x + 9)
    assert not calls
    assert torch.equal(m(x), x * 2)
    assert len(calls) == 1


def test_empty_modular_package_does_not_trigger_legacy_import_order_guard():
    root = "vllm_ascend.models.deepseek_v4"
    modules = {root: SimpleNamespace()}
    assert compat.loaded_model_modules(modules) == []
    modules[root + ".model"] = SimpleNamespace(DeepseekV4MoE=type("MoE", (), {}))
    assert compat.loaded_model_modules(modules) == [root + ".model"]
    modules = {root: SimpleNamespace(DeepseekV2DecoderLayer=type("Decoder", (), {}))}
    assert compat.loaded_model_modules(modules) == [root]


def test_kv_plan_patch_routes_gradients_and_preserves_no_grad_serving():
    from verl_speco.ops.ascend_dsa_compat import patch_kv_plan

    class Plan:
        uses_sparse_flash_mla = False
        uses_kv_compress_epilog = False
        layout_kv = "PA_ND"

        def dsa_kv_compress_scatter(self, cache, x, slots):
            return "native_scatter"

        def get_dsa_sparse_attn_op(self):
            return "native_attention"

    scatter_calls = []
    scatter = lambda *args: scatter_calls.append(args)
    attention = lambda *args: None
    patch_kv_plan(Plan, scatter, attention)
    plan = Plan()
    cache, values, slots = torch.zeros(2, 2), torch.ones(1, 2), torch.tensor([0])
    plan.dsa_kv_compress_scatter(cache, values, slots)
    assert (
        scatter_calls[0][0] is cache
        and scatter_calls[0][1] is slots
        and scatter_calls[0][2] is values
    )
    assert plan.get_dsa_sparse_attn_op() is attention
    assert plan.dsa_kv_compress_scatter(cache, None, slots) is None
    with torch.no_grad():
        assert plan.dsa_kv_compress_scatter(cache, values, slots) == "native_scatter"
        assert plan.get_dsa_sparse_attn_op() == "native_attention"
    plan.uses_sparse_flash_mla = True
    with pytest.raises(RuntimeError, match="uncompressed PA_ND"):
        plan.get_dsa_sparse_attn_op()
    with pytest.raises(RuntimeError, match="uncompressed PA_ND"):
        plan.dsa_kv_compress_scatter(cache, values, slots)


def test_split_expert_apply_supports_module_traversal_and_native_forward():
    import inspect

    class QuantMethod(nn.Module):
        def apply(
            self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        ):
            return (
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

    root = nn.Module()
    root.quant_method = QuantMethod()
    root.quant_method.child = nn.Linear(2, 2)
    with pytest.raises(TypeError, match="required positional"):
        root.apply(lambda module: None)
    signature = inspect.signature(QuantMethod.apply)
    compat.patch_module_apply_visitor(QuantMethod)
    patched = QuantMethod.apply
    compat.patch_module_apply_visitor(QuantMethod)
    assert QuantMethod.apply is patched
    assert inspect.signature(QuantMethod.apply) == signature
    visited = []
    assert root.apply(visited.append) is root
    assert visited == [root.quant_method.child, root.quant_method, root]
    positional = ("layer", "x", "weights", "ids", None, None)
    assert root.quant_method.apply(*positional) == positional
    assert (
        root.quant_method.apply(
            layer="layer",
            x="x",
            topk_weights="weights",
            topk_ids="ids",
            shared_experts=None,
            shared_experts_input=None,
        )
        == positional
    )
    with pytest.raises(TypeError):
        root.quant_method.apply("layer")


@pytest.mark.parametrize("legacy", [True, False])
def test_forward_context_preserves_supported_arguments(monkeypatch, legacy):
    import sys
    from contextlib import contextmanager

    ids = torch.tensor([3, 4])
    received = {}

    @contextmanager
    def old_context(*, attn_metadata, vllm_config, input_ids=None):
        received.update(metadata=attn_metadata, config=vllm_config, ids=input_ids)
        yield

    @contextmanager
    def new_context(*, attn_metadata, vllm_config):
        received.update(metadata=attn_metadata, config=vllm_config)
        yield

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(
            set_ascend_forward_context=old_context if legacy else new_context
        ),
    )
    with compat.ascend_forward_context(
        attn_metadata="attention", vllm_config="config", input_ids=ids
    ):
        assert received["metadata"] == "attention"
        assert received["config"] == "config"
        if legacy:
            assert received["ids"] is ids
        else:
            assert "ids" not in received
    with pytest.raises(TypeError, match="unexpected keyword"):
        compat.ascend_forward_context(
            attn_metadata=None, vllm_config=None, unknown=True
        )


def test_split_moe_uses_expert_layout_instead_of_global_ep_group(monkeypatch):
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    monkeypatch.setenv("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY", "0")
    torch.manual_seed(51)
    actual = model("sqrtsoftplus", True)
    actual.ep_size = 16  # A process group exists even though expert EP is disabled.
    expected = copy.deepcopy(actual)
    x = torch.randn(5, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    output = training_moe_forward(actual, x)
    ref_output = reference(expected, ref_x)
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, param in actual.named_parameters():
        torch.testing.assert_close(
            param.grad, dict(expected.named_parameters())[name].grad
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_ep", True),
        ("dp_size", 2),
        ("sp_size", 2),
        ("tp_size", 8),
    ],
)
def test_split_moe_rejects_incompatible_effective_layout(monkeypatch, field, value):
    monkeypatch.setattr(compat, "unquantized_moe_type", lambda: Method)
    m = model("sqrtsoftplus", True)
    setattr(m.experts.routed_experts.moe_config, field, value)
    with pytest.raises(RuntimeError, match="unsupported:"):
        training_moe_forward(m, torch.randn(5, 4))


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_npu_swiglu_preserves_autograd_and_custom_forward_fast_path(
    monkeypatch, grad_enabled
):
    import sys

    from verl_speco.ops.deepseek_v4_runtime import _routed_moe_swiglu

    class NpuDispatchTensor(torch.Tensor):
        # Keep actual storage/operations on CPU while selecting the NPU branch.
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    fused_calls = []

    def fused_swiglu(value):
        fused_calls.append(True)
        gate, up = value.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_swiglu=fused_swiglu)
    )
    monkeypatch.setenv("MTP_DS4_FUSED_ROUTED_MOE_SWIGLU", "1")
    base = torch.tensor(
        [[1.2, -0.4, -1.4, 0.2], [-0.6, 0.5, 1.3, -0.3]], requires_grad=True
    )
    ref = base.detach().clone().requires_grad_(True)
    gate, up = ref.chunk(2, dim=-1)
    expected = F.silu(gate.clamp(max=0.7)) * up.clamp(-0.7, 0.7)
    with torch.set_grad_enabled(grad_enabled):
        actual = _routed_moe_swiglu(
            base.as_subclass(NpuDispatchTensor), 0.7
        ).as_subclass(torch.Tensor)
    torch.testing.assert_close(actual, expected)
    if grad_enabled:
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(base.grad, ref.grad)
        torch.testing.assert_close(base.detach(), ref.detach())
    else:
        assert not actual.requires_grad
    assert fused_calls == [True]
