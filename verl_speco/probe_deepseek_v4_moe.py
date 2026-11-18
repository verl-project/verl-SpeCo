# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Measure local BF16 routed-MoE parity against installed Ascend kernels.

No checkpoint, target model, RL job or distributed process group is needed.
This reports exactness and errors; completing the probe is not a precision waiver.
"""

import argparse
import hashlib
import inspect
import json
import os
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def tensor_difference(actual, reference):
    import torch

    a, b = actual.detach().cpu().float(), reference.detach().cpu().float()
    if a.shape != b.shape:
        raise ValueError("Compared tensors have different shapes")
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a - b).abs()
    return {
        "exact": bool(torch.equal(a, b)),
        "finite": finite,
        "different_elements": int((a != b).sum()),
        "elements": a.numel(),
        "max_abs": float(delta.max()) if finite else None,
        "rms_abs": float(delta.square().mean().sqrt()) if finite else None,
        "relative_l2": float(
            torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-30)
        )
        if finite
        else None,
        "reference_max_abs": float(b.abs().max()) if finite else None,
    }


def combine_arithmetic_candidates(route_values, inverse_index, weights, expert_ids):
    """CPU arithmetic controls using the native token-slot -> expert-row map.

    Explicitly round products and running sums independently, so comparisons
    distinguish BF16 product rounding, BF16 accumulation and summation order.
    These emulate formulas; they do not replace the installed native operator.
    """
    import torch

    values = route_values.detach().cpu()
    probs = weights.detach().cpu()
    if values.dtype != torch.bfloat16 or probs.dtype != torch.bfloat16:
        raise ValueError("Combine arithmetic controls require BF16 values and weights")
    ids = expert_ids.detach().cpu()
    indices = inverse_index.detach().cpu().abs().long().reshape(-1)
    tokens, topk = probs.shape
    if ids.shape != probs.shape or indices.numel() != tokens * topk:
        raise ValueError("Native combine route schema differs")
    if values.shape[0] != indices.numel() or not torch.equal(
        indices.sort().values, torch.arange(indices.numel())
    ):
        raise ValueError(
            "Native combine indices must cover all route rows exactly once"
        )
    routed = values.index_select(0, indices).reshape(tokens, topk, -1)
    expert_order = ids.argsort(dim=-1, stable=True)
    slot_order = torch.arange(topk).expand(tokens, topk)
    result = {}
    for product_fp32 in (False, True):
        products = routed.float() * probs.float().unsqueeze(-1)
        if not product_fp32:
            products = products.to(values.dtype)
        for sum_fp32 in (False, True):
            for order_name, order in (("slot", slot_order), ("expert", expert_order)):
                ordered = products.gather(1, order.unsqueeze(-1).expand_as(products))
                accumulator = ordered[:, 0].float()
                if not sum_fp32:
                    accumulator = accumulator.to(values.dtype).float()
                for slot in range(1, topk):
                    accumulator = accumulator + ordered[:, slot].float()
                    if not sum_fp32:
                        accumulator = accumulator.to(values.dtype).float()
                name = f"product_{'fp32' if product_fp32 else 'bf16'}_sum_{'fp32' if sum_fp32 else 'bf16'}_{order_name}_order"
                result[name] = accumulator.to(values.dtype)
    return result


def native_components():
    import torch
    import torch_npu
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
        build_fused_experts_input,
    )
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
        build_token_dispatch_input,
    )
    from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp
    from vllm_ascend.ops.fused_moe.routed_experts import (
        AscendRoutedExperts,
        AscendUnquantizedFusedMoEMethod,
    )
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op() or not hasattr(
        torch.ops._C_ascend, "moe_gating_top_k_hash"
    ):
        raise RuntimeError(
            "Installed native sqrtsoftplus router custom op is unavailable"
        )
    return SimpleNamespace(
        torch_npu=torch_npu,
        activation=MoEActivation.SILU,
        quant_type=QuantType.NONE,
        build_input=build_fused_experts_input,
        build_dispatch=build_token_dispatch_input,
        mlp=unquant_apply_mlp,
        routed_type=AscendRoutedExperts,
        quant_method_type=AscendUnquantizedFusedMoEMethod,
        router_type=AscendFusedTopKRouter,
        dispatcher_type=TokenDispatcherWithAllGather,
    )


def make_fixture(
    native,
    *,
    stress,
    device,
    hidden_size=64,
    local_intermediate_size=64,
    expert_seed=None,
):
    import torch
    from torch import nn

    generator = torch.Generator().manual_seed(7301)
    experts, hidden, intermediate, tokens = (
        256,
        hidden_size,
        local_intermediate_size,
        16,
    )
    expert_generator = (
        None if expert_seed is None else torch.Generator().manual_seed(expert_seed)
    )

    def random(*shape, scale):
        return (torch.randn(*shape, generator=generator) * scale).to(
            device=device, dtype=torch.bfloat16
        )

    model = nn.Module()
    model.gate = nn.Module()
    model.gate.weight = nn.Parameter(random(experts, hidden, scale=0.02))
    model.gate.weight_fp32 = model.gate.weight.detach().float().clone()
    model.experts = nn.Module()
    model.experts.routed_scaling_factor = 1.0
    routed = model.experts.routed_experts = nn.Module()
    routed.moe_config = SimpleNamespace(
        ep_size=1,
        use_ep=False,
        dp_size=1,
        tp_size=1,
        sp_size=1,
        num_experts=experts,
        num_logical_experts=experts,
        swiglu_limit=10.0,
        activation=native.activation,
    )
    # Avoid config-dependent native construction: replay only checks the method
    # type. Reference computation uses the actual native MLP below.
    quant_method = native.quant_method_type.__new__(native.quant_method_type)
    if isinstance(quant_method, nn.Module):
        # The native method can itself be a Module. Skipping its config-heavy
        # constructor must not leave an uninitialized child in the module tree.
        nn.Module.__init__(quant_method)
    routed.quant_method = quant_method
    routed.activation = native.activation
    routed.router = native.router_type(
        top_k=6,
        global_num_experts=experts,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=torch.linspace(-0.2, 0.2, experts, device=device),
    )
    routed.w13_weight = nn.Parameter(
        (
            torch.randn(
                experts,
                hidden,
                intermediate * 2,
                generator=expert_generator or generator,
            )
            * (0.25 if expert_seed is None else hidden**-0.5)
        ).to(device=device, dtype=torch.bfloat16)
    )
    routed.w2_weight = nn.Parameter(
        (
            torch.randn(
                experts, intermediate, hidden, generator=expert_generator or generator
            )
            * (0.05 if expert_seed is None else intermediate**-0.5)
        ).to(device=device, dtype=torch.bfloat16)
    )
    routed.log2phy = None
    routed.n_shared_experts = 0
    routed.global_redundant_expert_num = 0
    model.tp_size, model.is_sequence_parallel = 1, False
    model.swiglu_limit, model.routed_scaling_factor = 10.0, 1.5
    model.shared_experts = None
    x = random(tokens, hidden, scale=4.0 if stress else 0.25).requires_grad_(True)
    dispatcher = native.dispatcher_type(
        top_k=6, num_experts=experts, num_local_experts=experts
    )
    return model, x, dispatcher


def measure_case(native, *, stress, grouped, device, result=None, mode=None):
    import torch
    from torch.nn import functional as F

    from verl_speco.ops import deepseek_v4_runtime as runtime
    from verl_speco.ops.ascend_moe_training import training_moe_forward

    result = {} if result is None else result
    mode = mode or ("grouped" if grouped else "expert")
    if mode not in {"expert", "grouped", "streaming", "budgeted"}:
        raise ValueError("Unknown MoE replay mode")
    result.update(
        {"stress": stress, "grouped": grouped, "replay_mode": mode, "phase": "started"}
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_REPLAY"] = str(int(grouped))
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING"] = str(
        int(mode in {"streaming", "budgeted"})
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP"] = (
        "4" if mode == "budgeted" else "1"
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET"] = "16"
    os.environ["MTP_DS4_FUSED_ROUTED_MOE_SWIGLU"] = "1"
    model, x, dispatcher = make_fixture(native, stress=stress, device=device)
    routed = model.experts.routed_experts
    with torch.no_grad():
        logits = F.linear(x.float(), model.gate.weight_fp32)
        weights, ids = native.routed_type._select_experts(routed, x, logits, False)
        fused = native.build_input(
            hidden_states=x.detach(),
            topk_weights=weights,
            topk_ids=ids,
            w1=routed.w13_weight.detach(),
            w2=routed.w2_weight.detach(),
            quant_type=native.quant_type,
            dynamic_eplb=False,
            activation=native.activation,
        )
        dispatched = dispatcher.token_dispatch(
            native.build_dispatch(fused_experts_input=fused)
        )
        native_experts, _ = native.mlp(
            dispatched.hidden_states,
            fused.weights.w1,
            fused.weights.w2,
            dispatched.group_list,
            activation=native.activation,
            need_trans=False,
            swiglu_limit=10.0,
            group_list_type=dispatched.group_list_type,
        )
        reference = dispatcher.token_combine(
            native_experts, dispatched.combine_metadata
        )
        per_expert = torch.empty_like(native_experts)
        offset = 0
        selected_max = 0.0
        for expert, count in enumerate(dispatched.group_list.cpu().tolist()):
            if count:
                projection = (
                    dispatched.hidden_states[offset : offset + count]
                    @ routed.w13_weight[expert]
                )
                selected_max = max(selected_max, float(projection.abs().max()))
                per_expert[offset : offset + count] = (
                    runtime._routed_moe_swiglu(projection, 0.0)
                    @ routed.w2_weight[expert]
                )
            offset += count
        if offset != native_experts.shape[0]:
            raise RuntimeError("Native dispatch count does not cover all expert rows")
        reference_combine = dispatcher.token_combine(
            per_expert, dispatched.combine_metadata
        )
        candidates = combine_arithmetic_candidates(
            per_expert, dispatched.combine_metadata.expanded_row_idx, weights, ids
        )
        sorted_routes, _, _, _ = runtime._prepare_grouped_moe_routes(
            ids, routed.w13_weight.shape[0]
        )
        native_indices = dispatched.combine_metadata.expanded_row_idx.abs().long()
        # Reorder the SAME expert outputs; compare the existing helper's map
        # with native dispatch independently of expert matmul accuracy.
        helper_values = per_expert.index_select(
            0, native_indices.index_select(0, sorted_routes)
        )
        with patch.dict(os.environ, {"MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE": "1"}):
            helper_output = runtime._combine_grouped_routes_token_unpermute(
                helper_values, sorted_routes, weights
            )
        if helper_output is None:
            raise RuntimeError("Existing native combine helper did not execute")

    core_name = "_GroupedConsolidatedRoutedMoE" if grouped else "_ConsolidatedRoutedMoE"
    core = getattr(runtime, core_name)
    captured = {}

    def combine(value, expert_ids, route_weights, *args):
        captured["ids"] = expert_ids.detach().clone()
        captured["weights"] = route_weights.detach().clone()
        captured["limit"] = args[-1]
        return core.apply(value, expert_ids, route_weights, *args)

    torch.npu.synchronize()
    memory_before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    diagnostic_before = runtime.get_vllm_ascend_training_diagnostics().get(
        "native_route_combine_forward_calls", 0
    )
    with patch.object(runtime, core_name, SimpleNamespace(apply=combine)):
        actual = training_moe_forward(model, x)
    torch.npu.synchronize()
    forward_peak = torch.npu.max_memory_allocated()
    native_calls = (
        runtime.get_vllm_ascend_training_diagnostics().get(
            "native_route_combine_forward_calls", 0
        )
        - diagnostic_before
    )
    result.update(
        {
            "stress": stress,
            "grouped": grouped,
            "tp_size": 1,
            "activation": str(native.activation),
            "effective_routed_limit": captured["limit"],
            "selected_projection_max_abs": selected_max,
            "router_ids_exact": bool(torch.equal(captured["ids"], ids.long())),
            "router_weights": tensor_difference(captured["weights"], weights),
            "routed_output": tensor_difference(actual, reference),
            "expert_gemm_vs_native_gmm": tensor_difference(per_expert, native_experts),
            "training_vs_native_combine_same_expert_values": tensor_difference(
                actual, reference_combine
            ),
            "combine_arithmetic_vs_native": {
                name: tensor_difference(value, reference_combine)
                for name, value in candidates.items()
            },
            "training_vs_bf16_expert_order_formula": tensor_difference(
                actual, candidates["product_bf16_sum_bf16_expert_order"]
            ),
            "existing_native_helper_vs_native_combine": tensor_difference(
                helper_output, reference_combine
            ),
            "phase": "forward_measured",
            "native_combine_calls": native_calls,
            "forward_peak_extra_device_bytes": max(0, forward_peak - memory_before),
            "theoretical_extra_route_buffer_bytes": 0
            if mode == "grouped"
            else weights.numel() * x.shape[-1] * x.element_size(),
        }
    )
    if native_calls != 1:
        raise RuntimeError("Training did not execute exactly one native route combine")
    actual.float().square().mean().backward()
    gradients = {
        "input": x.grad,
        **{name: p.grad for name, p in model.named_parameters()},
    }
    result["gradients"] = {
        name: {
            "present": value is not None,
            "finite": value is not None and bool(torch.isfinite(value).all()),
            "nonzero": value is not None and bool(value.abs().sum() > 0),
        }
        for name, value in gradients.items()
    }
    result["phase"] = "gradients_collected"
    if stress and selected_max <= 10.0:
        raise RuntimeError(
            "Stress input did not exercise values above the clip threshold"
        )
    torch.npu.synchronize()
    result["forward_backward_peak_extra_device_bytes"] = max(
        0, torch.npu.max_memory_allocated() - memory_before
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-interface-only", action="store_true")
    parser.add_argument("--require-forward-exact", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "started",
        "scope": "local_routed_moe_only_no_shared_no_tp_no_rl",
        "cases": [],
    }
    try:
        import torch

        native = native_components()
        from verl_speco.integration.vllm_ascend_implementation import (
            implementation_fingerprint,
        )

        report["implementation_fingerprint"] = implementation_fingerprint()
        report["native_source_sha256"] = {
            name: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
            for name, value in {
                "router": native.router_type,
                "dispatcher": native.dispatcher_type,
                "mlp": native.mlp,
            }.items()
        }
        report["versions"] = {
            name: version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
        }
        if args.check_interface_only:
            fixture, _, _ = make_fixture(native, stress=False, device="cpu")
            expected = {
                "gate.weight",
                "experts.routed_experts.w13_weight",
                "experts.routed_experts.w2_weight",
            }
            if (
                set(dict(fixture.named_parameters())) != expected
                or set(fixture.state_dict()) != expected
            ):
                raise RuntimeError("Probe fixture parameter registration differs")
            if torch.npu.is_initialized():
                raise RuntimeError("Interface check unexpectedly initialized NPU")
            for name in (
                "memory_allocated",
                "reset_peak_memory_stats",
                "max_memory_allocated",
            ):
                if not callable(getattr(torch.npu, name, None)):
                    raise TypeError(f"NPU memory diagnostic API is missing: {name}")
            report["status"] = "interface_ready_no_npu_execution"
        else:
            torch.npu.set_device(args.device)
            for mode in ("expert", "grouped", "streaming", "budgeted"):
                for stress in (False, True):
                    case = {}
                    report["cases"].append(case)
                    measure_case(
                        native,
                        stress=stress,
                        grouped=mode != "expert",
                        device=f"npu:{args.device}",
                        result=case,
                        mode=mode,
                    )
                    output.write_text(json.dumps(report, indent=2))
                    print(json.dumps(case), flush=True)
            report["status"] = "measurement_completed"
            report["forward_exact"] = all(
                c["routed_output"]["exact"] for c in report["cases"]
            )
            report["precision_acceptance"] = "not_assessed_no_tolerance_applied"
            if args.require_forward_exact:
                if not report["forward_exact"]:
                    raise RuntimeError(
                        "Local routed-MoE forward is not bitwise equal to native; inspect stage metrics"
                    )
                report["precision_acceptance"] = "exact_for_tested_local_cases_only"
            if not all(
                c["router_ids_exact"]
                and c["router_weights"]["exact"]
                and c["routed_output"]["finite"]
                and all(
                    v["present"] and v["finite"] and v["nonzero"]
                    for v in c["gradients"].values()
                )
                for c in report["cases"]
            ):
                raise RuntimeError(
                    "Routing/finite-gradient checks failed; inspect the report"
                )
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
