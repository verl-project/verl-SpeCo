# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Differentiable BF16 model boundary for Ascend's split MoE runner API.

Reuse SpeCo's consolidated expert autograd, retaining complete expert parameters
and optimizer storage. Serving forwards continue through the original runner.
"""

from functools import wraps

import torch
from torch.nn import functional as F

from verl_speco.ops.ascend_moe_combine import native_route_combine


class _NativeMoEFinalReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, runner):
        # This replay owns TP intermediate shards, so its output is unreduced.
        return runner._maybe_reduce_final_output(
            partial, partial.shape[-1], output_is_reduced=False
        )

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def training_moe_forward(model, hidden_states, input_ids=None):
    from verl_speco.backends.vllm_ascend_compat import unquantized_moe_type
    from verl_speco.ops.deepseek_v4_runtime import (
        _ConsolidatedRoutedMoE,
        _DeterministicTPRowReduce,
        _env_flag,
        _GroupedConsolidatedRoutedMoE,
        _increment_training_diagnostic,
        _routed_moe_swiglu,
        _vllm_ascend_moe_weights_module,
    )

    experts = _vllm_ascend_moe_weights_module(model.experts)
    router = getattr(experts, "router", None)
    if router is None or not isinstance(experts.quant_method, unquantized_moe_type()):
        raise RuntimeError(
            "Split Ascend MoE training requires a native router and BF16 expert weights"
        )
    # The model's EP process group can span TP/DP even with EP disabled.
    # Weight placement is defined by RoutedExperts' effective parallel config.
    layout = getattr(experts, "moe_config", None)
    if layout is None:
        raise RuntimeError(
            "Split Ascend MoE training requires an effective expert config"
        )
    unsupported = {
        "ep_size": int(layout.ep_size) != 1,
        "use_ep": bool(layout.use_ep),
        "dp_size": int(layout.dp_size) != 1,
        "tp_size_mismatch": int(layout.tp_size) != int(model.tp_size),
        "sequence_parallel": model.is_sequence_parallel or int(layout.sp_size) != 1,
        "dynamic_eplb": bool(getattr(experts, "dynamic_eplb", False)),
        "remapping": getattr(experts, "log2phy", None) is not None,
        "mix_placement": bool(getattr(experts, "mix_placement", False)),
        "lora": getattr(experts, "_ascend_moe_lora_context", None) is not None,
        "custom_routing": getattr(router, "custom_routing_function", None) is not None,
        "hash_router": getattr(router, "tid2eid", None) is not None,
        "hash_gate": getattr(model.gate, "tid2eid", None) is not None,
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise RuntimeError(
            "Split Ascend MoE training requires TP shards without EP/SP, EPLB, "
            "LoRA, hash routing or remapping; unsupported: " + ", ".join(rejected)
        )
    if bool(getattr(experts, "apply_router_weight_on_input", False)):
        raise RuntimeError("Split Ascend MoE training requires output routing weights")
    activation = getattr(experts, "activation", "silu")
    if str(getattr(activation, "value", activation)).lower() != "silu":
        raise RuntimeError("Split Ascend MoE training requires SiLU experts")
    # Native unquant_apply_mlp does not clip ordinary SILU, even if the
    # config carries swiglu_limit. RoutedExperts switches branches explicitly.
    clipped_routed = bool(getattr(experts, "swigluoai_uninterleave", False))
    if clipped_routed and (
        float(getattr(layout, "swiglu_alpha", None) or 1.0) != 1.0
        or float(getattr(layout, "swiglu_beta", None) or 0.0) != 0.0
    ):
        raise RuntimeError("Split Ascend clipped experts require alpha=1 and beta=0")
    routed_limit = (
        float(getattr(layout, "swiglu_limit", model.swiglu_limit) or 0.0)
        if clipped_routed
        else 0.0
    )
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(
            "Split Ascend MoE training requires BF16 or FP32 activations"
        )
    w13, w2 = experts.w13_weight, experts.w2_weight
    if (
        w13.ndim != 3
        or w2.ndim != 3
        or w13.shape[0] != w2.shape[0]
        or w13.shape[0] != model.gate.weight.shape[0]
        or w13.shape[1] != x.shape[-1]
        or w2.shape[2] != x.shape[-1]
        or w13.shape[2] != 2 * w2.shape[1]
        or w13.dtype != x.dtype
        or w2.dtype != x.dtype
    ):
        raise ValueError("Split Ascend expert weights must use postprocessed TP layout")
    logits = F.linear(x.float(), model.gate.weight.float())
    # Preserve the native choice of discrete expert IDs, including correction
    # bias. Only the selected unbiased weights get a differentiable replay.
    with torch.no_grad():
        native, ids = router._select_experts(
            hidden_states=x, router_logits=logits, input_ids=input_ids
        )
    router_scale = float(getattr(router, "routed_scaling_factor", 1.0))
    if router_scale <= 0:
        raise ValueError("MoE router scaling must be positive")
    output_scale = float(model.experts.routed_scaling_factor)
    scoring = router.scoring_func
    if scoring == "softmax":
        scores = logits.softmax(dim=-1)
    elif scoring == "sigmoid":
        scores = logits.sigmoid()
    elif scoring == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    else:
        raise RuntimeError(f"Unsupported split Ascend MoE scoring function {scoring!r}")
    replay = scores.gather(1, ids.long())
    if router.renormalize:
        replay = replay / replay.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    # Keep scale in the same place as the native factory: router scale BEFORE
    # the BF16 conversion, runner output scale AFTER expert combine. Moving it
    # across either operation changes rounding, even if their product is equal.
    replay = (replay * router_scale).to(native.dtype)
    weights = (native.detach() + (replay - replay.detach())).to(x.dtype)
    compute = (
        _GroupedConsolidatedRoutedMoE
        if _env_flag("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY")
        else _ConsolidatedRoutedMoE
    )
    with native_route_combine():
        output = compute.apply(
            x,
            ids.long(),
            weights,
            experts.w13_weight,
            experts.w2_weight,
            routed_limit,
        )
    if output_scale != 1.0:
        output = output * output_scale
    if model.shared_experts is not None:
        shared = model.shared_experts
        if getattr(shared.down_proj, "reduce_results", False):
            raise RuntimeError(
                "Shared experts must defer TP reduction to the model boundary"
            )
        if int(getattr(shared.gate_up_proj, "tp_size", model.tp_size)) != int(
            model.tp_size
        ):
            raise RuntimeError("Split Ascend shared expert weights must be TP sharded")
        gate_up, _ = shared.gate_up_proj(x)
        shared_out, _ = shared.down_proj(
            _routed_moe_swiglu(gate_up, float(model.swiglu_limit or 0.0))
        )
        output = output + shared_out
    # Every rank owns a shard of each expert's intermediate dimension. Sum
    # partial outputs exactly once, reusing the established row-reduce backward.
    if int(model.tp_size) > 1:
        if _env_flag("MTP_DS4_DETERMINISTIC_TP_MOE_FORWARD", False):
            from vllm.distributed import get_tp_group

            output = _DeterministicTPRowReduce.apply(output, get_tp_group())
        else:
            output = _NativeMoEFinalReduce.apply(output, model.experts)
    _increment_training_diagnostic("split_moe_model_training_calls")
    return output.reshape_as(hidden_states)


def patch_split_moe_model(moe_type):
    if getattr(moe_type, "_speco_split_moe_model_training_patch", False):
        return True
    original = moe_type.forward

    @wraps(original)
    def forward(self, hidden_states, input_ids=None):
        if not torch.is_grad_enabled():
            return original(self, hidden_states, input_ids=input_ids)
        return training_moe_forward(self, hidden_states, input_ids)

    moe_type.forward = forward
    moe_type._speco_split_moe_model_training_patch = True
    return True
