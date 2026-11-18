# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Lazy compatibility boundaries for reviewed Ascend model layouts.

Only a missing candidate module permits fallback. Import failures inside an
installed module propagate, so missing runtime dependencies are not hidden.
"""

from importlib import import_module

MODEL_MODULES = {
    "model": ("vllm_ascend.models.deepseek_v4.model", "vllm_ascend.models.deepseek_v4"),
    "dspark": (
        "vllm_ascend.models.deepseek_v4.dspark",
        "vllm_ascend.models.deepseek_v4_dspark",
    ),
    "mtp": ("vllm_ascend.models.deepseek_v4.mtp", "vllm_ascend.models.deepseek_v4_mtp"),
}
UNQUANTIZED_MOE_MODULES = (
    "vllm_ascend.ops.fused_moe.routed_experts",
    "vllm_ascend.ops.fused_moe.fused_moe",
)


def module_with_symbol(modules, names):
    for path in modules:
        try:
            module = import_module(path)
        except ModuleNotFoundError as error:
            if error.name and (path == error.name or path.startswith(error.name + ".")):
                continue
            raise
        for name in names:
            if hasattr(module, name):
                return module, getattr(module, name)
    raise ImportError(f"No reviewed Ascend symbol {names!r} in {modules!r}")


def model_symbol(variant, name):
    return module_with_symbol(MODEL_MODULES[variant], (name,))[1]


def decoder_layer_type():
    return module_with_symbol(
        MODEL_MODULES["model"], ("DeepseekV4DecoderLayer", "DeepseekV2DecoderLayer")
    )[1]


def unquantized_moe_type():
    return module_with_symbol(
        UNQUANTIZED_MOE_MODULES, ("AscendUnquantizedFusedMoEMethod",)
    )[1]


def split_moe_api():
    """0.27 moves selection out of quant_method.apply into RoutedExperts."""
    path = UNQUANTIZED_MOE_MODULES[0]
    try:
        module = import_module(path)
    except ModuleNotFoundError as error:
        if error.name and (path == error.name or path.startswith(error.name + ".")):
            return False
        raise
    method = getattr(module, "AscendUnquantizedFusedMoEMethod", None)
    if method is None:
        return False  # Older layouts keep this method in fused_moe.py.
    import inspect

    parameters = inspect.signature(method.apply).parameters
    if "topk_weights" in parameters and "topk_ids" in parameters:
        return True
    if "router_logits" in parameters:
        return False
    raise RuntimeError("Unsupported Ascend unquantized MoE apply signature")


def model_source(root, variant):
    """Resolve source layout without importing accelerator modules."""
    from pathlib import Path

    for module in MODEL_MODULES[variant]:
        candidate = Path(root).joinpath(*module.split(".")[1:]).with_suffix(".py")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No reviewed Ascend {variant} source under {root}")


def loaded_model_modules(modules):
    """An empty 0.27 package is safe; concrete model classes bind MoE early."""
    symbols = (
        "DeepseekV4MoE",
        "DeepseekV4DecoderLayer",
        "DeepseekV2DecoderLayer",
        "DeepSeekV4MTP",
        "DSparkDeepseekV4ForCausalLM",
    )
    return [
        name
        for paths in MODEL_MODULES.values()
        for name in paths
        if name in modules and any(hasattr(modules[name], symbol) for symbol in symbols)
    ]


def kv_plan_type():
    path = "vllm_ascend.attention.dsa_attn_kv_plan"
    try:
        module = import_module(path)
    except ModuleNotFoundError as error:
        if error.name and (path == error.name or path.startswith(error.name + ".")):
            return None
        raise
    return module.DsaAttnKvPlan


def patch_module_apply_visitor(method_type):
    """Keep torch Module.apply(fn) separate from an expert's apply(...) ABI."""
    from functools import wraps

    from torch import nn

    original = method_type.apply
    if getattr(original, "_speco_module_apply_visitor", False):
        return

    @wraps(original)
    def apply(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return nn.Module.apply(self, args[0])
        return original(self, *args, **kwargs)

    apply._speco_module_apply_visitor = True
    method_type.apply = apply


def ascend_forward_context(**kwargs):
    """Keep token IDs in legacy contexts; newer models receive them directly."""
    from inspect import signature

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    if "input_ids" not in signature(set_ascend_forward_context).parameters:
        kwargs.pop("input_ids", None)
    return set_ascend_forward_context(**kwargs)
