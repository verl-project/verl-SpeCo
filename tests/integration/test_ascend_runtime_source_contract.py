# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Audit installed Ascend source without importing model/driver extensions.

Set SPECO_ASCEND_RUNTIME_SOURCE_ROOT to an Ascend checkout. These checks execute
selected pure interface methods from that source, not copies of their bodies.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from verl_speco.integration.ascend_runtime_compat import CAPTURE_STATE_FIELDS


@pytest.fixture(scope="module")
def source_root():
    selected = os.environ.get("SPECO_ASCEND_RUNTIME_SOURCE_ROOT")
    if selected:
        root = Path(selected)
        return root if root.name == "vllm_ascend" else root / "vllm_ascend"
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None:
        pytest.skip("Install vllm-ascend or set SPECO_ASCEND_RUNTIME_SOURCE_ROOT")
    return Path(next(iter(spec.submodule_search_locations)))


def _class(source, name):
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _compile_class(node, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- execute selected pure methods from trusted installed source
        compile(
            ast.fix_missing_locations(module), "<installed-ascend-contract>", "exec"
        ),
        namespace,
    )
    return namespace[node.name]


def test_installed_unquant_silu_does_not_clip_merely_because_limit_is_set(source_root):
    path = source_root / "ops/fused_moe/moe_mlp.py"
    if not path.exists():
        pytest.skip("Installed runtime does not use the split MoE MLP module")
    method = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "unquant_apply_mlp"
    )

    def grouped_matmul(*, x, weight, **kwargs):
        return [x[0] @ weight[0][0]]

    activations = SimpleNamespace(
        SITU="situ",
        SWIGLUOAI="swigluoai",
        SWIGLUSTEP="swiglustep",
        GELU="gelu",
        GELU_TANH="gelu_tanh",
    )
    namespace = {
        "torch": torch,
        "MoEActivation": activations,
        "torch_npu": SimpleNamespace(
            npu_grouped_matmul=grouped_matmul,
            npu_swiglu=lambda x: (
                torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1]
            ),
        ),
    }
    exec(  # noqa: S102 -- exercise installed branch logic with CPU kernel substitutes
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        method,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    output, _ = namespace["unquant_apply_mlp"](
        torch.ones(1, 1),
        torch.tensor([[[20.0, 20.0]]]),
        torch.ones(1, 1, 1),
        torch.ones(1, dtype=torch.long),
        activation="silu",
        need_trans=False,
        swiglu_limit=10.0,
    )
    # This audits native Python branch selection, not NPU kernel accuracy.
    assert output.item() == 400.0


def test_installed_named_state_has_consumed_fields(source_root):
    source = (source_root / "worker/model_runner_v1.py").read_text()
    node = _class(source, "ExecuteModelState")
    state_type = _compile_class(node, {"NamedTuple": NamedTuple, "__name__": __name__})
    assert set(CAPTURE_STATE_FIELDS).issubset(state_type._fields)
    assert "aux_hidden_states" in state_type._fields
    values = [None] * len(state_type._fields)
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r": 2}
    )
    for name, value in {
        "positions": torch.arange(2),
        "scheduler_output": schedule,
    }.items():
        values[state_type._fields.index(name)] = value
    from verl_speco.integration.ascend_runtime_compat import capture_state

    state = state_type(*values)
    assert capture_state(SimpleNamespace(execute_model_state=state)) is state
    runner = _class(source, "NPUModelRunner")
    methods = {
        method.name: method
        for method in runner.body
        if isinstance(method, ast.FunctionDef)
    }
    assert {"sample_tokens", "_sample", "get_model"}.issubset(methods)
    sample = methods["sample_tokens"]
    clear = [
        node.lineno
        for node in ast.walk(sample)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "execute_model_state"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    calls = [
        node.lineno
        for node in ast.walk(sample)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_sample"
    ]
    assert clear and calls and max(clear) < min(calls), (
        "Sampler lifecycle changed: review capture placement"
    )


def test_installed_dspark_maps_decoder_ids_to_output_layers(source_root):
    source = (source_root / "worker/model_runner_v1.py").read_text()
    runner = _class(source, "NPUModelRunner")
    method = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_eagle3_aux_layers_from_config"
    )
    method.decorator_list = []

    class Base:
        def _get_eagle3_aux_layers_from_config(self):
            return None

    selected = ast.ClassDef(
        name="RunnerContract",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    runner_type = _compile_class(selected, {"Base": Base, "__name__": __name__})
    instance = runner_type()
    instance.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dspark_target_layer_ids=[0, 2])
        ),
    )
    assert instance._get_eagle3_aux_layers_from_config() == (1, 3)


def test_installed_target_getter_returns_pre_hc_buffer(source_root):
    from verl_speco.backends.vllm_ascend_compat import model_source

    source = model_source(source_root, "model").read_text()
    model = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(method, ast.FunctionDef)
            and method.name == "get_mtp_target_hidden_states"
            for method in node.body
        )
    )
    getter = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_mtp_target_hidden_states"
    )
    getter.decorator_list = []
    selected = ast.ClassDef(
        name="TargetContract", bases=[], keywords=[], body=[getter], decorator_list=[]
    )
    target_type = _compile_class(selected, {"__name__": __name__})
    target = target_type()
    pre_hc = torch.randn(3, 8)
    target.model = SimpleNamespace(_mtp_hidden_buffer=pre_hc)
    assert target.get_mtp_target_hidden_states() is pre_hc


def test_installed_dspark_and_decoder_symbols_match_training_loader(source_root):
    from verl_speco.backends.vllm_ascend_compat import model_source

    decoder = ast.parse(model_source(source_root, "model").read_text())
    classes = {
        node.name: node for node in decoder.body if isinstance(node, ast.ClassDef)
    }
    layer = classes.get("DeepseekV4DecoderLayer", classes.get("DeepseekV2DecoderLayer"))
    assert layer is not None
    methods = {
        node.name: node for node in layer.body if isinstance(node, ast.FunctionDef)
    }
    assert {"hc_pre", "hc_post", "forward"}.issubset(methods)
    assert [arg.arg for arg in methods["hc_pre"].args.args] == [
        "self",
        "x",
        "hc_fn",
        "hc_scale",
        "hc_base",
    ]
    assert [arg.arg for arg in methods["hc_post"].args.args] == [
        "self",
        "x",
        "residual",
        "post",
        "comb",
    ]
    draft = ast.parse(model_source(source_root, "dspark").read_text())
    classes = {node.name: node for node in draft.body if isinstance(node, ast.ClassDef)}
    assert "DSparkDeepseekV4ForCausalLM" in classes
    methods = {
        node.name
        for node in classes["DSparkDeepseekV4ForCausalLM"].body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "combine_hidden_states",
        "precompute_and_store_context_kv",
        "forward",
        "compute_logits",
        "load_weights",
    }.issubset(methods)


def test_split_moe_native_postprocess_preserves_training_matrix_layout(source_root):
    source_file = source_root / "ops/fused_moe/routed_experts.py"
    if not source_file.exists():
        pytest.skip("Legacy runtime uses the original fused_moe.py boundary")
    source = source_file.read_text()
    classes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
    }
    if "AscendUnquantizedFusedMoEMethod" not in classes:
        pytest.skip("Legacy runtime has no split expert quantization method")
    method_class = classes["AscendUnquantizedFusedMoEMethod"]
    methods = {n.name: n for n in method_class.body if isinstance(n, ast.FunctionDef)}
    args = {arg.arg for arg in methods["apply"].args.args}
    if "router_logits" in args:
        pytest.skip("Legacy runtime selects experts inside apply")
    assert {"topk_weights", "topk_ids", "shared_experts_input"}.issubset(args)

    class Parent:
        def process_weights_after_loading(self, layer):
            pass

    class Unquantized(Parent):
        def _maybe_pad_weight(self, value):
            return value

    def replace_parameter(layer, name, value):
        setattr(layer, name, torch.nn.Parameter(value))

    selected = ast.ClassDef(
        name="ExpertMethod",
        bases=[ast.Name(id="Unquantized", ctx=ast.Load())],
        keywords=[],
        body=[methods["process_weights_after_loading"]],
        decorator_list=[],
    )
    cls = _compile_class(
        selected,
        {
            "__name__": __name__,
            "Unquantized": Unquantized,
            "UnquantizedFusedMoEMethod": Unquantized,
            "replace_parameter": replace_parameter,
            "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=False),
            "maybe_trans_nz": lambda x: x,
        },
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.randn(3, 10, 4)),
        w2_weight=torch.nn.Parameter(torch.randn(3, 4, 5)),
    )
    cls().process_weights_after_loading(layer)
    assert layer.w13_weight.shape == (3, 4, 10)
    assert layer.w2_weight.shape == (3, 5, 4)


def test_kv_plan_reads_patched_training_ops_at_creation(source_root):
    source_file = source_root / "attention/dsa_attn_kv_plan.py"
    if not source_file.exists():
        pytest.skip("Legacy runtime uses DeviceOperator dispatch")
    from dataclasses import dataclass

    source = source_file.read_text()
    node = _class(source, "DsaAttnKvPlan")
    scatter_calls = []
    scatter = lambda *args: scatter_calls.append(args)
    sparse_attention = lambda *args, **kwargs: None
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "torch": SimpleNamespace(
            ops=SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_sparse_attn_sharedkv=sparse_attention,
                    npu_sparse_attn_sharedkv_metadata=lambda: None,
                    npu_scatter_nd_update_v2=scatter,
                )
            )
        ),
        "_supports_dsv4_compressed_cache": lambda: False,
        "DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET": 2,
    }
    _compile_class(node, namespace)
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_dsa_attn_kv_plan"
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(  # noqa: S102 -- selected pure methods from trusted installed source
        compile(ast.fix_missing_locations(module), "<installed-kv-plan>", "exec"),
        namespace,
    )
    plan = namespace["get_dsa_attn_kv_plan"](None)
    assert not plan.uses_sparse_flash_mla and not plan.uses_kv_compress_epilog
    assert plan.layout_kv == "PA_ND"
    assert plan.get_dsa_sparse_attn_op() is sparse_attention
    cache, values, slots = torch.zeros(2, 2), torch.ones(1, 2), torch.tensor([[0, 0]])
    plan.dsa_kv_compress_scatter(cache, values, slots)
    assert scatter_calls[0][0] is cache
    assert scatter_calls[0][1] is slots
    assert scatter_calls[0][2] is values


@pytest.mark.parametrize("include_input_ids", [True, False])
def test_installed_forward_context_accepts_training_call(
    source_root, monkeypatch, include_input_ids
):
    import copy
    import sys
    from contextlib import contextmanager

    from verl_speco.backends.vllm_ascend_compat import ascend_forward_context

    source = (source_root / "ascend_forward_context.py").read_text()
    node = copy.deepcopy(
        next(
            item
            for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef)
            and item.name == "set_ascend_forward_context"
        )
    )
    # Execute the installed signature without invoking distributed/NPU setup.
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    node.args.defaults = [ast.Constant(None) for _ in node.args.defaults]
    node.args.kw_defaults = [
        ast.Constant(None) if value is not None else None
        for value in node.args.kw_defaults
    ]
    node.body = [ast.Expr(ast.Yield(ast.Constant(None)))]
    namespace = {}
    exec(  # noqa: S102 -- bind the installed signature with a CPU-only stub body
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "<ascend-context-signature>",
            "exec",
        ),
        namespace,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(
            set_ascend_forward_context=contextmanager(namespace[node.name]),
        ),
    )
    kwargs = {
        "attn_metadata": {},
        "vllm_config": object(),
        "num_tokens": 2,
        "model_instance": object(),
        "is_draft_model": True,
    }
    if include_input_ids:
        kwargs["input_ids"] = torch.tensor([1, 2])
    with ascend_forward_context(**kwargs):
        pass
