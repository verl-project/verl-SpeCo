# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Forward-only routed + shared MoE comparison using native TP collectives.

Uses deterministic synthetic TP shards, installed native MLP/linear/activation
functions and the installed runner finalizer. No target checkpoint is loaded.
This is component validation, not a full DSpark or checkpoint-loading test.
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

from verl_speco.probe_deepseek_v4_moe import (
    make_fixture,
    native_components,
    tensor_difference,
)


def geometry(hidden, intermediate, tp, dp):
    if min(hidden, intermediate, tp, dp) < 1 or intermediate % tp:
        raise ValueError(
            "Positive dimensions and intermediate_size divisible by TP are required"
        )
    return {
        "hidden_size": hidden,
        "global_intermediate_size": intermediate,
        "local_intermediate_size": intermediate // tp,
        "tp": tp,
        "external_dp": dp,
    }


def add_shared(model, native, *, rank, tp, device, hidden, intermediate):
    import torch
    from torch import nn
    from vllm_ascend.ops.activation import AscendSiluAndMulWithClamp
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    class Projection(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight)
            self.tp_size = tp
            self.reduce_results = False

        def forward(self, x):
            # The installed native unquantized linear method uses precisely
            # this entry point. Explicit shards avoid constructing a loader.
            return AscendUnquantizedLinearMethod.apply(None, self, x), None

    class Activation(nn.Module):
        swiglu_limit = 10.0

        def forward(self, x):
            return AscendSiluAndMulWithClamp.forward_oot(self, x)

    generator = torch.Generator().manual_seed(8400 + rank)

    def random(rows, cols, scale):
        return (torch.randn(rows, cols, generator=generator) * scale).to(
            device=device, dtype=torch.bfloat16
        )

    shared = nn.Module()
    shared.gate_up_proj = Projection(random(intermediate * 2, hidden, hidden**-0.5))
    shared.down_proj = Projection(random(hidden, intermediate, intermediate**-0.5))
    shared.act_fn = Activation()
    model.shared_experts = shared
    model.tp_size = tp
    model.experts.routed_experts.moe_config.tp_size = tp
    model.experts.moe_config = SimpleNamespace(
        tp_size=tp, ep_size=1, skip_final_all_reduce=False, is_sequence_parallel=False
    )
    model.experts._maybe_reduce_final_output = MethodType(
        native.finalize, model.experts
    )


def native_reference(model, x, native, dispatcher, diagnostics=None):
    from torch.nn import functional as F

    routed = model.experts.routed_experts
    logits = F.linear(x.float(), model.gate.weight_fp32)
    weights, ids = native.routed_type._select_experts(routed, x, logits, False)
    fused = native.build_input(
        hidden_states=x,
        topk_weights=weights,
        topk_ids=ids,
        w1=routed.w13_weight,
        w2=routed.w2_weight,
        quant_type=native.quant_type,
        dynamic_eplb=False,
        activation=native.activation,
    )
    dispatched = dispatcher.token_dispatch(
        native.build_dispatch(fused_experts_input=fused)
    )
    expert_rows, _ = native.mlp(
        dispatched.hidden_states,
        fused.weights.w1,
        fused.weights.w2,
        dispatched.group_list,
        activation=native.activation,
        need_trans=False,
        swiglu_limit=10.0,
        group_list_type=dispatched.group_list_type,
    )
    if diagnostics is not None:
        from verl_speco.ops import deepseek_v4_runtime as runtime

        single_rows = expert_rows.new_empty(expert_rows.shape)
        offset = 0
        for expert, count in enumerate(dispatched.group_list.cpu().tolist()):
            if count:
                chunk = dispatched.hidden_states[offset : offset + count]
                counts = dispatched.group_list[expert : expert + 1].contiguous()
                projected = runtime._npu_grouped_moe_matmul(
                    chunk, fused.weights.w1[expert : expert + 1], counts
                )
                activated = runtime._routed_moe_swiglu(projected, 0.0)
                single_rows[offset : offset + count] = runtime._npu_grouped_moe_matmul(
                    activated, fused.weights.w2[expert : expert + 1], counts
                )
            offset += count
        diagnostics["single_expert_gmm_vs_native_gmm"] = tensor_difference(
            single_rows, expert_rows
        )
    local_routed = dispatcher.token_combine(expert_rows, dispatched.combine_metadata)
    local_shared = native.shared_forward(model.shared_experts, x)
    local_sum = local_shared + local_routed
    final = model.experts._maybe_reduce_final_output(
        local_sum.clone(), x.shape[-1], output_is_reduced=False
    )
    return local_routed, local_shared, local_sum, final


def worker(args):
    import torch
    import torch.distributed as dist
    import torch_npu
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_model_parallel, get_world_group
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.device.hardware_profile import get_current_hardware_profile
    from vllm_ascend.utils import maybe_trans_nz

    from verl_speco.backends.vllm_ascend_compat import model_symbol
    from verl_speco.integration.vllm_ascend_implementation import (
        implementation_fingerprint,
    )
    from verl_speco.ops import deepseek_v4_runtime as runtime
    from verl_speco.ops.ascend_moe_training import training_moe_forward

    sizes = geometry(
        args.hidden_size,
        args.intermediate_size,
        args.tensor_parallel_size,
        args.data_parallel_size,
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    tp_rank = rank % sizes["tp"]
    torch.npu.set_device(local_rank)
    torch.npu.config.allow_internal_format = True
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=sizes["tp"],
            data_parallel_size=1,
            disable_custom_all_reduce=True,
        )
    )
    init_ascend_config(config)
    native = native_components()
    native.finalize = MoERunner._maybe_reduce_final_output
    native.shared_forward = model_symbol("model", "DeepseekV2MLP").forward
    report = {
        "status": "started",
        "rank": rank,
        "tp_rank": tp_rank,
        "geometry": sizes,
        "scope": "synthetic_shards_native_component_forward_only",
        "cases": [],
        "implementation_fingerprint": implementation_fingerprint(),
    }
    report["native_shared_source_sha256"] = hashlib.sha256(
        Path(inspect.getfile(native.shared_forward)).read_bytes()
    ).hexdigest()
    report["probe_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["weight_layout_policy"] = str(
        get_current_hardware_profile().weight_layout_policy
    )
    path = Path(args.output) / f"rank_{rank}.json"
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=sizes["tp"] * sizes["external_dp"],
                rank=rank,
                local_rank=local_rank,
                backend="hccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=sizes["tp"], backend="hccl"
            )
            modes = (
                ("expert",)
                if args.diagnose_single_gmm
                else ("expert", "grouped", "streaming", "budgeted")
            )
            for mode in modes:
                os.environ["MTP_DS4_GROUPED_ROUTED_MOE_REPLAY"] = str(
                    int(mode != "expert")
                )
                os.environ["MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING"] = str(
                    int(mode in {"streaming", "budgeted"})
                )
                os.environ["MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP"] = (
                    "4" if mode == "budgeted" else "1"
                )
                os.environ["MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET"] = "16"
                for stress in (False, True):
                    case = {"mode": mode, "stress": stress}
                    report["cases"].append(case)
                    model, x, dispatcher = make_fixture(
                        native,
                        stress=stress,
                        device=f"npu:{local_rank}",
                        hidden_size=sizes["hidden_size"],
                        local_intermediate_size=sizes["local_intermediate_size"],
                        expert_seed=8300 + tp_rank,
                    )
                    add_shared(
                        model,
                        native,
                        rank=tp_rank,
                        tp=sizes["tp"],
                        device=f"npu:{local_rank}",
                        hidden=sizes["hidden_size"],
                        intermediate=sizes["local_intermediate_size"],
                    )
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.data = maybe_trans_nz(parameter.data)
                    case["weight_formats"] = {
                        name: int(torch_npu.get_npu_format(parameter))
                        for name, parameter in model.named_parameters()
                    }
                    with torch.no_grad():
                        routed_ref, shared_ref, local_ref, final_ref = native_reference(
                            model,
                            x,
                            native,
                            dispatcher,
                            case if args.diagnose_single_gmm else None,
                        )
                    capture = {}

                    def finalize(
                        self, partial, trunc_size, *, output_is_reduced, capture=capture
                    ):
                        capture["local_sum"] = partial.detach().clone()
                        return native.finalize(
                            self,
                            partial,
                            trunc_size,
                            output_is_reduced=output_is_reduced,
                        )

                    model.experts._maybe_reduce_final_output = MethodType(
                        finalize, model.experts
                    )
                    projected, _ = model.shared_experts.gate_up_proj(x)
                    shared_train, _ = model.shared_experts.down_proj(
                        runtime._routed_moe_swiglu(projected, 10.0)
                    )
                    actual = training_moe_forward(model, x)
                    with torch.no_grad():
                        repeated_ref = native.finalize(
                            model.experts,
                            local_ref.clone(),
                            x.shape[-1],
                            output_is_reduced=False,
                        )
                    torch.npu.synchronize()
                    case.update(
                        shared_output=tensor_difference(shared_train, shared_ref),
                        local_routed_shared_sum=tensor_difference(
                            actual if sizes["tp"] == 1 else capture["local_sum"],
                            local_ref,
                        ),
                        tp_final_output=tensor_difference(actual, final_ref),
                        native_collective_repeatability=tensor_difference(
                            repeated_ref, final_ref
                        ),
                    )
                    # Compare replicated outputs across both TP ranks and external DP replicas.
                    digest = hashlib.sha256(
                        actual.detach().cpu().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    digests = [None] * dist.get_world_size()
                    reference_digest = hashlib.sha256(
                        final_ref.detach().cpu().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    dist.all_gather_object(
                        digests,
                        (digest, reference_digest),
                        group=get_world_group().cpu_group,
                    )
                    case["all_rank_outputs_identical"] = (
                        len({d[0] for d in digests}) == 1
                    )
                    case["all_native_rank_outputs_identical"] = (
                        len({d[1] for d in digests}) == 1
                    )
                    path.write_text(json.dumps(report, indent=2))
                    # Break the bound-method self-reference before dropping a
                    # fixture with potentially GiB-scale expert parameters.
                    del model.experts._maybe_reduce_final_output
                    del (
                        model,
                        x,
                        actual,
                        projected,
                        shared_train,
                        routed_ref,
                        shared_ref,
                        local_ref,
                        final_ref,
                        repeated_ref,
                        capture,
                    )
            report["forward_exact"] = all(
                c["shared_output"]["exact"]
                and c["local_routed_shared_sum"]["exact"]
                and c["tp_final_output"]["exact"]
                and c["native_collective_repeatability"]["exact"]
                for c in report["cases"]
            )
            report["status"] = (
                "passed" if report["forward_exact"] else "differences_observed"
            )
            # All ranks finish and write their diagnostics before exactness gates the launcher.
            path.write_text(json.dumps(report, indent=2))
            dist.barrier()
        destroy_model_parallel()
        dist.destroy_process_group()
        if not report["forward_exact"]:
            raise RuntimeError(
                "Native shared/TP forward differences observed; inspect per-rank reports"
            )
    except Exception:
        report["error"] = traceback.format_exc()
        if report["status"] != "differences_observed":
            report["status"] = "failed"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--diagnose-single-gmm", action="store_true")
    args = parser.parse_args()
    sizes = geometry(
        args.hidden_size,
        args.intermediate_size,
        args.tensor_parallel_size,
        args.data_parallel_size,
    )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if (root / "launch.json").exists():
        raise FileExistsError("Use a fresh probe output directory")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={sizes['tp'] * sizes['external_dp']}",
        "-m",
        "verl_speco.probe_deepseek_v4_moe_tp",
        "--worker",
        "--output",
        str(root.resolve()),
        "--tensor-parallel-size",
        str(sizes["tp"]),
        "--data-parallel-size",
        str(sizes["external_dp"]),
        "--hidden-size",
        str(sizes["hidden_size"]),
        "--intermediate-size",
        str(sizes["global_intermediate_size"]),
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        TORCHDYNAMO_DISABLE="1",
        MTP_DS4_FUSED_ROUTED_MOE_SWIGLU="1",
        MTP_DS4_DETERMINISTIC_TP_ROW_FORWARD="0",
        MTP_DS4_DETERMINISTIC_TP_MOE_FORWARD="0",
    )
    if args.diagnose_single_gmm:
        command.append("--diagnose-single-gmm")
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
        map(str, range(sizes["tp"] * sizes["external_dp"]))
    )
    (root / "launch.json").write_text(
        json.dumps({"command": command, "geometry": sizes}, indent=2)
    )
    with (root / "run.log").open("w") as log:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    summary = {"returncode": completed.returncode, "geometry": sizes, "reports": {}}
    for rank in range(sizes["tp"] * sizes["external_dp"]):
        p = root / f"rank_{rank}.json"
        summary["reports"][str(rank)] = (
            json.loads(p.read_text()).get("status") if p.exists() else "missing"
        )
    (root / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
