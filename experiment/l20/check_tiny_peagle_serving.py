"""Small checkpoint loading and greedy-generation probe; not a quality test."""

import argparse
import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams


def capture_first_forward(worker, prefix):
    from vllm.forward_context import get_forward_context

    draft = worker.get_draft_model()
    signature = inspect.signature(draft.forward)
    records = []
    initial_inputs = {}

    def capture_inputs(module, inputs, kwargs):
        arguments = signature.bind(*inputs, **kwargs).arguments
        initial_inputs.clear()
        initial_inputs.update(
            (key, value.detach().cpu())
            for key, value in arguments.items()
            if isinstance(value, torch.Tensor)
        )

    def capture(module, inputs, kwargs, outputs):
        metadata = get_forward_context().attn_metadata
        record = {
            "inputs": dict(initial_inputs),
            "outputs": [value.detach().cpu() for value in outputs],
            "logits": module.compute_logits(outputs[0]).detach().cpu(),
            "metadata": {
                name: {
                    key: value.detach().cpu()
                    for key, value in vars(layer).items()
                    if isinstance(value, torch.Tensor)
                }
                for name, layer in metadata.items()
            },
        }
        records.append(record)
        torch.save(records, f"/experiment/evidence/l20-20260919/{prefix}-forwards.pt")
        if len(records) == 1:
            torch.save(
                record, f"/experiment/evidence/l20-20260919/{prefix}-first-forward.pt"
            )

    draft.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    draft.register_forward_hook(capture, with_kwargs=True)


def check_parameters(worker, checkpoint):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    tp = get_tensor_model_parallel_world_size()
    draft = worker.get_draft_model()
    expected = load_file(str(Path(checkpoint) / "model.safetensors"))
    parameters = dict(draft.named_parameters())
    used = set()
    compared = 0
    for name, value in expected.items():
        if name == "t2d":
            continue  # Runtime uses d2t offsets to expand logits to target IDs.
        if name == "mask_hidden":
            actual = draft.mask_hidden
            value = value.reshape_as(actual)
        else:
            target = "draft_id_to_target_id" if name == "d2t" else name
            if name not in {"d2t", "lm_head.weight"}:
                target = "model." + name
            projection = name.rpartition(".")[0].rpartition(".")[2]
            if projection in {"q_proj", "k_proj", "v_proj"}:
                target = target.replace(projection, "qkv_proj")
                widths = {
                    "q_proj": (0, 64 // tp),
                    "k_proj": (64 // tp, 96 // tp),
                    "v_proj": (96 // tp, 128 // tp),
                }
                start, end = widths[projection]
                actual = parameters[target][start:end]
                value = value.chunk(tp, dim=0)[rank]
            elif projection in {"gate_proj", "up_proj"}:
                target = target.replace(projection, "gate_up_proj")
                start = 0 if projection == "gate_proj" else 128 // tp
                actual = parameters[target][start : start + 128 // tp]
                value = value.chunk(tp, dim=0)[rank]
            else:
                actual = parameters[target]
                if projection in {"o_proj", "down_proj"}:
                    value = value.chunk(tp, dim=1)[rank]
                elif name in {"embed_tokens.weight", "lm_head.weight"}:
                    value = value.chunk(tp, dim=0)[rank]
            used.add(target)
        torch.testing.assert_close(actual, value.to(actual), rtol=0, atol=0)
        compared += 1
    runtime_only = sorted(parameters.keys() - used)
    assert runtime_only == ["model.layers.1.hidden_norm.weight"], runtime_only
    return {
        "rank": rank,
        "tp": tp,
        "class": type(draft).__name__,
        "compared_logical_tensors": compared,
        "runtime_only_parameters": runtime_only,
    }


def count_graph_replays(worker):
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    graph_roles = {
        id(entry.cudagraph): type(wrapper.runnable).__name__
        for wrapper in CUDAGraphWrapper._all_instances
        for entry in wrapper.concrete_cudagraph_entries.values()
        if entry.cudagraph is not None
    }
    worker.speco_graph_replays = 0
    worker.speco_graph_roles = {}
    original = torch.cuda.CUDAGraph.replay

    def replay(graph):
        worker.speco_graph_replays += 1
        role = graph_roles.get(id(graph), "unclassified")
        worker.speco_graph_roles[role] = worker.speco_graph_roles.get(role, 0) + 1
        return original(graph)

    torch.cuda.CUDAGraph.replay = replay


def graph_counts(worker):
    from vllm.compilation.counter import compilation_counter

    return {
        "captures": compilation_counter.num_cudagraph_captured,
        "replays": worker.speco_graph_replays,
        "roles": worker.speco_graph_roles,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["baseline", "draft-original", "draft-mapped", "draft-eagle3"]
    )
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=Path("/experiment/tiny-peagle/draft-original"),
    )
    parser.add_argument("--output-prefix", default="tiny-peagle")
    args = parser.parse_args()
    if (args.tp > 1 or args.graph) and not args.no_capture:
        parser.error(
            "TP and graph runs require --no-capture; use the eager logits oracle separately"
        )
    root = Path("/experiment/tiny-peagle")
    speculative = (
        None
        if args.mode == "baseline"
        else {
            "method": "eagle3",
            "model": str(args.draft_model or root / args.mode),
            "num_speculative_tokens": 3,
            "parallel_drafting": True,
        }
    )
    llm = LLM(
        model=str(root / "target"),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=not args.graph,
        tensor_parallel_size=args.tp,
        disable_custom_all_reduce=True,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]}
        if args.graph
        else {},
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        enable_prefix_caching=False,
        speculative_config=speculative,
    )
    prompts = [
        {"prompt_token_ids": tokens} for tokens in [[1, 4, 7], [1, 8, 9, 10, 11]]
    ]
    if speculative is not None and not args.no_capture:
        llm.collective_rpc(capture_first_forward, kwargs={"prefix": args.output_prefix})
    if args.graph:
        llm.collective_rpc(count_graph_replays)
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    repeated = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    assert [o.outputs[0].token_ids for o in outputs] == [
        o.outputs[0].token_ids for o in repeated
    ]
    report = {"mode": args.mode, "token_ids": [o.outputs[0].token_ids for o in outputs]}
    if speculative is not None:
        report["parameters"] = llm.collective_rpc(
            check_parameters, kwargs={"checkpoint": str(args.reference_model)}
        )
    report["tp"] = args.tp
    report["graph"] = args.graph
    if args.graph:
        report["graph_execution"] = llm.collective_rpc(graph_counts)
        assert all(row["replays"] > 0 for row in report["graph_execution"])
    Path(
        f"/experiment/evidence/l20-20260919/{args.output_prefix}-{args.mode}.json"
    ).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
