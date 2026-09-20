"""Publish a trained checkpoint at a drained engine boundary, then cold-check it."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from check_tiny_peagle_serving import (
    check_parameters,
    count_graph_replays,
    graph_counts,
)
from safetensors.torch import load_file
from vllm import LLM, SamplingParams


def target_digest(worker):
    digest = hashlib.sha256()
    for name, value in worker.model_runner.model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


def publish(worker, checkpoint):
    draft = worker.get_draft_model()
    target_addresses = {p.data_ptr() for p in worker.model_runner.model.parameters()}
    assert all(p.data_ptr() not in target_addresses for p in draft.parameters())
    before = target_digest(worker)
    draft.load_weights(load_file(str(Path(checkpoint) / "model.safetensors")).items())
    proposer = worker.model_runner.drafter
    cache = proposer.parallel_drafting_hidden_state_tensor
    expected = draft.combine_hidden_states(draft.mask_hidden.view(-1))
    stale_error = (cache - expected).abs().max().item()
    address = cache.data_ptr()
    cache.copy_(expected)
    assert cache.data_ptr() == address
    assert target_digest(worker) == before
    return {
        "parameters": check_parameters(worker, checkpoint),
        "target_unchanged": True,
        "stale_mask_cache_error": stale_error,
        "cache_address_preserved": True,
    }


def capture_logits(worker, output):
    from vllm.distributed import get_tensor_model_parallel_rank

    draft = worker.get_draft_model()

    def capture(module, inputs, kwargs, result):
        logits = module.compute_logits(result[0])
        if logits is not None:
            path = f"{output}.rank{get_tensor_model_parallel_rank()}.pt"
            torch.save(logits.detach().cpu(), path)
        handle.remove()

    handle = draft.register_forward_hook(capture, with_kwargs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hot", "cold"])
    parser.add_argument("target")
    parser.add_argument("initial")
    parser.add_argument("updated")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    llm = LLM(
        model=args.target,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        enforce_eager=not args.graph,
        disable_custom_all_reduce=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]}
        if args.graph
        else {},
        speculative_config={
            "method": "eagle3",
            "parallel_drafting": True,
            "num_speculative_tokens": 3,
            "model": args.initial if args.mode == "hot" else args.updated,
        },
    )
    prompts = [{"prompt_token_ids": [1, 4, 7]}, {"prompt_token_ids": [1, 8, 9, 10, 11]}]
    sampling = SamplingParams(temperature=0, max_tokens=16)
    if args.graph:
        llm.collective_rpc(count_graph_replays)
    first = llm.generate(prompts, sampling)
    graph_before = llm.collective_rpc(graph_counts) if args.graph else []
    target_before = llm.collective_rpc(target_digest)
    updates = (
        llm.collective_rpc(publish, kwargs={"checkpoint": args.updated})
        if args.mode == "hot"
        else []
    )
    if not args.graph:
        llm.collective_rpc(capture_logits, kwargs={"output": str(args.output)})
    result = llm.generate(prompts, sampling)
    tokens = [item.outputs[0].token_ids for item in result]
    assert tokens == [item.outputs[0].token_ids for item in first]
    assert llm.collective_rpc(target_digest) == target_before
    graph = llm.collective_rpc(graph_counts) if args.graph else []
    assert all(
        after["replays"] > before["replays"]
        for before, after in zip(graph_before, graph, strict=True)
    )
    report = {
        "mode": args.mode,
        "tp": args.tp,
        "graph": args.graph,
        "target_hashes": target_before,
        "tokens": tokens,
        "updates": updates,
        "graph_execution": graph,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
