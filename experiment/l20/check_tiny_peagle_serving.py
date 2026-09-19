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
                widths = {"q_proj": (0, 64), "k_proj": (64, 96), "v_proj": (96, 128)}
                start, end = widths[projection]
                actual = parameters[target][start:end]
            elif projection in {"gate_proj", "up_proj"}:
                target = target.replace(projection, "gate_up_proj")
                start = 0 if projection == "gate_proj" else 128
                actual = parameters[target][start : start + 128]
            else:
                actual = parameters[target]
            used.add(target)
        torch.testing.assert_close(actual, value.to(actual), rtol=0, atol=0)
        compared += 1
    runtime_only = sorted(parameters.keys() - used)
    assert runtime_only == ["model.layers.1.hidden_norm.weight"], runtime_only
    return {
        "class": type(draft).__name__,
        "compared_logical_tensors": compared,
        "runtime_only_parameters": runtime_only,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["baseline", "draft-original", "draft-mapped", "draft-eagle3"]
    )
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=Path("/experiment/tiny-peagle/draft-original"),
    )
    parser.add_argument("--output-prefix", default="tiny-peagle")
    args = parser.parse_args()
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
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        speculative_config=speculative,
    )
    prompts = [
        {"prompt_token_ids": tokens} for tokens in [[1, 4, 7], [1, 8, 9, 10, 11]]
    ]
    if speculative is not None:
        llm.collective_rpc(capture_first_forward, kwargs={"prefix": args.output_prefix})
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    report = {"mode": args.mode, "token_ids": [o.outputs[0].token_ids for o in outputs]}
    if speculative is not None:
        report["parameters"] = llm.collective_rpc(
            check_parameters, kwargs={"checkpoint": str(args.reference_model)}
        )
    Path(
        f"/experiment/evidence/l20-20260919/{args.output_prefix}-{args.mode}.json"
    ).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
