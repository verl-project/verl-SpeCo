# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the native vLLM speculative-decoding runtime for the C5 fixture.

Loads the target with an EAGLE3 (or DFLASH) drafter under the same worker-extension path the
online loop uses, then resolves the draft model and reports the runtime configuration. This
catches missing architecture support before a full training run.
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--algorithm", default="EAGLE3", choices=["EAGLE3", "DFLASH"])
    parser.add_argument("--gpu", default="1,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    speculative_config = {
        "method": args.algorithm.lower(),
        "model": args.drafter,
        "num_speculative_tokens": 1,
    }
    llm = LLM(
        model=args.target,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.30,
        max_model_len=256,
        trust_remote_code=False,
        speculative_config=speculative_config,
    )
    outputs = llm.generate(
        ["Calculate 173 times 29. Explain your calculation in two sentences, then write #### followed by the final number."],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    text = outputs[0].outputs[0].text
    report = {
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "algorithm": args.algorithm,
        "generated_prefix": text[:120],
        "spec_metrics": sorted({k for k in outputs[0].metrics.__dict__} if hasattr(outputs[0], "metrics") else []),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
