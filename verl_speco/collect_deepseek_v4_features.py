# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real Ascend rollout capture into the existing SpeCo store, without RL."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

import torch

from verl_speco.integration.ascend_runtime_compat import require_ascend_runtime
from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
)
from verl_speco.trainer.feature_store import TorchShardFeatureStore
from verl_speco.transport.rollout_feature_payload import decode_feature_payload


def _rpc(engine, operation, **kwargs):
    return engine.collective_rpc(
        "speco_standalone_capture", kwargs={"operation": operation, **kwargs}
    )


def close_capture(llm, store):
    try:
        _rpc(llm.llm_engine, "close")
    finally:
        try:
            store.close()
        finally:
            shutdown = getattr(llm, "shutdown_capture_frontends", None)
            if callable(shutdown):
                shutdown()


def capture_request(engine, prompt_ids, sampling_params, settings):
    """Register before enqueueing; consume actual generation steps exactly once."""
    if engine.has_unfinished_requests():
        raise ValueError("Standalone capture requires an idle, dedicated engine")
    request_id = f"speco-smoke-{uuid.uuid4().hex}"
    try:
        _rpc(
            engine,
            "begin",
            request_id=request_id,
            prompt_ids=prompt_ids,
            settings=settings,
        )
        engine.add_request(
            request_id, {"prompt_token_ids": prompt_ids}, sampling_params
        )
        completed = []
        steps = 0
        while engine.has_unfinished_requests():
            steps += 1
            if steps > settings["max_rows_per_request"] + 16:
                raise RuntimeError("Standalone generation exceeded its step budget")
            completed.extend(output for output in engine.step() if output.finished)
        if len(completed) != 1 or len(completed[0].outputs) != 1:
            raise ValueError("Capture requires exactly one finished completion")
        result = completed[0]
        if list(result.prompt_token_ids) != list(prompt_ids):
            raise ValueError("Serving prompt tokens differ from registered capture")
        final_ids = [*prompt_ids, *result.outputs[0].token_ids]
        payloads = _rpc(engine, "finish", request_id=request_id, final_ids=final_ids)
        payloads = [value for value in payloads if value is not None]
        if len(payloads) != 1:
            raise ValueError("Expected exactly one TP-leader feature payload")
        return request_id, final_ids, decode_feature_payload(payloads[0])
    finally:
        # Abort the dedicated request on failure and always release host buffers.
        try:
            if engine.has_unfinished_requests():
                engine.abort_request([request_id])
        finally:
            _rpc(engine, "cancel", request_id=request_id)


def collect_features(llm, prompts, *, contract, output, max_tokens, sampling_params):
    """Use the same collector, contract and store as the RL rollout path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "model_family": "deepseek_v4",
        "target_layer_ids": list(contract.target_layer_ids),
        "hidden_size": contract.hidden_size,
        "hidden_states_layout": contract.identity()["layout"],
        "max_rows_per_request": llm.llm_engine.model_config.max_model_len,
    }
    store = TorchShardFeatureStore(
        output,
        metadata={
            "ds4_contract": contract.identity(),
            "source": "standalone_rollout",
            "runtime_target_version": 0,
        },
    )
    records = []
    try:
        for prompt in prompts:
            ids = prompt.get("prompt_token_ids")
            if ids is None:
                ids = llm.get_tokenizer().encode(prompt["prompt"])
            if not ids or any(
                type(token) is not int or not 0 <= token < contract.vocab_size
                for token in ids
            ):
                raise ValueError("Prompt must contain nonempty vocabulary token IDs")
            if len(ids) + max_tokens > settings["max_rows_per_request"]:
                raise ValueError("Prompt plus generation exceeds max_model_len")
            request_id, final, payload = capture_request(
                llm.llm_engine, ids, sampling_params, settings
            )
            rows = payload["input_ids"].numel()
            # The terminal sampled token has no forward/HS. The shared collector
            # validates the full confirmed prefix and excludes that terminal row.
            mask = torch.zeros(rows)
            mask[len(ids) :] = 1
            if not bool(mask.any()):
                raise ValueError("Completion contains no supervised forward rows")
            sample = from_engine_tensors(payload, mask, contract, sample_id=request_id)
            sample.metadata.update(
                source="standalone_rollout", runtime_target_version=0
            )
            store.write_many([sample])
            records.append(
                {
                    "sample_id": request_id,
                    "prompt_tokens": len(ids),
                    "generated_token_ids": final[len(ids) :],
                    "feature_rows": rows,
                    "supervised_rows": int(mask.sum()),
                }
            )
    finally:
        close_capture(llm, store)
    if not records:
        raise ValueError("Provide at least one prompt")
    report = {
        "status": "passed",
        "samples": records,
        "contract": contract.identity(),
        "source": "standalone_rollout",
        "rl_integration_validated": False,
    }
    (output / "capture_report.json").write_text(json.dumps(report, indent=2))
    return report


def build_capture_llm(
    model,
    contract,
    *,
    tensor_parallel_size,
    max_model_len,
    gpu_memory_utilization,
    max_num_batched_tokens=None,
    data_parallel_size=1,
    enable_expert_parallel=False,
    worker_extension_cls="verl_speco.integration.standalone_rollout_capture.StandaloneCaptureWorkerExtension",
    speculative_method="dspark",
):
    """Construct the reviewed eager Ascend capture runtime for both input modes."""
    require_ascend_runtime(
        *(importlib.metadata.version(name) for name in ("vllm", "vllm-ascend"))
    )
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    os.environ.setdefault("VLLM_ASCEND_APPLY_DSV4_PATCH", "1")
    if int(data_parallel_size) > 1:
        from verl_speco.integration.standalone_capture_parallel import (
            ParallelCaptureLLM,
        )

        return ParallelCaptureLLM(
            data_parallel_size=data_parallel_size,
            options={
                "model": model,
                "contract": contract,
                "tensor_parallel_size": tensor_parallel_size,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_num_batched_tokens": max_num_batched_tokens,
                "enable_expert_parallel": enable_expert_parallel,
                "worker_extension_cls": worker_extension_cls,
                "speculative_method": speculative_method,
            },
        )
    from vllm import LLM

    return LLM(
        model=model,
        dtype="bfloat16",
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=enable_expert_parallel,
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=min(
            max_model_len, max_num_batched_tokens or max_model_len
        ),
        block_size=128,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=bool(
            max_num_batched_tokens and max_num_batched_tokens < max_model_len
        ),
        async_scheduling=False,
        additional_config={"enable_flashcomm1": False},
        speculative_config={
            "method": speculative_method,
            "num_speculative_tokens": contract.block_size,
            "enforce_eager": True,
        },
        worker_extension_cls=worker_extension_cls,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="New feature-store directory")
    parser.add_argument(
        "--prompts", help='JSONL: {"prompt":"text"} or {"prompt_token_ids":[...]}'
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if (
        min(args.tensor_parallel_size, args.max_tokens) <= 0
        or args.max_model_len <= args.max_tokens
    ):
        parser.error(
            "TP and max-tokens must be positive; max-model-len must exceed max-tokens"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new directory")
    contract = DeepSeekV4Contract.from_model(args.model)
    config = json.loads((Path(args.model) / "config.json").read_text())
    if config.get("quantization_config"):
        parser.error("This smoke path requires the 0731 BF16 checkpoint")
    prompts = (
        [
            json.loads(line)
            for line in Path(args.prompts).read_text().splitlines()
            if line.strip()
        ]
        if args.prompts
        else [
            {"prompt": "Explain why the sky appears blue in several sentences."},
            {"prompt": "Explain how to solve a quadratic equation step by step."},
        ]
    )
    if not prompts or any(
        not isinstance(p, dict) or ("prompt" in p) == ("prompt_token_ids" in p)
        for p in prompts
    ):
        parser.error(
            "Each prompt must specify exactly one of prompt or prompt_token_ids"
        )
    llm = build_capture_llm(
        args.model,
        contract,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    from vllm import SamplingParams

    report = collect_features(
        llm,
        prompts,
        contract=contract,
        output=args.output,
        max_tokens=args.max_tokens,
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            ignore_eos=True,
            n=1,
        ),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(report["samples"]),
                "store": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
