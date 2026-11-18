# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Capture HS/pre-HC for fixed existing trajectories, without RL or new answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from verl_speco.collect_deepseek_v4_features import (
    build_capture_llm,
    capture_request,
    close_capture,
)
from verl_speco.producer.trajectory_reader import (
    iter_trajectories,
    prepare_trajectories,
)
from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
)
from verl_speco.trainer.feature_store import TorchShardFeatureStore


def replay_features(
    llm,
    records,
    *,
    contract,
    output,
    sampling_params,
    capture_max_bytes=4 * 1024**3,
    algorithm="dspark",
):
    """Prefill the entire fixed trajectory and discard the extra sampled token.

    AIR's DSpark capture script also feeds serialized trajectories as prefill
    and captures selected target aux states plus pre-HC. Here the maintained
    SpeCo collector validates absolute positions and the exact token prefix.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "model_family": "deepseek_v4",
        "target_layer_ids": list(getattr(contract, "target_layer_ids", ())),
        "hidden_size": contract.hidden_size,
        "hidden_states_layout": contract.identity()["layout"],
        "max_rows_per_request": llm.llm_engine.model_config.max_model_len,
        "max_pending_bytes": capture_max_bytes,
    }
    sample_factory = from_engine_tensors
    if algorithm == "mtp":
        from verl_speco.trainer.deepseek_v4_mtp_contract import (
            from_engine_tensors as mtp_sample,
        )

        sample_factory = mtp_sample
        settings.update(feature_kind="mtp", hc_mult=contract.hc_mult)
    store = TorchShardFeatureStore(
        output,
        max_samples_per_shard=1,
        metadata={
            "ds4_contract": contract.identity(),
            "source": "trajectory_replay",
        },
    )
    reports = []
    try:
        for record in records:
            _, final, payload = capture_request(
                llm.llm_engine, record.input_ids, sampling_params, settings
            )
            if len(final) != len(record.input_ids) + 1:
                raise ValueError(
                    "Trajectory replay must sample exactly one discarded token"
                )
            if payload["input_ids"].tolist() != record.input_ids:
                raise ValueError(
                    "Captured input IDs differ from the recorded trajectory"
                )
            if not torch.equal(
                payload["positions"], torch.arange(len(record.input_ids))
            ):
                raise ValueError("Replay positions differ from the recorded trajectory")
            feature = sample_factory(
                payload, record.loss_mask, contract, sample_id=record.sample_id
            )
            feature.metadata.update(
                source="trajectory_replay",
                source_line=record.line_number,
                source_file=record.source_file,
            )
            store.write_many([feature])
            reports.append(
                {
                    "sample_id": record.sample_id,
                    "source_line": record.line_number,
                    "source_file": record.source_file,
                    "feature_rows": len(record.input_ids),
                    "supervised_rows": int(sum(record.loss_mask)),
                }
            )
    finally:
        close_capture(llm, store)
    if not reports:
        raise ValueError("Trajectory input contains no records")
    report = {
        "status": "passed",
        "source": "trajectory_replay",
        "samples": reports,
        "contract": contract.identity(),
        "rl_integration_validated": False,
    }
    (output / "replay_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(*, algorithm="dspark"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--trajectories",
        required=True,
        help="JSONL file or directory of existing trajectories",
    )
    parser.add_argument(
        "--output", required=True, help="New SpeCo Feature Store directory"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CPU tokenization/mask preparation; no NPU model",
    )
    parser.add_argument("--capture-max-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if (
        args.tensor_parallel_size < 1
        or args.data_parallel_size < 1
        or args.max_num_batched_tokens < 1
        or args.capture_max_bytes < 1
        or args.max_model_len < 4
        or (args.max_samples is not None and args.max_samples < 1)
    ):
        parser.error(
            "TP, max-samples must be positive; max-model-len must be at least 4"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    if args.data_parallel_size > 1 and not args.enable_expert_parallel:
        parser.error("DP>1 capture requires --enable-expert-parallel")
    if Path(args.output).exists():
        parser.error("output already exists; use a new directory")
    if not Path(args.trajectories).exists():
        parser.error("trajectory JSONL file/directory does not exist")
    contract_type = DeepSeekV4Contract
    if algorithm == "mtp":
        from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract

        contract_type = DeepSeekV4MTPContract
    contract = contract_type.from_model(args.model)
    config = json.loads((Path(args.model) / "config.json").read_text())
    if config.get("quantization_config"):
        parser.error("Replay requires the 0731 BF16 checkpoint")
    if args.prepare_only:
        from vllm.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(args.model, tokenizer_mode="deepseek_v4")
        report = prepare_trajectories(
            iter_trajectories(
                args.trajectories,
                tokenizer,
                vocab_size=contract.vocab_size,
                max_model_len=args.max_model_len,
                max_samples=args.max_samples,
            ),
            args.output,
            metadata={
                "target_model_path": str(Path(args.model).resolve()),
                "contract": contract.identity(),
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "samples": report["sample_count"],
                    "max_tokens": report["max_tokens"],
                    "output": args.output,
                }
            )
        )
        return
    llm = build_capture_llm(
        args.model,
        contract,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        **({"speculative_method": "mtp"} if algorithm == "mtp" else {}),
    )
    from vllm import SamplingParams

    records = iter_trajectories(
        args.trajectories,
        llm.get_tokenizer(),
        vocab_size=contract.vocab_size,
        max_model_len=args.max_model_len,
        max_samples=args.max_samples,
    )
    report = replay_features(
        llm,
        records,
        contract=contract,
        output=args.output,
        sampling_params=SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, n=1
        ),
        capture_max_bytes=args.capture_max_bytes,
        **({"algorithm": "mtp"} if algorithm == "mtp" else {}),
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
