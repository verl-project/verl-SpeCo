# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone DSpark update probe using checkpoint RPC or real bucketed IPC."""

import argparse
import json
from pathlib import Path

from verl_speco.collect_deepseek_v4_features import build_capture_llm
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract


def update_phase(engine, operation, checkpoint=None):
    rpc = getattr(engine, "draft_update_rpc", engine.collective_rpc)
    return rpc(
        "speco_standalone_draft_update",
        kwargs={"operation": operation, "checkpoint": checkpoint},
    )


def require_phase(results, status, tp, dp):
    if len(results) != tp * dp or any(
        result.get("status") != status for result in results
    ):
        raise RuntimeError(f"Draft update {status} failed: {results}")
    if all("dp_rank" in result for result in results) and {
        (row["tp_rank"], row["dp_rank"]) for row in results
    } != {(t, d) for t in range(tp) for d in range(dp)}:
        raise RuntimeError("Draft update did not acknowledge every TP/DP worker")
    for rank in range(tp):
        if sum(result.get("tp_rank") == rank for result in results) != dp:
            raise RuntimeError("Draft update did not acknowledge every TP/DP worker")


def generate(engine, ids, request_id):
    from vllm import SamplingParams

    engine.add_request(
        request_id,
        {"prompt_token_ids": ids},
        SamplingParams(temperature=0, max_tokens=8),
    )
    results = []
    while engine.has_unfinished_requests():
        results.extend(result for result in engine.step() if result.finished)
    if len(results) != 1 or len(results[0].outputs) != 1:
        raise RuntimeError("Expected one finished greedy completion")
    return list(results[0].outputs[0].token_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--trajectories", required=True, help="Prepared JSONL containing input_ids"
    )
    parser.add_argument("--output", required=True, help="New report directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--transport", choices=("checkpoint", "ipc"), default="checkpoint"
    )
    parser.add_argument("--ipc-timeout", type=int, default=600)
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.data_parallel_size, args.ipc_timeout) <= 0:
        parser.error("TP, DP and IPC timeout must be positive")
    contract = DeepSeekV4Contract.from_model(args.model)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    metadata = json.loads((checkpoint / "speco_ds4.json").read_text())
    from verl_speco.integration.vllm_ascend_implementation import (
        compatible_implementation_fingerprint,
    )

    if not compatible_implementation_fingerprint(
        metadata["identity"]["source_fingerprint"]
    ):
        raise ValueError("Checkpoint implementation differs from the current backend")
    saved_topology = json.loads((checkpoint / "parallel_topology.json").read_text())
    for name in ("world_size", "tensor_parallel_size", "data_parallel_size"):
        if saved_topology[name] != metadata["topology"][name]:
            raise ValueError("Checkpoint topology metadata differs")
    if metadata["identity"]["contract"] != contract.identity():
        raise ValueError("Checkpoint target contract differs")
    if metadata["topology"]["tensor_parallel_size"] != args.tensor_parallel_size:
        raise ValueError("Training and serving attention TP must match")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with Path(args.trajectories).open() as source:
        row = next(json.loads(line) for line in source if line.strip())
    ids = row["input_ids"]
    if not ids or any(
        type(i) is not int or not 0 <= i < contract.vocab_size for i in ids
    ):
        raise ValueError("Invalid prepared token IDs")
    if len(ids) + 8 > args.max_model_len:
        raise ValueError("Trajectory exceeds probe context window")
    llm = None
    build_options = {}
    if args.transport == "ipc":
        from verl_speco.integration.drafter_config_env import set_drafter_config_env

        set_drafter_config_env(
            json.dumps(
                {
                    "training": {
                        "backend": "vllm_ascend",
                        "collect_hidden_states_from_rollout": True,
                        "rollout_capture": {"model_family": "deepseek_v4"},
                    }
                }
            )
        )
        build_options["worker_extension_cls"] = (
            "verl_speco.integration.deepseek_v4_ipc_probe.StandaloneIPCDraftWorkerExtension"
        )
    try:
        llm = build_capture_llm(
            args.model,
            contract,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            enable_expert_parallel=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.8,
            **build_options,
        )
        engine = llm.llm_engine
        before = generate(engine, ids, "speco-update-before")
        print("Preparing every runtime worker", flush=True)
        prepared = update_phase(engine, "prepare", str(checkpoint))
        require_phase(
            prepared, "prepared", args.tensor_parallel_size, args.data_parallel_size
        )
        if args.data_parallel_size > 1 and any(
            result["expert_tensors"] == 0 for result in prepared
        ):
            raise RuntimeError(
                "No EP expert tensor conversion observed on a runtime worker"
            )
        print("Committing verified runtime weights", flush=True)
        sent = []
        if args.transport == "ipc":
            from verl_speco.integration.deepseek_v4_ipc_probe import transfer_ipc_probe

            committed, sent = transfer_ipc_probe(
                engine,
                str(checkpoint),
                args.tensor_parallel_size,
                args.data_parallel_size,
                args.ipc_timeout,
            )
        else:
            committed = update_phase(engine, "commit")
            require_phase(
                committed,
                "committed",
                args.tensor_parallel_size,
                args.data_parallel_size,
            )
        after = generate(engine, ids, "speco-update-after")
        if before != after:
            raise RuntimeError("Greedy target tokens changed after draft-only update")
        finalized = update_phase(engine, "finalize")
        require_phase(
            finalized, "finalized", args.tensor_parallel_size, args.data_parallel_size
        )
        report = {
            "status": "passed",
            "prepared": prepared,
            "committed": committed,
            "target_tokens_unchanged": True,
            "tokens": after,
            "checkpoint": str(checkpoint),
            "ipc_validated": args.transport == "ipc",
            "runtime_revision_validated": args.transport == "ipc",
            "server_version_commit_validated": False,
            "transport": args.transport,
            "sent": sent,
            "rl_integration_validated": False,
        }
        (output / "update_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except Exception as error:
        rollback = None
        if llm is not None:
            try:
                if args.transport == "ipc":
                    rpc = getattr(
                        llm.llm_engine,
                        "draft_update_rpc",
                        llm.llm_engine.collective_rpc,
                    )
                    rollback = rpc(
                        "speco_standalone_ipc_update", kwargs={"operation": "rollback"}
                    )
                else:
                    rollback = update_phase(llm.llm_engine, "rollback")
            except Exception as cleanup:  # noqa: BLE001 -- preserve the original failure
                rollback = {"error": repr(cleanup)}
        (output / "failure.json").write_text(
            json.dumps({"error": repr(error), "rollback": rollback}, indent=2)
        )
        raise
    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown_capture_frontends", None)
            if callable(shutdown):
                shutdown()
            else:
                llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
