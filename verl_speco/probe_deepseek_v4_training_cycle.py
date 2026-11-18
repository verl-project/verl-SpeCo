# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""One automatic rollout -> train -> publish cycle, sharing devices without RL.

Each NPU phase runs in a separate process. This tests real data and weight
contracts; it does not claim to validate the resident online trainer lifecycle.
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def prompt_from_trajectory(path, *, vocab_size, max_model_len, response_tokens):
    """Use only the prepared prompt, never supervise the recorded answer."""
    with Path(path).open() as source:
        row = next(json.loads(line) for line in source if line.strip())
    ids, mask = row["input_ids"], row["loss_mask"]
    if (
        len(ids) != len(mask)
        or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
        or any(value not in (0, 1) for value in mask)
        or 1 not in mask
    ):
        raise ValueError(
            "Expected a prepared trajectory with valid IDs and SpeCo loss_mask"
        )
    boundary = mask.index(1)
    if boundary == 0 or len(ids[:boundary]) + response_tokens > max_model_len:
        raise ValueError(
            "Prompt must be nonempty and leave the requested response budget"
        )
    return ids[:boundary]


def cycle_plan(args):
    root = Path(args.output).resolve()
    repo = Path(__file__).resolve().parents[1]
    shared = [
        "--model",
        str(Path(args.model).resolve()),
        "--trajectories",
        str(Path(args.trajectories).resolve()),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
    ]
    capture = [
        sys.executable,
        "-m",
        "verl_speco.probe_deepseek_v4_training_cycle",
        "--stage",
        "capture",
        *shared,
        "--response-tokens",
        str(args.response_tokens),
        "--output",
        str(root / "capture"),
    ]
    script = str(
        repo / "examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh"
    )
    assertion = "draft_engine.target_runtime_version=0"
    train = ["bash", script, "train", assertion]
    preflight = ["bash", script, "preflight", assertion]
    update = [
        sys.executable,
        "-m",
        "verl_speco.probe_deepseek_v4_server_update",
        *shared,
        "--checkpoint",
        str(root / "train/checkpoints/step_00000001"),
        "--output",
        str(root / "update/report"),
    ]
    # Verify using the exact same prompt used for live feature acquisition.
    update[update.index("--trajectories") + 1] = str(root / "prompt.jsonl")
    env = {
        "MODEL_PATH": shared[1],
        "FEATURE_STORE": str(root / "capture/features/target_step_00000000"),
        "TRAIN_TP": str(args.tensor_parallel_size),
        "CARD_COUNT": str(args.tensor_parallel_size * args.data_parallel_size),
        "MAX_STEPS": "1",
        "PYTHON_BIN": sys.executable,
    }
    return {
        "repo": str(repo),
        "environment": env,
        "stages": [
            {"name": "capture", "command": capture, "environment": {}},
            {
                "name": "preflight",
                "command": preflight,
                "environment": {"RUN_DIR": str(root / "preflight")},
            },
            {
                "name": "train",
                "command": train,
                "environment": {"RUN_DIR": str(root / "train")},
            },
            {"name": "update", "command": update, "environment": {}},
        ],
    }


def run_stage(stage, plan, root):
    """Argument arrays only; fail immediately and preserve each phase's log."""
    env = {
        **os.environ,
        **plan["environment"],
        **stage["environment"],
        "PYTHONUNBUFFERED": "1",
    }
    print(f"Starting cycle phase: {stage['name']}", flush=True)
    with (
        (root / (stage["name"] + ".log")).open("w") as log,
        subprocess.Popen(
            stage["command"],
            cwd=plan["repo"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
    ):
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, stage["command"])
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def training_evidence(log_path):
    metrics = None
    coverage = None
    with Path(log_path).open() as source:
        for line in source:
            if "DSpark step=1 metrics=" in line:
                metrics = ast.literal_eval(
                    line.split("DSpark step=1 metrics=", 1)[1].strip()
                )
            if "DSpark first-backward gradient coverage:" in line:
                coverage = ast.literal_eval(
                    line.split("gradient coverage:", 1)[1].strip()
                )
    if (
        not metrics
        or "train_loss" not in metrics
        or any(not math.isfinite(float(v)) for v in metrics.values())
    ):
        raise RuntimeError("Missing or non-finite optimizer-step metrics")
    if (
        metrics.get("optimizer_lr", 0) <= 0
        or metrics.get("optimizer_probe_delta", 0) <= 0
    ):
        raise RuntimeError("No measured parameter change in this training step")
    if not coverage or any(v["missing"] or v["nonfinite"] for v in coverage.values()):
        raise RuntimeError("Missing or non-finite backward gradient coverage")
    return {"metrics": metrics, "gradient_coverage": coverage}


def validate_capture(root, prompt, contract, dp_size):
    """Read back actual durable samples, including the unforwarded terminal token rule."""
    import torch

    from verl_speco.trainer.feature_store import TorchShardFeatureStore

    report = json.loads((root / "capture_report.json").read_text())
    store = TorchShardFeatureStore(
        root / "features/target_step_00000000", read_only=True
    )
    try:
        metadata = store.get_metadata()
        if (
            metadata.get("source") != "verl_rollout"
            or metadata.get("runtime_target_version") != 0
            or metadata.get("ds4_contract") != contract.identity()
        ):
            raise RuntimeError("Captured target version/contract differs")
        keys = list(store.iter_keys())
        requests = report["requests"]
        if len(keys) != dp_size or len(requests) != dp_size:
            raise RuntimeError("Expected one real captured request per DP engine")
        if {row["dp_rank"] for row in requests} != set(range(dp_size)):
            raise RuntimeError("Capture did not cover every DP engine")
        for row in requests:
            sample = store.read(row["key"])
            contract.validate(sample)
            expected = [*prompt, *row["tokens"]][:-1]
            mask = [0] * len(prompt) + [1] * (len(row["tokens"]) - 1)
            if (
                sample.input_ids.tolist() != expected
                or sample.loss_mask.tolist() != mask
                or sample.metadata.get("runtime_target_version") != 0
                or int(sample.loss_mask.sum()) < contract.block_size + 2
                or not torch.isfinite(sample.hidden_states).all()
            ):
                raise RuntimeError(
                    "Live tokens, SpeCo mask, features or target version differ"
                )
        if set(keys) != {row["key"] for row in requests}:
            raise RuntimeError("Capture report keys differ from the durable store")
    finally:
        store.close()
    return report


def capture_phase(args, prompt):
    raise RuntimeError(
        "Rollout hidden-state capture has been removed; use standalone replay"
    )


def run_cycle(args, prompt, contract):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = cycle_plan(args)
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    (root / "prompt.jsonl").write_text(json.dumps({"input_ids": prompt}) + "\n")
    stage = None
    try:
        for stage in plan["stages"]:
            run_stage(stage, plan, root)
            if stage["name"] == "capture":
                validate_capture(
                    root / "capture", prompt, contract, args.data_parallel_size
                )
            elif stage["name"] == "train":
                training_evidence(root / "train.log")
        trained = json.loads((root / "train/training_report.json").read_text())
        updated = json.loads(
            (root / "update/report/server_update_report.json").read_text()
        )
        expected_checkpoint = root / "train/checkpoints/step_00000001"
        if (
            trained.get("status") != "passed"
            or not trained.get("hardware_training_validated")
            or trained["progress"]["global_step"] != 1
            or Path(trained["checkpoint"]).resolve() != expected_checkpoint
            or trained.get("target_runtime_version") != 0
            or trained.get("source_samples") != args.data_parallel_size
            or trained.get("tensor_parallel_size") != args.tensor_parallel_size
            or trained.get("data_parallel_size") != args.data_parallel_size
            or updated.get("target_version") != 0
            or updated.get("drafter_revision") != 1
            or not all(
                updated.get(key) is True
                for key in (
                    "ipc_validated",
                    "server_version_commit_validated",
                    "http_health_validated",
                    "admission_blocked_during_update",
                    "target_tokens_unchanged",
                )
            )
        ):
            raise RuntimeError("Training/publication phase report differs")
        report = {
            "status": "passed",
            "backend": "vllm_ascend",
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "rollout_capture_validated": True,
            "hardware_training_validated": True,
            "fresh_checkpoint_published": str(expected_checkpoint),
            "server_version_commit_validated": True,
            "target_version": 0,
            "drafter_revision": 1,
            "server_restarted_for_training": True,
            "resident_online_trainer_validated": False,
            "rl_integration_validated": False,
            "training_evidence": training_evidence(root / "train.log"),
        }
        (root / "cycle_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except BaseException as error:
        (root / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "stage": stage["name"] if stage else "setup",
                    "error": repr(error),
                },
                indent=2,
            )
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "trajectories", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument(
        "--stage", choices=("cycle", "capture"), default="cycle", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only validation and command plan; no output files",
    )
    args = parser.parse_args()
    parser.error(
        "This rollout-capture training probe was retired; use actor old-logprob or standalone replay"
    )
    if (
        min(args.tensor_parallel_size, args.data_parallel_size) < 1
        or args.response_tokens < 16
    ):
        parser.error("Positive TP/DP and at least 16 response tokens are required")
    from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

    args.model = str(Path(args.model).resolve(strict=True))
    args.output = str(Path(args.output).resolve())
    contract = DeepSeekV4Contract.from_model(args.model)
    prompt = prompt_from_trajectory(
        args.trajectories,
        vocab_size=contract.vocab_size,
        max_model_len=args.max_model_len,
        response_tokens=args.response_tokens,
    )
    if Path(args.output).exists():
        parser.error("Use a fresh output directory")
    if args.dry_run:
        print(json.dumps(cycle_plan(args), indent=2))
        return
    devices = [
        v.strip()
        for v in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if v.strip()
    ]
    if len(devices) != args.tensor_parallel_size * args.data_parallel_size or len(
        set(devices)
    ) != len(devices):
        parser.error("Set exactly TP*DP distinct visible NPUs")
    if args.stage == "capture":
        capture_phase(args, prompt)
    else:
        run_cycle(args, prompt, contract)


if __name__ == "__main__":
    main()
