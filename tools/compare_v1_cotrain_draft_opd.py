# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Compare V1 co-train and Draft-OPD on the same rollout schedule.

Use --dry-run to inspect commands, or --compose-only to resolve both Hydra
configs on the training host without starting Ray/workers. Each arm starts a
fresh process from the same model/drafter checkpoints. No GPU is auto-selected.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TRAINING = "actor_rollout_ref.rollout.drafter.training."
METRIC = re.compile(r"([\w./-]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def sync_recipe(args):
    script = ROOT / f"examples/run_qwen3-8b_drafter_{args.backend}_vllm.sh"
    common = [
        "--config-name=speco_v1_trainer",
        "trainer.v1.trainer_mode=sync",
        "trainer.v1.sync.parameter_sync_step=1",
        "transfer_queue.enable=True",
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.rollout.drafter.model_path={args.drafter}",
        f"data.train_files={args.train_file}",
        f"data.val_files={args.val_file}",
        f"data.seed={args.seed}",
        "data.shuffle=True",
        "data.train_batch_size=8",
        "data.max_prompt_length=512",
        "data.max_response_length=2048",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.resume_mode=disable",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.logger=[console]",
        f"trainer.total_training_steps={args.steps}",
        TRAINING + "collect_interval_steps=1",
        TRAINING + "training_interval_steps=1",
        TRAINING + "step=2",
        TRAINING + "publish_async=false",
        TRAINING + "scheduler.execution.strategy=sync",
        TRAINING + "packing.enable=false",
        TRAINING + "use_data_buffer=false",
        TRAINING + "sample_last_n_steps=1",
        TRAINING + "drafter_convergence_freeze.enabled=false",
        TRAINING + "target_lm_head_row_restricted_sync=false",
        TRAINING + f"lk_temperature={args.temperature}",
    ]
    if args.backend == "dflash":
        common += [
            TRAINING + "dflash_block_size=16",
            "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=15",
        ]
    return script, common


def async_recipe(args):
    script = (
        ROOT / f"examples/dspark_mrv2/run_qwen3-4b_drafter_dspark_{args.mode}_vllm.sh"
    )
    common = [
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.rollout.drafter.model_path={args.drafter}",
        f"data.train_files={args.train_file}",
        f"data.val_files={args.val_file}",
        f"data.seed={args.seed}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        f"trainer.total_training_steps={args.steps}",
        TRAINING + f"lk_temperature={args.temperature}",
    ]
    return script, common


def commands(args):
    mode = getattr(args, "mode", "sync")
    script, common = sync_recipe(args) if mode == "sync" else async_recipe(args)
    result = {}
    for arm in ("co_train", "draft_opd"):
        overrides = list(common)
        if arm == "draft_opd" or args.matched_full_vocab:
            overrides += [
                TRAINING + f"{args.backend}_loss_mode=full_vocab",
                TRAINING + f"{args.backend}_per_position_loss_weight=fixed_exp_decay",
            ]
        if arm == "draft_opd":
            overrides += [
                TRAINING + "drafter_opd.enable=true",
                TRAINING + "drafter_opd.capture_verify_trace=true",
                TRAINING + "drafter_opd.anchor_planner=mixed",
                TRAINING + "drafter_opd.random_anchor_ratio=0.25",
                TRAINING + "drafter_opd.loss=e2e_tv",
            ]
            if mode != "sync":
                overrides += [
                    TRAINING
                    + f"drafter_opd.max_policy_staleness={args.max_policy_staleness}",
                    TRAINING
                    + f"drafter_opd.max_drafter_staleness={args.max_drafter_staleness}",
                ]
        else:
            overrides += [
                TRAINING + "drafter_opd.enable=false",
                TRAINING + "drafter_opd.capture_verify_trace=false",
            ]
        overrides += [
            f"trainer.experiment_name=v1_{mode}_{args.backend}_{arm}",
            f"trainer.default_local_dir={json.dumps((args.output / arm / 'checkpoints').as_posix())}",
            f"trainer.rollout_data_dir={json.dumps((args.output / arm / 'rollouts').as_posix())}",
        ]
        # Shared user overrides are applied identically to both arms.
        overrides += args.override
        if args.compose_only:
            overrides += ["--cfg", "job", "--resolve"]
        result[arm] = [args.bash, script.as_posix(), *overrides]
    return result


def summarize(raw, steps):
    rows = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", raw).splitlines():
        step = re.search(r"\bstep:\s*(\d+)\s+-", line)
        if step and 1 <= int(step[1]) <= steps:
            rows.setdefault(int(step[1]), {}).update(
                {
                    key: float(value)
                    for key, value in METRIC.findall(line)
                    if math.isfinite(float(value))
                }
            )
    missing = sorted(set(range(1, steps + 1)) - rows.keys())
    if missing:
        raise ValueError(f"Incomplete training log; missing steps: {missing}")
    metrics = sorted({key for row in rows.values() for key in row if key != "step"})
    stats = {}
    for key in metrics:
        values = [row[key] for row in rows.values() if key in row]
        stats[key] = {
            "mean": mean(values),
            "count": len(values),
            "min": min(values),
            "max": max(values),
        }
    return stats


def check_training(stats, *, opd, backend):
    required = ["drafter/train_optimizer_step_max", "drafter/published"]
    if opd:
        required += [f"{backend}/opd/trace_blocks"]
    for key in required:
        if key not in stats or stats[key]["max"] <= 0:
            raise ValueError(f"Missing positive training evidence: {key}")
    optimizer = stats["drafter/train_optimizer_step_max"]
    if optimizer["count"] > 1 and optimizer["max"] <= optimizer["min"]:
        raise ValueError("Drafter optimizer step did not increase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["dspark", "dflash"], default="dspark")
    parser.add_argument(
        "--mode", choices=["sync", "colocate_async", "separate_async"], default="sync"
    )
    parser.add_argument("--max-policy-staleness", type=int, default=0)
    parser.add_argument("--max-drafter-staleness", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--bash", default="bash")
    parser.add_argument(
        "--matched-full-vocab",
        action="store_true",
        help="Also use full-vocab fixed-decay loss in the baseline; default keeps its existing loss.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Shared Hydra override (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    if args.mode != "sync" and args.backend != "dspark":
        parser.error("The existing V1 async examples use DSpark")
    if min(args.max_policy_staleness, args.max_drafter_staleness) < 0:
        parser.error("Staleness limits must be nonnegative")
    if args.temperature is None:
        args.temperature = 0.6 if args.mode == "sync" else 1.0
    if args.steps < 2 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Use --steps >= 2 and a finite positive --temperature")
    args.output = args.output.resolve()
    cmds = commands(args)
    if args.dry_run:
        print(json.dumps(cmds, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "commands.json"
    if manifest.exists():
        parser.error("Output directory already contains a run; choose a fresh --output")
    manifest.write_text(
        json.dumps(
            {
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "commands": cmds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    results = {}
    for arm, command in cmds.items():
        print(f"{arm}: {shlex.join(command)}", flush=True)
        log = args.output / (arm + (".yaml" if args.compose_only else ".log"))
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not args.compose_only:
            stats = summarize(log.read_text(encoding="utf-8"), args.steps)
            check_training(stats, opd=arm == "draft_opd", backend=args.backend)
            results[arm] = stats
    if not args.compose_only:
        (args.output / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
