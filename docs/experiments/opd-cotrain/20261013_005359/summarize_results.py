# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Audit the recorded comparison or summarize a fresh pair of console logs."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "eed8c3a9f1fb40f82fb5e7d0e84996389cd14ef8"
SOURCE_PATH = "docs/experiments/opd-cotrain/20261013_005359"
LOGS = {
    "fixed_drafter": "run_opd_cotrain_dspark_vllm_fsdp2_no_training.log.txt",
    "co_train": "run_opd_cotrain_dspark_vllm_fsdp2.log.txt",
}
METRICS = (
    "drafter/spec_decode/mean_acceptance_length",
    "critic/rewards/mean",
    "perf/time_per_step",
    "timing_s/gen",
    "timing_s/drafter",
    "actor/distillation/abs_loss",
    "actor/pg_loss",
)


def summarize(raw, steps, *, fixed):
    samples = {}
    text = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8"))
    for line in text.splitlines():
        step = re.search(r"\bstep:(\d+)\s+-", line)
        if step is None or not 1 <= int(step[1]) <= steps:
            continue
        values = {}
        for metric in METRICS:
            match = re.search(re.escape(metric) + r":(-?[\d.eE+]+)", line)
            if match:
                values[metric] = float(match[1])
        if values:
            samples[int(step[1])] = values
    if set(samples) != set(range(1, steps + 1)):
        raise ValueError(f"Expected every training step from 1 through {steps}")
    result = {}
    for metric in METRICS:
        values = [row[metric] for row in samples.values() if metric in row]
        if len(values) != steps and not (fixed and metric == "timing_s/drafter"):
            raise ValueError(f"Incomplete metric {metric}: {len(values)}/{steps}")
        result[metric] = {
            "mean": mean(values) if values else None,
            "count": len(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-log", type=Path)
    parser.add_argument("--co-train-log", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--check", action="store_true", help="Compare with summary.json"
    )
    args = parser.parse_args()
    if bool(args.fixed_log) != bool(args.co_train_log):
        parser.error("Supply both --fixed-log and --co-train-log, or neither")
    if args.steps < 1:
        parser.error("--steps must be positive")
    checksums = json.loads((ROOT / "sha256.json").read_text(encoding="utf-8"))
    repo = ROOT.parents[3]
    results = {}
    for mode, filename in LOGS.items():
        path = args.fixed_log if mode == "fixed_drafter" else args.co_train_log
        if path is not None:
            raw = path.read_bytes()
        else:
            raw = subprocess.check_output(
                ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}/{filename}"], cwd=repo
            )
            if hashlib.sha256(raw).hexdigest() != checksums[filename]:
                raise ValueError(f"Recorded log checksum mismatch: {filename}")
        results[mode] = summarize(raw, args.steps, fixed=mode == "fixed_drafter")
    if args.check:
        expected = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
        if results != expected:
            raise ValueError("Recomputed values differ from summary.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
