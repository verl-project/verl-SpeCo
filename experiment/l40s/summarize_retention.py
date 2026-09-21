#!/usr/bin/env python3
"""Summarize a SpeCo native online run: step times, train/publish counts, retention audit."""

import argparse
import json
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"timing_s/step:([0-9.]+)")
TRAINED_RE = re.compile(r"drafter/trained:(?:np\.float64\()?([0-9.]+)")
PUBLISHED_RE = re.compile(r"drafter/published:(?:np\.float64\()?([0-9.]+)")
RETAIN_RE = re.compile(
    r"after_target_sync rank=(\d+) latest_draft_retained=(True|False) expected=([0-9a-f]+) actual=([0-9a-f]+)"
)
PUBLIC_RE = re.compile(r"public_load rank=(\d+) fp=([0-9a-f]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    steps = [float(value) for value in STEP_RE.findall(text)]
    trained = [float(value) for value in TRAINED_RE.findall(text)]
    published = [float(value) for value in PUBLISHED_RE.findall(text)]
    retention = RETAIN_RE.findall(text)
    public_loads = PUBLIC_RE.findall(text)

    retained = sum(1 for _, kept, _, _ in retention if kept == "True")
    summary = {
        "log": str(args.log),
        "logged_steps": len(steps),
        "first_step_seconds": steps[0] if steps else None,
        "steady_mean_seconds": round(sum(steps[1:]) / len(steps[1:]), 3) if len(steps) > 1 else None,
        "total_step_seconds": round(sum(steps), 3) if steps else None,
        "drafter_trained_metrics": int(sum(trained)),
        "drafter_published_metrics": int(sum(published)),
        "public_load_events": len(public_loads),
        "retention_checks": len(retention),
        "retention_passed": retained,
        "retention_failed": len(retention) - retained,
        "retention_by_rank": {},
    }
    for rank, kept, _expected, _actual in retention:
        entry = summary["retention_by_rank"].setdefault(rank, {"passed": 0, "failed": 0})
        entry["passed" if kept == "True" else "failed"] += 1
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
