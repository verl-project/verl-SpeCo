"""Summarize the fixed-work standalone quartet without treating steps as runs."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("evidence", type=Path)
args = parser.parse_args()
rows = []
for arm in ("A0", "P0", "P1", "A1"):
    path = args.evidence / f"veomni-{arm}.log"
    log = path.read_text()
    assert "Traceback" not in log
    result = ast.literal_eval(
        log.split("Standalone SPECO draft training finished: ")[-1].strip()
    )
    assert result["successful_steps"] == result["optimizer_steps_total"] == 6
    assert log.count("cleanup complete") == 2
    times = [
        float(value)
        for value in re.findall(
            r"\[standalone drafter metrics\].*?step_time=([\d.]+)s", log
        )
    ]
    assert len(times) == 6
    rows.append(
        {
            "arm": arm,
            "engine": "fsdp" if arm.startswith("A") else "veomni",
            "wall_seconds": float(
                (args.evidence / f"veomni-{arm}.seconds").read_text()
            ),
            "step_seconds": times,
            "successful_steps": 6,
            "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
baseline = math.sqrt(rows[0]["wall_seconds"] * rows[3]["wall_seconds"])
candidate = math.sqrt(rows[1]["wall_seconds"] * rows[2]["wall_seconds"])
report = {
    "scope": "tiny dense P-EAGLE, standalone feature replay, two L20s, six steps and checkpoints; not RL",
    "order": "A0 P0 P1 A1; independent processes",
    "rows": rows,
    "baseline_geomean_seconds": baseline,
    "candidate_geomean_seconds": candidate,
    "observed_speedup_percent": 100 * (baseline / candidate - 1),
    "baseline_max_min_ratio": rows[0]["wall_seconds"] / rows[3]["wall_seconds"],
    "conclusion": "INCONCLUSIVE: one quartet on shared GPUs; no stable speedup claim",
}
(args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
