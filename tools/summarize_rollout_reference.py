"""Summarize PPO step counters and the last visible export snapshot per worker."""

import argparse
import ast
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    stages = {
        "gen": "timing_s/gen",
        "old_logprob": "timing_s/old_log_prob",
        "actor": "timing_s/update_actor",
        "step_s": "timing_s/step",
        "ref": "drafter/rollout_reference_committed",
        "inline": "drafter/rollout_inline_committed",
        "ready": "drafter/reference_ready_at_boundary",
        "waiting": "drafter/reference_waiting_at_boundary",
        "sync_n": "drafter/reference_sync_resolves",
        "sync_s": "timing_s/reference_sync_resolve",
        "future_s": "timing_s/reference_future_wait",
        "tail_s": "timing_s/drafter_reference_collection_tail",
        "begin_s": "timing_s/reference_begin_receipts",
        "dispatch_s": "timing_s/reference_collect_dispatch",
        "finish_s": "timing_s/reference_finish_receipts",
        "ack_submit_s": "timing_s/reference_ack_submit",
        "tokens": "perf/total_num_tokens",
        "accept_len": "drafter/spec_decode/mean_acceptance_length",
    }
    workers = {}
    print("step", *stages, sep="\t")
    with args.log.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            step = re.search(r"\bstep:(\d+) - ", line)
            if step:
                fields = dict(re.findall(r"([\w/]+):([-+.\deE]+)(?= - |\s*$)", line))
                print(
                    step[1],
                    *(fields.get(key, "-") for key in stages.values()),
                    sep="\t",
                )
            snapshot = re.search(
                r"Worker pid=(\d+).*?step=\('([^']*)', (\d+)\) process_totals=(\{[^}]*\})",
                line,
            )
            if snapshot:
                workers[(snapshot[2], int(snapshot[3]), int(snapshot[1]))] = (
                    ast.literal_eval(snapshot[4])
                )
    if workers:
        print(
            "\nLast visible cumulative snapshot per run, step and worker (overlapping work seconds):"
        )
        for key, totals in sorted(workers.items()):
            print(*key, totals)


if __name__ == "__main__":
    main()
