# Exploratory OPD co-training comparison

Last updated: October 28, 2026.

This directory preserves an auditable summary of the two supplied 200-step runs:
OPD with a fixed DSpark drafter and OPD with online drafter training. The student
is Qwen3-8B, the teacher is Qwen3-32B, and the data combines GSM8K and MATH.
Student training uses FSDP2; student rollout and teacher inference use vLLM.
The recorded allocation is four student training/rollout GPUs and two teacher GPUs.

## Evidence and limits

There is **one run per condition**, with no repeated-seed evaluation or confidence
interval. The 200 training steps are correlated observations, not 200 independent
experiments. Training reward is not held-out evaluation accuracy. The observed
reward difference of +0.0078125 does not establish an improvement beyond random
variation; its statistical significance is unknown. No multi-seed results are
available, and this directory does not claim otherwise.

Values below are arithmetic means of steps 1–200. Full precision and per-metric
counts are in [summary.json](summary.json). The fixed-drafter training timer was
not logged; absence is not treated as zero.

| Metric | Fixed drafter | Online co-training |
| --- | ---: | ---: |
| Mean acceptance length | 2.959672 | 3.170638 |
| Training reward mean | 0.407344 | 0.415156 |
| Time per step (s) | 56.363171 | 67.794074 |
| Generation time (s) | 17.887778 | 17.712187 |
| Drafter training time (s) | Not logged | 7.030057 |
| Actor distillation absolute loss | 0.216656 | 0.216678 |
| Actor policy-gradient loss | -0.331649 | -0.337015 |

For this pair, mean acceptance length is 7.13% higher, generation time is 0.98%
lower, and total step time is 20.28% higher. The approximately 0.18 s generation
saving does not offset the approximately 11.43 s increase in total step time.
Drafter training accounts for a logged 7.03 s per step; the remaining overhead
cannot be attributed precisely from these aggregate timers. These observations
do not demonstrate an end-to-end speedup or establish that the overhead is worthwhile.

Both logs reach step 200, then report a killed DataLoader worker during shutdown.
The co-training run also reports a Ray/OpenCensus segmentation fault after the
final step. The summary includes the logged training steps, not a clean-shutdown claim.

## Audit the historical results

The original logs and launch scripts are Git objects in this repository's history
at `eed8c3a9f1fb40f82fb5e7d0e84996389cd14ef8`. They are not duplicated here.
[sha256.json](sha256.json) records their original byte hashes. No fork-hosted raw
URLs or plotting dependencies are needed to recompute the table.

From a full checkout of this branch, run:

```bash
uv run --no-project python docs/experiments/opd-cotrain/20261013_005359/summarize_results.py --check
```

The script reads both logs using `git show`, verifies their SHA-256 hashes,
requires all 200 steps and required metrics, and compares the result with the
checked-in summary. A shallow checkout must first fetch enough history to include
the source commit, for example `git fetch --unshallow`.

The original launch scripts and resolved configurations can also be inspected:

```bash
git show eed8c3a:docs/experiments/opd-cotrain/20261013_005359/run_opd_cotrain_dspark_vllm_fsdp2.sh
git show eed8c3a:docs/experiments/opd-cotrain/20261013_005359/run_opd_cotrain_dspark_vllm_fsdp2_no_training.sh
```

## Launch new comparisons

The maintained [8B launch example](../../../../examples/run_qwen3-8b_opd_actor_fsdp2_drafter_dspark_vllm.sh)
uses placeholder model/data paths and explicitly selects synchronous execution.
It reflects the principal historical settings; current code, runtime dependencies,
random state and hardware can change results. It is not a bit-for-bit replay.

After installing the documented GPU runtime, set `STUDENT_MODEL`, `TEACHER_MODEL`,
`DRAFTER_MODEL`, `TRAIN_FILES` and `VAL_FILES` as described in the example.
[reproduce.sh](reproduce.sh) launches paired fixed/online conditions, with separate
output directories and checkpoint resumption disabled:

```bash
SEEDS="42 43 44" OUTPUT_DIR=/path/to/new-runs \
  bash docs/experiments/opd-cotrain/20261013_005359/reproduce.sh
```

These are proposed new data-shuffle and actor-data-loader seeds, not measured
replicates or recovered historical seeds. Other backend random generators and
hardware nondeterminism are not controlled by this wrapper. Each condition must
start from the same initial student/drafter checkpoints and use the same runtime
versions and hardware. Record those versions and resolved configs with new results.
Additional example overrides can be passed to the wrapper; it fixes the paired
condition, shuffle seeds, output directory and resume mode after those overrides.

For each new pair, recompute its summary from the saved logs:

```bash
uv run --no-project python docs/experiments/opd-cotrain/20261013_005359/summarize_results.py \
  --fixed-log /path/to/new-runs/seed-42/fixed_drafter/console.log \
  --co-train-log /path/to/new-runs/seed-42/co_train/console.log
```

Use independent paired runs and a documented analysis of run-level differences
before claiming repeatable reward or throughput improvements. Do not derive a
seed-level confidence interval by treating training steps as independent samples.

## Student-only feature boundary

The supported OPD co-training path validates synchronous FSDP2 execution and
collects features during the student's old-logprob forward. Teacher supervision
fields are excluded from that forward, and drafter samples are constructed from
explicit student fields. Configuration validation alone is not a payload guarantee.

The worker rejects any sample containing `teacher_ids` or `teacher_logprobs`, in
all modes that use its collection commit path. It raises rather than silently
dropping the sample. If earlier samples in that collection were already stored,
the scheduler's transaction path rolls back the collection and propagates the
error. The failing worker does not return its final `collection_profile`; the
exception and any enabled outer phase diagnostics remain available.

Directly calling the private commit helper bypasses transaction orchestration.
The boundary checks known top-level supervision keys, not the semantic provenance
of arbitrary tensors, nested payloads, or future upstream field names. Custom
producers must preserve the student-only contract; no claim is made that arbitrary
external callers can never submit teacher-derived data.
