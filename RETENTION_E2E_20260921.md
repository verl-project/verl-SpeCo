# SpeCo C5 retention audit on current upstream main

Author: 0z5a
Date: 2026-09-21
Host: one RTX 5090 node (8x RTX 5090 32 GB), GPUs 1 and 3 only; GPUs 0 and 4-7 belong to other
workloads on the shared machine and were never touched. No process was killed.

## Why this run exists

`CONTINUE_20260921.md` leaves one open item in the C5 scope: "fix drafter retention across target
synchronization, then rerun both fresh 20-step cases with the same audit. A valid fix requires
38/38 retained checks per algorithm, steps 1-20, successful all-rank loads and exit 0."

The previous C5 worktree was based on `cf297c4`. That revision carried a work-around in
`SpecoVLLMColocateWorkerExtension.update_weights_from_ipc` that reloaded the drafter from its
startup checkpoint after every target weight sync. That reload is what the audit caught: the
public draft load completed, and 14 seconds later the drafter fingerprint had reverted to the
checkpoint values (`latest_draft_retained=False`, 38/38 checks).

Upstream `main` has since replaced that work-around with a level-2 snapshot lifecycle (PR #72 and
follow-ups): the extension snapshots every draft parameter and buffer to host memory before a
level-2 sleep, restores them after the matching wake-up with an exact revision check, and refuses
to roll an online revision back to the checkpoint. This run verifies that mechanism end to end
instead of assuming it.

## What was run

Source: `verl-project/verl-SpeCo` `origin/main` at `18dd709`, clean checkout (no production edits).
Two adaptation overlays were applied, both recorded with SHA-256 manifests:

| Overlay | Change | Manifest |
|---|---|---|
| `padding-overlay` | Redirect only the `flash_attn.bert_padding` import in `verl/utils/attention_utils.py` to a pure-PyTorch equivalent (the RTX 5090 host has no `flash_attn`), and alias vLLM 0.29's relocated `MoERunner` as `FusedMoE` for `verl/utils/vllm/vllm_fp8_utils.py` | `src/current/padding-overlay/manifest.json` |
| `src/probe-overlay` | Audit only: fingerprint the rollout drafter after each committed online draft update and again after each target sync. Loader logic unchanged | `src/current/src/probe-overlay/manifest.json` |

The padding replacement was validated directly: `unpad_input`/`pad_input` round-trip is exact,
and `verl.workers.utils.padding.left_right_2_no_padding` returns exactly `attention_mask.sum()`
rows with identical content.

Fixture: synthetic arithmetic RL data (16 train / 4 val rows, 128 prompt / 64 response tokens),
`Qwen/Qwen3-4B` target with `AngelSlim/Qwen3-4B_eagle3` (EAGLE3) and `z-lab/Qwen3-4B-DFlash-b16`
(DFLASH) drafters, GRPO, tensor parallel 2, one drafter train and publish per step, 20 steps.

Environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, vLLM 0.29.0, Transformers 5.10.4,
driver 580.82.07, RTX 5090.

## Results

| Algorithm | Steps logged | Exit | Drafter trained | Drafter published | Retention checks | Retained |
|---|---:|---:|---:|---:|---:|---:|
| EAGLE3 | 20 | 0 | 20 | 20 | 38 | **38/38** |
| DFLASH | 20 | 0 | 0 | 0 | (not reached) | (not reached) |

| Algorithm | First step | Steady mean (steps 2-20) | Total logged step time |
|---|---:|---:|---:|
| EAGLE3 | 71.002 s | 29.747 s | 636.190 s |
| DFLASH (drafter training failed on every step) | 59.5 s | 30.662 s | n/a |

Per-rank retention detail for EAGLE3: rank 0 19/19, rank 1 19/19. Every check compares the
fingerprint of the drafter's published `fc` projection captured at commit time with the value read
back after the target sync. 40 public draft loads were observed; 12 distinct published
fingerprints appear across the run, so the checks are not comparing an unchanging drafter.

The C5 retention defect is therefore **not reproducible on current `main`** for EAGLE3, and the
fix is the upstream level-2 snapshot lifecycle, not a local patch. No production change is needed
for this item.

## DFLASH is blocked by a different, reproducible failure

DFLASH did not exercise the retention path at all. On every one of the 20 steps the drafter
training batch failed before any optimizer step:

```text
ValueError: DFlash input/hidden/mask row mismatch: input_rows=34, hidden_rows=33, mask_rows=34
  at verl_speco/backends/dflash_trainer_backend.py:1260 (preprocess_individual_items)
```

Consequences visible in the metrics: `drafter/trained:0.0`, `drafter/published:0.0`,
`drafter/train_no_trainable_batch:1.0`, `speco draft wake_up ... (revision=0)`, and the log never
contains `SPECO committed online drafter revision=`. The rollout served the startup drafter for
all 20 steps, so a DFLASH retention audit would be vacuous even if the checks were emitted.

The row counts prove this is not the padding shim: the shim produces exact row counts on the
same `left_right_2_no_padding` path. The 34-vs-33 mismatch is between captured hidden-state rows
and input/mask rows inside the DFLASH feature path. This is a separate defect from the C5
retention item and needs its own root cause before DFLASH can be reported on `main`.

## Evidence

| Artifact | Location |
|---|---|
| EAGLE3 20-step log | `evidence/full-e2e/online-eagle-20260921-041402.log` |
| DFLASH 20-step log | `evidence/full-e2e/online-dflash-20260921-041402.log` |
| Exit files | `evidence/full-e2e/online-{eagle,dflash}-20260921-041402.exit` |
| Summaries | `evidence/full-e2e/{eagle,dflash}-retention-summary.json` |
| Runners and overlays | `src/current/experiment/l40s/`, `src/current/padding-overlay/`, `src/current/src/probe-overlay/` |
| Local copy | `0z5a/speco-retention-e2e/evidence/rtx5090-20260921/` |

Raw logs stay on the host under `~/0z5a-work/speco-retention/evidence/full-e2e/`.

## Scope limits

These are single 20-step observations on a shared 8-GPU node using GPUs 1 and 3, not production
throughput measurements and not a performance comparison. EAGLE3's 29.747 s steady mean is a
shared-host observation. The audit proves retention of the newly trained drafter across target
synchronization for this fixture; it does not validate drafter quality, longer runs, other
speculative algorithms, or multi-node setups.

NFS was not touched, the machine was not restarted, and no process belonging to another workload
was stopped.
