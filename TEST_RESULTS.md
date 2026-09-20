# P-EAGLE partition lifecycle results

Submitter: 0z5a. Upstream: `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`.
This branch rebases the existing partition implementation onto the shared
P-EAGLE checkpoint fixes from `98ace9d`. Both arms select `engine=fsdp`;
VeOmni wrapping is never enabled in this comparison.

## Speed comparison

The complete standalone workflow ran in A0 → P0 → P1 → A1 order using fresh
processes, identical eight frozen target forwards, per-rank COD seeds, six
optimizer steps and checkpoints at steps 3 and 6. Times include startup,
materialization, training, checkpoint writes and process shutdown.

| Measurement | Flat baseline | Two partitions | Speed improvement |
|---|---:|---:|---:|
| Complete standalone, first launch | 18.587 s | 23.261 s | −20.09% raw ratio |
| Complete standalone, second launch | 23.780 s | 23.044 s | +3.19% raw ratio |
| Complete standalone, geometric mean | 21.024 s | 23.152 s | **−9.19% observed**, inconclusive magnitude |
| Prior full optimizer step, 256 tokens | 322.22 ms | 564.92 ms | −42.96% |
| Prior full optimizer step, 1024 tokens | 551.88 ms | 886.93 ms | −37.78% |

Speed improvement = `(baseline / candidate − 1) × 100%`. The new standalone
fixture is a 4-layer/64-hidden Llama target and 2-layer P-EAGLE, 33–40 input
tokens, vocabulary 256, BF16, two shared L20s. All four launches completed
6/6 updates and saved both checkpoints without traceback. Baseline wall time
varied 27.94%; one quartet does not establish a reliable speedup interval.
This is standalone training E2E, not the full RL actor/rollout pipeline.

The previous 1024-token test used the real Qwen3-4B target and measured memory
falling from 21.05 to 16.28 GiB (**22.69% less**) at the cost of slower training.
Those prior numbers retain their original evidence and scope; they are not
combined with the tiny-model runs. Partitioning remains opt-in and is a
memory/compute tradeoff, not an asynchronous efficiency improvement.

## Correctness and audit

- Current partition and P-EAGLE regression suites: **44 passed**.
- Prior fixed-COD FP32 CUDA matrix: 25 passed; prior two-rank FSDP2 gradient and
  SGD update oracle passed. BF16 quality/trajectory equivalence is not claimed.
- Eight repository sanity checks pass; no new `Any`, `getattr`, or exception
  handlers were introduced by the partition implementation.
- Actual commands: `experiment/l20/run_partition_ab.sh`; raw logs, exit status,
  per-launch timing and `summary.json` are in `evidence/l20-20260920/`.
- Full RL partition integration and multi-card capacity performance remain
  unverified. The old optimizer-step benchmarks are not relabeled as RL E2E.

## Context-only vocabulary head pruning — 2026-09-20

The loss previously ran the vocabulary head and KL on every retained context
position in each partition, then multiplied ignored positions by zero. The
change selects supervised positions after attention and before the vocabulary
head. Context attention and its gradients remain intact. This also reduces work
for masked positions in the flat path; the measurements below keep the old flat
implementation as the baseline.

All 44 existing P-EAGLE/partition checks passed. Two added tests compare the
pruned loss and every trainable gradient against the original full-head masked
loss, including an entirely masked batch and reduced draft vocabulary. Both
passed. Two-rank CUDA FSDP2 gradient and optimizer-step parity also passed.

A fresh-process A0 → O0 → P0 → P1 → O1 → A1 run used the same pinned Qwen3-4B,
1,024 input tokens, target-feature collection, COD seeds, eight optimizer steps
(three warmups), BF16 and one shared L20. A = old flat, O = old two-partition,
P = fixed two-partition. Input hashes and sampled token counts match in all six
runs. The source baseline is `9be33cc`.

| Optimizer-step comparison | Baseline | Fixed partition | Speed improvement |
|---|---:|---:|---:|
| Old partition, first independent run | 2,048.93 ms | 1,685.15 ms | +21.59% |
| Old partition, second independent run | 1,994.21 ms | 1,699.68 ms | +17.33% |
| Old partition, geometric mean | 2,021.38 ms | 1,692.40 ms | **+19.44%** |
| Old flat, geometric mean | 1,317.97 ms | 1,692.40 ms | **−22.12%** |
| Peak allocated GPU memory vs old partition | 16.28 GiB | 14.23 GiB | N/A: **12.56% less memory** |
| Peak allocated GPU memory vs old flat | 21.05 GiB | 14.23 GiB | N/A: **32.40% less memory** |

Speed improvement is `(baseline / candidate − 1) × 100%`. Thus the fixed
partition still takes 28.41% longer than flat; the remaining recomputation
tradeoff is not removed. Flat means differed by 0.15%, old partition by 2.74%,
and fixed partition by 0.86%. The old partition hit allocator retries during
both measured runs; its complete optimizer-step costs are included. These are
shared-node observations, not a dedicated-node confidence interval. Package
installation was active elsewhere on the host. BF16 multi-step trajectories
are not claimed bitwise identical; FP32 parity is covered separately.

Raw samples, source hashes, logs and summary are under
`evidence/l20-20260920/head-pruning/c2-head-pruning/`. Reproduction:
`experiment/l20/run_partition_head_ab.sh`. These timings cover complete drafter
optimizer steps after real target feature collection, not online RL E2E.

The complete two-rank standalone workflow was also repeated in A0 → P0 → P1 →
A1 order, where A uses the old two-partition code and P uses the fix. Each run
completed six optimizer steps and saved steps 3 and 6, with identical frozen
features and rank-local COD seeds.

| Complete standalone lifecycle | Old partition | Fixed partition | Speed improvement |
|---|---:|---:|---:|
| First launch | 81.773 s | 89.698 s | −8.84% raw |
| Second launch | 55.847 s | 49.619 s | +12.55% raw |
| Geometric mean | 67.578 s | 66.714 s | +1.30% raw; **inconclusive** |

Large startup/checkpoint I/O variation dominates this tiny-model lifecycle:
the old-arm times differ by 46.42% and fixed-arm times by 80.77%. No reliable
complete-E2E speedup follows from this quartet. The checkpoint from fixed P0 is
used by the subsequent C1 native serving validation. The optimizer-step gains
above are not relabeled as full RL or standalone lifecycle gains.

After these runs, 23 completed-task model/optimizer weight files were removed,
freeing 8,051,385,896 bytes. Hashes and paths are in `cleanup.json`. The small
checkpoint copy still in active C1 validation is retained until that test ends.
