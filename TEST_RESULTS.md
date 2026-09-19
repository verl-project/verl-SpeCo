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
