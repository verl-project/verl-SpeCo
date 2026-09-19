# VeOmni drafter L20 results

Submitter: 0z5a. Upstream baseline: `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`.
Implementation: `98ace9d`; `43691fa` only adds license headers.

The dedicated two-rank standalone path completes target-feature replay,
P-EAGLE training, model/optimizer checkpoint export, process shutdown, and
resume from optimizer step 6 to step 8. This is a tiny dense-model lifecycle
validation, not an online RL or large-model performance claim.

## Speed comparison

Four fresh processes, A0 → P0 → P1 → A1. Both engines use the same eight frozen
target forwards, rank-local random seeds, six optimizer steps, and checkpoints
at steps 3 and 6. Timing includes Python startup, model materialization,
training, checkpointing and process exit. Hardware: two shared L20s, BF16
parameters/FP32 reduction, verl 0.9.0, VeOmni 0.1.11, PyTorch 2.13.0+cu130.
Target: 4-layer Llama, hidden size 64, vocabulary 256; drafter: 2-layer P-EAGLE.

| Run | FSDP baseline | VeOmni | Speed comparison |
|---|---:|---:|---|
| First independent launch | 96.462 s | 21.617 s | Not interpretable: delayed baseline process exit |
| Second independent launch | 18.828 s | 19.585 s | −3.87% raw ratio; not a paired confirmation |
| Geometric mean | 42.617 s | 20.576 s | +107.12% raw ratio, **invalid as speed evidence** |

Speed ratio is `(baseline / candidate - 1) × 100%`. Baseline launches differ
by **5.12×**. Both completed six updates, but this timing block cannot support
a speedup claim. The earlier screening quartet gave +3.31% with 12.63%
baseline variation; it is preserved under `evidence/l20-20260920/screen/` and
is not substituted for final validation. No stable speed improvement is claimed.

## Correctness

| Check | Result |
|---|---|
| Production VeOmni adapter, FP32, two ranks | Three AdamW steps pass loss, all parameter gradients, clipping norm, optimizer moments and updated parameter comparisons (`atol=2e-5`, `rtol=2e-4`) |
| Materialization | Parameters and persistent/non-persistent buffers match exactly |
| Full standalone BF16 | 6/6 successful updates in all four launches, two checkpoints each, clean exit |
| Independent resume | Initial optimizer step 6, two further updates, exported step 8, clean exit |
| Focused CPU regressions | 61 passed, 1 dependency skip |
| Repository sanity checks | All eight enabled checks pass |
| Ruff / changed-helper mypy / whitespace | Pass |

The full RL integration, SP/EP, colocated actor use and other algorithms are
not enabled by this first adapter. Existing non-TQ resume does not restore the
feature cursor or COD RNG; no claim of exact uninterrupted-trajectory resume.
Full-repository type checking is not claimed.

Two P-EAGLE issues discovered during lifecycle validation are fixed: pretrained
export unwraps the training module, and checkpoint loading retains a trained
embedding instead of copying the target embedding over it. The same fixes are
present in both timing arms. The optional engine leaves default FSDP wrapping
unchanged and imports VeOmni only when selected. New production code adds no
`Any`, `getattr`, or exception handlers.

## Reproduction

In the pinned test container, bind the source at `/experiment/variants/veomni`
and the deterministic fixture at `/experiment/tiny-peagle`:

```bash
PYTHONPATH=/experiment/veomni-deps:/experiment/online-deps:/experiment/variants/veomni \
  OMP_NUM_THREADS=4 /experiment/.venv-clean/bin/python -m torch.distributed.run \
  --nproc-per-node=2 --master-port=29591 experiment/l20/check_veomni_drafter.py
bash experiment/l20/run_veomni_ab.sh
python experiment/l20/summarize_veomni.py evidence/l20-20260920
```

VeOmni staging weights are held in per-rank temporary directories and removed
immediately after materialization. Raw logs, exit status, per-launch timings,
source hashes and JUnit are stored in `evidence/l20-20260920/`.
