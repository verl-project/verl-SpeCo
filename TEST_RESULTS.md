# Frozen P-EAGLE serving — 2026-09-20

Submitter: 0z5a. Base: upstream main `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`.

The converter exports a full-vocabulary SpeCo P-EAGLE checkpoint to the configuration consumed by vLLM 0.29.0's EAGLE3 implementation with `parallel_drafting=True`. Weights are copied without transformation. The existing online P-EAGLE publishing guard is unchanged.

## Complete tested path

The checkpoint came from the actual two-rank VeOmni six-step run (`veomni-P1/draft_step_6`), then underwent conversion, loading, speculative generation, and a reference-model logits comparison. L20, TP1, BF16, eager mode; tiny target: four layers, hidden size 64, vocabulary 256; draft: two layers. This establishes a small frozen-model lifecycle, not 4B online RL, CUDA Graph, or TP2/TP4 support.

| Check | Result |
|---|---|
| Loaded logical tensors vs exported training checkpoint | 25/25 exact |
| Greedy output vs target-only | Both prompts, all 16 tokens identical |
| Cached draft forward vs original model | 32/32 pass |
| Maximum absolute logits error | 0.00390625 |
| Logits argmax | 32/32 identical |
| Masked tokens, padding, rejected suffix / KV reuse | Included in replay oracle |
| Process exit | 0 |
| Converter unit tests | 7 passed |
| Mypy / Ruff / repository sanity | Passed / passed / 8 passed |

The oracle captures inputs before the runtime mutates residual tensors, reconstructs the causal prefix at each cache position, excludes padding via slot mapping, and checks logits at atol 0.01 / rtol 0.02. Earlier post-forward captures were invalid evidence; the committed pre-hook captures replace them. Projection weights are verified separately by exact tensor comparison.

## Speed comparison

| Workload | Baseline | Candidate | Speed improvement |
|---|---:|---:|---|
| Two prompts × 16 tokens | 32 identical tokens | 32 identical tokens | N/A: correctness instrumentation; no controlled timing |

No speedup claim is made. The relevant training performance comparisons are in the separate VeOmni and partition branches. Raw output, logits checks, forward captures, exit code, and sanity results are in `evidence/l20-20260920/`.

## Reproduction

Use the pinned vLLM 0.29.0 / PyTorch 2.13.0+cu130 environment recorded by the parent L20 validation. Generate the tiny target/draft with the parent `prepare_tiny_peagle.py`, train via the VeOmni branch's `run_veomni_ab.sh`, then:

```bash
python -m verl_speco.convert_peagle_vllm \
  /experiment/evidence/l20-20260919/veomni-P1/draft_step_6 \
  /experiment/tiny-peagle/trained-serving --target-layer-ids 0 1 2
python experiment/l20/check_tiny_peagle_serving.py draft-eagle3 \
  --draft-model /experiment/tiny-peagle/trained-serving \
  --reference-model /experiment/evidence/l20-20260919/veomni-P1/draft_step_6 \
  --output-prefix trained-peagle
python experiment/l20/check_peagle_decode_logits.py \
  --prefix trained-peagle \
  --reference /experiment/evidence/l20-20260919/veomni-P1/draft_step_6
```

Run with `CUDA_VISIBLE_DEVICES=0`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on the isolated test container. The converter requires ordered target feature IDs, full vocabulary, and safetensors. Reduced vocabulary and other runtime versions have not been validated. Completed model weights were removed after evidence capture at the user's request; regeneration is required to rerun.

The raw serving log retains vLLM’s chat-template warmup warning (`skip_tokenizer_init=True`); token-ID generation and numerical checks completed successfully. Raw logs are preserved byte-for-byte, including their original whitespace.

## Continued validation: CUDA Graph (2026-09-20)

The deterministic original tiny fixture was regenerated after cleanup. This run uses the original fixture, rather than the previous six-step trained checkpoint. Both arms use vLLM V1 runner, TP1, BF16, 128 MiB KV cache, CUDA Graph capture sizes `[1, 2, 4, 8]`, and the same two prompts, generated twice in the same engine. Source and runtime remain vLLM 0.29.0 with the unmodified public loader.

| Complete generation correctness | Target-only | Frozen P-EAGLE | Speed improvement |
|---|---:|---:|---|
| Output tokens, two repeated rounds | 64 | 64 identical | N/A: instrumentation and shared GPUs |
| Actual CUDA graph replay calls | 80 | 272 | Not a timing metric |
| Captured CUDA graphs | 21 | 17 | Not a timing metric |
| Loaded logical tensors | N/A | 25/25 exact | N/A |
| Process exit | 0 | 0 | N/A |

Every graph-mode output also matches the earlier eager target-only token IDs. Replay calls were counted on real `torch.cuda.CUDAGraph.replay` invocations after capture. Baseline roles: `LlamaForCausalLM=60`, `PiecewiseBackend=20`; speculative roles: `LlamaForCausalLM=60`, `PiecewiseBackend=212`. Piecewise entries are not individually attributed to draft layers; these counts establish graph execution in the complete speculative engine, not a separate draft-only logits oracle under graph capture.

TP2 was also attempted. The target-only baseline stalled inside FlashAttention; switching off custom all-reduce and switching V2 to V1 did not resolve it. One separate startup failure was caused by insufficient free memory and is retained as such. Worker stack captures and raw failures are retained. TP2 parameter-shard assertions were added to the driver but have not yet passed on a live TP2 draft. This is not evidence of a P-EAGLE regression, because the failing arm does not load a drafter.

Evidence for this continuation is in `evidence/l20-20260920/continuation/`. `run_serving_matrix.sh` specifies the attempted matrix; it stops on failure so worker cleanup precedes any retry. The successful single-card graph runs were dispatched separately after cleaning the stalled TP2 workers.

A final TP2 V1/spawn attempt also stalled in target-only generation and reached its 240-second timeout (exit 124). Its remaining engine/worker processes were explicitly stopped before cleanup. Spawn did not resolve the failure; no TP2 draft result was obtained.

## Native latest-release validation — 2026-09-20

The new L20 native environment uses vLLM 0.29.0, PyTorch 2.13.0+cu130,
Transformers 5.17.0 and Python 3.13.14. The wheel SHA-256 matches official PyPI
metadata. The existing native 0.18 installation is unchanged. First startup
included a FlashInfer sampling-kernel build; compilation and per-forward capture
I/O are excluded from any speedup claim.

This checkpoint is from the C2 fix's actual two-rank FSDP2 standalone run,
`c2-pruned-P0/draft_step_6` (six optimizer updates). It was copied before completed
training weights were cleaned, then exported with the converter. TP1 BF16/eager
loads all 25 logical tensors exactly; both prompts generated twice match the
target-only reference. All 64 cached draft-forward comparisons pass, with a
maximum logits error of 0.00390625. This is a tiny frozen-serving E2E, not an
online RL or quality result.

The native target-only TP2 run reproduced the earlier FlashAttention 2 stall
at the first generation. Its worker stack is inside `flash_attn_varlen_func`;
no draft model is loaded in that failed arm. The run was explicitly terminated
and its workers cleaned. A subsequent run explicitly selected `FLEX_ATTENTION` and also failed.
The reproduction recipe retains FlashAttention for TP2 and records backend
selection; it does not imply the full matrix passed.

The native TP1 graph run uses this same six-step checkpoint. Both eager and
graph arms produce the same two rounds of token IDs, and graph mode again checks
all 25 logical tensors against the original checkpoint. Converter unit tests:
7 passed.

| Native C1 check | Target-only | Frozen trained P-EAGLE | Speed improvement |
|---|---:|---:|---|
| TP1 eager, generated tokens across two rounds | 64 | 64, identical | N/A: correctness capture |
| TP1 graph, generated tokens across two rounds | 64 | 64, identical | N/A: shared GPU correctness run |
| Actual graph replay calls | 80 | 272 | Not a timing metric |
| Cached draft logits comparisons | N/A | 64/64 pass; max error 0.00390625 | N/A |
| TP1 logical parameters, eager and graph | N/A | 25/25 exact in each mode | N/A |

Native TP2 diagnostics preserve the target-only failures: FlexAttention reached
its 300-second executor RPC timeout; disabling NCCL P2P, disabling async
scheduling, and moving to physical GPUs 2/3 did not unblock first generation.
The FlexAttention native stack waits in `cudaStreamSynchronize`; the FlashAttention
native stack waits in `cuLaunchKernel`. These observations identify blocking
locations, not a proven root cause. No TP2 draft result or TP2 speedup is claimed.

Enabling custom all-reduce and setting `CUDA_MODULE_LOADING=EAGER` (with and
without synchronous scheduling) also failed to produce the first token. Eager
module loading moved the sampled blocking location to output synchronization;
the synchronous variant waited in `_bookkeeping_sync` / `_to_list`. These bounded
attempts were explicitly stopped, with worker stacks retained. Only the
FlexAttention arm reached its own RPC timeout; other stops are not reported as
completed benchmark timings.

`evidence/l20-20260920/native-latest/matrix-summary.json` records the partial
matrix: TP1 eager/graph pass; TP2 baseline not passed; TP2 draft/graph not run.
All task-owned workers were stopped. Three completed tiny checkpoint weights
were deleted (1,757,048 bytes), with SHA-256 and paths in `cleanup.json`. Raw
forward captures remain as numerical evidence. Ruff check/format and shell
syntax checks pass; the seven converter unit tests pass. The unit tests used
the existing container; all native serving results above used the new native
environment.
