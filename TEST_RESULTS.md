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
python experiment/l20/check_tiny_peagle_serving.py \
  --draft-model /experiment/tiny-peagle/trained-serving \
  --reference-model /experiment/evidence/l20-20260919/veomni-P1/draft_step_6 \
  --output-prefix trained-peagle
python experiment/l20/check_peagle_decode_logits.py \
  --prefix trained-peagle \
  --reference /experiment/evidence/l20-20260919/veomni-P1/draft_step_6
```

Run with `CUDA_VISIBLE_DEVICES=0`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on the isolated test container. The converter requires ordered target feature IDs, full vocabulary, and safetensors. Reduced vocabulary and other runtime versions have not been validated. Completed model weights were removed after evidence capture at the user's request; regeneration is required to rerun.
