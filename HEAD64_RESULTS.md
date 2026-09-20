# Trained checkpoint serving on native vLLM 0.29

Author: 0z5a. Source before this harness update: `8e9427f`.

The two-layer draft was trained for six optimizer steps with the VeOmni engine
on two L20 ranks, then exported and converted. The target has four layers,
hidden size 256, head dimension 64, four query heads, two KV heads and vocabulary
256. The old head-dimension-16 fixture stalled at TP2 even without a draft;
real Qwen3-4B TP2 generation and this larger fixture both finish.

| Validation | TP1 eager | TP2 eager | TP1 graph | TP2 graph |
|---|---|---|---|---|
| Target-only versus speculative tokens, two rounds | Pass | Pass | Pass | Pass |
| Checkpoint logical tensors per rank | 25/25 exact | 25/25 exact on both ranks | 25/25 exact | 25/25 exact on both ranks |
| Actual graph replay | N/A | N/A | 80 baseline / 272 speculative | 80 baseline / 272 speculative on each rank |
| Independent cached draft-logit oracle | **Strict failure** | Not run | Not run | Not run |

All 64 captured TP1 calls meet the original `atol=0.01, rtol=0.02` numerical
tolerance, with maximum error 0.0078125. Eight calls fail the original exact
argmax gate. In every failing call, one side's highest two logits are tied;
the other side's top-two gap is 0.00390625 or 0.0078125. Compiling the reference
FlexAttention did not remove the failure. The gate has not been relaxed: the
diagnostic now evaluates every call and still exits nonzero for any mismatch.

The parameter oracle now derives packed Q/K/V and gate/up widths from the
checkpoint instead of assuming the old fixture dimensions. It checks each
rank's expected slice, not equality of different TP shards.

| Speed comparison | Baseline | Candidate | Speed change |
|---|---|---|---|
| Frozen generation | Two rounds × 32 tokens | Identical tokens | N/A: correctness instrumentation, no controlled timing |

Evidence: `evidence/l20-20260920/head64/serving/` and `head64/logits/`.
`run_head64_serving.sh` reproduces the original fail-fast test; the independent
remaining matrix was run after the strict oracle failure and is recorded
separately. The complete matrix does not override the failed numerical gate.

`check_trained_publish.py` additionally exercises a drained-engine checkpoint
change and a separate cold engine. Its in-place parallel-mask cache refresh is
test-side diagnosis, not a production online P-EAGLE implementation. Publication
results are now complete:

| Trained step 3 → step 6 publication | TP1 eager | TP2 eager | TP1 graph | TP2 graph |
|---|---|---|---|---|
| Hot-B versus independent cold-B tokens | Exact | Exact | Exact | Exact |
| Target hashes before/after and hot/cold | Unchanged | Unchanged on both ranks | Unchanged | Unchanged on both ranks |
| Hot-B versus cold-B draft logits | Max error 0 | Max error 0 on both ranks | Not independently captured | Not independently captured |
| Post-update logical checkpoint tensors | 25/25 | 25/25 on both ranks | 25/25 | 25/25 on both ranks |
| Graph replay after publication | N/A | N/A | Observed | Observed on both ranks |

The stale parallel-mask cache error before the test-side refresh is 0.0185546875;
the refresh preserves its allocation address. This confirms the cache dependency,
not a production online P-EAGLE adapter. No timing speedup is claimed.
The TP2 graph continuation and all-rank comparison exit 0; the earlier I/O
timeout is retained separately in `publish-before-timeout/`.

The six-step flat and sequence-partition checkpoints also each complete fresh
TP2 serving, checking 25 logical tensors on both ranks and matching target-only
tokens over two rounds. `partition-serving/suite.exit` is 0.

All completed fixture/export/checkpoint model weights were then removed:
27 files, 181,914,574 bytes. The manifest retains sizes and SHA-256 hashes;
forward/logit captures and raw logs remain. Real models required by C5 are kept.
