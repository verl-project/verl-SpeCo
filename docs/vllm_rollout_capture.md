# vLLM rollout hidden-state collection

Last updated: 2026-11-05

This experimental path captures target features during the existing RL rollout
forward. It does not replay the target model and does not replace the real
EAGLE3/DFlash drafter with `method=extract_hidden_states`.

## Supported configuration

- vLLM **0.22.0**, V1 GPUModelRunner (not model runner V2).
- CUDA, text-only Llama/Qwen2/Qwen3 targets with auxiliary hidden outputs.
- EAGLE3 or DFLASH; TP=1, PP=1, engine DP=1; no microbatching.
- Legacy SpeCo PPO trainer. The verl V1 trainer currently rejects this source.
- Synchronous scheduling, prefix caching disabled. Chunked prefill is allowed;
  captured positions are derived from each request's computed-token offset.
- `use_logits=false`; front/random bounded response windows. Prompt preservation
  and auxiliary selection of the unnormalized final-layer output are unsupported.

For an existing EAGLE3/DFlash vLLM example, apply these Hydra overrides:

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.enable_prefix_caching=false \
+actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false \
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_vllm=true \
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false \
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=false \
actor_rollout_ref.rollout.drafter.training.use_logits=false \
actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=64
```

Use `++` instead of `+` if `async_scheduling` is already present in the selected
example. Existing interval, sampling-rate and per-replica sample/token budget
settings gate collection. Validation/probe requests carry the existing skip hint.
Target weights must remain fixed during an individual generation request, as in
the supported synchronous rollout lifecycle.

## Data flow and alignment

The server adds a private `speco_rollout_capture` plan to
`SamplingParams.extra_args`. The worker extension installs runner hooks before
model loading. `sample_tokens` retains its ephemeral forward state and `_sample`
captures rows after sampling, before proposer execution can reuse the buffers.

For a request with K scheduled draft inputs, N valid sampled outputs imply N-1
accepted draft inputs. The retained input prefix has
`scheduled_tokens - K + N - 1` rows. A total rejection still retains the base
input row that predicts the replacement token. Prefill input rows remain valid
even when their sampled output is discarded for an incomplete prefill.

Window coordinates are absolute input positions. The first response prediction
uses position `prompt_length - 1`. The last eligible position is
`prompt_length + final_response_length - 2`; the final generated token's hidden
is not required. EOS/stop truncation is applied again at frontend assembly.

The bridge snapshots only intersecting windows. Acceptance counts stay on the
device and are copied asynchronously with features; the writer removes rejected
rows after its completion event. This avoids a capture-induced sampler
`cpu().tolist()` synchronization. A few rejected candidate rows inside the
window can cross D2H, but are never published as training rows.

The bridge saves CPU-only, windowed chunks under the same-node temporary
directory `verl-speco-rollout-capture/<sample-id>/`. Files are committed by
atomic rename. The frontend checks identity, coverage, token IDs, layer metadata,
and layout, assembles the existing `drafter_sample`, and removes temporary files.
Conflicting recomputed rows fail closed. Missing features do not silently fall
back to replay. Short samples below `min_rows` are skipped.

`drafter_sample` enters the existing rollout collection adapter, transactional
worker-buffer ingestion and drafter training. If configured, the existing event
bus carries a Ray ObjectRef for early consumption; the final rollout output
retains the existing fallback/deduplication contract. Independent-training
producer model calls are not used.

## Performance and validation limits

CUDA capture uses owned snapshots on the forward stream, pinned host buffers,
and a dedicated copy stream. A shared `FeatureExportQueue` waits on completion
events and writes files on background threads; the runner does not wait for
those copies or writes. The frontend waits for pending exports before assembling
a sample. SGLang and vLLM share `rollout_feature_fields` for the base sample
contract. Neither helper executes a model or invokes the independent producer.

Each worker limits retained export buffers to 256 MiB and 16 outstanding chunks,
with two writer threads. Reservations include device snapshots, pinned buffers
and compact writer payloads. Capacity exhaustion explicitly skips the whole
sample; partial samples are not trained. The output fields
`_speco_capture_skipped_capacity` and `_speco_capture_rows` expose this behavior.
The writer retains ownership until export completes, and failed exports are
reported to the frontend. No fallback reruns the target forward.

This avoids an additional target forward but is not a throughput guarantee:
GPU snapshot copies, CPU assembly, metadata filesystem operations and feature
files still cost time. CUDA performance and the new asynchronous lifetime
handling require hardware validation. Multi-rank transport remains future work.
Request completion/failure removes its capture directory. A hard process kill
can leave files in the temporary directory; remove abandoned run files only
after all engines using them have stopped.

## Relationship to vLLM PR #62411

[#62411](https://github.com/vllm-project/vllm/pull/62411) adds windowed file export
to `ExampleHiddenStatesConnector` with `method=extract_hidden_states`. It cannot
coexist with a real speculative drafter in that engine. SpeCo's direct path keeps
EAGLE3/DFlash active and does not enable that connector. Configuration explicitly
rejects mixing either extractor option into direct capture.

The two designs can be used by separate engines, but this implementation does
not require that PR. Applying its changes to a patched vLLM installation still
requires version-specific integration testing; clean patch application or runtime
compatibility has not been established here.

CPU tests exercise acceptance filtering, mixed-request offsets, buffer ownership,
window/stop clipping, token validation, cancellation, frontend lifecycle and
shared scheduler routing:

```bash
uv run python -m pytest tests/integration/test_vllm_rollout_capture.py -q
```

Before production use, run CUDA end-to-end validation against a same-weight
reference forward for every selected auxiliary layer and the post-norm final
state, with greedy/sampled decoding, full/partial rejection, EOS, preemption,
chunked prefill and CUDA graphs. Also complete a collect/train/publish cycle and
measure rollout throughput and end-to-end step time. CPU tests alone do not
establish this accelerator compatibility.
