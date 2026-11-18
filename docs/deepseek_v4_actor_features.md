# DeepSeek V4 actor features with the Ascend training backend

`training.backend=vllm_ascend` selects drafter execution. Online features now come from the actor engine's old-logprob forward. Direct
rollout collection was removed on 2026-11-11; standalone trajectory replay and
the existing PyTorch collection paths remain available.

## Configuration

Under `actor_rollout_ref.rollout.drafter.training`, set:

```yaml
backend: vllm_ascend
mode: online
collect_hidden_states_from_rollout: false
collect_hidden_states_from_old_logprob: true
collect_hidden_states_from_sgl: false
old_logprob_hidden_capture_impl: forward_hook
use_logits: false
use_data_buffer: false
```

Keep the existing Ascend `vllm_ascend` execution/work directory settings and
DSpark/target checkpoint configuration. Actor strategies use SpeCo's existing
FSDP/FSDP2, VeOmni, or Megatron old-logprob adapter. This addition currently
requires actor PP=1; the existing PyTorch PP collection remains available.
The actor model must expose decoder layers with complete HC residuals; a model
that exposes only post-HC states is rejected. `output_hidden_states` cannot
replace the raw pre-HC hook. Existing Ascend rollout PP/DP constraints still apply.

Keep `collect_hidden_states_from_rollout=false`; enabling it is rejected.

### Transformers eager actor

The installed Transformers DeepSeek V4 implementation supports eager attention,
not FlashAttention/SDPA. For that actor implementation use the existing verl
configuration, without changing the actor forward:

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: false
    override_config:
      attn_implementation: eager
  actor:
    ulysses_sequence_parallel_size: 1
```

The external batch remains jagged (`DatasetPadMode.NO_PADDING`); verl's native
FSDP engine right-pads each sample and creates its attention mask. The DeepSeek
V4 selector maps unpadded sample positions into that padded decoder output and
excludes padding. This opt-in mapping applies only to the DeepSeek V4 feature
layout. Other SpeCo feature layouts retain their existing selection requirements.
Do not concatenate multiple samples for an eager attention model that cannot
interpret packed sequence boundaries.

## Data and lifecycle

For the target checkpoint's `dspark_target_layer_ids`, capture decoder outputs
and take their mean over HC, matching native Ascend DSpark auxiliary features.
Capture the final decoder output before HC head and final norm as the teacher
state. Hook snapshots own their storage before actor buffers can be reused.

Concatenate auxiliary and teacher features for the existing old-logprob
selection, sequence parallel merge, CPU/object-reference/chunk transport and
worker routing. Split them at the Ascend training boundary. Hidden positions
index the unpadded full token sequence: a random/middle window must select the
matching token IDs, and SpeCo mask values are 1 for response tokens and 0 for
prompt tokens. Preserve absolute positions, including the prompt anchor row.
Ordinary post-HC hidden states cannot be reshaped into missing HC channels.

SpeCo continues to own collection transactions, target version checks, teacher
synchronization, training triggers, checkpoints and runtime weight publication.
Serving still uses embedded DSpark and its existing hot-update loader, even
when sampler feature capture is disabled. Actor mode does not enable rollout
capture hooks or disable serving prefix caching.

## Validation

```bash
python -m pytest -q tests/unit/test_deepseek_v4_actor_capture.py \
  tests/unit/test_vllm_ascend_online.py tests/unit/test_rollout_feature_capture.py \
  tests/unit/test_training_backend_isolation.py
```

CPU tests cover real decoder hooks, HC mean/pre-HC bytes, buffer reuse, packed
and singleton batch/sequence layouts, native transport consumption, nonzero
window alignment, masks, stale target rejection, source exclusivity and serving
configuration. Real actor/NPU training remains a hardware validation step.
Actor/NPU training needs its own validation; the historical rollout-capture
NPU runs do not validate this source.

2026-11-11: server regression selection (actor hooks, lifecycle, source isolation,
and existing old-logprob contracts): **124 passed**. This includes unequal-length
padded batches, singleton dimensions, padding exclusion and original-layout
rejection. It is CPU evidence, not a full-size actor/NPU acceptance result.

## Actor target synchronization

The Ascend path accepts native HC-head names and the Transformers names
`model.hc_head.hc_fn`, `hc_base`, and `hc_scale`. Converted bare checkpoint names
are exact matches only, so decoder/compressor norms cannot be mistaken for the
target's output norm. Duplicate matches are rejected.

The existing target-sync RPC now requires all five teacher tensors plus
`embed_tokens.weight`. Before training, it refreshes both teacher and frozen
drafter embedding/LM-head shards from the same actor version. Copies preserve
parameter storage and native vocabulary intervals, including padding. Missing
embedding data is rejected rather than silently retaining initial token weights.
Apply once per target version; a new payload or model initialization resets that
marker. Standard offline behavior and the PyTorch backend remain unchanged.

For FSDP2, enumerate named DTensors and gather only selected tensors. Offloaded
selected shards temporarily return to their mesh device for communication; the
rest of the actor stays offloaded unless the caller requests activation. Exported
CPU tensors own storage and carry no autograd graph. Legacy FSDP and VeOmni keep
their existing actor export/activation interfaces.

## NPU evidence on 2026-11-11

- `actor_shared_cpu_20261111.log`: **210 passed**, including original publish,
  old-logprob and resident lifecycle regressions (overlaps other test groups).
- `actor_api_20261111/shared_cache.log`: **8 passed** on NPU, covering ND/NZ,
  BF16/FP32, two shared-token updates, pointer preservation and FP32 caches.
- `actor_api_20261111/results/actor_capture_report.json`: two-rank real verl
  FSDP2 plus a **miniature, randomly initialized HF DeepSeek V4 fixture** passed
  decoder-tap/HS/logprob comparisons. This proves the API path, not full-target
  correctness or RL quality. The fixture is explicitly marked `api_only_not_trained_model`.
- Full target run `actor_fsdp_20261111` loaded all weights, then failed during
  FSDP initialization with `HcclBroadcast` connection timeout. cgroup OOM/fail
  counters were zero. Preparing/broadcasting a 284.33B model is much slower than
  the fixture; a full retry with longer HCCL connection time remains necessary.
- `actor_api_export_20261111/results/actor_capture_report.json`: the two-NPU
  fixture also passed the actual CPU-offloaded FSDP2 selective DTensor export.
  Six target tensors and actor HS transport payloads were saved; independent
  HC/logprob comparisons had maximum absolute error zero. These short synthetic
  inputs validate transport and APIs, not trained model quality.
- `actor_fsdp_retry_20261111/results/actor_capture_report.json`: the **full
  284.33B target passed on all 16 NPUs**, using real verl 0.9.0 FSDP2, BF16,
  CPU offload and two short synthetic sequences. Feature shape was
  `[2, 8, 28672]`; independent hook/HC/logprob comparisons had maximum absolute
  error zero. Selective export produced all six target tensors, with identical
  hashes across ranks, plus HS transport payloads. The process exited zero at
  17:18:01. Connection/execution timeouts were 600/1800 seconds; OOM counters
  remained zero. This validates the HF actor path, not parity against vLLM
  target inference or a complete RL training run.

All paths above are under `/data/zzc/speco_validation_archive`. These jobs used
the actual container memory limit 2,032,999,997,440 bytes; they do not validate a
1 TiB memory quota. Native serving cold startup, complete actor-driven drafter
training/publication, and a real RL task remain separate acceptance steps.
