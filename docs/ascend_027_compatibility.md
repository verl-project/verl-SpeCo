# Ascend 0.27.1 training compatibility

## Supported runtime combinations

Keep the existing vLLM/Ascend 0.25.1/0.25.1rc1 and 0.26.0/0.26.0rc1 paths.
Add the reviewed 0.27.1/0.27.1rc1 combination, including local build suffixes.
Mixed pairs and unreviewed release candidates remain rejected. Version checks
are shared by the Ascend training and rollout feature adapters.

## Changes isolated to the Ascend backend

- Resolve DSpark/MTP from `models.deepseek_v4.dspark` / `.mtp` in 0.27, retaining
  the original `deepseek_v4_dspark` / `deepseek_v4_mtp` modules for older versions.
- Resolve the HC decoder as `DeepseekV4DecoderLayer` or the original
  `DeepseekV2DecoderLayer`. Install the existing analytical HC backward on the
  actual decoder class, and the mul/add fallback on its actual model module.
  Allow an empty modular package to be imported before patch binding, while
  still rejecting concrete model classes imported before MoE preparation.
- Detect the split MoE ABI by its native method signature. The original
  `apply(..., router_logits, ...)` path remains intact. In 0.27 selection moved
  out of `apply` into `RoutedExperts`, with a different shared-expert runner.
- For grad-enabled DeepSeek V4 BF16 forwards on that split ABI, use the native
  router's selected IDs/forward weights and replay selected routing weights
  with autograd. Reuse SpeCo's consolidated or grouped expert backward, complete
  expert parameter storage, and deterministic TP row reduction. Compute shared
  expert activation through the existing trainable projections. Apply routed
  scaling once; the trainable gate uses its real weight, not an inference copy.
- No-grad serving forwards still call the native model. EP/SP expert execution,
  EPLB, hash routing, remapping and LoRA are explicitly rejected by the new
  numerical branch. SpeCo training DP uses independent TP groups and remains
  separate from those unsupported expert execution modes.
- Bind the existing DSA scatter/sparse backward to the new KV-plan dispatch
  methods as well as retaining the original DeviceOperator boundary. The A2/A3
  path uses `PA_ND`; no-grad plan calls retain native operators. A5
  compressed/SparseFlashMla training is still unsupported.
  Keep eager mode, V1 runner, and FlashComm1 disabled for feature capture.

No changes to upstream verl, the PyTorch drafter backend, SpeCo masks, collection
transactions, triggers or checkpoint/publication protocols are required.
New compatibility files are included in the numerical implementation fingerprint;
this does not make arbitrary old implementation fingerprints resumable.

## Verification

```bash
python3 -m pytest -q tests/unit/test_ascend_model_compat.py \
  tests/unit/test_ascend_runtime_compat.py \
  tests/integration/test_ascend_runtime_source_contract.py
```

The source contracts inspect/execute selected pure methods from the installed
runtime without constructing a large model. Tests cover both legacy/new import
layouts, failure propagation, real HC/DSpark signatures, native expert weight
postprocessing and DSA KV-plan dispatch. CPU numerical tests compare forward
values and gradients for gate, input, complete routed matrices and shared
expert matrices against an independent reference, including scoring functions,
normalization, scaling and clamping. The opt-in small NPU activation test compares
training with the installed serving activation and an independent FP32 reference:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 SPECO_RUN_NPU_TESTS=1 python3 -m pytest -q -s \
  tests/integration/test_ascend_swiglu_npu.py
```

Shared experts use serving-style non-mutating clamps, concatenation and native
`npu_swiglu` under autograd. Custom expert forwards retain their existing fusion.

## Current server

Repository: `/data/zzc/verl-SpeCo`; runtime source: `/vllm-workspace`.
Target: `/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16`.
Trajectory: `/data/zzc/speco_minimal_20261029_143417/one_trajectory.jsonl`.
The target config differs from the old machine; generate new features/checkpoints.
Verified on 2026-11-09:

- Live feature capture: one trajectory, 2608 tokens, 159 supervised positions.
- Native SwiGLU parity: four NPU cases passed (BF16/FP32, clipped/unclipped),
  exact training/serving forward and gradient agreement for these inputs.
- Full DSpark training: TP8/DP2, one optimizer step, loss `0.495410`;
  audited gradients were present, finite and nonzero, and the update probe
  recorded `0.0001220703125`. All 16 Adam state shards have step 1.
- Checkpoint: `speco_restore_20261109_151846/train_confidence_fix_20261109_164629/`
  `checkpoints/step_00000001` under `/data/zzc`; all 40 required shards exist.
- CPU regressions after the final fix: 433 passed. The unused native serving
  confidence head is frozen; the wrapper's optional training head remains usable.
- The earlier 16-rank/four-round small Linear/ZeRO parking test also passed.

Resume, hot-update acceptance and full RL remain separate checks. Image
NumPy/Transformers still fall outside verl's declared version bounds.

## Resume check without another full checkpoint

Keep the same numerical settings and implementation. `MAX_STEPS=2` is the
absolute step cap, so restoring step 1 performs exactly one additional update.
The recipe has two epochs and one batch per DP rank. Set these SpeCo scheduling
options for this validation run:

```bash
export RESUME_FROM=/data/zzc/speco_restore_20261109_151846/train_confidence_fix_20261109_164629/checkpoints/step_00000001
export MAX_STEPS=2
RUN_DIR="$RUN_ROOT/resume_step2_$(date +%Y%m%d_%H%M%S)" \
  bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh resume \
  actor_rollout_ref.rollout.drafter.training.save_interval_steps=100 \
  actor_rollout_ref.rollout.drafter.training.save_final_checkpoint=false
```

Reuse the established model/feature paths, TP8/DP2 and fused SwiGLU environment.
A passed report with `global_step=2` verifies restore plus another update; this
run deliberately produces a report without a second 354 GiB checkpoint.

## Export replica audit

Strict export of the step-1 checkpoint found a BF16 Markov W1 discrepancy:
flat index 5408595 in ranks 2/5/10/13 differs from TP0 by exactly one ULP
(`9.5367431640625e-7`); all other W1 elements match. All 16 rank states were
checked. The existing explicit `allow_markov_w1_one_ulp=True` option permits
at most one differing W1 element and keeps TP0 without averaging. It is not the
default export policy. This bounded export check does not establish strict TP
replica equality or complete model training/serving parity. The original
checkpoint is preserved. Audit: `/data/zzc/speco_validation_archive/`
`markov_w1_replica_audit_20261109.json`.


Bounded CPU export completed: 2375 tensors were saved as standard PyTorch shards
and read back exactly against the merged weights. Artifact:
`/data/zzc/speco_restore_20261109_151846/draft_pytorch_step_00000001`.
Its `speco_export_report.json` records `weights_verified=true`,
`strict_replica_parity=false` and `inference_validated=false`. Strict default
export remains blocked by the recorded W1 replica difference.
