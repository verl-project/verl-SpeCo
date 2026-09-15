# verl-SpeCo ROCm / MI355X Enablement

This document explains the changes on the `rocm-enablement` branch: what each change does
and which problem it solves. The goal was to run verl-SpeCo (the DFlash drafter co-training
overlay on verl `release/v0.8.0`) end-to-end on AMD ROCm / MI355X with the SGLang backend,
including a full drafter **collect → train → publish** cycle.

**Hard constraint:** verl-SpeCo is an *import-only overlay* on verl. verl source MUST NOT be
patched. Only `verl_speco/` source and the example run script / constraints are touched here.
Everything that could be expressed as a launch flag was kept config-only in the run script.

## Change summary

| File | Kind | Problem solved |
| --- | --- | --- |
| `verl_speco/main.py` | source (+34) | HIP/CUDA visibility abort on the Ray **driver** |
| `verl_speco/integration/sglang_runtime.py` | source (+103) | HIP/CUDA abort in spawned **SGLang TP** subprocesses; `rocminfo` memory-probe crash |
| `verl_speco/integration/sglang_patch.py` | source (+86) | `IndexError` from SGLang stream-output refactor |
| `verl_speco/trainer/base_trainer.py` | source (+8/-1) | Drafter train+publish never fires (all samples dropped) |
| `examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh` | new | ROCm smoke launch config (attention backend, all-reduce, GPU isolation) |
| `rocm_constraints.txt` | new | Pinned ROCm dependency set |

## Source changes (verl_speco)

### 1. `verl_speco/main.py` — `_neutralize_rocm_hip_visible_devices()` (driver side)

**Problem.** vLLM's ROCm platform module (`vllm/platforms/rocm.py`) runs a one-time
`_sync_hip_cuda_env_vars()` at import that copies `CUDA_VISIBLE_DEVICES` into
`HIP_VISIBLE_DEVICES`. If `HIP_VISIBLE_DEVICES` is present when Ray initializes, Ray's AMD
accelerator manager switches to HIP-keyword mode and rewrites only the HIP var per GPU worker,
leaving each worker's `CUDA_VISIBLE_DEVICES` at the full stale mask. The ROCm runtime then
aborts every GPU worker at `import torch` with *"Conflicting visibility ... between
HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES"*.

**Fix.** At the very start of the Hydra `main()`, pre-trigger the vLLM sync (so its module body
is cached and cannot re-run) and then pop `HIP_VISIBLE_DEVICES`. The driver is left with
`CUDA_VISIBLE_DEVICES` only, so Ray stays CUDA-native and assigns each worker a single physical
GPU without ever introducing a conflicting HIP mask. Guarded to no-op on non-ROCm builds.

### 2. `verl_speco/integration/sglang_runtime.py` — two SGLang-actor patches

Both are installed from `install_sglang_server_actor_runtime()`.

**2a. `_neutralize_rocm_hip_visible_devices()` (server-actor side).** verl's async SGLang server
spawns TP scheduler subprocesses with `multiprocessing` start-method `spawn`. Because SGLang runs
inside a Ray worker, each spawned process re-executes Ray's `default_worker.py`, which imports Ray
and calls `AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()` *before* any SGLang code
runs. That manager raises `ValueError: Inconsistent values ... HIP_VISIBLE_DEVICES or
CUDA_VISIBLE_DEVICES` when both env vars disagree. The mismatch is produced by the same vLLM
`_sync_hip_cuda_env_vars()` running inside the actor, after SGLang narrows `CUDA_VISIBLE_DEVICES`
to one physical GPU per TP child. This is the same class of bug as the driver fix (#1) but must be
applied inside the actor where the sync re-runs.

**2b. `_install_amdgpu_memory_capacity_patch()`.** SGLang's `get_amdgpu_memory_capacity`
(`sglang/srt/utils/common.py`) shells out to `rocminfo | grep ... | awk ...` and parses stdout as
floats. Inside some server-actor processes that pipeline occasionally returns empty stdout with
returncode 0 (the trailing `awk` masks an upstream failure), producing
`ValueError: could not convert string to float: ''` in `ServerArgs.__post_init__` and killing the
whole job. The patch wraps the function so it falls back to torch's device total memory (min across
visible devices, in MiB) whenever the original probe raises or yields nothing.

### 3. `verl_speco/integration/sglang_patch.py` — `patch_sglang_stream_accumulator_hidden_states()`

**Problem.** This SGLang version refactored streaming output out of
`scheduler_output_processor_mixin` (which SPECO's existing hidden-state patch targeted) into
`scheduler_components/output_streamer._GenerationStreamAccumulator.accept`. The old patcher's
import silently failed (`logger.debug` no-op), so the unpatched `accept` appended
`output_hidden_states` **only** for requests with `return_hidden_states`. The tokenizer manager
then indexes `output_hidden_states[i]` by full-batch position, so a partial (drafter-only)
collection produced a list shorter than `rids` and raised `IndexError: list index out of range`.

**Fix.** A source-rewrite patch (regex + `inspect.getsource` + `exec`) that replaces the
conditional append so non-collected positions are padded with `[]`. The list length then always
matches `rids`, or stays empty → `None`, in which case the consumer's `if getattr(...):` guard
short-circuits. Wired into `patch_sglang_hidden_states_tensor_output()`. The patcher is idempotent
and warns (rather than crashing) if the target block is not found.

### 4. `verl_speco/trainer/base_trainer.py` — DFlash positions waiver

**Problem.** The drafter collected samples every step (`collected_samples=16`) but training never
launched: `schedule_reason=9 (no_trainable_batch)` and `train_no_trainable_batch=1` on *every*
step, structurally (confirmed over 35+ steps — more steps did not help). Root cause chain:

1. DFlash-aux hidden states are collected **without positions by design** — the dflash-aux path in
   `sglang_patch.py` sets `logits_output.hidden_states` and the `_verl_dflash_aux_hidden_states`
   flag but never sets a positions tensor, and `sglang_runtime.py` deliberately skips its own
   positions fail-closed check for `uses_dflash_aux_hidden`.
2. The DFlash trainer backend (`backends/dflash_trainer_backend.py`) does **not** read
   `hidden_positions` at all.
3. But `base_trainer.py` unconditionally dropped any sample missing positions whenever
   `collect_hidden_states_from_sgl=True`, with no DFlash exemption. All 16 samples were dropped →
   `trainable_batches=0` → the scheduler (`training_trigger.py`) never launched training → nothing
   was ever published.

**Fix.** Gate the positions requirement on the backend:

```python
require_sglang_positions = bool(
    self.config.rollout.drafter.training.get("collect_hidden_states_from_sgl", False)
) and not self._is_block_drafter_backend()
```

`_is_block_drafter_backend()` is true for the block drafters (`dflash`, `dspark`, `domino`). This
mirrors the waiver already present in `sglang_runtime.py`. DFlash samples (with `hidden_positions =
None`) now flow into the existing `else` legacy-fallback window builder, which already handles the
`None` case. The EAGLE path (which does carry positions) is unaffected.

## Config-only changes (run script)

`examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh` is the ROCm smoke launcher. The
ROCm-specific parts are expressed as flags — **no source change required**:

- `+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=aiter` — the active verl
  (`verl-08`) hardcodes flashinfer for `sglang>=0.5.12` with no ROCm branch; this passthrough
  selects the AITER backend instead.
- `+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True` — avoids the
  `AiterCustomAllreduce` `hipIpcGetMemHandle` abort.
- `export CUDA_VISIBLE_DEVICES=0,1` and **never** setting any HIP var — keeps Ray CUDA-native (see
  the header comment in the script for the full Ray 2.56 AMD accelerator-manager rationale).
- Outputs and Ray temp go to `/dev/shm` because host disk is full.
- FSDP `param_offload` / `optimizer_offload` enabled to fit the smoke on 2 GPUs.
- `trainer.total_training_steps=${SPECO_TOTAL_TRAINING_STEPS:-3}` — step count is overridable for
  longer validation runs; the smoke default stays 3.

`rocm_constraints.txt` pins the validated ROCm dependency set:
`torch==2.11.0+rocm7.2`, `numpy==2.2.6`, `transformers==5.8.1`, `sglang==0.5.14`.

## Validation

Verified on MI355X (GPUs 0,1) with an 8-step run (`SPECO_TOTAL_TRAINING_STEPS=8`):

- All 8 GRPO + DFlash steps completed (`Training Progress: 100%|██████████| 8/8`).
- Every step: `schedule_launch=1`, `schedule_reason=10 (training_ready)`, `trained=1`,
  `published=1`, `train_no_trainable_batch=0`.
- Zero `Drop drafter sample: missing or mismatched SGLang hidden positions` warnings.
- Real draft-weight updates pushed back to SGLang each step.
- The trailing `DataLoader worker ... killed by signal` traceback at exit is benign teardown noise
  (it appears *after* `Final validation metrics: None`, i.e. after the run has finished).

## Reproduce

```bash
# from repo root, inside the verl-speco-rocm container
CUDA_VISIBLE_DEVICES=0,1 SPECO_TOTAL_TRAINING_STEPS=8 \
  bash examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh
```

Check GPU occupancy with `rocm-smi --showmemuse` first; the machine's GPUs are shared and
occupancy is dynamic.

### Bind mount / repo path

The repo is bind-mounted into the container, so the host path
`/home/zhenchen/projects/verl-SpeCo` and the container path
`/workspace/projects/verl-SpeCo` are the **same inode** (verify with `[ A -ef B ]`). Two
consequences:

- Editing the code on the host **immediately** changes what the running container imports —
  no copy step, but also no isolation, so a mid-run edit affects live code.
- The active verl-SpeCo import path is the container path. Always launch from
  `/workspace/projects/verl-SpeCo` (not a stray copy) and, when debugging, read/patch that
  path. (Note the separate "two verl installs" gotcha: the *active* verl is
  `/workspace/projects/verl-08/verl`, which hardcodes flashinfer with no ROCm branch — not
  `/workspace/verl/verl`.)
