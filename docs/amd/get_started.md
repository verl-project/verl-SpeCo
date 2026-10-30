# Getting Started on AMD GPUs (ROCm)

Last updated: 10/30/2026

This guide takes you from a bare AMD GPU node to a verified verl-SpeCo run:
GRPO on Qwen3-8B with a DFlash drafter co-trained during rollout. The run
includes a full drafter **collect → train → publish** cycle every step. Two
rollout engines are covered: **SGLang** and **vLLM**.

What is validated today:

| Item | Validated |
| --- | --- |
| GPU | 2 × AMD Instinct MI355X (gfx950, 288 GB) per run, single node |
| Image | `verl-speco:rocm`, built from [`docker/rocm/Dockerfile.rocm`](../../docker/rocm/Dockerfile.rocm) on `amdagi/verl-dev:rocm7.14_torch2.12_release_0724` |
| verl | `0.9.0.dev`, the build bundled in the base image, not reinstalled |
| Rollout engine | SGLang 0.5.16 (`attention_backend=aiter`) and vLLM 0.22.1rc1 |
| Drafter | DFlash (`z-lab/Qwen3-8B-DFlash-b16`) |
| Actor | FSDP2 with param/optimizer offload, rollout TP=2 |

The SGLang path was also run end to end on verl `release/v0.8.0` (see
[§2](#2-build-the-image)). Anything outside this table has not been run on ROCm
yet. That includes MI300X, other drafters, and multi-node. See
[Limitations](#limitations).

## 1. Prerequisites

- An AMD Instinct node with at least 2 free GPUs per run. The image is built
  for `gfx942` (MI300X/MI325X) and `gfx950` (MI350X/MI355X).
- A ROCm kernel driver (`amdgpu`) on the host; `/dev/kfd` and `/dev/dri` must exist.
  The ROCm user-space stack (7.14) ships inside the image.
- Docker, and about 70 GB free disk for the image (the base image is ~63 GB).
- About 20 GB for the Qwen3-8B and DFlash weights, plus the GSM8K dataset (small).
- Check which GPUs are free before you start. On a shared node, other users'
  jobs are the most common cause of OOMs and hangs:

  ```bash
  rocm-smi --showmemuse
  ```

## 2. Build the image

Docker is the recommended way to install verl-SpeCo on ROCm. By default, use
the environment that ships in the container as is. verl-SpeCo does not pin
ROCm dependencies, and installing torch, SGLang or vLLM from pip over the
container's builds is not supported.

The image is a thin layer on verl's official ROCm image
(`amdagi/verl-dev:rocm7.14_torch2.12_release_0724`, from verl's
[AMD quick start](https://github.com/verl-project/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst)).
That base already contains everything verl-SpeCo needs at runtime, including
**both** rollout engines:

| Component | Version in the base image |
| --- | --- |
| PyTorch | 2.12.0+rocm7.14 (Python 3.12, `/opt/venv`) |
| SGLang | 0.5.16 (`/workspace/sglang`) |
| vLLM | 0.22.1rc1 (`/workspace/vllm`) |
| verl | 0.9.0.dev (site-packages; source snapshot at `/workspace/verl`) |
| transformers / ray | 5.12.1 / 2.56.1 |

The Dockerfile keeps all of these unchanged, verl included. It only installs
this repo in editable mode at `/workspace/verl-SpeCo`. It uses `--no-deps`, so
pip never replaces the ROCm builds.

From the repo root:

```bash
docker build -f docker/rocm/Dockerfile.rocm -t verl-speco:rocm .
# optional: a different verl ROCm base image (untested)
docker build -f docker/rocm/Dockerfile.rocm --build-arg BASE_IMAGE=<image> -t verl-speco:rocm .
```

The build prints a sanity check at the end. You should see:

```
torch 2.12.0+rocm7.14.0a20260608 hip 7.14.60850
verl 0.9.0.dev /opt/venv/lib/python3.12/site-packages/verl/__init__.py
verl_speco 0.1.0
```

One image serves both rollout engines. You choose the engine at run time with
`actor_rollout_ref.rollout.name=sglang|vllm`. Each engine has its own launch
script (see [§6](#6-run-the-smoke-test)).

**About the bundled verl.** verl-SpeCo accepts the image's `0.9.0.dev` build
under the `release/v0.9.0` contract. `resolve_verl_compatibility()` reports
`matched release/v0.9.0 version/API contract`. The bundled build differs from
verl `release/v0.8.0` in three defaults that matter here, and the ROCm scripts
already account for all three:

- `rollout.yaml` presets `engine_kwargs.{sglang,vllm}.disable_custom_all_reduce: True`
  and `engine_kwargs.sglang.attention_backend: triton`. The scripts therefore
  override them with `++` (add or override), not `+` (add only).
- `rollout.calculate_log_probs` defaults to `True`. The scripts set it to
  `False`, because SGLang's DFLASH mode rejects `return_logprob`.

Optional, not the default: to run on verl `release/v0.8.0` instead, install it
over the bundled verl inside the container. The same scripts work unchanged on
either version:

```bash
git clone -b release/v0.8.0 https://github.com/verl-project/verl.git /workspace/verl-0.8
pip install --no-deps -e /workspace/verl-0.8
python -c "import verl; print(verl.__version__)"   # 0.8.0
```

## 3. Start the container

```bash
docker run -it --name verl-speco-rocm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --shm-size 128g \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  verl-speco:rocm
```

Notes:

- `--shm-size 128g` is needed: the smoke scripts put Ray's temp dir and the
  checkpoints under `/dev/shm`.
- To edit verl-SpeCo from the host, also mount your checkout over the baked-in copy:
  `-v $PWD:/workspace/verl-SpeCo`. The package is installed in editable mode, so
  no reinstall is needed.
- If a model lives in a Hugging Face cache (`~/.cache/huggingface/hub`), mount
  the whole `models--<org>--<name>` directory, not only `snapshots/<rev>`. The
  snapshot files are symlinks into `../../blobs`, and those links break if only
  the snapshot directory is mounted. Example:
  `-v ~/.cache/huggingface/hub/models--z-lab--Qwen3-8B-DFlash-b16:/models/hf-dflash:ro`,
  then `DRAFTER_PATH=/models/hf-dflash/snapshots/<rev>`.
- Do **not** pass `-e HIP_VISIBLE_DEVICES=...`. See
  [GPU selection](#gpu-selection-read-this-once).

## 4. Check the environment

Inside the container:

```bash
python - <<'EOF'
import torch, verl, verl_speco, sglang, vllm
from verl.utils.device import get_device_name
print("hip       :", torch.version.hip)                  # 7.14.x
print("gpus      :", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("verl dev  :", get_device_name())                   # "cuda"; ROCm torch reports as CUDA
print("verl      :", verl.__version__)                    # 0.9.0.dev
print("verl_speco:", verl_speco.__version__)
print("sglang    :", sglang.__version__)                  # 0.5.16
print("vllm      :", vllm.__version__)                    # 0.22.1rc1...
EOF
```

`get_device_name()` returning `cuda` is expected. ROCm PyTorch exposes the
`torch.cuda` API, and `nccl` maps to RCCL, so verl and verl-SpeCo use their
CUDA code paths on AMD.

## 5. Prepare data and models

**GSM8K.** Use verl's preprocessing script to write `train.parquet` and
`test.parquet`:

```bash
python /workspace/verl/examples/data_preprocess/gsm8k.py --local_save_dir /data/gsm8k
```

**Models.** The smoke scripts run with `HF_HUB_OFFLINE=1`, so download the
weights first and pass local paths:

```bash
hf download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B
hf download z-lab/Qwen3-8B-DFlash-b16 --local-dir /models/Qwen3-8B-DFlash-b16
```

## 6. Run the smoke test

There is one launcher per engine. Both use the same sizes: 8 prompts × 4 samples
per step, 256/512 prompt/response tokens, rollout TP=2, and 4 drafter optimizer
steps after every GRPO step.

| Engine | Script | Where the drafter's training hidden states come from |
| --- | --- | --- |
| SGLang | [`examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh`](../../examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh) | Returned by SGLang during rollout (`collect_hidden_states_from_sgl=True`) |
| vLLM | [`examples/run_qwen3-8b_drafter_dflash_vllm_rocm.sh`](../../examples/run_qwen3-8b_drafter_dflash_vllm_rocm.sh) | Captured from the FSDP actor during the old-log-prob pass (`collect_hidden_states_from_old_logprob=True`) |

The default paths inside the scripts are placeholders or site-specific, so
always override them:

```bash
cd /workspace/verl-SpeCo
# Keep the launcher's exit status through `| tee`.
set -o pipefail

# SGLang
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/models/Qwen3-8B \
DRAFTER_PATH=/models/Qwen3-8B-DFlash-b16 \
TRAIN_FILE=/data/gsm8k/train.parquet \
TEST_FILE=/data/gsm8k/test.parquet \
SPECO_TOTAL_TRAINING_STEPS=8 \
  bash examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh 2>&1 | tee /tmp/speco_rocm_sglang.log

# vLLM (same variables, different script)
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/models/Qwen3-8B \
DRAFTER_PATH=/models/Qwen3-8B-DFlash-b16 \
TRAIN_FILE=/data/gsm8k/train.parquet \
TEST_FILE=/data/gsm8k/test.parquet \
SPECO_TOTAL_TRAINING_STEPS=8 \
  bash examples/run_qwen3-8b_drafter_dflash_vllm_rocm.sh 2>&1 | tee /tmp/speco_rocm_vllm.log
```

Run them one after the other on the same 2 GPUs. To run both at once on a
node with 4 or more GPUs, give the vLLM run `CUDA_VISIBLE_DEVICES=2,3`; the two
GPU sets must not overlap, also when the runs are in separate containers.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | The 2 GPUs to use (TP=2). Pick free ones. |
| `MODEL_PATH` / `DRAFTER_PATH` | placeholder / site-specific | Target and drafter weights. |
| `TRAIN_FILE` / `TEST_FILE` | placeholder / site-specific | GSM8K parquet files. |
| `CKPTS_DIR` | `/dev/shm/verl_speco_ckpt` | Checkpoint dir (`save_freq=-1`, so nothing is saved by default). |
| `SPECO_TOTAL_TRAINING_STEPS` | `3` | GRPO steps. Use 8 to reproduce the reference numbers below. |
| `SPECO_RAY_NUM_CPUS` | `32` | Ray CPU pool. |
| `SPECO_RAY_WORKER_SOFT_LIMIT` | `8` | Ray idle worker soft limit. |

Any extra Hydra overrides you append are forwarded to `python -m verl_speco.main`.

### ROCm-specific settings in the scripts

Keep these unless you are working on the corresponding issue.

Both engines:

| Setting | Why |
| --- | --- |
| `rollout.calculate_log_probs=False` | Bundled verl defaults it to `True`. SGLang DFLASH rejects `return_logprob`. The actor recomputes old log-probs anyway. |
| `++...engine_kwargs.<engine>.disable_custom_all_reduce=True` | Custom all-reduce aborts in `hipIpcGetMemHandle` (seen on SGLang/AITER). `++` because the bundled `rollout.yaml` already defines the key. |
| `CUDA_VISIBLE_DEVICES` only | See [GPU selection](#gpu-selection-read-this-once). |

SGLang only:

| Setting | Why |
| --- | --- |
| `++...engine_kwargs.sglang.attention_backend=aiter` | verl picks `flashinfer` for SGLang ≥ 0.5.12, which does not exist on ROCm. The bundled `rollout.yaml` presets `triton`; `aiter` is the validated backend. |

vLLM only:

| Setting | Why |
| --- | --- |
| `rollout.max_num_seqs=256`, `rollout.max_num_batched_tokens=12288` | vLLM spec decoding needs `max_num_batched_tokens > max_num_seqs × num_speculative_tokens`. The DFlash-b16 drafter uses 15 draft tokens, so verl's defaults (1024 seqs, 8192 tokens) are rejected. |
| `drafter.training.hidden_state_window_tokens_per_sample=256`, `hidden_state_window_min_rows=256` | The old-log-prob collector only takes responses with at least `window + 1` tokens. The default window of 512 can never be met with `max_response_length=512`, so nothing would be collected. |

### GPU selection (read this once)

On ROCm, select GPUs **only** with `CUDA_VISIBLE_DEVICES`. Never set
`HIP_VISIBLE_DEVICES`, and do not set Ray's `NOSET_*_VISIBLE_DEVICES` flags.

The reason: at import time, vLLM copies `CUDA_VISIBLE_DEVICES` into
`HIP_VISIBLE_DEVICES`. Ray then narrows only the HIP mask for each worker. The
HIP runtime aborts with `Conflicting visibility of agent-N between
HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES`, and Ray rejects the mix with
`Inconsistent values found`.

verl-SpeCo strips the copied variable for you
(`verl_speco.integration.rocm_env.neutralize_hip_visible_devices`, called on
the Ray driver and inside the SGLang server actor). It cannot fix a
`HIP_VISIBLE_DEVICES` that you export yourself.

## 7. Expected results

Reference runs: `verl-speco:rocm` image (bundled verl 0.9.0.dev),
`SPECO_TOTAL_TRAINING_STEPS=8`, settings as in the scripts. Each run used 2 ×
MI355X; the two runs ran at the same time on disjoint GPUs.

| What | SGLang | vLLM |
| --- | --- | --- |
| Progress | `8/8 [08:50<00:00, 66.26s/it]` | `8/8 [06:51<00:00, 51.41s/it]` |
| Exit | `EXIT_CODE=0` after `Final validation metrics: None` | same |
| `drafter/collected_samples` | `16.0` every step | `16.0` every step (an occasional `15.0` is normal) |
| `drafter/schedule_reason` | `10.0` (`training_ready`) every step | same |
| `drafter/trained`, `drafter/published` | `1.0` every step | same |
| `drafter/train_no_trainable_batch` | `0.0` | `0.0` |
| `drafter/train_optimizer_step_max` | `32` after step 8 (4 optimizer steps per GRPO step) | same |
| `row mismatch` / `Drop drafter sample` messages | **none** | **none** |
| Draft weight publish | `[speco sglang target update] done global_steps=8 buckets=8 elapsed_sec≈0.5` | `[speco draft ipc] streamed 58 translated weights in 5 buckets into DFlashQwen3Model (method=dflash)`; the vLLM path uses the `compat` draft update (see below) |
| Speculative acceptance | SGLang decode log `accept len` 2.8–3.6 (median 3.2), `cuda graph: True`, up to ~3400 tok/s | `drafter/spec_decode/mean_acceptance_length` 3.09–3.26 |
| Drafter loss (step 8, 4 drafter steps) | `6.17 → 6.18 → 7.15 → 5.99` | `2.69 → 2.92 → 2.72 → 2.75` |
| Drafter top-1 / top-5 (step 8) | ≈ 0.08 / 0.25 | ≈ 0.25 / 0.49 |
| Actor memory | `max_memory_allocated_gb` ≈ 76.4, `max_memory_reserved_gb` ≈ 79.9 | same |

Throughput and loss values vary with GPU load and sampling. The node was shared
with other jobs during these runs. The drafter scheduling metrics
(`schedule_reason`, `trained`, `published`) should match exactly.

Two engine differences are expected:

- The **drafter loss and accuracy are not comparable** between engines. The two
  engines take training hidden states from different sources (see the table in
  [§6](#6-run-the-smoke-test)), so the SGLang numbers are much lower. See
  [Limitations](#limitations).
- vLLM logs `[speco draft update] backend=vllm native_draft_update=disabled ...
  path=compat fallback_reason=vllm: native draft update unavailable`. Upstream
  vLLM has no native draft-weight-update RPC, so verl-SpeCo streams the drafter
  weights through its IPC compat path. `drafter/published` still reads `1.0`.

Log lines from the reference runs:

```
# SGLang
(SpecoSGLangHttpServer) Decode batch, #running-req: 32, ... accept len: 3.45, accept rate: 0.16, cuda graph: True, gen throughput (token/s): 3432.72
(SpecoWorker) [drafter logits quality] valid_tokens=3796 top1_acc=0.077977 top5_acc=0.249210 ...
(SpecoWorker) [drafter loss] step=4 optimizer_step_total=32 lr=1.000e-05 loss=5.9938 vloss=0.0000 ploss=5.9938
(WorkerDict)  [speco sglang target update] done global_steps=8 buckets=8 elapsed_sec=0.482
(SpecoTaskRunner) Training Progress: 100%|██████████| 8/8 [08:50<00:00, 66.26s/it]

# vLLM
(WorkerDict)  [speco draft ipc] streamed 58 translated weights in 5 buckets into DFlashQwen3Model (method=dflash), ...
(SpecoWorker) [drafter logits quality] valid_tokens=3835 top1_acc=0.253716 top5_acc=0.491525 ...
(SpecoWorker) [drafter loss] step=4 optimizer_step_total=32 lr=1.000e-05 loss=2.7450 vloss=0.0000 ploss=2.7450
(SpecoTaskRunner) Training Progress: 100%|██████████| 8/8 [06:51<00:00, 51.41s/it]
```

### Checking a run

```bash
LOG=/tmp/speco_rocm_sglang.log   # or /tmp/speco_rocm_vllm.log
# step metrics are printed as e.g. drafter/trained:np.float64(1.0)
# for 8 steps expect exactly: 8 × published(1.0), 8 × schedule_reason(10.0), 8 × trained(1.0)
grep -oE "drafter/(schedule_reason|trained|published|collected_samples):np\.float64\([0-9.]+\)" $LOG | sort | uniq -c
# expect 0 and 0
grep -c "row mismatch" $LOG; grep -c "Drop drafter sample" $LOG
# speculative acceptance: SGLang, then vLLM
grep -oE "accept len: [0-9.]+" $LOG | tail -3
grep -oE "mean_acceptance_length:[0-9.]+" $LOG | tail -3
# drafter loss
grep "\[drafter loss\]" $LOG | tail -4
```

### Benign noise

These messages do not indicate failures:

- `Warning: Stream-K Data Parallel does not support GSU > 1, setting GSU to 1.`
  This is an informational hipBLASLt message, printed many times.
- `Warning: Failed to set NUMA affinity: NVML Shared Library Not Found`. There is
  no NVML on AMD.
- On vLLM, the `native_draft_update=disabled ... path=compat` message described
  above.
- A `DataLoader worker ... killed by signal` traceback during teardown, printed
  **after** `Final validation metrics: None`. The run has already finished.

## Troubleshooting

These are the failures seen while bringing the stack up. With the image and
scripts above, you should hit none of them.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Hydra: `Could not append to config. An item is already at 'actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend'` | The bundled verl's `rollout.yaml` already defines the key, so `+key=` fails | Use `++key=` (add or override). It also works on verl 0.8.0, where the key is absent. |
| SGLang: `DFLASH speculative decoding does not support return_logprob yet` | The bundled verl defaults `rollout.calculate_log_probs=True` | Set `actor_rollout_ref.rollout.calculate_log_probs=False`. |
| vLLM: `max_num_scheduled_tokens is set to -7168 based on the speculative decoding settings` | `max_num_batched_tokens` ≤ `max_num_seqs × 15` draft tokens | Set `rollout.max_num_seqs=256` and `rollout.max_num_batched_tokens=12288`. |
| vLLM: run completes, but every step shows `drafter/collected_samples` `0.0`, `drafter/schedule_reason` `4.0`, `drafter/trained` `0.0` | No response is longer than the old-log-prob hidden window (default 512 rows + 1) | Set `drafter.training.hidden_state_window_tokens_per_sample` and `hidden_state_window_min_rows` below `max_response_length` (the script uses 256). |
| Run completes, but `drafter/trained` and `drafter/published` are `0.0` every step, with `DFlash input/hidden/mask row mismatch: input_rows=301, hidden_rows=300, mask_rows=301` | Online samples carry one input row more than hidden rows (the trailing label token), which verl-SpeCo builds between #73 and #109 rejected | Fixed upstream in #109 (block drafters carry `label_ids`/`label_mask` separately). Update verl-SpeCo. |
| SGLang fails on `flashinfer` | verl hardcodes `flashinfer` for SGLang ≥ 0.5.12 | Keep `engine_kwargs.sglang.attention_backend=aiter`. |
| `Conflicting visibility of agent-N between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES`, then `Fatal Python error: Aborted` in every GPU worker | A `HIP_VISIBLE_DEVICES` is present alongside Ray's CUDA mask | Unset `HIP_VISIBLE_DEVICES` and use `CUDA_VISIBLE_DEVICES` only. Confirm you are running a verl-SpeCo version that has `rocm_env.py`. |
| Ray `ValueError: Inconsistent values found. Please use either HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES.` | Same as above, inside the SGLang TP subprocess | Same as above. |
| `ValueError: could not convert string to float: ''` in `ServerArgs.__post_init__` | SGLang's `rocminfo` memory probe returned empty output | Handled by verl-SpeCo's torch fallback. If it still appears, check that `rocminfo` works in the container. |
| `hipIpcGetMemHandle ... invalid argument`, then abort in `custom_all_reduce.cu` | AITER custom all-reduce | Keep `engine_kwargs.<engine>.disable_custom_all_reduce=True`. |
| `IndexError: list index out of range` at `tokenizer_manager.py` `meta_info["hidden_states"]` | SGLang stream accumulator shorter than `rids` | Handled by `patch_sglang_stream_accumulator_hidden_states()`. If it still appears, the SGLang version changed; check the patch still applies. |
| Run completes but `drafter/schedule_reason` is `9.0` and `drafter/trained` is `0.0` every step, with `Drop drafter sample: missing or mismatched SGLang hidden positions ... positions=None` | Positions required for a block drafter | Fixed in `base_trainer.py`. Update verl-SpeCo. |
| Drafter fails to load with a missing `config.json` / safetensors, or a broken-link error | Only `snapshots/<rev>` of an HF cache was mounted; its symlinks point into `../../blobs` | Mount the whole `models--<org>--<name>` directory (see [§3](#3-start-the-container)). |
| OOM, `invalid device ordinal`, or a hang at startup | GPUs busy or mask wrong | Check `rocm-smi --showmemuse`, pick 2 idle GPUs, and remove any `NOSET_*` or `HIP_VISIBLE_DEVICES` from the environment. |

## Limitations

- Only **DFlash** has been run end to end on ROCm, on both SGLang and vLLM.
  EAGLE3, P-EAGLE, DFlash2, DSpark and Domino are present in the image but
  unvalidated.
- The drafter's training accuracy differs by engine. With the pretrained
  DFlash-b16 drafter, the vLLM path (hidden states from the FSDP actor) starts
  at ≈ 0.75 top-1 for the first block position. The SGLang path (hidden states
  returned by SGLang) starts at ≈ 0.16, and its loss is ~2× higher. Rollout
  acceptance is similar on both engines (~3.1–3.3). The SGLang-collected hidden
  states have not yet been checked against a CUDA run, so treat SGLang drafter
  training quality on ROCm as unverified.
- The configuration is sized for 2 GPUs: FSDP param and optimizer offload are
  on, and custom all-reduce is off. It is a functional smoke, not a performance
  baseline.
- Single node, TP=2 only. MI300X (gfx942) is a build target but has not been run.
