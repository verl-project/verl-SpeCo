# Getting Started on AMD GPUs (ROCm)

Last updated: 11/03/2026

This guide takes you from a bare AMD GPU node to a verified verl-SpeCo run:
GRPO on Qwen3-8B with a DFlash drafter co-trained during rollout. The run
includes a full drafter **collect → train → publish** cycle every step. Two
rollout engines are covered: **SGLang** and **vLLM**.

What is validated today:

| Item | Validated |
| --- | --- |
| GPU | 2 × AMD Instinct MI355X (gfx950, 288 GB) per node; 1 node, and 2 nodes over TCP |
| SGLang image | `verl-speco:rocm-sglang`, built from [`docker/rocm/Dockerfile.sglang`](../../docker/rocm/Dockerfile.sglang): SGLang 0.5.12 (`attention_backend=triton`), verl `release/v0.9.0` |
| vLLM image | `verl-speco:rocm-vllm`, built from [`docker/rocm/Dockerfile.vllm`](../../docker/rocm/Dockerfile.vllm): vLLM 0.22.1rc1, the verl `0.9.0.dev` bundled in the base image |
| Drafter | DFlash (`z-lab/Qwen3-8B-DFlash-b16`) |
| Actor | FSDP2 with param/optimizer offload, rollout TP=2 |

Anything outside this table has not been run on ROCm yet. That includes MI300X, other drafters, and more than 2 nodes. See
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

Docker is the recommended way to install verl-SpeCo on ROCm. There is one image
per rollout engine. Use the environment that ships in the image as is:
installing torch, SGLang or vLLM from pip over the image's ROCm builds is not
supported.

**SGLang image.** The same SGLang release as the GPU image
(`verlai/verl:sgl0512.dev1`), from SGLang's official ROCm build:

| Component | Version |
| --- | --- |
| Base | `lmsysorg/sglang:v0.5.12-rocm720-mi35x@sha256:3be1556105cdfdede103dc053f002d4d0307bdb229d0e15bfef130d5c53af208` |
| SGLang | 0.5.12 (tag `v0.5.12`, `/sgl-workspace/sglang`) |
| PyTorch | 2.9.1+rocm7.2.0 (Python 3.10) |
| verl | `release/v0.9.0` from source at `/workspace/verl`, as in `docker/verl0.9.0/Dockerfile.sglang` |
| transformers / ray | 5.10.4 (verl 0.9.0 excludes 5.6.0) / 2.56.1 (pinned, see [GPU selection](#gpu-selection-read-this-once)) |

For MI300X, build on the `-mi30x` variant of the same release with
`--build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.12-rocm720-mi30x` (not run yet).

**vLLM image.** A thin layer on verl's ROCm image
(`amdagi/verl-dev:rocm7.14_torch2.12_release_0724`, from verl's
[AMD quick start](https://github.com/verl-project/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst)):

| Component | Version in the base image |
| --- | --- |
| PyTorch | 2.12.0+rocm7.14 (Python 3.12, `/opt/venv`) |
| vLLM | 0.22.1rc1 (`/workspace/vllm`) |
| verl | 0.9.0.dev (site-packages; source snapshot at `/workspace/verl`) |
| transformers / ray | 5.12.1 / 2.56.1 |

The Dockerfile keeps all of these unchanged, verl included, and only installs
this repo in editable mode with `--no-deps`. The base image also contains an
SGLang dev build newer than 0.5.12; do not use it for SGLang runs.

From the repo root:

```bash
docker build -f docker/rocm/Dockerfile.sglang -t verl-speco:rocm-sglang .
docker build -f docker/rocm/Dockerfile.vllm -t verl-speco:rocm-vllm .
```

Each build prints a sanity check at the end. You should see:

```
# Dockerfile.sglang
torch 2.9.1+rocm7.2.0.git7e1940d4 hip 7.2.26015-fc0010cf6a
sglang 0.5.12 transformers 5.10.4
verl 0.9.0 /workspace/verl/verl/__init__.py ray 2.56.1
verl_speco 0.1.0

# Dockerfile.vllm
torch 2.12.0+rocm7.14.0a20260608 hip 7.14.60850
verl 0.9.0.dev /opt/venv/lib/python3.12/site-packages/verl/__init__.py
verl_speco 0.1.0
```

You choose the engine at run time with `actor_rollout_ref.rollout.name`. Each
engine has its own launch script (see [§6](#6-run-the-smoke-test)); run it in
the matching image.

**About verl.** Both images use verl 0.9, and verl-SpeCo resolves both under the
`release/v0.9.0` contract. Two verl 0.9 settings matter here, and the ROCm
scripts already handle them:

- The bundled `0.9.0.dev` build's `rollout.yaml` presets
  `engine_kwargs.{sglang,vllm}.disable_custom_all_reduce: True` and
  `engine_kwargs.sglang.attention_backend: triton`. The scripts therefore set
  these keys with `++` (add or override), not `+` (add only), which works with
  or without the preset.
- `rollout.calculate_log_probs` defaults to `True`. The scripts set it to
  `False`, because SGLang's DFLASH mode rejects `return_logprob`.

## 3. Start the container

```bash
docker run -it --name verl-speco-rocm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 128g \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  verl-speco:rocm-sglang    # or verl-speco:rocm-vllm
```

Notes:

- `--shm-size 128g` is needed: the smoke scripts put Ray's temp dir and the
  checkpoints under `/dev/shm`.
- The command uses Docker's default (bridge) network and a private IPC
  namespace, which is how the reference runs were made. With
  `--network host`, two containers starting at the same time on one node have
  failed RCCL setup with `hipIpcGetMemHandle failed : invalid argument` (see
  [Troubleshooting](#troubleshooting)).
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
import importlib.metadata as md
import torch, verl, verl_speco
from verl.utils.device import get_device_name
print("hip       :", torch.version.hip)                  # 7.2.x (SGLang image) / 7.14.x (vLLM image)
print("gpus      :", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("verl dev  :", get_device_name())                   # "cuda"; ROCm torch reports as CUDA
print("verl      :", verl.__version__)                    # 0.9.0 / 0.9.0.dev
print("verl_speco:", verl_speco.__version__)
for pkg in ("sglang", "vllm", "ray"):                     # sglang 0.5.12 or vllm 0.22.1rc1...; ray 2.56.1
    try:
        print(f"{pkg:10s}:", md.version(pkg))
    except md.PackageNotFoundError:
        pass
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

The default paths inside the scripts are placeholders, so always override them:

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
| `MODEL_PATH` / `DRAFTER_PATH` | `/path/to/...` placeholder | Target and drafter weights. |
| `TRAIN_FILE` / `TEST_FILE` | `/path/to/...` placeholder | GSM8K parquet files. |
| `CKPTS_DIR` | `/dev/shm/verl_speco_ckpt` | Checkpoint dir (`save_freq=-1`, so nothing is saved by default). |
| `SPECO_TOTAL_TRAINING_STEPS` | `3` | GRPO steps. Use 8 to reproduce the reference numbers below. |
| `SPECO_RAY_NUM_CPUS` | `32` | Ray CPU pool. |
| `SPECO_RAY_WORKER_SOFT_LIMIT` | `8` | Ray idle worker soft limit. |
| `SPECO_NNODES` | `1` | Number of nodes. Above 1, the script joins the Ray cluster at `RAY_ADDRESS` (see [Multi-node](#multi-node)). |

Any extra Hydra overrides you append are forwarded to `python -m verl_speco.main`.

The scripts check the pipeline, not the learning. Qwen3-8B answers in thinking
mode by default, and almost every response hits the 512-token limit, so the
GSM8K reward stays near 0 and the GRPO advantages are 0. For a run with a
meaningful reward, turn thinking off:

```bash
bash examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh \
  +data.apply_chat_template_kwargs.enable_thinking=False
```

Responses are then about 200–300 tokens. On vLLM, also lower the hidden-state
window (see the vLLM settings below) so that enough responses are long enough
to collect:

```bash
bash examples/run_qwen3-8b_drafter_dflash_vllm_rocm.sh \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=128 \
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=128
```

### Multi-node

Each node contributes the 2 GPUs in its `CUDA_VISIBLE_DEVICES`; rollout stays
TP=2, so 2 nodes give 2 rollout replicas and a 4-GPU FSDP actor. Start one
container per node with the same image, models and data at the same paths, and
`--network host` (Ray and RCCL need the node addresses). Then:

```bash
# Every node: pin RCCL/Gloo to the interface that connects the nodes.
export NCCL_SOCKET_IFNAME=<iface> GLOO_SOCKET_IFNAME=<iface>

# Head node
ray start --head --port=6379 --num-gpus=2
# Each other node
ray start --address=<head-ip>:6379 --num-gpus=2

# Head node, after `ray status` lists every node
RAY_ADDRESS=<head-ip>:6379 SPECO_NNODES=2 \
MODEL_PATH=... DRAFTER_PATH=... TRAIN_FILE=... TEST_FILE=... \
SPECO_TOTAL_TRAINING_STEPS=8 \
  bash examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh
```

With `SPECO_NNODES` above 1, the script sets `trainer.nnodes` and lets Ray use
the existing cluster's CPUs instead of starting a local one. The reference
2-node runs used TCP between the nodes (`NCCL_IB_DISABLE=1`), so a step took
about 145 s instead of about 21 s; with RDMA it is much faster.

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
| `++...engine_kwargs.sglang.attention_backend=triton` | verl picks `flashinfer` for SGLang ≥ 0.5.12, which does not exist on ROCm. `triton` is the backend preset in verl's `rollout.yaml`. Do not use `aiter` with SGLang 0.5.12 under verl: the first rollout after the weight sync fails (see [Troubleshooting](#troubleshooting)). |

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

This relies on Ray honoring `CUDA_VISIBLE_DEVICES` on AMD GPUs. Ray 2.56.1
does; Ray 2.57 and newer use `HIP_VISIBLE_DEVICES` only. Both images use Ray
2.56.1, so do not upgrade Ray inside them.

verl-SpeCo strips the copied variable for you
(`verl_speco.integration.rocm_env.neutralize_hip_visible_devices`, called on
the Ray driver and inside the SGLang server actor). It cannot fix a
`HIP_VISIBLE_DEVICES` that you export yourself.

## 7. Expected results

Reference runs: the two images above, `SPECO_TOTAL_TRAINING_STEPS=8`, the
scripts unchanged (thinking mode on). Each run used 2 × MI355X on a shared node;
the two runs overlapped on disjoint GPUs.

| What | SGLang | vLLM |
| --- | --- | --- |
| Progress | `8/8 [03:30<00:00, 22.54s/it]` | `8/8 [03:22<00:00, 20.85s/it]` |
| Exit | `EXIT_CODE=0` after `Final validation metrics: None` | same |
| `drafter/collected_samples` | `16.0` every step | `16.0` every step (an occasional `15.0` is normal) |
| `drafter/schedule_reason` | `10.0` (`training_ready`) every step | same |
| `drafter/trained`, `drafter/published` | `1.0` every step | same |
| `drafter/train_no_trainable_batch` | `0.0` | `0.0` |
| `drafter/train_optimizer_step_max` | `32` after step 8 (4 optimizer steps per GRPO step) | same |
| `row mismatch` / `Drop drafter sample` messages | **none** | **none** |
| Draft weight publish | `[speco sglang draft update] done global_steps=8 buckets=5 bucket_mb=512 elapsed_sec≈0.5` | `[speco draft ipc] streamed 58 translated weights in 5 buckets into DFlashQwen3Model (method=dflash)` |
| Speculative acceptance | SGLang decode log `accept len` 2.80–3.77 (median 3.25), `cuda graph: True`, up to ~5000 tok/s | `drafter/spec_decode/mean_acceptance_length` 3.07–3.28 |
| Drafter loss (step 8, 4 drafter steps) | `2.41 → 2.37 → 2.55 → 2.31` | `2.64 → 2.84 → 2.75 → 2.78` |
| Drafter top-1 / top-5 (step 8) | ≈ 0.31 / 0.56 | ≈ 0.23 / 0.47 |
| Actor memory | `max_memory_allocated_gb` ≈ 76.4, `max_memory_reserved_gb` ≈ 83.2 | same |
| Reward | `critic/score/mean` 0.0, `response_length/clip_ratio` 1.0 (thinking mode, see [§6](#6-run-the-smoke-test)) | same |

Throughput and loss values vary with GPU load and sampling. The drafter
scheduling metrics (`schedule_reason`, `trained`, `published`) should match
exactly.

Log lines from the reference runs:

```
# SGLang
(SpecoSGLangHttpServer) Decode batch, #running-req: 32, ... accept len: 3.37, accept rate: 0.16, cuda graph: True, gen throughput (token/s): 4456.38
(SpecoWorker) [drafter logits quality] valid_tokens=3833 top1_acc=0.306548 top5_acc=0.559875 ...
(SpecoWorker) [drafter loss] step=8 drafter_step=4 optimizer_step_total=32 samples=4 loss_tokens=7666 lr=1.000e-05 loss=2.3069 vloss=0.0000 ploss=2.3069
(SpecoSGLangHttpServer) [speco sglang draft update] done global_steps=8 buckets=5 bucket_mb=512 elapsed_sec=0.509
(SpecoTaskRunner) Training Progress: 100%|██████████| 8/8 [03:30<00:00, 22.54s/it]

# vLLM
(SpecoVLLMHttpServer) [speco draft ipc] streamed 58 translated weights in 5 buckets into DFlashQwen3Model (method=dflash), ...
(SpecoWorker) [drafter logits quality] valid_tokens=3834 top1_acc=0.225352 top5_acc=0.470266 ...
(SpecoWorker) [drafter loss] step=8 drafter_step=4 optimizer_step_total=32 samples=4 loss_tokens=7668 lr=1.000e-05 loss=2.7830 vloss=0.0000 ploss=2.7830
(SpecoTaskRunner) Training Progress: 100%|██████████| 8/8 [03:22<00:00, 20.85s/it]
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
| SGLang fails on `flashinfer` | verl hardcodes `flashinfer` for SGLang ≥ 0.5.12 | Keep `engine_kwargs.sglang.attention_backend=triton`. |
| `Conflicting visibility of agent-N between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES`, then `Fatal Python error: Aborted` in every GPU worker | A `HIP_VISIBLE_DEVICES` is present alongside Ray's CUDA mask | Unset `HIP_VISIBLE_DEVICES` and use `CUDA_VISIBLE_DEVICES` only. Confirm you are running a verl-SpeCo version that has `rocm_env.py`. |
| Ray `ValueError: Inconsistent values found. Please use either HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES.` | Same as above, inside the SGLang TP subprocess | Same as above. |
| `AssertionError: Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values found: 0 and 4,5.` in verl's `_setup_env_cuda_visible_devices` | Ray 2.57 or newer: on AMD it ignores `CUDA_VISIBLE_DEVICES` and gives each worker a `HIP_VISIBLE_DEVICES` index | Use Ray 2.56.1, as pinned in both images. |
| `ValueError: could not convert string to float: ''` in `ServerArgs.__post_init__` | SGLang's `rocminfo` memory probe returned empty output | Handled by verl-SpeCo's torch fallback. If it still appears, check that `rocminfo` works in the container. |
| SGLang `torch.AcceleratorError: HIP error: unspecified launch failure` at `torch.multinomial` (`sampler.py`), after `HSA_STATUS_ERROR_EXCEPTION`, in the first rollout | `attention_backend=aiter` on SGLang 0.5.12 under verl. It also fails with the drafter disabled and with `free_cache_engine=False`, so it is not specific to verl-SpeCo or to sleep/wake-up | Use `attention_backend=triton`, as in the script. |
| `hipIpcGetMemHandle ... invalid argument`, then abort in `custom_all_reduce.cu` | AITER custom all-reduce | Keep `engine_kwargs.<engine>.disable_custom_all_reduce=True`. |
| `NCCL WARN hipIpcGetMemHandle failed : invalid argument` (`transport/p2p.cc`), then `DistBackendError: ... unhandled cuda error` in `actor_rollout_ref_init_model` | Seen when two containers started with `--network host` initialize RCCL at the same time on one node | Start the container without `--network host`, as in [§3](#3-start-the-container), or start the runs a few minutes apart. |
| Run completes but `drafter/schedule_reason` is `9.0` and `drafter/trained` is `0.0` every step, with `Drop drafter sample: missing or mismatched SGLang hidden positions ... positions=None` | The SGLang build does not return hidden-state positions, so every sample is dropped | Use the SGLang 0.5.12 image. Newer SGLang releases are not supported by this guide yet. |
| SGLang run fails at step ~10–12 with `HIP error: invalid argument` in `_share_cuda_` (weight sync) or `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (actor update); free GPU memory drops by about one model copy per step | ROCm 7.2 does not return CUDA-IPC-exported memory to the device after it is freed, and verl's SGLang weight sync exports fresh tensors every step | See [Limitations](#limitations). Keep SGLang runs to ≤ 8 steps on this image, or use the vLLM image. |
| Drafter fails to load with a missing `config.json` / safetensors, or a broken-link error | Only `snapshots/<rev>` of an HF cache was mounted; its symlinks point into `../../blobs` | Mount the whole `models--<org>--<name>` directory (see [§3](#3-start-the-container)). |
| OOM, `invalid device ordinal`, or a hang at startup | GPUs busy or mask wrong | Check `rocm-smi --showmemuse`, pick 2 idle GPUs, and remove any `NOSET_*` or `HIP_VISIBLE_DEVICES` from the environment. |

## Limitations

- Only **DFlash** has been run end to end on ROCm, on both SGLang and vLLM.
  EAGLE3, P-EAGLE, DFlash2, DSpark and Domino are present in the image but
  unvalidated.
- Drafter training quality is similar on both engines. With the pretrained
  DFlash-b16 drafter, top-1 accuracy at the first block position starts at
  ≈ 0.64–0.77 on both, and the token-averaged top-1 over the block stays at
  ≈ 0.2–0.37 over 8 steps. Rollout acceptance is ~3.1–3.3 on both. Neither engine
  has been compared against a CUDA run of the same configuration.
- **SGLang runs stop after about 10 steps on the SGLang 0.5.12 ROCm image.** The
  image ships torch 2.9.1 on ROCm 7.2. On that stack, GPU memory exported over
  CUDA IPC is not returned to the device after it is freed. verl's SGLang weight
  sync exports a fresh copy of the weights at every step, so about one model's
  worth of device memory is lost per step. With 2 × MI355X and Qwen3-8B, the run
  fails around step 10–12 with `HIP error: invalid argument` in `_share_cuda_` or
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. The vLLM image (ROCm 7.14) is not
  affected, and the vLLM path reuses one exported buffer anyway. Keep SGLang runs
  at `SPECO_TOTAL_TRAINING_STEPS` ≤ 8 on this image.
- The configuration is sized for 2 GPUs: FSDP param and optimizer offload are
  on, and custom all-reduce is off. It is a functional smoke, not a performance
  baseline.
- Rollout TP=2 only, on 1 node or 2 nodes (2 GPUs each, TCP between nodes).
  MI300X (gfx942) is a build target but has not been run.
