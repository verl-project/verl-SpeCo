# ROCm (AMD GPU) Dockerfile of verl-SpeCo

This directory provides the Docker recipe for running verl-SpeCo on **AMD GPUs
with the ROCm software stack**. The NVIDIA images under `docker/verl0.*` do not
work on AMD hardware, so use [`Dockerfile.rocm`](Dockerfile.rocm) instead. One
image serves both rollout engines, SGLang and vLLM.

For an end-to-end walkthrough (build, run, expected results), see
[`docs/amd/get_started.md`](../../docs/amd/get_started.md).

## Supported Hardware

- `gfx950`: MI350 series (MI350X / MI355X), validated.
- `gfx942`: MI300 series (MI300X / MI325X), a build target (`PYTORCH_ROCM_ARCH`),
  not run yet.

## Key Versions

| Component | Version |
| --------- | ------- |
| Base image | `rocm/primus:v26.4@sha256:8c8ecc6fe14b5061423cc82517831e3515f6eb15647ee6d746fd6a0c1963da24` |
| ROCm | 7.14 |
| Python | 3.12 (`/opt/venv`) |
| PyTorch | 2.12.0+rocm7.14 |
| SGLang | 0.5.12 source @ `127b9e3` (tag `v0.5.12`) |
| sglang-kernel | 0.4.2.post2, built from SGLang's `sgl-kernel` source |
| vLLM | 0.26.0 source @ `568afb3a1` (tag `v0.26.0`; reports `0.26.1.dev0+g568afb3a1`) |
| xgrammar | 0.2.1 |
| transformers | 5.6.2 |
| aiter | 0.1.12.post2.dev214 @ `b5e03ed19` (from the base image) |
| verl | 0.9.0.dev @ `7b19203c` |
| megatron-core | 0.18.0 |
| ray | 2.60.0 |

## What the Image Contains

`Dockerfile.rocm` is verl's
[`docker/rocm/Dockerfile.rocm`](https://github.com/verl-project/verl/blob/5eb4826694f35d8453ad465b67e3e02cb6b34f07/docker/rocm/Dockerfile.rocm)
(see verl's
[AMD quick start](https://github.com/verl-project/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst)),
built from its own base, `rocm/primus` (torch, aiter and ray come from there).
It does not depend on a published verl image. On top of verl's file it:

- builds vLLM v0.26.0 instead of 0.22.1rc1, with verl's ROCm patches, and
  installs xgrammar 0.2.1 (vLLM v0.26.0 requires >= 0.2.1). The patches leave
  the source tree dirty, so vLLM reports `0.26.1.dev0+g568afb3a1`;
- builds SGLang v0.5.12, the release verl-SpeCo supports, instead of verl's
  newer source commit, the same way verl does: the ROCm `layernorm.py` fix from
  sgl-project/sglang#32034, the Rust gRPC extension disabled,
  `pip install --no-deps -e python`, and `sglang-kernel` 0.4.2.post2 built from
  its `sgl-kernel` source for `gfx942` and `gfx950`;
- installs transformers 5.6.2. SGLang v0.5.12 pins 5.6.0, whose
  `flash_attention_forward` calls `s_aux.to()` on `None` and breaks every FSDP
  actor forward;
- installs `ipython` and `gguf`, which SGLang imports at engine start and the
  `--no-deps` install skips;
- puts `_rocm_sdk_core/lib` first on `LD_LIBRARY_PATH`. Otherwise SGLang's
  `torch_memory_saver` `LD_PRELOAD` loads a second LLVM and the scheduler
  aborts with `Option 'spirv-expand-step' registered more than once`;
- installs verl-SpeCo in editable mode with `--no-deps`.

verl's environment variables are kept (`HSA_ENABLE_IPC_MODE_LEGACY=1`,
`SGLANG_ATTENTION_BACKEND=triton`, `SGLANG_USE_AITER=0`, ...). SGLang's `aiter`
attention backend fails its JIT build in this image, so use `triton`.

## Building Locally

```sh
docker build -f docker/rocm/Dockerfile.rocm -t verl-speco:rocm .
```

A full build takes about 36 minutes on a 256-core node.

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `BASE_IMAGE` | pinned `rocm/primus:v26.4` digest above | ROCm/torch base image. |
| `VLLM_TAG` | `568afb3a13806beb53bb2e6bd518269357b237c0` (`v0.26.0`) | vLLM source commit. |
| `VERL_TAG` | `7b19203c52503760f92e0368067c7eb744236274` | verl source commit. |
| `SGLANG_TAG` | `127b9e3283f7c2a43234b852ff5c9f1796d53624` (`v0.5.12`) | SGLang source commit. |
