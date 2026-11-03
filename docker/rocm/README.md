# ROCm (AMD GPU) Dockerfile of verl-SpeCo

This directory provides the Docker recipe for running verl-SpeCo on **AMD GPUs
with the ROCm software stack**. The NVIDIA images under `docker/verl0.*` do not
work on AMD hardware, so use [`Dockerfile.rocm`](Dockerfile.rocm) instead. One
image serves both rollout engines, SGLang and vLLM.

For an end-to-end walkthrough (build, run, expected results), see
[`docs/amd/get_started.md`](../../docs/amd/get_started.md).

## Supported Hardware

- `gfx950`: MI350 series (MI350X / MI355X), validated.
- `gfx942`: MI300 series (MI300X / MI325X), a build target of the base image,
  not run yet.

## Key Versions

| Component | Version |
| --------- | ------- |
| Base image | `amdagi/verl-dev:rocm7.14_torch2.12_release_0724@sha256:919a8a9b462b3e637e3e50bd704b4d987846619ddb35e10b4f15643db179f0e6` |
| ROCm | 7.14 |
| Python | 3.12 |
| PyTorch | 2.12.0+rocm7.14 |
| SGLang | 0.5.12 source @ `127b9e3` (tag `v0.5.12`) |
| sglang-kernel | 0.4.2.post2 |
| vLLM | 0.22.1rc1 source @ `18d87a87d` |
| aiter | ROCm @ `b5e03ed19` |
| verl | 0.9.0.dev @ `7b19203c` |
| ray | 2.56.1 |

## What the Image Contains

The base is verl's ROCm release image, from verl's
[AMD quick start](https://github.com/verl-project/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst)
(built from verl's
[`docker/rocm/Dockerfile.rocm`](https://github.com/verl-project/verl/blob/main/docker/rocm/Dockerfile.rocm)).
On top of it, `Dockerfile.rocm`:

- replaces the base image's SGLang source build with SGLang v0.5.12, the
  release verl-SpeCo supports. It is installed from source the same way as in
  verl's Dockerfile: the ROCm `layernorm.py` fix from sgl-project/sglang#32034,
  the Rust gRPC extension disabled, `pip install --no-deps -e python`. The base
  image's `sglang-kernel` 0.4.2.post2 is the version v0.5.12 pins, so it is
  reused;
- installs verl-SpeCo in editable mode with `--no-deps`.

The base image's torch, vLLM, aiter, verl and environment variables
(`HSA_ENABLE_IPC_MODE_LEGACY=1`, `SGLANG_ATTENTION_BACKEND=triton`, ...) are kept.

## Building Locally

```sh
docker build -f docker/rocm/Dockerfile.rocm -t verl-speco:rocm .
```

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `BASE_IMAGE` | pinned digest above | verl ROCm base image. |
| `SGLANG_TAG` | `127b9e3283f7c2a43234b852ff5c9f1796d53624` (`v0.5.12`) | SGLang source commit. |
