# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROCm/HIP environment normalization shared by the driver and SGLang actors.

Kept in its own light module (only ``os``/``logging`` at import time) so it can run
before the driver imports Ray/vLLM without pulling in heavy dependencies.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _visible_devices(name: str) -> str | None:
    # Matches vllm's _sync_hip_cuda_env_vars(): an empty mask counts as unset.
    return os.environ.get(name) or None


def _is_rocm() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return os.environ.get("HIP_PLATFORM") == "amd"


def neutralize_hip_visible_devices(*, context: str = "") -> bool:
    """Fold HIP_VISIBLE_DEVICES into CUDA_VISIBLE_DEVICES on ROCm so Ray stays CUDA-native.

    vllm's ROCm platform module (``vllm/platforms/rocm.py``) runs a one-time
    ``_sync_hip_cuda_env_vars()`` at import that copies a HIP-only mask into
    CUDA_VISIBLE_DEVICES (and, in the vLLM bundled in the ROCm image, a CUDA-only mask into
    HIP_VISIBLE_DEVICES). If HIP_VISIBLE_DEVICES is still present when Ray initializes, Ray's
    AMD accelerator manager switches to HIP-keyword mode and rewrites only HIP per GPU worker,
    leaving each worker's CUDA_VISIBLE_DEVICES at the full stale mask. The ROCm HIP runtime
    then aborts every GPU worker at ``import torch`` with "Conflicting visibility ... between
    HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES". The same mismatch aborts spawned SGLang TP
    subprocesses at ``import ray`` (``ValueError: Inconsistent values ...``), because each
    respawn re-executes Ray's ``default_worker.py`` before any SGLang code runs.

    The masks are normalized here, independently of whether vllm imports:

    * only HIP_VISIBLE_DEVICES set: copy it to CUDA_VISIBLE_DEVICES, then drop HIP;
    * both set to the same value: drop HIP;
    * both set to different values: raise ``ValueError`` (the intended devices are ambiguous);
    * only CUDA_VISIBLE_DEVICES set, or neither: keep CUDA as is (dropping any HIP copy that
      importing vLLM just added).

    The device restriction is therefore always kept, as CUDA_VISIBLE_DEVICES only, so Ray
    assigns each worker a single physical GPU via CUDA and never introduces a conflicting HIP
    mask. vllm's ROCm platform is still imported first so its import-time sync runs once,
    against the original environment, and cannot re-run later. No-op on non-ROCm builds.

    ``context`` is an optional label (e.g. "in SGLang server actor"); when set and HIP is
    actually removed, that is logged at WARNING so the env fix-up is visible. Returns whether
    HIP_VISIBLE_DEVICES was present and removed.
    """

    if not _is_rocm():
        return False
    hip_mask = _visible_devices("HIP_VISIBLE_DEVICES")
    cuda_mask = _visible_devices("CUDA_VISIBLE_DEVICES")
    if hip_mask is not None and cuda_mask is not None and hip_mask != cuda_mask:
        raise ValueError(
            "Conflicting GPU visibility masks on ROCm: "
            f"HIP_VISIBLE_DEVICES={hip_mask!r} vs CUDA_VISIBLE_DEVICES={cuda_mask!r}. "
            "Set only CUDA_VISIBLE_DEVICES (or make both identical)."
        )
    try:
        import vllm.platforms.rocm  # noqa: F401  # runs the one-time HIP/CUDA sync
    except Exception as exc:  # noqa: BLE001
        logger.debug("vllm ROCm platform import failed during HIP env sync: %s", exc)
    if hip_mask is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_mask
    stripped = os.environ.pop("HIP_VISIBLE_DEVICES", None) is not None
    if stripped and context:
        logger.warning(
            "Moved HIP_VISIBLE_DEVICES=%r to CUDA_VISIBLE_DEVICES %s to keep spawned "
            "workers CUDA-native.",
            hip_mask or "",
            context,
        )
    return stripped


def _ray_amd_visible_devices_env_var() -> str | None:
    try:
        from ray._private.accelerators.amd_gpu import AMDGPUAcceleratorManager

        return AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
    except Exception as exc:  # noqa: BLE001
        logger.debug("cannot query Ray's AMD visible-devices keyword: %s", exc)
        return None


def align_visible_devices_with_ray() -> bool:
    """Hand the driver's CUDA_VISIBLE_DEVICES mask to Ray as HIP_VISIBLE_DEVICES when Ray needs it.

    Call in the driver after :func:`neutralize_hip_visible_devices` and before Ray starts.
    Ray <= 2.56 keys AMD GPUs on CUDA_VISIBLE_DEVICES when HIP_VISIBLE_DEVICES is unset, so the
    CUDA-only mask is kept. Newer Ray (2.60 in the ROCm 7.14 primus base) always keys AMD GPUs on
    HIP_VISIBLE_DEVICES: it ignores a CUDA mask when sizing the GPU pool and sets only HIP per GPU
    worker, so an inherited CUDA mask conflicts with it and the HIP runtime aborts every GPU worker
    at ``import torch``. In that case the mask moves to HIP_VISIBLE_DEVICES (CUDA dropped) for Ray
    and its workers; verl's Worker folds the per-worker HIP id back into CUDA_VISIBLE_DEVICES.
    Returns whether the mask was moved. No-op on non-ROCm builds or without a CUDA mask.
    """

    if not _is_rocm():
        return False
    cuda_mask = _visible_devices("CUDA_VISIBLE_DEVICES")
    if cuda_mask is None or _visible_devices("HIP_VISIBLE_DEVICES") is not None:
        return False
    if _ray_amd_visible_devices_env_var() != "HIP_VISIBLE_DEVICES":
        return False
    os.environ["HIP_VISIBLE_DEVICES"] = cuda_mask
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    logger.warning(
        "Ray keys AMD GPUs on HIP_VISIBLE_DEVICES; moved CUDA_VISIBLE_DEVICES=%r to "
        "HIP_VISIBLE_DEVICES for the Ray cluster.",
        cuda_mask,
    )
    return True
