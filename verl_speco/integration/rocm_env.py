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


def neutralize_hip_visible_devices(*, context: str = "") -> bool:
    """Strip HIP_VISIBLE_DEVICES on ROCm so Ray stays CUDA-native.

    vllm's ROCm platform module (``vllm/platforms/rocm.py``) runs a one-time
    ``_sync_hip_cuda_env_vars()`` at import that copies CUDA_VISIBLE_DEVICES into
    HIP_VISIBLE_DEVICES. If that HIP value is present when Ray initializes, Ray's AMD
    accelerator manager switches to HIP-keyword mode and rewrites only HIP per GPU worker,
    leaving each worker's CUDA_VISIBLE_DEVICES at the full stale mask. The ROCm HIP runtime
    then aborts every GPU worker at ``import torch`` with "Conflicting visibility ... between
    HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES". The same mismatch aborts spawned SGLang TP
    subprocesses at ``import ray`` (``ValueError: Inconsistent values ...``), because each
    respawn re-executes Ray's ``default_worker.py`` before any SGLang code runs.

    Pre-triggering the sync (so its module body is cached and cannot re-run) and then popping
    HIP_VISIBLE_DEVICES leaves the process -- and every subprocess it spawns -- with
    CUDA_VISIBLE_DEVICES only, so Ray assigns each worker a single physical GPU via CUDA and
    never introduces a conflicting HIP mask. No-op on non-ROCm builds.

    ``context`` is an optional label (e.g. "in SGLang server actor"); when set and a value is
    actually stripped, that is logged at WARNING so the env fix-up is visible. Returns whether
    HIP_VISIBLE_DEVICES was present and removed.
    """

    try:
        import torch

        is_rocm = bool(getattr(torch.version, "hip", None))
    except Exception:
        is_rocm = os.environ.get("HIP_PLATFORM") == "amd"
    if not is_rocm:
        return False
    try:
        import vllm.platforms.rocm  # noqa: F401  # runs the one-time HIP/CUDA sync
    except Exception as exc:  # noqa: BLE001
        logger.debug("vllm ROCm platform import failed during HIP env sync: %s", exc)
    stripped = os.environ.pop("HIP_VISIBLE_DEVICES", None) is not None
    if stripped and context:
        logger.warning(
            "Stripped HIP_VISIBLE_DEVICES %s to keep spawned workers CUDA-native.", context
        )
    return stripped
