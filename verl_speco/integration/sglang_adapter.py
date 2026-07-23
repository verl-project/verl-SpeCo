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
"""SGLang integration helpers for SPECO.

This module is intentionally explicit: importing it does not monkey patch
SGLang. Call ``install_sglang_speco_patches`` from a SPECO-owned rollout/server
adapter when hidden-state collection is required.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from packaging import version

DRAFTER_SAMPLE_KEY = "drafter_sample"
DRAFTER_RETURN_LAST_HIDDEN_PARAM = "_verl_drafter_return_last_hidden"
DFLASH_RETURN_AUX_HIDDEN_PARAM = "_verl_dflash_return_aux_hidden"
DRAFTER_RAW_TOP_LOGPROBS_PARAM = "_verl_drafter_raw_top_logprobs"
SGLANG_QWEN3_ROPE_COMPAT_PATCH = "qwen3_rope_compat"
SGLANG_EAGLE_UPDATE_WEIGHTS_PATCH = "eagle_update_weights"
SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCH = "hidden_states_tensor_output"
SGLANG_NPU_EAGLE_TARGET_SAMPLING_PATCH = "npu_eagle_target_sampling"


def speco_step_matches_interval(
    global_step: Any, interval_steps: Any, *, default_interval: int = 1
) -> bool:
    """Return whether a positive trainer step should run an interval-gated action."""

    try:
        interval = int(default_interval if interval_steps is None else interval_steps)
    except (TypeError, ValueError):
        return False
    if interval <= 0 or global_step is None:
        return False
    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return False
    return step > 0 and step % interval == 0


@dataclass(frozen=True)
class SGLangSpecoPatchConfig:
    """Configuration for explicitly installing SPECO SGLang runtime patches."""

    set_envs_and_config: Optional[Callable] = None
    target_weight_loader: Optional[str] = None
    draft_weight_loader: Optional[str] = None
    enable_original_logprobs: bool = True
    patches: Optional[Iterable[str]] = None


def install_sglang_speco_patches(
    config: Optional[SGLangSpecoPatchConfig] = None,
) -> None:
    """Install SPECO's SGLang patches.

    The imported patch module touches third-party SGLang runtime objects, so it
    is imported lazily and only when this explicit function is called.
    """

    from verl_speco.integration.sglang_patch import (
        enable_sglang_original_logprob_return,
        install_sglang_verl_patches,
    )

    config = config or SGLangSpecoPatchConfig()
    if config.enable_original_logprobs:
        enable_sglang_original_logprob_return()

    install_kwargs: dict[str, Any] = {
        "set_envs_and_config": config.set_envs_and_config,
        "target_weight_loader": config.target_weight_loader,
        "draft_weight_loader": config.draft_weight_loader,
    }
    if config.patches is not None:
        install_kwargs["patches"] = config.patches
    install_sglang_verl_patches(**install_kwargs)


def sglang_needs_qwen3_rope_compat_patch(sglang_version: Optional[str] = None) -> bool:
    if sglang_version is None:
        try:
            import sglang

            sglang_version = sglang.__version__
        except Exception:  # noqa: BLE001
            return False

    try:
        current_version = version.parse(str(sglang_version))
    except Exception:  # noqa: BLE001
        return False
    return version.parse("0.5.10") <= current_version < version.parse("0.5.12")


def install_sglang_qwen3_rope_compat_patch(
    set_envs_and_config: Optional[Callable] = None,
) -> None:
    """Install only the SGLang Qwen3 rope compatibility patch."""

    from verl_speco.integration.sglang_patch import (
        install_sglang_qwen3_rope_compat_patch as _install,
    )

    _install(set_envs_and_config=set_envs_and_config)


def build_hidden_state_request_params(
    *,
    return_last_hidden: bool = False,
    return_dflash_aux_hidden: bool = False,
    raw_top_logprobs: bool = False,
) -> dict[str, Any]:
    """Build custom SGLang request params used by the migrated hidden-state patch."""

    params: dict[str, Any] = {}
    if return_last_hidden:
        params[DRAFTER_RETURN_LAST_HIDDEN_PARAM] = True
    if return_dflash_aux_hidden:
        params[DFLASH_RETURN_AUX_HIDDEN_PARAM] = True
    if raw_top_logprobs:
        params[DRAFTER_RAW_TOP_LOGPROBS_PARAM] = True
    return params


def normalize_drafter_samples(samples_array: Any) -> list[dict]:
    """Normalize ``drafter_sample`` side-channel payloads into a list of dicts."""

    if samples_array is None:
        return []
    if isinstance(samples_array, dict):
        return [samples_array]
    if hasattr(samples_array, "tolist"):
        raw_samples = samples_array.tolist()
    else:
        raw_samples = list(samples_array)
    return [sample for sample in raw_samples if sample is not None]


def pop_drafter_samples(gen_batch_output: Any) -> list[dict]:
    """Pop SPECO rollout samples from a generation output object."""

    non_tensor_batch = getattr(gen_batch_output, "non_tensor_batch", None)
    if non_tensor_batch is None:
        return []
    samples_array = non_tensor_batch.pop(DRAFTER_SAMPLE_KEY, None)
    return normalize_drafter_samples(samples_array)


def bucket_drafter_samples_by_replica(
    samples: list[dict], num_replicas: int
) -> list[list[dict]]:
    """Bucket normalized samples by rollout replica rank."""

    buckets: list[list[dict]] = [[] for _ in range(num_replicas)]
    for sample in samples:
        replica_rank = sample.get("replica_rank")
        if replica_rank is None:
            raise ValueError("drafter_sample is missing replica_rank for owner routing")
        owner_rank = int(replica_rank)
        if owner_rank < 0 or owner_rank >= num_replicas:
            raise ValueError(
                "drafter_sample replica_rank is out of range for owner routing: "
                f"replica_rank={owner_rank}, num_replicas={num_replicas}"
            )
        buckets[owner_rank].append(sample)
    return buckets
