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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/scheduler.py.
import math

SUPPORTED_SCHEDULERS = {
    "cosine": "WarmupCosineLR",
    "linear": "WarmupDecayLR",
}


def resolve_warmup_steps(*, total_steps: int, warmup_ratio: float) -> int:
    """Convert a positive warmup ratio to at least one optimizer step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            f"train.warmup_ratio must be finite and in [0, 1), got {warmup_ratio!r}"
        )
    if warmup_ratio == 0.0:
        return 0
    if total_steps < 2:
        raise ValueError(
            "A positive train.warmup_ratio requires at least two optimizer "
            f"steps, got total_steps={total_steps}"
        )
    return min(total_steps - 1, math.ceil(total_steps * warmup_ratio))


def build_deepspeed_scheduler_config(
    scheduler: str | None,
    *,
    total_steps: int,
    warmup_steps: int,
    learning_rate: float,
) -> dict | None:
    """Build the explicit DeepSpeed warmup-and-decay scheduler config."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps == 0:
        return None
    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be smaller than total_steps when a scheduler "
            f"is enabled, got {warmup_steps}/{total_steps}"
        )

    normalized = "cosine" if scheduler is None else str(scheduler).strip().lower()
    scheduler_type = SUPPORTED_SCHEDULERS.get(normalized)
    if scheduler_type is None:
        choices = ", ".join(sorted(SUPPORTED_SCHEDULERS))
        raise ValueError(
            f"train.scheduler must be one of [{choices}], got {scheduler!r}"
        )

    params = {
        "total_num_steps": total_steps,
        "warmup_num_steps": warmup_steps,
        "warmup_type": "linear",
    }
    if normalized == "linear":
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "deepspeed.optimizer.params.lr must be a positive finite "
                f"number for the linear scheduler, got {learning_rate!r}"
            )
        # WarmupDecayLR otherwise defaults to 1e-3 rather than the optimizer LR.
        params["warmup_min_lr"] = 0.0
        params["warmup_max_lr"] = learning_rate

    return {
        "type": scheduler_type,
        "params": params,
    }
