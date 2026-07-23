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
import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


_RESUME_OPTIMIZER_STEPS_KEY = "_resume_optimizer_steps"


def _scheduler_last_epoch(optimizer: Optimizer, train_cfg: Any) -> int:
    optimizer_steps = int(train_cfg.get(_RESUME_OPTIMIZER_STEPS_KEY, 0) or 0)
    if optimizer_steps <= 0:
        return -1

    # LRScheduler performs one initial step in __init__. Passing step - 1
    # initializes the newly-created optimizer at the LR for completed `step`.
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])
    return optimizer_steps - 1


class ClampedGlobalCosineLR(LRScheduler):
    """Cosine decay over global successful optimizer steps with a fixed floor."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        decay_steps: int = 100,
        min_lr_ratio: float = 0.1,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        self.decay_steps = int(decay_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_steps = int(warmup_steps)
        if self.decay_steps <= 0:
            raise ValueError(f"lr_decay_steps must be positive, got {self.decay_steps}")
        if self.warmup_steps < 0:
            raise ValueError(
                f"lr_warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.warmup_steps >= self.decay_steps:
            raise ValueError(
                "lr_warmup_steps must be smaller than lr_decay_steps, "
                f"got warmup={self.warmup_steps}, decay={self.decay_steps}"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_ratio(self, step: int) -> float:
        step = max(int(step), 0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)

        decay_span = self.decay_steps - self.warmup_steps
        progress = min(max(step - self.warmup_steps, 0) / decay_span, 1.0)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_ratio

    def get_lr(self) -> list[float]:
        ratio = self._lr_ratio(self.last_epoch)
        return [base_lr * ratio for base_lr in self.base_lrs]


def build_drafter_lr_scheduler(optimizer: Optimizer, train_cfg: Any) -> LRScheduler:
    """Build a drafter scheduler while retaining legacy warmup_style overrides."""

    legacy_style = train_cfg.get("warmup_style", None)
    configured_scheduler_type = train_cfg.get("lr_scheduler_type", None)
    scheduler_type = legacy_style or configured_scheduler_type or "constant"
    scheduler_type = str(scheduler_type).strip().lower()
    default_warmup_steps = 0 if configured_scheduler_type is not None else 1000
    configured_warmup_steps = train_cfg.get("lr_warmup_steps", default_warmup_steps)
    warmup_steps = int(
        default_warmup_steps
        if configured_warmup_steps is None
        else configured_warmup_steps
    )
    last_epoch = _scheduler_last_epoch(optimizer, train_cfg)

    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.0)
        num_cycles = train_cfg.get("num_cycles", 0.5)
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(train_cfg.get("step", 0) or 0),
            min_lr_ratio=float(min_lr_ratio or 0.0),
            num_cycles=float(0.5 if num_cycles is None else num_cycles),
            last_epoch=last_epoch,
        )
    if scheduler_type in {"global_cosine", "clamped_global_cosine"}:
        decay_steps = train_cfg.get("lr_decay_steps", 100)
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.1)
        return ClampedGlobalCosineLR(
            optimizer,
            decay_steps=int(100 if decay_steps is None else decay_steps),
            min_lr_ratio=float(0.1 if min_lr_ratio is None else min_lr_ratio),
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    raise NotImplementedError(f"LR scheduler type {scheduler_type!r} is not supported")
