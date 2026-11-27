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
import logging
import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from verl.utils.torch_functional import get_constant_schedule_with_warmup


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
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.1,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        self.num_cycles = num_cycles
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
        if step >= self.decay_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / decay_span
        cosine_ratio = 0.5 * (1.0 + math.cos(2 * math.pi * self.num_cycles * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_ratio

    def get_lr(self) -> list[float]:
        ratio = self._lr_ratio(self.last_epoch)
        return [base_lr * ratio for base_lr in self.base_lrs]


class LinearWarmupDecayLR(LRScheduler):
    """Linear warmup followed by linear decay over successful optimizer steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        decay_steps: int,
        min_lr_ratio: float = 0.0,
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
        linear_ratio = 1.0 - progress
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * linear_ratio

    def get_lr(self) -> list[float]:
        ratio = self._lr_ratio(self.last_epoch)
        return [base_lr * ratio for base_lr in self.base_lrs]


# Older linear and clamped-cosine schedulers serialized these attributes in
# trainer_state.pt. LambdaLR closures did not serialize their parameters.
_PLAN_ATTRIBUTES = {
    "lr_decay_steps": "decay_steps",
    "lr_warmup_steps": "warmup_steps",
    "min_lr_ratio": "min_lr_ratio",
}


def _standalone_decay_budget(train_cfg):
    max_steps = int(train_cfg["_standalone_max_steps"])
    # A single pass has no known optimizer-step count before loading the data.
    return max_steps if max_steps > 0 else 100


def resolve_standalone_lr_plan(train_cfg, saved_plan, scheduler_state):
    """Resolve each field independently: explicit value, checkpoint, then built-in default."""
    defaults = {
        "lr_decay_steps": _standalone_decay_budget(train_cfg),
        "lr_warmup_steps": 0,
        "min_lr_ratio": 0.1,
    }
    plan = {}
    for key, attr in _PLAN_ATTRIBUTES.items():
        saved = saved_plan.get(key, scheduler_state.get(attr))
        explicit = train_cfg.get(key)
        value = explicit if explicit is not None else saved
        convert = float if key == "min_lr_ratio" else int
        plan[key] = convert(defaults[key] if value is None else value)
        if saved is not None and explicit is not None and plan[key] != saved:
            logging.getLogger(__name__).warning(
                "Overriding checkpoint %s from %s to %s; learning rate may jump",
                key,
                saved,
                plan[key],
            )
    return plan


def restore_drafter_scheduler(scheduler, state, *, standalone):
    if standalone:
        # The constructor already applied the merged plan. Restore progress
        # and the checkpoint LR without reverting explicit schedule overrides.
        state = {k: v for k, v in state.items() if k not in _PLAN_ATTRIBUTES.values()}
    scheduler.load_state_dict(state)


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
    if scheduler_type in {"cosine", "global_cosine", "clamped_global_cosine"}:
        legacy_alias = scheduler_type != "cosine"
        decay_steps = train_cfg.get("lr_decay_steps")
        if decay_steps is None:
            if "_standalone_max_steps" in train_cfg:
                decay_steps = _standalone_decay_budget(train_cfg)
            else:
                decay_steps = 100 if legacy_alias else train_cfg.get("step", 0)
        min_lr_ratio = train_cfg.get("min_lr_ratio")
        if min_lr_ratio is None:
            min_lr_ratio = 0.1 if legacy_alias else 0.0
        num_cycles = train_cfg.get("num_cycles")
        return ClampedGlobalCosineLR(
            optimizer,
            decay_steps=int(decay_steps),
            min_lr_ratio=float(min_lr_ratio),
            warmup_steps=warmup_steps,
            num_cycles=float(0.5 if num_cycles is None else num_cycles),
            last_epoch=last_epoch,
        )
    if scheduler_type in {"linear", "linear_decay"}:
        decay_steps = train_cfg.get("lr_decay_steps", train_cfg.get("step", 0))
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.0)
        return LinearWarmupDecayLR(
            optimizer,
            decay_steps=int(100 if decay_steps is None else decay_steps),
            min_lr_ratio=float(0.0 if min_lr_ratio is None else min_lr_ratio),
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    raise NotImplementedError(f"LR scheduler type {scheduler_type!r} is not supported")
