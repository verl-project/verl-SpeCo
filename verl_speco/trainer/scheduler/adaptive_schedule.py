# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded startup scheduling and feedback-driven session budgets.

This controller never freezes training. Publication and execution remain owned
by the scheduler lifecycle; convergence policies may veto its training plans.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

from .training_opportunity import startup_active


@dataclass(frozen=True)
class AdaptiveScheduleConfig:
    enable: bool = False
    warmup_max_steps: int = 20
    warmup_window_size: int = 5
    warmup_min_improvement: float = 0.1
    warmup_patience: int = 3
    warmup_train_steps: int = 20
    min_train_steps: int = 0
    max_train_steps: int = 25
    ema_fast_alpha: float = 0.3
    ema_slow_alpha: float = 0.05
    trend_tolerance: float = 0.02
    min_feedback_samples: int = 32
    max_step_change: int = 20

    def __post_init__(self) -> None:
        positive = (
            "warmup_window_size",
            "warmup_patience",
            "warmup_train_steps",
            "max_train_steps",
            "min_feedback_samples",
            "max_step_change",
        )
        for name in positive + ("warmup_max_steps", "min_train_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"adaptive_schedule.{name} must be an integer")
            if value < (1 if name in positive else 0):
                raise ValueError(f"adaptive_schedule.{name} is out of range")
        if self.warmup_window_size < 2:
            raise ValueError("warmup_window_size must be at least 2")
        if (
            isinstance(self.warmup_min_improvement, bool)
            or not math.isfinite(self.warmup_min_improvement)
            or self.warmup_min_improvement < 0
        ):
            raise ValueError("warmup_min_improvement must be finite and nonnegative")
        if self.min_train_steps > self.max_train_steps:
            raise ValueError("min_train_steps must not exceed max_train_steps")
        if not 0 < self.ema_slow_alpha < self.ema_fast_alpha <= 1:
            raise ValueError(
                "adaptive_schedule requires 0 < slow alpha < fast alpha <= 1"
            )
        if not 0 < self.trend_tolerance < 1:
            raise ValueError("trend_tolerance must be in (0, 1)")

    @classmethod
    def from_mapping(cls, value) -> AdaptiveScheduleConfig:
        return cls(**dict(value or {}))

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass(frozen=True)
class AcceptanceFeedback:
    step: int
    sample_count: float
    mean_acceptance_length: float

    def valid(self, config: AdaptiveScheduleConfig) -> bool:
        return (
            math.isfinite(self.sample_count)
            and self.sample_count >= config.min_feedback_samples
            and math.isfinite(self.mean_acceptance_length)
            and self.mean_acceptance_length >= 1.0
        )


@dataclass
class AdaptiveScheduleState:
    budget: int = 0
    fast: float | None = None
    slow: float | None = None
    observations: int = 0
    adaptive_started_step: int = -1
    last_decision_step: int = -1
    last_valid_feedback_step: int = -1
    last_feedback_step: int = -1
    last_train_step: int = -1
    last_publish_step: int = -1
    successful_updates: int = 0
    interval_trend_history: list[list[float]] = field(default_factory=list)
    warmup_acceptance_history: list[float] = field(default_factory=list)
    warmup_no_improvement_count: int = 0
    warmup_ended_after: int | None = None


class AdaptiveScheduleController:
    def __init__(self, config: AdaptiveScheduleConfig) -> None:
        self.config = config
        self.state = AdaptiveScheduleState(
            budget=min(
                config.max_train_steps,
                max(config.min_train_steps, config.warmup_train_steps),
            )
        )
        self.last_interval_trend: float | None = None
        self.last_budget_before = self.state.budget
        self.last_reason = "initial_budget"
        self.feedback_reason = "missing_feedback"

    def warmup_active(self, step: int) -> bool:
        return startup_active(step, self.config, self.state.warmup_ended_after)

    def training_steps(self, step: int) -> int:
        if self.warmup_active(step):
            return self.config.warmup_train_steps
        return self.state.budget

    def relative_trend(self) -> float:
        if self.state.fast is None or self.state.slow is None:
            return 0.0
        return (self.state.fast - self.state.slow) / max(abs(self.state.slow), 1e-8)

    def record_training(self, step: int, successful_steps: int) -> None:
        if successful_steps > 0 and step > self.state.last_train_step:
            self.state.last_train_step = step
            self.state.successful_updates += 1

    def record_publish(self, step: int) -> None:
        self.state.last_publish_step = max(step, self.state.last_publish_step)

    def observe(
        self,
        feedback: AcceptanceFeedback | None,
        *,
        current_step: int,
        training_interval_steps: int = 1,
    ) -> None:
        cfg, state = self.config, self.state
        if not cfg.enable:
            return
        # Keep only this fixed RL interval; invalid/missing steps cannot pull
        # observations from an older interval into the next decision.
        interval = max(int(training_interval_steps), 1)
        interval_start = ((current_step - 1) // interval) * interval
        state.interval_trend_history = [
            row
            for row in state.interval_trend_history
            if interval_start < row[0] <= current_step
        ]
        entering_adaptive = (
            not self.warmup_active(current_step) and state.adaptive_started_step < 0
        )
        if entering_adaptive:
            history = state.warmup_acceptance_history
            baseline = sum(history) / len(history) if history else None
            state.fast = state.slow = baseline
            state.last_decision_step = -1
            state.adaptive_started_step = current_step
            state.interval_trend_history.clear()
        if feedback is None:
            self.feedback_reason = "missing_feedback"
            return
        # Planning may run more than once. Never consume stale/duplicate evidence.
        if feedback.step != current_step or feedback.step <= state.last_feedback_step:
            return
        state.last_feedback_step = feedback.step
        if not feedback.valid(cfg):
            self.feedback_reason = "invalid_or_insufficient_feedback"
            state.warmup_no_improvement_count = 0
            return
        # The current rollout must follow publication of the last successful update.
        if (
            state.last_train_step > state.last_publish_step
            or feedback.step <= state.last_publish_step
        ):
            self.feedback_reason = "awaiting_published_feedback"
            return
        quality = feedback.mean_acceptance_length
        self.feedback_reason = "hold"
        state.last_valid_feedback_step = current_step
        state.observations += 1
        # The transition observation establishes the baseline; subsequent samples
        # measure post-warmup changes without carrying over the warmup EMA gap.
        if entering_adaptive:
            if state.fast is None:
                state.fast = state.slow = quality
            return
        state.fast = (
            quality
            if state.fast is None
            else (cfg.ema_fast_alpha * quality + (1 - cfg.ema_fast_alpha) * state.fast)
        )
        state.slow = (
            quality
            if state.slow is None
            else (cfg.ema_slow_alpha * quality + (1 - cfg.ema_slow_alpha) * state.slow)
        )
        if self.warmup_active(current_step) and state.warmup_ended_after is None:
            # Same convention as rollout mean_acceptance_length: include bonus.
            history = state.warmup_acceptance_history
            history.append(quality)
            del history[: -cfg.warmup_window_size]
            if len(history) == cfg.warmup_window_size:
                gain = history[-1] - history[0]
                state.warmup_no_improvement_count = (
                    state.warmup_no_improvement_count + 1
                    if gain < cfg.warmup_min_improvement
                    else 0
                )
                if state.warmup_no_improvement_count >= cfg.warmup_patience:
                    # Apply next step so collection and training remain consistent.
                    state.warmup_ended_after = current_step
        if self.warmup_active(current_step):
            return
        state.interval_trend_history.append([current_step, self.relative_trend()])

    def decide_budget(
        self, current_step: int, training_interval_steps: int = 1
    ) -> None:
        """Decide once at an opportunity using its complete fixed RL interval."""
        cfg, state = self.config, self.state
        if not cfg.enable or current_step <= state.last_decision_step:
            return
        self.last_budget_before = state.budget
        self.last_interval_trend = None
        state.last_decision_step = current_step
        interval = max(int(training_interval_steps), 1)
        history = state.interval_trend_history
        state.interval_trend_history = []
        if self.warmup_active(current_step):
            self.last_reason = "warmup_budget"
            return
        if current_step == state.adaptive_started_step:
            self.last_reason = "hold"
            return
        values = [
            trend
            for step, trend in history
            if current_step - interval < step <= current_step
        ]
        if len(values) != interval or state.last_valid_feedback_step != current_step:
            self.last_reason = (
                self.feedback_reason
                if state.last_valid_feedback_step != current_step
                and self.feedback_reason != "hold"
                else "invalid_or_insufficient_feedback"
            )
            return
        trend = sum(values) / len(values)
        self.last_interval_trend = trend
        self.last_reason = "hold"
        if abs(trend) <= cfg.trend_tolerance:
            return
        # A 20% relative change uses the full step-change budget.
        # Round up to whole optimizer steps after the tolerance guard.
        severity = min(abs(trend) / 0.2, 1.0)
        delta_steps = math.ceil(severity * cfg.max_step_change)
        change = delta_steps if trend < -cfg.trend_tolerance else -delta_steps
        self.last_reason = "acceptance_drop" if change > 0 else "acceptance_rise"
        state.budget = min(
            cfg.max_train_steps, max(cfg.min_train_steps, state.budget + change)
        )

    def metrics(self, step: int) -> dict[str, float | int]:
        reasons = {
            "initial_budget": 0,
            "missing_feedback": 1,
            "invalid_or_insufficient_feedback": 2,
            "awaiting_published_feedback": 3,
            "hold": 4,
            "acceptance_drop": 5,
            "acceptance_rise": 6,
            "warmup_budget": 7,
            "restored": 8,
        }
        result: dict[str, float | int] = {
            "drafter/adaptive_warmup_active": int(self.warmup_active(step)),
            "drafter/adaptive_valid_observations": self.state.observations,
            "drafter/adaptive_successful_updates": self.state.successful_updates,
            "drafter/adaptive_reason": reasons[self.last_reason],
            "drafter/adaptive_warmup_no_improvement_count": self.state.warmup_no_improvement_count,
            "drafter/adaptive_warmup_early_exit": int(
                self.state.warmup_ended_after is not None
            ),
        }
        if (
            self.state.last_decision_step == step
            and self.last_interval_trend is not None
        ):
            result["drafter/adaptive_interval_trend"] = self.last_interval_trend
        history = self.state.warmup_acceptance_history
        if history:
            result["drafter/adaptive_warmup_acceptance_length"] = history[-1]
        if len(history) == self.config.warmup_window_size:
            result["drafter/adaptive_warmup_gain"] = history[-1] - history[0]
        if self.state.fast is not None and self.state.slow is not None:
            result["drafter/adaptive_acceptance_fast"] = self.state.fast
            result["drafter/adaptive_acceptance_slow"] = self.state.slow
            result["drafter/adaptive_acceptance_trend"] = self.relative_trend()
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": 5,
                    "fingerprint": self.config.fingerprint(),
                    "state": asdict(self.state),
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def restore(self, path: Path) -> bool:
        """Missing/changed-config state recalibrates; malformed state is an error."""
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != 5
            or payload.get("fingerprint") != self.config.fingerprint()
        ):
            return False
        state = AdaptiveScheduleState(**payload["state"])
        for key, value in asdict(state).items():
            if key == "interval_trend_history":
                if not isinstance(value, list) or any(
                    not isinstance(row, list)
                    or len(row) != 2
                    or isinstance(row[0], bool)
                    or not isinstance(row[0], int)
                    or row[0] < 0
                    or row[0] > state.last_valid_feedback_step
                    or isinstance(row[1], bool)
                    or not isinstance(row[1], (int, float))
                    or not math.isfinite(row[1])
                    for row in value
                ):
                    raise ValueError(f"Invalid adaptive checkpoint {key}")
                if any(a[0] >= b[0] for a, b in pairwise(value)):
                    raise ValueError(f"Invalid adaptive checkpoint {key} order")
            elif key == "warmup_acceptance_history":
                if (
                    not isinstance(value, list)
                    or len(value) > self.config.warmup_window_size
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        or v < 1.0
                        for v in value
                    )
                ):
                    raise ValueError(f"Invalid adaptive checkpoint {key}")
            elif key in {"fast", "slow"}:
                if value is not None and (not math.isfinite(value) or value < 1.0):
                    raise ValueError(f"Invalid adaptive checkpoint {key}")
            elif key == "warmup_ended_after" and value is None:
                continue
            else:
                minimum = (
                    -1
                    if key.startswith("last_") or key == "adaptive_started_step"
                    else 0
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < minimum
                ):
                    raise ValueError(f"Invalid adaptive checkpoint {key}")
        if (state.fast is None) != (state.slow is None):
            raise ValueError("Adaptive checkpoint must contain both EMA values")
        if (
            not self.config.min_train_steps
            <= state.budget
            <= self.config.max_train_steps
        ):
            raise ValueError("Adaptive checkpoint budget is outside configured bounds")
        self.state = state
        self.last_reason = "restored"
        return True
