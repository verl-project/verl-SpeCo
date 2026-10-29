# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Shared, driver-owned Bubble boundary and quota accounting primitives."""

from __future__ import annotations

import threading
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuotaCostLedger:
    """Count completed work once; parallel rank durations use their maximum."""

    events: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def record(self, identity: str, seconds: float) -> None:
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("Bubble costs must be finite and nonnegative")
        self.events.setdefault(identity, float(seconds))

    def record_interval(self, identity: str, start: float, end: float) -> None:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid Bubble cost interval")
        if identity not in self.events:
            self.record(identity, end - start)
            self.intervals[identity] = (start, end)

    @property
    def total(self) -> float:
        total = sum(
            value for key, value in self.events.items() if key not in self.intervals
        )
        end = float("-inf")
        for start, stop in sorted(self.intervals.values()):
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
        return total

    def observe_training(self, plan: Any, outcome: Any) -> None:
        records = getattr(outcome, "raw_results", ()) or ()

        def flatten(items):
            for item in items:
                if isinstance(item, dict):
                    yield item
                elif isinstance(item, (list, tuple)):
                    yield from flatten(item)

        records = list(flatten(records))
        # elapsed_sec is worker service time, not driver future/queue latency.
        for phase, duration_key in (
            ("train", "elapsed_sec"),
            ("prepare", "preflight_elapsed_sec"),
        ):
            critical: dict[str, Any] = max(
                records,
                key=lambda item: float(item.get(duration_key, 0) or 0),
                default={},
            )
            duration = float(critical.get(duration_key, 0) or 0)
            if not records and phase == "train":
                duration = float(getattr(outcome, "elapsed_sec", 0) or 0)
            start = critical.get(f"cost_{phase}_started_ts")
            stop = critical.get(f"cost_{phase}_finished_ts")
            if start is not None and stop is not None:
                self.record_interval(
                    f"{phase}:{plan.plan_id}", float(start), float(stop)
                )
            else:
                self.record(f"{phase}:{plan.plan_id}", duration)


class BubbleLifecycle:
    """Boundary orchestration shared by the legacy and V1 adapters.

    Scheduler/worker implementations retain ownership of admission and leases.
    Closing a boundary never publishes: publication belongs to the target-sync
    transaction, after the caller has quiesced the appropriate serving engines.
    """

    def __init__(self, trainer: Any):
        self.trainer = trainer
        if trainer is not None:
            defaults: dict[str, Any] = {
                "_pending_drafter_publish_contexts": [],
                "_speco_published_drafter_version": None,
                "_speco_last_published_drafter_step": 0,
                "_pending_drafter_publish_step": None,
                "_speco_ready_target_lm_head_versions": set(),
                "_speco_ready_target_lm_head_workers": {},
                "_speco_last_known_trainable_batches": None,
                "_speco_last_collected_step": None,
                "_speco_bubble_lock": threading.RLock(),
                "_speco_rollout_completed_training_metrics": {},
                "_speco_idle_completed_publications": [],
                "_speco_idle_continuation_plan": None,
                "_speco_rollout_idle_generation_open": False,
                "_speco_idle_completion_error": None,
                "_speco_last_rollout_idle_metrics": {},
                "_speco_direct_sample_ids": set(),
                "_speco_direct_collected_samples": 0,
                "_speco_runtime_reclaim_requested_plan_id": None,
                "_speco_rollout_training_lease": None,
                "_speco_runtime_idle_callback_verified": False,
                "_speco_runtime_idle_events_this_generation": 0,
                "_speco_runtime_sample_events_this_generation": 0,
                "_speco_deferred_publish_count": 0,
                "_speco_freeze_state_sidecar": "speco_drafter_freeze_state.json",
            }
            for name, value in defaults.items():
                trainer.__dict__.setdefault(name, value)
        self.stop_event: Any = None
        self.thread: Any = None
        self.pending: list[tuple[Any, Any]] = []
        self.queued: set[str] = set()
        self.published_ids: set[str] = set()
        self.costs = QuotaCostLedger()
        self.receipts: list[dict[str, Any]] = []
        self.lock = threading.RLock()

    def enqueue(self, plan: Any, outcome: Any) -> dict[str, int]:
        with self.lock:
            if plan.plan_id in self.published_ids:
                return {"bubble/duplicate_completion_ignored": 1}
            self.costs.observe_training(plan, outcome)
            quota_complete = (
                self.trainer is None
                or self.trainer._speco_get_drafter_scheduler()._training_quota_debt_steps
                <= 0
            )
            if (
                outcome.trained
                and plan.publish_after_success
                and quota_complete
                and plan.plan_id not in self.queued
            ):
                self.pending.append((plan, outcome))
                self.queued.add(plan.plan_id)
        return {"bubble/completion_queued": int(bool(self.pending))}

    def start(self) -> None:
        owner = self.trainer
        with self.lock:
            if self.thread is not None or getattr(
                owner, "_speco_validation_in_progress", False
            ):
                return
            owner._speco_rollout_idle_generation_open = True
            owner._speco_idle_continuation_plan = None
            owner._speco_runtime_idle_events_this_generation = 0
            owner._speco_runtime_sample_events_this_generation = 0
            owner._speco_record_rollout_idle_metrics(
                owner._speco_emit_rollout_generation_started()
            )
            self.stop_event, self.thread = owner._speco_start_rollout_idle_event_loop()

    def close(self, boundary: str) -> dict[str, Any]:
        owner = self.trainer
        generation_was_open = owner._speco_rollout_idle_generation_open
        owner._speco_rollout_idle_generation_open = False
        owner._speco_stop_rollout_idle_event_loop(self.stop_event, self.thread)
        self.stop_event = self.thread = None
        diagnostic = getattr(owner, "_speco_v1_boundary_diagnostic", None)
        if diagnostic is not None:
            diagnostic("reclaim_begin", boundary)
        # A failed join raises before scheduler mutation or PPO can continue.
        with owner._speco_bubble_training_lock():
            metrics = owner._speco_service_rollout_idle_events(allow_launch=False)
            if generation_was_open:
                metrics.update(owner._speco_emit_rollout_generation_completed(None))
            metrics.update(
                owner._speco_reclaim_rollout_idle_workers(
                    boundary=boundary,
                    on_completed=self.enqueue,
                    force_drain=True,
                )
            )
            metrics.update(owner._speco_flush_idle_completed_publications(self.enqueue))
            metrics.update(owner._speco_park_drafter_training_memory(boundary))
            if diagnostic is not None:
                diagnostic("reclaim_complete", boundary)
            return metrics

    def published(self, plan_id: str) -> None:
        with self.lock:
            if not self.pending or self.pending[0][0].plan_id != plan_id:
                raise RuntimeError("Bubble publication queue changed during commit")
            self.pending.pop(0)
            self.published_ids.add(plan_id)
            self.queued.discard(plan_id)
            self.costs = QuotaCostLedger()
