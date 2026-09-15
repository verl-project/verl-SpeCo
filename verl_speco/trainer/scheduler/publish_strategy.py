# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Execution strategies for drafter weight publication."""

from __future__ import annotations

import time
from dataclasses import dataclass

from verl_speco.trainer.scheduler.publish_executor import DrafterPublishExecutor
from verl_speco.trainer.scheduler.schedule_types import PublishPlan


@dataclass(frozen=True)
class PublishOutcome:
    attempted: bool
    published: bool
    waited_submissions: int = 0
    wait_elapsed_sec: float = 0.0
    fetch_elapsed_sec: float = 0.0
    update_elapsed_sec: float = 0.0

    def metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "drafter/publish_attempted": int(self.attempted),
            "drafter/published": int(self.published),
        }
        if self.attempted:
            metrics.update(
                {
                    "timing_s/drafter_publish_wait_pending": self.wait_elapsed_sec,
                    "timing_s/drafter_publish_fetch_snapshot": self.fetch_elapsed_sec,
                    "timing_s/drafter_publish_update_weights": self.update_elapsed_sec,
                }
            )
        return metrics


class PublishExecutionStrategy:
    def execute(
        self, plan: PublishPlan, *, executor: DrafterPublishExecutor
    ) -> PublishOutcome:
        if not plan.publish:
            return PublishOutcome(attempted=False, published=False)

        started = time.perf_counter()
        waited = executor.wait_pending()
        wait_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        payload = executor.fetch_snapshot()
        fetch_elapsed = time.perf_counter() - started
        if payload is None:
            return PublishOutcome(
                attempted=True,
                published=False,
                waited_submissions=waited,
                wait_elapsed_sec=wait_elapsed,
                fetch_elapsed_sec=fetch_elapsed,
            )

        started = time.perf_counter()
        executor.update_rollout_weights(
            payload,
            global_step=plan.source_global_step,
            asynchronous=plan.asynchronous,
        )
        update_elapsed = time.perf_counter() - started
        return PublishOutcome(
            attempted=True,
            published=True,
            waited_submissions=waited,
            wait_elapsed_sec=wait_elapsed,
            fetch_elapsed_sec=fetch_elapsed,
            update_elapsed_sec=update_elapsed,
        )
