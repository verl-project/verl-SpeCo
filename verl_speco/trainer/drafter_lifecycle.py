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
"""Synchronous Drafter lifecycle boundaries shared by trainer adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _PendingPublish:
    actor_output: Any
    drafter_trained: bool


class SyncDrafterLifecycle:
    """Publish Drafter weights only after Student rollout weights are synchronized."""

    def __init__(
        self,
        *,
        defer_publish_until_rollout_weight_sync: bool,
        publish_drafter: Callable[[bool], dict[str, Any]],
        update_output_metrics: Callable[[Any, dict[str, Any]], Any],
    ) -> None:
        self._defer_publish = bool(defer_publish_until_rollout_weight_sync)
        self._publish_drafter = publish_drafter
        self._update_output_metrics = update_output_metrics
        self._pending_publish: _PendingPublish | None = None

    @property
    def has_pending_publish(self) -> bool:
        return self._pending_publish is not None

    def after_student_update(
        self, *, actor_output: Any, drafter_trained: bool
    ) -> dict[str, Any]:
        """Record or immediately publish a Drafter update."""

        if self._defer_publish and drafter_trained:
            if self._pending_publish is not None:
                raise RuntimeError(
                    "Student update reached Drafter lifecycle before the previous "
                    "Student rollout weight sync"
                )
            self._pending_publish = _PendingPublish(
                actor_output=actor_output,
                drafter_trained=drafter_trained,
            )
            return {}
        return self._publish_drafter(drafter_trained)

    def after_student_rollout_weight_sync(self) -> dict[str, Any]:
        """Publish the pending Drafter update after Student rollout weight sync."""

        pending = self._pending_publish
        if pending is None:
            return {}
        metrics = self._publish_drafter(pending.drafter_trained)
        self._update_output_metrics(pending.actor_output, metrics)
        self._pending_publish = None
        return metrics
