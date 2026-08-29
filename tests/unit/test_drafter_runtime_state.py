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
from __future__ import annotations

import pytest

from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    TrainingPlan,
)


def _plan(*, launch: bool = True) -> TrainingPlan:
    return TrainingPlan(
        launch=launch,
        reason="current_step_samples",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=8,
        max_batches=2,
        publish_after_success=True,
    )


def test_sync_runtime_lifecycle_and_metrics() -> None:
    state = DrafterRuntimeState()
    state.submit(_plan(), started_at=10.0)
    assert state.status is DrafterRuntimeStatus.SUBMITTED
    state.mark_running()
    state.mark_completed(completed_batches=2, elapsed_sec=1.8)

    assert state.status is DrafterRuntimeStatus.COMPLETED
    assert state.metrics() == {
        "drafter/runtime_state": int(DrafterRuntimeStatus.COMPLETED),
        "drafter/runtime_completed_batches": 2,
        "drafter/runtime_inflight": 0,
        "timing_s/drafter_runtime_batch_average": 0.9,
    }

    state.reset()
    assert state.status is DrafterRuntimeStatus.IDLE
    assert state.active_plan is None
    assert state.last_batch_duration_sec == 0.9


def test_runtime_rejects_inactive_plan_and_invalid_transitions() -> None:
    state = DrafterRuntimeState()
    with pytest.raises(ValueError):
        state.submit(_plan(launch=False), started_at=10.0)
    with pytest.raises(RuntimeError):
        state.mark_running()


def test_runtime_records_failure_before_reset() -> None:
    state = DrafterRuntimeState()
    state.submit(_plan(), started_at=10.0)
    state.mark_running()
    state.mark_failed("rpc failed")
    assert state.status is DrafterRuntimeStatus.FAILED
    assert state.last_error == "rpc failed"
    state.reset()
    assert state.status is DrafterRuntimeStatus.IDLE
