# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import replace

import pytest

from verl_speco.trainer.scheduler import (
    CallbackDrafterWorkerExecutor,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduler,
    TrainingPlan,
)


def _plan(*, launch: bool = True) -> TrainingPlan:
    return TrainingPlan(
        launch=launch,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=4,
        max_batches=2,
        publish_after_success=True,
        plan_id="plan-4",
        worker_snapshots={"0": {"buffer_version": 1}},
    )


def test_sync_execution_submits_payload_and_blocks_for_results() -> None:
    events = []
    state = DrafterRuntimeState()

    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append(("submit", payload)) or "ref",
            resolve=lambda ref: events.append(("resolve", ref))
            or (
                [{"participating": True, "ready": True, "worker_id": "0"}]
                if ref == "preflight-ref"
                else [{"trained": True}]
            ),
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: "preflight-ref",
            abort_preflight=lambda plan_id: [],
        )
    )
    outcome = scheduler.execute_training_plan(
        _plan(),
        runtime_state=state,
    )

    assert events[0] == ("resolve", "preflight-ref")
    assert events[1][0] == "submit"
    assert events[1][1]["max_batches"] == 2
    assert events[2] == ("resolve", "ref")
    assert outcome.raw_results == [{"trained": True}]
    assert state.status is DrafterRuntimeStatus.RUNNING


def test_sync_execution_marks_runtime_failed() -> None:
    state = DrafterRuntimeState()
    with pytest.raises(RuntimeError, match="rpc failed"):
        DrafterScheduler(
            CallbackDrafterWorkerExecutor(
                submit=lambda payload: "ref",
                resolve=lambda ref: (_ for _ in ()).throw(RuntimeError("rpc failed")),
                inspect_data=lambda sample_last_n_steps, require_full_batch: [],
                prepare=lambda plan: {},
                activate=lambda: [],
                preflight=lambda payload: [
                    {"participating": True, "ready": True, "worker_id": "0"}
                ],
                abort_preflight=lambda plan_id: [],
            )
        ).execute_training_plan(
            _plan(),
            runtime_state=state,
        )
    assert state.status is DrafterRuntimeStatus.FAILED


def test_sync_execution_rejects_inactive_plan() -> None:
    with pytest.raises(ValueError):
        DrafterScheduler(
            CallbackDrafterWorkerExecutor(
                submit=lambda payload: None,
                resolve=lambda ref: None,
                inspect_data=lambda sample_last_n_steps, require_full_batch: [],
                prepare=lambda plan: {},
                activate=lambda: [],
                preflight=lambda payload: [
                    {"participating": True, "ready": True, "worker_id": "0"}
                ],
                abort_preflight=lambda plan_id: [],
            )
        ).execute_training_plan(
            _plan(launch=False),
            runtime_state=DrafterRuntimeState(),
        )


def test_execution_requires_bound_worker_executor() -> None:
    with pytest.raises(RuntimeError, match="has not been bound"):
        DrafterScheduler().execute_training_plan(
            _plan(), runtime_state=DrafterRuntimeState()
        )


def test_prepare_plan_skips_worker_inspection_before_interval() -> None:
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: (
                (_ for _ in ()).throw(AssertionError("unexpected inspection"))
            ),
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [
                {"participating": True, "ready": True, "worker_id": "0"}
            ],
            abort_preflight=lambda plan_id: [],
        )
    )
    context = DrafterScheduleContext(
        global_step=3,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
    )

    plan = scheduler.prepare_training_plan(
        context, DrafterScheduleConfig(training_interval_steps=4)
    )

    assert not plan.launch
    assert plan.reason == "interval_not_reached"


def test_scheduler_validates_training_worker_activation() -> None:
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [{"activated": False, "reason": "activation_failed"}],
            preflight=lambda payload: [
                {"participating": True, "ready": True, "worker_id": "0"}
            ],
            abort_preflight=lambda plan_id: [],
        )
    )

    with pytest.raises(RuntimeError, match="activation failed"):
        scheduler.activate_training_workers()


def test_preflight_failure_aborts_all_workers_without_submitting_training() -> None:
    events = []
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append("submit"),
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [
                {"participating": True, "ready": True, "rank": 0, "worker_id": "0"},
                {
                    "participating": True,
                    "ready": False,
                    "rank": 1,
                    "worker_id": "1",
                    "reason": "buffer_version_changed",
                },
            ],
            abort_preflight=lambda plan_id: events.append(("abort", plan_id)) or [],
        )
    )

    outcome = scheduler.execute_training_plan(
        replace(_plan(), worker_snapshots={"0": {}, "1": {}}),
        runtime_state=DrafterRuntimeState(),
    )

    assert not outcome.launched
    assert outcome.reason == "worker_preflight_failed"
    assert events == [("abort", "plan-4")]


def test_preflight_rejects_actual_data_version_mismatch() -> None:
    events = []
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append("submit"),
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [
                {
                    "participating": True,
                    "ready": True,
                    "worker_id": "0",
                    "data_version": 5,
                    "target_version": 4,
                }
            ],
            abort_preflight=lambda plan_id: events.append(("abort", plan_id)) or [],
        )
    )
    plan = replace(_plan(), data_version=4, required_target_version=4)

    outcome = scheduler.execute_training_plan(
        plan,
        runtime_state=DrafterRuntimeState(),
    )

    assert not outcome.launched
    assert outcome.reason == "worker_preflight_failed"
    assert events == [("abort", "plan-4")]


def test_preflight_allows_actual_target_version_when_not_required() -> None:
    events = []
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append("submit") or [],
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [
                {
                    "participating": True,
                    "ready": True,
                    "worker_id": "0",
                    "data_version": None,
                    "target_version": 4,
                }
            ],
            abort_preflight=lambda plan_id: events.append(("abort", plan_id)) or [],
        )
    )

    outcome = scheduler.execute_training_plan(
        replace(_plan(), required_target_version=None),
        runtime_state=DrafterRuntimeState(),
    )

    assert outcome.launched
    assert outcome.reason == "completed"
    assert events == ["submit"]


def test_preflight_rejects_duplicate_worker_results() -> None:
    events = []
    duplicate = {
        "participating": True,
        "ready": True,
        "worker_id": "0",
        "data_version": None,
        "target_version": None,
    }
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append("submit"),
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [duplicate, dict(duplicate)],
            abort_preflight=lambda plan_id: events.append(("abort", plan_id)) or [],
        )
    )

    outcome = scheduler.execute_training_plan(
        _plan(),
        runtime_state=DrafterRuntimeState(),
    )

    assert not outcome.launched
    assert outcome.reason == "worker_preflight_failed"
    assert events == [("abort", "plan-4")]
