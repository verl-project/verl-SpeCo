# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Run production boundary methods with CPU fakes, without Ray/Torch imports."""

import ast
import asyncio
import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def continuation_driver_class():
    from verl_speco.trainer.scheduler import (
        CollectionFinalizeError,
        DrafterRuntimeStatus,
    )

    return _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_release_completed_rollout_training_lease",
            "_speco_merge_training_metrics",
            "_speco_pop_rollout_idle_metrics",
            "_speco_record_rollout_idle_metrics",
            "_speco_flush_idle_completed_publications",
            "_speco_service_rollout_idle_events",
            "_speco_try_launch_rollout_idle_training",
            "_speco_start_rollout_idle_event_loop",
            "_speco_stop_rollout_idle_event_loop",
        },
        {
            "Any": object,
            "replace": replace,
            "time": __import__("time"),
            "logger": logging.getLogger(__name__),
            "threading": __import__("threading"),
            "CollectionFinalizeError": CollectionFinalizeError,
            "DrafterRuntimeStatus": DrafterRuntimeStatus,
        },
    )


def _ready_idle_driver(cls, *, final=False, reclaimed=False):
    from verl_speco.trainer.scheduler import (
        DrafterRuntimeState,
        TrainingPlan,
        DrafterExecutionStrategy,
    )

    driver = cls()
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=2,
        deadline_ts=__import__("time").time() + 10,
        target_worker_ids=("0",),
        publish_after_success=final,
    )
    state = DrafterRuntimeState()
    state.submit(plan, started_at=0, training_ref="ready")
    state.mark_running()
    outcome = SimpleNamespace(
        trained=True,
        successful_steps=2,
        metrics={
            "drafter/trained": 1,
            "drafter/idle_trained": 1,
            "drafter/train_successful_steps_max": 2,
            "bubble/idle_optimizer_steps": 2,
            "timing_s/drafter_idle_control": 0.02,
            "bubble/train_stop_reclaim_requested": int(reclaimed),
        },
    )
    driver._speco_rollout_training_lease = (plan.plan_id, ("0",))
    driver._speco_get_drafter_runtime_state = lambda: state
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        pending_training_ready=lambda **kwargs: True,
    )
    calls = []

    def poll():
        calls.append("poll")
        state.mark_completed(completed_batches=2, elapsed_sec=0.3)
        driver._speco_rollout_training_lease = None
        return plan, outcome

    driver._speco_poll_pending_drafter_training = poll
    return driver, plan, outcome, state, calls


def test_ready_idle_plan_is_consumed_once_and_releases_running_state(
    continuation_driver_class,
):
    from verl_speco.trainer.scheduler import DrafterRuntimeStatus

    driver, plan, outcome, state, calls = _ready_idle_driver(continuation_driver_class)
    driver._speco_release_completed_rollout_training_lease()
    assert state.status is DrafterRuntimeStatus.IDLE
    assert driver._speco_idle_continuation_plan.deadline_ts == plan.deadline_ts
    assert not getattr(driver, "_speco_idle_completed_publications", None)
    assert driver._speco_release_completed_rollout_training_lease() == {}
    assert calls == ["poll"]
    metrics = driver._speco_pop_rollout_idle_metrics()
    assert metrics["bubble/idle_optimizer_steps"] == 2
    assert metrics["timing_s/drafter_idle_control"] == 0.02
    assert driver._speco_pop_rollout_idle_metrics() == {}


def test_service_retries_without_another_idle_event_after_ready_plan(
    continuation_driver_class,
):
    import threading
    from verl_speco.trainer.scheduler import DrafterRuntimeStatus

    driver, plan, outcome, state, calls = _ready_idle_driver(continuation_driver_class)
    lock = threading.RLock()
    driver._speco_bubble_training_lock = lambda: lock
    driver._speco_drain_rollout_idle_events = lambda: {}

    def launch():
        assert state.status is DrafterRuntimeStatus.IDLE
        assert driver._speco_idle_continuation_plan.deadline_ts == plan.deadline_ts
        calls.append("launch")
        return {}

    driver._speco_try_launch_rollout_idle_training = launch
    driver._speco_service_rollout_idle_events()
    assert calls == ["poll", "launch"]


def test_final_quota_is_queued_not_published_by_background_loop(
    continuation_driver_class,
):
    driver, plan, outcome, state, calls = _ready_idle_driver(
        continuation_driver_class, final=True
    )
    driver._speco_release_completed_rollout_training_lease()
    assert driver._speco_idle_completed_publications == [(plan, outcome)]

    def safe_point(p, result):
        assert p is plan and result is outcome
        calls.append("publish")
        return {"published": 1}

    assert driver._speco_flush_idle_completed_publications(safe_point) == {
        "published": 1
    }
    assert driver._speco_flush_idle_completed_publications(safe_point) == {}
    assert calls == ["poll", "publish"]


def test_reclaimed_plan_cannot_restart_same_window(continuation_driver_class):
    driver, plan, outcome, state, calls = _ready_idle_driver(
        continuation_driver_class, reclaimed=True
    )
    before = __import__("time").time()
    driver._speco_release_completed_rollout_training_lease()
    assert driver._speco_idle_continuation_plan.deadline_ts <= __import__("time").time()
    assert driver._speco_idle_continuation_plan.deadline_ts >= before


def test_final_publish_callback_failure_keeps_snapshot_queued(
    continuation_driver_class,
):
    driver, plan, outcome, state, calls = _ready_idle_driver(
        continuation_driver_class, final=True
    )
    driver._speco_release_completed_rollout_training_lease()

    def fail(*args):
        raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        driver._speco_flush_idle_completed_publications(fail)
    assert driver._speco_idle_completed_publications == [(plan, outcome)]


def test_consumed_outcomes_accumulate_optimizer_counts_and_control_time(
    continuation_driver_class,
):
    driver, plan, outcome, state, calls = _ready_idle_driver(continuation_driver_class)
    driver._speco_release_completed_rollout_training_lease()
    next_plan = replace(plan, plan_id="second", max_batches=3)
    state.submit(next_plan, started_at=0, training_ref="ready-again")
    state.mark_running()
    driver._speco_rollout_training_lease = ("second", ("0",))
    next_outcome = SimpleNamespace(
        trained=True,
        successful_steps=3,
        metrics={
            **outcome.metrics,
            "bubble/idle_optimizer_steps": 3,
            "drafter/train_successful_steps_max": 3,
        },
    )

    def poll():
        state.mark_completed(completed_batches=3, elapsed_sec=0.4)
        driver._speco_rollout_training_lease = None
        return next_plan, next_outcome

    driver._speco_poll_pending_drafter_training = poll
    driver._speco_release_completed_rollout_training_lease()
    metrics = driver._speco_pop_rollout_idle_metrics()
    assert metrics["bubble/idle_optimizer_steps"] == 5
    assert metrics["timing_s/drafter_idle_control"] == pytest.approx(0.04)
    assert metrics["drafter/train_successful_steps_max"] == 3


def test_completed_quota_blocks_relaunch_until_safe_point(continuation_driver_class):
    driver, plan, outcome, state, calls = _ready_idle_driver(
        continuation_driver_class, final=True
    )
    driver._speco_release_completed_rollout_training_lease()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._pending_drafter_publish_refs = None
    driver._speco_drafter_schedule_config = lambda: SimpleNamespace(
        idle_worker_max_pending_publish=1
    )
    assert (
        driver._speco_try_launch_rollout_idle_training()[
            "bubble/training_blocked_by_publish_backpressure"
        ]
        == 1
    )


def test_rollout_return_prevents_further_launch(continuation_driver_class):
    driver = continuation_driver_class()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_rollout_idle_generation_open = False
    assert driver._speco_try_launch_rollout_idle_training() == {
        "bubble/post_generation_launch_suppressed": 1
    }


def test_failed_future_releases_lease_and_remembers_boundary_error(
    continuation_driver_class,
):
    driver, plan, outcome, state, calls = _ready_idle_driver(continuation_driver_class)
    error = RuntimeError("failed future")

    def poll():
        state.mark_failed(error)
        raise error

    driver._speco_poll_pending_drafter_training = poll
    driver._speco_release_rollout_training_lease = lambda p: calls.append("release")
    with pytest.raises(RuntimeError, match="failed future"):
        driver._speco_release_completed_rollout_training_lease()
    assert calls == ["release"]
    assert driver._speco_idle_completion_error is error


def _load_methods(relative_path, class_name, names, namespace):
    path = Path(__file__).resolve().parents[2] / relative_path
    module = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    for method in methods:
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name)
            and decorator.id in {"staticmethod", "classmethod"}
        ]
    selected = ast.ClassDef(
        name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["Subject"]


@pytest.fixture
def worker_class():
    return _load_methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {
            "_bubble_group_control_value",
            "_bubble_optimizer_stop_reason",
            "_bubble_can_stage_snapshot",
            "prepare_hot_drafter_target_head",
            "request_drafter_training_reclaim",
            "_receive_drafter_reclaim",
            "_reclaim_control_lock",
            "train_drafter",
        },
        {
            "asyncio": asyncio,
            "time": __import__("time"),
            "logger": logging.getLogger(__name__),
            "threading": __import__("threading"),
            "_preserve_process_rng_state": lambda device: nullcontext(),
        },
    )


def test_control_collective_uses_only_explicit_groups(worker_class):
    groups = [object(), object()]
    calls = []

    class Tensor:
        def __init__(self, values):
            self.value = values[0]

        def item(self):
            return self.value

    def reduce(tensor, *, op, group):
        assert op == "max" and group in groups
        calls.append(group)
        tensor.value = max(tensor.value, 3)  # A peer requested reclaim.

    globals_ = worker_class._bubble_group_control_value.__globals__
    globals_["dist"] = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_reduce=reduce,
        ReduceOp=SimpleNamespace(MAX="max"),
    )
    globals_["torch"] = SimpleNamespace(
        tensor=lambda values, **kwargs: Tensor(values), int32="int32"
    )
    worker = worker_class()
    worker.trainer = SimpleNamespace(
        _get_sp_group=lambda: groups[0],
        _get_dp_group=lambda: groups[1],
        runtime_device="cpu",
    )
    assert worker._bubble_group_control_value(0) == 3
    assert calls == groups
    calls.clear()
    worker.trainer._get_dp_group = lambda: groups[0]
    worker._bubble_group_control_value(0)
    assert calls == [groups[0]]


@pytest.mark.parametrize(
    "peer_code,reason",
    [
        (1, "deadline_reached"),
        (2, "next_batch_budget_too_small"),
        (3, "reclaim_requested"),
    ],
)
def test_peer_stop_is_obeyed_even_when_local_rank_can_continue(
    worker_class, peer_code, reason
):
    worker = worker_class()
    worker._drafter_reclaim_requested = False
    worker._bubble_group_control_value = lambda code: max(code, peer_code)
    assert worker._bubble_optimizer_stop_reason(None, 0.1) == reason


@pytest.mark.parametrize(
    "remaining,peer_denied,expected",
    [
        (5, False, True),
        (0.1, False, False),
        (5, True, False),
    ],
)
def test_snapshot_staging_requires_budget_on_every_rank(
    worker_class, remaining, peer_denied, expected
):
    worker = worker_class()
    worker._drafter_reclaim_requested = False
    worker._bubble_group_control_value = lambda code: max(code, int(peer_denied))
    plan = {
        "deadline_ts": __import__("time").time() + remaining,
        "idle_snapshot_estimate_sec": 1.0,
    }
    assert worker._bubble_can_stage_snapshot(plan) is expected
    worker._drafter_reclaim_requested = True
    assert not worker._bubble_can_stage_snapshot(plan)


def _prepared_worker(worker_class, *, reclaim_after_first=False):
    worker = worker_class()
    for key, value in {
        "rank": 0,
        "worker_incarnation": "rank-0",
        "enable_drafter": True,
        "in_drafter_train_group": True,
        "is_global_publish_leader": True,
        "is_drafter_group_leader": True,
        "device_name": "cpu",
        "last_global_step": 10,
        "last_trained_step": None,
        "last_trained_step_version": None,
        "_prepared_training_plan_id": "test",
        "_prepared_training_data_version": 10,
        "_prepared_training_target_version": 10,
        "_prepared_training_activation_elapsed_sec": 0.0,
        "_prepared_training_preflight_elapsed_sec": 0.0,
        "_prepared_training_ready_ts": None,
        "_drafter_reclaim_requested": False,
    }.items():
        setattr(worker, key, value)
    worker.last_global_step = 10
    worker._bubble_group_control_value = lambda code: code
    worker._should_keep_drafter_training_hot = lambda *args, **kwargs: True
    worker._capture_training_rng_state = lambda: {"cpu": "canonical-training-rng"}
    callbacks = []

    async def train(step, accumulation, *, can_start_micro_batch):
        callbacks.append(can_start_micro_batch)
        worker.trainer.optimizer_steps_total += 1
        if reclaim_after_first and len(callbacks) == 1:
            asyncio.get_running_loop().call_soon(
                setattr, worker, "_drafter_reclaim_requested", True
            )
        return True

    async def cleanup(*args, **kwargs):
        return None

    worker.trainer = SimpleNamespace(
        optimizer_steps_total=1,
        clear_pending_publish_state_dict=lambda: None,
        reset_training_metrics=lambda: None,
        training_accumulation_step=train,
        get_training_metrics=lambda: {},
        release_training_data_reservation=lambda plan: 0,
        cleanup_training=cleanup,
        get_training_data_status=lambda **kwargs: {"trainable_samples": 20},
        prepare_model_state_dict_for_publish=lambda step: True,
    )
    plan = {
        "plan_id": "test",
        "source_global_step": 10,
        "max_batches": 3,
        "execution_strategy": "rollout_idle_worker",
        "reason": "training_ready",
        "target_worker_ids": ["0"],
        "worker_snapshots": {"0": {"trainable_samples": 20}},
        "gradient_accumulation_steps": 2,
        "idle_batch_estimate_sec": 0.1,
        "publish_after_success": False,
    }
    return worker, plan, callbacks


def test_reclaim_runs_between_atomic_optimizer_steps(worker_class):
    worker, plan, callbacks = _prepared_worker(worker_class, reclaim_after_first=True)
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 1
    assert result["stop_reason"] == "reclaim_requested"
    assert result["replay_session_retained"] == 1
    assert callbacks == [None]


def test_first_idle_optimizer_breakdown_uses_per_plan_deltas(worker_class, caplog):
    worker, plan, callbacks = _prepared_worker(worker_class)
    worker.trainer._training_metric_sums = {"timing_s/drafter_forward_loss": 100.0}
    original = worker.trainer.training_accumulation_step

    async def train(*args, **kwargs):
        worker.trainer._training_metric_sums["timing_s/drafter_forward_loss"] += 0.25
        worker.trainer._training_metric_sums["timing_s/drafter_prepare_batch"] = (
            worker.trainer._training_metric_sums.get(
                "timing_s/drafter_prepare_batch", 0
            )
            + 0.1
        )
        return await original(*args, **kwargs)

    worker.trainer.training_accumulation_step = train
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 3
    assert result["first_optimizer_step_stage_sec"] == {
        "timing_s/drafter_forward_loss": 0.25,
        "timing_s/drafter_prepare_batch": 0.1,
    }
    assert caplog.text.count("optimizer_first_step_breakdown:") == 1
    assert "timing_mode=host_wall_no_device_sync" in caplog.text


def test_all_rollout_groups_finished_skips_launch_before_head_or_data_rpc(
    continuation_driver_class,
):
    import threading
    from verl_speco.trainer.scheduler import DrafterRuntimeState

    driver = continuation_driver_class()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_drafter_schedule_config = lambda: SimpleNamespace(
        idle_worker_max_pending_publish=1
    )
    driver._pending_drafter_publish_refs = None
    driver._speco_get_drafter_runtime_state = lambda: DrafterRuntimeState()
    driver._speco_bubble_training_lock = lambda: threading.RLock()
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        rollout_generation_tail_complete=lambda: True
    )
    driver._speco_poll_target_lm_head_prefetch = lambda: pytest.fail(
        "No head/data/migration RPC should run after the final serving release"
    )
    assert driver._speco_try_launch_rollout_idle_training() == {
        "bubble/rollout_tail_launch_suppressed": 1
    }


def test_idle_yield_services_timer_delivered_reclaim_before_next_step(worker_class):
    worker, plan, callbacks = _prepared_worker(worker_class)
    original_step = worker.trainer.training_accumulation_step

    async def step(*args, **kwargs):
        result = await original_step(*args, **kwargs)
        if len(callbacks) == 1:
            # Incoming I/O may become ready after a sleep(0) continuation has
            # already been queued. A positive timer yield must service it.
            asyncio.get_running_loop().call_later(
                0.0005, worker.request_drafter_training_reclaim, ["0"]
            )
        return result

    worker.trainer.training_accumulation_step = step
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 1
    assert result["stop_reason"] == "reclaim_requested"
    assert callbacks == [None]
    assert result["idle_control_yield_elapsed_sec"] > 0


def test_reclaim_timestamps_ignore_non_writer_and_preserve_first_receive(worker_class):
    worker = worker_class()
    worker.rank = 0
    worker._prepared_training_plan_id = "quota-1"
    worker._drafter_reclaim_requested = False
    assert not worker.request_drafter_training_reclaim(["1"], sent_ts=10)["requested"]
    assert not worker._drafter_reclaim_requested
    assert not hasattr(worker, "_drafter_reclaim_received_perf")
    worker.request_drafter_training_reclaim(["0"], sent_ts=10)
    first = worker._drafter_reclaim_received_perf
    worker.request_drafter_training_reclaim(["0"], sent_ts=11)
    assert worker._drafter_reclaim_received_perf == first


def test_late_control_message_cannot_reclaim_next_plan(worker_class):
    worker = worker_class()
    worker.rank = 0
    worker._active_drafter_training_plan_id = "new"
    worker._drafter_reclaim_requested = False
    assert not worker._receive_drafter_reclaim("old", 123, "lease_bus")
    assert not worker._drafter_reclaim_requested
    assert worker._receive_drafter_reclaim("new", 124, "lease_bus")
    assert worker._drafter_reclaim_requested


def test_driver_control_bus_bypasses_busy_worker_rpc():
    calls = []
    cls = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"speco_request_drafter_training_reclaim"},
        {
            "time": __import__("time"),
            "logger": logging.getLogger(__name__),
            "request_rollout_training_reclaim": lambda *args: calls.append("bus")
            or True,
        },
    )
    driver = cls()
    driver.global_steps = 5
    driver._speco_rollout_idle_event_bus_name = lambda: "bus"
    driver._require_speco_worker_group = lambda: pytest.fail(
        "must not queue behind training"
    )
    assert driver.speco_request_drafter_training_reclaim(("0", "1")) == []
    assert calls == ["bus"]


def test_reclaim_training_reports_local_stop_and_cleanup_latency(worker_class):
    worker, plan, callbacks = _prepared_worker(worker_class)
    worker.request_drafter_training_reclaim(["0"])
    result = asyncio.run(worker.train_drafter(plan))
    assert not callbacks
    assert result["stop_reason"] == "reclaim_requested"
    assert (
        0
        <= result["reclaim_receive_to_stop_sec"]
        <= result["reclaim_receive_to_cleanup_sec"]
    )


def test_completed_idle_control_metrics_survive_no_launch_and_add_topup():
    driver_class = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_merge_training_metrics"},
        {},
    )
    metrics = {
        "drafter/trained": 1,
        "drafter/idle_trained": 1,
        "drafter/train_successful_steps_max": 5,
        "timing_s/drafter_idle_control": 0.12,
        "timing_s/drafter_reclaim_receive_to_stop": 0.3,
        "timing_s/drafter_reclaim_receive_to_cleanup": 0.4,
    }
    driver_class._speco_merge_training_metrics(
        metrics,
        {
            "timing_s/drafter_idle_control": 0.0,
            "drafter/trained": 0,
            "drafter/idle_trained": 0,
            "drafter/train_successful_steps_max": 0,
        },
        {},
    )
    assert metrics["timing_s/drafter_idle_control"] == 0.12
    assert metrics["timing_s/drafter_reclaim_receive_to_stop"] == 0.3
    assert metrics["drafter/trained"] == metrics["drafter/trained_any"] == 1
    assert metrics["drafter/idle_trained"] == 1
    assert metrics["drafter/train_successful_steps_max"] == 5
    driver_class._speco_merge_training_metrics(
        metrics, {}, {"timing_s/drafter_idle_control": 0.08}
    )
    assert metrics["timing_s/drafter_idle_control"] == pytest.approx(0.2)


def test_training_metric_max_is_not_a_sum_of_idle_and_topup():
    cls = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_merge_training_metrics"},
        {},
    )
    metrics = {"drafter/idle_trained": 1, "drafter/train_successful_steps_max": 3}
    cls._speco_merge_training_metrics(
        metrics,
        {"drafter/trained": 1, "drafter/train_successful_steps_max": 9},
        {},
    )
    assert metrics["drafter/train_successful_steps_max"] == 9
    assert metrics["drafter/idle_trained"] == metrics["drafter/trained"] == 1


@pytest.mark.parametrize(
    "resident,optimizer_steps", [(True, 20), (False, 20), (True, 0)]
)
def test_hot_head_prepare_never_reloads_training_state(
    worker_class, resident, optimizer_steps
):
    worker = worker_class()
    worker.rank = 0
    worker.enable_drafter = True
    calls = []
    trainer = SimpleNamespace(
        _training_device_resident=resident,
        optimizer_steps_total=optimizer_steps,
        select_target_lm_head_version=lambda version: calls.append(("select", version))
        or True,
        _target_lm_head_module=lambda: SimpleNamespace(
            weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))
        ),
        device_module=SimpleNamespace(synchronize=lambda: calls.append("fence")),
        _applied_target_lm_head_weight_step=2,
        activate_training_model=lambda: pytest.fail("must not restore model/optimizer"),
    )

    def apply():
        trainer._applied_target_lm_head_weight_step = 4
        calls.append("apply")

    trainer._apply_pending_target_lm_head_weight = apply
    worker.trainer = trainer
    result = worker.prepare_hot_drafter_target_head(("0", "1"), 4)
    assert result["prepared"] == (resident and optimizer_steps > 0)
    if result["prepared"]:
        assert calls == [("select", 4), "apply", "fence"]
    else:
        assert not calls
    calls.clear()
    assert not worker.prepare_hot_drafter_target_head(("1",), 4)["prepared"]
    assert not calls


@pytest.mark.parametrize("snapshot_estimate,staged", [(1.0, True), (100.0, False)])
def test_completed_quota_stages_or_defers_snapshot_without_new_training(
    worker_class, snapshot_estimate, staged
):
    worker, plan, callbacks = _prepared_worker(worker_class)
    plan.update(
        publish_after_success=True,
        publish_snapshot_deferred=True,
        deadline_ts=__import__("time").time() + 30,
        idle_snapshot_estimate_sec=snapshot_estimate,
    )
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 3
    assert result["publish_snapshot_staged_in_idle"] is staged
    assert result["publish_snapshot_deferred"]
    assert result["publish_snapshot_cached"] == int(staged)
    assert (getattr(worker, "_deferred_publish_snapshot_step", None) is None) == staged
    assert callbacks == [None, None, None]


def test_sync_does_not_enter_idle_control(worker_class):
    worker, plan, callbacks = _prepared_worker(worker_class)
    plan["execution_strategy"] = "sync"
    plan.update(min_sample_step=8, max_sample_step=10)

    async def sync_step(step, **bounds):
        callbacks.append(bounds)
        worker.trainer.optimizer_steps_total += 1
        return True

    worker.trainer.training_step = sync_step
    worker.trainer.training_accumulation_step = lambda *args, **kwargs: pytest.fail(
        "Sync must use its original training_step entry point"
    )
    worker._bubble_group_control_value = lambda code: pytest.fail(
        "Sync control changed"
    )
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 3
    assert callbacks == [{"min_sample_step": 8, "max_sample_step": 10}] * 3


@pytest.mark.parametrize(
    "reason", ["quota_topup_training_ready", "quota_forced_completion_ready"]
)
def test_topup_keeps_original_accumulation_path(worker_class, reason):
    worker, plan, callbacks = _prepared_worker(worker_class)
    plan["reason"] = reason
    worker._bubble_group_control_value = lambda value: pytest.fail(
        "top-up must not use idle control"
    )
    result = asyncio.run(worker.train_drafter(plan))
    assert result["successful_steps"] == 3
    assert all(callable(callback) for callback in callbacks)


def test_generation_history_uses_rollout_return_not_reclaim_end():
    from verl_speco.trainer.scheduler import DrafterScheduler

    cls = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_emit_rollout_generation_completed"},
        {
            "Any": object,
            "time": SimpleNamespace(time=lambda: 120.0),
            "logger": logging.getLogger(__name__),
        },
    )
    trainer = cls()
    scheduler = DrafterScheduler()
    scheduler._replica_idle_started_at[0] = 100.0
    trainer._speco_rollout_idle_worker_enabled = lambda: True
    trainer._speco_get_drafter_scheduler = lambda: scheduler
    trainer._speco_get_drafter_runtime_state = lambda: SimpleNamespace(active_plan=None)
    trainer._speco_emit_rollout_generation_completed(None, generation_complete_ts=102.0)
    assert scheduler._replica_last_observed_idle_window_sec[0] == 2.0


@pytest.mark.parametrize("acknowledged,version", [(True, 6), (False, 6), (True, 4)])
def test_safe_point_head_preparation_requires_every_writer_ack(acknowledged, version):
    from verl_speco.trainer.scheduler import DrafterScheduler

    cls = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_prepare_hot_writer_head_at_safe_point"},
        {
            "Any": object,
            "time": __import__("time"),
            "logger": logging.getLogger(__name__),
        },
    )
    scheduler = DrafterScheduler()
    group = ("0", "1")
    scheduler._idle_worker_writer_group = group
    scheduler._idle_worker_hot_prewarmed_groups.add(group)
    scheduler._training_quota_data_version = 6
    scheduler._training_quota_debt_steps = 20
    scheduler._idle_worker_last_target_version[group] = 2
    calls = []

    def prepare(workers, target):
        calls.append((workers, target))
        return [
            {
                "worker_id": "0",
                "prepared": True,
                "target_version": 6,
                "elapsed_sec": 0.4,
            },
            {
                "worker_id": "1",
                "prepared": acknowledged,
                "target_version": version,
                "elapsed_sec": 0.5,
            },
        ]

    trainer = cls()
    trainer.global_steps = 6
    trainer._pending_target_lm_head_sync = None
    trainer._pending_drafter_publish_refs = None
    trainer._speco_rollout_idle_worker_enabled = lambda: True
    trainer._speco_drafter_schedule_config = lambda: SimpleNamespace(
        idle_worker_keep_hot_between_plans=True
    )
    trainer._speco_get_drafter_scheduler = lambda: scheduler
    trainer._speco_get_drafter_runtime_state = lambda: SimpleNamespace(active_plan=None)
    trainer._speco_target_lm_head_ready_for_workers = lambda target, workers: True
    trainer._require_speco_worker_group = lambda: SimpleNamespace(
        prepare_hot_drafter_target_head=prepare
    )
    trainer._ray_get_if_needed = lambda refs: refs
    metrics = trainer._speco_prepare_hot_writer_head_at_safe_point()
    expected = acknowledged and version == 6
    assert metrics["bubble/hot_writer_head_prepared"] == int(expected)
    assert scheduler._idle_worker_last_target_version[group] == (6 if expected else 2)
    if expected:
        assert trainer._speco_prepare_hot_writer_head_at_safe_point() == {}
    assert calls == [(group, 6)]
    trainer._speco_rollout_idle_worker_enabled = lambda: False
    assert trainer._speco_prepare_hot_writer_head_at_safe_point() == {}
    assert calls == [(group, 6)]


def _subgroup_boundary_worker(rank, init_method):
    import datetime

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        writer_group = dist.new_group([0, 1])
        cls = _load_methods(
            "verl_speco/workers/speco_worker.py",
            "SpecoWorker",
            {"_bubble_group_control_value", "_bubble_optimizer_stop_reason"},
            {"torch": torch, "dist": dist, "time": __import__("time")},
        )
        dist.barrier()
        if rank in (0, 1):
            worker = cls()
            worker.trainer = SimpleNamespace(
                _get_sp_group=lambda: writer_group,
                _get_dp_group=lambda: None,
                runtime_device="cpu",
            )
            worker._drafter_reclaim_requested = rank == 1
            assert (
                worker._bubble_optimizer_stop_reason(None, 0.1) == "reclaim_requested"
            )
        # Non-writer ranks never enter the control collective. An accidental
        # WORLD reduction will time out rather than silently pass this test.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_subgroup_stop_does_not_require_other_rollout_ranks(tmp_path):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo distributed backend unavailable")
    torch.multiprocessing.spawn(
        _subgroup_boundary_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=4,
        join=True,
    )


def test_upstream_active_freeze_blocks_bubble_launch_before_data_or_preflight(
    continuation_driver_class,
):
    driver = continuation_driver_class()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_drafter_frozen = True
    # No scheduler, data or executor is installed: the gate must run first.
    assert driver._speco_try_launch_rollout_idle_training() == {
        "bubble/training_blocked_by_freeze": 1
    }


def test_failed_collection_blocks_further_idle_event_service(continuation_driver_class):
    from verl_speco.trainer.scheduler import CollectionFinalizeError, CollectionOutcome

    driver = continuation_driver_class()
    failure = CollectionFinalizeError(
        "partial",
        CollectionOutcome(
            attempted=True,
            collected=False,
            reason="collection_finalize_failed",
            source_global_step=2,
            committed_unfinalized=True,
        ),
    )
    driver._speco_idle_completion_error = failure
    # A fatal transaction must be raised before event drain or launch can run.
    with pytest.raises(CollectionFinalizeError) as caught:
        driver._speco_service_rollout_idle_events()
    assert caught.value is failure


def test_background_finalize_failure_is_not_swallowed(continuation_driver_class):
    from verl_speco.trainer.scheduler import CollectionFinalizeError, CollectionOutcome

    driver = continuation_driver_class()
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_rollout_idle_event_bus_name = lambda: "test-bus"
    driver._speco_rollout_idle_poll_interval_sec = lambda: 0.001
    failure = CollectionFinalizeError(
        "partial",
        CollectionOutcome(
            attempted=True,
            collected=False,
            reason="collection_finalize_failed",
            source_global_step=2,
            committed_unfinalized=True,
        ),
    )

    def fail_service():
        raise failure

    driver._speco_service_rollout_idle_events = fail_service
    stop, thread = driver._speco_start_rollout_idle_event_loop()
    try:
        assert stop.wait(1.0)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert driver._speco_idle_completion_error is failure
    finally:
        driver._speco_stop_rollout_idle_event_loop(stop, thread)


def test_event_thread_timeout_cannot_continue_into_ppo(continuation_driver_class):
    calls = []
    stop = SimpleNamespace(set=lambda: calls.append("stop"))
    thread = SimpleNamespace(
        join=lambda timeout: calls.append(timeout), is_alive=lambda: True
    )
    with pytest.raises(RuntimeError, match="did not drain before PPO"):
        continuation_driver_class._speco_stop_rollout_idle_event_loop(stop, thread)
    assert calls == ["stop", 60.0]
