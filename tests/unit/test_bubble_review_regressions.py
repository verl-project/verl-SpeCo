# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise review fixes through production methods without device dependencies."""

import ast
import asyncio
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    IdleWindowConfidence,
    TrainingDataStatus,
)


def _methods(path, class_name, names, **namespace):
    path = Path(__file__).resolve().parents[2] / path
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n
        for n in original.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    for method in methods:
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        ]
    body = ast.ClassDef(
        name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            body,
        ],
        type_ignores=[],
    )
    namespace.setdefault("logger", logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Subject"]


def _sampling_trainer():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sample_training_items", "_sample_replay_items"},
        random=random,
    )
    obj = cls()
    obj._active_training_target_version = 4
    obj._active_training_replay_cursor = 0
    obj.optimizer_steps_total = 0
    obj._sampling_accumulation_steps = 2
    obj._sampling_micro_index = 0
    obj._is_block_drafter_backend = lambda: True
    obj._block_drafter_metric_prefix = lambda: "dspark"
    obj._block_drafter_config_value = (
        lambda key, default: 0.1 if key == "hard_sample_ratio" else default
    )
    return obj


@pytest.mark.parametrize("runner", [None, "0", "1"])
@pytest.mark.parametrize("confidence_alpha", [0.0, 1.0])
def test_merged_dspark_publish_filters_only_untrained_confidence_head(
    monkeypatch, runner, confidence_alpha
):
    if runner is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_is_frozen_publish_param"},
    )
    worker = cls()
    worker.backend = SimpleNamespace(
        model_type="dspark", confidence_head_alpha=confidence_alpha
    )
    worker._frozen_param_names = []
    for name in (
        "draft_model.confidence_head.proj.weight",
        "draft_model.confidence_head.proj.bias",
    ):
        assert worker._is_frozen_publish_param(name) == (confidence_alpha == 0.0)
    assert not worker._is_frozen_publish_param("draft_model.fc.weight")
    worker._frozen_param_names = ["confidence_head."]
    assert worker._is_frozen_publish_param("draft_model.confidence_head.proj.weight")


def _draw_micro_batch(obj, pool):
    items = obj._sample_replay_items(pool, 2)
    if obj._sampling_micro_index == 1:
        obj.optimizer_steps_total += 1
    obj._sampling_micro_index = 1 - obj._sampling_micro_index
    return items


def test_quota_replay_keeps_hard_sample_fraction_across_accumulated_steps():
    obj = _sampling_trainer()
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    batches = [_draw_micro_batch(obj, pool) for _ in range(40)]
    assert sum(item["_verl_is_hard"] for batch in batches for item in batch) == 8
    assert obj._active_training_replay_cursor == 80
    assert len(pool) == 20


def test_replay_sampling_continues_identically_after_short_plan_handoff():
    uninterrupted = _sampling_trainer()
    split = _sampling_trainer()
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    expected = [_draw_micro_batch(uninterrupted, pool) for _ in range(12)]
    actual = [_draw_micro_batch(split, pool) for _ in range(6)]
    resumed = _sampling_trainer()
    resumed._active_training_replay_cursor = split._active_training_replay_cursor
    resumed.optimizer_steps_total = split.optimizer_steps_total
    actual.extend(_draw_micro_batch(resumed, pool) for _ in range(6))
    assert actual == expected


def test_hard_fractional_quota_does_not_restart_rounding_each_cycle():
    obj = _sampling_trainer()
    obj._block_drafter_config_value = (
        lambda key, default: 0.03 if key == "hard_sample_ratio" else default
    )
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    hard = 0
    for _ in range(3):
        obj._active_training_replay_cursor = 0
        for _ in range(40):
            hard += sum(item["_verl_is_hard"] for item in _draw_micro_batch(obj, pool))
    assert hard == 7  # round(3 cycles * 20 optimizer steps * 4 samples * .03)


@pytest.mark.parametrize("bubble,fsdp", [(True, True), (False, True), (True, False)])
def test_fsdp1_bubble_uses_full_model_norm_and_sync_retains_its_path(bubble, fsdp):
    calls = []

    class FakeFSDP:
        def parameters(self):
            return ["local shard"]

        def clip_grad_norm_(self, *, max_norm):
            calls.append(("full_norm", max_norm))
            return 5.0

    def local_clip(parameters, *, max_norm):
        calls.append(("local_norm", max_norm))
        return 4.0

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_clip_training_grad_norm"},
        FSDP=FakeFSDP,
        torch=SimpleNamespace(
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=local_clip))
        ),
    )
    obj = cls()
    obj._bubble_time_enabled = bubble
    obj.model = (
        FakeFSDP() if fsdp else SimpleNamespace(parameters=lambda: ["parameter"])
    )
    expected = "full_norm" if bubble and fsdp else "local_norm"
    assert obj._clip_training_grad_norm() == (5.0 if expected == "full_norm" else 4.0)
    assert calls == [(expected, 1.0)]


@pytest.mark.parametrize("fail", [False, True])
def test_cold_bubble_cleanup_requires_parking_and_preserves_training_progress(fail):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py", "DrafterBaseTrainer", {"cleanup_training"}
    )
    obj = cls()
    obj._bubble_time_enabled = True
    obj._pending_checkpoint_future = None
    obj._pending_full_checkpoint_future = None
    obj._full_checkpoint_executor = None
    obj.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None)
    obj.training_steps = 17
    obj.optimizer_steps_total = 17
    calls = []

    async def park(*, strict):
        calls.append(strict)
        if fail:
            raise RuntimeError("model offload failed")

    obj.release_training_memory_after_activation = park
    if fail:
        with pytest.raises(RuntimeError, match="offload failed"):
            asyncio.run(obj.cleanup_training(clear_data=False, keep_hot=False))
    else:
        asyncio.run(obj.cleanup_training(clear_data=False, keep_hot=False))
    assert calls == [True]
    assert obj.training_steps == obj.optimizer_steps_total == 17


def test_actor_timing_and_feedback_source_are_idempotent_across_publish_merges():
    key = "_speco_drafter_timing_deducted_from_update_actor"
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_update_output_metrics"},
        _speco_metric_float=lambda v: None if v is None else float(v),
        _SPECO_DRAFTER_TIMING_DEDUCTED_KEY=key,
    )
    obj = cls()
    obj.global_steps = 7
    feedback_steps = []
    obj._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        record_step_metrics=lambda metrics,
        config,
        *,
        global_step: feedback_steps.append(global_step) or {}
    )
    obj._speco_drafter_schedule_config = lambda: DrafterScheduleConfig()
    output = SimpleNamespace(
        meta_info={
            "metrics": {
                "timing_s/update_actor": 10,
                "timing_s/drafter": 2,
                "timing_per_token_ms/update_actor": 1,
            }
        }
    )
    obj._speco_update_output_metrics(output, {"drafter/trained": 1})
    obj.global_steps = 8
    obj._speco_update_output_metrics(output, {"drafter/published": 1})
    assert output.meta_info["metrics"]["timing_s/update_actor"] == 8
    assert output.meta_info["metrics"]["timing_per_token_ms/update_actor"] == 0.8
    assert feedback_steps == [7, 7]


def test_one_rollout_cannot_trigger_slowdown_patience_twice():
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_gen_slowdown_patience=2,
    )
    scheduler.record_step_metrics(
        {"timing_per_token_ms/gen": 1.0}, config, global_step=1
    )
    slow = {"timing_per_token_ms/gen": 2.0, "drafter/idle_trained": 1}
    scheduler.record_step_metrics(slow, config, global_step=2)
    cap = scheduler._effective_idle_dynamic_batch_cap(config)
    scheduler.record_step_metrics(slow, config, global_step=2)
    assert scheduler._idle_worker_gen_slowdown_streak == 1
    assert scheduler._effective_idle_dynamic_batch_cap(config) == cap
    scheduler.record_step_metrics(slow, config, global_step=3)
    assert scheduler._idle_worker_gen_slowdown_streak == 2


@pytest.mark.parametrize("buffered", [False, True])
def test_sync_requires_successful_current_oldlogprob_collection(buffered):
    context = DrafterScheduleContext(
        global_step=4,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=True,
        data_status=TrainingDataStatus(
            current_step=4,
            current_step_samples=0,
            buffer_samples=8,
            trainable_samples=8,
            trainable_batches=4,
            batch_size_per_gpu=2,
            partial_batch_available=False,
            oldest_sample_step=2,
            newest_sample_step=2,
            same_step_data_required=False,
        ),
    )
    plan = DrafterScheduler().plan_training(
        context,
        DrafterScheduleConfig(training_interval_steps=4, use_data_buffer=buffered),
    )
    assert not plan.launch
    assert plan.reason == "no_current_step_oldlogprob_samples"


@pytest.mark.parametrize("use_logits", [False, True])
def test_sync_plan_requires_current_head_not_pre_sync_status(use_logits):
    # Status is queried before the driver exports this step's actor head.
    # A successful previous Sync leaves target_version one step behind.
    context = DrafterScheduleContext(
        global_step=2,
        training_mode="online",
        collected_samples_this_step=64,
        oldlogprob_collection_requested=True,
        data_status=TrainingDataStatus(
            current_step=2,
            current_step_samples=64,
            buffer_samples=64,
            trainable_samples=64,
            trainable_batches=8,
            batch_size_per_gpu=8,
            partial_batch_available=False,
            oldest_sample_step=2,
            newest_sample_step=2,
            same_step_data_required=False,
            target_version=1,
            data_version=2,
        ),
    )
    plan = DrafterScheduler().plan_training(
        context,
        DrafterScheduleConfig(training_interval_steps=1, use_logits=use_logits),
    )
    assert plan.launch
    assert plan.required_target_version == (None if use_logits else 2)
    assert plan.data_version == 2


@pytest.mark.parametrize("group", [("0",), ("0", "2"), ("0", "1", "2", "3")])
def test_explicit_idle_groups_cannot_split_or_combine_runtime_collectives(group):
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    config = DrafterScheduleConfig(idle_worker_training_groups=(group,))
    with pytest.raises(ValueError, match="complete runtime training collectives"):
        scheduler._idle_training_groups(config)


def test_explicit_runtime_group_is_canonicalized_without_changing_topology():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    assert scheduler._idle_training_groups(
        DrafterScheduleConfig(idle_worker_training_groups=(("3", "2"),))
    ) == (("2", "3"),)


@pytest.mark.parametrize("bubble", [True, False])
def test_target_head_version_cache_is_bubble_only(bubble):
    class Tensor:
        shape = (3, 2)

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"sync_target_lm_head_weight"},
        torch=SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor)),
    )
    obj = cls()
    obj._bubble_time_enabled = bubble
    obj._target_lm_head_snapshots = {}
    obj._target_lm_head_snapshot_limit = 2
    for step in range(3):
        obj.sync_target_lm_head_weight(
            Tensor(), global_step=step, defer_device_apply=True
        )
    assert set(obj._target_lm_head_snapshots) == ({1, 2} if bubble else set())


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "stale", "phase", "empty"]
)
def test_bubble_publish_requires_complete_same_version_acknowledgements(
    staged, failure
):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_validate_bubble_publish_acks"},
    )
    obj = cls()
    obj.actor_rollout_wg = SimpleNamespace(world_size=4)
    obj._speco_get_drafter_scheduler = lambda: DrafterScheduler()
    version_key = "staged_version" if staged else "published_version"
    acks = [
        {"worker_rank": rank, version_key: 4, "staged": staged, "published": not staged}
        for rank in range(4)
    ]
    if failure == "missing":
        acks.pop()
    elif failure == "duplicate":
        acks[-1]["worker_rank"] = 0
    elif failure == "stale":
        acks[-1][version_key] = 3
    elif failure == "phase":
        acks[-1]["staged"] = not staged
        acks[-1]["published"] = staged
    elif failure == "empty":
        acks = []
    if failure:
        with pytest.raises(RuntimeError, match="Bubble publish ACK"):
            obj._speco_validate_bubble_publish_acks(acks, 4, staged=staged)
    else:
        obj._speco_validate_bubble_publish_acks(acks, 4, staged=staged)


def test_metadata_discovery_failure_cannot_fall_back_to_guessed_bubble_groups():
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_register_drafter_training_resource_metadata"},
    )
    obj = cls()
    obj.drafter_wg = object()
    obj._speco_rollout_idle_worker_enabled = lambda: True

    def fail_metadata():
        raise RuntimeError("worker metadata unavailable")

    obj.speco_get_drafter_training_resource_metadata = fail_metadata
    with pytest.raises(
        RuntimeError, match="Unsafe or unavailable Bubble training resource metadata"
    ):
        obj._speco_register_drafter_training_resource_metadata()


@pytest.mark.parametrize("module_name", ["sglang_runtime", "vllm_runtime"])
def test_runtime_recognizes_legacy_bubble_strategy(module_name):
    from importlib import import_module

    module = import_module(f"verl_speco.integration.{module_name}")
    config = {
        "training": {
            "execution_strategy": " ROLLOUT_IDLE_WORKER ",
            "scheduler": {"idle_worker": {"event_bus_name": "bus"}},
        }
    }
    assert module._rollout_idle_worker_config(config)["event_bus_name"] == "bus"


def test_inconsistent_updated_writer_shards_cannot_be_retried_as_zero_progress():
    from verl_speco.trainer.scheduler import TrainingOutcome, TrainingPlan

    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 20
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=4,
        max_batches=2,
        publish_after_success=False,
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1,
        reason="worker_result_inconsistent",
        metrics={},
    )
    with pytest.raises(RuntimeError, match="Inconsistent Bubble training state"):
        scheduler._record_training_outcome(plan, outcome)
    assert scheduler._training_quota_debt_steps == 20
    assert scheduler._idle_optimizer_steps_total == 0


def test_main_sample_bounds_survive_status_aggregation_and_plan_payload():
    from verl_speco.trainer.scheduler import ConservativeTrainingDataStatusPolicy

    values = {
        "current_step": 4,
        "current_step_samples": 2,
        "buffer_samples": 4,
        "trainable_samples": 4,
        "trainable_batches": 2,
        "batch_size_per_gpu": 2,
        "partial_batch_available": False,
        "oldest_sample_step": 2,
        "newest_sample_step": 4,
        "same_step_data_required": False,
        "min_sample_step": 2,
        "max_sample_step": 4,
        "worker_id": "0",
    }
    status = TrainingDataStatus.from_mapping(values)
    aggregate = ConservativeTrainingDataStatusPolicy().aggregate(
        [status], global_step=4
    )
    assert aggregate.min_sample_step == 2 and aggregate.max_sample_step == 4
    context = DrafterScheduleContext(
        global_step=4,
        training_mode="online",
        collected_samples_this_step=2,
        oldlogprob_collection_requested=False,
        data_status=aggregate,
    )
    plan = DrafterScheduler().plan_training(
        context, DrafterScheduleConfig(training_interval_steps=4)
    )
    assert plan.to_worker_payload()["min_sample_step"] == 2
    assert plan.to_worker_payload()["max_sample_step"] == 4


@pytest.mark.parametrize("local_flags", [(True, True), (True, False)])
def test_bubble_batch_readiness_uses_writer_group_not_missing_dp(local_flags):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2, timeout=3)
    tensors = []
    groups = []

    class Scalar:
        def __init__(self, values, **kwargs):
            self.value = values[0]

        def item(self):
            return self.value

    def reduce(tensor, *, op, group):
        tensors.append(tensor)
        groups.append(group)
        barrier.wait()
        value = min(item.value for item in tensors)
        barrier.wait()
        tensor.value = value

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
        time=time,
        torch=SimpleNamespace(tensor=Scalar, int32="int32"),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MIN="min")),
    )
    workers = []
    for _ in range(2):
        worker = cls()
        worker._bubble_time_enabled = True
        worker.runtime_device = "cpu"
        worker.record_training_timing = lambda name, elapsed: None
        worker._full_training_group = lambda: "writer-group"
        worker._full_training_world_size = lambda: 2
        # replica-local initialization has no DP group. It must not be used.
        worker._get_dp_group = lambda: None
        worker._get_dp_world_size = lambda: 1
        workers.append(worker)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(w._sync_batch_readiness, flag)
            for w, flag in zip(workers, local_flags, strict=True)
        ]
        results = [future.result(timeout=5) for future in futures]
    assert results == [all(local_flags)] * 2
    assert groups == ["writer-group", "writer-group"]


def test_sync_readiness_does_not_add_writer_collective():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
    )
    worker = cls()
    worker._bubble_time_enabled = False
    worker._get_dp_group = lambda: None
    worker._get_dp_world_size = lambda: 1
    assert worker._sync_batch_readiness(True)
    assert not worker._sync_batch_readiness(False)


def test_sync_readiness_keeps_dp_group_and_no_bubble_metrics():
    calls = []

    class Scalar:
        def __init__(self, values, **kwargs):
            self.value = values[0]

        def item(self):
            return self.value

    def reduce(tensor, *, op, group):
        calls.append(group)
        tensor.value = 0

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
        torch=SimpleNamespace(tensor=Scalar, int32="int32"),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MIN="min")),
    )
    worker = cls()
    worker._bubble_time_enabled = False
    worker.runtime_device = "cpu"
    worker._get_dp_group = lambda: "sync-dp"
    worker._get_dp_world_size = lambda: 2
    worker._full_training_group = lambda: pytest.fail(
        "Sync must not use the Bubble writer group"
    )
    worker.record_training_timing = lambda *args: pytest.fail(
        "Sync must not record Bubble readiness timing"
    )
    assert not worker._sync_batch_readiness(True)
    assert calls == ["sync-dp"]


@pytest.mark.parametrize("ready", [True, False])
def test_sync_step_preserves_preparation_timing_and_sample_bounds(ready):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_training_step_impl"},
        time=time,
    )
    worker = cls()
    worker.model = object()
    worker._bubble_time_enabled = False
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={"collect_hidden_states_from_old_logprob": True}
            )
        )
    )
    events = []

    def prepare(**bounds):
        events.append(("prepare", bounds))
        return object()

    worker.prepare_training_batch = prepare
    worker._prepare_training_batch_for_step = lambda **kwargs: pytest.fail(
        "Sync must not enter the Bubble preparation helper"
    )
    worker.record_training_timing = lambda name, elapsed: events.append(
        ("timing", name)
    )
    worker._sync_batch_readiness = (
        lambda flag: events.append(("readiness", flag)) or ready
    )

    async def train(batch, step):
        events.append(("train", step))
        return True

    worker._training_step_on_batch = train
    worker._consume_last_training_batch = lambda: events.append(("consume", None))
    assert (
        asyncio.run(worker._training_step_impl(4, min_sample_step=2, max_sample_step=3))
        is ready
    )
    assert events[:3] == [
        ("prepare", {"min_sample_step": 2, "max_sample_step": 3}),
        ("timing", "timing_s/drafter_prepare_batch"),
        ("readiness", True),
    ]
    assert events[3:] == ([("train", 4), ("consume", None)] if ready else [])


@pytest.mark.parametrize("local_error", [False, True])
def test_bubble_preparation_failure_participates_in_readiness(local_error):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_prepare_training_batch_for_step"},
    )
    worker = cls()
    worker._bubble_time_enabled = True
    flags = []

    def prepare():
        if local_error:
            raise MemoryError("rank-local preparation failure")
        return object()

    worker.prepare_training_batch = prepare
    worker._sync_batch_readiness = lambda flag: flags.append(flag) or False
    assert worker._prepare_training_batch_for_step() is None
    assert flags == [not local_error]
    if local_error:
        assert worker.last_training_error["type"] == "MemoryError"
    else:
        assert getattr(worker, "last_training_error", None) is None


def _status_trainer(
    *, bubble=False, use_logits=True, model_type="dspark", use_buffer=True
):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"get_training_data_status", "_filter_training_data_by_step"},
        torch=SimpleNamespace(is_tensor=lambda value: False),
    )
    worker = cls()
    samples = [
        {"step": version, "target_version": version, "loss_tokens": 32}
        for version in (2, 3, 4)
        for _ in range(4)
    ]
    worker.data_buffer = SimpleNamespace(
        get_all_data=lambda: list(samples),
        get_available_data=lambda **kwargs: list(samples),
        get_data_from_last_n_steps=lambda n: [s for s in samples if s["step"] >= 4 - n],
    )
    worker._bubble_time_enabled = bubble
    worker._target_lm_head_snapshots = {2: object(), 3: object(), 4: object()}
    worker._target_lm_head_weight_step = 9
    worker.current_rl_step = 4
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"use_logits": use_logits})
        )
    )
    worker.backend = SimpleNamespace(model_type=model_type)
    worker.use_data_buffer = use_buffer
    worker.collected_data = list(samples)
    worker.batch_size = 4
    worker.buffer_version = 1
    return worker


@pytest.mark.parametrize("use_logits", [False, True])
def test_sync_status_preserves_full_cross_step_pool(use_logits):
    worker = _status_trainer(use_logits=use_logits)
    status = worker.get_training_data_status(
        sample_last_n_steps=2, require_full_batch=True
    )
    assert status["trainable_samples"] == 12
    assert status["trainable_batches"] == 3
    assert (status["min_sample_step"], status["max_sample_step"]) == (2, 4)
    assert status["target_version"] == 9


@pytest.mark.parametrize("use_buffer", [False, True])
def test_sync_status_honors_explicit_sample_bounds(use_buffer):
    worker = _status_trainer(use_buffer=use_buffer)
    status = worker.get_training_data_status(min_sample_step=3, max_sample_step=3)
    assert status["trainable_samples"] == 4
    assert status["data_version"] == 3
    assert (status["min_sample_step"], status["max_sample_step"]) == (3, 3)


@pytest.mark.parametrize("use_buffer", [False, True])
def test_sync_batch_sampling_honors_the_same_status_bounds(use_buffer):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_prepare_training_batch", "_filter_training_data_by_step"},
        random=random,
    )
    worker = cls()
    status_worker = _status_trainer(use_buffer=use_buffer)
    worker.__dict__.update(status_worker.__dict__)
    data = status_worker.data_buffer

    class Buffer:
        def __len__(self):
            return len(data.get_all_data())

        def get_all_data(self):
            return data.get_all_data()

    worker.data_buffer = Buffer()
    worker._active_training_reservation_id = None
    worker.training_steps = 0
    selected = []

    class SelectionReached(Exception):
        pass

    def sample(items, batch_size, rng):
        selected.extend(items)
        raise SelectionReached()

    worker._sample_training_items = sample
    with pytest.raises(SelectionReached):
        worker._prepare_training_batch(min_sample_step=3, max_sample_step=3)
    assert len(selected) == 4
    assert {sample["step"] for sample in selected} == {3}


def test_sync_eagle_without_logits_keeps_current_step_only():
    worker = _status_trainer(use_logits=False, model_type="eagle3")
    status = worker.get_training_data_status(sample_last_n_steps=2)
    assert status["trainable_samples"] == 4
    assert status["min_sample_step"] == 4


@pytest.mark.parametrize("target_version, expected", [(None, 4), (3, 3)])
def test_bubble_status_keeps_fixed_single_version(target_version, expected):
    worker = _status_trainer(bubble=True)
    # Explicit Sync bounds must not override a retained Bubble version.
    status = worker.get_training_data_status(
        target_version=target_version, min_sample_step=2, max_sample_step=2
    )
    assert status["trainable_samples"] == 4
    assert status["target_version"] == expected
    assert status["min_sample_step"] == expected


@pytest.mark.parametrize("accumulation_steps", [2, 4, 6])
def test_streamed_accumulation_keeps_one_batch_and_sampling_order(accumulation_steps):
    import weakref
    from contextlib import nullcontext

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "training_accumulation_step",
            "_prepare_training_batch_for_step",
            "_sample_replay_items",
            "_sample_training_items",
        },
        torch=SimpleNamespace(enable_grad=nullcontext),
        random=random,
        time=time,
    )
    worker = cls()
    sampling = _sampling_trainer()
    worker.__dict__.update(sampling.__dict__)
    worker._bubble_time_enabled = True
    timings = []
    worker.record_training_timing = lambda key, value: timings.append((key, value))
    worker._sampling_accumulation_steps = accumulation_steps
    pool = [{"id": i, "_verl_is_hard": i < 5} for i in range(accumulation_steps * 4)]
    expected = []
    for index in range(accumulation_steps):
        sampling._sampling_accumulation_steps = accumulation_steps
        sampling._sampling_micro_index = index
        available = [
            s for s in pool if s["id"] not in {v for batch in expected for v in batch}
        ]
        expected.append([s["id"] for s in sampling._sample_replay_items(available, 2)])

    class Batch:
        pass

    live = weakref.WeakSet()
    sequence = []
    zeros = []
    consumed = []

    def prepare():
        assert len(live) == 0, (
            "previous device batch must be released before preparing the next"
        )
        available = [
            s for s in pool if id(s) not in worker._active_accumulation_item_ids
        ]
        worker._last_prepared_training_items = worker._sample_replay_items(available, 2)
        batch = Batch()
        batch.samples = list(worker._last_prepared_training_items)
        live.add(batch)
        return batch

    async def forward(batch, step, *, accumulation_steps, accumulation_index):
        assert len(live) == 1
        sequence.append([s["id"] for s in batch.samples])
        random.random()  # Forward RNG must not perturb deterministic replay sampling.
        if accumulation_index == accumulation_steps - 1:
            worker.optimizer_steps_total += 1
        return True

    worker.prepare_training_batch = prepare
    worker._sync_batch_readiness = lambda ready: ready
    worker.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: zeros.append(1))
    worker._training_step_on_batch = forward
    worker._consume_training_items = lambda items: consumed.extend(items)
    assert asyncio.run(worker.training_accumulation_step(4, accumulation_steps))
    assert sequence == expected
    assert worker.optimizer_steps_total == 1
    assert len(consumed) == 2 * accumulation_steps
    assert len({id(s) for s in consumed}) == len(consumed)
    assert len(live) == 0
    assert worker._active_accumulation_item_ids == set()
    assert zeros == [1]
    assert len(timings) == accumulation_steps
    assert all(
        key == "timing_s/drafter_prepare_batch" and value >= 0 for key, value in timings
    )


def test_streamed_accumulation_discards_partial_gradients_on_peer_failure():
    from contextlib import nullcontext

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"training_accumulation_step", "_prepare_training_batch_for_step"},
        torch=SimpleNamespace(enable_grad=nullcontext),
        time=time,
    )
    worker = cls()
    worker._bubble_time_enabled = True
    worker.record_training_timing = lambda *args: None
    worker.last_training_error = {"type": "StaleError", "message": "prior plan"}
    worker._last_prepared_training_items = [{"id": 1}]
    worker.prepare_training_batch = lambda: object()
    ready_calls = []
    worker._sync_batch_readiness = (
        lambda flag: ready_calls.append(flag) or len(ready_calls) == 1
    )
    gradients = []
    worker.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: gradients.clear())

    async def forward(*args, **kwargs):
        gradients.append(1)
        return True

    worker._training_step_on_batch = forward
    worker._consume_training_items = lambda items: pytest.fail(
        "failed step must not consume samples"
    )
    assert not asyncio.run(worker.training_accumulation_step(4, 2))
    assert len(ready_calls) == 2
    assert gradients == []
    assert worker._active_accumulation_item_ids == set()
    assert worker.last_training_error is None


@pytest.mark.parametrize("mode", ["poll", "wait"])
def test_bubble_async_completion_preserves_main_ci_evidence(mode):
    from verl_speco.integration.ci_result import CIResult

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_poll_pending_drafter_training",
            "_speco_wait_pending_drafter_training",
        },
    )
    driver = cls()
    plan = SimpleNamespace(
        source_global_step=4,
        plan_id="quota-4",
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
    )
    outcome = SimpleNamespace(
        trained=True,
        successful_steps=2,
        elapsed_sec=0.5,
        reason="trained",
        raw_results=[{"errors": [{"type": "TestError", "message": "test evidence"}]}],
        metrics={"drafter/trained": 1, "drafter/train_successful_steps_max": 2},
    )
    results = [outcome, None]

    def complete(**kwargs):
        return results.pop(0)

    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        poll_pending_training=complete, wait_pending_training=complete
    )
    driver._speco_get_drafter_runtime_state = lambda: SimpleNamespace(active_plan=plan)
    driver._speco_ci_result = CIResult(training=True)
    released = []
    driver._speco_release_rollout_training_lease = lambda p: released.append(p.plan_id)
    method = (
        driver._speco_poll_pending_drafter_training
        if mode == "poll"
        else driver._speco_wait_pending_drafter_training
    )
    assert method() == (plan, outcome)
    assert method() == (plan, None)
    assert released == ["quota-4"]
    assert driver._speco_ci_result.data["optimizer_steps"] == 2
    assert driver._speco_ci_result.data["errors"] == outcome.raw_results[0]["errors"]


def _freeze_publish_driver(*, bubble=True, asynchronous=True, failure=None):
    from typing import Any, cast

    from verl_speco.trainer.drafter_freeze_policy import DrafterFreezePolicy
    from verl_speco.trainer.scheduler import AfterWeightUpdateContext

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_publish_drafter_weights",
            "_speco_update_rollout_drafter_weights",
            "_speco_wait_pending_drafter_publish_rpc",
            "_speco_validate_bubble_publish_acks",
            "_speco_freeze_observe_publish",
        },
        time=time,
        cast=cast,
        Any=Any,
        AfterWeightUpdateContext=AfterWeightUpdateContext,
        DrafterExecutionStrategy=DrafterExecutionStrategy,
    )
    driver = cls()
    driver.global_steps = 4
    driver._speco_freeze_policy = DrafterFreezePolicy({"enabled": True})
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver._speco_drafter_schedule_config = DrafterScheduleConfig
    driver._speco_freeze_publish_cost_seconds = lambda metrics: None
    driver._pending_drafter_publish_refs = None
    driver._pending_drafter_publish_contexts = []
    driver._speco_record_writer_switch_decision = lambda: None
    driver._ray_get_if_needed = lambda value: value
    driver.actor_rollout_wg = SimpleNamespace(world_size=2)
    stages = []
    applied = []
    completed = []

    def acks(version, *, staged=False):
        records = [
            {
                "worker_id": str(rank),
                "staged": staged,
                "published": not staged,
                "staged_version" if staged else "published_version": version,
            }
            for rank in range(2)
        ]
        if staged and failure == "stage":
            records[1]["staged_version"] = version - 1
        if not staged and failure == "apply":
            records[1]["published"] = False
        return records

    def rpc(method):
        def update(payload, *, global_steps):
            if failure == "submit":
                raise RuntimeError("submit failed")
            staged = method == "stage_draft_weights_async"
            stages.append(method)
            return acks(global_steps, staged=staged)

        return update

    def commit(version):
        if failure == "commit_rpc":
            raise RuntimeError("commit RPC failed")
        applied.append(version)
        return acks(version)

    driver._speco_actor_rollout_method = rpc
    driver.speco_commit_staged_drafter_weights = commit
    scheduler = SimpleNamespace(
        _publish_executor=object(),
        _flatten_publish_acknowledgements=DrafterScheduler._flatten_publish_acknowledgements,
        record_rollout_drafter_publish_completed=lambda version,
        **kwargs: completed.append(version),
        record_training_quota_publish_completed=lambda step: None,
    )

    def publish(context):
        driver._speco_update_rollout_drafter_weights(
            {}, context.global_step, asynchronous
        )
        return SimpleNamespace(
            publish_plan=SimpleNamespace(
                asynchronous=asynchronous, source_global_step=context.global_step
            ),
            publish_outcome=SimpleNamespace(published=True),
            metrics={"drafter/published": 1},
        )

    scheduler.on_safe_point = publish
    driver._speco_get_drafter_scheduler = lambda: scheduler
    plan = SimpleNamespace(
        execution_strategy=(
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            if bubble
            else DrafterExecutionStrategy.SYNC
        ),
        plan_id="quota-1",
        target_worker_ids=("0", "1"),
    )
    return driver, plan, stages, applied, completed


def test_bubble_freeze_advances_once_only_after_valid_apply_ack():
    driver, plan, stages, applied, completed = _freeze_publish_driver()
    driver._speco_publish_drafter_weights(True, plan)
    assert stages == ["stage_draft_weights_async"]
    assert driver._speco_freeze_policy.drafter_version == 0
    assert applied == [] and completed == []
    assert driver._speco_wait_pending_drafter_publish_rpc() == 2
    assert applied == [4] and completed == [4]
    assert driver._speco_freeze_policy.drafter_version == 1
    assert driver._speco_wait_pending_drafter_publish_rpc() == 0
    assert driver._speco_freeze_policy.drafter_version == 1


@pytest.mark.parametrize("failure", ["submit", "stage", "apply", "commit_rpc"])
def test_bubble_freeze_does_not_advance_on_publish_failure(failure):
    driver, plan, _, _, completed = _freeze_publish_driver(failure=failure)
    with pytest.raises(RuntimeError):
        driver._speco_publish_drafter_weights(True, plan)
        assert driver._speco_freeze_policy.drafter_version == 0
        driver._speco_wait_pending_drafter_publish_rpc()
    assert driver._speco_freeze_policy.drafter_version == 0
    assert completed == []


def test_bubble_direct_publish_advances_freeze_after_valid_apply():
    driver, plan, stages, applied, completed = _freeze_publish_driver(
        asynchronous=False
    )
    driver._speco_publish_drafter_weights(True, plan)
    assert stages == ["update_draft_weights"]
    assert completed == [4]
    assert applied == []
    assert driver._speco_freeze_policy.drafter_version == 1


def test_bubble_failed_direct_apply_does_not_advance_freeze():
    driver, plan, _, _, completed = _freeze_publish_driver(
        asynchronous=False, failure="apply"
    )
    with pytest.raises(RuntimeError, match="Invalid Bubble publish ACK"):
        driver._speco_publish_drafter_weights(True, plan)
    assert driver._speco_freeze_policy.drafter_version == 0
    assert completed == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_publish_keeps_original_outer_freeze_observer(asynchronous):
    driver, plan, stages, applied, completed = _freeze_publish_driver(
        bubble=False, asynchronous=asynchronous
    )
    metrics = driver._speco_publish_drafter_weights(True, plan)
    if asynchronous:
        driver._speco_wait_pending_drafter_publish_rpc()
    assert driver._speco_freeze_policy.drafter_version == 0
    assert stages == [
        "update_draft_weights_async" if asynchronous else "update_draft_weights"
    ]
    assert applied == [] and completed == [4]
    # The original Sync fit hook remains responsible for its one notification.
    driver._speco_freeze_observe_publish(metrics)
    assert driver._speco_freeze_policy.drafter_version == 1


@pytest.mark.parametrize("bubble", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_skipped_or_empty_oldlogprob_collection_cannot_prefetch_stale_head(
    bubble, collect
):
    from contextlib import contextmanager
    from types import MethodType
    from verl_speco.trainer.scheduler import DrafterCollectionSource

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_online_fit_hooks", "_speco_should_prefetch_target_lm_head"},
        time=time,
        MethodType=MethodType,
        DrafterCollectionSource=DrafterCollectionSource,
        _select_policy_model_batch=lambda batch: batch,
    )
    driver = cls()
    driver.global_steps = 5
    driver._speco_last_raw_drafter_samples = 160
    driver._speco_last_collected_samples = 20
    driver._speco_last_collected_step = 4
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    target = SimpleNamespace(generate_sequences=lambda: None)
    driver._speco_rollout_generation_target = lambda: target
    driver.checkpoint_manager = SimpleNamespace(update_weights=lambda: None)
    driver._compute_old_log_prob = lambda batch: ("old-log-prob", 0.5)
    driver._update_actor = lambda batch: None
    driver._speco_oldlogprob_collection_requested = lambda: True
    driver._speco_oldlogprob_collection_enabled = lambda: True
    driver._speco_oldlogprob_entropy_hook_enabled = lambda: False
    driver._speco_plan_drafter_collection = lambda source: SimpleNamespace(
        collect=collect, collect_interval_matched=False
    )
    driver._speco_log_drafter_collection_plan = lambda plan: None
    driver._speco_build_oldlogprob_collect_plan = lambda batch: None
    driver._speco_wait_pending_drafter_publish = lambda: None
    with contextmanager(driver._speco_online_fit_hooks)():
        assert driver._compute_old_log_prob(object()) == ("old-log-prob", 0.5)
        assert not driver._speco_should_prefetch_target_lm_head()
    assert driver._speco_last_collected_samples == (0 if bubble else 20)
    assert driver._speco_last_raw_drafter_samples == (0 if bubble else 160)


def _head_lifecycle_worker():
    torch = pytest.importorskip("torch")
    from verl_speco.trainer.data_buffer import DataBuffer

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "sync_target_lm_head_weight",
            "select_target_lm_head_version",
            "get_training_data_status",
            "_filter_training_data_by_step",
            "reserve_training_data",
            "finalize_training_data_reservation",
            "release_training_data_reservation",
            "_mark_buffer_changed",
        },
        torch=torch,
        cast=cast,
        Any=Any,
    )
    worker = cls()
    worker.rank = 0
    worker._bubble_time_enabled = True
    worker._target_lm_head_snapshots = {}
    worker._target_lm_head_snapshot_limit = 2
    worker._active_training_reservation_id = None
    worker._active_training_target_version = None
    worker._applied_target_lm_head_weight_step = None
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={"use_logits": False}))
    )
    worker.backend = SimpleNamespace(model_type="dspark")
    worker.batch_size = 4
    worker.current_rl_step = 4
    worker.buffer_version = 1
    worker.use_data_buffer = True
    worker.collected_data = []
    worker.data_buffer = DataBuffer(max_size=32)
    worker.data_buffer.update_rl_step(4)
    for index in range(20):
        worker.data_buffer.add_batch(
            {"target_version": 4, "id": index, "loss_tokens": 512}
        )
    return worker


@pytest.mark.parametrize(
    "collected_step, count, expected",
    [
        (4, 20, False),
        (None, 20, False),
        (5, 0, False),
        (5, 20, True),
    ],
)
def test_bubble_prefetch_requires_positive_collection_in_this_step(
    collected_step, count, expected
):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_should_prefetch_target_lm_head"},
    )
    driver = cls()
    driver.global_steps = 5
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_last_collected_samples = count
    driver._speco_last_collected_step = collected_step
    assert driver._speco_should_prefetch_target_lm_head() is expected
    driver._speco_drafter_frozen = True
    assert not driver._speco_should_prefetch_target_lm_head()


@pytest.mark.parametrize("bubble", [False, True])
def test_worker_rpc_forwards_head_protection_only_for_bubble(bubble):
    cls = _methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"sync_target_lm_head_weight"},
    )
    worker = cls()
    worker.rank = 0
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.is_drafter_group_leader = False
    calls = []
    worker.trainer = SimpleNamespace(
        _bubble_time_enabled=bubble,
        sync_target_lm_head_weight=lambda weight, **kwargs: calls.append(kwargs)
        or {"accepted": True},
    )
    result = worker.sync_target_lm_head_weight(
        {
            "weight": "cpu-head",
            "protected_target_versions": (4,),
            "defer_device_apply": True,
        },
        global_step=6,
    )
    assert result["accepted"]
    assert calls[0].get("protected_target_versions") == ((4,) if bubble else None)


def test_worker_replay_session_protects_head_without_driver_metadata():
    torch = pytest.importorskip("torch")
    worker = _head_lifecycle_worker()
    worker.sync_target_lm_head_weight(
        torch.full((2, 2), 4.0),
        global_step=4,
        defer_device_apply=True,
        protected_target_versions=(4,),
    )
    worker.reserve_training_data(
        plan_id="short", target_version=4, max_batches=1, retain_replay_session=True
    )
    worker.finalize_training_data_reservation("short", consume=False)
    worker.release_training_data_reservation("short")
    for version in (5, 6):
        worker.sync_target_lm_head_weight(
            torch.ones(2, 2),
            global_step=version,
            defer_device_apply=True,
            protected_target_versions=(),
        )
    assert worker.select_target_lm_head_version(4)
    assert len(worker._target_lm_head_snapshots) == 2
    worker.reserve_training_data(plan_id="final", target_version=4, max_batches=20)
    worker.finalize_training_data_reservation("final")
    worker.release_training_data_reservation("final")
    worker.sync_target_lm_head_weight(
        torch.ones(2, 2),
        global_step=7,
        defer_device_apply=True,
        protected_target_versions=(),
    )
    assert not worker.select_target_lm_head_version(4)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_quota_head_survives_short_idle_plans_then_topup_and_publish(groups):
    from verl_speco.trainer.scheduler import (
        CallbackDrafterWorkerExecutor,
        TrainingOutcome,
        TrainingPlan,
    )
    from verl_speco.trainer.scheduler.lifecycle import BeforeActorUpdateContext

    torch = pytest.importorskip("torch")
    workers = [_head_lifecycle_worker() for _ in range(groups * 2)]

    def inspect_data(
        sample_last_n_steps, require_full_batch, worker_ids=None, target_version=None
    ):
        return [
            {
                **workers[int(rank)].get_training_data_status(
                    target_version=target_version
                ),
                "worker_id": rank,
                "worker_incarnation": f"worker-{rank}",
                "rank": int(rank),
                "available": True,
            }
            for rank in (worker_ids or tuple(str(i) for i in range(len(workers))))
        ]

    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            inspect_data=inspect_data,
            submit=lambda payload: None,
            resolve=lambda value: value,
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda plan: [],
            abort_preflight=lambda plan_id: [],
        )
    )
    scheduler._idle_worker_writer_group = ("0", "1")
    # Match the log: quota 2 already bootstrapped this writer before quota 4.
    scheduler._idle_optimizer_initialized_groups.add(("0", "1"))
    config = DrafterScheduleConfig(
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_training_groups=tuple(
            (str(i), str(i + 1)) for i in range(0, len(workers), 2)
        ),
        training_interval_steps=2,
        training_quota_enable=True,
        training_quota_target_steps=20,
        training_quota_allow_critical_path_fallback=True,
        training_quota_max_completion_lag_steps=3,
        training_quota_max_sync_topup_steps=20,
        use_logits=False,
    )
    scheduler.record_collection_outcome(
        SimpleNamespace(source_global_step=4),
        SimpleNamespace(collected=True, worker_results=[]),
        config,
    )
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_mark_target_lm_head_version_ready",
            "_speco_target_lm_head_ready_for_workers",
            "_speco_dispatch_target_lm_head_payload",
            "_speco_advance_target_lm_head_prefetch",
            "_speco_validate_target_lm_head_acks",
        },
        time=time,
        cast=cast,
        Any=Any,
        _SPECO_TARGET_LM_HEAD_READY_VERSION_LIMIT=2,
    )
    driver = cls()
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_ready_target_lm_head_versions = set()
    driver._speco_ready_target_lm_head_workers = {}
    driver._speco_ray_refs_ready = lambda refs: True
    driver._ray_get_if_needed = lambda refs: refs
    driver._speco_build_drafter_target_lm_head_sync_args = lambda payload: (
        payload,
        driver.global_steps,
        1,
    )

    def dispatch(payload, global_step):
        return [
            dict(
                worker.sync_target_lm_head_weight(
                    payload["weight"],
                    global_step=global_step,
                    defer_device_apply=payload["defer_device_apply"],
                    protected_target_versions=payload["protected_target_versions"],
                ),
                worker_id=str(rank),
            )
            for rank, worker in enumerate(workers)
        ]

    driver.speco_sync_target_lm_head_weight = dispatch
    for step in (4, 5, 6):
        driver.global_steps = step
        # Even an unexpected newer export must not evict the active quota.
        _, pending_head = driver._speco_dispatch_target_lm_head_payload(
            {"weight": torch.full((2, 2), float(step))},
            sync_started=time.perf_counter(),
            fetch_elapsed=0.0,
            selected_rows=0,
            source_vocab_size=2,
        )
        driver._pending_target_lm_head_sync = dict(
            pending_head, stage="dispatch", target_version=step
        )
        assert (
            driver._speco_advance_target_lm_head_prefetch()[
                "drafter/target_lm_head_synced"
            ]
            == 1
        )
        assert driver._speco_target_lm_head_ready_for_workers(4, ("0", "1"))
        for worker in workers:
            assert worker.select_target_lm_head_version(4)
            assert torch.equal(
                worker._pending_target_lm_head_weight, torch.full((2, 2), 4.0)
            )
            assert (
                worker.get_training_data_status(target_version=4)["trainable_batches"]
                == 5
            )
            assert len(worker._target_lm_head_snapshots) == min(step - 3, 2)
        if step > 4:
            count = 1 if step == 5 else 3
            # No new collection in this rollout: buffered quota data alone
            # must still admit real idle work after the per-step count reset.
            status = scheduler.inspect_training_data(
                global_step=step, config=config, worker_ids=("0", "1"), target_version=4
            )
            idle = scheduler.plan_training(
                DrafterScheduleContext(
                    global_step=step,
                    training_mode="online",
                    collected_samples_this_step=0,
                    oldlogprob_collection_requested=True,
                    data_status=status,
                ),
                config,
                resources=SimpleNamespace(
                    training_group_id="idle-group-0",
                    worker_ids=("0", "1"),
                    minimum_idle_window_sec=10.0,
                    idle_confidence=IdleWindowConfidence.CONFIRMED,
                ),
                require_interval=False,
            )
            assert idle.launch and idle.required_target_version == 4
            plan = TrainingPlan(
                launch=True,
                reason="training_ready",
                interval_matched=False,
                execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
                source_global_step=step,
                max_batches=count,
                target_worker_ids=("0", "1"),
                data_version=4,
                required_target_version=4,
                retain_replay_session=True,
                publish_after_success=False,
                plan_id=f"idle-{step}",
            )
            for worker in workers[:2]:
                worker.reserve_training_data(
                    plan_id=plan.plan_id,
                    target_version=4,
                    max_batches=count,
                    retain_replay_session=True,
                )
                worker._active_training_replay_used_items.update(
                    {
                        id(item): item
                        for item in worker.data_buffer.get_available_data(
                            target_version=4, reservation_id=plan.plan_id
                        )
                    }
                )
                worker.finalize_training_data_reservation(plan.plan_id, consume=False)
                worker.release_training_data_reservation(plan.plan_id)
            scheduler._record_training_outcome(
                plan,
                TrainingOutcome(
                    trained=True,
                    successful_steps=count,
                    worker_results=[],
                    raw_results=[],
                    elapsed_sec=0.1,
                    reason="completed",
                    metrics={},
                ),
            )

    assert scheduler._training_quota_debt_steps == 16
    # Prevent a new idle plan here: the hard deadline must still choose a
    # synchronous writer-local completion of the exact remaining 16 steps.
    scheduler.prepare_training_plan = lambda *a, **k: TrainingPlan(
        launch=False,
        reason="window_too_small",
        interval_matched=False,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=7,
        max_batches=0,
        publish_after_success=False,
    )
    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=DrafterScheduleContext(
                global_step=7,
                training_mode="online",
                collected_samples_this_step=0,
                oldlogprob_collection_requested=True,
            ),
            config=config,
        )
    )
    final = event.training_plan
    assert final.launch and final.max_batches == 16
    assert final.required_target_version == 4 and final.publish_after_success
    assert not final.retain_replay_session
    for worker in workers[:2]:
        worker.reserve_training_data(
            plan_id=final.plan_id, target_version=4, max_batches=16
        )
        assert worker.finalize_training_data_reservation(final.plan_id) == 20
        worker.release_training_data_reservation(final.plan_id)
    scheduler._record_training_outcome(
        final,
        TrainingOutcome(
            trained=True,
            successful_steps=16,
            worker_results=[],
            raw_results=[],
            elapsed_sec=1.0,
            reason="completed",
            metrics={},
        ),
    )
    assert scheduler._training_quota_debt_steps == 0
    assert scheduler.protected_target_lm_head_versions() == (4,)
    assert scheduler.plan_publish(
        global_step=7, drafter_trained=True, config=config, training_plan=final
    ).publish

    publisher, _, stages, applied, _ = _freeze_publish_driver()
    publisher._speco_get_drafter_scheduler().record_training_quota_publish_completed = (
        scheduler.record_training_quota_publish_completed
    )
    publisher._speco_publish_drafter_weights(True, final)
    assert applied == [] and scheduler.protected_target_lm_head_versions() == (4,)
    publisher._speco_wait_pending_drafter_publish_rpc()
    assert applied == [4] and scheduler.protected_target_lm_head_versions() == ()
    assert not scheduler.training_quota_blocks_collection(8, config)
    # The next cycle can evict the completed head normally on both sides.
    driver._speco_mark_target_lm_head_version_ready(8)
    assert 4 not in driver._speco_ready_target_lm_head_versions
    for worker in workers:
        worker.sync_target_lm_head_weight(
            torch.ones(2, 2),
            global_step=8,
            defer_device_apply=True,
            protected_target_versions=(),
        )
        assert 4 not in worker._target_lm_head_snapshots


def _head_ack_driver(results, *, bubble=True, targets=("0", "1")):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_validate_target_lm_head_acks",
            "_speco_advance_target_lm_head_prefetch",
            "_speco_mark_target_lm_head_version_ready",
            "_speco_target_lm_head_ready_for_workers",
            "_speco_finish_target_lm_head_weight_sync",
        },
        time=time,
        cast=cast,
        Any=Any,
        _SPECO_TARGET_LM_HEAD_READY_VERSION_LIMIT=2,
    )
    driver = cls()
    driver._pending_target_lm_head_sync = {
        "stage": "dispatch",
        "refs": results,
        "target_version": 4,
        "global_step": 4,
        "target_worker_ids": targets,
    }
    driver._speco_ready_target_lm_head_versions = set()
    driver._speco_ready_target_lm_head_workers = {}
    driver._speco_ray_refs_ready = lambda refs: True
    driver._ray_get_if_needed = lambda refs: refs
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver._speco_get_drafter_scheduler = DrafterScheduler
    return driver


def _head_ack(worker, **changes):
    return {
        "worker_id": str(worker),
        "global_step": 4,
        "accepted": True,
        "applied": False,
        "pending": True,
        **changes,
    }


@pytest.mark.parametrize(
    "failure",
    [
        "rejected",
        "stale",
        "missing",
        "duplicate",
        "unexpected",
        "unidentified",
        "empty",
        "null",
        "target_skipped",
    ],
)
@pytest.mark.parametrize("completion", ["poll", "blocking"])
def test_rejected_or_incomplete_head_acks_never_report_success(failure, completion):
    results = [_head_ack(0), _head_ack(1)]
    if failure == "rejected":
        results[1] = _head_ack(1, accepted=False, reason="missing_weight")
    elif failure == "stale":
        results[1]["global_step"] = 3
    elif failure == "missing":
        results.pop()
    elif failure == "duplicate":
        results[1] = _head_ack(0)
    elif failure == "unexpected":
        results.append(_head_ack(2))
    elif failure == "unidentified":
        results[1].pop("worker_id")
    elif failure == "empty":
        results = []
    elif failure == "null":
        results = None
    elif failure == "target_skipped":
        results[1] = _head_ack(1, accepted=False, reason="not_target_worker")
    driver = _head_ack_driver(results)
    with pytest.raises(RuntimeError, match="Bubble target lm_head ACK"):
        if completion == "poll":
            driver._speco_advance_target_lm_head_prefetch()
        else:
            driver._speco_finish_target_lm_head_weight_sync(
                driver._pending_target_lm_head_sync
            )
    assert 4 not in driver._speco_ready_target_lm_head_versions


def test_head_ack_validation_ignores_only_non_target_skips():
    driver = _head_ack_driver(
        [
            [_head_ack(0)],
            [_head_ack(1), _head_ack(2, accepted=False, reason="not_target_worker")],
        ]
    )
    assert (
        driver._speco_advance_target_lm_head_prefetch()["drafter/target_lm_head_synced"]
        == 1
    )
    assert driver._speco_target_lm_head_ready_for_workers(4, ("0", "1"))
    assert not driver._speco_target_lm_head_ready_for_workers(4, ("2",))


@pytest.mark.parametrize("targets", [None, ()])
def test_broadcast_head_ready_is_scoped_to_actual_acknowledged_workers(targets):
    driver = _head_ack_driver(
        [_head_ack(0), _head_ack(1, accepted=False, reason="not_in_training_group")],
        targets=targets,
    )
    driver._speco_advance_target_lm_head_prefetch()
    assert driver._speco_target_lm_head_ready_for_workers(4, ("0",))
    assert not driver._speco_target_lm_head_ready_for_workers(4, ("0", "1"))


def test_sync_target_head_retains_legacy_ack_contract():
    driver = _head_ack_driver([{"accepted": True}], bubble=False)
    metrics = driver._speco_finish_target_lm_head_weight_sync(
        driver._pending_target_lm_head_sync
    )
    assert metrics["drafter/target_lm_head_synced"] == 1


@pytest.mark.parametrize(
    "reason",
    [
        "accepted",
        "missing_weight",
        "disabled",
        "not_in_training_group",
        "not_target_worker",
    ],
)
def test_bubble_head_worker_ack_contains_rank_and_version_even_on_rejection(reason):
    cls = _methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"sync_target_lm_head_weight"},
    )
    worker = cls()
    worker.rank = 1
    worker.enable_drafter = reason != "disabled"
    worker.in_drafter_train_group = reason != "not_in_training_group"
    worker.is_drafter_group_leader = False
    worker.trainer = SimpleNamespace(
        _bubble_time_enabled=True,
        sync_target_lm_head_weight=lambda *args, **kwargs: {
            "accepted": reason == "accepted",
            "reason": reason,
        },
    )
    payload = {"weight": "cpu-head", "require_target_head_ack": True}
    if reason == "not_target_worker":
        payload["target_worker_ids"] = ("0",)
    result = worker.sync_target_lm_head_weight(payload, global_step=4)
    assert result["worker_id"] == "1" and result["global_step"] == 4
    assert result["accepted"] == (reason == "accepted")


def _quota_topup_plan(*, source_step=2, local=False, use_logits=False):
    from verl_speco.trainer.scheduler import TrainingPlan

    return TrainingPlan(
        launch=True,
        reason="quota_forced_completion_ready",
        interval_matched=True,
        execution_strategy=(
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            if local
            else DrafterExecutionStrategy.SYNC
        ),
        source_global_step=source_step,
        max_batches=20,
        min_batches=1,
        deadline_ts=None,
        require_full_batch=False,
        sample_last_n_steps=20,
        publish_after_success=True,
        required_target_version=None if use_logits else 2,
        data_version=2,
        plan_id=f"bootstrap-{source_step}",
        target_worker_ids=("0", "1") if local else (),
        worker_snapshots={"0": {}, "1": {}},
    )


def _topup_head_driver(*, bubble=True):
    from dataclasses import replace

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_prepare_drafter_training_rpc",
            "_speco_gate_bubble_quota_topup",
            "_speco_target_lm_head_ready_for_workers",
        },
        replace=replace,
        cast=cast,
        Any=Any,
    )
    obj = cls()
    obj._speco_rollout_idle_worker_enabled = lambda: bubble
    obj._pending_target_lm_head_sync = None
    obj._speco_ready_target_lm_head_versions = {2}
    obj._speco_ready_target_lm_head_workers = {2: ("0", "1")}
    obj._speco_set_drafter_global_step = lambda: pytest.fail(
        "Bubble top-up must not re-prepare Sync"
    )
    obj._speco_start_target_lm_head_weight_sync = lambda plan: pytest.fail(
        "Do not replace the pinned head with the live actor head"
    )
    return obj


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("step", [2, 3])
def test_bubble_topup_reuses_ready_quota_head_in_the_same_step(local, step):
    driver = _topup_head_driver()
    plan = _quota_topup_plan(source_step=step, local=local)
    scheduler = DrafterScheduler()
    scheduler._worker_executor = SimpleNamespace(
        prepare_training=lambda plan: pytest.fail("No new Sync head preparation")
    )
    assert scheduler.prepare_training_execution(plan) == {
        "drafter/target_lm_head_synced": 0
    }
    # Defense in depth if the worker-executor callback is invoked directly.
    assert driver._speco_prepare_drafter_training_rpc(plan) == {
        "drafter/target_lm_head_synced": 0
    }
    accepted, metrics = driver._speco_gate_bubble_quota_topup(plan)
    assert accepted is plan and accepted.launch
    assert accepted.max_batches == 20 and accepted.required_target_version == 2
    assert driver._pending_target_lm_head_sync is None
    assert not metrics


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("failure", ["pending", "missing", "partial", "wrong_version"])
def test_bubble_topup_still_waits_for_exact_head_on_every_rank(local, failure):
    driver = _topup_head_driver()
    plan = _quota_topup_plan(local=local)
    if failure == "pending":
        driver._pending_target_lm_head_sync = {"target_version": 2, "stage": "dispatch"}
    elif failure == "missing":
        driver._speco_ready_target_lm_head_versions.clear()
    elif failure == "partial":
        driver._speco_ready_target_lm_head_workers[2] = ("0",)
    else:
        driver._speco_ready_target_lm_head_versions = {3}
    pending = driver._pending_target_lm_head_sync
    driver._speco_prepare_drafter_training_rpc(plan)
    deferred, metrics = driver._speco_gate_bubble_quota_topup(plan)
    assert not deferred.launch and deferred.max_batches == 0
    assert deferred.required_target_version == 2 and plan.max_batches == 20
    assert driver._pending_target_lm_head_sync is pending
    assert metrics["bubble/training_quota_topup_deferred"] == 1
    assert metrics["scheduler/train_requested"] == 0
    assert metrics["scheduler/planned_batches"] == 0
    assert metrics["bubble/training_quota_topup_batches"] == 0
    driver._pending_target_lm_head_sync = None
    driver._speco_ready_target_lm_head_versions = {2}
    driver._speco_ready_target_lm_head_workers[2] = ("0", "1")
    assert driver._speco_gate_bubble_quota_topup(plan)[0] is plan


def test_logits_bubble_topup_does_not_require_head_prefetch():
    driver = _topup_head_driver()
    driver._speco_ready_target_lm_head_versions.clear()
    plan = _quota_topup_plan(use_logits=True)
    assert driver._speco_gate_bubble_quota_topup(plan) == (plan, {})


def test_normal_sync_keeps_legacy_head_prepare_and_pending_semantics():
    from dataclasses import replace

    driver = _topup_head_driver(bubble=False)
    plan = replace(_quota_topup_plan(), reason="training_ready")
    calls, pending = [], {"global_step": 2, "refs": "head-dispatch"}
    driver._speco_set_drafter_global_step = lambda: calls.append("set-step")
    driver._speco_start_target_lm_head_weight_sync = lambda received: (
        calls.append(received) or {"prepared": 1},
        pending,
    )
    scheduler = DrafterScheduler()
    scheduler._worker_executor = SimpleNamespace(
        prepare_training=driver._speco_prepare_drafter_training_rpc
    )
    assert scheduler.prepare_training_execution(plan) == {"prepared": 1}
    assert calls == ["set-step", plan]
    assert driver._pending_target_lm_head_sync is pending
    assert driver._speco_gate_bubble_quota_topup(plan) == (plan, {})


def _collection_alignment_driver(
    *,
    bubble,
    owners,
    routed,
    hard_ratio=0.0,
    token_budget=16384,
    mode="interval",
    algorithm="DSPARK",
):
    import hashlib
    import math
    import torch
    from verl_speco.trainer.hard_sample_selection import partition_hard_candidates
    from verl_speco.trainer.scheduler import DrafterCollectionSource

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_build_oldlogprob_collect_plan"},
        torch=torch,
        math=math,
        hashlib=hashlib,
        partition_hard_candidates=partition_hard_candidates,
        DrafterCollectionSource=DrafterCollectionSource,
        _get_nested=lambda cfg, path, default: cfg.get(path[0], default),
    )
    driver = cls()
    driver.global_steps = 5
    driver._speco_oldlogprob_collection_enabled = lambda: True
    driver._speco_plan_drafter_collection = lambda source: SimpleNamespace(
        collect=True,
        sample_rate=1.0,
        max_samples_per_replica=16,
        max_tokens_per_replica=token_budget,
    )
    driver._speco_log_drafter_collection_plan = lambda plan: None
    driver._speco_drafter_training_config = lambda: {
        "dspark_hard_sample_ratio": hard_ratio,
        "dspark_hard_candidate_ratio": 0.4,
    }
    driver._speco_oldlogprob_window_mode = lambda cfg: "random"
    driver._speco_oldlogprob_window_train_rows = lambda cfg: 512
    driver._speco_owner_bucket_count = lambda: owners
    driver._speco_bubble_collection_owner_ranks = (
        lambda **kwargs: routed if bubble else ()
    )
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver._speco_drafter_schedule_config = lambda: DrafterScheduleConfig(
        training_quota_enable=True,
        training_quota_trigger_mode=mode,
    )
    if not bubble:
        driver._speco_drafter_schedule_config = lambda: pytest.fail(
            "Sync selection must not depend on Bubble schedule configuration"
        )
    driver._speco_drafter_config = lambda: {"speculative_algorithm": algorithm}
    driver._speco_request_accept_lengths = lambda batch, size: [
        float(i + 1) for i in range(size)
    ]
    driver._speco_request_ids = lambda batch, size, **kw: [
        f"request-{i}" for i in range(size)
    ]
    driver._speco_request_completion_indices = lambda batch, size: list(range(size))
    driver._speco_request_elapsed_secs = lambda batch, size: [1.0] * size

    def hashed(key):
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")

    driver._speco_hash_fraction = lambda key: hashed(key) / 2**64
    driver._speco_hash_int = lambda key, maximum: hashed(key) % (maximum + 1)
    driver._speco_log_request_accept_lens = lambda **kw: None
    size, prompt, response = 160, 4, 520
    batch = SimpleNamespace(
        batch={
            "prompts": torch.ones(size, prompt, dtype=torch.long),
            "responses": torch.ones(size, response, dtype=torch.long),
            "attention_mask": torch.ones(size, prompt + response, dtype=torch.long),
            "response_mask": torch.ones(size, response, dtype=torch.long),
        }
    )
    return driver, batch


@pytest.mark.parametrize("owners", [2, 3, 4])
@pytest.mark.parametrize("hard_ratio", [0.0, 0.375])
@pytest.mark.parametrize("token_budget", [16384, 8192])
def test_interval_collection_matches_sync_before_routing(
    owners, hard_ratio, token_budget
):
    import torch

    common = dict(owners=owners, hard_ratio=hard_ratio, token_budget=token_budget)
    sync, batch = _collection_alignment_driver(bubble=False, routed=(), **common)
    expected = sync._speco_build_oldlogprob_collect_plan(batch)
    for routed in (tuple(range(owners)), (owners - 1,)):
        bubble, _ = _collection_alignment_driver(bubble=True, routed=routed, **common)
        actual = bubble._speco_build_oldlogprob_collect_plan(batch)
        assert actual["selection_metrics"]["sync_aligned_collection"] == 1
        assert "sync_aligned_collection" not in expected["selection_metrics"]
        assert actual["selected_count"] == expected["selected_count"]
        assert (
            actual["selected_source_indices_hash"]
            == expected["selected_source_indices_hash"]
        )
        for key in (
            "collect_mask",
            "hidden_positions",
            "hidden_position_mask",
            "request_is_hard",
        ):
            assert torch.equal(actual[key], expected[key])
        assert (
            actual["selection_metrics"]["selected_hard_total"]
            == expected["selection_metrics"]["selected_hard_total"]
        )
        selected = actual["collect_mask"]
        assert set(actual["owner_rank"][selected].tolist()) <= set(routed)
        assert sum(actual["owner_token_counts"]) == actual["selected_count"] * 513
        if len(routed) == owners:
            assert torch.equal(actual["owner_rank"], expected["owner_rank"])


def _bind_collection_writer(driver, *, owners, tp):
    from collections import deque

    scheduler = DrafterScheduler()
    groups = tuple(
        tuple(str(owner * tp + rank) for rank in range(tp)) for owner in range(owners)
    )
    scheduler._metadata_idle_training_groups = groups
    scheduler._replica_idle_worker_groups = dict(enumerate(groups))
    scheduler._replica_idle_windows_by_source = {
        owner: {"generation_completed": deque([float(owner + 1)] * 3)}
        for owner in range(owners)
    }
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_bubble_collection_owner_ranks"},
    )
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_owner_route_mapping = lambda: [
        owner for owner in range(owners) for _ in range(tp)
    ]
    driver._speco_bubble_collection_owner_ranks = (
        cls._speco_bubble_collection_owner_ranks.__get__(driver)
    )
    return scheduler, groups


@pytest.mark.parametrize("owners,tp", [(2, 1), (2, 2), (3, 2), (4, 4)])
@pytest.mark.parametrize("hard_ratio", [0.0, 0.375])
def test_first_quota_writer_replay_contains_entire_sync_sample_set(
    owners, tp, hard_ratio
):
    import torch
    from verl_speco.trainer.data_buffer import DataBuffer
    from verl_speco.trainer.scheduler.collection_adapter import _build_payload
    from verl_speco.trainer.scheduler import DrafterCollectionSource

    common = dict(owners=owners, hard_ratio=hard_ratio)
    sync, batch = _collection_alignment_driver(bubble=False, routed=(), **common)
    expected = sync._speco_build_oldlogprob_collect_plan(batch)
    driver, _ = _collection_alignment_driver(bubble=True, routed=(), **common)
    scheduler, groups = _bind_collection_writer(driver, owners=owners, tp=tp)
    # Queries/prewarm alone still cover all candidates and do not pin ownership.
    assert driver._speco_bubble_collection_owner_ranks() == tuple(range(owners))
    assert scheduler.idle_writer_group() is None
    actual = driver._speco_build_oldlogprob_collect_plan(batch)
    indices = actual["collect_mask"].nonzero(as_tuple=False).flatten().tolist()
    payload = _build_payload(
        source=DrafterCollectionSource.OLD_LOGPROB,
        samples=[{"id": i, "target_version": 5} for i in indices],
        owners=actual["owner_rank"][actual["collect_mask"]].tolist(),
        owner_count=owners,
        dispatch_bucket_count=owners + 1,
        raw_samples=len(indices),
        collection_id="first-quota",
    )
    assert scheduler.idle_writer_group() == groups[-1]
    assert scheduler._idle_worker_writer_state_version is None
    assert not scheduler._idle_optimizer_initialized_groups
    assert not scheduler._idle_worker_hot_prewarmed_groups
    assert torch.equal(actual["collect_mask"], expected["collect_mask"])
    assert torch.equal(actual["hidden_positions"], expected["hidden_positions"])
    # Exercise the actual reservation path, not just the global collect counter.
    worker = _head_lifecycle_worker()
    worker.data_buffer = DataBuffer(max_size=256)
    worker.data_buffer.update_rl_step(5)
    for sample in payload.buckets[-2]:
        worker.data_buffer.add_batch(sample)
    reservation = worker.reserve_training_data(
        plan_id="first", target_version=5, max_batches=1, retain_replay_session=True
    )
    expected_ids = set(
        expected["collect_mask"].nonzero(as_tuple=False).flatten().tolist()
    )
    assert reservation["reserved_samples"] == len(expected_ids)
    assert {
        sample["id"] for sample in worker._training_replay_sessions[5]["items"]
    } == expected_ids
    assert all(not bucket for bucket in payload.buckets[:-2])
    assert not payload.buckets[-1]  # Nonparticipating ranks receive no samples.
    assert scheduler.target_lm_head_sync_worker_ids() == groups[-1]
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_data_version = 5
    assert scheduler.needs_idle_optimizer_bootstrap(
        driver._speco_drafter_schedule_config()
    )
    # A later idle inversion must not silently move an existing optimizer owner.
    scheduler._idle_worker_writer_state_version = 5
    scheduler._replica_idle_windows_by_source[0]["generation_completed"].append(999.0)
    assert scheduler.prepare_collection_writer_group() == groups[-1]


def test_empty_interval_collection_does_not_pin_bootstrap_writer():
    driver, batch = _collection_alignment_driver(bubble=True, owners=2, routed=())
    scheduler, _ = _bind_collection_writer(driver, owners=2, tp=2)
    batch.batch["response_mask"].zero_()
    driver._speco_build_oldlogprob_collect_plan(batch)
    assert scheduler.idle_writer_group() is None


def test_adaptive_collection_does_not_elect_bootstrap_writer():
    driver, batch = _collection_alignment_driver(
        bubble=True, owners=2, routed=(), mode="adaptive"
    )
    scheduler, _ = _bind_collection_writer(driver, owners=2, tp=2)
    driver._speco_build_oldlogprob_collect_plan(batch)
    assert scheduler.idle_writer_group() is None


def test_adaptive_collection_retains_existing_writer_minimum():
    driver, batch = _collection_alignment_driver(
        bubble=True,
        owners=2,
        routed=(1,),
        mode="adaptive",
    )
    plan = driver._speco_build_oldlogprob_collect_plan(batch)
    assert plan["selected_count"] == 20
    assert set(plan["owner_rank"][plan["collect_mask"]].tolist()) == {1}


@pytest.mark.parametrize("algorithm", ["DFLASH", "EAGLE3"])
def test_interval_collection_alignment_is_not_dspark_specific(algorithm):
    import torch

    sync, batch = _collection_alignment_driver(
        bubble=False,
        owners=4,
        routed=(),
        algorithm=algorithm,
    )
    bubble, _ = _collection_alignment_driver(
        bubble=True,
        owners=4,
        routed=(2,),
        algorithm=algorithm,
    )
    expected = sync._speco_build_oldlogprob_collect_plan(batch)
    actual = bubble._speco_build_oldlogprob_collect_plan(batch)
    assert expected["selected_count"] == actual["selected_count"] == 64
    assert torch.equal(actual["collect_mask"], expected["collect_mask"])
    assert torch.equal(actual["hidden_positions"], expected["hidden_positions"])
    assert set(actual["owner_rank"][actual["collect_mask"]].tolist()) == {2}


@pytest.mark.parametrize("completed", [8, 20])
def test_quota_outcome_metrics_report_post_execution_debt(completed):
    from verl_speco.trainer.scheduler import TrainingOutcome

    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_data_version = 2
    scheduler._training_quota_oldest_cycle_step = 2
    plan = _quota_topup_plan()
    # Top-ups must complete their requested batch count; use an idle partial
    # plan for the eight-step case, not a deliberately incomplete top-up.
    if completed == 8:
        from dataclasses import replace

        plan = replace(
            plan,
            execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
            reason="training_ready",
            max_batches=8,
            target_worker_ids=("0", "1"),
        )
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=completed,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        reason="completed",
        metrics={},
    )
    scheduler._record_training_outcome(plan, outcome)
    assert outcome.metrics["bubble/training_quota_debt_before"] == 20
    assert outcome.metrics["bubble/training_quota_debt_after"] == 20 - completed
    assert outcome.metrics["bubble/training_quota_debt_steps"] == 20 - completed


def test_normal_sync_outcome_does_not_publish_bubble_quota_metrics():
    from dataclasses import replace
    from verl_speco.trainer.scheduler import TrainingOutcome

    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 20
    plan = replace(_quota_topup_plan(), reason="training_ready")
    outcome = TrainingOutcome(
        trained=True,
        successful_steps=20,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1.0,
        reason="completed",
        metrics={},
    )
    scheduler._record_training_outcome(plan, outcome)
    assert scheduler._training_quota_debt_steps == 20
    assert "bubble/training_quota_debt_steps" not in outcome.metrics
