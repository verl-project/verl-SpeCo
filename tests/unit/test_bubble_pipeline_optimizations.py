# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU contracts for independent head, publish and idle-budget optimizations."""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from test_bubble_early_publish_staging import _driver
from test_bubble_review_regressions import _methods
from test_idle_window_readmission import _hot_writer
from verl_speco.integration import rollout_publish as transport
from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    TrainingOutcome,
)


def _head_transport(monkeypatch):
    np = pytest.importorskip("numpy")

    class Tensor:
        def __init__(self, bits, dtype="bf16"):
            self.bits, self.dtype = bits, dtype
            self.device = SimpleNamespace(type="cpu")

        def view(self, dtype):
            return Tensor(self.bits, dtype)

        def numpy(self):
            return self.bits

    class Ref:
        def __init__(self, value):
            self.value = value

    gets = []

    def get(ref):
        gets.append(ref)
        return ref.value

    monkeypatch.setattr(
        transport,
        "_ray_module",
        lambda: SimpleNamespace(ObjectRef=Ref, put=Ref, get=get),
    )
    monkeypatch.setattr(
        transport,
        "_torch_module",
        lambda: SimpleNamespace(
            bfloat16="bf16",
            uint8="uint8",
            from_numpy=lambda bits: Tensor(bits, "uint8"),
        ),
    )
    bits = np.arange(24, dtype=np.uint8).reshape(4, 6)
    return (
        {
            "weight": Tensor(bits),
            "source_vocab_size": 4,
            "export_strategy": "direct_full",
        },
        bits,
        gets,
    )


def test_head_reference_forwards_no_driver_tensor_and_preserves_bits(monkeypatch):
    payload, bits, gets = _head_transport(monkeypatch)
    wrapped = transport.pack_target_head_reference(payload, 8)
    assert "weight" not in wrapped and not gets
    wrapped.update(
        defer_device_apply=True,
        require_target_head_ack=True,
        protected_target_versions=(8,),
        target_worker_ids=("0",),
    )
    decoded = transport.resolve_target_head_reference(wrapped, 8)
    assert decoded["weight"].dtype == "bf16"
    assert (decoded["weight"].bits == bits).all()
    assert decoded["protected_target_versions"] == (8,)
    assert decoded["defer_device_apply"] and len(gets) == 1


def test_head_reference_rejects_version_before_materialization(monkeypatch):
    payload, _, gets = _head_transport(monkeypatch)
    wrapped = transport.pack_target_head_reference(payload, 8)
    with pytest.raises(RuntimeError, match="reference/version"):
        transport.resolve_target_head_reference(wrapped, 9)
    assert not gets
    wrapped["target_head_ref"].value["target_version"] = 7
    with pytest.raises(RuntimeError, match="object-store snapshot"):
        transport.resolve_target_head_reference(wrapped, 8)


@pytest.mark.parametrize("rank", [0, 1])
def test_only_target_worker_resolves_head_and_ack_is_versioned(monkeypatch, rank):
    payload, bits, gets = _head_transport(monkeypatch)
    wrapped = transport.pack_target_head_reference(payload, 8)
    wrapped.update(
        target_worker_ids=("0",),
        require_target_head_ack=True,
        defer_device_apply=True,
        protected_target_versions=(8,),
    )
    cls = _methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"sync_target_lm_head_weight"},
    )
    worker = cls()
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.rank = rank
    worker.is_drafter_group_leader = False
    captured = []
    worker.trainer = SimpleNamespace(
        _bubble_time_enabled=True,
        sync_target_lm_head_weight=lambda weight, **kwargs: (
            captured.append((weight, kwargs)) or {"accepted": True}
        ),
    )
    result = worker.sync_target_lm_head_weight(wrapped, global_step=8)
    assert result["accepted"] == (rank == 0)
    assert result["worker_id"] == str(rank) and result["global_step"] == 8
    assert len(gets) == len(captured) == (1 if rank == 0 else 0)
    if captured:
        assert (captured[0][0].bits == bits).all()
        assert captured[0][1]["protected_target_versions"] == (8,)


def test_head_codec_real_bf16_roundtrip_when_torch_available(monkeypatch):
    torch = pytest.importorskip("torch")

    class Ref:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(
        transport,
        "_ray_module",
        lambda: SimpleNamespace(ObjectRef=Ref, put=Ref, get=lambda ref: ref.value),
    )
    weight = torch.tensor([[0, 1.25, -3.5, float("inf")]], dtype=torch.bfloat16)
    payload = transport.pack_target_head_reference({"weight": weight}, 8)
    restored = transport.resolve_target_head_reference(payload, 8)["weight"]
    assert torch.equal(restored.view(torch.uint8), weight.view(torch.uint8))


def test_reference_export_owns_snapshot_even_when_actor_head_is_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    weight = torch.ones((4, 6), dtype=torch.bfloat16)
    worker = SimpleNamespace(rank=0, _is_actor=True, actor=object(), config={})
    monkeypatch.setattr(
        transport,
        "_head_only_export_candidate",
        lambda *args: ("lm_head.weight", weight, "ready"),
    )
    snapshot = transport.export_actor_lm_head_weight(
        worker, require_head_only=True, copy_cpu_snapshot=True
    )
    weight.add_(10)
    assert torch.equal(snapshot["weight"], torch.ones_like(weight))


@pytest.mark.parametrize("reference", [False, True])
def test_async_reference_export_is_opt_in_and_does_not_change_legacy_kwargs(
    monkeypatch, reference
):
    worker = SimpleNamespace()
    monkeypatch.setattr(transport, "_worker_bubble_enabled", lambda worker: True)
    calls = []
    monkeypatch.setattr(
        transport,
        "export_actor_lm_head_weight",
        lambda worker, **kwargs: calls.append(kwargs) or {"weight": "snapshot"},
    )
    monkeypatch.setattr(
        transport,
        "pack_target_head_reference",
        lambda payload, version: {"target_head_ref": "ref", "target_version": version},
    )
    result = transport.DraftWeightPublishMixin.get_actor_lm_head_weight_async(
        worker, reference_transport=reference, target_version=8 if reference else None
    )
    assert calls[0]["require_head_only"]
    assert calls[0].get("copy_cpu_snapshot", False) is reference
    assert ("target_head_ref" in result) is reference


def test_reference_export_cannot_be_requested_on_sync(monkeypatch):
    monkeypatch.setattr(transport, "_worker_bubble_enabled", lambda worker: False)
    monkeypatch.setattr(
        transport,
        "export_actor_lm_head_weight",
        lambda *args, **kwargs: pytest.fail("Sync must never enter reference export"),
    )
    with pytest.raises(RuntimeError, match="Bubble version"):
        transport.DraftWeightPublishMixin.get_actor_lm_head_weight_async(
            SimpleNamespace(), reference_transport=True, target_version=8
        )


def test_driver_reference_dispatch_does_not_materialize_or_change_version():
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_dispatch_target_lm_head_payload"},
        time=time,
        target_lm_head_export_metrics=transport.target_lm_head_export_metrics,
    )
    driver = cls()
    driver.global_steps = 8
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        protected_target_lm_head_versions=lambda: (8,)
    )
    driver._speco_build_drafter_target_lm_head_sync_args = lambda payload, **kwargs: (
        payload,
        kwargs["global_step"],
        1,
    )
    calls = []
    driver.speco_sync_target_lm_head_weight = lambda payload, **kwargs: (
        calls.append((payload, kwargs)) or ["cpu-cache-ack"]
    )
    driver._ray_get_if_needed = lambda refs: pytest.fail(
        "Driver must not resolve head data"
    )
    payload = {
        "target_head_ref": object(),
        "target_version": 8,
        "head_payload_bytes": 48,
    }
    metrics, pending = driver._speco_dispatch_target_lm_head_payload(
        payload,
        sync_started=time.perf_counter(),
        fetch_elapsed=0.01,
        selected_rows=0,
        source_vocab_size=4,
        target_worker_ids=("0",),
        target_version=8,
    )
    assert metrics["bubble/head_reference_transport"] == 1
    assert metrics["bubble/head_driver_tensor_materialized"] == 0
    assert pending["global_step"] == calls[0][1]["global_step"] == 8
    assert calls[0][0]["protected_target_versions"] == (8,)
    assert "weight" not in calls[0][0]
    with pytest.raises(RuntimeError, match="quota version"):
        driver._speco_dispatch_target_lm_head_payload(
            payload,
            sync_started=time.perf_counter(),
            fetch_elapsed=0,
            selected_rows=0,
            source_vocab_size=4,
            target_version=9,
        )
    assert len(calls) == 1
    driver._speco_rollout_idle_worker_enabled = lambda: False
    driver._speco_get_drafter_scheduler = lambda: pytest.fail(
        "Sync must not read Bubble head protections"
    )
    metrics, _ = driver._speco_dispatch_target_lm_head_payload(
        {"weight": "legacy-cpu-head"},
        sync_started=time.perf_counter(),
        fetch_elapsed=0,
        selected_rows=0,
        source_vocab_size=4,
        target_version=8,
    )
    assert not any(key.startswith("bubble/") for key in metrics)
    assert calls[-1][0] == {"weight": "legacy-cpu-head", "defer_device_apply": True}


def test_snapshot_async_rpc_is_cached_cpu_only_and_checks_source():
    cls = _methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"maybe_publish_async"},
        time=time,
    )
    worker = cls()
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.trainer, worker.rank, worker.last_global_step = (
        SimpleNamespace(_bubble_time_enabled=True),
        0,
        8,
    )
    calls = []
    worker.maybe_publish = lambda **kwargs: (
        calls.append(kwargs) or {"weights_ref": "snapshot"}
    )
    assert worker.maybe_publish_async(source_step=7) is None and not calls
    assert worker.maybe_publish_async(source_step=8) == {"weights_ref": "snapshot"}
    assert calls == [{"cached_snapshot_only": True}]
    worker.trainer._bubble_time_enabled = False
    assert worker.maybe_publish_async(source_step=8) is None
    assert len(calls) == 1
    worker.enable_drafter, worker.last_global_step = False, None
    assert worker.maybe_publish_async(source_step=8) is None


@pytest.mark.parametrize("capability", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_driver_negotiates_reference_transport_without_changing_legacy_rpc(
    capability, enabled
):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_start_target_lm_head_weight_sync"},
        time=time,
        _get_nested=transport._get_nested,
        DrafterExecutionStrategy=DrafterExecutionStrategy,
    )
    driver = cls()
    driver.global_steps, driver.device_name = 8, "cuda"
    driver.config = {"actor_rollout_ref": {"actor": {"strategy": "fsdp2"}, "model": {}}}
    driver._pending_target_lm_head_sync = None
    driver._speco_ready_target_lm_head_versions = set()
    driver._speco_ready_target_lm_head_workers = {}
    driver._speco_drafter_training_config = lambda: {}
    driver._speco_rollout_idle_worker_enabled = lambda: True
    driver._speco_drafter_schedule_config = lambda: DrafterScheduleConfig(
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_head_reference_transport=enabled,
    )
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        idle_writer_migration_blocked=lambda: False,
        target_lm_head_sync_worker_ids=lambda **kwargs: ("0",),
    )
    driver._speco_get_drafter_target_lm_head_row_selection = lambda: None
    calls = []
    driver.actor_rollout_wg = SimpleNamespace(
        get_actor_lm_head_weight_async=lambda rows, **kwargs: (
            calls.append(kwargs) or ["export-ref"]
        ),
        get_actor_lm_head_export_capability=lambda: [
            {
                "participating": True,
                "supported": True,
                "reference_transport": capability,
            }
        ],
    )
    driver._speco_actor_rollout_method = lambda name: getattr(
        driver.actor_rollout_wg, name
    )
    driver._ray_get_if_needed = lambda result: result
    metrics, pending = driver._speco_start_target_lm_head_weight_sync()
    assert pending["reference_transport"] == (enabled and capability)
    assert metrics["bubble/head_reference_transport"] == int(enabled and capability)
    assert calls == [
        {
            "keep_model_on_device": False,
            **(
                {"reference_transport": True, "target_version": 8}
                if enabled and capability
                else {}
            ),
        }
    ]


def _snapshot_driver():
    driver, pending, config, calls = _driver()
    methods = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_get_published_drafter_weights"},
        time=time,
    )
    driver._speco_get_published_drafter_weights = (
        methods._speco_get_published_drafter_weights.__get__(driver)
    )
    ready, submissions, gets = [False], [], []
    driver._require_speco_worker_group = lambda: SimpleNamespace(
        maybe_publish_async=lambda **kwargs: (
            submissions.append(kwargs) or ["snapshot-ref"]
        )
    )
    driver._speco_ray_refs_ready = lambda refs: ready[0]
    driver._ray_get_if_needed = lambda refs: (
        gets.append(refs) or [{"weights_ref": "immutable-host-snapshot"}]
    )
    driver._first_non_null = lambda rows: next(
        (row for row in rows if row is not None), None
    )
    return driver, pending, config, calls, ready, submissions, gets


def test_publish_snapshot_pending_never_blocks_before_upstream_weight_sync():
    driver, pending, _, calls, ready, submissions, gets = _snapshot_driver()
    staged, metrics = driver._speco_try_early_stage_drafter_publish(pending)
    assert not staged
    assert metrics["bubble/publish_snapshot_prefetch_pending_before_weight_sync"] == 1
    assert "bubble/publish_snapshot_prefetch_consumed" not in metrics
    assert not calls and not gets
    # Repeated readiness checks cannot pop/export the same snapshot twice.
    driver._speco_try_early_stage_drafter_publish(pending)
    assert len(submissions) == 1 and not gets
    ready[0] = True

    def publish(trained, plan):
        assert driver._speco_get_published_drafter_weights() == {
            "weights_ref": "immutable-host-snapshot"
        }
        driver._pending_drafter_publish_refs = ["stage-ref"]
        calls.append(plan.plan_id)
        return {"drafter/published": 1}

    driver._speco_publish_drafter_weights = publish
    staged, _ = driver._speco_try_early_stage_drafter_publish(pending)
    assert staged and len(submissions) == len(gets) == len(calls) == 1
    assert driver._speco_publish_snapshot_prefetch is None


def test_normal_safe_point_consumes_pending_snapshot_without_duplicate_rpc():
    driver, pending, _, _, _, submissions, gets = _snapshot_driver()
    driver._speco_try_early_stage_drafter_publish(pending)
    assert driver._speco_get_published_drafter_weights() == {
        "weights_ref": "immutable-host-snapshot"
    }
    assert len(submissions) == len(gets) == 1
    driver._speco_publish_snapshot_prefetch = {"source_step": 3, "refs": ["old-ref"]}
    with pytest.raises(RuntimeError, match="source version"):
        driver._speco_get_published_drafter_weights()


def test_recent_writer_steps_do_not_inherit_stale_ema_or_another_writer(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)
    config = replace(config, idle_worker_initial_batch_estimate_sec=None)
    scheduler._idle_worker_batch_estimate_sec = 0.333
    scheduler._idle_worker_batch_samples_sec.extend([0.333] * 8)
    scheduler._idle_steady_by_group[group] = deque(
        [0.17, 0.175, 0.168, 0.172], maxlen=4
    )
    assert scheduler._effective_idle_batch_estimate_sec(config, group) == pytest.approx(
        0.175
    )
    assert scheduler._effective_idle_batch_estimate_sec(
        config, ("2", "3")
    ) == pytest.approx(0.333)
    assert scheduler._effective_idle_batch_estimate_sec(
        replace(config, idle_worker_calibrated_budgets=False), group
    ) == pytest.approx(0.333)
    assert scheduler._effective_idle_batch_estimate_sec(
        replace(config, execution_strategy=DrafterExecutionStrategy.SYNC), group
    ) == pytest.approx(0.333)
    assert (
        scheduler._effective_idle_batch_estimate_sec(
            replace(config, idle_worker_initial_batch_estimate_sec=0.5), group
        )
        == 0.5
    )


def test_cleanup_calibration_requires_recent_same_writer_and_mode(monkeypatch):
    scheduler, config, group = _hot_writer(monkeypatch)

    def record(seconds, *, hot=True, final=False, workers=group):
        scheduler.record_idle_training_outcome(
            TrainingOutcome(
                trained=True,
                successful_steps=1,
                worker_results=[],
                raw_results=[],
                elapsed_sec=seconds,
                reason="trained",
                metrics={
                    "timing_s/drafter_worker_cleanup_nonpark": seconds,
                    "bubble/training_residency_retained": int(hot),
                },
            ),
            final=final,
            worker_ids=workers,
        )

    record(0.62)
    for value in (0.03, 0.04, 0.035, 0.038):
        record(value)
    record(4, final=True)
    record(8, hot=False)
    record(12, workers=("2", "3"))
    assert scheduler._effective_idle_tail_reserve_sec(
        config, worker_ids=group
    ) == pytest.approx(0.04)
    # No mature mode-specific sample set: retain the conservative fallback.
    assert (
        scheduler._effective_idle_tail_reserve_sec(
            replace(config, idle_worker_keep_hot_between_plans=False), worker_ids=group
        )
        >= 8
    )
    assert (
        scheduler._effective_idle_tail_reserve_sec(config, final=True, worker_ids=group)
        >= 4
    )


def test_optimization_flags_parse_independently():
    config = DrafterScheduleConfig.from_mapping(
        {
            "scheduler": {
                "idle_worker": {
                    "head_reference_transport": False,
                    "calibrated_budgets": False,
                    "early_publish_staging": True,
                }
            }
        }
    )
    assert not config.idle_worker_head_reference_transport
    assert not config.idle_worker_calibrated_budgets
    assert config.idle_worker_early_publish_staging
