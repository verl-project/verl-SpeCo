# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU-only contracts for cached-host staging; no live-engine apply is allowed."""

import ast
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_bubble_review_regressions import _freeze_publish_driver, _methods
from verl_speco.trainer.scheduler import DrafterExecutionStrategy, DrafterScheduleConfig


def _driver():
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_try_early_stage_drafter_publish"},
        time=time,
        DrafterExecutionStrategy=DrafterExecutionStrategy,
    )
    driver = cls()
    driver.global_steps = 4
    driver._speco_rollout_idle_worker_enabled = lambda: True
    config = DrafterScheduleConfig(
        publish_async=True,
        training_quota_enable=True,
        idle_worker_early_publish_staging=True,
    )
    driver._speco_drafter_schedule_config = lambda: config
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        _training_quota_debt_steps=0
    )
    driver._pending_drafter_publish_refs = None
    plan = SimpleNamespace(
        plan_id="quota-4",
        source_global_step=4,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        launch=True,
        publish_after_success=True,
    )
    pending = {"training_plan": plan, "drafter_trained": True, "actor_output": None}
    calls = []

    def publish(trained, requested_plan):
        assert driver._speco_publish_cached_snapshot_only
        calls.append((trained, requested_plan.plan_id))
        driver._pending_drafter_publish_refs = ["stage-rpc"]
        return {"drafter/published": 1}

    driver._speco_publish_drafter_weights = publish
    return driver, pending, config, calls


def test_early_staging_is_opt_in_and_parses_from_idle_worker_config():
    assert not DrafterScheduleConfig().idle_worker_early_publish_staging
    config = DrafterScheduleConfig.from_mapping(
        {"scheduler": {"idle_worker": {"early_publish_staging": True}}}
    )
    assert config.idle_worker_early_publish_staging


@pytest.mark.parametrize(
    "guard",
    [
        "sync",
        "disabled",
        "synchronous_publish",
        "sync_plan",
        "incomplete",
        "not_trained",
        "not_launched",
        "stale",
        "pending",
        "generation",
        "validation",
        "checkpoint",
        "debt",
    ],
)
def test_early_staging_never_touches_ineligible_publications(guard):
    driver, pending, config, calls = _driver()
    if guard == "sync":
        driver._speco_rollout_idle_worker_enabled = lambda: False
        driver._speco_drafter_schedule_config = lambda: pytest.fail(
            "Sync reads no new config"
        )
    elif guard == "disabled":
        driver._speco_drafter_schedule_config = lambda: replace(
            config, idle_worker_early_publish_staging=False
        )
    elif guard == "synchronous_publish":
        driver._speco_drafter_schedule_config = lambda: replace(
            config, publish_async=False
        )
    elif guard == "sync_plan":
        pending["training_plan"].execution_strategy = DrafterExecutionStrategy.SYNC
    elif guard == "incomplete":
        pending["training_plan"].publish_after_success = False
    elif guard == "not_trained":
        pending["drafter_trained"] = False
    elif guard == "not_launched":
        pending["training_plan"].launch = False
    elif guard == "stale":
        pending["training_plan"].source_global_step = 3
    elif guard == "pending":
        driver._pending_drafter_publish_refs = ["previous-stage"]
    elif guard == "debt":
        driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
            _training_quota_debt_steps=1
        )
    else:
        name = {
            "generation": "_speco_rollout_idle_generation_open",
            "validation": "_speco_validation_in_progress",
            "checkpoint": "_speco_checkpoint_in_progress",
        }[guard]
        setattr(driver, name, True)
    assert driver._speco_try_early_stage_drafter_publish(pending) == (False, {})
    assert calls == []


def test_early_staging_only_submits_once_and_clears_cache_only_scope():
    driver, pending, _, calls = _driver()
    staged, metrics = driver._speco_try_early_stage_drafter_publish(pending)
    assert staged and metrics["bubble/early_publish_staging_submitted"] == 1
    assert not driver._speco_publish_cached_snapshot_only
    assert driver._speco_try_early_stage_drafter_publish(pending) == (False, {})
    assert calls == [(True, "quota-4")]


@pytest.mark.parametrize("fail", [False, True])
def test_cache_miss_retains_fallback_and_ambiguous_failure_is_not_retried(fail):
    driver, pending, _, _ = _driver()

    def publish(*args):
        if fail:
            raise RuntimeError("stage submission failed")
        return {"drafter/published": 0}

    driver._speco_publish_drafter_weights = publish
    if fail:
        with pytest.raises(RuntimeError, match="stage submission failed"):
            driver._speco_try_early_stage_drafter_publish(pending)
    else:
        staged, metrics = driver._speco_try_early_stage_drafter_publish(pending)
        assert not staged and metrics["bubble/early_publish_staging_cache_miss"] == 1
        assert pending["training_plan"].plan_id == "quota-4"
    assert not driver._speco_publish_cached_snapshot_only


@pytest.mark.parametrize("cached_only", [False, True])
def test_snapshot_rpc_preserves_legacy_sync_signature(cached_only):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"speco_maybe_publish", "_speco_get_published_drafter_weights"},
    )
    driver = cls()
    calls = []
    driver._speco_publish_cached_snapshot_only = cached_only
    driver._require_speco_worker_group = lambda: SimpleNamespace(
        maybe_publish=lambda **kwargs: calls.append(kwargs) or [None, "host-payload"]
    )
    driver._ray_get_if_needed = lambda value: value
    driver._first_non_null = lambda values: next(v for v in values if v is not None)
    assert driver._speco_get_published_drafter_weights() == "host-payload"
    assert calls == ([{"cached_snapshot_only": True}] if cached_only else [{}])


def test_cached_only_worker_cannot_export_deferred_gpu_snapshot():
    cls = _methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"maybe_publish"},
    )
    worker = cls()
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.last_global_step = worker.last_trained_step = 4
    worker._deferred_publish_snapshot_step = 4
    worker._deferred_publish_optimizer_step = 20
    worker.trainer = SimpleNamespace(
        prepare_model_state_dict_for_publish=lambda *args: pytest.fail("GPU export"),
        pop_model_state_dict_for_publish=lambda *args: pytest.fail("cache consumed"),
    )
    assert worker.maybe_publish(cached_snapshot_only=True) is None
    assert worker._deferred_publish_snapshot_step == 4
    assert worker._deferred_publish_optimizer_step == 20


def _weight_update_hook(driver, queue, original, drain, *, diagnostics=False):
    path = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hook = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "update_weights_with_speco"
    )
    namespace = dict(
        self=driver,
        pending_drafter_publishes=queue,
        original_checkpoint_update_weights=original,
        drain_deferred_drafter_publishes=drain,
        time=time,
        logger=SimpleNamespace(warning=lambda *args: None),
        pending_drafter_publish={"ready": False},
        phase_diagnostics=diagnostics,
        control_snapshot=lambda owner: {},
        emit_summary=lambda *args, **kwargs: None,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            hook,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[hook.name]


@pytest.mark.parametrize("bubble,staged", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_production_hook_orders_stage_before_sync_and_retains_fallback(
    bubble, staged, diagnostics
):
    driver, pending, _, _ = _driver()
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    events = []
    queue = [pending]
    driver._speco_record_rollout_idle_metrics = lambda metrics: None
    driver._speco_park_drafter_training_memory = lambda reason: {}
    output_metrics = []
    driver._speco_update_output_metrics = lambda output, metrics: (
        output_metrics.append(dict(metrics)) or output
    )
    driver._speco_try_early_stage_drafter_publish = lambda item: (
        events.append("stage") or staged,
        {
            "drafter/published": int(staged),
            "timing_s/drafter_publish_fetch_snapshot": 1.0,
        },
    )

    def original():
        events.append("upstream_sync")
        return "result"

    def drain(**kwargs):
        events.append("fallback_publish" if queue else "no_duplicate_publish")
        return (
            {"drafter/published": 1, "timing_s/drafter_publish_fetch_snapshot": 2.0}
            if queue
            else {}
        )

    assert _weight_update_hook(
        driver, queue, original, drain, diagnostics=diagnostics
    )(None) == "result"
    assert events == (
        ["upstream_sync"]
        if not bubble
        else [
            "stage",
            "upstream_sync",
            "no_duplicate_publish" if staged else "fallback_publish",
        ]
    )
    assert len(queue) == (0 if staged else 1)
    if bubble:
        assert output_metrics[-1]["drafter/published"] == 1
        assert output_metrics[-1]["timing_s/drafter_publish_fetch_snapshot"] == (
            1.0 if staged else 3.0
        )


def test_real_two_phase_publish_still_requires_apply_ack_after_staging():
    driver, plan, stages, applied, completed = _freeze_publish_driver()
    helper_driver, _, config, _ = _driver()
    driver._speco_drafter_schedule_config = lambda: config
    driver._speco_get_drafter_scheduler()._training_quota_debt_steps = 0
    plan.launch = plan.publish_after_success = True
    plan.source_global_step = 4
    staged, _ = type(helper_driver)._speco_try_early_stage_drafter_publish(
        driver, {"training_plan": plan, "drafter_trained": True}
    )
    assert staged
    assert not driver._speco_publish_cached_snapshot_only
    assert stages == ["stage_draft_weights_async"]
    assert applied == completed == []
    assert driver._speco_freeze_policy.drafter_version == 0
    driver._speco_wait_pending_drafter_publish_rpc()
    assert applied == completed == [4]
    assert driver._speco_freeze_policy.drafter_version == 1
