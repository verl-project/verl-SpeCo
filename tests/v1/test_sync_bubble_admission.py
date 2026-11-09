# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Sync boundaries must not pause the engine needed by probes and rollout."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from verl_speco.integration.rollout_admission import RolloutAdmission
from verl_speco.trainer.bubble_lifecycle import CostClock, QuotaCostLedger
from verl_speco.trainer.v1.bubble_mixin import V1BubbleMixin


def test_admission_timeout_identifies_pending_replica(monkeypatch):
    refs = [object(), object()]
    seen = []

    def wait(submitted, *, num_returns, timeout):
        assert submitted == refs
        assert num_returns == 2
        assert timeout == 3
        return refs[:1], refs[1:]

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(wait=wait, get=seen.append))
    driver = V1BubbleMixin()
    driver._speco_v1_idle_option = lambda name, default: 3
    driver.llm_server_manager = SimpleNamespace(
        server_handles=[
            SimpleNamespace(
                speco_wait_rollout_idle=SimpleNamespace(remote=lambda ref=ref: ref)
            )
            for ref in refs
        ]
    )
    with pytest.raises(TimeoutError, match=r"speco_wait_rollout_idle.*indices \[1\]"):
        driver._speco_v1_admission("speco_wait_rollout_idle")
    assert seen == refs[:1]


@pytest.mark.parametrize("restore_fails", [False, True])
def test_checkpoint_failure_restores_weights_before_leaving_boundary(restore_fails):
    driver = V1BubbleMixin()
    driver.global_steps = 1
    driver._speco_bubble_session().close = lambda boundary: {}
    events = []
    driver._speco_v1_admission = events.append

    def restore(step):
        assert step == 1
        events.append("restore")
        if restore_fails:
            raise OSError("restore failed")

    driver.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep"), update_weights=restore
    )
    message = "admission remains closed" if restore_fails else "save failed"
    with pytest.raises(RuntimeError, match=message):
        with driver._speco_v1_quiescent_boundary(charge_cost=False):
            with driver._speco_v1_checkpoint_replicas_asleep():
                events.append("save")
                raise RuntimeError("save failed")
    assert events == [
        "speco_close_rollout_admission",
        "speco_wait_rollout_idle",
        "sleep",
        "save",
        "restore",
    ]
    assert driver._speco_colocate_admission_closed


def test_admission_retry_waits_for_late_close(monkeypatch):
    close_ref, resume_ref = object(), object()
    calls = []
    close_finished = False

    def wait(refs, *, num_returns, timeout):
        assert num_returns == len(refs)
        if refs == [close_ref] and not close_finished:
            return [], refs
        return refs, []

    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(wait=wait, get=lambda ref: None)
    )
    driver = V1BubbleMixin()
    driver._speco_v1_idle_option = lambda name, default: default
    driver.llm_server_manager = SimpleNamespace(
        server_handles=[
            SimpleNamespace(
                speco_close_rollout_admission=SimpleNamespace(remote=lambda: close_ref),
                speco_resume_rollout_admission=SimpleNamespace(
                    remote=lambda: (calls.append("resume"), resume_ref)[1]
                ),
            )
        ]
    )
    with pytest.raises(TimeoutError, match="timed out"):
        driver._speco_v1_admission("speco_close_rollout_admission")
    with pytest.raises(TimeoutError, match="still in flight"):
        driver._speco_v1_admission("speco_resume_rollout_admission")
    assert calls == []
    close_finished = True
    driver._speco_v1_admission("speco_resume_rollout_admission")
    assert calls == ["resume"]


def test_revision_timeout_reuses_applied_publish_and_before_probe(monkeypatch):
    driver = V1BubbleMixin()
    session = driver._speco_bubble_session()
    session.pending.append(
        (SimpleNamespace(plan_id="quota"), SimpleNamespace(trained=True))
    )
    driver._speco_v1_idle_option = lambda name, default: default
    driver._speco_freeze_policy = SimpleNamespace(
        drafter_version=4, probe_due=lambda version: True
    )
    events = []
    revision_ready = False
    ref = object()

    def wait(refs, **kwargs):
        assert refs == [ref]
        return (refs, []) if revision_ready else ([], refs)

    def publish(*args, **kwargs):
        events.append("apply")
        driver._speco_v1_publish_ack = True
        return {"drafter/published": 1}

    def revision(version):
        events.append(version)
        return ref

    def probe():
        revision = driver._speco_v1_probe_expected_revision
        events.append(revision)
        return revision

    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(wait=wait, get=lambda refs: None)
    )
    driver.llm_server_manager = SimpleNamespace(
        server_handles=[
            SimpleNamespace(
                speco_set_serving_drafter_revision=SimpleNamespace(remote=revision)
            )
        ]
    )
    driver._speco_publish_drafter_weights = publish
    driver._speco_wait_pending_drafter_publish = lambda: None
    driver._speco_run_freeze_probe_arm = probe
    driver._speco_build_freeze_probe_comparison = lambda before, after, **kwargs: (
        before,
        after,
    )

    with pytest.raises(TimeoutError, match="revision sync"):
        driver._speco_v1_publish_bubble()
    assert session.pending and not session.receipts
    revision_ready = True
    assert driver._speco_v1_publish_bubble()["drafter/published"] == 1
    assert events == [4, "apply", 5, 5]
    assert not session.pending
    assert session.receipts[0]["probe"] == (4, 5)


def test_boundary_retry_recloses_admission_after_resume_failure():
    driver = V1BubbleMixin()
    driver._speco_bubble_session().close = lambda boundary: {}
    calls = []

    def admission(method):
        calls.append(method)
        if method == "speco_resume_rollout_admission" and len(calls) == 3:
            raise TimeoutError("partial resume")

    driver._speco_v1_admission = admission
    with pytest.raises(TimeoutError, match="partial resume"):
        with driver._speco_v1_quiescent_boundary(charge_cost=False):
            pass
    with driver._speco_v1_quiescent_boundary(charge_cost=False):
        assert calls[-2:] == [
            "speco_close_rollout_admission",
            "speco_wait_rollout_idle",
        ]


def test_step_end_fast_path_consumes_acknowledged_receipt():
    driver = V1BubbleMixin()
    driver.global_steps = 2
    driver.total_training_steps = 3
    session = driver._speco_bubble_session()
    session.close = lambda boundary: {}
    session.receipts.append({"costs": QuotaCostLedger(), "probe": None})
    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        _training_quota_debt_steps=0, training_work_metrics=lambda: {}
    )
    driver._speco_wait_target_lm_head_prefetch = lambda: None
    observations = []
    driver._speco_freeze_observe_publish = (
        lambda metrics, **kwargs: observations.append(metrics)
    )

    def parent():
        assert not session.receipts
        return "target-synced"

    assert driver._speco_v1_bubble_step_end(parent) == "target-synced"
    assert observations == [{"drafter/published": 1}]


def test_transition_log_failure_does_not_replay_acknowledged_receipt():
    driver = V1BubbleMixin()
    session = driver._speco_bubble_session()
    session.receipts.append({"costs": QuotaCostLedger(), "probe": None})
    driver._speco_freeze_policy_active = True
    events = []

    def observe(metrics, **kwargs):
        events.append("observe")
        driver._speco_last_freeze_decision = SimpleNamespace(should_train=False)

    def emit(decision):
        raise BrokenPipeError("transition log failed")

    driver._speco_freeze_observe_publish = observe
    driver._speco_record_freeze_branch_request = lambda decision: events.append("save")
    driver._speco_freeze_emit_transition = emit

    with pytest.raises(BrokenPipeError, match="transition log failed"):
        driver._speco_v1_finish_receipts()
    assert driver._speco_drafter_frozen
    assert not session.receipts
    assert driver._speco_v1_finish_receipts() == {}
    assert events == ["observe", "save"]


def test_closed_admission_drains_existing_request_and_blocks_new_requests():
    async def scenario():
        gate = RolloutAdmission()
        entered, finish = asyncio.Event(), asyncio.Event()

        async def inflight():
            async with gate.request():
                entered.set()
                await finish.wait()

        async def next_request():
            async with gate.request():
                return "next rollout"

        running = asyncio.create_task(inflight())
        await entered.wait()
        gate.close()
        draining = asyncio.create_task(gate.wait_idle())
        queued = asyncio.create_task(next_request())
        await asyncio.sleep(0)
        assert not draining.done()
        assert not queued.done()
        finish.set()
        await running
        await draining
        assert not queued.done()
        gate.resume()
        assert await queued == "next rollout"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_sync_boundary_drains_without_engine_pause(nested, fails):
    gate = RolloutAdmission()
    events = []
    engine = SimpleNamespace(paused=False)

    def abort():
        engine.paused = True
        pytest.fail("sync Bubble must not pause the vLLM engine")

    class Driver(V1BubbleMixin):
        def _speco_bubble_session(self):
            return session

        def _speco_v1_admission(self, method):
            events.append(method)
            if method == "speco_close_rollout_admission":
                gate.close()
            elif method == "speco_wait_rollout_idle":
                asyncio.run(gate.wait_idle())
            elif method == "speco_resume_rollout_admission":
                gate.resume()

    session = SimpleNamespace(
        close=lambda boundary: events.append("reclaim"),
        costs=QuotaCostLedger(),
        receipts=[],
        clock=CostClock(),
    )
    driver = Driver()
    driver.global_steps = 1
    driver.checkpoint_manager = SimpleNamespace(abort_replicas=abort)
    driver._speco_colocate_admission_closed = nested
    if nested:
        gate.close()

    def probe():
        assert not engine.paused
        assert not gate._open.is_set()
        assert "speco_wait_rollout_idle" in events
        events.append("probe")

    def boundary():
        with driver._speco_v1_quiescent_boundary():
            probe()  # Dedicated probe RPC bypasses ordinary admission.
            if fails:
                raise RuntimeError("publication failed")

    if fails:
        with pytest.raises(RuntimeError, match="publication failed"):
            boundary()
    else:
        boundary()
    assert not engine.paused
    assert gate._open.is_set() == (not nested and not fails)
    assert driver._speco_colocate_admission_closed == (nested or fails)
    if not nested and not fails:

        async def rollout():
            async with gate.request():
                assert not engine.paused
                return "generated"

        assert asyncio.run(rollout()) == "generated"


@pytest.mark.parametrize("final", [False, True])
def test_quota_topup_charges_each_replica_sleep_and_the_final_park(final):
    driver = V1BubbleMixin()
    session = driver._speco_bubble_session()
    now = 0.0
    session.clock = SimpleNamespace(now=lambda: now)
    session.close = lambda boundary: {}
    driver.global_steps = 1
    driver.total_training_steps = 1 if final else 2
    driver._speco_v1_admission = lambda method: None
    driver._speco_wait_target_lm_head_prefetch = lambda: None
    driver._speco_v1_prefetch_publish_snapshot = lambda: {}
    driver._speco_v1_publish_bubble = lambda: {}

    scheduler = SimpleNamespace(
        _training_quota_debt_steps=2,
        training_work_metrics=lambda: {},
        needs_idle_optimizer_bootstrap=lambda config: False,
        prepare_training_execution=lambda plan: {},
    )

    def plan():
        return SimpleNamespace(
            plan_id=f"plan-{scheduler._training_quota_debt_steps}",
            launch=True,
            publish_after_success=True,
        )

    scheduler._plan_training_quota_topup = lambda *args: plan()
    driver._speco_get_drafter_scheduler = lambda: scheduler
    driver._speco_drafter_schedule_context = lambda: None
    driver._speco_drafter_schedule_config = lambda: SimpleNamespace(
        training_quota_allow_critical_path_fallback=True,
    )
    driver._speco_on_before_actor_update = lambda **kwargs: SimpleNamespace(
        training_plan=plan()
    )

    def sleep():
        nonlocal now
        now += 2.0

    def park(boundary):
        nonlocal now
        now += 3.0
        return {}

    def train(plan):
        scheduler._training_quota_debt_steps -= 1
        return True, {}

    driver.checkpoint_manager = SimpleNamespace(sleep_replicas=sleep)
    driver._speco_park_drafter_training_memory = park
    driver._speco_train_drafter = train
    driver._speco_v1_bubble_step_end(lambda: None)

    expected = (
        {"final_sleep:2": (0.0, 2.0), "final_sleep:1": (2.0, 4.0), "park": (4.0, 7.0)}
        if final
        else {"sleep": (0.0, 2.0), "park": (2.0, 5.0)}
    )
    assert {
        key.removeprefix("topup:1:"): interval
        for key, interval in session.costs.intervals.items()
        if key.startswith("topup:1:")
    } == expected
