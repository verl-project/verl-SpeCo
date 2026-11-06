# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Sync boundaries must not pause the engine needed by probes and rollout."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration.rollout_admission import RolloutAdmission
from verl_speco.trainer.bubble_lifecycle import CostClock, QuotaCostLedger
from verl_speco.trainer.v1.bubble_mixin import V1BubbleMixin


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
        costs=QuotaCostLedger(), receipts=[], clock=CostClock(),
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
