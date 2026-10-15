# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from types import SimpleNamespace

import pytest

import verl_speco.integration.rollout_idle_events as rollout_idle_events


class _LocalRay:
    @staticmethod
    def remote(actor_class):
        return actor_class


def _local_event_bus():
    return rollout_idle_events._event_bus_actor_class(_LocalRay)()


def test_idle_training_requires_live_serving_overlap_but_sync_does_not():
    bus = _local_event_bus()
    assert bus.reserve_training(
        ("0", "1"), "late", 0.0, require_active_rollout=True
    ) == {"acquired": False, "reason": "no_active_rollout"}
    assert not bus._training_owners
    assert bus.acquire_rollout(("2", "3"), {"lease_request_id": "peer"})["acquired"]
    assert bus.reserve_training(
        ("0", "1"), "overlap", 0.0, require_active_rollout=True
    )["acquired"]
    bus.release_training(("0", "1"), "overlap")
    bus.release_rollout(("2", "3"), {})
    assert not bus.reserve_training(
        ("0", "1"), "too-late", 0.0, require_active_rollout=True
    )["acquired"]
    # Explicit non-overlapped reservations retain the legacy behavior.
    assert bus.reserve_training(("0", "1"), "sync", 0.0)["acquired"]


def test_overlap_check_does_not_allow_training_on_a_serving_group():
    bus = _local_event_bus()
    assert bus.acquire_rollout(("0", "1"), {"lease_request_id": "own"})["acquired"]
    assert not bus.reserve_training(
        ("0", "1"), "busy", 0.0, require_active_rollout=True
    )["acquired"]


@pytest.mark.parametrize("cancel", [False, True])
def test_last_serving_release_reclaims_overlap_without_driver_poll(cancel):
    bus = _local_event_bus()
    for worker in ("2", "3"):
        assert bus.acquire_rollout((worker,), {"lease_request_id": worker})["acquired"]
    assert bus.reserve_training(("0",), "overlap", 0.0, require_active_rollout=True)[
        "acquired"
    ]
    # Legacy explicit reservations are not opportunistic overlap plans.
    assert bus.reserve_training(("1",), "explicit", 0.0)["acquired"]
    bus.release_rollout(("2",), {})
    assert not bus.training_reclaim_status("overlap", "0")["requested"]
    if cancel:
        bus.cancel_rollout_wait("3")
    else:
        bus.release_rollout(("3",), {})
    bus.drain()
    assert bus.training_reclaim_status("overlap", "0")["requested"]
    assert not bus.training_reclaim_status("explicit", "1")["requested"]
    bus.release_training(("0",), "overlap")
    assert "overlap" not in bus._overlap_required_plans
    assert not bus.training_reclaim_status("overlap", "0")["active"]


def test_cancelled_waiter_does_not_starve_future_idle_training():
    bus = _local_event_bus()
    bus.reserve_training(("0",), "old", 0.0)
    for request in ("a", "b"):
        assert not bus.acquire_rollout(("0",), {"lease_request_id": request})[
            "acquired"
        ]
    bus.cancel_rollout_wait("a")
    bus.release_training(("0",), "old")
    assert not bus.reserve_training(("0",), "new", 0.0)["acquired"]
    bus.cancel_rollout_wait("b")
    assert bus.reserve_training(("0",), "new", 0.0)["acquired"]


def test_cancelled_acquire_releases_only_its_own_idempotent_grant():
    bus = _local_event_bus()
    assert bus.acquire_rollout(("0",), {"lease_request_id": "a"})["acquired"]
    assert bus.acquire_rollout(("0",), {"lease_request_id": "a"})["acquired"]
    assert bus._rollout_owners["0"] == 1
    bus.acquire_rollout(("0",), {"lease_request_id": "b"})
    bus.cancel_rollout_wait("a")
    assert bus._rollout_owners["0"] == 1
    assert not bus.reserve_training(("0",), "new", 0.0)["acquired"]
    bus.release_rollout(("0",), {})
    assert not bus._rollout_requests
    assert bus.reserve_training(("0",), "new", 0.0)["acquired"]


@pytest.mark.parametrize("grant_before_cancel", [False, True])
def test_async_lease_cancellation_cleans_pending_and_late_granted_ownership(
    monkeypatch, grant_before_cancel
):
    import asyncio
    import pytest

    bus = _local_event_bus()
    if not grant_before_cancel:
        bus.reserve_training(("0",), "old", 0.0)

    def acquire(workers, event):
        return ("acquire", bus.acquire_rollout(workers, event))

    async def resolve(ray, value):
        if isinstance(value, tuple):
            raise asyncio.CancelledError()
        return value

    actor = SimpleNamespace(
        acquire_rollout=SimpleNamespace(remote=acquire),
        cancel_rollout_wait=SimpleNamespace(remote=bus.cancel_rollout_wait),
    )
    monkeypatch.setattr(
        rollout_idle_events, "_event_bus", lambda name: (object(), actor)
    )
    monkeypatch.setattr(rollout_idle_events, "_await_ray_result", resolve)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            rollout_idle_events.acquire_rollout_resource_lease(
                "bus", worker_ids=("0",), event={}
            )
        )
    assert not bus._pending_rollout_groups
    assert not bus._rollout_owners
    assert not bus._rollout_requests


@pytest.mark.parametrize("grant_before_timeout", [False, True])
def test_async_lease_timeout_bounds_rpc_and_cleans_ownership(
    monkeypatch, grant_before_timeout
):
    import asyncio

    bus = _local_event_bus()
    if not grant_before_timeout:
        bus.reserve_training(("0",), "old", 0.0)

    def acquire(workers, event):
        return ("acquire", bus.acquire_rollout(workers, event))

    async def resolve(ray, value):
        if isinstance(value, tuple):
            await asyncio.Event().wait()  # Simulate an RPC whose reply never arrives.
        return value

    actor = SimpleNamespace(
        acquire_rollout=SimpleNamespace(remote=acquire),
        cancel_rollout_wait=SimpleNamespace(remote=bus.cancel_rollout_wait),
    )
    monkeypatch.setattr(
        rollout_idle_events, "_event_bus", lambda name: (object(), actor)
    )
    monkeypatch.setattr(rollout_idle_events, "_await_ray_result", resolve)

    async def run():
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                rollout_idle_events.acquire_rollout_resource_lease(
                    "bus", worker_ids=("0",), event={}, timeout_sec=1.0
                ),
                timeout=3.0,
            )
        assert asyncio.get_running_loop().time() - started < 2.0
        assert not bus._pending_rollout_groups
        assert not bus._rollout_owners
        assert not bus._rollout_requests

    asyncio.run(run())


def test_out_of_band_reclaim_is_scoped_to_live_plan_and_not_drained():
    bus = _local_event_bus()
    bus.reserve_training(("0", "1"), "old", 0.0)
    assert bus.request_training_reclaim(("0",), 123.0)["requested"]
    bus.drain()  # Draining lifecycle events must not consume the stop flag.
    assert bus.training_reclaim_status("old", "1")["requested"]
    bus.release_training(("0", "1"), "old")
    bus.reserve_training(("0", "1"), "new", 0.0)
    assert not bus.training_reclaim_status("old", "0")["active"]
    assert not bus.training_reclaim_status("new", "0")["requested"]


def test_rollout_acquisition_signals_reclaim_without_driver_event_poll():
    bus = _local_event_bus()
    bus.reserve_training(("0",), "train", 0.0)
    bus.acquire_rollout(("0",), {"replica_rank": 0})
    assert bus.training_reclaim_status("train", "0")["requested"]


def test_waiting_generation_has_priority_over_idle_continuation():
    bus = _local_event_bus()
    bus.reserve_training(("0",), "old", 0.0)
    assert not bus.acquire_rollout(("0",), {"replica_rank": 0})["acquired"]
    bus.release_training(("0",), "old")
    assert not bus.reserve_training(("0",), "continuation", 0.0)["acquired"]
    assert bus.acquire_rollout(("0",), {"replica_rank": 0})["acquired"]


def test_cpu_reclaim_watcher_delivers_without_busy_worker_rpc(monkeypatch):
    import threading

    bus = _local_event_bus()
    bus.reserve_training(("0",), "plan", 0.0)
    proxy = SimpleNamespace(
        training_reclaim_status=SimpleNamespace(remote=bus.training_reclaim_status)
    )
    monkeypatch.setattr(
        rollout_idle_events,
        "_ray_module",
        lambda: SimpleNamespace(
            get_actor=lambda name: proxy,
            get=lambda result, **kwargs: result,
        ),
    )
    received = threading.Event()
    calls = []

    def receive(*args):
        calls.append(args)
        received.set()

    watcher = rollout_idle_events.TrainingReclaimWatcher(
        "bus", "plan", "0", receive
    ).start()
    try:
        bus.request_training_reclaim(("0",), 123.0)
        assert received.wait(1.0)
        assert calls == [("plan", 123.0, "lease_bus")]
    finally:
        watcher.close()
        watcher._thread.join(timeout=1.0)


def test_reclaim_control_ack_timeout_allows_compatibility_fallback(monkeypatch):
    def get(result, *, timeout):
        assert timeout == 1.0
        raise TimeoutError("control actor unavailable")

    actor = SimpleNamespace(
        request_training_reclaim=SimpleNamespace(remote=lambda *args: "ref")
    )
    monkeypatch.setattr(
        rollout_idle_events,
        "_event_bus",
        lambda name: (SimpleNamespace(get=get), actor),
    )
    assert not rollout_idle_events.request_rollout_training_reclaim(
        "bus", ("0",), 123.0
    )


def test_ray_control_reaches_worker_while_its_async_execution_loop_is_busy():
    import os
    import time
    import uuid
    import pytest

    if os.name == "nt":
        pytest.skip("Ray process integration runs on the Linux training host")
    ray = pytest.importorskip("ray")
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=2, include_dashboard=False)
    name = "speco-control-test-" + uuid.uuid4().hex
    bus = rollout_idle_events.ensure_rollout_idle_event_bus(name)

    @ray.remote
    class BusyWorker:
        def __init__(self):
            self.stop = False

        def receive(self, *args):
            self.stop = True

        async def train(self):
            watcher = rollout_idle_events.TrainingReclaimWatcher(
                name, "plan", "0", self.receive
            ).start()
            try:
                await bus.emit.remote({"event_type": "started"})
                count = 0
                for _ in range(50):
                    if self.stop:
                        return count
                    time.sleep(0.1)  # Deliberately no asyncio yield for RPC delivery.
                    count += 1
                return count
            finally:
                watcher.close()

    worker = BusyWorker.remote()
    try:
        ray.get(bus.reserve_training.remote(("0",), "plan", 0.0))
        training = worker.train.remote()
        deadline = time.monotonic() + 15
        while not ray.get(bus.drain.remote()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        ray.get(bus.request_training_reclaim.remote(("0",), time.time()))
        assert ray.get(training, timeout=5) < 10
    finally:
        ray.kill(worker, no_restart=True)
        ray.kill(bus, no_restart=True)
        if started_ray:
            ray.shutdown()


def test_resource_lease_reclaims_training_before_rollout_starts() -> None:
    bus = _local_event_bus()
    bus.configure_replica_groups({0: ("0", "1")})

    reserved = bus.reserve_training(("0", "1"), "plan-1", 0.0)
    blocked = bus.acquire_rollout(
        ("0",),
        {"worker_id": "replica-0", "replica_rank": 0},
    )

    assert reserved["acquired"] is True
    assert blocked == {
        "acquired": False,
        "blocking_plan_ids": ["plan-1"],
        "retry_after_sec": 0.01,
    }
    blocked_events = bus.drain()
    assert len(blocked_events) == 1
    assert blocked_events[0]["worker_id"] == "replica-0"
    assert blocked_events[0]["replica_rank"] == 0
    assert blocked_events[0]["event_type"] == "GENERATION_STARTED"
    assert blocked_events[0]["release_source"] == "rollout_waiting_for_training"
    assert isinstance(blocked_events[0]["event_ts"], float)

    assert bus.release_training(("0", "1"), "plan-1")["released"] is True
    acquired = bus.acquire_rollout(
        ("0",),
        {"worker_id": "replica-0", "replica_rank": 0},
    )
    assert acquired["acquired"] is True
    # The blocked request already emitted the reclaim event, so retrying after
    # release must not emit a duplicate generation-start transition.
    assert bus.drain() == []

    released = bus.release_rollout(
        ("0",),
        {
            "worker_id": "replica-0",
            "replica_rank": 0,
            "memory_released": True,
        },
    )
    idle_events = bus.drain()
    assert released == {"released": True, "idle": True}
    assert len(idle_events) == 1
    assert idle_events[0]["event_type"] == "WORKER_IDLE"
    assert idle_events[0]["idle_confidence"] == "confirmed"


def test_resource_lease_rejects_training_while_rollout_is_active() -> None:
    bus = _local_event_bus()
    bus.configure_replica_groups({0: ("0", "1")})
    acquired = bus.acquire_rollout(
        ("0",),
        {"worker_id": "replica-0", "replica_rank": 0},
    )

    reserved = bus.reserve_training(("0", "1"), "plan-1", 0.0)

    assert acquired["acquired"] is True
    assert reserved["acquired"] is False
    assert reserved["rollout_busy_workers"] == ["0", "1"]


def test_rollout_group_becomes_idle_only_after_all_owners_release() -> None:
    bus = _local_event_bus()
    bus.configure_replica_groups({0: ("0", "1")})
    event = {"worker_id": "replica-0", "replica_rank": 0}

    assert bus.acquire_rollout(("0",), event)["acquired"] is True
    assert bus.acquire_rollout(("0",), event)["acquired"] is True
    # Only the first owner transition emits GENERATION_STARTED.
    assert [item["event_type"] for item in bus.drain()] == ["GENERATION_STARTED"]

    first_release = bus.release_rollout(
        ("0",),
        {**event, "memory_released": True},
    )
    assert first_release == {"released": True, "idle": False}
    assert bus.drain() == []
    assert bus.reserve_training(("0", "1"), "plan-1", 0.0)["acquired"] is False

    final_release = bus.release_rollout(
        ("0",),
        {**event, "memory_released": True},
    )
    assert final_release == {"released": True, "idle": True}
    assert [item["event_type"] for item in bus.drain()] == ["WORKER_IDLE"]
    assert bus.reserve_training(("0", "1"), "plan-1", 0.0)["acquired"] is True


def test_drafter_sample_event_uses_object_store_reference(monkeypatch) -> None:
    emitted = []
    sample_ref = object()
    actor = SimpleNamespace(
        emit=SimpleNamespace(remote=lambda event: emitted.append(event))
    )
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        get_actor=lambda name: actor,
        put=lambda sample: sample_ref,
    )
    monkeypatch.setattr(rollout_idle_events, "_ray_module", lambda: fake_ray)

    emitted_ok = rollout_idle_events.emit_rollout_drafter_sample(
        "bubble-bus",
        {"hidden_states": "large-payload"},
        sample_id="5:0:req-1",
        replica_rank=0,
        global_step=5,
    )

    assert emitted_ok is True
    assert emitted == [
        {
            "event_type": rollout_idle_events.DRAFTER_SAMPLE_READY_EVENT,
            "sample_id": "5:0:req-1",
            "sample_ref": sample_ref,
            "replica_rank": 0,
            "global_step": 5,
            "event_ts": emitted[0]["event_ts"],
        }
    ]
