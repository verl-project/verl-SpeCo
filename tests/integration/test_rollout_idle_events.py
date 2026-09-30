# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from types import SimpleNamespace

import verl_speco.integration.rollout_idle_events as rollout_idle_events


class _LocalRay:
    @staticmethod
    def remote(actor_class):
        return actor_class


def _local_event_bus():
    return rollout_idle_events._event_bus_actor_class(_LocalRay)()


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
