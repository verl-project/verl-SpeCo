# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from verl_speco.trainer.data_buffer import DataBuffer


def test_global_sample_id_is_deduplicated_and_released_after_eviction() -> None:
    buffer = DataBuffer(max_size=2)

    assert buffer.add_batch({"_speco_global_sample_id": "a"})
    assert not buffer.add_batch({"_speco_global_sample_id": "a"})
    assert buffer.add_batch({"_speco_global_sample_id": "b"})
    assert buffer.add_batch({"_speco_global_sample_id": "c"})
    assert buffer.add_batch({"_speco_global_sample_id": "a"})

    assert [item["_speco_global_sample_id"] for item in buffer.get_all_data()] == [
        "c",
        "a",
    ]


def test_consumed_global_sample_can_be_collected_again() -> None:
    buffer = DataBuffer(max_size=2)
    sample = {"_speco_global_sample_id": "a", "target_version": 1}
    assert buffer.add_batch(sample)
    reserved = buffer.reserve("plan", target_version=1, max_samples=1)

    assert buffer.consume("plan", reserved) == 1
    assert buffer.add_batch({"_speco_global_sample_id": "a"})


def test_full_buffer_rejects_late_samples_instead_of_evicting_reservations():
    buffer = DataBuffer(max_size=2)
    for sample_id in ("a", "b"):
        buffer.add_batch({"_speco_global_sample_id": sample_id, "target_version": 4})
    reserved = buffer.reserve("running", target_version=4, max_samples=2)
    assert not buffer.add_batch({"_speco_global_sample_id": "c", "target_version": 4})
    assert (
        buffer.get_available_data(target_version=4, reservation_id="running")
        == reserved
    )
    assert len(buffer) == 2
    buffer.release_reservation("running")
    assert buffer.add_batch({"_speco_global_sample_id": "c", "target_version": 4})


def test_replay_pins_survive_plan_release_and_allow_unprotected_fifo_eviction():
    buffer = DataBuffer(max_size=3)
    for sample_id in ("a", "b", "c"):
        buffer.add_batch({"_speco_global_sample_id": sample_id, "target_version": 4})
    snapshot = buffer.reserve("first", target_version=4, max_samples=2)
    buffer.pin_replay_samples(snapshot)
    buffer.release_reservation("first")
    assert buffer.add_batch({"_speco_global_sample_id": "d", "target_version": 4})
    assert [item["_speco_global_sample_id"] for item in buffer.get_all_data()] == [
        "a",
        "b",
        "d",
    ]
    assert buffer.reserve_samples("next", snapshot) == snapshot
    buffer.unpin_replay_samples(snapshot)
    assert buffer.consume("next", snapshot) == 2
    assert buffer.add_batch({"_speco_global_sample_id": "a", "target_version": 5})


def test_replay_unpin_restores_fifo_for_unused_quota_samples():
    buffer = DataBuffer(max_size=2)
    for index in range(2):
        buffer.add_batch({"id": index, "target_version": 4})
    snapshot = buffer.reserve("plan", target_version=4, max_samples=2)
    buffer.pin_replay_samples(snapshot)
    buffer.unpin_replay_samples(snapshot)
    buffer.release_reservation("plan")
    assert buffer.add_batch({"id": 2, "target_version": 5})
    assert [item["id"] for item in buffer.get_all_data()] == [1, 2]
