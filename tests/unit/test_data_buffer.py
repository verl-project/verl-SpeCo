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
