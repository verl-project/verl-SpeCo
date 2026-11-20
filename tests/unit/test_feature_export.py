# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Completion, capacity and cancellation contracts for computed features."""

import threading

import pytest
import torch

from verl_speco.producer.feature_export import (
    FeatureExportQueue,
    wait_for_exports,
    rollout_feature_fields,
)


@pytest.mark.parametrize("packed", [False, True])
def test_verified_export_roundtrip_and_corruption(tmp_path, packed):
    from verl_speco.producer.feature_export import (
        load_export_chunks,
        verify_feature_digest,
    )

    (tmp_path / "live").touch()
    queue = FeatureExportQueue(pack_chunks=packed)
    payload = {
        "sample_id": "sample",
        "layer_ids": [2],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
        "hidden_states": torch.randn(2, 8).bfloat16(),
        "positions": torch.tensor([3, 4]),
        "token_ids": torch.tensor([10, 11]),
        "transport_snapshot_verified": True,
    }
    assert queue.reserve(1024)
    queue.submit_reserved(tmp_path, payload, size=1024)
    queue.close()
    result = list(load_export_chunks(tmp_path))[0]
    assert torch.equal(
        result["hidden_states"].view(torch.int16),
        payload["hidden_states"].view(torch.int16),
    )
    result["positions"][0] += 1
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_feature_digest(result)


def test_unlimited_export_accepts_full_request_burst(tmp_path):
    (tmp_path / "live").touch()
    gate = threading.Event()
    queue = FeatureExportQueue(max_bytes=None, max_pending=None)
    size = 10 << 20
    try:
        for index in range(32):
            assert queue.reserve(size)
            queue.submit_reserved(
                tmp_path, {"index": index}, size=size, wait=lambda: gate.wait(10)
            )
        assert queue._pending == 32
        assert queue._bytes == 320 << 20
        assert not queue.reserve(-1)
    finally:
        gate.set()
        queue.close()
    assert queue._pending == queue._bytes == 0
    assert len(list(tmp_path.glob("*.pt"))) == 32
    assert not list(tmp_path.glob("*.pending"))


def test_export_is_nonblocking_bounded_and_waits_for_completion(tmp_path):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue(max_bytes=32, max_pending=1)
    entered, release = threading.Event(), threading.Event()
    payload = {"hidden": torch.tensor([1.0, 2.0])}

    def wait():
        entered.set()
        assert release.wait(5)

    try:
        assert queue.reserve(32)
        queue.submit_reserved(tmp_path, payload, size=32, wait=wait)
        assert entered.wait(5)
        assert not list(tmp_path.glob("*.pt"))
        assert not queue.reserve(1)
        with pytest.raises(TimeoutError):
            wait_for_exports(tmp_path, timeout=0)
    finally:
        release.set()
        queue.close()
    wait_for_exports(tmp_path)
    data = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert torch.equal(data["hidden"], payload["hidden"])
    assert queue.reserve(32)
    queue.release(32)


def test_export_failure_is_visible_and_releases_capacity(tmp_path, monkeypatch):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue(max_bytes=16)

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(torch, "save", fail)
    assert queue.reserve(16)
    queue.submit_reserved(tmp_path, {}, size=16)
    queue.close()
    assert (tmp_path / "error").exists()
    assert not list(tmp_path.glob("*.pending"))
    assert queue.reserve(16)
    queue.release(16)


def test_export_records_host_timing_without_changing_features(tmp_path):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue()
    hidden = torch.tensor([[1.0, 2.0]])
    assert queue.reserve(32)
    queue.submit_reserved(
        tmp_path, {"sample_id": "test", "hidden_states": hidden}, size=32
    )
    queue.close()
    result = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert torch.equal(result["hidden_states"], hidden)
    assert result["capture_timing"]["queue_s"] >= 0
    assert result["capture_timing"]["wait_filter_s"] >= 0


def test_cancellation_before_completion_does_not_publish(tmp_path):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue()
    gate = threading.Event()
    assert queue.reserve(16)
    queue.submit_reserved(tmp_path, {}, size=16, wait=lambda: gate.wait(5))
    (tmp_path / "live").unlink()
    gate.set()
    queue.close()
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.pending"))


def test_shared_fields_preserve_tokens_positions_and_storage():
    prompt, response = torch.tensor([1, 2]), torch.tensor([3, 4])
    hidden = torch.randn(2, 8)
    positions = torch.tensor([1, 2])
    sample = rollout_feature_fields(prompt, response, hidden, positions)
    assert sample["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert sample["hidden_states"].data_ptr() == hidden.data_ptr()
    assert sample["hidden_positions"].tolist() == [[1, 2]]
    with pytest.raises(ValueError, match="match feature rows"):
        rollout_feature_fields(prompt, response, hidden, positions[:1])


def test_failed_export_does_not_wait_for_unrelated_pending_chunk(tmp_path):
    (tmp_path / "other.pending").touch()
    (tmp_path / "error").touch()
    wait_for_exports(tmp_path, timeout=0)


def test_packed_export_preserves_chunks_and_releases_budget(tmp_path):
    from verl_speco.producer.feature_export import load_export_chunks

    (tmp_path / "live").touch()
    queue = FeatureExportQueue(pack_chunks=True, max_pending=64)
    for index in range(33):
        assert queue.reserve(128)
        queue.submit_reserved(
            tmp_path, {"index": index, "hidden": torch.tensor([index])}, size=128
        )
    queue.close()
    wait_for_exports(tmp_path)
    assert len(list(tmp_path.glob("*.ptpack"))) == 3
    chunks = list(load_export_chunks(tmp_path))
    assert sorted(chunk["index"] for chunk in chunks) == list(range(33))
    assert all(chunk["hidden"].item() == chunk["index"] for chunk in chunks)
    assert queue._bytes == queue._pending == 0


def test_packed_write_failure_is_not_silently_consumed(tmp_path, monkeypatch):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue(pack_chunks=True)

    def fail(*args):
        raise OSError("full")

    monkeypatch.setattr(queue, "_write_pack", fail)
    assert queue.reserve(16)
    queue.submit_reserved(tmp_path, {}, size=16)
    queue.close()
    assert (tmp_path / "error").exists()
    assert queue._bytes == queue._pending == 0


@pytest.mark.parametrize("packed", [False, True])
def test_acceptance_filtered_empty_chunk_releases_without_serialization(
    tmp_path, packed
):
    from verl_speco.producer.feature_export import load_export_chunks

    (tmp_path / "live").touch()
    queue = FeatureExportQueue(pack_chunks=packed)
    payload = {"positions": torch.tensor([4]), "hidden_states": torch.ones(1, 2)}

    def filter_rejected():
        payload["positions"] = payload["positions"][:0]
        payload["hidden_states"] = payload["hidden_states"][:0]

    assert queue.reserve(32)
    queue.submit_reserved(tmp_path, payload, size=32, wait=filter_rejected)
    queue.close()
    assert list(load_export_chunks(tmp_path)) == []
    assert not list(tmp_path.glob("*.pending"))
    assert not (tmp_path / "error").exists()
    assert queue._bytes == queue._pending == 0


def test_pack_rotation_uses_cached_index_without_directory_scan(tmp_path, monkeypatch):
    from verl_speco.producer.feature_export import load_export_chunks

    (tmp_path / "live").touch()
    queue = FeatureExportQueue(pack_chunks=True, workers=1)
    for index in range(16):
        assert queue.reserve(8)
        queue.submit_reserved(tmp_path, {"index": index}, size=8)
    queue.close()
    original_glob = type(tmp_path).glob

    def no_scan(path, pattern, *args, **kwargs):
        if path == tmp_path and pattern == "*.ptpack":
            raise AssertionError("Cached rotation must not scan directory")
        return original_glob(path, pattern, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), "glob", no_scan)
        queue._write_pack(tmp_path / "rotation.pending", {"index": 16})
    assert sorted(c["index"] for c in load_export_chunks(tmp_path)) == list(range(17))


def test_recycle_runs_after_write_and_not_in_serialized_payload(tmp_path):
    from verl_speco.producer.feature_export import load_export_chunks

    (tmp_path / "live").touch()
    events = []
    queue = FeatureExportQueue()
    assert queue.reserve(8)
    queue.submit_reserved(
        tmp_path,
        {"value": torch.ones(1), "_capture_recycle": lambda: events.append("recycled")},
        size=8,
        wait=lambda: events.append("copied"),
    )
    queue.close()
    assert events == ["copied", "recycled"]
    assert "_capture_recycle" not in list(load_export_chunks(tmp_path))[0]


def test_capture_pool_reuses_only_released_buffers_and_is_bounded():
    from verl_speco.producer.capture_buffer_pool import CaptureBufferPool

    pool = CaptureBufferPool(max_bytes=16)
    first = torch.ones(4)
    pool.release([first, torch.ones(4)])
    assert pool._bytes == 16
    reused = pool.acquire(first)
    assert reused.data_ptr() == first.data_ptr()
    assert pool._bytes == 0 and not pool._free


def test_capture_event_pool_does_not_reuse_until_explicit_release(monkeypatch):
    from verl_speco.producer.capture_buffer_pool import CaptureBufferPool

    created = []

    def event(**kwargs):
        result = object()
        created.append(result)
        return result

    monkeypatch.setattr(torch.cuda, "Event", event)
    pool = CaptureBufferPool()
    first = pool.acquire_event("cuda:0")
    second = pool.acquire_event("cuda:0")
    assert first is not second
    pool.release_events("cuda:0", [first])
    assert pool.acquire_event("cuda:0") is first
    assert pool.acquire_event("cuda:1") is not first


def test_cancelled_window_retains_source_and_budget_until_writer_finishes(
    tmp_path, monkeypatch
):
    from verl_speco.integration import vllm_rollout_capture as capture
    from verl_speco.integration import rollout_reference as reference

    root = tmp_path / "capture"
    root.mkdir()
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda plan: root)
    budget = reference._reserve({})
    plan = {"sample_id": "s", "reference_budget": budget}
    (root / "live").touch()
    (root / "owned-data").write_bytes(b"retained")
    gate = threading.Event()
    queue = FeatureExportQueue()
    try:
        assert queue.reserve(16)
        queue.submit_reserved(
            root, {"_window_plan": plan}, size=16, wait=lambda: gate.wait(5)
        )
        capture.cleanup(plan)
        assert (root / "owned-data").exists()
        assert len(list(tmp_path.glob(".reference-budget/*/*.reservation"))) == 1
    finally:
        gate.set()
        queue.close()
    assert not root.exists()
    assert not list(tmp_path.glob(".reference-budget/*/*.reservation"))


def test_window_batch_shares_pending_until_last_copy(tmp_path, monkeypatch):
    from verl_speco.producer import rollout_window_store

    (tmp_path / "live").touch()
    queue = FeatureExportQueue()
    seen = []
    pending_counts = []

    def record(directory, payload, plan):
        pending_counts.append(len(list(directory.glob("*.pending"))))
        seen.append(payload["capture_timing"]["group_size"])

    monkeypatch.setattr(rollout_window_store, "write_window", record)
    with queue.window_batch() as append:
        for _ in range(3):
            assert queue.reserve(8)
            append(
                tmp_path,
                {"_window_plan": {}, "capture_timing": {}},
                size=8,
                wait=lambda: None,
                owners=(),
            )
        assert len(list(tmp_path.glob("*.pending"))) == 1
    queue.close()
    assert seen == [3, 3, 3] and pending_counts == [1, 1, 1]
    assert not list(tmp_path.glob("*.pending"))
    assert queue._pending == queue._bytes == 0
