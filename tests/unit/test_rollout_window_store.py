# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import torch
import pytest
from verl_speco.producer.rollout_window_store import write_window, load_window
from verl_speco.integration import rollout_reference as reference
from verl_speco.integration import vllm_rollout_capture as capture


def test_window_publishes_before_request_completion_without_chunk_archives(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(capture, "plan_directory", lambda p: tmp_path)
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    plan = {
        "sample_id": "s",
        "global_step": 1,
        "replica_rank": 0,
        "start": 1,
        "end": 3,
        "window_store": True,
        "target_version": 0,
        "request_id": "r",
    }
    base = {
        "sample_id": "s",
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    for i in range(2):
        write_window(
            tmp_path,
            {
                **base,
                "positions": torch.tensor([i + 1]),
                "token_ids": torch.tensor([i + 2], dtype=torch.int32),
                "hidden_states": hidden[i : i + 1],
            },
            plan,
        )
        assert (tmp_path / "window.ready").exists() == (i == 1)
    assert not list(tmp_path.glob("*.pt"))
    handle = reference.submit(plan, [1, 2], [3, 4], "r", 0)
    result = reference.resolve(handle)
    assert torch.equal(result["hidden_states"][0], hidden)
    assert not (tmp_path / "assembled.ready").exists()
    assert result["target_version"] == 0
    with (tmp_path / "window.safetensors").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(RuntimeError, match="checksum"):
        load_window(tmp_path)


def test_window_rejects_conflicting_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    plan = {"start": 0, "end": 2, "sample_id": "s"}
    payload = {
        "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
        "positions": torch.tensor([0]),
        "token_ids": torch.tensor([1]),
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    write_window(tmp_path, payload, plan)
    payload["token_ids"][0] = 2
    with pytest.raises(RuntimeError, match="Conflicting"):
        write_window(tmp_path, payload, plan)


def test_window_budget_exhaustion_is_visible_not_partial_success(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setenv("SPECO_REFERENCE_MAX_BYTES", "1")
    monkeypatch.setenv("SPECO_REFERENCE_SAMPLE_BYTES", "1")
    payload = {
        "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
        "positions": torch.tensor([0]),
        "token_ids": torch.tensor([1]),
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    with pytest.raises(RuntimeError, match="exceeds"):
        write_window(tmp_path, payload, {"sample_id": "s", "start": 0, "end": 2})
    assert not (tmp_path / "window.ready").exists()


@pytest.mark.parametrize("corruption", [None, "tokens", "range", "finite"])
def test_consumer_repairs_interrupted_window_publication(
    tmp_path, monkeypatch, corruption
):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda p: tmp_path)
    plan = {
        "sample_id": "s",
        "global_step": 1,
        "replica_rank": 0,
        "window_store": True,
        "start": 0,
        "end": 1,
        "target_version": 0,
        "request_id": "r",
    }
    write_window(
        tmp_path,
        {
            "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
            "positions": torch.tensor([0]),
            "token_ids": torch.tensor([1]),
            "layer_ids": [1],
            "layout": "aux",
            "last_hidden_norm": "post_norm",
        },
        plan,
    )
    handle = {
        "kind": "speco_rollout_reference_v1",
        "plan": plan,
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }
    if corruption == "tokens":
        import json

        (tmp_path / "manifest.ready").write_text(
            json.dumps({"direct_window": True, "prompt": [1], "response": [99]})
        )
    elif corruption == "range":
        handle["plan"]["end"] = 2
        handle["response"] = [2, 3]
    elif corruption == "finite":
        import json
        from safetensors.torch import save_file
        from verl_speco.producer.rollout_window_store import _sha

        path = tmp_path / "window.safetensors"
        save_file(
            {
                "hidden_states": torch.full((1, 2), float("nan"), dtype=torch.bfloat16),
                "token_ids": torch.tensor([1]),
                "coverage": torch.tensor([1], dtype=torch.uint8),
            },
            str(path),
        )
        ready = tmp_path / "window.ready"
        metadata = json.loads(ready.read_text())
        metadata["digests"] = {path.name: _sha(path)}
        ready.write_text(json.dumps(metadata))
    if corruption:
        with pytest.raises(
            RuntimeError,
            match={
                "tokens": "token identity",
                "range": "position range",
                "finite": "non-finite",
            }[corruption],
        ):
            reference.resolve(handle)
        assert (tmp_path / "window.safetensors").exists()
        return
    result = reference.resolve(handle)
    assert result["target_version"] == 0
    assert (tmp_path / "manifest.ready").exists()
    assert not (tmp_path / "assembled.ready").exists()


def test_window_is_readable_by_existing_standalone_file_store(tmp_path, monkeypatch):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    write_window(
        tmp_path,
        {
            "hidden_states": source,
            "positions": torch.tensor([0]),
            "token_ids": torch.tensor([7]),
            "layer_ids": [1],
            "layout": "aux",
            "last_hidden_norm": "post_norm",
        },
        {"sample_id": "s", "start": 0, "end": 1},
    )
    payload, size = FileHiddenStatesStore(wait_for_writer=False).load(
        str(tmp_path / "window.safetensors")
    )
    assert torch.equal(
        payload["hidden_states"].view(torch.int16), source.view(torch.int16)
    )
    assert payload["token_ids"].tolist() == [7] and size > 0


@pytest.mark.parametrize("invalid", ["shape", "dtype", "finite", "range"])
def test_invalid_window_is_rejected_before_reservation(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    payload = {
        "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
        "positions": torch.tensor([0]),
        "token_ids": torch.tensor([1]),
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    if invalid == "shape":
        payload["token_ids"] = torch.tensor([1, 2])
    elif invalid == "dtype":
        payload["positions"] = torch.tensor([0.0])
    elif invalid == "finite":
        payload["hidden_states"][0, 0] = float("inf")
    else:
        payload["positions"][0] = 2
    with pytest.raises(RuntimeError):
        write_window(tmp_path, payload, {"sample_id": "s", "start": 0, "end": 1})
    assert not (tmp_path / "window.meta").exists()
    assert not (tmp_path / "window.safetensors").exists()


def test_window_and_handle_share_frontend_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda plan: tmp_path)
    budget = reference._reserve({})
    plan = {
        "sample_id": "s",
        "global_step": 1,
        "replica_rank": 0,
        "window_store": True,
        "start": 0,
        "end": 1,
        "target_version": 0,
        "request_id": "r",
        "reference_budget": budget,
    }
    handle = reference.submit(plan, [1], [2], "r", 0)
    write_window(
        tmp_path,
        {
            "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
            "positions": torch.tensor([0]),
            "token_ids": torch.tensor([1]),
            "layer_ids": [1],
            "layout": "aux",
            "last_hidden_norm": "post_norm",
        },
        plan,
    )
    assert handle["budget"] == budget
    import json

    assert json.loads((tmp_path / "window.meta").read_text())["budget"] == budget
    assert len(list(tmp_path.glob(".reference-budget/*/*.reservation"))) == 1
    reference._release(handle)


def test_immutable_ready_window_accepts_exact_replay_without_rewriting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    payload = {
        "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
        "positions": torch.tensor([0]),
        "token_ids": torch.tensor([1]),
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    plan = {"sample_id": "s", "start": 0, "end": 1}
    write_window(tmp_path, payload, plan)
    path = tmp_path / "window.safetensors"
    original = path.stat().st_mtime_ns, path.read_bytes()
    write_window(tmp_path, payload, plan)
    assert (path.stat().st_mtime_ns, path.read_bytes()) == original
    payload["hidden_states"][0, 0] = 2
    with pytest.raises(RuntimeError, match="Conflicting window replay"):
        write_window(tmp_path, payload, plan)
    assert (path.stat().st_mtime_ns, path.read_bytes()) == original


def test_reference_rejects_another_run_before_loading_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECO_CAPTURE_RUN_ID", "new-run")
    handle = {
        "kind": "speco_rollout_reference_v1",
        "plan": {
            "sample_id": "s",
            "window_store": True,
            "run_id": "old-run",
            "start": 0,
            "end": 1,
            "request_id": "r",
            "target_version": 0,
        },
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }
    with pytest.raises(ValueError, match="another run"):
        reference.resolve(handle)


def test_sparse_window_flushes_once_and_rejects_conflicting_fragment(
    tmp_path, monkeypatch
):
    import numpy as np

    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    flushes = []
    original = np.memmap.flush

    def flush(mapping):
        flushes.append(mapping.filename)
        return original(mapping)

    monkeypatch.setattr(np.memmap, "flush", flush)
    plan = {"sample_id": "s", "start": 0, "end": 3}
    payload = {
        "positions": torch.tensor([2, 0]),
        "token_ids": torch.tensor([12, 10], dtype=torch.int32),
        "hidden_states": torch.tensor([[2.0], [0.0]], dtype=torch.bfloat16),
        "layer_ids": [1],
        "layout": "aux",
        "last_hidden_norm": "post_norm",
    }
    write_window(tmp_path, payload, plan)
    assert not flushes
    with pytest.raises(RuntimeError, match="Conflicting"):
        write_window(tmp_path, {**payload, "positions": torch.tensor([1, 1])}, plan)
    write_window(
        tmp_path,
        {
            **payload,
            "positions": torch.tensor([1, 1]),
            "token_ids": torch.tensor([11, 11]),
            "hidden_states": torch.ones(2, 1, dtype=torch.bfloat16),
        },
        plan,
    )
    result, _ = load_window(tmp_path)
    assert result["token_ids"].tolist() == [10, 11, 12]
    assert len(flushes) == 1
