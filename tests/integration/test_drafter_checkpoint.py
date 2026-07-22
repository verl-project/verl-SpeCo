from __future__ import annotations

import gc
import json

import pytest

from verl_speco.trainer import checkpoint as checkpoint_utils
from verl_speco.trainer.checkpoint import (
    DrafterCheckpointMetadataError,
    get_drafter_checkpoint_metadata,
    get_drafter_checkpoint_step,
    get_drafter_optimizer_checkpoint_path,
    get_drafter_trainer_state,
    is_pretrained_drafter_checkpoint,
    release_checkpoint_host_memory,
    resolve_drafter_checkpoint_path,
    trim_process_host_memory,
)


def test_drafter_checkpoint_reads_nested_trainer_state(tmp_path) -> None:
    checkpoint = tmp_path / "draft_step_20"
    checkpoint.mkdir()
    metadata = {
        "step": 20,
        "format": "pretrained_drafter_checkpoint",
        "trainer_state": {
            "version": 1,
            "optimizer_steps_total": 80,
            "training_steps": 80,
            "lr_scheduler_last_epoch": 80,
            "current_lr": 1.859423525312737e-6,
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert get_drafter_checkpoint_metadata(checkpoint) == metadata
    assert get_drafter_checkpoint_step(checkpoint) == 20
    assert get_drafter_trainer_state(checkpoint) == metadata["trainer_state"]


def test_drafter_checkpoint_keeps_old_metadata_compatible(tmp_path) -> None:
    checkpoint = tmp_path / "draft_step_12"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 12}), encoding="utf-8")

    assert get_drafter_checkpoint_step(checkpoint) == 12
    assert get_drafter_trainer_state(checkpoint) == {}


def test_resolve_drafter_checkpoint_path_matches_resumed_global_step(tmp_path) -> None:
    original_model = tmp_path / "original"
    original_model.mkdir()
    checkpoint_root = tmp_path / "drafter"
    checkpoint = checkpoint_root / "draft_step_20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 20}), encoding="utf-8")

    assert resolve_drafter_checkpoint_path(original_model, checkpoint_root, 20) == str(checkpoint)


def test_corrupt_metadata_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "draft_step_20"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "metadata.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(DrafterCheckpointMetadataError, match="Invalid drafter checkpoint metadata"):
        get_drafter_checkpoint_step(checkpoint)
    with pytest.raises(DrafterCheckpointMetadataError, match="Invalid drafter checkpoint metadata"):
        is_pretrained_drafter_checkpoint(checkpoint)


def test_managed_checkpoint_without_metadata_is_incomplete(tmp_path) -> None:
    checkpoint = tmp_path / "draft_step_20"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")

    assert get_drafter_checkpoint_step(checkpoint) is None
    assert not is_pretrained_drafter_checkpoint(checkpoint)


def test_optimizer_manifest_requires_complete_dcp_state(tmp_path) -> None:
    checkpoint = tmp_path / "draft_step_20"
    optimizer = checkpoint / "optimizer"
    optimizer.mkdir(parents=True)
    metadata = {
        "step": 20,
        "complete": True,
        "optimizer": {
            "format": "torch_distributed_checkpoint",
            "path": "optimizer",
            "trainer_state_file": "trainer_state.pt",
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DrafterCheckpointMetadataError, match="missing .metadata"):
        get_drafter_optimizer_checkpoint_path(checkpoint)

    (optimizer / ".metadata").write_bytes(b"dcp metadata")
    with pytest.raises(DrafterCheckpointMetadataError, match="missing trainer state"):
        get_drafter_optimizer_checkpoint_path(checkpoint)

    (optimizer / "trainer_state.pt").write_bytes(b"trainer state")
    assert get_drafter_optimizer_checkpoint_path(checkpoint) == str(optimizer)


def test_resolve_rejects_checkpoint_with_mismatched_metadata_step(tmp_path) -> None:
    original_model = tmp_path / "original"
    original_model.mkdir()
    checkpoint_root = tmp_path / "drafter"
    checkpoint = checkpoint_root / "draft_step_20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 19}), encoding="utf-8")

    assert resolve_drafter_checkpoint_path(original_model, checkpoint_root, 20) == str(original_model)


def test_checkpoint_host_memory_reclaim_is_best_effort(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "actor"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"weights")
    events = []
    monkeypatch.setattr(
        "verl_speco.trainer.checkpoint.gc.collect",
        lambda: events.append("gc"),
    )
    monkeypatch.setattr(
        "verl_speco.trainer.checkpoint._trim_process_heap",
        lambda: events.append("trim") or True,
    )
    monkeypatch.setattr(
        "verl_speco.trainer.checkpoint._flush_and_drop_checkpoint_file_cache",
        lambda path: events.append(("drop", path)) or (1, 0),
    )

    result = release_checkpoint_host_memory(checkpoint, drop_file_cache=True)

    assert events == ["gc", "trim", ("drop", str(checkpoint))]
    assert result["heap_trimmed"] is True
    assert result["files_advised"] == 1
    assert result["files_failed"] == 0


def test_process_heap_reclaim_prefers_jemalloc(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(checkpoint_utils.sys, "platform", "linux")
    monkeypatch.setattr(checkpoint_utils, "_jemalloc_is_active", lambda: True)
    monkeypatch.setattr(
        checkpoint_utils,
        "_reclaim_jemalloc_heap",
        lambda: events.append("jemalloc") or True,
    )

    assert checkpoint_utils._trim_process_heap() is True
    assert events == ["jemalloc"]


def test_jemalloc_reclaim_flushes_tcache_and_decays_all_arenas(monkeypatch) -> None:
    controls = []
    monkeypatch.setattr(
        checkpoint_utils,
        "_jemalloc_mallctl",
        lambda name: controls.append(name) or True,
    )
    monkeypatch.delenv("SPECO_JEMALLOC_RECLAIM_MODE", raising=False)

    assert checkpoint_utils._reclaim_jemalloc_heap() is True
    assert controls == ["thread.tcache.flush", "arena.4096.decay"]


def test_trim_process_host_memory_reports_allocator(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_utils, "_host_allocator_name", lambda: "jemalloc")
    monkeypatch.setattr(checkpoint_utils, "_trim_process_heap", lambda: True)
    monkeypatch.setenv("SPECO_JEMALLOC_RECLAIM_MODE", "invalid")

    result = trim_process_host_memory()

    assert result["heap_trimmed"] is True
    assert result["allocator"] == "jemalloc"
    assert result["reclaim_action"] == "decay"


def test_collect_host_allocator_stats_reads_jemalloc_counters(monkeypatch) -> None:
    values = {
        "stats.allocated": 1,
        "stats.active": 2,
        "stats.resident": 3,
        "stats.retained": 4,
    }
    monkeypatch.setattr(checkpoint_utils, "_host_allocator_name", lambda: "jemalloc")
    monkeypatch.setattr(checkpoint_utils, "_jemalloc_refresh_stats", lambda: True)
    monkeypatch.setattr(checkpoint_utils, "_jemalloc_read_size", values.__getitem__)

    assert checkpoint_utils.collect_host_allocator_stats() == {
        "allocator": "jemalloc",
        "allocated": 1,
        "active": 2,
        "resident": 3,
        "retained": 4,
    }


def test_output_lifetime_diagnostics_use_only_weak_references(monkeypatch, capsys) -> None:
    gib = 1024**3
    stats = iter(
        {
            "allocator": "jemalloc",
            "allocated": value * gib,
            "active": value * gib,
            "resident": value * gib,
            "retained": value * gib,
        }
        for value in (1, 2, 3)
    )
    monkeypatch.setattr(checkpoint_utils, "collect_host_allocator_stats", lambda: next(stats))
    monkeypatch.setattr(
        checkpoint_utils,
        "_read_kib",
        lambda path, keys: {"VmRSS": 2 * 1024**2, "RssAnon": 1 * 1024**2},
    )

    class Owner:
        pass

    class Buffer:
        def numel(self):
            return 1024

        def element_size(self):
            return 2

    class Output:
        def __init__(self, buffer):
            self.batch = {"tensor": buffer}

    owner = Owner()
    buffer = Buffer()
    output = Output(buffer)

    first_call = checkpoint_utils.log_previous_output_lifetime(
        owner,
        "worker:method",
        role="worker",
        method="method",
    )
    checkpoint_utils.remember_output_lifetime(owner, "worker:method", first_call, output)
    checkpoint_utils.log_previous_output_lifetime(
        owner,
        "worker:method",
        role="worker",
        method="method",
    )
    live_log = capsys.readouterr().out
    assert "previous_output_alive=1" in live_log
    assert "alive_buffers=1" in live_log
    assert "jemalloc_allocated_delta_gib=+1.000" in live_log

    del output
    del buffer
    gc.collect()
    checkpoint_utils.log_previous_output_lifetime(
        owner,
        "worker:method",
        role="worker",
        method="method",
    )
    released_log = capsys.readouterr().out
    assert "previous_output_alive=0" in released_log
    assert "alive_buffers=0" in released_log
