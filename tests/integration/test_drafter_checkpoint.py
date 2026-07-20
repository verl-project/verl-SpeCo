from __future__ import annotations

import json

import pytest

from verl_speco.trainer.checkpoint import (
    DrafterCheckpointMetadataError,
    get_drafter_checkpoint_metadata,
    get_drafter_checkpoint_step,
    get_drafter_optimizer_checkpoint_path,
    get_drafter_trainer_state,
    is_pretrained_drafter_checkpoint,
    resolve_drafter_checkpoint_path,
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
