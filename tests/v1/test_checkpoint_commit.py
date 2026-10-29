# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Failure and retention invariants for joint checkpoint commits."""

import json

import pytest
from omegaconf import OmegaConf

from verl_speco.trainer.v1 import checkpoint_commit as commit


@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_native_save_scope_restores_structured_config_after_failure(has_checkpoint):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": False}} if has_checkpoint else {}
            },
            "trainer": {"max_actor_ckpt_to_keep": 1},
        }
    )
    OmegaConf.set_struct(config, True)
    before = OmegaConf.to_container(config)
    with pytest.raises(RuntimeError, match="failed"):
        with commit.defer_native_checkpoint_commit(config):
            assert config.actor_rollout_ref.actor.checkpoint.async_save
            assert config.trainer.max_actor_ckpt_to_keep is None
            assert config.trainer.max_critic_ckpt_to_keep is None
            raise RuntimeError("failed")
    assert OmegaConf.to_container(config) == before
    assert OmegaConf.is_struct(config.trainer)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_async_worker_checkpoint_cannot_publish_a_joint_pointer(role):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": role == "actor"}}
            },
            "critic": {"checkpoint": {"async_save": role == "critic"}},
            "trainer": {},
        }
    )
    before = OmegaConf.to_container(config)
    with pytest.raises(ValueError, match=f"synchronous {role}"):
        with commit.defer_native_checkpoint_commit(config, use_critic=True):
            pytest.fail("must reject before any component is saved")
    assert OmegaConf.to_container(config) == before


def test_failed_atomic_pointer_replace_keeps_previous_pointer(tmp_path, monkeypatch):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("3")
    folder = tmp_path / "global_step_7"
    folder.mkdir()
    (folder / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 7,
                "bubble": {"quota_complete": True},
            }
        )
    )

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(commit.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        commit.commit_latest(tmp_path, 7)
    assert tracker.read_text() == "3"
    assert not list(tmp_path.glob(".speco_latest.*"))


def test_pointer_cannot_commit_incomplete_manifest(tmp_path):
    folder = tmp_path / "global_step_7"
    folder.mkdir()
    (folder / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 7,
                "bubble": {"quota_complete": False},
            }
        )
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        commit.commit_latest(tmp_path, 7)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_retention_only_selects_completed_manifest_steps(tmp_path):
    for step, payload in (
        (3, {"format": "speco_v1_joint_checkpoint", "global_step": 3}),
        (4, {"format": "other", "global_step": 4}),
        (5, {"format": "speco_v1_joint_checkpoint", "global_step": 9}),
        (8, {"format": "speco_v1_joint_checkpoint", "global_step": 8}),
    ):
        folder = tmp_path / f"global_step_{step}"
        folder.mkdir()
        (folder / "speco_v1_manifest.json").write_text(json.dumps(payload))
    assert commit.committed_steps(tmp_path, 7) == [3]


def test_retention_rejects_path_escape_before_deleting_any_checkpoint(tmp_path):
    root = tmp_path / "checkpoints"
    safe = root / "global_step_3" / "actor"
    safe.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "actor"
    victim.mkdir()
    try:
        (root / "global_step_4").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("requires directory symlink support")
    with pytest.raises(ValueError, match="escapes"):
        commit.prune_checkpoint_roles(None, root, [3, 4], "actor")
    assert safe.is_dir()
    assert victim.is_dir()
