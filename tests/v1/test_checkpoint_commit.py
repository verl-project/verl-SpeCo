# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Failure and retention invariants for joint checkpoint commits."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.trainer.v1 import checkpoint_commit as commit
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


@pytest.mark.parametrize("step", [80, 100])
def test_rewind_cannot_write_components_in_published_checkpoint_root(tmp_path, step):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("100", encoding="utf-8")
    published = tmp_path / "global_step_100"
    published.mkdir()
    marker = published / "speco_v1_manifest.json"
    marker.write_text("previous manifest", encoding="utf-8")

    class Harness(SpecoV1Mixin):
        config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=tmp_path))
        global_steps = step
        _speco_colocate_admission_closed = True
        _speco_last_checkpoint_saved_step = None

        def _speco_rollout_idle_worker_enabled(self):
            return True

        def _speco_bubble_session(self):
            return SimpleNamespace(close=lambda boundary: None, pending=False)

        def _speco_checkpoint_pending_quota(self):
            return False

        def _speco_save_v1_checkpoint_components(self):
            pytest.fail("rewind must be rejected before any component writes")

    with pytest.raises(RuntimeError, match="latest already points to step 100"):
        Harness()._save_checkpoint()

    assert tracker.read_text(encoding="utf-8") == "100"
    assert marker.read_text(encoding="utf-8") == "previous manifest"
    assert not (tmp_path / "global_step_80").exists()


@pytest.mark.parametrize("previous", [None, b"3\n"])
@pytest.mark.parametrize("native_fails", [False, True])
def test_native_save_cannot_leave_an_early_latest_pointer(
    tmp_path, previous, native_fails
):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    if previous is not None:
        tracker.write_bytes(previous)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
            "trainer": {},
        }
    )
    with pytest.raises(RuntimeError, match="changed latest before the joint commit"):
        with commit.defer_native_checkpoint_commit(config, root=tmp_path):
            tracker.write_bytes(b"7")
            if native_fails:
                raise OSError("native save failed")
    assert (tracker.read_bytes() if tracker.exists() else None) == previous
    assert config.actor_rollout_ref.actor.checkpoint.async_save is False


def test_checkpoint_step_allows_fresh_root_and_forward_progress(tmp_path):
    commit.validate_checkpoint_step(tmp_path, 1)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("100\n", encoding="utf-8")
    commit.validate_checkpoint_step(tmp_path, 101)


@pytest.mark.parametrize("tracker_step", [None, 60])
def test_checkpoint_rewind_uses_complete_manifests_with_missing_or_stale_tracker(
    tmp_path, tracker_step
):
    if tracker_step is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(
            str(tracker_step), encoding="utf-8"
        )
    for step, complete in ((100, True), (200, False)):
        folder = tmp_path / f"global_step_{step}"
        folder.mkdir()
        (folder / "speco_v1_manifest.json").write_text(
            json.dumps(
                {
                    "format": "speco_v1_joint_checkpoint",
                    "version": 1,
                    "global_step": step,
                    "bubble": {"quota_complete": complete},
                    "drafter_checkpoints": [
                        {"saved": True, "path": f"draft_step_{step}"}
                    ],
                }
            ),
            encoding="utf-8",
        )
    for step in (80, 100):
        with pytest.raises(
            RuntimeError, match="complete joint manifest exists at step 100"
        ):
            commit.validate_checkpoint_step(tmp_path, step)
    commit.validate_checkpoint_step(tmp_path, 101)


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


@pytest.mark.parametrize("has_previous", [False, True])
def test_failed_atomic_pointer_replace_keeps_previous_pointer(
    tmp_path, monkeypatch, has_previous
):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    if has_previous:
        tracker.write_text("3")
    folder = tmp_path / "global_step_7"
    commit.begin_checkpoint_commit(tmp_path, 7)
    (folder / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 7,
                "bubble": {"quota_complete": True},
                "drafter_checkpoints": [{"saved": True, "path": "draft_step_7"}],
            }
        )
    )

    def fail(*args):
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(commit.os, "replace", fail)
        with pytest.raises(OSError, match="replace failed"):
            commit.commit_latest(tmp_path, 7)
    if has_previous:
        assert tracker.read_text() == "3"
    else:
        assert not tracker.exists()
    assert not list(tmp_path.glob(".speco_latest.*"))
    assert commit.is_unpublished_checkpoint(folder)
    assert commit.committed_steps(tmp_path, 7) == []
    commit.validate_checkpoint_step(tmp_path, 7)
    commit.begin_checkpoint_commit(tmp_path, 7)
    commit.commit_latest(tmp_path, 7)
    assert tracker.read_text() == "7"
    assert not commit.is_unpublished_checkpoint(folder)


@pytest.mark.parametrize("quota_complete", [False, True])
def test_pointer_cannot_commit_incomplete_manifest(tmp_path, quota_complete):
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
                "bubble": {"quota_complete": quota_complete},
            }
        )
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        commit.commit_latest(tmp_path, 7)
    assert tracker.read_text() == "3"


def test_retention_only_selects_completed_manifest_steps(tmp_path):
    for step, overrides in (
        (3, {}),
        (4, {"format": "other"}),
        (5, {"global_step": 9}),
        (6, {"drafter_checkpoints": []}),
        (7, {"bubble": {"quota_complete": False}}),
        (8, {}),
        (9, {"bubble": None}),
        (10, {}),
    ):
        payload = {
            "format": "speco_v1_joint_checkpoint",
            "version": 1,
            "global_step": step,
            "bubble": {"quota_complete": True},
            "drafter_checkpoints": [{"saved": True, "path": f"draft_step_{step}"}],
            **overrides,
        }
        folder = tmp_path / f"global_step_{step}"
        folder.mkdir()
        (folder / "speco_v1_manifest.json").write_text(json.dumps(payload))
    assert commit.committed_steps(tmp_path, 9) == [3, 8]


@pytest.mark.parametrize("same_root_alias", [False, True])
def test_retention_rejects_path_escape_before_deleting_any_checkpoint(
    tmp_path, same_root_alias
):
    root = tmp_path / "checkpoints"
    safe = root / "global_step_3" / "actor"
    safe.mkdir(parents=True)
    outside = root / "global_step_8" if same_root_alias else tmp_path / "outside"
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


def test_worker_retention_combines_private_manifests_and_driver_steps(tmp_path):
    for step in (3, 7, 9):
        folder = tmp_path / f"global_step_{step}"
        (folder / "actor").mkdir(parents=True)
        if step == 3:
            (folder / "speco_v1_manifest.json").write_text(
                json.dumps(
                    {
                        "format": "speco_v1_joint_checkpoint",
                        "version": 1,
                        "global_step": step,
                        "bubble": {"quota_complete": True},
                        "drafter_checkpoints": [
                            {"saved": True, "path": "draft_step_3"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
    assert commit.prune_checkpoint_roles(None, tmp_path, [7, 9], "actor", 2, 9) == {
        "pruned": 1,
        "failed": 0,
    }
    assert not (tmp_path / "global_step_3" / "actor").exists()
    assert (tmp_path / "global_step_7" / "actor").is_dir()
    assert (tmp_path / "global_step_9" / "actor").is_dir()


def test_worker_retention_does_not_count_unpublished_manifest(tmp_path):
    for step in (10, 20, 30):
        commit.begin_checkpoint_commit(tmp_path, step)
        folder = tmp_path / f"global_step_{step}"
        (folder / "actor").mkdir()
        (folder / "speco_v1_manifest.json").write_text(
            json.dumps(
                {
                    "format": "speco_v1_joint_checkpoint",
                    "version": 1,
                    "global_step": step,
                    "bubble": {"quota_complete": True},
                    "drafter_checkpoints": [
                        {"saved": True, "path": f"draft_step_{step}"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        if step != 20:
            commit.commit_latest(tmp_path, step)
    assert commit.prune_checkpoint_roles(None, tmp_path, [30], "actor", 2, 30) == {
        "pruned": 0,
        "failed": 0,
    }
    assert (tmp_path / "global_step_10" / "actor").is_dir()
    assert commit.committed_steps(tmp_path, 30) == [10, 30]


def test_retention_skips_path_already_claimed_by_another_worker(tmp_path):
    folder = tmp_path / "global_step_3"
    target = folder / "actor"
    target.mkdir(parents=True)
    # A stale claim from the directory-based implementation must not block pruning.
    (folder / ".actor.speco-prune").mkdir()
    with commit._checkpoint_prune_lock(target) as acquired:
        assert acquired
        assert commit.prune_checkpoint_roles(None, tmp_path, [3], "actor") == {
            "pruned": 0,
            "failed": 0,
        }
        assert target.is_dir()
    assert commit.prune_checkpoint_roles(None, tmp_path, [3], "actor") == {
        "pruned": 1,
        "failed": 0,
    }
    assert not target.exists()


def test_retention_reports_failed_deletions(tmp_path, monkeypatch, caplog):
    target = tmp_path / "global_step_3" / "actor"
    target.mkdir(parents=True)

    def fail(path):
        raise PermissionError("checkpoint is busy")

    monkeypatch.setattr(commit.shutil, "rmtree", fail)
    assert commit.prune_checkpoint_roles(None, tmp_path, [3], "actor") == {
        "pruned": 0,
        "failed": 1,
    }
    assert target.is_dir()
    assert "Checkpoint retention failed" in caplog.text
