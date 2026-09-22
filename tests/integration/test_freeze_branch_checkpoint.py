# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the freeze-transition branch checkpoint feature.

Design: freeze_transition_branch_checkpoint_design.md. The manifest helpers in
``verl_speco.trainer.checkpoint`` are pure stdlib and tested everywhere; the
trainer wiring needs the ray/verl/torch import stack and is skipped when it is
unavailable (same convention as test_drafter_convergence_freeze.py).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verl_speco.trainer.checkpoint import (
    FREEZE_BRANCH_MANIFEST_NAME,
    FREEZE_POLICY_SIDECAR_NAME,
    FreezeBranchManifestError,
    atomic_write_json,
    freeze_branch_manifest_path,
    freeze_policy_sidecar_path,
    read_freeze_branch_manifest,
    validate_freeze_branch_checkpoint,
    write_freeze_branch_manifest,
)
from verl_speco.trainer.drafter_freeze_policy import FreezeDecision, FreezeState

# Heavy import: the trainer needs ray/verl/torch. Guard it so the pure
# checkpoint-helper tests still run in minimal environments.
try:
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    _HAS_TRAINER = True
except Exception:  # pragma: no cover - environment-dependent
    SpecoRayPPOTrainer = None  # type: ignore[assignment]
    _HAS_TRAINER = False

_trainer_skip = pytest.mark.skipif(
    not _HAS_TRAINER, reason="trainer dependency stack (ray/verl/torch) unavailable"
)


def _manifest(step=56, **overrides) -> dict:
    payload = {
        "format_version": 1,
        "complete": True,
        "trigger_step": step,
        "checkpoint_step": step,
        "drafter_version": 8,
        "update_opportunity_id": 8,
        "freeze_reason": "confirmed_non_paying",
        "freeze_mode_at_save": "active",
        "compare_from_step": step + 1,
    }
    payload.update(overrides)
    return payload


def _make_drafter_checkpoint(root, step: int, *, complete=True, recorded_step=None):
    """Create a minimal but *complete* managed drafter checkpoint layout."""
    draft_dir = root / f"draft_step_{step}"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    (draft_dir / "metadata.json").write_text(
        json.dumps(
            {
                "step": recorded_step if recorded_step is not None else step,
                "complete": complete,
            }
        ),
        encoding="utf-8",
    )
    return draft_dir


def _make_actor_checkpoint(folder):
    (folder / "actor").mkdir(parents=True)
    (folder / "actor" / "model.safetensors").write_bytes(b"actor")
    (folder / "data.pt").write_bytes(b"dataloader-state")


# --------------------------------------------------------------------------- #
# Pure manifest helpers (run in every environment)
# --------------------------------------------------------------------------- #
def test_manifest_write_and_read_roundtrip(tmp_path) -> None:
    folder = tmp_path / "global_step_56"
    folder.mkdir()
    path = write_freeze_branch_manifest(folder, _manifest())
    assert path == freeze_branch_manifest_path(folder)

    loaded = read_freeze_branch_manifest(folder)
    assert loaded["complete"] is True
    assert loaded["trigger_step"] == 56
    assert loaded["compare_from_step"] == 57


def test_manifest_writer_rejects_non_complete_payload(tmp_path) -> None:
    folder = tmp_path / "global_step_9"
    folder.mkdir()
    with pytest.raises(FreezeBranchManifestError):
        write_freeze_branch_manifest(folder, _manifest(complete=False))
    assert not (folder / FREEZE_BRANCH_MANIFEST_NAME).exists()


def test_atomic_commit_leaves_no_recognizable_tmp_file(tmp_path) -> None:
    folder = tmp_path / "global_step_3"
    folder.mkdir()
    # A torn temp file (process died mid-write) must never be a fork point.
    (folder / f"{FREEZE_BRANCH_MANIFEST_NAME}.tmp.12345").write_text(
        "{not json", encoding="utf-8"
    )
    with pytest.raises(FreezeBranchManifestError):
        read_freeze_branch_manifest(folder)

    write_freeze_branch_manifest(folder, _manifest(step=3))
    assert read_freeze_branch_manifest(folder)["checkpoint_step"] == 3
    # The committed manifest is recognized even with a stale foreign-PID tmp
    # alongside it (foreign temps are never touched...).
    assert (folder / f"{FREEZE_BRANCH_MANIFEST_NAME}.tmp.12345").exists()
    # (... while the writer's own same-PID staging temp is always cleaned).
    import os

    assert not list(
        folder.glob(f"{FREEZE_BRANCH_MANIFEST_NAME}.tmp.{os.getpid()}")
    )


def test_read_rejects_missing_and_corrupt_manifest(tmp_path) -> None:
    with pytest.raises(FreezeBranchManifestError):
        read_freeze_branch_manifest(tmp_path)
    atomic_write_json(
        freeze_branch_manifest_path(tmp_path), {"complete": False}
    )
    with pytest.raises(FreezeBranchManifestError):
        read_freeze_branch_manifest(tmp_path)


def test_validate_happy_path(tmp_path) -> None:
    folder = tmp_path / "global_step_56"
    folder.mkdir()
    _make_actor_checkpoint(folder)
    drafter_root = tmp_path / "drafter"
    _make_drafter_checkpoint(drafter_root, 56)
    atomic_write_json(
        freeze_policy_sidecar_path(folder),
        {"state": FreezeState.FROZEN.value, "drafter_version": 8},
    )
    write_freeze_branch_manifest(folder, _manifest())

    manifest = validate_freeze_branch_checkpoint(
        folder, drafter_root=drafter_root, step=56
    )
    assert manifest["freeze_mode_at_save"] == "active"


@pytest.mark.parametrize(
    "damage",
    [
        "no_manifest",
        "no_actor",
        "no_data",
        "no_sidecar",
        "no_drafter",
        "drafter_incomplete",
        "drafter_step_mismatch",
        "manifest_step_mismatch",
    ],
)
def test_validate_fails_closed_on_damage(tmp_path, damage) -> None:
    folder = tmp_path / "global_step_56"
    folder.mkdir()
    _make_actor_checkpoint(folder)
    drafter_root = tmp_path / "drafter"
    _make_drafter_checkpoint(drafter_root, 56)
    atomic_write_json(
        freeze_policy_sidecar_path(folder), {"state": "FROZEN"}
    )
    write_freeze_branch_manifest(folder, _manifest())

    if damage == "no_manifest":
        (folder / FREEZE_BRANCH_MANIFEST_NAME).unlink()
    elif damage == "no_actor":
        import shutil

        shutil.rmtree(folder / "actor")
    elif damage == "no_data":
        (folder / "data.pt").unlink()
    elif damage == "no_sidecar":
        (folder / FREEZE_POLICY_SIDECAR_NAME).unlink()
    elif damage == "no_drafter":
        import shutil

        shutil.rmtree(drafter_root)
        drafter_root = tmp_path / "drafter_missing"
    elif damage == "drafter_incomplete":
        (drafter_root / "draft_step_56" / "metadata.json").write_text(
            json.dumps({"step": 56, "complete": False}), encoding="utf-8"
        )
    elif damage == "drafter_step_mismatch":
        (drafter_root / "draft_step_56" / "metadata.json").write_text(
            json.dumps({"step": 55, "complete": True}), encoding="utf-8"
        )
    elif damage == "manifest_step_mismatch":
        write_freeze_branch_manifest(folder, _manifest(trigger_step=55))

    with pytest.raises(FreezeBranchManifestError):
        validate_freeze_branch_checkpoint(
            folder, drafter_root=drafter_root, step=56
        )


# --------------------------------------------------------------------------- #
# Trainer wiring (requires ray/verl/torch)
# --------------------------------------------------------------------------- #
def _freeze_cfg(*, enabled=True, mode="active", method="marginal_utility_v1",
               branch=None):
    return {
        "enabled": enabled,
        "method": method,
        "mode": mode,
        "branch_checkpoint": {"enabled": False, "once": True,
                              "fail_on_error": True}
        if branch is None else branch,
    }


def _training_cfg(freeze_cfg):
    return {
        "save_full_drafter_checkpoint": True,
        "drafter_convergence_freeze": freeze_cfg,
    }


def _make_trainer(tmp_path, training_cfg, *, step=56, active=True,
                  resume_mode="disable", resume_from_path=None):
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = step
    trainer.default_local_dir = str(tmp_path)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            default_local_dir=str(tmp_path),
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=True,
                    enable_drafter_training=True,
                    training=training_cfg,
                )
            ),
        ),
    )
    trainer._speco_freeze_policy_active = active
    trainer._speco_freeze_branch_save_pending = None
    trainer._speco_freeze_branch_saved = False
    trainer._speco_last_checkpoint_saved_step = None
    trainer._speco_freeze_state_sidecar = FREEZE_POLICY_SIDECAR_NAME
    trainer._speco_last_freeze_decision = None
    trainer._speco_last_request_accept_len_records = []
    trainer._speco_drafter_frozen = False
    return trainer


def _decision(state=FreezeState.FROZEN, *, transitioned=True,
              should_train=False, reason="confirmed_non_paying", version=8):
    return FreezeDecision(
        state=state,
        should_train=should_train,
        should_collect=True,
        should_probe=False,
        transitioned=transitioned,
        reason=reason,
        metrics={},
        drafter_version=version,
        update_opportunity_id=version,
        valid_update_count=version,
    )


class _StubPolicy:
    def __init__(self, decision, *, version=8):
        self._decision = decision
        self.drafter_version = version
        self.state = FreezeState.FROZEN
        self.saved_states = []

    def observe(self, evidence):
        return self._decision

    def state_dict(self):
        return {"state": FreezeState.FROZEN.value, "drafter_version": 8}

    def load_state_dict(self, state):
        self.saved_states.append(state)
        if "state" in state:
            self.state = FreezeState(state["state"])
        if "drafter_version" in state:
            self.drafter_version = int(state["drafter_version"])


@_trainer_skip
def test_record_request_on_first_active_freeze_transition(tmp_path, capsys) -> None:
    trainer = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True}))
    )
    trainer._speco_record_freeze_branch_request(_decision())

    request = trainer._speco_freeze_branch_save_pending
    assert request is not None
    assert request["trigger_step"] == 56
    assert request["reason"] == "confirmed_non_paying"
    assert request["drafter_version"] == 8
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_requested"
    assert event["step"] == 56


@_trainer_skip
def test_branch_config_reads_raw_omegaconf_node(tmp_path, capsys) -> None:
    # Regression: the real trainer's self.config holds OmegaConf nodes, and
    # DictConfig is NOT a dict subclass. The branch config reader must
    # normalize it instead of silently returning {} (which skipped the save).
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "save_full_drafter_checkpoint": True,
                            "drafter_convergence_freeze": {
                                "enabled": True,
                                "method": "marginal_utility_v1",
                                "mode": "active",
                                "branch_checkpoint": {
                                    "enabled": True,
                                    "once": True,
                                    "fail_on_error": True,
                                },
                            },
                        },
                    }
                }
            }
        }
    )
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = 56
    trainer.default_local_dir = str(tmp_path)
    trainer.config = cfg
    trainer._speco_freeze_policy_active = True
    trainer._speco_freeze_branch_save_pending = None
    trainer._speco_freeze_branch_saved = False

    branch = trainer._speco_freeze_branch_config()
    assert isinstance(branch, dict)
    assert branch["enabled"] is True

    trainer._speco_record_freeze_branch_request(_decision())
    assert trainer._speco_freeze_branch_save_pending is not None
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_requested"


@_trainer_skip
def test_shadow_mode_never_records(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(mode="shadow", branch={"enabled": True})),
        active=False,
    )
    trainer._speco_record_freeze_branch_request(_decision())
    assert trainer._speco_freeze_branch_save_pending is None


@_trainer_skip
def test_disabled_branch_does_not_record(tmp_path) -> None:
    trainer = _make_trainer(tmp_path, _training_cfg(_freeze_cfg()))
    trainer._speco_record_freeze_branch_request(_decision())
    assert trainer._speco_freeze_branch_save_pending is None


@_trainer_skip
def test_non_transition_frozen_decision_does_not_record(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True}))
    )
    trainer._speco_record_freeze_branch_request(
        _decision(FreezeState.FROZEN, transitioned=False)
    )
    assert trainer._speco_freeze_branch_save_pending is None


@_trainer_skip
def test_non_frozen_transition_does_not_record(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True}))
    )
    trainer._speco_record_freeze_branch_request(
        _decision(FreezeState.RECOVERING, transitioned=True, should_train=True)
    )
    assert trainer._speco_freeze_branch_save_pending is None


@_trainer_skip
def test_second_freeze_cycle_after_save_records_nothing(tmp_path) -> None:
    """FROZEN -> RECOVERING -> FROZEN must not produce a second fork."""
    trainer = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True}))
    )
    trainer._speco_freeze_branch_saved = True
    trainer._speco_record_freeze_branch_request(_decision())
    assert trainer._speco_freeze_branch_save_pending is None


@_trainer_skip
def test_pending_request_is_not_overwritten(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True}))
    )
    trainer._speco_freeze_branch_save_pending = {"trigger_step": 50}
    trainer.global_steps = 56
    trainer._speco_record_freeze_branch_request(_decision())
    assert trainer._speco_freeze_branch_save_pending["trigger_step"] == 50


_BRANCH_ENABLED = {"enabled": True, "once": True, "fail_on_error": True}


@_trainer_skip
@pytest.mark.parametrize(
    "freeze_cfg",
    [
        _freeze_cfg(mode="shadow", branch=dict(_BRANCH_ENABLED)),
        _freeze_cfg(method="legacy", branch=dict(_BRANCH_ENABLED)),
        _freeze_cfg(
            enabled=False, branch=dict(_BRANCH_ENABLED)
        ),
        _freeze_cfg(branch={"enabled": True, "once": False, "fail_on_error": True}),
        _freeze_cfg(branch={"enabled": True, "once": True, "fail_on_error": False}),
    ],
)
def test_branch_config_rejects_unsupported_combinations(tmp_path, freeze_cfg) -> None:
    trainer = _make_trainer(tmp_path, _training_cfg(freeze_cfg))
    with pytest.raises(ValueError):
        trainer._speco_validate_freeze_branch_config()


@_trainer_skip
def test_branch_config_accepts_supported_and_disabled(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(branch={"enabled": True})),
    )
    assert trainer._speco_validate_freeze_branch_config() is None

    trainer_disabled = _make_trainer(
        tmp_path, _training_cfg(_freeze_cfg(mode="shadow"))
    )
    assert trainer_disabled._speco_validate_freeze_branch_config() is None


@_trainer_skip
def test_observe_records_pending_request_end_to_end(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(branch={"enabled": True})),
        step=56,
    )
    trainer._speco_freeze_policy = _StubPolicy(_decision())
    trainer._speco_should_train_drafter_this_step = lambda: False

    decision = trainer._speco_observe_rollout_evidence(
        SimpleNamespace(meta_info=None), 8, generation_seconds=1.0
    )
    assert decision.state == FreezeState.FROZEN
    # Active enforcement of should_train=False and the fork intent both fire.
    assert trainer._speco_drafter_frozen is True
    assert trainer._speco_freeze_branch_save_pending["trigger_step"] == 56


def _install_full_checkpoint_layout(trainer, tmp_path, step=56):
    """Replace the full save chain with filesystem layout creation."""

    def fake_save():
        folder = tmp_path / f"global_step_{step}"
        _make_actor_checkpoint(folder)
        _make_drafter_checkpoint(tmp_path / "drafter", step)
        trainer._speco_last_checkpoint_saved_step = step

    trainer._save_checkpoint = fake_save
    trainer._speco_ensure_drafter_checkpoint_path = lambda: str(
        tmp_path / "drafter"
    )


@_trainer_skip
def test_consume_saves_manifest_sidecar_and_clears_pending(
    tmp_path, capsys
) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(branch={"enabled": True})),
        step=56,
    )
    trainer._speco_freeze_policy = _StubPolicy(_decision())
    trainer._speco_freeze_branch_save_pending = {
        "trigger_step": 56,
        "reason": "confirmed_non_paying",
        "drafter_version": 8,
        "update_opportunity_id": 8,
    }
    _install_full_checkpoint_layout(trainer, tmp_path)

    trainer._speco_maybe_save_freeze_branch_checkpoint()

    folder = tmp_path / "global_step_56"
    manifest = read_freeze_branch_manifest(folder)
    assert manifest["compare_from_step"] == 57
    assert manifest["freeze_reason"] == "confirmed_non_paying"
    sidecar = json.loads(
        (folder / FREEZE_POLICY_SIDECAR_NAME).read_text(encoding="utf-8")
    )
    assert sidecar["drafter_version"] == 8
    assert trainer._speco_freeze_branch_saved is True
    assert trainer._speco_freeze_branch_save_pending is None
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_saved"
    # Validation passed for real (drafter + actor + data + manifest).
    assert event["path"].endswith("global_step_56")


@_trainer_skip
def test_consume_save_chain_with_raw_omegaconf_config(tmp_path, capsys) -> None:
    # End-to-end regression: with a real OmegaConf config (not plain dicts)
    # the full request -> save -> manifest chain must complete.
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "save_full_drafter_checkpoint": True,
                            "drafter_convergence_freeze": {
                                "enabled": True,
                                "method": "marginal_utility_v1",
                                "mode": "active",
                                "branch_checkpoint": {
                                    "enabled": True,
                                    "once": True,
                                    "fail_on_error": True,
                                },
                            },
                        },
                    }
                }
            }
        }
    )
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = 56
    trainer.default_local_dir = str(tmp_path)
    trainer.config = cfg
    trainer._speco_freeze_policy_active = True
    trainer._speco_freeze_branch_save_pending = None
    trainer._speco_freeze_branch_saved = False
    trainer._speco_last_checkpoint_saved_step = None
    trainer._speco_freeze_state_sidecar = FREEZE_POLICY_SIDECAR_NAME
    trainer._speco_freeze_policy = _StubPolicy(_decision())
    trainer._speco_freeze_branch_save_pending = {
        "trigger_step": 56,
        "reason": "confirmed_non_paying",
        "drafter_version": 8,
        "update_opportunity_id": 8,
    }
    _install_full_checkpoint_layout(trainer, tmp_path)

    trainer._speco_maybe_save_freeze_branch_checkpoint()

    folder = tmp_path / "global_step_56"
    assert read_freeze_branch_manifest(folder)["compare_from_step"] == 57
    assert trainer._speco_freeze_branch_saved is True
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_saved"


@_trainer_skip
def test_consume_resolves_local_dir_from_config(tmp_path, capsys) -> None:
    # Regression: base verl RayPPOTrainer does NOT set self.default_local_dir;
    # the saver must resolve it from config.trainer.default_local_dir.
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "save_full_drafter_checkpoint": True,
                            "drafter_convergence_freeze": {
                                "enabled": True,
                                "method": "marginal_utility_v1",
                                "mode": "active",
                                "branch_checkpoint": {
                                    "enabled": True,
                                    "once": True,
                                    "fail_on_error": True,
                                },
                            },
                        },
                    }
                }
            },
        }
    )
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = 56
    trainer.config = cfg
    trainer._speco_freeze_policy_active = True
    trainer._speco_freeze_branch_save_pending = None
    trainer._speco_freeze_branch_saved = False
    trainer._speco_last_checkpoint_saved_step = None
    trainer._speco_freeze_state_sidecar = FREEZE_POLICY_SIDECAR_NAME
    trainer._speco_freeze_policy = _StubPolicy(_decision())
    trainer._speco_freeze_branch_save_pending = {
        "trigger_step": 56,
        "reason": "confirmed_non_paying",
        "drafter_version": 8,
        "update_opportunity_id": 8,
    }
    assert not hasattr(trainer, "default_local_dir")
    assert trainer._speco_default_local_dir() == str(tmp_path)
    _install_full_checkpoint_layout(trainer, tmp_path)

    trainer._speco_maybe_save_freeze_branch_checkpoint()

    folder = tmp_path / "global_step_56"
    assert read_freeze_branch_manifest(folder)["compare_from_step"] == 57
    assert trainer._speco_freeze_branch_saved is True
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_saved"


@_trainer_skip
def test_consume_failure_fails_closed(tmp_path, capsys) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(branch={"enabled": True})),
        step=56,
    )

    def boom():
        raise RuntimeError("disk full")

    trainer._save_checkpoint = boom
    trainer._speco_ensure_drafter_checkpoint_path = lambda: str(
        tmp_path / "drafter"
    )
    trainer._speco_freeze_branch_save_pending = {
        "trigger_step": 56,
        "reason": "confirmed_non_paying",
        "drafter_version": 8,
        "update_opportunity_id": 8,
    }

    with pytest.raises(RuntimeError, match="disk full"):
        trainer._speco_maybe_save_freeze_branch_checkpoint()
    assert trainer._speco_freeze_branch_saved is False
    # Pending retained: the failure must not masquerade as a valid fork point.
    assert (
        trainer._speco_freeze_branch_save_pending["trigger_step"] == 56
    )
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_failed"
    assert event["error_type"] == "RuntimeError"


@_trainer_skip
def test_consume_at_wrong_step_raises(tmp_path) -> None:
    trainer = _make_trainer(
        tmp_path,
        _training_cfg(_freeze_cfg(branch={"enabled": True})),
        step=57,
    )
    trainer._speco_freeze_branch_save_pending = {
        "trigger_step": 56,
        "reason": "r",
        "drafter_version": 8,
        "update_opportunity_id": 8,
    }
    with pytest.raises(RuntimeError, match="consumed at step 57"):
        trainer._speco_maybe_save_freeze_branch_checkpoint()
    assert trainer._speco_freeze_branch_saved is False


@_trainer_skip
def test_event_save_and_periodic_save_coalesce_once_per_step(
    tmp_path, monkeypatch
) -> None:
    """The event save (inside patched update_actor) plus the base loop's
    periodic save at the same step must serialize the full chain once."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = _make_trainer(tmp_path, _training_cfg(_freeze_cfg()), step=7)
    base_calls = []

    def fake_base_save(self):
        base_calls.append(self.global_steps)

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", fake_base_save)
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_freeze_save_state = lambda: None
    trainer._speco_save_drafter_checkpoint = lambda wait=True: None

    trainer._save_checkpoint()
    trainer._save_checkpoint()  # base loop's coinciding periodic call
    assert base_calls == [7]
    assert trainer._speco_last_checkpoint_saved_step == 7


@_trainer_skip
def test_freeze_state_loads_from_resume_fork_folder(tmp_path) -> None:
    """Branch with a fresh default_local_dir restores the sidecar persisted
    inside the shared resume_from_path global_step_S folder."""
    parent_ckpt = tmp_path / "parent" / "global_step_56"
    parent_ckpt.mkdir(parents=True)
    fork_sidecar = parent_ckpt / FREEZE_POLICY_SIDECAR_NAME
    fork_sidecar.write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 9}), encoding="utf-8"
    )
    fork_manifest = parent_ckpt / FREEZE_BRANCH_MANIFEST_NAME
    fork_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "complete": True,
                "trigger_step": 56,
                "checkpoint_step": 56,
                "drafter_version": 9,
                "update_opportunity_id": 9,
                "freeze_reason": "plateau_confirmed",
                "freeze_mode_at_save": "active",
                "compare_from_step": 57,
            }
        ),
        encoding="utf-8",
    )
    branch_dir = tmp_path / "branch_shadow"
    branch_dir.mkdir()

    trainer = _make_trainer(
        branch_dir,
        _training_cfg(_freeze_cfg(mode="shadow")),
        active=False,
        resume_mode="resume_path",
        resume_from_path=str(parent_ckpt),
    )
    policy = _StubPolicy(_decision(), version=0)
    trainer._speco_freeze_load_state(policy)
    assert policy.saved_states == [
        {"state": "FROZEN", "drafter_version": 9}
    ]

    # A sidecar in the branch's own dir takes precedence (normal resume);
    # the fork-manifest cross-check only applies to resume-folder restores.
    root_sidecar = branch_dir / FREEZE_POLICY_SIDECAR_NAME
    root_sidecar.write_text(
        json.dumps({"state": "LEARNING", "drafter_version": 3}), encoding="utf-8"
    )
    policy2 = _StubPolicy(_decision(), version=0)
    trainer._speco_freeze_load_state(policy2)
    assert policy2.saved_states == [
        {"state": "LEARNING", "drafter_version": 3}
    ]


@_trainer_skip
@pytest.mark.parametrize(
    "restored",
    [
        {"state": "CALIBRATING", "drafter_version": 9},  # fingerprint reset
        {"state": "FROZEN", "drafter_version": 0},       # version mismatch
        {"state": "LEARNING", "drafter_version": 8},     # both wrong
    ],
)
def test_freeze_resume_failfast_on_manifest_mismatch(tmp_path, restored) -> None:
    """Regression (Bug 5): after resuming from a committed fork checkpoint the
    policy MUST be FROZEN at the manifest's drafter_version. A silent reset to
    CALIBRATING must abort startup instead of training the 'frozen' branch."""
    parent_ckpt = tmp_path / "parent" / "global_step_56"
    parent_ckpt.mkdir(parents=True)
    (parent_ckpt / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps(restored), encoding="utf-8"
    )
    (parent_ckpt / FREEZE_BRANCH_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format_version": 1,
                "complete": True,
                "trigger_step": 56,
                "checkpoint_step": 56,
                "drafter_version": 9,
                "update_opportunity_id": 9,
                "freeze_reason": "plateau_confirmed",
                "freeze_mode_at_save": "active",
                "compare_from_step": 57,
            }
        ),
        encoding="utf-8",
    )
    branch_dir = tmp_path / "branch_active"
    branch_dir.mkdir()
    trainer = _make_trainer(
        branch_dir,
        _training_cfg(_freeze_cfg(mode="active")),
        active=False,
        resume_mode="resume_path",
        resume_from_path=str(parent_ckpt),
    )
    policy = _StubPolicy(_decision(), version=0)
    with pytest.raises(RuntimeError, match="failed to restore the frozen policy"):
        trainer._speco_freeze_load_state(policy)


@_trainer_skip
def test_freeze_resume_failfast_without_manifest(tmp_path) -> None:
    """A resume folder with a sidecar but no committed manifest is not a fork
    point and must fail fast rather than be treated as one."""
    parent_ckpt = tmp_path / "parent" / "global_step_56"
    parent_ckpt.mkdir(parents=True)
    (parent_ckpt / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 9}), encoding="utf-8"
    )
    trainer = _make_trainer(
        tmp_path / "branch",
        _training_cfg(_freeze_cfg(mode="active")),
        active=False,
        resume_mode="resume_path",
        resume_from_path=str(parent_ckpt),
    )
    (tmp_path / "branch").mkdir()
    policy = _StubPolicy(_decision(), version=0)
    with pytest.raises(RuntimeError, match="not a committed freeze"):
        trainer._speco_freeze_load_state(policy)
