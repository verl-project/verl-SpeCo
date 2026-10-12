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
"""Tests for the freeze-transition branch checkpoint (design doc of the same
name). Manifest helpers are pure stdlib; trainer wiring needs ray/verl/torch
and is skipped when that stack is unavailable.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from verl_speco.trainer.checkpoint import (
    FREEZE_BRANCH_MANIFEST_NAME, FREEZE_POLICY_SIDECAR_NAME,
    FreezeBranchManifestError, atomic_write_json, freeze_branch_manifest_path,
    freeze_policy_sidecar_path, read_freeze_branch_manifest,
    validate_freeze_branch_checkpoint, write_freeze_branch_manifest,
)
from verl_speco.trainer.drafter_freeze_policy import FreezeDecision, FreezeState

try:  # heavy import: trainer needs ray/verl/torch
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    _HAS_TRAINER = True
except Exception:  # pragma: no cover - environment-dependent
    SpecoRayPPOTrainer = None  # type: ignore[assignment]
    _HAS_TRAINER = False

_trainer_skip = pytest.mark.skipif(
    not _HAS_TRAINER, reason="trainer dependency stack (ray/verl/torch) unavailable")

def _manifest(step=56, **ov) -> dict:
    payload = {"format_version": 1, "complete": True, "trigger_step": step,
               "checkpoint_step": step, "drafter_version": 8, "update_opportunity_id": 8,
               "freeze_reason": "confirmed_non_paying", "freeze_mode_at_save": "active",
               "compare_from_step": step + 1}
    payload.update(ov)
    return payload

def _make_layout(root, *, step=56, damage=None):
    """Complete global_step_S + managed drafter layout with optional damage."""
    folder = root / f"global_step_{step}"
    (folder / "actor").mkdir(parents=True)
    (folder / "actor" / "model.safetensors").write_bytes(b"actor")
    (folder / "data.pt").write_bytes(b"dataloader-state")
    drafter_root = root / "drafter"
    draft_dir = drafter_root / f"draft_step_{step}"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    metadata = {"step": step, "complete": True}
    if damage == "no_actor":
        import shutil; shutil.rmtree(folder / "actor")
    elif damage == "no_data":
        (folder / "data.pt").unlink()
    elif damage == "no_drafter":
        import shutil; shutil.rmtree(drafter_root); drafter_root = root / "drafter_missing"
    elif damage == "drafter_incomplete":
        metadata["complete"] = False
    elif damage == "drafter_step_mismatch":
        metadata["step"] = step - 1
    if damage != "no_drafter":  # tree removed: nothing to mark; validator fails on absence
        (draft_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder, drafter_root


def test_manifest_roundtrip_and_write_read_rejection(tmp_path) -> None:
    folder = tmp_path / "global_step_56"
    folder.mkdir()
    write_freeze_branch_manifest(folder, _manifest())
    loaded = read_freeze_branch_manifest(folder)
    assert (loaded["trigger_step"], loaded["compare_from_step"], loaded["complete"]) == (56, 57, True)

    with pytest.raises(FreezeBranchManifestError):  # incomplete never written
        write_freeze_branch_manifest(folder, _manifest(complete=False))
    with pytest.raises(FreezeBranchManifestError):  # missing
        read_freeze_branch_manifest(tmp_path / "nope")
    atomic_write_json(freeze_branch_manifest_path(tmp_path), {"complete": False})
    with pytest.raises(FreezeBranchManifestError):  # corrupt payload
        read_freeze_branch_manifest(tmp_path)

def test_atomic_commit_and_foreign_tmp_survival(tmp_path) -> None:
    folder = tmp_path / "global_step_3"
    folder.mkdir()
    foreign = folder / f"{FREEZE_BRANCH_MANIFEST_NAME}.tmp.12345"
    foreign.write_text("{not json", encoding="utf-8")  # torn foreign-PID temp
    with pytest.raises(FreezeBranchManifestError):
        read_freeze_branch_manifest(folder)

    write_freeze_branch_manifest(folder, _manifest(step=3))
    assert read_freeze_branch_manifest(folder)["checkpoint_step"] == 3
    assert foreign.exists()  # foreign temp untouched
    assert not list(folder.glob(f"{FREEZE_BRANCH_MANIFEST_NAME}.tmp.{os.getpid()}"))

def test_validate_happy_path(tmp_path) -> None:
    folder, drafter_root = _make_layout(tmp_path)
    atomic_write_json(freeze_policy_sidecar_path(folder),
                      {"state": FreezeState.FROZEN.value, "drafter_version": 8})
    write_freeze_branch_manifest(folder, _manifest())
    assert validate_freeze_branch_checkpoint(folder, drafter_root=drafter_root, step=56)[
        "freeze_mode_at_save"] == "active"


@pytest.mark.parametrize(
    "damage",
    ["no_manifest", "no_actor", "no_data", "no_sidecar", "no_drafter",
     "drafter_incomplete", "drafter_step_mismatch", "manifest_step_mismatch",
     "null_trigger_step", "null_checkpoint_step"])
def test_validate_fails_closed_on_damage(tmp_path, damage) -> None:
    folder, drafter_root = _make_layout(tmp_path, damage=damage)
    if damage != "no_sidecar":
        atomic_write_json(freeze_policy_sidecar_path(folder), {"state": "FROZEN"})
    if damage == "manifest_step_mismatch":
        write_freeze_branch_manifest(folder, _manifest(trigger_step=55))
    elif damage.startswith("null_"):
        atomic_write_json(freeze_branch_manifest_path(folder),
                          _manifest(**{damage.removeprefix("null_"): None}))
    elif damage != "no_manifest":
        write_freeze_branch_manifest(folder, _manifest())
    with pytest.raises(FreezeBranchManifestError):
        validate_freeze_branch_checkpoint(folder, drafter_root=drafter_root, step=56)


def _freeze_cfg(*, enabled=True, mode="active", method="marginal_utility_v1", branch=None):
    return {"enabled": enabled, "method": method, "mode": mode,
            "branch_checkpoint": (
                {"enabled": False, "once": True, "fail_on_error": True}
                if branch is None else branch)}

def _training_cfg(freeze_cfg):
    return {"save_full_drafter_checkpoint": True,
            "drafter_convergence_freeze": freeze_cfg}

def _bare_trainer(tmp_path, *, step=56, active=True):
    """Trainer via __new__ with only the speco runtime attrs set."""
    t = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    t.global_steps = step
    t._speco_freeze_policy_active = active
    t._speco_freeze_branch_save_pending = None
    t._speco_freeze_branch_saved = False
    t._speco_last_checkpoint_saved_step = None
    t._speco_freeze_state_sidecar = FREEZE_POLICY_SIDECAR_NAME
    t._speco_last_freeze_decision = None
    t._speco_last_request_accept_len_records = []
    t._speco_drafter_frozen = False
    return t

def _attach_plain_config(t, root, training):
    t.default_local_dir = str(root)
    t.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(root)),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(drafter=SimpleNamespace(
                enable=True, enable_drafter_training=True, training=training))))

def _decision(state=FreezeState.FROZEN, *, transitioned=True, should_train=False,
              reason="confirmed_non_paying", version=8):
    return FreezeDecision(
        state=state, should_train=should_train, should_probe=False,
        transitioned=transitioned, reason=reason, metrics={},
        drafter_version=version, update_opportunity_id=version, valid_update_count=version)

def _pending(step=56):
    return {"trigger_step": step, "reason": "confirmed_non_paying",
            "drafter_version": 8, "update_opportunity_id": 8}

class _StubPolicy:
    def __init__(self, decision, *, version=8):
        self._decision, self.drafter_version, self.state = decision, version, FreezeState.FROZEN
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
@pytest.mark.parametrize(
    ("scenario", "expected_step"),
    [("active_first_transition", 56), ("shadow", None), ("branch_disabled", None),
     ("frozen_non_transition", None), ("non_frozen_transition", None),
     ("already_saved", None), ("pending_kept", 50)])
def test_record_request_gating(tmp_path, scenario, expected_step) -> None:
    cfgs = {
        "active_first_transition": _freeze_cfg(branch={"enabled": True}),
        "shadow": _freeze_cfg(mode="shadow", branch={"enabled": True}),
        "branch_disabled": _freeze_cfg(),
        "frozen_non_transition": _freeze_cfg(branch={"enabled": True}),
        "non_frozen_transition": _freeze_cfg(branch={"enabled": True}),
        "already_saved": _freeze_cfg(branch={"enabled": True}),
        "pending_kept": _freeze_cfg(branch={"enabled": True})}
    t = _bare_trainer(tmp_path, active=scenario != "shadow")
    _attach_plain_config(t, tmp_path, _training_cfg(cfgs[scenario]))
    d = _decision()
    if scenario == "frozen_non_transition":
        d = _decision(transitioned=False)
    elif scenario == "non_frozen_transition":
        d = _decision(FreezeState.RECOVERING, should_train=True)
    elif scenario == "already_saved":
        t._speco_freeze_branch_saved = True
    elif scenario == "pending_kept":
        t._speco_freeze_branch_save_pending = {"trigger_step": 50}

    t._speco_record_freeze_branch_request(d)
    pending = t._speco_freeze_branch_save_pending
    assert (pending["trigger_step"] if pending else None) == expected_step


@_trainer_skip
@pytest.mark.parametrize(
    "freeze_cfg",
    [_freeze_cfg(mode="shadow", branch={"enabled": True}),
     _freeze_cfg(method="legacy", branch={"enabled": True}),
     _freeze_cfg(enabled=False, branch={"enabled": True}),
     _freeze_cfg(branch={"enabled": True, "once": False}),
     _freeze_cfg(branch={"enabled": True, "fail_on_error": False})])
def test_branch_config_rejects_unsupported(tmp_path, freeze_cfg) -> None:
    t = _bare_trainer(tmp_path)
    _attach_plain_config(t, tmp_path, _training_cfg(freeze_cfg))
    with pytest.raises(ValueError):
        t._speco_validate_freeze_branch_config()


@_trainer_skip
def test_branch_config_accepts_supported_and_disabled(tmp_path) -> None:
    t = _bare_trainer(tmp_path)
    _attach_plain_config(t, tmp_path,
                         _training_cfg(_freeze_cfg(branch={"enabled": True})))
    assert t._speco_validate_freeze_branch_config() is None
    t2 = _bare_trainer(tmp_path)
    _attach_plain_config(t2, tmp_path, _training_cfg(_freeze_cfg(mode="shadow")))
    assert t2._speco_validate_freeze_branch_config() is None


@_trainer_skip
def test_observe_e2e_enforces_gate_and_records_pending(tmp_path) -> None:
    t = _bare_trainer(tmp_path)
    _attach_plain_config(t, tmp_path,
                         _training_cfg(_freeze_cfg(branch={"enabled": True})))
    t._speco_freeze_policy = _StubPolicy(_decision())
    t._speco_should_train_drafter_this_step = lambda: False

    d = t._speco_observe_rollout_evidence(
        SimpleNamespace(meta_info=None), 8, generation_seconds=1.0)
    assert (d.state, t._speco_drafter_frozen) == (FreezeState.FROZEN, True)
    assert t._speco_freeze_branch_save_pending["trigger_step"] == 56

def _install_full_save(trainer, tmp_path, step=56):
    """Replace the full save chain with filesystem layout creation."""
    def fake_save():
        _make_layout(tmp_path, step=step)
        trainer._speco_last_checkpoint_saved_step = step

    trainer._save_checkpoint = fake_save
    trainer._speco_ensure_drafter_checkpoint_path = lambda: str(tmp_path / "drafter")


@_trainer_skip
@pytest.mark.parametrize("variant", ["plain", "resolve_dir"])
def test_consume_save_chain_config_variants(tmp_path, capsys, variant) -> None:
    """plain: namespace config + attr. resolve_dir: real OmegaConf nodes
    WITHOUT the attr -- covers OmegaConf normalization + local-dir resolution."""
    t = _bare_trainer(tmp_path)
    t._speco_freeze_branch_save_pending = _pending()
    t._speco_freeze_policy = _StubPolicy(_decision())
    if variant == "plain":
        _attach_plain_config(
            t, tmp_path, _training_cfg(_freeze_cfg(branch={"enabled": True})))
    else:
        pytest.importorskip("omegaconf")
        from omegaconf import OmegaConf

        t.config = OmegaConf.create({
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {"rollout": {"drafter": {
                "enable": True, "enable_drafter_training": True,
                "training": _training_cfg(_freeze_cfg(branch={"enabled": True}))}}}})
    _install_full_save(t, tmp_path)

    t._speco_maybe_save_freeze_branch_checkpoint()

    folder = tmp_path / "global_step_56"
    manifest = read_freeze_branch_manifest(folder)
    assert (manifest["compare_from_step"], manifest["freeze_reason"]) == (57, "confirmed_non_paying")
    sidecar = json.loads((folder / FREEZE_POLICY_SIDECAR_NAME).read_text(encoding="utf-8"))
    assert sidecar["drafter_version"] == 8
    assert (t._speco_freeze_branch_saved, t._speco_freeze_branch_save_pending) == (True, None)
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_saved"
    assert event["path"].endswith("global_step_56")


@_trainer_skip
@pytest.mark.parametrize(("failure", "match"),
                         [("save_raises", "disk full"), ("wrong_step", "consumed at step 57")])
def test_consume_failure_modes_fail_closed(tmp_path, capsys, failure, match) -> None:
    t = _bare_trainer(tmp_path, step=57 if failure == "wrong_step" else 56)
    _attach_plain_config(t, tmp_path,
                         _training_cfg(_freeze_cfg(branch={"enabled": True})))
    t._speco_freeze_branch_save_pending = _pending()
    if failure == "save_raises":
        t._save_checkpoint = lambda: (_ for _ in ()).throw(RuntimeError("disk full"))
        t._speco_ensure_drafter_checkpoint_path = lambda: str(tmp_path / "drafter")

    with pytest.raises(RuntimeError, match=match):
        t._speco_maybe_save_freeze_branch_checkpoint()
    assert t._speco_freeze_branch_saved is False  # no false fork point
    assert t._speco_freeze_branch_save_pending["trigger_step"] == 56
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event"] == "freeze_branch_checkpoint_failed"


@_trainer_skip
def test_event_and_periodic_save_coalesce_once_per_step(tmp_path, monkeypatch) -> None:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    t = _bare_trainer(tmp_path, step=7)
    _attach_plain_config(t, tmp_path, _training_cfg(_freeze_cfg()))
    calls = []
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint",
                        lambda self: calls.append(self.global_steps))
    t._speco_wait_pending_drafter_publish = lambda: None
    t._speco_freeze_save_state = lambda: None
    t._speco_save_drafter_checkpoint = lambda wait=True: None

    t._save_checkpoint()
    t._save_checkpoint()  # base loop's coinciding periodic call
    assert calls == [7]
    assert t._speco_last_checkpoint_saved_step == 7

def _fork_folder(tmp_path, *, sidecar, manifest=True):
    """Resumed parent global_step_56 with sidecar (+ optional manifest)."""
    parent = tmp_path / "parent" / "global_step_56"
    parent.mkdir(parents=True)
    (parent / FREEZE_POLICY_SIDECAR_NAME).write_text(json.dumps(sidecar), encoding="utf-8")
    if manifest:
        (parent / FREEZE_BRANCH_MANIFEST_NAME).write_text(
            json.dumps(_manifest(drafter_version=9, freeze_reason="plateau_confirmed")),
            encoding="utf-8")
    return parent

def _resume_trainer(tmp_path, parent=None, *, resume_mode="resume_path"):
    branch = tmp_path / "branch"
    branch.mkdir()
    t = _bare_trainer(branch, active=False)
    t.default_local_dir = str(branch)
    trainer_ns = {"default_local_dir": str(branch), "resume_mode": resume_mode}
    if resume_mode == "resume_path":
        trainer_ns["resume_from_path"] = str(parent) if parent is not None else None
    t.config = SimpleNamespace(
        trainer=SimpleNamespace(**trainer_ns),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace())))
    return t


@_trainer_skip
def test_freeze_state_loads_from_resume_fork_folder(tmp_path) -> None:
    """resume_path restores ONLY the sidecar carried inside the shared
    resume_from_path global_step_S; the fresh branch output dir contributes
    nothing."""
    parent = _fork_folder(tmp_path, sidecar={"state": "FROZEN", "drafter_version": 9})
    t = _resume_trainer(tmp_path, parent)
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == [{"state": "FROZEN", "drafter_version": 9}]


@_trainer_skip
def test_freeze_state_resume_path_ignores_stale_output_sidecar(tmp_path) -> None:
    """A stale/divergent sidecar in the branch output dir must never override
    the state carried by the explicitly chosen resume_from_path checkpoint."""
    parent = _fork_folder(tmp_path, sidecar={"state": "FROZEN", "drafter_version": 9})
    t = _resume_trainer(tmp_path, parent)
    (tmp_path / "branch" / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "LEARNING", "drafter_version": 3}), encoding="utf-8")
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == [{"state": "FROZEN", "drafter_version": 9}]


@_trainer_skip
def test_freeze_state_disable_mode_ignores_stale_sidecar(tmp_path) -> None:
    """Fresh training (resume_mode=disable) must not inherit a stale FROZEN
    sidecar left behind in the output directory."""
    t = _resume_trainer(tmp_path, resume_mode="disable")
    (tmp_path / "branch" / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 7}), encoding="utf-8")
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == []  # nothing restored
    assert policy.drafter_version == 0


@_trainer_skip
def test_freeze_state_auto_mode_resolves_tracked_checkpoint(tmp_path) -> None:
    """auto resume restores the sidecar committed INSIDE the checkpoint verl
    actually restores (the tracker-selected latest global_step_S); a stale
    root sidecar from an interrupted later save attempt is never consulted."""
    t = _resume_trainer(tmp_path, resume_mode="auto")
    branch = tmp_path / "branch"
    folder = branch / "global_step_5"
    folder.mkdir()
    (folder / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 5}), encoding="utf-8")
    (branch / "latest_checkpointed_iteration.txt").write_text("5", encoding="utf-8")
    # Stale newer state left at the root by an interrupted save of step 9:
    (branch / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 9}), encoding="utf-8")
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == [{"state": "FROZEN", "drafter_version": 5}]


@_trainer_skip
def test_freeze_state_auto_mode_without_checkpoint_loads_nothing(tmp_path) -> None:
    """No tracker / no checkpoint folder -> the base trainer starts from
    scratch, so the freeze policy must start fresh too; a stale root sidecar
    is NOT a valid state source (review repro: AUTO_WITH_NO_CHECKPOINT)."""
    t = _resume_trainer(tmp_path, resume_mode="auto")
    (tmp_path / "branch" / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "FROZEN", "drafter_version": 7}), encoding="utf-8")
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == []  # nothing restored
    assert policy.drafter_version == 0


@_trainer_skip
def test_freeze_state_resume_path_normal_checkpoint_restores(tmp_path) -> None:
    """An explicit resume from a NORMAL checkpoint (sidecar but no fork
    manifest) restores the committed policy state without any fork-specific
    FROZEN validation -- non-frozen states are perfectly valid here."""
    folder = tmp_path / "parent" / "global_step_40"
    folder.mkdir(parents=True)
    (folder / FREEZE_POLICY_SIDECAR_NAME).write_text(
        json.dumps({"state": "LEARNING", "drafter_version": 11}), encoding="utf-8")
    t = _resume_trainer(tmp_path, folder)
    policy = _StubPolicy(_decision(), version=0)
    t._speco_freeze_load_state(policy)
    assert policy.saved_states == [{"state": "LEARNING", "drafter_version": 11}]


@_trainer_skip
def test_freeze_state_checkpoint_without_sidecar_downgrades(tmp_path, caplog) -> None:
    """Historical checkpoints predate per-checkpoint freeze state: no sidecar
    -> explicit compatibility downgrade (fresh policy + loud warning), never
    a silent rebind of stale state."""
    import logging

    folder = tmp_path / "parent" / "global_step_40"
    folder.mkdir(parents=True)
    t = _resume_trainer(tmp_path, folder)
    policy = _StubPolicy(_decision(), version=0)
    with caplog.at_level(logging.WARNING):
        t._speco_freeze_load_state(policy)  # must not raise
    assert policy.saved_states == []
    assert "no policy state sidecar" in caplog.text


@_trainer_skip
def test_freeze_state_fork_checkpoint_missing_sidecar_fails(tmp_path) -> None:
    """A committed fork point (manifest present) without its policy sidecar
    is incomplete: fail closed instead of silently resuming with a fresh
    policy and training the supposedly-frozen branch."""
    parent = tmp_path / "parent" / "global_step_56"
    parent.mkdir(parents=True)
    (parent / FREEZE_BRANCH_MANIFEST_NAME).write_text(
        json.dumps(_manifest(drafter_version=9, freeze_reason="plateau_confirmed")),
        encoding="utf-8")
    t = _resume_trainer(tmp_path, parent)
    with pytest.raises(RuntimeError, match="missing its policy state sidecar"):
        t._speco_freeze_load_state(_StubPolicy(_decision(), version=0))


@_trainer_skip
def test_periodic_checkpoint_carries_policy_sidecar(tmp_path, monkeypatch) -> None:
    """Every ordinary checkpoint carries the policy sidecar committed into the
    same global_step_S folder; no root sidecar is written anymore."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    t = _bare_trainer(tmp_path, step=7)
    _attach_plain_config(t, tmp_path, _training_cfg(_freeze_cfg()))
    t._speco_freeze_policy = _StubPolicy(_decision(), version=8)
    t._speco_wait_pending_drafter_publish = lambda: None
    t._speco_save_drafter_checkpoint = lambda wait=True: None

    def fake_save(self):
        # The base save commits the tracker; policy must already be present.
        assert (tmp_path / "global_step_7" / FREEZE_POLICY_SIDECAR_NAME).exists()

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", fake_save)
    t._save_checkpoint()
    sidecar = tmp_path / "global_step_7" / FREEZE_POLICY_SIDECAR_NAME
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "state": "FROZEN", "drafter_version": 8}
    assert not (tmp_path / FREEZE_POLICY_SIDECAR_NAME).exists()  # no root sidecar


@_trainer_skip
def test_failed_checkpoint_preserves_previous_tracker(tmp_path, monkeypatch) -> None:
    """A failed model save leaves its folder uncommitted and preserves resume."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    t = _bare_trainer(tmp_path, step=7)
    _attach_plain_config(t, tmp_path, _training_cfg(_freeze_cfg()))
    t._speco_freeze_policy = _StubPolicy(_decision(), version=8)
    t._speco_wait_pending_drafter_publish = lambda: None
    t._speco_save_drafter_checkpoint = lambda wait=True: None

    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("6", encoding="utf-8")

    def failing_save(self):
        (tmp_path / "global_step_7").mkdir(exist_ok=True)
        raise RuntimeError("actor save boom")

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", failing_save)
    with pytest.raises(RuntimeError, match="actor save boom"):
        t._save_checkpoint()
    assert (tmp_path / "global_step_7" / FREEZE_POLICY_SIDECAR_NAME).exists()
    assert tracker.read_text(encoding="utf-8") == "6"
    assert not (tmp_path / FREEZE_POLICY_SIDECAR_NAME).exists()


@_trainer_skip
def test_policy_save_failure_prevents_checkpoint_commit(tmp_path, monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl_speco.trainer import checkpoint

    t = _bare_trainer(tmp_path, step=7)
    _attach_plain_config(t, tmp_path, _training_cfg(_freeze_cfg()))
    t._speco_freeze_policy = _StubPolicy(_decision(), version=8)
    t._speco_wait_pending_drafter_publish = lambda: None
    t._speco_save_drafter_checkpoint = lambda wait=True: None

    def fail_write(*args, **kwargs):
        raise OSError("policy save boom")

    def unexpected_commit(self):
        pytest.fail("base checkpoint must not commit after policy save failure")

    monkeypatch.setattr(checkpoint, "atomic_write_json", fail_write)
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", unexpected_commit)
    with pytest.raises(OSError, match="policy save boom"):
        t._save_checkpoint()


@_trainer_skip
@pytest.mark.parametrize(
    ("manifest", "sidecar", "match"),
    [(True, {"state": "CALIBRATING", "drafter_version": 9}, "failed to restore"),
     (True, {"state": "FROZEN", "drafter_version": 0}, "failed to restore"),
     (True, {"state": "LEARNING", "drafter_version": 8}, "failed to restore")])
def test_freeze_resume_failfast(tmp_path, manifest, sidecar, match) -> None:
    """A committed fork must restore FROZEN at the manifest version; a silent
    reset aborts startup. (Without a manifest the checkpoint is a NORMAL one:
    its sidecar is restored as-is -- covered by the resume_path test above.)"""
    parent = _fork_folder(tmp_path, sidecar=sidecar, manifest=manifest)
    t = _resume_trainer(tmp_path, parent)
    with pytest.raises(RuntimeError, match=match):
        t._speco_freeze_load_state(_StubPolicy(_decision(), version=0))
