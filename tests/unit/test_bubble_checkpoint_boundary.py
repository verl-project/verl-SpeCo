# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Checkpoint deferral must preserve training and upstream commit ordering."""

import ast
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bubble_review_regressions import _methods

from verl_speco.trainer.checkpoint import (
    DrafterCheckpointMetadataError,
    validate_checkpoint_buffer_ownership,
)
from verl_speco.trainer.data_buffer import DataBuffer


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    path = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SpecoRayPPOTrainer"
    )

    class Parent:
        def _save_checkpoint(self):
            self.calls.append(("actor_save", self.global_steps))
            if self.fail_actor_save:
                raise OSError("actor save failed")
            return "saved"

    names = {
        "_save_checkpoint",
        "_speco_checkpoint_pending_quota",
        "_speco_retry_deferred_checkpoint",
        "_speco_is_final_training_step",
    }
    selected = ast.ClassDef(
        name="Driver",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        decorator_list=[],
    )
    namespace = {"Parent": Parent, "logger": logging.getLogger(__name__)}
    exec(  # noqa: S102 - execute the production methods without optional GPU imports
        compile(
            ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    obj = namespace["Driver"]()
    obj.global_steps, obj.total_training_steps = 5, 50
    obj.calls, obj.fail_actor_save = [], False
    obj.scheduler = SimpleNamespace(
        _training_quota_debt_steps=0, _training_quota_collection_step=None
    )
    obj._speco_rollout_idle_worker_enabled = lambda: True
    obj._speco_get_drafter_scheduler = lambda: obj.scheduler
    obj._speco_quiesce_bubble_for_checkpoint = lambda: obj.calls.append("quiesce")
    obj._speco_wait_pending_drafter_publish = lambda: obj.calls.append("publish_wait")
    obj._speco_save_drafter_checkpoint = lambda **kwargs: obj.calls.append(
        "drafter_save"
    )
    obj._speco_freeze_save_state = lambda: obj.calls.append("freeze_save")
    return obj


@pytest.mark.parametrize("debt,collection", [(15, 4), (0, 4), (0, 0)])
def test_periodic_save_waits_for_quota_and_publication_without_io(
    driver, debt, collection
):
    driver.scheduler._training_quota_debt_steps = debt
    driver.scheduler._training_quota_collection_step = collection
    assert driver._save_checkpoint() is None
    assert driver._speco_deferred_checkpoint_step == 5
    assert not driver.calls  # No reclaim, head wait, training or file writes.
    driver.global_steps = 6
    driver._save_checkpoint()
    driver._speco_retry_deferred_checkpoint()
    assert driver._speco_deferred_checkpoint_step == 5
    assert driver.scheduler._training_quota_debt_steps == debt
    assert not driver.calls


def test_retry_saves_actual_step_once_after_publish_ack(driver):
    driver.scheduler._training_quota_debt_steps = 15
    driver.scheduler._training_quota_collection_step = 4
    driver._save_checkpoint()
    driver.scheduler._training_quota_debt_steps = 0
    driver._speco_retry_deferred_checkpoint()
    assert not driver.calls  # Quota complete, but publish ACK is still missing.
    driver.scheduler._training_quota_collection_step = None
    driver.global_steps = 7
    driver._speco_retry_deferred_checkpoint()
    assert driver.calls == [
        "quiesce",
        "publish_wait",
        "drafter_save",
        "freeze_save",
        ("actor_save", 7),
    ]
    assert driver._speco_deferred_checkpoint_step is None
    assert driver._speco_last_checkpoint_saved_step == 7
    driver._speco_retry_deferred_checkpoint()
    driver._save_checkpoint()  # Coinciding periodic save must be deduplicated.
    assert driver.calls.count(("actor_save", 7)) == 1


def test_retry_failure_keeps_request_and_does_not_mark_saved(driver):
    driver._speco_deferred_checkpoint_step = 5
    driver.fail_actor_save = True
    with pytest.raises(OSError, match="actor save failed"):
        driver._speco_retry_deferred_checkpoint()
    assert driver._speco_deferred_checkpoint_step == 5
    assert not hasattr(driver, "_speco_last_checkpoint_saved_step")
    assert not driver._speco_checkpoint_in_progress


@pytest.mark.parametrize("boundary", ["final", "freeze"])
def test_unfinished_final_or_freeze_save_fails_before_io(driver, boundary):
    driver.scheduler._training_quota_debt_steps = 15
    if boundary == "final":
        driver.global_steps = 50
    else:
        driver._speco_freeze_branch_save_pending = {"trigger_step": 5}
    with pytest.raises(RuntimeError, match="unfinished Bubble quota"):
        driver._save_checkpoint()
    assert not driver.calls


def test_sync_save_does_not_read_bubble_scheduler(driver):
    driver._speco_rollout_idle_worker_enabled = lambda: False
    driver._speco_get_drafter_scheduler = lambda: pytest.fail("Sync read Bubble state")
    assert driver._save_checkpoint() == "saved"
    assert driver.calls == [
        "quiesce",
        "publish_wait",
        "drafter_save",
        "freeze_save",
        ("actor_save", 5),
    ]


def test_quota_race_is_detected_before_checkpoint_writes(driver):
    def quiesce():
        driver.scheduler._training_quota_debt_steps = 1

    driver._speco_quiesce_bubble_for_checkpoint = quiesce
    with pytest.raises(RuntimeError, match="quota changed"):
        driver._save_checkpoint()
    assert driver.calls == ["publish_wait"]
    assert not driver._speco_checkpoint_in_progress


@pytest.mark.parametrize(
    "guard",
    [
        "_speco_checkpoint_in_progress",
        "_speco_validation_in_progress",
        "_speco_rollout_idle_generation_open",
    ],
)
def test_deferred_retry_does_not_enter_busy_boundaries(driver, guard):
    driver._speco_deferred_checkpoint_step = 5
    setattr(driver, guard, True)
    driver._speco_get_drafter_scheduler = lambda: pytest.fail(
        "busy boundary inspected scheduler"
    )
    driver._speco_retry_deferred_checkpoint()
    assert driver._speco_deferred_checkpoint_step == 5
    assert not driver.calls


@pytest.mark.parametrize(
    "ownership",
    [{"_drafter_replay_pinned": True}, {"_drafter_reserved_by": "old-plan"}],
)
def test_legacy_orphan_ownership_is_rejected_without_mutating_samples(ownership):
    state = {"data_buffer": [{"id": 1, **ownership}]}
    with pytest.raises(
        DrafterCheckpointMetadataError, match="unfinished Bubble replay"
    ):
        validate_checkpoint_buffer_ownership(state)
    assert state["data_buffer"] == [{"id": 1, **ownership}]


def test_completed_replay_checkpoint_roundtrip_accepts_new_samples():
    torch = pytest.importorskip("torch")
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_speco_checkpoint_buffer_state"},
    )
    worker = cls()
    worker.use_data_buffer = True
    worker.data_buffer = DataBuffer(max_size=2)
    worker.collected_data = []
    worker.buffer_version, worker.current_rl_step = 1, 7
    for index, sample_id in enumerate(("a", "b")):
        worker.data_buffer.add_batch(
            {
                "id": sample_id,
                "target_version": 4,
                "hidden_states": torch.arange(8, dtype=torch.bfloat16) + index,
            }
        )
    items = worker.data_buffer.reserve("quota", target_version=4, max_samples=2)
    worker.data_buffer.pin_replay_samples(items)
    with pytest.raises(DrafterCheckpointMetadataError):
        worker._speco_checkpoint_buffer_state()
    worker.data_buffer.unpin_replay_samples(items)
    worker.data_buffer.consume("quota", items[:1])
    worker.data_buffer.release_reservation("quota")
    stream = io.BytesIO()
    torch.save(worker._speco_checkpoint_buffer_state(), stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    validate_checkpoint_buffer_ownership(restored)
    buffer = DataBuffer(max_size=2)
    buffer.buffer.extend(restored["data_buffer"])
    assert [item["id"] for item in buffer.get_all_data()] == ["b"]
    assert torch.equal(
        buffer.get_all_data()[0]["hidden_states"], items[1]["hidden_states"]
    )
    assert buffer.add_batch({"id": "c", "target_version": 8})
    assert buffer.add_batch({"id": "d", "target_version": 8})
    assert [item["id"] for item in buffer.get_all_data()] == ["c", "d"]
