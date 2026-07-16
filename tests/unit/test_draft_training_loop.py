from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from verl_speco.trainer.draft_training_loop import _save_standalone_checkpoint


class _FakeTrainer:
    def __init__(self):
        self.checkpoint_dir = "/tmp/draft"
        self._pending_full_checkpoint_future = None
        self.future = Future()
        self.calls = 0

    def _save_checkpoint_async(self, step: int):
        self.calls += 1
        self.step = step
        self._pending_full_checkpoint_future = self.future
        return self.future


def test_standalone_checkpoint_schedules_without_waiting():
    trainer = _FakeTrainer()

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is True
    assert result["reason"] == "scheduled"
    assert trainer.calls == 1
    assert trainer._pending_full_checkpoint_future is trainer.future


def test_standalone_checkpoint_waits_when_requested():
    trainer = _FakeTrainer()
    trainer.future.set_result(None)

    result = _save_standalone_checkpoint(trainer, 5, wait=True)

    assert result["saved"] is True
    assert result["reason"] == "saved"
    assert trainer._pending_full_checkpoint_future is None


def test_standalone_checkpoint_skips_when_previous_save_is_running():
    trainer = SimpleNamespace(checkpoint_dir="/tmp/draft", _pending_full_checkpoint_future=Future())

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is False
    assert result["reason"] == "previous_save_running"
