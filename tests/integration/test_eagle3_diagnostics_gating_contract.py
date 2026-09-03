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
"""Contract tests for the EAGLE-3 drafter quality diagnostics.

They are log-only and every one of them forces a device sync, so they must not
run on the hot path unless the log will actually be emitted.
"""

from __future__ import annotations

import logging

import pytest

QUALITY_LOG_PREFIX = "[drafter logits quality]"
BACKEND_LOGGER = "verl_speco.backends.eagle3_trainer_backend"


def _backend_and_inputs():
    import torch
    from omegaconf import OmegaConf

    from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend

    vocab_size, seq_len, ttt_length = 6, 4, 2

    backend = Eagle3TrainerBackend(
        OmegaConf.create(
            {
                "rollout": {
                    "drafter": {"training": {"use_logits": False, "ttt_length": ttt_length}}
                },
                "model": {"path": "/tmp/none"},
            }
        ),
        OmegaConf.create({}),
    )
    backend.target_model = lambda last_hidden: torch.zeros(
        *last_hidden.shape[:-1], vocab_size
    )

    class _FakeDraft:
        t2d = torch.ones(vocab_size, dtype=torch.bool)

        def __call__(self, **kwargs):
            return {
                "logits": [
                    torch.randn(1, seq_len, vocab_size) for _ in range(ttt_length)
                ],
                "position_masks": [
                    torch.ones(1, seq_len) for _ in range(ttt_length)
                ],
            }

    batch = {
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "hidden_states": torch.zeros(1, seq_len, 8),
        "last_hidden_states": torch.zeros(1, seq_len, 8),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(1, seq_len),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
    }
    return backend, _FakeDraft(), batch


class _Recorder(logging.Handler):
    """Records everything the backend logger emits, whatever its level is.

    ``caplog.at_level`` would raise the logger to DEBUG, which is exactly the
    condition under test, so the handler is attached directly instead.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _compute_loss_at(level: int) -> list[logging.LogRecord]:
    backend, model, batch = _backend_and_inputs()
    backend_logger = logging.getLogger(BACKEND_LOGGER)
    previous_level = backend_logger.level
    recorder = _Recorder()
    backend_logger.setLevel(level)
    backend_logger.addHandler(recorder)
    try:
        backend.compute_loss(model, batch, 0)
    finally:
        backend_logger.removeHandler(recorder)
        backend_logger.setLevel(previous_level)
    return [r for r in recorder.records if QUALITY_LOG_PREFIX in r.getMessage()]


def test_quality_diagnostics_are_silent_above_debug() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    assert _compute_loss_at(logging.INFO) == []


def test_quality_diagnostics_are_emitted_at_debug() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    quality_records = _compute_loss_at(logging.DEBUG)

    assert quality_records
    # Routine per-step training metrics belong at DEBUG, not WARNING.
    assert all(r.levelno == logging.DEBUG for r in quality_records)


def test_quality_diagnostics_do_not_sync_above_debug() -> None:
    """Above DEBUG the loss must not pay for the diagnostics' device syncs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    original_item = torch.Tensor.item

    def run_at(level: int) -> int:
        backend, model, batch = _backend_and_inputs()
        backend_logger = logging.getLogger(BACKEND_LOGGER)
        previous_level = backend_logger.level
        backend_logger.setLevel(level)
        calls = 0

        def counting_item(self):
            nonlocal calls
            calls += 1
            return original_item(self)

        torch.Tensor.item = counting_item
        try:
            backend.compute_loss(model, batch, 0)
        finally:
            torch.Tensor.item = original_item
            backend_logger.setLevel(previous_level)
        return calls

    assert run_at(logging.INFO) < run_at(logging.DEBUG)
