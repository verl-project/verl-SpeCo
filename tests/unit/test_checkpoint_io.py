# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.checkpoint_io import (
    bounded_checkpoint_writes,
    checkpoint_io_slot,
)


class TorchCheckpointEngine:
    zipfile_serialization = False

    def save(self, state_dict, path):
        assert self.zipfile_serialization
        torch.save(state_dict, path)


def _writer(directory, active, peak):
    import time

    with checkpoint_io_slot(directory):
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
        time.sleep(0.05)
        with active.get_lock():
            active.value -= 1


def test_processes_share_one_serialization_slot(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    active, peak = ctx.Value("i", 0), ctx.Value("i", 0)
    workers = [
        ctx.Process(target=_writer, args=(tmp_path, active, peak)) for _ in range(3)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
            assert worker.exitcode == 0
        assert peak.value == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def test_write_roundtrip_and_restore_on_failure(tmp_path):
    backend = TorchCheckpointEngine()
    engine = SimpleNamespace(checkpoint_engine=backend)
    payload = {"weights": torch.arange(3), "optimizer": {"step": 1}}
    with (
        pytest.raises(RuntimeError, match="injected"),
        bounded_checkpoint_writes(engine, tmp_path),
    ):
        backend.save(payload, tmp_path / "state.pt")
        raise RuntimeError("injected")
    assert not backend.zipfile_serialization
    assert "save" not in vars(backend)
    restored = torch.load(tmp_path / "state.pt", weights_only=True)
    assert torch.equal(restored["weights"], payload["weights"])
    assert restored["optimizer"] == payload["optimizer"]
    with checkpoint_io_slot(tmp_path):
        pass
