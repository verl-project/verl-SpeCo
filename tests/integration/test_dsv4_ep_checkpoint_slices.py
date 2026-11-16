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
"""Fault-injection tests for EP expert-slice checkpoint writes.

A checkpoint is only complete when every EP rank's ``ep_experts/rank_<r>.pt``
slice exists. A local write failure must fail the checkpoint collectively, not
just log a warning and report success.
"""

from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

base_trainer = pytest.importorskip("verl_speco.trainer.base_trainer")
DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_save_checkpoint_propagates_slice_failure(tmp_path) -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.checkpoint_dir = str(tmp_path)

    def fail_dump(checkpoint_path):
        del checkpoint_path
        raise RuntimeError("injected EP expert slice failure")

    trainer._dump_ep_expert_slices_or_fail = fail_dump

    # The error must escape save_checkpoint, so no "saved" result can be
    # reported for an incomplete checkpoint.
    with pytest.raises(RuntimeError, match="injected EP expert slice failure"):
        trainer.save_checkpoint(1)


def test_ep_slice_dump_is_skipped_without_ep() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._is_ep_active = lambda: False

    def unexpected_dump(checkpoint_path):
        del checkpoint_path
        raise AssertionError("non-EP checkpoint must not dump expert slices")

    def unexpected_sync(error, phase):
        del error, phase
        raise AssertionError("non-EP checkpoint must not sync a slice status")

    trainer._dump_local_expert_slices = unexpected_dump
    trainer._sync_checkpoint_phase_error = unexpected_sync

    trainer._dump_ep_expert_slices_or_fail("/unused")


@pytest.mark.parametrize("slice_path", [None, "/nonexistent/rank_0.pt"])
def test_ep_slice_dump_missing_path_is_failure(slice_path) -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._is_ep_active = lambda: True
    trainer._dump_local_expert_slices = lambda checkpoint_path: slice_path
    recorded = {}

    def record_sync(error, phase):
        recorded["error"] = error
        recorded["phase"] = phase

    trainer._sync_checkpoint_phase_error = record_sync

    trainer._dump_ep_expert_slices_or_fail("/unused")

    assert recorded["phase"] == "EP expert slice dump"
    assert isinstance(recorded["error"], RuntimeError)


def _run_slice_failure(rank: int, world_size: int, port: int, checkpoint_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.runtime_device = "cpu"
        trainer._is_ep_active = lambda: True
        trainer._checkpoint_process_group = lambda: dist.group.WORLD

        def dump(checkpoint_path):
            if rank == 1:
                raise OSError("injected EP expert slice write failure")
            out_dir = os.path.join(checkpoint_path, "ep_experts")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"rank_{rank}.pt")
            with open(out_path, "wb") as handle:
                handle.write(b"slice")
            return out_path

        trainer._dump_local_expert_slices = dump

        raised = None
        try:
            trainer._dump_ep_expert_slices_or_fail(checkpoint_dir)
        except RuntimeError as exc:
            raised = exc
        dist.barrier()

        assert raised is not None, "rank did not observe the collective failure"
        if rank == 1:
            assert "injected EP expert slice write failure" in str(raised)
        else:
            assert "EP expert slice dump failed" in str(raised)
    finally:
        dist.destroy_process_group()


def test_ep_slice_dump_failure_fails_all_ranks(tmp_path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the EP slice failure test")
    import torch.multiprocessing as mp

    mp.spawn(
        _run_slice_failure,
        args=(WORLD_SIZE, _free_port(), str(tmp_path)),
        nprocs=WORLD_SIZE,
        join=True,
    )
