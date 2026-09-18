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
"""Distributed regression tests for the online DDP drafter path.

`_full_training_group` must span every rank that holds a drafter replica
(DP x SP), not just the SP group. Wrapping DDP over the SP group alone leaves
the rollout replicas diverged and makes the `reduce_world_size` loss scaling
wrong. The trainer state-dict export must also recognize the DDP wrapper
instead of routing it through the FSDP2 state-dict API.
"""
from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from verl_speco.trainer.base_trainer import (  # noqa: E402
    DrafterBaseTrainer,
    _DrafterOptimizerState,
)

WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(rank: int, world_size: int, port: int, checkpoint_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "sp"))

        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.training_device_mesh = mesh
        trainer.training_process_group = mesh["sp"].get_group()
        trainer.training_group_world_size = mesh["sp"].size()

        # The DDP sync domain must cover both DP replicas and SP ranks.
        full_group = trainer._full_training_group()
        assert dist.get_world_size(full_group) == world_size
        assert trainer._full_training_world_size() == world_size

        model = torch.nn.Linear(4, 4, bias=False)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, process_group=full_group, broadcast_buffers=False
        )
        trainer.model = ddp_model

        # Each rank contributes a different loss; after backward the grads must
        # agree across all ranks, including across the DP dimension.
        local_input = torch.full((1, 4), float(rank + 1))
        loss = (ddp_model(local_input) ** 2).sum()
        loss.backward()
        grad = model.weight.grad
        assert grad is not None
        gathered = [torch.empty_like(grad) for _ in range(world_size)]
        dist.all_gather(gathered, grad, group=full_group)
        for other in gathered[1:]:
            assert torch.allclose(gathered[0], other, atol=1e-6)

        # State-dict export must take the DDP branch, not the FSDP2 helper.
        state_dict = trainer._get_full_model_state_dict()
        assert set(state_dict) == {"weight"}
        assert set(trainer._get_full_export_state_dict()) == {"weight"}

        # Optimizer checkpointing must also avoid the FSDP-only DCP API.
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
        loss = (ddp_model(local_input) ** 2).sum()
        loss.backward()
        optimizer.step()
        adapter = _DrafterOptimizerState(ddp_model, optimizer)
        assert adapter._is_ddp()
        optimizer_state = adapter.state_dict()
        assert set(optimizer_state) == {"state", "param_groups"}
        adapter.load_state_dict(optimizer_state)

        # A real DCP round-trip through the guarded adapter.
        import torch.distributed.checkpoint as dcp

        checkpoint_id = os.path.join(checkpoint_dir, "optimizer.incomplete")
        dcp.save({"optimizer": adapter}, checkpoint_id=checkpoint_id)
        for group in optimizer.param_groups:
            for param in group["params"]:
                optimizer.state[param]["step"] = torch.tensor(123.0)
        dcp.load({"optimizer": adapter}, checkpoint_id=checkpoint_id)
        restored_steps = {
            float(optimizer.state[param]["step"])
            for group in optimizer.param_groups
            for param in group["params"]
        }
        assert restored_steps == {1.0}, restored_steps
    finally:
        dist.destroy_process_group()


def test_ddp_synchronizes_over_full_training_mesh(tmp_path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the DDP regression test")
    import torch.multiprocessing as mp

    port = _free_port()
    checkpoint_dir = str(tmp_path / "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    mp.spawn(
        _run,
        args=(WORLD_SIZE, port, checkpoint_dir),
        nprocs=WORLD_SIZE,
        join=True,
    )
