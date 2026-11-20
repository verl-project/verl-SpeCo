# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real history optimizer loop with a small deterministic loss."""

import __future__
import ast
import asyncio
import logging
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist

from verl_speco.trainer.data_buffer import DataBuffer, dspark_history_window


@pytest.mark.parametrize("make_config", [dict, SimpleNamespace])
def test_history_window_accepts_mapping_and_attribute_configs(make_config):
    drafter = make_config(
        speculative_algorithm="DSPARK",
        training=make_config(
            use_data_buffer=True,
            collect_hidden_states_from_old_logprob=True,
            sample_last_n_steps=4,
        ),
    )

    assert dspark_history_window(drafter, sync=True) == 4
    assert dspark_history_window(drafter, sync=False) == 0


def trainer_class(device_name):
    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DrafterBaseTrainer"
    )
    names = {
        "_dspark_history_collective",
        "_filter_training_data_by_step",
        "_prune_dspark_history_heads",
        "_require_dspark_history_ready",
        "_train_dspark_history_step",
        "_training_step_on_batch",
        "training_step",
    }
    trainer.body = [node for node in trainer.body if getattr(node, "name", "") in names]
    tree.body = [
        node for node in tree.body
        if getattr(node, "name", "") in {"DrafterBaseTrainer", "DSparkHistoryVersionError"}
    ]
    namespace = dict(
        Any=object,
        Optional=object,
        device_name=device_name,
        dist=dist,
        logger=logging.getLogger(__name__),
        random=random,
        time=time,
        torch=torch,
    )
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["DrafterBaseTrainer"]


def exercise_history(device, mesh=None):
    trainer = trainer_class(device.type)()
    distributed = mesh is not None
    rank = dist.get_rank() if distributed else 0
    trainer.rank = rank
    model = torch.nn.Linear(2, 2, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(0.5)
    if distributed:
        from torch.distributed.fsdp import fully_shard

        fully_shard(model, mesh=mesh)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 10)
    trainer.runtime_device = device
    trainer._full_training_world_size = lambda: dist.get_world_size() if distributed else 1
    trainer._full_training_group = lambda: dist.group.WORLD
    trainer._ulysses_group_context = nullcontext
    trainer._is_checkpoint_leader = lambda: False
    trainer._is_ep_active = lambda: False
    trainer._clip_training_grad_norm = lambda: torch.tensor(1.0, device=device)
    trainer.record_training_timing = lambda *_: None
    trainer.batch_size = 20
    trainer.dspark_history_window = 4
    trainer._dspark_history_micro_batches = 0
    trainer._dspark_history_selected_samples = 0
    trainer._dspark_history_prepared_samples = 0
    trainer.training_steps = 0
    trainer.optimizer_steps_total = 0
    trainer.data_buffer = DataBuffer(max_size=20)
    trainer._target_lm_head_snapshots = {}
    trainer._target_lm_head_weight_step = None

    def stage_head(version):
        trainer._target_lm_head_weight_step = version

    def apply_head():
        trainer._applied_target_lm_head_weight_step = trainer._target_lm_head_weight_step

    def prepare_batch(selected_items):
        assert all(
            item["target_version"] == trainer._applied_target_lm_head_weight_step
            for item in selected_items
        )
        if trainer.fail_version == trainer._applied_target_lm_head_weight_step:
            raise ValueError("injected preprocessing failure")
        return {"_speco_sample_ids": tuple(item["uid"] for item in selected_items), "hidden_states": torch.zeros(1, device=device)}

    def compute_loss(model, batch, _):
        version = trainer._applied_target_lm_head_weight_step
        value = model(torch.tensor([[1.0, 0.0]], device=device)).float().mean()
        weight = torch.tensor(version / 2, device=device)
        loss_sum = (value - version / 8).square() * weight
        return {
            "total_local_vloss": torch.zeros_like(loss_sum),
            "total_local_ploss": loss_sum,
            "local_num_tokens": weight,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "active_sample_indices": torch.arange(4, device=device),
        }

    def reduce_metrics(*values):
        metrics = torch.stack([value.detach() for value in values])
        world_size = dist.get_world_size() if distributed else 1
        if distributed:
            dist.all_reduce(metrics)
        return (*metrics.unbind(), world_size)

    trainer.select_target_lm_head_version = stage_head
    trainer._apply_pending_target_lm_head_weight = apply_head
    trainer._prepare_training_batch = prepare_batch
    trainer._reduce_loss_metrics = reduce_metrics
    trainer._sample_training_items = lambda items, size, _: items[:size]
    trainer.backend = SimpleNamespace(model_type="dspark", compute_loss=compute_loss)
    trainer.fail_version = None

    versions = list(range(1, 6))
    local_versions = versions[:-1] if rank % 2 else versions
    for version in local_versions:
        trainer.data_buffer.update_rl_step(version)
        trainer._target_lm_head_snapshots[version] = object()
        for index in range(4):
            trainer.data_buffer.add_batch(
                {"target_version": version, "uid": f"{version}-{index}"}
            )

    expected = 0.5
    all_versions = versions if not distributed else [
        version
        for worker in range(dist.get_world_size())
        for version in (versions[:-1] if worker % 2 else versions)
    ]
    for _ in range(5):
        assert asyncio.run(trainer.training_step(5))
        mean_target = sum(v * v / 8 for v in all_versions) / sum(all_versions)
        expected -= 0.05 * (expected - mean_target)
    assert trainer.optimizer_steps_total == trainer.lr_scheduler.last_epoch == 5
    assert trainer._dspark_history_micro_batches == 5 * len(local_versions)
    assert trainer._training_sample_draws == 5 * 4 * len(local_versions)
    assert len(trainer._training_unique_sample_ids) == 4 * len(local_versions)
    actual_weight = model.weight.full_tensor() if distributed else model.weight
    torch.testing.assert_close(actual_weight[:, 0].mean().item(), expected, rtol=0, atol=0.003)

    if rank == 0:
        trainer.fail_version = 2
    assert not asyncio.run(trainer.training_step(5))
    assert trainer.optimizer_steps_total == trainer.lr_scheduler.last_epoch == 5
    assert all(parameter.grad is None for parameter in model.parameters())
    trainer.fail_version = None

    if rank == 0:
        trainer._target_lm_head_snapshots.pop(2)
    assert not asyncio.run(trainer.training_step(5))
    assert trainer.optimizer_steps_total == trainer.lr_scheduler.last_epoch == 5
    if rank == 0:
        assert trainer.last_training_error["type"] == "DSparkHistoryVersionError"
    assert trainer._training_sample_draws == 5 * 4 * len(local_versions)

    assert trainer.data_buffer.discard_before_step(2) == 4
    trainer._prune_dspark_history_heads()
    assert 1 not in trainer._target_lm_head_snapshots
    print(f"PASS rank={rank} steps=5 versions={local_versions} samples={trainer._training_sample_draws}")


def test_history_objective_and_atomic_commit():
    exercise_history(torch.device("cpu"))


if __name__ == "__main__":
    import torch_npu  # noqa: F401
    from torch.distributed.device_mesh import init_device_mesh

    device_index = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{device_index}")
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    try:
        world_size = dist.get_world_size()
        mesh = init_device_mesh("npu", (world_size // 2, 2), mesh_dim_names=("dp", "shard"))
        exercise_history(device, mesh)
    finally:
        dist.destroy_process_group()
