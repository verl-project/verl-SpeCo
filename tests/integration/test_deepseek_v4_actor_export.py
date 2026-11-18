# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise selective actor export with real sharded DTensors on two CPU ranks."""

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


def _export_worker(rank, rendezvous):
    from verl_speco.integration.deepseek_v4_online import export_target_output

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = nn.Module()
        model.model = nn.Module()
        model.model.hc_head = nn.Module()
        model.model.norm = nn.Module()
        model.model.embed_tokens = nn.Module()
        model.lm_head = nn.Module()
        values = [
            (model.model.hc_head, "hc_fn", torch.arange(8).float().reshape(2, 4)),
            (model.model.hc_head, "hc_base", torch.tensor([2.0, 3.0])),
            (model.model.hc_head, "hc_scale", torch.tensor([4.0, 5.0])),
            (model.model.norm, "weight", torch.tensor([6.0, 7.0])),
            (
                model.model.embed_tokens,
                "weight",
                torch.arange(12).float().reshape(6, 2),
            ),
            (model.lm_head, "weight", torch.arange(12).float().reshape(6, 2) + 20),
        ]
        for module, name, value in values:
            module.register_parameter(
                name, nn.Parameter(distribute_tensor(value, mesh, [Shard(0)]))
            )
        # Unrelated sharded decoder weight must not be materialized.
        model.unrelated = nn.Parameter(
            distribute_tensor(torch.ones(4, 4), mesh, [Shard(0)])
        )
        pointers = [p.to_local().data_ptr() for p in model.parameters()]

        def forbidden_full_actor_export(**kwargs):
            raise AssertionError("Full actor export must not run")

        engine = SimpleNamespace(
            module=model,
            _is_offload_param=False,
            get_per_tensor_param=forbidden_full_actor_export,
        )
        worker = SimpleNamespace(
            actor=SimpleNamespace(engine=engine),
            config={"actor": {"strategy": "fsdp2"}},
        )
        state = export_target_output(worker)
        expected = dict(
            zip(
                [
                    "hc_head_fn",
                    "hc_head_base",
                    "hc_head_scale",
                    "norm_weight",
                    "embed_tokens.weight",
                    "lm_head.weight",
                ],
                [v for _, _, v in values],
                strict=True,
            )
        )
        assert set(state) == set(expected)
        for key, value in state.items():
            assert value.device.type == "cpu" and not value.requires_grad
            assert torch.equal(value, expected[key])
        assert pointers == [p.to_local().data_ptr() for p in model.parameters()]
    finally:
        dist.destroy_process_group()


def test_selective_export_materializes_real_dtensors_without_full_actor(tmp_path):
    mp.spawn(
        _export_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )
