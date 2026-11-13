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
"""Distributed test for the EP replicated-state broadcast.

Random initialization builds non-expert parameters with the process-local RNG,
which differs across ranks. EP must broadcast them from rank 0 while keeping the
rank-disjoint routed experts untouched.
"""

from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.zeros(2, 2, 2))
        self.w2 = torch.nn.Parameter(torch.zeros(2, 2, 2))
        self.register_buffer("expert_scale", torch.zeros(2))


class _FFN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = _FFN()


class _Draft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.router_weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("shared_bias", torch.zeros(2))


class _Wrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.draft_model = _Draft()
        self.wrapper_param = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("wrapper_bias", torch.zeros(2))


def _expert_ids(model: _Wrapper) -> tuple[set[int], set[int]]:
    params: set[int] = set()
    buffers: set[int] = set()
    for layer in model.draft_model.layers:
        params.update(id(p) for p in layer.ffn.experts.parameters())
        buffers.update(id(b) for b in layer.ffn.experts.buffers())
    return params, buffers


def _run(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from verl_speco.backends.dsv4_ep_utils import broadcast_replicated_state

        model = _Wrapper()
        expert_params, expert_buffers = _expert_ids(model)
        local_expert_value = float(rank + 1) * 10.0
        local_replicated_value = float(rank + 1)

        with torch.no_grad():
            for param in model.parameters():
                param.fill_(
                    local_expert_value
                    if id(param) in expert_params
                    else local_replicated_value
                )
            for buffer in model.buffers():
                buffer.fill_(
                    local_expert_value
                    if id(buffer) in expert_buffers
                    else local_replicated_value
                )

        broadcast_replicated_state(model, src=0)
        dist.barrier()

        for param in model.parameters():
            expected = local_expert_value if id(param) in expert_params else 1.0
            torch.testing.assert_close(param, torch.full_like(param, expected))
        for buffer in model.buffers():
            expected = local_expert_value if id(buffer) in expert_buffers else 1.0
            torch.testing.assert_close(buffer, torch.full_like(buffer, expected))
    finally:
        dist.destroy_process_group()


def test_ep_broadcast_replicated_state() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the EP broadcast test")
    import torch.multiprocessing as mp

    mp.spawn(
        _run,
        args=(WORLD_SIZE, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )
