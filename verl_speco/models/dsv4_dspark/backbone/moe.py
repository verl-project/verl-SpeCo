# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeekMoE layer for the DSV4 draft backbone.

``256`` routed experts (SwiGLU with a magnitude clamp) + ``1`` shared expert per
layer, with ``sqrt(softplus)`` routing affinity, top-k selection and an optional
load-balance bias that also breaks first-step routing symmetry.
"""

from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import get_kernel, torch_kernel

_MOE_OP = "moe_dispatch"


class Router(nn.Module):
    """Top-k router with sqrt(softplus) affinity and an optional load-balance bias."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.topk = cfg.n_activated_experts
        self.route_scale = cfg.route_scale
        self.score_func = cfg.score_func
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        self.register_buffer("bias", torch.zeros(cfg.n_routed_experts), persistent=True)
        self.n_routed_experts = cfg.n_routed_experts
        self._log_load = os.environ.get("DSPARK_LOG_EXPERT_LOAD") == "1"
        self._balance = os.environ.get("DSPARK_MOE_BALANCE") == "1"
        self._balance_rate = float(os.environ.get("DSPARK_MOE_BALANCE_RATE", "1e-3"))
        self._sel_counts: torch.Tensor | None = None
        self._step_load: torch.Tensor | None = None
        self._bias_initialized = False

    def _score(self, scores: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return scores.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return scores.sigmoid()
        return F.softplus(scores).clamp_min(1e-6).sqrt()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._score(F.linear(x.float(), self.weight.float()))
        selection = scores + self.bias.float()
        indices = selection.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weights = weights * self.route_scale
        if self._log_load or (self._balance and self.training):
            _cnt = (
                torch.bincount(indices.reshape(-1), minlength=self.n_routed_experts)
                .detach()
                .float()
            )
            if self._log_load:
                self._sel_counts = _cnt
            if self._balance and self.training:
                self._step_load = _cnt
        return weights, indices

    @torch.no_grad()
    def update_load_balance_bias(self) -> None:
        if not self._balance:
            return
        # First-step initialization: inject uniform bias to prevent expert collapse
        # on the very first forward (before any load is recorded)
        if not self._bias_initialized:
            self._bias_initialized = True
            # Small random perturbation around zero to break symmetry
            # but large enough to distribute tokens across all experts
            init_scale = self._balance_rate * 10
            self.bias.add_(torch.randn_like(self.bias) * init_scale - init_scale * 0.5)
            return
        if self._step_load is None:
            return
        load = self._step_load
        import torch.distributed as dist

        if dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)
        delta = self._balance_rate * torch.sign(load.mean() - load)
        delta = delta - delta.mean()
        self.bias.add_(delta.to(self.bias.dtype))
        self._step_load = None


class Expert(nn.Module):
    """A single SwiGLU expert (used for the shared expert)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        return self.w2(h.to(dtype))


def swiglu_grouped(
    xg: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    counts: torch.Tensor,
    w_flat: torch.Tensor,
    swiglu_limit: float,
    matmul,
) -> torch.Tensor:
    gate = matmul(xg, w1.transpose(1, 2), counts)
    up = matmul(xg, w3.transpose(1, 2), counts)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    h = w_flat[:, None] * h
    return matmul(h, w2.transpose(1, 2), counts)


class GroupedExperts(nn.Module):
    """Stacked routed-expert weights ``[n_local, ...]`` for grouped dispatch."""

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_local: int,
        swiglu_limit: float,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_local_experts = n_local
        self.swiglu_limit = swiglu_limit
        self.w1 = nn.Parameter(torch.empty(n_local, inter_dim, dim))
        self.w3 = nn.Parameter(torch.empty(n_local, inter_dim, dim))
        self.w2 = nn.Parameter(torch.empty(n_local, dim, inter_dim))
        b1, b2 = 1.0 / math.sqrt(dim), 1.0 / math.sqrt(inter_dim)
        ctx = (
            torch.random.fork_rng(devices=[])
            if seed is not None
            else contextlib.nullcontext()
        )
        with ctx, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            self.w1.uniform_(-b1, b1)
            self.w3.uniform_(-b1, b1)
            self.w2.uniform_(-b2, b2)

    def local_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def loc(p):
            return p.to_local() if hasattr(p, "to_local") else p

        return loc(self.w1), loc(self.w3), loc(self.w2)


@torch_kernel(_MOE_OP)
def _moe_dispatch_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
    n_routed_experts: int,
) -> torch.Tensor:
    w1, w3, w2 = experts.local_weights()
    lim = experts.swiglu_limit
    y = torch.zeros_like(x, dtype=torch.float32)
    for e in range(n_routed_experts):
        tok, slot = torch.where(indices == e)
        xe = x[tok].float()
        gate = xe @ w1[e].t().float()
        up = xe @ w3[e].t().float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up
        h = weights[tok, slot, None].float() * h
        y[tok] += h @ w2[e].t().float()
    return y


class MoE(nn.Module):
    """Router -> routed experts -> shared expert.

    Under EP the routed experts are sliced to the local rank at construction;
    otherwise all experts are held locally.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.n_routed_experts = cfg.n_routed_experts
        self.router = Router(cfg)
        from . import moe_ep

        ep = moe_ep._EP
        if ep is not None and ep.size > 1:
            n_local = cfg.n_routed_experts // ep.size
            seed = 0xE9E9 + ep.rank
        else:
            n_local = cfg.n_routed_experts
            seed = None
        self.experts = GroupedExperts(
            cfg.hidden_size, cfg.moe_inter_dim, n_local, cfg.swiglu_limit, seed=seed
        )
        if cfg.n_shared_experts != 1:
            raise ValueError(
                "only n_shared_experts == 1 is supported (matches the release)."
            )
        self.shared_experts = Expert(
            cfg.hidden_size, cfg.moe_inter_dim, cfg.swiglu_limit
        )

    def forward(self, x: torch.Tensor, backend: str | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.router(x)
        # On NPU the accelerated dispatch is only active once expert parallelism
        # is enabled (``moe_ep.enable()``); otherwise this falls back to the
        # 256-way torch reference loop. Enable EP for NPU training throughput.
        y = get_kernel(_MOE_OP, backend)(
            x, weights, indices, self.experts, self.n_routed_experts
        )
        y = y + self.shared_experts(x)
        return y.type_as(x).reshape(shape)
