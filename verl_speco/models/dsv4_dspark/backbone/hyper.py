# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Manifold-constrained hyper-connections (mHC).

Instead of a single residual stream, each block keeps ``hc_mult`` copies. Each
sublayer output is expanded back into the streams through a softmax mixing
matrix that is constrained to the Birkhoff polytope (doubly stochastic) via
Sinkhorn-Knopp iterations, bounding the spectral norm at 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import get_kernel, torch_kernel
from .norm import UnweightedRMSNorm

_HC_OP = "mhc_hyper_connection"


@torch_kernel(_HC_OP)
def _hyper_connection_torch(
    module: HyperConnection, streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc = module.hc_mult
    eps = module.hc_eps
    flat = module.input_norm(streams.flatten(start_dim=2).float())
    mix = F.linear(flat, module.fn.float())
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = module.base.float().split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = module.scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.reshape(
        hc, hc
    )
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return post, comb, collapsed


def place(
    out: torch.Tensor,
    residual_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    placed = post.unsqueeze(-1) * out.unsqueeze(-2)
    mixed = torch.sum(comb.unsqueeze(-1) * residual_streams.unsqueeze(-2), dim=2)
    return (placed + mixed).type_as(out)


class HyperConnection(nn.Module):
    """Collapse ``hc_mult`` streams for a sublayer input and return the
    ``(post, comb)`` mixing factors used to place the sublayer output back.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * cfg.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))

    def forward(
        self, streams: torch.Tensor, backend: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return get_kernel(_HC_OP, backend)(self, streams)


class HyperHead(nn.Module):
    """Final mHC collapse of the ``hc_mult`` streams into one hidden state."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * cfg.hidden_size)
        )
        self.hc_base = nn.Parameter(torch.zeros(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.ones(1))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.hc_eps
        )
        return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
