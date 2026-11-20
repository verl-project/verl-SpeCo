# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4 decoder block (mHC residual flow + latent attention + MoE).

The residual flow keeps ``hc_mult`` streams; each sublayer (latent attention,
then MoE) is mixed into the streams through a hyper-connection, so the block
never collapses to a single residual channel.
"""

from __future__ import annotations

import os
import time

import torch
from torch import nn
from verl.utils.device import get_device_name

from .attention import LatentAttention
from .hyper import HyperConnection, place
from .moe import MoE
from .norm import RMSNorm

_FWD_PROF = os.environ.get("DSPARK_PROFILE_FWD") == "1"
_FWD_PROF_MS = float(os.environ.get("DSPARK_PROFILE_FWD_MS", "2000"))


def _device_synchronize():
    device = get_device_name()
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _prof(tag, fn):
    if not _FWD_PROF:
        return fn()
    _device_synchronize()
    _t0 = time.perf_counter()
    out = fn()
    _device_synchronize()
    _dt = (time.perf_counter() - _t0) * 1000.0
    if _dt > _FWD_PROF_MS:
        print(f"[FWD_PROF] {tag}: {_dt:.0f} ms", flush=True)
    return out


class MhcDecoderBlock(nn.Module):
    """One DSV4 draft layer: latent attention + MoE, each wrapped in an mHC."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn = LatentAttention(cfg)
        self.ffn = MoE(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn_hc = HyperConnection(cfg)
        self.ffn_hc = HyperConnection(cfg)

    def forward(
        self,
        streams: torch.Tensor,
        context_x: torch.Tensor,
        block_freqs: torch.Tensor,
        context_freqs: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = streams
        post, comb, x = _prof("mHC.attn", lambda: self.attn_hc(streams))
        x = self.attn_norm(x)
        x = _prof(
            "MLA.attn",
            lambda: self.attn(x, context_x, block_freqs, context_freqs, attn_bias),
        )
        streams = place(x, residual, post, comb)

        residual = streams
        post, comb, x = _prof("mHC.ffn", lambda: self.ffn_hc(streams))
        x = self.ffn_norm(x)
        x = _prof("MoE.ffn", lambda: self.ffn(x))
        streams = place(x, residual, post, comb)
        return streams
