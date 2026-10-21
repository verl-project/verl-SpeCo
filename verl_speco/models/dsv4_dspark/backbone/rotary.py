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
"""Rotary position embedding for the DSV4 DSpark draft (interleaved, YaRN-capable)."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: str = "cpu",
) -> torch.Tensor:
    def correction_dim(num_rotations: float) -> float:
        return (
            dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def correction_range(low_rot: float, high_rot: float) -> tuple[int, int]:
        low = math.floor(correction_dim(low_rot))
        high = math.ceil(correction_dim(high_rot))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp(lo: float, hi: float, n: int) -> torch.Tensor:
        if lo == hi:
            hi += 0.001
        ramp = (torch.arange(n, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(ramp, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = correction_range(beta_fast, beta_slow)
        smooth = 1 - linear_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis.to(device)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply interleaved RoPE to ``x`` using per-position frequencies.

    ``freqs_cis`` must carry the sequence axis per batch row, i.e. shape
    ``[..., S, rope_dim // 2, 2]`` with leading dims that broadcast against
    ``x``'s leading dims (``x`` is ``[..., S, H, rope_dim]``). The sequence axis
    is kept explicit: gathering the flattened ``[batch * S]`` positions and
    taking ``cos[:seq_len]`` would otherwise apply row 0's rotation to every row.
    """
    if freqs_cis.is_complex():
        freqs_cis = torch.view_as_real(freqs_cis)
    rope_dim = x.shape[-1]
    if freqs_cis.shape[-2] != rope_dim // 2:
        freqs_cis = freqs_cis[..., : rope_dim // 2, :]
    # ``freqs_cis`` is ``[..., S, rope_dim // 2, 2]``; ``x`` is either
    # ``[..., S, H, rope_dim]`` (head axis present, e.g. q/o) or
    # ``[..., S, rope_dim]`` (head-less shared KV).
    head_axis = x.dim() == freqs_cis.dim()
    seq_axis = -3 if head_axis else -2
    seq_len = x.shape[seq_axis]
    if freqs_cis.shape[-3] != seq_len:
        # Tolerate a longer precomputed table by slicing to the sequence length.
        freqs_cis = freqs_cis[..., :seq_len, :, :]
    cos = freqs_cis[..., 0].repeat_interleave(2, dim=-1)
    sin = freqs_cis[..., 1].repeat_interleave(2, dim=-1)
    if inverse:
        sin = -sin
    # Insert the broadcast axis for the head/stream dim only when it exists.
    if head_axis:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    cos = cos.float()
    sin = sin.float()
    xf = x.float()
    pair = xf.unflatten(-1, (-1, 2))
    rotate_half = torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).flatten(-2)
    return (xf * cos + rotate_half * sin).to(x.dtype)
