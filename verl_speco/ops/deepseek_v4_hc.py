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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/hc_backward.py.
import torch
import torch.nn.functional as F


def hc_pre_explicit_backward_chunk(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    grad_collapsed: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    need_x: bool = True,
    need_hc_fn: bool = True,
    need_hc_scale: bool = True,
    need_hc_base: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Evaluate the HC-pre VJP explicitly for one bounded token chunk."""
    x_float = x.float()
    x_flat = x_float.flatten(start_dim=-2)
    weight = hc_fn.float()
    scale = hc_scale.float()
    base = hc_base.float()

    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    linear = F.linear(x_flat, weight)
    logits = linear * rsqrt
    pre_logits = logits[:, :hc_mult]
    post_logits = logits[:, hc_mult : 2 * hc_mult]
    comb_logits = logits[:, 2 * hc_mult :]

    pre_sigmoid = torch.sigmoid(scale[0] * pre_logits + base[:hc_mult])
    pre = pre_sigmoid + hc_eps
    post_sigmoid = torch.sigmoid(scale[1] * post_logits + base[hc_mult : 2 * hc_mult])

    comb_affine = (scale[2] * comb_logits + base[2 * hc_mult :]).reshape(
        -1, hc_mult, hc_mult
    )
    comb_softmax = torch.softmax(comb_affine, dim=-1)
    comb = comb_softmax + hc_eps
    normalizations: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for _ in range(sinkhorn_iters):
        denominator = comb.sum(dim=-1, keepdim=True) + hc_eps
        comb = comb / denominator
        normalizations.append((comb, denominator, -1))
        denominator = comb.sum(dim=-2, keepdim=True) + hc_eps
        comb = comb / denominator
        normalizations.append((comb, denominator, -2))

    grad_collapsed_float = grad_collapsed.float()
    grad_pre = torch.bmm(
        x_float,
        grad_collapsed_float.unsqueeze(-1),
    ).squeeze(-1)
    grad_pre_affine = grad_pre * pre_sigmoid * (1.0 - pre_sigmoid)
    grad_x_direct = None
    if need_x:
        grad_x_direct = grad_collapsed_float.unsqueeze(-2) * pre.unsqueeze(-1)

    grad_post_affine = grad_post.float() * 2.0 * post_sigmoid * (1.0 - post_sigmoid)

    grad_comb_value = grad_comb.float()
    for normalized, denominator, dim in reversed(normalizations):
        grad_comb_value = (
            grad_comb_value - (grad_comb_value * normalized).sum(dim=dim, keepdim=True)
        ) / denominator
    grad_comb_affine = comb_softmax * (
        grad_comb_value - (grad_comb_value * comb_softmax).sum(dim=-1, keepdim=True)
    )
    grad_comb_affine_flat = grad_comb_affine.reshape(
        -1,
        hc_mult * hc_mult,
    )

    grad_logits = torch.cat(
        (
            grad_pre_affine * scale[0],
            grad_post_affine * scale[1],
            grad_comb_affine_flat * scale[2],
        ),
        dim=-1,
    )
    grad_linear = grad_logits * rsqrt
    grad_x = None
    if need_x:
        grad_rsqrt = (grad_logits * linear).sum(dim=-1, keepdim=True)
        grad_x_flat = grad_linear.matmul(weight)
        grad_x_flat.add_(-grad_rsqrt * x_flat * rsqrt.pow(3) / x_flat.shape[-1])
        grad_x = grad_x_flat.reshape_as(x_float)
        grad_x.add_(grad_x_direct)

    grad_hc_fn = grad_linear.transpose(0, 1).matmul(x_flat) if need_hc_fn else None
    grad_hc_scale = None
    if need_hc_scale:
        grad_hc_scale = torch.stack(
            (
                (grad_pre_affine * pre_logits).sum(),
                (grad_post_affine * post_logits).sum(),
                (grad_comb_affine_flat * comb_logits).sum(),
            )
        )
    grad_hc_base = None
    if need_hc_base:
        grad_hc_base = torch.cat(
            (
                grad_pre_affine.sum(dim=0),
                grad_post_affine.sum(dim=0),
                grad_comb_affine_flat.sum(dim=0),
            )
        )
    return (
        grad_x.to(x.dtype) if grad_x is not None else None,
        grad_hc_fn.to(hc_fn.dtype) if grad_hc_fn is not None else None,
        (grad_hc_scale.to(hc_scale.dtype) if grad_hc_scale is not None else None),
        grad_hc_base.to(hc_base.dtype) if grad_hc_base is not None else None,
    )


def hc_post_explicit_backward_chunk(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    need_x: bool = True,
    need_residual: bool = True,
    need_post: bool = True,
    need_comb: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Evaluate the HC-post VJP without broadcast-sized intermediates."""
    grad_float = grad_output.float()
    grad_x = None
    if need_x:
        grad_x = (
            torch.bmm(
                post.float().unsqueeze(1),
                grad_float,
            )
            .squeeze(1)
            .to(x.dtype)
        )
    grad_residual = None
    if need_residual:
        grad_residual = torch.bmm(
            comb.float(),
            grad_float,
        ).to(residual.dtype)
    grad_post = None
    if need_post:
        grad_post = (
            torch.bmm(
                grad_float,
                x.float().unsqueeze(-1),
            )
            .squeeze(-1)
            .to(post.dtype)
        )
    grad_comb = None
    if need_comb:
        grad_comb = torch.bmm(
            residual.float(),
            grad_float.transpose(-1, -2),
        ).to(comb.dtype)
    return grad_x, grad_residual, grad_post, grad_comb
