# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NPU grouped-matmul primitives for the DSV4 MoE.

Replaces the 256-way per-expert eager loop with one grouped matmul per
projection. Forward and backward both bucket-pad the token count so the NPU
grouped matmul never sees a degenerate/empty group. Consumed by
``moe_ep.moe_dispatch_ep`` (the single registered ``moe_dispatch`` NPU kernel).
"""

from __future__ import annotations

import os

import torch

from .moe import GroupedExperts, swiglu_grouped

_MOE_BUCKET = int(os.environ.get("DSPARK_MOE_BUCKET", "512"))
_MOE_PROF = os.environ.get("DSPARK_PROFILE_MOE") == "1"
_MOE_PROF_MS = float(os.environ.get("DSPARK_PROFILE_MOE_MS", "2000"))
_MOE_EMPTY_MIN = 16


def _bucket_count(n: int) -> int:
    if n == 0:
        return _MOE_BUCKET if _MOE_BUCKET > 1 else _MOE_EMPTY_MIN
    if _MOE_BUCKET <= 1:
        return n
    return ((n + _MOE_BUCKET - 1) // _MOE_BUCKET) * _MOE_BUCKET


def _grouped_matmul_torch(
    x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    outs = []
    off = 0
    for e in range(weight.shape[0]):
        n = int(counts[e])
        outs.append(x[off : off + n] @ weight[e])
        off += n
    return torch.cat(outs, dim=0) if outs else x.new_zeros((0, weight.shape[-1]))


class _NpuGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, counts):
        import torch_npu

        n = x.shape[0]
        nb = _bucket_count(n)
        if nb > n:
            pad = nb - n
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            counts = counts.clone()
            counts[-1] = counts[-1] + pad
        group_list = torch.cumsum(counts, dim=0).to(torch.int64)
        out = torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
        ctx.save_for_backward(x, weight, group_list)
        ctx._orig_n = n
        return out[:n] if nb > n else out

    @staticmethod
    def backward(ctx, grad):
        import torch_npu

        x, weight, group_list = ctx.saved_tensors
        n = ctx._orig_n
        nb = x.shape[0]
        if nb > n:
            pad = nb - n
            grad = torch.nn.functional.pad(grad, (0, 0, 0, pad))
        dx = torch_npu.npu_grouped_matmul(
            [grad],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=2,
            group_list_type=0,
        )[0]
        dx = dx[:n] if nb > n else dx
        dw = dw.reshape(weight.shape)
        dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
        dw = torch.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
        return dx, dw, None


def _grouped_matmul(x, weight, counts):
    if x.device.type == "npu":
        return _NpuGroupedMatmul.apply(x, weight, counts)
    return _grouped_matmul_torch(x, weight, counts)


def _fused_permute_dispatch_npu(
    x: torch.Tensor,
    indices: torch.Tensor,
    w_flat: torch.Tensor,
    experts: GroupedExperts,
    n_experts: int,
) -> torch.Tensor:
    import torch.nn.functional as F
    import torch_npu

    w1, w3, w2 = experts.local_weights()
    n, dim = x.shape
    k = indices.shape[1]
    nb = _bucket_count(n)
    if nb > n:
        pad = nb - n
        x = F.pad(x, (0, 0, 0, pad))
        indices = F.pad(indices, (0, 0, 0, pad))
        w_flat = F.pad(w_flat.reshape(n, k), (0, 0, 0, pad)).reshape(-1)

    if _MOE_PROF:
        import time as _time

        torch.npu.synchronize()
        _t0 = _time.perf_counter()
    idx = indices.to(torch.int32)
    routed_input, sorted_idx = torch_npu.npu_moe_token_permute(x, idx)
    routed_scores, _ = torch_npu.npu_moe_token_permute(
        w_flat.reshape(-1, 1), idx.reshape(-1, 1)
    )
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts)
    if _MOE_PROF:
        torch.npu.synchronize()
        _t1 = _time.perf_counter()
    out = swiglu_grouped(
        routed_input.bfloat16(),
        w1.bfloat16(),
        w3.bfloat16(),
        w2.bfloat16(),
        counts,
        routed_scores.reshape(-1).bfloat16(),
        experts.swiglu_limit,
        _grouped_matmul,
    )
    if _MOE_PROF:
        torch.npu.synchronize()
        _t2 = _time.perf_counter()
    unpermuted = torch_npu.npu_moe_token_unpermute(
        out.to(routed_input.dtype), sorted_idx, None
    )
    if _MOE_PROF:
        torch.npu.synchronize()
        _t3 = _time.perf_counter()
        _pm, _gm, _um = (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000
        if max(_pm, _gm, _um) > _MOE_PROF_MS:
            try:
                _r = torch.distributed.get_rank()
            except Exception:  # noqa: BLE001
                _r = 0
            print(
                f">>> [MOE-PROF r{_r}] n={n} nb={nb} k={k}  permute+bincount={_pm:.0f}ms  "
                f"experts_gemm={_gm:.0f}ms  unpermute={_um:.0f}ms",
                flush=True,
            )
    if nb > n:
        unpermuted = unpermuted.reshape(nb, k, dim)[:n].reshape(n * k, dim)
    return unpermuted
