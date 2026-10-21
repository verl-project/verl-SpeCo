# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional torch.compile / inductor path for the MoE experts.

Requires a matched torch-2.12 + torch_npu + inductor_npu_ext + triton-ascend
stack. Disabled by default; enabled through the grouped-GEMM switch.
"""

from __future__ import annotations

import os

import torch

_ENABLED = False


def _install_compile_shims() -> None:
    def _safe_cap(_f):
        def inner(*a, **k):
            try:
                c = _f(*a, **k)
            except Exception:  # noqa: BLE001
                c = None
            return c if c is not None else (0, 0)

        return inner

    for _name in ("cuda", "npu"):
        mod = getattr(torch, _name, None)
        gc = getattr(mod, "get_device_capability", None) if mod is not None else None
        if mod is not None and gc is not None:
            mod.get_device_capability = _safe_cap(gc)
    torch._dynamo.config.capture_scalar_outputs = True
    _lim = int(os.environ.get("DSPARK_COMPILE_CACHE_LIMIT", "1024"))
    torch._dynamo.config.cache_size_limit = _lim
    torch._dynamo.config.accumulated_cache_size_limit = _lim * 2


def _register_grouped_mm_npu() -> None:
    import torch_npu

    try:

        @torch.library.impl("aten::_grouped_mm", "PrivateUse1")
        def _(self, mat2, offs, bias=None, out_dtype=None):
            split_along_k = self.ndim == 2 and mat2.ndim == 2
            return torch_npu.npu_grouped_matmul(
                [self],
                [mat2],
                group_list=offs.to(dtype=torch.int64),
                group_list_type=0,
                split_item=2,
                group_type=(2 if split_along_k else 0),
                bias=[bias] if bias is not None else None,
                output_dtype=out_dtype,
            )[0]
    except RuntimeError:
        pass


def _expert_activation(h, swiglu_limit, scores):
    import torch_npu

    if swiglu_limit > 0:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)
    return h * scores.reshape(-1, 1).to(h.dtype)


def enable() -> None:
    global _ENABLED
    if _ENABLED:
        return
    _install_compile_shims()
    _register_grouped_mm_npu()
    _ENABLED = True


def run(w1, w3, w2, x, counts, swiglu_limit, scores):
    w13 = torch.cat([w1, w3], dim=1)
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offs)
    h = _expert_activation(h, swiglu_limit, scores)
    return torch._grouped_mm(
        h.bfloat16(), w2.bfloat16().transpose(-2, -1), offs=offs
    ).float()
