# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright contributors to the vllm-project/speculators project
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
"""Memory-efficient DSpark CE and TV losses.

The online-softmax and TV forward/backward are adapted from
``vllm-project/speculators`` (Apache-2.0),
``src/speculators/losses/fused.py``.  SpeCo's CE variant consumes the observed
token label directly instead of taking ``argmax`` over target logits.
"""

# Triton kernels conventionally use uppercase constexpr names.
# ruff: noqa: N803, N806, PLR2004
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only development and eager fallback.
    triton = None
    tl = None


FusedLossBackend = Literal["cuda", "ascend"]
MAX_FUSED_SIZE_NPU = 4096
_N_STATS = 5


@dataclass(frozen=True)
class FusedLossCapability:
    available: bool
    backend: FusedLossBackend | None
    reason: str | None = None


def fused_loss_capability(device: torch.device) -> FusedLossCapability:
    device = torch.device(device)
    if device.type not in {"cuda", "npu"}:
        return FusedLossCapability(
            False, None, f"unsupported device type {device.type!r}"
        )
    backend: FusedLossBackend = "ascend" if device.type == "npu" else "cuda"
    if triton is None:
        return FusedLossCapability(False, backend, "Triton is not importable")
    if device.type == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            return FusedLossCapability(
                False, backend, f"Ascend runtime unavailable: {exc}"
            )
        try:
            if importlib.util.find_spec("triton.backends.ascend") is None:
                return FusedLossCapability(
                    False, backend, "triton.backends.ascend is unavailable"
                )
        except (ImportError, ValueError) as exc:
            return FusedLossCapability(
                False, backend, f"Ascend backend check failed: {exc}"
            )
    try:
        target_backend = triton.runtime.driver.active.get_current_target().backend
    except Exception as exc:  # noqa: BLE001
        return FusedLossCapability(False, backend, f"Triton target unavailable: {exc}")
    expected_backends = {"npu"} if device.type == "npu" else {"cuda", "hip"}
    if target_backend not in expected_backends:
        return FusedLossCapability(
            False,
            backend,
            f"Triton target {target_backend!r} does not match {device.type!r}",
        )
    return FusedLossCapability(True, backend)


def _validate_logits(logits: torch.Tensor, name: str) -> None:
    if logits.ndim != 2:
        raise ValueError(
            f"{name} must have shape [tokens, vocab], got {tuple(logits.shape)}"
        )
    if logits.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"{name} must be fp16, bf16, or fp32, got {logits.dtype}")
    capability = fused_loss_capability(logits.device)
    if not capability.available:
        raise RuntimeError(f"DSpark fused loss is unavailable: {capability.reason}")


if triton is not None:
    _OP_LABEL_CE = tl.constexpr(0)
    _OP_TV = tl.constexpr(1)

    def _is_npu_available() -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        npu = getattr(torch, "npu", None)
        return npu is not None and bool(npu.is_available())

    def _tile_configs():
        if _is_npu_available():
            # Ascend UB overflows above 4096 for the two-row TV kernel.
            return [triton.Config({"BLOCK_SIZE": MAX_FUSED_SIZE_NPU}, num_warps=4)]
        block_sizes = (512, 1024, 2048, 4096, 8192, 16384, 32768)
        threads = (128, 256, 512, 1024)
        warp_size = 64 if getattr(torch.version, "hip", None) is not None else 32
        return [
            triton.Config({"BLOCK_SIZE": block}, num_warps=n_threads // warp_size)
            for n_threads in threads
            for block in block_sizes
            if 4 <= block // n_threads <= 128
        ]

    def _prune_oversized_tiles(configs, nargs, **_):
        cap = triton.next_power_of_2(nargs["n_cols"])
        keep = [config for config in configs if config.kwargs["BLOCK_SIZE"] <= cap]
        if keep:
            return keep
        floor = min(config.kwargs["BLOCK_SIZE"] for config in configs)
        return [config for config in configs if config.kwargs["BLOCK_SIZE"] == floor]

    _autotune_tile = triton.autotune(
        configs=_tile_configs(),
        key=["n_cols", "OP"],
        prune_configs_by={"early_config_prune": _prune_oversized_tiles},
    )

    @triton.jit
    def _online_stats(row_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        m = float("-inf")
        d = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            block_max = tl.max(tl.where(mask, x, float("-inf")))
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
            m = m_new
        return m, d

    @_autotune_tile
    @triton.jit
    def _loss_forward_kernel(
        logits_ptr,
        targets_ptr,
        labels_ptr,
        loss_ptr,
        stats_ptr,
        stats_row,
        n_cols,
        OP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        logits_ptr += pid * n_cols
        m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
        lse_d = m_d + tl.log(z_d)
        tl.store(stats_ptr + pid, m_d)
        tl.store(stats_ptr + stats_row + pid, z_d)

        if OP == _OP_LABEL_CE:
            label = tl.load(labels_ptr + pid).to(tl.int64)
            target_logit = tl.load(logits_ptr + label).cast(tl.float32)
            tl.store(loss_ptr + pid, lse_d - target_logit)
            tl.store(stats_ptr + 4 * stats_row + pid, label.to(tl.float32))
        else:
            targets_ptr += pid * n_cols
            m_t, z_t = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)
            lse_t = m_t + tl.log(z_t)
            tl.store(stats_ptr + 2 * stats_row + pid, m_t)
            tl.store(stats_ptr + 3 * stats_row + pid, z_t)
            overlap = 0.0
            draft_small = 0.0
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
                y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(
                    tl.float32
                )
                dp = tl.exp(x - lse_d)
                tp = tl.exp(y - lse_t)
                overlap += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
                draft_small += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))
            tl.store(loss_ptr + pid, 1.0 - overlap)
            tl.store(stats_ptr + 4 * stats_row + pid, draft_small)

    @_autotune_tile
    @triton.jit
    def _loss_backward_kernel(
        logits_ptr,
        targets_ptr,
        grad_in_ptr,
        grad_out_ptr,
        stats_ptr,
        stats_row,
        n_cols,
        OP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        logits_ptr += pid * n_cols
        grad_in_ptr += pid * n_cols
        go = tl.load(grad_out_ptr + pid).cast(tl.float32)
        if go == 0.0:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
            return
        m_d = tl.load(stats_ptr + pid)
        z_d = tl.load(stats_ptr + stats_row + pid)
        lse_d = m_d + tl.log(z_d)
        extra = tl.load(stats_ptr + 4 * stats_row + pid)
        if OP == _OP_TV:
            targets_ptr += pid * n_cols
            m_t = tl.load(stats_ptr + 2 * stats_row + pid)
            z_t = tl.load(stats_ptr + 3 * stats_row + pid)
            lse_t = m_t + tl.log(z_t)
        else:
            target_idx = extra.to(tl.int32)

        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            dp = tl.exp(x - lse_d)
            if OP == _OP_LABEL_CE:
                grad = dp - (offsets == target_idx).to(tl.float32)
            else:
                y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(
                    tl.float32
                )
                tp = tl.exp(y - lse_t)
                grad = -dp * ((dp <= tp).to(tl.float32) - extra)
            tl.store(grad_in_ptr + offsets, go * grad, mask=mask)

    class _FusedLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, targets, labels, op):
            rows, vocab = logits.shape
            logits_flat = logits.contiguous()
            targets_flat = targets.contiguous() if targets is not None else None
            labels_flat = labels.contiguous() if labels is not None else None
            loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
            stats = torch.empty(
                _N_STATS, rows, device=logits.device, dtype=torch.float32
            )
            _loss_forward_kernel[(rows,)](
                logits_flat,
                targets_flat,
                labels_flat,
                loss,
                stats,
                stats.stride(0),
                vocab,
                OP=op,
            )
            ctx.save_for_backward(logits_flat, targets_flat, stats)
            ctx.op = op
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            logits, targets, stats = ctx.saved_tensors
            rows, vocab = logits.shape
            grad_logits = torch.empty_like(logits)
            _loss_backward_kernel[(rows,)](
                logits,
                targets,
                grad_logits,
                grad_output.contiguous(),
                stats,
                stats.stride(0),
                vocab,
                OP=ctx.op,
            )
            return grad_logits, None, None, None


def fused_label_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    _validate_logits(logits, "logits")
    if labels.shape != logits.shape[:1] or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [tokens]")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")
    if bool(((labels < 0) | (labels >= logits.size(1))).any()):
        raise ValueError("labels contain an index outside the vocabulary")
    return _FusedLoss.apply(logits, None, labels, _OP_LABEL_CE.value)


def fused_total_variation(
    draft_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    _validate_logits(draft_logits, "draft_logits")
    _validate_logits(target_logits, "target_logits")
    if draft_logits.shape != target_logits.shape:
        raise ValueError("draft_logits and target_logits must have identical shapes")
    if draft_logits.device != target_logits.device:
        raise ValueError("draft_logits and target_logits must be on the same device")
    return _FusedLoss.apply(draft_logits, target_logits.detach(), None, _OP_TV.value)
