# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Native route combination scoped to the split Ascend training boundary."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_NATIVE_COMBINE = ContextVar("speco_native_route_combine", default=False)


@contextmanager
def native_route_combine():
    token = _NATIVE_COMBINE.set(True)
    try:
        yield
    finally:
        _NATIVE_COMBINE.reset(token)


def native_route_combine_enabled():
    return _NATIVE_COMBINE.get()


def expert_matmul(x, weights, expert, counts=None, *, native=None):
    """Use the inference GMM primitive, including single-expert replay.

    Backward passes an explicit saved choice because its thread no longer has
    the forward ContextVar. Legacy calls and CPU references keep ordinary GEMM.
    """
    enabled = native_route_combine_enabled() if native is None else native
    if enabled and x.device.type == "npu":
        from verl_speco.ops.deepseek_v4_runtime import _npu_grouped_moe_matmul

        if counts is None:
            counts = torch.full((1,), x.shape[0], device=x.device, dtype=torch.int64)
        return _npu_grouped_moe_matmul(x, weights[expert : expert + 1], counts)
    return x.matmul(weights[expert])


def allocate_route_buffer(x, weights):
    """One unweighted output per route; no full expert activations are kept."""
    from verl_speco.ops.deepseek_v4_runtime import _record_max_training_diagnostic

    result = x.new_empty((weights.numel(), x.shape[-1]))
    _record_max_training_diagnostic(
        "native_route_combine_buffer_bytes", result.numel() * result.element_size()
    )
    return result


def combine_routes(values, sorted_route_ids, weights):
    """Use native FP32 multiply/accumulate and its final output conversion.

    ``sorted_route_ids[row]`` gives the token-major token/slot index of each
    unweighted row. CPU execution uses the same FP32 slot-order formula for
    reference tests; NPU execution requires the native operator to be available.
    """
    from verl_speco.ops.deepseek_v4_runtime import (
        _combine_grouped_routes_token_unpermute,
        _increment_training_diagnostic,
    )

    if (
        values.ndim != 2
        or weights.ndim != 2
        or sorted_route_ids.ndim != 1
        or values.shape[0] != weights.numel()
        or sorted_route_ids.numel() != weights.numel()
    ):
        raise ValueError("Native route combine layout differs")
    if weights.shape[0] == 0:
        return values.new_empty((0, values.shape[-1]))
    if values.device.type == "npu":
        result = _combine_grouped_routes_token_unpermute(
            values, sorted_route_ids, weights, force=True
        )
        if result is None:
            raise RuntimeError("Split Ascend training requires native token unpermute")
    elif values.device.type == "cpu":
        inverse = torch.empty_like(sorted_route_ids)
        inverse.index_copy_(
            0,
            sorted_route_ids,
            torch.arange(
                sorted_route_ids.numel(),
                device=sorted_route_ids.device,
                dtype=sorted_route_ids.dtype,
            ),
        )
        rows = (
            values.index_select(0, inverse)
            .reshape(*weights.shape, values.shape[-1])
            .float()
        )
        probs = weights.float()
        result = rows[:, 0] * probs[:, 0, None]
        for slot in range(1, weights.shape[1]):
            result = result + rows[:, slot] * probs[:, slot, None]
        result = result.to(values.dtype)
    else:
        raise RuntimeError(
            "Native route combine supports Ascend NPU or CPU reference only"
        )
    _increment_training_diagnostic("native_route_combine_forward_calls")
    return result
