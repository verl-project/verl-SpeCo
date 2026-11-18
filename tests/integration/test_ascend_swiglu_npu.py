# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in tiny NPU parity check; no model weights or distributed workers.

SPECO_RUN_NPU_TESTS=1 python3 -m pytest -q -s tests/integration/test_ascend_swiglu_npu.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECO_RUN_NPU_TESTS") != "1",
    reason="Set SPECO_RUN_NPU_TESTS=1 to explicitly enable NPU execution",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("limit", [0.7, 0.0])
def test_training_swiglu_matches_native_serving_and_gradients(
    monkeypatch, dtype, limit
):
    import torch_npu  # noqa: F401 -- register the installed native NPU operators
    from vllm_ascend.ops.activation import (
        AscendSiluAndMul,
        AscendSiluAndMulWithClamp,
    )

    from verl_speco.ops.deepseek_v4_runtime import _routed_moe_swiglu

    monkeypatch.setenv("MTP_DS4_FUSED_ROUTED_MOE_SWIGLU", "1")
    torch.npu.set_device(0)
    source = torch.linspace(-2, 2, 512).reshape(16, 32).to(dtype)
    train_input = source.to("npu").requires_grad_(True)
    native_input = source.to("npu").requires_grad_(True)
    before = train_input.detach().clone()

    def serving(value):
        # Run the installed implementation without constructing a vLLM model.
        if limit > 0:
            return AscendSiluAndMulWithClamp.forward_oot(
                SimpleNamespace(swiglu_limit=limit), value
            )
        return AscendSiluAndMul.forward_oot(None, value)

    actual = _routed_moe_swiglu(train_input, limit)
    native = serving(native_input)
    with torch.no_grad():
        inference = serving(before)
    torch.testing.assert_close(actual, inference, rtol=0, atol=0)
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    torch.testing.assert_close(train_input.detach(), before, rtol=0, atol=0)
    upstream = (
        torch.linspace(-1, 1, actual.numel()).reshape_as(actual).to(dtype).to("npu")
    )
    actual.backward(upstream)
    native.backward(upstream)
    torch.testing.assert_close(train_input.grad, native_input.grad, rtol=0, atol=0)
    assert torch.isfinite(train_input.grad).all().item()

    # Independent FP32 derivative checks the native backward, including clipping.
    reference_input = source.float().requires_grad_(True)
    gate, up = reference_input.chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    reference = F.silu(gate) * up
    reference.backward(upstream.cpu().float())
    rtol, atol = (0.02, 0.01) if dtype == torch.bfloat16 else (2e-5, 2e-6)
    torch.testing.assert_close(
        actual.detach().cpu().float(), reference.detach(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        train_input.grad.cpu().float(), reference_input.grad, rtol=rtol, atol=atol
    )
    print(
        f"PASS dtype={dtype} limit={limit}: native forward/gradient exact match, FP32 reference passed"
    )
