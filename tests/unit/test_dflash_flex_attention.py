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

"""Kernel selection regressions shared by DSpark and DFlash training."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
flex = pytest.importorskip("verl_speco.models.dflash.flex_attention")


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("grad_input", [0, 1, 2])
def test_short_cuda_training_avoids_decode(monkeypatch, compiling, grad_input):
    inputs = [
        SimpleNamespace(
            device=SimpleNamespace(type="cuda"), requires_grad=i == grad_input
        )
        for i in range(3)
    ]

    def strict_attention(*args, **kwargs):
        if not kwargs.get("kernel_options", {}).get("FORCE_USE_FLEX_ATTENTION"):
            raise RuntimeError("flex_decoding: No choices exist for backend")
        assert kwargs["enable_gqa"] is True
        return "training output"

    monkeypatch.setattr(flex, "is_torchdynamo_compiling", lambda: compiling)
    monkeypatch.setattr(flex, "flex_attention", strict_attention)
    monkeypatch.setattr(flex, "WrappedFlexAttention", lambda: lambda: strict_attention)
    assert (
        flex.compile_friendly_flex_attention(*inputs, enable_gqa=True)
        == "training output"
    )


@pytest.mark.parametrize(
    "device,requires_grad,grad_enabled",
    [
        ("cuda", False, True),
        ("cuda", True, False),
        ("npu", True, True),
        ("cpu", True, True),
    ],
)
def test_inference_and_non_cuda_keep_existing_options(
    monkeypatch,
    device,
    requires_grad,
    grad_enabled,
):
    query = SimpleNamespace(
        device=SimpleNamespace(type=device), requires_grad=requires_grad
    )
    calls = []
    monkeypatch.setattr(flex, "is_torchdynamo_compiling", lambda: True)
    monkeypatch.setattr(
        flex, "flex_attention", lambda *args, **kwargs: calls.append(kwargs)
    )
    with torch.set_grad_enabled(grad_enabled):
        flex.compile_friendly_flex_attention(query, query, query, enable_gqa=True)
    assert calls == [{"enable_gqa": True}]


@pytest.mark.parametrize(
    "options",
    [
        {"BLOCK_M": 32},
        {"BACKEND": "TRITON"},
        {"FORCE_USE_FLEX_ATTENTION": False},
    ],
)
def test_kernel_options_are_copied_and_overrides_preserved(monkeypatch, options):
    query = SimpleNamespace(device=SimpleNamespace(type="cuda"), requires_grad=True)
    calls = []
    original = dict(options)
    monkeypatch.setattr(flex, "is_torchdynamo_compiling", lambda: True)
    monkeypatch.setattr(
        flex, "flex_attention", lambda *args, **kwargs: calls.append(kwargs)
    )
    flex.compile_friendly_flex_attention(query, query, query, kernel_options=options)
    expected = dict(original)
    if "BACKEND" not in expected:
        expected.setdefault("FORCE_USE_FLEX_ATTENTION", True)
    assert calls == [{"kernel_options": expected}]
    assert options == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_logged_short_gqa_shape_forward_and_backward():
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.functional import scaled_dot_product_attention

    torch.manual_seed(0)
    q = torch.randn(
        1, 32, 56, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        1, 8, 568, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)

    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= 512 + (q_idx // 7) * 7

    block_mask = create_block_mask(
        mask_mod, B=1, H=None, Q_LEN=56, KV_LEN=568, device="cuda"
    )
    output = flex.compile_friendly_flex_attention(
        q, k, v, block_mask=block_mask, enable_gqa=True
    )
    dense_mask = torch.arange(568, device="cuda")[None, :] <= (
        512 + (torch.arange(56, device="cuda")[:, None] // 7) * 7
    )
    reference = scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=dense_mask,
    )
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    gradients = torch.autograd.grad(output.float().square().mean(), (q, k, v))
    assert all(torch.isfinite(grad).all() for grad in gradients)
