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
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl_speco.ops.dspark_fused_loss import (
    fused_label_cross_entropy,
    fused_loss_capability,
    fused_total_variation,
)


def _accelerator_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("CUDA and Ascend NPU are unavailable")
    if torch.npu.is_available():
        return torch.device("npu")
    pytest.skip("CUDA and Ascend NPU are unavailable")


def test_fused_label_ce_matches_eager_forward_and_backward() -> None:
    device = _accelerator_device()
    capability = fused_loss_capability(device)
    if not capability.available:
        pytest.skip(capability.reason)
    torch.manual_seed(7)
    labels = torch.tensor([0, 17, 256], device=device, dtype=torch.long)
    eager_logits = torch.randn(3, 257, device=device, requires_grad=True)
    fused_logits = eager_logits.detach().clone().requires_grad_(True)

    eager = torch.nn.functional.cross_entropy(
        eager_logits.float(), labels, reduction="none"
    )
    fused = fused_label_cross_entropy(fused_logits, labels)
    grad = torch.tensor([0.25, 1.0, -0.5], device=device)
    eager.backward(grad)
    fused.backward(grad)

    torch.testing.assert_close(fused, eager, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        fused_logits.grad, eager_logits.grad, atol=2e-5, rtol=2e-5
    )


def test_fused_tv_matches_eager_forward_and_draft_backward() -> None:
    device = _accelerator_device()
    capability = fused_loss_capability(device)
    if not capability.available:
        pytest.skip(capability.reason)
    torch.manual_seed(11)
    eager_draft = torch.randn(3, 257, device=device, requires_grad=True)
    fused_draft = eager_draft.detach().clone().requires_grad_(True)
    target = torch.randn(3, 257, device=device, requires_grad=True)

    eager_probs = torch.softmax(eager_draft.float(), dim=-1)
    target_probs = torch.softmax(target.detach().float(), dim=-1)
    eager = 0.5 * (eager_probs - target_probs).abs().sum(dim=-1)
    fused = fused_total_variation(fused_draft, target)
    grad = torch.tensor([0.25, 1.0, -0.5], device=device)
    eager.backward(grad)
    fused.backward(grad)

    torch.testing.assert_close(fused, eager, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        fused_draft.grad, eager_draft.grad, atol=3e-5, rtol=3e-5
    )
    assert target.grad is None
