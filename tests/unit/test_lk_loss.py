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

import pytest

torch = pytest.importorskip("torch")

from verl_speco.backends.lk_loss import (  # noqa: E402
    adaptive_diagnostics_to_rows,
    adaptive_hybrid_acceptance_loss,
    full_vocab_subset_probs,
    negative_log_acceptance_loss,
)


def test_negative_log_acceptance_loss_matches_probability_overlap():
    draft_probs = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float32)
    target_probs = torch.tensor([[0.45, 0.4, 0.15]], dtype=torch.float32)

    loss = negative_log_acceptance_loss(draft_probs, target_probs)

    expected_overlap = torch.minimum(draft_probs, target_probs).sum(dim=-1)
    assert torch.allclose(loss, -expected_overlap.log())


def test_negative_log_acceptance_loss_backpropagates():
    draft_logits = torch.randn(2, 8, requires_grad=True)
    target_probs = torch.softmax(torch.randn(2, 8), dim=-1)

    loss = negative_log_acceptance_loss(
        torch.softmax(draft_logits, dim=-1), target_probs
    ).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert draft_logits.grad is not None
    assert torch.isfinite(draft_logits.grad).all()


def test_full_vocab_subset_probs_uses_actor_log_partition():
    subset_logits = torch.tensor([[[2.0, 1.0], [0.5, -0.5]]])
    outside_logits = torch.tensor([[[3.0], [1.5]]])
    full_logits = torch.cat([subset_logits, outside_logits], dim=-1)
    temperature = 0.5
    target_logz = torch.logsumexp(full_logits / temperature, dim=-1)

    actual = full_vocab_subset_probs(
        subset_logits,
        target_logz,
        temperature=temperature,
    )
    expected = torch.softmax(full_logits / temperature, dim=-1)[..., :2]

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual.sum(dim=-1) < 1.0)


def test_full_vocab_subset_probs_rejects_inconsistent_log_partition():
    subset_logits = torch.tensor([[[2.0, 1.0]]])
    too_small_logz = torch.tensor([[1.0]])

    with pytest.raises(ValueError, match="smaller than a selected logit"):
        full_vocab_subset_probs(subset_logits, too_small_logz, temperature=1.0)


def test_adaptive_diagnostics_are_transferred_to_cpu_once(monkeypatch):
    diagnostics = [
        {
            "acceptance": torch.tensor(0.5),
            "kl_weight": torch.tensor(0.6),
            "forward_kl": torch.tensor(0.7),
            "tv": torch.tensor(0.8),
        },
        {
            "acceptance": torch.tensor(0.4),
            "kl_weight": torch.tensor(0.5),
            "forward_kl": torch.tensor(0.6),
            "tv": torch.tensor(0.7),
        },
    ]
    cpu_calls = 0
    original_cpu = torch.Tensor.cpu

    def counted_cpu(self, *args, **kwargs):
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu, raising=True)

    rows = adaptive_diagnostics_to_rows(diagnostics)

    assert cpu_calls == 1
    torch.testing.assert_close(
        torch.tensor(rows),
        torch.tensor([[0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7]]),
    )


def test_adaptive_hybrid_acceptance_loss_matches_kl_tv_formula():
    draft_probs = torch.tensor(
        [[0.50, 0.30, 0.20], [0.20, 0.30, 0.50]], dtype=torch.float32
    )
    target_acceptance_probs = torch.tensor(
        [[0.40, 0.35, 0.15], [0.10, 0.25, 0.45]], dtype=torch.float32
    )
    target_kl_probs = target_acceptance_probs / target_acceptance_probs.sum(
        dim=-1, keepdim=True
    )
    position_mask = torch.tensor([True, True])
    eta = 3.0

    loss, diagnostics = adaptive_hybrid_acceptance_loss(
        draft_probs=draft_probs,
        target_acceptance_probs=target_acceptance_probs,
        target_kl_probs=target_kl_probs,
        position_mask=position_mask,
        eta=eta,
    )

    acceptance = torch.minimum(draft_probs, target_acceptance_probs).sum(dim=-1)
    expected_weight = torch.exp(-eta * acceptance.mean())
    expected_kl = (target_kl_probs * (target_kl_probs.log() - draft_probs.log())).sum(
        dim=-1
    )
    expected = expected_weight * expected_kl + (1 - expected_weight) * (1 - acceptance)

    assert torch.allclose(loss, expected)
    assert torch.allclose(diagnostics["acceptance"], acceptance.mean())
    assert torch.allclose(diagnostics["kl_weight"], expected_weight)


def test_adaptive_hybrid_weight_is_stop_gradient_and_loss_backpropagates():
    draft_logits = torch.randn(3, 8, requires_grad=True)
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_acceptance_probs = torch.softmax(torch.randn(3, 8), dim=-1)

    loss, diagnostics = adaptive_hybrid_acceptance_loss(
        draft_probs=draft_probs,
        target_acceptance_probs=target_acceptance_probs,
        target_kl_probs=target_acceptance_probs,
        position_mask=torch.tensor([True, True, True]),
        eta=3.0,
    )
    loss.mean().backward()

    assert diagnostics["kl_weight"].requires_grad is False
    assert draft_logits.grad is not None
    assert torch.isfinite(draft_logits.grad).all()


def test_adaptive_hybrid_empty_mask_stays_finite_and_kl_dominated():
    draft_probs = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    target_probs = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    loss, diagnostics = adaptive_hybrid_acceptance_loss(
        draft_probs=draft_probs,
        target_acceptance_probs=target_probs,
        target_kl_probs=target_probs,
        position_mask=torch.tensor([False]),
        eta=3.0,
    )

    assert torch.isfinite(loss).all()
    assert diagnostics["acceptance"].item() == pytest.approx(0.0)
    assert diagnostics["kl_weight"].item() == pytest.approx(1.0)
