"""Unit temperature must avoid copies without changing either OPD objective."""

import pytest
import torch

from verl_speco.backends import opd_distribution as distribution
from verl_speco.backends import specopd_loss


def legacy_tv(draft_logits, target_logits, *, proposal_temperature=1.0,
              target_temperature=1.0, implementation="eager"):
    draft = draft_logits.float() / proposal_temperature
    target = target_logits.detach().float() / target_temperature
    return (draft.softmax(-1) - target.softmax(-1)).abs().sum(-1) * 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperatures", [(1.0, 1.0), (0.6, 1.0), (1.0, 1.7), (0.6, 1.7)])
@pytest.mark.parametrize("implementation", ["eager", "fused"])
def test_unit_temperature_preserves_outputs_gradients_and_teacher_stop(monkeypatch, dtype, temperatures, implementation):
    torch.manual_seed(23)
    draft = torch.randn(2, 3, 17).to(dtype).requires_grad_()
    target = torch.randn(2, 3, 17).to(dtype).requires_grad_()
    pointers = []
    monkeypatch.setattr(distribution, "_fused_available", lambda device: True)
    def fused(x, y, *, average_ties):
        assert average_ties is True and not y.requires_grad
        pointers.append((x.data_ptr(), y.data_ptr()))
        return (x.softmax(-1) - y.softmax(-1)).abs().sum(-1) * 0.5
    monkeypatch.setattr(distribution, "fused_total_variation", fused)
    kwargs = dict(proposal_temperature=temperatures[0], target_temperature=temperatures[1],
                  implementation=implementation)
    actual = distribution.token_total_variation(draft, target, **kwargs)
    expected = legacy_tv(draft, target, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual.sum(), draft, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), draft)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert torch.autograd.grad(actual.sum(), target, allow_unused=True)[0] is None
    if implementation == "fused" and dtype == torch.float32:
        # Fused dispatch sees the original FP32 storage only on the unit side.
        assert (pointers[0][0] == draft.data_ptr()) == (temperatures[0] == 1.0)
        assert (pointers[0][1] == target.data_ptr()) == (temperatures[1] == 1.0)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("temperatures", [(1.0, 1.0), (0.6, 1.7)])
def test_prefix_masks_and_weighted_anchor_loss_remain_exact(monkeypatch, empty, temperatures):
    torch.manual_seed(41)
    draft = torch.randn(3, 7, 29, requires_grad=True)
    target = torch.randn_like(draft, requires_grad=True)
    mask = torch.arange(7)[None] < torch.tensor([[7], [3], [0]])
    if empty:
        mask.zero_()
    weights = torch.ones(3, 7)
    weights[0, 2] = 2
    weights[1] = torch.exp(-torch.arange(7).float() / 7)
    kwargs = dict(position_weights=weights, distribution_loss_impl="eager",
                  proposal_temperature=temperatures[0], target_temperature=temperatures[1])
    actual, count, diagnostics = specopd_loss.e2e_tv_loss(draft, target, mask, **kwargs)
    monkeypatch.setattr(specopd_loss, "token_total_variation", legacy_tv)
    expected, expected_count, expected_diagnostics = specopd_loss.e2e_tv_loss(draft, target, mask, **kwargs)
    assert count == expected_count == (0 if empty else 2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key in diagnostics:
        torch.testing.assert_close(diagnostics[key], expected_diagnostics[key], atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual, draft, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, draft)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert not actual_grad[~mask].any()
    assert torch.autograd.grad(actual, target, allow_unused=True)[0] is None
