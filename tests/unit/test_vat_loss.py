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
import copy

import pytest
import torch
from torch.nn import functional as F

from verl_speco.backends.verification_aware import VATObjective


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_vat_matches_dense_reference_and_gradients(layout, accepted):
    torch.manual_seed(8)
    offset = int(layout == "dflash")
    mask = torch.ones(1, 1, 3 + offset)
    mask[..., :offset] = 0
    active = mask.reshape(-1).bool()
    target = torch.eye(3)
    weight = torch.eye(3)
    hidden = torch.randn(3, 3, requires_grad=True)
    ids = torch.arange(3)
    ids[accepted:] = (ids[accepted:] + 1) % 3
    logits = F.one_hot(ids, 3).float().requires_grad_()
    conf = torch.randn(3, requires_grad=True)
    objective = VATObjective(
        3,
        layout=layout,
        gamma=2.0,
        loss_mode="full_vocab",
        config={"kd_weight": 1.0, "self_cov_weight": 0.6, "chunk_size": 1},
    )
    reference = copy.deepcopy(objective)
    h2 = hidden.detach().clone().requires_grad_()
    l2 = logits.detach().clone().requires_grad_()
    c2 = conf.detach().clone().requires_grad_()
    weights, extra, metrics = objective(
        logits, hidden, target, weight, mask, active, torch.arange(3), conf
    )
    j = torch.arange(3)
    expected_w = torch.exp(-(j - accepted).clamp_min(0) / 2.0)
    torch.testing.assert_close(weights.reshape(-1)[active], expected_w)
    assert metrics["vat/mean_accepted"].item() == accepted
    kd = (
        F.kl_div(
            F.log_softmax(l2 / 2.0, -1), F.softmax(target / 2.0, -1), reduction="none"
        ).sum(-1)
        * 4
    )
    kd = (kd * expected_w).sum() / expected_w.sum()
    if layout == "dflash":
        bce = F.binary_cross_entropy_with_logits(
            reference.accept_head(h2).squeeze(-1), (j < accepted).float()
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            c2, (j < accepted).float(), reduction="none"
        )
        bce = (bce * (j <= accepted)).sum() / (j <= accepted).sum()
    prefix = (
        F.log_softmax(l2, -1).diagonal().cumsum(-1).clamp_min(torch.tensor(1e-6).log())
    )
    cov = -(prefix * (j >= accepted)).sum() / 3
    expected = kd + bce + 0.6 * cov
    torch.testing.assert_close(extra, expected)
    extra.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, l2.grad)
    if layout == "dflash":
        torch.testing.assert_close(hidden.grad, h2.grad)
        for a, b in zip(objective.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
    else:
        torch.testing.assert_close(conf.grad, c2.grad)


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_empty_active_rows_have_zero_head_gradients(layout):
    obj = VATObjective(2, layout=layout, gamma=7.0, config={}, loss_mode="full_vocab")
    hidden = torch.empty(0, 2, requires_grad=True)
    conf_head = torch.nn.Linear(2, 1)
    mask = torch.zeros(1, 1, 4)
    _, loss, _ = obj(
        hidden,
        hidden,
        hidden,
        torch.eye(2),
        mask,
        mask.reshape(-1).bool(),
        torch.empty(0, dtype=torch.long),
        conf_head(hidden).squeeze(-1),
    )
    loss.backward()
    head = obj.accept_head if layout == "dflash" else conf_head
    assert head.weight.grad is not None
    assert head.weight.grad.count_nonzero() == 0
    assert loss.item() == 0


def test_invalid_row_ends_acceptance_chain_and_has_no_weight():
    obj = VATObjective(2, layout="dspark", gamma=1.0, config={}, loss_mode="full_vocab")
    mask = torch.tensor([[[1.0, 0.0, 1.0]]])
    active = mask.reshape(-1).bool()
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]], requires_grad=True)
    weights, _, metrics = obj(
        logits,
        logits,
        logits.detach(),
        torch.eye(2),
        mask,
        active,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
    )
    torch.testing.assert_close(
        weights, torch.tensor([[[1.0, 0.0, torch.exp(torch.tensor(-1.0)).item()]]])
    )
    assert metrics["vat/mean_accepted"].item() == 1


@pytest.mark.parametrize(
    "config",
    [
        {"accept_mode": "probabilistic"},
        {"chunk_size": 0},
        {"kd_temperature": 0},
        {"kd_weight": -1},
    ],
)
def test_invalid_user_config_fails(config):
    with pytest.raises(ValueError):
        VATObjective(
            2, layout="dflash", gamma=7.0, config=config, loss_mode="full_vocab"
        )


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
def test_vat_requests_final_target_hidden(algorithm):
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
    )

    config = {
        f"{algorithm.lower()}_per_position_loss_weight": "vat",
        "dspark_l1_loss_alpha": 0.0,
        "dflash_lk_loss_alpha": 0.0,
    }
    assert (
        resolve_drafter_hidden_states_layout(algorithm, config)
        == "dflash_aux_plus_last"
    )


def test_self_coverage_uses_data_labels_not_teacher_argmax():
    obj = VATObjective(
        2,
        layout="dspark",
        gamma=7.0,
        loss_mode="full_vocab",
        config={"self_cov_weight": 1.0, "kd_weight": 0.0, "accept_head_weight": 0.0},
    )
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    mask = torch.ones(1, 1, 2)
    _, loss, _ = obj(
        logits,
        logits,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.eye(2),
        mask,
        mask.reshape(-1).bool(),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2),
    )
    expected = -F.log_softmax(logits, -1)[:, 1].cumsum(-1).mean()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_fp16_vat_without_coverage_is_finite(layout):
    obj = VATObjective(2, layout=layout, gamma=7., config={}, loss_mode="full_vocab")
    logits = torch.full((1, 1024), -100., dtype=torch.float16, requires_grad=True)
    hidden = torch.zeros(1, 2, requires_grad=True)
    mask = torch.ones(1, 1, 1 + int(layout == "dflash"))
    if layout == "dflash":
        mask[..., 0] = 0
    _, loss, _ = obj(logits, hidden, torch.zeros(1, 2), torch.zeros(1024, 2),
                     mask, mask.flatten().bool(), torch.zeros(1, dtype=torch.long),
                     hidden[:, 0])
    assert torch.isfinite(loss)
    loss.backward()
    if logits.grad is not None:
        assert torch.isfinite(logits.grad).all()


def test_zero_kd_skips_kl_and_preserves_acceptance(monkeypatch):
    logits = torch.tensor([[2., 0.], [0., 2.], [2., 0.]], requires_grad=True)
    mask = torch.ones(1, 1, 3)
    inputs = (logits, logits, logits.detach(), torch.eye(2), mask,
              mask.flatten().bool(), torch.zeros(3, dtype=torch.long), torch.zeros(3))
    enabled = VATObjective(2, layout="dspark", gamma=1., config={"kd_weight": 1.}, loss_mode="full_vocab")
    disabled = VATObjective(2, layout="dspark", gamma=1., config={}, loss_mode="full_vocab")
    expected_weights, _, expected_metrics = enabled(*inputs)
    def forbidden(*args, **kwargs):
        raise AssertionError("KL evaluated with zero KD weight")
    monkeypatch.setattr(F, "kl_div", forbidden)
    monkeypatch.setattr(F, "softmax", forbidden)
    monkeypatch.setattr(F, "log_softmax", forbidden)
    weights, loss, metrics = disabled(*inputs)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(metrics["vat/mean_accepted"], expected_metrics["vat/mean_accepted"])
    assert metrics["vat/kd_loss"] == 0
    loss.backward()
