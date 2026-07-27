from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

base_trainer_mod = pytest.importorskip("verl_speco.trainer.base_trainer")

DrafterBaseTrainer = base_trainer_mod.DrafterBaseTrainer


def _bare_trainer(sp_size=1, dp_size=1, sp_group=None, dp_group=None):
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.training_device_mesh = None
    trainer._get_sp_group = lambda: sp_group
    trainer._get_dp_group = lambda: dp_group
    trainer._get_sp_world_size = lambda: sp_size
    trainer._get_dp_world_size = lambda: dp_size
    return trainer


def test_reduce_loss_metrics_single_rank_passthrough():
    trainer = _bare_trainer()
    l_v = torch.tensor(2.0, requires_grad=True) * 1.0
    l_p = torch.tensor(3.0, requires_grad=True) * 1.0
    l_n = torch.tensor(4.0)

    global_vloss, global_ploss, global_tokens, world_size = trainer._reduce_loss_metrics(l_v, l_p, l_n)

    assert world_size == 1
    assert not global_vloss.requires_grad
    assert not global_ploss.requires_grad
    assert torch.equal(global_vloss, torch.tensor(2.0))
    assert torch.equal(global_ploss, torch.tensor(3.0))
    assert torch.equal(global_tokens, torch.tensor(4.0))


def test_reduce_loss_metrics_world_size_is_sp_times_dp(monkeypatch):
    calls = []
    monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", lambda tensor, group=None: calls.append(group))

    trainer = _bare_trainer(sp_size=4, dp_size=2, sp_group="sp", dp_group="dp")
    zeros = torch.zeros(())
    _, _, _, world_size = trainer._reduce_loss_metrics(zeros, zeros, zeros)

    assert world_size == 8
    assert calls == ["sp", "dp"]


def test_dp_local_loss_scaling_recovers_global_token_mean_gradient(monkeypatch):
    """Emulate 2 DP ranks: per-rank scaled local losses, backward, then an
    FSDP-style gradient mean must equal the global token-mean gradient."""
    torch.manual_seed(0)
    rank_inputs = [torch.randn(3, 4), torch.randn(5, 4)]
    total_tokens = sum(x.shape[0] for x in rank_inputs)

    def token_sum_loss(weight, inputs):
        return ((inputs @ weight) ** 2).sum()

    weight_ref = torch.randn(4, requires_grad=True)
    reference = (token_sum_loss(weight_ref, rank_inputs[0]) + token_sum_loss(weight_ref, rank_inputs[1])) / float(
        total_tokens
    )
    reference.backward()

    per_rank_grads = []
    for rank in (0, 1):
        weight = weight_ref.detach().clone().requires_grad_(True)
        l_v = torch.zeros(())
        l_p = token_sum_loss(weight, rank_inputs[rank])
        l_n = torch.tensor(float(rank_inputs[rank].shape[0]))

        other = 1 - rank
        other_metrics = torch.stack(
            [
                torch.zeros(()),
                token_sum_loss(weight_ref.detach(), rank_inputs[other]).detach(),
                torch.tensor(float(rank_inputs[other].shape[0])),
            ]
        )

        def fake_all_reduce(tensor, group=None, _other=other_metrics):
            tensor += _other

        monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", fake_all_reduce)
        trainer = _bare_trainer(dp_size=2, dp_group="dp")

        _, global_ploss, global_tokens, world_size = trainer._reduce_loss_metrics(l_v, l_p, l_n)
        assert world_size == 2
        assert torch.allclose(global_tokens, torch.tensor(float(total_tokens)))

        denom = global_tokens.clamp(min=1.0)
        local_loss = (0.0 * l_v + 1.0 * l_p) * (float(world_size) / denom)
        local_loss.backward()
        per_rank_grads.append(weight.grad)

    fsdp_mean_grad = (per_rank_grads[0] + per_rank_grads[1]) / 2.0
    assert torch.allclose(fsdp_mean_grad, weight_ref.grad, atol=1e-6)
