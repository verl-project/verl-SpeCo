# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU tests for the EP-aware gradient clipping helper."""

from __future__ import annotations

import torch

from verl_speco.trainer.base_trainer import _clip_dtensor_grad_norm_


def _grad_norm(params) -> torch.Tensor:
    return torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params))


def test_ep_clip_matches_plain_clip_without_dist():
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3)),
    ]
    for param in params:
        param.grad = torch.randn_like(param) * 20.0

    expected_norm = _grad_norm(params)
    returned_norm = _clip_dtensor_grad_norm_(params, max_norm=1.0)

    torch.testing.assert_close(returned_norm, expected_norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(_grad_norm(params), torch.tensor(1.0), atol=1e-4, rtol=1e-4)


def test_ep_clip_no_grads_returns_zero():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4))  # no .grad
    norm = _clip_dtensor_grad_norm_([param], max_norm=1.0)
    assert norm.item() == 0.0


def test_ep_clip_leaves_small_norm_untouched():
    param = torch.nn.Parameter(torch.randn(4))
    param.grad = torch.randn_like(param) * 1e-3
    before = param.grad.clone()
    _clip_dtensor_grad_norm_([param], max_norm=1.0)
    torch.testing.assert_close(param.grad, before)


def test_ep_ddp_clip_counts_replicated_params_once():
    # EP+DDP: sync_non_expert_grads has already world-averaged the replicated
    # non-expert grads, so their squared sum must be counted once (divided by
    # the world size), while the rank-disjoint expert grads count in full.
    torch.manual_seed(0)
    replicated = torch.nn.Parameter(torch.randn(6))
    replicated.grad = torch.randn_like(replicated) * 3.0
    expert = torch.nn.Parameter(torch.randn(4))
    expert.grad = torch.randn_like(expert) * 3.0

    ne_sq = replicated.grad.float().pow(2).sum()
    ex_sq = expert.grad.float().pow(2).sum()
    world = 4

    norm = _clip_dtensor_grad_norm_(
        [replicated, expert],
        max_norm=100.0,  # no clipping: inspect the returned norm only
        disjoint_plain_param_ids={id(expert)},
        replicated_world_size=world,
    )
    expected = torch.sqrt(ne_sq / world + ex_sq)
    torch.testing.assert_close(norm, expected, atol=1e-5, rtol=1e-5)


def test_ep_ddp_clip_treats_disjoint_plain_params_in_full():
    torch.manual_seed(0)
    expert_a = torch.nn.Parameter(torch.randn(4))
    expert_a.grad = torch.randn_like(expert_a) * 5.0
    expert_b = torch.nn.Parameter(torch.randn(4))
    expert_b.grad = torch.randn_like(expert_b) * 5.0

    expected = torch.sqrt(
        expert_a.grad.float().pow(2).sum() + expert_b.grad.float().pow(2).sum()
    )
    norm = _clip_dtensor_grad_norm_(
        [expert_a, expert_b],
        max_norm=100.0,
        disjoint_plain_param_ids={id(expert_a), id(expert_b)},
        replicated_world_size=4,
    )
    torch.testing.assert_close(norm, expected, atol=1e-5, rtol=1e-5)
