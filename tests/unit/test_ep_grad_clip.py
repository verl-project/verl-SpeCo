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
