# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Trace cache invalidation/isolation and shared TV value-gradient contracts."""

import pytest
import torch

from verl_speco.backends import drafter_opd as opd
from verl_speco.backends import opd_distribution as distribution
from verl_speco.integration.verify_trace import TRACE_KEY


def trace():
    return dict(layout="dspark", num_spec_tokens=3, prompt_length=3, response_length=4,
        steps=[dict(anchor=2, drafts=3, accepted=1, raw_generated=2, commit=2, truncated=False),
               dict(anchor=4, drafts=3, accepted=1, raw_generated=2, commit=2, truncated=False)])


def test_cached_coordinates_are_validated_once_and_caller_mutation_is_isolated(monkeypatch):
    opd._TRACE_PLAN_CACHE.clear()
    calls = []
    original = opd.validate_verify_trace
    def validate(*a, **k):
        calls.append(1)
        return original(*a, **k)
    monkeypatch.setattr(opd, "validate_verify_trace", validate)
    item = {TRACE_KEY: trace()}
    def prepare():
        return opd.preprocess_trace(item, start=0, end=7, layout="dspark", capacity=3, device="cpu")
    first = prepare()
    assert first[2].tolist() == [1, 2, 1]
    first.fill_(99)
    second = prepare()
    assert second[2].tolist() == [1, 2, 1]
    assert len(calls) == 1
    step = item[TRACE_KEY]["steps"][0]
    step.update(accepted=2, raw_generated=3, truncated=True)
    assert prepare()[2].tolist() == [2, 2, 1]
    assert len(calls) == 2
    step["anchor"] = 99
    with pytest.raises(ValueError):
        prepare()


def test_cached_plan_retains_layout_and_aligned_next_token_checks():
    item = {TRACE_KEY: trace()}
    kw = dict(start=2, end=3, layout="dspark", capacity=3, device="cpu")
    assert opd.preprocess_trace(item, **kw)[0].tolist() == [1, 0, 1]
    assert opd.preprocess_trace(item, **kw, aligned_next_tokens=True)[0].tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        opd.preprocess_trace(item, **dict(kw, layout="dflash"))


@pytest.mark.parametrize("temperature", [0.6, 1., 1.7])
def test_shared_eager_tv_matches_original_value_gradient_and_teacher_stop(temperature):
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 17, requires_grad=True)
    target = torch.randn(2, 3, 17, requires_grad=True)
    actual = distribution.token_total_variation(logits, target,
        proposal_temperature=temperature, target_temperature=temperature, implementation="eager")
    expected = ((logits.float()/temperature).softmax(-1) - (target.detach().float()/temperature).softmax(-1)).abs().sum(-1)/2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), logits)[0])
    assert torch.autograd.grad(actual.sum(), target, allow_unused=True)[0] is None


def test_shared_fused_dispatch_flattens_rows_and_detaches_teacher(monkeypatch):
    calls = []
    monkeypatch.setattr(distribution, "_fused_available", lambda device: True)
    def fused(draft, target, *, average_ties):
        assert average_ties is True
        calls.append((draft.shape, target.requires_grad))
        return (draft.softmax(-1) - target.softmax(-1)).abs().sum(-1)/2
    monkeypatch.setattr(distribution, "fused_total_variation", fused)
    draft = torch.randn(2, 3, 11, requires_grad=True)
    target = torch.randn(2, 3, 11, requires_grad=True)
    actual = distribution.token_total_variation(draft, target)
    expected = distribution.token_total_variation(draft, target, implementation="eager")
    torch.testing.assert_close(actual, expected)
    assert calls == [(torch.Size([6, 11]), False)]


def test_eager_opt_out_is_honored_by_both_opd_objectives(monkeypatch):
    from verl_speco.backends.specopd_loss import e2e_tv_loss
    monkeypatch.setattr(distribution, "_fused_available", lambda device: True)
    def forbidden(*a, **k):
        pytest.fail("eager opt-out entered fused TV")
    monkeypatch.setattr(distribution, "fused_total_variation", forbidden)
    objective = opd.DraftOPDObjective(dict(distribution_loss_impl="eager"), layout="dspark",
        loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay")
    x=torch.randn(2,7,requires_grad=True)
    objective._tv(x,torch.randn(2,3),torch.randn(7,3)).sum().backward()
    e2e_tv_loss(x.reshape(1,2,7),torch.randn(1,2,7),torch.ones(1,2,dtype=torch.bool),
        distribution_loss_impl="eager")[0].backward()
