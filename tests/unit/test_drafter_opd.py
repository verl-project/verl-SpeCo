# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from verl_speco.backends.drafter_opd import DraftOPDObjective, preprocess_trace
from verl_speco.integration.verify_trace import (
    TRACE_KEY,
    TURN_TRACES,
    collect_turn_trace,
    finish_verify_step,
    merge_turn_traces,
    snapshot_verify_step,
    token_digest,
    validate_verify_trace,
    align_trace_to_prompt,
)


def trace(layout="dspark", capacity=3):
    return {
        "layout": layout,
        "num_spec_tokens": capacity,
        "prompt_length": 3,
        "response_length": 6,
        "steps": [
            {
                "anchor": 2,
                "commit": 1,
                "drafts": 0,
                "accepted": 0,
                "raw_generated": 1,
                "truncated": False,
            },
            {
                "anchor": 3,
                "commit": 2,
                "drafts": 3,
                "accepted": 1,
                "raw_generated": 2,
                "truncated": False,
            },
            {
                "anchor": 5,
                "commit": 3,
                "drafts": 3,
                "accepted": 3,
                "raw_generated": 4,
                "truncated": True,
            },
        ],
    }


def test_scheduler_snapshot_preserves_raw_acceptance_after_stop_and_grammar():
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3], _output_token_ids=[4], is_finished=lambda: False
    )
    scheduler = SimpleNamespace(requests={"r": request})
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"r": 4},
        scheduled_spec_decode_tokens={"r": [-1] * 3},
        num_invalid_spec_tokens={"r": 1},
    )
    output = SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[5, 6, 7]])
    snapshot = snapshot_verify_step(scheduler, scheduled, output)
    # Scheduler truncates at EOS and removes the finished request.
    request._output_token_ids.append(5)
    scheduler.requests.clear()
    result = finish_verify_step(snapshot, layout="dspark", capacity=3)["r"]
    step = result["steps"][0]
    assert step == {
        "anchor": 3,
        "commit": 1,
        "drafts": 2,
        "accepted": 2,
        "raw_generated": 3,
        "truncated": True,
    }


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_trace_crop_retains_first_response_anchor_and_does_not_invent_rejection(layout):
    source = trace(layout)
    validate_verify_trace(source, layout=layout, capacity=3)
    plan = preprocess_trace(
        {TRACE_KEY: source, "_verl_feature_start": 2},
        start=1,
        end=7,
        layout=layout,
        capacity=3,
        device="cpu",
    )
    assert plan[0].tolist() == [1, 2, 1]
    assert plan[2].tolist() == [3, 3, 0]
    assert (plan[1] == -1).all()


@pytest.mark.parametrize(
    "field,value", [("accepted", 4), ("anchor", 4), ("truncated", True), ("commit", 0)]
)
def test_malformed_trace_fails(field, value):
    source = trace()
    source["steps"][1][field] = value
    with pytest.raises(ValueError):
        validate_verify_trace(source, layout="dspark", capacity=3)


def test_trace_layout_and_capacity_must_match():
    with pytest.raises(ValueError):
        validate_verify_trace(trace(), layout="dflash", capacity=3)
    with pytest.raises(ValueError):
        validate_verify_trace(trace(), layout="dspark", capacity=2)


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_e2e_tv_matches_dense_reference_and_gradient(layout):
    torch.manual_seed(7)
    offset = int(layout == "dflash")
    base = torch.zeros(1, 3, 3 + offset)
    base[0, 0, offset : offset + 2] = 1
    base[0, 1, offset:] = 1
    active = base.reshape(-1).bool()
    logits = torch.randn(5, 7, requires_grad=True)
    hidden = torch.randn(5, 4)
    weight = torch.randn(7, 4)
    objective = DraftOPDObjective(
        {"chunk_size": 2},
        layout=layout,
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
    )
    actual = objective(logits, hidden, weight, base, active, position_weights=base)
    tv = (F.softmax(hidden @ weight.T, -1) - F.softmax(logits, -1)).abs().sum(-1) / 2
    expected = torch.stack(
        [1 - (1 - tv[:2]).cumprod(0).mean(), 1 - (1 - tv[2:]).cumprod(0).mean()]
    ).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_empty_tv_keeps_gradient_graph():
    objective = DraftOPDObjective(
        {},
        layout="dspark",
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
    )
    logits = torch.empty(0, 7, requires_grad=True)
    loss = objective(
        logits,
        torch.empty(0, 4),
        torch.randn(7, 4),
        torch.zeros(1, 2, 3),
        torch.zeros(6, dtype=torch.bool),
        position_weights=torch.zeros(1, 2, 3),
    )
    loss.backward()
    assert loss.item() == 0
    assert logits.grad is not None


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
@pytest.mark.parametrize("gamma", [None, 7.0])
def test_e2e_tv_position_weights_match_dense_loss_and_gradient(layout, gamma):
    torch.manual_seed(17)
    offset = int(layout == "dflash")
    base = torch.zeros(1, 3, 3 + offset)
    base[0, 0, offset : offset + 2] = 1  # accepted token, then first rejection
    base[0, 1, offset:] = 1  # random block
    active = base.reshape(-1).bool()
    logits = torch.randn(5, 7, requires_grad=True)
    hidden, head = torch.randn(5, 4), torch.randn(7, 4)
    objective = DraftOPDObjective(
        {"first_reject_weight": 2.0, "chunk_size": 2}, layout=layout,
        loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
    )
    weights = objective.weights(
        base, torch.tensor([[1, -1, 0]]), torch.tensor([[True, False, True]]),
        gamma=gamma,
    )
    actual = objective(
        logits, hidden, head, base, active, position_weights=weights,
    )
    tv = ((hidden @ head.T).softmax(-1) - logits.softmax(-1)).abs().sum(-1) / 2
    random_weights = (
        torch.ones(3) if gamma is None else torch.exp(-torch.arange(3) / gamma)
    )
    expected = torch.stack([
        ((1 - (1 - tv[:2]).cumprod(0)) * torch.tensor([1.0, 2.0])).sum() / 2,
        ((1 - (1 - tv[2:]).cumprod(0)) * random_weights).sum() / 3,
    ]).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_e2e_tv_single_first_rejection_scales_loss_and_gradient(layout):
    offset = int(layout == "dflash")
    base = torch.ones(1, 1, 1 + offset)
    base[..., :offset] = 0
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    hidden, head = torch.ones(1, 1), torch.tensor([[0.0], [1.0]])
    results = []
    for multiplier in (1.0, 2.0):
        objective = DraftOPDObjective(
            {"first_reject_weight": multiplier}, layout=layout,
            loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
        )
        weights = objective.weights(base, torch.tensor([[0]]), torch.tensor([[True]]))
        loss = objective(
            logits, hidden, head, base, base.reshape(-1).bool(),
            position_weights=weights,
        )
        results.append((loss, torch.autograd.grad(loss, logits)[0]))
    torch.testing.assert_close(results[1][0], 2 * results[0][0])
    torch.testing.assert_close(results[1][1], 2 * results[0][1])


@pytest.mark.parametrize(
    "planner,expected_random", [("random", 4), ("trace", 0), ("mixed", 1)]
)
def test_planner_has_fixed_budget_and_explicit_random_quota(planner, expected_random):
    objective = DraftOPDObjective(
        {"anchor_planner": planner},
        layout="dspark",
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
    )
    coords = torch.full((1, 10, 3), -1)
    coords[0, 2] = torch.tensor([0, 1, 1])
    coords[0, 5] = torch.tensor([2, 3, 1])
    anchors, keep, accepted, visible, reject = objective.plan(
        torch.tensor([[1, 3, 6, 7]]), torch.ones(1, 4, dtype=torch.bool), coords
    )
    assert anchors.shape == (1, 4)
    assert ((accepted == -1) & keep).sum().item() == expected_random
    assert keep.sum().item() == expected_random + min(4 - expected_random, 2)
    if planner == "trace":
        assert set(anchors[keep].tolist()) == {2, 5}


def test_truncated_step_does_not_receive_false_first_reject_weight():
    objective = DraftOPDObjective(
        {},
        layout="dspark",
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
    )
    base = objective.mask(torch.ones(1, 2, 3), torch.tensor([[1, 2]]))
    weights = objective.weights(
        base, torch.tensor([[2, 1]]), torch.tensor([[True, True]])
    )
    assert weights.tolist() == [[[1, 0, 0], [1, 2, 0]]]


def test_tool_turn_and_partial_segments_keep_absolute_coordinates():
    tokens = list(range(12))
    first = trace()
    first["prefix_digest"] = token_digest(tokens[:9])
    second = {
        "layout": "dspark",
        "num_spec_tokens": 3,
        "prompt_length": 10,
        "response_length": 2,
        "steps": [
            {
                "anchor": 9,
                "commit": 2,
                "drafts": 3,
                "accepted": 1,
                "raw_generated": 2,
                "truncated": False,
            }
        ],
        "prefix_digest": token_digest(tokens),
    }
    context = TURN_TRACES.set([])
    try:
        for turn in (first, second):
            collect_turn_trace(SimpleNamespace(extra_fields={TRACE_KEY: turn}))
        output = SimpleNamespace(
            prompt_ids=tokens[:3], response_ids=tokens[3:], extra_fields={}
        )
        merge_turn_traces(output)
        plan = preprocess_trace(
            {TRACE_KEY: output.extra_fields[TRACE_KEY]},
            start=0,
            end=12,
            layout="dspark",
            capacity=3,
            device="cpu",
        )
        assert plan[9].tolist() == [1, 2, 1]
        output.response_ids[-1] = 99
        with pytest.raises(ValueError, match="tokens differ"):
            merge_turn_traces(output)
    finally:
        TURN_TRACES.reset(context)


def test_missing_trace_fails_instead_of_changing_anchor_policy():
    with pytest.raises(ValueError, match="requires a captured"):
        preprocess_trace({}, start=0, end=5, layout="dspark", capacity=3, device="cpu")


def test_left_padding_offset_is_removed_after_feature_and_window_crop():
    aligned = align_trace_to_prompt(trace(), prompt_width=7)
    plan = preprocess_trace(
        {TRACE_KEY: aligned, "_verl_feature_start": 5},
        start=2,
        end=8,
        layout="dspark",
        capacity=3,
        device="cpu",
    )
    assert plan[0].tolist() == [1, 2, 1]  # absolute anchor 3 + pad 4 - crop 7
    assert plan[2].tolist() == [3, 3, 0]


def test_tv_uses_configured_temperature():
    objective = DraftOPDObjective(
        {},
        layout="dspark",
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
        temperature=0.6,
    )
    logits = torch.tensor([[2.0, -1.0, 0.0]])
    hidden, weight = torch.ones(1, 1), torch.tensor([[0.0], [1.0], [2.0]])
    expected = (
        logits.div(0.6).softmax(-1) - (hidden @ weight.T).div(0.6).softmax(-1)
    ).abs().sum(-1) / 2
    torch.testing.assert_close(objective._tv(logits, hidden, weight), expected)


def test_chunked_prefill_and_failed_kv_load_produce_no_false_verify_step():
    request = SimpleNamespace(
        prompt_token_ids=[1, 2], _output_token_ids=[], is_finished=lambda: False
    )
    scheduler = SimpleNamespace(requests={"r": request})
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"r": 1},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens={},
    )
    output = SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[])
    snapshots = snapshot_verify_step(scheduler, scheduled, output)
    result = finish_verify_step(snapshots, layout="dspark", capacity=3)["r"]
    assert result["steps"] == []
    output.req_id_to_index.clear()
    assert snapshot_verify_step(scheduler, scheduled, output) == []
