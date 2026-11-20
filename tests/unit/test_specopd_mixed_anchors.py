# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Mixed-anchor sampling and value/gradient equivalence to Draft-OPD."""

from types import SimpleNamespace as NS

import pytest
import torch

from verl_speco.backends.drafter_opd import DraftOPDObjective
from verl_speco.backends.specopd_loss import e2e_tv_loss
from verl_speco.integration.specopd_anchors import retain_proposal, add_trunk_anchors
from verl_speco.joint_scoring import DraftBranch
from verl_speco.specopd_config import SpecOPDConfig


def config(anchor_count=32, **kw):
    proposal_count = anchor_count * 3 // 4
    return SpecOPDConfig(anchor_count=anchor_count, random_anchor_count=anchor_count // 4,
                         max_branches_per_request=proposal_count,
                         max_proposal_tokens_per_request=proposal_count * 7, **kw)


def bundle():
    return dict(request_id="request", actor_revision=5, drafter_revision=0, window=[0, 512])


def test_collection_cap_optout_cannot_reenter_training_as_random_trunk_anchors():
    assert add_trunk_anchors([], tuple(range(600)), 10,
                             dict(bundle(), collection_enabled=False), config()) == []


def proposals(n):
    return [DraftBranch(f"p{i}", 9 + i, (2, 3, 4), accepted_count=1) for i in range(n)]


@pytest.mark.parametrize("anchor_count", [32, 64])
def test_full_and_short_proposal_groups_fill_unique_positions(anchor_count):
    ids = tuple(range(600))
    spec = config(anchor_count)
    spec.validate()
    for n in [0, 3, spec.max_branches_per_request]:
        mixed = add_trunk_anchors(proposals(n), ids, 10, bundle(), spec)
        assert len(mixed) == len({b.anchor for b in mixed}) == anchor_count
        assert sum(b.source == "proposal" for b in mixed) == n
        assert all(9 <= b.anchor < 521 for b in mixed)
        for b in mixed[n:]:
            assert b.source == "trunk"
            assert b.token_ids == ids[b.anchor + 1:min(b.anchor + 8, 522)]
        assert mixed == add_trunk_anchors(proposals(n), ids, 10, bundle(), spec)


def test_short_response_never_fabricates_or_duplicates_anchors():
    mixed = add_trunk_anchors([], tuple(range(14)), 10, bundle(), config())
    assert len(mixed) == 4
    assert len(mixed[-1].token_ids) >= 1
    assert all(b.token_ids for b in mixed)


def test_reservoir_can_replace_early_rounds_and_is_uniform_across_candidates():
    counts = [0] * 80
    for seed in range(800):
        request = NS(_specopd_records=[])
        plan = dict(request_id=str(seed), actor_revision=5, drafter_revision=0, max_branches=24)
        for i in range(80):
            retain_proposal(request, plan, dict(round_id=i))
        kept = [x["round_id"] for x in request._specopd_records]
        assert len(set(kept)) == 24
        for i in kept:
            counts[i] += 1
    # Expected 240 selections per candidate; reject first-N/last-N/biased sampling.
    assert min(counts) > 170 and max(counts) < 310


def test_mixed_loss_and_gradient_match_draft_opd_with_padding_and_weights():
    torch.manual_seed(7)
    head = torch.randn(13, 5)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    accepted = torch.tensor([[1, -1, 3, -1]])
    reject = torch.tensor([[True, False, False, False]])
    objective = DraftOPDObjective(dict(loss="e2e_tv", first_reject_weight=2.0),
                                  layout="dspark", loss_mode="full_vocab",
                                  per_position_loss_weight="fixed_exp_decay")
    weights = objective.weights(valid.reshape(1, 4, 4).float(), accepted, reject, gamma=7.0)
    active = valid.flatten()
    # Work on all rows with matching inactive padding; Draft-OPD takes active rows.
    dense = torch.randn(4, 4, 13, requires_grad=True)
    target = torch.randn(4, 4, 5)
    expected = objective(dense.reshape(-1, 13)[active], target.reshape(-1, 5)[active], head,
                         valid.reshape(1, 4, 4).float(), active, position_weights=weights)
    numerator, count, _ = e2e_tv_loss(dense, torch.nn.functional.linear(target, head), valid,
                                     position_weights=weights.reshape_as(valid))
    actual = numerator / count
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.autograd.grad(actual, dense, retain_graph=True)[0],
                               torch.autograd.grad(expected, dense)[0], atol=1e-6, rtol=1e-5)
    assert count == 3


def test_mixed_config_requires_matching_budgets():
    config().validate()
    with pytest.raises(ValueError, match="24 proposal"):
        SpecOPDConfig(anchor_count=32).validate()


def test_async_reservoir_keeps_capturing_after_24_and_stops_at_window_end():
    from verl_speco.integration.specopd_async import snapshot_verified_proposals
    from verl_speco.integration.specopd_window import make_window
    plan = make_window(config(), "a", 5, 0, 3, 1024)
    batch = NS(req_ids=["a"], idx_mapping_np=[0], num_draft_tokens_per_req=[7],
               is_prefilling_np=[False], seq_lens=torch.tensor([100]))
    runner = NS(draft_tokens_handler=NS(_specopd_plans={"a": plan}, _specopd_result_counts={"a": 24}),
                execute_model_state=NS(input_batch=batch), is_last_pp_rank=True,
                req_states=NS(draft_tokens=torch.ones((1, 7), dtype=torch.int32)))
    assert "a" in snapshot_verified_proposals(runner)
    batch.seq_lens[0] = 700
    assert snapshot_verified_proposals(runner) == {}
    runner.req_states = None
    assert snapshot_verified_proposals(runner) == {}
