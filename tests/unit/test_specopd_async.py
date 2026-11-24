# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Verify result-local proposal identity under asynchronous batch reordering."""
import pickle
from types import SimpleNamespace as NS

import pytest
import torch

from verl_speco.integration.specopd_async import (
    VERIFIED_KEY, snapshot_result_round,
    snapshot_verified_proposals,
)
from verl_speco.integration.specopd_vllm import capture_scheduler_round
from verl_speco.integration.specopd_window import make_window
from verl_speco.specopd_config import SpecOPDConfig
from verl_speco.integration.specopd_protocol import PLAN_KEY, token_digest


def fixture():
    plans = {rid: make_window(SpecOPDConfig(), rid, 5, 0, 3, 1024)
             for rid in ["a", "b"]}
    batch = NS(req_ids=["b", "a"], idx_mapping_np=[4, 1],
               num_draft_tokens_per_req=[7, 7], is_prefilling_np=[False, False],
               seq_lens=torch.tensor([10, 11]))
    drafts = torch.zeros((5, 7), dtype=torch.int32)
    drafts[4] = torch.arange(11, 18)
    drafts[1] = torch.arange(21, 28)
    runner = NS(draft_tokens_handler=NS(_specopd_plans=plans),
                execute_model_state=NS(input_batch=batch),
                is_last_pp_rank=True, req_states=NS(draft_tokens=drafts))
    requests = {rid: NS(sampling_params=NS(extra_args={PLAN_KEY: plans[rid]}),
                        all_token_ids=prefix, num_output_placeholders=16,
                        is_finished=lambda: False)
                for rid, prefix in [("a", [1, 2, 3, 4]), ("b", [1, 2, 3])]}
    scheduled = NS(num_scheduled_tokens={"a": 8, "b": 8},
                   scheduled_spec_decode_tokens={"a": [-1]*7, "b": [-1]*7})
    output = NS(req_id_to_index={"a": 0, "b": 1},
                sampled_token_ids=[[21, 99], [11, 12, 99]])
    return runner, NS(requests=requests), scheduled, output


def test_result_keeps_its_proposal_after_next_batch_overwrites_worker_state():
    runner, scheduler, scheduled, output = fixture()
    evidence = snapshot_verified_proposals(runner)
    runner.req_states.draft_tokens.fill_(999)
    setattr(output, VERIFIED_KEY, evidence)
    # The real worker channel pickles ModelRunnerOutput, including metadata.
    output = pickle.loads(pickle.dumps(output))
    snapshots = snapshot_result_round(scheduler, scheduled, output)
    capture_scheduler_round(scheduled, output, snapshots)
    a = scheduler.requests["a"]._specopd_records[0]
    b = scheduler.requests["b"]._specopd_records[0]
    assert a["proposal_ids"] == tuple(range(21, 28))
    assert a["accepted_count"] == 1
    assert a["prefix_digest"] == token_digest([1, 2, 3, 4])
    assert b["proposal_ids"] == tuple(range(11, 18))
    assert b["accepted_count"] == 2
    assert b["prefix_digest"] == token_digest([1, 2, 3])


def test_worker_does_not_spend_scheduler_quota_on_stale_delivery():
    runner, scheduler, scheduled, output = fixture()
    for plan in runner.draft_tokens_handler._specopd_plans.values():
        plan["max_branches"] = 1
    setattr(output, VERIFIED_KEY, snapshot_verified_proposals(runner))
    a, b = scheduler.requests["a"], scheduler.requests["b"]
    a.num_stale_output_tokens = 8
    snapshots = snapshot_result_round(scheduler, scheduled, output)
    assert set(snapshots) == {"b"}
    capture_scheduler_round(scheduled, output, snapshots)
    assert not hasattr(a, "_specopd_records")
    assert len(b._specopd_records) == 1

    a.num_stale_output_tokens = 0
    setattr(output, VERIFIED_KEY, snapshot_verified_proposals(runner))
    capture_scheduler_round(scheduled, output, snapshot_result_round(scheduler, scheduled, output))
    assert len(a._specopd_records) == len(b._specopd_records) == 1


@pytest.mark.parametrize("reservoir", [False, True])
def test_worker_stops_copying_when_capture_window_closes(reservoir):
    runner, _, _, _ = fixture()
    if reservoir:
        for plan in runner.draft_tokens_handler._specopd_plans.values():
            plan.update(reservoir=True, max_branches=24)
        assert all(len(snapshot_verified_proposals(runner)) == 2 for _ in range(25))
    runner.execute_model_state.input_batch.seq_lens.fill_(600)
    assert snapshot_verified_proposals(runner) == {}
    runner.req_states = None  # any additional device access would fail
    assert snapshot_verified_proposals(runner) == {}


@pytest.mark.parametrize("bad", ["prefix", "round", "missing"])
def test_async_result_must_match_committed_prefix_and_scheduling_round(bad):
    runner, scheduler, scheduled, output = fixture()
    evidence = snapshot_verified_proposals(runner)
    if bad == "prefix":
        evidence["a"]["prefix_len"] += 1
    elif bad == "round":
        evidence["a"]["scheduled_drafts"] -= 1
    else:
        evidence.pop("a")
    setattr(output, VERIFIED_KEY, evidence)
    with pytest.raises(RuntimeError):
        snapshot_result_round(scheduler, scheduled, output)


@pytest.mark.parametrize("sampled", [None, []])
def test_empty_prefill_result_does_not_require_verified_proposal(sampled):
    _, scheduler, scheduled, output = fixture()
    output.sampled_token_ids = [sampled, [11, 99]] if sampled is not None else None
    if sampled is not None:
        output.req_id_to_index = {"a": 0, "b": 1}
    # No proposal evidence for a chunk with no sampled output.
    if sampled is not None:
        runner, _, _, _ = fixture()
        evidence = snapshot_verified_proposals(runner)
        evidence.pop("a")
        setattr(output, VERIFIED_KEY, evidence)
    snapshots = snapshot_result_round(scheduler, scheduled, output)
    assert snapshots["a"][2:] == ([], [])
    capture_scheduler_round(scheduled, output, snapshots)
    assert not getattr(scheduler.requests["a"], "_specopd_records", [])
