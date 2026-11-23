# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Carry the proposal being verified with its own MRV2 model output.

The next proposal overwrites worker state before the current result reaches
the scheduler. Snapshot before sampling, and attach to that result, never to
the scheduler's speculative placeholders or a mutable 'latest proposal'.
"""

from functools import wraps

VERIFIED_KEY = "_specopd_verified_proposals"


def snapshot_verified_proposals(runner):
    handler = runner.draft_tokens_handler
    plans = getattr(handler, "_specopd_plans", {})
    if not plans:
        return None
    state = runner.execute_model_state
    if state is None or not runner.is_last_pp_rank:
        return None
    batch = state.input_batch
    closed = getattr(handler, "_specopd_closed_windows", None)
    if closed is None:
        closed = handler._specopd_closed_windows = set()
    widths = batch.num_draft_tokens_per_req
    if widths is None:
        return {}
    selected = [
        (i, rid, plans[rid])
        for i, rid in enumerate(batch.req_ids)
        if rid in plans
        and int(widths[i]) > 0
        and not bool(batch.is_prefilling_np[i])
        and rid not in closed
    ]
    if not selected:
        return {}
    import torch

    # Only the scheduler can charge the branch budget: a preempted result
    # may be discarded before it ever reaches the committed trajectory.
    rows = [i for i, _, _ in selected]
    slots = torch.tensor(
        [int(batch.idx_mapping_np[i]) for i in rows],
        device=runner.req_states.draft_tokens.device,
        dtype=torch.long,
    )
    # Full fixed-width proposals, including tokens not admitted by the scheduler.
    drafts = runner.req_states.draft_tokens[slots]
    sizes = torch.tensor([int(widths[i]) for i in rows], device=drafts.device)
    prefix_lengths = batch.seq_lens[rows].to(sizes.dtype) - sizes
    # One bounded copy only for opt-in requests; no JSON parsing or device work
    # at all on non-collection steps. Copy now, before propose() reuses storage.
    packed = torch.cat((prefix_lengths[:, None], drafts.to(sizes.dtype)), dim=1)
    values = packed.detach().cpu().tolist()
    result = {}
    for (i, rid, plan), row in zip(selected, values, strict=True):
        prefix_len, *tokens = row
        offset = prefix_len - plan["prompt_len"]
        if not plan["start"] <= offset < plan["end"]:
            if offset >= plan["end"]:
                closed.add(rid)
            continue
        result[rid] = {
            "prefix_len": prefix_len,
            "proposal_ids": tokens,
            "scheduled_drafts": int(widths[i]),
        }
    return result


def patch_result_transport(runner_class):
    if getattr(runner_class, "_specopd_result_transport", False):
        return True
    original = runner_class.sample_tokens

    @wraps(original)
    def sampled(self, *args, **kwargs):
        proposals = snapshot_verified_proposals(self)
        result = original(self, *args, **kwargs)
        if proposals is not None and result is not None:
            # AsyncOutput owns this object until its normal get_output() copy
            # completes. WorkerProc then pickles the entire ModelRunnerOutput.
            output = getattr(result, "model_runner_output", result)
            setattr(output, VERIFIED_KEY, proposals)
        return result

    runner_class.sample_tokens = sampled
    runner_class._specopd_result_transport = True
    return True


def snapshot_result_round(scheduler, scheduled, output):
    """Use committed history at result delivery, just like Draft-OPD trace."""
    from verl_speco.integration.specopd_vllm import _capture_active, _plan

    evidence = getattr(output, VERIFIED_KEY, None)
    sampled = getattr(output, "sampled_token_ids", None)
    mapping = getattr(output, "req_id_to_index", {})
    snapshots = {}
    for rid in scheduled.num_scheduled_tokens:
        request = scheduler.requests.get(rid)
        if request is None or request.is_finished():
            continue
        plan = _plan(request)
        if not plan:
            continue
        # Stale deliveries from preempted requests must never create branches.
        if getattr(request, "num_stale_output_tokens", 0) > 0:
            continue
        if sampled is None or rid not in mapping or not len(sampled[mapping[rid]]):
            # Chunked prefill can reserve speculative placeholders without
            # producing a verification result. Keep final-bundle bookkeeping,
            # but require proposal evidence only for an actual sampled result.
            snapshots[rid] = (request, plan, [], [])
            continue
        offset = len(request.all_token_ids) - plan["prompt_len"]
        if not _capture_active(request, plan, offset):
            # Retain the request/plan for final bundle delivery without copying
            # an ever-growing history after the capture window/budget closes.
            snapshots[rid] = (request, plan, [], [])
            continue
        prefix = list(request.all_token_ids)
        raw = []
        row = evidence.get(rid) if evidence is not None else None
        if row is not None:
            if row["prefix_len"] != len(prefix):
                raise RuntimeError(
                    "SpecOPD verified proposal prefix differs from committed history"
                )
            if row["scheduled_drafts"] != len(
                scheduled.scheduled_spec_decode_tokens.get(rid, ())
            ):
                raise RuntimeError(
                    "SpecOPD verified proposal belongs to a different scheduling round"
                )
            raw = list(row["proposal_ids"])
        if scheduled.scheduled_spec_decode_tokens.get(rid) and not raw:
            raise RuntimeError(
                "SpecOPD async capture requires complete verified MRV2 proposal metadata: "
                f"request={rid}, offset={offset}, "
                f"scheduled_drafts={len(scheduled.scheduled_spec_decode_tokens[rid])}, "
                f"sampled_tokens={len(sampled[mapping[rid]])}, "
                f"metadata_present={evidence is not None}"
            )
        snapshots[rid] = (request, plan, prefix, raw)
    return snapshots
