# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""L1 verification evidence, in absolute token coordinates."""

import contextvars
import hashlib

TRACE_KEY = "_speco_verify_trace"
TURN_TRACES = contextvars.ContextVar("speco_verify_turns", default=None)


def snapshot_verify_step(scheduler, scheduled, output):
    """Read CPU counts before Scheduler mutates generated tokens or frees requests."""
    snapshots = []
    for rid in scheduled.num_scheduled_tokens:
        request = scheduler.requests.get(rid)
        if request is None or request.is_finished():
            continue
        if rid not in output.req_id_to_index:
            continue  # A failed KV load has no worker row and commits no tokens.
        index = output.req_id_to_index[rid]
        generated = (
            len(output.sampled_token_ids[index]) if output.sampled_token_ids else 0
        )
        drafts = len(scheduled.scheduled_spec_decode_tokens.get(rid, ()))
        if scheduled.num_invalid_spec_tokens:
            drafts -= scheduled.num_invalid_spec_tokens.get(rid, 0)
        snapshots.append(
            (rid, request, len(request._output_token_ids), drafts, generated)
        )
    return snapshots


def finish_verify_step(snapshots, *, layout, capacity):
    traces = {}
    for rid, request, before, drafts, generated in snapshots:
        prompt_len = len(request.prompt_token_ids)
        trace = getattr(request, TRACE_KEY, None)
        if trace is None:
            trace = {
                "layout": layout,
                "num_spec_tokens": capacity,
                "prompt_length": prompt_len,
                "response_length": 0,
                "steps": [],
            }
            setattr(request, TRACE_KEY, trace)
        commit = len(request._output_token_ids) - before
        if generated or commit:
            trace["steps"].append(
                {
                    "anchor": prompt_len + before - 1,
                    "commit": commit,
                    "drafts": drafts,
                    "accepted": max(generated - 1, 0) if drafts else 0,
                    "raw_generated": generated,
                    "truncated": commit < generated,
                }
            )
        trace["response_length"] = len(request._output_token_ids)
        traces[rid] = trace
    return traces


def validate_verify_trace(trace, *, layout, capacity):
    """Validate the scheduler/agent boundary; malformed evidence must not train."""
    if trace["layout"] != layout or trace["num_spec_tokens"] != capacity:
        raise ValueError(
            "Draft-OPD trace layout or speculative capacity mismatch: "
            f"trace layout={trace['layout']}, num_spec_tokens={trace['num_spec_tokens']}; "
            f"expected layout={layout}, num_spec_tokens={capacity}"
        )
    cursor = trace["prompt_length"] - 1
    for step in trace["steps"]:
        c, n, a, raw = (
            step[k] for k in ("commit", "drafts", "accepted", "raw_generated")
        )
        if (
            step["anchor"] != cursor
            or not 0 <= c <= raw
            or not 0 <= a <= n <= capacity
            or (n > 0 and raw != a + 1)
            or step["truncated"] != (c < raw)
        ):
            raise ValueError("Invalid Draft-OPD verify step or token coordinates")
        cursor += c
    if cursor + 1 - trace["prompt_length"] != trace["response_length"]:
        raise ValueError("Draft-OPD committed tokens do not cover the response")


def token_digest(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def align_trace_to_prompt(trace, prompt_width):
    first = trace["turns"][0] if "turns" in trace else trace
    padding = prompt_width - first["prompt_length"]
    if padding < 0:
        raise ValueError("Draft-OPD prompt width is shorter than the request prompt")
    return {**trace, "token_offset": padding}


def collect_turn_trace(output):
    turns = TURN_TRACES.get()
    fields = getattr(output, "extra_fields", None)
    if (
        turns is not None
        and isinstance(fields, dict)
        and fields.get(TRACE_KEY) is not None
    ):
        turns.append(fields[TRACE_KEY])


def merge_turn_traces(output):
    """Keep each request's absolute coordinates across tool turns and resumes."""
    turns = TURN_TRACES.get()
    if turns:
        ids = [*output.prompt_ids, *output.response_ids]
        for turn in turns:
            end = turn["prompt_length"] + turn["response_length"]
            if token_digest(ids[:end]) != turn["prefix_digest"]:
                raise ValueError(
                    "Draft-OPD request tokens differ from the final agent trajectory"
                )
        output.extra_fields[TRACE_KEY] = {"turns": list(turns)}
