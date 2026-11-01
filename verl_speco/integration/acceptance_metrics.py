# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Read existing host-side request counters without GPU reads or file I/O."""

import math
from numbers import Real


def _pair(fields, rounds_key, accepted_key):
    rounds_value = fields.get(rounds_key)
    accepted_value = fields.get(accepted_key)
    # Reject tensors before float() could perform a device scalar read.
    if not isinstance(rounds_value, (Real, str)) or not isinstance(
        accepted_value, (Real, str)
    ):
        return None
    try:
        rounds = float(rounds_value)
        accepted = float(accepted_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(rounds)
        or not math.isfinite(accepted)
        or rounds <= 0
        or accepted < 0
    ):
        return None
    return rounds, accepted


def request_acceptance_counters(fields):
    """Prefer native multi-turn totals; fall back to exact SpeCo request totals."""
    for rounds_key, accepted_key in (
        ("spec_num_verify_steps", "spec_num_accepted_tokens"),
        ("_speco_vllm_request_verify_rounds", "_speco_vllm_request_accepted_tokens"),
    ):
        if rounds_key == "spec_num_verify_steps" and fields.get(
            "_speco_acceptance_engine_fallback"
        ):
            continue
        pair = _pair(fields, rounds_key, accepted_key)
        if pair is not None:
            return pair
    return None


def bridge_request_acceptance_counters(fields):
    """Map authoritative request totals to verl's tool-turn accumulation keys.

    Run after SpeCo has attached the final per-request summary, so an aggregate
    engine-logger fallback cannot override that request's own counters.
    """
    pair = _pair(
        fields,
        "_speco_vllm_request_verify_rounds",
        "_speco_vllm_request_accepted_tokens",
    )
    if pair is None:
        return False
    rounds, accepted = pair
    fields.pop("_speco_acceptance_engine_fallback", None)
    fields["spec_num_verify_steps"] = rounds
    fields["spec_num_accepted_tokens"] = accepted
    draft_tokens = fields.get("_speco_vllm_request_draft_tokens")
    if (
        isinstance(draft_tokens, Real)
        and math.isfinite(draft_tokens)
        and draft_tokens >= 0
    ):
        fields["spec_num_draft_tokens"] = float(draft_tokens)
    fields["_verl_request_mean_accept_len"] = 1.0 + accepted / rounds
    return True
