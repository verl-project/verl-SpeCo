# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded uniform proposal reservoir and disjoint random trunk anchors."""

import hashlib
import random

from verl_speco.joint_scoring import DraftBranch


def _rng(key):
    return random.Random(
        int.from_bytes(hashlib.sha256(key.encode()).digest(), "little")
    )


def retain_proposal(request, plan, record):
    """Algorithm R: every eligible verification round has the same inclusion odds."""
    records = request._specopd_records
    seen = getattr(request, "_specopd_candidates_seen", 0) + 1
    request._specopd_candidates_seen = seen
    capacity = plan["max_branches"]
    if len(records) < capacity:
        records.append(record)
        return
    slot = _rng(
        f"{plan['request_id']}:{plan['actor_revision']}:{plan['drafter_revision']}:{seen}"
    ).randrange(seen)
    if slot < capacity:
        records[slot] = record


def add_trunk_anchors(branches, ids, prompt_len, bundle, config):
    """Fill the configured number of unique starts in the planned response window."""
    if bundle.get("collection_enabled", True) is False:
        if branches:
            raise ValueError(
                "disabled SpecOPD collection cannot contain proposal branches"
            )
        return []
    window = bundle.get("window")
    if (
        not isinstance(window, (list, tuple))
        or len(window) != 2
        or any(type(v) is not int for v in window)
        or not 0 <= window[0] < window[1]
    ):
        raise ValueError("mixed SpecOPD requires the actual rollout collection window")
    proposal_slots = config.anchor_count - config.random_anchor_count
    if len(branches) > proposal_slots:
        raise ValueError(
            f"mixed SpecOPD proposal reservoir exceeds its {proposal_slots}-anchor budget"
        )
    used = {b.anchor for b in branches}
    if len(used) != len(branches):
        raise ValueError("mixed SpecOPD cannot use duplicate proposal anchor positions")
    start, end = window
    candidates = [
        a
        for a in range(prompt_len - 1 + start, min(prompt_len - 1 + end, len(ids) - 1))
        if a not in used
    ]
    count = min(config.anchor_count - len(branches), len(candidates))
    rng = _rng(
        f"{bundle['request_id']}:{bundle['actor_revision']}:{bundle['drafter_revision']}:trunk"
    )
    result = list(branches)
    for anchor in rng.sample(candidates, count):
        tokens = tuple(
            ids[anchor + 1 : min(anchor + 1 + config.proposal_width, prompt_len + end)]
        )
        result.append(
            DraftBranch(
                f"{bundle['request_id']}:trunk:{anchor}", anchor, tokens, source="trunk"
            )
        )
    return result
