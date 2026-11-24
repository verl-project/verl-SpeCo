# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Deterministic routing for standalone Producer requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

_MAX_RECORDS = 5_000_000


def plan_source_routes(
    lengths, *, source_count, max_samples, max_tokens, max_sequence_tokens
):
    """Plan one bounded capture wave without changing logical sample order.

    Returned routes are indexed by logical position. Empty sources are valid;
    a wave that cannot fit is rejected rather than dropping or copying samples.
    """
    for name, value in (
        ("source_count", source_count),
        ("max_samples", max_samples),
        ("max_tokens", max_tokens),
        ("max_sequence_tokens", max_sequence_tokens),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if source_count > 4096 or source_count * max_samples > _MAX_RECORDS:
        raise ValueError("Source routing plan exceeds bounded planning capacity")
    sizes = []
    for length in lengths:
        if len(sizes) >= source_count * max_samples:
            raise ValueError("Capture wave exceeds source sample capacity")
        if type(length) is not int or not 0 < length <= min(
            max_sequence_tokens, max_tokens
        ):
            raise ValueError(
                "Sequence exceeds source token limits or has invalid length"
            )
        sizes.append(length)
    loads, counts = [0] * source_count, [0] * source_count
    routes = [0] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
        eligible = [
            source
            for source in range(source_count)
            if counts[source] < max_samples
            and loads[source] + sizes[index] <= max_tokens
        ]
        if not eligible:
            raise ValueError("Capture wave cannot fit source token and sample limits")
        source = min(
            eligible, key=lambda source: (loads[source], counts[source], source)
        )
        routes[index] = source
        loads[source] += sizes[index]
        counts[source] += 1
    plan = {
        "version": 1,
        "lengths": sizes,
        "routes": routes,
        "source_count": source_count,
        "max_samples": max_samples,
        "max_tokens": max_tokens,
        "max_sequence_tokens": max_sequence_tokens,
    }
    plan["sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def iter_source_route_waves(
    requests,
    *,
    wave_samples,
    endpoints,
    max_samples,
    max_tokens,
    max_sequence_tokens,
    consumed=(),
):
    """Route full logical waves before excluding committed occurrences."""
    if type(wave_samples) is not int or not 1 <= wave_samples <= _MAX_RECORDS:
        raise ValueError("Invalid routing wave sample limit")
    endpoints = tuple(endpoints)
    if not endpoints or len(endpoints) > 4096 or len(set(endpoints)) != len(endpoints):
        raise ValueError("Source routing requires unique ordered endpoints")
    plan_source_routes(
        [],
        source_count=len(endpoints),
        max_samples=max_samples,
        max_tokens=max_tokens,
        max_sequence_tokens=max_sequence_tokens,
    )
    if wave_samples > len(endpoints) * max_samples:
        raise ValueError("Routing wave exceeds source sample capacity")
    consumed = set(consumed)

    def emit(wave):
        costs = [
            len(request.prompt_token_ids) + max(0, getattr(request, "max_tokens", 0))
            for request in wave
        ]
        plan = plan_source_routes(
            costs,
            source_count=len(endpoints),
            max_samples=max_samples,
            max_tokens=max_tokens,
            max_sequence_tokens=max_sequence_tokens,
        )
        identity = {
            "plan": plan,
            "endpoints": endpoints,
            "occurrences": [
                (request.sequence_no, request.sample_id) for request in wave
            ],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for request, route in zip(wave, plan["routes"]):
            if request.sequence_no not in consumed:
                yield replace(request, source_route=route, source_plan_sha256=digest)

    wave, wave_id, previous, wave_tokens = [], None, -1, 0
    for request in requests:
        sequence = request.sequence_no
        if type(sequence) is not int or sequence <= previous:
            raise ValueError(
                "Source routing requires increasing occurrence sequence numbers"
            )
        current = sequence // wave_samples
        if wave and current != wave_id:
            yield from emit(wave)
            wave = []
            wave_tokens = 0
        cost = len(request.prompt_token_ids) + max(0, getattr(request, "max_tokens", 0))
        if not 0 < cost <= min(max_sequence_tokens, max_tokens):
            raise ValueError(
                "Sequence exceeds source token limits or has invalid length"
            )
        if wave_tokens + cost > len(endpoints) * max_tokens:
            raise ValueError("Routing wave exceeds aggregate source token capacity")
        wave.append(request)
        wave_tokens += cost
        wave_id, previous = current, sequence
    if wave:
        yield from emit(wave)
