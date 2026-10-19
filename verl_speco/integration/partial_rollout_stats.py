# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve serving evidence across native abort/resume without changing tokens."""

from __future__ import annotations

import contextvars
import json
from functools import wraps


_SEGMENTS = contextvars.ContextVar("speco_partial_rollout_segments", default=None)
_PREFIX = "_speco_vllm_request_"
REVISION_MIN = "_speco_drafter_revision_min"
REVISION_MAX = "_speco_drafter_revision_max"
REVISION_STATS = "_speco_drafter_revision_stats"


def record_serving_segment(output):
    segments = _SEGMENTS.get()
    if segments is not None and len(getattr(output, "token_ids", [])):
        segment = dict(getattr(output, "extra_fields", {}) or {})
        segment["response_tokens"] = len(output.token_ids)
        segments.append(segment)


def merge_serving_segments(fields, segments):
    if not segments:
        return
    fields[_PREFIX + "id"] = segments[0].get(_PREFIX + "id")
    for name in ("verify_rounds", "draft_tokens", "accepted_tokens", "elapsed_sec"):
        values = [segment.get(_PREFIX + name) for segment in segments]
        fields[_PREFIX + name] = (
            sum(values) if all(v is not None for v in values) else None
        )
    rounds = fields.get(_PREFIX + "verify_rounds")
    accepted = fields.get(_PREFIX + "accepted_tokens")
    fields["_verl_request_mean_accept_len"] = (
        1.0 + accepted / rounds if rounds and accepted is not None else None
    )
    # Missing provenance in any nonempty segment invalidates the whole sample.
    mins = [segment.get(REVISION_MIN) for segment in segments]
    maxs = [segment.get(REVISION_MAX) for segment in segments]
    fields[REVISION_MIN] = min(mins) if all(v is not None for v in mins) else None
    fields[REVISION_MAX] = max(maxs) if all(v is not None for v in maxs) else None
    fields[_PREFIX + "segments"] = len(segments)
    # A partial trajectory can span several draft revisions. Preserve its
    # per-revision sufficient statistics rather than assigning the whole mean
    # to the latest revision or dropping every async trajectory as "mixed".
    revisions = {}
    for segment in segments:
        lo, hi = segment.get(REVISION_MIN), segment.get(REVISION_MAX)
        rounds = segment.get(_PREFIX + "verify_rounds")
        accepted = segment.get(_PREFIX + "accepted_tokens")
        if lo is None or lo != hi or rounds is None or accepted is None:
            fields[REVISION_STATS] = None  # Missing evidence fails closed.
            break
        row = revisions.setdefault(
            int(lo),
            {
                "revision": int(lo),
                "verify_rounds": 0,
                "accepted_tokens": 0,
                "response_tokens": 0,
                "seconds": 0.0,
            },
        )
        row["verify_rounds"] += rounds
        row["accepted_tokens"] += accepted
        row["response_tokens"] += segment.get("response_tokens", 0)
        row["seconds"] += segment.get(_PREFIX + "elapsed_sec") or 0.0
    else:
        fields[REVISION_STATS] = json.dumps(list(revisions.values()), sort_keys=True)


def group_serving_revision_evidence(fields, request_count, current_revision):
    """Group real request statistics by serving revision without changing inputs."""
    groups, matched = {}, 0
    columns = (
        REVISION_MIN,
        REVISION_MAX,
        REVISION_STATS,
        _PREFIX + "id",
        "_verl_request_mean_accept_len",
    )
    for index in range(request_count):
        request = {key: fields[key][index] for key in columns if key in fields}
        rows = None
        encoded = request.get(REVISION_STATS)
        if encoded is not None:
            try:
                rows = json.loads(encoded)
            except (TypeError, ValueError):
                pass
        if rows is None:
            lo, hi, mean = (
                request.get(REVISION_MIN),
                request.get(REVISION_MAX),
                request.get("_verl_request_mean_accept_len"),
            )
            if lo is not None and lo == hi and mean is not None:
                rows = [{"revision": int(lo), "mean": float(mean)}]
        qualified = False
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            version = row.get("revision")
            if not isinstance(version, int) or not 0 <= version <= current_revision:
                continue
            rounds, accepted = row.get("verify_rounds"), row.get("accepted_tokens")
            mean = row.get("mean")
            if rounds is not None and accepted is not None:
                mean = 1.0 + float(accepted) / float(rounds) if rounds > 0 else None
            if mean is None or request.get(_PREFIX + "id") is None:
                continue
            groups.setdefault(version, []).append(
                {
                    "id": request[_PREFIX + "id"],
                    "mean": float(mean),
                    "tokens": int(row.get("response_tokens", 0)),
                    "seconds": float(row.get("seconds", 0.0)),
                }
            )
            qualified = True
        matched += int(qualified)
    return groups, matched


def patch_partial_rollout_client(client_cls):
    if client_cls.__dict__.get("_speco_partial_stats_patched", False):
        return
    original = client_cls.generate

    @wraps(original)
    async def generate(self, *args, **kwargs):
        segments = []
        token = _SEGMENTS.set(segments)
        try:
            output = await original(self, *args, **kwargs)
            merge_serving_segments(output.extra_fields, segments)
            return output
        finally:
            _SEGMENTS.reset(token)

    client_cls.generate = generate
    client_cls._speco_partial_stats_patched = True
