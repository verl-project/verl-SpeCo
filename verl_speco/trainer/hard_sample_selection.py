# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Pure hard-sample candidate partitioning helpers."""

from __future__ import annotations

import math
from typing import Any


def partition_hard_candidates(
    scored_candidates: list[dict[str, Any]],
    *,
    candidate_ratio: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return the lowest-acceptance hard candidates and their count."""
    if not math.isfinite(candidate_ratio) or not 0.0 <= candidate_ratio <= 1.0:
        raise ValueError(f"Invalid hard candidate ratio: {candidate_ratio}")

    ordered = sorted(
        scored_candidates,
        key=lambda row: (row["mean_accept_len"], row["hash"]),
    )
    candidate_count = round(len(ordered) * candidate_ratio)
    return ordered[:candidate_count], candidate_count
