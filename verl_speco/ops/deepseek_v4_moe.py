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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations


# Adapted from training_common/utils/moe_chunking.py.
def plan_budgeted_expert_groups(
    group_counts: list[int] | tuple[int, ...],
    route_budget: str | int,
    max_experts_per_group: int,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Pack adjacent active experts under route and expert-count budgets."""
    counts = [int(count) for count in group_counts]
    if any(count < 0 for count in counts):
        raise ValueError(f"Expert route counts must be non-negative: {counts}")
    if max_experts_per_group <= 0:
        raise ValueError(
            f"max_experts_per_group must be positive, got {max_experts_per_group}"
        )

    maximum_count = max(counts, default=0)
    normalized_budget = str(route_budget).strip().lower()
    if normalized_budget == "auto":
        effective_budget = max(1, maximum_count)
    else:
        configured_budget = int(normalized_budget)
        if configured_budget <= 0:
            raise ValueError(
                f"route_budget must be 'auto' or positive, got {route_budget!r}"
            )
        # Individual experts are not split, so the busiest expert is the
        # unavoidable lower bound for a route-row budget.
        effective_budget = max(configured_budget, maximum_count, 1)

    route_offsets = [0]
    for count in counts:
        route_offsets.append(route_offsets[-1] + count)
    active_experts = [expert_id for expert_id, count in enumerate(counts) if count > 0]
    if not active_experts:
        return [], effective_budget

    groups: list[tuple[int, int, int, int]] = []
    group_start = active_experts[0]
    group_end = group_start + 1
    group_routes = counts[group_start]
    for expert_id in active_experts[1:]:
        candidate_end = expert_id + 1
        candidate_experts = candidate_end - group_start
        candidate_routes = group_routes + counts[expert_id]
        if (
            candidate_experts <= max_experts_per_group
            and candidate_routes <= effective_budget
        ):
            group_end = candidate_end
            group_routes = candidate_routes
            continue
        groups.append(
            (
                group_start,
                group_end,
                route_offsets[group_start],
                route_offsets[group_end],
            )
        )
        group_start = expert_id
        group_end = expert_id + 1
        group_routes = counts[expert_id]
    groups.append(
        (
            group_start,
            group_end,
            route_offsets[group_start],
            route_offsets[group_end],
        )
    )
    return groups, effective_budget
