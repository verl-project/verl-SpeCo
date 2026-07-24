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
"""Conditional-On-Distribution (COD) sampling for P-EAGLE parallel drafting.

P-EAGLE trains all ``K`` draft depths in a single parallel forward rather than
EAGLE-3's sequential test-time-training unroll. COD subsamples deeper depths with
geometric decay: depth 0 keeps all ``n`` positions, depth ``d`` keeps
``n * r**d`` (floored at ``down_sample_ratio_min``), dropping the attention cost
from ``O((nK)^2)`` to ``O((n * sum r^i)^2)``.

Verbatim port of NeMo AutoModel's ``peagle_data.generate_cod_sample_indices``
(itself a port of speculators PR #480), so the trained draft matches the
distribution vLLM's parallel-drafting runtime samples at inference.
"""

from __future__ import annotations

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate COD sampling indices for one sequence.

    Returns ``(anchor_pos, depth)`` flat tensors of shape ``[total_sampled]``.
    The reference (target) position of each element is ``anchor_pos + depth``.
    """
    loss_mask = loss_mask.flatten()
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = all_valid_indices
    if filter_position_zero:
        # Position 0 has no preceding token; its chain start would be negative and
        # would index document_ids[-1] via negative wraparound, breaking document
        # isolation. Drop it from the depth>=1 candidate pool.
        prev_indices = prev_indices[prev_indices != 0]

    for d in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[
                :sample_size
            ]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]  # restore causal order
        else:
            sampled_idx = prev_indices

        # Next depth's candidate pool: shift by +1 (next-token targets).
        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx - d)  # store the chain start
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(n_per_depth)
        ]
    )
    return anchor_pos, depth
