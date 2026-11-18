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

# Adapted from training_common/utils/cache.py.
import math


def required_training_cache_blocks(
    num_requests: int,
    query_length: int,
    block_size: int,
    *,
    reserve_blocks: int = 1,
) -> int:
    """Return paged-cache blocks needed by one full-sequence training call."""
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if query_length < 0:
        raise ValueError("query_length must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if reserve_blocks < 0:
        raise ValueError("reserve_blocks must be non-negative")

    blocks_per_request = max(1, math.ceil(max(query_length, 1) / block_size))
    return max(1, num_requests * blocks_per_request + reserve_blocks)


def cache_shape_has_capacity(
    actual_shape: tuple[int, ...],
    required_shape: tuple[int, ...],
    *,
    grow_first_dim: bool,
) -> bool:
    """Check exact shape or grow-only capacity along the paged-block axis."""
    if len(actual_shape) != len(required_shape):
        return False
    if not grow_first_dim:
        return actual_shape == required_shape
    if not actual_shape:
        return True
    return (
        actual_shape[0] >= required_shape[0] and actual_shape[1:] == required_shape[1:]
    )
