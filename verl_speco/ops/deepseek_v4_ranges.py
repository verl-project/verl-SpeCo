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

# Adapted from training_common/utils/dsa_source_ranges.py.
"""Recognize exact source layouts; return only CPU integer slice descriptors."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceRanges:
    # (sequence position, source row, length), in ascending sequence order.
    spans: tuple[tuple[int, int, int], ...]


def recognize_source_ranges(
    slots, block_table, kv_lengths, q_lengths, block_size, num_cache_blocks=None
):
    if len(kv_lengths) != len(q_lengths) or not kv_lengths:
        return None
    if any(k < q or q < 0 or k <= 0 for k, q in zip(kv_lengths, q_lengths)):
        return None
    coords = np.asarray(slots, dtype=np.int64)
    if coords.ndim == 1:
        actual = coords
    elif coords.ndim == 2 and coords.shape[1] == 1:
        actual = coords[:, 0]
    elif coords.ndim == 2 and coords.shape[1] == 2:
        if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= block_size):
            return None
        actual = coords[:, 0] * block_size + coords[:, 1]
    else:
        return None
    table = np.asarray(block_table, dtype=np.int64)
    if table.ndim != 2 or len(table) != len(kv_lengths):
        return None
    block_counts = [(k + block_size - 1) // block_size for k in kv_lengths]
    if any(n > table.shape[1] for n in block_counts):
        return None
    blocks = np.concatenate([table[i, :n] for i, n in enumerate(block_counts)])
    # No aliasing or overlapping cache slots across any requests. Otherwise
    # the generic last-write mapping must remain authoritative.
    if np.any(blocks < 0) or len(np.unique(blocks)) != len(blocks):
        return None
    if num_cache_blocks is not None and np.any(blocks >= num_cache_blocks):
        return None
    contexts = [k - q for k, q in zip(kv_lengths, q_lengths)]
    context_total, query_total = sum(contexts), sum(q_lengths)
    candidates = []
    if len(actual) == context_total:
        candidates.append("context_only")
    if len(actual) == query_total:
        candidates.append("query_only")
    if len(actual) == sum(kv_lengths):
        candidates.extend(
            ("sequence_order", "context_then_query", "query_then_context")
        )
    for mode in candidates:
        ranges = []
        co = qo = so = 0
        for c, q, k in zip(contexts, q_lengths, kv_lengths):
            if mode == "context_only":
                spans = ((0, co, c),)
            elif mode == "query_only":
                spans = ((c, qo, q),)
            elif mode == "sequence_order":
                spans = ((0, so, k),)
            elif mode == "context_then_query":
                spans = ((0, co, c), (c, context_total + qo, q))
            else:
                spans = ((0, query_total + co, c), (c, qo, q))
            ranges.append(SourceRanges(tuple(span for span in spans if span[2])))
            co += c
            qo += q
            so += k
        valid = True
        for request, descriptor in enumerate(ranges):
            for logical, source, count in descriptor.spans:
                # One request's temporary integer positions at a time; no
                # cache data, tensor gradients or per-slot Python objects.
                positions = np.arange(logical, logical + count, dtype=np.int64)
                expected = (
                    table[request, positions // block_size] * block_size
                    + positions % block_size
                )
                if not np.array_equal(actual[source : source + count], expected):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return tuple(ranges)
    return None
