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

# Adapted from training_common/utils/dsa_backward.py.
from collections import OrderedDict
from typing import Any

import torch

HostMetadata = tuple[list[int], list[int], list[int] | None]
_DSA_BACKWARD_STRUCTURE_CACHE_LIMIT = 32
_DSA_BACKWARD_STRUCTURE_WORKSPACES: dict[str, _DSABackwardStructureWorkspace] = {}


class _DSABackwardStructureWorkspace:
    """Reuse small shape-only tensors during one analytical backward call."""

    def __init__(self) -> None:
        self._paged_indices: dict[
            tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._kv_head_ids: dict[tuple[str, int, int], torch.Tensor] = {}
        self._range_indices: dict[tuple[str, int, int], torch.Tensor] = {}
        self._lru: OrderedDict[tuple[str, tuple[str, int, int]], None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.contiguous_query_batches = 0

    def _touch(self, kind: str, key: tuple[str, int, int]) -> None:
        cache_key = (kind, key)
        self._lru.pop(cache_key, None)
        self._lru[cache_key] = None

    def _make_room(self) -> None:
        if len(self._lru) < _DSA_BACKWARD_STRUCTURE_CACHE_LIMIT:
            return
        (kind, key), _ = self._lru.popitem(last=False)
        if kind == "paged":
            self._paged_indices.pop(key, None)
        elif kind == "heads":
            self._kv_head_ids.pop(key, None)
        else:
            self._range_indices.pop(key, None)
        self.evictions += 1

    def paged_indices(
        self,
        cache: torch.Tensor,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = int(cache.shape[1])
        key = (str(cache.device), int(sequence_length), block_size)
        cached = self._paged_indices.get(key)
        if cached is not None:
            self.hits += 1
            self._touch("paged", key)
            return cached
        positions = torch.arange(
            sequence_length,
            device=cache.device,
            dtype=torch.long,
        )
        cached = (positions // block_size, positions % block_size)
        self._make_room()
        self._paged_indices[key] = cached
        self._touch("paged", key)
        self.misses += 1
        return cached

    def kv_head_ids(
        self,
        q: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        key = (str(q.device), int(num_q_heads), int(num_kv_heads))
        cached = self._kv_head_ids.get(key)
        if cached is not None:
            self.hits += 1
            self._touch("heads", key)
            return cached
        heads_per_kv = max(1, num_q_heads // num_kv_heads)
        cached = (
            torch.arange(
                num_q_heads,
                device=q.device,
                dtype=torch.long,
            )
            // heads_per_kv
        ).clamp_max(num_kv_heads - 1)
        self._make_room()
        self._kv_head_ids[key] = cached
        self._touch("heads", key)
        self.misses += 1
        return cached

    def range_indices(
        self,
        tensor: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        key = (str(tensor.device), int(length), 0)
        cached = self._range_indices.get(key)
        if cached is not None:
            self.hits += 1
            self._touch("range", key)
            return cached
        cached = torch.arange(
            length,
            device=tensor.device,
            dtype=torch.long,
        )
        self._make_room()
        self._range_indices[key] = cached
        self._touch("range", key)
        self.misses += 1
        return cached


def _generic_paged_kv_sequence_and_update_map(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    batch_index: int,
    sequence_length: int,
    update_source: torch.Tensor | None,
    update_slots: torch.Tensor | None,
    structure_workspace: _DSABackwardStructureWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    block_size = int(cache.shape[1])
    num_logical_blocks = (sequence_length + block_size - 1) // block_size
    physical_blocks = block_table[batch_index, :num_logical_blocks].long()
    sequence = cache.index_select(0, physical_blocks).reshape(
        -1,
        *cache.shape[2:],
    )[:sequence_length]
    if (
        update_source is None
        or update_slots is None
        or update_source.numel() == 0
        or update_slots.numel() == 0
    ):
        return sequence, None

    num_updates = int(update_source.shape[0])
    slot_coordinates = update_slots.reshape(num_updates, -1).long()
    if slot_coordinates.shape[1] == 1:
        slots = slot_coordinates[:, 0]
        update_blocks = torch.div(slots, block_size, rounding_mode="floor")
        update_offsets = torch.remainder(slots, block_size)
    elif slot_coordinates.shape[1] == 2:
        update_blocks = slot_coordinates[:, 0]
        update_offsets = slot_coordinates[:, 1]
    else:
        raise RuntimeError(
            "Unsupported paged-cache slot mapping layout: "
            f"source={tuple(update_source.shape)}, "
            f"slots={tuple(update_slots.shape)}"
        )
    block_matches = physical_blocks.unsqueeze(1) == update_blocks.unsqueeze(0)
    # Apply validity BEFORE nonzero, once. For flat slots remainder() already
    # guarantees offsets in [0, block_size). Two-column slots need validation.
    if slot_coordinates.shape[1] == 2:
        valid_offsets = (update_offsets >= 0) & (update_offsets < block_size)
        block_matches.logical_and_(valid_offsets.unsqueeze(0))
        del valid_offsets
    # With valid offsets, only the final partial block can exceed sequence_length.
    tail_length = sequence_length % block_size
    if tail_length and num_logical_blocks:
        block_matches[-1].logical_and_(update_offsets < tail_length)
    logical_blocks, update_indices = block_matches.nonzero(as_tuple=True)
    matched_offsets = update_offsets.index_select(0, update_indices)
    logical_indices = logical_blocks * block_size + matched_offsets
    update_map = torch.full(
        (sequence_length,),
        -1,
        dtype=torch.long,
        device=cache.device,
    )
    updates = update_source.reshape(
        num_updates,
        *cache.shape[2:],
    )
    # sequence owns the index_select allocation; no cache/source is mutated.
    # Keep the out-of-place path for callers that record an autograd graph.
    replacement = updates.index_select(0, update_indices)
    if torch.is_grad_enabled():
        sequence = sequence.index_copy(0, logical_indices, replacement)
    else:
        sequence.index_copy_(0, logical_indices, replacement)
    update_map.index_copy_(0, logical_indices, update_indices)
    return sequence, update_map


import itertools
import os
from contextvars import ContextVar
from functools import wraps

from verl_speco.ops.deepseek_v4_ranges import SourceRanges, recognize_source_ranges

_DIRECT_RANGE_CONTEXT = ContextVar("direct_dsa_source_ranges", default=None)
_DIRECT_RANGE_STATS = {"recognized": 0, "fallback": 0, "maps": 0, "writebacks": 0}


def _with_direct_source_ranges(function):
    @wraps(function)
    def run(*args, **kwargs):
        def argument(index, name):
            return args[index] if len(args) > index else kwargs.get(name)

        state = None
        if (
            os.environ.get("MTP_DS4_DSA_DIRECT_RANGES", "1") == "1"
            and not torch.is_grad_enabled()
        ):
            op_kwargs = argument(6, "op_kwargs")
            metadata = argument(7, "host_metadata")
            source = argument(2, "ori_source")
            slots = argument(3, "ori_slots")
            if (
                source is not None
                and slots is not None
                and source.numel()
                and metadata is not None
            ):
                state = {
                    "cache": op_kwargs.get("ori_kv"),
                    "table": op_kwargs.get("ori_block_table"),
                    "source": source,
                    "slots": slots,
                    "metadata": metadata,
                    "checked": False,
                    "ranges": None,
                }
        token = _DIRECT_RANGE_CONTEXT.set(state)
        try:
            return function(*args, **kwargs)
        finally:
            _DIRECT_RANGE_CONTEXT.reset(token)

    return run


def _paged_kv_sequence_and_update_map(
    cache,
    block_table,
    batch_index,
    sequence_length,
    update_source,
    update_slots,
    structure_workspace=None,
):
    state = _DIRECT_RANGE_CONTEXT.get()
    applicable = (
        state is not None
        and cache is state["cache"]
        and block_table is state["table"]
        and update_source is state["source"]
        and update_slots is state["slots"]
    )
    if applicable and not state["checked"]:
        state["checked"] = True
        q_offsets, kv_lengths, _ = state["metadata"]
        q_lengths = [b - a for a, b in itertools.pairwise(q_offsets)]
        state["ranges"] = recognize_source_ranges(
            update_slots.reshape(int(update_source.shape[0]), -1)
            .detach()
            .cpu()
            .numpy(),
            block_table.detach().cpu().numpy(),
            kv_lengths,
            q_lengths,
            int(cache.shape[1]),
            num_cache_blocks=int(cache.shape[0]),
        )
        key = "recognized" if state["ranges"] is not None else "fallback"
        _DIRECT_RANGE_STATS[key] += 1
        _checks = _DIRECT_RANGE_STATS["recognized"] + _DIRECT_RANGE_STATS["fallback"]
    if (
        not applicable
        or state["ranges"] is None
        or sequence_length != state["metadata"][1][batch_index]
    ):
        return _generic_paged_kv_sequence_and_update_map(
            cache,
            block_table,
            batch_index,
            sequence_length,
            update_source,
            update_slots,
            structure_workspace,
        )
    mapping = state["ranges"][batch_index]
    updates = update_source.reshape(int(update_source.shape[0]), *cache.shape[2:])
    covered = 0
    for logical, source, count in mapping.spans:
        if logical != covered:
            break
        covered += count
    if covered == sequence_length:
        # Every value would be replaced by source anyway. Reuse its read-only
        # view for one span, or concatenate two spans instead of loading cache.
        pieces = [
            updates.narrow(0, source, count) for _, source, count in mapping.spans
        ]
        sequence = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    else:
        block_size = int(cache.shape[1])
        physical_blocks = block_table[
            batch_index, : (sequence_length + block_size - 1) // block_size
        ].long()
        sequence = cache.index_select(0, physical_blocks).reshape(-1, *cache.shape[2:])[
            :sequence_length
        ]
        for logical, source, count in mapping.spans:
            sequence.narrow(0, logical, count).copy_(updates.narrow(0, source, count))
    _DIRECT_RANGE_STATS["maps"] += 1
    return sequence, mapping


def _collapse_query_head_gradients(
    gradient: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    num_q_heads, selected_length, head_dim = gradient.shape
    heads_per_kv = max(1, num_q_heads // num_kv_heads)
    if num_q_heads == num_kv_heads * heads_per_kv:
        return (
            gradient.reshape(
                num_kv_heads,
                heads_per_kv,
                selected_length,
                head_dim,
            )
            .sum(dim=1)
            .transpose(0, 1)
        )

    kv_head_ids = (
        torch.arange(
            num_q_heads,
            device=gradient.device,
            dtype=torch.long,
        )
        // heads_per_kv
    ).clamp_max(num_kv_heads - 1)
    collapsed = gradient.new_zeros(
        selected_length,
        num_kv_heads,
        head_dim,
    )
    collapsed.index_add_(1, kv_head_ids, gradient.transpose(0, 1))
    return collapsed


def _collapse_batched_query_head_gradients(
    gradient: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    batch_size, num_q_heads, selected_length, head_dim = gradient.shape
    heads_per_kv = max(1, num_q_heads // num_kv_heads)
    if num_q_heads == num_kv_heads * heads_per_kv:
        return (
            gradient.reshape(
                batch_size,
                num_kv_heads,
                heads_per_kv,
                selected_length,
                head_dim,
            )
            .sum(dim=2)
            .permute(0, 2, 1, 3)
        )
    return torch.stack(
        [_collapse_query_head_gradients(row, num_kv_heads) for row in gradient],
        dim=0,
    )


def _explicit_attention_batch_backward(
    queries: torch.Tensor,
    selected: torch.Tensor,
    sinks: torch.Tensor | None,
    output_gradients: torch.Tensor,
    softmax_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    selected_float = selected.float()
    query_float = queries.float()
    scores = (
        torch.einsum(
            "bhd,bhkd->bhk",
            query_float,
            selected_float,
        )
        * softmax_scale
    )
    has_sinks = sinks is not None and sinks.numel() > 0
    if has_sinks:
        sink_logits = (
            sinks.float()
            .reshape(1, scores.shape[1], 1)
            .expand(
                scores.shape[0],
                -1,
                -1,
            )
        )
        probabilities_all = torch.softmax(
            torch.cat((scores, sink_logits), dim=-1),
            dim=-1,
        )
        probabilities = probabilities_all[..., :-1]
    else:
        probabilities_all = None
        probabilities = torch.softmax(scores, dim=-1)

    output_gradients_float = output_gradients.float()
    grad_probabilities = torch.einsum(
        "bhd,bhkd->bhk",
        output_gradients_float,
        selected_float,
    )
    grad_sinks = None
    if probabilities_all is not None:
        grad_probabilities_all = torch.cat(
            (
                grad_probabilities,
                torch.zeros_like(grad_probabilities[..., :1]),
            ),
            dim=-1,
        )
        centered = grad_probabilities_all - (
            probabilities_all * grad_probabilities_all
        ).sum(dim=-1, keepdim=True)
        grad_logits = probabilities_all * centered
        grad_scores = grad_logits[..., :-1]
        grad_sinks = grad_logits[..., -1]
    else:
        centered = grad_probabilities - (probabilities * grad_probabilities).sum(
            dim=-1, keepdim=True
        )
        grad_scores = probabilities * centered

    grad_queries = torch.einsum(
        "bhk,bhkd->bhd",
        grad_scores,
        selected_float,
    ).mul(softmax_scale)
    if torch.is_grad_enabled():
        grad_selected = (
            probabilities.unsqueeze(-1) * output_gradients_float.unsqueeze(2)
            + grad_scores.unsqueeze(-1) * query_float.unsqueeze(2) * softmax_scale
        ).to(selected.dtype)
    else:
        # Preserve FP32 operations and their order; reuse only fresh buffers.
        grad_selected = probabilities.unsqueeze(-1) * output_gradients_float.unsqueeze(
            2
        )
        grad_selected_key = grad_scores.unsqueeze(-1) * query_float.unsqueeze(2)
        grad_selected_key.mul_(softmax_scale)
        grad_selected.add_(grad_selected_key)
        del grad_selected_key
        grad_selected = grad_selected.to(selected.dtype)
    return grad_queries, grad_sinks, grad_selected


_SHARED_KV_STATS = {"batches": 0}


def _shared_attention_batch_backward(
    queries: torch.Tensor,
    selected: torch.Tensor,
    sinks: torch.Tensor | None,
    output_gradients: torch.Tensor,
    softmax_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    # selected is [batch, KV heads, rows, dim]; no query-head expansion.
    _SHARED_KV_STATS["batches"] += 1
    batch, heads, dim = queries.shape
    kv_heads = selected.shape[1]
    groups = heads // kv_heads
    rows = selected.shape[2]
    selected_float = selected.float()
    query_float = queries.float()
    scores = (
        torch.matmul(
            query_float.reshape(batch, kv_heads, groups, dim),
            selected_float.transpose(-1, -2),
        ).reshape(batch, heads, rows)
        * softmax_scale
    )
    has_sinks = sinks is not None and sinks.numel() > 0
    if has_sinks:
        sink_logits = (
            sinks.float()
            .reshape(1, scores.shape[1], 1)
            .expand(
                scores.shape[0],
                -1,
                -1,
            )
        )
        probabilities_all = torch.softmax(
            torch.cat((scores, sink_logits), dim=-1),
            dim=-1,
        )
        probabilities = probabilities_all[..., :-1]
    else:
        probabilities_all = None
        probabilities = torch.softmax(scores, dim=-1)

    output_gradients_float = output_gradients.float()
    grad_probabilities = torch.matmul(
        output_gradients_float.reshape(batch, kv_heads, groups, dim),
        selected_float.transpose(-1, -2),
    ).reshape(batch, heads, rows)
    grad_sinks = None
    if probabilities_all is not None:
        grad_probabilities_all = torch.cat(
            (
                grad_probabilities,
                torch.zeros_like(grad_probabilities[..., :1]),
            ),
            dim=-1,
        )
        centered = grad_probabilities_all - (
            probabilities_all * grad_probabilities_all
        ).sum(dim=-1, keepdim=True)
        grad_logits = probabilities_all * centered
        grad_scores = grad_logits[..., :-1]
        grad_sinks = grad_logits[..., -1]
    else:
        centered = grad_probabilities - (probabilities * grad_probabilities).sum(
            dim=-1, keepdim=True
        )
        grad_scores = probabilities * centered

    grad_queries = (
        torch.matmul(
            grad_scores.reshape(batch, kv_heads, groups, rows),
            selected_float,
        )
        .reshape(batch, heads, dim)
        .mul(softmax_scale)
    )
    if torch.is_grad_enabled():
        grad_selected = (
            probabilities.unsqueeze(-1) * output_gradients_float.unsqueeze(2)
            + grad_scores.unsqueeze(-1) * query_float.unsqueeze(2) * softmax_scale
        ).to(selected.dtype)
    else:
        # Preserve FP32 operations and their order; reuse only fresh buffers.
        grad_selected = probabilities.unsqueeze(-1) * output_gradients_float.unsqueeze(
            2
        )
        grad_selected_key = grad_scores.unsqueeze(-1) * query_float.unsqueeze(2)
        grad_selected_key.mul_(softmax_scale)
        grad_selected.add_(grad_selected_key)
        del grad_selected_key
        grad_selected = grad_selected.to(selected.dtype)
    return grad_queries, grad_sinks, grad_selected


def _accumulate_sequence_update_gradient(
    accumulated: torch.Tensor | None,
    sequence_gradient: torch.Tensor,
    update_map: torch.Tensor | None,
    update_source: torch.Tensor | None,
) -> torch.Tensor | None:
    if accumulated is None or update_map is None or update_source is None:
        return accumulated
    if isinstance(update_map, SourceRanges):
        flat = accumulated.reshape(
            int(update_source.shape[0]), *sequence_gradient.shape[1:]
        )
        for logical, source, count in update_map.spans:
            flat.narrow(0, source, count).add_(
                sequence_gradient.narrow(0, logical, count).to(accumulated.dtype)
            )
        _DIRECT_RANGE_STATS["writebacks"] += 1
        return accumulated
    logical_indices = (update_map >= 0).nonzero(as_tuple=False).squeeze(-1)
    update_indices = update_map.index_select(0, logical_indices)
    accumulated.reshape(
        int(update_source.shape[0]),
        *sequence_gradient.shape[1:],
    ).index_add_(
        0,
        update_indices,
        sequence_gradient.index_select(0, logical_indices).to(accumulated.dtype),
    )
    return accumulated


@_with_direct_source_ranges
def explicit_shared_kv_attention_backward(
    q: torch.Tensor,
    sinks: torch.Tensor | None,
    ori_source: torch.Tensor | None,
    ori_slots: torch.Tensor | None,
    cmp_source: torch.Tensor | None,
    cmp_slots: torch.Tensor | None,
    op_kwargs: dict[str, Any],
    host_metadata: HostMetadata,
    grad_output: torch.Tensor,
    chunk_size: int,
    selected_kv_row_budget: int,
    max_queries_per_batch: int,
    *,
    need_q: bool,
    need_sinks: bool,
    need_ori_source: bool,
    need_cmp_source: bool,
    reuse_temporary_structures: bool = True,
    group_queries_by_kv_length: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if selected_kv_row_budget <= 0:
        raise ValueError(
            f"selected_kv_row_budget must be positive, got {selected_kv_row_budget}"
        )
    if max_queries_per_batch <= 0:
        raise ValueError(
            f"max_queries_per_batch must be positive, got {max_queries_per_batch}"
        )
    if op_kwargs.get("layout_q", "BSND") != "TND":
        raise RuntimeError("Explicit DSA backward requires layout_q='TND'")
    if op_kwargs.get("layout_kv", "PA_ND") != "PA_ND":
        raise RuntimeError("Explicit DSA backward requires layout_kv='PA_ND'")

    ori_kv = op_kwargs.get("ori_kv")
    ori_block_table = op_kwargs.get("ori_block_table")
    if not isinstance(ori_kv, torch.Tensor) or not isinstance(
        ori_block_table,
        torch.Tensor,
    ):
        raise RuntimeError("Explicit DSA backward requires paged ori_kv metadata")  # noqa: TRY004 -- preserve the established numerical validation exception contract
    cmp_kv = op_kwargs.get("cmp_kv")
    cmp_block_table = op_kwargs.get("cmp_block_table")
    cmp_sparse_indices = op_kwargs.get("cmp_sparse_indices")

    q_offsets, kv_lengths, cmp_offsets = host_metadata
    softmax_scale = float(op_kwargs.get("softmax_scale", 0.0))
    cmp_ratio = int(op_kwargs.get("cmp_ratio", 0))
    ori_win_left = int(op_kwargs.get("ori_win_left", 128))
    ori_win_right = int(op_kwargs.get("ori_win_right", 0))
    num_q_heads = int(q.shape[1])
    total_queries = int(q.shape[0])

    grad_q = torch.zeros_like(q) if need_q else None
    grad_sinks = torch.zeros_like(sinks) if need_sinks and sinks is not None else None
    grad_ori_source = (
        torch.zeros_like(ori_source)
        if need_ori_source and ori_source is not None
        else None
    )
    grad_cmp_source = (
        torch.zeros_like(cmp_source)
        if need_cmp_source and cmp_source is not None
        else None
    )
    query_chunks = 0
    query_batches = 0
    max_batched_queries = 0
    max_batched_selected_kv_rows = 0
    workspace_key = str(q.device)
    structure_workspace = (
        _DSA_BACKWARD_STRUCTURE_WORKSPACES.setdefault(
            workspace_key,
            _DSABackwardStructureWorkspace(),
        )
        if reuse_temporary_structures
        else _DSABackwardStructureWorkspace()
    )
    initial_structure_hits = structure_workspace.hits
    initial_structure_misses = structure_workspace.misses
    initial_structure_evictions = structure_workspace.evictions
    initial_contiguous_query_batches = structure_workspace.contiguous_query_batches

    for chunk_start in range(0, total_queries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_queries)
        chunk_sink_gradient = (
            torch.zeros_like(sinks)
            if grad_sinks is not None and sinks is not None
            else None
        )

        QueryDescriptor = tuple[
            int,
            int,
            int,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            int,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        grouped_queries: dict[tuple[int, int, int], list[QueryDescriptor]] = {}
        deferred_sequence_updates: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []

        def flush_query_batch(
            query_descriptors: list[QueryDescriptor],
            chunk_sink_gradient=chunk_sink_gradient,
        ) -> None:
            nonlocal query_batches
            nonlocal max_batched_queries
            nonlocal max_batched_selected_kv_rows
            if not query_descriptors:
                return
            batch_queries = len(query_descriptors)
            ori_selected_length = query_descriptors[0][2] - query_descriptors[0][1]
            sparse_selected_length = (
                int(query_descriptors[0][3].numel())
                if query_descriptors[0][3] is not None
                else 0
            )
            ori_selected_batch = torch.stack(
                [
                    row[4].narrow(0, row[1], ori_selected_length)
                    for row in query_descriptors
                ],
                dim=0,
            )
            if sparse_selected_length > 0:
                cmp_groups: dict[
                    int,
                    tuple[torch.Tensor, list[int], list[torch.Tensor]],
                ] = {}
                for row_index, descriptor in enumerate(query_descriptors):
                    sparse_ids = descriptor[3]
                    cmp_sequence = descriptor[5]
                    if sparse_ids is None or cmp_sequence is None:
                        raise RuntimeError(
                            "Batched DSA descriptor lost compressed-KV metadata"
                        )
                    group = cmp_groups.get(id(cmp_sequence))
                    if group is None:
                        group = (cmp_sequence, [], [])
                        cmp_groups[id(cmp_sequence)] = group
                    group[1].append(row_index)
                    group[2].append(sparse_ids)

                cmp_selected_batch = None
                for cmp_sequence, row_indices, sparse_indices in cmp_groups.values():
                    sparse_index_batch = torch.stack(sparse_indices, dim=0)
                    selected = cmp_sequence.index_select(
                        0,
                        sparse_index_batch.reshape(-1),
                    ).reshape(
                        len(row_indices),
                        sparse_selected_length,
                        *cmp_sequence.shape[1:],
                    )
                    if len(cmp_groups) == 1 and len(row_indices) == batch_queries:
                        cmp_selected_batch = selected
                        break
                    if cmp_selected_batch is None:
                        cmp_selected_batch = cmp_sequence.new_empty(
                            (
                                batch_queries,
                                sparse_selected_length,
                                *cmp_sequence.shape[1:],
                            )
                        )
                    row_index_tensor = torch.tensor(
                        row_indices,
                        dtype=torch.long,
                        device=cmp_sequence.device,
                    )
                    cmp_selected_batch.index_copy_(
                        0,
                        row_index_tensor,
                        selected,
                    )
                if cmp_selected_batch is None:
                    raise RuntimeError(
                        "Batched DSA compressed-KV gather produced no rows"
                    )
                selected_kv_batch = torch.cat(
                    (ori_selected_batch, cmp_selected_batch),
                    dim=1,
                )
            else:
                selected_kv_batch = ori_selected_batch
            kv_head_ids = query_descriptors[0][6]
            num_kv_heads = query_descriptors[0][7]
            shared_math = (
                os.environ.get("MTP_DS4_DSA_SHARED_KV", "1") == "1"
                and q.shape[1] > num_kv_heads
                and q.shape[1] % num_kv_heads == 0
            )
            if shared_math:
                selected_batch = selected_kv_batch.permute(0, 2, 1, 3)
            else:
                selected_batch = selected_kv_batch.index_select(
                    2,
                    kv_head_ids,
                ).permute(0, 2, 1, 3)
            attention_backward = (
                _shared_attention_batch_backward
                if shared_math
                else _explicit_attention_batch_backward
            )
            query_indices_list = [row[0] for row in query_descriptors]
            query_start_index = query_indices_list[0]
            contiguous_queries = all(
                query_index == query_start_index + offset
                for offset, query_index in enumerate(query_indices_list)
            )
            if reuse_temporary_structures and contiguous_queries:
                query_batch = q.narrow(0, query_start_index, batch_queries)
                grad_output_batch = grad_output.narrow(
                    0,
                    query_start_index,
                    batch_queries,
                )
            else:
                query_indices = torch.tensor(
                    query_indices_list,
                    dtype=torch.long,
                    device=q.device,
                )
                query_batch = q.index_select(0, query_indices)
                grad_output_batch = grad_output.index_select(
                    0,
                    query_indices,
                )
            (
                batched_grad_q,
                batched_grad_sinks,
                batched_grad_selected,
            ) = attention_backward(
                query_batch,
                selected_batch,
                sinks,
                grad_output_batch,
                softmax_scale,
            )
            if grad_q is not None:
                if reuse_temporary_structures and contiguous_queries:
                    grad_q.narrow(
                        0,
                        query_start_index,
                        batch_queries,
                    ).copy_(batched_grad_q.to(grad_q.dtype))
                else:
                    grad_q.index_copy_(
                        0,
                        query_indices,
                        batched_grad_q.to(grad_q.dtype),
                    )
            if reuse_temporary_structures and contiguous_queries:
                structure_workspace.contiguous_query_batches += 1
            if (
                chunk_sink_gradient is not None
                and batched_grad_sinks is not None
                and sinks is not None
            ):
                chunk_sink_gradient.add_(
                    batched_grad_sinks.reshape(
                        batch_queries,
                        *sinks.shape,
                    )
                    .sum(dim=0)
                    .to(chunk_sink_gradient.dtype)
                )

            collapsed = _collapse_batched_query_head_gradients(
                batched_grad_selected,
                num_kv_heads,
            )
            ori_update_groups: dict[
                tuple[int, int],
                tuple[torch.Tensor, list[int], list[int]],
            ] = {}
            cmp_update_groups: dict[
                int,
                tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]],
            ] = {}
            for row_index, descriptor in enumerate(query_descriptors):
                (
                    _,
                    ori_start,
                    ori_end,
                    sparse_ids,
                    _,
                    _,
                    _,
                    _,
                    ori_sequence_gradient,
                    cmp_sequence_gradient,
                ) = descriptor
                ori_selected_length = ori_end - ori_start
                if ori_sequence_gradient is not None:
                    group_key = (
                        id(ori_sequence_gradient),
                        ori_selected_length,
                    )
                    group = ori_update_groups.get(group_key)
                    if group is None:
                        group = (ori_sequence_gradient, [], [])
                        ori_update_groups[group_key] = group
                    group[1].append(row_index)
                    group[2].append(ori_start)
                if (
                    cmp_sequence_gradient is not None
                    and sparse_ids is not None
                    and sparse_ids.numel() > 0
                ):
                    group_key = id(cmp_sequence_gradient)
                    group = cmp_update_groups.get(group_key)
                    if group is None:
                        group = (cmp_sequence_gradient, [], [])
                        cmp_update_groups[group_key] = group
                    group[1].append(sparse_ids)
                    group[2].append(collapsed[row_index, ori_selected_length:])

            for (_, ori_selected_length), (
                ori_sequence_gradient,
                row_indices,
                ori_starts,
            ) in ori_update_groups.items():
                start_tensor = torch.tensor(
                    ori_starts,
                    dtype=torch.long,
                    device=collapsed.device,
                )
                offsets = (
                    structure_workspace.range_indices(
                        collapsed,
                        ori_selected_length,
                    )
                    if reuse_temporary_structures
                    else torch.arange(
                        ori_selected_length,
                        dtype=torch.long,
                        device=collapsed.device,
                    )
                )
                destination_indices = (
                    start_tensor.unsqueeze(1) + offsets.unsqueeze(0)
                ).reshape(-1)
                # Select original KV before gathering rows: do not copy the
                # compressed-KV tail only to discard it. Contiguous groups
                # (especially one anchor per group) need no row-index tensor.
                original_gradients = collapsed[:, :ori_selected_length]
                first_row = row_indices[0]
                if all(
                    row == first_row + offset for offset, row in enumerate(row_indices)
                ):
                    source_gradients = original_gradients.narrow(
                        0, first_row, len(row_indices)
                    )
                else:
                    row_index_tensor = torch.tensor(
                        row_indices,
                        dtype=torch.long,
                        device=collapsed.device,
                    )
                    source_gradients = original_gradients.index_select(
                        0, row_index_tensor
                    )
                source_gradients = source_gradients.reshape(-1, *collapsed.shape[2:])
                ori_sequence_gradient.index_add_(
                    0,
                    destination_indices,
                    source_gradients,
                )

            for (
                cmp_sequence_gradient,
                sparse_indices,
                source_gradients,
            ) in cmp_update_groups.values():
                cmp_sequence_gradient.index_add_(
                    0,
                    torch.cat(sparse_indices, dim=0),
                    torch.cat(source_gradients, dim=0),
                )
            query_batches += 1
            batch_rows = batch_queries * int(selected_kv_batch.shape[1])
            max_batched_queries = max(max_batched_queries, batch_queries)
            max_batched_selected_kv_rows = max(
                max_batched_selected_kv_rows,
                batch_rows,
            )
            query_descriptors.clear()

        for batch_index in range(len(q_offsets) - 1):
            q_start = int(q_offsets[batch_index])
            q_end = int(q_offsets[batch_index + 1])
            selected_query_start = max(q_start, chunk_start)
            selected_query_end = min(q_end, chunk_end)
            if selected_query_start >= selected_query_end:
                continue

            q_length = q_end - q_start
            kv_length = int(kv_lengths[batch_index])
            ori_sequence, ori_update_map = _paged_kv_sequence_and_update_map(
                ori_kv,
                ori_block_table,
                batch_index,
                kv_length,
                ori_source,
                ori_slots,
                structure_workspace if reuse_temporary_structures else None,
            )
            ori_sequence_gradient = (
                torch.zeros_like(ori_sequence) if grad_ori_source is not None else None
            )
            num_kv_heads = int(ori_sequence.shape[1])
            if reuse_temporary_structures:
                kv_head_ids = structure_workspace.kv_head_ids(
                    q,
                    num_q_heads,
                    num_kv_heads,
                )
            else:
                heads_per_kv = max(1, num_q_heads // num_kv_heads)
                kv_head_ids = (
                    torch.arange(
                        num_q_heads,
                        device=q.device,
                        dtype=torch.long,
                    )
                    // heads_per_kv
                ).clamp_max(num_kv_heads - 1)
            query_position_start = kv_length - q_length

            cmp_sequence = None
            cmp_update_map = None
            cmp_sequence_gradient = None
            cmp_length = 0
            if isinstance(cmp_kv, torch.Tensor) and isinstance(
                cmp_block_table,
                torch.Tensor,
            ):
                if cmp_offsets is not None and len(cmp_offsets) == len(q_offsets):
                    cmp_length = int(
                        cmp_offsets[batch_index + 1] - cmp_offsets[batch_index]
                    )
                elif cmp_ratio > 0:
                    cmp_length = kv_length // cmp_ratio
                if cmp_length > 0:
                    cmp_sequence, cmp_update_map = _paged_kv_sequence_and_update_map(
                        cmp_kv,
                        cmp_block_table,
                        batch_index,
                        cmp_length,
                        cmp_source,
                        cmp_slots,
                        (structure_workspace if reuse_temporary_structures else None),
                    )
                    if grad_cmp_source is not None:
                        cmp_sequence_gradient = torch.zeros_like(cmp_sequence)

            pending_queries: list[QueryDescriptor] = []

            for global_query_index in range(
                selected_query_start,
                selected_query_end,
            ):
                local_query_index = global_query_index - q_start
                query_position = query_position_start + local_query_index
                ori_start = max(0, query_position - ori_win_left)
                ori_end = min(
                    kv_length,
                    query_position + ori_win_right + 1,
                )
                sparse_ids = None
                if cmp_sequence is not None and isinstance(
                    cmp_sparse_indices,
                    torch.Tensor,
                ):
                    sparse_ids = cmp_sparse_indices[global_query_index]
                    if sparse_ids.ndim == 2:
                        sparse_ids = sparse_ids[0]
                    sparse_ids = sparse_ids.long()
                    sparse_ids = sparse_ids[
                        (sparse_ids >= 0) & (sparse_ids < cmp_length)
                    ]
                ori_selected_length = ori_end - ori_start
                sparse_selected_length = (
                    int(sparse_ids.numel()) if sparse_ids is not None else 0
                )
                selected_length = ori_selected_length + sparse_selected_length
                descriptor = (
                    global_query_index,
                    ori_start,
                    ori_end,
                    sparse_ids,
                    ori_sequence,
                    cmp_sequence,
                    kv_head_ids,
                    num_kv_heads,
                    ori_sequence_gradient,
                    cmp_sequence_gradient,
                )
                if group_queries_by_kv_length:
                    bucket = grouped_queries.setdefault(
                        (
                            ori_selected_length,
                            sparse_selected_length,
                            num_kv_heads,
                        ),
                        [],
                    )
                    bucket.append(descriptor)
                    query_capacity = max(
                        1,
                        min(
                            max_queries_per_batch,
                            selected_kv_row_budget // selected_length,
                        ),
                    )
                    if len(bucket) >= query_capacity:
                        flush_query_batch(bucket)
                else:
                    pending_length = (
                        (
                            pending_queries[0][2]
                            - pending_queries[0][1]
                            + (
                                int(pending_queries[0][3].numel())
                                if pending_queries[0][3] is not None
                                else 0
                            )
                        )
                        if pending_queries
                        else selected_length
                    )
                    would_exceed_budget = (
                        len(pending_queries) + 1
                    ) * selected_length > selected_kv_row_budget
                    if pending_queries and (
                        selected_length != pending_length
                        or len(pending_queries) >= max_queries_per_batch
                        or would_exceed_budget
                    ):
                        flush_query_batch(pending_queries)
                    pending_queries.append(descriptor)
            if group_queries_by_kv_length:
                deferred_sequence_updates.append(
                    (
                        ori_sequence_gradient,
                        ori_update_map,
                        ori_source,
                        cmp_sequence_gradient,
                        cmp_update_map,
                        cmp_source,
                    )
                )
            else:
                flush_query_batch(pending_queries)
                if ori_sequence_gradient is not None:
                    _accumulate_sequence_update_gradient(
                        grad_ori_source,
                        ori_sequence_gradient,
                        ori_update_map,
                        ori_source,
                    )
                if cmp_sequence_gradient is not None:
                    _accumulate_sequence_update_gradient(
                        grad_cmp_source,
                        cmp_sequence_gradient,
                        cmp_update_map,
                        cmp_source,
                    )

        if group_queries_by_kv_length:
            for bucket in grouped_queries.values():
                flush_query_batch(bucket)
            for (
                ori_sequence_gradient,
                ori_update_map,
                ori_source_for_update,
                cmp_sequence_gradient,
                cmp_update_map,
                cmp_source_for_update,
            ) in deferred_sequence_updates:
                if ori_sequence_gradient is not None:
                    _accumulate_sequence_update_gradient(
                        grad_ori_source,
                        ori_sequence_gradient,
                        ori_update_map,
                        ori_source_for_update,
                    )
                if cmp_sequence_gradient is not None:
                    _accumulate_sequence_update_gradient(
                        grad_cmp_source,
                        cmp_sequence_gradient,
                        cmp_update_map,
                        cmp_source_for_update,
                    )

        if grad_sinks is not None and chunk_sink_gradient is not None:
            grad_sinks.add_(chunk_sink_gradient)
        query_chunks += 1

    return (
        grad_q,
        grad_sinks,
        grad_ori_source,
        grad_cmp_source,
        query_chunks,
        query_batches,
        max_batched_queries,
        max_batched_selected_kv_rows,
        structure_workspace.hits - initial_structure_hits,
        structure_workspace.misses - initial_structure_misses,
        structure_workspace.contiguous_query_batches - initial_contiguous_query_batches,
        structure_workspace.evictions - initial_structure_evictions,
    )
