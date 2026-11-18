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
# Adapted original numerical validation; no external source imports.
import unittest

import torch

from verl_speco.ops.deepseek_v4_dsa import (
    _DSA_BACKWARD_STRUCTURE_WORKSPACES,
    _accumulate_sequence_update_gradient,
    _paged_kv_sequence_and_update_map,
    explicit_shared_kv_attention_backward,
)


def _reference_paged_sequence(
    cache,
    block_table,
    sequence_length,
    source,
    slots,
    batch_index=0,
):
    block_size = cache.shape[1]
    positions = torch.arange(sequence_length)
    physical_blocks = block_table[batch_index].long()[positions // block_size]
    offsets = positions % block_size
    sequence = cache[physical_blocks, offsets]
    if source is None or slots is None or slots.numel() == 0:
        return sequence
    flat_slots = slots[:, 0] * block_size + slots[:, 1]
    physical_slots = physical_blocks * block_size + offsets
    matches = physical_slots.unsqueeze(1) == flat_slots.unsqueeze(0)
    replaced = matches.any(dim=1)
    update_indices = matches[replaced].long().argmax(dim=1)
    return sequence.index_copy(
        0,
        replaced.nonzero(as_tuple=False).squeeze(-1),
        source.index_select(0, update_indices),
    )


def _reference_attention(
    q,
    sinks,
    ori_source,
    ori_slots,
    cmp_source,
    cmp_slots,
    op_kwargs,
    host_metadata,
):
    q_offsets, kv_lengths, cmp_offsets = host_metadata
    outputs = []
    for batch_index in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[batch_index : batch_index + 2]
        kv_length = kv_lengths[batch_index]
        ori_sequence = _reference_paged_sequence(
            op_kwargs["ori_kv"],
            op_kwargs["ori_block_table"],
            kv_length,
            ori_source,
            ori_slots,
            batch_index,
        )
        cmp_sequence = None
        if cmp_source is not None:
            cmp_length = cmp_offsets[1] - cmp_offsets[0]
            cmp_sequence = _reference_paged_sequence(
                op_kwargs["cmp_kv"],
                op_kwargs["cmp_block_table"],
                cmp_length,
                cmp_source,
                cmp_slots,
                batch_index,
            )
        num_q_heads = q.shape[1]
        num_kv_heads = ori_sequence.shape[1]
        kv_head_ids = torch.arange(num_q_heads) // (num_q_heads // num_kv_heads)
        query_position_start = kv_length - (q_end - q_start)
        for query_index in range(q_start, q_end):
            query_position = query_position_start + query_index - q_start
            ori_start = max(0, query_position - op_kwargs["ori_win_left"])
            ori_end = min(
                kv_length,
                query_position + op_kwargs["ori_win_right"] + 1,
            )
            selected = ori_sequence[ori_start:ori_end]
            if cmp_sequence is not None:
                sparse_ids = op_kwargs["cmp_sparse_indices"][query_index]
                sparse_ids = sparse_ids[
                    (sparse_ids >= 0) & (sparse_ids < cmp_sequence.shape[0])
                ]
                selected = torch.cat(
                    (selected, cmp_sequence.index_select(0, sparse_ids)),
                    dim=0,
                )
            selected = selected.index_select(1, kv_head_ids).transpose(0, 1)
            scores = (
                torch.einsum(
                    "hd,hkd->hk",
                    q[query_index].float(),
                    selected.float(),
                )
                * op_kwargs["softmax_scale"]
            )
            if sinks is not None:
                probabilities = torch.softmax(
                    torch.cat((scores, sinks.float()[:, None]), dim=-1),
                    dim=-1,
                )[..., :-1]
            else:
                probabilities = torch.softmax(scores, dim=-1)
            outputs.append(
                torch.einsum(
                    "hk,hkd->hd",
                    probabilities,
                    selected.float(),
                ).to(q.dtype)
            )
    return torch.stack(outputs)


class TestExplicitDSABackward(unittest.TestCase):
    def test_paged_sequence_maps_updates_at_block_granularity(self):
        cache = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
        block_table = torch.tensor([[2, 0, -1]], dtype=torch.int32)
        source = torch.tensor([[[101.0]], [[202.0]], [[303.0]]])
        slots = torch.tensor([[2, 1], [0, 0], [3, 2]])

        sequence, update_map = _paged_kv_sequence_and_update_map(
            cache,
            block_table,
            batch_index=0,
            sequence_length=6,
            update_source=source,
            update_slots=slots,
        )

        expected = torch.tensor([8.0, 101.0, 10.0, 11.0, 202.0, 1.0])
        torch.testing.assert_close(sequence[:, 0, 0], expected)
        self.assertEqual(update_map.tolist(), [-1, 0, -1, -1, 1, -1])

        accumulated = torch.zeros_like(source)
        sequence_gradient = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
        _accumulate_sequence_update_gradient(
            accumulated,
            sequence_gradient,
            update_map,
            source,
        )
        torch.testing.assert_close(
            accumulated[:, 0, 0],
            torch.tensor([1.0, 4.0, 0.0]),
        )

    def test_paged_sequence_accepts_flat_physical_slots(self):
        cache = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1, 1)
        block_table = torch.tensor([[1, 2]], dtype=torch.int32)
        source = torch.tensor([[[41.0]], [[82.0]]])
        slots = torch.tensor([[5], [10]])

        sequence, update_map = _paged_kv_sequence_and_update_map(
            cache,
            block_table,
            batch_index=0,
            sequence_length=7,
            update_source=source,
            update_slots=slots,
        )

        expected = torch.tensor([4.0, 41.0, 6.0, 7.0, 8.0, 9.0, 82.0])
        torch.testing.assert_close(sequence[:, 0, 0], expected)
        self.assertEqual(update_map.tolist(), [-1, 0, -1, -1, -1, -1, 1])

    def _run_case(
        self,
        *,
        with_compressed_cache,
        chunk_size,
        dtype=torch.float32,
        ori_win_left=2,
        selected_kv_row_budget=16,
        max_queries_per_batch=4,
        reuse_temporary_structures=True,
        group_queries_by_kv_length=False,
    ):
        _DSA_BACKWARD_STRUCTURE_WORKSPACES.clear()
        generator = torch.Generator().manual_seed(20260730)
        q = torch.randn(
            3,
            4,
            3,
            generator=generator,
            dtype=dtype,
            requires_grad=True,
        )
        sinks = torch.randn(4, generator=generator, requires_grad=True)
        ori_source = torch.randn(
            4,
            2,
            3,
            generator=generator,
            dtype=dtype,
            requires_grad=True,
        )
        ori_slots = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]])
        ori_cache = torch.randn(
            1,
            4,
            2,
            3,
            generator=generator,
            dtype=dtype,
        )
        cmp_source = None
        cmp_slots = None
        cmp_cache = None
        cmp_offsets = None
        cmp_sparse_indices = None
        if with_compressed_cache:
            cmp_source = torch.randn(
                2,
                2,
                3,
                generator=generator,
                dtype=dtype,
                requires_grad=True,
            )
            cmp_slots = torch.tensor([[0, 0], [0, 1]])
            cmp_cache = torch.randn(
                1,
                4,
                2,
                3,
                generator=generator,
                dtype=dtype,
            )
            cmp_offsets = [0, 2]
            cmp_sparse_indices = torch.tensor([[0, 1], [1, 1], [1, -1]])
        op_kwargs = {
            "ori_kv": ori_cache,
            "ori_block_table": torch.tensor([[0]], dtype=torch.int32),
            "cmp_kv": cmp_cache,
            "cmp_block_table": (
                torch.tensor([[0]], dtype=torch.int32)
                if with_compressed_cache
                else None
            ),
            "cmp_sparse_indices": cmp_sparse_indices,
            "softmax_scale": 0.37,
            "cmp_ratio": 4,
            "ori_win_left": ori_win_left,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        host_metadata = ([0, 3], [4], cmp_offsets)
        grad_output = torch.randn(
            3,
            4,
            3,
            generator=generator,
            dtype=dtype,
        )
        output = _reference_attention(
            q,
            sinks,
            ori_source,
            ori_slots,
            cmp_source,
            cmp_slots,
            op_kwargs,
            host_metadata,
        )
        reference_inputs = [q, sinks, ori_source]
        if cmp_source is not None:
            reference_inputs.append(cmp_source)
        reference_grads = torch.autograd.grad(
            output,
            reference_inputs,
            grad_output,
        )

        actual = explicit_shared_kv_attention_backward(
            q.detach(),
            sinks.detach(),
            ori_source.detach(),
            ori_slots,
            cmp_source.detach() if cmp_source is not None else None,
            cmp_slots,
            op_kwargs,
            host_metadata,
            grad_output,
            chunk_size,
            selected_kv_row_budget=selected_kv_row_budget,
            max_queries_per_batch=max_queries_per_batch,
            need_q=True,
            need_sinks=True,
            need_ori_source=True,
            need_cmp_source=cmp_source is not None,
            reuse_temporary_structures=reuse_temporary_structures,
            group_queries_by_kv_length=group_queries_by_kv_length,
        )
        actual_grads = list(actual[:3])
        if cmp_source is not None:
            actual_grads.append(actual[3])
        for actual_grad, reference_grad in zip(
            actual_grads,
            reference_grads,
        ):
            if dtype == torch.bfloat16:
                torch.testing.assert_close(
                    actual_grad,
                    reference_grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    actual_grad,
                    reference_grad,
                    rtol=2e-5,
                    atol=2e-6,
                )
        self.assertEqual(actual[4], (3 + chunk_size - 1) // chunk_size)
        self.assertLessEqual(actual[6], max_queries_per_batch)
        self.assertLessEqual(
            actual[7],
            max(selected_kv_row_budget, actual[7] if actual[6] == 1 else 0),
        )
        if reuse_temporary_structures:
            self.assertGreater(actual[9], 0)
            if not group_queries_by_kv_length:
                self.assertEqual(actual[10], actual[5])
            else:
                self.assertLessEqual(actual[10], actual[5])
            if chunk_size < 3:
                self.assertGreater(actual[8], 0)
        else:
            self.assertEqual(actual[8:12], (0, 0, 0, 0))
        if not with_compressed_cache and ori_win_left == 0:
            expected_batch_size = min(
                3,
                selected_kv_row_budget,
                max_queries_per_batch,
            )
            expected_batches = (3 + expected_batch_size - 1) // expected_batch_size
            self.assertEqual(actual[5], expected_batches)
            self.assertEqual(actual[6], expected_batch_size)

    def test_without_compressed_cache(self):
        self._run_case(with_compressed_cache=False, chunk_size=2)

    def test_with_compressed_cache_and_duplicate_sparse_ids(self):
        self._run_case(with_compressed_cache=True, chunk_size=1)

    def test_full_query_chunk(self):
        self._run_case(with_compressed_cache=True, chunk_size=3)

    def test_compatible_queries_are_batched(self):
        self._run_case(
            with_compressed_cache=False,
            chunk_size=3,
            ori_win_left=0,
        )

    def test_temporary_structure_reuse_can_be_disabled(self):
        self._run_case(
            with_compressed_cache=True,
            chunk_size=1,
            reuse_temporary_structures=False,
        )

    def test_selected_kv_budget_bounds_batch(self):
        self._run_case(
            with_compressed_cache=False,
            chunk_size=3,
            ori_win_left=0,
            selected_kv_row_budget=2,
            max_queries_per_batch=8,
        )

    def test_bfloat16_compatible_queries_are_batched(self):
        self._run_case(
            with_compressed_cache=False,
            chunk_size=3,
            dtype=torch.bfloat16,
            ori_win_left=0,
        )

    def test_bfloat16_boundaries(self):
        self._run_case(
            with_compressed_cache=True,
            chunk_size=2,
            dtype=torch.bfloat16,
        )

    def test_queries_can_be_grouped_by_selected_kv_length(self):
        self._run_case(
            with_compressed_cache=True,
            chunk_size=3,
            group_queries_by_kv_length=True,
        )

    def test_query_grouping_spans_multiple_requests(self):
        generator = torch.Generator().manual_seed(109)
        q = torch.randn(6, 4, 3, generator=generator, requires_grad=True)
        sinks = torch.randn(4, generator=generator, requires_grad=True)
        ori_cache = torch.randn(2, 4, 2, 3, generator=generator)
        op_kwargs = {
            "ori_kv": ori_cache,
            "ori_block_table": torch.tensor([[0], [1]], dtype=torch.int32),
            "cmp_kv": None,
            "cmp_block_table": None,
            "cmp_sparse_indices": None,
            "softmax_scale": 0.37,
            "cmp_ratio": 4,
            "ori_win_left": 0,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        host_metadata = ([0, 3, 6], [4, 4], None)
        grad_output = torch.randn(6, 4, 3, generator=generator)
        reference = _reference_attention(
            q,
            sinks,
            None,
            None,
            None,
            None,
            op_kwargs,
            host_metadata,
        )
        reference_grads = torch.autograd.grad(
            reference,
            (q, sinks),
            grad_output,
        )
        actual = explicit_shared_kv_attention_backward(
            q.detach(),
            sinks.detach(),
            None,
            None,
            None,
            None,
            op_kwargs,
            host_metadata,
            grad_output,
            chunk_size=6,
            selected_kv_row_budget=16,
            max_queries_per_batch=4,
            need_q=True,
            need_sinks=True,
            need_ori_source=False,
            need_cmp_source=False,
            reuse_temporary_structures=True,
            group_queries_by_kv_length=True,
        )
        torch.testing.assert_close(actual[0], reference_grads[0])
        torch.testing.assert_close(actual[1], reference_grads[1])
        self.assertEqual(actual[5], 2)
        self.assertEqual(actual[6], 4)


if __name__ == "__main__":
    unittest.main()
