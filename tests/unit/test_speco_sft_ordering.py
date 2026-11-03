# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import torch

from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_SAMPLE_INDICES_KEY,
)


def _simulate_index_restore(
    sample_indices: torch.Tensor,
    ref_meta_map: dict,
    batch_size: int,
) -> list:
    """Simulate the index-restore logic in _speco_collect_sft_features.

    After balance_batch reordering, ``batch_idx`` is the post-reorder row
    position while ``meta["batch_idx"]`` (stored as keys in ``ref_meta_map``)
    is the pre-reorder original index.  ``sample_indices`` records the
    original index for each post-reorder row, so we must look up
    ``ref_meta_map[sample_indices[batch_idx]]`` instead of
    ``ref_meta_map[batch_idx]``.
    """
    matched = []
    for batch_idx in range(batch_size):
        orig_idx = int(sample_indices[batch_idx].item())
        matched.append(ref_meta_map.get(orig_idx))
    return matched


def test_balance_batch_reorder_preserves_token_hidden_alignment() -> None:
    """balance_batch 重排后，token 行与 hidden meta 仍能正确配对。"""
    batch_size = 4
    # 原始样本标识（同时作为 token 内容的代理）
    original_tokens = ["tok_A", "tok_B", "tok_C", "tok_D"]
    # meta["batch_idx"] 是重排前的原始索引，ref_meta_map 的 key 就是原始索引
    ref_meta_map = {i: f"hidden_{original_tokens[i]}" for i in range(batch_size)}

    # 模拟 balance_batch 重排：原始顺序 [0,1,2,3] → 重排后 [2,0,3,1]
    global_idx = torch.tensor([2, 0, 3, 1])

    # SAMPLE_INDICES 在重排前设为 arange，重排后随 data 一起被 index_select，
    # 因此 sample_indices[batch_idx] 保存了重排前行 batch_idx 对应的原始索引
    sample_indices = torch.arange(batch_size)[global_idx]

    # 重排后的 token 行
    reordered_tokens = [original_tokens[i] for i in global_idx.tolist()]

    # 验证：每个重排后的 token 行能通过 sample_indices 找到对应的 hidden
    matched = _simulate_index_restore(sample_indices, ref_meta_map, batch_size)

    for batch_idx in range(batch_size):
        expected_hidden = f"hidden_{reordered_tokens[batch_idx]}"
        assert matched[batch_idx] == expected_hidden, (
            f"batch_idx={batch_idx}: token={reordered_tokens[batch_idx]} "
            f"mapped to hidden={matched[batch_idx]}, expected={expected_hidden}"
        )


def test_no_reorder_keeps_direct_mapping() -> None:
    """未开启 balance_batch 时，sample_indices[i]==i，直接映射即可。"""
    batch_size = 3
    ref_meta_map = {i: f"hidden_{i}" for i in range(batch_size)}
    sample_indices = torch.arange(batch_size)

    matched = _simulate_index_restore(sample_indices, ref_meta_map, batch_size)

    for batch_idx in range(batch_size):
        assert matched[batch_idx] == f"hidden_{batch_idx}"


def test_sample_indices_key_name_stable() -> None:
    """确保 SAMPLE_INDICES 的 key 名称与 driver stamp 逻辑一致。"""
    assert OLD_LOGPROB_SAMPLE_INDICES_KEY == "speco_oldlogprob_sample_indices"
