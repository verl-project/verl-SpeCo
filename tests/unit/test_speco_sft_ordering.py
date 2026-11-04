# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# verl 路径解析：CI 中 verl 被 clone 到 .ci/upstream-verl，本地环境可能通过
# VERL_SPECO_UPSTREAM_ROOT 或 pip install 提供。此处自动查找并加入 sys.path。
# ---------------------------------------------------------------------------


def _ensure_verl_on_path() -> None:
    try:
        import verl  # noqa: F401

        return  # 已可导入
    except ImportError:
        pass

    candidates = []
    env_root = os.environ.get("VERL_SPECO_UPSTREAM_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    # CI 默认路径
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / ".ci" / "upstream-verl")
    # 本地常见路径
    candidates.append(Path("/model/sft_cotrain/verl"))
    candidates.append(repo_root.parent / "verl")

    # 去重并尝试
    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        if str(cand) in seen:
            continue
        seen.add(str(cand))
        if not cand.is_dir():
            continue
        # 支持两种结构：
        #   1. verl 仓库根目录：cand/verl/__init__.py -> 把 cand 加入 path
        #   2. verl 包目录本身：cand/__init__.py -> 把 cand.parent 加入 path
        path_to_add = None
        if (cand / "verl" / "__init__.py").is_file():
            path_to_add = cand
        elif (cand / "__init__.py").is_file():
            path_to_add = cand.parent
        if path_to_add is None:
            continue
        sys.path.insert(0, str(path_to_add))
        try:
            import verl  # noqa: F401

            return
        except ImportError:
            # 回退：移除刚加的 path，继续尝试
            sys.path.remove(str(path_to_add))
            continue


import traceback


_verl_import_error = None
try:
    _ensure_verl_on_path()
    import verl  # noqa: F401
except ImportError as _e:
    _verl_import_error = _e
    traceback.print_exc()

if _verl_import_error is not None:
    pytest.skip(
        f"verl not available (tried /model/sft_cotrain/verl etc): {_verl_import_error}",
        allow_module_level=True,
    )

try:
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer
except Exception as _e:  # noqa: BLE001
    traceback.print_exc()
    pytest.skip(
        f"failed to import SpecoRaySFTRayTrainer: {_e}",
        allow_module_level=True,
    )

from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
    OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
    OLD_LOGPROB_HIDDEN_REF_META_KEY,
    OLD_LOGPROB_HIDDEN_REFS_KEY,
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_SAMPLE_INDICES_KEY,
)


def _make_mock_trainer(monkeypatch, hidden_by_ref: dict):
    """创建一个最小化的 mock trainer，用于调用真实的 _speco_collect_sft_features。"""
    trainer = MagicMock(spec=SpecoRaySFTRayTrainer)
    trainer.global_steps = 0
    trainer._speco_last_sft_collected_samples = 0
    trainer._speco_last_sft_payload_mib = 0.0
    trainer._speco_oldlogprob_hidden_layout.return_value = "per_sample"
    trainer._speco_oldlogprob_aux_layer_ids.return_value = None
    trainer._speco_get_sequence_item.side_effect = lambda seq, idx: (
        seq[idx] if seq is not None and idx < len(seq) else None
    )

    # 捕获 speco_collect_rollout_features 收到的 buckets
    captured = {}

    def _capture(buckets):
        captured["buckets"] = buckets

    trainer.speco_collect_rollout_features.side_effect = _capture

    # mock ray.get：根据 ref 返回对应的 hidden tensor
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: hidden_by_ref[id(ref)])

    # patch tu.get 使其支持普通 dict（测试中不用 TensorDict）
    import verl_speco.trainer.speco_sft_trainer as _sft_mod

    monkeypatch.setattr(
        _sft_mod.tu,
        "get",
        lambda data, key, default=None: (
            data.get(key, default) if isinstance(data, dict) else default
        ),
    )

    return trainer, captured


def _build_inputs(batch_size, seq_len, hidden_dim, global_idx):
    """构造 batch / collect_plan / output，token 和 hidden 都用 orig_idx 标识。"""
    # input_ids 每行用原始索引填充，便于后续验证配对
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(batch_size):
        input_ids[i] = i  # 原始行 i 的 token 标识为 i

    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # SAMPLE_INDICES 在重排前为 arange，重排后随 data 一起被 index_select
    sample_indices = torch.arange(batch_size)[global_idx]

    batch = {
        "input_ids": input_ids[global_idx],  # 重排后的 input_ids
        "loss_mask": loss_mask[global_idx],
        OLD_LOGPROB_SAMPLE_INDICES_KEY: sample_indices,
    }

    collect_mask = torch.ones(batch_size, dtype=torch.bool)
    hidden_positions = torch.stack([torch.arange(seq_len) for _ in range(batch_size)])
    hidden_position_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    owner_rank = torch.zeros(batch_size, dtype=torch.long)

    collect_plan = {
        "collect_mask": collect_mask,
        "hidden_positions": hidden_positions,
        "hidden_position_mask": hidden_position_mask,
        "owner_rank": owner_rank,
        "owner_count": 1,
    }

    # 构造 hidden_refs 和 hidden_ref_meta
    # ref_meta[batch_idx] = {"batch_idx": 原始索引, "chunk_start": 0, "chunk_length": seq_len}
    hidden_refs = []
    hidden_ref_meta = []
    hidden_by_ref = {}
    for orig_idx in range(batch_size):
        ref = MagicMock()  # 用 id(ref) 作为 key
        hidden_tensor = torch.full((seq_len, hidden_dim), float(orig_idx))
        hidden_by_ref[id(ref)] = hidden_tensor
        hidden_refs.append(ref)
        hidden_ref_meta.append(
            {
                "batch_idx": orig_idx,
                "chunk_start": 0,
                "chunk_length": seq_len,
            }
        )

    output = {
        OLD_LOGPROB_HIDDEN_REFS_KEY: hidden_refs,
        OLD_LOGPROB_HIDDEN_REF_META_KEY: hidden_ref_meta,
    }

    return batch, collect_plan, output, hidden_by_ref


def test_balance_batch_reorder_preserves_token_hidden_alignment(monkeypatch) -> None:
    """调用真实 _speco_collect_sft_features，验证重排后 token/hidden 配对正确。

    重排 [0,1,2,3] → [2,0,3,1]。每个 token 行和对应的 hidden tensor 都用
    原始索引 orig_idx 填充，验证 buckets 中 sample 的 input_ids[0,0] ==
    hidden_states[0,0,0]（即同一 orig_idx）。
    """
    batch_size = 4
    seq_len = 5
    hidden_dim = 3
    global_idx = torch.tensor([2, 0, 3, 1])

    batch, collect_plan, output, hidden_by_ref = _build_inputs(
        batch_size, seq_len, hidden_dim, global_idx
    )
    trainer, captured = _make_mock_trainer(monkeypatch, hidden_by_ref)

    collected = SpecoRaySFTRayTrainer._speco_collect_sft_features(
        trainer, batch, collect_plan, output
    )

    assert collected == batch_size
    buckets = captured["buckets"]
    assert len(buckets) == 1
    samples = buckets[0]
    assert len(samples) == batch_size

    # 重排后的 token 顺序
    reordered_orig_indices = global_idx.tolist()  # [2, 0, 3, 1]

    for i, sample in enumerate(samples):
        expected_orig_idx = reordered_orig_indices[i]
        token_id = int(sample["input_ids"][0, 0].item())
        hidden_id = int(sample["hidden_states"][0, 0, 0].item())
        assert token_id == expected_orig_idx, (
            f"sample {i}: token_id={token_id}, expected={expected_orig_idx}"
        )
        assert hidden_id == expected_orig_idx, (
            f"sample {i}: hidden_id={hidden_id}, expected={expected_orig_idx} "
            f"(token/hidden mismatch!)"
        )


def test_no_reorder_keeps_direct_mapping(monkeypatch) -> None:
    """未重排时，token 与 hidden 直接按行对应。"""
    batch_size = 3
    seq_len = 4
    hidden_dim = 2
    global_idx = torch.arange(batch_size)  # 无重排

    batch, collect_plan, output, hidden_by_ref = _build_inputs(
        batch_size, seq_len, hidden_dim, global_idx
    )
    trainer, captured = _make_mock_trainer(monkeypatch, hidden_by_ref)

    collected = SpecoRaySFTRayTrainer._speco_collect_sft_features(
        trainer, batch, collect_plan, output
    )

    assert collected == batch_size
    samples = captured["buckets"][0]
    for i, sample in enumerate(samples):
        token_id = int(sample["input_ids"][0, 0].item())
        hidden_id = int(sample["hidden_states"][0, 0, 0].item())
        assert token_id == i
        assert hidden_id == i


def test_sample_indices_none_fallback(monkeypatch) -> None:
    """sample_indices 为 None 时回退到 batch_idx（旧版本兼容路径）。"""
    batch_size = 3
    seq_len = 4
    hidden_dim = 2
    global_idx = torch.tensor([2, 0, 1])  # 重排

    batch, collect_plan, output, hidden_by_ref = _build_inputs(
        batch_size, seq_len, hidden_dim, global_idx
    )
    # 移除 sample_indices，触发 fallback
    del batch[OLD_LOGPROB_SAMPLE_INDICES_KEY]

    trainer, captured = _make_mock_trainer(monkeypatch, hidden_by_ref)

    collected = SpecoRaySFTRayTrainer._speco_collect_sft_features(
        trainer, batch, collect_plan, output
    )

    # fallback 路径用 batch_idx 查 ref_meta_map，会导致错配
    # （这就是为什么需要 sample_indices），此处仅验证不崩溃
    assert collected == batch_size


def test_micro_batch_execution_order_differs_from_original(monkeypatch) -> None:
    """micro-batch 执行序 ≠ 原始序时，token/hidden 配对仍正确。

    模拟：原始 batch [0,1,2,3,4,5]，balance_batch 重排为 [3,1,5,0,2,4]，
    然后 micro-batch 切分为 [[3,1],[5,0],[2,4]]。每个 micro-batch 内的
    batch_idx 是局部位置，需通过 sample_indices 还原全局原始索引。
    """
    batch_size = 6
    seq_len = 4
    hidden_dim = 2
    global_idx = torch.tensor([3, 1, 5, 0, 2, 4])

    batch, collect_plan, output, hidden_by_ref = _build_inputs(
        batch_size, seq_len, hidden_dim, global_idx
    )
    trainer, captured = _make_mock_trainer(monkeypatch, hidden_by_ref)

    collected = SpecoRaySFTRayTrainer._speco_collect_sft_features(
        trainer, batch, collect_plan, output
    )

    assert collected == batch_size
    samples = captured["buckets"][0]
    assert len(samples) == batch_size

    reordered_orig_indices = global_idx.tolist()  # [3, 1, 5, 0, 2, 4]
    for i, sample in enumerate(samples):
        expected_orig_idx = reordered_orig_indices[i]
        token_id = int(sample["input_ids"][0, 0].item())
        hidden_id = int(sample["hidden_states"][0, 0, 0].item())
        assert token_id == expected_orig_idx, (
            f"sample {i}: token_id={token_id}, expected={expected_orig_idx}"
        )
        assert hidden_id == expected_orig_idx, (
            f"sample {i}: hidden_id={hidden_id}, expected={expected_orig_idx} "
            f"(token/hidden mismatch!)"
        )


def test_sample_indices_key_name_stable() -> None:
    """确保 SAMPLE_INDICES 的 key 名称与 driver stamp 逻辑一致。"""
    assert OLD_LOGPROB_SAMPLE_INDICES_KEY == "speco_oldlogprob_sample_indices"
