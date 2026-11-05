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
    # P2: 让 _speco_find_continuous_segments 使用真实实现，而非返回 MagicMock
    trainer._speco_find_continuous_segments = (
        SpecoRaySFTRayTrainer._speco_find_continuous_segments
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


def test_lm_head_sync_all_training_ranks_accepted():
    """所有参与训练的 rank 都接受时，同步成功。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = [
        {"accepted": True, "applied": True},
        {"accepted": True, "applied": True},
    ]
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is True
    assert info["accepted_count"] == 2
    assert info["total_count"] == 2


def test_lm_head_sync_partial_acceptance_rejected():
    """一个 rank 接受、另一个拒绝时，同步失败（不得用混合 head 版本训练）。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = [
        {"accepted": True, "applied": True},
        {"accepted": False, "applied": False, "reason": "missing_payload"},
    ]
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is False
    assert info["accepted_count"] == 1
    assert info["total_count"] == 2
    assert len(info["rejected"]) == 1


def test_lm_head_sync_non_training_ranks_excluded():
    """非训练 rank（disabled/not_in_training_group/not_target_worker）的拒绝
    不影响同步结果，只校验实际参与训练的 rank。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = [
        {"accepted": False, "applied": False, "reason": "disabled"},
        {"accepted": False, "applied": False, "reason": "not_in_training_group"},
        {"accepted": False, "applied": False, "reason": "not_target_worker"},
        {"accepted": True, "applied": True},
    ]
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is True
    assert info["total_count"] == 1
    assert info["accepted_count"] == 1


def test_lm_head_sync_no_training_ranks_responded():
    """所有响应都来自非训练 rank 时，同步失败。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = [
        {"accepted": False, "applied": False, "reason": "disabled"},
        {"accepted": False, "applied": False, "reason": "not_in_training_group"},
    ]
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is False
    assert info["total_count"] == 0


def test_lm_head_sync_single_worker_accepted():
    """单 worker 响应（非 list）且接受时，同步成功。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = {"accepted": True, "applied": True}
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is True


def test_lm_head_sync_single_worker_rejected():
    """单 worker 响应（非 list）且拒绝时，同步失败。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    result = {"accepted": False, "applied": False, "reason": "missing_payload"}
    synced, info = SpecoRaySFTRayTrainer._validate_lm_head_sync_result(result)
    assert synced is False


# ---------------------------------------------------------------------------
# P2: 多轮 SFT 不连续 hidden positions 拆段测试
# ---------------------------------------------------------------------------


def test_find_continuous_segments_splits_discontinuous_positions():
    """不连续 positions 被拆分为多个连续区间。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    find = SpecoRaySFTRayTrainer._speco_find_continuous_segments

    # 两段不连续：[1,2,5,6] -> [(0,2),(2,4)]
    segments = find(torch.tensor([1, 2, 5, 6]))
    assert segments == [(0, 2), (2, 4)]

    # 完全连续
    assert find(torch.tensor([3, 4, 5, 6])) == [(0, 4)]

    # 三段不连续
    assert find(torch.tensor([0, 1, 3, 7, 8])) == [(0, 2), (2, 3), (3, 5)]

    # 单元素
    assert find(torch.tensor([5])) == [(0, 1)]

    # 空
    assert find(torch.tensor([], dtype=torch.long)) == []


def test_multi_turn_collect_splits_into_continuous_segments(monkeypatch) -> None:
    """多轮对话两段 assistant 回复：采集后拆成 2 个 sample，两段监督都保留。

    场景：loss_mask=[0,0,1,1,0,0,1,1]，shifted_mask=[0,1,1,0,0,1,1,0]，
    collect_positions=[1,2,5,6]。下游 collect_online_data 要求 hidden_positions
    连续，因此必须拆成 [1,2] 和 [5,6] 两个 sample，否则第二段监督会被 truncate 丢弃。
    """
    batch_size = 1
    seq_len = 8
    hidden_dim = 3

    # 构造多轮 loss_mask：两段 assistant 回复（位置 2-3 和 6-7）
    loss_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.long)
    input_ids = torch.arange(seq_len).unsqueeze(0).long()

    batch = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        OLD_LOGPROB_SAMPLE_INDICES_KEY: torch.tensor([0]),
    }

    # collect_plan：positions=[1,2,5,6]（不连续），对应 4 个 valid rows
    collect_positions = torch.tensor([1, 2, 5, 6], dtype=torch.long)
    collect_plan = {
        "collect_mask": torch.tensor([True]),
        "hidden_positions": collect_positions.unsqueeze(0),
        "hidden_position_mask": torch.ones(1, 4, dtype=torch.bool),
        "owner_rank": torch.tensor([0]),
        "owner_count": 1,
    }

    # hidden tensor 共 4 行，每行用行号填充便于验证
    hidden_tensor = (
        torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, hidden_dim)
    )
    hidden_by_ref = {}
    ref = MagicMock()
    hidden_by_ref[id(ref)] = hidden_tensor

    hidden_ref_meta = [
        {"batch_idx": 0, "chunk_start": 0, "chunk_length": 4},
    ]
    output = {
        OLD_LOGPROB_HIDDEN_REFS_KEY: [ref],
        OLD_LOGPROB_HIDDEN_REF_META_KEY: hidden_ref_meta,
    }

    trainer, captured = _make_mock_trainer(monkeypatch, hidden_by_ref)

    collected = SpecoRaySFTRayTrainer._speco_collect_sft_features(
        trainer, batch, collect_plan, output
    )

    # 拆成 2 个 sample
    assert collected == 2
    samples = captured["buckets"][0]
    assert len(samples) == 2

    # 第一个 sample：positions=[1,2]，hidden 前 2 行（行号 0,1）
    seg0 = samples[0]
    assert seg0["hidden_positions"].reshape(-1).tolist() == [1, 2]
    assert seg0["hidden_states"].shape == (1, 2, hidden_dim)
    assert seg0["hidden_position_start"] == 1
    assert seg0["hidden_position_end"] == 3
    # 验证 hidden 行与 positions 一一对应
    assert seg0["hidden_states"][0, :, 0].tolist() == [0.0, 1.0]

    # 第二个 sample：positions=[5,6]，hidden 后 2 行（行号 2,3）
    seg1 = samples[1]
    assert seg1["hidden_positions"].reshape(-1).tolist() == [5, 6]
    assert seg1["hidden_states"].shape == (1, 2, hidden_dim)
    assert seg1["hidden_position_start"] == 5
    assert seg1["hidden_position_end"] == 7
    assert seg1["hidden_states"][0, :, 0].tolist() == [2.0, 3.0]


# ---------------------------------------------------------------------------
# P1: 直接调用 speco_train_drafter() 入口测试（无 backend 属性）
# ---------------------------------------------------------------------------


def test_speco_train_drafter_no_backend_dspark_enforces_same_step(monkeypatch):
    """直接调用真实 speco_train_drafter()，验证无需 backend 属性，且 DSpark
    场景下 plan 的 min_sample_step = max_sample_step = global_step。

    PR review 要求：仅手动向 base trainer 传入范围的测试捕获不到 self.backend
    AttributeError，必须直接调用 SFT driver 的 speco_train_drafter() 入口。
    """
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    captured_plan = {}

    def _capture_preflight(plan):
        captured_plan["plan"] = plan
        return [{"ready": True, "reason": None}]

    def _capture_train(plan):
        return [{"trained": True, "successful_steps": 1}]

    # 构造 mock trainer，不设置 backend 属性
    trainer = MagicMock(spec=SpecoRaySFTRayTrainer)
    # 显式删除 backend（spec= 会创建该属性，需删除以模拟真实 SFT trainer）
    del trainer.backend
    trainer.global_steps = 5

    # 配置：DSpark，use_logits=False，allow_cross_step_hidden=False
    drafter_cfg = MagicMock()
    drafter_cfg.speculative_algorithm = "dspark"
    training_cfg = {
        "sample_last_n_steps": 10,
        "require_full_batch": False,
        "step": 10,
        "min_trainable_batches": 1,
        "use_logits": False,
        "allow_cross_step_hidden": False,
    }
    drafter_cfg.training = training_cfg
    trainer.config.rollout.drafter = drafter_cfg

    # mock worker group
    wg = MagicMock()
    wg.get_drafter_training_data_status.return_value = [
        {
            "available": True,
            "rank": 0,
            "worker_incarnation": 1,
            "buffer_version": 1,
            "data_version": 1,
            "trainable_samples": 16,
            "min_sample_step": 0,
            "max_sample_step": 5,
        }
    ]
    wg.preflight_drafter_training.side_effect = _capture_preflight
    wg.train_drafter.side_effect = _capture_train
    trainer._require_speco_worker_group.return_value = wg

    # mock _ray_get_if_needed 直接返回值
    trainer._ray_get_if_needed.side_effect = lambda x: x

    # mock _speco_publish_during_sft
    trainer._speco_publish_during_sft.return_value = False

    # 调用真实方法
    result = SpecoRaySFTRayTrainer.speco_train_drafter(trainer)

    # 验证训练成功执行
    assert result[0]["trained"] is True

    # 验证 plan 中 min/max_sample_step 被强制设为 global_step=5
    plan = captured_plan["plan"]
    assert plan["min_sample_step"] == 5
    assert plan["max_sample_step"] == 5
    assert plan["required_target_version"] is None  # plan 不传版本，靠 step 范围约束


def test_speco_train_drafter_no_backend_eagle3_does_not_enforce_same_step(monkeypatch):
    """eagle3 场景下不应强制同 step，min/max_sample_step 来自 worker 状态。"""
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    captured_plan = {}

    def _capture_preflight(plan):
        captured_plan["plan"] = plan
        return [{"ready": True, "reason": None}]

    trainer = MagicMock(spec=SpecoRaySFTRayTrainer)
    del trainer.backend
    trainer.global_steps = 5

    drafter_cfg = MagicMock()
    drafter_cfg.speculative_algorithm = "eagle3"
    training_cfg = {
        "sample_last_n_steps": 10,
        "require_full_batch": False,
        "step": 10,
        "min_trainable_batches": 1,
        "use_logits": False,
        "allow_cross_step_hidden": False,
    }
    drafter_cfg.training = training_cfg
    trainer.config.rollout.drafter = drafter_cfg

    wg = MagicMock()
    wg.get_drafter_training_data_status.return_value = [
        {
            "available": True,
            "rank": 0,
            "worker_incarnation": 1,
            "buffer_version": 1,
            "data_version": 1,
            "trainable_samples": 16,
            "min_sample_step": 0,
            "max_sample_step": 5,
        }
    ]
    wg.preflight_drafter_training.side_effect = _capture_preflight
    wg.train_drafter.return_value = [{"trained": True, "successful_steps": 1}]
    trainer._require_speco_worker_group.return_value = wg
    trainer._ray_get_if_needed.side_effect = lambda x: x
    trainer._speco_publish_during_sft.return_value = False

    result = SpecoRaySFTRayTrainer.speco_train_drafter(trainer)
    assert result[0]["trained"] is True

    # eagle3 不强制同 step，使用 worker 返回的实际范围 [0, 5]
    plan = captured_plan["plan"]
    assert plan["min_sample_step"] == 0
    assert plan["max_sample_step"] == 5
