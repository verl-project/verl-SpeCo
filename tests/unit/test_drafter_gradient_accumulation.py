# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
from collections import deque
from types import MethodType, SimpleNamespace

import pytest


def test_dflash_lm_head_rows_prefer_newer_collected_version_when_step_lags() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter row selection needs the trainer dependency stack",
    )
    torch = pytest.importorskip("torch")
    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.backend = SimpleNamespace(model_type="dflash")
    trainer.current_rl_step = 1
    trainer.use_data_buffer = False
    trainer.data_buffer = []
    trainer.collected_data = deque(
        [
            {"step": 1, "target_version": 1, "input_ids": torch.tensor([[1, 2]])},
            {"step": 2, "target_version": 2, "input_ids": torch.tensor([[7, 8]])},
            {"step": 2, "target_version": 2, "input_ids": torch.tensor([[8, 9]])},
        ]
    )
    trainer._block_drafter_config_value = (
        lambda suffix, default: "restricted_ce" if suffix == "loss_mode" else default
    )
    trainer._target_lm_head_vocab_size = lambda: 100

    rows = trainer._build_target_lm_head_row_indices_from_dflash_data()

    assert rows is not None
    assert rows["selected_rows"] == 3
    assert rows["row_indices"].tolist() == [7, 8, 9]


def test_dflash_lm_head_rows_fall_back_to_latest_collected_version() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter row selection needs the trainer dependency stack",
    )
    torch = pytest.importorskip("torch")
    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.backend = SimpleNamespace(model_type="dflash")
    trainer.current_rl_step = 1
    trainer.use_data_buffer = False
    trainer.data_buffer = []
    trainer.collected_data = deque(
        [
            {"step": 2, "target_version": 2, "input_ids": torch.tensor([[7, 8]])},
            {"step": 2, "target_version": 2, "input_ids": torch.tensor([[8, 9]])},
        ]
    )
    trainer._block_drafter_config_value = (
        lambda suffix, default: "restricted_ce" if suffix == "loss_mode" else default
    )
    trainer._target_lm_head_vocab_size = lambda: 100

    rows = trainer._build_target_lm_head_row_indices_from_dflash_data()

    assert rows is not None
    assert rows["selected_rows"] == 3
    assert rows["row_indices"].tolist() == [7, 8, 9]


def test_single_micro_batch_bubble_reservation_is_replayed_until_finalize() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter replay needs the trainer dependency stack",
    )

    class _Buffer:
        def __init__(self) -> None:
            self.consume_calls = []

        def consume(self, plan_id, items):
            self.consume_calls.append((plan_id, list(items)))
            return len(items)

        def __len__(self) -> int:
            return 2

    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.rank = 0
    trainer.data_buffer = _Buffer()
    trainer._active_training_reservation_id = "quota-plan"
    trainer._active_training_replay_cursor = 0
    trainer._active_training_replay_used_items = {}
    trainer._last_prepared_training_items = []
    trainer._mark_buffer_changed = lambda: None
    first = {"sample": 1}
    second = {"sample": 2}

    # gradient_accumulation_steps == 1 reaches this helper after every
    # optimizer step. The samples must remain reserved and replayable.
    trainer._last_prepared_training_items = [first, second]
    assert trainer._consume_last_training_batch() == 0
    trainer._last_prepared_training_items = [first, second]
    assert trainer._consume_last_training_batch() == 0
    assert trainer.data_buffer.consume_calls == []

    consumed = trainer.finalize_training_data_reservation("quota-plan")

    assert consumed == 2
    assert trainer.data_buffer.consume_calls == [
        ("quota-plan", [first, second])
    ]


def test_quota_cycle_reuses_full_snapshot_across_bubble_plans() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter replay needs the trainer dependency stack",
    )
    data_buffer_module = pytest.importorskip("verl_speco.trainer.data_buffer")

    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.rank = 0
    trainer.batch_size = 4
    trainer.data_buffer = data_buffer_module.DataBuffer(max_size=32)
    trainer._mark_buffer_changed = lambda: None
    for sample_id in range(8):
        trainer.data_buffer._current_step = 4
        trainer.data_buffer.add_batch(
            {
                "target_version": 4,
                "_speco_global_sample_id": sample_id,
            }
        )

    first = trainer.reserve_training_data(
        plan_id="bubble-1",
        target_version=4,
        max_batches=1,
        retain_replay_session=True,
    )
    assert first["reserved_samples"] == 8
    trainer._active_training_replay_cursor = 4
    assert trainer.finalize_training_data_reservation("bubble-1", consume=False) == 0
    trainer.release_training_data_reservation("bubble-1")

    second = trainer.reserve_training_data(
        plan_id="bubble-2",
        target_version=4,
        max_batches=1,
        retain_replay_session=False,
    )
    assert second["reserved_samples"] == 8
    assert trainer._active_training_replay_cursor == 4
    trainer._active_training_replay_used_items = {
        id(item): item for item in trainer.data_buffer.get_all_data()
    }
    assert trainer.finalize_training_data_reservation("bubble-2", consume=True) == 8
    trainer.release_training_data_reservation("bubble-2")
    assert len(trainer.data_buffer) == 0


@pytest.mark.parametrize(
    ("model_type", "expected_input_rows"),
    [
        ("dflash", 5),
        ("dflash2", 5),
        ("dspark", 5),
        ("domino", 5),
        ("eagle3", 6),
    ],
)
def test_online_collection_uses_backend_specific_input_alignment(
    monkeypatch, model_type: str, expected_input_rows: int
) -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="online collection needs the trainer dependency stack",
    )
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    trainer = base_trainer.DrafterBaseTrainer.__new__(base_trainer.DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type=model_type)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "use_logits": False,
                    "collect_hidden_states_from_sgl": False,
                },
                rollout={},
            )
        )
    )
    trainer.copy_stream = None
    trainer.rank = 0
    trainer.pad_token_id = 0
    trainer.model_config = SimpleNamespace(pad_token_id=0)
    trainer.current_rl_step = 1
    trainer.use_data_buffer = False
    trainer.collected_data = []
    trainer.buffer_version = 0

    assert trainer.collect_online_data(
        {"input_ids": torch.arange(6).unsqueeze(0)},
        torch.zeros(1, 5, 4),
    )

    item = trainer.collected_data[0]
    assert item["input_ids"].size(0) == expected_input_rows
    assert item["hidden_states"].size(0) == 5
    assert item["loss_mask"].size(0) == expected_input_rows
    if model_type in {"dflash", "dflash2", "dspark", "domino"}:
        assert item["input_ids"].tolist() == [0, 1, 2, 3, 4]


def test_block_collection_uses_explicit_hidden_positions_as_source_of_truth(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="online collection needs the trainer dependency stack",
    )
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    trainer = base_trainer.DrafterBaseTrainer.__new__(base_trainer.DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dflash")
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "use_logits": False,
                    "collect_hidden_states_from_sgl": False,
                },
                rollout={},
            )
        )
    )
    trainer.copy_stream = None
    trainer.rank = 0
    trainer.pad_token_id = 0
    trainer.model_config = SimpleNamespace(pad_token_id=0)
    trainer.current_rl_step = 1
    trainer.use_data_buffer = False
    trainer.collected_data = []
    trainer.buffer_version = 0

    assert trainer.collect_online_data(
        {
            "input_ids": torch.arange(6).unsqueeze(0),
            "hidden_positions": torch.arange(1, 6).unsqueeze(0),
        },
        torch.zeros(1, 5, 4),
    )

    item = trainer.collected_data[0]
    assert item["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert item["input_ids"].size(0) == item["hidden_states"].size(0) == 5
    assert item["loss_mask"].size(0) == 5


def test_accumulation_uses_combined_valid_token_mean(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter accumulation needs the trainer dependency stack",
    )
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    class _Backend:
        model_type = "test"

        def __init__(self, model):
            self.model = model

        def compute_loss(self, _model, batch, _pad_size):
            tokens = batch["tokens"].float()
            total = self.model.weight * batch["loss_sum_factor"].float()
            return {
                "total_local_vloss": total * 0.0,
                "total_local_ploss": total,
                "local_num_tokens": tokens,
                "v_weight": 0.0,
                "p_weight": 1.0,
            }

    trainer = base_trainer.DrafterBaseTrainer.__new__(base_trainer.DrafterBaseTrainer)
    trainer.model = _Model()
    trainer.backend = _Backend(trainer.model)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.lr_scheduler = None
    trainer.use_ulysses_sp = False
    trainer.training_steps = 0
    trainer.optimizer_steps_total = 0
    trainer._current_pad_size = 0
    trainer._current_accumulation_valid_tokens = 0
    trainer._current_accumulation_vloss_sum = 0.0
    trainer._current_accumulation_ploss_sum = 0.0
    trainer._last_optimizer_valid_tokens = 0
    trainer.record_training_timing = lambda *_args, **_kwargs: None
    trainer._record_dflash_training_metrics = lambda *_args, **_kwargs: None
    trainer._get_sp_group = lambda: None
    trainer._reduce_loss_metrics = MethodType(
        lambda _self, l_v, l_p, l_n: (l_v, l_p, l_n, 1), trainer
    )

    async def _run() -> None:
        assert await trainer._training_step_on_batch(
            {
                "tokens": torch.tensor(1.0),
                "loss_sum_factor": torch.tensor(1.0),
            },
            1,
            accumulation_steps=2,
            accumulation_index=0,
        )
        assert await trainer._training_step_on_batch(
            {
                "tokens": torch.tensor(3.0),
                "loss_sum_factor": torch.tensor(6.0),
            },
            1,
            accumulation_steps=2,
            accumulation_index=1,
        )

    asyncio.run(_run())

    # (1 + 6) / (1 + 3) = 1.75; averaging micro-batch means would be 1.5.
    assert trainer.model.weight.item() == pytest.approx(-1.75)
    assert trainer._last_optimizer_valid_tokens == 4
