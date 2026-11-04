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
from __future__ import annotations

from collections import deque
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
base_trainer = pytest.importorskip("verl_speco.trainer.base_trainer")

DrafterBaseTrainer = base_trainer.DrafterBaseTrainer


def _metric_trainer():
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"dspark_block_size": 2})
        )
    )
    trainer.optimizer = None
    trainer.optimizer_steps_total = 7
    trainer._reduce_training_metric = MethodType(
        lambda _self, value: value.detach().float(), trainer
    )
    trainer.reset_training_metrics()
    return trainer


def test_step_accuracy_and_aggregate_metrics_preserve_lk_diagnostics():
    trainer = _metric_trainer()
    diagnostics = {
        "correct_count": torch.tensor(3.0),
        "eval_token_count": torch.tensor(4.0),
        "top1_correct_count": torch.tensor(2.0),
        "top5_correct_count": torch.tensor(3.0),
        "quality_token_count": torch.tensor(4.0),
        "ce_loss_sum": torch.tensor(2.0),
        "ce_weighted_token_count": torch.tensor(4.0),
        "l1_loss_sum": torch.tensor(1.0),
        "l1_weighted_token_count": torch.tensor(4.0),
        "lk_loss_sum": torch.tensor(0.5),
        "lk_weighted_token_count": torch.tensor(4.0),
        "lk_diagnostic_token_count": torch.tensor(4.0),
        "lk_acceptance_sum": torch.tensor(3.0),
        "lk_kl_weight_sum": torch.tensor(2.0),
        "lk_forward_kl_sum": torch.tensor(1.0),
        "lk_tv_sum": torch.tensor(1.0),
        "loss_sum_per_position": torch.tensor([1.0, 3.0]),
        "correct_per_position": torch.tensor([1.0, 2.0]),
        "count_per_position": torch.tensor([2.0, 2.0]),
    }

    step = trainer._record_dflash_training_metrics({"diagnostics": diagnostics})

    assert step["accuracy"] == pytest.approx(0.75)
    assert step["top1_acc"] == pytest.approx(0.5)
    assert step["top5_acc"] == pytest.approx(0.75)
    assert step["acc_per_position"] == pytest.approx([0.5, 1.0])

    trainer._training_sample_draws = 6
    trainer._training_unique_sample_ids.update({11, 12, 13})
    metrics = trainer.get_training_metrics()
    assert metrics["dspark/ce_loss"] == pytest.approx(0.5)
    assert metrics["dspark/l1_loss"] == pytest.approx(0.25)
    assert metrics["dspark/lk_loss"] == pytest.approx(0.125)
    assert metrics["dspark/lk/acceptance"] == pytest.approx(0.75)
    assert metrics["dspark/accuracy_per_position/0"] == pytest.approx(0.5)
    assert metrics["dspark/accuracy_per_position/1"] == pytest.approx(1.0)
    assert metrics["drafter/train_sample_draws"] == pytest.approx(6.0)
    assert metrics["drafter/train_unique_samples"] == pytest.approx(3.0)


def test_training_data_stats_select_current_step_without_data_buffer():
    trainer = _metric_trainer()
    trainer.use_data_buffer = False
    trainer.current_rl_step = 5
    trainer.batch_size = 2
    trainer.collected_data = deque(
        [
            {"step": 4, "sample": "old"},
            {"step": 5, "sample": "a"},
            {"step": 5, "sample": "b"},
        ]
    )

    stats = trainer.get_training_data_stats()

    assert stats == {
        "buffer_total": 3,
        "eligible_samples": 2,
        "eligible_step_counts": {5: 2},
        "sample_last_n_steps": 2,
        "batch_size": 2,
    }
