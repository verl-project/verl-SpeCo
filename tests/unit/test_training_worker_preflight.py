# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("torch")

from verl_speco.workers.speco_worker import SpecoWorker


class _FakeTrainer:
    def __init__(self, *, data_version: int) -> None:
        self.buffer_version = 3
        self._target_lm_head_weight_step = 4
        self.optimizer_steps_total = 0
        self.data_version = data_version
        self.activation_calls = 0
        self.status_kwargs = []

    def get_training_data_status(self, **kwargs):
        self.status_kwargs.append(kwargs)
        return {
            "trainable_batches": 1,
            "trainable_samples": 4,
            "data_version": self.data_version,
        }

    async def activate_training_model(self) -> bool:
        self.activation_calls += 1
        return True


def _worker(*, data_version: int) -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker.rank = 0
    worker.worker_incarnation = "worker-0"
    worker.last_global_step = 4
    worker.device_name = "cpu"
    worker.trainer = _FakeTrainer(data_version=data_version)
    worker._prepared_training_plan_id = None
    worker._prepared_training_data_version = None
    worker._prepared_training_target_version = None
    return worker


def _plan() -> dict[str, object]:
    return {
        "launch": True,
        "execution_strategy": "sync",
        "source_global_step": 4,
        "plan_id": "plan-4",
        "data_version": 4,
        "required_target_version": 4,
        "sample_last_n_steps": 2,
        "require_full_batch": False,
        "min_sample_step": 2,
        "max_sample_step": 4,
        "data_filter_reason": "recent_buffer_window",
        "min_batches": 1,
        "worker_snapshots": {
            "0": {
                "worker_incarnation": "worker-0",
                "buffer_version": 3,
                "data_version": 4,
            }
        },
    }


def test_worker_preflight_rejects_changed_data_version_before_activation() -> None:
    worker = _worker(data_version=5)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "data_version_changed"
    assert result["data_version"] == 5
    assert worker.trainer.activation_calls == 0
    assert worker._prepared_training_plan_id is None


def test_worker_preflight_records_actual_versions_for_training_result() -> None:
    worker = _worker(data_version=4)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert result["ready"]
    assert result["data_version"] == 4
    assert result["target_version"] == 4
    assert worker._prepared_training_plan_id == "plan-4"
    assert worker._prepared_training_data_version == 4
    assert worker._prepared_training_target_version == 4
    assert worker.trainer.status_kwargs[-1]["min_sample_step"] == 2
    assert worker.trainer.status_kwargs[-1]["max_sample_step"] == 4
