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
import copy
import time
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from omegaconf import OmegaConf

from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.data_buffer import DataBuffer
from verl_speco.workers.speco_worker import SpecoWorker


class _FakeTrainer:
    def __init__(self, *, data_version: int) -> None:
        self.buffer_version = 3
        self._target_lm_head_weight_step = 4
        self._bubble_time_enabled = True
        self.optimizer_steps_total = 0
        self.data_version = data_version
        self.activation_calls = 0
        self.cleanup_calls = 0
        self.release_after_activation_calls = 0
        self.reserved_plan_id = None
        self.requested_target_version = None

    def select_target_lm_head_version(self, global_step: int) -> bool:
        self._target_lm_head_weight_step = global_step
        return True

    def reserve_training_data(self, *, plan_id: str, **kwargs):
        del kwargs
        self.reserved_plan_id = plan_id
        return {"reserved_samples": 4}

    def release_training_data_reservation(self, plan_id: str) -> int:
        if self.reserved_plan_id != plan_id:
            return 0
        self.reserved_plan_id = None
        return 4

    def get_training_data_status(self, **kwargs):
        self.requested_target_version = kwargs.get("target_version")
        return {
            "trainable_batches": 1,
            "trainable_samples": 4,
            "data_version": self.data_version,
        }

    async def activate_training_model(self) -> bool:
        self.activation_calls += 1
        return True

    async def cleanup_training(
        self, clear_data: bool = True, *, keep_hot: bool = False
    ):
        del clear_data
        self.cleanup_calls += 1
        self.keep_hot = keep_hot

    async def release_training_memory_after_activation(self, *, strict=False):
        self.release_after_activation_calls += 1
        self.release_strict = strict

    def pop_model_state_dict_for_publish(self, global_step: int):
        return global_step == 4, {"weight": global_step}


def test_park_training_worker_preserves_optimizer_and_replay_progress() -> None:
    worker = _worker(data_version=4)
    worker.trainer.model = object()
    worker.trainer._training_device_resident = True
    worker.trainer.optimizer_steps_total = 9
    worker.trainer.reserved_plan_id = "quota-snapshot"

    result = asyncio.run(worker.park_drafter_training_model())

    assert result["parked"]
    assert worker.trainer.release_after_activation_calls == 1
    assert worker.trainer.release_strict
    assert worker.trainer.optimizer_steps_total == 9
    assert worker.trainer.reserved_plan_id == "quota-snapshot"
    assert worker.trainer.cleanup_calls == 0


def test_park_training_worker_skips_already_parked_model() -> None:
    worker = _worker(data_version=4)
    worker.trainer.model = object()
    worker.trainer._training_device_resident = False

    result = asyncio.run(worker.park_drafter_training_model())

    assert not result["parked"]
    assert worker.trainer.release_after_activation_calls == 0


def test_parking_preserves_optimizer_replay_and_training_counters(monkeypatch) -> None:
    import verl_speco.trainer.base_trainer as base_trainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = object()
    trainer.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None)
    trainer.optimizer_steps_total = 9
    trainer.training_steps = 9
    trainer._last_ckpt_step = 7
    trainer._training_active = True
    trainer._training_device_resident = True
    trainer.data_buffer = ["quota-snapshot"]
    trainer._active_training_reservation_id = "quota"
    operations = []
    monkeypatch.setattr(base_trainer, "device_name", "cpu")
    monkeypatch.setattr(
        base_trainer,
        "offload_fsdp_model_to_cpu",
        lambda model: operations.append("model"),
    )
    trainer._offload_optimizer_state_to_cpu = lambda: operations.append("optimizer")
    trainer._move_target_lm_head = lambda device: operations.append("head")
    trainer._park_idle_drafter_hccl = lambda: None

    asyncio.run(trainer.release_training_memory_after_activation())

    assert operations == ["model", "optimizer", "head"]
    assert not trainer._training_device_resident
    assert trainer.optimizer_steps_total == trainer.training_steps == 9
    assert trainer._last_ckpt_step == 7
    assert trainer.data_buffer == ["quota-snapshot"]
    assert trainer._active_training_reservation_id == "quota"


@pytest.mark.parametrize("strict", [False, True])
def test_parking_failure_continues_cleanup_and_retains_residency(
    monkeypatch, strict
) -> None:
    import verl_speco.trainer.base_trainer as base_trainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = object()
    trainer.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None)
    trainer._training_device_resident = True
    operations = []
    trainer._offload_optimizer_state_to_cpu = lambda: operations.append("optimizer")
    trainer._move_target_lm_head = lambda device: operations.append("head")
    trainer._park_idle_drafter_hccl = lambda: operations.append("hccl")
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    def fail_offload(model):
        raise RuntimeError("device transfer failed")

    monkeypatch.setattr(base_trainer, "offload_fsdp_model_to_cpu", fail_offload)
    if strict:
        with pytest.raises(RuntimeError, match="offload_model"):
            asyncio.run(trainer.release_training_memory_after_activation(strict=True))
    else:
        assert asyncio.run(trainer.release_training_memory_after_activation()) is False
    assert operations == ["optimizer", "head", "hccl"]
    assert trainer._training_device_resident


def test_cpu_writer_handoff_preserves_adam_scheduler_and_next_update() -> None:
    import torch

    def trainer(seed):
        torch.manual_seed(seed)
        obj = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        obj.model = torch.nn.Linear(3, 2)
        obj.optimizer = torch.optim.AdamW(obj.model.parameters(), lr=0.01)
        obj.lr_scheduler = torch.optim.lr_scheduler.StepLR(obj.optimizer, 1, 0.9)
        obj._training_active = obj._training_device_resident = False
        obj._active_training_reservation_id = None
        obj._target_lm_head_snapshots = {}
        obj.data_buffer = DataBuffer()
        obj.collected_data = []
        obj.buffer_version = 1
        obj.optimizer_steps_total = obj.training_steps = 0
        obj._mark_buffer_changed = lambda: None
        return obj

    source, target = trainer(1), trainer(2)

    def step(obj):
        obj.optimizer.zero_grad(set_to_none=True)
        obj.model(torch.ones(2, 3)).square().mean().backward()
        obj.optimizer.step()
        obj.lr_scheduler.step()
        obj.optimizer_steps_total += 1
        obj.training_steps += 1

    step(source)
    target.load_writer_state(source.local_writer_state(export=True))
    assert target.optimizer_steps_total == 1
    assert target.lr_scheduler.state_dict() == source.lr_scheduler.state_dict()
    step(source)
    step(target)
    for left, right in zip(
        source.model.parameters(), target.model.parameters(), strict=True
    ):
        assert torch.equal(left, right)


def test_cpu_writer_checkpoint_roundtrip_preserves_rng_next_loss_and_optimizer():
    import io
    import torch

    def trainer(seed):
        torch.manual_seed(seed)
        obj = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        obj.model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.2))
        obj.optimizer = torch.optim.AdamW(obj.model.parameters(), lr=0.01)
        obj.lr_scheduler = torch.optim.lr_scheduler.StepLR(obj.optimizer, 1, 0.9)
        obj._training_active = obj._training_device_resident = False
        obj._active_training_reservation_id = None
        obj._target_lm_head_snapshots = {}
        obj.data_buffer = DataBuffer()
        obj.collected_data = []
        obj.buffer_version = 1
        obj.optimizer_steps_total = obj.training_steps = 0
        obj._last_ckpt_step = 5
        obj._mark_buffer_changed = lambda: None
        return obj

    def step(obj):
        obj.optimizer.zero_grad(set_to_none=True)
        loss = obj.model(torch.randn(2, 3)).square().mean()
        loss.backward()
        obj.optimizer.step()
        obj.lr_scheduler.step()
        obj.optimizer_steps_total += 1
        obj.training_steps += 1
        return loss.detach()

    source, target = trainer(1), trainer(2)
    step(source)
    state = source.local_writer_state(export=True)
    state["rng"] = torch.get_rng_state()
    checkpoint = io.BytesIO()
    torch.save(state, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    target.load_writer_state(restored)
    assert target._last_ckpt_step == source._last_ckpt_step
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            source.model.parameters(), target.model.parameters(), strict=True
        )
    )
    torch.set_rng_state(restored["rng"])
    source_loss = step(source)
    torch.set_rng_state(restored["rng"])
    target_loss = step(target)
    assert torch.equal(source_loss, target_loss)
    assert target.optimizer_steps_total == source.optimizer_steps_total == 2
    assert target.lr_scheduler.state_dict() == source.lr_scheduler.state_dict()
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            source.model.parameters(), target.model.parameters(), strict=True
        )
    )
    for a, b in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True
    ):
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_fsdp1_optimizer_checkpoint_conversion_uses_writer_group(monkeypatch) -> None:
    from contextlib import nullcontext
    import verl_speco.trainer.base_trainer as module

    group = object()
    seen = []

    class FakeFSDP:
        @staticmethod
        def state_dict_type(*args):
            return nullcontext()

        @staticmethod
        def optim_state_dict(model, optimizer, *, group):
            seen.append(group)
            return {"state": {}}

        @staticmethod
        def optim_state_dict_to_load(model, optimizer, state, *, group):
            seen.append(group)
            return state

    monkeypatch.setattr(module, "FSDP", FakeFSDP)
    optimizer = SimpleNamespace(load_state_dict=lambda state: None)
    adapter = module._DrafterOptimizerState(FakeFSDP(), optimizer, group)
    adapter._is_ddp = lambda: False
    state = adapter.state_dict()
    adapter.load_state_dict(state)
    assert seen == [group, group]


def test_sync_optimizer_checkpoint_retains_dcp_conversion(monkeypatch) -> None:
    import torch.distributed.checkpoint.state_dict as state_dict_api
    import verl_speco.trainer.base_trainer as module

    class FakeFSDP:
        @staticmethod
        def optim_state_dict(*args, **kwargs):
            pytest.fail("Sync must retain its original DCP API")

        @staticmethod
        def optim_state_dict_to_load(*args, **kwargs):
            pytest.fail("Sync must retain its original DCP API")

    monkeypatch.setattr(module, "FSDP", FakeFSDP)
    calls = []
    monkeypatch.setattr(
        state_dict_api,
        "get_optimizer_state_dict",
        lambda model, optimizer, **kwargs: calls.append("save") or {},
    )
    monkeypatch.setattr(
        state_dict_api,
        "set_optimizer_state_dict",
        lambda model, optimizer, state, **kwargs: calls.append("load"),
    )
    adapter = module._DrafterOptimizerState(FakeFSDP(), object())
    adapter._is_ddp = lambda: False
    adapter.load_state_dict(adapter.state_dict())
    assert calls == ["save", "load"]


def _worker(*, data_version: int) -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker._rank = 0
    worker.worker_incarnation = "worker-0"
    worker.last_global_step = 4
    worker.device_name = "cpu"
    worker.trainer = _FakeTrainer(data_version=data_version)
    worker._prepared_training_plan_id = None
    worker._prepared_training_data_version = None
    worker._prepared_training_target_version = None
    worker.replica_rank = 0
    worker.last_trained_step = None
    worker.is_drafter_group_leader = True
    worker.is_global_publish_leader = True
    worker._last_trained_execution_strategy = None
    worker._last_trained_target_worker_ids = ()
    worker.training_group_ranks = [0]
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "scheduler": {"execution": {"strategy": "rollout_idle_worker"}}
                    }
                }
            }
        }
    )
    return worker


def test_select_target_lm_head_reuses_already_applied_version() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer._target_lm_head_snapshots = {
        4: {
            "weight": object(),
            "row_indices": None,
            "source_vocab_size": None,
            "chunked_apply": True,
        }
    }
    trainer._applied_target_lm_head_weight_step = 4
    trainer._target_lm_head_weight_step = 4
    trainer._pending_target_lm_head_weight = object()
    trainer._pending_target_lm_head_row_indices = object()
    trainer._pending_target_lm_head_source_vocab_size = 10
    trainer._pending_target_lm_head_chunked_apply = True
    trainer._target_lm_head_module = lambda: object()

    assert trainer.select_target_lm_head_version(4)
    assert trainer._target_lm_head_weight_step == 4
    assert trainer._pending_target_lm_head_weight is None
    assert trainer._pending_target_lm_head_row_indices is None
    assert trainer._pending_target_lm_head_source_vocab_size is None
    assert not trainer._pending_target_lm_head_chunked_apply


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
        "min_batches": 1,
        "max_batches": 1,
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


def test_sync_preflight_uses_staged_head_without_bubble_history() -> None:
    worker = _worker(data_version=4)
    worker.trainer._bubble_time_enabled = False

    def reject_history_lookup(version):
        pytest.fail("Sync must not select from Bubble's historical head cache")

    worker.trainer.select_target_lm_head_version = reject_history_lookup
    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert result["ready"]
    assert result["target_version"] == 4
    assert worker.trainer.requested_target_version is None
    assert worker.trainer.activation_calls == 1


def test_sync_preflight_still_rejects_wrong_staged_head() -> None:
    worker = _worker(data_version=4)
    worker.trainer._bubble_time_enabled = False
    worker.trainer._target_lm_head_weight_step = 3

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "target_version_mismatch"
    assert worker.trainer.activation_calls == 0


def test_bubble_sync_topup_still_requires_pinned_history() -> None:
    worker = _worker(data_version=4)
    worker.trainer._bubble_time_enabled = True
    worker.trainer.select_target_lm_head_version = lambda version: False

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "target_version_unavailable"
    assert worker.trainer.activation_calls == 0


def test_bubble_preflight_allows_newer_buffer_append_and_pins_target_version() -> None:
    worker = _worker(data_version=4)
    worker.trainer.buffer_version = 4
    plan = _plan()
    plan.update(
        {
            "execution_strategy": "rollout_idle_worker",
            "target_worker_ids": ("0",),
        }
    )

    result = asyncio.run(worker.preflight_drafter_training(plan))

    assert result["ready"]
    assert worker.trainer.requested_target_version == 4
    assert worker.trainer.reserved_plan_id == "plan-4"


def test_bubble_preflight_rejects_expired_plan_before_activation() -> None:
    worker = _worker(data_version=4)
    plan = _plan()
    plan.update(
        {
            "execution_strategy": "rollout_idle_worker",
            "target_worker_ids": ("0",),
            "deadline_ts": time.time() - 1.0,
            "idle_startup_reserve_sec": 2.0,
            "idle_batch_estimate_sec": 1.0,
            "idle_tail_reserve_sec": 1.0,
        }
    )

    result = asyncio.run(worker.preflight_drafter_training(plan))

    assert not result["ready"]
    assert result["reason"] == "plan_expired_before_preflight"
    assert result["remaining_sec"] < 0.0
    # deadline_ts already reserves idle_tail_reserve_sec, so preflight must not
    # count that reserve a second time.
    assert result["required_remaining_sec"] == 3.0
    assert worker.trainer.activation_calls == 0
    assert worker.trainer.reserved_plan_id is None


def test_worker_idle_prewarm_keeps_training_model_hot() -> None:
    worker = _worker(data_version=4)

    result = asyncio.run(worker.prewarm_drafter_training_model())

    assert result["activated"]
    assert result["reason"] == "prewarmed"
    assert worker.trainer.activation_calls == 1
    assert worker.trainer.release_after_activation_calls == 0
    assert worker.trainer.cleanup_calls == 0


def test_aborted_bubble_preflight_cleans_up_training_residency() -> None:
    worker = _worker(data_version=4)
    plan = _plan()
    plan.update(
        {
            "execution_strategy": "rollout_idle_worker",
            "target_worker_ids": ("0",),
        }
    )

    result = asyncio.run(worker.preflight_drafter_training(plan))
    abort_result = asyncio.run(worker.abort_drafter_training_preflight("plan-4"))

    assert result["ready"]
    assert abort_result["aborted"]
    assert worker.trainer.cleanup_calls == 1
    assert not worker.trainer.keep_hot


def test_full_collective_hot_bootstrap_retains_only_selected_worker() -> None:
    selected = _worker(data_version=4)
    selected._rank = 1
    other = _worker(data_version=4)
    plan = {"hot_bootstrap_worker_ids": ("1",)}

    assert selected._should_keep_drafter_training_hot(
        plan,
        execution_strategy="sync",
        successful_steps=10,
    )
    assert not other._should_keep_drafter_training_hot(
        plan,
        execution_strategy="sync",
        successful_steps=10,
    )
    assert not selected._should_keep_drafter_training_hot(
        plan,
        execution_strategy="sync",
        successful_steps=0,
    )


@pytest.mark.parametrize("trim_cache", [False, True])
def test_cleanup_keep_hot_does_not_offload_training_state(
    monkeypatch, trim_cache
) -> None:
    class _Optimizer:
        def __init__(self) -> None:
            self.zeroed = False

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            self.zeroed = True

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.model = object()
    trainer.optimizer = _Optimizer()
    trainer.collected_data = [object()]
    trainer.data_buffer = [object()]
    trainer._pending_checkpoint_future = None
    trainer._pending_full_checkpoint_future = None
    trainer._full_checkpoint_executor = None
    trainer._training_initialized = True
    trainer._training_active = True
    trainer._last_ckpt_step = 3
    trainer.training_steps = 7
    trainer._mark_buffer_changed = lambda: None
    cache_calls = []
    trainer.device_module = SimpleNamespace(
        empty_cache=lambda: cache_calls.append(True)
    )
    monkeypatch.setattr("verl_speco.trainer.base_trainer.device_name", "cuda")
    monkeypatch.setenv("VERL_DRAFTER_KEEP_HOT_TRIM_CACHE", str(trim_cache).lower())
    monkeypatch.delenv("VERL_DRAFTER_MEMORY_DIAGNOSTICS", raising=False)

    monkeypatch.setattr(
        "verl_speco.trainer.base_trainer.offload_fsdp_model_to_cpu",
        lambda model: pytest.fail("hot cleanup must not offload the model"),
    )

    asyncio.run(trainer.cleanup_training(clear_data=False, keep_hot=True))

    assert trainer.optimizer.zeroed
    assert trainer.collected_data
    assert trainer.data_buffer
    assert trainer._training_initialized
    assert not trainer._training_active
    assert trainer._last_ckpt_step == 3
    assert trainer.training_steps == 7
    assert len(cache_calls) == int(trim_cache)


def test_memory_diagnostics_counts_shared_storage_once(monkeypatch, caplog) -> None:
    import torch

    monkeypatch.setenv("VERL_DRAFTER_MEMORY_DIAGNOSTICS", "true")
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.runtime_device = torch.device("cpu")
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    trainer._target_lm_head_module = lambda: trainer.model
    trainer.optimizer = SimpleNamespace(state={0: {"shared": trainer.model.weight}})
    trainer.collected_data = [{"view": trainer.model.weight.detach().view(-1)}]
    trainer.device_module = SimpleNamespace(
        memory_allocated=lambda: 32,
        memory_reserved=lambda: 64,
        max_memory_allocated=lambda: 128,
    )
    trainer._log_training_memory("test")
    assert "'target_head': 16" in caplog.text
    assert "'model': 0" in caplog.text
    assert "'optimizer': 0" in caplog.text
    assert "'replay': 0" in caplog.text
    assert "cache_bytes=32" in caplog.text
    assert "unattributed_allocated_bytes=16" in caplog.text


def test_memory_diagnostics_failure_does_not_interrupt_training(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("VERL_DRAFTER_MEMORY_DIAGNOSTICS", "true")
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._log_training_memory("test")
    assert "Drafter memory diagnostics failed" in caplog.text


def test_memory_diagnostics_locates_internal_and_replay_storage(
    monkeypatch, caplog
) -> None:
    import torch

    monkeypatch.setenv("VERL_DRAFTER_MEMORY_DIAGNOSTICS", "true")
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.runtime_device = torch.device("cpu")
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    trainer.model.weight._full_param_padded = torch.ones(3)
    trainer.model._handle = SimpleNamespace(flat_param=trainer.model.weight)
    trainer._target_lm_head_module = lambda: trainer.model
    trainer.optimizer = SimpleNamespace(state={})
    trainer._training_replay_sessions = {2: {"items": [torch.ones(2)]}}
    trainer._scratch = torch.ones(1)
    trainer.device_module = SimpleNamespace(
        memory_allocated=lambda: 40,
        memory_reserved=lambda: 64,
        max_memory_allocated=lambda: 128,
    )
    trainer._log_training_memory("parked")
    assert "revision=3" in caplog.text
    assert "'replay': 8" in caplog.text
    assert "'model_internal': 12" in caplog.text
    assert "'trainer_runtime': 4" in caplog.text
    assert "flat_param._full_param_padded" in caplog.text
    assert "unattributed_allocated_bytes=0" in caplog.text


def test_cache_trim_releases_only_unused_cache(monkeypatch, caplog) -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    model = object()
    optimizer = object()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.optimizer_steps_total = 19
    state = {"reserved": 256}
    trainer.device_module = SimpleNamespace(
        memory_reserved=lambda: state["reserved"],
        memory_allocated=lambda: 128,
        empty_cache=lambda: state.update(reserved=128),
    )
    monkeypatch.setattr("verl_speco.trainer.base_trainer.device_name", "cuda")
    trainer._trim_keep_hot_cache()
    assert trainer.model is model and trainer.optimizer is optimizer
    assert trainer.optimizer_steps_total == 19
    assert "freed_reserved_bytes=128" in caplog.text
    assert "allocated_before_bytes=128 allocated_after_bytes=128" in caplog.text


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_bubble_fsdp1_publish_filters_before_host_copy(monkeypatch, dtype) -> None:
    import torch
    from contextlib import nullcontext

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.backend = SimpleNamespace(trains_draft_lm_head=False)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={"publish_dtype": dtype},
            )
        )
    )
    trainer.model = torch.nn.Module()
    trainer.model.embed_tokens = torch.nn.Embedding(4, 3)
    trainer.model.layer = torch.nn.Linear(3, 3)
    trainer.model.alias = trainer.model.layer
    trainer.model.register_buffer("scale", torch.ones(3))
    trainer.model.register_buffer("temporary", torch.ones(7), persistent=False)
    trainer._is_checkpoint_leader = lambda: True
    trainer._is_frozen_publish_param = lambda name: name.startswith("embed_tokens.")
    monkeypatch.setattr(
        "verl_speco.trainer.base_trainer.FSDP.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )
    expected = {
        name
        for name in trainer.model.state_dict()
        if not name.startswith("embed_tokens.")
    }
    result = trainer._get_bubble_fsdp1_publish_state()
    assert set(result) == expected
    assert all(value.device.type == "cpu" for value in result.values())
    assert all(
        value.dtype == (torch.bfloat16 if dtype == "bf16" else torch.float32)
        for value in result.values()
    )
    original = result["layer.weight"].clone()
    with torch.no_grad():
        trainer.model.layer.weight.add_(1)
    assert torch.equal(original, result["layer.weight"])
    trainer._is_checkpoint_leader = lambda: False
    assert trainer._get_bubble_fsdp1_publish_state() == {}


def test_bubble_publish_uses_replica_local_group_leader(monkeypatch) -> None:
    worker = _worker(data_version=4)
    worker.last_trained_step = 4
    worker.is_global_publish_leader = False
    worker.is_drafter_group_leader = True
    worker._last_trained_execution_strategy = "rollout_idle_worker"
    worker._last_trained_target_worker_ids = ("2", "3")
    # ray.put serializes a snapshot before the producer clears its host payload.
    monkeypatch.setattr("verl_speco.workers.speco_worker.ray.put", copy.deepcopy)

    result = worker.maybe_publish()

    assert result == {"weights_ref": {"weight": 4}}


def test_sync_publish_still_uses_global_leader(monkeypatch) -> None:
    worker = _worker(data_version=4)
    worker.last_trained_step = 4
    worker.is_global_publish_leader = False
    worker.is_drafter_group_leader = True
    worker._last_trained_execution_strategy = "sync"
    monkeypatch.setattr("verl_speco.workers.speco_worker.ray.put", copy.deepcopy)

    assert worker.maybe_publish() is None


def test_deferred_publish_stages_once_and_rejects_changed_optimizer(
    monkeypatch,
) -> None:
    worker = _worker(data_version=4)
    worker.last_global_step = worker.last_trained_step = 4
    worker.is_drafter_group_leader = True
    worker._last_trained_execution_strategy = "rollout_idle_worker"
    worker._last_trained_target_worker_ids = ("0",)
    worker._deferred_publish_snapshot_step = 4
    worker._deferred_publish_optimizer_step = 20
    worker.trainer.optimizer_steps_total = 21
    with pytest.raises(RuntimeError, match="changed before snapshot"):
        worker.maybe_publish()
    worker.trainer.optimizer_steps_total = 20
    calls = []
    worker.trainer.prepare_model_state_dict_for_publish = lambda step: calls.append(
        step
    )
    monkeypatch.setattr("verl_speco.workers.speco_worker.ray.put", copy.deepcopy)
    result = worker.maybe_publish()
    assert calls == [4]
    assert worker._deferred_publish_snapshot_step is None
    assert result is not None


def test_replica_local_idle_trainer_config_enables_use_orig_params() -> None:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker._rank = 2
    worker.config = OmegaConf.create(
        {
            "actor": {
                "fsdp_config": {
                    "use_orig_params": False,
                    "forward_prefetch": False,
                }
            },
            "rollout": {
                "drafter": {
                    "training": {
                        "fsdp_config": {
                            "use_orig_params": False,
                            "forward_prefetch": True,
                        }
                    }
                }
            },
        }
    )

    trainer_config = worker._replica_local_idle_trainer_config()

    assert trainer_config.actor.fsdp_config.use_orig_params is True
    assert trainer_config.rollout.drafter.training.fsdp_config.use_orig_params is True
    assert worker.config.actor.fsdp_config.use_orig_params is False
    assert worker.config.rollout.drafter.training.fsdp_config.use_orig_params is False


def test_base_trainer_orig_params_override_is_replica_local_bubble_only() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._bubble_time_enabled = True
    trainer.training_device_mesh = None
    trainer.training_process_group = object()
    trainer.training_group_world_size = 2

    assert trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer._bubble_time_enabled = False
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer._bubble_time_enabled = True
    trainer.training_device_mesh = object()
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer.training_device_mesh = None
    trainer.training_process_group = None
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()


def test_training_data_status_can_pin_an_older_target_version() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.current_rl_step = 5
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"use_logits": False}}}}
    )
    trainer.backend = SimpleNamespace(model_type="eagle3")
    trainer._bubble_time_enabled = True
    trainer.collected_data = deque()
    trainer.use_data_buffer = True
    trainer.data_buffer = DataBuffer(max_size=8)
    trainer.data_buffer.update_rl_step(4)
    trainer.data_buffer.add_batch({"target_version": 4, "loss_tokens": 10})
    trainer.data_buffer.add_batch({"target_version": 4, "loss_tokens": 20})
    trainer.data_buffer.update_rl_step(5)
    trainer.data_buffer.add_batch({"target_version": 5, "loss_tokens": 30})
    trainer._target_lm_head_snapshots = {4: object(), 5: object()}
    trainer.batch_size = 2
    trainer.buffer_version = 3

    status = trainer.get_training_data_status(
        sample_last_n_steps=2,
        target_version=4,
    )

    assert status["target_version"] == 4
    assert status["data_version"] == 4
    assert status["trainable_samples"] == 2
    assert status["trainable_batches"] == 1
    assert status["trainable_valid_tokens"] == 30


def test_training_data_status_handles_empty_buffer() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.current_rl_step = 5
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"use_logits": False}}}}
    )
    trainer.backend = SimpleNamespace(model_type="eagle3")
    trainer._bubble_time_enabled = True
    trainer.collected_data = deque()
    trainer.use_data_buffer = True
    trainer.data_buffer = DataBuffer(max_size=8)
    trainer._target_lm_head_snapshots = {}
    trainer.batch_size = 2
    trainer.buffer_version = 0

    status = trainer.get_training_data_status(
        sample_last_n_steps=2,
        target_version=4,
    )

    assert status["target_version"] is None
    assert status["data_version"] is None
    assert status["trainable_samples"] == 0
    assert status["trainable_batches"] == 0


def test_replica_local_checkpoint_leader_is_local_group_rank_zero() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._bubble_time_enabled = True
    trainer.training_device_mesh = None
    trainer.training_process_group = object()
    trainer.rank = 0
    trainer.rollout_dp_rank = 1

    assert trainer._is_checkpoint_leader()
