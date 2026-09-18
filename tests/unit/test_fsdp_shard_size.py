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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from verl_speco.trainer.base_trainer import DrafterBaseTrainer  # noqa: E402


def _trainer(actor_fsdp=None, training=None, actor_strategy="fsdp2") -> DrafterBaseTrainer:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = SimpleNamespace(
        actor=OmegaConf.create(
            {"fsdp_config": actor_fsdp or {}, "strategy": actor_strategy}
        ),
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training=OmegaConf.create(training or {}))
        ),
    )
    return trainer


def _fake_mesh(world_size: int = 4):
    return SimpleNamespace(
        size=lambda: world_size,
        mesh=torch.arange(world_size, dtype=torch.int64).reshape(1, world_size),
        device_type="cpu",
    )


def test_shard_size_from_actor_fsdp_config() -> None:
    trainer = _trainer(actor_fsdp={"fsdp_shard_size": 2})
    assert trainer._resolve_fsdp_shard_size() == 2


def test_shard_size_falls_back_to_drafter_training() -> None:
    trainer = _trainer(actor_fsdp={"fsdp_shard_size": None}, training={"fsdp_shard_size": 1})
    assert trainer._resolve_fsdp_shard_size() == 1


def test_shard_size_defaults_to_none() -> None:
    assert _trainer()._resolve_fsdp_shard_size() is None
    assert _trainer(actor_fsdp={"fsdp_shard_size": -1})._resolve_fsdp_shard_size() is None


def test_standalone_default_replicates_but_preserves_explicit() -> None:
    from verl_speco.trainer.draft_training_loop import (
        _apply_standalone_fsdp_shard_default,
    )

    unset = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"fsdp_shard_size": None}}}}
    )
    _apply_standalone_fsdp_shard_default(unset)
    assert unset.rollout.drafter.training.fsdp_shard_size == 1

    explicit = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"fsdp_shard_size": 4}}}}
    )
    _apply_standalone_fsdp_shard_default(explicit)
    assert explicit.rollout.drafter.training.fsdp_shard_size == 4


def test_unset_or_full_shard_keeps_mesh() -> None:
    mesh = _fake_mesh(4)
    assert _trainer(training={"fsdp_shard_size": None})._shard_sized_fsdp_mesh(mesh) is mesh
    assert _trainer(training={"fsdp_shard_size": 4})._shard_sized_fsdp_mesh(mesh) is mesh


def test_non_divisible_shard_size_raises() -> None:
    trainer = _trainer(training={"fsdp_shard_size": 3})
    with pytest.raises(ValueError, match="must divide"):
        trainer._shard_sized_fsdp_mesh(_fake_mesh(4))


def test_ddp_full_training_group_spans_full_mesh() -> None:
    sentinel = object()
    mesh = _fake_mesh(8)
    mesh.get_group = lambda: sentinel
    trainer = _trainer()
    trainer.training_device_mesh = mesh
    trainer.training_process_group = object()
    trainer.training_group_world_size = 4

    assert trainer._full_training_group() is sentinel
    assert trainer._full_training_world_size() == 8


def test_ddp_full_training_group_without_mesh() -> None:
    group = object()
    trainer = _trainer()
    trainer.training_device_mesh = None
    trainer.training_process_group = group
    trainer.training_group_world_size = 2

    assert trainer._full_training_group() is group
    assert trainer._full_training_world_size() == 2


def test_ddp_state_dict_avoids_fsdp2_helper(monkeypatch) -> None:
    import verl_speco.trainer.base_trainer as base_trainer

    trainer = _trainer()
    # Online DDP still has a non-null mesh; it must not be routed through FSDP2.
    trainer.training_device_mesh = object()
    inner = torch.nn.Linear(4, 4)
    trainer.model = SimpleNamespace(module=inner)
    monkeypatch.setattr(trainer, "_is_ddp_wrapped", lambda: True)

    def _fail(*args, **kwargs):
        raise AssertionError("DDP must not use get_fsdp_full_state_dict")

    monkeypatch.setattr(base_trainer, "get_fsdp_full_state_dict", _fail)

    full_state = trainer._get_full_model_state_dict()
    assert set(full_state) == set(inner.state_dict())
    assert set(trainer._get_full_export_state_dict()) == set(inner.state_dict())


def test_full_state_dict_uses_fsdp_when_distributed(monkeypatch) -> None:
    import verl_speco.trainer.base_trainer as base_trainer

    sentinel = {"weight": torch.zeros(1)}
    calls = []

    def _fake_fsdp(model, offload_to_cpu=True, rank0_only=True):
        calls.append((model, offload_to_cpu, rank0_only))
        return sentinel

    monkeypatch.setattr(base_trainer, "get_fsdp_full_state_dict", _fake_fsdp)
    monkeypatch.setattr(base_trainer.dist, "is_initialized", lambda: True)

    trainer = _trainer()
    trainer.training_device_mesh = object()
    trainer.model = torch.nn.Linear(2, 2)

    assert trainer._get_full_model_state_dict() is sentinel
    assert calls and calls[0][1:] == (True, True)


def test_drafter_strategy_prefers_drafter_specific_setting() -> None:
    trainer = _trainer(actor_strategy="fsdp2", training={"strategy": "ddp"})
    assert trainer._drafter_strategy() == "ddp"


def test_drafter_strategy_falls_back_to_actor_strategy() -> None:
    assert _trainer(actor_strategy="veomni", training={"strategy": None})._drafter_strategy() == "veomni"
    assert _trainer(actor_strategy="fsdp2")._drafter_strategy() == "fsdp2"


def test_resolve_drafter_strategy_prefers_training() -> None:
    from verl_speco.trainer.base_trainer import resolve_drafter_strategy

    explicit = OmegaConf.create(
        {
            "actor": {"strategy": "fsdp2"},
            "rollout": {"drafter": {"training": {"strategy": "ddp"}}},
        }
    )
    assert resolve_drafter_strategy(explicit) == "ddp"

    inherited = OmegaConf.create(
        {
            "actor": {"strategy": "fsdp2"},
            "rollout": {"drafter": {"training": {}}},
        }
    )
    assert resolve_drafter_strategy(inherited) == "fsdp2"


def test_shard_size_prefers_drafter_over_actor() -> None:
    trainer = _trainer(
        actor_fsdp={"fsdp_shard_size": 2},
        training={"strategy": "fsdp2", "fsdp_shard_size": 4},
    )
    assert trainer._resolve_fsdp_shard_size() == 4


def test_npu_veomni_helpers_follow_drafter_strategy(monkeypatch) -> None:
    # A drafter using fsdp2/ddp must not inherit an actor-side veomni path.
    import verl_speco.trainer.base_trainer as base_trainer

    monkeypatch.setattr(base_trainer, "device_name", "npu")

    def _npu_trainer(strategy):
        trainer = _trainer(actor_strategy="veomni", training={"strategy": strategy})
        trainer.backend = SimpleNamespace(model_type="dspark")
        trainer.training_device_mesh = object()
        trainer.dp_group_world_size = 2
        trainer.park_hccl_after_drafter_training = True
        return trainer

    ddp = _npu_trainer("ddp")
    assert ddp._drafter_strategy() == "ddp"
    assert not ddp._use_flattened_drafter_fsdp_mesh()
    assert not ddp._use_blocking_npu_optimizer_offload()
    assert not ddp._should_park_drafter_hccl()

    veomni = _npu_trainer("veomni")
    assert veomni._use_flattened_drafter_fsdp_mesh()
    assert veomni._use_blocking_npu_optimizer_offload()
    assert veomni._should_park_drafter_hccl()



