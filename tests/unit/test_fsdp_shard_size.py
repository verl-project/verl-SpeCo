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


def _trainer(actor_fsdp=None, training=None) -> DrafterBaseTrainer:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = SimpleNamespace(
        actor=OmegaConf.create({"fsdp_config": actor_fsdp or {}}),
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
