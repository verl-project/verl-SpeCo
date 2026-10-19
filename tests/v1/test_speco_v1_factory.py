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

import sys
import types

import pytest


def _install_fake_verl_v1(monkeypatch):
    """Provide enough of the V1 API to test factory selection without Ray."""
    verl = types.ModuleType("verl")
    trainer = types.ModuleType("verl.trainer")
    ppo = types.ModuleType("verl.trainer.ppo")
    v1 = types.ModuleType("verl.trainer.ppo.v1")

    class Base:
        pass

    v1.PPOTrainerSync = Base
    v1.PPOTrainerColocateAsync = type("Colocate", (Base,), {})
    v1.PPOTrainerSeparateAsync = type("Separate", (Base,), {})
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.trainer", trainer)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", ppo)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1", v1)


def test_factory_selects_all_v1_modes(monkeypatch):
    _install_fake_verl_v1(monkeypatch)
    from verl_speco.trainer.v1.factory import get_speco_v1_trainer_cls

    assert get_speco_v1_trainer_cls("sync").__name__ == "SpecoV1SyncTrainer"
    assert get_speco_v1_trainer_cls("colocate_async").__name__ == "SpecoV1ColocateAsyncTrainer"
    assert get_speco_v1_trainer_cls("separate_async").__name__ == "SpecoV1SeparateAsyncTrainer"


def test_factory_rejects_unknown_mode(monkeypatch):
    _install_fake_verl_v1(monkeypatch)
    from verl_speco.trainer.v1.factory import get_speco_v1_trainer_cls

    with pytest.raises(ValueError, match="Unknown SPECO V1 trainer mode"):
        get_speco_v1_trainer_cls("unsupported")
