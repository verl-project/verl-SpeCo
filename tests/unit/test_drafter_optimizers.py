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

import copy

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from verl_speco.backends.optimizers import (
    MuonAdamW,
    build_drafter_optimizer,
    split_named_params_for_muon,
)


class _DraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, hidden):
        return self.fc2(torch.relu(self.fc1(self.norm(hidden))))


def test_split_routes_2d_hidden_weights_to_muon() -> None:
    model = _DraftModel()

    muon_params, adamw_params = split_named_params_for_muon(model)

    muon_names = {name for name, _ in muon_params}
    adamw_names = {name for name, _ in adamw_params}

    assert muon_names == {"fc1.weight", "fc2.weight"}
    assert "norm.weight" in adamw_names
    assert "norm.bias" in adamw_names
    assert "embed_tokens.weight" in adamw_names
    assert "lm_head.weight" in adamw_names
    assert not muon_names & adamw_names


def test_split_excludes_frozen_parameters() -> None:
    model = _DraftModel()
    model.embed_tokens.weight.requires_grad_(False)
    model.fc1.weight.requires_grad_(False)

    muon_params, adamw_params = split_named_params_for_muon(model)

    names = {name for name, _ in muon_params} | {name for name, _ in adamw_params}
    assert "embed_tokens.weight" not in names
    assert "fc1.weight" not in names


def test_build_optimizer_defaults_to_adamw() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"lr": 1e-4})

    optimizer = build_drafter_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1


def test_build_optimizer_muon_splits_lrs() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "muon", "lr": 1e-4, "weight_decay": 1e-2})

    optimizer = build_drafter_optimizer(model, config)

    assert isinstance(optimizer, MuonAdamW)
    assert len(optimizer.param_groups) == 2
    muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
    adamw_group = next(g for g in optimizer.param_groups if not g["use_muon"])
    assert muon_group["lr"] == pytest.approx(10 * 1e-4)
    assert adamw_group["lr"] == pytest.approx(1e-4)


def test_muon_optimizer_steps_both_groups_and_round_trips_state() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "muon", "lr": 1e-3, "weight_decay": 0.0})
    optimizer = build_drafter_optimizer(model, config)

    muon_before = model.fc1.weight.detach().clone()
    adamw_before = model.norm.weight.detach().clone()
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()

    assert not torch.equal(muon_before, model.fc1.weight)
    assert not torch.equal(adamw_before, model.norm.weight)
    assert "momentum_buffer" in optimizer.state[model.fc1.weight]
    assert "exp_avg" in optimizer.state[model.norm.weight]

    state_dict = copy.deepcopy(optimizer.state_dict())
    restored = build_drafter_optimizer(model, config)
    restored.load_state_dict(state_dict)
    assert len(restored.state) == len(optimizer.state)


def test_unsupported_optimizer_raises() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "sgd", "lr": 1e-4})

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_drafter_optimizer(model, config)
