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
# Adapted original numerical validation; no external source imports.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl_speco.models.deepseek_v4.modeling_dspark import (
    DeepseekV4DSparkServingWrapper,
)


class _ToyDSparkModel(nn.Module):
    freeze_params = DeepseekV4DSparkServingWrapper.freeze_params

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.ModuleDict(
            {
                "gate": nn.Linear(4, 2, bias=False),
                "shared_experts": nn.Linear(4, 4, bias=False),
                "routed_experts": nn.Linear(4, 4, bias=False),
            }
        )
        self.self_attn = nn.Linear(4, 4, bias=False)

    def freeze_shared_token_weights(self) -> None:
        pass


def _config(*, freeze_mlp: bool, params_to_freeze=None):
    return SimpleNamespace(
        train=SimpleNamespace(
            dspark_freeze_mlp=freeze_mlp,
            params_to_freeze=list(params_to_freeze or []),
        )
    )


def test_dspark_mlp_freeze_is_disabled_by_default() -> None:
    model = _ToyDSparkModel()

    model.freeze_params(_config(freeze_mlp=False))

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.dspark_mlp_freeze_summary == {
        "enabled": False,
        "tensors": 0,
        "parameters": 0,
    }


def test_dspark_mlp_freeze_excludes_all_mlp_named_parameters() -> None:
    model = _ToyDSparkModel()

    model.freeze_params(_config(freeze_mlp=True))

    named_parameters = dict(model.named_parameters())
    mlp_parameters = {
        name: parameter
        for name, parameter in named_parameters.items()
        if "mlp" in name.lower()
    }
    assert mlp_parameters
    assert all(not parameter.requires_grad for parameter in mlp_parameters.values())
    assert named_parameters["self_attn.weight"].requires_grad
    assert model.dspark_mlp_freeze_summary == {
        "enabled": True,
        "tensors": len(mlp_parameters),
        "parameters": sum(parameter.numel() for parameter in mlp_parameters.values()),
    }


def test_dspark_mlp_freeze_preserves_explicit_freeze_rules() -> None:
    model = _ToyDSparkModel()

    model.freeze_params(
        _config(
            freeze_mlp=True,
            params_to_freeze=["self_attn.weight"],
        )
    )

    assert not model.self_attn.weight.requires_grad


@pytest.mark.parametrize("confidence_enabled", [False, True])
def test_unused_serving_confidence_is_frozen_without_disabling_training_head(
    confidence_enabled,
):
    model = _ToyDSparkModel()
    model.model = nn.Module()
    model.model.model = nn.Module()
    native_head = model.model.model.confidence_head = nn.Module()
    native_head.proj = nn.Linear(8, 1, bias=False)
    original = native_head.proj.weight.detach().clone()
    model.confidence_head = nn.Linear(8, 1) if confidence_enabled else None

    model.freeze_params(_config(freeze_mlp=False))

    assert not native_head.proj.weight.requires_grad
    torch.testing.assert_close(
        model.state_dict()["model.model.confidence_head.proj.weight"], original
    )
    assert model.self_attn.weight.requires_grad
    if confidence_enabled:
        assert all(
            parameter.requires_grad for parameter in model.confidence_head.parameters()
        )
        model.confidence_head(torch.randn(3, 8)).sum().backward()
        assert all(
            parameter.grad is not None
            for parameter in model.confidence_head.parameters()
        )
    assert native_head.proj.weight.grad is None
