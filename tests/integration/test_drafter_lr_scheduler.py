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

import math

import pytest

torch = pytest.importorskip("torch")

from verl_speco.backends.lr_scheduler import (  # noqa: E402
    ClampedGlobalCosineLR,
    build_drafter_lr_scheduler,
)


def _optimizer(lr: float = 1e-5):
    parameter = torch.nn.Parameter(torch.ones(()))
    return torch.optim.SGD([parameter], lr=lr)


def _step(optimizer, scheduler, count: int) -> None:
    for _ in range(count):
        optimizer.step()
        scheduler.step()


def test_clamped_global_cosine_uses_global_steps_and_holds_floor() -> None:
    optimizer = _optimizer()
    scheduler = ClampedGlobalCosineLR(
        optimizer,
        decay_steps=100,
        min_lr_ratio=0.1,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    _step(optimizer, scheduler, 50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-6)

    _step(optimizer, scheduler, 50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)

    _step(optimizer, scheduler, 25)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert scheduler.last_epoch == 125


def test_scheduler_builder_allows_legacy_constant_override() -> None:
    optimizer = _optimizer()
    scheduler = build_drafter_lr_scheduler(
        optimizer,
        {
            "lr_scheduler_type": "global_cosine",
            "warmup_style": "constant",
            "lr_warmup_steps": 0,
        },
    )

    assert not isinstance(scheduler, ClampedGlobalCosineLR)
    _step(optimizer, scheduler, 125)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_scheduler_builder_uses_configured_global_cosine_values() -> None:
    optimizer = _optimizer(lr=2e-5)
    scheduler = build_drafter_lr_scheduler(
        optimizer,
        {
            "lr_scheduler_type": "global_cosine",
            "lr_decay_steps": 20,
            "min_lr_ratio": 0.25,
            "lr_warmup_steps": 0,
        },
    )

    _step(optimizer, scheduler, 20)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)

    _step(optimizer, scheduler, 5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


def test_scheduler_builder_resumes_from_successful_optimizer_steps() -> None:
    optimizer = _optimizer()
    scheduler = build_drafter_lr_scheduler(
        optimizer,
        {
            "lr_scheduler_type": "global_cosine",
            "lr_decay_steps": 100,
            "min_lr_ratio": 0.1,
            "lr_warmup_steps": 0,
            "_resume_optimizer_steps": 50,
        },
    )

    assert scheduler.last_epoch == 50
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-6)

    _step(optimizer, scheduler, 1)
    expected_ratio = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 0.51))
    assert scheduler.last_epoch == 51
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5 * expected_ratio)


def test_scheduler_builder_does_not_replace_explicit_invalid_decay() -> None:
    with pytest.raises(ValueError, match="lr_decay_steps"):
        build_drafter_lr_scheduler(
            _optimizer(),
            {
                "lr_scheduler_type": "global_cosine",
                "lr_decay_steps": 0,
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decay_steps": 0}, "lr_decay_steps"),
        ({"decay_steps": 10, "warmup_steps": 10}, "lr_warmup_steps"),
        ({"min_lr_ratio": 1.1}, "min_lr_ratio"),
    ],
)
def test_clamped_global_cosine_rejects_invalid_config(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ClampedGlobalCosineLR(_optimizer(), **kwargs)
