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
    LinearWarmupDecayLR,
    build_drafter_lr_scheduler,
    resolve_standalone_lr_plan,
    restore_drafter_scheduler,
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


@pytest.mark.parametrize("standalone", [True, False], ids=["standalone", "online"])
def test_cosine_scheduler_follows_optimizer_step_budget(standalone) -> None:
    config = {
        "lr_scheduler_type": "cosine",
        "step": 10,
        "max_steps": 1000,
        "lr_warmup_steps": 100 if standalone else 2,
        "min_lr_ratio": 0.1,
    }
    if standalone:
        config["_standalone_max_steps"] = config["max_steps"]

    total = config["max_steps"] if standalone else config["step"]
    warmup = config["lr_warmup_steps"]
    optimizer = _optimizer(lr=1.0)
    scheduler = build_drafter_lr_scheduler(optimizer, config)
    for step in range(total + 1):
        if step < warmup:
            expected = step / warmup
        else:
            progress = (step - warmup) / (total - warmup)
            expected = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
        if step < total:
            _step(optimizer, scheduler, 1)


@pytest.mark.parametrize(("total", "warmup"), [(10, -1), (10, 10), (10, 11)])
def test_cosine_scheduler_rejects_invalid_step_budget(total, warmup) -> None:
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        build_drafter_lr_scheduler(
            _optimizer(),
            {
                "lr_scheduler_type": "cosine",
                "_standalone_max_steps": total,
                "step": 100,
                "lr_warmup_steps": warmup,
            },
        )


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


def test_linear_warmup_decay_reaches_zero_after_decay_steps() -> None:
    optimizer = _optimizer()
    scheduler = LinearWarmupDecayLR(
        optimizer,
        decay_steps=100,
        min_lr_ratio=0.0,
        warmup_steps=10,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    _step(optimizer, scheduler, 5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)

    _step(optimizer, scheduler, 5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    _step(optimizer, scheduler, 45)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)

    _step(optimizer, scheduler, 45)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    _step(optimizer, scheduler, 10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_scheduler_builder_uses_linear_warmup_decay_values() -> None:
    optimizer = _optimizer(lr=2e-5)
    scheduler = build_drafter_lr_scheduler(
        optimizer,
        {
            "lr_scheduler_type": "linear",
            "lr_decay_steps": 100,
            "min_lr_ratio": 0.1,
            "lr_warmup_steps": 10,
        },
    )

    _step(optimizer, scheduler, 10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)

    _step(optimizer, scheduler, 45)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.1e-5)

    _step(optimizer, scheduler, 45)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-6)


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


def test_linear_scheduler_builder_resumes_from_successful_optimizer_steps() -> None:
    optimizer = _optimizer()
    scheduler = build_drafter_lr_scheduler(
        optimizer,
        {
            "lr_scheduler_type": "linear",
            "lr_decay_steps": 100,
            "min_lr_ratio": 0.0,
            "lr_warmup_steps": 10,
            "_resume_optimizer_steps": 55,
        },
    )

    assert scheduler.last_epoch == 55
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)

    _step(optimizer, scheduler, 1)
    assert scheduler.last_epoch == 56
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4.888888888888889e-6)


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decay_steps": 0}, "lr_decay_steps"),
        ({"decay_steps": 10, "warmup_steps": 10}, "lr_warmup_steps"),
        ({"decay_steps": 10, "min_lr_ratio": -0.1}, "min_lr_ratio"),
    ],
)
def test_linear_warmup_decay_rejects_invalid_config(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        LinearWarmupDecayLR(_optimizer(), **kwargs)


@pytest.mark.parametrize("kind", ["cosine", "global_cosine", "linear", "constant"])
@pytest.mark.parametrize("override", [False, True])
def test_resume_extended_run_uses_resolved_plan(kind, override, caplog):
    config = {
        "_standalone_max_steps": 10,
        "lr_scheduler_type": kind,
        "lr_warmup_steps": 2,
        "min_lr_ratio": 0.25,
    }
    plan = resolve_standalone_lr_plan(config, {}, {})
    assert plan["lr_decay_steps"] == 10
    opt = _optimizer(1.0)
    scheduler = build_drafter_lr_scheduler(opt, {**config, **plan})
    _step(opt, scheduler, 6)

    resume = {"_standalone_max_steps": 20, "lr_scheduler_type": kind}
    if override:
        resume.update(lr_decay_steps=16, min_lr_ratio=0.0)
    state = scheduler.state_dict()
    resolved = resolve_standalone_lr_plan(resume, plan, state)
    assert ("learning rate may jump" in caplog.text) == override
    expected_plan = {
        **plan,
        **({"lr_decay_steps": 16, "min_lr_ratio": 0.0} if override else {}),
    }
    assert resolved == expected_plan

    resumed_opt = _optimizer(1.0)
    resumed = build_drafter_lr_scheduler(resumed_opt, {**resume, **resolved})
    restore_drafter_scheduler(resumed, state, standalone=True)
    resumed_opt.param_groups[0]["lr"] = resumed.get_last_lr()[0]
    assert resumed.get_last_lr() == scheduler.get_last_lr()
    reference_opt = _optimizer(1.0)
    reference = build_drafter_lr_scheduler(reference_opt, {**config, **expected_plan})
    _step(reference_opt, reference, 6)
    for _ in range(14):
        _step(reference_opt, reference, 1)
        _step(resumed_opt, resumed, 1)
        assert resumed.get_last_lr() == pytest.approx(reference.get_last_lr())
    if kind != "constant":
        assert resumed.get_last_lr() == pytest.approx([expected_plan["min_lr_ratio"]])


def test_legacy_linear_checkpoint_inherits_plan_except_explicit_warmup(caplog):
    scheduler = LinearWarmupDecayLR(
        _optimizer(), decay_steps=12, warmup_steps=2, min_lr_ratio=0.3
    )
    resolved = resolve_standalone_lr_plan(
        {"_standalone_max_steps": 20, "lr_warmup_steps": 0}, {}, scheduler.state_dict()
    )
    assert resolved == {"lr_decay_steps": 12, "lr_warmup_steps": 0, "min_lr_ratio": 0.3}
    assert "lr_warmup_steps" in caplog.text


@pytest.mark.parametrize("kind", ["cosine", "global_cosine", "clamped_global_cosine"])
def test_cosine_aliases_hold_floor_after_custom_cycles(kind):
    opt = _optimizer(1.0)
    scheduler = build_drafter_lr_scheduler(
        opt,
        {
            "lr_scheduler_type": kind,
            "lr_decay_steps": 8,
            "step": 4,
            "num_cycles": 1.0,
            "min_lr_ratio": 0.2,
        },
    )
    _step(opt, scheduler, 8)
    assert scheduler.get_last_lr() == pytest.approx([0.2])
    _step(opt, scheduler, 4)
    assert scheduler.get_last_lr() == pytest.approx([0.2])


@pytest.mark.parametrize("stop", [3, 0, -1])
def test_standalone_entry_resumes_decay_plan(tmp_path, monkeypatch, stop):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from verl_speco.standalone_ray_runtime import compose_runtime_config
    from verl_speco.trainer import base_trainer, draft_training_loop as loop

    class TinyModel(torch.nn.Linear):
        def save_pretrained(self, directory, *, state_dict, **kwargs):
            torch.save(state_dict, Path(directory) / "pytorch_model.bin")

    backend = SimpleNamespace(
        model_type="test",
        build_model=lambda: (TinyModel(1, 1), SimpleNamespace(pad_token_id=0)),
        setup_optimizer=lambda model, cfg: torch.optim.AdamW(
            model.parameters(), lr=1.0
        ),
        setup_scheduler=build_drafter_lr_scheduler,
    )
    rates = []

    async def train_batch(trainer, batch, step):
        trainer.model(torch.ones(1)).sum().backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.lr_scheduler.step()
        trainer.optimizer_steps_total += 1
        trainer.training_steps += 1
        rates.append(trainer.optimizer.param_groups[0]["lr"])
        return True

    monkeypatch.setattr(base_trainer, "get_device_id", lambda: 0)
    monkeypatch.setattr(base_trainer, "device_name", "cpu")
    monkeypatch.setattr(loop, "_configure_device", lambda rank: None)
    monkeypatch.setattr(loop, "_build_backend", lambda config: backend)
    monkeypatch.setattr(loop, "_build_standalone_tracking", lambda *args, **kw: [])
    monkeypatch.setattr(loop, "DraftFeatureDataLoader", lambda *args: [[1]] * 5)
    monkeypatch.setattr(
        loop.DrafterBaseTrainer,
        "prepare_training_batch_from_samples",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        loop.DrafterBaseTrainer, "training_step_from_batch", train_batch
    )

    def run(name, max_steps, *overrides):
        rates.clear()
        prefix = "actor_rollout_ref.rollout.drafter."
        config = compose_runtime_config(
            [
                prefix + f"checkpoint_path={tmp_path / name}",
                prefix + f"training.feature_store.path={tmp_path}",
                prefix + f"training.max_steps={max_steps}",
                prefix + "training.feature_store.repeat=false",
                prefix + "training.lr_scheduler_type=cosine",
                *[prefix + override for override in overrides],
            ]
        )
        return loop.run_standalone_draft_training(config)

    budget = ["training.lr_decay_steps=3"] if stop <= 0 else []
    result = run(
        "first",
        stop,
        *budget,
        "training.lr_warmup_steps=1",
        "training.min_lr_ratio=0.2",
    )
    completed = 5 if stop <= 0 else stop
    assert result["successful_steps"] == completed
    assert rates == pytest.approx([1, 0.6, 0.2, 0.2, 0.2][:completed])
    checkpoint = Path(result["last_save"]["path"])
    plan = json.loads((checkpoint / "metadata.json").read_text())["trainer_state"][
        "lr_schedule_plan"
    ]
    assert plan == {"lr_decay_steps": 3, "lr_warmup_steps": 1, "min_lr_ratio": 0.2}

    for explicit in (False, True):
        overrides = ["training.lr_decay_steps=9"] if explicit else []
        result = run(f"resume-{explicit}", 7, f"model_path={checkpoint}", *overrides)
        assert result["initial_optimizer_step"] == completed
        expected = (
            [
                0.6 + 0.4 * math.cos(math.pi * (s - 1) / 8)
                for s in range(completed + 1, 8)
            ]
            if explicit
            else [0.2] * (7 - completed)
        )
        assert rates == pytest.approx(expected)


@pytest.mark.parametrize("alias", ["global_cosine", "clamped_global_cosine"])
def test_legacy_cosine_alias_defaults(alias):
    opt = _optimizer(1.0)
    scheduler = build_drafter_lr_scheduler(opt, {"lr_scheduler_type": alias, "step": 7})
    assert scheduler.decay_steps == 100
    _step(opt, scheduler, 100)
    assert scheduler.get_last_lr() == pytest.approx([0.1])


def test_unbounded_run_without_decay_override_uses_finite_plan():
    config = {"lr_scheduler_type": "cosine", "_standalone_max_steps": -1}
    plan = resolve_standalone_lr_plan(config, {}, {})
    assert plan["lr_decay_steps"] == 100
    scheduler = build_drafter_lr_scheduler(_optimizer(), {**config, **plan})
    assert scheduler.decay_steps == 100
