from __future__ import annotations

from types import SimpleNamespace

import pytest


_speco_ray_trainer = pytest.importorskip(
    "verl_speco.trainer.speco_ray_trainer",
    reason="drafter runtime control contract needs the trainer dependency stack",
)
SpecoRayPPOTrainer = _speco_ray_trainer.SpecoRayPPOTrainer


def _trainer(training_cfg: dict, *, step: int = 1) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=True,
                    enable_drafter_training=True,
                    training=training_cfg,
                )
            )
        )
    )
    trainer._pending_drafter_publish_refs = None
    trainer._speco_last_collected_samples = 0
    trainer._ray_get_if_needed = lambda value: value
    return trainer


def _no_drafter_trainer(*, calculate_entropy=Ellipsis) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    actor = SimpleNamespace()
    if calculate_entropy is not Ellipsis:
        actor.calculate_entropy = calculate_entropy
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=actor,
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=False,
                    enable_drafter_training=False,
                    training={},
                )
            ),
        )
    )
    return trainer


def test_drafter_collect_train_and_publish_intervals() -> None:
    trainer = _trainer(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
        },
        step=6,
    )

    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is True
    assert trainer._speco_should_publish_drafter_weights(True) is False

    trainer.global_steps = 8
    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is False
    assert trainer._speco_should_publish_drafter_weights(True) is True
    assert trainer._speco_should_publish_drafter_weights(False) is False
    print("tests/integration/test_drafter_runtime_control_contract.py::test_drafter_collect_train_and_publish_intervals", flush=True)


def test_drafter_training_attempt_requires_interval_and_samples() -> None:
    trainer = _trainer({"training_interval_steps": 5}, step=4)
    trainer._speco_last_collected_samples = 10
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer.global_steps = 5
    trainer._speco_last_collected_samples = 0
    trainer._speco_oldlogprob_collection_requested = lambda: True
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer._speco_last_collected_samples = 1
    assert trainer._speco_should_attempt_drafter_train_this_step() is True
    print("tests/integration/test_drafter_runtime_control_contract.py::test_drafter_training_attempt_requires_interval_and_samples", flush=True)


def test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config() -> None:
    assert _no_drafter_trainer(calculate_entropy=False)._speco_oldlogprob_entropy_hook_enabled() is True
    assert _no_drafter_trainer(calculate_entropy=True)._speco_oldlogprob_entropy_hook_enabled() is False
    assert _no_drafter_trainer()._speco_oldlogprob_entropy_hook_enabled() is False
    print("tests/integration/test_drafter_runtime_control_contract.py::test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config", flush=True)


def test_async_publish_sets_pending_ref_and_waits_before_next_publish() -> None:
    calls: list[tuple[str, object, int]] = []
    waited: list[object] = []
    trainer = _trainer({"publish_interval_steps": 1, "publish_async": True}, step=10)
    trainer._pending_drafter_publish_refs = ["old-ref"]
    trainer._ray_get_if_needed = lambda value: waited.append(value) or value
    trainer._speco_get_published_drafter_weights = lambda: {"weights": 1}
    trainer._speco_actor_rollout_method = lambda name: (
        lambda payload, global_steps=None: calls.append((name, payload, global_steps)) or ["new-ref"]
    )

    metrics = trainer._speco_publish_drafter_weights(True)

    assert waited == [["old-ref"]]
    assert calls == [("update_draft_weights_async", {"weights": 1}, 10)]
    assert trainer._pending_drafter_publish_refs == ["new-ref"]
    assert metrics == {"drafter/publish_attempted": 1, "drafter/published": 1}
    print("tests/integration/test_drafter_runtime_control_contract.py::test_async_publish_sets_pending_ref_and_waits_before_next_publish", flush=True)


def test_disabled_or_untrained_drafter_does_not_publish() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=1)
    assert trainer._speco_publish_drafter_weights(False) == {
        "drafter/publish_attempted": 0,
        "drafter/published": 0,
    }
    print("tests/integration/test_drafter_runtime_control_contract.py::test_disabled_or_untrained_drafter_does_not_publish", flush=True)
