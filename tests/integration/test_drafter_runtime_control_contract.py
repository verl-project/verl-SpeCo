from __future__ import annotations

from types import SimpleNamespace

import pytest


_speco_ray_trainer = pytest.importorskip(
    "verl_speco.trainer.speco_ray_trainer",
    reason="drafter runtime control contract needs the trainer dependency stack",
)
SpecoRayPPOTrainer = _speco_ray_trainer.SpecoRayPPOTrainer


class _FakeOldLogProbBatch:
    non_tensor_batch = {}

    def __init__(self) -> None:
        self.selected_non_tensor_keys = None

    def select(self, *, non_tensor_batch_keys=None, **kwargs):
        self.selected_non_tensor_keys = non_tensor_batch_keys
        return self

    def to_tensordict(self):
        raise AssertionError("non-collect old-logprob steps should not enter the collection compute path")


class _FakeRolloutWorkerGroup:
    def __init__(self) -> None:
        self.compute_log_prob_calls = 0

    def generate_sequences(self, *args, **kwargs):
        return SimpleNamespace(meta_info={"metrics": {}})

    def compute_log_prob(self, batch):
        self.compute_log_prob_calls += 1
        raise AssertionError("non-collect old-logprob steps should use the original compute path")


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


def test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config() -> None:
    assert _no_drafter_trainer(calculate_entropy=False)._speco_oldlogprob_entropy_hook_enabled() is True
    assert _no_drafter_trainer(calculate_entropy=True)._speco_oldlogprob_entropy_hook_enabled() is False
    assert _no_drafter_trainer()._speco_oldlogprob_entropy_hook_enabled() is False


def test_no_drafter_vllm_path_disables_async_scheduling_without_hiding_config(monkeypatch) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter scheduler contract needs verl and Ray",
    )
    from omegaconf import OmegaConf
    from verl_speco.integration import vllm_runtime

    bridge_calls = []
    monkeypatch.setattr(
        vllm_runtime,
        "install_upstream_vllm_runtime_bridge",
        lambda: bridge_calls.append("installed") or True,
    )

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"enable": False},
                    "engine_kwargs": {"vllm": {}},
                }
            }
        }
    )

    with task_runner._prepare_no_drafter_runtime_config(config):
        from verl_speco.integration.vllm_runtime import SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS

        assert config.actor_rollout_ref.rollout.drafter.enable is False
        assert config.actor_rollout_ref.rollout.engine_kwargs.vllm["no-async-scheduling"] is True
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm["worker_extension_cls"]
            == SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS
        )
    assert bridge_calls == ["installed"]

    assert "drafter" in config.actor_rollout_ref.rollout
    assert "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert "worker_extension_cls" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm


def test_task_runner_installs_vllm_import_compat_in_its_own_process(monkeypatch) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="task-runner import compatibility needs verl and Ray",
    )
    from omegaconf import OmegaConf
    from verl_speco.integration import verl_npu_vllm_compat

    calls = []
    monkeypatch.setattr(
        verl_npu_vllm_compat,
        "install_verl_npu_vllm_import_compat",
        lambda: calls.append("compat") or True,
    )

    assert task_runner._install_vllm_import_compat_for_task_runner(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm"}}})
    )
    assert not task_runner._install_vllm_import_compat_for_task_runner(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "sglang"}}})
    )
    assert calls == ["compat"]


def test_no_drafter_run_keeps_speco_entropy_control(monkeypatch) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter trainer contract needs verl and Ray",
    )
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"enable": False},
                    "engine_kwargs": {"vllm": {}},
                }
            }
        }
    )
    runner = task_runner.SpecoTaskRunner.__new__(task_runner.SpecoTaskRunner)
    observed = {}

    def fake_run_with_speco_trainer(self, active_config):
        del self
        observed["drafter_present"] = "drafter" in active_config.actor_rollout_ref.rollout
        observed["no_async"] = active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
            "no-async-scheduling"
        ]
        observed["worker_extension_cls"] = active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
            "worker_extension_cls"
        ]
        return "ran"

    monkeypatch.setattr(task_runner.SpecoTaskRunner, "_run_with_speco_trainer", fake_run_with_speco_trainer)

    assert runner.run(config) == "ran"
    from verl_speco.integration.vllm_runtime import SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS

    assert observed == {
        "drafter_present": True,
        "no_async": True,
        "worker_extension_cls": SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    }
    assert "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert "worker_extension_cls" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm


def test_oldlogprob_non_collect_step_uses_original_compute_path() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 2,
            "training_interval_steps": 1,
        },
        step=1,
    )
    trainer.config.actor_rollout_ref.actor.calculate_entropy = True
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp"
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer._update_actor = lambda *args, **kwargs: SimpleNamespace(meta_info={"metrics": {}})
    original_calls = []

    def original_compute_old_log_prob(batch):
        original_calls.append(batch)
        return "old-log-prob", 0.5

    trainer._compute_old_log_prob = original_compute_old_log_prob
    batch = _FakeOldLogProbBatch()

    with trainer._speco_online_fit_hooks():
        result = trainer._compute_old_log_prob(batch)

    assert result == ("old-log-prob", 0.5)
    assert original_calls == [batch]
    assert batch.selected_non_tensor_keys is None
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 0
    assert trainer._speco_last_collect_interval_matched == 0


def test_dspark_l1_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({"dspark_l1_loss_alpha": 0.9}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"


def test_dspark_default_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"
    assert trainer._speco_get_drafter_target_lm_head_row_selection() is None


def test_dspark_ce_only_oldlogprob_layout_keeps_aux_only_hidden() -> None:
    trainer = _trainer({"dspark_l1_loss_alpha": 0.0}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux"


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


def test_disabled_or_untrained_drafter_does_not_publish() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=1)
    assert trainer._speco_publish_drafter_weights(False) == {
        "drafter/publish_attempted": 0,
        "drafter/published": 0,
    }


def test_drafter_checkpoint_results_require_a_successful_training_replica() -> None:
    SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
        [
            {"saved": True, "reason": "saved"},
            {"saved": False, "reason": "not_checkpoint_replica"},
            {"saved": False, "reason": "not_in_training_group"},
        ],
        require_saved=True,
    )

    with pytest.raises(RuntimeError, match="produced no saved state"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "not_checkpoint_replica"}],
            require_saved=True,
        )


def test_drafter_checkpoint_results_propagate_save_failure() -> None:
    with pytest.raises(RuntimeError, match="missing_checkpoint_dir"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "missing_checkpoint_dir"}],
            require_saved=True,
        )
