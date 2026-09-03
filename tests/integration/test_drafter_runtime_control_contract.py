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
        raise AssertionError(
            "non-collect old-logprob steps should not enter the collection compute path"
        )


class _FakeRolloutWorkerGroup:
    def __init__(self) -> None:
        self.compute_log_prob_calls = 0

    def generate_sequences(self, *args, **kwargs):
        return SimpleNamespace(meta_info={"metrics": {}})

    def compute_log_prob(self, batch):
        self.compute_log_prob_calls += 1
        raise AssertionError(
            "non-collect old-logprob steps should use the original compute path"
        )


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
            ),
        )
    )
    trainer._pending_drafter_publish_refs = None
    trainer._speco_last_collected_samples = 0
    trainer._ray_get_if_needed = lambda value: value
    trainer.speco_get_drafter_training_data_status = lambda *args: [
        {
            "available": True,
            "current_step": trainer.global_steps,
            "current_step_samples": trainer._speco_last_collected_samples,
            "buffer_samples": trainer._speco_last_collected_samples,
            "trainable_samples": trainer._speco_last_collected_samples,
            "trainable_batches": int(trainer._speco_last_collected_samples > 0),
            "batch_size_per_gpu": 1,
            "partial_batch_available": False,
            "oldest_sample_step": trainer.global_steps,
            "newest_sample_step": trainer.global_steps,
            "same_step_data_required": False,
            "target_version": trainer.global_steps,
        }
    ]
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


def _validation_config(
    *,
    val_batch_size=None,
    validation_batch_size=8,
    algorithm="DSPARK",
    enable_training=True,
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {"val_batch_size": val_batch_size},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": enable_training,
                        "speculative_algorithm": algorithm,
                        "training": {
                            "mode": "online",
                            "validation_batch_size": validation_batch_size,
                        },
                    },
                }
            },
        }
    )


def test_online_dspark_caps_unset_validation_batch_before_dataloader_init(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _validation_config()

    assert _speco_ray_trainer._speco_cap_online_dspark_validation_batch_size(config) == 8
    assert config.data.val_batch_size == 8


def test_online_dspark_preserves_explicit_validation_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _validation_config(val_batch_size=3)

    assert _speco_ray_trainer._speco_cap_online_dspark_validation_batch_size(config) == 3
    assert config.data.val_batch_size == 3


@pytest.mark.parametrize(
    ("algorithm", "enable_training"),
    [("EAGLE3", True), ("DSPARK", False)],
)
def test_validation_cap_does_not_change_other_runtime_modes(
    algorithm: str, enable_training: bool, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _validation_config(
        algorithm=algorithm, enable_training=enable_training
    )

    assert (
        _speco_ray_trainer._speco_cap_online_dspark_validation_batch_size(config)
        is None
    )
    assert config.data.val_batch_size is None


@pytest.mark.parametrize("value", [0, -1, True, "invalid"])
def test_online_dspark_rejects_invalid_validation_batch_size(
    value, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _validation_config(validation_batch_size=value)

    with pytest.raises(ValueError, match="must be a positive integer or null"):
        _speco_ray_trainer._speco_cap_online_dspark_validation_batch_size(config)


def test_validation_cap_preserves_mrv1_and_verl080_behavior(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    config = _validation_config()

    assert (
        _speco_ray_trainer._speco_cap_online_dspark_validation_batch_size(config)
        is None
    )
    assert config.data.val_batch_size is None


def test_drafter_collect_train_and_publish_intervals() -> None:
    trainer = _trainer(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
        },
        step=6,
    )

    config = trainer._speco_drafter_schedule_config()
    scheduler = trainer._speco_get_drafter_scheduler()
    assert scheduler.should_collect(trainer.global_steps, config) is True
    assert scheduler.training_interval_matched(trainer.global_steps, config) is True
    assert not scheduler.plan_publish(
        global_step=trainer.global_steps, drafter_trained=True, config=config
    ).publish

    trainer.global_steps = 8
    assert scheduler.should_collect(trainer.global_steps, config) is True
    assert scheduler.training_interval_matched(trainer.global_steps, config) is False
    assert scheduler.plan_publish(
        global_step=trainer.global_steps, drafter_trained=True, config=config
    ).publish
    assert not scheduler.plan_publish(
        global_step=trainer.global_steps, drafter_trained=False, config=config
    ).publish


def test_drafter_training_attempt_requires_interval_and_samples() -> None:
    trainer = _trainer({"training_interval_steps": 5}, step=4)
    trainer._speco_last_collected_samples = 10
    scheduler = trainer._speco_get_drafter_scheduler()
    config = trainer._speco_drafter_schedule_config()
    def plan():
        return scheduler.prepare_training_plan(
            trainer._speco_drafter_schedule_context(), config
        )
    assert plan().launch is False

    trainer.global_steps = 5
    trainer._speco_last_collected_samples = 0
    trainer._speco_oldlogprob_collection_requested = lambda: True
    assert plan().launch is False

    trainer._speco_last_collected_samples = 1
    assert plan().launch is True


def test_sync_scheduler_preserves_released_training_call_order() -> None:
    trainer = _trainer(
        {
            "training_interval_steps": 1,
            "publish_interval_steps": 0,
        },
        step=5,
    )
    trainer._speco_last_collected_samples = 1
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer._compute_old_log_prob = lambda batch: batch
    events = []

    trainer._speco_set_drafter_global_step = lambda **kwargs: events.append(
        "set_global_step"
    )
    trainer._speco_sync_target_lm_head_weight = lambda plan: events.append(
        "sync_target_lm_head"
    ) or {"drafter/target_lm_head_synced": 1}
    trainer._update_actor = lambda *args, **kwargs: events.append(
        "update_actor"
    ) or SimpleNamespace(meta_info={"metrics": {}})
    trainer._speco_train_drafter = lambda plan: events.append(
        ("train_drafter", plan.max_batches, plan.publish_after_success)
    ) or (
        True,
        {"drafter/trained": 1},
    )
    trainer._speco_publish_drafter_weights = lambda trained, plan: events.append(
        ("publish", trained)
    ) or {"drafter/publish_attempted": 1, "drafter/published": 1}

    with trainer._speco_online_fit_hooks():
        output = trainer._update_actor("batch")

    assert events == [
        "set_global_step",
        "sync_target_lm_head",
        "update_actor",
        ("train_drafter", 100, True),
        ("publish", True),
    ]
    assert output.meta_info["metrics"]["drafter/trained"] == 1
    assert output.meta_info["metrics"]["drafter/scheduler_used"] == 1
    assert output.meta_info["metrics"]["drafter/schedule_strategy"] == 0
    assert output.meta_info["metrics"]["drafter/schedule_reason"] == 3


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni"])
def test_oldlogprob_collection_accepts_supported_actor_backends(strategy: str) -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_hidden_states_from_sgl": False,
            "use_logits": False,
            "old_logprob_hidden_capture_impl": "forward_hook",
        }
    )
    trainer.config.actor_rollout_ref.actor.strategy = strategy

    assert trainer._speco_oldlogprob_collection_enabled() is True


def test_oldlogprob_collection_rejects_unknown_actor_backend() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_hidden_states_from_sgl": False,
            "use_logits": False,
        }
    )
    trainer.config.actor_rollout_ref.actor.strategy = "unknown"

    with pytest.raises(ValueError, match="fsdp/fsdp2/veomni"):
        trainer._speco_oldlogprob_collection_enabled()


def test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config() -> None:
    assert (
        _no_drafter_trainer(
            calculate_entropy=False
        )._speco_oldlogprob_entropy_hook_enabled()
        is True
    )
    assert (
        _no_drafter_trainer(
            calculate_entropy=True
        )._speco_oldlogprob_entropy_hook_enabled()
        is False
    )
    assert _no_drafter_trainer()._speco_oldlogprob_entropy_hook_enabled() is False


def test_no_drafter_vllm_path_disables_async_scheduling_without_hiding_config(
    monkeypatch,
) -> None:
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
        from verl_speco.integration.vllm_runtime import (
            SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
        )

        assert config.actor_rollout_ref.rollout.drafter.enable is False
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm["no-async-scheduling"]
            is True
        )
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm["worker_extension_cls"]
            == SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS
        )
    assert bridge_calls == ["installed"]

    assert "drafter" in config.actor_rollout_ref.rollout
    assert (
        "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )
    assert (
        "worker_extension_cls"
        not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )


def test_task_runner_installs_vllm_import_compat_in_its_own_process(
    monkeypatch,
) -> None:
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
        observed["drafter_present"] = (
            "drafter" in active_config.actor_rollout_ref.rollout
        )
        observed["no_async"] = (
            active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
                "no-async-scheduling"
            ]
        )
        observed["worker_extension_cls"] = (
            active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
                "worker_extension_cls"
            ]
        )
        return "ran"

    monkeypatch.setattr(
        task_runner.SpecoTaskRunner,
        "_run_with_speco_trainer",
        fake_run_with_speco_trainer,
    )

    assert runner.run(config) == "ran"
    from verl_speco.integration.vllm_runtime import (
        SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    )

    assert observed == {
        "drafter_present": True,
        "no_async": True,
        "worker_extension_cls": SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    }
    assert (
        "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )
    assert (
        "worker_extension_cls"
        not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )


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
    trainer._update_actor = lambda *args, **kwargs: SimpleNamespace(
        meta_info={"metrics": {}}
    )
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


@pytest.mark.parametrize(
    ("algorithm", "actor_backend", "actor_device_type", "export_strategy"),
    [
        ("DSPARK", "veomni", "npu", "veomni_lm_head_full"),
        ("DFLASH", "veomni", "npu", "veomni_lm_head_sparse"),
        ("EAGLE3", "veomni", "cuda", "veomni_lm_head_full"),
        ("EAGLE1", "fsdp", "npu", "engine_full_param"),
        ("DOMINO", "fsdp2", "cuda", "engine_full_param"),
    ],
)
def test_target_head_sync_defers_for_all_lm_head_drafters(
    algorithm: str,
    actor_backend: str,
    actor_device_type: str,
    export_strategy: str,
) -> None:
    trainer = _trainer({"training_interval_steps": 1}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = algorithm
    payload = {
        "weight": "cpu-weight",
        "actor_backend": actor_backend,
        "actor_device_type": actor_device_type,
        "export_strategy": export_strategy,
    }
    received = []
    trainer._speco_get_drafter_target_lm_head_row_selection = lambda: None
    trainer._speco_actor_rollout_method = lambda name: lambda rows: [payload]
    trainer._speco_build_drafter_target_lm_head_sync_args = (
        lambda value: (value, trainer.global_steps, 1)
    )
    trainer.speco_sync_target_lm_head_weight = (
        lambda value, global_step=None: received.append((value, global_step))
    )

    metrics = trainer._speco_sync_target_lm_head_weight()

    assert metrics["drafter/target_lm_head_apply_deferred"] == 1
    assert received[0][0].get("defer_device_apply", False) is True
    assert received[0][1] == 1


def test_target_head_transfer_waits_after_actor_update() -> None:
    trainer = _trainer({"training_interval_steps": 1}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"
    payload = {
        "weight": "cpu-weight",
        "actor_backend": "veomni",
        "actor_device_type": "npu",
        "export_strategy": "veomni_lm_head_full",
    }
    pending_refs = ["pending-target-sync"]
    resolved = []
    trainer._ray_get_if_needed = lambda value: resolved.append(value) or value
    trainer._speco_get_drafter_target_lm_head_row_selection = lambda: None
    trainer._speco_actor_rollout_method = lambda name: lambda rows: [payload]
    trainer._speco_build_drafter_target_lm_head_sync_args = (
        lambda value: (value, trainer.global_steps, 1)
    )
    trainer.speco_sync_target_lm_head_weight = (
        lambda value, global_step=None: pending_refs
    )

    metrics, pending = trainer._speco_start_target_lm_head_weight_sync()

    assert pending is not None
    assert metrics["drafter/target_lm_head_apply_deferred"] == 1
    assert resolved == [[payload]]

    metrics.update(trainer._speco_finish_target_lm_head_weight_sync(pending))

    assert resolved == [[payload], pending_refs]
    assert metrics["drafter/target_lm_head_synced"] == 1


def test_target_head_sync_is_skipped_when_training_uses_logits() -> None:
    trainer = _trainer({"training_interval_steps": 1, "use_logits": True}, step=1)

    def unexpected_actor_method(name):
        raise AssertionError(f"unexpected actor method lookup: {name}")

    trainer._speco_actor_rollout_method = unexpected_actor_method

    metrics, pending = trainer._speco_start_target_lm_head_weight_sync()

    assert metrics["drafter/target_lm_head_synced"] == 0
    assert pending is None


def test_target_head_worker_dispatch_is_nonblocking() -> None:
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl_speco.workers.speco_worker import SpecoWorker

    attrs = getattr(SpecoWorker.sync_target_lm_head_weight, MAGIC_ATTR)

    assert attrs["blocking"] is False


def test_async_publish_sets_pending_ref_and_waits_before_next_publish() -> None:
    calls: list[tuple[str, object, int]] = []
    waited: list[object] = []
    trainer = _trainer({"publish_interval_steps": 1, "publish_async": True}, step=10)
    trainer._pending_drafter_publish_refs = ["old-ref"]
    trainer._ray_get_if_needed = lambda value: waited.append(value) or value
    trainer._speco_get_published_drafter_weights = lambda: {"weights": 1}
    trainer._speco_actor_rollout_method = lambda name: (
        lambda payload, global_steps=None: calls.append((name, payload, global_steps))
        or ["new-ref"]
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


def test_drafter_checkpoint_saves_before_actor_checkpoint(monkeypatch) -> None:
    trainer = _trainer({}, step=20)
    events = []
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: events.append(
        ("drafter", kwargs)
    )
    parent_cls = SpecoRayPPOTrainer.__mro__[1]
    monkeypatch.setattr(
        parent_cls,
        "_save_checkpoint",
        lambda self: events.append(("actor", {})) or "saved",
    )

    assert trainer._save_checkpoint() == "saved"
    assert events == [
        ("drafter", {"wait": True}),
        ("actor", {}),
    ]


def test_actor_checkpoint_failure_preserves_previous_drafter(monkeypatch) -> None:
    trainer = _trainer({}, step=20)
    events = []
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: events.append(
        ("drafter", kwargs)
    )
    parent_cls = SpecoRayPPOTrainer.__mro__[1]

    def fail_actor_checkpoint(self):
        del self
        events.append(("actor", {}))
        raise RuntimeError("actor save failed")

    monkeypatch.setattr(parent_cls, "_save_checkpoint", fail_actor_checkpoint)

    with pytest.raises(RuntimeError, match="actor save failed"):
        trainer._save_checkpoint()
    assert events == [
        ("drafter", {"wait": True}),
        ("actor", {}),
    ]
