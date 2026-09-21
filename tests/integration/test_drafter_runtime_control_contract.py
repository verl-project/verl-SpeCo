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

from verl_speco.trainer.scheduler import CallbackDrafterWorkerExecutor


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
    trainer._pending_target_lm_head_sync = None
    trainer._speco_last_collected_samples = 0
    trainer._speco_get_drafter_scheduler().bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: [],
            resolve=lambda value: value,
            inspect_data=lambda last_n, full_batch: (
                trainer.speco_get_drafter_training_data_status(last_n, full_batch)
            ),
            prepare=trainer._speco_prepare_drafter_training_rpc,
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )
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


def test_oldlogprob_collection_uses_collect_interval_with_data_buffer() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "use_data_buffer": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 100,
        }
    )
    trainer._speco_online_enabled = lambda: True

    plan = trainer._speco_plan_drafter_collection(
        _speco_ray_trainer.DrafterCollectionSource.OLD_LOGPROB
    )

    assert plan.collect is True
    assert plan.reason == "collection_enabled"


def test_oldlogprob_collection_without_data_buffer_keeps_training_gate() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "use_data_buffer": False,
            "collect_interval_steps": 1,
            "training_interval_steps": 100,
        }
    )
    trainer._speco_online_enabled = lambda: True

    plan = trainer._speco_plan_drafter_collection(
        _speco_ray_trainer.DrafterCollectionSource.OLD_LOGPROB
    )

    assert plan.collect is False
    assert plan.reason == "training_interval_not_reached"


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
    trainer._speco_start_target_lm_head_weight_sync = lambda plan: (
        events.append("sync_target_lm_head")
        or ({"drafter/target_lm_head_synced": 1}, None)
    )
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
    assert output.meta_info["metrics"]["drafter/schedule_launch"] == 1


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

    with pytest.raises(ValueError, match="fsdp/fsdp2/megatron/veomni"):
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


def test_no_drafter_run_refuses_and_leaves_vllm_config_untouched(
    monkeypatch,
) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter runner contract needs verl and Ray",
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
    runner = task_runner.SpecoTaskRunner.__new__(task_runner.SpecoTaskRunner)

    with pytest.raises(RuntimeError, match="drafter.enable=true"):
        runner.run(config)

    # The no-drafter path is bypassed at the entry point.  Even if the SPECO
    # runner is reused by accident, it must refuse before installing the vLLM
    # runtime bridge, forcing async scheduling off, or injecting the SPECO
    # weight-sync compat extension.
    assert bridge_calls == []
    vllm_engine = config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert "no-async-scheduling" not in vllm_engine
    assert "worker_extension_cls" not in vllm_engine



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


def test_no_drafter_run_does_not_install_vllm_import_compat(monkeypatch) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter runner contract needs verl and Ray",
    )
    from omegaconf import OmegaConf
    from verl_speco.integration import verl_npu_vllm_compat

    compat_calls = []
    monkeypatch.setattr(
        verl_npu_vllm_compat,
        "install_verl_npu_vllm_import_compat",
        lambda: compat_calls.append("compat") or True,
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
    runner = task_runner.SpecoTaskRunner.__new__(task_runner.SpecoTaskRunner)

    with pytest.raises(RuntimeError, match="drafter.enable=true"):
        runner.run(config)

    # A no-drafter run must never install the SPECO vLLM import-compat mixin;
    # the runner refuses before reaching the import-compat step.
    assert compat_calls == []



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
    trainer._speco_actor_rollout_method = lambda name: lambda rows, **kwargs: [payload]
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
    trainer._speco_actor_rollout_method = lambda name: lambda rows, **kwargs: [payload]
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


def test_legacy_skip_compat_does_not_require_recipe_v1_fields() -> None:
    from omegaconf import OmegaConf

    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"use_v1": False}})

    trainer._speco_ensure_legacy_skip_config()

    assert trainer.config.trainer.v1.trainer_mode == "sync"


def test_legacy_skip_compat_leaves_v1_config_untouched() -> None:
    from omegaconf import OmegaConf

    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"trainer": {"use_v1": True, "v1": {"trainer_mode": "separate_async"}}}
    )

    trainer._speco_ensure_legacy_skip_config()

    assert trainer.config.trainer.v1.trainer_mode == "separate_async"


def test_sync_publish_failure_restores_last_committed_drafter_payload() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=8)
    trainer._speco_last_published_drafter_payload = {"weights": "old"}
    trainer._speco_last_published_drafter_step = 7
    calls = []

    def update(payload, global_steps=None):
        calls.append((payload, global_steps))
        return "new-ref" if payload["weights"] == "new" else "rollback-ref"

    trainer._speco_actor_rollout_method = lambda _name: update

    def resolve(value):
        if value == "new-ref":
            raise RuntimeError("injected replica failure")
        return value

    trainer._ray_get_if_needed = resolve

    with pytest.raises(RuntimeError, match="injected replica failure"):
        trainer._speco_update_rollout_drafter_weights(
            {"weights": "new"}, global_step=8, asynchronous=False
        )

    assert calls == [({"weights": "new"}, 8), ({"weights": "old"}, 7)]


def test_async_publish_failure_restores_last_committed_drafter_payload() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=8)
    trainer._speco_last_published_drafter_payload = {"weights": "old"}
    trainer._speco_last_published_drafter_step = 7
    calls = []

    def update(payload, global_steps=None):
        calls.append((payload, global_steps))
        return "new-ref" if payload["weights"] == "new" else "rollback-ref"

    trainer._speco_actor_rollout_method = lambda _name: update

    def resolve(value):
        if value == "new-ref":
            raise RuntimeError("injected async replica failure")
        return value

    trainer._ray_get_if_needed = resolve
    trainer._speco_update_rollout_drafter_weights(
        {"weights": "new"}, global_step=8, asynchronous=True
    )

    with pytest.raises(RuntimeError, match="injected async replica failure"):
        trainer._speco_wait_pending_drafter_publish_rpc()

    assert calls == [({"weights": "new"}, 8), ({"weights": "old"}, 7)]


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
    assert metrics["drafter/publish_attempted"] == 1
    assert metrics["drafter/published"] == 1


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


def _oldlogprob_collect_plan_trainer() -> SpecoRayPPOTrainer:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 2,
        }
    )
    trainer._speco_oldlogprob_collection_enabled = lambda: True
    trainer._speco_plan_drafter_collection = lambda source: SimpleNamespace(
        collect=True,
        sample_rate=1.0,
        max_samples_per_replica=None,
        max_tokens_per_replica=None,
        collection_id="response-mask-contract",
    )
    trainer._speco_log_drafter_collection_plan = lambda plan: None
    trainer._speco_owner_bucket_count = lambda: 1
    trainer._speco_dispatch_bucket_count = lambda: None
    return trainer


def _oldlogprob_batch(response_mask: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        batch={
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22, 23]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "response_mask": torch.tensor([response_mask], dtype=torch.long),
        }
    )


def test_oldlogprob_collect_plan_honors_response_mask_before_window_selection() -> None:
    trainer = _oldlogprob_collect_plan_trainer()

    plan = trainer._speco_build_oldlogprob_collect_plan(
        _oldlogprob_batch([1, 1, 0])
    )

    assert plan is None
    assert trainer._speco_last_oldlogprob_candidate_samples == 0
    assert trainer._speco_last_oldlogprob_short_response_skipped == 1


def test_oldlogprob_collect_plan_tracks_short_response_skip_metric_state() -> None:
    trainer = _oldlogprob_collect_plan_trainer()

    plan = trainer._speco_build_oldlogprob_collect_plan(
        _oldlogprob_batch([1, 1, 1])
    )

    assert plan is not None
    assert plan["response_lens"] == [3]
    assert plan["selected_count"] == 1
    assert trainer._speco_last_oldlogprob_short_response_skipped == 0


def test_oldlogprob_collection_omits_masked_response_token_from_payload() -> None:
    trainer = _oldlogprob_collect_plan_trainer()
    batch = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22, 23, 24]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }
    )
    collect_plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert collect_plan is not None
    captured = {}
    trainer._speco_execute_collection = lambda plan, payload: (
        captured.setdefault("payload", payload)
        or SimpleNamespace(collected_samples=1)
    )
    output = {
        _speco_ray_trainer.OLD_LOGPROB_HIDDEN_STATES_KEY: torch.arange(
            6, dtype=torch.float32
        ).reshape(1, 3, 2)
    }

    assert trainer._speco_collect_oldlogprob_features(batch, collect_plan, output) == 1
    sample = captured["payload"].buckets[0][0]
    assert sample["responses"].tolist() == [[21, 22, 23]]
    assert sample["input_ids"].tolist() == [[11, 12, 21, 22, 23]]
