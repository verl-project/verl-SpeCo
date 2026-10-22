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

import json
from types import SimpleNamespace

import pytest

from verl_speco.trainer.scheduler import CallbackDrafterWorkerExecutor

torch = pytest.importorskip(
    "torch",
    reason="drafter runtime control contract needs PyTorch",
)

_speco_ray_trainer = pytest.importorskip(
    "verl_speco.trainer.speco_ray_trainer",
    reason="drafter runtime control contract needs the trainer dependency stack",
)
SpecoRayPPOTrainer = _speco_ray_trainer.SpecoRayPPOTrainer
_base_trainer = pytest.importorskip("verl_speco.trainer.base_trainer")
DrafterBaseTrainer = _base_trainer.DrafterBaseTrainer


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


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def _oldlogprob_plan_batch(batch_size: int, *, non_tensor_batch=None):
    prompt_width = 4
    response_width = 520
    return SimpleNamespace(
        batch={
            "prompts": torch.ones(batch_size, prompt_width, dtype=torch.long),
            "responses": torch.ones(batch_size, response_width, dtype=torch.long),
            "attention_mask": torch.ones(
                batch_size, prompt_width + response_width, dtype=torch.long
            ),
            "response_mask": torch.ones(batch_size, response_width, dtype=torch.long),
        },
        non_tensor_batch=non_tensor_batch or {},
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
    trainer._speco_ready_target_lm_head_versions = set()
    trainer._speco_ready_target_lm_head_workers = {}
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

    trainer._speco_pending_request_accept_len_payloads = []
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


@pytest.mark.parametrize(
    ("strategy", "expected_collect", "expected_reason"),
    [
        ("rollout_idle_worker", True, "collection_enabled"),
        ("sync", False, "training_interval_not_reached"),
    ],
)
def test_bubble_oldlogprob_collection_uses_only_collection_interval(
    strategy: str,
    expected_collect: bool,
    expected_reason: str,
) -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 4,
            "training_interval_steps": 6,
            "scheduler": {"execution": {"strategy": strategy}},
        },
        step=4,
    )

    plan = trainer._speco_plan_drafter_collection(
        _speco_ray_trainer.DrafterCollectionSource.OLD_LOGPROB
    )

    assert plan.collect is expected_collect
    assert plan.collect_interval_matched
    assert not plan.training_interval_matched
    assert plan.reason == expected_reason


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


def test_deferred_bubble_publish_drains_before_next_generation() -> None:
    """A completed Bubble plan publishes before serving the next rollout."""
    trainer = _trainer(
        {"scheduler": {"execution": {"strategy": "rollout_idle_worker"}}},
        step=5,
    )
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer._compute_old_log_prob = lambda batch: batch
    trainer._update_actor = lambda batch: batch
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda: None)
    events = []
    trainer.actor_rollout_wg.generate_sequences = lambda: (
        events.append("generate") or SimpleNamespace(meta_info={"metrics": {}})
    )
    trainer._speco_get_drafter_scheduler().plan_publish = lambda **kwargs: (
        SimpleNamespace(publish=True)
    )
    completed = [SimpleNamespace(plan_id="completed-quota", publish_after_success=True)]

    def reclaim(*, on_completed):
        if completed:
            return on_completed(completed.pop(), SimpleNamespace(trained=True))
        return {}

    trainer._speco_reclaim_rollout_idle_workers_before_post_rollout = reclaim
    trainer._speco_set_drafter_global_step = lambda **kwargs: None
    trainer._speco_publish_drafter_weights = lambda trained, plan, **kwargs: (
        events.append(("publish", trained, kwargs.get("after_weight_update")))
        or {"drafter/publish_attempted": 1, "drafter/published": 1}
    )
    trainer._speco_reclaim_rollout_idle_workers_before_generation = lambda: {}
    trainer._speco_emit_rollout_generation_started = lambda: {}
    trainer._speco_start_rollout_idle_event_loop = lambda: (None, None)
    trainer._speco_stop_rollout_idle_event_loop = lambda *args: None
    trainer._speco_service_rollout_idle_events = lambda **kwargs: {}
    trainer._speco_emit_rollout_generation_completed = lambda output, **kwargs: {}
    trainer._speco_store_rollout_metrics = lambda output: None
    trainer._speco_collect_generation_samples = lambda output: 0

    with trainer._speco_online_fit_hooks():
        trainer.actor_rollout_wg.generate_sequences()
        assert events == ["generate"]
        output = trainer.actor_rollout_wg.generate_sequences()

    assert events == ["generate", ("publish", True, True), "generate"]
    assert output.meta_info["metrics"]["drafter/published"] == 1
    assert output.meta_info["metrics"]["timing_s/drafter_publish_critical_path"] >= 0.0


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
    trainer._speco_build_drafter_target_lm_head_sync_args = lambda value: (
        value,
        trainer.global_steps,
        1,
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
    trainer._speco_build_drafter_target_lm_head_sync_args = lambda value: (
        value,
        trainer.global_steps,
        1,
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


def test_bubble_target_head_prefetch_resolves_fetch_before_dispatch() -> None:
    trainer = _trainer(
        {"scheduler": {"execution": {"strategy": "rollout_idle_worker"}}},
        step=1,
    )
    payload = {"weight": "cpu-weight", "export_strategy": "full"}
    pending_refs = [{"worker_id": "0", "accepted": True, "global_step": 1}]
    resolved = []
    trainer._ray_get_if_needed = lambda value: resolved.append(value) or value
    trainer._speco_get_drafter_target_lm_head_row_selection = lambda: None
    trainer._speco_actor_rollout_method = lambda name: lambda rows, **kwargs: [payload]
    trainer._speco_build_drafter_target_lm_head_sync_args = lambda value, global_step=None: (
        value,
        trainer.global_steps,
        1,
    )
    trainer.speco_sync_target_lm_head_weight = (
        lambda value, global_step=None: pending_refs
    )

    metrics, pending = trainer._speco_start_target_lm_head_weight_sync()

    assert pending is not None
    assert pending["fetch_refs"] == [payload]
    assert "refs" not in pending
    assert pending["stage"] == "fetch"
    assert pending["target_version"] == trainer.global_steps
    # Bubble prefetch submits the actor export without blocking. The finish
    # boundary resolves it before dispatching the snapshot to the drafter.
    assert resolved == []
    assert metrics["timing_s/drafter_target_lm_head_fetch_submit"] >= 0.0
    assert metrics["timing_s/drafter_target_lm_head_prefetch_submit_critical_path"] >= 0.0

    metrics.update(trainer._speco_finish_target_lm_head_weight_sync(pending))

    assert resolved == [[payload], pending_refs]
    assert metrics["drafter/target_lm_head_synced"] == 1


def test_bubble_target_head_prefetch_ignores_transient_fallback_plan() -> None:
    trainer = _trainer(
        {"scheduler": {"execution": {"strategy": "rollout_idle_worker"}}},
        step=4,
    )
    trainer._speco_last_collected_samples = 31
    trainer._speco_last_collected_step = trainer.global_steps

    # A scheduler event may temporarily select a synchronous fallback plan
    # when no version is trainable.  Prefetch eligibility must continue to
    # follow the configured Bubble mode so that the next window can recover.
    assert trainer._speco_should_prefetch_target_lm_head()

    trainer._speco_last_collected_samples = 0
    assert not trainer._speco_should_prefetch_target_lm_head()

    trainer.config.actor_rollout_ref.rollout.drafter.training = {
        "scheduler": {"execution": {"strategy": "sync"}}
    }
    trainer._speco_last_collected_samples = 31
    assert not trainer._speco_should_prefetch_target_lm_head()


def test_bubble_target_head_ready_is_scoped_to_synced_workers() -> None:
    trainer = _trainer(
        {"scheduler": {"execution": {"strategy": "rollout_idle_worker"}}},
        step=1,
    )

    trainer._speco_mark_target_lm_head_version_ready(
        2,
        worker_ids=("0", "1"),
    )

    assert trainer._speco_target_lm_head_ready_for_workers(2, ("0", "1"))
    assert not trainer._speco_target_lm_head_ready_for_workers(2, ("2", "3"))

    trainer._speco_mark_target_lm_head_version_ready(
        4,
        worker_ids=("0", "1", "2", "3"),
    )

    assert trainer._speco_target_lm_head_ready_for_workers(4, ("2", "3"))


def test_sglang_bubble_owner_count_covers_routed_writer_owner() -> None:
    trainer = _trainer(
        {"scheduler": {"execution": {"strategy": "rollout_idle_worker"}}},
        step=6,
    )
    trainer._speco_owner_bucket_count = lambda: 1

    owner_count = trainer._speco_sglang_collection_owner_count(
        [{"replica_rank": 0}],
        [1],
    )

    assert owner_count == 2


def test_sglang_sync_owner_count_keeps_replica_derived_count() -> None:
    trainer = _trainer({"scheduler": {"execution": {"strategy": "sync"}}}, step=6)
    trainer._speco_owner_bucket_count = lambda: 4

    owner_count = trainer._speco_sglang_collection_owner_count(
        [{"replica_rank": 0}],
        None,
    )

    assert owner_count == 1


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


def test_dspark_oldlogprob_collect_plan_uses_request_hard_quota() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "max_collect_tokens_per_step_per_replica": 16384,
            "batch_size_per_gpu": 8,
            "dspark_hard_candidate_ratio": 0.20,
            "dspark_hard_sample_ratio": 0.375,
            "hidden_state_window_mode": "random",
        },
        step=20,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 5

    batch_size = 100
    batch = _oldlogprob_plan_batch(
        batch_size,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [
                1.0 + index / 1000.0 for index in range(batch_size)
            ],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
            "_speco_vllm_request_completion_index": list(reversed(range(batch_size))),
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["hard_enabled"] is True
    assert plan["selected_count"] == int(plan["collect_mask"].sum()) == 80
    assert int((plan["request_is_hard"] & plan["collect_mask"]).sum()) == 20
    assert plan["selection_metrics"]["hard_shortfall"] == 10
    assert plan["selection_metrics"]["selected_normal_total"] == 60
    assert [
        int((plan["collect_mask"] & (plan["owner_rank"] == owner)).sum())
        for owner in range(5)
    ] == [16] * 5
    records = trainer._speco_last_request_accept_len_records
    assert records[0]["request_id"] == "req-99"
    assert records[-1]["request_id"] == "req-0"
    assert plan["request_completion_indices"][0] == 99


def test_oldlogprob_collect_plan_accepts_array_like_request_stats() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 8,
            "max_collect_tokens_per_step_per_replica": 8192,
            "batch_size_per_gpu": 4,
            "dspark_hard_candidate_ratio": 0.5,
            "dspark_hard_sample_ratio": 0.5,
            "hidden_state_window_mode": "front",
        },
        step=21,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2

    batch_size = 12
    batch = _oldlogprob_plan_batch(
        batch_size,
        non_tensor_batch={
            "_verl_request_mean_accept_len": _ArrayLike(
                [1.0 + index for index in range(batch_size)]
            ),
            "_speco_vllm_request_id": _ArrayLike(
                [f"req-{index}" for index in range(batch_size)]
            ),
            "_speco_vllm_request_completion_index": _ArrayLike(range(batch_size)),
            "_speco_vllm_request_elapsed_sec": _ArrayLike(
                [0.1 * index for index in range(batch_size)]
            ),
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["selected_count"] == 12
    assert (
        int(plan["request_is_hard"].logical_and(plan["collect_mask"]).sum().item()) == 6
    )
    assert plan["request_is_hard"][:6].all()
    assert not plan["request_is_hard"][6:].any()
    records = trainer._speco_last_request_accept_len_records
    assert len(records) == batch_size
    assert records[0]["request_id"] == "req-0"
    assert records[-1]["request_id"] == "req-11"
    assert trainer._speco_last_request_accept_len_var > 0.0


def test_dspark_oldlogprob_collect_plan_selects_across_dp_shards() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "max_collect_tokens_per_step_per_replica": 16384,
            "batch_size_per_gpu": 4,
            "dspark_hard_candidate_ratio": 0.4,
            "dspark_hard_sample_ratio": 0.5,
            "hidden_state_window_mode": "front",
        },
        step=30,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 5
    batch_size = 100
    window_size = 50
    accept_lens = [
        1.0 + ((index * 7) % window_size) / 100.0 if index < window_size else 0.1
        for index in range(batch_size)
    ]
    batch = _oldlogprob_plan_batch(
        batch_size,
        non_tensor_batch={
            "_verl_request_mean_accept_len": accept_lens,
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
            "_speco_vllm_request_completion_index": list(range(batch_size)),
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["usable_window_size"] == batch_size
    assert plan["selected_count"] == 80
    selected_indices = plan["collect_mask"].nonzero(as_tuple=False).reshape(-1)
    assert int(selected_indices.max()) >= window_size
    assert [
        int(((plan["owner_rank"] == owner) & plan["collect_mask"]).sum())
        for owner in range(5)
    ] == [16] * 5
    assert int((plan["request_is_hard"] & plan["collect_mask"]).sum()) == 40
    assert plan["selection_metrics"]["duplicate_hard_count"] == 0


def test_dspark_oldlogprob_collect_plan_fills_normal_without_duplicates() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "batch_size_per_gpu": 4,
            "dspark_hard_candidate_ratio": 0.1,
            "dspark_hard_sample_ratio": 0.5,
            "hidden_state_window_mode": "front",
        },
        step=31,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2

    batch_size = 40
    batch = _oldlogprob_plan_batch(
        batch_size,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [
                1.0 + index / 1000.0 for index in range(batch_size)
            ],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["selected_count"] == int(plan["collect_mask"].sum()) == 32
    assert plan["selection_metrics"]["hard_shortfall"] == 12
    assert plan["selection_metrics"]["selected_normal_total"] == 28
    assert plan["selection_metrics"]["duplicate_hard_count"] == 0
    assert [
        int((plan["collect_mask"] & (plan["owner_rank"] == owner)).sum())
        for owner in range(2)
    ] == [16, 16]


def test_oldlogprob_collect_plan_hard_disabled_matches_baseline_order() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 3,
            "batch_size_per_gpu": 4,
            "hidden_state_window_mode": "front",
        },
        step=22,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2

    batch_size = 10
    batch = _oldlogprob_plan_batch(
        batch_size,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [
                1.0 + index for index in range(batch_size)
            ],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    # Hard sampling is disabled: selection is the baseline sequential
    # round-robin over the first 2 owners x 3 samples candidates in batch
    # order, and request stats do not influence it.
    assert plan["hard_enabled"] is False
    assert plan["usable_window_size"] == batch_size
    assert plan["selected_count"] == 6
    selected_indices = plan["collect_mask"].nonzero(as_tuple=False).reshape(-1).tolist()
    assert [int(index) for index in selected_indices] == [0, 1, 2, 3, 4, 5]
    assert [int(plan["owner_rank"][index].item()) for index in range(6)] == [
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    assert int(plan["hidden_positions"][0, 0].item()) == 3
    assert not plan["request_is_hard"].any()


def test_oldlogprob_collect_plan_uses_lowest_acceptance_hard_candidates() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 4,
            "dspark_collect_samples_per_step": 8,
            "dspark_hard_candidate_ratio": 0.4,
            "dspark_hard_sample_ratio": 0.5,
            "hidden_state_window_mode": "front",
        },
        step=23,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2
    batch = _oldlogprob_plan_batch(
        20,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [float(index) for index in range(20)],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(20)],
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    hard = set(
        (plan["collect_mask"] & plan["request_is_hard"])
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    assert hard.issubset(set(range(8)))
    assert plan["request_is_hard"].nonzero(as_tuple=False).flatten().tolist() == list(
        range(8)
    )
    assert len(hard) == 4
    assert plan["selected_count"] == 8


def test_oldlogprob_collect_plan_zero_ratio_preserves_baseline_order() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 3,
            "dspark_hard_candidate_ratio": 0.4,
            "dspark_hard_sample_ratio": 0.0,
            "hidden_state_window_mode": "front",
        },
        step=24,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2
    batch = _oldlogprob_plan_batch(
        10,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [float(index) for index in range(10)],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(10)],
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["hard_enabled"] is False
    assert plan["collect_mask"].nonzero(as_tuple=False).flatten().tolist() == list(
        range(6)
    )


@pytest.mark.parametrize("window_size", [0, 33, 999])
def test_oldlogprob_legacy_window_setting_does_not_restrict_collection(
    window_size,
) -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "dspark_hard_candidate_ratio": 0.4,
            "dspark_hard_sample_ratio": 0.1,
        },
        step=5,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 8
    batch = _oldlogprob_plan_batch(
        160,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [float(index + 1) for index in range(160)],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(160)],
        },
    )
    expected = trainer._speco_build_oldlogprob_collect_plan(batch)
    trainer.config.actor_rollout_ref.rollout.drafter.training[
        "dspark_hard_collect_window_size"
    ] = window_size
    actual = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert actual["selected_count"] == expected["selected_count"] == 80
    for key in (
        "collect_mask",
        "owner_rank",
        "hidden_positions",
        "hidden_position_mask",
        "request_is_hard",
    ):
        assert torch.equal(actual[key], expected[key])
    assert actual["selection_metrics"] == expected["selection_metrics"]


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE3", "DFLASH"])
@pytest.mark.parametrize("collection_limit", [None, 16, 80, 0])
def test_oldlogprob_collection_budget_is_scoped_to_dspark(
    algorithm, collection_limit
) -> None:
    training_cfg = {
        "collect_hidden_states_from_old_logprob": True,
        "hidden_state_window_tokens_per_sample": 512,
        "max_collect_samples_per_step_per_replica": 16,
    }
    if collection_limit is not None:
        training_cfg["dspark_collect_samples_per_step"] = collection_limit
    trainer = _trainer(training_cfg, step=5)
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = algorithm
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 8
    batch = _oldlogprob_plan_batch(160)
    if algorithm == "DSPARK" and collection_limit == 0:
        with pytest.raises(
            ValueError, match="dspark_collect_samples_per_step must be positive"
        ):
            trainer._speco_build_oldlogprob_collect_plan(batch)
        return
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    expected = (
        (80 if collection_limit is None else collection_limit)
        if algorithm == "DSPARK"
        else 128
    )
    assert plan["selected_count"] == expected
    assert not plan["hard_enabled"]
    assert not plan["request_is_hard"].any()
    for owner in range(8):
        assert (
            int(((plan["owner_rank"] == owner) & plan["collect_mask"]).sum())
            == expected // 8
        )


@pytest.mark.parametrize("algorithm", ["EAGLE3", "DFLASH"])
def test_explicit_hard_sampling_keeps_collection_budget_for_other_algorithms(
    algorithm,
) -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "dspark_collect_samples_per_step": 80,
            "dspark_hard_candidate_ratio": 0.4,
            "dspark_hard_sample_ratio": 0.1,
        },
        step=5,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = algorithm
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 8
    batch = _oldlogprob_plan_batch(
        160,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [float(index + 1) for index in range(160)],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(160)],
        },
    )
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert plan["selected_count"] == 80
    assert int((plan["collect_mask"] & plan["request_is_hard"]).sum()) == 8


def test_request_accept_len_variance_logs_without_hard_sampling(
    capsys, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH",
        str(tmp_path / "hist.jsonl"),
    )
    trainer = _trainer(
        {
            "request_accept_len_variance_interval_steps": 2,
            "dspark_hard_candidate_ratio": 0.0,
            "dspark_hard_sample_ratio": 0.0,
        },
        step=4,
    )

    batch_size = 4
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 2.0, 3.0, 4.0],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
        },
    )

    logged = trainer._speco_log_request_accept_len_variance_from_batch(
        batch, source="test"
    )
    trainer._speco_flush_request_accept_len_payloads()

    captured = capsys.readouterr()
    assert logged is True
    assert "source=test" in captured.out
    assert "request_accept_len_var=1.250" in captured.out
    assert trainer._speco_last_request_accept_len_var == 1.25


def test_request_accept_len_variance_respects_log_interval(
    capsys, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH",
        str(tmp_path / "hist.jsonl"),
    )
    trainer = _trainer(
        {
            "request_accept_len_variance_interval_steps": 3,
        },
        step=4,
    )

    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 3.0],
            "_speco_vllm_request_id": ["req-0", "req-1"],
        },
    )

    logged = trainer._speco_log_request_accept_len_variance_from_batch(
        batch, source="test"
    )
    trainer._speco_flush_request_accept_len_payloads()

    assert logged is True
    assert "[speco request accept len]" not in capsys.readouterr().out
    assert trainer._speco_last_request_accept_len_var == 1.0


@pytest.mark.parametrize("interval", [0, 3])
def test_request_accept_len_history_is_independent_of_print_interval(
    interval, tmp_path, monkeypatch
) -> None:
    history_path = tmp_path / "hist.jsonl"
    monkeypatch.setenv("VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH", str(history_path))
    trainer = _trainer({"request_accept_len_variance_interval_steps": interval}, step=4)
    batch = _oldlogprob_plan_batch(
        2,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 3.0],
            "_speco_vllm_request_id": ["req-0", "req-1"],
        },
    )
    assert trainer._speco_log_request_accept_len_variance_from_batch(
        batch, source="rollout"
    )
    trainer._speco_flush_request_accept_len_payloads()
    payloads = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(payloads) == 1
    assert payloads[0]["step"] == 4
    assert payloads[0]["source"] == "rollout"
    assert payloads[0]["count"] == 2
    assert payloads[0]["summary"]["var"] == 1.0


@pytest.mark.parametrize("request_ids", [["req-0", None], ["req-0", "req-0"]])
def test_rollout_diagnostic_failure_does_not_skip_collection(
    request_ids, caplog
) -> None:
    trainer = _trainer({"collect_hidden_states_from_old_logprob": True}, step=5)
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 1
    batch = _oldlogprob_plan_batch(
        2,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 3.0],
            "_speco_vllm_request_id": request_ids,
        },
    )
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer.actor_rollout_wg.generate_sequences = lambda *args, **kwargs: batch
    trainer._compute_old_log_prob = lambda value: value
    trainer._update_actor = lambda value: value
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_set_drafter_global_step = lambda **kwargs: None
    trainer._speco_store_rollout_metrics = lambda output: None
    collected_outputs = []
    trainer._speco_collect_generation_samples = (
        lambda output: collected_outputs.append(output) or 0
    )
    with trainer._speco_online_fit_hooks():
        assert trainer.actor_rollout_wg.generate_sequences() is batch
    assert collected_outputs == [batch]
    assert "skipped rollout diagnostics" in caplog.text
    assert trainer._speco_pending_request_accept_len_payloads == []
    with pytest.raises(RuntimeError, match="request id"):
        trainer._speco_build_oldlogprob_collect_plan(batch)


@pytest.mark.parametrize("all_unscored", [False, True])
def test_request_acceptance_diagnostics_skip_unscored_rows(all_unscored) -> None:
    trainer = _trainer({"request_accept_len_variance_interval_steps": 0})
    scores = [None, None, None] if all_unscored else [1.0, None, 3.0]
    request_ids = [None, None, None] if all_unscored else ["req-0", None, "req-2"]
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": scores,
            "_speco_vllm_request_id": request_ids,
        },
    )
    logged = trainer._speco_log_request_accept_len_variance_from_batch(
        batch, source="test"
    )
    assert logged is not all_unscored
    if not all_unscored:
        assert trainer._speco_last_request_accept_len_var == 1.0
        assert [
            record["request_id"]
            for record in trainer._speco_last_request_accept_len_records
        ] == ["req-0", "req-2"]


@pytest.mark.parametrize("request_ids", [["req-0", None], ["req-0", "req-0"]])
def test_request_acceptance_diagnostics_still_reject_invalid_scored_ids(
    request_ids,
) -> None:
    trainer = _trainer({})
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 3.0],
            "_speco_vllm_request_id": request_ids,
        },
    )
    with pytest.raises(RuntimeError, match="request ids?|request id"):
        trainer._speco_log_request_accept_len_variance_from_batch(batch, source="test")


@pytest.mark.parametrize("hard_enabled", [False, True])
def test_oldlogprob_collection_materializes_whole_refs_from_all_dp_shards(
    monkeypatch, hard_enabled
) -> None:
    import ray
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl_speco.integration.oldlogprob_runtime import (
        OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY,
        OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY,
    )

    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "max_collect_tokens_per_step_per_replica": 16384,
            "dspark_hard_candidate_ratio": 0.5 if hard_enabled else 0.0,
            "dspark_hard_sample_ratio": 0.5 if hard_enabled else 0.0,
        }
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 1
    batch = _oldlogprob_plan_batch(
        4,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, None, 1.2, 1.3],
            "_speco_vllm_request_id": ["req-0", None, "req-2", "req-3"],
        },
    )
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert plan["selected_count"] == 4
    stored = {
        "dp-0": torch.arange(2 * 513 * 4, dtype=torch.float32).reshape(2, 513, 4),
        "dp-1": torch.arange(2 * 513 * 4, dtype=torch.float32).reshape(2, 513, 4)
        + 10000,
    }
    outputs = []
    for shard_index in range(2):
        output = TensorDict({"log_probs": torch.zeros(2, 1)}, batch_size=[2])
        tu.assign_non_tensor_stack(
            output, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY, [f"dp-{shard_index}"] * 2
        )
        tu.assign_non_tensor_stack(
            output,
            OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY,
            [{"row_index": row_index} for row_index in range(2)],
        )
        outputs.append(output)
    merged = tu.concat_tensordict(outputs)
    get_calls = []

    def get(refs):
        get_calls.append(refs)
        return [stored[ref] for ref in refs]

    monkeypatch.setattr(ray, "get", get)
    payloads = []
    trainer._speco_dispatch_bucket_count = lambda: 1
    trainer._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        prepare_collection_payload=lambda **kwargs: kwargs
    )

    def execute_collection(collection_plan, payload):
        payloads.append(payload)
        return SimpleNamespace(collected_samples=len(payload["samples"]))

    trainer._speco_execute_collection = execute_collection
    assert trainer._speco_collect_oldlogprob_features(batch, plan, merged) == 4
    assert get_calls == [["dp-0", "dp-1"]]
    hidden = torch.cat([sample["hidden_states"] for sample in payloads[0]["samples"]])
    torch.testing.assert_close(hidden, torch.cat([stored["dp-0"], stored["dp-1"]]))
    assert tu.get(merged, OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY) is None
    assert tu.get(merged, OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY) is None


def _dspark_sampler_trainer(hard_sample_ratio: float):
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.rank = 0
    trainer.training_steps = 0
    trainer.optimizer_steps_total = 0
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "dspark_hard_sample_ratio": hard_sample_ratio,
                }
            )
        )
    )
    return trainer


def test_block_drafter_training_sampler_dspark_enforces_ratio() -> None:
    trainer = _dspark_sampler_trainer(0.375)
    items = [{"_verl_is_hard": True, "id": f"hard-{index}"} for index in range(16)]
    items.extend(
        {"_verl_is_hard": False, "id": f"normal-{index}"} for index in range(64)
    )

    import random as _random

    selected = trainer._sample_training_items(items, 8, _random.Random(1))

    assert len(selected) == 8
    assert sum(item["_verl_is_hard"] for item in selected) == 3


def test_block_drafter_training_sampler_dspark_fractional_quota_accumulates() -> None:
    trainer = _dspark_sampler_trainer(0.1)
    items = [{"_verl_is_hard": True, "id": f"hard-{index}"} for index in range(32)]
    items.extend(
        {"_verl_is_hard": False, "id": f"normal-{index}"} for index in range(48)
    )

    import random as _random

    hard_counts = []
    for optimizer_step in range(5):
        trainer.training_steps = optimizer_step
        trainer.optimizer_steps_total = optimizer_step
        selected = trainer._sample_training_items(
            items, 4, _random.Random(optimizer_step)
        )
        hard_counts.append(sum(item["_verl_is_hard"] for item in selected))

    assert hard_counts == [0, 1, 0, 1, 0]
    assert sum(hard_counts) == 2


@pytest.mark.parametrize("steps_per_trigger", [1, 2])
@pytest.mark.parametrize("resume_step", [0, 1])
def test_dspark_fractional_quota_survives_cleanup_and_resume(
    steps_per_trigger, resume_step
) -> None:
    import asyncio
    import random

    trainer = _dspark_sampler_trainer(0.1)
    trainer.optimizer_steps_total = resume_step
    trainer._pending_checkpoint_future = None
    trainer._pending_full_checkpoint_future = None
    trainer.optimizer = None
    trainer.skip_heavy_cleanup_after_drafter_training = True
    items = [{"_verl_is_hard": index < 32} for index in range(80)]
    hard_count = 0
    for update_index in range(20):
        selected = trainer._sample_training_items(items, 4, random.Random(update_index))
        hard_count += sum(item["_verl_is_hard"] for item in selected)
        trainer.training_steps += 1
        trainer.optimizer_steps_total += 1
        if (update_index + 1) % steps_per_trigger == 0:
            asyncio.run(trainer.cleanup_training(clear_data=False))
            assert trainer.training_steps == 0
    assert hard_count == 8
    assert trainer.optimizer_steps_total == resume_step + 20


def test_block_drafter_training_sampler_dspark_disabled_is_baseline_random() -> None:
    trainer = _dspark_sampler_trainer(0.0)
    items = [{"_verl_is_hard": True, "id": f"hard-{index}"} for index in range(32)]
    items.extend(
        {"_verl_is_hard": False, "id": f"normal-{index}"} for index in range(48)
    )

    import random as _random

    selected = trainer._sample_training_items(items, 8, _random.Random(1))

    assert selected == _random.Random(1).sample(items, 8)


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

    plan = trainer._speco_build_oldlogprob_collect_plan(_oldlogprob_batch([1, 1, 0]))

    assert plan is None
    assert trainer._speco_last_oldlogprob_candidate_samples == 0
    assert trainer._speco_last_oldlogprob_short_response_skipped == 1


def test_oldlogprob_collect_plan_tracks_short_response_skip_metric_state() -> None:
    trainer = _oldlogprob_collect_plan_trainer()

    plan = trainer._speco_build_oldlogprob_collect_plan(_oldlogprob_batch([1, 1, 1]))

    assert plan is not None
    assert plan["response_lens"] == [3]
    assert plan["selected_count"] == 1
    assert trainer._speco_last_oldlogprob_short_response_skipped == 0


def test_oldlogprob_local_batches_have_distinct_retry_stable_sample_ids() -> None:
    from verl_speco.trainer.data_buffer import DataBuffer

    trainer = _oldlogprob_collect_plan_trainer()
    buffer = DataBuffer(max_size=10)
    captured = []

    def execute(plan, payload):
        for bucket in payload.buckets:
            for sample in bucket:
                captured.append(sample["_speco_global_sample_id"])
                buffer.add_batch(sample)
        return SimpleNamespace(collected_samples=1)

    trainer._speco_execute_collection = execute
    output = {_speco_ray_trainer.OLD_LOGPROB_HIDDEN_STATES_KEY: torch.ones(1, 3, 2)}
    plans = []
    batches = []
    # V1 parameter_sync_step > 1 keeps global_steps fixed across local batches.
    for token in (21, 31, 41):
        batch = SimpleNamespace(
            batch={
                "prompts": torch.tensor([[11, 12]]),
                "responses": torch.tensor([[token, token + 1, token + 2]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }
        )
        plan = trainer._speco_build_oldlogprob_collect_plan(batch)
        assert plan is not None
        trainer._speco_collect_oldlogprob_features(batch, plan, output)
        plans.append(plan)
        batches.append(batch)

    assert len(buffer.get_all_data()) == 3
    assert len(set(captured)) == 3
    trainer._speco_collect_oldlogprob_features(batches[0], plans[0], output)
    assert captured[-1] == captured[0]
    assert len(buffer.get_all_data()) == 3


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
        captured.setdefault("payload", payload) or SimpleNamespace(collected_samples=1)
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


@pytest.mark.parametrize(
    "candidate,sample,expected_hard",
    [(0, 0, 0), (0.1, 0.4, 32), (0.4, 0.1, 8), (0.2, 0.3, 24), (0.3, 0.2, 16)],
)
def test_dspark_hard_sample_ratio_matrix(candidate, sample, expected_hard):
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "dspark_collect_samples_per_step": 80,
            "dspark_hard_candidate_ratio": candidate,
            "dspark_hard_sample_ratio": sample,
        },
        step=5,
    )
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 8
    batch = _oldlogprob_plan_batch(
        320,
        non_tensor_batch={
            "_verl_request_mean_accept_len": [
                1 + ((i * 37) % 320) / 320 for i in range(320)
            ],
            "_speco_vllm_request_id": [f"req-{i}" for i in range(320)],
        },
    )
    original = {key: value.clone() for key, value in batch.batch.items()}
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    repeat = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert torch.equal(plan["collect_mask"], repeat["collect_mask"])
    assert torch.equal(plan["owner_rank"], repeat["owner_rank"])
    assert plan["selected_count"] == int(plan["collect_mask"].sum()) == 80
    assert int((plan["collect_mask"] & plan["request_is_hard"]).sum()) == expected_hard
    assert plan["selection_metrics"]["hard_shortfall"] == 0
    for owner in range(8):
        mask = plan["collect_mask"] & (plan["owner_rank"] == owner)
        assert int(mask.sum()) == 10
        assert int((mask & plan["request_is_hard"]).sum()) == expected_hard // 8
    for key, value in original.items():
        assert torch.equal(batch.batch[key], value)
    if candidate:
        assert int(plan["collect_mask"][80:].sum()) > 0
