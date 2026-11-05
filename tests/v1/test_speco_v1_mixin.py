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
import sys
import types

import pytest

from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _config(*, mode="sync", bypass=False, rollout_name="vllm"):
    return AttrDict(
        actor_rollout_ref=AttrDict(
            actor=AttrDict(),
            rollout=AttrDict(
                name=rollout_name,
                temperature=1.0,
                drafter=AttrDict(
                    enable=True,
                    enable_drafter_training=True,
                    training=AttrDict(
                        collect_hidden_states_from_sgl=False,
                        collect_hidden_states_from_old_logprob=True,
                    ),
                ),
            ),
        ),
        algorithm=AttrDict(
            rollout_correction=AttrDict(bypass_mode=bypass),
        ),
        trainer=AttrDict(v1=AttrDict(trainer_mode=mode)),
    )


def test_worker_group_facade_delegates_attach_and_require(monkeypatch):
    attached = []
    worker_group = object()

    class FakeSpecoRayPPOTrainer:
        def attach_speco_worker_group(self, group):
            attached.append((self, group))

        def _require_speco_worker_group(self):
            return worker_group

    fake_module = types.ModuleType("verl_speco.trainer.speco_ray_trainer")
    fake_module.SpecoRayPPOTrainer = FakeSpecoRayPPOTrainer
    monkeypatch.setitem(
        sys.modules, "verl_speco.trainer.speco_ray_trainer", fake_module
    )

    class Harness(SpecoV1Mixin):
        pass

    trainer = Harness()
    trainer.attach_speco_worker_group(worker_group)
    assert attached == [(trainer, worker_group)]
    assert trainer._require_speco_worker_group() is worker_group


def test_update_actor_skips_drafter_execution_when_plan_does_not_launch():
    class Plan:
        launch = False
        reason = "training_interval_not_reached"

    class Event:
        training_plan = Plan()
        metrics = {"drafter/training_plan_launch": 0}

    class Upstream:
        def _update_actor(self, batch, metrics):
            metrics["upstream/updated"] = 1
            return "updated"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_on_before_actor_update(self):
            return Event()

        def _speco_train_drafter(self, plan):
            raise AssertionError("non-launch plan must not execute drafter training")

    metrics = {}
    assert Harness()._update_actor(None, metrics) == "updated"
    assert metrics["drafter/trained"] == 0
    assert metrics["drafter/train_no_trainable_batch"] == 0
    assert metrics["upstream/updated"] == 1


def test_online_drafter_trains_only_on_last_local_update_of_global_step():
    events = []

    class Plan:
        launch = False
        reason = "training_interval_not_reached"

    class Event:
        training_plan = Plan()
        metrics = {}

    class Upstream:
        def _update_actor(self, batch, metrics):
            events.append("actor")
            return "updated"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()
        parameter_sync_step = 3
        local_trigger_step = 0

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_on_before_actor_update(self):
            events.append("drafter_schedule")
            return Event()

    trainer = Harness()
    metrics = {}
    assert trainer._update_actor(None, metrics) == "updated"
    assert events == ["actor"]
    assert metrics["drafter/training_deferred_to_global_step_end"] == 1

    trainer.local_trigger_step = 2
    assert trainer._update_actor(None, {}) == "updated"
    assert events == ["actor", "drafter_schedule", "actor"]


def test_v1_acceptance_row_mismatch_uses_sidecar(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        types.SimpleNamespace(kv_batch_get=lambda **_: {"extra_fields": [{}]}),
    )

    class Harness(SpecoV1Mixin):
        def _speco_v1_spec_decode_sidecar_metrics(self):
            return {"drafter/spec_decode/mean_acceptance_length": 3.0}

    batch = types.SimpleNamespace(keys=["a", "b"], partition_id="train")
    assert Harness()._speco_v1_spec_decode_metrics(batch) == {
        "drafter/spec_decode/mean_acceptance_length": 3.0,
        "drafter/spec_decode/extra_fields_row_mismatch": 1,
    }


def test_v1_metrics_report_speco_mean_acceptance_length(monkeypatch):
    class ExtraFields(list):
        def tolist(self):
            return list(self)

    class ObjectBackedExtraFields:
        def __init__(self, data):
            self.data = data

    fake_tq = types.SimpleNamespace(
        kv_batch_get=lambda **_: {
            "extra_fields": ExtraFields(
                [
                    {
                        "spec_num_draft_tokens": 28,
                        "spec_num_accepted_tokens": 6,
                        "spec_num_verify_steps": 4,
                    },
                    # This is the shape returned by TransferQueue in the
                    # live V1 replay-buffer path, rather than a raw dict.
                    ObjectBackedExtraFields(
                        {
                            "spec_num_draft_tokens": 21,
                            "spec_num_accepted_tokens": 3,
                            "spec_num_verify_steps": 3,
                        }
                    ),
                    {
                        "spec_num_draft_tokens": 99,
                        "spec_num_accepted_tokens": 99,
                        "spec_num_verify_steps": 99,
                    },
                ]
            )
        }
    )
    monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)

    class Upstream:
        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            metrics["upstream/metrics"] = 1

    class Harness(SpecoV1Mixin, Upstream):
        # Isolate metrics from the legacy Ray facade in CPU-only tests.
        _speco_freeze_policy = None

    batch = types.SimpleNamespace(
        keys=["a", "b", "padding"],
        partition_id="train",
        tags=[{}, {}, {"is_padding": True}],
    )
    metrics = {}
    trainer = Harness()
    ci_metrics = []
    trainer._speco_ci_result = types.SimpleNamespace(record_metrics=ci_metrics.append)
    trainer._compute_metrics(batch, metrics, {}, global_steps=1, epoch=0)

    # 1 + (6 + 3) / (4 + 3); the padding record must be ignored.
    assert metrics["drafter/spec_decode/mean_acceptance_length"] == pytest.approx(
        1 + 9 / 7
    )
    assert metrics["upstream/metrics"] == 1
    assert ci_metrics == [{"training/global_step": 1}]


def test_v1_metrics_fall_back_to_engine_core_acceptance_sidecars(monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_SPECO_DRAFTER_CONFIG", raising=False)
    stats_dir = tmp_path / ".spec_decode_stats"
    stats_dir.mkdir()
    (stats_dir / "engine-10.counters").write_text("4 6 28\n", encoding="ascii")
    (stats_dir / "engine-11.counters").write_text("3 3 21\n", encoding="ascii")
    fake_tq = types.SimpleNamespace(
        kv_batch_get=lambda **_: {"extra_fields": [{"global_steps": 0}]}
    )
    monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)

    class Harness(SpecoV1Mixin):
        config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(default_local_dir=str(tmp_path))
        )

    batch = types.SimpleNamespace(keys=["a"], partition_id="train", tags=[{}])
    trainer = Harness()
    assert trainer._speco_v1_spec_decode_metrics(batch)[
        "drafter/spec_decode/mean_acceptance_length"
    ] == pytest.approx(1 + 9 / 7)
    # Cumulative files are differenced, so the same snapshot is not reused by
    # the next global step.
    assert trainer._speco_v1_spec_decode_metrics(batch) == {
        "drafter/spec_decode/requests_with_stats": 0,
        "drafter/spec_decode/requests_missing_stats": 1,
        "drafter/spec_decode/request_stats_coverage": 0,
    }


def test_acceptance_uses_exact_request_counters_with_coverage(monkeypatch):
    cells = [
        {
            "_speco_vllm_request_verify_rounds": 2,
            "_speco_vllm_request_accepted_tokens": 6,
        },
        {
            "spec_num_verify_steps": 4,
            "spec_num_accepted_tokens": 2,
            "_speco_vllm_request_verify_rounds": 99,
            "_speco_vllm_request_accepted_tokens": 99,
        },
        {"spec_num_verify_steps": float("nan"), "spec_num_accepted_tokens": 3},
        {"spec_num_verify_steps": 100, "spec_num_accepted_tokens": 100},
    ]
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        types.SimpleNamespace(kv_batch_get=lambda **kwargs: {"extra_fields": cells}),
    )
    batch = types.SimpleNamespace(
        keys=list(range(4)),
        partition_id="train",
        tags=[{}, {}, {}, {"is_padding": True}],
    )
    metrics = SpecoV1Mixin()._speco_v1_spec_decode_metrics(batch)
    assert metrics["drafter/spec_decode/mean_acceptance_length"] == pytest.approx(
        1 + 8 / 6
    )
    assert metrics["drafter/spec_decode/num_verify_steps"] == 6
    assert metrics["drafter/spec_decode/num_accepted_tokens"] == 8
    assert metrics["drafter/spec_decode/requests_missing_stats"] == 1
    assert metrics["drafter/spec_decode/request_stats_coverage"] == pytest.approx(2 / 3)


def test_online_training_rejects_rollout_correction_bypass():
    with pytest.raises(ValueError, match="bypass_mode=true"):
        SpecoV1Mixin._speco_validate_v1_training_config(_config(bypass=True))


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_online_training_accepts_all_v1_modes(mode):
    SpecoV1Mixin._speco_validate_v1_training_config(_config(mode=mode))


def test_v1_session_owns_shared_runtime_state():
    trainer = SpecoV1Mixin()
    trainer.config = _config(mode="sync")
    trainer.config.actor_rollout_ref.rollout.drafter.training["scheduler"] = {
        "execution": {"strategy": "rollout_idle_worker"},
    }
    trainer._speco_last_collected_step = 3
    session = trainer._speco_bubble_session()
    assert "_speco_last_collected_step" not in trainer.__dict__
    assert trainer._speco_last_collected_step == 3
    trainer._speco_last_collected_step = 4
    assert session.runtime_state["_speco_last_collected_step"] == 4


@pytest.mark.parametrize("bubble", [False, True])
def test_v1_bubble_requires_full_drafter_checkpoint(bubble):
    config = _config()
    training = config.actor_rollout_ref.rollout.drafter.training
    training["save_full_drafter_checkpoint"] = False
    if bubble:
        training["scheduler"] = {"execution": {"strategy": "rollout_idle_worker"}}
        with pytest.raises(ValueError, match="save_full_drafter_checkpoint=true"):
            SpecoV1Mixin._speco_validate_v1_training_config(config)
    else:
        SpecoV1Mixin._speco_validate_v1_training_config(config)


@pytest.mark.parametrize(
    "resume_option",
    ["resume_trainer_state_from_checkpoint", "resume_lr_scheduler_from_checkpoint"],
)
def test_v1_bubble_rejects_disabling_drafter_restore_on_resume(resume_option):
    class Harness(SpecoV1Mixin):
        def _speco_resume_global_step_hint(self):
            return 10

    trainer = Harness()
    trainer.config = _config()
    training = trainer.config.actor_rollout_ref.rollout.drafter.training
    training["scheduler"] = {"execution": {"strategy": "rollout_idle_worker"}}
    training[resume_option] = False

    with pytest.raises(
        ValueError, match="requires resume_trainer_state_from_checkpoint"
    ):
        trainer._setup()


def test_v1_bubble_rejects_removed_head_preparation_override():
    config = _config(mode="sync")
    config.actor_rollout_ref.rollout.drafter.training["scheduler"] = {
        "execution": {"strategy": "rollout_idle_worker"},
        "idle_worker": {"v1_head_preparation_enable": True},
    }
    with pytest.raises(ValueError, match="no longer supported"):
        SpecoV1Mixin._speco_validate_v1_training_config(config)


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("nested", [True, False])
def test_online_training_accepts_bubble_configuration(mode, nested):
    config = _config(mode=mode)
    training = config.actor_rollout_ref.rollout.drafter.training
    if nested:
        training["scheduler"] = {"execution": {"strategy": "rollout_idle_worker"}}
    else:
        training["execution_strategy"] = "rollout_idle_worker"

    trainer = SpecoV1Mixin()
    trainer.config = config
    if mode != "sync":
        with pytest.raises(ValueError, match="supports only trainer_mode=sync"):
            trainer._speco_validate_v1_training_config(config)
        with pytest.raises(ValueError, match="supports only trainer_mode=sync"):
            trainer._speco_rollout_idle_worker_enabled()
    else:
        trainer._speco_validate_v1_training_config(config)
        assert trainer._speco_rollout_idle_worker_enabled()



@pytest.mark.parametrize("mode", ["sync"])
@pytest.mark.parametrize("nested", [False, True])
def test_bubble_rejects_local_policy_head_aliasing_before_worker_creation(mode, nested):
    config = _config(mode=mode)
    config.trainer.v1[mode] = {"parameter_sync_step": 2}
    training = config.actor_rollout_ref.rollout.drafter.training
    if nested:
        training["scheduler"] = {"execution": {"strategy": "rollout_idle_worker"}}
    else:
        training["execution_strategy"] = "rollout_idle_worker"

    class Harness(SpecoV1Mixin):
        def _speco_install_v1_standalone_publish_worker(self):
            pytest.fail("reject head aliasing before creating workers")

    trainer = Harness()
    trainer.config = config
    with pytest.raises(ValueError, match="requires parameter_sync_step=1"):
        trainer._setup()


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
@pytest.mark.parametrize("nested", [False, True])
def test_async_bubble_rejected_before_worker_creation(mode, nested):
    config = _config(mode=mode)
    training = config.actor_rollout_ref.rollout.drafter.training
    if nested:
        training["scheduler"] = {"execution": {"strategy": "rollout_idle_worker"}}
    else:
        training["execution_strategy"] = "rollout_idle_worker"

    class Harness(SpecoV1Mixin):
        def _speco_install_v1_standalone_publish_worker(self):
            pytest.fail("async Bubble must fail before creating workers")

    trainer = Harness()
    trainer.config = config
    with pytest.raises(ValueError, match="supports only trainer_mode=sync"):
        trainer._setup()


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("bubble", [False, True])
def test_local_update_limit_only_applies_to_bubble_without_fixed_old_policy(
    mode, bubble
):
    config = _config(mode=mode)
    config.trainer.v1[mode] = {
        "parameter_sync_step": 1 if bubble and mode != "separate_async" else 4
    }
    if bubble:
        config.actor_rollout_ref.rollout.drafter.training["scheduler"] = {
            "execution": {"strategy": "rollout_idle_worker"}
        }
    if bubble and mode != "sync":
        with pytest.raises(ValueError, match="supports only trainer_mode=sync"):
            SpecoV1Mixin._speco_validate_v1_training_config(config)
    else:
        SpecoV1Mixin._speco_validate_v1_training_config(config)


@pytest.fixture
def old_policy_trainer():
    class Worker:
        weight = 10
        failure = None

        def __init__(self):
            self.snapshots = {}
            self.observed_weights = []
            self.restored_slots = []

        def save_model_to_cpu(self, slot):
            self.snapshots[slot] = self.weight

        def restore_model_from_cpu(self, slot):
            self.restored_slots.append(slot)
            if self.failure == ("restore_old" if slot == 0 else "restore_current"):
                raise RuntimeError(self.failure)
            self.weight = self.snapshots[slot]

        def clear_cpu_model(self, slot):
            del self.snapshots[slot]

        def compute_log_prob(self, control):
            self.observed_weights.append(self.weight)
            if self.failure == "forward":
                raise RuntimeError(self.failure)
            return self.weight

    trainer = SpecoV1Mixin()
    trainer.config = _config(mode="separate_async")
    trainer.actor_rollout_wg = Worker()
    return trainer


@pytest.mark.parametrize("mode", ["sync", "colocate_async"])
def test_collection_forward_does_not_snapshot_other_modes(mode, old_policy_trainer):
    trainer = old_policy_trainer
    trainer.config = _config(mode=mode)
    assert trainer._speco_compute_collection_log_prob(object()) == 10
    assert trainer.actor_rollout_wg.observed_weights == [10]
    assert trainer.actor_rollout_wg.snapshots == {}
    assert trainer.actor_rollout_wg.restored_slots == []


def test_bubble_head_export_is_fenced_before_restoring_current_policy(
    old_policy_trainer,
):
    trainer = old_policy_trainer
    trainer.local_trigger_step = 1
    worker = trainer.actor_rollout_wg
    worker.weight, worker.snapshots = 20, {0: 10}
    heads = []
    trainer._speco_v1_cache_old_policy_head = lambda: heads.append(worker.weight)
    assert trainer._speco_compute_collection_log_prob("batch") == 10
    assert heads == [10]
    assert worker.weight == 20


@pytest.mark.parametrize(
    "collect_steps", [(True, True, True), (True, False, True), (False, True, False)]
)
def test_separate_async_collection_shares_parent_old_policy_snapshots(
    collect_steps, old_policy_trainer
):
    class Parent:
        # Same CPU-slot contract as native PPOTrainerSeparateAsync. Its path
        # must remain compatible when collection is skipped between forwards.
        def _compute_old_log_prob(self, batch, metrics):
            worker = self.actor_rollout_wg
            local_step = self.local_trigger_step
            if local_step == 0:
                worker.save_model_to_cpu(0)
                return worker.compute_log_prob(batch)
            worker.save_model_to_cpu(local_step)
            worker.restore_model_from_cpu(0)
            output = worker.compute_log_prob(batch)
            worker.restore_model_from_cpu(local_step)
            worker.clear_cpu_model(local_step)
            return output

    class Harness(SpecoV1Mixin, Parent):
        def _speco_oldlogprob_collection_enabled(self):
            return False

    trainer = Harness()
    trainer.config = old_policy_trainer.config
    trainer.actor_rollout_wg = worker = old_policy_trainer.actor_rollout_wg
    for local_step, collect in enumerate(collect_steps):
        trainer.local_trigger_step = local_step
        if collect:
            assert trainer._speco_compute_collection_log_prob("batch") == 10
        else:
            assert trainer._compute_old_log_prob("batch", {}) == 10
        assert worker.weight == 10 + local_step
        assert worker.snapshots == {0: 10}
        worker.weight += 1  # Simulate a main-policy optimizer update.
    assert worker.observed_weights == [10, 10, 10]
    # A new sync cycle must replace rather than reuse the previous anchor.
    trainer.local_trigger_step = 0
    assert trainer._speco_compute_collection_log_prob("next-cycle") == 13
    assert worker.snapshots == {0: 13}


@pytest.mark.parametrize("failure", ["forward", "restore_old", "restore_current"])
def test_separate_async_collection_restores_weights_on_failure(
    failure, old_policy_trainer
):
    trainer = old_policy_trainer
    trainer.local_trigger_step = 1
    worker = trainer.actor_rollout_wg
    worker.weight, worker.snapshots, worker.failure = 20, {0: 10}, failure
    with pytest.raises(RuntimeError, match=failure):
        trainer._speco_compute_collection_log_prob("batch")
    assert worker.restored_slots == [0, 1]
    if failure == "restore_current":
        assert worker.snapshots[1] == 20
    else:
        assert worker.weight == 20
        assert worker.snapshots == {0: 10}


def test_old_logprob_collection_forward_runs_under_old_policy(
    monkeypatch, old_policy_trainer
):
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils import padding

    # Exercise the actual collection branch, not only its snapshot helper.
    monkeypatch.setitem(
        sys.modules, "transfer_queue", types.ModuleType("transfer_queue")
    )
    monkeypatch.setattr(padding, "left_right_2_no_padding", lambda control: control)
    monkeypatch.setattr(tu, "assign_non_tensor_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(tu, "assign_non_tensor", lambda *args, **kwargs: None)

    class Batch:
        def __len__(self):
            return 1

        def to_tensordict(self):
            return {"input_ids": torch.tensor([[1, 2]])}

    trainer = old_policy_trainer
    trainer.local_trigger_step = 1
    trainer._speco_oldlogprob_collection_enabled = lambda: True
    trainer._speco_v1_batch_data = lambda batch: (None, Batch())
    trainer._speco_build_oldlogprob_collect_plan = lambda batch: dict.fromkeys(
        ["collect_mask", "hidden_positions", "hidden_position_mask", "owner_rank"]
    )
    trainer._speco_oldlogprob_aux_layer_ids = lambda: []
    trainer._speco_oldlogprob_hidden_capture_impl = lambda: "forward_hook"
    trainer._speco_oldlogprob_hidden_layout = lambda: "BSH"

    def fail_collection(batch, plan, output):
        assert output == 10
        assert trainer.actor_rollout_wg.weight == 20
        raise RuntimeError("feature collection failed")

    trainer._speco_collect_oldlogprob_features = fail_collection
    trainer.actor_rollout_wg.weight = 20
    trainer.actor_rollout_wg.snapshots = {0: 10}
    with pytest.raises(RuntimeError, match="feature collection failed"):
        trainer._compute_old_log_prob(object(), {})
    assert trainer.actor_rollout_wg.weight == 20
    assert trainer.actor_rollout_wg.snapshots == {0: 10}


def test_async_publish_waits_after_upstream_weight_sync():
    events = []

    class Upstream:
        def on_step_end(self):
            events.append("upstream_weight_sync_and_resume")
            return "upstream"

    class Harness(SpecoV1Mixin, Upstream):
        _speco_v1_pending_training = True
        _speco_v1_training_plan = object()

        def _speco_publish_drafter_weights(self, *args, **kwargs):
            events.append(("publish", kwargs["after_weight_update"]))
            return {"drafter/published": 1}

        def _speco_wait_pending_drafter_publish(self):
            events.append("wait_publish")
            return 2

    trainer = Harness()
    assert trainer.on_step_end() == "upstream"
    assert events == [
        "upstream_weight_sync_and_resume",
        ("publish", True),
        "wait_publish",
    ]
    assert (
        trainer._pending_sync_metrics["drafter/publish_waited_after_weight_sync"] == 2
    )
    assert trainer._speco_v1_pending_training is False


@pytest.mark.parametrize("failure", [None, "abort", "publish"])
def test_colocate_probe_boundary_holds_admission_and_releases_on_failure(failure):
    events = []

    class Upstream:
        def on_step_end(self):
            events.append("target_sync_engine_resume")

    class Checkpoints:
        def abort_replicas(self):
            events.append("abort")
            if failure == "abort":
                raise RuntimeError("abort failed")

        def sleep_replicas(self):
            events.append("sleep")

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="colocate_async")
        checkpoint_manager = Checkpoints()

        def _speco_rollout_admission_rpc(self, method):
            events.append(method)

        def _speco_finalize_v1_drafter_publish(self):
            assert self._speco_colocate_admission_closed
            events.append("paired_probe_publish")
            if failure == "publish":
                raise RuntimeError("publish failed")
            return {}

    trainer = Harness()
    trainer.config.actor_rollout_ref.rollout.drafter.training[
        "drafter_convergence_freeze"
    ] = {"enabled": True}
    if failure == "abort":
        with pytest.raises(RuntimeError, match="abort failed"):
            trainer.on_sample_end()
        assert events == [
            "speco_close_rollout_admission",
            "abort",
            "speco_resume_rollout_admission",
        ]
    else:
        trainer.on_sample_end()
        if failure:
            with pytest.raises(RuntimeError, match="publish failed"):
                trainer.on_step_end()
        else:
            trainer.on_step_end()
        assert events == [
            "speco_close_rollout_admission",
            "abort",
            "speco_wait_rollout_idle",
            "sleep",
            "target_sync_engine_resume",
            "paired_probe_publish",
            "speco_resume_rollout_admission",
        ]
    assert trainer._speco_colocate_admission_closed is False


@pytest.mark.parametrize("saved", [False, True])
def test_async_branch_checkpoint_requires_real_tq_snapshot(tmp_path, saved):
    class Parent:
        def _save_checkpoint(self):
            if saved:
                queue = tmp_path / "transfer_queue"
                queue.mkdir()
                (queue / "metadata.json").write_text('{"global_steps":3}')

    class Harness(SpecoV1Mixin, Parent):
        global_steps = 3
        config = _config(mode="colocate_async")
        _speco_colocate_admission_closed = True

        def _speco_online_enabled_from_config(self, config):
            return False

        def _speco_freeze_branch_config(self):
            return {"enabled": True}

        def _speco_v1_checkpoint_directory(self):
            return str(tmp_path)

    trainer = Harness()
    if not saved:
        with pytest.raises(RuntimeError, match="TransferQueue snapshot"):
            trainer._save_checkpoint()
    else:
        trainer._save_checkpoint()
        assert trainer._speco_v1_async_checkpoint_evidence["checkpoint_step"] == 3
        assert trainer._speco_v1_async_checkpoint_evidence[
            "inflight_prompts_reissued_on_resume"
        ]


def test_final_colocate_async_step_does_not_prefetch_an_unconsumed_rollout():
    events = []

    class Upstream:
        def prepare_step(self):
            events.append("upstream_prefetch")
            return {"prefetched": 1}

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="colocate_async")
        global_steps = 3
        total_training_steps = 3

    assert Harness().prepare_step() == {}
    assert events == []


def test_final_sync_step_still_generates_its_current_batch():
    class Upstream:
        def prepare_step(self):
            return {"prefetched": 1}

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="sync")
        global_steps = 3
        total_training_steps = 3

    assert Harness().prepare_step() == {"prefetched": 1}


def test_final_separate_async_step_keeps_its_required_prefetch_and_switch():
    events = []

    class Upstream:
        def prepare_step(self):
            events.append("upstream_prefetch")
            return {"prefetched": 1}

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="separate_async")
        global_steps = 3
        total_training_steps = 3

        def _wait_for_sampleable_and_switch(self):
            events.append("wait_and_switch")
            return {"switch_wait": 1.0}

    assert Harness().prepare_step() == {"prefetched": 1}
    assert events == ["upstream_prefetch"]


def test_nonfinal_async_step_keeps_prefetching():
    class Upstream:
        def prepare_step(self):
            return {"prefetched": 1}

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="separate_async")
        global_steps = 2
        total_training_steps = 3

    assert Harness().prepare_step() == {"prefetched": 1}


def test_fit_drains_agent_loop_before_releasing_drafter_runtime():
    events = []

    class Upstream:
        def fit(self, manager):
            events.append("upstream_fit")
            return "done"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="colocate_async")
        _speco_prepared_for_fit = True

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_v1_drain_agent_loop(self, manager):
            events.append(("drain", manager))
            return 1

        def _speco_wait_pending_drafter_publish(self):
            events.append("publish")

        def _speco_wait_pending_drafter_checkpoint(self):
            events.append("checkpoint")

    manager = object()
    assert Harness().fit(manager) == "done"
    assert events == [
        "upstream_fit",
        ("drain", manager),
        "publish",
        "checkpoint",
    ]


def test_fixed_drafter_async_fit_also_drains_agent_loop():
    events = []
    config = _config(mode="colocate_async")
    config.actor_rollout_ref.rollout.drafter["enable_drafter_training"] = False

    class Upstream:
        def fit(self, manager):
            events.append("upstream_fit")
            return "done"

    class Harness(SpecoV1Mixin, Upstream):
        _speco_prepared_for_fit = True

        def __init__(self):
            self.config = config

        def _speco_v1_drain_agent_loop(self, manager):
            events.append(("drain", manager))

        def _speco_wait_pending_drafter_publish(self):
            events.append("publish")

        def _speco_wait_pending_drafter_checkpoint(self):
            events.append("checkpoint")

    manager = object()
    assert Harness().fit(manager) == "done"
    assert events == ["upstream_fit", ("drain", manager)]


def test_fit_shuts_down_stateful_dataloader_workers_before_ray_teardown():
    events = []

    class Iterator:
        def __init__(self, name):
            self.name = name

        def _shutdown_workers(self):
            events.append(("shutdown", self.name))

    train_iterator = Iterator("train")
    val_iterator = Iterator("val")

    class Upstream:
        def fit(self, manager):
            events.append("upstream_fit")
            return "done"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config(mode="sync")
        train_dataloader_it = train_iterator
        train_dataloader = types.SimpleNamespace(_iterator=train_iterator)
        val_dataloader = types.SimpleNamespace(_iterator=val_iterator)

        def _speco_online_enabled_from_config(self, config):
            return False

    trainer = Harness()
    assert trainer.fit(object()) == "done"
    assert events == ["upstream_fit", ("shutdown", "train"), ("shutdown", "val")]
    assert trainer.train_dataloader_it is None
    assert trainer.train_dataloader._iterator is None
    assert trainer.val_dataloader._iterator is None


def test_async_prefit_warmup_is_sampleable_before_upstream_fit(monkeypatch):
    events = []
    fake_skip_manager = types.SimpleNamespace(
        init=lambda config: events.append("skip_init"),
        set_step=lambda step: events.append(("skip_step", step)),
    )
    fake_skip_module = types.ModuleType("verl.utils.skip")
    fake_skip_module.SkipManager = fake_skip_manager
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.skip",
        fake_skip_module,
    )

    config = _config(mode="colocate_async")
    config.trainer.v1["pre_fit_rollout_warmup"] = True
    config.trainer.v1["colocate_async"] = AttrDict(num_warmup_batches=1)
    config.skip = AttrDict(rollout_tq=AttrDict(enable=False))
    config.data = AttrDict(train_batch_size=60)

    class ReplayBuffer:
        def wait_for_sampleable(self, global_steps, partition_id, target_count):
            events.append(("sampleable", global_steps, partition_id, target_count))
            return set(), {}

    class Upstream:
        def _reissue_inflight_prompts(self):
            events.append(("reissue", self.global_steps))
            return 0

        def on_train_begin(self):
            events.append(("submit_warmup", self.global_steps))

        def fit(self, manager):
            events.append(("upstream_fit", manager))
            # Mirror the two calls made by upstream PPOTrainer.fit(). They must
            # be consumed rather than submitting the same pre-fit batch twice.
            self._reissue_inflight_prompts()
            self.on_train_begin()
            return "done"

    class Harness(SpecoV1Mixin, Upstream):
        def __init__(self):
            self.config = config
            self.global_steps = 0
            self.replay_buffer = ReplayBuffer()

        def _speco_activate_drafter_training_model_before_fit(self):
            events.append("activate")

        def _speco_v1_drain_agent_loop(self, manager):
            events.append(("drain", manager))

        def _speco_wait_pending_drafter_publish(self):
            events.append("publish")

        def _speco_wait_pending_drafter_checkpoint(self):
            events.append("checkpoint")

    manager = object()
    trainer = Harness()
    trainer.prepare_for_fit(manager)

    assert trainer.global_steps == 0
    assert events == [
        "activate",
        "skip_init",
        ("skip_step", 1),
        ("reissue", 1),
        ("submit_warmup", 1),
        ("sampleable", 1, "train", 60),
    ]

    assert trainer.fit(manager) == "done"
    assert events[-4:] == [
        ("upstream_fit", manager),
        ("drain", manager),
        "publish",
        "checkpoint",
    ]


def test_joint_checkpoint_manifest_records_drafter_and_feature_store(tmp_path):
    class Harness(SpecoV1Mixin):
        global_steps = 7
        trainer_mode = "colocate_async"
        _speco_last_published_drafter_step = 5
        config = AttrDict(trainer=AttrDict(default_local_dir=str(tmp_path)))

    path = Harness()._speco_write_v1_joint_checkpoint_manifest(
        drafter_results=[{"saved": True, "path": "draft_step_7"}],
        feature_store=[{"saved": True, "cursor": {"version": 1}}],
    )
    payload = __import__("json").loads(__import__("pathlib").Path(path).read_text())
    assert payload["global_step"] == 7
    assert payload["trainer_mode"] == "colocate_async"
    assert payload["drafter_version"] == 5
    assert payload["drafter_checkpoints"] == [{"saved": True, "path": "draft_step_7"}]
    assert payload["feature_store"] == [{"saved": True, "cursor": {"version": 1}}]


def test_bubble_manifest_restores_writer_and_versions_without_feature_store(tmp_path):
    trainer = SpecoV1Mixin()
    trainer.config = _config(mode="sync")
    trainer.config.trainer.default_local_dir = str(tmp_path)
    trainer.config.actor_rollout_ref.rollout.drafter.training["scheduler"] = {
        "execution": {"strategy": "rollout_idle_worker"}
    }
    trainer.global_steps = 7
    trainer.trainer_mode = "sync"
    trainer._speco_last_published_drafter_step = 5
    trainer._speco_freeze_policy = types.SimpleNamespace(drafter_version=3)
    scheduler = types.SimpleNamespace(
        _training_quota_debt_steps=0,
        _training_quota_collection_step=None,
        _metadata_idle_training_groups=[("2",)],
        _idle_worker_writer_group=("2",),
    )
    scheduler.idle_writer_group = lambda: scheduler._idle_worker_writer_group
    trainer._speco_get_drafter_scheduler = lambda: scheduler
    trainer._speco_bubble_session().published_ids.add("published-once")
    trainer._speco_write_v1_joint_checkpoint_manifest(
        drafter_results=[{"saved": True, "path": str(tmp_path / "draft_step_7")}],
        feature_store=[],
    )
    trainer._speco_bubble_session().published_ids.clear()
    trainer._speco_last_published_drafter_step = 0
    scheduler._idle_worker_writer_group = None
    trainer._speco_resume_global_step_hint = lambda: 7
    trainer._speco_restore_v1_feature_store_checkpoint()
    assert scheduler._idle_worker_writer_group == ("2",)
    assert trainer._speco_last_published_drafter_step == 5
    assert trainer._speco_v1_restored_policy_version == 3
    assert trainer._speco_bubble_session().published_ids == {"published-once"}


@pytest.mark.parametrize("relocated", [False, True])
def test_v1_bubble_resume_rejects_same_step_drafter_from_another_run(tmp_path, relocated):
    source = tmp_path / "source"
    resume_directory = source / "global_step_10"
    resume_directory.mkdir(parents=True)
    expected = source / ("custom-drafts" if relocated else "drafter") / "draft_step_10"
    other = tmp_path / "other" / "drafter" / "draft_step_10"
    for path in (expected, other):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").touch()
        (path / "metadata.json").write_text('{"step": 10}', encoding="utf-8")
    record = {"saved": True, "path": str(expected)}
    if relocated:
        record.update(
            path=str(tmp_path / "old-location" / "custom-drafts" / expected.name),
            run_relative_path=f"custom-drafts/{expected.name}",
        )
    (resume_directory / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 10,
                "drafter_checkpoints": [record],
            }
        ),
        encoding="utf-8",
    )

    class Harness(SpecoV1Mixin):
        def _speco_resume_global_step_hint(self):
            return 10

        def _speco_set_config_value(self, config, key, value):
            config[key] = value

    trainer = Harness()
    trainer.config = _config()
    trainer.config.trainer.update(
        resume_mode="resume_path",
        resume_from_path=str(resume_directory),
        default_local_dir=str(tmp_path / "new-run"),
    )
    drafter = trainer.config.actor_rollout_ref.rollout.drafter
    drafter["model_path"] = str(other)
    with pytest.raises(RuntimeError, match="does not match a saved drafter checkpoint"):
        trainer._speco_validate_v1_drafter_resume_source()
    drafter["model_path"] = str(expected)
    trainer._speco_validate_v1_drafter_resume_source()
    if relocated:
        drafter["model_path"] = record["path"]
        drafter["checkpoint_path"] = str(tmp_path / "old-location" / "custom-drafts")
        trainer._speco_validate_v1_drafter_resume_source()
        assert drafter["model_path"] == str(expected)
        assert drafter["checkpoint_path"] == str(tmp_path / "new-run" / "custom-drafts")
        drafter["checkpoint_path"] = str(tmp_path / "explicit-destination")
        trainer._speco_validate_v1_drafter_resume_source()
        assert drafter["checkpoint_path"] == str(tmp_path / "explicit-destination")


@pytest.mark.parametrize("writer_index", [1, 3])
@pytest.mark.parametrize("explicit_resume", [False, True])
def test_feature_store_resume_deduplicates_shared_worker_cursors(
    tmp_path, writer_index, explicit_resume
):
    cursor = {
        "format": "torch_shard_feature_store_cursor",
        "version": 1,
        "path": str(tmp_path / "feature-store"),
        "num_samples": 8,
        "num_shards": 2,
        "next_shard_index": 1,
    }

    class Harness(SpecoV1Mixin):
        config = AttrDict(
            actor_rollout_ref=AttrDict(
                rollout=AttrDict(
                    drafter=AttrDict(
                        training=AttrDict(feature_store=AttrDict(path=cursor["path"]))
                    )
                )
            ),
            trainer=AttrDict(default_local_dir=str(tmp_path)),
        )

        def _speco_resume_global_step_hint(self):
            return 1

        def _ray_get_if_needed(self, value):
            return value

        def speco_restore_feature_store_checkpoint_state(self, state):
            self.restored_cursor = state["cursor"]
            return [{"restored": True}]

    manifest_path = tmp_path / "global_step_1" / "speco_v1_manifest.json"
    manifest_path.parent.mkdir()
    records = [
        {"saved": True, "cursor": cursor},
        {"saved": True, "cursor": {**cursor, "next_shard_index": writer_index}},
    ]
    if explicit_resume:
        records = [
            {
                **record,
                "run_relative_path": "feature-store",
                "cursor": {
                    **record["cursor"],
                    "path": str(tmp_path / "old-location" / "feature-store"),
                },
            }
            for record in records
        ]
    manifest_path.write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 1,
                "feature_store": records,
            }
        ),
        encoding="utf-8",
    )

    harness = Harness()
    if explicit_resume:
        output = tmp_path / "new-run"
        decoy = output / "global_step_1" / manifest_path.name
        decoy.parent.mkdir(parents=True)
        decoy.write_text("{}", encoding="utf-8")
        harness.config.trainer.update(
            default_local_dir=str(output),
            resume_mode="resume_path",
            resume_from_path=str(manifest_path.parent),
        )
    harness._speco_restore_v1_feature_store_checkpoint()
    assert harness.restored_cursor == {**cursor, "next_shard_index": writer_index}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("submission_failure", [False, True])
def test_separate_async_publish_failure_rolls_back_standalone_workers(
    monkeypatch, asynchronous, submission_failure
):
    pytest.importorskip("ray")
    from verl.single_controller.ray import base

    calls = []
    workers = [object(), object()]

    class WorkerGroup:
        def __init__(self, *, worker_handles, ray_cls_with_init):
            assert worker_handles == workers

        def update_draft_weights(self, payload, global_steps):
            calls.append(("sync", payload, global_steps))
            if submission_failure and payload == "new":
                raise RuntimeError("replica failure")
            return payload

        def update_draft_weights_async(self, payload, global_steps):
            calls.append(("async", payload, global_steps))
            if submission_failure:
                raise RuntimeError("replica failure")
            return payload

    monkeypatch.setattr(base, "RayWorkerGroup", WorkerGroup)
    monkeypatch.setattr(base, "RayClassWithInitArgs", lambda **kwargs: kwargs)

    class Harness(SpecoV1Mixin):
        config = _config(mode="separate_async")
        standalone_server_manager = types.SimpleNamespace(
            get_replicas=lambda: [types.SimpleNamespace(workers=workers)]
        )
        _speco_last_published_drafter_payload = "old"
        _speco_last_published_drafter_step = 7

        @staticmethod
        def _speco_v1_standalone_publish_worker_cls():
            return object()

        def _speco_actor_rollout_method(self, name):
            pytest.fail("separate_async must not publish to the hybrid actor worker")

        def _ray_get_if_needed(self, value):
            if value == "new":
                raise RuntimeError("replica failure")
            return value

    trainer = Harness()
    with pytest.raises(RuntimeError, match="replica failure"):
        trainer._speco_update_rollout_drafter_weights("new", 8, asynchronous)
        if asynchronous:
            trainer._speco_wait_pending_drafter_publish_rpc()

    assert calls == [
        ("async" if asynchronous else "sync", "new", 8),
        ("sync", "old", 7),
    ]
    assert trainer._speco_last_published_drafter_payload == "old"
    assert trainer._speco_last_published_drafter_step == 7


def test_feature_store_resume_rejects_conflicting_worker_cursors(tmp_path):
    cursor = {"version": 1, "path": str(tmp_path / "feature-store")}

    class Harness(SpecoV1Mixin):
        config = AttrDict(
            actor_rollout_ref=AttrDict(
                rollout=AttrDict(
                    drafter=AttrDict(
                        training=AttrDict(feature_store=AttrDict(path=cursor["path"]))
                    )
                )
            ),
            trainer=AttrDict(default_local_dir=str(tmp_path)),
        )

        def _speco_resume_global_step_hint(self):
            return 1

    manifest_path = tmp_path / "global_step_1" / "speco_v1_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 1,
                "feature_store": [
                    {"saved": True, "cursor": cursor},
                    {"saved": True, "cursor": {"version": 2}},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="one unique Feature Store cursor"):
        Harness()._speco_restore_v1_feature_store_checkpoint()


def test_collect_only_resume_skips_drafter_checkpoint_resolution():
    pytest.importorskip("ray")
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    class Harness:
        def _speco_drafter_config(self):
            return {"model_path": "/base-model", "checkpoint_path": "/checkpoint"}

        def _speco_drafter_training_mode(self):
            return "collect_only"

    SpecoRayPPOTrainer._speco_prepare_drafter_checkpoint_for_worker_init(Harness())


@pytest.mark.parametrize("bubble", [False, True])
def test_setup_resolves_drafter_before_upstream_and_creates_worker_after(
    monkeypatch, bubble
):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append(
        "configure"
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    class Upstream:
        def _setup(self):
            events.append("upstream")
            return "ready"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()

        def _speco_install_v1_standalone_publish_worker(self):
            events.append("install_standalone_publish_worker")

        def _speco_validate_v1_training_config(self, config):
            events.append("validate")

        def _speco_init_state(self):
            events.append("state")

        def _speco_configure_rollout_idle_event_bus(self):
            events.append("bus")

        def _speco_prepare_drafter_checkpoint_for_worker_init(self):
            events.append("resume")

        def _init_v1_speco_drafter_workers(self):
            events.append("workers")

    trainer = Harness()
    if bubble:
        trainer.config = _config()
        trainer.config.actor_rollout_ref.rollout.drafter.training["scheduler"] = {
            "execution": {"strategy": "rollout_idle_worker"}
        }
    assert trainer._setup() == "ready"
    assert events == [
        "validate",
        "install_standalone_publish_worker",
        "state",
        *(["bus"] if bubble else []),
        "resume",
        "configure",
        "upstream",
        "workers",
    ]


def test_init_creates_drafter_before_first_rollout_weight_update(monkeypatch):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append(
        "configure"
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    class Upstream:
        def init(self):
            self._setup()
            self.on_init_end()

        def _setup(self):
            events.append("upstream")

        def on_init_end(self):
            events.append("update_weights")

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()
        trainer_mode = "sync"

        def _speco_validate_v1_training_config(self, config):
            events.append("validate")

        def _speco_init_state(self):
            events.append("state")

        def _speco_prepare_drafter_checkpoint_for_worker_init(self):
            events.append("resume")

        def _init_v1_speco_drafter_workers(self):
            events.append("workers")

    Harness().init()
    assert events.index("resume") < events.index("upstream")
    assert events.index("workers") < events.index("update_weights")


def test_checkpoint_result_flattening_is_available_on_v1_trainer() -> None:
    nested = [{"saved": True}, ({"saved": False}, [{"saved": True}])]

    assert SpecoV1Mixin._speco_flatten_checkpoint_results(nested) == [
        {"saved": True},
        {"saved": False},
        {"saved": True},
    ]


def test_v1_adaptive_budget_uses_deduplicated_local_rollout_counts(monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock

    from verl_speco.trainer.scheduler import (
        AcceptanceFeedback, AdaptiveScheduleConfig, DrafterScheduleConfig,
        DrafterScheduleContext, DrafterScheduler,
    )

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from test_bubble_review_regressions import _methods

    drafts_key = "_speco_vllm_spec_decode_drafts"
    accepted_key = "_speco_vllm_spec_decode_accepted_tokens"
    facade = _methods(
        "verl_speco/trainer/speco_ray_trainer.py", "SpecoRayPPOTrainer",
        {"_speco_acceptance_feedback"},
        AcceptanceFeedback=AcceptanceFeedback,
        _SPECO_VLLM_INVALID_FEEDBACK_KEY="_speco_vllm_invalid_acceptance_feedback",
        _SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY=drafts_key,
        _SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY=accepted_key,
        SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC="length",
        _speco_metric_float=lambda value: float(value) if value is not None else None,
        _speco_rollout_metrics_from_stats=lambda stats: {
            "length": 1 + stats[accepted_key] / stats[drafts_key]
        },
    )
    records = {}
    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(
        kv_batch_get=lambda keys, **kwargs: {"extra_fields": [records[k] for k in keys]}
    ))
    cfg = DrafterScheduleConfig(
        training_interval_steps=1,
        adaptive_schedule=AdaptiveScheduleConfig(
            enable=True, warmup_max_steps=0,
            ema_fast_alpha=1.0, ema_slow_alpha=0.01,
        ),
    )
    scheduler = DrafterScheduler(worker_executor=Mock(
        get_training_data_status=Mock(return_value=[])
    ))
    controller = scheduler.configure_adaptive(cfg)
    controller.state.budget = 5
    observed = []

    class Upstream:
        def _update_actor(self, batch, metrics):
            return None

    class Harness(SpecoV1Mixin, facade, Upstream):
        _speco_freeze_policy = None
        config = _config()
        parameter_sync_step = 3

        def _speco_on_before_actor_update(self):
            feedback = self._speco_acceptance_feedback()
            observed.append(feedback)
            scheduler.prepare_training_plan(DrafterScheduleContext(
                global_step=self.global_steps, training_mode="online",
                collected_samples_this_step=1, oldlogprob_collection_requested=True,
                acceptance_feedback=feedback,
            ), cfg)
            return types.SimpleNamespace(training_plan=None, metrics={})

    trainer = Harness()
    trainer.config.actor_rollout_ref.rollout.drafter.training["adaptive_schedule"] = {"enable": True}
    for step, accepted in [(1, 140), (2, 0)]:
        trainer.global_steps = step
        for key in ("a", "b", "c", "padding"):
            records[key] = {"spec_num_verify_steps": 20,
                            "spec_num_accepted_tokens": accepted,
                            "spec_num_draft_tokens": 140}
        for local, keys in enumerate([
            ["a", "b"], ["b", "c"], ["a", "padding"],
        ]):
            trainer.local_trigger_step = local
            batch = types.SimpleNamespace(
                keys=keys, partition_id="train",
                tags=[{"is_padding": k == "padding"} for k in keys],
            )
            trainer._update_actor(batch, {})
            trainer._speco_v1_spec_decode_metrics(batch)
        assert observed[-1].sample_count == 60
        assert observed[-1].mean_acceptance_length == 1 + accepted / 20
    assert len(observed) == 2
    assert controller.state.budget > 5


def test_v1_checkpoint_saves_and_resumes_adaptive_state(monkeypatch, tmp_path):
    import os
    from dataclasses import asdict
    from pathlib import Path

    from verl_speco.integration.vllm_runtime import _get_nested
    from verl_speco.trainer.scheduler import AdaptiveScheduleConfig, DrafterScheduleConfig, DrafterScheduler
    from verl_speco.trainer.scheduler.adaptive_schedule import AdaptiveScheduleState

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from test_bubble_review_regressions import _methods

    facade = _methods(
        "verl_speco/trainer/speco_ray_trainer.py", "SpecoRayPPOTrainer",
        {"_speco_save_adaptive_state", "_speco_get_drafter_scheduler",
         "_speco_restore_adaptive_state",
         "_speco_freeze_state_resume_path"},
        Path=Path, os=os, _get_nested=_get_nested, DrafterScheduler=DrafterScheduler,
    )
    cfg = DrafterScheduleConfig(adaptive_schedule=AdaptiveScheduleConfig(enable=True))
    checkpoint = tmp_path / "global_step_10"

    class Parent:
        def _save_checkpoint(self):
            checkpoint.mkdir()
            return "saved"

    class Harness(SpecoV1Mixin, facade, Parent):
        global_steps = 10
        _speco_last_checkpoint_saved_step = None
        _speco_colocate_admission_closed = False
        _speco_freeze_state_sidecar = "freeze_policy_state.json"

        def _speco_drafter_schedule_config(self):
            return cfg

        def _speco_wait_pending_drafter_publish(self):
            pass

        def _speco_save_drafter_checkpoint(self, **kwargs):
            return []

        def _speco_freeze_save_state(self):
            pass

        def _speco_snapshot_v1_feature_store_checkpoint(self):
            return None

        def _speco_write_v1_joint_checkpoint_manifest(self, **kwargs):
            pass

    trainer = Harness()
    trainer.config = _config()
    trainer.config.trainer.update(default_local_dir=str(tmp_path), resume_mode="disable")
    trainer.config.actor_rollout_ref.rollout.drafter.training["adaptive_schedule"] = {"enable": True}
    trainer._speco_init_state()
    controller = trainer._speco_get_drafter_scheduler().adaptive_controller
    controller.state = AdaptiveScheduleState(
        budget=0, fast=2.0, slow=2.5, observations=3,
        last_valid_feedback_step=10, interval_trend_history=[[10, 2.0]],
        warmup_acceptance_history=[2.0, 2.0], warmup_ended_after=5,
    )
    expected = asdict(controller.state)
    assert trainer._save_checkpoint() == "saved"
    assert (checkpoint / "adaptive_schedule.json").is_file()

    resumed = Harness()
    resumed.config = _config()
    resumed.config.trainer.update(
        default_local_dir=str(tmp_path), resume_mode="resume_path",
        resume_from_path=str(checkpoint),
    )
    resumed.config.actor_rollout_ref.rollout.drafter.training["adaptive_schedule"] = {"enable": True}
    resumed._speco_init_state()
    restored = resumed._speco_get_drafter_scheduler().adaptive_controller
    assert asdict(restored.state) == expected
    assert restored.training_steps(11) == 0


def test_v1_emits_trainer_phase_spans_for_prepare_train_and_publish(monkeypatch):
    from contextlib import nullcontext

    import verl_speco.trainer.v1.speco_mixin as mixin

    spans = []

    def fake_trace(state_name, *, state_lane_id=None, **labels):
        spans.append((state_name, state_lane_id, labels))
        return nullcontext()

    monkeypatch.setattr(mixin, "_rl_insight_trace_state", fake_trace)

    class Plan:
        launch = True
        reason = "training_ready"
        execution_strategy = "sync"
        max_batches = 4

    class Event:
        training_plan = Plan()
        metrics = {}

    class Upstream:
        def _update_actor(self, batch, metrics):
            return "updated"

        def on_step_end(self):
            return "upstream"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()
        global_steps = 12
        _speco_freeze_policy = None
        _speco_v1_pending_training = False

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_on_before_actor_update(self):
            return Event()

        def _speco_train_drafter(self, plan):
            self._speco_v1_pending_training = True
            return True, {"drafter/trained": 1}

        def _speco_publish_drafter_weights(self, *args, **kwargs):
            return {"drafter/published": 1}

        def _speco_wait_pending_drafter_publish(self):
            return 0

        def _speco_flush_request_accept_len_payloads(self):
            pass

    trainer = Harness()
    assert trainer._update_actor(None, {}) == "updated"
    assert [span[0] for span in spans] == ["drafter/prepare", "drafter/train"]
    assert {span[1] for span in spans} == {"speco_trainer"}
    assert spans[1][2]["strategy"] == "sync"
    assert spans[1][2]["max_batches"] == 4

    spans.clear()
    assert trainer.on_step_end() == "upstream"
    assert [span[0] for span in spans] == ["drafter/publish"]
    assert spans[0][1] == "speco_trainer"
    assert spans[0][2]["deferred"] is True
