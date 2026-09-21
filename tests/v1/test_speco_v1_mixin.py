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
        pass

    batch = types.SimpleNamespace(
        keys=["a", "b", "padding"],
        partition_id="train",
        tags=[{}, {}, {"is_padding": True}],
    )
    metrics = {}
    Harness()._compute_metrics(batch, metrics, {}, global_steps=1, epoch=0)

    # 1 + (6 + 3) / (4 + 3); the padding record must be ignored.
    assert metrics["drafter/spec_decode/mean_acceptance_length"] == pytest.approx(
        1 + 9 / 7
    )
    assert metrics["upstream/metrics"] == 1


def test_v1_metrics_fall_back_to_engine_core_acceptance_sidecars(
    monkeypatch, tmp_path
):
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
    assert trainer._speco_v1_spec_decode_metrics(batch) == {}


def test_online_training_rejects_rollout_correction_bypass():
    with pytest.raises(ValueError, match="bypass_mode=true"):
        SpecoV1Mixin._speco_validate_v1_training_config(_config(bypass=True))


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_online_training_accepts_all_v1_modes(mode):
    SpecoV1Mixin._speco_validate_v1_training_config(_config(mode=mode))


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
    assert trainer._pending_sync_metrics["drafter/publish_waited_after_weight_sync"] == 2
    assert trainer._speco_v1_pending_training is False


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


def test_feature_store_resume_deduplicates_identical_worker_cursors(tmp_path):
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
    manifest_path.write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 1,
                "feature_store": [
                    {"saved": True, "cursor": cursor},
                    {"saved": True, "cursor": dict(cursor)},
                ],
            }
        ),
        encoding="utf-8",
    )

    harness = Harness()
    harness._speco_restore_v1_feature_store_checkpoint()
    assert harness.restored_cursor == cursor


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


def test_setup_resolves_drafter_before_upstream_and_creates_worker_after(monkeypatch):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append("configure")
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

        def _speco_prepare_drafter_checkpoint_for_worker_init(self):
            events.append("resume")

        def _init_v1_speco_drafter_workers(self):
            events.append("workers")

    assert Harness()._setup() == "ready"
    assert events == [
        "install_standalone_publish_worker",
        "validate",
        "state",
        "resume",
        "configure",
        "upstream",
        "workers",
    ]


def test_init_creates_drafter_before_first_rollout_weight_update(monkeypatch):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append("configure")
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
