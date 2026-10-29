# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Regressions against real V1 hooks and native checkpoint ordering."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle
from verl_speco.trainer.scheduler import DrafterScheduler
from verl_speco.trainer.v1.factory import _trainer_types

try:
    TRAINERS = _trainer_types()
except (ImportError, RuntimeError):
    if os.getenv("SPECO_REQUIRE_V1") == "1":
        raise
    pytest.skip("requires upstream verl 0.9 and TransferQueue", allow_module_level=True)


def config(mode="sync"):
    return OmegaConf.create(
        {
            "trainer": {"device": "cpu", "v1": {"trainer_mode": mode}},
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": False}},
                "rollout": {
                    "data_parallel_size": 1,
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "scheduler": {
                                "execution": {"strategy": "rollout_idle_worker"}
                            }
                        },
                    },
                },
            },
        }
    )


def trainer_for(mode):
    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = 7
    trainer.trainer_mode = mode
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._speco_bubble_lifecycle = BubbleLifecycle(trainer)
    return trainer


@pytest.mark.parametrize("mode", ["sync", "colocate_async"])
@pytest.mark.parametrize("requested", [None, False, True])
def test_step_end_never_prepares_head_even_with_legacy_flag(
    monkeypatch, mode, requested
):
    trainer = trainer_for(mode)
    trainer._speco_init_state()
    trainer.global_steps = 11
    trainer.total_training_steps = 50
    idle = {"keep_hot_between_plans": True}
    if requested is not None:
        idle["v1_head_preparation_enable"] = requested
    trainer.config.actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker = (
        idle
    )
    scheduler = trainer._drafter_scheduler
    group = ("2", "3")
    scheduler._idle_worker_writer_group = group
    scheduler._idle_worker_hot_prewarmed_groups.add(group)
    scheduler._training_quota_debt_steps = 20
    scheduler._training_quota_data_version = 8
    trainer._speco_ready_target_lm_head_versions.add(8)
    trainer._speco_ready_target_lm_head_workers[8] = group
    session = trainer._speco_bubble_session()
    events = []
    session.close = lambda boundary: events.append("close") or {}
    trainer._speco_wait_target_lm_head_prefetch = lambda: events.append("ack")
    trainer._speco_prepare_hot_writer_head_at_safe_point = lambda: pytest.fail(
        "V1 must prepare heads only at training preflight"
    )
    native = TRAINERS[mode].__bases__[1]
    monkeypatch.setattr(
        native, "on_step_end", lambda self: events.append("target_resume")
    )
    trainer.on_step_end()
    assert events == ["close", "ack", "target_resume"]
    assert group not in scheduler._idle_worker_last_target_version
    assert scheduler._training_quota_debt_steps == 20
    assert scheduler._training_quota_data_version == 8
    assert not session.costs.events


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("git_available", [False, True])
def test_runtime_provenance_survives_logger_filters(
    monkeypatch, capsys, mode, git_available
):
    import hashlib
    import subprocess
    from verl_speco.trainer.v1 import bubble_mixin

    trainer = trainer_for(mode)
    trainer.config.actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker = {
        "v1_head_preparation_enable": True,
        "boundary_diagnostics_enable": True,
    }

    def revision(*args, **kwargs):
        assert kwargs["timeout"] == 2
        if not git_available:
            raise FileNotFoundError("git unavailable")
        return SimpleNamespace(returncode=0, stdout="revision\n")

    monkeypatch.setattr(subprocess, "run", revision)
    monkeypatch.setattr(bubble_mixin.logger, "disabled", True)
    trainer._speco_v1_log_runtime_provenance()
    record = capsys.readouterr().out
    report = json.loads(record.split("v1_runtime: ", 1)[1])
    assert report["commit"] == ("revision" if git_available else "unavailable")
    assert report["mode"] == mode
    assert report["head_preparation_requested"] is True
    assert report["head_preparation_effective"] is False
    assert report["boundary_diagnostics_enable"] is True
    assert len(report["source_sha256"]) == 3
    for filename, digest in report["source_sha256"].items():
        path = Path(filename)
        assert path.is_absolute()
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_runtime_provenance_precedes_worker_setup(mode):
    trainer = trainer_for(mode)
    events = []
    trainer._speco_validate_v1_training_config = lambda config: None
    trainer._speco_v1_log_runtime_provenance = lambda: events.append("provenance")

    def install():
        events.append("worker_setup")
        raise RuntimeError("stop before worker creation")

    trainer._speco_install_v1_standalone_publish_worker = install
    with pytest.raises(RuntimeError, match="stop before worker creation"):
        trainer._setup()
    assert events == ["provenance", "worker_setup"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_boundary_diagnostics_preserve_actor_order_and_failure(caplog, enabled, fails):
    trainer = trainer_for("sync")
    trainer.config.actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker = {
        "boundary_diagnostics_enable": enabled
    }
    trainer._pending_target_lm_head_sync = None
    trainer._speco_rollout_idle_generation_open = True
    event_thread = object()
    trainer._speco_bubble_session().thread = event_thread

    def stop(*args):
        assert trainer._speco_rollout_idle_generation_open is False
        assert args[1] is event_thread

    trainer._speco_stop_rollout_idle_event_loop = stop
    trainer._speco_service_rollout_idle_events = lambda **kwargs: {}
    trainer._speco_emit_rollout_generation_completed = lambda *args: {}
    events = []
    trainer._speco_reclaim_rollout_idle_workers = (
        lambda **kwargs: events.append("reclaim") or {}
    )
    trainer._speco_flush_idle_completed_publications = lambda *args: {}
    trainer._speco_park_drafter_training_memory = lambda *args: {}
    trainer._speco_pop_rollout_idle_metrics = lambda: {}

    def update(batch, metrics):
        events.append("actor")
        if fails:
            raise RuntimeError("actor failure")
        return "result"

    if fails:
        with pytest.raises(RuntimeError, match="actor failure"):
            trainer._speco_v1_bubble_update_actor(None, {}, update)
    else:
        assert trainer._speco_v1_bubble_update_actor(None, {}, update) == "result"
    assert events == ["reclaim", "actor"]
    records = [r.message for r in caplog.records if "v1_boundary:" in r.message]
    expected = ["reclaim_begin", "reclaim_complete", "actor_begin"] + [
        "actor_failed" if fails else "actor_end"
    ]
    assert len(records) == (4 if enabled else 0)
    if enabled:
        assert all(
            f"event={event} " in record for event, record in zip(expected, records)
        )
        assert "event_thread_active=False generation_open=False" in records[0]
        assert "event_thread_active=False generation_open=False" in records[1]


def test_final_metrics_override_stale_event_quota_snapshot(monkeypatch):
    trainer = trainer_for("sync")
    trainer.global_steps = 42
    scheduler = trainer._drafter_scheduler
    scheduler._idle_optimizer_steps_total = 27
    scheduler._critical_path_topup_optimizer_steps_total = 158
    scheduler._training_quota_debt_steps = 15
    scheduler._training_quota_data_version = 40
    scheduler._training_quota_collection_step = 40
    scheduler._training_quota_oldest_cycle_step = 40
    trainer._speco_v1_spec_decode_metrics = lambda batch: {}
    native = TRAINERS["sync"].__bases__[1]
    monkeypatch.setattr(native, "_compute_metrics", lambda *args, **kwargs: None)
    metrics = {
        "bubble/training_quota_debt_steps": 19,
        "bubble/idle_optimizer_steps_total": 23,
    }
    trainer._compute_metrics(None, metrics, {}, global_steps=42, epoch=0)
    assert metrics["bubble/training_quota_debt_steps"] == 15
    assert metrics["bubble/idle_optimizer_steps_total"] == 27
    assert metrics["bubble/training_quota_debt_age_steps"] == 2
    assert metrics["bubble/training_quota_data_version"] == 40


@pytest.mark.parametrize("mode", ["sync", "colocate_async"])
def test_sample_end_keeps_training_outcome_from_forced_drain(monkeypatch, mode):
    trainer = trainer_for(mode)
    result = {
        "drafter/trained": 1,
        "drafter/train_successful_steps_max": 4,
        "bubble/idle_optimizer_steps": 4,
        "timing_s/drafter_async_training_work": 1.2,
        "bubble/training_quota_debt_steps": 15,
    }
    trainer._speco_bubble_session().close = lambda boundary: dict(result)
    trainer._speco_v1_admission = lambda method: None
    native = TRAINERS[mode].__bases__[1]
    monkeypatch.setattr(native, "on_sample_end", lambda self: None)
    trainer.on_sample_end()
    metrics = trainer._speco_pop_rollout_idle_metrics()
    for key, value in result.items():
        assert metrics[key] == value
    assert trainer._speco_pop_rollout_idle_metrics() == {}


@pytest.mark.parametrize("rl_step,quota_version", [(5, 4), (11, 8)])
def test_sync_skipped_collection_advances_control_step_without_replacing_quota_head(
    monkeypatch, rl_step, quota_version
):
    import asyncio
    from verl.trainer.ppo.v1 import PPOTrainerSync
    from verl_speco.workers.speco_worker import SpecoWorker

    worker = SpecoWorker.__new__(SpecoWorker)
    worker._rank = 2
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.worker_incarnation = "worker-2"
    worker.last_global_step = quota_version
    worker.device_name = "cpu"
    selected = []

    def select_head(version):
        selected.append(version)
        worker.trainer._target_lm_head_weight_step = version
        return True

    async def activate():
        return True

    worker.trainer = SimpleNamespace(
        buffer_version=3,
        _bubble_time_enabled=True,
        _target_lm_head_weight_step=rl_step,
        select_target_lm_head_version=select_head,
        get_training_data_status=lambda **kwargs: {
            "data_version": quota_version,
            "trainable_batches": 20,
        },
        activate_training_model=activate,
        clear_pending_publish_state_dict=lambda: None,
        increment_rl_step=lambda step: None,
    )
    payload = {
        "launch": True,
        "execution_strategy": "sync",
        "source_global_step": rl_step,
        "plan_id": "quota-topup",
        "data_version": quota_version,
        "required_target_version": quota_version,
        "max_batches": 20,
        "worker_snapshots": {
            "2": {
                "worker_incarnation": "worker-2",
                "buffer_version": 3,
                "data_version": quota_version,
            }
        },
    }
    stale = asyncio.run(worker.preflight_drafter_training(payload))
    assert stale["reason"] == "stale_training_plan"
    assert stale["worker_global_step"] == quota_version
    assert not selected

    trainer = trainer_for("sync")
    trainer.global_steps = rl_step
    trainer._require_speco_worker_group = lambda: SimpleNamespace(
        set_global_step=worker.set_global_step
    )
    events = []
    trainer._speco_bubble_lifecycle = SimpleNamespace(
        start=lambda: events.append(worker.last_global_step)
    )
    monkeypatch.setattr(
        PPOTrainerSync,
        "step",
        lambda *args: asyncio.run(worker.preflight_drafter_training(payload)),
    )
    ready = trainer.step({}, {})
    assert events == [rl_step]
    assert ready["ready"]
    assert ready["data_version"] == ready["target_version"] == quota_version
    assert selected == [quota_version]


@pytest.mark.parametrize("role", ["actor_rollout", "actor_rollout_ref"])
def test_resource_identity_survives_native_colocated_worker_export(monkeypatch, role):
    import ray
    from verl.single_controller.base.worker import Worker
    from verl.single_controller.ray.base import (
        RayClassWithInitArgs,
        create_colocated_worker_cls,
    )
    from verl_speco.integration import resource_identity
    from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

    class RolloutWorker(DraftWeightPublishMixin, Worker):
        pass

    # Use the real 0.9 exporter: undecorated methods disappear, while ordinary
    # ONE_TO_ALL exports have a role prefix unavailable on replica.workers.
    exported = create_colocated_worker_cls(
        {role: RayClassWithInitArgs(cls=ray.remote(RolloutWorker))}
    )
    remote_cls = exported.cls
    assert "speco_resource_identity" in remote_cls.__ray_metadata__.method_meta.methods
    raw = remote_cls.__ray_actor_class__
    identity = {"rank": 0, "node_id": "node-a", "device_ids": ("3",)}
    monkeypatch.setattr(resource_identity, "resource_identity", lambda rank: identity)
    inner = RolloutWorker.__new__(RolloutWorker)
    inner._rank = 0
    outer = raw.__new__(raw)
    outer.worker_dict = {role: inner}
    assert outer.speco_resource_identity() == identity

    trainer = trainer_for("sync")
    trainer._require_speco_worker_group = lambda: SimpleNamespace(
        get_drafter_training_resource_metadata=lambda: [
            {
                **identity,
                "available": True,
                "worker_id": "0",
                "training_group_ranks": [0],
            }
        ]
    )
    replica = SimpleNamespace(
        replica_rank=7,
        workers=[
            SimpleNamespace(
                speco_resource_identity=SimpleNamespace(
                    remote=outer.speco_resource_identity
                )
            )
        ],
    )
    trainer.llm_server_manager = SimpleNamespace(get_replicas=lambda: [replica])
    monkeypatch.setattr(ray, "get", lambda values: values)
    result = trainer.speco_get_drafter_training_resource_metadata()
    assert result[0]["replica_rank"] == 7
    assert result[0]["device_ids"] == ("3",)


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_generation_start_uses_authoritative_replica_groups(mode):
    trainer = trainer_for(mode)
    scheduler = trainer._drafter_scheduler
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": replica,
                "training_group_ranks": ranks,
                "full_collective_ranks": ranks,
            }
            for replica, ranks in ((7, [0, 1]), (8, [2, 3]))
        ]
    )
    trainer._speco_start_rollout_idle_event_loop = lambda: (None, object())
    trainer._speco_bubble_session().start()
    assert {
        worker: (state.replica_rank, state.status)
        for worker, state in scheduler._idle_workers.items()
    } == {
        "0": (7, "generating"),
        "1": (7, "generating"),
        "2": (8, "generating"),
        "3": (8, "generating"),
    }


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_only_separate_pool_enables_strict_resource_scope(monkeypatch, mode):
    from verl_speco.integration import rollout_idle_events
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    trainer = trainer_for(mode)
    trainer._drafter_scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": 7,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )
    monkeypatch.setattr(
        SpecoRayPPOTrainer,
        "_speco_register_drafter_training_resource_metadata",
        lambda self: {},
    )
    trainer._speco_rollout_idle_event_bus_name = lambda: "standalone-bus"
    calls = []
    monkeypatch.setattr(
        rollout_idle_events,
        "configure_rollout_resource_groups",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    trainer._speco_register_drafter_training_resource_metadata()
    assert calls == (
        [(("standalone-bus", {7: ("0", "1")}), {"strict_groups": True})]
        if mode == "separate_async"
        else []
    )


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("bad_ack", [False, True])
def test_head_poll_counts_transfer_only_after_complete_validated_ack(
    monkeypatch, mode, bad_ack
):
    from verl_speco.trainer import speco_ray_trainer as legacy
    from verl_speco.trainer.v1 import bubble_mixin

    trainer = trainer_for(mode)
    clock = [0.0]
    time = SimpleNamespace(time=lambda: 1000 + clock[0], perf_counter=lambda: clock[0])
    monkeypatch.setattr(legacy, "time", time)
    monkeypatch.setattr(bubble_mixin, "time", time)
    ready = [True]
    trainer._speco_ray_refs_ready = lambda refs: ready[0]
    trainer._speco_start_target_lm_head_weight_sync = lambda plan: (
        {},
        {
            "stage": "fetch",
            "target_version": 7,
            "target_worker_ids": ("0", "1"),
            "fetch_refs": ["fetch"],
            "sync_started": 0.0,
            "fetch_started": 0.0,
            "selected_rows": 0,
            "source_vocab_size": 0,
        },
    )

    def resolve(refs):
        if refs == ["fetch"]:
            clock[0] = 2.0
            return [{"export_strategy": "test"}]
        assert refs == ["ack"]
        clock[0] = 10.0
        return [
            {
                "worker_id": worker,
                "accepted": True,
                "global_step": 6 if bad_ack and worker == "1" else 7,
            }
            for worker in ("0", "1")
        ]

    trainer._ray_get_if_needed = resolve
    trainer._speco_build_drafter_target_lm_head_sync_args = (
        lambda payload, global_step: (
            payload,
            global_step,
            None,
        )
    )
    trainer.speco_sync_target_lm_head_weight = lambda *args, **kwargs: ["ack"]
    trainer._speco_v1_cache_old_policy_head()
    costs = trainer._speco_bubble_session().costs
    assert costs.total == 2
    dispatch = trainer._pending_target_lm_head_sync
    assert dispatch["sync_started"] == 0
    ready[0] = False
    clock[0] = 7
    trainer._speco_poll_target_lm_head_prefetch()
    assert costs.total == 2
    ready[0] = True
    if bad_ack:
        with pytest.raises(RuntimeError, match="Mismatched"):
            trainer._speco_wait_target_lm_head_prefetch()
        assert not any(key.startswith("head_transfer:") for key in costs.events)
        assert 7 not in trainer._speco_ready_target_lm_head_versions
    else:
        trainer._speco_wait_target_lm_head_prefetch()
        assert costs.total == 10
        assert costs.events["head_transfer:7:0.0"] == 10
        trainer._speco_poll_target_lm_head_prefetch()
        trainer._speco_finish_target_lm_head_weight_sync(dispatch)
        assert costs.total == 10
        assert trainer._speco_ready_target_lm_head_workers[7] == ("0", "1")
        clock[0] = 25
        trainer._speco_record_target_head_transfer_cost(
            {"target_version": 7, "sync_started": 20.0}
        )
        assert costs.total == 15


@pytest.mark.parametrize(
    "failure", [None, "drafter", "actor", "features", "manifest", "commit"]
)
@pytest.mark.parametrize("use_critic", [False, True])
@pytest.mark.parametrize("remove_previous", [False, True])
def test_native_joint_checkpoint_commits_last_and_preserves_previous_on_failure(
    tmp_path, monkeypatch, failure, use_critic, remove_previous
):
    from verl_speco.trainer.v1 import checkpoint_commit

    trainer = trainer_for("sync")
    trainer.config.trainer.update(
        {
            "default_local_dir": str(tmp_path),
            "default_hdfs_dir": None,
            "remove_previous_ckpt_in_save": remove_previous,
            "max_actor_ckpt_to_keep": 1,
            "max_critic_ckpt_to_keep": 2,
        }
    )
    trainer.use_critic = use_critic
    trainer.config["critic"] = {"checkpoint": {"async_save": False}}
    original = OmegaConf.to_container(trainer.config, resolve=True)
    trainer._speco_colocate_admission_closed = True
    trainer._speco_bubble_session().close = lambda boundary: {}
    trainer._speco_checkpoint_pending_quota = lambda: False
    trainer._speco_freeze_policy = None
    trainer._speco_freeze_save_state = lambda: None
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_freeze_branch_config = lambda: {}
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"step": 7})
    previous = tmp_path / "global_step_3"
    (previous / "actor").mkdir(parents=True)
    (previous / "actor" / "complete").write_text("previous")
    from verl.trainer.ppo.utils import Role

    critic_role = str(Role.Critic)
    (previous / critic_role).mkdir()
    (previous / critic_role / "complete").write_text("previous critic")
    (previous / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 3,
                "bubble": {
                    "quota_complete": True,
                    "writer_group": [],
                    "published_plan_ids": [],
                    "policy_drafter_version": None,
                },
            }
        )
    )
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("3")
    failing = [failure]

    def check(stage):
        assert tracker.read_text() == "3"
        assert (previous / "actor" / "complete").exists()
        if failing[0] == stage:
            raise RuntimeError(stage)

    def actor_save(path, remote_path, step, max_ckpt_to_keep):
        check("actor")
        assert max_ckpt_to_keep is None
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "complete").write_text(str(step))

    worker = SimpleNamespace(
        execute_with_func_generator=SimpleNamespace(
            remote=lambda func, *args: func(None, *args),
        )
    )
    trainer.actor_rollout_wg = SimpleNamespace(
        save_checkpoint=actor_save, workers=[worker]
    )
    trainer.critic_wg = SimpleNamespace(save_checkpoint=actor_save, workers=[worker])
    trainer._ray_get_if_needed = lambda refs: refs
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: check("drafter") or []
    trainer._speco_snapshot_v1_feature_store_checkpoint = (
        lambda: check("features") or []
    )
    write_manifest = trainer._speco_write_v1_joint_checkpoint_manifest

    def manifest(**kwargs):
        check("manifest")
        return write_manifest(**kwargs)

    trainer._speco_write_v1_joint_checkpoint_manifest = manifest
    commit = checkpoint_commit.commit_latest

    def checked_commit(root, step):
        check("commit")
        assert (tmp_path / "global_step_7" / "speco_v1_manifest.json").is_file()
        return commit(root, step)

    monkeypatch.setattr(checkpoint_commit, "commit_latest", checked_commit)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            trainer._save_checkpoint()
        assert tracker.read_text() == "3"
        assert (previous / "actor" / "complete").exists()
        assert getattr(trainer, "_speco_last_checkpoint_saved_step", None) is None
        assert OmegaConf.to_container(trainer.config, resolve=True) == original
        failing[0] = None
    trainer._save_checkpoint()
    assert tracker.read_text() == "7"
    assert (tmp_path / "global_step_7" / "actor" / "complete").is_file()
    assert not (previous / "actor").exists()
    assert (previous / critic_role / "complete").is_file() == (
        not use_critic or not remove_previous
    )
    assert trainer._speco_last_checkpoint_saved_step == 7
    assert OmegaConf.to_container(trainer.config, resolve=True) == original
