# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise the real migration driver without importing GPU/Ray dependencies."""

import ast
import logging
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest


def _config(*, enabled=True, use_logits=False):
    return SimpleNamespace(
        idle_worker_writer_migration_enable=enabled, use_logits=use_logits
    )


def _methods(path, name, methods, **namespace):
    path = Path(__file__).resolve().parents[2] / path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    for node in cls.body:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    cls.bases = []
    namespace.setdefault("logger", logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _driver(*, target_hot=True, rejection=None, source_resident=False):
    target = ("1", "2")
    # Equal topology is checked by the scheduler; use two shards below to test
    # that duplicate ACKs cannot stand in for a missing destination rank.
    source = ("3", "4")
    parked, decisions, leases, released = [], [], [], []
    state = {
        "bytes": 64,
        "optimizer_steps": 20,
        "worker_progress": {"last_trained_step": 4},
        "heads": {4: {}},
    }
    hot = {source, target} if target_hot else {source}
    scheduler = SimpleNamespace(
        idle_writer_group=lambda: source,
        _writer_boundary_migration_allowed=True,
        _training_quota_debt_steps=20,
        _training_quota_data_version=6,
        _latest_published_drafter_version=4,
        _idle_worker_writer_group=source,
        _idle_worker_writer_state_version=4,
        _metadata_idle_training_groups=(source, target),
        _idle_worker_hot_prewarmed_groups=hot,
        _idle_optimizer_initialized_groups={source},
        _idle_worker_park_samples_sec=deque([0.75]),
        record_memory_park_elapsed=lambda value: None,
        _effective_idle_tail_reserve_sec=lambda config: 0,
        _effective_idle_batch_estimate_sec=lambda config: 0.25,
        _idle_gradient_accumulation_steps=lambda config, workers: 1,
        _effective_idle_startup_reserve_sec=lambda config, workers: 0
        if workers in hot
        else 3,
        select_idle_training_resources=lambda config: SimpleNamespace(
            available=True, minimum_idle_window_sec=30
        ),
    )

    def decide(config, *, cost_sec=0):
        decisions.append(cost_sec)
        return target, {"bubble/writer_switch_estimated_cost_s": cost_sec}

    scheduler.writer_switch_decision = decide

    class Group:
        exports = 0
        metadata_calls = 0
        imports = 0
        target_resident = target_hot

        def export_drafter_writer_state(self, workers, export=False):
            if export:
                self.exports += 1
                assert not self.target_resident
                return [
                    {"rank": int(worker), "available": True, "state": state}
                    for worker in workers
                ]
            self.metadata_calls += 1
            return [
                {
                    "rank": int(worker),
                    "available": rejection != "unsupported",
                    "state": {
                        "bytes": 64,
                        "device_resident": source_resident
                        if worker in source
                        else self.target_resident,
                    },
                }
                for worker in workers
            ]

        def park_drafter_training_model(self, workers):
            parked.append(tuple(workers))
            if set(target).issubset(workers):
                self.target_resident = False
            return [
                {"rank": int(worker), "parked": True, "elapsed_sec": 0.5}
                for worker in workers
            ]

        def import_drafter_writer_state(self, workers, states, validate_only=False):
            self.imports += 1
            assert not self.target_resident
            if rejection == "layout":
                raise RuntimeError("Writer shard layout differs")
            if rejection == "apply" and not validate_only:
                raise RuntimeError("Partial apply failed")
            acks = [
                {"rank": int(worker), "validated": True, "optimizer_steps": 20}
                for worker in workers
            ]
            if rejection == "validation_ack" and validate_only:
                acks[-1] = acks[0]
            if rejection == "apply_ack" and not validate_only:
                acks[-1] = acks[0]
            return acks

    group = Group()
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_migrate_idle_writer_if_useful",
            "_speco_execute_writer_state_transfer",
        },
        time=time,
        ray=SimpleNamespace(put=lambda value: value),
        reserve_rollout_training_resources=lambda bus, **kwargs: leases.append(kwargs)
        or True,
        release_rollout_training_resources=lambda bus, **kwargs: released.append(kwargs)
        or True,
    )
    obj = cls()
    obj._speco_get_drafter_scheduler = lambda: scheduler
    obj._require_speco_worker_group = lambda: group
    obj._ray_get_if_needed = lambda value: value
    obj._speco_target_lm_head_ready_for_workers = lambda version, workers: True
    obj._speco_rollout_idle_event_bus_name = lambda: "migration-test"
    obj._speco_ready_target_lm_head_workers = {4: source, 6: source}
    return obj, scheduler, group, source, target, parked, decisions, leases, released


@pytest.mark.parametrize("target_hot", [False, True])
@pytest.mark.parametrize("source_resident", [False, True])
def test_migration_parks_both_sides_and_budgets_cold_target(
    target_hot, source_resident
):
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver(
        target_hot=target_hot, source_resident=source_resident
    )
    result = obj._speco_migrate_idle_writer_if_useful(_config())
    assert result["bubble/writer_state_migrated"] == 1
    assert result["timing_s/drafter_writer_migration"] >= 0
    assert parked == [source + target]
    assert costs[-1] >= 3.5  # Do not price the target's next activation as hot.
    assert group.exports == 1
    assert scheduler._idle_worker_writer_group == target
    assert not scheduler._writer_boundary_migration_allowed
    assert source not in scheduler._idle_worker_hot_prewarmed_groups
    assert target not in scheduler._idle_worker_hot_prewarmed_groups
    assert set(leases[0]["worker_ids"]) == set(
        target + (source if source_resident else ())
    )
    assert leases[0]["worker_ids"] == released[0]["worker_ids"]


@pytest.mark.parametrize("reason", ["layout", "unsupported", "validation_ack"])
def test_failed_migration_is_not_reexported_on_every_poll(reason):
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver(
        rejection=reason
    )
    config = _config()
    obj._speco_migrate_idle_writer_if_useful(config)
    calls = (group.metadata_calls, group.exports, len(parked))
    result = obj._speco_migrate_idle_writer_if_useful(config)
    assert result["bubble/writer_migration_retry_suppressed"] == 1
    assert (group.metadata_calls, group.exports, len(parked)) == calls
    assert scheduler._idle_worker_writer_group == source
    assert len(leases) == len(released)
    scheduler._training_quota_data_version = 8
    obj._speco_migrate_idle_writer_if_useful(config)
    assert group.metadata_calls == calls[0] + 1


@pytest.mark.parametrize("reason", ["apply_ack", "apply"])
def test_migration_never_moves_owner_on_bad_ack_or_partial_apply(reason):
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver(
        rejection=reason
    )
    with pytest.raises(RuntimeError):
        obj._speco_migrate_idle_writer_if_useful(_config())
    assert scheduler._idle_worker_writer_group == source
    assert len(leases) == len(released) == 1
    assert isinstance(obj._speco_idle_completion_error, RuntimeError)


def test_migration_disabled_does_no_work():
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver()
    assert obj._speco_migrate_idle_writer_if_useful(_config(enabled=False)) == {}
    assert not (group.metadata_calls or parked or costs or leases)


def test_untrained_writer_needs_no_state_migration():
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver()
    scheduler._idle_worker_writer_state_version = None
    assert obj._speco_migrate_idle_writer_if_useful(_config()) == {}
    assert not (group.metadata_calls or parked or costs or leases)


@pytest.mark.parametrize(
    "blocked", ["target_busy", "source_busy", "lease_raced", "cost"]
)
def test_migration_rejection_never_parks_a_busy_group(blocked):
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver(
        source_resident=True
    )
    if blocked in {"target_busy", "source_busy"}:
        busy_group = target if blocked == "target_busy" else source
        scheduler.select_idle_training_resources = lambda config: SimpleNamespace(
            available=scheduler._metadata_idle_training_groups != (busy_group,),
            minimum_idle_window_sec=30,
        )
    elif blocked == "lease_raced":
        namespace = obj._speco_migrate_idle_writer_if_useful.__func__.__globals__
        namespace["reserve_rollout_training_resources"] = lambda *args, **kwargs: False
    else:
        scheduler.writer_switch_decision = lambda config, cost_sec=0: (
            target if cost_sec == 0 else None,
            {},
        )
    result = obj._speco_migrate_idle_writer_if_useful(_config())
    assert not result.get("bubble/writer_state_migrated")
    assert not parked and not group.exports
    assert scheduler._idle_worker_writer_group == source
    assert target in scheduler._idle_worker_hot_prewarmed_groups
    assert scheduler._metadata_idle_training_groups == (source, target)
    assert not released


def test_uninitialized_migration_state_is_unavailable_without_building_model():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"local_writer_state"},
    )
    obj = cls()
    obj.model = obj.optimizer = None
    assert obj.local_writer_state() is None
    assert obj.local_writer_state(export=True) is None


@pytest.mark.parametrize("use_logits", [False, True])
def test_only_hidden_state_training_requires_head_ack_before_migration(use_logits):
    obj, scheduler, group, source, target, parked, costs, leases, released = _driver()
    obj._speco_target_lm_head_ready_for_workers = lambda version, workers: False
    result = obj._speco_migrate_idle_writer_if_useful(_config(use_logits=use_logits))
    assert bool(result.get("bubble/writer_state_migrated")) is use_logits
    assert bool(group.exports) is use_logits
