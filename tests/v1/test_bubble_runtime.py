# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Resource and accounting invariants without optional trainer dependencies."""

from types import SimpleNamespace

import pytest

from verl_speco.integration.resource_identity import map_training_replicas
from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle, QuotaCostLedger
from verl_speco.trainer.v1.bubble_workers import normalize_publish_ack


def test_cost_ledger_deduplicates_completion_and_uses_maximum_rank_cost():
    ledger = QuotaCostLedger()
    plan = SimpleNamespace(plan_id="one")
    outcome = SimpleNamespace(
        raw_results=[{"elapsed_sec": 2}, {"elapsed_sec": 4}], elapsed_sec=20
    )
    ledger.observe_training(plan, outcome)
    ledger.observe_training(plan, outcome)
    ledger.record("head:1", 1)
    ledger.record("publish:1", 3)
    assert ledger.total == 8
    with pytest.raises(ValueError):
        ledger.record("bad", -1)


def test_physical_identity_mapping_is_independent_of_logical_replica_rank():
    records = [
        {
            "available": True,
            "worker_id": str(rank),
            "node_id": "node",
            "device_ids": [str(rank)],
            "training_group_ranks": [0, 1],
        }
        for rank in range(2)
    ]
    replicas = {7: [{"node_id": "node", "device_ids": ["0", "1"]}]}
    assert [
        record["replica_rank"] for record in map_training_replicas(records, replicas)
    ] == [7, 7]
    with pytest.raises(RuntimeError, match="spans"):
        map_training_replicas(
            records,
            {
                7: [{"node_id": "node", "device_ids": ["0"]}],
                8: [{"node_id": "node", "device_ids": ["1"]}],
            },
        )


@pytest.mark.parametrize(
    "replicas",
    [
        {7: [{"node_id": "other", "device_ids": ["0"]}]},
        {
            7: [{"node_id": "node", "device_ids": ["0"]}],
            8: [{"node_id": "node", "device_ids": ["0"]}],
        },
    ],
)
def test_invalid_resource_ownership_fails_before_training(replicas):
    with pytest.raises(RuntimeError):
        map_training_replicas(
            [
                {
                    "available": True,
                    "worker_id": "0",
                    "node_id": "node",
                    "device_ids": ["0"],
                    "training_group_ranks": [0],
                }
            ],
            replicas,
        )


def test_distinct_training_ranks_cannot_share_one_physical_device():
    records = [
        {
            "available": True,
            "worker_id": str(rank),
            "node_id": "node",
            "device_ids": ["0"],
            "training_group_ranks": [0, 1],
        }
        for rank in range(2)
    ]
    with pytest.raises(RuntimeError, match="overlap physical"):
        map_training_replicas(
            records,
            {7: [{"node_id": "node", "device_ids": ["0", "1"]}]},
        )


def test_same_device_number_on_different_nodes_is_independent():
    records = [
        {
            "available": True,
            "worker_id": str(rank),
            "node_id": f"node-{rank}",
            "device_ids": ["0"],
            "training_group_ranks": [0, 1],
        }
        for rank in range(2)
    ]
    assert len(map_training_replicas(records, {7: records})) == 2


def test_publish_rank_domains_do_not_alias():
    ack = {"worker_id": "0", "published": True, "published_version": 9}
    assert normalize_publish_ack(ack, "hybrid", 0)["worker_id"] == "hybrid:0"
    assert normalize_publish_ack(ack, "standalone", 0)["worker_id"] == "standalone:0"
    with pytest.raises(RuntimeError):
        normalize_publish_ack(ack, "standalone", 1)
    assert (
        normalize_publish_ack({"worker_rank": 0}, "standalone:7", 0)["worker_id"]
        == "standalone:7:0"
    )
    assert (
        normalize_publish_ack({"worker_rank": 0}, "standalone:8", 0)["worker_id"]
        == "standalone:8:0"
    )


def test_completion_queue_is_idempotent_and_owns_its_cost_snapshot():
    session = BubbleLifecycle(None)
    plan = SimpleNamespace(plan_id="one", publish_after_success=True)
    outcome = SimpleNamespace(trained=True, raw_results=[], elapsed_sec=2)
    session.enqueue(plan, outcome)
    session.enqueue(plan, outcome)
    assert len(session.pending) == 1
    costs = session.costs
    session.published("one")
    assert costs.total == 2
    assert session.costs.total == 0
    assert not session.pending
    session.enqueue(plan, outcome)
    assert not session.pending
    assert session.costs.total == 0


def test_partial_quota_keeps_costs_without_publishing():
    scheduler = SimpleNamespace(_training_quota_debt_steps=2)
    owner = SimpleNamespace(_speco_get_drafter_scheduler=lambda: scheduler)
    session = BubbleLifecycle(owner)
    outcome = SimpleNamespace(
        trained=True, raw_results=[[{"elapsed_sec": 3}]], elapsed_sec=9
    )
    session.enqueue(
        SimpleNamespace(plan_id="partial", publish_after_success=True), outcome
    )
    assert not session.pending
    scheduler._training_quota_debt_steps = 0
    session.enqueue(
        SimpleNamespace(plan_id="final", publish_after_success=True), outcome
    )
    assert len(session.pending) == 1
    assert session.costs.total == 6


def test_failed_event_thread_join_prevents_reclaim_and_parking():
    def failed_join(*args):
        raise RuntimeError("thread still active")

    owner = SimpleNamespace(_speco_stop_rollout_idle_event_loop=failed_join)
    session = BubbleLifecycle(owner)
    with pytest.raises(RuntimeError, match="still active"):
        session.close("ppo")
    assert not owner._speco_rollout_idle_generation_open


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1])
def test_invalid_cost_cannot_become_roi_evidence(cost):
    with pytest.raises(ValueError):
        QuotaCostLedger().record("invalid", cost)


def test_overlapping_phases_and_parallel_preparation_are_not_double_charged():
    ledger = QuotaCostLedger()
    ledger.record_interval("head", 0, 4)
    ledger.observe_training(
        SimpleNamespace(plan_id="one"),
        SimpleNamespace(
            raw_results=[
                {
                    "elapsed_sec": 3,
                    "preflight_elapsed_sec": 2,
                    "cost_train_started_ts": 4,
                    "cost_train_finished_ts": 7,
                    "cost_prepare_started_ts": 2,
                    "cost_prepare_finished_ts": 4,
                },
                {
                    "elapsed_sec": 1,
                    "preflight_elapsed_sec": 1,
                    "cost_train_started_ts": 4,
                    "cost_train_finished_ts": 5,
                    "cost_prepare_started_ts": 3,
                    "cost_prepare_finished_ts": 4,
                },
            ],
            elapsed_sec=100,
        ),
    )
    ledger.record_interval("publish", 7, 8)
    ledger.record_interval("head", 0, 99)
    assert ledger.total == 8
    assert ledger.events["prepare:one"] == 2
    assert ledger.events["train:one"] == 3
