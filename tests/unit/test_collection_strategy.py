# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import pytest

from verl_speco.trainer.scheduler import (
    CallbackDrafterCollectionExecutor,
    CollectionPayload,
    CollectionPlan,
    CollectionFinalizeError,
    DrafterCollectionSource,
    DrafterScheduler,
)


def _plan(*, collect: bool = True) -> CollectionPlan:
    return CollectionPlan(
        collect=collect,
        reason="collection_enabled" if collect else "interval_not_reached",
        source=DrafterCollectionSource.SGLANG,
        source_global_step=4,
        collect_interval_matched=collect,
        training_interval_matched=True,
        sample_rate=1.0,
        max_samples_per_replica=4,
        max_tokens_per_replica=None,
        hidden_window_mode="front",
        hidden_window_tokens_per_sample=512,
        hidden_window_min_rows=512,
        collection_id="collection-4",
    )


def _payload(*, source=DrafterCollectionSource.SGLANG) -> CollectionPayload:
    return CollectionPayload(
        source=source,
        buckets=[[{"input_ids": [1, 2]}]],
        collected_samples=1,
        raw_samples=2,
        collection_id="collection-4",
    )


def _worker_result(**overrides):
    value = {
        "worker_id": "0",
        "worker_incarnation": "worker-0",
        "source_global_step": 4,
        "accepted_samples": 1,
        "rejected_samples": 0,
        "buffer_version_before": 3,
        "buffer_version_after": 4,
        "data_version": 4,
        "collected": True,
        "reason": "collection_completed",
        "collection_id": "collection-4",
    }
    value.update(overrides)
    return value


def _staged_result(**overrides):
    values = {
        "accepted_samples": 0,
        "buffer_version_after": 3,
        "collected": False,
        "reason": "collection_staged",
        "staged_samples": 1,
    }
    values.update(overrides)
    return _worker_result(**values)


def _executor(*, commit_results=None, stage_results=None):
    return CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda buckets: stage_results or [_staged_result()],
        commit_submit=lambda buckets: commit_results or [_worker_result()],
        abort_submit=lambda buckets: [],
        rollback_submit=lambda buckets: [],
        finalize_submit=lambda buckets: [
            _worker_result(
                worker_id=str(owner), reason="collection_finalized", accepted_samples=0
            )
            for owner in range(len(buckets))
        ],
        resolve=lambda value: value,
    )


def test_collection_sets_step_submits_buckets_and_returns_outcome() -> None:
    events = []
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: events.append(("set_step", step)) or "step-ref",
        stage_submit=lambda buckets: events.append(("stage", buckets)) or "stage-ref",
        commit_submit=lambda buckets: events.append(("commit", buckets))
        or "commit-ref",
        abort_submit=lambda buckets: events.append(("abort", buckets)) or "abort-ref",
        rollback_submit=lambda buckets: events.append(("rollback", buckets))
        or "rollback-ref",
        finalize_submit=lambda buckets: events.append(("finalize", buckets))
        or "finalize-ref",
        resolve=lambda value: events.append(("resolve", value))
        or ([_staged_result()] if value == "stage-ref" else [])
        or ([_worker_result()] if value == "commit-ref" else [])
        or (
            [_worker_result(reason="collection_finalized")]
            if value == "finalize-ref"
            else []
        ),
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert events[0:2] == [("set_step", 4), ("resolve", "step-ref")]
    assert events[2][0] == "stage"
    assert events[3] == ("resolve", "stage-ref")
    assert events[4][0] == "commit"
    assert any(event[0] == "finalize" for event in events if isinstance(event, tuple))
    assert outcome.attempted
    assert outcome.collected
    assert outcome.collected_samples == 1
    assert outcome.raw_samples == 2


def test_partial_worker_acceptance_fails_closed() -> None:
    executor = _executor(commit_results=[_worker_result(accepted_samples=0)])

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert outcome.attempted
    assert not outcome.collected
    assert outcome.collected_samples == 0
    assert outcome.reason == "collection_result_incomplete"


def test_worker_rejection_fails_closed() -> None:
    executor = _executor(
        commit_results=[
            _worker_result(
                accepted_samples=0,
                rejected_samples=1,
                collected=False,
                reason="samples_rejected",
            )
        ]
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert not outcome.collected
    assert outcome.reason == "collection_worker_rejected_samples"


def test_worker_version_mismatch_fails_closed() -> None:
    executor = _executor(commit_results=[_worker_result(data_version=3)])

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert not outcome.collected
    assert outcome.collected_samples == 0
    assert outcome.reason == "collection_version_mismatch"


def test_missing_owner_result_fails_closed_even_when_total_matches() -> None:
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[{"value": "a"}], [{"value": "b"}]],
        collected_samples=2,
        raw_samples=2,
        collection_id="collection-4",
    )
    executor = _executor(
        stage_results=[
            _staged_result(worker_id="0"),
            _staged_result(worker_id="1"),
        ],
        commit_results=[_worker_result(accepted_samples=2)],
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), payload
    )

    assert not outcome.collected
    assert outcome.reason == "collection_result_incomplete"


def test_stage_failure_aborts_before_commit() -> None:
    events = []
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda buckets: [
            _staged_result(staged_samples=0, reason="collection_already_staged")
        ],
        commit_submit=lambda buckets: events.append("commit"),
        abort_submit=lambda buckets: events.append("abort") or [],
        rollback_submit=lambda buckets: events.append("rollback") or [],
        finalize_submit=lambda buckets: events.append("finalize") or [],
        resolve=lambda value: value,
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert not outcome.collected
    assert outcome.reason == "collection_stage_failed"
    assert events == ["abort"]


def test_collection_id_must_match_before_worker_rpc() -> None:
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[{"value": "a"}]],
        collected_samples=1,
        collection_id="another-collection",
    )

    with pytest.raises(ValueError, match="collection_id"):
        DrafterScheduler(collection_executor=_executor()).execute_collection_plan(
            _plan(), payload
        )


def test_commit_validation_failure_rolls_back_all_workers() -> None:
    events = []
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda buckets: [_staged_result()],
        commit_submit=lambda buckets: [_worker_result(data_version=3)],
        abort_submit=lambda buckets: events.append("abort") or [],
        rollback_submit=lambda buckets: events.append("rollback") or [],
        finalize_submit=lambda buckets: events.append("finalize") or [],
        resolve=lambda value: value,
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), _payload()
    )

    assert not outcome.collected
    assert outcome.collected_samples == 0
    assert events == ["rollback"]


def test_inactive_collection_plan_does_not_call_executor() -> None:
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: (_ for _ in ()).throw(AssertionError("unexpected")),
        stage_submit=lambda buckets: (_ for _ in ()).throw(
            AssertionError("unexpected")
        ),
        commit_submit=lambda buckets: (_ for _ in ()).throw(
            AssertionError("unexpected")
        ),
        abort_submit=lambda buckets: (_ for _ in ()).throw(
            AssertionError("unexpected")
        ),
        rollback_submit=lambda buckets: (_ for _ in ()).throw(
            AssertionError("unexpected")
        ),
        finalize_submit=lambda buckets: (_ for _ in ()).throw(
            AssertionError("unexpected")
        ),
        resolve=lambda value: value,
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(collect=False), _payload()
    )

    assert not outcome.attempted
    assert not outcome.collected
    assert outcome.reason == "interval_not_reached"


def test_empty_collection_payload_is_skipped() -> None:
    executor = _executor()
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[]],
        collected_samples=0,
        collection_id="collection-4",
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), payload
    )

    assert not outcome.attempted
    assert outcome.reason == "empty_collection_payload"


def test_collection_executor_sends_control_request_for_empty_worker_bucket() -> None:
    submitted = {}

    def capture(phase):
        def submit(requests):
            submitted[phase] = requests
            return []

        return submit

    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=capture("stage"),
        commit_submit=capture("commit"),
        abort_submit=capture("abort"),
        rollback_submit=capture("rollback"),
        finalize_submit=capture("finalize"),
        resolve=lambda value: value,
    )
    sample = {"input_ids": [1, 2]}
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[sample], []],
        collected_samples=1,
        collection_id="collection-4",
    )

    executor.stage(payload)
    executor.commit(payload)
    executor.abort(payload)
    executor.rollback(payload)
    executor.finalize(payload)

    assert submitted["stage"] == [
        [{"collection_id": "collection-4", "samples": [sample]}],
        [{"collection_id": "collection-4", "samples": []}],
    ]
    expected_control_requests = [
        [{"collection_id": "collection-4", "samples": None}],
        [{"collection_id": "collection-4", "samples": None}],
    ]
    for phase in ("commit", "abort", "rollback", "finalize"):
        assert submitted[phase] == expected_control_requests


def test_collection_accepts_empty_worker_bucket_for_single_writer_routing() -> None:
    sample = {"input_ids": [1, 2]}
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[sample], []],
        collected_samples=1,
        raw_samples=1,
        collection_id="collection-4",
    )
    executor = _executor(
        stage_results=[
            _staged_result(worker_id="0", staged_samples=1),
            _staged_result(worker_id="1", staged_samples=0),
        ],
        commit_results=[
            _worker_result(worker_id="0", accepted_samples=1),
            _worker_result(worker_id="1", accepted_samples=0),
        ],
    )

    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), payload
    )

    assert outcome.collected
    assert outcome.reason == "collection_completed"
    assert outcome.collected_samples == 1
    assert outcome.finalized


def test_collection_rejects_payload_from_another_source() -> None:
    executor = _executor()

    with pytest.raises(ValueError, match="does not match"):
        DrafterScheduler(collection_executor=executor).execute_collection_plan(
            _plan(), _payload(source=DrafterCollectionSource.OLD_LOGPROB)
        )


@pytest.mark.parametrize(
    "failure", ["rpc_error", "missing_ack", "wrong_id", "worker_restarted"]
)
def test_finalize_failure_is_committed_unfinalized_and_stops_scheduling(failure):
    events = []

    def finalize(requests):
        events.append("finalize")
        if failure == "rpc_error":
            raise RuntimeError("partial finalize RPC failed")
        if failure == "missing_ack":
            return []
        overrides = (
            {"collection_id": "another"}
            if failure == "wrong_id"
            else {"worker_incarnation": "restarted"}
        )
        return [_worker_result(reason="collection_finalized", **overrides)]

    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda requests: [_staged_result()],
        commit_submit=lambda requests: events.append("commit") or [_worker_result()],
        finalize_submit=finalize,
        abort_submit=lambda requests: events.append("abort"),
        rollback_submit=lambda requests: events.append("rollback"),
        resolve=lambda value: value,
    )
    with pytest.raises(
        CollectionFinalizeError, match="collection_finalize_failed"
    ) as caught:
        DrafterScheduler(collection_executor=executor).execute_collection_plan(
            _plan(), _payload()
        )
    outcome = caught.value.outcome
    assert outcome.reason == "collection_finalize_failed"
    assert outcome.committed_unfinalized
    assert not outcome.collected and not outcome.finalized
    assert outcome.collected_samples == 0
    assert outcome.metrics()["drafter/collection_completed"] == 0
    assert outcome.metrics()["drafter/collection_committed_unfinalized"] == 1
    assert events == ["commit", "finalize"]


def test_missing_nonempty_owner_commit_rolls_back_without_finalize():
    events = []
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[{"value": "a"}], [{"value": "b"}]],
        collected_samples=2,
        collection_id="collection-4",
    )
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda requests: [
            _staged_result(worker_id="0"),
            _staged_result(worker_id="1"),
        ],
        commit_submit=lambda requests: [_worker_result()],
        finalize_submit=lambda requests: events.append("finalize") or [],
        abort_submit=lambda requests: events.append("abort"),
        rollback_submit=lambda requests: events.append("rollback"),
        resolve=lambda value: value,
    )
    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), payload
    )
    assert not outcome.collected
    assert outcome.reason == "collection_result_incomplete"
    assert events == ["rollback"]


def test_partial_finalize_preserves_remaining_journal_instead_of_split_rollback():
    journals = {"0", "1"}
    committed = {"0", "1"}

    def finalize(requests):
        journals.remove("0")  # owner 0 finalized before owner 1's RPC failed
        raise RuntimeError("owner 1 unavailable")

    def rollback(requests):
        committed.difference_update(journals)

    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda requests: [
            _staged_result(worker_id=str(owner)) for owner in range(2)
        ],
        commit_submit=lambda requests: [
            _worker_result(worker_id=str(owner)) for owner in range(2)
        ],
        finalize_submit=finalize,
        abort_submit=lambda requests: None,
        rollback_submit=rollback,
        resolve=lambda value: value,
    )
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[{"value": "a"}], [{"value": "b"}]],
        collected_samples=2,
        collection_id="collection-4",
    )
    with pytest.raises(CollectionFinalizeError):
        DrafterScheduler(collection_executor=executor).execute_collection_plan(
            _plan(), payload
        )
    assert committed == {"0", "1"}
    assert journals == {"1"}


def test_missing_nonempty_owner_stage_aborts_before_commit():
    events = []
    executor = CallbackDrafterCollectionExecutor(
        set_step=lambda step: None,
        stage_submit=lambda requests: [_staged_result()],
        commit_submit=lambda requests: events.append("commit") or [],
        finalize_submit=lambda requests: events.append("finalize") or [],
        abort_submit=lambda requests: events.append("abort"),
        rollback_submit=lambda requests: events.append("rollback"),
        resolve=lambda value: value,
    )
    payload = CollectionPayload(
        source=DrafterCollectionSource.SGLANG,
        buckets=[[{"value": "a"}], [{"value": "b"}]],
        collected_samples=2,
        collection_id="collection-4",
    )
    outcome = DrafterScheduler(collection_executor=executor).execute_collection_plan(
        _plan(), payload
    )
    assert not outcome.collected
    assert outcome.reason == "collection_stage_failed"
    assert events == ["abort"]


def test_collection_finalize_error_preserves_failure_state_when_serialized():
    import pickle
    from verl_speco.trainer.scheduler import CollectionOutcome

    failure = CollectionFinalizeError(
        "partial",
        CollectionOutcome(
            attempted=True,
            collected=False,
            reason="collection_finalize_failed",
            source_global_step=2,
            committed_unfinalized=True,
        ),
    )
    restored = pickle.loads(pickle.dumps(failure))
    assert restored.collection_id == "partial"
    assert restored.outcome == failure.outcome
    assert str(restored) == str(failure)
