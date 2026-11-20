# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import json

import pytest
import torch
from verl_speco.integration import rollout_reference as reference
from verl_speco.integration import vllm_rollout_capture as capture


@pytest.fixture
def window_reference(tmp_path, monkeypatch):
    from verl_speco.producer.rollout_window_store import write_window

    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    monkeypatch.setattr(capture, "plan_directory", lambda p: directory)
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    plan = {
        "sample_id": directory.name,
        "global_step": 2,
        "replica_rank": 0,
        "window_store": True,
        "start": 0,
        "end": 2,
        "request_id": "request",
        "target_version": 1,
    }
    write_window(
        directory,
        {
            "hidden_states": torch.ones(2, 3, dtype=torch.bfloat16),
            "positions": torch.tensor([0, 1]),
            "token_ids": torch.tensor([1, 2]),
            "layer_ids": [1],
            "layout": "aux",
            "last_hidden_norm": "post_norm",
        },
        plan,
    )
    return reference.submit(plan, [1], [2, 3], "request", 1), directory


def test_background_reference_roundtrip_and_cleanup(window_reference):
    handle, directory = window_reference
    handle["evidence"] = {"_verl_hard_score": 4.0}
    result = reference.resolve(handle)
    assert torch.equal(
        result["hidden_states"], torch.ones(1, 2, 3, dtype=torch.bfloat16)
    )
    replay = reference.resolve(handle)
    assert torch.equal(result["hidden_states"], replay["hidden_states"])
    assert (directory / "window.safetensors").exists()
    assert not (directory / "assembled.ready").exists()
    assert result["target_version"] == 1
    assert result["_verl_hard_score"] == 4.0
    assert directory.exists()
    reference.begin_collection([handle])
    reference.finish_collection([handle], collected=True)
    reference.acknowledge(handle)
    assert not directory.exists()


def test_reference_missing_ready_fails_instead_of_silent_skip(window_reference):
    handle, directory = window_reference
    (directory / "window.ready").unlink()
    with pytest.raises(TimeoutError):
        reference.resolve(handle, timeout=0)


def test_reference_rejects_wrong_version_before_cleanup(window_reference):
    handle, directory = window_reference
    handle["target_version"] = 9
    with pytest.raises(ValueError, match="mismatch"):
        reference.resolve(handle)
    assert directory.exists()


def test_inline_fallback_does_not_create_an_assembled_reference():
    assert reference.submit({}, [], [], "r", 0) is None


def test_byte_budget_held_until_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setenv("SPECO_REFERENCE_MAX_BYTES", "64")
    monkeypatch.setenv("SPECO_REFERENCE_SAMPLE_BYTES", "64")
    ticket = reference._reserve({})
    assert ticket is not None and reference._reserve({}) is None
    reference._release({"budget": ticket})
    reference._release({"budget": ticket})
    assert reference._reserve({}) is not None


def test_orphan_recovery_requires_stopped_run():
    with pytest.raises(RuntimeError, match="Stop the run"):
        reference.discard({})


def test_corrupt_window_file_is_rejected_before_load(window_reference):
    handle, directory = window_reference
    path = directory / "window.safetensors"
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(RuntimeError, match="checksum"):
        reference.resolve(handle)
    assert path.exists()


def test_stopped_recovery_releases_files_and_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda p: tmp_path / "sample")
    directory = tmp_path / "sample"
    directory.mkdir()
    (directory / "live").touch()
    handle = {
        "plan": {"sample_id": "s"},
        "target_version": 1,
        "budget": reference._reserve({}),
    }
    reference.discard(handle, preparation_stopped=True)
    reference.discard(handle, preparation_stopped=True)
    assert not directory.exists()
    assert not list(tmp_path.rglob("*.reservation"))


def test_nonthrowing_failed_collection_cannot_ack():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="source retained"):
        reference.require_collection_commit(
            SimpleNamespace(attempted=True, collected=False, reason="stage_failed")
        )
    reference.require_collection_commit(SimpleNamespace(attempted=True, collected=True))
    reference.require_collection_commit(
        SimpleNamespace(attempted=False, collected=False)
    )


def test_short_window_returns_without_waiting_for_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "plan_directory", lambda p: tmp_path)
    handle = {
        "kind": "speco_rollout_reference_v1",
        "plan": {
            "sample_id": "s",
            "window_store": True,
            "request_id": "r",
            "target_version": 0,
            "start": 0,
            "end": 8,
        },
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }
    assert reference.resolve(handle, timeout=0) is None


def test_direct_window_has_no_second_background_writer(window_reference, monkeypatch):
    handle, directory = window_reference

    def fail(*args):
        raise AssertionError("Window must not assemble an inline sample")

    monkeypatch.setattr(capture, "load_sample", fail)
    assert reference.resolve(handle) is not None
    assert not (directory / "assembled.ready").exists()
    assert (directory / "reference.json").exists()


def test_failed_cleanup_retains_budget_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(capture, "plan_directory", lambda p: source)
    monkeypatch.setattr(capture, "cleanup", lambda p: None)
    budget = reference._reserve({})
    handle = {
        "plan": {"sample_id": "s"},
        "request_id": "r",
        "target_version": 0,
        "budget": budget,
    }
    reference.begin_collection([handle])
    reference.finish_collection([handle], collected=True)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        reference.acknowledge(handle)
    assert list(tmp_path.rglob("*.reservation"))


@pytest.mark.parametrize("policy", ["wait", "defer"])
def test_sync_reference_parent_runs_once_before_resolution(monkeypatch, policy):
    from types import SimpleNamespace
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin
    from verl_speco.trainer.v1 import feature_adapter

    import sys
    import threading

    preparing = threading.Event()
    parent_finished = threading.Event()
    handle = {
        "kind": "speco_rollout_reference_v1",
        "plan": {
            "sample_id": "s",
            "window_store": True,
            "request_id": "r",
            "target_version": 0,
            "start": 0,
            "end": 1,
        },
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }

    def prepare(ref):
        preparing.set()
        assert parent_finished.wait(5)
        return {"prepared": True}

    monkeypatch.setattr(reference, "resolve", prepare)
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(
            kv_batch_get=lambda **kwargs: {"extra_fields": [{"drafter_sample": handle}]}
        ),
    )
    calls = []
    monkeypatch.setattr(
        feature_adapter,
        "validate_rollout_samples",
        lambda *a, **k: calls.append("validate"),
    )

    class Parent:
        def _compute_old_log_prob(self, batch, metrics):
            assert preparing.wait(5)
            calls.append("parent")
            parent_finished.set()
            return batch

    class Harness(SpecoV1Mixin, Parent):
        def __getattr__(self, name):
            raise AttributeError(name)

        def _speco_rollout_idle_worker_enabled(self):
            return False

        def _speco_async_rollout_capture_enabled(self):
            return False

        def _speco_v1_batch_data(self, batch):
            calls.append("resolve")
            assert self._speco_reference_prefetcher.take(handle) == {"prepared": True}
            return None, SimpleNamespace(meta_info={}, non_tensor_batch={})

        def _speco_collect_generation_samples(self, evidence):
            calls.append("collect")
            return 0

    trainer = Harness()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout={
                "drafter": {
                    "training": {
                        "vllm_capture_reference": True,
                        "vllm_capture_pending_policy": policy,
                        "collect_hidden_states_from_vllm": True,
                    }
                }
            }
        )
    )
    trainer.global_steps = 1
    trainer._speco_v1_state = {}
    trainer._speco_last_raw_drafter_samples = 0
    trainer.parameter_sync_step = 1
    metrics = {}
    batch = SimpleNamespace(keys=[], partition_id="test", tags=[])
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    if policy == "defer":
        assert calls == ["parent"]
        trainer._speco_finish_deferred_reference_collection(metrics)
        assert trainer._speco_reference_deferred_batch is None
        assert metrics["timing_s/drafter_reference_collection_tail"] >= 0
    assert calls == ["parent", "resolve", "validate", "collect"]
    trainer._speco_reference_prefetcher.close()


def test_prefetch_is_bounded_and_overlaps_parent_work():
    import threading

    entered, release = threading.Event(), threading.Event()
    calls = []

    def prepare(ref):
        calls.append(ref["id"])
        entered.set()
        assert release.wait(5)
        return ref["id"]

    pool = reference.ReferencePrefetcher(capacity=1, resolver=prepare)
    try:
        assert pool.submit({"id": 1})
        assert entered.wait(5)
        assert pool.submit({"id": 1})
        assert not pool.submit({"id": 2})
        assert pool.ready() == ()
        release.set()
        assert pool.take({"id": 1}) == 1
        assert calls == [1]
        assert pool.take({"id": 2}) == 2
    finally:
        release.set()
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit({"id": 3})


def test_prefetch_propagates_error_without_ack():
    def fail(ref):
        raise RuntimeError("source retained")

    pool = reference.ReferencePrefetcher(resolver=fail)
    pool.submit({"id": 1})
    with pytest.raises(RuntimeError, match="source retained"):
        pool.take({"id": 1})
    pool.close()


def test_reference_budget_is_shared_by_worker_processes(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("SPECO_CAPTURE_EXPORT_ROOT", str(tmp_path))
    monkeypatch.setenv("SPECO_REFERENCE_MAX_BYTES", "8192")
    monkeypatch.setenv("SPECO_REFERENCE_SAMPLE_BYTES", "4096")
    script = "from verl_speco.integration.rollout_reference import _reserve; print(int(_reserve({}) is not None))"

    def reserve(_):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return int(result.stdout.strip())

    with ThreadPoolExecutor(max_workers=6) as pool:
        admitted = list(pool.map(reserve, range(6)))
    assert sum(admitted) == 2
    assert len(list(tmp_path.glob(".reference-budget/*/*.reservation"))) == 2


def test_reference_budget_rejects_parent_directory_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="budget identity"):
        reference._release({"budget": {"owner": "..", "ticket": "x.reservation"}})


def test_prefetch_accounts_for_load_and_owned_clone_bytes():
    pool = reference.ReferencePrefetcher(max_bytes=8, resolver=lambda ref: ref)
    first = {"id": 1, "budget": {"bytes": 4}}
    second = {"id": 2, "budget": {"bytes": 1}}
    try:
        assert pool.submit(first)
        assert not pool.submit(second)
        assert pool.take(first) == first
        assert pool._bytes == 0
        assert pool.submit(second)
    finally:
        pool.close()


def test_transaction_intent_blocks_ack_and_replay_until_commit(tmp_path, monkeypatch):
    import json
    import shutil

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda plan: source)
    monkeypatch.setattr(capture, "cleanup", lambda plan: shutil.rmtree(source))
    handle = {
        "plan": {"sample_id": "s"},
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }
    reference.begin_collection([handle])
    with pytest.raises(RuntimeError, match="uncommitted"):
        reference.acknowledge(handle)
    assert source.exists()
    with pytest.raises(RuntimeError, match="already exists"):
        reference.begin_collection([handle])
    reference.finish_collection([handle], collected=True)
    altered = {**handle, "response": [99]}
    with pytest.raises(RuntimeError, match="mismatched"):
        reference.acknowledge(altered)
    assert source.exists()
    reference.acknowledge(handle)
    assert not source.exists()
    assert (
        json.loads(reference._receipt_path(handle).read_text())["state"] == "RELEASED"
    )
    with pytest.raises(RuntimeError, match="already exists"):
        reference.begin_collection([handle])


def test_stopped_recovery_cancels_ambiguous_transaction_without_commit_ack(
    tmp_path, monkeypatch
):
    import json

    source = tmp_path / "source"
    source.mkdir()
    (source / "stale.pending").touch()
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "plan_directory", lambda plan: source)
    handle = {
        "plan": {"sample_id": "s"},
        "request_id": "r",
        "target_version": 0,
        "prompt": [1],
        "response": [2],
    }
    reference.begin_collection([handle])
    reference.discard(handle, preparation_stopped=True)
    reference.discard(handle, preparation_stopped=True)
    assert not source.exists()
    assert not (tmp_path / ".reference-acks" / "s.json").exists()
    assert (
        json.loads(reference._receipt_path(handle).read_text())["state"] == "CANCELLED"
    )


def test_reference_keeps_inline_hard_sampling_score_rules():
    from types import SimpleNamespace

    handle = {"kind": "speco_rollout_reference_v1"}
    output = SimpleNamespace(
        extra_fields={
            "drafter_sample": handle,
            "_verl_request_mean_accept_len": 3.5,
            "_speco_sample_id": "must-not-override",
        }
    )
    capture.finalize_rollout_sample(output, {})
    assert handle["evidence"]["_verl_accept_len"] == 3.5
    assert handle["evidence"]["_verl_hard_score"] == -3.5
    assert "_speco_sample_id" not in handle["evidence"]
    output.extra_fields["_verl_hard_score"] = 9.0
    capture.finalize_rollout_sample(output, {})
    assert handle["evidence"]["_verl_hard_score"] == 9.0
    with pytest.raises(RuntimeError, match="feature identity"):
        reference._merge_evidence({}, {"evidence": {"target_version": 99}})


def test_fit_shutdown_closes_both_workers_and_preserves_uncommitted_sources():
    from types import SimpleNamespace
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

    calls = []

    def failed_close():
        calls.append("prefetch")
        raise RuntimeError("prepare failed")

    trainer = SpecoV1Mixin()
    trainer._speco_reference_prefetcher = SimpleNamespace(close=failed_close)
    trainer._speco_reference_finalizer = SimpleNamespace(
        close=lambda: calls.append("finalize")
    )
    trainer._speco_sync_reference_receipts = {"s": {"uncommitted": True}}
    with pytest.raises(RuntimeError, match="prepare failed"):
        trainer._speco_close_reference_workers()
    assert calls == ["prefetch", "finalize"]
    assert trainer._speco_sync_reference_receipts == {"s": {"uncommitted": True}}


def test_drain_ready_never_waits_or_changes_other_batch_ownership():
    import threading

    gate = threading.Event()

    def prepare(ref):
        if ref["id"] == 2:
            assert gate.wait(5)
        return ref["id"]

    pool = reference.ReferencePrefetcher(resolver=prepare)
    try:
        first, second = {"id": 1}, {"id": 2}
        pool.submit(first)
        pool.submit(second)
        pool._jobs[pool.identity(first)][0].result(timeout=5)
        assert pool.drain_ready([second]) == {}
        assert pool.drain_ready([first, second]) == {pool.identity(first): 1}
        assert pool.identity(second) in pool._jobs
    finally:
        gate.set()
        pool.close()


def test_reference_run_scopes_files_and_forwards_epoch_without_changing_actor_runs(
    tmp_path, monkeypatch
):
    from verl_speco.main import configure_reference_run, drafter_diagnostic_env

    monkeypatch.setenv("SPECO_CAPTURE_EXPORT_ROOT", str(tmp_path))
    # Track absence as well: configure_reference_run writes the real process
    # environment, and later tests must not inherit this test's ownership epoch.
    monkeypatch.setenv("SPECO_CAPTURE_RUN_ID", "temporary")
    monkeypatch.delenv("SPECO_CAPTURE_RUN_ID")
    configure_reference_run({})
    import os

    assert os.environ["SPECO_CAPTURE_EXPORT_ROOT"] == str(tmp_path)
    assert "SPECO_CAPTURE_RUN_ID" not in os.environ
    configure_reference_run(
        {
            "actor_rollout_ref": {
                "rollout": {"drafter": {"training": {"vllm_capture_reference": True}}}
            }
        }
    )
    epoch = os.environ["SPECO_CAPTURE_RUN_ID"]
    assert os.environ["SPECO_CAPTURE_EXPORT_ROOT"] == str(tmp_path / epoch)
    assert (tmp_path / epoch / "run.json").exists()
    assert drafter_diagnostic_env()["SPECO_CAPTURE_RUN_ID"] == epoch


def test_manifest_recovery_reads_the_full_window_once(window_reference, monkeypatch):
    from verl_speco.producer import rollout_window_store

    handle, directory = window_reference
    original = rollout_window_store.load_window
    calls = []

    def load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rollout_window_store, "load_window", load)
    assert reference.resolve(handle) is not None
    assert calls == [1]
    assert (directory / "manifest.ready").exists()


@pytest.mark.parametrize(
    "field,value", [("dtype", "torch.float16"), ("width", 0), ("rows", True)]
)
def test_window_metadata_contract_is_checked_before_consume(
    window_reference, field, value
):
    import json

    handle, directory = window_reference
    ready = directory / "window.ready"
    metadata = json.loads(ready.read_text())
    metadata[field] = value
    ready.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="tensor contract"):
        reference.resolve(handle)
    assert (directory / "window.safetensors").exists()


@pytest.mark.parametrize("empty_response", [False, True], ids=["normal", "empty"])
def test_export_error_rejects_resolve_and_ack(window_reference, empty_response):
    handle, directory = window_reference
    if empty_response:
        handle["response"] = []
    (directory / "error").touch()
    with pytest.raises(RuntimeError, match="export failed"):
        reference.resolve(handle)
    with pytest.raises(RuntimeError, match="source retained"):
        reference.acknowledge(handle)
    assert (directory / "error").exists()


def test_reference_reservation_identity_cannot_be_substituted(window_reference):
    handle, directory = window_reference
    handle["budget"] = {**handle["budget"], "bytes": 1}
    with pytest.raises(ValueError, match="reservation mismatch"):
        reference.resolve(handle)
    assert directory.exists()


def test_stopped_export_error_is_reclaimed_without_fabricating_ack(window_reference):
    handle, directory = window_reference
    (directory / "error").touch()
    (directory / "stale.pending").touch()
    reference.discard(handle, preparation_stopped=True)
    assert not directory.exists()
    assert not (
        directory.parent / ".reference-acks" / f"{directory.name}.json"
    ).exists()
    assert not list(directory.parent.glob(".reference-budget/*/*.reservation"))


def test_dispatch_receipts_match_routed_ids_before_ack():
    from types import SimpleNamespace

    refs = [{"plan": {"sample_id": key}} for key in ("A", "B")]
    outcome = SimpleNamespace(attempted=True, collected=True, collected_samples=1)
    accepted, metrics = reference.committed_reference_metrics(
        outcome, {"A": (512, False)}, refs
    )
    assert accepted == {"A"}
    assert metrics["drafter/rollout_reference_skipped"] == 1
    assert metrics["drafter/rollout_committed_hidden_rows"] == 512
    with pytest.raises(RuntimeError, match="Partial rollout dispatch"):
        reference.committed_reference_metrics(
            outcome, {"A": (512, False), "B": (512, False)}, refs
        )


def test_known_window_size_charged_against_shared_budget(tmp_path, monkeypatch):
    from verl_speco.integration import vllm_rollout_capture as capture

    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    monkeypatch.setenv("SPECO_REFERENCE_SAMPLE_BYTES", "100")
    monkeypatch.setenv("SPECO_REFERENCE_MAX_BYTES", "120")
    budgets = [reference._reserve({}, window_bytes=40) for _ in range(3)]
    assert [b["bytes"] for b in budgets] == [40, 40, 40]
    assert reference._reserve({}, window_bytes=1) is None
    for budget in budgets:
        reference._release({"budget": budget})
    assert reference._reserve({}, window_bytes=101) is None


def test_rolling_prefetch_refills_after_owned_result_transfer():
    import threading

    second_started = threading.Event()

    def resolve(ref):
        if ref["id"] == 2:
            second_started.set()
        return ref["id"]

    pool = reference.ReferencePrefetcher(capacity=1, max_bytes=20, resolver=resolve)
    refs = [{"id": i, "budget": {"bytes": 10}} for i in (1, 2)]
    try:
        assert pool.submit(refs[0], defer=True)
        assert not pool.submit(refs[1], defer=True)
        assert not second_started.is_set()
        assert pool.take(refs[0]) == 1
        assert second_started.wait(5)
        assert pool._bytes == 20
        assert pool.take(refs[1]) == 2 and pool._bytes == 0
    finally:
        pool.close()


def test_ack_batch_stops_on_first_failed_source(monkeypatch):
    calls = []

    def ack(item):
        calls.append(item["id"])
        if item["id"] == 2:
            raise RuntimeError("disk failure")

    monkeypatch.setattr(reference, "acknowledge", ack)
    with pytest.raises(RuntimeError, match="disk failure"):
        reference.acknowledge_batch([{"global_step": 3, "id": i} for i in (1, 2, 3)])
    assert calls == [1, 2]


def test_first_producer_failure_reaches_consumer(window_reference):
    from verl_speco.producer.feature_export import FeatureExportQueue

    handle, directory = window_reference
    (directory / "live").touch()
    queue = FeatureExportQueue(workers=1)

    def fail(reason):
        raise RuntimeError(reason)

    for reason in ("field=token_ids position=7", "field=hidden_states position=8"):
        assert queue.reserve(1)
        queue.submit_reserved(
            directory,
            {"_window_plan": handle["plan"]},
            size=1,
            wait=lambda reason=reason: fail(reason),
        )
    queue.close()
    with pytest.raises(RuntimeError, match="field=token_ids position=7") as caught:
        reference.resolve(handle)
    assert handle["plan"]["sample_id"] in str(caught.value)
    assert "position=8" not in str(caught.value)
    assert not list(directory.glob("error-*.tmp"))


def test_reference_cleanup_preserves_primary_failure(caplog):
    from types import SimpleNamespace
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

    def fail():
        raise ValueError("secondary")

    trainer = SpecoV1Mixin()
    trainer._speco_reference_prefetcher = SimpleNamespace(close=fail)
    trainer._speco_reference_finalizer = SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError, match="primary"):
        try:
            raise RuntimeError("primary")
        except RuntimeError as error:
            trainer._speco_close_reference_workers(primary_error=error)
            raise
    assert "secondary" in caplog.text


def test_receipt_required_before_ack(window_reference):
    handle, directory = window_reference
    with pytest.raises(RuntimeError, match="missing collection receipt"):
        reference.acknowledge(handle)
    assert directory.is_dir()
    reference.discard(handle, preparation_stopped=True)
    assert (
        json.loads(reference._receipt_path(handle).read_text())["state"] == "CANCELLED"
    )
    assert not (directory.parent / ".reference-acks").exists()


def test_stopped_root_recovers_unpublished_window_and_orphan_ticket(window_reference):
    handle, directory = window_reference
    (directory / "reference.json").unlink()
    reference._reserve({})
    result = reference.recover_stopped_root(directory.parent, preparation_stopped=True)
    assert result["unpublished_windows"] == result["orphan_tickets"] == 1
    assert not directory.exists()
    assert not list(directory.parent.rglob("*.reservation"))
    assert (
        reference.recover_stopped_root(directory.parent, preparation_stopped=True)[
            "orphan_tickets"
        ]
        == 0
    )


def test_stopped_root_retains_unidentified_window_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    source = tmp_path / ("b" * 32)
    source.mkdir()
    reference._reserve({})
    result = reference.recover_stopped_root(tmp_path, preparation_stopped=True)
    assert result["unresolved"] == [source.name]
    assert list(tmp_path.rglob("*.reservation"))
