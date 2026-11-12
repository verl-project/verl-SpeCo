# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import pytest

from verl_speco.trainer.scheduler import StandaloneQueueState
from verl_speco.config import prefetch_depth
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader, TQPrefetchIterator


@pytest.mark.parametrize("depth,workers,defer,host,batch", [
    (2, 1, True, 300, 100), (1, 2, True, 300, 100),
    (1, 1, False, 300, 100), (1, 1, True, 200, 100),
])
def test_node_owner_rejects_unsafe_prefetch_configuration(depth, workers, defer, host, batch):
    from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator
    loader = SimpleNamespace(prepare_batch=lambda entries: None,
                             store=SimpleNamespace(uses_transfer_queue=False), node_ranks_per_host=2)
    with pytest.raises(ValueError, match="Node reference owner"):
        ReferenceBatchPrefetchIterator(loader, depth, workers=workers, defer_cleanup=defer,
                                       max_host_bytes=host, max_batch_host_bytes=batch,
                                       node_ranks_per_host=2)


def test_node_receiver_enforces_materialized_and_pinned_budget(monkeypatch):
    import torch
    from verl_speco.trainer import reference_batch_prefetch as prefetch
    from verl_speco.trainer import reference_shared_frame as frame
    monkeypatch.setattr(prefetch, "_MAX_BATCH_HOST_BYTES", 100)
    monkeypatch.setattr(frame, "receive_node_batch", lambda **kwargs: ({"x": torch.zeros(20)}, 0.0))
    with pytest.raises(ValueError, match="consumer_prefetch_batch_bytes"):
        prefetch._node_frame_batch(None, binding={}, owner_rank=0, max_bytes=100, timeout=1)


def test_node_owner_rejects_early_source_cleanup(monkeypatch):
    from verl_speco.trainer import reference_batch_prefetch as prefetch
    monkeypatch.setattr(prefetch, "_CLEANUP_AFTER_LOAD", True)
    with pytest.raises(RuntimeError, match="cannot release payloads"):
        prefetch._node_frame_batch({0: ["sample"]}, binding={}, owner_rank=0, max_bytes=100, timeout=1)


@pytest.mark.parametrize("depth,ready_wait", [(0, 0.4), (1, 0.001)])
@pytest.mark.parametrize("file_path", [False, True])
def test_batch_timing_records_serial_and_prefetch_wait(depth, ready_wait, file_path):
    import ast
    from pathlib import Path

    from verl_speco.trainer.standalone_tracking import _select_standalone_tracking_metrics

    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_run_standalone_draft_training_async")
    # Exercise the real metrics block for both paths without the model/Ray stack.
    metrics_block = next(
        n for n in ast.walk(loop) if isinstance(n, ast.If)
        and any(isinstance(child, ast.Constant) and child.value == "perf/loader_plan_wait_time"
                for child in ast.walk(n))
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name) and n.test.left.id == "tq_local_batch"
    )
    metrics = {"dspark/loss": 1.0, "perf/step_time": ready_wait + 0.8,
               "perf/train_time": 0.65, "perf/batch_prepare_time": 0.08,
               "perf/batch_finalize_time": 0.06}
    namespace = dict(step_metrics=metrics, tq_local_batch=SimpleNamespace(
                         command_wait_seconds=10, prepared_batch={} if file_path else None,
                         batch_prepare_seconds=0.3, delivery_times={"command_delivery_time": 0.02}),
                     tq_prefetch=object() if depth else None,
                     batch_ready_wait_seconds=ready_wait, tq_get_seconds=0.3,
                     fetch_timings={"local_batch_wait_time": ready_wait * 0.75,
                                    "rank_ready_sync_time": ready_wait * 0.25})
    exec(compile(ast.Module(body=[metrics_block], type_ignores=[]), str(path), "exec"), namespace)
    selected = _select_standalone_tracking_metrics(metrics)
    assert selected["perf/batch_ready_wait_time"] == ready_wait
    assert selected["perf/loader_plan_wait_time"] == 10
    if file_path:
        assert selected["perf/loader_batch_prepare_time"] == 0.3
        assert "perf/loader_tq_get_time" not in selected
    else:
        assert selected["perf/loader_tq_get_time"] == 0.3
    assert selected["perf/other_time"] == pytest.approx(0.01)


@pytest.mark.parametrize("training_failed", [False, True])
@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("checkpoint", ["none", "callback", "save"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_prefetch_close_failure_still_cleans_resources(training_failed, running, checkpoint, close_fails):
    # Execute the actual finally block without importing the unavailable Ray /
    # verl model stack. Resource ordering and exception identity matter here.
    import ast
    import asyncio
    import concurrent.futures
    import logging
    import sys
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_run_standalone_draft_training_async")
    cleanup = next(n.finalbody for n in loop.body if isinstance(n, ast.Try) and n.finalbody)
    wrapper = ast.parse("async def exercise():\n    checkpoint_directory_lease = None\n    try:\n        if training_failed:\n"
                        "            raise training_error\n    finally:\n        pass\n")
    wrapper.body[0].body[1].finalbody = cleanup
    calls = []
    training_error = ValueError("training failed")
    close_error = RuntimeError("prefetch shutdown deadline")

    def close():
        calls.append("prefetch")
        if close_fails:
            raise close_error

    async def trainer_cleanup(**kwargs):
        calls.append("trainer")

    checkpoint_event = threading.Event() if checkpoint == "callback" else None
    checkpoint_future = concurrent.futures.Future() if checkpoint == "save" else None
    namespace = dict(
        sys=sys, asyncio=asyncio, time=time, logger=logging.getLogger(__name__), training_failed=training_failed,
        training_error=training_error, standalone_trackers=[], rank=0,
        current_stage="training_step", attempted_batches=1, successful_steps=0,
        optimizer_step=0, feature_producer=None, feature_replayer=None, device_prefetch=None,
        tq_prefetch=SimpleNamespace(close=close, has_running_work=running), tq_prefetch_group=object(), batch_finalizer=None,
        feature_store_type="tq",
        store=SimpleNamespace(close=lambda: calls.append("store")),
        trainer=SimpleNamespace(cleanup_training=trainer_cleanup, runtime_device="cpu",
                                _standalone_checkpoint_timeout=0.02,
                                _standalone_finalization_event=checkpoint_event,
                                _pending_full_checkpoint_future=checkpoint_future),
        dist=SimpleNamespace(destroy_process_group=lambda group: calls.append("group"),
                             is_initialized=lambda: False),
        _finish_standalone_tracking=lambda *a, **k: None,
        _sync_standalone_export_error=lambda *a, **k: False,
    )
    exec(compile(ast.fix_missing_locations(wrapper), str(path), "exec"), namespace)
    if training_failed or close_fails or running or checkpoint != "none":
        error_type = type(training_error) if training_failed else RuntimeError if close_fails or running else TimeoutError
        with pytest.raises(error_type) as caught:
            asyncio.run(namespace["exercise"]())
        if training_failed or close_fails:
            assert caught.value is (training_error if training_failed else close_error)
    else:
        asyncio.run(namespace["exercise"]())
    expected = ["prefetch"] if running else ["prefetch", "group", "store"]
    if not running and checkpoint == "none":
        expected.append("trainer")
    assert calls == expected
    if checkpoint_future is not None:
        assert not checkpoint_future.cancelled()


def test_prefetch_close_bounds_orphaned_executor_reads():
    import time
    from threading import Event
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator
    class Loader:
        rank, world_size, prepare_batch = 0, 1, lambda *args, **kwargs: None
        def __iter__(self): return iter(())
    loader = Loader()
    original = loader.prepare_batch
    source = TQPrefetchIterator(loader, 1, pin_memory=True, prepare_workers=1, close_timeout=0.1)
    started, release = Event(), Event()
    def read():
        started.set()
        assert release.wait(5)
    source._executor.submit(read)
    assert started.wait(5)
    try:
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="shutdown deadline"):
            source.close()
        assert time.monotonic() - before < 2 and source.has_running_work
        assert loader.prepare_batch is not original
    finally:
        release.set()
        source.close()
    assert not source.has_running_work and loader.prepare_batch == original


def test_prefetch_close_retains_blocked_coordinator_until_it_exits():
    from threading import Event
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator
    entered, release = Event(), Event()
    class Loader:
        rank, world_size, prepare_batch = 0, 1, None
        def __iter__(self):
            entered.set()
            assert release.wait(5)
            return iter(())
    source = TQPrefetchIterator(Loader(), 1, close_timeout=0.1)
    assert entered.wait(5)
    try:
        with pytest.raises(RuntimeError, match="shutdown deadline"):
            source.close()
        assert source.has_running_work
    finally:
        release.set()
        source.close()
    assert not source.has_running_work


def test_prefetch_iterator_creation_failure_reaches_consumer():
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator
    class Loader:
        rank, world_size, prepare_batch = 0, 1, None
        def __iter__(self): raise ValueError("source initialization failed")
    source = TQPrefetchIterator(Loader(), 1, close_timeout=1)
    try:
        with pytest.raises(RuntimeError, match="source initialization failed") as error:
            next(source)
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        source.close()


def test_process_prefetch_pool_shutdown_respects_deadline(monkeypatch):
    from collections import deque
    from threading import Condition, Event
    from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator

    release, finished = Event(), Event()
    class Pool:
        _pool = [SimpleNamespace(is_alive=lambda: not finished.is_set())]
        def terminate(self):
            assert release.wait(5)
        def join(self):
            finished.set()
    stopped = SimpleNamespace(is_alive=lambda: False, join=lambda **kwargs: None)
    source = ReferenceBatchPrefetchIterator.__new__(ReferenceBatchPrefetchIterator)
    source.close_timeout, source.stop, source._condition = 0.1, Event(), Condition()
    source._thread, source._pin_thread, source._executor = stopped, None, None
    source._collector, source._pool, source._pending = stopped, Pool(), deque()
    source._pool_shutdown_thread, source._pool_shutdown_error = None, None
    monkeypatch.setattr(TQPrefetchIterator, "close", lambda self, **kwargs: None)
    try:
        with pytest.raises(RuntimeError, match="shutdown deadline"):
            source.close()
        assert source.has_running_work
    finally:
        release.set()
        assert finished.wait(5)
    source.close()
    assert not source.has_running_work


def command(index):
    key = f"k{index}"
    return {"kind": "batch", "plan_id": index, "global_keys": [key],
            "global_sequence_nos": [index],
            "assignments": [[{"key": key, "tag": {"sequence_no": index}}]]}


class Store:
    def __init__(self, fail=False):
        self.calls = []
        self.fetched = threading.Condition()
        self.fail = fail

    def connect(self):
        pass

    def owner_ready(self):
        return True

    def get_many(self, entries):
        with self.fetched:
            self.calls.append([e.key for e in entries])
            self.fetched.notify_all()
        if self.fail:
            self.fail = False
            raise ConnectionError("temporary disconnect")
        return [SimpleNamespace(input_ids=[1])]

    def wait_calls(self, n):
        with self.fetched:
            assert self.fetched.wait_for(lambda: len(self.calls) >= n, timeout=3)


def test_scheduled_reference_batch_carries_input_identity():
    import torch
    from verl_speco.trainer.draft_training_loop import _next_batch_across_ranks
    identity = {"kind": "file", "sha256": "a" * 64, "size_bytes": 4}
    scheduled = Queue()
    item = command(0)
    item.update(receipt_token="token", input_identity=identity)
    scheduled.put(item)
    loader = TQFeatureDataLoader(Store(), batch_size=1, rank=0, world_size=1,
        scheduled_commands=scheduled, prepare_batch=lambda entries: {"input_ids": torch.ones(1)})
    result = _next_batch_across_ranks(iter(loader), rank=0, device=torch.device("cpu"), require_prepared=True)
    assert result.input_identity == identity


def make_prefetch(depth, store=None, commands=None):
    store = store or Store()
    commands = commands or Queue()
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1,
                                 scheduled_commands=commands)
    return TQPrefetchIterator(loader, depth, close_timeout=3), store, commands


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_prefetch_is_bounded_fifo_and_overlaps_training(depth):
    commands = Queue()
    for index in range(8):
        commands.put(command(index))
    commands.put({"kind": "stop"})
    source, store, _ = make_prefetch(depth, commands=commands)
    try:
        store.wait_calls(depth)
        with source._condition:
            assert source._condition.wait_for(lambda: len(source._batches) == depth, timeout=3)
            assert len(store.calls) == depth
        assert next(source).plan_id == 0
        # Simulate main thread training: it does not call next again, yet the
        # worker fills the newly freed slot. No timing-based speed assertion.
        store.wait_calls(depth + 1)
        assert [batch.plan_id for batch in source] == list(range(1, 8))
        assert store.calls == [[f"k{i}"] for i in range(8)]
    finally:
        source.close()


def test_close_unblocks_empty_command_queue():
    source, _, _ = make_prefetch(1)
    source.close()
    assert not source._thread.is_alive()


def test_empty_polls_do_not_end_iterator_or_fetch_tq():
    from queue import Empty

    class DelayedCommands:
        def __init__(self):
            self.calls = 0

        def get(self, **kwargs):
            self.calls += 1
            if self.calls <= 4:
                raise Empty
            return command(0) if self.calls == 5 else {"kind": "stop"}

    source, store, _ = make_prefetch(1, commands=DelayedCommands())
    try:
        assert [batch.plan_id for batch in source] == [0]
        assert store.calls == [["k0"]]
    finally:
        source.close()


def test_close_unblocks_full_payload_cache():
    commands = Queue()
    commands.put(command(0))
    commands.put(command(1))
    source, store, _ = make_prefetch(1, commands=commands)
    store.wait_calls(1)
    source.close()
    assert len(store.calls) == 1


def test_get_retry_and_duplicate_plan_do_not_fetch_success_twice(monkeypatch):
    monkeypatch.setattr("verl_speco.trainer.tq_retry.random.uniform", lambda *a: 0)
    commands = Queue()
    for value in (command(0), command(0), command(1), {"kind": "stop"}):
        commands.put(value)
    source, store, _ = make_prefetch(1, store=Store(fail=True), commands=commands)
    try:
        assert [batch.plan_id for batch in source] == [0, 1]
        assert store.calls == [["k0"], ["k0"], ["k1"]]
    finally:
        source.close()


def test_permanent_get_error_surfaces_without_clearing():
    store = Store()
    store.get_many = lambda entries: (_ for _ in ()).throw(ValueError("bad data"))
    commands = Queue()
    commands.put(command(0))
    source, _, _ = make_prefetch(1, store=store, commands=commands)
    try:
        with pytest.raises(RuntimeError, match="prefetch failed") as error:
            next(source)
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        source.close()


def entries(n):
    return [SimpleNamespace(key=f"k{i}", tag={"sequence_no": i}) for i in range(n)]


def test_plan_reservations_survive_refresh_and_completion_is_idempotent():
    plans = StandaloneQueueState(2, 2)
    plans.refresh(entries(6))
    first, _ = plans.reserve()
    second, _ = plans.reserve()
    assert not plans.has_capacity
    plans.refresh(entries(6))
    assert [e.key for e in plans.ready] == ["k4", "k5"]
    with pytest.raises(RuntimeError, match="FIFO"):
        plans.completion(second.plan_id, second.keys)
    assert plans.completion(first.plan_id, first.keys) is first
    plans.release(first)
    assert plans.completion(first.plan_id, first.keys) is None
    with pytest.raises(RuntimeError, match="Conflicting"):
        plans.completion(first.plan_id, ["wrong"])
    assert plans.has_capacity
    third, _ = plans.reserve()
    assert third.keys == ("k4", "k5")


def test_retry_inspects_causes_but_not_arbitrary_errors(monkeypatch):
    monkeypatch.setattr("verl_speco.trainer.tq_retry.time.sleep", lambda _: None)
    calls = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            try:
                raise TimeoutError("RPC timed out")
            except TimeoutError as error:
                raise RuntimeError("TQ wrapper") from error
        return 42

    assert retry_io(operation, name="test") == 42
    assert len(calls) == 3
    with pytest.raises(ValueError):
        retry_io(lambda: (_ for _ in ()).throw(ValueError("invalid sample")), name="test")


def test_shared_retry_preserves_clear_attempts_and_backoff(monkeypatch):
    delays = []
    monkeypatch.setattr("verl_speco.trainer.tq_retry.time.sleep", delays.append)
    calls = []

    def clear():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("clear failed")

    retry_io(clear, name="TQ clear", attempts=3, initial_backoff=0.5,
             jitter=False, retry_if=lambda error: True)
    assert len(calls) == 3
    assert delays == [0.5, 1.0]


def test_shared_retry_stops_at_attempt_limit(monkeypatch):
    delays = []
    monkeypatch.setattr("verl_speco.trainer.tq_retry.time.sleep", delays.append)
    calls = []

    def clear():
        calls.append(1)
        raise RuntimeError("clear failed")

    with pytest.raises(RuntimeError, match="clear failed"):
        retry_io(clear, name="TQ clear", attempts=3, initial_backoff=0.5,
                 jitter=False, retry_if=lambda error: True)
    assert len(calls) == 3
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_prefetch_config_rejects_invalid_depth(value):
    with pytest.raises(ValueError):
        prefetch_depth({"consumer_prefetch_batches": value})


def test_file_pin_thread_overlaps_next_prepare_and_keeps_one_bounded_fifo(monkeypatch):
    import torch

    first_pin = threading.Event()
    second_prepared = threading.Event()
    prepared = []
    pinned_on = []
    commands = Queue()
    for index in range(4):
        commands.put(command(index))
    commands.put({"kind": "stop"})

    def prepare(entries, *, pin_memory, executor):
        assert pin_memory is False and executor is not None
        index = entries[0].tag["sequence_no"]
        prepared.append(index)
        if index == 1:
            second_prepared.set()
        return {"hidden_states": torch.tensor([index]), "hidden_source": "standalone_vllm_reference"}

    def pin(tensor):
        pinned_on.append(threading.current_thread().name)
        if tensor.item() == 0:
            first_pin.set()
            assert second_prepared.wait(3), "Pinning blocked the prepare coordinator"
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    loader = TQFeatureDataLoader(Store(), batch_size=1, rank=0, world_size=1,
                                 scheduled_commands=commands, prepare_batch=prepare)
    source = TQPrefetchIterator(loader, 2, pin_memory=True, prepare_workers=2, close_timeout=3)
    try:
        assert first_pin.wait(3)
        assert second_prepared.wait(3)
        with source._condition:
            assert source._condition.wait_for(lambda: len(source._batches) == 2, timeout=3)
            assert prepared == [0, 1] and source._reserved == 2
        assert [batch.prepared_batch["hidden_states"].item() for batch in source] == [0, 1, 2, 3]
        assert all(name == "file-pin-0" for name in pinned_on)
    finally:
        source.close()
    assert not source._thread.is_alive() and not source._pin_thread.is_alive()
    assert loader.prepare_batch is prepare


@pytest.mark.parametrize("fail_pin", [False, True])
def test_file_pipeline_closes_when_full_or_pin_fails(monkeypatch, fail_pin):
    import torch

    commands = Queue()
    commands.put(command(0))
    commands.put(command(1))
    prepared = []

    def prepare(entries, **kwargs):
        prepared.append(entries[0].key)
        return {"hidden_states": torch.ones(2)}

    def pin(tensor):
        if fail_pin:
            raise RuntimeError("injected pin failure")
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    loader = TQFeatureDataLoader(Store(), batch_size=1, rank=0, world_size=1,
                                 scheduled_commands=commands, prepare_batch=prepare)
    source = TQPrefetchIterator(loader, 1, pin_memory=True, prepare_workers=0, close_timeout=3)
    try:
        if fail_pin:
            with pytest.raises(RuntimeError, match="injected pin failure"):
                next(source)
        else:
            with source._condition:
                assert source._condition.wait_for(lambda: len(source._batches) == 1, timeout=3)
                assert prepared == ["k0"]
    finally:
        source.close()
    assert not source._thread.is_alive() and not source._pin_thread.is_alive()


def test_file_worker_failure_surfaces_before_training_and_pool_closes(monkeypatch):
    import torch

    commands = Queue()
    commands.put(command(0))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)

    def fail(entry):
        raise ValueError("injected file worker failure")

    def prepare(entries, *, executor, **kwargs):
        return list(executor.map(fail, entries))

    loader = TQFeatureDataLoader(Store(), batch_size=1, rank=0, world_size=1,
                                 scheduled_commands=commands, prepare_batch=prepare)
    source = TQPrefetchIterator(loader, 2, pin_memory=True, prepare_workers=2, close_timeout=3)
    try:
        with pytest.raises(RuntimeError, match="prefetch failed") as caught:
            next(source)
        assert isinstance(caught.value.__cause__, ValueError)
    finally:
        source.close()
    assert not source._thread.is_alive() and not source._pin_thread.is_alive()


def test_pin_failure_keeps_rank_control_collectives_ordered_and_aborts_training(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import torch

    collective = threading.Barrier(2, timeout=5)
    slots = [None, None]

    def exchange(rank, kind, value):
        slots[rank] = (kind, value)
        collective.wait()
        assert slots[0][0] == slots[1][0], "Background rank collective order diverged"
        if kind == "broadcast":
            result = slots[0][1]
        else:
            result = [max(a, b) for a, b in zip(slots[0][1], slots[1][1])]
        collective.wait()
        return result

    class Loader(TQFeatureDataLoader):
        def _broadcast_command(self, command):
            return exchange(self.rank, "broadcast", command)

    class Source(TQPrefetchIterator):
        def _sync(self, values):
            return exchange(self.loader.rank, "reduce", values)

    def pin(tensor):
        if threading.current_thread().name == "file-pin-0":
            raise RuntimeError("injected rank-zero pin failure")
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    commands = Queue()
    commands.put({"kind": "batch", "plan_id": 0, "global_keys": ["k0", "k1"],
                  "global_sequence_nos": [0, 1],
                  "assignments": [[{"key": f"k{rank}", "tag": {"sequence_no": rank}}] for rank in range(2)]})
    commands.put({"kind": "stop"})
    sources = []
    for rank in range(2):
        loader = Loader(Store(), batch_size=1, rank=rank, world_size=2,
                        scheduled_commands=commands,
                        prepare_batch=lambda entries, **kwargs: {"hidden_states": torch.ones(2)})
        sources.append(Source(loader, 2, pin_memory=True, prepare_workers=0, close_timeout=6))

    agreement = threading.Barrier(2, timeout=5)
    failed = [False, False]
    trained = []

    def consumer(rank):
        try:
            try:
                next(sources[rank])
            except (RuntimeError, StopIteration):
                failed[rank] = True
            agreement.wait()  # Existing main-thread agreement precedes forward.
            if not any(failed):
                trained.append(rank)
        finally:
            sources[rank].close()

    with ThreadPoolExecutor(max_workers=2) as consumers:
        list(consumers.map(consumer, range(2)))
    assert failed[0] and trained == []
    assert all(not source._thread.is_alive() and not source._pin_thread.is_alive() for source in sources)


def test_completion_history_rejects_conflicts_and_expired_replays():
    plans = StandaloneQueueState(1, 1)
    plans._completion_history_limit = 2
    plans.refresh(entries(3))
    completed = []
    for _ in range(3):
        pending, _ = plans.reserve()
        assert plans.completion(pending.plan_id, pending.keys) is pending
        plans.release(pending)
        completed.append(pending)
    assert plans.completion(completed[1].plan_id, completed[1].keys) is None
    with pytest.raises(RuntimeError, match="Conflicting"):
        plans.completion(completed[1].plan_id, ["wrong"])
    with pytest.raises(RuntimeError, match="Expired"):
        plans.completion(completed[0].plan_id, completed[0].keys)
    with pytest.raises(RuntimeError, match="Invalid"):
        plans.completion(True, completed[2].keys)


def test_conflicting_duplicate_plan_is_rejected_before_fetch():
    commands = Queue()
    commands.put(command(0))
    altered = command(0)
    altered["global_sequence_nos"] = [99]
    commands.put(altered)
    commands.put({"kind": "stop"})
    source, store, _ = make_prefetch(1, commands=commands)
    try:
        assert next(source).plan_id == 0
        with pytest.raises(RuntimeError, match="prefetch failed") as error:
            next(source)
        assert "Conflicting duplicate" in str(error.value.__cause__)
        assert store.calls == [["k0"]]
    finally:
        source.close()


def test_receipt_token_fences_other_driver_incarnations():
    plans = StandaloneQueueState(1, 1)
    plans.refresh(entries(1))
    pending, _ = plans.reserve()
    pending.receipt_token = "current-dispatch"
    for stale in (None, "previous-dispatch"):
        with pytest.raises(RuntimeError, match="FIFO plan"):
            plans.completion(pending.plan_id, pending.keys, receipt_token=stale)
    assert plans.completion(pending.plan_id, pending.keys, receipt_token="current-dispatch") is pending
    plans.release(pending)
    assert plans.completion(pending.plan_id, pending.keys, receipt_token="current-dispatch") is None
    with pytest.raises(RuntimeError, match="Conflicting"):
        plans.completion(pending.plan_id, pending.keys, receipt_token="previous-dispatch")


def test_consumer_preserves_dispatch_receipt_token():
    commands = Queue()
    message = command(0)
    message["receipt_token"] = "dispatch-identity"
    commands.put(message)
    commands.put({"kind": "stop"})
    source, _, _ = make_prefetch(1, commands=commands)
    try:
        assert next(source).receipt_token == "dispatch-identity"
    finally:
        source.close()


def test_reference_command_frames_reassemble_and_retry_without_duplicates():
    from verl_speco.standalone_ray_runtime import _reference_command_frames
    from verl_speco.trainer.tq_sample_source import _ReferenceCommandAssembler

    value = {**command(0), "receipt_token": "identity", "large": "中" * 800}
    frames = list(_reference_command_frames(value, max_bytes=8192, frame_bytes=1024))
    assert len(frames) > 2 and all(len(frame["data"]) <= 1024 for frame in frames)
    assembler = _ReferenceCommandAssembler(8192, 1024)
    assert assembler.add(frames[0]) is None
    assert assembler.add(frames[0]) is None
    for frame in frames[1:-1]:
        assert assembler.add(frame) is None
    assert assembler.add(frames[-1]) == value
    assert assembler.add(frames[-1]) is None
    conflicting = {**frames[0], "data": b"x" * len(frames[0]["data"])}
    with pytest.raises(ValueError, match="Conflicting completed"):
        assembler.add(conflicting)


@pytest.mark.parametrize("failure", ["out_of_order", "identity", "checksum", "truncated"])
def test_reference_command_frames_reject_corruption(failure):
    from verl_speco.standalone_ray_runtime import _reference_command_frames
    from verl_speco.trainer.tq_sample_source import _ReferenceCommandAssembler

    value = {**command(0), "receipt_token": "identity", "large": "x" * 3000}
    frames = list(_reference_command_frames(value, max_bytes=8192, frame_bytes=1024))
    assembler = _ReferenceCommandAssembler(8192, 1024)
    if failure == "out_of_order":
        with pytest.raises(ValueError, match="out-of-order"):
            assembler.add(frames[1])
    elif failure == "identity":
        assembler.add(frames[0])
        with pytest.raises(ValueError, match="conflicting identities"):
            assembler.add({**frames[1], "receipt_token": "different"})
    elif failure == "truncated":
        with pytest.raises(ValueError, match="bounds"):
            assembler.add({**frames[0], "data": frames[0]["data"][:-1]})
    else:
        modified = {**frames[0], "data": b"x" * len(frames[0]["data"])}
        assembler.add(modified)
        with pytest.raises(ValueError, match="checksum mismatch"):
            for frame in frames[1:]:
                assembler.add(frame)


def test_fragmented_command_fetches_each_sample_once():
    from verl_speco.standalone_ray_runtime import _reference_command_frames

    first = {**command(0), "receipt_token": "identity", "large": "x" * 3000}
    frames = list(_reference_command_frames(first, max_bytes=8192, frame_bytes=1024))
    commands = Queue()
    for frame in [frames[0], frames[0], *frames[1:], frames[-1], command(1), {"kind": "stop"}]:
        commands.put(frame)
    store = Store()
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1,
                                scheduled_commands=commands, reference_command_max_bytes=8192,
                                reference_command_frame_bytes=1024)
    assert [batch.plan_id for batch in loader] == [0, 1]
    assert store.calls == [["k0"], ["k1"]]


def test_incomplete_fragmented_command_is_never_fetched():
    from verl_speco.standalone_ray_runtime import _reference_command_frames

    first = {**command(0), "receipt_token": "identity", "large": "x" * 3000}
    frames = list(_reference_command_frames(first, max_bytes=8192, frame_bytes=1024))
    commands = Queue()
    commands.put(frames[0])
    commands.put({"kind": "stop"})
    store = Store()
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1,
                                scheduled_commands=commands, reference_command_max_bytes=8192,
                                reference_command_frame_bytes=1024)
    with pytest.raises(RuntimeError, match="interrupted before all frames"):
        next(iter(loader))
    assert store.calls == []


def test_reference_command_frame_payload_must_match_header_identity():
    from verl_speco.standalone_ray_runtime import _reference_command_frames
    from verl_speco.trainer.tq_sample_source import _ReferenceCommandAssembler

    command_value = {**command(0), "receipt_token": "original", "large": "x" * 3000}
    frames = list(_reference_command_frames(command_value, max_bytes=8192, frame_bytes=1024))
    forged = [{**frame, "receipt_token": "forged"} for frame in frames]
    assembler = _ReferenceCommandAssembler(8192, 1024)
    with pytest.raises(ValueError, match="payload identity differs"):
        for frame in forged:
            assembler.add(frame)


def test_reference_command_total_limit_remains_enforced():
    from verl_speco.standalone_ray_runtime import _reference_command_frames

    value = {**command(0), "receipt_token": "identity", "large": "x" * 9000}
    with pytest.raises(ValueError, match="control metadata exceeds"):
        list(_reference_command_frames(value, max_bytes=8192, frame_bytes=1024))


def test_rank_ack_rejects_missing_producer_handle_before_sending():
    import torch
    from verl_speco.trainer.draft_training_loop import _ack_reference_materialization_across_ranks
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    events = Queue()
    batch = TQLocalBatch(local_keys=[], local_samples=[], global_keys=None, plan_id=0,
                         receipt_token="token", producer_incarnation="f" * 32)
    with pytest.raises(RuntimeError, match="acknowledgement failed") as error:
        _ack_reference_materialization_across_ranks(batch, rank=0, device=torch.device("cpu"),
                                                    events=events, config={}, per_rank=True)
    assert isinstance(error.value.__cause__, ValueError) and events.empty()


def test_plan_requires_all_rank_materialization_receipts():
    from verl_speco.trainer.scheduler.drafter_runtime_state import StandaloneQueueState
    state = StandaloneQueueState(capacity=1, batch_size=1)
    state.refresh([SimpleNamespace(key="a", tag={"sequence_no": 0})])
    pending, _ = state.reserve()
    pending.receipt_token = "token"
    assert state.mark_rank_materialized(0, "token", 0, 2)
    assert not state.mark_rank_materialized(0, "token", 0, 2)
    with pytest.raises(RuntimeError, match="every rank"):
        state.mark_materialized(0, "token", required_ranks=2)
    with pytest.raises(RuntimeError, match="Invalid rank"):
        state.mark_rank_materialized(0, "token", 2, 2)
    with pytest.raises(RuntimeError, match="does not match"):
        state.mark_rank_materialized(0, "wrong", 1, 2)
    assert state.mark_rank_materialized(0, "token", 1, 2)
    assert state.mark_materialized(0, "token", required_ranks=2)
    state.release(pending)
    assert not state.mark_rank_materialized(0, "token", 1, 2)


def test_materialization_acknowledgement_is_pending_idempotent_and_generation_bound():
    plans = StandaloneQueueState(2, 1)
    plans.refresh(entries(2))
    first, _ = plans.reserve()
    second, _ = plans.reserve()
    first.receipt_token, second.receipt_token = "first", "second"
    # Prefetch and asynchronous cleanup allow the next ready ACK before the
    # previous completion event reaches the Driver.
    with pytest.raises(RuntimeError, match="before materialization"):
        plans.mark_applied(first.plan_id, "first")
    assert plans.mark_materialized(second.plan_id, "second") is True
    with pytest.raises(RuntimeError, match="pending plan"):
        plans.mark_materialized(second.plan_id + 1, "future")
    with pytest.raises(RuntimeError, match="pending plan"):
        plans.mark_materialized(first.plan_id, "stale")
    assert plans.mark_materialized(first.plan_id, "first") is True
    assert plans.mark_materialized(first.plan_id, "first") is False
    assert plans.mark_applied(first.plan_id, "first") is True
    assert plans.mark_applied(first.plan_id, "first") is False
    with pytest.raises(RuntimeError, match="pending plan"):
        plans.mark_applied(second.plan_id, "stale")
    plans.release(first)
    assert plans.mark_applied(first.plan_id, "first") is False
    with pytest.raises(RuntimeError, match="conflicting"):
        plans.mark_applied(first.plan_id, "stale")
    assert plans.mark_materialized(first.plan_id, "first") is False
    with pytest.raises(RuntimeError, match="conflicting"):
        plans.mark_materialized(first.plan_id, "stale")
    assert plans.mark_materialized(second.plan_id, "second") is False


def test_reference_queue_metrics_follow_all_three_stages():
    from verl_speco.standalone_ray_runtime import _reference_queue_metrics
    plans = StandaloneQueueState(3, 1)
    plans.refresh(entries(4))
    first, _ = plans.reserve()
    second, _ = plans.reserve()
    third, _ = plans.reserve()
    for plan in (first, second, third):
        plan.receipt_token = str(plan.plan_id)
    plans.mark_materialized(first.plan_id, "0")
    plans.mark_applied(first.plan_id, "0")
    plans.mark_materialized(second.plan_id, "1")
    values = _reference_queue_metrics(plans, completed_samples=6, elapsed=2)
    assert values == {
        "scheduler/ready_samples": 1.0, "scheduler/reserved_samples": 3.0,
        "scheduler/pending_plans": 3.0, "scheduler/awaiting_materialization_plans": 1.0,
        "scheduler/awaiting_optimizer_plans": 1.0, "scheduler/awaiting_cleanup_plans": 1.0,
        "scheduler/completed_samples": 6.0, "scheduler/completed_samples_per_second": 3.0,
    }
    plans.release(first)
    values = _reference_queue_metrics(plans, completed_samples=7, elapsed=4)
    assert values["scheduler/awaiting_cleanup_plans"] == 0.0
    assert values["scheduler/completed_samples_per_second"] == 1.75


def _host_budget_gloo_worker(rank, port, result_queue, init_failure):
    import datetime
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=25))
    torch.Tensor.pin_memory = lambda tensor: tensor
    class Loader:
        world_size = 2
        process_group = dist.group.WORLD
        cleanup_after_load = False
        prepare_batch = staticmethod(lambda *args, **kwargs: None)
        def __init__(self):
            self.rank = rank
        def __iter__(self):
            if init_failure and rank == 1:
                raise ValueError("source initialization failed")
            return iter([SimpleNamespace(prepared_batch={"x": torch.zeros(4 if rank == 0 else 64)}, delivery_times={})])
    source = TQPrefetchIterator(Loader(), 2, pin_memory=True, prepare_workers=0,
                                max_host_bytes=128, max_batch_host_bytes=128, close_timeout=10)
    try:
        try:
            next(source)
            result_queue.put((rank, "unexpected_batch"))
        except RuntimeError as exc:
            result_queue.put((rank, str(exc), type(exc.__cause__).__name__))
    finally:
        source.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("init_failure", [False, True])
def test_host_budget_failure_stops_both_gloo_ranks(init_failure):
    import multiprocessing
    import socket
    import torch.distributed as dist
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_host_budget_gloo_worker, args=(rank, port, queue, init_failure)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=45) for _ in processes]
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        assert {item[0] for item in results} == {0, 1}
        message = "source initialization failed" if init_failure else "Consumer prefetch failed"
        assert all(message in item[1] for item in results)
        assert next(item for item in results if item[0] == 1)[2] == "ValueError"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _rank_materialization_gloo_worker(rank, port, fail_rank, events, results):
    import datetime
    import torch
    import torch.distributed as dist
    import ray
    from types import SimpleNamespace
    from verl_speco.trainer.draft_training_loop import _ack_reference_materialization_across_ranks
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=25))
    acknowledged = []
    def acknowledge(incarnation, plan_id, token, owner_rank):
        assert (incarnation, plan_id, token, owner_rank) == ("f" * 32, 0, "token", rank)
        if fail_rank == "source" and rank == 1:
            raise RuntimeError("injected Producer acknowledgement failure")
        acknowledged.append(owner_rank)
        return "ack"
    ray.get = lambda ref, timeout=None: ref
    producer = SimpleNamespace(acknowledge_reference_rank=SimpleNamespace(remote=acknowledge))
    class EventQueue:
        def put(self, event):
            if fail_rank == "event" and rank == 1:
                raise OSError("injected rank event failure")
            assert acknowledged == [rank]
            events.put(event)
    batch = TQLocalBatch(local_keys=[], local_samples=[], global_keys=None, plan_id=0, receipt_token="token",
                         producer_incarnation="f" * 32)
    try:
        try:
            _ack_reference_materialization_across_ranks(batch, rank=rank, device=torch.device("cpu"),
                                                        events=EventQueue(), config={"io_retry_max_attempts": 1}, per_rank=True,
                                                        producer_handle=producer)
            results.put((rank, "ok"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_rank", ["none", "event", "source"])
def test_per_rank_materialization_gloo_confirmation_and_failure(fail_rank):
    import multiprocessing
    import socket
    import torch.distributed as dist
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    ctx = multiprocessing.get_context("spawn")
    events, results = ctx.Queue(), ctx.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [ctx.Process(target=_rank_materialization_gloo_worker,
                             args=(rank, port, fail_rank, events, results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        rows = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        if fail_rank != "none":
            assert all("acknowledgement failed" in row[1] for row in rows)
            received = [events.get(timeout=5)]
            assert received[0]["kind"] == "reference_rank_materialized" and received[0]["rank"] == 0
        else:
            assert all(row[1] == "ok" for row in rows)
            received = [events.get(timeout=5) for _ in range(3)]
            assert {event["rank"] for event in received if event["kind"] == "reference_rank_materialized"} == {0, 1}
            assert sum(event["kind"] == "training_materialized" for event in received) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _fake_device_backend():
    from contextlib import nullcontext
    state = SimpleNamespace(copies=0, synchronized=threading.Event())
    class Stream:
        def synchronize(self):
            state.copies += 1
            if state.copies >= 2:
                state.synchronized.set()
    return SimpleNamespace(set_device=lambda index: None, Stream=Stream,
                           stream=lambda stream: nullcontext(), state=state)


def test_device_prefetch_preserves_order_and_copies_one_future_batch():
    import torch
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    class Source:
        def __init__(self):
            self.rows = iter([TQLocalBatch([], [], None, plan_id=i, receipt_token=f"t{i}",
                                          prepared_batch={"x": torch.tensor([i])}) for i in range(2)])
            self.closed = 0
        def __next__(self): return next(self.rows)
        def close(self): self.closed += 1
    source, backend = Source(), _fake_device_backend()
    prefetch = ReferenceDevicePrefetchIterator(source, device=torch.device("cpu"), backend=backend,
                                               max_batch_bytes=64, close_timeout=5)
    try:
        first = next(prefetch)
        assert first.plan_id == 0 and first.receipt_token == "t0"
        assert backend.state.synchronized.wait(5), "Next batch should copy while the current batch is in use"
        second = next(prefetch)
        assert second.plan_id == 1 and second.prepared_batch["x"].item() == 1
        with pytest.raises(StopIteration): next(prefetch)
    finally:
        prefetch.close()
        prefetch.close()
    assert source.closed == 1
    with pytest.raises(StopIteration):
        next(prefetch)


def test_device_prefetch_records_consumer_stream_before_return(monkeypatch):
    import torch
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    copies, recorded, devices = [], [], []
    source_tensor = torch.ones(2)
    def copy(self, target, **kwargs):
        copies.append((target.type, kwargs.get("non_blocking")))
        return self
    monkeypatch.setattr(torch.Tensor, "to", copy)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: recorded.append(stream))
    backend = _fake_device_backend()
    backend.set_device = devices.append
    backend.current_stream = lambda: "consumer-stream"
    class Source:
        def __init__(self): self.rows = iter([TQLocalBatch([], [], None, plan_id=0, prepared_batch={"x": source_tensor})])
        def __next__(self): return next(self.rows)
        def close(self): pass
    prefetch = ReferenceDevicePrefetchIterator(Source(), device=torch.device("cuda:0"), backend=backend,
                                               max_batch_bytes=64, close_timeout=5)
    try:
        batch = next(prefetch)
        assert batch.prepared_batch["x"] is source_tensor
        assert devices == [0] and copies == [("cuda", True)] and recorded == ["consumer-stream"]
    finally:
        prefetch.close()


def test_device_prefetch_syncs_partial_copy_failure_and_closes_source(monkeypatch):
    import torch
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    class Source:
        closed = False
        def __next__(self): return TQLocalBatch([], [], None, plan_id=0, prepared_batch={"x": torch.ones(2)})
        def close(self): self.closed = True
    def fail_copy(self, *args, **kwargs):
        raise RuntimeError("injected device copy failure")
    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    source, backend = Source(), _fake_device_backend()
    prefetch = ReferenceDevicePrefetchIterator(source, device=torch.device("cpu"), backend=backend,
                                               max_batch_bytes=64, close_timeout=5)
    try:
        with pytest.raises(RuntimeError, match="device copy failure"):
            next(prefetch)
    finally:
        prefetch.close()
    assert source.closed and backend.state.copies == 1


def test_device_prefetch_close_waits_for_inflight_copy():
    import torch
    from contextlib import nullcontext
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Stream:
        def synchronize(self):
            started.set()
            assert release.wait(5)
    backend = SimpleNamespace(set_device=lambda index: None, Stream=Stream, stream=lambda stream: nullcontext())
    class Source:
        def __next__(self): return TQLocalBatch([], [], None, plan_id=0, prepared_batch={"x": torch.ones(2)})
        def close(self): closed.set()
    prefetch = ReferenceDevicePrefetchIterator(Source(), device=torch.device("cpu"), backend=backend,
                                               max_batch_bytes=64, close_timeout=5)
    assert started.wait(5)
    errors = []
    def close():
        try: prefetch.close()
        except Exception as exc: errors.append(exc)
    thread = threading.Thread(target=close)
    thread.start()
    assert closed.wait(5) and thread.is_alive()
    release.set()
    thread.join(5)
    assert not thread.is_alive() and not errors


def test_device_prefetch_shares_close_deadline_with_source():
    import time
    import torch
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQPrefetchIterator
    started, release, timeouts = threading.Event(), threading.Event(), []
    class Source(TQPrefetchIterator):
        def __next__(self):
            started.set()
            assert release.wait(5)
            raise StopIteration
        @property
        def has_running_work(self): return not release.is_set()
        def close(self, *, timeout=None):
            timeouts.append(timeout)
            if not release.wait(timeout):
                raise RuntimeError("source is still running")
    source = Source.__new__(Source)
    prefetch = ReferenceDevicePrefetchIterator(source, device=torch.device("cpu"), backend=_fake_device_backend(),
                                               max_batch_bytes=64, close_timeout=0.1)
    assert started.wait(5)
    try:
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="did not stop"):
            prefetch.close()
        assert time.monotonic() - before < 1 and prefetch.has_running_work
        assert 0 < timeouts[0] <= 0.101
    finally:
        release.set()
        prefetch.close()
    assert not prefetch.has_running_work


def test_device_prefetch_rejects_oversize_before_copy():
    import torch
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    class Source:
        def __next__(self): return TQLocalBatch([], [], None, plan_id=0, prepared_batch={"x": torch.ones(4)})
        def close(self): pass
    backend = _fake_device_backend()
    prefetch = ReferenceDevicePrefetchIterator(Source(), device=torch.device("cpu"), backend=backend,
                                               max_batch_bytes=15, close_timeout=5)
    try:
        with pytest.raises(ValueError, match="consumer_device_batch_bytes"):
            next(prefetch)
    finally:
        prefetch.close()
    assert backend.state.copies == 0


def _device_prefetch_gloo_worker(rank, port, fail_copy, results):
    import datetime
    import torch
    import torch.distributed as dist
    from contextlib import contextmanager
    from verl_speco.trainer.reference_batch_prefetch import ReferenceDevicePrefetchIterator
    from verl_speco.trainer.draft_training_loop import _next_batch_across_ranks
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=25))
    @contextmanager
    def stream_context(stream):
        if fail_copy and rank == 1:
            raise RuntimeError("injected device transfer failure")
        yield
    class Stream:
        def synchronize(self): pass
    backend = SimpleNamespace(set_device=lambda index: None, Stream=Stream, stream=stream_context)
    class Source:
        def __init__(self):
            self.rows = iter([TQLocalBatch([], [], None, plan_id=0, receipt_token="same", prepared_batch={"x": torch.tensor([rank])})])
        def __next__(self): return next(self.rows)
        def close(self): pass
    prefetch = ReferenceDevicePrefetchIterator(Source(), device=torch.device("cpu"), backend=backend,
                                               max_batch_bytes=64, close_timeout=10)
    try:
        try:
            batch = _next_batch_across_ranks(prefetch, rank=rank, device=torch.device("cpu"), require_prepared=True)
            results.put((rank, "ok", batch.prepared_batch["x"].item()))
        except RuntimeError as exc:
            results.put((rank, str(exc), None))
    finally:
        prefetch.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_copy", [False, True])
def test_device_prefetch_failure_is_synchronized_across_gloo_ranks(fail_copy):
    import multiprocessing
    import socket
    import torch.distributed as dist
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [ctx.Process(target=_device_prefetch_gloo_worker, args=(rank, port, fail_copy, results))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        rows = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        if fail_copy:
            assert all("producer failed" in row[1] for row in rows)
        else:
            assert {(row[0], row[1], row[2]) for row in rows} == {(0, "ok", 0), (1, "ok", 1)}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
