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
                         batch_prepare_seconds=0.3),
                     tq_prefetch=object() if depth else None,
                     batch_ready_wait_seconds=ready_wait, tq_get_seconds=0.3)
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
def test_prefetch_close_failure_still_cleans_resources(training_failed):
    # Execute the actual finally block without importing the unavailable Ray /
    # verl model stack. Resource ordering and exception identity matter here.
    import ast
    import asyncio
    import logging
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_run_standalone_draft_training_async")
    cleanup = next(n.finalbody for n in loop.body if isinstance(n, ast.Try) and n.finalbody)
    wrapper = ast.parse("async def exercise():\n    try:\n        if training_failed:\n"
                        "            raise training_error\n    finally:\n        pass\n")
    wrapper.body[0].body[0].finalbody = cleanup
    calls = []
    training_error = ValueError("training failed")
    close_error = RuntimeError("prefetch shutdown deadline")

    def close():
        calls.append("prefetch")
        raise close_error

    async def trainer_cleanup(**kwargs):
        calls.append("trainer")

    namespace = dict(
        sys=sys, logger=logging.getLogger(__name__), training_failed=training_failed,
        training_error=training_error, standalone_trackers=[], rank=0,
        current_stage="training_step", attempted_batches=1, successful_steps=0,
        optimizer_step=0, feature_producer=None, feature_replayer=None,
        tq_prefetch=SimpleNamespace(close=close), tq_prefetch_group=object(), batch_finalizer=None,
        feature_store_type="tq",
        store=SimpleNamespace(close=lambda: calls.append("store")),
        trainer=SimpleNamespace(cleanup_training=trainer_cleanup, runtime_device="cpu"),
        dist=SimpleNamespace(destroy_process_group=lambda group: calls.append("group"),
                             is_initialized=lambda: False),
        _finish_standalone_tracking=lambda *a, **k: None,
        _sync_standalone_export_error=lambda *a, **k: False,
    )
    exec(compile(ast.fix_missing_locations(wrapper), str(path), "exec"), namespace)
    with pytest.raises(type(training_error if training_failed else close_error)) as caught:
        asyncio.run(namespace["exercise"]())
    assert caught.value is (training_error if training_failed else close_error)
    assert calls == ["prefetch", "group", "store", "trainer"]


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
