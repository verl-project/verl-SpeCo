# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
import ast
import logging
from pathlib import Path
from queue import Queue
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.standalone_batch_finalize import AsyncBatchFinalizer


def _complete_function():
    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_clear_tq_batch_across_ranks")
    namespace = dict(
        torch=torch, dist=SimpleNamespace(is_initialized=lambda: False),
        retry_io=lambda operation, **kwargs: operation(),
        put_queue=lambda queue, item, **kwargs: queue.put(item),
        _TQ_CLEAR_MAX_ATTEMPTS=3, _TQ_CLEAR_RETRY_BACKOFF_SECONDS=0.5,
        logger=logging.getLogger(__name__),
    )
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function.name]


def test_cleanup_overlaps_training_and_ack_follows_delete(tmp_path):
    path = tmp_path / "hidden"
    path.write_text("hidden")
    started, release = Event(), Event()
    caller = get_ident()
    events = Queue()

    def delete(keys):
        assert get_ident() != caller
        assert keys == ["sample"]
        started.set()
        assert release.wait(5)
        path.unlink()

    finalizer = AsyncBatchFinalizer(2)
    try:
        _complete_function()(
            SimpleNamespace(clear_completed_batch=delete), ["sample"], rank=0,
            device=torch.device("cpu"), training_events=events,
            completion_event={"kind": "training_completed", "plan_id": 1}, finalizer=finalizer,
        )
        assert started.wait(5)
        assert path.exists() and events.empty()  # Main thread can run the next step now.
        release.set()
        finalizer.drain()
        assert not path.exists()
        assert events.get_nowait() == {"kind": "training_completed", "plan_id": 1}
    finally:
        release.set()
        finalizer.close()


def test_backlog_is_bounded_and_fifo():
    finalizer = AsyncBatchFinalizer(1)
    started, release, submitted = Event(), Event(), Event()
    order = []

    def first():
        started.set()
        assert release.wait(5)
        order.append(1)

    finalizer.submit(first)
    assert started.wait(5)

    def submit_second():
        finalizer.submit(lambda: order.append(2))
        submitted.set()

    thread = Thread(target=submit_second)
    thread.start()
    try:
        assert not submitted.wait(0.05)
    finally:
        release.set()
        thread.join(5)
        finalizer.close()
    assert submitted.is_set() and order == [1, 2]


def test_failure_is_visible_to_driver_and_main_thread():
    events = Queue()
    finalizer = AsyncBatchFinalizer(2)

    def fail(keys):
        raise OSError("cleanup failed")

    _complete_function()(
        SimpleNamespace(clear_completed_batch=fail), ["sample"], rank=0,
        device=torch.device("cpu"), training_events=events,
        completion_event={"kind": "training_completed", "plan_id": 7}, finalizer=finalizer,
    )
    assert events.get(timeout=5) == {"kind": "finalization_failed", "plan_id": 7}
    with pytest.raises(OSError, match="cleanup failed"):
        finalizer.drain()
    with pytest.raises(OSError, match="cleanup failed"):
        finalizer.close()
    assert events.empty()


def test_legacy_path_stays_synchronous():
    calls = []
    _complete_function()(
        SimpleNamespace(clear_completed_batch=lambda keys: calls.append(keys)),
        ["sample"], rank=0, device=torch.device("cpu"),
    )
    assert calls == [["sample"]]


def test_file_ack_has_no_collective_and_does_not_delete_again():
    complete = _complete_function()
    calls, events = [], Queue()

    def forbidden(*args, **kwargs):
        pytest.fail("File ACK must not execute a distributed collective")

    complete.__globals__["dist"] = SimpleNamespace(
        is_initialized=lambda: True, get_world_size=lambda: 2, all_reduce=forbidden,
    )
    finalizer = AsyncBatchFinalizer(2)
    try:
        complete(SimpleNamespace(clear_completed_batch=lambda keys, **kw: calls.append((keys, kw))),
                 ["sample"], rank=0, device=torch.device("cpu"), training_events=events,
                 completion_event={"kind": "training_completed"}, finalizer=finalizer,
                 synchronize=False, cleanup_after_load=True)
        complete(None, None, rank=1, device=torch.device("cpu"), synchronize=False)
        finalizer.drain()
        assert calls == [(["sample"], {"cleanup_after_load": True})]
        assert events.get_nowait() == {"kind": "training_completed"}
    finally:
        finalizer.close()


def test_file_ack_retries_transport_without_repeating_metadata_release():
    from verl_speco.trainer.tq_retry import retry_io

    complete = _complete_function()
    complete.__globals__["retry_io"] = lambda op, **kw: retry_io(
        op, **dict(kw, initial_backoff=0, max_backoff=0),
    )
    attempts, cleared = [], []

    class Events:
        def put(self, event):
            attempts.append(event)
            if len(attempts) < 3:
                raise TimeoutError("temporary ACK timeout")

    complete(SimpleNamespace(clear_completed_batch=lambda *a, **kw: cleared.append(a)),
             ["sample"], rank=0, device=torch.device("cpu"), training_events=Events(),
             completion_event={"kind": "training_completed"}, synchronize=False,
             cleanup_after_load=True)
    assert len(attempts) == 3 and len(cleared) == 1
