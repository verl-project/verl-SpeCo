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


def _collect_gloo_workers(target, arguments, *, timeout=45):
    import multiprocessing
    import socket
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=target, args=arguments(rank, port, results)) for rank in range(2)]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        rows = [results.get(timeout=timeout) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        return rows
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


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


def test_reference_optimizer_ack_precedes_cleanup_and_completion():
    events, order = Queue(), []
    def clear(keys):
        order.append(events.get_nowait()["kind"])
        assert keys == ["sample"]
    _complete_function()(
        SimpleNamespace(clear_completed_batch=clear), ["sample"], rank=0,
        device=torch.device("cpu"), training_events=events,
        completion_event={"kind": "training_completed", "plan_id": 3, "receipt_token": "token"},
    )
    assert order == ["training_applied"]
    assert events.get_nowait()["kind"] == "training_completed"


def test_reference_optimizer_ack_failure_preserves_input():
    events, calls = Queue(), []
    def put(item):
        if item["kind"] == "training_applied":
            raise OSError("ack failed")
        calls.append(item)
    events.put = put
    with pytest.raises(RuntimeError, match="failed to clear"):
        _complete_function()(
            SimpleNamespace(clear_completed_batch=lambda keys: calls.append("deleted")), ["sample"],
            rank=0, device=torch.device("cpu"), training_events=events,
            completion_event={"kind": "training_completed", "plan_id": 3, "receipt_token": "token"},
        )
    assert calls == []


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


def test_finalization_timeout_retains_future_for_later_drain():
    finalizer = AsyncBatchFinalizer(1)
    started, release = Event(), Event()
    def slow():
        started.set()
        assert release.wait(5)
    finalizer.submit(slow)
    assert started.wait(5)
    try:
        with pytest.raises(TimeoutError):
            finalizer.drain(timeout=0.01)
        assert len(finalizer._pending) == 1
    finally:
        release.set()
        finalizer.close()
    assert not finalizer._pending


def test_backpressure_timeout_preserves_inflight_work():
    finalizer = AsyncBatchFinalizer(1, timeout=0.02)
    started, release, executed = Event(), Event(), []
    def slow():
        started.set()
        assert release.wait(5)
    finalizer.submit(slow)
    assert started.wait(5)
    try:
        with pytest.raises(TimeoutError):
            finalizer.submit(lambda: executed.append("unexpected"))
        assert len(finalizer._pending) == 1 and executed == []
        with pytest.raises(RuntimeError, match="stopped after a timeout"):
            finalizer.submit(lambda: executed.append("retry"))
    finally:
        release.set()
        finalizer.close()


def test_finalizer_close_timeout_cancels_queued_work_without_waiting_for_io():
    finalizer = AsyncBatchFinalizer(2)
    started, release, done, later = Event(), Event(), Event(), []
    def slow():
        started.set()
        try:
            assert release.wait(5)
        finally:
            done.set()
    finalizer.submit(slow)
    assert started.wait(5)
    finalizer.submit(lambda: later.append("unexpected"))
    try:
        with pytest.raises(TimeoutError):
            finalizer.close(timeout=0.02)
        assert finalizer._pending[1].cancelled() and later == [] and not done.is_set()
        assert finalizer.has_running_work
    finally:
        release.set()
        assert done.wait(5)
        finalizer._pending[0].result(timeout=5)
    assert later == []
    assert not finalizer.has_running_work


def test_finalizer_drain_timeout_fences_queued_work_before_close():
    finalizer = AsyncBatchFinalizer(2)
    started, release = Event(), Event()
    later = []
    def slow():
        started.set()
        assert release.wait(5)
    finalizer.submit(slow)
    assert started.wait(5)
    finalizer.submit(lambda: later.append("unexpected"))
    try:
        with pytest.raises(TimeoutError):
            finalizer.drain(timeout=0.02)
        with pytest.raises(RuntimeError, match="stopped after a timeout"):
            finalizer.submit(lambda: later.append("new"))
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="stopped after a timeout"):
        finalizer.close()
    assert later == []


def test_commit_gate_waits_for_previous_finalization_before_next_step():
    from verl_speco.trainer.draft_training_loop import _drain_batch_finalizer_across_ranks
    finalizer = AsyncBatchFinalizer(1)
    started, release, next_step = Event(), Event(), Event()
    def previous():
        started.set()
        assert release.wait(5)
    finalizer.submit(previous)
    assert started.wait(5)
    def gated_step():
        _drain_batch_finalizer_across_ranks(finalizer, torch.device("cpu"), timeout=3)
        next_step.set()
    thread = Thread(target=gated_step)
    thread.start()
    try:
        assert not next_step.wait(0.05)
        release.set()
        thread.join(5)
        assert not thread.is_alive() and next_step.is_set()
    finally:
        release.set()
        thread.join(5)
        finalizer.close()


def test_failure_fences_already_queued_finalizers():
    finalizer = AsyncBatchFinalizer(3)
    started, release = Event(), Event()
    cleared = []
    def first():
        started.set()
        assert release.wait(5)
        raise OSError("first cleanup failed")
    finalizer.submit(first)
    assert started.wait(5)
    finalizer.submit(lambda: cleared.append("second"))
    finalizer.submit(lambda: cleared.append("third"))
    release.set()
    with pytest.raises(OSError, match="first cleanup failed"):
        finalizer.close()
    assert cleared == []


def test_reference_finalization_failure_preserves_later_queued_references():
    finalizer = AsyncBatchFinalizer(2)
    started, release = Event(), Event()
    events, cleared = Queue(), []
    def cleanup(keys):
        if keys == ["first"]:
            started.set()
            assert release.wait(5)
            raise OSError("first cleanup failed")
        cleared.extend(keys)
    complete = _complete_function()
    loader = SimpleNamespace(clear_completed_batch=cleanup)
    for plan in (1, 2):
        complete(loader, ["first" if plan == 1 else "second"], rank=0,
                 device=torch.device("cpu"), training_events=events, finalizer=finalizer,
                 completion_event={"kind": "training_completed", "plan_id": plan,
                                   "receipt_token": str(plan)})
        if plan == 1:
            assert started.wait(5)
    release.set()
    assert events.get(timeout=5) == {"kind": "training_applied", "plan_id": 1, "receipt_token": "1"}
    assert events.get(timeout=5) == {"kind": "finalization_failed", "plan_id": 1}
    with pytest.raises(OSError, match="first cleanup failed"):
        finalizer.close()
    assert not cleared and events.empty()


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


def test_reference_async_ack_failure_wakes_driver_and_retains_input():
    events = Queue()
    calls = []
    complete = _complete_function()
    def put(queue, item, **kwargs):
        if item["kind"] == "training_applied":
            raise OSError("ack failed")
        queue.put(item)
    complete.__globals__["put_queue"] = put
    finalizer = AsyncBatchFinalizer(1)
    complete(SimpleNamespace(clear_completed_batch=lambda keys: calls.append(keys)), ["sample"],
             rank=0, device=torch.device("cpu"), training_events=events, finalizer=finalizer,
             completion_event={"kind": "training_completed", "plan_id": 7, "receipt_token": "token"})
    assert events.get(timeout=5) == {"kind": "finalization_failed", "plan_id": 7}
    with pytest.raises(OSError, match="ack failed"):
        finalizer.close()
    assert calls == [] and events.empty()


def test_reference_cleanup_failure_reports_applied_without_completion():
    events = Queue()
    finalizer = AsyncBatchFinalizer(1)
    def fail(keys):
        raise OSError("cleanup failed")
    _complete_function()(
        SimpleNamespace(clear_completed_batch=fail), ["sample"], rank=0,
        device=torch.device("cpu"), training_events=events, finalizer=finalizer,
        completion_event={"kind": "training_completed", "plan_id": 7, "receipt_token": "token"},
    )
    assert events.get(timeout=5) == {"kind": "training_applied", "plan_id": 7, "receipt_token": "token"}
    assert events.get(timeout=5) == {"kind": "finalization_failed", "plan_id": 7}
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


def test_multinode_cleanup_waits_for_all_ranks():
    import threading

    complete = _complete_function()
    barrier = threading.Barrier(2, timeout=5)
    cleared = []
    complete.__globals__["dist"] = SimpleNamespace(
        is_initialized=lambda: True, get_world_size=lambda: 2, barrier=barrier.wait,
    )
    loader = SimpleNamespace(clear_completed_batch=lambda keys: cleared.append(keys))
    rank0 = threading.Thread(target=lambda: complete(
        loader, ["sample"], rank=0, device=torch.device("cpu"),
        synchronize=False, synchronize_before_cleanup=True,
    ))
    rank0.start()
    assert not cleared
    complete(None, None, rank=1, device=torch.device("cpu"),
             synchronize=False, synchronize_before_cleanup=True)
    rank0.join(timeout=5)
    assert not rank0.is_alive() and cleared == [["sample"]]


def test_multinode_cleanup_refuses_uninitialized_group():
    complete = _complete_function()
    complete.__globals__["dist"] = SimpleNamespace(is_initialized=lambda: False)
    cleared = []
    with pytest.raises(RuntimeError, match="initialized distributed group"):
        complete(SimpleNamespace(clear_completed_batch=lambda keys: cleared.append(keys)),
                 ["sample"], rank=0, device=torch.device("cpu"),
                 synchronize=False, synchronize_before_cleanup=True)
    assert not cleared


def _gloo_cleanup_worker(rank, port, proceed, ready, cleaned):
    import torch.distributed as dist
    from datetime import timedelta

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=2, timeout=timedelta(seconds=20),
    )
    try:
        complete = _complete_function()
        complete.__globals__["dist"] = dist
        loader = SimpleNamespace(clear_completed_batch=lambda keys: cleaned.set())
        ready.put(rank)
        if rank == 1:
            assert proceed.wait(10)
        complete(loader if rank == 0 else None, ["sample"] if rank == 0 else None,
                 rank=rank, device=torch.device("cpu"), synchronize=False,
                 synchronize_before_cleanup=True)
        from verl_speco.trainer.standalone_tracking import StandaloneTokenWindow
        window = StandaloneTokenWindow(1)
        window.record_checkpoint(0.25 + rank * 0.25)
        metrics = window.update(
            {"source_tokens": 10 + 10 * rank, "window_tokens": 8,
             "supervised_tokens": 4, "padding_tokens": 2, "samples": 1,
             "window_train_input_positions": 10 + rank * 5, "window_statistics_input_positions": rank * 7},
            successful_steps=1, optimizer_step=1, max_steps=1,
            start_time=__import__("time").perf_counter() - (1 + rank),
            device=torch.device("cpu"), world_size=2,
        )
        ready.put((rank, metrics["tokens/source"], metrics["tokens/padding"],
                   metrics["tokens/source_per_second"], metrics["tokens/source_per_second_excluding_checkpoints"],
                   metrics["perf/checkpoint_blocking_time"], metrics["tokens/window_train_input_positions"],
                   metrics["tokens/window_statistics_input_positions"]))
        window.record_checkpoint(0.5)
        run = window.finish(device=torch.device("cpu"), world_size=2)
        ready.put((rank, run["tokens/run_source"], run["perf/run_checkpoint_blocking_time"],
                   run["tokens/run_source_per_second"], run["tokens/run_window_train_input_positions"],
                   run["tokens/run_window_statistics_input_positions"]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_real_two_process_group_waits_before_reference_cleanup():
    import multiprocessing
    import socket

    context = multiprocessing.get_context("spawn")
    proceed, cleaned, ready = context.Event(), context.Event(), context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_gloo_cleanup_worker,
                                 args=(rank, port, proceed, ready, cleaned))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        assert {ready.get(timeout=20), ready.get(timeout=20)} == {0, 1}
        assert not cleaned.wait(0.5)
        proceed.set()
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        assert cleaned.is_set()
        metrics = [ready.get(timeout=5), ready.get(timeout=5)]
        assert {row[:3] for row in metrics} == {(0, 30.0, 4.0), (1, 30.0, 4.0)}
        assert metrics[0][3:] == metrics[1][3:]
        assert 14.0 < metrics[0][3] <= 15.0
        assert 19.0 < metrics[0][4] <= 20.0
        assert metrics[0][5] == 0.5
        assert metrics[0][6:] == (25.0, 7.0)
        run_metrics = [ready.get(timeout=5), ready.get(timeout=5)]
        assert {row[:3] for row in run_metrics} == {(0, 30.0, 1.0), (1, 30.0, 1.0)}
        assert run_metrics[0][3] == run_metrics[1][3]
        assert run_metrics[0][4:] == run_metrics[1][4:] == (25.0, 7.0)
    finally:
        proceed.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        ready.close()


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


def _gloo_uneven_training_worker(rank, port, results, empty_rank, combine):
    import asyncio
    from contextlib import nullcontext
    from datetime import timedelta
    import torch.distributed as dist
    import verl_speco.trainer.base_trainer as trainer_module
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        trainer_module.device_name = "cpu"
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        model = torch.nn.parallel.DistributedDataParallel(model)
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer.lr_scheduler = None
        trainer.optimizer_steps_total = 0
        trainer.training_steps = 0
        trainer._standalone_validate_grad_graph = True
        trainer.combine_standalone_metrics = combine
        trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
        trainer.record_training_timing = lambda *args: None
        trainer._ulysses_group_context = nullcontext
        trainer._record_dflash_training_metrics = lambda losses: {}
        trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
        trainer._is_checkpoint_leader = lambda: False
        def reduce_statistics(values):
            dist.all_reduce(values)
            return values, 2
        trainer._reduce_loss_statistics = reduce_statistics
        def compute_loss(model, batch, pad):
            prediction = model(batch["x"])
            loss = ((prediction.float() - batch["y"]) ** 2).sum()
            count = float(batch["y"].numel())
            if empty_rank and rank == 1:
                loss, count = torch.tensor(0.0), 0.0
            return {"total_local_vloss": torch.tensor(0.0), "total_local_ploss": loss,
                    "local_num_tokens": torch.tensor(count),
                    "v_weight": 0.0, "p_weight": 1.0,
                    "diagnostics": {"correct_count": torch.tensor(count)}}
        trainer.backend = SimpleNamespace(model_type="test", compute_loss=compute_loss)
        labels = [0.0, 1.0] if rank == 0 else [2.0]
        batch = {"x": torch.ones((len(labels), 1)), "y": torch.tensor(labels).reshape(-1, 1),
                 "_speco_sample_count": len(labels)}
        try:
            assert asyncio.run(trainer._training_step_on_batch(batch, 0))
        except RuntimeError as exc:
            if not empty_rank or "no loss gradient graph" not in str(exc):
                raise
            results.put((rank, "no graph"))
        else:
            results.put((rank, float(model.module.weight.item())))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("combine", [False, True])
def test_uneven_rank_tail_uses_global_token_weighted_gradient(empty_rank, combine):
    values = dict(_collect_gloo_workers(_gloo_uneven_training_worker,
        lambda rank, port, results: (rank, port, results, empty_rank, combine), timeout=40))
    if empty_rank:
        assert values == {0: "no graph", 1: "no graph"}
    else:
        # Mean over targets [0, 1, 2] has gradient -2 at weight zero.
        assert values == pytest.approx({0: 0.2, 1: 0.2}, abs=0.002)


def _gloo_reference_parts_worker(rank, port, results, mode):
    import asyncio
    from contextlib import nullcontext
    from datetime import timedelta
    import torch.distributed as dist
    import verl_speco.trainer.base_trainer as trainer_module
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.draft_training_loop import _reference_training_parts_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        trainer_module.device_name = "cpu"
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.zero_()
        model = torch.nn.parallel.DistributedDataParallel(model)
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model, trainer.runtime_device = model, torch.device("cpu")
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer.lr_scheduler = None
        trainer.optimizer_steps_total = trainer.training_steps = 0
        trainer._standalone_validate_grad_graph = True
        trainer.combine_standalone_metrics = False
        trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
        trainer.record_training_timing = lambda *args: None
        trainer._ulysses_group_context = nullcontext
        trainer._record_dflash_training_metrics = lambda losses: {}
        trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
        trainer._is_checkpoint_leader = lambda: False
        def reduce_statistics(values):
            dist.all_reduce(values)
            return values, 2
        trainer._reduce_loss_statistics = reduce_statistics
        def compute_loss(module, batch, pad):
            loss = (((module(batch["x"]).float() - batch["y"]) ** 2) * batch["mask"]).sum()
            if mode == "zero_missing_graph" and rank == 1:
                loss = loss.detach()
            return {"total_local_vloss": torch.tensor(0.0), "total_local_ploss": loss,
                    "local_num_tokens": batch["mask"].sum(), "v_weight": 0.0, "p_weight": 1.0}
        trainer.backend = SimpleNamespace(model_type="test", compute_loss=compute_loss, reference_parts_additive=True)
        if mode in {"unsupported", "prepass_mismatch"}:
            trainer.backend.model_type = "dspark"
            model.module.loss_mode = "full_vocab"
            model.module.per_position_loss_weight = "fixed_exp_decay"
            model.module.ce_target_source = "labels"
            model.module.l1_loss_alpha = model.module.confidence_head_alpha = 0.0
            model.module.lk_loss_alpha = 0.5 if rank == 1 else 0.0
            model.module.loss_decay_gamma, model.module.block_size = 7.0, 2
            if mode == "prepass_mismatch":
                model.module.lk_loss_type, model.module.lk_hybrid_eta = "adaptive_hybrid", 3.0
        def batch(label, mask=1.0):
            if mode in {"unsupported", "prepass_mismatch"}:
                raise RuntimeError("unsupported recipe unexpectedly read a part")
            return {"x": torch.ones((1, 1)), "y": torch.tensor([[label]]), "mask": torch.tensor([[mask]])}
        def failed_read():
            raise OSError("injected part failure")
        scale = 0.25 if mode == "small_weight" else 0.5 if mode == "weighted" else 1.0
        parts = [lambda: batch(1.0 if rank == 0 else 2.0, scale),
                 failed_read if mode == "failure" and rank == 1 else
                 lambda: batch(3.0 if rank == 0 else 0.0, scale if rank == 0 else 0.0)]
        if mode in {"zero_first", "zero_last", "all_zero", "zero_missing_graph"}:
            parts = [lambda: batch(1.0 if rank == 0 else 2.0, 0.0 if mode != "zero_last" else 1.0),
                     lambda: batch(3.0 if rank == 0 else 0.0, 1.0 if mode == "zero_first" and rank == 0 else 0.0)]
        if mode == "unequal" and rank == 1:
            parts.pop()
        if mode == "unequal":
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        try:
            asyncio.run(_reference_training_parts_across_ranks(trainer, parts, 0, device=torch.device("cpu")))
            result = "ok"
        except RuntimeError as exc:
            result = str(exc)
        results.put((rank, result, trainer.optimizer_steps_total, float(model.module.weight.item()),
                     all(parameter.grad is None for parameter in model.parameters())))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("mode", ["ok", "weighted", "small_weight", "failure", "unequal", "unsupported", "prepass_mismatch",
                                  "zero_first", "zero_last", "all_zero", "zero_missing_graph"])
def test_reference_parts_one_optimizer_step_and_failure_agreement(mode):
    rows = _collect_gloo_workers(_gloo_reference_parts_worker,
        lambda rank, port, results: (rank, port, results, mode))
    assert {row[0] for row in rows} == {0, 1}
    for _, result, steps, weight, gradients_cleared in rows:
        assert gradients_cleared
        if mode in {"ok", "weighted", "small_weight", "zero_first", "zero_last"}:
            expected = 0.6 if mode == "zero_first" else 0.3 if mode in {"small_weight", "zero_last"} else 0.4
            assert result == "ok" and steps == 1 and weight == pytest.approx(expected, abs=0.002)
        else:
            assert {"failure": "materialization", "unequal": "part count", "unsupported": "loss recipe",
                    "prepass_mismatch": "prepass", "all_zero": "training failed",
                    "zero_missing_graph": "training failed"}[mode] in result
            assert steps == 0 and weight == 0.0


def _gloo_real_reference_parts_worker(rank, port, results, mode, strategy):
    import asyncio
    from contextlib import nullcontext
    from copy import deepcopy
    from datetime import timedelta
    import torch.distributed as dist
    import verl_speco.trainer.base_trainer as trainer_module
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.draft_training_loop import _reference_training_parts_across_ranks
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(912)
        trainer_module.device_name = "cpu"
        if mode.startswith("dpace_zero"):
            import torch.nn.functional as functional
            original_nll = functional.nll_loss
            def extreme_nll(log_probs, targets, *args, **kwargs):
                result = original_nll(log_probs, targets, *args, **kwargs)
                return result + (targets.remainder(2) == 0).to(result.dtype) * 200.0
            functional.nll_loss = extreme_nll
        confidence = mode in {"confidence", "dpace_aux", "dpace_aux_changed", "dpace_aux_failure", "restricted_aux", "sampled_parts_aux"}
        config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                              vocab_size=16, num_target_layers=4, num_context_layers=1,
                              target_hidden_size=8, target_num_hidden_layers=4, target_layer_ids=[1],
                              mask_token_id=15, markov_rank=4, markov_head_type="vanilla",
                              enable_confidence_head=confidence, confidence_head_with_markov=True,
                              sliding_window=3, use_sliding_window=True, layer_types=["sliding_attention"])
        adaptive = mode in {"adaptive", "adaptive_failure", "adaptive_changed", "confidence",
                            "dpace_aux", "dpace_aux_changed", "dpace_aux_failure", "restricted_aux", "sampled_parts_aux", "dpace_zero_aux"}
        if confidence:
            predict = DSparkDraftModel.predict_confidence
            def with_invalid_rows(self, draft_hidden, *, prev_token_ids):
                logits = predict(self, draft_hidden, prev_token_ids=prev_token_ids)
                return torch.where(prev_token_ids.remainder(2) == 0, torch.full_like(logits, float("inf")), logits)
            DSparkDraftModel.predict_confidence = with_invalid_rows
        raw = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                  num_anchors=3, l1_loss_alpha=0.2 if mode.startswith("dpace_aux") or mode in {"restricted_aux", "sampled_parts_aux", "dpace_zero_aux"} else 0.0,
                                  per_position_loss_weight="dpace" if mode.startswith("dpace") or mode in {"restricted_aux", "sampled_parts_aux"} else "fixed_exp_decay",
                                  dpace_alpha=0.0 if mode.startswith("dpace_zero") else 0.5,
                                  loss_mode="restricted_ce" if mode.startswith("restricted") else "sampled_ce" if mode.startswith("sampled_parts") else "full_vocab",
                                  sampled_ce_negatives=9,
                                  lk_loss_alpha=0.3 if adaptive else 0.0,
                                  lk_loss_type="adaptive_hybrid" if adaptive else "alpha",
                                  confidence_head_alpha=0.4 if confidence else 0.0,
                                  verifier_norm=torch.nn.LayerNorm(8) if adaptive else None)
        baseline = deepcopy(raw)
        head = torch.randn(16, 8)
        ids = torch.arange(12).reshape(1, 12)
        hidden = torch.randn(1, 12, 8)
        mask = torch.ones(1, 12)
        docs = torch.zeros(1, 12, dtype=torch.long)
        if mode.startswith("sampled_parts"):
            torch.manual_seed(542)
            sampled_vocab = torch.unique(torch.cat((torch.arange(12), torch.randint(0, 16, (9,)))), sorted=True)
            torch.manual_seed(542)
        def make_batch(start, end, anchors, keep):
            batch = {"input_ids": ids[:, start:end], "hidden_states": hidden[:, start:end],
                    "loss_mask": mask[:, start:end] if any(keep) else torch.zeros_like(mask[:, start:end]),
                    "label_ids": ids[:, start:end], "label_mask": mask[:, start:end] if any(keep) else torch.zeros_like(mask[:, start:end]),
                    "document_ids": docs[:, start:end],
                    "preselected_anchor_positions": torch.tensor([[a - start for a in anchors]]),
                    "preselected_anchor_keep_mask": torch.tensor([keep])}
            if adaptive:
                batch.update(target_last_hidden_states=hidden[:, start:end],
                             last_hidden_state_norm="raw_target_residual", hidden_source="standalone_vllm_reference")
            if mode.startswith("restricted"):
                batch["preselected_restricted_vocab"] = torch.arange(12)
            if mode.startswith("sampled_parts"):
                batch["preselected_restricted_vocab"] = sampled_vocab
            return batch
        backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
        backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=head))
        backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
            "mode": "offline", "feature_store": {"type": "ray_reference"},
            "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}},
        })))
        selected = ([3, 9] if rank == 0 else [6]) if adaptive else [3, 6, 9]
        original = make_batch(0, 12, selected, [True] * len(selected))
        original.pop("preselected_restricted_vocab", None)
        reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            reference = backend.compute_loss(baseline, original, 0)
            if confidence:
                diagnostic = reference["diagnostics"]
                assert 0 < diagnostic["confidence_weighted_token_count"] < diagnostic["ce_weighted_token_count"]
            denominator = reference["local_num_tokens"].detach().clone()
            if adaptive:
                dist.all_reduce(denominator)
            reference_loss = reference["total_local_ploss"] / denominator.clamp(min=1)
        reference_loss.backward()
        if adaptive:
            for parameter in baseline.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
        reference_optimizer.step()
        if strategy == "fsdp1":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(raw, use_orig_params=True, device_id=torch.device("cpu"))
        elif strategy == "fsdp2":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard
            model = raw
            fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
        else:
            model = torch.nn.parallel.DistributedDataParallel(raw, find_unused_parameters=True)
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model, trainer.runtime_device = model, torch.device("cpu")
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer.lr_scheduler = None
        trainer.optimizer_steps_total = trainer.training_steps = 0
        trainer._standalone_validate_grad_graph = True
        trainer.combine_standalone_metrics = False
        trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
        trainer.record_training_timing = lambda *args: None
        trainer._ulysses_group_context = nullcontext
        trainer._record_dflash_training_metrics = lambda losses: {}
        trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
        trainer._is_checkpoint_leader = lambda: False
        def reduce_statistics(values):
            dist.all_reduce(values)
            return values, 2
        trainer._reduce_loss_statistics = reduce_statistics
        trainer.backend = backend
        if adaptive:
            trainer._stage_standalone_cpu_batch = lambda batch: batch
        parts = ([make_batch(0, 6, [3], [True]), make_batch(6, 12, [9], [True])] if rank == 0 else
                 [make_batch(3, 9, [6], [True]), make_batch(3, 9, [6], [False])])
        if mode == "invalid_anchor" and rank == 1:
            parts[0]["preselected_anchor_positions"] = torch.tensor([[99]])
        if mode == "invalid_hidden" and rank == 1:
            parts[0]["hidden_states"] = parts[0]["hidden_states"][..., :4]
        if mode == "restricted_changed" and rank == 1:
            parts[1]["preselected_restricted_vocab"] = torch.arange(11)
        if mode in {"adaptive_failure", "dpace_aux_failure"} and rank == 1:
            def fail_part():
                raise OSError("injected adaptive prepass read failure")
            parts[1] = fail_part
        if mode in {"adaptive_changed", "dpace_aux_changed"} and rank == 1:
            original_part, reads = parts[0], [0]
            def changed_part():
                reads[0] += 1
                batch = dict(original_part)
                if reads[0] > 1:
                    batch["hidden_states"] = batch["hidden_states"].clone()
                    batch["hidden_states"][0, 0, 0] += 1
                return batch
            parts[0] = changed_part
        try:
            asyncio.run(_reference_training_parts_across_ranks(trainer, parts, 0, device=torch.device("cpu")))
        except RuntimeError as exc:
            if mode not in {"invalid_anchor", "invalid_hidden", "adaptive_failure", "adaptive_changed",
                            "dpace_aux_failure", "dpace_aux_changed", "restricted_changed"}:
                raise
            results.put((rank, trainer.optimizer_steps_total, str(exc)))
            return
        context = FSDP.summon_full_params(model, writeback=False) if strategy == "fsdp1" else nullcontext()
        with context:
            differences = []
            for a, b in zip(raw.parameters(), baseline.parameters(), strict=True):
                value = a.detach().full_tensor() if hasattr(a, "full_tensor") else a.detach()
                differences.append(float((value - b.detach()).abs().max()))
        results.put((rank, trainer.optimizer_steps_total, max(differences)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("strategy,mode", [("ddp", "valid"), ("ddp", "invalid_anchor"),
                                           ("ddp", "invalid_hidden"), ("fsdp1", "valid"),
                                           ("fsdp1", "invalid_anchor"), ("fsdp2", "valid"),
                                           ("fsdp2", "invalid_anchor"), ("fsdp1", "adaptive"),
                                           ("fsdp2", "adaptive"), ("ddp", "adaptive_failure"),
                                           ("ddp", "adaptive_changed"), ("ddp", "confidence"),
                                           ("fsdp1", "confidence"), ("fsdp2", "confidence"),
                                           ("ddp", "dpace"), ("fsdp1", "dpace"), ("fsdp2", "dpace"),
                                           ("ddp", "dpace_aux"), ("fsdp1", "dpace_aux"), ("fsdp2", "dpace_aux"),
                                           ("ddp", "dpace_aux_changed"), ("ddp", "dpace_aux_failure"),
                                           ("ddp", "restricted"), ("fsdp1", "restricted_aux"),
                                           ("fsdp2", "restricted_aux"), ("ddp", "restricted_changed"),
                                           ("ddp", "sampled_parts"), ("fsdp1", "sampled_parts_aux"),
                                           ("fsdp2", "sampled_parts_aux"), ("ddp", "dpace_zero"),
                                           ("fsdp1", "dpace_zero_aux"), ("fsdp2", "dpace_zero_aux")])
def test_real_reference_parts_match_full_batch_optimizer_step(mode, strategy):
    rows = _collect_gloo_workers(_gloo_real_reference_parts_worker,
        lambda rank, port, results: (rank, port, results, mode, strategy), timeout=60)
    assert {rank for rank, _, _ in rows} == {0, 1}
    if mode in {"valid", "adaptive", "confidence", "dpace", "dpace_aux", "restricted", "restricted_aux",
                "sampled_parts", "sampled_parts_aux", "dpace_zero", "dpace_zero_aux"}:
        assert all(steps == 1 and difference < 0.002 for _, steps, difference in rows)
    else:
        assert all(steps == 0 and "materialization or staging failed" in message for _, steps, message in rows)


def _gloo_fsdp_gradient_clip_worker(rank, port, results, strategy):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        if strategy == "fsdp1":
            model = FSDP(torch.nn.Linear(4, 1, bias=False), use_orig_params=True, device_id=torch.device("cpu"))
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard
            model = torch.nn.Linear(4, 1, bias=False)
            fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
        model(torch.tensor([[1., 2., 3., 4.]])).sum().backward()
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model = model
        trainer._standalone_validate_grad_graph = True
        trainer._bubble_time_enabled = False
        def global_grad_norm():
            grads = [p.grad.detach().to_local() if hasattr(p.grad, "to_local") else p.grad.detach()
                     for p in model.parameters() if p.grad is not None]
            total = sum((gradient.float().square().sum() for gradient in grads),
                        torch.tensor(0.0))
            dist.all_reduce(total)
            return float(total.sqrt())
        before = global_grad_norm()
        clipped_norm = trainer._clip_training_grad_norm()
        assert bool(torch.isfinite(clipped_norm))
        reported = float(clipped_norm)
        after = global_grad_norm()
        results.put((rank, before, reported, after))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("strategy", ["fsdp1", "fsdp2"])
def test_standalone_fsdp_gradient_clipping_uses_global_norm(strategy):
    rows = _collect_gloo_workers(_gloo_fsdp_gradient_clip_worker,
        lambda rank, port, results: (rank, port, results, strategy))
    assert {rank for rank, _, _, _ in rows} == {0, 1}
    for _, before, reported, after in rows:
        assert before == pytest.approx(30 ** 0.5, abs=1e-5)
        assert reported == pytest.approx(before, abs=1e-5)
        assert after == pytest.approx(1.0, abs=1e-5)


def _gloo_reference_agreement_worker(rank, port, mode, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _next_batch_across_ranks
    from verl_speco.trainer.tq_sample_source import TQLocalBatch

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        token = f"dispatch-{rank}" if mode == "identity" else "dispatch"
        batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                             plan_id=1, receipt_token=token)
        source = iter([] if mode == "exhaust" and rank == 1 else [batch])
        try:
            _next_batch_across_ranks(source, rank=rank, device=torch.device("cpu"), require_prepared=True)
            results.put((rank, "ok"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("mode,expected", [("match", "ok"), ("identity", "dispatch identity differs"),
                                           ("exhaust", "exhausted inconsistently")])
def test_real_two_rank_reference_readiness_agreement(mode, expected):
    outputs = _collect_gloo_workers(_gloo_reference_agreement_worker,
        lambda rank, port, results: (rank, port, mode, results), timeout=40)
    assert {rank for rank, _ in outputs} == {0, 1}
    assert all(expected in message for _, message in outputs)


def _gloo_checkpoint_publish_worker(rank, port, marker, entered, proceed, results, fail_rank, wait):
    from concurrent.futures import Future
    from datetime import timedelta
    import json
    import torch.distributed as dist
    import verl_speco.trainer.draft_training_loop as loop

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        class Trainer:
            _pending_full_checkpoint_future = None
            _standalone_export_error = None
            def save_checkpoint(self, step, *, wait, defer_completion):
                assert defer_completion
                if rank == 0:
                    Path(marker).write_text(json.dumps({"complete": False}), encoding="utf-8")
                    if not wait:
                        self._pending_full_checkpoint_future = Future()
                        self._pending_full_checkpoint_future.set_result(None)
                    entered.set()
                    return {"saved": True, "path": marker, "reason": "saved" if wait else "scheduled"}
                assert proceed.wait(15)
                if fail_rank:
                    raise OSError("injected rank save failure")
                return {"saved": True, "path": marker, "reason": "optimizer_shard_saved"}
        def finalize(trainer, path, **kwargs):
            Path(path).write_text(json.dumps({"complete": True}), encoding="utf-8")
        loop._finalize_standalone_checkpoint_files = finalize
        try:
            loop._save_standalone_checkpoint_across_ranks(Trainer(), 1, device=torch.device("cpu"), wait=wait)
            results.put((rank, "ok"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("fail_rank,wait", [(False, False), (True, False), (True, True)])
def test_real_two_rank_checkpoint_publication_waits_for_all(tmp_path, fail_rank, wait):
    import json
    import multiprocessing
    import socket

    context = multiprocessing.get_context("spawn")
    marker = str(tmp_path / "metadata.json")
    entered, proceed, results = context.Event(), context.Event(), context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_gloo_checkpoint_publish_worker,
        args=(rank, port, marker, entered, proceed, results, fail_rank, wait)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        assert entered.wait(30)
        assert json.loads(Path(marker).read_text(encoding="utf-8"))["complete"] is False
        proceed.set()
        outcomes = [results.get(timeout=30) for _ in processes]
        assert {rank for rank, _ in outcomes} == {0, 1}
        assert all(("save failed" in value) if fail_rank else value == "ok" for _, value in outcomes)
        assert json.loads(Path(marker).read_text(encoding="utf-8"))["complete"] is (not fail_rank)
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        proceed.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


def _gloo_directory_lease_worker(rank, port, directory, mode, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _acquire_checkpoint_directory_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        path = directory if mode != "not_shared" or rank == 0 else directory + "-missing"
        try:
            lease = _acquire_checkpoint_directory_across_ranks(
                path, not (mode == "different_setting" and rank == 1), rank=rank, device=torch.device("cpu"))
            assert mode == "shared"
            assert lease is not None
            lease.verify()
            dist.barrier()
            if rank == 0:
                lease.release()
            results.put((rank, "ok"))
        except (RuntimeError, ValueError):
            assert mode != "shared"
            results.put((rank, "rejected"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("mode", ["shared", "not_shared", "different_setting"])
def test_checkpoint_directory_ownership_agrees_across_ranks(tmp_path, mode):
    outputs = _collect_gloo_workers(_gloo_directory_lease_worker,
        lambda rank, port, results: (rank, port, str(tmp_path), mode, results))
    assert set(outputs) == {(0, "ok" if mode == "shared" else "rejected"),
                            (1, "ok" if mode == "shared" else "rejected")}
    assert (tmp_path / ".standalone-checkpoint-owner").exists() == (mode == "not_shared")


def _gloo_input_binding_worker(rank, port, source, results):
    from datetime import timedelta
    import torch.distributed as dist
    import verl_speco.trainer.draft_training_loop as loop
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        identity = loop._input_identity_across_ranks(source if rank == 0 else source + ".missing", 2)
        assert loop._input_identity_across_ranks(source + ".missing", 2) is None
        if rank == 0:
            def fail(*args, **kwargs):
                raise PermissionError("injected access error")
            loop.build_input_fingerprint = fail
        with pytest.raises(RuntimeError, match="Failed to bind standalone input"):
            loop._input_identity_across_ranks(source, 2)
        from verl_speco.trainer.tq_sample_source import TQLocalBatch
        batch_identity = dict(identity)
        if rank == 1:
            batch_identity["sha256"] = "0" * 64
        batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                             plan_id=0, receipt_token="receipt", input_identity=batch_identity)
        with pytest.raises(RuntimeError, match="target-feature producer failed"):
            loop._next_batch_across_ranks(iter([batch]), rank=rank, device=torch.device("cpu"),
                                         require_prepared=True, expected_input_identity=identity)
        batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                             plan_id=0, receipt_token="receipt", input_identity=identity,
                             assignment_digest=str(rank) * 64)
        with pytest.raises(RuntimeError, match="identity differs"):
            loop._next_batch_across_ranks(iter([batch]), rank=rank, device=torch.device("cpu"),
                                         require_prepared=True, expected_input_identity=identity)
        results.put((rank, identity))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_input_identity_and_failure_are_broadcast_to_all_ranks(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"input\n")
    gathered = _collect_gloo_workers(_gloo_input_binding_worker,
        lambda rank, port, results: (rank, port, str(source), results))
    assert {rank for rank, _ in gathered} == {0, 1}
    assert gathered[0][1] == gathered[1][1] and gathered[0][1]["kind"] == "file"


def _gloo_nonzero_checkpoint_writer_worker(rank, port, results):
    from datetime import timedelta
    import torch.distributed as dist
    import verl_speco.trainer.draft_training_loop as loop
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        finalized = []
        class Trainer:
            _standalone_export_error = None
            def __init__(self, partial=False):
                self.partial = partial
            def save_checkpoint(self, step, *, wait, defer_completion):
                if self.partial and rank == 0:
                    return {"saved": False, "reason": "missing_checkpoint_dir"}
                return {"saved": True, "path": "unused", "reason": "saved" if rank == 1 else "optimizer_shard_saved"}
        def finalize(trainer, path, **kwargs):
            snapshot = kwargs["consumed_snapshot"]
            assert rank == 1 and torch.equal(snapshot, torch.arange(kwargs["step"], dtype=torch.int64))
            finalized.append(kwargs["step"])
        loop._finalize_standalone_checkpoint_files = finalize
        for size in (0, 65539):
            progress = set(range(size)) if rank == 0 else {999999}
            loop._save_standalone_checkpoint_across_ranks(Trainer(), size, device=torch.device("cpu"),
                                                          wait=True, consumed_sequence_nos=progress)
        with pytest.raises(RuntimeError, match="not saved on every rank"):
            loop._save_standalone_checkpoint_across_ranks(Trainer(partial=True), 1,
                device=torch.device("cpu"), wait=True, consumed_sequence_nos={0} if rank == 0 else set())
        original_empty = loop.torch.empty
        if rank == 1:
            def fail_target(*args, **kwargs):
                if args and args[0] == 65539 and kwargs.get("dtype") == torch.int64 and "device" not in kwargs:
                    raise MemoryError("injected progress allocation failure")
                return original_empty(*args, **kwargs)
            loop.torch.empty = fail_target
        try:
            with pytest.raises(RuntimeError, match="allocate checkpoint progress transfer"):
                loop._save_standalone_checkpoint_across_ranks(Trainer(), 65539, device=torch.device("cpu"),
                    wait=True, consumed_sequence_nos=set(range(65539)) if rank == 0 else set())
        finally:
            loop.torch.empty = original_empty
        results.put((rank, finalized))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_nonzero_checkpoint_writer_receives_rank_zero_progress():
    outputs = _collect_gloo_workers(_gloo_nonzero_checkpoint_writer_worker,
        lambda rank, port, results: (rank, port, results))
    assert dict(outputs) == {0: [], 1: [0, 65539]}


def _gloo_real_checkpoint_worker(rank, port, directory, source, release, entered, results, fail_rank):
    from concurrent.futures import Future
    from datetime import timedelta
    import json
    from types import SimpleNamespace
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from verl_speco.trainer.draft_training_loop import _save_standalone_checkpoint_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        checkpoint = Path(directory) / "draft_step_1"
        class Trainer:
            _pending_full_checkpoint_future = None
            _standalone_retention_keep_last = 2
            _standalone_retention_run_id = "a" * 32
            backend = SimpleNamespace(model_type="eagle3")
            def save_checkpoint(self, step, *, wait, defer_completion):
                assert step == 1 and defer_completion and not wait
                metadata = dcp.save({"weights": torch.tensor([7])}, checkpoint_id=str(checkpoint / "optimizer"))
                if rank == 0:
                    from verl_speco.trainer.checkpoint import write_standalone_optimizer_layout
                    write_standalone_optimizer_layout(checkpoint / "optimizer", metadata)
                    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
                    torch.save({"weight": torch.tensor([7])}, checkpoint / "pytorch_model.bin")
                    torch.save({"optimizer_steps_total": 1}, checkpoint / "optimizer" / "trainer_state.pt")
                    (checkpoint / "metadata.json").write_text(json.dumps({
                        "step": 1, "complete": False,
                        "optimizer": {"format": "torch_distributed_checkpoint", "path": "optimizer",
                                      "trainer_state_file": "trainer_state.pt"}}), encoding="utf-8")
                    self._pending_full_checkpoint_future = Future()
                    self._pending_full_checkpoint_future.set_result(None)
                    entered.set()
                    return {"saved": True, "path": str(checkpoint), "reason": "scheduled"}
                assert release.wait(20)
                if fail_rank:
                    raise OSError("injected post-DCP failure")
                return {"saved": True, "path": str(checkpoint), "reason": "optimizer_shard_saved"}
        try:
            _save_standalone_checkpoint_across_ranks(
                Trainer(), 1, device=torch.device("cpu"), consumed_sequence_nos={0, 2} if rank == 0 else set(),
                input_path=source)
            restored = {"weights": torch.tensor([0])}
            dcp.load(restored, checkpoint_id=str(checkpoint / "optimizer"))
            results.put((rank, int(restored["weights"].item())))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("fail_rank", [False, True])
def test_real_dcp_checkpoint_publication_and_recovery(tmp_path, fail_rank):
    import json
    import multiprocessing
    import socket
    from verl_speco.trainer.checkpoint import is_pretrained_drafter_checkpoint, verify_standalone_checkpoint_integrity
    from verl_speco.trainer.standalone_resume import load_standalone_resume

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "draft_step_1"
    context = multiprocessing.get_context("spawn")
    entered, release, results = context.Event(), context.Event(), context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_gloo_real_checkpoint_worker,
        args=(rank, port, str(tmp_path), str(source), release, entered, results, fail_rank)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        assert entered.wait(40)
        assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["complete"] is False
        release.set()
        outputs = [results.get(timeout=40) for _ in processes]
        assert {rank for rank, _ in outputs} == {0, 1}
        if fail_rank:
            assert all("save failed" in value for _, value in outputs)
            assert not is_pretrained_drafter_checkpoint(checkpoint)
            assert not (checkpoint / "standalone_resume.json").exists()
        else:
            assert {value for _, value in outputs} == {7}
            assert is_pretrained_drafter_checkpoint(checkpoint)
            assert verify_standalone_checkpoint_integrity(checkpoint, require_optimizer_layout=True)
            assert load_standalone_resume(checkpoint, input_path=source)[0] == {0, 2}
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


def _gloo_materialization_ack_worker(rank, port, fail_send, events, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _ack_reference_materialization_across_ranks, _next_batch_across_ranks
    from verl_speco.trainer.tq_sample_source import TQLocalBatch

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        class Events:
            def put(self, event):
                if fail_send:
                    raise ConnectionError("ack transport failure")
                events.put(event)
        batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                             plan_id=2, receipt_token="dispatch")
        batch = _next_batch_across_ranks(iter([batch]), rank=rank, device=torch.device("cpu"), require_prepared=True)
        try:
            _ack_reference_materialization_across_ranks(batch, rank=rank, device=torch.device("cpu"),
                events=Events(), config={"io_retry_max_attempts": 1, "io_timeout_seconds": 1})
            results.put((rank, "ready for training"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("fail_send", [False, True])
def test_real_two_rank_materialization_acknowledgement(fail_send):
    import multiprocessing
    import socket
    from queue import Empty

    context = multiprocessing.get_context("spawn")
    events, results = context.Queue(), context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_gloo_materialization_ack_worker,
        args=(rank, port, fail_send, events, results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        outputs = [results.get(timeout=40) for _ in processes]
        assert {rank for rank, _ in outputs} == {0, 1}
        if fail_send:
            assert all("acknowledgement failed" in value for _, value in outputs)
            with pytest.raises(Empty):
                events.get(timeout=0.5)
        else:
            assert all(value == "ready for training" for _, value in outputs)
            assert events.get(timeout=5) == {"kind": "training_materialized", "plan_id": 2,
                                              "receipt_token": "dispatch"}
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        events.close()
        results.close()


def _gloo_reference_step_worker(rank, port, source, failure, results):
    import asyncio
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _reference_training_step_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        class Trainer:
            def _stage_standalone_cpu_batch(self, batch):
                if rank == 1 and failure == "stage":
                    raise RuntimeError("injected staging failure")
                return dict(batch)
            async def training_step_from_batch(self, batch, step, *, raise_on_error):
                assert raise_on_error and batch["_speco_reference_staged"] is True
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()
                if rank == 1 and failure == "raise":
                    raise RuntimeError("injected optimizer failure")
                return not (rank == 1 and failure == "false")
        try:
            asyncio.run(_reference_training_step_across_ranks(Trainer(), {"hidden_source": "standalone_vllm_reference"},
                                                               0, device=torch.device("cpu")))
            if rank == 0:
                Path(source).unlink()
            results.put((rank, "committed"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("failure", ["none", "raise", "false", "stage"])
def test_real_two_rank_reference_step_gates_release(tmp_path, failure):
    source = tmp_path / "reference"
    source.write_bytes(b"payload")
    outputs = _collect_gloo_workers(_gloo_reference_step_worker,
        lambda rank, port, results: (rank, port, str(source), failure, results), timeout=40)
    assert {rank for rank, _ in outputs} == {0, 1}
    if failure == "none":
        assert all(value == "committed" for _, value in outputs)
        assert not source.exists()
    else:
        expected = "device staging failed" if failure == "stage" else "training failed on at least one rank"
        assert all(expected in value for _, value in outputs)
        assert source.exists()


def _gloo_commit_gate_worker(rank, port, failure, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _drain_batch_finalizer_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    finalizer = AsyncBatchFinalizer(1) if rank == 0 else None
    release = Event()
    try:
        if finalizer is not None:
            def finish_previous():
                if failure == "timeout":
                    assert release.wait(10)
                elif failure:
                    raise OSError("injected finalization failure")
            finalizer.submit(finish_previous)
        try:
            _drain_batch_finalizer_across_ranks(finalizer, torch.device("cpu"),
                                                timeout=0.1 if failure == "timeout" else 5)
            results.put((rank, "next step allowed"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        if finalizer is not None:
            release.set()
            try:
                finalizer.close()
            except OSError:
                pass
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("failure", [False, True, "timeout"])
def test_real_two_rank_previous_finalization_gates_next_step(failure):
    outputs = _collect_gloo_workers(_gloo_commit_gate_worker,
        lambda rank, port, results: (rank, port, failure, results), timeout=40)
    assert {rank for rank, _ in outputs} == {0, 1}
    if failure:
        assert all("finaliz" in value.lower() for _, value in outputs)
    else:
        assert all(value == "next step allowed" for _, value in outputs)


def _gloo_checkpoint_timeout_worker(rank, port, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _sync_standalone_export_error
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    trainer = SimpleNamespace(_standalone_finalization_event=Event() if rank == 0 else None,
                              _standalone_checkpoint_timeout=0.1)
    try:
        try:
            _sync_standalone_export_error(trainer, torch.device("cpu"))
            results.put((rank, "unexpected success"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_checkpoint_finalization_timeout_stops_both_gloo_ranks():
    outputs = _collect_gloo_workers(_gloo_checkpoint_timeout_worker,
                                     lambda rank, port, results: (rank, port, results), timeout=40)
    assert {rank for rank, _ in outputs} == {0, 1}
    assert all("checkpoint export failed" in value.lower() for _, value in outputs)


def _gloo_checkpoint_save_timeout_worker(rank, port, results):
    from concurrent.futures import Future
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _save_standalone_checkpoint_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    class Trainer:
        checkpoint_dir = "unused"
        _pending_full_checkpoint_future = None
        _standalone_checkpoint_timeout = 0.1
        def _save_checkpoint_async(self, step):
            self._pending_full_checkpoint_future = Future()
            if rank:
                self._pending_full_checkpoint_future.set_result(None)
            return self._pending_full_checkpoint_future
    trainer = Trainer()
    try:
        try:
            _save_standalone_checkpoint_across_ranks(trainer, 1, device=torch.device("cpu"), wait=True)
            results.put((rank, "unexpected success", False))
        except RuntimeError as exc:
            pending = trainer._pending_full_checkpoint_future
            results.put((rank, str(exc), pending is not None and not pending.done()))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_checkpoint_save_timeout_stops_both_gloo_ranks():
    outputs = _collect_gloo_workers(_gloo_checkpoint_save_timeout_worker,
                                     lambda rank, port, results: (rank, port, results), timeout=40)
    assert {rank for rank, _, _ in outputs} == {0, 1}
    assert all("save failed on at least one rank" in value for _, value, _ in outputs)
    assert next(pending for rank, _, pending in outputs if rank == 0)
