# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Execute production methods without the optional verl training stack."""
import ast
import asyncio
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from threading import Event, current_thread, get_ident
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
import torch

from verl_speco.trainer.standalone_batch_finalize import AsyncBatchFinalizer
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader, TQPrefetchIterator

ROOT = Path(__file__).resolve().parents[2]


def _trainer():
    path = ROOT / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DrafterBaseTrainer")
    names = {
        "_pack_training_diagnostics", "_accumulate_training_diagnostics", "_summarize_step_metrics",
        "_record_dflash_training_metrics",
        "_reduce_training_metric", "_reduce_loss_statistics", "_reduce_loss_metrics",
        "_reduce_standalone_step_statistics", "_training_step_on_batch", "_clip_training_grad_norm",
        "_sanitize_sequence_tensor", "_mask_loss_for_bad_inputs", "_sanitize_training_batch",
    }
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    calls = []

    def reduce(value, group=None):
        calls.append((group, value.dtype))
        value.mul_(2)

    namespace = dict(torch=torch, logging=logging, logger=logging.getLogger("overhead-test"),
                     Any=Any, Optional=Optional, cast=cast, time=time, device_name="cpu",
                     dist=SimpleNamespace(all_reduce=reduce))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    trainer = namespace[cls.name]()
    trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
    trainer.training_device_mesh = None
    trainer._get_sp_group = lambda: "sp"
    trainer._get_dp_group = lambda: "dp"
    trainer._get_sp_world_size = trainer._get_dp_world_size = lambda: 2
    trainer._block_drafter_metric_prefix = lambda: "dspark"
    return trainer, calls


@pytest.mark.parametrize("dtype,expected_calls", [(torch.float32, 2), (torch.float64, 4)])
def test_combined_statistics_preserve_values_groups_and_precision(dtype, expected_calls):
    legacy, legacy_calls = _trainer()
    combined, calls = _trainer()
    values = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (2., 3., 5.)]
    data = dict(zip(("total_local_vloss", "total_local_ploss", "local_num_tokens"), values))
    data["diagnostics"] = {"correct_count": torch.tensor(2.),
                           "loss_sum_per_position": torch.tensor([1., 2., 3.])}
    legacy._record_dflash_training_metrics(data)
    expected = legacy._reduce_loss_metrics(*values)
    result = combined._reduce_standalone_step_statistics(data)
    assert len(legacy_calls) == 4 and len(calls) == expected_calls
    assert [group for group, _ in calls] == ["sp", "dp"] * (expected_calls // 2)
    assert result[3] == expected[3] == 4
    for actual, old in zip(result[:3], expected[:3]):
        assert actual.dtype == dtype and not actual.requires_grad
        torch.testing.assert_close(actual, old)
    assert combined._training_metric_steps == legacy._training_metric_steps == 1
    assert combined._training_metric_sums == legacy._training_metric_sums


def test_combined_statistics_preserve_actual_optimizer_update():
    trainers = [_trainer()[0], _trainer()[0]]
    for index, trainer in enumerate(trainers):
        trainer.model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            trainer.model.weight.fill_(0.25)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.lr_scheduler = None
        trainer.training_steps = trainer.optimizer_steps_total = 0
        trainer.combine_standalone_metrics = bool(index)
        trainer._ulysses_group_context = nullcontext
        trainer.record_training_timing = lambda *args: None
        trainer._is_checkpoint_leader = lambda: False

        def compute_loss(model, batch, pad):
            loss = model(batch["hidden_states"]).float().square().sum()
            return {"total_local_vloss": loss, "total_local_ploss": loss * 0,
                    "local_num_tokens": torch.tensor(3.), "v_weight": 1., "p_weight": 0.,
                    "diagnostics": {"ce_loss_sum": loss.detach(), "correct_count": torch.tensor(2.)}}

        trainer.backend = SimpleNamespace(model_type="dspark", compute_loss=compute_loss,
                                          enable_standalone_training_metrics=True)
        for step in range(3):
            assert asyncio.run(trainer._training_step_on_batch({"hidden_states": torch.ones(3, 2)}, step))
    torch.testing.assert_close(trainers[0].model.weight, trainers[1].model.weight, rtol=0, atol=0)
    assert trainers[0]._training_metric_sums == trainers[1]._training_metric_sums


def test_sanitizing_does_not_read_debug_counters_back_to_cpu(monkeypatch):
    trainer, _ = _trainer()
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer._get_hidden_state_clip_value = lambda: None
    trainer.rank = 0
    batch = {"hidden_states": torch.tensor([[[float("nan"), 1.], [2., 3.]]]),
             "target_last_hidden_states": torch.tensor([[[1., 1.], [float("inf"), 2.]]]),
             "loss_mask": torch.ones(1, 2), "attention_mask": torch.ones(1, 2)}
    logger = logging.getLogger("overhead-test")
    monkeypatch.setattr(logger, "isEnabledFor", lambda level: False)

    def forbidden(*args, **kwargs):
        pytest.fail("DEBUG counters must not synchronize with CPU when DEBUG is disabled")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        result = trainer._sanitize_training_batch(batch)
    assert torch.isfinite(result["hidden_states"]).all()
    assert torch.isfinite(result["target_last_hidden_states"]).all()
    assert result["loss_mask"].tolist() == [[0., 0.]]
    assert result["attention_mask"].tolist() == [[0., 1.]]


def _metrics_function():
    path = ROOT / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_submit_standalone_metrics")
    namespace = dict(retry_io=lambda op, **kw: retry_io(op, **dict(kw, initial_backoff=0, max_backoff=0)),
                     put_queue=lambda queue, event, **kw: queue.put(event), logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def test_metrics_use_existing_background_executor_and_preserve_every_snapshot():
    started, release = Event(), Event()
    order, threads = [], []
    caller = get_ident()

    class Events:
        def put(self, event):
            threads.append(get_ident())
            started.set()
            assert release.wait(5)
            order.append(event["step"])

    finalizer = AsyncBatchFinalizer(3)
    try:
        finalizer.submit(lambda: order.append("ack"))
        send = _metrics_function()
        for step in (1, 2):
            send(finalizer, Events(), {"kind": "training_metrics", "step": step}, {})
        assert started.wait(5) and order == ["ack"]
        release.set()
        finalizer.drain()
        assert order == ["ack", 1, 2]
        assert len(set(threads)) == 1 and threads[0] != caller
    finally:
        release.set()
        finalizer.close()


def test_metrics_retry_and_report_permanent_failure():
    attempts, events = [], Queue()

    class Events:
        def put(self, event):
            if event["kind"] == "training_metrics":
                attempts.append(1)
                raise TimeoutError("metrics queue unavailable")
            events.put(event)

    finalizer = AsyncBatchFinalizer(2)
    _metrics_function()(finalizer, Events(), {"kind": "training_metrics", "step": 9}, {})
    with pytest.raises(TimeoutError):
        finalizer.drain()
    with pytest.raises(TimeoutError):
        finalizer.close()
    assert len(attempts) == 3
    assert events.get_nowait() == {"kind": "finalization_failed", "phase": "metrics", "step": 9}


def test_pin_thread_limits_its_threads_without_changing_main_thread(monkeypatch):
    before = torch.get_num_threads()
    calls = []
    original = torch.set_num_threads

    def set_threads(count):
        calls.append((current_thread().name, count))
        original(count)

    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    commands = Queue()
    commands.put({"kind": "batch", "assignments": [[{"key": "k", "tag": {"sequence_no": 0}}]],
                  "global_keys": ["k"], "global_sequence_nos": [0]})
    commands.put({"kind": "stop"})
    store = SimpleNamespace(connect=lambda: None, owner_ready=lambda: True)
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
                                 prepare_batch=lambda entries, **kw: {"hidden_states": torch.ones(1)})
    source = TQPrefetchIterator(loader, 2, pin_memory=True, prepare_workers=0)
    try:
        assert next(source).prepared_batch is not None
    finally:
        source.close()
    assert ("file-pin-0", 1) in calls
    assert torch.get_num_threads() == before
