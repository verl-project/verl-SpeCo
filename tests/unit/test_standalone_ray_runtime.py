# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import sys
from types import SimpleNamespace

import pytest

from verl_speco.standalone_ray_runtime import (
    StandaloneProducerActor,
    StandaloneRayTrainer,
    _configure_driver_logging,
)
from verl_speco.standalone_tq_training_launcher import (
    PipelineCommands,
    _validate_runtime_backend_topology,
)


def _commands() -> PipelineCommands:
    return PipelineCommands(
        vllm=None,
        vllm_endpoints=("http://127.0.0.1:8000/v1",),
        owner=["python", "-m", "verl_speco.tq_owner"],
        producer=["producer"],
        consumer=["consumer"],
        producer_overrides=("producer=true",),
        consumer_overrides=("consumer=true",),
    )


def test_reference_events_retry_same_batch_id_and_use_full_batch():
    class Queue:
        def __init__(self):
            self.items = []

        def put(self, event):
            self.items.append(event)
            if len(self.items) == 1:
                raise ConnectionError("Delivery succeeded but acknowledgement was lost")

    queue = Queue()
    actor = StandaloneProducerActor({}, queue, reference_batch_size=64)
    assert actor._notification_batch_size == 64

    async def run():
        first = await actor._on_references([{"key": "first", "tag": {}}])
        second = await actor._on_references([{"key": "second", "tag": {}}])
        assert first["attempts"] == 2 and second["attempts"] == 1
        assert first["retry_wait_seconds"] > 0 and second["retry_wait_seconds"] == 0

    asyncio.run(run())
    assert [event["batch_id"] for event in queue.items] == [0, 0, 1]
    assert queue.items[0] == queue.items[1]


def test_file_actor_sends_completion_after_all_metadata_and_metrics(monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer

    config = _runtime_config()
    training = config.actor_rollout_ref.rollout.drafter.training
    monkeypatch.setattr(training, "feature_store", SimpleNamespace(type="ray_reference"), raising=False)

    async def run(config, **kwargs):
        assert kwargs["reference_batch_size"] == 64
        assert kwargs["on_published"] is None
        await kwargs["on_references"]([{"key": "first", "tag": {}}])
        await kwargs["on_references"]([{"key": "tail", "tag": {}}])
        await kwargs["on_metrics"](2, {"perf/test": 1.0})
        return producer.ProducerStats(published_count=2)

    monkeypatch.setattr(producer, "run_producer", run)
    events = Queue()
    actor = StandaloneProducerActor(config, events, reference_batch_size=64)
    result = asyncio.run(actor._run_producer())
    messages = [events.get_nowait() for _ in range(events.qsize())]
    assert result.published_count == 2
    assert [event["kind"] for event in messages] == [
        "samples_ready", "samples_ready", "producer_metrics", "producer_done",
    ]


def test_ray_reference_pipeline_never_starts_tq_and_preserves_untrained_files(tmp_path, monkeypatch):
    from verl_speco import standalone_ray_runtime as runtime

    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    untrained = hidden_dir / "untrained.safetensors"
    untrained.write_bytes(b"untrained")
    commands = _commands()
    commands = PipelineCommands(
        **{**commands.__dict__, "consumer_overrides": ("training.feature_store.type=ray_reference",)}
    )
    original_mkdtemp = runtime.tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") == "speco-vllm-hidden-states-":
            return str(hidden_dir)
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(runtime.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(runtime, "validate_tq_backend", lambda *args: pytest.fail("TQ validation"))
    monkeypatch.setattr(runtime, "run_ray_actors", lambda *args, **kwargs: 0)
    result = runtime.run_ray_pipeline(
        commands, ray_module=object(), ray_address="unused", environ={},
        endpoint_ready=lambda _: True,
        popen=lambda *args, **kwargs: pytest.fail("Started a TQ owner"),
    )
    assert result == 0 and untrained.exists()


def test_producer_actor_reuses_existing_producer_and_supports_stop(monkeypatch) -> None:
    started = asyncio.Event()

    async def fake_run_producer(config, **kwargs):
        assert config == {"producer": True}
        assert set(kwargs) == {
            "before_request",
            "on_published",
            "on_metrics",
            "get_runtime_state",
        }
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    actor = StandaloneProducerActor({"producer": True})

    async def exercise():
        assert await actor.start() == {"started": True}
        await started.wait()
        assert await actor.start() == {"started": False}
        assert await actor.stop() == {"stopped": True}

    asyncio.run(exercise())


def test_producer_actor_coalesces_publish_notifications_as_cumulative_total(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class _Stats:
        published_count: int

    class _Events:
        def __init__(self):
            self.items = []

        def put(self, value):
            self.items.append(value)

    async def fake_run_producer(config, **kwargs):
        for sequence_no in range(3):
            await kwargs["on_published"](sequence_no)
        return _Stats(published_count=3)

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    events = _Events()
    actor = StandaloneProducerActor({"producer": True}, events)

    async def exercise():
        assert await actor.start() == {"started": True}
        assert await actor.wait() == {"published_count": 3}

    asyncio.run(exercise())

    assert events.items == [
        {"kind": "samples_published", "published_total": 3}
    ]


def test_producer_actor_delivers_partial_metrics_when_stopped(monkeypatch):
    started = asyncio.Event()
    events = []

    async def fake_run_producer(config, **kwargs):
        try:
            await kwargs["on_published"](0)
            started.set()
            await asyncio.Event().wait()
        finally:
            await kwargs["on_metrics"](1, {"producer/sample_total_time": 2.0})

    monkeypatch.setitem(sys.modules, "verl_speco.standalone_tq_producer",
                        SimpleNamespace(run_producer=fake_run_producer))
    actor = StandaloneProducerActor({"producer": True}, SimpleNamespace(put=events.append))

    async def exercise():
        await actor.start()
        await started.wait()
        await actor.stop()

    asyncio.run(exercise())
    assert {"kind": "producer_metrics", "published_total": 1,
            "metrics": {"producer/sample_total_time": 2.0}} in events


def test_ray_backend_rejects_multi_node_local_runtime() -> None:
    with pytest.raises(ValueError, match="nnodes=1"):
        _validate_runtime_backend_topology(
            "ray", ["speco.draft_training.nnodes=2"]
        )

    _validate_runtime_backend_topology(
        "subprocess", ["speco.draft_training.nnodes=2"]
    )


@dataclass(frozen=True)
class _Ref:
    name: str


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self):
        return self.callback()


class _ProducerHandle:
    def __init__(self):
        self.stopped = False
        self.start = _RemoteMethod(lambda: _Ref("start"))
        self.wait = _RemoteMethod(lambda: _Ref("producer"))
        self.stop = _RemoteMethod(self._stop)

    def _stop(self):
        self.stopped = True
        return _Ref("stop")


class _RemoteProducerClass:
    def __init__(self, handle):
        self.handle = handle

    def options(self, **kwargs):
        assert kwargs["name"] == "speco_standalone_producer"
        return self

    def remote(self, config, events):
        assert config is _runtime_config()
        return self.handle


class _WorkerGroup:
    def run_standalone_training(self):
        return [_Ref("consumer-0"), _Ref("consumer-1")]


class _FakeRay:
    def __init__(self, producer):
        self.producer = producer

    def remote(self, **kwargs):
        assert kwargs == {"max_concurrency": 4}
        return lambda cls: _RemoteProducerClass(self.producer)

    def get(self, ref):
        return {"ref": ref.name}

    def wait(self, refs, num_returns, timeout=0):
        assert num_returns == 1
        producer = next((ref for ref in refs if ref.name == "producer"), None)
        completed = producer or next(
            ref for ref in refs if ref.name.startswith("consumer")
        )
        return [completed], [ref for ref in refs if ref != completed]


class _Queue:
    def put(self, value):
        pass

    def get(self, *, block, timeout=None):
        if not block:
            from queue import Empty
            raise Empty
        raise AssertionError("event queue should not be read in this test")


class _Store:
    def connect(self):
        pass

    def list_ready(self):
        return []

    def close(self):
        pass


class _ReadyStore(_Store):
    def __init__(self):
        self.ready = [
            SimpleNamespace(key=f"sample-{index}", tag={"sequence_no": index})
            for index in range(4)
        ]

    def list_ready(self):
        return list(self.ready)


class _RecordingQueue(_Queue):
    def __init__(self, *, events=None, on_event=None):
        self.items = [] if events is None else list(events)
        self.on_event = on_event

    def put(self, value):
        self.items.append(value)

    def get(self, *, block, timeout=None):
        if not self.items:
            from queue import Empty

            raise Empty
        value = self.items.pop(0)
        if self.on_event is not None:
            self.on_event(value)
        return value


class _FlowRay(_FakeRay):
    def __init__(self, producer):
        super().__init__(producer)
        self.wait_calls = 0

    def wait(self, refs, num_returns, timeout=0):
        self.wait_calls += 1
        if self.wait_calls == 1:
            return [], list(refs)
        producer = next(ref for ref in refs if ref.name == "producer")
        return [producer], [ref for ref in refs if ref != producer]


def test_driver_stops_on_background_finalization_failure(monkeypatch):
    queues = iter((_RecordingQueue(), _RecordingQueue(events=[
        {"kind": "finalization_failed", "plan_id": 0},
    ])))
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=_FlowRay(_ProducerHandle()),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda config: _ReadyStore(),
    )
    with pytest.raises(RuntimeError, match="failed to finalize trained plan=0"):
        trainer.run()


def test_file_driver_kills_group_when_one_rank_reports_training_failure(monkeypatch):
    from copy import deepcopy

    config = deepcopy(_runtime_config())
    training = config.actor_rollout_ref.rollout.drafter.training
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.get = lambda key, default=None: getattr(training, key, default)
    training.transfer_queue = {"consumer_prefetch_batches": 2}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    queues = iter((_RecordingQueue(), _RecordingQueue(events=[
        {"kind": "consumer_failed", "rank": 1, "error": "injected optimizer failure"},
    ])))
    ray = _FlowRay(_ProducerHandle())
    killed = []
    ray.kill = lambda worker, **kw: killed.append((worker, kw))
    group = _WorkerGroup()
    group.workers = ["rank0", "rank1"]
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=ray, worker_group_factory=lambda *a, **k: group,
        queue_factory=lambda **kwargs: next(queues), feature_store_factory=lambda _: _ReadyStore(),
    )
    trainer.producer = ray.producer
    trainer.consumer_wg = group
    trainer.commands_queue, trainer.events_queue = next(queues), next(queues)
    original_wait = ray.wait
    ray.wait = lambda refs, **kw: ((refs, []) if refs[0].name == "stop" else original_wait(refs, **kw))
    with pytest.raises(RuntimeError, match="rank=1 failed: injected optimizer failure"):
        trainer.run()
    assert killed == [("rank0", {"no_restart": True}), ("rank1", {"no_restart": True})]


def test_scheduler_executes_one_complete_standalone_batch(monkeypatch) -> None:
    store = _ReadyStore()
    commands = _RecordingQueue()
    events = _RecordingQueue(
        events=[
            {
                "kind": "training_completed",
                "keys": [f"sample-{index}" for index in range(4)],
                "successful": True,
            }
        ],
        on_event=lambda event: store.ready.clear(),
    )
    queues = iter((commands, events))
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=_FlowRay(_ProducerHandle()),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=lambda: next(queues),
        feature_store_factory=lambda config: store,
    )

    assert trainer.run() == 0
    assert commands.items[0]["kind"] == "batch"
    assert commands.items[0]["global_keys"] == [
        f"sample-{index}" for index in range(4)
    ]
    assert commands.items[-1] == {"kind": "stop"}


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_scheduler_prefills_capacity_and_uses_cached_keys(monkeypatch, depth):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    config.actor_rollout_ref.rollout.drafter.training.transfer_queue = {"consumer_prefetch_batches": depth}
    # deepcopy preserves the fixture's get closure, so bind it to this object.
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    store.ready = [SimpleNamespace(key=f"sample-{i}", tag={"sequence_no": i}) for i in range(12)]
    list_calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (list_calls.append(1), original_list())[1]
    commands = _RecordingQueue()

    def on_event(event):
        if event["kind"] == "training_completed":
            if depth == 0:
                # No later plan is submitted until this batch completes.
                assert sum(c["kind"] == "batch" for c in commands.items) == event["plan_id"] + 1
            if event["plan_id"] == 0:
                assert sum(c["kind"] == "batch" for c in commands.items) == depth + 1
                assert len(list_calls) == 1  # all initial plans use one snapshot
            keys = set(event["keys"])
            store.ready = [e for e in store.ready if e.key not in keys]

    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        *[{"kind": "training_completed", "plan_id": i,
           "keys": [f"sample-{j}" for j in range(i * 4, (i + 1) * 4)], "successful": True}
          for i in range(3)],
    ], on_event=on_event)

    class EventRay(_FakeRay):
        def wait(self, refs, **kwargs):
            if events.items:
                return [], list(refs)
            return super().wait(refs, **kwargs)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=EventRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(), queue_factory=lambda: next(queues),
        feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    batches = [c for c in commands.items if c["kind"] == "batch"]
    assert [c["plan_id"] for c in batches] == [0, 1, 2]
    assert len({key for c in batches for key in c["global_keys"]}) == 12


@pytest.mark.parametrize("initial_step", [8, 9])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_scheduler_respects_restored_remaining_budget(monkeypatch, initial_step, depth):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.transfer_queue = {"consumer_prefetch_batches": depth}
    training.max_steps = 9
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    commands = _RecordingQueue()
    event_items = [{"kind": "consumer_ready", "initial_optimizer_step": initial_step}]
    if initial_step < 9:
        event_items.append({"kind": "training_completed", "plan_id": 0,
                            "keys": [f"sample-{i}" for i in range(4)], "successful": True})
    events = _RecordingQueue(events=event_items)

    class WaitingRay(_FakeRay):
        def wait(self, refs, **kwargs):
            return [], list(refs)

    queues = iter((commands, events))
    producer = _ProducerHandle()
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=WaitingRay(producer), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    assert sum(c["kind"] == "batch" for c in commands.items) == 9 - initial_step
    assert commands.items[-1] == {"kind": "stop"}
    assert producer.stopped


def test_fast_producer_watermarks_and_duplicate_events_do_not_block_training(monkeypatch):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    store.ready.clear()
    calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (calls.append(1), original_list())[1]
    commands = _RecordingQueue()
    published = 0
    list_count_before_full_publication = None

    def on_event(event):
        nonlocal published, list_count_before_full_publication
        if event["kind"] == "samples_published":
            total = event["published_total"]
            if total == 24:
                if list_count_before_full_publication is None:
                    list_count_before_full_publication = len(calls)
                else:
                    assert len(calls) == list_count_before_full_publication
            store.ready.extend(SimpleNamespace(key=f"sample-{i}", tag={"sequence_no": i})
                               for i in range(published, total))
            published = total
        elif event["kind"] == "training_completed":
            keys = set(event["keys"])
            store.ready = [e for e in store.ready if e.key not in keys]

    completions = [{"kind": "training_completed", "plan_id": i,
                    "keys": [f"sample-{j}" for j in range(i * 4, (i + 1) * 4)], "successful": True}
                   for i in range(6)]
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "samples_published", "published_total": 12},
        {"kind": "samples_published", "published_total": 24},
        {"kind": "samples_published", "published_total": 24},
        completions[0], completions[0], *completions[1:],
    ], on_event=on_event)
    producer = _ProducerHandle()
    actions = []
    producer.pause = _RemoteMethod(lambda: (actions.append("pause"), _Ref("pause"))[1])
    producer.resume = _RemoteMethod(lambda: (actions.append("resume"), _Ref("resume"))[1])

    class EventRay(_FakeRay):
        def wait(self, refs, **kwargs):
            if events.items:
                return [], list(refs)
            return super().wait(refs, **kwargs)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=EventRay(producer), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    assert actions == ["pause", "resume"]
    assert [c["plan_id"] for c in commands.items if c["kind"] == "batch"] == list(range(6))


def test_driver_logs_producer_snapshot_and_drains_final_metrics(monkeypatch):
    store = _ReadyStore()
    list_calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (list_calls.append(1), original_list())[1]
    commands = _RecordingQueue()
    def on_event(event):
        if event["kind"] == "producer_metrics":
            # One initialization list before this low-frequency metric snapshot.
            assert len(list_calls) == 1
        elif event["kind"] == "training_completed":
            assert len(list_calls) == 2
            store.ready.clear()

    events = _RecordingQueue(events=[
        {"kind": "producer_metrics", "published_total": 100,
         "metrics": {"producer/samples_per_second": 2.0}},
        {"kind": "training_completed", "keys": [f"sample-{i}" for i in range(4)],
         "successful": True},
        {"kind": "training_metrics", "step": 17, "metrics": {"dspark/loss": 0.5}},
    ], on_event=on_event)

    class MetricsRay(_FlowRay):
        def wait(self, refs, num_returns, timeout=0):
            self.wait_calls += 1
            if self.wait_calls <= 2:
                return [], list(refs)
            producer = next(ref for ref in refs if ref.name == "producer")
            return [producer], [ref for ref in refs if ref != producer]

    queues = iter((commands, events))
    calls = []
    module = "verl_speco.standalone_ray_runtime."
    monkeypatch.setattr(module + "compose_runtime_config", lambda _: _runtime_config())
    monkeypatch.setattr(module + "_build_standalone_tracking", lambda *a, **k: ["tracker"])
    monkeypatch.setattr(module + "_log_producer_tracking_metrics",
                        lambda trackers, metrics, **k: calls.append(("producer", dict(metrics), k)))
    monkeypatch.setattr(module + "_log_standalone_tracking_metrics",
                        lambda trackers, metrics, **k: calls.append(("train", dict(metrics), k)))
    monkeypatch.setattr(module + "_finish_standalone_tracking",
                        lambda trackers, **k: calls.append(("finish", trackers)))
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=MetricsRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    assert calls == [
        ("producer", {"producer/samples_per_second": 2.0, "tq/ready_samples": 4.0},
         {"published_total": 100}),
        ("train", {"dspark/loss": 0.5}, {"step": 17}),
        ("finish", ["tracker"]),
    ]


_CONFIG = None


def _runtime_config():
    global _CONFIG
    if _CONFIG is None:
        training = SimpleNamespace(batch_size_per_gpu=4, transfer_queue={})
        training.get = lambda key, default=None: getattr(training, key, default)
        draft_training = SimpleNamespace(
            nproc_per_node=1,
            nnodes=1,
            scheduler={},
        )
        draft_training.get = lambda key, default=None: getattr(
            draft_training, key, default
        )
        _CONFIG = SimpleNamespace(
            speco=SimpleNamespace(
                draft_training=draft_training,
                standalone_tq_producer=SimpleNamespace(max_pending_samples=20),
            ),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    drafter=SimpleNamespace(training=training)
                )
            ),
        )
    return _CONFIG


def test_queue_config_defaults_low_watermark_to_half_producer_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace())

    config = trainer._queue_config()

    assert config.global_batch_size == 4
    assert config.low_watermark_samples == 10
    assert config.high_watermark_samples == 20


def test_driver_logging_overrides_preinstalled_warning_handler(monkeypatch) -> None:
    monkeypatch.delenv("SPECO_STANDALONE_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]
        root.setLevel(logging.WARNING)

        _configure_driver_logging()

        assert root.level == logging.INFO
        assert handler.level == logging.INFO
        assert (
            logging.getLogger("verl_speco.standalone_ray_runtime").level
            == logging.INFO
        )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_driver_logging_supports_scheduler_debug(monkeypatch) -> None:
    monkeypatch.setenv("SPECO_STANDALONE_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    handler = logging.StreamHandler()
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]

        _configure_driver_logging()

        assert root.level == logging.DEBUG
        assert handler.level == logging.DEBUG
        assert module_logger.level == logging.DEBUG
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_trainer_finishes_after_producer_and_consumers_complete(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FakeRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    assert trainer.run() == 0
    assert producer.stopped is False


class _FailingRay(_FakeRay):
    def get(self, ref):
        if ref.name == "consumer-0":
            raise RuntimeError("consumer failed")
        return super().get(ref)


class _EarlyConsumerRay(_FakeRay):
    def wait(self, refs, num_returns, timeout=0):
        assert num_returns == 1
        consumer = next(ref for ref in refs if ref.name.startswith("consumer"))
        return [consumer], [ref for ref in refs if ref != consumer]

    def get(self, ref):
        if ref.name.startswith("consumer"):
            return {"optimizer_steps_total": 0}
        return super().get(ref)


def test_trainer_rejects_unexpected_early_consumer_exit(monkeypatch) -> None:
    producer = _ProducerHandle()
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=_EarlyConsumerRay(producer),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    with pytest.raises(RuntimeError, match="before receiving"):
        trainer.run()

    assert producer.stopped is True


def test_step_budget_survives_consumer_exit_before_completion_event(monkeypatch):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.step = 1
    training.get = lambda key, default=None: getattr(training, key, default)
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "training_completed", "plan_id": 0,
         "keys": [f"sample-{i}" for i in range(4)], "successful": True},
    ])
    commands = _RecordingQueue()

    class FinishedConsumerRay(_FakeRay):
        calls = 0

        def wait(self, refs, **kwargs):
            self.calls += 1
            if self.calls == 2:
                consumer = next(r for r in refs if r.name.startswith("consumer"))
                return [consumer], [r for r in refs if r != consumer]
            return [], list(refs)

        def get(self, ref):
            if ref.name.startswith("consumer"):
                return {"optimizer_steps_total": 1}
            return super().get(ref)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=FinishedConsumerRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: _ReadyStore(),
    )
    assert trainer.run() == 0
    assert [c["kind"] for c in commands.items] == ["batch", "stop"]


def test_trainer_propagates_consumer_error_and_stops_producer(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FailingRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        trainer.run()


@pytest.mark.parametrize("end", ["eos", "max_steps"])
def test_mooncake_driver_cleans_tail_and_late_shutdown_references(monkeypatch, end):
    from copy import deepcopy
    from verl_speco.producer import hidden_states_store
    from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore
    from verl_speco.transport.drafter_sample_protocol import (
        PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key,
    )

    config = deepcopy(_runtime_config())
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 2 if end == "eos" else 1
    training.max_steps = 0 if end == "eos" else 1
    training.transfer_queue = {"run_id": "run", "ray": {"address": "unused"}, "consumer_prefetch_batches": 1}
    training.target_feature_replay = {"hidden_states_store": {"backend": "mooncake", "mooncake": {"master_server_address": "host:50051"}}}
    deleted = []
    backend = SimpleNamespace(backend="mooncake", release=deleted.append, close=lambda: None)
    monkeypatch.setattr(hidden_states_store, "build_hidden_states_store", lambda cfg: backend)

    def entry(index):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(
            meta, backend="mooncake", handle=f"hs-{index}",
            request={"input_ids": [1, 2], "loss_mask": [0, 1], "position_ids": [0, 1],
                     "feature_positions": [0, 1], "draft_position_ids": [1, 2],
                     "vllm_prompt_token_ids": [1, 2], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"},
        )
        return {"key": make_sample_key(meta), "tag": tag}

    items = [{"kind": "consumer_ready", "initial_optimizer_step": 0},
             {"kind": "samples_ready", "batch_id": 0, "entries": [entry(0)]}]
    if end == "eos":
        items.append({"kind": "producer_done"})
        expected = ["hs-0"]
    else:
        # Budget allows only plan 0; the other references must be discarded.
        items[1]["entries"].append(entry(1))
        items.extend([
            {"kind": "training_completed", "plan_id": 0, "keys": [entry(0)["key"]], "successful": True},
            {"kind": "samples_ready", "batch_id": 1, "entries": [entry(2)]},
        ])
        expected = ["hs-1", "hs-2"]
    commands, events = _RecordingQueue(), _RecordingQueue(events=items)

    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return refs[:1], refs[1:]
            if events.items:
                return [], refs
            consumers = [ref for ref in refs if ref.name.startswith("consumer")]
            return consumers[:1], [ref for ref in refs if ref not in consumers[:1]]

    configs = []
    def factory(cfg):
        configs.append(cfg)
        return ReferenceFeatureStore.from_config(cfg)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=Ray(_ProducerHandle()), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda **kw: next(queues), feature_store_factory=factory,
    )
    trainer.producer = trainer.ray.producer
    trainer.consumer_wg = _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    assert trainer.run() == 0
    assert configs[0]["hidden_states_store"]["mooncake"]["master_server_address"] == "host:50051"
    assert sorted(deleted) == expected
