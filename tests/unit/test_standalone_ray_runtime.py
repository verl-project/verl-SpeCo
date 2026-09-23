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


def test_producer_actor_reuses_existing_producer_and_supports_stop(monkeypatch) -> None:
    started = asyncio.Event()

    async def fake_run_producer(config, **kwargs):
        assert config == {"producer": True}
        assert set(kwargs) == {
            "before_request",
            "on_published",
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

    def get(self, *, block, timeout):
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

    def get(self, *, block, timeout):
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
