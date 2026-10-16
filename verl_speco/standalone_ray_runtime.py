# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Ray execution backend for standalone Producer/TQ/Consumer training.

This module deliberately owns execution only. Scheduling and backpressure are
introduced in later phases; the training worker calls the existing standalone
loop without changing its numerical path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verl_speco.config import prefetch_depth
from verl_speco.standalone_tq_training_launcher import (
    _VLLM_HIDDEN_STATES_DIR,
    PipelineCommands,
    _stop_process,
    _vllm_is_ready,
    _wait_for_owner_ready,
    _wait_for_vllm_ready,
    validate_hidden_states_store,
    validate_tq_backend,
)
from verl_speco.trainer.scheduler import (
    CallbackStandaloneCollectionExecutor,
    CallbackStandaloneTrainingExecutor,
    DrafterScheduler,
    QueueScheduleConfig,
    QueueScheduleContext,
    QueueStatus,
    StandaloneQueueState,
)
from verl_speco.trainer.standalone_tracking import (
    _build_standalone_tracking,
    _finish_standalone_tracking,
    _log_producer_tracking_metrics,
    _log_standalone_tracking_metrics,
)
from verl_speco.trainer.tq_retry import retry_io

logger = logging.getLogger(__name__)
_SCHEDULER_INFO_INTERVAL_STEPS = 50
_PRODUCER_NOTIFICATION_MAX_BATCH = 32
_PRODUCER_NOTIFICATION_MAX_DELAY_SECONDS = 0.2


def put_queue(queue: Any, item: Any, *, timeout: float = 30.0) -> None:
    """Bound the Ray RPC too: Queue.put(timeout) only bounds queue capacity.

    Callers retry the same ID/cumulative event, making ambiguous delivery safe.
    Local queues keep their usual API for tests and single-process adapters.
    """
    actor = getattr(queue, "actor", None)
    if actor is None:
        queue.put(item)
        return
    import ray

    ref = actor.put.remote(item, timeout)
    try:
        ray.get(ref, timeout=timeout + 5)
    except ray.exceptions.GetTimeoutError:
        ray.cancel(ref, force=False)
        raise


def get_queue(queue: Any, *, timeout: float = 0.5, rpc_timeout: float = 30.0) -> Any:
    actor = getattr(queue, "actor", None)
    if actor is None:
        return queue.get(block=True, timeout=timeout)
    import ray

    ref = actor.get.remote(timeout)
    try:
        return ray.get(ref, timeout=rpc_timeout + 5)
    except ray.exceptions.GetTimeoutError:
        ray.cancel(ref, force=False)
        # A timed-out get may already have removed an item. Do not issue a new
        # get and silently skip that plan; terminate with its keys uncleared.
        raise RuntimeError(
            "Ray queue receive RPC did not respond to its deadline"
        ) from None


def _configure_driver_logging() -> None:
    """Keep Driver INFO logs visible after Ray/Hydra install handlers."""

    level_name = os.environ.get("SPECO_STANDALONE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(
            "SPECO_STANDALONE_LOG_LEVEL must be a standard Python logging level"
        )
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level)
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
    logger.disabled = False
    logger.setLevel(level)


def compose_runtime_config(overrides: Sequence[str]):
    """Compose the same Hydra config used by the former module entrypoints."""

    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="verl_speco.config", version_base=None):
        return compose(config_name="draft_trainer", overrides=list(overrides))


def _producer_notification_batch_size(config: Any) -> int:
    """Bound publish notifications while retaining one-batch responsiveness."""

    try:
        training = config.actor_rollout_ref.rollout.drafter.training
        runtime = config.speco.draft_training
        global_batch_size = (
            int(training.batch_size_per_gpu)
            * int(runtime.nproc_per_node)
            * int(runtime.nnodes)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        global_batch_size = _PRODUCER_NOTIFICATION_MAX_BATCH
    return max(1, min(global_batch_size, _PRODUCER_NOTIFICATION_MAX_BATCH))


class StandaloneProducerActor:
    """Async Ray actor facade around the existing standalone Producer."""

    def __init__(self, config: Any, events: Any | None = None):
        # The former module entrypoint configured INFO logging before calling
        # run_producer(). Ray invokes this class directly, so preserve that
        # behavior without replacing Ray's own log handler and actor prefix.
        _configure_driver_logging()
        self.config = config
        self._task: asyncio.Task[Any] | None = None
        self._run_gate = asyncio.Event()
        self._run_gate.set()
        self._events = events
        self._paused_at: float | None = None
        self._published_total = 0
        self._notified_published_total = 0
        self._notification_pending_since: float | None = None
        self._notification_finishing = False
        self._notification_wakeup = asyncio.Event()
        self._notification_task: asyncio.Task[None] | None = None
        self._notification_batch_size = _producer_notification_batch_size(config)
        self._metrics_pending: list[dict[str, Any]] = []

    async def _on_metrics(
        self, published_total: int, metrics: dict[str, float]
    ) -> None:
        if self._events is not None:
            self._metrics_pending.append(
                {
                    "kind": "producer_metrics",
                    "published_total": published_total,
                    "metrics": dict(metrics),
                }
            )
            self._notification_wakeup.set()

    async def _before_request(self) -> None:
        await self._run_gate.wait()

    async def _on_published(self, sequence_no: int) -> None:
        del sequence_no
        if self._events is None:
            return
        self._published_total += 1
        if self._notification_pending_since is None:
            self._notification_pending_since = time.monotonic()
        self._notification_wakeup.set()

    async def _notify_published_totals(self) -> None:
        """Coalesce successful publishes without blocking the publish path."""

        events = self._events
        if events is None:
            return

        while True:
            if self._metrics_pending:
                # Keep the snapshot until delivery succeeds. The producer's
                # publish tasks only append scalars; queue RPCs run here.
                await asyncio.to_thread(
                    retry_io,
                    lambda: put_queue(events, self._metrics_pending[0]),
                    name="Producer metrics event",
                )
                self._metrics_pending.pop(0)
                continue
            pending = self._published_total - self._notified_published_total
            if pending <= 0:
                if self._notification_finishing:
                    return
                self._notification_wakeup.clear()
                await self._notification_wakeup.wait()
                continue

            pending_since = self._notification_pending_since or time.monotonic()
            delay_remaining = max(
                0.0,
                pending_since
                + _PRODUCER_NOTIFICATION_MAX_DELAY_SECONDS
                - time.monotonic(),
            )
            if (
                not self._notification_finishing
                and pending < self._notification_batch_size
                and delay_remaining > 0
            ):
                self._notification_wakeup.clear()
                try:
                    await asyncio.wait_for(
                        self._notification_wakeup.wait(), timeout=delay_remaining
                    )
                except TimeoutError:
                    pass
                continue

            published_total = self._published_total
            published_event = {
                "kind": "samples_published",
                "published_total": published_total,
            }
            await asyncio.to_thread(
                retry_io,
                lambda event=published_event: put_queue(events, event),
                name="Producer publication event",
            )
            self._notified_published_total = published_total
            self._notification_pending_since = (
                time.monotonic()
                if self._published_total > self._notified_published_total
                else None
            )

    async def _run_producer(self) -> Any:
        from verl_speco.standalone_tq_producer import run_producer

        if self._events is not None:
            self._notification_task = asyncio.create_task(
                self._notify_published_totals(),
                name="producer-publish-notifier",
            )
        try:
            result = await run_producer(
                self.config,
                before_request=self._before_request,
                on_published=self._on_published,
                on_metrics=self._on_metrics,
                get_runtime_state=lambda: (
                    "running" if self._run_gate.is_set() else "paused"
                ),
            )
        except BaseException:
            if self._notification_task is not None:
                # run_producer flushes its partial window before cancellation
                # reaches here; deliver it before returning from stop().
                self._notification_finishing = True
                self._notification_wakeup.set()
                await self._notification_task
            raise
        self._notification_finishing = True
        self._notification_wakeup.set()
        if self._notification_task is not None:
            await self._notification_task
        return result

    async def start(self) -> dict[str, bool]:
        if self._task is not None and not self._task.done():
            return {"started": False}
        self._published_total = 0
        self._notified_published_total = 0
        self._notification_pending_since = None
        self._notification_finishing = False
        self._notification_wakeup.clear()
        self._notification_task = None
        self._metrics_pending.clear()
        self._task = asyncio.create_task(self._run_producer())
        return {"started": True}

    async def wait(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("Producer has not been started")
        return asdict(await self._task)

    async def stop(self) -> dict[str, bool]:
        task = self._task
        if task is None or task.done():
            return {"stopped": False}
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return {"stopped": True}

    async def pause(self) -> dict[str, bool]:
        changed = self._run_gate.is_set()
        self._run_gate.clear()
        if changed:
            import time

            self._paused_at = time.monotonic()
            logger.info("Standalone Producer paused accepting new requests")
        return {"paused": changed}

    async def resume(self) -> dict[str, bool]:
        changed = not self._run_gate.is_set()
        self._run_gate.set()
        if changed:
            import time

            paused_seconds = (
                time.monotonic() - self._paused_at
                if self._paused_at is not None
                else 0.0
            )
            logger.info(
                "Standalone Producer resumed accepting new requests "
                "paused_seconds=%.3f",
                paused_seconds,
            )
            self._paused_at = None
        return {"resumed": changed}


def _ray_controller_types():
    """Import upstream verl Ray APIs lazily so basic imports stay lightweight."""

    from verl.single_controller.ray import RayClassWithInitArgs
    from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup

    return RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def build_consumer_worker_group(
    config: Any,
    *,
    ray_module: Any,
    scheduled_commands: Any | None = None,
    training_events: Any | None = None,
):
    """Create one standalone training WorkerGroup from ordinary draft config."""

    RayClassWithInitArgs, RayResourcePool, RayWorkerGroup = _ray_controller_types()
    from verl.utils.device import get_device_name

    from verl_speco.workers.standalone_drafter_worker import StandaloneDrafterWorker

    draft_training = config.speco.draft_training
    nnodes = int(draft_training.nnodes)
    nproc_per_node = int(draft_training.nproc_per_node)
    resource_pool = RayResourcePool(
        process_on_nodes=[nproc_per_node] * nnodes,
        use_gpu=True,
        max_colocate_count=1,
        name_prefix="speco_standalone_pool",
    )
    worker_cls = RayClassWithInitArgs(
        cls=ray_module.remote(StandaloneDrafterWorker),
        config=config,
        scheduled_commands=scheduled_commands,
        training_events=training_events,
    )
    return RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_cls,
        name_prefix="speco_standalone_consumer",
        device_name=get_device_name(),
    )


def _flatten_refs(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class StandaloneRayTrainer:
    """Local driver matching ``SpecoRayPPOTrainer`` WorkerGroup ownership."""

    def __init__(
        self,
        commands: PipelineCommands,
        *,
        ray_module: Any,
        worker_group_factory: Callable[..., Any] = build_consumer_worker_group,
        queue_factory: Callable[[], Any] | None = None,
        feature_store_factory: Callable[[Any], Any] | None = None,
    ):
        self.ray = ray_module
        self.producer_config = compose_runtime_config(commands.producer_overrides)
        self.consumer_config = compose_runtime_config(commands.consumer_overrides)
        self.worker_group_factory = worker_group_factory
        self.queue_factory = queue_factory
        self.feature_store_factory = feature_store_factory
        self.producer = None
        self.consumer_wg = None
        self.commands_queue = None
        self.events_queue = None

    def _queue_config(self) -> QueueScheduleConfig:
        training = self.consumer_config.actor_rollout_ref.rollout.drafter.training
        global_batch = (
            int(training.batch_size_per_gpu)
            * int(self.consumer_config.speco.draft_training.nproc_per_node)
            * int(self.consumer_config.speco.draft_training.nnodes)
        )
        scheduler_cfg = self.consumer_config.speco.draft_training.get("scheduler") or {}
        producer_cfg = self.producer_config.speco.standalone_tq_producer
        max_pending = int(producer_cfg.max_pending_samples)
        default_low = max(global_batch, max_pending // 2)
        low = int(scheduler_cfg.get("low_watermark_samples", 0) or default_low)
        default_high = max(max_pending, low + global_batch)
        high = int(scheduler_cfg.get("high_watermark_samples", 0) or default_high)
        return QueueScheduleConfig(low, high, global_batch)

    def init_workers(self) -> None:
        """Create actors once, like cotrain initializes and retains drafter_wg."""

        if self.queue_factory is None:
            from ray.util.queue import Queue

            self.queue_factory = Queue
        self.commands_queue = self.queue_factory()
        self.events_queue = self.queue_factory()
        producer_cls = self.ray.remote(max_concurrency=4)(StandaloneProducerActor)
        self.producer = producer_cls.options(
            name="speco_standalone_producer", num_cpus=1
        ).remote(self.producer_config, self.events_queue)
        self.consumer_wg = self.worker_group_factory(
            self.consumer_config,
            ray_module=self.ray,
            scheduled_commands=self.commands_queue,
            training_events=self.events_queue,
        )

    def run(self) -> int:
        if self.producer is None or self.consumer_wg is None:
            self.init_workers()
        producer = self.producer
        consumer_wg = self.consumer_wg
        from queue import Empty

        from verl_speco.trainer.tq_feature_store import TQFeatureStore
        from verl_speco.trainer.tq_sample_source import build_assignments

        assert self.commands_queue is not None and self.events_queue is not None
        config = self._queue_config()
        training_cfg = self.consumer_config.actor_rollout_ref.rollout.drafter.training
        tq_cfg = training_cfg.transfer_queue
        store = (
            self.feature_store_factory(tq_cfg)
            if self.feature_store_factory is not None
            else TQFeatureStore.from_config(tq_cfg)
        )
        store.connect()
        scheduler = DrafterScheduler()
        depth = prefetch_depth(tq_cfg)
        io_attempts = int(tq_cfg.get("io_retry_max_attempts", 3))
        plans = StandaloneQueueState(depth + 1, config.global_batch_size)
        max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 0)) or 0)
        # The restored optimizer step is known only after Consumer activation.
        consumer_ready = not (depth or max_steps > 0)
        initial_optimizer_step = 0
        producer_done = False
        producer_paused = False
        consumer_stop_requested = False
        pipeline_started = time.monotonic()
        initial_list_started = time.monotonic()
        initial_ready = retry_io(
            store.list_ready, name="TQ scheduler list", attempts=io_attempts
        )
        plans.refresh(initial_ready)
        ready_hint = len(initial_ready)
        tq_list_count = 1
        tq_list_seconds = time.monotonic() - initial_list_started
        producer_pause_count = 0
        tail_dropped = 0
        completed_steps = 0
        last_published_total = 0
        trackers: list[Any] = []

        def log_tracking_event(event: dict[str, Any]) -> None:
            nonlocal tq_list_count, tq_list_seconds
            if event.get("kind") == "training_metrics":
                _log_standalone_tracking_metrics(
                    trackers,
                    event["metrics"],
                    step=int(event["step"]),
                )
            elif event.get("kind") == "producer_metrics":
                metrics = dict(event["metrics"])
                # Low-frequency producer statistics keep an actual TQ snapshot.
                # This query does not refresh selection caches or watermarks.
                started = time.monotonic()
                try:
                    metrics["tq/ready_samples"] = float(len(store.list_ready()))
                except Exception:
                    logger.exception("Failed to sample TQ size for Producer metrics")
                finally:
                    tq_list_count += 1
                    tq_list_seconds += time.monotonic() - started
                _log_producer_tracking_metrics(
                    trackers,
                    metrics,
                    published_total=int(event["published_total"]),
                )

        world_size = int(
            self.consumer_config.speco.draft_training.nproc_per_node
        ) * int(self.consumer_config.speco.draft_training.nnodes)

        def submit_training_command(plan, selected_entries):
            assignments = build_assignments(
                selected_entries,
                batch_size=config.global_batch_size // world_size,
                world_size=world_size,
            )
            command = {
                "kind": "batch",
                "plan_id": plans.pending[-1].plan_id,
                "global_keys": list(plan.selected_keys),
                "global_sequence_nos": [
                    int(entry.tag["sequence_no"]) for entry in selected_entries
                ],
                "assignments": [
                    [
                        {"key": entry.key, "tag": dict(entry.tag)}
                        for entry in rank_entries
                    ]
                    for rank_entries in assignments
                ],
            }
            retry_io(
                lambda: put_queue(
                    self.commands_queue,
                    command,
                    timeout=float(tq_cfg.get("io_timeout_seconds", 30)),
                ),
                name="Ray training command",
                attempts=io_attempts,
            )
            return command

        scheduler.bind_standalone_collection_executor(
            CallbackStandaloneCollectionExecutor(
                pause=lambda: producer.pause.remote(),
                resume=lambda: producer.resume.remote(),
                stop=lambda: producer.stop.remote(),
                resolve=self.ray.get,
            )
        )
        scheduler.bind_standalone_training_executor(
            CallbackStandaloneTrainingExecutor(
                submit=submit_training_command,
                stop=lambda: retry_io(
                    lambda: put_queue(self.commands_queue, {"kind": "stop"}),
                    name="Consumer stop command",
                ),
            )
        )
        logger.info(
            "Standalone scheduler initialized global_batch_size=%s "
            "low_watermark=%s high_watermark=%s ready_samples=%s",
            config.global_batch_size,
            config.low_watermark_samples,
            config.high_watermark_samples,
            ready_hint,
        )

        self.ray.get(producer.start.remote())
        producer_ref = producer.wait.remote()
        consumer_refs = _flatten_refs(consumer_wg.run_standalone_training())

        def context(confirmed_ready: int) -> QueueScheduleContext:
            return QueueScheduleContext(
                queue_status=QueueStatus(confirmed_ready),
                config=config,
                producer_done=producer_done,
                producer_paused=producer_paused,
                consumer_training=bool(plans.pending),
                consumer_ready=consumer_ready,
                outstanding_batches=len(plans.pending),
                max_outstanding_batches=plans.capacity,
                remaining_steps=(
                    max(0, max_steps - initial_optimizer_step - completed_steps)
                    if max_steps > 0
                    else None
                ),
            )

        def reconcile_and_schedule(trigger: str) -> bool:
            nonlocal producer_paused, consumer_stop_requested
            nonlocal tq_list_count, tq_list_seconds, producer_pause_count, tail_dropped
            refreshed = False

            def refresh():
                nonlocal refreshed, tq_list_count, tq_list_seconds
                started = time.monotonic()
                try:
                    entries = retry_io(
                        store.list_ready, name="TQ scheduler list", attempts=io_attempts
                    )
                    plans.refresh(entries)
                finally:
                    tq_list_count += 1
                    tq_list_seconds += time.monotonic() - started
                refreshed = True

            collection = scheduler.plan_queue_collection(context(max(0, ready_hint)))
            collection_outcome = scheduler.execute_standalone_collection_plan(
                collection,
                producer_paused=producer_paused,
                producer_done=producer_done,
            )
            producer_paused = collection_outcome.producer_paused
            if collection_outcome.changed:
                producer_pause_count += int(producer_paused)
                logger.info(
                    "Standalone producer %s ready_hint=%s",
                    "paused" if producer_paused else "resumed",
                    ready_hint,
                )

            while True:
                training = scheduler.plan_queue_training(
                    context(len(plans.ready)),
                    selected_keys=tuple(
                        entry.key
                        for entry in islice(plans.ready, config.global_batch_size)
                    ),
                )
                if not training.launch:
                    if (
                        training.reason == "insufficient_ready_samples"
                        and not refreshed
                    ):
                        refresh()
                        continue
                    break
                if not consumer_refs:
                    raise RuntimeError(
                        "Standalone Consumer exited before the Scheduler could submit the next TrainingPlan"
                    )
                pending, selected = plans.reserve()
                scheduler.execute_standalone_training_plan(
                    training,
                    runtime_state=pending.runtime,
                    selected_entries=selected,
                )
                logger.debug(
                    "Standalone dispatched plan=%s outstanding=%s cached_samples=%s trigger=%s",
                    pending.plan_id,
                    len(plans.pending),
                    len(plans.ready),
                    trigger,
                )

            if scheduler.queue_budget_exhausted(context(len(plans.ready))):
                scheduler.stop_standalone_consumer()
                consumer_stop_requested = True
                return True
            if producer_done and not plans.pending and consumer_ready:
                # Consumer clear has succeeded for every completed plan. Verify
                # the tail once, even if a publication notification was delayed.
                if not refreshed:
                    refresh()
                if len(plans.ready) >= config.global_batch_size:
                    return reconcile_and_schedule("final_batch_ready")
                if plans.ready:
                    keys = [entry.key for entry in plans.ready]
                    retry_io(
                        lambda: store.clear_many(keys),
                        name="TQ tail clear",
                        attempts=io_attempts,
                    )
                    tail_dropped += len(keys)
                    plans.ready.clear()
                scheduler.stop_standalone_consumer()
                consumer_stop_requested = True
                return True
            return False

        try:
            trackers = _build_standalone_tracking(self.consumer_config, rank=0)
            finished = reconcile_and_schedule("initialization")
            while not finished:
                wait_refs = list(consumer_refs)
                if producer_ref is not None:
                    wait_refs.insert(0, producer_ref)
                if wait_refs:
                    completed, _ = self.ray.wait(
                        wait_refs,
                        num_returns=1,
                        timeout=0,
                    )
                else:
                    completed = []
                if completed:
                    ref = completed[0]
                    try:
                        result = self.ray.get(ref)
                    except Exception as error:
                        if ref != producer_ref and plans.pending:
                            plans.pending[0].runtime.mark_failed(error)
                        raise
                    if ref == producer_ref:
                        producer_ref = None
                        producer_done = True
                        logger.info(
                            "Standalone scheduler producer completed ready_hint=%s",
                            ready_hint,
                        )
                        finished = reconcile_and_schedule("producer_done")
                        continue
                    if not consumer_stop_requested:
                        results = result if isinstance(result, list) else [result]
                        expected_completion = (
                            bool(results)
                            and max_steps > 0
                            and all(
                                isinstance(item, dict)
                                and int(item.get("optimizer_steps_total", -1))
                                >= max_steps
                                for item in results
                            )
                        )
                        if not expected_completion:
                            raise RuntimeError(
                                "Standalone Consumer exited before receiving the "
                                "Scheduler stop command"
                            )
                    consumer_refs.remove(ref)
                    continue
                try:
                    event = get_queue(self.events_queue, timeout=1.0)
                except Empty:
                    if not wait_refs:
                        raise RuntimeError(
                            "Standalone pipeline has no live Producer or Consumer "
                            "and did not receive the expected completion event"
                        )
                    continue
                kind = event.get("kind")
                if kind in {"training_metrics", "producer_metrics"}:
                    log_tracking_event(event)
                    continue
                if kind == "consumer_ready":
                    initial_optimizer_step = int(event["initial_optimizer_step"])
                    consumer_ready = True
                    finished = reconcile_and_schedule("consumer_ready")
                elif kind == "samples_published":
                    published_total = int(event.get("published_total", 0))
                    if published_total <= last_published_total:
                        continue
                    # initial_ready was captured before producer.start(). Keep
                    # event counts independent of subsequent list snapshots.
                    ready_hint += published_total - last_published_total
                    last_published_total = published_total
                    finished = reconcile_and_schedule("samples_published")
                elif kind == "training_completed":
                    pending = plans.completion(
                        event.get("plan_id"), event.get("keys", ())
                    )
                    if pending is None:
                        continue
                    training_outcome = scheduler.complete_standalone_training(
                        runtime_state=pending.runtime,
                        completed_keys=event.get("keys", ()),
                        successful=bool(event.get("successful", False)),
                    )
                    plans.release(pending)
                    completed_steps += 1
                    # Do not clamp the accumulator: late publication events
                    # must cancel the earlier subtraction exactly once.
                    ready_hint -= training_outcome.assigned_samples
                    if completed_steps % _SCHEDULER_INFO_INTERVAL_STEPS == 0:
                        logger.info(
                            "Standalone scheduler step=%s outstanding=%s ready_hint=%s tq_list_count=%s tq_list_time=%.3fs",
                            completed_steps,
                            len(plans.pending),
                            ready_hint,
                            tq_list_count,
                            tq_list_seconds,
                        )
                    finished = reconcile_and_schedule("training_completed")
            if finished:
                for ref in consumer_refs:
                    self.ray.get(ref)
            logger.info(
                "Standalone pipeline completed consumer_steps=%s tail_dropped=%s "
                "producer_pauses=%s tq_list_count=%s tq_list_time=%.3fs elapsed=%.3fs",
                completed_steps,
                tail_dropped,
                producer_pause_count,
                tq_list_count,
                tq_list_seconds,
                time.monotonic() - pipeline_started,
            )
            return 0
        except Exception:
            logger.exception(
                "Standalone pipeline failed producer_state=%s consumer_training=%s "
                "ready_hint=%s completed_steps=%s",
                "paused" if producer_paused else "done" if producer_done else "running",
                bool(plans.pending),
                ready_hint,
                completed_steps,
            )
            raise
        finally:
            if producer_ref is not None:
                try:
                    self.ray.get(producer.stop.remote())
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to stop standalone Producer actor")
            # Producer.stop() waits for the partial-window notification; workers
            # already awaited on normal completion may have final metric events.
            try:
                while True:
                    try:
                        event = self.events_queue.get(block=False)
                    except Empty:
                        break
                    if event.get("kind") in {"training_metrics", "producer_metrics"}:
                        log_tracking_event(event)
            finally:
                _finish_standalone_tracking(trackers, rank=0)
                store.close()


def run_ray_actors(
    commands: PipelineCommands,
    *,
    ray_module: Any,
    worker_group_factory: Callable[..., Any] = build_consumer_worker_group,
) -> int:
    _configure_driver_logging()
    logger.info("Standalone Ray scheduler Driver starting")
    trainer = StandaloneRayTrainer(
        commands,
        ray_module=ray_module,
        worker_group_factory=worker_group_factory,
    )
    return trainer.run()


def run_ray_pipeline(
    commands: PipelineCommands,
    *,
    ray_module: Any,
    ray_address: str,
    environ: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    endpoint_ready: Callable[[str], bool] = _vllm_is_ready,
) -> int:
    """Keep service ownership local while Ray executes Producer and Consumer."""

    base_env = dict(os.environ if environ is None else environ)
    base_env["RAY_ADDRESS"] = ray_address
    # Mirror the subprocess path: fail fast when the selected TQ transport or
    # handle-based store points at an unreachable Mooncake master. This is the
    # default runtime mode.
    validate_tq_backend(base_env)
    validate_hidden_states_store(base_env)
    owner = None
    vllm = None
    hidden_states_temp = None
    try:
        with tempfile.TemporaryDirectory(prefix="speco-tq-launch-") as temp_dir:
            ready_file = Path(temp_dir) / "owner.ready"
            hidden_states_temp = tempfile.TemporaryDirectory(
                prefix="speco-vllm-hidden-states-"
            )
            unavailable = [e for e in commands.vllm_endpoints if not endpoint_ready(e)]
            if unavailable:
                if commands.vllm is None:
                    raise RuntimeError(
                        "The configured hidden-state vLLM endpoints are unavailable: "
                        + ", ".join(unavailable)
                    )
                vllm_command = [
                    part.replace(_VLLM_HIDDEN_STATES_DIR, hidden_states_temp.name)
                    for part in commands.vllm
                ]
                vllm = popen(vllm_command, env=base_env)
            owner_env = {**base_env, "SPECO_TQ_OWNER_READY_FILE": str(ready_file)}
            owner = popen(commands.owner, env=owner_env)
            _wait_for_owner_ready(owner, ready_file, timeout_seconds=120)
            if vllm is not None:
                _wait_for_vllm_ready(
                    vllm,
                    commands.vllm_endpoints[0],
                    timeout_seconds=900,
                    endpoint_ready=endpoint_ready,
                )
            return run_ray_actors(commands, ray_module=ray_module)
    except KeyboardInterrupt:
        logger.warning("Standalone Ray training interrupted")
        return 130
    finally:
        _stop_process(owner)
        _stop_process(vllm)
        if hidden_states_temp is not None:
            hidden_states_temp.cleanup()


__all__ = [
    "StandaloneProducerActor",
    "StandaloneRayTrainer",
    "build_consumer_worker_group",
    "compose_runtime_config",
    "run_ray_actors",
    "run_ray_pipeline",
]
