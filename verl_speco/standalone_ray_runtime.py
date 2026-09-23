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
from dataclasses import asdict
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verl_speco.trainer.scheduler import (
    CallbackStandaloneCollectionExecutor,
    CallbackStandaloneTrainingExecutor,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduler,
    QueueScheduleConfig,
    QueueScheduleContext,
    QueueStatus,
)

from verl_speco.standalone_tq_training_launcher import (
    PipelineCommands,
    _VLLM_HIDDEN_STATES_DIR,
    _stop_process,
    _wait_for_owner_ready,
    _wait_for_vllm_ready,
    _vllm_is_ready,
)

logger = logging.getLogger(__name__)
_SCHEDULER_INFO_INTERVAL_STEPS = 50
_PRODUCER_NOTIFICATION_MAX_BATCH = 32
_PRODUCER_NOTIFICATION_MAX_DELAY_SECONDS = 0.2


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
            await asyncio.to_thread(
                events.put,
                {
                    "kind": "samples_published",
                    "published_total": published_total,
                },
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
                get_runtime_state=lambda: (
                    "running" if self._run_gate.is_set() else "paused"
                ),
            )
        except BaseException:
            if self._notification_task is not None:
                self._notification_task.cancel()
                try:
                    await self._notification_task
                except asyncio.CancelledError:
                    pass
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
        runtime_state = DrafterRuntimeState()
        producer_done = False
        producer_paused = False
        consumer_stop_requested = False
        pipeline_started = time.monotonic()
        initial_list_started = time.monotonic()
        ready_hint = len(store.list_ready())
        tq_list_count = 1
        tq_list_seconds = time.monotonic() - initial_list_started
        producer_pause_count = 0
        tail_dropped = 0
        completed_steps = 0
        last_published_total = 0
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
            self.commands_queue.put(command)
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
                stop=lambda: self.commands_queue.put({"kind": "stop"}),
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
                consumer_training=runtime_state.status
                in {DrafterRuntimeStatus.SUBMITTED, DrafterRuntimeStatus.RUNNING},
            )

        def reconcile_and_schedule(trigger: str) -> bool:
            nonlocal ready_hint, producer_paused, consumer_stop_requested
            nonlocal tq_list_count, tq_list_seconds, producer_pause_count, tail_dropped
            hint_before = ready_hint
            list_started = time.monotonic()
            ready = store.list_ready()
            list_elapsed = time.monotonic() - list_started
            tq_list_count += 1
            tq_list_seconds += list_elapsed
            ready_hint = len(ready)
            logger.debug(
                "Standalone scheduler TQ reconciled trigger=%s ready_hint=%s "
                "actual_ready=%s delta=%s elapsed=%.3fs",
                trigger,
                hint_before,
                ready_hint,
                ready_hint - hint_before,
                list_elapsed,
            )
            collection = scheduler.plan_queue_collection(context(ready_hint))
            collection_outcome = scheduler.execute_standalone_collection_plan(
                collection,
                producer_paused=producer_paused,
                producer_done=producer_done,
            )
            producer_paused = collection_outcome.producer_paused
            if collection_outcome.changed and producer_paused:
                producer_pause_count += 1
                logger.info(
                    "Standalone scheduler producer paused reason=%s ready=%s "
                    "high_watermark=%s",
                    collection.reason,
                    ready_hint,
                    config.high_watermark_samples,
                )
            elif collection_outcome.changed and not producer_paused:
                logger.info(
                    "Standalone scheduler producer resumed reason=%s ready=%s "
                    "low_watermark=%s",
                    collection.reason,
                    ready_hint,
                    config.low_watermark_samples,
                )

            training = scheduler.plan_queue_training(
                context(ready_hint),
                selected_keys=tuple(
                    entry.key for entry in ready[: config.global_batch_size]
                ),
            )
            if training.launch:
                selected = ready[: config.global_batch_size]
                if not consumer_refs:
                    raise RuntimeError(
                        "Standalone Consumer exited before the Scheduler could "
                        "submit the next TrainingPlan"
                    )
                training_outcome = scheduler.execute_standalone_training_plan(
                    training,
                    runtime_state=runtime_state,
                    selected_entries=selected,
                )
                sequence_nos = [int(entry.tag["sequence_no"]) for entry in selected]
                logger.debug(
                    "Standalone scheduler training dispatched step=%s "
                    "batch_samples=%s ready_before=%s producer_state=%s "
                    "sequence_range=%s-%s",
                    completed_steps + 1,
                    training_outcome.assigned_samples,
                    ready_hint,
                    "paused"
                    if producer_paused
                    else "done"
                    if producer_done
                    else "running",
                    min(sequence_nos),
                    max(sequence_nos),
                )
            if (
                producer_done
                and runtime_state.status
                not in {DrafterRuntimeStatus.SUBMITTED, DrafterRuntimeStatus.RUNNING}
                and ready_hint < config.global_batch_size
            ):
                if ready:
                    tail_dropped += len(ready)
                    logger.info(
                        "Standalone scheduler dropping incomplete tail "
                        "tail_samples=%s required_batch_size=%s",
                        len(ready),
                        config.global_batch_size,
                    )
                    store.clear_many([entry.key for entry in ready])
                    ready_hint = 0
                scheduler.stop_standalone_consumer()
                consumer_stop_requested = True
                return True
            return False

        try:
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
                        if ref != producer_ref and runtime_state.status in {
                            DrafterRuntimeStatus.SUBMITTED,
                            DrafterRuntimeStatus.RUNNING,
                        }:
                            runtime_state.mark_failed(error)
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
                        max_steps = int(training_cfg.get("max_steps", 0) or 0)
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
                    event = self.events_queue.get(block=True, timeout=1.0)
                except Empty:
                    if not wait_refs:
                        raise RuntimeError(
                            "Standalone pipeline has no live Producer or Consumer "
                            "and did not receive the expected completion event"
                        )
                    continue
                kind = event.get("kind")
                if kind == "samples_published":
                    published_total = int(event.get("published_total", 0))
                    if published_total <= last_published_total:
                        continue
                    ready_hint += published_total - last_published_total
                    last_published_total = published_total
                    consumer_training = runtime_state.status in {
                        DrafterRuntimeStatus.SUBMITTED,
                        DrafterRuntimeStatus.RUNNING,
                    }
                    if (
                        not consumer_training and ready_hint >= config.global_batch_size
                    ) or (
                        not producer_paused
                        and ready_hint >= config.high_watermark_samples
                    ):
                        trigger = (
                            "batch_ready" if not consumer_training else "high_watermark"
                        )
                        finished = reconcile_and_schedule(trigger)
                elif kind == "training_completed":
                    training_outcome = scheduler.complete_standalone_training(
                        runtime_state=runtime_state,
                        completed_keys=event.get("keys", ()),
                        successful=bool(event.get("successful", False)),
                    )
                    consumed = training_outcome.assigned_samples
                    completed_steps += 1
                    ready_hint = max(0, ready_hint - consumed)
                    logger.debug(
                        "Standalone scheduler training completed step=%s consumed=%s "
                        "ready_hint=%s producer_state=%s",
                        completed_steps,
                        consumed,
                        ready_hint,
                        "paused"
                        if producer_paused
                        else "done"
                        if producer_done
                        else "running",
                    )
                    if completed_steps % _SCHEDULER_INFO_INTERVAL_STEPS == 0:
                        logger.info(
                            "Standalone scheduler summary step=%s ready_hint=%s "
                            "producer_state=%s tq_list_count=%s "
                            "tq_list_time=%.3fs producer_pauses=%s",
                            completed_steps,
                            ready_hint,
                            (
                                "paused"
                                if producer_paused
                                else "done"
                                if producer_done
                                else "running"
                            ),
                            tq_list_count,
                            tq_list_seconds,
                            producer_pause_count,
                        )
                    if (
                        ready_hint >= config.global_batch_size
                        or (
                            producer_paused
                            and ready_hint <= config.low_watermark_samples
                        )
                        or producer_done
                    ):
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
                runtime_state.status
                in {DrafterRuntimeStatus.SUBMITTED, DrafterRuntimeStatus.RUNNING},
                ready_hint,
                completed_steps,
            )
            raise
        finally:
            store.close()
            if producer_ref is not None:
                try:
                    self.ray.get(producer.stop.remote())
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to stop standalone Producer actor")


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
