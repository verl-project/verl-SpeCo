# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed streaming sample source for a TQ-backed standalone Consumer."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from queue import Empty
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.standalone_ray_runtime import get_queue
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.tq_feature_store import ReadyEntry, TQFeatureStore
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TQLocalBatch:
    """The payload owned by one rank plus rank-0's global cleanup keys."""

    local_keys: list[str]
    local_samples: list[DraftFeatureSample]
    global_keys: list[str] | None
    global_sequence_nos: list[int] | None = None
    command_wait_seconds: float = 0.0
    tq_get_seconds: float = 0.0
    cycle_started_at: float = 0.0
    plan_id: int | None = None
    prepared_batch: dict[str, Any] | None = None
    batch_prepare_seconds: float = 0.0
    file_entries: list[ReadyEntry] | None = None
    cleanup_after_load: bool = False


def build_assignments(
    entries: Sequence[ReadyEntry], *, batch_size: int, world_size: int
) -> list[list[ReadyEntry]]:
    """Split one complete global batch into disjoint contiguous rank batches."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    expected = batch_size * world_size
    if len(entries) != expected:
        raise ValueError(
            f"Expected exactly {expected} ready entries for one global batch, got {len(entries)}"
        )
    return [
        list(entries[rank * batch_size : (rank + 1) * batch_size])
        for rank in range(world_size)
    ]


class TQFeatureDataLoader:
    """Poll TQ on rank 0, distribute keys, and fetch payloads on owner ranks."""

    def __init__(
        self,
        store: TQFeatureStore,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        poll_interval_seconds: float = 0.5,
        drop_last: bool = True,
        scheduled_commands: Any | None = None,
        process_group: Any | None = None,
        stop_event: threading.Event | None = None,
        io_retry_max_attempts: int = 3,
        io_timeout_seconds: float = 30.0,
        prepare_batch: Any | None = None,
        cleanup_after_load: bool = False,
    ):
        self.store = store
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.01)
        self.drop_last = bool(drop_last)
        self.scheduled_commands = scheduled_commands
        self.process_group = process_group
        self.stop_event = stop_event
        self.io_retry_max_attempts = io_retry_max_attempts
        self.io_timeout_seconds = io_timeout_seconds
        self.prepare_batch = prepare_batch
        self.cleanup_after_load = cleanup_after_load
        if cleanup_after_load and (prepare_batch is None or store.uses_transfer_queue):
            raise ValueError(
                "Cleanup after load requires ray_reference batch preparation"
            )
        self._cleanup_entries: dict[str, ReadyEntry] = {}
        if prepare_batch is not None and scheduled_commands is None:
            raise ValueError(
                "File preparation requires scheduler-assigned batch commands"
            )
        if io_retry_max_attempts < 1 or io_timeout_seconds <= 0:
            raise ValueError("TQ retry attempts and I/O timeout must be positive")
        if self.batch_size <= 0:
            raise ValueError("TQ Consumer batch_size_per_gpu must be positive")
        if self.world_size <= 0 or not (0 <= self.rank < self.world_size):
            raise ValueError(
                "Invalid TQ Consumer rank/world_size: "
                f"rank={self.rank}, world_size={self.world_size}"
            )
        if not self.drop_last:
            raise ValueError(
                "TQ Consumer first version requires transfer_queue.drop_last=true"
            )
        if dist.is_initialized() and dist.get_world_size() != self.world_size:
            raise ValueError(
                "TQFeatureDataLoader world_size does not match torch.distributed world size"
            )

    def __iter__(self) -> Iterator[TQLocalBatch]:
        return self._iterate(materialize=True)

    def iter_assignments(self) -> Iterator[TQLocalBatch]:
        """Receive metadata only; CPU workers materialize these batches later."""
        if self.prepare_batch is None or self.scheduled_commands is None:
            raise ValueError("Deferred preparation requires scheduled file assignments")
        return self._iterate(materialize=False)

    def _iterate(self, *, materialize: bool) -> Iterator[TQLocalBatch]:
        self.store.connect()
        global_batch_size = self.batch_size * self.world_size
        owner_ready = False
        last_plan_id = -1
        cycle_started_at = time.perf_counter()
        command_wait_seconds = 0.0
        while True:
            command: dict[str, Any] | None = None
            if self.rank == 0:
                try:
                    if not owner_ready:
                        owner_ready = retry_io(
                            self.store.owner_ready,
                            name="Consumer source readiness",
                            attempts=self.io_retry_max_attempts,
                            stop=self.stop_event,
                        )
                        if not owner_ready and not (
                            self.stop_event is not None and self.stop_event.is_set()
                        ):
                            time.sleep(self.poll_interval_seconds)
                            if self.stop_event is None:
                                continue
                            command = {"kind": "wait"}
                    if command is None and self.scheduled_commands is not None:
                        command_wait_started = time.perf_counter()
                        if self.stop_event is None:
                            command = self.scheduled_commands.get(block=True)
                        else:
                            if not self.stop_event.is_set():
                                try:
                                    command = get_queue(
                                        self.scheduled_commands,
                                        rpc_timeout=self.io_timeout_seconds,
                                    )
                                except Empty:
                                    # Other ranks must not spend the entire
                                    # producer idle period in one timed Gloo
                                    # broadcast. This is control traffic only.
                                    command = {"kind": "wait"}
                            if command is None:
                                command = {"kind": "stop"}
                        command_wait_seconds += (
                            time.perf_counter() - command_wait_started
                        )
                        plan_id = command.get("plan_id")
                        if plan_id is not None and int(plan_id) <= last_plan_id:
                            # Queue put may have succeeded before a response timeout.
                            continue
                        if command.get("kind") == "stop":
                            pass
                        elif command.get("kind") not in {"batch", "wait"}:
                            raise RuntimeError(
                                f"Unsupported scheduled TQ command: {command!r}"
                            )
                    elif command is None:
                        ready = self.store.list_ready()
                        if len(ready) >= global_batch_size:
                            selected = ready[:global_batch_size]
                            command = self._batch_command(selected)
                        else:
                            eos = self.store.read_eos()
                            if eos is not None:
                                tail_keys = [entry.key for entry in ready]
                                if tail_keys:
                                    logger.info(
                                        "Dropping %s TQ tail samples after EOS because one "
                                        "global batch requires %s",
                                        len(tail_keys),
                                        global_batch_size,
                                    )
                                    cleanup_standalone_samples(
                                        self.store, ready, reason="tail_dropped"
                                    )
                                command = {"kind": "stop"}
                            else:
                                time.sleep(self.poll_interval_seconds)
                                continue
                except BaseException as exc:  # noqa: BLE001
                    command = {
                        "kind": "error",
                        "message": f"rank 0 failed while discovering TQ samples: {exc}",
                    }

            command = self._broadcast_command(command)
            if command.get("kind") == "wait":
                stop_event = self.stop_event
                stopped = torch.tensor(
                    int(stop_event is not None and stop_event.is_set()),
                    dtype=torch.int32,
                )
                if self.world_size > 1:
                    dist.all_reduce(
                        stopped, op=dist.ReduceOp.MAX, group=self.process_group
                    )
                if stopped.item():
                    return
                continue
            if command.get("kind") == "error":
                raise RuntimeError(str(command.get("message") or "TQ discovery failed"))
            if command.get("kind") == "stop":
                return
            if command.get("kind") != "batch":
                raise RuntimeError(f"Unsupported TQ loader command: {command!r}")
            if command.get("plan_id") is not None:
                last_plan_id = int(command["plan_id"])
            wire_assignments = command.get("assignments")
            if (
                not isinstance(wire_assignments, list)
                or len(wire_assignments) != self.world_size
            ):
                raise RuntimeError("TQ loader received malformed rank assignments")
            local_entries = [
                _entry_from_wire(item) for item in wire_assignments[self.rank]
            ]
            tq_get_started = time.perf_counter()
            prepared_batch = None
            fetched: Any = (
                retry_io(
                    partial(self.prepare_batch or self.store.get_many, local_entries),
                    name="File batch preparation"
                    if self.prepare_batch is not None
                    else "TQ get",
                    attempts=self.io_retry_max_attempts,
                    stop=self.stop_event,
                )
                if materialize
                else None
            )
            tq_get_seconds = time.perf_counter() - tq_get_started
            batch_prepare_seconds = 0.0
            samples: list[DraftFeatureSample]
            cleanup_after_load = False
            if self.prepare_batch is not None:
                prepared_batch, samples = fetched, []
                if materialize and self.cleanup_after_load:
                    cleanup_standalone_samples(self.store, local_entries)
                    cleanup_after_load = True
                batch_prepare_seconds = (
                    time.perf_counter() - tq_get_started if materialize else 0.0
                )
                tq_get_seconds = 0.0
                if self.rank == 0:
                    self._cleanup_entries.update(
                        {
                            item["key"]: _entry_from_wire(item)
                            for rank_items in wire_assignments
                            for item in rank_items
                        }
                    )
            else:
                samples = fetched
            global_keys = (
                [str(key) for key in command.get("global_keys", [])]
                if self.rank == 0
                else None
            )
            global_sequence_nos = (
                [int(value) for value in command.get("global_sequence_nos", [])]
                if self.rank == 0
                else None
            )
            yield TQLocalBatch(
                local_keys=[entry.key for entry in local_entries],
                local_samples=samples,
                global_keys=global_keys,
                global_sequence_nos=global_sequence_nos,
                command_wait_seconds=command_wait_seconds,
                tq_get_seconds=tq_get_seconds,
                cycle_started_at=cycle_started_at,
                plan_id=command.get("plan_id"),
                prepared_batch=prepared_batch,
                batch_prepare_seconds=batch_prepare_seconds,
                file_entries=local_entries if not materialize else None,
                cleanup_after_load=cleanup_after_load,
            )
            cycle_started_at = time.perf_counter()
            command_wait_seconds = 0.0

    def _batch_command(self, selected: Sequence[ReadyEntry]) -> dict[str, Any]:
        assignments = build_assignments(
            selected,
            batch_size=self.batch_size,
            world_size=self.world_size,
        )
        return {
            "kind": "batch",
            "global_keys": [entry.key for entry in selected],
            "global_sequence_nos": [
                int(entry.tag["sequence_no"]) for entry in selected
            ],
            "assignments": [
                [_entry_to_wire(entry) for entry in rank_entries]
                for rank_entries in assignments
            ],
        }

    def clear_completed_batch(
        self, global_keys: Sequence[str] | None, *, cleanup_after_load=False
    ) -> None:
        if self.rank != 0:
            return
        if not global_keys:
            raise ValueError(
                "rank 0 requires global_keys to clear a completed TQ batch"
            )
        if self.prepare_batch is None:
            self.store.clear_many(global_keys)
        else:
            # The training completion boundary already retries TQ clear.
            if not cleanup_after_load:
                entries = [self._cleanup_entries[key] for key in global_keys]
                cleanup_standalone_samples(self.store, entries, attempts=1)
            for key in global_keys:
                self._cleanup_entries.pop(key, None)

    def _broadcast_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        if not dist.is_initialized() or self.world_size == 1:
            if command is None:
                raise RuntimeError("rank 0 did not create a TQ loader command")
            return command
        payload: list[Any] = [command if self.rank == 0 else None]
        if self.process_group is None:
            dist.broadcast_object_list(payload, src=0)
        else:
            dist.broadcast_object_list(payload, src=0, group=self.process_group)
        received = payload[0]
        if not isinstance(received, dict):
            raise RuntimeError("TQ loader broadcast did not contain a command mapping")
        return received


def _entry_to_wire(entry: ReadyEntry) -> dict[str, Any]:
    return {"key": entry.key, "tag": dict(entry.tag)}


def _entry_from_wire(value: Any) -> ReadyEntry:
    if not isinstance(value, dict) or "key" not in value or "tag" not in value:
        raise TypeError(f"Invalid serialized ReadyEntry: {value!r}")
    if not isinstance(value["tag"], dict):
        raise TypeError("Serialized ReadyEntry.tag must be a mapping")
    return ReadyEntry(key=str(value["key"]), tag=dict(value["tag"]))


class TQPrefetchIterator:
    """One ordered coordinator per rank, bounded payloads, separate CPU collectives.

    Each fetch occupies a slot before next(source). The main thread taking a
    batch frees a slot. EOS/errors live outside the payload buffer so shutdown
    does not depend on a consumer draining a full queue.
    """

    def __init__(
        self,
        loader: TQFeatureDataLoader,
        depth: int,
        *,
        close_timeout: float = 180.0,
        pin_memory: bool = False,
        prepare_workers: int = 4,
        pin_device: torch.device | None = None,
    ):
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        if prepare_workers < 0:
            raise ValueError("prepare_workers must be non-negative")
        if pin_memory and getattr(loader, "prepare_batch", None) is None:
            raise ValueError("Independent pinning requires file batch preparation")
        self.loader = loader
        self.depth = depth
        self.close_timeout = close_timeout
        self.stop = loader.stop_event = threading.Event()
        self._condition = threading.Condition()
        self._batches: deque[TQLocalBatch] = deque()
        self._pin_batches: deque[TQLocalBatch] = deque()
        self._reserved = 0
        self._source_done = False
        self._pin_device = pin_device
        self._original_prepare: Any | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._pin_thread: threading.Thread | None = None
        if pin_memory:
            assert loader.prepare_batch is not None
            self._original_prepare = loader.prepare_batch
            if prepare_workers:
                self._executor = ThreadPoolExecutor(
                    max_workers=prepare_workers,
                    thread_name_prefix=f"file-prepare-{loader.rank}",
                )
            loader.prepare_batch = partial(
                loader.prepare_batch,
                pin_memory=False,
                executor=self._executor,
            )
            self._pin_thread = threading.Thread(
                target=self._pin_loop,
                name=f"file-pin-{loader.rank}",
                daemon=True,
            )
        self._done = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"{'file' if getattr(loader, 'prepare_batch', None) is not None else 'tq'}-prefetch-{loader.rank}",
            daemon=True,
        )
        if self._pin_thread is not None:
            self._pin_thread.start()
        self._thread.start()

    def _sync(self, values: list[int]) -> list[int]:
        if self.loader.world_size > 1:
            if self.loader.process_group is None:
                raise RuntimeError(
                    "Distributed prefetch requires its own Gloo process group"
                )
            state = torch.tensor(values, dtype=torch.int32)
            dist.all_reduce(
                state, op=dist.ReduceOp.MAX, group=self.loader.process_group
            )
            return state.tolist()
        return values

    def _run(self) -> None:
        source = iter(self.loader)
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stop.is_set() or self._reserved < self.depth
                    )
                    if not self.stop.is_set():
                        self._reserved += 1
                if self._sync([int(self.stop.is_set())])[0]:
                    return
                batch = None
                error = None
                exhausted = False
                try:
                    batch = next(source)
                except StopIteration:
                    exhausted = True
                except BaseException as exc:  # noqa: BLE001 - deliver thread failures to every rank
                    error = exc
                failed, ended = self._sync([int(error is not None), int(exhausted)])
                if failed:
                    raise RuntimeError(
                        f"Consumer prefetch failed on rank={self.loader.rank}"
                    ) from error
                if ended:
                    return
                if batch is None:
                    raise RuntimeError(
                        "Consumer prefetch returned no batch without EOS"
                    )
                with self._condition:
                    if self._pin_thread is None:
                        self._batches.append(batch)
                    else:
                        self._pin_batches.append(batch)
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the main thread
            with self._condition:
                self._error = exc
        finally:
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            finally:
                with self._condition:
                    self._source_done = True
                    if self._pin_thread is None:
                        self._done = True
                    self._condition.notify_all()

    def _pin_loop(self) -> None:
        """Pin FIFO batches while the coordinator prepares the next assignment.

        This thread never performs distributed collectives. Its errors surface
        through next(), then the existing main-thread rank agreement aborts all
        ranks before a mismatched training step can start.
        """
        try:
            # Match torch DataLoader's pin thread, avoiding parallel CPU copies.
            torch.set_num_threads(1)
            if self._pin_device is not None and self._pin_device.type != "cpu":
                getattr(torch, self._pin_device.type).set_device(self._pin_device)
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stop.is_set()
                        or self._pin_batches
                        or self._source_done
                    )
                    if self.stop.is_set() or not self._pin_batches:
                        return
                    batch = self._pin_batches.popleft()
                started = time.perf_counter()
                if batch.prepared_batch is None:
                    raise RuntimeError("File pinning received no prepared batch")
                pinned = {
                    key: value.pin_memory() if torch.is_tensor(value) else value
                    for key, value in batch.prepared_batch.items()
                }
                batch = replace(
                    batch,
                    prepared_batch=pinned,
                    batch_prepare_seconds=batch.batch_prepare_seconds
                    + time.perf_counter()
                    - started,
                )
                with self._condition:
                    self._batches.append(batch)
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surface pin failures in the main thread
            with self._condition:
                self._error = exc
                self.stop.set()
                self._condition.notify_all()
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> TQLocalBatch:
        with self._condition:
            self._condition.wait_for(lambda: bool(self._batches) or self._done)
            if self._batches:
                batch = self._batches.popleft()
                self._reserved -= 1
                self._condition.notify_all()
                return batch
            if self._error is not None:
                raise self._error
            raise StopIteration

    def close(self) -> None:
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=self.close_timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                "TQ prefetch thread did not stop before its shutdown deadline"
            )
        if self._pin_thread is not None:
            self._pin_thread.join(timeout=self.close_timeout)
            if self._pin_thread.is_alive():
                raise RuntimeError(
                    "Consumer pin thread did not stop before its shutdown deadline"
                )
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._original_prepare is not None:
            self.loader.prepare_batch = self._original_prepare
        self._batches.clear()
        self._pin_batches.clear()
        self._reserved = 0


__all__ = [
    "TQFeatureDataLoader",
    "TQLocalBatch",
    "TQPrefetchIterator",
    "build_assignments",
]
