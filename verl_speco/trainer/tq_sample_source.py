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
from collections.abc import Iterator, Sequence
from functools import partial
from queue import Empty
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.producer.metrics import current_ray_node_id
from verl_speco.standalone_ray_runtime import get_queue
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.standalone_data import (
    StandaloneLocalBatch as TQLocalBatch,
)
from verl_speco.trainer.standalone_data import (
    _entry_from_wire,
    build_assignments,
)
from verl_speco.trainer.standalone_prefetch import (
    StandalonePrefetchIterator as TQPrefetchIterator,
)
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples
from verl_speco.trainer.tq_feature_store import ReadyEntry, TQFeatureStore
from verl_speco.trainer.tq_retry import retry_io

logger = logging.getLogger(__name__)


class TQFeatureDataLoader:
    """Receive Ray scheduler assignments and fetch TQ payloads on owner ranks."""

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
        self._clock_node_id = current_ray_node_id()
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
        if scheduled_commands is None:
            raise ValueError(
                "TQ consumption requires scheduler-assigned batch commands"
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
            delivery_times = dict(command.get("delivery_times", {}))
            if (
                self._clock_node_id is not None
                and command.get("clock_node_id") == self._clock_node_id
                and "dispatched_at" in command
            ):
                delivery_times["command_delivery_time"] = (
                    time.monotonic() - command["dispatched_at"]
                )
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
                delivery_times=delivery_times,
            )
            cycle_started_at = time.perf_counter()
            command_wait_seconds = 0.0

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


__all__ = [
    "TQFeatureDataLoader",
    "TQLocalBatch",
    "TQPrefetchIterator",
    "build_assignments",
]
