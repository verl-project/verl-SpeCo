# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Scheduler-only reference consumption with no TQ polling or payload reads."""

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
from verl_speco.trainer.reference_feature_store import ReferenceFeatureStore
from verl_speco.trainer.standalone_data import (
    ReadyEntry,
    StandaloneLocalBatch,
    _entry_from_wire,
    _entry_to_wire,
    build_assignments,
)
from verl_speco.trainer.standalone_retry import retry_io
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples


from verl_speco.transport.reference_dispatch import (
    ReferenceCommandAssembler,
    command_fingerprint,
)

logger = logging.getLogger(__name__)


class ScheduledFeatureDataLoader:
    """Poll TQ on rank 0, distribute keys, and fetch payloads on owner ranks."""

    def __init__(
        self,
        store: Any,
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
        reference_command_max_bytes: int = 256 * 1024 * 1024,
        reference_command_frame_bytes: int = 4 * 1024 * 1024,
        node_ranks_per_host: int = 0,
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
        self._reference_command_max_bytes = reference_command_max_bytes
        self._reference_command_frame_bytes = reference_command_frame_bytes
        if (
            type(node_ranks_per_host) is not int
            or node_ranks_per_host < 0
            or (
                node_ranks_per_host
                and (
                    node_ranks_per_host > self.world_size
                    or self.world_size % node_ranks_per_host
                )
            )
        ):
            raise ValueError("Invalid local reference owner topology")
        self.node_ranks_per_host = node_ranks_per_host
        ReferenceCommandAssembler(
            reference_command_max_bytes, reference_command_frame_bytes
        )
        if cleanup_after_load and (prepare_batch is None or store.uses_transfer_queue):
            raise ValueError(
                "Cleanup after load requires ray_reference batch preparation"
            )
        self._cleanup_entries: dict[str, ReadyEntry] = {}
        self._released_cleanup_keys: set[str] = set()
        if scheduled_commands is None:
            raise ValueError(
                "Feature consumption requires scheduler-assigned batch commands"
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
                "ScheduledFeatureDataLoader world_size does not match torch.distributed world size"
            )

    def __iter__(self) -> Iterator[StandaloneLocalBatch]:
        return self._iterate(materialize=True)

    def iter_assignments(self) -> Iterator[StandaloneLocalBatch]:
        """Receive metadata only; CPU workers materialize these batches later."""
        if self.prepare_batch is None or self.scheduled_commands is None:
            raise ValueError("Deferred preparation requires scheduled file assignments")
        return self._iterate(materialize=False)

    def _iterate(self, *, materialize: bool) -> Iterator[StandaloneLocalBatch]:
        self.store.connect()
        global_batch_size = self.batch_size * self.world_size
        owner_ready = False
        last_plan_id = -1
        command_history: dict[int, bytes] = {}
        assembler = ReferenceCommandAssembler(
            self._reference_command_max_bytes, self._reference_command_frame_bytes
        )
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
                        if command.get("kind") == "stop":
                            pass
                        elif command.get("kind") not in {
                            "batch",
                            "batch_fragment",
                            "wait",
                        }:
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
            if command.get("kind") == "batch_fragment":
                command = assembler.add(command)
                if command is None:
                    continue
            elif assembler.active is not None and command.get("kind") not in {
                "wait",
                "error",
            }:
                raise RuntimeError(
                    "Reference command was interrupted before all frames arrived"
                )
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
                    if assembler.active is not None:
                        raise RuntimeError(
                            "Reference command was interrupted before all frames arrived"
                        )
                    return
                continue
            if command.get("kind") == "error":
                raise RuntimeError(str(command.get("message") or "TQ discovery failed"))
            if command.get("kind") == "stop":
                return
            if command.get("kind") != "batch":
                raise RuntimeError(f"Unsupported TQ loader command: {command!r}")
            plan_id = command.get("plan_id")
            if plan_id is not None:
                if type(plan_id) is not int or plan_id < 0:
                    raise RuntimeError("Invalid scheduled TQ plan ID")
                if plan_id <= last_plan_id:
                    saved = command_history.get(plan_id)
                    if saved is None:
                        raise RuntimeError(
                            "Expired scheduled TQ command cannot be verified"
                        )
                    if saved != command_fingerprint(command):
                        raise RuntimeError("Conflicting duplicate scheduled TQ command")
                    continue
                last_plan_id = plan_id
                command_history[last_plan_id] = command_fingerprint(command)
                if len(command_history) > 4096:
                    del command_history[next(iter(command_history))]
            wire_assignments = command.get("assignments")
            if (
                not isinstance(wire_assignments, list)
                or len(wire_assignments) != self.world_size
            ):
                raise RuntimeError("TQ loader received malformed rank assignments")
            assignment_digest = None
            if not getattr(self.store, "uses_transfer_queue", True):
                keys, sequences = (
                    command.get("global_keys"),
                    command.get("global_sequence_nos"),
                )
                if (
                    not isinstance(keys, list)
                    or not keys
                    or any(not isinstance(key, str) or not key for key in keys)
                    or len(set(keys)) != len(keys)
                    or not isinstance(sequences, list)
                    or len(sequences) != len(keys)
                    or any(type(seq) is not int or seq < 0 for seq in sequences)
                    or len(set(sequences)) != len(sequences)
                    or any(not isinstance(group, list) for group in wire_assignments)
                ):
                    raise RuntimeError("Invalid reference batch assignment manifest")
                expected = dict(zip(keys, sequences))
                assigned = {}
                for item in (item for group in wire_assignments for item in group):
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("key"), str)
                        or not isinstance(item.get("tag"), dict)
                    ):
                        raise RuntimeError("Invalid reference batch assignment entry")
                    key, sequence = item["key"], item["tag"].get("sequence_no")
                    if (
                        key in assigned
                        or key not in expected
                        or type(sequence) is not int
                        or sequence != expected[key]
                    ):
                        raise RuntimeError(
                            "Reference batch assignments do not match manifest"
                        )
                    assigned[key] = sequence
                if assigned != expected:
                    raise RuntimeError(
                        "Reference batch assignments do not match manifest"
                    )
                assignment_digest = command_fingerprint(command).hex()
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
            padding_ranks = command.get("padding_ranks", [])
            if (
                not isinstance(padding_ranks, list)
                or any(
                    type(item) is not int or not 0 <= item < self.world_size
                    for item in padding_ranks
                )
                or len(set(padding_ranks)) != len(padding_ranks)
            ):
                raise RuntimeError("TQ loader received malformed padding ranks")
            is_padding_rank = self.rank in padding_ranks
            if padding_ranks:
                template = command.get("padding_reference")
                real_items = [item for group in wire_assignments for item in group]
                if (
                    self.prepare_batch is None
                    or self.store.uses_transfer_queue
                    or self.cleanup_after_load
                    or template not in real_items
                    or not isinstance(template, dict)
                    or template.get("key") not in command.get("global_keys", [])
                    or str(
                        template.get("tag", {})
                        .get("feature_contract", {})
                        .get("algorithm", "")
                    ).upper()
                    not in {"DSPARK", "DSV4_DSPARK", "DFLASH", "EAGLE3"}
                    or any(
                        bool(group) == (rank in padding_ranks)
                        for rank, group in enumerate(wire_assignments)
                    )
                ):
                    raise RuntimeError(
                        "Invalid padding reference or early cleanup configuration"
                    )
                if is_padding_rank:
                    local_entries = [_entry_from_wire(template)]
            node_entries = None
            if self.node_ranks_per_host and self.rank % self.node_ranks_per_host == 0:
                template = command.get("padding_reference")
                if any(
                    not wire_assignments[rank] and rank not in padding_ranks
                    for rank in range(self.rank, self.rank + self.node_ranks_per_host)
                ):
                    raise RuntimeError(
                        "Empty local reference assignment lacks a padding rank"
                    )
                node_entries = {
                    rank: [
                        _entry_from_wire(item)
                        for item in (wire_assignments[rank] or [template])
                    ]
                    for rank in range(self.rank, self.rank + self.node_ranks_per_host)
                }
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
                    remaining = cleanup_standalone_samples(self.store, local_entries)
                    if remaining:
                        raise RuntimeError(
                            f"Failed to release {len(remaining)} prepared references"
                        )
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
            yield StandaloneLocalBatch(
                local_keys=[]
                if is_padding_rank
                else [entry.key for entry in local_entries],
                local_samples=samples,
                global_keys=global_keys,
                global_sequence_nos=global_sequence_nos,
                command_wait_seconds=command_wait_seconds,
                tq_get_seconds=tq_get_seconds,
                cycle_started_at=cycle_started_at,
                plan_id=command.get("plan_id"),
                receipt_token=command.get("receipt_token"),
                input_identity=command.get("input_identity"),
                producer_incarnation=command.get("producer_incarnation"),
                assignment_digest=assignment_digest,
                node_file_entries=node_entries if not materialize else None,
                prepared_batch=prepared_batch,
                batch_prepare_seconds=batch_prepare_seconds,
                file_entries=local_entries if not materialize else None,
                cleanup_after_load=cleanup_after_load,
                is_padding_rank=is_padding_rank,
                has_padding_ranks=bool(padding_ranks),
                delivery_times=delivery_times,
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
            # Retain failed references for the outer retry, but do not delete a
            # successfully released handle twice if another release failed.
            if not cleanup_after_load:
                unknown = [
                    key
                    for key in global_keys
                    if key not in self._cleanup_entries
                    and key not in self._released_cleanup_keys
                ]
                if unknown:
                    raise ValueError("Completed batch includes an unknown cleanup key")
                entries = [
                    self._cleanup_entries[key]
                    for key in global_keys
                    if key in self._cleanup_entries
                ]
                remaining = set(
                    cleanup_standalone_samples(self.store, entries, attempts=1)
                )
                for entry in entries:
                    tag = entry.tag
                    reference = (
                        tag.get("hidden_states_path")
                        if tag.get("backend") == "file"
                        else tag.get("handle")
                    )
                    if reference not in remaining:
                        self._cleanup_entries.pop(entry.key, None)
                        self._released_cleanup_keys.add(entry.key)
                if remaining:
                    raise RuntimeError(
                        f"Failed to release {len(remaining)} completed references"
                    )
            for key in global_keys:
                self._cleanup_entries.pop(key, None)
                self._released_cleanup_keys.discard(key)

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


class ReferenceFeatureDataLoader(ScheduledFeatureDataLoader):
    """Prepare hidden references on their assigned ranks without TQ polling."""

    def __init__(self, store: ReferenceFeatureStore, *, prepare_batch=None, **options):
        if prepare_batch is None:
            raise ValueError(
                "Reference preparation requires scheduler-assigned batch commands"
            )
        super().__init__(store, prepare_batch=prepare_batch, **options)
