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

import hashlib
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from functools import partial
from queue import Empty
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.standalone_ray_runtime import get_queue
from verl_speco.producer.metrics import current_ray_node_id
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.tq_feature_store import ReadyEntry, TQFeatureStore
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples

logger = logging.getLogger(__name__)


def _command_fingerprint(command: dict[str, Any]) -> bytes:
    payload = json.dumps(command, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).digest()


class _ReferenceCommandAssembler:
    """Validate and reassemble bounded, duplicate-safe command fragments."""

    def __init__(self, max_bytes: int, frame_bytes: int):
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 1024**3:
            raise ValueError("Invalid reference command byte limit")
        if type(frame_bytes) is not int or not 1024 <= frame_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid reference command frame limit")
        self.max_bytes, self.frame_bytes = max_bytes, frame_bytes
        self.active = None
        self.recent = {}
        self.recent_hashes = 0

    def add(self, frame):
        fields = {"kind", "plan_id", "receipt_token", "index", "count", "nbytes", "sha256", "data"}
        if not isinstance(frame, dict) or set(frame) != fields or frame["kind"] != "batch_fragment":
            raise ValueError("Malformed reference command frame")
        plan, token = frame["plan_id"], frame["receipt_token"]
        index, count, size, digest, data = (frame[key] for key in ("index", "count", "nbytes", "sha256", "data"))
        if (type(plan) is not int or plan < 0 or not isinstance(token, str) or not 1 <= len(token) <= 128
                or type(size) is not int or not self.frame_bytes < size <= self.max_bytes
                or type(count) is not int or count != (size + self.frame_bytes - 1) // self.frame_bytes
                or type(index) is not int or not 0 <= index < count
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or type(data) is not bytes
                or len(data) != min(self.frame_bytes, size - index * self.frame_bytes)):
            raise ValueError("Invalid reference command frame bounds or identity")
        identity = (plan, token, count, size, digest)
        chunk_hash = hashlib.sha256(data).digest()
        prior = self.recent.get(plan)
        if prior is not None:
            if prior[0] != identity or prior[1][index] != chunk_hash:
                raise ValueError("Conflicting completed reference command frame")
            return None
        if self.active is None:
            if index != 0:
                raise ValueError("Reference command starts with an out-of-order frame")
            self.active = {"identity": identity, "data": bytearray(), "hashes": []}
        active = self.active
        if active["identity"] != identity:
            raise ValueError("Reference command frames have conflicting identities")
        next_index = len(active["hashes"])
        if index < next_index:
            if active["hashes"][index] != chunk_hash:
                raise ValueError("Conflicting duplicate reference command frame")
            return None
        if index != next_index:
            raise ValueError("Reference command frame is out of order")
        active["data"].extend(data)
        active["hashes"].append(chunk_hash)
        if len(active["hashes"]) < count:
            return None
        payload = active["data"]
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Reference command frame checksum mismatch")
        try:
            command = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Reference command frame payload is not valid JSON") from exc
        if (not isinstance(command, dict) or command.get("kind") != "batch"
                or command.get("plan_id") != plan or command.get("receipt_token") != token):
            raise ValueError("Reference command frame payload identity differs")
        hashes = tuple(active["hashes"])
        self.active = None
        if len(hashes) <= 8192:
            self.recent[plan] = (identity, hashes)
            self.recent_hashes += len(hashes)
            while len(self.recent) > 64 or self.recent_hashes > 8192:
                oldest = next(iter(self.recent))
                _, old_hashes = self.recent.pop(oldest)
                self.recent_hashes -= len(old_hashes)
        return command


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
    is_padding_rank: bool = False
    has_padding_ranks: bool = False
    delivery_times: dict[str, float] = field(default_factory=dict)
    receipt_token: str | None = None
    input_identity: dict[str, Any] | None = None
    producer_incarnation: str | None = None
    assignment_digest: str | None = None
    node_file_entries: dict[int, list[ReadyEntry]] | None = None


def build_assignments(
    entries: Sequence[ReadyEntry], *, batch_size: int, world_size: int,
    balance_by_tokens: bool = False, allow_uneven: bool = False,
    allow_empty: bool = False,
) -> list[list[ReadyEntry]]:
    """Split a complete batch; optionally balance feature tokens across ranks."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    expected = batch_size * world_size
    if type(allow_uneven) is not bool or type(allow_empty) is not bool:
        raise ValueError("allow_uneven and allow_empty must be booleans")
    if allow_empty and not allow_uneven:
        raise ValueError("Empty ranks require uneven assignment")
    minimum = 1 if allow_empty else world_size
    if len(entries) != expected and not (
        allow_uneven and minimum <= len(entries) < expected
    ):
        raise ValueError(
            f"Expected exactly {expected} ready entries for one global batch, got {len(entries)}"
        )
    if type(balance_by_tokens) is not bool:
        raise ValueError("balance_by_tokens must be a boolean")
    base, extra = divmod(len(entries), world_size)
    capacities = [base + int(rank < extra) for rank in range(world_size)]
    if balance_by_tokens:
        lengths = []
        for entry in entries:
            request = entry.tag.get("request")
            positions = request.get("feature_positions") if isinstance(request, dict) else None
            if not isinstance(positions, list) or not positions:
                raise ValueError("Token-balanced assignment requires feature position metadata")
            lengths.append(len(positions))
        assignments: list[list[tuple[int, ReadyEntry]]] = [[] for _ in range(world_size)]
        totals = [0] * world_size
        # LPT is deterministic; restore input order within each rank afterwards.
        for index in sorted(range(len(entries)), key=lambda i: (-lengths[i], i)):
            rank = min((r for r in range(world_size) if len(assignments[r]) < capacities[r]),
                       key=lambda r: (totals[r], r))
            assignments[rank].append((index, entries[index]))
            totals[rank] += lengths[index]
        return [[entry for _, entry in sorted(group)] for group in assignments]
    result = []
    offset = 0
    for count in capacities:
        result.append(list(entries[offset : offset + count]))
        offset += count
    return result


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
        if type(node_ranks_per_host) is not int or node_ranks_per_host < 0 or (node_ranks_per_host and
                (node_ranks_per_host > self.world_size or self.world_size % node_ranks_per_host)):
            raise ValueError("Invalid local reference owner topology")
        self.node_ranks_per_host = node_ranks_per_host
        _ReferenceCommandAssembler(reference_command_max_bytes, reference_command_frame_bytes)
        if cleanup_after_load and (prepare_batch is None or store.uses_transfer_queue):
            raise ValueError(
                "Cleanup after load requires ray_reference batch preparation"
            )
        self._cleanup_entries: dict[str, ReadyEntry] = {}
        self._released_cleanup_keys: set[str] = set()
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
        command_history: dict[int, bytes] = {}
        assembler = _ReferenceCommandAssembler(self._reference_command_max_bytes, self._reference_command_frame_bytes)
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
                        elif command.get("kind") not in {"batch", "batch_fragment", "wait"}:
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
            elif assembler.active is not None and command.get("kind") not in {"wait", "error"}:
                raise RuntimeError("Reference command was interrupted before all frames arrived")
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
                        raise RuntimeError("Reference command was interrupted before all frames arrived")
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
                        raise RuntimeError("Expired scheduled TQ command cannot be verified")
                    if saved != _command_fingerprint(command):
                        raise RuntimeError("Conflicting duplicate scheduled TQ command")
                    continue
                last_plan_id = plan_id
                command_history[last_plan_id] = _command_fingerprint(command)
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
                keys, sequences = command.get("global_keys"), command.get("global_sequence_nos")
                if (not isinstance(keys, list) or not keys
                        or any(not isinstance(key, str) or not key for key in keys)
                        or len(set(keys)) != len(keys)
                        or not isinstance(sequences, list) or len(sequences) != len(keys)
                        or any(type(seq) is not int or seq < 0 for seq in sequences)
                        or len(set(sequences)) != len(sequences)
                        or any(not isinstance(group, list) for group in wire_assignments)):
                    raise RuntimeError("Invalid reference batch assignment manifest")
                expected = dict(zip(keys, sequences))
                assigned = {}
                for item in (item for group in wire_assignments for item in group):
                    if (not isinstance(item, dict) or not isinstance(item.get("key"), str)
                            or not isinstance(item.get("tag"), dict)):
                        raise RuntimeError("Invalid reference batch assignment entry")
                    key, sequence = item["key"], item["tag"].get("sequence_no")
                    if key in assigned or key not in expected or type(sequence) is not int or sequence != expected[key]:
                        raise RuntimeError("Reference batch assignments do not match manifest")
                    assigned[key] = sequence
                if assigned != expected:
                    raise RuntimeError("Reference batch assignments do not match manifest")
                assignment_digest = _command_fingerprint(command).hex()
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
            if (not isinstance(padding_ranks, list)
                    or any(type(item) is not int or not 0 <= item < self.world_size for item in padding_ranks)
                    or len(set(padding_ranks)) != len(padding_ranks)):
                raise RuntimeError("TQ loader received malformed padding ranks")
            is_padding_rank = self.rank in padding_ranks
            if padding_ranks:
                template = command.get("padding_reference")
                real_items = [item for group in wire_assignments for item in group]
                if (self.prepare_batch is None or self.store.uses_transfer_queue
                        or self.cleanup_after_load or template not in real_items
                        or not isinstance(template, dict)
                        or template.get("key") not in command.get("global_keys", [])
                        or str(template.get("tag", {}).get("feature_contract", {}).get("algorithm", "")).upper()
                           not in {"DSPARK", "DFLASH", "EAGLE3"}
                        or any(bool(group) == (rank in padding_ranks)
                               for rank, group in enumerate(wire_assignments))):
                    raise RuntimeError("Invalid padding reference or early cleanup configuration")
                if is_padding_rank:
                    local_entries = [_entry_from_wire(template)]
            node_entries = None
            if self.node_ranks_per_host and self.rank % self.node_ranks_per_host == 0:
                template = command.get("padding_reference")
                if any(not wire_assignments[rank] and rank not in padding_ranks
                       for rank in range(self.rank, self.rank + self.node_ranks_per_host)):
                    raise RuntimeError("Empty local reference assignment lacks a padding rank")
                node_entries = {
                    rank: [_entry_from_wire(item) for item in (wire_assignments[rank] or [template])]
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
                        raise RuntimeError(f"Failed to release {len(remaining)} prepared references")
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
                local_keys=[] if is_padding_rank else [entry.key for entry in local_entries],
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
                unknown = [key for key in global_keys if key not in self._cleanup_entries
                           and key not in self._released_cleanup_keys]
                if unknown:
                    raise ValueError("Completed batch includes an unknown cleanup key")
                entries = [self._cleanup_entries[key] for key in global_keys if key in self._cleanup_entries]
                remaining = set(cleanup_standalone_samples(self.store, entries, attempts=1))
                for entry in entries:
                    tag = entry.tag
                    reference = (tag.get("hidden_states_path") if tag.get("backend") == "file"
                                 else tag.get("handle"))
                    if reference not in remaining:
                        self._cleanup_entries.pop(entry.key, None)
                        self._released_cleanup_keys.add(entry.key)
                if remaining:
                    raise RuntimeError(f"Failed to release {len(remaining)} completed references")
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


def _entry_to_wire(entry: ReadyEntry) -> dict[str, Any]:
    return {"key": entry.key, "tag": dict(entry.tag)}


def _entry_from_wire(value: Any) -> ReadyEntry:
    if not isinstance(value, dict) or "key" not in value or "tag" not in value:
        raise TypeError(f"Invalid serialized ReadyEntry: {value!r}")
    if not isinstance(value["tag"], dict):
        raise TypeError("Serialized ReadyEntry.tag must be a mapping")
    return ReadyEntry(key=str(value["key"]), tag=dict(value["tag"]))


def _prepared_host_bytes(batch):
    """Bound original tensor storages plus the copies made by the pin thread."""
    storages, pin_bytes = {}, 0
    for value in batch.values():
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
            pin_bytes += max(storage.nbytes(), value.numel() * value.element_size())
    return sum(storages.values()) + pin_bytes


def _reference_device_transfer_bytes(batch):
    """Conservatively account every tensor copied by the device staging path."""
    return sum(max(value.untyped_storage().nbytes(), value.numel() * value.element_size())
               for value in batch.values() if torch.is_tensor(value))


def _host_budget_depth(depth, max_host_bytes, max_batch_host_bytes):
    if (type(max_host_bytes) is not int or max_host_bytes < 0
            or type(max_batch_host_bytes) is not int or max_batch_host_bytes < 0
            or bool(max_host_bytes) != bool(max_batch_host_bytes)):
        raise ValueError("Prefetch host byte limits must both be non-negative integers and enabled together")
    if max_host_bytes:
        if max_host_bytes < max_batch_host_bytes:
            raise ValueError("Prefetch host bytes must cover one maximum-sized batch")
        depth = min(depth, max_host_bytes // max_batch_host_bytes)
    return depth


class _TrackedPrefetchExecutor(ThreadPoolExecutor):
    """Keep in-flight reads visible after the coordinator has stopped."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracked, self._tracked_lock = set(), threading.Lock()

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        with self._tracked_lock:
            self._tracked.add(future)
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future):
        with self._tracked_lock:
            self._tracked.discard(future)

    def pending(self):
        with self._tracked_lock:
            return {future for future in self._tracked if not future.done()}


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
        max_host_bytes: int = 0,
        max_batch_host_bytes: int = 0,
    ):
        depth = _host_budget_depth(depth, max_host_bytes, max_batch_host_bytes)
        if max_host_bytes and (not pin_memory or getattr(loader, "cleanup_after_load", False)):
            raise ValueError("Prefetch host limits require deferred-cleanup reference batches")
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        if prepare_workers < 0:
            raise ValueError("prepare_workers must be non-negative")
        if (isinstance(close_timeout, bool) or not isinstance(close_timeout, (int, float))
                or not math.isfinite(close_timeout) or close_timeout <= 0):
            raise ValueError("Prefetch close timeout must be a positive finite number")
        if pin_memory and getattr(loader, "prepare_batch", None) is None:
            raise ValueError("Independent pinning requires file batch preparation")
        self.loader = loader
        self.depth = depth
        self._max_batch_host_bytes = max_batch_host_bytes
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
                self._executor = _TrackedPrefetchExecutor(
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

    def _open_source(self, factory):
        source, error = None, None
        try:
            source = factory()
        except BaseException as exc:
            error = exc
        if self._sync([int(error is not None)])[0]:
            raise RuntimeError("Consumer prefetch source initialization failed on at least one rank") from error
        return source

    def _run(self) -> None:
        source = None
        try:
            source = self._open_source(lambda: iter(self.loader))
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
                    if self._max_batch_host_bytes:
                        if batch.prepared_batch is None:
                            raise RuntimeError("Reference prefetch returned no prepared batch")
                        if _prepared_host_bytes(batch.prepared_batch) > self._max_batch_host_bytes:
                            raise ValueError("Prepared reference batch exceeds consumer_prefetch_batch_bytes")
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
                        batch.delivery_times["_pin_queued_at"] = time.monotonic()
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
                delivery_times = dict(batch.delivery_times)
                queued_at = delivery_times.pop("_pin_queued_at", None)
                if queued_at is not None:
                    delivery_times["loader_pin_queue_time"] = (
                        time.monotonic() - queued_at
                    )
                if batch.prepared_batch is None:
                    raise RuntimeError("File pinning received no prepared batch")
                pinned = {
                    key: value.pin_memory() if torch.is_tensor(value) else value
                    for key, value in batch.prepared_batch.items()
                }
                delivery_times["loader_pin_time"] = time.perf_counter() - started
                delivery_times["_ready_at"] = time.monotonic()
                batch = replace(
                    batch,
                    delivery_times=delivery_times,
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
                if self._pin_thread is not None:
                    ready_at = batch.delivery_times.pop("_ready_at", None)
                    if ready_at is not None:
                        batch.delivery_times["loader_ready_queue_time"] = (
                            time.monotonic() - ready_at
                        )
                self._reserved -= 1
                self._condition.notify_all()
                return batch
            if self._error is not None:
                raise self._error
            raise StopIteration

    def _thread_work_running(self) -> bool:
        return (self._thread.is_alive() or (self._pin_thread is not None and self._pin_thread.is_alive())
                or (self._executor is not None and bool(self._executor.pending())))

    @property
    def has_running_work(self) -> bool:
        return self._thread_work_running()

    def close(self, *, timeout: float | None = None) -> None:
        timeout = self.close_timeout if timeout is None else timeout
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Prefetch close timeout must be a positive finite number")
        deadline = time.monotonic() + timeout
        self.stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._pin_thread is not None:
            self._pin_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            wait(self._executor.pending(), timeout=max(0.0, deadline - time.monotonic()))
        if self._thread_work_running():
            raise RuntimeError("TQ prefetch work did not stop before its shutdown deadline")
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
