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
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch.distributed as dist

from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.tq_feature_store import ReadyEntry, TQFeatureStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TQLocalBatch:
    """The payload owned by one rank plus rank-0's global cleanup keys."""

    local_keys: list[str]
    local_samples: list[DraftFeatureSample]
    global_keys: list[str] | None
    global_sequence_nos: list[int] | None = None


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
    ):
        self.store = store
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.01)
        self.drop_last = bool(drop_last)
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
        self.store.connect()
        global_batch_size = self.batch_size * self.world_size
        owner_ready = False
        while True:
            command: dict[str, Any] | None = None
            if self.rank == 0:
                try:
                    if not owner_ready:
                        owner_ready = self.store.owner_ready()
                        if not owner_ready:
                            time.sleep(self.poll_interval_seconds)
                            continue
                    ready = self.store.list_ready()
                    if len(ready) >= global_batch_size:
                        selected = ready[:global_batch_size]
                        assignments = build_assignments(
                            selected,
                            batch_size=self.batch_size,
                            world_size=self.world_size,
                        )
                        command = {
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
                                self.store.clear_many(tail_keys)
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
            if command.get("kind") == "error":
                raise RuntimeError(str(command.get("message") or "TQ discovery failed"))
            if command.get("kind") == "stop":
                return
            if command.get("kind") != "batch":
                raise RuntimeError(f"Unsupported TQ loader command: {command!r}")
            wire_assignments = command.get("assignments")
            if (
                not isinstance(wire_assignments, list)
                or len(wire_assignments) != self.world_size
            ):
                raise RuntimeError("TQ loader received malformed rank assignments")
            local_entries = [
                _entry_from_wire(item) for item in wire_assignments[self.rank]
            ]
            samples = self.store.get_many(local_entries)
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
            )

    def clear_completed_batch(self, global_keys: Sequence[str] | None) -> None:
        if self.rank != 0:
            return
        if not global_keys:
            raise ValueError(
                "rank 0 requires global_keys to clear a completed TQ batch"
            )
        self.store.clear_many(global_keys)

    def _broadcast_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        if not dist.is_initialized() or self.world_size == 1:
            if command is None:
                raise RuntimeError("rank 0 did not create a TQ loader command")
            return command
        payload: list[Any] = [command if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
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


__all__ = ["TQFeatureDataLoader", "TQLocalBatch", "build_assignments"]
