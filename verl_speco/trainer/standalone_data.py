# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Transport-independent assignments and rank-local standalone batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from verl_speco.trainer.feature_store import DraftFeatureSample


def validate_standalone_training_source(
    training: Any, *, scheduled_commands: Any = None
) -> None:
    """Reject retired training paths before activating either model or transport."""
    source = (
        str((training.get("feature_store") or {}).get("type") or "torch_shard")
        .strip()
        .lower()
    )
    if source in {"token_replay", "jsonl_token_replay", "jsonl"}:
        raise ValueError(
            "Token replay training was removed. Generate complete features with "
            "verl_speco.vllm_hidden_states_generate, or use the Ray standalone launcher."
        )
    if training.get("target_feature_pipeline"):
        raise ValueError(
            "target_feature_pipeline was removed; use the Ray standalone launcher"
        )
    if source in {"tq", "ray_reference"} and scheduled_commands is None:
        raise ValueError(
            "Streaming standalone training requires the Ray Producer/Consumer launcher "
            "verl_speco.standalone_tq_training_launcher."
        )


@dataclass(frozen=True)
class ReadyEntry:
    """One discoverable sample record; payload tensors are not loaded yet."""

    key: str
    tag: dict[str, Any]


@dataclass(frozen=True)
class StandaloneLocalBatch:
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
    entries: Sequence[ReadyEntry],
    *,
    batch_size: int,
    world_size: int,
    balance_by_tokens: bool = False,
    allow_uneven: bool = False,
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
            positions = (
                request.get("feature_positions") if isinstance(request, dict) else None
            )
            if not isinstance(positions, list) or not positions:
                raise ValueError(
                    "Token-balanced assignment requires feature position metadata"
                )
            lengths.append(len(positions))
        assignments: list[list[tuple[int, ReadyEntry]]] = [
            [] for _ in range(world_size)
        ]
        totals = [0] * world_size
        # LPT is deterministic; restore input order within each rank afterwards.
        for index in sorted(range(len(entries)), key=lambda i: (-lengths[i], i)):
            rank = min(
                (r for r in range(world_size) if len(assignments[r]) < capacities[r]),
                key=lambda r: (totals[r], r),
            )
            assignments[rank].append((index, entries[index]))
            totals[rank] += lengths[index]
        return [[entry for _, entry in sorted(group)] for group in assignments]
    result = []
    offset = 0
    for count in capacities:
        result.append(list(entries[offset : offset + count]))
        offset += count
    return result


def _entry_to_wire(entry: ReadyEntry) -> dict[str, Any]:
    return {"key": entry.key, "tag": dict(entry.tag)}


def _entry_from_wire(value: Any) -> ReadyEntry:
    if not isinstance(value, dict) or "key" not in value or "tag" not in value:
        raise TypeError(f"Invalid serialized ReadyEntry: {value!r}")
    if not isinstance(value["tag"], dict):
        raise TypeError("Serialized ReadyEntry.tag must be a mapping")
    return ReadyEntry(key=str(value["key"]), tag=dict(value["tag"]))
