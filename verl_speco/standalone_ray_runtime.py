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
import hashlib
import ipaddress
import json
import logging
import math
import os
import sqlite3
import subprocess
import tempfile
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verl_speco.config import prefetch_depth
from verl_speco.producer.metrics import current_ray_node_id
from verl_speco.producer.hidden_states_store import file_reference_tensor_bytes
from verl_speco.standalone_tq_training_launcher import (
    _VLLM_HIDDEN_STATES_DIR,
    PipelineCommands,
    _stop_process,
    _start_process,
    _cleanup_hidden_states_directory,
    _vllm_is_ready,
    _wait_for_owner_ready,
    _wait_for_vllm_ready,
    _verify_local_model_cache,
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
from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity, validate_input_identity
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples

logger = logging.getLogger(__name__)
_SCHEDULER_INFO_INTERVAL_STEPS = 50
_PRODUCER_NOTIFICATION_MAX_BATCH = 32
_PRODUCER_NOTIFICATION_MAX_DELAY_SECONDS = 0.2


_DEFAULT_REFERENCE_EVENT_BYTES = 16 * 1024 * 1024
_DEFAULT_REFERENCE_COMMAND_BYTES = 256 * 1024 * 1024
_DEFAULT_REFERENCE_FRAME_BYTES = 4 * 1024 * 1024


def _config_member(value, name, default=None):
    if hasattr(value, "get"):
        return value.get(name, default)
    return getattr(value, name, default)


def _reference_queue_metrics(plans, *, completed_samples: int, elapsed: float) -> dict[str, float]:
    """Snapshot driver-local state; never compare monotonic clocks across nodes."""
    materialized = sum(plan.materialized for plan in plans.pending)
    applied = sum(plan.optimizer_applied for plan in plans.pending)
    return {
        "scheduler/ready_samples": float(len(plans.ready)),
        "scheduler/reserved_samples": float(len(plans.reserved)),
        "scheduler/pending_plans": float(len(plans.pending)),
        "scheduler/awaiting_materialization_plans": float(len(plans.pending) - materialized),
        "scheduler/awaiting_optimizer_plans": float(materialized - applied),
        "scheduler/awaiting_cleanup_plans": float(applied),
        "scheduler/completed_samples": float(completed_samples),
        "scheduler/completed_samples_per_second": completed_samples / max(elapsed, 1e-9),
    }


def _reference_event_limit(config):
    speco = _config_member(config, "speco", {})
    producer = _config_member(speco, "standalone_tq_producer", {})
    limit = _config_member(producer, "reference_event_max_bytes", _DEFAULT_REFERENCE_EVENT_BYTES)
    if type(limit) is not int or not 1024 <= limit <= 256 * 1024 * 1024:
        raise ValueError("reference_event_max_bytes must be an integer between 1024 and 268435456")
    return limit


def _bounded_metadata_size(value, limit):
    """Count canonical JSON bytes incrementally before issuing a control RPC."""
    encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    size = 0
    for part in encoder.iterencode(value):
        size += len(part)  # ensure_ascii produces only single-byte characters
        if size > limit:
            raise ValueError(f"Reference control metadata exceeds {limit} bytes")
    return size


def _split_reference_metadata(entries, limit, clock_node_id, producer_incarnation=None):
    if not isinstance(entries, list) or not entries or len(entries) > 100_000:
        raise ValueError("Reference metadata requires 1 to 100000 entries")
    envelope = {"kind": "samples_ready", "batch_id": 2**63 - 1, "entries": [],
                "sent_at": -1.7976931348623157e308, "clock_node_id": clock_node_id}
    if producer_incarnation is not None:
        envelope["producer_incarnation"] = producer_incarnation
    overhead = _bounded_metadata_size(envelope, limit)
    sizes = [_bounded_metadata_size(entry, limit) for entry in entries]
    if any(overhead + size > limit for size in sizes):
        raise ValueError("One reference metadata entry exceeds the event byte limit")
    groups, group, used = [], [], overhead
    for entry, size in zip(entries, sizes, strict=True):
        extra = size + int(bool(group))
        if used + extra > limit:
            groups.append(group)
            group, used, extra = [], overhead, size
        group.append(entry)
        used += extra
    groups.append(group)
    return groups


def _reference_command_frames(command, *, max_bytes, frame_bytes):
    """Serialize large commands into ordered, independently retryable frames."""
    size = _bounded_metadata_size(command, max_bytes)
    if size <= frame_bytes:
        yield command
        return
    encoded = json.dumps(command, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(encoded) != size:
        raise RuntimeError("Reference command encoding size changed")
    digest = hashlib.sha256(encoded).hexdigest()
    count = (size + frame_bytes - 1) // frame_bytes
    for index in range(count):
        yield {"kind": "batch_fragment", "plan_id": command["plan_id"],
               "receipt_token": command["receipt_token"], "index": index, "count": count,
               "nbytes": size, "sha256": digest,
               "data": encoded[index * frame_bytes:(index + 1) * frame_bytes]}


class _ReferenceLeaseJournal:
    """Optional local durable lease ledger; unresolved owners require fencing.

    This is independent of optimizer progress. It never grants training credit
    or authorizes reusing a buffer from a prior unresolved incarnation.
    """

    def __init__(self, path, run_id, incarnation):
        path = Path(path).expanduser()
        if not path.is_absolute() or path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Reference lease journal requires an absolute regular-file path")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        try:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.execute("CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY CHECK(id=1), run_id TEXT, incarnation TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS leases (plan INTEGER PRIMARY KEY, phase TEXT NOT NULL, identity TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS payloads (key TEXT PRIMARY KEY, identity TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS acknowledgements (plan INTEGER, rank INTEGER, PRIMARY KEY(plan, rank)) WITHOUT ROWID")
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                previous = self.db.execute("SELECT run_id, incarnation FROM owner WHERE id=1").fetchone()
                if previous is not None:
                    # Even an empty ledger can have an active writer. Explicit
                    # ownership handoff is required before accepting a new one.
                    raise RuntimeError("Reference lease journal already has an owner; fence and reconcile it before reuse")
                self.db.execute("INSERT INTO owner VALUES (1, ?, ?)", (run_id, incarnation))
        except BaseException:
            self.db.close()
            raise

    def transition(self, plan_id, phase, identity):
        if phase not in {"DISPATCH_INTENT", "LEASED", "MATERIALIZED", "RELEASE_INTENT", "RELEASED"}:
            raise ValueError("Invalid persistent lease phase")
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 16 * 1024**2:
            raise ValueError("Reference lease journal record exceeds metadata budget")
        with self.db:
            previous = self.db.execute("SELECT identity, phase FROM leases WHERE plan=?", (plan_id,)).fetchone()
            if previous is not None and previous[0] != encoded:
                raise ValueError("Persistent reference lease identity differs")
            allowed = {None: {"DISPATCH_INTENT"}, "DISPATCH_INTENT": {"LEASED"},
                       "LEASED": {"MATERIALIZED", "RELEASE_INTENT"},
                       "MATERIALIZED": {"RELEASE_INTENT"}, "RELEASE_INTENT": {"RELEASED"}, "RELEASED": set()}
            old_phase = previous[1] if previous else None
            if phase != old_phase and phase not in allowed[old_phase]:
                raise ValueError("Invalid persistent lease transition")
            if phase in {"MATERIALIZED", "RELEASE_INTENT"}:
                owners = identity["owner_count"]
                count = self.db.execute("SELECT COUNT(*) FROM acknowledgements WHERE plan=?", (plan_id,)).fetchone()[0]
                if count != owners:
                    raise ValueError("Persistent lease requires every owner acknowledgement")
            self.db.execute("INSERT OR REPLACE INTO leases VALUES (?, ?, ?)", (plan_id, phase, encoded))
            if phase == "RELEASED":
                self.db.execute("DELETE FROM leases WHERE phase='RELEASED' AND plan < ?", (plan_id - 4096,))
                self.db.execute("DELETE FROM acknowledgements WHERE plan NOT IN (SELECT plan FROM leases)")

    def acknowledge(self, plan_id, rank):
        with self.db:
            record = self.db.execute("SELECT phase, identity FROM leases WHERE plan=?", (plan_id,)).fetchone()
            if record is None or record[0] not in {"LEASED", "MATERIALIZED"}:
                raise ValueError("Persistent acknowledgement requires a dispatched lease")
            owners = json.loads(record[1])["owner_count"]
            if type(rank) is not int or not 0 <= rank < owners:
                raise ValueError("Persistent acknowledgement owner differs")
            self.db.execute("INSERT OR IGNORE INTO acknowledgements VALUES (?, ?)", (plan_id, rank))

    def admit(self, entries):
        with self.db:
            for entry in entries:
                encoded = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if len(encoded.encode()) > 16 * 1024**2:
                    raise ValueError("Persistent payload identity exceeds metadata budget")
                self.db.execute("INSERT INTO payloads VALUES (?, ?)", (entry["key"], encoded))

    def retire(self, keys):
        with self.db:
            self.db.executemany("DELETE FROM payloads WHERE key=?", ((key,) for key in keys))

    def snapshot(self, max_bytes=16 * 1024**2):
        owner = self.db.execute("SELECT run_id, incarnation FROM owner WHERE id=1").fetchone()
        if owner is None:
            raise RuntimeError("Persistent lease snapshot requires an owned journal")
        snapshot = {"version": 1, "run_id": owner[0], "incarnation": owner[1],
                    "leases": [], "payloads": [], "acknowledgements": []}
        size = 1024
        for name, query in (
            ("leases", "SELECT plan, phase, identity FROM leases ORDER BY plan"),
            ("payloads", "SELECT key, identity FROM payloads ORDER BY key"),
            ("acknowledgements", "SELECT plan, rank FROM acknowledgements ORDER BY plan, rank"),
        ):
            for row in self.db.execute(query):
                size += len(json.dumps(row, separators=(",", ":")).encode()) + 1
                if size > max_bytes:
                    raise ValueError("Persistent lease snapshot exceeds metadata budget")
                snapshot[name].append(list(row))
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) + 80 > max_bytes:
            raise ValueError("Persistent lease snapshot exceeds metadata budget")
        snapshot["sha256"] = hashlib.sha256(encoded).hexdigest()
        return snapshot

    def close(self, *, drained=False):
        if drained:
            with self.db:
                unresolved = self.db.execute("SELECT 1 FROM leases WHERE phase != 'RELEASED' LIMIT 1").fetchone()
                if unresolved is not None or self.db.execute("SELECT 1 FROM payloads LIMIT 1").fetchone() is not None:
                    raise RuntimeError("Cannot clear a journal with unresolved leases")
                self.db.execute("DELETE FROM leases")
                self.db.execute("DELETE FROM acknowledgements")
                self.db.execute("DELETE FROM owner")
        self.db.close()


class _ReferenceIdentityLedger:
    """Per-run disk-backed identity index with a bounded SQLite page cache."""

    def __init__(self, max_entries=0):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("identity_ledger_max_entries must be a non-negative integer")
        self.max_entries = max_entries
        self.count = 0
        self.directory = tempfile.TemporaryDirectory(prefix="speco-identities-")
        try:
            self.db = sqlite3.connect(str(Path(self.directory.name) / "identities.sqlite"))
            try:
                self.db.execute("PRAGMA cache_size=-2048")
                self.db.execute("PRAGMA temp_store=FILE")
                # The index is scoped to this live Driver and is never a
                # recovery artifact; no durable fsync is needed per event.
                self.db.execute("PRAGMA journal_mode=MEMORY")
                self.db.execute("PRAGMA synchronous=OFF")
                self.db.execute("CREATE TABLE identities (key_hash BLOB PRIMARY KEY, fingerprint TEXT NOT NULL) WITHOUT ROWID")
                self.db.execute("CREATE TABLE batches (batch_id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL)")
            except BaseException:
                self.db.close()
                raise
        except BaseException:
            self.directory.cleanup()
            raise

    @staticmethod
    def _key(key):
        return hashlib.sha256(key.encode("utf-8")).digest()

    def get(self, key, default=None):
        row = self.db.execute("SELECT fingerprint FROM identities WHERE key_hash = ?", (self._key(key),)).fetchone()
        return row[0] if row is not None else default

    def update(self, values):
        if self.max_entries and self.count + len(values) > self.max_entries:
            raise ValueError("Reference identity ledger entry limit exceeded")
        with self.db:
            self.db.executemany(
                "INSERT INTO identities (key_hash, fingerprint) VALUES (?, ?)",
                ((self._key(key), value) for key, value in values.items()),
            )
        self.count += len(values)

    def record_batch(self, batch_id, entries):
        if type(batch_id) is not int or batch_id < 0:
            raise ValueError("Invalid reference publication batch ID")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Reference publication batch must not be empty")
        digest = hashlib.sha256()
        for item in entries:
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        fingerprint = digest.hexdigest()
        row = self.db.execute("SELECT fingerprint FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if row is not None:
            if row[0] != fingerprint:
                raise ValueError("Repeated metadata batch ID has conflicting contents")
            return False
        with self.db:
            self.db.execute("INSERT INTO batches (batch_id, fingerprint) VALUES (?, ?)", (batch_id, fingerprint))
        return True

    def close(self):
        try:
            self.db.close()
        finally:
            self.directory.cleanup()


def _accept_reference_entries(items, *, run_id, seen_keys, seen_references):
    """Validate a whole delivery before mutating the scheduler's identity ledger.

    One reference may belong to only one active sample at a time.
    This prevents an earlier batch from freeing data still needed by a later one.
    Retried deliveries of the same sample must have identical metadata.
    """
    from verl_speco.trainer.tq_feature_store import ReadyEntry
    from verl_speco.transport.drafter_sample_protocol import (
        SampleMetadata,
        make_sample_key,
        validate_reference_envelope,
    )

    new_keys, new_references, incoming = {}, {}, []
    for item in items:
        tag = item["tag"]
        validate_reference_envelope(tag)
        metadata = SampleMetadata.from_dict(tag)
        key = str(item["key"])
        if metadata.run_id != run_id or key != make_sample_key(metadata):
            raise ValueError("Producer file reference identity mismatch")
        fingerprint = hashlib.sha256(json.dumps(
            tag, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        previous = new_keys.get(key, seen_keys.get(key))
        if previous is not None:
            if previous != fingerprint:
                raise ValueError("Repeated sample has conflicting reference metadata")
            continue
        reference = (tag["backend"], tag["hidden_states_path"] if tag["backend"] == "file"
                     else tag["handle"])
        owner = new_references.get(reference, seen_references.get(reference))
        if owner is not None and owner != key:
            raise ValueError("Hidden reference belongs to multiple samples")
        new_keys[key] = fingerprint
        new_references[reference] = key
        incoming.append(ReadyEntry(key, tag))
    seen_keys.update(new_keys)
    seen_references.update(new_references)
    return incoming


def _release_reference_entries(entries, seen_references):
    """Forget ownership only after the completed batch has released its payloads."""
    for entry in entries:
        tag = entry.tag
        reference = (tag["backend"], tag["hidden_states_path"] if tag["backend"] == "file"
                     else tag["handle"])
        if seen_references.get(reference) == entry.key:
            del seen_references[reference]


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


def _uses_ray_reference(config: Any) -> bool:
    try:
        return (
            config.actor_rollout_ref.rollout.drafter.training.feature_store.type
            == "ray_reference"
        )
    except (AttributeError, KeyError):
        return False


def _balance_assignments_by_tokens(config: Any) -> bool:
    scheduler = config.speco.draft_training.get("scheduler") or {}
    enabled = scheduler.get("balance_by_tokens", False)
    if type(enabled) is not bool:
        raise ValueError("scheduler.balance_by_tokens must be a boolean")
    if enabled and not _uses_ray_reference(config):
        raise ValueError("Token-balanced assignment requires ray_reference input")
    return enabled


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
    return max(
        1,
        global_batch_size
        if _uses_ray_reference(config)
        else min(global_batch_size, _PRODUCER_NOTIFICATION_MAX_BATCH),
    )


class StandaloneProducerActor:
    """Async Ray actor facade around the existing standalone Producer."""

    def __init__(
        self,
        config: Any,
        events: Any | None = None,
        reference_batch_size: int | None = None,
        native_transport: Any | None = None,
    ):
        # The former module entrypoint configured INFO logging before calling
        # run_producer(). Ray invokes this class directly, so preserve that
        # behavior without replacing Ray's own log handler and actor prefix.
        _configure_driver_logging()
        self.config = config
        self._task: asyncio.Task[Any] | None = None
        self._stopping = False
        self._start_lock = asyncio.Lock()
        self._input_identity: dict[str, Any] | None = None
        self._run_gate = asyncio.Event()
        self._run_gate.set()
        self._events = events
        self._clock_node_id = current_ray_node_id()
        self._paused_at: float | None = None
        self._published_total = 0
        self._notified_published_total = 0
        self._notification_pending_since: float | None = None
        self._notification_finishing = False
        self._notification_wakeup = asyncio.Event()
        self._notification_task: asyncio.Task[None] | None = None
        self._notification_batch_size = _producer_notification_batch_size(config)
        if reference_batch_size is not None:
            self._notification_batch_size = max(1, int(reference_batch_size))
        self._metrics_pending: list[dict[str, Any]] = []
        self._reference_batch_id = 0
        self._reference_event_max_bytes = _reference_event_limit(config)
        speco = _config_member(config, "speco", {})
        producer = _config_member(speco, "standalone_tq_producer", {})
        file_root = _config_member(_config_member(producer, "hidden_states_store", {}), "file_root", None)
        self._reference_file_store = None
        if file_root is not None:
            from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
            self._reference_file_store = FileHiddenStatesStore(file_root)
        self._shared_file_probe = None
        self._reference_limit = _config_member(producer, "max_published_references", 0)
        parent_commit = _config_member(producer, "publication_after_parent_commit", False)
        if type(parent_commit) is not bool:
            raise ValueError("publication_after_parent_commit must be a boolean")
        if parent_commit:
            if type(reference_batch_size) is not int or reference_batch_size < 1:
                raise ValueError("publication_after_parent_commit requires an explicit global batch size")
            if (type(self._reference_limit) is not int or self._reference_limit < 0
                    or 0 < self._reference_limit < reference_batch_size):
                raise ValueError("max_published_references must cover a global batch")
            self._reference_limit = reference_batch_size
        self._execution_limit = _config_member(producer, "max_reference_execution_slots", 0)
        self._file_bytes_limit = _config_member(producer, "max_published_file_tensor_bytes", 0)
        self._single_file_bytes_limit = _config_member(producer, "max_single_file_tensor_bytes", 0)
        self._reserved_file_bytes_limit = _config_member(producer, "max_reserved_file_tensor_bytes", 0)
        if (type(self._reserved_file_bytes_limit) is not int or self._reserved_file_bytes_limit < 0
                or (self._reserved_file_bytes_limit and not self._file_bytes_limit)):
            raise ValueError("max_reserved_file_tensor_bytes requires enabled file byte limits")
        if (self._reserved_file_bytes_limit and
                self._reserved_file_bytes_limit < self._single_file_bytes_limit * self._notification_batch_size):
            raise ValueError("Reserved file tensor bytes must cover one maximum-sized global batch")
        self._slot_tracking = bool(self._execution_limit or self._reserved_file_bytes_limit)
        if (type(self._file_bytes_limit) is not int or self._file_bytes_limit < 0
                or type(self._single_file_bytes_limit) is not int or self._single_file_bytes_limit < 0
                or bool(self._file_bytes_limit) != bool(self._single_file_bytes_limit)):
            raise ValueError("File tensor byte limits must both be non-negative integers and enabled together")
        if self._file_bytes_limit and self._file_bytes_limit < self._single_file_bytes_limit * self._notification_batch_size:
            raise ValueError("File tensor byte limit must cover one maximum-sized global batch")
        if type(self._execution_limit) is not int or self._execution_limit < 0:
            raise ValueError("max_reference_execution_slots must be a non-negative integer")
        if self._execution_limit and self._execution_limit < self._notification_batch_size:
            raise ValueError("max_reference_execution_slots must cover a global training batch")
        self._credits_enabled = bool(self._reference_limit or self._execution_limit or self._file_bytes_limit)
        if type(self._reference_limit) is not int or self._reference_limit < 0:
            raise ValueError("max_published_references must be a non-negative integer")
        if self._reference_limit and self._reference_limit < self._notification_batch_size:
            raise ValueError("max_published_references must cover a global training batch")
        self._reference_condition = asyncio.Condition()
        self._reference_publication_lock = asyncio.Lock()
        self._reference_publication_started = False
        self._active_references: set[str] = set()
        if isinstance(native_transport, Mapping):
            from verl_speco.transport.native_feature_transport import (
                NativeSourceControlClient, NativeSourceControlRouter, NativeSourceDirectoryRouter,
            )
            native_transport = (NativeSourceDirectoryRouter(**native_transport) if "directory" in native_transport
                                else NativeSourceControlRouter(**native_transport) if "routes" in native_transport
                                else NativeSourceControlClient(**native_transport))
        self._native_transport = native_transport
        self._native_references: dict[str, dict[str, Any]] = {}
        self._active_file_bytes: dict[str, int] = {}
        self._published_file_bytes = 0
        self._execution_pending: set[int] = set()
        self._execution_waiters: list[object] = []
        self._active_reference_sequences: dict[str, int] = {}
        self._registered_reference_plans: dict[int, tuple[str, tuple[str, ...]]] = {}
        self._rank_ack_counts: dict[int, int] = {}
        self._rank_acknowledgements: dict[int, set[int]] = {}
        self._assigned_references: set[str] = set()
        self._last_registered_reference_plan = -1
        self._released_reference_plans: dict[int, tuple[str, tuple[str, ...]]] = {}
        self._last_released_reference_plan = -1
        self._discarded_reference_batches: dict[str, str] = {}
        self._reference_closed = False
        self._incarnation = uuid.uuid4().hex
        self._reference_aborted = False
        self._lease_journal = None
        self._native_sealed = False
        self._lease_journal_path = _config_member(producer, "reference_lease_journal_path", None)
        if self._lease_journal_path is not None and not self._credits_enabled:
            raise ValueError("Persistent reference leases require publication credits")

    def _abort_reference_state(self):
        """Set the common failure state while holding the reference condition."""
        self._reference_aborted = True
        self._reference_condition.notify_all()
        self._run_gate.set()

    def _ensure_lease_journal(self):
        if self._lease_journal_path is None:
            return
        try:
            if self._lease_journal is None:
                node = self.config
                for key in ("actor_rollout_ref", "rollout", "drafter", "training", "transfer_queue"):
                    node = _config_member(node, key, {})
                run_id = _config_member(node, "run_id", None)
                if not isinstance(run_id, str) or not run_id:
                    raise ValueError("Persistent leases require a run identity")
                self._lease_journal = _ReferenceLeaseJournal(self._lease_journal_path, run_id, self._incarnation)
        except BaseException:
            self._abort_reference_state()
            raise

    def _journal_lease(self, plan_id, phase, identity, owner_count=None, rank=None):
        if self._lease_journal_path is None:
            return
        try:
            self._ensure_lease_journal()
            record = {"receipt": identity[0], "keys": identity[1],
                      "owner_count": self._rank_ack_counts.get(plan_id, 0) if owner_count is None else owner_count,
                      "native_sources": {key: self._native_references[key] for key in identity[1] if key in self._native_references}}
            if rank is not None:
                self._lease_journal.acknowledge(plan_id, rank)
            else:
                self._lease_journal.transition(plan_id, phase, record)
        except BaseException:
            self._abort_reference_state()
            raise

    def _close_drained_lease_journal(self):
        if (self._lease_journal is not None and self._reference_closed and not self._reference_aborted
                and (self._native_transport is None or self._native_sealed)
                and not self._active_references and not self._execution_pending and not self._registered_reference_plans):
            try:
                self._lease_journal.close(drained=True)
                self._lease_journal = None
            except BaseException:
                self._abort_reference_state()
                raise

    async def reference_lease_snapshot(self):
        async with self._reference_condition:
            if self._lease_journal_path is None:
                raise ValueError("Persistent reference lease snapshot is not configured")
            if self._reference_aborted:
                raise RuntimeError("Cannot checkpoint an aborted reference lease owner")
            if self._lease_journal is None:
                if not self._reference_closed or self._active_references or self._execution_pending:
                    raise RuntimeError("Reference lease journal is unavailable before drain")
                snapshot = {"version": 1, "incarnation": self._incarnation, "drained": True}
                encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
                return {**snapshot, "sha256": hashlib.sha256(encoded).hexdigest()}
            return self._lease_journal.snapshot()

    def _journal_payloads(self, *, entries=None, retired=None):
        if self._lease_journal_path is None:
            return
        try:
            self._ensure_lease_journal()
            if entries is not None:
                self._lease_journal.admit(entries)
            if retired is not None:
                self._lease_journal.retire(retired)
        except BaseException:
            self._abort_reference_state()
            raise

    async def _on_metrics(
        self, published_total: int, metrics: dict[str, float]
    ) -> None:
        if self._events is not None:
            snapshot = dict(metrics)
            if self._credits_enabled:
                async with self._reference_condition:
                    snapshot.update({
                        "producer/reference_active": float(len(self._active_references)),
                        "producer/reference_assigned": float(len(self._assigned_references)),
                        "producer/reference_plans": float(len(self._registered_reference_plans)),
                    })
                    if self._slot_tracking:
                        snapshot.update({
                            "producer/reference_execution_pending": float(len(self._execution_pending)),
                            "producer/reference_execution_waiters": float(len(self._execution_waiters)),
                        })
                    if self._file_bytes_limit:
                        snapshot["producer/reference_file_tensor_bytes"] = float(self._published_file_bytes)
                    if self._native_transport is not None:
                        # Descriptor span includes its header/alignment; it excludes
                        # unpublished sources and allocator registration overhead.
                        snapshot["producer/reference_native_span_bytes"] = float(sum(
                            descriptor["nbytes"] for descriptor in self._native_references.values()
                        ))
                    if self._reserved_file_bytes_limit:
                        snapshot["producer/reference_reserved_file_tensor_bytes"] = float(
                            self._published_file_bytes + len(self._execution_pending) * self._single_file_bytes_limit
                        )
            self._metrics_pending.append(
                {
                    "kind": "producer_metrics",
                    "published_total": published_total,
                    "metrics": snapshot,
                }
            )
            self._notification_wakeup.set()

    async def _before_request(self) -> None:
        await self._run_gate.wait()
        async with self._reference_condition:
            if self._reference_aborted or self._reference_closed:
                raise RuntimeError("Reference admission stopped after pipeline failure or Producer closure")

    async def _close_reference_admission(self) -> None:
        async with self._reference_condition:
            self._reference_closed = True
            self._reference_condition.notify_all()
            self._close_drained_lease_journal()
        self._run_gate.set()

    async def abort_references(self, incarnation: str | None) -> None:
        """Fence new admission without releasing any existing source references."""
        async with self._reference_condition:
            if ((self._credits_enabled and incarnation != self._incarnation)
                    or (not self._credits_enabled and incarnation is not None and incarnation != self._incarnation)):
                raise ValueError("Reference abort has a stale producer incarnation")
            self._abort_reference_state()
        await self._stop_native_admission()

    async def _stop_native_admission(self) -> None:
        from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter

        if isinstance(self._native_transport, NativeSourceDirectoryRouter):
            await asyncio.to_thread(self._native_transport.stop_admission)

    async def _fail_native_transition(self) -> None:
        """Called under the reference condition; never return existing credits."""
        self._abort_reference_state()
        try:
            await self._stop_native_admission()
        except BaseException:
            logger.exception("Native admission freeze was not confirmed; retain source processes and buffers")

    async def _acquire_reference_slot(self, request: Any) -> None:
        sequence = request.sequence_no
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Reference execution slot requires a valid sequence number")
        async with self._reference_condition:
            if sequence in self._execution_pending or sequence in self._active_reference_sequences.values():
                raise ValueError("Duplicate reference execution sequence")
            ticket = object()
            self._execution_waiters.append(ticket)
            try:
                await self._reference_condition.wait_for(
                        lambda: self._reference_aborted or self._reference_closed or (self._execution_waiters[0] is ticket
                    and (not self._execution_limit or
                         len(self._execution_pending) + len(self._active_references) < self._execution_limit)
                    and (not self._reserved_file_bytes_limit or
                         self._published_file_bytes + (len(self._execution_pending) + 1) * self._single_file_bytes_limit
                         <= self._reserved_file_bytes_limit))
                )
                if self._reference_aborted or self._reference_closed:
                    raise RuntimeError("Reference admission stopped after pipeline failure or Producer closure")
                if sequence in self._execution_pending or sequence in self._active_reference_sequences.values():
                    raise ValueError("Duplicate reference execution sequence")
                self._execution_pending.add(sequence)
            finally:
                self._execution_waiters.remove(ticket)
                self._reference_condition.notify_all()

    async def _transfer_unpublished_slot(self, old_request: Any, new_request: Any) -> None:
        async with self._reference_condition:
            if self._reference_aborted:
                raise RuntimeError("Reference admission stopped after pipeline failure")
            old, new = old_request.sequence_no, new_request.sequence_no
            if (type(old) is not int or old not in self._execution_pending
                    or type(new) is not int or new < 0 or new in self._execution_pending
                    or new in self._active_reference_sequences.values()):
                raise ValueError("Invalid unpublished reference execution slot transfer")
            self._execution_pending.remove(old)
            self._execution_pending.add(new)

    async def _release_unpublished_slot(self, request: Any) -> None:
        async with self._reference_condition:
            sequence = request.sequence_no
            if type(sequence) is not int or sequence not in self._execution_pending:
                raise ValueError("Unpublished reference execution slot is not active")
            self._execution_pending.remove(sequence)
            self._reference_condition.notify_all()

    async def _on_references(self, entries: list[dict[str, Any]]) -> dict[str, float]:
        # Batch IDs and partial sends must stay ordered even if callers overlap.
        async with self._reference_publication_lock:
            try:
                return await self._publish_references(entries)
            except BaseException:
                # An RPC can have succeeded even if its acknowledgement was lost.
                if self._reference_publication_started:
                    await self.abort_references(self._incarnation if self._credits_enabled else None)
                raise
            finally:
                self._reference_publication_started = False

    async def _publish_references(self, entries: list[dict[str, Any]]) -> dict[str, float]:
        if self._events is None:
            raise RuntimeError("Ray file publication requires the Driver event queue")
        entries = [deepcopy(entry) if entry.get("tag", {}).get("backend") == "native" else entry
                   for entry in entries]
        native_references = {}
        native_fingerprints = {value["sha256"] for value in self._native_references.values()}
        for entry in entries:
            tag = entry.get("tag", {})
            if tag.get("backend") != "native":
                continue
            if self._native_transport is None or not self._credits_enabled or self._file_bytes_limit:
                raise ValueError("Native references require an in-process source and reference credits")
            from verl_speco.transport.drafter_sample_protocol import validate_reference_envelope
            validate_reference_envelope(tag)
            descriptor = json.loads(tag["handle"])
            key = entry.get("key")
            if (not isinstance(key, str) or not key or key in native_references
                    or descriptor["sha256"] in native_fingerprints):
                raise ValueError("Invalid or duplicate native source reference")
            await asyncio.to_thread(self._native_transport.validate_published, descriptor)
            native_references[key] = descriptor
            native_fingerprints.add(descriptor["sha256"])
        if self._reference_file_store is not None:
            for entry in entries:
                tag = entry.get("tag", {})
                if tag.get("backend") != "file" or not isinstance(tag.get("hidden_states_path"), str):
                    raise ValueError("Shared file root requires file references")
                self._reference_file_store.validate_reference(tag["hidden_states_path"])
        groups = _split_reference_metadata(
            entries, self._reference_event_max_bytes, self._clock_node_id,
            self._incarnation if self._credits_enabled else None,
        )
        file_sizes = {}
        if self._file_bytes_limit:
            keys = [entry.get("key") for entry in entries]
            if (any(not isinstance(key, str) or not key for key in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError("Invalid or duplicate file reference credit")
            for entry in entries:
                tag = entry.get("tag", {})
                if tag.get("backend") != "file" or not isinstance(tag.get("hidden_states_path"), str):
                    raise ValueError("File tensor byte accounting requires file references")
                size = await asyncio.to_thread(file_reference_tensor_bytes, tag["hidden_states_path"])
                if size > self._single_file_bytes_limit:
                    raise ValueError("File reference exceeds max_single_file_tensor_bytes")
                file_sizes[entry["key"]] = size
        metrics = {"attempts": 0.0, "transport_seconds": 0.0, "retry_wait_seconds": 0.0}
        for group in groups:
            while group:
                async with self._reference_condition:
                    await self._reference_condition.wait_for(
                        lambda: self._reference_aborted or self._reference_closed or ((not self._reference_limit or len(self._active_references) < self._reference_limit)
                        and (not self._file_bytes_limit or
                             self._published_file_bytes + file_sizes[group[0]["key"]] <= self._file_bytes_limit))
                    )
                    if self._reference_aborted:
                        raise RuntimeError("Reference publication stopped after pipeline failure")
                    if self._reference_closed:
                        raise RuntimeError("Reference publication stopped after Producer closure")
                    self._ensure_lease_journal()
                    self._reference_publication_started = True
                    count = min(len(group), self._reference_limit - len(self._active_references)) if self._reference_limit else len(group)
                    if self._file_bytes_limit:
                        available = self._file_bytes_limit - self._published_file_bytes
                        for index, item in enumerate(group[:count]):
                            size = file_sizes[item["key"]]
                            if size > available:
                                count = index
                                break
                            available -= size
                    portion, group = group[:count], group[count:]
                    keys = [item["key"] for item in portion]
                    if self._credits_enabled:
                        if (any(not isinstance(key, str) or not key for key in keys)
                                or len(set(keys)) != len(keys)
                                or any(key in self._active_references for key in keys)):
                            raise ValueError("Invalid or duplicate published reference credit")
                        self._journal_payloads(entries=portion)
                        if self._slot_tracking:
                            sequences = [item.get("tag", {}).get("sequence_no") for item in portion]
                            if (any(type(seq) is not int for seq in sequences)
                                    or len(set(sequences)) != len(sequences)
                                    or not set(sequences) <= self._execution_pending):
                                raise ValueError("Reference publication lacks its execution slot")
                            self._execution_pending.difference_update(sequences)
                            self._active_reference_sequences.update(zip(keys, sequences, strict=True))
                        self._active_references.update(keys)
                        self._native_references.update((key, native_references[key]) for key in keys
                                                       if key in native_references)
                        if self._file_bytes_limit:
                            self._active_file_bytes.update((key, file_sizes[key]) for key in keys)
                            self._published_file_bytes += sum(file_sizes[key] for key in keys)
                        if self._slot_tracking:
                            self._reference_condition.notify_all()
                if self._reference_batch_id >= 2**63:
                    raise ValueError("Reference metadata batch ID exceeded its bound")
                event = {"kind": "samples_ready", "batch_id": self._reference_batch_id,
                         "entries": portion, "sent_at": time.monotonic(), "clock_node_id": self._clock_node_id}
                if self._credits_enabled:
                    event["producer_incarnation"] = self._incarnation
                last_finished: float | None = None

                def submit():
                    nonlocal last_finished
                    started = time.monotonic()
                    if last_finished is not None:
                        metrics["retry_wait_seconds"] += started - last_finished
                    metrics["attempts"] += 1
                    try:
                        put_queue(self._events, event)
                    finally:
                        last_finished = time.monotonic()
                        metrics["transport_seconds"] += last_finished - started

                # Failed or ambiguous sends retain their credits until the actor stops.
                await asyncio.to_thread(retry_io, submit, name="Producer file metadata batch")
                self._reference_batch_id += 1
        return metrics

    async def create_shared_file_probe(self) -> dict[str, str]:
        """Publish a small identity witness from the Producer's actual filesystem."""
        if self._reference_file_store is None:
            raise ValueError("Shared file probe requires a configured shared root")
        if self._shared_file_probe is not None:
            return dict(self._shared_file_probe)
        token = uuid.uuid4().hex
        path = self._reference_file_store.validate_reference(
            str(self._reference_file_store.root / f".speco-shared-probe-{token}"))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                stream.write(token)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._shared_file_probe = {"path": str(path), "token": token}
        return dict(self._shared_file_probe)

    async def remove_shared_file_probe(self, path: str, token: str) -> None:
        if self._shared_file_probe != {"path": path, "token": token}:
            raise ValueError("Shared file probe identity differs")
        checked = self._reference_file_store.validate_reference(path)
        if not checked.is_file() or checked.read_text(encoding="ascii") != token:
            raise ValueError("Shared file probe changed before cleanup")
        checked.unlink()
        self._shared_file_probe = None

    async def _native_transition(self, keys: Sequence[str], method: str, **kwargs: Any) -> None:
        """Perform a source transition under the reference condition; retain credits on uncertainty."""
        try:
            for key in keys:
                if key in self._native_references:
                    await asyncio.to_thread(getattr(self._native_transport, method), self._native_references[key], **kwargs)
        except BaseException:
            await self._fail_native_transition()
            raise

    async def discard_unassigned_references(self, incarnation: str, token: str, keys: list[str]) -> None:
        """Retire never-dispatched references after Producer admission is closed."""
        if (not isinstance(token, str) or not 1 <= len(token) <= 256 or not isinstance(keys, list)
                or not keys or len(keys) > 100_000 or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("Invalid unassigned reference discard")
        digest = hashlib.sha256()
        for key in keys:
            encoded = key.encode("utf-8", "surrogatepass")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        identity = digest.hexdigest()
        async with self._reference_condition:
            if incarnation != self._incarnation:
                raise ValueError("Reference discard has a stale producer incarnation")
            if self._reference_aborted or not self._reference_closed:
                raise RuntimeError("Reference discard requires a closed, non-aborted Producer")
            previous = self._discarded_reference_batches.get(token)
            if previous is not None:
                if previous != identity:
                    raise ValueError("Conflicting reference discard")
                return
            if (not self._credits_enabled or not set(keys) <= self._active_references
                    or set(keys) & self._assigned_references):
                raise ValueError("Reference discard requires unassigned active references")
            if self._slot_tracking and not set(keys) <= self._active_reference_sequences.keys():
                raise RuntimeError("Reference execution slot accounting is inconsistent")
            if self._file_bytes_limit and (not set(keys) <= self._active_file_bytes.keys()
                    or sum(self._active_file_bytes[key] for key in keys) > self._published_file_bytes):
                raise RuntimeError("File tensor byte accounting is inconsistent")
            await self._native_transition(keys, "discard_unbound")
            self._journal_payloads(retired=keys)
            for key in keys:
                self._native_references.pop(key, None)
                if self._slot_tracking:
                    del self._active_reference_sequences[key]
                if self._file_bytes_limit:
                    self._published_file_bytes -= self._active_file_bytes.pop(key)
            self._active_references.difference_update(keys)
            self._discarded_reference_batches[token] = identity
            self._close_drained_lease_journal()
            if len(self._discarded_reference_batches) > 4096:
                del self._discarded_reference_batches[next(iter(self._discarded_reference_batches))]
            self._reference_condition.notify_all()

    async def seal_native_sources(self, incarnation: str) -> None:
        """Stop future source publications only after all references are retired."""
        from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter

        async with self._reference_condition:
            if incarnation != self._incarnation:
                raise ValueError("Native source seal has a stale producer incarnation")
            if (self._reference_aborted or not self._reference_closed or self._active_references
                    or self._execution_pending or self._registered_reference_plans
                    or not isinstance(self._native_transport, NativeSourceDirectoryRouter)):
                raise RuntimeError("Native source seal requires a stopped, drained Producer")
            try:
                await asyncio.to_thread(self._native_transport.freeze_and_seal)
                self._native_sealed = True
                self._close_drained_lease_journal()
            except BaseException:
                await self._fail_native_transition()
                raise

    async def register_dispatch(self, incarnation: str, plan_id: int, receipt_token: str, keys: list[str],
                                owner_count: int = 0) -> None:
        """Bind active references to a plan before its training command is sent."""
        if (type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str)
                or not receipt_token or not isinstance(keys, list) or not keys
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys) or type(owner_count) is not int
                or not 0 <= owner_count <= 65536):
            raise ValueError("Invalid reference dispatch")
        identity = (receipt_token, tuple(keys))
        async with self._reference_condition:
            if incarnation != self._incarnation:
                raise ValueError("Reference dispatch has a stale producer incarnation")
            if self._reference_aborted:
                raise RuntimeError("Reference dispatch stopped after pipeline failure")
            if self._lease_journal_path is not None and owner_count == 0:
                raise ValueError("Persistent reference leases require per-rank acknowledgements")
            previous = self._registered_reference_plans.get(plan_id, self._released_reference_plans.get(plan_id))
            if previous is not None:
                if previous != identity or self._rank_ack_counts.get(plan_id) != owner_count:
                    raise ValueError("Conflicting reference dispatch")
                return
            if plan_id != self._last_registered_reference_plan + 1:
                raise ValueError("Expired or out-of-order reference dispatch")
            if (not self._credits_enabled or not set(keys) <= self._active_references
                    or set(keys) & self._assigned_references):
                raise ValueError("Reference dispatch does not match unassigned active references")
            native = [self._native_references[key] for key in keys if key in self._native_references]
            if native and (not 1 <= owner_count <= 4096 or len(receipt_token) > 256):
                raise ValueError("Native reference dispatch requires bounded per-rank acknowledgements and receipt")
            self._journal_lease(plan_id, "DISPATCH_INTENT", identity, owner_count)
            if native:
                await self._native_transition(keys, "bind_readers", readers=list(range(owner_count)),
                                              plan_id=plan_id, receipt_token=receipt_token)
            self._registered_reference_plans[plan_id] = identity
            self._rank_ack_counts[plan_id] = owner_count
            self._rank_acknowledgements[plan_id] = set()
            self._assigned_references.update(keys)
            self._last_registered_reference_plan = plan_id
            self._journal_lease(plan_id, "LEASED", identity)

    async def acknowledge_reference_rank(self, incarnation: str, plan_id: int, receipt_token: str, rank: int) -> None:
        """Record an idempotent rank confirmation for a registered dispatch."""
        if (type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str)
                or not receipt_token or type(rank) is not int):
            raise ValueError("Invalid reference rank acknowledgement")
        async with self._reference_condition:
            if incarnation != self._incarnation:
                raise ValueError("Reference rank acknowledgement has a stale producer incarnation")
            if self._reference_aborted:
                raise RuntimeError("Reference acknowledgement stopped after pipeline failure")
            identity = self._registered_reference_plans.get(plan_id, self._released_reference_plans.get(plan_id))
            owner_count = self._rank_ack_counts.get(plan_id, 0)
            if identity is None or identity[0] != receipt_token or not 0 <= rank < owner_count:
                raise ValueError("Reference rank acknowledgement does not match registered dispatch")
            if plan_id in self._registered_reference_plans:
                if rank in self._rank_acknowledgements[plan_id]:
                    return
                await self._native_transition(identity[1], "acknowledge", reader=rank,
                                              plan_id=plan_id, receipt_token=receipt_token)
                self._journal_lease(plan_id, "LEASED", identity, rank=rank)
                self._rank_acknowledgements[plan_id].add(rank)
                if len(self._rank_acknowledgements[plan_id]) == owner_count:
                    self._journal_lease(plan_id, "MATERIALIZED", identity)
                    # Readers may now proceed to optimizer/finalization, whose
                    # cleanup can run before Driver completion polling.
                    self._journal_lease(plan_id, "RELEASE_INTENT", identity)

    async def release_references(self, incarnation: str, plan_id: int, receipt_token: str, keys: list[str]) -> None:
        """Return payload credits for an identity-bound plan; grant no optimizer credit."""
        if (type(plan_id) is not int or plan_id < 0 or not isinstance(receipt_token, str)
                or not receipt_token or not isinstance(keys, list) or not keys
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("Invalid reference credit release")
        identity = (receipt_token, tuple(keys))
        async with self._reference_condition:
            if incarnation != self._incarnation:
                raise ValueError("Reference credit release has a stale producer incarnation")
            if self._reference_aborted:
                raise RuntimeError("Reference credit release stopped after pipeline failure")
            previous = self._released_reference_plans.get(plan_id)
            if previous is not None:
                if previous != identity:
                    raise ValueError("Conflicting reference credit release")
                return
            if plan_id != self._last_released_reference_plan + 1:
                raise ValueError("Expired or out-of-order reference credit release")
            if not self._credits_enabled or not set(keys) <= self._active_references:
                raise ValueError("Reference credit release does not match active references")
            if self._registered_reference_plans.get(plan_id) != identity:
                raise ValueError("Reference credit release does not match registered dispatch")
            if len(self._rank_acknowledgements[plan_id]) != self._rank_ack_counts[plan_id]:
                raise ValueError("Reference credit release requires every rank acknowledgement")
            if self._slot_tracking and not set(keys) <= self._active_reference_sequences.keys():
                raise RuntimeError("Reference execution slot accounting is inconsistent")
            if self._file_bytes_limit and (
                not set(keys) <= self._active_file_bytes.keys()
                or sum(self._active_file_bytes[key] for key in keys) > self._published_file_bytes
            ):
                raise RuntimeError("File tensor byte accounting is inconsistent")
            self._journal_lease(plan_id, "RELEASE_INTENT", identity)
            await self._native_transition(keys, "release", plan_id=plan_id, receipt_token=receipt_token)
            self._journal_lease(plan_id, "RELEASED", identity)
            self._journal_payloads(retired=keys)
            for key in keys:
                self._native_references.pop(key, None)
            del self._registered_reference_plans[plan_id]
            del self._rank_acknowledgements[plan_id]
            self._assigned_references.difference_update(keys)
            self._active_references.difference_update(keys)
            if self._file_bytes_limit:
                self._published_file_bytes -= sum(self._active_file_bytes.pop(key) for key in keys)
            if self._slot_tracking:
                for key in keys:
                    del self._active_reference_sequences[key]
            self._released_reference_plans[plan_id] = identity
            self._last_released_reference_plan = plan_id
            self._close_drained_lease_journal()
            if len(self._released_reference_plans) > 4096:
                oldest = next(iter(self._released_reference_plans))
                del self._released_reference_plans[oldest]
                del self._rank_ack_counts[oldest]
            self._reference_condition.notify_all()

    async def _on_published(self, sequence_no: int) -> None:
        del sequence_no
        if self._events is None:
            return
        self._published_total += 1
        if self._notification_pending_since is None:
            self._notification_pending_since = time.monotonic()
        self._notification_wakeup.set()

    async def _finish_notifications(self) -> None:
        self._notification_finishing = True
        self._notification_wakeup.set()
        if self._notification_task is None:
            return
        speco = _config_member(self.config, "speco", {})
        training = _config_member(speco, "draft_training", {})
        scheduler = _config_member(training, "scheduler", {}) or {}
        timeout = _config_member(scheduler, "producer_stop_timeout_seconds", 180)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("producer_stop_timeout_seconds must be a positive finite number")
        try:
            # Timeout does not cancel a queue RPC already running in a thread.
            await asyncio.wait_for(asyncio.shield(self._notification_task), timeout=timeout)
        except BaseException:
            async with self._reference_condition:
                await self._fail_native_transition()
            raise

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
        file_options: dict[str, Any] = {}
        if _uses_ray_reference(self.config):
            file_options = {
                "on_references": self._on_references,
                "reference_batch_size": self._notification_batch_size,
            }
            if self._input_identity is not None:
                file_options["expected_input_identity"] = self._input_identity
            if self._slot_tracking:
                file_options["acquire_reference_slot"] = self._acquire_reference_slot
                file_options["release_unpublished_slot"] = self._release_unpublished_slot
                file_options["transfer_unpublished_slot"] = self._transfer_unpublished_slot
            if self._native_transport is not None:
                from verl_speco.transport.native_feature_transport import NativeProducerReferenceStore, NativeSourceDirectoryRouter
                if isinstance(self._native_transport, NativeSourceDirectoryRouter):
                    file_options["hidden_states_store"] = NativeProducerReferenceStore(self._native_transport)
        try:
            if self._credits_enabled and not file_options:
                raise ValueError("Reference capacity limits require ray_reference mode")
            result = await run_producer(
                self.config,
                **file_options,
                before_request=self._before_request,
                on_published=None if file_options else self._on_published,
                on_metrics=self._on_metrics,
                get_runtime_state=lambda: (
                    "running" if self._run_gate.is_set() else "paused"
                ),
            )
        except BaseException as error:
            # Intentional stop retains the existing tail-drain protocol. Other
            # failures quarantine admission immediately, before Driver polling.
            if not isinstance(error, asyncio.CancelledError) or not self._stopping:
                try:
                    await self.abort_references(self._incarnation if self._credits_enabled else None)
                except BaseException:
                    logger.exception("Producer failed and source admission freeze was not confirmed")
            try:
                await self._finish_notifications()
            except BaseException:
                if isinstance(error, asyncio.CancelledError):
                    raise
                logger.exception("Producer notification shutdown failed after the primary failure")
            raise
        await self._finish_notifications()
        await self._close_reference_admission()
        if file_options:
            done_event = {"kind": "producer_done"}
            if self._credits_enabled:
                done_event["producer_incarnation"] = self._incarnation
            await asyncio.to_thread(
                retry_io,
                lambda: put_queue(self._events, done_event),
                name="Producer completion event",
            )
        return result

    async def start(self) -> dict[str, Any]:
        async with self._start_lock:
            if self._reference_aborted:
                raise RuntimeError("Cannot restart an aborted Producer actor")
            if self._native_sealed:
                raise RuntimeError("Cannot restart a Producer with sealed native sources")
            if self._task is not None and not self._task.done():
                return {"started": False,
                        **({"producer_incarnation": self._incarnation} if self._credits_enabled else {}),
                        **({"input_identity": self._input_identity} if self._input_identity is not None else {})}
            speco = _config_member(self.config, "speco", {})
            producer_cfg = _config_member(speco, "standalone_tq_producer", {})
            input_path = _config_member(producer_cfg, "input_path", None)
            input_identity = None
            if _uses_ray_reference(self.config) and input_path:
                fingerprint = await asyncio.to_thread(build_input_fingerprint, input_path, include_content_hash=True)
                input_identity = input_content_identity(fingerprint)
            async with self._reference_condition:
                if self._reference_aborted:
                    raise RuntimeError("Cannot restart an aborted Producer actor")
                if (self._active_references or self._execution_pending or self._registered_reference_plans
                        or self._native_references or self._active_file_bytes or self._published_file_bytes):
                    raise RuntimeError("Cannot restart producer with outstanding reference credits")
                self._input_identity = input_identity
                self._released_reference_plans.clear()
                self._discarded_reference_batches.clear()
                self._rank_ack_counts.clear()
                self._rank_acknowledgements.clear()
                self._assigned_references.clear()
                self._last_registered_reference_plan = -1
                self._last_released_reference_plan = -1
                self._incarnation = uuid.uuid4().hex
                self._ensure_lease_journal()
                self._reference_closed = False
            self._published_total = 0
            self._reference_batch_id = 0
            self._notified_published_total = 0
            self._notification_pending_since = None
            self._notification_finishing = False
            self._notification_wakeup.clear()
            self._notification_task = None
            self._metrics_pending.clear()
            self._stopping = False
            self._task = asyncio.create_task(self._run_producer())
            return {"started": True,
                    **({"producer_incarnation": self._incarnation} if self._credits_enabled else {}),
                    **({"input_identity": self._input_identity} if self._input_identity is not None else {})}

    async def wait(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("Producer has not been started")
        return asdict(await self._task)

    async def stop(self) -> dict[str, bool]:
        # Wait for an in-progress start handshake before deciding there is no task.
        async with self._start_lock:
            task = self._task
            if task is None or task.done():
                await self._close_reference_admission()
                return {"stopped": False}
            self._stopping = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await self._close_reference_admission()
            return {"stopped": True}

    async def pause(self) -> dict[str, bool]:
        if self._reference_aborted:
            return {"paused": False}
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


def _create_pinned_trainer_groups(ray_module, *, node_ips, processes_per_node,
                                  device_name, name, timeout, placement_api=None):
    """Reserve one strict-packed trainer group per explicitly selected Ray node."""
    if (not isinstance(node_ips, (list, tuple)) or not node_ips
            or any(not isinstance(ip, str) or not ip.strip() or ip != ip.strip() for ip in node_ips)
            or len(set(node_ips)) != len(node_ips)):
        raise ValueError("trainer_node_ips must list distinct Ray node addresses")
    try:
        node_ips = [str(ipaddress.ip_address(ip)) for ip in node_ips]
    except ValueError:
        raise ValueError("trainer_node_ips must contain valid IP addresses") from None
    if len(set(node_ips)) != len(node_ips):
        raise ValueError("trainer_node_ips must list distinct Ray node addresses")
    node_ips.sort()  # Match verl's deterministic placement-group rank order.
    if (type(processes_per_node) is not int or processes_per_node < 1
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("trainer placement requires positive process count and timeout")
    resource = {"cuda": "GPU", "npu": "NPU"}.get(device_name)
    if resource is None:
        raise ValueError(f"Unsupported distributed trainer device: {device_name}")
    live = {}
    for node in ray_module.nodes():
        if node.get("Alive"):
            ip = node.get("NodeManagerAddress")
            if ip in live:
                raise ValueError(f"Multiple live Ray nodes share address {ip!r}")
            live[ip] = node.get("Resources", {})
    for ip in node_ips:
        resources = live.get(ip)
        if (resources is None or resources.get(f"node:{ip}", 0) < 0.001
                or resources.get(resource, 0) < processes_per_node
                or resources.get("CPU", 0) < processes_per_node):
            raise ValueError(f"Ray node {ip!r} lacks required {resource}/CPU capacity or node resource")
    if placement_api is None:
        from ray.util.placement_group import placement_group, remove_placement_group
    else:
        placement_group, remove_placement_group = placement_api
    groups = []
    try:
        for index, ip in enumerate(node_ips):
            bundles = [{"CPU": 1, resource: 1} for _ in range(processes_per_node)]
            bundles[0][f"node:{ip}"] = 0.001
            groups.append(placement_group(bundles, strategy="STRICT_PACK", name=f"{name}_{index}"))
        ray_module.get([group.ready() for group in groups], timeout=timeout)
        return groups
    except BaseException:
        for group in groups:
            try:
                remove_placement_group(group)
            except Exception:
                logger.exception("Failed to remove partially allocated trainer group")
        raise


def _create_unpinned_trainer_group(ray_module, *, processes_per_node, device_name,
                                   name, timeout, placement_api=None):
    """Reserve the single-node group with a bounded, cleanup-safe wait."""
    if (type(processes_per_node) is not int or processes_per_node < 1
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("trainer placement requires positive process count and timeout")
    resource = {"cuda": "GPU", "npu": "NPU"}.get(device_name, device_name)
    if not isinstance(resource, str) or not resource:
        raise ValueError("Unsupported trainer device")
    if placement_api is None:
        from ray.util.placement_group import placement_group, remove_placement_group
    else:
        placement_group, remove_placement_group = placement_api
    group = placement_group([{"CPU": 1, resource: 1} for _ in range(processes_per_node)],
                            strategy="STRICT_PACK", name=name)
    try:
        ray_module.get(group.ready(), timeout=timeout)
        return [group]
    except BaseException:
        try:
            remove_placement_group(group)
        except Exception:
            logger.exception("Failed to remove partially allocated trainer group")
        raise


def build_consumer_worker_group(
    config: Any,
    *,
    ray_module: Any,
    scheduled_commands: Any | None = None,
    training_events: Any | None = None,
    producer_handle: Any | None = None,
):
    """Create one standalone training WorkerGroup from ordinary draft config."""

    RayClassWithInitArgs, RayResourcePool, RayWorkerGroup = _ray_controller_types()
    from verl.utils.device import get_device_name

    from verl_speco.workers.standalone_drafter_worker import StandaloneDrafterWorker

    draft_training = config.speco.draft_training
    nnodes = int(draft_training.nnodes)
    nproc_per_node = int(draft_training.nproc_per_node)
    if nnodes < 1 or nproc_per_node < 1:
        raise ValueError("Trainer node and process counts must be positive")
    group_name = f"speco_standalone_{uuid.uuid4().hex}"
    resource_pool = RayResourcePool(
        process_on_nodes=[nproc_per_node] * nnodes,
        use_gpu=True,
        max_colocate_count=1,
        name_prefix=f"{group_name}_pool",
    )
    device_name = get_device_name()
    placement_timeout = float(draft_training.get("placement_timeout_seconds", 120))
    if nnodes > 1:
        node_ips = draft_training.get("trainer_node_ips")
        if node_ips is None or len(node_ips) != nnodes:
            raise ValueError("trainer_node_ips must contain one address per trainer node")
        owned_groups = _create_pinned_trainer_groups(
            ray_module, node_ips=list(node_ips), processes_per_node=nproc_per_node,
            device_name=device_name, name=group_name, timeout=placement_timeout,
        )
    else:
        owned_groups = _create_unpinned_trainer_group(
            ray_module, processes_per_node=nproc_per_node, device_name=device_name,
            name=group_name, timeout=placement_timeout,
        )
    resource_pool.pgs = owned_groups
    try:
        worker_cls = RayClassWithInitArgs(
            cls=ray_module.remote(StandaloneDrafterWorker),
            config=config,
            scheduled_commands=scheduled_commands,
            training_events=training_events,
            producer_handle=producer_handle,
        )
        return RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            name_prefix=f"{group_name}_consumer",
            device_name=device_name,
        )
    except BaseException:
        if owned_groups:
            from ray.util.placement_group import remove_placement_group
            for group in owned_groups:
                try:
                    remove_placement_group(group)
                except Exception:
                    logger.exception("Failed to remove trainer group after worker setup failure")
        raise


def _flatten_refs(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _raise_consumer_failure(event):
    if event.get("kind") == "consumer_failed":
        raise RuntimeError(
            f"Consumer rank={event.get('rank')} failed: {event.get('error')}"
        )
    if event.get("kind") == "finalization_failed":
        raise RuntimeError(
            f"Consumer failed to finalize trained plan={event.get('plan_id')}; "
            "see Consumer logs for the cleanup or completion-event error"
        )


def _pipeline_run_metrics(run_metrics: Mapping[str, Any], elapsed: float) -> dict[str, float]:
    """Combine rank-aggregated token counts with one Driver clock interval."""
    source = run_metrics.get("tokens/run_source")
    checkpoint = run_metrics.get("perf/run_checkpoint_blocking_time")
    if (isinstance(source, bool) or not isinstance(source, (int, float))
            or not math.isfinite(source) or source < 0 or not math.isfinite(elapsed) or elapsed <= 0):
        return {}
    result = {"perf/pipeline_active_seconds": elapsed, "tokens/pipeline_source_per_second": source / elapsed}
    if (not isinstance(checkpoint, bool) and isinstance(checkpoint, (int, float))
            and math.isfinite(checkpoint) and 0 <= checkpoint < elapsed):
        result["tokens/pipeline_source_per_second_excluding_checkpoint_wait"] = source / (elapsed - checkpoint)
    return result


class StandaloneRayTrainer:
    """Local driver matching ``SpecoRayPPOTrainer`` WorkerGroup ownership."""

    def __init__(
        self,
        commands: PipelineCommands,
        *,
        ray_module: Any,
        worker_group_factory: Callable[..., Any] = build_consumer_worker_group,
        queue_factory: Callable[..., Any] | None = None,
        feature_store_factory: Callable[[Any], Any] | None = None,
        pipeline_started_at: float | None = None,
        native_transport: Mapping[str, Any] | None = None,
    ):
        self.ray = ray_module
        self.producer_config = compose_runtime_config(commands.producer_overrides)
        self.consumer_config = compose_runtime_config(commands.consumer_overrides)
        self.worker_group_factory = worker_group_factory
        self.queue_factory = queue_factory
        self.feature_store_factory = feature_store_factory
        self.pipeline_started_at = pipeline_started_at
        self.native_transport = native_transport
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

        _balance_assignments_by_tokens(self.consumer_config)
        if self.queue_factory is None:
            from ray.util.queue import Queue

            self.queue_factory = Queue
        try:
            self.commands_queue = self.queue_factory()
            self.events_queue = (
                self.queue_factory(maxsize=128)
                if _uses_ray_reference(self.consumer_config)
                else self.queue_factory()
            )
            producer_cls = self.ray.remote(max_concurrency=4)(StandaloneProducerActor)
            producer_args = [self.producer_config, self.events_queue]
            if _uses_ray_reference(self.consumer_config):
                producer_args.append(self._queue_config().global_batch_size)
            if self.native_transport is not None:
                producer_args.append(self.native_transport)
            self.producer = producer_cls.options(
                name=f"speco_standalone_producer_{uuid.uuid4().hex}", num_cpus=1
            ).remote(*producer_args)
            worker_options = {"ray_module": self.ray, "scheduled_commands": self.commands_queue,
                              "training_events": self.events_queue}
            scheduler_cfg = self.consumer_config.speco.draft_training.get("scheduler") or {}
            source_store = _config_member(self.producer_config.speco.standalone_tq_producer, "hidden_states_store", {}) or {}
            if (scheduler_cfg.get("per_rank_reference_ack", False) or _config_member(source_store, "file_root", None) is not None
                    or _config_member(self.producer_config.speco.standalone_tq_producer, "reference_lease_journal_path", None) is not None):
                worker_options["producer_handle"] = self.producer
            self.consumer_wg = self.worker_group_factory(self.consumer_config, **worker_options)

        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release only this task's actors/queues, also on borrowed Ray sessions."""
        readers = list(getattr(self.consumer_wg, "workers", ()))
        for actor in [self.producer, *readers]:
            if actor is not None:
                try:
                    self.ray.kill(actor, no_restart=True)
                except Exception:
                    logger.exception("Failed to release standalone actor")
        self.reader_actors_exit_confirmed = False
        speco = _config_member(getattr(self, "consumer_config", {}), "speco", {})
        training = _config_member(speco, "draft_training", {})
        scheduler = _config_member(training, "scheduler", {}) or {}
        timeout = _config_member(scheduler, "reader_exit_timeout_seconds", 0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            logger.error("Invalid reader_exit_timeout_seconds; reader exit remains unconfirmed")
        elif timeout and readers:
            try:
                from ray.util.state import get_actor

                pending = {actor._actor_id.hex() for actor in readers}
                deadline = time.monotonic() + timeout
                while pending and time.monotonic() < deadline:
                    for actor_id in tuple(pending):
                        remaining = deadline - time.monotonic()
                        if remaining < 1:
                            break
                        state = get_actor(actor_id, timeout=max(1, min(5, int(remaining))))
                        # Missing/truncated state is not evidence of termination.
                        if state is not None and state.state == "DEAD":
                            pending.remove(actor_id)
                    if pending:
                        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                self.reader_actors_exit_confirmed = not pending
                if pending:
                    logger.error("Reader actor termination unconfirmed for %s task-owned actors", len(pending))
            except Exception:
                logger.exception("Could not confirm task-owned reader actor termination")
        # Actor DEAD does not prove orphan subprocesses or device RPCs are fenced.
        # It never authorizes source-buffer reuse by itself.
        self.producer = None
        for queue in (self.commands_queue, self.events_queue):
            if queue is not None:
                try:
                    queue.shutdown(force=True)
                except Exception:
                    logger.exception("Failed to release standalone queue")
        pool = getattr(self.consumer_wg, "resource_pool", None)
        if getattr(pool, "pgs", None):
            from ray.util.placement_group import remove_placement_group

            for group in pool.pgs:
                try:
                    remove_placement_group(group)
                except Exception:
                    logger.exception("Failed to release standalone placement group")
        self.consumer_wg = None
        self.commands_queue = self.events_queue = None

    def run(self) -> int:
        job_started_at = getattr(self, "pipeline_started_at", None)
        if job_started_at is None:
            job_started_at = time.monotonic()
        balance_by_tokens = _balance_assignments_by_tokens(self.consumer_config)
        scheduler_cfg = self.consumer_config.speco.draft_training.get("scheduler") or {}
        start_timeout = scheduler_cfg.get("producer_start_timeout_seconds", 1800)
        if (isinstance(start_timeout, bool) or not isinstance(start_timeout, (int, float))
                or not math.isfinite(start_timeout) or start_timeout <= 0):
            raise ValueError("scheduler.producer_start_timeout_seconds must be a positive finite number")
        stop_timeout = scheduler_cfg.get("producer_stop_timeout_seconds", 180)
        if (isinstance(stop_timeout, bool) or not isinstance(stop_timeout, (int, float))
                or not math.isfinite(stop_timeout) or stop_timeout <= 0):
            raise ValueError("scheduler.producer_stop_timeout_seconds must be a positive finite number")
        consumer_timeout = scheduler_cfg.get("consumer_shutdown_timeout_seconds", 1800)
        if (isinstance(consumer_timeout, bool) or not isinstance(consumer_timeout, (int, float))
                or not math.isfinite(consumer_timeout) or consumer_timeout <= 0):
            raise ValueError("scheduler.consumer_shutdown_timeout_seconds must be a positive finite number")
        commit_gate_timeout = scheduler_cfg.get("commit_gate_timeout_seconds", 180)
        if (isinstance(commit_gate_timeout, bool) or not isinstance(commit_gate_timeout, (int, float))
                or not math.isfinite(commit_gate_timeout) or commit_gate_timeout <= 0):
            raise ValueError("scheduler.commit_gate_timeout_seconds must be a positive finite number")
        per_rank_ack = scheduler_cfg.get("per_rank_reference_ack", False)
        if type(per_rank_ack) is not bool:
            raise ValueError("scheduler.per_rank_reference_ack must be a boolean")
        producer_options = _config_member(self.producer_config.speco, "standalone_tq_producer", {})
        if _config_member(producer_options, "reference_lease_journal_path", None) is not None and not per_rank_ack:
            raise ValueError("Persistent reference leases require per_rank_reference_ack=true")
        if per_rank_ack and not _uses_ray_reference(self.consumer_config):
            raise ValueError("per_rank_reference_ack requires ray_reference input")
        tail_policy = scheduler_cfg.get("tail_policy", "drop")
        if not isinstance(tail_policy, str) or tail_policy not in {"drop", "partial"}:
            raise ValueError("scheduler.tail_policy must be drop or partial")
        if tail_policy == "partial" and not _uses_ray_reference(self.consumer_config):
            raise ValueError("Partial tail batches require ray_reference input")
        source_settings = self.producer_config.speco.standalone_tq_producer
        source_store = _config_member(source_settings, "hidden_states_store", {}) or {}
        target_training = self.consumer_config.actor_rollout_ref.rollout.drafter.training
        target_replay = _config_member(target_training, "target_feature_replay", {}) or {}
        target_store = _config_member(target_replay, "hidden_states_store", {}) or {}
        source_root, target_root = _config_member(source_store, "file_root", None), _config_member(target_store, "file_root", None)
        if source_root != target_root or (source_root is not None and not _uses_ray_reference(self.consumer_config)):
            raise ValueError("Producer and consumer require the same shared file root with ray_reference input")
        parent_commit = _config_member(source_settings, "publication_after_parent_commit", False)
        if type(parent_commit) is not bool:
            raise ValueError("publication_after_parent_commit must be a boolean")
        if parent_commit and not _uses_ray_reference(self.consumer_config):
            raise ValueError("publication_after_parent_commit requires ray_reference input")
        reference_credits_enabled = bool(
            parent_commit or _config_member(source_settings, "max_published_references", 0)
            or _config_member(source_settings, "max_reference_execution_slots", 0)
            or _config_member(source_settings, "max_published_file_tensor_bytes", 0)
        )
        if per_rank_ack and not reference_credits_enabled:
            raise ValueError("per_rank_reference_ack requires Producer reference credits")
        native_source = _config_member(source_store, "backend", "file") == "native"
        native_target = _config_member(target_store, "backend", "file") == "native"
        release_on_materialization = scheduler_cfg.get("release_payload_on_materialization", False)
        if type(release_on_materialization) is not bool:
            raise ValueError("release_payload_on_materialization must be a boolean")
        if release_on_materialization and not (per_rank_ack and reference_credits_enabled and _uses_ray_reference(self.consumer_config)):
            raise ValueError("Early payload release requires reference input, credits and per-rank ACK")
        if native_source or native_target or self.native_transport is not None:
            if not (native_source and native_target and self.native_transport is not None
                    and _uses_ray_reference(self.consumer_config) and per_rank_ack and reference_credits_enabled):
                raise ValueError("Native transport requires matching stores, directory, reference credits and per-rank ACK")
        if self.producer is None or self.consumer_wg is None:
            self.init_workers()
        producer = self.producer
        consumer_wg = self.consumer_wg
        from queue import Empty

        from verl_speco.trainer.tq_feature_store import (
            ReferenceFeatureStore,
            ReadyEntry,
            TQFeatureStore,
        )
        from verl_speco.trainer.tq_sample_source import build_assignments

        assert self.commands_queue is not None and self.events_queue is not None
        config = self._queue_config()
        training_cfg = self.consumer_config.actor_rollout_ref.rollout.drafter.training
        tq_cfg = training_cfg.transfer_queue
        hidden_references = _uses_ray_reference(self.consumer_config)
        reference_event_limit = _reference_event_limit(self.consumer_config) if hidden_references else 0
        scheduler_settings = _config_member(self.consumer_config.speco.draft_training, "scheduler", {}) or {}
        command_limit = _config_member(scheduler_settings, "reference_command_max_bytes", _DEFAULT_REFERENCE_COMMAND_BYTES)
        if hidden_references and (type(command_limit) is not int or not 1024 <= command_limit <= 1024**3):
            raise ValueError("reference_command_max_bytes must be an integer between 1024 and 1073741824")
        frame_limit = _config_member(scheduler_settings, "reference_command_frame_bytes", _DEFAULT_REFERENCE_FRAME_BYTES)
        if hidden_references and (type(frame_limit) is not int or not 1024 <= frame_limit <= 32 * 1024 * 1024):
            raise ValueError("reference_command_frame_bytes must be an integer between 1024 and 33554432")
        clock_node_id = current_ray_node_id()
        source_cfg = tq_cfg
        if hidden_references:
            source_cfg = {
                **tq_cfg,
                "hidden_states_store": (
                    training_cfg.get("target_feature_replay") or {}
                ).get("hidden_states_store"),
            }
        if self.feature_store_factory is not None:
            store = self.feature_store_factory(source_cfg)
        elif native_source:
            from verl_speco.transport.native_feature_transport import NativeRetiredReferenceStore
            store = ReferenceFeatureStore(source_cfg, hidden_states_store=NativeRetiredReferenceStore())
        else:
            store = (ReferenceFeatureStore if hidden_references else TQFeatureStore).from_config(source_cfg)
        store.connect()
        scheduler = DrafterScheduler()
        depth = prefetch_depth(tq_cfg)
        io_attempts = int(tq_cfg.get("io_retry_max_attempts", 3))
        plans = StandaloneQueueState(depth + 1, config.global_batch_size)
        max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 0)) or 0)
        # The restored optimizer step is known only after Consumer activation.
        producer_cfg = self.producer_config.speco.standalone_tq_producer
        input_identity_required = hidden_references and bool(_config_member(producer_cfg, "input_path", None))
        consumer_ready = not (depth or max_steps > 0 or input_identity_required)
        initial_optimizer_step = 0
        producer_done = False
        producer_paused = False
        consumer_stop_requested = False
        pipeline_started = time.monotonic()
        initial_list_started = time.monotonic()
        initial_ready = (
            []
            if hidden_references
            else retry_io(
                store.list_ready, name="TQ scheduler list", attempts=io_attempts
            )
        )
        plans.refresh(initial_ready)
        ready_hint = len(initial_ready)
        tq_list_count = 0 if hidden_references else 1
        tq_list_seconds = (
            0.0 if hidden_references else time.monotonic() - initial_list_started
        )
        producer_pause_count = 0
        tail_dropped = 0
        completed_steps = 0
        completed_samples = 0
        last_published_total = 0
        last_reference_batch_id = -1
        producer_incarnation = None
        producer_input_identity = None
        seen_file_keys = None
        seen_file_references: dict[tuple[str, str], str] = {}
        reference_received_times: dict[str, tuple[float, float | None]] = {}
        shutdown_entries: dict[str, ReadyEntry] = {}
        pipeline_failed = False
        trackers: list[Any] = []
        final_run_metrics: dict[str, Any] = {}
        final_run_step: int | None = None

        def log_tracking_event(event: dict[str, Any]) -> None:
            nonlocal tq_list_count, tq_list_seconds, final_run_metrics, final_run_step
            if event.get("kind") == "training_metrics":
                metrics = dict(event["metrics"])
                if "tokens/run_source" in metrics:
                    final_run_metrics = dict(metrics)
                    final_run_step = int(event["step"])
                if hidden_references:
                    metrics.update(_reference_queue_metrics(
                        plans, completed_samples=completed_samples,
                        elapsed=time.monotonic() - pipeline_started,
                    ))
                _log_standalone_tracking_metrics(
                    trackers,
                    metrics,
                    step=int(event["step"]),
                )
            elif event.get("kind") == "producer_metrics":
                metrics = dict(event["metrics"])
                # Low-frequency producer statistics keep an actual TQ snapshot.
                # This query does not refresh selection caches or watermarks.
                started = time.monotonic()
                try:
                    if hidden_references:
                        metrics["queue/ready_samples"] = float(len(plans.ready))
                        metrics["queue/pending_samples"] = float(len(plans.reserved))
                    else:
                        metrics["tq/ready_samples"] = float(len(store.list_ready()))
                except Exception:
                    logger.exception("Failed to sample TQ size for Producer metrics")
                finally:
                    if not hidden_references:
                        tq_list_count += 1
                        tq_list_seconds += time.monotonic() - started
                _log_producer_tracking_metrics(
                    trackers,
                    metrics,
                    published_total=int(event["published_total"]),
                )

        def validate_producer_incarnation(event):
            if reference_credits_enabled:
                incarnation = event.get("producer_incarnation")
                if (not isinstance(incarnation, str) or len(incarnation) != 32
                        or any(char not in "0123456789abcdef" for char in incarnation)):
                    raise ValueError("Reference event requires a valid producer incarnation")
                if producer_incarnation != incarnation:
                    raise ValueError("Reference event has a stale producer incarnation")

        def discard_unassigned(entries):
            if not reference_credits_enabled or not entries:
                return
            keys = [entry.key for entry in entries]
            for start in range(0, len(keys), 4096):
                batch, token = keys[start:start + 4096], uuid.uuid4().hex
                retry_io(lambda: self.ray.get(producer.discard_unassigned_references.remote(
                    producer_incarnation, token, batch), timeout=float(tq_cfg.get("io_timeout_seconds", 30))),
                    name="Producer unassigned reference discard", attempts=io_attempts)

        def handle_shutdown_event(event):
            if pipeline_failed:
                return  # Unfenced references must remain untouched after failure.
            if event.get("kind") == "samples_ready":
                validate_producer_incarnation(event)
                entries = event.get("entries")
                if not isinstance(entries, list) or not 1 <= len(entries) <= 100_000:
                    raise ValueError("Reference publication batch must contain 1 to 100000 entries")
                _bounded_metadata_size(event, reference_event_limit)
                if seen_file_keys is None:
                    raise RuntimeError("Cannot verify shutdown reference identity")
                incoming = _accept_reference_entries(
                    entries, run_id=store.run_id,
                    seen_keys=seen_file_keys, seen_references=seen_file_references,
                )
                shutdown_entries.update((entry.key, entry) for entry in incoming)
            elif event.get("kind") == "producer_done":
                validate_producer_incarnation(event)
            elif event.get("kind") in {"training_metrics", "producer_metrics"}:
                log_tracking_event(event)

        world_size = int(
            self.consumer_config.speco.draft_training.nproc_per_node
        ) * int(self.consumer_config.speco.draft_training.nnodes)
        algorithm = str(self.consumer_config.actor_rollout_ref.rollout.drafter.get(
            "speculative_algorithm", ""
        )).upper() if hasattr(self.consumer_config.actor_rollout_ref.rollout.drafter, "get") else str(
            getattr(self.consumer_config.actor_rollout_ref.rollout.drafter, "speculative_algorithm", "")
        ).upper()
        supports_empty_ranks = algorithm in {"DSPARK", "DFLASH", "EAGLE3"}

        def submit_training_command(plan, selected_entries):
            assignments = build_assignments(
                selected_entries,
                batch_size=config.global_batch_size // world_size,
                world_size=world_size,
                balance_by_tokens=balance_by_tokens,
                allow_uneven=tail_policy == "partial",
                allow_empty=tail_policy == "partial" and supports_empty_ranks,
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
            if hidden_references and input_identity_required:
                command["input_identity"] = producer_input_identity
            if hidden_references:
                # A fresh capability distinguishes this live dispatch from an
                # old Driver incarnation even if a plan number is reused.
                plans.pending[-1].receipt_token = uuid.uuid4().hex
                command["receipt_token"] = plans.pending[-1].receipt_token
                if per_rank_ack:
                    command["producer_incarnation"] = producer_incarnation
            padding_ranks = [rank for rank, group in enumerate(assignments) if not group]
            if padding_ranks:
                template = selected_entries[0]
                command["padding_ranks"] = padding_ranks
                command["padding_reference"] = {"key": template.key, "tag": dict(template.tag)}
            if hidden_references:
                now = time.monotonic()
                received = [
                    reference_received_times.pop(entry.key)
                    for entry in selected_entries
                ]
                command["delivery_times"] = {
                    "scheduler_ready_wait_time": now - max(at for at, _ in received),
                }
                if all(age is not None for _, age in received):
                    command["delivery_times"]["metadata_event_age_time"] = max(
                        age for _, age in received
                    )
                command["dispatched_at"] = now
                command["clock_node_id"] = clock_node_id
            frames = (_reference_command_frames(command, max_bytes=command_limit, frame_bytes=frame_limit)
                      if hidden_references else (command,))
            if hidden_references and reference_credits_enabled:
                pending = plans.pending[-1]
                retry_io(
                    lambda: self.ray.get(producer.register_dispatch.remote(
                        producer_incarnation, pending.plan_id, pending.receipt_token, list(pending.keys),
                        *([world_size] if per_rank_ack else [])),
                        timeout=float(tq_cfg.get("io_timeout_seconds", 30))),
                    name="Producer reference dispatch registration", attempts=io_attempts,
                )
            for frame in frames:
                retry_io(
                    lambda frame=frame: put_queue(
                        self.commands_queue, frame,
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

        producer_ref = None
        producer_start_attempted = False
        producer_completed = False
        consumer_refs = []

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

        def release_materialized_payloads():
            # Keep source release FIFO even if prefetch receipts arrive early.
            # This does not remove a training plan or advance optimizer credit.
            for pending in plans.pending:
                if pending.payload_released:
                    continue
                if not pending.materialized:
                    break
                remaining = cleanup_standalone_samples(store, list(pending.entries), attempts=io_attempts,
                                                       reason="materialized")
                if remaining:
                    raise RuntimeError("Materialized source payload cleanup failed; source credits retained")
                retry_io(
                    lambda pending=pending: self.ray.get(producer.release_references.remote(
                        producer_incarnation, pending.plan_id, pending.receipt_token, list(pending.keys)),
                        timeout=float(tq_cfg.get("io_timeout_seconds", 30))),
                    name="Materialized source payload release", attempts=io_attempts,
                )
                plans.mark_payload_released(pending)

        def reconcile_and_schedule(trigger: str) -> bool:
            nonlocal producer_paused, consumer_stop_requested
            nonlocal tq_list_count, tq_list_seconds, producer_pause_count, tail_dropped
            refreshed = False

            def refresh():
                nonlocal refreshed, tq_list_count, tq_list_seconds
                if hidden_references:
                    refreshed = True
                    return
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
                if plans.ready and tail_policy == "partial":
                    tail_count = len(plans.ready)
                    if tail_count < world_size and not supports_empty_ranks:
                        raise RuntimeError(
                            f"Tail has {tail_count} samples for {world_size} ranks; "
                            f"empty-rank training is unsupported for {algorithm}; references retained"
                        )
                    if not consumer_refs:
                        raise RuntimeError("Standalone Consumer exited before the partial tail")
                    tail_context = replace(
                        context(tail_count), config=replace(config, global_batch_size=tail_count)
                    )
                    tail_plan = scheduler.plan_queue_training(
                        tail_context, selected_keys=tuple(entry.key for entry in plans.ready)
                    )
                    if not tail_plan.launch:
                        raise RuntimeError(f"Cannot schedule partial tail: {tail_plan.reason}")
                    pending, selected = plans.reserve(size=tail_count)
                    scheduler.execute_standalone_training_plan(
                        tail_plan, runtime_state=pending.runtime, selected_entries=selected,
                    )
                    logger.info("Scheduled final partial batch samples=%s ranks=%s", tail_count, world_size)
                    return False
                if plans.ready:
                    keys = [entry.key for entry in plans.ready]
                    if hidden_references:
                        discard_unassigned(list(plans.ready))
                    remaining = cleanup_standalone_samples(
                        store,
                        list(plans.ready),
                        reason="tail_dropped",
                        attempts=io_attempts,
                    )
                    if remaining:
                        raise RuntimeError(f"Failed to release {len(remaining)} dropped tail references")
                    tail_dropped += len(keys)
                    plans.ready.clear()
                scheduler.stop_standalone_consumer()
                consumer_stop_requested = True
                return True
            return False

        completed_normally = False
        try:
            if hidden_references:
                scheduler_cfg = self.consumer_config.speco.draft_training.get("scheduler") or {}
                seen_file_keys = _ReferenceIdentityLedger(scheduler_cfg.get("identity_ledger_max_entries", 0))
            producer_start_attempted = True
            try:
                started = self.ray.get(producer.start.remote(), timeout=start_timeout)
            except TimeoutError as error:
                raise TimeoutError("Standalone Producer start timed out; shutdown will be attempted") from error
            # Even a malformed startup receipt may follow a successful start.
            # Track the live actor before validating, so failure triggers stop.
            producer_ref = producer.wait.remote()
            if input_identity_required:
                producer_input_identity = validate_input_identity(started.get("input_identity") if isinstance(started, dict) else None)
            if hidden_references and reference_credits_enabled:
                producer_incarnation = started.get("producer_incarnation") if isinstance(started, dict) else None
                if (not isinstance(producer_incarnation, str) or len(producer_incarnation) != 32
                        or any(char not in "0123456789abcdef" for char in producer_incarnation)):
                    raise RuntimeError("Producer start did not establish a valid incarnation")
            consumer_refs = _flatten_refs(consumer_wg.run_standalone_training())
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
                        producer_completed = True
                        if hidden_references:
                            # The ordered completion event follows every metadata batch.
                            continue
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
                    if input_identity_required:
                        if event.get("input_identity_unavailable") is True and event.get("input_identity") is None:
                            logger.warning("Consumer cannot access the input path; binding Producer content identity from scheduled batches")
                        else:
                            consumer_identity = validate_input_identity(event.get("input_identity"))
                            if consumer_identity != producer_input_identity:
                                raise ValueError("Producer and Consumer input content differs")
                    initial_optimizer_step = int(event["initial_optimizer_step"])
                    consumer_ready = True
                    finished = reconcile_and_schedule("consumer_ready")
                elif kind == "samples_ready":
                    if not hidden_references:
                        raise RuntimeError("Unexpected file metadata on the TQ path")
                    if not isinstance(event.get("entries"), list) or not 1 <= len(event["entries"]) <= 100_000:
                        raise ValueError("Reference publication batch must contain 1 to 100000 entries")
                    _bounded_metadata_size(event, reference_event_limit)
                    validate_producer_incarnation(event)
                    batch_id = int(event["batch_id"])
                    if batch_id <= last_reference_batch_id:
                        if seen_file_keys.record_batch(batch_id, event["entries"]):
                            raise RuntimeError("Unknown historical Producer metadata batch")
                        continue
                    if batch_id != last_reference_batch_id + 1:
                        raise RuntimeError("Missing Producer metadata batch")
                    received_at = time.monotonic()
                    # Different Ray nodes need not share a monotonic clock origin.
                    event_age = (
                        received_at - float(event["sent_at"])
                        if clock_node_id is not None
                        and event.get("clock_node_id") == clock_node_id
                        and "sent_at" in event
                        else None
                    )
                    incoming = _accept_reference_entries(
                        event["entries"], run_id=store.run_id,
                        seen_keys=seen_file_keys, seen_references=seen_file_references,
                    )
                    if len(incoming) != len(event["entries"]):
                        raise ValueError("A new metadata batch repeats a previously published sample")
                    if not seen_file_keys.record_batch(batch_id, event["entries"]):
                        raise RuntimeError("New Producer metadata batch was already recorded")
                    for entry in incoming:
                        reference_received_times[entry.key] = (received_at, event_age)
                    plans.refresh([*plans.ready, *incoming])
                    ready_hint += len(incoming)
                    last_reference_batch_id = batch_id
                    finished = reconcile_and_schedule("samples_ready")
                elif kind == "producer_done":
                    if not hidden_references:
                        raise RuntimeError("Unexpected file Producer completion")
                    validate_producer_incarnation(event)
                    producer_done = True
                    finished = reconcile_and_schedule("producer_done")
                elif kind == "samples_published":
                    published_total = int(event.get("published_total", 0))
                    if published_total <= last_published_total:
                        continue
                    # initial_ready was captured before producer.start(). Keep
                    # event counts independent of subsequent list snapshots.
                    ready_hint += published_total - last_published_total
                    last_published_total = published_total
                    finished = reconcile_and_schedule("samples_published")
                elif kind in {"consumer_failed", "finalization_failed"}:
                    _raise_consumer_failure(event)
                elif kind == "reference_rank_materialized":
                    if not hidden_references or not per_rank_ack:
                        raise RuntimeError("Unexpected reference rank materialization acknowledgement")
                    plan_id, token, rank = event.get("plan_id"), event.get("receipt_token"), event.get("rank")
                    if release_on_materialization and event.get("source_payload_safe") is not True:
                        raise RuntimeError("Early payload release requires every rank's independent-copy receipt")
                    plans.mark_rank_materialized(plan_id, token, rank, world_size)
                elif kind == "training_materialized":
                    if not hidden_references:
                        raise RuntimeError("Unexpected reference materialization acknowledgement")
                    if release_on_materialization and event.get("source_payload_safe") is not True:
                        raise RuntimeError("Early payload release requires an aggregate source-safety receipt")
                    plans.mark_materialized(event.get("plan_id"), event.get("receipt_token"),
                                            required_ranks=world_size if per_rank_ack else 0)
                    if release_on_materialization:
                        release_materialized_payloads()
                elif kind == "training_applied":
                    if not hidden_references:
                        raise RuntimeError("Unexpected reference optimizer acknowledgement")
                    plans.mark_applied(event.get("plan_id"), event.get("receipt_token"))
                elif kind == "training_completed":
                    if event.get("successful") is not True:
                        raise RuntimeError("Invalid or unsuccessful training completion")
                    if hidden_references and (type(event.get("plan_id")) is not int or event["plan_id"] < 0):
                        raise RuntimeError("Reference completion requires a valid plan ID")
                    pending = plans.completion(
                        event.get("plan_id"), event.get("keys", ()),
                        receipt_token=event.get("receipt_token") if hidden_references else None,
                    )
                    if pending is None:
                        continue
                    if hidden_references and not pending.materialized:
                        raise RuntimeError("Reference training completed before all-rank materialization acknowledgement")
                    if hidden_references and not pending.optimizer_applied:
                        raise RuntimeError("Reference training completed before optimizer acknowledgement")
                    if hidden_references and reference_credits_enabled and not pending.payload_released:
                        retry_io(
                            lambda: self.ray.get(producer.release_references.remote(
                                producer_incarnation, pending.plan_id, pending.receipt_token, list(pending.keys)),
                                timeout=float(tq_cfg.get("io_timeout_seconds", 30))),
                            name="Producer reference credit release", attempts=io_attempts,
                        )
                        plans.mark_payload_released(pending)
                    training_outcome = scheduler.complete_standalone_training(
                        runtime_state=pending.runtime,
                        completed_keys=event.get("keys", ()),
                        successful=bool(event.get("successful", False)),
                    )
                    plans.release(pending)
                    if hidden_references:
                        _release_reference_entries(pending.entries, seen_file_references)
                    completed_steps += 1
                    completed_samples += training_outcome.assigned_samples
                    # Do not clamp the accumulator: late publication events
                    # must cancel the earlier subtraction exactly once.
                    ready_hint -= training_outcome.assigned_samples
                    if completed_steps % _SCHEDULER_INFO_INTERVAL_STEPS == 0:
                        logger.info(
                            "Standalone scheduler step=%s outstanding=%s buffered_samples=%s ready_list_count=%s ready_list_time=%.3fs",
                            completed_steps,
                            len(plans.pending),
                            ready_hint,
                            tq_list_count,
                            tq_list_seconds,
                        )
                    finished = reconcile_and_schedule("training_completed")
            if finished:
                # Consumers may emit events while exiting; drain the bounded
                # queue and bound shutdown for both supported store types.
                consumer_deadline = time.monotonic() + consumer_timeout
                while consumer_refs:
                    if time.monotonic() >= consumer_deadline:
                        raise TimeoutError("Standalone Consumer shutdown timed out")
                    completed, _ = self.ray.wait(consumer_refs, num_returns=1, timeout=0.1)
                    for ref in completed:
                        self.ray.get(ref)
                        consumer_refs.remove(ref)
                    try:
                        event = get_queue(self.events_queue, timeout=0.0)
                    except Empty:
                        continue
                    _raise_consumer_failure(event)
                    handle_shutdown_event(event)
            logger.info(
                "Standalone pipeline completed consumer_steps=%s tail_dropped=%s "
                "producer_pauses=%s ready_list_count=%s ready_list_time=%.3fs elapsed=%.3fs",
                completed_steps,
                tail_dropped,
                producer_pause_count,
                tq_list_count,
                tq_list_seconds,
                time.monotonic() - pipeline_started,
            )
            completed_normally = True
            return 0
        except Exception:
            pipeline_failed = True
            if hidden_references and producer_start_attempted and (producer_incarnation is not None or not reference_credits_enabled):
                try:
                    self.ray.get(producer.abort_references.remote(producer_incarnation), timeout=min(stop_timeout, 5.0))
                except Exception:
                    logger.exception("Failed to stop new reference admission before Producer shutdown")
            # These actors belong to this standalone task. Stop peers that
            # could otherwise remain blocked in training collectives.
            for worker in getattr(self.consumer_wg, "workers", ()):
                try:
                    self.ray.kill(worker, no_restart=True)
                except Exception:
                    logger.exception("Failed to stop standalone Consumer actor")
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
            shutdown_stop_error = None
            if producer_ref is not None or (producer_start_attempted and not producer_completed):
                try:
                    stop_ref = producer.stop.remote()
                    # Drain bounded events while stop() finishes, but never
                    # wait indefinitely for a blocked Producer in either mode.
                    stop_deadline = time.monotonic() + stop_timeout
                    while not self.ray.wait([stop_ref], num_returns=1, timeout=0.1)[0]:
                        if time.monotonic() >= stop_deadline:
                            raise TimeoutError("Standalone Producer stop timed out")
                        try:
                            event = get_queue(self.events_queue, timeout=0.0)
                        except Empty:
                            continue
                        handle_shutdown_event(event)
                    self.ray.get(stop_ref)
                except Exception as error:  # noqa: BLE001
                    shutdown_stop_error = error
                    logger.exception("Failed to stop standalone Producer actor")
            # Producer.stop() waits for the partial-window notification; workers
            # already awaited on normal completion may have final metric events.
            try:
                if completed_normally and shutdown_stop_error is not None:
                    raise RuntimeError("Failed to stop standalone Producer safely; unresolved references were not cleaned") from shutdown_stop_error
                while True:
                    try:
                        event = self.events_queue.get(block=False)
                    except Empty:
                        break
                    handle_shutdown_event(event)
                if hidden_references and completed_normally:
                    if plans.pending:
                        raise RuntimeError("Cannot discard references while dispatched plans remain unresolved")
                    shutdown_entries.update((entry.key, entry) for entry in plans.ready)
                    discard_unassigned(list(shutdown_entries.values()))
                    remaining = cleanup_standalone_samples(
                        store,
                        list(shutdown_entries.values()),
                        reason="shutdown",
                        attempts=io_attempts,
                    )
                    if remaining:
                        raise RuntimeError(f"Failed to release {len(remaining)} shutdown references")
                    if native_source:
                        self.ray.get(producer.seal_native_sources.remote(producer_incarnation), timeout=stop_timeout)
                elif hidden_references:
                    logger.warning(
                        "Preserving unresolved references after failure; "
                        "readers have not been fenced"
                    )
                if completed_normally and final_run_step is not None:
                    metrics = _pipeline_run_metrics(final_run_metrics, time.monotonic() - job_started_at)
                    if metrics:
                        _log_standalone_tracking_metrics(trackers, metrics, step=final_run_step)
                        logger.info("Standalone pipeline throughput: %s", metrics)
            finally:
                try:
                    _finish_standalone_tracking(trackers, rank=0)
                finally:
                    try:
                        store.close()
                    finally:
                        if seen_file_keys is not None:
                            seen_file_keys.close()


def run_ray_actors(
    commands: PipelineCommands,
    *,
    ray_module: Any,
    worker_group_factory: Callable[..., Any] = build_consumer_worker_group,
    pipeline_started_at: float | None = None,
    native_transport: Mapping[str, Any] | None = None,
) -> int:
    _configure_driver_logging()
    logger.info("Standalone Ray scheduler Driver starting")
    trainer = StandaloneRayTrainer(
        commands,
        ray_module=ray_module,
        worker_group_factory=worker_group_factory,
        pipeline_started_at=pipeline_started_at,
        native_transport=native_transport,
    )
    try:
        return trainer.run()
    finally:
        trainer.close()


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

    pipeline_started_at = time.monotonic()
    base_env = dict(os.environ if environ is None else environ)
    base_env["RAY_ADDRESS"] = ray_address
    # Mirror the subprocess path: fail fast when the selected TQ transport or
    # handle-based store points at an unreachable Mooncake master. This is the
    # default runtime mode.
    hidden_references = any(
        item.endswith("feature_store.type=ray_reference")
        for item in commands.consumer_overrides
    )
    if not hidden_references:
        validate_tq_backend(base_env)
    validate_hidden_states_store(base_env)
    owner = None
    vllm = None
    native = commands.native_settings
    native_directory = None
    native_drained = False
    native_unavailable = None
    if native is not None:
        native_unavailable = [endpoint for endpoint in commands.vllm_endpoints if not endpoint_ready(endpoint)]
        if commands.vllm is not None and not native_unavailable and not native["reuse_existing"]:
            raise RuntimeError("An existing native vLLM requires explicit SPECO_NATIVE_DIRECTORY and SPECO_NATIVE_RUN_ID")
    hidden_states_temp = None
    retained_hidden_dir = None
    try:
        if native is not None:
            from verl_speco.transport.native_feature_transport import NativeSourceDirectory

            native_directory = ray_module.remote(max_concurrency=128)(NativeSourceDirectory).options(
                name=native["directory"], namespace=native["namespace"], lifetime="detached", num_cpus=0,
            ).remote(run_id=native["run_id"], generation=native["generation"],
                     max_sources=native["max_sources"], max_records=native["max_records"],
                     max_descriptor_bytes=native["max_descriptor_bytes"], source_requirements={
                         "max_bytes": native["source_max_bytes"], "max_slots": native["source_max_slots"],
                         "chunk_bytes": native["chunk_bytes"], "metadata_server": native["metadata_server"],
                         "reader_endpoints": native["reader_endpoints"],
                     })
            scope = ray_module.get(native_directory.scope.remote(), timeout=native["timeout"])
            if scope != {"run_id": native["run_id"], "generation": native["generation"]}:
                raise RuntimeError("Native source directory scope differs")
            base_env.update({
                "SPECO_NATIVE_DIRECTORY": native["directory"], "SPECO_NATIVE_RAY_NAMESPACE": native["namespace"],
                "SPECO_NATIVE_RAY_ADDRESS": ray_address, "SPECO_NATIVE_RUN_ID": native["run_id"],
                "SPECO_NATIVE_GENERATION": str(native["generation"]),
                "SPECO_NATIVE_SOURCE_ENDPOINT": native["source_endpoint"],
                "SPECO_NATIVE_METADATA_SERVER": native["metadata_server"],
                "SPECO_NATIVE_SOURCE_MAX_BYTES": str(native["source_max_bytes"]),
                "SPECO_NATIVE_SOURCE_MAX_SLOTS": str(native["source_max_slots"]),
                "SPECO_NATIVE_CHUNK_BYTES": str(native["chunk_bytes"]),
                "SPECO_NATIVE_CONTROL_TIMEOUT": str(native["timeout"]),
            })
        with tempfile.TemporaryDirectory(prefix="speco-tq-launch-") as temp_dir:
            ready_file = Path(temp_dir) / "owner.ready"
            if hidden_references:
                retained_hidden_dir = Path(
                    tempfile.mkdtemp(prefix="speco-vllm-hidden-states-")
                )
                hidden_states_dir = str(retained_hidden_dir)
            else:
                hidden_states_temp = Path(tempfile.mkdtemp(
                    prefix="speco-vllm-hidden-states-"
                ))
                hidden_states_dir = str(hidden_states_temp)
            unavailable = (native_unavailable if native_unavailable is not None
                           else [e for e in commands.vllm_endpoints if not endpoint_ready(e)])
            if commands.local_model_cache and not unavailable:
                raise RuntimeError("Local model cache requires a new launcher-owned vLLM service")
            if unavailable:
                if commands.local_model_cache:
                    _verify_local_model_cache(*commands.local_model_cache)
                if commands.vllm is None:
                    raise RuntimeError(
                        "The configured hidden-state vLLM endpoints are unavailable: "
                        + ", ".join(unavailable)
                    )
                vllm_command = [
                    part.replace(_VLLM_HIDDEN_STATES_DIR, hidden_states_dir)
                    for part in commands.vllm
                ]
                vllm = _start_process(popen, vllm_command, env=base_env)
            if not hidden_references:
                owner_env = {**base_env, "SPECO_TQ_OWNER_READY_FILE": str(ready_file)}
                owner = _start_process(popen, commands.owner, env=owner_env)
                _wait_for_owner_ready(owner, ready_file, timeout_seconds=120)
            if vllm is not None:
                _wait_for_vllm_ready(
                    vllm,
                    commands.vllm_endpoints[0],
                    timeout_seconds=900,
                    endpoint_ready=endpoint_ready,
                )
            kwargs = {"ray_module": ray_module, "pipeline_started_at": pipeline_started_at}
            if native_directory is not None:
                kwargs["native_transport"] = {"directory": native_directory, "timeout": native["timeout"]}
            result = run_ray_actors(commands, **kwargs)
            native_drained = result == 0
            return result
    except KeyboardInterrupt:
        logger.warning("Standalone Ray training interrupted")
        return 130
    finally:
        if native_directory is not None and not native_drained:
            # Startup can fail before the Producer exists to propagate abort.
            # Preserve descriptors but stop further directory admission first.
            try:
                ray_module.get(native_directory.freeze.remote(False), timeout=min(5.0, native["timeout"]))
            except Exception:
                logger.exception("Native directory admission freeze was not confirmed during pipeline shutdown")
        shutdown_failures = []
        for process in (owner, vllm if native is None or native_drained else None):
            try:
                _stop_process(process)
            except Exception as exc:
                shutdown_failures.append(exc)
                logger.exception("Ray pipeline task-owned process shutdown was not confirmed")
        if shutdown_failures:
            native_drained = False
        if native is not None and not native_drained and vllm is not None:
            logger.error("Preserving native vLLM process pid=%s after incomplete shutdown; "
                         "fence all readers before stopping it", getattr(vllm, "pid", "unknown"))
        if native_directory is not None:
            if native_drained:
                try:
                    ray_module.kill(native_directory, no_restart=True)
                except Exception:
                    logger.exception("Failed to remove drained native source directory")
            else:
                logger.error("Preserving native Ray directory %s in namespace %s after incomplete shutdown; "
                             "remove it only after fencing readers and sources", native["directory"], native["namespace"])
                try:
                    status = ray_module.get(native_directory.status.remote(), timeout=min(5.0, native["timeout"]))
                    logger.error("Preserved native directory status: %s", status)
                except Exception:
                    logger.warning("Could not inspect the preserved native directory")
        if shutdown_failures and hidden_states_temp is not None:
            logger.error("Preserving hidden-state directory after incomplete process shutdown: %s", hidden_states_temp)
        elif hidden_states_temp is not None:
            _cleanup_hidden_states_directory(hidden_states_temp)
        if retained_hidden_dir is not None:
            try:
                retained_hidden_dir.rmdir()
            except OSError:
                logger.warning(
                    "Preserving remaining hidden files in %s", retained_hidden_dir
                )
        if shutdown_failures:
            raise RuntimeError("Standalone Ray subprocess shutdown was not confirmed") from shutdown_failures[0]


__all__ = [
    "StandaloneProducerActor",
    "StandaloneRayTrainer",
    "build_consumer_worker_group",
    "compose_runtime_config",
    "run_ray_actors",
    "run_ray_pipeline",
]
