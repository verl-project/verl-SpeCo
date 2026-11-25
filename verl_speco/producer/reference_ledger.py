# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Durable Producer leases and temporary Driver identity tracking."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path


class _ReferenceLeaseJournal:
    """Optional local durable lease ledger; unresolved owners require fencing.

    This is independent of optimizer progress. It never grants training credit
    or authorizes reusing a buffer from a prior unresolved incarnation.
    """

    def __init__(self, path, run_id, incarnation):
        path = Path(path).expanduser()
        if (
            not path.is_absolute()
            or path.is_symlink()
            or (path.exists() and not path.is_file())
        ):
            raise ValueError(
                "Reference lease journal requires an absolute regular-file path"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        try:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY CHECK(id=1), run_id TEXT, incarnation TEXT)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS leases (plan INTEGER PRIMARY KEY, phase TEXT NOT NULL, identity TEXT NOT NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS payloads (key TEXT PRIMARY KEY, identity TEXT NOT NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS acknowledgements (plan INTEGER, rank INTEGER, PRIMARY KEY(plan, rank)) WITHOUT ROWID"
            )
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                previous = self.db.execute(
                    "SELECT run_id, incarnation FROM owner WHERE id=1"
                ).fetchone()
                if previous is not None:
                    # Even an empty ledger can have an active writer. Explicit
                    # ownership handoff is required before accepting a new one.
                    raise RuntimeError(
                        "Reference lease journal already has an owner; fence and reconcile it before reuse"
                    )
                self.db.execute(
                    "INSERT INTO owner VALUES (1, ?, ?)", (run_id, incarnation)
                )
        except BaseException:
            self.db.close()
            raise

    def transition(self, plan_id, phase, identity):
        if phase not in {
            "DISPATCH_INTENT",
            "LEASED",
            "MATERIALIZED",
            "RELEASE_INTENT",
            "RELEASED",
        }:
            raise ValueError("Invalid persistent lease phase")
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(encoded.encode()) > 16 * 1024**2:
            raise ValueError("Reference lease journal record exceeds metadata budget")
        with self.db:
            previous = self.db.execute(
                "SELECT identity, phase FROM leases WHERE plan=?", (plan_id,)
            ).fetchone()
            if previous is not None and previous[0] != encoded:
                raise ValueError("Persistent reference lease identity differs")
            allowed = {
                None: {"DISPATCH_INTENT"},
                "DISPATCH_INTENT": {"LEASED"},
                "LEASED": {"MATERIALIZED", "RELEASE_INTENT"},
                "MATERIALIZED": {"RELEASE_INTENT"},
                "RELEASE_INTENT": {"RELEASED"},
                "RELEASED": set(),
            }
            old_phase = previous[1] if previous else None
            if phase != old_phase and phase not in allowed[old_phase]:
                raise ValueError("Invalid persistent lease transition")
            if phase in {"MATERIALIZED", "RELEASE_INTENT"}:
                owners = identity["owner_count"]
                count = self.db.execute(
                    "SELECT COUNT(*) FROM acknowledgements WHERE plan=?", (plan_id,)
                ).fetchone()[0]
                if count != owners:
                    raise ValueError(
                        "Persistent lease requires every owner acknowledgement"
                    )
            self.db.execute(
                "INSERT OR REPLACE INTO leases VALUES (?, ?, ?)",
                (plan_id, phase, encoded),
            )
            if phase == "RELEASED":
                self.db.execute(
                    "DELETE FROM leases WHERE phase='RELEASED' AND plan < ?",
                    (plan_id - 4096,),
                )
                self.db.execute(
                    "DELETE FROM acknowledgements WHERE plan NOT IN (SELECT plan FROM leases)"
                )

    def acknowledge(self, plan_id, rank):
        with self.db:
            record = self.db.execute(
                "SELECT phase, identity FROM leases WHERE plan=?", (plan_id,)
            ).fetchone()
            if record is None or record[0] not in {"LEASED", "MATERIALIZED"}:
                raise ValueError(
                    "Persistent acknowledgement requires a dispatched lease"
                )
            owners = json.loads(record[1])["owner_count"]
            if type(rank) is not int or not 0 <= rank < owners:
                raise ValueError("Persistent acknowledgement owner differs")
            self.db.execute(
                "INSERT OR IGNORE INTO acknowledgements VALUES (?, ?)", (plan_id, rank)
            )

    def admit(self, entries):
        with self.db:
            for entry in entries:
                encoded = json.dumps(
                    entry, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                if len(encoded.encode()) > 16 * 1024**2:
                    raise ValueError(
                        "Persistent payload identity exceeds metadata budget"
                    )
                self.db.execute(
                    "INSERT INTO payloads VALUES (?, ?)", (entry["key"], encoded)
                )

    def retire(self, keys):
        with self.db:
            self.db.executemany(
                "DELETE FROM payloads WHERE key=?", ((key,) for key in keys)
            )

    def snapshot(self, max_bytes=16 * 1024**2):
        owner = self.db.execute(
            "SELECT run_id, incarnation FROM owner WHERE id=1"
        ).fetchone()
        if owner is None:
            raise RuntimeError("Persistent lease snapshot requires an owned journal")
        snapshot = {
            "version": 1,
            "run_id": owner[0],
            "incarnation": owner[1],
            "leases": [],
            "payloads": [],
            "acknowledgements": [],
        }
        size = 1024
        for name, query in (
            (
                "leases",
                "SELECT plan, phase, identity FROM leases WHERE phase != 'RELEASED' ORDER BY plan",
            ),
            ("payloads", "SELECT key, identity FROM payloads ORDER BY key"),
            (
                "acknowledgements",
                "SELECT a.plan, a.rank FROM acknowledgements a "
                "JOIN leases l ON l.plan = a.plan WHERE l.phase != 'RELEASED' "
                "ORDER BY a.plan, a.rank",
            ),
        ):
            for row in self.db.execute(query):
                size += len(json.dumps(row, separators=(",", ":")).encode()) + 1
                if size > max_bytes:
                    raise ValueError(
                        "Persistent lease snapshot exceeds metadata budget"
                    )
                snapshot[name].append(list(row))
        encoded = json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(encoded) + 80 > max_bytes:
            raise ValueError("Persistent lease snapshot exceeds metadata budget")
        snapshot["sha256"] = hashlib.sha256(encoded).hexdigest()
        return snapshot

    def close(self, *, drained=False):
        if drained:
            with self.db:
                unresolved = self.db.execute(
                    "SELECT 1 FROM leases WHERE phase != 'RELEASED' LIMIT 1"
                ).fetchone()
                if (
                    unresolved is not None
                    or self.db.execute("SELECT 1 FROM payloads LIMIT 1").fetchone()
                    is not None
                ):
                    raise RuntimeError("Cannot clear a journal with unresolved leases")
                self.db.execute("DELETE FROM leases")
                self.db.execute("DELETE FROM acknowledgements")
                self.db.execute("DELETE FROM owner")
        self.db.close()


class _ReferenceIdentityLedger:
    """Per-run disk-backed identity index with a bounded SQLite page cache."""

    def __init__(self, max_entries=0):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError(
                "identity_ledger_max_entries must be a non-negative integer"
            )
        self.max_entries = max_entries
        self.count = 0
        self.directory = tempfile.TemporaryDirectory(prefix="speco-identities-")
        try:
            self.db = sqlite3.connect(
                str(Path(self.directory.name) / "identities.sqlite")
            )
            try:
                self.db.execute("PRAGMA cache_size=-2048")
                self.db.execute("PRAGMA temp_store=FILE")
                # The index is scoped to this live Driver and is never a
                # recovery artifact; no durable fsync is needed per event.
                self.db.execute("PRAGMA journal_mode=MEMORY")
                self.db.execute("PRAGMA synchronous=OFF")
                self.db.execute(
                    "CREATE TABLE identities (key_hash BLOB PRIMARY KEY, fingerprint TEXT NOT NULL) WITHOUT ROWID"
                )
                self.db.execute(
                    "CREATE TABLE batches (batch_id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL)"
                )
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
        row = self.db.execute(
            "SELECT fingerprint FROM identities WHERE key_hash = ?", (self._key(key),)
        ).fetchone()
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
            encoded = json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        fingerprint = digest.hexdigest()
        row = self.db.execute(
            "SELECT fingerprint FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is not None:
            if row[0] != fingerprint:
                raise ValueError("Repeated metadata batch ID has conflicting contents")
            return False
        with self.db:
            self.db.execute(
                "INSERT INTO batches (batch_id, fingerprint) VALUES (?, ?)",
                (batch_id, fingerprint),
            )
        return True

    def close(self):
        try:
            self.db.close()
        finally:
            self.directory.cleanup()
