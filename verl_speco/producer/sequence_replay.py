# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Committed, checksummed token sequences for standalone draft training.

The manifest is the commit record. A segment is never visible to a reader
until its descriptor has been atomically published. Only one process may write
one directory at a time; readers operate on an immutable manifest snapshot.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import sqlite3
import tempfile
import threading
import uuid
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_ROW_BYTES = 16 * 1024 * 1024
_MAX_RECORDS = 5_000_000


def plan_source_routes(lengths, *, source_count, max_samples, max_tokens, max_sequence_tokens):
    """Plan one bounded capture wave without changing logical sample order.

    Returned routes are indexed by logical position. Empty sources are valid;
    a wave that cannot fit is rejected rather than dropping or copying samples.
    """
    for name, value in (("source_count", source_count), ("max_samples", max_samples),
                        ("max_tokens", max_tokens), ("max_sequence_tokens", max_sequence_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if source_count > 4096 or source_count * max_samples > _MAX_RECORDS:
        raise ValueError("Source routing plan exceeds bounded planning capacity")
    sizes = []
    for length in lengths:
        if len(sizes) >= source_count * max_samples:
            raise ValueError("Capture wave exceeds source sample capacity")
        if type(length) is not int or not 0 < length <= min(max_sequence_tokens, max_tokens):
            raise ValueError("Sequence exceeds source token limits or has invalid length")
        sizes.append(length)
    loads, counts = [0] * source_count, [0] * source_count
    routes = [0] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
        eligible = [source for source in range(source_count)
                    if counts[source] < max_samples and loads[source] + sizes[index] <= max_tokens]
        if not eligible:
            raise ValueError("Capture wave cannot fit source token and sample limits")
        source = min(eligible, key=lambda source: (loads[source], counts[source], source))
        routes[index] = source
        loads[source] += sizes[index]
        counts[source] += 1
    plan = {"version": 1, "lengths": sizes, "routes": routes, "source_count": source_count,
            "max_samples": max_samples, "max_tokens": max_tokens,
            "max_sequence_tokens": max_sequence_tokens}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def iter_source_route_waves(requests, *, wave_samples, endpoints, max_samples, max_tokens,
                            max_sequence_tokens, consumed=()):
    """Route full logical waves before excluding committed occurrences."""
    if type(wave_samples) is not int or not 1 <= wave_samples <= _MAX_RECORDS:
        raise ValueError("Invalid routing wave sample limit")
    endpoints = tuple(endpoints)
    if not endpoints or len(endpoints) > 4096 or len(set(endpoints)) != len(endpoints):
        raise ValueError("Source routing requires unique ordered endpoints")
    plan_source_routes([], source_count=len(endpoints), max_samples=max_samples,
                       max_tokens=max_tokens, max_sequence_tokens=max_sequence_tokens)
    if wave_samples > len(endpoints) * max_samples:
        raise ValueError("Routing wave exceeds source sample capacity")
    consumed = set(consumed)
    def emit(wave):
        costs = [len(request.prompt_token_ids) + max(0, getattr(request, "max_tokens", 0)) for request in wave]
        plan = plan_source_routes(costs, source_count=len(endpoints), max_samples=max_samples,
                                  max_tokens=max_tokens, max_sequence_tokens=max_sequence_tokens)
        identity = {"plan": plan, "endpoints": endpoints,
                    "occurrences": [(request.sequence_no, request.sample_id) for request in wave]}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for request, route in zip(wave, plan["routes"]):
            if request.sequence_no not in consumed:
                yield replace(request, source_route=route, source_plan_sha256=digest)
    wave, wave_id, previous, wave_tokens = [], None, -1, 0
    for request in requests:
        sequence = request.sequence_no
        if type(sequence) is not int or sequence <= previous:
            raise ValueError("Source routing requires increasing occurrence sequence numbers")
        current = sequence // wave_samples
        if wave and current != wave_id:
            yield from emit(wave)
            wave = []
            wave_tokens = 0
        cost = len(request.prompt_token_ids) + max(0, getattr(request, "max_tokens", 0))
        if not 0 < cost <= min(max_sequence_tokens, max_tokens):
            raise ValueError("Sequence exceeds source token limits or has invalid length")
        if wave_tokens + cost > len(endpoints) * max_tokens:
            raise ValueError("Routing wave exceeds aggregate source token capacity")
        wave.append(request)
        wave_tokens += cost
        wave_id, previous = current, sequence
    if wave:
        yield from emit(wave)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_row(row: Mapping[str, Any]) -> None:
    if (
        not isinstance(row, Mapping)
        or not {
            "sequence_no",
            "sample_id",
            "input_ids",
            "loss_mask",
            "source_metadata",
        }.issubset(row)
        or set(row)
        - {
            "sequence_no",
            "sample_id",
            "input_ids",
            "loss_mask",
            "source_metadata",
            "prefill_excludes_last",
        }
        or type(row["sequence_no"]) is not int
        or row["sequence_no"] < 0
        or not isinstance(row["sample_id"], str)
        or not row["sample_id"]
        or not isinstance(row["source_metadata"], dict)
        or (
            "prefill_excludes_last" in row
            and type(row["prefill_excludes_last"]) is not bool
        )
    ):
        raise ValueError("Invalid sequence replay identity or metadata")
    ids, mask = row["input_ids"], row["loss_mask"]
    if (
        not isinstance(ids, list)
        or len(ids) < 2
        or any(type(token) is not int or token < 0 for token in ids)
        or not isinstance(mask, list)
        or len(mask) != len(ids)
        or any(type(value) not in (int, float) or value not in (0, 1) for value in mask)
    ):
        raise ValueError("Invalid sequence replay tokens or loss mask")
    try:
        boundary = mask.index(1)
    except ValueError:
        raise ValueError("Sequence replay requires a supervised response") from None
    if boundary == 0 or any(value != 1 for value in mask[boundary:]):
        raise ValueError("Sequence replay loss mask must identify a response suffix")


def _lock_file(path: Path):
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except BaseException:
        stream.close()
        raise


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _ReplayIndex(Mapping):
    """Rebuildable disk-backed identity index with a bounded SQLite page cache."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="speco-replay-index-")
        self._db = None
        self._count = 0
        try:
            self._db = sqlite3.connect(str(Path(self._directory.name) / "index.sqlite"), check_same_thread=False)
            self._db.execute("PRAGMA cache_size=-4096")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA journal_mode=OFF")
            self._db.execute("PRAGMA synchronous=OFF")
            self._db.execute("CREATE TABLE rows (seq TEXT PRIMARY KEY, sample TEXT NOT NULL, digest BLOB NOT NULL) WITHOUT ROWID")
        except BaseException:
            self.close()
            raise

    def add(self, sequence_no: int, sample_id: str, digest: bytes) -> bool:
        try:
            self._db.execute("INSERT INTO rows VALUES (?, ?, ?)", (str(sequence_no), sample_id, digest))
        except sqlite3.IntegrityError:
            return False
        self._count += 1
        return True

    def get(self, sequence_no: int, default=None):
        row = self._db.execute("SELECT sample, digest FROM rows WHERE seq = ?", (str(sequence_no),)).fetchone()
        return default if row is None else (row[0], bytes(row[1]).hex())

    def __len__(self):
        return self._count

    def __getitem__(self, sequence_no):
        value = self.get(sequence_no)
        if value is None:
            raise KeyError(sequence_no)
        return value

    def __iter__(self):
        return (int(row[0]) for row in self._db.execute("SELECT seq FROM rows"))

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        self._directory.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SequenceReplayReader:
    """Verify a manifest and index committed rows without loading token arrays."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any] | None = None,
        allow_incomplete: bool = False,
    ):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / _MANIFEST
        if (
            manifest_path.is_symlink()
            or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            raise ValueError("Invalid sequence replay manifest")
        manifest_bytes = manifest_path.read_bytes()
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("Invalid sequence replay manifest")
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes)
        if (
            not isinstance(self.manifest, dict)
            or self.manifest.get("schema_version") != 1
            or not isinstance(self.manifest.get("identity"), dict)
            or not isinstance(self.manifest.get("segments"), list)
            or type(self.manifest.get("complete")) is not bool
        ):
            raise ValueError("Invalid sequence replay manifest schema")
        if not allow_incomplete and not self.manifest["complete"]:
            raise ValueError("Sequence replay generation has not completed")
        if identity is not None and dict(identity) != self.manifest["identity"]:
            raise ValueError("Sequence replay identity differs from the requested run")
        self.files: list[Path] = []
        self.file_ids = array("I")
        self.offsets = array("Q")
        self.row_hashes = bytearray()
        self.seen = _ReplayIndex()
        try:
            seen_files = set()
            for segment in self.manifest["segments"]:
                if not isinstance(segment, dict):
                    raise ValueError("Invalid sequence replay segment descriptor")  # noqa: TRY004
                name = segment.get("file")
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.startswith("segment-")
                    or not name.endswith(".jsonl")
                    or name in seen_files
                ):
                    raise ValueError("Invalid or duplicate sequence replay segment")
                path = self.directory / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"Sequence replay segment missing or checksum mismatch: {name}"
                    )
                seen_files.add(name)
                index = len(self.files)
                self.files.append(path)
                count = 0
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while True:
                        offset = stream.tell()
                        line = stream.readline(_MAX_ROW_BYTES + 1)
                        if not line:
                            break
                        if len(line) > _MAX_ROW_BYTES or not line.endswith(b"\n"):
                            raise ValueError(f"Invalid sequence replay row in {name}")
                        digest.update(line)
                        row = json.loads(line)
                        _validate_row(row)
                        sequence_no = row["sequence_no"]
                        row_digest = hashlib.sha256(_encoded(row)).digest()
                        if not self.seen.add(sequence_no, row["sample_id"], row_digest):
                            raise ValueError("Duplicate sequence number in replay manifest")
                        self.row_hashes.extend(row_digest)
                        self.file_ids.append(index)
                        self.offsets.append(offset)
                        count += 1
                        if len(self.offsets) > _MAX_RECORDS:
                            raise ValueError("Sequence replay exceeds record limit")
                if (
                    type(segment.get("rows")) is not int
                    or count != segment["rows"]
                    or count == 0
                ):
                    raise ValueError(f"Sequence replay segment row count mismatch: {name}")
                if digest.hexdigest() != segment.get("sha256"):
                    raise ValueError(
                        f"Sequence replay segment missing or checksum mismatch: {name}"
                    )
        except BaseException:
            self.seen.close()
            raise
    def close(self) -> None:
        """Release the temporary identity index; row iteration remains available."""
        self.seen.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def iter_rows(self, order: Sequence[int] | None = None) -> Iterator[dict[str, Any]]:
        current_id, stream = None, None
        try:
            for position in range(len(self)) if order is None else order:
                try:
                    position = operator.index(position)
                except TypeError:
                    raise ValueError(
                        "Sequence replay order must contain integer indices"
                    ) from None
                if not 0 <= position < len(self):
                    raise ValueError("Sequence replay order contains an invalid index")
                index = self.file_ids[position]
                if index != current_id:
                    if stream is not None:
                        stream.close()
                    stream = self.files[index].open("rb")
                    current_id = index
                stream.seek(self.offsets[position])
                row = json.loads(stream.readline(_MAX_ROW_BYTES + 1))
                _validate_row(row)
                start = position * 32
                if hashlib.sha256(_encoded(row)).digest() != self.row_hashes[start:start + 32]:
                    raise ValueError("Sequence replay row changed after indexing")
                yield row
        finally:
            if stream is not None:
                stream.close()


class SequenceReplayWriter:
    """Publish immutable segments and recover only rows committed in the manifest."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any],
        segment_rows: int = 100,
        segment_bytes: int = 64 * 1024 * 1024,
    ):
        if (
            type(segment_rows) is not int
            or segment_rows < 1
            or not identity
            or type(segment_bytes) is not int
            or segment_bytes < _MAX_ROW_BYTES
        ):
            raise ValueError(
                "Sequence replay requires an identity and positive segment size"
            )
        _encoded(identity)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = _lock_file(self.directory / ".writer.lock")
        try:
            manifest_path = self.directory / _MANIFEST
            if manifest_path.exists():
                reader = SequenceReplayReader(
                    self.directory, identity=identity, allow_incomplete=True
                )
                self.manifest = reader.manifest
                self.seen = reader.seen
            else:
                self.manifest = {
                    "schema_version": 1,
                    "identity": dict(identity),
                    "segments": [],
                    "complete": False,
                }
                self.seen = _ReplayIndex()
            self.segment_rows = segment_rows
            self.segment_bytes = segment_bytes
            self.pending: list[tuple[dict[str, Any], bytes]] = []
            self.pending_bytes = 0
            self._mutex = threading.Lock()
            self._closed = False
            self._broken = False
        except BaseException:
            try:
                if hasattr(self, "seen"):
                    self.seen.close()
            finally:
                self.lock.close()
            raise

    def has_record(self, sequence_no: int, sample_id: str) -> bool:
        with self._mutex:
            if self._closed or self._broken:
                raise RuntimeError("Sequence replay writer is closed or failed")
            existing = self.seen.get(sequence_no)
            if existing and existing[0] != sample_id:
                raise ValueError("Sequence replay sample identity changed")
            return existing is not None

    def append(self, row: Mapping[str, Any]) -> bool:
        _validate_row(row)
        data = _encoded(row)
        if len(data) + 1 > _MAX_ROW_BYTES:
            raise ValueError("Sequence replay row exceeds byte limit")
        digest = hashlib.sha256(data).hexdigest()
        with self._mutex:
            if self._closed or self._broken:
                raise RuntimeError("Sequence replay writer is closed or failed")
            seq = row["sequence_no"]
            existing = self.seen.get(seq)
            pending = next(
                (item for item, _ in self.pending if item["sequence_no"] == seq), None
            )
            if pending is not None:
                existing = (
                    pending["sample_id"],
                    hashlib.sha256(_encoded(pending)).hexdigest(),
                )
            if existing is not None:
                if existing != (row["sample_id"], digest):
                    raise ValueError(
                        "Sequence replay sequence number has conflicting content"
                    )
                return False
            if self.manifest["complete"]:
                raise ValueError("Cannot append to a completed sequence replay")
            if len(self.seen) + len(self.pending) >= _MAX_RECORDS:
                raise ValueError("Sequence replay exceeds record limit")
            if self.pending and self.pending_bytes + len(data) + 1 > self.segment_bytes:
                self._flush_locked()
            self.pending.append((json.loads(data), data + b"\n"))
            self.pending_bytes += len(data) + 1
            if len(self.pending) >= self.segment_rows:
                self._flush_locked()
            return True

    def _flush_locked(self) -> None:
        if not self.pending:
            return
        payload = b"".join(data for _, data in self.pending)
        name = f"segment-{uuid.uuid4().hex}.jsonl"
        descriptor = {
            "file": name,
            "rows": len(self.pending),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        manifest = {
            **self.manifest,
            "segments": [*self.manifest["segments"], descriptor],
        }
        encoded_manifest = _encoded(manifest) + b"\n"
        if len(encoded_manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("Sequence replay manifest exceeds byte limit")
        try:
            _atomic_write(self.directory / name, payload)
            _atomic_write(self.directory / _MANIFEST, encoded_manifest)
            for row, data in self.pending:
                if not self.seen.add(row["sequence_no"], row["sample_id"], hashlib.sha256(data[:-1]).digest()):
                    raise RuntimeError("Sequence replay index conflicts after publication")
        except BaseException:
            self._broken = True
            raise
        self.manifest = manifest
        self.pending.clear()
        self.pending_bytes = 0

    def close(self, *, commit: bool = True) -> None:
        with self._mutex:
            if self._closed:
                return
            try:
                if commit:
                    if self._broken:
                        raise RuntimeError("Cannot commit a failed sequence replay writer")
                    self._flush_locked()
                    if not self.seen:
                        raise ValueError("Cannot complete an empty sequence replay")
                    if not self.manifest["complete"]:
                        complete = {**self.manifest, "complete": True}
                        _atomic_write(
                            self.directory / _MANIFEST, _encoded(complete) + b"\n"
                        )
                        self.manifest = complete
            finally:
                self._closed = True
                try:
                    self.seen.close()
                finally:
                    self.lock.close()


async def generate_sequences(input_path, *, writer, tokenizer, generate, config) -> int:
    """Generate or tokenize one input pass; committed sequence IDs are skipped."""
    from .input_reader import (
        iter_input_records,
        prepare_generation_request,
        tokenize_record,
    )

    generated_count = 0
    input_count = 0
    for record in iter_input_records(input_path, on_error="raise"):
        input_count += 1
        if writer.has_record(record.sequence_no, record.sample_id):
            continue
        if record.response is None:
            request = prepare_generation_request(record, tokenizer, config)
            response = await generate(request)
            if (
                not isinstance(response, (list, tuple))
                or not response
                or len(response) > request.max_tokens
                or any(type(token) is not int or token < 0 for token in response)
            ):
                raise ValueError("Generation endpoint returned invalid token IDs")
            ids = [*request.prompt_token_ids, *response]
            mask = [0] * len(request.prompt_token_ids) + [1] * len(response)
        else:
            tokenized = tokenize_record(record, tokenizer, config)
            ids = tokenized.input_ids.tolist()
            mask = tokenized.loss_mask.tolist()
        writer.append(
            {
                "sequence_no": record.sequence_no,
                "sample_id": record.sample_id,
                "input_ids": ids,
                "loss_mask": mask,
                "source_metadata": record.source_metadata,
                "prefill_excludes_last": record.response is None,
            }
        )
        generated_count += 1
    if not input_count:
        raise ValueError("Sequence replay source contains no samples")
    return generated_count


def prepare_replay_request(
    row: Mapping[str, Any], *, sequence_no: int, config: Mapping[str, Any]
):
    """Recreate training's token request with the source's original prefill boundary."""
    from .input_reader import _build_tokenized_request

    _validate_row(row)
    ids, mask = row["input_ids"], row["loss_mask"]
    generated = row.get("prefill_excludes_last", True)
    return _build_tokenized_request(
        sequence_no=sequence_no,
        sample_id=row["sample_id"],
        prompt_length=mask.index(1),
        full_ids=ids,
        source_metadata=dict(row["source_metadata"]),
        config=config,
        vllm_prompt_token_ids=ids[:-1] if generated else None,
        feature_end_limit=len(ids) - 1 if generated else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Collect a committed replay corpus from a standard vLLM generation endpoint."""
    import argparse
    import asyncio
    import math

    parser = argparse.ArgumentParser(
        description="Generate a standalone token replay corpus"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-fingerprint")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--segment-rows", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.temperature)
        or args.temperature < 0
        or not math.isfinite(args.top_p)
        or not 0 < args.top_p <= 1
        or args.max_tokens < 1
        or args.max_sequence_length < 2
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("sampling, length and timeout arguments are invalid")
    from urllib.parse import urlparse

    endpoint = urlparse(args.endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        parser.error("endpoint must be an HTTP(S) vLLM API base URL")
    input_path = Path(args.input).resolve()
    target_revision = (
        args.model_revision
        or "target-path-sha256-" + hashlib.sha256(args.model.encode()).hexdigest()
    )
    tokenizer_fingerprint = (
        args.tokenizer_fingerprint
        or "tokenizer-path-sha256-"
        + hashlib.sha256(args.tokenizer.encode()).hexdigest()
    )
    identity = {
        "input_sha256": _sha256(input_path),
        "model": args.model,
        "model_revision": target_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_sequence_length": args.max_sequence_length,
    }
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    writer = SequenceReplayWriter(
        args.output, identity=identity, segment_rows=args.segment_rows
    )

    async def execute():
        async with httpx.AsyncClient(
            base_url=args.endpoint.rstrip("/") + "/"
        ) as client:

            async def generate(request):
                response = await client.post(
                    "completions",
                    json={
                        "model": args.model,
                        "prompt": list(request.prompt_token_ids),
                        "max_tokens": request.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "return_token_ids": True,
                    },
                    timeout=args.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("kv_transfer_params"):
                    raise ValueError(
                        "Sequence-only generation requires an endpoint without hidden-state transfer"
                    )
                choices = payload.get("choices") or []
                if not choices or not isinstance(choices[0].get("token_ids"), list):
                    raise ValueError("Generation endpoint must return token_ids")
                actual_prompt = choices[0].get("prompt_token_ids")
                if actual_prompt is not None and actual_prompt != list(
                    request.prompt_token_ids
                ):
                    raise ValueError(
                        "Generation endpoint returned different prompt token IDs"
                    )
                return choices[0]["token_ids"]

            return await generate_sequences(
                input_path,
                writer=writer,
                tokenizer=tokenizer,
                generate=generate,
                config={
                    "generation_max_tokens": args.max_tokens,
                    "max_sequence_length": args.max_sequence_length,
                    "max_feature_length": 0,
                    "preserve_prompt_context": True,
                },
            )

    try:
        count = asyncio.run(execute())
        if _sha256(input_path) != identity["input_sha256"]:
            raise ValueError("Input changed during sequence replay generation")
        writer.close()
    except BaseException:
        writer.close(commit=False)
        raise
    print(f"Committed {count} new sequences to {writer.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
