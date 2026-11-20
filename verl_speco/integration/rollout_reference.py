# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in fixed-window rollout references; CPU preparation never calls a model."""

import os
import threading
import logging
import time
import hashlib
import json
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_BUDGET_LOCK = threading.Lock()
_OWNER = uuid.uuid4().hex


class ReferencePrefetcher:
    """Bounded CPU-only preparation; scheduler retains collection ownership."""

    def __init__(self, *, capacity=None, workers=2, resolver=None, max_bytes=None):
        if capacity is None:
            capacity = int(os.getenv("SPECO_REFERENCE_PREFETCH_CAPACITY", "4"))
        if max_bytes is None:
            max_bytes = int(
                os.getenv("SPECO_REFERENCE_PREFETCH_MAX_BYTES", str(256 << 20))
            )
        if capacity <= 0 or workers <= 0 or max_bytes <= 0:
            raise ValueError(
                "Prefetch capacity, workers and byte limit must be positive"
            )
        self.capacity = capacity
        self.max_bytes = max_bytes
        self._bytes = 0
        self._resolve = resolver or resolve
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rollout-consume"
        )
        self._jobs = {}
        self._waiting = {}
        self.sync_resolves = 0
        self.sync_resolve_s = 0.0
        self.future_wait_s = 0.0
        self._closed = False

    @staticmethod
    def identity(reference):
        return json.dumps(reference, sort_keys=True, separators=(",", ":"))

    def submit(self, reference, *, defer=False):
        if self._closed:
            raise RuntimeError("Reference prefetcher is closed")
        key = self.identity(reference)
        if key in self._jobs:
            return True
        size = 2 * int(reference.get("budget", {}).get("bytes", 0))
        if (
            size < 0
            or len(self._jobs) >= self.capacity
            or self._bytes + size > self.max_bytes
        ):
            if defer:
                self._waiting.setdefault(key, None)
            return False
        self._waiting.pop(key, None)
        self._jobs[key] = (self._executor.submit(self._resolve, json.loads(key)), size)
        self._bytes += size
        return True

    def _refill(self):
        for key in tuple(self._waiting):
            if len(self._jobs) >= self.capacity:
                break
            self.submit(json.loads(key))

    def take(self, reference):
        key = self.identity(reference)
        self._waiting.pop(key, None)
        job = self._jobs.pop(key, None)
        if job is None:
            started = time.perf_counter()
            self.sync_resolves += 1
            try:
                return self._resolve(reference)
            finally:
                self.sync_resolve_s += time.perf_counter() - started
        future, size = job
        started = time.perf_counter()
        try:
            return future.result()
        finally:
            self.future_wait_s += time.perf_counter() - started
            self._bytes -= size
            self._refill()

    def boundary_metrics(self):
        ready = len(self.ready())
        return {
            "drafter/reference_ready_at_boundary": ready,
            "drafter/reference_inflight_at_boundary": len(self._jobs) - ready,
            "drafter/reference_waiting_at_boundary": len(self._waiting),
            "drafter/reference_prefetch_reserved_bytes": self._bytes,
        }

    def ready(self):
        return tuple(key for key, (future, _) in self._jobs.items() if future.done())

    def drain_ready(self, references):
        """Transfer only completed work; preserve original batch selection/order."""
        eligible = {self.identity(ref) for ref in references}
        return {
            key: self.take(json.loads(key)) for key in self.ready() if key in eligible
        }

    def close(self):
        self._closed = True
        self._waiting.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        jobs, self._jobs = self._jobs, {}
        self._bytes = 0
        failures = []
        for key, (future, _) in jobs.items():
            if not future.cancelled():
                try:
                    future.result()
                except Exception as exc:
                    sample = json.loads(key).get("plan", {}).get("sample_id")
                    failures.append((sample, exc))
        if failures:
            logger.error(
                "Reference prefetch cleanup failures: %s",
                [(sample, str(exc)) for sample, exc in failures],
            )
            raise failures[0][1]


@contextmanager
def _ledger_lock(root):
    """OS-owned lock survives worker death without a stale lock-directory lease."""
    root.mkdir(parents=True, exist_ok=True)
    with _BUDGET_LOCK, (root / ".ledger.lock").open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reserve(plan, *, window_bytes=None):
    from verl_speco.integration.vllm_rollout_capture import capture_root

    limit = int(os.getenv("SPECO_REFERENCE_MAX_BYTES", str(256 << 20)))
    allowance = int(os.getenv("SPECO_REFERENCE_SAMPLE_BYTES", str(64 << 20)))
    if allowance <= 0 or limit <= 0 or allowance > limit:
        raise ValueError("Invalid reference byte budget")
    if window_bytes is not None:
        if window_bytes <= 0 or window_bytes > allowance:
            return None
        allowance = window_bytes
    root = capture_root() / ".reference-budget"
    ledger = root / _OWNER
    with _ledger_lock(root):
        ledger.mkdir(parents=True, exist_ok=True)
        used = sum(int(p.read_text()) for p in root.glob("*/*.reservation"))
        if used + allowance > limit:
            return None
        ticket = ledger / f"{uuid.uuid4().hex}.reservation"
        ticket.write_text(str(allowance))
    return {"owner": _OWNER, "ticket": ticket.name, "bytes": allowance}


def _release(reference):
    from verl_speco.integration.vllm_rollout_capture import capture_root

    budget = reference.get("budget")
    if budget is None:
        return
    owner, ticket = budget.get("owner"), budget.get("ticket")
    if (
        not isinstance(owner, str)
        or len(owner) != 32
        or any(c not in "0123456789abcdef" for c in owner)
        or not isinstance(ticket, str)
        or not ticket.endswith(".reservation")
        or len(ticket.removesuffix(".reservation")) != 32
        or any(c not in "0123456789abcdef" for c in ticket.removesuffix(".reservation"))
    ):
        raise RuntimeError("Invalid reference budget identity")
    root = capture_root() / ".reference-budget"
    with _ledger_lock(root):
        (root / owner / ticket).unlink(missing_ok=True)


def submit(plan, prompt, response, request_id, target_version):
    """Transfer a fixed-window handle; inline fallback never assembles a second file."""
    if not plan.get("window_store"):
        return None
    from verl_speco.integration.vllm_rollout_capture import plan_directory
    from verl_speco.transport.drafter_sample_protocol import validate_rollout_reference

    metadata = plan_directory(plan) / "window.meta"
    budget = plan.get("reference_budget") or (
        json.loads(metadata.read_text())["budget"]
        if metadata.exists()
        else _reserve(plan)
    )
    if budget is None:
        logger.warning(
            "[rollout reference] event=fallback reason=byte_capacity sample=%s",
            plan.get("sample_id"),
        )
        return None
    # The caller keeps the producer reservation if handle publication fails;
    # cancellation releases it only once the final export writer has drained.
    plan["reference_budget"] = budget
    reference = {
        "budget": budget,
        "prompt": list(prompt),
        "response": list(response),
        "kind": "speco_rollout_reference_v1",
        "plan": dict(plan),
        "request_id": request_id,
        "target_version": target_version,
        "global_step": plan["global_step"],
    }
    validate_rollout_reference(reference)
    directory = plan_directory(plan)
    temporary = directory / f"reference-{uuid.uuid4().hex}.tmp"
    with temporary.open("w") as stream:
        json.dump(reference, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "reference.json")
    logger.warning(
        "[rollout reference] event=submit sample=%s target=%s",
        plan["sample_id"],
        target_version,
    )
    return reference


def _ready_tag(reference, prompt, response, directory):
    from verl_speco.transport.drafter_sample_protocol import (
        SampleMetadata,
        PROTOCOL_SCHEMA_VERSION,
        make_reference_tag,
        validate_reference_tag,
    )

    plan = reference["plan"]
    sequence = prompt + response
    positions = list(range(plan["start"], plan["end"]))
    tag = make_reference_tag(
        SampleMetadata(
            PROTOCOL_SCHEMA_VERSION,
            plan.get("run_id")
            or hashlib.sha256(str(directory.parent).encode()).hexdigest(),
            plan["sample_id"],
            0,
        ),
        path=str(directory / "window.safetensors"),
        request={
            "input_ids": sequence,
            "loss_mask": [0] * len(prompt) + [1] * len(response),
            "position_ids": list(range(len(sequence))),
            "feature_positions": positions,
            "draft_position_ids": positions,
            "vllm_prompt_token_ids": prompt,
            "source_metadata": {
                "target_version": reference["target_version"],
                "request_id": reference["request_id"],
            },
        },
        feature_contract={
            "adapter": "rollout_window",
            "layout": plan.get("layout", "aux"),
            "target_version": reference["target_version"],
        },
    )
    validate_reference_tag(tag)
    return tag


def _merge_evidence(result, reference):
    for key, value in reference.get("evidence", {}).items():
        if (
            not isinstance(key, str)
            or not key.startswith(("_speco_", "_verl_"))
            or key
            in {
                "_speco_sample_id",
                "_speco_global_sample_id",
                "_speco_direct_event_emitted",
            }
        ):
            raise RuntimeError(
                "Rollout evidence may not replace feature identity or contract"
            )
        result[key] = value
    return result


def _check_export_error(directory, plan):
    try:
        detail = (directory / "error").read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    raise RuntimeError(
        f"Window export failed step={plan.get('global_step')} sample={plan['sample_id']}"
        + (f": {detail}" if detail else "")
    )


def resolve(reference, timeout=120):
    import torch
    from verl_speco.integration.vllm_rollout_capture import plan_directory
    from verl_speco.transport.drafter_sample_protocol import validate_rollout_reference
    from verl_speco.producer.rollout_window_store import load_window
    from verl_speco.producer.feature_export import (
        rollout_feature_fields,
        hidden_states_are_finite,
    )

    if reference.get("kind") != "speco_rollout_reference_v1":
        return reference
    if not reference.get("plan", {}).get("window_store"):
        raise ValueError(
            "Legacy assembled references are unsupported; use inline capture or fixed windows"
        )
    validate_rollout_reference(reference)
    if _receipt_path(reference).exists():
        raise RuntimeError(
            "Rollout reference transaction already exists; refusing duplicate or ambiguous replay"
        )
    plan = reference["plan"]
    directory = plan_directory(plan)
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    _check_export_error(directory, plan)
    if len(reference["prompt"]) + len(reference["response"]) - 1 < plan["end"]:
        logger.warning(
            "[rollout reference] event=skip reason=short_response sample=%s",
            plan["sample_id"],
        )
        return None
    while True:
        _check_export_error(directory, plan)
        if (directory / "skipped").exists():
            return None
        if (directory / "window.ready").exists() and not any(
            directory.glob("*.pending")
        ):
            break
        if time.monotonic() >= deadline:
            logger.error(
                "[rollout reference] event=timeout sample=%s target=%s directory=%s",
                plan["sample_id"],
                reference["target_version"],
                directory,
            )
            raise TimeoutError("Rollout reference not ready or export root not shared")
        time.sleep(0.005)
    ready_wait = time.perf_counter() - started
    manifest_path = directory / "manifest.ready"
    recovered = not manifest_path.exists()
    if recovered:
        manifest = {
            "direct_window": True,
            "prompt": reference["prompt"],
            "response": reference["response"],
        }
    else:
        while True:
            try:
                manifest = json.loads(manifest_path.read_text())
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)
    if manifest.get("direct_window") is not True:
        raise RuntimeError("Invalid rollout window manifest schema")
    # Load and clone once. Recovery publishes only after all data and identity
    # checks pass, without a second complete file read or assembled copy.
    window, _ = load_window(directory)
    metadata = window["metadata"]
    if "budget" in reference and metadata.get("budget") != reference["budget"]:
        raise RuntimeError("Window reference reservation mismatch")
    if (
        metadata["sample_id"] != plan["sample_id"]
        or metadata.get("run_id") != plan.get("run_id")
        or metadata.get("target_version") != reference["target_version"]
        or metadata.get("request_id") != reference["request_id"]
        or ("layout" in plan and metadata["layout"] != plan["layout"])
    ):
        raise RuntimeError("Window sample mismatch")
    prompt, response = manifest["prompt"], manifest["response"]
    if prompt != reference["prompt"] or response != reference["response"]:
        raise RuntimeError("Window manifest token identity mismatch")
    if (metadata["start"], metadata["end"]) != (plan["start"], plan["end"]):
        raise RuntimeError("Window position range mismatch")
    if not hidden_states_are_finite(window["hidden_states"]):
        raise RuntimeError("Window contains non-finite hidden states")
    _ready_tag(reference, prompt, response, directory)
    sequence = prompt + response
    positions = window["positions"].tolist()
    if any(
        p >= len(sequence) - 1 or int(window["token_ids"][i]) != sequence[p]
        for i, p in enumerate(positions)
    ):
        raise RuntimeError("Window token alignment mismatch")
    result = {
        **rollout_feature_fields(
            torch.tensor(prompt),
            torch.tensor(response),
            window["hidden_states"],
            window["positions"],
        ),
        "hidden_states_layout": metadata["layout"],
        "hidden_layer_ids": metadata["layer_ids"],
        "last_hidden_state_norm": metadata["norm"],
        "hidden_position_start": metadata["start"],
        "hidden_position_end": metadata["end"],
        "hidden_source": "vllm_rollout",
        "_speco_sample_id": plan["sample_id"],
        "global_step": plan["global_step"],
        "replica_rank": plan["replica_rank"],
        "request_id": reference["request_id"],
        "target_version": reference["target_version"],
    }
    _merge_evidence(result, reference)
    if recovered:
        temporary = directory / f"repair-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(manifest))
        os.replace(temporary, manifest_path)
        logger.warning(
            "[rollout reference] event=recovered sample=%s target=%s",
            plan["sample_id"],
            reference["target_version"],
        )
    logger.warning(
        "[rollout reference] event=resolved backend=window sample=%s target=%s resolve_s=%.6f writer_ready_wait_s=%.6f",
        plan["sample_id"],
        reference["target_version"],
        time.perf_counter() - started,
        ready_wait,
    )
    return result


def _receipt_identity(reference):
    return hashlib.sha256(
        json.dumps(
            {
                "plan": reference["plan"],
                "request_id": reference.get("request_id"),
                "target_version": reference.get("target_version"),
                "prompt": reference.get("prompt"),
                "response": reference.get("response"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _receipt_path(reference):
    from verl_speco.integration.vllm_rollout_capture import capture_root

    sample = str(reference["plan"]["sample_id"])
    key = hashlib.sha256(sample.encode()).hexdigest()
    return capture_root() / ".reference-transactions" / f"{key}.json"


def _write_receipt(reference, state):
    path = _receipt_path(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(
            {"identity": _receipt_identity(reference), "owner": _OWNER, "state": state},
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def begin_collection(references):
    """Persist intent before any worker mutation; ambiguous replay fails closed.

    Worker buffers and this journal are not an atomic distributed store. An
    interrupted COMMITTING record therefore requires a consistent checkpoint,
    rather than guessing whether a worker already consumed the sample.
    """
    references = list(references)
    if not references:
        return
    with _ledger_lock(_receipt_path(references[0]).parent):
        for reference in references:
            if _receipt_path(reference).exists():
                raise RuntimeError(
                    "Rollout reference transaction already exists; restore a consistent checkpoint"
                )
        for reference in references:
            _write_receipt(reference, "COMMITTING")


def finish_collection(references, *, collected):
    references = list(references)
    if not references:
        return
    with _ledger_lock(_receipt_path(references[0]).parent):
        for reference in references:
            record = json.loads(_receipt_path(reference).read_text())
            if (
                record.get("identity") != _receipt_identity(reference)
                or record.get("owner") != _OWNER
                or record.get("state") != "COMMITTING"
            ):
                raise RuntimeError("Rollout reference transaction ownership mismatch")
        for reference in references:
            _write_receipt(reference, "COMMITTED" if collected else "SKIPPED")


def require_collection_commit(outcome):
    """A non-throwing failed transaction is not permission to ACK sources."""
    if outcome is not None and outcome.attempted and not outcome.collected:
        raise RuntimeError(
            f"Reference collection failed without commit: {outcome.reason}; source retained"
        )


def committed_reference_metrics(outcome, dispatch_rows, references):
    """The collection strategy validates every owner's accepted count before success."""
    require_collection_commit(outcome)
    if len(dispatch_rows) != getattr(outcome, "collected_samples", 0):
        raise RuntimeError("Partial rollout dispatch; retaining reference sources")
    reference_ids = {ref["plan"]["sample_id"] for ref in references}
    committed = reference_ids.intersection(dispatch_rows)
    return committed, {
        "drafter/rollout_reference_committed": len(committed),
        "drafter/rollout_inline_committed": len(dispatch_rows) - len(committed),
        "drafter/rollout_capacity_fallback_committed": sum(
            fallback for _, fallback in dispatch_rows.values()
        ),
        "drafter/rollout_reference_skipped": len(reference_ids) - len(committed),
        "drafter/rollout_committed_hidden_rows": sum(
            rows for rows, _ in dispatch_rows.values()
        ),
    }


def acknowledge_batch(references):
    """Preserve individual durable ACKs and stop on the first failed source."""
    started = time.perf_counter()
    finished = 0
    try:
        for reference in references:
            acknowledge(reference)
            finished += 1
    finally:
        logger.warning(
            "[rollout reference] event=ack_batch step=%s completed=%s total=%s seconds=%.6f",
            references[0]["global_step"],
            finished,
            len(references),
            time.perf_counter() - started,
        )


def acknowledge(reference):
    """Release source only after collection has succeeded (or explicitly skipped)."""
    from verl_speco.integration.vllm_rollout_capture import cleanup

    from verl_speco.integration.vllm_rollout_capture import plan_directory, capture_root

    # READY does not guarantee every export worker has completed its finally
    # block. Never remove source mappings while a pending writer still owns them.
    directory = plan_directory(reference["plan"])
    deadline = time.monotonic() + 120
    while any(directory.glob("*.pending")):
        if time.monotonic() >= deadline:
            raise TimeoutError("ACK refused while export writers remain pending")
        time.sleep(0.005)
    if (directory / "error").exists():
        raise RuntimeError("ACK refused: window export failed; source retained")
    transaction = _receipt_path(reference)
    if not transaction.exists():
        raise RuntimeError("ACK refused: missing collection receipt; source retained")
    record = json.loads(transaction.read_text())
    if record.get("identity") != _receipt_identity(reference) or record.get(
        "state"
    ) not in {"COMMITTED", "SKIPPED", "RELEASED"}:
        raise RuntimeError(
            "ACK refused for uncommitted or mismatched reference transaction"
        )
    journal = capture_root() / ".reference-acks"
    journal.mkdir(parents=True, exist_ok=True)
    temporary = journal / f"{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "sample_id": reference["plan"]["sample_id"],
                "request_id": reference.get("request_id"),
                "target_version": reference["target_version"],
                "state": "acknowledged",
            }
        )
    )
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, journal / f"{reference['plan']['sample_id']}.json")
    cleanup(reference["plan"])
    if directory.exists():
        raise RuntimeError(
            "Reference source cleanup incomplete; inspect files before retry"
        )
    _release(reference)
    _write_receipt(reference, "RELEASED")
    logger.warning(
        "[rollout reference] event=ack sample=%s target=%s",
        reference["plan"]["sample_id"],
        reference["target_version"],
    )


def discard(reference, *, preparation_stopped=False):
    """Operator recovery only: never race a live writer or consumer.

    Call after the complete associated training run has stopped. No TTL-based
    deletion of active files is performed. Repeated recovery is idempotent.
    """
    if not preparation_stopped:
        raise RuntimeError("Stop the run before reclaiming orphan references")
    from verl_speco.integration.vllm_rollout_capture import cleanup, plan_directory

    directory = plan_directory(reference["plan"])
    transaction = _receipt_path(reference)
    cancelled = not transaction.exists() or (directory / "error").exists()
    if transaction.exists():
        record = json.loads(transaction.read_text())
        if record.get("identity") != _receipt_identity(reference):
            raise RuntimeError("Stopped recovery reference identity mismatch")
        cancelled = cancelled or record.get("state") in {"COMMITTING", "CANCELLED"}
    # This stopped-run operation cancels ownership, never fabricates a worker
    # commit. Stale markers may now be removed because no writer remains live.
    for marker in directory.glob("*.pending"):
        marker.unlink()
    if cancelled:
        _write_receipt(reference, "CANCELLED")
        cleanup(reference["plan"])
        if directory.exists():
            raise RuntimeError("Stopped reference cleanup incomplete")
        _release(reference)
        return
    acknowledge(reference)


def recover_stopped_root(root, *, preparation_stopped=False):
    """Reconcile a dedicated export root after all users of the root have stopped."""
    from verl_speco.integration.vllm_rollout_capture import capture_root, cleanup
    from verl_speco.transport.drafter_sample_protocol import validate_rollout_reference

    root = Path(root).resolve(strict=True)
    if not preparation_stopped or root != capture_root().resolve():
        raise RuntimeError(
            "Stopped recovery requires the configured stopped export root"
        )

    # Validate journal and ledger paths before cleanup can follow one of them.
    for name in (".reference-budget", ".reference-transactions", ".reference-acks"):
        journal = root / name
        for entry in (journal, *journal.rglob("*")):
            if entry.is_symlink() or entry.resolve() != entry:
                raise RuntimeError(f"Linked recovery path: {entry}")

    unresolved = []
    recovered = 0
    unpublished = 0
    for directory in sorted(root.iterdir()):
        if directory.resolve() != directory or directory.is_symlink():
            raise RuntimeError(f"Linked recovery path: {directory}")
        if len(directory.name) != 32 or any(
            c not in "0123456789abcdef" for c in directory.name
        ):
            if directory.is_dir() and (
                (directory / "reference.json").exists()
                or (directory / "window.meta").exists()
            ):
                unresolved.append(directory.name)
            continue
        if not directory.is_dir():
            continue
        if any(p.resolve() != p or p.is_symlink() for p in directory.iterdir()):
            raise RuntimeError(f"Linked sample contents: {directory}")
        manifest = directory / "reference.json"
        meta = directory / "window.meta"
        if manifest.exists():
            handle = json.loads(manifest.read_text())
            validate_rollout_reference(handle)
            if handle["plan"]["sample_id"] != directory.name:
                raise RuntimeError("Recovery reference directory mismatch")
            discard(handle, preparation_stopped=True)
            recovered += 1
        elif (
            meta.exists()
            and json.loads(meta.read_text()).get("sample_id") == directory.name
        ):
            # No published handle exists. An existing transaction needs manual reconciliation.
            plan = {"sample_id": directory.name}
            if _receipt_path({"plan": plan}).exists():
                unresolved.append(directory.name)
                continue
            for marker in directory.glob("*.pending"):
                marker.unlink()
            cleanup(plan)
            if directory.exists():
                raise RuntimeError("Unpublished window cleanup incomplete")
            unpublished += 1
        else:
            unresolved.append(directory.name)

    # Legacy tickets contain only a byte count. Release them only when every
    # sample directory in this stopped root has been accounted for.
    budget = root / ".reference-budget"
    tickets = 0
    if budget.is_dir() and not unresolved:
        with _ledger_lock(budget):
            for ticket in budget.glob("*/*.reservation"):
                if ticket.resolve() != ticket or ticket.is_symlink():
                    raise RuntimeError(f"Linked reservation: {ticket}")
                owner = ticket.parent.name
                if (
                    len(owner) != 32
                    or len(ticket.stem) != 32
                    or any(c not in "0123456789abcdef" for c in owner + ticket.stem)
                ):
                    unresolved.append(str(ticket.relative_to(root)))
                    continue
                ticket.unlink()
                tickets += 1
    return {
        "recovered_references": recovered,
        "unpublished_windows": unpublished,
        "orphan_tickets": tickets,
        "unresolved": unresolved,
        "root": str(root),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recover reference files only after the associated run is stopped"
    )
    parser.add_argument("--recover-stopped", action="store_true", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    os.environ["SPECO_CAPTURE_EXPORT_ROOT"] = str(Path(args.root).resolve())
    result = recover_stopped_root(args.root, preparation_stopped=args.recover_stopped)
    print(json.dumps(result))
    raise SystemExit(1 if result["unresolved"] else 0)
