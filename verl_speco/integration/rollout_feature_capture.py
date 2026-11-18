# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Request-local target features from rollout forwards, without replay.

The collector receives device rows before serving buffers can be reused. It
keeps only rejection-sampler-confirmed inputs and validates against the final
trajectory before allowing any training consumer to see them.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch


@dataclass
class HostTransfer:
    tensor: torch.Tensor
    event: object = None
    source: object = None

    def result(self):
        if self.event is not None:
            self.event.synchronize()
        self.source = None
        return self.tensor


def handoff(tensor):
    """Own serving storage immediately; wait for DtoH on the capture thread."""
    source = tensor.detach().clone()
    if source.device.type == "cpu":
        return HostTransfer(source)
    backend = getattr(torch, source.device.type, None)
    if backend is None:
        raise RuntimeError(f"No stream backend for capture device {source.device}")
    host = torch.empty_like(source, device="cpu", pin_memory=True)
    host.copy_(source, non_blocking=True)
    event = backend.Event()
    event.record(backend.current_stream(source.device))
    return HostTransfer(host, event, source)


@dataclass
class RequestCapture:
    prompt_ids: tuple
    chunks: list = field(default_factory=list)
    futures: list = field(default_factory=list)
    byte_count: int = 0
    failure: str | None = None
    runtime_ids: set = field(default_factory=set)
    bytes_released: bool = False


class RolloutFeatureCollector:
    def __init__(
        self, *, max_requests=64, max_bytes=1024**3, max_rows=8192, timeout=120
    ):
        if min(max_requests, max_bytes, max_rows, timeout) <= 0:
            raise ValueError("Rollout capture capacity and timeout must be positive")
        self.max_requests, self.max_bytes, self.max_rows = (
            max_requests,
            max_bytes,
            max_rows,
        )
        self.timeout = timeout
        self.requests = {}
        self.pending_bytes = 0
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="speco-rollout-features"
        )
        self.lock = threading.RLock()

    def begin(self, request_id, prompt_ids):
        with self.lock:
            if request_id in self.requests:
                raise ValueError("Duplicate rollout capture request ID")
            if len(self.requests) >= self.max_requests:
                raise RuntimeError("Rollout capture request capacity exhausted")
            self.requests[request_id] = RequestCapture(tuple(prompt_ids))

    def _matching_request(self, request_id):
        matches = [
            key
            for key in self.requests
            if request_id == key or request_id.startswith(key + "-")
        ]
        return max(matches, key=len) if matches else None

    def watches_any(self, request_ids):
        with self.lock:
            return any(
                self._matching_request(str(request_id)) is not None
                for request_id in request_ids
            )

    def submit(
        self,
        request_id,
        *,
        input_ids,
        positions,
        hidden_states,
        target_head_hidden_states,
        sampled_token_ids,
        num_draft_tokens,
    ):
        with self.lock:
            key = self._matching_request(request_id)
            if key is None:
                return
            session = self.requests[key]
            if session.failure:
                return
            session.runtime_ids.add(request_id)
            if len(session.runtime_ids) != 1:
                session.failure = (
                    "Multiple completions share one capture session; require n=1"
                )
                return
            tensors = (
                input_ids,
                positions,
                hidden_states,
                target_head_hidden_states,
                sampled_token_ids,
            )
            size = sum(t.numel() * t.element_size() for t in tensors if t is not None)
            if self.pending_bytes + size > self.max_bytes:
                session.failure = "Rollout capture byte capacity exhausted"
                return
            self.pending_bytes += size
            session.byte_count += size
            try:
                handles = [handoff(t) if t is not None else None for t in tensors]
                session.futures.append(
                    self.pool.submit(self._consume, session, handles, num_draft_tokens)
                )
            except Exception as error:  # noqa: BLE001 -- invalidate, never expose partial rows
                session.failure = str(error)

    def fail(self, error):
        with self.lock:
            for session in self.requests.values():
                session.failure = str(error)

    def _consume(self, session, handles, drafts):
        try:
            ids, positions, hidden, head, sampled = [
                h.result() if h is not None else None for h in handles
            ]
            ids, positions, sampled = (
                ids.reshape(-1),
                positions.reshape(-1),
                sampled.reshape(-1),
            )
            count = ids.numel()
            if (
                positions.numel() != count
                or hidden.size(0) != count
                or (head is not None and head.size(0) != count)
            ):
                raise ValueError("Rollout feature rows do not match input token rows")
            if drafts:
                valid = sampled != -1
                keep = int(valid.sum())
                if not 1 <= keep <= drafts + 1 or count != drafts + 1:
                    raise ValueError("Invalid speculative verifier row count")
                if not torch.equal(valid, torch.arange(sampled.numel()) < keep):
                    raise ValueError("Verifier tokens must form a valid prefix")
                if bool((sampled[:keep] < 0).any()):
                    raise ValueError("Verifier returned negative token IDs")
                if not torch.equal(ids[1:keep], sampled[: keep - 1]):
                    raise ValueError("Accepted drafts differ from verifier output")
                ids, positions, hidden = ids[:keep], positions[:keep], hidden[:keep]
                head = head[:keep] if head is not None else None
            if (
                sum(chunk[0].numel() for chunk in session.chunks) + ids.numel()
                > self.max_rows
            ):
                raise ValueError("Rollout capture row capacity exhausted")
            session.chunks.append((ids, positions, hidden, head))
        except Exception as error:  # noqa: BLE001 -- failure is checked before feature publication
            session.failure = str(error)

    def finish(self, request_id, final_ids):
        with self.lock:
            session = self.requests.get(request_id)
        if session is None:
            return None
        try:
            for future in session.futures:
                future.result(timeout=self.timeout)
            if session.failure:
                raise ValueError(session.failure)
            if not session.chunks:
                return None  # This DP replica did not execute the request.
            final = torch.as_tensor(final_ids, dtype=torch.int64)
            prompt = torch.tensor(session.prompt_ids, dtype=torch.int64)
            if not torch.equal(final[: prompt.numel()], prompt):
                raise ValueError(
                    "Final rollout prompt differs from capture registration"
                )
            ids, positions, hidden = [
                torch.cat([chunk[i] for chunk in session.chunks]) for i in range(3)
            ]
            # The terminal sampled token has not been forwarded. Stop/EOS may
            # also truncate a verifier block: retain only the final trajectory.
            rows = final.numel() - 1
            if (
                rows < 1
                or ids.numel() < rows
                or not torch.equal(ids[:rows], final[:rows])
            ):
                raise ValueError(
                    "Rollout features do not cover the final confirmed token prefix"
                )
            if not torch.equal(positions[:rows].long(), torch.arange(rows)):
                raise ValueError("Rollout feature positions have a gap or duplicate")
            if not bool(torch.isfinite(hidden[:rows]).all()):
                raise ValueError("Rollout hidden states contain NaN/Inf")
            head_present = [chunk[3] is not None for chunk in session.chunks]
            if any(head_present) and not all(head_present):
                raise ValueError("Rollout target-head schema changed within a request")
            head = (
                torch.cat([chunk[3] for chunk in session.chunks])[:rows]
                if all(head_present)
                else None
            )
            if head is not None and not bool(torch.isfinite(head).all()):
                raise ValueError("Rollout target-head states contain NaN/Inf")
            return {
                "input_ids": ids[:rows].long().contiguous(),
                "positions": positions[:rows].long().contiguous(),
                "hidden_states": hidden[:rows],
                "target_head_hidden_states": head,
            }
        finally:
            self.cancel(request_id)

    def cancel(self, request_id):
        with self.lock:
            session = self.requests.pop(request_id, None)
            if session is not None:
                for future in session.futures:
                    future.cancel()
                # A cancelled request may still own an in-flight device copy.
                # Retain its byte reservation until every transfer has finished.
                for future in session.futures:
                    future.add_done_callback(lambda _, s=session: self._release(s))
                self._release(session)

    def _release(self, session):
        with self.lock:
            if not session.bytes_released and all(f.done() for f in session.futures):
                self.pending_bytes -= session.byte_count
                session.bytes_released = True
                session.chunks.clear()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.requests.clear()
        self.pending_bytes = 0
