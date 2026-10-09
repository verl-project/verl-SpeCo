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
"""Handle-based hidden-state connector loaded by vLLM out-of-tree.

Wraps ``hs_connectors.MooncakeHiddenStatesConnector`` for vLLM 0.26 with fixes
required by the store-only, TP>1 setup:

* ``get_finished_count`` returns 1 because only TP rank 0 publishes; the default
  ``world_size`` leaves the request's blocks unfreed.
* ``get_finished`` only reports requests whose publish actually completed, and
  never reports from non-rank-0 workers. Upstream reports any accumulated
  finished id even without a submitted write, which trips
  ``assert req_id in self.requests`` in the scheduler and kills the engine.
* ``_ensure_store`` creates the accelerator context Mooncake's Ascend transport
  needs before allocating its local segment.
* ``register_kv_caches`` and the DtoH publish path are reimplemented on the
  device abstraction so the connector runs on non-CUDA backends (the upstream
  versions hard-code CUDA streams/events).

Observability (opt-in): when ``SPECO_HS_PROBE=1`` the connector appends a
write-path breakdown to ``SPECO_HS_PROBE_FILE`` (default
``/tmp/speco_hs_probe.log``): one line per ``SPECO_HS_PROBE_INTERVAL`` writes and
one line per store put. This is deliberately file-based because the vLLM worker
logger may not surface it at the configured level.

Loaded through ``kv_connector_module_path`` in ``--kv-transfer-config``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import torch
from hs_connectors import mooncake_hidden_states_connector as _upstream_hs
from hs_connectors.mooncake_hidden_states_connector import (
    MooncakeConnectorMetadata,
    MooncakeHiddenStatesConnector as _UpstreamMooncakeHiddenStatesConnector,
)
from vllm.logger import init_logger

from verl.utils.device import get_torch_device
from verl_speco.producer.hidden_states_store import ensure_accelerator_context

logger = init_logger(__name__)

_PROBE_LOCK = threading.Lock()
_PROBE_FH: Any = None


def _probe_enabled() -> bool:
    # Opt-in: set SPECO_HS_PROBE=1 to append a write-path breakdown to
    # SPECO_HS_PROBE_FILE (default /tmp/speco_hs_probe.log).
    return os.environ.get("SPECO_HS_PROBE", "0") == "1"


def _probe_interval() -> int:
    return int(os.environ.get("SPECO_HS_PROBE_INTERVAL", "100") or "100")


def _probe_write(line: str) -> None:
    """Append a timestamped probe line to the probe file (best effort)."""
    if not _probe_enabled():
        return
    global _PROBE_FH
    with _PROBE_LOCK:
        try:
            if _PROBE_FH is None:
                path = (
                    os.environ.get("SPECO_HS_PROBE_FILE") or "/tmp/speco_hs_probe.log"
                )
                _PROBE_FH = open(path, "a", buffering=1)
            _PROBE_FH.write(f"{time.time():.3f} {os.getpid()} {line}\n")
        except OSError:
            pass


class SpecoMooncakeHiddenStatesConnector(_UpstreamMooncakeHiddenStatesConnector):
    """Mooncake hidden-state connector with rank-0-only completion tracking."""

    def get_finished_count(self) -> int:
        return 1

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Reimplemented instead of delegating: the upstream version hard-codes a
        # CUDA stream. The device abstraction resolves the copy stream on the
        # active backend (e.g. NPU on Ascend).
        self._is_tp_rank_zero = _upstream_hs.get_tensor_model_parallel_rank() == 0

        from vllm.model_executor.models.extract_hidden_states import (  # noqa: PLC0415
            CacheOnlyAttentionLayer,
        )

        layers = _upstream_hs.get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        cache_layers = list(layers.keys())
        assert len(cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(cache_layers)}"
        )
        self._kv_cache = kv_caches[cache_layers[0]]
        self._copy_stream = get_torch_device().Stream()

    def _ensure_store(self) -> None:
        # Mooncake's Ascend transport needs an active device context to
        # allocate its local segment.
        ensure_accelerator_context()
        super()._ensure_store()
        if not _probe_enabled() or self._store is None or hasattr(self, "_probe"):
            return
        probe = _WriteProbe(interval=_probe_interval())
        probe.wrap(self._store)
        self._probe = probe
        _probe_write("probe_enabled")
        logger.warning("[hs-probe] write-path probe enabled")

    def _write_sample(self, pending: Any, ready_event: Any) -> None:
        probe = getattr(self, "_probe", None)
        if probe is None:
            return self._publish_sample(pending, ready_event)
        started = time.perf_counter()
        try:
            return self._publish_sample(pending, ready_event)
        finally:
            probe.record(
                total_ms=(time.perf_counter() - started) * 1000.0,
                tokens=int(pending.token_ids.shape[0]),
                num_layers=int(getattr(self, "num_hidden_states", 0) or 0),
            )

    def _publish_sample(self, pending: Any, ready_event: Any) -> None:
        # Device-agnostic reimplementation of the upstream DtoH publish. The
        # only backend-specific call is entering the copy stream; everything else
        # (tensor ops, wait_event, synchronize) is shared across backends.
        assert self._kv_cache is not None
        assert self._copy_stream is not None

        copy_stream = self._copy_stream
        copy_stream.wait_event(ready_event)

        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        num_blocks = block_ids_t.shape[0]
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()

        num_tokens = pending.token_ids.shape[0]

        try:
            with get_torch_device().stream(copy_stream):
                slot_mapping = slot_mapping.to(self._kv_cache.device, non_blocking=True)
                hidden_states = _upstream_hs.extract_from_kv_cache(
                    self._kv_cache,
                    slot_mapping,
                    num_tokens,
                )
                _upstream_hs.assert_finite("hidden_states", hidden_states)
                # Async DtoH copy into pinned host memory.
                pinned_hs = torch.empty_like(
                    hidden_states, device="cpu", pin_memory=True
                )
                pinned_hs.copy_(hidden_states, non_blocking=True)

            # Wait for the DtoH copy to complete before handing data to the store.
            copy_stream.synchronize()

            self._store.put_sample(
                pending.mooncake_key,
                {"hidden_states": pinned_hs, "token_ids": pending.token_ids},
            )
        except Exception as exc:
            try:
                # Store error marker instead of the sample, so consumer can
                # re-request.
                self._store.put_error(pending.mooncake_key, str(exc))
            except Exception:
                logger.exception(
                    "Failed to publish Mooncake error marker for %s",
                    pending.req_id,
                )
            raise

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Only rank 0 writes, so only it may report completion.
        if not self._is_tp_rank_zero:
            return None, None

        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, MooncakeConnectorMetadata):
                for pending in metadata.pending_saves:
                    if pending.req_id in self._req_futures:
                        continue
                    self._ensure_store()
                    ready_event = get_torch_device().Event()
                    ready_event.record()
                    self._req_futures[pending.req_id] = self._get_executor().submit(
                        self._write_sample, pending, ready_event
                    )

        self._accumulated_finished_req_ids.update(finished_req_ids)
        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            future = self._req_futures.get(req_id)
            if future is None:
                # Never submitted a publish (e.g. aborted before scheduling);
                # reporting it would free a request the scheduler already forgot.
                self._accumulated_finished_req_ids.discard(req_id)
                continue
            if not future.done():
                continue
            exc = future.exception()
            if exc is not None:
                # Report as finished even though the handle was never written:
                # keeping it pending would leave the engine waiting on a publish
                # that cannot succeed. The consumer then fails to fetch the
                # handle and the producer drops the sample (instead of the
                # engine dying).
                logger.error("Mooncake write failed for %s: %r", req_id, exc)
            self._req_futures.pop(req_id, None)
            done_sending.add(req_id)
            self._accumulated_finished_req_ids.discard(req_id)

        return done_sending or None, None


class _WriteProbe:
    """Aggregate write-path timings for the Mooncake connector (file-based)."""

    def __init__(self, *, interval: int = 100) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._interval = max(int(interval), 1)
        self._count = 0
        self._total_ms = 0.0
        self._put_ms = 0.0
        self._max_total = 0.0
        self._max_put = 0.0
        self._tokens = 0

    def wrap(self, store: Any) -> None:
        put = store.put_sample

        def _timed(key: str, tensors: dict[str, torch.Tensor], *args: Any, **kw: Any):
            started = time.perf_counter()
            try:
                return put(key, tensors, *args, **kw)
            finally:
                put_ms = (time.perf_counter() - started) * 1000.0
                self._local.put_ms = put_ms
                _probe_write(f"put put_ms={put_ms:.2f}")

        store.put_sample = _timed

    def record(self, *, total_ms: float, tokens: int, num_layers: int) -> None:
        put_ms = float(getattr(self._local, "put_ms", 0.0))
        with self._lock:
            self._count += 1
            self._total_ms += total_ms
            self._put_ms += put_ms
            self._max_total = max(self._max_total, total_ms)
            self._max_put = max(self._max_put, put_ms)
            self._tokens += tokens
            if self._count % self._interval != 0:
                return
            n = self._count
            _probe_write(
                "SUMMARY "
                f"n={n} avg_total={self._total_ms / n:.1f} "
                f"max_total={self._max_total:.1f} avg_put={self._put_ms / n:.1f} "
                f"max_put={self._max_put:.1f} "
                f"avg_extract_dtoh={(self._total_ms - self._put_ms) / n:.1f} "
                f"avg_tokens={self._tokens // n} layers={num_layers}"
            )


__all__ = ["SpecoMooncakeHiddenStatesConnector"]
