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
"""Native vLLM draft weight-update selection and thin control-plane client.

verl/vLLM expose a draft-target weight-update session on V1 backends::

    start_draft_weight_update()
    update_weights(update_info)
    finish_weight_update()

SPECO keeps ownership of drafter training and publish scheduling, but defers
the actual transfer to that native session when the backend advertises it.  In
``auto`` (the default) the native path is used only when capability detection
succeeds; otherwise SPECO falls back to its draft-only compat implementation
(``speco_vllm_update_draft_weights``).  ``native`` fails fast when the backend
cannot do a draft update, and ``compat`` always forces the SPECO compat path.

The capability checks never rely on the HTTP endpoint existing alone: the
client API, the server RPC and the engine/worker capability must all be
present, because a backend can expose the endpoint while still rejecting the
draft update target.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WEIGHT_UPDATE_MODES = ("auto", "native", "compat")


def _get_nested(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return default if current is None else current


def resolve_weight_update_mode(config: Any) -> str:
    """Return the configured ``speco.runtime.weight_update_mode`` (default auto)."""

    raw = _get_nested(config, ("speco", "runtime", "weight_update_mode"), "auto")
    mode = str(raw).strip().lower() if raw is not None else "auto"
    if mode not in _WEIGHT_UPDATE_MODES:
        raise ValueError(
            "speco.runtime.weight_update_mode must be one of "
            f"{_WEIGHT_UPDATE_MODES}, got {raw!r}"
        )
    return mode


def _server_supports_draft_update(rollout: Any) -> bool:
    """Best-effort check that the vLLM server exposes the draft RPC."""

    server = getattr(rollout, "server_handle", None)
    if server is None:
        return False
    try:
        method = getattr(server, "start_draft_weight_update")
    except AttributeError:
        return False
    return callable(method) and hasattr(method, "remote")


def _worker_reports_draft_weight_support(rollout: Any) -> bool:
    """Read the engine/worker capability flag the runtime bridge publishes."""

    value = getattr(rollout, "supports_draft_weight_update", None)
    if value is None:
        return False
    if callable(value):
        try:
            return bool(value())
        except Exception:  # noqa: BLE001
            return False
    return bool(value)


def _ascend_worker_supports_draft_updates(rollout: Any) -> bool:
    """vLLM-Ascend Worker contract: ``supports_draft_weight_updates()`` (plural)."""

    method = getattr(rollout, "supports_draft_weight_updates", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception:  # noqa: BLE001
        return False


def resolve_draft_update_backend(config: Any, rollout: Any = None) -> str:
    """Return the rollout backend identity (``vllm`` or ``vllm-ascend``).

    Mirrors the light signals of ``vllm_runtime._is_vllm_ascend_runtime_hint``:
    Ascend env vars, ``VLLM_TARGET_DEVICE``/``VLLM_PLATFORM`` hints, the
    configured ``trainer.device`` and loaded ``vllm_ascend``/``torch_npu``
    modules.  The heavier device-name / vLLM-platform probes are intentionally
    skipped so the strategy decision stays import-light.
    """

    ascend_env_hints = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
    )
    if any(os.getenv(name) for name in ascend_env_hints):
        return "vllm-ascend"
    if str(os.getenv("VLLM_TARGET_DEVICE", "")).strip().lower() == "npu":
        return "vllm-ascend"
    if "ascend" in str(os.getenv("VLLM_PLATFORM", "")).strip().lower():
        return "vllm-ascend"
    device = _get_nested(config, ("trainer", "device"), None)
    if str(device).strip().lower() == "npu":
        return "vllm-ascend"
    if "vllm_ascend" in sys.modules and "torch_npu" in sys.modules:
        return "vllm-ascend"
    return "vllm"


def native_draft_update_available(rollout: Any, backend: Optional[str] = None) -> bool:
    """Return whether the rollout adapter can run a native draft update.

    ``backend="vllm-ascend"`` additionally requires the Ascend Worker contract
    (``supports_draft_weight_updates``) on top of the shared engine flag, so a
    backend that only ships the actor ``start_weight_update`` session is not
    misreported as draft-capable.
    """

    if backend == "vllm-ascend":
        return ascend_native_draft_update_available(rollout)
    if rollout is None:
        return False
    has_client_api = callable(getattr(rollout, "start_draft_weight_update", None))
    has_finish = callable(getattr(rollout, "finish_weight_update", None))
    has_update = callable(getattr(rollout, "update_weights", None))
    has_server_api = _server_supports_draft_update(rollout)
    supports_engine = _worker_reports_draft_weight_support(rollout)
    return (
        has_client_api
        and has_finish
        and has_update
        and has_server_api
        and supports_engine
    )


def ascend_native_draft_update_available(rollout: Any) -> bool:
    """Ascend-native draft update contract (Worker + WeightTransferEngine)."""

    if rollout is None:
        return False
    has_client_api = callable(getattr(rollout, "start_draft_weight_update", None))
    has_finish = callable(getattr(rollout, "finish_weight_update", None))
    has_update = callable(getattr(rollout, "update_weights", None))
    has_worker_supports = _ascend_worker_supports_draft_updates(rollout)
    engine_flag = _worker_reports_draft_weight_support(rollout)
    has_server_api = _server_supports_draft_update(rollout)
    return (
        has_client_api
        and has_finish
        and has_update
        and has_worker_supports
        and engine_flag
        and has_server_api
    )


def _describe_native_shortfall(rollout: Any, backend: str = "vllm") -> str:
    if rollout is None:
        return f"{backend}: rollout adapter not available"
    missing: list[str] = []
    if not callable(getattr(rollout, "start_draft_weight_update", None)):
        missing.append("start_draft_weight_update")
    if not callable(getattr(rollout, "finish_weight_update", None)):
        missing.append("finish_weight_update")
    if not _server_supports_draft_update(rollout):
        missing.append("server start_draft_weight_update RPC")
    if not _worker_reports_draft_weight_support(rollout):
        missing.append("supports_draft_weight_update")
    if backend == "vllm-ascend" and not _ascend_worker_supports_draft_updates(rollout):
        missing.append("Worker.supports_draft_weight_updates")
    return f"{backend}: " + (", ".join(missing) or "unknown")


@dataclass(frozen=True)
class DraftUpdateDecision:
    """Resolved draft weight-update strategy for a rollout adapter."""

    path: str
    mode: str
    backend: str
    native_available: bool
    reason: str


def select_draft_update_strategy(config: Any, rollout: Any) -> DraftUpdateDecision:
    """Resolve the draft update path from config and backend capability.

    ``auto`` uses the native session when available and otherwise falls back to
    the SPECO draft-only compat implementation.  ``native`` fails fast when the
    backend cannot do a draft update.  ``compat`` always forces the compat path.
    """

    backend = resolve_draft_update_backend(config, rollout)
    mode = resolve_weight_update_mode(config)
    native_available = (
        native_draft_update_available(rollout, backend=backend)
        if mode != "compat"
        else False
    )

    if mode == "compat":
        return DraftUpdateDecision(
            path="compat",
            mode=mode,
            backend=backend,
            native_available=native_available,
            reason=f"{backend}: weight_update_mode=compat forces SPECO draft-only compat",
        )

    if mode == "native":
        if not native_available:
            raise RuntimeError(
                "native draft update requested, but backend does not support: "
                + _describe_native_shortfall(rollout, backend)
            )
        return DraftUpdateDecision(
            path="native",
            mode=mode,
            backend=backend,
            native_available=True,
            reason=f"{backend}: weight_update_mode=native with native capability",
        )

    if native_available:
        return DraftUpdateDecision(
            path="native",
            mode=mode,
            backend=backend,
            native_available=True,
            reason=f"{backend}: native draft update available",
        )
    return DraftUpdateDecision(
        path="compat",
        mode=mode,
        backend=backend,
        native_available=False,
        reason=f"{backend}: native draft update unavailable: "
        + _describe_native_shortfall(rollout, backend),
    )


def _log_draft_update_decision(decision: DraftUpdateDecision) -> None:
    native_state = "enabled" if decision.native_available else "disabled"
    logger.warning(
        "[speco draft update] backend=%s native_draft_update=%s "
        "weight_update_mode=%s path=%s fallback_reason=%s",
        decision.backend,
        native_state,
        decision.mode,
        decision.path,
        decision.reason,
    )


class NativeVLLMDraftWeightSyncClient:
    """Thin control-plane client driving the vLLM native draft update session.

    The transport (bucketed sender, pause/resume, KV-cache cleanup) is owned by
    verl/vLLM's ``update_weights``; this client only orders the session start /
    transfer / finish calls and preserves the publish lifecycle (pause
    generation, flush KV cache, set global step, resume) around them.
    """

    def __init__(self, rollout: Any, decision: Optional[DraftUpdateDecision] = None):
        self.rollout = rollout
        self.decision = decision

    async def update(
        self, weights: Any, *args, global_steps: int | None = None, **kwargs
    ) -> None:
        del args, kwargs
        if not weights:
            return

        from verl_speco.integration.vllm_runtime import (
            _load_env_drafter_config,
            _maybe_call_vllm_server_method,
        )

        training_cfg = (_load_env_drafter_config() or {}).get("training") or {}
        pause_generation = bool(training_cfg.get("draft_update_pause_generation", True))
        flush_before = bool(training_cfg.get("draft_update_flush_before", True))
        flush_after = bool(training_cfg.get("draft_update_flush_after", True))

        rollout = self.rollout
        is_rank_zero = (
            getattr(rollout, "replica_rank", -1) == 0
            and getattr(rollout, "rollout_rank", -1) == 0
        )
        if is_rank_zero:
            logger.warning(
                "[speco vllm native draft update] starting global_steps=%s reason=%s",
                global_steps,
                self.decision.reason if self.decision else "native",
            )

        generation_paused = False
        try:
            if pause_generation:
                await _maybe_call_vllm_server_method(
                    rollout, "abort_all_requests", reset_prefix_cache=flush_before
                )
                generation_paused = True
            elif flush_before:
                await _maybe_call_vllm_server_method(rollout, "clear_kv_cache")

            await rollout.start_draft_weight_update()
            try:
                await rollout.update_weights(weights, global_steps=global_steps)
            finally:
                await rollout.finish_weight_update()

            if flush_after:
                await _maybe_call_vllm_server_method(rollout, "clear_kv_cache")
            if global_steps is not None:
                await _maybe_call_vllm_server_method(
                    rollout, "set_global_steps", global_steps
                )
        finally:
            if generation_paused:
                await _maybe_call_vllm_server_method(rollout, "resume_generation")


def _default_compat_attacher(rollout: Any) -> None:
    """Attach the SPECO draft-only compat ``update_draft_weights``."""

    from verl_speco.integration.vllm_runtime import (
        attach_update_draft_weights_to_rollout,
    )

    attach_update_draft_weights_to_rollout(rollout)


def attach_draft_weight_updater(config: Any, rollout: Any) -> Any:
    """Attach a draft ``update_draft_weights`` to a vLLM rollout adapter.

    Selects the native vLLM draft session or the SPECO compat path based on
    ``speco.runtime.weight_update_mode`` and the backend capability.  The
    decision is logged once; subsequent calls are idempotent so a rollout that
    already has ``update_draft_weights`` keeps its first attachment.
    """

    if rollout is None:
        return rollout
    if callable(getattr(rollout, "update_draft_weights", None)):
        return rollout

    decision = select_draft_update_strategy(config, rollout)
    _log_draft_update_decision(decision)
    if decision.path == "native":
        rollout.update_draft_weights = NativeVLLMDraftWeightSyncClient(
            rollout, decision
        ).update
    else:
        _default_compat_attacher(rollout)
    return rollout
