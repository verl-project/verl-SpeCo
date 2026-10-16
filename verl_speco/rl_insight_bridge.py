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
"""Optional RL-Insight bridge for SPECO drafter co-training observability.

RL-Insight (``rl_insight``) is an optional dependency. Every helper in this
module degrades to a cheap no-op unless monitoring is explicitly enabled through
``VERL_RL_INSIGHT_ENABLE=1`` and the ``rl_insight`` package is importable, so
instrumentation can never change training results or crash a run.

The bridge is intentionally independent of ``verl.utils.tracking`` so it can be
used from Ray worker processes (for example ``SpecoWorker``) that never build a
``Tracking`` instance. It exposes only the two primitives SPECO needs — scalar
gauges (``log_scalar_metrics``) and state intervals (``trace_state``) — over the
same public rl-insight API that verl's ``RLInsightLogger`` uses, and it lazily
imports the package on first use so importing ``verl_speco`` never pulls in
``rl_insight``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"
PROJECT_ENV = "VERL_RL_INSIGHT_PROJECT_NAME"
EXPERIMENT_ENV = "VERL_RL_INSIGHT_EXPERIMENT_NAME"

__all__ = [
    "ENABLE_ENV",
    "PROJECT_ENV",
    "EXPERIMENT_ENV",
    "log_scalar_metrics",
    "trace_state",
    "reset_for_tests",
]

# Process-local module cache and one-shot init state. Kept as module globals so
# tests can reset them; values are only ever read/written from the owning
# process.
_module: Any | None = None
_init_state = "uninitialized"  # "uninitialized" | "ready" | "unavailable"


def _get_module() -> Any:
    global _module
    if _module is None:
        import rl_insight

        _module = rl_insight
    return _module


def _module_is_enabled() -> bool:
    """Best-effort probe of rl-insight's process state after ``init()``.

    ``rl_insight.init()`` returns silently without raising when the Ray backend
    has no server URL, leaving the module disabled. Detecting that lets callers
    stop paying the emit cost, but the probe falls back to ``True`` when the
    internal state object is missing so an upstream refactor cannot accidentally
    disable a working monitor.
    """
    state = getattr(_module, "_STATE", None)
    if state is None:
        return True
    return bool(getattr(state, "enabled", True))


def _ensure_ready() -> bool:
    """Lazily initialize rl-insight once; return False unless it is truly active.

    Safe to call from every emit helper: disabled runs, import failures, init
    failures, and the Ray backend's silent "no server URL" bail-out all collapse
    to a no-op.

    Initialization forwards the resolved project/experiment names from
    ``VERL_RL_INSIGHT_PROJECT_NAME`` / ``VERL_RL_INSIGHT_EXPERIMENT_NAME``. This
    is what gives Ray worker processes, which never build
    ``Tracking``/``RLInsightLogger``, the same ``project``/``experiment_name``
    labels as the trainer: rl-insight stores labels per process at init time and
    merges them into every event, and the shared hub does not add them. In the
    intended flow the trainer process builds ``Tracking``/``RLInsightLogger``
    first, so the duplicate call here is ignored by rl-insight and the labels
    already set by that first init win.
    """
    global _init_state
    if os.getenv(ENABLE_ENV) != "1" or _init_state == "unavailable":
        return False
    if _init_state == "ready":
        return True
    try:
        # verl's Tracking/RLInsightLogger may already have initialized the same
        # process; rl-insight warns and ignores the duplicate, which is benign.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _get_module().init(
                project=os.getenv(PROJECT_ENV) or None,
                experiment_name=os.getenv(EXPERIMENT_ENV) or None,
            )
    except Exception:
        _init_state = "unavailable"
        logger.warning(
            "[speco][rl-insight] init failed; monitoring is disabled for this process",
            exc_info=True,
        )
        return False
    if not _module_is_enabled():
        # init() bailed out (e.g. no server URL) without raising; record the
        # degraded state so later emits short-circuit instead of re-checking.
        _init_state = "unavailable"
        return False
    _init_state = "ready"
    return True


def _finite_float(value: Any) -> float | None:
    """Coerce a scalar (including 0-dim tensors/numpy scalars) to a finite float.

    Returns ``None`` for non-scalars, booleans, and non-finite values so they are
    skipped rather than exported as misleading zeros.
    """
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError, RuntimeError):
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sanitize_metric_name(name: str) -> str:
    """Convert an internal metric key or prefix into a Prometheus-legal name.

    Prometheus only accepts ``[a-zA-Z_:][a-zA-Z0-9_:]*``. SPECO keys and prefixes
    use ``/`` and sometimes ``-`` as separators and may carry ``.`` or spaces, so
    every character outside ``[a-zA-Z0-9_]`` is folded to ``_`` (``:`` is folded
    too, since recording-rule colons are reserved). A leading digit is prefixed
    with ``_``. Both the prefix and each key are sanitized before concatenation.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def log_scalar_metrics(
    prefix: str,
    metrics: Mapping[str, Any],
    **labels: Any,
) -> None:
    """Emit every finite scalar in ``metrics`` as a prefixed gauge.

    Non-scalar and non-finite values are skipped, matching the defensiveness
    RL-Insight's own logging path has toward tensor/NaN payloads. The ``prefix``
    and every key are sanitized before concatenation, and rl-insight is
    initialized at most once per process.
    """
    if not isinstance(metrics, Mapping) or not _ensure_ready():
        return
    metric_gauge = _get_module().metric_gauge
    sanitized_prefix = _sanitize_metric_name(prefix)
    for key, value in metrics.items():
        scalar = _finite_float(value)
        if scalar is None:
            continue
        name = f"{sanitized_prefix}{_sanitize_metric_name(str(key))}"
        try:
            metric_gauge(name, scalar, **labels)
        except Exception:
            logger.warning(
                "[speco][rl-insight] metric_gauge failed for %s", name, exc_info=True
            )


@contextmanager
def trace_state(
    state_name: str, *, state_lane_id: str | int | None = None, **labels: Any
) -> Iterator[None]:
    """Record a named SPECO state interval via rl-insight, no-op when disabled.

    The monitoring span is entered/exited around the ``with`` body but its
    errors are swallowed so a broken monitor never masks or replaces a training
    exception.
    """
    if not _ensure_ready():
        yield
        return
    try:
        ctx = _get_module().trace_state(
            state_name, state_lane_id=state_lane_id, **labels
        )
    except Exception:
        logger.warning(
            "[speco][rl-insight] trace_state(%s) setup failed",
            state_name,
            exc_info=True,
        )
        yield
        return
    try:
        ctx.__enter__()
    except Exception:
        logger.warning(
            "[speco][rl-insight] trace_state(%s) enter failed",
            state_name,
            exc_info=True,
        )
        yield
        return
    try:
        yield
    finally:
        # Forward the in-flight exception (if any) so monitoring can reflect the
        # failure; a monitor error is swallowed so it never masks the training
        # exception or replaces the body result.
        exc_info = sys.exc_info()
        try:
            ctx.__exit__(*exc_info)
        except Exception:
            logger.warning(
                "[speco][rl-insight] trace_state(%s) exit failed",
                state_name,
                exc_info=True,
            )


def reset_for_tests() -> None:
    """Reset process-local bridge state. Test-only helper."""
    global _module, _init_state
    _module = None
    _init_state = "uninitialized"
