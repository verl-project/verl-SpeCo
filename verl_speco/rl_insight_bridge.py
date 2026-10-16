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
import warnings
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"

__all__ = ["ENABLE_ENV", "log_scalar_metrics", "trace_state", "reset_for_tests"]

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


def _ensure_ready() -> bool:
    """Lazily initialize rl-insight once; return False when unavailable.

    Safe to call from every emit helper: disabled runs, import failures, and
    init failures all collapse to a no-op with at most one warning.

    Initialization is intentionally argument-less. In the intended flow the
    trainer process builds ``Tracking``/``RLInsightLogger`` first, which calls
    ``rl_insight.init(project=..., experiment_name=..., config=...)`` and creates
    the job-scoped hub; worker-side calls then attach to that existing hub and
    inherit its project/experiment labels. Calling ``init`` here is therefore a
    benign duplicate in the common case.
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
            _get_module().init()
    except Exception:
        _init_state = "unavailable"
        logger.warning(
            "[speco][rl-insight] init failed; monitoring is disabled for this process",
            exc_info=True,
        )
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
    """Convert an internal metric key into a Prometheus-legal metric name.

    Prometheus only accepts ``[a-zA-Z_:][a-zA-Z0-9_:]*``; SPECO metric keys use
    ``/`` and sometimes ``-`` as separators, so both are folded to ``_``.
    """
    return str(name).replace("/", "_").replace("-", "_")


def log_scalar_metrics(
    prefix: str,
    metrics: Mapping[str, Any],
    **labels: Any,
) -> None:
    """Emit every finite scalar in ``metrics`` as a prefixed gauge.

    Non-scalar and non-finite values are skipped, matching the defensiveness
    RL-Insight's own logging path has toward tensor/NaN payloads. ``prefix`` and
    keys are sanitized once, and rl-insight is initialized at most once per
    process.
    """
    if not isinstance(metrics, Mapping) or not _ensure_ready():
        return
    metric_gauge = _get_module().metric_gauge
    for key, value in metrics.items():
        scalar = _finite_float(value)
        if scalar is None:
            continue
        name = f"{prefix}{_sanitize_metric_name(str(key))}"
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
        try:
            ctx.__exit__(None, None, None)
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
