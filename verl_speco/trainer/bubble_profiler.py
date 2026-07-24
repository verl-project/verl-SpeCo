# Copyright 2026 SPECO Authors
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
"""Bubble-time profiling helpers for SPECO online training.

SPECO inserts an extra draft-training stage into each RL step on top of the base
``rollout -> old_log_prob -> update_actor`` pipeline. Because that stage runs
serially after ``update_actor``, part of every step is spent with the training
GPUs doing work that could in principle overlap the generation window (where the
same GPUs are busy serving tokens instead of training). This module turns the
per-stage ``timing_s/*`` metrics the trainer already emits into a small set of
derived ``bubble/*`` metrics that make that opportunity measurable:

* ``bubble/unaccounted_s`` / ``bubble/unaccounted_ratio`` -- wall-clock time not
  attributed to any instrumented stage (pipeline gaps, sync, host overhead).
* ``bubble/drafter_s`` / ``bubble/drafter_ratio`` -- the serial draft-training
  add-on relative to the step.
* ``bubble/overlap_headroom_s`` / ``bubble/overlap_headroom_ratio`` -- an upper
  bound on the wall-clock recoverable by overlapping draft training into the
  generation window, i.e. ``min(gen, drafter)``.

It is pure bookkeeping over a metrics mapping: it never runs a model and imports
nothing beyond the standard library, so it can be unit tested on CPU and adds no
work to a step when the profiler is disabled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

BUBBLE_PREFIX = "bubble/"

STEP_KEY = "timing_s/step"
DRAFTER_KEY = "timing_s/drafter"

# The generation stage has gone by different metric names across verl releases;
# the first key that is present wins so we never double count it.
GEN_KEYS = ("timing_s/gen", "timing_s/generate_sequences", "timing_s/generation")

# Top-level (non-overlapping) stages whose durations add up to the step wall
# clock. Sub-stage breakdowns such as ``timing_s/drafter_train_rpc`` are
# deliberately excluded so they are not counted twice. Only the keys actually
# present in the metrics contribute, so the set can safely list more names than
# any single verl release emits.
DEFAULT_TOP_LEVEL_STAGES = (
    "timing_s/gen",
    "timing_s/generate_sequences",
    "timing_s/generation",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/values",
    "timing_s/adv",
    "timing_s/reward",
    "timing_s/update_actor",
    "timing_s/update_critic",
    "timing_s/drafter",
)


def _as_float(value: Any) -> float | None:
    """Coerce a metric value to a finite float, or ``None`` if not possible.

    Handles plain numbers and zero-dim tensors / numpy scalars (anything with a
    scalar ``.item()``). NaN and +/-inf are treated as missing.
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
    if not math.isfinite(result):
        return None
    return result


def _first_present(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def compute_bubble_metrics(
    metrics: Mapping[str, Any],
    *,
    stage_keys: tuple[str, ...] = DEFAULT_TOP_LEVEL_STAGES,
    prefix: str = BUBBLE_PREFIX,
) -> dict[str, float]:
    """Derive ``bubble/*`` metrics from a step's ``timing_s/*`` breakdown.

    Returns an empty dict when the step wall clock (``timing_s/step``) is absent
    or non-positive, e.g. for validation-only log payloads, so callers can treat
    an empty result as "nothing to add".
    """
    if not isinstance(metrics, Mapping):
        return {}
    step = _as_float(metrics.get(STEP_KEY))
    if step is None or step <= 0.0:
        return {}

    accounted = 0.0
    for key in stage_keys:
        value = _as_float(metrics.get(key))
        if value is not None and value > 0.0:
            accounted += value
    # Instrumented stages can never legitimately exceed the wall clock; clamp so
    # the unaccounted remainder stays non-negative even with timer skew.
    accounted = min(accounted, step)
    unaccounted = max(0.0, step - accounted)

    gen = _first_present(metrics, GEN_KEYS)
    gen = 0.0 if gen is None else max(0.0, min(gen, step))

    drafter = _as_float(metrics.get(DRAFTER_KEY))
    drafter = 0.0 if drafter is None else max(0.0, drafter)
    drafter_capped = min(drafter, step)

    overlap_headroom = min(gen, drafter)

    return {
        f"{prefix}step_s": step,
        f"{prefix}accounted_s": accounted,
        f"{prefix}unaccounted_s": unaccounted,
        f"{prefix}unaccounted_ratio": unaccounted / step,
        f"{prefix}gen_s": gen,
        f"{prefix}gen_ratio": gen / step,
        f"{prefix}drafter_s": drafter,
        f"{prefix}drafter_ratio": drafter_capped / step,
        f"{prefix}overlap_headroom_s": overlap_headroom,
        f"{prefix}overlap_headroom_ratio": overlap_headroom / step,
    }


def inject_bubble_metrics(
    data: Any,
    *,
    stage_keys: tuple[str, ...] = DEFAULT_TOP_LEVEL_STAGES,
    prefix: str = BUBBLE_PREFIX,
) -> Any:
    """Return ``data`` augmented with ``bubble/*`` metrics.

    ``data`` is returned unchanged (the same object) when it is not a dict or
    there is nothing to add, so this is a cheap no-op on non-training log
    payloads. When metrics are added a shallow copy is made; keys already present
    in ``data`` are never overwritten.
    """
    if not isinstance(data, dict):
        return data
    bubble = compute_bubble_metrics(data, stage_keys=stage_keys, prefix=prefix)
    if not bubble:
        return data
    merged = dict(data)
    for key, value in bubble.items():
        merged.setdefault(key, value)
    return merged
