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
"""Drafter marginal-utility freeze policy (``marginal_utility_v1``).

This module replaces the single-metric, global-step tracker with a deeper
decision module that answers:

    Even under a statistically *optimistic* estimate, is the next drafter
    train + publish still worth its wall-clock cost for the overall request
    stream and for the hard (low accept-length) tail?

Design reference: ``drafter_marginal_utility_freeze_rfc.md``. The key ideas:

* The clock is the **successfully published drafter version** / the **update
  opportunity**, not the RL global step (RFC sec. 3).
* Quality is aggregated per deployed drafter version from the requests that
  version actually served (RFC sec. 4): ``Q_all`` is a 5% trimmed mean and
  ``Q_hard`` is the bottom-10% conditional mean (lower-tail CVaR).
* A relative gain is recorded only across a *real, successful* train+publish
  between two *valid* adjacent version windows (RFC sec. 5). A failed
  train/publish or an invalid window never becomes a zero-gain update.
* Freeze uses the 95% upper confidence bound of the recent gain so that
  "looks flat" is distinguished from "even the optimistic estimate is
  practically flat" (RFC sec. 7). CIs come from a moving-block bootstrap over
  rollout steps; the noise floor is calibrated from same-version split-window
  pseudo gains (RFC sec. 7.3).
* A quality drawdown guard forbids freezing while current quality is
  significantly below the recent best (RFC sec. 7.4).
* Freeze requires several consecutive confirming updates; a quality drift
  versus the freeze anchor resumes training (RFC sec. 9/10).
* Missing or invalid evidence always fails *open*: keep training.

The module is pure Python (no numpy/torch/ray) so it runs under ``unittest``
in any environment. The trainer only supplies facts (:class:`FreezeEvidence`)
and enforces the returned decision; it never computes UCB/ROI/state itself.

Phase 3b scope (RFC sec. 6/8): fixed paired probes run before/after every
``probe.interval_updates``-th successful publish on the identical target
checkpoint/prompt set (greedy decoding, so both arms produce the same verified
token stream -- speculative decoding is lossless). Per-request aligned
accept-length and timing pairs drive a paired bootstrap over request ids
(RFC sec. 7.2); its UCB95 of ``ROI_next`` is the *only* evidence the
economics gate accepts. Online-window speed economics stay telemetry-only.
The gate fails open (keeps training) whenever complete, fresh paired probe
evidence is missing. ``mode`` (shadow/active) is intentionally *not* known to
this class: shadow and active run identical decision code; the trainer decides
whether to enforce ``should_train=False`` (RFC sec. 17.4).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field, replace as dataclass_replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "FreezeState",
    "ProbeComparison",
    "FreezeEvidence",
    "FreezeDecision",
    "VersionEvidence",
    "DrafterFreezePolicy",
    "trimmed_mean",
    "bottom_tail_mean",
]

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Pure quality statistics (RFC sec. 4)
# --------------------------------------------------------------------------- #
def trimmed_mean(values: Sequence[float], trim_fraction: float = 0.05) -> float:
    """Mean of ``values`` after trimming ``trim_fraction`` from *each* tail.

    ``Q_all`` in the RFC (default 5%/5%). More robust than a plain mean to a
    few anomalous requests while retaining more continuous information than a
    median.
    """
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(float(v) for v in values)
    k = int(trim_fraction * n)  # floor: never trim more than we are sure about
    core = ordered[k : n - k] if n - 2 * k > 0 else ordered
    return statistics.fmean(core)


def bottom_tail_mean(values: Sequence[float], quantile: float = 0.10) -> float:
    """Mean of the worst ``ceil(quantile*n)`` requests.

    ``Q_hard`` in the RFC (lower-tail CVaR). Uses ``ceil`` (not round) so even
    tiny samples always contribute >= 1 hard request.
    """
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(float(v) for v in values)
    m = max(1, math.ceil(quantile * n))
    return statistics.fmean(ordered[:m])


def _plain_quality(values: Sequence[float], trim: float, tau: float) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "q_all": 0.0, "q_hard": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "n": float(n),
        "q_all": trimmed_mean(values, trim),
        "q_hard": bottom_tail_mean(values, tau),
        "mean": statistics.fmean(float(v) for v in values),
        "median": statistics.median(float(v) for v in values),
    }


def _quantile(sorted_values: Sequence[float], prob: float) -> float:
    """Linear-interpolation quantile over an already-sorted sample."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, prob)) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


class FreezeState(str, Enum):
    CALIBRATING = "CALIBRATING"
    LEARNING = "LEARNING"
    PLATEAU_CANDIDATE = "PLATEAU_CANDIDATE"
    NO_RESPONSE_CANDIDATE = "NO_RESPONSE_CANDIDATE"
    FROZEN = "FROZEN"
    RECOVERING = "RECOVERING"


# --------------------------------------------------------------------------- #
# Evidence / decision data structures (RFC sec. 12.2/12.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeComparison:
    """Paired before/after drafter probe on a fixed target checkpoint (RFC 6).

    All arrays are aligned by probe request id (row ``i`` is the same fixed
    prompt served by drafter ``version_before`` and ``version_after``). The
    executor runs greedy decoding on an identical target checkpoint, so both
    arms produce the same verified token stream; per-request generated target
    tokens and engine wall seconds therefore compare the same token workload
    (RFC sec. 6.4).

    ``tokens_*`` are generated TARGET token counts per request
    (accepted draft tokens + verify rounds); ``seconds_*`` are engine request
    lifetimes. Pairs with missing/non-positive timing are excluded from speed
    statistics but kept for accept-length statistics.
    """

    request_ids: Sequence[str]
    accept_before: Sequence[float]
    accept_after: Sequence[float]
    tokens_before: Optional[Sequence[float]] = None
    seconds_before: Optional[Sequence[float]] = None
    tokens_after: Optional[Sequence[float]] = None
    seconds_after: Optional[Sequence[float]] = None
    version_before: Optional[int] = None
    version_after: Optional[int] = None
    global_step: Optional[int] = None
    wall_seconds: Optional[float] = None

    def _aligned_rows(self) -> list[tuple[str, Optional[float], Optional[float]]]:
        return [
            (str(rid), before, after)
            for rid, before, after in zip(
                self.request_ids, self.accept_before, self.accept_after
            )
        ]

    def paired_relative_gains(self) -> list[float]:
        """Per-pair relative accept-length gain (after-before)/before."""
        out = []
        for _rid, before, after in self._aligned_rows():
            if before is None or after is None:
                continue
            denom = max(abs(float(before)), _EPS)
            out.append((float(after) - float(before)) / denom)
        return out

    def accept_pairs(self) -> list[tuple[str, float, float, float]]:
        """Complete accept rows: (id, accept_before, accept_after, gain)."""
        out = []
        for rid, before, after in self._aligned_rows():
            if before is None or after is None:
                continue
            before_f, after_f = float(before), float(after)
            out.append(
                (rid, before_f, after_f, (after_f - before_f) / max(abs(before_f), _EPS))
            )
        return out

    def timing_pairs(self) -> list[tuple[str, float, float, float, float]]:
        """Complete timing rows: (id, tokens_b, seconds_b, tokens_a, seconds_a)."""
        if (
            self.tokens_before is None
            or self.tokens_after is None
            or self.seconds_before is None
            or self.seconds_after is None
        ):
            return []
        out = []
        for rid, tok_b, sec_b, tok_a, sec_a in zip(
            self.request_ids,
            self.tokens_before,
            self.seconds_before,
            self.tokens_after,
            self.seconds_after,
        ):
            values = (tok_b, sec_b, tok_a, sec_a)
            if any(v is None for v in values):
                continue
            tok_b_f, sec_b_f, tok_a_f, sec_a_f = (float(v) for v in values)
            if tok_b_f <= 0 or sec_b_f <= 0 or tok_a_f <= 0 or sec_a_f <= 0:
                continue
            out.append((str(rid), tok_b_f, sec_b_f, tok_a_f, sec_a_f))
        return out

    def n_requests(self) -> int:
        return min(
            len(self.request_ids), len(self.accept_before), len(self.accept_after)
        )

    def timing_coverage(self) -> float:
        n = self.n_requests()
        if n <= 0:
            return 0.0
        return len(self.timing_pairs()) / n

    def to_dict(self) -> dict[str, Any]:
        def floats(values):
            return (
                None
                if values is None
                else [None if v is None else float(v) for v in values]
            )

        return {
            "request_ids": [str(v) for v in self.request_ids],
            "accept_before": floats(self.accept_before),
            "accept_after": floats(self.accept_after),
            "tokens_before": floats(self.tokens_before),
            "tokens_after": floats(self.tokens_after),
            "seconds_before": floats(self.seconds_before),
            "seconds_after": floats(self.seconds_after),
            "version_before": self.version_before,
            "version_after": self.version_after,
            "global_step": self.global_step,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProbeComparison":
        return cls(
            request_ids=list(data.get("request_ids", [])),
            accept_before=list(data.get("accept_before", [])),
            accept_after=list(data.get("accept_after", [])),
            tokens_before=data.get("tokens_before"),
            seconds_before=data.get("seconds_before"),
            tokens_after=data.get("tokens_after"),
            seconds_after=data.get("seconds_after"),
            version_before=data.get("version_before"),
            version_after=data.get("version_after"),
            global_step=data.get("global_step"),
            wall_seconds=data.get("wall_seconds"),
        )


@dataclass(frozen=True)
class FreezeEvidence:
    """One fact handed to the policy.

    Two event kinds share this type, selected by ``publish_completed``:

    * rollout event (``publish_completed=False``): the per-request accept lens
      and timing for the requests the *currently serving* version produced.
      ``drafter_version`` here is the version that actually served the rollout
      (captured at generation time, before any end-of-step publish).
    * publish event (``publish_completed=True``): a real train+publish finished
      and the rollout runtime has switched to ``drafter_version`` (the new
      version). This closes the previous version window and advances the clock.
      ``update_cost_seconds`` is the critical-path cost of that update.
    """

    global_step: int
    drafter_version: int
    request_accept_lens: Sequence[float] = field(default_factory=list)
    rollout_steps: int = 1
    response_tokens: int = 0
    generation_seconds: float = 0.0
    opportunity_this_step: bool = False
    low_fraction: Optional[float] = None
    update_succeeded: bool = False
    publish_succeeded: bool = False
    publish_completed: bool = False
    update_cost_seconds: Optional[float] = None
    probe: Optional[ProbeComparison] = None


@dataclass(frozen=True)
class FreezeDecision:
    state: FreezeState
    should_train: bool
    should_collect: bool
    should_probe: bool
    transitioned: bool
    reason: str
    metrics: Mapping[str, float]
    drafter_version: int
    update_opportunity_id: int
    valid_update_count: int

    @property
    def would_freeze(self) -> bool:
        """Intrinsic freeze intent regardless of shadow/active enforcement."""
        return self.state == FreezeState.FROZEN


@dataclass
class VersionEvidence:
    """Finalized, JSON-serializable summary of one deployed version window."""

    version: int
    global_step: int
    n_requests: int
    n_rollout_steps: int
    valid: bool
    q_all: float = 0.0
    q_hard: float = 0.0
    q_speed: float = 0.0
    plain_mean: float = 0.0
    median: float = 0.0
    response_tokens: int = 0
    generation_seconds: float = 0.0
    low_fraction: Optional[float] = None
    g_all: Optional[float] = None
    g_hard: Optional[float] = None
    g_speed: Optional[float] = None
    cost_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionEvidence":
        return cls(**dict(data))


# --------------------------------------------------------------------------- #
# Internal window bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class _StepStat:
    accept_lens: list[float]
    response_tokens: int
    generation_seconds: float
    low_fraction: Optional[float]


@dataclass
class _Window:
    version: int
    opened_step: int
    steps: list[_StepStat] = field(default_factory=list)

    def pooled_lens(self) -> list[float]:
        out: list[float] = []
        for step in self.steps:
            out.extend(step.accept_lens)
        return out

    def total_tokens(self) -> int:
        return int(sum(s.response_tokens for s in self.steps))

    def total_seconds(self) -> float:
        return float(sum(s.generation_seconds for s in self.steps))

    def latest_low_fraction(self) -> Optional[float]:
        for step in reversed(self.steps):
            if step.low_fraction is not None:
                return step.low_fraction
        return None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PolicyConfig:
    # observation
    min_requests_per_version: int = 256
    min_rollout_steps_per_version: int = 2
    trim_fraction: float = 0.05
    hard_quantile: float = 0.10
    spec_verify_tokens: Optional[int] = None
    # gain / plateau
    window_updates: int = 3
    patience_updates: int = 3
    confidence_level: float = 0.95
    bootstrap_samples: int = 300
    practical_gain_all: float = 0.005
    practical_gain_hard: float = 0.005
    growth_epsilon: float = 0.005
    freeze_guard_drop_all: float = 0.02
    freeze_guard_drop_hard: float = 0.03
    # noise floor
    noise_quantile: float = 0.95
    noise_min_pseudo_samples: int = 20
    # economics
    economics_enabled: bool = False
    cost_margin: float = 1.0
    cost_window_updates: int = 3
    # probe
    probe_enabled: bool = False
    probe_interval_updates: int = 3
    # Minimum fraction of probe requests with complete paired timing for the
    # evidence to make the economics gate ready (RFC 7.2 item 3: insufficient
    # paired coverage -> no freeze decision via the economics gate).
    probe_min_coverage: float = 0.8
    probe_min_timing_pairs: int = 16
    # resume / drift
    resume_drop_all: float = 0.05
    resume_drop_hard: float = 0.05
    resume_patience_opportunities: int = 3
    low_fraction_accelerator: float = 0.10
    missing_evidence_opportunities: int = 3
    max_frozen_opportunities: int = 12
    # enforcement scope
    freeze_scope: str = "soft"
    bootstrap_seed: int = 20260915

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> "_PolicyConfig":
        def block(name: str) -> Mapping[str, Any]:
            value = raw.get(name, {}) if raw is not None else {}
            return value if isinstance(value, Mapping) else {}

        obs, gain = block("observation"), block("gain")
        noise, econ = block("noise"), block("economics")
        probe, resume = block("probe"), block("resume")

        def f(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
            return mapping[key] if key in mapping else default

        return _PolicyConfig(
            min_requests_per_version=int(f(obs, "min_requests_per_version", 256)),
            min_rollout_steps_per_version=int(
                f(obs, "min_rollout_steps_per_version", 2)
            ),
            trim_fraction=float(f(obs, "trim_fraction", 0.05)),
            hard_quantile=float(f(obs, "hard_quantile", 0.10)),
            spec_verify_tokens=raw.get("spec_verify_tokens", None) if raw else None,
            window_updates=int(f(gain, "window_updates", 3)),
            patience_updates=int(f(gain, "patience_updates", 3)),
            confidence_level=float(f(gain, "confidence_level", 0.95)),
            bootstrap_samples=int(f(gain, "bootstrap_samples", 300)),
            practical_gain_all=float(f(gain, "practical_gain_all", 0.005)),
            practical_gain_hard=float(f(gain, "practical_gain_hard", 0.005)),
            growth_epsilon=float(f(gain, "growth_epsilon", 0.005)),
            freeze_guard_drop_all=float(f(gain, "freeze_guard_drop_all", 0.02)),
            freeze_guard_drop_hard=float(f(gain, "freeze_guard_drop_hard", 0.03)),
            noise_quantile=float(f(noise, "quantile", 0.95)),
            noise_min_pseudo_samples=int(f(noise, "min_pseudo_samples", 20)),
            economics_enabled=bool(f(econ, "enabled", False)),
            cost_margin=float(f(econ, "cost_margin", 1.0)),
            cost_window_updates=int(f(econ, "cost_window_updates", 3)),
            probe_enabled=bool(f(probe, "enabled", False)),
            probe_interval_updates=int(f(probe, "interval_updates", 3)),
            probe_min_coverage=float(f(probe, "min_coverage", 0.8)),
            probe_min_timing_pairs=int(f(probe, "min_timing_pairs", 16)),
            resume_drop_all=float(f(resume, "drop_all", 0.05)),
            resume_drop_hard=float(f(resume, "drop_hard", 0.05)),
            resume_patience_opportunities=int(
                f(resume, "patience_opportunities", 3)
            ),
            low_fraction_accelerator=float(
                f(resume, "low_fraction_accelerator", 0.10)
            ),
            missing_evidence_opportunities=int(
                f(resume, "missing_evidence_opportunities", 3)
            ),
            max_frozen_opportunities=int(f(resume, "max_frozen_opportunities", 12)),
            freeze_scope=str(raw.get("freeze_scope", "soft") if raw else "soft"),
            bootstrap_seed=int(f(probe, "seed", 20260915)),
        )

    def fingerprint_parts(self) -> dict[str, Any]:
        """Values that, if changed on resume, force re-calibration (RFC 12.4)."""
        return {
            "min_requests_per_version": self.min_requests_per_version,
            "min_rollout_steps_per_version": self.min_rollout_steps_per_version,
            "trim_fraction": self.trim_fraction,
            "hard_quantile": self.hard_quantile,
            "window_updates": self.window_updates,
            "patience_updates": self.patience_updates,
            "confidence_level": self.confidence_level,
            "practical_gain_all": self.practical_gain_all,
            "practical_gain_hard": self.practical_gain_hard,
            "growth_epsilon": self.growth_epsilon,
            "freeze_guard_drop_all": self.freeze_guard_drop_all,
            "freeze_guard_drop_hard": self.freeze_guard_drop_hard,
            "noise_quantile": self.noise_quantile,
            "noise_min_pseudo_samples": self.noise_min_pseudo_samples,
            "economics_enabled": self.economics_enabled,
            "cost_margin": self.cost_margin,
            "cost_window_updates": self.cost_window_updates,
            "probe_enabled": self.probe_enabled,
            "probe_interval_updates": self.probe_interval_updates,
            "probe_min_coverage": self.probe_min_coverage,
            "probe_min_timing_pairs": self.probe_min_timing_pairs,
            "resume_drop_all": self.resume_drop_all,
            "resume_drop_hard": self.resume_drop_hard,
            "resume_patience_opportunities": self.resume_patience_opportunities,
            "freeze_scope": self.freeze_scope,
            "spec_verify_tokens": self.spec_verify_tokens,
        }

    def fingerprint(self) -> str:
        blob = json.dumps(
            self.fingerprint_parts(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
class DrafterFreezePolicy:
    """Version-clock marginal-utility freeze state machine.

    The trainer calls :meth:`observe` with one :class:`FreezeEvidence` at a time
    and enforces the returned :class:`FreezeDecision`. The policy owns the
    drafter version and update-opportunity counters so there is a single source
    of truth for the clock.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config_raw: Optional[Mapping[str, Any]] = config
        self.cfg = _PolicyConfig.from_mapping(config or {})

        # Clock. ``drafter_version`` advances the instant a publish is staged
        # (the authoritative published version the *next* rollout will serve);
        # the served window is only closed once rollout evidence tagged with the
        # new version arrives, which makes attribution independent of whether
        # the trainer calls publish before/after logging the step (RFC 3.2).
        self.drafter_version = 0
        self.update_opportunity_id = 0
        self.valid_update_count = 0
        self._pending_next_version: Optional[int] = None
        self._pending_cost: Optional[float] = None
        self.dropped_late_rollouts = 0

        # Windows / evidence.
        self._windows: dict[int, _Window] = {}
        self._open_version = 0
        self._windows[self._open_version] = _Window(0, opened_step=-1)
        self.versions: list[VersionEvidence] = []
        self._gains: list[dict[str, Optional[float]]] = []  # adjacent valid pairs
        self._consecutive_valid_gains = 0
        self._growth_seen_all = False
        self._growth_seen_hard = False

        # Empirical noise floor (same-version split-window pseudo gains).
        self._pseudo_noise_all: list[float] = []
        self._pseudo_noise_hard: list[float] = []

        # State machine.
        self.state = FreezeState.CALIBRATING
        self._candidate_streak = 0
        self._candidate_kind: Optional[str] = None  # plateau | no_response
        self._resume_streak = 0
        self._missing_streak = 0
        self._frozen_opportunities = 0
        self._anchor: Optional[VersionEvidence] = None
        self._anchor_low_fraction: Optional[float] = None

        # Economics bookkeeping.
        self._costs: list[float] = []
        self.cumulative_cost_seconds = 0.0
        self.cumulative_saved_seconds_estimate = 0.0

        # Latest completed fixed paired probe (RFC 6). Carried on publish
        # events; the economics gate may use only a complete, fresh probe.
        self._latest_probe: Optional[ProbeComparison] = None
        self._latest_probe_stats: Optional[dict[str, Any]] = None
        # Probes completed / skipped / rejected as incomplete (telemetry).
        self.probe_completed = 0
        self.probe_incomplete = 0
        # Total wall seconds spent running probe arms (telemetry).
        self.probe_wall_seconds_total = 0.0

        self._last_reason = "insufficient_evidence"
        self._last_transition: Optional[dict[str, Any]] = None
        self._bootstrap_seen = 0  # only used to vary nothing (seed is derived)

        self._config_fingerprint = self.cfg.fingerprint()

    # ------------------------------------------------------------------ #
    # Public API (RFC 12.1)
    # ------------------------------------------------------------------ #
    def observe(self, evidence: FreezeEvidence) -> FreezeDecision:
        if evidence.publish_completed:
            return self._handle_publish(evidence)
        return self._handle_rollout(evidence)

    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable policy state (RFC 12.4)."""
        retain_versions = self._versions_to_retain_steps()
        raw_windows = {
            version: {
                "opened_step": win.opened_step,
                "steps": [
                    {
                        "accept_lens": list(step.accept_lens),
                        "response_tokens": step.response_tokens,
                        "generation_seconds": step.generation_seconds,
                        "low_fraction": step.low_fraction,
                    }
                    for step in win.steps
                ],
            }
            for version, win in self._windows.items()
            if version in retain_versions
        }
        return {
            "config_fingerprint": self._config_fingerprint,
            "state": self.state.value,
            "drafter_version": self.drafter_version,
            "update_opportunity_id": self.update_opportunity_id,
            "valid_update_count": self.valid_update_count,
            "open_version": self._open_version,
            "pending_next_version": self._pending_next_version,
            "pending_cost": self._pending_cost,
            "dropped_late_rollouts": self.dropped_late_rollouts,
            "windows": raw_windows,
            "versions": [v.to_dict() for v in self.versions],
            "gains": list(self._gains),
            "consecutive_valid_gains": self._consecutive_valid_gains,
            "growth_seen_all": self._growth_seen_all,
            "growth_seen_hard": self._growth_seen_hard,
            "pseudo_noise_all": list(self._pseudo_noise_all),
            "pseudo_noise_hard": list(self._pseudo_noise_hard),
            "candidate_streak": self._candidate_streak,
            "candidate_kind": self._candidate_kind,
            "resume_streak": self._resume_streak,
            "missing_streak": self._missing_streak,
            "frozen_opportunities": self._frozen_opportunities,
            "anchor": self._anchor.to_dict() if self._anchor else None,
            "anchor_low_fraction": self._anchor_low_fraction,
            "costs": list(self._costs),
            "cumulative_cost_seconds": self.cumulative_cost_seconds,
            "cumulative_saved_seconds_estimate": self.cumulative_saved_seconds_estimate,
            "latest_probe": (
                self._latest_probe.to_dict() if self._latest_probe is not None else None
            ),
            "probe_completed": self.probe_completed,
            "probe_incomplete": self.probe_incomplete,
            "probe_wall_seconds_total": self.probe_wall_seconds_total,
            "last_reason": self._last_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        fingerprint = state.get("config_fingerprint")
        if fingerprint is not None and fingerprint != self.cfg.fingerprint():
            # Config changed since the checkpoint: old threshold state is
            # meaningless. Fail safe by rebuilding calibration state from
            # scratch while keeping the *current* config (RFC 12.4).
            raw = self._config_raw
            self.__init__(config=raw)
            self._last_transition = {
                "event": "drafter_freeze_transition",
                "to": FreezeState.CALIBRATING.value,
                "reason": "config_fingerprint_mismatch",
                "global_step": 0,
                "update_opportunity_id": 0,
                "drafter_version": 0,
            }
            return
        self._load_state_dict_unchecked(state)

    def _load_state_dict_unchecked(self, state: Mapping[str, Any]) -> None:
        self.drafter_version = int(state.get("drafter_version", 0))
        self.update_opportunity_id = int(state.get("update_opportunity_id", 0))
        self.valid_update_count = int(state.get("valid_update_count", 0))
        self._open_version = int(state.get("open_version", self.drafter_version))
        self._pending_next_version = state.get("pending_next_version", None)
        self._pending_cost = state.get("pending_cost", None)
        self.dropped_late_rollouts = int(state.get("dropped_late_rollouts", 0))
        self._windows = {}
        for version, raw in (state.get("windows", {}) or {}).items():
            win = _Window(int(version), opened_step=int(raw.get("opened_step", -1)))
            for step in raw.get("steps", []):
                win.steps.append(
                    _StepStat(
                        accept_lens=[float(v) for v in step.get("accept_lens", [])],
                        response_tokens=int(step.get("response_tokens", 0)),
                        generation_seconds=float(step.get("generation_seconds", 0.0)),
                        low_fraction=step.get("low_fraction"),
                    )
                )
            self._windows[int(version)] = win
        self._windows.setdefault(
            self._open_version, _Window(self._open_version, opened_step=-1)
        )
        self.versions = [
            VersionEvidence.from_dict(v) for v in state.get("versions", [])
        ]
        self._gains = [dict(g) for g in state.get("gains", [])]
        self._consecutive_valid_gains = int(
            state.get("consecutive_valid_gains", 0)
        )
        self._growth_seen_all = bool(state.get("growth_seen_all", False))
        self._growth_seen_hard = bool(state.get("growth_seen_hard", False))
        self._pseudo_noise_all = [float(v) for v in state.get("pseudo_noise_all", [])]
        self._pseudo_noise_hard = list(
            float(v) for v in state.get("pseudo_noise_hard", [])
        )
        self._candidate_streak = int(state.get("candidate_streak", 0))
        self._candidate_kind = state.get("candidate_kind", None)
        self._resume_streak = int(state.get("resume_streak", 0))
        self._missing_streak = int(state.get("missing_streak", 0))
        self._frozen_opportunities = int(state.get("frozen_opportunities", 0))
        anchor = state.get("anchor", None)
        self._anchor = VersionEvidence.from_dict(anchor) if anchor else None
        self._anchor_low_fraction = state.get("anchor_low_fraction", None)
        self._costs = [float(v) for v in state.get("costs", [])]
        self.cumulative_cost_seconds = float(
            state.get("cumulative_cost_seconds", 0.0)
        )
        self.cumulative_saved_seconds_estimate = float(
            state.get("cumulative_saved_seconds_estimate", 0.0)
        )
        probe_state = state.get("latest_probe", None)
        if isinstance(probe_state, Mapping) and probe_state.get("request_ids"):
            self._latest_probe = ProbeComparison.from_dict(probe_state)
            # Stats are deterministic functions of the probe + config; rebuild
            # immediately so the gate is available without a re-ingest.
            self._latest_probe_stats = self._summarize_probe(self._latest_probe)
        else:
            self._latest_probe = None
            self._latest_probe_stats = None
        self.probe_completed = int(state.get("probe_completed", 0))
        self.probe_incomplete = int(state.get("probe_incomplete", 0))
        self.probe_wall_seconds_total = float(
            state.get("probe_wall_seconds_total", 0.0)
        )
        self.state = FreezeState(state.get("state", FreezeState.CALIBRATING.value))
        self._last_reason = str(state.get("last_reason", "insufficient_evidence"))

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def _handle_rollout(self, ev: FreezeEvidence) -> FreezeDecision:
        if ev.opportunity_this_step:
            self.update_opportunity_id += 1

        serving = int(ev.drafter_version)
        lens = [float(v) for v in ev.request_accept_lens if v is not None]
        pending = self._pending_next_version
        transitioned = False
        reason = self._last_reason

        if pending is not None and serving >= pending:
            # The new version is already serving and the publishing step had
            # no post-publish old-version observation: close the old window
            # first, then ingest into the newly-serving window.
            transitioned, reason = self._advance_version(
                pending, self._pending_cost, ev
            )
            self._pending_next_version = None
            self._pending_cost = None
            win = self._windows.setdefault(
                self._open_version,
                _Window(self._open_version, opened_step=ev.global_step),
            )
            self._append_step(win, lens, ev)
        elif pending is not None and serving == self._open_version and serving < pending:
            # The publishing step's own evidence still carries the OLD serving
            # version: it is the final observation of that window -- ingest it,
            # then close and open the published version.
            win = self._windows.setdefault(
                serving, _Window(serving, opened_step=ev.global_step)
            )
            self._append_step(win, lens, ev)
            transitioned, reason = self._advance_version(
                pending, self._pending_cost, ev
            )
            self._pending_next_version = None
            self._pending_cost = None
        else:
            # Ordinary accumulation. Late evidence for an already-finalized
            # window is dropped rather than folded into a closed version.
            win = self._windows.get(serving)
            if win is None and serving == self._open_version:
                win = self._windows.setdefault(
                    serving, _Window(serving, opened_step=ev.global_step)
                )
            if win is None:
                self.dropped_late_rollouts += 1
            else:
                self._append_step(win, lens, ev)

        # Drift is evaluated on opportunities that occur WHILE frozen. The
        # observation that just confirmed the freeze itself does not count.
        if self.state == FreezeState.FROZEN and not transitioned:
            drift_transitioned, drift_reason = self._evaluate_drift_when_frozen(ev)
            transitioned = transitioned or drift_transitioned
            if drift_transitioned:
                reason = drift_reason
        return self._decision(transitioned=transitioned, reason=reason)

    @staticmethod
    def _append_step(win: _Window, lens: list[float], ev: FreezeEvidence) -> None:
        if not lens:
            return
        win.steps.append(
            _StepStat(
                accept_lens=lens,
                response_tokens=int(ev.response_tokens or 0),
                generation_seconds=float(ev.generation_seconds or 0.0),
                low_fraction=ev.low_fraction,
            )
        )

    def _handle_publish(self, ev: FreezeEvidence) -> FreezeDecision:
        """Stage a successful train+publish; advance the authoritative version.

        The window is not finalized here -- the step whose rollout used the old
        version is logged *after* this publish call (update_actor precedes
        Tracking.log in the trainer). Finalization happens on the next rollout
        observation: old-version evidence from the publishing step closes the
        window directly; otherwise the first evidence tagged with the new
        version closes it before being ingested.
        """
        new_version = int(ev.drafter_version)
        if new_version <= self.drafter_version:
            # Duplicate / out-of-order publish event; do not move the clock.
            return self._decision(transitioned=False, reason=self._last_reason)

        cost = (
            float(ev.update_cost_seconds)
            if ev.update_cost_seconds is not None and ev.update_cost_seconds >= 0
            else None
        )
        if cost is not None:
            self._costs.append(cost)
            cap = max(self.cfg.cost_window_updates, 1) * 4
            if len(self._costs) > cap:
                self._costs = self._costs[-cap:]
            self.cumulative_cost_seconds += cost

        self._ingest_probe(ev.probe, new_version)

        self.drafter_version = new_version
        # An unresolved staged version (publish without any intervening new
        # rollout, unusual): flush it before staging the next one.
        if self._pending_next_version is not None and new_version > self._pending_next_version:
            self._advance_version(
                self._pending_next_version, self._pending_cost, ev
            )
        self._pending_next_version = new_version
        self._pending_cost = cost
        return self._decision(transitioned=False, reason=self._last_reason)

    def _advance_version(
        self, new_version: int, cost: Optional[float], ev: FreezeEvidence
    ) -> tuple[bool, str]:
        """Close the served window, open ``new_version``, run the state machine."""
        closed = self._finalize_window(
            self._open_version, ev.global_step, cost=cost
        )
        self._open_version = new_version
        self._windows[new_version] = _Window(
            new_version, opened_step=ev.global_step
        )

        if self.state in (FreezeState.FROZEN, FreezeState.RECOVERING):
            # A real update after freeze/recovery completes the recovery once a
            # valid window has been observed.
            if closed is not None and closed.valid:
                transitioned = self._transition(
                    FreezeState.LEARNING, "learning", ev, closed=closed
                )
                self._candidate_streak = 0
                self._resume_streak = 0
                self._missing_streak = 0
                self._anchor = None
                self._anchor_low_fraction = None
                return transitioned, "learning"
            return False, self._last_reason

        # Calibrating / learning / candidate paths: evaluate on the new gain.
        return self._evaluate_plateau(ev, closed, new_version)

    # ------------------------------------------------------------------ #
    # Window finalization, gains, noise
    # ------------------------------------------------------------------ #
    def _window_valid(self, win: _Window) -> bool:
        n_requests = sum(len(s.accept_lens) for s in win.steps)
        return (
            n_requests >= self.cfg.min_requests_per_version
            and len(win.steps) >= self.cfg.min_rollout_steps_per_version
        )

    def _speed(self, tokens: int, seconds: float) -> float:
        return float(tokens) / seconds if seconds and seconds > 0 else 0.0

    def _finalize_window(
        self, version: int, global_step: int, *, cost: Optional[float]
    ) -> Optional[VersionEvidence]:
        win = self._windows.get(version)
        if win is None:
            return None
        lens = win.pooled_lens()
        q = _plain_quality(lens, self.cfg.trim_fraction, self.cfg.hard_quantile)
        tokens, seconds = win.total_tokens(), win.total_seconds()
        valid = self._window_valid(win)
        evidence = VersionEvidence(
            version=version,
            global_step=global_step,
            n_requests=int(q["n"]),
            n_rollout_steps=len(win.steps),
            valid=valid,
            q_all=q["q_all"],
            q_hard=q["q_hard"],
            q_speed=self._speed(tokens, seconds),
            plain_mean=q["mean"],
            median=q["median"],
            response_tokens=tokens,
            generation_seconds=seconds,
            low_fraction=win.latest_low_fraction(),
            cost_seconds=cost,
        )

        prev = self.versions[-1] if self.versions else None
        if valid and prev is not None and prev.valid and prev.version == version - 1:
            evidence.g_all = self._rel(evidence.q_all, prev.q_all)
            evidence.g_hard = self._rel(evidence.q_hard, prev.q_hard)
            evidence.g_speed = self._rel(evidence.q_speed, prev.q_speed)
            self._gains.append(
                {
                    "version": version,
                    "g_all": evidence.g_all,
                    "g_hard": evidence.g_hard,
                    "g_speed": evidence.g_speed,
                    "cost": cost,
                }
            )
            self._consecutive_valid_gains += 1
            self.valid_update_count += 1
            self._record_growth(prev.version, win.version)
        elif not valid or prev is None or not prev.valid:
            # Break adjacency: cannot attribute a gain across an invalid gap.
            self._consecutive_valid_gains = 0

        if valid:
            self._record_split_window_noise(win)

        self.versions.append(evidence)
        # Bound summaries; gains needed for plateau live in self._gains.
        if len(self.versions) > max(self.cfg.window_updates + 2, 8) * 4:
            self.versions = self.versions[-self.cfg.window_updates * 8 :]
        self._prune_window_steps()
        return evidence

    @staticmethod
    def _rel(new: float, old: float) -> Optional[float]:
        denom = max(abs(old), _EPS)
        return (new - old) / denom

    def _record_growth(self, prev_version: int, cur_version: int) -> None:
        """Latch 'significant positive gain ever observed' via gain LCB (9.3)."""
        versions = [prev_version, cur_version]
        steps_map = self._steps_for_bootstrap(versions)
        if not self._bootstrap_available(steps_map):
            return
        lcb_all = self._gain_bound(versions, "q_all", upper=False)
        lcb_hard = self._gain_bound(versions, "q_hard", upper=False)
        if lcb_all is not None and lcb_all > self.cfg.growth_epsilon:
            self._growth_seen_all = True
        if lcb_hard is not None and lcb_hard > self.cfg.growth_epsilon:
            self._growth_seen_hard = True

    def _record_split_window_noise(self, win: _Window) -> None:
        """Same-version pseudo gain between two equivalent sub-windows (7.3).

        Alternating rollout steps form the two sub-windows (even/odd), which
        balances slow within-version drift better than a first/second split.
        Requires >= 2 observed rollout steps.
        """
        steps = win.steps
        if len(steps) < 2:
            return
        group_a = [s for i, s in enumerate(steps) if i % 2 == 0]
        group_b = [s for i, s in enumerate(steps) if i % 2 == 1]
        if not group_a or not group_b:
            return
        lens_a = [v for s in group_a for v in s.accept_lens]
        lens_b = [v for s in group_b for v in s.accept_lens]
        if not lens_a or not lens_b:
            return
        for store, value_a, value_b in (
            (
                self._pseudo_noise_all,
                trimmed_mean(lens_a, self.cfg.trim_fraction),
                trimmed_mean(lens_b, self.cfg.trim_fraction),
            ),
            (
                self._pseudo_noise_hard,
                bottom_tail_mean(lens_a, self.cfg.hard_quantile),
                bottom_tail_mean(lens_b, self.cfg.hard_quantile),
            ),
        ):
            pseudo = self._rel(value_b, value_a)
            if pseudo is not None:
                store.append(abs(pseudo))

    def _noise_floor(self, samples: Sequence[float]) -> Optional[float]:
        if len(samples) < max(1, self.cfg.noise_min_pseudo_samples):
            return None
        return _quantile(sorted(samples), self.cfg.noise_quantile)

    # ------------------------------------------------------------------ #
    # Plateau / freeze evaluation (RFC 9)
    # ------------------------------------------------------------------ #
    def _recent_gain_versions(self) -> list[int]:
        """Versions participating in the last W gains (W+1 windows)."""
        recent = self._gains[-self.cfg.window_updates :]
        if len(recent) < self.cfg.window_updates:
            return []
        last_version = int(recent[-1]["version"])
        first_version = last_version - self.cfg.window_updates
        return list(range(first_version, last_version + 1))

    def _median_gain(self, key: str) -> Optional[float]:
        recent = self._gains[-self.cfg.window_updates :]
        vals = [g.get(key) for g in recent]
        if len(vals) < self.cfg.window_updates or any(v is None for v in vals):
            return None
        return statistics.median(float(v) for v in vals)  # type: ignore[arg-type]

    def _evaluate_plateau(
        self,
        ev: FreezeEvidence,
        closed: Optional[VersionEvidence],
        new_version: int,
    ) -> tuple[bool, str]:
        # Need W consecutive, adjacent valid gains.
        if (
            closed is None
            or not closed.valid
            or self._consecutive_valid_gains < self.cfg.window_updates
        ):
            if self.state != FreezeState.CALIBRATING:
                self._candidate_streak = 0
            return False, "insufficient_evidence"

        versions = self._recent_gain_versions()
        if len(versions) < self.cfg.window_updates + 1:
            return False, "insufficient_evidence"

        learned = False
        if self.state == FreezeState.CALIBRATING:
            learned = self._transition(
                FreezeState.LEARNING, "learning", ev, closed=closed
            )

        point_all = self._median_gain("g_all")
        point_hard = self._median_gain("g_hard")
        point_speed = self._median_gain("g_speed")
        ucb_all = self._aggregate_gain_bound(versions, "q_all", upper=True)
        ucb_hard = self._aggregate_gain_bound(versions, "q_hard", upper=True)
        ucb_speed = self._aggregate_gain_bound(versions, "q_speed", upper=True)

        noise_all = self._noise_floor(self._pseudo_noise_all)
        noise_hard = self._noise_floor(self._pseudo_noise_hard)
        eps_all = max(
            self.cfg.practical_gain_all, noise_all if noise_all is not None else 0.0
        )
        eps_hard = max(
            self.cfg.practical_gain_hard, noise_hard if noise_hard is not None else 0.0
        )

        drop_all_b, drop_hard_b = self._quality_drawdown_bounds(versions)

        # Disabled economics is telemetry-only. Once explicitly enabled, the
        # gate fails open (keeps training) until a *complete, fresh* fixed
        # paired probe covering this advance predicts the next update's ROI
        # (RFC sec. 8.3); the online quality evidence above never substitutes.
        probe_stats = (
            self._fresh_probe_stats(new_version)
            if self.cfg.economics_enabled
            else None
        )
        probe_roi = self._probe_roi(probe_stats) if probe_stats else None
        roi_ucb = probe_roi["roi_ucb95"] if probe_roi is not None else None
        economics_missing = self.cfg.economics_enabled and probe_roi is None
        economic_ok = True
        if self.cfg.economics_enabled:
            economic_ok = (
                probe_roi is not None and roi_ucb <= self.cfg.cost_margin
            )

        gain_all_flat = ucb_all is not None and ucb_all <= eps_all
        gain_hard_flat = ucb_hard is not None and ucb_hard <= eps_hard
        stable_all = drop_all_b is not None and drop_all_b <= self.cfg.freeze_guard_drop_all
        stable_hard = (
            drop_hard_b is not None and drop_hard_b <= self.cfg.freeze_guard_drop_hard
        )

        growth_seen = self._growth_seen_all or self._growth_seen_hard

        normal_candidate = (
            growth_seen
            and gain_all_flat
            and gain_hard_flat
            and stable_all
            and stable_hard
            and economic_ok
        )
        no_response_candidate = (
            (not growth_seen)
            and gain_all_flat
            and gain_hard_flat
            and economic_ok
        )

        candidate = normal_candidate or no_response_candidate
        kind = "plateau" if normal_candidate else (
            "no_response" if no_response_candidate else None
        )

        # Choose the most informative blocking reason when not a candidate.
        if not candidate:
            self._candidate_streak = 0
            self._candidate_kind = None
            if not gain_all_flat:
                reason = "gain_above_threshold"
            elif not gain_hard_flat:
                reason = "hard_gain_above_threshold"
            elif not (stable_all and stable_hard):
                reason = "quality_regression"
            elif economics_missing:
                reason = "insufficient_economics"
            elif not economic_ok:
                reason = "economic_value_positive"
            else:
                reason = "learning"
            if self.state in (
                FreezeState.PLATEAU_CANDIDATE,
                FreezeState.NO_RESPONSE_CANDIDATE,
            ):
                moved = self._transition(
                    FreezeState.LEARNING, reason, ev, closed=closed
                )
            else:
                self.state = FreezeState.LEARNING
                moved = False
            return learned or moved, reason

        # A confirming candidate update.
        if self._candidate_kind not in (None, kind):
            # Switched candidate path: restart the streak rather than mixing.
            self._candidate_streak = 1
        else:
            self._candidate_streak += 1
        self._candidate_kind = kind

        if self._candidate_streak < self.cfg.patience_updates:
            target = (
                FreezeState.PLATEAU_CANDIDATE
                if kind == "plateau"
                else FreezeState.NO_RESPONSE_CANDIDATE
            )
            reason = (
                "plateau_candidate" if kind == "plateau" else "no_response_candidate"
            )
            moved = self._transition(target, reason, ev, closed=closed)
            return learned or moved, reason

        # Confirmed -> freeze.
        self.state = FreezeState.FROZEN
        self._anchor = closed
        self._anchor_low_fraction = closed.low_fraction if closed else None  # type: ignore[union-attr]
        self._resume_streak = 0
        self._missing_streak = 0
        self._frozen_opportunities = 0
        confirm_reason = (
            "plateau_confirmed" if kind == "plateau" else "no_response_confirmed"
        )
        self._emit_transition(
            FreezeState.FROZEN,
            confirm_reason,
            ev,
            extra={
                "gain_all_ucb95": ucb_all,
                "gain_hard_ucb95": ucb_hard,
                "gain_speed_ucb95": ucb_speed,
                "roi_next": probe_roi["roi"] if probe_roi else None,
                "roi_next_ucb95": roi_ucb,
                "benefit_next_seconds": (
                    probe_roi["benefit"] if probe_roi else None
                ),
                "benefit_next_ucb95_seconds": (
                    probe_roi["benefit_ucb95"] if probe_roi else None
                ),
                "probe_version_after": (
                    probe_stats["version_after"] if probe_stats else None
                ),
                "probe_timing_pairs": (
                    probe_stats["timing_pairs"] if probe_stats else None
                ),
            },
        )
        return True, confirm_reason

    def _quality_drawdown_bounds(
        self, versions: Sequence[int]
    ) -> tuple[Optional[float], Optional[float]]:
        """UCB95 of (best_recent - current)/best_recent for all and hard."""
        evid = {v.version: v for v in self.versions if v.version in versions}
        valids = [evid[v] for v in versions if v in evid and evid[v].valid]
        if len(valids) < 2:
            return None, None
        current = valids[-1]
        best = max(valids[:-1] if len(valids) > 1 else valids, key=lambda e: e.q_all)
        if best.version == current.version or best.q_all <= 0:
            return 0.0, 0.0
        pair = [best.version, current.version]
        steps_map = self._steps_for_bootstrap(pair)
        if not self._bootstrap_available(steps_map):
            return None, None
        drop_all = self._drop_bound(steps_map, "q_all", upper=True)
        drop_hard = self._drop_bound(steps_map, "q_hard", upper=True)
        return drop_all, drop_hard

    # ------------------------------------------------------------------ #
    # Drift / resume when frozen (RFC 10)
    # ------------------------------------------------------------------ #
    def _current_frozen_window(self) -> _Window:
        return self._windows.setdefault(
            self._open_version, _Window(self._open_version, opened_step=-1)
        )

    def _evaluate_drift_when_frozen(
        self, ev: FreezeEvidence
    ) -> tuple[bool, str]:
        if not ev.opportunity_this_step:
            return False, "frozen_stable"
        self._frozen_opportunities += 1

        # Credit the avoided update cost against the saved-time estimate.
        avoided = self._robust_update_cost()
        if avoided is not None:
            self.cumulative_saved_seconds_estimate += avoided

        # Optional safety valve: periodic forced probe (monitoring only).
        should_probe = (
            self.cfg.probe_enabled
            and self._frozen_opportunities % max(1, self.cfg.max_frozen_opportunities)
            == 0
        )

        win = self._current_frozen_window()
        if not self._window_valid(win) or self._anchor is None:
            self._missing_streak += 1
            if self._missing_streak >= self.cfg.missing_evidence_opportunities:
                self._transition(
                    FreezeState.RECOVERING,
                    "resume_missing_observability",
                    ev,
                )
                return True, "resume_missing_observability"
            return False, "frozen_stable"
        self._missing_streak = 0

        steps_map = {
            self._anchor.version: self._windows.get(self._anchor.version),
            self._open_version: win,
        }
        steps_map = {k: v for k, v in steps_map.items() if v is not None}
        if not self._bootstrap_available(steps_map) or self._anchor.version == self._open_version:
            # Anchor window steps were pruned: cannot test drift, fail open.
            if self._anchor.version not in self._windows:
                self._transition(
                    FreezeState.RECOVERING,
                    "resume_missing_observability",
                    ev,
                )
                return True, "resume_missing_observability"
            return False, "frozen_stable"

        d_all = self._drift_bound(steps_map, "q_all", upper=False)
        d_hard = self._drift_bound(steps_map, "q_hard", upper=False)

        low_fraction_now = win.latest_low_fraction()
        accelerated = self._low_fraction_accelerated(low_fraction_now)
        required = max(
            1,
            self.cfg.resume_patience_opportunities
            - (1 if accelerated else 0),
        )

        drift_all = d_all is not None and d_all > self.cfg.resume_drop_all
        drift_hard = d_hard is not None and d_hard > self.cfg.resume_drop_hard
        hard_noise = (
            d_hard is not None and d_hard > self.cfg.practical_gain_hard
        )

        if accelerated and hard_noise and not drift_all:
            drift_hard = True  # low-fraction tail alarm + real hard regression

        if drift_all or drift_hard:
            self._resume_streak += 1
            if self._resume_streak >= required:
                if drift_all and drift_hard:
                    reason = "resume_quality_drift"
                elif drift_all:
                    reason = "resume_quality_drift"
                else:
                    reason = (
                        "resume_low_fraction_and_hard_drift"
                        if accelerated
                        else "resume_hard_drift"
                    )
                self._transition(FreezeState.RECOVERING, reason, ev)
                return True, reason
            return False, "frozen_stable"

        self._resume_streak = 0
        return False, "frozen_stable"

    def _low_fraction_accelerated(self, low_fraction_now: Optional[float]) -> bool:
        if low_fraction_now is None or self._anchor_low_fraction is None:
            return False
        return (
            low_fraction_now
            > self._anchor_low_fraction + self.cfg.low_fraction_accelerator
        )

    # ------------------------------------------------------------------ #
    # Bootstrap (RFC 7.2): moving-block over rollout steps within a version
    # ------------------------------------------------------------------ #
    def _versions_to_retain_steps(self) -> set[int]:
        retain: set[int] = set()
        retain.add(self._open_version)
        recent = self._gains[-(self.cfg.window_updates + 1) :]
        if recent:
            last = int(recent[-1]["version"])
            for version in range(
                last - self.cfg.window_updates - 1, last + 1
            ):
                retain.add(version)
        if self._anchor is not None:
            retain.add(self._anchor.version)
        return retain

    def _prune_window_steps(self) -> None:
        retain = self._versions_to_retain_steps()
        for version in list(self._windows.keys()):
            if version != self._open_version and version not in retain:
                # Keep a zeroed placeholder? Drop raw steps but keep the window
                # key only if a VersionEvidence summary exists.
                if self.versions and any(
                    e.version == version for e in self.versions
                ):
                    self._windows[version].steps = []
                else:
                    del self._windows[version]

    def _steps_for_bootstrap(self, versions: Sequence[int]) -> dict[int, _Window]:
        return {
            v: self._windows[v]
            for v in versions
            if v in self._windows and self._windows[v].steps
        }

    @staticmethod
    def _bootstrap_available(steps_map: Mapping[int, _Window]) -> bool:
        return bool(steps_map) and all(
            len(win.steps) >= 1 and win.pooled_lens() for win in steps_map.values()
        )

    def _rng_for(self, tag: int) -> random.Random:
        # Deterministic per (decision point): reproducible regardless of how
        # many times observe() is called (RFC 7.2 / 17.4).
        base = (
            self.cfg.bootstrap_seed
            + 1_000_003 * (self.drafter_version + 1)
            + 97 * tag
        )
        return random.Random(base)

    @staticmethod
    def _resample_window(win: _Window, rng: random.Random) -> _Window:
        steps = win.steps
        n = len(steps)
        picked = [steps[rng.randrange(n)] for _ in range(n)]
        return _Window(
            win.version,
            opened_step=win.opened_step,
            steps=picked,  # type: ignore[arg-type]
        )

    def _quality_of(self, win: Optional[_Window], key: str) -> float:
        if win is None:
            return 0.0
        lens = win.pooled_lens()
        if key == "q_all":
            return trimmed_mean(lens, self.cfg.trim_fraction)
        if key == "q_hard":
            return bottom_tail_mean(lens, self.cfg.hard_quantile)
        if key == "q_speed":
            return self._speed(win.total_tokens(), win.total_seconds())
        return 0.0

    def _bootstrap_bounds(
        self,
        versions: Sequence[int],
        statistic,  # callable(dict[int,_Window]) -> Optional[float]
        *,
        tag: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (point, lcb, ucb) of ``statistic`` via block bootstrap."""
        base_map = self._steps_for_bootstrap(versions)
        if not self._bootstrap_available(base_map):
            return None, None, None
        point = statistic(base_map)
        if point is None:
            return None, None, None
        b = max(2, int(self.cfg.bootstrap_samples))
        rng = self._rng_for(tag)
        samples: list[float] = []
        for _ in range(b):
            resampled = {
                version: self._resample_window(win, rng)
                for version, win in base_map.items()
            }
            value = statistic(resampled)
            if value is not None and math.isfinite(value):
                samples.append(value)
        if len(samples) < 2:
            return point, point, point
        ordered = sorted(samples)
        tail = (1.0 - self.cfg.confidence_level) / 2.0
        return point, _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)

    def _aggregate_gain_bound(
        self, versions: Sequence[int], key: str, *, upper: bool
    ) -> Optional[float]:
        """Bootstrap UCB/LCB of the *median* relative gain over W transitions."""
        w = self.cfg.window_updates

        def statistic(win_map: dict[int, _Window]) -> Optional[float]:
            qvals = {v: self._quality_of(win_map.get(v), key) for v in versions}
            gains = []
            for idx in range(1, len(versions)):
                prev_v, cur_v = versions[idx - 1], versions[idx]
                rel = self._rel(qvals[cur_v], qvals[prev_v])
                if rel is not None:
                    gains.append(rel)
            if len(gains) < w:
                return None
            return statistics.median(gains[-w:])

        tag = (1 if upper else 2) * (
            11 if key == "q_all" else 22 if key == "q_hard" else 33
        )
        point, lcb, ucb = self._bootstrap_bounds(versions, statistic, tag=tag)
        return ucb if upper else lcb

    def _gain_bound(self, versions, key, *, upper):
        """Single-transition gain bound used for the growth latch."""

        def statistic(win_map: dict[int, _Window]) -> Optional[float]:
            prev_v, cur_v = versions[0], versions[1]
            return self._rel(
                self._quality_of(win_map.get(cur_v), key),
                self._quality_of(win_map.get(prev_v), key),
            )

        tag = (5 if upper else 6) * (
            11 if key == "q_all" else 22 if key == "q_hard" else 33
        )
        point, lcb, ucb = self._bootstrap_bounds(versions, statistic, tag=tag)
        return ucb if upper else lcb

    def _drop_bound(
        self, steps_map: Mapping[int, _Window], key: str, *, upper: bool
    ) -> Optional[float]:
        versions = list(steps_map.keys())
        anchor_v = min(versions)
        cur_v = max(versions)

        def statistic(win_map: dict[int, _Window]) -> Optional[float]:
            q_anchor = self._quality_of(win_map.get(anchor_v), key)
            q_cur = self._quality_of(win_map.get(cur_v), key)
            if q_anchor <= 0:
                return None
            return (q_anchor - q_cur) / q_anchor

        tag = (7 if upper else 8) * (11 if key == "q_all" else 22)
        point, lcb, ucb = self._bootstrap_bounds(
            [anchor_v, cur_v], statistic, tag=tag
        )
        return ucb if upper else lcb

    def _drift_bound(self, steps_map, key, *, upper):
        return self._drop_bound(steps_map, key, upper=upper)

    # ------------------------------------------------------------------ #
    # Fixed paired probes (RFC 6) and economics gate (RFC 8)
    # ------------------------------------------------------------------ #
    def probe_due(self, after_version: int) -> bool:
        """Whether the trainer should run a probe around this publish.

        Cadence is "every ``probe.interval_updates`` successful publishes"
        (RFC sec. 6.3). Probes are version-aligned: the probe for version
        ``after_version`` runs before/after that exact publish.
        """
        if not self.cfg.probe_enabled:
            return False
        interval = max(1, self.cfg.probe_interval_updates)
        return int(after_version) >= 1 and int(after_version) % interval == 0

    def _ingest_probe(
        self, probe: Optional[ProbeComparison], new_version: int
    ) -> None:
        """Attach a probe returned by the executor to the version it compares."""
        if probe is None:
            return
        if not probe.accept_pairs():
            # No aligned accept evidence at all: do not overwrite the previous
            # usable probe with a broken one.
            return
        tagged_after = probe.version_after
        if tagged_after is not None and int(tagged_after) != int(new_version):
            # Probe/version mismatch (late or duplicated event); ignore rather
            # than let the gate attribute another update's speed delta.
            return
        if tagged_after is None:
            probe = dataclass_replace(probe, version_after=int(new_version))
        stats = self._summarize_probe(probe)
        self._latest_probe = probe
        self._latest_probe_stats = stats
        self.probe_completed += 1
        complete = (
            stats["coverage"] >= self.cfg.probe_min_coverage
            and stats["timing_pairs"] >= self.cfg.probe_min_timing_pairs
        )
        if not complete:
            self.probe_incomplete += 1
        if probe.wall_seconds is not None and probe.wall_seconds > 0:
            self.probe_wall_seconds_total += float(probe.wall_seconds)

    def _paired_bootstrap(
        self,
        rows: Sequence[Any],
        statistic,  # callable(list[rows]) -> Optional[float]
        *,
        tag: int,
        version_key: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Paired bootstrap over request ids (RFC 7.2, priority 1).

        Unlike the online gain CI, requests are i.i.d., so pairs are resampled
        directly (no moving block). Seed derives from the probe's after-version
        and the statistic tag, so the decision is reproducible regardless of
        how many times it is evaluated.
        """
        rows = list(rows)
        n = len(rows)
        if n == 0:
            return None, None, None
        point = statistic(rows)
        if point is None or not math.isfinite(point):
            return None, None, None
        b = max(2, int(self.cfg.bootstrap_samples))
        rng = random.Random(
            self.cfg.bootstrap_seed
            + 1_000_003 * (int(version_key) + 1)
            + 97 * tag
        )
        samples: list[float] = []
        for _ in range(b):
            picked = [rows[rng.randrange(n)] for _ in range(n)]
            value = statistic(picked)
            if value is not None and math.isfinite(value):
                samples.append(value)
        if len(samples) < 2:
            return point, point, point
        ordered = sorted(samples)
        tail = (1.0 - self.cfg.confidence_level) / 2.0
        return point, _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)

    def _summarize_probe(self, probe: ProbeComparison) -> dict[str, Any]:
        """Immutable probe statistics; computed once when the probe arrives."""
        accept_rows = probe.accept_pairs()
        timing_rows = probe.timing_pairs()
        version_key = (
            int(probe.version_after)
            if probe.version_after is not None
            else self.drafter_version
        )

        gain_all_p, gain_all_l, gain_all_u = self._paired_bootstrap(
            accept_rows,
            lambda rows: trimmed_mean([r[3] for r in rows], self.cfg.trim_fraction),
            tag=101,
            version_key=version_key,
        )

        def hard_gain(rows: list[tuple[str, float, float, float]]) -> Optional[float]:
            if not rows:
                return None
            m = max(1, math.ceil(self.cfg.hard_quantile * len(rows)))
            worst = sorted(rows, key=lambda r: r[1])[:m]
            return statistics.fmean(r[3] for r in worst)

        gain_hard_p, gain_hard_l, gain_hard_u = self._paired_bootstrap(
            accept_rows, hard_gain, tag=102, version_key=version_key
        )

        def delta_sec_per_token(
            rows: list[tuple[str, float, float, float, float]],
        ) -> Optional[float]:
            if not rows:
                return None
            deltas = [
                sec_b / max(tok_b, _EPS) - sec_a / max(tok_a, _EPS)
                for _rid, tok_b, sec_b, tok_a, sec_a in rows
            ]
            return trimmed_mean(deltas, self.cfg.trim_fraction)

        delta_p, delta_l, delta_u = self._paired_bootstrap(
            timing_rows, delta_sec_per_token, tag=103, version_key=version_key
        )

        # Aggregate arm sec/token (diagnostics; gate uses paired distribution).
        sec_per_token_before = None
        sec_per_token_after = None
        if timing_rows:
            total_tok_b = sum(r[1] for r in timing_rows)
            total_sec_b = sum(r[2] for r in timing_rows)
            total_tok_a = sum(r[3] for r in timing_rows)
            total_sec_a = sum(r[4] for r in timing_rows)
            sec_per_token_before = (
                total_sec_b / total_tok_b if total_tok_b > 0 else None
            )
            sec_per_token_after = (
                total_sec_a / total_tok_a if total_tok_a > 0 else None
            )

        return {
            "version_before": probe.version_before,
            "version_after": probe.version_after,
            "global_step": probe.global_step,
            "wall_seconds": probe.wall_seconds,
            "request_count": probe.n_requests(),
            "accept_pairs": len(accept_rows),
            "timing_pairs": len(timing_rows),
            "coverage": probe.timing_coverage(),
            "gain_all": gain_all_p,
            "gain_all_lcb95": gain_all_l,
            "gain_all_ucb95": gain_all_u,
            "gain_hard": gain_hard_p,
            "gain_hard_lcb95": gain_hard_l,
            "gain_hard_ucb95": gain_hard_u,
            "delta_sec_per_token": delta_p,
            "delta_lcb95": delta_l,
            "delta_ucb95": delta_u,
            "sec_per_token_before": sec_per_token_before,
            "sec_per_token_after": sec_per_token_after,
        }

    def _fresh_probe_stats(self, new_version: int) -> Optional[dict[str, Any]]:
        """Complete paired-probe evidence fresh enough to gate this advance.

        RFC 8.2 uses the most recent valid probe to predict the next update;
        "most recent" is bounded by the probe interval so a plateau cannot be
        confirmed on stale speed evidence. Everything missing fails closed for
        the gate (the caller keeps training).
        """
        if not self.cfg.economics_enabled:
            return None
        probe = self._latest_probe
        stats = self._latest_probe_stats
        if probe is None or stats is None:
            return None
        after = probe.version_after
        if after is None:
            return None
        age = int(new_version) - int(after)
        if age < 0 or age > max(1, self.cfg.probe_interval_updates):
            return None
        if stats["coverage"] < self.cfg.probe_min_coverage:
            return None
        if stats["timing_pairs"] < self.cfg.probe_min_timing_pairs:
            return None
        if stats.get("delta_ucb95") is None:
            return None
        return stats

    def _probe_roi(
        self, stats: Optional[dict[str, Any]]
    ) -> Optional[dict[str, float]]:
        """B_next/ROI point + UCB from one probe summary and current horizon."""
        if stats is None:
            return None
        horizon = self._estimated_tokens_to_next_opportunity()
        cost = self._robust_update_cost()
        if horizon is None or cost is None or cost <= 0:
            return None
        delta = stats.get("delta_sec_per_token")
        delta_ucb = stats.get("delta_ucb95")
        if delta is None or delta_ucb is None:
            return None
        return {
            "horizon_tokens": float(horizon),
            "update_cost": float(cost),
            "delta_sec_per_token": float(delta),
            "delta_ucb95": float(delta_ucb),
            "benefit": float(horizon) * float(delta),
            "benefit_ucb95": float(horizon) * float(delta_ucb),
            "roi": float(horizon) * float(delta) / float(cost),
            "roi_ucb95": float(horizon) * float(delta_ucb) / float(cost),
        }

    def _economics_gate_ready(self) -> bool:
        if not self.cfg.economics_enabled:
            return False
        stats = self._fresh_probe_stats(self.drafter_version)
        return stats is not None and self._probe_roi(stats) is not None

    # ------------------------------------------------------------------ #
    # Economics (RFC 8). Probe-driven ROI gates; online proxy is reported.
    # ------------------------------------------------------------------ #
    def _robust_update_cost(self) -> Optional[float]:
        costs = self._costs[-self.cfg.cost_window_updates :]
        if not costs:
            return None
        return statistics.median(costs)

    def _estimated_tokens_to_next_opportunity(self) -> Optional[float]:
        """Mean response tokens served per deployed version (one interval)."""
        recent = [e for e in self.versions if e.valid][-self.cfg.cost_window_updates :]
        if not recent:
            return None
        return float(statistics.fmean(e.response_tokens for e in recent))

    def _online_economics_proxy(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Observed speed value per update; telemetry only, never a gate.

        Online windows do not hold policy/prompts fixed, so this is deliberately
        kept separate from the probe-driven economics gate. It still makes the
        units auditable: seconds/token -> seconds/update interval -> ROI.
        """
        horizon_tokens = self._estimated_tokens_to_next_opportunity()
        cost = self._robust_update_cost()
        recent = [
            evidence
            for evidence in self.versions
            if evidence.valid and evidence.q_speed > 0.0
        ]
        deltas = [
            1.0 / before.q_speed - 1.0 / after.q_speed
            for before, after in zip(recent, recent[1:])
            if after.version == before.version + 1
        ][-self.cfg.cost_window_updates :]
        if not deltas:
            return horizon_tokens, None, None, None
        delta_sec_per_token = float(statistics.median(deltas))
        benefit_seconds = (
            horizon_tokens * delta_sec_per_token
            if horizon_tokens is not None
            else None
        )
        roi = (
            benefit_seconds / cost
            if benefit_seconds is not None and cost is not None and cost > 0.0
            else None
        )
        return horizon_tokens, delta_sec_per_token, benefit_seconds, roi

    # ------------------------------------------------------------------ #
    # Transitions / decision assembly / metrics
    # ------------------------------------------------------------------ #
    def _emit_transition(
        self,
        to: FreezeState,
        reason: str,
        ev: FreezeEvidence,
        extra: Optional[Mapping[str, Optional[float]]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "drafter_freeze_transition",
            "to": to.value,
            "reason": reason,
            "global_step": int(ev.global_step),
            "update_opportunity_id": int(self.update_opportunity_id),
            "drafter_version": int(self.drafter_version),
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = round(float(value), 6)
        self._last_transition = payload

    def _transition(
        self,
        to: FreezeState,
        reason: str,
        ev: FreezeEvidence,
        *,
        closed: Optional[VersionEvidence] = None,
        extra: Optional[Mapping[str, Optional[float]]] = None,
    ) -> bool:
        old = self.state
        self._emit_transition(to, reason, ev, extra=extra)
        self.state = to
        self._last_reason = reason
        return old != to

    @property
    def last_transition(self) -> Optional[Mapping[str, Any]]:
        return self._last_transition

    def _live_quality(self) -> dict[str, float]:
        """Quality of the serving (open) window, else of the last closed one."""
        win = self._windows.get(self._open_version)
        if win is not None and win.steps:
            lens = win.pooled_lens()
            q = _plain_quality(lens, self.cfg.trim_fraction, self.cfg.hard_quantile)
            tokens, seconds = win.total_tokens(), win.total_seconds()
            q["q_speed"] = self._speed(tokens, seconds)
            q["low_fraction"] = win.latest_low_fraction()
            return q
        last = self.versions[-1] if self.versions else None
        if last is None:
            return {}
        return {
            "n": float(last.n_requests),
            "q_all": last.q_all,
            "q_hard": last.q_hard,
            "q_speed": last.q_speed,
            "mean": last.plain_mean,
            "median": last.median,
            "low_fraction": last.low_fraction,
        }

    def _q_normalized(self, value: Optional[float]) -> Optional[float]:
        k = self.cfg.spec_verify_tokens
        if k is None or value is None:
            return None
        denom = max(k - 1.0, _EPS)
        return max(0.0, min(1.0, (value - 1.0) / denom))

    def _decision(
        self, *, transitioned: bool, reason: str
    ) -> FreezeDecision:
        self._last_reason = reason
        frozen = self.state == FreezeState.FROZEN
        should_collect = True
        if frozen:
            should_collect = self.cfg.freeze_scope != "hard"
        metrics = self._metrics(reason)
        return FreezeDecision(
            state=self.state,
            should_train=not frozen,
            should_collect=should_collect,
            should_probe=False,
            transitioned=bool(transitioned),
            reason=reason,
            metrics=metrics,
            drafter_version=self.drafter_version,
            update_opportunity_id=self.update_opportunity_id,
            valid_update_count=self.valid_update_count,
        )

    def _metrics(self, reason: str) -> Mapping[str, Optional[float]]:
        m: dict[str, Optional[float]] = {
            "drafter/freeze_state": float(_STATE_CODE.get(self.state, 0.0)),
            "drafter/frozen": float(self.state == FreezeState.FROZEN),
            "drafter/drafter_version": float(self.drafter_version),
            "drafter/update_opportunity_id": float(self.update_opportunity_id),
            "drafter/valid_update_count": float(self.valid_update_count),
            "drafter/plateau_streak_updates": float(self._candidate_streak),
            "drafter/resume_streak_opportunities": float(self._resume_streak),
            "drafter/freeze_reason_code": float(_reason_code(reason)),
            "drafter/consecutive_valid_gains": float(self._consecutive_valid_gains),
            "drafter/frozen_opportunities": float(self._frozen_opportunities),
        }

        # Current live quality. Prefer the open (serving) window; right after a
        # publish it is empty, so fall back to the last closed version to keep
        # the metric continuous.
        open_q = self._live_quality()
        if open_q:
            m["drafter/q_all"] = open_q["q_all"]
            m["drafter/q_hard"] = open_q["q_hard"]
            m["drafter/q_speed"] = open_q["q_speed"]
            m["drafter/q_all_request_count"] = open_q["n"]
            m["drafter/q_hard_request_count"] = open_q["n"]
            m["drafter/request_mean_accept_len"] = open_q["mean"]
            m["drafter/request_accept_len_median"] = open_q["median"]
            m["drafter/q_all_normalized"] = self._q_normalized(open_q["q_all"])
            m["drafter/q_hard_normalized"] = self._q_normalized(open_q["q_hard"])
            if open_q.get("low_fraction") is not None:
                m["drafter/low_fraction"] = open_q["low_fraction"]

        # Latest finalized version gains (point estimates) and last bounds.
        last_closed = self.versions[-1] if self.versions else None
        if last_closed is not None:
            m["drafter/q_all_last_version"] = last_closed.q_all
            m["drafter/q_hard_last_version"] = last_closed.q_hard
            m["drafter/gain_all"] = last_closed.g_all
            m["drafter/gain_hard"] = last_closed.g_hard
            m["drafter/gain_speed"] = last_closed.g_speed

        g_all = self._median_gain("g_all")
        g_hard = self._median_gain("g_hard")
        g_speed = self._median_gain("g_speed")
        m["drafter/gain_all_recent_median"] = g_all
        m["drafter/gain_hard_recent_median"] = g_hard
        m["drafter/gain_speed_recent_median"] = g_speed

        noise_all = self._noise_floor(self._pseudo_noise_all)
        noise_hard = self._noise_floor(self._pseudo_noise_hard)
        m["drafter/noise_all"] = noise_all
        m["drafter/noise_hard"] = noise_hard
        m["drafter/noise_sample_count"] = float(len(self._pseudo_noise_all))
        m["drafter/effective_epsilon_all"] = float(
            max(
                self.cfg.practical_gain_all,
                noise_all if noise_all is not None else 0.0,
            )
        )
        m["drafter/effective_epsilon_hard"] = float(
            max(
                self.cfg.practical_gain_hard,
                noise_hard if noise_hard is not None else 0.0,
            )
        )

        cost = self._robust_update_cost()
        m["drafter/update_cost_seconds"] = cost
        m["drafter/cumulative_cost_seconds"] = self.cumulative_cost_seconds
        m["drafter/cumulative_saved_seconds_estimate"] = (
            self.cumulative_saved_seconds_estimate
        )
        horizon, delta_per_token, benefit, online_roi = self._online_economics_proxy()
        m["drafter/economics_horizon_tokens"] = horizon
        m["drafter/economics_online_delta_sec_per_token"] = delta_per_token
        m["drafter/economics_online_benefit_seconds"] = benefit
        m["drafter/economics_online_roi"] = online_roi
        m["drafter/economics_gate_enabled"] = float(self.cfg.economics_enabled)
        self._emit_probe_metrics(m)
        return m

    def _emit_probe_metrics(self, m: dict[str, Optional[float]]) -> None:
        """Latest fixed paired probe diagnostics + gate-acceptable ROI."""
        stats = self._latest_probe_stats
        m["drafter/probe_enabled"] = float(self.cfg.probe_enabled)
        m["drafter/probe_completed"] = float(self.probe_completed)
        m["drafter/probe_incomplete"] = float(self.probe_incomplete)
        m["drafter/probe_wall_seconds_total"] = (
            self.probe_wall_seconds_total
            if self.probe_wall_seconds_total > 0
            else None
        )
        if stats is None:
            m["drafter/economics_gate_ready"] = 0.0
            return

        after = stats.get("version_after")
        m["drafter/probe_drafter_version"] = (
            float(after) if after is not None else None
        )
        m["drafter/probe_age_updates"] = (
            float(self.drafter_version - int(after))
            if after is not None
            else None
        )
        m["drafter/probe_request_count"] = float(stats["request_count"])
        m["drafter/probe_accept_pair_count"] = float(stats["accept_pairs"])
        m["drafter/probe_timing_pair_count"] = float(stats["timing_pairs"])
        m["drafter/probe_timing_coverage"] = float(stats["coverage"])
        m["drafter/probe_wall_seconds"] = stats.get("wall_seconds")
        m["drafter/probe_gain_all"] = stats.get("gain_all")
        m["drafter/probe_gain_all_ucb95"] = stats.get("gain_all_ucb95")
        m["drafter/probe_gain_all_lcb95"] = stats.get("gain_all_lcb95")
        m["drafter/probe_gain_hard"] = stats.get("gain_hard")
        m["drafter/probe_gain_hard_ucb95"] = stats.get("gain_hard_ucb95")
        m["drafter/probe_gain_hard_lcb95"] = stats.get("gain_hard_lcb95")
        m["drafter/probe_sec_per_token_before"] = stats.get(
            "sec_per_token_before"
        )
        m["drafter/probe_sec_per_token_after"] = stats.get(
            "sec_per_token_after"
        )

        # Gate view: only a probe that is complete AND fresh for the current
        # version contributes ROI metrics (RFC 8.3).
        fresh = self._fresh_probe_stats(self.drafter_version)
        roi = self._probe_roi(fresh) if fresh is not None else None
        m["drafter/economics_gate_ready"] = float(roi is not None)
        if roi is not None:
            m["drafter/probe_delta_sec_per_token"] = roi["delta_sec_per_token"]
            m["drafter/probe_delta_sec_per_token_ucb95"] = roi["delta_ucb95"]
            m["drafter/predicted_benefit_next_seconds"] = roi["benefit"]
            m["drafter/predicted_benefit_next_ucb95_seconds"] = (
                roi["benefit_ucb95"]
            )
            m["drafter/roi_next"] = roi["roi"]
            m["drafter/roi_next_ucb95"] = roi["roi_ucb95"]


_STATE_CODE = {
    FreezeState.CALIBRATING: 0.0,
    FreezeState.LEARNING: 1.0,
    FreezeState.PLATEAU_CANDIDATE: 2.0,
    FreezeState.NO_RESPONSE_CANDIDATE: 3.0,
    FreezeState.FROZEN: 4.0,
    FreezeState.RECOVERING: 5.0,
}

_REASON_CODES = {
    "insufficient_evidence": 0,
    "learning": 1,
    "gain_above_threshold": 2,
    "hard_gain_above_threshold": 3,
    "quality_regression": 4,
    "economic_value_positive": 5,
    "plateau_candidate": 6,
    "plateau_confirmed": 7,
    "no_response_candidate": 8,
    "no_response_confirmed": 9,
    "frozen_stable": 10,
    "resume_quality_drift": 11,
    "resume_hard_drift": 12,
    "resume_low_fraction_and_hard_drift": 13,
    "resume_missing_observability": 14,
    "config_fingerprint_mismatch": 15,
    "insufficient_economics": 16,
}


def _reason_code(reason: str) -> int:
    return _REASON_CODES.get(reason, -1)
