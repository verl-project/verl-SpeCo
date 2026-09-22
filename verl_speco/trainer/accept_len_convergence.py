"""Drafter convergence freeze: detect when drafter training stops reducing rollout time.

The RL policy keeps training, but the speculative-decoding drafter is frozen once
its training no longer shortens rollout time. The decision is driven by a
length-normalized rollout throughput metric (tokens/sec), which removes the
response-length drift confound that pollutes raw ``gen_time`` and raw
``mean_accept_len`` trends.

This module is pure logic (no trainer dependency) so it can be unit-tested in
isolation. See ``drafter_convergence_freeze_design.md`` for the full design and
the empirical calibration on the qwen3-8b dspark run.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Optional

__all__ = [
    "hard_tail_mean",
    "rollout_throughput",
    "relative_slope",
    "bimodal_metrics",
    "ConvergenceTracker",
]


def hard_tail_mean(accept_lens: list[float], quantile: float = 0.10) -> float:
    """Mean of the bottom-``quantile`` fraction of per-request accept lengths.

    Represents the "hard samples" the drafter struggles with (and that PR #52
    prioritizes collecting). Robust to response length (length-coupling is weak,
    r ~= -0.16) and has the largest dynamic range among accept-length aggregates,
    so its plateau is the easiest to detect.
    """
    if not accept_lens:
        return 0.0
    sorted_vals = sorted(accept_lens)
    k = max(1, int(round(quantile * len(sorted_vals))))
    return statistics.fmean(sorted_vals[:k])


def rollout_throughput(count: int, response_length_mean: float, gen_time: float) -> float:
    """Length-normalized rollout speed [tokens/sec].

    ``= total_response_tokens / gen_time = count * response_length_mean / gen_time``.

    This is the objective the drafter is meant to improve. It is immune to the
    policy shortening responses (which drops raw ``gen_time`` without the drafter
    doing anything) because both numerator and denominator scale with length.
    Empirically it plateaus well before raw accept-length metrics, correctly
    flagging that further drafter training stops paying off in rollout speed.
    """
    if gen_time is None or gen_time <= 0:
        return 0.0
    if response_length_mean is None or response_length_mean <= 0 or count <= 0:
        return 0.0
    return count * response_length_mean / gen_time


def relative_slope(values: list[float], window: int) -> Optional[float]:
    """Relative linear-regression slope over the last ``window`` values.

    Returns ``slope / mean`` (per-step, normalized by level) so the threshold is
    comparable across runs and metrics. ``None`` if fewer than ``window`` samples.
    A 30-step regression averages out the +/-3-5% noise dips that cause naive
    "new high" / two-window-mean plateau tests to flicker.
    """
    if window <= 0 or len(values) < window:
        return None
    ys = values[-window:]
    n = len(ys)
    xs = list(range(n))
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    if my == 0:
        return 0.0
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return (num / den) / my


# --------------------------------------------------------------------------- #
# Bimodal accept-length detection ("hard sample" low peak). Pure Python KDE.
#   Mirrors the method documented in accept_len_hard_sample_and_convergence_metric.md
#   (Gaussian KDE -> peak detection -> valley split -> low_fraction), kept free
#   of numpy so the module stays dependency-light and unit-testable anywhere.
# --------------------------------------------------------------------------- #
_GRID_POINTS = 128          # KDE density grid resolution
_PEAK_HEIGHT_MIN = 0.10     # candidate peak height >= 10% of the tallest peak
_PEAK_SPACING_MIN = 0.5     # peaks must be > 0.5 accept-length apart
_LOW_PEAK_FRACTION_MIN = 0.05  # else the distribution is treated as unimodal


def _silverman_bandwidth(values: list[float]) -> float:
    """Silverman's rule-of-thumb KDE bandwidth, robust to a degenerate IQR."""
    n = len(values)
    sd = statistics.pstdev(values)
    if sd <= 0.0:
        return 1.0
    ordered = sorted(values)
    q25 = ordered[int(0.25 * (n - 1))]
    q75 = ordered[int(0.75 * (n - 1))]
    iqr = q75 - q25
    robust_scale = sd if iqr <= 0.0 else min(sd, iqr / 1.349)
    if robust_scale <= 0.0:
        robust_scale = sd
    return 0.9 * robust_scale * n ** (-0.2)


def _kde_density(values: list[float], x_grid: list[float], bandwidth: float) -> list[float]:
    """Gaussian KDE of ``values`` evaluated on ``x_grid`` (pure Python)."""
    n = len(values)
    inv_h = 1.0 / bandwidth
    norm = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    density = []
    for xi in x_grid:
        acc = 0.0
        for v in values:
            z = (xi - v) * inv_h
            acc += math.exp(-0.5 * z * z)
        density.append(norm * acc)
    return density


def _local_maxima(density: list[float]) -> list[int]:
    """Indices of strict local maxima of a 1-D density array."""
    return [
        i
        for i in range(1, len(density) - 1)
        if density[i] > density[i - 1] and density[i] > density[i + 1]
    ]


def _unimodal_result(values: list[float]) -> dict:
    """Safe result for a (practically) unimodal accept-length distribution."""
    mean = statistics.fmean(values) if values else 0.0
    return {
        "low_fraction": 0.0,
        "low_mean": 0.0,
        "high_mean": mean,
        "valley": None,
        "is_bimodal": False,
    }


def bimodal_metrics(accept_lens: list[float]) -> dict:
    """Split the per-request accept-length distribution into a low (hard) peak.

    Returns a dict with:
      - ``low_fraction`` : share of requests whose accept length is at/below the
        KDE valley between the two peaks; ``0.0`` when the distribution is (or
        has become) unimodal.
      - ``low_mean`` / ``high_mean`` : mean accept length split at the valley.
      - ``valley``        : the valley accept-length value, or ``None``.
      - ``is_bimodal``    : whether a significant low peak was detected.

    This is the "distribution-level convergence" signal from the design doc: as
    the drafter learns the hard samples, the low peak shrinks and ``low_fraction``
    decays toward ``0.0`` — which is the freeze trigger for the
    ``bimodal_low_fraction`` gate.
    """
    if not accept_lens:
        return _unimodal_result([])
    values = [float(v) for v in accept_lens]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        # Degenerate single-value distribution -> single peak by construction.
        return _unimodal_result(values)

    x_grid = [lo + (hi - lo) * i / (_GRID_POINTS - 1) for i in range(_GRID_POINTS)]
    bandwidth = _silverman_bandwidth(values)
    density = _kde_density(values, x_grid, bandwidth)

    peaks = _local_maxima(density)
    tall = [p for p in peaks if density[p] >= _PEAK_HEIGHT_MIN * max(density)]
    if len(tall) < 2:
        return _unimodal_result(values)

    low_p, high_p = min(tall), max(tall)
    if x_grid[high_p] - x_grid[low_p] <= _PEAK_SPACING_MIN:
        # Peaks are too close to be a real bimodal split.
        return _unimodal_result(values)

    in_between = density[low_p : high_p + 1]
    valley_idx = low_p + in_between.index(min(in_between))
    valley = x_grid[valley_idx]

    below = [v for v in values if v <= valley]
    low_fraction = len(below) / len(values)
    # A low group that thin is just the tail of the main peak, not a distinct
    # hard sub-population: reclassify as unimodal (and 0 freeze signal).
    if low_fraction < _LOW_PEAK_FRACTION_MIN or low_fraction > 0.95:
        return _unimodal_result(values)

    high_vals = [v for v in values if v > valley]
    return {
        "low_fraction": low_fraction,
        "low_mean": statistics.fmean(below) if below else 0.0,
        "high_mean": statistics.fmean(high_vals) if high_vals else valley,
        "valley": valley,
        "is_bimodal": True,
    }


class ConvergenceTracker:
    """Stateful freeze tracker: TRAINING -> FROZEN, with optional hysteresis.

    Each step call :meth:`update` with the current throughput and hard_tail; it
    returns the metrics dict to report and mutates :attr:`frozen`. The trainer
    gates drafter training on ``frozen`` (one-step lag: step *k* decides from
    data through step *k-1*, which is correct).

    Parameters mirror the ``drafter_convergence_freeze`` config block.
    """

    def __init__(
        self,
        *,
        gate_metric: str = "rollout_throughput",
        window: int = 30,
        slope_eps: float = 0.001,
        patience: int = 10,
        throughput_floor: Optional[float] = None,
        hard_tail_floor: float = 2.5,
        low_fraction_floor: float = 0.05,
        low_fraction_resume: float = 0.10,
        hysteresis: bool = True,
        hysteresis_drop: float = 0.15,
        hysteresis_window: int = 5,
        hysteresis_patience: int = 5,
    ) -> None:
        if gate_metric not in (
            "rollout_throughput",
            "hard_tail",
            "bimodal_low_fraction",
        ):
            raise ValueError(
                "gate_metric must be 'rollout_throughput', 'hard_tail' or "
                f"'bimodal_low_fraction', got {gate_metric!r}"
            )
        self.gate_metric = gate_metric
        self.window = max(1, int(window))
        self.slope_eps = float(slope_eps)
        self.patience = max(1, int(patience))
        self.throughput_floor = (
            None if throughput_floor is None else float(throughput_floor)
        )
        self.hard_tail_floor = float(hard_tail_floor)
        self.low_fraction_floor = float(low_fraction_floor)
        self.low_fraction_resume = float(low_fraction_resume)
        self.hysteresis = bool(hysteresis)
        self.hysteresis_drop = float(hysteresis_drop)
        self.hysteresis_window = max(1, int(hysteresis_window))
        self.hysteresis_patience = max(1, int(hysteresis_patience))

        self.frozen: bool = False
        self.freeze_step: Optional[int] = None
        self.freeze_value: Optional[float] = None
        self._gate_values: deque[float] = deque(maxlen=self.window)
        self._streak: int = 0
        self._hyst_streak: int = 0

    def _gate_value(
        self, throughput: float, hard_tail: float, low_fraction: Optional[float] = None
    ) -> Optional[float]:
        if self.gate_metric == "rollout_throughput":
            return throughput
        if self.gate_metric == "hard_tail":
            return hard_tail
        return low_fraction

    def _floor(self) -> Optional[float]:
        if self.gate_metric == "rollout_throughput":
            return self.throughput_floor
        return self.hard_tail_floor

    def update(
        self,
        throughput: float,
        hard_tail: float,
        step: int,
        low_fraction: Optional[float] = None,
    ) -> dict[str, float]:
        """Feed one step's metrics; return the metrics dict to report.

        Returns ``drafter/rollout_throughput``, ``drafter/hard_tail``,
        ``drafter/gate_rel_slope`` (when available), ``drafter/frozen``, and for
        ``gate_metric="bimodal_low_fraction"`` also ``drafter/low_fraction``.
        """
        gate_value = self._gate_value(throughput, hard_tail, low_fraction)
        self._gate_values.append(gate_value)

        metrics: dict[str, float] = {
            "drafter/rollout_throughput": float(throughput),
            "drafter/hard_tail": float(hard_tail),
            "drafter/frozen": float(self.frozen),
        }

        if self.gate_metric == "bimodal_low_fraction":
            return self._bimodal_step(gate_value, step, metrics)

        if len(self._gate_values) < self.window:
            # Not enough history to estimate a slope yet.
            return metrics

        rel_slope = relative_slope(list(self._gate_values), self.window)
        if rel_slope is not None:
            metrics["drafter/gate_rel_slope"] = rel_slope

        floor = self._floor()
        reached = gate_value >= floor if floor is not None else True
        flat = rel_slope is not None and rel_slope <= self.slope_eps and reached

        if not self.frozen:
            self._streak = self._streak + 1 if flat else 0
            if self._streak >= self.patience:
                self.frozen = True
                self.freeze_step = step
                self.freeze_value = gate_value
        elif self.hysteresis and self.freeze_value is not None:
            # Policy drift: the frozen drafter no longer keeps up with the
            # drifted policy and the gate metric genuinely degrades. We require
            # a *sustained* decline (mean of the last ``hysteresis_window``
            # steps below ``(1-drop) * freeze_value`` for ``hysteresis_patience``
            # consecutive steps) so a single noisy step does not unfreeze.
            recent = (
                statistics.fmean(list(self._gate_values)[-self.hysteresis_window:])
                if len(self._gate_values) >= self.hysteresis_window
                else gate_value
            )
            if recent < (1.0 - self.hysteresis_drop) * self.freeze_value:
                self._hyst_streak += 1
            else:
                self._hyst_streak = 0
            if self._hyst_streak >= self.hysteresis_patience:
                self.frozen = False
                self.freeze_step = None
                self.freeze_value = None
                self._streak = 0
                self._hyst_streak = 0

        metrics["drafter/frozen"] = float(self.frozen)
        return metrics

    def _bimodal_step(
        self,
        gate_value: Optional[float],
        step: int,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """Freeze decision for the ``bimodal_low_fraction`` gate.

        Unlike the throughput/hard_tail gates (which freeze when an *increasing*
        signal plateaus), this gate's signal is the hard-sample ``low_fraction``,
        which *decays to 0* as the drafter converges. There is no slope test: the
        signal is bounded below by 0, so we freeze purely on ``low_fraction``
        staying at/below ``low_fraction_floor`` for ``patience`` consecutive
        steps. Hysteresis unfreezes when the low peak reappears (policy drift),
        i.e. ``low_fraction`` climbing back above ``low_fraction_resume``.
        """
        if gate_value is not None:
            metrics["drafter/low_fraction"] = float(gate_value)
        if len(self._gate_values) >= self.window:
            rel_slope = relative_slope(list(self._gate_values), self.window)
            if rel_slope is not None:
                metrics["drafter/gate_rel_slope"] = rel_slope

        if gate_value is None:
            return metrics
        reached_floor = gate_value <= self.low_fraction_floor

        if not self.frozen:
            self._streak = self._streak + 1 if reached_floor else 0
            if self._streak >= self.patience:
                self.frozen = True
                self.freeze_step = step
                self.freeze_value = gate_value
        elif self.hysteresis:
            # The hard group reappearing (low_fraction back above resume) in a
            # sustained way means the policy drifted and generated new hard
            # samples -> resume drafter training.
            if gate_value >= self.low_fraction_resume:
                self._hyst_streak += 1
            else:
                self._hyst_streak = 0
            if self._hyst_streak >= self.hysteresis_patience:
                self.frozen = False
                self.freeze_step = None
                self.freeze_value = None
                self._streak = 0
                self._hyst_streak = 0

        metrics["drafter/frozen"] = float(self.frozen)
        return metrics
