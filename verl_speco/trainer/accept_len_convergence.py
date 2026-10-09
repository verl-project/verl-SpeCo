"""Bimodal accept-length detection ("hard sample" low peak).

Pure-Python Gaussian KDE used by the marginal-utility freeze policy
(``method=marginal_utility_v1``): as the drafter learns the hard samples, the
low accept-length peak shrinks and ``low_fraction`` decays toward ``0.0``.

Mirrors the method documented in
``accept_len_hard_sample_and_convergence_metric.md`` (Gaussian KDE -> peak
detection -> valley split -> low_fraction), kept free of numpy so the module
stays dependency-light and unit-testable anywhere.
"""

from __future__ import annotations

import math
import statistics

__all__ = [
    "bimodal_metrics",
]

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
    the drafter learns the hard samples, the low peak shrinks and
    ``low_fraction`` decays toward ``0.0``.
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
