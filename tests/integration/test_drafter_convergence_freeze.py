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
"""Tests for the shared bimodal accept-length metrics.

``bimodal_metrics`` (pure-Python KDE) feeds the ``low_fraction`` evidence into
the marginal-utility freeze policy. The legacy single-metric tracker and its
tests were removed together with the legacy freeze implementation.
"""
from __future__ import annotations

import math

import pytest

from verl_speco.trainer.accept_len_convergence import bimodal_metrics


# --------------------------------------------------------------------------- #
# Bimodal accept-length detection (hard-sample / distribution convergence)
# --------------------------------------------------------------------------- #
def _gaussian_mixture(low_n: int, high_n: int, low_mu=2.3, high_mu=3.6) -> list[float]:
    """Deterministic two-cluster accept-length sample: low (hard) + high peaks."""
    vals = []
    for i in range(low_n):
        vals.append(low_mu + 0.18 * math.sin(i * 1.3))
    for i in range(high_n):
        vals.append(high_mu + 0.18 * math.cos(i * 1.1))
    return vals


def test_bimodal_metrics_splits_mixture_into_low_and_high() -> None:
    vals = _gaussian_mixture(40, 280)          # 12.5% hard peak
    m = bimodal_metrics(vals)
    assert m["is_bimodal"] is True
    assert m["low_fraction"] == pytest.approx(40 / 320, abs=0.05)
    assert m["valley"] is not None
    assert m["low_mean"] < m["high_mean"]


def test_bimodal_metrics_unimodal_is_zero() -> None:
    # A single cluster around 3.6 -> one peak -> no hard sub-population.
    vals = [3.6 + 0.18 * math.sin(i * 1.1) for i in range(320)]
    m = bimodal_metrics(vals)
    assert m["is_bimodal"] is False
    assert m["low_fraction"] == 0.0


def test_bimodal_metrics_tiny_low_peak_is_unimodal() -> None:
    # ~1.6% hard group (< the 5% fallback) -> reclassified unimodal, zero signal.
    vals = _gaussian_mixture(5, 315)
    m = bimodal_metrics(vals)
    assert m["low_fraction"] == 0.0


def test_bimodal_metrics_edge_cases() -> None:
    assert bimodal_metrics([])["low_fraction"] == 0.0
    assert bimodal_metrics([3.5] * 50)["low_fraction"] == 0.0
    assert bimodal_metrics([3.5])["low_fraction"] == 0.0
