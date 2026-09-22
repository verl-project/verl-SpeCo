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
"""Tests for the drafter convergence freeze metric and gate.

The pure-logic tests (metric functions + ``ConvergenceTracker`` state machine)
run without the trainer dependency stack. The trainer-gate contract tests are
skipped when ``verl_speco.trainer.speco_ray_trainer`` cannot be imported.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from verl_speco.trainer.accept_len_convergence import (
    ConvergenceTracker,
    bimodal_metrics,
    hard_tail_mean,
    relative_slope,
    rollout_throughput,
)

# Heavy import: the trainer needs ray/verl/torch. Guard it so the pure-logic
# tests above still run in minimal environments.
try:
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer
    _HAS_TRAINER = True
except Exception:  # pragma: no cover - environment-dependent
    SpecoRayPPOTrainer = None  # type: ignore[assignment]
    _HAS_TRAINER = False

_trainer_skip = pytest.mark.skipif(
    not _HAS_TRAINER, reason="trainer dependency stack (ray/verl/torch) unavailable"
)


# --------------------------------------------------------------------------- #
# Pure metric functions
# --------------------------------------------------------------------------- #
def test_hard_tail_mean_bottom_quantile() -> None:
    # 20 values 1..20, bottom 10% = 2 values (1, 2) -> mean 1.5
    assert hard_tail_mean([float(v) for v in range(1, 21)], quantile=0.10) == pytest.approx(1.5)


def test_hard_tail_mean_empty_and_small() -> None:
    assert hard_tail_mean([]) == 0.0
    # With 3 values and q=0.10, round(0.3)=0 -> clamped to 1 sample.
    assert hard_tail_mean([5.0, 1.0, 9.0]) == pytest.approx(1.0)


def test_rollout_throughput_basic_and_guards() -> None:
    assert rollout_throughput(320, 6000.0, 120.0) == pytest.approx(16000.0)
    assert rollout_throughput(320, 6000.0, 0.0) == 0.0          # div-by-zero
    assert rollout_throughput(0, 6000.0, 120.0) == 0.0          # empty batch
    assert rollout_throughput(320, 0.0, 120.0) == 0.0           # no length
    assert rollout_throughput(320, None, 120.0) == 0.0          # missing length


def test_relative_slope_rising_flat_and_short() -> None:
    rising = [float(v) for v in range(1, 41)]          # slope +1/step
    assert relative_slope(rising, window=30) > 0.02     # clearly positive
    flat = [10.0] * 30
    assert relative_slope(flat, window=30) == pytest.approx(0.0)
    assert relative_slope([1.0, 2.0], window=30) is None  # not enough samples


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


# --------------------------------------------------------------------------- #
# ConvergenceTracker state machine
# --------------------------------------------------------------------------- #
def _run(tracker: ConvergenceTracker, gate_values, *, throughput=True) -> bool:
    """Feed a synthetic series; return final frozen state."""
    for step, v in enumerate(gate_values, start=1):
        thr = v if throughput else 1.0
        ht = v if not throughput else 1.0
        tracker.update(thr, ht, step)
    return tracker.frozen


def test_rising_series_does_not_freeze() -> None:
    tracker = ConvergenceTracker(window=30, patience=10, throughput_floor=None)
    # Linear rise 1..60: slope stays strongly positive -> never flat -> no freeze.
    assert _run(tracker, [float(v) for v in range(1, 61)]) is False


def test_plateau_freezes_after_window_plus_patience() -> None:
    tracker = ConvergenceTracker(window=30, patience=10, throughput_floor=None)
    # Constant from the start: slope == 0 once window fills -> freeze at step 39
    # (window=30 first yields a slope at step 30; patience=10 reaches at step 39).
    series = [100.0] * 50
    frozen = _run(tracker, series)
    assert frozen is True
    assert tracker.freeze_step == 39           # window(30) + patience(10) - 1
    assert tracker.freeze_value == pytest.approx(100.0)


def test_hysteresis_unfreezes_on_sustained_decline() -> None:
    tracker = ConvergenceTracker(
        window=30, patience=10, hysteresis=True, hysteresis_drop=0.10,
        throughput_floor=None,
    )
    # Rise to a plateau -> freeze, then a sustained large decline -> unfreeze.
    # Decline must outlast hysteresis_window(5) + hysteresis_patience(5) so the
    # smoothed recent mean stays below the threshold long enough.
    series = [100.0] * 45 + [50.0] * 10        # 50 is < 0.9*100 = 90
    assert _run(tracker, series) is False       # unfroze on the sustained decline
    assert tracker.freeze_step is None


def test_hysteresis_does_not_unfreeze_on_noise() -> None:
    tracker = ConvergenceTracker(
        window=30, patience=10, hysteresis=True, hysteresis_drop=0.10,
        throughput_floor=None,
    )
    # Plateau -> freeze, then a small +/-3% wobble (within noise floor) -> stay
    # frozen: the recent mean never crosses the (1-drop) threshold.
    series = [100.0] * 45 + [97.0, 100.0, 98.0, 101.0, 97.0]
    assert _run(tracker, series) is True


def test_hard_tail_gate_floor_blocks_garbage_freeze() -> None:
    # gate_metric=hard_tail, floor=2.5: a stuck-at-1.0 drafter must NOT freeze
    # (the "constant-1 garbage" case from the design doc).
    tracker = ConvergenceTracker(
        gate_metric="hard_tail", window=30, patience=10, hard_tail_floor=2.5,
    )
    assert _run(tracker, [1.0] * 50, throughput=False) is False


def test_throughput_floor_blocks_low_freeze() -> None:
    tracker = ConvergenceTracker(
        gate_metric="rollout_throughput", window=30, patience=10,
        throughput_floor=100.0,
    )
    # Constant throughput of 50 (below floor) -> reached=False -> no freeze.
    assert _run(tracker, [50.0] * 50) is False


def test_update_returns_metrics_dict() -> None:
    tracker = ConvergenceTracker(window=5, patience=2, throughput_floor=None)
    metrics = tracker.update(throughput=16000.0, hard_tail=3.1, step=1)
    assert metrics["drafter/rollout_throughput"] == pytest.approx(16000.0)
    assert metrics["drafter/hard_tail"] == pytest.approx(3.1)
    assert metrics["drafter/frozen"] == 0.0
    # slope only reported once the window fills
    for step in range(2, 6):
        tracker.update(16000.0, 3.1, step)
    metrics = tracker.update(16000.0, 3.1, step=6)
    assert "drafter/gate_rel_slope" in metrics


def test_invalid_gate_metric_raises() -> None:
    with pytest.raises(ValueError):
        ConvergenceTracker(gate_metric="bogus")


def test_bimodal_gate_freezes_on_sustained_low_fraction() -> None:
    tracker = ConvergenceTracker(
        gate_metric="bimodal_low_fraction", patience=5, low_fraction_floor=0.05
    )
    # Bimodal (high low_fraction) -> never reaches the floor -> no freeze.
    for step in range(1, 12):
        tracker.update(1.0, 1.0, step, low_fraction=0.15)
    assert tracker.frozen is False
    # Low peak disappears (low_fraction -> 0) and stays for patience(5) -> freeze.
    for step in range(12, 25):
        tracker.update(1.0, 1.0, step, low_fraction=0.0)
    assert tracker.frozen is True
    assert tracker.freeze_value == pytest.approx(0.0)


def test_bimodal_gate_rejects_single_low_step() -> None:
    tracker = ConvergenceTracker(
        gate_metric="bimodal_low_fraction", patience=5, low_fraction_floor=0.05
    )
    values = [0.0] + [0.13] * 60           # one brief dip, then recovered
    for step, v in enumerate(values, start=1):
        tracker.update(1.0, 1.0, step, low_fraction=v)
    assert tracker.frozen is False


def test_bimodal_gate_hysteresis_unfreezes_on_reappearing_peak() -> None:
    tracker = ConvergenceTracker(
        gate_metric="bimodal_low_fraction",
        patience=5, low_fraction_floor=0.05, low_fraction_resume=0.10,
        hysteresis=True, hysteresis_patience=5,
    )
    for step in range(1, 30):              # floor sustained -> freeze
        tracker.update(1.0, 1.0, step, low_fraction=0.0)
    assert tracker.frozen is True
    for step in range(30, 60):             # hard peak reappears -> unfreeze
        tracker.update(1.0, 1.0, step, low_fraction=0.15)
    assert tracker.frozen is False
    assert tracker.freeze_step is None


def test_bimodal_gate_reports_low_fraction_metric() -> None:
    tracker = ConvergenceTracker(
        gate_metric="bimodal_low_fraction", window=5, patience=2,
        low_fraction_floor=0.05,
    )
    metrics = tracker.update(1.0, 1.0, step=1, low_fraction=0.0)
    assert metrics["drafter/low_fraction"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Trainer gate contract (skipped without the full dependency stack)
# --------------------------------------------------------------------------- #
def _trainer(training_cfg: dict, *, step: int = 1):
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=True,
                    enable_drafter_training=True,
                    training=training_cfg,
                )
            ),
        )
    )
    trainer._pending_drafter_publish_refs = None
    trainer._speco_last_collected_samples = 0
    trainer._ray_get_if_needed = lambda value: value
    return trainer


@_trainer_skip
def test_frozen_flag_short_circuits_drafter_training() -> None:
    """When _speco_drafter_frozen is True, drafter training is skipped (cascade freeze)."""
    trainer = _trainer({"training_interval_steps": 5})
    trainer.global_steps = 5
    trainer._speco_last_collected_samples = 10
    # Without the freeze flag, the interval matches (5%5==0) and collected
    # samples > 0 -> the method attempts drafter training.
    trainer._speco_drafter_frozen = False
    assert trainer._speco_should_attempt_drafter_train_this_step() is True
    # With the freeze flag, it short-circuits at the top regardless of the
    # other conditions (cascade: no drafter train / lm_head sync / publish).
    trainer._speco_drafter_frozen = True
    assert trainer._speco_should_attempt_drafter_train_this_step() is False


@_trainer_skip
def test_init_convergence_tracker_disabled_returns_none() -> None:
    trainer = _trainer({"drafter_convergence_freeze": {"enabled": False}})
    assert trainer._speco_init_convergence_tracker() is None


@_trainer_skip
def test_init_convergence_tracker_enabled_builds_tracker() -> None:
    trainer = _trainer({
        "drafter_convergence_freeze": {
            "enabled": True,
            "gate_metric": "hard_tail",
            "window_steps": 15,
            "patience_steps": 5,
            "hard_tail_floor": 2.5,
        }
    })
    tracker = trainer._speco_init_convergence_tracker()
    assert isinstance(tracker, ConvergenceTracker)
    assert tracker.gate_metric == "hard_tail"
    assert tracker.window == 15
    assert tracker.patience == 5


@_trainer_skip
def test_convergence_metrics_updates_tracker_and_freeze_flag() -> None:
    trainer = _trainer({"drafter_convergence_freeze": {"enabled": True, "window_steps": 5, "patience_steps": 2}})
    trainer._speco_convergence_tracker = trainer._speco_init_convergence_tracker()
    trainer._speco_last_convergence_step = None
    trainer._speco_drafter_frozen = False
    # Simulate 320 rollout requests with accept lengths around 3.5.
    trainer._speco_last_request_accept_len_records = [
        {"mean_accept_len": 3.5} for _ in range(320)
    ]
    data = {"response_length/mean": 6000.0, "timing_s/gen": 120.0}
    # Feed enough constant steps to trigger a plateau freeze.
    for step in range(1, 10):
        trainer.global_steps = step
        metrics = trainer._speco_convergence_metrics(data)
    assert metrics["drafter/rollout_throughput"] == pytest.approx(16000.0)
    assert metrics["drafter/hard_tail"] == pytest.approx(3.5)
    # Constant throughput -> plateau -> frozen after window(5)+patience(2)-1=6 steps.
    assert trainer._speco_drafter_frozen is True
    assert metrics["drafter/frozen"] == 1.0


@_trainer_skip
def test_convergence_metrics_skips_when_no_records() -> None:
    trainer = _trainer({"drafter_convergence_freeze": {"enabled": True}})
    trainer._speco_convergence_tracker = trainer._speco_init_convergence_tracker()
    trainer._speco_last_convergence_step = None
    trainer._speco_drafter_frozen = False
    trainer._speco_last_request_accept_len_records = []
    assert trainer._speco_convergence_metrics({"response_length/mean": 6000.0, "timing_s/gen": 120.0}) == {}


@_trainer_skip
def test_convergence_metrics_bimodal_gate_freezes_on_unimodal() -> None:
    trainer = _trainer({
        "drafter_convergence_freeze": {
            "enabled": True,
            "gate_metric": "bimodal_low_fraction",
            "window_steps": 5,
            "patience_steps": 2,
            "low_fraction_floor": 0.05,
        }
    })
    trainer._speco_convergence_tracker = trainer._speco_init_convergence_tracker()
    trainer._speco_last_convergence_step = None
    trainer._speco_drafter_frozen = False
    # Unimodal accept lengths around 3.5 -> low_fraction 0 -> freeze after patience(2).
    trainer._speco_last_request_accept_len_records = [
        {"mean_accept_len": 3.5 + 0.05 * ((i * 37) % 7)} for i in range(320)
    ]
    data = {"response_length/mean": 6000.0, "timing_s/gen": 120.0}
    for step in range(1, 8):
        trainer.global_steps = step
        metrics = trainer._speco_convergence_metrics(data)
    assert trainer._speco_drafter_frozen is True
    assert metrics["drafter/low_fraction"] == pytest.approx(0.0)
    assert metrics["drafter/is_bimodal"] == 0.0
