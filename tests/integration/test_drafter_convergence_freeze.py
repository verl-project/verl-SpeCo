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
"""Bimodal accept-length metrics (pure-Python KDE feeding ``low_fraction``
into the marginal-utility policy) plus the hidden-state gate contract: the
collection gate follows the single ``_speco_drafter_frozen`` training gate,
so an active freeze stops hidden-state collection while shadow keeps it.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from verl_speco.trainer.accept_len_convergence import bimodal_metrics
from verl_speco.trainer.scheduler import DrafterCollectionSource

try:  # contract test calls a real trainer method; guard the heavy import
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    _HAS_TRAINER = True
except Exception:  # pragma: no cover - environment-dependent
    SpecoRayPPOTrainer = None  # type: ignore[assignment]
    _HAS_TRAINER = False

_trainer_skip = pytest.mark.skipif(
    not _HAS_TRAINER, reason="trainer dependency stack (ray/verl/torch) unavailable")


def _gaussian_mixture(low_n: int, high_n: int) -> list[float]:
    """Deterministic two-cluster sample: low (hard) + high accept peaks."""
    return ([2.3 + 0.18 * math.sin(i * 1.3) for i in range(low_n)]
            + [3.6 + 0.18 * math.cos(i * 1.1) for i in range(high_n)])

def test_bimodal_metrics_splits_mixture_into_low_and_high() -> None:
    m = bimodal_metrics(_gaussian_mixture(40, 280))  # 12.5% hard peak
    assert m["is_bimodal"] is True
    assert m["low_fraction"] == pytest.approx(40 / 320, abs=0.05)
    assert m["valley"] is not None
    assert m["low_mean"] < m["high_mean"]

def test_unimodal_and_tiny_low_peak() -> None:
    m = bimodal_metrics([3.6 + 0.18 * math.sin(i * 1.1) for i in range(320)])
    assert m["is_bimodal"] is False and m["low_fraction"] == 0.0
    m2 = bimodal_metrics(_gaussian_mixture(5, 315))  # 1.6% < 5% fallback
    assert m2["low_fraction"] == 0.0

def test_bimodal_metrics_edge_cases() -> None:
    for vals in ([], [3.5] * 50, [3.5]):
        assert bimodal_metrics(vals)["low_fraction"] == 0.0


def _gate_trainer(*, mode: str, frozen_gate: bool):
    t = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    t.global_steps = 56
    t.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            drafter=SimpleNamespace(enable=True, enable_drafter_training=True, training={
                "collect_hidden_states_from_sgl": True,
                "drafter_convergence_freeze": {
                    "enabled": True, "method": "marginal_utility_v1", "mode": mode}}))))
    t._speco_freeze_policy_active = mode == "active"
    # The single gate driven by observe() in active mode; shadow leaves it
    # False even when the policy's own decision would freeze.
    t._speco_drafter_frozen = frozen_gate
    return t


@_trainer_skip
@pytest.mark.parametrize(
    ("mode", "frozen_gate", "expected"),
    [("active", True, False),    # active freeze: hidden-state collection stops
     ("active", False, True),    # active, not frozen: normal collection
     ("shadow", False, True)])   # shadow never sets the gate: keeps collecting
def test_hidden_state_gate_follows_training_gate(mode, frozen_gate, expected) -> None:
    t = _gate_trainer(mode=mode, frozen_gate=frozen_gate)
    plan = t._speco_plan_drafter_collection(DrafterCollectionSource.SGLANG)
    assert plan.collect is expected
    assert plan.reason == ("collection_enabled" if expected else "drafter_frozen")
