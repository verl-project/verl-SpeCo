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


def _probe_trainer(probe):
    t = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    t.global_steps = 1
    t.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            drafter=SimpleNamespace(enable=True, enable_drafter_training=True, training={
                "drafter_convergence_freeze": {
                    "enabled": True, "method": "marginal_utility_v1",
                    "mode": "active", "probe": probe}}))))
    return t


@_trainer_skip
def test_probe_config_normalizes_dictconfig() -> None:
    """Real Hydra config nodes are DictConfig (NOT dict subclasses): without
    OmegaConf conversion the method silently returned {} and probe arms ran on
    executor defaults instead of the user's prompts/token settings."""
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    probe = OmegaConf.create(
        {"prompts_file": "/pool/x.parquet", "prompt_count": 32,
         "batch_size": 4, "max_new_tokens": 512, "indirect": "${prompt_count}"})
    out = _probe_trainer(probe)._speco_freeze_probe_config()
    assert out == {"prompts_file": "/pool/x.parquet", "prompt_count": 32,
                   "batch_size": 4, "max_new_tokens": 512, "indirect": 32}


@_trainer_skip
@pytest.mark.parametrize(("probe", "expected"),
                         [({"prompt_count": 7}, {"prompt_count": 7}),
                          (None, {})])
def test_probe_config_plain_dict_and_default(probe, expected) -> None:
    assert _probe_trainer(probe)._speco_freeze_probe_config() == expected


@_trainer_skip
def test_probe_config_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="probe must resolve to a mapping"):
        _probe_trainer(["not", "a", "mapping"])._speco_freeze_probe_config()


# --------------------------------------------------------------------- #
# Async publish completion-wait attribution (economics C_update)         #
# --------------------------------------------------------------------- #

def _wait_trainer(*, policy):
    t = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    t._speco_freeze_policy = policy
    t._speco_pending_async_publish_version = None
    t._speco_publish_completion_wait = None
    t._pending_drafter_publish_refs = None
    t._speco_last_freeze_decision = None
    t.global_steps = 10
    return t


@_trainer_skip
def test_async_publish_wait_amends_staged_update_cost() -> None:
    """publish_async wait completing AFTER the publish was staged (next
    generation entry / checkpoint save) is folded into THAT update's staged
    cost -- never into a later update's."""
    amends = []

    class _Policy:
        drafter_version = 5  # the publish already staged v5

        def amend_pending_update_cost(self, version, extra):
            amends.append((version, extra))
            return True

    t = _wait_trainer(policy=_Policy())
    t._speco_pending_async_publish_version = 5
    t._pending_drafter_publish_refs = ["ref-1"]
    import time as _time
    t._ray_get_if_needed = lambda refs: _time.sleep(0.01)

    t._speco_wait_pending_drafter_publish_rpc()
    assert len(amends) == 1 and amends[0][0] == 5 and amends[0][1] > 0.0
    assert t._speco_pending_async_publish_version is None  # consumed
    assert t._speco_publish_completion_wait is None        # nothing carried


@_trainer_skip
def test_async_publish_wait_before_observe_carried_into_cost() -> None:
    """Deferred probe path: the completion wait lands BEFORE observe_publish
    (probe after-arm). It is carried into the observed update cost for the
    SAME version, so the economics gate sees train+publish+wait."""
    observed = []

    class _Policy:
        drafter_version = 4  # publish not staged yet: observe stages v5

        def amend_pending_update_cost(self, version, extra):
            return False

        def observe(self, evidence):
            observed.append(evidence)
            return SimpleNamespace(transitioned=False)

    t = _wait_trainer(policy=_Policy())
    t._speco_pending_async_publish_version = 5
    t._pending_drafter_publish_refs = ["ref-1"]
    import time as _time
    t._ray_get_if_needed = lambda refs: _time.sleep(0.01)

    t._speco_wait_pending_drafter_publish_rpc()
    carried = t._speco_publish_completion_wait
    assert carried is not None and carried[0] == 5 and carried[1] > 0.0

    t._speco_freeze_observe_publish(
        {"drafter/published": 1}, update_cost_seconds=10.0)
    assert len(observed) == 1
    evidence = observed[0]
    assert int(evidence.drafter_version) == 5
    assert evidence.update_cost_seconds == pytest.approx(10.0 + carried[1])
    assert t._speco_publish_completion_wait is None


@_trainer_skip
def test_stale_carried_wait_dropped_loudly_never_misattributed(caplog) -> None:
    """A carry whose version does not match the observed publish is dropped
    with a warning -- it must never inflate a different update's cost."""
    import logging
    observed = []

    class _Policy:
        drafter_version = 7  # observe will stage v8

        def observe(self, evidence):
            observed.append(evidence)
            return SimpleNamespace(transitioned=False)

    t = _wait_trainer(policy=_Policy())
    t._speco_publish_completion_wait = (5, 3.0)  # stale: belongs to v5
    with caplog.at_level(logging.WARNING):
        t._speco_freeze_observe_publish(
            {"drafter/published": 1}, update_cost_seconds=10.0)
    assert observed[0].update_cost_seconds == 10.0  # untouched
    assert t._speco_publish_completion_wait is None
    assert "dropping async publish wait" in caplog.text


@_trainer_skip
def test_async_submit_stashes_next_version_and_sync_does_not() -> None:
    """The async submit path stashes clock+1 as the version to attribute the
    completion wait to; the synchronous path resolves immediately and stashes
    nothing."""
    class _Policy:
        drafter_version = 4

    t = _wait_trainer(policy=_Policy())
    calls = []
    t._speco_actor_rollout_method = lambda name: (
        lambda payload, global_steps=None: calls.append(name) or ["ref"])
    t._ray_get_if_needed = lambda refs: None

    t._speco_update_rollout_drafter_weights("p", 10, asynchronous=True)
    assert t._speco_pending_async_publish_version == 5
    assert t._pending_drafter_publish_refs == ["ref"]

    t._speco_pending_async_publish_version = None
    t._pending_drafter_publish_refs = None
    t._speco_update_rollout_drafter_weights("p", 10, asynchronous=False)
    assert t._speco_pending_async_publish_version is None
    assert t._pending_drafter_publish_refs is None  # sync: resolved in place


@_trainer_skip
def test_publish_cost_excludes_foreign_wait_pending() -> None:
    """``drafter_publish_wait_pending`` measures the PREVIOUS async publish's
    completion wait; summing it into this update's cost would double count it
    (the wait funnel already attributes it to the update that caused it)."""
    cost = SpecoRayPPOTrainer._speco_freeze_publish_cost_seconds(
        {"timing_s/drafter_publish_wait_pending": 9.0,
         "timing_s/drafter_publish_fetch_snapshot": 1.0,
         "timing_s/drafter_publish_update_weights": 2.0})
    assert cost == pytest.approx(3.0)
