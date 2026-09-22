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
"""Pure-logic tests for the marginal-utility drafter freeze policy.

Runs with the stdlib (``python -m unittest``) so it executes in the minimal
server Python without ray/torch/pytest/numpy. Covers RFC sec. 18.1/18.2:
quality statistics, version-window rules, update-gain attribution, bootstrap
CI direction/determinism, plateau/no-response state machine, drawdown guard,
drift resume, fail-open on missing evidence, state round-trip, fingerprint
mismatch, update-clock (vs global-step) invariance, and invariance to the
relative ordering of publish vs. rollout-log events.

Trainer event ordering modelled here (sync ``publish_async=false``; the
publishing step's rollout used the OLD weights and Tracking.log runs AFTER
update_actor)::

    steps 1..K-1 : rollout evidence tagged with serving version v
    step K       : publish(v+1) event, then rollout evidence tagged with v
    step K+1 ... : rollout evidence tagged with v+1
"""
from __future__ import annotations

import random
import unittest

from verl_speco.trainer.drafter_freeze_policy import (
    DrafterFreezePolicy,
    FreezeEvidence,
    FreezeState,
    ProbeComparison,
    _Window,
    bottom_tail_mean,
    trimmed_mean,
)


def make_config(**overrides):
    cfg = {
        "freeze_scope": "soft",
        "observation": {
            "min_requests_per_version": 50,
            "min_rollout_steps_per_version": 2,
            "trim_fraction": 0.05,
            "hard_quantile": 0.10,
        },
        "gain": {
            "window_updates": 3,
            "patience_updates": 2,
            "confidence_level": 0.95,
            "bootstrap_samples": 100,
            "practical_gain_all": 0.01,
            "practical_gain_hard": 0.01,
            "growth_epsilon": 0.02,
            "freeze_guard_drop_all": 0.05,
            "freeze_guard_drop_hard": 0.08,
        },
        "noise": {"quantile": 0.95, "min_pseudo_samples": 99},
        "economics": {"enabled": False, "cost_margin": 1.0, "cost_window_updates": 3},
        "probe": {"enabled": False, "interval_updates": 3, "seed": 20260915},
        "resume": {
            "drop_all": 0.05,
            "drop_hard": 0.10,
            "patience_opportunities": 2,
            "low_fraction_accelerator": 0.10,
            "missing_evidence_opportunities": 2,
            "max_frozen_opportunities": 12,
        },
    }
    for key, value in overrides.items():
        cfg[key].update(value)
    return cfg


class Simulator:
    """Drives a policy through rollout steps and synchronous publishes.

    ``cycle`` reproduces the real trainer call order (publish event before the
    publishing step's old-version evidence); ``rollout``/``publish`` remain
    available for out-of-order and no-publish scenarios.
    """

    def __init__(self, cfg, seed=0, steps_per_version=4, per_step=60, sd=0.02):
        self.policy = DrafterFreezePolicy(cfg)
        self.rng = random.Random(seed)
        self.steps_per_version = steps_per_version
        self.per_step = per_step
        self.sd = sd
        self.step = 1

    def lens(self, mean, sd=None):
        sd = self.sd if sd is None else sd
        return [max(1.0, self.rng.gauss(mean, sd)) for _ in range(self.per_step)]

    def rollout(self, mean, *, opportunity=True, sd=None, low_fraction=0.2,
                tokens=60000, seconds=5.0, n=None, version=None):
        n = self.steps_per_version if n is None else n
        tag = self.policy.drafter_version if version is None else version
        decision = None
        for k in range(n):
            ev = FreezeEvidence(
                global_step=self.step,
                drafter_version=tag,
                request_accept_lens=self.lens(mean, sd=sd) if n > 0 else [],
                response_tokens=tokens // max(n, 1),
                generation_seconds=seconds / max(n, 1),
                opportunity_this_step=bool(opportunity and k == n - 1),
                low_fraction=low_fraction,
            )
            decision = self.policy.observe(ev)
            self.step += 1
        return decision

    def empty_rollout(self, *, opportunity=True, version=None):
        """An update opportunity with NO usable per-request evidence."""
        tag = self.policy.drafter_version if version is None else version
        ev = FreezeEvidence(
            global_step=self.step,
            drafter_version=tag,
            request_accept_lens=[],
            opportunity_this_step=opportunity,
        )
        decision = self.policy.observe(ev)
        self.step += 1
        return decision

    def publish(self, cost=10.0, *, update_succeeded=True, version=None,
                probe=None):
        new_version = (
            self.policy.drafter_version + 1 if version is None else version
        )
        ev = FreezeEvidence(
            global_step=self.step,
            drafter_version=new_version,
            publish_completed=True,
            update_succeeded=update_succeeded,
            publish_succeeded=update_succeeded,
            update_cost_seconds=cost,
            probe=probe,
        )
        decision = self.policy.observe(ev)
        self.step += 1
        return decision

    def cycle(self, mean, *, low_fraction=0.2, sd=None, cost=10.0):
        """One full served window in the real trainer event order.

        Pre-publish steps (serving v) -> publish(v+1) -> the publishing step's
        own evidence, tagged with the OLD serving version v, which finalizes
        the window. Returns the final decision (the closing one on transition).
        """
        n = self.steps_per_version
        serving = self.policy.drafter_version
        pre_tokens = int(60000 * (n - 1) / n)
        close_tokens = 60000 - pre_tokens
        pre_seconds = 5.0 * (n - 1) / n
        if n > 1:
            self.rollout(
                mean, opportunity=False, sd=sd, low_fraction=low_fraction,
                tokens=pre_tokens, seconds=pre_seconds, n=n - 1,
                version=serving,
            )
        self.publish(cost, version=serving + 1)
        closing = self.rollout(
            mean, opportunity=True, sd=sd, low_fraction=low_fraction,
            tokens=close_tokens, seconds=5.0 - pre_seconds, n=1,
            version=serving,
        )
        # The closing decision post-dates (and supersedes) the publish decision.
        return closing

    def run_versions(self, means, **kwargs):
        """One served window + one publish per mean; returns final decision."""
        decision = None
        for mean in means:
            decision = self.cycle(mean, **kwargs)
        return decision


class TestQualityStatistics(unittest.TestCase):
    def test_trimmed_mean_trims_both_tails(self):
        values = [float(v) for v in range(1, 101)]  # 1..100
        # trim 5% -> drop 5 each side -> mean(6..95)
        self.assertAlmostEqual(trimmed_mean(values, 0.05), sum(range(6, 96)) / 90.0)

    def test_trimmed_mean_empty_and_degenerate(self):
        self.assertEqual(trimmed_mean([]), 0.0)
        self.assertAlmostEqual(trimmed_mean([3.0, 3.0, 3.0]), 3.0)

    def test_bottom_tail_mean_uses_ceil_exact_10pct(self):
        values = [float(v) for v in range(1, 101)]  # ceil(10)=10 worst -> 1..10
        self.assertAlmostEqual(bottom_tail_mean(values, 0.10), 5.5)

    def test_bottom_tail_mean_small_sample_at_least_one(self):
        # ceil(0.10*3)=1 -> the single worst request.
        self.assertAlmostEqual(bottom_tail_mean([5.0, 1.0, 9.0], 0.10), 1.0)


class TestVersionClock(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(make_config())

    def test_opportunity_without_publish_does_not_advance_version_or_gain(self):
        for _ in range(5):
            self.sim.rollout(3.0)  # opportunity flag set, but no publish
        self.assertEqual(self.sim.policy.drafter_version, 0)
        self.assertEqual(self.sim.policy.valid_update_count, 0)
        self.assertEqual(self.sim.policy.state, FreezeState.CALIBRATING)
        self.assertTrue(self.sim.empty_rollout().should_train)

    def test_failed_publish_does_not_increment_version(self):
        self.sim.rollout(3.0)
        # Train attempted but the publish event never arrives (failed publish):
        # version must not move and no gain may be recorded.
        self.assertEqual(self.sim.policy.drafter_version, 0)
        self.sim.rollout(3.1)
        self.assertEqual(self.sim.policy.drafter_version, 0)
        self.assertEqual(self.sim.policy.valid_update_count, 0)

    def test_duplicate_publish_event_does_not_double_advance(self):
        self.sim.rollout(3.0)
        self.sim.publish()
        self.assertEqual(self.sim.policy.drafter_version, 1)
        # Re-send the same (already-seen) version: must be ignored.
        ev = FreezeEvidence(
            global_step=self.sim.step,
            drafter_version=1,
            publish_completed=True,
            update_cost_seconds=10.0,
        )
        self.sim.policy.observe(ev)
        self.assertEqual(self.sim.policy.drafter_version, 1)

    def test_invalid_window_too_few_requests_fails_open(self):
        cfg = make_config()
        sim = Simulator(cfg, per_step=10, steps_per_version=4)  # 40 < min 50
        for _ in range(6):
            sim.cycle(3.0)
        # Invalid windows never form gains; policy keeps training.
        self.assertEqual(sim.policy.valid_update_count, 0)
        self.assertNotEqual(sim.policy.state, FreezeState.FROZEN)
        self.assertTrue(sim.empty_rollout().should_train)


class TestPlateauStateMachine(unittest.TestCase):
    GROW = [3.0, 3.7, 4.4, 4.7]

    def test_growth_then_plateau_freezes_after_patience(self):
        sim = Simulator(make_config())
        decision = None
        frozen_version = None
        means = self.GROW + [4.72, 4.73, 4.735, 4.74]
        for v, mean in enumerate(means, start=1):
            decision = sim.cycle(mean)
            if decision.state == FreezeState.FROZEN:
                frozen_version = v
                break
        self.assertIsNotNone(frozen_version)
        self.assertEqual(decision.reason, "plateau_confirmed")
        self.assertFalse(decision.should_train)
        self.assertTrue(decision.should_collect)  # soft freeze keeps collection
        self.assertEqual(decision.drafter_version, frozen_version)

    def test_hard_freeze_stops_collection(self):
        cfg = make_config()
        cfg["freeze_scope"] = "hard"
        sim = Simulator(cfg)
        decision = None
        for mean in self.GROW + [4.72, 4.73, 4.735, 4.74]:
            decision = sim.cycle(mean)
            if decision.state == FreezeState.FROZEN:
                break
        self.assertEqual(decision.state, FreezeState.FROZEN)
        self.assertFalse(decision.should_collect)

    def test_single_invalid_window_resets_candidate_streak(self):
        # RFC 18.1: a single piece of unusable evidence must interrupt the
        # consecutive-confirmation chain and fail open (keep training).
        sim = Simulator(make_config())
        for mean in self.GROW + [4.7, 4.7]:
            sim.cycle(mean)
        self.assertEqual(sim.policy._candidate_streak, 1)
        self.assertEqual(sim.policy.state, FreezeState.PLATEAU_CANDIDATE)
        # One under-sampled version window (< min_requests) is invalid evidence.
        sim.per_step = 5
        decision = sim.cycle(4.7)
        sim.per_step = 60
        self.assertEqual(sim.policy._candidate_streak, 0)
        self.assertEqual(sim.policy._consecutive_valid_gains, 0)
        self.assertEqual(decision.reason, "insufficient_evidence")
        self.assertNotEqual(decision.state, FreezeState.FROZEN)
        self.assertTrue(decision.should_train)

    def test_single_point_spike_is_absorbed_by_median(self):
        # RFC 18.4 scenario 3 / 18.2: one isolated gain spike inside an
        # established plateau must not be read as "still growing" -- the median
        # over the last W gains absorbs it (no gain_above_threshold, no flicker).
        sim = Simulator(make_config())
        decision = None
        for mean in self.GROW + [4.7, 4.7, 5.2]:
            decision = sim.cycle(mean)
        # The spike window confirms the plateau rather than resetting it.
        self.assertNotEqual(decision.reason, "gain_above_threshold")
        self.assertIn(
            decision.reason, ("plateau_confirmed", "plateau_candidate")
        )
        median_gain = sim.policy._median_gain("g_all")
        self.assertIsNotNone(median_gain)
        self.assertLessEqual(abs(median_gain), 0.01)  # absorbed: below practical eps

    def test_quality_drawdown_blocks_freeze(self):
        sim = Simulator(make_config())
        # Reach an established plateau (recent median gain ~0), then regress
        # the current version well below the recent best (4.72 -> 4.4, ~6.8%).
        for mean in self.GROW + [4.72, 4.72]:
            sim.cycle(mean)
        decision = sim.cycle(4.4)
        self.assertEqual(decision.reason, "quality_regression")
        self.assertEqual(sim.policy._candidate_streak, 0)
        self.assertNotEqual(decision.state, FreezeState.FROZEN)
        self.assertTrue(decision.should_train)

    def test_low_fraction_alone_does_not_freeze_when_growth_continues(self):
        sim = Simulator(make_config())
        # low_fraction is 0 (low peak gone) but Q keeps jumping up.
        decision = sim.run_versions(
            [3.0, 3.8, 4.7, 5.6, 6.6, 7.7], low_fraction=0.0
        )
        self.assertEqual(decision.reason, "gain_above_threshold")
        self.assertNotEqual(decision.state, FreezeState.FROZEN)


class TestNoResponsePath(unittest.TestCase):
    def test_flat_from_start_freezes_via_no_response(self):
        sim = Simulator(make_config())
        decision = None
        frozen_version = None
        for v in range(1, 9):
            decision = sim.cycle(3.0)
            if decision.state == FreezeState.FROZEN:
                frozen_version = v
                break
        self.assertIsNotNone(frozen_version)
        self.assertEqual(decision.reason, "no_response_confirmed")
        self.assertFalse(sim.policy._growth_seen_all)


class TestDriftResume(unittest.TestCase):
    def _frozen_sim(self):
        sim = Simulator(make_config())
        means = [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]
        for mean in means:
            decision = sim.cycle(mean)
            if decision.state == FreezeState.FROZEN:
                return sim, decision
        raise AssertionError("did not freeze")

    def test_quality_drift_resumes_training(self):
        sim, decision = self._frozen_sim()
        self.assertEqual(decision.state, FreezeState.FROZEN)
        # Two degraded frozen windows -> resume patience (2) met.
        sim.rollout(4.0)  # opportunity at end, resume streak -> 1
        self.assertEqual(sim.policy.state, FreezeState.FROZEN)
        decision = sim.rollout(4.0)  # streak -> 2 -> RECOVERING
        self.assertEqual(sim.policy.state, FreezeState.RECOVERING)
        self.assertTrue(decision.should_train)
        self.assertIn(decision.reason, ("resume_quality_drift", "resume_hard_drift"))

    def test_missing_evidence_while_frozen_fails_open(self):
        sim, _ = self._frozen_sim()
        sim.empty_rollout()
        self.assertEqual(sim.policy.state, FreezeState.FROZEN)
        decision = sim.empty_rollout()  # missing streak -> threshold
        self.assertEqual(sim.policy.state, FreezeState.RECOVERING)
        self.assertEqual(decision.reason, "resume_missing_observability")

    def test_recovering_returns_to_learning_after_valid_update(self):
        sim, _ = self._frozen_sim()
        sim.rollout(4.0)
        sim.rollout(4.0)
        self.assertEqual(sim.policy.state, FreezeState.RECOVERING)
        # A fresh successful update + observed window completes recovery.
        decision = sim.cycle(4.05)
        self.assertEqual(sim.policy.state, FreezeState.LEARNING)
        self.assertTrue(decision.should_train)


class TestEventOrderInvariance(unittest.TestCase):
    MEANS = [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]

    def _freeze_with_closing_after_publish(self, steps_per_version, seed=5):
        """Order A (real trainer): all of v's steps but the last, publish(v+1),
        then the publishing step's old-tagged evidence closes the window."""
        sim = Simulator(
            make_config(), seed=seed, steps_per_version=steps_per_version
        )
        for mean in self.MEANS:
            decision = sim.cycle(mean)
            if decision.state == FreezeState.FROZEN:
                return sim, decision
        return sim, decision

    def _freeze_with_close_on_new_version_evidence(self, steps_per_version, seed=5):
        """Order B: every one of v's steps (incl. the opportunity step) is
        logged BEFORE publish(v+1); the window closes on the FIRST evidence
        tagged with v+1. Windows must end up byte-identical to order A."""
        sim = Simulator(
            make_config(), seed=seed, steps_per_version=steps_per_version
        )
        decision = None
        for v, mean in enumerate(self.MEANS):
            if v == 0:
                # Version 0 has no predecessor trigger: all steps up front.
                sim.rollout(mean, opportunity=True, n=steps_per_version)
            else:
                # This version's first step already arrived as the previous
                # iteration's trigger; supply the remaining n-1 observations.
                sim.rollout(mean, opportunity=True, n=steps_per_version - 1)
            sim.publish()  # pending v+1; window v not closed yet
            if v + 1 < len(self.MEANS):
                # First evidence tagged with the new version closes window v.
                decision = sim.rollout(
                    self.MEANS[v + 1], opportunity=False, n=1
                )
                if decision.state == FreezeState.FROZEN:
                    return sim, decision
        return sim, decision

    def test_both_event_orders_freeze_same_version_same_reason(self):
        sim_a, d_a = self._freeze_with_closing_after_publish(4)
        sim_b, d_b = self._freeze_with_close_on_new_version_evidence(4)
        self.assertEqual(d_a.state, FreezeState.FROZEN)
        self.assertEqual(d_b.state, FreezeState.FROZEN)
        self.assertEqual(d_a.reason, d_b.reason)
        self.assertEqual(
            d_a.drafter_version, d_b.drafter_version,
            "freeze clock must not depend on publish/log call ordering",
        )
        self.assertEqual(sim_a.policy.valid_update_count,
                         sim_b.policy.valid_update_count)
        # Per-version quality summaries must match window-for-window.
        self.assertEqual(len(sim_a.policy.versions), len(sim_b.policy.versions))
        for va, vb in zip(sim_a.policy.versions, sim_b.policy.versions):
            self.assertEqual(va.version, vb.version)
            self.assertAlmostEqual(va.q_all, vb.q_all)
            self.assertAlmostEqual(va.q_hard, vb.q_hard)


class TestBootstrap(unittest.TestCase):
    def test_bounds_are_deterministic_and_ordered(self):
        cfg = make_config()
        means = [3.0, 3.7, 4.4, 4.7]
        sim_a = Simulator(cfg, seed=3)
        sim_b = Simulator(cfg, seed=3)
        for mean in means:
            d_a = sim_a.cycle(mean)
            d_b = sim_b.cycle(mean)
        self.assertEqual(d_a.metrics.get("drafter/drafter_version"),
                         d_b.metrics.get("drafter/drafter_version"))
        # Reach into the aggregate bound helper to check LCB <= UCB determinism.
        versions = sim_a.policy._recent_gain_versions()
        ucb = sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=True)
        lcb = sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=False)
        ucb2 = sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=True)
        self.assertIsNotNone(ucb)
        self.assertLessEqual(lcb, ucb)
        self.assertEqual(ucb, ucb2)  # reproducible regardless of call count

    def test_flat_gain_ucb_below_practical_threshold(self):
        cfg = make_config()
        sim = Simulator(cfg, sd=0.005)
        for mean in [3.0, 3.7, 4.4, 4.7]:
            sim.cycle(mean)
        for _ in range(3):
            sim.cycle(4.7)
        versions = sim.policy._recent_gain_versions()
        ucb_all = sim.policy._aggregate_gain_bound(versions, "q_all", upper=True)
        ucb_hard = sim.policy._aggregate_gain_bound(versions, "q_hard", upper=True)
        self.assertLessEqual(ucb_all, cfg["gain"]["practical_gain_all"])
        self.assertLessEqual(ucb_hard, cfg["gain"]["practical_gain_hard"])


class TestStatePersistence(unittest.TestCase):
    def test_state_dict_round_trip_preserves_clock_and_state(self):
        sim = Simulator(make_config())
        for mean in [3.0, 3.7, 4.4, 4.7, 4.72]:
            sim.cycle(mean)
        state = sim.policy.state_dict()
        restored = DrafterFreezePolicy(make_config())
        restored.load_state_dict(state)
        self.assertEqual(restored.state, sim.policy.state)
        self.assertEqual(restored.drafter_version, sim.policy.drafter_version)
        self.assertEqual(restored.valid_update_count, sim.policy.valid_update_count)
        self.assertEqual(restored._candidate_streak, sim.policy._candidate_streak)
        # Continuing on both policies reaches the same freeze: mirror each
        # cycle on the restored policy by copying the closed window's steps,
        # then replaying publish + old-version closing events.
        for mean in [4.73, 4.735]:
            expected = sim.cycle(mean)
            serving = restored.drafter_version
            src = sim.policy._windows.get(serving)
            dst = restored._windows.setdefault(
                serving, _Window(serving, opened_step=-1)
            )
            if src is not None:
                dst.steps = list(src.steps)
            restored.observe(
                FreezeEvidence(
                    global_step=sim.step,
                    drafter_version=serving + 1,
                    publish_completed=True,
                    update_cost_seconds=10.0,
                )
            )
            got = restored.observe(
                FreezeEvidence(
                    global_step=sim.step,
                    drafter_version=serving,
                    request_accept_lens=[],
                    opportunity_this_step=True,
                )
            )
            self.assertEqual(got.state, expected.state)
            if expected.state == FreezeState.FROZEN:
                self.assertEqual(got.reason, expected.reason)
                return
        self.fail("expected freeze after state restore")

    def test_config_fingerprint_mismatch_forces_recalibration(self):
        sim = Simulator(make_config())
        for mean in [3.0, 3.7, 4.4]:
            sim.cycle(mean)
        state = sim.policy.state_dict()
        changed = make_config()
        changed["gain"]["window_updates"] = 4  # alters fingerprint
        restored = DrafterFreezePolicy(changed)
        restored.load_state_dict(state)
        self.assertEqual(restored.state, FreezeState.CALIBRATING)
        self.assertEqual(restored.drafter_version, 0)

    def test_state_dict_is_json_serializable(self):
        import json

        sim = Simulator(make_config())
        sim.cycle(3.0)
        blob = json.dumps(sim.policy.state_dict())
        self.assertIn("drafter_version", blob)


class TestUpdateClockInvariance(unittest.TestCase):
    def test_same_update_sequence_freezes_same_version(self):
        """Stretching global steps per version must not move the freeze version."""
        means = [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]

        def freeze_version(steps_per_version):
            sim = Simulator(
                make_config(), seed=5, steps_per_version=steps_per_version
            )
            for v, mean in enumerate(means, start=1):
                decision = sim.cycle(mean)
                if decision.state == FreezeState.FROZEN:
                    return v
            return None

        self.assertEqual(freeze_version(2), freeze_version(8))


class TestEconomicsAndProbe(unittest.TestCase):
    def test_roi_does_not_gate_when_economics_disabled_or_no_probe(self):
        # economics disabled: no probe -> freezes on quality alone.
        sim = Simulator(make_config())
        decision = None
        for mean in [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]:
            decision = sim.cycle(mean)
            if decision.state == FreezeState.FROZEN:
                break
        self.assertEqual(decision.state, FreezeState.FROZEN)

    def test_paired_probe_relative_gains(self):
        probe = ProbeComparison(
            request_ids=["a", "b", "c"],
            accept_before=[2.0, 4.0, 5.0],
            accept_after=[3.0, 4.0, 6.0],
        )
        gains = probe.paired_relative_gains()
        self.assertAlmostEqual(gains[0], 0.5)
        self.assertAlmostEqual(gains[1], 0.0)
        self.assertAlmostEqual(gains[2], 0.2)


def make_probe(
    version_after,
    *,
    n=64,
    accept_before=3.0,
    accept_after=3.0,
    sec_per_tok_before=0.010,
    sec_per_tok_after=0.010,
    tokens=512,
    timing_holes=0,
    global_step=0,
    wall=12.0,
):
    """A complete (or partially holed) paired probe with flat accept lens."""
    ids = [f"probe-{version_after}-{i}" for i in range(n)]
    ab = [float(accept_before)] * n
    aa = [float(accept_after)] * n
    tb = [float(tokens)] * n
    ta = [float(tokens)] * n
    sb = [float(tokens) * float(sec_per_tok_before)] * n
    sa = [float(tokens) * float(sec_per_tok_after)] * n
    for i in range(timing_holes):
        # Missing after-arm timing for some pairs: excluded from timing stats.
        sa[i] = None
    return ProbeComparison(
        request_ids=ids,
        accept_before=ab,
        accept_after=aa,
        tokens_before=tb,
        seconds_before=sb,
        tokens_after=ta,
        seconds_after=sa,
        version_before=int(version_after) - 1,
        version_after=int(version_after),
        global_step=global_step,
        wall_seconds=wall,
    )


def economics_config(**probe_overrides):
    cfg = make_config()
    cfg["economics"] = {"enabled": True, "cost_margin": 1.0,
                        "cost_window_updates": 3}
    cfg["probe"] = {"enabled": True, "interval_updates": 3, "seed": 20260915}
    cfg["probe"].update(probe_overrides)
    return cfg


class TestPairedProbeBootstrap(unittest.TestCase):
    def test_probe_due_cadence(self):
        policy = DrafterFreezePolicy(economics_config())
        due = [v for v in range(1, 10) if policy.probe_due(v)]
        self.assertEqual(due, [3, 6, 9])
        disabled = DrafterFreezePolicy(make_config())
        self.assertFalse(any(disabled.probe_due(v) for v in range(1, 10)))

    def test_statistics_deterministic_and_pair_ordered(self):
        policy = DrafterFreezePolicy(economics_config())
        probe = make_probe(3, sec_per_tok_before=0.010,
                           sec_per_tok_after=0.012)
        s1 = policy._summarize_probe(probe)
        s2 = policy._summarize_probe(probe)
        self.assertEqual(s1, s2)  # seed derives from (version, tag)
        self.assertEqual(s1["timing_pairs"], 64)
        self.assertAlmostEqual(s1["coverage"], 1.0)
        # after arm slower -> delta (before - after) is negative at every CI level
        self.assertLess(s1["delta_sec_per_token"], 0.0)
        self.assertLess(s1["delta_ucb95"], 0.0)
        # Arms swapped: sign flips, magnitude matches.
        swapped = make_probe(3, sec_per_tok_before=0.012,
                             sec_per_tok_after=0.010)
        s3 = policy._summarize_probe(swapped)
        self.assertAlmostEqual(s3["delta_sec_per_token"],
                               -s1["delta_sec_per_token"], places=12)
        self.assertAlmostEqual(s3["delta_ucb95"], -s1["delta_lcb95"],
                               places=12)
        self.assertGreater(s3["delta_lcb95"], 0.0)

    def test_hard_gain_uses_before_bottom_tail(self):
        # Bottom-decile before requests start at 1.0 and jump to 3.0 after;
        # G_hard must read the big gain while G_all stays near zero.
        n = 100
        policy = DrafterFreezePolicy(economics_config())
        probe = make_probe(3, n=n)
        ab = [1.0] * 10 + [3.0] * 90
        import dataclasses
        probe = dataclasses.replace(probe, accept_before=ab,
                                    accept_after=[3.0] * n)
        stats = policy._summarize_probe(probe)
        self.assertAlmostEqual(stats["gain_hard"], 2.0, places=6)
        # 5% trimming drops 5 of the 10 jumpers: G_all keeps 5 gains of 2.0,
        # i.e. 10/90 -- a small leak, nowhere near the hard-tail reading.
        self.assertAlmostEqual(stats["gain_all"], 10.0 / 90.0, places=6)


class TestEconomicsGate(unittest.TestCase):
    def _flat_once(self, sim, probe=None):
        """One flat-from-start served window; probe rides its publish event."""
        serving = sim.policy.drafter_version
        sim.rollout(3.0, opportunity=False, n=sim.steps_per_version - 1,
                    version=serving)
        sim.publish(version=serving + 1, probe=probe)
        return sim.rollout(3.0, opportunity=True, n=1, version=serving)

    def _flat_versions(self, sim, versions, probes=None):
        """Run flat cycles 1..versions; probes maps publish version -> probe."""
        probes = probes or {}
        decision = None
        for v in range(1, versions + 1):
            decision = self._flat_once(sim, probes.get(v))
        return decision

    def test_no_probe_blocks_freeze_with_insufficient_economics(self):
        sim = Simulator(economics_config())
        reasons = []
        decision = None
        for _ in range(9):
            decision = self._flat_once(sim)
            reasons.append(decision.reason)
        self.assertNotEqual(decision.state, FreezeState.FROZEN)
        self.assertEqual(sim.policy._candidate_streak, 0)
        self.assertEqual(decision.metrics["drafter/economics_gate_ready"], 0.0)
        # At plateau-shaped advances the blocker must be named explicitly.
        self.assertIn("insufficient_economics", reasons)

    def test_complete_non_paying_probe_allows_freeze(self):
        # after arm slightly SLOWER: ROI UCB <= 0 <= margin -> gate passes.
        probes = {
            3: make_probe(3, sec_per_tok_before=0.010,
                          sec_per_tok_after=0.012),
            6: make_probe(6, sec_per_tok_before=0.010,
                          sec_per_tok_after=0.012),
        }
        sim = Simulator(economics_config())
        decision = None
        frozen_at = None
        for v in range(1, 8):
            decision = self._flat_once(sim, probes.get(v))
            if decision.state == FreezeState.FROZEN:
                frozen_at = v
                break
        # Probe at v3 (ingested at the v3 publish) stays fresh (age <=
        # interval=3) through both confirming advances; closing v4 then
        # completes patience, so the freeze lands at iteration 5 / version 5.
        self.assertEqual(frozen_at, 5)
        self.assertEqual(decision.reason, "no_response_confirmed")
        self.assertEqual(decision.metrics["drafter/economics_gate_ready"], 1.0)
        self.assertLessEqual(
            decision.metrics["drafter/roi_next_ucb95"], 1.0
        )
        self.assertEqual(sim.policy.probe_completed, 1)
        self.assertEqual(sim.policy.probe_incomplete, 0)

    def test_positive_roi_probe_blocks_freeze(self):
        # after arm twice as fast per token -> large positive ROI UCB.
        probes = {
            3: make_probe(3, sec_per_tok_before=0.020,
                          sec_per_tok_after=0.010),
            6: make_probe(6, sec_per_tok_before=0.020,
                          sec_per_tok_after=0.010),
        }
        sim = Simulator(economics_config())
        decision = self._flat_versions(sim, 7, probes=probes)
        self.assertNotEqual(decision.state, FreezeState.FROZEN)
        self.assertEqual(decision.reason, "economic_value_positive")
        self.assertGreater(
            decision.metrics["drafter/roi_next_ucb95"], 1.0
        )

    def test_incomplete_probe_fails_closed(self):
        sim = Simulator(economics_config())
        self._flat_versions(
            sim, 3,
            probes={3: make_probe(3, n=20, timing_holes=20)},
        )
        # coverage 0.0 < 0.8 and 0 timing pairs < 16: gate must not be ready.
        self.assertEqual(sim.policy.probe_incomplete, 1)
        self.assertFalse(sim.policy._economics_gate_ready())
        decision = self._flat_versions(sim, 4)
        self.assertNotEqual(decision.state, FreezeState.FROZEN)
        self.assertEqual(decision.metrics["drafter/economics_gate_ready"], 0.0)

    def test_stale_probe_fails_closed(self):
        policy = DrafterFreezePolicy(economics_config())
        policy.observe(FreezeEvidence(
            global_step=1, drafter_version=3, publish_completed=True,
            update_succeeded=True, publish_succeeded=True,
            update_cost_seconds=10.0, probe=make_probe(3),
        ))
        self.assertIsNotNone(policy._fresh_probe_stats(6))  # age == interval
        self.assertIsNone(policy._fresh_probe_stats(7))     # one cycle stale
        self.assertIsNone(policy._fresh_probe_stats(2))     # from the future

    def test_wrong_version_probe_is_ignored(self):
        policy = DrafterFreezePolicy(economics_config())
        policy.observe(FreezeEvidence(
            global_step=1, drafter_version=4, publish_completed=True,
            update_succeeded=True, publish_succeeded=True,
            update_cost_seconds=10.0, probe=make_probe(3),
        ))
        self.assertIsNone(policy._latest_probe)
        self.assertEqual(policy.probe_completed, 0)

    def test_probe_state_round_trip(self):
        sim = Simulator(economics_config())
        self._flat_versions(sim, 3, probes={3: make_probe(3)})
        stats_before = sim.policy._latest_probe_stats
        state = sim.policy.state_dict()
        restored = DrafterFreezePolicy(economics_config())
        restored.load_state_dict(state)
        self.assertEqual(restored._latest_probe_stats, stats_before)
        self.assertEqual(restored.probe_completed, 1)
        self.assertEqual(
            restored._fresh_probe_stats(3),
            sim.policy._fresh_probe_stats(3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
