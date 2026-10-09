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
"""Pure-stdlib tests for the marginal-utility drafter freeze policy (no
ray/torch/numpy needed): run this file directly. Covers quality stats,
version-window rules, plateau/no-response machine, bootstrap bounds, drawdown
guard, drift resume, state round-trip, fingerprint mismatch, update-clock
invariance, paired probes and the economics ROI gate. ``cycle`` models the
real event order: pre-steps(v) -> publish(v+1) -> closing OLD-v evidence.
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

def make_config(**ov):
    cfg = {
        "observation": {"min_requests_per_version": 50, "min_rollout_steps_per_version": 2,
                        "trim_fraction": 0.05, "hard_quantile": 0.10},
        "gain": {"window_updates": 3, "patience_updates": 2, "confidence_level": 0.95,
                 "bootstrap_samples": 100, "practical_gain_all": 0.01, "practical_gain_hard": 0.01,
                 "growth_epsilon": 0.02, "freeze_guard_drop_all": 0.05, "freeze_guard_drop_hard": 0.08},
        "noise": {"quantile": 0.95, "min_pseudo_samples": 99},
        "economics": {"enabled": False, "cost_margin": 1.0, "cost_window_updates": 3},
        "probe": {"enabled": False, "interval_updates": 3, "seed": 20260915},
        "resume": {"drop_all": 0.05, "drop_hard": 0.10, "patience_opportunities": 2,
                   "low_fraction_accelerator": 0.10, "missing_evidence_opportunities": 2, "max_frozen_opportunities": 12},
    }
    for k, v in ov.items():  # sub-dicts merge, top-level scalars replace
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg

class Simulator:
    """Drives rollout steps; ``cycle`` = real order: pre-steps(v) -> publish
    (v+1) -> closing OLD-v evidence finalizes the window."""
    def __init__(self, cfg, seed=0, spv=4, per_step=60, sd=0.02):
        self.policy = DrafterFreezePolicy(cfg)
        self.rng = random.Random(seed)
        self.spv, self.per_step, self.sd, self.step = spv, per_step, sd, 1
    def lens(self, mean, sd=None):
        return [max(1.0, self.rng.gauss(mean, self.sd if sd is None else sd))
                for _ in range(self.per_step)]
    def rollout(self, mean, *, opportunity=True, sd=None, low_fraction=0.2,
                n=None, version=None):
        n = self.spv if n is None else n
        tag = self.policy.drafter_version if version is None else version
        d = None
        for k in range(n):
            d = self.policy.observe(FreezeEvidence(
                global_step=self.step, drafter_version=tag,
                request_accept_lens=self.lens(mean, sd) if n else [],
                response_tokens=60000 // max(n, 1),
                generation_seconds=5.0 / max(n, 1),
                opportunity_this_step=bool(opportunity and k == n - 1),
                low_fraction=low_fraction))
            self.step += 1
        return d
    def empty_rollout(self, *, opportunity=True, version=None):
        tag = self.policy.drafter_version if version is None else version
        d = self.policy.observe(FreezeEvidence(
            global_step=self.step, drafter_version=tag, request_accept_lens=[],
            opportunity_this_step=opportunity))
        self.step += 1
        return d
    def publish(self, cost=10.0, *, ok=True, version=None, probe=None):
        ver = self.policy.drafter_version + 1 if version is None else version
        d = self.policy.observe(FreezeEvidence(
            global_step=self.step, drafter_version=ver, publish_completed=True,
            update_succeeded=ok, publish_succeeded=ok,
            update_cost_seconds=cost, probe=probe))
        self.step += 1
        return d
    def cycle(self, mean, *, low_fraction=0.2, sd=None, cost=10.0):
        n, serving = self.spv, self.policy.drafter_version
        if n > 1:
            self.rollout(mean, opportunity=False, sd=sd, low_fraction=low_fraction,
                         n=n - 1, version=serving)
        self.publish(cost, version=serving + 1)
        return self.rollout(mean, opportunity=True, sd=sd, low_fraction=low_fraction,
                            n=1, version=serving)

class TestQualityStatistics(unittest.TestCase):
    def test_trimmed_mean_and_bottom_tail_mean(self):
        vals = [float(v) for v in range(1, 101)]
        self.assertAlmostEqual(trimmed_mean(vals, 0.05), sum(range(6, 96)) / 90.0)
        self.assertEqual(trimmed_mean([]), 0.0)
        self.assertAlmostEqual(trimmed_mean([3.0] * 3), 3.0)
        self.assertAlmostEqual(bottom_tail_mean(vals, 0.10), 5.5)  # ceil keeps one
        self.assertAlmostEqual(bottom_tail_mean([5.0, 1.0, 9.0], 0.10), 1.0)

class TestVersionClock(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(make_config())
    def test_opportunity_without_publish_keeps_clock_in_calibrating(self):
        for _ in range(5):
            self.sim.rollout(3.0)  # opportunity flag set, but no publish
        p = self.sim.policy
        self.assertEqual((p.drafter_version, p.valid_update_count, p.state),
                         (0, 0, FreezeState.CALIBRATING))
        self.assertTrue(self.sim.empty_rollout().should_train)
    def test_rollout_only_and_duplicate_publish_do_not_advance(self):
        self.sim.rollout(3.0); self.sim.rollout(3.1)  # failed-publish case
        self.assertEqual(self.sim.policy.drafter_version, 0)
        self.sim.publish()
        self.assertEqual(self.sim.policy.drafter_version, 1)
        self.sim.policy.observe(FreezeEvidence(  # re-send seen version: ignored
            global_step=self.sim.step, drafter_version=1, publish_completed=True,
            update_cost_seconds=10.0))
        self.assertEqual(self.sim.policy.drafter_version, 1)
    def test_invalid_window_too_few_requests_fails_open(self):
        sim = Simulator(make_config(), per_step=10)  # 4*10 = 40 < 50
        for _ in range(6):
            sim.cycle(3.0)
        p = sim.policy
        self.assertEqual((p.valid_update_count, p.state == FreezeState.FROZEN),
                         (0, False))
        self.assertTrue(sim.empty_rollout().should_train)

class TestPlateauStateMachine(unittest.TestCase):
    GROW = [3.0, 3.7, 4.4, 4.7]
    def _run_to_freeze(self, cfg=None, means_extra=None):
        sim = Simulator(cfg or make_config())
        means_extra = means_extra or [4.72, 4.73, 4.735, 4.74]
        d = None
        for v, mean in enumerate(self.GROW + means_extra, start=1):
            d = sim.cycle(mean)
            if d.state == FreezeState.FROZEN:
                return sim, d, v
        return sim, d, None
    def test_growth_then_plateau_freezes_after_patience(self):
        _, d, fv = self._run_to_freeze()
        self.assertEqual((d.reason, d.should_train, d.drafter_version),
                         ("plateau_confirmed", False, fv))
    def test_single_invalid_window_resets_candidate_streak(self):
        sim = Simulator(make_config())
        for mean in self.GROW + [4.7, 4.7]:
            sim.cycle(mean)
        self.assertEqual((sim.policy._candidate_streak, sim.policy.state),
                         (1, FreezeState.PLATEAU_CANDIDATE))
        sim.per_step = 5  # this window 4*5 = 20 < 50
        d = sim.cycle(4.7)
        self.assertEqual((sim.policy._candidate_streak, sim.policy._consecutive_valid_gains),
                         (0, 0))
        self.assertEqual((d.reason, d.should_train), ("insufficient_evidence", True))
    def test_single_point_spike_is_absorbed_by_median(self):
        sim = Simulator(make_config())
        d = None
        for mean in self.GROW + [4.7, 4.7, 5.2]:
            d = sim.cycle(mean)
        self.assertIn(d.reason, ("plateau_confirmed", "plateau_candidate"))
        mg = sim.policy._median_gain("g_all")
        self.assertIsNotNone(mg)
        self.assertLessEqual(abs(mg), 0.01)  # below practical eps
    def test_quality_drawdown_blocks_freeze(self):
        sim = Simulator(make_config())
        for mean in self.GROW + [4.72, 4.72]:  # plateau established
            sim.cycle(mean)
        d = sim.cycle(4.4)  # regress below recent best 4.72
        self.assertEqual((d.reason, sim.policy._candidate_streak, d.should_train),
                         ("quality_regression", 0, True))
    def test_low_fraction_alone_does_not_freeze_when_growth_continues(self):
        sim = Simulator(make_config())
        d = None
        for mean in [3.0, 3.8, 4.7, 5.6, 6.6, 7.7]:
            d = sim.cycle(mean, low_fraction=0.0)  # low peak gone, Q keeps rising
        self.assertEqual((d.reason, d.state == FreezeState.FROZEN),
                         ("gain_above_threshold", False))

class TestNoResponsePath(unittest.TestCase):
    def test_flat_from_start_freezes_via_no_response(self):
        sim = Simulator(make_config())
        d = None
        for v in range(1, 9):
            d = sim.cycle(3.0)
            if d.state == FreezeState.FROZEN:
                break
        else:
            self.fail("did not freeze")
        self.assertEqual(d.reason, "no_response_confirmed")
        self.assertFalse(sim.policy._growth_seen_all)

class TestDriftResume(unittest.TestCase):
    def _frozen_sim(self):
        sim = Simulator(make_config())
        for mean in [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]:
            d = sim.cycle(mean)
            if d.state == FreezeState.FROZEN:
                return sim, d
        raise AssertionError("did not freeze")
    def test_quality_drift_resumes_training(self):
        sim, d = self._frozen_sim()
        self.assertEqual(d.state, FreezeState.FROZEN)
        self.assertEqual(sim.rollout(4.0).state, FreezeState.FROZEN)  # streak 1
        d = sim.rollout(4.0)                                          # streak 2
        self.assertEqual((sim.policy.state, d.should_train), (FreezeState.RECOVERING, True))
        self.assertIn(d.reason, ("resume_quality_drift", "resume_hard_drift"))
    def test_missing_evidence_resumes_and_recovery_completes(self):
        sim, _ = self._frozen_sim()
        sim.empty_rollout()
        d = sim.empty_rollout()  # missing streak hits patience
        self.assertEqual((sim.policy.state, d.reason),
                         (FreezeState.RECOVERING, "resume_missing_observability"))
        d = sim.cycle(4.05)  # fresh update + observed window completes recovery
        self.assertEqual((sim.policy.state, d.should_train), (FreezeState.LEARNING, True))

class TestEventOrderInvariance(unittest.TestCase):
    MEANS = [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]
    def _order_a(self, spv, seed=5):
        """Real trainer: publish(v+1), then old-tagged evidence closes v."""
        sim = Simulator(make_config(), seed=seed, spv=spv)
        d = None
        for mean in self.MEANS:
            d = sim.cycle(mean)
            if d.state == FreezeState.FROZEN:
                return sim, d
        return sim, d
    def _order_b(self, spv, seed=5):
        """v closes on the FIRST evidence tagged v+1; all v steps precede publish."""
        sim = Simulator(make_config(), seed=seed, spv=spv)
        d = None
        for v, mean in enumerate(self.MEANS):
            sim.rollout(mean, opportunity=True, n=spv if v == 0 else spv - 1)
            sim.publish()  # pending v+1; window v not closed yet
            if v + 1 < len(self.MEANS):
                d = sim.rollout(self.MEANS[v + 1], opportunity=False, n=1)
                if d.state == FreezeState.FROZEN:
                    return sim, d
        return sim, d
    def test_both_event_orders_freeze_same_version_same_reason(self):
        sim_a, da = self._order_a(4)
        sim_b, db = self._order_b(4)
        self.assertEqual((da.state, db.state), (FreezeState.FROZEN, FreezeState.FROZEN))
        self.assertEqual((da.reason, da.drafter_version, db.reason, db.drafter_version),
                         (db.reason, db.drafter_version, da.reason, da.drafter_version))
        self.assertEqual((sim_a.policy.valid_update_count, len(sim_a.policy.versions)),
                         (sim_b.policy.valid_update_count, len(sim_b.policy.versions)))
        for va, vb in zip(sim_a.policy.versions, sim_b.policy.versions):
            self.assertEqual(va.version, vb.version)
            self.assertAlmostEqual(va.q_all, vb.q_all)
            self.assertAlmostEqual(va.q_hard, vb.q_hard)

class TestBootstrap(unittest.TestCase):
    def test_bounds_deterministic_ordered_and_flat_ucb_below_threshold(self):
        cfg = make_config()
        sim_a, sim_b = Simulator(cfg, seed=3), Simulator(cfg, seed=3)
        means = [3.0, 3.7, 4.4, 4.7]
        for mean in means:
            da, db = sim_a.cycle(mean), sim_b.cycle(mean)
        self.assertEqual(da.metrics.get("drafter/drafter_version"),
                         db.metrics.get("drafter/drafter_version"))
        versions = sim_a.policy._recent_gain_versions()
        ucb = sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=True)
        lcb = sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=False)
        self.assertIsNotNone(ucb)
        self.assertLessEqual(lcb, ucb)
        self.assertEqual(ucb, sim_a.policy._aggregate_gain_bound(versions, "q_all", upper=True))

        flat = Simulator(make_config(), sd=0.005)
        for mean in means:
            flat.cycle(mean)
        for _ in range(3):
            flat.cycle(4.7)
        recent = flat.policy._recent_gain_versions()
        self.assertLessEqual(flat.policy._aggregate_gain_bound(recent, "q_all", upper=True),
                             cfg["gain"]["practical_gain_all"])
        self.assertLessEqual(flat.policy._aggregate_gain_bound(recent, "q_hard", upper=True),
                             cfg["gain"]["practical_gain_hard"])

class TestStatePersistence(unittest.TestCase):
    def test_state_dict_round_trip_preserves_clock_and_state(self):
        import json

        sim = Simulator(make_config())
        for mean in [3.0, 3.7, 4.4, 4.7, 4.72]:
            sim.cycle(mean)
        self.assertIn("drafter_version", json.dumps(sim.policy.state_dict()))
        restored = DrafterFreezePolicy(make_config())
        restored.load_state_dict(sim.policy.state_dict())
        self.assertEqual((restored.state, restored.drafter_version,
                          restored.valid_update_count, restored._candidate_streak),
                         (sim.policy.state, sim.policy.drafter_version,
                          sim.policy.valid_update_count, sim.policy._candidate_streak))

        # Replay next two versions; the restored policy keeps only recent
        # windows, so mirror each window's steps before publish + closing event.
        for mean in [4.73, 4.735]:
            expected = sim.cycle(mean)
            serving = restored.drafter_version
            src = sim.policy._windows.get(serving)
            dst = restored._windows.setdefault(serving, _Window(serving, opened_step=-1))
            if src is not None:
                dst.steps = list(src.steps)
            restored.observe(FreezeEvidence(
                global_step=sim.step, drafter_version=serving + 1, publish_completed=True,
                update_cost_seconds=10.0))
            got = restored.observe(FreezeEvidence(
                global_step=sim.step, drafter_version=serving, request_accept_lens=[],
                opportunity_this_step=True))
            if got.state == FreezeState.FROZEN:
                self.assertEqual((got.state, got.reason), (expected.state, expected.reason))
                return
        self.fail("expected freeze after state restore")
    def test_config_fingerprint_mismatch_forces_recalibration(self):
        sim = Simulator(make_config())
        for mean in [3.0, 3.7, 4.4]:
            sim.cycle(mean)
        restored = DrafterFreezePolicy(make_config(gain={"window_updates": 4}))
        restored.load_state_dict(sim.policy.state_dict())
        self.assertEqual((restored.state, restored.drafter_version),
                         (FreezeState.CALIBRATING, 0))

class TestUpdateClockInvariance(unittest.TestCase):
    def test_same_update_sequence_freezes_same_version(self):
        """Stretching global steps per version must not move the freeze version."""
        means = [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]

        def freeze_version(spv):
            sim = Simulator(make_config(), seed=5, spv=spv)
            for v, mean in enumerate(means, start=1):
                if sim.cycle(mean).state == FreezeState.FROZEN:
                    return v

        self.assertEqual(freeze_version(2), freeze_version(8))

def make_probe(version_after, *, n=64, accept_before=3.0, accept_after=3.0,
               sec_before=0.010, sec_after=0.010, tokens=512, timing_holes=0):
    """A complete (or partially holed) paired probe with flat accept lens."""
    ids = [f"probe-{version_after}-{i}" for i in range(n)]
    ab, aa = [float(accept_before)] * n, [float(accept_after)] * n
    tb = [float(tokens)] * n
    sb, sa = [float(tokens) * float(sec_before)] * n, [float(tokens) * float(sec_after)] * n
    for i in range(timing_holes):
        sa[i] = None  # missing after-arm timing excluded from stats
    return ProbeComparison(
        request_ids=ids, accept_before=ab, accept_after=aa,
        tokens_before=tb, seconds_before=sb, tokens_after=tb, seconds_after=sa,
        version_before=int(version_after) - 1, version_after=int(version_after))

def economics_config(**probe_overrides):
    cfg = make_config()
    cfg["economics"] = {"enabled": True, "cost_margin": 1.0, "cost_window_updates": 3}
    cfg["probe"] = {"enabled": True, "interval_updates": 3, "seed": 20260915}
    cfg["probe"].update(probe_overrides)
    return cfg

class TestEconomicsAndProbe(unittest.TestCase):
    def test_no_gate_when_disabled_and_paired_relative_gains(self):
        sim = Simulator(make_config())
        d = None
        for mean in [3.0, 3.7, 4.4, 4.7, 4.72, 4.73, 4.735, 4.74]:
            d = sim.cycle(mean)
            if d.state == FreezeState.FROZEN:
                break
        self.assertEqual(d.state, FreezeState.FROZEN)  # no economics gate at all
        gains = ProbeComparison(
            request_ids=["a", "b", "c"], accept_before=[2.0, 4.0, 5.0],
            accept_after=[3.0, 4.0, 6.0]).paired_relative_gains()
        self.assertEqual(tuple(round(g, 6) for g in gains), (0.5, 0.0, 0.2))

class TestPairedProbeBootstrap(unittest.TestCase):
    def test_probe_due_cadence(self):
        p = DrafterFreezePolicy(economics_config())
        self.assertEqual([v for v in range(1, 10) if p.probe_due(v)], [3, 6, 9])
        self.assertFalse(any(DrafterFreezePolicy(make_config()).probe_due(v)
                             for v in range(1, 10)))
    def test_statistics_deterministic_and_pair_ordered(self):
        p = DrafterFreezePolicy(economics_config())
        s1 = p._summarize_probe(make_probe(3, sec_before=0.010, sec_after=0.012))
        self.assertEqual(s1, p._summarize_probe(
            make_probe(3, sec_before=0.010, sec_after=0.012)))
        self.assertEqual((s1["timing_pairs"], s1["coverage"]), (64, 1.0))
        self.assertLess(s1["delta_sec_per_token"], 0.0)  # after arm slower
        self.assertLess(s1["delta_ucb95"], 0.0)
        s3 = p._summarize_probe(make_probe(3, sec_before=0.012, sec_after=0.010))
        self.assertAlmostEqual(s3["delta_sec_per_token"], -s1["delta_sec_per_token"], places=12)
        self.assertAlmostEqual(s3["delta_ucb95"], -s1["delta_lcb95"], places=12)
        self.assertGreater(s3["delta_lcb95"], 0.0)  # arms swap -> sign flips
    def test_hard_gain_uses_before_bottom_tail(self):
        # Bottom-decile before at 1.0 jumps to 3.0: G_hard big, G_all near zero.
        import dataclasses

        probe = dataclasses.replace(
            make_probe(3, n=100), accept_before=[1.0] * 10 + [3.0] * 90,
            accept_after=[3.0] * 100)
        stats = DrafterFreezePolicy(economics_config())._summarize_probe(probe)
        self.assertAlmostEqual(stats["gain_hard"], 2.0, places=6)
        self.assertAlmostEqual(stats["gain_all"], 10.0 / 90.0, places=6)

class TestEconomicsGate(unittest.TestCase):
    def _flat_once(self, sim, probe=None):
        """One flat-from-start served window; probe rides its publish event."""
        serving = sim.policy.drafter_version
        sim.rollout(3.0, opportunity=False, n=sim.spv - 1, version=serving)
        sim.publish(version=serving + 1, probe=probe)
        return sim.rollout(3.0, opportunity=True, n=1, version=serving)
    def _flat_versions(self, sim, versions, probes=None):
        d = None
        for v in range(1, versions + 1):
            d = self._flat_once(sim, (probes or {}).get(v))
        return d
    def test_no_probe_blocks_freeze_with_insufficient_economics(self):
        sim = Simulator(economics_config())
        reasons = []
        d = None
        for _ in range(9):
            d = self._flat_once(sim)
            reasons.append(d.reason)
        self.assertNotEqual(d.state, FreezeState.FROZEN)
        self.assertEqual(sim.policy._candidate_streak, 0)
        self.assertEqual(d.metrics["drafter/economics_gate_ready"], 0.0)
        self.assertIn("insufficient_economics", reasons)  # blocker named
    def test_complete_non_paying_probe_allows_freeze(self):
        # after arm slightly SLOWER: ROI UCB <= 0 <= margin -> gate passes.
        probes = {v: make_probe(v, sec_before=0.010, sec_after=0.012)
                  for v in (3, 6)}
        sim = Simulator(economics_config())
        d, frozen_at = None, None
        for v in range(1, 8):
            d = self._flat_once(sim, probes.get(v))
            if d.state == FreezeState.FROZEN:
                frozen_at = v
                break
        # v3 probe stays fresh (age <= 3); closing v4 completes patience.
        self.assertEqual(frozen_at, 5)
        self.assertEqual((d.reason, d.metrics["drafter/economics_gate_ready"]),
                         ("no_response_confirmed", 1.0))
        self.assertLessEqual(d.metrics["drafter/roi_next_ucb95"], 1.0)
        self.assertEqual((sim.policy.probe_completed, sim.policy.probe_incomplete), (1, 0))
    def test_positive_roi_probe_blocks_freeze(self):
        probes = {v: make_probe(v, sec_before=0.020, sec_after=0.010)  # 2x faster
                  for v in (3, 6)}
        d = self._flat_versions(Simulator(economics_config()), 7, probes=probes)
        self.assertNotEqual(d.state, FreezeState.FROZEN)
        self.assertEqual(d.reason, "economic_value_positive")
        self.assertGreater(d.metrics["drafter/roi_next_ucb95"], 1.0)
    def test_incomplete_probe_fails_closed(self):
        sim = Simulator(economics_config())
        self._flat_versions(sim, 3, probes={3: make_probe(3, n=20, timing_holes=20)})
        # coverage 0.0 < 0.8 and 0 timing pairs < 16: gate not ready.
        self.assertEqual(sim.policy.probe_incomplete, 1)
        self.assertFalse(sim.policy._economics_gate_ready())
        d = self._flat_versions(sim, 4)
        self.assertNotEqual(d.state, FreezeState.FROZEN)
        self.assertEqual(d.metrics["drafter/economics_gate_ready"], 0.0)
    def test_stale_and_wrong_version_probes(self):
        p = DrafterFreezePolicy(economics_config())
        p.observe(FreezeEvidence(
            global_step=1, drafter_version=3, publish_completed=True,
            update_succeeded=True, publish_succeeded=True,
            update_cost_seconds=10.0, probe=make_probe(3)))
        self.assertIsNotNone(p._fresh_probe_stats(6))  # age == interval
        self.assertIsNone(p._fresh_probe_stats(7))     # one cycle stale
        self.assertIsNone(p._fresh_probe_stats(2))     # from the future
        p2 = DrafterFreezePolicy(economics_config())
        p2.observe(FreezeEvidence(  # probe version != publish version: ignored
            global_step=1, drafter_version=4, publish_completed=True,
            update_succeeded=True, publish_succeeded=True,
            update_cost_seconds=10.0, probe=make_probe(3)))
        self.assertEqual((p2._latest_probe, p2.probe_completed), (None, 0))
    def test_probe_state_round_trip(self):
        sim = Simulator(economics_config())
        self._flat_versions(sim, 3, probes={3: make_probe(3)})
        stats_before = sim.policy._latest_probe_stats
        restored = DrafterFreezePolicy(economics_config())
        restored.load_state_dict(sim.policy.state_dict())
        self.assertEqual(restored._latest_probe_stats, stats_before)
        self.assertEqual(restored.probe_completed, 1)
        self.assertEqual(restored._fresh_probe_stats(3), sim.policy._fresh_probe_stats(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
