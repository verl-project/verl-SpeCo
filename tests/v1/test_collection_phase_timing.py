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
"""V1 collection timings preserve outputs, failure behavior and phase boundaries."""

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.v1 import speco_mixin


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, value):
        self.now += value


@pytest.fixture
def collection_harness(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(speco_mixin, "time", clock)
    for name in (
        "transfer_queue",
        "verl",
        "verl.utils",
        "verl.utils.tensordict_utils",
        "verl.workers",
        "verl.workers.utils",
        "verl.workers.utils.padding",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.trainer.ppo.core_algos",
    ):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    tu = sys.modules["verl.utils.tensordict_utils"]
    monkeypatch.setattr(
        tu, "assign_non_tensor_data", lambda *args, **kwargs: None, raising=False
    )
    monkeypatch.setattr(
        tu, "assign_non_tensor", lambda *args, **kwargs: None, raising=False
    )
    padding = sys.modules["verl.workers.utils.padding"]
    for name in (
        "left_right_2_no_padding",
        "no_padding_2_padding",
        "response_to_nested",
    ):
        monkeypatch.setattr(padding, name, lambda value, *args: value, raising=False)
    puts = []
    monkeypatch.setattr(
        sys.modules["transfer_queue"],
        "kv_batch_put",
        lambda **kwargs: puts.append(kwargs) or "stored",
        raising=False,
    )

    def aggregate(**kwargs):
        clock.advance(4)
        return kwargs["loss_mat"].mean()

    monkeypatch.setattr(
        sys.modules["verl.trainer.ppo.core_algos"], "agg_loss", aggregate, raising=False
    )
    log_probs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    entropy = torch.ones(2, 3)
    response_mask = torch.ones(2, 3)

    class Nested(dict):
        def select(self, *names):
            return {name: self[name] for name in names}

    class Batch:
        def __init__(self):
            self.batch = {"response_mask": response_mask}

        def __len__(self):
            return 2

        def to_tensordict(self):
            return {"input_ids": torch.ones(2, 3, dtype=torch.long)}

    class Upstream:
        def _compute_old_log_prob(self, batch, metrics):
            return "upstream"

    class Harness(speco_mixin.SpecoV1Mixin, Upstream):
        _speco_freeze_policy = None
        _speco_last_collection_outcome = None
        _speco_last_hard_selection_metrics = None
        global_steps = 6
        fail_collection = False
        emit_worker_timing = True
        no_plan = False
        _speco_last_oldlogprob_candidate_samples = 2
        _speco_last_oldlogprob_short_response_skipped = 0
        _speco_last_oldlogprob_planned_samples = 2
        _speco_last_oldlogprob_collected_samples = 2
        _speco_last_oldlogprob_collected_rows = 6

        def __getattr__(self, name):
            raise AttributeError(name)

        def _speco_oldlogprob_collection_enabled(self):
            return True

        def _speco_v1_batch_data(self, batch):
            clock.advance(1)
            return Nested(response_mask=response_mask), Batch()

        def _speco_build_oldlogprob_collect_plan(self, batch):
            if self.no_plan:
                return None
            return {
                "collect_mask": torch.ones(2, dtype=torch.bool),
                "hidden_positions": torch.zeros(2, 3, dtype=torch.long),
                "hidden_position_mask": torch.ones(2, 3, dtype=torch.bool),
                "owner_rank": torch.zeros(2, dtype=torch.long),
                "collection_plan": SimpleNamespace(metrics=dict),
            }

        def _speco_oldlogprob_aux_layer_ids(self):
            return [1]

        def _speco_oldlogprob_hidden_capture_impl(self):
            return "forward_hook"

        def _speco_oldlogprob_hidden_layout(self):
            return "BSH"

        def _speco_compute_collection_log_prob(self, control):
            clock.advance(2)
            return {"log_probs": log_probs, "entropy": entropy}

        def _speco_collect_oldlogprob_features(self, batch, plan, output):
            clock.advance(3)
            if self.fail_collection:
                raise RuntimeError("collection failed")
            if self.emit_worker_timing:
                self._speco_last_oldlogprob_select_elapsed_sec = 0.1
                self._speco_last_oldlogprob_control_bytes = 1000
                self._speco_last_oldlogprob_bulk_host_contexts = 2
            return 2

    trainer = Harness()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=AttrDict(loss_agg_mode="token-mean", loss_scale_factor=1),
            rollout=AttrDict(temperature=1.0),
        )
    )
    trainer._speco_v1_state = {"features_collected": 0}
    return trainer, puts, log_probs, entropy


def test_collection_timings_partition_wall_time_and_do_not_change_outputs(
    collection_harness,
):
    trainer, puts, log_probs, entropy = collection_harness
    metrics = {}
    assert (
        trainer._compute_old_log_prob(
            SimpleNamespace(keys=[1, 2], partition_id="p"), metrics
        )
        == "stored"
    )
    assert [
        metrics[f"timing_s/oldlogprob_{name}"]
        for name in ("prepare", "compute_with_capture", "collect", "restore")
    ] == [1, 2, 3, 4]
    assert metrics["timing_s/oldlogprob_collection_path_total"] == 10
    assert metrics["timing_s/oldlogprob_worker_select_sum"] == 0.1
    assert metrics["oldlogprob/bulk_host_contexts"] == 2
    assert trainer._speco_v1_state["features_collected"] == 2
    torch.testing.assert_close(
        puts[0]["fields"]["old_log_probs"], log_probs, rtol=0, atol=0
    )
    torch.testing.assert_close(puts[0]["fields"]["entropy"], entropy, rtol=0, atol=0)
    trainer.emit_worker_timing = False
    metrics = {}
    trainer._compute_old_log_prob(
        SimpleNamespace(keys=[1, 2], partition_id="p"), metrics
    )
    assert metrics["timing_s/oldlogprob_worker_select_sum"] == 0
    assert (
        metrics["oldlogprob/bulk_host_contexts"]
        == metrics["oldlogprob/host_control_bytes"]
        == 0
    )


def test_failed_collection_still_raises_before_tq_write(collection_harness):
    trainer, puts, _, _ = collection_harness
    trainer.fail_collection = True
    with pytest.raises(RuntimeError, match="collection failed"):
        trainer._compute_old_log_prob(SimpleNamespace(), {})
    assert puts == []


def test_absent_collection_plan_keeps_upstream_path(collection_harness):
    trainer, puts, _, _ = collection_harness
    trainer.no_plan = True
    metrics = {}
    assert trainer._compute_old_log_prob(SimpleNamespace(), metrics) == "upstream"
    assert puts == []
    assert "timing_s/oldlogprob_compute_with_capture" not in metrics
