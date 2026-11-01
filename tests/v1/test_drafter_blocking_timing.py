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
"""Keep drafter blocking totals separate from actor work and target sync."""

from types import SimpleNamespace

import pytest

from verl_speco.trainer.v1 import speco_mixin


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("local_updates", [1, 3])
def test_drafter_total_excludes_actor_and_target_sync_and_resets(
    monkeypatch, local_updates
):
    clock = Clock()
    monkeypatch.setattr(speco_mixin, "time", clock)

    class Upstream:
        def _update_actor(self, batch, metrics):
            clock.advance(100)
            return "actor"

        def on_step_end(self):
            clock.advance(200)
            return "target_sync"

    class Harness(speco_mixin.SpecoV1Mixin, Upstream):
        config = None
        _speco_freeze_policy = None
        _speco_colocate_admission_closed = False
        _speco_v1_pending_training = False
        launch = True

        def __getattr__(self, name):
            raise AttributeError(name)

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_on_before_actor_update(self):
            clock.advance(1)
            self._pending_target_lm_head_sync = object()
            return SimpleNamespace(
                training_plan=SimpleNamespace(launch=self.launch, reason="interval"),
                metrics={"timing_s/drafter_sync_target_lm_head_overlap_window": 50},
            )

        def _speco_finish_target_lm_head_weight_sync(self, pending):
            clock.advance(0.2)
            return {}

        def _speco_train_drafter(self, plan):
            clock.advance(6)
            return True, {"drafter/trained": 1}

        def _speco_publish_drafter_weights(self, *args, **kwargs):
            clock.advance(3)
            return {"drafter/published": 1}

        def _speco_wait_pending_drafter_publish(self):
            clock.advance(0.5)
            return 1

    trainer = Harness()
    trainer.parameter_sync_step = local_updates
    metrics = {}
    for local in range(local_updates):
        trainer.local_trigger_step = local
        assert trainer._update_actor(None, metrics) == "actor"
    assert metrics["timing_s/actor_update_without_drafter"] == pytest.approx(
        100 * local_updates
    )
    assert metrics["timing_s/drafter"] == pytest.approx(7.2)
    trainer.on_step_end()
    metrics.update(trainer._pending_sync_metrics)
    assert metrics["timing_s/drafter"] == pytest.approx(10.7)
    assert metrics["timing_s/drafter_v1_publish_blocking"] == pytest.approx(3.5)
    assert trainer._speco_finalize_v1_drafter_publish() == {}
    trainer.launch = False
    for local in range(local_updates):
        trainer.local_trigger_step = local
        trainer._update_actor(None, metrics)
    assert metrics["timing_s/drafter"] == pytest.approx(1.2)
    assert metrics["timing_s/actor_update_without_drafter"] == pytest.approx(
        100 * local_updates
    )
