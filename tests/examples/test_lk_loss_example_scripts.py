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

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backend", ("dflash", "dspark", "eagle3"))
def test_lk_loss_examples_enable_pure_lk_loss(backend: str) -> None:
    source = (
        ROOT / "examples" / f"run_qwen3-8b-lk-loss_drafter_{backend}_vllm.sh"
    ).read_text(encoding="utf-8")

    assert "python3 -m verl_speco.main" in source
    assert (
        f"actor_rollout_ref.rollout.drafter.training.{backend}_ce_loss_alpha=0.0"
        in source
    )
    assert (
        f"actor_rollout_ref.rollout.drafter.training.{backend}_lk_loss_alpha=1.0"
        in source
    )
    assert (
        "actor_rollout_ref.rollout.drafter.vllm.draft_sample_method=probabilistic"
        in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.lk_temperature=0.6" in source
    assert "actor_rollout_ref.rollout.temperature=0.6" in source


def test_adaptive_hybrid_lk_example_selects_hybrid_objective() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b-adaptive-hybrid-lk_drafter_eagle3_vllm.sh"
    ).read_text(encoding="utf-8")

    assert "eagle3_ce_loss_alpha=0.0" in source
    assert "eagle3_lk_loss_alpha=1.0" in source
    assert "eagle3_lk_loss_type=adaptive_hybrid" in source
    assert "eagle3_lk_hybrid_eta=1.0" in source
