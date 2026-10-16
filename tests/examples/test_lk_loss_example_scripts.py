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


@pytest.mark.parametrize(
    "loss,backend,ce_alpha,lk_alpha",
    (
        ("lk-loss", "eagle3", "0.0", "1.0"),
        ("adaptive-hybrid-lk", "dspark", "0.5", "0.5"),
        ("dpace", "dflash", "1.0", "0.0"),
    ),
)
def test_loss_examples_select_objective_and_resources(
    loss: str, backend: str, ce_alpha: str, lk_alpha: str
) -> None:
    source = (
        ROOT / "examples" / f"run_qwen3-8b-{loss}_drafter_{backend}_vllm.sh"
    ).read_text(encoding="utf-8")

    assert "python3 -m verl_speco.main" in source
    assert (
        f"actor_rollout_ref.rollout.drafter.training.{backend}_ce_loss_alpha={ce_alpha}"
        in source
    )
    assert (
        f"actor_rollout_ref.rollout.drafter.training.{backend}_lk_loss_alpha={lk_alpha}"
        in source
    )
    assert (
        "actor_rollout_ref.rollout.drafter.vllm.draft_sample_method=probabilistic"
        in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.lk_temperature=0.6" in source
    assert "actor_rollout_ref.rollout.temperature=0.6" in source
    assert "MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}" in source
    assert "train_sp=${SPECO_TRAIN_SP:-4}" in source
    assert "trainer.n_gpus_per_node=${SPECO_N_GPUS_PER_NODE:-4}" in source


def test_adaptive_hybrid_lk_example_selects_hybrid_objective() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b-adaptive-hybrid-lk_drafter_dspark_vllm.sh"
    ).read_text(encoding="utf-8")

    assert "dspark_ce_loss_alpha=0.5" in source
    assert "dspark_l1_loss_alpha=0.25" in source
    assert "dspark_lk_loss_alpha=0.5" in source
    assert "dspark_lk_loss_type=adaptive_hybrid" in source
    assert "dspark_lk_hybrid_eta=3.0" in source


def test_dpace_example_selects_position_weighting() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b-dpace_drafter_dflash_vllm.sh"
    ).read_text(encoding="utf-8")

    assert "dflash_per_position_loss_weight=dpace" in source
    assert "dflash_dpace_alpha=0.5" in source
    assert "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=4" in source
    assert "actor_rollout_ref.rollout.drafter.training.publish_async=False" in source
