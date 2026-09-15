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
"""CPU contract for the shared drafter dispatch used by online and standalone training."""

from __future__ import annotations

import pytest

from verl_speco.backends.factory import SUPPORTED_DRAFTER_ALGORITHMS, build_trainer_backend
from verl_speco.integration.oldlogprob_layer_ids import resolve_drafter_hidden_states_layout


class _DrafterConfig:
    """Minimal stand-in for the actor_rollout_ref config the factory reads."""

    def __init__(self, algorithm: str):
        self.rollout = type("_Rollout", (), {"drafter": type("_Drafter", (), {"speculative_algorithm": algorithm})})


def test_factory_lists_every_supported_algorithm() -> None:
    assert set(SUPPORTED_DRAFTER_ALGORITHMS) == {
        "EAGLE1",
        "EAGLE2",
        "EAGLE3",
        "DFLASH",
        "DFLASH2",
        "DSPARK",
        "DOMINO",
        "PEAGLE",
    }


def test_factory_rejects_unknown_algorithm_before_importing_a_backend() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_trainer_backend(_DrafterConfig("NOT_AN_ALGORITHM"), None)

    message = str(excinfo.value)
    assert "Unsupported drafter algorithm" in message
    for algorithm in SUPPORTED_DRAFTER_ALGORITHMS:
        assert algorithm in message


@pytest.mark.parametrize(
    ("algorithm", "training_cfg", "expected"),
    [
        ("EAGLE3", {}, "eagle3_aux_plus_last"),
        ("EAGLE1", {}, "eagle3_aux_plus_last"),
        ("PEAGLE", {}, "eagle3_aux_plus_last"),
        ("DFLASH", {}, "dflash_aux"),
        # Domino is a DFlash variant and consumes the same aux context layers.
        # Tagging it eagle3_aux_plus_last makes DFlash preprocessing fail closed.
        ("DOMINO", {}, "dflash_aux"),
        ("domino", {}, "dflash_aux"),
        # DFlash2 is likewise a DFlash variant on the same context layers.
        ("DFLASH2", {}, "dflash_aux"),
        ("dflash2", {}, "dflash_aux"),
        ("DSPARK", {}, "dflash_aux_plus_last"),
        ("DSPARK", {"dspark_l1_loss_alpha": 0.0}, "dflash_aux"),
        ("DSPARK", {"dspark_l1_loss_alpha": None}, "dflash_aux"),
    ],
)
def test_hidden_states_layout_per_algorithm(algorithm, training_cfg, expected) -> None:
    assert resolve_drafter_hidden_states_layout(algorithm, training_cfg) == expected


def test_hidden_states_layout_defaults_to_eagle_layout_without_algorithm() -> None:
    assert resolve_drafter_hidden_states_layout(None, {}) == "eagle3_aux_plus_last"
