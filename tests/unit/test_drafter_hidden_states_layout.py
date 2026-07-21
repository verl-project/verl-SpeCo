from __future__ import annotations

import pytest

from verl_speco.integration.oldlogprob_layer_ids import resolve_drafter_hidden_states_layout


@pytest.mark.parametrize(
    ("algorithm", "training_cfg", "expected"),
    [
        ("EAGLE3", {}, "eagle3_aux_plus_last"),
        ("EAGLE1", {}, "eagle3_aux_plus_last"),
        ("PEAGLE", {}, "eagle3_aux_plus_last"),
        ("DFLASH", {}, "dflash_aux"),
        # Domino is a DFlash variant and consumes the same aux context layers.
        ("DOMINO", {}, "dflash_aux"),
        ("domino", {}, "dflash_aux"),
        ("DSPARK", {}, "dflash_aux_plus_last"),
        ("DSPARK", {"dspark_l1_loss_alpha": 0.0}, "dflash_aux"),
    ],
)
def test_hidden_states_layout_per_algorithm(algorithm, training_cfg, expected):
    assert resolve_drafter_hidden_states_layout(algorithm, training_cfg) == expected


def test_hidden_states_layout_defaults_to_eagle_layout_without_algorithm():
    assert resolve_drafter_hidden_states_layout(None, {}) == "eagle3_aux_plus_last"
