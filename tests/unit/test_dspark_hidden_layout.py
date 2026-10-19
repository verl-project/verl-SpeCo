# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

# Load the configuration helper without importing optional runtime backends.
_spec = importlib.util.spec_from_file_location(
    "oldlogprob_layer_ids",
    Path(__file__).resolve().parents[2] / "verl_speco/integration/oldlogprob_layer_ids.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
resolve_drafter_hidden_states_layout = _module.resolve_drafter_hidden_states_layout


@pytest.mark.parametrize("algorithm", ["DOMINO", "domino"])
@pytest.mark.parametrize("key", ["dflash_lk_loss_alpha", "dspark_lk_loss_alpha"])
def test_domino_rejects_lk_loss(algorithm, key):
    with pytest.raises(ValueError, match=f"Domino does not support LK loss.*{key}"):
        resolve_drafter_hidden_states_layout(algorithm, {key: 0.5})


@pytest.mark.parametrize("alpha", [0.0, None, "0.0"])
def test_domino_without_lk_keeps_aux_layout(alpha):
    assert (
        resolve_drafter_hidden_states_layout(
            "DOMINO", {"dflash_lk_loss_alpha": alpha, "dspark_lk_loss_alpha": alpha}
        )
        == "dflash_aux"
    )


@pytest.mark.parametrize(
    "alpha,source,expected",
    [
        (0.0, "labels", "dflash_aux"),
        (0.0, "target_argmax", "dflash_aux_plus_last"),
        (0.0, "TARGET_ARGMAX", "dflash_aux_plus_last"),
        (0.0, "Target_Argmax", "dflash_aux_plus_last"),
        (0.0, "LABELS", "dflash_aux"),
        (0.0, None, "dflash_aux"),
        (0.9, "labels", "dflash_aux_plus_last"),
        (0.9, "target_argmax", "dflash_aux_plus_last"),
    ],
)
def test_dspark_target_hidden_requirement(alpha, source, expected):
    config = {"dspark_l1_loss_alpha": alpha, "ce_target_source": source}
    assert resolve_drafter_hidden_states_layout("DSPARK", config) == expected
    assert resolve_drafter_hidden_states_layout("DFLASH", config) == "dflash_aux"
