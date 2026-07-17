from __future__ import annotations

import json

import pytest

from verl_speco.integration.oldlogprob_runtime import (
    _select_and_merge_concatenated_hidden,
    oldlogprob_hidden_runtime_enabled,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    eagle3_num_aux_hidden_states_from_config,
    resolve_oldlogprob_aux_layer_ids,
)


def _drafter(enabled: bool) -> dict:
    return {
        "enable": True,
        "enable_drafter_training": True,
        "training": {"collect_hidden_states_from_old_logprob": enabled},
    }


def test_oldlogprob_collection_is_disabled_by_default() -> None:
    assert oldlogprob_hidden_runtime_enabled({}) is False
    assert oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": _drafter(False)}}) is False
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_oldlogprob_collection_is_disabled_by_default", flush=True)


def test_oldlogprob_collection_requires_online_drafter_training() -> None:
    rollout_disabled = _drafter(True)
    rollout_disabled["enable"] = False
    training_disabled = _drafter(True)
    training_disabled["enable_drafter_training"] = False

    assert oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": rollout_disabled}}) is False
    assert oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": training_disabled}}) is False
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_oldlogprob_collection_requires_online_drafter_training", flush=True)


def test_oldlogprob_collection_accepts_both_config_shapes() -> None:
    assert oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": _drafter(True)}})
    assert oldlogprob_hidden_runtime_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": _drafter(True)}}}
    )
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_oldlogprob_collection_accepts_both_config_shapes", flush=True)


def test_oldlogprob_collection_can_be_enabled_from_worker_environment() -> None:
    payload = json.dumps(_drafter(True))

    assert oldlogprob_hidden_runtime_enabled({}, drafter_env=payload)
    assert oldlogprob_hidden_runtime_enabled({}, drafter_env="{invalid") is False
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_oldlogprob_collection_can_be_enabled_from_worker_environment", flush=True)


def test_eagle3_oldlogprob_accepts_three_explicit_aux_layers() -> None:
    drafter_cfg = {
        "speculative_algorithm": "EAGLE3",
        "eagle_aux_hidden_state_layer_ids": [2, 18, 33],
    }

    assert resolve_oldlogprob_aux_layer_ids(
        drafter_cfg,
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [2, 18, 33]
    assert eagle3_num_aux_hidden_states_from_config(drafter_cfg) == 3
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_eagle3_oldlogprob_accepts_three_explicit_aux_layers", flush=True)


def test_eagle3_oldlogprob_falls_back_to_default_three_layers() -> None:
    assert resolve_oldlogprob_aux_layer_ids(
        {"speculative_algorithm": "EAGLE3"},
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [2, 18, 33]
    assert eagle3_num_aux_hidden_states_from_config({"speculative_algorithm": "EAGLE3"}) is None
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_eagle3_oldlogprob_falls_back_to_default_three_layers", flush=True)


def test_eagle3_oldlogprob_accepts_top_level_target_layer_ids() -> None:
    drafter_cfg = {
        "speculative_algorithm": "EAGLE3",
        "target_layer_ids": [1, 9, 17, 25, 33],
    }

    assert resolve_oldlogprob_aux_layer_ids(
        drafter_cfg,
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [1, 9, 17, 25, 33]
    assert eagle3_num_aux_hidden_states_from_config(drafter_cfg) == 5
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_eagle3_oldlogprob_accepts_top_level_target_layer_ids", flush=True)


def test_dflash_oldlogprob_ignores_dspark_training_defaults() -> None:
    drafter_cfg = {
        "speculative_algorithm": "DFLASH",
        "training": {
            "dflash_num_target_layers": 3,
            "dspark_num_target_layers": 5,
        },
    }

    assert resolve_oldlogprob_aux_layer_ids(
        drafter_cfg,
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [1, 17, 33]
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_dflash_oldlogprob_ignores_dspark_training_defaults", flush=True)


def test_dspark_oldlogprob_uses_dspark_training_defaults() -> None:
    drafter_cfg = {
        "speculative_algorithm": "DSPARK",
        "training": {
            "dspark_num_target_layers": 5,
        },
    }

    assert resolve_oldlogprob_aux_layer_ids(
        drafter_cfg,
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [1, 9, 17, 25, 33]
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_dspark_oldlogprob_uses_dspark_training_defaults", flush=True)


def _selection_context(*, batch_size: int, hidden_rows: int) -> dict:
    return {
        "batch_size": batch_size,
        "output_batch_size": batch_size,
        "hidden_rows": hidden_rows,
        "local_positions": [0, hidden_rows],
        "local_batch_indices": [0, min(batch_size - 1, 1)],
        "local_row_indices": [0, 0],
        "max_local_position": hidden_rows,
        "sparse_sp_merge": False,
        "sp_group": None,
        "timing_us": {"select": 0.0, "sp_merge": 0.0, "concat": 0.0},
    }


def test_forward_hook_merges_already_selected_batch_hidden_without_reselection() -> None:
    torch = pytest.importorskip("torch")
    context = _selection_context(batch_size=4, hidden_rows=129)
    aux_hidden = torch.randn(4, 129, 8)
    final_hidden = torch.randn(4, 129, 4)

    selected, owner_mask = _select_and_merge_concatenated_hidden(
        context,
        [aux_hidden, final_hidden],
        already_selected=True,
    )

    assert selected.shape == (4, 129, 12)
    torch.testing.assert_close(selected[..., :8], aux_hidden)
    torch.testing.assert_close(selected[..., 8:], final_hidden)
    assert owner_mask.shape == (4, 129)
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_forward_hook_merges_already_selected_batch_hidden_without_reselection", flush=True)


def test_unselected_flat_hidden_keeps_original_row_selection() -> None:
    torch = pytest.importorskip("torch")
    context = _selection_context(batch_size=2, hidden_rows=2)
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    selected, _owner_mask = _select_and_merge_concatenated_hidden(context, [hidden])

    assert selected.shape == (2, 2, 2)
    torch.testing.assert_close(selected[0, 0], hidden[0])
    torch.testing.assert_close(selected[1, 0], hidden[2])
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_unselected_flat_hidden_keeps_original_row_selection", flush=True)


def test_forward_hook_rejects_malformed_selected_hidden() -> None:
    torch = pytest.importorskip("torch")
    context = _selection_context(batch_size=4, hidden_rows=129)

    with pytest.raises(RuntimeError, match="invalid selected hidden tensor"):
        _select_and_merge_concatenated_hidden(
            context,
            [torch.randn(4, 128, 8)],
            already_selected=True,
        )
    print("tests/integration/test_oldlogprob_runtime_contract.py::test_forward_hook_rejects_malformed_selected_hidden", flush=True)
