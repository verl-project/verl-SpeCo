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

import json
from types import SimpleNamespace

import pytest

from verl_speco.integration import oldlogprob_runtime
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
    _find_layers_and_final_norm,
    _install_oldlogprob_fsdp_batch_postprocess_patch,
    _select_and_merge_concatenated_hidden,
    _to_cpu_transfer_tensor,
    oldlogprob_hidden_runtime_enabled,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    eagle3_num_aux_hidden_states_from_config,
    resolve_oldlogprob_aux_layer_ids,
)


def test_fsdp2_runtime_install_does_not_import_veomni(monkeypatch) -> None:
    imported_modules = []
    fsdp_module = SimpleNamespace(__name__="verl.workers.engine.fsdp.transformer_impl")

    def fake_import_module(name: str):
        imported_modules.append(name)
        if name == "verl.workers.engine.fsdp.transformer_impl":
            return fsdp_module
        raise AssertionError(f"unexpected backend import: {name}")

    monkeypatch.setattr(oldlogprob_runtime, "_PATCHED", True)
    monkeypatch.setattr(oldlogprob_runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        oldlogprob_runtime,
        "_install_oldlogprob_fsdp_batch_postprocess_patch",
        lambda module: module is fsdp_module,
    )
    monkeypatch.setattr(
        oldlogprob_runtime,
        "_install_oldlogprob_training_worker_postprocess_patch",
        lambda: True,
    )

    assert oldlogprob_runtime.install_oldlogprob_hidden_runtime_patch(
        actor_backend="fsdp2"
    )
    assert imported_modules == ["verl.workers.engine.fsdp.transformer_impl"]


def _drafter(enabled: bool) -> dict:
    return {
        "enable": True,
        "enable_drafter_training": True,
        "training": {"collect_hidden_states_from_old_logprob": enabled},
    }


def test_oldlogprob_collection_is_disabled_by_default() -> None:
    assert oldlogprob_hidden_runtime_enabled({}) is False
    assert (
        oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": _drafter(False)}})
        is False
    )


def test_oldlogprob_collection_requires_online_drafter_training() -> None:
    rollout_disabled = _drafter(True)
    rollout_disabled["enable"] = False
    training_disabled = _drafter(True)
    training_disabled["enable_drafter_training"] = False

    assert (
        oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": rollout_disabled}})
        is False
    )
    assert (
        oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": training_disabled}})
        is False
    )


def test_oldlogprob_collection_accepts_both_config_shapes() -> None:
    assert oldlogprob_hidden_runtime_enabled({"rollout": {"drafter": _drafter(True)}})
    assert oldlogprob_hidden_runtime_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": _drafter(True)}}}
    )


def test_oldlogprob_collection_can_be_enabled_from_worker_environment() -> None:
    payload = json.dumps(_drafter(True))

    assert oldlogprob_hidden_runtime_enabled({}, drafter_env=payload)
    assert oldlogprob_hidden_runtime_enabled({}, drafter_env="{invalid") is False


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


def test_eagle3_oldlogprob_falls_back_to_default_three_layers() -> None:
    assert resolve_oldlogprob_aux_layer_ids(
        {"speculative_algorithm": "EAGLE3"},
        target_num_hidden_layers=36,
        model_configs=[],
    ) == [2, 18, 33]
    assert (
        eagle3_num_aux_hidden_states_from_config({"speculative_algorithm": "EAGLE3"})
        is None
    )


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


def test_forward_hook_merges_already_selected_batch_hidden_without_reselection() -> (
    None
):
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


def test_unselected_flat_hidden_keeps_original_row_selection() -> None:
    torch = pytest.importorskip("torch")
    context = _selection_context(batch_size=2, hidden_rows=2)
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    selected, _owner_mask = _select_and_merge_concatenated_hidden(context, [hidden])

    assert selected.shape == (2, 2, 2)
    torch.testing.assert_close(selected[0, 0], hidden[0])
    torch.testing.assert_close(selected[1, 0], hidden[2])


def test_cpu_transfer_tensor_detaches_cpu_tensor_without_copy() -> None:
    torch = pytest.importorskip("torch")
    source = torch.tensor([1.0], requires_grad=True)

    transferred = _to_cpu_transfer_tensor(source)

    assert transferred.device.type == "cpu"
    assert transferred.requires_grad is False
    assert transferred.data_ptr() == source.data_ptr()
    torch.testing.assert_close(transferred, source.detach())


def test_cpu_transfer_tensor_detaches_sparse_payload_rows() -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "rows": torch.tensor([[1.0]], requires_grad=True),
        "batch_indices": [0],
        "row_indices": [[0]],
    }

    transferred = _to_cpu_transfer_tensor(payload)

    assert transferred["rows"].device.type == "cpu"
    assert transferred["rows"].requires_grad is False
    assert transferred["rows"].data_ptr() == payload["rows"].data_ptr()
    assert transferred["batch_indices"] == [0]


def test_forward_hook_rejects_malformed_selected_hidden() -> None:
    torch = pytest.importorskip("torch")
    context = _selection_context(batch_size=4, hidden_rows=129)

    with pytest.raises(RuntimeError, match="invalid selected hidden tensor"):
        _select_and_merge_concatenated_hidden(
            context,
            [torch.randn(4, 128, 8)],
            already_selected=True,
        )


@pytest.mark.parametrize("root_attr", ["model", "thinker"])
def test_hidden_layer_discovery_supports_veomni_multimodal_wrappers(
    root_attr: str,
) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    text_model = SimpleNamespace(
        layers=nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)]),
        norm=nn.LayerNorm(4),
    )
    if root_attr == "model":
        module = SimpleNamespace(model=SimpleNamespace(language_model=text_model))
    else:
        module = SimpleNamespace(thinker=SimpleNamespace(model=text_model))

    layers, final_norm = _find_layers_and_final_norm(SimpleNamespace(module=module))

    assert len(layers) == 2
    assert layers[0] is text_model.layers[0]
    assert final_norm is text_model.norm


def test_batch_postprocess_patch_is_installed_per_engine_module() -> None:
    module_a = SimpleNamespace(
        __name__="fake.fsdp.transformer_impl",
        postprocess_batch_func=lambda output_lst, indices, data: {"engine": "fsdp"},
    )
    module_b = SimpleNamespace(
        __name__="fake.veomni.transformer_impl",
        postprocess_batch_func=lambda output_lst, indices, data: {"engine": "veomni"},
    )

    assert _install_oldlogprob_fsdp_batch_postprocess_patch(module_a)
    assert _install_oldlogprob_fsdp_batch_postprocess_patch(module_b)
    assert module_a.postprocess_batch_func([], None, None) == {"engine": "fsdp"}
    assert module_b.postprocess_batch_func([], None, None) == {"engine": "veomni"}


def test_veomni_batch_postprocess_keeps_router_replay_output() -> None:
    def native_postprocess(output_lst, indices, data):
        model_output = output_lst[0]["model_output"]
        assert model_output["routed_experts"] == "routes"
        assert OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY not in model_output
        return {"model_output": {"routed_experts": "routes"}}

    module = SimpleNamespace(
        __name__="fake.veomni.router_replay.transformer_impl",
        postprocess_batch_func=native_postprocess,
    )
    assert _install_oldlogprob_fsdp_batch_postprocess_patch(module)

    result = module.postprocess_batch_func(
        [
            {
                "model_output": {
                    "routed_experts": "routes",
                    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY: ["hidden-ref"],
                }
            }
        ],
        None,
        None,
    )

    assert result["model_output"]["routed_experts"] == "routes"
    assert result["model_output"][OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY] == [
        "hidden-ref"
    ]
