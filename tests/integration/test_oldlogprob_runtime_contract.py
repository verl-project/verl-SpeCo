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
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl_speco.integration import oldlogprob_runtime
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
    OLD_LOGPROB_TARGET_LOGZ_TEMPERATURE_KEY,
    _compute_oldlogprob_target_logz,
    _find_layers_and_final_norm,
    _hidden_state_capture_target,
    _install_oldlogprob_fsdp_batch_postprocess_patch,
    _put_oldlogprob_hidden_refs,
    _resolve_hidden_state,
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
    monkeypatch.setattr(
        oldlogprob_runtime.importlib, "import_module", fake_import_module
    )
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


def test_eagle3_output_ids_select_same_physical_hidden_states_as_vllm() -> None:
    torch = pytest.importorskip("torch")
    hidden_states = tuple(torch.tensor([index]) for index in range(37))

    selected = [
        _resolve_hidden_state(hidden_states, layer_id, layer_id_space="output")
        for layer_id in [2, 18, 33]
    ]

    assert [int(value.item()) for value in selected] == [2, 18, 33]
    assert _hidden_state_capture_target(2, 36, layer_id_space="output") == (
        "layer",
        1,
    )


def test_decoder_ids_keep_hf_embedding_offset_for_dflash_and_eagle12() -> None:
    torch = pytest.importorskip("torch")
    hidden_states = tuple(torch.tensor([index]) for index in range(37))

    assert (
        int(_resolve_hidden_state(hidden_states, 2, layer_id_space="decoder").item())
        == 3
    )
    assert _hidden_state_capture_target(2, 36, layer_id_space="decoder") == (
        "layer",
        2,
    )
    assert _hidden_state_capture_target(35, 36, layer_id_space="decoder") == (
        "final",
        None,
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
    assert result["model_output"][OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY] == ["hidden-ref"]


def test_oldlogprob_target_logz_is_temperature_aware() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]])
    micro_batch = {OLD_LOGPROB_TARGET_LOGZ_TEMPERATURE_KEY: 0.5}

    actual = _compute_oldlogprob_target_logz({"logits": logits}, micro_batch)

    torch.testing.assert_close(
        actual,
        torch.logsumexp(logits.float() / 0.5, dim=-1, keepdim=True),
    )


def test_oldlogprob_target_logz_is_disabled_without_temperature() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.randn(1, 2, 4)

    assert _compute_oldlogprob_target_logz({"logits": logits}, {}) is None


@pytest.mark.parametrize("capture_impl", ["forward_hook", "output_hidden_states"])
@pytest.mark.parametrize("inplace_scaling", [True, False])
@pytest.mark.parametrize("collect_logz", [True, False])
@pytest.mark.parametrize("lk_temperature", [0.6, 1.0])
def test_fsdp_logz_capture_precedes_upstream_temperature_scaling(
    monkeypatch, capture_impl, inplace_scaling, collect_logz, lk_temperature
) -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.backends.lk_loss import (
        full_vocab_subset_probs,
        negative_log_acceptance_loss,
    )

    raw_logits = torch.tensor(
        [
            [
                [30.0, 29.0, 28.0],
                [31.0, 29.0, 28.0],
                [28.0, 30.0, 29.0],
                [32.0, 29.0, 28.0],
            ]
        ]
    )
    temperature = 0.6

    class FakeEngine:
        def prepare_model_inputs(self, micro_batch):
            return {}, {}

        def prepare_model_outputs(
            self, output, output_args, micro_batch, logits_processor_func
        ):
            logits = output.logits.squeeze(0)
            if inplace_scaling:
                logits.div_(temperature)
            else:
                logits = logits / temperature
            return {"log_probs": torch.log_softmax(logits, dim=-1)}

    module = SimpleNamespace(FSDPEngineWithLMHead=FakeEngine)
    monkeypatch.setattr(oldlogprob_runtime, "_PATCHED", False)
    monkeypatch.setattr(
        oldlogprob_runtime.importlib, "import_module", lambda name: module
    )
    for name in (
        "_install_oldlogprob_fsdp_batch_postprocess_patch",
        "_install_oldlogprob_backend_batch_postprocess_patch",
        "_install_oldlogprob_training_worker_postprocess_patch",
    ):
        monkeypatch.setattr(oldlogprob_runtime, name, lambda *args: True)
    monkeypatch.setattr(
        oldlogprob_runtime,
        "_put_oldlogprob_hidden_refs",
        lambda payload, batch: payload,
    )
    monkeypatch.setattr(
        oldlogprob_runtime, "_build_selection_context", lambda *args: {}
    )
    monkeypatch.setattr(
        oldlogprob_runtime,
        "_timing_tensor_from_context",
        lambda context, device: torch.zeros(1, 1, device=device),
    )
    monkeypatch.setattr(
        oldlogprob_runtime, "_oldlogprob_aux_layer_ids_from_batch", lambda batch: [0]
    )
    monkeypatch.setattr(
        oldlogprob_runtime,
        "_select_and_merge_concatenated_hidden",
        lambda context, tensors, **kwargs: (
            torch.cat([t.unsqueeze(0) if t.dim() == 2 else t for t in tensors], dim=-1)[
                :, 1:3
            ],
            torch.tensor([True]),
        ),
    )
    assert oldlogprob_runtime.install_oldlogprob_hidden_runtime_patch(
        actor_backend="fsdp2"
    )
    engine = FakeEngine()
    hidden = torch.zeros(1, 4, 2)
    if capture_impl == "forward_hook":
        engine._speco_oldlogprob_hidden_context = {
            "aux_keys": ["aux"],
            "final_key": "final",
            "captures": {"aux": hidden, "final": hidden},
            "handles": [],
        }
    batch = {
        oldlogprob_runtime.OLD_LOGPROB_COLLECT_MASK_KEY: torch.tensor([True]),
        oldlogprob_runtime.OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY: capture_impl,
    }
    if collect_logz:
        batch[OLD_LOGPROB_TARGET_LOGZ_TEMPERATURE_KEY] = lk_temperature
    output = SimpleNamespace(logits=raw_logits.clone(), hidden_states=(hidden, hidden))
    result = engine.prepare_model_outputs(output, {}, batch, None)
    torch.testing.assert_close(
        result["log_probs"], torch.log_softmax(raw_logits.squeeze(0) / temperature, -1)
    )
    if collect_logz:
        logz = result[oldlogprob_runtime.OLD_LOGPROB_TARGET_LOGZ_KEY]
        torch.testing.assert_close(
            logz, torch.logsumexp(raw_logits[:, 1:3] / lk_temperature, -1, keepdim=True)
        )
        target = full_vocab_subset_probs(
            raw_logits[:, 1:3], logz.squeeze(-1), temperature=lk_temperature
        )
        torch.testing.assert_close(target.sum(-1), torch.ones(1, 2))
        draft_logits = torch.zeros_like(target, requires_grad=True)
        loss = negative_log_acceptance_loss(
            torch.softmax(draft_logits / lk_temperature, -1), target
        ).mean()
        loss.backward()
        assert loss.item() < 13.0
        assert draft_logits.grad.abs().max().item() > 0
    else:
        assert oldlogprob_runtime.OLD_LOGPROB_TARGET_LOGZ_KEY not in result
    assert not hasattr(engine, "_speco_oldlogprob_hidden_context")


def test_oldlogprob_object_ref_payload_keeps_fp32_target_logz(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    fake_ray = ModuleType("ray")
    fake_ray.put = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(oldlogprob_runtime, "_BATCH_POSTPROCESS_PATCHED", True)
    monkeypatch.setattr(oldlogprob_runtime, "_POSTPROCESS_PATCHED", True)
    hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
    target_logz = torch.tensor([[[2.25], [3.5]]], dtype=torch.float32)
    micro_batch = {
        "speco_oldlogprob_collect_mask": torch.tensor([True]),
        "speco_oldlogprob_hidden_position_mask": torch.tensor([[True, True]]),
        "speco_oldlogprob_owner_rank": torch.tensor([0]),
        "speco_oldlogprob_hidden_object_ref": True,
    }

    result = _put_oldlogprob_hidden_refs(
        {
            "speco_oldlogprob_hidden_states": hidden,
            "speco_oldlogprob_target_logz": target_logz,
            "speco_oldlogprob_sp_size": 1,
            "speco_oldlogprob_sp_rank": 0,
        },
        micro_batch,
    )

    payload = result["speco_oldlogprob_hidden_chunk_refs"][0]
    assert payload["hidden_states"].dtype == torch.bfloat16
    assert payload["target_logz"].dtype == torch.float32
    torch.testing.assert_close(
        payload["target_logz"].reshape(-1), target_logz.reshape(-1)
    )


def test_row_refs_survive_dynamic_batch_and_dp_concat(monkeypatch):
    import importlib

    torch = pytest.importorskip("torch")
    from verl.utils import tensordict_utils as tu
    from tensordict import TensorDict

    refs_key = oldlogprob_runtime.OLD_LOGPROB_HIDDEN_REFS_KEY
    meta_key = oldlogprob_runtime.OLD_LOGPROB_HIDDEN_REF_META_KEY
    chunk_key = oldlogprob_runtime.OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY

    class Worker:
        def _postprocess_output(self, output):
            return TensorDict(
                {"log_probs": torch.arange(3).reshape(3, 1)}, batch_size=[3]
            )

    module = SimpleNamespace(TrainingWorker=Worker)
    real_import = importlib.import_module

    def fake_import(name):
        if name == "verl.workers.engine_workers":
            return module
        if name == "verl.workers.engine.utils":
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(oldlogprob_runtime.importlib, "import_module", fake_import)
    monkeypatch.setattr(oldlogprob_runtime, "_POSTPROCESS_PATCHED", False)
    assert oldlogprob_runtime._install_oldlogprob_training_worker_postprocess_patch()
    outputs = []
    for offset in (0, 3):
        # Dynamic micro-batches process local rows [2, 0], then [1].
        indices = [[2, 0], [1]]
        micro_outputs = []
        for rows in indices:
            micro_outputs.append(
                {
                    "model_output": {
                        refs_key: [f"ref-{offset + i}" for i in rows],
                        meta_key: [
                            {
                                "source_batch_index": offset + i,
                                "owner": (offset + i) % 2,
                                "chunk_start": i * 2,
                                "chunk_length": 2,
                            }
                            for i in rows
                        ],
                    }
                }
            )
        extracted = oldlogprob_runtime._extract_oldlogprob_non_tensor_model_output(
            micro_outputs, indices
        )
        for key in (refs_key, meta_key):
            extracted[key] = oldlogprob_runtime._restore_non_tensor_dynamic_batch(
                extracted[key], indices
            )
        extracted[chunk_key] = ["worker-local-chunk"]
        outputs.append(Worker()._postprocess_output({"model_output": extracted}))
    merged = tu.concat_tensordict(outputs)
    assert tu.get(merged, refs_key) == [f"ref-{i}" for i in range(6)]
    metas = tu.get(merged, meta_key)
    assert [m["source_batch_index"] for m in metas] == list(range(6))
    assert [m["owner"] for m in metas] == [i % 2 for i in range(6)]
    assert chunk_key not in merged


def test_pp_whole_refs_survive_postprocess_dp_concat_and_row_reordering(monkeypatch):
    import importlib

    ray = pytest.importorskip("ray")
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    runtime = oldlogprob_runtime
    refs_key = runtime.OLD_LOGPROB_HIDDEN_WHOLE_REF_KEY
    meta_key = runtime.OLD_LOGPROB_HIDDEN_WHOLE_REF_META_KEY

    class Worker:
        def _postprocess_output(self, output):
            return TensorDict({"log_probs": torch.zeros(3, 1)}, batch_size=[3])

    real_import = importlib.import_module

    def import_module(name):
        if name == "verl.workers.engine_workers":
            return SimpleNamespace(TrainingWorker=Worker)
        return real_import(name)

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime, "_POSTPROCESS_PATCHED", False)
    assert runtime._install_oldlogprob_training_worker_postprocess_patch()
    outputs = []
    stored = {}
    for shard_index in range(2):
        ref = f"whole-{shard_index}"
        stored[ref] = (
            torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + 100 * shard_index
        )
        outputs.append(
            Worker()._postprocess_output(
                {
                    "model_output": {
                        refs_key: ref,
                        meta_key: {"num_samples": 3, "shape": (3, 2, 4)},
                    }
                }
            )
        )
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    merged = tu.concat_tensordict(outputs)[order]
    assert tu.get(merged, refs_key) == [
        "whole-1",
        "whole-0",
        "whole-1",
        "whole-0",
        "whole-1",
        "whole-0",
    ]
    assert [meta["row_index"] for meta in tu.get(merged, meta_key)] == [
        1,
        1,
        2,
        0,
        0,
        2,
    ]
    calls = []

    def get(refs):
        calls.append(refs)
        return [stored[ref] for ref in refs]

    monkeypatch.setattr(ray, "get", get)
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    hidden_rows = trainer._speco_materialize_oldlogprob_whole_refs(merged)
    torch.testing.assert_close(
        torch.stack(hidden_rows),
        torch.cat([stored["whole-0"], stored["whole-1"]])[order],
    )
    assert calls == [["whole-1", "whole-0"]]
    assert tu.get(merged, refs_key) is None
    assert tu.get(merged, meta_key) is None


def test_hidden_export_preserves_source_owner_and_hidden_slices(monkeypatch):
    ray = pytest.importorskip("ray")
    torch = pytest.importorskip("torch")
    runtime = oldlogprob_runtime
    monkeypatch.setattr(runtime, "_BATCH_POSTPROCESS_PATCHED", True)
    monkeypatch.setattr(runtime, "_POSTPROCESS_PATCHED", True)
    stored = {}

    def put(value):
        ref = f"ref-{len(stored)}"
        stored[ref] = value
        return ref

    monkeypatch.setattr(ray, "put", put)
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    batch = {
        runtime.OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY: True,
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: torch.tensor([True, False, True]),
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: torch.ones(
            3, 2, dtype=torch.bool
        ),
        runtime.OLD_LOGPROB_OWNER_RANK_KEY: torch.tensor([1, -1, 1]),
        runtime.OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY: torch.tensor([205, 11, 93]),
    }
    output = runtime._put_oldlogprob_hidden_refs(
        {runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: hidden}, batch
    )
    refs = output[runtime.OLD_LOGPROB_HIDDEN_REFS_KEY]
    metas = output[runtime.OLD_LOGPROB_HIDDEN_REF_META_KEY]
    assert refs[1] is None and metas[1] is None
    for row, source in ((0, 205), (2, 93)):
        meta = metas[row]
        assert meta["source_batch_index"] == source and meta["owner"] == 1
        start, length = meta["chunk_start"], meta["chunk_length"]
        torch.testing.assert_close(
            stored[refs[row]][start : start + length], hidden[row]
        )
