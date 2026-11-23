# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reloads keep the derived storage already captured by graphs."""

import pytest

from verl_speco.integration.vllm_runtime import SpecoVLLMColocateWorkerExtension


@pytest.mark.parametrize("existing_buffers", [False, True])
@pytest.mark.parametrize("norm_layout", ["tensor", "list", "tuple"])
def test_checkpoint_loader_internal_rebuild_keeps_captured_storage(
    monkeypatch, tmp_path, existing_buffers, norm_layout
):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_value = {"value": 4.0}
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(
        safetensors_torch, "load_file",
        lambda *args, **kwargs: {"weight": torch.full((2, 3), checkpoint_value["value"])},
    )

    class Inner:
        value = 1.0

        def _build_fused_kv_buffers(self):
            self._fused_kv_weight = torch.cat([torch.full((1, 3), self.value)] * 2)
            self._fused_kv_bias = torch.full((2,), self.value + 1)
            norms = [torch.full((3,), self.value + 2), torch.full((3,), self.value + 3)]
            if norm_layout == "tensor":
                self._k_norm_weights = torch.stack(norms)
            elif norm_layout == "tuple":
                self._k_norm_weights = tuple(norms)
            else:
                self._k_norm_weights = norms

    class Draft:
        def __init__(self):
            self.model = Inner()

        def load_weights(self, weights):
            # Native Qwen3DSparkForCausalLM.load_weights also rebuilds internally.
            self.model.value = dict(weights)["weight"][0, 0].item()
            self.model._build_fused_kv_buffers()

    draft = Draft()
    if existing_buffers:
        draft.model._build_fused_kv_buffers()
    extension = SpecoVLLMColocateWorkerExtension()
    extension._speco_is_dflash_draft = lambda: True
    extension._speco_resolve_draft_model = lambda: (draft, None)
    extension._speco_get_draft_checkpoint_path = lambda: str(tmp_path)
    names = ("_fused_kv_weight", "_fused_kv_bias", "_k_norm_weights")

    def tensors(value):
        return value if isinstance(value, (list, tuple)) else [value]

    for expected_value in (4.0, 5.0):
        checkpoint_value["value"] = expected_value
        old = {name: getattr(draft.model, name) for name in names if hasattr(draft.model, name)}
        old_pointers = {name: [t.data_ptr() for t in tensors(value)] for name, value in old.items()}
        assert extension._speco_reload_draft_from_checkpoint() == 1
        for name, value in old.items():
            assert getattr(draft.model, name) is value
            assert [t.data_ptr() for t in tensors(value)] == old_pointers[name]
        assert torch.all(draft.model._fused_kv_weight == expected_value)
        assert torch.all(draft.model._fused_kv_bias == expected_value + 1)
        for index, norm in enumerate(tensors(draft.model._k_norm_weights)):
            if norm_layout == "tensor":
                assert torch.all(norm[0] == expected_value + 2)
                assert torch.all(norm[1] == expected_value + 3)
            else:
                assert torch.all(norm == expected_value + index + 2)
        assert extension._speco_draft_weight_source == "checkpoint"
