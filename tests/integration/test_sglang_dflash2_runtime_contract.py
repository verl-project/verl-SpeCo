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
"""SGLang co-training contract for the DFlash2 drafter (served through DFLASH)."""

from __future__ import annotations

import json
import sys
import types

import pytest

from verl_speco.integration import sglang_runtime
from verl_speco.integration.sglang_runtime import (
    _assert_sglang_supports_dflash2,
    _drafter_uses_dflash_aux_hidden,
    _is_sglang_draft_model,
    _server_args_overrides_from_drafter,
    _validate_sglang_dflash2_block_size,
)

_SUPPORTED_FIELDS = {
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "enable_return_hidden_states",
    "enable_weights_cpu_backup",
    "enable_draft_weights_cpu_backup",
}

_DFLASH2_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "dflash_config": {
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "mask_token_id": 151669,
        "target_layer_ids": [0, 8, 16, 24, 32],
    },
}


def _write_drafter(tmp_path):
    model_path = tmp_path / "dflash2-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(_DFLASH2_CONFIG), encoding="utf-8"
    )
    return model_path


def _drafter(model_path, **overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DFLASH2",
        "model_path": str(model_path),
        "rollout": {"spec_steps": 1, "spec_topk": 1, "spec_verify_tokens": 8},
        "training": {
            "dflash2_block_size": 8,
            "collect_hidden_states_from_old_logprob": True,
        },
    }
    config.update(overrides)
    return config


@pytest.fixture
def sglang_has_dflash2(monkeypatch):
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: True)


def test_dflash2_maps_to_the_dflash_server_algorithm(
    tmp_path, sglang_has_dflash2
) -> None:
    overrides = _server_args_overrides_from_drafter(
        _drafter(_write_drafter(tmp_path)), _SUPPORTED_FIELDS
    )

    assert overrides["speculative_algorithm"] == "DFLASH"
    assert overrides["speculative_num_draft_tokens"] == 8
    # sglang rejects return_hidden_states for the DFLASH worker; DFlash2 runs
    # collect hidden states from the old-logprob pass instead.
    assert overrides["enable_return_hidden_states"] is False


def test_dflash2_rejects_engine_hidden_state_collection(
    tmp_path, sglang_has_dflash2
) -> None:
    """sglang rejects return_hidden_states for DFLASH, so fail in the overlay."""
    with pytest.raises(ValueError, match="collect_hidden_states_from_old_logprob"):
        _server_args_overrides_from_drafter(
            _drafter(
                _write_drafter(tmp_path),
                training={
                    "dflash2_block_size": 8,
                    "collect_hidden_states_from_sgl": True,
                },
            ),
            _SUPPORTED_FIELDS,
        )


def test_dflash2_refuses_an_sglang_without_the_draft_class(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: False)

    with pytest.raises(ValueError, match="sglang main"):
        _server_args_overrides_from_drafter(
            _drafter(_write_drafter(tmp_path)), _SUPPORTED_FIELDS
        )


def test_dflash2_capability_probe_is_skipped_without_sglang(monkeypatch) -> None:
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: None)
    _assert_sglang_supports_dflash2()


def test_dflash2_capability_probe_reads_the_installed_sglang(monkeypatch) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert sglang_runtime._sglang_supports_dflash2() is None

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    for parent in ("sglang", "sglang.srt", "sglang.srt.models"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    dflash_module = types.ModuleType(sglang_runtime._SGLANG_DFLASH_MODULE)
    monkeypatch.setitem(
        sys.modules, sglang_runtime._SGLANG_DFLASH_MODULE, dflash_module
    )
    assert sglang_runtime._sglang_supports_dflash2() is False

    dflash_module.DFlash2DraftModel = object
    assert sglang_runtime._sglang_supports_dflash2() is True


def test_dflash2_rejects_a_plain_dflash_checkpoint(
    tmp_path, sglang_has_dflash2
) -> None:
    model_path = tmp_path / "dflash-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["DFlashDraftModel"]}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="DFlash2 drafter checkpoint"):
        _server_args_overrides_from_drafter(_drafter(model_path), _SUPPORTED_FIELDS)


def test_dflash2_block_size_equals_sglang_num_draft_tokens(
    tmp_path, sglang_has_dflash2
) -> None:
    """SGLang uses num_draft_tokens AS the block size (no bonus-token offset)."""
    with pytest.raises(ValueError, match="spec_verify_tokens=7 but block_size=8"):
        _server_args_overrides_from_drafter(
            _drafter(
                _write_drafter(tmp_path),
                rollout={"spec_steps": 1, "spec_topk": 1, "spec_verify_tokens": 7},
            ),
            _SUPPORTED_FIELDS,
        )


def test_dflash2_block_size_falls_back_to_the_checkpoint(tmp_path) -> None:
    drafter = _drafter(_write_drafter(tmp_path), training={})
    _validate_sglang_dflash2_block_size(drafter, 8)
    with pytest.raises(ValueError, match="block_size=8"):
        _validate_sglang_dflash2_block_size(drafter, 4)
    # Training config wins; nothing to compare against is accepted.
    _validate_sglang_dflash2_block_size(
        {**drafter, "training": {"dflash2_block_size": 4}}, 4
    )
    _validate_sglang_dflash2_block_size({"model_path": None, "training": {}}, 4)
    _validate_sglang_dflash2_block_size(drafter, None)


def test_dflash2_uses_the_dflash_aux_hidden_layout() -> None:
    assert _drafter_uses_dflash_aux_hidden(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": "DFLASH2",
            "training": {"collect_hidden_states_from_sgl": True},
        }
    )
    assert not _drafter_uses_dflash_aux_hidden(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": "DFLASH2",
            "training": {"collect_hidden_states_from_sgl": True, "use_logits": True},
        }
    )


class _FakeConfig:
    def __init__(self, architectures=None, draft_vocab_size=None):
        self.architectures = architectures or []
        self.draft_vocab_size = draft_vocab_size


def test_draft_model_detection_covers_the_dflash_family() -> None:
    """SGLang's DFlash draft classes spell neither "eagle" nor draft_vocab_size.

    Without the dflash checks the draft weight loader silently dropped every
    published DFlash-family tensor.
    """

    class DFlash2DraftModel:
        config = _FakeConfig(architectures=["DFlash2DraftModel"])

    class DFlashDraftModel:
        config = _FakeConfig(architectures=["DFlashDraftModel"])

    class EagleDraft:
        config = _FakeConfig(architectures=["Eagle3LlamaForCausalLM"])

    class Qwen3ForCausalLM:
        config = _FakeConfig(architectures=["Qwen3ForCausalLM"])

    assert _is_sglang_draft_model(DFlash2DraftModel())
    assert _is_sglang_draft_model(DFlashDraftModel())
    assert _is_sglang_draft_model(EagleDraft())
    assert not _is_sglang_draft_model(Qwen3ForCausalLM())
