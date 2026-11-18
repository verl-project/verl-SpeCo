# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Actual format readers and native-loader dispatch; no simulated NPU math."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from verl_speco.backends.vllm_ascend_runtime import load_initial_draft_weights
from verl_speco.models.deepseek_v4 import artifact_loading as loading
from verl_speco.models.deepseek_v4.artifact import save_draft_artifact


@pytest.fixture
def sources(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    config = {"model_type": "deepseek_v4", "n_mtp_layers": 1, "hidden_size": 2}
    (base / "config.json").write_text(json.dumps(config))
    weights = {
        "embed.weight": torch.ones(3, 2),
        "head.weight": torch.full((3, 2), 2.0),
        "mtp.0.confidence_head.proj.weight": torch.full((1, 2), 3.0),
    }
    save_file(weights, str(base / "model.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in weights}})
    )
    return base, config


class ContractValue(int):
    @classmethod
    def from_config(cls, config):
        return cls(config.hidden_size)


def wrapper():
    model = torch.nn.Module()
    for name in ["draft", "embed", "head", "confidence"]:
        model.register_parameter(name, torch.nn.Parameter(torch.zeros(1)))
    destinations = {
        "mtp.0.weight": "draft",
        "embed.weight": "embed",
        "head.weight": "head",
        "mtp.0.confidence_head.proj.weight": "confidence",
    }
    consumed = {}

    def load(weights):
        for name, value in weights:
            consumed[name] = value.clone()
        return {destinations[key] for key in consumed}

    model.load_weights = load
    value = SimpleNamespace(
        model=model,
        runtime_contract=ContractValue(2),
        _weights_postprocessed=False,
        vllm_config=None,
        _set_current_vllm_config=lambda _: nullcontext(),
        _keep_hc_parameters_float32=lambda: None,
        freeze_shared_token_weights=lambda: None,
        _ensure_vllm_ascend_tp_equivalent=lambda **kw: None,
    )
    return value, consumed


@pytest.mark.parametrize("serialization", ["safetensors", "pytorch"])
def test_both_readers_use_native_loader_and_explicit_frozen_base(
    sources, tmp_path, monkeypatch, serialization
):
    base, config = sources
    artifact = tmp_path / "artifact"
    save_draft_artifact(
        {"mtp.0.weight": torch.full((2, 2), 7.0, dtype=torch.bfloat16)},
        output=artifact,
        config=config,
        step=1,
        serialization=serialization,
    )
    model, consumed = wrapper()
    finished = []
    monkeypatch.setattr(loading, "_finish_loading", lambda obj: finished.append(obj))
    result = load_initial_draft_weights(
        model,
        OmegaConf.create(
            {"model": {"path": str(base), "initial_draft_artifact": str(artifact)}}
        ),
    )
    assert result["loaded"] == 4 and finished == [model]
    assert consumed["mtp.0.weight"].dtype == torch.bfloat16
    assert consumed["mtp.0.weight"].eq(7).all()
    assert consumed["head.weight"].eq(2).all()
    assert consumed["mtp.0.confidence_head.proj.weight"].eq(3).all()
    assert result["base_fallback"]["embed.weight"] == "embed.weight"


def test_unset_initial_artifact_preserves_existing_full_target_loader():
    calls = []
    model = SimpleNamespace(load_weights=lambda **kwargs: calls.append(kwargs))
    load_initial_draft_weights(model, OmegaConf.create({"model": {"path": "/base"}}))
    assert calls == [{"path": "/base", "mode": "full"}]


def test_initial_artifact_path_does_not_change_optimizer_recovery_recipe():
    from verl_speco.integration.vllm_ascend_config import training_fingerprint

    cfg = OmegaConf.create({"model": {"path": "/base"}, "train": {"epochs": 2}})
    original = training_fingerprint(cfg)
    cfg.model.initial_draft_artifact = "/first_artifact"
    assert training_fingerprint(cfg) == original
    cfg.model.initial_draft_artifact = "/moved_artifact"
    assert training_fingerprint(cfg) == original
    cfg.train.epochs = 3
    assert training_fingerprint(cfg) != original


def test_wrong_base_config_rejected_before_native_load(sources, tmp_path):
    base, config = sources
    artifact = tmp_path / "artifact"
    save_draft_artifact(
        {"mtp.0.weight": torch.ones(2)},
        output=artifact,
        config={**config, "hidden_size": 3},
        step=1,
    )
    model, consumed = wrapper()
    with pytest.raises(ValueError, match="configurations differ"):
        loading.load_standard_draft_artifact(model, artifact, base_checkpoint=base)
    assert not consumed


def test_live_wrapper_cannot_be_reinitialized():
    with pytest.raises(RuntimeError, match="fresh wrapper"):
        loading.load_standard_draft_artifact(
            SimpleNamespace(_weights_postprocessed=True),
            "/absent",
            base_checkpoint="/absent",
        )


def test_unsupported_confidence_bias_is_rejected_before_native_load(sources, tmp_path):
    base, config = sources
    artifact = tmp_path / "artifact"
    save_draft_artifact(
        {"mtp.0.confidence_head.proj.bias": torch.ones(1)},
        output=artifact,
        config=config,
        step=1,
    )
    model, consumed = wrapper()
    with pytest.raises(ValueError, match="confidence schema"):
        loading.load_standard_draft_artifact(model, artifact, base_checkpoint=base)
    assert not consumed


def test_incomplete_native_parameter_coverage_never_postprocesses(
    sources, tmp_path, monkeypatch
):
    base, config = sources
    artifact = tmp_path / "artifact"
    save_draft_artifact(
        {"mtp.0.weight": torch.ones(2)}, output=artifact, config=config, step=1
    )
    model, _ = wrapper()
    model.model.register_parameter("missing", torch.nn.Parameter(torch.ones(1)))
    monkeypatch.setattr(
        loading, "_finish_loading", lambda _: pytest.fail("partial model processed")
    )
    with pytest.raises(ValueError, match="missing"):
        loading.load_standard_draft_artifact(model, artifact, base_checkpoint=base)
    assert model._artifact_load_failed
    with pytest.raises(RuntimeError, match="fresh wrapper"):
        loading.load_standard_draft_artifact(model, artifact, base_checkpoint=base)
