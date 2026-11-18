# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone SpeCo installation and execution without an AIR checkout."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from verl_speco.integration.vllm_ascend_implementation import (
    LEGACY_IMPLEMENTATION_FINGERPRINT,
    implementation_fingerprint,
    implementation_identity_matches,
    implementation_paths,
)
from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
)
from verl_speco.trainer.feature_store import TorchShardFeatureStore

ROOT = Path(__file__).resolve().parents[2]


def test_native_runtime_imports_without_external_namespaces():
    script = """
import importlib.abc, sys
class NoExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'AIRDrafter', 'training_common', 'offline_training', 'tuner'}:
            raise AssertionError('External source import attempted: ' + fullname)
sys.meta_path.insert(0, NoExternal())
from verl_speco.backends import vllm_ascend_api
from verl_speco.models.deepseek_v4 import modeling_dspark
from verl_speco.models.deepseek_v4 import export
assert vllm_ascend_api.run_dspark_training.__module__.startswith('verl_speco.')
assert modeling_dspark._RUNTIME_IMPORT_ERROR is None
assert callable(export.export_patched_model)
assert not any(name.startswith('AIRDrafter') for name in sys.modules)
print('native imports OK')
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DRAFT_ENGINE_SOURCE_ROOT", "PYTHONPATH"}
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "native imports OK" in result.stdout


def test_implementation_has_no_imports_from_external_source_tree():
    for path in implementation_paths():
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    (
                        "AIRDrafter",
                        "training_common",
                        "offline_training",
                        "co_training",
                        "tuner",
                    )
                ), path
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith(
                        (
                            "AIRDrafter",
                            "training_common",
                            "offline_training",
                            "co_training",
                            "tuner",
                        )
                    )
                    for alias in node.names
                ), path
    assert all(path.is_file() for path in implementation_paths())


def test_numerical_recovery_requires_exact_implementation_identity():
    current = {
        "source_fingerprint": implementation_fingerprint(),
        "contract": {"revision": "0731"},
        "training_fingerprint": "numerics",
    }
    previous = {**current, "source_fingerprint": LEGACY_IMPLEMENTATION_FINGERPRINT}
    assert implementation_identity_matches(dict(current), current)
    assert not implementation_identity_matches(None, None)
    assert not implementation_identity_matches(previous, current)
    assert not implementation_identity_matches(
        {**previous, "training_fingerprint": "different"}, current
    )
    assert not implementation_identity_matches(
        {**previous, "source_fingerprint": "unknown-external-source"}, current
    )


def test_offline_cli_preflight_without_source_checkout(tmp_path):
    model = tmp_path / "0731 model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 5,
                "dspark_target_layer_ids": [0, 2],
                "dspark_block_size": 2,
            }
        )
    )
    contract = DeepSeekV4Contract.from_model(model)
    store_path = tmp_path / "features"
    store = TorchShardFeatureStore(
        store_path, metadata={"ds4_contract": contract.identity()}
    )
    for index in range(4):
        sample = from_engine_tensors(
            {
                "input_ids": torch.arange(8, dtype=torch.int64),
                "positions": torch.arange(8, dtype=torch.int64),
                "hidden_states": torch.ones(8, 8),
                "target_head_hidden_states": torch.ones(8, 8),
            },
            [0, 0, 1, 1, 1, 1, 1, 1],
            contract,
            sample_id=str(index),
        )
        store.write_many([sample])
    store.close()
    output = tmp_path / "preflight"
    args = [
        sys.executable,
        "-m",
        "verl_speco.draft_train",
        "--config-name=deepseek_v4_0731_draft_trainer",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.rollout.drafter.training.feature_store.path={store_path}",
        f"draft_engine.output_dir={output}",
        "draft_engine.action=preflight",
        "draft_engine.require_runtime=false",
        "++draft_engine.overrides.model.vllm_tensor_parallel_size=1",
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DRAFT_ENGINE_SOURCE_ROOT",
            "MTP_TENSOR_PARALLEL_SIZE",
            "WORLD_SIZE",
            "PYTHONPATH",
        }
    }
    subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    report = json.loads((output / "preflight.json").read_text())
    assert report["status"] == "passed"
    assert report["source_samples"] == 4
    assert report["source_fingerprint"] == implementation_fingerprint()
    assert report["mask_semantics"] == "speco_1_supervised_0_ignored"


def test_old_source_path_is_no_longer_a_runtime_dependency(tmp_path):
    from verl_speco.integration.deepseek_v4_online import execution_config

    cfg = OmegaConf.create(
        {
            "model": {"path": str(tmp_path)},
            "rollout": {
                "drafter": {
                    "training": {
                        "vllm_ascend": {
                            "work_dir": str(tmp_path / "work"),
                            "total_optimizer_steps": 10,
                        }
                    }
                }
            },
        }
    )
    resolved = execution_config(cfg)
    assert resolved.draft_engine.source_root is None
    assert resolved.draft_engine.config_path is None


def test_native_model_accepts_speco_mask_directly():
    from types import SimpleNamespace

    from verl_speco.models.deepseek_v4.modeling_dspark import (
        DeepseekV4DSparkServingWrapper,
        build_block_targets,
    )

    model = SimpleNamespace(
        _training_config=SimpleNamespace(dspark_anchor_min_spacing=2),
        runtime_contract=SimpleNamespace(
            block_size=2, target_offset=2, noise_token_id=63
        ),
    )
    ids = torch.arange(10).reshape(1, 10)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]])
    positions = ids + 50
    actual, _ = DeepseekV4DSparkServingWrapper.prepare_training_inputs(
        model,
        input_ids=ids,
        raw_mask=mask,
        positions=positions,
        num_anchors=2,
        generator=torch.Generator().manual_seed(7),
    )
    expected = build_block_targets(
        ids,
        mask,
        block_size=2,
        num_anchors=2,
        target_offset=2,
        min_anchor_spacing=2,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(actual.anchor_positions, expected.anchor_positions)
    assert torch.equal(actual.eval_mask, expected.eval_mask)


def test_target_head_loader_reuses_speco_index_and_rejects_duplicate_keys(tmp_path):
    import pytest
    from safetensors.torch import save_file

    from verl_speco.models.target.deepseek_v4_head import _find_safetensor_weights

    weights = {
        "model.hc_head_fn": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        "model.norm.weight": torch.ones(4),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, shard)})
    )
    candidates = {"hc": ("model.hc_head_fn",), "norm": ("model.norm.weight",)}
    actual, selected = _find_safetensor_weights(str(tmp_path), candidates)
    assert selected == {"hc": "model.hc_head_fn", "norm": "model.norm.weight"}
    assert torch.equal(actual["hc"], weights["model.hc_head_fn"])
    save_file(
        {"model.hc_head_fn": weights["model.hc_head_fn"]},
        tmp_path / "duplicate.safetensors",
    )
    with pytest.raises(ValueError, match="Duplicate target output weight"):
        _find_safetensor_weights(str(tmp_path), candidates)
