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
"""Compare with the selected real AIR checkout using synthetic CPU tensors.

Set DRAFT_ENGINE_SOURCE_ROOT to opt in. No model weights or NPU dependencies are
needed; these tests import AIR's actual data/sampler/TP/export implementations.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from _draft_reference import activate_source, engine_module
from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
)
from verl_speco.trainer.deepseek_v4_dataset import StoreDataset, build_loader
from verl_speco.trainer.feature_store import TorchShardFeatureStore


@pytest.fixture(scope="module")
def air():
    root = os.environ.get("DRAFT_ENGINE_SOURCE_ROOT")
    if not root:
        pytest.skip("DRAFT_ENGINE_SOURCE_ROOT is not set")
    activate_source(root)
    return engine_module("training_common.utils.data")


@pytest.fixture
def prepared(air, tmp_path):
    root = Path(os.environ["DRAFT_ENGINE_SOURCE_ROOT"])
    cfg = OmegaConf.load(root / "configs/ds4_dspark_train_smoke.yaml")
    cfg.train.segment_length = 8
    cfg.train.segment_overlap = 2
    cfg.train.segment_min_loss_tokens = 1
    cfg.train.num_workers = 0
    cfg.train.dspark_anchor_min_spacing = 2
    cfg.train.dspark_num_anchors = 2
    cfg.deepspeed.train_micro_batch_size_per_gpu = 1
    model = tmp_path / "model"
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
    cfg.model.path = str(model)
    contract = DeepSeekV4Contract.from_model(model)
    store = TorchShardFeatureStore(
        tmp_path / "store",
        max_samples_per_shard=1,
        metadata={"ds4_contract": contract.identity()},
    )
    masks = {}
    capture_root = tmp_path / "capture"
    for i in range(4):
        size = 11 + i * 3
        key = f"request_{i:06d}"
        tensors = {
            "input_ids": torch.arange(size, dtype=torch.int64),
            "positions": torch.arange(50, 50 + size, dtype=torch.int64),
            "hidden_states": torch.randn(size, 8),
            "target_head_hidden_states": torch.randn(size, 8),
        }
        raw = [1, 1] + [0] * (size - 2)
        sample = from_engine_tensors(
            tensors, [1 - value for value in raw], contract, sample_id=key
        )
        store.write_many([sample])
        directory = capture_root / key
        directory.mkdir(parents=True)
        for name, value in tensors.items():
            torch.save(value, directory / f"step_0_{name}.cpu.pt")
        masks[key] = raw
    store.close()
    original = air.MTPDataset(
        str(capture_root),
        masks,
        __import__("logging").getLogger(__name__),
        segment_length=8,
        segment_overlap=2,
        segment_min_loss_tokens=1,
        ttt_length=1,
        required_tensor_keys=(
            "input_ids",
            "hidden_states",
            "positions",
            "target_head_hidden_states",
        ),
    )
    dataset = StoreDataset(str(tmp_path / "store"), contract, cfg)
    return cfg, dataset, original


def test_store_segments_match_real_air_dataset(prepared):
    _, dataset, original = prepared
    assert dataset.sequence_lengths == original.sequence_lengths
    assert dataset.source_ids == original.source_ids
    for index in range(len(dataset)):
        actual, expected = dataset[index], original[index]
        for field in actual:
            if field == "loss_mask":
                assert torch.equal(actual[field], 1 - expected["raw_mask"]), index
            else:
                assert torch.equal(actual[field], expected[field]), (index, field)


def test_speco_padding_and_anchor_capacity_match_supervised_tokens(air, prepared):
    from verl_speco.trainer.vllm_ascend_mask import collate_features

    _, dataset, original = prepared
    batch = collate_features([dataset[0], dataset[-1]], pad_to_length=12)
    for row, index in enumerate((0, len(dataset) - 1)):
        length = dataset[index]["input_ids"].numel()
        assert torch.equal(
            batch["loss_mask"][row, :length], dataset[index]["loss_mask"]
        )
        assert not bool(batch["loss_mask"][row, length:].any())
    assert batch["raw_mask"] is batch["loss_mask"]
    expected = [
        air.estimate_dspark_anchor_capacity(
            original.samples[segment.sample_index]["mask"],
            segment,
            num_anchors=3,
            min_anchor_spacing=2,
        )
        for segment in original.segments
    ]
    assert dataset.anchor_capacities(num_anchors=3, spacing=2) == expected


def test_speco_model_preparation_reuses_native_targets_without_mask_inversion(air):
    from dataclasses import fields
    from types import SimpleNamespace

    from verl_speco.trainer.vllm_ascend_mask import install_speco_mask

    api = engine_module("training_common.modeling.ds4_dspark")
    model = SimpleNamespace(
        _training_config=SimpleNamespace(dspark_anchor_min_spacing=2),
        runtime_contract=SimpleNamespace(
            block_size=2, target_offset=2, noise_token_id=31
        ),
        prepare_training_inputs=lambda **kwargs: None,
    )
    ids = torch.arange(10).unsqueeze(0)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 1, 0, 0]], dtype=torch.float32)
    positions = ids + 100
    original = mask.clone()
    install_speco_mask(model)
    actual, runtime = model.prepare_training_inputs(
        input_ids=ids,
        raw_mask=mask,
        positions=positions,
        num_anchors=3,
        generator=torch.Generator().manual_seed(42),
    )
    expected = api.build_block_targets(
        ids,
        mask,
        block_size=2,
        num_anchors=3,
        target_offset=2,
        min_anchor_spacing=2,
        generator=torch.Generator().manual_seed(42),
    )
    expected_runtime = api.build_runtime_block_inputs(
        ids, expected, noise_token_id=31, positions=positions
    )
    for result, reference in ((actual, expected), (runtime, expected_runtime)):
        for field in fields(reference):
            a, b = getattr(result, field.name), getattr(reference, field.name)
            if torch.is_tensor(b):
                assert torch.equal(a, b), field.name
            else:
                assert a == b, field.name
    assert torch.equal(mask, original)


@pytest.mark.parametrize(
    "buckets,locality", [(True, False), (False, True), (False, False)]
)
def test_tp_replicas_share_batch_order_and_dp_uses_air_sampler(
    air, prepared, buckets, locality
):
    cfg, dataset, _ = prepared
    cfg.optimizations.common.lazy_segment_source_locality = locality
    cfg.train.length_bucket_multiplier = 2 if buckets else 0
    loaders = []
    for global_rank in range(4):
        coordinates = engine_module(
            "training_common.utils.parallel"
        ).parallel_coordinates(global_rank, 4, 2)
        sampler, loader = build_loader(
            dataset,
            cfg,
            dp_rank=coordinates.data_parallel_rank,
            dp_size=coordinates.data_parallel_size,
        )
        sampler.set_epoch(3)
        loaders.append(list(loader))
    assert len(loaders[0]) == len(loaders[2]) > 0
    for a, b in ((0, 1), (2, 3)):
        for left, right in zip(loaders[a], loaders[b], strict=True):
            assert left.keys() == right.keys()
            assert all(torch.equal(left[key], right[key]) for key in left)
    assert any(
        not torch.equal(a["input_ids"], b["input_ids"])
        for a, b in zip(loaders[0], loaders[2], strict=True)
    )


def test_real_air_exporter_imports_without_npu(air):
    exporter = engine_module("loading_trained_mtp.loading_ds4")
    assert callable(exporter.export_patched_model)
    assert callable(exporter.verify_patched_model_weights)


def test_no_training_runtime_is_imported_by_data_path(air):
    import sys

    assert "torch_npu" not in sys.modules
    assert "deepspeed" not in sys.modules


def test_native_launcher_builds_opt_in_command_without_native_config():
    from verl_speco.draft_train_launcher import (
        build_torch_distributed_command,
        normalize_training_args,
        resolve_launch_config,
    )

    args = ["speco.draft_training.nproc_per_node=8", "draft_engine.action=train"]
    config = resolve_launch_config(args)
    command = build_torch_distributed_command(
        config, normalize_training_args(args, config)
    )
    assert "verl_speco.draft_train" in command
    assert "--nproc_per_node=8" in command


def test_smoke_script_uses_existing_launcher_without_loading_models(tmp_path):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "DRAFT_ENGINE_SOURCE_ROOT": str(tmp_path / "AIR source"),
        "MODEL_PATH": str(tmp_path / "0731 model"),
        "FEATURE_STORE": str(tmp_path / "features"),
        "PYTHON_BIN": sys.executable,
    }
    result = subprocess.run(
        [
            "bash",
            str(
                repo
                / "examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh"
            ),
            "train",
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "torch.distributed.run" in result.stdout
    assert "-m verl_speco.draft_train" in result.stdout
    assert "--nproc_per_node=8" in result.stdout


def test_full_preflight_and_numerical_config_guards(prepared, tmp_path):
    from verl_speco.backends.training_backend import run_standalone_training
    from verl_speco.integration.vllm_ascend_config import load_vllm_ascend_config

    cfg, dataset, _ = prepared
    repo = Path(__file__).resolve().parents[2]
    config = OmegaConf.load(
        repo / "verl_speco/config/deepseek_v4_0731_draft_trainer.yaml"
    )
    config.draft_engine.source_root = os.environ["DRAFT_ENGINE_SOURCE_ROOT"]
    config.actor_rollout_ref.model.path = cfg.model.path
    config.actor_rollout_ref.rollout.drafter.training.feature_store.path = str(
        dataset.store.path
    )
    config.draft_engine.overrides = {
        "train": OmegaConf.to_container(cfg.train),
        "model": {"vllm_tensor_parallel_size": 2},
        "deepspeed": {"train_micro_batch_size_per_gpu": 1},
    }
    config.speco.draft_training.nproc_per_node = 4
    config.draft_engine.output_dir = str(tmp_path / "report")
    result = run_standalone_training(config)
    assert result["status"] == "passed"
    assert result["data_parallel_size"] == 2 and result["tensor_parallel_size"] == 2
    assert result["hardware_training_validated"] is False
    saved = json.loads((tmp_path / "report/preflight.json").read_text())
    assert saved["contract"]["target_revision"] == "0731"
    metadata_path = dataset.store.metadata_path
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"source": "verl_rollout", "runtime_target_version": 7})
    metadata_path.write_text(json.dumps(metadata))
    config.draft_engine.output_dir = str(tmp_path / "rollout_report")
    with pytest.raises(ValueError, match="target_runtime_version"):
        run_standalone_training(config)
    config.draft_engine.target_runtime_version = 7
    assert run_standalone_training(config)["target_runtime_version"] == 7
    config.draft_engine.overrides["deepspeed"]["zero_optimization"] = {"stage": 2}
    with pytest.raises(ValueError, match="ZeRO-1"):
        load_vllm_ascend_config(config)
