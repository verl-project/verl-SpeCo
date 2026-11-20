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

import pytest

from verl_speco.draft_train_launcher import (
    build_torch_distributed_command,
    normalize_training_args,
    resolve_launch_config,
    validate_feature_store_launch_config,
)


def test_launcher_resolves_python_friendly_gpu_count_override() -> None:
    config = resolve_launch_config(
        [
            "speco.draft_training.num_gpus_per_node=8",
            "actor_rollout_ref.rollout.drafter.model_path=/draft",
        ]
    )

    assert config.nproc_per_node == "8"
    assert config.nnodes == "1"
    assert config.standalone is True

    command = build_torch_distributed_command(
        config, ["foo=bar"], python_executable="python"
    )

    assert command[:6] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--standalone",
    ]
    assert command[-3:] == ["-m", "verl_speco.draft_train", "foo=bar"]


def test_launcher_normalizes_gpu_count_alias_for_hydra() -> None:
    overrides = [
        "speco.draft_training.num_gpus_per_node=8",
        "actor_rollout_ref.rollout.drafter.training.max_steps=10",
    ]
    config = resolve_launch_config(overrides)

    normalized = normalize_training_args(overrides, config)

    assert "speco.draft_training.num_gpus_per_node=8" not in normalized
    assert "speco.draft_training.nproc_per_node=8" in normalized
    assert "speco.draft_training.nnodes=1" in normalized
    assert "speco.draft_training.standalone=true" in normalized
    assert "actor_rollout_ref.rollout.drafter.training.max_steps=10" in normalized


def test_launcher_resolves_multinode_settings() -> None:
    config = resolve_launch_config(
        [
            "speco.draft_training.nproc_per_node=4",
            "speco.draft_training.nnodes=2",
            "speco.draft_training.node_rank=1",
            "speco.draft_training.master_addr=10.0.0.1",
            "speco.draft_training.master_port=29511",
            "speco.draft_training.standalone=false",
        ]
    )

    command = build_torch_distributed_command(config, [], python_executable="python")

    assert "--nnodes=2" in command
    assert "--nproc_per_node=4" in command
    assert "--node_rank=1" in command
    assert "--master_addr=10.0.0.1" in command
    assert "--master_port=29511" in command
    assert "--standalone" not in command


def test_launcher_uses_explicit_port_without_standalone() -> None:
    config = resolve_launch_config(
        [
            "speco.draft_training.num_gpus_per_node=8",
            "speco.draft_training.master_port=29511",
        ]
    )

    command = build_torch_distributed_command(config, [], python_executable="python")

    assert "--nproc_per_node=8" in command
    assert "--master_port=29511" in command
    assert "--standalone" not in command


def test_launcher_rejects_standalone_multinode() -> None:
    with pytest.raises(ValueError, match="standalone=true requires nnodes=1"):
        resolve_launch_config(
            [
                "speco.draft_training.nnodes=2",
                "speco.draft_training.standalone=true",
            ]
        )






@pytest.mark.parametrize("store_type", ["tq", "ray_reference", "token_replay", "jsonl_token_replay", "jsonl"])
def test_direct_launcher_rejects_streaming_and_token_replay(store_type):
    with pytest.raises(ValueError, match="standalone_tq_training_launcher|vllm_hidden_states_generate"):
        validate_feature_store_launch_config([
            f"actor_rollout_ref.rollout.drafter.training.feature_store.type={store_type}",
        ])


@pytest.mark.parametrize("store_type", ["torch_shard", "vllm_safetensors", "safetensors"])
def test_direct_launcher_keeps_complete_offline_features(store_type):
    validate_feature_store_launch_config([
        f"actor_rollout_ref.rollout.drafter.training.feature_store.type={store_type}",
    ])
