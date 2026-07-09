from __future__ import annotations

import pytest

from verl_speco.draft_train_launcher import (
    build_torch_distributed_command,
    resolve_launch_config,
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

    command = build_torch_distributed_command(config, ["foo=bar"], python_executable="python")

    assert command[:6] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--standalone",
    ]
    assert command[-3:] == ["-m", "verl_speco.draft_train", "foo=bar"]


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
