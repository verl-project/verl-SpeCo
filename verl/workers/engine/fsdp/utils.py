# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.device import get_device_name, is_npu_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class RolloutParallelLayout:
    infer_tp: int
    infer_pp: int
    rollout_world_size: int
    num_replicas: int
    replica_training_ranks: list[list[int]]


def apply_npu_fsdp_patches():
    """Apply NPU patches for FSDP backend if NPU is available."""
    if is_npu_available:
        try:
            import verl.models.transformers.npu_patch  # noqa

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info("Applied NPU patches for FSDP backend")
        except Exception as e:
            logger.warning(f"Failed to apply NPU patches: {e}")


def create_device_mesh(world_size, fsdp_size):
    """
    Create a device mesh for distributed training based on the world size and FSDP size.

    Args:
        world_size (int): Total number of processes in the distributed training setup.
        fsdp_size (int): Size of the Fully Sharded Data Parallel (FSDP) group.

    Returns:
        torch.distributed.device_mesh.DeviceMesh: The initialized device mesh.
    """
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def build_rollout_parallel_layout(
    world_size: int,
    rollout_tp: int,
    rollout_dp: int,
    rollout_pp: int,
) -> RolloutParallelLayout:
    infer_tp = int(rollout_tp) * int(rollout_dp)
    infer_pp = int(rollout_pp)
    rollout_world_size = infer_tp * infer_pp
    if rollout_world_size <= 0:
        raise ValueError(
            "rollout_world_size must be positive: "
            f"rollout_tp={rollout_tp}, rollout_dp={rollout_dp}, rollout_pp={rollout_pp}"
        )
    if world_size % rollout_world_size != 0:
        raise ValueError(
            "world_size must be divisible by rollout replica world size: "
            f"world_size={world_size}, rollout_world_size={rollout_world_size}"
        )

    num_replicas = world_size // rollout_world_size
    replica_training_ranks = []
    for replica_rank in range(num_replicas):
        replica_base = replica_rank * rollout_world_size
        replica_training_ranks.append([replica_base + tp_rank * infer_pp for tp_rank in range(int(rollout_tp))])

    return RolloutParallelLayout(
        infer_tp=infer_tp,
        infer_pp=infer_pp,
        rollout_world_size=rollout_world_size,
        num_replicas=num_replicas,
        replica_training_ranks=replica_training_ranks,
    )


def build_drafter_training_device_mesh(device_type: str, layout: RolloutParallelLayout) -> DeviceMesh:
    return DeviceMesh(
        device_type=device_type,
        mesh=torch.tensor(layout.replica_training_ranks, dtype=torch.int64),
        mesh_dim_names=("dp", "sp"),
    )


def get_sharding_strategy(device_mesh):
    """
    Determine the appropriate sharding strategy based on the number of dimensions of the device mesh.

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): The device mesh used for distributed training.

    Returns:
        torch.distributed.fsdp.ShardingStrategy: The sharding strategy to be used with FSDP.

    Raises:
        NotImplementedError: If the number of dimensions of the device mesh is neither 1 nor 2.
    """
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
