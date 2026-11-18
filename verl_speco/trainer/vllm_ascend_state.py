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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

import json

# Adapted from training_common/utils/ds_utils.py.
import os

import torch
import torch.distributed as dist

from verl_speco.backends.vllm_ascend_parallel import is_main_process
from verl_speco.trainer.checkpoint_io import (
    bounded_checkpoint_writes,
    checkpoint_io_slot,
)


def load_deepspeed_checkpoint(model_engine, resume_ckpt, logger):
    if resume_ckpt is None:
        return {}

    if os.path.basename(resume_ckpt).startswith("epoch_"):
        load_dir = os.path.dirname(resume_ckpt)
        tag = os.path.basename(resume_ckpt)
    else:
        load_dir = resume_ckpt
        tag = None

    if is_main_process():
        logger.info(
            "Resuming DeepSpeed checkpoint | dir: %s, tag: %s",
            load_dir,
            tag or "latest",
        )

    load_path, client_state = model_engine.load_checkpoint(
        load_dir=load_dir,
        tag=tag,
    )
    if load_path is None:
        raise FileNotFoundError(
            f"Failed to load DeepSpeed checkpoint from {resume_ckpt}"
        )
    from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

    refresh_fp32_weight_caches(model_engine.module)
    if is_main_process():
        logger.info("Successfully loaded checkpoint from %s", load_path)
    return client_state or {}


def save_deepspeed_checkpoint(
    model_engine,
    base_dir,
    tag,
    logger,
    *,
    client_state=None,
    save_full_model_per_rank=True,
    save_individual_model_state=True,
    parallel_topology=None,
):
    rank = model_engine.global_rank
    main_process = rank == 0

    if main_process:
        os.makedirs(base_dir, exist_ok=True)
        logger.info(
            "Saving DeepSpeed checkpoint (tag: %s) to directory: %s...",
            tag,
            base_dir,
        )

    with bounded_checkpoint_writes(model_engine, base_dir):
        success = model_engine.save_checkpoint(
            base_dir,
            tag=tag,
            client_state=client_state or {},
        )
    checkpoint_failure = torch.tensor(
        int(not success),
        dtype=torch.int32,
        device=model_engine.device,
    )
    if dist.is_initialized():
        dist.all_reduce(checkpoint_failure, op=dist.ReduceOp.MAX)
    if checkpoint_failure.item():
        raise RuntimeError(f"DeepSpeed failed to save checkpoint {tag} in {base_dir}")

    if dist.is_initialized():
        dist.barrier()

    if main_process and parallel_topology is not None:
        topology_path = os.path.join(base_dir, tag, "parallel_topology.json")
        topology = parallel_topology.metadata()
        topology.pop("global_rank", None)
        topology.pop("tensor_parallel_rank", None)
        topology.pop("data_parallel_rank", None)
        topology["rank_order"] = "dp_major_tp_minor"
        with open(topology_path, "w", encoding="utf-8") as target:
            json.dump(topology, target, indent=2)

    should_save_full_model = save_individual_model_state and (
        save_full_model_per_rank or rank == 0
    )
    ckpt_dir = os.path.join(base_dir, tag)
    model_state_file = os.path.join(ckpt_dir, f"model_states_rank_{rank}.pt")
    save_error = None
    if should_save_full_model:
        try:
            with checkpoint_io_slot(base_dir):
                state_dict = model_engine.module.state_dict()
                cpu_state_dict = {
                    name: value.detach().cpu() for name, value in state_dict.items()
                }
                torch.save(cpu_state_dict, model_state_file)
                del cpu_state_dict, state_dict
            logger.info(
                "[Rank %d] Individual weights successfully preserved in %s",
                rank,
                model_state_file,
            )
        except Exception as exc:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
            save_error = exc

    if dist.is_initialized():
        failure = torch.tensor(
            int(save_error is not None),
            dtype=torch.int32,
            device=model_engine.device,
        )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if failure.item():
            detail = f": {save_error}" if save_error is not None else " on another rank"
            raise RuntimeError(f"Failed to save full model state{detail}")
        dist.barrier()
    elif save_error is not None:
        raise RuntimeError(
            f"Failed to save full model state: {save_error}"
        ) from save_error
