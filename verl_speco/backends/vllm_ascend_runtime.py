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
"""Ascend-only construction, kept out of import and CPU preflight paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os

import torch
from omegaconf import OmegaConf

from verl_speco.backends.vllm_ascend_engine import VllmAscendDSparkEngine
from verl_speco.integration.ascend_runtime_compat import require_ascend_runtime
from verl_speco.trainer.vllm_ascend_mask import install_speco_mask


def runtime_versions(*, require: bool = False) -> dict:
    versions: dict[str, str | None] = {}
    for package in (
        "torch",
        "torch-npu",
        "deepspeed",
        "vllm",
        "vllm-ascend",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if require:
        missing = [name for name, version in versions.items() if version is None]
        if missing:
            raise RuntimeError(f"Missing Ascend backend dependencies: {missing}")
        require_ascend_runtime(versions["vllm"], versions["vllm-ascend"])
    return versions


def initialize_distributed():
    """Use the existing SpeCo launcher's rank environment, without another launcher."""
    import deepspeed
    import torch.distributed as dist
    import torch_npu  # noqa: F401 -- registers torch.npu and HCCL

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.npu.is_available():
        raise RuntimeError(
            "Real DeepSeek V4 forward/backward requires an available Ascend NPU"
        )
    torch.npu.set_device(local_rank)
    owned = not dist.is_initialized()
    if owned:
        # DeepSpeed needs an initialized process group even for TP=DP=1.
        deepspeed.init_distributed(dist_backend="hccl")
    return torch.device(f"npu:{local_rank}"), owned


def load_initial_draft_weights(model, cfg):
    """Opt in to standard draft artifacts; keep the target as the shared base."""
    artifact = cfg.model.get("initial_draft_artifact")
    if artifact:
        if cfg.model.get("type") == "ds4":
            from verl_speco.models.deepseek_v4.mtp_artifact import load_mtp_artifact

            return load_mtp_artifact(model, artifact, base_checkpoint=cfg.model.path)
        from verl_speco.models.deepseek_v4.artifact_loading import (
            load_standard_draft_artifact,
        )

        return load_standard_draft_artifact(
            model, artifact, base_checkpoint=cfg.model.path
        )
    return model.load_weights(path=cfg.model.path, mode="full")


def warmup_metadata_collectives(topology, device):
    """Initialize HCCL before the first 0.27.1 custom AICPU metadata task.

    On the validated runtime, initializing HCCL between custom metadata calls
    invalidates subsequent metadata execution. One scalar AllGather before any
    metadata task avoids that transition, including for later HCCL groups.
    It consumes no RNG and changes no model values or forward operations.
    """
    from packaging.version import Version

    if (
        topology.tensor_parallel_size <= 1
        or Version(importlib.metadata.version("vllm-ascend")).public != "0.27.1rc1"
    ):
        return False
    import torch
    import torch.distributed as dist

    token = torch.zeros(1, dtype=torch.bfloat16, device=device)
    received = torch.empty(
        topology.tensor_parallel_size, dtype=token.dtype, device=device
    )
    dist.all_gather_into_tensor(received, token, group=topology.tensor_parallel_group)
    torch.npu.synchronize(device)
    return True


def build_engine(
    *, cfg, topology, loader, dataset, device, tracker, logger, checkpoint_dir
):
    """Reuse the packaged model, teacher, optimizer and scheduler constructors."""
    import deepspeed

    from verl_speco.backends import vllm_ascend_api as api
    from verl_speco.integration.ascend_state_config import (
        require_deterministic_training_environment,
    )

    if topology.tensor_parallel_size > 1:
        require_deterministic_training_environment()

    threads = int(
        os.environ.get("MTP_OMP_NUM_THREADS", cfg.train.get("omp_num_threads", 2))
    )
    if threads <= 0:
        raise ValueError("MTP_OMP_NUM_THREADS must be positive")
    torch.set_num_threads(threads)
    if cfg.debug.get("seed") is not None:
        api.set_seed(int(cfg.debug.seed))
    api.export_memory_chunking_environment(api.resolve_memory_chunking(cfg))
    padded_length = (
        dataset.segment_length if cfg.train.get("pad_segment_tails", False) else None
    )
    required_tokens = int(cfg.deepspeed.train_micro_batch_size_per_gpu) * max(
        max(dataset.sequence_lengths), int(padded_length or 0)
    )
    if int(cfg.model.vllm_max_num_batched_tokens) < required_tokens:
        raise ValueError(
            f"vllm_max_num_batched_tokens must cover the batch: need {required_tokens}"
        )
    is_mtp = cfg.model.type == "ds4"
    if is_mtp:
        from verl_speco.backends.vllm_ascend_mtp_model import load_mtp_model

        model = load_mtp_model(cfg, logger)
    else:
        model = api.load_mtp_draft_model(cfg, logger)
    if warmup_metadata_collectives(topology, device):
        logger.info("Initialized HCCL before native DSpark metadata execution")
    if not is_mtp:
        install_speco_mask(model)
    load_initial_draft_weights(model, cfg)
    model.freeze_params(cfg)
    api.audit_data_parallel_parameter_replicas(model, topology)
    if (
        not cfg.model.get("vllm_training_dp_replicate_moe", True)
        and topology.data_parallel_size > 1
    ):
        raise ValueError("DS4 TP+DP training requires replicated MoE weights across DP")
    groups = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            groups.setdefault(parameter.dtype, []).append(parameter)
    if not groups:
        raise ValueError("The model exposes no trainable parameters")
    model_parameters = [
        {"params": values, "name": f"mtp_model_{str(dtype).removeprefix('torch.')}"}
        for dtype, values in groups.items()
    ]
    ds_config = OmegaConf.to_container(cfg.deepspeed, resolve=True)
    total_steps = math.ceil(
        len(loader) / int(cfg.deepspeed.gradient_accumulation_steps)
    ) * int(cfg.train.epochs)
    total_steps = int(cfg.train.get("scheduler_total_steps", total_steps))
    warmup = api.resolve_warmup_steps(
        total_steps=total_steps, warmup_ratio=float(cfg.train.warmup_ratio)
    )
    scheduler = api.build_deepspeed_scheduler_config(
        cfg.train.get("scheduler", "cosine"),
        total_steps=total_steps,
        warmup_steps=warmup,
        learning_rate=float(ds_config["optimizer"]["params"]["lr"]),
    )
    if scheduler is None:
        ds_config.pop("scheduler", None)
    else:
        ds_config["scheduler"] = scheduler
    args = argparse.Namespace(
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        deepspeed=True,
        deepspeed_config=None,
    )
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model_parameters,
        config=ds_config,
        mpu=topology,
    )
    optimizations = api.resolve_optimization_config(cfg)
    teacher = api.load_target_lm_head(
        cfg.model.path,
        logger,
        device,
        next(engine.parameters()).dtype,
        parallel=optimizations.parallel_lm_head,
        model_type="ds4",
        tensor_parallel_group=topology.tensor_parallel_group,
        tensor_parallel_rank=topology.tensor_parallel_rank,
        tensor_parallel_size=topology.tensor_parallel_size,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    train_function = api.run_dspark_training
    if is_mtp:
        from verl_speco.trainer.deepseek_v4_mtp_training import run_mtp_training

        train_function = run_mtp_training
    return VllmAscendDSparkEngine(
        model_engine=engine,
        teacher=teacher,
        cfg=cfg,
        topology=topology,
        device=device,
        tracker=tracker,
        logger=logger,
        train_function=train_function,
        checkpoint_dir=checkpoint_dir,
    )
