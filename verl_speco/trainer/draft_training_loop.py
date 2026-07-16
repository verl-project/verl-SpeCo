"""Standalone torchrun training loop for SPECO draft models."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import OmegaConf, open_dict
from verl.utils.device import get_device_name, get_torch_device

from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend
from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.draft_dataset import DraftFeatureDataLoader, DraftFeatureDataLoaderConfig
from verl_speco.trainer.feature_store import build_feature_store_from_config

logger = logging.getLogger(__name__)


def run_standalone_draft_training(config) -> dict[str, Any]:
    """Run independent draft training from a feature store."""
    return asyncio.run(_run_standalone_draft_training_async(config))


async def _run_standalone_draft_training_async(config) -> dict[str, Any]:
    rank, local_rank, world_size = _init_distributed()
    draft_config = config.actor_rollout_ref
    drafter_cfg = draft_config.rollout.drafter
    training_cfg = drafter_cfg.training
    feature_store_cfg = training_cfg.feature_store
    if not feature_store_cfg.get("path"):
        raise ValueError("actor_rollout_ref.rollout.drafter.training.feature_store.path is required")
    _disable_standalone_sequence_parallel(draft_config)

    _configure_device(local_rank)
    backend = _build_backend(draft_config)
    training_device_mesh = _build_training_device_mesh(draft_config, world_size)
    trainer = DrafterBaseTrainer(
        config=draft_config,
        world_size=world_size,
        # Standalone ranks form one training replica. Keep rollout_dp_rank at
        # zero on every rank so all ranks participate in optimizer DCP while
        # _is_checkpoint_leader still selects SP rank zero for metadata/model IO.
        rollout_dp_rank=0,
        training_device_mesh=training_device_mesh,
        training_process_group=(
            None
            if training_device_mesh is not None
            else dist.group.WORLD if dist.is_initialized() and world_size > 1 else None
        ),
        data_parallel_process_group=None,
        backend=backend,
    )

    max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 1000)) or 0)
    save_interval = int(training_cfg.get("save_interval_steps", 0) or 0)
    successful_steps = 0
    initial_optimizer_step = 0
    optimizer_step = 0
    attempted_batches = 0
    last_save_result: dict[str, Any] | None = None
    last_saved_step = 0
    store = None
    try:
        activated = await trainer.activate_training_model()
        if not activated:
            raise RuntimeError(f"Failed to activate standalone drafter trainer on rank={rank}")
        initial_optimizer_step = int(trainer.optimizer_steps_total)
        optimizer_step = initial_optimizer_step
        last_saved_step = optimizer_step

        store = build_feature_store_from_config(feature_store_cfg, read_only=True)
        loader = DraftFeatureDataLoader(
            store,
            DraftFeatureDataLoaderConfig(
                batch_size=int(training_cfg.get("batch_size_per_gpu", 4)),
                rank=rank,
                world_size=world_size,
                shuffle=bool(feature_store_cfg.get("shuffle", True)),
                repeat=bool(feature_store_cfg.get("repeat", True)),
                seed=int(training_cfg.get("seed", 0) or 0),
            ),
        )
        for samples in loader:
            if max_steps > 0 and successful_steps >= max_steps:
                break
            attempted_batches += 1
            batch = trainer.prepare_training_batch_from_samples(samples, step=optimizer_step)
            has_batch = batch is not None
            if not _all_ranks_true(has_batch, trainer.runtime_device):
                if rank == 0:
                    logger.warning("Skipping standalone drafter batch: at least one rank has no valid batch")
                continue
            if batch is None:
                continue
            ok = await trainer.training_step_from_batch(batch, optimizer_step)
            if not _all_ranks_true(ok, trainer.runtime_device):
                continue
            successful_steps += 1
            optimizer_step = int(trainer.optimizer_steps_total)
            if optimizer_step <= initial_optimizer_step:
                optimizer_step = initial_optimizer_step + successful_steps
            if save_interval > 0 and optimizer_step % save_interval == 0:
                last_save_result = _save_standalone_checkpoint(trainer, optimizer_step)
                if last_save_result.get("saved"):
                    last_saved_step = optimizer_step
                _barrier()
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0 and optimizer_step != last_saved_step:
            last_save_result = _save_standalone_checkpoint(trainer, optimizer_step, wait=True)
            _barrier()
    finally:
        if store is not None:
            store.close()
        await trainer.cleanup_training(clear_data=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    return {
        "rank": rank,
        "world_size": world_size,
        "attempted_batches": attempted_batches,
        "successful_steps": successful_steps,
        "initial_optimizer_step": initial_optimizer_step,
        "optimizer_steps_total": optimizer_step,
        "last_save": last_save_result,
    }


def _build_backend(draft_config):
    algo = str(draft_config.rollout.drafter.speculative_algorithm).upper()
    if algo == "EAGLE3":
        return Eagle3TrainerBackend(draft_config, draft_config.model)
    if algo == "DFLASH":
        return DFlashTrainerBackend(draft_config, draft_config.model)
    if algo == "DSPARK":
        from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

        return DSparkTrainerBackend(draft_config, draft_config.model)
    raise ValueError(f"Unsupported drafter algorithm {algo!r}; expected EAGLE3, DFLASH or DSPARK")


def _save_standalone_checkpoint(trainer: DrafterBaseTrainer, step: int, *, wait: bool = False) -> dict[str, Any]:
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if callable(save_checkpoint):
        return save_checkpoint(int(step), wait=wait)

    # Keep the small PR #13 test double and older trainer adapters usable.
    checkpoint_dir = getattr(trainer, "checkpoint_dir", None)
    if not checkpoint_dir:
        return {"saved": False, "reason": "missing_checkpoint_dir"}
    checkpoint_path = os.path.join(checkpoint_dir, f"draft_step_{int(step)}")
    pending_full_checkpoint = getattr(trainer, "_pending_full_checkpoint_future", None)
    pending_done = getattr(pending_full_checkpoint, "done", None)
    if callable(pending_done) and not pending_done():
        return {"saved": False, "path": checkpoint_path, "reason": "previous_save_running"}

    save_async = getattr(trainer, "_save_checkpoint_async", None)
    if not callable(save_async):
        return {"saved": False, "path": checkpoint_path, "reason": "unsupported_trainer"}
    future = save_async(int(step))
    if future is not None and wait:
        future.result()
        trainer._pending_full_checkpoint_future = None
    return {
        "saved": future is not None,
        "path": checkpoint_path,
        "reason": "saved" if future is not None and wait else "scheduled" if future is not None else "not_checkpoint_leader",
    }


def _disable_standalone_sequence_parallel(draft_config) -> None:
    rollout_cfg = draft_config.rollout
    rollout_tp_size = int(rollout_cfg.get("tensor_model_parallel_size", 1) or 1)
    if rollout_tp_size <= 1:
        return
    logger.warning(
        "Standalone draft training disables Ulysses sequence parallelism: "
        "actor_rollout_ref.rollout.tensor_model_parallel_size=%s is treated as 1 for offline drafter training",
        rollout_tp_size,
    )
    with open_dict(rollout_cfg):
        rollout_cfg.tensor_model_parallel_size = 1


def _build_training_device_mesh(draft_config, world_size: int) -> DeviceMesh | None:
    if world_size <= 1 or not dist.is_initialized():
        return None
    strategy = str(draft_config.actor.get("strategy", "") if hasattr(draft_config, "actor") else "").lower()
    if strategy != "fsdp2":
        return None
    return DeviceMesh(
        device_type=get_device_name(),
        mesh=torch.arange(world_size, dtype=torch.int64).reshape(1, world_size),
        mesh_dim_names=("dp", "sp"),
    )


def _init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        _configure_device(local_rank)
        device_name = str(get_device_name()).lower()
        if device_name == "npu":
            backend = "hccl"
        elif device_name == "cuda":
            backend = "nccl"
        elif device_name == "cpu":
            backend = "gloo"
        else:
            raise ValueError(f"Unsupported standalone drafter device_name={device_name!r}")
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _configure_device(local_rank: int) -> None:
    device_name = get_device_name()
    device_module = get_torch_device()
    if device_name == "cpu":
        return
    set_device = getattr(device_module, "set_device", None)
    if callable(set_device):
        set_device(int(local_rank))


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return bool(value)
    ready = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    return bool(ready.item())


def log_resolved_config(config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.warning("Resolved SPECO standalone draft trainer config:\n%s", OmegaConf.to_yaml(config))
