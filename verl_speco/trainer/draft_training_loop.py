"""Standalone torchrun training loop for SPECO draft models."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
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

    _configure_device(local_rank)
    backend = _build_backend(draft_config)
    trainer = DrafterBaseTrainer(
        config=draft_config,
        world_size=world_size,
        rollout_dp_rank=rank,
        training_device_mesh=None,
        training_process_group=dist.group.WORLD if dist.is_initialized() and world_size > 1 else None,
        data_parallel_process_group=None,
        backend=backend,
    )

    activated = await trainer.activate_training_model()
    if not activated:
        raise RuntimeError(f"Failed to activate standalone drafter trainer on rank={rank}")

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

    max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 1000)) or 0)
    save_interval = int(training_cfg.get("save_interval_steps", 0) or 0)
    successful_steps = 0
    attempted_batches = 0
    last_save_result: dict[str, Any] | None = None
    try:
        for samples in loader:
            if max_steps > 0 and successful_steps >= max_steps:
                break
            attempted_batches += 1
            batch = trainer.prepare_training_batch_from_samples(samples, step=successful_steps)
            has_batch = batch is not None
            if not trainer._sync_batch_readiness(has_batch):
                if rank == 0:
                    logger.warning("Stopping standalone drafter training: at least one rank has no batch")
                break
            if batch is None:
                continue
            ok = await trainer.training_step_from_batch(batch, successful_steps)
            if not ok:
                continue
            successful_steps += 1
            if save_interval > 0 and successful_steps % save_interval == 0:
                last_save_result = trainer.save_checkpoint(successful_steps, wait=True)
                _barrier()
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0:
            last_save_result = trainer.save_checkpoint(successful_steps, wait=True)
            _barrier()
    finally:
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


def _init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
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


def log_resolved_config(config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.warning("Resolved SPECO standalone draft trainer config:\n%s", OmegaConf.to_yaml(config))
