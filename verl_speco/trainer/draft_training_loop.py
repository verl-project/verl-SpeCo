"""Standalone torchrun training loop for SPECO draft models."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
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
        rollout_dp_rank=rank,
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
    attempted_batches = 0
    last_save_result: dict[str, Any] | None = None
    last_saved_step = 0
    store = None
    try:
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
                last_save_result = _save_standalone_checkpoint(trainer, successful_steps)
                if _sync_any_rank_saved_checkpoint(last_save_result.get("saved")):
                    last_saved_step = successful_steps
                _barrier()
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0 and successful_steps != last_saved_step:
            last_save_result = _save_standalone_checkpoint(trainer, successful_steps, wait=True)
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
    if not trainer.checkpoint_dir:
        return {"saved": False, "reason": "missing_checkpoint_dir"}

    checkpoint_path = os.path.join(trainer.checkpoint_dir, f"draft_step_{int(step)}")
    pending_full_checkpoint = getattr(trainer, "_pending_full_checkpoint_future", None)
    pending_done = getattr(pending_full_checkpoint, "done", None)
    if callable(pending_done) and not pending_done():
        return {
            "saved": False,
            "path": checkpoint_path,
            "reason": "previous_save_running",
        }

    future = trainer._save_checkpoint_async(int(step))
    if future is not None and wait:
        future.result()
        trainer._pending_full_checkpoint_future = None
        _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
    elif future is not None:
        future.add_done_callback(lambda completed: _rewrite_standalone_block_runtime_config(trainer, checkpoint_path, completed))

    return {
        "saved": future is not None,
        "path": checkpoint_path,
        "reason": "saved" if future is not None and wait else "scheduled" if future is not None else "not_checkpoint_leader",
    }


def _ensure_dict_child(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    config[key] = value
    return value


def _load_source_drafter_config(trainer: DrafterBaseTrainer) -> dict[str, Any] | None:
    model_path = getattr(getattr(getattr(trainer, "config", None), "rollout", None), "drafter", None)
    model_path = getattr(model_path, "model_path", None)
    if not model_path:
        return None
    config_path = os.path.join(os.fspath(model_path), "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load source drafter config %s: %s", config_path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


def _fill_if_missing(dst: dict[str, Any], src: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in src and key not in dst:
            dst[key] = deepcopy(src[key])


def _rewrite_standalone_block_runtime_config(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future=None,
) -> None:
    """Export standalone DFlash/DSpark checkpoints with runtime-facing config.

    The training wrapper saves an internal SpeCo config.  For standalone
    checkpoints we keep the original drafter ``config.json`` as the runtime
    contract and only merge the alias fields needed by vLLM/SGLang.
    """
    backend_type = getattr(getattr(trainer, "backend", None), "model_type", None)
    if backend_type not in {"dflash", "dspark"}:
        return

    if completed_future is not None:
        try:
            completed_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip standalone runtime config rewrite because checkpoint save failed: %s", exc)
            return

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning("Cannot rewrite standalone runtime config: missing %s", config_path)
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot rewrite standalone runtime config %s: %s", config_path, exc)
        return
    if not isinstance(training_config, dict):
        logger.warning("Cannot rewrite standalone runtime config %s: expected object", config_path)
        return

    training_config_path = os.path.join(checkpoint_path, "speco_training_config.json")
    try:
        with open(training_config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Failed to write standalone training config copy %s: %s", training_config_path, exc)

    runtime_config = _load_source_drafter_config(trainer)
    if runtime_config is None:
        runtime_config = deepcopy(training_config)
        logger.warning(
            "Source drafter config is unavailable; standalone checkpoint keeps SpeCo training config as runtime config"
        )

    runtime_config["speco_training_model_type"] = backend_type
    common_alias_keys = ("target_layer_ids", "mask_token_id", "num_context_layers")
    _fill_if_missing(runtime_config, training_config, common_alias_keys)

    dflash_config = _ensure_dict_child(runtime_config, "dflash_config")
    _fill_if_missing(dflash_config, training_config, common_alias_keys)

    if backend_type == "dspark":
        dspark_config = _ensure_dict_child(runtime_config, "dspark_config")
        _fill_if_missing(
            dspark_config,
            training_config,
            (
                "block_size",
                "num_anchors",
                "markov_rank",
                "markov_head_type",
                "confidence_head_alpha",
                "confidence_head_with_markov",
                "ce_loss_alpha",
                "l1_loss_alpha",
                "loss_decay_gamma",
                "target_layer_ids",
                "num_context_layers",
                "num_target_layers",
                "target_num_hidden_layers",
                "mask_token_id",
            ),
        )
    else:
        dspark_config = {}

    target_layer_ids = (
        runtime_config.get("target_layer_ids")
        or dflash_config.get("target_layer_ids")
        or dspark_config.get("target_layer_ids")
    )
    if target_layer_ids is not None and "eagle_aux_hidden_state_layer_ids" not in runtime_config:
        try:
            runtime_config["eagle_aux_hidden_state_layer_ids"] = [int(layer_id) + 1 for layer_id in target_layer_ids]
        except (TypeError, ValueError):
            logger.warning("Invalid target_layer_ids in standalone exported config: %r", target_layer_ids)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        logger.warning("Failed to write standalone runtime config %s: %s", config_path, exc)


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


def _sync_any_rank_saved_checkpoint(saved: Any) -> bool:
    if not dist.is_initialized():
        return bool(saved)
    device_name = get_device_name()
    if device_name == "cpu":
        device = torch.device("cpu")
    else:
        current_device = getattr(get_torch_device(), "current_device", None)
        device_index = current_device() if callable(current_device) else 0
        device = torch.device(f"{device_name}:{int(device_index)}")
    flag = torch.tensor([1 if saved else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def log_resolved_config(config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.warning("Resolved SPECO standalone draft trainer config:\n%s", OmegaConf.to_yaml(config))
