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
"""Standalone torchrun training loop for SPECO draft models."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import math
import os
import time
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import OmegaConf, open_dict
from verl.utils.device import get_device_name, get_torch_device

from verl_speco.backends.factory import build_trainer_backend
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.draft_dataset import (
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import (
    DraftReplaySample,
    build_feature_store_from_config,
)
from verl_speco.trainer.standalone_resume import (
    load_standalone_resume,
    save_standalone_resume,
)
from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader, TQLocalBatch

logger = logging.getLogger(__name__)


def _should_log_batch_progress(attempted_batches: int) -> bool:
    return attempted_batches <= 3 or attempted_batches % 100 == 0


def _is_out_of_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    if "out of memory" in message or "oom" in message:
        return True
    return error.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}


def _contains_replay_samples(samples: list[Any]) -> bool:
    return any(isinstance(sample, DraftReplaySample) for sample in samples)


def run_standalone_draft_training(config) -> dict[str, Any]:
    """Run independent draft training from a feature store."""
    return asyncio.run(_run_standalone_draft_training_async(config))


async def _run_standalone_draft_training_async(config) -> dict[str, Any]:
    rank, local_rank, world_size = _init_distributed()
    logger.info(
        "[standalone rank=%s] distributed runtime initialized local_rank=%s world_size=%s",
        rank,
        local_rank,
        world_size,
    )
    draft_config = config.actor_rollout_ref
    drafter_cfg = draft_config.rollout.drafter
    training_cfg = drafter_cfg.training
    feature_store_cfg = training_cfg.feature_store
    feature_store_type = (
        str(feature_store_cfg.get("type", "torch_shard") or "torch_shard")
        .strip()
        .lower()
    )
    training_mode = (
        str(training_cfg.get("mode", "offline") or "offline").strip().lower()
    )
    replay_feature_store_types = {"token_replay", "jsonl_token_replay", "jsonl"}
    if feature_store_type != "tq" and not feature_store_cfg.get("path"):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.feature_store.path is required"
        )
    if feature_store_type == "tq" and training_mode != "offline":
        raise ValueError(
            "feature_store.type=tq requires standalone training.mode=offline"
        )
    if feature_store_type in replay_feature_store_types and training_mode != "offline":
        raise ValueError(
            f"feature_store.type={feature_store_type} is supported only by "
            "standalone training.mode=offline"
        )
    _disable_standalone_sequence_parallel(draft_config)

    _configure_device(local_rank)
    backend = _build_backend(draft_config)
    setattr(backend, "enable_standalone_training_metrics", True)
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
            else dist.group.WORLD
            if dist.is_initialized() and world_size > 1
            else None
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
    consumed_sequence_nos: set[int] = set()
    standalone_input_path = _standalone_input_path(config)
    store = None
    feature_replayer = None
    feature_producer = None
    current_stage = "activate_training_model"
    try:
        stage_started = time.perf_counter()
        logger.info(
            "[standalone rank=%s] activating drafter model algorithm=%s",
            rank,
            drafter_cfg.speculative_algorithm,
        )
        activated = await trainer.activate_training_model()
        if not activated:
            raise RuntimeError(
                f"Failed to activate standalone drafter trainer on rank={rank}"
            )
        logger.info(
            "[standalone rank=%s] drafter model activated elapsed=%.3fs",
            rank,
            time.perf_counter() - stage_started,
        )
        initial_optimizer_step = int(trainer.optimizer_steps_total)
        optimizer_step = initial_optimizer_step
        last_saved_step = optimizer_step
        if feature_store_type == "tq":
            consumed_sequence_nos, resume_metadata = load_standalone_resume(
                drafter_cfg.get("model_path"),
                input_path=standalone_input_path,
            )
            if initial_optimizer_step > 0 and resume_metadata is None:
                raise ValueError(
                    "Standalone TQ checkpoint restored optimizer state but has no "
                    "standalone_resume.json; exact data resume is unavailable"
                )
            logger.info(
                "[standalone rank=%s] resume progress optimizer_step=%s consumed=%s",
                rank,
                initial_optimizer_step,
                len(consumed_sequence_nos),
            )

        current_stage = "open_feature_store"
        stage_started = time.perf_counter()
        logger.info(
            "[standalone rank=%s] opening feature store type=%s path=%s",
            rank,
            feature_store_type,
            feature_store_cfg.get("path"),
        )
        if feature_store_type in {"jsonl_token_replay", "jsonl"} and not (
            feature_store_cfg.get("tokenizer_path")
        ):
            tokenizer_path = draft_config.model.path
            try:
                feature_store_cfg.tokenizer_path = tokenizer_path
            except AttributeError:
                feature_store_cfg["tokenizer_path"] = tokenizer_path
        store = build_feature_store_from_config(
            feature_store_cfg,
            read_only=True,
            transfer_queue_cfg=training_cfg.get("transfer_queue"),
        )
        if feature_store_type == "tq":
            current_stage = "connect_tq_feature_store"
            _connect_tq_store_across_ranks(
                store,
                rank=rank,
                device=trainer.runtime_device,
            )
        logger.info(
            "[standalone rank=%s] feature store opened elapsed=%.3fs",
            rank,
            time.perf_counter() - stage_started,
        )
        if feature_store_type in replay_feature_store_types:
            # Keep the large target model entirely outside online training imports
            # and lifetime. The standalone loop materializes ordinary feature
            # samples before handing them to the shared trainer.
            from verl_speco.trainer.target_feature_replay import (
                TargetFeatureReplayer,
            )

            current_stage = "initialize_target_feature_replayer"
            stage_started = time.perf_counter()
            logger.info(
                "[standalone rank=%s] initializing target feature replayer",
                rank,
            )
            feature_replayer = TargetFeatureReplayer(
                config,
                rank=rank,
                world_size=world_size,
                device=trainer.runtime_device,
            )
            logger.info(
                "[standalone rank=%s] target feature replayer initialized "
                "backend=%s elapsed=%.3fs",
                rank,
                feature_replayer.backend,
                time.perf_counter() - stage_started,
            )
        current_stage = "create_dataloader"
        loader: Any
        if feature_store_type == "tq":
            tq_cfg = training_cfg.get("transfer_queue") or {}
            loader = TQFeatureDataLoader(
                store,
                batch_size=int(training_cfg.get("batch_size_per_gpu", 4)),
                rank=rank,
                world_size=world_size,
                poll_interval_seconds=float(
                    tq_cfg.get("poll_interval_seconds", 0.5) or 0.5
                ),
                drop_last=bool(tq_cfg.get("drop_last", True)),
            )
        else:
            loader = DraftFeatureDataLoader(
                store,
                DraftFeatureDataLoaderConfig(
                    batch_size=int(training_cfg.get("batch_size_per_gpu", 4)),
                    rank=rank,
                    world_size=world_size,
                    shuffle=bool(feature_store_cfg.get("shuffle", True)),
                    repeat=bool(feature_store_cfg.get("repeat", True)),
                    seed=int(training_cfg.get("seed", 0) or 0),
                    min_sample_step=_optional_int(
                        feature_store_cfg.get("min_sample_step")
                    ),
                    max_sample_step=_optional_int(
                        feature_store_cfg.get("max_sample_step")
                    ),
                ),
            )
        logger.info(
            "[standalone rank=%s] dataloader ready batch_size_per_gpu=%s "
            "shuffle=%s repeat=%s",
            rank,
            int(training_cfg.get("batch_size_per_gpu", 4)),
            bool(feature_store_cfg.get("shuffle", True)),
            bool(feature_store_cfg.get("repeat", True)),
        )
        sample_source = loader
        pipeline_cfg = training_cfg.get("target_feature_pipeline", {}) or {}
        pipeline_enabled = bool(pipeline_cfg.get("enabled", False))
        if feature_store_type == "tq" and pipeline_enabled:
            raise ValueError(
                "feature_store.type=tq already contains target hidden states and cannot be "
                "combined with target_feature_pipeline.enabled=true"
            )
        if pipeline_enabled:
            if feature_replayer is None or not feature_replayer.backend.startswith(
                "vllm_"
            ):
                raise ValueError(
                    "target_feature_pipeline.enabled=true requires a vLLM replay "
                    "backend (vllm_file)"
                )
            from verl_speco.trainer.target_feature_pipeline import (
                TargetFeatureProducer,
            )

            feature_producer = TargetFeatureProducer(
                loader,
                feature_replayer,
                rank=rank,
                concurrency=max(
                    math.ceil(
                        int(pipeline_cfg.get("concurrency", 16) or 16) / world_size
                    ),
                    1,
                ),
                producer_prefetch_depth=int(
                    pipeline_cfg.get("producer_prefetch_depth", 4) or 4
                ),
                prefetch_depth=int(pipeline_cfg.get("prefetch_depth", 2) or 2),
                queue_timeout=float(pipeline_cfg.get("queue_timeout", 300.0) or 300.0),
            )
            sample_source = feature_producer
        sample_iterator = iter(sample_source)
        while max_steps <= 0 or optimizer_step < max_steps:
            current_stage = "load_next_batch"
            loaded_batch = _next_batch_across_ranks(
                sample_iterator,
                rank=rank,
                device=trainer.runtime_device,
            )
            if loaded_batch is None:
                break
            tq_local_batch = (
                loaded_batch if isinstance(loaded_batch, TQLocalBatch) else None
            )
            samples = (
                tq_local_batch.local_samples
                if tq_local_batch is not None
                else loaded_batch
            )
            step_started = time.perf_counter()
            attempted_batches += 1
            log_batch_progress = _should_log_batch_progress(attempted_batches)
            if log_batch_progress:
                logger.info(
                    "[standalone rank=%s] batch=%s loaded samples=%s "
                    "successful_steps=%s",
                    rank,
                    attempted_batches,
                    len(samples),
                    successful_steps,
                )
            current_stage = "materialize_target_features"
            if feature_replayer is None and _contains_replay_samples(samples):
                logger.warning(
                    "[standalone rank=%s] feature store type=%s yielded replay "
                    "samples without an initialized target feature replayer; "
                    "initializing replayer lazily",
                    rank,
                    feature_store_type,
                )
                from verl_speco.trainer.target_feature_replay import (
                    TargetFeatureReplayer,
                )

                feature_replayer = TargetFeatureReplayer(
                    config,
                    rank=rank,
                    world_size=world_size,
                    device=trainer.runtime_device,
                )
            if feature_replayer is not None and feature_producer is None:
                materialize_started = time.perf_counter()
                if log_batch_progress:
                    logger.info(
                        "[standalone rank=%s] batch=%s materializing target features "
                        "backend=%s",
                        rank,
                        attempted_batches,
                        feature_replayer.backend,
                    )
                materialized_samples = feature_replayer.materialize(samples)
                if log_batch_progress:
                    logger.info(
                        "[standalone rank=%s] batch=%s target features materialized "
                        "samples=%s elapsed=%.3fs",
                        rank,
                        attempted_batches,
                        len(materialized_samples),
                        time.perf_counter() - materialize_started,
                    )
            else:
                materialized_samples = samples
            current_stage = "prepare_training_batch"
            batch = trainer.prepare_training_batch_from_samples(
                cast(list[Any], materialized_samples),
                step=optimizer_step,
            )
            has_batch = batch is not None
            current_stage = "synchronize_batch_readiness"
            if not _all_ranks_true(has_batch, trainer.runtime_device):
                if tq_local_batch is not None:
                    raise RuntimeError(
                        "TQ Consumer could not prepare a valid batch on every rank; "
                        "the TQ keys were intentionally not cleared"
                    )
                if rank == 0:
                    logger.warning(
                        "Skipping standalone drafter batch: at least one rank has no valid batch"
                    )
                continue
            if batch is None:
                continue
            trainer.reset_training_metrics()
            current_stage = "training_step"
            if log_batch_progress:
                logger.info(
                    "[standalone rank=%s] batch=%s starting drafter training step "
                    "optimizer_step=%s",
                    rank,
                    attempted_batches,
                    optimizer_step,
                )
            ok = await trainer.training_step_from_batch(batch, optimizer_step)
            step_error = getattr(trainer, "last_standalone_training_error", None)
            if step_error is not None and _is_out_of_memory_error(step_error):
                raise RuntimeError(
                    "Standalone drafter training hit an unrecoverable OOM during "
                    f"batch={attempted_batches} optimizer_step={optimizer_step}. "
                    "Reduce batch_size_per_gpu, feature_store.max_seq_len, "
                    "dspark_num_anchors/block_size or disable DSpark L1 loss."
                ) from step_error
            current_stage = "synchronize_training_step"
            if not _all_ranks_true(ok, trainer.runtime_device):
                if tq_local_batch is not None:
                    raise RuntimeError(
                        "TQ Consumer training_step_from_batch failed on at least one rank; "
                        "the TQ keys were intentionally not cleared"
                    )
                continue
            if tq_local_batch is not None:
                current_stage = "clear_tq_batch"
                _clear_tq_batch_across_ranks(
                    cast(TQFeatureDataLoader, loader),
                    tq_local_batch.global_keys,
                    rank=rank,
                    device=trainer.runtime_device,
                )
                if rank == 0:
                    consumed_sequence_nos.update(
                        tq_local_batch.global_sequence_nos or []
                    )
            successful_steps += 1
            optimizer_step = int(trainer.optimizer_steps_total)
            if optimizer_step <= initial_optimizer_step:
                optimizer_step = initial_optimizer_step + successful_steps
            step_metrics = _standalone_step_metrics(
                trainer,
                successful_steps=successful_steps,
                attempted_batches=attempted_batches,
                step_elapsed_sec=time.perf_counter() - step_started,
            )
            if feature_replayer is not None:
                step_metrics.update(feature_replayer.metrics())
            if feature_producer is not None:
                step_metrics.update(feature_producer.metrics())
            _log_standalone_step_metrics(step_metrics, rank=rank)
            if save_interval > 0 and optimizer_step % save_interval == 0:
                current_stage = "save_checkpoint"
                last_save_result = _save_standalone_checkpoint(
                    trainer,
                    optimizer_step,
                    consumed_sequence_nos=consumed_sequence_nos,
                    input_path=(
                        standalone_input_path if feature_store_type == "tq" else None
                    ),
                )
                if _sync_any_rank_saved_checkpoint(last_save_result.get("saved")):
                    last_saved_step = optimizer_step
                _barrier()
            current_stage = "load_next_batch"
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0 and optimizer_step != last_saved_step:
            current_stage = "save_final_checkpoint"
            last_save_result = _save_standalone_checkpoint(
                trainer,
                optimizer_step,
                wait=True,
                consumed_sequence_nos=consumed_sequence_nos,
                input_path=(
                    standalone_input_path if feature_store_type == "tq" else None
                ),
            )
            _barrier()
    except Exception:
        logger.exception(
            "[standalone rank=%s] training failed stage=%s attempted_batches=%s "
            "successful_steps=%s optimizer_step=%s",
            rank,
            current_stage,
            attempted_batches,
            successful_steps,
            optimizer_step,
        )
        raise
    finally:
        logger.info(
            "[standalone rank=%s] cleanup starting stage=%s attempted_batches=%s "
            "successful_steps=%s",
            rank,
            current_stage,
            attempted_batches,
            successful_steps,
        )
        if feature_producer is not None:
            feature_producer.close()
        if feature_replayer is not None:
            feature_replayer.close()
        if store is not None:
            store.close()
        logger.info("[standalone rank=%s] cleaning trainer resources", rank)
        await trainer.cleanup_training(clear_data=True)
        if dist.is_initialized():
            logger.info(
                "[standalone rank=%s] entering final process-group barrier", rank
            )
            dist.barrier()
            logger.info(
                "[standalone rank=%s] final process-group barrier complete", rank
            )
            dist.destroy_process_group()
        logger.info("[standalone rank=%s] cleanup complete", rank)

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
    return build_trainer_backend(draft_config, draft_config.model)


def _save_standalone_checkpoint(
    trainer: DrafterBaseTrainer,
    step: int,
    *,
    wait: bool = False,
    consumed_sequence_nos: set[int] | None = None,
    input_path: str | None = None,
) -> dict[str, Any]:
    consumed_snapshot = torch.tensor(
        sorted(consumed_sequence_nos or ()), dtype=torch.int64
    )
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if callable(save_checkpoint):
        result = save_checkpoint(int(step), wait=wait)
        checkpoint_path = result.get("path")
        is_export_leader = result.get("reason") in {"saved", "scheduled"}
        if result.get("saved") and checkpoint_path and is_export_leader:
            if wait:
                _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
                _save_resume_sidecar(
                    checkpoint_path,
                    consumed_snapshot,
                    step=step,
                    input_path=input_path,
                )
            else:
                future = getattr(trainer, "_pending_full_checkpoint_future", None)
                if future is not None:
                    future.add_done_callback(
                        lambda completed: _finalize_standalone_checkpoint(
                            trainer,
                            checkpoint_path,
                            completed,
                            consumed_snapshot=consumed_snapshot,
                            step=step,
                            input_path=input_path,
                        )
                    )
        return result

    # Keep the small PR #13 test double and older trainer adapters usable.
    checkpoint_dir = getattr(trainer, "checkpoint_dir", None)
    if not checkpoint_dir:
        return {"saved": False, "reason": "missing_checkpoint_dir"}
    checkpoint_path = os.path.join(checkpoint_dir, f"draft_step_{int(step)}")
    pending_full_checkpoint = getattr(trainer, "_pending_full_checkpoint_future", None)
    pending_done = getattr(pending_full_checkpoint, "done", None)
    if callable(pending_done) and not pending_done():
        return {
            "saved": False,
            "path": checkpoint_path,
            "reason": "previous_save_running",
        }

    save_async = getattr(trainer, "_save_checkpoint_async", None)
    if not callable(save_async):
        return {
            "saved": False,
            "path": checkpoint_path,
            "reason": "unsupported_trainer",
        }
    future = save_async(int(step))
    if future is not None and wait:
        future.result()
        trainer._pending_full_checkpoint_future = None
        _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
        _save_resume_sidecar(
            checkpoint_path,
            consumed_snapshot,
            step=step,
            input_path=input_path,
        )
    elif future is not None:
        future.add_done_callback(
            lambda completed: _finalize_standalone_checkpoint(
                trainer,
                checkpoint_path,
                completed,
                consumed_snapshot=consumed_snapshot,
                step=step,
                input_path=input_path,
            )
        )
    return {
        "saved": future is not None,
        "path": checkpoint_path,
        "reason": (
            "saved"
            if future is not None and wait
            else "scheduled"
            if future is not None
            else "not_checkpoint_leader"
        ),
    }


def _load_tensor_from_safetensors(
    path: str, keys: tuple[str, ...]
) -> tuple[str, torch.Tensor] | None:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            available_keys = set(f.keys())
            for key in keys:
                if key in available_keys:
                    return key, f.get_tensor(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load any of %s from %s: %s", keys, path, exc)
    return None


def _finalize_standalone_checkpoint(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future,
    *,
    consumed_snapshot: torch.Tensor | None = None,
    step: int | None = None,
    input_path: str | None = None,
) -> None:
    try:
        completed_future.result()
    except Exception:
        _rewrite_standalone_block_runtime_config(
            trainer, checkpoint_path, completed_future
        )
        return

    _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
    _save_resume_sidecar(
        checkpoint_path,
        consumed_snapshot,
        step=step,
        input_path=input_path,
    )


def _save_resume_sidecar(
    checkpoint_path: str,
    consumed_snapshot: torch.Tensor | None,
    *,
    step: int | None,
    input_path: str | None,
) -> None:
    if consumed_snapshot is None or step is None or input_path is None:
        return
    save_standalone_resume(
        checkpoint_path,
        consumed_snapshot,
        optimizer_step=step,
        input_path=input_path,
    )


def _standalone_input_path(config: Any) -> str | None:
    data_cfg = getattr(config, "data", None)
    train_files = getattr(data_cfg, "train_files", None)
    if train_files is None and isinstance(data_cfg, dict):
        train_files = data_cfg.get("train_files")
    if isinstance(train_files, str):
        return train_files
    if train_files is not None and len(train_files) == 1:
        return str(train_files[0])
    return None


def _ensure_dict_child(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    config[key] = value
    return value


def _load_source_drafter_config(trainer: DrafterBaseTrainer) -> dict[str, Any] | None:
    model_path = getattr(
        getattr(getattr(trainer, "config", None), "rollout", None), "drafter", None
    )
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


def _fill_if_missing(
    dst: dict[str, Any], src: dict[str, Any], keys: tuple[str, ...]
) -> None:
    for key in keys:
        if key in src and key not in dst:
            dst[key] = deepcopy(src[key])


# Extra runtime-facing config a DFlash variant needs on top of the shared DFlash
# aliases, as (config child, keys copied from the training config). Domino writes
# into dflash_config because engines serve it as a DFlash projector sub-mode
# (dflash_config.projector_type="domino"), while DSpark is its own serve method.
_VARIANT_RUNTIME_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "domino": (
        "dflash_config",
        (
            "block_size",
            "num_anchors",
            "loss_decay_gamma",
            "emb_dim",
            "gru_hidden_dim",
            "pure_draft_prefix_len",
            "num_target_layers",
            "target_num_hidden_layers",
        ),
    ),
    # DFlash2 is served as a DFlash checkpoint whose dflash_config carries the
    # selector/conv hyperparameters, matching the released z-lab layout.
    "dflash2": (
        "dflash_config",
        (
            "block_size",
            "num_anchors",
            "loss_decay_gamma",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "target_layer_ids",
            "num_context_layers",
            "num_target_layers",
            "target_num_hidden_layers",
            "mask_token_id",
        ),
    ),
    "dspark": (
        "dspark_config",
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
    ),
}


def _rewrite_standalone_block_runtime_config(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future=None,
) -> None:
    """Export standalone DFlash/DSpark/Domino checkpoints with runtime-facing config.

    The training wrapper saves an internal SpeCo config.  For standalone
    checkpoints we keep the original drafter ``config.json`` as the runtime
    contract and only merge the alias fields needed by vLLM/SGLang.
    """
    backend_type = getattr(getattr(trainer, "backend", None), "model_type", None)
    if backend_type not in {"dflash", "dflash2", "dspark", "domino"}:
        return

    if completed_future is not None:
        try:
            completed_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skip standalone runtime config rewrite because checkpoint save failed: %s",
                exc,
            )
            return

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning(
            "Cannot rewrite standalone runtime config: missing %s", config_path
        )
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Cannot rewrite standalone runtime config %s: %s", config_path, exc
        )
        return
    if not isinstance(training_config, dict):
        logger.warning(
            "Cannot rewrite standalone runtime config %s: expected object", config_path
        )
        return

    training_config_path = os.path.join(checkpoint_path, "speco_training_config.json")
    try:
        with open(training_config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning(
            "Failed to write standalone training config copy %s: %s",
            training_config_path,
            exc,
        )

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

    variant_child_key, variant_alias_keys = _VARIANT_RUNTIME_ALIASES.get(
        backend_type, (None, ())
    )
    variant_config = (
        _ensure_dict_child(runtime_config, variant_child_key)
        if variant_child_key
        else {}
    )
    _fill_if_missing(variant_config, training_config, variant_alias_keys)
    if backend_type == "domino":
        variant_config["projector_type"] = str(
            training_config.get("projector_type", "domino") or "domino"
        )

    target_layer_ids = (
        runtime_config.get("target_layer_ids")
        or dflash_config.get("target_layer_ids")
        or variant_config.get("target_layer_ids")
    )
    if (
        target_layer_ids is not None
        and "eagle_aux_hidden_state_layer_ids" not in runtime_config
    ):
        try:
            runtime_config["eagle_aux_hidden_state_layer_ids"] = [
                int(layer_id) + 1 for layer_id in target_layer_ids
            ]
        except (TypeError, ValueError):
            logger.warning(
                "Invalid target_layer_ids in standalone exported config: %r",
                target_layer_ids,
            )

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        logger.warning(
            "Failed to write standalone runtime config %s: %s", config_path, exc
        )


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
    strategy = str(
        draft_config.actor.get("strategy", "") if hasattr(draft_config, "actor") else ""
    ).lower()
    if strategy != "fsdp2":
        return None
    return DeviceMesh(
        device_type=get_device_name(),
        mesh=torch.arange(world_size, dtype=torch.int64).reshape(1, world_size),
        mesh_dim_names=("dp", "sp"),
    )


def _block_metric_prefix(trainer: DrafterBaseTrainer) -> str | None:
    model_type = str(getattr(getattr(trainer, "backend", None), "model_type", "") or "")
    if model_type in {"dflash", "dspark", "eagle3"}:
        return model_type
    return None


def _current_learning_rate(trainer: DrafterBaseTrainer) -> float:
    optimizer = getattr(trainer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return 0.0
    return float(param_groups[0].get("lr", 0.0))


def _position_metric_series(
    metrics: dict[str, float], prefix: str, name: str
) -> list[float]:
    values: list[float] = []
    pos = 0
    while True:
        key = f"{prefix}/{name}/{pos}"
        if key not in metrics:
            break
        values.append(float(metrics[key]))
        pos += 1
    return values


def _weighted_average(values: list[float], counts: list[float]) -> float | None:
    if not values or not counts:
        return None
    total_count = sum(counts[: len(values)])
    if total_count <= 0:
        return None
    return (
        sum(value * count for value, count in zip(values, counts, strict=False))
        / total_count
    )


def _simulated_accept_length(accuracies: list[float]) -> float:
    cumulative = 1.0
    simulated = 0.0
    for accuracy in accuracies:
        cumulative *= max(0.0, min(1.0, float(accuracy)))
        simulated += cumulative
    return simulated


def _standalone_step_metrics(
    trainer: DrafterBaseTrainer,
    *,
    successful_steps: int,
    attempted_batches: int,
    step_elapsed_sec: float,
) -> dict[str, float]:
    raw_metrics = trainer.get_training_metrics()
    metrics: dict[str, float] = {
        key: float(value) for key, value in raw_metrics.items()
    }
    prefix = _block_metric_prefix(trainer)
    if prefix is not None:
        anchor_offset = 1 if prefix == "dflash" else 0
        losses = _position_metric_series(raw_metrics, prefix, "loss_per_position")
        accuracies = _position_metric_series(
            raw_metrics, prefix, "accuracy_per_position"
        )
        counts = _position_metric_series(raw_metrics, prefix, "count_per_position")
        pred_losses = losses[anchor_offset:]
        pred_accuracies = accuracies[anchor_offset:]
        pred_counts = counts[anchor_offset:]

        avg_loss = _weighted_average(pred_losses, pred_counts)
        avg_acc = _weighted_average(pred_accuracies, pred_counts)
        if avg_loss is not None:
            metrics["train/avg_loss"] = avg_loss
        if avg_acc is not None:
            metrics["train/avg_acc"] = avg_acc
        if f"{prefix}/simulated_acc_len" in raw_metrics:
            metrics["train/simulated_acc_len"] = float(
                raw_metrics[f"{prefix}/simulated_acc_len"]
            )
        elif pred_accuracies:
            metrics["train/simulated_acc_len"] = _simulated_accept_length(
                pred_accuracies
            )
        if f"{prefix}/top1_acc" in raw_metrics:
            metrics["train/top1_acc"] = float(raw_metrics[f"{prefix}/top1_acc"])
        if f"{prefix}/top5_acc" in raw_metrics:
            metrics["train/top5_acc"] = float(raw_metrics[f"{prefix}/top5_acc"])
        for idx, value in enumerate(pred_losses):
            metrics[f"train/ploss_{idx}"] = float(value)
        for idx, value in enumerate(pred_accuracies):
            metrics[f"train/acc_{idx}"] = float(value)
    metrics["train/step"] = float(successful_steps)
    metrics["train/global_step"] = float(
        getattr(trainer, "training_steps", successful_steps)
    )
    metrics["train/lr"] = _current_learning_rate(trainer)
    metrics["drafter/train_successful_steps"] = float(successful_steps)
    metrics["drafter/train_attempted_batches"] = float(attempted_batches)
    metrics["perf/step_time"] = float(step_elapsed_sec)
    return metrics


def _log_standalone_step_metrics(metrics: dict[str, float], *, rank: int) -> None:
    if rank != 0:
        return
    fields = [f"step={int(metrics.get('train/step', 0.0))}"]
    for key, label in (
        ("train/avg_loss", "avg_loss"),
        ("train/avg_acc", "avg_acc"),
        ("train/top1_acc", "top1"),
        ("train/top5_acc", "top5"),
        ("train/simulated_acc_len", "sim_acc_len"),
        ("train/lr", "lr"),
        ("perf/step_time", "step_time"),
        ("replay/cache_hit_ratio", "cache_hit"),
        ("replay/target_forward_time_total", "target_forward_total"),
        ("replay/vllm_request_time_total", "vllm_request_total"),
        ("producer/consumer_wait_time_total", "producer_wait_total"),
        ("producer/ready_queue_size", "ready_batches"),
    ):
        if key not in metrics:
            continue
        value = float(metrics[key])
        if key == "train/lr":
            fields.append(f"{label}={value:.3e}")
        elif key.endswith("_time_total") or key in {
            "perf/step_time",
            "replay/target_forward_time_total",
        }:
            fields.append(f"{label}={value:.3f}s")
        else:
            fields.append(f"{label}={value:.4f}")
    logger.warning("[standalone drafter metrics] %s", " ".join(fields))


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
            raise ValueError(
                f"Unsupported standalone drafter device_name={device_name!r}"
            )
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


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(Any, value))


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return bool(value)
    ready = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    return bool(ready.item())


def _clear_tq_batch_across_ranks(
    loader: TQFeatureDataLoader,
    global_keys: list[str] | None,
    *,
    rank: int,
    device: torch.device,
) -> None:
    """Clear once on rank 0 and report a clear failure to every training rank."""

    local_error: BaseException | None = None
    if rank == 0:
        try:
            loader.clear_completed_batch(global_keys)
        except BaseException as exc:  # noqa: BLE001
            local_error = exc
    failed = torch.tensor(
        1 if local_error is not None else 0,
        dtype=torch.int32,
        device=device,
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if bool(failed.item()):
        if local_error is not None:
            raise RuntimeError(
                "rank 0 failed to clear a completed TQ batch"
            ) from local_error
        raise RuntimeError("rank 0 failed to clear a completed TQ batch")


def _connect_tq_store_across_ranks(store, *, rank: int, device: torch.device) -> None:
    """Connect every rank before any rank enters TQ key-discovery broadcasts."""

    local_error: BaseException | None = None
    try:
        store.connect()
    except BaseException as exc:  # noqa: BLE001
        local_error = exc
    connected = _all_ranks_true(local_error is None, device)
    if connected:
        return
    if local_error is not None:
        raise RuntimeError(
            f"TQ Consumer failed to connect on rank={rank}"
        ) from local_error
    raise RuntimeError(
        f"TQ Consumer failed to connect on another rank; rank={rank} is stopping"
    )


def _next_batch_across_ranks(
    source,
    *,
    rank: int,
    device: torch.device,
) -> Any | None:
    """Fetch one batch and make producer failures visible to every rank.

    Producer and replay errors happen before the FSDP training step. Every
    rank therefore reports its fetch result through the same collective before
    any rank is allowed to enter model collectives.  This prevents healthy
    ranks from waiting in FSDP after another rank has already started cleanup.
    """
    samples: Any | None = None
    local_error: BaseException | None = None
    exhausted = False
    try:
        samples = next(source)
    except StopIteration:
        exhausted = True
    except BaseException as exc:  # noqa: BLE001
        local_error = exc

    state = torch.tensor(
        [1 if local_error is not None else 0, 1 if exhausted else 0],
        dtype=torch.int32,
        device=device,
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(state, op=dist.ReduceOp.MAX)

    any_failed = bool(state[0].item())
    any_exhausted = bool(state[1].item())
    if any_failed:
        if local_error is not None:
            raise RuntimeError(
                f"Standalone target-feature producer failed on rank={rank}; "
                "all ranks are stopping before the next training collective"
            ) from local_error
        raise RuntimeError(
            "Standalone target-feature producer failed on another rank; "
            f"rank={rank} is stopping before the next training collective"
        )
    if any_exhausted:
        if not exhausted:
            logger.warning(
                "[standalone rank=%s] discarding a prefetched batch because "
                "another rank exhausted its data source",
                rank,
            )
        return None
    if samples is None:
        raise RuntimeError(
            f"Rank={rank} reported a successful batch fetch without samples"
        )
    return samples


def _sync_any_rank_saved_checkpoint(saved: Any) -> bool:
    if not dist.is_initialized():
        return bool(saved)
    device = torch.device(get_device_name())
    flag = torch.tensor([1 if saved else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def log_resolved_config(config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.warning(
            "Resolved SPECO standalone draft trainer config:\n%s",
            OmegaConf.to_yaml(config),
        )
