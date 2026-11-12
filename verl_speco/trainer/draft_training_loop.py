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
import contextlib
import hashlib
import ipaddress
import json
import logging
import math
import os
import random
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
from torch.distributed.device_mesh import DeviceMesh
from verl.utils.device import get_device_name, get_torch_device

from verl_speco.backends.factory import build_trainer_backend
from verl_speco.config import config_int, prefetch_depth
from verl_speco.standalone_ray_runtime import put_queue
from verl_speco.standalone_layer_ids import normalize_standalone_layer_ids
from verl_speco.trainer.base_trainer import (
    DrafterBaseTrainer,
    resolve_drafter_strategy,
)
from verl_speco.trainer.draft_dataset import (
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import (
    DraftReplaySample,
    build_feature_store_from_config,
)
from verl_speco.trainer.standalone_resume import (
    build_input_fingerprint,
    producer_data_recipe,
    input_content_identity,
    model_cache_content_sha256,
    load_standalone_resume,
    save_standalone_resume,
    validate_input_identity,
)
from verl_speco.trainer.standalone_tracking import (
    StandaloneTokenWindow,
    _build_standalone_tracking,
    _finish_standalone_tracking,
    _log_standalone_tracking_metrics,
    _select_standalone_tracking_metrics,
)
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.trainer.standalone_batch_finalize import AsyncBatchFinalizer
from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator, ReferenceDevicePrefetchIterator
from verl_speco.trainer.tq_sample_source import (
    TQFeatureDataLoader,
    TQLocalBatch,
    TQPrefetchIterator,
    _reference_device_transfer_bytes,
)

logger = logging.getLogger(__name__)

_TQ_GET_MAX_SYNC_INTERVAL_STEPS = 50
_TQ_CLEAR_MAX_ATTEMPTS = 3
_TQ_CLEAR_RETRY_BACKOFF_SECONDS = 0.5


def _should_log_batch_progress(attempted_batches: int) -> bool:
    return attempted_batches <= 3 or attempted_batches % 100 == 0


def _is_out_of_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    if "out of memory" in message or "oom" in message:
        return True
    return error.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}


def _contains_replay_samples(samples: list[Any]) -> bool:
    return any(isinstance(sample, DraftReplaySample) for sample in samples)


def _verify_rollback_checkpoint_across_ranks(training_cfg, model_path, world_size: int) -> None:
    """Require every rank to see the same intact, launcher-selected checkpoint."""
    enabled = training_cfg.get("checkpoint_auto_rollback", False)
    if type(enabled) is not bool:
        raise ValueError("checkpoint_auto_rollback must be a boolean")
    if not enabled:
        return
    from verl_speco.trainer.checkpoint import verify_standalone_checkpoint_integrity

    try:
        path = Path(model_path)
        if not verify_standalone_checkpoint_integrity(path):
            raise ValueError("checkpoint integrity verification failed")
        identity = hashlib.sha256((path / "standalone_integrity.json").read_bytes()).hexdigest()
        local = (identity, None)
    except Exception as exc:  # noqa: BLE001 - every rank must enter the collective
        local = (None, f"{type(exc).__name__}: {exc}")
    gathered = [None] * world_size
    if dist.is_initialized() and world_size > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if any(item[1] is not None for item in gathered) or len({item[0] for item in gathered}) != 1:
        raise ValueError(f"Standalone rollback checkpoint differs or is invalid across ranks: {gathered}")


def _zero_standalone_padding_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep tensor geometry for collectives while excluding placeholder work."""
    result = dict(batch)
    for name in ("loss_mask", "label_mask", "next_token_loss_mask"):
        if name in result:
            result[name] = torch.zeros_like(result[name])
    if "loss_mask" not in result:
        raise ValueError("Padding rank requires a loss mask")
    if "_speco_token_counts" in result:
        counts = {key: 0 for key in result["_speco_token_counts"]}
        # The placeholder contributes no samples or supervision, but the
        # actual model still processes these positions as padding work.
        if "padding_tokens" in counts and torch.is_tensor(result.get("input_ids")):
            counts["padding_tokens"] = int(result["input_ids"].numel())
        result["_speco_token_counts"] = counts
    result["_speco_sample_count"] = 0
    result["_speco_sample_ids"] = ()
    return result


def _seed_standalone_step(base_seed: int, optimizer_step: int, rank: int) -> None:
    """Derive per-rank RNG state without depending on process or run identity."""
    import numpy as np

    data = f"{base_seed}:{optimizer_step}:{rank}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.manual_seed_all(seed)


def _standalone_rng_identity(enabled, device):
    if type(enabled) is not bool:
        raise ValueError("deterministic_step_rng must be a boolean")
    if not enabled:
        return {}
    device_type = getattr(device, "type", None) or torch.device(device).type
    if device_type not in {"cpu", "cuda", "npu"}:
        raise ValueError("deterministic_step_rng requires a supported CPU, CUDA or NPU device")
    return {"rng_device_type": device_type}


def _valid_checkpoint_retention(keep, delete, exclusive, inherit=False):
    return (type(keep) is int and (keep == 0 or keep >= 2)
            and type(delete) is bool and type(exclusive) is bool and type(inherit) is bool
            and (not (delete or inherit) or (keep >= 2 and exclusive)))


def _validate_node_window_host_budget(host_bytes, frame_bytes):
    if (type(host_bytes) is not int or type(frame_bytes) is not int
            or frame_bytes <= 0 or host_bytes < 3 * frame_bytes):
        raise ValueError("Node-owned windows require a positive frame limit and a three-frame host reservation")


def _validate_checkpoint_retention_across_ranks(settings, *, rank, device):
    leader_settings = [settings if rank == 0 else None]
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list(leader_settings, src=0)
    if not _all_ranks_true(_valid_checkpoint_retention(*settings)
                           and settings == leader_settings[0], device):
        raise ValueError("Checkpoint retention settings must agree across ranks; deletion or inheritance requires "
                         "keep_last >= 2 and cooperative exclusive directory ownership")


def _retention_resume_identity_across_ranks(root, checkpoint, step, resume_metadata, world_size):
    from verl_speco.trainer.checkpoint import standalone_retention_resume_identity
    try:
        if resume_metadata is None or resume_metadata.get("optimizer_step") != step:
            raise ValueError("Retention inheritance requires verified restored replay progress")
        identity = standalone_retention_resume_identity(root, checkpoint, optimizer_step=step)
        local = {"identity": identity}
    except Exception as exc:
        local = {"error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if any(item.get("error") for item in gathered) or any(item != gathered[0] for item in gathered):
        raise ValueError(f"Checkpoint retention inheritance failed across ranks: {gathered}")
    return local["identity"]


def _verify_shared_file_root_across_ranks(store, producer_handle, *, rank, device, timeout):
    """Check that every trainer can read a witness created on the Producer node."""
    hidden_store = getattr(store, "hidden_states_store", None)
    root = getattr(hidden_store, "root", None)
    local_root = str(root) if root is not None else None
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    selected = [local_root if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(selected, src=0)
    if not _all_ranks_true(local_root == selected[0] and (local_root is None or producer_handle is not None), device):
        raise RuntimeError("Shared file root or Producer handle differs across ranks")
    if local_root is None:
        return
    response = [None]
    if rank == 0:
        try:
            import ray
            response[0] = {"probe": ray.get(producer_handle.create_shared_file_probe.remote(), timeout=timeout)}
        except Exception as exc:
            response[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(response, src=0)
    if "error" in response[0]:
        raise RuntimeError("Failed to create Producer shared-file witness: " + response[0]["error"])
    probe, error = response[0]["probe"], None
    try:
        if (not isinstance(probe, dict) or set(probe) != {"path", "token"}
                or not isinstance(probe["token"], str) or len(probe["token"]) != 32
                or any(c not in "0123456789abcdef" for c in probe["token"])):
            raise ValueError("Invalid Producer shared-file witness")
        path = hidden_store.validate_reference(probe["path"])
        if (path.name != ".speco-shared-probe-" + probe["token"] or path.stat().st_size != 32
                or path.read_text(encoding="ascii") != probe["token"]):
            raise ValueError("Producer shared-file witness is not visible or differs")
    except Exception as exc:
        error = exc
    visible = _all_ranks_true(error is None, device)
    cleanup_error = None
    if rank == 0:
        try:
            import ray
            ray.get(producer_handle.remove_shared_file_probe.remote(probe["path"], probe["token"]), timeout=timeout)
        except Exception as exc:
            cleanup_error = exc
    cleaned = _all_ranks_true(cleanup_error is None, device)
    if not visible or not cleaned:
        raise RuntimeError("Shared file root is not consistently visible or its witness cleanup failed") from (error or cleanup_error)


def _acquire_checkpoint_directory_across_ranks(directory, enabled, *, rank, device):
    """Optionally reserve one shared directory until every writer has stopped."""
    from verl_speco.trainer.checkpoint import StandaloneCheckpointDirectoryLease
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    setting = [enabled if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(setting, src=0)
    if not _all_ranks_true(type(enabled) is bool and enabled == setting[0], device):
        raise ValueError("checkpoint_directory_exclusive must match on every rank and be boolean")
    if not enabled:
        return None
    response = [None]
    owner = None
    if rank == 0:
        try:
            if not isinstance(directory, (str, os.PathLike)) or not str(directory):
                raise ValueError("Exclusive checkpoint directory requires a path")
            owner = StandaloneCheckpointDirectoryLease(directory, uuid.uuid4().hex)
            owner.acquire()
            response[0] = {"token": owner.token}
        except Exception as exc:
            response[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(response, src=0)
    if "error" in response[0]:
        raise RuntimeError("Cannot acquire checkpoint directory: " + response[0]["error"])
    try:
        lease = StandaloneCheckpointDirectoryLease(directory, response[0]["token"])
        lease.verify()
        visible = True
    except Exception:
        visible = False
    if not _all_ranks_true(visible, device):
        raise RuntimeError("Checkpoint directory marker is not visible to every rank; it remains reserved")
    return lease


def _standalone_recipe(training_cfg: Any, target_model_path: str | None) -> dict[str, Any]:
    """Capture configuration that changes training data, numerics or optimizer state."""
    explicit = {
        "strategy", "execution_strategy", "fsdp_shard_size", "ddp_find_unused_parameters", "lr", "optimizer", "weight_decay",
        "adamw_betas", "muon_lr", "muon_momentum", "muon_nesterov", "muon_weight_decay",
        "muon_ns_steps", "muon_adjust_lr_fn", "muon_adamw_name_hints", "lr_warmup_steps",
        "lr_scheduler_type", "lr_decay_steps", "min_lr_ratio", "warmup_style", "use_logits",
        "ttt_length", "packing", "group_by_length", "group_by_length_megabatch", "lk_temperature",
        "lk_target_logz_tolerance", "lk_chunk_size", "ce_target_source", "vocab_mapping_path",
        "preserve_prompt_hidden_states", "max_seq_len", "hidden_state_clip_value",
        "resume_trainer_state_from_checkpoint", "logits_topk", "logits_sparse_min_intersection",
        "logits_sparse_min_mass", "logits_coverage_mask_min_ratio", "logits_coverage_mask_require_top1",
    }
    prefixes = ("dspark_", "dflash_", "dflash2_", "eagle1_", "eagle3_", "domino_", "peagle_")
    selected = {
        str(key): training_cfg.get(key) for key in training_cfg.keys()
        if key in explicit or (str(key).startswith(prefixes) and "debug" not in str(key))
    }
    if training_cfg.get("reference_owned_windows", False):
        selected["reference_owned_windows"] = True
    feature_store = training_cfg.get("feature_store") or {}
    selected["feature_store"] = {
        key: feature_store.get(key) for key in ("type", "max_seq_len", "window_mode", "train_on",
                                               "min_supervised_tokens", "strict_token_alignment")
    }
    tq = training_cfg.get("transfer_queue") or {}
    expected_feature = tq.get("expected_feature") or {}
    cache_identity = training_cfg.get("target_model_cache_content_identity", False)
    if type(cache_identity) is not bool:
        raise ValueError("target_model_cache_content_identity must be a boolean")
    if cache_identity:
        logical_id = training_cfg.get("target_model_logical_id")
        if not isinstance(logical_id, str) or not logical_id.strip() or logical_id != logical_id.strip():
            raise ValueError("Content-verified model cache requires target_model_logical_id")
        if not isinstance(target_model_path, str) or not target_model_path:
            raise ValueError("Content-verified model cache requires a local target model path")
        expected_feature = dict(expected_feature)
        if expected_feature.get("target_model_path") == target_model_path:
            expected_feature["target_model_path"] = logical_id
        selected["target_model_path"] = logical_id
    else:
        selected["target_model_path"] = target_model_path
    selected["expected_feature"] = expected_feature
    target_config = Path(target_model_path) / "config.json" if target_model_path else None
    if target_config is not None and target_config.is_file():
        before = target_config.stat()
        if before.st_size > 8 * 1024 * 1024:
            raise ValueError("Target model config exceeds 8 MiB")
        content = target_config.read_bytes()
        after = target_config.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("Target model config changed during fingerprinting")
        selected["target_config_sha256"] = hashlib.sha256(content).hexdigest()
    else:
        selected["target_config_sha256"] = None
    if cache_identity:
        selected["target_model_cache_sha256"] = model_cache_content_sha256(target_model_path)
        if hashlib.sha256(target_config.read_bytes()).hexdigest() != selected["target_config_sha256"]:
            raise ValueError("Target model config changed while fingerprinting its cache")

    def plain(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    normalized = plain(selected)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > 65536:
        raise ValueError("Standalone training recipe exceeds 64 KiB")
    return {"version": 1, "sha256": hashlib.sha256(encoded).hexdigest(), "settings": normalized}


def _standalone_recipe_across_ranks(training_cfg, target_model_path, world_size):
    """Reject local recipe failures or mismatches before ranks continue training."""
    recipe, error = None, None
    try:
        snapshot_interval = training_cfg.get("trainable_snapshot_interval_steps", 0)
        if type(snapshot_interval) is not int or snapshot_interval < 0:
            raise ValueError("trainable_snapshot_interval_steps must be a non-negative integer")
        snapshot_max_bytes = training_cfg.get("trainable_snapshot_max_bytes", 0)
        if type(snapshot_max_bytes) is not int or snapshot_max_bytes < 0:
            raise ValueError("trainable_snapshot_max_bytes must be a non-negative integer")
        snapshot_init = training_cfg.get("trainable_snapshot_initialization_path")
        if snapshot_init is not None and (not isinstance(snapshot_init, str) or not snapshot_init.strip()):
            raise ValueError("trainable_snapshot_initialization_path must be a nonempty path or null")
        recipe = _standalone_recipe(training_cfg, target_model_path)
    except Exception as exc:  # All ranks must reach the same startup collective.
        error = f"{type(exc).__name__}: {exc}"
    local = {"error": error, "sha256": recipe["sha256"] if recipe else None,
             "snapshot_interval": snapshot_interval if error is None else None,
             "snapshot_max_bytes": snapshot_max_bytes if error is None else None,
             "snapshot_init": snapshot_init if error is None else None}
    if dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if any(item["error"] for item in gathered) or any(item != gathered[0] for item in gathered):
        raise ValueError(f"Standalone recipe differs or failed across ranks: {gathered}")
    return recipe


def _validate_resume_training_plan(metadata, optimizer_step: int, training_plan: dict[str, Any]) -> None:
    if metadata is None:
        return
    if int(metadata.get("optimizer_step", -1)) != optimizer_step:
        raise ValueError("Standalone resume progress does not match restored optimizer step")
    saved_plan = metadata.get("training_plan")
    if (isinstance(saved_plan, dict) and "tail_policy" not in saved_plan
            and "tail_policy" in training_plan):
        # Checkpoints created before tail support always used the drop policy.
        saved_plan = {**saved_plan, "tail_policy": "drop"}
    if isinstance(saved_plan, dict) and "recipe" not in saved_plan:
        # Legacy sidecars did not record these settings. Check their known
        # fields but cannot retroactively establish recipe equivalence.
        training_plan = {key: value for key, value in training_plan.items() if key != "recipe"}
    if (isinstance(saved_plan, dict) and saved_plan.get("deterministic_step_rng") is True
            and "rng_device_type" not in saved_plan and "rng_device_type" in training_plan):
        logger.warning("Legacy checkpoint does not record the RNG device type; cross-device reproducibility cannot be verified")
        training_plan = {key: value for key, value in training_plan.items() if key != "rng_device_type"}
    if saved_plan is not None and saved_plan != training_plan:
        raise ValueError(f"Standalone resume training plan changed: saved={saved_plan}, current={training_plan}")


def _load_resume_across_ranks(model_path, input_path, optimizer_step, training_plan, world_size):
    """Make local resume failures and differing progress visible on every rank."""
    consumed, metadata = set(), None
    try:
        consumed, metadata = load_standalone_resume(model_path, input_path=input_path)
        if optimizer_step > 0 and metadata is None:
            raise ValueError("Standalone TQ checkpoint restored optimizer state but has no standalone_resume.json")
        _validate_resume_training_plan(metadata, optimizer_step, training_plan)
        digest = hashlib.sha256()
        for sequence_no in sorted(consumed):
            digest.update(f"{sequence_no},".encode("ascii"))
        local = {"error": None, "progress": digest.hexdigest(), "step": optimizer_step, "plan": training_plan}
    except Exception as exc:  # noqa: BLE001 - all ranks must enter the collective
        local = {"error": f"{type(exc).__name__}: {exc}"}
    gathered = [None] * world_size
    if dist.is_initialized() and world_size > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if any(item.get("error") for item in gathered) or any(item != gathered[0] for item in gathered):
        raise ValueError(f"Standalone resume differs or failed across ranks: {gathered}")
    return consumed, metadata


def _input_identity_across_ranks(input_path, world_size: int):
    """Bind all trainers to rank zero's input content before accepting batches."""
    distributed = dist.is_initialized() and world_size > 1
    rank = dist.get_rank() if distributed else 0
    result = [None]
    if rank == 0:
        try:
            result[0] = {"identity": input_content_identity(
                build_input_fingerprint(input_path, include_content_hash=True)) if input_path else None}
        except FileNotFoundError as exc:
            # A wholly absent source can be mounted only on the Producer node.
            # An existing but incomplete replay directory is corruption instead.
            try:
                exists = bool(input_path and Path(input_path).exists())
            except Exception as probe_error:  # Still notify every live rank.
                result[0] = {"error": f"{type(probe_error).__name__}: {probe_error}"}
            else:
                result[0] = ({"error": f"{type(exc).__name__}: {exc}"} if exists else {"identity": None})
        except Exception as exc:  # Other ranks must receive the failure too.
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[0].get("error"):
        raise RuntimeError(f"Failed to bind standalone input: {result[0]['error']}")
    return result[0]["identity"]


def _validate_live_trainer_topology(records, node_ips, processes_per_node):
    """Verify Ray's actual physical placements before loading a model."""
    if type(processes_per_node) is not int or processes_per_node < 1:
        raise ValueError("Invalid trainer processes per node")
    try:
        nodes = sorted(str(ipaddress.ip_address(str(ip))) for ip in node_ips)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid trainer node IP list") from exc
    if not nodes or len(set(nodes)) != len(nodes) or len(records) != len(nodes) * processes_per_node:
        raise ValueError("Live trainer topology does not match configured world size")
    seen_ranks, node_ids, devices = set(), {}, set()
    for item in records:
        if not isinstance(item, dict) or item.get("error"):
            raise ValueError(f"Cannot verify live trainer placement: {item}")
        rank = item.get("rank")
        if type(rank) is not int or not 0 <= rank < len(records) or rank in seen_ranks:
            raise ValueError("Live trainer ranks are invalid or duplicated")
        seen_ranks.add(rank)
        try:
            address = str(ipaddress.ip_address(item.get("ip")))
        except (ValueError, TypeError) as exc:
            raise ValueError("Live trainer node address is invalid") from exc
        node_id, assigned = item.get("node_id"), item.get("accelerators")
        if (address != nodes[rank // processes_per_node] or not isinstance(node_id, str) or not node_id
                or not isinstance(assigned, list) or len(assigned) != 1
                or not isinstance(assigned[0], str) or not assigned[0]):
            raise ValueError("Live trainer placement or accelerator assignment differs from the plan")
        previous = node_ids.setdefault(address, node_id)
        if previous != node_id:
            raise ValueError("One trainer address resolved to multiple Ray nodes")
        key = (node_id, assigned[0])
        if key in devices:
            raise ValueError("Trainer ranks were assigned the same accelerator")
        devices.add(key)
    if len(set(node_ids.values())) != len(nodes):
        raise ValueError("Multiple trainer addresses resolved to one Ray node")


def _verify_live_trainer_topology(config, *, rank, world_size, scheduled,
                                  events_available=None, producer_available=None):
    if not scheduled or world_size <= 1:
        return
    record = {"rank": rank}
    try:
        settings = config.speco.draft_training
        nnodes = int(settings.nnodes)
        nproc = int(settings.nproc_per_node)
        if nnodes < 1 or nproc < 1 or nnodes * nproc != world_size:
            raise ValueError("Configured trainer topology differs from distributed world size")
        raw_nodes = settings.trainer_node_ips
        nodes = None if raw_nodes is None else sorted(str(ip) for ip in raw_nodes)
        training = config.actor_rollout_ref.rollout.drafter.training
        transport = training.get("transfer_queue") or {}
        budget_keys = ("consumer_device_batch_bytes", "consumer_device_prefetch_bytes", "consumer_prefetch_host_bytes",
                       "consumer_prefetch_batch_bytes", "consumer_prefetch_batches", "consumer_batch_workers",
                       "consumer_node_reference_owner")
        record["plan"] = {"nnodes": nnodes, "nproc_per_node": nproc, "node_ips": nodes,
                          "feature_store": (training.get("feature_store") or {}).get("type"),
                          "reference_owned_windows": training.get("reference_owned_windows", False),
                          "per_rank_reference_ack": (getattr(settings, "scheduler", None) or {}).get("per_rank_reference_ack", False),
                          "release_payload_on_materialization": (getattr(settings, "scheduler", None) or {}).get("release_payload_on_materialization", False),
                          "commit_gate_timeout": (getattr(settings, "scheduler", None) or {}).get("commit_gate_timeout_seconds", 180),
                          "prefetch": {key: transport.get(key) for key in budget_keys}}
        hidden_store = (training.get("target_feature_replay") or {}).get("hidden_states_store") or {}
        record["plan"]["hidden_backend"] = str(hidden_store.get("backend", "file")).strip().lower()
        if record["plan"]["hidden_backend"] == "native":
            from verl_speco.trainer.tq_feature_store import _plain_dict
            producer = config.speco.get("standalone_tq_producer") or {}
            record["plan"]["native_reader"] = {"settings": _plain_dict(hidden_store.get("native")),
                                                "run_id": transport.get("run_id"),
                                                "publication_after_parent_commit": producer.get("publication_after_parent_commit", False),
                                                "max_published_references": producer.get("max_published_references", 0),
                                                "max_reference_execution_slots": producer.get("max_reference_execution_slots", 0)}
        if record["plan"]["per_rank_reference_ack"]:
            record["plan"]["events_available"] = events_available
            record["plan"]["producer_available"] = producer_available
        if transport.get("consumer_node_reference_owner", False):
            import ray
            node_id = ray.get_runtime_context().get_node_id()
            if node_id is None or not str(node_id).strip():
                raise ValueError("Node reference owner requires a verified Ray node ID")
            record["node_owner_node_id"] = str(node_id)
        if nnodes > 1:
            import ray
            context = ray.get_runtime_context()
            kind = {"cuda": "GPU", "npu": "NPU"}.get(str(get_device_name()).lower())
            if kind is None:
                raise ValueError("Multi-node Ray requires CUDA or NPU accelerator placement")
            record.update(ip=ray.util.get_node_ip_address().strip("[]"),
                          node_id=str(context.get_node_id()),
                          accelerators=[str(value) for value in context.get_accelerator_ids().get(kind, [])])
    except Exception as exc:  # All ranks must still enter the collective.
        record["error"] = f"{type(exc).__name__}: {exc}"
    gathered = [None] * world_size
    dist.all_gather_object(gathered, record)
    if any(not isinstance(item, dict) or item.get("error") for item in gathered):
        raise ValueError(f"Cannot verify live trainer placement: {gathered}")
    if {item.get("rank") for item in gathered} != set(range(world_size)):
        raise ValueError("Live trainer ranks are invalid or duplicated")
    plan = gathered[0].get("plan")
    if any(item.get("plan") != plan for item in gathered):
        raise ValueError("Live trainer ranks disagree on the configured topology")
    if plan["prefetch"].get("consumer_node_reference_owner", False):
        ordered = sorted(gathered, key=lambda item: item["rank"])
        node_ids = []
        for start in range(0, world_size, plan["nproc_per_node"]):
            ids = {item.get("node_owner_node_id") for item in ordered[start:start + plan["nproc_per_node"]]}
            if len(ids) != 1 or not next(iter(ids)):
                raise ValueError("Node reference owner ranks must share one verified Ray node")
            node_ids.append(next(iter(ids)))
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Different node reference owner groups cannot share a Ray node")
    if plan["nnodes"] > 1:
        if not plan["node_ips"] or len(plan["node_ips"]) != plan["nnodes"]:
            raise ValueError("Configured trainer node count does not match its IP list")
        _validate_live_trainer_topology(gathered, plan["node_ips"], plan["nproc_per_node"])


def run_standalone_draft_training(config) -> dict[str, Any]:
    """Run independent draft training from a feature store."""
    return asyncio.run(_run_standalone_draft_training_async(config))


def _validate_native_reader(*, per_rank_ack, node_reference_owner, owned_windows, tq_cfg, producer_cfg):
    if (not per_rank_ack or node_reference_owner or owned_windows
            or int(tq_cfg.get("consumer_batch_workers", 2)) != 0
            or not (producer_cfg.get("publication_after_parent_commit", False)
                    or producer_cfg.get("max_published_references", 0)
                    or producer_cfg.get("max_reference_execution_slots", 0))
            or (tq_cfg.get("expected_feature") or {}).get("payload_kind") != "vllm_reference"):
        raise ValueError("Native reader requires scheduled per-rank ACK, producer credits and in-process reference prefetch")


async def _run_standalone_draft_training_async(
    config,
    *,
    scheduled_commands: Any | None = None,
    training_events: Any | None = None,
    producer_handle: Any | None = None,
) -> dict[str, Any]:
    rank, local_rank, world_size = _init_distributed()
    standalone_trackers: list[Any] = []
    logger.info(
        "[standalone rank=%s] distributed runtime initialized local_rank=%s world_size=%s",
        rank,
        local_rank,
        world_size,
    )
    _verify_live_trainer_topology(config, rank=rank, world_size=world_size,
                                  scheduled=scheduled_commands is not None,
                                  events_available=training_events is not None,
                                  producer_available=producer_handle is not None)
    draft_config = config.actor_rollout_ref
    drafter_cfg = draft_config.rollout.drafter
    training_cfg = drafter_cfg.training
    resume_format = training_cfg.get("checkpoint_resume_format", "tensor")
    if not isinstance(resume_format, str) or resume_format not in {"tensor", "gzip"}:
        raise ValueError("checkpoint_resume_format must be tensor or gzip")
    _verify_rollback_checkpoint_across_ranks(training_cfg, drafter_cfg.get("model_path"), world_size)
    feature_store_cfg = training_cfg.feature_store
    feature_store_type = (
        str(feature_store_cfg.get("type", "torch_shard") or "torch_shard")
        .strip()
        .lower()
    )
    tq_cfg = training_cfg.get("transfer_queue") or {}
    node_reference_owner = tq_cfg.get("consumer_node_reference_owner", False)
    if type(node_reference_owner) is not bool:
        raise ValueError("consumer_node_reference_owner must be a boolean")
    node_ranks_per_host = int(config.speco.draft_training.get("nproc_per_node", world_size)) if node_reference_owner else 0
    if node_reference_owner and (feature_store_type != "ray_reference" or node_ranks_per_host < 2
            or world_size % node_ranks_per_host or not hasattr(os, "memfd_create")):
        raise ValueError("Node reference owner requires Linux ray_reference with at least two ranks per node")
    device_batch_bytes = _reference_device_budget(training_cfg.get("transfer_queue") or {}, feature_store_type)
    device_prefetch_bytes = _reference_device_prefetch_budget(
        training_cfg.get("transfer_queue") or {}, feature_store_type, str(get_device_name()).lower(), device_batch_bytes)
    owned_windows = training_cfg.get("reference_owned_windows", False)
    if type(owned_windows) is not bool:
        raise ValueError("reference_owned_windows must be a boolean")
    if owned_windows and (feature_store_type != "ray_reference" or scheduled_commands is None or training_events is None
            or str(drafter_cfg.get("speculative_algorithm", "")).lower() != "dspark"
            or str(((training_cfg.get("target_feature_replay") or {}).get("hidden_states_store") or {}).get("backend", "file")).lower() != "file"
            or device_prefetch_bytes or (not node_reference_owner and tq_cfg.get("consumer_prefetch_host_bytes", 0))):
        raise ValueError("Owned windows require scheduled file references, block training and no device prefetch")
    scheduler_cfg = config.speco.draft_training.get("scheduler") or {}
    commit_gate_timeout = scheduler_cfg.get("commit_gate_timeout_seconds", 180)
    if (isinstance(commit_gate_timeout, bool) or not isinstance(commit_gate_timeout, (int, float))
            or not math.isfinite(commit_gate_timeout) or commit_gate_timeout <= 0):
        raise ValueError("commit_gate_timeout_seconds must be a positive finite number")
    checkpoint_timeout = scheduler_cfg.get("checkpoint_finalization_timeout_seconds", 1800)
    if (isinstance(checkpoint_timeout, bool) or not isinstance(checkpoint_timeout, (int, float))
            or not math.isfinite(checkpoint_timeout) or checkpoint_timeout <= 0):
        raise ValueError("checkpoint_finalization_timeout_seconds must be a positive finite number")
    producer_options = (config.speco.get("standalone_tq_producer") if hasattr(config.speco, "get")
                        else getattr(config.speco, "standalone_tq_producer", None)) or {}
    journal_path = (producer_options.get("reference_lease_journal_path") if hasattr(producer_options, "get")
                    else getattr(producer_options, "reference_lease_journal_path", None))
    lease_snapshot_provider = None
    if journal_path is not None:
        if (producer_handle is None or scheduled_commands is None or feature_store_type != "ray_reference"
                or scheduler_cfg.get("per_rank_reference_ack", False) is not True):
            raise ValueError("Persistent lease checkpoint requires scheduled reference input and a Producer handle")
        def lease_snapshot_provider():
            import ray
            return ray.get(producer_handle.reference_lease_snapshot.remote(), timeout=checkpoint_timeout)
    per_rank_ack = scheduler_cfg.get("per_rank_reference_ack", False)
    early_payload_release = scheduler_cfg.get("release_payload_on_materialization", False)
    if type(early_payload_release) is not bool or (early_payload_release and not per_rank_ack):
        raise ValueError("release_payload_on_materialization requires per-rank ACK")
    if (type(per_rank_ack) is not bool or (per_rank_ack and
            (feature_store_type != "ray_reference" or scheduled_commands is None
             or training_events is None or producer_handle is None))):
        raise ValueError("per_rank_reference_ack requires scheduled ray_reference input, events and Producer handle")
    hidden_store_cfg = (training_cfg.get("target_feature_replay") or {}).get("hidden_states_store") or {}
    native_reader = str(hidden_store_cfg.get("backend", "file")).strip().lower() == "native"
    if early_payload_release and not native_reader and not owned_windows:
        batch_limit = tq_cfg.get("consumer_prefetch_batch_bytes", 0)
        host_limit = tq_cfg.get("consumer_prefetch_host_bytes", 0)
        if (device_prefetch_bytes or type(batch_limit) is not int or batch_limit <= 0
                or type(host_limit) is not int or host_limit < 2 * batch_limit):
            raise ValueError("Early file/Mooncake release requires a batch byte limit, twice its host budget and no device prefetch")
    if native_reader:
        _validate_native_reader(per_rank_ack=per_rank_ack, node_reference_owner=node_reference_owner,
                                owned_windows=owned_windows, tq_cfg=tq_cfg,
                                producer_cfg=config.speco.get("standalone_tq_producer") or {})
    training_mode = (
        str(training_cfg.get("mode", "offline") or "offline").strip().lower()
    )
    replay_feature_store_types = {"token_replay", "jsonl_token_replay", "jsonl"}
    if feature_store_type not in {"tq", "ray_reference"} and not feature_store_cfg.get(
        "path"
    ):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.feature_store.path is required"
        )
    if feature_store_type in {"tq", "ray_reference"} and training_mode != "offline":
        raise ValueError(
            f"feature_store.type={feature_store_type} requires standalone training.mode=offline"
        )
    if feature_store_type in replay_feature_store_types and training_mode != "offline":
        raise ValueError(
            f"feature_store.type={feature_store_type} is supported only by "
            "standalone training.mode=offline"
        )
    _disable_standalone_sequence_parallel(draft_config)
    _apply_standalone_fsdp_shard_default(draft_config)

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
    if lease_snapshot_provider is not None:
        trainer._standalone_lease_snapshot_provider = lease_snapshot_provider
    trainer._standalone_resume_format = resume_format
    trainer._standalone_preserve_packed_samples = True
    trainer._standalone_validate_grad_graph = True
    trainer._standalone_training_plan = {
        "world_size": world_size,
        "nnodes": int(config.speco.draft_training.get("nnodes", 1)),
        "nproc_per_node": int(config.speco.draft_training.get("nproc_per_node", world_size)),
        "algorithm": str(drafter_cfg.get("speculative_algorithm", "")),
        "batch_size_per_gpu": int(training_cfg.get("batch_size_per_gpu", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "seed": int(training_cfg.get("seed", 0) or 0),
        "deterministic_step_rng": training_cfg.get("deterministic_step_rng", False),
        **_standalone_rng_identity(training_cfg.get("deterministic_step_rng", False), trainer.runtime_device),
        "balance_by_tokens": scheduler_cfg.get("balance_by_tokens", False),
        "tail_policy": scheduler_cfg.get("tail_policy", "drop"),
        "recipe": _standalone_recipe_across_ranks(training_cfg, draft_config.model.get("path"), world_size),
    }
    retention_keep = training_cfg.get("checkpoint_retention_keep_last", 0)
    retention_delete = training_cfg.get("checkpoint_retention_delete", False)
    retention_inherit = training_cfg.get("checkpoint_retention_inherit_run", False)
    retention_settings = (retention_keep, retention_delete, training_cfg.get("checkpoint_directory_exclusive", False),
                          retention_inherit)
    _validate_checkpoint_retention_across_ranks(retention_settings, rank=rank, device=trainer.runtime_device)
    trainer._standalone_retention_keep_last = retention_keep
    trainer._standalone_retention_delete = retention_delete
    trainer._standalone_retention_run_id = uuid.uuid4().hex
    # Fail in the main thread before the first batch when a resumed checkpoint
    # declares another layer-ID convention. The export path repeats this check
    # while rewriting the runtime config, but only on the writer thread.
    _assert_standalone_layer_migration(trainer, getattr(backend, "model_type", None))
    max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 1000)) or 0)
    save_interval = int(training_cfg.get("save_interval_steps", 0) or 0)
    snapshot_interval = training_cfg.get("trainable_snapshot_interval_steps", 0)
    snapshot_max_bytes = training_cfg.get("trainable_snapshot_max_bytes", 0)
    if type(snapshot_interval) is not int or snapshot_interval < 0:
        raise ValueError("trainable_snapshot_interval_steps must be a non-negative integer")
    if snapshot_interval and not _all_ranks_true(bool(getattr(trainer, "checkpoint_dir", None)), trainer.runtime_device):
        raise ValueError("Trainable snapshot export requires a checkpoint directory")
    token_window = StandaloneTokenWindow(training_cfg.get("token_metrics_interval_steps", 50))
    successful_steps = 0
    initial_optimizer_step = 0
    optimizer_step = 0
    attempted_batches = 0
    last_save_result: dict[str, Any] | None = None
    last_saved_step = 0
    consumed_sequence_nos: set[int] = set()
    resume_metadata = None
    standalone_input_path = _standalone_input_path(config)
    trainer._standalone_retention_protected = tuple(
        str(path) for path in (drafter_cfg.get("model_path"), standalone_input_path) if path
    )
    standalone_producer_shuffle = _standalone_producer_shuffle(config)
    trainer.combine_standalone_metrics = feature_store_type == "ray_reference"
    trainer._standalone_checkpoint_timeout = float(checkpoint_timeout)
    store = None
    feature_replayer = None
    feature_producer = None
    tq_prefetch: TQPrefetchIterator | ReferenceBatchPrefetchIterator | None = None
    device_prefetch: ReferenceDevicePrefetchIterator | None = None
    tq_prefetch_group = None
    batch_finalizer = None
    checkpoint_directory_lease = None
    current_stage = "activate_training_model"
    try:
        checkpoint_directory_lease = _acquire_checkpoint_directory_across_ranks(
            getattr(trainer, "checkpoint_dir", None), training_cfg.get("checkpoint_directory_exclusive", False),
            rank=rank, device=trainer.runtime_device,
        )
        trainer._standalone_checkpoint_directory_lease = checkpoint_directory_lease
        # Scheduled Ray training delegates tracking to its Driver, which also
        # receives Producer windows. Subprocess/file training logs locally.
        if training_events is None:
            standalone_trackers = _build_standalone_tracking(config, rank=rank)
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
        snapshot_init = training_cfg.get("trainable_snapshot_initialization_path")
        if feature_store_type in {"tq", "ray_reference"}:
            consumed_sequence_nos, resume_metadata = _load_resume_across_ranks(
                drafter_cfg.get("model_path"), standalone_input_path, initial_optimizer_step,
                trainer._standalone_training_plan, world_size,
            )
            if feature_store_type == "ray_reference":
                trainer._standalone_input_identity = _input_identity_across_ranks(standalone_input_path, world_size)
            logger.info(
                "[standalone rank=%s] resume progress optimizer_step=%s consumed=%s",
                rank,
                initial_optimizer_step,
                len(consumed_sequence_nos),
            )
        if retention_inherit:
            trainer._standalone_retention_run_id = _retention_resume_identity_across_ranks(
                getattr(trainer, "checkpoint_dir", None), drafter_cfg.get("model_path"), initial_optimizer_step,
                resume_metadata, world_size)
        if snapshot_init is not None:
            current_stage = "initialize_trainable_snapshot"
            optimizer = getattr(trainer, "optimizer", None)
            fresh = (initial_optimizer_step == 0 and not consumed_sequence_nos and optimizer is not None
                     and hasattr(optimizer, "state") and not bool(optimizer.state))
            if not _all_ranks_true(fresh, trainer.runtime_device):
                raise ValueError("Trainable snapshot initialization requires fresh optimizer and data progress on every rank")
            _load_trainable_snapshot_across_ranks(trainer, snapshot_init, device=trainer.runtime_device,
                                                   max_bytes=snapshot_max_bytes)

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
        store_error = None
        native_store = None
        try:
            if native_reader:
                from verl_speco.producer.hidden_states_store import build_managed_native_reader
                from verl_speco.trainer.tq_feature_store import _plain_dict
                native_store = build_managed_native_reader(
                    _plain_dict(hidden_store_cfg.get("native")), run_id=str(tq_cfg.get("run_id") or ""),
                    rank=rank, world_size=world_size, device=trainer.runtime_device)
            store = build_feature_store_from_config(
                feature_store_cfg,
                read_only=True,
                transfer_queue_cfg=training_cfg.get("transfer_queue"),
                hidden_states_store_cfg=(
                    training_cfg.get("target_feature_replay") or {}
                ).get("hidden_states_store"),
                hidden_states_store=native_store,
            )
        except Exception as exc:
            if native_store is not None:
                native_store.close()
            store_error = exc
        if feature_store_type in {"tq", "ray_reference"}:
            if not _all_ranks_true(store_error is None, trainer.runtime_device):
                raise RuntimeError("Feature store construction failed on at least one rank") from store_error
        elif store_error is not None:
            raise store_error
        if feature_store_type in {"tq", "ray_reference"}:
            current_stage = "connect_feature_store"
            _connect_tq_store_across_ranks(
                store,
                rank=rank,
                device=trainer.runtime_device,
            )
        if feature_store_type == "ray_reference":
            current_stage = "verify_shared_file_root"
            _verify_shared_file_root_across_ranks(store, producer_handle, rank=rank, device=trainer.runtime_device,
                                                   timeout=float(tq_cfg.get("io_timeout_seconds", 30)))
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
        if feature_store_type in {"tq", "ray_reference"}:
            depth = prefetch_depth(tq_cfg)
            if feature_store_type == "ray_reference" and rank == 0:
                batch_finalizer = AsyncBatchFinalizer(capacity=depth + 1, timeout=commit_gate_timeout)
            hidden_references = (tq_cfg.get("expected_feature") or {}).get(
                "payload_kind"
            ) == "vllm_reference"
            if feature_store_type == "ray_reference" and not hidden_references:
                raise ValueError(
                    "ray_reference requires launcher-provided hidden reference metadata"
                )
            prepare_batch = None
            if hidden_references:
                if not depth or scheduled_commands is None:
                    raise ValueError(
                        "Reference input requires scheduler-controlled Consumer prefetch"
                    )
                if int(training_cfg.get("gradient_accumulation_steps", 1)) != 1:
                    raise ValueError(
                        "Reference input currently requires one optimizer step per scheduled batch"
                    )
                algorithm = str(drafter_cfg.speculative_algorithm).lower()
                packing_cfg = training_cfg.get("packing") or {}
                streaming_prepare = training_cfg.get("reference_streaming_prepare", False)
                if type(streaming_prepare) is not bool:
                    raise ValueError("reference_streaming_prepare must be a boolean")
                prepare_batch = partial(
                    store.prepare_batch,
                    max_window=training_cfg.get(
                        f"{algorithm}_max_window",
                        512 if algorithm == "dflash" else None,
                    ),
                    packing=bool(packing_cfg.get("enable", False)),
                    block_size=int(
                        getattr(
                            trainer.model_config,
                            "block_size",
                            training_cfg.get(f"{algorithm}_block_size", 16),
                        )
                    ),
                    max_packed_len=int(packing_cfg.get("max_packed_len", 0) or 0),
                    streaming=streaming_prepare,
                    max_output_bytes=tq_cfg.get("consumer_prefetch_batch_bytes", 0),
                )
            if depth and scheduled_commands is None:
                raise ValueError(
                    "consumer_prefetch_batches requires the scheduled Ray Consumer"
                )
            if depth:
                io_timeout = float(tq_cfg.get("io_timeout_seconds", 30))
                io_attempts = int(tq_cfg.get("io_retry_max_attempts", 3))
                collective_timeout = float(
                    tq_cfg.get("prefetch_collective_timeout_seconds", 180)
                )
                retry_deadline = io_attempts * (io_timeout + 5) + sum(
                    min(0.2 * 2**attempt, 2.0) * 1.1
                    for attempt in range(max(0, io_attempts - 1))
                )
                if (
                    io_timeout <= 0
                    or io_attempts < 1
                    or collective_timeout <= retry_deadline
                ):
                    raise ValueError(
                        "Prefetch collective timeout must exceed the complete data preparation retry deadline"
                    )
            if depth and world_size > 1:
                tq_prefetch_group = dist.new_group(
                    backend="gloo",
                    timeout=timedelta(
                        seconds=float(
                            tq_cfg.get("prefetch_collective_timeout_seconds", 180)
                        )
                    ),
                )
            loader = TQFeatureDataLoader(
                store,
                batch_size=int(training_cfg.get("batch_size_per_gpu", 4)),
                rank=rank,
                world_size=world_size,
                poll_interval_seconds=float(
                    tq_cfg.get("poll_interval_seconds", 0.5) or 0.5
                ),
                drop_last=bool(tq_cfg.get("drop_last", True)),
                scheduled_commands=scheduled_commands,
                process_group=tq_prefetch_group,
                io_retry_max_attempts=int(tq_cfg.get("io_retry_max_attempts", 3)),
                io_timeout_seconds=float(tq_cfg.get("io_timeout_seconds", 30)),
                prepare_batch=prepare_batch,
                reference_command_max_bytes=scheduler_cfg.get("reference_command_max_bytes", 256 * 1024 * 1024),
                reference_command_frame_bytes=scheduler_cfg.get("reference_command_frame_bytes", 4 * 1024 * 1024),
                node_ranks_per_host=node_ranks_per_host,
                # References are released only after materialization ACK and
                # all-rank optimizer success, never in background readers.
                cleanup_after_load=False,
            )
            if training_events is not None and rank == 0:
                ready_error = None
                try:
                    retry_io(
                        lambda: put_queue(
                            training_events,
                            {
                                "kind": "consumer_ready",
                                "initial_optimizer_step": initial_optimizer_step,
                                **({"input_identity": trainer._standalone_input_identity}
                                   if getattr(trainer, "_standalone_input_identity", None) is not None
                                   else {"input_identity_unavailable": True} if feature_store_type == "ray_reference" else {}),
                            },
                        ),
                        name="Consumer ready event",
                        attempts=int(tq_cfg.get("io_retry_max_attempts", 3)),
                    )
                except Exception as exc:  # noqa: BLE001 - communicate event failure to every rank
                    ready_error = exc
            else:
                ready_error = None
            if not _all_ranks_true(ready_error is None, trainer.runtime_device):
                raise RuntimeError(
                    "Failed to notify Scheduler that Consumer is ready"
                ) from ready_error
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
                    group_by_length=bool(training_cfg.get("group_by_length", False)),
                    group_by_length_megabatch=int(
                        training_cfg.get("group_by_length_megabatch", 8) or 8
                    ),
                    on_error=str(feature_store_cfg.get("on_error", "skip") or "skip"),
                    max_consecutive_errors=config_int(
                        feature_store_cfg, "max_consecutive_errors", 20
                    ),
                    min_supervised_tokens=int(
                        feature_store_cfg.get("min_supervised_tokens", 1) or 0
                    ),
                    strict_token_alignment=str(
                        feature_store_cfg.get("strict_token_alignment", "warn")
                        or "warn"
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
        prefetch_host_bytes = tq_cfg.get("consumer_prefetch_host_bytes", 0)
        prefetch_batch_bytes = tq_cfg.get("consumer_prefetch_batch_bytes", 0)
        if early_payload_release and not native_reader and not owned_windows:
            # Reserve one independent training copy outside the prefetch queue.
            prefetch_host_bytes -= prefetch_batch_bytes
        if owned_windows and node_reference_owner:
            _validate_node_window_host_budget(prefetch_host_bytes, prefetch_batch_bytes)
        if device_prefetch_bytes and prefetch_host_bytes:
            # One pinned batch is held outside the source queue during copy.
            prefetch_host_bytes -= prefetch_batch_bytes
        if ((prefetch_host_bytes or prefetch_batch_bytes) and
                (feature_store_type != "ray_reference" or (not depth and not early_payload_release and not (owned_windows and node_reference_owner)))):
            raise ValueError("Prefetch host byte limits require ray_reference prefetch")
        if owned_windows:
            sample_source = loader.iter_assignments()
        elif feature_store_type in {"tq", "ray_reference"} and depth:
            batch_workers = int(tq_cfg.get("consumer_batch_workers", 2))
            if batch_workers < 0:
                raise ValueError("consumer_batch_workers must be non-negative")
            prefetch_options = {
                "close_timeout": float(
                    tq_cfg.get("prefetch_collective_timeout_seconds", 180)
                )
                + 10,
                "pin_device": trainer.runtime_device,
            }
            if feature_store_type == "ray_reference" and batch_workers:
                assert prepare_batch is not None
                tq_prefetch = ReferenceBatchPrefetchIterator(
                    loader,
                    depth,
                    workers=batch_workers,
                    prepare_options=prepare_batch.keywords,
                    **prefetch_options,
                    defer_cleanup=True,
                    max_host_bytes=prefetch_host_bytes,
                    max_batch_host_bytes=prefetch_batch_bytes,
                    node_ranks_per_host=node_ranks_per_host,
                )
            else:
                if node_reference_owner:
                    raise ValueError("Node reference owner requires a process prefetch worker")
                tq_prefetch = TQPrefetchIterator(
                    loader,
                    depth,
                    pin_memory=hidden_references,
                    prepare_workers=int(tq_cfg.get("consumer_prepare_workers", 4)),
                    max_host_bytes=prefetch_host_bytes,
                    max_batch_host_bytes=prefetch_batch_bytes,
                    **prefetch_options,
                )
            sample_source = tq_prefetch
            if device_prefetch_bytes:
                device_prefetch = ReferenceDevicePrefetchIterator(
                    tq_prefetch, device=trainer.runtime_device, backend=get_torch_device(),
                    max_batch_bytes=device_batch_bytes, close_timeout=prefetch_options["close_timeout"],
                )
                sample_source = device_prefetch
        pipeline_cfg = training_cfg.get("target_feature_pipeline", {}) or {}
        pipeline_enabled = bool(pipeline_cfg.get("enabled", False))
        if feature_store_type in {"tq", "ray_reference"} and pipeline_enabled:
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
            load_started = time.perf_counter()
            commit_gate_seconds = 0.0
            if feature_store_type == "ray_reference" and successful_steps:
                current_stage = "wait_previous_batch_commit"
                _drain_batch_finalizer_across_ranks(batch_finalizer, trainer.runtime_device,
                                                    timeout=commit_gate_timeout)
                commit_gate_seconds = time.perf_counter() - load_started
            current_stage = "load_next_batch"
            fetch_started = time.perf_counter()
            fetch_timings: dict[str, float] = {}
            loaded_batch = _next_batch_across_ranks(
                sample_iterator,
                rank=rank,
                device=trainer.runtime_device,
                require_prepared=feature_store_type == "ray_reference" and not owned_windows,
                require_reference_metadata=owned_windows,
                node_ranks_per_host=node_ranks_per_host if owned_windows else 0,
                expected_input_identity=getattr(trainer, "_standalone_input_identity", None),
                timings=fetch_timings,
            )
            batch_ready_wait_seconds = time.perf_counter() - fetch_started
            if loaded_batch is None:
                break
            tq_local_batch = (
                loaded_batch if isinstance(loaded_batch, TQLocalBatch) else None
            )
            if feature_store_type == "ray_reference" and tq_local_batch is not None and tq_local_batch.input_identity is not None:
                trainer._standalone_input_identity = tq_local_batch.input_identity
            if feature_store_type == "ray_reference" and training_events is not None and not owned_windows:
                if early_payload_release and not native_reader:
                    detach_error = None
                    try:
                        tq_local_batch = _detach_reference_batch(tq_local_batch, tq_cfg)
                        loaded_batch = tq_local_batch
                    except Exception as exc:
                        detach_error = exc
                    if not _all_ranks_true(detach_error is None, trainer.runtime_device):
                        raise RuntimeError("Independent reference copy failed on at least one rank; source retained") from detach_error
                _ack_reference_materialization_across_ranks(
                    tq_local_batch, rank=rank, device=trainer.runtime_device,
                    events=training_events, config=tq_cfg, per_rank=per_rank_ack,
                    producer_handle=producer_handle,
                    **({"source_payload_safe": True} if early_payload_release else {}),
                )
            samples = (
                tq_local_batch.local_samples
                if tq_local_batch is not None
                else loaded_batch
            )
            processing_started = time.perf_counter()
            batch_prepare_seconds = 0.0
            train_seconds = 0.0
            tq_clear_seconds = 0.0
            tq_get_seconds = (
                float(tq_local_batch.tq_get_seconds)
                if tq_local_batch is not None
                else 0.0
            )
            attempted_batches += 1
            # Computing the slowest-rank TQ fetch time requires a collective.
            # Keep per-step precision while debugging, but avoid paying for a
            # logging-only all-reduce on every normal training step.
            if (
                tq_local_batch is not None
                and tq_local_batch.prepared_batch is None
                and (
                    logger.isEnabledFor(logging.DEBUG)
                    or attempted_batches % _TQ_GET_MAX_SYNC_INTERVAL_STEPS == 0
                )
            ):
                tq_get_seconds = _max_across_ranks_float(
                    tq_get_seconds,
                    trainer.runtime_device,
                )
            log_batch_progress = _should_log_batch_progress(attempted_batches)
            if log_batch_progress:
                logger.info(
                    "[standalone rank=%s] batch=%s loaded samples=%s "
                    "successful_steps=%s",
                    rank,
                    attempted_batches,
                    len(tq_local_batch.local_keys)
                    if tq_local_batch is not None
                    else len(samples),
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
            batch: dict[str, Any] | None
            batch_prepare_started = time.perf_counter()
            if owned_windows:
                batch = {}  # File parts are materialized during the training step.
            elif tq_local_batch is not None and tq_local_batch.prepared_batch is not None:
                batch = tq_local_batch.prepared_batch
                # Background work is not part of this main-thread wall time.
                batch_prepare_seconds = 0.0
            else:
                batch = trainer.prepare_training_batch_from_samples(
                    cast(list[Any], materialized_samples),
                    step=optimizer_step,
                )
                batch_prepare_seconds = time.perf_counter() - batch_prepare_started
            has_batch = batch is not None
            current_stage = "synchronize_batch_readiness"
            if feature_store_type != "ray_reference" and not _all_ranks_true(
                has_batch, trainer.runtime_device
            ):
                if tq_local_batch is not None:
                    raise RuntimeError(
                        "Standalone Consumer could not prepare a valid batch on every rank; "
                        "the source samples were intentionally retained"
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
            if trainer._standalone_training_plan["deterministic_step_rng"]:
                _seed_standalone_step(trainer._standalone_training_plan["seed"], optimizer_step, rank)
            train_started = time.perf_counter()
            if owned_windows:
                token_counts = {}
                ok = await _execute_owned_reference_batch(
                    trainer, store, tq_local_batch, optimizer_step, rank=rank, device=trainer.runtime_device,
                    events=training_events, transport_config=tq_cfg, per_rank=per_rank_ack,
                    **({"source_payload_safe": True} if early_payload_release else {}),
                    producer_handle=producer_handle, max_window=training_cfg.get("dspark_max_window"),
                    max_host_bytes=prefetch_batch_bytes, max_device_bytes=device_batch_bytes, token_counts=token_counts,
                    packing=prepare_batch.keywords["packing"], packing_block_size=prepare_batch.keywords["block_size"],
                    max_packed_len=prepare_batch.keywords["max_packed_len"], node_ranks_per_host=node_ranks_per_host,
                )
                batch["_speco_token_counts"] = token_counts
            elif feature_store_type == "ray_reference":
                ok = await _reference_training_step_across_ranks(
                    trainer, batch, optimizer_step, device=trainer.runtime_device,
                    max_device_bytes=device_batch_bytes,
                )
            else:
                ok = await trainer.training_step_from_batch(batch, optimizer_step)
            train_seconds = time.perf_counter() - train_started
            step_error = getattr(trainer, "last_standalone_training_error", None)
            if step_error is not None and _is_out_of_memory_error(step_error):
                raise RuntimeError(
                    "Standalone drafter training hit an unrecoverable OOM during "
                    f"batch={attempted_batches} optimizer_step={optimizer_step}. "
                    "Reduce batch_size_per_gpu, feature_store.max_seq_len, "
                    "dspark_num_anchors/block_size or disable DSpark L1 loss."
                ) from step_error
            current_stage = "synchronize_training_step"
            if feature_store_type != "ray_reference" and not _all_ranks_true(
                ok, trainer.runtime_device
            ):
                if tq_local_batch is not None:
                    raise RuntimeError(
                        "Standalone Consumer training_step_from_batch failed on at least one rank; "
                        "the source samples were intentionally retained"
                    )
                continue
            if tq_local_batch is not None:
                current_stage = "finalize_completed_batch"
                tq_clear_started = time.perf_counter()
                _clear_tq_batch_across_ranks(
                    cast(TQFeatureDataLoader, loader),
                    tq_local_batch.global_keys,
                    rank=rank,
                    device=trainer.runtime_device,
                    training_events=training_events,
                    completion_event={
                        "kind": "training_completed",
                        "plan_id": tq_local_batch.plan_id,
                        "receipt_token": tq_local_batch.receipt_token,
                        "keys": list(tq_local_batch.global_keys or []),
                        "successful": True,
                    },
                    finalizer=batch_finalizer,
                    synchronize=feature_store_type != "ray_reference",
                    cleanup_after_load=tq_local_batch.cleanup_after_load or early_payload_release,
                    synchronize_before_cleanup=(
                        feature_store_type == "ray_reference"
                        and (int(config.speco.draft_training.nnodes) > 1
                             or tq_local_batch.has_padding_ranks)
                    ),
                )
                tq_clear_seconds = time.perf_counter() - tq_clear_started
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
                step_elapsed_sec=(
                    time.perf_counter() - load_started
                    if tq_local_batch is not None
                    else time.perf_counter() - processing_started
                ),
            )
            step_metrics["perf/batch_prepare_time"] = batch_prepare_seconds
            step_metrics["perf/train_time"] = train_seconds
            step_metrics["perf/batch_finalize_time"] = tq_clear_seconds
            if feature_store_type == "ray_reference":
                step_metrics["perf/commit_gate_wait_time"] = commit_gate_seconds
            if feature_store_type == "ray_reference":
                step_metrics.update(token_window.update(
                    batch.get("_speco_token_counts"), successful_steps=successful_steps,
                    optimizer_step=optimizer_step, max_steps=max_steps, start_time=load_started,
                    device=trainer.runtime_device, world_size=world_size,
                ))
            if tq_local_batch is not None:
                step_metrics["perf/batch_ready_wait_time"] = batch_ready_wait_seconds
                step_metrics.update(
                    (f"perf/{key}", value) for key, value in fetch_timings.items()
                )
                step_metrics.update(
                    (f"perf/{key}", value)
                    for key, value in tq_local_batch.delivery_times.items()
                )
                step_metrics["perf/loader_plan_wait_time"] = float(
                    tq_local_batch.command_wait_seconds
                )
                if getattr(tq_local_batch, "prepared_batch", None) is None:
                    step_metrics["perf/loader_tq_get_time"] = tq_get_seconds
                else:
                    step_metrics["perf/loader_batch_prepare_time"] = float(
                        tq_local_batch.batch_prepare_seconds
                    )
            if feature_replayer is not None:
                step_metrics.update(feature_replayer.metrics())
            if feature_producer is not None:
                step_metrics.update(feature_producer.metrics())
            _log_standalone_step_metrics(step_metrics, rank=rank)
            tracking_metrics = _select_standalone_tracking_metrics(step_metrics)
            if training_events is not None and rank == 0:
                event = {
                    "kind": "training_metrics",
                    "step": optimizer_step,
                    "metrics": tracking_metrics,
                }
                if batch_finalizer is not None:
                    _submit_standalone_metrics(
                        batch_finalizer, training_events, event, tq_cfg
                    )
                else:
                    training_events.put(event)
            else:
                _log_standalone_tracking_metrics(
                    standalone_trackers,
                    tracking_metrics,
                    step=optimizer_step,
                )
            if snapshot_interval and optimizer_step % snapshot_interval == 0:
                current_stage = "save_trainable_snapshot"
                snapshot_started = time.perf_counter()
                if feature_store_type == "ray_reference":
                    _drain_batch_finalizer_across_ranks(batch_finalizer, trainer.runtime_device,
                                                        timeout=commit_gate_timeout)
                _save_trainable_snapshot_across_ranks(trainer, optimizer_step, device=trainer.runtime_device,
                                                       max_bytes=snapshot_max_bytes)
                if feature_store_type == "ray_reference":
                    token_window.record_checkpoint(time.perf_counter() - snapshot_started)
            if save_interval > 0 and optimizer_step % save_interval == 0:
                checkpoint_block_started = time.perf_counter()
                if feature_store_type == "ray_reference":
                    _drain_batch_finalizer_across_ranks(
                        batch_finalizer, trainer.runtime_device, timeout=commit_gate_timeout
                    )
                _sync_standalone_export_error(trainer, trainer.runtime_device)
                current_stage = "save_checkpoint"
                checkpoint_started = time.perf_counter()
                if rank == 0:
                    logger.info(
                        "[standalone rank=%s] checkpoint saving step=%s",
                        rank,
                        optimizer_step,
                    )
                last_save_result = _save_standalone_checkpoint_across_ranks(
                    trainer,
                    optimizer_step,
                    device=trainer.runtime_device,
                    consumed_sequence_nos=consumed_sequence_nos,
                    input_path=(
                        standalone_input_path
                        if feature_store_type in {"tq", "ray_reference"}
                        else None
                    ),
                    producer_shuffle=(
                        standalone_producer_shuffle
                        if feature_store_type in {"tq", "ray_reference"}
                        else None
                    ),
                )
                if rank == 0:
                    logger.info(
                        "[standalone rank=%s] checkpoint saved step=%s elapsed=%.3fs",
                        rank,
                        optimizer_step,
                        time.perf_counter() - checkpoint_started,
                    )
                if _sync_any_rank_saved_checkpoint(last_save_result.get("saved")):
                    last_saved_step = optimizer_step
                _barrier()
                if feature_store_type == "ray_reference":
                    token_window.record_checkpoint(time.perf_counter() - checkpoint_block_started)
            if device_prefetch is not None:
                # Release the previous input before taking the next one, so the
                # lookahead owns no more than the current and one future batch.
                del batch, loaded_batch, tq_local_batch
            current_stage = "load_next_batch"
        if feature_store_type == "ray_reference":
            _drain_batch_finalizer_across_ranks(batch_finalizer, trainer.runtime_device, timeout=commit_gate_timeout)
        # Catch an export failure reported after the last in-loop check, before
        # the final save can succeed and end the run as a silent partial export.
        _sync_standalone_export_error(trainer, trainer.runtime_device)
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0 and optimizer_step != last_saved_step:
            current_stage = "save_final_checkpoint"
            checkpoint_started = time.perf_counter()
            if rank == 0:
                logger.info(
                    "[standalone rank=%s] final checkpoint saving step=%s",
                    rank,
                    optimizer_step,
                )
            last_save_result = _save_standalone_checkpoint_across_ranks(
                trainer,
                optimizer_step,
                device=trainer.runtime_device,
                wait=True,
                consumed_sequence_nos=consumed_sequence_nos,
                input_path=(
                    standalone_input_path
                    if feature_store_type in {"tq", "ray_reference"}
                    else None
                ),
                producer_shuffle=(
                    standalone_producer_shuffle
                    if feature_store_type in {"tq", "ray_reference"}
                    else None
                ),
            )
            if rank == 0:
                logger.info(
                    "[standalone rank=%s] final checkpoint saved step=%s elapsed=%.3fs",
                    rank,
                    optimizer_step,
                    time.perf_counter() - checkpoint_started,
                )
            _barrier()
            if feature_store_type == "ray_reference":
                token_window.record_checkpoint(time.perf_counter() - checkpoint_started)
        if feature_store_type == "ray_reference" and successful_steps:
            run_metrics = token_window.finish(device=trainer.runtime_device, world_size=world_size)
            if run_metrics and training_events is not None and rank == 0:
                event = {"kind": "training_metrics", "step": optimizer_step, "metrics": run_metrics}
                if batch_finalizer is not None:
                    _submit_standalone_metrics(batch_finalizer, training_events, event, tq_cfg)
                else:
                    training_events.put(event)
            elif run_metrics and training_events is None:
                _log_standalone_tracking_metrics(standalone_trackers, run_metrics, step=optimizer_step)
    except Exception as training_error:
        logger.exception(
            "[standalone rank=%s] training failed stage=%s attempted_batches=%s "
            "successful_steps=%s optimizer_step=%s",
            rank,
            current_stage,
            attempted_batches,
            successful_steps,
            optimizer_step,
        )
        if feature_store_type == "ray_reference" and training_events is not None:
            try:
                put_queue(
                    training_events,
                    {
                        "kind": "consumer_failed",
                        "rank": rank,
                        "error": str(training_error),
                    },
                    timeout=30,
                )
            except Exception:
                logger.exception("Failed to notify Driver of Consumer failure")
        raise
    finally:
        original_error = sys.exc_info()[1]
        prefetch_cleanup_error = None
        if batch_finalizer is not None:
            try:
                batch_finalizer.close(timeout=commit_gate_timeout)
            except Exception as error:  # noqa: BLE001 - still close the remaining resources
                prefetch_cleanup_error = error
                logger.exception("Failed to drain completed reference batches")
        _finish_standalone_tracking(standalone_trackers, rank=rank)
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
        if device_prefetch is not None:
            try:
                device_prefetch.close()
            except Exception as error:
                prefetch_cleanup_error = error
                logger.exception("Failed to close device prefetch")
        # Device prefetch owns and closes its underlying CPU prefetch source.
        if tq_prefetch is not None and device_prefetch is None:
            try:
                tq_prefetch.close()
            except Exception as error:  # noqa: BLE001 - still release the remaining resources
                prefetch_cleanup_error = error
                logger.exception("Failed to close TQ prefetch thread")
        prefetch_running = any(bool(getattr(prefetch, "has_running_work", False))
                               for prefetch in (device_prefetch, tq_prefetch) if prefetch is not None)
        if prefetch_running and prefetch_cleanup_error is None:
            prefetch_cleanup_error = RuntimeError("Prefetch work is still running after shutdown")
        if tq_prefetch_group is not None and not prefetch_running:
            try:
                dist.destroy_process_group(tq_prefetch_group)
            except Exception as error:  # noqa: BLE001 - still close the client and trainer
                if prefetch_cleanup_error is None:
                    prefetch_cleanup_error = error
                logger.exception("Failed to destroy TQ prefetch process group")
        if feature_replayer is not None and not prefetch_running:
            feature_replayer.close()
        if store is not None:
            if prefetch_running or (batch_finalizer is not None and batch_finalizer.has_running_work):
                logger.error("Preserving the reference store while background I/O remains in flight")
            else:
                store.close()
        finalization = getattr(trainer, "_standalone_finalization_event", None)
        checkpoint_wait_started = time.monotonic()
        checkpoint_wait_timeout = getattr(trainer, "_standalone_checkpoint_timeout", None)
        checkpoint_running = False
        if finalization is not None:
            if original_error is not None or prefetch_cleanup_error is not None or prefetch_running:
                checkpoint_running = not finalization.is_set()
            else:
                checkpoint_running = not finalization.wait(timeout=checkpoint_wait_timeout)
                if checkpoint_running and prefetch_cleanup_error is None:
                    prefetch_cleanup_error = TimeoutError("Checkpoint finalization did not stop before its deadline")
        pending_saves = [future for future in (getattr(trainer, "_pending_full_checkpoint_future", None),
                                                getattr(trainer, "_pending_checkpoint_future", None))
                         if future is not None and not future.done()]
        if pending_saves and not (original_error is not None or prefetch_cleanup_error is not None
                                  or prefetch_running or checkpoint_running):
            wrapped = [asyncio.wrap_future(future) for future in pending_saves]
            remaining = (None if checkpoint_wait_timeout is None else
                         max(0.0, checkpoint_wait_timeout - (time.monotonic() - checkpoint_wait_started)))
            finished, pending = await asyncio.wait(wrapped, timeout=remaining)
            for future in finished:
                if not future.cancelled():
                    future.exception()  # The trainer will report the original save error.
            checkpoint_running = bool(pending)
            if checkpoint_running and prefetch_cleanup_error is None:
                prefetch_cleanup_error = TimeoutError("Checkpoint save did not stop before its deadline")
        else:
            checkpoint_running = checkpoint_running or bool(pending_saves)
        resource_running = prefetch_running or checkpoint_running
        logger.info("[standalone rank=%s] cleaning trainer resources", rank)
        if resource_running:
            logger.error("Preserving trainer and communication resources until background work is fenced")
        else:
            await trainer.cleanup_training(clear_data=True)
        failed_file_run = (
            feature_store_type == "ray_reference" and (original_error is not None or resource_running)
        )
        final_export_failed = (
            False
            if failed_file_run or resource_running
            else _sync_standalone_export_error(
                trainer, trainer.runtime_device, raise_on_error=False
            )
        )
        all_ranks_clean = (original_error is None and prefetch_cleanup_error is None
                           and not final_export_failed and not resource_running)
        if dist.is_initialized() and not resource_running:
            if not failed_file_run:
                if training_cfg.get("checkpoint_directory_exclusive", False):
                    all_ranks_clean = _all_ranks_true(all_ranks_clean, trainer.runtime_device)
                logger.info(
                    "[standalone rank=%s] entering final process-group barrier", rank
                )
                dist.barrier()
                logger.info(
                    "[standalone rank=%s] final process-group barrier complete", rank
                )
            if rank == 0 and checkpoint_directory_lease is not None and all_ranks_clean:
                checkpoint_directory_lease.release()
                checkpoint_directory_lease = None
            dist.destroy_process_group()
        elif not dist.is_initialized() and checkpoint_directory_lease is not None and all_ranks_clean:
            checkpoint_directory_lease.release()
        logger.info("[standalone rank=%s] cleanup complete", rank)
        # cleanup_training waits for the pending full-checkpoint writer. Its
        # standalone finalize callback may only report an export failure while
        # that wait is completing, after the last check in the training loop.
        if final_export_failed:
            _raise_standalone_export_error(trainer)
            raise RuntimeError("Standalone checkpoint export failed on another rank")
        if prefetch_cleanup_error is not None and original_error is None:
            raise prefetch_cleanup_error

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


def _trainable_snapshot_parameters(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    source = model if isinstance(model, FSDP) else getattr(model, "module", model)
    parameters = {}
    for raw_name, parameter in source.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            name = raw_name.replace("_fsdp_wrapped_module.", "")
            if not name or name in parameters:
                raise ValueError("Trainable snapshot parameter names are ambiguous")
            parameters[name] = parameter
    if not parameters:
        raise ValueError("Trainable snapshot contains no trainable parameters")
    return parameters


def _save_trainable_snapshot_across_ranks(trainer, step, *, device, max_bytes=0):
    """Collect trainable parameters after a committed step and publish on rank zero."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl_speco.trainer.checkpoint import write_standalone_trainable_snapshot

    model = trainer.model
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    context = (FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=False)
               if isinstance(model, FSDP) else contextlib.nullcontext())
    parameters, weights, error = {}, {}, None
    try:
        with context:
            try:
                if type(max_bytes) is not int or max_bytes < 0:
                    raise ValueError("Trainable snapshot max_bytes must be a non-negative integer")
                parameters = _trainable_snapshot_parameters(model)
                logical_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters.values())
                if max_bytes and logical_bytes > max_bytes:
                    raise ValueError("Trainable snapshot exceeds its logical tensor byte limit")
            except Exception as exc:  # noqa: BLE001
                error = exc
            if not _all_ranks_true(error is None, device):
                raise RuntimeError("Trainable snapshot parameter discovery failed on a rank") from error
            schema = [(name, list(parameter.shape), str(parameter.dtype))
                      for name, parameter in sorted(parameters.items())]
            if distributed:
                schemas = [None] * dist.get_world_size()
                dist.all_gather_object(schemas, (schema, max_bytes))
                if any(item != schemas[0] for item in schemas):
                    raise ValueError("Trainable snapshot parameter schemas differ across ranks")
            for name, parameter in sorted(parameters.items()):
                error = None
                try:
                    value = parameter.detach()
                    if callable(getattr(value, "full_tensor", None)):
                        value = value.full_tensor()
                    if rank == 0:
                        weights[name] = value.to(device="cpu").contiguous().clone()
                except Exception as exc:  # noqa: BLE001
                    error = exc
                if not _all_ranks_true(error is None, device):
                    raise RuntimeError(f"Trainable snapshot gathering failed for {name}") from error
    except Exception as exc:  # noqa: BLE001
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Trainable snapshot collection failed on a rank") from error
    result = [None]
    if rank == 0:
        try:
            path = Path(trainer.checkpoint_dir) / f"trainable_step_{step}_{uuid.uuid4().hex}"
            result[0] = {"path": write_standalone_trainable_snapshot(
                path, step=step, tensors=weights, training_plan=trainer._standalone_training_plan), "error": None}
        except Exception as exc:  # noqa: BLE001
            result[0] = {"path": None, "error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[0]["error"] is not None:
        raise RuntimeError(f"Trainable snapshot publication failed: {result[0]['error']}")
    logger.info("Standalone trainable-only snapshot step=%s path=%s", step, result[0]["path"])
    return result[0]["path"]


def _load_trainable_snapshot_across_ranks(trainer, path, *, device, max_bytes=0):
    """Initialize a matching model; all ranks validate and stage before any copy."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from verl_speco.trainer.checkpoint import read_standalone_trainable_snapshot

    model = trainer.model
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    context = (FSDP.summon_full_params(model, recurse=True, writeback=True, rank0_only=False)
               if isinstance(model, FSDP) else contextlib.nullcontext())
    error, manifest, parameters, staged = None, None, {}, {}
    try:
        with context:
            try:
                if type(max_bytes) is not int or max_bytes < 0:
                    raise ValueError("Trainable snapshot max_bytes must be a non-negative integer")
                weights, manifest = read_standalone_trainable_snapshot(
                    path, expected_training_plan=trainer._standalone_training_plan)
                if max_bytes and sum(tensor.numel() * tensor.element_size() for tensor in weights.values()) > max_bytes:
                    raise ValueError("Trainable snapshot exceeds its logical tensor byte limit")
                parameters = _trainable_snapshot_parameters(model)
                if set(weights) != set(parameters):
                    raise ValueError("Trainable snapshot parameter names differ")
                aliases = {}
                for name, parameter in parameters.items():
                    tensor = weights[name]
                    if tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != parameter.dtype:
                        raise ValueError(f"Trainable snapshot tensor differs: {name}")
                    previous = aliases.setdefault(id(parameter), name)
                    if previous != name and not torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),
                                                             weights[previous].contiguous().reshape(-1).view(torch.uint8)):
                        raise ValueError("Trainable snapshot contains conflicting tied parameters")
                    local = parameter
                    if hasattr(parameter, "placements"):
                        if any(not isinstance(item, Replicate) and not (isinstance(item, Shard) and item.dim == 0)
                               for item in parameter.placements):
                            raise ValueError("Trainable snapshot supports only replication and shard(0)")
                        shape, offset = compute_local_shape_and_global_offset(
                            parameter.shape, parameter.device_mesh, parameter.placements)
                        tensor = tensor[tuple(slice(start, start + length) for start, length in zip(offset, shape, strict=True))]
                        local = parameter.to_local()
                    if tuple(tensor.shape) != tuple(local.shape):
                        raise ValueError(f"Trainable snapshot local tensor differs: {name}")
                    staged[name] = tensor.to(device=local.device).contiguous().clone()
            except Exception as exc:  # noqa: BLE001
                error = exc
            if not _all_ranks_true(error is None, device):
                raise RuntimeError("Trainable snapshot validation failed on a rank") from error
            if distributed:
                identities = [None] * dist.get_world_size()
                dist.all_gather_object(identities, (manifest["weights_sha256"], manifest["optimizer_step"], max_bytes))
                if any(item != identities[0] for item in identities):
                    raise ValueError("Trainable snapshot differs across ranks")
            try:
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        local = parameter.to_local() if hasattr(parameter, "placements") else parameter
                        local.copy_(staged[name])
            except Exception as exc:  # noqa: BLE001
                error = exc
            if not _all_ranks_true(error is None, device):
                raise RuntimeError("Trainable snapshot apply failed; discard the model") from error
    except Exception as exc:  # noqa: BLE001
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Trainable snapshot initialization failed on a rank; discard the model") from error
    return manifest


def _wait_standalone_finalization(trainer) -> None:
    event = getattr(trainer, "_standalone_finalization_event", None)
    timeout = getattr(trainer, "_standalone_checkpoint_timeout", None)
    if event is not None and not event.wait(timeout=timeout):
        error = TimeoutError("Standalone checkpoint finalization timed out; checkpoint remains unresolved")
        if getattr(trainer, "_standalone_export_error", None) is None:
            trainer._standalone_export_error = error
        raise error


def _save_standalone_checkpoint(
    trainer: DrafterBaseTrainer,
    step: int,
    *,
    wait: bool = False,
    consumed_sequence_nos: set[int] | None = None,
    input_path: str | None = None,
    producer_shuffle: dict[str, Any] | None = None,
    defer_finalization: bool = False,
) -> dict[str, Any]:
    # A Future becomes done before its callbacks finish. Do not overlap a new
    # save with the preceding sidecar/manifest publication or lose its error.
    _wait_standalone_finalization(trainer)
    _raise_standalone_export_error(trainer)
    consumed_snapshot = (
        torch.tensor(sorted(consumed_sequence_nos or ()), dtype=torch.int64)
        if not defer_finalization else None
    )
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if callable(save_checkpoint):
        result = save_checkpoint(int(step), wait=wait, defer_completion=True)
        checkpoint_path = result.get("path")
        is_export_leader = result.get("reason") in {"saved", "scheduled"}
        if result.get("saved") and checkpoint_path and is_export_leader and not defer_finalization:
            if wait:
                _finalize_standalone_checkpoint_files(
                    trainer,
                    checkpoint_path,
                    consumed_snapshot=consumed_snapshot,
                    step=step,
                    input_path=input_path,
                    producer_shuffle=producer_shuffle,
                )
            else:
                future = getattr(trainer, "_pending_full_checkpoint_future", None)
                if future is not None:
                    _attach_standalone_finalizer(
                        trainer, future, checkpoint_path,
                        consumed_snapshot=consumed_snapshot, step=step,
                        input_path=input_path, producer_shuffle=producer_shuffle,
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
        try:
            future.result(timeout=getattr(trainer, "_standalone_checkpoint_timeout", None))
        finally:
            if future.done():
                trainer._pending_full_checkpoint_future = None
        if not defer_finalization:
            _finalize_standalone_checkpoint_files(
                trainer,
                checkpoint_path,
                consumed_snapshot=consumed_snapshot,
                step=step,
                input_path=input_path,
                producer_shuffle=producer_shuffle,
            )
    elif future is not None and not defer_finalization:
        _attach_standalone_finalizer(
            trainer, future, checkpoint_path,
            consumed_snapshot=consumed_snapshot, step=step,
            input_path=input_path, producer_shuffle=producer_shuffle,
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


def _broadcast_checkpoint_progress(consumed_sequence_nos, *, writer_rank: int, device: torch.device):
    """Copy rank-zero progress to a nonzero writer in bounded device chunks."""
    rank = dist.get_rank()
    count = torch.tensor([len(consumed_sequence_nos or ()) if rank == 0 else 0],
                         dtype=torch.int64, device=device)
    dist.broadcast(count, src=0)
    total = int(count.item())
    error = None
    source, target, scratch = None, None, None
    try:
        if rank == 0:
            source = torch.tensor(sorted(consumed_sequence_nos or ()), dtype=torch.int64)
        if rank == writer_rank:
            target = torch.empty(total, dtype=torch.int64)
        if total:
            scratch = torch.empty(min(total, 65536), dtype=torch.int64, device=device)
    except Exception as exc:  # Every live rank must agree before broadcasting chunks.
        error = exc
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError("Failed to allocate checkpoint progress transfer") from error
    for start in range(0, total, 65536):
        length = min(65536, total - start)
        chunk = scratch[:length]
        if rank == 0:
            chunk.copy_(source[start:start + length])
        dist.broadcast(chunk, src=0)
        if rank == writer_rank:
            target[start:start + length].copy_(chunk)
    return target


def _save_standalone_checkpoint_across_ranks(
    trainer: DrafterBaseTrainer,
    step: int,
    *,
    device: torch.device,
    wait: bool = False,
    consumed_sequence_nos: set[int] | None = None,
    input_path: str | None = None,
    producer_shuffle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish only after every rank reports that its save phase returned."""
    result: dict[str, Any] = {}
    local_error: BaseException | None = None
    try:
        result = _save_standalone_checkpoint(
            trainer, step, wait=wait, consumed_sequence_nos=consumed_sequence_nos,
            input_path=input_path, producer_shuffle=producer_shuffle, defer_finalization=True,
        )
    except Exception as exc:  # all live ranks must reach the agreement
        local_error = exc
    writer = bool(result.get("saved") and result.get("path")
                  and result.get("reason") in {"saved", "scheduled"})
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    state = torch.tensor([int(local_error is not None), int(writer), int(bool(result.get("saved"))),
                          rank + 1 if writer else 0], dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
    failures, writers, saved, writer_rank_plus_one = state.tolist()
    if failures:
        raise RuntimeError("Standalone checkpoint save failed on at least one rank") from local_error
    if saved and writers != 1:
        raise RuntimeError(f"Standalone checkpoint requires exactly one export writer, got {writers}")
    if saved and saved != (dist.get_world_size() if distributed else 1):
        raise RuntimeError("Standalone checkpoint was not saved on every rank")
    writer_rank = writer_rank_plus_one - 1
    transferred_snapshot = None
    if distributed and writers == 1 and writer_rank != 0:
        transferred_snapshot = _broadcast_checkpoint_progress(
            consumed_sequence_nos, writer_rank=writer_rank, device=device,
        )
    publication_error: BaseException | None = None
    if writer:
        try:
            snapshot = (transferred_snapshot if transferred_snapshot is not None
                        else torch.tensor(sorted(consumed_sequence_nos or ()), dtype=torch.int64))
            kwargs = dict(consumed_snapshot=snapshot, step=step, input_path=input_path,
                          producer_shuffle=producer_shuffle)
            if wait:
                _finalize_standalone_checkpoint_files(trainer, result["path"], **kwargs)
            else:
                future = getattr(trainer, "_pending_full_checkpoint_future", None)
                if future is None:
                    raise RuntimeError("Scheduled checkpoint has no export future")
                _attach_standalone_finalizer(trainer, future, result["path"], **kwargs)
                publication_error = getattr(trainer, "_standalone_export_error", None)
        except Exception as exc:
            publication_error = exc
    state = torch.tensor([int(publication_error is not None)], dtype=torch.int32, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(state, op=dist.ReduceOp.MAX)
    if bool(state.item()):
        raise RuntimeError("Standalone checkpoint publication failed on at least one rank") from publication_error
    return result


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


def _attach_standalone_finalizer(trainer, future, checkpoint_path, **kwargs):
    """Track callback completion separately from the underlying export future."""
    complete = threading.Event()
    trainer._standalone_finalization_event = complete

    def finalize(finished):
        try:
            _finalize_standalone_checkpoint(trainer, checkpoint_path, finished, **kwargs)
        finally:
            complete.set()

    try:
        future.add_done_callback(finalize)
    except BaseException:
        complete.set()
        raise


def _finalize_standalone_checkpoint(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future,
    *,
    consumed_snapshot: torch.Tensor | None = None,
    step: int | None = None,
    input_path: str | None = None,
    producer_shuffle: dict[str, Any] | None = None,
) -> None:
    try:
        completed_future.result()
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[standalone] checkpoint writer failed step=%s path=%s",
            step,
            checkpoint_path,
        )
        try:
            _set_standalone_checkpoint_complete(checkpoint_path, complete=False)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[standalone] failed to invalidate checkpoint step=%s path=%s",
                step,
                checkpoint_path,
            )
        if getattr(trainer, "_standalone_export_error", None) is None:
            setattr(trainer, "_standalone_export_error", exc)
        return

    # This callback runs on the checkpoint writer thread, where concurrent.futures
    # only logs a raise. Record it so the training loop can fail from the main
    # thread instead of finishing with an unmigrated runtime config.
    try:
        _finalize_standalone_checkpoint_files(
            trainer,
            checkpoint_path,
            consumed_snapshot=consumed_snapshot,
            step=step,
            input_path=input_path,
            producer_shuffle=producer_shuffle,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[standalone] checkpoint export failed step=%s path=%s",
            step,
            checkpoint_path,
        )
        if getattr(trainer, "_standalone_export_error", None) is None:
            setattr(trainer, "_standalone_export_error", exc)


def _finalize_standalone_checkpoint_files(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    *,
    consumed_snapshot: torch.Tensor | None,
    step: int | None,
    input_path: str | None,
    producer_shuffle: dict[str, Any] | None = None,
) -> None:
    """Commit standalone-only files before publishing a managed checkpoint."""

    _set_standalone_checkpoint_complete(checkpoint_path, complete=False)
    _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
    lease_snapshot = None
    provider = getattr(trainer, "_standalone_lease_snapshot_provider", None)
    if provider is not None:
        from verl_speco.trainer.standalone_resume import validate_reference_lease_snapshot
        from verl_speco.trainer.checkpoint import atomic_write_json
        lease_snapshot = validate_reference_lease_snapshot(provider())
        atomic_write_json(os.path.join(checkpoint_path, "reference_leases.json"), lease_snapshot, compact=True)
    elif os.path.lexists(os.path.join(checkpoint_path, "reference_leases.json")):
        raise ValueError("Cannot publish a checkpoint containing an unbound reference lease snapshot")
    _save_resume_sidecar(
        checkpoint_path,
        consumed_snapshot,
        step=step,
        input_path=input_path,
        producer_shuffle=producer_shuffle,
        training_plan=getattr(trainer, "_standalone_training_plan", None),
        storage_format=getattr(trainer, "_standalone_resume_format", "tensor"),
        expected_input_identity=getattr(trainer, "_standalone_input_identity", None),
        **({"lease_snapshot_sha256": lease_snapshot["sha256"]} if lease_snapshot is not None else {}),
    )
    retention_keep = getattr(trainer, "_standalone_retention_keep_last", 0)
    if retention_keep:
        from verl_speco.trainer.checkpoint import write_standalone_integrity_manifest

        write_standalone_integrity_manifest(
            checkpoint_path, run_id=trainer._standalone_retention_run_id,
            require_resume=input_path is not None and consumed_snapshot is not None,
        )
    _set_standalone_checkpoint_complete(checkpoint_path, complete=True)
    if retention_keep:
        from verl_speco.trainer.checkpoint import retire_standalone_checkpoints

        try:
            lease = getattr(trainer, "_standalone_checkpoint_directory_lease", None)
            if getattr(trainer, "_standalone_retention_delete", False):
                if lease is None or lease.directory.resolve() != Path(checkpoint_path).resolve().parent:
                    raise RuntimeError("Checkpoint retention requires the owned checkpoint directory")
                lease.verify()
            candidates = retire_standalone_checkpoints(
                os.path.dirname(checkpoint_path), current_step=int(step), keep_last=retention_keep,
                protected_paths=getattr(trainer, "_standalone_retention_protected", ()),
                delete=getattr(trainer, "_standalone_retention_delete", False),
                ownership_check=lease.verify if lease is not None and getattr(trainer, "_standalone_retention_delete", False) else None,
            )
            logger.info("Standalone checkpoint retention delete=%s candidates=%s",
                        trainer._standalone_retention_delete, candidates)
        except Exception:
            # Retention must not invalidate a successfully committed checkpoint.
            logger.exception("Standalone checkpoint retention skipped after verification failure")


def _set_standalone_checkpoint_complete(
    checkpoint_path: str, *, complete: bool
) -> None:
    """Atomically publish or invalidate an existing managed checkpoint."""

    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot update standalone checkpoint metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Cannot update standalone checkpoint metadata {metadata_path}: "
            "expected object"
        )
    metadata["complete"] = bool(complete)
    DrafterBaseTrainer._atomic_json_dump(metadata, metadata_path)


def _save_resume_sidecar(
    checkpoint_path: str,
    consumed_snapshot: torch.Tensor | None,
    *,
    step: int | None,
    input_path: str | None,
    producer_shuffle: dict[str, Any] | None = None,
    training_plan: dict[str, Any] | None = None,
    storage_format: str = "tensor",
    expected_input_identity: dict[str, Any] | None = None,
    lease_snapshot_sha256: str | None = None,
) -> None:
    if consumed_snapshot is None or step is None or input_path is None:
        if expected_input_identity is not None or lease_snapshot_sha256 is not None:
            raise ValueError("Cannot publish a bound checkpoint without input path and progress")
        return
    save_standalone_resume(
        checkpoint_path,
        consumed_snapshot,
        optimizer_step=step,
        input_path=input_path,
        producer_shuffle=producer_shuffle,
        training_plan=training_plan,
        storage_format=storage_format,
        expected_input_identity=expected_input_identity,
        **({"lease_snapshot_sha256": lease_snapshot_sha256} if lease_snapshot_sha256 is not None else {}),
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


def _standalone_producer_shuffle(config: Any) -> dict[str, Any] | None:
    """Producer shuffle settings to persist beside standalone checkpoints.

    Consumed sequence numbers are permutation positions, so a resumed Producer
    must see the settings that produced them (see ``_assert_resume_shuffle_
    alignment`` in the producer).
    """
    speco_cfg = getattr(config, "speco", None)
    if speco_cfg is None and isinstance(config, dict):
        speco_cfg = config.get("speco")
    producer_cfg = getattr(speco_cfg, "standalone_tq_producer", None)
    if producer_cfg is None and isinstance(speco_cfg, dict):
        producer_cfg = speco_cfg.get("standalone_tq_producer")
    if producer_cfg is None:
        return None
    settings = {
        "shuffle": bool(producer_cfg.get("shuffle", False)),
        "shuffle_seed": int(producer_cfg.get("shuffle_seed", 42)),
        "data_recipe": producer_data_recipe(producer_cfg),
    }
    if producer_cfg.get("deterministic_publication", False):
        settings["deterministic_publication"] = True
    if producer_cfg.get("source_proportional_field") is not None:
        settings["source_proportional_field"] = producer_cfg.get("source_proportional_field")
    return settings


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

_STANDALONE_TRAINING_LAYER_KEYS = {
    "dflash": "dflash_target_layer_ids",
    "dflash2": "dflash2_target_layer_ids",
    "dspark": "dspark_target_layer_ids",
    "domino": "domino_target_layer_ids",
}


def _standalone_training_layer_ids(
    trainer: DrafterBaseTrainer, backend_type: str
) -> list[int] | None:
    drafter_cfg = getattr(
        getattr(getattr(trainer, "config", None), "rollout", None), "drafter", None
    )
    training_cfg = getattr(drafter_cfg, "training", None)
    key = _STANDALONE_TRAINING_LAYER_KEYS.get(backend_type)
    if training_cfg is None or key is None:
        return None
    value = (
        training_cfg.get(key, None)
        if hasattr(training_cfg, "get")
        else getattr(training_cfg, key, None)
    )
    if value is None:
        return None
    layer_ids = [int(layer_id) for layer_id in value]
    if not layer_ids or any(layer_id < 0 for layer_id in layer_ids):
        raise ValueError(
            f"Invalid standalone {key}={layer_ids}: expected non-negative "
            "decoder-layer IDs"
        )
    return layer_ids


def _source_vllm_aux_layer_ids(
    source_config: dict[str, Any] | None,
) -> list[int] | None:
    if source_config is None:
        return None
    for key in (
        "aux_hidden_state_layer_ids",
        "eagle_aux_hidden_state_layer_ids",
    ):
        value = source_config.get(key)
        if value is not None:
            return [int(layer_id) for layer_id in value]
    return None


def _source_target_layer_ids(
    source_config: dict[str, Any], backend_type: str
) -> list[int] | None:
    """Read the layer IDs a source drafter config declares as decoder indices."""

    candidates = [source_config]
    for child_key in ("dflash_config", f"{backend_type}_config"):
        child = source_config.get(child_key)
        if isinstance(child, dict):
            candidates.append(child)
    for candidate in candidates:
        value = candidate.get("target_layer_ids")
        if value is not None:
            return [int(layer_id) for layer_id in value]
    return None


def _assert_standalone_layer_migration(
    trainer: DrafterBaseTrainer, backend_type: str | None
) -> None:
    """Reject a resumed checkpoint whose layer IDs use another convention.

    The checkpoint export path performs the same comparison while rewriting the
    runtime config, but that runs on the async checkpoint writer thread where a
    raise cannot reach the training loop.  Running the check up front keeps the
    failure in the main thread, before the run spends any GPU time.
    """

    if backend_type is None:
        return
    launcher_layer_ids = _standalone_training_layer_ids(trainer, backend_type)
    if launcher_layer_ids is None:
        return
    source_config = _load_source_drafter_config(trainer)
    if source_config is None:
        return
    _, expected_vllm_ids = normalize_standalone_layer_ids(
        backend_type, launcher_layer_ids, None
    )
    source_vllm_ids = _source_vllm_aux_layer_ids(source_config)
    if source_vllm_ids is not None:
        if source_vllm_ids != list(expected_vllm_ids):
            raise ValueError(
                "Standalone layer-ID migration mismatch: source checkpoint "
                f"vLLM IDs are {source_vllm_ids}, but launcher decoder IDs "
                f"{launcher_layer_ids} imply {list(expected_vllm_ids)}"
            )
        return
    source_target_ids = _source_target_layer_ids(source_config, backend_type)
    if source_target_ids is not None and source_target_ids != launcher_layer_ids:
        raise ValueError(
            "Cannot safely migrate standalone checkpoint target_layer_ids "
            f"{source_target_ids}: source config has no explicit vLLM auxiliary "
            f"IDs and launcher expects {launcher_layer_ids}"
        )


def _raise_standalone_export_error(trainer: DrafterBaseTrainer) -> None:
    """Re-raise a checkpoint export failure captured on the writer thread."""

    error = getattr(trainer, "_standalone_export_error", None)
    if error is not None:
        raise RuntimeError(
            "Standalone checkpoint export failed or timed out"
        ) from error


def _sync_standalone_export_error(
    trainer: DrafterBaseTrainer,
    device: torch.device,
    *,
    raise_on_error: bool = True,
) -> bool:
    """Make a checkpoint export failure visible to every distributed rank."""

    try:
        _wait_standalone_finalization(trainer)
    except TimeoutError:
        pass  # All ranks must participate in the failure agreement below.
    error = getattr(trainer, "_standalone_export_error", None)
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        failed = error is not None
    else:
        failed_tensor = torch.tensor(
            [1 if error is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(failed_tensor, op=dist.ReduceOp.MAX)
        failed = bool(failed_tensor.item())

    if failed and raise_on_error:
        if error is not None:
            _raise_standalone_export_error(trainer)
        raise RuntimeError("Standalone checkpoint export failed on another rank")
    return failed


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
        raise FileNotFoundError(
            f"Cannot rewrite standalone runtime config: missing {config_path}"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot rewrite standalone runtime config {config_path}: {exc}"
        ) from exc
    if not isinstance(training_config, dict):
        raise TypeError(
            f"Cannot rewrite standalone runtime config {config_path}: expected object"
        )

    variant_child_key, variant_alias_keys = _VARIANT_RUNTIME_ALIASES.get(
        backend_type, (None, ())
    )
    source_runtime_config = _load_source_drafter_config(trainer)
    training_dflash_config = training_config.get("dflash_config")
    training_variant_config = (
        training_config.get(variant_child_key) if variant_child_key else None
    )
    training_target_layer_ids = (
        training_config.get("target_layer_ids")
        or (
            training_dflash_config.get("target_layer_ids")
            if isinstance(training_dflash_config, dict)
            else None
        )
        or (
            training_variant_config.get("target_layer_ids")
            if isinstance(training_variant_config, dict)
            else None
        )
    )
    launcher_target_layer_ids = _standalone_training_layer_ids(trainer, backend_type)
    if launcher_target_layer_ids is not None:
        expected_vllm_ids = [layer_id + 1 for layer_id in launcher_target_layer_ids]
        source_vllm_ids = _source_vllm_aux_layer_ids(source_runtime_config)
        saved_layer_ids = (
            [int(layer_id) for layer_id in training_target_layer_ids]
            if training_target_layer_ids is not None
            else None
        )
        if source_vllm_ids is not None:
            if source_vllm_ids != expected_vllm_ids:
                raise ValueError(
                    "Standalone layer-ID migration mismatch: source checkpoint "
                    f"vLLM IDs are {source_vllm_ids}, but launcher decoder IDs "
                    f"{launcher_target_layer_ids} imply {expected_vllm_ids}"
                )
        elif saved_layer_ids not in (None, launcher_target_layer_ids):
            raise ValueError(
                "Cannot safely migrate standalone checkpoint target_layer_ids "
                f"{saved_layer_ids}: source config has no explicit vLLM auxiliary "
                f"IDs and launcher expects {launcher_target_layer_ids}"
            )
        # save_pretrained may have serialized a released speculators config whose
        # target_layer_ids were copied verbatim from output-index aux IDs. The
        # standalone launcher's decoder IDs are authoritative for this run.
        training_target_layer_ids = launcher_target_layer_ids
        training_config["target_layer_ids"] = list(launcher_target_layer_ids)
        if isinstance(training_dflash_config, dict):
            training_dflash_config["target_layer_ids"] = list(launcher_target_layer_ids)
        if isinstance(training_variant_config, dict):
            training_variant_config["target_layer_ids"] = list(
                launcher_target_layer_ids
            )
    training_decoder_layer_ids: list[int] | None = None
    if training_target_layer_ids is not None:
        try:
            training_decoder_layer_ids = [
                int(layer_id) for layer_id in training_target_layer_ids
            ]
        except (TypeError, ValueError):
            logger.warning(
                "Invalid target_layer_ids in standalone exported config: %r",
                training_target_layer_ids,
            )
        else:
            if any(layer_id < 0 for layer_id in training_decoder_layer_ids):
                raise ValueError(
                    "Cannot export standalone DFlash runtime config with "
                    f"target_layer_ids={training_decoder_layer_ids}: each decoder "
                    "layer id must be non-negative."
                )

    training_config_path = os.path.join(checkpoint_path, "speco_training_config.json")
    DrafterBaseTrainer._atomic_json_dump(training_config, training_config_path)

    runtime_config = source_runtime_config
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

    if training_decoder_layer_ids is not None:
        # Training and runtime configs use zero-based decoder-layer indices.
        # vLLM's EAGLE auxiliary alias addresses layer outputs, so it is the
        # same capture plan shifted by one.
        runtime_target_layer_ids = list(training_decoder_layer_ids)
        runtime_config["eagle_aux_hidden_state_layer_ids"] = [
            layer_id + 1 for layer_id in training_decoder_layer_ids
        ]
        runtime_config["target_layer_ids"] = runtime_target_layer_ids
        dflash_config["target_layer_ids"] = runtime_target_layer_ids
        if variant_child_key:
            variant_config["target_layer_ids"] = runtime_target_layer_ids

    DrafterBaseTrainer._atomic_json_dump(runtime_config, config_path)


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


def _apply_standalone_fsdp_shard_default(draft_config) -> None:
    """Standalone drafters replicate by default; fsdp_shard_size>1 opts into sharding."""

    training_cfg = draft_config.rollout.drafter.training
    with open_dict(training_cfg):
        if training_cfg.get("fsdp_shard_size", None) is None:
            training_cfg.fsdp_shard_size = 1


def _build_training_device_mesh(draft_config, world_size: int) -> DeviceMesh | None:
    if world_size <= 1 or not dist.is_initialized():
        return None
    strategy = resolve_drafter_strategy(draft_config)
    if strategy != "fsdp2":
        return None
    return DeviceMesh(
        device_type=get_device_name(),
        mesh=torch.arange(world_size, dtype=torch.int64).reshape(1, world_size),
        mesh_dim_names=("dp", "sp"),
    )


def _block_metric_prefix(trainer: DrafterBaseTrainer) -> str | None:
    model_type = str(getattr(getattr(trainer, "backend", None), "model_type", "") or "")
    if model_type in {"dflash", "dspark", "eagle3", "dflash2"}:
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
        anchor_offset = 1 if prefix in {"dflash", "dflash2"} else 0
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
        if f"{prefix}/mean_acceptance_length" in raw_metrics:
            metrics["train/mean_accept_length"] = float(
                raw_metrics[f"{prefix}/mean_acceptance_length"]
            )
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
        ("train/mean_accept_length", "mean_accept_length"),
        ("train/lr", "lr"),
        ("perf/step_time", "step_time"),
        ("perf/loader_plan_wait_time", "loader_plan_wait_time"),
        ("perf/loader_tq_get_time", "loader_tq_get_time"),
        ("perf/loader_batch_prepare_time", "loader_batch_prepare_time"),
        ("perf/batch_ready_wait_time", "batch_ready_wait_time"),
        *[
            (f"perf/{key}", key)
            for key in (
                "local_batch_wait_time",
                "rank_ready_sync_time",
                "metadata_event_age_time",
                "scheduler_ready_wait_time",
                "command_delivery_time",
                "loader_worker_result_wait_time",
                "loader_ipc_collect_time",
                "loader_pin_queue_time",
                "loader_pin_time",
                "loader_ready_queue_time",
            )
        ],
        ("perf/batch_prepare_time", "batch_prepare_time"),
        ("perf/train_time", "train_time"),
        ("perf/batch_finalize_time", "batch_finalize_time"),
        ("perf/commit_gate_wait_time", "commit_gate_wait_time"),
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
        elif (
            key.endswith("_time_total")
            or key.startswith("perf/")
            or key
            in {
                "perf/step_time",
                "replay/target_forward_time_total",
            }
        ):
            fields.append(f"{label}={value:.3f}s")
        else:
            fields.append(f"{label}={value:.4f}")
    logger.info("[standalone drafter metrics] %s", " ".join(fields))


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


def _max_across_ranks_float(value: float, device: torch.device) -> float:
    """Return the slowest-rank duration for a distributed standalone stage."""

    maximum = torch.tensor(float(value), dtype=torch.float32, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def _reference_device_budget(transfer_config, feature_store_type):
    value = transfer_config.get("consumer_device_batch_bytes", 0)
    if type(value) is not int or value < 0:
        raise ValueError("consumer_device_batch_bytes must be a non-negative integer")
    if value and feature_store_type != "ray_reference":
        raise ValueError("consumer_device_batch_bytes requires ray_reference input")
    return value


def _reference_device_prefetch_budget(transfer_config, feature_store_type, device_type, batch_limit):
    value = transfer_config.get("consumer_device_prefetch_bytes", 0)
    if type(value) is not int or value < 0:
        raise ValueError("consumer_device_prefetch_bytes must be a non-negative integer")
    if value and (feature_store_type != "ray_reference" or device_type not in {"cuda", "npu"}
                  or not batch_limit or value < 2 * batch_limit):
        raise ValueError("Device prefetch requires ray_reference, CUDA/NPU and two device batch budgets")
    if value:
        if not prefetch_depth(transfer_config):
            raise ValueError("Device prefetch requires a positive Consumer prefetch depth")
        host = transfer_config.get("consumer_prefetch_host_bytes", 0)
        per_batch = transfer_config.get("consumer_prefetch_batch_bytes", 0)
        if (type(host) is not int or host < 0 or type(per_batch) is not int or per_batch < 0
                or bool(host) != bool(per_batch) or (host and host < 2 * per_batch)):
            raise ValueError("Device prefetch requires host budget for two maximum batches")
    return value


async def _reference_training_step_across_ranks(trainer, batch, step, *, device, max_device_bytes=0) -> bool:
    """Do not release or credit a reference batch unless all live ranks succeed."""
    if max_device_bytes:
        error = None
        try:
            if _reference_device_transfer_bytes(batch) > max_device_bytes:
                raise ValueError("Reference batch exceeds consumer_device_batch_bytes")
        except Exception as exc:
            error = exc
        if not _all_ranks_true(error is None, device):
            raise RuntimeError("Reference device byte budget exceeded or invalid on at least one rank; source references retained") from error
    error = None
    staged = batch
    try:
        if batch.get("hidden_source") == "standalone_vllm_reference":
            staged = trainer._stage_standalone_cpu_batch(batch)
            staged["_speco_reference_staged"] = True
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Reference device staging failed on at least one rank; source references retained") from error
    error = None
    ok = False
    try:
        ok = bool(await trainer.training_step_from_batch(staged, step, raise_on_error=True))
    except Exception as exc:  # ranks that returned must agree before cleanup
        error = exc
    if not _all_ranks_true(ok and error is None, device):
        raise RuntimeError("Reference training failed on at least one rank; source references retained") from error
    return True


def _validate_reference_part_loss_recipe(trainer):
    """Refuse known non-additive model recipes until their reduction is implemented."""
    backend = getattr(trainer, "backend", None)
    if getattr(backend, "model_type", None) != "dspark":
        if getattr(backend, "reference_parts_additive", False) is True:
            return
        raise ValueError("Reference part loss recipe is not validated for this backend")
    model = trainer.model
    seen = set()
    while not hasattr(model, "l1_loss_alpha") and id(model) not in seen:
        seen.add(id(model))
        inner = getattr(model, "module", None)
        if inner is None:
            inner = getattr(model, "_orig_mod", None)
        if inner is None:
            raise ValueError("Cannot inspect the reference part loss recipe")
        model = inner
    if (getattr(model, "loss_mode", None) not in {"full_vocab", "restricted_ce", "sampled_ce"}
            or getattr(model, "per_position_loss_weight", None) not in {"fixed_exp_decay", "dpace"}
            or getattr(model, "ce_target_source", None) not in {"labels", "target_argmax"}
            or (getattr(model, "loss_mode", None) in {"restricted_ce", "sampled_ce"}
                and getattr(model, "ce_target_source", None) != "labels")
            or (getattr(model, "ce_target_source", None) == "target_argmax"
                and getattr(model, "l1_chunk_size", None) != 0)):
        raise ValueError("Reference parts require full-vocabulary fixed-weight CE or D-PACE CE, or restricted CE with labels")
    for name in ("l1_loss_alpha", "lk_loss_alpha", "confidence_head_alpha"):
        value = getattr(model, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Reference part auxiliary weights must be finite and non-negative")
    if model.lk_loss_alpha > 0 and getattr(model, "lk_loss_type", None) not in {"alpha", "adaptive_hybrid"}:
        raise ValueError("Reference parts do not support this LK loss type")
    adaptive = model.lk_loss_alpha > 0 and getattr(model, "lk_loss_type", None) == "adaptive_hybrid"
    confidence = model.confidence_head_alpha > 0
    if model.per_position_loss_weight == "dpace":
        alpha = getattr(model, "dpace_alpha", None)
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha)
                or not 0 <= alpha <= 1):
            raise ValueError("Reference D-PACE requires alpha in [0, 1]")
    if confidence and (getattr(model, "confidence_loss_enabled", False) is not True
                       or getattr(getattr(model, "draft_model", None), "confidence_head", None) is None):
        raise ValueError("Reference confidence loss requires an enabled confidence head")
    if adaptive:
        eta = getattr(model, "lk_hybrid_eta", None)
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0:
            raise ValueError("Reference adaptive LK coefficient must be finite and non-negative")
    if adaptive or confidence or model.loss_mode == "sampled_ce" or model.per_position_loss_weight == "dpace":
        if any((isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0)
               or (isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats)
               or (isinstance(module, torch.nn.RNNBase) and module.dropout > 0)
               or (isinstance(getattr(module, "attention_dropout", None), (int, float))
                   and module.attention_dropout > 0) for module in model.modules()):
            raise ValueError("Reference loss prepass requires deterministic model dropout and state")
    gamma, block_size = getattr(model, "loss_decay_gamma", None), getattr(model, "block_size", None)
    if (type(block_size) is not int or block_size < 1
            or (gamma is not None and (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
                or not math.isfinite(gamma)))
            or (model.per_position_loss_weight != "dpace" and gamma is not None and gamma > 0
                and (block_size - 1) / gamma >= -math.log(1e-6))):
        raise ValueError("Reference part position weights can fall below the supported normalization floor")
    return model


async def _execute_owned_reference_batch(trainer, store, batch, step, *, rank, device, events,
                                         transport_config, per_rank=False, producer_handle=None,
                                         max_window=None, max_host_bytes=0, max_device_bytes=0, token_counts=None,
                                         packing=False, packing_block_size=None, max_packed_len=0,
                                         node_ranks_per_host=0, source_payload_safe=False):
    """Read all owned parts before acknowledging materialization to the producer."""
    contract, clip_value, read_timeout, error = None, None, None, None
    try:
        contract = store.reference_contract(batch.file_entries)
        clip_value = trainer._get_hidden_state_clip_value()
        read_timeout = float(transport_config.get("io_timeout_seconds", 30))
        if (type(max_host_bytes) is not int or max_host_bytes < 0
                or type(max_device_bytes) is not int or max_device_bytes < 0):
            raise ValueError("Owned window byte limits must be non-negative integers")
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Owned reference contract failed on at least one rank; source references retained") from error
    ok = await _owned_reference_training_step(
        trainer, [entry.tag for entry in batch.file_entries], contract, step, device=device,
        max_window=max_window, clip_value=clip_value, read_timeout=read_timeout,
        max_host_bytes=max_host_bytes, max_device_bytes=max_device_bytes, padding_rank=batch.is_padding_rank,
        token_counts=token_counts, packing=packing, packing_block_size=packing_block_size, max_packed_len=max_packed_len,
        hidden_states_store=getattr(store, "hidden_states_store", None),
        node_context=({"rank": rank, "ranks_per_host": node_ranks_per_host,
                       "entries": batch.node_file_entries, "run_id": store.run_id,
                       "plan_id": batch.plan_id, "receipt_token": batch.receipt_token}
                      if node_ranks_per_host else None),
    )
    _ack_reference_materialization_across_ranks(batch, rank=rank, device=device, events=events,
                                                config=transport_config, per_rank=per_rank,
                                                producer_handle=producer_handle,
                                                **({"source_payload_safe": True} if source_payload_safe else {}))
    return ok


async def _owned_reference_training_step(trainer, tags, feature_contract, step, *, device, windows=None,
                                         max_window=None, clip_value=1.0e4, chunk_rows=512,
                                         read_timeout=30.0, max_host_bytes=0, max_device_bytes=0,
                                         padding_rank=False, token_counts=None, packing=False,
                                         packing_block_size=None, max_packed_len=0, node_context=None,
                                         hidden_states_store=None):
    """Sample a local file assignment and execute globally padded lazy parts.

    This primitive does not acknowledge or release references. Its caller must
    retain them through the last read and coordinate the scheduler lifecycle.
    """
    from verl_speco.trainer.standalone_batch_prepare import (
        prepare_empty_owned_window, prepare_owned_anchor_window, sample_owned_anchor_windows,
    )
    error, model, plans, vocabulary = None, None, None, None
    try:
        model = _validate_reference_part_loss_recipe(trainer)
        if model is None:
            raise ValueError("Owned reference sampling requires the block training backend")
        if windows is None:
            from verl_speco.backends.dflash_trainer_backend import _resolve_sliding_windows
            windows = _resolve_sliding_windows(model.draft_model.config)
        if model.loss_mode in {"restricted_ce", "sampled_ce"}:
            vocabulary = {"size": int(trainer.backend.target_lm_head.fc.weight.shape[0])}
        _, _, plans = sample_owned_anchor_windows(
            tags, feature_contract, model, device=device, windows=windows, max_window=max_window,
            clip_value=clip_value, chunk_rows=chunk_rows, read_timeout=read_timeout, padding_rank=padding_rank,
            token_counts=token_counts, packing=packing, packing_block_size=packing_block_size, max_packed_len=max_packed_len,
            vocabulary=vocabulary, hidden_states_store=hidden_states_store,
        )
        if len(plans) > 4096:
            raise ValueError("Owned reference batch exceeds the part limit")
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Owned reference planning failed on at least one rank; source references retained") from error
    sampled_probe = vocabulary is not None and model.loss_mode == "sampled_ce" and model.sampled_ce_negatives > 0
    count = torch.tensor([len(plans), int(sampled_probe), -int(sampled_probe)], dtype=torch.int32, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    if int(count[1]) != -int(count[2]):
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Sampled vocabulary probe differs across ranks")
    count = max(int(count[0]), 1)
    parts = [partial(prepare_owned_anchor_window, tags[plan["sample_index"]], feature_contract,
                     start=plan["start"], end=plan["end"], anchor=plan["anchor"],
                     block_size=model.block_size, windows=windows, source_window=plan["source_window"],
                     read_timeout=read_timeout, max_output_bytes=max_host_bytes,
                     hidden_states_store=hidden_states_store) for plan in plans]
    parts.extend(partial(prepare_empty_owned_window, tags[0], feature_contract,
                         read_timeout=read_timeout, max_output_bytes=max_host_bytes, hidden_states_store=hidden_states_store)
                 for _ in range(count - len(parts)))
    if node_context is not None:
        try:
            parts = _node_owned_window_parts(
                plans, count, node_context, feature_contract, windows=windows, block_size=model.block_size,
                read_timeout=read_timeout, max_host_bytes=max_host_bytes, device=device,
                hidden_states_store=hidden_states_store)
        except Exception:
            if getattr(trainer, "optimizer", None) is not None:
                trainer.optimizer.zero_grad(set_to_none=True)
            raise
    if vocabulary is not None:
        def with_vocabulary(part):
            return {**part(), "preselected_restricted_vocab": vocabulary["ids"]}
        parts = [partial(with_vocabulary, part) for part in parts]
    probe_counts, probe_fingerprints = None, None
    if sampled_probe:
        statistics = {"values": torch.zeros(7, dtype=torch.float32, device=device), "fingerprints": [],
                      "input_fingerprints": []}
        probe_counts = {"window_statistics_input_positions": 0} if token_counts is not None else None
        await _reference_training_parts_across_ranks(trainer, parts, step, device=device, max_device_bytes=max_device_bytes,
                                                     work_counts=probe_counts, _loss_statistics=statistics)
        probe_fingerprints = statistics["input_fingerprints"]
        if statistics["values"][6] > 0:
            negatives = torch.randint(0, vocabulary["size"], (model.sampled_ce_negatives,),
                                      dtype=torch.long, device=device).cpu()
            vocabulary["ids"] = torch.unique(torch.cat((vocabulary["ids"], negatives)), sorted=True)
        del statistics
    ok = await _reference_training_parts_across_ranks(trainer, parts, step, device=device,
                                                       max_device_bytes=max_device_bytes, work_counts=token_counts,
                                                       _input_fingerprints=probe_fingerprints)
    if probe_counts is not None:
        token_counts["window_statistics_input_positions"] += probe_counts["window_statistics_input_positions"]
    return ok


def _node_owned_window_parts(plans, count, context, feature_contract, *, windows, block_size,
                             read_timeout, max_host_bytes, device, hidden_states_store=None):
    """Route each aligned window through its node owner using bounded local frames."""
    from verl_speco.trainer.reference_shared_frame import serve_node_batches, receive_node_batch
    from verl_speco.trainer.standalone_batch_prepare import prepare_empty_owned_window, prepare_owned_anchor_window

    error, encoded = None, None
    try:
        rank, local_size = context["rank"], context["ranks_per_host"]
        world = dist.get_world_size() if dist.is_initialized() else 1
        if (type(rank) is not int or type(local_size) is not int or local_size < 2
                or not 0 <= rank < world or world % local_size or not dist.is_initialized()
                or rank != dist.get_rank() or type(max_host_bytes) is not int or max_host_bytes <= 0
                or type(context["plan_id"]) is not int or context["plan_id"] < 0
                or not isinstance(context["run_id"], str) or not 1 <= len(context["run_id"]) <= 256
                or not isinstance(context["receipt_token"], str) or not 1 <= len(context["receipt_token"]) <= 256
                or isinstance(read_timeout, bool) or not isinstance(read_timeout, (int, float))
                or not math.isfinite(read_timeout) or read_timeout <= 0):
            raise ValueError("Invalid node-owned window context")
        owner = rank // local_size * local_size
        entries = context.get("entries")
        if rank == owner and (not isinstance(entries, dict)
                or set(entries) != set(range(owner, owner + local_size))
                or any(not isinstance(value, list) or not value for value in entries.values())):
            raise ValueError("Node owner lacks its rank assignments")
        common = {"run_id": context["run_id"], "plan_id": context["plan_id"],
                  "receipt_token": context["receipt_token"], "windows": windows,
                  "block_size": block_size, "frame_bytes": max_host_bytes, "read_timeout": read_timeout,
                  "ranks_per_host": local_size}
        encoded = json.dumps({"common": common, "plans": plans}, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError("Node-owned window plan exceeds its metadata limit")
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Node-owned window plan is invalid on at least one rank") from error
    metadata_bytes = torch.tensor([len(encoded.encode())], dtype=torch.int64, device=device)
    dist.all_reduce(metadata_bytes, op=dist.ReduceOp.SUM)
    if int(metadata_bytes.item()) > 16 * 1024 * 1024:
        raise RuntimeError("Node-owned window plans exceed the global metadata limit")
    all_plans = [None] * world
    dist.all_gather_object(all_plans, encoded)
    error, groups = None, None
    try:
        records = [json.loads(item) for item in all_plans]
        if any(not isinstance(item, dict) or item.get("common") != records[0].get("common") for item in records):
            raise ValueError("Node-owned window identity or configuration differs across ranks")
        parsed = [item.get("plans") for item in records]
        if any(not isinstance(items, list) or len(items) > count for items in parsed):
            raise ValueError("Node-owned window part count differs")
        if rank == owner:
            groups = {}
            for peer in range(owner, owner + local_size):
                tags = [entry.tag for entry in entries[peer]]
                peer_plans = parsed[peer]
                for plan in peer_plans:
                    if (not isinstance(plan, dict) or type(plan.get("sample_index")) is not int
                            or not 0 <= plan["sample_index"] < len(tags)):
                        raise ValueError("Node-owned window references an unassigned sample")
                groups[peer] = (tags, peer_plans)
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Node-owned window assignments differ across ranks") from error

    counter = [0]
    timeout = max(30.0, float(read_timeout) * (local_size + 1))

    def materialize(index):
        serial = counter[0]
        counter[0] += 1
        receipt = hashlib.sha256(f"{context['receipt_token']}:{serial}".encode()).hexdigest()
        binding = {"run_id": context["run_id"], "generation": 0, "batch_id": context["plan_id"],
                   "rank": rank, "receipt_token": receipt}
        if rank != owner:
            return receive_node_batch(binding=binding, owner_rank=owner, max_bytes=max_host_bytes, timeout=timeout)[0]

        requests = {peer: [(tags, peer_plans[index] if index < len(peer_plans) else None)]
                    for peer, (tags, peer_plans) in groups.items()}
        def prepare(request):
            tags, plan = request[0]
            if plan is None:
                batch = prepare_empty_owned_window(tags[0], feature_contract, read_timeout=read_timeout,
                                                   max_output_bytes=max_host_bytes, hidden_states_store=hidden_states_store)
            else:
                batch = prepare_owned_anchor_window(
                    tags[plan["sample_index"]], feature_contract, start=plan["start"], end=plan["end"],
                    anchor=plan["anchor"], block_size=block_size, windows=windows,
                    source_window=plan["source_window"], read_timeout=read_timeout, max_output_bytes=max_host_bytes,
                    hidden_states_store=hidden_states_store)
            return batch, 0.0
        return serve_node_batches(requests, binding=binding, prepare=prepare,
                                  max_bytes=max_host_bytes, timeout=timeout)[0]

    return [partial(materialize, index) for index in range(count)]


def _reference_part_fingerprint(batch):
    """Bind a two-pass loss calculation to the exact staged input bytes."""
    digest = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name]
        if torch.is_tensor(value):
            if value.layout != torch.strided or not value.is_contiguous():
                raise ValueError("Adaptive reference inputs must be contiguous")
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
            raw = value.detach().reshape(-1).view(torch.uint8)
            for offset in range(0, raw.numel(), 8 * 1024 * 1024):
                chunk = raw[offset:offset + 8 * 1024 * 1024].to(device="cpu", copy=True)
                digest.update(memoryview(chunk.numpy()))
        elif name in {"hidden_source", "last_hidden_state_norm", "_speco_pad_size"}:
            digest.update(json.dumps([name, value], separators=(",", ":"), allow_nan=False).encode())
    return digest.digest()


async def _reference_training_parts_across_ranks(trainer, parts, step, *, device, max_device_bytes=0,
                                                 work_counts=None, _loss_statistics=None, _input_fingerprints=None) -> bool:
    """Train equal-length lazy part plans with one globally normalized optimizer step.

    Each plan entry is a batch or a zero-argument materializer. Callers must
    pad empty ranks with a graph-preserving zero-loss batch and establish that
    their loss recipe is additive across parts before using this primitive.
    """
    error, part_model = None, None
    try:
        part_model = _validate_reference_part_loss_recipe(trainer)
    except Exception as exc:
        error = exc
    if not _all_ranks_true(error is None, device):
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Reference part loss recipe is unsupported on at least one rank") from error
    count = len(parts) if isinstance(parts, (list, tuple)) else 0
    valid = 0 < count <= 4096 and type(max_device_bytes) is int and max_device_bytes >= 0
    adaptive = (part_model is not None and getattr(part_model, "lk_loss_alpha", 0) > 0
                and getattr(part_model, "lk_loss_type", None) == "adaptive_hybrid")
    confidence = part_model is not None and getattr(part_model, "confidence_head_alpha", 0) > 0
    dpace = part_model is not None and getattr(part_model, "per_position_loss_weight", None) == "dpace"
    counts = torch.tensor([count if valid else 0, -count if valid else 0,
                           int(adaptive), -int(adaptive), int(confidence), -int(confidence),
                           int(dpace), -int(dpace)],
                          dtype=torch.int32, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    if not valid or int(counts[0]) == 0 or int(counts[0]) != -int(counts[1]):
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Reference part count or device budget differs across ranks")
    if any(int(counts[i]) != -int(counts[i + 1]) for i in (2, 4, 6)):
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Reference loss statistics prepass differs across ranks")
    lk_weight = confidence_scale = l1_scale = lk_scale = ce_scale = None
    fingerprints = None
    vocabulary_fingerprint = None
    if work_counts is not None and _loss_statistics is None:
        work_counts.update(window_train_input_positions=0, window_statistics_input_positions=0)
    if (adaptive or confidence or dpace) and _loss_statistics is None:
        statistics = {"values": torch.zeros(7, dtype=torch.float32, device=device), "fingerprints": []}
        await _reference_training_parts_across_ranks(trainer, parts, step, device=device,
                                                     max_device_bytes=max_device_bytes, work_counts=work_counts,
                                                     _loss_statistics=statistics, _input_fingerprints=_input_fingerprints)
        values, fingerprints = statistics["values"], statistics["fingerprints"]
        if adaptive:
            lk_weight = torch.exp(-part_model.lk_hybrid_eta * (values[0] / values[1].clamp_min(1.0))).detach()
        if confidence:
            confidence_scale = (values[2] / values[3].clamp_min(1e-6)).detach()
        if dpace:
            ce_scale = (values[2] / values[2].clamp_min(1e-6)).detach()
        if dpace and part_model.l1_loss_alpha > 0:
            l1_scale = (values[2] / values[4].clamp_min(1e-6)).detach()
        if dpace and part_model.lk_loss_alpha > 0:
            lk_scale = (values[2] / values[5].clamp_min(1e-6)).detach()
        del statistics, values
    try:
        for index, item in enumerate(parts):
            error, staged = None, None
            try:
                batch = item() if callable(item) else item
                if not isinstance(batch, dict):
                    raise ValueError("Reference part must materialize a batch mapping")
                if max_device_bytes and (_reference_device_transfer_bytes(batch)
                                         + (lk_weight.numel() * lk_weight.element_size() if lk_weight is not None else 0)
                                         + (confidence_scale.numel() * confidence_scale.element_size() if confidence_scale is not None else 0)
                                         + (l1_scale.numel() * l1_scale.element_size() if l1_scale is not None else 0)
                                         + (lk_scale.numel() * lk_scale.element_size() if lk_scale is not None else 0)
                                         + (ce_scale.numel() * ce_scale.element_size() if ce_scale is not None else 0)
                                         > max_device_bytes):
                    raise ValueError("Reference part exceeds consumer_device_batch_bytes")
                if batch.get("hidden_source") == "standalone_vllm_reference":
                    staged = dict(trainer._stage_standalone_cpu_batch(batch))
                    staged["_speco_reference_staged"] = True
                else:
                    staged = dict(batch)
                if part_model is not None:
                    ids, mask, hidden = staged["input_ids"], staged["loss_mask"], staged["hidden_states"]
                    if part_model.loss_mode in {"restricted_ce", "sampled_ce"}:
                        vocab = staged.get("preselected_restricted_vocab")
                        if not isinstance(vocab, torch.Tensor) or staged.get("next_token_ids") is not None:
                            raise ValueError("Reference restricted CE requires the original input vocabulary and labels")
                        current_vocab = _reference_part_fingerprint({"preselected_restricted_vocab": vocab})
                        if vocabulary_fingerprint is not None and current_vocab != vocabulary_fingerprint:
                            raise ValueError("Reference restricted vocabulary changed between parts")
                        vocabulary_fingerprint = current_vocab
                    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.dtype != torch.long
                            or not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or mask.device != ids.device
                            or not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                            or hidden.shape[:2] != ids.shape or hidden.device != ids.device):
                        raise ValueError("Reference part input, mask and hidden tensors are incompatible")
                    draft = part_model.draft_model
                    if hidden.size(-1) != draft.num_context_layers * draft.target_hidden_size:
                        raise ValueError("Reference part hidden feature width is incompatible")
                    target = staged.get("target_last_hidden_states")
                    if getattr(part_model, "ce_target_source", None) == "target_argmax" and target is None:
                        raise ValueError("Reference part target-argmax loss requires target hidden states")
                    # Raw targets are normalized and sanitized before the model
                    # forms CE and distribution masks. With fixed position
                    # weights these losses then share the same denominator.
                    if (getattr(part_model, "l1_loss_alpha", 0) > 0 or getattr(part_model, "lk_loss_alpha", 0) > 0
                            or getattr(part_model, "confidence_head_alpha", 0) > 0) and (
                            target is None or staged.get("last_hidden_state_norm") != "raw_target_residual"):
                        raise ValueError("Reference part distribution losses require normalized raw target supervision")
                    if target is not None and (not isinstance(target, torch.Tensor) or target.ndim != 3
                            or target.shape[:2] != ids.shape or target.size(-1) != draft.target_hidden_size
                            or target.device != ids.device):
                        raise ValueError("Reference part target hidden tensor is incompatible")
                    if staged.get("last_hidden_state_norm") == "raw_target_residual" and (
                            staged.get("hidden_source") != "standalone_vllm_reference"
                            or getattr(part_model, "verifier_norm", None) is None or target is None):
                        raise ValueError("Reference part raw target requires a verifier norm and target tensor")
                    labels = staged.get("label_ids", ids)
                    label_mask = staged.get("label_mask", mask)
                    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.ndim != 2
                            or labels.size(0) != ids.size(0) or labels.device != ids.device
                            or not isinstance(label_mask, torch.Tensor) or label_mask.shape != labels.shape
                            or label_mask.device != ids.device):
                        raise ValueError("Reference part label tensors are incompatible")
                    next_mask = staged.get("next_token_loss_mask")
                    part_model._validate_preselected_anchors(
                        staged.get("preselected_anchor_positions"), staged.get("preselected_anchor_keep_mask"),
                        seq_len=ids.size(1), label_mask=mask if next_mask is not None else label_mask,
                        document_ids=staged.get("document_ids"), next_token_loss_mask=next_mask,
                    )
                if (_input_fingerprints is not None
                        or (_loss_statistics is not None and "input_fingerprints" in _loss_statistics)):
                    fingerprint = _reference_part_fingerprint(
                        {key: value for key, value in staged.items() if key != "preselected_restricted_vocab"})
                    if _input_fingerprints is not None and fingerprint != _input_fingerprints[index]:
                        raise ValueError("Reference part changed after vocabulary sampling probe")
                    if _loss_statistics is not None and "input_fingerprints" in _loss_statistics:
                        _loss_statistics["input_fingerprints"].append(fingerprint)
                if _loss_statistics is not None:
                    _loss_statistics["fingerprints"].append(_reference_part_fingerprint(staged))
                elif fingerprints is not None and _reference_part_fingerprint(staged) != fingerprints[index]:
                    raise ValueError("Reference part changed after loss statistics")
            except Exception as exc:
                error = exc
            if not _all_ranks_true(error is None, device):
                raise RuntimeError("Reference part materialization or staging failed; source references retained") from error
            if _loss_statistics is not None:
                error = None
                try:
                    trainer.model.train()
                    inputs = {key: value for key, value in staged.items() if not key.startswith("_speco_")}
                    with torch.no_grad(), trainer._ulysses_group_context(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        diagnostic = trainer.backend.compute_loss(trainer.model, inputs, int(staged.get("_speco_pad_size", 0) or 0))["diagnostics"]
                    values = [diagnostic[name].detach().float() for name in (
                        "lk_acceptance_sum", "lk_diagnostic_token_count", "ce_weighted_token_count",
                        "confidence_weighted_token_count", "l1_weighted_token_count", "lk_weighted_token_count",
                        "active_target_count")]
                    if any(value.numel() != 1 or not bool(torch.isfinite(value).all()) or float(value) < 0 for value in values):
                        raise ValueError("Invalid reference loss statistics")
                    _loss_statistics["values"].add_(torch.stack([value.reshape(()) for value in values]))
                except Exception as exc:
                    error = exc
                if not _all_ranks_true(error is None, device):
                    raise RuntimeError("Reference loss statistics failed; source references retained") from error
                if work_counts is not None:
                    work_counts["window_statistics_input_positions"] += staged["input_ids"].numel()
                del staged, batch
                continue
            if lk_weight is not None:
                staged["preselected_lk_weight"] = lk_weight
            if confidence_scale is not None:
                staged["preselected_confidence_scale"] = confidence_scale
            if l1_scale is not None:
                staged["preselected_l1_scale"] = l1_scale
            if lk_scale is not None:
                staged["preselected_lk_scale"] = lk_scale
            if ce_scale is not None:
                staged["preselected_ce_scale"] = ce_scale
            staged["_speco_allow_zero_accumulation"] = True
            error, ok = None, False
            try:
                ok = bool(await trainer.training_step_from_batch(
                    staged, step, raise_on_error=True, accumulation_steps=count, accumulation_index=index))
            except Exception as exc:
                error = exc
            if not _all_ranks_true(ok and error is None, device):
                raise RuntimeError("Reference part training failed; source references retained") from error
            if work_counts is not None:
                work_counts["window_train_input_positions"] += staged["input_ids"].numel()
            del staged, batch
    except BaseException:
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        raise
    return True


def _detach_reference_batch(batch, config):
    limit = config.get("consumer_prefetch_batch_bytes", 0)
    if not isinstance(batch, TQLocalBatch) or batch.prepared_batch is None or batch.local_samples:
        raise ValueError("Early source release requires a prepared batch without source samples")
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
    size = 0
    for value in tensors(batch.prepared_batch):
        if value.device.type != "cpu":
            raise ValueError("Detached source materialization must precede device prefetch")
        size += 2 * max(value.untyped_storage().nbytes(), value.numel() * value.element_size())
    if type(limit) is not int or limit <= 0 or size > limit:
        raise ValueError("Detached materialization requires an explicit sufficient batch byte budget")
    def detach(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Detached source materialization must precede device prefetch")
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: detach(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(detach(item) for item in value)
        return value
    return replace(batch, prepared_batch=detach(batch.prepared_batch))


def _ack_reference_materialization_across_ranks(batch, *, rank, device, events, config, per_rank=False,
                                                producer_handle=None, source_payload_safe=False) -> None:
    """Confirm rank preparation before sending the aggregate materialization event."""
    def agree(error):
        failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("Reference materialization acknowledgement failed on at least one rank") from error

    error = None
    try:
        if (not isinstance(batch, TQLocalBatch) or type(batch.plan_id) is not int or batch.plan_id < 0
                or not isinstance(batch.receipt_token, str) or not batch.receipt_token):
            raise ValueError("Reference materialization requires a plan and receipt token")
        if type(source_payload_safe) is not bool:
            raise ValueError("Source payload safety receipt must be a boolean")
        if per_rank:
            incarnation = batch.producer_incarnation
            if (not isinstance(incarnation, str) or len(incarnation) != 32
                    or any(char not in "0123456789abcdef" for char in incarnation)
                    or producer_handle is None):
                raise ValueError("Reference rank acknowledgement requires Producer identity and handle")
            import ray
            retry_io(
                lambda: ray.get(producer_handle.acknowledge_reference_rank.remote(
                    incarnation, batch.plan_id, batch.receipt_token, rank),
                    timeout=float(config.get("io_timeout_seconds", 30))),
                name="Producer reference rank acknowledgement",
                attempts=int(config.get("io_retry_max_attempts", 3)),
            )
            event = {"kind": "reference_rank_materialized", "plan_id": batch.plan_id,
                     "receipt_token": batch.receipt_token, "rank": rank}
            if source_payload_safe:
                event["source_payload_safe"] = True
            retry_io(lambda: put_queue(events, event, timeout=float(config.get("io_timeout_seconds", 30))),
                     name="Reference rank materialization acknowledgement",
                     attempts=int(config.get("io_retry_max_attempts", 3)))
    except Exception as exc:
        error = exc
    if per_rank:
        agree(error)
        error = None
    if error is None and rank == 0:
        try:
            event = {"kind": "training_materialized", "plan_id": batch.plan_id,
                     "receipt_token": batch.receipt_token}
            if source_payload_safe:
                event["source_payload_safe"] = True
            retry_io(lambda: put_queue(events, event, timeout=float(config.get("io_timeout_seconds", 30))),
                     name="Reference materialization acknowledgement",
                     attempts=int(config.get("io_retry_max_attempts", 3)))
        except Exception as exc:
            error = exc
    agree(error)


def _submit_standalone_metrics(finalizer, events, event, config) -> None:
    """Keep every metrics snapshot on the bounded ACK executor, never on GPU."""

    def send() -> None:
        try:
            retry_io(
                lambda: put_queue(
                    events, event, timeout=config.get("io_timeout_seconds", 30)
                ),
                name="Standalone metrics event",
                attempts=int(config.get("io_retry_max_attempts", 3)),
            )
        except Exception:
            try:
                put_queue(
                    events,
                    {
                        "kind": "finalization_failed",
                        "step": event["step"],
                        "phase": "metrics",
                    },
                    timeout=config.get("io_timeout_seconds", 30),
                )
            except Exception:
                logger.exception(
                    "Failed to notify Driver of metrics publication failure"
                )
            raise

    finalizer.submit(send)


def _clear_tq_batch_across_ranks(
    loader: TQFeatureDataLoader,
    global_keys: list[str] | None,
    *,
    rank: int,
    device: torch.device,
    training_events: Any | None = None,
    completion_event: dict[str, Any] | None = None,
    finalizer: AsyncBatchFinalizer | None = None,
    synchronize: bool = True,
    cleanup_after_load: bool = False,
    synchronize_before_cleanup: bool = False,
) -> None:
    """Finalize on rank 0; ray_reference uses async ACK without a per-step collective."""

    def complete() -> None:
        if training_events is not None and completion_event and completion_event.get("receipt_token"):
            applied_event = {"kind": "training_applied", "plan_id": completion_event["plan_id"],
                             "receipt_token": completion_event["receipt_token"]}
            retry_io(
                lambda: put_queue(training_events, applied_event,
                                  timeout=getattr(loader, "io_timeout_seconds", 30)),
                name="Reference optimizer acknowledgement",
                attempts=getattr(loader, "io_retry_max_attempts", 3),
            )
        retry_io(
            lambda: (
                loader.clear_completed_batch(global_keys, cleanup_after_load=True)
                if cleanup_after_load
                else loader.clear_completed_batch(global_keys)
            ),
            name="Consumer batch cleanup",
            attempts=_TQ_CLEAR_MAX_ATTEMPTS,
            initial_backoff=_TQ_CLEAR_RETRY_BACKOFF_SECONDS,
            jitter=False,
            # Keep the existing clear policy: retry ordinary exceptions,
            # including TQ's RuntimeError wrappers, not only network errors.
            retry_if=lambda error: True,
        )
        if training_events is not None:
            retry_io(
                lambda: put_queue(
                    training_events,
                    completion_event,
                    timeout=getattr(loader, "io_timeout_seconds", 30),
                ),
                name="Training completion event",
                attempts=getattr(loader, "io_retry_max_attempts", 3),
            )

    if synchronize_before_cleanup:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            raise RuntimeError("Multi-node cleanup requires an initialized distributed group")
        # No owner may release shared payloads until all ranks finish this step.
        dist.barrier()

    local_error: BaseException | None = None
    if rank == 0:
        try:
            if finalizer is None:
                complete()
            else:

                def complete_async() -> None:
                    try:
                        complete()
                    except Exception:
                        # Wake the Driver even if training is waiting for a new plan.
                        # The Future retains the original failure for the main thread.
                        if training_events is not None:
                            try:
                                put_queue(
                                    training_events,
                                    {
                                        "kind": "finalization_failed",
                                        "plan_id": (completion_event or {}).get(
                                            "plan_id"
                                        ),
                                    },
                                    timeout=getattr(loader, "io_timeout_seconds", 30),
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to notify Driver of file finalization failure"
                                )
                        raise

                finalizer.submit(complete_async)
        except BaseException as exc:  # noqa: BLE001 - synchronize failures before the next training collective
            local_error = exc
    if not synchronize:
        if local_error is not None:
            raise local_error
        return
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
                "rank 0 failed to clear a completed batch"
            ) from local_error
        raise RuntimeError("rank 0 failed to clear a completed batch")


def _drain_batch_finalizer_across_ranks(finalizer, device, *, timeout=None) -> None:
    error = None
    if finalizer is not None:
        try:
            if timeout is None:
                finalizer.drain()
            else:
                finalizer.drain(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - every rank must stop before proceeding
            error = exc
    if not _all_ranks_true(error is None, device):
        raise RuntimeError("Failed to finalize completed reference batches") from error


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
    require_prepared: bool = False,
    require_reference_metadata: bool = False,
    node_ranks_per_host: int = 0,
    expected_input_identity: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
) -> Any | None:
    """Fetch one batch and make producer failures visible to every rank.

    Producer and replay errors happen before the FSDP training step. Every
    rank therefore reports its fetch result through the same collective before
    any rank is allowed to enter model collectives.  This prevents healthy
    ranks from waiting in FSDP after another rank has already started cleanup.
    """
    wait_started = time.perf_counter() if timings is not None else 0.0
    samples: Any | None = None
    local_error: BaseException | None = None
    exhausted = False
    identity_words = [0, 0, 0, 0]
    try:
        if type(require_reference_metadata) is not bool or (require_reference_metadata and require_prepared):
            raise ValueError("Reference fetch must select one batch representation")
        if type(node_ranks_per_host) is not int or node_ranks_per_host < 0 or (node_ranks_per_host and not require_reference_metadata):
            raise ValueError("Node reference assignments require deferred reference metadata")
        samples = next(source)
        if samples is None:
            raise RuntimeError("Successful batch fetch returned no samples")
        if require_prepared and (
            not isinstance(samples, TQLocalBatch) or not samples.prepared_batch
        ):
            raise RuntimeError("Reference input returned no prepared training batch")
        if require_reference_metadata:
            from verl_speco.trainer.tq_feature_store import ReadyEntry
            if (not isinstance(samples, TQLocalBatch) or samples.prepared_batch is not None
                    or not isinstance(samples.file_entries, list) or not samples.file_entries
                    or any(not isinstance(entry, ReadyEntry) for entry in samples.file_entries)
                    or samples.local_samples or samples.cleanup_after_load
                    or (not samples.is_padding_rank and samples.local_keys != [entry.key for entry in samples.file_entries])
                    or (samples.is_padding_rank and samples.local_keys)):
                raise RuntimeError("Reference input returned invalid deferred file assignments")
            node_entries = samples.node_file_entries
            if node_ranks_per_host:
                world = dist.get_world_size() if dist.is_initialized() else 1
                owner = rank // node_ranks_per_host * node_ranks_per_host
                if (node_ranks_per_host < 2 or world % node_ranks_per_host or not 0 <= rank < world
                        or (rank != owner and node_entries is not None)
                        or (rank == owner and (not isinstance(node_entries, dict)
                            or set(node_entries) != set(range(owner, owner + node_ranks_per_host))
                            or any(not isinstance(entries, list) or not entries
                                   or any(not isinstance(entry, ReadyEntry) for entry in entries)
                                   for entries in node_entries.values())
                            or node_entries[rank] != samples.file_entries))):
                    raise RuntimeError("Reference input returned invalid node assignments")
            elif node_entries is not None:
                raise RuntimeError("Reference input returned unexpected node assignments")
            if (type(samples.plan_id) is not int or samples.plan_id < 0
                    or not isinstance(samples.receipt_token, str) or not samples.receipt_token
                    or samples.assignment_digest is None):
                raise RuntimeError("Deferred reference assignment requires complete dispatch identity")
        if (require_prepared or require_reference_metadata) and isinstance(samples, TQLocalBatch):
            plan_id, token = samples.plan_id, samples.receipt_token
            if ((plan_id is not None and (type(plan_id) is not int or plan_id < 0))
                    or (token is not None and (not isinstance(token, str) or not token or len(token) > 128))):
                raise RuntimeError("Invalid reference dispatch identity")
            input_identity = samples.input_identity
            if input_identity is not None:
                input_identity = validate_input_identity(input_identity)
                if expected_input_identity is not None and input_identity != expected_input_identity:
                    raise RuntimeError("Reference batch input content differs from training baseline")
            incarnation = samples.producer_incarnation
            if incarnation is not None and (not isinstance(incarnation, str) or len(incarnation) != 32
                    or any(char not in "0123456789abcdef" for char in incarnation)):
                raise RuntimeError("Invalid reference Producer incarnation")
            assignment_digest = samples.assignment_digest
            if assignment_digest is not None and (not isinstance(assignment_digest, str) or len(assignment_digest) != 64
                    or any(char not in "0123456789abcdef" for char in assignment_digest)):
                raise RuntimeError("Invalid reference assignment digest")
            encoded = json.dumps([plan_id, token, input_identity, incarnation, assignment_digest], sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(encoded).digest()
            identity_words = [int.from_bytes(digest[i:i + 4], "big") & 0x7fffffff for i in range(0, 16, 4)]
        if isinstance(samples, TQLocalBatch) and samples.is_padding_rank:
            if require_reference_metadata:
                pass  # Mask the template after sampling geometry is known.
            elif not require_prepared or samples.prepared_batch is None:
                raise RuntimeError("Padding rank requires a prepared reference batch")
            else:
                samples = replace(samples, prepared_batch=_zero_standalone_padding_batch(samples.prepared_batch))
    except StopIteration:
        exhausted = True
    except BaseException as exc:  # noqa: BLE001
        local_error = exc

    sync_started = time.perf_counter() if timings is not None else 0.0
    values = [1 if local_error is not None else 0, 1 if exhausted else 0,
              1 if samples is not None and local_error is None else 0]
    if require_prepared or require_reference_metadata:
        # MAX of both positive and negative words gives each word's max/min in
        # the readiness collective, without exchanging payloads or another RPC.
        for word in identity_words:
            values.extend((word, -word))
    state = torch.tensor(values, dtype=torch.int32, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(state, op=dist.ReduceOp.MAX)

    state_values = state.tolist()
    any_failed = bool(state_values[0])
    any_exhausted = bool(state_values[1])
    if timings is not None:
        timings["local_batch_wait_time"] = sync_started - wait_started
        timings["rank_ready_sync_time"] = time.perf_counter() - sync_started
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
        if (require_prepared or require_reference_metadata) and bool(state_values[2]):
            raise RuntimeError(
                "Standalone reference source exhausted inconsistently across ranks; "
                "the unconfirmed batch is retained"
            )
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
    if (require_prepared or require_reference_metadata) and any(state_values[i] != -state_values[i + 1] for i in range(3, 11, 2)):
        raise RuntimeError("Standalone reference dispatch identity differs across ranks")
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
