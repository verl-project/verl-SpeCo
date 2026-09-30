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
"""External SPECO worker.

Adapted from the current in-tree
``verl/workers/engine_workers.py::DrafterWorker``. This module keeps drafter
worker behavior in ``verl_speco`` while importing upstream ``verl`` as a
dependency.
"""

import asyncio
import logging
import os
import random
import time
import uuid
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import DictConfig, open_dict

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.distributed import (
    initialize_global_process_group_ray,
    set_numa_affinity,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.integration.rollout_publish import release_draft_weights_payload
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DRAFTER_OWNER_ROUTE_MESH = "drafter_owner_route"
DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"


def _config_str(value, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return default if text in {"", "None", "null"} else text


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    get = config.get if hasattr(config, "get") else None
    if get is not None:
        return get(key, default)
    return getattr(config, key, default)


def _config_has(config: Any, key: str) -> bool:
    if config is None:
        return False
    if isinstance(config, dict):
        return key in config
    if isinstance(config, DictConfig):
        return key in config
    return hasattr(config, key)


def _config_set(config: Any, key: str, value: Any) -> None:
    if isinstance(config, DictConfig):
        with open_dict(config):
            config[key] = value
        return
    if isinstance(config, dict):
        config[key] = value
        return
    setattr(config, key, value)


def _is_oom_error(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "out of memory" in text or "oom" in text or "memory allocation" in text


def _is_ray_object_ref(value) -> bool:
    object_ref_type = getattr(ray, "ObjectRef", ())
    return bool(object_ref_type) and isinstance(value, object_ref_type)


async def _resolve_ray_object_ref(value):
    """Resolve an ObjectRef without blocking this async Ray actor's event loop."""
    if _is_ray_object_ref(value):
        return await value
    return value


async def _resolve_hidden_state_chunks(chunks, expected_rows: int | None = None):
    if not chunks:
        return None
    unique_refs: dict[Any, Any] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ref = chunk.get("ref")
        if ref is not None:
            unique_refs.setdefault(id(ref), ref)
    resolved_values = await asyncio.gather(
        *(_resolve_ray_object_ref(ref) for ref in unique_refs.values())
    )
    resolved_cache = dict(zip(unique_refs, resolved_values, strict=True))
    pieces = []
    full_rows = int(expected_rows or 0)
    hidden_size = None
    dtype = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ref = chunk.get("ref")
        if ref is None:
            continue
        cache_key = id(ref)
        tensor = resolved_cache[cache_key]
        if not torch.is_tensor(tensor):
            continue
        tensor = cast(torch.Tensor, tensor)
        start = int(chunk.get("chunk_start", 0) or 0)
        length = int(chunk.get("chunk_length", 0) or 0)
        if length <= 0:
            continue
        part = tensor[start : start + length]
        row_indices = chunk.get("chunk_row_indices")
        if torch.is_tensor(row_indices):
            row_indices = cast(torch.Tensor, row_indices)
            row_indices = row_indices.detach().cpu().long().reshape(-1)
        elif isinstance(row_indices, (list, tuple)):
            row_indices = torch.tensor(
                [int(idx) for idx in row_indices], dtype=torch.long
            )
        else:
            row_indices = torch.arange(length, dtype=torch.long)
        if int(row_indices.numel()) != int(part.shape[0]):
            logger.debug(
                "Skip malformed SPECO hidden chunk: rows=%s tensor_rows=%s",
                int(row_indices.numel()),
                int(part.shape[0]),
            )
            continue
        pieces.append((row_indices, part))
        full_rows = max(
            full_rows,
            int(row_indices.max().item()) + 1 if int(row_indices.numel()) > 0 else 0,
        )
        hidden_size = int(part.shape[-1])
        dtype = part.dtype
    if not pieces or hidden_size is None:
        return None
    output = torch.zeros((full_rows, hidden_size), dtype=dtype)
    for row_indices, part in pieces:
        output[row_indices] = part.to(device=output.device, dtype=output.dtype)
    return output.unsqueeze(0)


@dataclass(frozen=True)
class RolloutParallelLayout:
    infer_tp: int
    infer_pp: int
    rollout_world_size: int
    num_replicas: int
    replica_training_ranks: list[list[int]]


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
        replica_training_ranks.append(
            [replica_base + tp_rank * infer_pp for tp_rank in range(int(rollout_tp))]
        )

    return RolloutParallelLayout(
        infer_tp=infer_tp,
        infer_pp=infer_pp,
        rollout_world_size=rollout_world_size,
        num_replicas=num_replicas,
        replica_training_ranks=replica_training_ranks,
    )


def build_drafter_training_device_mesh(
    device_type: str, layout: RolloutParallelLayout
) -> DeviceMesh:
    return DeviceMesh(
        device_type=device_type,
        mesh=torch.tensor(layout.replica_training_ranks, dtype=torch.int64),
        mesh_dim_names=("dp", "sp"),
    )


def _dispatch_nd_compute(
    dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs
):
    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)

    def dispatch_value(value):
        if not isinstance(value, (tuple, list)):
            return [value for _ in range(worker_group.world_size)]
        assert len(value) == dp_size
        max_workers = max(1, min(len(value), os.cpu_count()))
        value_refs = parallel_put(value, max_workers=max_workers)
        return [value_refs[dp_rank_mapping[i]] for i in range(worker_group.world_size)]

    all_args = [dispatch_value(arg) for arg in args]
    all_kwargs = {key: dispatch_value(value) for key, value in kwargs.items()}

    return tuple(all_args), all_kwargs


def _collect_nd_compute(collect_mask: list[bool], worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return [
        output[global_rank]
        for global_rank in range(worker_group.world_size)
        if collect_mask[global_rank]
    ]


def _dispatch_lazy_compute(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(
            mesh_name
        )
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    dp_size = max(dp_rank_mapping) + 1
    return _dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


def _collect_lazy_compute(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(
            mesh_name
        )
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    return _collect_nd_compute(
        worker_group._collect_info[mesh_name], worker_group, *args, **kwargs
    )


def make_nd_compute_dispatch_fn(mesh_name):
    return {
        "dispatch_fn": partial(_dispatch_lazy_compute, mesh_name),
        "collect_fn": partial(_collect_lazy_compute, mesh_name),
    }


def _resolve_drafter_init_backend(device_name: str) -> str:
    device_name = str(device_name).lower()
    if device_name == "npu":
        return "cpu:gloo,npu:hccl"
    if device_name == "cuda":
        return "cpu:gloo,cuda:nccl"
    if device_name == "cpu":
        return "cpu:gloo"
    raise ValueError(f"Unsupported drafter device_name={device_name!r}")


@contextmanager
def _preserve_process_rng_state(device_name: str):
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_cpu_rng_state = torch.get_rng_state()
    torch_device_rng_state = None
    torch_device = None

    if str(device_name).lower() != "cpu":
        torch_device = get_torch_device()
        try:
            torch_device_rng_state = torch_device.get_rng_state()
        except (AttributeError, RuntimeError):
            torch_device_rng_state = None

    try:
        yield
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_cpu_rng_state)
        if torch_device is not None and torch_device_rng_state is not None:
            try:
                torch_device.set_rng_state(torch_device_rng_state)
            except (AttributeError, RuntimeError):
                logger.warning(
                    "Failed to restore %s RNG state after SPECO training.", device_name
                )


class SpecoWorker(Worker):
    """Standalone SPECO drafter worker.

    The worker receives CPU rollout features from the PPO trainer and trains the
    drafter model periodically according to global RL steps.
    """

    def __init__(
        self,
        config: DictConfig,
        role: str = "speco",
        device_name: Optional[str] = None,
        **kwargs,
    ):
        Worker.__init__(self)
        self.config = config
        self.role = role
        if device_name is None:
            raise ValueError(
                "SpecoWorker requires an explicit device_name from the trainer initialization path"
            )
        self.device_name = str(device_name).lower()
        self.trainer: Any = None
        self.feature_writer: Optional[TorchShardFeatureStore] = None
        self.feature_writer_path: Optional[str] = None
        self.last_global_step: Optional[int] = None
        self.last_trained_step: Optional[int] = None
        self.worker_incarnation = uuid.uuid4().hex
        self._staged_rollout_features: dict[str, dict[str, object]] = {}
        self._collection_commit_journals: dict[str, dict[str, object]] = {}
        self._prepared_training_plan_id: Optional[str] = None
        self._prepared_training_data_version: Optional[int] = None
        self._prepared_training_target_version: Optional[int] = None
        # These timestamps make the Bubble admission path observable.  In
        # particular, an idle plan can be reclaimed after activation but before
        # its first optimizer batch.  That must not be indistinguishable from a
        # normal zero-batch training result.
        self._prepared_training_activation_elapsed_sec: float = 0.0
        self._prepared_training_preflight_elapsed_sec: float = 0.0
        self._prepared_training_ready_ts: Optional[float] = None
        self._drafter_reclaim_requested = False
        self.training_process_group = None
        self.dp_process_group = None
        self.training_group_ranks: list[int] = []
        self.training_group_world_size = 1
        self.dp_group_ranks: list[int] = []
        self.dp_group_world_size = 1
        self.full_collective_ranks: list[int] = []
        self.sync_collective_ranks: list[int] = []
        self.num_rollout_replicas = 1
        self.training_device_mesh = None
        self._process_group_initialized = False
        self._training_group_initialized = False
        self._last_trained_execution_strategy: Optional[str] = None
        self._last_trained_target_worker_ids: tuple[str, ...] = ()

        self.rollout_tp = int(self.config.rollout.tensor_model_parallel_size)
        self.rollout_dp = int(self.config.rollout.data_parallel_size)
        self.infer_tp = self.rollout_tp * self.rollout_dp
        self.rollout_pp = int(self.config.rollout.pipeline_model_parallel_size)
        self.rollout_world_size = self.infer_tp * self.rollout_pp
        self.rollout_rank = self.rank % self.rollout_world_size
        self.replica_rank = self.rank // self.rollout_world_size
        self.local_infer_tp_rank = self.rollout_rank // self.rollout_pp
        self.local_infer_pp_rank = self.rollout_rank % self.rollout_pp
        self.local_drafter_sp_rank = None
        self.in_drafter_train_group = False
        self.is_drafter_group_leader = False
        self.global_publish_leader_rank = None
        self.is_global_publish_leader = False

        self.enable_drafter = bool(
            self.config.rollout.drafter.enable
            and self.config.rollout.drafter.enable_drafter_training
        )

    def _drafter_execution_strategy(self) -> str:
        training_cfg = self.config.rollout.drafter.training
        scheduler_cfg = _config_get(training_cfg, "scheduler", {}) or {}
        execution_cfg = _config_get(scheduler_cfg, "execution", {}) or {}
        strategy = _config_get(
            execution_cfg,
            "strategy",
            _config_get(training_cfg, "execution_strategy", "sync"),
        )
        return str(strategy or "sync").strip().lower()

    def _use_replica_local_idle_training(self) -> bool:
        return self._drafter_execution_strategy() == "rollout_idle_worker"

    def _replica_local_idle_trainer_config(self) -> DictConfig:
        """Return a Bubble-only trainer config.

        Replica-local Bubble training may wrap only the drafter trainable subset
        with FSDP.  Some drafter variants freeze target/auxiliary parameters
        while leaving the draft head trainable; FSDP1 with ``use_orig_params``
        disabled cannot flatten such mixed ``requires_grad`` groups.  Keep the
        sync/full-collective config untouched, but force the replica-local
        Bubble trainer copy to preserve original params.
        """

        trainer_config = deepcopy(self.config)
        changed_paths: list[str] = []

        def _force_use_orig_params(fsdp_config: Any, path: str) -> None:
            if fsdp_config is None or not _config_has(fsdp_config, "use_orig_params"):
                return
            old_value = _config_get(fsdp_config, "use_orig_params", None)
            if bool(old_value) is True:
                return
            _config_set(fsdp_config, "use_orig_params", True)
            changed_paths.append(f"{path}.use_orig_params:{old_value}->True")

        actor_cfg = _config_get(trainer_config, "actor", None)
        _force_use_orig_params(
            _config_get(actor_cfg, "fsdp_config", None),
            "actor.fsdp_config",
        )

        rollout_cfg = _config_get(trainer_config, "rollout", None)
        drafter_cfg = _config_get(rollout_cfg, "drafter", None)
        training_cfg = _config_get(drafter_cfg, "training", None)
        _force_use_orig_params(
            _config_get(training_cfg, "fsdp_config", None),
            "rollout.drafter.training.fsdp_config",
        )

        if changed_paths:
            logger.warning(
                "[BubbleTime] replica_local_fsdp_override: rank=%s paths=%s "
                "reason=mixed_requires_grad_requires_use_orig_params",
                self.rank,
                tuple(changed_paths),
            )
            print(
                "[BubbleTime] replica_local_fsdp_override: "
                f"rank={self.rank} paths={tuple(changed_paths)} "
                "reason=mixed_requires_grad_requires_use_orig_params",
                flush=True,
            )
        return trainer_config

    def _ensure_process_group_initialized(self):
        if not dist.is_initialized():
            initialize_global_process_group_ray(
                timeout_second=None,
                backend=_resolve_drafter_init_backend(self.device_name),
            )
        if not self._process_group_initialized:
            set_numa_affinity()
            self._process_group_initialized = True
        if dist.is_initialized() and dist.get_rank() != self.rank:
            raise RuntimeError(
                f"SpecoWorker rank mismatch: worker_rank={self.rank}, dist_rank={dist.get_rank()}"
            )

    def _ensure_training_group_initialized(self):
        if self._training_group_initialized:
            return

        self._ensure_process_group_initialized()
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        rollout_layout = build_rollout_parallel_layout(
            world_size=world_size,
            rollout_tp=self.rollout_tp,
            rollout_dp=self.rollout_dp,
            rollout_pp=self.config.rollout.pipeline_model_parallel_size,
        )
        self.num_rollout_replicas = rollout_layout.num_replicas

        self.global_publish_leader_rank = (
            rollout_layout.replica_training_ranks[0][0]
            if rollout_layout.replica_training_ranks
            else None
        )
        self.is_global_publish_leader = self.rank == self.global_publish_leader_rank
        self.training_device_mesh = build_drafter_training_device_mesh(
            self.device_name, rollout_layout
        )
        owner_route_rank = self.num_rollout_replicas
        owner_route_collect = False

        mesh_coordinate = self.training_device_mesh.get_coordinate()
        self.in_drafter_train_group = mesh_coordinate is not None
        if self.in_drafter_train_group:
            mesh_dp_rank, mesh_sp_rank = mesh_coordinate
            if mesh_dp_rank != self.replica_rank:
                raise ValueError(
                    "SPECO mesh dp coordinate does not match rollout replica rank: "
                    f"mesh_dp_rank={mesh_dp_rank}, rollout_replica_rank={self.replica_rank}, "
                    f"global_rank={self.rank}"
                )
            self.training_process_group = self.training_device_mesh["sp"].get_group()
            self.dp_process_group = self.training_device_mesh["dp"].get_group()
            self.training_group_ranks = list(
                rollout_layout.replica_training_ranks[mesh_dp_rank]
            )
            self.dp_group_ranks = [
                rollout_layout.replica_training_ranks[replica_rank][mesh_sp_rank]
                for replica_rank in range(rollout_layout.num_replicas)
            ]
            self.training_group_world_size = self.training_device_mesh["sp"].size()
            self.dp_group_world_size = self.training_device_mesh["dp"].size()
            self.sync_collective_ranks = [
                rank
                for replica_ranks in rollout_layout.replica_training_ranks
                for rank in replica_ranks
            ]
            if self._use_replica_local_idle_training():
                self.full_collective_ranks = list(self.training_group_ranks)
            elif self.dp_group_world_size > 1:
                self.full_collective_ranks = list(self.sync_collective_ranks)
            else:
                self.full_collective_ranks = list(self.training_group_ranks)
            self.local_drafter_sp_rank = mesh_sp_rank
            self.is_drafter_group_leader = mesh_sp_rank == 0
            owner_route_rank = self.replica_rank
            owner_route_collect = self.is_drafter_group_leader

        self._register_dispatch_collect_info(
            mesh_name=DRAFTER_OWNER_ROUTE_MESH,
            dp_rank=owner_route_rank,
            is_collect=owner_route_collect,
        )
        self._register_dispatch_collect_info(
            mesh_name=DRAFTER_TARGET_SYNC_MESH,
            dp_rank=0,
            is_collect=True,
        )
        self._training_group_initialized = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if not self.enable_drafter:
            return

        self._ensure_training_group_initialized()
        if not self.in_drafter_train_group:
            return

        from verl_speco.backends.factory import build_trainer_backend
        from verl_speco.trainer.base_trainer import DrafterBaseTrainer

        if self._use_replica_local_idle_training():
            trainer_config = self._replica_local_idle_trainer_config()
            trainer_backend = build_trainer_backend(
                trainer_config, trainer_config.model
            )
            logger.warning(
                "[BubbleTime] replica_local_training_mesh: rank=%s replica_rank=%s "
                "training_group_ranks=%s sync_collective_ranks=%s "
                "training_group_world_size=%s dp_group_world_size=%s",
                self.rank,
                self.replica_rank,
                tuple(self.training_group_ranks),
                tuple(self.sync_collective_ranks or self.full_collective_ranks),
                self.training_group_world_size,
                self.dp_group_world_size,
            )
            print(
                "[BubbleTime] replica_local_training_mesh: "
                f"rank={self.rank} replica_rank={self.replica_rank} "
                f"training_group_ranks={tuple(self.training_group_ranks)} "
                "sync_collective_ranks="
                f"{tuple(self.sync_collective_ranks or self.full_collective_ranks)} "
                f"training_group_world_size={self.training_group_world_size} "
                f"dp_group_world_size={self.dp_group_world_size}",
                flush=True,
            )
            self.trainer = DrafterBaseTrainer(
                config=trainer_config,
                world_size=self.training_group_world_size,
                rollout_dp_rank=self.replica_rank,
                training_device_mesh=None,
                training_process_group=self.training_process_group,
                data_parallel_process_group=None,
                backend=trainer_backend,
            )
            return

        trainer_backend = build_trainer_backend(self.config, self.config.model)
        self.trainer = DrafterBaseTrainer(
            config=self.config,
            world_size=self.training_group_world_size,
            rollout_dp_rank=self.replica_rank,
            training_device_mesh=self.training_device_mesh,
            backend=trainer_backend,
        )

    def _store_rollout_sample(
        self,
        batch: dict,
        hidden_states: torch.Tensor,
        target_logprobs: Optional[torch.Tensor] = None,
    ) -> bool:
        if (
            not self.enable_drafter
            or not self.in_drafter_train_group
            or self.trainer is None
        ):
            return False
        if self._drafter_training_mode() == "collect_only":
            return self._write_rollout_feature_sample(
                batch, hidden_states, target_logprobs
            )
        if hidden_states is None:
            raise RuntimeError(
                "Online drafter training requires collected hidden states"
            )
        return bool(
            self.trainer.collect_online_data(batch, hidden_states, target_logprobs)
        )

    def _drafter_training_mode(self) -> str:
        return (
            str(self.config.rollout.drafter.training.get("mode", "online") or "online")
            .strip()
            .lower()
        )

    def _get_feature_writer(self) -> Optional[TorchShardFeatureStore]:
        feature_store_cfg = self.config.rollout.drafter.training.get(
            "feature_store", None
        )
        if feature_store_cfg is None:
            return None
        path = _config_str(feature_store_cfg.get("path", None))
        if not path:
            return None
        if self.feature_writer is not None and self.feature_writer_path == path:
            return self.feature_writer
        model_cfg = self.config.get("model", None)
        target_model_path = (
            _config_str(model_cfg.get("path", None)) if model_cfg is not None else ""
        )
        self.feature_writer = TorchShardFeatureStore(
            path,
            max_samples_per_shard=int(
                feature_store_cfg.get("max_samples_per_shard", 1024)
            ),
            strict_schema=bool(feature_store_cfg.get("strict_schema", True)),
            metadata={
                "algorithm": str(
                    self.config.rollout.drafter.speculative_algorithm
                ).upper(),
                "target_model_path": target_model_path,
                "drafter_model_path": _config_str(
                    self.config.rollout.drafter.get("model_path", None)
                ),
                "source": "rl_collect_only",
            },
            shard_prefix=f"rank{int(self.rank):05d}_pid{int(os.getpid())}",
        )
        self.feature_writer_path = path
        return self.feature_writer

    def _build_rollout_loss_mask(
        self, batch: dict, input_ids: torch.Tensor
    ) -> torch.Tensor:
        loss_mask_value = batch.get("loss_mask")
        if torch.is_tensor(loss_mask_value):
            loss_mask_value = cast(torch.Tensor, loss_mask_value)
            return loss_mask_value.detach().cpu().float().reshape(-1)
        ids = input_ids.detach().cpu().reshape(-1)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        prompts = batch.get("prompts")
        responses = batch.get("responses")
        if torch.is_tensor(prompts) and torch.is_tensor(responses):
            prompts = cast(torch.Tensor, prompts)
            responses = cast(torch.Tensor, responses)
            prompt_len = int(prompts.reshape(-1).numel())
            response_ids = responses.detach().cpu().reshape(-1)
            model_cfg = self.config.get("model", None)
            pad_token_id = (
                int(model_cfg.get("pad_token_id", 0) or 0)
                if model_cfg is not None
                else 0
            )
            max_response = max(
                0, min(int(response_ids.numel()), int(ids.numel()) - prompt_len)
            )
            if max_response > 0:
                loss_mask[prompt_len : prompt_len + max_response] = (
                    response_ids[:max_response] != pad_token_id
                ).float()
        else:
            loss_mask[:] = 1.0
        return loss_mask

    def _write_rollout_feature_sample(
        self,
        batch: dict,
        hidden_states: torch.Tensor,
        target_logprobs: Optional[torch.Tensor],
    ) -> bool:
        writer = self._get_feature_writer()
        if writer is None:
            logger.warning(
                "[SpecoWorker rank=%s] training.mode=collect_only but feature_store.path is empty; drop sample",
                self.rank,
            )
            return False
        full_input_ids = batch["input_ids"].detach().cpu().reshape(-1)
        full_loss_mask = self._build_rollout_loss_mask(batch, full_input_ids)
        hidden_states = hidden_states.detach().cpu()
        hidden_rows = int(
            hidden_states.size(1)
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1
            else hidden_states.size(0)
        )
        hidden_positions = batch.get("hidden_positions")
        if torch.is_tensor(hidden_positions):
            hidden_positions = cast(torch.Tensor, hidden_positions)
            hidden_positions = hidden_positions.detach().cpu().long().reshape(-1)
        else:
            hidden_positions = None
        feature_start, feature_end, position_ids = self._resolve_rollout_feature_window(
            full_input_ids,
            hidden_rows,
            hidden_positions=hidden_positions,
            hidden_position_start=batch.get("hidden_position_start"),
            hidden_position_end=batch.get("hidden_position_end"),
        )
        input_ids = full_input_ids[feature_start:feature_end]
        loss_mask = full_loss_mask[feature_start:feature_end]
        target_logprobs = self._align_rollout_target_logprobs(
            target_logprobs,
            feature_start=feature_start,
            train_rows=max(int(input_ids.numel()) - 1, 0),
            target_position_start=batch.get("target_logprobs_position_start"),
            target_position_end=batch.get("target_logprobs_position_end"),
        )
        model_cfg = self.config.get("model", None)
        target_model_path = (
            _config_str(model_cfg.get("path", None)) if model_cfg is not None else ""
        )
        algorithm = str(self.config.rollout.drafter.speculative_algorithm).upper()
        default_hidden_layout = resolve_drafter_hidden_states_layout(
            algorithm,
            self.config.rollout.drafter.training,
        )
        metadata = {
            "source": batch.get("hidden_target_logprobs_source", "rl_rollout"),
            "global_step": batch.get("global_step", self.last_global_step),
            "target_model_path": target_model_path,
            "drafter_model_path": _config_str(
                self.config.rollout.drafter.get("model_path", None)
            ),
            "hidden_states_layout": batch.get("hidden_states_layout")
            or default_hidden_layout,
            "target_layer_ids": batch.get("target_layer_ids"),
            "use_logits": bool(
                self.config.rollout.drafter.training.get("use_logits", False)
            ),
            "sequence_length": int(input_ids.numel()),
            "loss_tokens": int(loss_mask.sum().item()),
            "full_sequence_length": int(full_input_ids.numel()),
            "feature_start": int(feature_start),
            "feature_end": int(feature_end),
        }
        for key in (
            "hidden_position_start",
            "hidden_position_end",
            "hidden_positions",
            "hidden_prefix_cache_rows",
            "hidden_window_start",
            "hidden_window_end",
            "hidden_lm_head_fingerprint",
            "hidden_last_hidden_logprob_check",
            "hidden_raw_topk_logprob_check",
            "hidden_last_hidden_filter",
            "hidden_last_hidden_select",
            "target_logprobs_position_start",
            "target_logprobs_position_end",
        ):
            if key in batch:
                metadata[key] = batch[key]
        feature_sample = DraftFeatureSample(
            algorithm=algorithm,
            input_ids=input_ids,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            target_logprobs=target_logprobs,
            position_ids=position_ids,
            metadata=metadata,
        )
        writer.write_many([feature_sample])
        return True

    @staticmethod
    def _align_rollout_target_logprobs(
        target_logprobs: Optional[torch.Tensor],
        *,
        feature_start: int,
        train_rows: int,
        target_position_start,
        target_position_end,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(target_logprobs):
            return None
        target_logprobs = cast(torch.Tensor, target_logprobs)
        target = target_logprobs.detach().cpu()
        while target.dim() > 3 and target.size(0) == 1:
            target = target.squeeze(0)
        if target.dim() != 3:
            return target.contiguous()

        try:
            position_start = int(target_position_start)
        except (TypeError, ValueError):
            position_start = int(feature_start) + 1
        try:
            position_end = int(target_position_end)
        except (TypeError, ValueError):
            position_end = position_start + int(target.size(0))
        position_end = min(
            max(position_end, position_start), position_start + int(target.size(0))
        )

        desired_start = int(feature_start) + 1
        desired_end = desired_start + max(int(train_rows), 0)
        slice_start = min(max(desired_start - position_start, 0), int(target.size(0)))
        slice_end = min(
            max(desired_end - position_start, slice_start),
            int(position_end - position_start),
        )
        return target[slice_start:slice_end].contiguous()

    @staticmethod
    def _resolve_rollout_feature_window(
        input_ids: torch.Tensor,
        hidden_rows: int,
        *,
        hidden_positions: Optional[torch.Tensor],
        hidden_position_start,
        hidden_position_end,
    ) -> tuple[int, int, torch.Tensor]:
        input_len = int(input_ids.numel())
        hidden_rows = max(int(hidden_rows), 0)
        if hidden_positions is not None and int(hidden_positions.numel()) > 0:
            positions = hidden_positions[:hidden_rows].long()
            start = int(positions[0].item())
            if int(positions.numel()) == hidden_rows and bool(
                torch.all(positions[1:] == positions[:-1] + 1).item()
            ):
                end = int(positions[-1].item()) + 1
                if 0 <= start < end <= input_len:
                    return start, end, positions + 1
        else:
            positions = None

        try:
            start = int(hidden_position_start)
        except (TypeError, ValueError):
            start = 0
        try:
            end = int(hidden_position_end)
        except (TypeError, ValueError):
            end = start + hidden_rows
        start = min(max(start, 0), input_len)
        end = min(max(end, start), input_len)
        if end - start != hidden_rows:
            end = min(start + hidden_rows, input_len)
        if end <= start:
            start = 0
            end = min(hidden_rows, input_len)
        position_ids = torch.arange(start + 1, end + 1, dtype=torch.long)
        return start, end, position_ids

    def _flush_rollout_features_for_step(self) -> None:
        if (
            self._drafter_training_mode() != "collect_only"
            or self.feature_writer is None
        ):
            return
        feature_store_cfg = self.config.rollout.drafter.training.get(
            "feature_store", {}
        )
        flush_interval = int(feature_store_cfg.get("flush_interval_steps", 1))
        self.feature_writer.flush_on_step(self.last_global_step, flush_interval)

    def _collection_worker_result(
        self,
        *,
        collection_id: str,
        staged_samples: int = 0,
        accepted_samples: int = 0,
        rejected_samples: int = 0,
        collected: bool = False,
        reason: str,
        buffer_version_before: int | None = None,
        expired_stages: int = 0,
    ) -> dict[str, Any]:
        if self.last_global_step is None:
            raise RuntimeError(
                "Cannot collect drafter samples before setting global_step"
            )
        source_global_step = int(self.last_global_step)
        current_buffer_version = int(
            self.trainer.buffer_version if self.trainer is not None else 0
        )
        if buffer_version_before is None:
            buffer_version_before = current_buffer_version
        return {
            "collection_id": collection_id,
            "worker_id": str(self.replica_rank),
            "worker_incarnation": self.worker_incarnation,
            "source_global_step": source_global_step,
            "staged_samples": staged_samples,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
            "buffer_version_before": buffer_version_before,
            "buffer_version_after": current_buffer_version,
            "data_version": source_global_step,
            "collected": collected,
            "reason": reason,
            "expired_stages": expired_stages,
        }

    @staticmethod
    def _collection_request(requests: list[dict]) -> dict:
        if len(requests) != 1 or not isinstance(requests[0], dict):
            raise ValueError("Expected exactly one drafter collection request")
        return requests[0]

    def _cleanup_expired_collection_stages(self) -> int:
        ttl_sec = float(
            cast(
                Any,
                self.config.rollout.drafter.training.get(
                    "collection_stage_ttl_sec", 300.0
                )
                or 0.0,
            )
        )
        if ttl_sec <= 0:
            return 0
        deadline = time.monotonic() - ttl_sec
        expired = [
            collection_id
            for collection_id, entry in self._staged_rollout_features.items()
            if float(cast(Any, entry.get("staged_at", 0.0))) < deadline
        ]
        for collection_id in expired:
            self._staged_rollout_features.pop(collection_id, None)
        return len(expired)

    def _snapshot_collection_buffer(self) -> dict[str, object]:
        trainer = self.trainer
        if trainer is None:
            return {}
        return {
            "buffer_version": int(trainer.buffer_version),
            "collected_data": deque(
                trainer.collected_data, maxlen=trainer.collected_data.maxlen
            ),
            "data_buffer": deque(
                trainer.data_buffer.buffer,
                maxlen=trainer.data_buffer.buffer.maxlen,
            ),
            "data_buffer_step": trainer.data_buffer._current_step,
        }

    def _restore_collection_buffer(self, snapshot: dict[str, object]) -> None:
        trainer = self.trainer
        if trainer is None or not snapshot:
            return
        trainer.collected_data = cast(deque, snapshot["collected_data"])
        trainer.data_buffer.buffer = cast(deque, snapshot["data_buffer"])
        trainer.data_buffer._current_step = cast(
            Optional[int], snapshot["data_buffer_step"]
        )
        trainer.buffer_version = int(cast(Any, snapshot["buffer_version"]))

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH)
    )
    def stage_rollout_features(self, requests: list[dict]):
        expired_stages = self._cleanup_expired_collection_stages()
        request = self._collection_request(requests)
        collection_id = str(request.get("collection_id", ""))
        samples = request.get("samples")
        if (
            not self.enable_drafter
            or not self.in_drafter_train_group
            or self.trainer is None
        ):
            return self._collection_worker_result(
                collection_id=collection_id, reason="worker_not_ready"
            )
        if not collection_id:
            return self._collection_worker_result(
                collection_id="", reason="missing_collection_id"
            )
        if (
            collection_id in self._staged_rollout_features
            or collection_id in self._collection_commit_journals
        ):
            return self._collection_worker_result(
                collection_id=collection_id,
                reason="collection_already_staged",
                expired_stages=expired_stages,
            )
        if not isinstance(samples, list):
            return self._collection_worker_result(
                collection_id=collection_id, reason="invalid_collection_samples"
            )
        self._staged_rollout_features[collection_id] = {
            "samples": samples,
            "staged_at": time.monotonic(),
        }
        return self._collection_worker_result(
            collection_id=collection_id,
            staged_samples=len(samples),
            reason="collection_staged",
            expired_stages=expired_stages,
        )

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH)
    )
    async def commit_rollout_features(self, requests: list[dict]):
        request = self._collection_request(requests)
        collection_id = str(request.get("collection_id", ""))
        staged = self._staged_rollout_features.pop(collection_id, None)
        if staged is None:
            return self._collection_worker_result(
                collection_id=collection_id, reason="collection_not_staged"
            )
        samples = staged.get("samples")
        if not isinstance(samples, list):
            return self._collection_worker_result(
                collection_id=collection_id, reason="invalid_staged_collection"
            )
        snapshot = self._snapshot_collection_buffer()
        self._collection_commit_journals[collection_id] = snapshot
        return await self._commit_rollout_features(collection_id, samples)

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH)
    )
    def abort_rollout_features(self, requests: list[dict]):
        request = self._collection_request(requests)
        collection_id = str(request.get("collection_id", ""))
        removed = self._staged_rollout_features.pop(collection_id, None)
        return self._collection_worker_result(
            collection_id=collection_id,
            reason="collection_aborted" if removed is not None else "collection_absent",
        )

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH)
    )
    def rollback_rollout_features(self, requests: list[dict]):
        request = self._collection_request(requests)
        collection_id = str(request.get("collection_id", ""))
        self._staged_rollout_features.pop(collection_id, None)
        snapshot = self._collection_commit_journals.pop(collection_id, None)
        if snapshot is not None:
            self._restore_collection_buffer(snapshot)
        return self._collection_worker_result(
            collection_id=collection_id,
            reason=(
                "collection_rolled_back"
                if snapshot is not None
                else "collection_rollback_absent"
            ),
        )

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH)
    )
    def finalize_rollout_features(self, requests: list[dict]):
        request = self._collection_request(requests)
        collection_id = str(request.get("collection_id", ""))
        journal = self._collection_commit_journals.pop(collection_id, None)
        return self._collection_worker_result(
            collection_id=collection_id,
            reason=(
                "collection_finalized"
                if journal is not None
                else "collection_finalize_absent"
            ),
        )

    async def _commit_rollout_features(
        self, collection_id: str, samples: list[dict]
    ) -> dict[str, Any]:
        buffer_version_before = int(
            self.trainer.buffer_version if self.trainer is not None else 0
        )
        result = self._collection_worker_result(
            collection_id=collection_id,
            buffer_version_before=buffer_version_before,
            collected=True,
            reason="collection_completed",
        )
        if not samples:
            return result
        for sample in samples:
            if not sample:
                result["rejected_samples"] += 1
                continue
            batch = {
                "input_ids": sample["input_ids"],
                "prompts": sample["prompts"],
                "responses": sample["responses"],
            }
            for key in (
                "hidden_position_start",
                "hidden_position_end",
                "hidden_positions",
                "hidden_prefix_cache_rows",
                "hidden_window_start",
                "hidden_window_end",
                "hidden_lm_head_fingerprint",
                "hidden_last_hidden_logprob_check",
                "hidden_target_logprobs_source",
                "hidden_raw_topk_logprob_check",
                "hidden_raw_target_logprobs",
                "hidden_raw_target_logprobs_positions",
                "hidden_raw_target_logprobs_position_start",
                "hidden_raw_target_logprobs_position_end",
                "hidden_last_hidden_filter",
                "hidden_last_hidden_select",
                "hidden_states_layout",
                "target_logprobs_position_start",
                "target_logprobs_position_end",
                "global_step",
                "source_replica_rank",
                "_speco_global_sample_id",
            ):
                if key in sample:
                    batch[key] = sample[key]
            hidden = sample.get("hidden_states")
            if hidden is None:
                hidden_chunks = sample.get("hidden_states_ref_chunks")
                if hidden_chunks:
                    expected_rows = None
                    hidden_positions = batch.get("hidden_positions")
                    if torch.is_tensor(hidden_positions):
                        hidden_positions = cast(torch.Tensor, hidden_positions)
                        expected_rows = int(hidden_positions.numel())
                    hidden = await _resolve_hidden_state_chunks(
                        hidden_chunks, expected_rows=expected_rows
                    )
                else:
                    hidden = await _resolve_ray_object_ref(
                        sample.get("hidden_states_ref")
                    )
            target_logprobs = sample.get("target_logprobs")
            if target_logprobs is None:
                target_logprobs = await _resolve_ray_object_ref(
                    sample.get("target_logprobs_ref")
                )
            if hidden is None:
                result["rejected_samples"] += 1
                continue
            stored = self._store_rollout_sample(
                batch=batch,
                hidden_states=hidden,
                target_logprobs=target_logprobs,
            )
            if stored:
                result["accepted_samples"] += 1
            else:
                result["rejected_samples"] += 1
        self._flush_rollout_features_for_step()
        result["buffer_version_after"] = int(
            self.trainer.buffer_version if self.trainer is not None else 0
        )
        if result["rejected_samples"]:
            result["collected"] = False
            result["reason"] = "samples_rejected"
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_global_step(self, global_step: int):
        if (
            not self.enable_drafter
            or not self.in_drafter_train_group
            or self.trainer is None
        ):
            return
        if global_step is None:
            return
        if self.last_global_step == global_step:
            return
        self.last_global_step = global_step
        self.trainer.clear_pending_publish_state_dict()
        self.trainer.increment_rl_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        global_step: int,
        wait: bool = True,
        worker_ids: Optional[tuple[str, ...]] = None,
    ):
        if not self.enable_drafter:
            return {"saved": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"saved": False, "reason": "not_in_training_group"}
        if worker_ids and str(self.rank) not in {
            str(worker_id) for worker_id in worker_ids
        }:
            return {"saved": False, "reason": "not_checkpoint_group"}
        if global_step is None:
            return {"saved": False, "reason": "missing_global_step"}
        result = self.trainer.save_checkpoint(
            int(global_step),
            wait=wait,
        )
        if self.is_drafter_group_leader:
            logger.debug(
                "[speco checkpoint] replica=%s global_step=%s workers=%s result=%s",
                self.replica_rank,
                global_step,
                worker_ids,
                result,
            )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_checkpoint(self):
        if not self.enable_drafter:
            return {"waited": False, "completed": True, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {
                "waited": False,
                "completed": True,
                "reason": "not_in_training_group",
            }
        result = self.trainer.wait_checkpoint()
        if self.is_drafter_group_leader:
            logger.debug(
                "[speco checkpoint wait] replica=%s result=%s",
                self.replica_rank,
                result,
            )
        return result

    @register(
        dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_TARGET_SYNC_MESH),
        blocking=False,
    )
    def sync_target_lm_head_weight(
        self, payload: Optional[dict], global_step: Optional[int] = None
    ):
        if not self.enable_drafter:
            return {"accepted": False, "applied": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {
                "accepted": False,
                "applied": False,
                "reason": "not_in_training_group",
            }
        if not payload:
            return {"accepted": False, "applied": False, "reason": "missing_payload"}
        target_worker_ids = payload.get("target_worker_ids")
        if target_worker_ids:
            target_worker_id_set = {str(worker_id) for worker_id in target_worker_ids}
            if str(self.rank) not in target_worker_id_set:
                return {
                    "accepted": False,
                    "applied": False,
                    "reason": "not_target_worker",
                    "worker_id": str(self.rank),
                    "target_worker_ids": tuple(sorted(target_worker_id_set)),
                }

        weight = payload.get("weight")
        row_indices = payload.get("row_indices")
        source_vocab_size = payload.get("source_vocab_size")
        defer_device_apply = bool(payload.get("defer_device_apply", False))
        name = payload.get("name")
        result = self.trainer.sync_target_lm_head_weight(
            weight,
            global_step=global_step,
            row_indices=row_indices,
            source_vocab_size=source_vocab_size,
            defer_device_apply=defer_device_apply,
        )
        if self.is_drafter_group_leader:
            logger.debug(
                "[speco target lm_head sync] replica=%s source=%s global_step=%s result=%s",
                self.replica_rank,
                name,
                global_step,
                result,
            )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_drafter_target_lm_head_row_indices(self):
        if (
            not self.enable_drafter
            or not self.in_drafter_train_group
            or self.trainer is None
        ):
            return None
        result = self.trainer.get_target_lm_head_row_indices()
        if result is not None and self.is_drafter_group_leader:
            logger.debug(
                "[speco target lm_head rows] replica=%s target_vocab=%s selected_rows=%s",
                self.replica_rank,
                result.get("source_vocab_size"),
                result.get("selected_rows"),
            )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def activate_drafter_training_model(self):
        result = {
            "activated": False,
            "elapsed_sec": 0.0,
            "reason": "",
        }
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result

        with _preserve_process_rng_state(self.device_name):
            start_ts = time.time()
            activation_ts = time.time()
            result["activated"] = bool(await self.trainer.activate_training_model())
            result["activation_elapsed_sec"] = time.time() - activation_ts
            release_ts = time.time()
            await self.trainer.release_training_memory_after_activation()
            result["release_elapsed_sec"] = time.time() - release_ts
            result["elapsed_sec"] = time.time() - start_ts
            result["reason"] = (
                "activated" if result["activated"] else "activation_failed"
            )
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def prewarm_drafter_training_model(self, worker_ids=None):
        target_worker_ids = {str(worker_id) for worker_id in (worker_ids or ())}
        result = {
            "activated": False,
            "elapsed_sec": 0.0,
            "reason": "",
            "rank": self.rank,
            "worker_id": str(self.rank),
            "worker_incarnation": self.worker_incarnation,
            "replica_rank": self.replica_rank,
            "training_group_ranks": list(getattr(self, "training_group_ranks", [])),
            "target_worker_ids": tuple(sorted(target_worker_ids)),
        }
        if target_worker_ids and str(self.rank) not in target_worker_ids:
            result["reason"] = "not_in_training_group"
            return result
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result
        if not self._use_replica_local_idle_training():
            result["reason"] = "not_rollout_idle_worker"
            return result

        start_ts = time.time()
        activation_error: BaseException | None = None
        with _preserve_process_rng_state(self.device_name):
            try:
                result["activated"] = bool(await self.trainer.activate_training_model())
            except Exception as error:  # noqa: BLE001
                activation_error = error
                result["activated"] = False
        result["elapsed_sec"] = time.time() - start_ts
        result["reason"] = "prewarmed" if result["activated"] else "activation_failed"

        if not result["activated"]:
            try:
                await self.trainer.cleanup_training(clear_data=False)
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "[BubbleTime] idle_prewarm_cleanup_failed: rank=%s error=%s",
                    self.rank,
                    repr(cleanup_error),
                )
            replica_local_oom = bool(
                activation_error is not None and _is_oom_error(activation_error)
            )
            result["replica_local_oom"] = replica_local_oom
            result["replica_local_unavailable"] = True
            logger.error(
                "[BubbleTime] idle_prewarm_failed: worker_id=%s rank=%s "
                "replica_rank=%s oom=%s elapsed_s=%.3f group=%s error=%s",
                self.rank,
                self.rank,
                self.replica_rank,
                replica_local_oom,
                result["elapsed_sec"],
                tuple(getattr(self, "training_group_ranks", [])),
                repr(activation_error) if activation_error is not None else "",
            )
            print(
                "[BubbleTime] idle_prewarm_failed: "
                f"worker_id={self.rank} rank={self.rank} "
                f"replica_rank={self.replica_rank} oom={replica_local_oom} "
                f"elapsed_s={result['elapsed_sec']:.3f} "
                f"group={tuple(getattr(self, 'training_group_ranks', []))} "
                f"target_workers={tuple(sorted(target_worker_ids))}",
                flush=True,
            )
            return result

        logger.warning(
            "[BubbleTime] idle_prewarm_succeeded: worker_id=%s rank=%s "
            "replica_rank=%s elapsed_s=%.3f group=%s target_workers=%s "
            "resident_until_training=True",
            self.rank,
            self.rank,
            self.replica_rank,
            result["elapsed_sec"],
            tuple(getattr(self, "training_group_ranks", [])),
            tuple(sorted(target_worker_ids)),
        )
        print(
            "[BubbleTime] idle_prewarm_succeeded: "
            f"worker_id={self.rank} rank={self.rank} "
            f"replica_rank={self.replica_rank} elapsed_s={result['elapsed_sec']:.3f} "
            f"group={tuple(getattr(self, 'training_group_ranks', []))} "
            f"target_workers={tuple(sorted(target_worker_ids))} "
            "resident_until_training=True",
            flush=True,
        )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_drafter_training_data_status(
        self,
        sample_last_n_steps: int = 2,
        require_full_batch: bool = False,
        target_version: int | None = None,
    ):
        if not self.enable_drafter:
            return {"available": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"available": False, "reason": "not_in_training_group"}
        status = self.trainer.get_training_data_status(
            sample_last_n_steps=sample_last_n_steps,
            require_full_batch=require_full_batch,
            target_version=target_version,
        )
        status.update(
            {
                "available": True,
                "replica_rank": self.replica_rank,
                "rank": self.rank,
                "worker_id": str(self.rank),
                "worker_incarnation": self.worker_incarnation,
            }
        )
        return status

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_drafter_training_resource_metadata(self):
        """Return the true drafter training mesh membership for Bubble Time."""

        result = {
            "available": False,
            "rank": self.rank,
            "worker_id": str(self.rank),
            "worker_incarnation": self.worker_incarnation,
            "replica_rank": self.replica_rank,
            "in_drafter_train_group": False,
            "training_group_ranks": [],
            "training_group_world_size": 0,
            "dp_group_ranks": [],
            "dp_group_world_size": 0,
            "full_collective_ranks": [],
            "sync_collective_ranks": [],
            "idle_collective_scope": "",
            "is_global_publish_leader": False,
            "reason": "",
        }
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        self._ensure_training_group_initialized()
        if not self.in_drafter_train_group:
            result["reason"] = "not_in_training_group"
            return result
        result.update(
            {
                "available": True,
                "in_drafter_train_group": True,
                "training_group_ranks": list(self.training_group_ranks),
                "training_group_world_size": int(self.training_group_world_size),
                "dp_group_ranks": list(self.dp_group_ranks),
                "dp_group_world_size": int(self.dp_group_world_size),
                "full_collective_ranks": list(
                    self.full_collective_ranks or self.training_group_ranks
                ),
                "sync_collective_ranks": list(
                    self.sync_collective_ranks or self.full_collective_ranks
                ),
                "idle_collective_scope": (
                    "replica_local"
                    if self._use_replica_local_idle_training()
                    else "full_collective"
                ),
                "is_global_publish_leader": bool(self.is_global_publish_leader),
                "reason": "ok",
            }
        )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def preflight_drafter_training(self, training_plan=None):
        preflight_started_ts = time.time()
        result = {
            "ready": False,
            "participating": False,
            "activated": False,
            "rank": self.rank,
            "worker_id": str(self.rank),
            "worker_incarnation": self.worker_incarnation,
            "reason": "",
        }
        self._prepared_training_plan_id = None
        self._prepared_training_data_version = None
        self._prepared_training_target_version = None
        self._prepared_training_activation_elapsed_sec = 0.0
        self._prepared_training_preflight_elapsed_sec = 0.0
        self._prepared_training_ready_ts = None
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result
        result["participating"] = True
        if not isinstance(training_plan, dict) or not training_plan.get("launch"):
            result["reason"] = "missing_or_inactive_training_plan"
            return result
        execution_strategy = str(training_plan.get("execution_strategy", "sync"))
        if execution_strategy not in {"sync", "rollout_idle_worker"}:
            result["reason"] = "unsupported_execution_strategy"
            return result
        target_worker_ids = {
            str(worker_id) for worker_id in training_plan.get("target_worker_ids", ())
        }
        if (
            execution_strategy == "rollout_idle_worker"
            and str(self.rank) not in target_worker_ids
        ):
            result["participating"] = False
            result["reason"] = "not_in_training_group"
            return result
        if training_plan.get("source_global_step") != self.last_global_step:
            result["reason"] = "stale_training_plan"
            return result

        if execution_strategy == "rollout_idle_worker":
            deadline_ts = training_plan.get("deadline_ts")
            now_ts = time.time()
            startup_reserve_sec = max(
                float(training_plan.get("idle_startup_reserve_sec", 0.0) or 0.0),
                0.0,
            )
            batch_estimate_sec = max(
                float(training_plan.get("idle_batch_estimate_sec", 0.0) or 0.0),
                0.0,
            )
            # ``deadline_ts`` is already the scheduler's latest safe training
            # boundary after reserving cleanup/tail time.  Requiring the tail
            # again here rejects otherwise valid windows by double-counting it.
            required_remaining_sec = startup_reserve_sec + batch_estimate_sec
            # Ray dispatch and Python scheduling can consume a few milliseconds
            # between the trainer-side admission decision and this worker-side
            # preflight.  Treat tiny underflows as still admissible; the tail
            # reserve plus cooperative reclaim remain the safety boundary.
            deadline_tolerance_sec = min(
                max(batch_estimate_sec * 0.10, 0.05),
                0.25,
            )
            if deadline_ts is not None and (
                float(deadline_ts) - now_ts
                < required_remaining_sec - deadline_tolerance_sec
            ):
                remaining_sec = float(deadline_ts) - now_ts
                result.update(
                    {
                        "reason": "plan_expired_before_preflight",
                        "deadline_ts": float(deadline_ts),
                        "remaining_sec": remaining_sec,
                        "required_remaining_sec": required_remaining_sec,
                    }
                )
                print(
                    "[BubbleTime] training_launch_stale: "
                    f"plan_id={training_plan.get('plan_id', '')} "
                    f"worker_id={self.rank} rank={self.rank} "
                    "reason=plan_expired_before_preflight "
                    f"remaining_s={remaining_sec:.4f} "
                    f"required_remaining_s={required_remaining_sec:.4f} "
                    f"deadline_ts={float(deadline_ts):.6f} now_ts={now_ts:.6f}",
                    flush=True,
                )
                return result

        snapshot = (training_plan.get("worker_snapshots") or {}).get(str(self.rank))
        if not isinstance(snapshot, dict):
            result["reason"] = "missing_worker_snapshot"
            return result
        if snapshot.get("worker_incarnation") != self.worker_incarnation:
            result["reason"] = "worker_restarted"
            return result
        # Bubble plans reserve a version-homogeneous sample snapshot below.
        # Appending a newer version between scheduling and preflight is safe
        # and must not invalidate the older plan.  A backwards version change
        # still indicates a reset/restart and remains fail-closed.
        snapshot_buffer_version = int(snapshot.get("buffer_version", -1))
        current_buffer_version = int(self.trainer.buffer_version)
        if (
            execution_strategy != "rollout_idle_worker"
            and snapshot_buffer_version != current_buffer_version
        ) or (
            execution_strategy == "rollout_idle_worker"
            and current_buffer_version < snapshot_buffer_version
        ):
            result["reason"] = "buffer_version_changed"
            return result
        required_target_version = training_plan.get("required_target_version")
        if (
            required_target_version is not None
            and not self.trainer.select_target_lm_head_version(
                int(required_target_version)
            )
        ):
            logger.warning(
                "[BubbleTime] preflight rejected: rank=%s reason=target_version_unavailable "
                "required=%s cached=%s",
                self.rank,
                required_target_version,
                sorted(getattr(self.trainer, "_target_lm_head_snapshots", {})),
            )
            result.update(
                {
                    "reason": "target_version_unavailable",
                    "required_target_version": required_target_version,
                }
            )
            print(
                "[BubbleTime] target_lm_head_cache_miss: "
                f"plan_id={training_plan.get('plan_id', '')} worker_id={self.rank} "
                f"rank={self.rank} required_target_version={required_target_version} "
                "reason=target_version_unavailable",
                flush=True,
            )
            return result
        current_target_version = getattr(
            self.trainer, "_target_lm_head_weight_step", None
        )
        if required_target_version is not None and int(required_target_version) != int(
            current_target_version if current_target_version is not None else -1
        ):
            print(
                "[BubbleTime] target_lm_head_cache_miss: "
                f"plan_id={training_plan.get('plan_id', '')} worker_id={self.rank} "
                f"rank={self.rank} required_target_version={required_target_version} "
                f"current_target_version={current_target_version} "
                "reason=target_version_mismatch",
                flush=True,
            )
            result["reason"] = "target_version_mismatch"
            return result
        data_status = self.trainer.get_training_data_status(
            sample_last_n_steps=int(training_plan.get("sample_last_n_steps", 2)),
            require_full_batch=bool(training_plan.get("require_full_batch", False)),
            target_version=(
                int(required_target_version)
                if required_target_version is not None
                else None
            ),
        )
        actual_data_version = data_status.get("data_version")
        planned_data_version = training_plan.get("data_version")
        snapshot_data_version = snapshot.get("data_version")
        if (
            actual_data_version != planned_data_version
            or actual_data_version != snapshot_data_version
        ):
            result.update(
                {
                    "reason": "data_version_changed",
                    "data_version": actual_data_version,
                    "target_version": current_target_version,
                }
            )
            return result
        if int(data_status["trainable_batches"]) < int(
            training_plan.get("min_batches", 1)
        ):
            result["reason"] = "insufficient_worker_data"
            return result

        if execution_strategy == "rollout_idle_worker":
            reservation = self.trainer.reserve_training_data(
                plan_id=str(training_plan.get("plan_id", "")),
                target_version=int(
                    required_target_version
                    if required_target_version is not None
                    else actual_data_version
                ),
                max_batches=(
                    int(training_plan.get("max_batches", 0))
                    * max(
                        int(training_plan.get("gradient_accumulation_steps", 1)),
                        1,
                    )
                ),
                require_full_batch=bool(training_plan.get("require_full_batch", False)),
                retain_replay_session=bool(
                    training_plan.get("retain_replay_session", False)
                ),
            )
            if int(reservation.get("reserved_samples", 0)) <= 0:
                logger.warning(
                    "[BubbleTime] preflight rejected: rank=%s plan_id=%s "
                    "reason=data_reservation_failed target_version=%s",
                    self.rank,
                    training_plan.get("plan_id", ""),
                    required_target_version,
                )
                result["reason"] = "data_reservation_failed"
                return result
            # This method contains no await between the directed status query
            # and reservation, so the Ray actor cannot interleave a collection
            # call here.  The reservation is the atomic plan-local ownership
            # boundary; later appends cannot change what this plan trains on.
            print(
                "[BubbleTime] training_replay_snapshot: "
                f"plan_id={training_plan.get('plan_id', '')} rank={self.rank} "
                f"samples={reservation.get('reserved_samples', 0)} "
                f"planned_optimizer_steps={training_plan.get('max_batches', 0)} "
                "mode=quota_cycle_replay",
                flush=True,
            )

        self._prepared_training_plan_id = str(training_plan.get("plan_id", ""))
        self._prepared_training_data_version = actual_data_version
        self._prepared_training_target_version = current_target_version
        self._drafter_reclaim_requested = False
        activation_started_ts = time.time()
        activation_error: BaseException | None = None
        with _preserve_process_rng_state(self.device_name):
            try:
                activated = bool(await self.trainer.activate_training_model())
            except Exception as error:  # noqa: BLE001
                activation_error = error
                activated = False
        result["activated"] = activated
        result["activation_elapsed_sec"] = time.time() - activation_started_ts
        if not activated:
            self.trainer.release_training_data_reservation(
                str(training_plan.get("plan_id", ""))
            )
            result["reason"] = "activation_failed"
            if (
                execution_strategy == "rollout_idle_worker"
                and self._use_replica_local_idle_training()
            ):
                replica_local_oom = bool(
                    activation_error is not None and _is_oom_error(activation_error)
                )
                result.update(
                    {
                        "replica_local_unavailable": True,
                        "replica_local_oom": replica_local_oom,
                        "training_group_ranks": list(self.training_group_ranks),
                        "sync_collective_ranks": list(
                            self.sync_collective_ranks or self.full_collective_ranks
                        ),
                    }
                )
                logger.error(
                    "[BubbleTime] replica_local_unavailable: plan_id=%s "
                    "worker_id=%s rank=%s replica_rank=%s reason=activation_failed "
                    "oom=%s activation_s=%.3f training_group_ranks=%s "
                    "sync_collective_ranks=%s error=%s",
                    training_plan.get("plan_id", ""),
                    self.rank,
                    self.rank,
                    self.replica_rank,
                    replica_local_oom,
                    result["activation_elapsed_sec"],
                    tuple(self.training_group_ranks),
                    tuple(self.sync_collective_ranks or self.full_collective_ranks),
                    repr(activation_error) if activation_error is not None else "",
                )
                print(
                    "[BubbleTime] replica_local_unavailable: "
                    f"plan_id={training_plan.get('plan_id', '')} "
                    f"worker_id={self.rank} rank={self.rank} "
                    f"replica_rank={self.replica_rank} reason=activation_failed "
                    f"oom={replica_local_oom} "
                    f"activation_s={result['activation_elapsed_sec']:.3f} "
                    f"training_group_ranks={tuple(self.training_group_ranks)} "
                    "sync_collective_ranks="
                    f"{tuple(self.sync_collective_ranks or self.full_collective_ranks)}",
                    flush=True,
                )
                if activation_error is not None and not replica_local_oom:
                    logger.error(
                        "[BubbleTime] replica-local activation failed with non-OOM exception",
                        exc_info=(
                            type(activation_error),
                            activation_error,
                            activation_error.__traceback__,
                        ),
                    )
            return result
        if execution_strategy == "rollout_idle_worker":
            deadline_ts = training_plan.get("deadline_ts")
            now_ts = time.time()
            # The scheduler has already moved ``deadline_ts`` earlier by the
            # tail reserve, so only one optimizer batch must fit at this point.
            required_remaining_sec = max(
                float(training_plan.get("idle_batch_estimate_sec", 0.0) or 0.0),
                0.0,
            )
            deadline_tolerance_sec = min(
                max(required_remaining_sec * 0.10, 0.05),
                0.25,
            )
            if deadline_ts is not None and (
                float(deadline_ts) - now_ts
                < required_remaining_sec - deadline_tolerance_sec
            ):
                remaining_sec = float(deadline_ts) - now_ts
                self.trainer.release_training_data_reservation(
                    str(training_plan.get("plan_id", ""))
                )
                await self.trainer.cleanup_training(clear_data=False)
                result.update(
                    {
                        "reason": "plan_expired_during_preflight",
                        "deadline_ts": float(deadline_ts),
                        "remaining_sec": remaining_sec,
                        "required_remaining_sec": required_remaining_sec,
                    }
                )
                print(
                    "[BubbleTime] training_launch_stale: "
                    f"plan_id={training_plan.get('plan_id', '')} "
                    f"worker_id={self.rank} rank={self.rank} "
                    "reason=plan_expired_during_preflight "
                    f"remaining_s={remaining_sec:.4f} "
                    f"required_remaining_s={required_remaining_sec:.4f} "
                    f"activation_s={result['activation_elapsed_sec']:.4f}",
                    flush=True,
                )
                return result
        ready_ts = time.time()
        preflight_elapsed_sec = ready_ts - preflight_started_ts
        self._prepared_training_activation_elapsed_sec = float(
            result["activation_elapsed_sec"]
        )
        self._prepared_training_preflight_elapsed_sec = preflight_elapsed_sec
        self._prepared_training_ready_ts = ready_ts
        result.update(
            {
                "ready": True,
                "reason": "ready",
                "buffer_version": int(self.trainer.buffer_version),
                "data_version": actual_data_version,
                "target_version": current_target_version,
                "preflight_elapsed_sec": preflight_elapsed_sec,
            }
        )
        logger.info(
            "[BubbleTime] training_preflight_ready: plan_id=%s worker_id=%s "
            "rank=%s activation_s=%.3f preflight_s=%.3f",
            training_plan.get("plan_id", ""),
            self.rank,
            self.rank,
            result["activation_elapsed_sec"],
            preflight_elapsed_sec,
        )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def abort_drafter_training_preflight(self, plan_id: str):
        was_prepared = self._prepared_training_plan_id == str(plan_id)
        self._prepared_training_plan_id = None
        self._prepared_training_data_version = None
        self._prepared_training_target_version = None
        self._prepared_training_activation_elapsed_sec = 0.0
        self._prepared_training_preflight_elapsed_sec = 0.0
        self._prepared_training_ready_ts = None
        if was_prepared and self.trainer is not None:
            self.trainer.release_training_data_reservation(str(plan_id))
            await self.trainer.cleanup_training(clear_data=False)
        return {"aborted": was_prepared, "rank": self.rank}

    def _should_keep_drafter_training_hot(
        self,
        training_plan: dict[str, object],
        *,
        execution_strategy: str,
        successful_steps: int,
    ) -> bool:
        if int(successful_steps) <= 0:
            return False
        if execution_strategy == "rollout_idle_worker":
            return True
        bootstrap_workers = {
            str(worker_id)
            for worker_id in cast(
                Any, training_plan.get("hot_bootstrap_worker_ids", ())
            )
        }
        return str(self.rank) in bootstrap_workers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def train_drafter(self, training_plan=None):
        result = {
            "trained": False,
            "triggered": False,
            "successful_steps": 0,
            "attempted_steps": 0,
            "successful_valid_tokens": 0,
            "elapsed_sec": 0.0,
            "reason": "",
            "worker_id": str(self.rank),
            "worker_incarnation": self.worker_incarnation,
            "is_publish_leader": self.is_global_publish_leader,
        }
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result
        execution_strategy = (
            str(training_plan.get("execution_strategy", "sync"))
            if isinstance(training_plan, dict)
            else "sync"
        )
        publish_leader = (
            self.is_drafter_group_leader
            if execution_strategy == "rollout_idle_worker"
            else self.is_global_publish_leader
        )
        result["is_publish_leader"] = publish_leader
        target_worker_ids = {
            str(worker_id)
            for worker_id in (
                training_plan.get("target_worker_ids", ())
                if isinstance(training_plan, dict)
                else ()
            )
        }
        if (
            execution_strategy == "rollout_idle_worker"
            and str(self.rank) not in target_worker_ids
        ):
            result["reason"] = "not_in_training_group"
            return result
        plan_id = (
            str(training_plan.get("plan_id", ""))
            if isinstance(training_plan, dict)
            else ""
        )
        if not plan_id or self._prepared_training_plan_id != plan_id:
            result["reason"] = "preflight_not_ready"
            return result
        prepared_data_version = self._prepared_training_data_version
        prepared_target_version = self._prepared_training_target_version
        prepared_activation_elapsed_sec = self._prepared_training_activation_elapsed_sec
        prepared_preflight_elapsed_sec = self._prepared_training_preflight_elapsed_sec
        prepared_ready_ts = self._prepared_training_ready_ts
        self._prepared_training_plan_id = None
        self._prepared_training_data_version = None
        self._prepared_training_target_version = None
        self._prepared_training_activation_elapsed_sec = 0.0
        self._prepared_training_preflight_elapsed_sec = 0.0
        self._prepared_training_ready_ts = None
        max_batches = max(int(training_plan.get("max_batches", 0)), 0)
        gradient_accumulation_steps = max(
            int(training_plan.get("gradient_accumulation_steps", 1)), 1
        )
        idle_batch_estimate_sec = max(
            float(training_plan.get("idle_batch_estimate_sec", 0.0) or 0.0),
            0.0,
        )
        prepare_publish = bool(training_plan.get("publish_after_success", False))
        snapshot = (training_plan.get("worker_snapshots") or {})[str(self.rank)]
        buffer_size_before = int(snapshot.get("trainable_samples", 0))
        result.update(
            {
                "source_global_step": int(training_plan["source_global_step"]),
                "plan_id": plan_id,
                "data_version": prepared_data_version,
                "target_version": prepared_target_version,
                "execution_strategy": str(training_plan["execution_strategy"]),
                "buffer_size_before": buffer_size_before,
                "buffer_size_after": buffer_size_before,
                "optimizer_step": int(self.trainer.optimizer_steps_total),
                "publish_snapshot_cached": 0,
                "activation_elapsed_sec": prepared_activation_elapsed_sec,
                "preflight_elapsed_sec": prepared_preflight_elapsed_sec,
                "first_batch_started": 0,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "planned_valid_tokens": int(
                    training_plan.get("planned_valid_tokens", 0) or 0
                ),
                "stop_reason": "",
            }
        )

        with _preserve_process_rng_state(self.device_name):
            result["triggered"] = True
            start_ts = time.time()
            self.trainer.clear_pending_publish_state_dict()
            try:
                train_loop_ts = time.time()
                self.trainer.reset_training_metrics()
                for batch_index in range(max_batches):
                    deadline_ts = training_plan.get("deadline_ts")
                    now_ts = time.time()
                    if deadline_ts is not None and now_ts >= float(deadline_ts):
                        result["reason"] = "deadline_reached"
                        result["stop_reason"] = "deadline_reached"
                        if prepared_ready_ts is not None:
                            result["preflight_to_stop_sec"] = max(
                                now_ts - prepared_ready_ts, 0.0
                            )
                        log_prefix = (
                            "training_not_started"
                            if not result["first_batch_started"]
                            else "training_stopped"
                        )
                        logger.warning(
                            "[BubbleTime] %s: plan_id=%s "
                            "worker_id=%s rank=%s reason=deadline_reached "
                            "now_ts=%.6f deadline_ts=%.6f remaining_s=%.4f "
                            "reclaim_requested=%s",
                            log_prefix,
                            plan_id,
                            self.rank,
                            self.rank,
                            now_ts,
                            float(deadline_ts),
                            float(deadline_ts) - now_ts,
                            self._drafter_reclaim_requested,
                        )
                        print(
                            f"[BubbleTime] {log_prefix}: "
                            f"plan_id={plan_id} worker_id={self.rank} rank={self.rank} "
                            "reason=deadline_reached "
                            f"now_ts={now_ts:.6f} deadline_ts={float(deadline_ts):.6f} "
                            f"remaining_s={float(deadline_ts) - now_ts:.4f} "
                            f"reclaim_requested={self._drafter_reclaim_requested}",
                            flush=True,
                        )
                        break
                    if (
                        deadline_ts is not None
                        and idle_batch_estimate_sec > 0.0
                        and now_ts + idle_batch_estimate_sec > float(deadline_ts)
                    ):
                        result["reason"] = "next_batch_budget_too_small"
                        result["stop_reason"] = "deadline_reached"
                        remaining_s = float(deadline_ts) - now_ts
                        if (
                            prepared_ready_ts is not None
                            and not result["first_batch_started"]
                        ):
                            result["preflight_to_stop_sec"] = max(
                                now_ts - prepared_ready_ts, 0.0
                            )
                        log_prefix = (
                            "training_not_started"
                            if not result["first_batch_started"]
                            else "training_stopped"
                        )
                        logger.warning(
                            "[BubbleTime] %s: plan_id=%s worker_id=%s rank=%s "
                            "reason=next_batch_budget_too_small batch_index=%s "
                            "successful_steps=%s now_ts=%.6f deadline_ts=%.6f "
                            "remaining_s=%.4f batch_estimate_s=%.4f "
                            "reclaim_requested=%s",
                            log_prefix,
                            plan_id,
                            self.rank,
                            self.rank,
                            batch_index,
                            result["successful_steps"],
                            now_ts,
                            float(deadline_ts),
                            remaining_s,
                            idle_batch_estimate_sec,
                            self._drafter_reclaim_requested,
                        )
                        print(
                            f"[BubbleTime] {log_prefix}: "
                            f"plan_id={plan_id} worker_id={self.rank} rank={self.rank} "
                            "reason=next_batch_budget_too_small "
                            f"batch_index={batch_index} "
                            f"successful_steps={result['successful_steps']} "
                            f"now_ts={now_ts:.6f} "
                            f"deadline_ts={float(deadline_ts):.6f} "
                            f"remaining_s={remaining_s:.4f} "
                            f"batch_estimate_s={idle_batch_estimate_sec:.4f} "
                            f"reclaim_requested={self._drafter_reclaim_requested}",
                            flush=True,
                        )
                        break
                    if self._drafter_reclaim_requested:
                        result["reason"] = "reclaim_requested"
                        result["stop_reason"] = "reclaim_requested"
                        if prepared_ready_ts is not None:
                            result["preflight_to_stop_sec"] = max(
                                now_ts - prepared_ready_ts, 0.0
                            )
                        log_prefix = (
                            "training_not_started"
                            if not result["first_batch_started"]
                            else "training_stopped"
                        )
                        logger.warning(
                            "[BubbleTime] %s: plan_id=%s "
                            "worker_id=%s rank=%s reason=reclaim_requested "
                            "now_ts=%.6f deadline_ts=%s reclaim_requested=True",
                            log_prefix,
                            plan_id,
                            self.rank,
                            self.rank,
                            now_ts,
                            deadline_ts,
                        )
                        print(
                            f"[BubbleTime] {log_prefix}: "
                            f"plan_id={plan_id} worker_id={self.rank} rank={self.rank} "
                            "reason=reclaim_requested "
                            f"now_ts={now_ts:.6f} deadline_ts={deadline_ts} "
                            f"activation_s={prepared_activation_elapsed_sec:.3f} "
                            f"preflight_s={prepared_preflight_elapsed_sec:.3f} "
                            "preflight_to_stop_s="
                            f"{result.get('preflight_to_stop_sec', 0.0):.3f} "
                            "reclaim_requested=True",
                            flush=True,
                        )
                        break
                    if not result["first_batch_started"]:
                        result["first_batch_started"] = 1
                        if prepared_ready_ts is not None:
                            result["preflight_to_first_batch_sec"] = max(
                                now_ts - prepared_ready_ts, 0.0
                            )
                        logger.warning(
                            "[BubbleTime] training_first_batch_start: plan_id=%s "
                            "worker_id=%s rank=%s activation_s=%.3f "
                            "preflight_s=%.3f preflight_to_first_batch_s=%.3f",
                            plan_id,
                            self.rank,
                            self.rank,
                            prepared_activation_elapsed_sec,
                            prepared_preflight_elapsed_sec,
                            float(
                                result.get("preflight_to_first_batch_sec", 0.0) or 0.0
                            ),
                        )
                        print(
                            "[BubbleTime] training_first_batch_start: "
                            f"plan_id={plan_id} worker_id={self.rank} rank={self.rank} "
                            f"activation_s={prepared_activation_elapsed_sec:.3f} "
                            f"preflight_s={prepared_preflight_elapsed_sec:.3f} "
                            "preflight_to_first_batch_s="
                            f"{result.get('preflight_to_first_batch_sec', 0.0):.3f}",
                            flush=True,
                        )
                    result["attempted_steps"] += 1
                    step_error: BaseException | None = None
                    accumulation_stop_reason: str | None = None

                    def can_start_micro_batch(micro_index: int) -> bool:
                        nonlocal accumulation_stop_reason
                        if self._drafter_reclaim_requested:
                            accumulation_stop_reason = "reclaim_requested"
                            return False
                        if deadline_ts is None:
                            return True
                        remaining_micro_batches = max(
                            gradient_accumulation_steps - int(micro_index), 1
                        )
                        micro_batch_estimate_sec = idle_batch_estimate_sec / max(
                            gradient_accumulation_steps, 1
                        )
                        if (
                            float(deadline_ts) - time.time()
                            < micro_batch_estimate_sec * remaining_micro_batches
                        ):
                            accumulation_stop_reason = "deadline_reached"
                            return False
                        return True

                    try:
                        step_ok = await self.trainer.training_accumulation_step(
                            self.last_global_step,
                            gradient_accumulation_steps,
                            can_start_micro_batch=can_start_micro_batch,
                        )
                    except Exception as error:  # noqa: BLE001
                        step_error = error
                        step_ok = False
                    if step_ok:
                        result["successful_steps"] += 1
                        result["successful_valid_tokens"] += int(
                            getattr(self.trainer, "_last_optimizer_valid_tokens", 0)
                            or 0
                        )
                    else:
                        result["reason"] = (
                            accumulation_stop_reason or "training_step_returned_false"
                        )
                        result["stop_reason"] = result["reason"]
                        if (
                            execution_strategy == "rollout_idle_worker"
                            and self._use_replica_local_idle_training()
                            and step_error is not None
                            and _is_oom_error(step_error)
                        ):
                            result.update(
                                {
                                    "reason": "replica_local_oom",
                                    "stop_reason": "replica_local_oom",
                                    "replica_local_unavailable": True,
                                    "replica_local_oom": True,
                                    "training_group_ranks": list(
                                        self.training_group_ranks
                                    ),
                                    "sync_collective_ranks": list(
                                        self.sync_collective_ranks
                                        or self.full_collective_ranks
                                    ),
                                }
                            )
                        now_after_step_ts = time.time()
                        logger.warning(
                            "[BubbleTime] training_stopped: plan_id=%s "
                            "worker_id=%s rank=%s reason=%s "
                            "batch_index=%s attempted_steps=%s successful_steps=%s "
                            "now_ts=%.6f deadline_ts=%s reclaim_requested=%s error=%s",
                            plan_id,
                            self.rank,
                            self.rank,
                            result["reason"],
                            batch_index,
                            result["attempted_steps"],
                            result["successful_steps"],
                            now_after_step_ts,
                            deadline_ts,
                            self._drafter_reclaim_requested,
                            repr(step_error) if step_error is not None else "",
                        )
                        print(
                            "[BubbleTime] training_stopped: "
                            f"plan_id={plan_id} worker_id={self.rank} rank={self.rank} "
                            f"reason={result['reason']} "
                            f"batch_index={batch_index} "
                            f"attempted_steps={result['attempted_steps']} "
                            f"successful_steps={result['successful_steps']} "
                            f"now_ts={now_after_step_ts:.6f} deadline_ts={deadline_ts} "
                            f"reclaim_requested={self._drafter_reclaim_requested}",
                            flush=True,
                        )
                        break
                if (
                    result["successful_steps"] > 0
                    and not result.get("stop_reason")
                    and result["successful_steps"] >= max_batches
                ):
                    result["stop_reason"] = "max_batches_reached"
                result["training_loop_elapsed_sec"] = time.time() - train_loop_ts
                result.update(self.trainer.get_training_metrics())
                if result["successful_steps"] > 0:
                    publish_quota_completed = bool(
                        result["successful_steps"] >= max_batches
                        and not result.get("error")
                    )
                    if prepare_publish and publish_quota_completed:
                        snapshot_ts = time.time()
                        cached = self.trainer.prepare_model_state_dict_for_publish(
                            self.last_global_step
                        )
                        result["publish_snapshot_cached"] = int(cached)
                        result["publish_snapshot_elapsed_sec"] = (
                            time.time() - snapshot_ts
                        )
                        if hasattr(self.trainer, "record_training_timing"):
                            self.trainer.record_training_timing(
                                "timing_s/drafter_publish_snapshot",
                                result["publish_snapshot_elapsed_sec"],
                            )
                    else:
                        # A final Bubble or forced-completion plan may still be
                        # interrupted. Never cache a partially trained snapshot;
                        # the scheduler keeps its quota debt and retries it.
                        self.trainer.clear_pending_publish_state_dict()
                else:
                    self.trainer.clear_pending_publish_state_dict()
                result.update(self.trainer.get_training_metrics())
            finally:
                cleanup_ts = time.time()
                finalize_reservation = getattr(
                    self.trainer, "finalize_training_data_reservation", None
                )
                completed_plan = bool(
                    result.get("successful_steps", 0) >= max_batches
                    and not result.get("error")
                )
                retain_replay_session = bool(
                    training_plan.get("retain_replay_session", False)
                    or not completed_plan
                )
                replay_consumed = (
                    int(
                        finalize_reservation(
                            plan_id,
                            consume=not retain_replay_session,
                        )
                    )
                    if callable(finalize_reservation)
                    else 0
                )
                result["replay_session_retained"] = int(retain_replay_session)
                result["replay_consumed_samples"] = replay_consumed
                if replay_consumed:
                    print(
                        "[BubbleTime] training_replay_finalized: "
                        f"plan_id={plan_id} rank={self.rank} "
                        f"unique_samples={replay_consumed}",
                        flush=True,
                    )
                self.trainer.release_training_data_reservation(plan_id)
                # Keep only a canonical state that has actually advanced.
                # Expired, aborted, or zero-step attempts are cleaned up by
                # their respective paths and never leave extra hot replicas.
                keep_hot = self._should_keep_drafter_training_hot(
                    training_plan,
                    execution_strategy=execution_strategy,
                    successful_steps=int(result.get("successful_steps", 0) or 0),
                )
                await self.trainer.cleanup_training(
                    clear_data=False,
                    keep_hot=keep_hot,
                )
                result["training_residency_retained"] = int(keep_hot)
                if keep_hot and execution_strategy == "sync":
                    print(
                        "[BubbleTime] full_collective_worker_kept_hot: "
                        f"rank={self.rank} plan_id={plan_id} "
                        f"source_step={training_plan.get('source_global_step')} "
                        "reason=authoritative_publish_replica_bootstrap",
                        flush=True,
                    )
                result["cleanup_elapsed_sec"] = time.time() - cleanup_ts

            result["trained"] = result["successful_steps"] > 0
            if not result["trained"] and not result.get("stop_reason"):
                result["stop_reason"] = result["reason"] or "no_trainable_batch"
            result["reason"] = (
                "trained"
                if result["trained"]
                else result["reason"] or "no_trainable_batch"
            )
            if result["trained"]:
                self.last_trained_step = self.last_global_step
                self._last_trained_execution_strategy = execution_strategy
                self._last_trained_target_worker_ids = tuple(
                    sorted(target_worker_ids, key=str)
                )
            data_status_after = self.trainer.get_training_data_status(
                sample_last_n_steps=int(training_plan.get("sample_last_n_steps", 2)),
                require_full_batch=bool(training_plan.get("require_full_batch", False)),
            )
            result["buffer_size_after"] = int(data_status_after["trainable_samples"])
            result["optimizer_step"] = int(self.trainer.optimizer_steps_total)
            result["elapsed_sec"] = time.time() - start_ts
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def request_drafter_training_reclaim(self, worker_ids=None):
        target_worker_ids = {str(worker_id) for worker_id in (worker_ids or ())}
        if target_worker_ids and str(self.rank) not in target_worker_ids:
            return {"rank": self.rank, "worker_id": str(self.rank), "requested": False}
        self._drafter_reclaim_requested = True
        logger.warning(
            "[BubbleTime] reclaim_requested: worker_id=%s rank=%s target_workers=%s",
            self.rank,
            self.rank,
            tuple(sorted(target_worker_ids)),
        )
        print(
            "[BubbleTime] reclaim_requested: "
            f"worker_id={self.rank} rank={self.rank} "
            f"target_workers={tuple(sorted(target_worker_ids))}",
            flush=True,
        )
        return {"rank": self.rank, "worker_id": str(self.rank), "requested": True}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def maybe_publish(self):
        if (
            not self.enable_drafter
            or not self.in_drafter_train_group
            or self.trainer is None
        ):
            return None
        if self.last_global_step is None:
            return None
        if self.last_trained_step != self.last_global_step:
            return None
        has_snapshot, weights = self.trainer.pop_model_state_dict_for_publish(
            self.last_global_step
        )
        if not has_snapshot or not weights:
            logger.debug(
                "[SpecoWorker replica=%s rank=%s] missing cached publish snapshot at step %s; skip publish.",
                self.replica_rank,
                self.rank,
                self.last_global_step,
            )
            return None
        publish_leader = self.is_global_publish_leader
        if self._last_trained_execution_strategy == "rollout_idle_worker":
            publish_leader = self.is_drafter_group_leader
        if not publish_leader:
            release_draft_weights_payload(weights)
            return None

        print(
            "[BubbleTime] publish_snapshot_ready: "
            f"rank={self.rank} replica_rank={self.replica_rank} "
            f"step={self.last_global_step} "
            f"execution_strategy={self._last_trained_execution_strategy or 'sync'} "
            f"target_worker_ids={self._last_trained_target_worker_ids}",
            flush=True,
        )
        try:
            weights_ref = ray.put(weights)
        finally:
            reclaim = release_draft_weights_payload(weights)
            logger.warning(
                "[speco publish reclaim] role=producer global_steps=%s "
                "num_weights=%s payload_cleared=%s allocator=%s action=%s "
                "heap_trimmed=%s elapsed_sec=%.3f memory_before=(%s) "
                "memory_after=(%s)",
                self.last_global_step,
                reclaim["num_weights"],
                reclaim["payload_cleared"],
                reclaim.get("allocator"),
                reclaim.get("reclaim_action"),
                reclaim.get("heap_trimmed"),
                float(reclaim.get("elapsed_sec", 0.0) or 0.0),
                reclaim.get("memory_before"),
                reclaim.get("memory_after"),
            )

        return {"weights_ref": weights_ref}
