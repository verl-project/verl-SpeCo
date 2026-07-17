"""External SPECO worker.

Adapted from the current in-tree
``verl/workers/engine_workers.py::DrafterWorker``. This module keeps drafter
worker behavior in ``verl_speco`` while importing upstream ``verl`` as a
dependency.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
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


def _is_ray_object_ref(value) -> bool:
    object_ref_type = getattr(ray, "ObjectRef", ())
    return bool(object_ref_type) and isinstance(value, object_ref_type)


def _resolve_ray_object_ref(value):
    if _is_ray_object_ref(value):
        return ray.get(value)
    return value


def _resolve_hidden_state_chunks(chunks, expected_rows: int | None = None):
    if not chunks:
        return None
    resolved_cache = {}
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
        if cache_key not in resolved_cache:
            resolved_cache[cache_key] = _resolve_ray_object_ref(ref)
        tensor = resolved_cache[cache_key]
        if not torch.is_tensor(tensor):
            continue
        start = int(chunk.get("chunk_start", 0) or 0)
        length = int(chunk.get("chunk_length", 0) or 0)
        if length <= 0:
            continue
        part = tensor[start : start + length]
        row_indices = chunk.get("chunk_row_indices")
        if torch.is_tensor(row_indices):
            row_indices = row_indices.detach().cpu().long().reshape(-1)
        elif isinstance(row_indices, (list, tuple)):
            row_indices = torch.tensor([int(idx) for idx in row_indices], dtype=torch.long)
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
        full_rows = max(full_rows, int(row_indices.max().item()) + 1 if int(row_indices.numel()) > 0 else 0)
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


def _dispatch_nd_compute(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
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
    return [output[global_rank] for global_rank in range(worker_group.world_size) if collect_mask[global_rank]]


def _dispatch_lazy_compute(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    dp_size = max(dp_rank_mapping) + 1
    return _dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


def _collect_lazy_compute(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    return _collect_nd_compute(worker_group._collect_info[mesh_name], worker_group, *args, **kwargs)


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
                logger.warning("Failed to restore %s RNG state after SPECO training.", device_name)


class SpecoWorker(Worker):
    """Standalone SPECO drafter worker.

    The worker receives CPU rollout features from the PPO trainer and trains the
    drafter model periodically according to global RL steps.
    """

    def __init__(self, config: DictConfig, role: str = "speco", device_name: Optional[str] = None, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        if device_name is None:
            raise ValueError("SpecoWorker requires an explicit device_name from the trainer initialization path")
        self.device_name = str(device_name).lower()
        self.trainer = None
        self.feature_writer = None
        self.feature_writer_path = None
        self.last_global_step = None
        self.last_trained_step = None
        self.training_process_group = None
        self.dp_process_group = None
        self.training_group_ranks = []
        self.training_group_world_size = 1
        self.dp_group_ranks = []
        self.dp_group_world_size = 1
        self.num_rollout_replicas = 1
        self.training_device_mesh = None
        self._process_group_initialized = False
        self._training_group_initialized = False

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
            self.config.rollout.drafter.enable and self.config.rollout.drafter.enable_drafter_training
        )
        self.training_interval_steps = int(self.config.rollout.drafter.training.get("training_interval_steps", 1))
        self.publish_interval_steps = int(self.config.rollout.drafter.training.get("publish_interval_steps", 0))
        self.train_steps_per_trigger = int(self.config.rollout.drafter.training.get("step", 100))

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
            raise RuntimeError(f"SpecoWorker rank mismatch: worker_rank={self.rank}, dist_rank={dist.get_rank()}")

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
            rollout_layout.replica_training_ranks[0][0] if rollout_layout.replica_training_ranks else None
        )
        self.is_global_publish_leader = self.rank == self.global_publish_leader_rank
        self.training_device_mesh = build_drafter_training_device_mesh(self.device_name, rollout_layout)
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
            self.training_group_ranks = list(rollout_layout.replica_training_ranks[mesh_dp_rank])
            self.dp_group_ranks = [
                rollout_layout.replica_training_ranks[replica_rank][mesh_sp_rank]
                for replica_rank in range(rollout_layout.num_replicas)
            ]
            self.training_group_world_size = self.training_device_mesh["sp"].size()
            self.dp_group_world_size = self.training_device_mesh["dp"].size()
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

        from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend
        from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
        from verl_speco.trainer.base_trainer import DrafterBaseTrainer

        algo = str(self.config.rollout.drafter.speculative_algorithm).upper()
        if algo == "EAGLE3":
            trainer_backend = Eagle3TrainerBackend(self.config, self.config.model)
        elif algo in ("EAGLE1", "EAGLE2"):
            from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend

            trainer_backend = Eagle1TrainerBackend(self.config, self.config.model)
        elif algo == "DFLASH":
            trainer_backend = DFlashTrainerBackend(self.config, self.config.model)
        elif algo == "DSPARK":
            from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

            trainer_backend = DSparkTrainerBackend(self.config, self.config.model)
        elif algo == "DOMINO":
            from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend

            trainer_backend = DominoTrainerBackend(self.config, self.config.model)
        else:
            raise ValueError(
                "Unsupported drafter algorithm "
                f"{self.config.rollout.drafter.speculative_algorithm!r}; "
                "supported algorithms are EAGLE1, EAGLE2, EAGLE3, DFLASH, DSPARK and DOMINO"
            )

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
    ):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return
        if self._drafter_training_mode() == "collect_only":
            self._write_rollout_feature_sample(batch, hidden_states, target_logprobs)
            return
        self.trainer.collect_online_data(batch, hidden_states, target_logprobs)

    def _drafter_training_mode(self) -> str:
        return str(self.config.rollout.drafter.training.get("mode", "online") or "online").strip().lower()

    def _get_feature_writer(self) -> Optional[TorchShardFeatureStore]:
        feature_store_cfg = self.config.rollout.drafter.training.get("feature_store", None)
        if feature_store_cfg is None:
            return None
        path = _config_str(feature_store_cfg.get("path", None))
        if not path:
            return None
        if self.feature_writer is not None and self.feature_writer_path == path:
            return self.feature_writer
        model_cfg = self.config.get("model", None)
        target_model_path = _config_str(model_cfg.get("path", None)) if model_cfg is not None else ""
        self.feature_writer = TorchShardFeatureStore(
            path,
            max_samples_per_shard=int(feature_store_cfg.get("max_samples_per_shard", 1024)),
            strict_schema=bool(feature_store_cfg.get("strict_schema", True)),
            metadata={
                "algorithm": str(self.config.rollout.drafter.speculative_algorithm).upper(),
                "target_model_path": target_model_path,
                "drafter_model_path": _config_str(self.config.rollout.drafter.get("model_path", None)),
                "source": "rl_collect_only",
            },
            shard_prefix=f"rank{int(self.rank):05d}_pid{int(os.getpid())}",
        )
        self.feature_writer_path = path
        return self.feature_writer

    def _build_rollout_loss_mask(self, batch: dict, input_ids: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(batch.get("loss_mask")):
            return batch["loss_mask"].detach().cpu().float().reshape(-1)
        ids = input_ids.detach().cpu().reshape(-1)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        prompts = batch.get("prompts")
        responses = batch.get("responses")
        if torch.is_tensor(prompts) and torch.is_tensor(responses):
            prompt_len = int(prompts.reshape(-1).numel())
            response_ids = responses.detach().cpu().reshape(-1)
            model_cfg = self.config.get("model", None)
            pad_token_id = int(model_cfg.get("pad_token_id", 0) or 0) if model_cfg is not None else 0
            max_response = max(0, min(int(response_ids.numel()), int(ids.numel()) - prompt_len))
            if max_response > 0:
                loss_mask[prompt_len : prompt_len + max_response] = (response_ids[:max_response] != pad_token_id).float()
        else:
            loss_mask[:] = 1.0
        return loss_mask

    def _write_rollout_feature_sample(
        self,
        batch: dict,
        hidden_states: torch.Tensor,
        target_logprobs: Optional[torch.Tensor],
    ) -> None:
        writer = self._get_feature_writer()
        if writer is None:
            logger.warning(
                "[SpecoWorker rank=%s] training.mode=collect_only but feature_store.path is empty; drop sample",
                self.rank,
            )
            return
        full_input_ids = batch["input_ids"].detach().cpu().reshape(-1)
        full_loss_mask = self._build_rollout_loss_mask(batch, full_input_ids)
        hidden_states = hidden_states.detach().cpu()
        hidden_rows = int(hidden_states.size(1) if hidden_states.dim() == 3 and hidden_states.size(0) == 1 else hidden_states.size(0))
        hidden_positions = batch.get("hidden_positions")
        if torch.is_tensor(hidden_positions):
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
        target_model_path = _config_str(model_cfg.get("path", None)) if model_cfg is not None else ""
        algorithm = str(self.config.rollout.drafter.speculative_algorithm).upper()
        dspark_l1_enabled = (
            algorithm == "DSPARK"
            and float(self.config.rollout.drafter.training.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0
        )
        default_hidden_layout = (
            "dflash_aux_plus_last"
            if dspark_l1_enabled
            else "dflash_aux"
            if algorithm in {"DFLASH", "DSPARK"}
            else "eagle3_aux_plus_last"
        )
        metadata = {
            "source": batch.get("hidden_target_logprobs_source", "rl_rollout"),
            "global_step": batch.get("global_step", self.last_global_step),
            "target_model_path": target_model_path,
            "drafter_model_path": _config_str(self.config.rollout.drafter.get("model_path", None)),
            "hidden_states_layout": batch.get("hidden_states_layout") or default_hidden_layout,
            "target_layer_ids": batch.get("target_layer_ids"),
            "use_logits": bool(self.config.rollout.drafter.training.get("use_logits", False)),
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
        sample = DraftFeatureSample(
            algorithm=str(self.config.rollout.drafter.speculative_algorithm).upper(),
            input_ids=input_ids,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            target_logprobs=target_logprobs,
            position_ids=position_ids,
            metadata=metadata,
        )
        writer.write_many([sample])

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
        position_end = min(max(position_end, position_start), position_start + int(target.size(0)))

        desired_start = int(feature_start) + 1
        desired_end = desired_start + max(int(train_rows), 0)
        slice_start = min(max(desired_start - position_start, 0), int(target.size(0)))
        slice_end = min(max(desired_end - position_start, slice_start), int(position_end - position_start))
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
            if int(positions.numel()) == hidden_rows and bool(torch.all(positions[1:] == positions[:-1] + 1).item()):
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
        if self._drafter_training_mode() != "collect_only" or self.feature_writer is None:
            return
        feature_store_cfg = self.config.rollout.drafter.training.get("feature_store", {})
        flush_interval = int(feature_store_cfg.get("flush_interval_steps", 1))
        self.feature_writer.flush_on_step(self.last_global_step, flush_interval)

    @register(dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH))
    def collect_rollout_features(self, samples: list[dict]):
        if not samples:
            return
        for sample in samples:
            if not sample:
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
                        expected_rows = int(hidden_positions.numel())
                    hidden = _resolve_hidden_state_chunks(hidden_chunks, expected_rows=expected_rows)
                else:
                    hidden = _resolve_ray_object_ref(sample.get("hidden_states_ref"))
            target_logprobs = sample.get("target_logprobs")
            if target_logprobs is None:
                target_logprobs = _resolve_ray_object_ref(sample.get("target_logprobs_ref"))
            if hidden is None:
                continue
            self._store_rollout_sample(
                batch=batch,
                hidden_states=hidden,
                target_logprobs=target_logprobs,
            )
        self._flush_rollout_features_for_step()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_global_step(self, global_step: int):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return
        if global_step is None:
            return
        if self.last_global_step == global_step:
            return
        self.last_global_step = global_step
        self.trainer.clear_pending_publish_state_dict()
        self.trainer.increment_rl_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, global_step: int, wait: bool = True):
        if not self.enable_drafter:
            return {"saved": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"saved": False, "reason": "not_in_training_group"}
        if global_step is None:
            return {"saved": False, "reason": "missing_global_step"}
        result = self.trainer.save_checkpoint(int(global_step), wait=wait)
        if self.is_drafter_group_leader:
            logger.debug(
                "[speco checkpoint] replica=%s global_step=%s result=%s",
                self.replica_rank,
                global_step,
                result,
            )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_checkpoint(self):
        if not self.enable_drafter:
            return {"waited": False, "completed": True, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"waited": False, "completed": True, "reason": "not_in_training_group"}
        result = self.trainer.wait_checkpoint()
        if self.is_drafter_group_leader:
            logger.debug("[speco checkpoint wait] replica=%s result=%s", self.replica_rank, result)
        return result

    @register(dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_TARGET_SYNC_MESH))
    def sync_target_lm_head_weight(self, payload: Optional[dict], global_step: Optional[int] = None):
        if not self.enable_drafter:
            return {"accepted": False, "applied": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"accepted": False, "applied": False, "reason": "not_in_training_group"}
        if not payload:
            return {"accepted": False, "applied": False, "reason": "missing_payload"}

        weight = payload.get("weight")
        row_indices = payload.get("row_indices")
        source_vocab_size = payload.get("source_vocab_size")
        name = payload.get("name")
        result = self.trainer.sync_target_lm_head_weight(
            weight,
            global_step=global_step,
            row_indices=row_indices,
            source_vocab_size=source_vocab_size,
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
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
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
            result["reason"] = "activated" if result["activated"] else "activation_failed"
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def train_drafter(self):
        result = {
            "trained": False,
            "triggered": False,
            "successful_steps": 0,
            "attempted_steps": 0,
            "elapsed_sec": 0.0,
            "reason": "",
        }
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result
        if self.last_global_step is None:
            result["reason"] = "missing_global_step"
            return result
        if self.training_interval_steps <= 0:
            result["reason"] = "invalid_interval"
            return result
        if self.last_global_step % self.training_interval_steps != 0:
            result["reason"] = "interval_not_reached"
            return result

        with _preserve_process_rng_state(self.device_name):
            result["triggered"] = True
            start_ts = time.time()
            self.trainer.clear_pending_publish_state_dict()
            activation_ts = time.time()
            success = await self.trainer.activate_training_model()
            result["activation_elapsed_sec"] = time.time() - activation_ts
            if not success:
                logger.error(
                    "[SpecoWorker replica=%s] failed to activate trainer at step %s",
                    self.replica_rank,
                    self.last_global_step,
                )
                self.trainer.clear_pending_publish_state_dict()
                cleanup_ts = time.time()
                await self.trainer.cleanup_training(clear_data=False)
                result["cleanup_elapsed_sec"] = time.time() - cleanup_ts
                result["reason"] = "activation_failed"
                result["elapsed_sec"] = time.time() - start_ts
                return result

            try:
                train_loop_ts = time.time()
                self.trainer.reset_training_metrics()
                for _ in range(self.train_steps_per_trigger):
                    result["attempted_steps"] += 1
                    step_ok = await self.trainer.training_step(self.last_global_step)
                    if step_ok:
                        result["successful_steps"] += 1
                result["training_loop_elapsed_sec"] = time.time() - train_loop_ts
                result.update(self.trainer.get_training_metrics())
                if result["successful_steps"] > 0:
                    should_prepare_publish = (
                        self.publish_interval_steps <= 0 or self.last_global_step % self.publish_interval_steps == 0
                    )
                    if should_prepare_publish:
                        snapshot_ts = time.time()
                        cached = self.trainer.prepare_model_state_dict_for_publish(self.last_global_step)
                        result["publish_snapshot_cached"] = int(cached)
                        result["publish_snapshot_elapsed_sec"] = time.time() - snapshot_ts
                        if hasattr(self.trainer, "record_training_timing"):
                            self.trainer.record_training_timing(
                                "timing_s/drafter_publish_snapshot",
                                result["publish_snapshot_elapsed_sec"],
                            )
                    else:
                        self.trainer.clear_pending_publish_state_dict()
                else:
                    self.trainer.clear_pending_publish_state_dict()
                result.update(self.trainer.get_training_metrics())
            finally:
                cleanup_ts = time.time()
                await self.trainer.cleanup_training(clear_data=result["successful_steps"] > 0)
                result["cleanup_elapsed_sec"] = time.time() - cleanup_ts

            result["trained"] = result["successful_steps"] > 0
            result["reason"] = "trained" if result["trained"] else "no_trainable_batch"
            if result["trained"]:
                self.last_trained_step = self.last_global_step
            result["elapsed_sec"] = time.time() - start_ts
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def maybe_publish(self):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return None
        if self.last_global_step is None:
            return None
        if self.training_interval_steps <= 0 or self.last_global_step % self.training_interval_steps != 0:
            return None
        if self.last_trained_step != self.last_global_step:
            return None
        has_snapshot, weights = self.trainer.pop_model_state_dict_for_publish(self.last_global_step)
        if not has_snapshot:
            logger.debug(
                "[SpecoWorker replica=%s rank=%s] missing cached publish snapshot at step %s; skip publish.",
                self.replica_rank,
                self.rank,
                self.last_global_step,
            )
            return None
        if not self.is_global_publish_leader:
            return None

        return {"weights_ref": ray.put(weights)}
