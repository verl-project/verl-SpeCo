import logging
import json
import os
import time
import asyncio
import random
import fnmatch
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any
from omegaconf import open_dict
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.nn import SmoothL1Loss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.device import get_device_name, get_torch_device
from verl.workers.drafter.data_buffer import DataBuffer
from verl.utils.fsdp_utils import (
    get_fsdp_full_state_dict,
    get_fsdp_wrap_policy,
    get_device_id,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    MixedPrecisionPolicy,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

_ALIGNMENT_DEBUG_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG"
_ALIGNMENT_DEBUG_EVERY_N_STEPS_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_EVERY_N_STEPS"
_ALIGNMENT_DEBUG_MAX_SAMPLES_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_MAX_SAMPLES_PER_STEP"
_ALIGNMENT_DEBUG_TOKEN_WINDOW_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_TOKEN_WINDOW"
_ALIGNMENT_DEBUG_RANKS_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_RANKS"


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def alignment_debug_enabled() -> bool:
    return _env_flag_enabled(_ALIGNMENT_DEBUG_ENV, default=False)


def alignment_debug_every_n_steps() -> int:
    return _env_int(_ALIGNMENT_DEBUG_EVERY_N_STEPS_ENV, default=50, minimum=1)


def alignment_debug_max_samples_per_step() -> int:
    return _env_int(_ALIGNMENT_DEBUG_MAX_SAMPLES_ENV, default=2, minimum=1)


def alignment_debug_token_window() -> int:
    return _env_int(_ALIGNMENT_DEBUG_TOKEN_WINDOW_ENV, default=3, minimum=1)


def alignment_debug_rank_selected(rank: int | None) -> bool:
    raw_value = os.getenv(_ALIGNMENT_DEBUG_RANKS_ENV, "0").strip().lower()
    if raw_value in {"*", "all"}:
        return True
    if rank is None:
        return False

    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        return False

    for item in raw_value.replace(",", " ").split():
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            try:
                if int(start) <= rank_int <= int(end):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(item) == rank_int:
                    return True
            except ValueError:
                continue
    return False


def should_log_alignment(
    step: int | None,
    rank: int | None,
    sample_index: int | None = 0,
    *,
    force: bool = False,
) -> bool:
    if not alignment_debug_enabled() or not alignment_debug_rank_selected(rank):
        return False

    if force:
        return True

    if sample_index is not None:
        try:
            if int(sample_index) >= alignment_debug_max_samples_per_step():
                return False
        except (TypeError, ValueError):
            return False

    if step is None:
        return False

    try:
        step_int = int(step)
    except (TypeError, ValueError):
        return False
    return step_int % alignment_debug_every_n_steps() == 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def log_alignment_event(logger: logging.Logger, payload: dict[str, Any], level: int = logging.INFO) -> None:
    if not alignment_debug_enabled():
        return
    logger.log(
        level,
        "DRAFTER_ALIGNMENT %s",
        json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")),
    )


def _tensor_shape(tensor: Optional[torch.Tensor]) -> list[int] | None:
    if torch.is_tensor(tensor):
        return list(tensor.shape)
    return None


def _batch_item_int(value: Any, index: int = 0) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        flat = value.detach().view(-1).cpu()
        index = min(max(int(index), 0), flat.numel() - 1)
        return int(flat[index].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        index = min(max(int(index), 0), len(value) - 1)
        value = value[index]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tensor_sum_int(tensor: torch.Tensor) -> int:
    return int(tensor.detach().float().sum().cpu().item())


def _tensor_scalar_int(tensor: torch.Tensor) -> int | None:
    if tensor.numel() == 0:
        return None
    return int(tensor.detach().view(-1)[0].cpu().item())


def _target_row_valid_mask(target_logprobs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if target_logprobs is None or target_logprobs.numel() == 0:
        return None
    return torch.isfinite(target_logprobs[..., 0]).any(dim=-1) & (target_logprobs[..., 1] >= 0).any(dim=-1)


def _target_top_ids(target_logprobs: torch.Tensor, row: int, limit: int) -> list[int]:
    if target_logprobs is None or target_logprobs.numel() == 0 or row >= target_logprobs.size(0):
        return []
    token_ids = target_logprobs[row, :, 1].detach().cpu()
    top_ids = []
    for token_id in token_ids[:limit].tolist():
        try:
            token_id_int = int(token_id)
        except (TypeError, ValueError):
            continue
        if token_id_int >= 0:
            top_ids.append(token_id_int)
    return top_ids


def _eagle_target_logprobs_train_start(source_item: dict) -> int:
    feature_start = source_item.get("_verl_feature_start")
    target_position_start = source_item.get("_verl_target_position_start")
    try:
        return max(int(feature_start) + 1 - int(target_position_start), 0)
    except (TypeError, ValueError):
        # Legacy collected samples include an anchor target row at feature_start.
        return 1


def _first_index(mask: torch.Tensor) -> int | None:
    indices = torch.nonzero(mask, as_tuple=False).view(-1)
    if indices.numel() == 0:
        return None
    return int(indices[0].detach().cpu().item())


def _last_index(mask: torch.Tensor) -> int | None:
    indices = torch.nonzero(mask, as_tuple=False).view(-1)
    if indices.numel() == 0:
        return None
    return int(indices[-1].detach().cpu().item())


def _alignment_window_rows(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    target_logprobs: Optional[torch.Tensor],
    *,
    feature_start: int,
    prompt_len: int | None,
    response_len: int | None,
) -> list[dict]:
    if target_logprobs is None:
        return []
    target_rows = min(target_logprobs.size(0), max(input_ids.size(0) - 1, 0), max(loss_mask.size(0) - 1, 0))
    if target_rows <= 0:
        return []

    row_valid = _target_row_valid_mask(target_logprobs[:target_rows])
    if row_valid is None:
        return []
    active_mask = loss_mask[1 : 1 + target_rows].bool()
    invalid_active_mask = active_mask & ~row_valid

    candidates = [
        ("first_active", _first_index(active_mask)),
        ("first_invalid_active", _first_index(invalid_active_mask)),
        ("last_active", _last_index(active_mask)),
    ]

    rows = []
    seen_rows = set()
    topk_limit = alignment_debug_token_window()
    prompt_len_int = int(prompt_len) if prompt_len is not None else None
    response_len_int = int(response_len) if response_len is not None else None
    for kind, local_row in candidates:
        if local_row is None or local_row in seen_rows:
            continue
        seen_rows.add(local_row)
        predict_token = _tensor_scalar_int(input_ids[local_row + 1])
        orig_hidden_pos = int(feature_start) + local_row
        predict_pos = orig_hidden_pos + 1
        response_idx = None
        if prompt_len_int is not None:
            response_idx_candidate = predict_pos - prompt_len_int
            if response_idx_candidate >= 0 and (
                response_len_int is None or response_idx_candidate < response_len_int
            ):
                response_idx = response_idx_candidate
        top_ids = _target_top_ids(target_logprobs, local_row, topk_limit)
        rows.append(
            {
                "kind": kind,
                "local_row": local_row,
                "orig_hidden_pos": orig_hidden_pos,
                "predict_pos": predict_pos,
                "predict_token": predict_token,
                "response_idx": response_idx,
                "loss": int(loss_mask[local_row + 1].detach().cpu().item() > 0),
                "target_row": orig_hidden_pos,
                "target_valid": bool(row_valid[local_row].detach().cpu().item()),
                "top_ids": top_ids,
                "contains_predict": predict_token in top_ids if predict_token is not None else None,
            }
        )
    return rows

class DrafterBaseTrainer:
    def __init__(
        self,
        config,
        world_size: int,
        rollout_dp_rank: int,
        training_device_mesh: Optional[DeviceMesh],
        backend,
        training_process_group=None,
        data_parallel_process_group=None,
    ):   
        self.config = config
        self.world_size = world_size
        self.rollout_dp_rank = rollout_dp_rank
        self.backend = backend
        self.pad_token_id = 0
        model_cfg = getattr(config, "model", None)
        if model_cfg is not None:
            try:
                configured_pad_id = model_cfg.get("pad_token_id", None)
            except AttributeError:
                configured_pad_id = getattr(model_cfg, "pad_token_id", None)
            if configured_pad_id is not None:
                self.pad_token_id = int(configured_pad_id)
        self.training_device_mesh = training_device_mesh
        if self.training_device_mesh is not None:
            self.training_process_group = self.training_device_mesh["sp"].get_group()
            self.data_parallel_process_group = self.training_device_mesh["dp"].get_group()
            self.training_group_world_size = self.training_device_mesh["sp"].size()
            self.dp_group_world_size = self.training_device_mesh["dp"].size()
            self.rank = self.training_device_mesh["sp"].get_local_rank()
            self.dp_rank = self.training_device_mesh["dp"].get_local_rank()
        else:
            self.training_process_group = training_process_group
            self.data_parallel_process_group = data_parallel_process_group
            self.training_group_world_size = (
                dist.get_world_size(training_process_group) if training_process_group is not None else 1
            )
            self.dp_group_world_size = (
                dist.get_world_size(data_parallel_process_group) if data_parallel_process_group is not None else 1
            )
            self.rank = dist.get_rank(training_process_group) if training_process_group is not None else (
                dist.get_rank() if dist.is_initialized() else 0
            )
            self.dp_rank = dist.get_rank(data_parallel_process_group) if data_parallel_process_group is not None else 0
        self.use_data_buffer = bool(config.rollout.drafter.training.get("use_data_buffer", False))
        self.current_rl_step = 0
        
        self.device_id = get_device_id()
        self.device_module = get_torch_device()
        self.runtime_device = (
            torch.device(f"{device_name}:{self.device_id}") if device_name != "cpu" else torch.device("cpu")
        )
        self.copy_stream = self._create_copy_stream()

        self.is_offload_param = False
        self.is_offload_optimizer = False
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0
        self._alignment_debug_step = None
        self._alignment_debug_counts = {}

        self.collected_data = deque(maxlen=int(self.config.rollout.drafter.training.get("current_max_samples", 2000)))
        self.shared_data_buffer = None
        self.batch_size = int(self.config.rollout.drafter.training.get("batch_size_per_gpu", 4))

        # Initialize DataBuffer for storing data across RL steps
        buffer_max_size = int(self.config.rollout.drafter.training.get("data_buffer_max_size", 10000))
        # Only store hidden states in buffer if we're collecting them during generation
        collect_hidden_states_from_sgl = bool(self.config.rollout.drafter.training.get("collect_hidden_states_from_sgl", False))

        #DataBuffer define
        self.data_buffer = DataBuffer(max_size=buffer_max_size, store_hidden_states=collect_hidden_states_from_sgl)

        self.criterion = SmoothL1Loss(reduction="none")

        self._last_ckpt_step = -1
        # New: optional per-step barrier (default False to avoid stalls)
        self.enable_mesh_barrier = bool(self.config.rollout.drafter.training.get("enable_step_barrier", False))

        # Track the last pending async checkpoint save future
        self._pending_checkpoint_future = None
        self._pending_full_checkpoint_future = None
        self._full_checkpoint_executor = None
        self._pending_publish_state_dict = None
        self._pending_publish_step = None
        self._pending_publish_ready = False
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.drafter_train_config = None
        self._pending_target_lm_head_weight = None
        self._target_lm_head_weight_step = None
        self._frozen_param_names = {"model.embed_tokens.weight"}

        # Ulysses Sequence Parallelism configuration
        self.ulysses_sequence_parallel_size = min(
            int(self.config.rollout.get("tensor_model_parallel_size", 1)),
            self.training_group_world_size,
        )
        self.use_ulysses_sp = self.training_group_world_size > 1 and self.ulysses_sequence_parallel_size > 1
        setattr(self.backend, "use_ulysses_sp", self.use_ulysses_sp)
        self.use_native_dp_sp = self.training_group_world_size > 1 and self.dp_group_world_size > 1

        self.checkpoint_dir = self.config.rollout.drafter.get("checkpoint_path")
        self.step = self.config.rollout.drafter.training.step

    def _create_copy_stream(self):
        if device_name == "cpu":
            return None
        stream_cls = getattr(self.device_module, "Stream", None)
        if stream_cls is None:
            return None
        try:
            return stream_cls()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to create drafter copy stream on {device_name}: {exc}")
            return None

    def _has_mesh_dim(self, dim_name: str) -> bool:
        return (
            self.training_device_mesh is not None
            and getattr(self.training_device_mesh, "mesh_dim_names", None) is not None
            and dim_name in self.training_device_mesh.mesh_dim_names
        )

    def _get_sp_group(self):
        if self._has_mesh_dim("sp"):
            return self.training_device_mesh["sp"].get_group()
        return self.training_process_group

    def _get_dp_group(self):
        if self._has_mesh_dim("dp"):
            return self.training_device_mesh["dp"].get_group()
        return self.data_parallel_process_group

    def _get_sp_world_size(self) -> int:
        if self._has_mesh_dim("sp"):
            return self.training_device_mesh["sp"].size()
        return self.training_group_world_size

    def _get_dp_world_size(self) -> int:
        if self._has_mesh_dim("dp"):
            return self.training_device_mesh["dp"].size()
        return self.dp_group_world_size

    def _get_sp_local_rank(self) -> int:
        if self._has_mesh_dim("sp"):
            return self.training_device_mesh["sp"].get_local_rank()
        return self.rank

    def _get_dp_local_rank(self) -> int:
        if self._has_mesh_dim("dp"):
            return self.training_device_mesh["dp"].get_local_rank()
        return self.dp_rank

    def _resolve_fsdp_config(self):
        # Primary source: actor fsdp config used across PPO training stacks.
        fsdp_config = None
        if hasattr(self.config, "actor"):
            fsdp_config = self.config.actor.get("fsdp_config")
        # Optional fallback for drafter-local overrides.
        if fsdp_config is None:
            fsdp_config = self.config.rollout.drafter.training.get("fsdp_config")
        if fsdp_config is None:
            raise ValueError("FSDP config is missing: expect actor_rollout_ref.actor.fsdp_config or drafter override")
        return fsdp_config

    def _build_draft_model(self):
        """build draft model"""
        logger.info(f"[Rank {self.rollout_dp_rank}] Building drafter model...")
        # A. 实例化模型（委托给backend）
        raw_model, drafter_model_config = self.backend.build_model()
        raw_model.to(self.runtime_device)

        # B. 获取全量状态用于 FSDP 初始化

        # C. FSDP包装
        if self.training_device_mesh is not None and dist.is_initialized():
            fsdp_config = self._resolve_fsdp_config()
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.training_device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": None,
            }
            logger.info("Building drafter model with mesh-centered dp x sp FSDP2")

            full_state = raw_model.state_dict()
            apply_fsdp2(raw_model, fsdp_kwargs, fsdp_config)

            # Load full state dict using the same mesh as used by drafter FSDP wrapping
            fsdp2_load_full_state_dict(raw_model, full_state, self.training_device_mesh, None)
            self.model = raw_model
            del full_state
        elif self.training_process_group is not None and self.training_group_world_size > 1 and dist.is_initialized():
            fsdp_config = self._resolve_fsdp_config()
            auto_wrap_policy = get_fsdp_wrap_policy(module=raw_model, config=fsdp_config.wrap_policy)
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
            sharding_strategy = ShardingStrategy.FULL_SHARD
            process_group = self.training_process_group
            if self.use_native_dp_sp:
                logger.info("Building drafter model with native dp x sp hybrid-shard FSDP")
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
                process_group = (self.training_process_group, self.data_parallel_process_group)
            else:
                logger.info("Building drafter model with subgroup FSDP")
            self.model = FSDP(
                raw_model,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                process_group=process_group,
                use_orig_params=fsdp_config.use_orig_params,
                forward_prefetch=fsdp_config.forward_prefetch,
                cpu_offload=None,
            )
        else:
            logger.info("Building drafter model without FSDP (standalone/local trainer mode)")
            self.model = raw_model

        # D. 构建优化器和调度器
        drafter_train_config = self._prepare_training_config(self.config.rollout)
        setattr(self.backend, "train_config", drafter_train_config)

        self.optimizer = self.backend.setup_optimizer(self.model, drafter_train_config)
        self.lr_scheduler = self.backend.setup_scheduler(self.optimizer, drafter_train_config)
        self.drafter_train_config = drafter_train_config
        self.model_config = drafter_model_config
        self.pad_token_id = int(getattr(drafter_model_config, "pad_token_id", self.pad_token_id) or self.pad_token_id)
        self._apply_pending_target_lm_head_weight()
        
    def _prepare_training_config(self, rollout_config):
        """
        Prepare the training configuration for drafter module.

        Args:
            rollout_config (dict): The rollout configuration.

        Returns:
            dict: The prepared training configuration.
        """
        drafter_train_config = rollout_config['drafter']['training'].copy()

        # Open the dictionary for modification
        with open_dict(drafter_train_config):
            # Update the configuration with required values
            drafter_train_config.update(
                {
                    "speculative_algorithm": rollout_config['drafter']['speculative_algorithm'],
                    "model_path": rollout_config['drafter']['model_path'],
                    "is_offload_optimizer": False,
                    "is_offload_param": False,
                    "vloss_weight": 1.0,
                    "ploss_weight": 0.1,
                    "data_augment_std": 0.2,
                }
            )

        return drafter_train_config

    
    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get floating state dict entries excluding weights shared with the target model."""
        if isinstance(self.model, FSDP) or (self.training_device_mesh is not None and dist.is_initialized()):
            full_state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
        else:
            full_state_dict = self.model.state_dict()
        if not full_state_dict:
            return {}
        trainable_state_dict = {}

        for name, param in full_state_dict.items():
            # EAGLE3 vocab mapping buffers are static and can break SGLang hot update device assumptions.
            if isinstance(param, torch.Tensor) and not torch.is_floating_point(param):
                logger.debug(f"Skipping non-floating drafter state: {name}, dtype={param.dtype}")
                continue
            # EAGLE shares target lm_head, while EAGLE3 trains and publishes its own lm_head.
            if any(frozen_name in name for frozen_name in self._frozen_param_names) or (
                getattr(self.backend, "model_type", None) == "dflash" and "embed_tokens.weight" in name
            ) or (
                "lm_head.weight" in name and getattr(self.backend, "model_type", None) != "eagle3"
            ):
                logger.debug(f"Skipping frozen parameter: {name}")
                continue
            trainable_state_dict[name] = param

        return trainable_state_dict

    def _get_full_export_state_dict(self) -> dict[str, torch.Tensor]:
        """Get a complete CPU state dict for offline drafter export.

        Unlike `_get_trainable_state_dict`, this keeps frozen parameters and
        persistent buffers such as vocab mappings. It is intentionally not used
        by hot publish.
        """
        if isinstance(self.model, FSDP) or (self.training_device_mesh is not None and dist.is_initialized()):
            full_state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
        else:
            full_state_dict = self.model.state_dict()
        if not full_state_dict:
            return {}
        return {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in full_state_dict.items()
            if isinstance(tensor, torch.Tensor)
        }

    def _is_checkpoint_leader(self) -> bool:
        return self.rollout_dp_rank == 0 and self._get_sp_local_rank() == 0

    def _should_save_full_drafter_checkpoint(self, is_final: bool) -> bool:
        training_cfg = self.config.rollout.drafter.training
        return bool(is_final and training_cfg.get("save_full_drafter_checkpoint", False))

    def _copy_drafter_config_files(self, output_dir: str) -> None:
        spec_model_path = self.config.rollout.drafter.model_path
        if not spec_model_path or not os.path.isdir(spec_model_path):
            return
        for filename in ("config.json", "generation_config.json"):
            src = os.path.join(spec_model_path, filename)
            dst = os.path.join(output_dir, filename)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    logger.warning("Failed to copy drafter %s to full checkpoint: %s", filename, exc)

    def _save_full_export_checkpoint_async(self, checkpoint_path: str, step: int, is_final: bool = False):
        if not self._should_save_full_drafter_checkpoint(is_final):
            return None
        if self._pending_full_checkpoint_future is not None:
            if not self._pending_full_checkpoint_future.done():
                logger.warning(
                    "[Rank %s] Previous full drafter checkpoint save is still running; skip step=%s",
                    self.rank,
                    step,
                )
                return None
            try:
                self._pending_full_checkpoint_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Previous full drafter checkpoint save failed: %s", exc)
            self._pending_full_checkpoint_future = None

        model_state_dict = self._get_full_export_state_dict()
        if not self._is_checkpoint_leader() or not model_state_dict:
            return None

        export_dir = os.path.join(checkpoint_path, "full_drafter")
        output_path = os.path.join(export_dir, "pytorch_model.bin")
        metadata_path = os.path.join(export_dir, "metadata.json")

        def _write_full_checkpoint():
            os.makedirs(export_dir, exist_ok=True)
            torch.save(model_state_dict, output_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"step": step, "format": "full_drafter_export"}, f, indent=2)
            self._copy_drafter_config_files(export_dir)

        if self._full_checkpoint_executor is None:
            self._full_checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drafter-full-ckpt")
        future = self._full_checkpoint_executor.submit(_write_full_checkpoint)
        self._pending_full_checkpoint_future = future
        logger.info("[Rank %s] Scheduled full drafter checkpoint export to %s", self.rank, export_dir)
        return future

    
    def _save_checkpoint_async(self, step: int, is_final: bool = False):
        """Asynchronously save checkpoint using DCP's async_save.

        Args:
            step: Current training step
            is_final: Whether this is the final checkpoint during cleanup

        Returns:
            Future object from dcp.async_save that can be awaited or checked for completion
        """
        if not self.checkpoint_dir:
            return None

        checkpoint_path = os.path.join(self.checkpoint_dir, f"draft_step_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Get trainable state dict (excluding frozen layers)
        model_state_dict = self._get_trainable_state_dict()
        is_fsdp_wrapped = isinstance(self.model, FSDP) or self.training_device_mesh is not None
        is_checkpoint_leader = self.rollout_dp_rank == 0 and self._get_sp_local_rank() == 0
        optimizer_state_dict = self.optimizer.state_dict() if self.optimizer and is_checkpoint_leader else {}

        state_dict = {"model": model_state_dict, "optimizer": optimizer_state_dict, "step": step}
        self._save_full_export_checkpoint_async(checkpoint_path, step, is_final=is_final)

        if is_fsdp_wrapped:
            if is_checkpoint_leader and model_state_dict:
                torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
            return None

        # Standalone mode: no distributed mesh, save locally.
        if self.training_device_mesh is None or not dist.is_initialized():
            torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
            return None

        # Use DCP async_save - returns a future that can be checked later
        future = dcp.async_save(
            state_dict=state_dict,
            checkpoint_id=checkpoint_path,
            process_group=self.training_device_mesh.get_group(),
        )
        return future

    async def activate_training_model(self) -> bool:
        # 将模型和优化器状态从CPU加载到GPU，激活草稿模型进入训练状态
        start_ts = time.time()
        try:        
            logger.info(
                f"[Trainer rank {getattr(self, 'rank', -1)}] activate_training_model enter "
            )

            if self.model is None:
                logger.info("Draft Model not initialized, calling build_draft_model during activation...")
                self._build_draft_model()

            # 只有当配置了 offload 或者当前模型不在 CUDA 上时执行加载
            first_param = next(self.model.parameters(), None)
            is_on_cuda = first_param is not None and first_param.device.type == device_name

            if self.is_offload_param or not is_on_cuda:
                # 调用工具将 FSDP 分片移动到 GPU
                load_fsdp_model_to_gpu(self.model)
                logger.debug("Loaded drafter model to GPU for training")
            
            if self.optimizer is not None:
                # 获取 device_id,否则在多卡环境优化器状态可能全部挤在 cuda:0 导致 OOM
                current_dev_id = get_device_id()
                load_fsdp_optimizer(optimizer=self.optimizer, device_id=current_dev_id)
                logger.debug("Loaded drafter optimizer to GPU for training")

            target_model = getattr(self.backend, "target_model", None)
            if target_model is not None:
                target_model.to(self.runtime_device)
            self._apply_pending_target_lm_head_weight()

            # 先标记初始化完成，然后开启 active 开关，确保训练循环不会读到中间状态
            self._training_initialized = True
            self._training_active = True

            logger.info(
                f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model success "
                f"elapsed={time.time() - start_ts:.2f}s"
            )
            return True
        
        except Exception as e:
            logger.error(f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model failed: {e}")
            self._training_active = False
            return False

    def sync_target_lm_head_weight(self, weight: torch.Tensor, global_step: Optional[int] = None) -> dict[str, Any]:
        """Update the frozen target lm_head used by last-hidden drafter training."""
        if weight is None or not torch.is_tensor(weight):
            return {"accepted": False, "applied": False, "reason": "missing_weight"}

        self._pending_target_lm_head_weight = weight.detach().cpu().contiguous()
        self._target_lm_head_weight_step = global_step
        applied = self._apply_pending_target_lm_head_weight()
        return {
            "accepted": True,
            "applied": applied,
            "pending": not applied,
            "global_step": global_step,
            "shape": tuple(self._pending_target_lm_head_weight.shape),
        }

    def _target_lm_head_module(self):
        target_model = getattr(self.backend, "target_model", None)
        if target_model is not None and getattr(target_model, "fc", None) is not None:
            return target_model.fc
        target_lm_head = getattr(self.backend, "target_lm_head", None)
        if target_lm_head is not None and getattr(target_lm_head, "fc", None) is not None:
            return target_lm_head.fc
        return None

    @torch.no_grad()
    def _apply_pending_target_lm_head_weight(self) -> bool:
        if self._pending_target_lm_head_weight is None:
            return False

        lm_head = self._target_lm_head_module()
        if lm_head is None:
            return False

        target_weight = lm_head.weight
        source_weight = self._pending_target_lm_head_weight
        if tuple(source_weight.shape) != tuple(target_weight.shape):
            raise ValueError(
                "Target lm_head weight shape mismatch for drafter training: "
                f"source={tuple(source_weight.shape)}, target={tuple(target_weight.shape)}"
            )

        target_weight.copy_(source_weight.to(device=target_weight.device, dtype=target_weight.dtype, non_blocking=True))
        lm_head.requires_grad_(False)
        logger.warning(
            "[drafter target lm_head sync] applied global_step=%s shape=%s dtype=%s device=%s",
            self._target_lm_head_weight_step,
            tuple(target_weight.shape),
            target_weight.dtype,
            target_weight.device,
        )
        return True

    def collect_online_data(self, batch: dict, hidden_states: torch.Tensor, target_logprobs: List = None) -> None:
        """Collect online data from inference for Eagle training.

        This method stores hidden states in the cross-step DataBuffer only when
        use_data_buffer=True. Otherwise it keeps only the current-step samples.
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            logger.warning(
                f"[Rank {self.rank}] Non-batched data in input_ids"
            )
            return

        # 1、异步拷贝，GPU在后台进行数据搬运，避免阻塞Rollout Stream
        if target_logprobs is not None and not isinstance(target_logprobs, torch.Tensor):
            logger.warning(f"[Rank {self.rank}] Unsupported target_logprobs type: {type(target_logprobs)}")
            target_logprobs = None

        source_tensors = [input_ids, hidden_states]
        if target_logprobs is not None:
            source_tensors.append(target_logprobs)
        if "responses" in batch and batch["responses"] is not None:
            source_tensors.append(batch["responses"])
        if "prompts" in batch and batch["prompts"] is not None:
            source_tensors.append(batch["prompts"])

        use_copy_stream = self.copy_stream is not None and any(
            isinstance(t, torch.Tensor) and t.device.type == device_name for t in source_tensors
        )

        if use_copy_stream:
            with self.device_module.stream(self.copy_stream):
                cpu_input_ids = input_ids.to('cpu', non_blocking=True)
                cpu_h_states = hidden_states.to('cpu', non_blocking=True)
                cpu_target_logprobs = (
                    target_logprobs.to('cpu', non_blocking=True) if target_logprobs is not None else None
                )
                cpu_responses = batch.get("responses").to('cpu', non_blocking=True) if "responses" in batch else None
                cpu_prompts = batch.get("prompts").to('cpu', non_blocking=True) if "prompts" in batch else None

            self.device_module.current_stream().wait_stream(self.copy_stream)
        else:
            cpu_input_ids = input_ids.to('cpu')
            cpu_h_states = hidden_states.to('cpu')
            cpu_target_logprobs = target_logprobs.to('cpu') if target_logprobs is not None else None
            cpu_responses = batch.get("responses").to('cpu') if "responses" in batch else None
            cpu_prompts = batch.get("prompts").to('cpu') if "prompts" in batch else None

        batch_size = cpu_input_ids.size(0)

        input_seq_length = cpu_input_ids.size(1)
        hidden_seq_length = cpu_h_states.size(1)
        if min(input_seq_length, hidden_seq_length) <= 0:
            return

        model_config = getattr(self, "model_config", None)
        pad_id = int(getattr(model_config, "pad_token_id", self.pad_token_id) or self.pad_token_id)
        for i in range(batch_size):
            expected_hidden_rows = max(input_seq_length - 1, 0)
            hidden_position_start = _batch_item_int(batch.get("hidden_position_start"), i)
            if hidden_position_start is None:
                hidden_position_start = max(expected_hidden_rows - hidden_seq_length, 0)
            # Hidden rows are next-token features for original positions. If
            # prefix cache reused leading prompt tokens, the first returned
            # hidden row starts after that reused prefix; keep the remaining
            # rows head-aligned from that original position onward.
            feature_start = min(max(hidden_position_start, 0), input_seq_length)
            hidden_start = 0
            hidden_feature_length = min(hidden_seq_length, max(input_seq_length - feature_start - 1, 0))
            hidden_end = hidden_feature_length
            feature_end = feature_start + hidden_feature_length + 1

            target_logprobs_position_start = None
            target_logprobs_position_end = None
            if cpu_target_logprobs is not None:
                target_rows = int(cpu_target_logprobs.size(1))
                target_logprobs_position_start = _batch_item_int(batch.get("target_logprobs_position_start"), i)
                target_logprobs_position_end = _batch_item_int(batch.get("target_logprobs_position_end"), i)
                if target_logprobs_position_start is None:
                    target_logprobs_position_start = 0
                if target_logprobs_position_end is None:
                    target_logprobs_position_end = target_logprobs_position_start + target_rows
                target_logprobs_position_end = min(
                    max(target_logprobs_position_end, target_logprobs_position_start),
                    target_logprobs_position_start + target_rows,
                )

                # For EAGLE/EAGLE3, the loss target for hidden row p is the
                # target row at original position p + 1. A compact tensor may
                # therefore start one row after the hidden window without
                # requiring us to drop the corresponding hidden/input row.
                target_row_offset = (
                    1
                    if getattr(self.backend, "model_type", None) in ("eagle", "eagle3")
                    else 0
                )
                required_target_start = feature_start + target_row_offset
                if target_logprobs_position_start > required_target_start:
                    hidden_shift = target_logprobs_position_start - required_target_start
                    hidden_start += hidden_shift
                    hidden_feature_length -= hidden_shift
                    feature_start += hidden_shift

                if hidden_feature_length <= 0:
                    hidden_feature_length = 0
                    hidden_end = hidden_start
                    feature_end = feature_start + 1
                else:
                    hidden_end = hidden_start + hidden_feature_length
                    feature_end = feature_start + hidden_feature_length + 1

            input_feature_length = feature_end - feature_start
            if input_feature_length <= 0 or hidden_feature_length <= 0:
                continue

            full_loss_mask = torch.zeros(input_seq_length, dtype=torch.float32)
            if cpu_prompts is not None and cpu_responses is not None:
                prompt_len = min(cpu_prompts[i].numel(), input_seq_length)
                response_len = min(cpu_responses[i].numel(), max(0, input_seq_length - prompt_len))
                if response_len > 0:
                    response_mask = (cpu_responses[i, :response_len] != pad_id).float()
                    full_loss_mask[prompt_len : prompt_len + response_len] = response_mask
            elif cpu_responses is not None:
                response_len = min(cpu_responses[i].numel(), input_seq_length)
                response_start = input_seq_length - response_len
                full_loss_mask[response_start:] = (cpu_responses[i, :response_len] != pad_id).float()
            else:
                full_loss_mask[:] = 1.0

            target_logprobs_item = None
            target_start = None
            target_end = None
            if cpu_target_logprobs is not None:
                # target_logprobs row p stores the next-token target for the
                # original position p. For shifted EAGLE training, the first
                # usable target row is feature_start + 1.
                target_base = target_logprobs_position_start or 0
                target_row_offset = (
                    1
                    if getattr(self.backend, "model_type", None) in ("eagle", "eagle3")
                    else 0
                )
                target_limit = min(
                    cpu_target_logprobs.size(1),
                    max((target_logprobs_position_end or target_base) - target_base, 0),
                )
                target_start = min(max(feature_start + target_row_offset - target_base, 0), target_limit)
                target_end = min(max(feature_end - 1 - target_base, target_start), target_limit)
                target_logprobs_item = cpu_target_logprobs[i, target_start:target_end, ...]

            data_item = {
                "input_ids": cpu_input_ids[i, feature_start:feature_end],
                "hidden_states": cpu_h_states[i, hidden_start:hidden_end, :],
                "loss_mask": full_loss_mask[feature_start:feature_end],
                "position_ids": torch.arange(input_feature_length, dtype=torch.long),
                "target_logprobs": target_logprobs_item,
                "responses": cpu_responses[i] if cpu_responses is not None else None,
                "prompts": cpu_prompts[i] if cpu_prompts is not None else None,
                "_verl_feature_start": feature_start,
                "_verl_feature_end": feature_end,
                "_verl_hidden_start": hidden_start,
                "_verl_hidden_end": hidden_end,
                "_verl_hidden_position_start": hidden_position_start,
                "_verl_target_start": target_start if cpu_target_logprobs is not None else None,
                "_verl_target_end": target_end if cpu_target_logprobs is not None else None,
                "_verl_target_position_start": (
                    (target_logprobs_position_start or 0) + target_start if target_start is not None else None
                ),
                "_verl_target_position_end": (
                    (target_logprobs_position_start or 0) + target_end if target_end is not None else None
                ),
                "_verl_target_tensor_position_start": target_logprobs_position_start,
                "_verl_target_tensor_position_end": target_logprobs_position_end,
                "_verl_prompt_len": prompt_len if cpu_prompts is not None else None,
                "_verl_response_len": response_len if cpu_responses is not None else None,
                "_verl_input_seq_length": input_seq_length,
            }

            if alignment_debug_enabled():
                sample_index = self._alignment_debug_sample_index(self.current_rl_step, "collect")
                row_valid_count = None
                active_rows = None
                active_valid = None
                if target_logprobs_item is not None:
                    row_valid = _target_row_valid_mask(target_logprobs_item)
                    if row_valid is not None and row_valid.numel() > 0:
                        target_debug_offset = (
                            2
                            if getattr(self.backend, "model_type", None) in ("eagle", "eagle3")
                            else 1
                        )
                        active_mask = data_item["loss_mask"][
                            target_debug_offset : target_debug_offset + row_valid.size(0)
                        ].bool()
                        row_valid_count = int(row_valid.detach().sum().cpu().item())
                        active_rows = int(active_mask.detach().sum().cpu().item())
                        active_valid = int((row_valid & active_mask).detach().sum().cpu().item())
                force_alignment_log = target_logprobs_item is not None and active_valid is not None and active_valid <= 0
                if should_log_alignment(self.current_rl_step, self.rank, sample_index, force=force_alignment_log):
                    log_alignment_event(
                        logger,
                        {
                            "event": "drafter_align_collect",
                            "step": self.current_rl_step,
                            "rank": self.rank,
                            "sample": sample_index,
                            "input_len": input_seq_length,
                            "hidden_len": hidden_seq_length,
                            "hidden_raw_len": hidden_seq_length,
                            "hidden_kept_len": hidden_feature_length,
                            "feature_start": feature_start,
                            "feature_end": feature_end,
                            "hidden_start": hidden_start,
                            "hidden_end": hidden_end,
                            "hidden_position_start": hidden_position_start,
                            "target_start": target_start if cpu_target_logprobs is not None else None,
                            "target_end": target_end if cpu_target_logprobs is not None else None,
                            "target_position_start": data_item.get("_verl_target_position_start"),
                            "target_position_end": data_item.get("_verl_target_position_end"),
                            "target_tensor_position_start": target_logprobs_position_start,
                            "target_tensor_position_end": target_logprobs_position_end,
                            "prompt_len": prompt_len if cpu_prompts is not None else None,
                            "response_len": response_len if cpu_responses is not None else None,
                            "loss_before": int(full_loss_mask.sum().item()),
                            "loss_after_slice": int(data_item["loss_mask"].sum().item()),
                            "target_rows": int(target_logprobs_item.size(0)) if target_logprobs_item is not None else None,
                            "row_valid": row_valid_count,
                            "active_rows": active_rows,
                            "active_valid": active_valid,
                            "target_shape": _tensor_shape(target_logprobs_item),
                        },
                    )
                    window_input_ids = data_item["input_ids"]
                    window_loss_mask = data_item["loss_mask"]
                    window_feature_start = feature_start
                    if getattr(self.backend, "model_type", None) in ("eagle", "eagle3"):
                        window_input_ids = data_item["input_ids"][1:]
                        window_loss_mask = data_item["loss_mask"][1:]
                        window_feature_start = feature_start + 1
                    window_rows = _alignment_window_rows(
                        window_input_ids,
                        window_loss_mask,
                        target_logprobs_item,
                        feature_start=window_feature_start,
                        prompt_len=prompt_len if cpu_prompts is not None else None,
                        response_len=response_len if cpu_responses is not None else None,
                    )
                    if window_rows:
                        log_alignment_event(
                            logger,
                            {
                                "event": "drafter_align_window",
                                "stage": "collect",
                                "step": self.current_rl_step,
                                "rank": self.rank,
                                "sample": sample_index,
                                "rows": window_rows,
                            },
                        )

            # 同步 DataBuffer
            if self.use_data_buffer:
                self.data_buffer.add_batch(data_item)

            # 同步 collect_data (当前步训练直接使用)
            else:
                data_item["step"] = self.current_rl_step
                self.collected_data.append(data_item)

    def _get_hidden_state_clip_value(self) -> Optional[float]:
        clip_value = self.config.rollout.drafter.training.get("hidden_state_clip_value", 1.0e4)
        if clip_value is None:
            return None
        clip_value = float(clip_value)
        return clip_value if clip_value > 0 else None

    def _sanitize_sequence_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        clip_value: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.dim() < 3:
            raise ValueError(f"{name} must have shape [batch, seq, ...], got {tuple(tensor.shape)}")

        bad_rows = ~torch.isfinite(tensor).flatten(start_dim=2).all(dim=-1)
        safe_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_value is not None:
            row_abs_max = safe_tensor.detach().float().abs().flatten(start_dim=2).amax(dim=-1)
            bad_rows = bad_rows | (row_abs_max > clip_value)
            safe_tensor = safe_tensor.clamp(min=-clip_value, max=clip_value)

        dropped_rows = int(bad_rows.detach().sum().cpu().item())
        if dropped_rows > 0:
            logger.warning(
                f"[Rank {self.rank}] Sanitized {dropped_rows} drafter {name} rows with non-finite or clipped values"
            )

        return safe_tensor, bad_rows

    def _mask_loss_for_bad_inputs(
        self,
        loss_mask: torch.Tensor,
        bad_input_rows: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bad_input_rows.any():
            return loss_mask, attention_mask

        ttt_length = 1
        if getattr(self.backend, "model_type", None) == "eagle3":
            ttt_length = int(self.config.rollout.drafter.training.get("ttt_length", 1))
        ttt_length = max(1, ttt_length)

        bad_target_rows = torch.zeros_like(bad_input_rows, dtype=torch.bool)
        seq_len = bad_input_rows.size(1)
        for offset in range(min(ttt_length, seq_len)):
            if offset == 0:
                bad_target_rows |= bad_input_rows
            else:
                bad_target_rows[:, offset:] |= bad_input_rows[:, : seq_len - offset]

        masked_tokens = int(((loss_mask > 0) & bad_target_rows).detach().sum().cpu().item())
        if masked_tokens > 0:
            logger.warning(f"[Rank {self.rank}] Masked {masked_tokens} drafter targets due to bad input hidden rows")

        loss_mask = loss_mask.masked_fill(bad_target_rows, 0.0)
        attention_mask = attention_mask.masked_fill(bad_input_rows, 0)
        return loss_mask, attention_mask

    def _sanitize_training_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clip_value = self._get_hidden_state_clip_value()

        loss_mask = torch.nan_to_num(batch["loss_mask"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        batch["loss_mask"] = torch.where(loss_mask > 0, torch.ones_like(loss_mask), torch.zeros_like(loss_mask))

        hidden_states, bad_input_rows = self._sanitize_sequence_tensor(
            batch["hidden_states"], "hidden_states", clip_value
        )
        batch["hidden_states"] = hidden_states
        batch["loss_mask"], batch["attention_mask"] = self._mask_loss_for_bad_inputs(
            batch["loss_mask"], bad_input_rows, batch["attention_mask"]
        )

        for target_key in ("target", "last_hidden_states"):
            if target_key not in batch:
                continue
            target_tensor, bad_target_rows = self._sanitize_sequence_tensor(batch[target_key], target_key, clip_value)
            batch[target_key] = target_tensor
            masked_tokens = int(((batch["loss_mask"] > 0) & bad_target_rows).detach().sum().cpu().item())
            if masked_tokens > 0:
                logger.warning(
                    f"[Rank {self.rank}] Masked {masked_tokens} drafter targets due to bad {target_key} rows"
                )
            batch["loss_mask"] = batch["loss_mask"].masked_fill(bad_target_rows, 0.0)

        return batch

    def _alignment_debug_sample_index(self, step: int | None, stage: str) -> int:
        if self._alignment_debug_step != step:
            self._alignment_debug_step = step
            self._alignment_debug_counts = {}
        sample_index = int(self._alignment_debug_counts.get(stage, 0))
        self._alignment_debug_counts[stage] = sample_index + 1
        return sample_index

    def _prepare_training_batch(
        self, buffer_steps: int = 2
    ) -> Optional[dict[str, torch.Tensor]]:
        """Prepare a batch for training using Ulysses SP to remove padding.

        Args:
            buffer_steps: Number of recent RL steps to include data from (only used if use_buffer_data=True)

        Returns:
            Dictionary containing batch tensors for training
        """
        effective_batch_size = self.batch_size

        current_step = int(self.current_rl_step)

        min_items_for_batch = 1

        use_logits = bool(self.config.rollout.drafter.training.get("use_logits", False))
        same_step_target_head_required = self.backend.model_type == "eagle3" and not use_logits

        # Determine data source: DataBuffer (cross-step) or collected_data (current step only).
        # last-hidden supervision can only be reconstructed with the exact target
        # head version that produced those hidden states, so older buffered Eagle3
        # samples are not valid for the actor head synced for this rollout step.
        if self.use_data_buffer and len(self.data_buffer) > 0:
            if same_step_target_head_required:
                buffer_steps = 0
            else:
                # Use data from last N RL steps via DataBuffer
                buffer_steps = int(self.config.rollout.drafter.training.get("sample_last_n_steps", buffer_steps))
            available_data = self.data_buffer.get_data_from_last_n_steps(buffer_steps)
            if len(available_data) < effective_batch_size:
                if len(available_data) >= min_items_for_batch:
                    items = available_data
                else:
                    return None
            else:
                # Randomly sample from available data to ensure diversity
                rng = random.Random((int(self.current_rl_step) << 16) + int(self.training_steps))
                items = rng.sample(available_data, min(len(available_data), effective_batch_size))
        else:
            # Fall back to current step data only. collected_data can contain
            # older rollout steps when drafter training is triggered sparsely.
            current_step_data = [
                item for item in self.collected_data if int(item.get("step", current_step)) == current_step
            ]
            if len(current_step_data) < effective_batch_size:
                if len(current_step_data) >= min_items_for_batch:
                    items = current_step_data
                else:
                    return None
            else:
                rng = random.Random((current_step << 16) + int(self.training_steps))
                items = rng.sample(current_step_data, effective_batch_size)

        # Filter out items without the tensors required by the selected loss path.
        items = [item for item in items if "hidden_states" in item]
        if self.backend.model_type == "eagle3" and use_logits:
            items = [item for item in items if item.get("target_logprobs") is not None]
        if len(items) == 0:
            logger.warning(f"[Rank {self.rank}] No items with hidden_states found, cannot prepare batch")
            return None
        elif len(items) < min_items_for_batch:
            logger.warning(
                f"[Rank {self.rank}] Only {len(items)} items with hidden_states found "
                f"(need at least {min_items_for_batch}), cannot prepare batch"
            )
            return None

        dev = next(self.model.parameters()).device
        if self.backend.model_type == "dflash" and self.use_ulysses_sp:
            raise NotImplementedError("DFlash drafter training does not support Ulysses sequence parallel yet")
        
        preprocessed_lists = self.backend.preprocess_individual_items(items, dev, self.model_config)
        items_seen = len(items)
        items_used = 0
        items_dropped_short = 0
        items_dropped_missing_target = 0
        packed_tokens_before_shift = 0
        packed_loss_tokens = 0
       
        # Build training chunks inside each sample before packing. EAGLE-style
        # models use next-token chunks, while DFlash keeps same-position blocks
        # per sample so sampled anchors never cross sample boundaries.
        input_id_chunks = []
        loss_mask_chunks = []
        hidden_state_chunks = []
        position_id_chunks = []
        target_chunks = []
        last_hidden_state_chunks = []
        target_logprob_chunks = []

        ids_list = preprocessed_lists["ids"]
        hidden_list = preprocessed_lists["h_states"]
        mask_list = preprocessed_lists["masks"]
        position_list = preprocessed_lists.get("position_ids")
        if position_list is None:
            position_list = [torch.arange(ids.size(0), device=ids.device, dtype=torch.long) for ids in ids_list]
        for item_idx, (ids, h_states, item_loss_mask, item_position_ids) in enumerate(
            zip(ids_list, hidden_list, mask_list, position_list)
        ):
            source_item = items[item_idx] if item_idx < len(items) else {}
            uses_shifted_eagle_inputs = self.backend.model_type in ("eagle", "eagle3")
            seq_len = min(ids.size(0), h_states.size(0), item_loss_mask.size(0), item_position_ids.size(0))
            if seq_len < 1:
                items_dropped_short += 1
                continue

            target_logprobs_item = None
            target_logprobs_train_start = 0
            if self.backend.model_type == "eagle3" and use_logits:
                target_logprobs_item = preprocessed_lists["target_logprobs"][item_idx]
                target_logprobs_train_start = _eagle_target_logprobs_train_start(source_item)
                train_seq_len = min(
                    max(ids.size(0) - 2, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 2, 0),
                    item_position_ids.size(0),
                    max(target_logprobs_item.size(0) - target_logprobs_train_start, 0),
                )
            elif self.backend.model_type == "eagle3":
                last_h_states = preprocessed_lists["last_h_states"][item_idx]
                train_seq_len = min(
                    max(ids.size(0) - 2, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 2, 0),
                    item_position_ids.size(0),
                    max(last_h_states.size(0) - 1, 0),
                )
            elif self.backend.model_type == "eagle":
                train_seq_len = min(
                    max(ids.size(0) - 2, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 2, 0),
                    item_position_ids.size(0),
                    max(h_states.size(0) - 1, 0),
                )
            elif self.backend.model_type == "dflash":
                train_seq_len = seq_len
            else:
                train_seq_len = seq_len - 1

            if train_seq_len < 1:
                items_dropped_missing_target += 1
                continue

            items_used += 1
            packed_tokens_before_shift += train_seq_len
            if self.backend.model_type == "dflash":
                packed_loss_tokens += int(item_loss_mask[:train_seq_len].detach().float().sum().cpu().item())
            elif uses_shifted_eagle_inputs:
                packed_loss_tokens += int(item_loss_mask[2 : 2 + train_seq_len].detach().float().sum().cpu().item())
            else:
                packed_loss_tokens += int(item_loss_mask[1 : 1 + train_seq_len].detach().float().sum().cpu().item())

            if alignment_debug_enabled():
                sample_index = self._alignment_debug_sample_index(current_step, "prepare_item")
                train_target_logprobs = None
                if self.backend.model_type == "eagle3" and use_logits:
                    train_target_logprobs = target_logprobs_item[
                        target_logprobs_train_start : target_logprobs_train_start + train_seq_len
                    ]
                row_valid_count = None
                active_rows = None
                active_valid = None
                if train_target_logprobs is not None:
                    row_valid = _target_row_valid_mask(train_target_logprobs)
                    if row_valid is not None and row_valid.numel() > 0:
                        if uses_shifted_eagle_inputs:
                            active_mask = item_loss_mask[2 : 2 + train_seq_len].bool()
                        else:
                            active_mask = item_loss_mask[1 : 1 + train_seq_len].bool()
                        active_mask = active_mask[: row_valid.size(0)]
                        row_valid_count = int(row_valid.detach().sum().cpu().item())
                        active_rows = int(active_mask.detach().sum().cpu().item())
                        active_valid = int((row_valid & active_mask).detach().sum().cpu().item())
                force_alignment_log = train_target_logprobs is not None and active_valid is not None and active_valid <= 0
                if should_log_alignment(current_step, self.rank, sample_index, force=force_alignment_log):
                    log_alignment_event(
                        logger,
                        {
                            "event": "drafter_align_prepare_item",
                            "step": current_step,
                            "rank": self.rank,
                            "sample": sample_index,
                            "item_idx": item_idx,
                            "seq_len": seq_len,
                            "train_seq_len": train_seq_len,
                            "input_len": int(ids.size(0)),
                            "hidden_len": int(h_states.size(0)),
                            "loss_len": int(item_loss_mask.size(0)),
                            "target_len": int(target_logprobs_item.size(0)) if target_logprobs_item is not None else None,
                            "prompt_len": source_item.get("_verl_prompt_len"),
                            "response_len": source_item.get("_verl_response_len"),
                            "feature_start": source_item.get("_verl_feature_start"),
                            "feature_end": source_item.get("_verl_feature_end"),
                            "hidden_start": source_item.get("_verl_hidden_start"),
                            "hidden_end": source_item.get("_verl_hidden_end"),
                            "hidden_position_start": source_item.get("_verl_hidden_position_start"),
                            "target_start": source_item.get("_verl_target_start"),
                            "target_end": source_item.get("_verl_target_end"),
                            "target_position_start": source_item.get("_verl_target_position_start"),
                            "target_position_end": source_item.get("_verl_target_position_end"),
                            "target_train_start": target_logprobs_train_start,
                            "target_tensor_position_start": source_item.get("_verl_target_tensor_position_start"),
                            "target_tensor_position_end": source_item.get("_verl_target_tensor_position_end"),
                            "loss_after_shift": int(
                                (
                                    item_loss_mask[2 : 2 + train_seq_len]
                                    if uses_shifted_eagle_inputs
                                    else item_loss_mask[1 : 1 + train_seq_len]
                                )
                                .detach()
                                .float()
                                .sum()
                                .cpu()
                                .item()
                            ),
                            "target_rows": int(train_target_logprobs.size(0)) if train_target_logprobs is not None else None,
                            "row_valid": row_valid_count,
                            "active_rows": active_rows,
                            "active_valid": active_valid,
                            "target_shape": _tensor_shape(train_target_logprobs),
                        },
                    )
                    window_input_ids = ids
                    window_loss_mask = item_loss_mask
                    window_feature_start = int(source_item.get("_verl_feature_start", 0) or 0)
                    if uses_shifted_eagle_inputs:
                        window_input_ids = ids[1 : 2 + train_seq_len]
                        window_loss_mask = item_loss_mask[1 : 2 + train_seq_len]
                        window_feature_start += 1
                    window_rows = _alignment_window_rows(
                        window_input_ids,
                        window_loss_mask,
                        train_target_logprobs,
                        feature_start=window_feature_start,
                        prompt_len=source_item.get("_verl_prompt_len"),
                        response_len=source_item.get("_verl_response_len"),
                    )
                    if window_rows:
                        log_alignment_event(
                            logger,
                            {
                                "event": "drafter_align_window",
                                "stage": "prepare_item",
                                "step": current_step,
                                "rank": self.rank,
                                "sample": sample_index,
                                "item_idx": item_idx,
                                "rows": window_rows,
                            },
                        )

            if uses_shifted_eagle_inputs:
                input_id_chunks.append(ids[1 : 1 + train_seq_len])
                hidden_state_chunks.append(h_states[:train_seq_len])
                position_id_chunks.append(item_position_ids[:train_seq_len])
            else:
                input_id_chunks.append(ids[:train_seq_len])
                hidden_state_chunks.append(h_states[:train_seq_len])
                position_id_chunks.append(item_position_ids[:train_seq_len])
            if self.backend.model_type == "dflash":
                loss_mask_chunks.append(item_loss_mask[:train_seq_len])
            elif uses_shifted_eagle_inputs:
                loss_mask_chunks.append(item_loss_mask[2 : 2 + train_seq_len])
            else:
                loss_mask_chunks.append(item_loss_mask[1 : 1 + train_seq_len])

            if self.backend.model_type == "eagle3":
                if use_logits:
                    target_logprob_chunks.append(
                        target_logprobs_item[
                            target_logprobs_train_start : target_logprobs_train_start + train_seq_len
                        ]
                    )
                else:
                    last_hidden_state_chunks.append(last_h_states[1 : 1 + train_seq_len])
            elif self.backend.model_type == "eagle":
                target_chunks.append(h_states[1 : 1 + train_seq_len])

        if not input_id_chunks:
            return None

        if self.backend.model_type == "dflash":
            max_train_len = max(chunk.size(0) for chunk in input_id_chunks)
            hidden_dim = hidden_state_chunks[0].size(-1)
            input_ids = torch.zeros(len(input_id_chunks), max_train_len, dtype=input_id_chunks[0].dtype, device=dev)
            loss_mask = torch.zeros(len(loss_mask_chunks), max_train_len, dtype=loss_mask_chunks[0].dtype, device=dev)
            base_h = torch.zeros(
                len(hidden_state_chunks),
                max_train_len,
                hidden_dim,
                dtype=hidden_state_chunks[0].dtype,
                device=dev,
            )
            position_ids = torch.zeros(len(position_id_chunks), max_train_len, dtype=position_id_chunks[0].dtype, device=dev)
            attn_mask = torch.zeros_like(input_ids, dtype=torch.long, device=dev)
            for row_idx, (ids_chunk, mask_chunk, h_chunk, pos_chunk) in enumerate(
                zip(input_id_chunks, loss_mask_chunks, hidden_state_chunks, position_id_chunks)
            ):
                row_len = ids_chunk.size(0)
                input_ids[row_idx, :row_len] = ids_chunk
                loss_mask[row_idx, :row_len] = mask_chunk
                base_h[row_idx, :row_len] = h_chunk
                position_ids[row_idx, :row_len] = pos_chunk
                attn_mask[row_idx, :row_len] = 1
        else:
            input_ids = torch.cat(input_id_chunks, dim=0).unsqueeze(0).contiguous()
            loss_mask = torch.cat(loss_mask_chunks, dim=0).unsqueeze(0).contiguous()
            base_h = torch.cat(hidden_state_chunks, dim=0).unsqueeze(0).contiguous()
            attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)
            position_ids = torch.cat(position_id_chunks, dim=0).unsqueeze(0).contiguous()

        if self.backend.model_type == "eagle3":
            if use_logits:
                if not target_logprob_chunks:
                    return None
                target_logprobs = torch.cat(target_logprob_chunks, dim=0).unsqueeze(0).contiguous()
            else:
                if not last_hidden_state_chunks:
                    return None
                last_hidden_states = torch.cat(last_hidden_state_chunks, dim=0).unsqueeze(0).contiguous()
        elif self.backend.model_type == "eagle":
            if not target_chunks:
                return None
            target = torch.cat(target_chunks, dim=0).unsqueeze(0).contiguous()

        batch = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "hidden_states": base_h,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self.backend.model_type == "eagle3":
            if use_logits:
                batch["target_logprobs"] = target_logprobs
            else:
                batch["last_hidden_states"] = last_hidden_states
        elif self.backend.model_type == "eagle":
            batch["target"] = target

        batch = self._sanitize_training_batch(batch)
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        base_h = batch["hidden_states"]
        loss_mask = batch["loss_mask"]
        position_ids = batch["position_ids"]
        if self.backend.model_type == "eagle3":
            if use_logits:
                target_logprobs = batch["target_logprobs"]
            else:
                last_hidden_states = batch["last_hidden_states"]
        elif self.backend.model_type == "eagle":
            target = batch["target"]

        # Use Ulysses SP to pad and slice if needed
        if self.use_ulysses_sp:
            from verl.utils.ulysses import slice_input_tensor, ulysses_pad_and_slice_inputs
            # Pad to be divisible by SP size and slice across ranks
            input_ids, position_ids, pad_size = ulysses_pad_and_slice_inputs(
                input_ids, position_ids, sp_size=self.ulysses_sequence_parallel_size
            )

            # Pad loss_mask and hidden_states to match
            if pad_size > 0:
                loss_mask = torch.nn.functional.pad(loss_mask, (0, pad_size), value=0.0)
                base_h = torch.nn.functional.pad(base_h, (0, 0, 0, pad_size), value=0.0)
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_size), value=0)
                if self.backend.model_type == "eagle3":
                    if use_logits:
                        pad_shape = list(target_logprobs.shape)
                        pad_shape[1] = pad_size
                        target_logprobs_pad = torch.zeros(
                            pad_shape,
                            dtype=target_logprobs.dtype,
                            device=target_logprobs.device,
                        )
                        target_logprobs_pad[..., 0] = float("-inf")
                        target_logprobs_pad[..., 1] = -1.0
                        target_logprobs = torch.cat((target_logprobs, target_logprobs_pad), dim=1)
                    else:
                        last_hidden_states = torch.nn.functional.pad(
                            last_hidden_states, (0, 0, 0, pad_size), value=0.0
                        )
                elif self.backend.model_type == "eagle":
                    target = torch.nn.functional.pad(target, (0, 0, 0, pad_size), value=0.0)

            # Slice for this rank
            loss_mask = slice_input_tensor(loss_mask, dim=1, padding=False)
            base_h = slice_input_tensor(base_h, dim=1, padding=False)
            attn_mask = slice_input_tensor(attn_mask, dim=1, padding=False)
            if self.backend.model_type == "eagle3":
                if use_logits:
                    target_logprobs = slice_input_tensor(target_logprobs, dim=1, padding=False)
                else:
                    last_hidden_states = slice_input_tensor(last_hidden_states, dim=1, padding=False)
            elif self.backend.model_type == "eagle":
                target = slice_input_tensor(target, dim=1, padding=False)

            # Store pad_size for later gathering
            self._current_pad_size = pad_size
        else:
            self._current_pad_size = 0

        batch = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "hidden_states": base_h,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if self.backend.model_type == "eagle3":
            if use_logits:
                batch["target_logprobs"] = target_logprobs
            else:
                batch["last_hidden_states"] = last_hidden_states
        elif self.backend.model_type == "eagle":
            batch["target"] = target

        if alignment_debug_enabled():
            final_target = None
            if self.backend.model_type == "eagle3" and use_logits:
                final_target = batch.get("target_logprobs")
            elif self.backend.model_type == "eagle3":
                final_target = batch.get("last_hidden_states")
            elif self.backend.model_type == "eagle":
                final_target = batch.get("target")

            final_loss_mask = batch["loss_mask"]
            final_target_rows = int(final_target.size(1)) if torch.is_tensor(final_target) and final_target.dim() >= 2 else None
            final_row_valid = None
            final_active_rows = None
            final_active_valid = None
            if torch.is_tensor(final_target) and final_target.dim() >= 3 and final_target_rows is not None:
                if self.backend.model_type == "eagle3" and use_logits:
                    final_row_valid_mask = _target_row_valid_mask(final_target.squeeze(0))
                    if final_row_valid_mask is not None and final_row_valid_mask.numel() > 0:
                        final_loss_mask_slice = final_loss_mask.squeeze(0)[: final_row_valid_mask.size(0)].bool()
                        final_row_valid = int(final_row_valid_mask.detach().sum().cpu().item())
                        final_active_rows = int(final_loss_mask_slice.detach().sum().cpu().item())
                        final_active_valid = int((final_row_valid_mask & final_loss_mask_slice).detach().sum().cpu().item())
            force_alignment_log = final_active_valid is not None and final_active_valid <= 0
            sample_index = self._alignment_debug_sample_index(current_step, "prepare_summary")
            if should_log_alignment(current_step, self.rank, sample_index, force=force_alignment_log):
                log_alignment_event(
                    logger,
                    {
                        "event": "drafter_align_prepare",
                        "step": current_step,
                        "rank": self.rank,
                        "items_seen": items_seen,
                        "items_used": items_used,
                        "items_dropped_short": items_dropped_short,
                        "items_dropped_missing_target": items_dropped_missing_target,
                        "packed_tokens_before_shift": packed_tokens_before_shift,
                        "packed_loss_tokens": packed_loss_tokens,
                        "input_shape": _tensor_shape(batch["input_ids"]),
                        "hidden_shape": _tensor_shape(batch["hidden_states"]),
                        "loss_shape": _tensor_shape(final_loss_mask),
                        "target_shape": _tensor_shape(final_target),
                        "target_rows": final_target_rows,
                        "row_valid": final_row_valid,
                        "active_rows": final_active_rows,
                        "active_valid": final_active_valid,
                        "loss_sum": int(final_loss_mask.detach().float().sum().cpu().item()),
                    },
                )

        return batch

    def _sync_batch_readiness(self, has_local_batch: bool) -> bool:
        dp_group = self._get_dp_group()
        if dp_group is None or self._get_dp_world_size() <= 1:
            return has_local_batch

        readiness = torch.tensor(
            [1 if has_local_batch else 0],
            device=self.runtime_device,
            dtype=torch.int32,
        )
        dist.all_reduce(readiness, op=dist.ReduceOp.MIN, group=dp_group)
        return bool(readiness.item())

    async def training_step(self, step: int) -> bool:
        try:
            with torch.enable_grad():
                return await self._training_step_impl(step)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Training step {step} failed with error: {e}")
            return False

    @contextmanager
    def _ulysses_group_context(self):
        sp_group = self._get_sp_group()
        if not self.use_ulysses_sp or sp_group is None:
            with nullcontext():
                yield
            return

        prev_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(sp_group)
        try:
            yield
        finally:
            set_ulysses_sequence_parallel_group(prev_group)
        
    async def _training_step_impl(self, step: int) -> bool:
        """Execute a single training step."""
        if not self.model:
            logger.warning("No model available for training")
            return False

        # Skip training if we're not collecting hidden states (since we can't train without them)
        collect_hidden_states_from_sgl = bool(self.config.rollout.drafter.training.get("collect_hidden_states_from_sgl", False))
        if not collect_hidden_states_from_sgl:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Skipping training step {step} "
                f"because collect_hidden_states_from_sgl=False"
            )
            return False

        with self._ulysses_group_context():
            batch = self._prepare_training_batch()
        if not self._sync_batch_readiness(batch is not None):
            logger.debug(f"[EagleTrainer rank {self.rank}] Skipping step {step} due to missing drafter batch")
            return False
        if batch is None:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Not enough data at step {step} "
                f"(have={len(self.collected_data)} need>={self.batch_size})"
            )
            return False
        
        # 开启训练模式
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 前向传播
        with self._ulysses_group_context():
            with torch.amp.autocast(device_type=device_name, dtype=torch.bfloat16):
                loss_dict = self.backend.compute_loss(self.model, batch, self._current_pad_size)

                l_v = loss_dict["total_local_vloss"]
                l_p = loss_dict["total_local_ploss"]
                l_n = loss_dict["local_num_tokens"]

        # 分布式同步（Global Reduction）,如果使用序列并行，仅在这里进行一次标量同步
        sp_group = self._get_sp_group()
        dp_group = self._get_dp_group()
        if sp_group is not None and self._get_sp_world_size() > 1:
            metrics = torch.stack([l_v, l_p, l_n])
            dist.all_reduce(metrics, group=sp_group)
            if dp_group is not None and self._get_dp_world_size() > 1:
                dist.all_reduce(metrics, group=dp_group)
            global_vloss, global_ploss, global_tokens = metrics[0], metrics[1], metrics[2]
        elif dp_group is not None and self._get_dp_world_size() > 1:
            metrics = torch.stack([l_v, l_p, l_n])
            dist.all_reduce(metrics, group=dp_group)
            global_vloss, global_ploss, global_tokens = metrics[0], metrics[1], metrics[2]
        elif self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            metrics = torch.stack([l_v, l_p, l_n])
            dist.all_reduce(metrics, group=self.training_device_mesh.get_group())
            global_vloss, global_ploss, global_tokens = metrics[0], metrics[1], metrics[2]
        else:
            global_vloss, global_ploss, global_tokens = l_v, l_p, l_n
        
        # 最终 Loss 平滑处理
        if float(global_tokens.detach().float().item()) <= 0:
            logger.warning(
                f"Step {self.training_steps + 1}: no finite drafter target tokens, skipping optimizer step"
            )
            return False

        denom = global_tokens.clamp(min=1.0)
        vloss = global_vloss / denom
        ploss = global_ploss / denom

        # 使用 backend 传回的权重合成最终 Loss
        loss = loss_dict["v_weight"] * vloss + loss_dict["p_weight"] * ploss
        if not torch.isfinite(loss):
            logger.error(
                f"Step {self.training_steps + 1}: non-finite drafter loss, "
                f"loss={float(loss.detach().float().item())}, "
                f"vloss={float(vloss.detach().float().item())}, "
                f"ploss={float(ploss.detach().float().item())}, "
                f"tokens={float(global_tokens.detach().float().item())}"
            )
            return False

        # 反向传播
        loss.backward()

        # 更新权重
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            logger.error(
                f"Step {self.training_steps + 1}: non-finite drafter grad norm, "
                f"grad_norm={float(grad_norm.detach().float().item())}"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return False
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.training_steps += 1
        if self.training_steps % 10 == 0:
            logger.info(
                f"Step {self.training_steps}: loss={float(loss.item()):.4f}, vloss={float(vloss.item()):.4f}, ploss={float(ploss.item()):.4f}"
            )
        return True
    
    def increment_rl_step(self, global_step: Optional[int] = None):
        """Increment the RL step counter in the data buffer.

        Should be called at the end of each RL training step to mark the boundary.
        """
        previous_step = self.current_rl_step
        if global_step is None:
            self.current_rl_step += 1
        else:
            self.current_rl_step = int(global_step)
        if not self.use_data_buffer and self.current_rl_step != previous_step:
            self.collected_data.clear()
        self.data_buffer.update_rl_step(self.current_rl_step)
        logger.debug(
            f"[Rank {self.rank}] DataBuffer RL step incremented to {self.data_buffer.get_current_step()}, "
            f"total samples: {len(self.data_buffer)}"
        )
    
    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Get trainable model state dict (excluding frozen layers)."""
        if not self.model:
            return None

        first_param = next(self.model.parameters(), None)
        was_on_device = first_param is not None and first_param.device.type == device_name
        is_fsdp_wrapped = isinstance(self.model, FSDP) or self.training_device_mesh is not None
        if is_fsdp_wrapped and not was_on_device:
            load_fsdp_model_to_gpu(self.model)

        try:
            trainable_state = self._get_trainable_state_dict()
            if not trainable_state:
                return None
            training_cfg = self.config.rollout.drafter.training
            patterns = training_cfg.get("publish_param_name_patterns", None)
            if isinstance(patterns, str):
                patterns = [patterns]
            if patterns:
                trainable_state = {
                    k: v
                    for k, v in trainable_state.items()
                    if any(fnmatch.fnmatch(k, pattern) for pattern in patterns)
                }
                if not trainable_state:
                    logger.warning(
                        "[Rank %s] No drafter parameters matched publish_param_name_patterns=%s",
                        self.rank,
                        patterns,
                    )
                    return None

            publish_dtype = training_cfg.get("publish_dtype", None)
            dtype = None
            if publish_dtype in {"float32", "fp32"}:
                dtype = torch.float32
            elif publish_dtype in {"float16", "fp16"}:
                dtype = torch.float16
            elif publish_dtype in {"bfloat16", "bf16"}:
                dtype = torch.bfloat16

            return {
                k: (
                    v.detach().to(dtype=dtype, device="cpu").contiguous()
                    if dtype is not None
                    else v.detach().cpu().contiguous()
                )
                for k, v in trainable_state.items()
            }
        finally:
            if is_fsdp_wrapped and not was_on_device:
                offload_fsdp_model_to_cpu(self.model)

    def clear_pending_publish_state_dict(self) -> None:
        self._pending_publish_state_dict = None
        self._pending_publish_step = None
        self._pending_publish_ready = False

    def prepare_model_state_dict_for_publish(self, global_step: Optional[int]) -> bool:
        """Snapshot trainable drafter weights before cleanup/offload for fast publish."""
        self.clear_pending_publish_state_dict()
        step = int(global_step) if global_step is not None else None
        state_dict = self.get_model_state_dict()
        self._pending_publish_state_dict = state_dict
        self._pending_publish_step = step
        self._pending_publish_ready = True
        return bool(state_dict)

    def pop_model_state_dict_for_publish(
        self,
        global_step: Optional[int],
    ) -> tuple[bool, Optional[dict[str, torch.Tensor]]]:
        if not self._pending_publish_ready:
            return False, None

        step = int(global_step) if global_step is not None else None
        if self._pending_publish_step != step:
            logger.warning(
                "[Rank %s] Drop stale drafter publish snapshot: snapshot_step=%s, requested_step=%s",
                self.rank,
                self._pending_publish_step,
                step,
            )
            self.clear_pending_publish_state_dict()
            return False, None

        state_dict = self._pending_publish_state_dict
        self.clear_pending_publish_state_dict()
        return True, state_dict
    
    async def cleanup_training(self, clear_data: bool = True):
        # First set training as inactive to prevent further steps
        self._training_active = False

        # Wait for any pending async checkpoint save to complete
        if self._pending_checkpoint_future is not None:
            logger.debug(f"[Rank {self.rank}] Waiting for pending checkpoint save to complete...")
            try:
                # Run the blocking .result() call in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._pending_checkpoint_future.result)
                logger.debug(f"[Rank {self.rank}] Pending checkpoint save completed")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Pending checkpoint save failed: {e}")
            self._pending_checkpoint_future = None
        if self._pending_full_checkpoint_future is not None:
            logger.debug(f"[Rank {self.rank}] Waiting for pending full drafter checkpoint save to complete...")
            try:
                await asyncio.wrap_future(self._pending_full_checkpoint_future)
                logger.debug(f"[Rank {self.rank}] Pending full drafter checkpoint save completed")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Pending full drafter checkpoint save failed: {e}")
            self._pending_full_checkpoint_future = None

        # Save final checkpoint and wait for it to complete
        if self.checkpoint_dir and self.model is not None and self.training_steps > 0:
            final_ckpt_step = self.current_rl_step if self.current_rl_step > 0 else self.training_steps
            final_future = self._save_checkpoint_async(final_ckpt_step, is_final=True)
            if final_future is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, final_future.result)
                    logger.info(f"[Rank {self.rank}] Final checkpoint save completed")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Final checkpoint save failed: {e}")
            if self._pending_full_checkpoint_future is not None:
                try:
                    await asyncio.wrap_future(self._pending_full_checkpoint_future)
                    logger.info(f"[Rank {self.rank}] Final full drafter checkpoint save completed")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Final full drafter checkpoint save failed: {e}")
                self._pending_full_checkpoint_future = None

        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to clear drafter gradients during cleanup: {e}")

        # Clean up distributed resources gracefully
        sp_group = self._get_sp_group()
        dp_group = self._get_dp_group()
        if sp_group is not None and self._get_sp_world_size() > 1:
            try:
                await asyncio.sleep(0.1)
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: torch.distributed.barrier(sp_group)
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Rank {self.rank} subgroup barrier timeout during cleanup, continuing anyway")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Subgroup cleanup error (expected): {e}")
        if dp_group is not None and self._get_dp_world_size() > 1:
            try:
                await asyncio.sleep(0.1)
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: torch.distributed.barrier(dp_group)
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Rank {self.rank} dp-group barrier timeout during cleanup, continuing anyway")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"DP-group cleanup error (expected): {e}")
        elif self.training_device_mesh is not None:
            try:
                # Give a moment for any pending operations to complete
                await asyncio.sleep(0.1)
                if self.training_device_mesh.size() > 1:
                    # Try to destroy the process group if possible
                    try:
                        # Run barrier with timeout to avoid hanging
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, lambda: torch.distributed.barrier(self.training_device_mesh.get_group())
                            ),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Rank {self.rank} barrier timeout during cleanup, continuing anyway")
                    except Exception:
                        pass  # Ignore barrier errors during cleanup
            except Exception as e:
                logger.debug(f"Process group cleanup error (expected): {e}")

        if self.model is not None:
            try:
                offload_fsdp_model_to_cpu(self.model)
                logger.debug("Offloaded drafter model to CPU after training")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter model during cleanup: {e}")

        if self.optimizer is not None:
            try:
                offload_fsdp_optimizer(self.optimizer)
                logger.debug("Offloaded drafter optimizer state to CPU after training")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter optimizer during cleanup: {e}")

        target_model = getattr(self.backend, "target_model", None)
        if target_model is not None:
            try:
                target_model.to("cpu")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter target model during cleanup: {e}")
        
        if clear_data:
            self.collected_data.clear()
            self.data_buffer.clear()  # Clear the cross-step data buffer
        if self._full_checkpoint_executor is not None:
            self._full_checkpoint_executor.shutdown(wait=False)
            self._full_checkpoint_executor = None
        if device_name != "cpu" and hasattr(self.device_module, "empty_cache"):
            if hasattr(self.device_module, "synchronize"):
                self.device_module.synchronize()
            self.device_module.empty_cache()
        self._training_initialized = False
        self._last_ckpt_step = -1
        self.training_steps = 0
