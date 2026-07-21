import logging
import json
import os
import glob
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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.nn import SmoothL1Loss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.device import get_device_name, get_torch_device
from verl_speco.trainer.data_buffer import DataBuffer
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
from verl_speco.trainer.feature_store import DraftFeatureSample

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

_ALIGNMENT_DEBUG_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG"
_ALIGNMENT_DEBUG_EVERY_N_STEPS_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_EVERY_N_STEPS"
_ALIGNMENT_DEBUG_MAX_SAMPLES_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_MAX_SAMPLES_PER_STEP"
_ALIGNMENT_DEBUG_TOKEN_WINDOW_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_TOKEN_WINDOW"
_ALIGNMENT_DEBUG_RANKS_ENV = "VERL_DRAFTER_ALIGNMENT_DEBUG_RANKS"
_LAST_HIDDEN_LOGPROB_CHECK_ENV = "VERL_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK"


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


def last_hidden_logprob_check_enabled() -> bool:
    return _env_flag_enabled(_LAST_HIDDEN_LOGPROB_CHECK_ENV, default=False)


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


class _DrafterOptimizerState:
    """DCP Stateful adapter for FSDP1/FSDP2 optimizer state dicts."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @staticmethod
    def _options():
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        return StateDictOptions(full_state_dict=False, cpu_offload=True)

    def state_dict(self):
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        return get_optimizer_state_dict(self.model, self.optimizer, options=self._options())

    def load_state_dict(self, state_dict):
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict,
            options=self._options(),
        )


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


def _batch_item_float(value: Any, index: int = 0) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        flat = value.detach().view(-1).float().cpu()
        index = min(max(int(index), 0), flat.numel() - 1)
        return float(flat[index].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        index = min(max(int(index), 0), len(value) - 1)
        value = value[index]
    try:
        return float(value)
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
    hidden_positions = source_item.get("hidden_positions")
    if isinstance(hidden_positions, torch.Tensor) and int(hidden_positions.numel()) > 0:
        feature_start = int(hidden_positions.reshape(-1)[0].item())
    else:
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

        training_cfg = config.rollout.drafter.training
        self.is_offload_param = bool(training_cfg.get("is_offload_param", False))
        self.is_offload_optimizer = bool(training_cfg.get("is_offload_optimizer", False))
        self.skip_heavy_cleanup_after_drafter_training = bool(
            training_cfg.get("skip_heavy_cleanup_after_drafter_training", False)
        )
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0
        self.optimizer_steps_total = 0
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
        self._pending_target_lm_head_row_indices = None
        self._pending_target_lm_head_source_vocab_size = None
        self._target_lm_head_weight_step = None
        self._cached_target_lm_head_row_indices = None
        self._training_timing_accumulator = {}
        self._training_timing_steps = 0
        self._training_metric_sums = {}
        self._training_metric_steps = 0
        self._frozen_param_names = {"model.embed_tokens.weight"}

        # Ulysses Sequence Parallelism configuration. EAGLE3 can slice
        # token-wise training tensors by the rollout TP size, but DFlash anchor
        # sampling needs full local sequences and does not implement SP loss yet.
        rollout_tp_size = int(self.config.rollout.get("tensor_model_parallel_size", 1))
        if self._is_block_drafter_backend():
            self.ulysses_sequence_parallel_size = 1
            if rollout_tp_size > 1 and self.training_group_world_size > 1:
                logger.debug(
                    "[Rank %s] Disable Ulysses SP for %s drafter training: "
                    "rollout_tp=%s training_group_world_size=%s",
                    self.rank,
                    self.backend.model_type,
                    rollout_tp_size,
                    self.training_group_world_size,
                )
        elif not getattr(self.backend, "supports_ulysses_sp", True):
            # Backends that do not implement the SP loss path (e.g. EAGLE-1/2)
            # opt out here so SP is never enabled under them, rather than aborting
            # in compute_loss when rollout_tp > 1 on a multi-rank training group.
            self.ulysses_sequence_parallel_size = 1
            if rollout_tp_size > 1 and self.training_group_world_size > 1:
                logger.debug(
                    "[Rank %s] Disable Ulysses SP for %s drafter training: "
                    "rollout_tp=%s training_group_world_size=%s",
                    self.rank,
                    self.config.rollout.drafter.get("speculative_algorithm", self.backend.model_type),
                    rollout_tp_size,
                    self.training_group_world_size,
                )
        else:
            self.ulysses_sequence_parallel_size = min(
                rollout_tp_size,
                self.training_group_world_size,
            )
        self.use_ulysses_sp = self.training_group_world_size > 1 and self.ulysses_sequence_parallel_size > 1
        setattr(self.backend, "use_ulysses_sp", self.use_ulysses_sp)
        self.use_native_dp_sp = self.training_group_world_size > 1 and self.dp_group_world_size > 1

        self.checkpoint_dir = self.config.rollout.drafter.get("checkpoint_path")
        self.step = self.config.rollout.drafter.training.step

    def _optimizer_state_on_runtime_device(self) -> bool:
        if self.optimizer is None:
            return True
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for value in state.values():
                if torch.is_tensor(value) and value.device.type != self.runtime_device.type:
                    return False
        return True

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

    def _is_block_drafter_backend(self) -> bool:
        return getattr(self.backend, "model_type", None) in {"dflash", "dspark", "domino"}

    def _block_drafter_metric_prefix(self) -> str:
        model_type = str(getattr(self.backend, "model_type", "dflash") or "dflash")
        if model_type in {"dspark", "domino"}:
            return model_type
        return "dflash"

    def _block_drafter_config_value(self, suffix: str, default: Any) -> Any:
        training_cfg = self.config.rollout.drafter.training
        prefix = self._block_drafter_metric_prefix()
        value = training_cfg.get(f"{prefix}_{suffix}", None)
        if value is not None:
            return value
        return training_cfg.get(f"dflash_{suffix}", default)

    def reset_training_metrics(self) -> None:
        self._training_metric_sums = {}
        self._training_metric_steps = 0

    def get_training_metrics(self) -> dict[str, float]:
        sums = dict(self._training_metric_sums)
        steps = max(int(self._training_metric_steps), 1)
        metrics: dict[str, float] = {}
        for key, value in sums.items():
            if key.startswith("timing_s/drafter_"):
                metrics[key] = value
        prefix = self._block_drafter_metric_prefix()
        correct = sums.get(f"{prefix}/correct_count", 0.0)
        eval_tokens = sums.get(f"{prefix}/eval_token_count", 0.0)
        if eval_tokens > 0:
            metrics[f"{prefix}/accuracy"] = correct / eval_tokens
        quality_tokens = sums.get(f"{prefix}/quality_token_count", 0.0)
        if quality_tokens > 0:
            metrics[f"{prefix}/top1_acc"] = sums.get(f"{prefix}/top1_correct_count", 0.0) / quality_tokens
            metrics[f"{prefix}/top5_acc"] = sums.get(f"{prefix}/top5_correct_count", 0.0) / quality_tokens
        ce_tokens = sums.get(f"{prefix}/ce_weighted_token_count", 0.0)
        if ce_tokens > 0:
            metrics[f"{prefix}/ce_loss"] = sums.get(f"{prefix}/ce_loss_sum", 0.0) / ce_tokens
        l1_tokens = sums.get(f"{prefix}/l1_weighted_token_count", 0.0)
        if l1_tokens > 0:
            metrics[f"{prefix}/l1_loss"] = sums.get(f"{prefix}/l1_loss_sum", 0.0) / l1_tokens
        for key in (
            f"{prefix}/valid_token_count",
            f"{prefix}/weighted_token_count",
            f"{prefix}/ce_weighted_token_count",
            f"{prefix}/l1_weighted_token_count",
            f"{prefix}/quality_token_count",
            f"{prefix}/sanitized_rows",
            f"{prefix}/masked_rows",
            f"{prefix}/sampled_vocab_size",
            f"{prefix}/loss_mode_id",
        ):
            if key in sums:
                metrics[key] = sums[key] / steps
        metrics[f"{prefix}/metric_steps"] = float(self._training_metric_steps)
        metrics["drafter/optimizer_steps_total"] = float(self.optimizer_steps_total)
        if self.optimizer is not None and self.optimizer.param_groups:
            metrics["drafter/current_lr"] = float(self.optimizer.param_groups[0]["lr"])

        for pos in range(int(self._block_drafter_config_value("block_size", 16))):
            count_key = f"{prefix}/count_per_position/{pos}"
            count = sums.get(count_key, 0.0)
            if count <= 0:
                continue
            metrics[count_key] = count / steps
            loss_sum = sums.get(f"{prefix}/loss_sum_per_position/{pos}", 0.0)
            correct_sum = sums.get(f"{prefix}/correct_per_position/{pos}", 0.0)
            metrics[f"{prefix}/loss_per_position/{pos}"] = loss_sum / count
            metrics[f"{prefix}/accuracy_per_position/{pos}"] = correct_sum / count
        return metrics

    def record_training_timing(self, metric_name: str, elapsed_sec: float) -> None:
        if elapsed_sec < 0:
            return
        key = metric_name if metric_name.startswith("timing_s/") else f"timing_s/{metric_name}"
        self._training_metric_sums[key] = self._training_metric_sums.get(key, 0.0) + float(elapsed_sec)

    def _reduce_training_metric(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        sp_group = self._get_sp_group()
        dp_group = self._get_dp_group()
        if sp_group is not None and self._get_sp_world_size() > 1:
            dist.all_reduce(reduced, group=sp_group)
            if dp_group is not None and self._get_dp_world_size() > 1:
                dist.all_reduce(reduced, group=dp_group)
        elif dp_group is not None and self._get_dp_world_size() > 1:
            dist.all_reduce(reduced, group=dp_group)
        elif self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            dist.all_reduce(reduced, group=self.training_device_mesh.get_group())
        return reduced

    def _record_dflash_training_metrics(self, loss_dict: dict[str, Any]) -> None:
        diagnostics = loss_dict.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return
        self._training_metric_steps += 1
        prefix = self._block_drafter_metric_prefix()
        scalar_keys = {
            "correct_count": f"{prefix}/correct_count",
            "eval_token_count": f"{prefix}/eval_token_count",
            "top1_correct_count": f"{prefix}/top1_correct_count",
            "top5_correct_count": f"{prefix}/top5_correct_count",
            "quality_token_count": f"{prefix}/quality_token_count",
            "valid_token_count": f"{prefix}/valid_token_count",
            "weighted_token_count": f"{prefix}/weighted_token_count",
            "ce_loss_sum": f"{prefix}/ce_loss_sum",
            "ce_weighted_token_count": f"{prefix}/ce_weighted_token_count",
            "l1_loss_sum": f"{prefix}/l1_loss_sum",
            "l1_weighted_token_count": f"{prefix}/l1_weighted_token_count",
            "sanitized_rows": f"{prefix}/sanitized_rows",
            "masked_rows": f"{prefix}/masked_rows",
            "sampled_vocab_size": f"{prefix}/sampled_vocab_size",
            "loss_mode_id": f"{prefix}/loss_mode_id",
        }
        for source_key, metric_key in scalar_keys.items():
            value = diagnostics.get(source_key)
            if not torch.is_tensor(value):
                continue
            reduced = self._reduce_training_metric(value.reshape(()))
            self._training_metric_sums[metric_key] = self._training_metric_sums.get(metric_key, 0.0) + float(
                reduced.cpu().item()
            )

        vector_keys = {
            "loss_sum_per_position": f"{prefix}/loss_sum_per_position",
            "correct_per_position": f"{prefix}/correct_per_position",
            "count_per_position": f"{prefix}/count_per_position",
        }
        for source_key, metric_prefix in vector_keys.items():
            value = diagnostics.get(source_key)
            if not torch.is_tensor(value):
                continue
            reduced = self._reduce_training_metric(value)
            for idx, item in enumerate(reduced.detach().cpu().tolist()):
                metric_key = f"{metric_prefix}/{idx}"
                self._training_metric_sums[metric_key] = self._training_metric_sums.get(metric_key, 0.0) + float(item)

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
        logger.debug(f"[Rank {self.rollout_dp_rank}] Building drafter model...")
        # A. 实例化模型（委托给backend）
        pending_target_weight = self._pending_target_lm_head_weight
        if (
            getattr(self.backend, "model_type", None) in {"eagle3", "dflash", "dspark", "domino"}
            and torch.is_tensor(pending_target_weight)
            and pending_target_weight.dim() == 2
        ):
            setattr(self.backend, "_initial_target_lm_head_shape", tuple(pending_target_weight.shape))
        else:
            setattr(self.backend, "_initial_target_lm_head_shape", None)
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
            logger.debug("Building drafter model with mesh-centered dp x sp FSDP2")

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
                logger.debug("Building drafter model with native dp x sp hybrid-shard FSDP")
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
                process_group = (self.training_process_group, self.data_parallel_process_group)
            else:
                logger.debug("Building drafter model with subgroup FSDP")
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
            logger.debug("Building drafter model without FSDP (standalone/local trainer mode)")
            self.model = raw_model

        # D. 构建优化器和调度器
        drafter_train_config = self._prepare_training_config(self.config.rollout)
        setattr(self.backend, "train_config", drafter_train_config)

        resume_metadata = {}
        resume_state = {}
        resume_setting = drafter_train_config.get("resume_trainer_state_from_checkpoint", None)
        if resume_setting is None:
            resume_setting = drafter_train_config.get("resume_lr_scheduler_from_checkpoint", True)
        resume_enabled = bool(resume_setting)
        spec_model_path = self.config.rollout.drafter.get("model_path")
        if resume_enabled:
            from verl_speco.trainer.checkpoint import (
                get_drafter_checkpoint_step,
                get_drafter_checkpoint_metadata,
                get_drafter_trainer_state,
            )

            resume_metadata = get_drafter_checkpoint_metadata(spec_model_path)
            resume_state = get_drafter_trainer_state(spec_model_path)
            if not resume_state and get_drafter_checkpoint_step(spec_model_path) is not None:
                logger.warning(
                    "[drafter resume] checkpoint=%s has no trainer_state; optimizer and LR scheduler start fresh",
                    spec_model_path,
                )

        resume_optimizer_steps = max(int(resume_state.get("optimizer_steps_total", 0) or 0), 0)
        resume_training_steps = max(int(resume_state.get("training_steps", resume_optimizer_steps) or 0), 0)
        with open_dict(drafter_train_config):
            drafter_train_config["_resume_optimizer_steps"] = resume_optimizer_steps

        self.optimizer = self.backend.setup_optimizer(self.model, drafter_train_config)
        restored_trainer_state = None
        restored_optimizer_lrs = None
        if resume_enabled and resume_metadata.get("optimizer") is not None:
            restored_trainer_state = self._load_optimizer_checkpoint(spec_model_path, resume_metadata)
            restored_optimizer_lrs = [float(param_group["lr"]) for param_group in self.optimizer.param_groups]
        self.lr_scheduler = self.backend.setup_scheduler(self.optimizer, drafter_train_config)
        if restored_trainer_state is not None:
            scheduler_state = restored_trainer_state.get("lr_scheduler")
            if self.lr_scheduler is not None and isinstance(scheduler_state, dict):
                self.lr_scheduler.load_state_dict(scheduler_state)
                restored_optimizer_lrs = [float(lr) for lr in self.lr_scheduler.get_last_lr()]
            if restored_optimizer_lrs is not None:
                if len(restored_optimizer_lrs) != len(self.optimizer.param_groups):
                    raise RuntimeError(
                        "Drafter optimizer checkpoint parameter-group count does not match the current optimizer"
                    )
                for param_group, restored_lr in zip(
                    self.optimizer.param_groups,
                    restored_optimizer_lrs,
                    strict=True,
                ):
                    param_group["lr"] = restored_lr
            for key, expected in (
                ("optimizer_steps_total", resume_optimizer_steps),
                ("training_steps", resume_training_steps),
            ):
                restored_value = int(restored_trainer_state.get(key, expected) or 0)
                if restored_value != expected:
                    raise RuntimeError(
                        f"Drafter checkpoint trainer state mismatch for {key}: "
                        f"metadata={expected} optimizer_checkpoint={restored_value}"
                    )
        self.optimizer_steps_total = resume_optimizer_steps
        self.training_steps = resume_training_steps
        self.drafter_train_config = drafter_train_config
        self.model_config = drafter_model_config
        self.pad_token_id = int(getattr(drafter_model_config, "pad_token_id", self.pad_token_id) or self.pad_token_id)
        self._apply_pending_target_lm_head_weight()
        if resume_optimizer_steps > 0 or restored_trainer_state is not None:
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            logger.info(
                "[drafter resume] checkpoint=%s optimizer_steps_total=%s training_steps=%s "
                "scheduler_last_epoch=%s lr=%.3e optimizer_state=%s",
                spec_model_path,
                self.optimizer_steps_total,
                self.training_steps,
                getattr(self.lr_scheduler, "last_epoch", None),
                current_lr,
                "restored" if restored_trainer_state is not None else "fresh",
            )
        
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

    def _is_frozen_publish_param(self, name: str) -> bool:
        """Return whether a drafter state entry should be excluded from hot publish."""

        if any(frozen_name in name for frozen_name in self._frozen_param_names):
            return True
        return name == "embed_tokens.weight" or name.endswith(".embed_tokens.weight")

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
            # EAGLE3 trains and publishes its own lm_head; other backends skip lm_head by default.
            if self._is_frozen_publish_param(name) or (
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

    def _get_pretrained_export_model(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        # Trainer wrappers (the DFlash family and P-EAGLE) keep the exportable
        # draft as ``draft_model``; the checkpoint must hold the draft itself, so
        # unwrap whenever a wrapper is present rather than per backend type.
        draft_model = getattr(model, "draft_model", None)
        if draft_model is not None:
            return draft_model, ("draft_model.", "module.draft_model.", "_orig_mod.draft_model.")
        return model, ()

    def _get_pretrained_export_state_dict(self) -> dict[str, torch.Tensor]:
        full_state_dict = self._get_full_export_state_dict()
        if not full_state_dict:
            return {}

        _, strip_prefixes = self._get_pretrained_export_model()
        if not strip_prefixes:
            return full_state_dict

        stripped_state_dict = {}
        for name, tensor in full_state_dict.items():
            stripped_name = None
            for prefix in strip_prefixes:
                if name.startswith(prefix):
                    stripped_name = name[len(prefix) :]
                    break
            if stripped_name is not None:
                stripped_state_dict[stripped_name] = tensor

        return stripped_state_dict or full_state_dict

    def _is_checkpoint_leader(self) -> bool:
        return self.rollout_dp_rank == 0 and self._get_sp_local_rank() == 0

    def _infer_pretrained_save_kwargs(self) -> dict[str, Any]:
        # Save as HuggingFace-compatible PyTorch weights so SGLang and the
        # drafter trainer can load the checkpoint directory directly.
        save_kwargs: dict[str, Any] = {"safe_serialization": False}
        spec_model_path = self.config.rollout.drafter.model_path
        if not spec_model_path or not os.path.isdir(spec_model_path):
            return save_kwargs

        shard_files = []
        for index_path in sorted(glob.glob(os.path.join(spec_model_path, "*.index.json"))):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_json = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to inspect drafter weight index %s: %s", index_path, exc)
                continue
            shard_files.extend(index_json.get("weight_map", {}).values())

        if shard_files:
            unique_shards = sorted(set(shard_files))
            shard_sizes = []
            for shard in unique_shards:
                shard_path = os.path.join(spec_model_path, shard)
                if os.path.exists(shard_path):
                    shard_sizes.append(os.path.getsize(shard_path))
            if len(unique_shards) > 1 and shard_sizes:
                save_kwargs["max_shard_size"] = max(shard_sizes)
        return save_kwargs

    def _copy_drafter_auxiliary_files(self, output_dir: str) -> None:
        spec_model_path = self.config.rollout.drafter.model_path
        if not spec_model_path or not os.path.isdir(spec_model_path):
            return
        for filename in ("generation_config.json",):
            src = os.path.join(spec_model_path, filename)
            dst = os.path.join(output_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    logger.warning("Failed to copy drafter %s to full checkpoint: %s", filename, exc)

    def _clear_existing_pretrained_weight_files(self, output_dir: str) -> None:
        for pattern in (
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "pytorch_model.bin.index.json",
            "model.pt",
        ):
            for path in glob.glob(os.path.join(output_dir, pattern)):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Failed to remove stale drafter checkpoint file %s: %s", path, exc)

    @staticmethod
    def _atomic_torch_save(payload: Any, output_path: str) -> None:
        temporary_path = f"{output_path}.tmp-{os.getpid()}"
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @staticmethod
    def _atomic_json_dump(payload: dict[str, Any], output_path: str) -> None:
        temporary_path = f"{output_path}.tmp-{os.getpid()}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _checkpoint_process_group(self):
        if not dist.is_initialized():
            return None
        group = self._get_sp_group()
        if group is None:
            raise RuntimeError("Drafter checkpoint requires an explicit SP process group")
        return group

    def _sync_checkpoint_phase_error(self, error: Optional[BaseException], phase: str) -> None:
        group = self._checkpoint_process_group()
        if group is None:
            if error is not None:
                raise RuntimeError(f"Drafter checkpoint {phase} failed: {error}") from error
            return

        failure = torch.tensor([1 if error is not None else 0], dtype=torch.int32, device=self.runtime_device)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
        if int(failure.item()) != 0:
            if error is not None:
                raise RuntimeError(f"Drafter checkpoint {phase} failed: {error}") from error
            raise RuntimeError(f"Drafter checkpoint {phase} failed on another SP rank")

    def _save_optimizer_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("Cannot save drafter optimizer before it is initialized")

        optimizer_path = os.path.join(checkpoint_path, "optimizer")
        temporary_path = os.path.join(checkpoint_path, "optimizer.incomplete")
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        trainer_state_file = "trainer_state.pt"
        is_leader = self._is_checkpoint_leader()

        prepare_error = None
        if is_leader:
            try:
                os.makedirs(checkpoint_path, exist_ok=True)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                shutil.rmtree(temporary_path, ignore_errors=True)
                shutil.rmtree(optimizer_path, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                prepare_error = exc
        self._sync_checkpoint_phase_error(prepare_error, "prepare")

        save_error = None
        try:
            import torch.distributed.checkpoint as dcp

            dcp.save(
                {"optimizer": _DrafterOptimizerState(self.model, self.optimizer)},
                checkpoint_id=temporary_path,
                process_group=self._checkpoint_process_group(),
            )
        except Exception as exc:  # noqa: BLE001
            save_error = exc
        self._sync_checkpoint_phase_error(save_error, "optimizer save")

        finalize_error = None
        if is_leader:
            try:
                self._atomic_torch_save(
                    {
                        "version": 1,
                        "optimizer_steps_total": int(self.optimizer_steps_total),
                        "training_steps": int(self.training_steps),
                        "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    },
                    os.path.join(temporary_path, trainer_state_file),
                )
                os.replace(temporary_path, optimizer_path)
            except Exception as exc:  # noqa: BLE001
                finalize_error = exc
        self._sync_checkpoint_phase_error(finalize_error, "optimizer finalize")

        manifest = {
            "format": "torch_distributed_checkpoint",
            "path": "optimizer",
            "trainer_state_file": trainer_state_file,
            "save_sp_world_size": int(self._get_sp_world_size()),
            "save_dp_world_size": int(self._get_dp_world_size()),
        }
        if is_leader:
            try:
                total_bytes = sum(
                    os.path.getsize(os.path.join(root, filename))
                    for root, _, filenames in os.walk(optimizer_path)
                    for filename in filenames
                )
                logger.info(
                    "[drafter checkpoint] optimizer saved path=%s size_gib=%.2f sp=%s dp=%s",
                    optimizer_path,
                    total_bytes / (1024**3),
                    self._get_sp_world_size(),
                    self._get_dp_world_size(),
                )
            except OSError as exc:
                logger.warning("Unable to measure drafter optimizer checkpoint size at %s: %s", optimizer_path, exc)
        return manifest

    def _load_optimizer_checkpoint(self, checkpoint_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
        from verl_speco.trainer.checkpoint import get_drafter_optimizer_checkpoint_path

        optimizer_path = get_drafter_optimizer_checkpoint_path(checkpoint_path)
        if optimizer_path is None:
            raise RuntimeError(f"Drafter checkpoint metadata declares no optimizer state: {checkpoint_path}")

        try:
            import torch.distributed.checkpoint as dcp

            dcp.load(
                {"optimizer": _DrafterOptimizerState(self.model, self.optimizer)},
                checkpoint_id=optimizer_path,
                process_group=self._checkpoint_process_group(),
            )
            trainer_state_file = metadata["optimizer"]["trainer_state_file"]
            trainer_state_path = os.path.join(optimizer_path, trainer_state_file)
            try:
                trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            except TypeError:
                trainer_state = torch.load(trainer_state_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to restore drafter optimizer checkpoint from {optimizer_path}: {exc}") from exc

        if not isinstance(trainer_state, dict) or int(trainer_state.get("version", 0) or 0) != 1:
            raise RuntimeError(f"Invalid drafter trainer state in {trainer_state_path}")
        return trainer_state

    def _save_pretrained_checkpoint_async(
        self,
        checkpoint_path: str,
        step: int,
        optimizer_manifest: dict[str, Any],
    ):
        if self._pending_full_checkpoint_future is not None:
            if not self._pending_full_checkpoint_future.done():
                logger.warning(
                    "[Rank %s] Previous drafter checkpoint save is still running; skip step=%s",
                    self.rank,
                    step,
                )
                return None
            try:
                self._pending_full_checkpoint_future.result()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Previous full drafter checkpoint save failed") from exc
            self._pending_full_checkpoint_future = None

        export_model, _ = self._get_pretrained_export_model()
        model_state_dict = self._get_pretrained_export_state_dict()
        if not self._is_checkpoint_leader() or not model_state_dict:
            return None
        if not hasattr(export_model, "save_pretrained"):
            raise TypeError(f"Drafter export model does not support save_pretrained: {type(export_model)}")

        save_kwargs = self._infer_pretrained_save_kwargs()
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        trainer_state = {
            "version": 2,
            "optimizer_steps_total": int(self.optimizer_steps_total),
            "training_steps": int(self.training_steps),
            "lr_scheduler_last_epoch": (
                int(self.lr_scheduler.last_epoch) if self.lr_scheduler is not None else None
            ),
            "current_lr": (
                float(self.optimizer.param_groups[0]["lr"])
                if self.optimizer is not None and self.optimizer.param_groups
                else None
            ),
        }

        def _write_full_checkpoint():
            os.makedirs(checkpoint_path, exist_ok=True)
            self._clear_existing_pretrained_weight_files(checkpoint_path)
            export_model.save_pretrained(checkpoint_path, state_dict=model_state_dict, **save_kwargs)
            self._copy_drafter_auxiliary_files(checkpoint_path)
            self._atomic_json_dump(
                {
                    "step": step,
                    "format": "pretrained_drafter_checkpoint",
                    "serialization": "pytorch",
                    "complete": True,
                    "trainer_state": trainer_state,
                    "optimizer": optimizer_manifest,
                },
                metadata_path,
            )

        if self._full_checkpoint_executor is None:
            self._full_checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drafter-full-ckpt")
        future = self._full_checkpoint_executor.submit(_write_full_checkpoint)
        self._pending_full_checkpoint_future = future
        logger.debug("[Rank %s] Scheduled drafter checkpoint export to %s", self.rank, checkpoint_path)
        return future

    
    def _save_checkpoint_async(
        self,
        step: int,
        optimizer_manifest: dict[str, Any],
        is_final: bool = False,
    ):
        """Asynchronously save a directly loadable drafter checkpoint.

        Args:
            step: Current training step
            is_final: Whether this is the final checkpoint during cleanup

        Returns:
            Future object for the background save, or None on non-leader ranks
        """
        if not self.checkpoint_dir:
            return None

        checkpoint_path = os.path.join(self.checkpoint_dir, f"draft_step_{step}")
        return self._save_pretrained_checkpoint_async(checkpoint_path, step, optimizer_manifest)

    def save_checkpoint(self, step: int, wait: bool = True) -> dict[str, Any]:
        if not self.checkpoint_dir:
            return {"saved": False, "reason": "missing_checkpoint_dir"}

        checkpoint_path = os.path.join(self.checkpoint_dir, f"draft_step_{int(step)}")
        if self.rollout_dp_rank != 0:
            return {
                "saved": False,
                "path": checkpoint_path,
                "reason": "not_checkpoint_replica",
            }
        pending_full_checkpoint = getattr(self, "_pending_full_checkpoint_future", None)
        previous_error = None
        if self._is_checkpoint_leader() and pending_full_checkpoint is not None:
            try:
                pending_full_checkpoint.result()
            except Exception as exc:  # noqa: BLE001
                previous_error = exc
            finally:
                self._pending_full_checkpoint_future = None
        self._sync_checkpoint_phase_error(previous_error, "previous save wait")

        if self.model is None:
            self._build_draft_model()

        first_param = next(self.model.parameters(), None)
        was_on_device = first_param is not None and first_param.device.type == device_name
        is_fsdp_wrapped = isinstance(self.model, FSDP) or self.training_device_mesh is not None
        if is_fsdp_wrapped and not was_on_device:
            load_fsdp_model_to_gpu(self.model)

        future = None
        optimizer_manifest = None
        try:
            optimizer_manifest = self._save_optimizer_checkpoint(checkpoint_path)
            future = self._save_checkpoint_async(int(step), optimizer_manifest)
            if wait and future is not None:
                future.result()
                self._pending_full_checkpoint_future = None
        finally:
            if is_fsdp_wrapped and not was_on_device:
                offload_fsdp_model_to_cpu(self.model)

        if future is None:
            reason = "optimizer_shard_saved"
        elif wait:
            reason = "saved"
        else:
            reason = "scheduled"
        return {
            "saved": optimizer_manifest is not None,
            "path": checkpoint_path,
            "reason": reason,
        }

    def wait_checkpoint(self) -> dict[str, Any]:
        pending_full_checkpoint = getattr(self, "_pending_full_checkpoint_future", None)
        if pending_full_checkpoint is None:
            return {"waited": False, "completed": True, "reason": "no_pending_checkpoint"}
        try:
            pending_full_checkpoint.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pending full drafter checkpoint save failed: %s", exc)
            return {"waited": True, "completed": False, "reason": "failed", "error": str(exc)}
        finally:
            self._pending_full_checkpoint_future = None
        return {"waited": True, "completed": True, "reason": "completed"}

    async def activate_training_model(self) -> bool:
        # 将模型和优化器状态从CPU加载到GPU，激活草稿模型进入训练状态
        start_ts = time.time()
        try:        
            logger.debug(
                f"[Trainer rank {getattr(self, 'rank', -1)}] activate_training_model enter "
            )

            if self._training_active and self.model is not None:
                self._apply_pending_target_lm_head_weight()
                logger.debug(
                    f"[DrafterTrainer rank {getattr(self, 'rank', -1)}] activate_training_model reused active model "
                    f"elapsed={time.time() - start_ts:.2f}s"
                )
                return True

            if self.model is None:
                logger.debug("Draft Model not initialized, calling build_draft_model during activation...")
                self._build_draft_model()

            # 只有当配置了 offload 或者当前模型不在 CUDA 上时执行加载
            first_param = next(self.model.parameters(), None)
            is_on_cuda = first_param is not None and first_param.device.type == device_name

            if self.is_offload_param or not is_on_cuda:
                # 调用工具将 FSDP 分片移动到 GPU
                load_fsdp_model_to_gpu(self.model)
                logger.debug("Loaded drafter model to GPU for training")
            
            if self.optimizer is not None and (self.is_offload_optimizer or not self._optimizer_state_on_runtime_device()):
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

            logger.debug(
                f"[DrafterTrainer rank {getattr(self, 'rank', -1)}] activate_training_model success "
                f"elapsed={time.time() - start_ts:.2f}s"
            )
            return True
        
        except Exception as e:
            logger.error(f"[DrafterTrainer rank {getattr(self, 'rank', -1)}] activate_training_model failed: {e}")
            self._training_active = False
            return False

    def sync_target_lm_head_weight(
        self,
        weight: torch.Tensor,
        global_step: Optional[int] = None,
        row_indices: Optional[torch.Tensor] = None,
        source_vocab_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Update the frozen target lm_head used by last-hidden drafter training."""
        if weight is None or not torch.is_tensor(weight):
            return {"accepted": False, "applied": False, "reason": "missing_weight"}

        weight_shape = tuple(weight.shape)
        self._pending_target_lm_head_weight = weight.detach().cpu().contiguous()
        if row_indices is not None and torch.is_tensor(row_indices):
            self._pending_target_lm_head_row_indices = row_indices.detach().cpu().long().reshape(-1).contiguous()
        elif isinstance(row_indices, (list, tuple)):
            self._pending_target_lm_head_row_indices = torch.tensor(
                [int(idx) for idx in row_indices], dtype=torch.long
            ).reshape(-1).contiguous()
        else:
            self._pending_target_lm_head_row_indices = None
        self._pending_target_lm_head_source_vocab_size = (
            int(source_vocab_size) if source_vocab_size is not None else None
        )
        self._target_lm_head_weight_step = global_step
        selected_rows = (
            int(self._pending_target_lm_head_row_indices.numel())
            if self._pending_target_lm_head_row_indices is not None
            else None
        )
        pending_source_vocab_size = self._pending_target_lm_head_source_vocab_size
        applied = self._apply_pending_target_lm_head_weight()
        return {
            "accepted": True,
            "applied": applied,
            "pending": not applied,
            "global_step": global_step,
            "shape": weight_shape,
            "selected_rows": selected_rows,
            "source_vocab_size": pending_source_vocab_size,
        }

    def get_target_lm_head_row_indices(self) -> Optional[dict[str, Any]]:
        """Return target lm_head row indices needed by the current drafter loss."""
        if self._is_block_drafter_backend():
            return self._build_target_lm_head_row_indices_from_dflash_data()
        if getattr(self.backend, "model_type", None) != "eagle3":
            return None
        if self.model is not None:
            draft_model = self.model.module if hasattr(self.model, "module") else self.model
            t2d = getattr(draft_model, "t2d", None)
            row_info = self._build_target_lm_head_row_indices_from_t2d(t2d)
            if row_info is None:
                row_info = self._build_target_lm_head_row_indices_from_drafter_config()
            if row_info is not None:
                self._cached_target_lm_head_row_indices = row_info
            return row_info
        if self._cached_target_lm_head_row_indices is not None:
            return self._cached_target_lm_head_row_indices

        row_info = self._load_target_lm_head_row_indices_from_checkpoint()
        if row_info is not None:
            self._cached_target_lm_head_row_indices = row_info
            return row_info
        row_info = self._build_target_lm_head_row_indices_from_drafter_config()
        if row_info is not None:
            self._cached_target_lm_head_row_indices = row_info
        return row_info

    def _target_lm_head_vocab_size(self) -> Optional[int]:
        lm_head = self._target_lm_head_module()
        if lm_head is not None and getattr(lm_head, "out_features", None) is not None:
            return int(lm_head.out_features)
        model_config = getattr(self, "model_config", None)
        vocab_size = getattr(model_config, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        try:
            return int(self.config.actor_rollout_ref.model.get("vocab_size"))
        except Exception:  # noqa: BLE001
            return None

    def _dflash_target_lm_head_row_source_items(self) -> list[dict[str, Any]]:
        current_step = int(self.current_rl_step)
        if self.use_data_buffer and len(self.data_buffer) > 0:
            sample_last_n = int(self.config.rollout.drafter.training.get("sample_last_n_steps", 2))
            items = self.data_buffer.get_data_from_last_n_steps(sample_last_n)
            if items:
                return items
        return [
            item
            for item in self.collected_data
            if int(item.get("step", current_step)) == current_step
        ]

    def _build_target_lm_head_row_indices_from_dflash_data(self) -> Optional[dict[str, Any]]:
        """Return a conservative DFlash restricted-CE row set from collected tokens.

        DFlash restricted CE builds its vocab from the per-batch ``input_ids`` and
        active targets.  The active targets are gathered from the same sample
        windows, so syncing all valid token ids from candidate DFlash training
        samples preserves the original loss semantics while avoiding a full
        lm_head transfer.
        """
        training_cfg = self.config.rollout.drafter.training
        loss_mode = str(self._block_drafter_config_value("loss_mode", "full_vocab"))
        if loss_mode != "restricted_ce":
            return None

        source_vocab_size = self._target_lm_head_vocab_size()
        if source_vocab_size is None or source_vocab_size <= 0:
            return None

        token_chunks = []
        for item in self._dflash_target_lm_head_row_source_items():
            input_ids = item.get("input_ids")
            if not torch.is_tensor(input_ids):
                continue
            flat_ids = input_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
            valid = flat_ids[(flat_ids >= 0) & (flat_ids < int(source_vocab_size))]
            if int(valid.numel()) > 0:
                token_chunks.append(valid)
        if not token_chunks:
            return None

        row_indices = torch.unique(torch.cat(token_chunks), sorted=True).to(dtype=torch.long).contiguous()
        selected_rows = int(row_indices.numel())
        if selected_rows <= 0 or selected_rows >= int(source_vocab_size):
            return None
        return {
            "row_indices": row_indices,
            "source_vocab_size": int(source_vocab_size),
            "selected_rows": selected_rows,
            "source": f"{self._block_drafter_metric_prefix()}_collected_tokens",
        }

    def _build_target_lm_head_row_indices_from_t2d(self, t2d: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
        if t2d is None or not torch.is_tensor(t2d):
            return None

        t2d_cpu = t2d.detach().to(device="cpu", dtype=torch.bool).flatten()
        source_vocab_size = int(t2d_cpu.numel())
        row_indices = torch.nonzero(t2d_cpu, as_tuple=False).flatten().to(dtype=torch.long).contiguous()
        selected_rows = int(row_indices.numel())
        if selected_rows <= 0 or selected_rows >= source_vocab_size:
            return None
        return {
            "row_indices": row_indices,
            "source_vocab_size": source_vocab_size,
            "selected_rows": selected_rows,
        }

    def _build_target_lm_head_row_indices_from_drafter_config(self) -> Optional[dict[str, Any]]:
        """Fallback for EAGLE3 checkpoints that use the default prefix draft vocab."""
        model_path = self.config.rollout.drafter.get("model_path")
        if not model_path:
            return None
        config_path = os.path.join(str(model_path), "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                drafter_config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(drafter_config, dict):
            return None

        vocab_size = drafter_config.get("vocab_size")
        draft_vocab_size = drafter_config.get("draft_vocab_size", vocab_size)
        if vocab_size is None or draft_vocab_size is None:
            return None
        vocab_size = int(vocab_size)
        draft_vocab_size = int(draft_vocab_size)
        if draft_vocab_size <= 0 or draft_vocab_size >= vocab_size:
            return None

        row_indices = torch.arange(draft_vocab_size, dtype=torch.long).contiguous()
        logger.debug(
            "[drafter target lm_head rows] using prefix vocab from drafter config model_path=%s "
            "target_vocab=%s selected_rows=%s",
            model_path,
            vocab_size,
            draft_vocab_size,
        )
        return {
            "row_indices": row_indices,
            "source_vocab_size": vocab_size,
            "selected_rows": draft_vocab_size,
        }

    def _load_target_lm_head_row_indices_from_checkpoint(self) -> Optional[dict[str, Any]]:
        model_path = self.config.rollout.drafter.get("model_path")
        if not model_path or not os.path.isdir(model_path):
            return None
        try:
            t2d = self._load_checkpoint_tensor(model_path, "t2d")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to load EAGLE3 t2d from drafter checkpoint %s: %s", model_path, exc)
            return None
        row_info = self._build_target_lm_head_row_indices_from_t2d(t2d)
        if row_info is not None:
            logger.debug(
                "[drafter target lm_head rows] loaded t2d from checkpoint model_path=%s "
                "target_vocab=%s selected_rows=%s",
                model_path,
                row_info["source_vocab_size"],
                row_info["selected_rows"],
            )
        return row_info

    def _load_checkpoint_tensor(self, model_path: str, logical_name: str) -> torch.Tensor:
        index_paths = glob.glob(os.path.join(model_path, "*.index.json"))
        if len(index_paths) > 1:
            raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
        if index_paths:
            with open(index_paths[0], "r") as f:
                index_json = json.load(f)
            weight_map = index_json.get("weight_map", {})
            selected_key = self._select_checkpoint_tensor_key(weight_map.keys(), logical_name)
            if selected_key is None:
                raise KeyError(f"Cannot find {logical_name} in checkpoint index {index_paths[0]}")
            ckpt_file = os.path.join(model_path, weight_map[selected_key])
            return self._load_tensor_from_checkpoint_file(ckpt_file, selected_key)

        for filename in ("model.safetensors", "pytorch_model.bin"):
            ckpt_file = os.path.join(model_path, filename)
            if os.path.exists(ckpt_file):
                return self._load_tensor_from_checkpoint_file(ckpt_file, logical_name)
        raise FileNotFoundError(f"No model index, model.safetensors or pytorch_model.bin found in {model_path}")

    @staticmethod
    def _select_checkpoint_tensor_key(keys, logical_name: str) -> Optional[str]:
        keys = list(keys)
        if logical_name in keys:
            return logical_name
        suffix = f".{logical_name}"
        return next((key for key in keys if str(key).endswith(suffix)), None)

    def _load_tensor_from_checkpoint_file(self, ckpt_file: str, logical_name: str) -> torch.Tensor:
        if ckpt_file.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                selected_key = self._select_checkpoint_tensor_key(f.keys(), logical_name)
                if selected_key is None:
                    raise KeyError(f"Cannot find {logical_name} in {ckpt_file}")
                return f.get_tensor(selected_key)

        try:
            state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint object type for {ckpt_file}: {type(state_dict)}")
        selected_key = self._select_checkpoint_tensor_key(state_dict.keys(), logical_name)
        if selected_key is None:
            raise KeyError(f"Cannot find {logical_name} in {ckpt_file}")
        return state_dict[selected_key]

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
        source_row_indices = self._pending_target_lm_head_row_indices
        source_vocab_size = self._pending_target_lm_head_source_vocab_size
        if source_row_indices is not None:
            if source_weight.dim() != 2:
                raise ValueError(
                    "Sparse target lm_head sync expects a 2D weight tensor, "
                    f"got source={tuple(source_weight.shape)}"
                )
            if int(source_row_indices.numel()) != int(source_weight.shape[0]):
                raise ValueError(
                    "Sparse target lm_head sync row count mismatch: "
                    f"rows={int(source_row_indices.numel())}, weight_rows={int(source_weight.shape[0])}"
                )
            if source_vocab_size is not None and int(target_weight.shape[0]) != int(source_vocab_size):
                lm_head = self._resize_target_lm_head_to_vocab_size(lm_head, int(source_vocab_size))
                target_weight = lm_head.weight
            if int(source_row_indices.numel()) <= 0:
                self._pending_target_lm_head_weight = None
                self._pending_target_lm_head_row_indices = None
                self._pending_target_lm_head_source_vocab_size = None
                return True
            if int(source_row_indices.max().item()) >= int(target_weight.shape[0]) or int(source_row_indices.min().item()) < 0:
                raise ValueError(
                    "Sparse target lm_head sync row index out of range: "
                    f"target_rows={int(target_weight.shape[0])}, "
                    f"min={int(source_row_indices.min().item())}, max={int(source_row_indices.max().item())}"
                )
            target_weight.index_copy_(
                0,
                source_row_indices.to(device=target_weight.device, dtype=torch.long),
                source_weight.to(device=target_weight.device, dtype=target_weight.dtype, non_blocking=True),
            )
            lm_head.requires_grad_(False)
            logger.debug(
                "[drafter target lm_head sync] applied sparse global_step=%s rows=%s/%s dtype=%s device=%s",
                self._target_lm_head_weight_step,
                int(source_row_indices.numel()),
                int(target_weight.shape[0]),
                target_weight.dtype,
                target_weight.device,
            )
            self._pending_target_lm_head_weight = None
            self._pending_target_lm_head_row_indices = None
            self._pending_target_lm_head_source_vocab_size = None
            return True

        if tuple(source_weight.shape) != tuple(target_weight.shape):
            lm_head = self._resize_target_lm_head_to_weight(lm_head, source_weight)
            target_weight = lm_head.weight
            if tuple(source_weight.shape) != tuple(target_weight.shape):
                raise ValueError(
                    "Target lm_head weight shape mismatch for drafter training: "
                    f"source={tuple(source_weight.shape)}, target={tuple(target_weight.shape)}"
                )

        target_weight.copy_(source_weight.to(device=target_weight.device, dtype=target_weight.dtype, non_blocking=True))
        lm_head.requires_grad_(False)
        if logger.isEnabledFor(logging.DEBUG):
            block = target_weight[
                : min(8, int(target_weight.size(0))),
                : min(8, int(target_weight.size(1))),
            ].detach().float()
            row0_norm = target_weight[0].detach().float().norm() if int(target_weight.size(0)) > 0 else None
            row_last_norm = target_weight[-1].detach().float().norm() if int(target_weight.size(0)) > 0 else None
            logger.debug(
                "[drafter target lm_head sync] applied global_step=%s shape=%s dtype=%s device=%s "
                "block_sum=%.6g row0_norm=%s row_last_norm=%s",
                self._target_lm_head_weight_step,
                tuple(target_weight.shape),
                target_weight.dtype,
                target_weight.device,
                float(block.sum().detach().cpu().item()),
                None if row0_norm is None else round(float(row0_norm.detach().cpu().item()), 6),
                None if row_last_norm is None else round(float(row_last_norm.detach().cpu().item()), 6),
            )
        if last_hidden_logprob_check_enabled() and int(target_weight.size(0)) > 0:
            probe_indices = sorted(
                {
                    0,
                    max(int(target_weight.size(0)) // 2 - 1, 0),
                    min(int(target_weight.size(0)) // 2, int(target_weight.size(0)) - 1),
                    int(target_weight.size(0)) - 1,
                }
            )
            probe_norms = {}
            for row_idx in probe_indices:
                row_norm = target_weight[row_idx].detach().float().norm()
                probe_norms[f"row_{row_idx}_norm"] = round(float(row_norm.detach().cpu().item()), 6)
            logger.debug(
                "[drafter target lm_head sync debug] global_step=%s probe_norms=%s",
                self._target_lm_head_weight_step,
                probe_norms,
            )
        self._pending_target_lm_head_weight = None
        self._pending_target_lm_head_row_indices = None
        self._pending_target_lm_head_source_vocab_size = None
        return True

    def _resize_target_lm_head_to_vocab_size(self, lm_head, vocab_size: int):
        if vocab_size <= 0 or int(lm_head.out_features) == int(vocab_size):
            return lm_head

        new_head = torch.nn.Linear(
            lm_head.in_features,
            int(vocab_size),
            bias=False,
            device=lm_head.weight.device,
            dtype=lm_head.weight.dtype,
        )
        new_head.requires_grad_(False)

        target_model = getattr(self.backend, "target_model", None)
        target_lm_head = getattr(self.backend, "target_lm_head", None)
        if target_model is not None and getattr(target_model, "fc", None) is lm_head:
            target_model.fc = new_head
            return target_model.fc
        if target_lm_head is not None and getattr(target_lm_head, "fc", None) is lm_head:
            target_lm_head.fc = new_head
            return target_lm_head.fc
        return lm_head

    def _resize_target_lm_head_to_weight(self, lm_head, source_weight: torch.Tensor):
        if source_weight.dim() != 2 or int(source_weight.shape[1]) != int(lm_head.in_features):
            return lm_head
        new_vocab_size = int(source_weight.shape[0])
        if new_vocab_size == int(lm_head.out_features):
            return lm_head

        new_head = torch.nn.Linear(
            lm_head.in_features,
            new_vocab_size,
            bias=False,
            device=lm_head.weight.device,
            dtype=lm_head.weight.dtype,
        )
        new_head.requires_grad_(False)

        target_model = getattr(self.backend, "target_model", None)
        target_lm_head = getattr(self.backend, "target_lm_head", None)
        if target_model is not None and getattr(target_model, "fc", None) is lm_head:
            target_model.fc = new_head
            logger.debug(
                "[drafter target lm_head sync] resized target_model.fc from out_features=%s to out_features=%s",
                lm_head.out_features,
                new_vocab_size,
            )
            return target_model.fc
        if target_lm_head is not None and getattr(target_lm_head, "fc", None) is lm_head:
            target_lm_head.fc = new_head
            logger.debug(
                "[drafter target lm_head sync] resized target_lm_head.fc from out_features=%s to out_features=%s",
                lm_head.out_features,
                new_vocab_size,
            )
            return target_lm_head.fc
        return lm_head

    def collect_online_data(self, batch: dict, hidden_states: torch.Tensor, target_logprobs: List = None) -> None:
        """Collect online data from inference for drafter training.

        This method stores hidden states in the cross-step DataBuffer only when
        use_data_buffer=True. Otherwise it keeps only the current-step samples.
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            logger.debug(
                f"[Rank {self.rank}] Non-batched data in input_ids"
            )
            return

        # 1、异步拷贝，GPU在后台进行数据搬运，避免阻塞Rollout Stream
        use_logits = bool(self.config.rollout.drafter.training.get("use_logits", False))
        if not use_logits:
            # Phase 4: use_logits=False reconstructs supervision from the
            # synced target_lm_head(last_hidden_states). Do not keep stale
            # SGLang output_top_logprobs payloads in collected samples.
            target_logprobs = None
        elif target_logprobs is not None and not isinstance(target_logprobs, torch.Tensor):
            logger.debug(f"[Rank {self.rank}] Unsupported target_logprobs type: {type(target_logprobs)}")
            target_logprobs = None
        hidden_states_layout = batch.get("hidden_states_layout")
        if hidden_states_layout is None:
            hidden_states_layout = (
                "dflash_aux"
                if self._is_block_drafter_backend()
                else "eagle3_aux_plus_last"
            )

        source_tensors = [input_ids, hidden_states]
        hidden_positions = batch.get("hidden_positions")
        if isinstance(hidden_positions, torch.Tensor):
            source_tensors.append(hidden_positions)
        else:
            hidden_positions = None
        hidden_raw_target_logprobs = batch.get("hidden_raw_target_logprobs")
        if isinstance(hidden_raw_target_logprobs, torch.Tensor):
            source_tensors.append(hidden_raw_target_logprobs)
        else:
            hidden_raw_target_logprobs = None
        hidden_raw_target_logprobs_positions = batch.get("hidden_raw_target_logprobs_positions")
        if isinstance(hidden_raw_target_logprobs_positions, torch.Tensor):
            source_tensors.append(hidden_raw_target_logprobs_positions)
        else:
            hidden_raw_target_logprobs_positions = None
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
                cpu_hidden_positions = (
                    hidden_positions.to('cpu', non_blocking=True) if hidden_positions is not None else None
                )
                cpu_hidden_raw_target_logprobs = (
                    hidden_raw_target_logprobs.to('cpu', non_blocking=True)
                    if hidden_raw_target_logprobs is not None
                    else None
                )
                cpu_hidden_raw_target_logprobs_positions = (
                    hidden_raw_target_logprobs_positions.to('cpu', non_blocking=True)
                    if hidden_raw_target_logprobs_positions is not None
                    else None
                )
                cpu_responses = batch.get("responses").to('cpu', non_blocking=True) if "responses" in batch else None
                cpu_prompts = batch.get("prompts").to('cpu', non_blocking=True) if "prompts" in batch else None

            self.device_module.current_stream().wait_stream(self.copy_stream)
        else:
            cpu_input_ids = input_ids.to('cpu')
            cpu_h_states = hidden_states.to('cpu')
            cpu_target_logprobs = target_logprobs.to('cpu') if target_logprobs is not None else None
            cpu_hidden_positions = hidden_positions.to('cpu') if hidden_positions is not None else None
            cpu_hidden_raw_target_logprobs = (
                hidden_raw_target_logprobs.to('cpu') if hidden_raw_target_logprobs is not None else None
            )
            cpu_hidden_raw_target_logprobs_positions = (
                hidden_raw_target_logprobs_positions.to('cpu')
                if hidden_raw_target_logprobs_positions is not None
                else None
            )
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
            raw_positions_item_for_alignment = None
            if cpu_hidden_raw_target_logprobs is not None and cpu_hidden_raw_target_logprobs_positions is not None:
                raw_rows_for_alignment = int(cpu_hidden_raw_target_logprobs[i].size(0))
                candidate_raw_positions = cpu_hidden_raw_target_logprobs_positions[i].reshape(-1).long()
                if int(candidate_raw_positions.numel()) >= raw_rows_for_alignment:
                    raw_positions_item_for_alignment = candidate_raw_positions[:raw_rows_for_alignment]
            hidden_positions_item = None
            if cpu_hidden_positions is not None:
                hidden_positions_item = cpu_hidden_positions[i].reshape(-1)[:hidden_seq_length].long()
                if hidden_positions_item.numel() <= 0:
                    hidden_positions_item = None
            hidden_position_start = _batch_item_int(batch.get("hidden_position_start"), i)
            uses_hidden_positions = hidden_positions_item is not None
            require_sglang_positions = bool(
                self.config.rollout.drafter.training.get("collect_hidden_states_from_sgl", False)
            )
            if require_sglang_positions and (
                hidden_positions_item is None or int(hidden_positions_item.numel()) != hidden_seq_length
            ):
                logger.warning(
                    "[Rank %s] Drop drafter sample: missing or mismatched SGLang hidden positions "
                    "sample=%s hidden_rows=%s positions=%s",
                    self.rank,
                    i,
                    hidden_seq_length,
                    int(hidden_positions_item.numel()) if hidden_positions_item is not None else None,
                )
                continue
            if hidden_positions_item is not None:
                selected_hidden_row_end = int(hidden_positions_item.numel())
                contiguous_mask = hidden_positions_item[1:] == hidden_positions_item[:-1] + 1
                if contiguous_mask.numel() > 0 and not bool(contiguous_mask.all()):
                    first_break = int(torch.nonzero(~contiguous_mask, as_tuple=False)[0].item()) + 1
                    selected_hidden_row_end = min(selected_hidden_row_end, first_break)
                    logger.warning(
                        "[Rank %s] Truncate drafter sample at first hidden-position break rows=%s/%s",
                        self.rank,
                        selected_hidden_row_end,
                        int(hidden_positions_item.numel()),
                    )

                if cpu_hidden_raw_target_logprobs is not None:
                    raw_rows = int(cpu_hidden_raw_target_logprobs[i].size(0))
                    if raw_positions_item_for_alignment is None:
                        logger.warning("[Rank %s] Drop logits sample with missing raw positions", self.rank)
                        continue
                    common_rows = min(
                        selected_hidden_row_end,
                        raw_rows,
                        int(raw_positions_item_for_alignment.numel()),
                    )
                    raw_contiguous = (
                        raw_positions_item_for_alignment[1:common_rows]
                        == raw_positions_item_for_alignment[: max(common_rows - 1, 0)] + 1
                    )
                    if raw_contiguous.numel() > 0 and not bool(raw_contiguous.all()):
                        raw_break = int(torch.nonzero(~raw_contiguous, as_tuple=False)[0].item()) + 1
                        common_rows = min(common_rows, raw_break)
                    aligned = (
                        raw_positions_item_for_alignment[:common_rows]
                        == hidden_positions_item[:common_rows] + 1
                    )
                    if aligned.numel() > 0 and not bool(aligned.all()):
                        first_mismatch = int(torch.nonzero(~aligned, as_tuple=False)[0].item())
                        common_rows = min(common_rows, first_mismatch)
                    selected_hidden_row_end = min(selected_hidden_row_end, common_rows)
                elif use_logits:
                    logger.warning("[Rank %s] Drop logits sample without raw top-k rows", self.rank)
                    continue

                if selected_hidden_row_end <= 0:
                    logger.warning("[Rank %s] Drop sample with no aligned continuous prefix", self.rank)
                    continue

                hidden_position_start = max(int(hidden_positions_item[0].item()), 0)
                # Phase 3: SGLang hidden_positions is the source of truth.
                # Hidden row p supervises token p+1 and target row p+1, and
                # the loss row is p+2, so keep only rows with that token window.
                max_hidden_rows = min(
                    selected_hidden_row_end,
                    hidden_seq_length,
                    max(input_seq_length - hidden_position_start - 1, 0),
                )
                hidden_start = 0
                hidden_feature_length = max_hidden_rows
                hidden_end = hidden_feature_length
                feature_start = hidden_position_start
                feature_end = min(input_seq_length, feature_start + hidden_feature_length + 1)
            else:
                if hidden_position_start is None:
                    hidden_position_start = max(expected_hidden_rows - hidden_seq_length, 0)
                # Legacy fallback for non-SGLang or older buffered samples.
                # New SGLang EAGLE3 samples should always carry hidden_positions.
                feature_start = min(max(hidden_position_start, 0), input_seq_length)
                hidden_start = 0
                hidden_feature_length = min(hidden_seq_length, max(input_seq_length - feature_start - 1, 0))
                hidden_end = hidden_feature_length
                feature_end = min(input_seq_length, feature_start + hidden_feature_length + 1)

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

                # For EAGLE3, the loss target for hidden row p is the
                # target row at original position p + 1. A compact tensor may
                # therefore start one row after the hidden window without
                # requiring us to drop the corresponding hidden/input row.
                target_row_offset = 1 if getattr(self.backend, "model_type", None) == "eagle3" else 0
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
                    feature_end = min(input_seq_length, feature_start + hidden_feature_length + 1)

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
                # original position p. For shifted EAGLE3 training, the first
                # usable target row is feature_start + 1.
                target_base = target_logprobs_position_start or 0
                target_row_offset = 1 if getattr(self.backend, "model_type", None) == "eagle3" else 0
                target_limit = min(
                    cpu_target_logprobs.size(1),
                    max((target_logprobs_position_end or target_base) - target_base, 0),
                )
                target_start = min(max(feature_start + target_row_offset - target_base, 0), target_limit)
                target_end = min(max(feature_end - 1 - target_base, target_start), target_limit)
                target_logprobs_item = cpu_target_logprobs[i, target_start:target_end, ...]

            kept_hidden_positions = (
                hidden_positions_item[hidden_start:hidden_end] if hidden_positions_item is not None else None
            )
            hidden_raw_target_logprobs_item = None
            raw_target_logprobs_positions_item = None
            if cpu_hidden_raw_target_logprobs is not None:
                raw_target_logprobs_item = cpu_hidden_raw_target_logprobs[i]
                if hidden_positions_item is not None:
                    raw_target_logprobs_item = raw_target_logprobs_item[:selected_hidden_row_end]
                raw_rows = int(raw_target_logprobs_item.size(0))
                if cpu_hidden_raw_target_logprobs_positions is not None:
                    candidate_raw_positions = cpu_hidden_raw_target_logprobs_positions[i].reshape(-1).long()
                    if int(candidate_raw_positions.numel()) >= raw_rows:
                        raw_target_logprobs_positions_item = candidate_raw_positions[:raw_rows]
                raw_position_start = _batch_item_int(batch.get("hidden_raw_target_logprobs_position_start"), i)
                raw_position_end = _batch_item_int(batch.get("hidden_raw_target_logprobs_position_end"), i)
                if raw_position_start is None:
                    raw_position_start = hidden_position_start
                if raw_position_end is None:
                    raw_position_end = raw_position_start + raw_rows
                raw_position_end = min(
                    max(raw_position_end, raw_position_start),
                    raw_position_start + raw_rows,
                )
                if raw_target_logprobs_item.dim() == 3 and hidden_feature_length > 0:
                    hidden_raw_target_logprobs_item = torch.zeros(
                        hidden_feature_length,
                        raw_target_logprobs_item.size(1),
                        raw_target_logprobs_item.size(2),
                        dtype=raw_target_logprobs_item.dtype,
                    )
                    hidden_raw_target_logprobs_item[..., 0] = float("-inf")
                    if int(hidden_raw_target_logprobs_item.size(-1)) > 1:
                        hidden_raw_target_logprobs_item[..., 1] = -1
                    row_positions = (
                        kept_hidden_positions.long() + 1
                        if kept_hidden_positions is not None
                        else torch.arange(
                            feature_start + 1,
                            feature_start + 1 + hidden_feature_length,
                            dtype=torch.long,
                        )
                    )
                    if raw_target_logprobs_positions_item is not None:
                        raw_index_by_position = {
                            int(position): raw_index
                            for raw_index, position in enumerate(raw_target_logprobs_positions_item.tolist())
                            if int(position) >= 0
                        }
                        local_rows = []
                        raw_indices = []
                        for local_row, position in enumerate(row_positions.tolist()):
                            raw_index = raw_index_by_position.get(int(position))
                            if raw_index is not None:
                                local_rows.append(local_row)
                                raw_indices.append(raw_index)
                        if local_rows:
                            hidden_raw_target_logprobs_item[torch.tensor(local_rows, dtype=torch.long)] = (
                                raw_target_logprobs_item[torch.tensor(raw_indices, dtype=torch.long)]
                            )
                    else:
                        # Legacy metadata has no explicit positions. Keep the
                        # compact row-order fallback for older rollout samples.
                        raw_slice_start = min(max(int(hidden_start), 0), raw_rows)
                        raw_slice_end = min(max(int(hidden_end), raw_slice_start), raw_rows)
                        copy_rows = min(raw_slice_end - raw_slice_start, hidden_feature_length)
                        if copy_rows > 0:
                            hidden_raw_target_logprobs_item[:copy_rows] = raw_target_logprobs_item[
                                raw_slice_start : raw_slice_start + copy_rows
                            ]
            item_position_ids = (
                kept_hidden_positions + 1
                if kept_hidden_positions is not None
                else torch.arange(feature_start + 1, feature_start + 1 + hidden_feature_length, dtype=torch.long)
            )
            accept_len = _batch_item_float(batch.get("accept_lens"), i)
            if accept_len is None:
                num_correct = _batch_item_float(batch.get("num_correct_drafts_per_req_cpu"), i)
                if num_correct is not None:
                    accept_len = max(num_correct + 1.0, 1.0)
            accept_rate = None
            if accept_len is not None:
                verify_tokens = int(self.config.rollout.drafter.rollout.get("spec_verify_tokens", 0) or 0)
                if verify_tokens > 0:
                    accept_rate = accept_len / float(verify_tokens)

            data_item = {
                "input_ids": cpu_input_ids[i, feature_start:feature_end],
                "hidden_states": cpu_h_states[i, hidden_start:hidden_end, :],
                "hidden_states_layout": hidden_states_layout,
                "hidden_positions": kept_hidden_positions,
                "loss_mask": full_loss_mask[feature_start:feature_end],
                "position_ids": item_position_ids,
                "target_logprobs": target_logprobs_item,
                "responses": cpu_responses[i] if cpu_responses is not None else None,
                "prompts": cpu_prompts[i] if cpu_prompts is not None else None,
                "_verl_feature_start": feature_start,
                "_verl_feature_end": feature_end,
                "_verl_hidden_start": hidden_start,
                "_verl_hidden_end": hidden_end,
                "_verl_hidden_position_start": hidden_position_start,
                "_verl_uses_hidden_positions": uses_hidden_positions,
                "_verl_hidden_positions": kept_hidden_positions,
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
                "_verl_hidden_raw_target_position_start": (
                    _batch_item_int(batch.get("hidden_raw_target_logprobs_position_start"), i)
                    if cpu_hidden_raw_target_logprobs is not None
                    else None
                ),
                "_verl_hidden_raw_target_position_end": (
                    _batch_item_int(batch.get("hidden_raw_target_logprobs_position_end"), i)
                    if cpu_hidden_raw_target_logprobs is not None
                    else None
                ),
                "_verl_prompt_len": prompt_len if cpu_prompts is not None else None,
                "_verl_response_len": response_len if cpu_responses is not None else None,
                "_verl_accept_len": accept_len,
                "_verl_accept_rate": accept_rate,
                "_verl_input_seq_length": input_seq_length,
                "hidden_lm_head_fingerprint": batch.get("hidden_lm_head_fingerprint"),
                "hidden_last_hidden_logprob_check": batch.get("hidden_last_hidden_logprob_check"),
                "hidden_target_logprobs_source": batch.get("hidden_target_logprobs_source"),
                "hidden_raw_topk_logprob_check": batch.get("hidden_raw_topk_logprob_check"),
                "hidden_raw_target_logprobs": hidden_raw_target_logprobs_item,
                "hidden_raw_target_logprobs_positions": raw_target_logprobs_positions_item,
                "hidden_last_hidden_filter": batch.get("hidden_last_hidden_filter"),
                "hidden_last_hidden_select": batch.get("hidden_last_hidden_select"),
                "global_step": _batch_item_int(batch.get("global_step"), i),
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
                            if getattr(self.backend, "model_type", None) == "eagle3"
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
                            "hidden_lm_head_fingerprint": data_item.get("hidden_lm_head_fingerprint"),
                            "hidden_last_hidden_logprob_check": data_item.get("hidden_last_hidden_logprob_check"),
                            "hidden_target_logprobs_source": data_item.get("hidden_target_logprobs_source"),
                            "hidden_raw_topk_logprob_check": data_item.get("hidden_raw_topk_logprob_check"),
                            "hidden_raw_target_shape": _tensor_shape(data_item.get("hidden_raw_target_logprobs")),
                            "hidden_raw_target_positions_shape": _tensor_shape(
                                data_item.get("hidden_raw_target_logprobs_positions")
                            ),
                            "hidden_raw_target_position_start": data_item.get(
                                "_verl_hidden_raw_target_position_start"
                            ),
                            "hidden_raw_target_position_end": data_item.get("_verl_hidden_raw_target_position_end"),
                            "hidden_last_hidden_filter": data_item.get("hidden_last_hidden_filter"),
                            "hidden_last_hidden_select": data_item.get("hidden_last_hidden_select"),
                            "item_global_step": data_item.get("global_step"),
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
                    if getattr(self.backend, "model_type", None) == "eagle3":
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
            logger.debug(
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
            logger.debug(f"[Rank {self.rank}] Masked {masked_tokens} drafter targets due to bad input hidden rows")

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

        for target_key in ("target", "last_hidden_states", "target_last_hidden_states"):
            if target_key not in batch:
                continue
            target_tensor, bad_target_rows = self._sanitize_sequence_tensor(batch[target_key], target_key, clip_value)
            batch[target_key] = target_tensor
            masked_tokens = int(((batch["loss_mask"] > 0) & bad_target_rows).detach().sum().cpu().item())
            if masked_tokens > 0:
                logger.debug(
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

    @staticmethod
    def _item_float(item: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
        for key in keys:
            value = item.get(key)
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _dflash_hard_sample_score(self, item: dict[str, Any]) -> Optional[float]:
        """Return a larger-is-harder score for block-drafter sample selection."""
        explicit_score = self._item_float(
            item,
            (
                "_verl_dflash_hard_score",
                "dflash_hard_score",
                "_verl_sample_loss",
                "sample_loss",
            ),
        )
        if explicit_score is not None:
            return explicit_score

        accept_len = self._item_float(
            item,
            (
                "_verl_accept_len",
                "accept_len",
                "accepted_len",
                "num_correct_drafts",
                "num_correct_drafts_per_req",
            ),
        )
        if accept_len is not None:
            return -accept_len

        accept_rate = self._item_float(item, ("_verl_accept_rate", "accept_rate"))
        if accept_rate is not None:
            return -accept_rate

        return None

    def _sample_training_items(
        self,
        available_data: list[dict[str, Any]],
        batch_size: int,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        if len(available_data) <= batch_size:
            return list(available_data)

        hard_ratio = float(self._block_drafter_config_value("hard_sample_ratio", 0.0) or 0.0)
        if not self._is_block_drafter_backend() or hard_ratio <= 0:
            return rng.sample(available_data, batch_size)

        hard_count = min(batch_size, max(0, round(batch_size * hard_ratio)))
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for item in available_data:
            score = self._dflash_hard_sample_score(item)
            if score is not None:
                scored.append((score, rng.random(), item))

        selected: list[dict[str, Any]] = []
        if hard_count > 0 and scored:
            scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
            hard_pool_size = min(len(scored), max(hard_count, hard_count * 4))
            hard_pool = [entry[2] for entry in scored[:hard_pool_size]]
            selected.extend(rng.sample(hard_pool, min(hard_count, len(hard_pool))))

        selected_ids = {id(item) for item in selected}
        remaining = [item for item in available_data if id(item) not in selected_ids]
        random_count = batch_size - len(selected)
        if random_count > 0:
            selected.extend(rng.sample(remaining, min(random_count, len(remaining))))

        return selected

    def _prepare_training_batch(
        self,
        buffer_steps: int = 2,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Prepare a batch for training using Ulysses SP to remove padding.

        Args:
            buffer_steps: Number of recent RL steps to include data from (only used if use_buffer_data=True)

        Returns:
            Dictionary containing batch tensors for training
        """
        effective_batch_size = self.batch_size

        current_step = int(self.current_rl_step)
        sample_seed_step = int(self.training_steps)

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
                rng = random.Random((int(self.current_rl_step) << 16) + sample_seed_step)
                items = self._sample_training_items(
                    available_data,
                    min(len(available_data), effective_batch_size),
                    rng,
                )
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
                rng = random.Random((current_step << 16) + sample_seed_step)
                items = self._sample_training_items(current_step_data, effective_batch_size, rng)

        # Filter out items without the tensors required by the selected loss path.
        items = [item for item in items if "hidden_states" in item]
        if self.backend.model_type == "eagle3" and use_logits:
            items = [item for item in items if item.get("target_logprobs") is not None]
        if len(items) == 0:
            logger.debug(f"[Rank {self.rank}] No items with hidden_states found, cannot prepare batch")
            return None
        elif len(items) < min_items_for_batch:
            logger.debug(
                f"[Rank {self.rank}] Only {len(items)} items with hidden_states found "
                f"(need at least {min_items_for_batch}), cannot prepare batch"
            )
            return None

        dev = next(self.model.parameters()).device
        if self._is_block_drafter_backend() and self.use_ulysses_sp:
            raise NotImplementedError(f"{self.backend.model_type} drafter training does not support Ulysses sequence parallel yet")
        
        preprocessed_lists = self.backend.preprocess_individual_items(items, dev, self.model_config)
        items_seen = len(items)
        items_used = 0
        items_dropped_short = 0
        items_dropped_missing_target = 0
        packed_tokens_before_shift = 0
        packed_loss_tokens = 0
       
        # Build training chunks inside each sample before packing. EAGLE3-style
        # models use next-token chunks, while DFlash keeps same-position blocks
        # per sample so sampled anchors never cross sample boundaries.
        input_id_chunks = []
        loss_mask_chunks = []
        hidden_state_chunks = []
        position_id_chunks = []
        last_hidden_state_chunks = []
        target_logprob_chunks = []
        target_last_hidden_state_chunks = []

        ids_list = preprocessed_lists["ids"]
        hidden_list = preprocessed_lists["h_states"]
        mask_list = preprocessed_lists["masks"]
        position_list = preprocessed_lists.get("position_ids")
        target_last_h_list = preprocessed_lists.get("target_last_h_states")
        dspark_l1_enabled = (
            self.backend.model_type == "dspark"
            and float(self.config.rollout.drafter.training.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0
        )
        if position_list is None:
            position_list = [torch.arange(ids.size(0), device=ids.device, dtype=torch.long) for ids in ids_list]
        for item_idx, (ids, h_states, item_loss_mask, item_position_ids) in enumerate(
            zip(ids_list, hidden_list, mask_list, position_list)
        ):
            source_item = items[item_idx] if item_idx < len(items) else {}
            uses_shifted_eagle_inputs = self.backend.model_type == "eagle3"
            target_last_h_states = (
                target_last_h_list[item_idx]
                if target_last_h_list is not None and item_idx < len(target_last_h_list)
                else None
            )
            seq_len_limits = [ids.size(0), h_states.size(0), item_loss_mask.size(0), item_position_ids.size(0)]
            if torch.is_tensor(target_last_h_states):
                seq_len_limits.append(target_last_h_states.size(0))
            seq_len = min(seq_len_limits)
            if seq_len < 1:
                items_dropped_short += 1
                continue
            if (
                dspark_l1_enabled
                and not torch.is_tensor(target_last_h_states)
            ):
                items_dropped_missing_target += 1
                continue

            target_logprobs_item = None
            target_logprobs_train_start = 0
            if self.backend.model_type == "eagle3" and use_logits:
                target_logprobs_item = preprocessed_lists["target_logprobs"][item_idx]
                if target_logprobs_item is not None:
                    target_logprobs_train_start = _eagle_target_logprobs_train_start(source_item)
            if self.backend.model_type == "eagle3" and use_logits:
                train_seq_len = min(
                    max(ids.size(0) - 2, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 2, 0),
                    item_position_ids.size(0),
                    max(target_logprobs_item.size(0) - target_logprobs_train_start, 0),
                )
            elif self.backend.model_type == "eagle3":
                last_h_states = preprocessed_lists["last_h_states"][item_idx]
                train_seq_len_limits = [
                    max(ids.size(0) - 2, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 2, 0),
                    item_position_ids.size(0),
                    max(last_h_states.size(0) - 1, 0),
                ]
                train_seq_len = min(train_seq_len_limits)
            elif self.backend.model_type == "peagle":
                # P-EAGLE mirrors the reference target-wrapper shift: row p pairs
                # aux f[p] with the NEXT token x[p+1], supervised against the
                # distribution of x[p+2] from last_hidden[p+1] and gated by
                # loss_mask[p+1]. Only aux stays unshifted.
                last_h_states = preprocessed_lists["last_h_states"][item_idx]
                train_seq_len = min(
                    max(ids.size(0) - 1, 0),
                    h_states.size(0),
                    max(item_loss_mask.size(0) - 1, 0),
                    item_position_ids.size(0),
                    max(last_h_states.size(0) - 1, 0),
                )
            elif self._is_block_drafter_backend():
                train_seq_len = seq_len
            else:
                train_seq_len = seq_len - 1

            if train_seq_len < 1:
                items_dropped_missing_target += 1
                continue

            items_used += 1
            packed_tokens_before_shift += train_seq_len
            if self._is_block_drafter_backend():
                packed_loss_tokens += int(item_loss_mask[:train_seq_len].detach().float().sum().cpu().item())
            elif uses_shifted_eagle_inputs:
                packed_loss_tokens += int(item_loss_mask[2 : 2 + train_seq_len].detach().float().sum().cpu().item())
            else:
                packed_loss_tokens += int(item_loss_mask[1 : 1 + train_seq_len].detach().float().sum().cpu().item())

            if alignment_debug_enabled():
                sample_index = self._alignment_debug_sample_index(current_step, "prepare_item")
                train_target_logprobs = None
                if self.backend.model_type == "eagle3" and use_logits and target_logprobs_item is not None:
                    train_target_logprobs = target_logprobs_item[
                        target_logprobs_train_start : target_logprobs_train_start + train_seq_len
                    ]
                source_hidden_positions = source_item.get("hidden_positions")
                if source_hidden_positions is None:
                    source_hidden_positions = source_item.get("_verl_hidden_positions")
                if isinstance(source_hidden_positions, torch.Tensor):
                    source_hidden_positions = source_hidden_positions.reshape(-1)
                    train_hidden_positions = source_hidden_positions[:train_seq_len]
                    base_hidden_position_start = (
                        int(train_hidden_positions[0].item()) if int(train_hidden_positions.numel()) > 0 else None
                    )
                    base_hidden_position_end = (
                        int(train_hidden_positions[-1].item()) + 1 if int(train_hidden_positions.numel()) > 0 else None
                    )
                    last_hidden_position_start = (
                        int(source_hidden_positions[1].item())
                        if uses_shifted_eagle_inputs and int(source_hidden_positions.numel()) > 1
                        else None
                    )
                    last_hidden_position_end = (
                        int(source_hidden_positions[train_seq_len].item()) + 1
                        if uses_shifted_eagle_inputs and int(source_hidden_positions.numel()) > train_seq_len
                        else None
                    )
                else:
                    base_hidden_position_start = source_item.get("_verl_feature_start")
                    base_hidden_position_end = int(source_item.get("_verl_feature_start", 0) or 0) + train_seq_len
                    last_hidden_position_start = (
                        int(source_item.get("_verl_feature_start", 0) or 0) + 1
                    ) if uses_shifted_eagle_inputs else None
                    last_hidden_position_end = (
                        int(source_item.get("_verl_feature_start", 0) or 0) + 1 + train_seq_len
                    ) if uses_shifted_eagle_inputs else None
                effective_target_position_start = source_item.get("_verl_target_position_start")
                effective_target_position_end = source_item.get("_verl_target_position_end")
                if self.backend.model_type == "eagle3" and not use_logits:
                    effective_target_position_start = last_hidden_position_start
                    effective_target_position_end = last_hidden_position_end
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
                            "uses_hidden_positions": source_item.get("_verl_uses_hidden_positions"),
                            "base_hidden_position_start": base_hidden_position_start,
                            "base_hidden_position_end": base_hidden_position_end,
                            "last_hidden_position_start": last_hidden_position_start,
                            "last_hidden_position_end": last_hidden_position_end,
                            "target_start": source_item.get("_verl_target_start"),
                            "target_end": source_item.get("_verl_target_end"),
                            "target_position_start": effective_target_position_start,
                            "target_position_end": effective_target_position_end,
                            "target_train_start": target_logprobs_train_start,
                            "target_train_position_start": (
                                int(source_item.get("_verl_target_position_start", 0) or 0)
                                + int(target_logprobs_train_start)
                            )
                            if source_item.get("_verl_target_position_start") is not None
                            else None,
                            "target_train_position_end": (
                                int(source_item.get("_verl_target_position_start", 0) or 0)
                                + int(target_logprobs_train_start)
                                + train_seq_len
                            )
                            if source_item.get("_verl_target_position_start") is not None
                            else None,
                            "target_tensor_position_start": source_item.get("_verl_target_tensor_position_start"),
                            "target_tensor_position_end": source_item.get("_verl_target_tensor_position_end"),
                            "item_global_step": source_item.get("global_step"),
                            "hidden_lm_head_fingerprint": source_item.get("hidden_lm_head_fingerprint"),
                            "hidden_last_hidden_logprob_check": source_item.get("hidden_last_hidden_logprob_check"),
                            "hidden_target_logprobs_source": source_item.get("hidden_target_logprobs_source"),
                            "hidden_raw_topk_logprob_check": source_item.get("hidden_raw_topk_logprob_check"),
                            "hidden_raw_target_shape": _tensor_shape(source_item.get("hidden_raw_target_logprobs")),
                            "hidden_raw_target_positions_shape": _tensor_shape(
                                source_item.get("hidden_raw_target_logprobs_positions")
                            ),
                            "hidden_raw_target_position_start": source_item.get(
                                "_verl_hidden_raw_target_position_start"
                            ),
                            "hidden_raw_target_position_end": source_item.get(
                                "_verl_hidden_raw_target_position_end"
                            ),
                            "hidden_last_hidden_filter": source_item.get("hidden_last_hidden_filter"),
                            "hidden_last_hidden_select": source_item.get("hidden_last_hidden_select"),
                            "target_lm_head_weight_step": getattr(self, "_target_lm_head_weight_step", None),
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

            if uses_shifted_eagle_inputs or self.backend.model_type == "peagle":
                input_id_chunks.append(ids[1 : 1 + train_seq_len])
                hidden_state_chunks.append(h_states[:train_seq_len])
                position_id_chunks.append(item_position_ids[:train_seq_len])
            else:
                input_id_chunks.append(ids[:train_seq_len])
                hidden_state_chunks.append(h_states[:train_seq_len])
                position_id_chunks.append(item_position_ids[:train_seq_len])
            if self._is_block_drafter_backend():
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
            elif self.backend.model_type == "peagle":
                # Reference-shifted: last_hidden[p+1] scores x[p+2], the token
                # after the drafted input token x[p+1] at row p.
                last_hidden_state_chunks.append(last_h_states[1 : 1 + train_seq_len])
            elif dspark_l1_enabled and torch.is_tensor(target_last_h_states):
                target_last_hidden_state_chunks.append(target_last_h_states[:train_seq_len])

        if not input_id_chunks:
            return None

        if self._is_block_drafter_backend():
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
            target_last_hidden_states = None
            if target_last_hidden_state_chunks:
                if len(target_last_hidden_state_chunks) != len(input_id_chunks):
                    logger.warning(
                        "[dspark-trainer] dropping batch with partial target_last_hidden_states: "
                        "target_rows=%s batch_rows=%s",
                        len(target_last_hidden_state_chunks),
                        len(input_id_chunks),
                    )
                    return None
                target_hidden_dim = target_last_hidden_state_chunks[0].size(-1)
                target_last_hidden_states = torch.zeros(
                    len(target_last_hidden_state_chunks),
                    max_train_len,
                    target_hidden_dim,
                    dtype=target_last_hidden_state_chunks[0].dtype,
                    device=dev,
                )
                for row_idx, target_h_chunk in enumerate(target_last_hidden_state_chunks):
                    target_last_hidden_states[row_idx, : target_h_chunk.size(0)] = target_h_chunk
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
        elif self.backend.model_type == "peagle":
            if not last_hidden_state_chunks:
                return None
            last_hidden_states = torch.cat(last_hidden_state_chunks, dim=0).unsqueeze(0).contiguous()

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
        elif self.backend.model_type == "peagle":
            batch["last_hidden_states"] = last_hidden_states
            # Preserve the per-document chunk lengths so the P-EAGLE COD mask can
            # isolate documents. The flat batch concatenates every chunk into one
            # length-`sum` sequence with an all-ones attention_mask, so the mask
            # cannot recover document boundaries from attention_mask alone.
            batch["seq_lengths"] = torch.tensor(
                [chunk.size(0) for chunk in input_id_chunks], dtype=torch.long, device=dev
            )
        elif self.backend.model_type == "dspark" and target_last_hidden_state_chunks:
            batch["target_last_hidden_states"] = target_last_hidden_states

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
        elif self.backend.model_type == "peagle":
            # Carry the sanitized P-EAGLE tensors across the batch rebuild below.
            # They are not padded/sliced here: the backend rejects Ulysses SP in
            # compute_loss (supports_ulysses_sp=False), so peagle batches must
            # reach it unsliced.
            peagle_last_hidden_states = batch["last_hidden_states"]
            peagle_seq_lengths = batch["seq_lengths"]
        elif self.backend.model_type == "dspark" and "target_last_hidden_states" in batch:
            target_last_hidden_states = batch["target_last_hidden_states"]

        # Use Ulysses SP to pad and slice if needed.
        pad_size_for_batch = 0
        if self.use_ulysses_sp:
            from verl.utils.ulysses import slice_input_tensor, ulysses_pad_and_slice_inputs
            # Pad to be divisible by SP size and slice across ranks
            input_ids, position_ids, pad_size = ulysses_pad_and_slice_inputs(
                input_ids, position_ids, sp_size=self.ulysses_sequence_parallel_size
            )
            pad_size_for_batch = pad_size

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
                elif self.backend.model_type == "dspark" and "target_last_hidden_states" in batch:
                    target_last_hidden_states = torch.nn.functional.pad(
                        target_last_hidden_states, (0, 0, 0, pad_size), value=0.0
                    )

            # Slice for this rank
            loss_mask = slice_input_tensor(loss_mask, dim=1, padding=False)
            base_h = slice_input_tensor(base_h, dim=1, padding=False)
            attn_mask = slice_input_tensor(attn_mask, dim=1, padding=False)
            if self.backend.model_type == "eagle3":
                if use_logits:
                    target_logprobs = slice_input_tensor(target_logprobs, dim=1, padding=False)
                else:
                    last_hidden_states = slice_input_tensor(last_hidden_states, dim=1, padding=False)
            elif self.backend.model_type == "dspark" and "target_last_hidden_states" in batch:
                target_last_hidden_states = slice_input_tensor(target_last_hidden_states, dim=1, padding=False)

            # Store pad_size for later gathering
            self._current_pad_size = pad_size_for_batch
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
        elif self.backend.model_type == "peagle":
            batch["last_hidden_states"] = peagle_last_hidden_states
            batch["seq_lengths"] = peagle_seq_lengths
        elif self.backend.model_type == "dspark" and target_last_hidden_state_chunks:
            batch["target_last_hidden_states"] = target_last_hidden_states
        batch["_speco_pad_size"] = pad_size_for_batch

        if alignment_debug_enabled():
            final_target = None
            if self.backend.model_type == "eagle3" and use_logits:
                final_target = batch.get("target_logprobs")
            elif self.backend.model_type == "eagle3":
                final_target = batch.get("last_hidden_states")

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
                        "source_global_steps": [item.get("global_step") for item in items[:8]],
                        "source_hidden_lm_head_fingerprints": [
                            item.get("hidden_lm_head_fingerprint") for item in items[:2]
                        ],
                        "source_last_hidden_logprob_checks": [
                            item.get("hidden_last_hidden_logprob_check") for item in items[:2]
                        ],
                        "source_target_logprobs_sources": [
                            item.get("hidden_target_logprobs_source") for item in items[:2]
                        ],
                        "source_raw_topk_checks": [
                            item.get("hidden_raw_topk_logprob_check") for item in items[:2]
                        ],
                        "source_last_hidden_filters": [item.get("hidden_last_hidden_filter") for item in items[:2]],
                        "source_last_hidden_selects": [item.get("hidden_last_hidden_select") for item in items[:2]],
                        "target_lm_head_weight_step": getattr(self, "_target_lm_head_weight_step", None),
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

    def prepare_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        with self._ulysses_group_context():
            return self._prepare_training_batch()

    def add_feature_sample(self, sample: DraftFeatureSample | dict[str, Any]) -> None:
        """Append a normalized standalone sample to the in-memory training buffer."""
        if isinstance(sample, DraftFeatureSample):
            item = sample.to_training_item()
        elif isinstance(sample, dict):
            item = DraftFeatureSample.from_dict(sample, strict=False).to_training_item()
        else:
            raise TypeError(f"Unsupported draft feature sample type: {type(sample)!r}")
        item["step"] = int(item.get("step", self.current_rl_step) or self.current_rl_step)
        self.collected_data.append(item)

    def prepare_training_batch_from_samples(
        self,
        samples: list[DraftFeatureSample | dict[str, Any]],
        *,
        step: Optional[int] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Prepare a training batch directly from standalone feature samples."""
        current_step = int(self.current_rl_step if step is None else step)
        previous_collected_data = self.collected_data
        previous_current_step = self.current_rl_step
        maxlen = max(len(samples), int(self.config.rollout.drafter.training.get("current_max_samples", 2000)))
        self.collected_data = deque(maxlen=maxlen)
        self.current_rl_step = current_step
        try:
            for sample in samples:
                if isinstance(sample, DraftFeatureSample):
                    item = sample.to_training_item()
                elif isinstance(sample, dict):
                    item = DraftFeatureSample.from_dict(sample, strict=False).to_training_item()
                else:
                    raise TypeError(f"Unsupported draft feature sample type: {type(sample)!r}")
                item["step"] = current_step
                self.collected_data.append(item)
            with self._ulysses_group_context():
                return self._prepare_training_batch()
        finally:
            self.collected_data = previous_collected_data
            self.current_rl_step = previous_current_step

    async def training_step_from_batch(self, batch: dict[str, torch.Tensor], step: int) -> bool:
        """Execute one optimizer step from a pre-built standalone batch."""
        try:
            with torch.enable_grad():
                return await self._training_step_on_batch(batch, step)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Standalone training step {step} failed with error: {e}")
            return False

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
            logger.debug("No model available for training")
            return False

        # Skip training only when no hidden-state collection path is enabled.
        training_cfg = self.config.rollout.drafter.training
        collect_hidden_states_from_sgl = bool(training_cfg.get("collect_hidden_states_from_sgl", False))
        collect_hidden_states_from_old_logprob = bool(
            training_cfg.get("collect_hidden_states_from_old_logprob", False)
        )
        if not (collect_hidden_states_from_sgl or collect_hidden_states_from_old_logprob):
            logger.debug(
                f"[DrafterTrainer rank {self.rank}] Skipping training step {step} "
                "because hidden-state collection is disabled"
            )
            return False

        prepare_ts = time.time()
        batch = self.prepare_training_batch()
        self.record_training_timing("timing_s/drafter_prepare_batch", time.time() - prepare_ts)
        if not self._sync_batch_readiness(batch is not None):
            logger.debug(f"[DrafterTrainer rank {self.rank}] Skipping step {step} due to missing drafter batch")
            return False
        if batch is None:
            logger.debug(
                f"[DrafterTrainer rank {self.rank}] Not enough data at step {step} "
                f"(have={len(self.collected_data)} need>={self.batch_size})"
            )
            return False
        
        return await self._training_step_on_batch(batch, step)

    def _reduce_loss_metrics(
        self, l_v: torch.Tensor, l_p: torch.Tensor, l_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """All-reduce detached loss/token sums across the SP/DP groups.

        Returns the global sums plus the number of ranks that participated in
        the reduction. The all-reduce is detached, so the caller must scale its
        local loss by that world size to cancel FSDP's gradient averaging over
        the same ranks.
        """
        metrics = torch.stack([l_v.detach(), l_p.detach(), l_n.detach()])
        reduce_world_size = 1
        sp_group = self._get_sp_group()
        dp_group = self._get_dp_group()
        if sp_group is not None and self._get_sp_world_size() > 1:
            dist.all_reduce(metrics, group=sp_group)
            reduce_world_size *= self._get_sp_world_size()
            if dp_group is not None and self._get_dp_world_size() > 1:
                dist.all_reduce(metrics, group=dp_group)
                reduce_world_size *= self._get_dp_world_size()
        elif dp_group is not None and self._get_dp_world_size() > 1:
            dist.all_reduce(metrics, group=dp_group)
            reduce_world_size *= self._get_dp_world_size()
        elif self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            dist.all_reduce(metrics, group=self.training_device_mesh.get_group())
            reduce_world_size *= self.training_device_mesh.size()
        return metrics[0], metrics[1], metrics[2], reduce_world_size

    async def _training_step_on_batch(self, batch: dict[str, torch.Tensor], step: int) -> bool:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass.
        forward_ts = time.time()
        with self._ulysses_group_context():
            with torch.amp.autocast(device_type=device_name, dtype=torch.bfloat16):
                pad_size = int(batch.get("_speco_pad_size", self._current_pad_size) or 0)
                loss_batch = {key: value for key, value in batch.items() if not key.startswith("_speco_")}
                loss_dict = self.backend.compute_loss(self.model, loss_batch, pad_size)

                l_v = loss_dict["total_local_vloss"]
                l_p = loss_dict["total_local_ploss"]
                l_n = loss_dict["local_num_tokens"]
                self._record_dflash_training_metrics(loss_dict)
        self.record_training_timing("timing_s/drafter_forward_loss", time.time() - forward_ts)

        # Reduce scalar loss statistics once across SP/DP groups.
        reduce_ts = time.time()
        global_vloss, global_ploss, global_tokens, reduce_world_size = self._reduce_loss_metrics(l_v, l_p, l_n)
        self.record_training_timing("timing_s/drafter_reduce_loss", time.time() - reduce_ts)

        if float(global_tokens.detach().float().item()) <= 0:
            logger.debug(
                f"Step {self.training_steps + 1}: no finite drafter target tokens, skipping optimizer step"
            )
            return False

        denom = global_tokens.clamp(min=1.0)
        vloss = global_vloss / denom
        ploss = global_ploss / denom

        # Global token-mean loss, identical on every rank; used for guards and logging.
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

        # Backward on this rank's local loss sums: the metric all-reduce above is
        # outside the autograd graph, so each rank's backward carries only its own
        # contribution, and FSDP then averages gradients across the same
        # `reduce_world_size` ranks. Scaling by `reduce_world_size` cancels that
        # mean, making the synchronized gradient the exact global token-mean
        # gradient regardless of world size.
        local_loss = (loss_dict["v_weight"] * l_v + loss_dict["p_weight"] * l_p) * (
            float(reduce_world_size) / denom
        )
        backward_ts = time.time()
        local_loss.backward()
        self.record_training_timing("timing_s/drafter_backward", time.time() - backward_ts)

        # 更新权重
        optimizer_ts = time.time()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            logger.error(
                f"Step {self.training_steps + 1}: non-finite drafter grad norm, "
                f"grad_norm={float(grad_norm.detach().float().item())}"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return False
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer_steps_total += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.record_training_timing("timing_s/drafter_optimizer", time.time() - optimizer_ts)

        self.training_steps += 1
        logger.warning(
            "[drafter loss] step=%s optimizer_step_total=%s lr=%.3e loss=%.4f vloss=%.4f ploss=%.4f",
            self.training_steps,
            self.optimizer_steps_total,
            current_lr,
            float(loss.item()),
            float(vloss.item()),
            float(ploss.item()),
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
                    logger.debug(
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

            result = {
                k: (
                    v.detach().to(dtype=dtype, device="cpu").contiguous()
                    if dtype is not None
                    else v.detach().cpu().contiguous()
                )
                for k, v in trainable_state.items()
            }

            return result
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
            logger.debug(
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

        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to clear drafter gradients during cleanup: {e}")

        if self.skip_heavy_cleanup_after_drafter_training:
            if clear_data:
                self.collected_data.clear()
                self.data_buffer.clear()
            self._training_initialized = False
            self._training_active = False
            self._last_ckpt_step = -1
            self.training_steps = 0
            logger.debug(
                "[Rank %s] Skipped heavy drafter cleanup; model/optimizer stay on runtime device",
                self.rank,
            )
            return

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
        self._training_active = False
        self._last_ckpt_step = -1
        self.training_steps = 0

    async def release_training_memory_after_activation(self):
        """Release runtime-device memory after a pre-fit activation warmup."""

        self._training_active = False

        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to clear drafter gradients after activation warmup: {e}")

        if self.model is not None:
            try:
                offload_fsdp_model_to_cpu(self.model)
                logger.debug("Offloaded drafter model to CPU after activation warmup")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter model after activation warmup: {e}")

        if self.optimizer is not None:
            try:
                offload_fsdp_optimizer(self.optimizer)
                logger.debug("Offloaded drafter optimizer state to CPU after activation warmup")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter optimizer after activation warmup: {e}")

        target_model = getattr(self.backend, "target_model", None)
        if target_model is not None:
            try:
                target_model.to("cpu")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter target model after activation warmup: {e}")

        if device_name != "cpu" and hasattr(self.device_module, "empty_cache"):
            if hasattr(self.device_module, "synchronize"):
                self.device_module.synchronize()
            self.device_module.empty_cache()

        self._training_initialized = False
