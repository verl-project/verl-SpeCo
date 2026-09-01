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
"""Reconstruct standalone draft-training features from compact token samples."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import torch
from torch import nn

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.trainer.feature_store import DraftFeatureSample, DraftReplaySample

logger = logging.getLogger(__name__)


class HiddenStateAlignmentError(ValueError):
    """The returned hidden states cannot cover the requested training sample."""


@dataclass(frozen=True)
class FeatureContract:
    """Explicit inputs for converting one vLLM payload into a training sample."""

    algorithm: str
    target_layer_ids: list[int]
    hidden_states_layout: str
    dtype: torch.dtype
    target_model_id: str
    target_model_revision: str | None
    tokenizer_fingerprint: str
    use_logits: bool = False
    target_config_fingerprint: str | None = None
    source: str = "standalone_tq_producer"
    require_full_alignment: bool = False


@dataclass
class _VllmEndpointState:
    index: int
    url: str
    client: Any | None = None
    model: str | None = None
    inflight: int = 0
    requests: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    request_seconds: float = 0.0
    cooldown_until: float = 0.0


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_vllm_endpoints(config: Any) -> list[str]:
    configured = _config_value(config, "vllm_endpoints", None)
    if configured is None:
        configured = [
            _config_value(config, "vllm_endpoint", "http://localhost:8000/v1")
        ]
    elif isinstance(configured, str):
        configured = [configured]
    endpoints: list[str] = []
    for value in configured:
        endpoint = str(value or "").strip().rstrip("/")
        if endpoint and endpoint not in endpoints:
            endpoints.append(endpoint)
    if not endpoints:
        raise ValueError(
            "target_feature_replay.vllm_endpoints must contain at least one URL"
        )
    return endpoints


def _parse_dtype(value: Any) -> torch.dtype:
    normalized = str(value or "bfloat16").strip().lower()
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtypes:
        raise ValueError(
            f"Unsupported target feature replay dtype {value!r}; "
            "expected bfloat16, float16, or float32"
        )
    return dtypes[normalized]


def _tensor_from_module_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return cast(torch.Tensor, value)
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return cast(torch.Tensor, value[0])
    last_hidden_state = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(last_hidden_state):
        return cast(torch.Tensor, last_hidden_state)
    raise TypeError(
        f"Target feature replay expected tensor-like module output, got {type(value)!r}"
    )


def _get_module_by_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not part:
            continue
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _find_layers_and_final_norm(model: nn.Module) -> tuple[list[nn.Module], nn.Module]:
    roots: list[nn.Module] = [model]
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module) and base_model is not model:
        roots.append(base_model)

    candidates = (
        ("model.layers", "model.norm"),
        ("base_model.model.layers", "base_model.model.norm"),
        ("model.decoder.layers", "model.decoder.final_layer_norm"),
        ("transformer.h", "transformer.ln_f"),
        ("gpt_neox.layers", "gpt_neox.final_layer_norm"),
    )
    for root in roots:
        for layers_path, norm_path in candidates:
            layers = _get_module_by_path(root, layers_path)
            norm = _get_module_by_path(root, norm_path)
            if (
                isinstance(layers, (nn.ModuleList, list, tuple))
                and len(layers) > 0
                and isinstance(norm, nn.Module)
            ):
                return list(layers), norm

        for name, child in root.named_modules():
            if not isinstance(child, nn.ModuleList) or len(child) <= 0:
                continue
            if not (name.endswith("layers") or name.endswith("h")):
                continue
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            for norm_name in ("norm", "final_layer_norm", "ln_f"):
                norm_path = f"{parent_path}.{norm_name}" if parent_path else norm_name
                norm = _get_module_by_path(root, norm_path)
                if isinstance(norm, nn.Module):
                    return list(child), norm

    raise RuntimeError(
        "Target feature replay could not find transformer layers and final norm"
    )


def _hidden_capture_target(layer_id: int, num_layers: int) -> tuple[str, int | None]:
    hidden_state_index = (
        int(layer_id) + 1 if int(layer_id) >= 0 else num_layers + 1 + int(layer_id)
    )
    if hidden_state_index <= 0 or hidden_state_index > num_layers:
        raise IndexError(
            f"Target replay layer id {layer_id} resolved to hidden-state index "
            f"{hidden_state_index}, but the model has {num_layers} layers"
        )
    if hidden_state_index == num_layers:
        return "final", None
    return "layer", hidden_state_index - 1


def _load_json_config(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    config_path = os.path.join(os.fspath(path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            value = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_for_lock(lock_path: Path, timeout: float = 30.0) -> None:
    if not lock_path.exists():
        return
    try:
        import fcntl
    except ImportError:
        deadline = time.monotonic() + float(timeout)
        while lock_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for hidden-states lock: {lock_path}"
                )
            time.sleep(0.1)
        return

    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + float(timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for hidden-states lock: {lock_path}"
                    ) from None
                time.sleep(0.1)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    try:
        lock_path.unlink()
    except OSError:
        pass


class BoundedReplayCache:
    """Per-rank disk cache with a hard least-recently-used size budget."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_size_gb: float,
        rank: int,
        world_size: int,
    ):
        global_max_bytes = max(int(float(max_size_gb) * 1024**3), 0)
        self.max_bytes = global_max_bytes // max(int(world_size), 1)
        self.path = Path(path) / f"rank{int(rank):05d}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._entries: dict[Path, tuple[int, float]] = {}
        self._total_bytes = 0
        self._scan()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def _scan(self) -> None:
        self._entries = {}
        self._total_bytes = 0
        for path in self.path.glob("*.pt"):
            try:
                stat = path.stat()
            except OSError:
                continue
            size = int(stat.st_size)
            self._entries[path] = (size, float(stat.st_mtime))
            self._total_bytes += size

    def get(self, key: str) -> DraftFeatureSample | None:
        if not self.enabled:
            return None
        path = self.path / f"{key}.pt"
        if not path.exists():
            return None
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            sample = DraftFeatureSample.from_dict(payload, strict=True)
            now = time.time()
            os.utime(path, (now, now))
            size = int(path.stat().st_size)
            self._entries[path] = (size, now)
            return sample
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Discard invalid target replay cache entry %s: %s", path, exc
            )
            self._forget(path)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def put(self, key: str, sample: DraftFeatureSample) -> bool:
        if not self.enabled:
            return False
        path = self.path / f"{key}.pt"
        if path.exists():
            return True
        with tempfile.NamedTemporaryFile(
            prefix=path.name,
            suffix=".tmp",
            dir=self.path,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            torch.save(sample.to_dict(), tmp_path)
            size = int(tmp_path.stat().st_size)
            if size > self.max_bytes:
                return False
            self._evict_until_fits(size)
            os.replace(tmp_path, path)
            now = time.time()
            self._entries[path] = (size, now)
            self._total_bytes += size
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write target replay cache entry %s: %s", path, exc
            )
            return False
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _evict_until_fits(self, incoming_size: int) -> None:
        entries = sorted(self._entries.items(), key=lambda item: item[1][1])
        for path, _ in entries:
            if self._total_bytes + incoming_size <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            self._forget(path)

    def _forget(self, path: Path) -> None:
        previous = self._entries.pop(path, None)
        if previous is not None:
            self._total_bytes = max(self._total_bytes - int(previous[0]), 0)

    def metrics(self) -> dict[str, float]:
        return {
            "replay/cache_size_gb": self._total_bytes / float(1024**3),
            "replay/cache_budget_gb_per_rank": self.max_bytes / float(1024**3),
        }


def feature_from_vllm_payload(
    payload: Mapping[str, Any] | Any,
    request: DraftReplaySample | Any,
    feature_config: FeatureContract,
) -> DraftFeatureSample:
    """Pure vLLM payload conversion shared by replay and standalone Producer."""

    values = getattr(payload, "payload", payload)
    if not isinstance(values, Mapping):
        raise TypeError("vLLM hidden-states payload must be a mapping")
    token_ids = values.get("token_ids")
    hidden = values.get("hidden_states")
    if token_ids is None or hidden is None:
        raise ValueError(
            "vLLM hidden-states payload must contain token_ids and hidden_states"
        )
    if not torch.is_tensor(token_ids) or not torch.is_tensor(hidden):
        raise ValueError(
            "vLLM hidden-states payload must contain token_ids and hidden_states"
        )
    feature_positions = request.feature_positions.detach().cpu().long()
    if int(feature_positions.numel()) <= 0:
        raise ValueError("vLLM feature positions must not be empty")
    feature_end_for_request = int(feature_positions[-1].item()) + 1
    expected_prompt_ids = (
        list(request.prompt_token_ids)
        if hasattr(request, "prompt_token_ids")
        else request.input_ids[:feature_end_for_request].detach().cpu().long().tolist()
    )
    if token_ids.detach().cpu().long().tolist() != expected_prompt_ids:
        raise HiddenStateAlignmentError(
            "vLLM hidden-states token_ids do not match replay input"
        )
    if hidden.dim() != 3:
        raise ValueError(
            "vLLM hidden_states must have shape [seq, layers, hidden], "
            f"got {tuple(hidden.shape)}"
        )

    algorithm = str(feature_config.algorithm).strip().upper()
    if algorithm not in {"EAGLE3", "DFLASH", "DSPARK"}:
        raise ValueError(f"Unsupported vLLM feature algorithm {algorithm!r}")
    target_layer_ids = [int(layer_id) for layer_id in feature_config.target_layer_ids]
    if not target_layer_ids:
        raise ValueError("FeatureContract.target_layer_ids must not be empty")
    hidden_layout = str(feature_config.hidden_states_layout)
    if hidden_layout not in {
        "eagle3_aux_plus_last",
        "dflash_aux",
        "dflash_aux_plus_last",
    }:
        raise ValueError(f"Unsupported vLLM hidden_states_layout {hidden_layout!r}")

    hidden_position_offset = max(len(expected_prompt_ids) - int(hidden.size(0)), 0)
    include_final = hidden_layout in {
        "eagle3_aux_plus_last",
        "dflash_aux_plus_last",
    }
    required_layers = len(target_layer_ids) + (1 if include_final else 0)
    if int(hidden.size(1)) < required_layers:
        raise HiddenStateAlignmentError(
            "vLLM hidden_states layer count is too small: "
            f"got {int(hidden.size(1))}, need at least {required_layers}. "
            "Start vLLM with target layer ids plus the final layer when the "
            "training layout needs last hidden states."
        )
    relative_positions = feature_positions - hidden_position_offset
    keep_mask = (relative_positions >= 0) & (relative_positions < int(hidden.size(0)))
    filtered = not bool(keep_mask.all().item())
    if filtered:
        if feature_config.require_full_alignment:
            raise HiddenStateAlignmentError(
                "vLLM hidden-state rows do not cover the complete feature window: "
                f"hidden_rows={int(hidden.size(0))}, "
                f"hidden_position_offset={hidden_position_offset}, "
                f"dropped={int((~keep_mask).sum().item())}, "
                f"feature_min={int(feature_positions.min().item())}, "
                f"feature_max={int(feature_positions.max().item())}"
            )
        logger.warning(
            "Dropping vLLM feature positions outside hidden rows dropped=%s "
            "hidden_rows=%s hidden_offset=%s feature_min=%s feature_max=%s",
            int((~keep_mask).sum().item()),
            int(hidden.size(0)),
            hidden_position_offset,
            int(feature_positions.min().item()),
            int(feature_positions.max().item()),
        )
        feature_positions = feature_positions[keep_mask]
        relative_positions = relative_positions[keep_mask]
        if int(feature_positions.numel()) <= 0:
            raise HiddenStateAlignmentError(
                "vLLM hidden_states contain no rows for replay feature positions: "
                f"hidden_rows={int(hidden.size(0))}, "
                f"hidden_position_offset={hidden_position_offset}"
            )

    selected = hidden.index_select(0, relative_positions).to(dtype=feature_config.dtype)
    aux_hidden = selected[:, : len(target_layer_ids), :].flatten(1)
    if include_final:
        final_hidden = selected[:, required_layers - 1, :]
        output_hidden = torch.cat([aux_hidden, final_hidden], dim=-1)
    else:
        output_hidden = aux_hidden
    selected_input_ids = request.input_ids.index_select(0, feature_positions).long()
    selected_loss_mask = request.loss_mask.index_select(0, feature_positions).float()
    draft_position_ids = request.draft_position_ids.detach().cpu().long()
    if filtered:
        draft_position_ids = draft_position_ids[keep_mask]

    source_metadata = getattr(request, "source_metadata", None)
    if source_metadata is None:
        source_metadata = getattr(request, "metadata", {})
    metadata = dict(source_metadata or {})
    feature_start = int(feature_positions[0].item())
    feature_end = int(feature_positions[-1].item()) + 1
    metadata.update(
        {
            "source": feature_config.source,
            "target_model_path": feature_config.target_model_id,
            "target_revision": feature_config.target_model_revision,
            "target_config_fingerprint": feature_config.target_config_fingerprint,
            "tokenizer_fingerprint": feature_config.tokenizer_fingerprint,
            "target_layer_ids": target_layer_ids,
            "vllm_hidden_layers": int(hidden.size(1)),
            "vllm_hidden_rows": int(hidden.size(0)),
            "vllm_hidden_position_offset": hidden_position_offset,
            "hidden_states_layout": hidden_layout,
            "feature_start": feature_start,
            "feature_end": feature_end,
            "hidden_position_start": feature_start,
            "hidden_position_end": feature_end,
            "hidden_positions": feature_positions,
            "sequence_length": int(selected_input_ids.numel()),
            "full_sequence_length": int(request.input_ids.numel()),
            "use_logits": feature_config.use_logits,
        }
    )
    return DraftFeatureSample(
        algorithm=algorithm,
        input_ids=selected_input_ids,
        loss_mask=selected_loss_mask,
        hidden_states=output_hidden.cpu().contiguous(),
        position_ids=draft_position_ids,
        metadata=metadata,
    )


class TargetFeatureReplayer:
    """Materialize target hidden states only for standalone token replay."""

    def __init__(
        self,
        config: Any,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.draft_config = config.actor_rollout_ref
        self.drafter_cfg = self.draft_config.rollout.drafter
        self.training_cfg = self.drafter_cfg.training
        self.replay_cfg = self.training_cfg.get("target_feature_replay", {}) or {}
        self.backend = (
            str(_config_value(self.replay_cfg, "backend", "torch") or "torch")
            .strip()
            .lower()
        )
        if self.backend not in {"torch", "vllm_file"}:
            raise ValueError(
                f"Unsupported target_feature_replay.backend={self.backend!r}; "
                "expected 'torch' or 'vllm_file'"
            )
        configured_model_path = _config_value(self.replay_cfg, "model_path", None)
        model_path = configured_model_path or self.draft_config.model.path
        if not model_path:
            raise ValueError(
                "Token replay requires target_feature_replay.model_path or "
                "actor_rollout_ref.model.path"
            )
        self.model_path = os.fspath(model_path)
        self.target_revision = str(
            _config_value(self.replay_cfg, "target_revision", None) or self.model_path
        )
        self.dtype = _parse_dtype(_config_value(self.replay_cfg, "dtype", "bfloat16"))
        self.trust_remote_code = bool(
            _config_value(self.replay_cfg, "trust_remote_code", False)
        )
        self.strict_target_model_path = bool(
            _config_value(self.replay_cfg, "strict_target_model_path", False)
        )
        self.algorithm = str(self.drafter_cfg.speculative_algorithm).upper()
        if self.algorithm not in {"EAGLE3", "DFLASH", "DSPARK"}:
            raise ValueError(
                f"Token replay does not support drafter algorithm {self.algorithm!r}"
            )
        self.use_logits = bool(self.training_cfg.get("use_logits", False))
        self.logits_topk = int(self.training_cfg.get("logits_topk", 128) or 128)
        self.logits_chunk_rows = max(
            int(_config_value(self.replay_cfg, "logits_chunk_rows", 32) or 32), 1
        )
        self.vllm_endpoints = _normalize_vllm_endpoints(self.replay_cfg)
        self.vllm_endpoint = self.vllm_endpoints[0]
        self.vllm_model = _config_value(self.replay_cfg, "vllm_model", None)
        self.vllm_timeout = float(
            _config_value(self.replay_cfg, "request_timeout", 120.0) or 120.0
        )
        self.vllm_max_retries = max(
            int(_config_value(self.replay_cfg, "max_retries", 3) or 0), 0
        )
        self.vllm_endpoint_cooldown = max(
            float(_config_value(self.replay_cfg, "endpoint_cooldown", 5.0) or 0.0),
            0.0,
        )
        self.vllm_on_generate = (
            str(_config_value(self.replay_cfg, "on_generate", "delete") or "delete")
            .strip()
            .lower()
        )
        if self.vllm_on_generate not in {"delete", "keep"}:
            raise ValueError(
                "target_feature_replay.on_generate must be 'delete' or 'keep'"
            )
        self.vllm_require_arange_positions = bool(
            _config_value(self.replay_cfg, "require_arange_positions", True)
        )

        from transformers import AutoConfig

        self.target_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.target_num_hidden_layers = int(
            getattr(
                getattr(self.target_config, "text_config", self.target_config),
                "num_hidden_layers",
            )
        )
        model_configs = [
            value
            for value in (
                _load_json_config(self.drafter_cfg.get("model_path", None)),
                _load_json_config(self.drafter_cfg.get("checkpoint_path", None)),
            )
            if value is not None
        ]
        layer_ids = resolve_oldlogprob_aux_layer_ids(
            self.drafter_cfg,
            target_num_hidden_layers=self.target_num_hidden_layers,
            model_configs=model_configs,
        )
        if not layer_ids:
            raise RuntimeError(
                "Token replay could not resolve target auxiliary layer ids"
            )
        self.target_layer_ids = [int(layer_id) for layer_id in layer_ids]
        dspark_l1_enabled = (
            self.algorithm == "DSPARK"
            and float(self.training_cfg.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0
        )
        self.hidden_layout = (
            "dflash_aux_plus_last"
            if dspark_l1_enabled
            else "dflash_aux"
            if self.algorithm in {"DFLASH", "DSPARK"}
            else "eagle3_aux_plus_last"
        )
        config_json = json.dumps(
            self.target_config.to_dict(), sort_keys=True, default=str
        ).encode()
        self.target_config_fingerprint = hashlib.sha256(config_json).hexdigest()

        self.cache: BoundedReplayCache | None = None
        self._cache_lock = threading.Lock()
        cache_cfg = _config_value(self.replay_cfg, "cache", {}) or {}
        if bool(_config_value(cache_cfg, "enabled", False)):
            cache_path = _config_value(cache_cfg, "path", None)
            if not cache_path:
                feature_path = os.fspath(self.training_cfg.feature_store.path)
                cache_path = f"{feature_path}.hidden_cache"
            self.cache = BoundedReplayCache(
                cache_path,
                max_size_gb=float(_config_value(cache_cfg, "max_size_gb", 0.0) or 0.0),
                rank=self.rank,
                world_size=self.world_size,
            )

        self.model: nn.Module | None = None
        self.layers: list[nn.Module] = []
        self.final_norm: nn.Module | None = None
        self.backbone: nn.Module | None = None
        self.output_embedding: nn.Module | None = None
        self.vllm_client: Any | None = None
        self.vllm_resolved_model: str | None = None
        self._vllm_endpoint_states = [
            _VllmEndpointState(index=index, url=endpoint)
            for index, endpoint in enumerate(self.vllm_endpoints)
        ]
        self._vllm_clients_initialized = False
        self._client_lock = threading.Lock()
        self._endpoint_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.materialized_samples = 0
        self.target_forward_seconds = 0.0
        self.vllm_request_seconds = 0.0
        self.vllm_requests = 0
        self._warned_replay_algorithm_mismatch = False
        self._warned_replay_layer_mismatch = False
        self._warned_replay_layout_mismatch = False
        logger.info(
            "[target replay rank=%s] initialized backend=%s algorithm=%s "
            "target_layers=%s hidden_layout=%s use_logits=%s endpoints=%s cache=%s",
            self.rank,
            self.backend,
            self.algorithm,
            self.target_layer_ids,
            self.hidden_layout,
            self.use_logits,
            self.vllm_endpoints if self.backend.startswith("vllm_") else None,
            self.cache is not None,
        )

    def materialize(
        self, samples: Iterable[DraftReplaySample | DraftFeatureSample]
    ) -> list[DraftFeatureSample]:
        materialized: list[DraftFeatureSample] = []
        for sample_index, sample in enumerate(samples):
            try:
                if isinstance(sample, DraftFeatureSample):
                    materialized.append(sample)
                    continue
                if not isinstance(sample, DraftReplaySample):
                    raise TypeError(
                        "Target feature replay expected DraftReplaySample, "
                        f"got {type(sample)!r}"
                    )
                self._validate_target_path(sample)
                key = self._cache_key(sample)
                with self._cache_lock:
                    cached = self.cache.get(key) if self.cache is not None else None
                if cached is not None:
                    with self._metrics_lock:
                        self.cache_hits += 1
                    materialized.append(cached)
                    continue
                with self._metrics_lock:
                    self.cache_misses += 1
                replayed = self._materialize_one(sample)
                if self.cache is not None:
                    with self._cache_lock:
                        self.cache.put(key, replayed)
                materialized.append(replayed)
            except Exception:
                metadata = getattr(sample, "metadata", {}) or {}
                logger.exception(
                    "[target replay rank=%s] sample materialization failed "
                    "sample_index=%s algorithm=%s source=%s global_step=%s",
                    self.rank,
                    sample_index,
                    getattr(sample, "algorithm", None),
                    metadata.get("source"),
                    metadata.get("global_step"),
                )
                raise
        with self._metrics_lock:
            self.materialized_samples += len(materialized)
        return materialized

    def _validate_target_path(self, sample: DraftReplaySample) -> None:
        if sample.algorithm.upper() != self.algorithm:
            if not self._warned_replay_algorithm_mismatch:
                logger.warning(
                    "[target replay rank=%s] token replay algorithm differs from "
                    "training algorithm; using the training algorithm for "
                    "materialized features sample=%s training=%s",
                    self.rank,
                    sample.algorithm,
                    self.algorithm,
                )
                self._warned_replay_algorithm_mismatch = True
        collected_layer_ids = sample.metadata.get("target_layer_ids")
        if collected_layer_ids is not None:
            normalized_layer_ids = (
                [int(collected_layer_ids)]
                if isinstance(collected_layer_ids, int)
                else [int(value) for value in collected_layer_ids]
            )
            if normalized_layer_ids != self.target_layer_ids:
                if not self._warned_replay_layer_mismatch:
                    logger.warning(
                        "[target replay rank=%s] token replay target layers differ "
                        "from replay target layers; recomputing hidden states with "
                        "the training configuration collected=%s replay=%s",
                        self.rank,
                        normalized_layer_ids,
                        self.target_layer_ids,
                    )
                    self._warned_replay_layer_mismatch = True
        collected_layout = sample.metadata.get("hidden_states_layout")
        if collected_layout and str(collected_layout) != self.hidden_layout:
            if not self._warned_replay_layout_mismatch:
                logger.warning(
                    "[target replay rank=%s] token replay hidden layout differs "
                    "from replay layout; recomputing hidden states with the "
                    "training configuration collected=%s replay=%s",
                    self.rank,
                    collected_layout,
                    self.hidden_layout,
                )
                self._warned_replay_layout_mismatch = True
        if not self.strict_target_model_path:
            return
        collected_path = sample.metadata.get("target_model_path")
        if collected_path and os.path.normpath(
            os.fspath(collected_path)
        ) != os.path.normpath(self.model_path):
            raise ValueError(
                "Token replay target model path mismatch: "
                f"collected={collected_path!r} replay={self.model_path!r}"
            )

    def _cache_key(self, sample: DraftReplaySample) -> str:
        digest = hashlib.sha256()
        contract = {
            "target_revision": self.target_revision,
            "target_config": self.target_config_fingerprint,
            "algorithm": self.algorithm,
            "target_layer_ids": self.target_layer_ids,
            "hidden_layout": self.hidden_layout,
            "dtype": str(self.dtype),
            "use_logits": self.use_logits,
            "logits_topk": self.logits_topk,
        }
        digest.update(json.dumps(contract, sort_keys=True).encode())
        for tensor in (
            sample.input_ids,
            sample.attention_mask,
            sample.position_ids,
            sample.feature_positions,
            sample.draft_position_ids,
            sample.loss_mask,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM

        logger.warning(
            "Loading frozen target model for standalone token replay: path=%s dtype=%s device=%s",
            self.model_path,
            self.dtype,
            self.device,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self.layers, self.final_norm = _find_layers_and_final_norm(model)
        base_model_prefix = str(getattr(model, "base_model_prefix", "") or "")
        backbone = (
            getattr(model, base_model_prefix, None) if base_model_prefix else None
        )
        self.backbone = backbone if isinstance(backbone, nn.Module) else model
        output_embedding = model.get_output_embeddings()
        self.output_embedding = (
            output_embedding if isinstance(output_embedding, nn.Module) else None
        )
        self.model = model

    def _materialize_one(self, sample: DraftReplaySample) -> DraftFeatureSample:
        if self.backend == "vllm_file":
            return self._materialize_one_vllm_file(sample)
        return self._materialize_one_torch(sample)

    def _materialize_one_torch(self, sample: DraftReplaySample) -> DraftFeatureSample:
        self._ensure_model()
        assert self.backbone is not None
        assert self.final_norm is not None

        feature_positions = sample.feature_positions.detach().cpu().long()
        feature_end = int(feature_positions[-1].item()) + 1
        input_ids = sample.input_ids[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        attention_mask = sample.attention_mask[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        position_ids = sample.position_ids[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        captures: dict[str, torch.Tensor] = {}
        handles = []

        def capture(key: str):
            def hook(_module, _inputs, output):
                captures[key] = _tensor_from_module_output(output)

            return hook

        aux_keys: list[str] = []
        modules: dict[str, nn.Module] = {}
        for layer_id in self.target_layer_ids:
            kind, layer_index = _hidden_capture_target(
                layer_id, self.target_num_hidden_layers
            )
            if kind == "final":
                key = "final"
                module = self.final_norm
            else:
                assert layer_index is not None
                key = f"layer:{layer_index}"
                module = self.layers[layer_index]
            aux_keys.append(key)
            modules[key] = module
        include_final = self.hidden_layout in {
            "eagle3_aux_plus_last",
            "dflash_aux_plus_last",
        }
        need_final = include_final or (self.algorithm == "EAGLE3" and self.use_logits)
        if need_final:
            modules["final"] = self.final_norm
        for key, module in modules.items():
            handles.append(module.register_forward_hook(capture(key)))

        started = time.perf_counter()
        try:
            forward_kwargs = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "position_ids": position_ids.unsqueeze(0),
                "use_cache": False,
                "return_dict": True,
            }
            forward_kwargs = _supported_forward_kwargs(
                self.backbone.forward, forward_kwargs
            )
            with torch.inference_mode():
                self.backbone(**forward_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        self.target_forward_seconds += time.perf_counter() - started

        required_keys = list(aux_keys)
        if need_final:
            required_keys.append("final")
        missing = [key for key in required_keys if key not in captures]
        if missing:
            raise RuntimeError(
                f"Target feature replay missed hidden-state captures: {missing}"
            )

        device_positions = feature_positions.to(self.device)
        hidden_parts = [
            captures[key].squeeze(0).index_select(0, device_positions)
            for key in aux_keys
        ]
        selected_final = (
            captures["final"].squeeze(0).index_select(0, device_positions)
            if need_final
            else None
        )
        if include_final:
            assert selected_final is not None
            hidden_parts.append(selected_final)
        hidden_states = torch.cat(hidden_parts, dim=-1).to(
            device="cpu", dtype=self.dtype
        )

        target_logprobs = None
        if self.algorithm == "EAGLE3" and self.use_logits:
            assert selected_final is not None
            target_logprobs = self._build_sparse_target_logprobs(selected_final[:-1])

        selected_input_ids = sample.input_ids.index_select(0, feature_positions).long()
        selected_loss_mask = sample.loss_mask.index_select(0, feature_positions).float()
        metadata = dict(sample.metadata)
        feature_start = int(feature_positions[0].item())
        feature_end = int(feature_positions[-1].item()) + 1
        metadata.update(
            {
                "source": "token_replay",
                "target_model_path": self.model_path,
                "target_revision": self.target_revision,
                "target_config_fingerprint": self.target_config_fingerprint,
                "target_layer_ids": list(self.target_layer_ids),
                "hidden_states_layout": self.hidden_layout,
                "feature_start": feature_start,
                "feature_end": feature_end,
                "hidden_position_start": feature_start,
                "hidden_position_end": feature_end,
                "hidden_positions": feature_positions,
                "sequence_length": int(selected_input_ids.numel()),
                "full_sequence_length": int(sample.input_ids.numel()),
                "use_logits": self.use_logits,
            }
        )
        if target_logprobs is not None:
            metadata["target_logprobs_position_start"] = feature_start + 1
            metadata["target_logprobs_position_end"] = feature_end

        return DraftFeatureSample(
            algorithm=self.algorithm,
            input_ids=selected_input_ids,
            loss_mask=selected_loss_mask,
            hidden_states=hidden_states,
            target_logprobs=target_logprobs,
            position_ids=sample.draft_position_ids.long(),
            metadata=metadata,
        )

    def _materialize_one_vllm_file(
        self, sample: DraftReplaySample
    ) -> DraftFeatureSample:
        if self.use_logits:
            raise NotImplementedError(
                "target_feature_replay.backend=vllm_file does not yet support "
                "training.use_logits=true; use backend=torch for EAGLE3 logits."
            )
        self._validate_vllm_positions(sample)
        feature_positions = sample.feature_positions.detach().cpu().long()
        feature_end = int(feature_positions[-1].item()) + 1
        prompt_ids = sample.input_ids[:feature_end].detach().cpu().long().tolist()
        hidden_payload = self._request_vllm_hidden_states(prompt_ids)
        try:
            feature = self._feature_from_vllm_payload(
                sample,
                hidden_payload,
                prompt_ids=prompt_ids,
                source="token_replay_vllm_file",
            )
        finally:
            path = hidden_payload.get("_path")
            if self.vllm_on_generate == "delete" and path:
                try:
                    Path(os.fspath(path)).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete vLLM hidden-states file %s", path)
        return feature

    def _validate_vllm_positions(self, sample: DraftReplaySample) -> None:
        if not self.vllm_require_arange_positions:
            return
        feature_positions = sample.feature_positions.detach().cpu().long()
        feature_end = int(feature_positions[-1].item()) + 1
        expected = torch.arange(feature_end, dtype=torch.long)
        actual = sample.position_ids[:feature_end].detach().cpu().long()
        if not torch.equal(actual, expected):
            raise ValueError(
                "vLLM target replay currently requires "
                "position_ids to be contiguous arange positions for the replay prefix"
            )

    def _ensure_vllm_clients(self) -> None:
        if self._vllm_clients_initialized:
            return
        with self._client_lock:
            if self._vllm_clients_initialized:
                return
            started = time.perf_counter()
            logger.info(
                "[target replay rank=%s] initializing vLLM endpoint pool=%s "
                "configured_model=%s",
                self.rank,
                self.vllm_endpoints,
                self.vllm_model,
            )
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM target replay requires the openai package"
                ) from exc
            resolved_models: set[str] = set()
            for state in self._vllm_endpoint_states:
                state.client = openai.OpenAI(
                    base_url=state.url,
                    api_key="EMPTY",
                    max_retries=0,
                )
                state.model = (
                    os.fspath(self.vllm_model) if self.vllm_model else self.model_path
                )
                try:
                    models = state.client.models.list(timeout=self.vllm_timeout)
                    if not self.vllm_model and models.data:
                        state.model = str(models.data[0].id)
                    resolved_models.add(str(state.model))
                    logger.info(
                        "[target replay rank=%s] vLLM endpoint[%s] ready url=%s model=%s",
                        self.rank,
                        state.index,
                        state.url,
                        state.model,
                    )
                except Exception as exc:  # noqa: BLE001
                    state.failures += 1
                    state.consecutive_failures += 1
                    state.cooldown_until = (
                        time.monotonic() + self.vllm_endpoint_cooldown
                    )
                    logger.warning(
                        "[target replay rank=%s] vLLM endpoint[%s] health check "
                        "failed; requests may retry it after cooldown url=%s error=%r",
                        self.rank,
                        state.index,
                        state.url,
                        exc,
                    )
            self._vllm_clients_initialized = True
            self.vllm_client = self._vllm_endpoint_states[0].client
            self.vllm_resolved_model = self._vllm_endpoint_states[0].model
            if len(resolved_models) > 1:
                logger.warning(
                    "[target replay rank=%s] vLLM endpoints advertise different "
                    "models=%s; set target_feature_replay.vllm_model explicitly "
                    "after confirming all servers use identical target weights",
                    self.rank,
                    sorted(resolved_models),
                )
            logger.info(
                "[target replay rank=%s] vLLM endpoint pool initialized "
                "endpoints=%s elapsed=%.3fs",
                self.rank,
                len(self._vllm_endpoint_states),
                time.perf_counter() - started,
            )

    def _acquire_vllm_endpoint(
        self, excluded: set[int] | None = None
    ) -> _VllmEndpointState:
        self._ensure_vllm_clients()
        excluded = excluded or set()
        now = time.monotonic()
        with self._endpoint_lock:
            candidates = [
                state
                for state in self._vllm_endpoint_states
                if state.client is not None
                and state.model is not None
                and state.index not in excluded
                and state.cooldown_until <= now
            ]
            if not candidates:
                candidates = [
                    state
                    for state in self._vllm_endpoint_states
                    if state.client is not None
                    and state.model is not None
                    and state.index not in excluded
                ]
            if not candidates:
                candidates = [
                    state
                    for state in self._vllm_endpoint_states
                    if state.client is not None and state.model is not None
                ]
            if not candidates:
                raise RuntimeError("No configured vLLM endpoint has a usable client")
            state = min(
                candidates,
                key=lambda item: (
                    item.inflight,
                    item.consecutive_failures,
                    item.cooldown_until,
                    item.index,
                ),
            )
            state.inflight += 1
            return state

    def _release_vllm_endpoint(
        self,
        state: _VllmEndpointState,
        *,
        elapsed: float,
        succeeded: bool,
    ) -> None:
        with self._endpoint_lock:
            state.inflight = max(state.inflight - 1, 0)
            state.request_seconds += float(elapsed)
            if succeeded:
                state.requests += 1
                state.consecutive_failures = 0
                state.cooldown_until = 0.0
            else:
                state.failures += 1
                state.consecutive_failures += 1
                state.cooldown_until = time.monotonic() + self.vllm_endpoint_cooldown

    def _request_vllm_response(self, prompt_ids: list[int]) -> Any:
        last_error: Exception | None = None
        started = time.perf_counter()
        attempted_endpoints: set[int] = set()
        for attempt in range(self.vllm_max_retries + 1):
            state = self._acquire_vllm_endpoint(attempted_endpoints)
            if state.client is None:
                raise RuntimeError(f"vLLM endpoint {state.url} has no client")
            attempt_started = time.perf_counter()
            try:
                response = state.client.completions.create(
                    model=state.model,
                    prompt=prompt_ids,
                    max_tokens=1,
                    extra_body={"return_token_ids": True},
                    timeout=self.vllm_timeout,
                )
                choices = getattr(response, "choices", None) or []
                if choices:
                    actual = getattr(choices[0], "prompt_token_ids", None)
                    if actual is not None and list(actual) != prompt_ids:
                        raise ValueError("vLLM prompt_token_ids mismatch")
                with self._metrics_lock:
                    self.vllm_requests += 1
                    self.vllm_request_seconds += time.perf_counter() - started
                self._release_vllm_endpoint(
                    state,
                    elapsed=time.perf_counter() - attempt_started,
                    succeeded=True,
                )
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempted_endpoints.add(state.index)
                self._release_vllm_endpoint(
                    state,
                    elapsed=time.perf_counter() - attempt_started,
                    succeeded=False,
                )
                if attempt >= self.vllm_max_retries:
                    break
                time.sleep(float(2**attempt))
        with self._metrics_lock:
            self.vllm_request_seconds += time.perf_counter() - started
        raise RuntimeError(
            "Failed to request vLLM hidden states after "
            f"{self.vllm_max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _request_vllm_hidden_states(self, prompt_ids: list[int]) -> dict[str, Any]:
        last_error: Exception | None = None
        started = time.perf_counter()
        request_index = self.vllm_requests + 1
        log_request = request_index <= 2 or request_index % 100 == 0
        attempted_endpoints: set[int] = set()
        for attempt in range(self.vllm_max_retries + 1):
            state = self._acquire_vllm_endpoint(attempted_endpoints)
            if state.client is None:
                raise RuntimeError(f"vLLM endpoint {state.url} has no client")
            try:
                attempt_started = time.perf_counter()
                if log_request:
                    logger.info(
                        "[target replay rank=%s] vLLM request starting request=%s "
                        "attempt=%s/%s endpoint=%s prompt_tokens=%s",
                        self.rank,
                        request_index,
                        attempt + 1,
                        self.vllm_max_retries + 1,
                        state.url,
                        len(prompt_ids),
                    )
                response = state.client.completions.create(
                    model=state.model,
                    prompt=prompt_ids,
                    max_tokens=1,
                    extra_body={"return_token_ids": True},
                    timeout=self.vllm_timeout,
                )
                path = self._extract_hidden_states_path(response, prompt_ids)
                payload = self._load_vllm_hidden_states(path)
                payload["_path"] = path
                with self._metrics_lock:
                    self.vllm_requests += 1
                    self.vllm_request_seconds += time.perf_counter() - started
                self._release_vllm_endpoint(
                    state,
                    elapsed=time.perf_counter() - attempt_started,
                    succeeded=True,
                )
                if log_request:
                    hidden_states = payload.get("hidden_states")
                    hidden_shape = (
                        tuple(hidden_states.shape)
                        if hidden_states is not None and torch.is_tensor(hidden_states)
                        else None
                    )
                    logger.info(
                        "[target replay rank=%s] vLLM request completed request=%s "
                        "attempt=%s endpoint=%s path=%s hidden_shape=%s elapsed=%.3fs",
                        self.rank,
                        request_index,
                        attempt + 1,
                        state.url,
                        path,
                        hidden_shape,
                        time.perf_counter() - attempt_started,
                    )
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempted_endpoints.add(state.index)
                self._release_vllm_endpoint(
                    state,
                    elapsed=time.perf_counter() - attempt_started,
                    succeeded=False,
                )
                logger.warning(
                    "[target replay rank=%s] vLLM request failed request=%s "
                    "attempt=%s/%s endpoint=%s prompt_tokens=%s elapsed=%.3fs error=%r",
                    self.rank,
                    request_index,
                    attempt + 1,
                    self.vllm_max_retries + 1,
                    state.url,
                    len(prompt_ids),
                    time.perf_counter() - started,
                    exc,
                )
                if attempt >= self.vllm_max_retries:
                    break
                time.sleep(float(2**attempt))
        with self._metrics_lock:
            self.vllm_request_seconds += time.perf_counter() - started
        raise RuntimeError(
            f"Failed to request vLLM hidden states after "
            f"{self.vllm_max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _extract_hidden_states_path(self, response: Any, prompt_ids: list[int]) -> str:
        choices = getattr(response, "choices", None) or []
        if choices:
            prompt_token_ids = getattr(choices[0], "prompt_token_ids", None)
            if prompt_token_ids is not None and list(prompt_token_ids) != prompt_ids:
                raise ValueError(
                    "vLLM prompt_token_ids mismatch while extracting hidden states"
                )
        kv_transfer_params = getattr(response, "kv_transfer_params", None)
        if kv_transfer_params is None:
            raise ValueError("vLLM response missing kv_transfer_params")
        path = kv_transfer_params.get("hidden_states_path")
        if not path:
            raise ValueError("vLLM response missing hidden_states_path")
        return os.fspath(path)

    def _load_vllm_hidden_states(self, path: str) -> dict[str, Any]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("vLLM hidden-state replay requires safetensors") from exc
        file_path = Path(path)
        lock_path = Path(f"{path}.lock")
        if lock_path.exists():
            _wait_for_lock(lock_path)
        if not file_path.exists():
            raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
        return dict(load_file(str(file_path), device="cpu"))

    def _feature_from_vllm_payload(
        self,
        sample: DraftReplaySample,
        payload: dict[str, Any],
        *,
        prompt_ids: list[int],
        source: str,
    ) -> DraftFeatureSample:
        expected_prompt_ids = (
            sample.input_ids[: int(sample.feature_positions[-1].item()) + 1]
            .detach()
            .cpu()
            .long()
            .tolist()
        )
        if expected_prompt_ids != prompt_ids:
            raise ValueError("prompt_ids do not match replay feature window")
        return feature_from_vllm_payload(
            payload,
            sample,
            FeatureContract(
                algorithm=getattr(self, "algorithm", sample.algorithm),
                target_layer_ids=list(self.target_layer_ids),
                hidden_states_layout=self.hidden_layout,
                dtype=self.dtype,
                target_model_id=self.model_path,
                target_model_revision=self.target_revision,
                tokenizer_fingerprint=str(
                    getattr(self, "tokenizer_fingerprint", "replay-unspecified")
                ),
                use_logits=self.use_logits,
                target_config_fingerprint=self.target_config_fingerprint,
                source=source,
            ),
        )

    def _build_sparse_target_logprobs(
        self, final_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.output_embedding is None:
            raise RuntimeError(
                "EAGLE3 token replay with use_logits=true requires target output embeddings"
            )
        rows: list[torch.Tensor] = []
        topk = max(self.logits_topk, 1)
        with torch.inference_mode():
            for start in range(
                0, int(final_hidden_states.size(0)), self.logits_chunk_rows
            ):
                hidden = final_hidden_states[start : start + self.logits_chunk_rows]
                logits = self.output_embedding(hidden).float()
                local_topk = min(topk, int(logits.size(-1)))
                values, ids = logits.topk(local_topk, dim=-1)
                values = values - torch.logsumexp(logits, dim=-1, keepdim=True)
                rows.append(
                    torch.stack((values, ids.to(dtype=values.dtype)), dim=-1).cpu()
                )
        if not rows:
            return torch.empty(0, topk, 2, dtype=torch.float32)
        return torch.cat(rows, dim=0).contiguous()

    def metrics(self) -> dict[str, float]:
        metrics = {
            "replay/cache_hits_total": float(self.cache_hits),
            "replay/cache_misses_total": float(self.cache_misses),
            "replay/materialized_samples_total": float(self.materialized_samples),
            "replay/target_forward_time_total": float(self.target_forward_seconds),
        }
        if self.backend.startswith("vllm_"):
            metrics["replay/vllm_requests_total"] = float(self.vllm_requests)
            metrics["replay/vllm_request_time_total"] = float(self.vllm_request_seconds)
            with self._endpoint_lock:
                metrics["replay/vllm_endpoints_total"] = float(
                    len(self._vllm_endpoint_states)
                )
                for state in self._vllm_endpoint_states:
                    prefix = f"replay/vllm_endpoint_{state.index}"
                    metrics[f"{prefix}_inflight"] = float(state.inflight)
                    metrics[f"{prefix}_requests_total"] = float(state.requests)
                    metrics[f"{prefix}_failures_total"] = float(state.failures)
                    metrics[f"{prefix}_request_time_total"] = float(
                        state.request_seconds
                    )
        total = self.cache_hits + self.cache_misses
        if total > 0:
            metrics["replay/cache_hit_ratio"] = self.cache_hits / float(total)
        if self.cache is not None:
            metrics.update(self.cache.metrics())
        return metrics

    def close(self) -> None:
        for state in self._vllm_endpoint_states:
            client = state.client
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to close vLLM endpoint client %s",
                        state.url,
                        exc_info=True,
                    )
            state.client = None
        if self.model is None:
            return
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        self.model = None
        self.layers = []
        self.final_norm = None
        self.backbone = None
        self.output_embedding = None


def _supported_forward_kwargs(forward: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return kwargs
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
