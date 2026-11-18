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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

import os

# Adapted from training_common/utils/memory_chunking.py.
from dataclasses import dataclass

_FALSE_VALUES = {"0", "false", "off", "no"}


@dataclass(frozen=True)
class MemoryChunkingConfig:
    enabled: bool
    vocab_parallel_loss_token_chunk_size: int
    dsa_backward_query_chunk_size: int
    hc_token_chunk_size: int
    grouped_routed_moe_expert_streaming: bool
    grouped_routed_moe_route_budget: str
    grouped_routed_moe_max_experts_per_group: int


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _nonnegative_int(value, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _route_budget(value) -> str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return normalized
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError(
            "grouped_routed_moe_route_budget must be 'auto' or a positive "
            f"integer, got {value!r}"
        )
    return str(parsed)


def resolve_memory_chunking(cfg) -> MemoryChunkingConfig:
    optimizations = resolve_optimization_config(cfg)
    section = getattr(optimizations, "memory_chunking", None)
    if section is not None:
        enabled_default = bool(getattr(section, "enabled", False))
    else:
        enabled_default = any(
            (
                int(getattr(optimizations, "dsa_backward_query_chunk_size", 0)) > 0,
                int(getattr(optimizations, "hc_token_chunk_size", 0)) > 0,
                bool(
                    getattr(
                        optimizations,
                        "grouped_routed_moe_expert_streaming",
                        False,
                    )
                ),
            )
        )
    enabled = _env_flag("MTP_MEMORY_CHUNKING", enabled_default)

    def configured(name: str, legacy_name: str, default):
        if section is not None and hasattr(section, name):
            return getattr(section, name)
        return getattr(optimizations, legacy_name, default)

    vocab_chunk = _nonnegative_int(
        os.environ.get(
            "MTP_VOCAB_PARALLEL_LOSS_TOKEN_CHUNK_SIZE",
            configured(
                "vocab_parallel_loss_token_chunk_size",
                "vocab_parallel_loss_token_chunk_size",
                0,
            ),
        ),
        "vocab_parallel_loss_token_chunk_size",
    )
    dsa_chunk = _nonnegative_int(
        os.environ.get(
            "MTP_DS4_DSA_BACKWARD_QUERY_CHUNK_SIZE",
            configured(
                "dsa_backward_query_chunk_size",
                "dsa_backward_query_chunk_size",
                0,
            ),
        ),
        "dsa_backward_query_chunk_size",
    )
    hc_chunk = _nonnegative_int(
        os.environ.get(
            "MTP_DS4_HC_TOKEN_CHUNK_SIZE",
            os.environ.get(
                "MTP_DS4_HC_POST_TOKEN_CHUNK_SIZE",
                configured(
                    "hc_token_chunk_size",
                    "hc_token_chunk_size",
                    getattr(optimizations, "hc_post_token_chunk_size", 0),
                ),
            ),
        ),
        "hc_token_chunk_size",
    )
    if section is not None and hasattr(section, "grouped_routed_moe_expert_streaming"):
        expert_default = bool(section.grouped_routed_moe_expert_streaming)
    elif hasattr(optimizations, "grouped_routed_moe_expert_streaming"):
        expert_default = bool(optimizations.grouped_routed_moe_expert_streaming)
    elif hasattr(optimizations, "grouped_routed_moe_full_backward"):
        expert_default = not bool(optimizations.grouped_routed_moe_full_backward)
    else:
        expert_default = False
    if "MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING" in os.environ:
        expert_streaming = _env_flag(
            "MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING",
            expert_default,
        )
    elif "MTP_DS4_GROUPED_ROUTED_MOE_FULL_BACKWARD" in os.environ:
        expert_streaming = not _env_flag(
            "MTP_DS4_GROUPED_ROUTED_MOE_FULL_BACKWARD",
            not expert_default,
        )
    else:
        expert_streaming = expert_default
    route_budget = _route_budget(
        os.environ.get(
            "MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET",
            configured(
                "grouped_routed_moe_route_budget",
                "grouped_routed_moe_route_budget",
                "auto",
            ),
        )
    )
    max_experts_per_group = _nonnegative_int(
        os.environ.get(
            "MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP",
            configured(
                "grouped_routed_moe_max_experts_per_group",
                "grouped_routed_moe_max_experts_per_group",
                1,
            ),
        ),
        "grouped_routed_moe_max_experts_per_group",
    )
    if max_experts_per_group == 0:
        raise ValueError("grouped_routed_moe_max_experts_per_group must be positive")

    if not enabled:
        vocab_chunk = 0
        dsa_chunk = 0
        hc_chunk = 0
        expert_streaming = False
        max_experts_per_group = 1

    return MemoryChunkingConfig(
        enabled=enabled,
        vocab_parallel_loss_token_chunk_size=vocab_chunk,
        dsa_backward_query_chunk_size=dsa_chunk,
        hc_token_chunk_size=hc_chunk,
        grouped_routed_moe_expert_streaming=expert_streaming,
        grouped_routed_moe_route_budget=route_budget,
        grouped_routed_moe_max_experts_per_group=max_experts_per_group,
    )


def export_memory_chunking_environment(config: MemoryChunkingConfig) -> None:
    os.environ["MTP_MEMORY_CHUNKING"] = "1" if config.enabled else "0"
    os.environ["MTP_VOCAB_PARALLEL_LOSS_TOKEN_CHUNK_SIZE"] = str(
        config.vocab_parallel_loss_token_chunk_size
    )
    os.environ["MTP_DS4_DSA_BACKWARD_QUERY_CHUNK_SIZE"] = str(
        config.dsa_backward_query_chunk_size
    )
    os.environ["MTP_DS4_HC_TOKEN_CHUNK_SIZE"] = str(config.hc_token_chunk_size)
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING"] = (
        "1" if config.grouped_routed_moe_expert_streaming else "0"
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_FULL_BACKWARD"] = (
        "0" if config.grouped_routed_moe_expert_streaming else "1"
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET"] = (
        config.grouped_routed_moe_route_budget
    )
    os.environ["MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP"] = str(
        config.grouped_routed_moe_max_experts_per_group
    )


# Adapted from training_common/utils/optimization_config.py.

from collections.abc import Mapping, MutableMapping

_MISSING = object()
_SCOPES = {"common", "mtp", "dspark"}
_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_DS4_MODEL_CONSTRUCTION_BOOL_ENV = {
    "cache_dsa_metadata_structures": ("MTP_DS4_CACHE_DSA_METADATA_STRUCTURES"),
    "reuse_dsa_backward_host_metadata": ("MTP_DS4_REUSE_DSA_BACKWARD_HOST_METADATA"),
    "fused_qnorm_unquantized_backward": ("MTP_DS4_FUSED_QNORM_UNQUANTIZED_BACKWARD"),
}


def _get_value(section, name, default=_MISSING):
    if section is None:
        return default
    if isinstance(section, Mapping):
        return section[name] if name in section else default  # noqa: SIM401 -- retain mandatory DictConfig value errors
    return getattr(section, name, default)


def infer_optimization_scope(cfg) -> str:
    model = getattr(cfg, "model", None)
    model_type = _get_value(model, "type", "")
    return "dspark" if model_type == "ds4_dspark" else "mtp"


class OptimizationConfigView:
    """Attribute view over scoped, common, and legacy optimization settings."""

    def __init__(self, *layers):
        self._layers = tuple(layer for layer in layers if layer is not None)

    def __getattr__(self, name):
        for layer in self._layers:
            value = _get_value(layer, name)
            if value is not _MISSING:
                return value
        raise AttributeError(name)

    def __dir__(self):
        names = set(super().__dir__())
        for layer in self._layers:
            if isinstance(layer, Mapping):
                names.update(layer)
            else:
                names.update(vars(layer))
        return sorted(names - _SCOPES)


def resolve_optimization_config(cfg, scope: str | None = None):
    """Resolve optimization settings for one training algorithm.

    New configurations use ``optimizations.common``, ``optimizations.mtp``,
    and ``optimizations.dspark``. Flat keys remain the lowest-priority layer so
    existing external configurations continue to work.
    """

    if scope is None:
        scope = infer_optimization_scope(cfg)
    if scope not in {"mtp", "dspark"}:
        raise ValueError(f"Unsupported optimization scope: {scope!r}")

    optimizations = getattr(cfg, "optimizations", None)
    common = _get_value(optimizations, "common", None)
    scoped = _get_value(optimizations, scope, None)
    return OptimizationConfigView(scoped, common, optimizations)


def export_ds4_model_construction_environment(
    cfg,
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, bool]:
    """Export DS4 flags that the runtime caches during model construction.

    Explicit environment overrides retain priority over Hydra configuration.
    The returned values are canonicalized and can also be copied into the
    Transformers config so the constructor has one consistent resolved view.
    """

    if environment is None:
        environment = os.environ
    optimizations = resolve_optimization_config(cfg)
    resolved: dict[str, bool] = {}
    for config_name, environment_name in _DS4_MODEL_CONSTRUCTION_BOOL_ENV.items():
        configured = bool(getattr(optimizations, config_name, False))
        raw_value = environment.get(environment_name)
        enabled = (
            configured
            if raw_value is None
            else str(raw_value).strip().lower() not in _FALSE_ENV_VALUES
        )
        environment[environment_name] = "1" if enabled else "0"
        resolved[config_name] = enabled
    return resolved


def get_optimization(
    cfg,
    name: str,
    default=None,
    *,
    scope: str | None = None,
    legacy_train_name: str | None = None,
):
    optimizations = resolve_optimization_config(cfg, scope)
    value = getattr(optimizations, name, _MISSING)
    if value is not _MISSING:
        return value
    if legacy_train_name is None:
        return default
    train = getattr(cfg, "train", None)
    return _get_value(train, legacy_train_name, default)


def get_dspark_optimization(cfg, name: str, default=None):
    """Read a DSpark optimization with the former train key as fallback."""

    return get_optimization(
        cfg,
        name,
        default,
        scope="dspark",
        legacy_train_name=f"dspark_{name}",
    )
