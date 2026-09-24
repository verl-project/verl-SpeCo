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
"""Single-entry launcher for Producer -> TransferQueue -> draft training.

The example script keeps the ordinary standalone-training interface. This
module owns the internal Ray/TQ identity and the owner, Producer and Consumer
process lifecycle so transport-specific overrides do not leak into examples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid

from omegaconf import OmegaConf

from verl_speco.draft_train_launcher import (
    normalize_training_args,
    resolve_launch_config,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.standalone_layer_ids import normalize_standalone_layer_ids
from verl_speco.trainer.standalone_resume import load_standalone_resume


logger = logging.getLogger(__name__)

_MODEL_PATH_KEY = "actor_rollout_ref.model.path"
_DRAFTER_PATH_KEY = "actor_rollout_ref.rollout.drafter.model_path"
_ALGORITHM_KEY = "actor_rollout_ref.rollout.drafter.speculative_algorithm"
_TRAIN_FILES_KEY = "data.train_files"
_TOKENIZER_PATH_KEY = (
    "actor_rollout_ref.rollout.drafter.training.feature_store.tokenizer_path"
)
_PRODUCER_TARGET_LAYER_IDS_KEY = "speco.standalone_tq_producer.target_layer_ids"
_PRODUCER_VLLM_AUX_LAYER_IDS_KEY = (
    "speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids"
)
_PRODUCER_HIDDEN_DTYPE_KEY = "speco.standalone_tq_producer.hidden_dtype"
_DSPARK_L1_LOSS_ALPHA_KEY = (
    "actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha"
)
_ALGORITHM_TARGET_LAYER_IDS_KEYS = {
    "EAGLE3": "actor_rollout_ref.rollout.drafter.training.eagle3_target_layer_ids",
    "DFLASH": "actor_rollout_ref.rollout.drafter.training.dflash_target_layer_ids",
    "DSPARK": "actor_rollout_ref.rollout.drafter.training.dspark_target_layer_ids",
    "DOMINO": "actor_rollout_ref.rollout.drafter.training.domino_target_layer_ids",
}
_MAX_STEPS_KEY = "actor_rollout_ref.rollout.drafter.training.max_steps"
_BATCH_SIZE_PER_GPU_KEY = (
    "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu"
)
_NPROC_KEYS = (
    "speco.draft_training.nproc_per_node",
    "speco.draft_training.num_gpus_per_node",
    "actor_rollout_ref.rollout.drafter.training.nproc_per_node",
    "actor_rollout_ref.rollout.drafter.training.num_gpus_per_node",
)
_NNODES_KEYS = (
    "speco.draft_training.nnodes",
    "speco.draft_training.num_nodes",
    "actor_rollout_ref.rollout.drafter.training.nnodes",
    "actor_rollout_ref.rollout.drafter.training.num_nodes",
)

_TQ_PREFIX = "actor_rollout_ref.rollout.drafter.training.transfer_queue"
_FEATURE_STORE_PREFIX = "actor_rollout_ref.rollout.drafter.training.feature_store"
_PRODUCER_PREFIX = "speco.standalone_tq_producer"
_RUNTIME_BACKEND_KEY = "speco.draft_training.runtime_backend"
# Every user override under this prefix is forwarded to the Producer, except the
# fields the unified launcher computes and sets itself (paths, identities,
# endpoints, and the sample budget). Forwarding the whole prefix keeps newly
# added Producer knobs usable through the launcher without extending an
# allow-list for each one.
_PRODUCER_LAUNCHER_OWNED_KEYS = frozenset(
    {
        f"{_PRODUCER_PREFIX}.input_path",
        f"{_PRODUCER_PREFIX}.resume_checkpoint_path",
        f"{_PRODUCER_PREFIX}.tokenizer_path",
        f"{_PRODUCER_PREFIX}.tokenizer_fingerprint",
        f"{_PRODUCER_PREFIX}.target_model_id",
        f"{_PRODUCER_PREFIX}.target_model_revision",
        f"{_PRODUCER_PREFIX}.target_layer_ids",
        f"{_PRODUCER_PREFIX}.vllm_aux_hidden_state_layer_ids",
        f"{_PRODUCER_PREFIX}.vllm_endpoints",
        f"{_PRODUCER_PREFIX}.vllm_model",
        f"{_PRODUCER_PREFIX}.max_samples",
    }
)
_PRODUCER_CONFIG_PATH = Path(__file__).with_name("config") / "speco_base.yaml"
_INTERNAL_OVERRIDE_KEYS = frozenset(
    {
        f"{_FEATURE_STORE_PREFIX}.type",
        f"{_FEATURE_STORE_PREFIX}.path",
        f"{_FEATURE_STORE_PREFIX}.shuffle",
        f"{_FEATURE_STORE_PREFIX}.repeat",
        f"{_TQ_PREFIX}.enable",
        f"{_TQ_PREFIX}.ray.address",
        f"{_TQ_PREFIX}.ray.namespace",
        f"{_TQ_PREFIX}.partition_id",
        f"{_TQ_PREFIX}.run_id",
        f"{_TQ_PREFIX}.drop_last",
        f"{_TQ_PREFIX}.backend.storage_backend",
        f"{_TQ_PREFIX}.backend.SimpleStorage.total_storage_size",
        f"{_TQ_PREFIX}.backend.SimpleStorage.num_data_storage_units",
        f"{_TQ_PREFIX}.expected_feature.algorithm",
        f"{_TQ_PREFIX}.expected_feature.target_model_path",
        f"{_TQ_PREFIX}.expected_feature.target_revision",
        f"{_TQ_PREFIX}.expected_feature.tokenizer_fingerprint",
        f"{_TQ_PREFIX}.expected_feature.target_layer_ids",
        f"{_TQ_PREFIX}.expected_feature.hidden_states_layout",
        f"{_TQ_PREFIX}.expected_feature.hidden_dtype",
        _RUNTIME_BACKEND_KEY,
    }
)

# These are the output IDs passed verbatim to vLLM serve. DFlash-family
# training IDs are derived from them by subtracting one.
_DEFAULT_VLLM_AUX_LAYER_IDS = (1, 9, 17, 25, 33)
_DEFAULT_VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
_DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = "0.4"
_VLLM_HIDDEN_STATES_DIR = "__SPECO_HIDDEN_STATES_DIR__"
_TQ_NAMESPACE = "speco-drafter"
_TQ_PARTITION = "speco_drafter_features"
_TQ_STORAGE_BACKENDS = ("SimpleStorage", "MooncakeStore")
_MOONCAKE_MASTER_DEFAULT = "127.0.0.1:50051"
_MOONCAKE_AUTO_INIT_DEFAULT = False


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    model_path: str
    tokenizer_path: str
    algorithm: str
    target_layer_ids: tuple[int, ...]
    vllm_aux_hidden_state_layer_ids: tuple[int, ...]
    vllm_endpoints: tuple[str, ...]
    run_id: str


@dataclass(frozen=True)
class PipelineCommands:
    vllm: list[str] | None
    vllm_endpoints: tuple[str, ...]
    owner: list[str]
    producer: list[str]
    consumer: list[str]
    producer_overrides: tuple[str, ...]
    consumer_overrides: tuple[str, ...]


@dataclass(frozen=True)
class RaySession:
    module: Any
    address: str

    def close(self) -> None:
        self.module.shutdown()


def _split_override(item: str) -> tuple[str, str] | None:
    if "=" not in item or item.startswith("-"):
        return None
    key, value = item.split("=", 1)
    return key, value


def _is_forwarded_producer_override(key: str) -> bool:
    """Whether a key selects an option under the Producer config prefix."""

    return key.startswith(f"{_PRODUCER_PREFIX}.")


@lru_cache(maxsize=1)
def _producer_config_keys() -> frozenset[str]:
    """Every override path the Producer accepts, taken from its config schema.

    Includes intermediate nodes so a nested override such as
    ``render_boundary`` or a leaf such as ``render_boundary.enabled`` both
    validate.
    """

    node: Any = OmegaConf.load(_PRODUCER_CONFIG_PATH)
    for part in _PRODUCER_PREFIX.split("."):
        node = node[part]
    keys: set[str] = set()

    def walk(path: str, value: Any) -> None:
        keys.add(path)
        items = getattr(value, "items", None)
        if callable(items):
            for key, child in items():
                walk(f"{path}.{key}", child)

    walk(_PRODUCER_PREFIX, node)
    return frozenset(keys)


def _find_override(overrides: Sequence[str], key: str) -> str | None:
    for item in reversed(overrides):
        parsed = _split_override(item)
        if parsed is not None and parsed[0] == key:
            return parsed[1]
    return None


def _find_first_override(overrides: Sequence[str], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _find_override(overrides, key)
        if value is not None:
            return value
    return None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _single_train_file(value: str | None) -> str:
    if value is None:
        raise ValueError(f"Standalone TQ training requires {_TRAIN_FILES_KEY}")
    text = _strip_quotes(value)
    if text.startswith("[") and text.endswith("]"):
        items = [_strip_quotes(item) for item in text[1:-1].split(",") if item.strip()]
        if len(items) != 1:
            raise ValueError("Standalone TQ Producer requires exactly one train file")
        text = items[0]
    if not text:
        raise ValueError("Standalone TQ Producer train file must not be empty")
    return text


def _parse_layer_ids(
    value: str | None, *, config_key: str, default: tuple[int, ...] | None = None
) -> tuple[int, ...] | None:
    if value is None or _strip_quotes(value).lower() in {"", "null", "none"}:
        return default
    text = _strip_quotes(value)
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"{config_key} must be a Hydra integer list")
    try:
        result = tuple(int(item.strip()) for item in text[1:-1].split(","))
    except ValueError as exc:
        raise ValueError(f"{config_key} must contain only integers") from exc
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{config_key} must contain non-negative IDs")
    return result


def _resolve_layer_ids(
    training_args: Sequence[str], algorithm: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve vLLM output IDs and canonical drafter decoder-layer IDs."""

    raw_vllm = _find_override(training_args, _PRODUCER_VLLM_AUX_LAYER_IDS_KEY)
    raw_producer_target = _find_override(training_args, _PRODUCER_TARGET_LAYER_IDS_KEY)
    algorithm_key = _ALGORITHM_TARGET_LAYER_IDS_KEYS.get(algorithm)
    raw_algorithm_target = (
        _find_override(training_args, algorithm_key)
        if algorithm_key is not None
        else None
    )
    vllm_ids = _parse_layer_ids(
        raw_vllm,
        config_key=_PRODUCER_VLLM_AUX_LAYER_IDS_KEY,
        default=None,
    )
    producer_target_ids = _parse_layer_ids(
        raw_producer_target,
        config_key=_PRODUCER_TARGET_LAYER_IDS_KEY,
        default=None,
    )
    algorithm_target_ids = _parse_layer_ids(
        raw_algorithm_target,
        config_key=algorithm_key or _PRODUCER_TARGET_LAYER_IDS_KEY,
        default=None,
    )
    if (
        producer_target_ids is not None
        and algorithm_target_ids is not None
        and producer_target_ids != algorithm_target_ids
    ):
        raise ValueError(
            f"{_PRODUCER_TARGET_LAYER_IDS_KEY}={list(producer_target_ids)} does "
            f"not match {algorithm_key}={list(algorithm_target_ids)}"
        )
    target_ids = (
        producer_target_ids if producer_target_ids is not None else algorithm_target_ids
    )
    normalized_target_ids, normalized_vllm_ids = normalize_standalone_layer_ids(
        algorithm,
        target_ids,
        vllm_ids,
        default_vllm_ids=_DEFAULT_VLLM_AUX_LAYER_IDS,
    )
    return normalized_vllm_ids, normalized_target_ids


def _parse_vllm_endpoints(env: Mapping[str, str]) -> tuple[str, ...]:
    """Read a Hydra-style endpoint list while preserving the singular fallback."""

    configured = str(env.get("SPECO_VLLM_ENDPOINTS", "")).strip()
    if configured:
        text = _strip_quotes(configured)
        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError(
                "SPECO_VLLM_ENDPOINTS must be a Hydra-style list, for example "
                "[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]"
            )
        endpoints = tuple(
            _strip_quotes(item).rstrip("/")
            for item in text[1:-1].split(",")
            if item.strip()
        )
    else:
        endpoint = str(env.get("SPECO_VLLM_ENDPOINT", _DEFAULT_VLLM_ENDPOINT)).strip()
        endpoints = (endpoint.rstrip("/"),) if endpoint else ()
    if not endpoints:
        raise ValueError("At least one hidden-state vLLM endpoint is required")
    for endpoint in endpoints:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid vLLM endpoint: {endpoint!r}")
    return endpoints


def _required_override(overrides: Sequence[str], key: str) -> str:
    value = _find_override(overrides, key)
    normalized = _strip_quotes(value or "")
    if not normalized or normalized.startswith("/path/to/"):
        raise ValueError(f"Standalone TQ training requires a real {key}")
    return normalized


def _positive_int_override(
    overrides: Sequence[str], keys: Sequence[str], *, default: int
) -> int:
    raw = _find_first_override(overrides, keys)
    value = int(_strip_quotes(raw)) if raw is not None else int(default)
    if value <= 0:
        raise ValueError(f"{keys[0]} must be positive, got {value}")
    return value


def _producer_max_samples(
    training_args: Sequence[str], *, resumed_optimizer_step: int = 0
) -> int:
    """Return samples needed for exactly max_steps complete global batches."""

    raw_max_steps = _find_override(training_args, _MAX_STEPS_KEY)
    max_steps = int(_strip_quotes(raw_max_steps)) if raw_max_steps is not None else 1000
    if max_steps <= 0:
        # An unbounded training run cannot have a finite Producer target. Keep
        # the direct Producer's one-pass behavior instead of looping forever.
        return 0
    batch_size = _positive_int_override(
        training_args, (_BATCH_SIZE_PER_GPU_KEY,), default=4
    )
    nproc = _positive_int_override(training_args, _NPROC_KEYS, default=1)
    nnodes = _positive_int_override(training_args, _NNODES_KEYS, default=1)
    remaining_steps = max(max_steps - int(resumed_optimizer_step), 0)
    return remaining_steps * batch_size * nproc * nnodes


def _stable_path_identity(kind: str, path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return f"{kind}-path-sha256-{digest}"


def _target_final_layer_id(model_path: str, target_layer_ids: Sequence[int]) -> int:
    """Resolve the final transformer-layer output ID from a local HF config."""

    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read target model config: {config_path}") from exc
        for candidate in (
            model_config.get("num_hidden_layers"),
            (model_config.get("text_config") or {}).get("num_hidden_layers"),
        ):
            if candidate is not None and int(candidate) > 0:
                return int(candidate)
        raise ValueError(f"Target model config has no num_hidden_layers: {config_path}")
    # Keep dry-run and model-registry IDs usable. The configured Qwen3-4B/8B
    # vLLM outputs end at 33 and use transformer output 36 as the final state.
    return max(int(layer_id) for layer_id in target_layer_ids) + 3


def _vllm_capture_layer_ids(
    target_layer_ids: Sequence[int], final_layer_id: int
) -> list[int]:
    """Build the vLLM capture plan without requesting the final output twice."""

    capture_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    if int(final_layer_id) not in capture_layer_ids:
        capture_layer_ids.append(int(final_layer_id))
    return capture_layer_ids


def resolve_pipeline_config(
    training_args: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> PipelineConfig:
    """Derive all Producer/TQ settings from ordinary training arguments."""

    env = os.environ if environ is None else environ
    model_path = _required_override(training_args, _MODEL_PATH_KEY)
    input_path = _single_train_file(_find_override(training_args, _TRAIN_FILES_KEY))
    tokenizer_path = _strip_quotes(
        _find_override(training_args, _TOKENIZER_PATH_KEY) or model_path
    )
    algorithm = _strip_quotes(
        _find_override(training_args, _ALGORITHM_KEY) or "DSPARK"
    ).upper()
    if not algorithm:
        raise ValueError(f"{_ALGORITHM_KEY} must not be empty")
    vllm_aux_layer_ids, target_layer_ids = _resolve_layer_ids(training_args, algorithm)
    endpoints = _parse_vllm_endpoints(env)
    return PipelineConfig(
        input_path=input_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        algorithm=algorithm,
        target_layer_ids=target_layer_ids,
        vllm_aux_hidden_state_layer_ids=vllm_aux_layer_ids,
        vllm_endpoints=endpoints,
        run_id=f"{algorithm.lower()}-{uuid.uuid4().hex}",
    )


def start_ray_session(
    *,
    environ: Mapping[str, str] | None = None,
    ray_module: Any | None = None,
) -> RaySession:
    """Create a task-local Ray control plane for the hidden TQ pipeline."""

    env = os.environ if environ is None else environ
    ray_runtime = ray_module
    if ray_runtime is None:
        try:
            ray_runtime = importlib.import_module("ray")
        except ImportError as exc:
            raise RuntimeError(
                "Standalone TQ training requires Ray and TransferQueue==0.1.10"
            ) from exc
    # ``ray.init()`` consults RAY_ADDRESS when no explicit address is supplied.
    # This launcher owns the complete Producer/TQ/Consumer lifetime, so an
    # inherited address (often left by another job) must never select its
    # control plane.  ``local`` explicitly starts this task's Ray runtime.
    init_kwargs: dict[str, Any] = {
        "address": "local",
        "namespace": _TQ_NAMESPACE,
        "include_dashboard": False,
    }
    num_cpus = str(env.get("SPECO_RAY_NUM_CPUS", "")).strip()
    if num_cpus:
        init_kwargs["num_cpus"] = int(num_cpus)
    ray_runtime.init(**init_kwargs)
    address = str(ray_runtime.get_runtime_context().gcs_address).strip()
    if not address:
        ray_runtime.shutdown()
        raise RuntimeError("Ray did not report a GCS address for TQ clients")
    return RaySession(module=ray_runtime, address=address)


def _validate_runtime_backend_topology(
    runtime_backend: str, training_args: Sequence[str]
) -> None:
    """Reject multi-node Ray jobs until external-cluster ownership is supported."""

    if runtime_backend != "ray":
        return
    nnodes = _positive_int_override(training_args, _NNODES_KEYS, default=1)
    if nnodes != 1:
        raise ValueError(
            "Standalone Ray backend currently requires "
            "speco.draft_training.nnodes=1 because the launcher starts a "
            "task-local single-node Ray runtime"
        )


def _hydra_list(values: Sequence[Any]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _replace_internal_overrides(
    training_args: Sequence[str], internal: Sequence[str]
) -> list[str]:
    cleaned: list[str] = []
    for item in training_args:
        parsed = _split_override(item)
        if parsed is not None and parsed[0] in _INTERNAL_OVERRIDE_KEYS:
            continue
        cleaned.append(item)
    return [*cleaned, *internal]


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_tq_backend(env: Mapping[str, str]) -> str:
    """Return the selected TQ storage backend, rejecting unknown values.

    A typo or case mismatch must not silently fall back to the in-memory
    ``SimpleStorage`` backend when a remote store was intended.
    """
    backend = str(env.get("SPECO_TQ_STORAGE_BACKEND", "SimpleStorage")).strip()
    if backend not in _TQ_STORAGE_BACKENDS:
        raise RuntimeError(
            f"Unsupported SPECO_TQ_STORAGE_BACKEND={backend!r}; expected one of "
            f"{list(_TQ_STORAGE_BACKENDS)}."
        )
    return backend


def _mooncake_master_address(env: Mapping[str, str]) -> tuple[str, int]:
    raw = str(env.get("SPECO_TQ_MOONCAKE_MASTER", _MOONCAKE_MASTER_DEFAULT)).strip()
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        raise RuntimeError(f"SPECO_TQ_MOONCAKE_MASTER must be host:port, got {raw!r}")
    return host, int(port)


def validate_tq_backend(
    env: Mapping[str, str],
    *,
    connect: Callable[..., Any] | None = None,
    timeout: float = 2.0,
) -> None:
    """Fail fast when the selected TQ transport cannot possibly work.

    ``MooncakeStore`` with ``auto_init=false`` needs an external
    ``mooncake_master``; TransferQueue only starts one itself when auto-init is
    on. Probe the master address up front so the pipeline raises an actionable
    error instead of failing deep inside the owner.
    """
    backend = _resolve_tq_backend(env)
    if backend != "MooncakeStore":
        return
    if _env_flag(env, "SPECO_TQ_MOONCAKE_AUTO_INIT", _MOONCAKE_AUTO_INIT_DEFAULT):
        return
    if _env_flag(env, "SPECO_TQ_MOONCAKE_SKIP_PRECHECK", False):
        # Multi-node setups may not be able to reach the master from the
        # launcher host even though the training workers can.
        return
    host, port = _mooncake_master_address(env)
    probe = connect if connect is not None else socket.create_connection
    try:
        probe((host, port), timeout=timeout).close()
    except OSError as exc:
        raise RuntimeError(
            "MooncakeStore backend selected with SPECO_TQ_MOONCAKE_AUTO_INIT=false, "
            f"but no mooncake_master is reachable at {host}:{port} ({exc}). Start "
            "mooncake_master there, or set SPECO_TQ_MOONCAKE_AUTO_INIT=true to let "
            "TransferQueue start one."
        ) from exc


def _tq_backend_overrides(env: Mapping[str, str]) -> list[str]:
    """Internal TQ storage-backend overrides.

    Transport backend selection is not exposed through the Hydra CLI. It
    defaults to the in-memory ``SimpleStorage``; set
    ``SPECO_TQ_STORAGE_BACKEND=MooncakeStore`` to use a Mooncake store
    (``SPECO_TQ_MOONCAKE_*`` tune its client configuration).
    """
    backend = _resolve_tq_backend(env)

    def _env(name: str, default: str) -> str:
        return str(env.get(name, default)).strip()

    if backend == "MooncakeStore":
        auto_init = _env_flag(
            env, "SPECO_TQ_MOONCAKE_AUTO_INIT", _MOONCAKE_AUTO_INIT_DEFAULT
        )
        return [
            f"{_TQ_PREFIX}.backend.storage_backend=MooncakeStore",
            f"{_TQ_PREFIX}.backend.MooncakeStore.auto_init={str(auto_init).lower()}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.metadata_server="
            f"{_env('SPECO_TQ_MOONCAKE_METADATA_SERVER', 'P2PHANDSHAKE')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.master_server_address="
            f"{_env('SPECO_TQ_MOONCAKE_MASTER', _MOONCAKE_MASTER_DEFAULT)}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.local_hostname="
            f"{_env('SPECO_TQ_MOONCAKE_LOCAL_HOSTNAME', '')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.protocol="
            f"{_env('SPECO_TQ_MOONCAKE_PROTOCOL', 'tcp')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.global_segment_size="
            f"{_env('SPECO_TQ_MOONCAKE_GLOBAL_SEGMENT_BYTES', str(4 * 1024**3))}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.local_buffer_size="
            f"{_env('SPECO_TQ_MOONCAKE_LOCAL_BUFFER_BYTES', str(2 * 1024**3))}",
        ]
    return [
        f"{_TQ_PREFIX}.backend.storage_backend=SimpleStorage",
        f"{_TQ_PREFIX}.backend.SimpleStorage.total_storage_size=17179869184",
        f"{_TQ_PREFIX}.backend.SimpleStorage.num_data_storage_units=8",
    ]


def build_pipeline_commands(
    config: PipelineConfig,
    training_args: Sequence[str],
    *,
    ray_address: str,
    python_executable: str = sys.executable,
    env: Mapping[str, str] | None = None,
) -> PipelineCommands:
    """Build the internal commands without exposing transport options.

    ``env`` selects the TQ storage backend (defaults to ``os.environ``); pass
    the same mapping to :func:`run_pipeline` so backend selection and master
    validation read one source.
    """
    backend_env = os.environ if env is None else env

    drafter_path = _strip_quotes(_find_override(training_args, _DRAFTER_PATH_KEY) or "")
    _, resume_metadata = load_standalone_resume(
        drafter_path or None,
        input_path=config.input_path,
    )
    resumed_optimizer_step = (
        int(resume_metadata.get("optimizer_step", 0))
        if resume_metadata is not None
        else 0
    )
    tq_overrides = [
        f"{_TQ_PREFIX}.enable=true",
        f"{_TQ_PREFIX}.ray.address={ray_address}",
        f"{_TQ_PREFIX}.ray.namespace={_TQ_NAMESPACE}",
        f"{_TQ_PREFIX}.partition_id={_TQ_PARTITION}",
        f"{_TQ_PREFIX}.run_id={config.run_id}",
        f"{_TQ_PREFIX}.drop_last=true",
        *_tq_backend_overrides(backend_env),
    ]
    parsed_endpoint = urlparse(config.vllm_endpoints[0])
    vllm_port = parsed_endpoint.port or (
        443 if parsed_endpoint.scheme == "https" else 80
    )
    # extract_hidden_states uses the model's layer-output convention. Qwen3-4B/8B
    # have 36 transformer layers; the default DSpark auxiliary selection ends at
    # 33 and requests the final layer output as 36.
    final_layer_id = _target_final_layer_id(
        config.model_path, config.vllm_aux_hidden_state_layer_ids
    )
    capture_layer_ids = _vllm_capture_layer_ids(
        config.vllm_aux_hidden_state_layer_ids, final_layer_id
    )
    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": capture_layer_ids}
        },
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": _VLLM_HIDDEN_STATES_DIR,
            "use_synchronization_lock": True,
        },
    }
    vllm = None
    if len(config.vllm_endpoints) == 1 and parsed_endpoint.hostname in {
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
    }:
        vllm = [
            "vllm",
            "serve",
            config.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(vllm_port),
            "--gpu-memory-utilization",
            _DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
            "--speculative-config",
            json.dumps(speculative_config, separators=(",", ":")),
            "--kv-transfer-config",
            json.dumps(kv_transfer_config, separators=(",", ":")),
            "--no-enable-chunked-prefill",
        ]
    owner = [
        python_executable,
        "-m",
        "verl_speco.tq_owner",
        *tq_overrides,
    ]
    producer_tuning_overrides = []
    for item in training_args:
        parsed = _split_override(item)
        if parsed is None or not _is_forwarded_producer_override(parsed[0]):
            continue
        key = parsed[0]
        if key in _PRODUCER_LAUNCHER_OWNED_KEYS:
            # The launcher computes these and sets them on the Producer command.
            continue
        if key not in _producer_config_keys():
            raise ValueError(
                f"Unknown Producer override {key!r}; it is not defined under "
                f"{_PRODUCER_PREFIX} in speco_base.yaml"
            )
        producer_tuning_overrides.append(item)
    producer_overrides = [
        f"{_ALGORITHM_KEY}={config.algorithm}",
        *tq_overrides,
        *producer_tuning_overrides,
        f"speco.standalone_tq_producer.input_path={config.input_path}",
        "speco.standalone_tq_producer.resume_checkpoint_path="
        + (drafter_path if resume_metadata is not None else "null"),
        f"speco.standalone_tq_producer.tokenizer_path={config.tokenizer_path}",
        "speco.standalone_tq_producer.tokenizer_fingerprint="
        + _stable_path_identity("tokenizer", config.tokenizer_path),
        f"speco.standalone_tq_producer.target_model_id={config.model_path}",
        "speco.standalone_tq_producer.target_model_revision="
        + _stable_path_identity("target", config.model_path),
        "speco.standalone_tq_producer.target_layer_ids="
        + _hydra_list(config.target_layer_ids),
        "speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids="
        + _hydra_list(config.vllm_aux_hidden_state_layer_ids),
        "speco.standalone_tq_producer.vllm_endpoints="
        + _hydra_list(config.vllm_endpoints),
        f"speco.standalone_tq_producer.vllm_model={config.model_path}",
        "speco.standalone_tq_producer.max_samples="
        + str(
            _producer_max_samples(
                training_args,
                resumed_optimizer_step=resumed_optimizer_step,
            )
        ),
    ]
    consumer_internal = [
        f"{_FEATURE_STORE_PREFIX}.type=tq",
        f"{_FEATURE_STORE_PREFIX}.path=null",
        f"{_FEATURE_STORE_PREFIX}.shuffle=false",
        f"{_FEATURE_STORE_PREFIX}.repeat=false",
        *tq_overrides,
    ]
    hidden_dtype = _strip_quotes(
        _find_override(training_args, _PRODUCER_HIDDEN_DTYPE_KEY) or "bfloat16"
    )
    dspark_l1_loss_alpha = float(
        _strip_quotes(_find_override(training_args, _DSPARK_L1_LOSS_ALPHA_KEY) or "0.9")
    )
    hidden_layout = resolve_drafter_hidden_states_layout(
        config.algorithm,
        {"dspark_l1_loss_alpha": dspark_l1_loss_alpha},
    )
    consumer_internal.extend(
        [
            f"{_TQ_PREFIX}.expected_feature.algorithm={config.algorithm}",
            f"{_TQ_PREFIX}.expected_feature.target_model_path="
            + json.dumps(config.model_path),
            f"{_TQ_PREFIX}.expected_feature.target_revision="
            + _stable_path_identity("target", config.model_path),
            f"{_TQ_PREFIX}.expected_feature.tokenizer_fingerprint="
            + _stable_path_identity("tokenizer", config.tokenizer_path),
            f"{_TQ_PREFIX}.expected_feature.target_layer_ids="
            + _hydra_list(config.target_layer_ids),
            f"{_TQ_PREFIX}.expected_feature.hidden_states_layout={hidden_layout}",
            f"{_TQ_PREFIX}.expected_feature.hidden_dtype={hidden_dtype}",
        ]
    )
    algorithm_layer_ids_key = _ALGORITHM_TARGET_LAYER_IDS_KEYS.get(config.algorithm)
    if algorithm_layer_ids_key is not None:
        consumer_internal.append(
            f"{algorithm_layer_ids_key}={_hydra_list(config.target_layer_ids)}"
        )
    # The subprocess path normally passes through draft_train_launcher, which
    # removes user-facing aliases such as num_gpus_per_node before Hydra sees
    # them. Ray composes Hydra directly, so apply the exact same normalization.
    launch_config = resolve_launch_config(list(training_args))
    normalized_training_args = normalize_training_args(
        list(training_args), launch_config
    )
    consumer_overrides = _replace_internal_overrides(
        normalized_training_args, consumer_internal
    )
    producer = [
        python_executable,
        "-m",
        "verl_speco.standalone_tq_producer",
        *producer_overrides,
    ]
    consumer = [
        python_executable,
        "-m",
        "verl_speco.draft_train_launcher",
        *consumer_overrides,
    ]
    return PipelineCommands(
        vllm=vllm,
        vllm_endpoints=config.vllm_endpoints,
        owner=owner,
        producer=producer,
        consumer=consumer,
        producer_overrides=tuple(producer_overrides),
        consumer_overrides=tuple(consumer_overrides),
    )


def _wait_for_owner_ready(
    owner: subprocess.Popen[Any],
    ready_file: Path,
    *,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + timeout_seconds
    while not ready_file.is_file():
        returncode = owner.poll()
        if returncode is not None:
            raise RuntimeError(f"TransferQueue owner exited early ({returncode})")
        if monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for TransferQueue owner readiness")
        sleep(0.1)


def _vllm_is_ready(endpoint: str, *, timeout_seconds: float = 1.0) -> bool:
    try:
        with urlopen(
            f"{endpoint.rstrip('/')}/models", timeout=timeout_seconds
        ) as response:
            return 200 <= int(response.status) < 300
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


def _wait_for_vllm_ready(
    process: subprocess.Popen[Any],
    endpoint: str,
    *,
    timeout_seconds: float,
    endpoint_ready: Callable[[str], bool] = _vllm_is_ready,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + timeout_seconds
    while not endpoint_ready(endpoint):
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"hidden-state vLLM exited before becoming ready ({returncode})"
            )
        if monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for hidden-state vLLM at {endpoint}")
        sleep(1.0)


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_pipeline(
    commands: PipelineCommands,
    *,
    ray_address: str,
    environ: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    poll_interval_seconds: float = 0.2,
    owner_ready_timeout_seconds: float = 120,
    vllm_ready_timeout_seconds: float = 900,
    endpoint_ready: Callable[[str], bool] = _vllm_is_ready,
) -> int:
    """Run the internal processes and return the training Consumer status."""

    base_env = dict(os.environ if environ is None else environ)
    # Ray itself also reads RAY_ADDRESS.  Pin every child (including the
    # torchrun ranks created by the Consumer launcher) to the control plane
    # created above instead of allowing a stale inherited value to win.
    base_env["RAY_ADDRESS"] = ray_address
    validate_tq_backend(base_env)
    owner: subprocess.Popen[Any] | None = None
    producer: subprocess.Popen[Any] | None = None
    consumer: subprocess.Popen[Any] | None = None
    vllm: subprocess.Popen[Any] | None = None
    hidden_states_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="speco-tq-launch-") as temp_dir:
            ready_file = Path(temp_dir) / "owner.ready"
            hidden_states_temp = tempfile.TemporaryDirectory(
                prefix="speco-vllm-hidden-states-"
            )
            hidden_states_dir = Path(hidden_states_temp.name)
            unavailable_endpoints = [
                endpoint
                for endpoint in commands.vllm_endpoints
                if not endpoint_ready(endpoint)
            ]
            if unavailable_endpoints:
                if commands.vllm is None:
                    raise RuntimeError(
                        "The configured hidden-state vLLM endpoints are unavailable: "
                        + ", ".join(unavailable_endpoints)
                    )
                config_endpoint = commands.vllm_endpoints[0]
                vllm_command = [
                    part.replace(_VLLM_HIDDEN_STATES_DIR, str(hidden_states_dir))
                    for part in commands.vllm
                ]
                logger.info("Starting hidden-state vLLM at %s", config_endpoint)
                vllm = popen(vllm_command, env=base_env)
            owner_env = {**base_env, "SPECO_TQ_OWNER_READY_FILE": str(ready_file)}
            logger.info("Starting TransferQueue owner")
            owner = popen(commands.owner, env=owner_env)
            _wait_for_owner_ready(
                owner,
                ready_file,
                timeout_seconds=owner_ready_timeout_seconds,
            )
            if vllm is not None:
                _wait_for_vllm_ready(
                    vllm,
                    config_endpoint,
                    timeout_seconds=vllm_ready_timeout_seconds,
                    endpoint_ready=endpoint_ready,
                )
            else:
                unavailable_endpoints = [
                    endpoint
                    for endpoint in commands.vllm_endpoints
                    if not endpoint_ready(endpoint)
                ]
                if unavailable_endpoints:
                    raise RuntimeError(
                        "hidden-state vLLM became unavailable at: "
                        + ", ".join(unavailable_endpoints)
                    )
            logger.info("Starting standalone DSpark Consumer")
            consumer = popen(commands.consumer, env=base_env)
            logger.info("Starting standalone vLLM Producer")
            producer = popen(commands.producer, env=base_env)

            while True:
                owner_status = owner.poll()
                vllm_status = None if vllm is None else vllm.poll()
                producer_status = producer.poll()
                consumer_status = consumer.poll()
                if owner_status is not None:
                    raise RuntimeError(
                        f"TransferQueue owner exited during training ({owner_status})"
                    )
                if producer_status is not None and producer_status != 0:
                    return int(producer_status)
                if vllm_status is not None:
                    raise RuntimeError(
                        f"hidden-state vLLM exited during training ({vllm_status})"
                    )
                if consumer_status is not None:
                    return int(consumer_status)
                time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        logger.warning("Standalone DSpark training interrupted")
        return 130
    finally:
        _stop_process(producer)
        _stop_process(consumer)
        _stop_process(owner)
        _stop_process(vllm)
        if hidden_states_temp is not None:
            hidden_states_temp.cleanup()


def _format_command(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def _preflight_input_file(input_path: str) -> None:
    path = Path(input_path)
    if not path.is_file():
        raise ValueError(f"Training file does not exist: {input_path}")
    from verl_speco.producer.input_reader import iter_input_records

    try:
        next(iter_input_records(path))
    except StopIteration as exc:
        raise ValueError(f"Training file contains no samples: {input_path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch standalone DSpark training through Producer/TQ/Consumer."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--runtime-backend",
        choices=("subprocess", "ray"),
        default=None,
        help="Override speco.draft_training.runtime_backend.",
    )
    args, training_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO)
    runtime_backend = (
        args.runtime_backend
        or _strip_quotes(
            _find_override(training_args, _RUNTIME_BACKEND_KEY) or "ray"
        ).lower()
    )
    if runtime_backend not in {"subprocess", "ray"}:
        parser.error(f"{_RUNTIME_BACKEND_KEY} must be either subprocess or ray")

    try:
        config = resolve_pipeline_config(training_args)
        _validate_runtime_backend_topology(runtime_backend, training_args)
        if args.dry_run:
            commands = build_pipeline_commands(
                config,
                training_args,
                ray_address="127.0.0.1:6379",
                python_executable=args.python_executable,
            )
            printable_commands = [
                ("vllm", commands.vllm),
                ("owner", commands.owner),
                ("producer", commands.producer),
                ("consumer", commands.consumer),
            ]
            for role, command in printable_commands:
                if command is None:
                    print(
                        f"{role}: external services at "
                        + ", ".join(commands.vllm_endpoints)
                    )
                    continue
                print(f"{role}: {_format_command(command)}")
            return 0
        _preflight_input_file(config.input_path)
        ray_session = start_ray_session()
        try:
            commands = build_pipeline_commands(
                config,
                training_args,
                ray_address=ray_session.address,
                python_executable=args.python_executable,
            )
            logger.info("Using task-local Ray control plane at %s", ray_session.address)
            if runtime_backend == "ray":
                from verl_speco.standalone_ray_runtime import run_ray_pipeline

                return run_ray_pipeline(
                    commands,
                    ray_module=ray_session.module,
                    ray_address=ray_session.address,
                )
            return run_pipeline(commands, ray_address=ray_session.address)
        finally:
            ray_session.close()
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        logger.error("Standalone TQ training failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PipelineCommands",
    "PipelineConfig",
    "RaySession",
    "build_pipeline_commands",
    "main",
    "resolve_pipeline_config",
    "run_pipeline",
    "start_ray_session",
]
