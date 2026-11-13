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
import ipaddress
import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import socket
import stat
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


def _local_ip() -> str:
    from verl_speco.producer.hidden_states_store import _local_ip as resolve

    return resolve()


def load_standalone_resume(*args, **kwargs):
    from verl_speco.trainer.standalone_resume import load_standalone_resume as load

    return load(*args, **kwargs)


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
_DSPARK_CONFIDENCE_LOSS_ALPHA_KEY = (
    "actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha"
)
_DFLASH_LK_LOSS_ALPHA_KEY = (
    "actor_rollout_ref.rollout.drafter.training.dflash_lk_loss_alpha"
)
_DSPARK_LK_LOSS_ALPHA_KEY = (
    "actor_rollout_ref.rollout.drafter.training.dspark_lk_loss_alpha"
)
_DSPARK_CE_TARGET_SOURCE_KEY = (
    "actor_rollout_ref.rollout.drafter.training.ce_target_source"
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
_DRAFT_TRAINER_CONFIG_PATH = Path(__file__).with_name("config") / "draft_trainer.yaml"
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
        f"{_TQ_PREFIX}.expected_feature.payload_kind",
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
_HS_STORES = ("file", "mooncake", "native")
_HS_STORE_ENV = "SPECO_VLLM_HIDDEN_STATES_STORE"
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
    local_model_cache: tuple[str, str] | None = None
    native_settings: dict[str, Any] | None = None


@dataclass(frozen=True)
class RaySession:
    module: Any
    address: str
    owns_connection: bool = True

    def close(self) -> None:
        if self.owns_connection:
            # Disconnect this driver, never stop an external Ray cluster.
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


def _find_override(
    overrides: Sequence[str], key: str, *, normalize_key: bool = True
) -> str | None:
    for item in reversed(overrides):
        parsed = _split_override(item)
        if parsed is not None:
            candidate = parsed[0].lstrip("+") if normalize_key else parsed[0]
            if candidate == key:
                return parsed[1]
    return None


def _find_last_override(overrides: Sequence[str], keys: Sequence[str]) -> str | None:
    wanted = set(keys)
    for item in reversed(overrides):
        parsed = _split_override(item)
        if parsed and parsed[0].lstrip("+") in wanted:
            return parsed[1]
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
    raw = _find_last_override(overrides, keys)
    value = int(_strip_quotes(raw)) if raw is not None else int(default)
    if value <= 0:
        raise ValueError(f"{keys[0]} must be positive, got {value}")
    return value


def _producer_max_samples(
    training_args: Sequence[str], *, resumed_optimizer_step: int = 0
) -> int:
    """Return samples needed for exactly max_steps complete global batches."""

    raw_max_steps = _find_override(training_args, _MAX_STEPS_KEY, normalize_key=True)
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


def _verify_local_model_cache(source: str, cache: str) -> None:
    """Compare every local model file before launching a cache-backed service."""
    def inventory(root_name: str) -> dict[str, Path]:
        root = Path(root_name)
        if root.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(root) or not root.is_dir():
            raise ValueError(f"Model cache comparison requires a real directory: {root}")
        files: dict[str, Path] = {}
        def fail(error: OSError) -> None:
            raise error
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
            for name in dirs:
                candidate = Path(directory) / name
                if candidate.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(candidate):
                    raise ValueError(f"Model cache directory contains a link: {candidate}")
            for name in names:
                candidate = Path(directory) / name
                if not stat.S_ISREG(candidate.lstat().st_mode):
                    raise ValueError(f"Model cache contains a non-regular file: {candidate}")
                files[candidate.relative_to(root).as_posix()] = candidate
                if len(files) > 100_000:
                    raise ValueError("Model cache has too many files")
        if not files:
            raise ValueError(f"Model cache directory contains no files: {root}")
        return files

    def signature(path: Path) -> tuple[int, int, int, int, int]:
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"Model cache file changed type: {path}")
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def digest(path: Path) -> bytes:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.digest()

    source_files, cache_files = inventory(source), inventory(cache)
    if source_files.keys() != cache_files.keys():
        raise ValueError("Local model cache file inventory differs from the source model")
    signatures = {}
    for name in sorted(source_files):
        original, replica = source_files[name], cache_files[name]
        before = signature(original), signature(replica)
        if before[0][2] != before[1][2] or digest(original) != digest(replica):
            raise ValueError(f"Local model cache content differs for {name}")
        if (signature(original), signature(replica)) != before:
            raise ValueError(f"Model file changed during cache verification: {name}")
        signatures[name] = before
    if inventory(source).keys() != source_files.keys() or inventory(cache).keys() != cache_files.keys():
        raise ValueError("Model file inventory changed during cache verification")
    for name, before in signatures.items():
        if (signature(source_files[name]), signature(cache_files[name])) != before:
            raise ValueError(f"Model file changed during cache verification: {name}")


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
        run_id=(str(env.get("SPECO_NATIVE_RUN_ID", "")).strip()
                if _resolve_hidden_states_store(env) == "native" and env.get("SPECO_NATIVE_RUN_ID")
                else f"{algorithm.lower()}-{uuid.uuid4().hex}"),
    )


def start_ray_session(
    *,
    environ: Mapping[str, str] | None = None,
    ray_module: Any | None = None,
) -> RaySession:
    """Own a local runtime or explicitly connect to an external control plane."""

    env = os.environ if environ is None else environ
    external_address = _external_ray_address(env)
    ray_runtime = ray_module
    if ray_runtime is None:
        try:
            ray_runtime = importlib.import_module("ray")
        except ImportError as exc:
            raise RuntimeError(
                "Standalone TQ training requires Ray and TransferQueue==0.1.10"
            ) from exc
    if getattr(ray_runtime, "is_initialized", lambda: False)():
        context = ray_runtime.get_runtime_context()
        if (
            not external_address
            or str(context.gcs_address).strip() != external_address
            or context.namespace != _TQ_NAMESPACE
        ):
            raise ValueError(
                "Ray is already initialized with a different address or namespace; "
                "use a fresh driver or explicitly select its address"
            )
        return RaySession(ray_runtime, external_address, owns_connection=False)
    # Ignore stale inherited RAY_ADDRESS unless this task explicitly opts in.
    init_kwargs: dict[str, Any] = {
        "address": external_address or "local",
        "namespace": _TQ_NAMESPACE,
    }
    if not external_address:
        init_kwargs["include_dashboard"] = False
    num_cpus = str(env.get("SPECO_RAY_NUM_CPUS", "")).strip()
    if num_cpus and not external_address:
        if int(num_cpus) <= 0:
            raise ValueError("SPECO_RAY_NUM_CPUS must be positive")
        init_kwargs["num_cpus"] = int(num_cpus)
    ray_runtime.init(**init_kwargs)
    try:
        address = str(ray_runtime.get_runtime_context().gcs_address).strip()
        if not address:
            raise RuntimeError("Ray did not report a GCS address for TQ clients")
    except Exception:
        ray_runtime.shutdown()
        raise
    return RaySession(module=ray_runtime, address=address)


def _external_ray_address(env: Mapping[str, str]) -> str:
    """Only explicit GCS host:port addresses are valid for TQ subprocess clients."""
    address = str(env.get("SPECO_EXTERNAL_RAY_ADDRESS", "")).strip()
    if not address:
        return ""
    try:
        parsed = urlparse("//" + address)
        if (
            not parsed.hostname or parsed.port is None or not 0 < parsed.port < 65536
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or any(char.isspace() for char in address)
        ):
            raise ValueError
    except ValueError:
        raise ValueError(
            "SPECO_EXTERNAL_RAY_ADDRESS requires an explicit GCS host:port "
            "(not auto, local or a ray:// client URL)"
        ) from None
    return address


def _validate_runtime_backend_topology(
    runtime_backend: str, training_args: Sequence[str],
    *, environ: Mapping[str, str] | None = None,
) -> None:
    """Keep unsupported multi-node data paths from failing after model startup."""

    if runtime_backend != "ray":
        return
    env = os.environ if environ is None else environ
    external_address = _external_ray_address(env)
    nnodes = _positive_int_override(training_args, _NNODES_KEYS, default=1)
    _positive_int_override(training_args, _NPROC_KEYS, default=1)
    if nnodes != 1 and not external_address:
        raise ValueError(
            "Standalone local Ray requires speco.draft_training.nnodes=1; "
            "multi-node training needs SPECO_EXTERNAL_RAY_ADDRESS"
        )
    if nnodes != 1:
        _trainer_node_ips(training_args, nnodes)
    shared_root = _shared_hidden_states_root(env)
    if external_address:
        backend = _resolve_hidden_states_store(env)
        if backend not in {"mooncake", "native"} and shared_root is None:
            raise ValueError(
                "External Ray requires the mooncake hidden-state store or an explicit shared file root; "
                "local temporary file references are not shared across nodes"
            )
        endpoints = _parse_vllm_endpoints(env)
        if any(urlparse(endpoint).hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
               for endpoint in endpoints):
            raise ValueError("Multi-node Ray requires remotely reachable vLLM endpoints")
        if backend == "mooncake":
            settings = _hidden_states_mooncake_settings(env)
            host = urlparse("//" + settings["master_server_address"]).hostname
            if host in {None, "localhost", "127.0.0.1", "0.0.0.0", "::1"}:
                raise ValueError("Multi-node Ray requires a remotely reachable Mooncake master")
            if settings["local_hostname"]:
                raise ValueError(
                    "Multi-node Ray must resolve Mooncake local_hostname on each worker; "
                    "unset the shared local-hostname override"
                )


def _trainer_node_ips(training_args: Sequence[str], nnodes: int) -> list[str]:
    raw = _find_override(training_args, "speco.draft_training.trainer_node_ips", normalize_key=True)
    try:
        value = _strip_quotes(raw or "")
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError
        parts = value[1:-1].split(",")
        if len(parts) != nnodes:
            raise ValueError
        nodes = [str(ipaddress.ip_address(_strip_quotes(part))) for part in parts]
        if len(set(nodes)) != nnodes:
            raise ValueError
        # verl's WorkerGroup orders placement groups by node IP for rank assignment.
        return sorted(nodes)
    except (ValueError, TypeError):
        raise ValueError("trainer_node_ips must contain one distinct IP per trainer node") from None


def _trainer_topology_plan(runtime_backend: str, training_args: Sequence[str]) -> dict[str, Any]:
    """Describe logical trainer ranks without allocating Ray or devices."""
    nnodes = _positive_int_override(training_args, _NNODES_KEYS, default=1)
    nproc = _positive_int_override(training_args, _NPROC_KEYS, default=1)
    addresses = _trainer_node_ips(training_args, nnodes) if nnodes > 1 and runtime_backend == "ray" else None
    nodes = [
        {"node_rank": rank, "address": addresses[rank] if addresses else None,
         "ranks": [{"local_rank": local, "global_rank": rank * nproc + local}
                   for local in range(nproc)]}
        for rank in range(nnodes)
    ]
    plan = {"schema_version": 1, "backend": runtime_backend, "nodes": nodes,
            "processes_per_node": nproc, "world_size": nnodes * nproc}
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plan["topology_sha256"] = hashlib.sha256(payload).hexdigest()
    return plan


def _hydra_list(values: Sequence[Any]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _replace_internal_overrides(
    training_args: Sequence[str], internal: Sequence[str]
) -> list[str]:
    cleaned: list[str] = []
    for item in training_args:
        parsed = _split_override(item)
        if parsed is not None and parsed[0].lstrip("+") in _INTERNAL_OVERRIDE_KEYS:
            continue
        cleaned.append(item)
    return [*cleaned, *internal]


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    """Return ``env[name]`` stripped, or ``default`` when it is unset."""
    value = env.get(name)
    return default if value is None else str(value).strip()


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

    if backend == "MooncakeStore":
        auto_init = _env_flag(
            env, "SPECO_TQ_MOONCAKE_AUTO_INIT", _MOONCAKE_AUTO_INIT_DEFAULT
        )
        return [
            f"{_TQ_PREFIX}.backend.storage_backend=MooncakeStore",
            f"{_TQ_PREFIX}.backend.MooncakeStore.auto_init={str(auto_init).lower()}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.metadata_server="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_METADATA_SERVER', 'P2PHANDSHAKE')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.master_server_address="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_MASTER', _MOONCAKE_MASTER_DEFAULT)}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.local_hostname="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_LOCAL_HOSTNAME', '')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.protocol="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_PROTOCOL', 'tcp')}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.global_segment_size="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_GLOBAL_SEGMENT_BYTES', str(4 * 1024**3))}",
            f"{_TQ_PREFIX}.backend.MooncakeStore.local_buffer_size="
            f"{_env(env, 'SPECO_TQ_MOONCAKE_LOCAL_BUFFER_BYTES', str(2 * 1024**3))}",
        ]
    return [
        f"{_TQ_PREFIX}.backend.storage_backend=SimpleStorage",
        f"{_TQ_PREFIX}.backend.SimpleStorage.total_storage_size=17179869184",
        f"{_TQ_PREFIX}.backend.SimpleStorage.num_data_storage_units=8",
    ]


def _resolve_hidden_states_store(env: Mapping[str, str]) -> str:
    """Return the selected hidden-state transfer backend.

    ``file`` keeps the legacy safetensors connector; ``mooncake`` swaps in the
    out-of-tree handle-based connector from ``hs_connectors``.
    """
    backend = str(env.get(_HS_STORE_ENV, "file")).strip().lower()
    if backend not in _HS_STORES:
        raise RuntimeError(
            f"Unsupported {_HS_STORE_ENV}={backend!r}; expected one of "
            f"{list(_HS_STORES)}."
        )
    return backend


def _shared_hidden_states_root(env: Mapping[str, str]) -> str | None:
    raw = str(env.get("SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT", "")).strip()
    if not raw:
        return None
    if not Path(raw).is_absolute() or _resolve_hidden_states_store(env) != "file":
        raise ValueError("SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT requires an absolute path and file store")
    return raw


def _hidden_states_mooncake_settings(env: Mapping[str, str]) -> dict[str, Any]:
    """Resolve Mooncake settings, falling back to the TQ Mooncake variables."""

    def pick(name: str, fallback: str, default: str) -> str:
        return _env(env, name) or _env(env, fallback) or default

    def pick_int(name: str, fallback: str, default: int) -> int:
        raw = pick(name, fallback, str(default))
        try:
            return int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} (or {fallback}) must be an integer, got {raw!r}"
            ) from exc

    local_hostname = (
        pick(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_HOSTNAME",
            "SPECO_TQ_MOONCAKE_LOCAL_HOSTNAME",
            "",
        )
        or ("" if _external_ray_address(env) else _local_ip())
    )
    return {
        "local_hostname": local_hostname,
        "metadata_server": pick(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_METADATA_SERVER",
            "SPECO_TQ_MOONCAKE_METADATA_SERVER",
            "P2PHANDSHAKE",
        ),
        "master_server_address": pick(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER",
            "SPECO_TQ_MOONCAKE_MASTER",
            _MOONCAKE_MASTER_DEFAULT,
        ),
        "global_segment_size": pick_int(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_GLOBAL_SEGMENT_BYTES",
            "SPECO_TQ_MOONCAKE_GLOBAL_SEGMENT_BYTES",
            4 * 1024**3,
        ),
        "local_buffer_size": pick_int(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_BUFFER_BYTES",
            "SPECO_TQ_MOONCAKE_LOCAL_BUFFER_BYTES",
            2 * 1024**3,
        ),
        "protocol": pick(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL",
            "SPECO_TQ_MOONCAKE_PROTOCOL",
            "tcp",
        ),
        "device_name": pick(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_DEVICE_NAME",
            "SPECO_TQ_MOONCAKE_DEVICE_NAME",
            "",
        ),
        "num_writer_threads": pick_int(
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_WRITER_THREADS",
            "SPECO_TQ_MOONCAKE_WRITER_THREADS",
            8,
        ),
    }


def _vllm_kv_transfer_config(env: Mapping[str, str]) -> dict[str, Any]:
    shared_root = _shared_hidden_states_root(env)
    """Build the vLLM ``--kv-transfer-config`` for the selected backend."""
    backend = _resolve_hidden_states_store(env)
    if backend in {"mooncake", "native"}:
        return {
            "kv_connector": ("SpecoNativeHiddenStatesConnector" if backend == "native"
                             else "SpecoMooncakeHiddenStatesConnector"),
            "kv_connector_module_path": "verl_speco.mooncake_hidden_states_connector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "mooncake": _hidden_states_mooncake_settings(env),
            },
        }
    return {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": shared_root or _VLLM_HIDDEN_STATES_DIR,
            "use_synchronization_lock": True,
        },
    }


def _hidden_states_store_overrides(env: Mapping[str, str]) -> list[str]:
    """Producer-side overrides that select the consumer store backend."""
    root = _shared_hidden_states_root(env)
    backend = _resolve_hidden_states_store(env)
    if backend == "native":
        return [f"{_PRODUCER_PREFIX}.hidden_states_store.backend=native"]
    if backend != "mooncake":
        return [] if root is None else [f"{_PRODUCER_PREFIX}.hidden_states_store.file_root={json.dumps(root, ensure_ascii=False)}"]
    settings = _hidden_states_mooncake_settings(env)
    prefix = f"{_PRODUCER_PREFIX}.hidden_states_store"
    return [
        f"{prefix}.backend=mooncake",
        f"{prefix}.mooncake.local_hostname={settings['local_hostname']}",
        f"{prefix}.mooncake.metadata_server={settings['metadata_server']}",
        f"{prefix}.mooncake.master_server_address={settings['master_server_address']}",
        f"{prefix}.mooncake.global_segment_size={settings['global_segment_size']}",
        f"{prefix}.mooncake.local_buffer_size={settings['local_buffer_size']}",
        f"{prefix}.mooncake.protocol={settings['protocol']}",
        f"{prefix}.mooncake.device_name={settings['device_name']}",
        f"{prefix}.mooncake.num_writer_threads={settings['num_writer_threads']}",
    ]


def _native_launch_settings(config: PipelineConfig, training_args: Sequence[str], env: Mapping[str, str],
                            *, launcher_owned: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    """Resolve bounded source/reader resources and scheduler preconditions."""
    if _strip_quotes(_find_override(training_args, _RUNTIME_BACKEND_KEY, normalize_key=True) or "ray") != "ray":
        raise ValueError("Native hidden states require the Ray runtime")
    world_size = (_positive_int_override(training_args, _NNODES_KEYS, default=1)
                  * _positive_int_override(training_args, _NPROC_KEYS, default=1))
    batch = world_size * _positive_int_override(training_args, (_BATCH_SIZE_PER_GPU_KEY,), default=4)

    def integer(name, default, minimum, maximum):
        try:
            value = int(_env(env, name, str(default)))
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    try:
        endpoints = json.loads(_env(env, "SPECO_NATIVE_READER_ENDPOINTS", "[]"))
    except (TypeError, ValueError):
        raise ValueError("SPECO_NATIVE_READER_ENDPOINTS must be a JSON array") from None
    if (not isinstance(endpoints, list) or len(endpoints) != world_size
            or any(not isinstance(item, str) or not item.strip() or item != item.strip() or len(item) > 256
                   for item in endpoints) or len(set(endpoints)) != world_size):
        raise ValueError("SPECO_NATIVE_READER_ENDPOINTS must contain one distinct endpoint per rank")
    source_endpoint = _env(env, "SPECO_NATIVE_SOURCE_ENDPOINT")
    if launcher_owned and (not source_endpoint or len(source_endpoint) > 256 or source_endpoint in endpoints):
        raise ValueError("Launcher-owned native vLLM needs a source endpoint distinct from reader endpoints")
    directory = _env(env, "SPECO_NATIVE_DIRECTORY")
    reuse_existing = bool(directory and _env(env, "SPECO_NATIVE_RUN_ID"))
    if not directory:
        if not launcher_owned:
            raise ValueError("External native vLLM requires SPECO_NATIVE_DIRECTORY and SPECO_NATIVE_RUN_ID")
        directory = f"speco_native_{uuid.uuid4().hex}"
    if not launcher_owned and not _env(env, "SPECO_NATIVE_RUN_ID"):
        raise ValueError("External native vLLM requires SPECO_NATIVE_DIRECTORY and SPECO_NATIVE_RUN_ID")
    if reuse_existing and _env(env, "SPECO_NATIVE_RUN_ID") != config.run_id:
        raise ValueError("Native vLLM and launcher must use the same run identity")
    if len(directory) > 256 or not config.run_id or len(config.run_id) > 256:
        raise ValueError("Invalid native directory name or run identity")
    align = 2 * 1024**2
    source_slots = integer("SPECO_NATIVE_SOURCE_MAX_SLOTS", max(batch, 16), batch, 4096)
    source_bytes = integer("SPECO_NATIVE_SOURCE_MAX_BYTES", source_slots * (2 * align - 1),
                           batch * (2 * align - 1), 2**63 - 1)
    reader_bytes = integer("SPECO_NATIVE_READER_MAX_BYTES", 2 * 1024**3, 2 * align - 1, 2**63 - 1)
    chunk = integer("SPECO_NATIVE_CHUNK_BYTES", align, align, 2**63 - 1)
    if chunk % align:
        raise ValueError("SPECO_NATIVE_CHUNK_BYTES must be a multiple of 2 MiB")
    try:
        timeout = float(_env(env, "SPECO_NATIVE_CONTROL_TIMEOUT", "30"))
    except ValueError:
        raise ValueError("SPECO_NATIVE_CONTROL_TIMEOUT must be a positive finite number") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("SPECO_NATIVE_CONTROL_TIMEOUT must be a positive finite number")
    settings = {"directory": directory, "run_id": config.run_id, "reuse_existing": reuse_existing,
                "namespace": _env(env, "SPECO_NATIVE_RAY_NAMESPACE", "transfer_queue"),
                "generation": integer("SPECO_NATIVE_GENERATION", 0, 0, 2**63 - 1),
                "max_sources": integer("SPECO_NATIVE_MAX_SOURCES", len(config.vllm_endpoints), 1, 4096),
                "max_records": integer("SPECO_NATIVE_MAX_RECORDS", 1_000_000, 1, 10_000_000),
                "max_descriptor_bytes": integer("SPECO_NATIVE_MAX_DESCRIPTOR_BYTES", 256 * 1024**2, 1, 16 * 1024**3),
                "metadata_server": _env(env, "SPECO_NATIVE_METADATA_SERVER", "P2PHANDSHAKE"),
                "source_endpoint": source_endpoint, "reader_endpoints": endpoints, "source_max_bytes": source_bytes,
                "source_max_slots": source_slots, "chunk_bytes": chunk, "timeout": timeout}
    if not settings["namespace"] or not settings["metadata_server"]:
        raise ValueError("Native Ray namespace and metadata server must not be empty")
    source_overrides, consumer_overrides = [], []
    for key in ("max_published_references", "max_reference_execution_slots"):
        full = f"{_PRODUCER_PREFIX}.{key}"
        raw = _find_override(training_args, full, normalize_key=True)
        value = int(_strip_quotes(raw)) if raw is not None else batch
        if value < batch or (key == "max_reference_execution_slots" and value > source_slots):
            raise ValueError(f"{full} must cover one batch and execution slots must fit the source")
        if key == "max_reference_execution_slots" and source_bytes < value * (2 * align - 1):
            raise ValueError("SPECO_NATIVE_SOURCE_MAX_BYTES must cover the minimum allocation for each execution slot")
        if raw is None:
            source_overrides.append(f"{full}={value}")
            consumer_overrides.append(f"{full}={value}")
    for key, required in (("speco.draft_training.scheduler.per_rank_reference_ack", "true"),
                          (f"{_TQ_PREFIX}.consumer_batch_workers", "0")):
        raw = _find_override(training_args, key, normalize_key=True)
        if raw is None:
            consumer_overrides.append(f"{key}={required}")
        elif _strip_quotes(raw).lower() != required:
            raise ValueError(f"Native hidden states require {key}={required}")
    for key in (f"{_TQ_PREFIX}.consumer_node_reference_owner",
                "actor_rollout_ref.rollout.drafter.training.reference_owned_windows"):
        raw = _find_override(training_args, key, normalize_key=True)
        if raw is not None and _strip_quotes(raw).lower() != "false":
            raise ValueError(f"Native hidden states require {key}=false")
    prefix = "actor_rollout_ref.rollout.drafter.training.target_feature_replay.hidden_states_store.native"
    consumer_overrides.extend([f"{prefix}.endpoints={json.dumps(endpoints, separators=(',', ':'))}",
        f"{prefix}.metadata_server={json.dumps(settings['metadata_server'])}", f"{prefix}.generation={settings['generation']}",
        f"{prefix}.max_bytes={reader_bytes}", f"{prefix}.chunk_bytes={chunk}"])
    return settings, source_overrides, consumer_overrides


def validate_hidden_states_store(
    env: Mapping[str, str],
    *,
    connect: Callable[..., Any] | None = None,
    timeout: float = 2.0,
) -> None:
    """Fail fast when the handle-based store cannot reach its Mooncake master."""
    if _resolve_hidden_states_store(env) != "mooncake":
        return
    if _env_flag(env, "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_SKIP_PRECHECK", False):
        return
    raw = str(_hidden_states_mooncake_settings(env)["master_server_address"])
    host, _, port = raw.rpartition(":")
    host = host.strip("[]")
    if not host or not port.isdigit():
        raise RuntimeError(
            f"hidden-states Mooncake master address must be host:port, got {raw!r}"
        )
    probe = connect if connect is not None else socket.create_connection
    try:
        probe((host, int(port)), timeout=timeout).close()
    except OSError as exc:
        raise RuntimeError(
            "hidden-states Mooncake store selected, but no mooncake_master is "
            f"reachable at {host}:{port} ({exc}). Start mooncake_master there, or "
            "set SPECO_VLLM_HIDDEN_STATES_MOONCAKE_SKIP_PRECHECK=true."
        ) from exc


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
    kv_transfer_config = _vllm_kv_transfer_config(backend_env)
    cache_override = _find_override(training_args, "speco.draft_training.local_target_cache_path", normalize_key=True)
    cache_path = _strip_quotes(cache_override) if cache_override else ""
    if cache_path.lower() in {"null", "none"}:
        cache_path = ""
    vllm = None
    if len(config.vllm_endpoints) == 1 and parsed_endpoint.hostname in {
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
    }:
        vllm = [
            "vllm",
            "serve",
            cache_path or config.model_path,
            *(["--served-model-name", config.model_path] if cache_path else []),
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
    if cache_path and vllm is None:
        raise ValueError("Local target cache requires a launcher-owned local vLLM endpoint")
    owner = [
        python_executable,
        "-m",
        "verl_speco.tq_owner",
        *tq_overrides,
    ]
    producer_tuning_overrides = []
    for item in training_args:
        parsed = _split_override(item)
        if parsed is None or not _is_forwarded_producer_override(parsed[0].lstrip("+")):
            continue
        key = parsed[0].lstrip("+")
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
        "actor_rollout_ref.rollout.drafter.training.dflash_block_size="
        + (
            _find_override(
                training_args,
                "actor_rollout_ref.rollout.drafter.training.dflash_block_size",
                normalize_key=True,
            )
            or "16"
        ),
        "actor_rollout_ref.rollout.drafter.training.preserve_prompt_hidden_states="
        + (
            _find_override(
                training_args,
                "actor_rollout_ref.rollout.drafter.training.preserve_prompt_hidden_states",
                normalize_key=True,
            )
            or "true"
        ),
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
        *_hidden_states_store_overrides(backend_env),
    ]
    consumer_internal = [
        f"{_FEATURE_STORE_PREFIX}.type=tq",
        f"{_FEATURE_STORE_PREFIX}.path=null",
        f"{_FEATURE_STORE_PREFIX}.shuffle=false",
        f"{_FEATURE_STORE_PREFIX}.repeat=false",
        *tq_overrides,
    ]
    hidden_references = (
        _strip_quotes(_find_override(training_args, _RUNTIME_BACKEND_KEY) or "ray")
        == "ray"
    )
    native_settings = None
    if _resolve_hidden_states_store(backend_env) == "native":
        controlled = (f"{_PRODUCER_PREFIX}.hidden_states_store",
                      "actor_rollout_ref.rollout.drafter.training.target_feature_replay.hidden_states_store")
        for item in training_args:
            parsed = _split_override(item)
            if parsed and any(parsed[0].lstrip("+") == prefix or parsed[0].lstrip("+").startswith(prefix + ".")
                              for prefix in controlled):
                raise ValueError("Native hidden-state store settings are configured by SPECO_NATIVE_* environment variables")
        native_settings, source_native, consumer_native = _native_launch_settings(
            config, training_args, backend_env, launcher_owned=vllm is not None)
        sample_budget = _producer_max_samples(training_args, resumed_optimizer_step=resumed_optimizer_step)
        if sample_budget > native_settings["max_records"]:
            raise ValueError("SPECO_NATIVE_MAX_RECORDS must cover the requested sample budget; "
                             "allow extra records for filtered samples")
        producer_overrides.extend(source_native)
        consumer_internal.extend(consumer_native)
    if hidden_references:
        consumer_internal.extend(
            item.replace(
                f"{_PRODUCER_PREFIX}.hidden_states_store",
                "actor_rollout_ref.rollout.drafter.training.target_feature_replay.hidden_states_store",
            )
            for item in _hidden_states_store_overrides(backend_env)
        )
        owner = []
        consumer_internal[0] = f"{_FEATURE_STORE_PREFIX}.type=ray_reference"
        producer_overrides.extend(
            [
                f"{_FEATURE_STORE_PREFIX}.type=ray_reference",
                f"{_TQ_PREFIX}.enable=false",
            ]
        )
        consumer_internal.append(f"{_TQ_PREFIX}.enable=false")
        kind_override = f"{_TQ_PREFIX}.expected_feature.payload_kind=vllm_reference"
        producer_overrides.append(kind_override)
        consumer_internal.append(kind_override)
        depth_key = f"{_TQ_PREFIX}.consumer_prefetch_batches"
        requested_depth = _find_override(training_args, depth_key, normalize_key=True)
        if requested_depth is not None and int(_strip_quotes(requested_depth)) < 1:
            raise ValueError(
                "Standalone vLLM file input requires consumer_prefetch_batches >= 1"
            )
        if requested_depth is None:
            consumer_internal.append(f"{depth_key}=4")
    hidden_dtype = _strip_quotes(
        _find_override(training_args, _PRODUCER_HIDDEN_DTYPE_KEY) or "bfloat16"
    )
    # The Producer's collected layout and the Consumer's expected layout must
    # agree, so resolve the loss settings that select it (dflash_aux vs
    # dflash_aux_plus_last) once here and forward the same values to both.
    dspark_l1_loss_alpha = _strip_quotes(
        _find_override(training_args, _DSPARK_L1_LOSS_ALPHA_KEY, normalize_key=True)
        or "0.9"
    )
    dspark_confidence_loss_alpha = _strip_quotes(
        _find_override(
            training_args, _DSPARK_CONFIDENCE_LOSS_ALPHA_KEY, normalize_key=True
        )
        or "0.0"
    )
    dspark_lk_loss_alpha = _strip_quotes(
        _find_override(training_args, _DSPARK_LK_LOSS_ALPHA_KEY, normalize_key=True)
        or "0.0"
    )
    dflash_lk_loss_alpha = _strip_quotes(
        _find_override(training_args, _DFLASH_LK_LOSS_ALPHA_KEY, normalize_key=True)
        or "0.0"
    )
    ce_target_source = _strip_quotes(
        _find_override(training_args, _DSPARK_CE_TARGET_SOURCE_KEY, normalize_key=True)
        or "labels"
    )
    producer_overrides.extend(
        [
            f"{_DSPARK_L1_LOSS_ALPHA_KEY}={dspark_l1_loss_alpha}",
            f"{_DSPARK_CONFIDENCE_LOSS_ALPHA_KEY}={dspark_confidence_loss_alpha}",
            f"{_DSPARK_LK_LOSS_ALPHA_KEY}={dspark_lk_loss_alpha}",
            f"{_DFLASH_LK_LOSS_ALPHA_KEY}={dflash_lk_loss_alpha}",
            f"{_DSPARK_CE_TARGET_SOURCE_KEY}={ce_target_source}",
        ]
    )
    hidden_layout = resolve_drafter_hidden_states_layout(
        config.algorithm,
        {
            "dspark_l1_loss_alpha": float(dspark_l1_loss_alpha),
            "dspark_lk_loss_alpha": float(dspark_lk_loss_alpha),
            "dflash_lk_loss_alpha": float(dflash_lk_loss_alpha),
            "dspark_confidence_loss_alpha": float(dspark_confidence_loss_alpha),
            "ce_target_source": ce_target_source,
        },
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
        local_model_cache=(config.model_path, cache_path) if cache_path else None,
        native_settings=native_settings,
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


def _cleanup_hidden_states_directory(path):
    import shutil
    path = Path(path)
    resolved = path.resolve()
    if (path.is_symlink() or resolved.parent != Path(tempfile.gettempdir()).resolve()
            or not resolved.name.startswith("speco-vllm-hidden-states-")):
        raise RuntimeError("Refusing to remove an unowned hidden-state directory")
    shutil.rmtree(resolved)


def _start_process(popen, command, *, env):
    # Only real, task-owned Linux children get a dedicated process session.
    grouped = os.name == "posix" and popen is subprocess.Popen
    process = popen(command, env=env, **({"start_new_session": True} if grouped else {}))
    if grouped:
        process._speco_process_group = process.pid
    return process


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    import signal

    group = getattr(process, "_speco_process_group", None)

    def confirm_group_exit():
        if os.name == "posix" and group is not None and group == process.pid:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return
            # The leader may already have been reaped and its PID reused. Do
            # not signal an unproven group; keep resources for explicit recovery.
            raise RuntimeError(f"Process group {group} exit is unconfirmed; retaining resources")

    if process.poll() is not None:
        confirm_group_exit()
        return

    def stop(kill=False):
        group = getattr(process, "_speco_process_group", None)
        if os.name == "posix" and group is not None and group == process.pid:
            try:
                if os.getpgid(process.pid) == group:
                    os.killpg(group, signal.SIGKILL if kill else signal.SIGTERM)
                    return
            except ProcessLookupError:
                return
        process.kill() if kill else process.terminate()
    stop()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        stop(kill=True)
        process.wait(timeout=10)
    confirm_group_exit()


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
    validate_hidden_states_store(base_env)
    owner: subprocess.Popen[Any] | None = None
    producer: subprocess.Popen[Any] | None = None
    consumer: subprocess.Popen[Any] | None = None
    vllm: subprocess.Popen[Any] | None = None
    hidden_states_temp: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="speco-tq-launch-") as temp_dir:
            ready_file = Path(temp_dir) / "owner.ready"
            hidden_states_temp = Path(tempfile.mkdtemp(
                prefix="speco-vllm-hidden-states-"
            ))
            hidden_states_dir = hidden_states_temp
            unavailable_endpoints = [
                endpoint
                for endpoint in commands.vllm_endpoints
                if not endpoint_ready(endpoint)
            ]
            if commands.local_model_cache and not unavailable_endpoints:
                raise RuntimeError("Local model cache requires a new launcher-owned vLLM service")
            if unavailable_endpoints:
                if commands.local_model_cache:
                    _verify_local_model_cache(*commands.local_model_cache)
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
                vllm = _start_process(popen, vllm_command, env=base_env)
            owner_env = {**base_env, "SPECO_TQ_OWNER_READY_FILE": str(ready_file)}
            logger.info("Starting TransferQueue owner")
            owner = _start_process(popen, commands.owner, env=owner_env)
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
            consumer = _start_process(popen, commands.consumer, env=base_env)
            logger.info("Starting standalone vLLM Producer")
            producer = _start_process(popen, commands.producer, env=base_env)

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
        failures = []
        for process in (producer, consumer, owner, vllm):
            try:
                _stop_process(process)
            except Exception as exc:
                failures.append(exc)
                logger.exception("Task-owned process shutdown was not confirmed")
        if failures and hidden_states_temp is not None:
            logger.error("Preserving hidden-state directory after incomplete process shutdown: %s", hidden_states_temp)
        elif hidden_states_temp is not None:
            _cleanup_hidden_states_directory(hidden_states_temp)
        if failures:
            raise RuntimeError("Standalone subprocess shutdown was not confirmed") from failures[0]


def _format_command(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def _preflight_input_file(input_path: str, *, max_jsonl_line_bytes: int | None = None) -> None:
    path = Path(input_path)
    if path.is_dir():
        from verl_speco.producer.sequence_replay import SequenceReplayReader

        if not len(SequenceReplayReader(path)):
            raise ValueError(f"Sequence replay contains no samples: {input_path}")
        return
    if not path.is_file():
        raise ValueError(f"Training file does not exist: {input_path}")
    from verl_speco.producer.input_reader import iter_input_records

    try:
        kwargs = {} if max_jsonl_line_bytes is None else {"max_jsonl_line_bytes": max_jsonl_line_bytes}
        next(iter_input_records(path, **kwargs))
    except StopIteration as exc:
        raise ValueError(f"Training file contains no samples: {input_path}") from exc


def _resolve_checkpoint_rollback(
    training_args: Sequence[str], *, require_input: bool = True,
) -> list[str]:
    """Resolve an explicitly enabled rollback before Producer and Consumer diverge."""
    key = "actor_rollout_ref.rollout.drafter.training.checkpoint_auto_rollback"
    raw = _find_override(training_args, key, normalize_key=True)
    if raw is None:
        base = OmegaConf.load(_DRAFT_TRAINER_CONFIG_PATH)
        default = OmegaConf.select(base, key, default=False)
        if type(default) is not bool:
            raise ValueError(f"{key} must be true or false")
        value = str(default).lower()
    else:
        value = _strip_quotes(raw).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{key} must be true or false")
    if value == "false":
        return list(training_args)
    requested = _strip_quotes(_find_override(training_args, _DRAFTER_PATH_KEY, normalize_key=True) or "")
    if not requested:
        raise ValueError("Automatic rollback requires a requested drafter checkpoint")
    input_path = _single_train_file(_find_override(training_args, _TRAIN_FILES_KEY)) if require_input else None
    from verl_speco.trainer.checkpoint import resolve_verified_standalone_checkpoint

    selected = resolve_verified_standalone_checkpoint(requested, input_path=input_path)
    if os.path.normcase(os.path.abspath(selected)) == os.path.normcase(os.path.abspath(requested)):
        return list(training_args)
    logger.warning("Rolling back standalone checkpoint from %s to %s", requested, selected)
    filtered = [item for item in training_args if not (
        _split_override(item) and _split_override(item)[0].lstrip("+") == _DRAFTER_PATH_KEY
    )]
    return [*filtered, f"{_DRAFTER_PATH_KEY}={json.dumps(selected.replace(chr(92), '/'), ensure_ascii=False)}"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch standalone DSpark training through Producer/TQ/Consumer."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-topology", action="store_true",
        help="Print the logical topology without loading models or connecting to services.",
    )
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
    # Keep command composition consistent with the explicit CLI backend too.
    training_args = [*training_args, f"{_RUNTIME_BACKEND_KEY}={runtime_backend}"]

    try:
        if args.dry_run_topology:
            _validate_runtime_backend_topology(runtime_backend, training_args)
            plan = _trainer_topology_plan(runtime_backend, training_args)
            plan["ray_address"] = _external_ray_address(os.environ) or "local"
            plan["hidden_states_store"] = _resolve_hidden_states_store(os.environ)
            plan["vllm_endpoints"] = _parse_vllm_endpoints(os.environ)
            canonical = {key: value for key, value in plan.items() if key != "topology_sha256"}
            plan["topology_sha256"] = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            print(json.dumps(plan, sort_keys=True))
            return 0
        training_args = _resolve_checkpoint_rollback(training_args)
        config = resolve_pipeline_config(training_args)
        _validate_runtime_backend_topology(runtime_backend, training_args)
        if args.dry_run:
            nnodes = _positive_int_override(training_args, _NNODES_KEYS, default=1)
            nproc = _positive_int_override(training_args, _NPROC_KEYS, default=1)
            print("runtime: " + json.dumps({
                "backend": runtime_backend,
                "ray_address": _external_ray_address(os.environ) or "local",
                "trainer_nodes": nnodes,
                "trainer_node_ips": _trainer_node_ips(training_args, nnodes) if nnodes > 1 and runtime_backend == "ray" else None,
                "processes_per_node": nproc,
                "world_size": nnodes * nproc,
                "trainer_topology": _trainer_topology_plan(runtime_backend, training_args),
                "hidden_states_store": _resolve_hidden_states_store(os.environ),
                "vllm_endpoints": config.vllm_endpoints,
            }, sort_keys=True))
            commands = build_pipeline_commands(
                config,
                training_args,
                ray_address=_external_ray_address(os.environ) or "127.0.0.1:6379",
                python_executable=args.python_executable,
            )
            printable_commands = [
                ("vllm", commands.vllm),
                ("owner", commands.owner),
                ("producer", commands.producer),
                ("consumer", commands.consumer),
            ]
            for role, command in printable_commands:
                if role == "owner" and not command:
                    continue
                if command is None:
                    print(
                        f"{role}: external services at "
                        + ", ".join(commands.vllm_endpoints)
                    )
                    continue
                print(f"{role}: {_format_command(command)}")
            return 0
        line_limit = _find_override(training_args, "speco.standalone_tq_producer.max_jsonl_line_bytes", normalize_key=True)
        if line_limit is None:
            _preflight_input_file(config.input_path)
        else:
            _preflight_input_file(config.input_path, max_jsonl_line_bytes=int(_strip_quotes(line_limit)))
        ray_session = start_ray_session()
        try:
            commands = build_pipeline_commands(
                config,
                training_args,
                ray_address=ray_session.address,
                python_executable=args.python_executable,
            )
            logger.info("Using Ray control plane at %s", ray_session.address)
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
