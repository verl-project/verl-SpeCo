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
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid

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
_ALGORITHM_TARGET_LAYER_IDS_KEYS = {
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
_PRODUCER_TUNING_KEYS = frozenset(
    {
        f"{_PRODUCER_PREFIX}.request_timeout",
        f"{_PRODUCER_PREFIX}.max_inflight_requests",
        f"{_PRODUCER_PREFIX}.per_endpoint_concurrency",
        f"{_PRODUCER_PREFIX}.input_queue_size",
        f"{_PRODUCER_PREFIX}.publish_queue_size",
        f"{_PRODUCER_PREFIX}.max_pending_samples",
        f"{_PRODUCER_PREFIX}.pending_poll_interval_seconds",
        f"{_PRODUCER_PREFIX}.max_sequence_length",
        f"{_PRODUCER_PREFIX}.max_feature_length",
        f"{_PRODUCER_PREFIX}.generation_max_tokens",
    }
)
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
    }
)

_DEFAULT_TARGET_LAYER_IDS = (1, 9, 17, 25, 33)
_DEFAULT_VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
_DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = "0.4"
_VLLM_HIDDEN_STATES_DIR = "__SPECO_HIDDEN_STATES_DIR__"
_TQ_NAMESPACE = "speco-drafter"
_TQ_PARTITION = "speco_drafter_features"


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    model_path: str
    tokenizer_path: str
    algorithm: str
    target_layer_ids: tuple[int, ...]
    vllm_endpoints: tuple[str, ...]
    run_id: str


@dataclass(frozen=True)
class PipelineCommands:
    vllm: list[str] | None
    vllm_endpoints: tuple[str, ...]
    owner: list[str]
    producer: list[str]
    consumer: list[str]


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


def _parse_layer_ids(value: str | None, *, config_key: str) -> tuple[int, ...]:
    if value is None or _strip_quotes(value).lower() in {"", "null", "none"}:
        return _DEFAULT_TARGET_LAYER_IDS
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


def _resolve_target_layer_ids(
    training_args: Sequence[str], algorithm: str
) -> tuple[int, ...]:
    """Resolve Producer layers without making the launcher DSpark-specific."""

    algorithm_key = _ALGORITHM_TARGET_LAYER_IDS_KEYS.get(algorithm)
    candidate_keys = (
        (_PRODUCER_TARGET_LAYER_IDS_KEY, algorithm_key)
        if algorithm_key is not None
        else (_PRODUCER_TARGET_LAYER_IDS_KEY,)
    )
    raw = _find_first_override(training_args, candidate_keys)
    return _parse_layer_ids(raw, config_key=candidate_keys[0])


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
    # Keep dry-run and model-registry IDs usable. The formal Qwen3-4B/8B
    # defaults select layer 33 and use transformer output 36 as the final state.
    return max(int(layer_id) for layer_id in target_layer_ids) + 3


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
    target_layer_ids = _resolve_target_layer_ids(training_args, algorithm)
    endpoints = _parse_vllm_endpoints(env)
    return PipelineConfig(
        input_path=input_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        algorithm=algorithm,
        target_layer_ids=target_layer_ids,
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


def build_pipeline_commands(
    config: PipelineConfig,
    training_args: Sequence[str],
    *,
    ray_address: str,
    python_executable: str = sys.executable,
) -> PipelineCommands:
    """Build the internal commands without exposing transport options."""

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
        f"{_TQ_PREFIX}.backend.storage_backend=SimpleStorage",
        f"{_TQ_PREFIX}.backend.SimpleStorage.total_storage_size=17179869184",
        f"{_TQ_PREFIX}.backend.SimpleStorage.num_data_storage_units=8",
    ]
    parsed_endpoint = urlparse(config.vllm_endpoints[0])
    vllm_port = parsed_endpoint.port or (
        443 if parsed_endpoint.scheme == "https" else 80
    )
    # extract_hidden_states uses the model's layer-output convention. Qwen3-4B/8B
    # have 36 transformer layers; the default DSpark auxiliary selection ends at
    # 33 and requests the final layer output as 36.
    final_layer_id = _target_final_layer_id(config.model_path, config.target_layer_ids)
    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [
                    *config.target_layer_ids,
                    final_layer_id,
                ]
            }
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
    producer_tuning_overrides = [
        item
        for item in training_args
        if (parsed := _split_override(item)) is not None
        and parsed[0] in _PRODUCER_TUNING_KEYS
    ]
    producer = [
        python_executable,
        "-m",
        "verl_speco.standalone_tq_producer",
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
    algorithm_layer_ids_key = _ALGORITHM_TARGET_LAYER_IDS_KEYS.get(config.algorithm)
    if algorithm_layer_ids_key is not None:
        consumer_internal.append(
            f"{algorithm_layer_ids_key}={_hydra_list(config.target_layer_ids)}"
        )
    consumer = [
        python_executable,
        "-m",
        "verl_speco.draft_train_launcher",
        *_replace_internal_overrides(training_args, consumer_internal),
    ]
    return PipelineCommands(
        vllm=vllm,
        vllm_endpoints=config.vllm_endpoints,
        owner=owner,
        producer=producer,
        consumer=consumer,
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
    args, training_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        config = resolve_pipeline_config(training_args)
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
