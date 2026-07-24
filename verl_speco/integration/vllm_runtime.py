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
"""Runtime bridge from SPECO drafter config to upstream verl vLLM rollout.

The upstream verl v0.8.0 rollout config does not know about
``rollout.drafter``.  SPECO therefore injects only the vLLM-native launch
arguments under ``rollout.engine_kwargs.vllm`` before upstream validation, and
keeps draft-only weight publishing as a runtime method on the rollout adapter.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable

from verl_speco.integration.verl_npu_vllm_compat import (
    install_verl_npu_vllm_import_compat,
)
from verl_speco.trainer.checkpoint import trim_process_host_memory

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"
SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV = "VERL_SPECO_VLLM_DRAFT_UPDATE_USE_SHM"
SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS = (
    "verl_speco.integration.vllm_runtime.SpecoVLLMWeightSyncCompatExtension"
)
SPECO_VLLM_WORKER_EXTENSION_CLS = (
    "verl_speco.integration.vllm_runtime.SpecoVLLMColocateWorkerExtension"
)
SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_ENV = (
    "VERL_SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_SECONDS"
)
SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX = "_speco_vllm_spec_decode"
SPECO_VLLM_DRAFT_DIAG_ENV = "VERL_SPECO_VLLM_DRAFT_DIAG"
SPECO_VLLM_NPU_STAGING_ENV = "VERL_SPECO_VLLM_NPU_STAGING"
SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES = 64 << 20

_VLLM_REPLICA_PATCHED = False
_VLLM_DFLASH_CONFIG_ALIASES_PATCHED = False
_VLLM_DSPARK_RUNTIME_PATCHED = False
_VLLM_DSPARK_REGISTRY_ALIAS_PATCHED = False
_NPU_TARGET_STAGING_STATE = threading.local()

_DSPARK_VLLM_ARCHITECTURES = {
    "DSparkDraftModel",
    "Qwen3DSparkModel",
    "DeepSeekDSparkModel",
}
_TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK = (
    "attention",
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",
)


def _speco_is_npu_vllm_worker(worker: Any) -> bool:
    try:
        from vllm.platforms import current_platform

        return str(getattr(current_platform, "device_type", "")).lower() == "npu"
    except Exception:  # noqa: BLE001
        device = getattr(worker, "device", None)
        return str(getattr(device, "type", "")).lower() == "npu"


def _get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _plain_container(value: Any):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        pass

    if isinstance(value, dict):
        return {key: _plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_container(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: _plain_container(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _open_dict_if_needed(config: Any):
    try:
        from omegaconf import OmegaConf, open_dict

        if OmegaConf.is_config(config):
            return open_dict(config)
    except Exception:  # noqa: BLE001
        pass
    return nullcontext()


def _set_child(container: Any, key: str, value: Any) -> None:
    with _open_dict_if_needed(container):
        if hasattr(container, "__setitem__"):
            container[key] = value
        else:
            setattr(container, key, value)


def _has_config_field(config: Any, key: str) -> bool:
    if config is None:
        return False
    if hasattr(config, "get"):
        try:
            return key in config
        except TypeError:
            return config.get(key, None) is not None
    return hasattr(config, key)


def _ensure_child_mapping(container: Any, key: str) -> Any:
    child = _get_nested(container, (key,), None)
    if child is None:
        child = {}
        _set_child(container, key, child)
    return child


def _ensure_nested_mapping(config: Any, path: tuple[str, ...]) -> Any:
    current = config
    for key in path:
        current = _ensure_child_mapping(current, key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            continue
        return value
    return None


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "on", "yes", "y"}:
            return True
        if normalized in {"0", "false", "off", "no", "n", ""}:
            return False
    return bool(value)


def _describe_vllm_draft_logits(draft_logits: Any, *, missing: bool = False) -> str:
    if missing:
        return "missing"
    if draft_logits is None:
        return "None(greedy)"
    shape = getattr(draft_logits, "shape", None)
    if shape is not None:
        try:
            return f"tensor{tuple(shape)}"
        except TypeError:
            return f"shape={shape}"
    return type(draft_logits).__name__


def _resolve_torch_rebuild_func(func: Any):
    if callable(func):
        return func
    if isinstance(func, str):
        from torch.multiprocessing import reductions

        resolved = getattr(reductions, func.rsplit(".", 1)[-1], None)
        if callable(resolved):
            return resolved
    raise TypeError(f"Unsupported IPC rebuild function: {func!r}")


def _speco_rebuild_ipc_compat(handle: tuple[Any, tuple], device_id: int | None = None):
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        if len(list_args) <= 6:
            raise ValueError(
                f"IPC rebuild args do not include a device id slot: len={len(list_args)}"
            )
        list_args[6] = device_id
    return _resolve_torch_rebuild_func(func)(*list_args)


_speco_rebuild_ipc_compat._speco_compat = True


def patch_verl_bucketed_weight_transfer_rebuild_ipc(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Patch verl's bucketed IPC rebuild helper for serialized rebuild names.

    Some environments deserialize the first element of a torch IPC handle as a
    rebuild function name string instead of the callable object. SPECO installs
    this runtime compatibility patch without modifying the vendored verl tree.
    """

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer
        except Exception:  # noqa: BLE001
            return False

    current = getattr(bucketed_weight_transfer, "rebuild_ipc", None)
    if getattr(current, "_speco_compat", False):
        return False
    bucketed_weight_transfer.rebuild_ipc = _speco_rebuild_ipc_compat
    return True


def _speco_persistent_weight_shm_name(zmq_handle: str, bucket_size: int) -> str:
    """Return a job/rank/bucket-scoped shared-memory name.

    ``zmq_handle`` contains the Ray job id, replica id, and local rank in verl.
    Including the bucket size keeps target and draft transfers isolated when
    they use different bucket sizes.
    """

    identity = f"{zmq_handle}\0{int(bucket_size)}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return f"verl_weights_speco_{digest}"


def patch_verl_bucketed_weight_transfer_shm_reuse(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Reuse one stable SHM mapping per verl weight-transfer channel.

    NPU cannot use torch device IPC, so verl falls back to POSIX shared memory.
    Upstream creates a new UUID-named bucket for every actor update. If an NPU
    runtime keeps the old mmap/pinned registration alive after ``close()``, the
    node can retain one full bucket per rank and update. Keeping the mapping
    open and reusing it bounds host memory to one bucket per channel.

    CUDA/device IPC is delegated to the untouched upstream implementation.
    """

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer
        except Exception:  # noqa: BLE001
            return False

    sender_cls = getattr(bucketed_weight_transfer, "BucketedWeightSender", None)
    receiver_cls = getattr(bucketed_weight_transfer, "BucketedWeightReceiver", None)
    if sender_cls is None or receiver_cls is None:
        return False

    sender_init = getattr(sender_cls, "_init_buffer", None)
    receiver_init = getattr(receiver_cls, "_init_buffer", None)
    sender_cleanup = getattr(sender_cls, "_cleanup", None)
    receiver_cleanup = getattr(receiver_cls, "_cleanup", None)
    methods = (sender_init, receiver_init, sender_cleanup, receiver_cleanup)
    if not all(callable(method) for method in methods):
        return False
    if all(getattr(method, "_speco_shm_reuse", False) for method in methods):
        return False

    cache = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_cache", None
    )
    if cache is None:
        cache = {}
        bucketed_weight_transfer._speco_persistent_weight_shm_cache = cache
    owner_names = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_owner_names", None
    )
    if owner_names is None:
        owner_names = set()
        bucketed_weight_transfer._speco_persistent_weight_shm_owner_names = owner_names
    reuse_logged = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_reuse_logged", None
    )
    if reuse_logged is None:
        reuse_logged = set()
        bucketed_weight_transfer._speco_persistent_weight_shm_reuse_logged = (
            reuse_logged
        )
    cache_lock = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_lock", None
    )
    if cache_lock is None:
        cache_lock = threading.Lock()
        bucketed_weight_transfer._speco_persistent_weight_shm_lock = cache_lock

    if not getattr(
        bucketed_weight_transfer,
        "_speco_persistent_weight_shm_cleanup_registered",
        False,
    ):

        def _cleanup_persistent_weight_shm() -> None:
            with cache_lock:
                entries = list(cache.items())
                cache.clear()
                owned = set(owner_names)
                owner_names.clear()

            # Drop torch.frombuffer views before closing their mmap objects.
            shm_entries = [(name, entry[1]) for name, entry in entries]
            entries.clear()
            gc.collect()
            for name, shm in shm_entries:
                try:
                    shm.close()
                except (BufferError, OSError) as exc:
                    logger.warning(
                        "[speco weight shm] close failed name=%s: %s", name, exc
                    )
                if name in owned:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning(
                            "[speco weight shm] unlink failed name=%s: %s", name, exc
                        )

        bucketed_weight_transfer._speco_cleanup_persistent_weight_shm = (
            _cleanup_persistent_weight_shm
        )
        bucketed_weight_transfer._speco_persistent_weight_shm_cleanup_registered = True
        atexit.register(_cleanup_persistent_weight_shm)

    def _get_cached_buffer(shm_name: str, shm_size: int, *, owner: bool):
        with cache_lock:
            cached = cache.get(shm_name)
            if cached is not None:
                buffer, shm = cached
                if int(getattr(shm, "size", shm_size)) < shm_size:
                    raise RuntimeError(
                        f"Persistent weight SHM {shm_name!r} is smaller than requested: "
                        f"{getattr(shm, 'size', None)} < {shm_size}"
                    )
                if owner:
                    owner_names.add(shm_name)
                reuse_key = ("sender" if owner else "receiver", shm_name)
                if reuse_key not in reuse_logged:
                    reuse_logged.add(reuse_key)
                    logger.warning(
                        "[speco weight shm] persistent mapping reused role=%s name=%s "
                        "size_mb=%.1f pid=%s cache_entries=%s",
                        reuse_key[0],
                        shm_name,
                        shm_size / (1 << 20),
                        os.getpid(),
                        len(cache),
                    )
                return buffer, shm

            if owner:
                shm = bucketed_weight_transfer.create_shared_memory(shm_size, shm_name)
                buffer = bucketed_weight_transfer.torch.frombuffer(
                    shm.buf,
                    dtype=bucketed_weight_transfer.torch.uint8,
                )
                owner_names.add(shm_name)
                role = "sender"
            else:
                buffer, shm = bucketed_weight_transfer.rebuild_shared_memory(
                    shm_name,
                    shm_size,
                    dtype=bucketed_weight_transfer.torch.uint8,
                )
                role = "receiver"
            cache[shm_name] = (buffer, shm)
            logger.warning(
                "[speco weight shm] persistent mapping ready role=%s name=%s size_mb=%.1f",
                role,
                shm_name,
                shm_size / (1 << 20),
            )
            return buffer, shm

    def _sender_init_buffer_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return sender_init(self)
        shm_name = _speco_persistent_weight_shm_name(self.zmq_handle, self.bucket_size)
        buffer, shm = _get_cached_buffer(shm_name, self.bucket_size, owner=True)
        self.socket.send_pyobj({"name": shm_name, "size": self.bucket_size})
        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _receiver_init_buffer_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return receiver_init(self)
        comm_metadata = self.socket.recv_pyobj()
        shm_name = comm_metadata["name"]
        shm_size = int(comm_metadata["size"])
        buffer, shm = _get_cached_buffer(shm_name, shm_size, owner=False)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _sender_cleanup_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return sender_cleanup(self)
        # The module-level cache owns the mapping. Let upstream clean sockets
        # and device caches without closing or unlinking the persistent SHM.
        self.buffer = None
        self.shm = None
        return sender_cleanup(self)

    def _receiver_cleanup_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return receiver_cleanup(self)
        self.buffer = None
        self.shm = None
        return receiver_cleanup(self)

    for method in (
        _sender_init_buffer_with_shm_reuse,
        _receiver_init_buffer_with_shm_reuse,
        _sender_cleanup_with_shm_reuse,
        _receiver_cleanup_with_shm_reuse,
    ):
        method._speco_shm_reuse = True

    sender_cls._init_buffer = _sender_init_buffer_with_shm_reuse
    receiver_cls._init_buffer = _receiver_init_buffer_with_shm_reuse
    sender_cls._cleanup = _sender_cleanup_with_shm_reuse
    receiver_cls._cleanup = _receiver_cleanup_with_shm_reuse
    return True


def patch_verl_bucketed_weight_transfer_npu_staging(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Use one reusable NPU allocation for all SHM buckets in a target reload."""

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer
        except Exception:  # noqa: BLE001
            return False

    receiver_cls = getattr(bucketed_weight_transfer, "BucketedWeightReceiver", None)
    if receiver_cls is None:
        return False
    original_receive = getattr(receiver_cls, "receive_weights", None)
    if not callable(original_receive) or getattr(
        original_receive, "_speco_npu_staging", False
    ):
        return False

    logged_pids = getattr(
        bucketed_weight_transfer, "_speco_npu_staging_logged_pids", None
    )
    if logged_pids is None:
        logged_pids = set()
        bucketed_weight_transfer._speco_npu_staging_logged_pids = logged_pids

    def _receive_weights_with_npu_staging(self, on_bucket_received):
        enabled = bool(getattr(_NPU_TARGET_STAGING_STATE, "enabled", False))
        device_type = str(getattr(getattr(self, "device", None), "type", "")).lower()
        if (
            not enabled
            or not bool(getattr(self, "use_shm", False))
            or device_type != "npu"
        ):
            return original_receive(self, on_bucket_received)

        staging_buffer = None
        weights = None
        tensor = None
        metadata = None
        bucket_meta = None
        try:
            self._init_socket()
            self._init_buffer()
            capacity = int(self.buffer.numel())
            staging_buffer = bucketed_weight_transfer.torch.empty(
                capacity,
                dtype=bucketed_weight_transfer.torch.uint8,
                device=self.device,
            )
            if os.getpid() not in logged_pids:
                logged_pids.add(os.getpid())
                logger.warning(
                    "[speco vllm weight sync] using one reusable NPU staging allocation for SHM target reload "
                    "pid=%s capacity_mb=%.1f copy_chunk_mb=%.1f",
                    os.getpid(),
                    capacity / (1 << 20),
                    SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES / (1 << 20),
                )

            while True:
                metadata = self.socket.recv_pyobj()
                bucket_meta = metadata["bucket_meta"]
                used_bytes = max(
                    (
                        int(meta["offset"])
                        + int(meta["dtype"].itemsize * meta["shape"].numel())
                        for meta in bucket_meta.values()
                        if meta["handle"] is None
                    ),
                    default=0,
                )
                if used_bytes > capacity:
                    raise RuntimeError(
                        f"NPU staging bucket overflow: {used_bytes} > {capacity}"
                    )
                if used_bytes:
                    for start in range(
                        0, used_bytes, SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES
                    ):
                        end = min(
                            start + SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES, used_bytes
                        )
                        staging_buffer[start:end].copy_(
                            self.buffer[start:end], non_blocking=False
                        )

                weights = []
                for name, meta in bucket_meta.items():
                    shape = meta["shape"]
                    dtype = meta["dtype"]
                    offset = int(meta["offset"])
                    handle = meta["handle"]
                    if handle is not None:
                        tensor = bucketed_weight_transfer.rebuild_ipc(
                            handle, self.device.index
                        )
                    else:
                        size = int(dtype.itemsize * shape.numel())
                        tensor = (
                            staging_buffer[offset : offset + size]
                            .view(dtype=dtype)
                            .view(shape)
                        )
                    weights.append((name, tensor))

                on_bucket_received(weights)
                bucketed_weight_transfer.get_torch_device().synchronize()
                self.socket.send(b"")
                weights = None
                tensor = None
                if metadata["is_last"]:
                    break
        finally:
            weights = None
            tensor = None
            bucket_meta = None
            metadata = None
            staging_buffer = None
            self._cleanup()

    _receive_weights_with_npu_staging._speco_npu_staging = True
    receiver_cls.receive_weights = _receive_weights_with_npu_staging
    return True


def _speco_npu_target_staging_decision(
    worker: Any,
    *,
    peft_config: dict | None,
    use_shm: bool,
) -> tuple[bool, str]:
    if _bool_or_none(os.getenv(SPECO_VLLM_NPU_STAGING_ENV)) is False:
        return False, "disabled_by_env"
    if not use_shm:
        return False, "not_shm"
    if peft_config is not None:
        return False, "peft"
    if not _speco_is_npu_vllm_worker(worker):
        return False, "not_npu"
    if bool(getattr(worker, "_is_qat_model", False)) or bool(
        getattr(worker, "_is_modelopt_qat", False)
    ):
        return False, "qat"
    use_mtp_sync = getattr(worker, "_use_mtp_drafter_weight_sync", None)
    if callable(use_mtp_sync) and use_mtp_sync():
        return False, "mtp"
    runner = getattr(worker, "model_runner", None)
    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is None:
        return False, "missing_vllm_config"
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is not None:
        return False, f"quantized:{type(quant_config).__name__}"
    return True, "eligible"


def _speco_can_use_npu_target_staging(
    worker: Any, *, peft_config: dict | None, use_shm: bool
) -> bool:
    enabled, _ = _speco_npu_target_staging_decision(
        worker,
        peft_config=peft_config,
        use_shm=use_shm,
    )
    return enabled


@contextmanager
def _speco_npu_target_staging(worker: Any, *, peft_config: dict | None, use_shm: bool):
    previous = bool(getattr(_NPU_TARGET_STAGING_STATE, "enabled", False))
    enabled, reason = _speco_npu_target_staging_decision(
        worker,
        peft_config=peft_config,
        use_shm=use_shm,
    )
    _NPU_TARGET_STAGING_STATE.enabled = enabled
    if not bool(getattr(worker, "_speco_npu_staging_decision_logged", False)):
        worker._speco_npu_staging_decision_logged = True
        print(
            "[speco vllm weight sync] NPU staging decision "
            f"enabled={int(enabled)} reason={reason} pid={os.getpid()} "
            f"local_rank={getattr(worker, 'local_rank', None)} use_shm={int(bool(use_shm))}",
            flush=True,
        )
    try:
        yield
    finally:
        _NPU_TARGET_STAGING_STATE.enabled = previous


def _int_list_or_none(value: Any, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or isinstance(value, dict):
        raise TypeError(f"{field_name} must be a list of integers")
    try:
        return [int(item) for item in value]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a list of integers") from exc
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain only integers") from exc


def _is_dspark_config(config: Any) -> bool:
    architectures = _get_nested(config, ("architectures",), None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture_names = {str(name) for name in architectures}
    return bool(
        _has_config_field(config, "markov_head_type")
        or _has_config_field(config, "dspark_config")
        or architecture_names.intersection(_DSPARK_VLLM_ARCHITECTURES)
    )


def _normalize_dflash_target_layer_aliases(config: Any) -> bool:
    """Mirror DFlash/DSpark target layer ids into vLLM 0.23 alias fields."""

    target_layer_ids = _int_list_or_none(
        _get_nested(config, ("target_layer_ids",), None),
        "target_layer_ids",
    )
    dflash_config = _get_nested(config, ("dflash_config",), None)
    if dflash_config is not None and not hasattr(dflash_config, "get"):
        raise TypeError("DFlash dflash_config must be a mapping when provided")
    nested_target_layer_ids = _int_list_or_none(
        _get_nested(dflash_config, ("target_layer_ids",), None),
        "dflash_config.target_layer_ids",
    )
    dspark_config = _get_nested(config, ("dspark_config",), None)
    if dspark_config is not None and not hasattr(dspark_config, "get"):
        raise TypeError("DSpark dspark_config must be a mapping when provided")
    dspark_target_layer_ids = _int_list_or_none(
        _get_nested(dspark_config, ("target_layer_ids",), None),
        "dspark_config.target_layer_ids",
    )

    if (
        target_layer_ids is not None
        and nested_target_layer_ids is not None
        and target_layer_ids != nested_target_layer_ids
    ):
        raise ValueError(
            "DFlash target_layer_ids conflict with dflash_config.target_layer_ids: "
            f"{target_layer_ids} != {nested_target_layer_ids}"
        )
    if (
        target_layer_ids is not None
        and dspark_target_layer_ids is not None
        and target_layer_ids != dspark_target_layer_ids
    ):
        raise ValueError(
            "DSpark target_layer_ids conflict with dspark_config.target_layer_ids: "
            f"{target_layer_ids} != {dspark_target_layer_ids}"
        )
    if (
        nested_target_layer_ids is not None
        and dspark_target_layer_ids is not None
        and nested_target_layer_ids != dspark_target_layer_ids
    ):
        raise ValueError(
            "DSpark dflash_config.target_layer_ids conflict with dspark_config.target_layer_ids: "
            f"{nested_target_layer_ids} != {dspark_target_layer_ids}"
        )

    selected_layer_ids = _first_present(
        target_layer_ids, nested_target_layer_ids, dspark_target_layer_ids
    )
    if selected_layer_ids is None:
        return False

    changed = False
    if dflash_config is None:
        dflash_config = {}
        _set_child(config, "dflash_config", dflash_config)
        changed = True
    if nested_target_layer_ids is None:
        _set_child(dflash_config, "target_layer_ids", selected_layer_ids)
        changed = True
    if (
        _is_dspark_config(config)
        and _get_nested(dflash_config, ("mask_token_id",), None) is None
        and _get_nested(config, ("mask_token_id",), None) is not None
    ):
        _set_child(
            dflash_config,
            "mask_token_id",
            _get_nested(config, ("mask_token_id",), None),
        )
        changed = True

    expected_aux_layer_ids = [layer_id + 1 for layer_id in selected_layer_ids]
    existing_aux_layer_ids = _int_list_or_none(
        _get_nested(config, ("eagle_aux_hidden_state_layer_ids",), None),
        "eagle_aux_hidden_state_layer_ids",
    )
    if (
        existing_aux_layer_ids is not None
        and existing_aux_layer_ids != expected_aux_layer_ids
    ):
        raise ValueError(
            "DFlash eagle_aux_hidden_state_layer_ids conflict with target_layer_ids: "
            f"{existing_aux_layer_ids} != {expected_aux_layer_ids}"
        )
    if existing_aux_layer_ids is None:
        _set_child(config, "eagle_aux_hidden_state_layer_ids", expected_aux_layer_ids)
        changed = True

    return changed


def _drafter_algorithm(drafter_cfg: dict[str, Any]) -> str:
    return (
        str(drafter_cfg.get("speculative_algorithm", "EAGLE3") or "EAGLE3")
        .strip()
        .upper()
    )


def _validate_vllm_dflash_drafter_config(
    spec_model_path: Any, algorithm: str = "DFLASH"
) -> None:
    if not spec_model_path:
        return

    config_path = os.path.join(os.fspath(spec_model_path), "config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid DFlash drafter config.json at {config_path}: {exc}"
        ) from exc

    architectures = config.get("architectures") or []
    algorithm = str(algorithm or "DFLASH").strip().upper()
    if algorithm == "DSPARK":
        if not _is_dspark_config(config):
            raise ValueError(
                "vLLM DSpark uses the DFlash speculative path, but requires "
                "actor_rollout_ref.rollout.drafter.model_path to point to a DSpark drafter "
                "checkpoint with markov_head_type or a DSpark architecture in config.json; "
                f"got architectures={architectures!r} from {config_path}."
            )
        return

    if architectures and "DFlashDraftModel" not in architectures:
        raise ValueError(
            "vLLM DFlash requires actor_rollout_ref.rollout.drafter.model_path "
            "to point to a DFlash drafter checkpoint with architectures=['DFlashDraftModel']; "
            f"got architectures={architectures!r} from {config_path}. "
            "Do not use an EAGLE/EAGLE3 drafter path with speculative_algorithm=DFLASH."
        )


def _load_env_drafter_config() -> dict[str, Any]:
    raw = os.getenv(SPECO_DRAFTER_CONFIG_ENV)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {SPECO_DRAFTER_CONFIG_ENV}: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def _vllm_drafter_env_payload(drafter_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(drafter_cfg)


def _rollout_name(config: Any) -> str | None:
    return _get_nested(
        config, ("actor_rollout_ref", "rollout", "name"), None
    ) or _get_nested(config, ("rollout", "name"), None)


def _drafter_config_from_config(config: Any) -> dict[str, Any]:
    drafter = _get_nested(config, ("actor_rollout_ref", "rollout", "drafter"), None)
    if drafter is None:
        drafter = _get_nested(config, ("rollout", "drafter"), {})
    drafter = _plain_container(drafter) or {}
    return drafter if isinstance(drafter, dict) else {}


def _rollout_config_from_config(config: Any) -> Any:
    return _get_nested(config, ("actor_rollout_ref", "rollout"), None) or _get_nested(
        config, ("rollout",), None
    )


def _speculative_method_from_drafter(drafter_cfg: dict[str, Any]) -> str:
    algorithm = _drafter_algorithm(drafter_cfg)
    if algorithm == "PEAGLE":
        # P-EAGLE trains against vLLM's parallel-drafting runtime (speculators
        # PR #480). This overlay wires the training path; enabling vLLM rollout
        # requires that runtime, which is not asserted here yet.
        raise ValueError(
            "P-EAGLE vLLM rollout requires the parallel-drafting runtime and is not wired in this overlay yet; "
            "keep actor_rollout_ref.rollout.drafter.enable=false (train the drafter, serve separately)."
        )
    if algorithm == "DOMINO":
        # Domino is a DFlash variant, not an engine-level method: engines expose it as
        # "dflash" and enable the causal correction head (prefix_gru + embed_proj) from
        # the checkpoint's dflash_config.projector_type="domino" (vllm-project/vllm#48241,
        # sgl-project/sglang#31328). DOMINO is never a valid engine algorithm string, so
        # fail loud and point at DFLASH.
        raise ValueError(
            "DOMINO is not an engine-level speculative algorithm; Domino is served as a DFlash "
            "projector sub-mode. Set actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH "
            "for the rollout/serve path; the trained checkpoint's dflash_config.projector_type=domino "
            "enables the Domino correction head on engines that support it, keeping DOMINO for "
            "drafter training."
        )
    if algorithm == "DSPARK":
        return "dflash" if _is_vllm_ascend_runtime_hint() else "dspark"

    method_map = {
        # EAGLE-1 and EAGLE-2 share vLLM's native EAGLE draft (method="eagle");
        # EAGLE-2 is a dynamic-tree decoding policy over the same draft head.
        "EAGLE1": "eagle",
        "EAGLE2": "eagle",
        "EAGLE3": "eagle3",
        "DFLASH": "dflash",
        "DRAFT": "draft_model",
        "DRAFT_MODEL": "draft_model",
        "MTP": "mtp",
    }
    if algorithm not in method_map:
        raise ValueError(
            f"Unsupported SPECO speculative_algorithm for vLLM 0.23: {algorithm}"
        )
    return method_map[algorithm]


def _should_force_eager(drafter_cfg: dict[str, Any]) -> bool:
    explicit = _first_present(
        _get_nested(drafter_cfg, ("vllm", "enforce_eager"), None),
        _get_nested(drafter_cfg, ("vllm", "force_eager"), None),
        _get_nested(drafter_cfg, ("training", "draft_update_enforce_eager"), None),
    )
    return explicit is not None and bool(_bool_or_none(explicit))


# Acceptance settings that break the exact target-distribution guarantee RL relies
# on. greedy DRAFT sampling is lossless (a one-hot proposal fed into rejection
# sampling); only ACCEPTANCE-relaxing knobs are listed here. This is a best-effort
# denylist of the lossy modes we can name confidently, NOT a proof of losslessness --
# it fails closed on the known silent-degradation paths (config overrides via
# drafter.vllm.speculative_config_overrides or engine_kwargs.vllm.speculative_config).
_LOSSY_VLLM_ACCEPTANCE_CHECKS = (
    (
        "acceptance_method",
        lambda v: str(v).strip().lower() == "typical_acceptance_sampler",
        "typical_acceptance_sampler trades exactness for speed",
    ),
    (
        "spec_decoding_acceptance_method",
        lambda v: str(v).strip().lower() == "typical_acceptance_sampler",
        "typical_acceptance_sampler trades exactness for speed",
    ),
    (
        "rejection_sample_method",
        lambda v: str(v).strip().lower() == "synthetic",
        "synthetic acceptance does not sample from the corrected residual distribution",
    ),
    (
        "posterior_threshold",
        lambda v: v is not None and float(v) > 0.0,
        "a nonzero posterior_threshold enables typical/Medusa relaxed acceptance",
    ),
    (
        "posterior_alpha",
        lambda v: v is not None and float(v) > 0.0,
        "a nonzero posterior_alpha enables typical/Medusa relaxed acceptance",
    ),
)


def assert_lossless_vllm_speculative_config(config: Any, *, allow_lossy: bool) -> None:
    """Fail closed on known-lossy vLLM speculative acceptance settings.

    RL rollout under speculative decoding is only unbiased when the verifier samples
    exactly from the target policy. SPECO recomputes PPO's ``old_log_probs`` as the
    target logprob with no importance-sampling correction, so a relaxed acceptance
    method silently miscalibrates the PPO ratio (uncorrected off-policy bias). This
    turns that implicit assumption into an enforced contract for the acceptance modes
    we can name; set ``allow_lossy`` to opt in knowingly.
    """
    if allow_lossy or not isinstance(config, dict):
        return
    offenders = []
    for key, is_lossy, why in _LOSSY_VLLM_ACCEPTANCE_CHECKS:
        if key not in config:
            continue
        try:
            lossy = bool(is_lossy(config[key]))
        except (TypeError, ValueError):
            lossy = False
        if lossy:
            offenders.append(f"{key}={config[key]!r} ({why})")
    if offenders:
        raise ValueError(
            "SPECO refuses a lossy speculative-decoding config that would break the "
            "target-distribution guarantee RL relies on: " + "; ".join(offenders) + ". "
            "The generated tokens would no longer be exactly sampled from the target policy, so "
            "PPO's old_log_probs (recomputed as the target logprob, with no importance-sampling "
            "correction) would be miscalibrated. Set "
            "actor_rollout_ref.rollout.drafter.vllm.allow_lossy_speculative_sampling=true to opt in knowingly."
        )


def build_vllm_speculative_config_from_drafter(
    drafter_cfg: dict[str, Any],
    rollout_cfg: Any = None,
) -> dict[str, Any]:
    """Build a vLLM 0.23 ``speculative_config`` from SPECO drafter config."""

    if not bool(drafter_cfg.get("enable")):
        return {}

    algorithm = _drafter_algorithm(drafter_cfg)
    method = _speculative_method_from_drafter(drafter_cfg)
    spec_model_path = _first_present(
        drafter_cfg.get("model_path"),
        drafter_cfg.get("checkpoint_path"),
        _get_nested(drafter_cfg, ("spec_model", "path"), None),
        _get_nested(drafter_cfg, ("model", "path"), None),
        drafter_cfg.get("spec_model_path"),
    )
    if (
        method in {"eagle", "eagle3", "draft_model", "dflash", "dspark"}
        and spec_model_path is None
    ):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.model_path is required for vLLM speculative decoding"
        )

    rollout_drafter_cfg = drafter_cfg.get("rollout") or {}
    if method in ("dflash", "dspark"):
        if method == "dflash" or algorithm == "DSPARK":
            _validate_vllm_dflash_drafter_config(spec_model_path, algorithm=algorithm)
        num_speculative_tokens = _positive_int_or_none(
            rollout_drafter_cfg.get("spec_verify_tokens")
        )
        if num_speculative_tokens is None:
            raise ValueError(
                "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens "
                f"must be positive for vLLM {method.upper()} speculative decoding"
            )
    else:
        num_speculative_tokens = _positive_int_or_none(
            _first_present(
                rollout_drafter_cfg.get("spec_steps"),
                rollout_drafter_cfg.get("spec_verify_tokens"),
                drafter_cfg.get("num_speculative_tokens"),
            )
        )
        if num_speculative_tokens is None:
            raise ValueError(
                "actor_rollout_ref.rollout.drafter.rollout.spec_steps or spec_verify_tokens "
                "must be positive for vLLM speculative decoding"
            )

    vllm_cfg = drafter_cfg.get("vllm") or {}
    speculative_config: dict[str, Any] = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_sample_method": "greedy",
    }
    if spec_model_path is not None:
        speculative_config["model"] = spec_model_path

    draft_tp = _positive_int_or_none(
        _first_present(
            vllm_cfg.get("draft_tensor_parallel_size"),
            drafter_cfg.get("draft_tensor_parallel_size"),
        )
    )
    if draft_tp is not None:
        speculative_config["draft_tensor_parallel_size"] = draft_tp

    max_model_len = _positive_int_or_none(
        _first_present(vllm_cfg.get("max_model_len"), drafter_cfg.get("max_model_len"))
    )
    if max_model_len is not None:
        speculative_config["max_model_len"] = max_model_len

    if _should_force_eager(drafter_cfg):
        speculative_config["enforce_eager"] = True

    # Keep draft sampling greedy by default. This preserves the NPU/vLLM-Ascend
    # DFlash-family behavior where draft probabilities should not affect
    # rejection sampling; native GPU DSpark can opt in through overrides.

    overrides = vllm_cfg.get("speculative_config_overrides") or {}
    if not isinstance(overrides, dict):
        raise TypeError(
            "drafter.vllm.speculative_config_overrides must be a mapping when provided"
        )
    speculative_config.update(_plain_container(overrides))
    assert_lossless_vllm_speculative_config(
        speculative_config,
        allow_lossy=bool(
            _bool_or_none(vllm_cfg.get("allow_lossy_speculative_sampling", False))
        ),
    )
    return speculative_config


def _merge_speculative_config(
    existing: Any, injected: dict[str, Any]
) -> dict[str, Any]:
    if existing in (None, ""):
        return dict(injected)
    if isinstance(existing, str):
        existing = json.loads(existing)
    existing = _plain_container(existing)
    if not isinstance(existing, dict):
        raise TypeError(
            "rollout.engine_kwargs.vllm.speculative_config must be a mapping for SPECO merge"
        )
    merged = dict(injected)
    merged.update(existing)
    return merged


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_env_or_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _get_vllm_spec_decode_logger():
    try:
        from vllm.logger import init_logger

        spec_logger = init_logger("vllm.spec_decode.acceptance")
    except Exception:  # noqa: BLE001
        spec_logger = logging.getLogger("vllm.spec_decode.acceptance")

    level_name = os.getenv("VLLM_LOGGING_LEVEL")
    if level_name:
        try:
            spec_logger.setLevel(level_name.upper())
        except ValueError:
            pass
    return spec_logger


def _is_vllm_ascend_runtime_hint() -> bool:
    env_hints = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
    )
    if any(os.getenv(name) for name in env_hints):
        return True
    if str(os.getenv("VLLM_TARGET_DEVICE", "")).strip().lower() == "npu":
        return True
    if "ascend" in str(os.getenv("VLLM_PLATFORM", "")).strip().lower():
        return True

    try:
        from verl.utils.device import get_device_name

        if str(get_device_name()).lower() == "npu":
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        from vllm.platforms import current_platform

        return (
            str(
                _first_present(
                    getattr(current_platform, "device_type", None),
                    getattr(current_platform, "device_name", None),
                )
            ).lower()
            == "npu"
        )
    except Exception:  # noqa: BLE001
        return "vllm_ascend" in sys.modules and "torch_npu" in sys.modules


def _maybe_apply_vllm_ascend_global_patch() -> bool:
    patch_transformers_attention_layer_type_constants()
    if not _is_vllm_ascend_runtime_hint():
        return False
    try:
        from vllm_ascend.utils import adapt_patch
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to import vLLM-Ascend global patch hook: %s", exc)
        return False

    try:
        adapt_patch(is_global_patch=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to apply vLLM-Ascend global patch hook: %s", exc)
        return False
    return True


def patch_transformers_attention_layer_type_constants() -> bool:
    """Provide the attention layer type aliases expected by mixed vLLM builds."""

    try:
        from transformers import configuration_utils
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install transformers attention layer type compat: %s", exc
        )
        return False

    has_v5_name = hasattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES")
    has_v4_name = hasattr(configuration_utils, "ALLOWED_LAYER_TYPES")
    if has_v5_name and has_v4_name:
        return False

    existing = None
    if has_v5_name:
        existing = getattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES", None)
    elif has_v4_name:
        existing = getattr(configuration_utils, "ALLOWED_LAYER_TYPES", None)

    try:
        allowed_layer_types = (
            tuple(existing)
            if existing is not None
            else _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK
        )
    except TypeError:
        allowed_layer_types = _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK
    if not allowed_layer_types:
        allowed_layer_types = _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK

    patched_names = []
    if not has_v5_name:
        configuration_utils.ALLOWED_ATTENTION_LAYER_TYPES = allowed_layer_types
        patched_names.append("ALLOWED_ATTENTION_LAYER_TYPES")
    if not has_v4_name:
        configuration_utils.ALLOWED_LAYER_TYPES = allowed_layer_types
        patched_names.append("ALLOWED_LAYER_TYPES")

    logger.warning(
        "[speco vllm compat] patched transformers.configuration_utils missing %s for vLLM import",
        ", ".join(patched_names),
    )
    return True


# Ray imports this module to deserialize SpecoVLLMHttpServer before the normal
# worker runtime hooks run. Install both import guards before any top-level
# verl/vLLM import below, including Worker_TP extension class resolution.
patch_transformers_attention_layer_type_constants()
install_verl_npu_vllm_import_compat()


def _is_dspark_hf_config(hf_config: Any) -> bool:
    architectures = _get_nested(hf_config, ("architectures",), None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture_names = {str(name).replace("DFlash", "", 1) for name in architectures}
    return bool(
        _has_config_field(hf_config, "markov_head_type")
        or _has_config_field(hf_config, "dspark_config")
        or architecture_names.intersection(_DSPARK_VLLM_ARCHITECTURES)
    )


def _ensure_dspark_dflash_aliases(hf_config: Any) -> bool:
    """Make DSpark HF config consumable by vLLM's DFlash draft model."""

    if not _is_dspark_hf_config(hf_config):
        return False

    target_layer_ids = _int_list_or_none(
        _first_present(
            _get_nested(hf_config, ("target_layer_ids",), None),
            _get_nested(hf_config, ("dflash_config", "target_layer_ids"), None),
            _get_nested(hf_config, ("dspark_config", "target_layer_ids"), None),
        ),
        "DSpark target_layer_ids",
    )
    if target_layer_ids is None:
        return False

    changed = False
    dflash_config = _get_nested(hf_config, ("dflash_config",), None)
    if dflash_config is None:
        dflash_config = {}
        _set_child(hf_config, "dflash_config", dflash_config)
        changed = True
    if _get_nested(dflash_config, ("target_layer_ids",), None) is None:
        _set_child(dflash_config, "target_layer_ids", target_layer_ids)
        changed = True
    if _get_nested(dflash_config, ("mask_token_id",), None) is None:
        mask_token_id = _get_nested(hf_config, ("mask_token_id",), None)
        if mask_token_id is not None:
            _set_child(dflash_config, "mask_token_id", mask_token_id)
            changed = True

    if _get_nested(hf_config, ("eagle_aux_hidden_state_layer_ids",), None) is None:
        _set_child(
            hf_config,
            "eagle_aux_hidden_state_layer_ids",
            [layer_id + 1 for layer_id in target_layer_ids],
        )
        changed = True
    return changed


def _dspark_hf_config_from_vllm_config(vllm_config: Any) -> Any:
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    draft_model_cfg = (
        getattr(spec_cfg, "draft_model_config", None) if spec_cfg is not None else None
    )
    return getattr(draft_model_cfg, "hf_config", None)


def _dspark_hf_config_from_proposer(proposer: Any) -> Any:
    draft_model_cfg = getattr(proposer, "draft_model_config", None)
    if draft_model_cfg is not None:
        hf_config = getattr(draft_model_cfg, "hf_config", None)
        if hf_config is not None:
            return hf_config
    spec_cfg = getattr(proposer, "speculative_config", None)
    draft_model_cfg = (
        getattr(spec_cfg, "draft_model_config", None) if spec_cfg is not None else None
    )
    return getattr(draft_model_cfg, "hf_config", None)


def _patch_vllm_dspark_parallel_token() -> bool:
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark parallel-token patch: %s", exc)
        return False

    current = getattr(SpecDecodeBaseProposer, "_init_parallel_drafting_params", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_parallel_token", False):
        return True

    def patched_init_parallel_drafting_params(self):
        model_hf_config = self.draft_model_config.hf_config
        if _is_dspark_hf_config(model_hf_config):
            _ensure_dspark_dflash_aliases(model_hf_config)
            mask_token_id = _get_nested(model_hf_config, ("mask_token_id",), None)
            if mask_token_id is not None:
                self.parallel_drafting_token_id = int(mask_token_id)
                return
        current(self)

    patched_init_parallel_drafting_params._speco_dspark_parallel_token = True
    patched_init_parallel_drafting_params._speco_original_init_parallel_drafting_params = current
    SpecDecodeBaseProposer._init_parallel_drafting_params = (
        patched_init_parallel_drafting_params
    )
    return True


def _patch_vllm_dspark_qwen3_heads() -> bool:
    try:
        import torch
        from torch import nn
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark Qwen3 head patch: %s", exc)
        return False

    class DSparkConfidenceHead(nn.Module):
        def __init__(self, vllm_config: Any, prefix: str) -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            rank = int(
                getattr(
                    config, "markov_rank", getattr(config, "dspark_markov_rank", 256)
                )
            )
            self.proj = ReplicatedLinear(
                config.hidden_size + rank,
                1,
                bias=True,
                params_dtype=torch.float32,
                quant_config=None,
                prefix=f"{prefix}.proj",
            )

        def forward(self, hidden_states: Any, markov_embeds: Any) -> Any:
            x = torch.cat([hidden_states, markov_embeds], dim=-1)
            confidence, _ = self.proj(x.float())
            return confidence.squeeze(-1)

    class DSparkMarkovHead(nn.Module):
        def __init__(self, vllm_config: Any, prefix: str) -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            rank = int(
                getattr(
                    config, "markov_rank", getattr(config, "dspark_markov_rank", 256)
                )
            )
            self.markov_w1 = VocabParallelEmbedding(
                config.vocab_size,
                rank,
                prefix=f"{prefix}.markov_w1",
            )
            self.markov_w2 = ParallelLMHead(
                config.vocab_size,
                rank,
                params_dtype=torch.float32,
                org_num_embeddings=config.vocab_size,
                prefix=f"{prefix}.markov_w2",
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)

        def forward(self, token_ids: Any) -> tuple[Any, Any]:
            embeds = self.markov_w1(token_ids)
            logits = self.logits_processor(
                self.markov_w2,
                embeds.view(-1, embeds.shape[-1]).float(),
            )
            return logits.view(*embeds.shape[:-1], -1), embeds

    current = getattr(DFlashQwen3Model, "__init__", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_qwen3_heads", False):
        return True

    def patched_dflash_qwen3_init(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        prefix = kwargs.get("prefix", "")
        hf_config = _dspark_hf_config_from_vllm_config(vllm_config)
        is_dspark = _is_dspark_hf_config(hf_config)
        if is_dspark:
            _ensure_dspark_dflash_aliases(hf_config)

        current(self, *args, **kwargs)

        if is_dspark:
            if not hasattr(self, "markov_head"):
                self.markov_head = DSparkMarkovHead(
                    vllm_config, prefix=f"{prefix}.markov_head"
                )
            if not hasattr(self, "confidence_head"):
                self.confidence_head = DSparkConfidenceHead(
                    vllm_config, prefix=f"{prefix}.confidence_head"
                )

    patched_dflash_qwen3_init._speco_dspark_qwen3_heads = True
    patched_dflash_qwen3_init._speco_original_dflash_qwen3_init = current
    DFlashQwen3Model.__init__ = patched_dflash_qwen3_init
    return True


def _import_vllm_ascend_dspark_patch() -> bool:
    try:
        import importlib

        importlib.import_module("vllm_ascend.patch.platform.patch_dspark_proposer")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to import vLLM-Ascend DSpark platform patch: %s", exc)
        return False
    return True


def _source_contains_all(obj: Any, markers: tuple[str, ...]) -> bool:
    if obj is None:
        return False
    try:
        import inspect

        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return False
    return all(marker in source for marker in markers)


def _vllm_ascend_has_dspark_pr11153_k_query_runtime() -> bool:
    """Detect the latest PR #11153 DSpark layout in vLLM-Ascend.

    The current PR logic differs from the older fallback in two important ways:
    DSpark uses K query tokens per request instead of DFlash's K+1, and the
    vLLM-Ascend proposer samples the anchor position plus K-1 mask positions.
    """

    try:
        import vllm_ascend.spec_decode.dflash_proposer as dflash_module
        import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
    except Exception:  # noqa: BLE001
        return False

    dflash_cls = getattr(dflash_module, "AscendDflashProposer", None)
    if dflash_cls is None:
        return False

    has_k_query_proposer = _source_contains_all(
        getattr(dflash_cls, "_num_query_per_req", None),
        (
            "self._is_dspark",
            "self.num_speculative_tokens",
            "1 + self.num_speculative_tokens",
        ),
    ) and _source_contains_all(
        getattr(dflash_cls, "set_inputs_first_pass", None),
        ("_num_query_per_req", "IS_DSPARK"),
    )
    if not has_k_query_proposer:
        return False

    for candidate in vars(proposer_module).values():
        if not isinstance(candidate, type) or "Proposer" not in candidate.__name__:
            continue
        method = getattr(candidate, "_run_merged_draft", None)
        if _source_contains_all(
            method,
            (
                "markov_head_type",
                "blk = self.num_speculative_tokens",
                "markov_head",
                "draft_token_ids[:, 1:]",
            ),
        ):
            return True
    return False


def patch_vllm_dspark_runtime() -> bool:
    """Install DSpark hooks for vLLM-Ascend PR #11153's K-query runtime."""

    global _VLLM_DSPARK_RUNTIME_PATCHED

    ascend_has_pr11153_k_query = _vllm_ascend_has_dspark_pr11153_k_query_runtime()
    if not ascend_has_pr11153_k_query:
        logger.debug(
            "vLLM-Ascend DSpark runtime does not match PR #11153's latest K-query "
            "layout; SpeCo will not install legacy DSpark fallback patches."
        )
        return False

    patched = bool(_VLLM_DSPARK_RUNTIME_PATCHED)
    patched = _import_vllm_ascend_dspark_patch() or patched
    patched = _patch_vllm_dspark_parallel_token() or patched
    patched = _patch_vllm_dspark_qwen3_heads() or patched
    if patched:
        _VLLM_DSPARK_RUNTIME_PATCHED = True
    return patched


def _record_vllm_spec_decode_acceptance(
    scheduler: Any,
    *,
    request_id: Any,
    num_draft_tokens: Any,
    num_accepted_tokens: Any,
    num_invalid_spec_tokens: Any,
) -> None:
    if not getattr(scheduler, "log_stats", True):
        return

    del request_id, num_invalid_spec_tokens
    draft_tokens = _int_or_zero(num_draft_tokens)
    accepted = max(0, _int_or_zero(num_accepted_tokens))
    if draft_tokens <= 0:
        return

    total_drafts = int(getattr(scheduler, "_speco_spec_decode_log_drafts", 0)) + 1
    total_accepted = (
        int(getattr(scheduler, "_speco_spec_decode_log_accepted", 0)) + accepted
    )
    scheduler._speco_spec_decode_log_drafts = total_drafts
    scheduler._speco_spec_decode_log_accepted = total_accepted

    now = time.monotonic()
    last_log_time = float(getattr(scheduler, "_speco_spec_decode_last_log_time", 0.0))
    interval = _float_env_or_default(SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_ENV, 10.0)
    if last_log_time > 0.0 and interval > 0.0 and now - last_log_time < interval:
        return

    spec_logger = getattr(scheduler, "_speco_spec_decode_logger", None)
    if spec_logger is None:
        spec_logger = _get_vllm_spec_decode_logger()
        scheduler._speco_spec_decode_logger = spec_logger

    spec_logger.info(
        "[speco vllm spec decode] mean_acceptance_length=%.3f",
        1.0 + total_accepted / max(1, total_drafts),
    )
    scheduler._speco_spec_decode_last_log_time = now
    scheduler._speco_spec_decode_log_drafts = 0
    scheduler._speco_spec_decode_log_accepted = 0


def patch_vllm_spec_decode_acceptance_logging() -> bool:
    """Lightly restore vLLM speculative acceptance logging as INFO logs."""

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install vLLM spec decode acceptance logging patch: %s", exc
        )
        return False

    original = getattr(Scheduler, "make_spec_decoding_stats", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_spec_decode_acceptance_logging", False):
        return True

    def patched_make_spec_decoding_stats(
        self,
        spec_decoding_stats,
        num_draft_tokens,
        num_accepted_tokens,
        num_invalid_spec_tokens,
        request_id,
    ):
        result = original(
            self,
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )
        try:
            _record_vllm_spec_decode_acceptance(
                self,
                request_id=request_id,
                num_draft_tokens=num_draft_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_invalid_spec_tokens=num_invalid_spec_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log vLLM spec decode acceptance stats: %s", exc)
        return result

    patched_make_spec_decoding_stats._speco_spec_decode_acceptance_logging = True
    patched_make_spec_decoding_stats._speco_original_make_spec_decoding_stats = original
    Scheduler.make_spec_decoding_stats = patched_make_spec_decoding_stats
    return True


def patch_vllm_dflash_config_aliases() -> bool:
    """Let vLLM 0.23 consume SPECO DFlash top-level target layer ids."""

    global _VLLM_DFLASH_CONFIG_ALIASES_PATCHED
    if _VLLM_DFLASH_CONFIG_ALIASES_PATCHED:
        return True
    try:
        from vllm.transformers_utils.configs.eagle import EAGLEConfig
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DFlash config alias patch: %s", exc)
        return False

    current = getattr(EAGLEConfig, "__init__", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dflash_config_aliases", False):
        _VLLM_DFLASH_CONFIG_ALIASES_PATCHED = True
        return True

    def patched_eagle_config_init(self, *args, **kwargs):
        method = kwargs.get("method", None)
        if method is None and len(args) >= 3:
            method = args[2]
        current(self, *args, **kwargs)
        if str(method or "eagle").strip().lower() == "dflash":
            _normalize_dflash_target_layer_aliases(self)
            if _is_dspark_hf_config(self):
                _set_child(self, "architectures", ["DFlashDraftModel"])

    patched_eagle_config_init._speco_dflash_config_aliases = True
    patched_eagle_config_init._speco_original_eagle_config_init = current
    EAGLEConfig.__init__ = patched_eagle_config_init
    _VLLM_DFLASH_CONFIG_ALIASES_PATCHED = True
    return True


def patch_vllm_dspark_registry_aliases() -> bool:
    """Let vLLM resolve DSpark draft architectures through the DFlash model."""

    global _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED
    if _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED:
        return True
    try:
        from vllm.model_executor.models import registry
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark registry aliases: %s", exc)
        return False

    model_registry = getattr(registry, "ModelRegistry", None)
    register_model = getattr(model_registry, "register_model", None)
    if not callable(register_model):
        return False

    existing_models = getattr(model_registry, "models", {})
    for architecture in sorted(
        _DSPARK_VLLM_ARCHITECTURES
        | {"DFlashDSparkDraftModel", "DFlashQwen3DSparkModel"}
    ):
        if architecture not in existing_models:
            register_model(
                architecture,
                "vllm.model_executor.models.qwen3_dflash:DFlashQwen3ForCausalLM",
            )
    _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED = True
    return True


def _speco_vllm_run_engine_core_with_acceptance_logging(*args, **kwargs):
    _maybe_apply_vllm_ascend_global_patch()
    patch_vllm_dflash_config_aliases()
    patch_vllm_dspark_registry_aliases()
    patch_vllm_dspark_runtime()
    patch_vllm_spec_decode_acceptance_logging()
    patch_vllm_worker_proc_entrypoint()

    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(EngineCoreProc, "_speco_original_run_engine_core", None)
    if original is None:
        original = EngineCoreProc.run_engine_core
    return original(*args, **kwargs)


def patch_vllm_engine_core_entrypoint() -> bool:
    """Install the scheduler logging patch inside vLLM EngineCore subprocesses."""

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM EngineCore entrypoint patch: %s", exc)
        return False

    current = getattr(EngineCoreProc, "run_engine_core", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_engine_core_acceptance_logging", False):
        return True

    EngineCoreProc._speco_original_run_engine_core = current
    _speco_vllm_run_engine_core_with_acceptance_logging._speco_engine_core_acceptance_logging = True
    EngineCoreProc.run_engine_core = staticmethod(
        _speco_vllm_run_engine_core_with_acceptance_logging
    )
    return True


def _speco_vllm_worker_main_with_runtime_observability(*args, **kwargs):
    _maybe_apply_vllm_ascend_global_patch()
    patch_vllm_dflash_config_aliases()
    patch_vllm_dspark_registry_aliases()
    patch_vllm_dspark_runtime()
    patch_vllm_spec_decode_acceptance_logging()

    original = getattr(
        _speco_vllm_worker_main_with_runtime_observability,
        "_speco_original_worker_main",
        None,
    )
    if not callable(original):
        from vllm.v1.executor import multiproc_executor

        original = multiproc_executor.WorkerProc.worker_main
    return original(*args, **kwargs)


def patch_vllm_worker_proc_entrypoint() -> bool:
    """Install runtime patches inside vLLM worker subprocesses."""

    if not _maybe_apply_vllm_ascend_global_patch():
        return False
    try:
        from vllm.v1.executor import multiproc_executor
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM WorkerProc entrypoint patch: %s", exc)
        return False

    WorkerProc = getattr(multiproc_executor, "WorkerProc", None)
    if WorkerProc is None:
        return False
    current = getattr(WorkerProc, "worker_main", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_worker_proc_runtime_observability", False):
        return True

    WorkerProc._speco_original_worker_main = current
    _speco_vllm_worker_main_with_runtime_observability._speco_worker_proc_runtime_observability = True
    _speco_vllm_worker_main_with_runtime_observability._speco_original_worker_main = (
        current
    )
    WorkerProc.worker_main = staticmethod(
        _speco_vllm_worker_main_with_runtime_observability
    )
    return True


def install_vllm_spec_decode_acceptance_logging() -> bool:
    """Install speculative acceptance logs in parent and EngineCore processes."""

    scheduler_patched = patch_vllm_spec_decode_acceptance_logging()
    engine_core_patched = patch_vllm_engine_core_entrypoint()
    return scheduler_patched or engine_core_patched


def install_vllm_runtime_observability() -> bool:
    """Install lightweight vLLM logging hooks for all rollout modes."""

    _maybe_apply_vllm_ascend_global_patch()
    dflash_config_patched = patch_vllm_dflash_config_aliases()
    dspark_registry_patched = patch_vllm_dspark_registry_aliases()
    dspark_runtime_patched = patch_vllm_dspark_runtime()
    acceptance_patched = install_vllm_spec_decode_acceptance_logging()
    worker_proc_patched = patch_vllm_worker_proc_entrypoint()
    return (
        dflash_config_patched
        or dspark_registry_patched
        or dspark_runtime_patched
        or acceptance_patched
        or worker_proc_patched
    )


def _new_vllm_spec_decode_stats() -> dict[str, float]:
    return {
        "drafts": 0,
        "accepted_tokens": 0,
    }


def _record_vllm_spec_decode_scheduler_stats(
    target: dict[str, float], scheduler_stats: Any
) -> None:
    spec_stats = getattr(scheduler_stats, "spec_decoding_stats", None)
    if spec_stats is None:
        return
    drafts = _int_or_zero(getattr(spec_stats, "num_drafts", 0))
    accepted_tokens = _int_or_zero(getattr(spec_stats, "num_accepted_tokens", 0))
    if drafts <= 0 and accepted_tokens <= 0:
        return
    target["drafts"] += drafts
    target["accepted_tokens"] += accepted_tokens


def _vllm_spec_decode_stats_to_metrics(stats: dict[str, float]) -> dict[str, float]:
    drafts = float(stats.get("drafts", 0.0) or 0.0)
    accepted_tokens = float(stats.get("accepted_tokens", 0.0) or 0.0)
    return {
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": drafts,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": accepted_tokens,
    }


def _build_speco_vllm_stat_logger(server: Any):
    from vllm.v1.metrics.loggers import StatLoggerBase

    class SpecoVLLMSpecDecodeStatLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0):
            del vllm_config
            self.engine_index = engine_index

        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx: int = 0,
        ):
            del iteration_stats, mm_cache_stats
            stats = getattr(server, "_speco_vllm_spec_decode_pending_stats", None)
            if not isinstance(stats, dict):
                stats = _new_vllm_spec_decode_stats()
                server._speco_vllm_spec_decode_pending_stats = stats
            _record_vllm_spec_decode_scheduler_stats(stats, scheduler_stats)

        def log_engine_initialized(self):
            return None

    SpecoVLLMSpecDecodeStatLogger.__module__ = __name__
    return SpecoVLLMSpecDecodeStatLogger


def _ensure_vllm_drafter_speculative_config_from_env(rollout_cfg: Any) -> None:
    drafter_cfg = _load_env_drafter_config()
    if not bool(drafter_cfg.get("enable")):
        return

    speculative_config = build_vllm_speculative_config_from_drafter(
        drafter_cfg, rollout_cfg=rollout_cfg
    )
    engine_kwargs_root = _ensure_child_mapping(rollout_cfg, "engine_kwargs")
    engine_kwargs = _ensure_child_mapping(engine_kwargs_root, "vllm")
    existing_spec = _get_nested(engine_kwargs, ("speculative_config",), None)
    merged_speculative_config = _merge_speculative_config(
        existing_spec, speculative_config
    )
    # Authoritative check: engine_kwargs.vllm.speculative_config (existing_spec) takes
    # priority in the merge, so a lossy acceptance mode injected there must be caught here.
    assert_lossless_vllm_speculative_config(
        merged_speculative_config,
        allow_lossy=bool(
            _bool_or_none(
                _get_nested(
                    drafter_cfg, ("vllm", "allow_lossy_speculative_sampling"), False
                )
            )
        ),
    )
    _set_child(engine_kwargs, "speculative_config", merged_speculative_config)
    if bool(merged_speculative_config.get("enforce_eager")):
        _set_child(engine_kwargs, "enforce_eager", True)


class _SpecoVLLMHttpServerMixin:
    def _speco_pop_vllm_spec_decode_stats(self) -> dict[str, float]:
        stats = getattr(self, "_speco_vllm_spec_decode_pending_stats", None)
        if not isinstance(stats, dict):
            return _new_vllm_spec_decode_stats()
        snapshot = dict(stats)
        self._speco_vllm_spec_decode_pending_stats = _new_vllm_spec_decode_stats()
        return snapshot

    def _speco_add_vllm_spec_decode_extra_fields(
        self, extra_fields: dict[str, Any]
    ) -> None:
        stats = self._speco_pop_vllm_spec_decode_stats()
        extra_fields.update(_vllm_spec_decode_stats_to_metrics(stats))

    async def launch_server(self, *args, **kwargs):
        self._speco_vllm_spec_decode_pending_stats = _new_vllm_spec_decode_stats()
        install_vllm_runtime_observability()
        _ensure_vllm_drafter_speculative_config_from_env(self.config)
        return await super().launch_server(*args, **kwargs)

    async def run_server(self, args):
        try:
            import inspect

            from vllm.v1.engine.async_llm import AsyncLLM
        except Exception:  # noqa: BLE001
            return await super().run_server(args)

        original_from_vllm_config_attr = inspect.getattr_static(
            AsyncLLM, "from_vllm_config"
        )
        original_from_vllm_config = AsyncLLM.from_vllm_config
        try:
            original_signature = inspect.signature(
                original_from_vllm_config_attr.__func__
            )
        except (AttributeError, TypeError, ValueError):
            original_signature = None

        def from_vllm_config_with_speco_stats(cls, *call_args, **call_kwargs):
            del cls
            install_vllm_runtime_observability()
            stat_loggers = list(call_kwargs.get("stat_loggers") or [])
            stat_loggers.append(_build_speco_vllm_stat_logger(self))
            call_kwargs["stat_loggers"] = stat_loggers
            return original_from_vllm_config(*call_args, **call_kwargs)

        if original_signature is not None:
            from_vllm_config_with_speco_stats.__signature__ = original_signature
        AsyncLLM.from_vllm_config = classmethod(from_vllm_config_with_speco_stats)
        try:
            return await super().run_server(args)
        finally:
            AsyncLLM.from_vllm_config = original_from_vllm_config_attr

    async def generate(self, *args, **kwargs):
        output = await super().generate(*args, **kwargs)
        extra_fields = getattr(output, "extra_fields", None)
        if isinstance(extra_fields, dict):
            self._speco_add_vllm_spec_decode_extra_fields(extra_fields)
        return output


def _build_speco_vllm_http_server_class(upstream_module: Any):
    upstream_cls = upstream_module.vLLMHttpServer
    if issubclass(upstream_cls, _SpecoVLLMHttpServerMixin):
        return upstream_cls
    return type(
        "SpecoVLLMHttpServer",
        (_SpecoVLLMHttpServerMixin, upstream_cls),
        {"__module__": __name__},
    )


def install_upstream_vllm_runtime_bridge() -> bool:
    """Patch upstream verl vLLM rollout classes in the current process."""

    global _VLLM_REPLICA_PATCHED
    install_vllm_runtime_observability()
    if _VLLM_REPLICA_PATCHED:
        return True

    try:
        import ray

        from verl.workers.rollout import replica as replica_module
        from verl.workers.rollout.vllm_rollout import vllm_async_server
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install SPECO vLLM runtime bridge: %s", exc)
        return False

    upstream_replica = getattr(vllm_async_server, "vLLMReplica", None)
    if upstream_replica is None:
        return False

    speco_http_server_cls = _build_speco_vllm_http_server_class(vllm_async_server)

    class SpecoVLLMReplica(upstream_replica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(speco_http_server_cls)

    SpecoVLLMReplica.__module__ = __name__
    vllm_async_server.vLLMReplica = SpecoVLLMReplica
    registry = getattr(replica_module, "RolloutReplicaRegistry", None)
    if registry is not None and hasattr(registry, "_registry"):
        registry._registry["vllm"] = lambda: SpecoVLLMReplica
    patch_vllm_server_adapter_update()
    _VLLM_REPLICA_PATCHED = True
    return True


def configure_vllm_runtime_from_config(config: Any) -> dict[str, Any]:
    """Inject vLLM launch kwargs needed by SPECO online drafter rollout."""

    if _rollout_name(config) != "vllm":
        return {}
    drafter_cfg = _drafter_config_from_config(config)
    enabled = bool(drafter_cfg.get("enable"))
    if not enabled:
        os.environ.pop(SPECO_DRAFTER_CONFIG_ENV, None)
        return {}

    os.environ[SPECO_DRAFTER_CONFIG_ENV] = json.dumps(
        _vllm_drafter_env_payload(drafter_cfg), sort_keys=True
    )
    rollout_cfg = _rollout_config_from_config(config)
    speculative_config = build_vllm_speculative_config_from_drafter(
        drafter_cfg, rollout_cfg=rollout_cfg
    )
    install_upstream_vllm_runtime_bridge()

    engine_kwargs = _ensure_nested_mapping(
        config, ("actor_rollout_ref", "rollout", "engine_kwargs", "vllm")
    )
    existing_spec = _get_nested(engine_kwargs, ("speculative_config",), None)
    merged_speculative_config = _merge_speculative_config(
        existing_spec, speculative_config
    )
    assert_lossless_vllm_speculative_config(
        merged_speculative_config,
        allow_lossy=bool(
            _bool_or_none(
                _get_nested(
                    drafter_cfg, ("vllm", "allow_lossy_speculative_sampling"), False
                )
            )
        ),
    )
    _set_child(engine_kwargs, "speculative_config", merged_speculative_config)
    if bool(drafter_cfg.get("enable")):
        _set_child(
            engine_kwargs, "worker_extension_cls", SPECO_VLLM_WORKER_EXTENSION_CLS
        )
    if bool(merged_speculative_config.get("enforce_eager")):
        _set_child(engine_kwargs, "enforce_eager", True)
    return speculative_config


def _draft_zmq_handle_from_base(zmq_handle: str) -> str:
    if zmq_handle.endswith(".sock"):
        return f"{zmq_handle[:-5]}-draft.sock"
    return f"{zmq_handle}-draft"


def _named_weight_iter(weights: Any) -> Iterable[tuple[str, Any]]:
    if hasattr(weights, "items"):
        return weights.items()
    return weights


def _resolve_vllm_draft_update_use_shm(
    adapter: Any, training_cfg: dict[str, Any]
) -> bool:
    env_forced = _bool_or_none(os.getenv(SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV))
    if env_forced is not None:
        return env_forced
    forced = _bool_or_none(training_cfg.get("draft_update_use_shm", None))
    if forced is not None:
        return forced
    config_forced = _bool_or_none(
        _first_present(
            _get_nested(
                getattr(adapter, "config", None),
                ("drafter", "training", "draft_update_use_shm"),
                None,
            ),
            _get_nested(
                getattr(adapter, "config", None),
                ("rollout", "drafter", "training", "draft_update_use_shm"),
                None,
            ),
            _get_nested(
                getattr(adapter, "config", None),
                (
                    "actor_rollout_ref",
                    "rollout",
                    "drafter",
                    "training",
                    "draft_update_use_shm",
                ),
                None,
            ),
        )
    )
    if config_forced is not None:
        return config_forced
    return bool(getattr(adapter, "use_shm", False)) or _is_vllm_ascend_runtime_hint()


def _draft_param_name_candidates(name: str) -> list[str]:
    prefixes = ("module.", "_orig_mod.", "draft_model.", "model.draft_model.")
    bases = []
    pending = [name]
    while pending:
        candidate = pending.pop(0)
        if candidate in bases:
            continue
        bases.append(candidate)
        for prefix in prefixes:
            if candidate.startswith(prefix):
                pending.append(candidate[len(prefix) :])

    candidates = []
    for candidate in bases:
        candidates.append(candidate)
        if "midlayer." in candidate:
            candidates.append(candidate.replace("midlayer.", "model.layers.0."))
    for candidate in list(candidates):
        if not candidate.startswith("model."):
            candidates.append(f"model.{candidate}")
    return list(dict.fromkeys(candidates))


def _draft_fused_param_candidates(name: str) -> list[tuple[str, Any]]:
    mappings = (
        (".qkv_proj.", ".q_proj.", "q"),
        (".qkv_proj.", ".k_proj.", "k"),
        (".qkv_proj.", ".v_proj.", "v"),
        (".gate_up_proj.", ".gate_proj.", 0),
        (".gate_up_proj.", ".up_proj.", 1),
    )
    candidates = []
    for candidate in _draft_param_name_candidates(name):
        for fused_name, shard_name, shard_id in mappings:
            if shard_name in candidate:
                candidates.append((candidate.replace(shard_name, fused_name), shard_id))
    return list(dict.fromkeys(candidates))


def _load_draft_param(param: Any, tensor: Any, shard_id: Any = None) -> None:
    param_data = getattr(param, "data", None)
    device = getattr(param, "device", getattr(param_data, "device", None))
    dtype = getattr(param, "dtype", getattr(param_data, "dtype", None))
    tensor = tensor.to(device=device, dtype=dtype)
    weight_loader = getattr(param, "weight_loader", None)
    if callable(weight_loader):
        if shard_id is None:
            weight_loader(param, tensor)
        else:
            weight_loader(param, tensor, shard_id)
        return
    param.data.copy_(tensor, non_blocking=True)


def _ensure_vllm_server_handle(adapter: Any) -> None:
    if (
        getattr(adapter, "rollout_rank", None) != 0
        or getattr(adapter, "server_handle", None) is not None
    ):
        return
    import ray

    prefix = adapter._get_server_name_prefix()
    adapter.server_handle = ray.get_actor(
        f"{prefix}server_{adapter.replica_rank}_{adapter.node_rank}"
    )


async def _maybe_call_vllm_server_method(
    adapter: Any, method_name: str, *args, **kwargs
) -> Any:
    if getattr(adapter, "rollout_rank", None) != 0:
        return None
    _ensure_vllm_server_handle(adapter)
    method = getattr(adapter.server_handle, method_name, None)
    if method is None or not hasattr(method, "remote"):
        return None
    return await method.remote(*args, **kwargs)


async def speco_vllm_update_draft_weights(
    self, weights: Any, *args, global_steps: int = None, **kwargs
):
    """Update only vLLM draft/speculative model weights from a ServerAdapter."""

    del args
    if not weights:
        return

    drafter_cfg = _load_env_drafter_config()
    training_cfg = drafter_cfg.get("training") or {}
    bucket_mb = _positive_int_or_none(
        training_cfg.get("draft_update_weights_bucket_megabytes")
    )
    if bucket_mb is None:
        bucket_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
    pause_generation = bool(training_cfg.get("draft_update_pause_generation", True))
    flush_before = bool(training_cfg.get("draft_update_flush_before", True))
    flush_after = bool(training_cfg.get("draft_update_flush_after", True))
    generation_paused = False
    use_shm = _resolve_vllm_draft_update_use_shm(self, training_cfg)
    if (
        getattr(self, "replica_rank", -1) == 0
        and getattr(self, "rollout_rank", -1) == 0
    ):
        logger.warning(
            "[speco vllm draft update] starting global_steps=%s transfer=%s env_%s=%r cfg_draft_update_use_shm=%r adapter_use_shm=%r",
            global_steps,
            "shm" if use_shm else "ipc",
            SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV,
            os.getenv(SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV),
            training_cfg.get("draft_update_use_shm", None),
            getattr(self, "use_shm", None),
        )

    patch_verl_bucketed_weight_transfer_shm_reuse()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        BucketedWeightSender,
    )

    start_time = time.time()
    try:
        if self.rollout_rank == 0 and pause_generation:
            await _maybe_call_vllm_server_method(
                self, "abort_all_requests", reset_prefix_cache=flush_before
            )
            generation_paused = True
        elif self.rollout_rank == 0 and flush_before:
            await _maybe_call_vllm_server_method(self, "clear_kv_cache")

        future = await self._execute_method(
            "update_draft_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": use_shm},
        )

        sender = BucketedWeightSender(
            zmq_handle=_draft_zmq_handle_from_base(self.zmq_handle),
            bucket_size_mb=int(bucket_mb),
            use_shm=use_shm,
        )
        await sender.async_send_weights(_named_weight_iter(weights))

        if future is not None:
            await future

        if self.rollout_rank == 0:
            if flush_after:
                await _maybe_call_vllm_server_method(self, "clear_kv_cache")
            if global_steps is not None:
                await _maybe_call_vllm_server_method(
                    self, "set_global_steps", global_steps
                )

        if (
            getattr(self, "replica_rank", -1) == 0
            and getattr(self, "rollout_rank", -1) == 0
        ):
            logger.warning(
                "[speco vllm draft update] done global_steps=%s transfer=%s bucket_mb=%s elapsed_sec=%.3f",
                global_steps,
                "shm" if use_shm else "ipc",
                bucket_mb,
                time.time() - start_time,
            )
    finally:
        if generation_paused:
            await _maybe_call_vllm_server_method(self, "resume_generation")


def attach_update_draft_weights_to_rollout(rollout: Any) -> Any:
    """Attach ``update_draft_weights`` to an upstream vLLM ServerAdapter."""

    if rollout is not None and not callable(
        getattr(rollout, "update_draft_weights", None)
    ):
        rollout.update_draft_weights = speco_vllm_update_draft_weights.__get__(
            rollout, type(rollout)
        )
    return rollout


def patch_vllm_server_adapter_update() -> None:
    patch_verl_bucketed_weight_transfer_rebuild_ipc()
    patch_verl_bucketed_weight_transfer_shm_reuse()
    try:
        from verl.workers.rollout.vllm_rollout import vllm_rollout
    except Exception:  # noqa: BLE001
        return

    server_adapter = getattr(vllm_rollout, "ServerAdapter", None)
    if server_adapter is not None and not callable(
        getattr(server_adapter, "update_draft_weights", None)
    ):
        server_adapter.update_draft_weights = speco_vllm_update_draft_weights


def install_vllm_runtime_for_worker(worker: Any) -> None:
    """Install SPECO vLLM runtime hooks inside an actor-rollout worker process."""

    drafter_env = getattr(type(worker), "_speco_sglang_drafter_config_env", None)
    if drafter_env:
        os.environ[SPECO_DRAFTER_CONFIG_ENV] = drafter_env
    install_vllm_runtime_observability()
    patch_verl_bucketed_weight_transfer_rebuild_ipc()
    patch_verl_bucketed_weight_transfer_shm_reuse()
    patch_verl_bucketed_weight_transfer_npu_staging()
    patch_vllm_server_adapter_update()


try:
    from verl.workers.rollout.vllm_rollout.utils import (
        vLLMColocateWorkerExtension as _VLLMWorkerExtensionBase,
    )
except Exception:  # noqa: BLE001
    _VLLMWorkerExtensionBase = object


class SpecoVLLMWeightSyncCompatExtension(_VLLMWorkerExtensionBase):
    """Install the serialized NPU IPC-handle compatibility before target weight sync."""

    def update_weights_from_ipc(
        self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False
    ):
        patched = patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        patch_verl_bucketed_weight_transfer_npu_staging()
        if patched and int(getattr(self, "local_rank", 0) or 0) == 0:
            logger.warning(
                "[speco vllm weight sync] installed IPC rebuild compatibility"
            )
        if not _speco_is_npu_vllm_worker(self):
            return super().update_weights_from_ipc(
                peft_config=peft_config,
                base_sync_done=base_sync_done,
                use_shm=use_shm,
            )

        try:
            with _speco_npu_target_staging(
                self, peft_config=peft_config, use_shm=use_shm
            ):
                return super().update_weights_from_ipc(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                    use_shm=use_shm,
                )
        finally:
            trim_process_host_memory()


class SpecoVLLMColocateWorkerExtension(_VLLMWorkerExtensionBase):
    """vLLM worker extension that can update only the speculative draft model."""

    def __new__(cls, **kwargs):
        try:
            instance = super().__new__(cls, **kwargs)
        except TypeError:
            instance = super().__new__(cls)
        # vLLM's extension mechanism forbids overriding methods that already
        # exist on Worker (e.g. wake_up). Use __new__ (dunder, skipped by the
        # conflict check) to install an instance-level wrapper instead.
        # Python resolves instance attributes before class methods.
        _orig_wake_up = getattr(type(instance), "wake_up", None)
        if not callable(_orig_wake_up):
            return instance

        def _speco_wake_up_hook(*args, **kwargs):
            result = _orig_wake_up(instance, *args, **kwargs)
            reloaded = instance._speco_reload_draft_from_checkpoint()
            if reloaded > 0:
                logger.warning(
                    "[speco draft wake_up] drafter weights restored after wake_up (%d tensors)",
                    reloaded,
                )
            return result

        instance.wake_up = _speco_wake_up_hook
        return instance

    def _get_speco_draft_zmq_handle(self) -> str:
        get_base = getattr(self, "_get_zmq_handle", None)
        if callable(get_base):
            return _draft_zmq_handle_from_base(get_base())
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{self.local_rank}-draft.sock"

    def _speco_resolve_draft_proposer(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return None
        for attr in ("drafter", "speculator"):
            proposer = getattr(runner, attr, None)
            if proposer is not None:
                return proposer
        return None

    def _speco_resolve_draft_model(self):
        proposer = self._speco_resolve_draft_proposer()
        if proposer is None:
            return None, None
        get_model = getattr(proposer, "get_model", None)
        if callable(get_model):
            try:
                return get_model(), proposer
            except AttributeError:
                return None, proposer
        model = getattr(proposer, "model", None)
        if model is not None:
            return model, proposer
        return None, proposer

    def _speco_draft_method(self) -> str:
        proposer = self._speco_resolve_draft_proposer()
        spec_cfg = (
            getattr(proposer, "speculative_config", None)
            if proposer is not None
            else None
        )
        method = str(getattr(spec_cfg, "method", "") or "").strip().lower()
        if method:
            return method
        draft_model, _ = self._speco_resolve_draft_model()
        model_type = (
            type(draft_model).__name__.lower() if draft_model is not None else ""
        )
        if "dflash" in model_type:
            return "dflash"
        if "eagle3" in model_type:
            return "eagle3"
        return method

    def _speco_is_dflash_draft(self) -> bool:
        return self._speco_draft_method() in ("dflash", "dspark")

    def _speco_update_draft_weights(self, weights: list[tuple[str, Any]]) -> int:
        draft_model, proposer = self._speco_resolve_draft_model()
        if draft_model is None:
            return 0
        del proposer
        named_parameters = getattr(draft_model, "named_parameters", None)
        if not callable(named_parameters):
            raise RuntimeError(
                "Resolved vLLM draft model does not expose named_parameters() for graph-safe update"
            )

        named_params = dict(named_parameters())
        updated = 0
        missing = []
        incompatible = []
        for name, tensor in weights:
            param = None
            matched_name = None
            for candidate in _draft_param_name_candidates(str(name)):
                param = named_params.get(candidate)
                if param is not None:
                    matched_name = candidate
                    break
            shard_id = None
            if param is None:
                for candidate, candidate_shard_id in _draft_fused_param_candidates(
                    str(name)
                ):
                    param = named_params.get(candidate)
                    if param is not None:
                        matched_name = candidate
                        shard_id = candidate_shard_id
                        break
                if param is None:
                    missing.append(name)
                    continue
            if (
                shard_id is None
                and not callable(getattr(param, "weight_loader", None))
                and tuple(param.shape) != tuple(tensor.shape)
            ):
                incompatible.append(
                    f"{name}->{matched_name}: expected {tuple(param.shape)}, got {tuple(tensor.shape)}"
                )
                continue
            try:
                _load_draft_param(param, tensor, shard_id=shard_id)
            except Exception as exc:  # noqa: BLE001
                incompatible.append(
                    f"{name}->{matched_name}: loader failed for shape {tuple(tensor.shape)}: {exc}"
                )
                continue
            updated += 1

        if missing or incompatible:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}")
            if incompatible:
                details.append(f"incompatible={incompatible[:8]}")
            raise RuntimeError(
                "SPECO vLLM graph-safe draft update could not load all weights: "
                + "; ".join(details)
            )
        return updated

    def update_draft_weights_from_ipc(self, use_shm: bool = False):
        """Receive and load draft-model weights through the verl bucketed IPC path.

        Uses draft_model.load_weights() (the same path as checkpoint reload)
        instead of per-param copy_() to ensure weights are correctly placed in
        cumem-managed memory after sleep/wake_up cycles.
        """

        import torch
        from vllm.platforms import current_platform

        patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        is_npu = str(getattr(current_platform, "device_type", "")).lower() == "npu"
        use_shm = bool(use_shm)
        if is_npu and not use_shm:
            raise RuntimeError(
                "SPECO vLLM draft weight update on NPU requires shared-memory transfer"
            )

        if is_npu and getattr(self, "device", None) is None:
            self.device = torch.device(f"npu:{self.local_rank}")
        assert self.device is not None

        all_weights: list[tuple[str, torch.Tensor]] = []

        def finish_update(result, translated_weights=None):
            if not is_npu:
                return result
            if translated_weights is not None:
                translated_weights.clear()
            all_weights.clear()
            trim_process_host_memory()
            return result

        def on_bucket_received(bucket_weights):
            # Clone immediately: bucket views may be freed (IPC) or overwritten
            # by the next update (persistent SHM). Give each tensor independent
            # device storage before receive_weights() returns.
            all_weights.extend(
                [(name, t.detach().clone()) for name, t in bucket_weights]
            )

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_speco_draft_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(on_bucket_received=on_bucket_received)

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None or not all_weights:
            return finish_update(
                {"loaded_params": 0, "has_draft_model": draft_model is not None}
            )

        draft_method = self._speco_draft_method()
        is_dflash = draft_method == "dflash"
        is_dspark = draft_method == "dspark"
        is_eagle3 = draft_method == "eagle3"

        # Translate training-side names.  DFlash/DSpark publish into the inner
        # DFlashQwen3Model/Qwen3DSparkModel, while EAGLE3 publishes into the
        # outer Eagle3LlamaForCausalLM because lm_head.weight lives outside
        # ``draft_model.model`` in vLLM.
        _strip_prefixes = (
            "module.",
            "_orig_mod.",
            "draft_model.",
            "model.draft_model.",
        )
        if is_dflash or is_dspark:
            _strip_prefixes = (*_strip_prefixes, "model.")
        translated_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in all_weights:
            n = name
            changed = True
            while changed:
                changed = False
                for pfx in _strip_prefixes:
                    if n.startswith(pfx):
                        n = n[len(pfx) :]
                        changed = True
            if "midlayer." in n:
                n = n.replace("midlayer.", "layers.0.")
            if (
                is_eagle3
                and n != "lm_head.weight"
                and "." in n
                and not n.startswith("model.")
            ):
                n = f"model.{n}"
            translated_weights.append((n, tensor))

        loaded_params = len(translated_weights)
        if is_eagle3:
            loaded_params = self._speco_update_draft_weights(translated_weights)
        else:
            inner_model = getattr(draft_model, "model", None)
            if inner_model is None:
                return finish_update(
                    {"loaded_params": 0, "has_draft_model": True},
                    translated_weights,
                )
            logger.warning(
                "[speco draft ipc] loading %d translated weights into %s (method=%s), first 5 keys: %s",
                len(translated_weights),
                type(inner_model).__name__,
                draft_method,
                [n for n, _ in translated_weights[:5]],
            )
            inner_model.load_weights(iter(translated_weights))

            # Rebuild fused KV buffers (torch.cat snapshot, not a view)
            try:
                inner_model._build_fused_kv_buffers()
            except Exception as exc:
                logger.warning(
                    "[speco draft update] _build_fused_kv_buffers failed: %s", exc
                )

        self._speco_diag_draft_state("after_draft_ipc_update")
        # One-time diagnostic: check whether probabilistic sampling is active
        proposer = self._speco_resolve_draft_proposer()
        if proposer is not None and not getattr(
            self, "_speco_logged_sampling_mode", False
        ):
            self._speco_logged_sampling_mode = True
            missing_draft_logits = not hasattr(proposer, "draft_logits")
            draft_logits = getattr(proposer, "draft_logits", None)
            spec_cfg = getattr(
                getattr(getattr(self, "model_runner", None), "vllm_config", None),
                "speculative_config",
                None,
            )
            dsm = (
                getattr(spec_cfg, "draft_sample_method", "UNKNOWN")
                if spec_cfg
                else "NO_SPEC_CFG"
            )
            logger.warning(
                "[speco-diag:sampling_mode] draft_sample_method=%s draft_logits=%s proposer=%s",
                dsm,
                _describe_vllm_draft_logits(draft_logits, missing=missing_draft_logits),
                type(proposer).__name__,
            )
        return finish_update(
            {"loaded_params": loaded_params, "has_draft_model": True},
            translated_weights,
        )

    # ----------------------------------------------------------------
    # Fix: reload DFlash drafter weights from checkpoint after wake_up
    # ----------------------------------------------------------------

    def _speco_get_draft_checkpoint_path(self) -> str | None:
        """Resolve the DFlash drafter checkpoint path from speculative_config."""
        if not self._speco_is_dflash_draft():
            return None
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return None
        vllm_cfg = getattr(runner, "vllm_config", None)
        spec_cfg = getattr(vllm_cfg, "speculative_config", None) if vllm_cfg else None
        draft_model_cfg = (
            getattr(spec_cfg, "draft_model_config", None) if spec_cfg else None
        )
        if draft_model_cfg is None:
            return None
        return getattr(draft_model_cfg, "model", None)

    def _speco_reload_draft_from_checkpoint(self) -> int:
        """Reload DFlash drafter weights from its checkpoint (safetensors).

        Called after target model wake_up to restore drafter weights that were
        lost during sleep(level=2). Returns the number of weight tensors loaded.
        """
        import glob as _glob

        if not self._speco_is_dflash_draft():
            return 0

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None:
            logger.warning("[speco draft reload] no draft model found, skip")
            return 0

        ckpt_path = self._speco_get_draft_checkpoint_path()
        if not ckpt_path:
            logger.warning(
                "[speco draft reload] no draft checkpoint path configured, skip"
            )
            return 0

        st_files = sorted(_glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if not st_files:
            logger.warning(
                "[speco draft reload] no safetensors files in %s, skip", ckpt_path
            )
            return 0

        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.warning("[speco draft reload] safetensors not available, skip")
            return 0

        weights_iter = []
        for st_file in st_files:
            state_dict = load_file(st_file, device="cpu")
            for name, tensor in state_dict.items():
                weights_iter.append((name, tensor))

        if not weights_iter:
            logger.warning("[speco draft reload] checkpoint empty, skip")
            return 0

        try:
            draft_model.load_weights(iter(weights_iter))
            loaded_count = len(weights_iter)
        except Exception as exc:
            logger.warning("[speco draft reload] load_weights failed: %s", exc)
            return 0

        return loaded_count

    def update_weights_from_ipc(
        self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False
    ):
        """Override target weight sync to also reload drafter from checkpoint."""
        patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        patch_verl_bucketed_weight_transfer_npu_staging()
        is_npu = _speco_is_npu_vllm_worker(self)
        # Diagnostic: check draft state BEFORE target sync
        self._speco_diag_draft_state("before_target_sync")
        try:
            with _speco_npu_target_staging(
                self, peft_config=peft_config, use_shm=use_shm
            ):
                result = super().update_weights_from_ipc(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                    use_shm=use_shm,
                )
            # Diagnostic: check draft state AFTER target sync (may be zeroed by wake_up)
            self._speco_diag_draft_state("after_target_sync")
            reloaded = self._speco_reload_draft_from_checkpoint()
            if reloaded > 0:
                logger.warning(
                    "[speco draft reload] drafter weights restored after target sync (%d tensors)",
                    reloaded,
                )
            # Diagnostic: check draft state AFTER reload
            self._speco_diag_draft_state("after_draft_reload")
            return result
        finally:
            if is_npu:
                trim_process_host_memory()

    def _speco_diag_draft_state(self, phase: str):
        """Log norms of key draft model parameters for debugging."""
        if not bool(_bool_or_none(os.getenv(SPECO_VLLM_DRAFT_DIAG_ENV))):
            return

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None:
            logger.warning("[speco-diag:%s] no draft model found", phase)
            return
        try:
            params = dict(draft_model.named_parameters())
            diag_keys = [
                k
                for k in params
                if any(s in k for s in ("markov", "fc.", "norm.", "layers.0."))
            ]
            if not diag_keys:
                diag_keys = list(params.keys())[:5]
            norms = {
                k: f"{params[k].data.float().norm().item():.4f}" for k in diag_keys[:6]
            }
            logger.warning("[speco-diag:%s] draft param norms: %s", phase, norms)
        except Exception as exc:
            logger.warning("[speco-diag:%s] failed: %s", phase, exc)
