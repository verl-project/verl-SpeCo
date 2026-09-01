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

"""TransferQueue bridge for SPECO drafter feature transport.

This module lets SPECO route large per-sample drafter-training tensors (hidden
states, target logprobs) through TransferQueue (TQ) instead of funneling them
through the ``SpecoRayPPOTrainer`` driver process and the Ray object store. It
is the SpeCo-side analog of verl's ``transferqueue_utils.py``, but used as a
standalone transport library -- it does **not** depend on verl's
``main_ppo_sync`` TQ integration and does **not** modify upstream verl.

Design:
- P0: offload SGLang-collected ``hidden_states`` (a1 path).
- P1: offload old-logprob-collected ``hidden_states`` (a2 path) -- replaces the
  ``ray.put`` chunk + driver relay.
- P2: also offload ``target_logprobs`` / ``hidden_raw_target_logprobs``.
- Default ``enable: false`` -> behavior is bit-identical to the current Ray
  path. TQ is only touched when explicitly enabled and the ``transfer_queue``
  package is importable.

A sample remains in TQ until task cleanup: a SPECO drafter replica contains
multiple TP/SP ranks and each rank reads the same sample (the owner-route
dispatch duplicates a DP bucket to all SP ranks of one replica; only the SP
leader is ``is_collect``). Deleting a sample after the first read would race the
remaining ranks. Garbage collection is therefore deferred to task teardown; a
finer-grained leader-clears-after-barrier is future work.

Note: the TQ call sites follow the documented KV API (``kv_put`` /
``kv_batch_get`` / ``kv_close``) of TransferQueue 0.1.10. The exact signatures
(keyword names, return shapes) must be verified against the installed TQ
version on first run; the bridge fails loud, never silently.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Partition under which SPECO drafter samples are stored.
_SPECO_TQ_PARTITION = "speco_drafter_features"

try:
    import transfer_queue as tq  # type: ignore
    from transfer_queue import KVBatchMeta  # noqa: F401  (re-exported for symmetry)

    _TQ_IMPORTABLE = True
except ImportError:
    _TQ_IMPORTABLE = False

    class KVBatchMeta:  # type: ignore[no-redef]
        """Stand-in used only when TransferQueue is not installed."""

    class _MockTQ:
        """Mock that raises on any use; only hit if enabled without TQ installed."""

        def __getattr__(self, name: str) -> Any:
            def _raise(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    f"transfer_queue is not installed. Cannot call tq.{name}(). "
                    "Install with `pip install TransferQueue==0.1.10` or disable "
                    "actor_rollout_ref.rollout.drafter.training.transfer_queue.enable."
                )

            return _raise

    tq = _MockTQ()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "enabled": False,  # config says enable=true
    "configured": False,  # configure_transfer_queue has run
    "initialized": False,  # tq.init() has run in this process
    "config": None,  # the transfer_queue sub-config (plain dict)
    "owner": False,  # this process created the task-level TQ system
    "ray_initialized_here": False,
    "ray_address": None,
    "ray_namespace": None,
}


def configure_transfer_queue(training_cfg: Any) -> bool:
    """Read the ``transfer_queue`` sub-config from the drafter training config.

    Called from both the SGLang server process (via env-serialized drafter
    config) and the drafter worker process (via full hydra config). Idempotent.
    Returns whether TQ transport is usable in this process.
    """

    global _state
    tq_cfg = _extract_tq_config(training_cfg)
    with _state_lock:
        _state["config"] = tq_cfg
        _state["enabled"] = bool(tq_cfg.get("enable")) if tq_cfg else False
        _state["configured"] = True
    usable = _state["enabled"] and _TQ_IMPORTABLE
    if _state["enabled"] and not _TQ_IMPORTABLE:
        logger.warning(
            "[SpeCo TQ] transfer_queue.enable=true but transfer_queue package is "
            "not installed; falling back to inline Ray transport."
        )
    return usable


def _extract_tq_config(training_cfg: Any) -> Optional[dict]:
    if training_cfg is None:
        return None
    transfer_queue_cfg = None
    if hasattr(training_cfg, "get"):
        transfer_queue_cfg = training_cfg.get("transfer_queue", None)
    elif isinstance(training_cfg, dict):
        transfer_queue_cfg = training_cfg.get("transfer_queue", None)
    if transfer_queue_cfg is None:
        plain = _to_plain_dict(training_cfg)
        if isinstance(plain, dict) and any(
            key in plain for key in ("enable", "backend", "controller", "ray")
        ):
            transfer_queue_cfg = plain
        else:
            return None
    return _to_plain_dict(transfer_queue_cfg)


def _to_plain_dict(value: Any) -> dict:
    """Convert OmegaConf DictConfig / nested mapping to a plain dict."""

    if hasattr(value, "to_container"):
        try:
            import omegaconf

            return omegaconf.OmegaConf.to_container(value, resolve=True)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    return value


def is_transfer_queue_enabled() -> bool:
    """True only if configured enabled AND the TQ package is importable."""

    return bool(_state["enabled"]) and _TQ_IMPORTABLE


def connect_ray_cluster(
    ray_address: str | None,
    namespace: str | None = None,
) -> None:
    """Connect this ordinary process to the Ray cluster hosting TQ.

    PR #48 workers are already Ray actors and therefore do not call this
    function.  Standalone owner, Producer and torchrun ranks must call it
    before ``tq.init`` so TQ 0.1.10 can discover its named Controller actor.
    """

    try:
        import ray
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Ray is required by TransferQueue 0.1.10. Install TransferQueue==0.1.10 "
            "and connect all standalone processes to the same Ray cluster."
        ) from exc

    if ray.is_initialized():
        return
    address = str(ray_address or "auto").strip() or "auto"
    kwargs: dict[str, Any] = {"address": address}
    if namespace:
        kwargs["namespace"] = str(namespace)
    ray.init(**kwargs)
    with _state_lock:
        _state["ray_initialized_here"] = True
        _state["ray_address"] = address
        _state["ray_namespace"] = namespace


def start_transfer_queue_owner(tq_config: Any) -> None:
    """Create the task-level named TQ Controller in the current Ray cluster."""

    plain = _extract_tq_config(tq_config)
    if plain is None:
        raise ValueError("TransferQueue owner configuration is missing")
    if not bool(plain.get("enable", True)):
        raise ValueError("TransferQueue owner requires transfer_queue.enable=true")
    if not _TQ_IMPORTABLE:
        raise RuntimeError("TransferQueue==0.1.10 is required to start the TQ owner")
    with _state_lock:
        if _state["initialized"]:
            raise RuntimeError("TransferQueue is already initialized in this process")
    tq.init(_as_tq_config(_native_tq_config(plain)))
    with _state_lock:
        _state["config"] = plain
        _state["enabled"] = True
        _state["configured"] = True
        _state["initialized"] = True
        _state["owner"] = True
    logger.info("[SpeCo TQ] standalone owner started (partition=%s)", _partition_id())


def connect_transfer_queue_client() -> None:
    """Attach this process to the named TQ Controller on its Ray cluster."""

    if not _TQ_IMPORTABLE:
        raise RuntimeError("TransferQueue==0.1.10 is required to connect a TQ client")
    if not bool(_state["enabled"]):
        raise RuntimeError(
            "configure_transfer_queue() must enable TQ before client connect"
        )
    _ensure_initialized()


def init_transfer_queue(config: Any) -> bool:
    """Cluster-wide TQ bootstrap, called once from the SpecoTaskRunner.

    Mirrors verl ``main_ppo_sync`` calling ``tq.init(config.transfer_queue)``
    in the TaskRunner before workers spawn. Other Ray processes lazily call
    ``tq.init(config)`` with the same native configuration and connect to the
    named TransferQueue controller. Returns whether TQ is usable; no-op
    (returns False) when disabled or not installed.
    """

    tq_cfg = _extract_tq_config(_drafter_training_cfg(config))
    if tq_cfg is None or not bool(tq_cfg.get("enable")) or not _TQ_IMPORTABLE:
        return False
    tq.init(_as_tq_config(_native_tq_config(tq_cfg)))
    with _state_lock:
        _state["config"] = _to_plain_dict(tq_cfg)
        _state["enabled"] = True
        _state["initialized"] = True
        _state["owner"] = True
    logger.info(
        "[SpeCo TQ] TransferQueue bootstrapped in task runner (partition=%s)",
        _SPECO_TQ_PARTITION,
    )
    return True


def _drafter_training_cfg(config: Any) -> Any:
    try:
        return config.actor_rollout_ref.rollout.drafter.training
    except AttributeError:
        return None


def _ensure_initialized() -> None:
    """Lazily ``tq.init(config)`` once per worker process.

    TransferQueue 0.1.10 first tries to discover the named controller and ignores
    the supplied configuration when one already exists. Supplying the same
    native configuration in every process is therefore safe for ordinary
    clients and also prevents an unexpectedly early client from creating a
    default-configured controller.
    """

    if _state["initialized"]:
        return
    with _state_lock:
        if _state["initialized"]:
            return
        configured = _state.get("config")
        if not isinstance(configured, Mapping):
            raise RuntimeError(
                "configure_transfer_queue() must provide TQ configuration before init"
            )
        tq.init(_as_tq_config(_native_tq_config(configured)))
        _state["initialized"] = True


def _native_tq_config(tq_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove SPECO-only connection/protocol fields before ``tq.init``."""

    project_keys = {
        "enable",
        "package_version",
        "ray",
        "partition_id",
        "run_id",
        "schema_version",
        "connect_timeout_seconds",
        "poll_interval_seconds",
        "drop_last",
    }
    return {key: value for key, value in tq_cfg.items() if key not in project_keys}


def _as_tq_config(value: Mapping[str, Any]) -> Any:
    """TQ 0.1.10 annotates its config as DictConfig; keep tests dependency-light."""

    try:
        from omegaconf import OmegaConf

        return OmegaConf.create(dict(value))
    except ImportError:  # pragma: no cover - project normally depends on OmegaConf
        return dict(value)


def _partition_id() -> str:
    config = _state.get("config") or {}
    value = config.get("partition_id") if isinstance(config, Mapping) else None
    return str(value or _SPECO_TQ_PARTITION)


# ---------------------------------------------------------------------------
# Key / put / get / close
# ---------------------------------------------------------------------------


def make_sample_key(global_step: Any, replica_rank: Any, request_id: Any) -> str:
    """Build a deterministic, cluster-unique key for one drafter sample.

    Uniqueness space: (global_step, replica_rank, request_id). Each rollout
    request produces exactly one drafter_sample, so this is unique per sample.
    """

    return f"speco:{global_step}:{replica_rank}:{request_id}"


def put_sample(
    key: str,
    tensor_dict: dict,
    *,
    tag: Optional[dict] = None,
) -> None:
    """Store a dict of CPU tensors under ``key`` in the SPECO TQ partition.

    ``tensor_dict`` values must be CPU ``torch.Tensor`` (or None). None values
    are dropped. Raises if TQ is enabled but the call fails -- never silently
    degrades, so a transport failure surfaces immediately rather than dropping
    a sample.
    """

    if not is_transfer_queue_enabled():
        raise RuntimeError("put_sample called while TransferQueue is not enabled.")
    payload = {k: v for k, v in tensor_dict.items() if torch.is_tensor(v)}
    if not payload:
        return
    _ensure_initialized()
    # Pass a plain single-sample dict of columns. TQ's kv_put adds its required
    # batch dimension internally; constructing a scalar TensorDict here would be
    # incorrect. (Exact kwarg names verified against TQ 0.1.10 on first run.)
    tq.kv_put(
        key=key,
        partition_id=_partition_id(),
        fields=payload,
        tag=tag or {},
    )


def get_sample(key: str) -> dict:
    """Retrieve one tensor dict without deleting it.

    A drafter replica may execute this method on multiple TP/SP ranks; each
    rank reads the same key. TQ storage is released once at task shutdown by
    the process that initialized it, after every consumer has finished.
    """

    if not is_transfer_queue_enabled():
        raise RuntimeError("get_sample called while TransferQueue is not enabled.")
    _ensure_initialized()
    # TQ returns the stored sample (TensorDict-like). Return shape is version
    # dependent, so handle both a direct value and a {key: value} mapping.
    result = tq.kv_batch_get(keys=[key], partition_id=_partition_id())
    value = _extract_value(result, key)
    if value is None:
        return {}
    return _tensordict_to_dict(value)


def list_samples() -> dict[str, dict[str, Any]]:
    """Return key -> tag for the configured partition without fetching fields."""

    if not is_transfer_queue_enabled():
        raise RuntimeError("list_samples called while TransferQueue is not enabled.")
    _ensure_initialized()
    result = tq.kv_list(partition_id=_partition_id())
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError(f"tq.kv_list returned unsupported type {type(result)!r}")
    # 0.1.10 returns key -> tag when partition_id is supplied.  Accept the
    # partition -> (key -> tag) wrapper as well to keep the bridge version-safe.
    nested = result.get(_partition_id())
    if isinstance(nested, Mapping) and all(
        isinstance(v, Mapping) for v in nested.values()
    ):
        result = nested
    records: dict[str, dict[str, Any]] = {}
    for key, tag in result.items():
        if tag is None:
            records[str(key)] = {}
        elif isinstance(tag, Mapping):
            records[str(key)] = dict(tag)
        else:
            raise TypeError(
                f"TQ tag for key {key!r} must be a mapping, got {type(tag)!r}"
            )
    return records


def get_samples(keys: Sequence[str]) -> list[tuple[str, dict[str, Any]]]:
    """Batch-fetch records and return one plain field dict per input key."""

    normalized_keys = [str(key) for key in keys]
    if not normalized_keys:
        return []
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("get_samples keys must be unique")
    if not is_transfer_queue_enabled():
        raise RuntimeError("get_samples called while TransferQueue is not enabled.")
    _ensure_initialized()
    result = tq.kv_batch_get(keys=normalized_keys, partition_id=_partition_id())
    values = _split_batch_result(result, normalized_keys)
    return [
        (key, _tensordict_to_dict(value))
        for key, value in zip(normalized_keys, values, strict=True)
    ]


def clear_samples(keys: Sequence[str]) -> None:
    """Delete consumed records from the configured partition."""

    normalized_keys = [str(key) for key in keys]
    if not normalized_keys:
        return
    if not is_transfer_queue_enabled():
        raise RuntimeError("clear_samples called while TransferQueue is not enabled.")
    _ensure_initialized()
    tq.kv_clear(keys=normalized_keys, partition_id=_partition_id())


def close_transfer_queue_client() -> None:
    """Close only this process's TQ client; never kill the shared Controller."""

    with _state_lock:
        if _state["owner"]:
            raise RuntimeError("TQ owner must use close_transfer_queue_owner()")
        initialized = bool(_state["initialized"])
        _state["initialized"] = False
    if initialized and _TQ_IMPORTABLE:
        try:
            client = tq.get_client()
            if client is not None:
                client.close()
        except Exception:  # noqa: BLE001
            logger.debug("[SpeCo TQ] local client close raised", exc_info=True)
    _shutdown_local_ray_connection()


def close_transfer_queue_owner() -> None:
    """Close global TQ resources.  Only the process that started them may call."""

    close_transfer_queue()
    _shutdown_local_ray_connection()


def _shutdown_local_ray_connection() -> None:
    with _state_lock:
        initialized_here = bool(_state["ray_initialized_here"])
        _state["ray_initialized_here"] = False
    if not initialized_here:
        return
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except ImportError:  # pragma: no cover
        return


def close_transfer_queue() -> None:
    """Close task-level TQ resources if this process initialized them.

    Only the TaskRunner owns controller/storage teardown. Worker-side lazy
    clients must not call this because they may still be serving other ranks.
    """

    with _state_lock:
        if not _state["owner"]:
            return
        _state["owner"] = False
        _state["initialized"] = False
    try:
        tq.close()
    except AttributeError:
        # tq.close() is not part of the public TQ API on some versions; nothing
        # to tear down explicitly. The named controller/storage actors are
        # reaped when the Ray job exits.
        logger.debug("[SpeCo TQ] tq.close() unavailable; skipping explicit teardown")
    except Exception:  # noqa: BLE001
        logger.debug("[SpeCo TQ] tq.close() raised; ignoring shutdown error")


def _split_batch_result(result: Any, keys: Sequence[str]) -> list[Any]:
    if result is None:
        raise KeyError(f"TQ returned no payload for keys={list(keys)!r}")
    # TensorDict exposes a batch_size and indexes rows with ``result[index]``.
    # Check this before the generic Mapping branch because some TensorDict
    # versions also satisfy mapping-like protocols for their field columns.
    if getattr(result, "batch_size", None) is not None:
        return _index_batch_rows(result, len(keys))
    if isinstance(result, Mapping):
        if all(key in result for key in keys):
            return [result[key] for key in keys]
    if isinstance(result, (list, tuple)):
        if len(result) != len(keys):
            raise RuntimeError(
                f"TQ returned {len(result)} rows for {len(keys)} requested keys"
            )
        return list(result)
    if len(keys) == 1:
        return [result]
    return _index_batch_rows(result, len(keys))


def _index_batch_rows(result: Any, expected_rows: int) -> list[Any]:
    try:
        rows = [result[index] for index in range(expected_rows)]
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"Unable to split TQ batch result of type {type(result)!r} into {expected_rows} rows"
        ) from exc
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"TQ returned {len(rows)} rows for {expected_rows} requested keys"
        )
    return rows


def _extract_value(result: Any, key: str) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get(key)
    if isinstance(result, (list, tuple)):
        return result[0] if len(result) > 0 else None
    return result


def _tensordict_to_dict(value: Any) -> dict:
    if hasattr(value, "items"):
        return {k: v for k, v in value.items()}
    return dict(value)


__all__ = [
    "KVBatchMeta",
    "configure_transfer_queue",
    "connect_ray_cluster",
    "connect_transfer_queue_client",
    "start_transfer_queue_owner",
    "close_transfer_queue",
    "close_transfer_queue_client",
    "close_transfer_queue_owner",
    "clear_samples",
    "init_transfer_queue",
    "is_transfer_queue_enabled",
    "list_samples",
    "make_sample_key",
    "put_sample",
    "get_sample",
    "get_samples",
]
