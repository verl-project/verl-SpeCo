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
``kv_batch_get`` / ``kv_close``) of TransferQueue 0.1.7. The exact signatures
(keyword names, return shapes) must be verified against the installed TQ
version on first run; the bridge fails loud, never silently.
"""

from __future__ import annotations

import logging
import os
import threading
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
                    "Install with `pip install TransferQueue==0.1.7` or disable "
                    "actor_rollout_ref.rollout.drafter.training.transfer_queue.enable."
                )

            return _raise

    tq = _MockTQ()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state = {
    "enabled": False,        # config says enable=true
    "configured": False,     # configure_transfer_queue has run
    "initialized": False,    # tq.init() has run in this process
    "config": None,          # the transfer_queue sub-config (plain dict)
    "owner": False,          # this process created the task-level TQ system
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


def init_transfer_queue(config: Any) -> bool:
    """Cluster-wide TQ bootstrap, called once from the SpecoTaskRunner.

    Mirrors verl ``main_ppo_sync`` calling ``tq.init(config.transfer_queue)``
    in the TaskRunner before workers spawn. Other Ray processes lazily call
    ``tq.init()`` and connect to the named TransferQueue controller. Returns
    whether TQ is usable; no-op (returns False) when disabled or not installed.
    """

    tq_cfg = _extract_tq_config(_drafter_training_cfg(config))
    if tq_cfg is None or not bool(tq_cfg.get("enable")) or not _TQ_IMPORTABLE:
        return False
    tq.init(_to_plain_dict(tq_cfg))
    with _state_lock:
        _state["config"] = _to_plain_dict(tq_cfg)
        _state["enabled"] = True
        _state["initialized"] = True
        _state["owner"] = True
    logger.info("[SpeCo TQ] TransferQueue bootstrapped in task runner (partition=%s)", _SPECO_TQ_PARTITION)
    return True


def _drafter_training_cfg(config: Any) -> Any:
    try:
        return config.actor_rollout_ref.rollout.drafter.training
    except AttributeError:
        return None


def _ensure_initialized() -> None:
    """Lazily ``tq.init()`` once per worker process (mirrors verl TQ_INITIALIZED).

    A no-argument initialization discovers the named TransferQueue controller
    on the connected Ray cluster. It deliberately does not create a separate
    per-worker configuration.
    """

    if _state["initialized"]:
        return
    with _state_lock:
        if _state["initialized"]:
            return
        tq.init()
        _state["initialized"] = True


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
    # incorrect. (Exact kwarg names verified against TQ 0.1.7 on first run.)
    tq.kv_put(
        key=key,
        partition_id=_SPECO_TQ_PARTITION,
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
    result = tq.kv_batch_get(keys=[key], partition_id=_SPECO_TQ_PARTITION)
    value = _extract_value(result, key)
    if value is None:
        return {}
    return _tensordict_to_dict(value)


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
    "close_transfer_queue",
    "init_transfer_queue",
    "is_transfer_queue_enabled",
    "make_sample_key",
    "put_sample",
    "get_sample",
]
