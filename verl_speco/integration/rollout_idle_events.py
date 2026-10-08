# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Ray-backed event bus for fine-grained rollout idle-worker events."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, cast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV = "VERL_SPECO_ROLLOUT_IDLE_EVENT_BUS"
DRAFTER_SAMPLE_READY_EVENT = "drafter_sample_ready"


def _ray_module():
    try:
        import ray
    except Exception:  # noqa: BLE001
        return None
    return ray


def _event_bus_actor_class(ray):
    @ray.remote
    class RolloutIdleEventBus:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []
            # Rollout serving and replica-local drafter training share the same
            # devices.  Keep the ownership decision in this actor so a request
            # cannot race a just-admitted Bubble plan.
            self._rollout_owners: dict[str, int] = {}
            self._training_owners: dict[str, tuple[str, float]] = {}
            self._pending_rollout_groups: set[tuple[str, ...]] = set()
            self._replica_groups: dict[int, tuple[str, ...]] = {}

        @staticmethod
        def _worker_group(worker_ids: Any) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(str(worker_id) for worker_id in worker_ids or ())
            )

        def _append_event(self, event: dict[str, Any]) -> None:
            event = dict(event)
            event.setdefault("event_ts", time.time())
            self._events.append(event)

        def _worker_group_for_event(
            self,
            worker_ids: Any,
            event: dict[str, Any],
        ) -> tuple[str, ...]:
            try:
                replica_rank = int(cast(Any, event.get("replica_rank")))
            except (TypeError, ValueError):
                replica_rank = -1
            return self._replica_groups.get(
                replica_rank,
                self._worker_group(worker_ids),
            )

        def _expire_training_owners(self, now: float) -> None:
            expired = [
                worker_id
                for worker_id, (_, expires_at) in self._training_owners.items()
                if expires_at > 0.0 and expires_at <= now
            ]
            for worker_id in expired:
                self._training_owners.pop(worker_id, None)

        def emit(self, event: dict[str, Any]) -> int:
            self._append_event(event)
            return len(self._events)

        def configure_replica_groups(
            self,
            groups: dict[int, list[str] | tuple[str, ...]],
        ) -> dict[int, list[str]]:
            self._replica_groups = {
                int(replica_rank): self._worker_group(worker_ids)
                for replica_rank, worker_ids in dict(groups or {}).items()
                if self._worker_group(worker_ids)
            }
            return {
                replica_rank: list(worker_ids)
                for replica_rank, worker_ids in self._replica_groups.items()
            }

        def acquire_rollout(
            self,
            worker_ids: list[str] | tuple[str, ...],
            event: dict[str, Any],
        ) -> dict[str, Any]:
            """Atomically acquire one rollout replica's colocated resources."""

            group = self._worker_group_for_event(worker_ids, event)
            now = time.time()
            self._expire_training_owners(now)
            blocking_plans = sorted(
                {
                    self._training_owners[worker_id][0]
                    for worker_id in group
                    if worker_id in self._training_owners
                }
            )
            if blocking_plans:
                # One busy notification is sufficient to make the trainer
                # cooperatively reclaim the in-flight Bubble batch.
                if group not in self._pending_rollout_groups:
                    requested = dict(event)
                    requested["event_type"] = "GENERATION_STARTED"
                    requested["release_source"] = "rollout_waiting_for_training"
                    requested["event_ts"] = now
                    self._append_event(requested)
                    self._pending_rollout_groups.add(group)
                return {
                    "acquired": False,
                    "blocking_plan_ids": blocking_plans,
                    "retry_after_sec": 0.01,
                }

            was_idle = all(
                self._rollout_owners.get(worker_id, 0) <= 0 for worker_id in group
            )
            for worker_id in group:
                self._rollout_owners[worker_id] = (
                    self._rollout_owners.get(worker_id, 0) + 1
                )
            if was_idle and group not in self._pending_rollout_groups:
                started = dict(event)
                started["event_type"] = "GENERATION_STARTED"
                started["event_ts"] = now
                self._append_event(started)
            self._pending_rollout_groups.discard(group)
            return {"acquired": True, "blocking_plan_ids": []}

        def release_rollout(
            self,
            worker_ids: list[str] | tuple[str, ...],
            event: dict[str, Any],
        ) -> dict[str, Any]:
            """Release serving ownership and publish an authoritative idle event."""

            group = self._worker_group_for_event(worker_ids, event)
            for worker_id in group:
                remaining = max(self._rollout_owners.get(worker_id, 0) - 1, 0)
                if remaining:
                    self._rollout_owners[worker_id] = remaining
                else:
                    self._rollout_owners.pop(worker_id, None)
            idle = all(
                self._rollout_owners.get(worker_id, 0) <= 0 for worker_id in group
            )
            if idle:
                released = dict(event)
                released.update(
                    {
                        "event_type": "WORKER_IDLE",
                        "idle_confidence": "confirmed",
                        "event_ts": time.time(),
                    }
                )
                self._append_event(released)
            return {"released": True, "idle": idle}

        def reserve_training(
            self,
            worker_ids: list[str] | tuple[str, ...],
            plan_id: str,
            expires_at: float,
        ) -> dict[str, Any]:
            """Atomically reserve idle rollout resources for one Bubble plan."""

            group = self._worker_group(worker_ids)
            now = time.time()
            self._expire_training_owners(now)
            rollout_busy = [
                worker_id
                for worker_id in group
                if self._rollout_owners.get(worker_id, 0) > 0
            ]
            training_busy = [
                worker_id
                for worker_id in group
                if worker_id in self._training_owners
                and self._training_owners[worker_id][0] != str(plan_id)
            ]
            if rollout_busy or training_busy:
                return {
                    "acquired": False,
                    "rollout_busy_workers": rollout_busy,
                    "training_busy_workers": training_busy,
                }
            # A zero expiry is deliberately fail-closed. If the trainer dies,
            # the distributed job should fail instead of silently allowing
            # rollout and drafter training to overlap on the same devices.
            requested_expiry = float(expires_at)
            expiry = max(requested_expiry, now + 1.0) if requested_expiry > 0.0 else 0.0
            for worker_id in group:
                self._training_owners[worker_id] = (str(plan_id), expiry)
            return {"acquired": True, "worker_ids": list(group)}

        def release_training(
            self,
            worker_ids: list[str] | tuple[str, ...],
            plan_id: str,
        ) -> dict[str, Any]:
            group = self._worker_group(worker_ids)
            released = []
            for worker_id in group:
                owner = self._training_owners.get(worker_id)
                if owner is not None and owner[0] == str(plan_id):
                    self._training_owners.pop(worker_id, None)
                    released.append(worker_id)
            return {"released": bool(released), "worker_ids": released}

        def drain(self) -> list[dict[str, Any]]:
            events = self._events
            self._events = []
            return events

        def clear(self) -> int:
            count = len(self._events)
            self._events = []
            return count

    return RolloutIdleEventBus


def ensure_rollout_idle_event_bus(name: str):
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return None
    try:
        return ray.get_actor(name)
    except Exception:  # noqa: BLE001
        pass
    try:
        actor_cls = _event_bus_actor_class(ray)
        return actor_cls.options(name=name, lifetime="detached").remote()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to create SPECO rollout idle event bus %s: %s", name, exc)
        return None


def emit_rollout_idle_event(name: str | None, event: dict[str, Any]) -> bool:
    if not name:
        return False
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return False
    try:
        actor = ray.get_actor(name)
        actor.emit.remote(dict(event))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to emit SPECO rollout idle event to %s: %s", name, exc)
        return False


def _event_bus(name: str | None):
    if not name:
        return None, None
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return ray, None
    try:
        return ray, ray.get_actor(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to resolve SPECO rollout event bus %s: %s", name, exc)
        return ray, None


async def _await_ray_result(ray, result_ref):
    if hasattr(result_ref, "__await__"):
        return await result_ref
    return ray.get(result_ref)


async def acquire_rollout_resource_lease(
    name: str | None,
    *,
    worker_ids: tuple[str, ...],
    event: dict[str, Any],
    timeout_sec: float = 300.0,
) -> tuple[bool, float]:
    """Wait until a rollout replica can safely reclaim its colocated devices."""

    ray, actor = _event_bus(name)
    if actor is None:
        return False, 0.0
    acquire = getattr(actor, "acquire_rollout", None)
    remote = getattr(acquire, "remote", None)
    if not callable(remote):
        return False, 0.0
    started = time.monotonic()
    while True:
        result = await _await_ray_result(
            ray,
            remote(list(worker_ids), dict(event)),
        )
        if isinstance(result, dict) and bool(result.get("acquired", False)):
            return True, time.monotonic() - started
        if time.monotonic() - started >= max(float(timeout_sec), 1.0):
            raise TimeoutError(
                "Timed out waiting for SPECO Bubble training to release rollout "
                f"workers={worker_ids}"
            )
        retry_after = (
            float(result.get("retry_after_sec", 0.01))
            if isinstance(result, dict)
            else 0.01
        )
        await asyncio.sleep(max(retry_after, 0.001))


async def release_rollout_resource_lease(
    name: str | None,
    *,
    worker_ids: tuple[str, ...],
    event: dict[str, Any],
) -> bool:
    ray, actor = _event_bus(name)
    if actor is None:
        return False
    release = getattr(actor, "release_rollout", None)
    remote = getattr(release, "remote", None)
    if not callable(remote):
        return False
    result = await _await_ray_result(ray, remote(list(worker_ids), dict(event)))
    return bool(isinstance(result, dict) and result.get("released", False))


def reserve_rollout_training_resources(
    name: str | None,
    *,
    worker_ids: tuple[str, ...],
    plan_id: str,
    expires_at: float,
) -> bool:
    ray, actor = _event_bus(name)
    if actor is None:
        return False
    reserve = getattr(actor, "reserve_training", None)
    remote = getattr(reserve, "remote", None)
    if not callable(remote):
        return False
    try:
        result = ray.get(remote(list(worker_ids), str(plan_id), float(expires_at)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to reserve rollout resources for Bubble training: %s", exc)
        return False
    return bool(isinstance(result, dict) and result.get("acquired", False))


def configure_rollout_resource_groups(
    name: str | None,
    groups: dict[int, tuple[str, ...]],
) -> bool:
    ray, actor = _event_bus(name)
    if actor is None:
        return False
    configure = getattr(actor, "configure_replica_groups", None)
    remote = getattr(configure, "remote", None)
    if not callable(remote):
        return False
    try:
        result = ray.get(
            remote(
                {
                    int(replica_rank): list(worker_ids)
                    for replica_rank, worker_ids in groups.items()
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to configure rollout resource groups: %s", exc)
        return False
    return isinstance(result, dict) and len(result) == len(groups)


def release_rollout_training_resources(
    name: str | None,
    *,
    worker_ids: tuple[str, ...],
    plan_id: str,
) -> bool:
    ray, actor = _event_bus(name)
    if actor is None:
        return False
    release = getattr(actor, "release_training", None)
    remote = getattr(release, "remote", None)
    if not callable(remote):
        return False
    try:
        result = ray.get(remote(list(worker_ids), str(plan_id)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to release rollout Bubble training resources: %s", exc)
        return False
    return bool(isinstance(result, dict) and result.get("released", False))


def emit_rollout_drafter_sample(
    name: str | None,
    sample: dict[str, Any],
    *,
    sample_id: str,
    replica_rank: int,
    global_step: object,
) -> bool:
    """Publish one SGLang sample without copying it through the final rollout batch.

    The large tensor payload is placed in Ray's object store first.  The event
    bus only carries the ObjectRef and routing/version metadata.  The original
    sample remains attached to ``TokenOutput`` as a lossless fallback until the
    trainer confirms that this event was transactionally committed.
    """

    if not name:
        return False
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return False
    try:
        actor = ray.get_actor(name)
        sample_ref = ray.put(sample)
        actor.emit.remote(
            {
                "event_type": DRAFTER_SAMPLE_READY_EVENT,
                "sample_id": str(sample_id),
                "sample_ref": sample_ref,
                "replica_rank": int(replica_rank),
                "global_step": global_step,
                "event_ts": time.time(),
            }
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to emit SPECO drafter sample to %s: %s", name, exc)
        return False


def drain_rollout_idle_events(name: str | None) -> list[dict[str, Any]]:
    if not name:
        return []
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return []
    try:
        actor = ray.get_actor(name)
        return list(ray.get(actor.drain.remote()) or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to drain SPECO rollout idle events from %s: %s", name, exc)
        return []
