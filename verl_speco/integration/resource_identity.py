# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Physical accelerator identity for independently ranked Ray worker groups."""

from __future__ import annotations

from typing import Any


def resource_identity(rank: int) -> dict[str, Any]:
    import ray

    context = ray.get_runtime_context()
    devices = tuple(
        sorted(
            str(device)
            for assigned in context.get_accelerator_ids().values()
            for device in assigned
        )
    )
    if not devices:
        raise RuntimeError("Bubble worker has no authoritative accelerator assignment")
    return {"rank": int(rank), "node_id": context.get_node_id(), "device_ids": devices}


def map_training_replicas(records, replicas):
    """Validate complete collectives against actual rollout device ownership."""
    ownership = {}
    for replica_rank, identities in replicas.items():
        for identity in identities:
            for device in identity["device_ids"]:
                key = (identity["node_id"], str(device))
                if key in ownership:
                    raise RuntimeError("Rollout replicas overlap physical accelerators")
                ownership[key] = int(replica_rank)
    mapped = []
    owners = {}
    training_devices = set()
    for record in records:
        if not record.get("available"):
            if record.get("reason") == "not_in_training_group":
                mapped.append(record)
                continue
            raise RuntimeError("Bubble training worker metadata is unavailable")
        assigned = {
            ownership.get((record["node_id"], str(device)))
            for device in record["device_ids"]
        }
        if len(assigned) != 1 or None in assigned:
            raise RuntimeError("Training worker does not match one rollout replica")
        owner = next(iter(assigned))
        if str(record["worker_id"]) in owners:
            raise RuntimeError("Duplicate Bubble training worker identity")
        devices = {(record["node_id"], str(device)) for device in record["device_ids"]}
        if devices & training_devices:
            raise RuntimeError("Bubble training workers overlap physical accelerators")
        training_devices.update(devices)
        owners[str(record["worker_id"])] = owner
        mapped.append({**record, "replica_rank": owner})
    for record in mapped:
        if not record.get("available"):
            continue
        group = tuple(str(rank) for rank in record["training_group_ranks"])
        if (
            str(record["worker_id"]) not in group
            or not group
            or any(owners.get(rank) != record["replica_rank"] for rank in group)
        ):
            raise RuntimeError("Bubble collective spans different rollout replicas")
    return mapped
