# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Stage immutable CPU features on all ranks before admitting device work."""

import hashlib
import json
import logging
import time

import torch

from verl_speco.integration.tensor_wire import (
    encode_cpu_tensors,
    decode_cpu_tensors,
    tensor_payload_bytes,
)
from verl_speco.joint_scoring import BranchScores, JointScores

logger = logging.getLogger(__name__)


def _training_payload_identity(samples, tensors):
    def signature(name):
        tensor = tensors[name]
        return tuple(tensor.shape), str(tensor.dtype)

    identity = [
        [
            r.sample_id,
            r.trunk_ids,
            r.response_start,
            [
                (
                    b.branch_id,
                    b.anchor,
                    b.token_ids,
                    b.source,
                    b.accepted_count,
                    b.verified_count,
                )
                for b in r.branches
            ],
            [(i, signature(name)) for i, name in sorted(features.items())],
            [(bid, signature(name)) for bid, name in branches],
        ]
        for r, features, branches in samples
    ]
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


def encode_training_payload(samples):
    """No torch.Tensor is pickled by value; bf16 travels as lossless NumPy bytes."""
    tensors, metadata = {}, []

    def store(name, tensor):
        if tensor.device.type != "cpu":
            raise ValueError(
                "SpecOPD training transport requires independent CPU scores"
            )
        tensors[name] = tensor.detach().contiguous()
        return name

    for index, (request, scores) in enumerate(samples):
        if scores.sample_id != request.sample_id:
            raise ValueError("SpecOPD training transport sample identity mismatch")
        end = max(b.anchor for b in request.branches) + 1
        features = {
            i: store(f"{index}/f/{i}", value[:end])
            for i, value in scores.trunk_features.items()
        }
        branches = [
            (b.branch_id, store(f"{index}/b/{j}/h", b.target_hidden))
            for j, b in enumerate(scores.branches)
        ]
        metadata.append((request, features, branches))
    wire = encode_cpu_tensors(tensors)
    if wire is None:
        raise ValueError("SpecOPD training tensors cannot use the CPU wire format")
    payload = {
        "samples": metadata,
        "tensors": wire,
        "identity": _training_payload_identity(metadata, tensors),
    }
    return payload, tensor_payload_bytes(tensors)


def decode_training_payload(payload):
    tensors = decode_cpu_tensors(payload["tensors"])
    samples = payload["samples"]
    if _training_payload_identity(samples, tensors) != payload["identity"]:
        raise ValueError("SpecOPD staged training payload identity mismatch")
    return tuple(
        (
            request,
            JointScores(
                request.sample_id,
                torch.empty(0),
                tuple(
                    BranchScores(bid, None, None, tensors[name])
                    for bid, name in branches
                ),
                {i: tensors[name] for i, name in features.items()},
            ),
        )
        for request, features, branches in samples
    )


def preparation_key(step, revision, feature_indices):
    return int(step), int(revision), tuple(feature_indices)


async def stage_training_group(
    worker, payload_refs, step, revision, feature_indices, *, spec=None
):
    """CPU-only readiness RPC: never enter HCCL while peers deserialize data."""
    key = preparation_key(step, revision, feature_indices)
    if key in getattr(worker, "_specopd_cancelled_prepares", set()):
        raise RuntimeError("SpecOPD preparation was cancelled")
    if getattr(worker, "_specopd_prepared_group", None) is not None:
        raise RuntimeError("SpecOPD already has an unconsumed prepared group")
    started = time.perf_counter()
    logger.warning("[SpecOPD stage] enter rank=%s step=%s", worker.rank, step)
    if len(payload_refs) != 1:
        raise ValueError("SpecOPD expects one shared training object reference")
    reference = payload_refs[0]
    payload = reference if isinstance(reference, dict) else await reference
    samples = decode_training_payload(payload)
    from verl_speco.specopd_config import SpecOPDConfig
    from verl_speco.backends.specopd_forward import plan_specopd_training_groups
    from verl_speco.integration.specopd_collectives import training_schedule_digest

    spec = spec or SpecOPDConfig()
    policy = (
        spec.optimizer_updates_per_collection,
        spec.min_scored_positions_per_update,
        spec.training_microbatch_trajectories,
        spec.training_microbatch_token_budget,
    )
    if tuple(payload.get("policy", policy)) != policy:
        raise RuntimeError("SpecOPD preparation policy differs from the driver")
    selected = [(r, s) for r, s in samples if r.branches]
    groups = plan_specopd_training_groups(
        selected,
        worker.trainer._get_dp_world_size(),
        spec.training_microbatch_trajectories,
        spec.training_microbatch_token_budget,
    )
    schedule = training_schedule_digest(
        worker.trainer,
        step=step,
        payload_identity=payload["identity"],
        spec=spec,
        feature_indices=feature_indices,
        group_sizes=groups,
    )
    if key in getattr(worker, "_specopd_cancelled_prepares", set()):
        raise RuntimeError("SpecOPD preparation was cancelled during data transfer")
    worker._specopd_prepared_group = (key, samples, payload)
    sp, dp = worker.trainer._get_sp_world_size(), worker.trainer._get_dp_world_size()
    elapsed = time.perf_counter() - started
    logger.warning(
        "[SpecOPD stage] ready rank=%s step=%s samples=%s elapsed_s=%.4f",
        worker.rank,
        step,
        len(samples),
        elapsed,
    )
    return {
        "rank": worker.rank,
        "step": int(step),
        "revision": int(revision),
        "features": tuple(feature_indices),
        "identity": payload["identity"],
        "schedule": schedule,
        "samples": len(samples),
        "sp": sp,
        "dp": dp,
        "seconds": elapsed,
    }


def take_training_group(worker, step, revision, feature_indices):
    staged = getattr(worker, "_specopd_prepared_group", None)
    if staged is None or staged[0] != preparation_key(step, revision, feature_indices):
        raise RuntimeError("SpecOPD training has no matching prepared input group")
    worker._specopd_prepared_group = None
    return staged[1]


def discard_training_group(worker, step, revision, feature_indices):
    key = preparation_key(step, revision, feature_indices)
    cancelled = getattr(worker, "_specopd_cancelled_prepares", None)
    if cancelled is None:
        cancelled = worker._specopd_cancelled_prepares = set()
    cancelled.add(key)
    # Sequential RL keeps at most one outstanding group. Bound bookkeeping.
    if len(cancelled) > 16:
        cancelled.remove(min(cancelled))
    staged = getattr(worker, "_specopd_prepared_group", None)
    if staged is not None and staged[0] == key:
        worker._specopd_prepared_group = None
    return True


def validate_prepared_group(
    statuses, *, step, revision, feature_indices, identity, samples
):
    active = [s for s in statuses if s is not None]
    if not active:
        raise RuntimeError("SpecOPD no training ranks prepared their inputs")
    expected = active[0]["sp"] * active[0]["dp"]
    if len(active) != expected or len({s["rank"] for s in active}) != expected:
        raise RuntimeError("SpecOPD not all training ranks prepared their inputs")
    for s in active:
        if (
            s["step"] != step
            or s["revision"] != revision
            or tuple(s["features"]) != tuple(feature_indices)
            or s["identity"] != identity
            or s["schedule"] != active[0]["schedule"]
            or s["samples"] != samples
            or s["sp"] != active[0]["sp"]
            or s["dp"] != active[0]["dp"]
        ):
            raise RuntimeError("SpecOPD ranks prepared inconsistent input plans")
    return active


def dispatch_prepared_group(
    trainer, samples, step, revision, feature_indices, metrics, spec=None
):
    import ray

    samples = tuple((r, s) for r, s in samples if r.branches)
    group = trainer._require_speco_worker_group()
    started = time.perf_counter()
    payload, size = encode_training_payload(samples)
    if spec is not None:
        payload["policy"] = (
            spec.optimizer_updates_per_collection,
            spec.min_scored_positions_per_update,
            spec.training_microbatch_trajectories,
            spec.training_microbatch_token_budget,
        )
    reference = ray.put(payload)
    metrics["timing_s/specopd_transport_encode"] = time.perf_counter() - started
    metrics["specopd/transport_tensor_bytes"] = size
    try:
        started = time.perf_counter()
        prepared = trainer._ray_get_if_needed(
            group.prepare_specopd_group(
                (reference,),
                step,
                revision,
                feature_indices,
            )
        )
        ready = validate_prepared_group(
            prepared,
            step=step,
            revision=revision,
            feature_indices=feature_indices,
            identity=payload["identity"],
            samples=len(samples),
        )
        metrics["timing_s/specopd_transport_ready"] = time.perf_counter() - started
        metrics["specopd/prepared_ranks"] = len(ready)
        # Only this small RPC enters device collectives. Bulk data is resident
        # everywhere and every participant has acknowledged the same plan.
        output = group.train_specopd_group(step, revision, feature_indices)
        statuses = [s for s in trainer._ray_get_if_needed(output) if s is not None]
    except BaseException:
        group.discard_specopd_group(step, revision, feature_indices)
        raise
    trainer._ray_get_if_needed(
        group.discard_specopd_group(step, revision, feature_indices)
    )
    return statuses
