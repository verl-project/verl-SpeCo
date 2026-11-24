# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Stage immutable CPU features on all ranks before admitting device work."""

from dataclasses import asdict, dataclass
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedTrainingGroup:
    key: tuple
    samples: tuple
    group_sizes: tuple
    spec: object
    payload: dict  # Retain the object-store owner through all optimizer updates.

    @property
    def positions(self):
        return sum(b["_speco_position_count"] for b in self.samples)


def _training_payload_identity(samples, tensors, spec):
    # Include the small conditioning tensors, never the large hidden features.
    identity = (
        samples,
        asdict(spec),
        [
            (
                name,
                tuple(value.shape),
                str(value.dtype),
                None
                if name.endswith(("/hidden_states", "/target_last_hidden_states"))
                else value.tolist(),
            )
            for name, value in sorted(tensors.items())
        ],
    )
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


def encode_training_payload(samples, spec):
    tensors, metadata = {}, []
    for index, batch in enumerate(samples):
        metadata.append(
            {key: value for key, value in batch.items() if not torch.is_tensor(value)}
        )
        tensors.update(
            {
                f"{index}/{key}": value.detach().contiguous()
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
        )
    wire = encode_cpu_tensors(tensors)
    if wire is None:
        raise ValueError("SpecOPD training tensors cannot use the CPU wire format")
    return dict(
        samples=metadata,
        tensors=wire,
        spec=spec,
        identity=_training_payload_identity(metadata, tensors, spec),
    ), tensor_payload_bytes(tensors)


def decode_training_payload(payload):
    tensors = decode_cpu_tensors(payload["tensors"])
    if (
        _training_payload_identity(payload["samples"], tensors, payload["spec"])
        != payload["identity"]
    ):
        raise ValueError("SpecOPD staged training payload identity mismatch")
    samples = [dict(fields) for fields in payload["samples"]]
    for name, value in tensors.items():
        index, key = name.split("/", 1)
        samples[int(index)][key] = value
    return tuple(samples)


def preparation_key(step, revision, feature_indices):
    return int(step), int(revision), tuple(feature_indices)


async def stage_training_group(worker, payload_refs, step, revision, feature_indices):
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
    from verl_speco.backends.specopd_forward import plan_specopd_training_groups

    spec = payload["spec"]
    if not spec.enable:
        raise RuntimeError("SpecOPD preparation contains a disabled training policy")
    spec.validate()
    groups = plan_specopd_training_groups(
        samples,
        worker.trainer._get_dp_world_size(),
        spec.training_microbatch_trajectories,
        spec.training_microbatch_token_budget,
    )
    if key in getattr(worker, "_specopd_cancelled_prepares", set()):
        raise RuntimeError("SpecOPD preparation was cancelled during data transfer")
    worker._specopd_prepared_group = PreparedTrainingGroup(
        key, samples, tuple(groups), spec, payload
    )
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
        "groups": tuple(groups),
        "samples": len(samples),
        "sp": sp,
        "dp": dp,
        "seconds": elapsed,
    }


def take_training_group(worker, step, revision, feature_indices):
    staged = getattr(worker, "_specopd_prepared_group", None)
    if staged is None or staged.key != preparation_key(step, revision, feature_indices):
        raise RuntimeError("SpecOPD training has no matching prepared input group")
    worker._specopd_prepared_group = None
    return staged


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
    if staged is not None and staged.key == key:
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
    expected_plan = dict(
        step=step,
        revision=revision,
        identity=identity,
        samples=samples,
        groups=active[0]["groups"],
        sp=active[0]["sp"],
        dp=active[0]["dp"],
    )
    if any(
        tuple(s["features"]) != tuple(feature_indices)
        or any(s[key] != value for key, value in expected_plan.items())
        for s in active
    ):
        raise RuntimeError("SpecOPD ranks prepared inconsistent input plans")
    return active


def dispatch_prepared_group(
    trainer, samples, step, revision, feature_indices, metrics, spec
):
    import ray

    group = trainer._require_speco_worker_group()
    started = time.perf_counter()
    payload, size = encode_training_payload(samples, spec)
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
