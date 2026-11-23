# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Admit an identical SpecOPD schedule before rank-local preparation/forward."""

import hashlib
import json
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def training_schedule_digest(
    trainer, *, step, payload_identity, spec, feature_indices, group_sizes
):
    """CPU-only plan signature, checked by the all-rank prepare acknowledgements."""
    identity = dict(
        step=int(step),
        payload=payload_identity,
        features=list(feature_indices),
        groups=list(group_sizes),
        updates=spec.optimizer_updates_per_collection,
        minimum=spec.min_scored_positions_per_update,
        anchor_loss=(
            spec.anchor_count,
            spec.random_anchor_count,
            spec.first_reject_weight,
            spec.random_position_decay_gamma,
        ),
        sp=trainer._get_sp_world_size(),
        dp=trainer._get_dp_world_size(),
    )
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


def warm_specopd_training_collectives(trainer, *, step):
    """Warm the existing loss groups only after all CPU preparation ACKs arrive."""
    groups = []
    for group, size in (
        (trainer._get_sp_group(), trainer._get_sp_world_size()),
        (trainer._get_dp_group(), trainer._get_dp_world_size()),
    ):
        if size > 1:
            if group is None:
                raise RuntimeError(
                    "SpecOPD multi-rank training is missing its process group"
                )
            if all(group is not existing for existing in groups):
                groups.append(group)
    if not groups:
        return
    rank = dist.get_rank()
    logger.warning("[SpecOPD collective] enter rank=%s step=%s", rank, step)
    ready = torch.ones(1, dtype=torch.int32, device=trainer.runtime_device)
    for group in groups:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
    logger.warning("[SpecOPD collective] ready rank=%s step=%s", rank, step)
