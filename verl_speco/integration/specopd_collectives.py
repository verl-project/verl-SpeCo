# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Warm loss-reduction groups after every rank acknowledges the shared plan."""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


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
