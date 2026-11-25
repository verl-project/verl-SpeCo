# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Shared exact full-support TV objective, with the existing fused backend."""

from functools import lru_cache

import torch

from verl_speco.ops.dspark_fused_loss import (
    fused_loss_capability,
    fused_total_variation,
)


@lru_cache(maxsize=8)
def _fused_available(device):
    return fused_loss_capability(torch.device(device)).available


def use_fused_opd_loss(device, implementation):
    if implementation not in {"auto", "eager", "fused"}:
        raise ValueError("invalid OPD distribution implementation")
    fused = implementation != "eager" and _fused_available(str(device))
    if implementation == "fused" and not fused:
        raise RuntimeError("OPD fused loss is unavailable on this device")
    return fused


def token_total_variation(
    draft_logits,
    target_logits,
    *,
    proposal_temperature=1.0,
    target_temperature=1.0,
    implementation="auto",
):
    if draft_logits.shape != target_logits.shape:
        raise ValueError("proposal and target supports differ")
    draft = draft_logits.float()
    if proposal_temperature != 1.0:
        draft = draft / proposal_temperature
    target = target_logits.detach().float()
    if target_temperature != 1.0:
        target = target / target_temperature
    fused = use_fused_opd_loss(draft.device, implementation)
    if fused and draft.numel():
        return fused_total_variation(
            draft.reshape(-1, draft.shape[-1]),
            target.reshape(-1, target.shape[-1]),
            average_ties=True,
        ).reshape(draft.shape[:-1])
    with torch.no_grad():
        p = target.softmax(-1)
    q = draft.softmax(-1)
    return (p - q).abs().sum(-1) * 0.5
