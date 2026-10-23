# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Memory-bounded target feature alignment for block drafters."""

import torch


def gather_aligned_target_hidden(
    target_last_hidden_states: torch.Tensor | None,
    label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor | None:
    if target_last_hidden_states is None:
        return None
    if target_last_hidden_states.dim() != 3:
        raise ValueError(
            "target_last_hidden_states must have shape [batch, seq, hidden], "
            f"got {tuple(target_last_hidden_states.shape)}"
        )
    batch_size, seq_len, hidden_size = target_last_hidden_states.shape
    if seq_len <= 0:
        return None
    if label_indices.size(0) != batch_size:
        raise ValueError(
            "label_indices batch size must match target_last_hidden_states: "
            f"{label_indices.size(0)} != {batch_size}"
        )
    indices = (label_indices - 1).clamp(min=0, max=seq_len - 1)
    # Dropped blocks point at row 0; callers must mask these rows out of the loss.
    indices = torch.where(block_keep_mask.unsqueeze(-1), indices, 0)
    offsets = (
        torch.arange(batch_size, device=indices.device, dtype=indices.dtype).view(
            batch_size, 1, 1
        )
        * seq_len
    )
    # Avoid the [batch, anchors, seq, hidden] expanded gather input, which
    # Ascend may materialize before performing the gather.
    return (
        target_last_hidden_states.reshape(batch_size * seq_len, hidden_size)
        .index_select(0, (indices + offsets).reshape(-1))
        .view(*indices.shape, hidden_size)
    )


def normalize_raw_target_hidden(
    verifier_norm: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen verifier final norm to raw vLLM final hidden rows.

    Returns the sanitized hidden states and a boolean mask of rows that were
    non-finite or exceeded ``clip`` so callers can drop them from the loss.
    """
    with (
        torch.no_grad(),
        torch.autocast(device_type=hidden.device.type, enabled=False),
    ):
        verifier_norm.eval()
        hidden = verifier_norm(hidden)
        bad_rows = ~torch.isfinite(hidden).all(dim=-1)
        if clip is not None:
            bad_rows |= hidden.float().abs().amax(dim=-1) > clip
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            hidden = hidden.clamp(-clip, clip)
    return hidden, bad_rows
