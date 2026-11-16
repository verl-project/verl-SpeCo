# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Inference-only Ulysses tree attention, scoped to a single joint forward.

Token activations are sequence-sharded. QKV all-to-all gathers sequence while
scattering heads; the reverse exchange restores the local sequence shard.
Normal PPO/ref attention forwards are restored unchanged after this call.
"""

from contextlib import contextmanager
import math
from types import MethodType

import torch
import torch.distributed as dist

from verl_speco.joint_scoring import BranchScores, JointScores, pack_joint_batch


def validate_ulysses_model(model, size):
    cfg = model.config
    if cfg.num_attention_heads % size or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("Q heads must divide SP and be an integer multiple of KV heads")
    # Adaptive RoPE updates its frequencies from the largest observed position.
    # Local sequence shards can observe different maxima; do not silently use
    # rank-dependent frequencies. The deployed Qwen3-8B uses static RoPE.
    for module in model.modules():
        rope_type = str(getattr(module, "rope_type", "default"))
        if "dynamic" in rope_type or rope_type == "longrope":
            raise ValueError("joint Ulysses adaptive RoPE requires a separate global-frequency implementation")


def exchange(x, scatter_dim, gather_dim, group):
    size = dist.get_world_size(group)
    if x.shape[scatter_dim] % size:
        raise ValueError("Ulysses exchange dimension must divide the SP size")
    send = torch.stack(torch.chunk(x, size, dim=scatter_dim)).contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return torch.cat(tuple(received.unbind(0)), dim=gather_dim).contiguous()


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


@contextmanager
def tree_attention(model, mask, group):
    """Override instance methods only, preserving global upstream SP patches."""
    changed = []
    size = dist.get_world_size(group)

    def forward(attn, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("joint Ulysses attention is inference-only")
        if kwargs.get("past_key_values") is not None or kwargs.get("past_key_value") is not None:
            raise ValueError("joint Ulysses does not accept KV caches")
        shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(hidden_states).view(shape)
        k = attn.k_proj(hidden_states).view(shape)
        v = attn.v_proj(hidden_states).view(shape)
        if hasattr(attn, "q_norm"):
            q, k = attn.q_norm(q), attn.k_norm(k)
        if position_embeddings is None:
            raise ValueError("joint Ulysses requires explicit logical RoPE embeddings")
        cos, sin = (x.unsqueeze(2) for x in position_embeddings)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        # Minimal KV replication that makes the KV head axis divisible by SP.
        repeats = size // math.gcd(size, k.shape[2])
        k, v = k.repeat_interleave(repeats, dim=2), v.repeat_interleave(repeats, dim=2)
        q, k, v = (exchange(x, 2, 1, group) for x in (q, k, v))
        groups = q.shape[2] // k.shape[2]
        k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        weights = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
        weights = (weights + mask.to(weights.dtype)).softmax(-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        output = exchange(output, 1, 2, group)
        output = attn.o_proj(output.reshape(*hidden_states.shape[:-1], -1))
        return output, None

    try:
        for layer in model.modules():
            if all(hasattr(layer, x) for x in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "scaling")):
                own = "forward" in layer.__dict__
                changed.append((layer, own, layer.__dict__.get("forward")))
                layer.forward = MethodType(forward, layer)
        if len(changed) != model.config.num_hidden_layers:
            raise ValueError("joint Ulysses requires one supported full-attention module per layer")
        yield
    finally:
        for layer, own, original in reversed(changed):
            if own:
                layer.forward = original
            else:
                del layer.forward


def _gather_rows(local, rows, start, group):
    """Gather selected rows only, without gathering the full vocabulary output."""
    output = local.new_zeros((len(rows), *local.shape[1:]))
    rows = torch.as_tensor(rows, dtype=torch.long, device=local.device)
    owned = (rows >= start) & (rows < start + len(local))
    output[owned] = local.index_select(0, rows[owned] - start)
    dist.all_reduce(output, group=group)
    return output


@torch.no_grad()
def score_ulysses_joint_batch(
    model, requests, group, *, actor_temperature=1.0, target_temperature=1.0,
    hidden_state_indices=(), calculate_entropy=False, return_target_logits=True,
    capture_empty_trunk_features=True, max_packed_tokens=8192, max_mask_elements=16777216,
):
    for t in (actor_temperature, target_temperature):
        if not math.isfinite(t) or t <= 0:
            raise ValueError("scoring temperatures must be finite and positive")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    cfg = model.config
    validate_ulysses_model(model, size)
    parameter = next(model.parameters())
    packed = pack_joint_batch(
        requests, device=parameter.device, dtype=parameter.dtype, vocab_size=cfg.vocab_size,
        max_packed_tokens=max_packed_tokens, max_mask_elements=max_mask_elements,
        sequence_multiple=size,
    )
    width = packed.input_ids.shape[1] // size
    start = rank * width
    states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with tree_attention(model, packed.attention_mask, group):
            output = model(
                input_ids=packed.input_ids[:, start:start + width],
                position_ids=packed.position_ids[:, start:start + width],
                attention_mask=None, use_cache=False, output_hidden_states=True, return_dict=True,
            )
    finally:
        for module, training in states:
            module.training = training
    results = []
    for i, request in enumerate(requests):
        n = len(request.trunk_ids)
        rows = torch.arange(request.response_start - 1, n - 1, device=parameter.device)
        owned = (rows >= start) & (rows < start + width)
        logits = output.logits[i].index_select(0, rows[owned] - start).float() / actor_temperature
        labels = torch.tensor(request.trunk_ids[request.response_start:], device=parameter.device)[owned]
        log_probs = logits.new_zeros(len(rows))
        log_probs[owned] = logits.gather(-1, labels[:, None]).squeeze(-1) - logits.logsumexp(-1)
        if len(rows):
            dist.all_reduce(log_probs, group=group)
        entropy = None
        if calculate_entropy:
            local_logp = logits.log_softmax(-1)
            entropy = logits.new_zeros(len(rows))
            entropy[owned] = -(local_logp.exp() * local_logp).sum(-1)
            if len(rows):
                dist.all_reduce(entropy, group=group)
        branches = []
        readouts = packed.branch_rows[i]
        flat_rows = [row for readout in readouts for row in readout]
        if flat_rows:
            # Compute vocabulary normalization only at branch readout rows owned
            # by this rank; never normalize the entire trunk a second time.
            row_ids = torch.tensor(flat_rows, device=parameter.device)
            owned_readouts = (row_ids >= start) & (row_ids < start + width)
            flat_logz = output.logits.new_zeros(len(flat_rows), dtype=torch.float32)
            flat_logz[owned_readouts] = (
                output.logits[i].index_select(0, row_ids[owned_readouts] - start).float()
                / target_temperature
            ).logsumexp(-1)
            dist.all_reduce(flat_logz, group=group)
            flat_hidden = _gather_rows(output.hidden_states[-1][i], flat_rows, start, group)
            flat_logits = _gather_rows(output.logits[i], flat_rows, start, group) if return_target_logits else None
            offset = 0
            for branch, readout in zip(request.branches, readouts, strict=True):
                end = offset + len(readout)
                branches.append(BranchScores(
                    branch.branch_id,
                    flat_logits[offset:end] if flat_logits is not None else None,
                    flat_logz[offset:end], flat_hidden[offset:end],
                ))
                offset = end
        features = {
            index: _gather_rows(output.hidden_states[index][i], range(n), start, group)
            for index in hidden_state_indices
        } if request.branches or capture_empty_trunk_features else {}
        results.append(JointScores(request.sample_id, log_probs, tuple(branches), features, entropy))
    return tuple(results), {
        "model_forward_calls": 1, "logical_rows": packed.logical_rows,
        "independent_rows": packed.independent_rows, "padded_rows": packed.input_ids.numel(),
        "mask_elements": packed.attention_mask.numel(), "sp_size": size,
    }
