# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Inference-only Ulysses tree attention, scoped to a single joint forward.

Token activations are sequence-sharded. QKV all-to-all gathers sequence while
scattering heads; the reverse exchange restores the local sequence shard.
Normal PPO/ref attention forwards are restored unchanged after this call.
"""

from contextlib import contextmanager
import math
import os
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F

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


def _pack_qkv(q, k, v, size):
    batch, width, _, head_dim = q.shape
    heads = tuple(x.shape[2] // size for x in (q, k, v))
    # The leading dimension selects the destination rank. Within each rank's
    # payload, keep separate Q, K and V head ranges to support GQA.
    packed = torch.cat(
        tuple(x.reshape(batch, width, size, h, head_dim).permute(2, 0, 1, 3, 4)
              for x, h in zip((q, k, v), heads, strict=True)),
        dim=3,
    )
    return packed, heads


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _dense_attention(q, k, v, mask, scale, *, pv_rows=None):
    groups = q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    weights = torch.matmul(q, k.transpose(-1, -2)) * scale
    weights = (weights + mask.to(weights.dtype)).softmax(-1, dtype=torch.float32).to(q.dtype)
    if pv_rows is not None:
        prefix, trunk = pv_rows
        width = k.shape[-2]
        end = trunk + weights.shape[-2] - prefix
        # CANN's BF16 value matmul can pick different accumulation with fewer
        # queries. Restore the full geometry and physical row offsets for P@V.
        full = weights.new_zeros((*weights.shape[:-2], width, width))
        full[..., :prefix, :] = weights[..., :prefix, :]
        full[..., trunk:end, :] = weights[..., prefix:, :]
        weights = full
    return torch.matmul(weights, v).transpose(1, 2)


def _causal_attention(q, k, v, scale):
    tail_only = q.shape[1] < k.shape[1]
    if q.device.type == "npu" and q.dtype in (torch.float16, torch.bfloat16):
        # Reuse the same causal NPU kernel and compressed mask as ordinary FA2.
        from transformers.integrations.npu_flash_attention import npu_flash_attn_func

        if tail_only:
            from torch_npu import npu_fusion_attention
            from transformers.integrations.npu_flash_attention import (
                DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE,
                get_attn_mask_npu,
            )

            # Shorter Q is the suffix of K/V. Force down-right alignment
            # independently of the square FA2 environment setting.
            return npu_fusion_attention(
                q, k, v, q.shape[2], "BSND", keep_prob=1.0, scale=scale,
                atten_mask=get_attn_mask_npu(q.device),
                sparse_mode=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE,
            )[0]
        return npu_flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
    groups = q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    mask = None
    if tail_only:
        from torch.nn.attention.bias import causal_lower_right

        mask = causal_lower_right(q.shape[-2], k.shape[-2])
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=not tail_only, scale=scale,
    ).transpose(1, 2)


def _segment_masks(packed, dtype, *, dense_prefix=False):
    device = packed.input_ids.device
    width = packed.input_ids.shape[1]
    segments = []
    for request, readouts in zip(packed.requests, packed.branch_rows, strict=True):
        trunk_length = len(request.trunk_ids)
        anchors, starts = [], []
        for anchor, *tail in readouts:
            if tail:
                anchors.extend([anchor] * len(tail))
                starts.extend([tail[0] - trunk_length] * len(tail))
        prefix = max((rows[0] for rows in readouts) if dense_prefix else anchors, default=-1) + 1
        if not prefix:
            segments.append((trunk_length, 0, None))
            continue
        trunk_columns = torch.arange(prefix, device=device)
        tail_columns = torch.arange(len(starts), device=device)
        trunk_allowed = trunk_columns[None, :] <= torch.tensor(anchors, dtype=torch.long, device=device)[:, None]
        tail_allowed = (
            (tail_columns[None, :] >= torch.tensor(starts, dtype=torch.long, device=device)[:, None])
            & (tail_columns[None, :] <= tail_columns[:, None])
        )
        if dense_prefix:
            # Preserve the original key span and reduction length for every
            # query whose output can feed a branch in a later layer.
            allowed = torch.zeros((prefix + len(starts), width), dtype=torch.bool, device=device)
            allowed[:prefix, :prefix] = trunk_columns[None, :] <= trunk_columns[:, None]
            allowed[prefix:, :prefix] = trunk_allowed
            allowed[prefix:, trunk_length:trunk_length + len(starts)] = tail_allowed
        else:
            allowed = torch.cat((trunk_allowed, tail_allowed), dim=-1)
        allowed = allowed[None, None, :, :]
        mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        segments.append((trunk_length, prefix, mask))
    return tuple(segments)


def _segmented_attention(q, k, v, segments, scale, *, dense_prefix=False, preserve_pv=False):
    output = torch.zeros_like(q)
    for i, (trunk, prefix, mask) in enumerate(segments):
        row = slice(i, i + 1)
        if dense_prefix and mask is not None:
            if prefix < trunk:
                output[row, prefix:trunk] = _causal_attention(
                    q[row, prefix:trunk], k[row, :trunk], v[row, :trunk], scale,
                )
            tail = slice(trunk, trunk + mask.shape[-2] - prefix)
            dense_queries = torch.cat((q[row, :prefix], q[row, tail]), dim=1)
            attended = _dense_attention(
                dense_queries, k[row], v[row], mask, scale,
                pv_rows=(prefix, trunk) if preserve_pv else None,
            )
            output[row, :prefix] = attended[:, :prefix]
            output[row, tail] = attended[:, tail if preserve_pv else slice(prefix, None)]
            continue
        output[row, :trunk] = _causal_attention(q[row, :trunk], k[row, :trunk], v[row, :trunk], scale)
        if mask is not None:
            tail = slice(trunk, trunk + mask.shape[-2])
            branch_keys = torch.cat((k[row, :prefix], k[row, tail]), dim=1)
            branch_values = torch.cat((v[row, :prefix], v[row, tail]), dim=1)
            output[row, tail] = _dense_attention(q[row, tail], branch_keys, branch_values, mask, scale)
    return output


@contextmanager
def tree_attention(
    model, mask, group, *, causal_rows=(), tree_rows=(), pack_qkv=False, segments=None,
    dense_prefix=False, preserve_pv=False,
):
    """Override instance methods only, preserving global upstream SP patches."""
    changed = []
    size = dist.get_world_size(group)
    causal_index = tree_index = None
    if causal_rows and tree_rows:
        causal_index = torch.tensor(causal_rows, device=mask.device)
        tree_index = torch.tensor(tree_rows, device=mask.device)

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
        if pack_qkv:
            send, heads = _pack_qkv(q, k, v, size)
            del q, k, v
            received = torch.empty_like(send)
            dist.all_to_all_single(received, send, group=group)
            del send
            # The leading dimension now selects the source rank. Concatenate
            # the local sequence blocks in rank order as in exchange().
            q, k, v = (torch.cat(chunk.unbind(0), dim=1) for chunk in received.split(heads, dim=3))
            del received
        else:
            q, k, v = (exchange(x, 2, 1, group) for x in (q, k, v))
        if segments is not None:
            output = _segmented_attention(
                q, k, v, segments, attn.scaling, dense_prefix=dense_prefix, preserve_pv=preserve_pv,
            )
        elif mask is None:
            output = _causal_attention(q, k, v, attn.scaling)
        elif causal_index is not None:
            output = torch.empty_like(q)
            cq, ck, cv = (x.index_select(0, causal_index) for x in (q, k, v))
            output.index_copy_(0, causal_index, _causal_attention(cq, ck, cv, attn.scaling))
            tq, tk, tv = (x.index_select(0, tree_index) for x in (q, k, v))
            output.index_copy_(0, tree_index, _dense_attention(tq, tk, tv, mask, attn.scaling))
        else:
            output = _dense_attention(q, k, v, mask, attn.scaling)
        output = exchange(output.contiguous(), 1, 2, group)
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
    return_target_logz=True,
    capture_empty_trunk_features=True, trunk_feature_prefix_only=False,
    max_packed_tokens=8192, max_mask_elements=16777216,
    use_causal_fast_path=None, use_packed_qkv=None, use_segmented_attention=None,
    use_segmented_dense_prefix=None, use_preserve_pv=None,
):
    for t in (actor_temperature, target_temperature):
        if not math.isfinite(t) or t <= 0:
            raise ValueError("scoring temperatures must be finite and positive")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    cfg = model.config
    validate_ulysses_model(model, size)
    parameter = next(model.parameters())
    if use_causal_fast_path is None:
        use_causal_fast_path = os.environ.get("VERL_SPECO_JOINT_CAUSAL_FAST_PATH") == "1"
    if use_packed_qkv is None:
        use_packed_qkv = os.environ.get("VERL_SPECO_JOINT_PACK_QKV") == "1"
    if use_segmented_attention is None:
        use_segmented_attention = os.environ.get("VERL_SPECO_JOINT_SEGMENTED_ATTENTION") == "1"
    if use_segmented_dense_prefix is None:
        use_segmented_dense_prefix = os.environ.get("VERL_SPECO_JOINT_SEGMENTED_DENSE_PREFIX") == "1"
    if use_preserve_pv is None:
        use_preserve_pv = os.environ.get("VERL_SPECO_JOINT_PRESERVE_PV") == "1"
    if use_preserve_pv and len(requests) != 1:
        # Only one local row has been validated with the original PV geometry.
        use_preserve_pv = use_causal_fast_path = use_segmented_attention = use_segmented_dense_prefix = False
    elif use_preserve_pv:
        use_preserve_pv = bool(requests[0].branches)
    dense_prefix = use_segmented_dense_prefix or use_preserve_pv
    segmented = use_segmented_attention or dense_prefix
    tree_rows = tuple(i for i, request in enumerate(requests) if request.branches)
    causal_rows = (
        tuple(i for i, request in enumerate(requests) if not request.branches)
        if use_causal_fast_path and not segmented else ()
    )
    # Valid trunk queries cannot see right padding, and padded outputs are never returned.
    packed = pack_joint_batch(
        requests, device=parameter.device, dtype=parameter.dtype, vocab_size=cfg.vocab_size,
        max_packed_tokens=max_packed_tokens, max_mask_elements=max_mask_elements,
        sequence_multiple=size,
        omit_trunk_only_masks=bool(causal_rows),
        build_attention_mask=not segmented,
    )
    segments = _segment_masks(packed, parameter.dtype, dense_prefix=dense_prefix) if segmented else None
    width = packed.input_ids.shape[1] // size
    start = rank * width
    states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with tree_attention(
            model, packed.attention_mask, group, causal_rows=causal_rows, tree_rows=tree_rows,
            pack_qkv=use_packed_qkv, segments=segments, dense_prefix=dense_prefix, preserve_pv=use_preserve_pv,
        ):
            output = model(
                input_ids=packed.input_ids[:, start:start + width],
                position_ids=packed.position_ids[:, start:start + width],
                attention_mask=None, use_cache=False, return_dict=True,
                output_hidden_states=(not (use_causal_fast_path or segmented) or bool(
                    tree_rows or (capture_empty_trunk_features and hidden_state_indices)
                )),
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
            flat_logz = None
            if return_target_logz:
                # Only normalize branch rows owned by this SP rank.
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
                    flat_logz[offset:end] if flat_logz is not None else None,
                    flat_hidden[offset:end],
                ))
                offset = end
        feature_rows = (max(b.anchor for b in request.branches) + 1
                        if trunk_feature_prefix_only and request.branches else n)
        features = {
            index: _gather_rows(output.hidden_states[index][i], range(feature_rows), start, group)
            for index in hidden_state_indices
        } if request.branches or capture_empty_trunk_features else {}
        results.append(JointScores(request.sample_id, log_probs, tuple(branches), features, entropy))
    mask_elements = packed.attention_mask.numel() if packed.attention_mask is not None else 0
    if segments is not None:
        mask_elements = sum(mask.numel() for _, _, mask in segments if mask is not None)
    return tuple(results), {
        "model_forward_calls": 1, "logical_rows": packed.logical_rows,
        "independent_rows": packed.independent_rows, "padded_rows": packed.input_ids.numel(),
        "mask_elements": mask_elements,
        "sp_size": size,
    }
