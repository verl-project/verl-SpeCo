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

from verl_speco.joint_scoring import (
    pack_joint_batch,
    score_joint_batch,
)


ULYSSES_ENV_FLAGS = {
    "use_causal_fast_path": "VERL_SPECO_JOINT_CAUSAL_FAST_PATH",
    "use_packed_qkv": "VERL_SPECO_JOINT_PACK_QKV",
    "use_segmented_attention": "VERL_SPECO_JOINT_SEGMENTED_ATTENTION",
    "use_segmented_dense_prefix": "VERL_SPECO_JOINT_SEGMENTED_DENSE_PREFIX",
    "use_preserve_pv": "VERL_SPECO_JOINT_PRESERVE_PV",
}


def resolve_ulysses_flag(name, value):
    return os.environ.get(ULYSSES_ENV_FLAGS[name]) == "1" if value is None else value


def validate_ulysses_model(model, size):
    cfg = model.config
    if (
        cfg.num_attention_heads % size
        or cfg.num_attention_heads % cfg.num_key_value_heads
    ):
        raise ValueError(
            "Q heads must divide SP and be an integer multiple of KV heads"
        )
    # Adaptive RoPE updates its frequencies from the largest observed position.
    # Local sequence shards can observe different maxima; do not silently use
    # rank-dependent frequencies. The deployed Qwen3-8B uses static RoPE.
    for module in model.modules():
        rope_type = str(getattr(module, "rope_type", "default"))
        if "dynamic" in rope_type or rope_type == "longrope":
            raise ValueError(
                "joint Ulysses adaptive RoPE requires a separate global-frequency implementation"
            )


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
        tuple(
            x.reshape(batch, width, size, h, head_dim).permute(2, 0, 1, 3, 4)
            for x, h in zip((q, k, v), heads, strict=True)
        ),
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
    weights = (
        (weights + mask.to(weights.dtype)).softmax(-1, dtype=torch.float32).to(q.dtype)
    )
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
                q,
                k,
                v,
                q.shape[2],
                "BSND",
                keep_prob=1.0,
                scale=scale,
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
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=not tail_only,
        scale=scale,
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
        prefix = (
            max((rows[0] for rows in readouts) if dense_prefix else anchors, default=-1)
            + 1
        )
        if not prefix:
            segments.append((trunk_length, 0, None))
            continue
        trunk_columns = torch.arange(prefix, device=device)
        tail_columns = torch.arange(len(starts), device=device)
        trunk_allowed = (
            trunk_columns[None, :]
            <= torch.tensor(anchors, dtype=torch.long, device=device)[:, None]
        )
        tail_allowed = (
            tail_columns[None, :]
            >= torch.tensor(starts, dtype=torch.long, device=device)[:, None]
        ) & (tail_columns[None, :] <= tail_columns[:, None])
        if dense_prefix:
            # Preserve the original key span and reduction length for every
            # query whose output can feed a branch in a later layer.
            allowed = torch.zeros(
                (prefix + len(starts), width), dtype=torch.bool, device=device
            )
            allowed[:prefix, :prefix] = trunk_columns[None, :] <= trunk_columns[:, None]
            allowed[prefix:, :prefix] = trunk_allowed
            allowed[prefix:, trunk_length : trunk_length + len(starts)] = tail_allowed
        else:
            allowed = torch.cat((trunk_allowed, tail_allowed), dim=-1)
        allowed = allowed[None, None, :, :]
        mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        segments.append((trunk_length, prefix, mask))
    return tuple(segments)


def _segmented_attention(
    q, k, v, segments, scale, *, dense_prefix=False, preserve_pv=False
):
    output = torch.zeros_like(q)
    for i, (trunk, prefix, mask) in enumerate(segments):
        row = slice(i, i + 1)
        if dense_prefix and mask is not None:
            if prefix < trunk:
                output[row, prefix:trunk] = _causal_attention(
                    q[row, prefix:trunk],
                    k[row, :trunk],
                    v[row, :trunk],
                    scale,
                )
            tail = slice(trunk, trunk + mask.shape[-2] - prefix)
            dense_queries = torch.cat((q[row, :prefix], q[row, tail]), dim=1)
            attended = _dense_attention(
                dense_queries,
                k[row],
                v[row],
                mask,
                scale,
                pv_rows=(prefix, trunk) if preserve_pv else None,
            )
            output[row, :prefix] = attended[:, :prefix]
            output[row, tail] = attended[
                :, tail if preserve_pv else slice(prefix, None)
            ]
            continue
        output[row, :trunk] = _causal_attention(
            q[row, :trunk], k[row, :trunk], v[row, :trunk], scale
        )
        if mask is not None:
            tail = slice(trunk, trunk + mask.shape[-2])
            branch_keys = torch.cat((k[row, :prefix], k[row, tail]), dim=1)
            branch_values = torch.cat((v[row, :prefix], v[row, tail]), dim=1)
            output[row, tail] = _dense_attention(
                q[row, tail], branch_keys, branch_values, mask, scale
            )
    return output


@contextmanager
def tree_attention(
    model,
    mask,
    group,
    *,
    causal_rows=(),
    tree_rows=(),
    pack_qkv=False,
    segments=None,
    dense_prefix=False,
    preserve_pv=False,
):
    """Override instance methods only, preserving global upstream SP patches."""
    changed = []
    size = dist.get_world_size(group)
    causal_index = tree_index = None
    if causal_rows and tree_rows:
        causal_index = torch.tensor(causal_rows, device=mask.device)
        tree_index = torch.tensor(tree_rows, device=mask.device)

    def forward(
        attn, hidden_states, position_embeddings=None, attention_mask=None, **kwargs
    ):
        if torch.is_grad_enabled():
            raise RuntimeError("joint Ulysses attention is inference-only")
        if (
            kwargs.get("past_key_values") is not None
            or kwargs.get("past_key_value") is not None
        ):
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
            q, k, v = (
                torch.cat(chunk.unbind(0), dim=1)
                for chunk in received.split(heads, dim=3)
            )
            del received
        else:
            q, k, v = (exchange(x, 2, 1, group) for x in (q, k, v))
        if segments is not None:
            output = _segmented_attention(
                q,
                k,
                v,
                segments,
                attn.scaling,
                dense_prefix=dense_prefix,
                preserve_pv=preserve_pv,
            )
        elif mask is None:
            output = _causal_attention(q, k, v, attn.scaling)
        elif causal_index is not None:
            output = torch.empty_like(q)
            cq, ck, cv = (x.index_select(0, causal_index) for x in (q, k, v))
            output.index_copy_(
                0, causal_index, _causal_attention(cq, ck, cv, attn.scaling)
            )
            tq, tk, tv = (x.index_select(0, tree_index) for x in (q, k, v))
            output.index_copy_(
                0, tree_index, _dense_attention(tq, tk, tv, mask, attn.scaling)
            )
        else:
            output = _dense_attention(q, k, v, mask, attn.scaling)
        output = exchange(output.contiguous(), 1, 2, group)
        output = attn.o_proj(output.reshape(*hidden_states.shape[:-1], -1))
        return output, None

    try:
        for layer in model.modules():
            if all(
                hasattr(layer, x)
                for x in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "scaling")
            ):
                own = "forward" in layer.__dict__
                changed.append((layer, own, layer.__dict__.get("forward")))
                layer.forward = MethodType(forward, layer)
        if len(changed) != model.config.num_hidden_layers:
            raise ValueError(
                "joint Ulysses requires one supported full-attention module per layer"
            )
        yield
    finally:
        for layer, own, original in reversed(changed):
            if own:
                layer.forward = original
            else:
                del layer.forward


def prepare_ulysses_batch(
    model, requests, group, pack_options, needs_features, **options
):
    """Prepare only the SP-specific layout and scoped attention override."""
    unknown = options.keys() - ULYSSES_ENV_FLAGS.keys()
    if unknown:
        raise TypeError(f"unexpected Ulysses scoring options: {sorted(unknown)}")
    flags = {
        name: resolve_ulysses_flag(name, options.get(name))
        for name in ULYSSES_ENV_FLAGS
    }
    size = dist.get_world_size(group)
    validate_ulysses_model(model, size)
    if flags["use_preserve_pv"] and len(requests) != 1:
        # Only one local row has been validated with the original PV geometry.
        for name in (
            "use_preserve_pv",
            "use_causal_fast_path",
            "use_segmented_attention",
            "use_segmented_dense_prefix",
        ):
            flags[name] = False
    elif flags["use_preserve_pv"]:
        flags["use_preserve_pv"] = bool(requests[0].branches)
    dense_prefix = flags["use_segmented_dense_prefix"] or flags["use_preserve_pv"]
    segmented = flags["use_segmented_attention"] or dense_prefix
    tree_rows = tuple(i for i, request in enumerate(requests) if request.branches)
    causal_rows = (
        tuple(i for i, request in enumerate(requests) if not request.branches)
        if flags["use_causal_fast_path"] and not segmented
        else ()
    )
    packed = pack_joint_batch(
        requests,
        sequence_multiple=size,
        omit_trunk_only_masks=bool(causal_rows),
        build_attention_mask=not segmented,
        **pack_options,
    )
    segments = (
        _segment_masks(packed, pack_options["dtype"], dense_prefix=dense_prefix)
        if segmented
        else None
    )
    attention = tree_attention(
        model,
        packed.attention_mask,
        group,
        causal_rows=causal_rows,
        tree_rows=tree_rows,
        pack_qkv=flags["use_packed_qkv"],
        segments=segments,
        dense_prefix=dense_prefix,
        preserve_pv=flags["use_preserve_pv"],
    )
    elements = packed.attention_mask.numel() if packed.attention_mask is not None else 0
    if segments is not None:
        elements = sum(mask.numel() for _, _, mask in segments if mask is not None)
    return (
        packed,
        attention,
        not (flags["use_causal_fast_path"] or segmented) or needs_features,
        {"mask_elements": elements, "sp_size": size},
    )


def score_ulysses_joint_batch(model, requests, group, **options):
    return score_joint_batch(model, requests, _sp_group=group, **options)
