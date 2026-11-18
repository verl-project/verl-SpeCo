# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""A bounded AIR-style MTP cycle behind SpeCo's persistent engine lifecycle.

Tokens shift by k, target distributions by k, positions by k-1. Unlike AIR's
prompt mask, SpeCo supervision stays positive throughout this implementation.
"""

import itertools
import math
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl_speco.backends.vllm_ascend_options import resolve_optimization_config
from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches
from verl_speco.ops.deepseek_v4_loss import vocab_parallel_log_probs
from verl_speco.ops.deepseek_v4_mtp_loss import (
    select_valid_loss_tokens,
    vocab_parallel_eagle_loss,
)


@contextmanager
def first_backward_audit(model, logger):
    """Observe hooks before ZeRO can clear gradients; never inspect param.grad."""
    if getattr(model, "_speco_mtp_backward_audited", False):
        yield
        return
    from verl_speco.trainer.deepseek_v4_diagnostics import capture_dspark_gradient_probe

    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    seen, handles = {}, []

    def hook(name):
        def observe(gradient):
            finite, maximum, _ = capture_dspark_gradient_probe(gradient)
            if name in seen:
                old_finite, old_maximum = seen[name]
                finite, maximum = (
                    old_finite & finite,
                    torch.maximum(old_maximum, maximum),
                )
            seen[name] = finite, maximum
            return gradient

        return observe

    for name, value in model.named_parameters():
        if value.requires_grad:
            handles.append(value.register_hook(hook(name)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(expected - seen.keys())
    if missing:
        raise RuntimeError(f"MTP backward omitted trainable parameters: {missing}")
    if not seen or not bool(torch.stack([value[0] for value in seen.values()]).all()):
        raise RuntimeError("MTP backward produced missing or non-finite gradients")
    nonzero = int(torch.stack([value[1] > 0 for value in seen.values()]).sum())
    if not nonzero:
        raise RuntimeError("MTP backward produced only zero gradients")
    logger.info(
        "MTP first-backward audit: parameters=%d present=%d nonzero=%d nonfinite=0",
        len(expected),
        len(seen),
        nonzero,
    )
    model._speco_mtp_backward_audited = True


def valid_token_counts(batch, ttt_length):
    mask = batch["loss_mask"]
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("MTP loss mask must use SpeCo binary supervision")
    padded = F.pad(mask, (0, ttt_length), value=0)
    return [
        float(padded[..., k : k + mask.shape[-1]].sum())
        for k in range(1, ttt_length + 1)
    ]


def step_inputs(batch, k, ttt_length):
    """Shift IDs without incorrectly shifting the hidden state's RoPE position."""
    length = batch["input_ids"].shape[-1]
    return (
        F.pad(batch["input_ids"], (0, ttt_length), value=0)[..., k : k + length],
        F.pad(batch["positions"], (0, ttt_length), value=0)[
            ..., k - 1 : k - 1 + length
        ],
        F.pad(batch["loss_mask"], (0, ttt_length), value=0)[..., k : k + length],
    )


def token_loss_scale(local_count, global_count, dp_size, window_size):
    # AIR's strict_token_loss_scale, including epsilon cancellation of each
    # local mean. DeepSpeed subsequently divides by GAS and by DP size.
    if global_count <= 0:
        return 0.0
    return dp_size * window_size * (local_count + 1e-6) / (global_count + 1e-6)


def run_mtp_training(
    *,
    model_engine,
    target_lm_head,
    dataloader,
    sampler,
    cfg,
    logger,
    tracker,
    device,
    parallel_topology,
    checkpoint_dir,
    initial_global_step=0,
    initial_global_examples=0,
    save_checkpoints=False,
):
    del sampler, checkpoint_dir, initial_global_examples
    if save_checkpoints:
        raise ValueError("MTP checkpoints are owned by the SpeCo outer lifecycle")
    if cfg.train.loss_type != "eagle":
        raise ValueError("MTP currently supports Eagle KL")
    options = resolve_optimization_config(cfg)
    steps = int(cfg.train.max_optimizer_steps)
    gas = int(cfg.deepspeed.gradient_accumulation_steps)
    ttt = int(cfg.train.ttt_length)
    dp = parallel_topology.data_parallel_size
    group = parallel_topology.tensor_parallel_group
    model = model_engine.module
    probes = [
        p
        for p in model.parameters()
        if p.requires_grad and p.dtype == torch.float32 and p.numel() <= 4096
    ]
    if not probes:
        probes = [next(p for p in model.parameters() if p.requires_grad)]
    iterator, completed, examples = iter(dataloader), 0, 0
    observed_update, observed_lr = False, False
    while completed < steps:
        window = list(itertools.islice(iterator, gas))
        if not window:
            raise RuntimeError("MTP loader exhausted before planned optimizer updates")
        counts = [valid_token_counts(batch, ttt) for batch in window]
        global_counts = torch.tensor(
            [sum(row[k] for row in counts) for k in range(ttt)],
            dtype=torch.float32,
            device=device,
        )
        if dist.is_initialized() and dp > 1:
            dist.all_reduce(global_counts, group=parallel_topology.data_parallel_group)
        global_counts = global_counts.cpu().tolist()
        if not any(global_counts):
            raise ValueError("MTP accumulation window has no supervised future targets")
        numerators = torch.zeros(ttt, dtype=torch.float32, device=device)
        for i, batch in enumerate(window):
            boundary = i == len(window) - 1
            model_engine.set_gradient_accumulation_boundary(boundary)
            host = {key: batch[key] for key in ("input_ids", "positions", "loss_mask")}
            hidden = batch["hidden_states"].to(device)
            with torch.no_grad():
                probabilities = (
                    vocab_parallel_log_probs(
                        target_lm_head.local_logits(hidden), process_group=group
                    )
                    .exp()
                    .to(target_lm_head.dtype)
                )
                probabilities = F.pad(probabilities, (0, 0, 0, ttt), value=0)
            current = hidden.requires_grad_(True)
            losses = []
            for step in range(ttt):
                k = step + 1
                ids, positions, mask = step_inputs(host, k, ttt)
                mask = mask.to(device)
                new_hidden = model_engine(
                    input_ids=ids.to(device),
                    positions=positions.to(device),
                    positions_cpu=positions,
                    hidden_states=current,
                    attention_mask=None,
                    spec_step_idx=step,
                )
                teacher = probabilities[:, k : k + hidden.shape[1]]
                loss_hidden = new_hidden
                if options.valid_token_logits_only and counts[i][step] > 0:
                    loss_hidden, teacher, _, mask, _ = select_valid_loss_tokens(
                        loss_hidden, teacher, None, mask
                    )
                logits = model_engine.compute_local_logits(
                    loss_hidden, spec_step_idx=step
                ).reshape(*teacher.shape)
                loss = vocab_parallel_eagle_loss(
                    logits,
                    teacher,
                    mask,
                    step,
                    process_group=group,
                    token_chunk_size=int(
                        getattr(options, "vocab_parallel_loss_token_chunk_size", 0)
                    ),
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("MTP reported a non-finite loss")
                numerators[step] += loss.detach().float() * (counts[i][step] + 1e-6)
                scale = token_loss_scale(
                    counts[i][step], global_counts[step], dp, len(window)
                )
                losses.append(loss * (gas / len(window)) * scale)
                current = (
                    new_hidden.detach() if cfg.train.detach_hs_graph else new_hidden
                )
            with first_backward_audit(model, logger):
                model_engine.backward(sum(losses))
            before = (
                [p.detach().reshape(-1)[:64].clone() for p in probes]
                if boundary
                else []
            )
            rates = model_engine.get_lr() if boundary else []
            model_engine.step()
            examples += int(hidden.shape[0])
            if boundary:
                refresh_fp32_weight_caches(model)
                delta = max(
                    float((p.detach().reshape(-1)[:64] - old).abs().max())
                    for p, old in zip(probes, before, strict=True)
                )
                if not math.isfinite(delta):
                    raise RuntimeError("MTP update produced non-finite parameters")
                observed_update |= delta > 0
                observed_lr |= any(float(rate) > 0 for rate in rates)
            del losses, probabilities, current, new_hidden
        if dist.is_initialized() and dp > 1:
            dist.all_reduce(numerators, group=parallel_topology.data_parallel_group)
        per_step = [
            float(numerators[k]) / (global_counts[k] + 1e-6) for k in range(ttt)
        ]
        completed += 1
        metrics = {
            "train_loss": sum(per_step),
            "optimizer_lr": float(rates[0]),
            "optimizer_probe_delta": delta,
            **{f"mtp_loss_t{k + 1}": value for k, value in enumerate(per_step)},
            **{
                f"mtp_valid_tokens_t{k + 1}": value
                for k, value in enumerate(global_counts)
            },
        }
        tracker.log_metrics(metrics, step=initial_global_step + completed)
        logger.info(
            "MTP optimizer step=%s metrics=%s", initial_global_step + completed, metrics
        )
    return {
        "global_step": initial_global_step + completed,
        "global_examples": examples,
        "observed_update": observed_update,
        "observed_nonzero_optimizer_lr": observed_lr,
    }
