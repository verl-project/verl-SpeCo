# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Trace anchors and first-rejection objectives shared by DFlash and DSpark."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from verl_speco.integration.verify_trace import TRACE_KEY, validate_verify_trace


def preprocess_trace(
    item, *, start, end, layout, capacity, device, aligned_next_tokens=False
):
    trace = item.get(TRACE_KEY)
    if trace is None:
        raise ValueError("Draft-OPD requires a captured verification trace")
    plan = torch.full((end - start, 3), -1, dtype=torch.long, device=device)
    origin = (
        int(item.get("_verl_feature_start", item.get("feature_start", 0)))
        + start
        - trace.get("token_offset", 0)
    )
    turns = trace["turns"] if "turns" in trace else [trace]
    last_end = -1
    for turn in turns:
        validate_verify_trace(turn, layout=layout, capacity=capacity)
        if turn["prompt_length"] < last_end:
            raise ValueError("Draft-OPD turn traces overlap")
        last_end = turn["prompt_length"] + turn["response_length"]
        for step in turn["steps"]:
            anchor = step["anchor"] - origin
            if step["drafts"] and step["commit"] and 0 <= anchor < end - start:
                visible = min(
                    step["commit"],
                    step["drafts"],
                    end - start - anchor - 1 + int(aligned_next_tokens),
                )
                plan[anchor] = torch.tensor(
                    [step["accepted"], visible, int(step["accepted"] < step["drafts"])],
                    device=device,
                )
    return plan


class DraftOPDObjective(nn.Module):
    def __init__(
        self, config, *, layout, loss_mode, per_position_loss_weight, temperature=1.0
    ):
        super().__init__()
        self.layout = layout
        self.temperature = float(temperature)
        self.planner = config.get("anchor_planner", "mixed")
        self.random_ratio = float(config.get("random_anchor_ratio", 0.25))
        self.loss_type = config.get("loss", "e2e_tv")
        self.reject_weight = float(config.get("first_reject_weight", 2.0))
        self.chunk_size = int(config.get("chunk_size", 1024))
        if self.planner not in {"random", "trace", "mixed"} or self.loss_type not in {
            "ce",
            "e2e_tv",
        }:
            raise ValueError("Invalid Draft-OPD anchor planner or loss")
        if (
            not 0 <= self.random_ratio <= 1
            or not math.isfinite(self.reject_weight)
            or self.reject_weight <= 0
            or self.chunk_size <= 0
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Invalid Draft-OPD ratio, rejection weight or chunk size")
        if loss_mode != "full_vocab" or per_position_loss_weight not in {
            "fixed_exp_decay",
            "vat",
        }:
            raise ValueError("Draft-OPD requires full_vocab and fixed_exp_decay or vat")

    def plan(self, anchors, keep, trace):
        if trace is None:
            raise ValueError("Draft-OPD requires preprocessed trace coordinates")
        bsz, budget = anchors.shape
        random_count = (
            budget
            if self.planner == "random"
            else 0
            if self.planner == "trace"
            else math.ceil(budget * self.random_ratio)
        )
        accepted = torch.full_like(anchors, -1)
        visible = torch.full_like(anchors, torch.iinfo(torch.long).max)
        reject = torch.zeros_like(keep)
        result = anchors.clone()
        result_keep = keep.clone()
        for row in range(bsz):
            if self.planner == "mixed":
                # Upstream samplers sort coordinates; choose the random quota
                # uniformly from valid slots before packing the trace slots.
                valid_slots = torch.nonzero(keep[row], as_tuple=True)[0]
                random_slots = valid_slots[
                    torch.randperm(valid_slots.numel(), device=anchors.device)[
                        :random_count
                    ]
                ]
                result_keep[row, :random_count] = False
                result[row, : random_slots.numel()] = anchors[row, random_slots]
                result_keep[row, : random_slots.numel()] = True
            candidates = torch.nonzero(trace[row, :, 1] > 0, as_tuple=True)[0]
            selected = candidates[
                torch.randperm(candidates.numel(), device=anchors.device)[
                    : budget - random_count
                ]
            ]
            result_keep[row, random_count:] = False
            count = selected.numel()
            slots = slice(random_count, random_count + count)
            result[row, slots] = selected
            result_keep[row, slots] = True
            accepted[row, slots] = trace[row, selected, 0]
            visible[row, slots] = trace[row, selected, 1]
            reject[row, slots] = trace[row, selected, 2].bool()
        return result, result_keep, accepted, visible, reject

    def mask(self, base_mask, visible):
        offset = int(self.layout == "dflash")
        j = torch.arange(base_mask.shape[-1], device=base_mask.device) - offset
        return base_mask * (j < visible.unsqueeze(-1))

    def weights(self, weights, accepted, reject, gamma=None):
        offset = int(self.layout == "dflash")
        j = torch.arange(weights.shape[-1], device=weights.device) - offset
        first = (j == accepted.unsqueeze(-1)) & reject.unsqueeze(-1)
        weights = weights * torch.where(first, self.reject_weight, 1.0)
        if gamma is not None and gamma > 0:
            decay = torch.exp(-j.clamp_min(0).float() / gamma)
            weights = weights * torch.where(accepted.unsqueeze(-1) < 0, decay, 1.0)
        return weights

    def _tv(self, logits, target_hidden, weight):
        with torch.no_grad():
            target_probs = F.softmax(
                F.linear(target_hidden, weight).float() / self.temperature, -1
            )
        draft_probs = F.softmax(logits.float() / self.temperature, -1)
        return (target_probs - draft_probs).abs().sum(-1) * 0.5

    def forward(
        self,
        logits,
        target_hidden,
        weight,
        base_mask,
        active_mask,
        *,
        position_weights,
    ):
        if target_hidden is None:
            raise ValueError("Draft-OPD E2E TV requires final target hidden states")
        tv = torch.zeros_like(base_mask)
        terms = []
        for start in range(0, logits.shape[0], self.chunk_size):
            terms.append(
                checkpoint(
                    self._tv,
                    logits[start : start + self.chunk_size],
                    target_hidden[start : start + self.chunk_size],
                    weight,
                    use_reentrant=False,
                )
            )
        if terms:
            tv.reshape(-1)[active_mask] = torch.cat(terms)
        offset = int(self.layout == "dflash")
        valid = base_mask[..., offset:].bool().long().cumprod(-1)
        survival = (1.0 - tv[..., offset:]).clamp(0, 1).cumprod(-1) * valid
        counts = valid.sum(-1)
        blocks = counts > 0
        # Weight prefix failures, not the TV probabilities inside the product.
        # Keep the count denominator so a single first rejection is amplified too.
        failures = (valid - survival) * position_weights[..., offset:]
        losses = failures.sum(-1) / counts.clamp_min(1)
        return (losses * blocks).sum() / blocks.sum().clamp_min(1) + logits[:0].sum()

    def metrics(self, loss, accepted, base_mask):
        valid = base_mask.bool().any(-1)
        traced = valid & (accepted >= 0)
        return {
            "opd/batch_count": loss.new_tensor(1.0),
            "opd/loss": loss.detach(),
            "opd/trace_blocks": traced.sum().float(),
            "opd/random_blocks": (valid & ~traced).sum().float(),
            "opd/visible_tokens": base_mask.sum().detach(),
        }
