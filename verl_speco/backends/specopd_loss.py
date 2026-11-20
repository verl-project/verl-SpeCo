# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distribution losses with fixed sampled histories and prefix masks."""

import math

import torch
import torch.nn.functional as F

from verl_speco.backends.opd_distribution import (
    token_total_variation,
    use_fused_opd_loss,
)
from verl_speco.ops.dspark_fused_loss import (
    fused_label_cross_entropy,
    fused_total_variation_and_teacher_ce,
)


def specopd_distribution_loss(
    draft_logits,
    target_logits,
    valid_mask,
    *,
    proposal_temperature=1.0,
    target_temperature=1.0,
    position_weights=None,
    distribution_loss_impl="auto",
    use_l1=False,
    distribution_weight=1.0,
    ce_weight=0.0,
):
    if (
        draft_logits.shape[:-1] != valid_mask.shape
        or target_logits.shape[:-1] != valid_mask.shape
    ):
        raise ValueError("SpecOPD logits and prefix mask dimensions differ")
    if (
        valid_mask.dtype != torch.bool
        or valid_mask.ndim != 2
        or (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any()
    ):
        raise ValueError("valid_mask must describe contiguous branch prefixes")
    if not all(
        math.isfinite(t) and t > 0 for t in (proposal_temperature, target_temperature)
    ):
        raise ValueError("positive scoring temperatures required")
    joint_ce = (
        ce_weight > 0
        and proposal_temperature == target_temperature == 1.0
        and draft_logits.numel() > 0
        and use_fused_opd_loss(draft_logits.device, distribution_loss_impl)
    )
    ce = None
    if joint_ce:
        tv, ce = fused_total_variation_and_teacher_ce(
            draft_logits.flatten(0, 1), target_logits.flatten(0, 1)
        )
        tv, ce = tv.reshape_as(valid_mask), ce.reshape_as(valid_mask)
        alpha = 1.0 - tv
    else:
        tv = token_total_variation(
            draft_logits,
            target_logits,
            proposal_temperature=proposal_temperature,
            target_temperature=target_temperature,
            implementation=distribution_loss_impl,
        )
        alpha = 1.0 - tv
    alpha = torch.where(valid_mask, alpha, torch.ones_like(alpha))
    survival = alpha.cumprod(dim=-1)
    distance = 2.0 * tv if use_l1 else 1.0 - survival
    numerator, count = reduce_specopd_loss(distance, valid_mask, position_weights)
    if ce_weight:
        if ce is None:
            logits = draft_logits.flatten(0, 1)
            labels = target_logits.detach().argmax(-1).flatten()
            if logits.numel() and use_fused_opd_loss(
                logits.device, distribution_loss_impl
            ):
                ce = fused_label_cross_entropy(logits, labels)
            else:
                labels = labels.masked_fill(~valid_mask.flatten(), -100)
                ce = F.cross_entropy(logits, labels, reduction="none")
            ce = ce.reshape_as(valid_mask)
        ce_numerator, _ = reduce_specopd_loss(ce, valid_mask, position_weights)
        numerator = distribution_weight * numerator + ce_weight * ce_numerator
    elif distribution_weight != 1.0:
        numerator = distribution_weight * numerator
    return numerator, count, {"overlap": alpha.detach(), "survival": survival.detach()}


def reduce_specopd_loss(loss, valid_mask, position_weights):
    if position_weights is None:
        return (loss * valid_mask).sum(), valid_mask.sum()
    if position_weights.shape != valid_mask.shape:
        raise ValueError("SpecOPD anchor weight dimensions differ")
    # Match DraftOPDObjective: divide by valid length, then mean over nonempty blocks.
    counts = valid_mask.sum(-1)
    blocks = counts > 0
    weighted = (loss * valid_mask * position_weights).sum(-1)
    return ((weighted / counts.clamp_min(1)) * blocks).sum(), blocks.sum()
