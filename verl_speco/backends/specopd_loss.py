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
"""Differentiable E2E-TV with fixed sampled histories and prefix masks."""

import torch
import math

from verl_speco.backends.opd_math import prefix_survival


def e2e_tv_loss(
    draft_logits,
    target_logits,
    valid_mask,
    *,
    proposal_temperature=1.0,
    target_temperature=1.0,
    candidate_ids=None,
    target_logz=None,
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
    q = (draft_logits.float() / proposal_temperature).softmax(-1)
    with torch.no_grad():
        scaled_target = target_logits.detach().float() / target_temperature
        logz = (
            scaled_target.logsumexp(-1)
            if target_logz is None
            else target_logz.detach().float()
        )
        if candidate_ids is not None:
            scaled_target = scaled_target.gather(-1, candidate_ids)
        p = (scaled_target - logz.unsqueeze(-1)).exp()
    if q.shape != p.shape:
        raise ValueError("proposal and target supports differ")
    alpha = torch.minimum(q, p).sum(-1)
    alpha = alpha.clamp(max=1.0)
    alpha = torch.where(valid_mask, alpha, torch.ones_like(alpha))
    survival = prefix_survival(alpha)
    numerator = ((1.0 - survival) * valid_mask).sum()
    count = valid_mask.sum()
    return numerator, count, {"overlap": alpha.detach(), "survival": survival.detach()}
