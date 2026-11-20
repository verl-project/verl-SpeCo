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
"""Training-only VAT objective; acceptance is a greedy proxy, not rollout AL."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class VATObjective(nn.Module):
    def __init__(self, hidden_size, *, layout, gamma, config, loss_mode):
        super().__init__()
        self.layout = layout
        self.gamma = gamma
        self.kd_weight = config.get("kd_weight")
        if self.kd_weight is None:
            self.kd_weight = 1.0 if layout == "dflash" else 0.0
        self.temperature = float(config.get("kd_temperature", 2.0))
        self.accept_weight = float(config.get("accept_head_weight", 1.0))
        self.cov_weight = float(config.get("self_cov_weight", 0.0))
        self.floor = float(config.get("self_cov_floor", 1e-6))
        self.chunk_size = int(config.get("chunk_size", 1024))
        if loss_mode != "full_vocab" or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("VAT requires full_vocab loss and finite gamma > 0")
        if config.get("accept_mode", "greedy") != "greedy":
            raise ValueError("VAT currently supports only the greedy acceptance proxy")
        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
            or not 0 < self.floor <= 1
            or self.chunk_size <= 0
            or any(
                not math.isfinite(w) or w < 0
                for w in (self.kd_weight, self.accept_weight, self.cov_weight)
            )
        ):
            raise ValueError(
                "Invalid VAT temperature, floor, chunk size or loss weight"
            )
        self.accept_head = nn.Linear(hidden_size, 1) if layout == "dflash" else None

    def _teacher_terms(self, logits, target_hidden, weight):
        with torch.no_grad():
            teacher = F.linear(target_hidden, weight).float()
        if self.kd_weight == 0:
            return teacher.argmax(-1), logits.new_zeros(
                logits.shape[0], dtype=torch.float32
            )
        kd = F.kl_div(
            F.log_softmax(logits.float() / self.temperature, -1),
            F.softmax(teacher / self.temperature, -1),
            reduction="none",
        ).sum(-1)
        return teacher.argmax(-1), kd * self.temperature**2

    def forward(
        self,
        logits,
        hidden,
        target_hidden,
        weight,
        base_mask,
        active_mask,
        target_ids,
        confidence_logits=None,
        observed_accepted=None,
        position_weights=None,
    ):
        if target_hidden is None:
            raise ValueError("VAT requires dflash_aux_plus_last target hidden states")
        matches = torch.zeros_like(base_mask, dtype=torch.bool)
        kd = logits.new_zeros(logits.shape[0], dtype=torch.float32)
        top1 = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        for start in range(0, logits.shape[0], self.chunk_size):
            end = start + self.chunk_size
            args = (logits[start:end], target_hidden[start:end], weight)
            ids, terms = checkpoint(self._teacher_terms, *args, use_reentrant=False)
            top1[start:end] = ids
            kd[start:end] = terms
        matches.reshape(-1)[active_mask] = logits.detach().argmax(-1) == top1
        offset = 1 if self.layout == "dflash" else 0
        valid = base_mask[..., offset:] > 0
        match = matches[..., offset:]
        accepted = (match & valid).long().cumprod(-1).sum(-1, keepdim=True)
        sim_accepted = accepted
        if observed_accepted is not None:
            accepted = torch.where(
                observed_accepted.unsqueeze(-1) >= 0,
                observed_accepted.unsqueeze(-1),
                accepted,
            )
            match = torch.where(
                observed_accepted.unsqueeze(-1) >= 0,
                torch.arange(valid.shape[-1], device=logits.device) < accepted,
                match,
            )
        j = torch.arange(valid.shape[-1], device=logits.device)
        decay = torch.exp(-(j - accepted).clamp_min(0).float() / self.gamma)
        weights = base_mask.clone()
        weights[..., offset:] *= decay
        loss_weights = (
            weights if position_weights is None else weights * position_weights
        )
        active_weights = loss_weights.reshape(-1)[active_mask]
        kd_loss = (kd * active_weights).sum() / active_weights.sum().clamp_min(1e-6)
        targets = torch.zeros_like(base_mask)
        head_mask = base_mask.clone()
        if self.layout == "dflash":
            targets[..., offset:] = (j < accepted).float()
            accept_logits = self.accept_head(hidden).squeeze(-1)
        else:
            targets[..., offset:] = match.float()
            head_mask[..., offset:] *= j <= accepted
            accept_logits = confidence_logits
        if position_weights is not None:
            head_mask = head_mask * position_weights
        hw = head_mask.reshape(-1)[active_mask]
        accept_loss = (
            F.binary_cross_entropy_with_logits(
                accept_logits.float(),
                targets.reshape(-1)[active_mask],
                reduction="none",
            )
            * hw
        ).sum()
        accept_loss = accept_loss / hw.sum().clamp_min(1e-6)
        cov_loss = logits[:0].sum()
        if self.cov_weight > 0:
            neg_log_q = torch.zeros_like(base_mask)
            neg_log_q.reshape(-1)[active_mask] = (
                -F.log_softmax(logits.float(), -1)
                .gather(-1, target_ids[:, None])
                .squeeze(-1)
            )
            row_valid = valid.long().cumprod(-1).bool()
            log_prefix = (
                (-neg_log_q[..., offset:] * row_valid)
                .cumsum(-1)
                .clamp_min(math.log(self.floor))
            )
            gate = (j >= accepted) & row_valid
            blocks = row_valid[..., 0].float()
            cov_loss = ((-(log_prefix * gate).sum(-1) / valid.shape[-1]) * blocks).sum()
            cov_loss = cov_loss / blocks.sum().clamp_min(1)
        loss = (
            self.kd_weight * kd_loss
            + self.accept_weight * accept_loss
            + self.cov_weight * cov_loss
        )
        blocks = valid[..., 0].float()
        metrics = {
            "vat/batch_count": logits.new_tensor(1.0),
            "vat/kd_loss": kd_loss.detach(),
            "vat/accept_loss": accept_loss.detach(),
            "vat/self_cov_loss": cov_loss.detach(),
            "vat/mean_accepted": (accepted.squeeze(-1) * blocks).sum()
            / blocks.sum().clamp_min(1),
        }
        if observed_accepted is not None:
            traced = (observed_accepted >= 0) * blocks
            metrics["vat/sim_vs_obs_visible_accept_gap"] = (
                (
                    sim_accepted.squeeze(-1)
                    - torch.minimum(observed_accepted, valid.long().cumprod(-1).sum(-1))
                )
                .float()
                .abs()
                * traced
            ).sum() / traced.sum().clamp_min(1)
        return weights, loss, metrics
