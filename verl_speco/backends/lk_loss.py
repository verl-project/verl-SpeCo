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

import math

import torch


def full_vocab_subset_probs(
    subset_logits: torch.Tensor,
    target_logz: torch.Tensor,
    *,
    temperature: float = 1.0,
    validation_tolerance: float = 5e-2,
) -> torch.Tensor:
    """Recover exact full-vocabulary probabilities for selected logits.

    ``target_logz`` must be computed by the actor from the same model snapshot
    as ``subset_logits`` using ``logsumexp(full_logits / temperature)``.  This
    keeps probability mass outside the draft vocabulary in the denominator
    without copying the full target LM head to the drafter.

    The consistency checks are evaluated with a single scalar-tensor transfer
    and raise a regular ``ValueError`` on every device (a device-side assert
    would poison the accelerator context). ``validation_tolerance`` must absorb
    bf16 numerical noise between the actor and drafter-side logits, so it is
    deliberately loose: it only needs to catch gross snapshot/temperature
    mismatches, not rounding differences.
    """

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(
            f"temperature must be finite and positive, got {temperature!r}"
        )
    if tuple(target_logz.shape) != tuple(subset_logits.shape[:-1]):
        raise ValueError(
            "target_logz must match the non-vocabulary dimensions of subset_logits, "
            f"got logz={tuple(target_logz.shape)} logits={tuple(subset_logits.shape)}"
        )

    scaled_logits = subset_logits.float() / float(temperature)
    log_probs = scaled_logits - target_logz.float().unsqueeze(-1)
    tolerance = max(float(validation_tolerance), 0.0)
    probs = torch.exp(log_probs)
    with torch.no_grad():
        checks = torch.stack(
            [
                torch.isfinite(target_logz).all(),
                log_probs.detach().amax() <= tolerance,
                (probs.detach().sum(dim=-1) <= 1.0 + tolerance).all(),
            ]
        ).cpu()
    finite_ok, logz_ok, mass_ok = (bool(v) for v in checks)
    if not finite_ok:
        raise ValueError("target_logz contains non-finite values")
    if not logz_ok:
        raise ValueError(
            "target_logz is smaller than a selected logit; the actor log partition "
            "and synchronized LM-head rows are not from the same snapshot"
        )
    if not mass_ok:
        raise ValueError(
            "selected target probability mass exceeds one; target_logz alignment "
            "or temperature is inconsistent"
        )
    return probs


def adaptive_diagnostics_to_rows(
    diagnostics: list[dict[str, torch.Tensor]],
) -> list[list[float]]:
    """Transfer per-position adaptive diagnostics with one device sync."""

    if not diagnostics:
        return []
    rows = torch.stack(
        [
            torch.stack(
                [
                    item["acceptance"],
                    item["kl_weight"],
                    item["forward_kl"],
                    item["tv"],
                ]
            )
            for item in diagnostics
        ]
    )
    return rows.detach().float().cpu().tolist()


def negative_log_acceptance_loss(
    draft_probs: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    acceptance = torch.minimum(draft_probs.float(), target_probs.float()).sum(dim=-1)
    # Clamp at float32 tiny (not a coarse epsilon) so gradients survive early
    # training when the draft/target overlap is still very small.
    return -torch.log(acceptance.clamp_min(torch.finfo(torch.float32).tiny))


def adaptive_hybrid_loss_components(
    draft_probs: torch.Tensor,
    target_acceptance_probs: torch.Tensor,
    target_kl_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-token acceptance, forward KL, and TV components."""

    draft_probs = draft_probs.float()
    target_acceptance_probs = target_acceptance_probs.float()
    target_kl_probs = target_kl_probs.float()
    acceptance = torch.minimum(draft_probs, target_acceptance_probs).sum(dim=-1)
    tiny = torch.finfo(torch.float32).tiny
    forward_kl = (
        target_kl_probs
        * (target_kl_probs.clamp_min(tiny).log() - draft_probs.clamp_min(tiny).log())
    ).sum(dim=-1)
    tv = 1.0 - acceptance
    return acceptance, forward_kl, tv


def adaptive_hybrid_acceptance_loss(
    draft_probs: torch.Tensor,
    target_acceptance_probs: torch.Tensor,
    target_kl_probs: torch.Tensor,
    position_mask: torch.Tensor,
    *,
    eta: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend forward KL and TV using the current mean acceptance.

    The blend weight ``lambda = exp(-eta * stop_grad(mean_acceptance))`` uses the
    *unweighted* mean acceptance over the valid positions passed in one call (a
    TTT step for EAGLE3, all active rows for DFlash/DSpark), independent of any
    per-position loss weights (fixed decay / D-PACE) applied to the final loss.

    ``target_acceptance_probs`` keeps the target's full-vocabulary
    normalization after selecting draft-vocabulary rows, so
    ``1 - sum(min(p, q))`` is the full-vocabulary TV distance. In contrast,
    ``target_kl_probs`` is normalized inside the draft vocabulary, matching
    the truncated-vocabulary KL objective used by EAGLE3.
    """
    if draft_probs.shape != target_acceptance_probs.shape:
        raise ValueError(
            "draft_probs and target_acceptance_probs must have the same shape, "
            f"got {tuple(draft_probs.shape)} and "
            f"{tuple(target_acceptance_probs.shape)}"
        )
    if draft_probs.shape != target_kl_probs.shape:
        raise ValueError(
            "draft_probs and target_kl_probs must have the same shape, "
            f"got {tuple(draft_probs.shape)} and {tuple(target_kl_probs.shape)}"
        )
    if tuple(position_mask.shape) != tuple(draft_probs.shape[:-1]):
        raise ValueError(
            "position_mask must match the non-vocabulary dimensions, "
            f"got mask={tuple(position_mask.shape)} and "
            f"probs={tuple(draft_probs.shape)}"
        )
    if not math.isfinite(float(eta)) or eta < 0:
        raise ValueError(f"eta must be finite and non-negative, got {eta!r}")

    draft_probs = draft_probs.float()
    target_acceptance_probs = target_acceptance_probs.float()
    target_kl_probs = target_kl_probs.float()
    valid = position_mask.to(device=draft_probs.device, dtype=torch.bool)

    acceptance, forward_kl, tv = adaptive_hybrid_loss_components(
        draft_probs,
        target_acceptance_probs,
        target_kl_probs,
    )
    valid_count = valid.float().sum()
    mean_acceptance = torch.where(
        valid_count > 0,
        (acceptance * valid.float()).sum() / valid_count.clamp_min(1.0),
        acceptance.new_zeros(()),
    )
    kl_weight = torch.exp(-float(eta) * mean_acceptance.detach())

    per_token_loss = kl_weight * forward_kl + (1.0 - kl_weight) * tv
    per_token_loss = torch.where(
        valid, per_token_loss, torch.zeros_like(per_token_loss)
    )

    def _masked_mean(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            valid_count > 0,
            (values * valid.float()).sum() / valid_count.clamp_min(1.0),
            values.new_zeros(()),
        )

    diagnostics = {
        "acceptance": mean_acceptance.detach(),
        "kl_weight": kl_weight,
        "forward_kl": _masked_mean(forward_kl).detach(),
        "tv": _masked_mean(tv).detach(),
    }
    return per_token_loss, diagnostics
