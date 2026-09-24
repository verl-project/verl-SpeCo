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

import torch


def dpace_position_weights(
    per_token_ce: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    dpace_alpha: float = 0.5,
) -> torch.Tensor:
    """Compute detached D-PACE weights for the last (block-position) axis.

    ``per_token_ce`` and ``loss_mask`` must have identical shapes ending in the
    speculative block size. Invalid positions are neutral in the cumulative
    product and receive zero final weight. Position zero can therefore be
    masked for DFlash anchors while remaining active for DSpark blocks.
    """
    if per_token_ce.shape != loss_mask.shape:
        raise ValueError(
            "D-PACE per_token_ce and loss_mask must have identical shapes, got "
            f"{tuple(per_token_ce.shape)} and {tuple(loss_mask.shape)}"
        )
    if per_token_ce.ndim == 0 or per_token_ce.shape[-1] == 0:
        raise ValueError("D-PACE inputs must include a non-empty block-position axis")
    alpha = float(dpace_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha!r}")

    with torch.no_grad():
        mask = loss_mask.to(dtype=torch.float32)
        confidence = torch.exp(-per_token_ce.detach().float())
        smooth = (1.0 - alpha) * confidence + alpha
        smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights * mask
