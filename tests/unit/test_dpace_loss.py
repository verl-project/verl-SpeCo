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

import pytest
import torch

from verl_speco.backends.dpace_loss import dpace_position_weights


def test_dpace_weights_match_smoothed_prefix_suffix_formula():
    confidence = torch.tensor([[[0.0, 0.8, 0.5, 0.25]]], dtype=torch.float32)
    per_token_ce = -confidence.clamp_min(1e-12).log()
    loss_mask = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)

    weights = dpace_position_weights(per_token_ce, loss_mask, dpace_alpha=0.5)

    # Anchor position is neutral. Smoothed prediction confidences are
    # [0.9, 0.75, 0.625], with prefixes [0.9, 0.675, 0.421875].
    expected = torch.tensor(
        [[[0.0, 0.9 + 0.675 + 0.421875, 0.675 + 0.421875, 0.421875]]]
    )
    torch.testing.assert_close(weights, expected)
    assert weights.requires_grad is False


def test_dpace_masked_suffix_does_not_reduce_valid_prefix():
    per_token_ce = torch.tensor([[[0.0, 0.2, 99.0, 99.0]]])
    loss_mask = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    weights = dpace_position_weights(per_token_ce, loss_mask, dpace_alpha=0.5)

    expected = 0.5 * torch.exp(torch.tensor(-0.2)) + 0.5
    torch.testing.assert_close(weights[0, 0, 1], expected)
    assert weights[0, 0, 0].item() == 0.0
    assert weights[0, 0, 2:].count_nonzero().item() == 0


@pytest.mark.parametrize("alpha", (-0.1, 1.1))
def test_dpace_rejects_invalid_smoothing(alpha: float):
    with pytest.raises(ValueError, match="dpace_alpha"):
        dpace_position_weights(
            torch.zeros(1, 1, 2), torch.ones(1, 1, 2), dpace_alpha=alpha
        )


def test_dpace_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        dpace_position_weights(torch.zeros(1, 2), torch.ones(1, 3))
