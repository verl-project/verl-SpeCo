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
from __future__ import annotations

import torch

from verl_speco.models.dspark.modeling_dspark import (
    DSparkConfidenceHead,
    DSparkDraftModel,
)

_HIDDEN_DIM = 8


def _bare_model(*, detach_backbone: bool) -> DSparkDraftModel:
    """Build a minimal DSparkDraftModel exposing only the confidence head."""

    model = DSparkDraftModel.__new__(DSparkDraftModel)
    torch.nn.Module.__init__(model)
    model.confidence_head = DSparkConfidenceHead(_HIDDEN_DIM)
    model.confidence_head_with_markov = False
    model.markov_head = None
    model.confidence_detach_backbone = detach_backbone
    return model


def _backward_through_head(model: DSparkDraftModel) -> torch.Tensor:
    hidden = torch.randn(4, _HIDDEN_DIM, requires_grad=True)
    logits = model.predict_confidence(
        hidden, prev_token_ids=torch.zeros(4, dtype=torch.long)
    )
    logits.sum().backward()
    return hidden


def test_confidence_loss_updates_shared_backbone_by_default() -> None:
    model = _bare_model(detach_backbone=False)
    hidden = _backward_through_head(model)
    assert model.confidence_head.proj.weight.grad is not None
    assert hidden.grad is not None


def test_confidence_detach_backbone_isolates_shared_backbone() -> None:
    model = _bare_model(detach_backbone=True)
    hidden = _backward_through_head(model)
    # The head still trains, but the shared backbone/Markov features do not.
    assert model.confidence_head.proj.weight.grad is not None
    assert hidden.grad is None
