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

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

target_head = pytest.importorskip("verl_speco.models.target.target_head")

TargetFinalNorm = target_head.TargetFinalNorm


@pytest.mark.parametrize(
    ("model_type", "norm_key", "text_config", "expect_fold"),
    [
        ("qwen3", "model.norm.weight", False, False),
        ("qwen3_5_text", "model.language_model.norm.weight", False, True),
        ("qwen3_5", "model.norm.weight", True, True),
        ("gemma3", "model.norm.weight", False, True),
    ],
)
def test_target_final_norm_from_pretrained_folding(
    tmp_path, model_type, norm_key, text_config, expect_fold
):
    eps = 5e-6
    config = {
        "model_type": model_type,
        "rms_norm_eps": eps,
        "hidden_size": 8,
    }
    if text_config:
        config = {"model_type": f"{model_type}_vl", "text_config": dict(config)}
    (tmp_path / "config.json").write_text(json.dumps(config))
    raw_norm = torch.randn(8)
    safetensors_torch.save_file(
        {norm_key: raw_norm, "lm_head.weight": torch.randn(32, 8)},
        str(tmp_path / "model.safetensors"),
    )

    final_norm = TargetFinalNorm.from_pretrained(str(tmp_path))

    assert final_norm.eps == pytest.approx(eps)
    expected_weight = raw_norm + 1.0 if expect_fold else raw_norm
    assert torch.allclose(final_norm.weight, expected_weight.float())
    assert not final_norm.weight.requires_grad


def test_target_final_norm_missing_weight_raises(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "rms_norm_eps": 1e-6})
    )
    safetensors_torch.save_file(
        {"lm_head.weight": torch.randn(32, 8)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(KeyError, match="final-norm"):
        TargetFinalNorm.from_pretrained(str(tmp_path))
