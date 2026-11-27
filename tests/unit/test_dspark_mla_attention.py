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
"""GLM-5 style MLA draft backbone used by DSpark training."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl_speco.models.dflash.modeling_dflash import (  # noqa: E402
    DFlashMLAAttention,
    DFlashMLADecoderLayer,
    _apply_rotary_interleaved,
    _apply_rotary_neox,
    _yarn_apply_mscale,
    _yarn_get_mscale,
)
from verl_speco.models.dspark.configuration_dspark import DSparkConfig  # noqa: E402
from verl_speco.models.dspark.modeling_dspark import DSparkDraftModel  # noqa: E402

_MLA_DIMS = dict(
    q_lora_rank=8,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
)


def _mla_config(**overrides) -> DSparkConfig:
    kwargs = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_target_layers=2,
        num_context_layers=2,
        target_hidden_size=16,
        target_num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=32,
        block_size=2,
        num_anchors=4,
        markov_rank=0,
        draft_attn="mla",
        **_MLA_DIMS,
    )
    kwargs.update(overrides)
    return DSparkConfig(**kwargs)


def _tiny_attention(**overrides) -> DFlashMLAAttention:
    torch.manual_seed(0)
    return DFlashMLAAttention(_mla_config(**overrides)).float().eval()


def test_mla_config_normalizes_topology_and_architecture() -> None:
    config = _mla_config()

    assert config.draft_attn == "mla"
    assert config.head_dim == config.qk_rope_head_dim
    assert config.num_key_value_heads == config.num_attention_heads
    assert config.architectures == ["DSparkDraftModel"]


def test_gqa_config_keeps_qwen3_dspark_architecture() -> None:
    config = _mla_config(
        draft_attn="gqa",
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
    )

    assert config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"kv_lora_rank": None}, "kv_lora_rank"),
        ({"qk_rope_head_dim": None}, "qk_rope_head_dim"),
        ({"qk_rope_head_dim": 3}, "even"),
        ({"v_head_dim": 0}, "v_head_dim"),
        ({"use_sliding_window": True, "sliding_window": 8}, "sliding-window"),
    ],
)
def test_mla_config_rejects_invalid_topology(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _mla_config(**overrides)


def test_mla_config_round_trips_through_dict() -> None:
    config = _mla_config()

    exported = config.to_dict()
    assert exported["architectures"] == ["DSparkDraftModel"]
    assert exported["draft_attn"] == "mla"

    reloaded = DSparkConfig.from_dspark_dict(exported)
    assert reloaded.draft_attn == "mla"
    assert reloaded.architectures == ["DSparkDraftModel"]
    assert reloaded.kv_lora_rank == config.kv_lora_rank


def test_from_dspark_dict_detects_glm5_snapshots() -> None:
    reloaded = DSparkConfig.from_dspark_dict(
        {
            "model_type": "glm5_dspark",
            "architectures": ["DSparkDraftModel"],
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            **_MLA_DIMS,
        }
    )

    assert reloaded.draft_attn == "mla"
    assert reloaded.architectures == ["DSparkDraftModel"]


def test_dspark_model_dispatches_to_mla_decoder_layers() -> None:
    model = DSparkDraftModel(_mla_config())
    layer = model.layers[0]

    assert isinstance(layer, DFlashMLADecoderLayer)
    attention = layer.self_attn
    assert isinstance(attention, DFlashMLAAttention)
    assert attention.q_a_proj.out_features == _MLA_DIMS["q_lora_rank"]
    assert attention.kv_a_proj_with_mqa.out_features == (
        _MLA_DIMS["kv_lora_rank"] + _MLA_DIMS["qk_rope_head_dim"]
    )
    assert attention.kv_b_proj.in_features == _MLA_DIMS["kv_lora_rank"]
    assert attention.o_proj.in_features == 2 * _MLA_DIMS["v_head_dim"]


def test_interleaved_rotary_pairs_adjacent_channels() -> None:
    cos_half = torch.randn(1, 1, 5, 2)
    sin_half = torch.randn(1, 1, 5, 2)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    x = torch.randn(1, 2, 5, 4)

    output = _apply_rotary_interleaved(x, cos, sin)

    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    expected = torch.stack(
        [x_even * cos_half - x_odd * sin_half, x_odd * cos_half + x_even * sin_half],
        dim=-1,
    ).flatten(-2)
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, _apply_rotary_neox(x, cos, sin))


def test_yarn_mscale_scales_attention_by_squared_factor() -> None:
    rope_parameters = {"rope_type": "yarn", "factor": 2.0, "mscale_all_dim": 1.0}

    expected = 0.5 * _yarn_get_mscale(2.0, 1.0) ** 2
    assert _yarn_apply_mscale(rope_parameters, 0.5) == pytest.approx(expected)
    assert _yarn_apply_mscale(None, 0.5) == 0.5
    assert _yarn_apply_mscale({"rope_type": "default"}, 0.5) == 0.5


def test_mla_forward_matches_explicit_masked_softmax(monkeypatch) -> None:
    attention = _tiny_attention()
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    captured: dict[str, torch.Tensor | float] = {}

    def capture_sdpa(query, key, value, **kwargs):
        captured.update(q=query, k=key, v=value, scale=kwargs["scale"])
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", capture_sdpa
    )
    # Compare the attention output directly; the projection is tested by shape.
    attention.o_proj = torch.nn.Identity()

    draft_hidden = torch.randn(1, 3, 16)
    context_hidden = torch.randn(1, 2, 16)
    draft_position_ids = torch.tensor([[2, 3, 4]])
    context_position_ids = torch.tensor([[0, 1]])
    allowed = torch.tensor(
        [
            [
                [
                    [True, True, False, False, False],
                    [True, True, True, False, False],
                    [True, False, False, True, False],
                ]
            ]
        ]
    )

    output = attention(
        draft_hidden,
        context_hidden,
        draft_position_ids,
        context_position_ids,
        dense_attention_mask=allowed,
    )

    query = captured["q"]
    key = captured["k"]
    value = captured["v"]
    assert key.shape[-1] == attention.qk_head_dim
    # v is zero-padded from v_head_dim to qk_head_dim before attention.
    assert value.shape[-1] == attention.qk_head_dim

    scores = query @ key.transpose(-1, -2) * captured["scale"]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    reference = scores.softmax(dim=-1) @ value
    reference = reference[..., : attention.v_head_dim]
    reference = reference.transpose(1, 2).reshape(1, 3, -1)

    assert output.shape == (1, 3, 2 * _MLA_DIMS["v_head_dim"])
    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


def test_mla_attention_dispatches_to_flex_without_gqa(monkeypatch) -> None:
    attention = _tiny_attention()
    calls: dict[str, object] = {}

    def fake_flex(query, key, value, **kwargs):
        calls.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(
        "verl_speco.models.dflash.modeling_dflash.compile_friendly_flex_attention",
        fake_flex,
    )

    output = attention(
        torch.randn(1, 2, 16),
        torch.randn(1, 2, 16),
        torch.tensor([[2, 3]]),
        torch.tensor([[0, 1]]),
        block_mask=object(),
    )

    assert calls["enable_gqa"] is False
    assert calls["scale"] == pytest.approx(attention.scaling)
    assert output.shape == (1, 2, 16)


def test_mla_attention_rejects_oversized_value_heads() -> None:
    with pytest.raises(ValueError, match="v_head_dim"):
        _tiny_attention(v_head_dim=16)


def test_mla_attention_rejects_two_masks() -> None:
    attention = _tiny_attention()
    with pytest.raises(ValueError, match="both block_mask and dense_attention_mask"):
        attention(
            torch.randn(1, 2, 16),
            torch.randn(1, 2, 16),
            torch.tensor([[2, 3]]),
            torch.tensor([[0, 1]]),
            block_mask=object(),
            dense_attention_mask=torch.ones(1, 1, 2, 4, dtype=torch.bool),
        )
