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
"""CPU tests for DSV4 DSpark released-weight mapping and dequantization."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

dsv4_weights = pytest.importorskip("verl_speco.models.dsv4_dspark.weights")

map_released_key = dsv4_weights.map_released_key
_unpack_fp4 = dsv4_weights._unpack_fp4
_expand_scale = dsv4_weights._expand_scale
_dequant_weight = dsv4_weights._dequant_weight
_dequant_fp8 = dsv4_weights._dequant_fp8
load_released_draft = dsv4_weights.load_released_draft


def test_map_released_key_covers_every_namespace():
    assert map_released_key("embed.weight") == "embed_tokens.weight"
    assert map_released_key("head.weight") == "lm_head.weight"
    assert map_released_key("mtp.0.main_proj.weight") == "fc.weight"
    assert map_released_key("mtp.0.main_norm.weight") == "hidden_norm.weight"
    assert map_released_key("mtp.1.norm.weight") == "norm.weight"
    assert map_released_key("mtp.0.markov_head.markov_w1.weight") == (
        "markov_head.markov_w1.weight"
    )
    assert map_released_key("mtp.0.confidence_head.proj.weight") == (
        "confidence_head.proj.weight"
    )
    assert map_released_key("mtp.0.hc_head_fn") == "hc_head.hc_fn"
    assert map_released_key("mtp.0.hc_attn_fn") == "layers.0.attn_hc.fn"
    assert map_released_key("mtp.1.ffn.gate.weight") == "layers.1.ffn.router.weight"
    assert map_released_key("mtp.2.ffn.experts.3.w2.weight") == (
        "layers.2.ffn.experts.w2_idx_3"
    )
    # Non-draft keys are skipped.
    assert map_released_key("layers.0.foo.weight") is None
    assert map_released_key("mtp.0.ffn.experts.0.w1.scale") is None


def test_expand_scale_standard_block():
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    expanded = _expand_scale(scale, (2, 2), block=(2, 2))
    assert torch.equal(expanded, torch.full((2, 2), 2.0))


def test_unpack_fp4_nibble_order():
    # Byte 0x21 -> low nibble 1 (0.5), high nibble 2 (1.0).
    out = _unpack_fp4(torch.tensor([[0x21]], dtype=torch.int8))
    assert out.shape == (1, 2)
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0]]))


def test_dequant_weight_int8_scale_offset():
    weight = torch.tensor([[2, -4]], dtype=torch.int8)
    scale = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    offset = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    out = _dequant_weight(weight, scale, offset, target_dtype=torch.float32)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0]]))


def test_dequant_fp8_block_scale():
    weight = torch.tensor([[2.0, 3.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    out = _dequant_fp8(weight, scale, target_dtype=torch.float32, block=(1, 1))
    torch.testing.assert_close(out, torch.tensor([[2.0, 1.5]]))


def _tiny_dsv4_model():
    from verl_speco.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel

    config = DSV4DSparkConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        target_hidden_size=64,
        target_num_hidden_layers=8,
        num_context_layers=2,
        target_layer_ids=[5, 6],
        mask_token_id=127,
        block_size=4,
        num_anchors=2,
        markov_rank=8,
        dsv4_num_heads=4,
        dsv4_head_dim=16,
        dsv4_rope_head_dim=8,
        dsv4_q_lora_rank=16,
        dsv4_o_lora_rank=16,
        dsv4_o_groups=2,
        dsv4_n_routed_experts=4,
        dsv4_n_activated_experts=2,
        dsv4_moe_inter_dim=32,
        dsv4_hc_mult=2,
    )
    return DSV4DSparkDraftModel(config)


def test_load_released_draft_bf16(tmp_path):
    model = _tiny_dsv4_model()
    released = {
        "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
        "mtp.0.ffn.gate.weight": model.layers[0].ffn.router.weight.detach().clone(),
        "mtp.0.ffn.experts.0.w1.weight": model.layers[
            0
        ].ffn.experts.w1.detach()[0].clone(),
    }
    # Perturb so a successful load is observable.
    for value in released.values():
        value.add_(1.0)
    safetensors_torch.save_file(released, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    key: "model-00001.safetensors" for key in released
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_released_draft(
        model, str(tmp_path), n_draft_layers=2, verbose=False, target_dtype=torch.float32
    )

    assert loaded
    torch.testing.assert_close(model.fc.weight, released["mtp.0.main_proj.weight"])
    assert torch.equal(
        model.layers[0].ffn.experts.w1[0], released["mtp.0.ffn.experts.0.w1.weight"]
    )


def _write_index(tmp_path, tensors, filename="model-00001.safetensors"):
    safetensors_torch.save_file(tensors, str(tmp_path / filename))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}}),
        encoding="utf-8",
    )


def test_load_released_draft_rejects_shape_mismatch(tmp_path):
    model = _tiny_dsv4_model()
    _write_index(
        tmp_path,
        {"mtp.0.main_proj.weight": torch.zeros(4, 4, dtype=torch.float32)},
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_load_released_draft_rejects_unknown_model_key(tmp_path):
    model = _tiny_dsv4_model()
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            # maps to layers.0.ffn.shared_experts.w4.* which the model lacks.
            "mtp.0.ffn.shared_experts.w4.weight": torch.zeros(4, 64),
        },
    )
    with pytest.raises(ValueError, match="no matching model parameter"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_load_released_draft_rejects_expert_overflow(tmp_path):
    model = _tiny_dsv4_model()
    expert_shape = tuple(model.layers[0].ffn.experts.w1.shape[1:])
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            "mtp.0.ffn.experts.99.w1.weight": torch.zeros(expert_shape),
        },
    )
    with pytest.raises(ValueError, match="out of range"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_load_released_draft_rejects_index_without_mtp(tmp_path):
    model = _tiny_dsv4_model()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.attn.wq_a.weight": "a.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no mtp"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_load_released_draft_requires_essential_key(tmp_path):
    model = _tiny_dsv4_model()
    _write_index(
        tmp_path,
        {"mtp.0.ffn.gate.weight": model.layers[0].ffn.router.weight.detach().clone()},
    )
    with pytest.raises(ValueError, match="required key"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_load_released_draft_routes_experts_to_owner_rank(tmp_path):
    from verl_speco.models.dsv4_dspark.backbone import moe_ep

    # 4 global experts over 2 ranks -> rank 1 builds 2 local experts at slots 0,1.
    moe_ep.configure(group=None, rank=1, size=2, experts_per_rank=2)
    model = _tiny_dsv4_model()
    assert model.layers[0].ffn.experts.w1.shape[0] == 2
    expert_shape = tuple(model.layers[0].ffn.experts.w1.shape[1:])

    tensors = {"mtp.0.main_proj.weight": model.fc.weight.detach().clone()}
    for expert_idx in range(4):
        tensors[f"mtp.0.ffn.experts.{expert_idx}.w1.weight"] = torch.full(
            expert_shape, float(expert_idx)
        )
    _write_index(tmp_path, tensors)

    load_released_draft(model, str(tmp_path), n_draft_layers=2, target_dtype=torch.float32)

    # Rank 1 owns global experts 2 and 3; they land in local slots 0 and 1.
    torch.testing.assert_close(
        model.layers[0].ffn.experts.w1[0], torch.full(expert_shape, 2.0)
    )
    torch.testing.assert_close(
        model.layers[0].ffn.experts.w1[1], torch.full(expert_shape, 3.0)
    )


def test_dsv4_load_draft_state_dict_selects_mtp_index(tmp_path):
    from verl_speco.backends.dsv4_dspark_trainer_backend import (
        DSV4DSparkTrainerBackend,
    )

    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    safetensors_torch.save_file(
        {"mtp.0.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "draft.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.ffn.gate.weight": "draft.safetensors"}}),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {"layers.0.attn.wq_a.weight": torch.zeros(1, 1)},
        str(tmp_path / "main.safetensors"),
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"layers.0.attn.wq_a.weight": "main.safetensors"}}
        ),
        encoding="utf-8",
    )

    state = backend._load_draft_state_dict(str(tmp_path))

    assert set(state) == {"mtp.0.ffn.gate.weight"}


def test_load_released_draft_rejects_mismatched_ep_context(tmp_path):
    model = _tiny_dsv4_model()  # holds all 4 experts
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            "mtp.0.ffn.experts.0.w1.weight": torch.zeros_like(
                model.layers[0].ffn.experts.w1[0]
            ),
        },
    )
    with pytest.raises(ValueError, match="expert-parallel context"):
        load_released_draft(
            model,
            str(tmp_path),
            n_draft_layers=2,
            target_dtype=torch.float32,
            ep_rank=0,
            experts_per_rank=2,
        )


def test_dsv4_load_draft_state_dict_merges_split_mtp_indices(tmp_path):
    from verl_speco.backends.dsv4_dspark_trainer_backend import (
        DSV4DSparkTrainerBackend,
    )

    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    safetensors_torch.save_file(
        {"mtp.0.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "a.safetensors"),
    )
    safetensors_torch.save_file(
        {"mtp.1.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "b.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.ffn.gate.weight": "a.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.1.ffn.gate.weight": "b.safetensors"}}),
        encoding="utf-8",
    )

    state = backend._load_draft_state_dict(str(tmp_path))

    assert set(state) == {"mtp.0.ffn.gate.weight", "mtp.1.ffn.gate.weight"}


def test_dsv4_confidence_bias_key_mapping_both_directions():
    from verl_speco.tools.convert_checkpoint import _map_key

    assert (
        map_released_key("mtp.2.confidence_head.proj.weight")
        == "confidence_head.proj.weight"
    )
    assert (
        map_released_key("mtp.2.confidence_head.proj.bias")
        == "confidence_head.proj.bias"
    )
    assert _map_key("confidence_head.proj.weight", 3) == [
        "mtp.2.confidence_head.proj.weight"
    ]
    assert _map_key("confidence_head.proj.bias", 3) == [
        "mtp.2.confidence_head.proj.bias"
    ]


@pytest.mark.parametrize(
    "model_key",
    [
        "embed_tokens.weight",
        "lm_head.weight",
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
        "hc_head.hc_fn",
        "hc_head.hc_base",
        "hc_head.hc_scale",
        "layers.0.attn_hc.fn",
        "layers.0.ffn_hc.scale",
        "layers.0.ffn.router.weight",
        "layers.0.ffn.router.bias",
        "layers.1.attn.wq_a.weight",
        "layers.1.attn.attn_sink",
        "layers.1.attn_norm.weight",
        "layers.1.ffn_norm.weight",
        "layers.0.ffn.shared_experts.w1.weight",
    ],
)
def test_dsv4_key_mapping_roundtrip(model_key):
    from verl_speco.tools.convert_checkpoint import _map_key

    released = _map_key(model_key, 3)
    assert released is not None and len(released) == 1, model_key
    assert map_released_key(released[0]) == model_key


def test_dsv4_expert_key_mapping_roundtrip():
    from verl_speco.tools.convert_checkpoint import _map_key

    released = _map_key("layers.0.ffn.experts.w1", 3)
    assert released == ["mtp.0.ffn.experts.{e}.w1.weight"]
    assert map_released_key(released[0].format(e=7)) == "layers.0.ffn.experts.w1_idx_7"


def test_load_released_draft_loads_mhc_scale_params(tmp_path):
    model = _tiny_dsv4_model()
    attn_scale = model.layers[0].attn_hc.scale.detach().clone().add_(1.0)
    head_scale = model.hc_head.hc_scale.detach().clone().add_(1.0)
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            "mtp.0.hc_attn_scale": attn_scale,
            "mtp.0.hc_head_scale": head_scale,
        },
    )

    load_released_draft(
        model, str(tmp_path), n_draft_layers=2, target_dtype=torch.float32
    )

    torch.testing.assert_close(model.layers[0].attn_hc.scale, attn_scale)
    torch.testing.assert_close(model.hc_head.hc_scale, head_scale)


def test_dsv4_key_mapping_roundtrip_exhaustive():
    from verl_speco.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel
    from verl_speco.tools.convert_checkpoint import _map_key

    # Include the optional confidence head so its keys are covered too.
    config = DSV4DSparkConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        target_hidden_size=64,
        target_num_hidden_layers=8,
        num_context_layers=2,
        target_layer_ids=[5, 6],
        mask_token_id=127,
        block_size=4,
        num_anchors=2,
        markov_rank=8,
        enable_confidence_head=True,
        dsv4_num_heads=4,
        dsv4_head_dim=16,
        dsv4_rope_head_dim=8,
        dsv4_q_lora_rank=16,
        dsv4_o_lora_rank=16,
        dsv4_o_groups=2,
        dsv4_n_routed_experts=4,
        dsv4_n_activated_experts=2,
        dsv4_moe_inter_dim=32,
        dsv4_hc_mult=2,
    )
    model = DSV4DSparkDraftModel(config)

    unmapped = []
    for key in model.state_dict():
        if key.startswith("layers.") and ".ffn.experts.w" in key:
            released = _map_key(key, 3)
            assert released is not None and "{e}" in released[0], key
            assert map_released_key(released[0].format(e=0)) == f"{key}_idx_0"
            continue
        released = _map_key(key, 3)
        if released is None:
            unmapped.append(key)
            continue
        assert len(released) == 1, (key, released)
        assert map_released_key(released[0]) == key, (key, released[0])
    assert not unmapped, f"unmapped model keys: {unmapped}"


def test_convert_checkpoint_end_to_end():
    from verl_speco.tools.convert_checkpoint import convert

    model = _tiny_dsv4_model()  # 2 draft layers, 4 experts
    out, skipped, n_unstacked = convert(model.state_dict(), 2)

    # Context projection / final norm / expert unstacking / confidence head.
    assert "mtp.0.main_proj.weight" in out
    assert "mtp.1.norm.weight" in out
    assert "mtp.0.ffn.experts.0.w1.weight" in out
    assert "mtp.0.ffn.experts.3.w1.weight" in out
    assert n_unstacked == 6  # per layer: w1, w3, w2
    # Every emitted key maps back to a model parameter name (round-trip).
    for key in out:
        assert key.startswith("mtp.") or key in ("embed.weight", "head.weight"), key
        mapped = map_released_key(key)
        assert mapped is not None, key
    # Only the (unused) lm_head is expected to be skipped, if present at all.
    assert all("freqs_cis" not in key for key in skipped)


def _tiny_dsv4_model_with_confidence():
    from verl_speco.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel

    config = DSV4DSparkConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        target_hidden_size=64,
        target_num_hidden_layers=8,
        num_context_layers=2,
        target_layer_ids=[5, 6],
        mask_token_id=127,
        block_size=4,
        num_anchors=2,
        markov_rank=8,
        enable_confidence_head=True,
        dsv4_num_heads=4,
        dsv4_head_dim=16,
        dsv4_rope_head_dim=8,
        dsv4_q_lora_rank=16,
        dsv4_o_lora_rank=16,
        dsv4_o_groups=2,
        dsv4_n_routed_experts=4,
        dsv4_n_activated_experts=2,
        dsv4_moe_inter_dim=32,
        dsv4_hc_mult=2,
    )
    return DSV4DSparkDraftModel(config)


def test_load_released_draft_zeroes_bias_for_biasless_confidence_head(tmp_path):
    model = _tiny_dsv4_model_with_confidence()
    # nn.Linear initialises the bias to a non-zero value.
    assert torch.count_nonzero(model.confidence_head.proj.bias).item() > 0
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            # Bias-less released head: weight only.
            "mtp.0.confidence_head.proj.weight": model.confidence_head.proj.weight.detach().clone(),
        },
    )

    load_released_draft(
        model, str(tmp_path), n_draft_layers=2, target_dtype=torch.float32
    )

    assert torch.count_nonzero(model.confidence_head.proj.bias).item() == 0


def test_dsv4_load_draft_state_dict_filters_shards_in_single_combined_index(tmp_path):
    from verl_speco.backends.dsv4_dspark_trainer_backend import (
        DSV4DSparkTrainerBackend,
    )

    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    safetensors_torch.save_file(
        {"mtp.0.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "draft.safetensors"),
    )
    safetensors_torch.save_file(
        {"layers.0.attn.wq_a.weight": torch.zeros(8, 8)},
        str(tmp_path / "target.safetensors"),
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.0.ffn.gate.weight": "draft.safetensors",
                    "layers.0.attn.wq_a.weight": "target.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    state = backend._load_draft_state_dict(str(tmp_path))

    assert set(state) == {"mtp.0.ffn.gate.weight"}


def test_dsv4_load_draft_state_metadata_reads_only_drafter_keys(tmp_path):
    from verl_speco.backends.dsv4_dspark_trainer_backend import (
        DSV4DSparkTrainerBackend,
        _TensorMeta,
    )

    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    safetensors_torch.save_file(
        {"mtp.0.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "draft.safetensors"),
    )
    safetensors_torch.save_file(
        {"layers.0.attn.wq_a.weight": torch.zeros(8, 8)},
        str(tmp_path / "target.safetensors"),
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.0.ffn.gate.weight": "draft.safetensors",
                    "layers.0.attn.wq_a.weight": "target.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    meta = backend._load_draft_state_metadata(str(tmp_path))

    assert set(meta) == {"mtp.0.ffn.gate.weight"}
    entry = meta["mtp.0.ffn.gate.weight"]
    assert isinstance(entry, _TensorMeta)
    assert entry.shape == (4, 64) and entry.size(1) == 64 and entry.ndim == 2
    assert entry.dtype == torch.float32


def test_dsv4_load_draft_state_metadata_tolerates_missing_shard(tmp_path):
    from verl_speco.backends.dsv4_dspark_trainer_backend import (
        DSV4DSparkTrainerBackend,
    )

    backend = DSV4DSparkTrainerBackend.__new__(DSV4DSparkTrainerBackend)
    safetensors_torch.save_file(
        {"mtp.0.ffn.gate.weight": torch.zeros(4, 64)},
        str(tmp_path / "present.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.0.ffn.gate.weight": "present.safetensors",
                    "mtp.0.ffn.gate2.weight": "missing.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    meta = backend._load_draft_state_metadata(str(tmp_path))

    assert "mtp.0.ffn.gate.weight" in meta


def test_load_released_draft_dequant_fp8_with_weight_scale_key(tmp_path):
    model = _tiny_dsv4_model()
    expert_shape = tuple(model.layers[0].ffn.experts.w1.shape[1:])
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            "mtp.0.ffn.experts.0.w1.weight": torch.ones(
                expert_shape, dtype=torch.float8_e4m3fn
            ),
            # FP8 scale spelled ``weight_scale`` (not ``.scale``).
            "mtp.0.ffn.experts.0.w1.weight_scale": torch.full(
                (expert_shape[0], 1), 2.0, dtype=torch.float32
            ),
        },
    )

    load_released_draft(
        model, str(tmp_path), n_draft_layers=2, target_dtype=torch.float32
    )

    # 1.0 (fp8) * 2.0 (scale) == 2.0 everywhere.
    torch.testing.assert_close(
        model.layers[0].ffn.experts.w1[0], torch.full(expert_shape, 2.0)
    )


def test_load_released_draft_rejects_fp8_without_scale(tmp_path):
    model = _tiny_dsv4_model()
    expert_shape = tuple(model.layers[0].ffn.experts.w1.shape[1:])
    _write_index(
        tmp_path,
        {
            "mtp.0.main_proj.weight": model.fc.weight.detach().clone(),
            "mtp.0.ffn.experts.0.w1.weight": torch.zeros(
                expert_shape, dtype=torch.float8_e4m3fn
            ),
        },
    )

    with pytest.raises(ValueError, match="no companion scale"):
        load_released_draft(model, str(tmp_path), n_draft_layers=2)


def test_target_head_recognizes_deepseek_keys_across_multiple_index(tmp_path):
    from verl_speco.models.target.target_head import TargetHead

    safetensors_torch.save_file(
        {"head.weight": torch.zeros(8, 4)}, str(tmp_path / "a.safetensors")
    )
    safetensors_torch.save_file(
        {"embed.weight": torch.ones(8, 4)}, str(tmp_path / "b.safetensors")
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"head.weight": "a.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed.weight": "b.safetensors"}}),
        encoding="utf-8",
    )

    head = TargetHead.from_pretrained(str(tmp_path))

    torch.testing.assert_close(head.fc.weight, torch.zeros(8, 4))


def test_target_head_falls_back_to_embed_weight(tmp_path):
    from verl_speco.models.target.target_head import TargetHead

    safetensors_torch.save_file(
        {"embed.weight": torch.ones(8, 4)}, str(tmp_path / "a.safetensors")
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed.weight": "a.safetensors"}}),
        encoding="utf-8",
    )

    head = TargetHead.from_pretrained(str(tmp_path))

    torch.testing.assert_close(head.fc.weight, torch.ones(8, 4))


def test_convert_checkpoint_merges_ep_expert_slices(tmp_path):
    from verl_speco.tools.convert_checkpoint import (
        _load_state_dict,
        _merge_ep_expert_slices,
    )

    # Leader-only export: the stacked expert weight holds just the leader slice.
    leader_slice = torch.zeros(2, 3, 2)
    leader_slice[0] = 0.0
    leader_slice[1] = 1.0
    safetensors_torch.save_file(
        {
            # The router weight pins n_routed_experts = 4 (2 ranks x 2 local).
            "layers.0.ffn.router.weight": torch.zeros(4, 2),
            "layers.0.ffn.experts.w1": leader_slice,
        },
        str(tmp_path / "model.safetensors"),
    )
    ep_dir = tmp_path / "ep_experts"
    ep_dir.mkdir()
    for rank in range(2):
        sl = torch.zeros(2, 3, 2)
        sl[0] = 10.0 * rank
        sl[1] = 10.0 * rank + 1.0
        torch.save({"layers.0.ffn.experts.w1": sl}, ep_dir / f"rank_{rank}.pt")

    sd = _load_state_dict(tmp_path)
    assert sd["layers.0.ffn.experts.w1"].shape == (2, 3, 2)  # leader slice only

    merged = _merge_ep_expert_slices(sd, ep_dir)
    assert merged["layers.0.ffn.experts.w1"].shape == (4, 3, 2)
    # Rank r owns experts [r*local, (r+1)*local): rank0 rows 0-1, rank1 rows 2-3.
    torch.testing.assert_close(merged["layers.0.ffn.experts.w1"][0], torch.zeros(3, 2))
    torch.testing.assert_close(
        merged["layers.0.ffn.experts.w1"][1], torch.ones(3, 2)
    )
    torch.testing.assert_close(
        merged["layers.0.ffn.experts.w1"][2], torch.full((3, 2), 10.0)
    )
    torch.testing.assert_close(
        merged["layers.0.ffn.experts.w1"][3], torch.full((3, 2), 11.0)
    )

    from verl_speco.tools.convert_checkpoint import convert

    out, _skipped, n_unstacked = convert(merged, 1)
    assert n_unstacked == 1
    for e in range(4):
        assert f"mtp.0.ffn.experts.{e}.w1.weight" in out


def test_convert_checkpoint_merge_rejects_wrong_expert_count(tmp_path):
    from verl_speco.tools.convert_checkpoint import (
        _load_state_dict,
        _merge_ep_expert_slices,
    )

    safetensors_torch.save_file(
        {
            # Router says 4 experts...
            "layers.0.ffn.router.weight": torch.zeros(4, 2),
            "layers.0.ffn.experts.w1": torch.zeros(2, 3, 2),
        },
        str(tmp_path / "model.safetensors"),
    )
    ep_dir = tmp_path / "ep_experts"
    ep_dir.mkdir()
    # ...but only one 2-expert slice is present -> merged rows 2 != 4.
    torch.save(
        {"layers.0.ffn.experts.w1": torch.zeros(2, 3, 2)}, ep_dir / "rank_0.pt"
    )
    sd = _load_state_dict(tmp_path)

    with pytest.raises(SystemExit, match="n_routed_experts"):
        _merge_ep_expert_slices(sd, ep_dir)


def test_convert_checkpoint_merge_validates_against_config_rows(
    tmp_path, capsys
):
    from verl_speco.tools.convert_checkpoint import (
        _load_state_dict,
        _merge_ep_expert_slices,
    )

    # No router weight in the export (e.g. a prefixed export): the config's
    # dsv4_n_routed_experts is the fallback validation source.
    safetensors_torch.save_file(
        {"layers.0.ffn.experts.w1": torch.zeros(2, 3, 2)},
        str(tmp_path / "model.safetensors"),
    )
    ep_dir = tmp_path / "ep_experts"
    ep_dir.mkdir()
    torch.save(
        {"layers.0.ffn.experts.w1": torch.zeros(2, 3, 2)}, ep_dir / "rank_0.pt"
    )
    sd = _load_state_dict(tmp_path)

    merged = _merge_ep_expert_slices(sd, ep_dir, expected_expert_rows=2)
    assert merged["layers.0.ffn.experts.w1"].shape == (2, 3, 2)

    with pytest.raises(SystemExit, match="dsv4_n_routed_experts 4"):
        _merge_ep_expert_slices(sd, ep_dir, expected_expert_rows=4)

    # Neither source available -> loud warning instead of a silent pass.
    _merge_ep_expert_slices(sd, ep_dir)
    assert "could not validate the merged EP expert count" in capsys.readouterr().out


def test_convert_checkpoint_inspect_ignores_malformed_config(
    tmp_path, capsys, monkeypatch
):
    # --inspect is a dry run and must not parse config.json.
    import sys

    from verl_speco.tools import convert_checkpoint

    safetensors_torch.save_file(
        {
            "layers.0.ffn.router.weight": torch.zeros(4, 2),
            "layers.0.ffn.experts.w1": torch.zeros(2, 3, 2),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_checkpoint", "--in", str(tmp_path), "--inspect"],
    )

    convert_checkpoint.main()

    assert "Sample mtp.* keys" in capsys.readouterr().out
