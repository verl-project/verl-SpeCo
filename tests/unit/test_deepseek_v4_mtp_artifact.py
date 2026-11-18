# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Strict native MTP TP merge, serialization and fresh initialization."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl_speco.models.deepseek_v4 import export as native
from verl_speco.models.deepseek_v4 import mtp_artifact as mtp
from verl_speco.models.deepseek_v4.artifact import (
    iter_draft_weights,
    save_draft_artifact,
)
from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract


def rank_state(rank):
    state = {
        key: torch.ones(4, 4) for key in native.VLLM_ASCEND_REQUIRED_TRAINABLE_SOURCES
    }
    p = native.LAYER_PREFIX
    state.update(
        {
            p + name: torch.ones(4)
            for name in (
                "enorm.weight",
                "hnorm.weight",
                "shared_head.norm.weight",
                "mtp_block.input_layernorm.weight",
                "mtp_block.post_attention_layernorm.weight",
            )
        }
    )
    state.update(
        {
            p + name: torch.ones(2)
            for name in (
                "mtp_block.self_attn.q_norm.weight",
                "mtp_block.self_attn.kv_norm.weight",
            )
        }
    )
    state.update(
        {
            p + name: torch.ones(2, 4)
            for name in (
                "mtp_block.self_attn.wq_a.weight",
                "mtp_block.self_attn.wkv.weight",
                "mtp_block.mlp.gate.weight",
            )
        }
    )
    state.update(
        {p + name: torch.ones(4, 4) for name in ("e_proj.weight", "h_proj.weight")}
    )
    state.update(
        {
            "model.embed_tokens.weight": torch.full((6, 4), float(rank + 1)),
            p + "mtp_block.self_attn.attn_sink": torch.tensor([float(rank + 1)]),
            p + "mtp_block.self_attn.wq_b.weight": torch.full((2, 2), float(rank + 1)),
            p + "mtp_block.self_attn.wo_a.weight": torch.full(
                (1, 2, 4), float(rank + 1)
            ),
            p + "mtp_block.self_attn.wo_b.weight": torch.full((4, 2), float(rank + 1)),
            p + "mtp_block.mlp.shared_experts.gate_up_proj.weight": torch.full(
                (2, 4), float(rank + 1)
            ),
            p + "mtp_block.mlp.shared_experts.down_proj.weight": torch.full(
                (4, 1), float(rank + 1)
            ),
            p + "mtp_block.mlp.experts.w13_weight": torch.full(
                (2, 4, 2), float(rank + 1)
            ),
            p + "mtp_block.mlp.experts.w2_weight": torch.full(
                (2, 1, 4), float(rank + 1)
            ),
        }
    )
    return state


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_nextn_predict_layers": 1,
                "hidden_size": 4,
                "hc_mult": 2,
                "vocab_size": 12,
            }
        )
    )
    for rank in range(2):
        torch.save(rank_state(rank), root / f"model_states_rank_{rank}.pt")
    topology = {
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "world_size": 2,
        "rank_order": "dp_major_tp_minor",
    }
    (root / "parallel_topology.json").write_text(json.dumps(topology))
    (root / "speco_ds4.json").write_text(
        json.dumps(
            {
                "identity": {
                    "contract": DeepSeekV4MTPContract.from_model(model).identity(),
                    "source_fingerprint": mtp.mtp_implementation_fingerprint(),
                },
                "topology": topology,
                "progress": {"global_step": 1},
            }
        )
    )
    return root, model


@pytest.mark.parametrize(
    "suffix",
    [
        "hc_head_fn",
        "e_proj.weight",
        "mtp_block.hc_ffn_scale",
        "mtp_block.self_attn.wq_a.weight",
        "mtp_block.mlp.gate.weight",
    ],
)
def test_mtp_export_rejects_replica_drift_before_writing(checkpoint, tmp_path, suffix):
    root, model = checkpoint
    assert mtp.validate_mtp_replicas(root, 0)["tp_ranks"] == 2
    state = rank_state(1)
    state[native.LAYER_PREFIX + suffix].reshape(-1)[0] += 1
    torch.save(state, root / "model_states_rank_1.pt")
    with pytest.raises(ValueError, match="replicated tensor values differ"):
        mtp.export_mtp_checkpoint(
            checkpoint=root, model=model, output=tmp_path / "invalid"
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("serialization", ["pytorch", "safetensors"])
def test_standard_mtp_export_has_distinct_identity_and_complete_readback(
    checkpoint, tmp_path, serialization
):
    from verl_speco.export_deepseek_v4_draft import export_checkpoint

    root, model = checkpoint
    out = tmp_path / "artifact"
    report = export_checkpoint(
        checkpoint=str(root),
        model=str(model),
        output=str(out),
        format="draft-" + serialization,
    )
    assert report["status"] == "passed" and report["algorithm"] == "MTP"
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["weight_layout"] == "deepseek_v4_mtp_v1"
    values = dict(iter_draft_weights(out))
    assert values["mtp.0.emb.tok_emb.weight"].shape == (12, 4)
    assert values["mtp.0.ffn.experts.0.w1.weight"].shape == (2, 4)
    assert torch.equal(values["mtp.0.e_proj.weight"], torch.ones(4, 4))


def test_mtp_artifact_init_uses_native_loader_shared_target_and_postprocessing(
    tmp_path, monkeypatch
):
    base = tmp_path / "base"
    base.mkdir()
    config = {"model_type": "deepseek_v4"}
    (base / "config.json").write_text(json.dumps(config))
    save_file(
        {"embed.weight": torch.ones(3, 2), "head.weight": torch.full((3, 2), 2.0)},
        str(base / "model.safetensors"),
    )
    artifact = tmp_path / "artifact"
    save_draft_artifact(
        {"mtp.0.weight": torch.full((2, 2), 7.0)},
        output=artifact,
        config=config,
        step=1,
        weight_layout="deepseek_v4_mtp_v1",
    )
    model = torch.nn.Module()
    for name in ("draft", "embed", "head"):
        model.register_parameter(name, torch.nn.Parameter(torch.zeros(1)))
    consumed = []

    def load(stream):
        consumed.extend(stream)
        return {"draft"}

    model.load_weights = load
    wrapper = SimpleNamespace(
        model=model,
        vllm_config=None,
        _weights_postprocessed=False,
        _set_current_vllm_config=lambda _: nullcontext(),
        _load_shared_main_weights=lambda values, **kwargs: {
            values_to_alias[key] for key in values
        },
        _keep_hc_parameters_float32=lambda: None,
        _ensure_vllm_ascend_tp_equivalent=lambda **kwargs: None,
    )
    values_to_alias = {
        "model.embed_tokens.weight": "embed",
        "model.layers.0.shared_head.head.weight": "head",
    }
    import verl_speco.models.deepseek_v4.artifact_loading as loading

    finished = []
    monkeypatch.setattr(
        loading, "_finish_loading", lambda value: finished.append(value)
    )
    result = mtp.load_mtp_artifact(
        SimpleNamespace(model=wrapper), artifact, base_checkpoint=base
    )
    assert result["loaded"] == 3 and finished == [wrapper]
    assert consumed[0][0] == "mtp.0.weight" and consumed[0][1].eq(7).all()


def test_moe_shared_expert_alias_is_checked_against_its_own_tp_shard(checkpoint):
    root, _ = checkpoint
    canonical = native.LAYER_PREFIX + "mtp_block.mlp.shared_experts.down_proj.weight"
    alias = canonical.replace(
        ".mlp.shared_experts.", ".mlp.experts._shared_experts._layer."
    )
    for rank in range(2):
        state = rank_state(rank)
        state[alias] = state[canonical].clone()
        torch.save(state, root / f"model_states_rank_{rank}.pt")
    assert mtp.validate_mtp_replicas(root, 0)["tp_ranks"] == 2
    state[alias][0, 0] += 1
    torch.save(state, root / "model_states_rank_1.pt")
    with pytest.raises(ValueError, match="shared-expert alias differs"):
        mtp.validate_mtp_replicas(root, 0)
