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
# Adapted original artifact validation; no external checkout imports.
import json
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from verl_speco.models.deepseek_v4.export import (
    LAYER_PREFIX,
    _add_attention_projection_tp_shards,
    _add_eh_projection_tp_shards,
    _dspark_projection_layout,
    _load_export_weights,
    _validate_replicated_tensors,
)
from verl_speco.models.deepseek_v4.export_validation import validate_tp_export_replicas


@pytest.mark.parametrize(
    "suffix",
    [
        "hc_attn_fn",
        "hc_ffn_base",
        "self_attn.wq_a.weight",
        "self_attn.wkv.weight",
        "self_attn.q_norm.weight",
        "mlp.gate.weight",
        "main_norm.weight",
    ],
)
def test_full_export_guard_rejects_previously_unchecked_replica_drift(tmp_path, suffix):
    (tmp_path / "parallel_topology.json").write_text(
        json.dumps(
            {
                "tensor_parallel_size": 2,
                "data_parallel_size": 1,
                "world_size": 2,
                "rank_order": "dp_major_tp_minor",
            }
        )
    )
    for rank in range(2):
        state = _rank_state(rank)
        torch.save(state, tmp_path / f"model_states_rank_{rank}.pt")
    # Markov weights in this fixture are vocab shards, while all listed
    # parameters are actual replicas. Legitimate attention/MoE shards differ.
    report = validate_tp_export_replicas(tmp_path, dp_rank=0, target_shapes=None)
    assert report["tp_ranks"] == 2
    state[f"model.model.layers.43.{suffix}"].reshape(-1)[0] += 1
    torch.save(state, tmp_path / "model_states_rank_1.pt")
    with pytest.raises(ValueError, match="replicated tensor values differ"):
        validate_tp_export_replicas(tmp_path, dp_rank=0, target_shapes=None)


def test_full_export_guard_requires_explicit_tp_dp_topology(tmp_path):
    for rank in range(2):
        torch.save(_rank_state(rank), tmp_path / f"model_states_rank_{rank}.pt")
    with pytest.raises(ValueError, match="parallel_topology"):
        validate_tp_export_replicas(tmp_path, dp_rank=0, target_shapes=None)


# Synthetic TP fixture adapted from the original DSpark export validation.
def _filled(shape: tuple[int, ...], value: int) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _rank_state(tp_rank: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    hidden = 4
    experts = 2
    for stage, layer_idx in enumerate((43, 44, 45)):
        prefix = f"model.model.layers.{layer_idx}."
        base = 100 * stage
        state.update(
            {
                f"{prefix}hc_attn_fn": _filled((2, 8), base + 1),
                f"{prefix}hc_attn_base": _filled((2,), base + 2),
                f"{prefix}hc_attn_scale": _filled((1,), base + 3),
                f"{prefix}hc_ffn_fn": _filled((2, 8), base + 4),
                f"{prefix}hc_ffn_base": _filled((2,), base + 5),
                f"{prefix}hc_ffn_scale": _filled((1,), base + 6),
                f"{prefix}self_attn.attn_sink": _filled((1,), base + 10 + tp_rank),
                f"{prefix}self_attn.wq_a.weight": _filled((2, hidden), base + 11),
                f"{prefix}self_attn.q_norm.weight": _filled((2,), base + 12),
                f"{prefix}self_attn.wq_b.weight": _filled((2, 2), base + 20 + tp_rank),
                f"{prefix}self_attn.wkv.weight": _filled((2, hidden), base + 13),
                f"{prefix}self_attn.kv_norm.weight": _filled((2,), base + 14),
                f"{prefix}self_attn.wo_a.weight": _filled(
                    (1, 2, 2), base + 30 + tp_rank
                ),
                f"{prefix}self_attn.wo_b.weight": _filled(
                    (hidden, 2), base + 40 + tp_rank
                ),
                f"{prefix}mlp.gate.weight": _filled((experts, hidden), base + 15),
                f"{prefix}mlp.gate.e_score_correction_bias": _filled(
                    (experts,), base + 16
                ),
                f"{prefix}mlp.shared_experts.gate_up_proj.weight": torch.cat(
                    (
                        _filled((1, hidden), base + 50 + tp_rank),
                        _filled((1, hidden), base + 60 + tp_rank),
                    ),
                    dim=0,
                ),
                f"{prefix}mlp.shared_experts.down_proj.weight": _filled(
                    (hidden, 1), base + 70 + tp_rank
                ),
                f"{prefix}mlp.experts.routed_experts.w13_weight": torch.cat(
                    (
                        _filled((experts, hidden, 1), base + 80 + tp_rank),
                        _filled((experts, hidden, 1), base + 90 + tp_rank),
                    ),
                    dim=2,
                ),
                f"{prefix}mlp.experts.routed_experts.w2_weight": _filled(
                    (experts, 1, hidden), base + 95 + tp_rank
                ),
                f"{prefix}input_layernorm.weight": _filled((hidden,), base + 17),
                f"{prefix}post_attention_layernorm.weight": _filled(
                    (hidden,), base + 18
                ),
            }
        )

    state["model.model.layers.43.main_proj.weight"] = _filled((hidden, 12), 401)
    state["model.model.layers.43.main_norm.weight"] = _filled((hidden,), 402)
    state["model.model.layers.45.norm.weight"] = _filled((hidden,), 403)
    state["model.model.layers.45.markov_head.markov_w1.weight"] = _filled(
        (3, 2), 410 + tp_rank
    )
    state["model.model.layers.45.markov_head.markov_w2.weight"] = _filled(
        (3, 2), 420 + tp_rank
    )
    state["confidence_head.weight"] = _filled((1, 7), 499)
    state["confidence_head.bias"] = _filled((1,), 500)
    state["model.model.hc_head_fn"] = _filled((2, 8), 430)
    state["model.model.hc_head_base"] = _filled((2,), 431)
    state["model.model.hc_head_scale"] = _filled((1,), 432)
    return state


class TestDS4TPProjectionExport(unittest.TestCase):
    def test_bounded_bf16_w1_ulp_exception_checks_every_element(self):
        w1 = "model.layers.45.markov_head.markov_w1.weight"
        reference = torch.tensor(
            [0.34765625, -0.34765625, 0.34765625], dtype=torch.bfloat16
        )
        candidate = torch.tensor(
            [0.349609375, -0.349609375, 0.34765625], dtype=torch.bfloat16
        )

        def validate(key=w1, **kwargs):
            _validate_replicated_tensors(
                {key: reference},
                {key: candidate},
                [key],
                candidate_path=Path("rank5.pt"),
                allow_markov_w1_one_ulp=True,
                **kwargs,
            )

        original = reference.clone()
        with self.assertRaises(ValueError):
            validate()
        validate(markov_w1_max_differing_elements=2)
        self.assertTrue(torch.equal(reference, original))
        for key in (
            "model.layers.45.markov_head.markov_w2.weight",
            "model.layers.45.other.weight",
        ):
            with self.assertRaises(ValueError):
                validate(key, markov_w1_max_differing_elements=2)
        candidate[2] = 0.349609375
        with self.assertRaises(ValueError):
            validate(markov_w1_max_differing_elements=2)
        candidate[2] = reference[2]
        candidate[1] = -0.3515625
        with self.assertRaises(ValueError):
            validate(markov_w1_max_differing_elements=2)
        for invalid in (0, -1):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                validate(markov_w1_max_differing_elements=invalid)
        candidate[1] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate(markov_w1_max_differing_elements=2)

    def test_single_bf16_w1_ulp_exception_does_not_allow_other_drift(self):
        w1 = "model.layers.45.markov_head.markov_w1.weight"
        w2 = "model.layers.45.markov_head.markov_w2.weight"
        reference = torch.tensor([[0.34765625, -0.34765625]], dtype=torch.bfloat16)
        candidate = reference.clone()
        candidate[0, 0] = 0.349609375

        def validate(key, left, right, **kwargs):
            _validate_replicated_tensors(
                {key: left},
                {key: right},
                [key],
                candidate_path=Path("rank4.pt"),
                **kwargs,
            )

        with self.assertRaises(ValueError):
            validate(w1, reference, candidate)
        validate(w1, reference, candidate, allow_markov_w1_one_ulp=True)
        with self.assertRaises(ValueError):
            validate(w2, reference, candidate, allow_markov_w1_one_ulp=True)
        with self.assertRaises(ValueError):
            validate(
                w1, reference.float(), candidate.float(), allow_markov_w1_one_ulp=True
            )
        candidate[0, 1] = -0.349609375
        with self.assertRaises(ValueError):
            validate(w1, reference, candidate, allow_markov_w1_one_ulp=True)
        candidate = reference.clone()
        candidate[0, 0] = 0.3515625
        with self.assertRaises(ValueError):
            validate(w1, reference, candidate, allow_markov_w1_one_ulp=True)
        candidate[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate(w1, reference, candidate, allow_markov_w1_one_ulp=True)

    def test_markov_replica_tolerance_is_explicit_and_scoped(self):
        key = "model.layers.45.markov_head.markov_w1.weight"
        reference = {key: torch.zeros(2, 2)}
        candidate = {key: torch.full((2, 2), 3.814697265625e-6)}
        with self.assertRaisesRegex(ValueError, "differing_elements=4/4"):
            _validate_replicated_tensors(
                reference, candidate, [key], candidate_path=Path("rank1.pt")
            )
        _validate_replicated_tensors(
            reference,
            candidate,
            [key],
            candidate_path=Path("rank1.pt"),
            markov_replica_atol=4e-6,
        )
        with self.assertRaises(ValueError):
            _validate_replicated_tensors(
                reference,
                candidate,
                [key],
                candidate_path=Path("rank1.pt"),
                markov_replica_atol=1e-6,
            )
        routing_key = "model.layers.45.mlp.gate.e_score_correction_bias"
        with self.assertRaises(ValueError):
            _validate_replicated_tensors(
                {routing_key: reference[key]},
                {routing_key: candidate[key]},
                [routing_key],
                candidate_path=Path("rank1.pt"),
                markov_replica_atol=4e-6,
            )
        reference[key].fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            _validate_replicated_tensors(
                reference,
                reference,
                [key],
                candidate_path=Path("rank1.pt"),
                markov_replica_atol=4e-6,
            )

    def test_dspark_merges_projection_but_keeps_full_markov_replicas(self):
        state = {
            "model.layers.43.main_proj.weight": torch.zeros(2, 12),
            "model.layers.45.markov_head.markov_w1.weight": torch.zeros(6, 2),
            "model.layers.45.markov_head.markov_w2.weight": torch.zeros(6, 2),
        }
        shapes = {
            "mtp.0.main_proj.weight": (4, 12),
            "mtp.2.markov_head.markov_w1.weight": (6, 2),
            "mtp.2.markov_head.markov_w2.weight": (6, 2),
        }
        sharded, replicated = _dspark_projection_layout(state, [43, 44, 45], 2, shapes)
        self.assertEqual(sharded, {"model.layers.43.main_proj.weight": 0})
        self.assertEqual(len(replicated), 2)
        shapes["mtp.0.main_proj.weight"] = (5, 12)
        with self.assertRaisesRegex(ValueError, "Cannot identify DSpark TP layout"):
            _dspark_projection_layout(state, [43, 44, 45], 2, shapes)

    def test_dspark_export_reconstructs_projection_and_validates_markov_replicas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parallel_topology.json").write_text(
                json.dumps(
                    {
                        "tensor_parallel_size": 2,
                        "data_parallel_size": 1,
                        "world_size": 2,
                        "rank_order": "dp_major_tp_minor",
                    }
                ),
                encoding="utf-8",
            )
            for rank in range(2):
                state = _rank_state(rank)
                state["model.model.layers.43.main_proj.weight"] = torch.full(
                    (2, 12), 401.0 + rank
                )
                for name in ("markov_w1", "markov_w2"):
                    state[f"model.model.layers.45.markov_head.{name}.weight"] = (
                        torch.full((6, 2), 410.0)
                    )
                torch.save(state, root / f"model_states_rank_{rank}.pt")
            shapes = {
                "mtp.0.main_proj.weight": (4, 12),
                "mtp.2.markov_head.markov_w1.weight": (6, 2),
                "mtp.2.markov_head.markov_w2.weight": (6, 2),
            }
            weights = _load_export_weights(root, "dspark", target_shapes=shapes)
            self.assertEqual(
                weights["mtp.0.main_proj.weight"][:, 0].tolist(), [401, 401, 402, 402]
            )
            self.assertEqual(
                tuple(weights["mtp.2.markov_head.markov_w1.weight"].shape), (6, 2)
            )
            state["model.model.layers.45.markov_head.markov_w1.weight"][0, 0] += 1
            torch.save(state, root / "model_states_rank_1.pt")
            with self.assertRaises(ValueError):
                _load_export_weights(root, "dspark", target_shapes=shapes)

            for rank in range(2):
                state = torch.load(
                    root / f"model_states_rank_{rank}.pt", weights_only=False
                )
                for name in ("markov_w1", "markov_w2"):
                    key = f"model.model.layers.45.markov_head.{name}.weight"
                    state[key] = torch.full((6, 2), 0.34765625, dtype=torch.bfloat16)
                if rank == 1:
                    state["model.model.layers.45.markov_head.markov_w1.weight"][
                        0, :
                    ] = 0.349609375
                torch.save(state, root / f"model_states_rank_{rank}.pt")
            with self.assertRaises(ValueError):
                _load_export_weights(
                    root, "dspark", target_shapes=shapes, allow_markov_w1_one_ulp=True
                )
            weights = _load_export_weights(
                root,
                "dspark",
                target_shapes=shapes,
                allow_markov_w1_one_ulp=True,
                markov_w1_max_differing_elements=2,
            )
            self.assertTrue(
                torch.all(weights["mtp.2.markov_head.markov_w1.weight"] == 0.34765625)
            )

    def test_detects_output_sharded_attention_weights(self):
        shards = {}
        prefix = f"{LAYER_PREFIX}mtp_block.self_attn."
        state = {
            f"{prefix}wq_a.weight": torch.empty(192, 4096),
            f"{prefix}q_norm.weight": torch.empty(1536),
            f"{prefix}wkv.weight": torch.empty(32, 4096),
            f"{prefix}kv_norm.weight": torch.empty(256),
        }
        _add_attention_projection_tp_shards(
            shards,
            state,
            tp_size=8,
            source=Path("rank0.pt"),
        )
        self.assertEqual(
            shards,
            {
                f"{prefix}wq_a.weight": 0,
                f"{prefix}wkv.weight": 0,
            },
        )

    def test_leaves_replicated_attention_weights_unmerged(self):
        shards = {}
        prefix = f"{LAYER_PREFIX}mtp_block.self_attn."
        state = {
            f"{prefix}wq_a.weight": torch.empty(16, 32),
            f"{prefix}q_norm.weight": torch.empty(16),
            f"{prefix}wkv.weight": torch.empty(8, 32),
            f"{prefix}kv_norm.weight": torch.empty(8),
        }
        _add_attention_projection_tp_shards(
            shards,
            state,
            tp_size=2,
            source=Path("rank0.pt"),
        )
        self.assertEqual(shards, {})

    def test_rejects_inconsistent_attention_projection_shard(self):
        prefix = f"{LAYER_PREFIX}mtp_block.self_attn."
        state = {
            f"{prefix}wq_a.weight": torch.empty(3, 32),
            f"{prefix}q_norm.weight": torch.empty(16),
            f"{prefix}wkv.weight": torch.empty(4, 32),
            f"{prefix}kv_norm.weight": torch.empty(8),
        }
        with self.assertRaisesRegex(
            ValueError,
            "Cannot identify TP output shard",
        ):
            _add_attention_projection_tp_shards(
                {},
                state,
                tp_size=2,
                source=Path("rank0.pt"),
            )

    def test_detects_output_sharded_eh_weights(self):
        shards = {}
        state = {
            f"{LAYER_PREFIX}e_proj.weight": torch.empty(512, 4096),
            f"{LAYER_PREFIX}h_proj.weight": torch.empty(512, 4096),
        }
        _add_eh_projection_tp_shards(
            shards,
            state,
            tp_size=8,
            source=Path("rank0.pt"),
        )
        self.assertEqual(
            shards,
            {
                f"{LAYER_PREFIX}e_proj.weight": 0,
                f"{LAYER_PREFIX}h_proj.weight": 0,
            },
        )

    def test_leaves_replicated_square_weights_unmerged(self):
        shards = {}
        state = {
            f"{LAYER_PREFIX}e_proj.weight": torch.empty(16, 16),
            f"{LAYER_PREFIX}h_proj.weight": torch.empty(16, 16),
        }
        _add_eh_projection_tp_shards(
            shards,
            state,
            tp_size=8,
            source=Path("rank0.pt"),
        )
        self.assertEqual(shards, {})

    def test_rejects_inconsistent_projection_shard(self):
        state = {
            f"{LAYER_PREFIX}e_proj.weight": torch.empty(513, 4096),
            f"{LAYER_PREFIX}h_proj.weight": torch.empty(512, 4096),
        }
        with self.assertRaisesRegex(ValueError, "Cannot identify TP output shard"):
            _add_eh_projection_tp_shards(
                {},
                state,
                tp_size=8,
                source=Path("rank0.pt"),
            )

    def test_accepts_equal_replicated_tensor(self):
        key = "model.layers.43.mlp.gate.e_score_correction_bias"
        state = {key: torch.tensor([1.0, 2.0], dtype=torch.float32)}
        _validate_replicated_tensors(
            state,
            {key: state[key].clone()},
            [key],
            candidate_path=Path("rank1.pt"),
        )

    def test_rejects_diverged_replicated_tensor(self):
        key = "model.layers.43.mlp.gate.e_score_correction_bias"
        reference = {key: torch.tensor([1.0, 2.0], dtype=torch.float32)}
        candidate = {key: torch.tensor([1.0, 2.25], dtype=torch.float32)}
        with self.assertRaisesRegex(
            ValueError,
            r"tensor=model\.layers\.43.*max_abs_delta=0\.25",
        ):
            _validate_replicated_tensors(
                reference,
                candidate,
                [key],
                candidate_path=Path("rank1.pt"),
            )


if __name__ == "__main__":
    unittest.main()


def test_tp_merge_to_standard_artifact_selects_only_one_dp_replica(tmp_path):
    from verl_speco.models.deepseek_v4.artifact import (
        iter_draft_weights,
        save_draft_artifact,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "parallel_topology.json").write_text(
        json.dumps(
            {
                "world_size": 4,
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "rank_order": "dp_major_tp_minor",
            }
        )
    )
    for dp in range(2):
        for tp in range(2):
            state = _rank_state(tp)
            # Keep the replicated Markov projection consistent for strict export.
            state["model.model.layers.45.markov_head.markov_w1.weight"].fill_(410)
            state["model.model.layers.45.markov_head.markov_w2.weight"].fill_(420)
            state["model.model.layers.43.main_norm.weight"].fill_(402 + dp)
            torch.save(state, checkpoint / f"model_states_rank_{dp * 2 + tp}.pt")
    expected = _load_export_weights(checkpoint, "dspark", data_parallel_rank=1)
    assert torch.equal(expected["mtp.0.main_norm.weight"], torch.full((4,), 403.0))
    output = tmp_path / "draft_step_1"
    save_draft_artifact(
        expected, output=output, config={}, step=1, max_shard_bytes=1024
    )
    actual = dict(iter_draft_weights(output))
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
