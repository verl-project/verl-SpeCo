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
"""Export contract: a LiLiCorr checkpoint must round-trip to the vLLM layout."""

from __future__ import annotations

import json
from types import SimpleNamespace

from verl_speco.trainer.draft_training_loop import (
    _rewrite_standalone_block_runtime_config,
)


def test_export_writes_vllm_servable_config(tmp_path):
    training_config = {
        "model_type": "lilicorr",
        "architectures": ["LiLiCorrDraftModel"],
        "hidden_size": 4096,
        "num_hidden_layers": 5,
        "vocab_size": 151936,
        "draft_vocab_size": 151936,
        "block_size": 16,
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "lilicorr_candidate_topk": 8,
        "lilicorr_hidden_size": 1024,
        "lilicorr_factor_dim": 1024,
        "lilicorr_num_layers": 2,
        "lilicorr_num_heads": 8,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_logit_scale": 8.0,
        "lilicorr_vector_eps": 1e-4,
        "sample_from_anchor": False,
        "causal": True,
        "dflash_config": {"block_size": 16},
    }
    (tmp_path / "config.json").write_text(json.dumps(training_config))

    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="lilicorr"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    model_path=None,
                    training={"lilicorr_target_layer_ids": [1, 9, 17, 25, 33]},
                )
            )
        ),
    )
    _rewrite_standalone_block_runtime_config(trainer, str(tmp_path))
    exported = json.loads((tmp_path / "config.json").read_text())

    assert exported["architectures"] == ["LiLiCorrDraftModel"]
    # Released layout keeps the target family so transformers AutoConfig parses
    # it; vLLM dispatches on the architecture instead.
    assert exported["model_type"] == "qwen3"
    assert exported["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert exported["draft_vocab_size"] == exported["vocab_size"]
    nested = exported["dflash_config"]
    for key in (
        "lilicorr_candidate_topk",
        "lilicorr_hidden_size",
        "lilicorr_factor_dim",
        "lilicorr_num_layers",
        "lilicorr_num_heads",
        "lilicorr_mlp_ratio",
        "lilicorr_logit_scale",
        "lilicorr_vector_eps",
        "mask_token_id",
    ):
        assert key in nested, key
    assert nested["target_layer_ids"] == [1, 9, 17, 25, 33]
