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
"""Tests for the SGLang last-layer pre-norm fail-closed guard.

SGLang's aux/context capture never applies the target's final norm, so a
target_layer_id equal to the last layer (or -1) is captured with different
semantics than the offline / old-logprob paths. These tests lock in that SPECO
refuses that combination unless the user opts in.
"""

from __future__ import annotations

import pytest

from verl_speco.integration.oldlogprob_layer_ids import (
    assert_sglang_aux_last_layer_norm_safe,
)


def test_intermediate_layers_pass() -> None:
    # Default-style ids never hit the last layer -> no raise.
    assert_sglang_aux_last_layer_norm_safe(
        [2, 18, 33],
        num_hidden_layers=36,
        collect_from_sgl=True,
        allow_prenorm_last=False,
    )


@pytest.mark.parametrize("layer_ids", [[40, 41, 42], [42], [2, 18, 42]])
def test_last_layer_is_rejected(layer_ids) -> None:
    # 42 == num_hidden_layers - 1 (the deepseek_v4 / glm_5.2 DSpark recipe pattern).
    with pytest.raises(ValueError, match="WITHOUT the target's final norm"):
        assert_sglang_aux_last_layer_norm_safe(
            layer_ids,
            num_hidden_layers=43,
            collect_from_sgl=True,
            allow_prenorm_last=False,
        )


def test_embedding_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="WITHOUT the target's final norm"):
        assert_sglang_aux_last_layer_norm_safe(
            [-1, 10],
            num_hidden_layers=36,
            collect_from_sgl=True,
            allow_prenorm_last=False,
        )


def test_opt_out_allows_last_layer() -> None:
    # Self-consistent SGLang-only train+serve: pre-norm on both sides is fine.
    assert_sglang_aux_last_layer_norm_safe(
        [40, 41, 42],
        num_hidden_layers=43,
        collect_from_sgl=True,
        allow_prenorm_last=True,
    )


def test_not_sgl_collection_is_ignored() -> None:
    # old-logprob collection handles the last layer as post-norm, so it is unaffected.
    assert_sglang_aux_last_layer_norm_safe(
        [40, 41, 42],
        num_hidden_layers=43,
        collect_from_sgl=False,
        allow_prenorm_last=False,
    )


def test_unresolvable_inputs_skip() -> None:
    # Best-effort: no raise when layer ids or target depth are unavailable.
    assert_sglang_aux_last_layer_norm_safe(
        None, 43, collect_from_sgl=True, allow_prenorm_last=False
    )
    assert_sglang_aux_last_layer_norm_safe(
        [40, 41, 42], None, collect_from_sgl=True, allow_prenorm_last=False
    )


def test_embedding_id_rejected_even_without_target_depth() -> None:
    # -1 (the embedding) is divergent regardless of depth, so it must still be
    # rejected when num_hidden_layers cannot be resolved.
    with pytest.raises(ValueError, match="WITHOUT the target's final norm"):
        assert_sglang_aux_last_layer_norm_safe(
            [-1, 10],
            num_hidden_layers=None,
            collect_from_sgl=True,
            allow_prenorm_last=False,
        )
