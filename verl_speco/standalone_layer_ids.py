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
"""Layer-ID normalization shared by standalone launch and direct Producer."""

from __future__ import annotations

from collections.abc import Sequence

_DECODER_INDEX_ALGORITHMS = frozenset({"DFLASH", "DFLASH2", "DSPARK", "DOMINO"})


def normalize_standalone_layer_ids(
    algorithm: str,
    target_layer_ids: Sequence[int] | None,
    vllm_aux_hidden_state_layer_ids: Sequence[int] | None,
    *,
    default_vllm_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return training IDs and vLLM output IDs using algorithm-native semantics."""

    algorithm = str(algorithm).strip().upper()
    shift_vllm_ids = algorithm in _DECODER_INDEX_ALGORITHMS
    target_ids = (
        tuple(int(layer_id) for layer_id in target_layer_ids)
        if target_layer_ids is not None
        else None
    )
    vllm_ids = (
        tuple(int(layer_id) for layer_id in vllm_aux_hidden_state_layer_ids)
        if vllm_aux_hidden_state_layer_ids is not None
        else None
    )
    if target_ids is None and vllm_ids is None and default_vllm_ids is not None:
        vllm_ids = tuple(int(layer_id) for layer_id in default_vllm_ids)
    if target_ids is None and vllm_ids is None:
        raise ValueError(
            "Configure target_layer_ids or vllm_aux_hidden_state_layer_ids"
        )
    if target_ids is not None and (
        not target_ids or any(layer_id < 0 for layer_id in target_ids)
    ):
        raise ValueError("target_layer_ids must contain non-negative decoder IDs")
    if vllm_ids is not None and (
        not vllm_ids or any(layer_id < 1 for layer_id in vllm_ids)
    ):
        raise ValueError(
            "vllm_aux_hidden_state_layer_ids must contain positive output IDs"
        )
    if target_ids is None:
        assert vllm_ids is not None
        target_ids = (
            tuple(layer_id - 1 for layer_id in vllm_ids) if shift_vllm_ids else vllm_ids
        )
    expected_vllm_ids = (
        tuple(layer_id + 1 for layer_id in target_ids) if shift_vllm_ids else target_ids
    )
    if vllm_ids is None:
        vllm_ids = expected_vllm_ids
    elif vllm_ids != expected_vllm_ids:
        raise ValueError(
            "Standalone layer IDs do not match the algorithm convention: "
            f"algorithm={algorithm}, "
            f"target_layer_ids={list(target_ids)}, "
            f"vllm_aux_hidden_state_layer_ids={list(vllm_ids)}, "
            f"expected={list(expected_vllm_ids)}"
        )
    return target_ids, vllm_ids


__all__ = ["normalize_standalone_layer_ids"]
