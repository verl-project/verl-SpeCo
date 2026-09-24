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

from verl_speco.standalone_layer_ids import normalize_standalone_layer_ids


@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH", "DFLASH2", "DOMINO"])
def test_dflash_family_maps_decoder_ids_to_vllm_output_ids(algorithm) -> None:
    assert normalize_standalone_layer_ids(algorithm, [0, 8, 16], None) == (
        (0, 8, 16),
        (1, 9, 17),
    )
    assert normalize_standalone_layer_ids(algorithm, None, [1, 9, 17]) == (
        (0, 8, 16),
        (1, 9, 17),
    )


def test_eagle3_keeps_training_and_vllm_ids_identical() -> None:
    assert normalize_standalone_layer_ids("EAGLE3", [1, 9, 17], None) == (
        (1, 9, 17),
        (1, 9, 17),
    )
    assert normalize_standalone_layer_ids("EAGLE3", None, [1, 9, 17]) == (
        (1, 9, 17),
        (1, 9, 17),
    )


def test_layer_id_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="algorithm convention"):
        normalize_standalone_layer_ids("DSPARK", [0, 8], [1, 10])
