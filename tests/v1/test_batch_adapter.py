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

torch = pytest.importorskip("torch")
tensordict = pytest.importorskip("tensordict")
TensorDict = tensordict.TensorDict

from verl_speco.trainer.v1.batch_adapter import to_legacy_padded_batch


def _batch(multimodal=False):
    prompts = [torch.tensor([11, 12]), torch.tensor([21, 22, 23, 24])]
    responses = [torch.tensor([31, 32, 33, 34]), torch.tensor([41, 42])]
    inputs = [torch.cat(pair) for pair in zip(prompts, responses)]
    positions = [torch.arange(6) for _ in inputs]
    if multimodal:
        positions = [torch.stack((row, row + 10, row + 20)) for row in positions]
    # A list of rows also exercises the adapter without requiring a particular
    # nested-tensor layout for multimodal position ids.
    class Rows(list):
        def unbind(self):
            return tuple(self)
    return {key: Rows(value) for key, value in {
        "prompts": prompts, "responses": responses, "input_ids": inputs,
        "position_ids": positions,
        "response_mask": [torch.tensor([1, 0, 1, 1]), torch.ones(2, dtype=torch.long)],
        "rollout_log_probs": [torch.arange(4).float(), torch.arange(2).float()],
        "routed_experts": [torch.arange(12).reshape(6, 2) for _ in inputs],
    }.items()}


@pytest.mark.parametrize("multimodal", [False, True])
def test_variable_prompt_boundaries_preserve_all_sequence_fields(multimodal):
    data = _batch(multimodal)
    result = to_legacy_padded_batch(data, 99)
    assert result["prompts"].tolist() == [[99, 99, 11, 12], [21, 22, 23, 24]]
    assert result["attention_mask"][:, :4].sum(-1).tolist() == [2, 4]
    for i in range(2):
        valid = result["attention_mask"][i].bool()
        assert torch.equal(result["input_ids"][i][valid], data["input_ids"][i])
        assert torch.equal(result["position_ids"][i][..., valid], data["position_ids"][i])
        assert torch.equal(result["routed_experts"][i][valid], data["routed_experts"][i])
    assert result["response_mask"].tolist() == [[1, 0, 1, 1], [1, 1, 0, 0]]
    assert data["prompts"][0].tolist() == [11, 12]


def test_upstream_unpad_and_oldlogprob_slice_round_trip():
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    rows = _batch()
    nested = TensorDict({key: torch.nested.as_nested_tensor(value, layout=torch.jagged)
                         for key, value in rows.items()}, batch_size=[2])
    padded = to_legacy_padded_batch(nested)
    control = left_right_2_no_padding(padded)
    assert [row.tolist() for row in control["input_ids"].unbind()] == [
        row.tolist() for row in rows["input_ids"]]
    # A mock prediction at position i carries i as its value. Response logits
    # must start at the last prompt token, independently for each sequence.
    logits = torch.nested.as_nested_tensor([torch.arange(6).float()] * 2, layout=torch.jagged)
    actual = no_padding_2_padding(logits, control)
    assert actual.tolist() == [[1, 2, 3, 4], [3, 4, 0, 0]]


def test_position_fallback_and_invalid_sequence():
    data = _batch()
    del data["position_ids"]
    result = to_legacy_padded_batch(data)
    assert result["position_ids"][0, 2:].tolist() == list(range(6))
    data["input_ids"][0] = torch.arange(6)
    with pytest.raises(ValueError, match="concatenation"):
        to_legacy_padded_batch(data)
