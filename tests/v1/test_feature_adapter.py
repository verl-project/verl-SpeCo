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

from verl_speco.trainer.v1.feature_adapter import from_transfer_queue_batch


def test_feature_adapter_keeps_optional_fields_explicit():
    batch = {"input_ids": [1, 2], "attention_mask": [1, 1], "response_mask": [0, 1]}
    view = from_transfer_queue_batch(batch, global_step=7)

    assert view.input_ids == [1, 2]
    assert view.attention_mask == [1, 1]
    assert view.response_mask == [0, 1]
    assert view.hidden_states is None
    assert view.target_logprobs is None
    assert view.global_step == 7
