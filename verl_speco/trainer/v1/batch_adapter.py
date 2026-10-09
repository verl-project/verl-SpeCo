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

"""Convert V1 variable-length records to the legacy prompt/response layout."""

import torch
from tensordict import TensorDict


def to_legacy_padded_batch(data, pad_token_id=0):
    """Left-pad prompts and right-pad responses, preserving sequence alignment.

    Padding each field independently loses the shared prompt boundary required
    by old-logprob slicing and hidden-state collection. Keep the source nested
    tensors intact because they are also used by TransferQueue consumers.
    """
    prompts = list(data["prompts"].unbind())
    responses = list(data["responses"].unbind())
    inputs = list(data["input_ids"].unbind())
    prompt_width = max(row.numel() for row in prompts)
    response_width = max(row.numel() for row in responses)
    width = prompt_width + response_width
    count = len(prompts)
    padded = {}
    for key, rows, field_width, fill in (
        ("prompts", prompts, prompt_width, pad_token_id),
        ("responses", responses, response_width, pad_token_id),
        ("input_ids", inputs, width, pad_token_id),
    ):
        padded[key] = rows[0].new_full((count, field_width), fill)
    padded["attention_mask"] = inputs[0].new_zeros((count, width))
    for i, (prompt, response, tokens) in enumerate(
        zip(prompts, responses, inputs, strict=True)
    ):
        length = prompt.numel() + response.numel()
        if tokens.numel() != length or not torch.equal(
            tokens, torch.cat((prompt, response))
        ):
            raise ValueError(
                "V1 input_ids must equal the concatenation of prompts and responses"
            )
        start = prompt_width - prompt.numel()
        padded["prompts"][i, start:] = prompt
        padded["responses"][i, : response.numel()] = response
        padded["input_ids"][i, start : start + length] = tokens
        padded["attention_mask"][i, start : start + length] = 1

    for key in ("response_mask", "rollout_log_probs", "position_ids", "routed_experts"):
        if key not in data.keys():
            continue
        rows = list(data[key].unbind())
        # Multimodal position ids have shape [axes, sequence]; router records
        # and response fields use their first dimension as the token axis.
        axis = rows[0].ndim - 1 if key == "position_ids" else 0
        shape = list(rows[0].shape)
        shape[axis] = (
            response_width if key in ("response_mask", "rollout_log_probs") else width
        )
        output = rows[0].new_zeros((count, *shape))
        for i, row in enumerate(rows):
            response_field = key in ("response_mask", "rollout_log_probs")
            expected = responses[i].numel() if response_field else inputs[i].numel()
            if row.shape[axis] != expected:
                raise ValueError(f"V1 {key} length does not match its token sequence")
            start = 0 if response_field else prompt_width - prompts[i].numel()
            output[i].narrow(axis, start, expected).copy_(row)
        padded[key] = output
    if "position_ids" not in padded:
        padded["position_ids"] = (padded["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    return TensorDict(padded, batch_size=[count])
