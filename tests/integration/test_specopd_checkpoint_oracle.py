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
"""Opt-in real-checkpoint NPU joint-vs-independent causal-forward oracle.

Set SPECO_CHECKPOINT_ORACLE_MODEL and reserve an ASCEND_RT_VISIBLE_DEVICES die.
This performs inference only on a deterministic synthetic token fixture.
"""

import json
import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
from transformers import AutoModelForCausalLM

from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch


@pytest.fixture(scope="module")
def checkpoint():
    path = os.getenv("SPECO_CHECKPOINT_ORACLE_MODEL")
    if not path or not torch.npu.is_available():
        pytest.skip("requires explicit checkpoint path and reserved visible NPU")
    if hasattr(torch.npu.matmul, "allow_hf32"):
        torch.npu.matmul.allow_hf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().to("npu:0")
    model.config.use_cache = False
    yield model
    del model
    torch.npu.empty_cache()


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpoint_joint_matches_independent_causal_forwards(checkpoint, dtype):
    model = checkpoint.to(dtype=getattr(torch, dtype))
    request = JointScoringRequest("oracle", tuple(range(1, 65)), 48, (
        DraftBranch("early", 47, (101, 102, 103, 104, 105, 106, 107)),
        DraftBranch("late", 55, (201, 202, 203, 204, 205, 206, 207)),
    ))
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    (scored,), metrics = score_joint_batch(
        model, [request], hidden_state_indices=(2, 10, 18, 26, 34),
        calculate_entropy=True,
    )
    hook.remove()
    assert len(calls) == metrics["model_forward_calls"] == 1
    errors = {}
    packed_length = metrics["logical_rows"]

    def independent(ids):
        # Equal physical shapes isolate tree/mapping errors from the NPU's
        # length-dependent BF16 GEMM choices. The oracle remains a standalone
        # ordinary causal sequence, with only right padding after its end.
        tensor = torch.tensor([ids], device="npu:0")
        mask = torch.ones_like(tensor)
        return model(F.pad(tensor, (0, packed_length - len(ids))),
                     attention_mask=F.pad(mask, (0, packed_length - len(ids))),
                     output_hidden_states=True)

    with torch.no_grad():
        trunk = independent(request.trunk_ids)
        rows = trunk.logits[0, request.response_start - 1:len(request.trunk_ids)-1].float()
        labels = torch.tensor(request.trunk_ids[request.response_start:], device="npu:0")
        expected = rows.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        errors["response_logprob_max_abs"] = (scored.response_log_probs - expected).abs().max().item()
        unpadded = model(torch.tensor([request.trunk_ids], device="npu:0"))
        raw = unpadded.logits[0, request.response_start-1:-1].float().log_softmax(-1)
        raw = raw.gather(-1, labels[:, None]).squeeze(-1)
        errors["ordinary_padding_logprob_max_abs"] = (expected - raw).abs().max().item()
        errors["aux_hidden_max_abs"] = max(
            (scored.trunk_features[i] - trunk.hidden_states[i][0, :len(request.trunk_ids)]).abs().max().item()
            for i in scored.trunk_features
        )
        tv, hidden, compact_tv = [], [], []
        branch_offset = len(request.trunk_ids)
        for branch, actual in zip(request.branches, scored.branches, strict=True):
            ids = request.trunk_ids[:branch.anchor + 1] + branch.token_ids[:-1]
            # An ordinary independent causal sequence with masked padding holes
            # preserves physical KV reduction order as well as logical positions.
            # No sibling or future-trunk token is present in this oracle input.
            tensor = torch.zeros((1, packed_length), dtype=torch.long, device="npu:0")
            mask = torch.zeros_like(tensor)
            positions = torch.zeros_like(tensor)
            prefix = branch.anchor + 1
            tensor[0, :prefix] = torch.tensor(request.trunk_ids[:prefix], device="npu:0")
            mask[0, :prefix] = 1
            positions[0, :prefix] = torch.arange(prefix, device="npu:0")
            tail = len(branch.token_ids) - 1
            tensor[0, branch_offset:branch_offset+tail] = torch.tensor(branch.token_ids[:-1], device="npu:0")
            mask[0, branch_offset:branch_offset+tail] = 1
            positions[0, branch_offset:branch_offset+tail] = torch.arange(prefix, prefix+tail, device="npu:0")
            reference = model(tensor, attention_mask=mask, position_ids=positions, output_hidden_states=True)
            readout = torch.tensor([branch.anchor, *range(branch_offset, branch_offset+tail)], device="npu:0")
            expected_logits = reference.logits[0].index_select(0, readout).float()
            expected_hidden = reference.hidden_states[-1][0].index_select(0, readout)
            tv.append((actual.target_logits.float().softmax(-1) - expected_logits.softmax(-1)).abs().sum(-1).max().item() / 2)
            compact = model(torch.tensor([ids], device="npu:0"))
            compact_logits = compact.logits[0, branch.anchor:].float()
            compact_tv.append((actual.target_logits.float().softmax(-1) - compact_logits.softmax(-1)).abs().sum(-1).max().item() / 2)
            hidden.append((actual.target_hidden - expected_hidden).abs().max().item())
            torch.testing.assert_close(actual.target_hidden, expected_hidden,
                                       atol=0.125 if dtype == "bfloat16" else 2e-4,
                                       rtol=0.02 if dtype == "bfloat16" else 1e-5)
            branch_offset += tail
        for i in scored.trunk_features:
            torch.testing.assert_close(scored.trunk_features[i], trunk.hidden_states[i][0, :len(request.trunk_ids)],
                                       atol=0.125 if dtype == "bfloat16" else 2e-4,
                                       rtol=0.02 if dtype == "bfloat16" else 1e-5)
        errors["branch_distribution_tv_max"] = max(tv)
        errors["compact_branch_distribution_tv_max"] = max(compact_tv)
        errors["branch_hidden_max_abs"] = max(hidden)
    print("CHECKPOINT_ORACLE", json.dumps({"dtype": dtype, **metrics, **errors}, sort_keys=True))
    lp_limit, tv_limit = (0.125, 0.03) if dtype == "bfloat16" else (1e-4, 5e-5)
    assert errors["response_logprob_max_abs"] <= lp_limit
    assert errors["branch_distribution_tv_max"] <= tv_limit
    assert all(parameter.grad is None for parameter in model.parameters())
