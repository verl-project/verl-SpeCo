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
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from verl_speco.joint_scoring import (
    DraftBranch,
    JointScoringRequest,
    pack_joint_batch,
    score_joint_batch,
)


def make_model(attention="eager", family="qwen3"):
    torch.manual_seed(17)
    config_class, model_class = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[family]
    config = config_class(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.2,
        use_cache=False,
        sliding_window=None,
    )
    config._attn_implementation = attention
    return model_class(config)


def requests():
    return (
        JointScoringRequest(
            "a",
            (1, 2, 3, 4, 5, 6),
            3,
            (
                DraftBranch("early", 2, (20, 21, 22, 23)),
                DraftBranch("late", 4, (30, 31, 32)),
                DraftBranch("single", 5, (25,)),
            ),
        ),
        JointScoringRequest("b", (7, 8, 9), 2, (DraftBranch("short", 1, (11, 12)),)),
        JointScoringRequest("empty-response", (1, 2), 2),
    )


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_real_joint_matches_independent_forwards(attention, family):
    model = make_model(attention, family)
    model.train()
    # Preserve a deliberately mixed state across scoring.
    model.model.layers[0].eval()
    states = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    scored, metrics = score_joint_batch(
        model,
        requests(),
        actor_temperature=0.7,
        target_temperature=1.2,
        hidden_state_indices=(1,),
        calculate_entropy=True,
    )
    hook.remove()
    assert len(calls) == metrics["model_forward_calls"] == 1
    assert metrics["logical_rows"] < metrics["independent_rows"]
    assert [module.training for module in model.modules()] == states
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        for request, result in zip(requests(), scored):
            trunk = model(torch.tensor([request.trunk_ids]), output_hidden_states=True)
            rows = (
                trunk.logits[
                    0, request.response_start - 1 : len(request.trunk_ids) - 1
                ].float()
                / 0.7
            )
            labels = torch.tensor(request.trunk_ids[request.response_start :])
            expected = rows.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
            torch.testing.assert_close(
                result.response_log_probs, expected, atol=2e-6, rtol=2e-5
            )
            log_distribution = rows.log_softmax(-1)
            torch.testing.assert_close(
                result.response_entropy,
                -(log_distribution.exp() * log_distribution).sum(-1),
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                result.trunk_features[1],
                trunk.hidden_states[1][0],
                atol=2e-6,
                rtol=2e-5,
            )
            for branch, actual in zip(request.branches, result.branches):
                ids = request.trunk_ids[: branch.anchor + 1] + branch.token_ids[:-1]
                reference = model(torch.tensor([ids]), output_hidden_states=True)
                start = branch.anchor
                expected_logits = reference.logits[0, start:]
                torch.testing.assert_close(
                    actual.target_logits, expected_logits, atol=2e-6, rtol=2e-5
                )
                torch.testing.assert_close(
                    actual.target_hidden,
                    reference.hidden_states[-1][0, start:],
                    atol=2e-6,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    actual.target_logz, (expected_logits.float() / 1.2).logsumexp(-1)
                )
                assert not actual.target_logits.requires_grad


def test_future_trunk_and_sibling_never_leak():
    model = make_model()
    original = requests()[0]
    changed = JointScoringRequest(
        "a",
        (1, 2, 3, 41, 42, 43),
        3,
        (
            original.branches[0],
            DraftBranch("other", 4, (35, 36, 37)),
        ),
    )
    before, _ = score_joint_batch(model, [original])
    after, _ = score_joint_batch(model, [changed])
    torch.testing.assert_close(
        before[0].branches[0].target_logits, after[0].branches[0].target_logits
    )


def test_mask_positions_and_predecessor_rows():
    batch = pack_joint_batch(
        requests(), device="cpu", dtype=torch.float32, vocab_size=48
    )
    assert batch.branch_rows[0][0] == (2, 6, 7, 8)
    assert batch.position_ids[0, 6:9].tolist() == [3, 4, 5]
    visible = batch.attention_mask[0, 0] == 0
    assert visible[6].nonzero().flatten().tolist() == [0, 1, 2, 6]
    assert visible[7].nonzero().flatten().tolist() == [0, 1, 2, 6, 7]
    assert visible[4].nonzero().flatten().tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "branch",
    [DraftBranch("x", -1, (1,)), DraftBranch("x", 2, ()), DraftBranch("x", 2, (999,))],
)
def test_invalid_branch_rejected(branch):
    with pytest.raises(ValueError):
        pack_joint_batch(
            [JointScoringRequest("a", (1, 2, 3), 2, (branch,))],
            device="cpu",
            dtype=torch.float32,
            vocab_size=48,
        )


def test_unsupported_backend_and_budget_fail_before_forward():
    model = make_model()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match="max_packed_tokens"):
        score_joint_batch(model, requests(), max_packed_tokens=1)
    with pytest.raises(ValueError, match="max_mask_elements"):
        score_joint_batch(model, requests(), max_mask_elements=1)
    model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="4D mask"):
        score_joint_batch(model, requests())
    hook.remove()
    assert calls == []


def test_training_state_restored_on_forward_failure(monkeypatch):
    model = make_model()
    states = [module.training for module in model.modules()]

    def fail(**kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        score_joint_batch(model, requests())
    assert [module.training for module in model.modules()] == states
