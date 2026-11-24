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
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl_speco.joint_actor import score_actor_joint
from verl_speco.joint_scoring import (
    DraftBranch,
    JointScoringRequest,
    score_joint_batch,
)


def make_model(attention="eager", *, vocab_size=48):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=vocab_size,
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
    return Qwen3ForCausalLM(config)


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


def test_response_scoring_matches_fp32_over_multiple_chunks():
    from verl_speco.joint_scoring import response_scores

    logits = torch.randn(257, 17, dtype=torch.bfloat16)
    labels = torch.randint(logits.shape[-1], (len(logits),))
    logp, entropy = response_scores(logits, labels, 0.7, True)
    expected = (logits.float() / 0.7).log_softmax(-1)
    torch.testing.assert_close(logp, expected.gather(-1, labels[:, None]).squeeze(-1))
    torch.testing.assert_close(entropy, -(expected.exp() * expected).sum(-1))


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_real_joint_matches_independent_forwards(attention):
    model = make_model(attention)
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


def worker():
    model = make_model(vocab_size=32)
    events = []

    @contextmanager
    def eval_mode():
        events.append("enter")
        model.eval()
        try:
            yield
        finally:
            events.append("exit")

    engine = SimpleNamespace(
        module=model,
        eval_mode=eval_mode,
        use_remove_padding=False,
        use_ulysses_sp=False,
        model_config=SimpleNamespace(use_fused_kernels=False),
        _autocast_dtype=torch.float32,
    )
    return SimpleNamespace(_is_actor=True, actor=SimpleNamespace(engine=engine)), events


def request():
    return [JointScoringRequest("x", (1, 2, 3, 4), 2, (DraftBranch("b", 2, (5, 6)),))]


def test_parallel_schedule_respects_padded_local_mask_budget():
    from dataclasses import replace

    from verl_speco.joint_actor import parallel_microbatch_sizes
    tree = JointScoringRequest("x", tuple(range(9)), 2)
    trees = tuple(replace(tree, sample_id=str(i)) for i in range(5))
    sizes = parallel_microbatch_sizes(trees, (2, 2, 1), 2, 4,
                                     {"max_packed_tokens": 24, "max_mask_elements": 144})
    assert sizes == (2, 2, 1)


@pytest.mark.parametrize("bad_late_batch", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
def test_schedule_checks_all_ranks_once_before_any_forward(monkeypatch, bad_late_batch, automatic):
    from dataclasses import replace

    actor, events = worker()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    collectives = []

    def gather(values, value):
        assert not events
        collectives.append(value)
        values[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    first = request()[0]
    final = replace(first, sample_id="final", trunk_ids=(1, 2, 3, 32)) if bad_late_batch else first
    trees = (first, first, final)
    schedule = dict(microbatch_trajectories=1) if automatic else dict(microbatch_sizes=(1, 1, 1))
    if bad_late_batch:
        with pytest.raises(RuntimeError, match="preflight failed.*model vocabulary"):
            score_actor_joint(actor, trees, **schedule)
        assert not events
    else:
        _, metrics = score_actor_joint(actor, trees, **schedule)
        assert metrics["model_forward_calls"] == 3
        assert events == ["enter", "exit"]
    assert len(collectives) == 2


def test_preflight_validation_allocates_no_quadratic_mask(monkeypatch):
    from verl_speco.joint_scoring import validate_joint_batch
    def fail(*args, **kwargs):
        raise AssertionError("preflight allocated a tensor")
    monkeypatch.setattr(torch, "zeros", fail)
    monkeypatch.setattr(torch, "eye", fail)
    monkeypatch.setattr(torch, "ones", fail)
    lengths, width, _ = validate_joint_batch(request(), dtype=torch.float32, vocab_size=32,
                                            sequence_multiple=4)
    assert lengths == [5] and width == 8
