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
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from verl_speco.joint_actor import score_actor_joint
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


def test_response_scoring_matches_fp32_over_multiple_chunks():
    from verl_speco.joint_scoring import response_scores

    logits = torch.randn(257, 17, dtype=torch.bfloat16)
    labels = torch.randint(logits.shape[-1], (len(logits),))
    logp, entropy = response_scores(logits, labels, 0.7, True)
    expected = (logits.float() / 0.7).log_softmax(-1)
    torch.testing.assert_close(logp, expected.gather(-1, labels[:, None]).squeeze(-1))
    torch.testing.assert_close(entropy, -(expected.exp() * expected).sum(-1))


def test_prefix_feature_export_keeps_complete_normal_response_scores():
    model = make_model()
    request = JointScoringRequest("crop", tuple((i % 40) + 1 for i in range(48)), 3,
                                  (DraftBranch("b", 5, (20, 21)),))
    full, _ = score_joint_batch(model, (request,), hidden_state_indices=(1, 2))
    cropped, _ = score_joint_batch(model, (request,), hidden_state_indices=(1, 2),
                                  trunk_feature_prefix_only=True)
    assert cropped[0].response_log_probs.numel() == 45
    torch.testing.assert_close(cropped[0].response_log_probs, full[0].response_log_probs)
    for layer in (1, 2):
        assert cropped[0].trunk_features[layer].shape[0] == 6
        torch.testing.assert_close(cropped[0].trunk_features[layer], full[0].trunk_features[layer][:6])


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


def worker():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
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


@pytest.mark.parametrize("return_target_logz", [True, False])
def test_engine_context_and_original_training_states_restored(return_target_logz):
    actor, events = worker()
    result, metrics = score_actor_joint(actor, request(), return_target_logz=return_target_logz)
    assert events == ["enter", "exit"]
    assert actor.actor.engine.module.training
    assert result[0].response_log_probs.shape == (2,)
    assert result[0].branches[0].target_logits.device.type == "cpu"
    assert (result[0].branches[0].target_logz is not None) == return_target_logz
    assert metrics["model_forward_calls"] == 1


@pytest.mark.parametrize("field", ["use_remove_padding", "use_ulysses_sp"])
def test_incompatible_engine_fails_before_context(field):
    actor, events = worker()
    setattr(actor.actor.engine, field, True)
    with pytest.raises(NotImplementedError):
        score_actor_joint(actor, request())
    assert not events


def test_multiple_ranks_are_not_silently_broadcast(monkeypatch):
    actor, events = worker()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    def mismatched(gathered, digest):
        gathered[:] = [digest, "another trajectory schedule"]
    monkeypatch.setattr(torch.distributed, "all_gather_object", mismatched)
    with pytest.raises(RuntimeError, match="same complete trajectory schedule"):
        score_actor_joint(actor, request())
    assert not events


def test_forward_error_exits_engine_and_restores_training(monkeypatch):
    actor, events = worker()

    def fail(**kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(actor.actor.engine.module, "forward", fail)
    with pytest.raises(RuntimeError, match="failed"):
        score_actor_joint(actor, request())
    assert events == ["enter", "exit"]
    assert actor.actor.engine.module.training


def test_actor_publish_mixin_exposes_joint_scoring():
    from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

    actor, events = worker()
    result, metrics = DraftWeightPublishMixin.compute_joint_old_log_prob(
        actor, request()
    )
    assert result[0].sample_id == "x"
    assert metrics["model_forward_calls"] == 1
    assert events == ["enter", "exit"]


def test_ulysses_rejects_adaptive_rotary_frequencies():
    from verl_speco.joint_ulysses import validate_ulysses_model

    actor, _ = worker()
    model = actor.actor.engine.module
    model.model.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="global-frequency"):
        validate_ulysses_model(model, 2)


def test_microbatch_schedule_uses_one_engine_residency_context():
    actor, events = worker()
    requests = request() + request()
    expected = [score_actor_joint(actor, [r])[0][0] for r in requests]
    events.clear()
    actual, measured = score_actor_joint(actor, requests, microbatch_sizes=(1, 1))
    assert events == ["enter", "exit"]
    assert actor.actor.engine.module.training
    assert measured["model_forward_calls"] == 2
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.response_log_probs, b.response_log_probs)
        torch.testing.assert_close(a.branches[0].target_logits, b.branches[0].target_logits)


def test_microbatch_schedule_error_restores_engine(monkeypatch):
    actor, events = worker()
    original = actor.actor.engine.module.forward
    calls = 0
    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chunk failed")
        return original(**kwargs)
    monkeypatch.setattr(actor.actor.engine.module, "forward", fail_second)
    with pytest.raises(RuntimeError, match="second chunk failed"):
        score_actor_joint(actor, request() + request(), microbatch_sizes=(1, 1))
    assert events == ["enter", "exit"]
    assert actor.actor.engine.module.training


@pytest.mark.parametrize("sizes", [(), (0, 2), (1,), (True, 1)])
def test_invalid_microbatch_schedule_does_not_enter_engine(sizes):
    actor, events = worker()
    with pytest.raises(ValueError, match="partition"):
        score_actor_joint(actor, request() + request(), microbatch_sizes=sizes)
    assert not events


def test_parallel_schedule_uses_all_dp_groups_for_long_trees():
    from dataclasses import replace

    from verl_speco.joint_actor import parallel_microbatch_sizes
    tree = JointScoringRequest("x", tuple(range(8192)), 128)
    trees = tuple(replace(tree, sample_id=str(i)) for i in range(320))
    sizes = parallel_microbatch_sizes(trees, (1,) * 320, 4, 4,
                                     {"max_packed_tokens": 9216, "max_mask_elements": 9216 ** 2})
    assert sizes == (4,) * 80


def test_parallel_schedule_respects_padded_local_mask_budget():
    from dataclasses import replace

    from verl_speco.joint_actor import parallel_microbatch_sizes
    tree = JointScoringRequest("x", tuple(range(9)), 2)
    trees = tuple(replace(tree, sample_id=str(i)) for i in range(5))
    sizes = parallel_microbatch_sizes(trees, (2, 2, 1), 2, 4,
                                     {"max_packed_tokens": 24, "max_mask_elements": 144})
    assert sizes == (2, 2, 1)


@pytest.mark.parametrize(
    "fast_path,tree_count,expected_branched,expected_ids",
    [
        (False, 8, [True, True, True, True], ["0", "2", "4", "6"]),
        (True, 8, [False, False, True, True], ["0", "1", "4", "5"]),
        (True, 6, [False, False, True], ["1", "2", "5"]),
    ],
)
def test_causal_schedule_groups_trunks_and_preserves_result_order(
    monkeypatch, fast_path, tree_count, expected_branched, expected_ids,
):
    from dataclasses import replace

    from verl_speco import joint_actor

    branched = request()[0]
    trees = tuple(
        replace(branched, sample_id=str(i), branches=branched.branches if i % 2 == 0 else ())
        for i in range(tree_count)
    )
    actor, _ = worker()
    engine = actor.actor.engine
    engine.ulysses_sequence_parallel_size = 2
    engine.ulysses_parallel_group = object()
    engine.use_remove_padding = True
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4 if group is None else 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(
        torch.distributed, "all_gather_object",
        lambda gathered, value: gathered.__setitem__(slice(None), [value] * 4),
    )
    monkeypatch.setenv("VERL_SPECO_JOINT_CAUSAL_FAST_PATH", str(int(fast_path)))
    batches = []

    def score(_, requests, sp_size, group, dummy, nonleader, *, use_causal_fast_path, **options):
        assert use_causal_fast_path == fast_path
        assert len(requests) == 1 and not dummy
        batches.append(bool(requests[0].branches))
        return requests, {}

    monkeypatch.setattr(joint_actor, "_score_local_joint_batch", score)
    actual, _ = score_actor_joint(actor, trees, microbatch_sizes=(1,) * tree_count)
    assert batches == expected_branched
    assert [tree.sample_id for tree in actual] == expected_ids


@pytest.mark.parametrize("bad_late_batch", [False, True])
def test_schedule_checks_all_ranks_once_before_any_forward(monkeypatch, bad_late_batch):
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
    if bad_late_batch:
        with pytest.raises(RuntimeError, match="preflight failed.*model vocabulary"):
            score_actor_joint(actor, trees, microbatch_sizes=(1, 1, 1))
        assert not events
    else:
        _, metrics = score_actor_joint(actor, trees, microbatch_sizes=(1, 1, 1))
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
