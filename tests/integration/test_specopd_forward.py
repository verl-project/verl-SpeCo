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
"""Real CPU actor/drafter numerics; only the unavailable verl device utility is shimmed."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch
from verl_speco.specopd_config import SpecOPDConfig
from verl_speco.backends.specopd_forward import (
    build_specopd_training_batch,
    compute_specopd_backend_loss,
)


@pytest.fixture
def classes(monkeypatch):
    if importlib.util.find_spec("verl") is None:
        device = ModuleType("verl.utils.device")
        device.get_device_name = lambda: "cpu"
        device.get_device_id = lambda: 0
        for name in ("verl", "verl.utils"):
            module = ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(sys.modules, "verl.utils.device", device)
        functional = ModuleType("verl.utils.torch_functional")

        def unused_scheduler(*args, **kwargs):
            raise AssertionError(
                "scheduler shim must not participate in model numerics"
            )

        functional.get_constant_schedule_with_warmup = unused_scheduler
        functional.get_cosine_schedule_with_warmup = unused_scheduler
        monkeypatch.setitem(sys.modules, "verl.utils.torch_functional", functional)
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel

    return {
        "dspark": (DSparkConfig, DSparkDraftModel, DSparkTrainingModel),
    }


@pytest.mark.parametrize("objective", ["e2e_tv", "tv", "l1"])
def test_joint_actor_to_real_drafter_backward_without_resampling(
    classes, objective, monkeypatch
):
    torch.manual_seed(21)
    target_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
    )
    target_config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(target_config).eval()
    request = JointScoringRequest(
        "a",
        (1, 2, 3, 4, 5, 6),
        3,
        (DraftBranch("b1", 2, (11, 12, 13)), DraftBranch("b2", 4, (14, 15, 16))),
    )
    scored, metrics = score_joint_batch(
        target, [request], hidden_state_indices=(1, 2), return_target_logits=False
    )
    config_class, draft_class, wrapper_class = classes["dspark"]
    block_size = 3
    cfg = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_context_layers=2,
        target_hidden_size=16,
        target_num_hidden_layers=2,
        mask_token_id=31,
        max_position_embeddings=128,
        block_size=block_size,
        selector_rank=8,
        selector_top_k=32,
        conv_group_size=8,
        markov_rank=8,
    )
    draft = draft_class(cfg)
    wrapper = wrapper_class(
        draft_model=draft,
        block_size=block_size,
        num_anchors=2,
    )
    assert wrapper.opd_objective is None
    monkeypatch.setattr(
        wrapper,
        "_sample_anchor_positions",
        lambda *a, **k: pytest.fail("anchors were resampled"),
    )
    spec = SpecOPDConfig(proposal_width=3, objective=objective)
    batch = build_specopd_training_batch(
        request, scored[0], next(wrapper.parameters()).device, spec, (1, 2)
    )
    fixed_inputs = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
    dist_key, ce_key = "dspark_l1_loss_alpha", "dspark_ce_loss_alpha"
    training = {dist_key: 0.9, ce_key: 0.1}
    drafter = SimpleNamespace(speculative_algorithm="DSPARK", training=training)
    backend = SimpleNamespace(
        target_lm_head=SimpleNamespace(fc=target.lm_head),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=drafter)),
    )
    # Teacher projection weights are inputs to draft logits, but the target is
    # frozen; backward must update only the actual draft network.
    target.requires_grad_(False)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)
    before = draft.fc.weight.detach().clone()
    loss = compute_specopd_backend_loss(backend, wrapper, batch)
    if objective != "e2e_tv":
        training.update({dist_key: 1.0, ce_key: 0.0})
        distribution = compute_specopd_backend_loss(backend, wrapper, batch)
        training.update({dist_key: 0.0, ce_key: 1.0})
        ce = compute_specopd_backend_loss(backend, wrapper, batch)
        training.update({dist_key: 0.9, ce_key: 0.1})
        assert ce["total_local_ploss"] > 0
        torch.testing.assert_close(
            loss["total_local_ploss"],
            0.9 * distribution["total_local_ploss"] + 0.1 * ce["total_local_ploss"],
        )
    changed = dict(batch)
    # Restore a full response suffix to compare against the cropped context.
    assert batch["input_ids"].shape[1] == 5
    changed["input_ids"] = torch.tensor([request.trunk_ids])
    changed["input_ids"][0, -1] = 22
    changed["hidden_states"] = torch.cat([scored[0].trunk_features[i] for i in (1, 2)], -1).unsqueeze(0).clone()
    changed["hidden_states"][:, -1] = torch.randn_like(changed["hidden_states"][:, -1])
    unchanged = compute_specopd_backend_loss(backend, wrapper, changed)
    torch.testing.assert_close(
        loss["total_local_ploss"], unchanged["total_local_ploss"]
    )
    parameters = tuple(wrapper.parameters())
    cropped_grad = torch.autograd.grad(loss["total_local_ploss"], parameters, retain_graph=True, allow_unused=True)
    lean_scores, _ = score_joint_batch(
        target, [request], hidden_state_indices=(1, 2),
        return_target_logits=False, return_target_logz=False,
    )
    assert all(branch.target_logz is None for branch in lean_scores[0].branches)
    lean_batch = build_specopd_training_batch(
        request, lean_scores[0], next(wrapper.parameters()).device,
        spec, (1, 2),
    )
    lean_loss = compute_specopd_backend_loss(backend, wrapper, lean_batch)
    torch.testing.assert_close(lean_loss["total_local_ploss"], loss["total_local_ploss"], atol=0, rtol=0)
    lean_grad = torch.autograd.grad(lean_loss["total_local_ploss"], parameters, allow_unused=True)
    for actual, expected in zip(lean_grad, cropped_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    full_grad = torch.autograd.grad(unchanged["total_local_ploss"], parameters, retain_graph=True, allow_unused=True)
    for a, b in zip(cropped_grad, full_grad, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert loss["local_num_tokens"] == 6
    assert metrics["model_forward_calls"] == 1
    (loss["total_local_ploss"] / loss["local_num_tokens"]).backward()
    assert (
        draft.fc.weight.grad is not None and torch.isfinite(draft.fc.weight.grad).all()
    )
    assert draft.fc.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in draft.markov_head.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, draft.fc.weight)

    # A retained input batch must remain immutable and agree with fresh input
    # reconstruction after the model weights have changed.
    for key, value in fixed_inputs.items():
        torch.testing.assert_close(batch[key], value)
    fresh = build_specopd_training_batch(
        request, scored[0], next(wrapper.parameters()).device, spec, (1, 2)
    )
    reused_loss = compute_specopd_backend_loss(backend, wrapper, batch)
    fresh_loss = compute_specopd_backend_loss(backend, wrapper, fresh)
    torch.testing.assert_close(reused_loss["total_local_ploss"], fresh_loss["total_local_ploss"])
    params = tuple(p for p in wrapper.parameters() if p.requires_grad)
    reused_grads = torch.autograd.grad(reused_loss["total_local_ploss"], params, allow_unused=True)
    fresh_grads = torch.autograd.grad(fresh_loss["total_local_ploss"], params, allow_unused=True)
    for a, b in zip(reused_grads, fresh_grads, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b)
    from dataclasses import replace
    from verl_speco.backends.specopd_forward import merge_specopd_training_batches

    ragged = []
    for index in range(4):
        branches = tuple(replace(b, branch_id=f"{index}-{b.branch_id}",
                                 token_ids=b.token_ids[:2] if index == 1 else b.token_ids)
                         for b in request.branches[:1 if index == 1 else 2])
        req = replace(request, sample_id=f"sample-{index}", branches=branches,
                      trunk_ids=request.trunk_ids[:-1] if index == 2 else request.trunk_ids)
        scores, _ = score_joint_batch(target, [req], hidden_state_indices=(1, 2), return_target_logits=False)
        item = build_specopd_training_batch(req, scores[0], 'cpu', spec, (1, 2))
        if index == 3:
            item['proposal_mask'].zero_()  # DP dummy row
        ragged.append(item)
    merged = merge_specopd_training_batches(ragged)
    combined = compute_specopd_backend_loss(backend, wrapper, merged)
    separate = [compute_specopd_backend_loss(backend, wrapper, item) for item in ragged]
    expected_sum = sum(item['total_local_ploss'] for item in separate)
    expected_count = sum(item['local_num_tokens'] for item in separate)
    assert combined['local_num_tokens'] == expected_count
    torch.testing.assert_close(combined['total_local_ploss'], expected_sum, atol=2e-5, rtol=2e-5)
    merged_grads = torch.autograd.grad(combined['total_local_ploss'] / expected_count, params, allow_unused=True)
    individual_grads = torch.autograd.grad(expected_sum / expected_count, params, allow_unused=True)
    for a, b in zip(merged_grads, individual_grads, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    dummy = dict(batch, proposal_mask=torch.zeros_like(batch["proposal_mask"]))
    dummy_loss = compute_specopd_backend_loss(backend, wrapper, dummy)
    assert dummy_loss["local_num_tokens"] == 0
    assert dummy_loss["total_local_ploss"] == 0
    assert torch.isfinite(dummy_loss["accuracy"])


def test_accumulated_prefix_loss_matches_concatenated_gradient():
    from verl_speco.backends.specopd_loss import e2e_tv_loss

    torch.manual_seed(4)
    logits = torch.randn(3, 3, 8, requires_grad=True)
    target = torch.randn_like(logits)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    numerator, count, _ = e2e_tv_loss(logits, target, mask)
    expected = torch.autograd.grad(numerator / count, logits)[0]
    copied = logits.detach().clone().requires_grad_()
    for start, end in ((0, 1), (1, 3)):
        local, _, _ = e2e_tv_loss(copied[start:end], target[start:end], mask[start:end])
        (local / count).backward()
    torch.testing.assert_close(copied.grad, expected)


@pytest.mark.parametrize("objective", ["e2e_tv", "tv", "l1"])
def test_mixed_dspark_joint_forward_backward_and_dummy_batch(classes, objective):
    from verl_speco.backends.specopd_forward import merge_specopd_training_batches

    torch.manual_seed(29)
    target_config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128)
    target_config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(target_config).eval().requires_grad_(False)
    from verl_speco.integration.specopd_anchors import add_trunk_anchors
    config = SpecOPDConfig(anchor_count=32, max_branches_per_request=24,
                          max_proposal_tokens_per_request=72, proposal_width=3,
                          objective=objective)
    ids = tuple(i % 30 for i in range(43))
    proposals = [DraftBranch(f"proposal{i}", 2+i, (11, 12, 13), accepted_count=1) for i in range(24)]
    branches = add_trunk_anchors(proposals, ids, 3,
        dict(request_id="mixed", actor_revision=0, drafter_revision=0, window=[0, 40]), config)
    request = JointScoringRequest("mixed", ids, 3, tuple(branches))
    scores, measured = score_joint_batch(target, [request], hidden_state_indices=(1, 2), return_target_logits=False)
    batch = build_specopd_training_batch(request, scores[0], "cpu", config, (1, 2))
    torch.testing.assert_close(batch["position_weights"][0], torch.tensor([1., 2., 1.]))
    torch.testing.assert_close(batch["position_weights"][24], torch.exp(-torch.arange(3).float()/7))
    cfg_cls, draft_cls, wrapper_cls = classes["dspark"]
    draft = draft_cls(cfg_cls(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_context_layers=2, target_hidden_size=16, target_num_hidden_layers=2,
        mask_token_id=31, max_position_embeddings=128, block_size=3,
        conv_group_size=8, markov_rank=8))
    model = wrapper_cls(draft_model=draft, block_size=3, num_anchors=2)
    drafter = SimpleNamespace(speculative_algorithm="DSPARK", training={})
    backend = SimpleNamespace(
        target_lm_head=SimpleNamespace(fc=target.lm_head),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=drafter)),
    )
    loss = compute_specopd_backend_loss(backend, model, batch)
    assert measured["model_forward_calls"] == 1
    assert loss["local_num_tokens"] == 32  # Equal anchor, rather than token, weight.
    dummy = dict(batch, proposal_mask=torch.zeros_like(batch["proposal_mask"]))
    combined = compute_specopd_backend_loss(backend, model, merge_specopd_training_batches([batch, dummy]))
    torch.testing.assert_close(combined["total_local_ploss"], loss["total_local_ploss"])
    assert combined["local_num_tokens"] == 32
    # The optional ten-trajectory profile must preserve the same complete-group
    # objective and gradients as the default [4, 4, 2] accumulation schedule.
    ten = compute_specopd_backend_loss(backend, model, merge_specopd_training_batches([batch] * 10))
    parts = [compute_specopd_backend_loss(backend, model, merge_specopd_training_batches([batch] * n))
             for n in (4, 4, 2)]
    parts_sum = sum(part["total_local_ploss"] for part in parts)
    parts_count = sum(part["local_num_tokens"] for part in parts)
    assert ten["local_num_tokens"] == parts_count == 320
    torch.testing.assert_close(ten["total_local_ploss"], parts_sum)
    parameters = tuple(model.parameters())
    ten_grads = torch.autograd.grad(ten["total_local_ploss"] / parts_count, parameters, allow_unused=True)
    parts_grads = torch.autograd.grad(parts_sum / parts_count, parameters, allow_unused=True)
    for actual, expected in zip(ten_grads, parts_grads, strict=True):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    (loss["total_local_ploss"] / loss["local_num_tokens"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in draft.markov_head.parameters())
    assert all(p.grad is None for p in target.parameters())
