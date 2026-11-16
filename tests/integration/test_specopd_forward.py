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
    from verl_speco.backends.dflash_trainer_backend import DFlashTrainingModel
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel
    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainingModel
    from verl_speco.models.dflash import DFlashConfig, DFlashDraftModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.models.dflash2 import DFlash2Config, DFlash2DraftModel

    return {
        "dflash": (DFlashConfig, DFlashDraftModel, DFlashTrainingModel),
        "dspark": (DSparkConfig, DSparkDraftModel, DSparkTrainingModel),
        "dflash2": (DFlash2Config, DFlash2DraftModel, DFlash2TrainingModel),
    }


@pytest.mark.parametrize("algorithm", ["dflash", "dspark", "dflash2"])
def test_joint_actor_to_real_drafter_backward_without_resampling(
    classes, algorithm, monkeypatch
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
    config_class, draft_class, wrapper_class = classes[algorithm]
    block_size = 3 if algorithm == "dspark" else 4
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
        drafter_opd_config={
            "enable": True,
            "method": "spec_opd",
            "capture_verify_trace": True,
        },
    )
    assert wrapper.opd_objective is None
    monkeypatch.setattr(
        wrapper,
        "_sample_anchor_positions",
        lambda *a, **k: pytest.fail("anchors were resampled"),
    )
    batch = build_specopd_training_batch(
        request, scored[0], next(wrapper.parameters()).device, SpecOPDConfig(proposal_width=3), (1, 2)
    )
    fixed_inputs = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
    backend = SimpleNamespace(target_lm_head=SimpleNamespace(fc=target.lm_head))
    # Teacher projection weights are inputs to draft logits, but the target is
    # frozen; backward must update only the actual draft network.
    target.requires_grad_(False)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)
    before = draft.fc.weight.detach().clone()
    loss = compute_specopd_backend_loss(backend, wrapper, batch)
    changed = dict(batch)
    changed["input_ids"] = batch["input_ids"].clone()
    changed["input_ids"][0, -1] = 22
    changed["hidden_states"] = batch["hidden_states"].clone()
    changed["hidden_states"][:, -1] = torch.randn_like(changed["hidden_states"][:, -1])
    unchanged = compute_specopd_backend_loss(backend, wrapper, changed)
    torch.testing.assert_close(
        loss["total_local_ploss"], unchanged["total_local_ploss"]
    )
    assert loss["local_num_tokens"] == 6
    assert metrics["model_forward_calls"] == 1
    (loss["total_local_ploss"] / loss["local_num_tokens"]).backward()
    assert (
        draft.fc.weight.grad is not None and torch.isfinite(draft.fc.weight.grad).all()
    )
    assert draft.fc.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    if algorithm == "dspark":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in draft.markov_head.parameters()
        )
    if algorithm == "dflash2":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in draft.candidate_selector.parameters()
        )
    optimizer.step()
    assert not torch.equal(before, draft.fc.weight)

    # A retained input batch must remain immutable and agree with fresh input
    # reconstruction after the model weights have changed.
    for key, value in fixed_inputs.items():
        torch.testing.assert_close(batch[key], value)
    fresh = build_specopd_training_batch(
        request, scored[0], next(wrapper.parameters()).device, SpecOPDConfig(proposal_width=3), (1, 2)
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


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("loss_type", ["ce", "e2e_tv"])
@pytest.mark.parametrize("empty", [False, True])
def test_preserved_draftopd_baseline_real_backward(
    classes, algorithm, loss_type, empty, monkeypatch
):
    from verl_speco.backends import specopd_forward

    monkeypatch.setattr(
        specopd_forward,
        "proposal_loss",
        lambda *a, **k: pytest.fail("SpecOPD entered baseline"),
    )
    config_class, draft_class, wrapper_class = classes[algorithm]
    cfg = config_class(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        markov_rank=4,
    )
    wrapper = wrapper_class(
        draft_class(cfg),
        block_size=3,
        num_anchors=2,
        per_position_loss_weight="fixed_exp_decay",
        drafter_opd_config={
            "enable": True,
            "method": "draft_opd",
            "anchor_planner": "trace",
            "loss": loss_type,
        },
    )
    assert wrapper.opd_objective is not None
    trace = torch.full((1, 5, 3), -1)
    if not empty:
        trace[0, 1] = torch.tensor([0, 1, 1])
        trace[0, 3] = torch.tensor([2, 1, 1])
    head = torch.randn(32, 8)
    loss, *_rest, diagnostics = wrapper(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        hidden_states_list=[torch.randn(1, 5, 8), torch.randn(1, 5, 8)],
        loss_mask=torch.ones(1, 5),
        lm_head_weight=head,
        target_last_hidden_states=torch.randn(1, 5, 8),
        opd_trace=trace,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["opd/visible_tokens"] == (0 if empty else 2)
    assert wrapper.draft_model.fc.weight.grad is not None
    assert torch.isfinite(wrapper.draft_model.fc.weight.grad).all()
    if not empty:
        assert wrapper.draft_model.fc.weight.grad.abs().sum() > 0
