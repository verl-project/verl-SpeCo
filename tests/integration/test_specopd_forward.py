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

import asyncio
import importlib.util
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers import Qwen3Config, Qwen3ForCausalLM

from verl_speco.backends.specopd_forward import (
    build_specopd_training_batch,
    compute_specopd_backend_loss,
)
from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch
from verl_speco.specopd_config import SpecOPDConfig


@pytest.fixture
def dspark_model(monkeypatch):
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

    def create_model(**overrides):
        config = DSparkConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            num_context_layers=2, target_hidden_size=16, target_num_hidden_layers=2,
            mask_token_id=31, max_position_embeddings=128, block_size=3,
            conv_group_size=8, markov_rank=8, **overrides,
        )
        return DSparkTrainingModel(DSparkDraftModel(config), block_size=3, num_anchors=2)

    return create_model


def _actor():
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize(
    "objective,head_type", [("e2e_tv", "vanilla"), ("tv", "gated"), ("l1", "vanilla")]
)
def test_joint_actor_to_real_drafter_backward_without_resampling(
    dspark_model, objective, head_type, monkeypatch
):
    torch.manual_seed(21)
    target = _actor()
    request = JointScoringRequest(
        "a",
        (1, 2, 3, 4, 5, 6),
        3,
        (
            DraftBranch("b1", 2, (11, 12, 13), accepted_count=1, verified_count=1),
            DraftBranch("b2", 4, (14, 15, 16), accepted_count=1, verified_count=2),
        ),
    )
    scored, metrics = score_joint_batch(
        target, [request], hidden_state_indices=(1, 2), return_target_logits=False
    )
    wrapper = dspark_model(selector_rank=8, selector_top_k=32, markov_head_type=head_type)
    draft = wrapper.draft_model
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
    from verl_speco.integration.specopd_training_transport import encode_training_payload, decode_training_payload

    payload, _ = encode_training_payload([batch], spec)
    [transported] = decode_training_payload(payload)
    assert transported.keys() == batch.keys()
    for key, value in batch.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(transported[key], value, atol=0, rtol=0)
        else:
            assert transported[key] == value
    batch = {**transported, **spec.batch_options}
    mixed_spec = SpecOPDConfig(proposal_width=3, objective=objective, anchor_count=32)
    weighted = build_specopd_training_batch(
        request, scored[0], next(wrapper.parameters()).device, mixed_spec, (1, 2)
    )
    torch.testing.assert_close(
        weighted["position_weights"], torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    )
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
    assert batch["input_ids"].shape[1] == 5
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



@pytest.mark.parametrize("objective", ["e2e_tv", "tv", "l1"])
def test_mixed_dspark_joint_forward_backward_and_dummy_batch(dspark_model, objective):
    from verl_speco.backends.specopd_forward import merge_specopd_training_batches

    torch.manual_seed(29)
    target = _actor().requires_grad_(False)
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
    batch = {**build_specopd_training_batch(request, scores[0], "cpu", config, (1, 2)), **config.batch_options}
    torch.testing.assert_close(batch["position_weights"][0], torch.tensor([1., 2., 1.]))
    torch.testing.assert_close(batch["position_weights"][24], torch.exp(-torch.arange(3).float()/7))
    model = dspark_model()
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
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.draft_model.markov_head.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_uncached_device_inputs_are_released_before_next_transfer_and_snapshot(monkeypatch):
    import ast
    import time
    from pathlib import Path

    import torch

    from verl_speco.integration import specopd_collectives

    # Exercise the production coroutine without importing the Ray deployment
    # stack, so this allocation-lifetime regression also runs on CPU hosts.
    source = Path(__file__).resolve().parents[2] / "verl_speco/trainer/base_trainer.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(module)
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "training_specopd_group")
    namespace = {"torch": torch, "time": time}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    monkeypatch.setattr(specopd_collectives, "warm_specopd_training_collectives", lambda *a, **kw: None)

    # CPU .to() normally aliases its input. Clone to emulate a distinct device
    # allocation, then ensure one microbatch never overlaps the next staging.
    live_inputs = []
    original_to = torch.Tensor.to
    def transfer(tensor, *args, **kwargs):
        assert all(reference() is None for reference in live_inputs)
        return original_to(tensor, *args, **kwargs).clone()
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    inputs = {
        "input_ids": torch.ones(1, 2), "proposal_mask": torch.ones(1, 2, dtype=torch.bool), "_speco_position_count": 2,
    }


    async def activate():
        return True
    def snapshot(step):
        assert all(reference() is None for reference in live_inputs)
    trainer = SimpleNamespace(
        runtime_device="cpu", optimizer_steps_total=0, activate_training_model=activate,
        _get_dp_world_size=lambda: 1, _get_dp_local_rank=lambda: 0,
        model=SimpleNamespace(draft_model=SimpleNamespace(num_context_layers=1, config=SimpleNamespace(attention_dropout=0))),
        prepare_model_state_dict_for_publish=snapshot,
    )
    async def train(batch, step, accumulation_steps, accumulation_index):
        live_inputs.append(weakref.ref(batch["input_ids"]))
        if accumulation_index == accumulation_steps - 1:
            trainer.optimizer_steps_total += 1
        return True
    trainer._training_step_on_batch = train
    from verl_speco.integration.specopd_training_transport import PreparedTrainingGroup

    spec = SpecOPDConfig(optimizer_updates_per_collection=20, training_input_cache_mib=0)
    group = PreparedTrainingGroup((5, 0, (1,)), (inputs, inputs), (1, 1), spec, {})
    assert asyncio.run(namespace["training_specopd_group"](trainer, group))
    assert trainer._specopd_group_metrics["specopd/input_cache_bytes"] == 0
    assert len(live_inputs) == 40 and trainer.optimizer_steps_total == 20
