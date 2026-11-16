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
from verl_speco.joint_scoring import DraftBranch, JointScoringRequest


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


def test_engine_context_and_original_training_states_restored():
    actor, events = worker()
    result, metrics = score_actor_joint(actor, request())
    assert events == ["enter", "exit"]
    assert actor.actor.engine.module.training
    assert result[0].response_log_probs.shape == (2,)
    assert result[0].branches[0].target_logits.device.type == "cpu"
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
