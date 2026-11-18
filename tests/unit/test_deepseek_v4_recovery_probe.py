# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Full persistent-state comparison, beyond sparse value samples."""

import copy
from types import SimpleNamespace

import pytest
import torch

from verl_speco.probe_deepseek_v4_recovery import (
    ReplicaGradientAudit,
    differences,
    persistent_state,
    tensor_chunks,
)


def test_replica_audit_records_inputs_and_gradients_without_changing_training():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Module()
            self.actor.mlp = torch.nn.Module()
            self.actor.mlp.gate = torch.nn.Linear(3, 2, bias=False)

        def prepare_training_inputs(self, *, input_ids):
            return SimpleNamespace(ids=input_ids), {"positions": input_ids + 1}

    model = Model()
    original = copy.deepcopy(model)
    audit = ReplicaGradientAudit(model)
    rng = torch.get_rng_state().clone()
    for _ in range(2):
        model.prepare_training_inputs(input_ids=torch.arange(3))
        model.actor.mlp.gate(torch.ones(1, 3)).sum().backward()
        original.actor.mlp.gate(torch.ones(1, 3)).sum().backward()
    name = "actor.mlp.gate.weight"
    assert torch.equal(
        model.actor.mlp.gate.weight.grad, original.actor.mlp.gate.weight.grad
    )
    assert torch.equal(audit.gradients[name], model.actor.mlp.gate.weight.grad)
    assert audit.calls[name] == 2
    report = audit.report()
    assert len(report["inputs"]) == 2
    assert report["initial_parameters"] == report["updated_parameters"]
    assert torch.equal(rng, torch.get_rng_state())
    audit.close()
    assert "prepare_training_inputs" not in vars(model)
    model.actor.mlp.gate(torch.ones(1, 3)).sum().backward()
    assert audit.calls[name] == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_chunked_hash_covers_noncontiguous_values_without_mutating_rng_or_storage(
    dtype,
):
    values = torch.arange(192).reshape(8, 24).to(dtype).transpose(0, 1)
    before = torch.get_rng_state().clone()
    pointer = values.data_ptr()
    chunks = list(tensor_chunks(values, 32))
    assert all(x.numel() * x.element_size() <= 32 for x in chunks)
    assert torch.equal(torch.cat([x.reshape(-1) for x in chunks]), values.reshape(-1))
    assert values.data_ptr() == pointer and torch.equal(torch.get_rng_state(), before)
    a = persistent_state(values, chunk_bytes=36)
    b = persistent_state(values.contiguous(), chunk_bytes=36)
    assert differences(a, b) == []


def test_full_state_comparison_detects_middle_value_and_metadata_changes():
    state = {"model": torch.zeros(10000), "scheduler": {"last_epoch": 1}}
    original = persistent_state(state, chunk_bytes=64)
    state["model"][4579] = 1
    changed = differences(original, persistent_state(state, chunk_bytes=64))
    assert any("/model/sha256" in p for p in changed)
    state["model"][4579] = 0
    state["scheduler"]["last_epoch"] = 2
    assert differences(original, persistent_state(state, chunk_bytes=64)) == [
        "/scheduler/last_epoch"
    ]


def test_real_adam_and_scheduler_state_restore_continue_identically():
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    def step(m, o, s):
        o.zero_grad()
        m(torch.ones(2, 4)).square().mean().backward()
        o.step()
        s.step()

    step(model, optimizer, scheduler)
    saved = copy.deepcopy(
        (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    )
    step(model, optimizer, scheduler)
    expected = persistent_state(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
    )
    restored = torch.nn.Linear(4, 3)
    other = torch.optim.AdamW(restored.parameters(), lr=0.01)
    other_scheduler = torch.optim.lr_scheduler.StepLR(other, 1, gamma=0.9)
    restored.load_state_dict(saved[0])
    other.load_state_dict(saved[1])
    other_scheduler.load_state_dict(saved[2])
    step(restored, other, other_scheduler)
    actual = persistent_state(
        {
            "model": restored.state_dict(),
            "optimizer": other.state_dict(),
            "scheduler": other_scheduler.state_dict(),
        }
    )
    assert differences(expected, actual) == []
    other.state[next(iter(restored.parameters()))]["exp_avg"].reshape(-1)[2] += 0.1
    assert differences(
        expected,
        persistent_state(
            {
                "model": restored.state_dict(),
                "optimizer": other.state_dict(),
                "scheduler": other_scheduler.state_dict(),
            }
        ),
    )


def test_state_preserves_scalar_object_schema_and_rejects_nonfinite():
    assert persistent_state(SimpleNamespace(scale=1.0))["state"] == {"scale": 1.0}
    with pytest.raises(ValueError, match="Non-finite"):
        persistent_state(torch.tensor([float("nan")]))


@pytest.mark.parametrize("with_scheduler", [False, True])
def test_engine_checksum_excludes_scratch_but_includes_router_cache(with_scheduler):
    from verl_speco.backends.vllm_ascend_engine import Progress
    from verl_speco.probe_deepseek_v4_recovery import engine_state

    model = torch.nn.Linear(2, 2)
    model.register_buffer("scratch", torch.zeros(5), persistent=False)
    model.weight_fp32 = model.weight.detach().float().clone()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    engine = SimpleNamespace(
        model_engine=SimpleNamespace(
            module=model, optimizer=optimizer, lr_scheduler=scheduler, global_steps=0
        ),
        teacher=torch.nn.Linear(2, 2),
        progress=Progress(),
    )
    store = SimpleNamespace(checkpoint_state=lambda: {"cursor": 0})
    if not with_scheduler:
        engine.model_engine.lr_scheduler = None
    before = engine_state(engine, store)
    model.scratch.fill_(3)
    assert differences(before, engine_state(engine, store)) == []
    model.weight_fp32.fill_(3)
    assert differences(before, engine_state(engine, store)) == [
        "/router_caches//sha256"
    ]
