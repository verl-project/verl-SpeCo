import asyncio
from types import SimpleNamespace
import weakref

import pytest

from verl_speco.specopd_config import SpecOPDConfig


@pytest.mark.parametrize("microbatch_size", [1, 4])
@pytest.mark.parametrize("updates,dp_rank,dp_size", [(1, 0, 1), (20, 0, 1), (20, 0, 2), (20, 1, 2), (20, 2, 3)])
def test_collected_group_executes_all_optimizer_updates_before_publication(monkeypatch, updates, dp_rank, dp_size, microbatch_size):
    pytest.importorskip("ray")
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.backends import specopd_forward
    from verl_speco.integration import specopd_collectives

    # This test isolates optimizer accounting; actual device admission is
    # covered by the16-rank Ray/HCCL probe.
    monkeypatch.setattr(specopd_collectives, "warm_specopd_training_collectives", lambda *a, **kw: None)

    import torch
    events = []
    async def activate():
        return True

    trainer = SimpleNamespace(
        runtime_device="cpu", optimizer_steps_total=0, activate_training_model=activate,
        _get_dp_world_size=lambda: dp_size, _get_dp_local_rank=lambda: dp_rank,
        model=SimpleNamespace(draft_model=SimpleNamespace(num_context_layers=1, config=SimpleNamespace(attention_dropout=0))),
        prepare_model_state_dict_for_publish=lambda step: events.append(("snapshot", step)),
    )
    async def train(batch, step, accumulation_steps, accumulation_index):
        assert bool(batch["proposal_mask"].any()) == (dp_rank < 2)
        events.append(("microbatch", accumulation_index))
        if accumulation_index == accumulation_steps - 1:
            trainer.optimizer_steps_total += 1
        return True

    trainer._training_step_on_batch = train
    builds = []
    def build(*args):
        builds.append(args[0])
        return dict(proposal_mask=torch.ones(1, 2, dtype=torch.bool),
                    input_ids=torch.ones(1, 3, dtype=torch.long), hidden_states=torch.ones(1, 3, 1),
                    anchor_positions=torch.ones(1, 1, dtype=torch.long), proposal_ids=torch.ones(1, 2, dtype=torch.long),
                    target_last_hidden_states=torch.ones(1, 2, 1), proposal_temperature=1.0, target_temperature=1.0,
                    proposal_block_width=7, _speco_sample_count=1, _speco_sample_ids=("b",), _specopd_trunk_digest="fixed")
    monkeypatch.setattr(specopd_forward, "build_specopd_training_batch", build)
    request = SimpleNamespace(trunk_ids=(1, 2, 3), branches=[SimpleNamespace(anchor=2, token_ids=(1, 2))])
    result = asyncio.run(DrafterBaseTrainer.training_specopd_group(
        trainer, [(request, None), (request, None)], 5,
        SpecOPDConfig(optimizer_updates_per_collection=updates, training_microbatch_trajectories=microbatch_size), (1,),
    ))
    local_count = (2 + dp_size - 1) // dp_size
    assert len(builds) == local_count  # fixed local inputs built once
    assert result and trainer.optimizer_steps_total == updates
    assert events.count(("snapshot", 5)) == 1
    assert [e[1] for e in events if e[0] == "microbatch"] == list(range((local_count + microbatch_size - 1) // microbatch_size)) * updates
    assert events[-1] == ("snapshot", 5)


@pytest.mark.parametrize("budget", [0, 16, 32])
def test_fixed_input_cache_respects_budget(monkeypatch, budget):
    import torch
    from verl_speco.backends import specopd_forward

    def build(request, *args):
        return {"input_ids": torch.tensor([request, request]), "temperature": 1.0}
    monkeypatch.setattr(specopd_forward, "build_specopd_training_batch", build)
    batches, used = specopd_forward.prepare_specopd_training_batches(
        [(1, None), (2, None)], "cpu", None, (), cache_bytes=budget,
    )
    assert used <= budget
    assert used == min(budget // 16, 2) * 16
    assert [batch["input_ids"].tolist() for batch in batches] == [[1, 1], [2, 2]]
    assert all(batch["temperature"] == 1.0 for batch in batches)


def test_uncached_device_inputs_are_released_before_next_transfer_and_snapshot(monkeypatch):
    import ast
    from pathlib import Path
    import time

    import torch
    from verl_speco.backends import specopd_forward
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
    monkeypatch.setattr(specopd_forward, "build_specopd_training_batch", lambda *args: {
        "input_ids": torch.ones(1, 2), "proposal_mask": torch.ones(1, 2, dtype=torch.bool),
    })

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
    request = SimpleNamespace(branches=[SimpleNamespace(anchor=1, token_ids=(1, 2))])
    assert asyncio.run(namespace["training_specopd_group"](
        trainer, [(request, None), (request, None)], 5,
        SpecOPDConfig(optimizer_updates_per_collection=20, training_microbatch_trajectories=1,
                      training_input_cache_mib=0), (1,),
    ))
    assert trainer._specopd_group_metrics["specopd/input_cache_bytes"] == 0
    assert len(live_inputs) == 40 and trainer.optimizer_steps_total == 20


@pytest.mark.parametrize("maximum,expected", [(1, [1, 1, 1, 1]), (2, [2, 2]), (4, [4])])
def test_global_training_groups_have_identical_dp_boundaries(maximum, expected):
    from verl_speco.backends.specopd_forward import plan_specopd_training_groups
    selected = [(SimpleNamespace(trunk_ids=tuple(range(n)), branches=[SimpleNamespace(anchor=n-1)]), None) for n in [7, 20, 9, 8, 15, 6, 5]]
    assert plan_specopd_training_groups(selected, 2, maximum, 1000) == expected
    assert plan_specopd_training_groups(selected, 2, 4, 30) == [1, 2, 1]
