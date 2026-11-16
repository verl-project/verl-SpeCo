import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.specopd_config import SpecOPDConfig


@pytest.mark.parametrize("updates,dp_rank,dp_size", [(1, 0, 1), (20, 0, 1), (20, 0, 2), (20, 1, 2), (20, 2, 3)])
def test_collected_group_executes_all_optimizer_updates_before_publication(monkeypatch, updates, dp_rank, dp_size):
    pytest.importorskip("ray")
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.backends import specopd_forward

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
        return {"proposal_mask": torch.ones(1, 2, dtype=torch.bool)}
    monkeypatch.setattr(specopd_forward, "build_specopd_training_batch", build)
    request = SimpleNamespace(branches=[SimpleNamespace(token_ids=(1, 2))])
    result = asyncio.run(DrafterBaseTrainer.training_specopd_group(
        trainer, [(request, None), (request, None)], 5,
        SpecOPDConfig(optimizer_updates_per_collection=updates), (1,),
    ))
    local_count = (2 + dp_size - 1) // dp_size
    assert len(builds) == local_count  # fixed local inputs built once
    assert result and trainer.optimizer_steps_total == updates
    assert events.count(("snapshot", 5)) == 1
    assert [e[1] for e in events if e[0] == "microbatch"] == list(range(local_count)) * updates
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
