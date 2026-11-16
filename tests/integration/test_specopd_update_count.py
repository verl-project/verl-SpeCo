import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.specopd_config import SpecOPDConfig


@pytest.mark.parametrize("updates", [1, 20])
def test_collected_group_executes_all_optimizer_updates_before_publication(monkeypatch, updates):
    pytest.importorskip("ray")
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.backends import specopd_forward

    events = []
    async def activate():
        return True

    trainer = SimpleNamespace(
        runtime_device="cpu", optimizer_steps_total=0, activate_training_model=activate,
        model=SimpleNamespace(draft_model=SimpleNamespace(num_context_layers=1, config=SimpleNamespace(attention_dropout=0))),
        prepare_model_state_dict_for_publish=lambda step: events.append(("snapshot", step)),
    )
    async def train(batch, step, accumulation_steps, accumulation_index):
        events.append(("microbatch", accumulation_index))
        if accumulation_index == accumulation_steps - 1:
            trainer.optimizer_steps_total += 1
        return True

    trainer._training_step_on_batch = train
    monkeypatch.setattr(specopd_forward, "build_specopd_training_batch", lambda *args: {})
    request = SimpleNamespace(branches=[SimpleNamespace(token_ids=(1, 2))])
    result = asyncio.run(DrafterBaseTrainer.training_specopd_group(
        trainer, [(request, None), (request, None)], 5,
        SpecOPDConfig(optimizer_updates_per_collection=updates), (1,),
    ))
    assert result and trainer.optimizer_steps_total == updates
    assert events.count(("snapshot", 5)) == 1
    assert [e[1] for e in events if e[0] == "microbatch"] == [0, 1] * updates
    assert events[-1] == ("snapshot", 5)
