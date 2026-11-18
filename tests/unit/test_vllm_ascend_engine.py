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
"""Real CPU autograd tests for orchestration; not an Ascend model audit."""

import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.backends.vllm_ascend_engine import (
    RemainingBatches,
    VllmAscendDSparkEngine,
)
from verl_speco.trainer.vllm_ascend_checkpoint import rank_rng_state, restore_rng_state


def make_engine(*, train_function=None):
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    runtime = SimpleNamespace(global_steps=0, parameter=parameter, optimizer=optimizer)
    epochs = []
    sampler = SimpleNamespace(set_epoch=epochs.append)
    cfg = OmegaConf.create(
        {
            "train": {"epochs": 3, "max_optimizer_steps": None},
            "deepspeed": {"gradient_accumulation_steps": 2},
        }
    )

    def cpu_numerics(
        *,
        model_engine,
        target_lm_head,
        dataloader,
        sampler,
        cfg,
        logger,
        tracker,
        device,
        parallel_topology,
        checkpoint_dir,
        initial_global_step,
        initial_global_examples,
        save_checkpoints,
    ):
        # Emulate a native callable that owns accumulation, backward and step.
        assert target_lm_head == "frozen-teacher" and sampler is None
        assert not save_checkpoints and cfg.train.epochs == 1
        batches = list(dataloader)
        for offset in range(0, len(batches), 2):
            window = batches[offset : offset + 2]
            loss = sum(
                (model_engine.parameter - value).square() for value in window
            ) / len(window)
            loss.backward()
            model_engine.optimizer.step()
            model_engine.optimizer.zero_grad(set_to_none=True)
            model_engine.global_steps += 1
        return {
            "global_step": model_engine.global_steps,
            "global_examples": len(batches),
            "observed_update": True,
            "observed_nonzero_optimizer_lr": True,
        }

    engine = VllmAscendDSparkEngine(
        model_engine=runtime,
        teacher="frozen-teacher",
        cfg=cfg,
        topology=None,
        device="cpu",
        tracker=None,
        logger=logging.getLogger(__name__),
        train_function=train_function or cpu_numerics,
        checkpoint_dir="unused",
    )
    return engine, sampler, epochs


def test_cycle_preserves_accumulation_tail_and_epoch_cursor():
    engine, sampler, epochs = make_engine()
    loader = [torch.tensor(float(i)) for i in range(5)]
    reference = torch.tensor(2.0, requires_grad=True)
    optimizer = torch.optim.SGD([reference], lr=0.1)
    for window in (loader[:2], loader[2:4], loader[4:]):
        (sum((reference - x).square() for x in window) / len(window)).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    engine.train_cycle(loader, sampler, max_steps=1)
    assert engine.progress.batch_cursor == 2 and engine.progress.global_step == 1
    engine.train_cycle(loader, sampler, max_steps=10)
    assert engine.progress.epoch == 1 and engine.progress.batch_cursor == 0
    assert engine.progress.global_step == 3 and engine.progress.global_examples == 5
    assert torch.equal(engine.model_engine.parameter, reference)
    assert epochs == [0, 0]
    assert engine.cfg.train.epochs == 3 and engine.cfg.train.max_optimizer_steps is None


def test_partial_failure_invalidates_session():
    engine, sampler, _ = make_engine()
    engine.train_function = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("partial update")
    )
    with pytest.raises(RuntimeError, match="partial update"):
        engine.train_cycle([torch.tensor(1.0)] * 4, sampler, max_steps=1)
    with pytest.raises(RuntimeError, match="restore"):
        engine.train_cycle([torch.tensor(1.0)] * 4, sampler, max_steps=1)


def test_changed_native_api_is_rejected():
    with pytest.raises(RuntimeError, match="API has changed"):
        make_engine(train_function=lambda model: None)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (4, 2)])
def test_bad_cursor_is_rejected(start, count):
    with pytest.raises(ValueError):
        RemainingBatches(range(5), start, count)


def test_rng_roundtrip_preserves_random_streams():
    import random

    import numpy as np

    state = rank_rng_state()
    expected = random.random(), np.random.rand(), torch.rand(4)
    restore_rng_state(state)
    actual = random.random(), np.random.rand(), torch.rand(4)
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
