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
"""First real accelerator offload blocks; subsequent transfers use verl."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer import base_trainer as base  # noqa: E402


class Tensor:
    def __init__(self, device="npu", *, fail=False):
        self.device = SimpleNamespace(type=device)
        self.fail = fail
        self.copies = []

    def to(self, device, *, non_blocking):
        self.copies.append(non_blocking)
        if self.fail:
            raise RuntimeError("copy failed")
        return Tensor(device)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(base, "device_name", "npu")
    monkeypatch.setattr(
        base.torch, "is_tensor", lambda value: isinstance(value, Tensor)
    )
    instance = base.DrafterBaseTrainer.__new__(base.DrafterBaseTrainer)
    instance._drafter_strategy = lambda: "fsdp2"
    instance.optimizer = SimpleNamespace(state={})
    return instance


@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_first_real_offload_then_original_and_new_optimizer(
    trainer, monkeypatch, device
):
    monkeypatch.setattr(base, "device_name", device)
    original_calls = []
    monkeypatch.setattr(base, "offload_fsdp_optimizer", original_calls.append)
    trainer._offload_optimizer_state_to_cpu()
    trainer.optimizer.state = {"p": {"step": Tensor("cpu")}, "metadata": None}
    trainer._offload_optimizer_state_to_cpu()
    assert not getattr(
        trainer.optimizer, "_speco_first_blocking_optimizer_offload_done", False
    )

    first = Tensor(device)
    trainer.optimizer.state["p"]["exp_avg"] = first
    trainer._offload_optimizer_state_to_cpu()
    assert first.copies == [False]
    assert trainer.optimizer._speco_first_blocking_optimizer_offload_done
    assert trainer.optimizer.state["metadata"] is None
    assert trainer.optimizer.state["p"]["step"].copies == []

    trainer.optimizer.state["p"]["exp_avg"] = Tensor(device)
    trainer._offload_optimizer_state_to_cpu()
    assert original_calls == [trainer.optimizer]

    replacement = Tensor(device)
    trainer.optimizer = SimpleNamespace(state={"p": {"exp_avg": replacement}})
    trainer._offload_optimizer_state_to_cpu()
    assert replacement.copies == [False]
    assert trainer.optimizer._speco_first_blocking_optimizer_offload_done


def test_failed_copy_does_not_mark_optimizer_complete(trainer):
    good, bad = Tensor(), Tensor(fail=True)
    trainer.optimizer.state = {"p": {"exp_avg": good, "exp_avg_sq": bad}}
    with pytest.raises(RuntimeError, match="copy failed"):
        trainer._offload_optimizer_state_to_cpu()
    assert not getattr(
        trainer.optimizer, "_speco_first_blocking_optimizer_offload_done", False
    )
    bad.fail = False
    trainer._offload_optimizer_state_to_cpu()
    assert trainer.optimizer._speco_first_blocking_optimizer_offload_done


@pytest.mark.parametrize(
    "device,strategy",
    [
        ("cpu", "fsdp2"),
        ("npu", "fsdp"),
        ("cuda", "fsdp"),
        ("npu", "ddp"),
        ("cuda", "ddp"),
        ("npu", "megatron"),
        ("cuda", "veomni"),
    ],
)
def test_other_devices_and_strategies_keep_original(
    trainer, monkeypatch, device, strategy
):
    monkeypatch.setattr(base, "device_name", device)
    trainer._drafter_strategy = lambda: strategy
    value = Tensor(device)
    trainer.optimizer.state = {"p": {"exp_avg": value}}
    calls = []
    monkeypatch.setattr(base, "offload_fsdp_optimizer", calls.append)
    trainer._offload_optimizer_state_to_cpu()
    assert calls == [trainer.optimizer]
    assert value.copies == []
    assert not getattr(
        trainer.optimizer, "_speco_first_blocking_optimizer_offload_done", False
    )


def test_veomni_still_blocks_every_offload(trainer):
    trainer._drafter_strategy = lambda: "veomni"
    for _ in range(2):
        value = Tensor()
        trainer.optimizer.state = {"p": {"exp_avg": value}}
        trainer._offload_optimizer_state_to_cpu()
        assert value.copies == [False]


def test_dtensor_uses_local_shard_to_detect_device(trainer):
    value = Tensor("cpu")
    value._local_tensor = Tensor("npu")
    trainer.optimizer.state = {"p": {"exp_avg": value}}
    trainer._offload_optimizer_state_to_cpu()
    assert value.copies == [False]
    assert trainer.optimizer._speco_first_blocking_optimizer_offload_done
