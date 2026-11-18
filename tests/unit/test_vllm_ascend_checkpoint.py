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
"""Recovery state survives process-equivalent reconstruction and rejects drift."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_speco.backends.vllm_ascend_engine import Progress
from verl_speco.trainer import vllm_ascend_checkpoint as cp
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore


class CPUCheckpointEngine:
    def __init__(self):
        self.module = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )
        self.global_steps = 0
        self.loads = 0

    def update(self):
        self.module(torch.ones(2, 2)).square().mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_steps += 1

    def save_checkpoint(self, path, *, tag, client_state):
        target = Path(path) / tag
        target.mkdir(parents=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.global_steps,
                "client": client_state,
            },
            target / "state.pt",
        )

    def load_checkpoint(self, path, *, tag):
        self.loads += 1
        value = torch.load(Path(path) / tag / "state.pt", weights_only=False)
        self.module.load_state_dict(value["model"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.scheduler.load_state_dict(value["scheduler"])
        self.global_steps = value["step"]
        return str(Path(path) / tag), value["client"]


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    store = TorchShardFeatureStore(tmp_path / "store", max_samples_per_shard=1)
    store.write_many(
        [DraftFeatureSample(torch.arange(4), torch.ones(4), torch.ones(4, 2))]
    )
    store.flush()
    topology = SimpleNamespace(
        metadata=lambda: {
            "world_size": 1,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "global_rank": 0,
        }
    )

    def wrapper():
        return SimpleNamespace(
            model_engine=CPUCheckpointEngine(),
            progress=Progress(),
            topology=topology,
            logger=logging.getLogger(__name__),
        )

    def save(engine, base, tag, logger, *, client_state, **kwargs):
        assert kwargs["save_full_model_per_rank"] is True
        engine.save_checkpoint(base, tag=tag, client_state=client_state)

    monkeypatch.setattr(
        cp,
        "save_deepspeed_checkpoint",
        save,
    )
    engine = wrapper()
    engine.model_engine.update()
    engine.progress = Progress(
        global_step=1, global_examples=2, epoch=0, batch_cursor=2
    )
    identity = {"source": "air-revision", "model": "0731", "settings": "numerics"}
    path = cp.save_checkpoint(
        engine,
        base_dir=tmp_path / "ckpt",
        tag="step_00000001",
        identity=identity,
        store=store,
    )
    cp.write_checkpoint_metadata(path, engine=engine, identity=identity, store=store)
    return engine, wrapper(), identity, path, store


def test_adam_scheduler_and_data_cursor_resume_match_continuous_update(recovery):
    original, restored, identity, path, store = recovery
    cp.restore_checkpoint(restored, path=path, identity=identity, store=store)
    assert restored.progress == original.progress
    assert restored.model_engine.optimizer.state
    assert (
        restored.model_engine.scheduler.state_dict()
        == original.model_engine.scheduler.state_dict()
    )
    original.model_engine.update()
    restored.model_engine.update()
    for a, b in zip(
        original.model_engine.module.parameters(),
        restored.model_engine.module.parameters(),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_changed_source_is_rejected_before_loading(recovery):
    _, restored, identity, path, store = recovery
    with pytest.raises(ValueError, match="identity"):
        cp.restore_checkpoint(
            restored, path=path, identity={**identity, "source": "other"}, store=store
        )
    assert restored.model_engine.loads == 0


def test_changed_topology_is_rejected_before_loading(recovery):
    _, restored, identity, path, store = recovery
    restored.topology = SimpleNamespace(
        metadata=lambda: {
            "world_size": 2,
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
        }
    )
    with pytest.raises(ValueError, match="topology"):
        cp.restore_checkpoint(restored, path=path, identity=identity, store=store)
    assert restored.model_engine.loads == 0


def test_deepspeed_step_mismatch_does_not_silently_resume(recovery):
    _, restored, identity, path, store = recovery
    native = restored.model_engine.load_checkpoint

    def bad_load(*args, **kwargs):
        result = native(*args, **kwargs)
        restored.model_engine.global_steps = 0
        return result

    restored.model_engine.load_checkpoint = bad_load
    with pytest.raises(RuntimeError, match="global step"):
        cp.restore_checkpoint(restored, path=path, identity=identity, store=store)


def test_original_numerical_implementation_cannot_resume_optimizer_into_new_math(
    recovery,
):
    import json

    from verl_speco.integration.vllm_ascend_implementation import (
        LEGACY_IMPLEMENTATION_FINGERPRINT,
        implementation_fingerprint,
    )

    _, restored, identity, path, store = recovery
    legacy = {**identity, "source": LEGACY_IMPLEMENTATION_FINGERPRINT}
    sidecar = Path(path) / "speco_ds4.json"
    metadata = json.loads(sidecar.read_text())
    metadata["identity"] = legacy
    sidecar.write_text(json.dumps(metadata))
    artifact = Path(path) / "state.pt"
    state = torch.load(artifact, weights_only=False)
    state["client"][cp.STATE_KEY]["identity"] = legacy
    torch.save(state, artifact)
    current = {**identity, "source": implementation_fingerprint()}
    with pytest.raises(ValueError, match="identity"):
        cp.restore_checkpoint(restored, path=path, identity=current, store=store)
    assert restored.model_engine.loads == 0


def test_checkpoint_restore_refreshes_unregistered_router_cache_in_place(recovery):
    original, restored, identity, path, store = recovery
    layer = restored.model_engine.module
    layer.precast_fp32_weight = True
    layer.weight_fp32 = torch.full_like(layer.weight, -123.0)
    cache, pointer = layer.weight_fp32, layer.weight_fp32.data_ptr()
    cp.restore_checkpoint(restored, path=path, identity=identity, store=store)
    assert layer.weight_fp32 is cache and cache.data_ptr() == pointer
    torch.testing.assert_close(cache, original.model_engine.module.weight, rtol=0, atol=0)
