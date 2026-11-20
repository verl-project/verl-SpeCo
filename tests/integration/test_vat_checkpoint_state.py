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
import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf
from verl_speco.backends.dflash_trainer_backend import DFlashTrainingModel
from verl_speco.models.dflash import DFlashConfig, DFlashDraftModel
from verl_speco.trainer.base_trainer import DrafterBaseTrainer


def _trainer():
    config = DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    model = DFlashTrainingModel(
        DFlashDraftModel(config),
        block_size=4,
        num_anchors=1,
        per_position_loss_weight="vat",
    )
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer.optimizer_steps_total = trainer.training_steps = 1
    trainer.lr_scheduler = None
    trainer.fsdp_device_mesh = None
    trainer._use_replica_local_bubble_group = lambda: False
    trainer._is_checkpoint_leader = lambda: True
    trainer._get_sp_world_size = trainer._get_dp_world_size = lambda: 1
    trainer._get_full_export_state_dict = lambda: model.state_dict()
    return trainer


def _step(trainer):
    trainer.optimizer.zero_grad()
    weight = next(p for p in trainer.model.draft_model.parameters() if p.requires_grad)
    x = weight.reshape(-1)[:8].expand(2, 8)
    loss = trainer.model.vat_objective.accept_head(x).square().sum()
    loss = loss + sum(
        p.square().mean() for p in trainer.model.parameters() if p.requires_grad
    )
    loss.backward()
    trainer.optimizer.step()
    return loss.detach()


def test_auxiliary_head_production_checkpoint_resume(tmp_path):
    source = _trainer()
    _step(source)
    snapshot = copy.deepcopy(source.model.state_dict())
    exported = source._get_pretrained_export_state_dict()
    assert set(exported) == set(source.model.draft_model.state_dict())
    assert not any("accept_head" in name for name in exported)
    manifest = source._save_optimizer_checkpoint(str(tmp_path))
    metadata = {"optimizer": manifest}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    resumed = _trainer()
    resumed.model.draft_model.load_state_dict(exported)
    original_load = resumed.optimizer.load_state_dict

    def load_optimizer(state_dict):
        torch.testing.assert_close(
            resumed.model.vat_objective.accept_head.weight,
            snapshot["vat_objective.accept_head.weight"],
        )
        original_load(state_dict)

    resumed.optimizer.load_state_dict = load_optimizer
    resumed._load_optimizer_checkpoint(str(tmp_path), metadata)
    for key, value in snapshot.items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value)
    torch.testing.assert_close(_step(source), _step(resumed))
    for a, b in zip(source.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b)
    del manifest["auxiliary_model"]
    with pytest.raises(RuntimeError, match="missing VAT acceptance-head"):
        resumed._load_optimizer_checkpoint(str(tmp_path), metadata)


@pytest.mark.parametrize("kind", ["eagle3", "dflash", "dspark"])
def test_opd_two_rank_ulysses_startup_policy(kind, monkeypatch):
    import verl_speco.trainer.base_trainer as base

    monkeypatch.setattr(base, "get_device_id", lambda: 0)
    monkeypatch.setattr(base, "device_name", "cpu")
    monkeypatch.setattr(DrafterBaseTrainer, "_create_copy_stream", lambda self: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    config = OmegaConf.create(
        {
            "rollout": {
                "tensor_model_parallel_size": 2,
                "drafter": {"training": {"drafter_opd": {"enable": True}}},
            }
        }
    )
    backend = SimpleNamespace(model_type=kind)

    class ReachedMesh(Exception):
        pass

    def mesh(self):
        assert not self.use_ulysses_sp
        raise ReachedMesh()

    monkeypatch.setattr(DrafterBaseTrainer, "_resolve_drafter_fsdp_device_mesh", mesh)
    expected = ValueError if kind == "eagle3" else ReachedMesh
    message = "Draft-OPD does not support Ulysses" if kind == "eagle3" else None
    with pytest.raises(expected, match=message):
        DrafterBaseTrainer(config, 2, 0, None, backend, training_process_group=object())
