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
from __future__ import annotations

from inspect import getsource
from types import SimpleNamespace

import pytest

from verl_speco.integration import rollout_publish


class _FakeObjectRef:
    pass


class _FakeRay:
    ObjectRef = _FakeObjectRef

    @staticmethod
    def get(value):
        return {"resolved": value}


def test_materialize_direct_and_object_ref_payloads(monkeypatch) -> None:
    monkeypatch.setattr(rollout_publish, "_ray_module", lambda: _FakeRay)
    direct = {"weight": 1}
    ref = _FakeObjectRef()

    assert rollout_publish.materialize_draft_weights_payload(direct) == (direct, False)
    assert rollout_publish.materialize_draft_weights_payload(ref) == (
        {"resolved": ref},
        True,
    )
    assert rollout_publish.materialize_draft_weights_payload({"weights_ref": ref}) == (
        {"resolved": ref},
        True,
    )


def test_rollout_backend_and_drafter_gates_support_both_config_shapes() -> None:
    assert rollout_publish.rollout_backend_name({"rollout": {"name": "vllm"}}) == "vllm"
    assert rollout_publish.actor_training_backend_name({}) == "fsdp"
    assert (
        rollout_publish.actor_training_backend_name(
            {"actor_rollout_ref": {"actor": {"strategy": "fsdp2"}}}
        )
        == "fsdp2"
    )
    assert (
        rollout_publish.rollout_backend_name(
            {"actor_rollout_ref": {"rollout": {"name": "sglang"}}}
        )
        == "sglang"
    )
    assert rollout_publish.drafter_rollout_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": {"enable": True}}}}
    )
    assert not rollout_publish.drafter_rollout_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": {"enable": False}}}}
    )


def test_veomni_backend_and_parallel_layout_support_worker_config_shape() -> None:
    config = {
        "actor": {
            "strategy": "veomni",
            "veomni": {
                "ulysses_parallel_size": 2,
                "expert_parallel_size": 4,
                "router_replay": {"mode": "R3"},
            },
        },
        "rollout": {"enable_rollout_routing_replay": True},
    }

    assert rollout_publish.actor_training_backend_name(config) == "veomni"
    assert rollout_publish.veomni_parallel_layout(config) == {
        "ulysses_parallel_size": 2,
        "expert_parallel_size": 4,
        "router_replay_mode": "R3",
        "rollout_routing_replay": True,
    }


@pytest.mark.parametrize(
    ("router_mode", "rollout_replay", "error_match"),
    [
        ("invalid", False, "disabled, R2, or R3"),
        ("R2", True, "R2 must not enable"),
        ("R3", False, "R3 requires"),
    ],
)
def test_veomni_router_replay_contract_fails_closed(
    router_mode: str, rollout_replay: bool, error_match: str
) -> None:
    config = {
        "actor": {
            "strategy": "veomni",
            "veomni": {"router_replay": {"mode": router_mode}},
        },
        "rollout": {"enable_rollout_routing_replay": rollout_replay},
    }

    with pytest.raises((ValueError, RuntimeError), match=error_match):
        rollout_publish.validate_veomni_parallel_layout(config)


def test_veomni_r3_contract_accepts_rollout_routing_replay() -> None:
    config = {
        "actor": {
            "strategy": "veomni",
            "veomni": {"router_replay": {"mode": "R3"}},
        },
        "rollout": {"enable_rollout_routing_replay": True},
    }

    layout = rollout_publish.validate_veomni_parallel_layout(config)

    assert layout["router_replay_mode"] == "R3"


def test_veomni_oldlogprob_runtime_skips_ref_only_worker(monkeypatch) -> None:
    from verl_speco.integration import oldlogprob_runtime

    install_called = False

    def fake_install(*, actor_backend=None):
        nonlocal install_called
        install_called = True
        return True

    monkeypatch.setattr(
        oldlogprob_runtime,
        "install_oldlogprob_hidden_runtime_patch",
        fake_install,
    )
    worker = SimpleNamespace(
        _is_actor=False,
        config={"actor": {"strategy": "veomni"}},
    )

    rollout_publish.install_oldlogprob_hidden_runtime_for_worker(worker)
    rollout_publish.validate_oldlogprob_hidden_runtime_for_worker(worker)

    assert install_called is False


@pytest.mark.parametrize(
    ("row_indices", "expected_strategy", "expected_rows"),
    [
        (None, "veomni_lm_head_full", 6),
        ([1, 4], "veomni_lm_head_sparse", 2),
    ],
)
def test_veomni_lm_head_export_avoids_full_engine_state_dict(
    row_indices, expected_strategy: str, expected_rows: int
) -> None:
    torch = pytest.importorskip("torch")

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(3, 6, bias=False)

    class _Engine:
        module = _Module()

        @staticmethod
        def get_per_tensor_param(**kwargs):
            raise AssertionError("VeOmni export must not enumerate the full state dict")

    worker = SimpleNamespace(
        _is_actor=True,
        rank=0,
        config={"actor": {"strategy": "veomni"}},
        actor=SimpleNamespace(engine=_Engine()),
    )

    payload = rollout_publish.export_actor_lm_head_weight(
        worker, row_indices=row_indices
    )

    assert payload["export_strategy"] == expected_strategy
    assert payload["actor_backend"] == "veomni"
    assert tuple(payload["weight"].shape) == (expected_rows, 3)


def test_veomni_runtime_validation_checks_initialized_model_contract() -> None:
    torch = pytest.importorskip("torch")

    class _TextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
            self.norm = torch.nn.LayerNorm(3)

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TextModel()
            self.lm_head = torch.nn.Linear(3, 6, bias=False)

    config = {
        "actor": {
            "strategy": "veomni",
            "veomni": {
                "ulysses_parallel_size": 2,
                "expert_parallel_size": 1,
                "router_replay": {"mode": "R2"},
            },
        },
        "rollout": {
            "drafter": {
                "enable": True,
                "enable_drafter_training": True,
                "training": {"collect_hidden_states_from_old_logprob": True},
            }
        },
    }
    worker = SimpleNamespace(
        rank=0,
        config=config,
        actor=SimpleNamespace(engine=SimpleNamespace(module=_Module())),
    )

    rollout_publish.validate_oldlogprob_hidden_runtime_for_worker(worker)


def test_publish_state_filter_keeps_eagle3_trainable_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(
        model_type="eagle3",
        trains_draft_lm_head=True,
        trains_draft_embeddings=False,
    )
    trainer.training_device_mesh = None
    trainer._frozen_param_names = ["target_model."]
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "embed_tokens.weight": torch.ones(2, 2),
            "target_model.fc.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "midlayer.fc.weight": torch.ones(2, 2),
            "t2d": torch.ones(2, dtype=torch.bool),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {
        "lm_head.weight",
        "midlayer.fc.weight",
    }


def test_publish_state_filter_skips_non_eagle_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(
        model_type="dflash",
        trains_draft_lm_head=False,
        trains_draft_embeddings=False,
    )
    trainer.training_device_mesh = None
    trainer._frozen_param_names = []
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "lm_head.weight": torch.ones(2, 2),
            "draft_model.fc.weight": torch.ones(2, 2),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {"draft_model.fc.weight"}


def test_publish_state_filter_excludes_block_drafter_embedding() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(
        model_type="dspark",
        trains_draft_lm_head=False,
        trains_draft_embeddings=False,
    )
    trainer.training_device_mesh = None
    trainer._frozen_param_names = []
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "draft_model.embed_tokens.weight": torch.ones(2, 2),
            "draft_model.fc.weight": torch.ones(2, 2),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {"draft_model.fc.weight"}


def test_publish_state_filter_keeps_peagle_trained_head_and_embedding() -> None:
    """P-EAGLE owns its lm_head and fine-tunes the draft embedding.

    Both are trained every drafter step, so hot publish has to ship them; the
    generic filter used to drop them and leave the rollout engine on the initial
    weights forever.
    """
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(
        model_type="peagle",
        trains_draft_lm_head=True,
        trains_draft_embeddings=True,
    )
    trainer.training_device_mesh = None
    trainer._frozen_param_names = ["target_model."]
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "embed_tokens.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "fc.weight": torch.ones(2, 2),
            "mask_hidden": torch.ones(1, 1, 2),
            "target_model.fc.weight": torch.ones(2, 2),
            "t2d": torch.ones(2, dtype=torch.bool),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {
        "embed_tokens.weight",
        "lm_head.weight",
        "fc.weight",
        "mask_hidden",
    }


def test_backend_publish_contract_matches_what_each_backend_trains() -> None:
    """The declared flags must track the real freeze/build calls in each backend.

    ``model_type`` is not a usable proxy here: EAGLE-1/2 deliberately report
    ``"eagle3"`` to reuse the data plumbing, and P-EAGLE is the only backend that
    skips ``freeze_embedding()``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend
    from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend
    from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend
    from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    expected = {
        Eagle3TrainerBackend: (True, False),
        Eagle1TrainerBackend: (False, False),
        PEagleTrainerBackend: (True, True),
        DFlashTrainerBackend: (False, False),
        DSparkTrainerBackend: (False, False),
        DominoTrainerBackend: (False, False),
    }
    for backend_cls, (lm_head, embeddings) in expected.items():
        assert backend_cls.trains_draft_lm_head is lm_head, backend_cls.__name__
        assert backend_cls.trains_draft_embeddings is embeddings, backend_cls.__name__


def test_target_lm_head_device_helper_handles_dflash_style_backend() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head device contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    class _FakeHead:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    head = _FakeHead()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(target_lm_head=head)

    assert trainer._move_target_lm_head("cpu") is True
    assert head.devices == ["cpu"]


def test_target_lm_head_device_helper_preserves_eagle_backend() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target model device contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    class _FakeHead:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    head = _FakeHead()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(target_model=head, target_lm_head=None)

    assert trainer._move_target_lm_head("npu:0") is True
    assert head.devices == ["npu:0"]


def test_drafter_state_is_offloaded_after_training_and_warmup() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head cleanup contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    assert '_move_target_lm_head("cpu")' in getsource(
        DrafterBaseTrainer.cleanup_training
    )
    assert '_move_target_lm_head("cpu")' in getsource(
        DrafterBaseTrainer.release_training_memory_after_activation
    )
    assert "self._offload_optimizer_state_to_cpu()" in getsource(
        DrafterBaseTrainer.cleanup_training
    )
    assert "self._offload_optimizer_state_to_cpu()" in getsource(
        DrafterBaseTrainer.release_training_memory_after_activation
    )


def test_drafter_full_shard_mesh_reuses_default_process_group() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter mesh contract needs the trainer dependency stack",
    )
    source = getsource(
        base_trainer.DrafterBaseTrainer._resolve_drafter_fsdp_device_mesh
    )

    assert 'getattr(DeviceMesh, "from_group", None)' in source
    assert "dist.group.WORLD" in source
    assert 'mesh._flatten(mesh_dim_name="fsdp")' in source


def test_target_lm_head_sync_can_defer_device_apply() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head sync needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    head = torch.nn.Linear(3, 4, bias=False)
    head.weight.data.zero_()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(
        model_type="dspark",
        target_model=None,
        target_lm_head=SimpleNamespace(fc=head),
    )
    trainer._pending_target_lm_head_weight = None
    trainer._pending_target_lm_head_row_indices = None
    trainer._pending_target_lm_head_source_vocab_size = None
    trainer._target_lm_head_weight_step = None
    source_weight = torch.ones_like(head.weight)

    result = trainer.sync_target_lm_head_weight(
        source_weight,
        global_step=3,
        defer_device_apply=True,
    )

    assert result["accepted"] is True
    assert result["applied"] is False
    assert result["pending"] is True
    assert result["deferred"] is True
    assert torch.count_nonzero(head.weight) == 0

    assert trainer._apply_pending_target_lm_head_weight() is True
    assert torch.equal(head.weight, source_weight)


def test_dspark_pretrained_export_strips_only_training_wrapper_prefix() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="checkpoint export needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.training_device_mesh = None
    trainer.model = SimpleNamespace(
        draft_model=SimpleNamespace(),
        state_dict=lambda: {
            "draft_model.fc.weight": torch.ones(2, 4),
            "draft_model.hidden_norm.weight": torch.ones(2),
            "draft_model.norm.weight": torch.ones(2),
            "draft_model.markov_head.markov_w1.weight": torch.ones(2, 2),
        },
    )

    exported_state = trainer._get_pretrained_export_state_dict()

    assert set(exported_state) == {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
    }
