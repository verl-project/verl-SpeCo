from __future__ import annotations

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
    assert rollout_publish.materialize_draft_weights_payload(ref) == ({"resolved": ref}, True)
    assert rollout_publish.materialize_draft_weights_payload({"weights_ref": ref}) == (
        {"resolved": ref},
        True,
    )


def test_rollout_backend_and_drafter_gates_support_both_config_shapes() -> None:
    assert rollout_publish.rollout_backend_name({"rollout": {"name": "vllm"}}) == "vllm"
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


def test_publish_state_filter_keeps_eagle3_trainable_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="eagle3")
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

    assert set(trainer._get_trainable_state_dict()) == {"lm_head.weight", "midlayer.fc.weight"}


def test_publish_state_filter_skips_non_eagle_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dflash")
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
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.training_device_mesh = None
    trainer._frozen_param_names = []
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "draft_model.embed_tokens.weight": torch.ones(2, 2),
            "draft_model.fc.weight": torch.ones(2, 2),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {"draft_model.fc.weight"}


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
