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

import asyncio
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


def test_release_draft_weights_payload_clears_before_host_reclaim(monkeypatch) -> None:
    payload = {"weight": object(), "bias": object()}
    reclaim_calls = []

    def fake_reclaim():
        reclaim_calls.append(dict(payload))
        return {
            "elapsed_sec": 0.25,
            "heap_trimmed": True,
            "allocator": "jemalloc",
            "reclaim_action": "purge",
        }

    monkeypatch.setattr(rollout_publish, "_release_publish_host_memory", fake_reclaim)

    result = rollout_publish.release_draft_weights_payload(payload)

    assert reclaim_calls == [{}]
    assert payload == {}
    assert result["num_weights"] == 2
    assert result["payload_cleared"] == 1


class _FakeDraftRollout:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.seen = []

    async def update_draft_weights(self, weights, global_steps=None):
        self.seen.append((dict(weights), global_steps))
        if self.fail:
            raise RuntimeError("publish failed")


def _publish_worker(rollout):
    worker = rollout_publish.DraftWeightPublishMixin()
    worker.config = {"rollout": {"drafter": {"enable": True}}}
    worker.rollout = rollout
    worker._attach_update_draft_weights_to_rollout = lambda: None
    return worker


@pytest.mark.parametrize("publish_async", [False, True])
def test_ref_backed_publish_releases_materialized_payload(
    monkeypatch, publish_async
) -> None:
    weight = object()
    materialized = {"weight": weight}
    rollout = _FakeDraftRollout()
    worker = _publish_worker(rollout)
    monkeypatch.setattr(
        rollout_publish,
        "materialize_draft_weights_payload",
        lambda _payload: (materialized, True),
    )
    monkeypatch.setattr(
        rollout_publish,
        "_release_publish_host_memory",
        lambda: {"elapsed_sec": 0.0, "allocator": "jemalloc"},
    )

    method = (
        worker.update_draft_weights_async
        if publish_async
        else worker.update_draft_weights
    )
    asyncio.run(method({"weights_ref": object()}, global_steps=2))

    assert rollout.seen == [({"weight": weight}, 2)]
    assert materialized == {}


def test_ref_backed_publish_releases_payload_after_failure(monkeypatch) -> None:
    materialized = {"weight": object()}
    worker = _publish_worker(_FakeDraftRollout(fail=True))
    monkeypatch.setattr(
        rollout_publish,
        "materialize_draft_weights_payload",
        lambda _payload: (materialized, True),
    )
    monkeypatch.setattr(
        rollout_publish,
        "_release_publish_host_memory",
        lambda: {"elapsed_sec": 0.0, "allocator": "glibc"},
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        asyncio.run(
            worker.update_draft_weights_async(
                {"weights_ref": object()}, global_steps=2
            )
        )

    assert materialized == {}


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


def test_explicit_disabled_drafter_wins_over_stale_runtime_env(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.sglang_runtime._load_env_drafter_config",
        lambda: {"enable": True},
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


def test_mrv2_dspark_publish_excludes_frozen_confidence_head(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
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
            "draft_model.confidence_head.proj.weight": torch.ones(1, 2),
            "draft_model.confidence_head.proj.bias": torch.ones(1),
            "draft_model.markov_head.markov_w1.weight": torch.ones(2, 2),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {
        "draft_model.markov_head.markov_w1.weight"
    }


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


@pytest.mark.parametrize(
    "release_method",
    ["release_training_memory_after_activation", "cleanup_training"],
)
def test_idle_drafter_lifecycle_offloads_dspark_target_lm_head(
    monkeypatch, release_method
) -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head lifecycle contract needs the trainer dependency stack",
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
    trainer.rank = 3
    trainer.backend = SimpleNamespace(
        model_type="dspark", target_model=None, target_lm_head=head
    )
    trainer.model = None
    trainer.optimizer = None
    trainer._training_active = True
    trainer._training_initialized = True
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    if release_method == "cleanup_training":
        trainer._pending_checkpoint_future = None
        trainer._pending_full_checkpoint_future = None
        trainer.skip_heavy_cleanup_after_drafter_training = False
        trainer._get_sp_group = lambda: None
        trainer._get_dp_group = lambda: None
        trainer.training_device_mesh = None
        trainer.collected_data = []
        trainer.data_buffer = []
        trainer._full_checkpoint_executor = None
        trainer._last_ckpt_step = 0
        trainer.training_steps = 1
        asyncio.run(trainer.cleanup_training(clear_data=True))
    else:
        asyncio.run(trainer.release_training_memory_after_activation())

    assert head.devices == ["cpu"]


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
