# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Verify opt-in Ascend behavior at shared native SpeCo entrypoints."""

import ast
import asyncio
import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.backends.training_backend import (
    is_vllm_ascend_drafter,
    rollout_capture_enabled,
)
from verl_speco.integration import vllm_runtime as runtime
from verl_speco.integration.ascend_state_config import training_worker_environment
from verl_speco.integration.rollout_admission import RolloutAdmission
from verl_speco.integration.serving_drafter_commit import AscendRolloutAdmission


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_deterministic_collectives_are_injected_only_into_ascend_trainer_workers(
    backend,
):
    config = {"rollout": {"drafter": drafter_config(backend, capture=False)}}
    previous = copy.deepcopy(config)
    expected = {"HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1"}
    assert training_worker_environment(config) == (
        expected if backend == "vllm_ascend" else {}
    )
    assert config == previous


def test_conflicting_training_communication_policy_is_rejected_before_ray_start():
    config = {
        "rollout": {
            "drafter": {
                "training": {
                    "backend": "vllm_ascend",
                    "vllm_ascend": {
                        "environment": {"HCCL_DETERMINISTIC": "false"},
                    },
                }
            }
        }
    }
    with pytest.raises(ValueError, match="HCCL_DETERMINISTIC=strict"):
        training_worker_environment(config)


def test_collective_policy_is_part_of_recovery_identity_and_requires_startup_env(
    monkeypatch,
):
    from verl_speco.integration.ascend_state_config import (
        require_deterministic_training_environment,
    )
    from verl_speco.integration.vllm_ascend_config import execution_environment

    config = SimpleNamespace(draft_engine=SimpleNamespace(environment={}))
    for name in ["HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC"]:
        monkeypatch.delenv(name, raising=False)
    original = execution_environment(config)
    with pytest.raises(RuntimeError, match="HCCL_DETERMINISTIC"):
        require_deterministic_training_environment()
    monkeypatch.setenv("HCCL_DETERMINISTIC", "strict")
    with pytest.raises(RuntimeError, match="LCCL_DETERMINISTIC"):
        require_deterministic_training_environment()
    monkeypatch.setenv("LCCL_DETERMINISTIC", "1")
    require_deterministic_training_environment()
    current = execution_environment(config)
    assert current != original
    assert current["HCCL_DETERMINISTIC"] == "strict"
    assert current["LCCL_DETERMINISTIC"] == "1"


def drafter_config(backend, capture=True):
    training = {
        "collect_hidden_states_from_rollout": capture,
        "rollout_capture": {"model_family": "deepseek_v4"},
    }
    if backend is not None:
        training["backend"] = backend
    return {"training": training}


def set_backend(monkeypatch, backend):
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps(drafter_config(backend)))
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
@pytest.mark.parametrize("capture", [False, True])
def test_capture_and_embedded_loader_require_ascend(monkeypatch, backend, capture):
    drafter = drafter_config(backend, capture)
    ascend = backend == "vllm_ascend"
    assert is_vllm_ascend_drafter(drafter) == ascend
    assert rollout_capture_enabled(drafter) is False
    monkeypatch.setattr(runtime, "_load_env_drafter_config", lambda: drafter)
    # A weight-only Ascend probe also needs the embedded loader.
    assert runtime._embedded_rollout_draft_enabled() == ascend


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_admission_class_and_new_commit_api_are_backend_specific(monkeypatch, backend):
    set_backend(monkeypatch, backend)

    async def run():
        server = runtime._SpecoVLLMHttpServerMixin()
        server.global_steps = 3
        gate = server._speco_rollout_admission()
        if backend == "vllm_ascend":
            assert type(gate) is AscendRolloutAdmission
            await server.speco_close_rollout_admission()
            result = await server.speco_commit_serving_drafter_revision(0, 1, 3)
            assert result["drafter_revision"] == 1
        else:
            assert type(gate) is RolloutAdmission
            assert not hasattr(gate, "waiting_requests")
            async with gate.request():
                assert gate._active == 1
            assert gate._active == 0
            with pytest.raises(RuntimeError, match="backend=vllm_ascend"):
                await server.speco_get_serving_versions()
            with pytest.raises(RuntimeError, match="backend=vllm_ascend"):
                await server.speco_commit_serving_drafter_revision(0, 1, 3)
            await server.speco_set_serving_drafter_revision(1)
            assert server._speco_serving_drafter_revision == 1

    asyncio.run(run())


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_shm_bucket_sizing_only_changes_for_ascend(monkeypatch, backend):
    from verl_speco.transport import bucketed_weight_transfer as transfer

    set_backend(monkeypatch, backend)
    sizes, adjustments, calls = [], [], []

    class Sender:
        def __init__(self, **kwargs):
            sizes.append(kwargs["bucket_size_mb"])

        async def async_send_weights(self, weights):
            calls.append("send")

    module = ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    module.BucketedWeightSender = Sender
    monkeypatch.setitem(sys.modules, module.__name__, module)

    async def call_server(adapter, method, *args, **kwargs):
        calls.append(method)

    async def complete():
        calls.append("commit")

    async def execute(method, **kwargs):
        calls.append(method)
        return complete()

    def required(weights, bucket):
        adjustments.append(bucket)
        return 1024

    monkeypatch.setattr(transfer, "required_bucket_megabytes", required)
    monkeypatch.setattr(runtime, "_maybe_call_vllm_server_method", call_server)
    monkeypatch.setattr(runtime, "_resolve_vllm_draft_update_use_shm", lambda *a: True)
    monkeypatch.setattr(
        runtime, "patch_verl_bucketed_weight_transfer_shm_reuse", lambda: False
    )
    monkeypatch.setattr(runtime, "_ipc_safe_allocator", lambda enabled: nullcontext())
    adapter = SimpleNamespace(
        rollout_rank=0,
        replica_rank=0,
        zmq_handle="ipc:///test",
        config=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=512)
        ),
        _execute_method=execute,
    )
    asyncio.run(
        runtime.speco_vllm_update_draft_weights(
            adapter, {"weight": object()}, global_steps=1
        )
    )
    assert sizes == ([1024] if backend == "vllm_ascend" else [512])
    assert adjustments == ([512] if backend == "vllm_ascend" else [])
    assert calls == [
        "abort_all_requests",
        "update_draft_weights_from_ipc",
        "send",
        "commit",
        "clear_kv_cache",
        "set_global_steps",
        "resume_generation",
    ]


def extract_class(path, name, methods, namespace=None):
    """Execute actual shared methods without importing Ray/NPU worker packages."""
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node = copy.deepcopy(node)
    node.bases, node.decorator_list = [], []
    node.body = [
        n
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods
    ]
    for method in node.body:
        method.decorator_list = []
    scope = {} if namespace is None else namespace
    exec(  # noqa: S102 -- trusted repository methods
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            path,
            "exec",
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_scheduler_registers_new_source_only_for_ascend(backend):
    from verl_speco.trainer.scheduler import DrafterCollectionSource, DrafterScheduler

    cls = extract_class(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_get_drafter_scheduler"},
        {"DrafterScheduler": DrafterScheduler},
    )
    trainer = cls()
    trainer.config = {"rollout": {"drafter": drafter_config(backend)}}
    scheduler = trainer._speco_get_drafter_scheduler()
    assert trainer._speco_get_drafter_scheduler() is scheduler
    assert "vllm" not in {source.value for source in scheduler._collection_adapters}
    assert DrafterCollectionSource.SGLANG in scheduler._collection_adapters
    assert DrafterCollectionSource.OLD_LOGPROB in scheduler._collection_adapters


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_v1_does_not_relax_native_oldlogprob_requirement(backend):
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

    drafter = drafter_config(backend)
    drafter.update(enable=True, enable_drafter_training=True)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"name": "vllm", "drafter": drafter}},
            "algorithm": {},
            "trainer": {"v1": {}},
        }
    )
    with pytest.raises(ValueError, match="old_logprob=true"):
        SpecoV1Mixin._speco_validate_v1_training_config(config)


@pytest.mark.parametrize("backend", [None, "pytorch"])
def test_native_v1_ignores_direct_capture_flag(backend):
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

    class Upstream:
        def _compute_old_log_prob(self, batch, metrics):
            return "native"

    class Trainer(SpecoV1Mixin, Upstream):
        _speco_freeze_policy = None

        def _speco_oldlogprob_collection_enabled(self):
            return False

        def _speco_v1_batch_data(self, batch):
            raise AssertionError("Ascend rollout collector entered native path")

    trainer = Trainer()
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"drafter": drafter_config(backend)}}}
    )
    trainer.global_steps = 1
    assert trainer._compute_old_log_prob(None, {}) == "native"
    assert "_speco_v1_rollout_collection_step" not in trainer.__dict__


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_worker_target_sync_selects_backend_rather_than_method_presence(backend):
    from typing import Optional

    cls = extract_class(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {"sync_target_lm_head_weight"},
        {"Optional": Optional},
    )
    worker = cls()
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.is_drafter_group_leader = False
    worker.config = {"rollout": {"drafter": drafter_config(backend)}}
    calls = []

    def sync_output(payload, step):
        calls.append("ascend")
        return {"applied": True}

    def sync_head(weight, **kwargs):
        assert weight == "weight" and kwargs["global_step"] == 5
        calls.append("native")
        return {"applied": True}

    worker.trainer = SimpleNamespace(
        sync_target_output=sync_output,
        sync_target_lm_head_weight=sync_head,
    )
    assert worker.sync_target_lm_head_weight({"weight": "weight"}, 5)["applied"]
    assert calls == (["ascend"] if backend == "vllm_ascend" else ["native"])


@pytest.mark.parametrize("backend", [None, "pytorch", "vllm_ascend"])
def test_parking_worker_environments_are_explicitly_opt_in(backend):
    from verl_speco.integration.ascend_state_config import parking_worker_environments

    drafter = drafter_config(backend)
    drafter["training"]["vllm_ascend"] = {"state_parking": "camem"}
    config = {"rollout": {"drafter": drafter}}
    training, serving = parking_worker_environments(config)
    if backend == "vllm_ascend":
        assert training["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
        assert training["HCCL_NPU_SOCKET_PORT_RANGE"] == "61000-61099"
        assert serving == {
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "HCCL_NPU_SOCKET_PORT_RANGE": "62000-62099",
            "HCCL_HOST_SOCKET_PORT_RANGE": "62100-62199",
        }
    else:
        assert training == serving == {}


def test_disabled_parking_and_offline_mode_leave_worker_environments_unchanged():
    from verl_speco.integration.ascend_state_config import parking_worker_environments

    config = {"rollout": {"drafter": drafter_config("vllm_ascend")}}
    assert parking_worker_environments(config) == ({}, {})
    config["rollout"]["drafter"]["training"].update(
        mode="offline", vllm_ascend={"state_parking": "camem"}
    )
    assert parking_worker_environments(config) == ({}, {})


@pytest.mark.parametrize(
    "backend,parking",
    [
        (None, "camem"),
        ("pytorch", "camem"),
        ("vllm_ascend", "none"),
        ("vllm_ascend", "camem"),
    ],
)
def test_native_worker_factory_only_passes_parking_env_to_selected_backend(
    backend, parking
):
    role = SimpleNamespace(ActorRolloutRef="actor_ref", ActorRollout="actor")
    cls = extract_class(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_init_speco_drafter_workers"},
        {"Role": role, "RayClassWithInitArgs": lambda **kwargs: kwargs},
    )
    trainer = cls()
    drafter = drafter_config(backend)
    drafter["training"]["vllm_ascend"] = {"state_parking": parking}
    trainer.config = SimpleNamespace(
        actor_rollout_ref={"rollout": {"drafter": drafter}}
    )
    trainer.drafter_wg = None
    trainer.speco_worker_cls = object()
    trainer.role_worker_mapping = {role.ActorRolloutRef: object()}
    trainer.resource_pool_manager = SimpleNamespace(
        get_resource_pool=lambda key: "pool"
    )
    trainer.device_name = "npu"
    captured = {}

    def worker_group(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(init_model=lambda: None)

    trainer.ray_worker_group_cls = worker_group
    trainer.attach_speco_worker_group = lambda group: setattr(
        trainer, "drafter_wg", group
    )
    trainer._init_speco_drafter_workers()
    assert ("worker_env" in captured) == (backend == "vllm_ascend")
    if "worker_env" in captured:
        assert captured["worker_env"]["HCCL_DETERMINISTIC"] == "strict"
        assert captured["worker_env"]["LCCL_DETERMINISTIC"] == "1"
        assert ("PYTORCH_NPU_ALLOC_CONF" in captured["worker_env"]) == (
            parking == "camem"
        )
        if parking == "camem":
            assert (
                captured["worker_env"]["PYTORCH_NPU_ALLOC_CONF"]
                == "expandable_segments:False"
            )


def test_serving_parking_environment_preserves_ray_setup_and_custom_options(
    monkeypatch,
):
    drafter = drafter_config("vllm_ascend")
    drafter["training"]["vllm_ascend"] = {"state_parking": "camem"}
    monkeypatch.setattr(runtime, "_load_env_drafter_config", lambda: drafter)
    captured = {}

    def options(**kwargs):
        captured.update(kwargs)
        return captured

    wrapper = runtime._SpecoVLLMHttpServerActorClass(SimpleNamespace(options=options))
    wrapper.options(
        runtime_env={"env_vars": {"USER_KEY": "keep"}, "working_dir": "/tmp/keep"}
    )
    env = captured["runtime_env"]
    assert env["env_vars"]["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert env["env_vars"]["HCCL_NPU_SOCKET_PORT_RANGE"] == "62000-62099"
    assert env["env_vars"]["USER_KEY"] == "keep" and env["working_dir"] == "/tmp/keep"
    assert env["worker_process_setup_hook"] == wrapper._SETUP_HOOK_PATH
    with pytest.raises(ValueError, match="Conflicting Ascend serving"):
        wrapper.options(
            runtime_env={"env_vars": {"HCCL_NPU_SOCKET_PORT_RANGE": "61000-61099"}}
        )


def test_serving_camem_callbacks_are_retained_before_model_loading(monkeypatch):
    drafter = drafter_config("vllm_ascend")
    drafter["training"]["vllm_ascend"] = {"state_parking": "camem"}
    monkeypatch.setattr(runtime, "_load_env_drafter_config", lambda: drafter)
    monkeypatch.setattr(runtime, "patch_vllm_dspark_draft_load_config", lambda: None)

    class Allocator:
        def python_malloc_callback(self, handle):
            return handle

        def python_free_callback(self, ptr):
            return ptr

    allocator = Allocator()
    module = ModuleType("vllm_ascend.device_allocator.camem")
    module.CaMemAllocator = SimpleNamespace(get_instance=lambda: allocator)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime.SpecoVLLMColocateWorkerExtension()
    assert allocator.python_malloc_callback is allocator.python_malloc_callback
    assert allocator.python_free_callback is allocator.python_free_callback
