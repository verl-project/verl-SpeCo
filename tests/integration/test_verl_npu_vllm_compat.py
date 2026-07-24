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
import importlib
import sys
import types

import pytest

from verl_speco.integration import verl_npu_vllm_compat as compat


def test_factory_fused_moe_survives_verl_npu_patch_import(monkeypatch) -> None:
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.23.0"
    fused_moe_module = types.ModuleType(compat._VLLM_FUSED_MOE_MODULE)

    def fused_moe_factory(*args, **kwargs):
        return args, kwargs

    fused_moe_module.FusedMoE = fused_moe_factory
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    monkeypatch.setitem(sys.modules, compat._VLLM_FUSED_MOE_MODULE, fused_moe_module)
    monkeypatch.setattr(compat, "_IMPORT_COMPAT_APPLIED", False)

    def module_importer(module_name: str):
        if module_name == compat._VERL_NPU_VLLM_PATCH_MODULE:
            assert hasattr(fused_moe_factory, "weight_loader")
            original = fused_moe_factory.weight_loader

            def wrapped_weight_loader(*args, **kwargs):
                return original(*args, **kwargs)

            fused_moe_factory.weight_loader = wrapped_weight_loader
            module = types.ModuleType(module_name)
            monkeypatch.setitem(sys.modules, module_name, module)
            return module
        return importlib.import_module(module_name)

    assert compat.install_verl_npu_vllm_import_compat(module_importer) is True
    assert not hasattr(fused_moe_factory, "weight_loader")
    assert compat._IMPORT_COMPAT_APPLIED is True


def test_worker_mixin_installs_compat_before_base_init(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        compat, "install_verl_npu_vllm_import_compat", lambda: events.append("compat")
    )
    monkeypatch.setattr(
        compat,
        "install_verl_fsdp_training_output_release_compat",
        lambda: events.append("training_output_release"),
    )
    monkeypatch.setattr(
        compat,
        "install_verl_npu_checkpoint_reclaim",
        lambda: events.append("reclaim"),
    )
    monkeypatch.setattr(
        compat,
        "install_verl_npu_fsdp2_weight_export_compat",
        lambda: events.append("fsdp2_export"),
    )

    class BaseWorker:
        def __init__(self):
            events.append("base")

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    WrappedWorker()
    assert events == [
        "compat",
        "training_output_release",
        "reclaim",
        "fsdp2_export",
        "base",
    ]


def test_worker_mixin_installs_shm_reuse_before_weight_update(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        compat,
        "_install_weight_transfer_shm_reuse",
        lambda: events.append(("shm_reuse",)) or True,
    )

    class BaseWorker:
        rank = 3

        async def update_weights(self, global_steps=None, mode="auto"):
            events.append(("update", global_steps, mode))
            return "updated"

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    worker = WrappedWorker.__new__(WrappedWorker)
    result = asyncio.run(worker.update_weights(global_steps=4, mode="naive"))

    assert result == "updated"
    assert events == [("shm_reuse",), ("update", 4, "naive")]


def test_worker_mixin_preserves_weight_update_failure(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        compat,
        "_install_weight_transfer_shm_reuse",
        lambda: events.append(("shm_reuse",)) or True,
    )

    class BaseWorker:
        rank = 1

        async def update_weights(self, global_steps=None, mode="auto"):
            del global_steps, mode
            raise RuntimeError("update failed")

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    worker = WrappedWorker.__new__(WrappedWorker)
    with pytest.raises(RuntimeError, match="update failed"):
        asyncio.run(worker.update_weights(global_steps=6))
    assert events == [("shm_reuse",)]


def test_npu_checkpoint_reclaim_preserves_native_save(monkeypatch) -> None:
    events = []
    engine_module = types.ModuleType(compat._VERL_FSDP_ENGINE_MODULE)

    class FSDPEngine:
        def save_checkpoint(self, *args, **kwargs):
            events.append(("original", args, kwargs))
            return "saved"

    engine_module.FSDPEngine = FSDPEngine
    device_module = types.SimpleNamespace(get_device_name=lambda: "npu")
    modules = {
        compat._VERL_FSDP_ENGINE_MODULE: engine_module,
        "verl.utils.device": device_module,
    }
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(compat, "_NPU_CHECKPOINT_RECLAIM_APPLIED", False)
    monkeypatch.setattr(
        compat,
        "release_checkpoint_host_memory",
        lambda path, drop_file_cache: events.append(("reclaim", path, drop_file_cache))
        or {"elapsed_sec": 0.1, "files_advised": 3, "files_failed": 0},
    )
    monkeypatch.setattr(compat, "format_checkpoint_memory_snapshot", lambda: "memory")

    assert compat.install_verl_npu_checkpoint_reclaim(modules.__getitem__) is True

    engine = FSDPEngine()
    engine.rank = 0

    assert (
        engine.save_checkpoint("/tmp/actor", global_step=20, max_ckpt_to_keep=1)
        == "saved"
    )
    assert events == [
        (
            "original",
            ("/tmp/actor",),
            {
                "hdfs_path": None,
                "global_step": 20,
                "max_ckpt_to_keep": 1,
            },
        ),
        ("reclaim", "/tmp/actor", True),
    ]


def test_npu_checkpoint_reclaim_runs_after_native_save_failure(monkeypatch) -> None:
    events = []
    engine_module = types.ModuleType(compat._VERL_FSDP_ENGINE_MODULE)

    class FSDPEngine:
        rank = 1

        def save_checkpoint(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("save failed")

    engine_module.FSDPEngine = FSDPEngine
    modules = {
        compat._VERL_FSDP_ENGINE_MODULE: engine_module,
        "verl.utils.device": types.SimpleNamespace(get_device_name=lambda: "npu"),
    }
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(compat, "_NPU_CHECKPOINT_RECLAIM_APPLIED", False)
    monkeypatch.setattr(
        compat,
        "release_checkpoint_host_memory",
        lambda path, drop_file_cache: events.append((path, drop_file_cache))
        or {"elapsed_sec": 0.1, "files_advised": 0, "files_failed": 0},
    )

    assert compat.install_verl_npu_checkpoint_reclaim(modules.__getitem__) is True
    with pytest.raises(RuntimeError, match="save failed"):
        FSDPEngine().save_checkpoint("/tmp/actor")
    assert events == [(None, False)]


def test_npu_checkpoint_skips_manual_move_with_fsdp2_cpu_offload(monkeypatch) -> None:
    events = []
    engine_module = types.ModuleType(compat._VERL_FSDP_ENGINE_MODULE)
    engine_module.torch = types.SimpleNamespace(
        distributed=types.SimpleNamespace(barrier=lambda: events.append(("barrier",)))
    )

    class CheckpointManager:
        def save_checkpoint(self, **kwargs):
            events.append(("manager", kwargs))

    class FSDPEngine:
        rank = 0
        _uses_fsdp2_cpu_offload_policy = True
        checkpoint_manager = CheckpointManager()

        def save_checkpoint(self, *args, **kwargs):
            events.append(("original", args, kwargs))

    engine_module.FSDPEngine = FSDPEngine
    modules = {
        compat._VERL_FSDP_ENGINE_MODULE: engine_module,
        "verl.utils.device": types.SimpleNamespace(get_device_name=lambda: "npu"),
    }
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(compat, "_NPU_CHECKPOINT_RECLAIM_APPLIED", False)
    monkeypatch.setattr(
        compat,
        "release_checkpoint_host_memory",
        lambda path, drop_file_cache: events.append(("reclaim", path, drop_file_cache))
        or {"elapsed_sec": 0.1, "files_advised": 0, "files_failed": 0},
    )
    monkeypatch.setattr(compat, "format_checkpoint_memory_snapshot", lambda: "memory")

    assert compat.install_verl_npu_checkpoint_reclaim(modules.__getitem__) is True
    FSDPEngine().save_checkpoint("/tmp/actor", global_step=20, max_ckpt_to_keep=1)

    assert events == [
        (
            "manager",
            {
                "local_path": "/tmp/actor",
                "hdfs_path": None,
                "global_step": 20,
                "max_ckpt_to_keep": 1,
            },
        ),
        ("barrier",),
        ("reclaim", "/tmp/actor", True),
    ]


def test_fsdp_training_output_release_preserves_forward_only(monkeypatch) -> None:
    engine_module = types.ModuleType(compat._VERL_FSDP_ENGINE_MODULE)

    class FSDPEngine:
        def forward_backward_batch(self):
            return None

    class FSDPEngineWithLMHead:
        def forward_step(self, micro_batch, loss_function, forward_only):
            del micro_batch, loss_function
            return "loss", {
                "model_output": {"log_probs": "tensor"},
                "forward_only": forward_only,
            }

    engine_module.FSDPEngine = FSDPEngine
    engine_module.FSDPEngineWithLMHead = FSDPEngineWithLMHead
    monkeypatch.setattr(compat, "_FSDP_TRAIN_OUTPUT_RELEASE_APPLIED", False)

    assert (
        compat.install_verl_fsdp_training_output_release_compat(lambda _: engine_module)
        is True
    )
    engine = FSDPEngineWithLMHead()

    _, training_output = engine.forward_step(None, None, False)
    _, inference_output = engine.forward_step(None, None, True)
    assert "model_output" not in training_output
    assert inference_output["model_output"] == {"log_probs": "tensor"}
