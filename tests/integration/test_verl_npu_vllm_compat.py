from __future__ import annotations

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
    monkeypatch.setattr(compat, "install_verl_npu_vllm_import_compat", lambda: events.append("compat"))
    monkeypatch.setattr(
        compat,
        "install_verl_npu_checkpoint_reclaim",
        lambda: events.append("reclaim"),
    )

    class BaseWorker:
        def __init__(self):
            events.append("base")

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    WrappedWorker()
    assert events == ["compat", "reclaim", "base"]


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

    assert engine.save_checkpoint("/tmp/actor", global_step=20, max_ckpt_to_keep=1) == "saved"
    assert events == [
        ("original", ("/tmp/actor",), {"global_step": 20, "max_ckpt_to_keep": 1}),
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
