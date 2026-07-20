from __future__ import annotations

import importlib
import sys
import types

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

    class BaseWorker:
        def __init__(self):
            events.append("base")

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    WrappedWorker()
    assert events == ["compat", "base"]
