"""Compatibility for verl release/v0.8.0 with the vLLM FusedMoE factory API."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from typing import Any, Callable

from packaging import version

logger = logging.getLogger(__name__)

_VERL_NPU_VLLM_PATCH_MODULE = "verl.utils.vllm.npu_vllm_patch"
_VLLM_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe"
_IMPORT_COMPAT_APPLIED = False


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _unused_factory_weight_loader(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("FusedMoE factory compatibility weight_loader must never be called")


def install_verl_npu_vllm_import_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Import verl's NPU patch without applying its obsolete class-only MoE hook.

    vLLM >= 0.18 exposes ``FusedMoE`` as a factory function. The verl v0.8.0
    patch still accesses ``FusedMoE.weight_loader`` during import. A temporary
    attribute lets the rest of verl's NPU initialization run; it is removed
    immediately because factory instances use their own runner weight loaders.
    """

    global _IMPORT_COMPAT_APPLIED
    if _IMPORT_COMPAT_APPLIED or _VERL_NPU_VLLM_PATCH_MODULE in sys.modules:
        return False
    # Match verl's own guard: its failing import path is enabled by torch_npu,
    # even before vllm_ascend itself has necessarily been imported.
    if not _module_available("torch_npu"):
        return False

    vllm = module_importer("vllm")
    if version.parse(str(getattr(vllm, "__version__", "0"))) < version.parse("0.18.0"):
        return False

    fused_moe_module = module_importer(_VLLM_FUSED_MOE_MODULE)
    fused_moe = getattr(fused_moe_module, "FusedMoE", None)
    if fused_moe is None or isinstance(fused_moe, type) or hasattr(fused_moe, "weight_loader"):
        return False

    fused_moe.weight_loader = _unused_factory_weight_loader
    try:
        module_importer(_VERL_NPU_VLLM_PATCH_MODULE)
    finally:
        if hasattr(fused_moe, "weight_loader"):
            delattr(fused_moe, "weight_loader")

    _IMPORT_COMPAT_APPLIED = True
    logger.warning(
        "Applied verl release/v0.8.0 NPU import compatibility for the vLLM FusedMoE factory"
    )
    return True


class VerlNPUVLLMImportCompatMixin:
    """Install import compatibility when WorkerDict constructs the worker."""

    def __init__(self, *args, **kwargs):
        install_verl_npu_vllm_import_compat()
        super().__init__(*args, **kwargs)
