"""NPU compatibility for verl release/v0.8.0 vLLM imports and checkpoints."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import sys
import time
from typing import Any, Callable

from packaging import version

from verl_speco.trainer.checkpoint import (
    format_checkpoint_memory_snapshot,
    release_checkpoint_host_memory,
)

logger = logging.getLogger(__name__)

_VERL_NPU_VLLM_PATCH_MODULE = "verl.utils.vllm.npu_vllm_patch"
_VLLM_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe"
_VERL_FSDP_ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"
_IMPORT_COMPAT_APPLIED = False
_NPU_CHECKPOINT_RECLAIM_APPLIED = False


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


def install_verl_npu_checkpoint_reclaim(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Preserve verl's native actor save path and reclaim host memory after it."""

    global _NPU_CHECKPOINT_RECLAIM_APPLIED
    if _NPU_CHECKPOINT_RECLAIM_APPLIED or not _module_available("torch_npu"):
        return False

    device_module = module_importer("verl.utils.device")
    if device_module.get_device_name() != "npu":
        return False

    engine_module = module_importer(_VERL_FSDP_ENGINE_MODULE)
    engine_cls = getattr(engine_module, "FSDPEngine", None)
    if engine_cls is None:
        return False
    original_save_checkpoint = getattr(engine_cls, "save_checkpoint", None)
    if original_save_checkpoint is None or getattr(
        original_save_checkpoint,
        "_speco_npu_checkpoint_reclaim",
        False,
    ):
        return False

    @functools.wraps(original_save_checkpoint)
    def save_checkpoint_with_reclaim(self, local_path: str, *args, **kwargs):
        started = time.perf_counter()
        saved = False
        try:
            result = original_save_checkpoint(self, local_path, *args, **kwargs)
            saved = True
            return result
        finally:
            is_leader = int(getattr(self, "rank", 0) or 0) == 0
            reclaim = release_checkpoint_host_memory(
                local_path if saved else None,
                drop_file_cache=saved and is_leader,
            )
            if is_leader:
                logger.warning(
                    "[actor checkpoint] native save reclaim saved=%s total=%.2fs "
                    "reclaim=%.2fs files=%s failed=%s %s",
                    int(saved),
                    time.perf_counter() - started,
                    reclaim["elapsed_sec"],
                    reclaim["files_advised"],
                    reclaim["files_failed"],
                    format_checkpoint_memory_snapshot(),
                )

    save_checkpoint_with_reclaim._speco_npu_checkpoint_reclaim = True
    engine_cls.save_checkpoint = save_checkpoint_with_reclaim
    _NPU_CHECKPOINT_RECLAIM_APPLIED = True
    logger.warning("Enabled post-save NPU actor checkpoint host-memory reclaim")
    return True


class VerlNPUVLLMImportCompatMixin:
    """Install import compatibility when WorkerDict constructs the worker."""

    def __init__(self, *args, **kwargs):
        install_verl_npu_vllm_import_compat()
        install_verl_npu_checkpoint_reclaim()
        super().__init__(*args, **kwargs)
