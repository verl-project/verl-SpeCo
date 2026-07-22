"""NPU compatibility for verl release/v0.8.0 vLLM imports and checkpoints."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import time
from types import MethodType
from typing import Any, Callable

from packaging import version

from verl_speco.trainer.checkpoint import (
    format_checkpoint_memory_snapshot,
    log_previous_output_lifetime,
    remember_output_lifetime,
    release_checkpoint_host_memory,
    trim_process_host_memory,
)

logger = logging.getLogger(__name__)

_VERL_NPU_VLLM_PATCH_MODULE = "verl.utils.vllm.npu_vllm_patch"
_VLLM_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe"
_VERL_FSDP_ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"
_IMPORT_COMPAT_APPLIED = False
_NPU_CHECKPOINT_RECLAIM_APPLIED = False
_NPU_FSDP2_WEIGHT_EXPORT_APPLIED = False
_FSDP_TRAIN_OUTPUT_RELEASE_APPLIED = False
_NPU_FSDP_HOST_MEMORY_RECLAIM_APPLIED = False

try:
    from verl.single_controller.base.decorator import Dispatch, register
except Exception:  # noqa: BLE001
    Dispatch = None

    def register(*args, **kwargs):
        del args, kwargs

        def decorator(func):
            return func

        return decorator


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
    """Preserve native actor saves while fixing FSDP2 CPU-offload staging."""

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

    code = getattr(original_save_checkpoint, "__code__", None)
    has_fsdp2_cpu_offload_guard = bool(
        code is not None and "_uses_fsdp2_cpu_offload_policy" in code.co_names
    )

    @functools.wraps(original_save_checkpoint)
    def save_checkpoint_with_reclaim(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ):
        started = time.perf_counter()
        saved = False
        try:
            uses_fsdp2_cpu_offload = bool(
                getattr(self, "_uses_fsdp2_cpu_offload_policy", False)
            )
            if uses_fsdp2_cpu_offload and not has_fsdp2_cpu_offload_guard:
                if int(getattr(self, "rank", 0) or 0) == 0 and not getattr(
                    self,
                    "_speco_npu_fsdp2_checkpoint_guard_logged",
                    False,
                ):
                    logger.warning(
                        "[actor checkpoint] skipping manual model move under FSDP2 CPUOffloadPolicy"
                    )
                    self._speco_npu_fsdp2_checkpoint_guard_logged = True
                self.checkpoint_manager.save_checkpoint(
                    local_path=local_path,
                    hdfs_path=hdfs_path,
                    global_step=global_step,
                    max_ckpt_to_keep=max_ckpt_to_keep,
                )
                engine_module.torch.distributed.barrier()
                result = None
            else:
                result = original_save_checkpoint(
                    self,
                    local_path,
                    hdfs_path=hdfs_path,
                    global_step=global_step,
                    max_ckpt_to_keep=max_ckpt_to_keep,
                    **kwargs,
                )
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


def install_verl_fsdp_training_output_release_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Drop unused per-micro-batch model outputs during FSDP actor training.

    verl release/v0.8.0 retains every training micro-batch's full-length
    log-probability and entropy outputs until the mini-batch finishes. The
    training worker discards these outputs after the call, so retaining them
    only keeps tensors and their autograd graphs alive. This mirrors upstream
    fixes dbbf0853 and 78bba31d while preserving forward-only inference output.
    """

    global _FSDP_TRAIN_OUTPUT_RELEASE_APPLIED
    if _FSDP_TRAIN_OUTPUT_RELEASE_APPLIED:
        return False

    engine_module = module_importer(_VERL_FSDP_ENGINE_MODULE)
    engine_cls = getattr(engine_module, "FSDPEngine", None)
    lm_head_cls = getattr(engine_module, "FSDPEngineWithLMHead", None)
    if engine_cls is None or lm_head_cls is None:
        return False

    forward_backward_batch = getattr(engine_cls, "forward_backward_batch", None)
    try:
        forward_backward_source = inspect.getsource(forward_backward_batch)
    except (OSError, TypeError):
        forward_backward_source = ""
    upstream_releases_training_output = (
        "meta_info.pop" in forward_backward_source
        and (
            '"model_output"' in forward_backward_source
            or "'model_output'" in forward_backward_source
        )
    )
    if upstream_releases_training_output:
        _FSDP_TRAIN_OUTPUT_RELEASE_APPLIED = True
        return False

    original_forward_step = getattr(lm_head_cls, "forward_step", None)
    if original_forward_step is None or getattr(
        original_forward_step,
        "_speco_training_output_release_compat",
        False,
    ):
        return False

    @functools.wraps(original_forward_step)
    def forward_step_without_retained_training_output(self, *args, **kwargs):
        result = original_forward_step(self, *args, **kwargs)
        forward_only = kwargs.get("forward_only")
        if forward_only is None and len(args) >= 3:
            forward_only = args[2]
        if forward_only is False and isinstance(result, tuple) and len(result) == 2:
            meta_info = result[1]
            if isinstance(meta_info, dict):
                meta_info.pop("model_output", None)
        return result

    forward_step_without_retained_training_output._speco_training_output_release_compat = True
    lm_head_cls.forward_step = forward_step_without_retained_training_output
    _FSDP_TRAIN_OUTPUT_RELEASE_APPLIED = True
    logger.warning("Enabled FSDP actor training output-release compatibility")
    return True


def install_verl_npu_fsdp_host_memory_reclaim(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Trim completed Ray-call allocations at the next NPU FSDP call entry.

    Ray may still be serializing the previous ``forward_backward_batch`` result
    when that call returns. Trimming at the next entry avoids retaining glibc's
    previous high-water mark without touching live return values or forcing a
    Python GC cycle.
    """

    global _NPU_FSDP_HOST_MEMORY_RECLAIM_APPLIED
    if _NPU_FSDP_HOST_MEMORY_RECLAIM_APPLIED or not _module_available("torch_npu"):
        return False

    device_module = module_importer("verl.utils.device")
    if device_module.get_device_name() != "npu":
        return False

    engine_module = module_importer(_VERL_FSDP_ENGINE_MODULE)
    engine_cls = getattr(engine_module, "FSDPEngine", None)
    original_forward_backward_batch = getattr(engine_cls, "forward_backward_batch", None)
    if engine_cls is None or not callable(original_forward_backward_batch):
        return False
    if getattr(original_forward_backward_batch, "_speco_npu_entry_host_memory_reclaim", False):
        _NPU_FSDP_HOST_MEMORY_RECLAIM_APPLIED = True
        return False

    @functools.wraps(original_forward_backward_batch)
    def forward_backward_batch_with_entry_reclaim(self, *args, **kwargs):
        trim_process_host_memory()
        return original_forward_backward_batch(self, *args, **kwargs)

    forward_backward_batch_with_entry_reclaim._speco_npu_entry_host_memory_reclaim = True
    engine_cls.forward_backward_batch = forward_backward_batch_with_entry_reclaim
    _NPU_FSDP_HOST_MEMORY_RECLAIM_APPLIED = True
    return True


def install_verl_npu_fsdp2_weight_export_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Skip verl's redundant whole-shard staging during NPU FSDP2 export.

    verl release/v0.8.0 moves every local FSDP2 shard to the device before
    ``state_dict()`` and back to CPU afterwards. FSDP2 only returns DTensor
    references there, and the returned generator already materializes each
    full tensor on the device lazily. The extra round trip increases weight
    sync latency and can leave NPU host-memory allocations at their high-water
    mark. This mirrors upstream verl fix b7ff88e3 while preserving FSDP1 and
    PEFT/LoRA behavior.
    """

    global _NPU_FSDP2_WEIGHT_EXPORT_APPLIED
    if _NPU_FSDP2_WEIGHT_EXPORT_APPLIED or not _module_available("torch_npu"):
        return False

    device_module = module_importer("verl.utils.device")
    if device_module.get_device_name() != "npu":
        return False

    engine_module = module_importer(_VERL_FSDP_ENGINE_MODULE)
    engine_cls = getattr(engine_module, "FSDPEngine", None)
    fsdp_version = getattr(engine_module, "fsdp_version", None)
    if engine_cls is None or not callable(fsdp_version):
        return False

    original_get_per_tensor_param = getattr(engine_cls, "get_per_tensor_param", None)
    if original_get_per_tensor_param is None or getattr(
        original_get_per_tensor_param,
        "_speco_npu_fsdp2_weight_export_compat",
        False,
    ):
        return False

    # Newer verl versions contain the upstream fix already.
    code = getattr(original_get_per_tensor_param, "__code__", None)
    if code is not None and "_skip_staging" in code.co_varnames:
        _NPU_FSDP2_WEIGHT_EXPORT_APPLIED = True
        return False

    @functools.wraps(original_get_per_tensor_param)
    def get_per_tensor_param_without_fsdp2_staging(self, *args, **kwargs):
        module = getattr(self, "module", None)
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        try:
            skip_staging = module is not None and fsdp_version(module) == 2 and not hasattr(
                peft_model,
                "peft_config",
            )
        except Exception:  # noqa: BLE001
            skip_staging = False

        if not skip_staging:
            return original_get_per_tensor_param(self, *args, **kwargs)

        uses_cpu_offload_policy = getattr(self, "_uses_fsdp2_cpu_offload_policy", False)
        is_offload_param = getattr(self, "_is_offload_param", False)
        self._uses_fsdp2_cpu_offload_policy = True
        self._is_offload_param = False
        try:
            if int(getattr(self, "rank", 0) or 0) == 0 and not getattr(
                self,
                "_speco_npu_fsdp2_weight_export_logged",
                False,
            ):
                logger.warning(
                    "[speco weight export] skipping redundant whole-shard staging for NPU FSDP2"
                )
                self._speco_npu_fsdp2_weight_export_logged = True
            return original_get_per_tensor_param(self, *args, **kwargs)
        finally:
            self._uses_fsdp2_cpu_offload_policy = uses_cpu_offload_policy
            self._is_offload_param = is_offload_param

    get_per_tensor_param_without_fsdp2_staging._speco_npu_fsdp2_weight_export_compat = True
    engine_cls.get_per_tensor_param = get_per_tensor_param_without_fsdp2_staging
    _NPU_FSDP2_WEIGHT_EXPORT_APPLIED = True
    logger.warning("Enabled NPU FSDP2 weight-export staging compatibility")
    return True


def _install_weight_transfer_shm_reuse() -> bool:
    """Install the sender-side SHM reuse patch in the WorkerDict process."""

    try:
        from verl_speco.integration.vllm_runtime import patch_verl_bucketed_weight_transfer_shm_reuse
    except Exception:  # noqa: BLE001
        return False
    return patch_verl_bucketed_weight_transfer_shm_reuse()


def _is_npu_worker() -> bool:
    if not _module_available("torch_npu"):
        return False
    try:
        device_module = importlib.import_module("verl.utils.device")
        return device_module.get_device_name() == "npu"
    except Exception:  # noqa: BLE001
        return False


def _install_npu_worker_output_lifetime_diagnostics(worker: Any) -> bool:
    if not _is_npu_worker() or int(getattr(worker, "rank", 0) or 0) != 0:
        return False
    if getattr(worker, "_speco_output_lifetime_diagnostics_installed", False):
        return False

    installed = []
    for method_name in ("compute_log_prob", "compute_ref_log_prob", "update_actor"):
        original = getattr(worker, method_name, None)
        if not callable(original) or inspect.iscoroutinefunction(original):
            continue

        @functools.wraps(original)
        def output_lifetime_wrapper(
            bound_worker,
            *args,
            _original=original,
            _method_name=method_name,
            **kwargs,
        ):
            call_index = log_previous_output_lifetime(
                bound_worker,
                f"worker_dict:{_method_name}",
                role="worker_dict",
                method=_method_name,
            )
            result = _original(*args, **kwargs)
            remember_output_lifetime(
                bound_worker,
                f"worker_dict:{_method_name}",
                call_index,
                result,
            )
            return result

        setattr(worker, method_name, MethodType(output_lifetime_wrapper, worker))
        installed.append(method_name)

    if not installed:
        return False
    worker._speco_output_lifetime_diagnostics_installed = True
    print(
        "[speco output lifetime] WorkerDict diagnostics installed "
        f"pid={os.getpid()} rank={getattr(worker, 'rank', None)} "
        f"methods={','.join(installed)}",
        flush=True,
    )
    return True


class VerlNPUVLLMImportCompatMixin:
    """Install import compatibility when WorkerDict constructs the worker."""

    def __init__(self, *args, **kwargs):
        install_verl_npu_vllm_import_compat()
        install_verl_fsdp_training_output_release_compat()
        install_verl_npu_fsdp_host_memory_reclaim()
        install_verl_npu_checkpoint_reclaim()
        install_verl_npu_fsdp2_weight_export_compat()
        super().__init__(*args, **kwargs)
        _install_npu_worker_output_lifetime_diagnostics(self)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None), blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # Both baseline and speculative runs send actor weights from this
        # WorkerDict process. Install immediately before the upstream sender is
        # constructed so no-drafter runs receive the same NPU SHM protection.
        _install_weight_transfer_shm_reuse()
        if not _is_npu_worker():
            return await super().update_weights(global_steps=global_steps, mode=mode)

        try:
            return await super().update_weights(global_steps=global_steps, mode=mode)
        finally:
            trim_process_host_memory()
