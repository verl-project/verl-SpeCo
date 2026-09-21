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
"""NPU compatibility for verl 0.8/0.9 vLLM imports and checkpoints."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from packaging import version

from verl_speco.trainer.checkpoint import (
    format_checkpoint_memory_snapshot,
    release_checkpoint_host_memory,
    trim_process_host_memory,
)

logger = logging.getLogger(__name__)

_VERL_NPU_VLLM_PATCH_MODULE = "verl.utils.vllm.npu_vllm_patch"
_VLLM_FUSED_MOE_PACKAGE = "vllm.model_executor.layers.fused_moe"
_VLLM_FUSED_MOE_LAYER_MODULE = "vllm.model_executor.layers.fused_moe.layer"
# Backward-compatible name used by the release/v0.8.0 import path and tests.
_VLLM_FUSED_MOE_MODULE = _VLLM_FUSED_MOE_PACKAGE
_VERL_FSDP_ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"
_IMPORT_COMPAT_APPLIED = False
_NPU_CHECKPOINT_RECLAIM_APPLIED = False
_NPU_FSDP2_WEIGHT_EXPORT_APPLIED = False
_FSDP_TRAIN_OUTPUT_RELEASE_APPLIED = False

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


def _unavailable_fused_moe(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "FusedMoE is unavailable in this vLLM build; the temporary symbol only "
        "allows verl release/v0.9.0 to skip its inapplicable legacy NPU patch"
    )


def _unused_factory_weight_loader(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "FusedMoE factory compatibility weight_loader must never be called"
    )


def _uses_verl_v090_runner() -> bool:
    """Identify the moved legacy runner without importing accelerator modules."""

    return _module_available("verl.trainer.main_ppo_v0")


def _install_verl_v080_npu_vllm_import_compat(
    module_importer: Callable[[str], Any],
) -> bool:
    """Import verl 0.8's NPU patch around its obsolete factory attribute."""

    vllm = module_importer("vllm")
    if version.parse(str(getattr(vllm, "__version__", "0"))) < version.parse("0.18.0"):
        return False

    fused_moe_module = module_importer(_VLLM_FUSED_MOE_MODULE)
    fused_moe = getattr(fused_moe_module, "FusedMoE", None)
    if (
        fused_moe is None
        or isinstance(fused_moe, type)
        or hasattr(fused_moe, "weight_loader")
    ):
        return False

    fused_moe.weight_loader = _unused_factory_weight_loader
    try:
        module_importer(_VERL_NPU_VLLM_PATCH_MODULE)
    finally:
        if getattr(fused_moe, "weight_loader", None) is _unused_factory_weight_loader:
            del fused_moe.weight_loader
    return True


@contextmanager
def _temporary_verl_v090_fused_moe_import(
    module_importer: Callable[[str], Any],
) -> Iterator[None]:
    """Provide only the temporary package export expected by verl's NPU patch.

    Some modular vLLM revisions keep ``FusedMoE`` in ``fused_moe.layer``
    without re-exporting it; newer revisions remove that factory entirely.
    verl v0.9 imports the package-level symbol before it can skip the obsolete
    class-level hook. Export the exact factory or a non-class sentinel only for
    that import, then restore the package namespace even when import fails.
    """

    fused_moe_package = module_importer(_VLLM_FUSED_MOE_PACKAGE)
    if hasattr(fused_moe_package, "FusedMoE"):
        yield
        return

    try:
        fused_moe_layer = module_importer(_VLLM_FUSED_MOE_LAYER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _VLLM_FUSED_MOE_LAYER_MODULE:
            raise
        fused_moe_layer = None
    fused_moe = getattr(fused_moe_layer, "FusedMoE", None)
    if fused_moe is None:
        fused_moe = _unavailable_fused_moe

    fused_moe_package.FusedMoE = fused_moe
    try:
        yield
    finally:
        if getattr(fused_moe_package, "FusedMoE", None) is fused_moe:
            del fused_moe_package.FusedMoE


def install_verl_npu_vllm_import_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Eagerly import the installed release's NPU vLLM initialization safely."""

    global _IMPORT_COMPAT_APPLIED
    if _IMPORT_COMPAT_APPLIED or _VERL_NPU_VLLM_PATCH_MODULE in sys.modules:
        return False
    if not _module_available("torch_npu"):
        return False

    if _uses_verl_v090_runner():
        with _temporary_verl_v090_fused_moe_import(module_importer):
            module_importer(_VERL_NPU_VLLM_PATCH_MODULE)
    elif not _install_verl_v080_npu_vllm_import_compat(module_importer):
        return False
    _IMPORT_COMPAT_APPLIED = True
    logger.warning(
        "Applied verl NPU vLLM import compatibility for legacy runner API %s",
        "0.9" if _uses_verl_v090_runner() else "0.8",
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

    setattr(save_checkpoint_with_reclaim, "_speco_npu_checkpoint_reclaim", True)
    engine_cls.save_checkpoint = save_checkpoint_with_reclaim
    _NPU_CHECKPOINT_RECLAIM_APPLIED = True
    logger.warning("Enabled post-save NPU actor checkpoint host-memory reclaim")
    return True


def install_verl_fsdp_training_output_release_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Drop unused per-micro-batch model outputs during FSDP actor training.

    The legacy trainer can retain every training micro-batch's full-length
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
        forward_backward_source = (
            inspect.getsource(forward_backward_batch)
            if forward_backward_batch is not None
            else ""
        )
    except (OSError, TypeError):
        forward_backward_source = ""
    upstream_releases_training_output = "meta_info.pop" in forward_backward_source and (
        '"model_output"' in forward_backward_source
        or "'model_output'" in forward_backward_source
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

    setattr(
        forward_step_without_retained_training_output,
        "_speco_training_output_release_compat",
        True,
    )
    lm_head_cls.forward_step = forward_step_without_retained_training_output
    _FSDP_TRAIN_OUTPUT_RELEASE_APPLIED = True
    logger.warning("Enabled FSDP actor training output-release compatibility")
    return True


def install_verl_npu_fsdp2_weight_export_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Skip verl's redundant whole-shard staging during NPU FSDP2 export.

    The legacy trainer moves every local FSDP2 shard to the device before
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
            skip_staging = (
                module is not None
                and fsdp_version(module) == 2
                and not hasattr(
                    peft_model,
                    "peft_config",
                )
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

    setattr(
        get_per_tensor_param_without_fsdp2_staging,
        "_speco_npu_fsdp2_weight_export_compat",
        True,
    )
    engine_cls.get_per_tensor_param = get_per_tensor_param_without_fsdp2_staging
    _NPU_FSDP2_WEIGHT_EXPORT_APPLIED = True
    logger.warning("Enabled NPU FSDP2 weight-export staging compatibility")
    return True


def _install_weight_transfer_shm_reuse() -> bool:
    """Install the sender-side SHM reuse patch in the WorkerDict process."""

    try:
        from verl_speco.integration.vllm_runtime import (
            patch_verl_bucketed_weight_transfer_shm_reuse,
        )
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


class VerlNPUVLLMImportCompatMixin:
    """Install import compatibility when WorkerDict constructs the worker."""

    def __init__(self, *args, **kwargs):
        from verl_speco.integration.compat import check_compatible_verl

        check_compatible_verl()
        install_verl_npu_vllm_import_compat()
        install_verl_fsdp_training_output_release_compat()
        install_verl_npu_checkpoint_reclaim()
        install_verl_npu_fsdp2_weight_export_compat()
        super().__init__(*args, **kwargs)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None), blocking=False)
    async def update_weights(self, global_steps: int | None = None, mode: str = "auto"):
        # Both baseline and speculative runs send actor weights from this
        # WorkerDict process. Install immediately before the upstream sender is
        # constructed so no-drafter runs receive the same NPU SHM protection.
        _install_weight_transfer_shm_reuse()
        if not _is_npu_worker():
            return await cast(Any, super()).update_weights(
                global_steps=global_steps, mode=mode
            )

        try:
            return await cast(Any, super()).update_weights(
                global_steps=global_steps, mode=mode
            )
        finally:
            trim_process_host_memory()
