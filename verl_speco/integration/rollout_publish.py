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
"""Draft-weight publishing helpers for SPECO rollout adapters."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, cast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl.single_controller.base.decorator import Dispatch, register
except Exception:  # noqa: BLE001
    Dispatch = None

    def register(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def _ray_module():
    import ray

    return ray


def _torch_module():
    import torch

    return torch


def _get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def rollout_backend_name(config: Any) -> Optional[str]:
    return _get_nested(config, ("rollout", "name"), None) or _get_nested(
        config, ("actor_rollout_ref", "rollout", "name"), None
    )


def actor_training_backend_name(config: Any) -> str:
    value = _get_nested(config, ("actor", "strategy"), None)
    if value is None:
        value = _get_nested(config, ("actor_rollout_ref", "actor", "strategy"), None)
    if value is None:
        value = _get_nested(config, ("model_engine",), None)
    return str(value or "fsdp").strip().lower()


def veomni_parallel_layout(config: Any) -> dict[str, Any]:
    actor_config = _get_nested(config, ("actor",), None)
    if actor_config is None:
        actor_config = _get_nested(config, ("actor_rollout_ref", "actor"), {})
    veomni_config = _get_nested(actor_config, ("veomni",), {}) or {}
    rollout_config = _get_nested(config, ("rollout",), None)
    if rollout_config is None:
        rollout_config = _get_nested(config, ("actor_rollout_ref", "rollout"), {})
    return {
        "ulysses_parallel_size": int(
            _get_nested(veomni_config, ("ulysses_parallel_size",), 1) or 1
        ),
        "expert_parallel_size": int(
            _get_nested(veomni_config, ("expert_parallel_size",), 1) or 1
        ),
        "router_replay_mode": str(
            _get_nested(veomni_config, ("router_replay", "mode"), "disabled")
            or "disabled"
        ).upper(),
        "rollout_routing_replay": bool(
            _get_nested(rollout_config, ("enable_rollout_routing_replay",), False)
        ),
    }


def validate_veomni_parallel_layout(config: Any) -> dict[str, Any]:
    layout = veomni_parallel_layout(config)
    router_mode = layout["router_replay_mode"]
    rollout_replay = layout["rollout_routing_replay"]
    if router_mode not in {"DISABLED", "R2", "R3"}:
        raise ValueError(
            "VeOmni router_replay.mode must be disabled, R2, or R3, got "
            f"{router_mode!r}"
        )
    if router_mode == "R3" and not rollout_replay:
        raise RuntimeError(
            "VeOmni router_replay.mode=R3 requires "
            "actor_rollout_ref.rollout.enable_rollout_routing_replay=True"
        )
    if router_mode == "R2" and rollout_replay:
        raise RuntimeError(
            "VeOmni router_replay.mode=R2 must not enable rollout routing replay"
        )
    return layout


def materialize_draft_weights_payload(weights: Any) -> tuple[Any, bool]:
    """Resolve direct tensor payloads or Ray ObjectRef-backed payloads."""

    try:
        ray = _ray_module()
        object_ref_type = ray.ObjectRef
    except Exception:  # noqa: BLE001
        ray = None
        object_ref_type = ()

    if isinstance(weights, dict) and "weights_ref" in weights:
        weights_ref = weights["weights_ref"]
        if object_ref_type and isinstance(weights_ref, object_ref_type):
            return ray.get(weights_ref), True
        return weights_ref, True
    if object_ref_type and isinstance(weights, object_ref_type):
        return ray.get(weights), True
    return weights, False


def resolve_drafter_publish_payload(published_payload: Any) -> Any:
    """Normalize direct or Ray-ref draft publish payload."""

    if isinstance(published_payload, dict) and "weights_ref" in published_payload:
        return published_payload

    return published_payload


def drafter_rollout_enabled(config: Any) -> bool:
    if bool(_get_nested(config, ("rollout", "drafter", "enable"), False)):
        return True
    if bool(
        _get_nested(
            config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False
        )
    ):
        return True
    try:
        from verl_speco.integration.sglang_runtime import _load_env_drafter_config

        return bool(_load_env_drafter_config().get("enable"))
    except Exception:  # noqa: BLE001
        return False


def drafter_speculative_algorithm(config: Any) -> str:
    value = _get_nested(config, ("rollout", "drafter", "speculative_algorithm"), None)
    if value is None:
        value = _get_nested(
            config,
            ("actor_rollout_ref", "rollout", "drafter", "speculative_algorithm"),
            None,
        )
    return str(value or "").upper()


def install_sglang_runtime_for_worker(worker: Any) -> None:
    """Install SPECO SGLang runtime hooks inside an actor-rollout worker process."""

    try:
        from verl_speco.integration.sglang_runtime import (
            SPECO_SGLANG_DRAFTER_CONFIG_ENV,
            patch_sglang_server_adapter_update,
        )
    except Exception:  # noqa: BLE001
        return

    drafter_env = getattr(type(worker), "_speco_sglang_drafter_config_env", None)
    if drafter_env:
        os.environ[SPECO_SGLANG_DRAFTER_CONFIG_ENV] = drafter_env
    if os.getenv(SPECO_SGLANG_DRAFTER_CONFIG_ENV):
        patch_sglang_server_adapter_update()


def install_vllm_runtime_for_worker(worker: Any) -> None:
    """Install SPECO vLLM runtime hooks inside an actor-rollout worker process."""

    try:
        from verl_speco.integration.vllm_runtime import (
            install_vllm_runtime_for_worker as _install,
        )
    except Exception:  # noqa: BLE001
        return

    _install(worker)


def install_rollout_runtime_for_worker(worker: Any) -> None:
    backend = rollout_backend_name(getattr(worker, "config", None))
    if backend == "vllm":
        install_vllm_runtime_for_worker(worker)
        return
    install_sglang_runtime_for_worker(worker)


def install_oldlogprob_hidden_runtime_for_worker(worker: Any) -> None:
    """Install old-logprob hidden collection hooks inside an actor worker process."""

    if getattr(worker, "_is_actor", True) is False:
        return

    try:
        from verl_speco.integration.oldlogprob_runtime import (
            install_oldlogprob_hidden_runtime_patch,
            install_oldlogprob_hidden_runtime_patch_megatron,
            oldlogprob_hidden_runtime_enabled,
        )
    except Exception:  # noqa: BLE001
        return

    drafter_env = (
        getattr(type(worker), "_speco_sglang_drafter_config_env", None) or None
    )
    if not oldlogprob_hidden_runtime_enabled(
        getattr(worker, "config", None), drafter_env=drafter_env
    ):
        return
    actor_backend = actor_training_backend_name(getattr(worker, "config", None))
    if actor_backend != "veomni":
        install_oldlogprob_hidden_runtime_patch()
        install_oldlogprob_hidden_runtime_patch_megatron()
        return

    patched = install_oldlogprob_hidden_runtime_patch(actor_backend="veomni")
    if not patched:
        raise RuntimeError(
            "SPECO could not install VeOmni old-logprob hidden collection. "
            "Verify the installed verl and VeOmni versions before enabling drafter co-training."
        )
    validate_veomni_parallel_layout(getattr(worker, "config", None))


def validate_oldlogprob_hidden_runtime_for_worker(worker: Any) -> None:
    """Validate VeOmni's initialized model contract before the first rollout."""

    if getattr(worker, "_is_actor", True) is False:
        return

    config = getattr(worker, "config", None)
    if actor_training_backend_name(config) != "veomni":
        return

    try:
        from verl_speco.integration.oldlogprob_runtime import (
            _find_layers_and_final_norm,
            oldlogprob_hidden_runtime_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SPECO could not import the VeOmni hidden-state validator"
        ) from exc

    drafter_env = (
        getattr(type(worker), "_speco_sglang_drafter_config_env", None) or None
    )
    if not oldlogprob_hidden_runtime_enabled(config, drafter_env=drafter_env):
        return

    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None) if actor is not None else None
    module = getattr(engine, "module", None) if engine is not None else None
    if engine is None or module is None:
        raise RuntimeError(
            "SPECO VeOmni validation requires an initialized actor engine"
        )

    layers, final_norm = _find_layers_and_final_norm(engine)
    if not layers or final_norm is None:
        raise RuntimeError(
            "SPECO could not locate VeOmni transformer layers and final norm; "
            "the installed VeOmni model layout is not supported"
        )
    lm_head_name, lm_head_weight = _select_lm_head_named_tensor(module)
    if lm_head_weight is None:
        raise RuntimeError(
            "SPECO could not locate VeOmni lm_head.weight or tied embed_tokens.weight"
        )

    layout = veomni_parallel_layout(config)
    if getattr(worker, "rank", None) == 0:
        logger.warning(
            "[speco actor backend] strategy=veomni hidden_capture=forward_hook "
            "lm_head_export=veomni_lm_head_only drafter_backend=fsdp2 "
            "layers=%s lm_head=%s sp=%s ep=%s router_replay=%s",
            len(layers),
            lm_head_name,
            layout["ulysses_parallel_size"],
            layout["expert_parallel_size"],
            layout["router_replay_mode"],
        )


def _normalize_lm_head_row_indices(row_indices: Any, *, device: Any = None):
    if row_indices is None:
        return None
    torch = _torch_module()
    if torch.is_tensor(row_indices):
        rows = row_indices.detach().to(dtype=torch.long).reshape(-1)
        return rows.to(device=device) if device is not None else rows
    if isinstance(row_indices, (list, tuple)):
        rows = torch.tensor([int(idx) for idx in row_indices], dtype=torch.long)
        return rows.to(device=device) if device is not None else rows
    return None


def _actor_module_candidates(worker: Any) -> list[Any]:
    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None) if actor is not None else None
    candidates = []
    for root in (actor, engine, worker):
        if root is None:
            continue
        candidates.append(root)
        for attr in (
            "module",
            "_module",
            "model",
            "_model",
            "actor_module",
            "actor_module_fsdp",
            "fsdp_module",
            "module_fsdp",
            "model_module",
            "transformer",
            "_orig_module",
            "_fsdp_wrapped_module",
            "_fully_sharded_module",
            "_checkpoint_wrapped_module",
            "_wrapped_module",
            "wrapped_module",
            "_forward_module",
        ):
            value = getattr(root, attr, None)
            if value is not None:
                candidates.append(value)

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        for attr in (
            "module",
            "_module",
            "model",
            "_model",
            "_orig_module",
            "_fsdp_wrapped_module",
            "_fully_sharded_module",
            "_checkpoint_wrapped_module",
            "_wrapped_module",
            "wrapped_module",
            "_forward_module",
            "lm_head",
        ):
            value = getattr(candidate, attr, None)
            if value is not None:
                expanded.append(value)

    # Preserve order while removing duplicate object identities.
    deduped = []
    seen = set()
    for candidate in expanded:
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(candidate)
    return deduped


def _select_lm_head_named_tensor(module: Any) -> tuple[str | None, Any | None]:
    torch = _torch_module()
    direct_weight = getattr(module, "weight", None)
    if torch.is_tensor(direct_weight):
        direct_weight = cast(Any, direct_weight)
        if direct_weight.dim() == 2:
            return "lm_head.weight", direct_weight

    for attr_name in ("lm_head", "embed_tokens"):
        child = getattr(module, attr_name, None)
        child_weight = getattr(child, "weight", None)
        if torch.is_tensor(child_weight):
            child_weight = cast(Any, child_weight)
            if child_weight.dim() == 2:
                return f"{attr_name}.weight", child_weight

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return None, None

    fallback: tuple[str | None, Any | None] = (None, None)
    try:
        iterator = named_parameters(recurse=True)
    except TypeError:
        iterator = named_parameters()
    except Exception:  # noqa: BLE001
        return None, None

    try:
        for name, tensor in iterator:
            if not torch.is_tensor(tensor) or tensor.dim() != 2:
                continue
            name = str(name)
            if name == "model.embed_tokens.weight" or name.endswith(
                ".embed_tokens.weight"
            ):
                fallback = (name, tensor)
            if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
                return name, tensor
    except Exception:  # noqa: BLE001
        return None, None
    return fallback


def _export_actor_lm_head_rows_direct(worker: Any, row_indices: Any) -> Optional[dict]:
    """Best-effort fast path for sparse target lm_head export.

    This avoids ``engine.get_per_tensor_param()``, which may materialize or
    enumerate the full actor parameter set before slicing.  If the current verl
    worker layout does not expose a full 2D lm_head/embedding tensor directly,
    callers fall back to the original engine path.
    """
    torch = _torch_module()
    row_indices_cpu = _normalize_lm_head_row_indices(row_indices)
    if row_indices_cpu is None or int(row_indices_cpu.numel()) <= 0:
        return None

    for module in _actor_module_candidates(worker):
        selected_name, selected_weight = _select_lm_head_named_tensor(module)
        if selected_weight is None:
            continue
        try:
            source_vocab_size = int(selected_weight.shape[0])
            if (
                int(row_indices_cpu.max().item()) >= source_vocab_size
                or int(row_indices_cpu.min().item()) < 0
            ):
                continue
            rows_on_device = row_indices_cpu.to(
                device=selected_weight.device, dtype=torch.long
            )
            selected_rows = selected_weight.detach().index_select(0, rows_on_device)
            if getattr(worker, "rank", None) != 0:
                return {"_speco_non_owner_direct_sparse": True}
            weight = selected_rows.to(device="cpu", dtype=torch.bfloat16).contiguous()
            logger.warning(
                "[actor lm_head export] direct_sparse name=%s shape=%s source_vocab=%s selected_rows=%s",
                selected_name,
                tuple(weight.shape),
                source_vocab_size,
                int(row_indices_cpu.numel()),
            )
            return {
                "name": selected_name,
                "weight": weight,
                "row_indices": row_indices_cpu.to(
                    device="cpu", dtype=torch.long
                ).contiguous(),
                "source_vocab_size": source_vocab_size,
                "selected_rows": int(row_indices_cpu.numel()),
                "export_strategy": "direct_sparse",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Direct sparse lm_head export failed for %s: %s", selected_name, exc
            )
            continue
    return None


def _is_veomni_actor_worker(worker: Any) -> bool:
    return actor_training_backend_name(getattr(worker, "config", None)) == "veomni"


def _materialize_veomni_lm_head_rows(selected_weight: Any, row_indices: Any):
    """Collect selected vocab rows without replicating the full DTensor."""

    torch = _torch_module()
    if not callable(getattr(selected_weight, "to_local", None)):
        rows_on_device = row_indices.to(device=selected_weight.device, dtype=torch.long)
        return selected_weight.detach().index_select(0, rows_on_device)

    try:
        import torch.distributed as dist
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "the installed PyTorch DTensor build does not expose shard offsets"
        ) from exc

    placements = tuple(getattr(selected_weight, "placements", ()))
    device_mesh = getattr(selected_weight, "device_mesh", None)
    if device_mesh is None or not placements:
        raise NotImplementedError("missing DTensor placements or device mesh")

    sharded_mesh_dims = []
    for mesh_dim, placement in enumerate(placements):
        placement_name = type(placement).__name__
        if placement_name == "Shard":
            if int(getattr(placement, "dim", -1)) != 0:
                raise NotImplementedError(
                    "lm_head DTensor is sharded on a non-vocab dimension"
                )
            sharded_mesh_dims.append(mesh_dim)
        elif placement_name != "Replicate":
            raise NotImplementedError(
                f"unsupported lm_head DTensor placement {placement_name}"
            )

    local_weight = selected_weight.to_local().detach()
    _local_shape, global_offset = compute_local_shape_and_global_offset(
        tuple(selected_weight.shape), device_mesh, placements
    )
    row_start = int(global_offset[0])
    row_end = row_start + int(local_weight.shape[0])
    global_rows = row_indices.to(device=local_weight.device, dtype=torch.long)
    local_mask = (global_rows >= row_start) & (global_rows < row_end)
    selected_rows = local_weight.new_zeros(
        (int(global_rows.numel()), int(local_weight.shape[1]))
    )
    if bool(local_mask.any().item()):
        output_positions = torch.nonzero(local_mask, as_tuple=False).flatten()
        local_rows = global_rows.index_select(0, output_positions) - row_start
        selected_rows.index_copy_(
            0,
            output_positions,
            local_weight.index_select(0, local_rows),
        )

    for mesh_dim in sharded_mesh_dims:
        if int(device_mesh.size(mesh_dim)) > 1:
            dist.all_reduce(selected_rows, group=device_mesh.get_group(mesh_dim))
    return selected_rows


def _export_veomni_actor_lm_head_weight(
    worker: Any,
    row_indices: Any = None,
    keep_model_on_device: bool = False,
) -> Optional[dict]:
    """Export only VeOmni's lm_head DTensor instead of its full state dict."""

    torch = _torch_module()
    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None) if actor is not None else None
    module = getattr(engine, "module", None) if engine is not None else None
    if engine is None or module is None:
        raise RuntimeError(
            "SPECO VeOmni lm_head export requires an initialized actor engine"
        )

    offload_model = None
    actor_device_type = None
    materialized_weight = None
    npu_lm_head_export = False
    reclaim_npu_staging = False
    restore_cpu_after_export = bool(getattr(engine, "_is_offload_param", False))
    keep_model_on_device_after_export = False
    if restore_cpu_after_export:
        try:
            from verl.workers.engine.veomni.utils import (
                load_veomni_model_to_gpu,
                offload_veomni_model_to_cpu,
            )

            offload_model = offload_veomni_model_to_cpu
            load_veomni_model_to_gpu(module)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SPECO failed to activate the VeOmni actor for lm_head-only export"
            ) from exc

    try:
        selected_name, selected_weight = _select_lm_head_named_tensor(module)
        if selected_weight is None:
            raise RuntimeError(
                "SPECO could not find VeOmni lm_head.weight or tied embed_tokens.weight"
            )

        if not torch.is_tensor(selected_weight) or selected_weight.dim() != 2:
            raise RuntimeError("SPECO VeOmni lm_head-only export expected a 2D tensor")

        source_vocab_size = int(selected_weight.shape[0])
        actor_device_type = str(selected_weight.device.type)
        npu_lm_head_export = actor_device_type == "npu"
        normalized_rows = _normalize_lm_head_row_indices(row_indices)
        exported_rows = None
        selected_rows = None
        export_strategy = "veomni_lm_head_full"
        if normalized_rows is not None and int(normalized_rows.numel()) > 0:
            min_row = int(normalized_rows.min().item())
            max_row = int(normalized_rows.max().item())
            if min_row < 0 or max_row >= source_vocab_size:
                raise ValueError(
                    "SPECO VeOmni lm_head row selection is outside the source vocabulary: "
                    f"min={min_row}, max={max_row}, vocab={source_vocab_size}"
                )
            if int(normalized_rows.numel()) < source_vocab_size:
                try:
                    materialized_weight = _materialize_veomni_lm_head_rows(
                        selected_weight,
                        normalized_rows,
                    )
                    if int(materialized_weight.shape[0]) != int(
                        normalized_rows.numel()
                    ):
                        raise RuntimeError(
                            "VeOmni sparse lm_head export returned an unexpected row count"
                        )
                    export_strategy = "veomni_lm_head_sparse"
                except (NotImplementedError, RuntimeError, TypeError) as exc:
                    logger.warning(
                        "VeOmni DTensor row-selective lm_head export is unavailable; "
                        "falling back to full lm_head materialization: %s",
                        exc,
                    )
                exported_rows = normalized_rows.to(
                    device="cpu", dtype=torch.long
                ).contiguous()
                selected_rows = int(normalized_rows.numel())

        if materialized_weight is None:
            full_tensor = getattr(selected_weight, "full_tensor", None)
            materialized_weight = (
                full_tensor() if callable(full_tensor) else selected_weight.detach()
            )
            if exported_rows is not None:
                rows_on_device = normalized_rows.to(
                    device=materialized_weight.device, dtype=torch.long
                )
                materialized_weight = materialized_weight.index_select(
                    0, rows_on_device
                )

        if not torch.is_tensor(materialized_weight) or materialized_weight.dim() != 2:
            raise RuntimeError(
                "SPECO VeOmni lm_head-only export expected a materialized 2D tensor, "
                f"got {type(materialized_weight).__name__}"
            )
        reclaim_npu_staging = bool(
            npu_lm_head_export and export_strategy == "veomni_lm_head_full"
        )

        # The caller enters actor update immediately after this RPC. Keeping the
        # already materialized actor on device avoids a full model offload/load
        # pair when VeOmni parameter offload is enabled.
        keep_model_on_device_after_export = bool(
            keep_model_on_device and restore_cpu_after_export
        )
        # DTensor.full_tensor() is collective, so every rank must execute it.
        # Only rank 0 retains the host payload consumed by the trainer.
        if getattr(worker, "rank", None) != 0:
            return None

        weight = (
            materialized_weight.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        logger.warning(
            "[actor lm_head export] veomni_lm_head_only name=%s shape=%s "
            "source_vocab=%s selected_rows=%s device=%s reclaim_staging=%s",
            selected_name,
            tuple(weight.shape),
            source_vocab_size,
            selected_rows,
            actor_device_type,
            int(reclaim_npu_staging),
        )
        return {
            "name": selected_name,
            "weight": weight,
            "row_indices": exported_rows,
            "source_vocab_size": source_vocab_size,
            "selected_rows": selected_rows,
            "export_strategy": export_strategy,
            "actor_backend": "veomni",
            "actor_device_type": actor_device_type,
        }
    finally:
        device_module = None
        if reclaim_npu_staging and materialized_weight is not None:
            device_module = getattr(torch, "npu", None)
            if device_module is not None:
                synchronize = getattr(device_module, "synchronize", None)
                if callable(synchronize):
                    synchronize()
        materialized_weight = None
        if (
            restore_cpu_after_export
            and offload_model is not None
            and not keep_model_on_device_after_export
        ):
            offload_model(module)
        if device_module is not None:
            empty_cache = getattr(device_module, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()


def export_actor_lm_head_weight(
    worker: Any,
    row_indices: Any = None,
    keep_model_on_device: bool = False,
) -> Optional[dict]:
    """Export actor lm_head or tied embedding rows from an actor-rollout worker."""

    torch = _torch_module()

    if (
        not getattr(worker, "_is_actor", False)
        or getattr(worker, "actor", None) is None
    ):
        return None

    if _is_veomni_actor_worker(worker):
        return _export_veomni_actor_lm_head_weight(
            worker,
            row_indices=row_indices,
            keep_model_on_device=keep_model_on_device,
        )

    normalized_row_indices = _normalize_lm_head_row_indices(row_indices)
    # Block drafters that train against the target's own lm_head rows.
    is_dflash = drafter_speculative_algorithm(getattr(worker, "config", None)) in {
        "DFLASH",
        "DFLASH2",
        "DSPARK",
    }
    if (
        is_dflash
        and normalized_row_indices is not None
        and int(normalized_row_indices.numel()) > 0
    ):
        direct_payload = _export_actor_lm_head_rows_direct(
            worker, normalized_row_indices
        )
        if isinstance(direct_payload, dict) and direct_payload.get(
            "_speco_non_owner_direct_sparse"
        ):
            return None
        if direct_payload is not None:
            return direct_payload

    per_tensor_param, _ = worker.actor.engine.get_per_tensor_param(
        layered_summon=getattr(worker, "layered_summon", False),
        base_sync_done=True,
    )
    selected_name = None
    selected_weight = None
    fallback_name = None
    fallback_weight = None

    for name, tensor in per_tensor_param:
        if not torch.is_tensor(tensor):
            continue
        name = str(name)
        if name == "model.embed_tokens.weight" or name.endswith(".embed_tokens.weight"):
            fallback_name = name
            fallback_weight = tensor
        if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
            selected_name = name
            selected_weight = tensor
            break

    if selected_weight is None:
        selected_name = fallback_name
        selected_weight = fallback_weight

    if getattr(worker, "rank", None) != 0:
        return None
    if selected_weight is None:
        logger.warning(
            "Unable to find actor lm_head.weight or tied model.embed_tokens.weight for SPECO sync"
        )
        return None

    selected_rows = None
    source_vocab_size = int(selected_weight.shape[0])
    exported_row_indices = None
    row_indices = normalized_row_indices
    if row_indices is not None:
        row_indices = row_indices.to(device=selected_weight.device, dtype=torch.long)
        if row_indices.numel() > 0 and row_indices.numel() < source_vocab_size:
            selected_weight = selected_weight.index_select(0, row_indices)
            exported_row_indices = (
                row_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            )
            selected_rows = int(row_indices.numel())
        elif row_indices.numel() == 0:
            logger.warning(
                "Received empty lm_head row_indices for SPECO sync; falling back to full lm_head export"
            )

    weight = (
        selected_weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    )
    logger.warning(
        "[actor lm_head export] name=%s shape=%s dtype=%s source_vocab=%s selected_rows=%s",
        selected_name,
        tuple(weight.shape),
        weight.dtype,
        source_vocab_size,
        selected_rows,
    )
    return {
        "name": selected_name,
        "weight": weight,
        "row_indices": exported_row_indices,
        "source_vocab_size": source_vocab_size,
        "selected_rows": selected_rows,
        "export_strategy": "engine_full_param",
    }


class DraftWeightPublishMixin:
    """Mixin for external actor-rollout workers that publish SPECO draft weights."""

    config: Any
    rollout: Any

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    def init_model(self, *args, **kwargs):
        install_rollout_runtime_for_worker(self)
        install_oldlogprob_hidden_runtime_for_worker(self)
        result = super().init_model(*args, **kwargs)
        validate_oldlogprob_hidden_runtime_for_worker(self)
        return result

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    def get_actor_lm_head_weight(
        self,
        row_indices: Any = None,
        keep_model_on_device: bool = False,
    ):
        return export_actor_lm_head_weight(
            self,
            row_indices=row_indices,
            keep_model_on_device=keep_model_on_device,
        )

    @staticmethod
    def _materialize_draft_weights_payload(weights):
        return materialize_draft_weights_payload(weights)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    async def update_draft_weights(
        self, weights: dict, global_steps: int | None = None
    ):
        if not drafter_rollout_enabled(self.config):
            return

        self._attach_update_draft_weights_to_rollout()
        materialize_ts = time.perf_counter()
        weights, used_ref = materialize_draft_weights_payload(weights)
        if used_ref:
            logger.warning(
                "[speco publish materialize] async=False global_steps=%s elapsed_sec=%.3f num_weights=%s",
                global_steps,
                time.perf_counter() - materialize_ts,
                len(weights) if weights else 0,
            )
        await self.rollout.update_draft_weights(weights, global_steps=global_steps)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None), blocking=False)
    async def update_draft_weights_async(
        self, weights: dict, global_steps: int | None = None
    ):
        if not drafter_rollout_enabled(self.config):
            return

        self._attach_update_draft_weights_to_rollout()
        materialize_ts = time.perf_counter()
        weights, used_ref = materialize_draft_weights_payload(weights)
        if used_ref:
            logger.warning(
                "[speco publish materialize] async=True global_steps=%s elapsed_sec=%.3f num_weights=%s",
                global_steps,
                time.perf_counter() - materialize_ts,
                len(weights) if weights else 0,
            )
        await self.rollout.update_draft_weights(weights, global_steps=global_steps)

    def _attach_update_draft_weights_to_rollout(self):
        backend = rollout_backend_name(getattr(self, "config", None))
        if backend == "vllm":
            from verl_speco.integration.vllm_runtime import (
                attach_update_draft_weights_to_rollout,
            )
        else:
            from verl_speco.integration.sglang_runtime import (
                attach_update_draft_weights_to_rollout,
            )

        attach_update_draft_weights_to_rollout(getattr(self, "rollout", None))
