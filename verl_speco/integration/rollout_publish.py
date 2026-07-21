"""Draft-weight publishing helpers for SPECO rollout adapters."""

import logging
import os
import time
from typing import Any, Optional

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
    if bool(_get_nested(config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False)):
        return True
    try:
        from verl_speco.integration.sglang_runtime import _load_env_drafter_config

        return bool(_load_env_drafter_config().get("enable"))
    except Exception:  # noqa: BLE001
        return False


def drafter_speculative_algorithm(config: Any) -> str:
    value = _get_nested(config, ("rollout", "drafter", "speculative_algorithm"), None)
    if value is None:
        value = _get_nested(config, ("actor_rollout_ref", "rollout", "drafter", "speculative_algorithm"), None)
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

    from verl_speco.integration.verl_npu_vllm_compat import install_verl_npu_vllm_import_compat

    # This must run before importing verl's vLLM rollout modules. verl
    # release/v0.8.0 still treats vLLM >= 0.18 FusedMoE as a class on NPU.
    install_verl_npu_vllm_import_compat()

    try:
        from verl_speco.integration.vllm_runtime import install_vllm_runtime_for_worker as _install
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

    try:
        from verl_speco.integration.oldlogprob_runtime import (
            install_oldlogprob_hidden_runtime_patch,
            oldlogprob_hidden_runtime_enabled,
        )
    except Exception:  # noqa: BLE001
        return

    drafter_env = getattr(type(worker), "_speco_sglang_drafter_config_env", None) or None
    if not oldlogprob_hidden_runtime_enabled(getattr(worker, "config", None), drafter_env=drafter_env):
        return
    install_oldlogprob_hidden_runtime_patch()


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


def _select_lm_head_named_tensor(module: Any):
    torch = _torch_module()
    direct_weight = getattr(module, "weight", None)
    if torch.is_tensor(direct_weight) and direct_weight.dim() == 2:
        return "lm_head.weight", direct_weight

    for attr_name in ("lm_head", "embed_tokens"):
        child = getattr(module, attr_name, None)
        child_weight = getattr(child, "weight", None)
        if torch.is_tensor(child_weight) and child_weight.dim() == 2:
            return f"{attr_name}.weight", child_weight

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return None, None

    fallback = (None, None)
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
            if name == "model.embed_tokens.weight" or name.endswith(".embed_tokens.weight"):
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
            if int(row_indices_cpu.max().item()) >= source_vocab_size or int(row_indices_cpu.min().item()) < 0:
                continue
            rows_on_device = row_indices_cpu.to(device=selected_weight.device, dtype=torch.long)
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
                "row_indices": row_indices_cpu.to(device="cpu", dtype=torch.long).contiguous(),
                "source_vocab_size": source_vocab_size,
                "selected_rows": int(row_indices_cpu.numel()),
                "export_strategy": "direct_sparse",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Direct sparse lm_head export failed for %s: %s", selected_name, exc)
            continue
    return None


def export_actor_lm_head_weight(worker: Any, row_indices: Any = None) -> Optional[dict]:
    """Export actor lm_head or tied embedding rows from an actor-rollout worker."""

    torch = _torch_module()

    if not getattr(worker, "_is_actor", False) or getattr(worker, "actor", None) is None:
        return None

    normalized_row_indices = _normalize_lm_head_row_indices(row_indices)
    is_dflash = drafter_speculative_algorithm(getattr(worker, "config", None)) in {"DFLASH", "DSPARK"}
    if is_dflash and normalized_row_indices is not None and int(normalized_row_indices.numel()) > 0:
        direct_payload = _export_actor_lm_head_rows_direct(worker, normalized_row_indices)
        if isinstance(direct_payload, dict) and direct_payload.get("_speco_non_owner_direct_sparse"):
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
        logger.warning("Unable to find actor lm_head.weight or tied model.embed_tokens.weight for SPECO sync")
        return None

    selected_rows = None
    source_vocab_size = int(selected_weight.shape[0])
    exported_row_indices = None
    row_indices = normalized_row_indices
    if row_indices is not None:
        row_indices = row_indices.to(device=selected_weight.device, dtype=torch.long)
        if row_indices.numel() > 0 and row_indices.numel() < source_vocab_size:
            selected_weight = selected_weight.index_select(0, row_indices)
            exported_row_indices = row_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            selected_rows = int(row_indices.numel())
        elif row_indices.numel() == 0:
            logger.warning("Received empty lm_head row_indices for SPECO sync; falling back to full lm_head export")

    weight = selected_weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
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

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    def init_model(self, *args, **kwargs):
        install_rollout_runtime_for_worker(self)
        install_oldlogprob_hidden_runtime_for_worker(self)
        return super().init_model(*args, **kwargs)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    def get_actor_lm_head_weight(self, row_indices: Any = None):
        return export_actor_lm_head_weight(self, row_indices=row_indices)

    @staticmethod
    def _materialize_draft_weights_payload(weights):
        return materialize_draft_weights_payload(weights)

    @register(dispatch_mode=getattr(Dispatch, "ONE_TO_ALL", None))
    async def update_draft_weights(self, weights: dict, global_steps: int = None):
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
    async def update_draft_weights_async(self, weights: dict, global_steps: int = None):
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
            from verl_speco.integration.vllm_runtime import attach_update_draft_weights_to_rollout
        else:
            from verl_speco.integration.sglang_runtime import attach_update_draft_weights_to_rollout

        attach_update_draft_weights_to_rollout(getattr(self, "rollout", None))
