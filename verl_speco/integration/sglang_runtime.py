"""Runtime bridge from external SPECO config to upstream verl SGLang rollout.

verl v0.8.0 does not expose ``rollout.drafter`` in ``RolloutConfig``.  SPECO
therefore keeps that subtree external to upstream config validation, and uses
this module to pass the relevant SGLang launch/update information to the Ray
actors that own the HTTP server and rollout adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
from dataclasses import fields
from typing import Any, Optional

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None
from verl_speco.integration.sglang_adapter import (
    DFLASH_RETURN_AUX_HIDDEN_PARAM,
    DRAFTER_RAW_TOP_LOGPROBS_PARAM,
    DRAFTER_RETURN_LAST_HIDDEN_PARAM,
    SGLANG_EAGLE_UPDATE_WEIGHTS_PATCH,
    SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCH,
    SGLANG_NPU_EAGLE_TARGET_SAMPLING_PATCH,
    SGLANG_QWEN3_ROPE_COMPAT_PATCH,
    sglang_needs_qwen3_rope_compat_patch,
    speco_step_matches_interval,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_SGLANG_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"
SPECO_SGLANG_RUNTIME_PATCHED_ENV = "VERL_SPECO_SGLANG_RUNTIME_PATCHED"
SPECO_TARGET_WEIGHT_LOADER = "verl_speco.integration.sglang_runtime.speco_sglang_target_weight_loader"
SPECO_DRAFT_WEIGHT_LOADER = "verl_speco.integration.sglang_runtime.speco_sglang_draft_weight_loader"
SPECO_DISABLE_SPECULATIVE_WEIGHT_SYNC_GUARD_ENV = "VERL_SPECO_DISABLE_SPECULATIVE_WEIGHT_SYNC_GUARD"
_DRAFTER_HIDDEN_WINDOW_PARAM = "_verl_drafter_hidden_state_window"
_HIDDEN_STATE_FRONT_TOKENS_PARAM = "_verl_hidden_state_front_tokens_per_sample"
_HIDDEN_STATE_PROMPT_LEN_PARAM = "_verl_prompt_len"
_HIDDEN_STATE_WINDOW_START_PARAM = "_verl_hidden_state_window_start"
_HIDDEN_STATE_WINDOW_END_PARAM = "_verl_hidden_state_window_end"
_HIDDEN_STATE_WINDOW_START_OFFSET_PARAM = "_verl_hidden_state_window_start_offset"
_HIDDEN_STATE_WINDOW_MIN_ROWS_PARAM = "_verl_hidden_state_window_min_rows"
_VERL_DRAFTER_RAW_TOP_LOGPROBS_ENV = "VERL_DRAFTER_RAW_TOP_LOGPROBS"

_SERVER_ARGS_PATCHED = False
_SGLANG_REPLICA_PATCHED = False


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


def _plain_container(value: Any):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        pass

    if isinstance(value, dict):
        return {key: _plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_container(item) for item in value]
    return value


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n", ""}:
        return False
    return default


def _is_npu_runtime() -> bool:
    try:
        from verl.utils.device import get_device_name

        return get_device_name() == "npu"
    except Exception:  # noqa: BLE001
        return False


def _load_env_drafter_config() -> dict[str, Any]:
    raw = os.getenv(SPECO_SGLANG_DRAFTER_CONFIG_ENV)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {SPECO_SGLANG_DRAFTER_CONFIG_ENV}: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def configure_sglang_runtime_from_config(config: Any) -> dict[str, Any]:
    """Serialize enabled SPECO drafter config for SGLang Ray actors."""

    drafter = _plain_container(_get_nested(config, ("actor_rollout_ref", "rollout", "drafter"), {})) or {}
    enabled = bool(drafter.get("enable"))
    if not enabled:
        os.environ.pop(SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)
        return {}

    os.environ[SPECO_SGLANG_DRAFTER_CONFIG_ENV] = json.dumps(drafter, sort_keys=True)
    return drafter


def clear_sglang_runtime_config() -> None:
    os.environ.pop(SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)


def _drafter_uses_eagle_last_hidden(drafter_cfg: dict[str, Any]) -> bool:
    algorithm = str(drafter_cfg.get("speculative_algorithm", "") or "").upper()
    training_cfg = drafter_cfg.get("training") or {}
    return bool(
        algorithm == "EAGLE3"
        and drafter_cfg.get("enable")
        and drafter_cfg.get("enable_drafter_training")
        and training_cfg.get("collect_hidden_states_from_sgl")
        and not training_cfg.get("use_logits")
    )


def _drafter_uses_dflash_aux_hidden(drafter_cfg: dict[str, Any]) -> bool:
    algorithm = str(drafter_cfg.get("speculative_algorithm", "") or "").upper()
    training_cfg = drafter_cfg.get("training") or {}
    return bool(
        algorithm in {"DFLASH", "DSPARK"}
        and drafter_cfg.get("enable")
        and drafter_cfg.get("enable_drafter_training")
        and training_cfg.get("collect_hidden_states_from_sgl")
        and not training_cfg.get("use_logits")
    )


def _positive_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return max(int(default), 0)


def _expected_full_hidden_rows(prompt_len: int, output_len: int) -> int:
    return max(int(prompt_len) + int(output_len) - 1, 0)


def _expected_collected_hidden_rows(
    prompt_len: int,
    output_len: int,
    front_tokens: Optional[int],
    tail_tokens: Optional[int],
) -> int:
    expected_full_rows = _expected_full_hidden_rows(prompt_len, output_len)
    if front_tokens is not None:
        train_window_start = max(int(prompt_len) - 1, 0)
        available_train_rows = max(expected_full_rows - train_window_start, 0)
        return min(available_train_rows, int(front_tokens))
    if tail_tokens is not None:
        return min(expected_full_rows, int(tail_tokens))
    return expected_full_rows


def _hidden_state_window_mode(training_cfg: dict[str, Any]) -> str:
    mode = str(training_cfg.get("hidden_state_window_mode", "random") or "random").strip().lower()
    return mode if mode in {"front", "random"} else "front"


def _deterministic_random_window_offset(
    *,
    request_id: str,
    collection_global_steps: Optional[int],
    replica_rank: int,
    prompt_len: int,
    max_new_tokens: int,
    max_start_offset: int,
    seed_by_step: bool,
) -> int:
    if max_start_offset <= 0:
        return 0
    step_key = collection_global_steps if seed_by_step else "request"
    seed_material = (
        f"{step_key}:{replica_rank}:{request_id}:{prompt_len}:{max_new_tokens}:{max_start_offset}"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little", signed=False)
    return seed % (max_start_offset + 1)


def _build_drafter_hidden_window_plan(
    *,
    training_cfg: dict[str, Any],
    request_id: str,
    collection_global_steps: Optional[int],
    replica_rank: int,
    prompt_len: int,
    max_new_tokens: int,
    front_tokens: Optional[int],
    tail_tokens: Optional[int],
) -> dict[str, Any]:
    mode = _hidden_state_window_mode(training_cfg)
    if mode == "random":
        window_size = _positive_int_or_none(training_cfg.get("hidden_state_window_tokens_per_sample"))
        if window_size is None:
            window_size = front_tokens
        if window_size is None:
            window_size = tail_tokens
        if window_size is None:
            window_size = max(int(max_new_tokens), 0)

        min_rows = max(_nonnegative_int(training_cfg.get("hidden_state_window_min_rows", 64), default=64), 1)
        max_offset_limit = _positive_int_or_none(training_cfg.get("hidden_state_random_max_offset"))
        if max_offset_limit is None:
            max_offset_limit = max(int(max_new_tokens), 0)
        max_offset_limit = min(max(max_offset_limit, 0), max(int(max_new_tokens), 0))

        target_window_rows = min(int(window_size), max_offset_limit, max(int(max_new_tokens), 0))
        if max_new_tokens <= 0 or target_window_rows < min_rows:
            return {"mode": mode, "estimated_rows": 0, "min_rows": min_rows}

        max_start_offset = max(max_offset_limit - target_window_rows, 0)
        random_offset = _deterministic_random_window_offset(
            request_id=request_id,
            collection_global_steps=collection_global_steps,
            replica_rank=replica_rank,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            max_start_offset=max_start_offset,
            seed_by_step=bool(training_cfg.get("hidden_state_random_seed_by_step", True)),
        )
        train_base = max(int(prompt_len) - 1, 0)
        window_start = train_base + random_offset
        window_end = train_base + random_offset + target_window_rows
        estimated_rows = max(window_end - window_start, 0)
        if estimated_rows <= 0:
            return {"mode": mode, "estimated_rows": 0, "min_rows": min_rows}
        return {
            "mode": mode,
            "estimated_rows": estimated_rows,
            "min_rows": min_rows,
            "window_start": window_start,
            "window_end": window_end,
            "window_start_offset": random_offset,
            "window_size": int(window_size),
            "target_window_rows": target_window_rows,
        }

    return {
        "mode": "front",
        "estimated_rows": _expected_collected_hidden_rows(prompt_len, max_new_tokens, front_tokens, tail_tokens),
        "min_rows": 0,
    }


def _select_drafter_hidden_state_window(
    hidden_states: torch.Tensor,
    *,
    expected_hidden_rows: int,
    prompt_len: int,
    front_tokens: Optional[int],
    tail_tokens: Optional[int],
) -> tuple[torch.Tensor, int, int, str, int]:
    hidden_raw_len = int(hidden_states.size(0))
    prefix_cache_rows = max(int(expected_hidden_rows) - hidden_raw_len, 0)
    crop_start = 0
    crop_end = hidden_raw_len
    crop_mode = "all"

    train_hidden_start = max(prefix_cache_rows, max(int(prompt_len) - 1, 0))
    if train_hidden_start > prefix_cache_rows:
        crop_start = min(train_hidden_start - prefix_cache_rows, hidden_raw_len)
        crop_mode = "front"

    if front_tokens is not None and max(hidden_raw_len - crop_start, 0) > front_tokens:
        crop_end = crop_start + front_tokens
        crop_mode = "front"
    elif tail_tokens is not None and hidden_raw_len > tail_tokens:
        crop_start = hidden_raw_len - tail_tokens
        crop_mode = "tail"

    hidden_position_start = prefix_cache_rows + crop_start
    hidden_position_end = prefix_cache_rows + crop_end
    return (
        hidden_states[crop_start:crop_end],
        hidden_position_start,
        hidden_position_end,
        crop_mode,
        prefix_cache_rows,
    )


def _hidden_state_metadata_from_chunk(hidden_state_chunk: Any) -> dict[str, Any]:
    if not isinstance(hidden_state_chunk, dict):
        return {}

    metadata: dict[str, Any] = {}
    for source_key, target_key in (
        ("position_start", "position_start"),
        ("position_end", "position_end"),
        ("hidden_position_start", "position_start"),
        ("hidden_position_end", "position_end"),
        ("prefix_cache_rows", "prefix_cache_rows"),
        ("window_start", "window_start"),
        ("window_end", "window_end"),
    ):
        if source_key not in hidden_state_chunk:
            continue
        value = _positive_int_or_none(hidden_state_chunk.get(source_key))
        if value is not None or hidden_state_chunk.get(source_key) == 0:
            metadata[target_key] = int(hidden_state_chunk[source_key])

    positions = hidden_state_chunk.get("positions")
    if positions is not None:
        try:
            if torch.is_tensor(positions):
                metadata["positions"] = positions.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            else:
                metadata["positions"] = torch.tensor(list(positions), dtype=torch.long).reshape(-1)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to normalize SGLang hidden positions metadata; ignoring positions")

    for key in (
        "lm_head_fingerprint",
        "last_hidden_logprob_check",
        "raw_topk_logprob_check",
        "last_hidden_filter",
        "last_hidden_select",
    ):
        value = hidden_state_chunk.get(key)
        if isinstance(value, dict):
            metadata[key] = value

    target_logprobs_source = hidden_state_chunk.get("target_logprobs_source")
    if isinstance(target_logprobs_source, str):
        metadata["target_logprobs_source"] = target_logprobs_source

    raw_target_logprobs = hidden_state_chunk.get("raw_target_logprobs")
    if raw_target_logprobs is not None:
        try:
            raw_tensor = (
                raw_target_logprobs.detach().to(device="cpu", dtype=torch.float32)
                if torch.is_tensor(raw_target_logprobs)
                else torch.tensor(raw_target_logprobs, dtype=torch.float32)
            )
            if raw_tensor.dim() == 3 and raw_tensor.size(-1) >= 2:
                metadata["raw_target_logprobs"] = raw_tensor.contiguous()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to normalize SGLang raw top-k metadata; ignoring raw target logprobs")

    raw_positions = hidden_state_chunk.get("raw_target_logprobs_positions")
    if raw_positions is not None and torch.is_tensor(metadata.get("raw_target_logprobs")):
        try:
            raw_positions_tensor = (
                raw_positions.detach().to(device="cpu", dtype=torch.long).reshape(-1)
                if torch.is_tensor(raw_positions)
                else torch.tensor(list(raw_positions), dtype=torch.long).reshape(-1)
            )
            if int(raw_positions_tensor.numel()) == int(metadata["raw_target_logprobs"].size(0)):
                metadata["raw_target_logprobs_positions"] = raw_positions_tensor.contiguous()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to normalize SGLang raw top-k positions metadata; ignoring positions")

    return metadata


def _hidden_state_chunk_payload(hidden_state_chunk: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(hidden_state_chunk, dict):
        return hidden_state_chunk, {}

    payload = None
    for key in ("hidden_states", "tensor", "data"):
        if key in hidden_state_chunk:
            payload = hidden_state_chunk[key]
            break
    return payload, _hidden_state_metadata_from_chunk(hidden_state_chunk)


def _hidden_state_chunk_to_tensor_and_metadata(
    hidden_state_chunk: Any,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    if hidden_state_chunk is None:
        return None, {}

    hidden_state_chunk, metadata = _hidden_state_chunk_payload(hidden_state_chunk)
    if hidden_state_chunk is None:
        return None, metadata

    if torch.is_tensor(hidden_state_chunk):
        h_states = hidden_state_chunk.detach().to(device="cpu", dtype=torch.bfloat16)
    elif isinstance(hidden_state_chunk, (bytes, bytearray, memoryview, str)):
        serialized_chunk = (
            hidden_state_chunk.tobytes()
            if isinstance(hidden_state_chunk, memoryview)
            else bytes(hidden_state_chunk)
            if isinstance(hidden_state_chunk, bytearray)
            else hidden_state_chunk
        )
        try:
            from sglang.srt.utils import MultiprocessingSerializer

            deserialized = MultiprocessingSerializer.deserialize(serialized_chunk)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deserialize SGLang hidden-state chunk; skipping chunk")
            return None, metadata
        tensor, nested_metadata = _hidden_state_chunk_to_tensor_and_metadata(deserialized)
        return tensor, {**metadata, **nested_metadata}
    else:
        h_states = torch.tensor(hidden_state_chunk, dtype=torch.bfloat16)

    if h_states.numel() == 0:
        return None, metadata
    if h_states.dim() == 1:
        h_states = h_states.unsqueeze(0)
    elif h_states.dim() == 3:
        h_states = h_states.squeeze(0)
    return h_states.contiguous(), metadata


def _iter_hidden_state_chunks(hidden_states_data: Any):
    if hidden_states_data is None:
        return []
    if (
        torch.is_tensor(hidden_states_data)
        or isinstance(hidden_states_data, dict)
        or isinstance(hidden_states_data, (bytes, bytearray, memoryview, str))
    ):
        return [hidden_states_data]
    return hidden_states_data


def _target_logprobs_invalid_rows_like(target_logprobs: torch.Tensor, rows: int) -> torch.Tensor:
    shape = (int(rows),) + tuple(target_logprobs.shape[1:])
    invalid = target_logprobs.new_empty(shape)
    invalid.zero_()
    if invalid.dim() >= 3 and int(invalid.size(-1)) >= 2:
        invalid[..., 0] = float("-inf")
        invalid[..., 1] = -1
    return invalid


def _select_target_logprobs_by_raw_positions(
    target_logprobs: Optional[torch.Tensor],
    raw_positions: Optional[torch.Tensor],
    *,
    desired_position_start: int,
    desired_position_end: int,
) -> tuple[Optional[torch.Tensor], Optional[int], Optional[int], int]:
    if target_logprobs is None or raw_positions is None:
        return None, None, None, 0
    row_count = int(target_logprobs.size(0))
    desired_rows = max(int(desired_position_end) - int(desired_position_start), 0)
    if row_count <= 0 or desired_rows <= 0 or not torch.is_tensor(raw_positions):
        return None, None, None, row_count

    raw_positions = raw_positions.reshape(-1).to(device=target_logprobs.device, dtype=torch.long)
    if int(raw_positions.numel()) < row_count:
        return None, None, None, row_count
    raw_positions = raw_positions[:row_count]
    keep_mask = (
        (raw_positions >= 0)
        & (raw_positions >= int(desired_position_start))
        & (raw_positions < int(desired_position_end))
    )
    if not bool(keep_mask.any()):
        return None, None, None, row_count

    selected = _target_logprobs_invalid_rows_like(target_logprobs, desired_rows)
    source_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    dest_indices = (raw_positions[source_indices] - int(desired_position_start)).to(dtype=torch.long)
    selected[dest_indices] = target_logprobs[source_indices]
    return (
        selected.contiguous(),
        int(desired_position_start),
        int(desired_position_start) + desired_rows,
        row_count - int(source_indices.numel()),
    )


def _server_args_overrides_from_drafter(drafter_cfg: dict[str, Any], supported_fields: set[str]) -> dict[str, Any]:
    """Build SGLang ``ServerArgs`` overrides from SPECO drafter config."""

    if not bool(drafter_cfg.get("enable")):
        return {}

    algorithm = str(drafter_cfg.get("speculative_algorithm", "") or "").strip().upper()
    if algorithm == "DOMINO":
        # Domino is a DFlash variant, not an engine-level method: engines expose it as
        # "dflash" and enable the causal correction head (prefix_gru + embed_proj) from
        # the checkpoint's dflash_config.projector_type="domino" (vllm-project/vllm#48241,
        # sgl-project/sglang#31328). DOMINO is never a valid SGLang ServerArgs algorithm,
        # so fail loud and point at DFLASH, mirroring
        # vllm_runtime._speculative_method_from_drafter.
        raise ValueError(
            "DOMINO is not an engine-level speculative algorithm; Domino is served as a DFlash "
            "projector sub-mode. Set actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH "
            "for the rollout/serve path; the trained checkpoint's dflash_config.projector_type=domino "
            "enables the Domino correction head on engines that support it, keeping DOMINO for "
            "drafter training."
        )

    rollout_cfg = drafter_cfg.get("rollout") or {}
    training_cfg = drafter_cfg.get("training") or {}
    overrides = {
        "speculative_algorithm": drafter_cfg.get("speculative_algorithm"),
        "speculative_draft_model_path": drafter_cfg.get("model_path"),
        "speculative_num_steps": rollout_cfg.get("spec_steps"),
        "speculative_eagle_topk": rollout_cfg.get("spec_topk"),
        "speculative_num_draft_tokens": rollout_cfg.get("spec_verify_tokens"),
        "enable_return_hidden_states": bool(
            drafter_cfg.get("enable_drafter_training") and training_cfg.get("collect_hidden_states_from_sgl")
        ),
        "enable_weights_cpu_backup": True,
        "enable_draft_weights_cpu_backup": True,
    }

    cuda_graph_max_bs = rollout_cfg.get("cuda_graph_max_bs")
    if cuda_graph_max_bs is not None:
        overrides["cuda_graph_max_bs"] = cuda_graph_max_bs

    if "custom_weight_loader" in supported_fields:
        overrides["custom_weight_loader"] = [SPECO_TARGET_WEIGHT_LOADER, SPECO_DRAFT_WEIGHT_LOADER]

    return {key: value for key, value in overrides.items() if key in supported_fields and value is not None}


def _set_sglang_patch_envs(drafter_cfg: dict[str, Any]) -> None:
    training_cfg = drafter_cfg.get("training") or {}
    os.environ["VERL_SGLANG_DRAFTER_RETURN_LAST_HIDDEN"] = "1" if _drafter_uses_eagle_last_hidden(drafter_cfg) else "0"
    raw_top_logprobs_enabled = bool(
        drafter_cfg.get("enable")
        and drafter_cfg.get("enable_drafter_training")
        and training_cfg.get("collect_hidden_states_from_sgl")
        and training_cfg.get("use_logits")
    )
    os.environ["VERL_DRAFTER_RAW_TOP_LOGPROBS"] = "1" if raw_top_logprobs_enabled else "0"
    if raw_top_logprobs_enabled:
        os.environ["VERL_DRAFTER_RAW_TOP_LOGPROBS_TOPK"] = str(max(int(training_cfg.get("logits_topk", 1)), 1))


def _default_sglang_verl_patches(drafter_cfg: dict[str, Any]) -> set[str]:
    patches: set[str] = set()
    if _is_npu_runtime():
        patches.add(SGLANG_NPU_EAGLE_TARGET_SAMPLING_PATCH)
    if sglang_needs_qwen3_rope_compat_patch():
        patches.add(SGLANG_QWEN3_ROPE_COMPAT_PATCH)

    if drafter_cfg.get("enable"):
        patches.add(SGLANG_EAGLE_UPDATE_WEIGHTS_PATCH)
    if drafter_cfg.get("enable") and drafter_cfg.get("enable_drafter_training"):
        training_cfg = drafter_cfg.get("training") or {}
        if training_cfg.get("collect_hidden_states_from_sgl"):
            patches.add(SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCH)
    return patches


def _install_server_args_patch(drafter_cfg: dict[str, Any] | None = None) -> None:
    """Patch SGLang ServerArgs construction inside SGLang server actors."""

    global _SERVER_ARGS_PATCHED
    if _SERVER_ARGS_PATCHED:
        return

    from sglang.srt.server_args import ServerArgs

    original_init = ServerArgs.__init__
    supported_fields = {field.name for field in fields(ServerArgs)}

    def speco_server_args_init(self, *args, **kwargs):
        cfg = drafter_cfg or _load_env_drafter_config()
        if cfg:
            custom_loaders = list(kwargs.get("custom_weight_loader") or [])
            for key, value in _server_args_overrides_from_drafter(cfg, supported_fields).items():
                if key == "custom_weight_loader":
                    for loader in value:
                        if loader not in custom_loaders:
                            custom_loaders.append(loader)
                    kwargs[key] = custom_loaders
                else:
                    kwargs[key] = value
        return original_init(self, *args, **kwargs)

    ServerArgs.__init__ = speco_server_args_init
    _SERVER_ARGS_PATCHED = True


def _install_sglang_hidden_state_patch(drafter_cfg: dict[str, Any]) -> None:
    from verl_speco.integration.sglang_adapter import SGLangSpecoPatchConfig, install_sglang_speco_patches

    training_cfg = drafter_cfg.get("training") or {}
    _set_sglang_patch_envs(drafter_cfg)
    default_patches = _default_sglang_verl_patches(drafter_cfg)
    if default_patches:
        logger.info("Installing SPECO SGLang patches: %s", ", ".join(sorted(default_patches)))
    install_sglang_speco_patches(
        SGLangSpecoPatchConfig(
            target_weight_loader=SPECO_TARGET_WEIGHT_LOADER,
            draft_weight_loader=SPECO_DRAFT_WEIGHT_LOADER,
            enable_original_logprobs=bool(training_cfg.get("use_logits", False)),
            patches=default_patches,
        )
    )


def install_sglang_server_actor_runtime() -> dict[str, Any]:
    """Install SPECO runtime hooks inside a SGLang HTTP server actor."""

    drafter_cfg = _load_env_drafter_config()
    if not drafter_cfg:
        if sglang_needs_qwen3_rope_compat_patch():
            from verl_speco.integration.sglang_adapter import install_sglang_qwen3_rope_compat_patch

            install_sglang_qwen3_rope_compat_patch()
            os.environ[SPECO_SGLANG_RUNTIME_PATCHED_ENV] = "1"
            logger.warning("Installed SPECO SGLang Qwen3 rope compatibility runtime patch.")
        return {}

    _install_server_args_patch(drafter_cfg)
    _install_sglang_hidden_state_patch(drafter_cfg)
    os.environ[SPECO_SGLANG_RUNTIME_PATCHED_ENV] = "1"
    logger.warning(
        "Installed SPECO SGLang runtime: algorithm=%s draft_model=%s",
        drafter_cfg.get("speculative_algorithm"),
        drafter_cfg.get("model_path"),
    )
    return drafter_cfg


def _is_sglang_eagle_draft_model(model) -> bool:
    config = getattr(model, "config", None)
    class_name = type(model).__name__.lower()
    architectures = getattr(config, "architectures", None) or []
    return (
        "eagle" in class_name
        or any("eagle" in str(architecture).lower() for architecture in architectures)
        or getattr(config, "draft_vocab_size", None) is not None
    )


def speco_sglang_target_weight_loader(model, named_tensors):
    """Load target weights without touching the SGLang draft model."""

    if _is_sglang_eagle_draft_model(model):
        return
    return model.load_weights(named_tensors)


def speco_sglang_draft_weight_loader(model, named_tensors):
    """Load draft weights without touching the SGLang target model."""

    if not _is_sglang_eagle_draft_model(model):
        return
    return model.load_weights(named_tensors)


async def _sgl_update_weights_with_route(
    *,
    engine,
    params_batch,
    device_mesh_key: str,
    device_mesh,
    disable_draft_model: bool | None,
    disable_target_model: bool | None,
    load_format: str | None = None,
    stage_cpu_tensors_to_device: bool = False,
    flush_cache: bool = False,
    abort_all_requests: bool = False,
):
    import torch
    import torch.distributed as dist
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights
    from verl.utils.device import get_device_name, get_torch_device

    infer_tp_mesh = device_mesh[device_mesh_key]
    infer_tp_size = infer_tp_mesh.mesh.size()[0]
    infer_tp_rank = infer_tp_mesh.get_local_rank()
    device_name = get_device_name()

    monkey_patch_torch_reductions()

    def _prepare_update_tensor(tensor):
        tensor = tensor.detach()
        if stage_cpu_tensors_to_device and tensor.device.type == "cpu" and device_name != "cpu":
            device_module = get_torch_device()
            tensor = tensor.to(
                torch.device(f"{device_name}:{device_module.current_device()}"),
                non_blocking=True,
            )
        return _preprocess_tensor_for_update_weights(tensor)

    named_tensors_batch = [
        (name, MultiprocessingSerializer.serialize(_prepare_update_tensor(tensor))) for name, tensor in params_batch
    ]
    gathered_serialized_batches = [None for _ in range(infer_tp_size)] if infer_tp_rank == 0 else None
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=infer_tp_mesh.mesh.tolist()[0],
        group=infer_tp_mesh.get_group(),
    )
    if infer_tp_rank != 0:
        return None

    logical_tensors = zip(*gathered_serialized_batches, strict=True)
    named_tensors = [
        (tensor_group[0][0], LocalSerializedTensor(values=[rank_part[1] for rank_part in tensor_group]))
        for tensor_group in logical_tensors
    ]
    update_weights_request = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[MultiprocessingSerializer.serialize(named_tensors) for _ in range(infer_tp_size)],
        load_format=load_format,
        flush_cache=flush_cache,
    )
    setattr(update_weights_request, "abort_all_requests", abort_all_requests)
    if disable_draft_model is not None:
        setattr(update_weights_request, "disable_draft_model", disable_draft_model)
    if disable_target_model is not None:
        setattr(update_weights_request, "disable_target_model", disable_target_model)
    result = await engine.update_weights_from_tensor(update_weights_request)
    if isinstance(result, dict):
        success = result.get("success")
        status = str(result.get("status", "")).lower()
        if success is False or status in {"error", "failed", "failure"} or bool(result.get("error")):
            route = "target" if disable_draft_model else "draft" if disable_target_model else "target/draft"
            raise RuntimeError(f"SGLang {route} weight update failed: {result.get('message') or result.get('error')}")
    return result


def _supports_sglang_custom_weight_loader() -> bool:
    try:
        from sglang.srt.server_args import ServerArgs

        return "custom_weight_loader" in getattr(ServerArgs, "__dataclass_fields__", {})
    except Exception:  # noqa: BLE001
        return False


def _speculative_weight_sync_guard_disabled() -> bool:
    return _env_flag_enabled(SPECO_DISABLE_SPECULATIVE_WEIGHT_SYNC_GUARD_ENV, default=False)


def _server_adapter_original_update_weights(adapter: Any):
    return getattr(type(adapter), "_speco_original_update_weights", None)


def _sglang_engine_has_method(engine: Any, method_name: str) -> bool:
    return callable(getattr(engine, method_name, None))


async def _maybe_call_sglang_engine_method(engine: Any, method_name: str, *args, **kwargs) -> bool:
    method = getattr(engine, method_name, None)
    if not callable(method):
        logger.debug("SGLang engine %s has no %s(); skip", type(engine).__name__, method_name)
        return False
    try:
        result = method(*args, **kwargs)
    except TypeError as exc:
        if not kwargs or "unexpected keyword argument" not in str(exc):
            raise
        result = method(*args)
    if inspect.isawaitable(result):
        await result
    return True


async def speco_update_target_weights(self, weights, *args, global_steps: int = None, **kwargs):
    """Update only SGLang target weights when speculative drafter is enabled."""

    original_update_weights = _server_adapter_original_update_weights(self)
    drafter_cfg = _load_env_drafter_config()
    if not bool(drafter_cfg.get("enable")):
        if original_update_weights is None:
            return None
        return await original_update_weights(self, weights, *args, global_steps=global_steps, **kwargs)

    peft_config = getattr(self, "peft_config", None)
    base_sync_done = getattr(self, "base_sync_done", False)
    if peft_config is not None and base_sync_done:
        if original_update_weights is None:
            return None
        return await original_update_weights(self, weights, *args, global_steps=global_steps, **kwargs)

    await self._init_server_adapter()
    update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
    generation_paused = False

    from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

    if getattr(self.config, "get", None) is not None and self.config.get("quantization") == "fp8":
        from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

        hf_config = self.model_config.hf_config
        fp8_quantizer = SGLangFP8QuantizerHelper(hf_config.quantization_config)
        weights = fp8_quantizer.quant_weights_by_name(weights, dtype=hf_config.dtype)

    total_ts = time.perf_counter()
    try:
        engine_has_flush_cache = _sglang_engine_has_method(self._engine, "flush_cache")
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and not _speculative_weight_sync_guard_disabled():
            generation_paused = await _maybe_call_sglang_engine_method(
                self._engine,
                "pause_generation",
                mode="retract",
            )
            if engine_has_flush_cache:
                await _maybe_call_sglang_engine_method(self._engine, "flush_cache")

        bucket_idx = 0
        async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
            bucket_idx += 1
            await _sgl_update_weights_with_route(
                engine=self._engine,
                params_batch=list(params_batch),
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
                disable_draft_model=True,
                disable_target_model=False,
                load_format=SPECO_TARGET_WEIGHT_LOADER if _supports_sglang_custom_weight_loader() else None,
                stage_cpu_tensors_to_device=False,
                flush_cache=not engine_has_flush_cache,
                abort_all_requests=False,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            if engine_has_flush_cache:
                await _maybe_call_sglang_engine_method(self._engine, "flush_cache")
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)
            logger.warning(
                "[speco sglang target update] done global_steps=%s buckets=%s elapsed_sec=%.3f",
                global_steps,
                bucket_idx,
                time.perf_counter() - total_ts,
            )
    finally:
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and generation_paused:
            await _maybe_call_sglang_engine_method(self._engine, "continue_generation")


async def speco_update_draft_weights(self, weights: dict[str, Any], *args, global_steps: int = None, **kwargs):
    """Update only SGLang draft weights from an upstream ServerAdapter instance."""

    del args, kwargs

    if not weights:
        return
    drafter_cfg = _load_env_drafter_config()
    if not bool(drafter_cfg.get("enable")):
        return

    await self._init_server_adapter()
    training_cfg = drafter_cfg.get("training") or {}
    bucket_mb = training_cfg.get("draft_update_weights_bucket_megabytes")
    if bucket_mb is None:
        bucket_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
    update_weights_bucket_bytes = int(bucket_mb) << 20
    pause_generation = bool(training_cfg.get("draft_update_pause_generation", True))
    flush_before = bool(training_cfg.get("draft_update_flush_before", True))
    flush_after = bool(training_cfg.get("draft_update_flush_after", True))
    generation_paused = False

    from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

    total_ts = time.perf_counter()
    try:
        engine_has_flush_cache = _sglang_engine_has_method(self._engine, "flush_cache")
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and pause_generation:
            generation_paused = await _maybe_call_sglang_engine_method(
                self._engine,
                "pause_generation",
                mode="retract",
            )
            if flush_before and engine_has_flush_cache:
                await _maybe_call_sglang_engine_method(self._engine, "flush_cache")

        bucket_idx = 0
        async for params_batch in get_named_tensor_buckets(weights.items(), update_weights_bucket_bytes):
            bucket_idx += 1
            await _sgl_update_weights_with_route(
                engine=self._engine,
                params_batch=list(params_batch),
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
                disable_draft_model=False,
                disable_target_model=True,
                load_format=SPECO_DRAFT_WEIGHT_LOADER if _supports_sglang_custom_weight_loader() else None,
                stage_cpu_tensors_to_device=True,
                flush_cache=bool(flush_after and not engine_has_flush_cache),
                abort_all_requests=False,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            if flush_after and engine_has_flush_cache:
                await _maybe_call_sglang_engine_method(self._engine, "flush_cache")
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)
            logger.warning(
                "[speco sglang draft update] done global_steps=%s buckets=%s bucket_mb=%s elapsed_sec=%.3f",
                global_steps,
                bucket_idx,
                bucket_mb,
                time.perf_counter() - total_ts,
            )
    finally:
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and generation_paused:
            await _maybe_call_sglang_engine_method(self._engine, "continue_generation")


def attach_update_draft_weights_to_rollout(rollout: Any) -> Any:
    """Attach ``update_draft_weights`` to an upstream SGLang ServerAdapter."""

    if rollout is not None and not callable(getattr(rollout, "update_draft_weights", None)):
        rollout.update_draft_weights = speco_update_draft_weights.__get__(rollout, type(rollout))
    return rollout


def patch_sglang_server_adapter_update() -> None:
    try:
        from verl.workers.rollout.sglang_rollout import sglang_rollout
    except Exception:  # noqa: BLE001
        return

    server_adapter = getattr(sglang_rollout, "ServerAdapter", None)
    if server_adapter is None:
        return
    if not callable(getattr(server_adapter, "update_draft_weights", None)):
        server_adapter.update_draft_weights = speco_update_draft_weights
    if not getattr(server_adapter, "_speco_patched_target_update", False):
        original_update_weights = getattr(server_adapter, "update_weights", None)
        if callable(original_update_weights):
            server_adapter._speco_original_update_weights = original_update_weights
            server_adapter.update_weights = speco_update_target_weights
            server_adapter._speco_patched_target_update = True


class _SpecoSGLangHttpServerMixin:
    async def launch_server(self, *args, **kwargs):
        self._speco_drafter_config = install_sglang_server_actor_runtime()
        self.global_steps = getattr(self, "global_steps", None)
        self._drafter_collection_step = None
        self._drafter_collection_samples = 0
        self._drafter_collection_tokens = 0
        self._speco_collection_skip_log_keys = set()
        self._speco_hidden_missing_log_keys = set()
        return await super().launch_server(*args, **kwargs)

    def _speco_drafter_cfg(self) -> dict[str, Any]:
        cached_cfg = getattr(self, "_speco_drafter_config", None)
        if isinstance(cached_cfg, dict) and cached_cfg:
            return cached_cfg
        return _load_env_drafter_config()

    @staticmethod
    def _speco_strip_internal_sampling_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
        clean_params = dict(sampling_params)
        clean_params.pop("_verl_global_steps", None)
        clean_params.pop("_verl_skip_drafter_collection", None)
        return clean_params

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    def _reset_drafter_collection_budget_if_needed(self, collection_global_steps: Optional[int]) -> None:
        if getattr(self, "_drafter_collection_step", None) == collection_global_steps:
            return
        self._drafter_collection_step = collection_global_steps
        self._drafter_collection_samples = 0
        self._drafter_collection_tokens = 0

    def _speco_mark_collection_skip(self, reason: str) -> bool:
        self._speco_last_collection_skip_reason = reason
        return False

    def _speco_log_collection_skip_once(
        self,
        reason: str,
        *,
        collection_global_steps: Optional[int],
        request_global_steps: Optional[int],
        prompt_len: int,
        max_new_tokens: Optional[int],
        hidden_window_plan: Optional[dict[str, Any]] = None,
    ) -> None:
        logged = getattr(self, "_speco_collection_skip_log_keys", None)
        if not isinstance(logged, set):
            logged = set()
            self._speco_collection_skip_log_keys = logged
        key = (collection_global_steps, reason)
        if key in logged:
            return
        logged.add(key)

        drafter_cfg = self._speco_drafter_cfg()
        training_cfg = drafter_cfg.get("training") or {}
        hidden_window_plan = hidden_window_plan or {}
        logger.warning(
            "[SPECO SGLang] skip drafter hidden-state collection: reason=%s step=%s "
            "request_step=%s server_step=%s prompt_len=%s max_new_tokens=%s "
            "estimated_rows=%s window_mode=%s collect_interval=%s sample_rate=%s "
            "reserved_samples=%s reserved_tokens=%s",
            reason,
            collection_global_steps,
            request_global_steps,
            getattr(self, "global_steps", None),
            prompt_len,
            max_new_tokens,
            hidden_window_plan.get("estimated_rows"),
            hidden_window_plan.get("mode"),
            training_cfg.get("collect_interval_steps", 1),
            training_cfg.get("collection_sample_rate", 1.0),
            getattr(self, "_drafter_collection_samples", None),
            getattr(self, "_drafter_collection_tokens", None),
        )

    def _speco_log_missing_hidden_states_once(
        self,
        *,
        collection_global_steps: Optional[int],
        meta_info: dict[str, Any],
        hidden_states_raw_type: str,
        hidden_states_raw_len: Optional[int],
    ) -> None:
        logged = getattr(self, "_speco_hidden_missing_log_keys", None)
        if not isinstance(logged, set):
            logged = set()
            self._speco_hidden_missing_log_keys = logged
        if collection_global_steps in logged:
            return
        logged.add(collection_global_steps)

        drafter_cfg = self._speco_drafter_cfg()
        logger.warning(
            "[SGLangHttpServer] No valid hidden states returned for drafter sample collection: "
            "meta_keys=%s hidden_raw_type=%s hidden_raw_len=%s algorithm=%s",
            sorted(meta_info.keys()),
            hidden_states_raw_type,
            hidden_states_raw_len,
            drafter_cfg.get("speculative_algorithm"),
        )

    def _speco_should_collect_drafter(self, request_global_steps: Optional[int], request_id: str, estimated_hidden_rows: int) -> bool:
        self._speco_last_collection_skip_reason = None
        drafter_cfg = self._speco_drafter_cfg()
        if not bool(drafter_cfg.get("enable") and drafter_cfg.get("enable_drafter_training")):
            return self._speco_mark_collection_skip("drafter_disabled")
        training_cfg = drafter_cfg.get("training") or {}
        if not bool(training_cfg.get("collect_hidden_states_from_sgl")):
            return self._speco_mark_collection_skip("hidden_collection_disabled")
        skip_drafter_collection = bool(getattr(self, "_verl_skip_drafter_collection", False))
        if skip_drafter_collection:
            return self._speco_mark_collection_skip("request_skip_flag")
        collection_global_steps = request_global_steps if request_global_steps is not None else self.global_steps
        self._reset_drafter_collection_budget_if_needed(collection_global_steps)
        if collection_global_steps is not None and not speco_step_matches_interval(
            collection_global_steps, training_cfg.get("collect_interval_steps", 1)
        ):
            return self._speco_mark_collection_skip("interval_mismatch")
        if estimated_hidden_rows <= 0:
            return self._speco_mark_collection_skip("empty_hidden_window")
        max_samples = training_cfg.get("max_collect_samples_per_step_per_replica")
        if max_samples is not None:
            max_samples = int(max_samples)
            current_samples = int(getattr(self, "_drafter_collection_samples", 0))
            if current_samples >= max_samples:
                return self._speco_mark_collection_skip("max_samples_budget")
        max_tokens = training_cfg.get("max_collect_tokens_per_step_per_replica")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            current_tokens = int(getattr(self, "_drafter_collection_tokens", 0))
            if current_tokens + int(estimated_hidden_rows) > max_tokens:
                return self._speco_mark_collection_skip("max_tokens_budget")
        sample_rate = float(training_cfg.get("collection_sample_rate", 1.0))
        if sample_rate <= 0:
            return self._speco_mark_collection_skip("sample_rate_zero")
        if sample_rate < 1.0:
            sampling_key = f"{collection_global_steps}:{self.replica_rank}:{request_id}".encode()
            digest = hashlib.blake2b(sampling_key, digest_size=8).digest()
            sample_value = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
            if sample_value >= sample_rate:
                return self._speco_mark_collection_skip("sample_rate_rejected")
        self._drafter_collection_samples = int(getattr(self, "_drafter_collection_samples", 0)) + 1
        self._drafter_collection_tokens = int(getattr(self, "_drafter_collection_tokens", 0)) + int(estimated_hidden_rows)
        self._speco_last_collection_skip_reason = None
        return True

    def _speco_request_hidden_state_params(
        self,
        sampling_params: dict[str, Any],
        *,
        prompt_len: int,
        request_id: str,
        collection_global_steps: Optional[int],
        max_new_tokens: int,
    ) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        drafter_cfg = self._speco_drafter_cfg()
        training_cfg = drafter_cfg.get("training") or {}
        front_hidden_tokens = _positive_int_or_none(training_cfg.get("hidden_state_window_tokens_per_sample", 512))
        max_hidden_tokens = _positive_int_or_none(training_cfg.get("hidden_state_max_tokens_per_sample"))
        if _drafter_uses_dflash_aux_hidden(drafter_cfg):
            algorithm = str(drafter_cfg.get("speculative_algorithm", "") or "").strip().upper()
            max_window_key = "dspark_max_window" if algorithm == "DSPARK" else "dflash_max_window"
            dflash_max_window = _positive_int_or_none(
                training_cfg.get(max_window_key, training_cfg.get("dflash_max_window"))
            )
            if dflash_max_window is not None:
                front_hidden_tokens = (
                    min(front_hidden_tokens, dflash_max_window)
                    if front_hidden_tokens is not None
                    else dflash_max_window
                )
        hidden_window_plan = _build_drafter_hidden_window_plan(
            training_cfg=training_cfg,
            request_id=request_id,
            collection_global_steps=collection_global_steps,
            replica_rank=self.replica_rank,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            front_tokens=front_hidden_tokens,
            tail_tokens=max_hidden_tokens,
        )
        estimated_hidden_rows = int(hidden_window_plan.get("estimated_rows", 0) or 0)
        should_collect = self._speco_should_collect_drafter(collection_global_steps, request_id, estimated_hidden_rows)
        if not should_collect:
            return False, hidden_window_plan, {}

        custom_params = sampling_params.get("custom_params")
        custom_params = dict(custom_params) if isinstance(custom_params, dict) else {}
        custom_params.update(
            {
                _DRAFTER_HIDDEN_WINDOW_PARAM: True,
                _HIDDEN_STATE_PROMPT_LEN_PARAM: prompt_len,
            }
        )
        if hidden_window_plan.get("mode") == "random":
            custom_params[_HIDDEN_STATE_WINDOW_START_PARAM] = int(hidden_window_plan["window_start"])
            custom_params[_HIDDEN_STATE_WINDOW_END_PARAM] = int(hidden_window_plan["window_end"])
            custom_params[_HIDDEN_STATE_WINDOW_START_OFFSET_PARAM] = int(hidden_window_plan["window_start_offset"])
            custom_params[_HIDDEN_STATE_WINDOW_MIN_ROWS_PARAM] = int(hidden_window_plan["min_rows"])
        elif front_hidden_tokens is not None:
            custom_params[_HIDDEN_STATE_FRONT_TOKENS_PARAM] = int(front_hidden_tokens)
        if (
            _env_flag_enabled(_VERL_DRAFTER_RAW_TOP_LOGPROBS_ENV, default=False)
            and bool(training_cfg.get("use_logits", False))
            and not _drafter_uses_dflash_aux_hidden(drafter_cfg)
        ):
            custom_params[DRAFTER_RAW_TOP_LOGPROBS_PARAM] = True
        if not bool(training_cfg.get("use_logits", False)):
            if _drafter_uses_dflash_aux_hidden(drafter_cfg):
                custom_params[DFLASH_RETURN_AUX_HIDDEN_PARAM] = True
            elif _drafter_uses_eagle_last_hidden(drafter_cfg):
                custom_params[DRAFTER_RETURN_LAST_HIDDEN_PARAM] = True
        sampling_params["custom_params"] = custom_params
        return True, hidden_window_plan, custom_params

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ):
        from sglang.srt.managers.io_struct import GenerateReqInput
        from verl.workers.rollout.replica import TokenOutput
        from verl.workers.rollout.sglang_rollout.utils import SGLANG_LORA_NAME

        drafter_cfg = self._speco_drafter_cfg()
        training_cfg = drafter_cfg.get("training") or {}
        uses_dflash_aux_hidden = _drafter_uses_dflash_aux_hidden(drafter_cfg)
        if not bool(
            drafter_cfg.get("enable")
            and drafter_cfg.get("enable_drafter_training")
            and training_cfg.get("collect_hidden_states_from_sgl")
        ):
            return await super().generate(
                prompt_ids,
                self._speco_strip_internal_sampling_params(sampling_params),
                request_id,
                image_data=image_data,
                video_data=video_data,
            )

        original_sampling_params = sampling_params
        request_global_steps = original_sampling_params.get("_verl_global_steps")
        if request_global_steps is not None:
            request_global_steps = int(request_global_steps)
        skip_drafter_collection = bool(original_sampling_params.get("_verl_skip_drafter_collection", False))
        collection_global_steps = request_global_steps if request_global_steps is not None else self.global_steps
        if skip_drafter_collection or (
            collection_global_steps is not None
            and not speco_step_matches_interval(collection_global_steps, training_cfg.get("collect_interval_steps", 1))
        ):
            self._speco_log_collection_skip_once(
                "request_skip_flag" if skip_drafter_collection else "interval_mismatch",
                collection_global_steps=collection_global_steps,
                request_global_steps=request_global_steps,
                prompt_len=len(prompt_ids),
                max_new_tokens=None,
                hidden_window_plan=None,
            )
            return await super().generate(
                prompt_ids,
                self._speco_strip_internal_sampling_params(original_sampling_params),
                request_id,
                image_data=image_data,
                video_data=video_data,
            )

        sampling_params = self._speco_strip_internal_sampling_params(original_sampling_params)

        max_possible_tokens = self.config.max_model_len - len(prompt_ids) - 1
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = min(self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids))
        max_new_tokens = max(0, min(int(max_new_tokens), max_possible_tokens))
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
        }
        if self.config.enable_rollout_routing_replay:
            request["return_routed_experts"] = True

        should_collect = False
        hidden_window_plan = {"mode": "front", "estimated_rows": 0, "min_rows": 0}
        should_collect, hidden_window_plan, _ = self._speco_request_hidden_state_params(
            sampling_params,
            prompt_len=len(prompt_ids),
            request_id=request_id,
            collection_global_steps=collection_global_steps,
            max_new_tokens=max_new_tokens,
        )
        if not should_collect:
            self._speco_log_collection_skip_once(
                getattr(self, "_speco_last_collection_skip_reason", "unknown"),
                collection_global_steps=collection_global_steps,
                request_global_steps=request_global_steps,
                prompt_len=len(prompt_ids),
                max_new_tokens=max_new_tokens,
                hidden_window_plan=hidden_window_plan,
            )
            return await super().generate(
                prompt_ids,
                self._speco_strip_internal_sampling_params(original_sampling_params),
                request_id,
                image_data=image_data,
                video_data=video_data,
            )
        request["return_hidden_states"] = True

        generate_request = GenerateReqInput(**request)
        if self.model_config.lora_rank > 0:
            generate_request.lora_path = SGLANG_LORA_NAME

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        meta_info = output.get("meta_info", {})
        finish_reason = meta_info.get("finish_reason")
        finish_reason = finish_reason["type"] if finish_reason else None

        token_ids = list(output.get("output_ids", []))
        if return_logprob:
            output_token_logprobs = meta_info.get("output_token_logprobs") or []
            if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
                log_probs = [float(log_prob) for log_prob, _, _ in output_token_logprobs]
            else:
                assert not token_ids, (
                    f"output_token_logprobs length ({len(output_token_logprobs)}) != "
                    f"output_ids length ({len(token_ids)}) for request {request_id}"
                )
                log_probs = []
        else:
            log_probs = None

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = output.get("meta_info", {}).get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        drafter_sample = None
        if should_collect:
            collect_target_logprobs = bool(training_cfg.get("use_logits", False)) and not uses_dflash_aux_hidden
            hidden_states_data = meta_info.pop("hidden_states", [])
            hidden_states_raw_type = type(hidden_states_data).__name__
            try:
                hidden_states_raw_len = len(hidden_states_data)
            except TypeError:
                hidden_states_raw_len = None
            hidden_states_list = []
            hidden_states_metadata = []
            for hs in _iter_hidden_state_chunks(hidden_states_data):
                h_states, metadata = _hidden_state_chunk_to_tensor_and_metadata(hs)
                if h_states is not None:
                    hidden_states_list.append(h_states)
                    if metadata:
                        hidden_states_metadata.append(metadata)
            has_hidden_states = bool(hidden_states_list)
            hidden_window_mode = str(hidden_window_plan.get("mode", "front") or "front")
            hidden_window_start = hidden_window_plan.get("window_start")
            hidden_window_end = hidden_window_plan.get("window_end")
            hidden_window_start_offset = hidden_window_plan.get("window_start_offset")
            hidden_window_min_rows = int(hidden_window_plan.get("min_rows", 0) or 0)
            hidden_window_target_rows = hidden_window_plan.get("target_window_rows")
            hidden_raw_len = 0
            hidden_kept_len = 0
            hidden_position_start = 0
            hidden_position_end = 0
            hidden_prefix_cache_rows = 0
            hidden_positions = None
            target_logprobs = None
            target_logprobs_position_start = None
            target_logprobs_position_end = None
            hidden_last_hidden_logprob_check = None
            hidden_target_logprobs_source = None
            hidden_raw_topk_logprob_check = None
            hidden_raw_target_logprobs = None
            hidden_raw_target_logprobs_positions = None
            hidden_raw_target_logprobs_position_start = None
            hidden_raw_target_logprobs_position_end = None
            hidden_last_hidden_filter = None
            hidden_last_hidden_select = None

            if has_hidden_states:
                prompt_tensor = torch.as_tensor(prompt_ids, dtype=torch.long).detach().cpu()
                response_tensor = torch.tensor(token_ids, dtype=torch.long)
                hidden_states = torch.cat(hidden_states_list, dim=0)
                hidden_raw_len = int(hidden_states.size(0))
                if hidden_states_metadata:
                    first_metadata = hidden_states_metadata[0]
                    last_metadata = hidden_states_metadata[-1]
                    position_chunks = [
                        metadata.get("positions")
                        for metadata in hidden_states_metadata
                        if torch.is_tensor(metadata.get("positions"))
                    ]
                    if position_chunks:
                        hidden_positions = torch.cat(position_chunks, dim=0).to(dtype=torch.long)
                        if int(hidden_positions.numel()) != hidden_raw_len:
                            hidden_positions = None
                    hidden_position_start = int(first_metadata.get("position_start", 0))
                    hidden_position_end = int(last_metadata.get("position_end", hidden_position_start + hidden_raw_len))
                    if hidden_positions is not None and int(hidden_positions.numel()) > 0:
                        hidden_position_start = int(hidden_positions[0].item())
                        hidden_position_end = int(hidden_positions[-1].item()) + 1
                    hidden_prefix_cache_rows = int(first_metadata.get("prefix_cache_rows", 0))
                    hidden_window_start = first_metadata.get("window_start", hidden_window_start)
                    hidden_window_end = first_metadata.get("window_end", hidden_window_end)
                    hidden_last_hidden_logprob_check = next(
                        (
                            metadata.get("last_hidden_logprob_check")
                            for metadata in reversed(hidden_states_metadata)
                            if metadata.get("last_hidden_logprob_check") is not None
                        ),
                        None,
                    )
                    hidden_target_logprobs_source = next(
                        (
                            metadata.get("target_logprobs_source")
                            for metadata in reversed(hidden_states_metadata)
                            if metadata.get("target_logprobs_source") is not None
                        ),
                        None,
                    )
                    hidden_raw_topk_logprob_check = next(
                        (
                            metadata.get("raw_topk_logprob_check")
                            for metadata in reversed(hidden_states_metadata)
                            if metadata.get("raw_topk_logprob_check") is not None
                        ),
                        None,
                    )
                    raw_target_logprob_chunks = [
                        metadata.get("raw_target_logprobs")
                        for metadata in hidden_states_metadata
                        if torch.is_tensor(metadata.get("raw_target_logprobs"))
                    ]
                    if raw_target_logprob_chunks:
                        hidden_raw_target_logprobs = torch.cat(raw_target_logprob_chunks, dim=0).contiguous()
                        raw_target_rows = int(hidden_raw_target_logprobs.size(0))
                        raw_target_position_chunks = []
                        if any(torch.is_tensor(metadata.get("raw_target_logprobs_positions")) for metadata in hidden_states_metadata):
                            for metadata in hidden_states_metadata:
                                raw_chunk = metadata.get("raw_target_logprobs")
                                if not torch.is_tensor(raw_chunk):
                                    continue
                                raw_position_chunk = metadata.get("raw_target_logprobs_positions")
                                if torch.is_tensor(raw_position_chunk) and int(raw_position_chunk.numel()) == int(raw_chunk.size(0)):
                                    raw_target_position_chunks.append(raw_position_chunk.reshape(-1).to(dtype=torch.long))
                                else:
                                    raw_target_position_chunks.append(torch.full((int(raw_chunk.size(0)),), -1, dtype=torch.long))
                        if raw_target_position_chunks:
                            hidden_raw_target_logprobs_positions = torch.cat(raw_target_position_chunks, dim=0).contiguous()
                        if torch.is_tensor(hidden_raw_target_logprobs_positions):
                            valid_raw_positions = hidden_raw_target_logprobs_positions[hidden_raw_target_logprobs_positions >= 0]
                            if int(valid_raw_positions.numel()) > 0:
                                hidden_raw_target_logprobs_position_start = int(valid_raw_positions[0].item())
                                hidden_raw_target_logprobs_position_end = int(valid_raw_positions[-1].item()) + 1
                        if hidden_raw_target_logprobs_position_start is None and not collect_target_logprobs:
                            hidden_raw_target_logprobs_position_start = int(hidden_position_start)
                            hidden_raw_target_logprobs_position_end = hidden_raw_target_logprobs_position_start + raw_target_rows
                        if torch.is_tensor(hidden_raw_target_logprobs_positions):
                            target_window_start = int(hidden_position_start) + 1
                            target_window_end = min(int(hidden_position_end), max(len(prompt_ids) - 1, 0) + len(token_ids))
                            target_logprobs, target_logprobs_position_start, target_logprobs_position_end, _ = _select_target_logprobs_by_raw_positions(
                                hidden_raw_target_logprobs,
                                hidden_raw_target_logprobs_positions,
                                desired_position_start=target_window_start,
                                desired_position_end=target_window_end,
                            )
                    hidden_last_hidden_filter = next(
                        (
                            metadata.get("last_hidden_filter")
                            for metadata in reversed(hidden_states_metadata)
                            if metadata.get("last_hidden_filter") is not None
                        ),
                        None,
                    )
                    hidden_last_hidden_select = next(
                        (
                            metadata.get("last_hidden_select")
                            for metadata in reversed(hidden_states_metadata)
                            if metadata.get("last_hidden_select") is not None
                        ),
                        None,
                    )
                else:
                    hidden_states, hidden_position_start, hidden_position_end, _, hidden_prefix_cache_rows = _select_drafter_hidden_state_window(
                        hidden_states,
                        expected_hidden_rows=_expected_full_hidden_rows(len(prompt_ids), len(token_ids)),
                        prompt_len=len(prompt_ids),
                        front_tokens=_positive_int_or_none(training_cfg.get("hidden_state_window_tokens_per_sample", 512)),
                        tail_tokens=_positive_int_or_none(training_cfg.get("hidden_state_max_tokens_per_sample")),
                )
                hidden_kept_len = int(hidden_states.size(0))
                fail_closed_alignment_reason = None
                if not uses_dflash_aux_hidden:
                    if hidden_positions is None or int(hidden_positions.numel()) != hidden_kept_len:
                        fail_closed_alignment_reason = "hidden_positions_missing_or_mismatched"
                    else:
                        aligned_prefix_rows = hidden_kept_len
                        hidden_contiguous = hidden_positions[1:] == hidden_positions[:-1] + 1
                        if hidden_contiguous.numel() > 0 and not bool(hidden_contiguous.all()):
                            first_break = int(
                                torch.nonzero(~hidden_contiguous, as_tuple=False).reshape(-1)[0].item()
                            ) + 1
                            aligned_prefix_rows = min(aligned_prefix_rows, first_break)

                        if collect_target_logprobs:
                            if not (
                                torch.is_tensor(hidden_raw_target_logprobs)
                                and torch.is_tensor(hidden_raw_target_logprobs_positions)
                            ):
                                aligned_prefix_rows = 0
                            else:
                                raw_positions = hidden_raw_target_logprobs_positions.reshape(-1).to(dtype=torch.long)
                                aligned_prefix_rows = min(
                                    aligned_prefix_rows,
                                    int(hidden_raw_target_logprobs.size(0)),
                                    int(raw_positions.numel()),
                                )
                                raw_contiguous = (
                                    raw_positions[1:aligned_prefix_rows]
                                    == raw_positions[: max(aligned_prefix_rows - 1, 0)] + 1
                                )
                                if raw_contiguous.numel() > 0 and not bool(raw_contiguous.all()):
                                    first_raw_break = int(
                                        torch.nonzero(~raw_contiguous, as_tuple=False).reshape(-1)[0].item()
                                    ) + 1
                                    aligned_prefix_rows = min(aligned_prefix_rows, first_raw_break)
                                # hidden position p produces the teacher row for token p + 1.
                                positions_match = (
                                    raw_positions[:aligned_prefix_rows]
                                    == hidden_positions[:aligned_prefix_rows] + 1
                                )
                                if positions_match.numel() > 0 and not bool(positions_match.all()):
                                    first_mismatch = int(
                                        torch.nonzero(~positions_match, as_tuple=False).reshape(-1)[0].item()
                                    )
                                    aligned_prefix_rows = min(aligned_prefix_rows, first_mismatch)

                        if aligned_prefix_rows <= 0:
                            fail_closed_alignment_reason = "no_aligned_continuous_prefix"
                        elif aligned_prefix_rows < hidden_kept_len:
                            logger.warning(
                                "[SpeCo SGLang] Truncate drafter sample to aligned prefix: kept=%s/%s",
                                aligned_prefix_rows,
                                hidden_kept_len,
                            )
                            hidden_states = hidden_states[:aligned_prefix_rows]
                            hidden_positions = hidden_positions[:aligned_prefix_rows]
                            hidden_kept_len = aligned_prefix_rows
                            hidden_position_start = int(hidden_positions[0].item())
                            hidden_position_end = int(hidden_positions[-1].item()) + 1
                            if torch.is_tensor(hidden_raw_target_logprobs):
                                hidden_raw_target_logprobs = hidden_raw_target_logprobs[:aligned_prefix_rows]
                            if torch.is_tensor(hidden_raw_target_logprobs_positions):
                                hidden_raw_target_logprobs_positions = hidden_raw_target_logprobs_positions[
                                    :aligned_prefix_rows
                                ]

                if fail_closed_alignment_reason is not None:
                    logger.warning(
                        "[SpeCo SGLang] Drop misaligned drafter sample: reason=%s hidden_rows=%s "
                        "hidden_positions=%s raw_rows=%s raw_positions=%s",
                        fail_closed_alignment_reason,
                        hidden_kept_len,
                        int(hidden_positions.numel()) if torch.is_tensor(hidden_positions) else None,
                        (
                            int(hidden_raw_target_logprobs.size(0))
                            if torch.is_tensor(hidden_raw_target_logprobs)
                            else None
                        ),
                        (
                            int(hidden_raw_target_logprobs_positions.numel())
                            if torch.is_tensor(hidden_raw_target_logprobs_positions)
                            else None
                        ),
                    )
                if collect_target_logprobs and hidden_raw_target_logprobs is not None:
                    target_window_start = int(hidden_position_start) + 1
                    target_window_end = min(int(hidden_position_end), max(len(prompt_ids) - 1, 0) + len(token_ids))
                    if torch.is_tensor(hidden_raw_target_logprobs_positions):
                        target_logprobs, target_logprobs_position_start, target_logprobs_position_end, _ = _select_target_logprobs_by_raw_positions(
                            hidden_raw_target_logprobs,
                            hidden_raw_target_logprobs_positions,
                            desired_position_start=target_window_start,
                            desired_position_end=target_window_end,
                        )
                if fail_closed_alignment_reason is not None:
                    drafter_sample = None
                elif hidden_window_mode == "random" and hidden_kept_len < hidden_window_min_rows:
                    drafter_sample = None
                else:
                    drafter_sample = {
                        "input_ids": torch.cat([prompt_tensor, response_tensor], dim=0).unsqueeze(0),
                        "prompts": prompt_tensor.unsqueeze(0),
                        "responses": response_tensor.unsqueeze(0),
                        "hidden_states": hidden_states.unsqueeze(0).cpu(),
                        "hidden_positions": hidden_positions.unsqueeze(0).cpu() if hidden_positions is not None else None,
                        "hidden_position_start": hidden_position_start,
                        "hidden_position_end": hidden_position_end,
                        "hidden_prefix_cache_rows": hidden_prefix_cache_rows,
                        "hidden_window_mode": hidden_window_mode,
                        "hidden_window_start": hidden_window_start,
                        "hidden_window_end": hidden_window_end,
                        "hidden_window_start_offset": hidden_window_start_offset,
                        "hidden_window_min_rows": hidden_window_min_rows,
                        "hidden_window_target_rows": hidden_window_target_rows,
                        "hidden_lm_head_fingerprint": next(
                            (
                                metadata.get("lm_head_fingerprint")
                                for metadata in hidden_states_metadata
                                if metadata.get("lm_head_fingerprint") is not None
                            ),
                            None,
                        ),
                        "hidden_last_hidden_logprob_check": hidden_last_hidden_logprob_check,
                        "hidden_target_logprobs_source": hidden_target_logprobs_source,
                        "hidden_raw_topk_logprob_check": hidden_raw_topk_logprob_check,
                        "hidden_raw_target_logprobs": (
                            hidden_raw_target_logprobs.unsqueeze(0).cpu() if torch.is_tensor(hidden_raw_target_logprobs) else None
                        ),
                        "hidden_raw_target_logprobs_positions": (
                            hidden_raw_target_logprobs_positions.unsqueeze(0).cpu()
                            if torch.is_tensor(hidden_raw_target_logprobs_positions)
                            else None
                        ),
                        "hidden_raw_target_logprobs_position_start": hidden_raw_target_logprobs_position_start,
                        "hidden_raw_target_logprobs_position_end": hidden_raw_target_logprobs_position_end,
                        "hidden_last_hidden_filter": hidden_last_hidden_filter,
                        "hidden_last_hidden_select": hidden_last_hidden_select,
                        "target_logprobs": target_logprobs.unsqueeze(0).cpu() if target_logprobs is not None else None,
                        "target_logprobs_position_start": target_logprobs_position_start,
                        "target_logprobs_position_end": target_logprobs_position_end,
                        "global_step": collection_global_steps,
                        "replica_rank": self.replica_rank,
                    }
            else:
                self._speco_log_missing_hidden_states_once(
                    collection_global_steps=collection_global_steps,
                    meta_info=meta_info,
                    hidden_states_raw_type=hidden_states_raw_type,
                    hidden_states_raw_len=hidden_states_raw_len,
                )
            extra_fields = {
                "global_steps": collection_global_steps,
                "drafter_sample": drafter_sample,
            }
            output = TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                routed_experts=routed_experts,
                stop_reason=finish_reason,
                extra_fields=extra_fields,
            )
            return output

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=finish_reason,
            extra_fields={"global_steps": collection_global_steps},
        )


def _build_speco_http_server_class(upstream_module):
    upstream_cls = upstream_module.SGLangHttpServer
    if issubclass(upstream_cls, _SpecoSGLangHttpServerMixin):
        return upstream_cls
    return type("SpecoSGLangHttpServer", (_SpecoSGLangHttpServerMixin, upstream_cls), {"__module__": __name__})


def _build_speco_replica_class(upstream_module):
    import ray
    from verl.utils.device import get_visible_devices_keyword
    from verl.utils.net_utils import is_valid_ipv6_address

    upstream_replica = upstream_module.SGLangReplica
    visible_devices_keyword = get_visible_devices_keyword()
    speco_http_server_cls = _build_speco_http_server_class(upstream_module)

    class SpecoSGLangReplica(upstream_replica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(speco_http_server_cls)

        async def launch_servers(self):
            assert len(self.workers) == self.world_size, (
                f"worker number {len(self.workers)} not equal to world size {self.world_size}"
            )
            worker_infos = await asyncio.gather(
                *[
                    worker.__ray_call__.remote(
                        lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                    )
                    for worker in self.workers
                ]
            )
            worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
            worker_node_ids = [worker_info[0] for worker_info in worker_infos]
            base_gpu_id = 0
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            replica_world_size = infer_tp * self.config.pipeline_model_parallel_size
            if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
                base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node

            drafter_env = os.getenv(SPECO_SGLANG_DRAFTER_CONFIG_ENV, "")
            for node_rank in range(self.nnodes):
                node_cuda_visible_devices_set = worker_cuda_visible_devices[
                    node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
                ]
                node_cuda_visible_devices = ",".join(
                    map(
                        str,
                        sorted(
                            set(
                                int(device)
                                for worker_devices_set in node_cuda_visible_devices_set
                                for device in worker_devices_set.split(",")
                                if device.strip()
                            )
                        ),
                    )
                )
                node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
                if self.is_reward_model:
                    name = f"sglang_server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
                elif self.is_teacher_model:
                    name = f"sglang_server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
                else:
                    name = f"sglang_server_{self.replica_rank}_{node_rank}{self.name_suffix}"

                env_vars = {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}
                if drafter_env:
                    env_vars[SPECO_SGLANG_DRAFTER_CONFIG_ENV] = drafter_env
                server = self.server_class.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                    runtime_env={"env_vars": env_vars},
                    name=name,
                    max_concurrency=self.max_concurrency,
                ).remote(
                    config=self.config,
                    model_config=self.model_config,
                    rollout_mode=self.rollout_mode,
                    workers=self.workers[
                        node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
                    ],
                    replica_rank=self.replica_rank,
                    node_rank=node_rank,
                    nnodes=self.nnodes,
                    cuda_visible_devices=node_cuda_visible_devices,
                    base_gpu_id=base_gpu_id,
                )
                self.servers.append(server)

            master_address, master_port = None, None
            if self.nnodes > 1:
                master_address, master_port = await self.servers[0].get_master_address.remote()
            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=master_address, master_port=master_port)
                    for server in self.servers
                ]
            )
            server_address, server_port = await self.servers[0].get_server_address.remote()
            self._server_handle = self.servers[0]
            self._server_address = (
                f"[{server_address}]:{server_port}"
                if is_valid_ipv6_address(server_address)
                else f"{server_address}:{server_port}"
            )

    SpecoSGLangReplica.__name__ = "SpecoSGLangReplica"
    SpecoSGLangReplica.__qualname__ = "SpecoSGLangReplica"
    SpecoSGLangReplica.__module__ = __name__
    return SpecoSGLangReplica


def should_install_sglang_base_compat_runtime(config: Any) -> bool:
    rollout_name = _get_nested(config, ("actor_rollout_ref", "rollout", "name"), None)
    return rollout_name == "sglang" and sglang_needs_qwen3_rope_compat_patch()


def install_upstream_sglang_runtime_bridge(*, base_compat_only: bool = False) -> bool:
    """Patch upstream verl v0.8.0 SGLang rollout classes in the current process."""

    global _SGLANG_REPLICA_PATCHED
    if _SGLANG_REPLICA_PATCHED:
        return True
    if not base_compat_only and not os.getenv(SPECO_SGLANG_DRAFTER_CONFIG_ENV):
        return False

    try:
        from verl.workers.rollout import replica as replica_module
        from verl.workers.rollout.sglang_rollout import async_sglang_server
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to install SPECO SGLang runtime bridge: %s", exc)
        return False

    speco_replica = _build_speco_replica_class(async_sglang_server)
    async_sglang_server.SGLangReplica = speco_replica
    registry = getattr(replica_module, "RolloutReplicaRegistry", None)
    if registry is not None and hasattr(registry, "_registry"):
        registry._registry["sglang"] = lambda: speco_replica
    patch_sglang_server_adapter_update()
    _SGLANG_REPLICA_PATCHED = True
    return True
