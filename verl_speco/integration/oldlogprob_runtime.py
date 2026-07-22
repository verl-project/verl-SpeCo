"""Runtime patch for SPECO old-logprob hidden-state collection.

The SPECO package must not edit the upstream ``verl`` source tree. This module
therefore installs a narrow in-process patch on the upstream FSDP LM-head engine
inside actor worker processes when old-logprob collection is explicitly enabled.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from functools import wraps
from typing import Any, cast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_SGLANG_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"

OLD_LOGPROB_COLLECT_MASK_KEY = "speco_oldlogprob_collect_mask"
OLD_LOGPROB_HIDDEN_POSITIONS_KEY = "speco_oldlogprob_hidden_positions"
OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY = "speco_oldlogprob_hidden_position_mask"
OLD_LOGPROB_OWNER_RANK_KEY = "speco_oldlogprob_owner_rank"
OLD_LOGPROB_HIDDEN_STATES_KEY = "speco_oldlogprob_hidden_states"
OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY = "speco_oldlogprob_hidden_object_ref"
OLD_LOGPROB_HIDDEN_REFS_KEY = "speco_oldlogprob_hidden_refs"
OLD_LOGPROB_HIDDEN_REF_META_KEY = "speco_oldlogprob_hidden_ref_meta"
OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY = "speco_oldlogprob_hidden_chunk_refs"
OLD_LOGPROB_HIDDEN_CHUNK_META_KEY = "speco_oldlogprob_hidden_chunk_meta"
OLD_LOGPROB_AUX_LAYER_IDS_KEY = "speco_oldlogprob_aux_layer_ids"
OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY = "speco_oldlogprob_hidden_capture_impl"
OLD_LOGPROB_HIDDEN_LAYOUT_KEY = "speco_oldlogprob_hidden_layout"
OLD_LOGPROB_TIMING_KEY = "speco_oldlogprob_timing"
OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY = "speco_oldlogprob_selected_batch_indices"

_TIMING_SELECT_US = 0
_TIMING_SP_MERGE_US = 1
_TIMING_CONCAT_US = 2
_TIMING_CPU_COPY_US = 3
_TIMING_RAY_PUT_US = 4
_TIMING_WIDTH = 5

_PATCHED = False
_BATCH_POSTPROCESS_PATCHED = False
_POSTPROCESS_PATCHED = False


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


def _load_drafter_env(raw: str | None = None) -> dict[str, Any]:
    raw = raw if raw is not None else os.getenv(SPECO_SGLANG_DRAFTER_CONFIG_ENV, "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid %s while checking old-logprob hidden runtime",
            SPECO_SGLANG_DRAFTER_CONFIG_ENV,
        )
        return {}
    return value if isinstance(value, dict) else {}


def _oldlogprob_enabled_from_drafter(drafter: Any) -> bool:
    if not bool(_get_nested(drafter, ("enable",), False)):
        return False
    if not bool(_get_nested(drafter, ("enable_drafter_training",), False)):
        return False
    training = _get_nested(drafter, ("training",), {}) or {}
    return bool(
        _get_nested(training, ("collect_hidden_states_from_old_logprob",), False)
    )


def oldlogprob_hidden_runtime_enabled(
    config: Any = None, *, drafter_env: str | None = None
) -> bool:
    """Return whether an actor worker should install the old-logprob runtime patch."""

    if _oldlogprob_enabled_from_drafter(_load_drafter_env(drafter_env)):
        return True

    drafter = _get_nested(config, ("rollout", "drafter"), None)
    if _oldlogprob_enabled_from_drafter(drafter):
        return True

    drafter = _get_nested(config, ("actor_rollout_ref", "rollout", "drafter"), None)
    return _oldlogprob_enabled_from_drafter(drafter)


def _tensor_key_present(container: Any, key: str) -> bool:
    try:
        return key in container.keys()
    except Exception:  # noqa: BLE001
        try:
            return key in container
        except Exception:  # noqa: BLE001
            return False


def _resolve_hidden_state(hidden_states: Any, layer_id: int):
    if hidden_states is None:
        return None
    num_states = len(hidden_states)
    index = int(layer_id)
    if index >= 0:
        # Config layer ids refer to transformer layer outputs. HF hidden_states
        # includes embeddings at index 0, so layer 0 is hidden_states[1].
        index += 1
    if index < 0:
        index = num_states + index
    if index < 0 or index >= num_states:
        raise IndexError(
            f"SPECO old-logprob hidden layer id {layer_id} resolved to index {index}, "
            f"but model returned {num_states} hidden states"
        )
    hidden = hidden_states[index]
    return hidden.squeeze(0) if hidden.dim() == 3 and hidden.size(0) == 1 else hidden


def _flat_local_range(
    total_tokens: int, pad_size: int, sp_size: int, sp_rank: int
) -> tuple[int, int]:
    if sp_size <= 1:
        return 0, int(total_tokens)
    total_padded = int(total_tokens) + int(pad_size)
    chunk = total_padded // int(sp_size)
    start = int(sp_rank) * chunk
    return start, start + chunk


def _oldlogprob_capture_impl(micro_batch: Any) -> str:
    value = "forward_hook"
    try:
        value = micro_batch.get(OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY, value)
    except Exception:  # noqa: BLE001
        if _tensor_key_present(micro_batch, OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY):
            value = micro_batch[OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY]
    value = getattr(value, "data", value)
    return str(value or "forward_hook")


def _oldlogprob_hidden_layout(micro_batch: Any) -> str:
    value = "eagle3_aux_plus_last"
    try:
        value = micro_batch.get(OLD_LOGPROB_HIDDEN_LAYOUT_KEY, value)
    except Exception:  # noqa: BLE001
        if _tensor_key_present(micro_batch, OLD_LOGPROB_HIDDEN_LAYOUT_KEY):
            value = micro_batch[OLD_LOGPROB_HIDDEN_LAYOUT_KEY]
    value = getattr(value, "data", value)
    layout = str(value or "eagle3_aux_plus_last")
    if layout not in {"eagle3_aux_plus_last", "dflash_aux", "dflash_aux_plus_last"}:
        raise ValueError(f"Unsupported SPECO old-logprob hidden layout: {layout!r}")
    return layout


def _oldlogprob_hidden_object_ref_enabled(micro_batch: Any) -> bool:
    if not (_BATCH_POSTPROCESS_PATCHED and _POSTPROCESS_PATCHED):
        return False

    value = False
    try:
        from verl.utils import tensordict_utils as tu

        value = tu.get_non_tensor_data(
            data=micro_batch, key=OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, default=False
        )
    except Exception:  # noqa: BLE001
        try:
            value = micro_batch.get(OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, False)
        except Exception:  # noqa: BLE001
            if _tensor_key_present(micro_batch, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY):
                value = micro_batch[OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY]
    value = getattr(value, "data", value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "n"}
    return bool(value)


def _is_sparse_sp_non_source_context(context: dict[str, Any]) -> bool:
    return (
        bool(context.get("sparse_sp_merge"))
        and int(context.get("sp_rank", 0) or 0) != 0
    )


def _is_sparse_selected(value: Any) -> bool:
    return isinstance(value, dict) and {
        "rows",
        "batch_indices",
        "row_indices",
    }.issubset(value)


def _selected_device(selected: Any):
    if _is_sparse_selected(selected):
        return selected["rows"].device
    return selected.device


def _row_indices_payload(row_indices: Any):
    if row_indices is None:
        return None
    try:
        import torch

        if torch.is_tensor(row_indices):
            return [int(idx) for idx in row_indices.detach().cpu().reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(row_indices, (list, tuple)):
        return [int(idx) for idx in row_indices]
    return None


def _sparse_rows_cover_full_window(row_indices: list[int], valid_rows: int) -> bool:
    return len(row_indices) == int(valid_rows) and row_indices == list(
        range(int(valid_rows))
    )


def _put_oldlogprob_hidden_refs(
    hidden_output: dict[str, Any], micro_batch: Any
) -> dict[str, Any]:
    if not _oldlogprob_hidden_object_ref_enabled(micro_batch):
        return hidden_output

    selected = hidden_output.get(OLD_LOGPROB_HIDDEN_STATES_KEY)
    sp_size = int(hidden_output.get("speco_oldlogprob_sp_size", 1) or 1)
    sp_rank = int(hidden_output.get("speco_oldlogprob_sp_rank", 0) or 0)
    if sp_size > 1 and sp_rank != 0:
        # Only the SP source rank is returned by the upstream worker dispatch.
        # Creating ObjectRefs on non-returned SP ranks wastes object-store memory,
        # and passing ObjectRefs through torch collectives loses Ray owner metadata.
        return hidden_output
    try:
        import ray
        import torch
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to export SPECO old-logprob hidden ObjectRefs: %s", exc)
        return hidden_output

    selected_is_sparse = _is_sparse_selected(selected)
    if not torch.is_tensor(selected) and not selected_is_sparse:
        return hidden_output
    if selected_is_sparse:
        selected_sparse = cast(dict[str, Any], selected)
        selected_tensor: Any = None
    else:
        selected_sparse = {}
        selected_tensor = cast(Any, selected)

    copy_us = 0.0
    ray_put_us = 0.0
    try:
        collect_mask = micro_batch[OLD_LOGPROB_COLLECT_MASK_KEY].bool().detach().cpu()
        position_mask = (
            micro_batch[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY].bool().detach().cpu()
        )
        owner_rank = micro_batch[OLD_LOGPROB_OWNER_RANK_KEY].long().detach().cpu()
        selected_batch_indices = hidden_output.get(
            OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY
        )
        if torch.is_tensor(selected_batch_indices):
            selected_batch_indices = cast(Any, selected_batch_indices)
            selected_batch_indices = [
                int(idx)
                for idx in selected_batch_indices.detach().cpu().reshape(-1).tolist()
            ]
        elif isinstance(selected_batch_indices, (list, tuple)):
            selected_batch_indices = [int(idx) for idx in selected_batch_indices]
        else:
            selected_batch_indices = None
        rows = (
            []
            if selected_is_sparse
            else (
                list(selected_tensor.unbind(0))
                if selected_tensor.dim() >= 3
                else [selected_tensor]
            )
        )
        if (
            not selected_is_sparse
            and selected_batch_indices is not None
            and len(selected_batch_indices) != len(rows)
        ):
            raise RuntimeError(
                "SPECO old-logprob compact hidden refs got mismatched selected batch indices: "
                f"indices={len(selected_batch_indices)} rows={len(rows)}"
            )
        owner_mask = hidden_output.get("speco_oldlogprob_owner_mask")
        owner_mask_cpu = (
            cast(Any, owner_mask).detach().bool().cpu()
            if torch.is_tensor(owner_mask)
            else None
        )
        sample_count = int(collect_mask.numel())
        refs: list[Any] = [None for _ in range(sample_count)]
        metas: list[Any] = [None for _ in range(sample_count)]
        owner_chunks: dict[int, list[dict[str, Any]]] = {}
        if selected_is_sparse:
            sparse_rows = selected_sparse["rows"]
            copy_started = time.perf_counter()
            sparse_rows_cpu = (
                sparse_rows.detach().to(device="cpu", copy=True).contiguous()
            )
            copy_us += (time.perf_counter() - copy_started) * 1_000_000.0
            sparse_batch_indices = (
                selected_sparse["batch_indices"].detach().cpu().reshape(-1).tolist()
            )
            sparse_row_indices = (
                selected_sparse["row_indices"].detach().cpu().reshape(-1).tolist()
            )
            hidden_dim = (
                int(sparse_rows_cpu.shape[-1]) if sparse_rows_cpu.dim() > 1 else 0
            )
            element_size = int(sparse_rows_cpu.element_size())
            owner_sample_chunks: dict[int, dict[int, dict[str, Any]]] = {}
            for source_idx, (batch_idx, row_idx) in enumerate(
                zip(sparse_batch_indices, sparse_row_indices, strict=False)
            ):
                batch_idx = int(batch_idx)
                row_idx = int(row_idx)
                if batch_idx >= sample_count or not bool(
                    collect_mask[batch_idx].item()
                ):
                    continue
                valid_rows = (
                    int(position_mask[batch_idx].sum().item())
                    if batch_idx < int(position_mask.size(0))
                    else 0
                )
                if valid_rows <= 0 or row_idx < 0 or row_idx >= valid_rows:
                    continue
                owner = (
                    int(owner_rank[batch_idx].item())
                    if batch_idx < int(owner_rank.numel())
                    else 0
                )
                sample_chunks = owner_sample_chunks.setdefault(owner, {})
                default_chunk: dict[str, Any] = {
                    "batch_idx": batch_idx,
                    "source_indices": [],
                    "valid_rows": int(valid_rows),
                    "row_indices": [],
                }
                chunk = sample_chunks.setdefault(batch_idx, default_chunk)
                chunk["source_indices"].append(int(source_idx))
                chunk["row_indices"].append(row_idx)
                metas[batch_idx] = {
                    "shape": (1, int(valid_rows), int(hidden_dim)),
                    "dtype": str(sparse_rows_cpu.dtype),
                    "nbytes": int(valid_rows) * int(hidden_dim) * element_size,
                    "rows": int(valid_rows),
                }
            for owner, sample_chunks in owner_sample_chunks.items():
                for chunk in sample_chunks.values():
                    if not chunk["source_indices"]:
                        continue
                    sorted_pairs = sorted(
                        zip(
                            chunk["row_indices"],
                            chunk.pop("source_indices"),
                            strict=False,
                        )
                    )
                    chunk["row_indices"] = [
                        row_idx for row_idx, _source_idx in sorted_pairs
                    ]
                    source_indices = torch.tensor(
                        [source_idx for _row_idx, source_idx in sorted_pairs],
                        dtype=torch.long,
                    )
                    chunk["hidden"] = sparse_rows_cpu.index_select(
                        0, source_indices
                    ).contiguous()
                    owner_chunks.setdefault(owner, []).append(chunk)
        else:
            for row_idx, hidden in enumerate(rows):
                batch_idx = (
                    selected_batch_indices[row_idx]
                    if selected_batch_indices is not None
                    else row_idx
                )
                if batch_idx >= sample_count or not bool(
                    collect_mask[batch_idx].item()
                ):
                    continue

                valid_rows = (
                    int(position_mask[batch_idx].sum().item())
                    if batch_idx < int(position_mask.size(0))
                    else 0
                )
                if valid_rows <= 0:
                    continue

                hidden = hidden[:valid_rows]
                row_indices = None
                if owner_mask_cpu is not None:
                    owner_mask_row = owner_mask_cpu[row_idx, :valid_rows]
                    if not bool(owner_mask_row.all().item()):
                        row_indices = owner_mask_row.nonzero(as_tuple=False).reshape(-1)
                        if int(row_indices.numel()) <= 0:
                            continue
                        hidden = hidden.index_select(
                            0, row_indices.to(device=hidden.device)
                        )
                        row_indices = row_indices.to(dtype=torch.long)

                copy_started = time.perf_counter()
                hidden_cpu = hidden.detach().to(device="cpu", copy=True).contiguous()
                copy_us += (time.perf_counter() - copy_started) * 1_000_000.0
                owner = (
                    int(owner_rank[batch_idx].item())
                    if batch_idx < int(owner_rank.numel())
                    else 0
                )
                owner_chunks.setdefault(owner, []).append(
                    {
                        "batch_idx": int(batch_idx),
                        "hidden": hidden_cpu,
                        "valid_rows": int(valid_rows),
                        "row_indices": row_indices,
                    }
                )
                metas[batch_idx] = {
                    "shape": (1, int(valid_rows), int(hidden_cpu.shape[-1])),
                    "dtype": str(hidden_cpu.dtype),
                    "nbytes": int(valid_rows)
                    * int(hidden_cpu.shape[-1])
                    * int(hidden_cpu.element_size()),
                    "rows": int(valid_rows),
                }

        chunk_refs: list[Any] = []
        chunk_meta: list[dict[str, Any]] = []
        for owner, chunks in sorted(owner_chunks.items()):
            if not chunks:
                continue
            tensors = [chunk["hidden"] for chunk in chunks]
            starts: list[int] = []
            lengths: list[int] = []
            sample_indices: list[int] = []
            row_indices_payload: list[Any] = []
            offset = 0
            for chunk in chunks:
                length = int(chunk["hidden"].shape[0])
                starts.append(offset)
                lengths.append(length)
                sample_indices.append(int(chunk["batch_idx"]))
                row_indices = chunk.get("row_indices")
                if isinstance(row_indices, list) and _sparse_rows_cover_full_window(
                    row_indices, chunk["valid_rows"]
                ):
                    row_indices_payload.append(None)
                else:
                    row_indices_payload.append(_row_indices_payload(row_indices))
                offset += length
            hidden_chunk = (
                torch.cat(tensors, dim=0).contiguous()
                if len(tensors) > 1
                else tensors[0].contiguous()
            )
            ray_put_started = time.perf_counter()
            chunk_ref = ray.put(hidden_chunk)
            ray_put_us += (time.perf_counter() - ray_put_started) * 1_000_000.0
            chunk_index = len(chunk_refs)
            chunk_refs.append(chunk_ref)
            chunk_meta.append(
                {
                    "owner": int(owner),
                    "shape": tuple(hidden_chunk.shape),
                    "dtype": str(hidden_chunk.dtype),
                    "nbytes": int(hidden_chunk.numel())
                    * int(hidden_chunk.element_size()),
                    "sample_indices": sample_indices,
                    "starts": starts,
                    "lengths": lengths,
                    "row_indices": row_indices_payload,
                }
            )
            for sample_pos, batch_idx in enumerate(sample_indices):
                refs[batch_idx] = chunk_ref
                metas[batch_idx]["chunk_index"] = chunk_index
                metas[batch_idx]["chunk_start"] = starts[sample_pos]
                metas[batch_idx]["chunk_length"] = lengths[sample_pos]
                row_indices = row_indices_payload[sample_pos]
                if row_indices is not None:
                    metas[batch_idx]["chunk_row_indices"] = row_indices
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to export SPECO old-logprob hidden ObjectRefs: %s", exc)
        if (
            selected_is_sparse
            or hidden_output.get(OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY) is not None
        ):
            raise RuntimeError(
                "SPECO old-logprob compact hidden ObjectRef export failed"
            ) from exc
        return hidden_output

    updated = dict(hidden_output)
    updated.pop(OLD_LOGPROB_HIDDEN_STATES_KEY, None)
    updated.pop(OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY, None)
    updated.pop("speco_oldlogprob_owner_mask", None)
    updated.pop("speco_oldlogprob_sp_group", None)
    updated.pop("speco_oldlogprob_sp_size", None)
    updated.pop("speco_oldlogprob_sp_rank", None)
    _add_timing_us(updated, _TIMING_CPU_COPY_US, copy_us)
    _add_timing_us(updated, _TIMING_RAY_PUT_US, ray_put_us)
    updated[OLD_LOGPROB_HIDDEN_REFS_KEY] = refs
    updated[OLD_LOGPROB_HIDDEN_REF_META_KEY] = metas
    updated[OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY] = chunk_refs
    updated[OLD_LOGPROB_HIDDEN_CHUNK_META_KEY] = chunk_meta
    return updated


def _install_oldlogprob_training_worker_postprocess_patch() -> bool:
    global _POSTPROCESS_PATCHED
    if _POSTPROCESS_PATCHED:
        return True

    try:
        module = importlib.import_module("verl.workers.engine_workers")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to install SPECO old-logprob postprocess patch: %s", exc)
        return False

    worker_cls = getattr(module, "TrainingWorker", None)
    postprocess_output = (
        getattr(worker_cls, "_postprocess_output", None)
        if worker_cls is not None
        else None
    )
    if not callable(postprocess_output):
        logger.warning(
            "Unable to install SPECO old-logprob postprocess patch: missing TrainingWorker._postprocess_output"
        )
        return False
    if getattr(worker_cls, "_speco_oldlogprob_patched_postprocess", False):
        _POSTPROCESS_PATCHED = True
        return True
    assert worker_cls is not None

    @wraps(postprocess_output)
    def speco_postprocess_output(self, output, *args, **kwargs):
        speco_non_tensor = {}
        model_output = output.get("model_output") if isinstance(output, dict) else None
        if isinstance(model_output, dict):
            for key in (
                OLD_LOGPROB_HIDDEN_REFS_KEY,
                OLD_LOGPROB_HIDDEN_REF_META_KEY,
                OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
                OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
            ):
                if key in model_output:
                    speco_non_tensor[key] = model_output.pop(key)

        final_output = postprocess_output(self, output, *args, **kwargs)
        if speco_non_tensor and final_output is not None:
            from verl.utils import tensordict_utils as tu

            for key, value in speco_non_tensor.items():
                # These ObjectRef side-channel payloads use original sample indices and
                # chunk-level metadata; they are not shaped like upstream tensor outputs.
                tu.assign_non_tensor_data(final_output, key, value)
        return final_output

    worker_cls._postprocess_output = speco_postprocess_output
    worker_cls._speco_oldlogprob_patched_postprocess = True
    _POSTPROCESS_PATCHED = True
    logger.warning("SPECO old-logprob hidden ObjectRef postprocess patch active")
    return True


def _restore_non_tensor_dynamic_batch(values: list[Any], indices: Any) -> list[Any]:
    if not indices:
        return values
    flat_indices = [int(idx) for partition in indices for idx in partition]
    if len(flat_indices) != len(values):
        return values
    reverse_indices = list(flat_indices)
    for reordered_idx, original_idx in enumerate(flat_indices):
        reverse_indices[original_idx] = reordered_idx
    return [values[idx] for idx in reverse_indices]


def _micro_batch_index_map(
    indices: Any, micro_batch_idx: int, fallback_start: int, sample_count: int
) -> list[int]:
    if indices is not None and micro_batch_idx < len(indices):
        values = indices[micro_batch_idx]
        try:
            import torch

            if torch.is_tensor(values):
                return [int(idx) for idx in values.detach().cpu().reshape(-1).tolist()]
        except Exception:  # noqa: BLE001
            pass
        if isinstance(values, (list, tuple)):
            return [int(idx) for idx in values]
    return list(range(int(fallback_start), int(fallback_start) + int(sample_count)))


def _remap_chunk_meta_sample_indices(
    chunk_meta: Any, index_map: list[int]
) -> list[Any]:
    remapped = []
    for meta in chunk_meta if isinstance(chunk_meta, (list, tuple)) else [chunk_meta]:
        if not isinstance(meta, dict):
            remapped.append(meta)
            continue
        new_meta = dict(meta)
        sample_indices = []
        for sample_idx in meta.get("sample_indices") or []:
            try:
                local_idx = int(sample_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= local_idx < len(index_map):
                sample_indices.append(int(index_map[local_idx]))
            else:
                sample_indices.append(local_idx)
        new_meta["sample_indices"] = sample_indices
        remapped.append(new_meta)
    return remapped


def _extract_oldlogprob_non_tensor_model_output(
    output_lst: list[dict[str, Any]],
    indices: Any = None,
) -> dict[str, list[Any]]:
    extracted: dict[str, list[Any]] = {
        OLD_LOGPROB_HIDDEN_REFS_KEY: [],
        OLD_LOGPROB_HIDDEN_REF_META_KEY: [],
        OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY: [],
        OLD_LOGPROB_HIDDEN_CHUNK_META_KEY: [],
    }
    found_speco_output = False
    batch_offset = 0
    for micro_batch_idx, output in enumerate(output_lst or []):
        model_output = output.get("model_output") if isinstance(output, dict) else None
        if not isinstance(model_output, dict):
            continue
        hidden_refs = model_output.get(OLD_LOGPROB_HIDDEN_REFS_KEY)
        sample_count = len(hidden_refs) if isinstance(hidden_refs, (list, tuple)) else 0
        index_map = _micro_batch_index_map(
            indices, micro_batch_idx, batch_offset, sample_count
        )
        for key in tuple(extracted):
            if key not in model_output:
                continue
            found_speco_output = True
            value = model_output.pop(key)
            if key == OLD_LOGPROB_HIDDEN_CHUNK_META_KEY:
                value = _remap_chunk_meta_sample_indices(value, index_map)
            if isinstance(value, (list, tuple)):
                extracted[key].extend(value)
            else:
                extracted[key].append(value)
        batch_offset += sample_count
    return extracted if found_speco_output else {}


def _install_oldlogprob_fsdp_batch_postprocess_patch(transformer_module: Any) -> bool:
    global _BATCH_POSTPROCESS_PATCHED
    if _BATCH_POSTPROCESS_PATCHED:
        return True

    postprocess_batch_func = getattr(transformer_module, "postprocess_batch_func", None)
    if not callable(postprocess_batch_func):
        logger.warning(
            "Unable to install SPECO old-logprob batch postprocess patch: missing postprocess_batch_func"
        )
        return False
    if getattr(
        transformer_module, "_speco_oldlogprob_patched_batch_postprocess", False
    ):
        _BATCH_POSTPROCESS_PATCHED = True
        return True

    @wraps(postprocess_batch_func)
    def speco_postprocess_batch_func(output_lst, indices, data):
        speco_non_tensor = _extract_oldlogprob_non_tensor_model_output(
            output_lst, indices
        )
        output = postprocess_batch_func(output_lst, indices, data)
        if speco_non_tensor and isinstance(output, dict):
            model_output = output.setdefault("model_output", {})
            if isinstance(model_output, dict):
                for key, value in speco_non_tensor.items():
                    if key in (
                        OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
                        OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
                    ):
                        model_output[key] = value
                    else:
                        model_output[key] = _restore_non_tensor_dynamic_batch(
                            value, indices
                        )
        return output

    transformer_module.postprocess_batch_func = speco_postprocess_batch_func
    transformer_module._speco_oldlogprob_patched_batch_postprocess = True
    _BATCH_POSTPROCESS_PATCHED = True
    logger.warning("SPECO old-logprob hidden ObjectRef batch postprocess patch active")
    return True


def _build_selection_context(
    engine: Any, output_args: dict[str, Any], micro_batch: Any
) -> dict[str, Any]:
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.ulysses import get_ulysses_sequence_parallel_group

    use_remove_padding = tu.get_non_tensor_data(
        data=micro_batch, key="use_remove_padding", default=True
    )
    pad_mode = tu.get_non_tensor_data(
        data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING
    )
    if not use_remove_padding or pad_mode != DatasetPadMode.NO_PADDING:
        raise NotImplementedError(
            "SPECO old-logprob hidden collection currently requires use_remove_padding=True "
            "and DatasetPadMode.NO_PADDING"
        )

    input_ids = micro_batch["input_ids"]
    collect_mask = micro_batch[OLD_LOGPROB_COLLECT_MASK_KEY].bool()
    hidden_positions = micro_batch[OLD_LOGPROB_HIDDEN_POSITIONS_KEY].long()
    hidden_position_mask = micro_batch[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY].bool()
    batch_size, hidden_rows = hidden_positions.shape
    offsets = input_ids.offsets().to(device=hidden_positions.device, dtype=torch.long)
    total_tokens = int(input_ids.values().numel())
    pad_size = int(output_args.get("pad_size", 0) or 0)
    if getattr(engine, "use_ulysses_sp", False):
        sp_group = get_ulysses_sequence_parallel_group()
        sp_size = int(getattr(engine, "ulysses_sequence_parallel_size", 1) or 1)
        sp_rank = torch.distributed.get_rank(group=sp_group)
    else:
        sp_group = None
        sp_size = 1
        sp_rank = 0
    object_ref_enabled = _oldlogprob_hidden_object_ref_enabled(micro_batch)
    compact_selected = bool(object_ref_enabled and sp_size <= 1)
    sparse_sp_merge = bool(object_ref_enabled and sp_size > 1)
    selected_batch_indices = [
        int(batch_idx)
        for batch_idx in range(int(batch_size))
        if bool(collect_mask[batch_idx].item())
        and bool(hidden_position_mask[batch_idx].detach().any().item())
    ]
    selected_index_by_batch = {
        batch_idx: selected_idx
        for selected_idx, batch_idx in enumerate(selected_batch_indices)
    }
    output_batch_size = (
        len(selected_batch_indices) if compact_selected else int(batch_size)
    )
    rank_start, rank_end = _flat_local_range(total_tokens, pad_size, sp_size, sp_rank)

    local_positions = []
    local_batch_indices = []
    local_row_indices = []
    local_3d_batch_indices = []
    local_3d_seq_positions = []
    selected_owner_rows: dict[int, list[int]] = {}
    for batch_idx in range(batch_size):
        if not bool(collect_mask[batch_idx].item()):
            continue
        if compact_selected:
            output_batch_idx = selected_index_by_batch.get(
                int(batch_idx), int(batch_idx)
            )
        else:
            output_batch_idx = int(batch_idx)
        seq_start = int(offsets[batch_idx].item())
        seq_end = int(offsets[batch_idx + 1].item())
        for row_idx in range(hidden_rows):
            if not bool(hidden_position_mask[batch_idx, row_idx].item()):
                continue
            seq_pos = int(hidden_positions[batch_idx, row_idx].item())
            if seq_pos < 0:
                continue
            global_flat_pos = seq_start + seq_pos
            if global_flat_pos < seq_start or global_flat_pos >= seq_end:
                continue
            if global_flat_pos < rank_start or global_flat_pos >= rank_end:
                continue
            local_pos = global_flat_pos - rank_start
            local_positions.append(local_pos)
            local_batch_indices.append(output_batch_idx)
            local_row_indices.append(row_idx)
            local_3d_batch_indices.append(int(batch_idx))
            local_3d_seq_positions.append(seq_pos)
            selected_owner_rows.setdefault(output_batch_idx, []).append(row_idx)

    return {
        "batch_size": int(batch_size),
        "output_batch_size": int(output_batch_size),
        "compact_selected": bool(compact_selected),
        "sparse_sp_merge": bool(sparse_sp_merge),
        "selected_batch_indices": selected_batch_indices,
        "hidden_rows": int(hidden_rows),
        "local_positions": local_positions,
        "local_batch_indices": local_batch_indices,
        "local_row_indices": local_row_indices,
        "local_3d_batch_indices": local_3d_batch_indices,
        "local_3d_seq_positions": local_3d_seq_positions,
        "max_local_position": max(local_positions, default=-1),
        "selected_owner_rows": selected_owner_rows,
        "sp_group": sp_group,
        "sp_size": sp_size,
        "sp_rank": sp_rank,
        "timing_us": {
            "select": 0.0,
            "sp_merge": 0.0,
            "concat": 0.0,
        },
    }


def _add_timing_us(
    hidden_output: dict[str, Any], timing_idx: int, value_us: float
) -> None:
    timing = hidden_output.get(OLD_LOGPROB_TIMING_KEY)
    if value_us <= 0 or timing is None:
        return
    try:
        import torch

        if (
            not torch.is_tensor(timing)
            or timing.numel() <= 0
            or timing.shape[-1] <= timing_idx
        ):
            return
        timing.reshape(-1, timing.shape[-1])[0, timing_idx] += float(value_us)
    except Exception:  # noqa: BLE001
        return


def _timing_tensor_from_context(context: dict[str, Any], device: Any):
    import torch

    timing = torch.zeros(
        context["batch_size"],
        _TIMING_WIDTH,
        dtype=torch.float32,
        device=device,
    )
    if context["batch_size"] > 0:
        timing[0, _TIMING_SELECT_US] = float(
            context.get("timing_us", {}).get("select", 0.0)
        )
        timing[0, _TIMING_SP_MERGE_US] = float(
            context.get("timing_us", {}).get("sp_merge", 0.0)
        )
        timing[0, _TIMING_CONCAT_US] = float(
            context.get("timing_us", {}).get("concat", 0.0)
        )
    return timing


def _owner_mask_for_context(context: dict[str, Any], device: Any):
    import torch

    owner_mask = torch.zeros(
        context.get("output_batch_size", context["batch_size"]),
        context["hidden_rows"],
        dtype=torch.int32,
        device=device,
    )
    local_positions, batch_indices, row_indices = _selection_index_tensors_for_device(
        context, device
    )
    if int(local_positions.numel()) > 0:
        owner_mask[batch_indices, row_indices] = 1
    return owner_mask


def _selection_index_tensors_for_device(context: dict[str, Any], device: Any):
    import torch

    cache = context.setdefault("_selection_index_tensor_cache", {})
    key = str(device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if context["local_positions"]:
        local_positions = torch.tensor(
            context["local_positions"], dtype=torch.long, device=device
        )
        batch_indices = torch.tensor(
            context["local_batch_indices"], dtype=torch.long, device=device
        )
        row_indices = torch.tensor(
            context["local_row_indices"], dtype=torch.long, device=device
        )
    else:
        local_positions = torch.empty(0, dtype=torch.long, device=device)
        batch_indices = torch.empty(0, dtype=torch.long, device=device)
        row_indices = torch.empty(0, dtype=torch.long, device=device)

    cached = (local_positions, batch_indices, row_indices)
    cache[key] = cached
    return cached


def _selection_3d_index_tensors_for_device(context: dict[str, Any], device: Any):
    import torch

    cache = context.setdefault("_selection_3d_index_tensor_cache", {})
    key = str(device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if context.get("local_3d_batch_indices"):
        hidden_batch_indices = torch.tensor(
            context["local_3d_batch_indices"], dtype=torch.long, device=device
        )
        hidden_seq_positions = torch.tensor(
            context["local_3d_seq_positions"], dtype=torch.long, device=device
        )
        output_batch_indices = torch.tensor(
            context["local_batch_indices"], dtype=torch.long, device=device
        )
        row_indices = torch.tensor(
            context["local_row_indices"], dtype=torch.long, device=device
        )
    else:
        hidden_batch_indices = torch.empty(0, dtype=torch.long, device=device)
        hidden_seq_positions = torch.empty(0, dtype=torch.long, device=device)
        output_batch_indices = torch.empty(0, dtype=torch.long, device=device)
        row_indices = torch.empty(0, dtype=torch.long, device=device)

    cached = (
        hidden_batch_indices,
        hidden_seq_positions,
        output_batch_indices,
        row_indices,
    )
    cache[key] = cached
    return cached


def _filter_selection_indices_for_hidden(context: dict[str, Any], hidden: Any):
    local_positions, batch_indices, row_indices = _selection_index_tensors_for_device(
        context, hidden.device
    )
    if int(local_positions.numel()) <= 0:
        return local_positions, batch_indices, row_indices

    if int(context.get("max_local_position", -1)) < int(hidden.size(0)):
        return local_positions, batch_indices, row_indices

    valid = local_positions < int(hidden.size(0))
    return local_positions[valid], batch_indices[valid], row_indices[valid]


def _filter_3d_selection_indices_for_hidden(context: dict[str, Any], hidden: Any):
    hidden_batch_indices, hidden_seq_positions, output_batch_indices, row_indices = (
        _selection_3d_index_tensors_for_device(context, hidden.device)
    )
    if int(hidden_batch_indices.numel()) <= 0:
        return (
            hidden_batch_indices,
            hidden_seq_positions,
            output_batch_indices,
            row_indices,
        )

    valid = (hidden_batch_indices < int(hidden.size(0))) & (
        hidden_seq_positions < int(hidden.size(1))
    )
    return (
        hidden_batch_indices[valid],
        hidden_seq_positions[valid],
        output_batch_indices[valid],
        row_indices[valid],
    )


def _extract_hidden_tensor(module_output: Any):
    if isinstance(module_output, tuple):
        module_output = module_output[0]
    elif isinstance(module_output, dict):
        module_output = module_output.get(
            "last_hidden_state", next(iter(module_output.values()), None)
        )
    if module_output is None:
        return None
    if module_output.dim() == 3 and module_output.size(0) == 1:
        return module_output.squeeze(0)
    return module_output


def _select_rows_from_local_hidden(context: dict[str, Any], hidden: Any):
    import torch

    started = time.perf_counter()
    hidden = _extract_hidden_tensor(hidden)
    try:
        if hidden is None:
            return None
        if hidden.dim() == 3:
            if context.get("sparse_sp_merge"):
                empty_indices = torch.empty(0, dtype=torch.long, device=hidden.device)
                (
                    hidden_batch_indices,
                    hidden_seq_positions,
                    batch_indices,
                    row_indices,
                ) = _filter_3d_selection_indices_for_hidden(context, hidden)
                if int(hidden_batch_indices.numel()) <= 0:
                    return {
                        "rows": hidden.new_empty((0, int(hidden.size(-1)))),
                        "batch_indices": empty_indices,
                        "row_indices": empty_indices,
                    }
                rows = hidden[hidden_batch_indices, hidden_seq_positions].detach()
                return {
                    "rows": rows,
                    "batch_indices": batch_indices,
                    "row_indices": row_indices,
                }

            selected = torch.zeros(
                context.get("output_batch_size", context["batch_size"]),
                context["hidden_rows"],
                int(hidden.size(-1)),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            hidden_batch_indices, hidden_seq_positions, batch_indices, row_indices = (
                _filter_3d_selection_indices_for_hidden(context, hidden)
            )
            if int(hidden_batch_indices.numel()) > 0:
                selected[batch_indices, row_indices] = hidden[
                    hidden_batch_indices, hidden_seq_positions
                ].detach()
            return selected
        if hidden.dim() != 2:
            raise RuntimeError(
                "SPECO old-logprob forward-hook capture expected a local 2D or 3D hidden tensor, "
                f"got shape={tuple(hidden.shape)}"
            )

        if context.get("sparse_sp_merge"):
            empty_indices = torch.empty(0, dtype=torch.long, device=hidden.device)
            local_positions, batch_indices, row_indices = (
                _filter_selection_indices_for_hidden(context, hidden)
            )
            if int(local_positions.numel()) <= 0:
                return {
                    "rows": hidden.new_empty((0, int(hidden.size(-1)))),
                    "batch_indices": empty_indices,
                    "row_indices": empty_indices,
                }
            rows = hidden.index_select(0, local_positions).detach()
            return {
                "rows": rows,
                "batch_indices": batch_indices,
                "row_indices": row_indices,
            }

        selected = torch.zeros(
            context.get("output_batch_size", context["batch_size"]),
            context["hidden_rows"],
            int(hidden.size(-1)),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        local_positions, batch_indices, row_indices = (
            _filter_selection_indices_for_hidden(context, hidden)
        )
        if int(local_positions.numel()) > 0:
            selected[batch_indices, row_indices] = hidden.index_select(
                0, local_positions
            ).detach()
        return selected
    finally:
        context.setdefault("timing_us", {}).setdefault("select", 0.0)
        context["timing_us"]["select"] += (time.perf_counter() - started) * 1_000_000.0


def _dense_selected_from_sparse(
    context: dict[str, Any], sparse_selected: dict[str, Any]
):
    import torch

    rows = sparse_selected["rows"]
    selected = torch.zeros(
        context.get("output_batch_size", context["batch_size"]),
        context["hidden_rows"],
        int(rows.size(-1)),
        dtype=rows.dtype,
        device=rows.device,
    )
    if int(rows.size(0)) > 0:
        selected[sparse_selected["batch_indices"], sparse_selected["row_indices"]] = (
            rows
        )
    return selected


def _owner_mask_from_sparse_selected(
    context: dict[str, Any], sparse_selected: dict[str, Any]
):
    import torch

    rows = sparse_selected["rows"]
    device = rows.device
    owner_mask = torch.zeros(
        context.get("output_batch_size", context["batch_size"]),
        context["hidden_rows"],
        dtype=torch.bool,
        device=device,
    )
    batch_indices = sparse_selected["batch_indices"]
    row_indices = sparse_selected["row_indices"]
    if batch_indices.device != device or batch_indices.dtype != torch.long:
        batch_indices = batch_indices.to(device=device, dtype=torch.long)
    if row_indices.device != device or row_indices.dtype != torch.long:
        row_indices = row_indices.to(device=device, dtype=torch.long)
    if int(batch_indices.numel()) > 0:
        owner_mask[batch_indices, row_indices] = True
    return owner_mask


def _gather_sparse_sp_selected_to_source(
    context: dict[str, Any], sparse_selected: dict[str, Any]
):
    import torch
    import torch.distributed as dist

    rows = sparse_selected["rows"]
    device = rows.device
    sp_group = context["sp_group"]
    sp_size = int(context.get("sp_size", 1) or 1)
    sp_rank = int(context.get("sp_rank", 0) or 0)
    source_sp_rank = 0
    source_global_rank = (
        dist.get_global_rank(sp_group, source_sp_rank)
        if hasattr(dist, "get_global_rank")
        else source_sp_rank
    )
    is_source = sp_rank == source_sp_rank
    local_count = torch.tensor([int(rows.size(0))], dtype=torch.long, device=device)
    max_count_tensor = local_count.clone()
    dist.all_reduce(max_count_tensor, op=dist.ReduceOp.MAX, group=sp_group)
    max_count = int(max_count_tensor.item())
    hidden_dim = int(rows.size(-1))
    if max_count <= 0:
        if not is_source:
            return None
        empty_indices = rows.new_empty((0,), dtype=torch.long)
        return rows.new_empty((0, hidden_dim)), empty_indices, empty_indices

    padded_rows = rows.new_zeros((max_count, hidden_dim))
    padded_batch_indices = torch.zeros(max_count, dtype=torch.long, device=device)
    padded_row_indices = torch.zeros(max_count, dtype=torch.long, device=device)
    if int(local_count.item()) > 0:
        count = int(local_count.item())
        padded_rows[:count] = rows
        padded_batch_indices[:count] = sparse_selected["batch_indices"]
        padded_row_indices[:count] = sparse_selected["row_indices"]

    gathered_counts = (
        [torch.empty_like(local_count) for _ in range(sp_size)] if is_source else None
    )
    gathered_rows = (
        [torch.empty_like(padded_rows) for _ in range(sp_size)] if is_source else None
    )
    gathered_batch_indices = (
        [torch.empty_like(padded_batch_indices) for _ in range(sp_size)]
        if is_source
        else None
    )
    gathered_row_indices = (
        [torch.empty_like(padded_row_indices) for _ in range(sp_size)]
        if is_source
        else None
    )
    dist.gather(
        local_count, gather_list=gathered_counts, dst=source_global_rank, group=sp_group
    )
    dist.gather(
        padded_rows, gather_list=gathered_rows, dst=source_global_rank, group=sp_group
    )
    dist.gather(
        padded_batch_indices,
        gather_list=gathered_batch_indices,
        dst=source_global_rank,
        group=sp_group,
    )
    dist.gather(
        padded_row_indices,
        gather_list=gathered_row_indices,
        dst=source_global_rank,
        group=sp_group,
    )
    if not is_source:
        return None
    assert gathered_counts is not None
    assert gathered_rows is not None
    assert gathered_batch_indices is not None
    assert gathered_row_indices is not None

    rows_parts: list[Any] = []
    batch_parts: list[Any] = []
    row_parts: list[Any] = []
    counts = [int(count.item()) for count in gathered_counts]
    for count, row_part, batch_part, row_idx_part in zip(
        counts,
        gathered_rows,
        gathered_batch_indices,
        gathered_row_indices,
        strict=False,
    ):
        if count <= 0:
            continue
        rows_parts.append(row_part[:count])
        batch_parts.append(batch_part[:count])
        row_parts.append(row_idx_part[:count])
    if not rows_parts:
        return (
            rows.new_empty((0, hidden_dim)),
            rows.new_empty((0,), dtype=torch.long),
            rows.new_empty((0,), dtype=torch.long),
        )
    return (
        torch.cat(rows_parts, dim=0),
        torch.cat(batch_parts, dim=0),
        torch.cat(row_parts, dim=0),
    )


def _merge_sp_selected(context: dict[str, Any], selected: Any):
    import torch

    if isinstance(selected, dict):
        if context.get("sparse_sp_merge") and context.get("sp_group") is not None:
            started = time.perf_counter()
            gathered = _gather_sparse_sp_selected_to_source(context, selected)
            context.setdefault("timing_us", {}).setdefault("sp_merge", 0.0)
            context["timing_us"]["sp_merge"] += (
                time.perf_counter() - started
            ) * 1_000_000.0
            if gathered is None:
                return None, None
            rows, batch_indices, row_indices = gathered
            selected = {
                "rows": rows,
                "batch_indices": batch_indices,
                "row_indices": row_indices,
            }
            return selected, None
        owner_mask = _owner_mask_from_sparse_selected(context, selected)
        selected = _dense_selected_from_sparse(context, selected)
        return selected, owner_mask

    owner_mask = _owner_mask_for_context(context, selected.device)
    sp_group = context.get("sp_group")
    if sp_group is not None and not context.get("sparse_sp_merge"):
        started = time.perf_counter()
        import torch.distributed as dist

        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(owner_mask, op=dist.ReduceOp.SUM, group=sp_group)
        owner_mask = owner_mask.to(dtype=torch.bool)
        context.setdefault("timing_us", {}).setdefault("sp_merge", 0.0)
        context["timing_us"]["sp_merge"] += (
            time.perf_counter() - started
        ) * 1_000_000.0
    return selected, owner_mask


def _select_and_merge_concatenated_hidden(
    context: dict[str, Any],
    hidden_parts: list[Any],
    *,
    already_selected: bool = False,
):
    import torch

    selected_items = []
    for hidden in hidden_parts:
        if _is_sparse_selected(hidden):
            selected_items.append(hidden)
        elif already_selected:
            expected_shape = (
                int(context.get("output_batch_size", context["batch_size"])),
                int(context["hidden_rows"]),
            )
            if (
                not torch.is_tensor(hidden)
                or hidden.dim() != 3
                or tuple(hidden.shape[:2]) != expected_shape
            ):
                shape = (
                    tuple(hidden.shape)
                    if torch.is_tensor(hidden)
                    else type(hidden).__name__
                )
                raise RuntimeError(
                    "SPECO old-logprob forward-hook capture produced an invalid selected hidden tensor: "
                    f"expected [batch, rows, hidden] with prefix={expected_shape}, got {shape}"
                )
            selected_items.append(hidden)
        else:
            selected_items.append(_select_rows_from_local_hidden(context, hidden))

    if not selected_items or any(selected is None for selected in selected_items):
        raise RuntimeError(
            "SPECO old-logprob hidden collection failed to resolve required hidden layers"
        )

    dict_selected = [
        selected for selected in selected_items if isinstance(selected, dict)
    ]
    if dict_selected and len(dict_selected) != len(selected_items):
        raise RuntimeError(
            "SPECO old-logprob hidden collection mixed sparse and dense selected rows"
        )

    concat_started = time.perf_counter()
    if dict_selected:
        selected_rows = torch.cat(
            [selected["rows"] for selected in dict_selected], dim=-1
        )
    else:
        selected_rows = torch.cat(selected_items, dim=-1)
    context.setdefault("timing_us", {}).setdefault("concat", 0.0)
    context["timing_us"]["concat"] += (
        time.perf_counter() - concat_started
    ) * 1_000_000.0

    if dict_selected:
        selected_template = dict_selected[0]
        merged_input = {
            "rows": selected_rows,
            "batch_indices": selected_template["batch_indices"],
            "row_indices": selected_template["row_indices"],
        }
    else:
        merged_input = selected_rows
    return _merge_sp_selected(context, merged_input)


def _unwrap_module(module: Any):
    current = module
    seen = set()
    while (
        current is not None and hasattr(current, "module") and id(current) not in seen
    ):
        seen.add(id(current))
        child = getattr(current, "module", None)
        if child is None or child is current:
            break
        current = child
    return current


def _get_module_by_path(root: Any, path: str):
    current = root
    for part in path.split("."):
        if current is None:
            return None
        if part.isdigit() and isinstance(current, (list, tuple)):
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            current = getattr(current, part, None)
    return current


def _find_layers_and_final_norm(engine: Any):
    import torch.nn as nn

    module = getattr(engine, "module", None)
    roots: list[Any] = []
    for root in (module, _unwrap_module(module)):
        if root is not None and all(root is not existing for existing in roots):
            roots.append(root)

    candidates = (
        ("model.layers", "model.norm"),
        ("base_model.model.layers", "base_model.model.norm"),
        ("model.decoder.layers", "model.decoder.final_layer_norm"),
        ("transformer.h", "transformer.ln_f"),
        ("gpt_neox.layers", "gpt_neox.final_layer_norm"),
    )
    for root in roots:
        for layers_path, norm_path in candidates:
            layers = _get_module_by_path(root, layers_path)
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return list(layers), _get_module_by_path(root, norm_path)

        for name, child in root.named_modules():
            if not isinstance(child, nn.ModuleList) or len(child) <= 0:
                continue
            if not (name.endswith("layers") or name.endswith("h")):
                continue
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            norm_names = ("norm", "final_layer_norm", "ln_f")
            for norm_name in norm_names:
                norm_path = f"{parent_path}.{norm_name}" if parent_path else norm_name
                final_norm = _get_module_by_path(root, norm_path)
                if final_norm is not None:
                    return list(child), final_norm
            return list(child), None

    return None, None


def _hidden_state_capture_target(
    layer_id: int, num_layers: int
) -> tuple[str, int | None]:
    num_states = num_layers + 1
    index = int(layer_id)
    hidden_state_index = index + 1 if index >= 0 else num_states + index
    if hidden_state_index <= 0 or hidden_state_index > num_layers:
        raise IndexError(
            f"SPECO old-logprob hidden layer id {layer_id} resolved to hidden-state index "
            f"{hidden_state_index}, but hook capture supports layer outputs/final norm for {num_layers} layers"
        )
    if hidden_state_index == num_layers:
        return "final", None
    return "layer", hidden_state_index - 1


def _oldlogprob_aux_layer_ids_from_batch(micro_batch: Any) -> list[int]:
    from verl.utils import tensordict_utils as tu

    aux_layer_ids = tu.get_non_tensor_data(
        data=micro_batch, key=OLD_LOGPROB_AUX_LAYER_IDS_KEY, default=None
    )
    if aux_layer_ids is None:
        raise RuntimeError(
            "SPECO old-logprob hidden collection is missing aux layer ids. "
            "The trainer must pass explicit EAGLE3 aux layers or derive them from target num_hidden_layers."
        )
    if isinstance(aux_layer_ids, int):
        return [int(aux_layer_ids)]
    return [int(layer_id) for layer_id in aux_layer_ids]


def _make_capture_hook(context: dict[str, Any], key: str):
    def hook(_module, _inputs, module_output):
        selected = _select_rows_from_local_hidden(context, module_output)
        if selected is not None:
            context["captures"][key] = selected

    return hook


def _cleanup_oldlogprob_hidden_capture(engine: Any) -> None:
    context = getattr(engine, "_speco_oldlogprob_hidden_context", None)
    if not context:
        return
    for handle in context.get("handles", []):
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            pass
    try:
        delattr(engine, "_speco_oldlogprob_hidden_context")
    except AttributeError:
        pass


def _install_oldlogprob_hidden_hooks(
    engine: Any, output_args: dict[str, Any], micro_batch: Any
) -> None:
    if not _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
        return

    layers, final_norm = _find_layers_and_final_norm(engine)
    if not layers:
        raise RuntimeError(
            "SPECO old-logprob forward-hook capture could not find transformer layers"
        )

    selection_context = _build_selection_context(engine, output_args, micro_batch)
    aux_layer_ids = _oldlogprob_aux_layer_ids_from_batch(micro_batch)
    hidden_layout = _oldlogprob_hidden_layout(micro_batch)
    aux_keys = []
    required_modules = {}
    for layer_id in aux_layer_ids:
        kind, layer_index = _hidden_state_capture_target(layer_id, len(layers))
        key = "final" if kind == "final" else f"layer:{layer_index}"
        aux_keys.append(key)
        if kind == "final":
            if final_norm is None:
                raise RuntimeError(
                    "SPECO old-logprob forward-hook capture could not find final norm module"
                )
            required_modules[key] = final_norm
        else:
            required_modules[key] = layers[layer_index]

    final_key = None
    if hidden_layout in {"eagle3_aux_plus_last", "dflash_aux_plus_last"}:
        if final_norm is None:
            raise RuntimeError(
                "SPECO old-logprob forward-hook capture could not find final norm module"
            )
        final_key = "final"
        required_modules[final_key] = final_norm

    _cleanup_oldlogprob_hidden_capture(engine)
    context = {
        **selection_context,
        "aux_keys": aux_keys,
        "final_key": final_key,
        "hidden_layout": hidden_layout,
        "captures": {},
        "handles": [],
    }
    for key, module in required_modules.items():
        context["handles"].append(
            module.register_forward_hook(_make_capture_hook(context, key))
        )
    engine._speco_oldlogprob_hidden_context = context


def _consume_oldlogprob_hidden_capture(engine: Any):
    import torch

    context = getattr(engine, "_speco_oldlogprob_hidden_context", None)
    if not context:
        return {}

    try:
        captures = context.get("captures", {})
        required_keys = list(context["aux_keys"])
        if context.get("final_key") is not None:
            required_keys.append(context["final_key"])
        missing = [key for key in required_keys if key not in captures]
        if missing:
            raise RuntimeError(
                f"SPECO old-logprob forward-hook capture missed hidden tensors: {missing}"
            )

        hidden_parts = [captures[key] for key in context["aux_keys"]]
        if context.get("final_key") is not None:
            hidden_parts.append(captures[context["final_key"]])
        selected, owner_mask = _select_and_merge_concatenated_hidden(
            context,
            hidden_parts,
            already_selected=True,
        )
        if _is_sparse_sp_non_source_context(context):
            return {}
        if selected is None:
            return {}
        output = {
            OLD_LOGPROB_HIDDEN_STATES_KEY: selected,
            OLD_LOGPROB_TIMING_KEY: _timing_tensor_from_context(
                context, _selected_device(selected)
            ),
            "speco_oldlogprob_owner_mask": owner_mask,
            "speco_oldlogprob_sp_group": context.get("sp_group"),
            "speco_oldlogprob_sp_size": int(context.get("sp_size", 1) or 1),
            "speco_oldlogprob_sp_rank": int(context.get("sp_rank", 0) or 0),
        }
        if context.get("compact_selected"):
            output[OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY] = torch.tensor(
                context.get("selected_batch_indices", []),
                dtype=torch.long,
                device=_selected_device(selected),
            )
        return output
    finally:
        _cleanup_oldlogprob_hidden_capture(engine)


def _select_oldlogprob_hidden_states(
    engine: Any, output: Any, output_args: dict[str, Any], micro_batch: Any
):
    if not _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
        return {}

    import torch

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None and isinstance(output, dict):
        hidden_states = output.get("hidden_states")
    if hidden_states is None:
        return {}

    context = _build_selection_context(engine, output_args, micro_batch)
    aux_layer_ids = _oldlogprob_aux_layer_ids_from_batch(micro_batch)
    hidden_layout = _oldlogprob_hidden_layout(micro_batch)

    aux_hidden_list = [
        _resolve_hidden_state(hidden_states, layer_id) for layer_id in aux_layer_ids
    ]
    selected_hidden_list = list(aux_hidden_list)
    if hidden_layout in {"eagle3_aux_plus_last", "dflash_aux_plus_last"}:
        final_hidden = hidden_states[-1]
        final_hidden = (
            final_hidden.squeeze(0)
            if final_hidden.dim() == 3 and final_hidden.size(0) == 1
            else final_hidden
        )
        selected_hidden_list.append(final_hidden)
    selected, owner_mask = _select_and_merge_concatenated_hidden(
        context, selected_hidden_list
    )
    if _is_sparse_sp_non_source_context(context):
        return {}
    if selected is None:
        return {}
    output = {
        OLD_LOGPROB_HIDDEN_STATES_KEY: selected,
        OLD_LOGPROB_TIMING_KEY: _timing_tensor_from_context(
            context, _selected_device(selected)
        ),
        "speco_oldlogprob_owner_mask": owner_mask,
        "speco_oldlogprob_sp_group": context.get("sp_group"),
        "speco_oldlogprob_sp_size": int(context.get("sp_size", 1) or 1),
        "speco_oldlogprob_sp_rank": int(context.get("sp_rank", 0) or 0),
    }
    if context.get("compact_selected"):
        output[OLD_LOGPROB_SELECTED_BATCH_INDICES_KEY] = torch.tensor(
            context.get("selected_batch_indices", []),
            dtype=torch.long,
            device=_selected_device(selected),
        )
    return output


def install_oldlogprob_hidden_runtime_patch() -> bool:
    """Patch upstream FSDP LM-head engine methods in the current process."""

    global _PATCHED
    if _PATCHED:
        module = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
        _install_oldlogprob_fsdp_batch_postprocess_patch(module)
        _install_oldlogprob_training_worker_postprocess_patch()
        return True

    try:
        module = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to install SPECO old-logprob hidden runtime patch: %s", exc
        )
        return False

    engine_cls = getattr(module, "FSDPEngineWithLMHead", None)
    if engine_cls is None:
        logger.warning(
            "Unable to install SPECO old-logprob hidden runtime patch: missing FSDPEngineWithLMHead"
        )
        return False

    prepare_model_inputs = getattr(engine_cls, "prepare_model_inputs", None)
    prepare_model_outputs = getattr(engine_cls, "prepare_model_outputs", None)
    forward_step = getattr(engine_cls, "forward_step", None)
    if not callable(prepare_model_inputs) or not callable(prepare_model_outputs):
        logger.warning(
            "Unable to install SPECO old-logprob hidden runtime patch: missing engine methods"
        )
        return False

    _install_oldlogprob_fsdp_batch_postprocess_patch(module)

    if not getattr(engine_cls, "_speco_oldlogprob_patched_inputs", False):

        @wraps(prepare_model_inputs)
        def speco_prepare_model_inputs(self, micro_batch, *args, **kwargs):
            model_inputs, output_args = prepare_model_inputs(
                self, micro_batch, *args, **kwargs
            )
            if _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
                capture_impl = _oldlogprob_capture_impl(micro_batch)
                model_inputs["return_dict"] = True
                if capture_impl == "output_hidden_states":
                    model_inputs["output_hidden_states"] = True
                elif capture_impl == "forward_hook":
                    _install_oldlogprob_hidden_hooks(self, output_args, micro_batch)
                else:
                    raise ValueError(
                        f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}"
                    )
            return model_inputs, output_args

        engine_cls.prepare_model_inputs = speco_prepare_model_inputs
        engine_cls._speco_oldlogprob_patched_inputs = True

    if not getattr(engine_cls, "_speco_oldlogprob_patched_outputs", False):

        @wraps(prepare_model_outputs)
        def speco_prepare_model_outputs(
            self,
            output,
            output_args,
            micro_batch,
            logits_processor_func,
            *args,
            **kwargs,
        ):
            try:
                model_output = prepare_model_outputs(
                    self,
                    output,
                    output_args,
                    micro_batch,
                    logits_processor_func,
                    *args,
                    **kwargs,
                )
                if _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
                    capture_impl = _oldlogprob_capture_impl(micro_batch)
                    if capture_impl == "forward_hook":
                        hidden_output = _consume_oldlogprob_hidden_capture(self)
                    elif capture_impl == "output_hidden_states":
                        hidden_output = _select_oldlogprob_hidden_states(
                            self, output, output_args, micro_batch
                        )
                    else:
                        raise ValueError(
                            f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}"
                        )
                    if not hidden_output:
                        if _oldlogprob_hidden_object_ref_enabled(micro_batch):
                            return model_output
                        raise RuntimeError(
                            "SPECO old-logprob hidden collection did not produce selected hidden states"
                        )
                    hidden_output = _put_oldlogprob_hidden_refs(
                        hidden_output, micro_batch
                    )
                    model_output.update(hidden_output)
                return model_output
            except Exception:
                _cleanup_oldlogprob_hidden_capture(self)
                raise

        engine_cls.prepare_model_outputs = speco_prepare_model_outputs
        engine_cls._speco_oldlogprob_patched_outputs = True

    if callable(forward_step) and not getattr(
        engine_cls, "_speco_oldlogprob_patched_forward_step", False
    ):

        @wraps(forward_step)
        def speco_forward_step(self, *args, **kwargs):
            try:
                return forward_step(self, *args, **kwargs)
            finally:
                _cleanup_oldlogprob_hidden_capture(self)

        engine_cls.forward_step = speco_forward_step
        engine_cls._speco_oldlogprob_patched_forward_step = True

    _install_oldlogprob_training_worker_postprocess_patch()

    _PATCHED = True
    logger.warning(
        "SPECO old-logprob hidden runtime patch active for upstream FSDP LM-head engine"
    )
    return True
