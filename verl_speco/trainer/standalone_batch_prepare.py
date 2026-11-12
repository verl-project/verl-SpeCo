# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU-only file reading, alignment, collation and pinning for standalone prefetch."""

from __future__ import annotations

import math
import time
from concurrent.futures import Executor
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from verl_speco.producer.hidden_states_store import (
    FileHiddenStatesStore,
    HiddenStatesStore,
)
from verl_speco.trainer.target_feature_replay import (
    FeatureContract,
    feature_from_vllm_payload,
)
from verl_speco.transport.drafter_sample_protocol import validate_reference_tag


@torch.no_grad()
def prepare_standalone_cpu_batch(
    tags: Sequence[Mapping[str, Any]],
    feature_contract: FeatureContract,
    *,
    max_window: int | None = None,
    packing: bool = False,
    block_size: int = 16,
    max_packed_len: int = 0,
    pin_memory: bool = True,
    executor: Executor | None = None,
    hidden_states_store: HiddenStatesStore | None = None,
    read_timeout: float = 30.0,
    streaming: bool = False,
    max_output_bytes: int = 0,
) -> dict[str, Any]:
    """Prepare one assigned batch without trainer state, GPU work or deletion.

    Ordering is the scheduler's ordering, not a second random sample. CUDA-less
    tests may set pin_memory=False; production pinning failures propagate rather
    than silently returning pageable buffers. Cleanup references stay in the
    caller's batch envelope. No file is released here, including on failure.
    """
    store = (
        hidden_states_store
        if hidden_states_store is not None
        else FileHiddenStatesStore()
    )
    algorithm = _validate_reference_contracts(tags, feature_contract, store)
    if type(streaming) is not bool:
        raise ValueError("streaming must be a boolean")
    if type(max_output_bytes) is not int or max_output_bytes < 0:
        raise ValueError("max_output_bytes must be a non-negative integer")
    if (isinstance(read_timeout, bool) or not isinstance(read_timeout, (int, float))
            or not math.isfinite(read_timeout) or read_timeout <= 0):
        raise ValueError("read_timeout must be a positive finite number")
    read_deadline = time.monotonic() + read_timeout
    if not tags or (max_window is not None and max_window < 0):
        raise ValueError("A non-empty batch and non-negative max_window are required")
    if block_size <= 0 or max_packed_len < 0:
        raise ValueError("Invalid packing limits")
    if algorithm == "EAGLE3" and packing:
        raise ValueError("Block packing is not an EAGLE3 mode")

    prepare = partial(
        _prepare_chunk,
        feature_contract=feature_contract,
        algorithm=algorithm,
        max_window=max_window,
        store=store,
        read_deadline=read_deadline,
        partial_file_reads=streaming,
    )
    if streaming:
        plan = _stream_plan(tags, algorithm=algorithm, packing=packing, block_size=block_size,
                            max_packed_len=max_packed_len, max_window=max_window)
        if max_output_bytes and type(store) is FileHiddenStatesStore:
            remaining = read_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Reference batch read deadline exceeded")
            _, _, width = store.hidden_shape(tags[0]["hidden_states_path"], timeout=remaining)
            # Count the final CPU tensors and one pinning copy, matching the
            # existing post-materialization _prepared_host_bytes budget.
            aux = len(feature_contract.vllm_aux_hidden_state_layer_ids or feature_contract.target_layer_ids)
            final = int(feature_contract.hidden_states_layout.endswith("_plus_last"))
            scalar_bytes = 28 + (8 if packing else 12 if algorithm != "EAGLE3" else 0)
            output_bytes = 2 * math.prod(plan[2]) * (scalar_bytes + 2 * width * (aux + final))
            if output_bytes > max_output_bytes:
                raise ValueError("Planned reference batch exceeds consumer_prefetch_batch_bytes")
        batch, window_tokens, supervised_tokens = _stream_batch(
            tags, prepare, algorithm=algorithm, packing=packing, plan=plan,
        )
    else:
        chunks = list(executor.map(prepare, tags)) if executor is not None else list(map(prepare, tags))
        if algorithm == "EAGLE3":
            batch = _concatenate(chunks)
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        elif packing:
            batch = _pack(chunks, block_size, max_packed_len)
        else:
            batch = _pad(chunks)
            batch["label_ids"] = batch["input_ids"].clone()
            batch["label_mask"] = batch["loss_mask"].clone()
        window_tokens = sum(int(chunk["input_ids"].numel()) for chunk in chunks)
        supervised_tokens = sum(int(torch.count_nonzero(chunk["loss_mask"]).item()) for chunk in chunks)
    if pin_memory:
        batch = {key: value.pin_memory() for key, value in batch.items()}
    batch["_speco_token_counts"] = {
        "source_tokens": sum(len(tag["request"]["input_ids"]) for tag in tags),
        "window_tokens": window_tokens,
        "supervised_tokens": supervised_tokens,
        "padding_tokens": int(batch["input_ids"].numel()) - window_tokens,
        "samples": len(tags),
    }
    batch["hidden_source"] = "standalone_vllm_reference"
    batch["_speco_pad_size"] = 0
    if feature_contract.hidden_states_layout.endswith("_plus_last"):
        batch["last_hidden_state_norm"] = "raw_target_residual"
    return batch


def _validate_reference_contracts(tags, feature_contract, store):
    algorithm = feature_contract.algorithm.upper()
    if algorithm not in {"DSPARK", "DFLASH", "EAGLE3"} or feature_contract.use_logits:
        raise ValueError(
            "Reference prefetch supports DSPARK/DFLASH/EAGLE3 hidden supervision only"
        )
    if not feature_contract.require_full_alignment:
        raise ValueError("Reference prefetch requires full feature alignment")
    valid_layouts = {
        "DSPARK": {"dflash_aux", "dflash_aux_plus_last"},
        "DFLASH": {"dflash_aux", "dflash_aux_plus_last"},
        "EAGLE3": {"eagle3_aux_plus_last"},
    }
    if feature_contract.hidden_states_layout not in valid_layouts[algorithm]:
        raise ValueError("Algorithm and hidden layout do not match")
    expected = asdict(feature_contract)
    expected["dtype"] = str(feature_contract.dtype)
    # Validate every reference before opening files; never combine contracts.
    for tag in tags:
        validate_reference_tag(tag)
        if tag["backend"] != store.backend:
            raise ValueError("Hidden reference backend does not match consumer store")
        if tag["backend"] == "file" and callable(getattr(store, "validate_reference", None)):
            store.validate_reference(tag["hidden_states_path"])
        for name in (
            "algorithm",
            "target_layer_ids",
            "hidden_states_layout",
            "dtype",
            "target_model_id",
            "target_model_revision",
            "tokenizer_fingerprint",
            "target_num_hidden_layers",
            "vllm_aux_hidden_state_layer_ids",
            "use_logits",
            "require_full_alignment",
            "target_config_fingerprint",
        ):
            if tag["feature_contract"].get(name) != expected[name]:
                raise ValueError(f"Hidden reference feature contract mismatch: {name}")

    return algorithm


def _stream_plan(tags, *, algorithm, packing, block_size, max_packed_len, max_window):
    """Plan all rows from metadata before materializing features."""
    lengths = [len(tag["request"]["feature_positions"]) for tag in tags]
    if algorithm == "EAGLE3":
        lengths = [length - 2 for length in lengths]
    elif max_window:
        lengths = [min(length, max_window) for length in lengths]
    if any(length <= 0 for length in lengths):
        raise ValueError("Reference training sample has no usable feature rows")
    placements = []
    row_sizes = [0]
    for index, length in enumerate(lengths):
        if algorithm == "EAGLE3":
            placements.append((0, row_sizes[0], length, index))
            row_sizes[0] += length
        elif not packing:
            placements.append((index, 0, length, 0))
        else:
            padded = length + (-length) % block_size
            if max_packed_len and padded > max_packed_len:
                raise ValueError("One sample exceeds max_packed_len after block alignment")
            if max_packed_len and row_sizes[-1] + padded > max_packed_len:
                row_sizes.append(0)
            row = len(row_sizes) - 1
            document = 0 if not row_sizes[row] else placements[-1][3] + 1
            placements.append((row, row_sizes[row], padded, document))
            row_sizes[row] += padded
    shape = ((1, sum(lengths)) if algorithm == "EAGLE3" else
             (len(row_sizes), max(row_sizes)) if packing else (len(lengths), max(lengths)))
    return lengths, placements, shape


def plan_owned_anchor_windows(document_ids, anchors, keep_mask, *, block_size, label_shift, windows, world_size):
    """Route already-sampled anchors with their exact document-local context.

    This planner does not sample anchors, materialize tensors, or choose a loss
    reduction. The caller must preserve the original optimizer semantics.
    Each item describes one packed row; end offsets are exclusive.
    """
    if (not isinstance(document_ids, (list, tuple)) or not document_ids
            or any(type(doc) is not int or doc < -1 for doc in document_ids)
            or not isinstance(anchors, (list, tuple)) or not isinstance(keep_mask, (list, tuple))
            or len(anchors) != len(keep_mask) or any(type(value) is not bool for value in keep_mask)
            or type(block_size) is not int or block_size < 1 or type(label_shift) is not int or label_shift not in (0, 1)
            or type(world_size) is not int or world_size < 1
            or not isinstance(windows, (list, tuple)) or not windows
            or any(window is not None and (type(window) is not int or window <= 0) for window in windows)):
        raise ValueError("Invalid owned-anchor window plan")
    spans, index = {}, 0
    while index < len(document_ids):
        doc = document_ids[index]
        end = index + 1
        while end < len(document_ids) and document_ids[end] == doc:
            end += 1
        if doc >= 0:
            if doc in spans:
                raise ValueError("Packed document IDs must be contiguous")
            spans[doc] = (index, end)
        index = end
    horizon = None if any(window is None for window in windows) else max(windows)
    assignments = [[] for _ in range(world_size)]
    seen = set()
    ordinal = 0
    for slot, (anchor, keep) in enumerate(zip(anchors, keep_mask, strict=True)):
        if not keep:
            continue
        if type(anchor) is not int or not 0 <= anchor < len(document_ids) or anchor in seen:
            raise ValueError("Active anchors must be unique valid positions")
        seen.add(anchor)
        doc = document_ids[anchor]
        tail = min(anchor + block_size - 1 + label_shift, len(document_ids) - 1)
        if doc < 0 or document_ids[tail] != doc:
            raise ValueError("Active anchor label span crosses a document boundary")
        doc_start, _ = spans[doc]
        start = doc_start if horizon is None else max(doc_start, anchor - horizon)
        end = tail + 1
        assignments[ordinal % world_size].append({
            "anchor_slot": slot, "anchor": anchor, "document_id": doc,
            "start": start, "end": end, "local_anchor": anchor - start,
        })
        ordinal += 1
    return assignments


@torch.no_grad()
def prepare_owned_anchor_window(tag, feature_contract, *, start, end, anchor, block_size,
                                windows, source_window=None, read_timeout=30.0, max_output_bytes=0,
                                hidden_states_store=None):
    """Read one file-backed owned window for an already sampled anchor.

    This low-level materializer does not assign global ownership or combine
    per-part gradients. The caller retains the source reference until commit.
    """
    store = hidden_states_store if hidden_states_store is not None else FileHiddenStatesStore()
    if _validate_reference_contracts([tag], feature_contract, store) != "DSPARK":
        raise ValueError("Owned anchor windows currently require the block training backend")
    if (isinstance(read_timeout, bool) or not isinstance(read_timeout, (int, float))
            or not math.isfinite(read_timeout) or read_timeout <= 0
            or type(max_output_bytes) is not int or max_output_bytes < 0):
        raise ValueError("Invalid owned-window read timeout or byte limit")
    positions = tag["request"]["feature_positions"]
    if source_window is None:
        source_window = (0, len(positions))
    if (not isinstance(source_window, (tuple, list)) or len(source_window) != 2
            or any(type(value) is not int for value in source_window)
            or not 0 <= source_window[0] < source_window[1] <= len(positions)
            or type(anchor) is not int or not source_window[0] <= anchor < source_window[1]):
        raise ValueError("Invalid owned source window or anchor")
    source_start, source_end = source_window
    plan = plan_owned_anchor_windows([0] * (source_end - source_start), [anchor - source_start], [True], block_size=block_size,
                                     label_shift=1, windows=windows, world_size=1)[0][0]
    if (type(start) is not int or type(end) is not int or not source_start <= start <= source_start + plan["start"]
            or not source_start + plan["end"] <= end <= source_end):
        raise ValueError("Owned window omits required context or labels")
    mask = tag["request"]["loss_mask"]
    supervised = torch.tensor([mask[positions[anchor]], mask[positions[min(anchor + 1, source_end - 1)]]],
                              dtype=torch.float32) > 0
    if anchor + 1 >= source_end or not bool(supervised.all()):
        raise ValueError("Owned window anchor is not supervised")
    deadline = time.monotonic() + read_timeout
    if max_output_bytes:
        _, _, width = store.hidden_shape(tag["hidden_states_path"], timeout=max(deadline - time.monotonic(), 0.0))
        aux = len(feature_contract.vllm_aux_hidden_state_layer_ids or feature_contract.target_layer_ids)
        final = int(feature_contract.hidden_states_layout.endswith("_plus_last"))
        if 2 * ((end - start) * (48 + 2 * width * (aux + final)) + 9) > max_output_bytes:
            raise ValueError("Planned owned window exceeds consumer_prefetch_batch_bytes")
    chunk = _prepare_chunk(tag, feature_contract=feature_contract, algorithm="DSPARK", max_window=None,
                           store=store, read_deadline=deadline, partial_file_reads=True,
                           owned_window=(start, end))
    batch = _pad([chunk])
    batch["label_ids"] = batch["input_ids"].clone()
    batch["label_mask"] = batch["loss_mask"].clone()
    batch["document_ids"] = torch.zeros_like(batch["input_ids"])
    batch["preselected_anchor_positions"] = torch.tensor([[anchor - start]], dtype=torch.long)
    batch["preselected_anchor_keep_mask"] = torch.ones((1, 1), dtype=torch.bool)
    batch["hidden_source"] = "standalone_vllm_reference"
    batch["_speco_pad_size"] = 0
    if feature_contract.hidden_states_layout.endswith("_plus_last"):
        batch["last_hidden_state_norm"] = "raw_target_residual"
    return batch


@torch.no_grad()
def scan_owned_anchor_inputs(tag, feature_contract, *, max_window=None, clip_value=1.0e4,
                             chunk_rows=512, read_timeout=30.0, hidden_states_store=None):
    """Build pre-sampling inputs after a bounded scan, without retaining hidden rows."""
    store = hidden_states_store if hidden_states_store is not None else FileHiddenStatesStore()
    if _validate_reference_contracts([tag], feature_contract, store) != "DSPARK":
        raise ValueError("Owned anchor scanning currently requires the block training backend")
    if max_window is not None and (type(max_window) is not int or max_window < 0):
        raise ValueError("Owned anchor max_window must be a non-negative integer")
    request = tag["request"]
    positions = request["feature_positions"]
    original_mask = torch.tensor([request["loss_mask"][position] for position in positions], dtype=torch.float32)
    start, end = _window((original_mask > 0).float(), max_window)
    selected = positions[start:end]
    aux_ids = (feature_contract.vllm_aux_hidden_state_layer_ids
               if feature_contract.vllm_aux_hidden_state_layer_ids is not None else feature_contract.target_layer_ids)
    final_id = feature_contract.target_num_hidden_layers
    needs_final = feature_contract.hidden_states_layout.endswith("_plus_last")
    required = len(aux_ids) + int(needs_final and (final_id is None or final_id not in aux_ids))
    bad = store.scan_bad_auxiliary_rows(
        tag["hidden_states_path"], all_positions=positions, selected_positions=selected,
        prompt_ids=request["vllm_prompt_token_ids"], auxiliary_layers=len(aux_ids), required_layers=required,
        feature_dtype=feature_contract.dtype, clip_value=clip_value, chunk_rows=chunk_rows, timeout=read_timeout,
    )
    invalid = torch.tensor(bad, dtype=torch.bool)
    loss_mask = (original_mask[start:end] > 0).float().masked_fill(invalid, 0.0).unsqueeze(0)
    return {
        "input_ids": torch.tensor([[request["input_ids"][position] for position in selected]], dtype=torch.long),
        "loss_mask": loss_mask,
        "label_mask": loss_mask.clone(),
        "attention_mask": (~invalid).long().unsqueeze(0),
        "feature_window": (start, end),
        "supervised_tokens": int(torch.count_nonzero(original_mask[start:end] > 0).item()),
    }


@torch.no_grad()
def sample_owned_anchor_windows(tags, feature_contract, model, *, device, windows,
                                max_window=None, clip_value=1.0e4, chunk_rows=512, read_timeout=30.0,
                                padding_rank=False, token_counts=None, packing=False,
                                packing_block_size=None, max_packed_len=0, vocabulary=None, hidden_states_store=None):
    """Sample once using the original padded batch geometry, then plan file windows.

    It does not materialize parts or distribute them across ranks; callers
    must retain every source reference.
    """
    if (not isinstance(tags, (tuple, list)) or not tags or type(padding_rank) is not bool
            or type(packing) is not bool or type(max_packed_len) is not int or max_packed_len < 0):
        raise ValueError("Owned anchor sampling requires a nonempty sample batch")
    if packing_block_size is None:
        packing_block_size = model.block_size
    if type(packing_block_size) is not int or packing_block_size <= 0:
        raise ValueError("Owned anchor packing block size must be positive")
    scans = [scan_owned_anchor_inputs(tag, feature_contract, max_window=max_window,
                                     clip_value=clip_value, chunk_rows=chunk_rows,
                                     read_timeout=read_timeout, hidden_states_store=hidden_states_store) for tag in tags]
    lengths, placements, shape = _stream_plan(tags, algorithm="DSPARK", packing=packing,
                                               block_size=packing_block_size, max_packed_len=max_packed_len,
                                               max_window=max_window)
    if any(item["loss_mask"].size(1) != length for item, length in zip(scans, lengths, strict=True)):
        raise ValueError("Owned anchor scan differs from the batch geometry")
    if vocabulary is not None:
        if not isinstance(vocabulary, dict) or type(vocabulary.get("size")) is not int or vocabulary["size"] <= 0:
            raise ValueError("Invalid restricted vocabulary size")
        present = torch.zeros(vocabulary["size"], dtype=torch.bool)
        for item in scans:
            ids = item["input_ids"].reshape(-1)
            present[ids[(ids >= 0) & (ids < vocabulary["size"])]] = True
        if math.prod(shape) > sum(lengths):
            present[0] = True
        vocabulary["ids"] = torch.nonzero(present).flatten()
    if token_counts is not None:
        if not isinstance(token_counts, dict):
            raise ValueError("Owned anchor token counts must be a mapping")
        token_counts.update({
            "source_tokens": sum(len(tag["request"]["input_ids"]) for tag in tags),
            "window_tokens": sum(lengths),
            "supervised_tokens": sum(item["supervised_tokens"] for item in scans),
            "padding_tokens": math.prod(shape) - sum(lengths),
            "samples": len(tags),
        })
        if padding_rank:
            token_counts.update({key: 0 for key in token_counts})
            token_counts["padding_tokens"] = math.prod(shape)
    masks = torch.zeros(shape, dtype=torch.float32, device=device)
    documents = torch.full(shape, -1, dtype=torch.long, device=device) if packing else None
    for item, length, (row, offset, _, document) in zip(scans, lengths, placements, strict=True):
        mask = item["loss_mask"][0]
        masks[row, offset:offset + length] = mask.to(device=device)
        if documents is not None:
            documents[row, offset:offset + length] = document
    if padding_rank:
        masks.zero_()
    anchors, keep = model._sample_anchor_positions(shape[1], masks, torch.device(device), document_ids=documents)
    positions = anchors.detach().cpu().tolist()
    active = keep.detach().cpu().tolist()
    result = []
    by_row = [[] for _ in range(shape[0])]
    for index, (length, (row, offset, _, _)) in enumerate(zip(lengths, placements, strict=True)):
        by_row[row].append((offset, offset + length, index))
    for row, (row_positions, row_active) in enumerate(zip(positions, active, strict=True)):
        for slot, (anchor, selected) in enumerate(zip(row_positions, row_active, strict=True)):
            if not selected:
                continue
            matches = [(start, index) for start, end, index in by_row[row] if start <= anchor < end]
            if len(matches) != 1:
                raise ValueError("Sampled anchor does not identify one source document")
            offset, sample_index = matches[0]
            item = scans[sample_index]
            source_start, source_end = item["feature_window"]
            plan = plan_owned_anchor_windows(
                [0] * (source_end - source_start), [anchor - offset], [True],
                block_size=model.block_size, label_shift=1, windows=windows, world_size=1,
            )[0][0]
            result.append({"sample_index": sample_index, "batch_row": row, "anchor_slot": slot,
                           "anchor": source_start + plan["anchor"],
                           "start": source_start + plan["start"], "end": source_start + plan["end"],
                           "source_window": (source_start, source_end)})
    return anchors, keep, result


@torch.no_grad()
def prepare_empty_owned_window(tag, feature_contract, *, read_timeout=30.0, max_output_bytes=0, hidden_states_store=None):
    """Build a graph-compatible zero-loss part without reading a full sample."""
    batch = prepare_standalone_cpu_batch([tag], feature_contract, max_window=1, pin_memory=False,
                                         read_timeout=read_timeout, streaming=True,
                                         max_output_bytes=max_output_bytes, hidden_states_store=hidden_states_store)
    batch["loss_mask"].zero_()
    batch["label_mask"].zero_()
    batch["document_ids"] = torch.zeros_like(batch["input_ids"])
    batch["preselected_anchor_positions"] = torch.zeros((1, 1), dtype=torch.long)
    batch["preselected_anchor_keep_mask"] = torch.zeros((1, 1), dtype=torch.bool)
    return batch


def _stream_batch(tags, prepare, *, algorithm, packing, plan):
    """Materialize a planned batch while holding at most one sample."""
    lengths, placements, shape = plan
    batch = None
    supervised = 0
    for tag, length, (row, offset, padded, document) in zip(tags, lengths, placements, strict=True):
        chunk = prepare(tag)
        if chunk["input_ids"].numel() != length:
            raise ValueError("Reference feature length differs from planned metadata")
        if batch is None:
            batch = _batch_buffers(chunk, shape)
            if packing:
                batch["document_ids"] = torch.full(shape, -1, dtype=torch.long, device=chunk["input_ids"].device)
            batch["attention_mask"] = torch.zeros(shape, dtype=torch.long, device=chunk["input_ids"].device)
        _copy_chunk(batch, chunk, row, offset)
        supervised += int(torch.count_nonzero(chunk["loss_mask"]).item())
        if packing:
            batch["document_ids"][row, offset:offset + length] = document
        batch["attention_mask"][row, offset:offset + padded] = 1
        del chunk
    if algorithm != "EAGLE3" and not packing:
        batch["label_ids"] = batch["input_ids"].clone()
        batch["label_mask"] = batch["loss_mask"].clone()
    return batch, sum(lengths), supervised


@torch.no_grad()
def _prepare_chunk(
    tag, *, feature_contract, algorithm, max_window, store, read_deadline,
    partial_file_reads=False, owned_window=None,
):
    """One scheduler-assigned sample; workers perform CPU work only."""
    reference = (
        # The Mooncake store consumes the full transfer manifest the producer
        # recorded; the bare handle lacks the tensor layout get_sample needs.
        tag["transfer_params"]
        if tag["backend"] == "mooncake"
        else tag["hidden_states_path"] if tag["backend"] == "file" else tag["handle"]
    )
    request = dict(tag["request"])
    row_offset = None
    row_positions = None
    positions = request["feature_positions"]
    remaining = read_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Reference batch read deadline exceeded")
    if partial_file_reads and type(store) is FileHiddenStatesStore:
        start, end = owned_window if owned_window is not None else (0, len(positions))
        if owned_window is None and algorithm != "EAGLE3" and max_window and len(positions) > max_window:
            mask = torch.tensor([request["loss_mask"][position] for position in positions], dtype=torch.float32)
            start, end = _window((mask > 0).float(), max_window)
        selected_positions = positions[start:end]
        payload, row_index = store.load_window(
            reference, all_positions=positions, selected_positions=selected_positions,
            prompt_length=len(request["vllm_prompt_token_ids"]), timeout=remaining, compact=True,
        )
        if isinstance(row_index, list):
            row_positions = row_index
        else:
            row_offset = row_index
        request["feature_positions"] = selected_positions
        request["draft_position_ids"] = request["draft_position_ids"][start:end]
    elif store.backend in {"mooncake", "native"}:
        payload, _ = store.load(reference, timeout=remaining)
    elif type(store) is FileHiddenStatesStore:
        payload, _ = store.load(reference, timeout=remaining)
    else:
        payload, _ = store.load(reference)
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in payload.values()
    ):
        raise ValueError("Reference prefetch requires CPU tensor payloads")
    request["prompt_token_ids"] = request.pop("vllm_prompt_token_ids")
    for name in (
        "input_ids",
        "position_ids",
        "feature_positions",
        "draft_position_ids",
    ):
        request[name] = torch.tensor(request[name], dtype=torch.long, device="cpu")
    request["loss_mask"] = torch.tensor(
        request["loss_mask"], dtype=torch.float32, device="cpu"
    )
    sample = feature_from_vllm_payload(
        payload,
        SimpleNamespace(**request),
        feature_contract,
        normalize_final_hidden=False,
        split_hidden_states=True,
        validate_finite=False,
        hidden_row_offset=row_offset,
        hidden_row_positions=row_positions,
    )
    # Match backend dtype. Preserve values until training sanitation so it
    # can identify over-limit rows and mask their loss, not just clip them.
    aux = sample.hidden_states
    if not isinstance(aux, torch.Tensor) or sample.position_ids is None:
        raise ValueError("Reference preparation requires tensor features and positions")
    final = sample.last_hidden_states
    ids, mask = sample.input_ids, (sample.loss_mask > 0).float()
    if algorithm == "EAGLE3":
        if final is None:
            raise ValueError("EAGLE3 file input requires final hidden states")
        length = ids.numel() - 2
        if length <= 0:
            raise ValueError("EAGLE3 requires at least three feature rows")
        return {
            "input_ids": ids[1 : 1 + length],
            "hidden_states": aux[:length],
            "loss_mask": mask[2 : 2 + length],
            "position_ids": sample.position_ids[:length],
            "last_hidden_states": final[1 : 1 + length],
        }
    else:
        start, end = _window(mask, max_window)
        chunk = {
            "input_ids": ids[start:end],
            "hidden_states": aux[start:end],
            "loss_mask": mask[start:end],
            # Existing block backends reset positions after cropping.
            "position_ids": torch.arange(end - start, dtype=torch.long, device="cpu"),
        }
        if final is not None:
            chunk["target_last_hidden_states"] = final[start:end]
        return chunk


def _window(mask: torch.Tensor, max_window: int | None) -> tuple[int, int]:
    if not max_window:
        return 0, mask.numel()
    active = torch.nonzero(mask).flatten()
    start = (
        min(max(int(active[0]) - max_window // 2, 0), max(0, mask.numel() - max_window))
        if active.numel()
        else max(0, mask.numel() - max_window)
    )
    return start, min(start + max_window, mask.numel())


def _batch_buffers(
    example: dict[str, torch.Tensor], rows: tuple[int, int], *, cast_hidden_bfloat: bool = True,
    flatten_hidden: bool = True,
) -> dict[str, torch.Tensor]:
    """Allocate final buffers once; hidden conversion happens during copy_."""
    buffers = {}
    for key, tensor in example.items():
        trailing = tuple(tensor.shape[1:])
        if flatten_hidden and key == "hidden_states" and tensor.dim() == 3:
            trailing = (tensor.size(1) * tensor.size(2),)
        dtype = torch.bfloat16 if cast_hidden_bfloat and "hidden_states" in key else tensor.dtype
        buffers[key] = torch.zeros((*rows, *trailing), dtype=dtype, device=tensor.device)
    return buffers


def _copy_chunk(
    batch: dict[str, torch.Tensor],
    chunk: dict[str, torch.Tensor],
    row: int,
    offset: int,
) -> None:
    length = chunk["input_ids"].numel()
    for key, tensor in chunk.items():
        # Match the unflattened source through a destination view: no aux copy.
        destination = batch[key][row, offset : offset + length]
        destination.view(tensor.shape).copy_(tensor)


def _concatenate(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    total = sum(c["input_ids"].numel() for c in chunks)
    batch = _batch_buffers(chunks[0], (1, total))
    offset = 0
    for chunk in chunks:
        _copy_chunk(batch, chunk, 0, offset)
        offset += chunk["input_ids"].numel()
    return batch


def _pad(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    longest = max(c["input_ids"].numel() for c in chunks)
    batch = _batch_buffers(chunks[0], (len(chunks), longest))
    batch["attention_mask"] = torch.zeros_like(batch["input_ids"])
    for row, chunk in enumerate(chunks):
        _copy_chunk(batch, chunk, row, 0)
        batch["attention_mask"][row, : chunk["input_ids"].numel()] = 1
    return batch


def _pack(
    chunks: list[dict[str, torch.Tensor]], block_size: int, limit: int,
    *, cast_hidden_bfloat: bool = True, flatten_hidden: bool = True,
) -> dict[str, torch.Tensor]:
    """Pack every document into ordered rows without exceeding the row limit."""
    rows: list[list[tuple[dict[str, torch.Tensor], int]]] = [[]]
    lengths = [0]
    for chunk in chunks:
        length = chunk["input_ids"].numel()
        padded = length + (-length) % block_size
        if limit and padded > limit:
            raise ValueError("One sample exceeds max_packed_len after block alignment")
        if limit and lengths[-1] + padded > limit:
            rows.append([])
            lengths.append(0)
        rows[-1].append((chunk, padded))
        lengths[-1] += padded
    if not chunks:
        raise ValueError("No samples fit max_packed_len")
    shape = (len(rows), max(lengths))
    batch = _batch_buffers(chunks[0], shape, cast_hidden_bfloat=cast_hidden_bfloat,
                           flatten_hidden=flatten_hidden)
    device = chunks[0]["input_ids"].device
    batch["document_ids"] = torch.full(shape, -1, dtype=torch.long, device=device)
    batch["attention_mask"] = torch.zeros(shape, dtype=torch.long, device=device)
    for row, packed in enumerate(rows):
        offset = 0
        for document, (chunk, padded) in enumerate(packed):
            _copy_chunk(batch, chunk, row, offset)
            batch["document_ids"][row, offset : offset + chunk["input_ids"].numel()] = document
            offset += padded
        batch["attention_mask"][row, :offset] = 1
    return batch
