# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU-only file reading, alignment, collation and pinning for standalone prefetch."""

from __future__ import annotations

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
    read_deadline = time.monotonic() + read_timeout
    algorithm = feature_contract.algorithm.upper()
    if algorithm not in {"DSPARK", "DFLASH", "EAGLE3"} or feature_contract.use_logits:
        raise ValueError(
            "Reference prefetch supports DSPARK/DFLASH/EAGLE3 hidden supervision only"
        )
    if not feature_contract.require_full_alignment:
        raise ValueError("Reference prefetch requires full feature alignment")
    valid_layouts = {
        "DSPARK": {"dflash_aux", "dflash_aux_plus_last"},
        "DFLASH": {"dflash_aux"},
        "EAGLE3": {"eagle3_aux_plus_last"},
    }
    if feature_contract.hidden_states_layout not in valid_layouts[algorithm]:
        raise ValueError("Algorithm and hidden layout do not match")
    if not tags or (max_window is not None and max_window < 0):
        raise ValueError("A non-empty batch and non-negative max_window are required")
    if block_size <= 0 or max_packed_len < 0:
        raise ValueError("Invalid packing limits")
    if algorithm == "EAGLE3" and packing:
        raise ValueError("Block packing is not an EAGLE3 mode")

    expected = asdict(feature_contract)
    expected["dtype"] = str(feature_contract.dtype)
    # Validate every reference before opening files; never combine contracts.
    for tag in tags:
        validate_reference_tag(tag)
        if tag["backend"] != store.backend:
            raise ValueError("Hidden reference backend does not match consumer store")
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

    prepare = partial(
        _prepare_chunk,
        feature_contract=feature_contract,
        algorithm=algorithm,
        max_window=max_window,
        store=store,
        read_deadline=read_deadline,
    )
    chunks = (
        list(executor.map(prepare, tags))
        if executor is not None
        else list(map(prepare, tags))
    )

    if algorithm == "EAGLE3":
        batch = _concatenate(chunks)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    elif packing:
        batch = _pack(chunks, block_size, max_packed_len)
    else:
        batch = _pad(chunks)
        batch["label_ids"] = batch["input_ids"].clone()
        batch["label_mask"] = batch["loss_mask"].clone()
    if pin_memory:
        batch = {key: value.pin_memory() for key, value in batch.items()}
    batch["hidden_source"] = "standalone_vllm_reference"
    batch["_speco_pad_size"] = 0
    if feature_contract.hidden_states_layout.endswith("_plus_last"):
        batch["last_hidden_state_norm"] = "raw_target_residual"
    return batch


@torch.no_grad()
def _prepare_chunk(
    tag, *, feature_contract, algorithm, max_window, store, read_deadline
):
    """One scheduler-assigned sample; workers perform CPU work only."""
    reference = (
        tag["handle"] if tag["backend"] == "mooncake" else tag["hidden_states_path"]
    )
    if store.backend == "mooncake":
        remaining = read_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Mooncake batch read deadline exceeded")
        payload, _ = store.load(reference, timeout=remaining)
    else:
        payload, _ = store.load(reference)
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in payload.values()
    ):
        raise ValueError("Reference prefetch requires CPU tensor payloads")
    request = dict(tag["request"])
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
    example: dict[str, torch.Tensor], rows: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Allocate final buffers once; hidden conversion happens during copy_."""
    buffers = {}
    for key, tensor in example.items():
        trailing = tuple(tensor.shape[1:])
        if key == "hidden_states" and tensor.dim() == 3:
            trailing = (tensor.size(1) * tensor.size(2),)
        dtype = torch.bfloat16 if "hidden_states" in key else tensor.dtype
        buffers[key] = torch.zeros((*rows, *trailing), dtype=dtype, device="cpu")
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
    chunks: list[dict[str, torch.Tensor]], block_size: int, limit: int
) -> dict[str, torch.Tensor]:
    packed: list[tuple[dict[str, torch.Tensor], int]] = []
    total = 0
    for chunk in chunks:
        length = chunk["input_ids"].numel()
        padded = length + (-length) % block_size
        if limit and total + padded > limit:
            continue
        packed.append((chunk, padded))
        total += padded
    if not packed:
        raise ValueError("No samples fit max_packed_len")
    batch = _batch_buffers(packed[0][0], (1, total))
    batch["document_ids"] = torch.full((1, total), -1, dtype=torch.long, device="cpu")
    offset = 0
    for document, (chunk, padded) in enumerate(packed):
        _copy_chunk(batch, chunk, 0, offset)
        batch["document_ids"][0, offset : offset + chunk["input_ids"].numel()] = (
            document
        )
        offset += padded
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return batch
