# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Idempotent cleanup for prepared or discarded standalone samples."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
from verl_speco.trainer.tq_retry import retry_io
from verl_speco.transport.drafter_sample_protocol import validate_reference_tag

logger = logging.getLogger(__name__)


def cleanup_standalone_samples(
    store: Any, entries: Sequence[Any], *, attempts: int = 3, reason: str = "trained"
) -> list[str]:
    """Clear TQ first when used, then release hidden references.

    All references are validated before mutation. An uncertain TQ clear preserves
    payloads. Release failures have independent retries and never republish data.
    """
    references = []
    hidden_store = getattr(store, "hidden_states_store", None)
    if hidden_store is None:
        hidden_store = FileHiddenStatesStore()
    for entry in entries:
        if entry.tag.get("payload_kind") == "vllm_reference":
            validate_reference_tag(entry.tag)
            if entry.tag["backend"] != hidden_store.backend:
                raise ValueError("Cleanup reference backend does not match store")
            references.append(
                entry.tag["handle"]
                if entry.tag["backend"] == "mooncake"
                else entry.tag["hidden_states_path"]
            )
    if not entries:
        return []
    if getattr(store, "uses_transfer_queue", True):
        retry_io(
            lambda: store.clear_many([e.key for e in entries]),
            name=f"TQ cleanup ({reason})",
            attempts=attempts,
            retry_if=lambda exc: True,
        )
    remaining = []
    for reference in dict.fromkeys(references):
        try:
            retry_io(
                partial(hidden_store.release, reference),
                name=f"Hidden reference cleanup ({reason})",
                attempts=attempts,
                retry_if=lambda exc: True,
            )
        except Exception:
            remaining.append(reference)
            logger.exception(
                "Completed sample; hidden reference cleanup exhausted retries: %s",
                reference,
            )
    return remaining
