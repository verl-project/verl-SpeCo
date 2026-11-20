# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Idempotent cleanup for prepared or discarded standalone samples."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
from verl_speco.trainer.standalone_retry import retry_io
from verl_speco.transport.drafter_sample_protocol import validate_reference_envelope

logger = logging.getLogger(__name__)


def cleanup_standalone_samples(
    store: Any, entries: Sequence[Any], *, attempts: int = 3, reason: str = "trained"
) -> list[str]:
    """Clear TQ first when used, then release hidden references.

    All references are validated before mutation. An uncertain TQ clear preserves
    payloads. Release failures have independent retries and never republish data.
    """
    if not entries:
        return []
    if not getattr(store, "uses_transfer_queue", True) and not getattr(
        store, "release_hidden_states_after_use", True
    ):
        # Permanent file references have no storage cleanup. The caller still
        # completes scheduler bookkeeping and sends its training acknowledgement.
        return []
    references = []
    hidden_store = getattr(store, "hidden_states_store", None)
    if hidden_store is None:
        hidden_store = FileHiddenStatesStore()
    for entry in entries:
        if entry.tag.get("payload_kind") == "vllm_reference":
            validate_reference_envelope(entry.tag)
            if entry.tag["backend"] != hidden_store.backend:
                raise ValueError("Cleanup reference backend does not match store")
            references.append(
                entry.tag["handle"]
                if entry.tag["backend"] == "mooncake"
                else entry.tag["hidden_states_path"]
            )
    if getattr(store, "uses_transfer_queue", True):
        retry_io(
            lambda: store.clear_many([e.key for e in entries]),
            name=f"TQ cleanup ({reason})",
            attempts=attempts,
            retry_if=lambda exc: True,
        )
    if not getattr(store, "release_hidden_states_after_use", True):
        # Completion releases scheduler metadata, never permanent dataset files.
        return []
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
