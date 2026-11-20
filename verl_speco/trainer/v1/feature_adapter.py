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

"""Small, dependency-light V1 feature representation.

This is intentionally limited to ordinary V1 PPO batches in Phase 1.  Agent
request/turn fields and context-plus-assistant slicing are added in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V1DrafterBatch:
    """Explicit names for the fields needed by future V1 collection hooks."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any | None = None
    hidden_states: Any | None = None
    target_logprobs: Any | None = None
    global_step: int | None = None


def from_transfer_queue_batch(
    batch: Any, *, global_step: int | None = None
) -> V1DrafterBatch:
    """Build a feature view without assuming a legacy upstream batch container.

    ``batch`` may be a TensorDict-like object or a mapping. Missing optional
    fields stay ``None``; no padding or token-window policy is applied here.
    """

    def get(name: str, default=None):
        if hasattr(batch, "get"):
            return batch.get(name, default)
        try:
            return batch[name]
        except (KeyError, TypeError):
            return default

    return V1DrafterBatch(
        input_ids=get("input_ids"),
        attention_mask=get("attention_mask"),
        response_mask=get("response_mask"),
        hidden_states=get("hidden_states"),
        target_logprobs=get("target_logprobs"),
        global_step=global_step,
    )


def _rollout_hidden_isfinite(hidden):
    from verl_speco.producer.feature_export import hidden_states_are_finite

    return hidden_states_are_finite(hidden)


def validate_rollout_samples(batch, global_step, *, allow_stale=False):
    """Fail closed on stale or misrouted TQ side-channel samples."""
    import torch

    values = batch.non_tensor_batch.get("drafter_sample", [])
    if len(values) != len(batch):
        raise ValueError("V1 rollout samples must preserve every TQ request position")
    seen = set()
    for index, sample in enumerate(values):
        if sample is None:
            continue
        if sample.get("hidden_source") != "vllm_rollout":
            raise ValueError("V1 expected a vLLM rollout feature sample")
        source_step = int(sample.get("global_step", -1))
        if (
            source_step < 0
            or source_step > int(global_step)
            or (not allow_stale and source_step != int(global_step))
        ):
            raise ValueError(
                "V1 rollout sample collection step does not match trainer step"
            )
        identity = sample.get("_speco_sample_id")
        if not identity or identity in seen:
            raise ValueError("V1 rollout sample identity is missing or duplicated")
        seen.add(identity)
        response = batch.batch["responses"][index]
        captured = sample["responses"].reshape(-1)
        if captured.numel() > response.numel() or not torch.equal(
            captured.cpu(), response[: captured.numel()].cpu()
        ):
            raise ValueError("V1 rollout sample response does not match its TQ row")
        prompt = sample.get("prompts")
        if prompt is not None and "prompts" in batch.batch:
            source_prompt = batch.batch["prompts"][index]
            prompt = prompt.reshape(-1)
            if (
                prompt.numel() == 0
                or prompt.numel() > source_prompt.numel()
                or not torch.equal(prompt.cpu(), source_prompt[-prompt.numel() :].cpu())
            ):
                raise ValueError("V1 rollout sample prompt does not match its TQ row")
        hidden, positions = sample["hidden_states"], sample["hidden_positions"]
        flat = positions.reshape(-1)
        if flat.numel() == 0 or flat.dtype not in (torch.int32, torch.int64):
            raise ValueError("V1 rollout hidden positions must be nonempty integers")
        if flat.numel() > 1 and not torch.all(flat[1:] == flat[:-1] + 1):
            raise ValueError("V1 rollout hidden positions must be contiguous")
        if prompt is not None:
            start = prompt.numel() - 1
            end = prompt.numel() + captured.numel() - 1
            if int(flat[0]) < start or int(flat[-1]) >= end:
                raise ValueError(
                    "V1 rollout hidden positions are outside the response prediction range"
                )
        if hidden.shape[1] != positions.numel() or not _rollout_hidden_isfinite(hidden):
            raise ValueError("V1 rollout hidden rows are invalid or nonfinite")


def validate_async_rollout_versions(samples, head_paths, current_step):
    """Reject ambiguous provenance before any worker receives a sample."""
    versions = {s.get("target_version") for s in samples if s is not None}
    if versions and (None in versions or len(versions) != 1):
        raise RuntimeError("Async rollout batch requires one explicit target version")
    source = next(iter(versions)) if versions else None
    if source is not None and (
        type(source) is not int or source < 0 or source > int(current_step)
    ):
        raise RuntimeError("Invalid or future async rollout target version")
    if source is not None and (
        source not in head_paths or not head_paths[source].is_file()
    ):
        raise RuntimeError("Async rollout target head snapshot is missing")
    return source
