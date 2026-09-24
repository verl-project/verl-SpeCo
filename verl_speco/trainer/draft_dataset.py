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
"""Dataset helpers for standalone draft feature stores."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, cast

import torch

from verl_speco.trainer.feature_store import DraftFeatureStore, DraftStoredSample

logger = logging.getLogger(__name__)

FEATURE_ALIGNMENT_WARN = "warn"
FEATURE_ALIGNMENT_STRICT = "strict"


def validate_draft_sample(
    sample: Any,
    *,
    min_supervised_tokens: int = 0,
) -> str | None:
    """Return a rejection reason for a stored draft sample, or ``None``.

    Mirrors the Producer-side checks on the read path: text/mask alignment,
    hidden-state row counts, NaN/Inf, and a minimum supervised-token budget.
    Backend-specific shape validation stays in ``DraftFeatureSample``.
    """
    input_ids = getattr(sample, "input_ids", None)
    loss_mask = getattr(sample, "loss_mask", None)
    if not torch.is_tensor(input_ids) or not torch.is_tensor(loss_mask):
        return "input_ids/loss_mask must be tensors"
    input_ids = cast(torch.Tensor, input_ids)
    loss_mask = cast(torch.Tensor, loss_mask)
    rows = int(input_ids.numel())
    mask_rows = int(loss_mask.numel())
    if mask_rows != rows:
        return f"input_ids/loss_mask length mismatch ({rows} vs {mask_rows})"
    if min_supervised_tokens > 0:
        supervised = int((loss_mask == 1).sum().item())
        if supervised < min_supervised_tokens:
            return (
                f"only {supervised} supervised tokens, below "
                f"min_supervised_tokens={min_supervised_tokens}"
            )

    for field_name in ("hidden_states", "last_hidden_states", "target"):
        value = getattr(sample, field_name, None)
        if value is None:
            continue
        tensors = value if isinstance(value, (list, tuple)) else [value]
        for index, tensor in enumerate(tensors):
            if not torch.is_tensor(tensor):
                continue
            if tensor.dim() == 0 or int(tensor.size(0)) != rows:
                return (
                    f"{field_name}[{index}] has {tuple(tensor.shape)} rows, expected "
                    f"{rows} to match input_ids"
                )
            if torch.is_floating_point(tensor) and not bool(
                torch.isfinite(tensor).all().item()
            ):
                return f"{field_name}[{index}] contains NaN/Inf"

    target_logprobs = getattr(sample, "target_logprobs", None)
    if (
        torch.is_tensor(target_logprobs)
        and torch.is_floating_point(target_logprobs)
        and not bool(torch.isfinite(target_logprobs).all().item())
    ):
        return "target_logprobs contains NaN/Inf"
    return None


def _collective_device() -> torch.device:
    """Device to use for collective ops, matching the process-group backend.

    Accelerator backends reject CPU tensors while gloo rejects accelerator
    tensors, so resolve the device from the backend instead of hard-coding one.
    """
    try:
        import torch.distributed as dist

        backend = str(dist.get_backend()).lower()
    except Exception:  # pragma: no cover - no distributed context
        return torch.device("cpu")
    if backend != "gloo":
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None:
            try:
                device = accelerator.current_accelerator()
            except Exception:
                device = None
            if device is not None:
                return device
    return torch.device("cpu")


@dataclass(frozen=True)
class DraftFeatureDataLoaderConfig:
    batch_size: int
    rank: int = 0
    world_size: int = 1
    shuffle: bool = True
    seed: int = 0
    repeat: bool = True
    min_sample_step: int | None = None
    max_sample_step: int | None = None
    group_by_length: bool = False
    group_by_length_megabatch: int = 8
    # Read-side filters, equivalent to the standalone TQ Producer input path.
    on_error: str = "skip"
    max_consecutive_errors: int = 20
    min_supervised_tokens: int = 1
    strict_token_alignment: str = FEATURE_ALIGNMENT_WARN


class DraftFeatureDataLoader:
    """Small iterable loader over a DraftFeatureStore.

    The first implementation deliberately keeps sharding simple:
    ``rank_keys = keys[rank::world_size]``. Distributed ranks are truncated to
    the same sample count so every rank executes the same number of FSDP
    collectives when the store size is not divisible by ``world_size``.
    """

    def __init__(self, store: DraftFeatureStore, config: DraftFeatureDataLoaderConfig):
        self.store = store
        self.config = config
        rank = int(config.rank)
        world_size = int(config.world_size)
        if world_size <= 0:
            raise ValueError(f"Invalid world_size: {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(
                f"Invalid rank/world_size configuration: rank={rank}, world_size={world_size}"
            )
        if config.on_error not in {"skip", "raise"}:
            raise ValueError(
                f"on_error must be 'skip' or 'raise', got {config.on_error!r}"
            )
        if config.strict_token_alignment not in {
            FEATURE_ALIGNMENT_WARN,
            FEATURE_ALIGNMENT_STRICT,
        }:
            raise ValueError(
                "strict_token_alignment must be 'warn' or 'strict', got "
                f"{config.strict_token_alignment!r}"
            )
        if int(config.min_supervised_tokens) < 0:
            raise ValueError("min_supervised_tokens must be >= 0")
        self.filter_stats = {
            "kept": 0,
            "read_errors": 0,
            "dropped_misaligned": 0,
            "dropped_min_supervised": 0,
        }

    def format_filter_stats(self) -> str:
        stats = self.filter_stats
        return (
            f"kept={stats['kept']} read_errors={stats['read_errors']} "
            f"dropped_misaligned={stats['dropped_misaligned']} "
            f"dropped_min_supervised={stats['dropped_min_supervised']}"
        )

    def _read_filtered(self, keys: list[str]) -> list[DraftStoredSample]:
        """Read and filter samples, applying the read-side policy."""
        samples: list[DraftStoredSample] = []
        consecutive_errors = 0
        for key in keys:
            try:
                sample = self.store.read(key)
            except Exception as error:
                if self.config.on_error == "raise":
                    raise RuntimeError(
                        f"Failed to read feature sample {key}: {error}"
                    ) from error
                consecutive_errors += 1
                self.filter_stats["read_errors"] += 1
                logger.warning(
                    "Skipping unreadable feature sample %s (consecutive: %d/%d): %s",
                    key,
                    consecutive_errors,
                    self.config.max_consecutive_errors,
                    error,
                )
                if consecutive_errors > int(self.config.max_consecutive_errors):
                    raise RuntimeError(
                        f"Exceeded max_consecutive_errors="
                        f"{self.config.max_consecutive_errors} while reading feature "
                        f"store (last: {key}): {error}"
                    ) from error
                continue

            reason = validate_draft_sample(
                sample,
                min_supervised_tokens=int(self.config.min_supervised_tokens),
            )
            if reason is not None:
                if self.config.strict_token_alignment == FEATURE_ALIGNMENT_STRICT:
                    raise ValueError(
                        f"Feature sample {key} failed validation: {reason}"
                    )
                metric = (
                    "dropped_min_supervised"
                    if "min_supervised_tokens" in reason
                    else "dropped_misaligned"
                )
                self.filter_stats[metric] += 1
                logger.warning("Dropping feature sample %s: %s", key, reason)
                continue

            consecutive_errors = 0
            self.filter_stats["kept"] += 1
            samples.append(sample)
        return samples

    def _synchronize_length(
        self, samples: list[DraftStoredSample]
    ) -> list[DraftStoredSample]:
        """Truncate all ranks to the smallest post-filter sample count.

        Each rank reads its own keys, so filtering can leave ranks with
        different sample counts and desync FSDP collectives. When torch
        distributed is initialized we take the global minimum.
        """
        if int(self.config.world_size) <= 1:
            return samples
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - torch always ships distributed
            return samples
        if not (dist.is_available() and dist.is_initialized()):
            return samples
        count = torch.tensor(
            [len(samples)], dtype=torch.long, device=_collective_device()
        )
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
        return samples[: int(count.item())]

    def _sample_in_step_window(self, sample: DraftStoredSample) -> bool:
        if self.config.min_sample_step is None and self.config.max_sample_step is None:
            return True
        step = self._sample_step(sample)
        if step is None:
            return True
        if self.config.min_sample_step is not None and step < int(
            self.config.min_sample_step
        ):
            return False
        if self.config.max_sample_step is not None and step > int(
            self.config.max_sample_step
        ):
            return False
        return True

    @staticmethod
    def _sample_step(sample: DraftStoredSample) -> int | None:
        raw_step = sample.metadata.get("global_step", sample.metadata.get("step"))
        if raw_step is None:
            return None
        try:
            return int(raw_step)
        except (TypeError, ValueError):
            return None

    @property
    def _uses_step_window(self) -> bool:
        return (
            self.config.min_sample_step is not None
            or self.config.max_sample_step is not None
        )

    @staticmethod
    def _sample_seq_length(sample: DraftStoredSample) -> int:
        """Prefer the recorded seq_len; fall back to materialized input_ids."""
        metadata = getattr(sample, "metadata", None) or {}
        for key in ("seq_len", "seq_length", "num_tokens"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        input_ids = getattr(sample, "input_ids", None)
        if input_ids is None:
            return 0
        if hasattr(input_ids, "numel"):
            return int(input_ids.numel())
        return len(input_ids)

    def _maybe_group_by_length(
        self, samples: list[DraftStoredSample]
    ) -> list[DraftStoredSample]:
        """Sort samples by length inside a shuffled megabatch.

        The dataset keys are already shuffled per epoch, so sorting within
        ``group_by_length_megabatch`` batches keeps epoch-level variety while
        making each yielded batch near-uniform in length.
        """
        if not self.config.group_by_length or not samples:
            return samples
        megabatch = max(
            int(self.config.batch_size),
            int(self.config.batch_size)
            * max(1, int(self.config.group_by_length_megabatch)),
        )
        grouped: list[DraftStoredSample] = []
        for start in range(0, len(samples), megabatch):
            chunk = samples[start : start + megabatch]
            grouped.extend(sorted(chunk, key=self._sample_seq_length))
        return grouped

    def __iter__(self) -> Iterator[list[DraftStoredSample]]:
        epoch = 0
        while True:
            keys = list(
                self.store.iter_keys(
                    shuffle=bool(self.config.shuffle),
                    seed=int(self.config.seed) + epoch,
                )
            )
            if not keys:
                return
            rank = int(self.config.rank)
            world_size = int(self.config.world_size)
            if self._uses_step_window:
                samples = [
                    sample
                    for sample in self._read_filtered(keys)
                    if self._sample_in_step_window(sample)
                ]
                if not samples:
                    return
                rank_samples = samples[rank::world_size]
                if world_size > 1:
                    rank_samples = rank_samples[: len(samples) // world_size]
            else:
                rank_keys = keys[rank::world_size]
                if world_size > 1:
                    rank_keys = rank_keys[: len(keys) // world_size]
                rank_samples = self._read_filtered(rank_keys)
            rank_samples = self._synchronize_length(rank_samples)
            rank_samples = self._maybe_group_by_length(rank_samples)
            if not rank_samples:
                raise RuntimeError(
                    f"DraftFeatureDataLoader epoch={epoch} rank={rank} produced no "
                    "usable samples after filtering "
                    f"({self.format_filter_stats()}); check the feature store and "
                    "read-side filters"
                )
            batch: list[DraftStoredSample] = []
            for sample in rank_samples:
                batch.append(sample)
                if len(batch) >= int(self.config.batch_size):
                    yield batch
                    batch = []
            if batch:
                yield batch
            logger.info(
                "DraftFeatureDataLoader epoch=%s rank=%s %s",
                epoch,
                rank,
                self.format_filter_stats(),
            )
            if not self.config.repeat:
                return
            epoch += 1
