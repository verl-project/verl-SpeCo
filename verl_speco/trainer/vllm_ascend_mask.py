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
"""SpeCo supervision semantics for the in-package DSpark implementation.

All masks are 1=supervised and 0=ignored, including padding and context overlap.
Only the native runner's argument name ``raw_mask`` is retained for ABI reuse.
No tensors are inverted and no external modules or files are modified.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def build_segments(dataset):
    """Preserve source windows and loss ownership using positive supervision."""
    from verl_speco.data.deepseek_v4_sampler import MTPSegment

    segment_type = MTPSegment
    segments = []
    for index, sample in enumerate(dataset.samples):
        mask = sample["mask"]
        if dataset.segment_length is None:
            end = min(len(mask), dataset.max_len) if dataset.max_len else len(mask)
            segments.append(segment_type(index, 0, end, 0, end))
            continue
        stride = dataset.segment_length - dataset.segment_overlap
        for start in range(0, len(mask), stride):
            end = min(start + dataset.segment_length, len(mask))
            loss_start = start + dataset.segment_overlap if start else 0
            if loss_start >= end:
                break
            if (
                sum(value > 0.5 for value in mask[loss_start:end])
                >= dataset.segment_min_loss_tokens
            ):
                segments.append(segment_type(index, start, end, loss_start, end))
    return segments


def estimate_anchor_capacity(
    mask, segment, *, num_anchors, min_anchor_spacing, target_offset=2
):
    """Count spaced anchors with supervised predecessor and target tokens."""
    if num_anchors <= 0 or min_anchor_spacing <= 0 or target_offset < 1:
        raise ValueError("Anchor count, spacing and target offset must be positive")
    supervised = [False] * (segment.end - segment.start)
    for position in range(segment.loss_start, segment.loss_end):
        supervised[position - segment.start] = mask[position] > 0.5
    selected, last = 0, -min_anchor_spacing
    for anchor in range(max(0, len(supervised) - target_offset)):
        if (
            supervised[anchor + target_offset - 1]
            and supervised[anchor + target_offset]
            and anchor - last >= min_anchor_spacing
        ):
            selected, last = selected + 1, anchor
            if selected >= num_anchors:
                break
    return selected


def collate_features(batch, pad_to_length=None):
    """Pad features and SpeCo supervision masks with zero."""
    observed = max(item["input_ids"].size(0) for item in batch)
    length = int(pad_to_length or observed)
    if length < observed:
        raise ValueError("pad_to_length is smaller than the longest sequence")
    result = {}
    for key in (
        "input_ids",
        "positions",
        "hidden_states",
        "target_head_hidden_states",
        "loss_mask",
    ):
        values = []
        for item in batch:
            value = item[key]
            padding = length - value.size(0)
            values.append(
                F.pad(
                    value,
                    (0, padding) if value.ndim == 1 else (0, 0, 0, padding),
                    value=0,
                )
            )
        result[key] = torch.stack(values)
    # Reuse the native runner ABI without adopting its mask semantics.
    result["raw_mask"] = result["loss_mask"]
    return result


def prepare_training_inputs(
    self, *, input_ids, raw_mask, positions, num_anchors, generator=None
):
    """Use native target builders directly with a SpeCo supervision mask."""
    if input_ids.shape != raw_mask.shape or positions.shape != input_ids.shape:
        raise ValueError(
            "DSpark tokens, loss mask and positions must have equal shapes"
        )
    from verl_speco.models.deepseek_v4 import modeling_dspark as api

    spacing = getattr(self._training_config, "dspark_anchor_min_spacing", None)
    spacing = self.runtime_contract.block_size if spacing is None else int(spacing)
    if spacing <= 0:
        raise ValueError("dspark_anchor_min_spacing must be positive")
    targets = api.build_block_targets(
        input_ids,
        raw_mask,
        block_size=self.runtime_contract.block_size,
        num_anchors=num_anchors,
        generator=generator,
        target_offset=self.runtime_contract.target_offset,
        min_anchor_spacing=spacing,
    )
    runtime_inputs = api.build_runtime_block_inputs(
        input_ids,
        targets,
        noise_token_id=int(self.runtime_contract.noise_token_id),
        positions=positions,
    )
    return targets, runtime_inputs


def install_speco_mask(model):
    """Override this instance's input preparation; retain native parameter names."""
    if not callable(getattr(model, "prepare_training_inputs", None)):
        raise TypeError("DSpark model must expose prepare_training_inputs")
    model.prepare_training_inputs = MethodType(prepare_training_inputs, model)
