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
"""One full-attention forward for RL trajectory logprobs and draft branches.

This module deliberately has no verl/vLLM imports. It is the dense reference
executor for the actor-side joint-scoring contract, not a FlashAttention or
recurrent-state implementation. Token IDs are already sampled; no generation or
rejection sampling takes place here.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class DraftBranch:
    branch_id: str
    anchor: int  # Zero-based token index on the unpadded trunk.
    token_ids: tuple[int, ...]  # Full valid proposal, excluding anchor/bonus.
    source: str = "proposal"
    accepted_count: int = -1  # -1 for unknown legacy diagnostics or trunk anchors.
    verified_count: int | None = None


@dataclass(frozen=True)
class JointScoringRequest:
    sample_id: str
    trunk_ids: tuple[int, ...]  # Exact prompt + final rollout response.
    response_start: int  # Index of first response token, not prediction row.
    branches: tuple[DraftBranch, ...] = ()

    def packed_size(self, sequence_multiple):
        """Return logical token count and SP-aligned mask width."""
        length = len(self.trunk_ids) + sum(len(b.token_ids) - 1 for b in self.branches)
        return length, (
            (length + sequence_multiple - 1) // sequence_multiple
        ) * sequence_multiple


@dataclass
class PackedJointBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    branch_rows: tuple[tuple[tuple[int, ...], ...], ...]
    requests: tuple[JointScoringRequest, ...]
    logical_rows: int
    independent_rows: int


@dataclass
class BranchScores:
    branch_id: str
    target_logits: torch.Tensor | None  # Optional raw predecessor logits.
    target_logz: (
        torch.Tensor | None
    )  # Optional logsumexp(raw logits / target_temperature).
    target_hidden: torch.Tensor


@dataclass
class JointScores:
    sample_id: str
    response_log_probs: torch.Tensor
    branches: tuple[BranchScores, ...]
    # HF hidden-state tuple indices: index 0 is embeddings, final index is normed.
    trunk_features: dict[int, torch.Tensor]
    response_entropy: torch.Tensor | None = None


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_joint_batch(
    requests,
    *,
    dtype,
    vocab_size,
    max_packed_tokens=8192,
    max_mask_elements=16_777_216,
    sequence_multiple=1,
    max_position_embeddings=None,
):
    """Check a local packed tree layout without allocating quadratic masks."""
    if not requests:
        raise ValueError("joint scoring requires at least one request")
    if dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        raise ValueError("attention mask requires a floating dtype")
    lengths = []
    seen_samples = set()
    independent_rows = 0
    for request in requests:
        if not request.sample_id or request.sample_id in seen_samples:
            raise ValueError("sample IDs must be nonempty and unique")
        seen_samples.add(request.sample_id)
        n = len(request.trunk_ids)
        if (
            n < 1
            or not _integer(request.response_start)
            or not 1 <= request.response_start <= n
        ):
            raise ValueError(
                "response_start must follow a nonempty prompt on the trunk"
            )
        ids = list(request.trunk_ids)
        seen_branches = set()
        length = n
        independent_rows += n
        for branch in request.branches:
            if not branch.branch_id or branch.branch_id in seen_branches:
                raise ValueError(
                    "branch IDs must be nonempty and unique within a sample"
                )
            seen_branches.add(branch.branch_id)
            if (
                not _integer(branch.anchor)
                or not request.response_start - 1 <= branch.anchor < n
            ):
                raise ValueError("anchor must be at the prompt end or on the response")
            if not branch.token_ids:
                raise ValueError("a proposal must contain at least one valid token")
            if (
                branch.source not in {"proposal", "trunk"}
                or not _integer(branch.accepted_count)
                or not -1 <= branch.accepted_count <= len(branch.token_ids)
                or (
                    branch.verified_count is not None
                    and (
                        not _integer(branch.verified_count)
                        or not 0
                        <= branch.accepted_count
                        <= branch.verified_count
                        <= len(branch.token_ids)
                    )
                )
            ):
                raise ValueError("invalid branch source or acceptance diagnostics")
            if (
                branch.source == "trunk"
                and tuple(
                    request.trunk_ids[
                        branch.anchor + 1 : branch.anchor + 1 + len(branch.token_ids)
                    ]
                )
                != branch.token_ids
            ):
                raise ValueError(
                    "trunk anchor tokens differ from the rollout trajectory"
                )
            ids.extend(branch.token_ids)
            length += len(branch.token_ids) - 1
            independent_rows += branch.anchor + len(branch.token_ids)
        if any(not _integer(token) or not 0 <= token < vocab_size for token in ids):
            raise ValueError("token IDs must be integers inside the model vocabulary")
        max_position = max(
            [n - 1] + [b.anchor + len(b.token_ids) - 1 for b in request.branches]
        )
        if (
            max_position_embeddings is not None
            and max_position >= max_position_embeddings
        ):
            raise ValueError(
                "joint actor exceeds max_position_embeddings; prefixes must not be truncated"
            )
        lengths.append(length)
    if not _integer(sequence_multiple) or sequence_multiple < 1:
        raise ValueError("sequence_multiple must be a positive integer")
    width = math.ceil(max(lengths) / sequence_multiple) * sequence_multiple
    if sum(lengths) > max_packed_tokens:
        raise ValueError(
            "joint batch exceeds max_packed_tokens; split into microbatches"
        )
    if len(requests) * width * width > max_mask_elements:
        raise ValueError("dense tree mask exceeds max_mask_elements")

    return lengths, width, independent_rows


def pack_joint_batch(
    requests: Sequence[JointScoringRequest],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    vocab_size: int,
    max_packed_tokens: int = 8192,
    max_mask_elements: int = 16_777_216,
    sequence_multiple: int = 1,
    omit_trunk_only_masks: bool = False,
    build_attention_mask: bool = True,
    max_position_embeddings: int | None = None,
) -> PackedJointBatch:
    """Compile ancestry, semantic RoPE positions and predecessor readouts.

    Each trunk occurs once. Branches append W-1 rows because the anchor scores
    the first proposal token, and the last proposal token has no successor to
    score. Padding queries attend only themselves to avoid all-masked rows.
    With ``omit_trunk_only_masks``, the mask contains only the rows with branches,
    in request order. Trunk-only rows then need implicit causal attention.
    Set ``build_attention_mask=False`` when the executor compiles its own masks.
    """
    requests = tuple(requests)
    lengths, width, independent_rows = validate_joint_batch(
        requests,
        dtype=dtype,
        vocab_size=vocab_size,
        max_packed_tokens=max_packed_tokens,
        max_mask_elements=max_mask_elements,
        sequence_multiple=sequence_multiple,
        max_position_embeddings=max_position_embeddings,
    )

    input_ids = torch.zeros((len(requests), width), dtype=torch.long, device=device)
    positions = torch.zeros_like(input_ids)
    mask_rows = (
        sum(bool(request.branches) for request in requests)
        if omit_trunk_only_masks
        else len(requests)
    )
    allowed = None
    if build_attention_mask and mask_rows:
        allowed = (
            torch.eye(width, dtype=torch.bool, device=device)
            .expand(mask_rows, -1, -1)
            .clone()
        )
    all_rows: list[tuple[tuple[int, ...], ...]] = []
    mask_row = 0
    for batch_index, request in enumerate(requests):
        n = len(request.trunk_ids)
        input_ids[batch_index, :n] = torch.tensor(request.trunk_ids, device=device)
        positions[batch_index, :n] = torch.arange(n, device=device)
        if omit_trunk_only_masks and not request.branches:
            all_rows.append(())
            continue
        if allowed is not None:
            allowed[mask_row, :n, :n] = torch.ones(
                (n, n), dtype=torch.bool, device=device
            ).tril()
        offset = n
        branch_rows = []
        for branch in request.branches:
            count = len(branch.token_ids) - 1
            branch_rows.append((branch.anchor, *range(offset, offset + count)))
            if count:
                input_ids[batch_index, offset : offset + count] = torch.tensor(
                    branch.token_ids[:-1], device=device
                )
                positions[batch_index, offset : offset + count] = torch.arange(
                    branch.anchor + 1, branch.anchor + 1 + count, device=device
                )
                if allowed is not None:
                    allowed[mask_row, offset : offset + count, : branch.anchor + 1] = (
                        True
                    )
                    allowed[
                        mask_row, offset : offset + count, offset : offset + count
                    ] = torch.ones(
                        (count, count), dtype=torch.bool, device=device
                    ).tril()
            offset += count
        all_rows.append(tuple(branch_rows))
        mask_row += 1
    mask = None
    if allowed is not None:
        mask = torch.zeros((mask_rows, 1, width, width), dtype=dtype, device=device)
        mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return PackedJointBatch(
        input_ids,
        positions,
        mask,
        tuple(all_rows),
        requests,
        sum(lengths),
        independent_rows,
    )


def response_scores(logits, labels, temperature, calculate_entropy):
    """Normalize bounded chunks of vocabulary rows in float32."""
    log_probs = logits.new_empty(len(labels), dtype=torch.float32)
    entropy = torch.empty_like(log_probs) if calculate_entropy else None
    for start in range(0, len(labels), 128):
        sl = slice(start, start + 128)
        scaled = logits[sl].float()
        if temperature != 1.0:
            scaled = scaled / temperature
        log_probs[sl] = scaled.gather(-1, labels[sl, None]).squeeze(
            -1
        ) - scaled.logsumexp(-1)
        if entropy is not None:
            log_distribution = scaled.log_softmax(-1)
            entropy[sl] = -(log_distribution.exp() * log_distribution).sum(-1)
    return log_probs, entropy


def parallel_microbatch_sizes(requests, sizes, dp_size, sp_size, options):
    """Apply token/mask limits per DP rank, keeping each tree in one forward."""
    token_budget = options.get("max_packed_tokens", 8192)
    mask_budget = options.get("max_mask_elements", 16777216)
    local_limit = max(sizes)
    groups, count = [], 0
    totals, widths, counts = [0] * dp_size, [0] * dp_size, [0] * dp_size
    for request in requests:
        length, width = request.packed_size(sp_size)
        if length > token_budget or width * width > mask_budget:
            raise ValueError(
                "complete trajectory exceeds its per-rank token/mask budget"
            )
        rank = count % dp_size
        if (
            totals[rank] + length > token_budget
            or (counts[rank] + 1) * max(widths[rank], width) ** 2 > mask_budget
            or counts[rank] + 1 > local_limit
        ):
            groups.append(count)
            count = 0
            totals, widths, counts = [0] * dp_size, [0] * dp_size, [0] * dp_size
            rank = 0
        totals[rank] += length
        widths[rank] = max(widths[rank], width)
        counts[rank] += 1
        count += 1
    if count:
        groups.append(count)
    return tuple(groups)


def validate_joint_model(config, indices, sp_size=1):
    """Share model restrictions between collective preflight and execution."""
    if config.model_type not in {"qwen2", "qwen3", "llama"}:
        raise ValueError(
            "joint scoring currently supports full-attention Qwen2/Qwen3/Llama only"
        )
    if getattr(config, "_attn_implementation", None) not in (
        {"eager", "sdpa", "flash_attention_2"} if sp_size > 1 else {"eager", "sdpa"}
    ):
        raise ValueError(
            "joint scoring requires eager or SDPA with an explicit 4D mask"
        )
    if getattr(config, "sliding_window", None) is not None or any(
        layer != "full_attention"
        for layer in (getattr(config, "layer_types", None) or ())
    ):
        raise ValueError("sliding-window and hybrid/recurrent models are not supported")
    if any(not _integer(i) or not 0 <= i <= config.num_hidden_layers for i in indices):
        raise ValueError(
            "hidden-state indices must use the HF embedding/layer tuple convention"
        )


def _gather_rows(local, rows, start=0, group=None):
    """Gather readout rows only; the unsharded path is an independent snapshot."""
    rows = torch.as_tensor(rows, dtype=torch.long, device=local.device)
    if group is None:
        return local.index_select(0, rows)
    owned = (rows >= start) & (rows < start + len(local))
    output = local.new_zeros((len(rows), *local.shape[1:]))
    output[owned] = local.index_select(0, rows[owned] - start)
    torch.distributed.all_reduce(output, group=group)
    return output


@torch.no_grad()
def score_joint_batch(
    model,
    requests: Sequence[JointScoringRequest],
    *,
    actor_temperature: float = 1.0,
    target_temperature: float = 1.0,
    hidden_state_indices: Sequence[int] = (),
    calculate_entropy: bool = False,
    return_target_logits: bool = True,
    return_target_logz: bool = True,
    capture_empty_trunk_features: bool = True,
    trunk_feature_prefix_only: bool = False,
    max_packed_tokens: int = 8192,
    max_mask_elements: int = 16_777_216,
    sequence_multiple: int = 1,
    _sp_group=None,
    **sp_options,
) -> tuple[tuple[JointScores, ...], dict[str, int]]:
    """Score trunks and branches in one forward, using HF or Ulysses attention."""
    for temperature in (actor_temperature, target_temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("scoring temperatures must be finite and positive")
    indices = tuple(hidden_state_indices)
    group = _sp_group
    size = torch.distributed.get_world_size(group) if group is not None else 1
    rank = torch.distributed.get_rank(group) if group is not None else 0
    validate_joint_model(model.config, indices, size)
    parameter = next(model.parameters())
    pack_options = dict(
        device=parameter.device,
        dtype=parameter.dtype,
        vocab_size=model.config.vocab_size,
        max_position_embeddings=model.config.max_position_embeddings,
        max_packed_tokens=max_packed_tokens,
        max_mask_elements=max_mask_elements,
    )
    if group is None:
        if sp_options:
            raise TypeError(f"unexpected joint scoring options: {sorted(sp_options)}")
        packed = pack_joint_batch(
            requests, sequence_multiple=sequence_multiple, **pack_options
        )
        attention, capture_hidden = nullcontext(), True
        assert packed.attention_mask is not None
        metrics = {"mask_elements": packed.attention_mask.numel()}
    else:
        from verl_speco.joint_ulysses import prepare_ulysses_batch

        packed, attention, capture_hidden, metrics = prepare_ulysses_batch(
            model,
            requests,
            group,
            pack_options,
            bool(
                any(r.branches for r in requests)
                or (capture_empty_trunk_features and indices)
            ),
            **sp_options,
        )
    width = packed.input_ids.shape[1] // size
    start = rank * width
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with attention:
            output = model(
                input_ids=packed.input_ids[:, start : start + width],
                position_ids=packed.position_ids[:, start : start + width],
                attention_mask=packed.attention_mask if group is None else None,
                use_cache=False,
                output_hidden_states=capture_hidden,
                return_dict=True,
            )
    finally:
        # Preserve mixed child states without recursively resetting dropout.
        for module, training in states:
            module.training = training

    results = []
    for i, request in enumerate(packed.requests):
        n = len(request.trunk_ids)
        rows = torch.arange(request.response_start - 1, n - 1, device=parameter.device)
        owned = (rows >= start) & (rows < start + width)
        labels = torch.tensor(
            request.trunk_ids[request.response_start :],
            device=parameter.device,
            dtype=torch.long,
        )
        local_probs, local_entropy = response_scores(
            output.logits[i].index_select(0, rows[owned] - start),
            labels[owned],
            actor_temperature,
            calculate_entropy,
        )
        responses = [local_probs, local_entropy]
        if group is not None:
            for index, local in enumerate(responses):
                if local is not None:
                    values = local.new_zeros(len(rows))
                    values[owned] = local
                    if len(rows):
                        torch.distributed.all_reduce(values, group=group)
                    responses[index] = values
        readouts = packed.branch_rows[i]
        flat_rows = [row for readout in readouts for row in readout]
        branches = []
        if flat_rows:
            flat_logz = None
            if return_target_logz:
                row_ids = torch.tensor(flat_rows, device=parameter.device)
                owned_readouts = (row_ids >= start) & (row_ids < start + width)
                flat_logz = output.logits.new_zeros(len(flat_rows), dtype=torch.float32)
                flat_logz[owned_readouts] = (
                    output.logits[i]
                    .index_select(0, row_ids[owned_readouts] - start)
                    .float()
                    / target_temperature
                ).logsumexp(-1)
                if group is not None:
                    torch.distributed.all_reduce(flat_logz, group=group)
            flat_hidden = _gather_rows(
                output.hidden_states[-1][i], flat_rows, start, group
            )
            flat_logits = (
                _gather_rows(output.logits[i], flat_rows, start, group)
                if return_target_logits
                else None
            )
            offset = 0
            for branch, readout in zip(request.branches, readouts, strict=True):
                end = offset + len(readout)
                branches.append(
                    BranchScores(
                        branch.branch_id,
                        flat_logits[offset:end] if flat_logits is not None else None,
                        flat_logz[offset:end] if flat_logz is not None else None,
                        flat_hidden[offset:end],
                    )
                )
                offset = end
        feature_rows = (
            max(b.anchor for b in request.branches) + 1
            if trunk_feature_prefix_only and request.branches
            else n
        )
        features = (
            {
                index: _gather_rows(
                    output.hidden_states[index][i], range(feature_rows), start, group
                )
                for index in indices
            }
            if request.branches or capture_empty_trunk_features
            else {}
        )
        results.append(
            JointScores(
                request.sample_id, responses[0], tuple(branches), features, responses[1]
            )
        )
    return tuple(results), {
        "model_forward_calls": 1,
        "logical_rows": packed.logical_rows,
        "independent_rows": packed.independent_rows,
        "padded_rows": packed.input_ids.numel(),
        **metrics,
    }
