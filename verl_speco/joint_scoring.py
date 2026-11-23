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
) -> tuple[tuple[JointScores, ...], dict[str, int]]:
    """Evaluate all trunks and branches in ONE model call, without gradients.

    Call on an unpatched HF full-attention causal LM (eager or SDPA). This
    explicit reference API does not accept verl unpadded/fused model forwards,
    sharded/offloaded workers, linear attention, sliding windows, or KV caches.
    Old logprobs have response-token order, with no prompt/branch/padding rows.
    """
    for temperature in (actor_temperature, target_temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("scoring temperatures must be finite and positive")
    config = model.config
    if config.model_type not in {"qwen2", "qwen3", "llama"}:
        raise ValueError(
            "joint scoring currently supports full-attention Qwen2/Qwen3/Llama only"
        )
    if getattr(config, "_attn_implementation", None) not in {"eager", "sdpa"}:
        raise ValueError(
            "joint scoring requires eager or SDPA with an explicit 4D mask"
        )
    if getattr(config, "sliding_window", None) is not None or any(
        layer != "full_attention"
        for layer in (getattr(config, "layer_types", None) or ())
    ):
        raise ValueError("sliding-window and hybrid/recurrent models are not supported")
    indices = tuple(hidden_state_indices)
    if any(not _integer(i) or not 0 <= i <= config.num_hidden_layers for i in indices):
        raise ValueError(
            "hidden-state indices must use the HF embedding/layer tuple convention"
        )
    parameter = next(model.parameters())
    packed = pack_joint_batch(
        requests,
        device=parameter.device,
        dtype=parameter.dtype,
        vocab_size=config.vocab_size,
        max_packed_tokens=max_packed_tokens,
        max_mask_elements=max_mask_elements,
        sequence_multiple=sequence_multiple,
    )
    mask = packed.attention_mask
    assert mask is not None
    if int(packed.position_ids.max()) >= config.max_position_embeddings:
        raise ValueError(
            "a branch exceeds max_position_embeddings; prefixes must not be truncated"
        )
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        output = model(
            input_ids=packed.input_ids,
            position_ids=packed.position_ids,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        # Restore mixed child states, including models whose dropout was already
        # in eval while their parent was in train. No recursive train() reset.
        for module, training in training_states:
            module.training = training

    results = []
    for batch_index, request in enumerate(packed.requests):
        n = len(request.trunk_ids)
        rows = output.logits[batch_index, request.response_start - 1 : n - 1]
        labels = torch.tensor(
            request.trunk_ids[request.response_start :],
            device=rows.device,
            dtype=torch.long,
        )
        log_probs, entropy = response_scores(
            rows, labels, actor_temperature, calculate_entropy
        )
        branches = []
        for branch, readout in zip(
            request.branches, packed.branch_rows[batch_index], strict=True
        ):
            row_indices = torch.tensor(readout, device=rows.device)
            logits = (
                output.logits[batch_index].index_select(0, row_indices)
                if return_target_logits or return_target_logz
                else None
            )
            hidden = output.hidden_states[-1][batch_index].index_select(0, row_indices)
            branches.append(
                BranchScores(
                    branch.branch_id,
                    logits if return_target_logits else None,
                    (logits.float() / target_temperature).logsumexp(-1)
                    if return_target_logz and logits is not None
                    else None,
                    hidden,
                )
            )
        feature_rows = (
            max(b.anchor for b in request.branches) + 1
            if trunk_feature_prefix_only and request.branches
            else n
        )
        features = (
            {
                i: output.hidden_states[i][batch_index, :feature_rows].clone()
                for i in indices
            }
            if request.branches or capture_empty_trunk_features
            else {}
        )
        results.append(
            JointScores(
                request.sample_id, log_probs, tuple(branches), features, entropy
            )
        )
    return tuple(results), {
        "model_forward_calls": 1,
        "logical_rows": packed.logical_rows,
        "independent_rows": packed.independent_rows,
        "padded_rows": packed.input_ids.numel(),
        "mask_elements": mask.numel(),
    }
