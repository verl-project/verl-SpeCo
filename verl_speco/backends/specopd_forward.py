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
"""Fixed-anchor block proposal forward and branch-conditioned distillation."""

import torch
import torch.nn.functional as F

from verl_speco.backends.specopd_loss import specopd_distribution_loss
from verl_speco.specopd_config import specopd_loss_weights


def proposal_loss(wrapper, batch, lm_head_weight):
    ids = batch["input_ids"]
    anchors = batch["anchor_positions"]
    valid = batch["proposal_mask"]
    proposal = batch["proposal_ids"]
    bsz, n = anchors.shape
    if (
        ids.shape[0] != bsz
        or proposal.shape != valid.shape
        or proposal.shape[0] != bsz * n
    ):
        raise ValueError(
            "SpecOPD proposal rows must match the padded trajectory/branch layout"
        )
    if wrapper.block_size != batch["proposal_block_width"]:
        raise ValueError(
            "drafter mask-block width differs from rollout; probabilities cannot be replayed"
        )
    width = proposal.shape[1]
    if width > wrapper.block_size:
        raise ValueError("SpecOPD proposal width exceeds the drafter block capacity")
    keep = valid.any(-1).reshape(bsz, n)
    base = batch["hidden_states"]
    layers = wrapper.draft_model.num_context_layers
    if base.shape[-1] % layers:
        raise ValueError(
            "SpecOPD concatenated auxiliary feature dimensions are invalid"
        )
    context = wrapper.draft_model.extract_context_feature(
        list(base.chunk(layers, dim=-1))
    )
    hidden = wrapper._forward_anchor_blocks(ids, context, anchors, keep).reshape(
        bsz * n, wrapper.block_size, -1
    )[:, :width]
    head = lm_head_weight
    if head.shape[0] != wrapper.draft_model.config.vocab_size:
        raise ValueError("SpecOPD currently requires the complete target LM head")
    unary = F.linear(hidden, head.to(hidden.dtype)).float()
    anchor_tokens = ids.gather(1, anchors).reshape(-1)
    predecessors = torch.cat((anchor_tokens[:, None], proposal[:, :-1]), dim=-1)
    if wrapper.draft_model.markov_head is None:
        raise ValueError("SpecOPD DSpark requires a conditional Markov head")
    logits = wrapper.draft_model.apply_markov_logits(
        unary,
        prev_token_ids=predecessors,
        draft_hidden=hidden,
    )
    with torch.no_grad():
        teacher = F.linear(
            batch["target_last_hidden_states"].to(head.dtype), head
        ).float()
    distribution_weight, ce_weight = batch.get("loss_weights", (1.0, 0.0))
    return specopd_distribution_loss(
        logits,
        teacher,
        valid,
        position_weights=batch.get("position_weights"),
        distribution_loss_impl=getattr(wrapper, "distribution_loss_impl", "auto"),
        proposal_temperature=batch["proposal_temperature"],
        target_temperature=batch["target_temperature"],
        use_l1=batch.get("objective") == "l1",
        distribution_weight=distribution_weight,
        ce_weight=ce_weight,
    )


def compute_specopd_backend_loss(backend, model, batch):
    if backend.target_lm_head is None:
        raise RuntimeError("SpecOPD target head must be synchronized before training")
    if batch.get("objective") in {"l1", "tv"}:
        drafter = backend.config.rollout.drafter
        batch = {
            **batch,
            "loss_weights": specopd_loss_weights(
                drafter.training, drafter.speculative_algorithm
            ),
        }
    numerator, count, diagnostics = model(
        input_ids=batch["input_ids"],
        hidden_states_list=[],
        loss_mask=batch["proposal_mask"],
        lm_head_weight=backend.target_lm_head.fc.weight,
        specopd_batch=batch,
    )
    zero = numerator.new_zeros(())
    return {
        "total_local_vloss": zero,
        "total_local_ploss": numerator,
        "local_num_tokens": count.to(numerator.dtype),
        "v_weight": 0.0,
        "p_weight": 1.0,
        "accuracy": diagnostics["overlap"].masked_fill(~batch["proposal_mask"], 0).sum()
        / batch["proposal_mask"].sum().clamp_min(1),
        "diagnostics": {},
    }


def build_specopd_training_batch(request, scores, device, config, feature_indices):
    if scores.sample_id != request.sample_id or len(scores.branches) != len(
        request.branches
    ):
        raise ValueError("SpecOPD scoring/request identity mismatch")
    if not request.branches:
        raise ValueError("empty trajectory has no drafter training batch")
    width = max(len(branch.token_ids) for branch in request.branches)
    proposal = torch.zeros(
        (len(request.branches), width), dtype=torch.long, device=device
    )
    mask = torch.zeros_like(proposal, dtype=torch.bool)
    targets = []
    for row, (branch, target) in enumerate(
        zip(request.branches, scores.branches, strict=True)
    ):
        if branch.branch_id != target.branch_id:
            raise ValueError("SpecOPD branch outputs are reordered")
        length = len(branch.token_ids)
        proposal[row, :length] = torch.tensor(branch.token_ids, device=device)
        mask[row, :length] = True
        hidden = target.target_hidden.to(device)
        targets.append(F.pad(hidden, (0, 0, 0, width - length)))
    if not feature_indices or any(
        i not in scores.trunk_features for i in feature_indices
    ):
        raise ValueError("SpecOPD is missing the drafter's target auxiliary layers")
    position_weights = None
    if config.anchor_count:
        position_weights = torch.ones_like(proposal, dtype=torch.float32)
        for row, branch in enumerate(request.branches):
            if branch.source == "trunk":
                if config.random_position_decay_gamma > 0:
                    position_weights[row] = torch.exp(
                        -torch.arange(width, device=device).float()
                        / config.random_position_decay_gamma
                    )
            else:
                if branch.accepted_count < 0:
                    raise ValueError(
                        "mixed SpecOPD proposal is missing acceptance diagnostics"
                    )
                verified = branch.verified_count
                if verified is None:
                    verified = len(branch.token_ids)  # Legacy proposal records.
                if branch.accepted_count < verified:
                    position_weights[row, branch.accepted_count] = (
                        config.first_reject_weight
                    )
    context_rows = max(b.anchor for b in request.branches) + 1
    if any(scores.trunk_features[i].shape[0] < context_rows for i in feature_indices):
        raise ValueError(
            "SpecOPD auxiliary features do not cover the last branch anchor"
        )
    result = {
        "input_ids": torch.tensor(
            [request.trunk_ids[:context_rows]], dtype=torch.long, device=device
        ),
        "hidden_states": torch.cat(
            [
                scores.trunk_features[i][:context_rows].to(device)
                for i in feature_indices
            ],
            -1,
        ).unsqueeze(0),
        "anchor_positions": torch.tensor(
            [[b.anchor for b in request.branches]], device=device
        ),
        "proposal_ids": proposal,
        "proposal_mask": mask,
        "target_last_hidden_states": torch.stack(targets),
        "_speco_sample_count": len(request.branches),
        "_speco_position_count": sum(len(b.token_ids) for b in request.branches),
        "_specopd_request_id": request.sample_id,
        "_specopd_proposal_anchors": sum(
            b.source == "proposal" for b in request.branches
        ),
        "_speco_sample_ids": tuple(b.branch_id for b in request.branches),
    }
    if position_weights is not None:
        result["position_weights"] = position_weights
    return result


def plan_specopd_training_groups(selected, dp_size, max_trajectories, token_budget):
    """Use global slot maxima so every DP/SP rank chooses identical groups."""
    lengths = [
        max(b["input_ids"].shape[1] for b in selected[i : i + dp_size])
        for i in range(0, len(selected), dp_size)
    ]
    sizes, current, longest = [], 0, 0
    for length in lengths:
        if length > token_budget:
            raise ValueError(
                f"SpecOPD trajectory context length {length} exceeds "
                f"training_microbatch_token_budget={token_budget}"
            )
        if current and (
            current == max_trajectories
            or (current + 1) * max(longest, length) > token_budget
        ):
            sizes.append(current)
            current, longest = 0, 0
        current += 1
        longest = max(longest, length)
    if current:
        sizes.append(current)
    return sizes


def merge_specopd_training_batches(batches):
    """Pad independent trajectory batch rows; branches never share context."""
    if len(batches) == 1:
        return batches[0]
    first = batches[0]
    length = max(b["input_ids"].shape[1] for b in batches)
    anchors = max(b["anchor_positions"].shape[1] for b in batches)
    width = max(b["proposal_ids"].shape[1] for b in batches)
    result = dict(first)
    if any(("position_weights" in b) != ("position_weights" in first) for b in batches):
        raise ValueError("cannot merge different SpecOPD loss reductions")
    # Pad from the trailing dimensions, preserving each independent trajectory.
    shapes = {
        "input_ids": (length,),
        "hidden_states": (length, None),
        "anchor_positions": (anchors,),
        "proposal_ids": (anchors, width),
        "proposal_mask": (anchors, width),
        "target_last_hidden_states": (anchors, width, None),
    }
    if "position_weights" in first:
        shapes["position_weights"] = (anchors, width)
    for key, shape in shapes.items():
        padded = []
        for batch in batches:
            tensor = batch[key]
            padding = []
            for actual, target in zip(reversed(tensor.shape), reversed(shape)):
                padding.extend((0, 0 if target is None else target - actual))
            padded.append(F.pad(tensor, padding))
        result[key] = torch.cat(padded)
    for key in (
        "_speco_sample_count",
        "_speco_position_count",
        "_specopd_proposal_anchors",
    ):
        if key in first:
            result[key] = sum(b[key] for b in batches)
    result["_speco_sample_ids"] = tuple(
        x for b in batches for x in b["_speco_sample_ids"]
    )
    return result


def prepare_specopd_training_batches(
    selected,
    device,
    *,
    cache_bytes=64 * 1024**2,
    group_sizes=None,
    real_slots=None,
):
    """Merge fixed CPU inputs and cache within the configured device budget."""
    prepared = []
    for index, inputs in enumerate(selected):
        batch = dict(inputs)
        if real_slots is not None and not real_slots[index]:
            batch["proposal_mask"] = torch.zeros_like(batch["proposal_mask"])
            for key in (
                "_speco_sample_count",
                "_speco_position_count",
                "_specopd_proposal_anchors",
            ):
                batch[key] = 0
            batch["_speco_sample_ids"] = ()
        prepared.append(batch)
    sizes = group_sizes if group_sizes is not None else [1] * len(prepared)
    if any(n <= 0 for n in sizes) or sum(sizes) != len(prepared):
        raise ValueError("invalid SpecOPD training microbatch partition")
    batches, used, offset = [], 0, 0
    for length in sizes:
        batch = merge_specopd_training_batches(prepared[offset : offset + length])
        offset += length
        size = sum(
            v.numel() * v.element_size() for v in batch.values() if torch.is_tensor(v)
        )
        if used + size <= cache_bytes:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            used += size
        batches.append(batch)
    return batches, used


def partition_specopd_samples(selected, dp_rank, dp_size):
    """Each DP replica trains distinct trees; SP peers share a tree.

    Zero-loss dummy slots keep the FSDP collective count equal for uneven
    groups. Global token-sum normalization preserves the complete group loss.
    """
    if not selected or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid SpecOPD data-parallel schedule")
    indices = range(
        dp_rank, ((len(selected) + dp_size - 1) // dp_size) * dp_size, dp_size
    )
    slots = [
        (
            selected[i] if i < len(selected) else selected[(i // dp_size) * dp_size],
            i < len(selected),
        )
        for i in indices
    ]
    return [sample for sample, _ in slots], [real for _, real in slots]
