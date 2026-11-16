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
"""Fixed-anchor block proposal forward and E2E-TV, without reference-token CE."""

import torch
import torch.nn.functional as F

from verl_speco.backends.specopd_loss import e2e_tv_loss


def proposal_loss(wrapper, batch, lm_head_weight):
    from verl_speco.backends.dflash_trainer_backend import (
        build_dflash_attention_masks,
        _resolve_sliding_windows,
    )

    ids = batch["input_ids"]
    anchors = batch["anchor_positions"]
    valid = batch["proposal_mask"]
    proposal = batch["proposal_ids"]
    n = anchors.shape[1]
    if (
        ids.shape[0] != 1
        or anchors.shape[0] != 1
        or proposal.shape != valid.shape
        or proposal.shape[0] != n
    ):
        raise ValueError(
            "SpecOPD microbatches contain one shared trunk and all its branches"
        )
    spark = hasattr(wrapper.draft_model, "markov_head")
    start = 0 if spark else 1
    if wrapper.block_size != batch["proposal_block_width"] + start:
        raise ValueError(
            "drafter mask-block width differs from rollout; probabilities cannot be replayed"
        )
    width = proposal.shape[1]
    if width + start > wrapper.block_size:
        raise ValueError("SpecOPD proposal width exceeds the drafter block capacity")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    base = batch["hidden_states"]
    layers = wrapper.draft_model.num_context_layers
    if base.shape[-1] % layers:
        raise ValueError(
            "SpecOPD concatenated auxiliary feature dimensions are invalid"
        )
    context = wrapper.draft_model.extract_context_feature(
        list(base.chunk(layers, dim=-1))
    )
    context_pos, draft_pos = wrapper._create_position_ids(anchors, ids.shape[1])
    mask, dense = build_dflash_attention_masks(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=ids.shape[1],
        block_size=wrapper.block_size,
        device=ids.device,
        windows=_resolve_sliding_windows(wrapper.draft_model.config),
    )
    hidden = wrapper.draft_model(
        draft_input_ids=None,
        context_feature=context,
        draft_position_ids=draft_pos,
        context_position_ids=context_pos,
        block_mask=mask,
        dense_attention_mask=dense,
        noise_embedding=wrapper._create_noise_embed(ids, anchors, keep),
    ).reshape(n, wrapper.block_size, -1)[:, start : start + width]
    head = lm_head_weight
    if head.shape[0] != wrapper.draft_model.config.vocab_size:
        raise ValueError("SpecOPD currently requires the complete target LM head")
    unary = F.linear(hidden, head.to(hidden.dtype)).float()
    anchor_tokens = ids[0, anchors[0]]
    predecessors = torch.cat((anchor_tokens[:, None], proposal[:, :-1]), dim=-1)
    candidate_ids = None
    if hasattr(wrapper.draft_model, "candidate_selector"):
        selector = wrapper.draft_model.candidate_selector
        _, candidate_ids = unary.detach().topk(
            min(selector.top_k, unary.shape[-1]), dim=-1, sorted=False
        )
        supported = (candidate_ids == proposal.unsqueeze(-1)).any(-1)
        if not supported[valid].all():
            raise ValueError(
                "sampled proposal is outside reconstructed DFlash2 candidate support"
            )
        selected_unary = unary.gather(-1, candidate_ids)
        logits = selector.pair_scores(
            hidden.reshape(-1, hidden.shape[-1]),
            selected_unary.reshape(-1, candidate_ids.shape[-1]),
            candidate_ids.reshape(-1, candidate_ids.shape[-1]),
            predecessors.reshape(-1),
        ).reshape_as(selected_unary)
    elif spark:
        if wrapper.draft_model.markov_head is None:
            raise ValueError("SpecOPD DSpark requires a conditional Markov head")
        logits = wrapper.draft_model.apply_markov_logits(
            unary,
            prev_token_ids=predecessors,
            draft_hidden=hidden,
        )
    else:
        logits = unary
    with torch.no_grad():
        teacher = F.linear(
            batch["target_last_hidden_states"].to(head.dtype), head
        ).float()
    numerator, count, diagnostics = e2e_tv_loss(
        logits,
        teacher,
        valid,
        candidate_ids=candidate_ids,
        proposal_temperature=batch["proposal_temperature"],
        target_temperature=batch["target_temperature"],
    )
    return numerator, count, diagnostics


def compute_specopd_backend_loss(backend, model, batch):
    if backend.target_lm_head is None:
        raise RuntimeError("SpecOPD target head must be synchronized before training")
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
        "accuracy": diagnostics["overlap"][batch["proposal_mask"]].mean(),
        "diagnostics": {},
    }


def build_specopd_training_batch(request, scores, device, config, feature_indices):
    from verl_speco.transport.specopd_protocol import token_digest

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
    return {
        "input_ids": torch.tensor([request.trunk_ids], dtype=torch.long, device=device),
        "hidden_states": torch.cat(
            [scores.trunk_features[i].to(device) for i in feature_indices], -1
        ).unsqueeze(0),
        "anchor_positions": torch.tensor(
            [[b.anchor for b in request.branches]], device=device
        ),
        "proposal_ids": proposal,
        "proposal_mask": mask,
        "target_last_hidden_states": torch.stack(targets),
        "proposal_temperature": config.proposal_temperature,
        "proposal_block_width": config.proposal_width,
        "target_temperature": config.target_temperature,
        "specopd": True,
        "_speco_sample_count": len(request.branches),
        "_speco_sample_ids": tuple(b.branch_id for b in request.branches),
        "_specopd_trunk_digest": token_digest(request.trunk_ids),
    }
