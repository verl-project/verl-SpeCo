"""Flex-attention mask for P-EAGLE parallel-group prediction.

P-EAGLE flattens all COD-sampled depths into one sequence and runs a single
attention forward. The cross-depth visibility is not plain causal: an element
attends to (a) any earlier-position element at depth 0 (committed real context)
and (b) earlier-or-equal depths of its own rollout (the masked multi-token
chain). Documents are isolated so padded rows never cross-attend.

Verbatim port of NeMo AutoModel's ``peagle_attention.create_peagle_mask_mod``
(itself a port of speculators PR #480).
"""

from __future__ import annotations

import torch


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,  # [total_sampled]
    depth: torch.Tensor,  # [total_sampled]
    lengths: torch.Tensor,  # [num_documents]
    total_seq_len: int,
):
    """Build a ``flex_attention`` ``mask_mod`` for P-EAGLE parallel groups."""
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long), lengths
    )
    document_ids = torch.cat(
        [
            document_ids,
            torch.full((total_seq_len - document_ids.shape[0],), -1, device=lengths.device, dtype=torch.long),
        ]
    ).contiguous()

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_anchor_pos = anchor_pos[q_idx]
        kv_anchor_pos = anchor_pos[kv_idx]
        q_depth = depth[q_idx]
        kv_depth = depth[kv_idx]

        same_document = document_ids[q_anchor_pos] == document_ids[kv_anchor_pos]
        is_not_padding = document_ids[q_anchor_pos] != -1
        same_rollout = q_anchor_pos == kv_anchor_pos
        kv_depth0 = kv_depth == 0
        in_depth_order = q_depth >= kv_depth
        is_anchor_causal = q_anchor_pos >= kv_anchor_pos

        return is_not_padding & same_document & ((kv_depth0 & is_anchor_causal) | (same_rollout & in_depth_order))

    return peagle_mask_mod
