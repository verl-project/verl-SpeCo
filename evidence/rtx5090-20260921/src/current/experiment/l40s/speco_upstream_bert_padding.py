# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch replacements for the ``flash_attn.bert_padding`` helpers verl imports.

The GPU host has no ``flash_attn`` (RTX 5090, sm_120). verl imports four helpers from
``flash_attn.bert_padding``; these are plain tensor operations with no kernel dependency.
This module keeps the upstream semantics, including return arity:

    index_first_axis(x, indices) -> Tensor
    unpad_input(hidden_states, attention_mask, _zu=None, use_actual_seqlen=False)
        -> (x_unpad, indices, cu_seqlens, max_seqlen_in_batch)
    pad_input(hidden_states, indices, batch, seqlen) -> Tensor
    rearrange -> einops.rearrange
"""

from einops import rearrange  # noqa: F401  (re-exported for verl)

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]


def index_first_axis(x, indices):
    """Gather rows of ``x`` (already flattened over batch*seqlen) at ``indices``."""
    if indices.numel() == 0:
        return x.new_empty((0, *x.shape[1:]))
    return x[indices]


def _seq_lens_from_mask(attention_mask):
    import torch

    mask = attention_mask.bool()
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)
    return seqlens_in_batch, int(mask.shape[0]), int(mask.shape[1])


def unpad_input(hidden_states, attention_mask, _zu=None, use_actual_seqlen=False):
    """Drop padding positions, matching ``flash_attn.bert_padding.unpad_input``."""
    import torch

    seqlens_in_batch, batch, seqlen = _seq_lens_from_mask(attention_mask)
    if int(seqlens_in_batch.sum()) != int(batch * seqlen):
        # Slow path: build the flat keep-index list for the ragged batch.
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    else:
        indices = None

    if indices is None:
        x_unpad = hidden_states.reshape(-1, *hidden_states.shape[2:])
        cu_seqlens = torch.arange(
            0, (batch + 1) * seqlen, step=seqlen, dtype=torch.int32, device=hidden_states.device
        )
    else:
        x_unpad = index_first_axis(hidden_states.reshape(-1, *hidden_states.shape[2:]), indices)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
            ]
        )
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if batch else 0
    return x_unpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """Scatter unpadded rows back into a ``(batch, seqlen, ...)`` padded tensor."""
    import torch

    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    if indices is None:
        output = hidden_states
    else:
        output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
