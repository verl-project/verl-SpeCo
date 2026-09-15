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
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from verl_speco.models.dflash import DFlashDraftModel

from .configuration_dflash2 import DFlash2Config


def grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Depthwise causal convolution with a content-adaptive per-group correction.

    ``base`` is a static per-channel kernel of shape ``[kernel_size, hidden]``;
    ``dynamic`` carries one extra coefficient per (position, tap, group) so every
    ``group_size`` channels share a correction. Faithful to the upstream
    ``_grouped_dynamic_convolve`` in z-lab/dflash.

    Args:
        hidden: ``[batch, length, hidden]`` activations. ``length`` must already
            be block-local, since tap ``offset`` reads position ``t - offset``.
        dynamic: ``[batch, length, kernel_size, groups]`` correction coefficients.
        base: ``[kernel_size, hidden]`` static kernel.
        group_size: Number of channels sharing one dynamic coefficient.

    Returns:
        torch.Tensor: Same shape as ``hidden``.
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        if offset == 0:
            values = blocks
        else:
            values = F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset].to(hidden.dtype), values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    """Two-tap dynamic depthwise convolution wrapped around a transformer sublayer.

    ``prepare`` runs before the sublayer and also emits the dynamic kernel that
    ``finish`` applies after it, so one projection feeds both taps.

    Unlike the upstream inference implementation, which only ever sees a single
    draft block, the training path packs ``n_blocks`` blocks into one flat
    sequence. The convolution is causal along the sequence axis, so it is applied
    block-locally here; otherwise tap 1 at a block's first position would read the
    last position of the previous, unrelated block.
    """

    def __init__(
        self, hidden_size: int, kernel_size: int, group_size: int, block_size: int
    ):
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        self.block_size = int(block_size)
        groups = hidden_size // group_size
        # Start as an identity passthrough (tap 0 = 1, later taps = 0) with a
        # zeroed projection, so a freshly built DFlash2 is numerically identical
        # to DFlash and learns the correction from there.
        base_kernel = torch.zeros(2, self.kernel_size, hidden_size)
        base_kernel[:, 0, :] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.kernel_size * groups, bias=False
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _to_block_local(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        batch, length, hidden_size = hidden.shape
        if length % self.block_size != 0:
            raise ValueError(
                f"DFlash2 conv expects the draft sequence length ({length}) to be a "
                f"multiple of block_size ({self.block_size}); the causal taps must not "
                "cross a block boundary."
            )
        n_blocks = length // self.block_size
        return (
            hidden.reshape(batch * n_blocks, self.block_size, hidden_size),
            (batch, length, hidden_size),
        )

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_hidden, shape = self._to_block_local(hidden)
        groups = self.hidden_size // self.group_size
        dynamic = self.kernel_projection(block_hidden).view(
            *block_hidden.shape[:-1], 2, self.kernel_size, groups
        )
        convolved = grouped_dynamic_convolve(
            block_hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
        )
        return convolved.reshape(*shape), dynamic[..., 1, :, :]

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        block_hidden, shape = self._to_block_local(hidden)
        convolved = grouped_dynamic_convolve(
            block_hidden, dynamic, self.base_kernel[1], self.group_size
        )
        return convolved.reshape(*shape)


class CandidateSelector(nn.Module):
    """Scores adjacent (predecessor, candidate) pairs to pick one coherent path.

    For block position ``t``, candidate ``b`` and the token ``a`` chosen at
    ``t - 1``::

        S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>

    where ``U_t(b)`` is the drafter's own logit for ``b``, ``A``/``B`` are the
    predecessor/successor codebooks and ``H`` projects the backbone hidden state.
    """

    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.rank = int(config.selector_rank)
        self.top_k = int(config.selector_top_k)
        self.predecessor_codebook = nn.Embedding(config.vocab_size, self.rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, self.rank)
        self.hidden_projection = nn.Linear(config.hidden_size, self.rank, bias=False)

    def pair_scores(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidates: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate against a known predecessor, all positions at once.

        Training teacher-forces the predecessor (it is the ground-truth previous
        token), so unlike :meth:`select` this needs no sequential loop.

        Args:
            hidden: ``[n, hidden]`` backbone states for the scored positions.
            unary: ``[n, k]`` drafter logits for the candidates.
            candidates: ``[n, k]`` candidate token ids.
            predecessor_ids: ``[n]`` ground-truth previous token per position.

        Returns:
            torch.Tensor: ``[n, k]`` pair scores.
        """
        projected = self.hidden_projection(hidden)
        gated = self.predecessor_codebook(predecessor_ids) * projected
        successor = self.successor_codebook(candidates)
        return unary + torch.einsum("nr,nkr->nk", gated, successor)

    @torch.no_grad()
    def select(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy/sampled path trace through the per-position candidate sets.

        Mirrors the upstream inference-time selector: the predecessor at position
        ``t`` is whatever was chosen at ``t - 1``, seeded by ``anchor_ids``.

        Args:
            hidden: ``[batch, block, hidden]`` backbone states.
            logits: ``[batch, block, vocab]`` drafter logits.
            anchor_ids: ``[batch]`` token preceding the block.
            temperature: 0 for argmax, otherwise softmax sampling.

        Returns:
            tuple: ``(path [batch, block], candidates [batch, block, k])``.
        """
        top_k = min(self.top_k, logits.shape[-1])
        unary, candidates = torch.topk(logits, top_k, dim=-1, sorted=False)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            gated = self.predecessor_codebook(predecessor) * projected[:, position]
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk", gated, self.successor_codebook(candidates[:, position])
            )
            if temperature > 0:
                probs = torch.softmax(scores.float() / temperature, dim=-1)
                index = torch.multinomial(probs, num_samples=1)[:, 0]
            else:
                index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash block drafter plus the DFlash2 dynamic convolutions and selector.

    The convolutions are attached to the existing :class:`DFlashDecoderLayer`
    hooks, which are inert (``None``) for every other DFlash-family drafter, so
    DFlash / Domino / DSpark / JetSpec keep their exact behaviour.
    """

    config_class = DFlash2Config

    def __init__(self, config: DFlash2Config):
        super().__init__(config)
        self.block_size = int(getattr(config, "block_size", 8))
        self.conv_kernel_size = int(getattr(config, "conv_kernel_size", 2))
        self.conv_group_size = int(getattr(config, "conv_group_size", 16))
        for layer in self.layers:
            layer.attention_conv = GroupedDynamicCausalConv(
                config.hidden_size,
                self.conv_kernel_size,
                self.conv_group_size,
                self.block_size,
            )
            layer.mlp_conv = GroupedDynamicCausalConv(
                config.hidden_size,
                self.conv_kernel_size,
                self.conv_group_size,
                self.block_size,
            )
        self.candidate_selector = CandidateSelector(config)


__all__ = [
    "CandidateSelector",
    "DFlash2Config",
    "DFlash2DraftModel",
    "GroupedDynamicCausalConv",
    "grouped_dynamic_convolve",
]
