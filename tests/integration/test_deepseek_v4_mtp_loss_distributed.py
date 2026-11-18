# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Two-process Gloo test of real vocabulary-sharded MTP autograd.

SPECO_RUN_DISTRIBUTED_TESTS=1 torchrun --standalone --nproc-per-node=2 \
  -m pytest -q tests/integration/test_deepseek_v4_mtp_loss_distributed.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl_speco.ops.deepseek_v4_mtp_loss import vocab_parallel_eagle_loss

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECO_RUN_DISTRIBUTED_TESTS") != "1",
    reason="opt-in two-rank Gloo test",
)


def test_uneven_vocab_shards_match_independent_dense_kl_and_gradients():
    assert int(os.environ.get("WORLD_SIZE", "0")) == 2
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(34)
        dense = torch.randn(2, 5, 13, requires_grad=True)
        target = torch.randn_like(dense).softmax(-1).detach()
        mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
        expected = (
            (
                F.kl_div(dense.log_softmax(-1), target, reduction="none").sum(-1) * mask
            ).sum()
            / (mask.sum() + 1e-6)
            * 0.8
        )
        expected.backward()
        rank = dist.get_rank()
        local = dense.detach().tensor_split(2, dim=-1)[rank].clone().requires_grad_()
        teacher = target.tensor_split(2, dim=-1)[rank].clone()
        original = teacher.clone()
        actual = vocab_parallel_eagle_loss(
            local, teacher, mask, 1, process_group=dist.group.WORLD, token_chunk_size=3
        )
        actual.backward()
        torch.testing.assert_close(actual, expected.detach(), atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(
            local.grad, dense.grad.tensor_split(2, dim=-1)[rank], atol=2e-8, rtol=2e-6
        )
        assert torch.equal(teacher, original)
    finally:
        dist.destroy_process_group()
