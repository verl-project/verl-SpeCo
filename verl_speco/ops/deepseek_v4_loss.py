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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/losses.py.
import torch
import torch.distributed as dist


class _ReplicatedAllReduceSum(torch.autograd.Function):
    """Forward SUM with identity backward for a TP-replicated logical loss."""

    @staticmethod
    def forward(ctx, value, process_group):
        if not dist.is_initialized() or dist.get_world_size(group=process_group) == 1:
            return value.clone()
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=process_group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _DifferentiableAllReduceSum(torch.autograd.Function):
    """SUM in both passes for expressions coupled across TP vocab shards."""

    @staticmethod
    def forward(ctx, value, process_group):
        ctx.process_group = process_group
        if not dist.is_initialized() or dist.get_world_size(group=process_group) == 1:
            return value.clone()
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=process_group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if (
            not dist.is_initialized()
            or dist.get_world_size(group=ctx.process_group) == 1
        ):
            return grad_output, None
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(
            grad_input,
            op=dist.ReduceOp.SUM,
            group=ctx.process_group,
        )
        return grad_input, None


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = float(scale)
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class _AllReduceGradient(torch.autograd.Function):
    """Identity forward with a SUM of replicated-feature gradients over TP."""

    @staticmethod
    def forward(ctx, value, process_group):
        ctx.process_group = process_group
        return value

    @staticmethod
    def backward(ctx, grad_output):
        if (
            not dist.is_initialized()
            or dist.get_world_size(group=ctx.process_group) == 1
        ):
            return grad_output, None
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(
            grad_input,
            op=dist.ReduceOp.SUM,
            group=ctx.process_group,
        )
        return grad_input, None


def differentiable_all_reduce_sum(value, process_group=None):
    return _DifferentiableAllReduceSum.apply(value, process_group)


def scale_gradient(value, scale):
    return _ScaleGradient.apply(value, scale)


def all_reduce_gradient(value, process_group=None):
    return _AllReduceGradient.apply(value, process_group)


def differentiable_vocab_parallel_log_probs(local_logits, process_group=None):
    """Compute global log-probs with exact cross-shard autograd coupling."""

    logits = local_logits.float()
    global_max = _global_vocab_max(logits, process_group)
    local_exp_sum = torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1)
    global_exp_sum = differentiable_all_reduce_sum(
        local_exp_sum,
        process_group,
    )
    return logits - (global_max + torch.log(global_exp_sum)).unsqueeze(-1)


def _global_vocab_max(local_logits, process_group):
    local_max = local_logits.detach().amax(dim=-1)
    if dist.is_initialized() and dist.get_world_size(group=process_group) > 1:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=process_group)
    return local_max


def _global_vocab_logsumexp(local_logits, process_group):
    global_max = _global_vocab_max(local_logits, process_group)
    local_exp_sum = torch.exp(local_logits - global_max.unsqueeze(-1)).sum(dim=-1)
    global_exp_sum = _ReplicatedAllReduceSum.apply(local_exp_sum, process_group)
    return global_max + torch.log(global_exp_sum)


def vocab_parallel_log_probs(local_logits, process_group=None):
    """Compute exact log probabilities while retaining only a local vocab shard."""
    logits = local_logits.float()
    global_logsumexp = _global_vocab_logsumexp(logits, process_group)
    return logits - global_logsumexp.unsqueeze(-1)
