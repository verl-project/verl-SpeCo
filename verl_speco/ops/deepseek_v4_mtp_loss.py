# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ported AIR Eagle KL; shared vocabulary normalization stays in SpeCo."""

import torch
import torch.distributed as dist

_LOSS_DIAGNOSTICS = {}


def _increment_loss_diagnostic(name: str, amount: int = 1) -> None:
    _LOSS_DIAGNOSTICS[name] = _LOSS_DIAGNOSTICS.get(name, 0) + int(amount)


def get_loss_diagnostics() -> dict[str, int]:
    return dict(_LOSS_DIAGNOSTICS)


def _distributed_world_size(process_group) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=process_group)


def _all_reduce_in_place(value, process_group, op=dist.ReduceOp.SUM):
    if _distributed_world_size(process_group) > 1:
        dist.all_reduce(value, op=op, group=process_group)
    return value


class _ChunkedVocabParallelEagleLoss(torch.autograd.Function):
    """Exact TP-sharded Eagle KL with bounded FP32 token workspaces."""

    @staticmethod
    def forward(
        ctx,
        student_local_logits,
        teacher_local_probs,
        generation_mask,
        step_idx,
        process_group,
        token_chunk_size,
    ):
        if teacher_local_probs.requires_grad:
            raise ValueError(
                "Chunked Eagle loss requires detached teacher probabilities"
            )
        if student_local_logits.shape != teacher_local_probs.shape:
            raise ValueError(
                "Student and teacher local vocab shards must have identical shapes: "
                f"student={tuple(student_local_logits.shape)}, "
                f"teacher={tuple(teacher_local_probs.shape)}"
            )

        vocab_size = int(student_local_logits.shape[-1])
        student = student_local_logits.reshape(-1, vocab_size)
        teacher = teacher_local_probs.reshape(-1, vocab_size)
        mask = generation_mask.reshape(-1).float()
        if student.shape[0] != mask.numel():
            raise ValueError(
                "generation_mask must contain one value per token: "
                f"logits={tuple(student_local_logits.shape)}, "
                f"mask={tuple(generation_mask.shape)}"
            )

        requested_chunk_size = int(token_chunk_size)
        chunk_size = (
            min(requested_chunk_size, max(1, int(student.shape[0])))
            if requested_chunk_size > 0
            else max(1, int(student.shape[0]))
        )
        denominator = mask.sum() + 1.0e-6
        numerator = torch.zeros((), dtype=torch.float32, device=student.device)
        for start in range(0, int(student.shape[0]), chunk_size):
            end = min(start + chunk_size, int(student.shape[0]))
            logits = student[start:end].float()
            probabilities = teacher[start:end].float().clone()
            global_max = logits.amax(dim=-1)
            _all_reduce_in_place(global_max, process_group, dist.ReduceOp.MAX)

            reduced = torch.stack(
                (
                    torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1),
                    torch.xlogy(probabilities, probabilities).sum(dim=-1),
                    probabilities.sum(dim=-1),
                    (probabilities * logits).sum(dim=-1),
                ),
                dim=-1,
            )
            _all_reduce_in_place(reduced, process_group)
            global_kl = (
                reduced[:, 1]
                - reduced[:, 3]
                + reduced[:, 2] * (global_max + torch.log(reduced[:, 0]))
            ).to(student_local_logits.dtype)
            numerator.add_((global_kl * mask[start:end]).sum())

        ctx.save_for_backward(
            student_local_logits,
            teacher_local_probs,
            generation_mask,
        )
        ctx.process_group = process_group
        ctx.token_chunk_size = chunk_size
        ctx.loss_weight = 0.8 ** int(step_idx)
        _increment_loss_diagnostic("vocab_parallel_loss_chunked_forward_calls")
        _increment_loss_diagnostic(
            "vocab_parallel_loss_forward_token_chunks",
            (int(student.shape[0]) + chunk_size - 1) // chunk_size,
        )
        return numerator.div(denominator).mul(ctx.loss_weight)

    @staticmethod
    def backward(ctx, grad_output):
        student_local_logits, teacher_local_probs, generation_mask = ctx.saved_tensors
        vocab_size = int(student_local_logits.shape[-1])
        student = student_local_logits.reshape(-1, vocab_size)
        teacher = teacher_local_probs.reshape(-1, vocab_size)
        mask = generation_mask.reshape(-1).float()
        denominator = mask.sum() + 1.0e-6
        loss_scale = grad_output.float() * ctx.loss_weight / denominator
        grad_student = torch.empty_like(student)
        token_chunks = 0

        for start in range(0, int(student.shape[0]), ctx.token_chunk_size):
            end = min(start + ctx.token_chunk_size, int(student.shape[0]))
            logits = student[start:end].float()
            probabilities = teacher[start:end].float().clone()
            token_scale = (
                (mask[start:end] * loss_scale).to(student_local_logits.dtype).float()
            )
            probabilities.mul_(token_scale.unsqueeze(-1))
            global_max = logits.amax(dim=-1)
            _all_reduce_in_place(
                global_max,
                ctx.process_group,
                dist.ReduceOp.MAX,
            )

            local_softmax = torch.exp(logits - global_max.unsqueeze(-1))
            reduced = torch.stack(
                (
                    local_softmax.sum(dim=-1),
                    probabilities.sum(dim=-1),
                ),
                dim=-1,
            )
            _all_reduce_in_place(reduced, ctx.process_group)
            local_softmax.div_(reduced[:, 0].unsqueeze(-1))
            local_softmax.mul_(reduced[:, 1].unsqueeze(-1))
            local_softmax.sub_(probabilities)
            grad_student[start:end].copy_(local_softmax)
            token_chunks += 1

        _increment_loss_diagnostic("vocab_parallel_loss_chunked_backward_calls")
        _increment_loss_diagnostic(
            "vocab_parallel_loss_backward_token_chunks",
            token_chunks,
        )
        _increment_loss_diagnostic(
            "vocab_parallel_loss_explicit_backward_calls",
        )
        _increment_loss_diagnostic(
            "vocab_parallel_loss_explicit_backward_token_chunks",
            token_chunks,
        )
        return (
            grad_student.reshape_as(student_local_logits),
            None,
            None,
            None,
            None,
            None,
        )


def vocab_parallel_eagle_loss(
    student_local_logits,
    teacher_local_probs,
    generation_mask,
    step_idx,
    process_group=None,
    token_chunk_size=0,
):
    """Exact Eagle KL over vocabulary shards without gathering full logits."""
    return _ChunkedVocabParallelEagleLoss.apply(
        student_local_logits,
        teacher_local_probs,
        generation_mask,
        step_idx,
        process_group,
        token_chunk_size,
    )


def select_valid_loss_tokens(
    hidden_states,
    p_target,
    ground_truth,
    generation_mask,
    current_s=None,
):
    """Select tokens after the sequence model, before full-vocabulary logits."""
    valid_tokens = generation_mask.to(dtype=torch.bool)
    selected_ground_truth = (
        ground_truth[valid_tokens] if ground_truth is not None else None
    )
    selected_current_s = current_s[valid_tokens] if current_s is not None else None
    selected_mask = torch.ones_like(generation_mask[valid_tokens])
    return (
        hidden_states[valid_tokens],
        p_target[valid_tokens],
        selected_ground_truth,
        selected_mask,
        selected_current_s,
    )
