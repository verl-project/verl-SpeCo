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
# Adapted original numerical validation; no external source imports.
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from verl_speco.models.deepseek_v4.modeling_dspark import (
    DeepseekV4DSparkServingWrapper,
    DSparkBlockTargets,
    compute_dspark_loss,
    compute_vocab_parallel_dspark_loss,
    distributed_vocab_argmax,
)
from verl_speco.ops.deepseek_v4_loss import all_reduce_gradient


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loss_targets() -> DSparkBlockTargets:
    target_ids = torch.tensor([[[1, 7, 10], [6, 2, 9]]])
    return DSparkBlockTargets(
        anchor_positions=torch.tensor([[0, 3]]),
        block_keep_mask=torch.ones((1, 2), dtype=torch.bool),
        label_indices=torch.tensor([[[1, 2, 3], [4, 5, 6]]]),
        target_ids=target_ids,
        previous_token_ids=torch.zeros_like(target_ids),
        eval_mask=torch.ones_like(target_ids, dtype=torch.bool),
    )


def _distributed_worker(
    rank,
    world_size,
    port,
    draft_logits,
    target_logits,
    shared_feature,
    projection_weight,
    output_dir,
):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    shard_size = 6
    start = rank * shard_size
    end = min(start + shard_size, draft_logits.shape[-1])
    local_draft = draft_logits[..., start:end].clone().requires_grad_(True)
    local_target = target_logits[..., start:end].clone()
    confidence_logits = torch.randn(
        draft_logits.shape[:-1],
        generator=torch.Generator().manual_seed(202),
        requires_grad=True,
    )
    loss = compute_vocab_parallel_dspark_loss(
        local_draft,
        _loss_targets(),
        target_local_logits=local_target,
        vocab_start_index=start,
        process_group=dist.group.WORLD,
        loss_decay_gamma=4.0,
        distribution_kl_weight=0.35,
        confidence_logits=confidence_logits,
        confidence_weight=0.2,
        compute_distribution_metrics=True,
    )
    loss.total.backward()
    replicated_feature = torch.tensor(
        [1.0, 2.0],
        requires_grad=True,
    )
    reduced_feature = all_reduce_gradient(
        replicated_feature,
        dist.group.WORLD,
    )
    ((rank + 1) * reduced_feature.sum()).backward()
    local_shared_feature = shared_feature.clone().requires_grad_(True)
    local_projection_weight = projection_weight[start:end].clone().requires_grad_(True)
    local_projected_logits = F.linear(
        all_reduce_gradient(local_shared_feature, dist.group.WORLD),
        local_projection_weight,
    )
    projected_loss = compute_vocab_parallel_dspark_loss(
        local_projected_logits,
        _loss_targets(),
        target_local_logits=local_target,
        vocab_start_index=start,
        process_group=dist.group.WORLD,
        loss_decay_gamma=4.0,
        distribution_kl_weight=0.35,
    )
    projected_loss.total.backward()
    greedy_ids = distributed_vocab_argmax(
        local_target,
        vocab_start_index=start,
        process_group=dist.group.WORLD,
    )
    torch.save(
        {
            "total": loss.total.detach(),
            "ce": loss.cross_entropy.detach(),
            "l1": loss.distribution_l1.detach(),
            "kl": loss.distribution_kl.detach(),
            "distribution_position_counts": (
                loss.distribution_position_counts.detach()
            ),
            "distribution_l1_position_sums": (
                loss.distribution_l1_position_sums.detach()
            ),
            "top1_correct_counts": loss.top1_correct_counts.detach(),
            "gradient": local_draft.grad,
            "confidence_gradient": confidence_logits.grad,
            "replicated_feature_gradient": replicated_feature.grad,
            "shared_feature_gradient": local_shared_feature.grad,
            "projection_weight_gradient": local_projection_weight.grad,
            "greedy_ids": greedy_ids,
        },
        Path(output_dir) / f"rank_{rank}.pt",
    )
    dist.destroy_process_group()


class TestDSparkVocabParallelLoss(unittest.TestCase):
    def test_markov_logits_processor_falls_back_to_serving_model(self):
        processor = object()
        resolved = DeepseekV4DSparkServingWrapper._resolve_markov_logits_processor(
            SimpleNamespace(),
            SimpleNamespace(logits_processor=processor),
        )

        self.assertIs(resolved, processor)

    def test_zero_kl_weight_preserves_existing_dense_objective(self):
        generator = torch.Generator().manual_seed(77)
        draft = torch.randn(1, 2, 3, 11, generator=generator)
        target = torch.randn(1, 2, 3, 11, generator=generator)
        loss = compute_dspark_loss(
            draft,
            _loss_targets(),
            target_logits=target,
        )

        weights = _loss_targets().eval_mask.to(torch.float32)
        expected_l1 = (
            draft.float().softmax(dim=-1) - target.float().softmax(dim=-1)
        ).abs().sum(dim=-1).mul(weights).sum() / weights.sum()
        torch.testing.assert_close(loss.distribution_l1, expected_l1)
        torch.testing.assert_close(
            loss.distribution_kl,
            torch.zeros_like(loss.distribution_kl),
        )
        torch.testing.assert_close(
            loss.total,
            0.1 * loss.cross_entropy + 0.9 * loss.distribution_l1,
        )

    def test_two_rank_loss_and_gradient_match_dense(self):
        generator = torch.Generator().manual_seed(101)
        draft = torch.randn(1, 2, 3, 11, generator=generator, requires_grad=True)
        target = torch.randn(1, 2, 3, 11, generator=generator)
        confidence = torch.randn(
            draft.shape[:-1],
            generator=torch.Generator().manual_seed(202),
            requires_grad=True,
        )
        shared_feature = torch.randn(
            *draft.shape[:-1],
            4,
            generator=torch.Generator().manual_seed(303),
            requires_grad=True,
        )
        projection_weight = torch.randn(
            draft.shape[-1],
            4,
            generator=torch.Generator().manual_seed(404),
            requires_grad=True,
        )
        dense = compute_dspark_loss(
            draft,
            _loss_targets(),
            target_logits=target,
            loss_decay_gamma=4.0,
            confidence_logits=confidence,
            confidence_weight=0.2,
            distribution_kl_weight=0.35,
            compute_distribution_metrics=True,
        )
        dense.total.backward()
        dense_projected = compute_dspark_loss(
            F.linear(shared_feature, projection_weight),
            _loss_targets(),
            target_logits=target,
            loss_decay_gamma=4.0,
            distribution_kl_weight=0.35,
        )
        dense_projected.total.backward()

        with tempfile.TemporaryDirectory() as output_dir:
            mp.spawn(
                _distributed_worker,
                args=(
                    2,
                    _find_free_port(),
                    draft.detach(),
                    target,
                    shared_feature.detach(),
                    projection_weight.detach(),
                    output_dir,
                ),
                nprocs=2,
                join=True,
            )
            outputs = [
                torch.load(
                    Path(output_dir) / f"rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for rank in range(2)
            ]

        for output in outputs:
            torch.testing.assert_close(output["total"], dense.total)
            torch.testing.assert_close(output["ce"], dense.cross_entropy)
            torch.testing.assert_close(output["l1"], dense.distribution_l1)
            torch.testing.assert_close(output["kl"], dense.distribution_kl)
            torch.testing.assert_close(
                output["distribution_position_counts"],
                dense.distribution_position_counts,
            )
            torch.testing.assert_close(
                output["distribution_l1_position_sums"],
                dense.distribution_l1_position_sums,
            )
            torch.testing.assert_close(
                output["top1_correct_counts"],
                dense.top1_correct_counts,
            )
            torch.testing.assert_close(
                output["confidence_gradient"],
                confidence.grad,
            )
            torch.testing.assert_close(
                output["replicated_feature_gradient"],
                torch.full((2,), 3.0),
            )
            torch.testing.assert_close(
                output["shared_feature_gradient"],
                shared_feature.grad,
            )
            torch.testing.assert_close(
                output["greedy_ids"],
                target.argmax(dim=-1),
            )
        distributed_gradient = torch.cat(
            [output["gradient"] for output in outputs],
            dim=-1,
        )
        torch.testing.assert_close(distributed_gradient, draft.grad)
        distributed_weight_gradient = torch.cat(
            [output["projection_weight_gradient"] for output in outputs],
            dim=0,
        )
        torch.testing.assert_close(
            distributed_weight_gradient,
            projection_weight.grad,
        )


if __name__ == "__main__":
    unittest.main()
