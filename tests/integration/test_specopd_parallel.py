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
"""Real multi-process Ulysses collectives vs full causal/tree inference."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from verl_speco.backends.specopd_forward import partition_specopd_samples
from verl_speco.backends.specopd_loss import specopd_distribution_loss
from verl_speco.joint_actor import score_actor_joint
from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch
from verl_speco.joint_ulysses import score_ulysses_joint_batch


def _requests():
    return (
        JointScoringRequest("a", (1, 2, 3, 4, 5, 6, 7, 8, 9), 3, (
            DraftBranch("a0", 3, (10, 11, 12)), DraftBranch("a1", 5, (13, 14, 15)),
        )),
        JointScoringRequest("b", (3, 4, 5, 6, 7), 2, (DraftBranch("b0", 1, (8, 9)),)),
    )


def _assert_scores(actual, expected, *, atol=2e-5, rtol=2e-5):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert a.sample_id == b.sample_id
        torch.testing.assert_close(a.response_log_probs, b.response_log_probs, atol=atol, rtol=rtol)
        torch.testing.assert_close(a.response_entropy, b.response_entropy, atol=atol, rtol=rtol)
        for index in a.trunk_features:
            torch.testing.assert_close(a.trunk_features[index], b.trunk_features[index], atol=atol, rtol=rtol)
        for x, y in zip(a.branches, b.branches, strict=True):
            assert x.branch_id == y.branch_id
            for field in ("target_logits", "target_logz", "target_hidden"):
                torch.testing.assert_close(getattr(x, field), getattr(y, field), atol=atol, rtol=rtol)


def _run(rank, size, init_file):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=size)
    try:
        torch.manual_seed(193)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        group = dist.group.WORLD
        requests = _requests()
        options = dict(hidden_state_indices=(0, 1, 2), calculate_entropy=True)
        expected, _ = score_joint_batch(model, requests, sequence_multiple=size, **options)
        fallback, metrics = score_ulysses_joint_batch(model, requests, group, use_preserve_pv=True, **options)
        _assert_scores(fallback, expected)
        assert metrics["sp_size"] == size and metrics["mask_elements"] == 2 * 16**2

        tree = requests[:1]
        dense, _ = score_ulysses_joint_batch(model, tree, group, use_preserve_pv=False, **options)
        with patch.object(torch, "matmul", wraps=torch.matmul) as matmul:
            preserved, metrics = score_ulysses_joint_batch(model, tree, group, use_preserve_pv=True, **options)
        _assert_scores(preserved, dense)
        assert metrics["model_forward_calls"] == 1 and metrics["mask_elements"] == (6 + 4) * 16
        square_values = [call for call in matmul.call_args_list if call.args[0].shape[-2:] == (16, 16)]
        assert len(square_values) == config.num_hidden_layers

        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        engine = SimpleNamespace(
            module=model,
            eval_mode=nullcontext,
            use_remove_padding=True,
            ulysses_sequence_parallel_size=2,
            ulysses_parallel_group=groups[rank // 2],
        )
        worker = SimpleNamespace(_is_actor=True, actor=SimpleNamespace(engine=engine))
        exported, metrics = score_actor_joint(worker, tree, microbatch_trajectories=1, **options)
        if rank == 0:
            _assert_scores(exported, expected[:1])
            assert metrics["model_forward_calls"] == 1
        else:
            assert not exported and metrics["model_forward_calls"] == 0
    finally:
        dist.destroy_process_group()


def test_ulysses_preserves_branch_scores_and_exports_each_tree_once(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("requires CPU Gloo collectives")
    mp.spawn(_run, args=(4, str(tmp_path / "init")), nprocs=4, join=True)


def _check(rank, world, sp, sample_count, init_file, aligned):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=world)
    try:
        torch.manual_seed(321)
        base = torch.nn.Linear(3, 5)
        reference = deepcopy(base)
        model = DistributedDataParallel(base)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        samples = []
        for i in range(sample_count):
            x, teacher = torch.randn(1, 3, 3), torch.randn(1, 3, 5)
            mask = torch.arange(3)[None] < (1 + i % 3)
            samples.append((x, teacher, mask))
        local, real_slots = partition_specopd_samples(samples, rank // sp, world // sp)
        for _ in range(20):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            token_total = 0
            for (x, teacher, mask), real in zip(local, real_slots, strict=True):
                valid = mask if real else torch.zeros_like(mask)
                numerator, count, _ = specopd_distribution_loss(model(x), teacher, valid,
                    position_weights=torch.tensor([[1., 2., 1.]]) if aligned else None)
                counts = count.detach().clone()
                dist.all_reduce(counts)
                token_total += int(counts)
                (numerator * world).backward()
            for parameter in model.parameters():
                parameter.grad.div_(token_total)
            ref_numerator, ref_count = 0, 0
            for x, teacher, mask in samples:
                n, c, _ = specopd_distribution_loss(reference(x), teacher, mask,
                    position_weights=torch.tensor([[1., 2., 1.]]) if aligned else None)
                ref_numerator += n
                ref_count += c
            (ref_numerator / ref_count).backward()
            for actual, expected in zip(model.module.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.module.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('world,sp,samples', [(2, 1, 3), (4, 2, 1)])
def test_dp_partition_preserves_twenty_group_updates(tmp_path, world, sp, samples, aligned):
    if not dist.is_gloo_available():
        pytest.skip('requires CPU Gloo')
    mp.spawn(_check, args=(world, sp, samples, str(tmp_path / 'init'), aligned), nprocs=world, join=True)
