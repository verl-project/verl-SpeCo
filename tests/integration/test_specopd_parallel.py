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
from dataclasses import replace
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


def _run(rank, size, init_file, kv_heads, kind):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=size)
    try:
        torch.manual_seed(193)
        config_cls, model_cls = {
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "llama": (LlamaConfig, LlamaForCausalLM),
        }[kind]
        q_heads = 2 * size
        cfg = config_cls(vocab_size=32, hidden_size=q_heads * 8, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=q_heads, num_key_value_heads=kv_heads, head_dim=8,
                         max_position_embeddings=128, attention_dropout=0.0)
        cfg._attn_implementation = "eager"
        model = model_cls(cfg).eval()
        before = [layer.self_attn.forward.__func__ for layer in model.model.layers]
        options = dict(hidden_state_indices=(0, 1, 2), calculate_entropy=True, return_target_logits=True)
        expected, _ = score_joint_batch(model, _requests(), sequence_multiple=size, **options)
        with patch.object(dist, "all_to_all_single", wraps=dist.all_to_all_single) as collective:
            actual, metrics = score_ulysses_joint_batch(
                model, _requests(), dist.group.WORLD, use_packed_qkv=False, **options,
            )
            assert collective.call_count == 4 * cfg.num_hidden_layers
            collective.reset_mock()
            packed, _ = score_ulysses_joint_batch(
                model, _requests(), dist.group.WORLD, use_packed_qkv=True, **options,
            )
            assert collective.call_count == 2 * cfg.num_hidden_layers
        _assert_scores(packed, actual, atol=0, rtol=0)
        _assert_scores(actual, expected)
        if kind == "qwen3" and size == 2:
            training_options = dict(options, return_target_logits=False, use_packed_qkv=False)
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collective:
                original, _ = score_ulysses_joint_batch(model, _requests(), dist.group.WORLD, **training_options)
                original_reductions = collective.call_count
                collective.reset_mock()
                lean, _ = score_ulysses_joint_batch(
                    model, _requests(), dist.group.WORLD, return_target_logz=False, **training_options,
                )
                assert original_reductions - collective.call_count == len(_requests())
            expected_lean = tuple(
                replace(row, branches=tuple(replace(branch, target_logz=None) for branch in row.branches))
                for row in original
            )
            _assert_scores(lean, expected_lean, atol=0, rtol=0)
        assert metrics["sp_size"] == size and metrics["model_forward_calls"] == 1
        assert before == [layer.self_attn.forward.__func__ for layer in model.model.layers]
        # Future trunk and sibling history must never leak into branch a0.
        req = _requests()[0]
        changed = replace(req, trunk_ids=(*req.trunk_ids[:-1], 23), branches=(req.branches[0], replace(req.branches[1], token_ids=(24, 25, 26))))
        isolated, _ = score_ulysses_joint_batch(model, (changed,), dist.group.WORLD, **options)
        torch.testing.assert_close(isolated[0].branches[0].target_logits, actual[0].branches[0].target_logits, atol=2e-5, rtol=2e-5)
        if size == 4:
            trunks = tuple(replace(r, branches=()) for r in _requests())
            mixed = (_requests()[0], trunks[1])
            for batch, has_tree in ((trunks, False), (mixed, True)):
                reference, baseline_metrics = score_ulysses_joint_batch(
                    model, batch, dist.group.WORLD, use_causal_fast_path=False, **options,
                )
                fast, measured = score_ulysses_joint_batch(
                    model, batch, dist.group.WORLD, use_causal_fast_path=True, **options,
                )
                _assert_scores(fast, reference)
                assert (measured["mask_elements"] > 0) == has_tree
                if has_tree:
                    assert measured["mask_elements"] * 2 == baseline_metrics["mask_elements"]
                assert measured["model_forward_calls"] == 1
                if has_tree:
                    # A neighboring trunk-only row cannot change the tree's scores.
                    for a, b in zip(fast[0].branches, reference[0].branches, strict=True):
                        torch.testing.assert_close(a.target_logits, b.target_logits, atol=0, rtol=0)
            short = replace(mixed[1], sample_id="short", branches=(DraftBranch("anchor-only", 4, (10,)),))
            mixed = (*mixed, short)
            reference, dense_mixed = score_ulysses_joint_batch(model, mixed, dist.group.WORLD, **options)
            segmented, measured = score_ulysses_joint_batch(
                model, mixed, dist.group.WORLD, use_segmented_attention=True, use_packed_qkv=True, **options,
            )
            _assert_scores(segmented, reference)
            assert measured["model_forward_calls"] == 1
            # Four branch-tail queries see six prefix and four branch columns.
            assert measured["mask_elements"] == 4 * (6 + 4)
            separate, _ = score_ulysses_joint_batch(
                model, (short,), dist.group.WORLD, use_segmented_attention=True, **options,
            )
            _assert_scores(separate, segmented[2:])
            isolated, _ = score_ulysses_joint_batch(
                model, (changed, *mixed[1:]), dist.group.WORLD,
                use_segmented_attention=True, use_packed_qkv=True, **options,
            )
            torch.testing.assert_close(
                isolated[0].branches[0].target_logits, segmented[0].branches[0].target_logits, atol=0, rtol=0,
            )
            precise, measured = score_ulysses_joint_batch(
                model, mixed, dist.group.WORLD, use_segmented_dense_prefix=True,
                use_packed_qkv=True, **options,
            )
            _assert_scores(precise, reference)
            assert measured["model_forward_calls"] == 1
            width = 16
            assert measured["mask_elements"] == (6 + 4 + 5) * width
            for row, readers in ((0, 6), (2, 5)):
                # Full key width protects the dense prefix and branch states.
                for index in precise[row].trunk_features:
                    torch.testing.assert_close(
                        precise[row].trunk_features[index][:readers],
                        reference[row].trunk_features[index][:readers], atol=2e-6, rtol=2e-6,
                    )
                for actual_branch, expected_branch in zip(precise[row].branches, reference[row].branches, strict=True):
                    torch.testing.assert_close(
                        actual_branch.target_logits, expected_branch.target_logits, atol=2e-6, rtol=2e-6,
                    )
            precise_changed, _ = score_ulysses_joint_batch(
                model, (changed, *mixed[1:]), dist.group.WORLD,
                use_segmented_dense_prefix=True, use_packed_qkv=True, **options,
            )
            torch.testing.assert_close(
                precise_changed[0].branches[0].target_logits, precise[0].branches[0].target_logits, atol=0, rtol=0,
            )
            dense_one, _ = score_ulysses_joint_batch(model, (req,), dist.group.WORLD, **options)
            with patch.object(torch, "matmul", wraps=torch.matmul) as matmul:
                preserved, measured = score_ulysses_joint_batch(
                    model, (req,), dist.group.WORLD, use_preserve_pv=True, **options,
                )
            assert measured["model_forward_calls"] == 1
            assert measured["mask_elements"] == (6 + 4) * width
            # Preserve the original, square query/key geometry specifically for P@V.
            value_calls = [call.args for call in matmul.call_args_list if call.args[0].shape[-2:] == (width, width)]
            assert len(value_calls) == cfg.num_hidden_layers
            _assert_scores(preserved, dense_one)
            fallback, measured = score_ulysses_joint_batch(
                model, mixed, dist.group.WORLD, use_preserve_pv=True,
                use_segmented_dense_prefix=True, use_causal_fast_path=True, **options,
            )
            assert measured["mask_elements"] == dense_mixed["mask_elements"]
            _assert_scores(fallback, reference, atol=0, rtol=0)
            engine_schedule = SimpleNamespace(module=model, eval_mode=nullcontext,
                use_remove_padding=True, use_ulysses_sp=True, ulysses_sequence_parallel_size=size,
                ulysses_parallel_group=dist.group.WORLD, model_config=SimpleNamespace(use_fused_kernels=False),
                _autocast_dtype=torch.float32)
            schedule_worker = SimpleNamespace(_is_actor=True, actor=SimpleNamespace(engine=engine_schedule))
            with patch.dict("os.environ", {"VERL_SPECO_JOINT_PRESERVE_PV": str(int(rank == 0))}):
                with pytest.raises(RuntimeError, match="same complete trajectory schedule"):
                    score_actor_joint(schedule_worker, (req,), **options)
            exported, measured = score_actor_joint(schedule_worker, _requests(), microbatch_sizes=(1, 1), **options)
            if rank == 0:
                _assert_scores(exported, expected)
                assert measured["sp_size"] == size and measured["model_forward_calls"] == 2
            else:
                assert not exported and measured["model_forward_calls"] == 0
            groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            group = groups[rank // 2]
            engine = SimpleNamespace(module=model, eval_mode=nullcontext,
                use_remove_padding=True, use_ulysses_sp=True, ulysses_sequence_parallel_size=2,
                ulysses_parallel_group=group, model_config=SimpleNamespace(use_fused_kernels=False),
                _autocast_dtype=torch.float32)
            worker = SimpleNamespace(_is_actor=True, actor=SimpleNamespace(engine=engine))
            exported, measured = score_actor_joint(worker, _requests(), **options)
            if rank % 2 == 0:
                _assert_scores(exported, expected[rank // 2:rank // 2 + 1])
                assert measured["model_forward_calls"] == 1
            else:
                assert exported == () and measured["model_forward_calls"] == 0
            ids = [None] * size
            dist.all_gather_object(ids, [r.sample_id for r in exported])
            assert [sample for rows in ids for sample in rows] == ["a", "b"]
            exported, measured = score_actor_joint(worker, _requests()[:1], **options)
            dist.all_gather_object(ids, [r.sample_id for r in exported])
            assert [sample for rows in ids for sample in rows] == ["a"]
            if rank >= 2:
                assert not exported and measured["logical_rows"] == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size,kv_heads,kind", [(2, 2, "qwen3"), (4, 1, "qwen3"), (2, 1, "qwen2"), (2, 1, "llama")])
def test_ulysses_matches_joint_reference(tmp_path, size, kv_heads, kind):
    if not dist.is_gloo_available():
        pytest.skip("requires CPU Gloo collectives")
    mp.spawn(_run, args=(size, str(tmp_path / "init"), kv_heads, kind), nprocs=size, join=True)


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
