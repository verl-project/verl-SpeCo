"""Real multi-process Ulysses collectives vs full causal/tree inference."""

from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch
from verl_speco.joint_ulysses import score_ulysses_joint_batch
from verl_speco.joint_actor import score_actor_joint


def _requests():
    return (
        JointScoringRequest("a", (1, 2, 3, 4, 5, 6, 7, 8, 9), 3, (
            DraftBranch("a0", 3, (10, 11, 12)), DraftBranch("a1", 5, (13, 14, 15)),
        )),
        JointScoringRequest("b", (3, 4, 5, 6, 7), 2, (DraftBranch("b0", 1, (8, 9)),)),
    )


def _assert_scores(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert a.sample_id == b.sample_id
        torch.testing.assert_close(a.response_log_probs, b.response_log_probs, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a.response_entropy, b.response_entropy, atol=2e-5, rtol=2e-5)
        for index in a.trunk_features:
            torch.testing.assert_close(a.trunk_features[index], b.trunk_features[index], atol=2e-5, rtol=2e-5)
        for x, y in zip(a.branches, b.branches, strict=True):
            assert x.branch_id == y.branch_id
            for field in ("target_logits", "target_logz", "target_hidden"):
                torch.testing.assert_close(getattr(x, field), getattr(y, field), atol=2e-5, rtol=2e-5)


def _run(rank, size, init_file, kv_heads, kind):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=size)
    try:
        torch.manual_seed(193)
        config_cls, model_cls = {
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "llama": (LlamaConfig, LlamaForCausalLM),
        }[kind]
        cfg = config_cls(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=8,
                         max_position_embeddings=128, attention_dropout=0.0)
        cfg._attn_implementation = "eager"
        model = model_cls(cfg).eval()
        before = [layer.self_attn.forward.__func__ for layer in model.model.layers]
        options = dict(hidden_state_indices=(0, 1, 2), calculate_entropy=True, return_target_logits=True)
        expected, _ = score_joint_batch(model, _requests(), sequence_multiple=size, **options)
        actual, metrics = score_ulysses_joint_batch(model, _requests(), dist.group.WORLD, **options)
        _assert_scores(actual, expected)
        assert metrics["sp_size"] == size and metrics["model_forward_calls"] == 1
        assert before == [layer.self_attn.forward.__func__ for layer in model.model.layers]
        # Future trunk and sibling history must never leak into branch a0.
        req = _requests()[0]
        changed = replace(req, trunk_ids=(*req.trunk_ids[:-1], 23), branches=(req.branches[0], replace(req.branches[1], token_ids=(24, 25, 26))))
        isolated, _ = score_ulysses_joint_batch(model, (changed,), dist.group.WORLD, **options)
        torch.testing.assert_close(isolated[0].branches[0].target_logits, actual[0].branches[0].target_logits, atol=2e-5, rtol=2e-5)
        if size == 4:
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
