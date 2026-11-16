"""Opt-in torchrun HCCL oracle for tiny real Qwen3 tree attention."""
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch
from verl_speco.joint_ulysses import score_ulysses_joint_batch
from verl_speco.joint_actor import score_actor_joint


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    if hasattr(torch.npu.matmul, "allow_hf32"):
        torch.npu.matmul.allow_hf32 = False
    dist.init_process_group("hccl")
    size = dist.get_world_size()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(93)
            cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=256, attention_dropout=0)
            cfg._attn_implementation = "eager"
            model = Qwen3ForCausalLM(cfg).eval().to(dtype=dtype, device=f"npu:{rank}")
            request = JointScoringRequest("npu", tuple(range(1, 18)), 4, (
                DraftBranch("b0", 5, (21, 22, 23, 24)), DraftBranch("b1", 10, (31, 32, 33)),
            ))
            options = dict(hidden_state_indices=(0, 1, 2), calculate_entropy=True)
            reference, _ = score_joint_batch(model, (request,), sequence_multiple=size, **options)
            actual, metrics = score_ulysses_joint_batch(model, (request,), dist.group.WORLD, **options)
            errors = {
                "response_logprob": (actual[0].response_log_probs-reference[0].response_log_probs).abs().max().item(),
                "branch_logits": max((a.target_logits.float()-b.target_logits.float()).abs().max().item() for a,b in zip(actual[0].branches,reference[0].branches,strict=True)),
                "branch_hidden": max((a.target_hidden.float()-b.target_hidden.float()).abs().max().item() for a,b in zip(actual[0].branches,reference[0].branches,strict=True)),
            }
            tolerance = 5e-5 if dtype==torch.float32 else 0.03
            assert max(errors.values()) < tolerance, errors
            if rank==0:
                print(json.dumps(dict(dtype=str(dtype), sp=size, errors=errors, tolerance=tolerance, metrics=metrics)),flush=True)
            # Four physical ranks, two independent SP=2 groups; the second DP
            # group executes a dummy tree and must export no duplicate rows.
            if size == 4:
                groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
                engine = SimpleNamespace(module=model, eval_mode=nullcontext, use_remove_padding=True,
                    use_ulysses_sp=True, ulysses_sequence_parallel_size=2,
                    ulysses_parallel_group=groups[rank // 2],
                    model_config=SimpleNamespace(use_fused_kernels=False), _autocast_dtype=dtype)
                worker = SimpleNamespace(_is_actor=True, actor=SimpleNamespace(engine=engine))
                exported, measured = score_actor_joint(worker, (request,), **options)
                identities = [None] * size
                dist.all_gather_object(identities, [r.sample_id for r in exported])
                assert [item for rows in identities for item in rows] == ["npu"]
                if rank == 0:
                    error = (exported[0].response_log_probs.cpu()-reference[0].response_log_probs.cpu()).abs().max().item()
                    assert error < tolerance,error
                    print(json.dumps(dict(dtype=str(dtype),dp=2,sp=2,leader_exports=identities,logprob_error=error)),flush=True)
                elif rank >= 2:
                    assert not exported and measured['logical_rows']==0
            del model
            torch.npu.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
