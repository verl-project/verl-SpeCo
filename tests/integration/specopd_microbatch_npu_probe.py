"""Opt-in NPU oracle for four ragged/dummy DSpark trajectories vs accumulation."""
import json
import torch
import torch_npu  # noqa: F401
from types import SimpleNamespace
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel
from verl_speco.backends.specopd_forward import merge_specopd_training_batches, compute_specopd_backend_loss


def main():
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(41)
        cfg = DSparkConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_context_layers=2, target_hidden_size=32, target_num_hidden_layers=2,
            mask_token_id=127, max_position_embeddings=128, block_size=3,
            conv_group_size=8, markov_rank=8)
        wrapper = DSparkTrainingModel(draft_model=DSparkDraftModel(cfg), block_size=3,
            num_anchors=2, drafter_opd_config=dict(enable=True,method='spec_opd',capture_verify_trace=True)).to(device='npu',dtype=dtype)
        head = torch.nn.Linear(32,128,bias=False).to(device='npu',dtype=dtype).requires_grad_(False)
        backend=SimpleNamespace(target_lm_head=SimpleNamespace(fc=head))
        batches=[]
        for i in range(4):
            length, anchors = 8+i, 1+i%2
            mask=torch.ones(anchors,3,dtype=torch.bool)
            if i==1:
                mask[-1,-1]=False
            if i==3:
                mask.zero_()
            batch=dict(input_ids=torch.randint(1,100,(1,length)),hidden_states=torch.randn(1,length,64),
                anchor_positions=torch.tensor([[2,5][:anchors]]),proposal_ids=torch.randint(1,100,(anchors,3)),
                proposal_mask=mask,target_last_hidden_states=torch.randn(anchors,3,32),
                proposal_temperature=1.0,target_temperature=1.0,proposal_block_width=3,
                specopd=True,_speco_sample_count=anchors,_speco_sample_ids=(str(i),),_specopd_trunk_digest=str(i))
            batches.append({k:v.to(device='npu',dtype=dtype if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v for k,v in batch.items()})
        params=tuple(p for p in wrapper.parameters() if p.requires_grad)
        separate=[compute_specopd_backend_loss(backend,wrapper,b) for b in batches]
        count=sum(b['local_num_tokens'] for b in separate)
        loss=sum(b['total_local_ploss'] for b in separate)/count
        expected=torch.autograd.grad(loss,params,allow_unused=True)
        merged=compute_specopd_backend_loss(backend,wrapper,merge_specopd_training_batches(batches))
        actual_loss=merged['total_local_ploss']/merged['local_num_tokens']
        actual=torch.autograd.grad(actual_loss,params,allow_unused=True)
        # Predeclared tolerances: full precision and BF16 batching roundoff.
        atol,rtol=(1e-4,1e-3) if dtype==torch.float32 else (0.01,0.05)
        torch.testing.assert_close(actual_loss,loss,atol=atol,rtol=rtol)
        maximum=0.0
        for a,b in zip(actual,expected,strict=True):
            if a is None or b is None:
                assert a is b
            else:
                maximum=max(maximum,(a.float()-b.float()).abs().max().item())
                torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        print(json.dumps(dict(dtype=str(dtype),loss_error=abs((actual_loss-loss).item()),max_grad_abs_error=maximum,atol=atol,rtol=rtol)),flush=True)


if __name__=='__main__':
    main()
