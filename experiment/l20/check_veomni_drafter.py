"""Dedicated two-rank VeOmni 0.1.11 materialization and optimizer oracle."""

from copy import deepcopy
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from verl_speco.trainer.veomni_drafter import wrap_veomni_drafter

from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
torch.manual_seed(17)
config = PeagleConfig(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=2,
    num_hidden_layers=2,
    num_draft_layers=2,
    target_hidden_size=32,
    num_aux_hidden_states=3,
    vocab_size=64,
    num_depths=3,
    mask_token_id=63,
    max_position_embeddings=64,
)
reference = PEagleTrainingModel(LlamaForCausalLMPeagle(config), num_depths=3)
candidate = wrap_veomni_drafter(deepcopy(reference), mesh, mixed_precision=False)
reference = reference.cuda()
fully_shard(reference)
for name, parameter in candidate.named_parameters():
    torch.testing.assert_close(
        parameter.full_tensor(),
        dict(reference.named_parameters())[name].full_tensor(),
        rtol=0,
        atol=0,
        msg=name,
    )
for name, buffer in candidate.named_buffers():
    torch.testing.assert_close(
        buffer, dict(reference.named_buffers())[name], rtol=0, atol=0
    )
torch.manual_seed(23 + rank)
mask = torch.ones(1, 12, device="cuda")
mask[:, : rank + 1] = 0
batch = dict(
    input_ids=torch.randint(0, 63, (1, 12), device="cuda"),
    aux_hidden=torch.randn(1, 12, 96, device="cuda"),
    loss_mask=mask,
    attention_mask=torch.ones_like(mask),
    target_logits=torch.randn(1, 12, 64, device="cuda"),
    seq_lengths=torch.tensor([5, 7], device="cuda"),
)
optimizers = [
    torch.optim.AdamW(model.parameters(), lr=0.001) for model in (reference, candidate)
]
for step in range(3):
    losses = []
    norms = []
    for model, optimizer in zip((reference, candidate), optimizers, strict=True):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(41 + rank + step)
        numerator, denominator, _ = model(**deepcopy(batch))
        global_denominator = denominator.detach().clone()
        dist.all_reduce(global_denominator)
        loss = numerator * dist.get_world_size() / global_denominator
        losses.append(loss.detach())
        loss.backward()
        norms.append(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).full_tensor()
        )
        optimizer.step()
    torch.testing.assert_close(*losses, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(*norms, atol=2e-5, rtol=2e-4)
    for name, actual in candidate.named_parameters():
        expected = dict(reference.named_parameters())[name]
        torch.testing.assert_close(
            actual.full_tensor(), expected.full_tensor(), atol=2e-5, rtol=2e-4, msg=name
        )
        torch.testing.assert_close(
            actual.grad.full_tensor(),
            expected.grad.full_tensor(),
            atol=2e-5,
            rtol=2e-4,
            msg=name,
        )
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizers[1].state[actual][key].full_tensor(),
                optimizers[0].state[expected][key].full_tensor(),
                atol=2e-5,
                rtol=2e-4,
                msg=f"{name}/{key}",
            )
    print(
        f"VeOmni rank={rank} step={step + 1}: checkpoint, buffers, loss, gradients, clip, AdamW states PASS",
        flush=True,
    )
dist.destroy_process_group()
