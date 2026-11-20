"""Check complete-group gradients after DP partitioning, including dummy slots."""
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from verl_speco.backends.specopd_forward import partition_specopd_samples
from verl_speco.backends.specopd_loss import e2e_tv_loss


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
                numerator, count, _ = e2e_tv_loss(model(x), teacher, valid,
                    position_weights=torch.tensor([[1., 2., 1.]]) if aligned else None)
                counts = count.detach().clone()
                dist.all_reduce(counts)
                token_total += int(counts)
                (numerator * world).backward()
            for parameter in model.parameters():
                parameter.grad.div_(token_total)
            ref_numerator, ref_count = 0, 0
            for x, teacher, mask in samples:
                n, c, _ = e2e_tv_loss(reference(x), teacher, mask,
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
