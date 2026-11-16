"""Opt-in HCCL publish regression, including a 2.32GiB frozen embedding."""
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.tensor import DTensor, Shard
from types import SimpleNamespace


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    spec = importlib.util.spec_from_file_location('publish_contract', Path(__file__).with_name('test_selective_publish_state.py'))
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    # Legacy/selected state, dtype, patterns, checkpoint, consumer SGD and
    # versioned snapshot lifecycle on real NPU DTensors and HCCL.
    contract._distributed_check(rank, os.environ['PUBLISH_RENDEZVOUS'], 'hccl')
    dist.init_process_group('hccl')
    try:
        mesh = dist.device_mesh.init_device_mesh('npu', (2,))
        rows, columns = 151936, 4096
        local = torch.zeros((rows // 2, columns), device=f'npu:{rank}')
        weight = DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                                    shape=torch.Size((rows, columns)), stride=(columns, 1))
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(1, 1)
        model.embed_tokens.weight = torch.nn.Parameter(weight, requires_grad=False)
        model.fc = torch.nn.Linear(4, 4, bias=False)
        source = torch.arange(16, dtype=torch.float32, device=f'npu:{rank}').reshape(4, 4)
        model.fc.weight = torch.nn.Parameter(DTensor.from_local(source[rank*2:rank*2+2].clone(), mesh, [Shard(0)],
            run_check=False, shape=torch.Size((4, 4)), stride=(4, 1)))
        trainer = contract._trainer_type('npu')()
        trainer.model, trainer.training_device_mesh, trainer.rank = model, mesh, rank
        trainer._is_ddp_wrapped = lambda: False
        trainer._frozen_param_names = []
        trainer.backend = SimpleNamespace(model_type='dspark',trains_draft_lm_head=False,trains_draft_embeddings=False)
        trainer.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
            'drafter_opd':{'enable':True,'method':'spec_opd'}, 'publish_dtype':'bf16'})))
        stats = {}
        trainer._record_gpu_transfer = lambda scope, values: stats.update(values)
        dist.barrier()
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        # Limit only this process allocator: 1GiB headroom, below the 2.32GiB
        # full embedding gather. No device-wide filler allocations or OOM.
        capacity = torch.npu.get_device_properties(rank).total_memory
        torch.npu.set_per_process_memory_fraction(float((before + 1024**3) / capacity))
        original = DTensor.full_tensor
        gathered = []
        def track(tensor, *args, **kwargs):
            gathered.append(tuple(tensor.shape))
            assert tuple(tensor.shape) != (rows, columns), 'frozen embedding must never gather'
            return original(tensor, *args, **kwargs)
        DTensor.full_tensor = track
        try:
            trainer.clear_pending_publish_state_dict()
            assert trainer.prepare_model_state_dict_for_publish(1) == (rank == 0)
            ready, snapshot = trainer.pop_model_state_dict_for_publish(1)
            assert ready
            assert gathered == [(4, 4)]
            if rank == 0:
                assert set(snapshot) == {'fc.weight'}
                torch.testing.assert_close(snapshot['fc.weight'], source.cpu().bfloat16(), rtol=0,atol=0)
                assert stats['skipped_bytes'] == rows*columns*4
            else:
                assert snapshot is None
            torch.npu.synchronize()
            peak_delta = torch.npu.max_memory_allocated() - before
            assert peak_delta < 1024**3
            print(json.dumps(dict(rank=rank,frozen_bytes=rows*columns*4,headroom_bytes=1024**3,
                                  gathered_shapes=gathered,peak_delta_bytes=peak_delta)),flush=True)
        finally:
            DTensor.full_tensor = original
            torch.npu.set_per_process_memory_fraction(1.0)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
