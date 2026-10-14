# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two-rank HCCL regression: gradients, updates, and optimizer resume.

Run with torchrun --standalone --nproc-per-node=2 ci/npu_hccl_smoke.py.
Uses a small model to isolate communication from model-cache loading.
"""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.nn.parallel import DistributedDataParallel


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        assert dist.get_world_size() == 2, "exactly two ranks required"
        torch.manual_seed(42)
        model = DistributedDataParallel(
            torch.nn.Linear(4, 2).to(f"npu:{rank}"), device_ids=[rank]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.full((3, 4), float(rank + 1), device=f"npu:{rank}")

        def step():
            optimizer.zero_grad()
            loss = model(inputs).square().mean()
            assert torch.isfinite(loss).all()
            loss.backward()
            for parameter in model.parameters():
                gradient = parameter.grad.detach().clone()
                reference = gradient.clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(gradient, reference)
            optimizer.step()
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(parameter, reference)

        initial = [p.detach().clone() for p in model.parameters()]
        step()
        assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
        with tempfile.TemporaryDirectory(prefix="speco-hccl-") as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )
            step()
            expected = [p.detach().clone() for p in model.parameters()]
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.module.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            step()
            for actual, reference in zip(model.parameters(), expected):
                torch.testing.assert_close(actual, reference)
            assert all(int(state["step"]) == 2 for state in optimizer.state.values())
        dist.barrier()
        print(f"HCCL rank {rank}: gradient/update/checkpoint resume passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
