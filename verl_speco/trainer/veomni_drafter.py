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
"""VeOmni wrapping for a dense drafter on a fixed, full-world FSDP2 mesh."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


def wrap_veomni_drafter(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    *,
    mixed_precision: bool = True,
) -> torch.nn.Module:
    """Preserve the algorithm wrapper's complete state while VeOmni shards it.

    VeOmni 0.1.11 materializes from meta and owns a process-wide mesh. Only a
    one-dimensional mesh spanning the default group is supported here. The
    temporary checkpoint bridges its loader without changing parameter names,
    buffers, or the trainer's optimizer and publication contracts.
    """
    ranks = mesh.mesh.reshape(-1)
    if sum(size > 1 for size in mesh.shape) > 1 or ranks.tolist() != list(
        range(dist.get_world_size())
    ):
        raise ValueError("VeOmni drafter requires a one-dimensional full-world mesh")

    from safetensors.torch import save_file
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.parallel_state import (
        get_parallel_state,
        init_parallel_state,
    )
    from veomni.distributed.torch_parallelize import build_parallelize_model

    init_parallel_state(dp_size=dist.get_world_size(), dp_mode="fsdp2")
    state = get_parallel_state()
    if state.dp_mode != "fsdp2" or not torch.equal(state.fsdp_mesh.mesh, ranks):
        raise ValueError("Existing VeOmni parallel state differs from the drafter mesh")

    # Each rank owns its staging directory; it is removed immediately after load.
    buffers = {
        name: value.detach().cpu().clone() for name, value in model.named_buffers()
    }
    with TemporaryDirectory(prefix="speco-veomni-") as directory:
        save_file(
            {
                name: value.detach().cpu().contiguous().clone()
                for name, value in model.state_dict().items()
            },
            str(Path(directory) / "model.safetensors"),
        )
        model.to("meta")
        # VeOmni's loader copies existing non-persistent buffers during loading.
        for name, value in buffers.items():
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, value.to(mesh.device_type))
        model = build_parallelize_model(
            model,
            weights_path=directory,
            init_device="meta",
            mixed_precision=MixedPrecisionConfig(enable=mixed_precision),
            enable_gradient_checkpointing=False,
        )
    return model
