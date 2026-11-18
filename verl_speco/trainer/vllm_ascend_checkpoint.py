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
"""Native DeepSpeed recovery with a versioned SpeCo data/implementation cursor."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from verl_speco.backends.vllm_ascend_engine import Progress
from verl_speco.integration.vllm_ascend_implementation import (
    implementation_identity_matches,
)
from verl_speco.trainer.vllm_ascend_state import save_deepspeed_checkpoint

# Persisted identifier: preserve existing checkpoints across module renames.
STATE_KEY = "speco_ds4_v1"


def rank_rng_state() -> dict:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        value["npu"] = npu.get_rng_state()
    return value


def restore_rng_state(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if "npu" in value:
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("Cannot restore NPU RNG state without torch_npu")
        npu.set_rng_state(value["npu"].cpu())


def save_checkpoint(engine, *, base_dir, tag: str, identity: dict, store) -> str:
    import torch.distributed as dist

    state = rank_rng_state()
    rank_states: list[Any] = [state]
    if dist.is_initialized():
        rank_states = [None] * dist.get_world_size()
        dist.all_gather_object(rank_states, state)
    payload = {
        "identity": identity,
        "progress": asdict(engine.progress),
        "store": store.checkpoint_state(),
        "rng_by_rank": rank_states,
        "topology": engine.topology.metadata(),
    }
    save_deepspeed_checkpoint(
        engine.model_engine,
        str(base_dir),
        tag,
        engine.logger,
        client_state={STATE_KEY: payload},
        save_full_model_per_rank=True,
        parallel_topology=engine.topology,
    )
    from pathlib import Path

    return str(Path(base_dir) / tag)


def restore_checkpoint(engine, *, path, identity: dict, store) -> None:
    """Preflight the sidecar before mutating the engine, then verify restored state."""
    import json
    from pathlib import Path

    # AIR's loader recognizes epoch_* tags; other tags must use a parent/latest.
    checkpoint_path = Path(path).resolve(strict=True)
    metadata = json.loads((checkpoint_path / "speco_ds4.json").read_text())
    if not implementation_identity_matches(metadata["identity"], identity):
        raise ValueError(
            "Recovery identity differs (implementation, model, data settings or feature contract)"
        )
    expected = engine.topology.metadata()
    for key in ("world_size", "tensor_parallel_size", "data_parallel_size"):
        if metadata["topology"].get(key) != expected[key]:
            raise ValueError(f"Recovery topology differs at {key}")
    store.restore_checkpoint_state(metadata["store"])
    # DeepSpeed accepts explicit tags, including step_* tags used by SpeCo.
    loaded, client = engine.model_engine.load_checkpoint(
        str(checkpoint_path.parent), tag=checkpoint_path.name
    )
    if loaded is None or STATE_KEY not in (client or {}):
        raise RuntimeError("Checkpoint does not contain SpeCo Ascend recovery state")
    from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

    refresh_fp32_weight_caches(engine.model_engine.module)
    state = client[STATE_KEY]
    if (
        not implementation_identity_matches(state["identity"], identity)
        or state["progress"] != metadata["progress"]
        or state["store"] != metadata["store"]
    ):
        raise ValueError("DeepSpeed state differs from recovery metadata")
    store.restore_checkpoint_state(state["store"])
    engine.progress = Progress(**state["progress"])
    if int(engine.model_engine.global_steps) != engine.progress.global_step:
        raise RuntimeError("DeepSpeed global step was not restored")
    # DeepSpeed restores optimizer and scheduler by default. RNG is rank-local.
    rank = int(engine.topology.metadata().get("global_rank", 0))
    restore_rng_state(state["rng_by_rank"][rank])


def write_checkpoint_metadata(path, *, engine, identity: dict, store) -> None:
    """Mark a collectively saved checkpoint usable only after save succeeds."""
    import json
    import os
    from pathlib import Path

    target = Path(path) / "speco_ds4.json"
    temporary = target.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(
            {
                "identity": identity,
                "progress": asdict(engine.progress),
                "topology": engine.topology.metadata(),
                "store": store.checkpoint_state(),
            },
            indent=2,
        )
    )
    os.replace(temporary, target)
