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
"""Restricted actor-engine adapter for the dense joint scoring executor."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from verl_speco.joint_scoring import (
    JointScoringRequest,
    pack_joint_batch,
    score_joint_batch,
)


def partition_joint_requests(requests, rank, world_size):
    """Never split a tree; empty ranks still participate in one FSDP forward."""
    local = tuple(requests[rank::world_size])
    if local:
        return local, False
    if not requests:
        raise ValueError("joint distributed schedule is empty")
    dummy = JointScoringRequest(
        f"__specopd_dummy_{rank}", (requests[0].trunk_ids[0],), 1
    )
    return (dummy,), True


def score_actor_joint(worker, requests, **options):
    """Run inside the RL actor's own eval/offload context, never the SFT worker.

    ONE_TO_ALL receives the same global schedule; each FSDP rank owns complete
    trajectories. Empty ranks execute a one-token dummy to match collectives.
    Runtime FSDP/NPU validation is separate from these dispatch contracts.
    """
    distributed = torch.distributed.is_initialized()
    if distributed:
        signature = repr(
            [
                (
                    r.sample_id,
                    r.trunk_ids,
                    r.response_start,
                    [(b.branch_id, b.anchor, b.token_ids) for b in r.branches],
                )
                for r in requests
            ]
        ) + repr(sorted(options.items()))
        import hashlib

        digest = hashlib.sha256(signature.encode()).hexdigest()
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise RuntimeError(
                "joint actor ranks must receive the same complete trajectory schedule"
            )
    engine = getattr(getattr(worker, "actor", None), "engine", None)
    error = None
    try:
        if not getattr(worker, "_is_actor", False):
            raise ValueError("joint scoring must run on the RL actor worker")
        if engine is None or not callable(getattr(engine, "eval_mode", None)):
            raise ValueError("actor engine must expose module and eval_mode")
        if getattr(engine, "use_remove_padding", True):
            raise NotImplementedError(
                "joint scoring requires model.use_remove_padding=false"
            )
        if getattr(engine, "use_ulysses_sp", False):
            raise NotImplementedError(
                "joint scoring does not support sequence parallelism"
            )
        model_config = getattr(engine, "model_config", None)
        if getattr(model_config, "use_fused_kernels", False):
            raise NotImplementedError("joint scoring requires use_fused_kernels=false")
        if getattr(model_config, "lora_rank", 0) or getattr(
            model_config, "lora_adapter_path", None
        ):
            raise NotImplementedError(
                "joint scoring has not validated LoRA actor wrappers"
            )
        # Validate every complete tree before partitioning and before FSDP
        # collectives. CPU preflight allocations are bounded by the same mask
        # budget; a bad tree on one rank cannot leave another inside forward.
        config = engine.module.config
        if config.model_type not in {"qwen2", "qwen3", "llama"} or getattr(
            config, "_attn_implementation", None
        ) not in {"eager", "sdpa"}:
            raise ValueError(
                "joint actor requires a supported full-attention eager/SDPA model"
            )
        if getattr(config, "sliding_window", None) is not None or any(
            layer != "full_attention"
            for layer in (getattr(config, "layer_types", None) or ())
        ):
            raise ValueError("joint actor does not support sliding/hybrid attention")
        packed = pack_joint_batch(
            requests,
            device="cpu",
            dtype=torch.float32,
            vocab_size=config.vocab_size,
            max_packed_tokens=options.get("max_packed_tokens", 8192),
            max_mask_elements=options.get("max_mask_elements", 16777216),
        )
        if int(packed.position_ids.max()) >= config.max_position_embeddings:
            raise ValueError("joint actor exceeds max_position_embeddings")
        del packed
        for i in options.get("hidden_state_indices", ()):
            if type(i) is not int or not 0 <= i <= config.num_hidden_layers:
                raise ValueError("joint actor auxiliary hidden-state index is invalid")
    except Exception as exc:
        error = exc
    if distributed:
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, str(error) if error else None)
        if any(errors):
            raise RuntimeError(f"joint actor preflight failed: {errors}")
    elif error:
        raise error
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    requests, dummy = partition_joint_requests(requests, rank, world)
    model = engine.module
    dtype = getattr(engine, "_autocast_dtype", torch.float32)
    states = [(module, module.training) for module in model.modules()]
    try:
        with engine.eval_mode():
            device_type = next(model.parameters()).device.type
            autocast = (
                nullcontext()
                if dtype == torch.float32
                else torch.autocast(device_type, dtype=dtype)
            )
            with autocast:
                result, metrics = score_joint_batch(model, requests, **options)
            # Materialize independent CPU snapshots before the engine offloads.
            for sample in result:
                sample.response_log_probs = sample.response_log_probs.cpu()
                if sample.response_entropy is not None:
                    sample.response_entropy = sample.response_entropy.cpu()
                sample.trunk_features = {
                    i: value.cpu() for i, value in sample.trunk_features.items()
                }
                for branch in sample.branches:
                    if branch.target_logits is not None:
                        branch.target_logits = branch.target_logits.cpu()
                    branch.target_logz = branch.target_logz.cpu()
                    branch.target_hidden = branch.target_hidden.cpu()
            metrics["dummy_rows"] = metrics["padded_rows"] if dummy else 0
            if dummy:
                result = ()
                metrics["logical_rows"] = metrics["independent_rows"] = 0
            return result, metrics
    finally:
        for module, training in states:
            module.training = training
