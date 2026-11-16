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


def score_actor_joint(worker, requests, *, microbatch_sizes=None, **options):
    """Keep one actor residency context across a bounded microbatch schedule."""
    if microbatch_sizes is None:
        return _score_actor_joint(worker, requests, **options)
    sizes = tuple(microbatch_sizes)
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes) or sum(sizes) != len(requests):
        raise ValueError("joint microbatch sizes must partition the complete request schedule")
    if torch.distributed.is_initialized():
        import hashlib

        digest = hashlib.sha256(repr((requests, sizes, sorted(options.items()))).encode()).hexdigest()
        schedules = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(schedules, digest)
        if len(set(schedules)) != 1:
            raise RuntimeError("joint actor ranks received different microbatch schedules")
    engine = worker.actor.engine
    states = [(m, m.training) for m in engine.module.modules()]
    results, totals, offset = [], {}, 0
    try:
        with engine.eval_mode():
            for length in sizes:
                scored, measured = _score_actor_joint(
                    worker, requests[offset:offset + length], manage_engine=False, **options,
                )
                results.extend(scored)
                for key, value in measured.items():
                    if key == "sp_size":
                        if key in totals and totals[key] != value:
                            raise RuntimeError("joint microbatch SP size changed")
                        totals[key] = value
                    else:
                        totals[key] = totals.get(key, 0) + value
                offset += length
    finally:
        for module, training in states:
            module.training = training
    return tuple(results), totals


def _score_actor_joint(worker, requests, *, manage_engine=True, **options):
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
    sp_size = int(getattr(engine, "ulysses_sequence_parallel_size", 1) or 1)
    sp_group = getattr(engine, "ulysses_parallel_group", None)
    error = None
    try:
        if not getattr(worker, "_is_actor", False):
            raise ValueError("joint scoring must run on the RL actor worker")
        if engine is None or not callable(getattr(engine, "eval_mode", None)):
            raise ValueError("actor engine must expose module and eval_mode")
        if getattr(engine, "use_remove_padding", True) and sp_size == 1:
            raise NotImplementedError(
                "joint scoring requires model.use_remove_padding=false"
            )
        if getattr(engine, "use_ulysses_sp", False) and sp_size == 1:
            raise NotImplementedError("joint Ulysses requires an explicit SP size")
        if sp_size > 1:
            if not distributed or sp_group is None or torch.distributed.get_world_size(sp_group) != sp_size:
                raise ValueError("joint Ulysses requires the actor engine's initialized SP group")
            if not getattr(engine, "use_remove_padding", False):
                raise ValueError("upstream Ulysses requires use_remove_padding=true")
            members = torch.distributed.get_process_group_ranks(sp_group)
            global_rank = torch.distributed.get_rank()
            expected_members = list(range(global_rank // sp_size * sp_size, (global_rank // sp_size + 1) * sp_size))
            if members != expected_members:
                raise ValueError("joint Ulysses requires the upstream contiguous [dp, sp] mesh")
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
        ) not in ({"eager", "sdpa", "flash_attention_2"} if sp_size > 1 else {"eager", "sdpa"}):
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
            sequence_multiple=sp_size,
        )
        if sp_size > 1:
            from verl_speco.joint_ulysses import validate_ulysses_model

            validate_ulysses_model(engine.module, sp_size)
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
    requests, dummy = partition_joint_requests(requests, rank // sp_size, world // sp_size)
    model = engine.module
    dtype = getattr(engine, "_autocast_dtype", torch.float32)
    states = [(module, module.training) for module in model.modules()]
    try:
        with engine.eval_mode() if manage_engine else nullcontext():
            device_type = next(model.parameters()).device.type
            autocast = (
                nullcontext()
                if dtype == torch.float32
                else torch.autocast(device_type, dtype=dtype)
            )
            with autocast:
                if sp_size > 1:
                    from verl_speco.joint_ulysses import score_ulysses_joint_batch

                    result, metrics = score_ulysses_joint_batch(model, requests, sp_group, **options)
                else:
                    result, metrics = score_joint_batch(model, requests, **options)
            # Only one SP rank exports a tree. Other members finish the same
            # forward/collectives but avoid duplicate host feature transfers.
            nonleader = sp_size > 1 and torch.distributed.get_rank(sp_group) != 0
            if dummy or nonleader:
                result = ()
                metrics["logical_rows"] = metrics["independent_rows"] = 0
                metrics["model_forward_calls"] = 0
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
            return result, metrics
    finally:
        for module, training in states:
            module.training = training
