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

import hashlib
from contextlib import nullcontext

import torch

from verl_speco.joint_scoring import (
    JointScoringRequest,
    validate_joint_batch,
    score_joint_batch,
    validate_joint_model,
    parallel_microbatch_sizes,
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


def score_actor_joint(
    worker, requests, *, microbatch_sizes=None, microbatch_trajectories=None, **options
):
    """Validate all scheduled rounds before entering the resident FSDP actor."""
    sizes = tuple(microbatch_sizes) if microbatch_sizes is not None else None
    engine = getattr(getattr(worker, "actor", None), "engine", None)
    sp_size = int(getattr(engine, "ulysses_sequence_parallel_size", 1) or 1)
    sp_group = getattr(engine, "ulysses_parallel_group", None)
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    if sp_size > 1:
        from verl_speco.joint_ulysses import ULYSSES_ENV_FLAGS, resolve_ulysses_flag

        for name in ULYSSES_ENV_FLAGS:
            options[name] = resolve_ulysses_flag(name, options.get(name))
    if distributed:
        signature = (
            requests,
            sizes,
            microbatch_trajectories,
            sp_size,
            sorted(options.items()),
        )
        digest = hashlib.sha256(repr(signature).encode()).hexdigest()
        schedules = [None] * world
        torch.distributed.all_gather_object(schedules, digest)
        if len(set(schedules)) != 1:
            message = (
                "joint actor ranks must receive the same complete trajectory schedule"
                if sizes is None
                else "joint actor ranks received different microbatch schedules"
            )
            raise RuntimeError(message)

    batches = []
    ordered_indices = None
    error = None
    # Preflight every local batch before any rank enters an FSDP forward.
    try:
        rank = torch.distributed.get_rank() if distributed else 0
        _validate_actor_engine(
            worker, engine, sp_size, sp_group, rank, world, distributed, options
        )
        dp_size, dp_rank = world // sp_size, rank // sp_size
        nonleader = sp_size > 1 and torch.distributed.get_rank(sp_group) != 0
        if microbatch_trajectories is not None:
            if (
                sizes is not None
                or type(microbatch_trajectories) is not int
                or microbatch_trajectories < 1
            ):
                raise ValueError(
                    "joint scoring requires one positive microbatch trajectory limit"
                )
            requests, sizes, ordered_indices = _plan_actor_batches(
                requests,
                (microbatch_trajectories,),
                dp_size,
                sp_size,
                options,
                auto=True,
            )
        elif sizes is None:
            sizes = (len(requests),)
        else:
            if (
                not sizes
                or any(type(n) is not int or n <= 0 for n in sizes)
                or sum(sizes) != len(requests)
            ):
                raise ValueError(
                    "joint microbatch sizes must partition the complete request schedule"
                )
            requests, sizes, ordered_indices = _plan_actor_batches(
                requests, sizes, dp_size, sp_size, options
            )
        offset = 0
        for length in sizes:
            batch = requests[offset : offset + length]
            if len({request.sample_id for request in batch}) != len(batch):
                raise ValueError("sample IDs must be nonempty and unique")
            local_requests, dummy = partition_joint_requests(batch, dp_rank, dp_size)
            validate_joint_batch(
                local_requests,
                dtype=torch.float32,
                vocab_size=engine.module.config.vocab_size,
                max_position_embeddings=engine.module.config.max_position_embeddings,
                max_packed_tokens=options.get("max_packed_tokens", 8192),
                max_mask_elements=options.get("max_mask_elements", 16777216),
                sequence_multiple=sp_size,
            )
            indices = (
                ordered_indices[offset + dp_rank : offset + length : dp_size]
                if ordered_indices is not None
                else ()
            )
            batches.append((local_requests, dummy, indices))
            offset += length
    except Exception as exc:
        error = exc
    if distributed:
        errors = [None] * world
        torch.distributed.all_gather_object(errors, str(error) if error else None)
        if any(errors):
            raise RuntimeError(f"joint actor preflight failed: {errors}")
    elif error:
        raise error

    states = [(m, m.training) for m in engine.module.modules()]
    results, totals = [], {}
    result_indices = []
    try:
        with engine.eval_mode():
            for local_requests, dummy, indices in batches:
                scored, measured = _score_local_joint_batch(
                    engine,
                    local_requests,
                    sp_size,
                    sp_group,
                    dummy,
                    nonleader,
                    **options,
                )
                results.extend(scored)
                if scored:
                    result_indices.extend(indices)
                for key, value in measured.items():
                    if key == "sp_size":
                        if key in totals and totals[key] != value:
                            raise RuntimeError("joint microbatch SP size changed")
                        totals[key] = value
                    else:
                        totals[key] = totals.get(key, 0) + value
    finally:
        for module, training in states:
            module.training = training
    if ordered_indices is not None:
        results = [
            sample for _, sample in sorted(zip(result_indices, results, strict=True))
        ]
    return tuple(results), totals


def _plan_actor_batches(requests, sizes, dp_size, sp_size, options, *, auto=False):
    ordered_indices = None
    if sp_size > 1 and dp_size > 1 and options.get("use_causal_fast_path"):
        trunk_indices = []
        branch_indices = []
        for index, request in enumerate(requests):
            (branch_indices if request.branches else trunk_indices).append(index)
        if (
            trunk_indices
            and branch_indices
            and len({r.sample_id for r in requests}) == len(requests)
        ):
            # Group trunk-only work into the same DP rounds. Use the original
            # packing rule at the class boundary to avoid an extra dummy round.
            ordered_indices = trunk_indices + branch_indices
            requests = tuple(requests[i] for i in ordered_indices)
    return (
        requests,
        sizes
        if dp_size == 1 and not auto
        else parallel_microbatch_sizes(requests, sizes, dp_size, sp_size, options),
        ordered_indices,
    )


def _validate_actor_engine(
    worker, engine, sp_size, sp_group, rank, world, distributed, options
):
    if not getattr(worker, "_is_actor", False):
        raise ValueError("joint scoring must run on the RL actor worker")
    if engine is None or not callable(getattr(engine, "eval_mode", None)):
        raise ValueError("actor engine must expose module and eval_mode")
    if sp_size < 1 or world % sp_size:
        raise ValueError("joint actor SP size must divide the actor world size")
    if getattr(engine, "use_remove_padding", True) and sp_size == 1:
        raise NotImplementedError(
            "joint scoring requires model.use_remove_padding=false"
        )
    if getattr(engine, "use_ulysses_sp", False) and sp_size == 1:
        raise NotImplementedError("joint Ulysses requires an explicit SP size")
    if sp_size > 1:
        if (
            not distributed
            or sp_group is None
            or torch.distributed.get_world_size(sp_group) != sp_size
        ):
            raise ValueError(
                "joint Ulysses requires the actor engine's initialized SP group"
            )
        if not getattr(engine, "use_remove_padding", False):
            raise ValueError("upstream Ulysses requires use_remove_padding=true")
        members = torch.distributed.get_process_group_ranks(sp_group)
        expected_members = list(
            range(rank // sp_size * sp_size, (rank // sp_size + 1) * sp_size)
        )
        if members != expected_members:
            raise ValueError(
                "joint Ulysses requires the upstream contiguous [dp, sp] mesh"
            )
    model_config = getattr(engine, "model_config", None)
    if getattr(model_config, "use_fused_kernels", False):
        raise NotImplementedError("joint scoring requires use_fused_kernels=false")
    if getattr(model_config, "lora_rank", 0) or getattr(
        model_config, "lora_adapter_path", None
    ):
        raise NotImplementedError("joint scoring has not validated LoRA actor wrappers")
    validate_joint_model(
        engine.module.config, options.get("hidden_state_indices", ()), sp_size
    )
    if sp_size > 1:
        from verl_speco.joint_ulysses import validate_ulysses_model

        validate_ulysses_model(engine.module, sp_size)


def _score_local_joint_batch(
    engine, requests, sp_size, sp_group, dummy, nonleader, **options
):
    model = engine.module
    dtype = getattr(engine, "_autocast_dtype", torch.float32)
    device_type = next(model.parameters()).device.type
    autocast = (
        nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type, dtype=dtype)
    )
    with autocast:
        if sp_size > 1:
            from verl_speco.joint_ulysses import score_ulysses_joint_batch

            result, metrics = score_ulysses_joint_batch(
                model, requests, sp_group, **options
            )
        else:
            result, metrics = score_joint_batch(model, requests, **options)
    # Only one SP rank exports a tree. Other members finish the same
    # forward/collectives but avoid duplicate host feature transfers.
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
            if branch.target_logz is not None:
                branch.target_logz = branch.target_logz.cpu()
            branch.target_hidden = branch.target_hidden.cpu()
    metrics["dummy_rows"] = metrics["padded_rows"] if dummy else 0
    return result, metrics
