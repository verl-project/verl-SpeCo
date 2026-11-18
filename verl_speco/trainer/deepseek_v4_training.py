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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from offline_training/train_mtp.py.
import math
import os
from time import perf_counter

import torch
import torch.distributed as dist

from verl_speco.backends.vllm_ascend_options import (
    get_dspark_optimization,
    resolve_optimization_config,
)
from verl_speco.backends.vllm_ascend_parallel import is_main_process
from verl_speco.models.deepseek_v4.modeling_dspark import (
    build_combined_target_replay_loss,
    build_dspark_loss_weights,
    dspark_backward_sync_context,
    should_synchronize_dspark_backward,
)
from verl_speco.ops.deepseek_v4_runtime import get_vllm_ascend_training_diagnostics
from verl_speco.trainer.deepseek_v4_diagnostics import (
    capture_dspark_gradient_probe,
    dspark_parameter_audit_group,
)
from verl_speco.trainer.vllm_ascend_state import save_deepspeed_checkpoint


def synchronize_device_for_timing(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def dspark_runtime_trace(stage: str) -> None:
    """Emit rank-local DSpark runtime checkpoints for DP synchronization stalls."""
    if os.environ.get("SPECO_DSPARK_RUNTIME_TRACE", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[SpeCo DSpark runtime rank={rank}] {stage}", flush=True)


def optimizer_learning_rates(model_engine) -> list[float]:
    learning_rates = [float(value) for value in model_engine.get_lr()]
    if not learning_rates:
        raise RuntimeError("DeepSpeed optimizer returned no learning rates")
    if not all(math.isfinite(value) and value >= 0 for value in learning_rates):
        raise RuntimeError(f"Invalid optimizer learning rates: {learning_rates}")
    reference = learning_rates[0]
    if any(
        not math.isclose(value, reference, rel_tol=1e-12, abs_tol=0.0)
        for value in learning_rates[1:]
    ):
        raise RuntimeError(
            "DS4 optimizer parameter groups must use the same learning rate; "
            f"got {learning_rates}"
        )
    return learning_rates


def audit_dspark_first_backward(
    *,
    model_engine,
    device,
    logger,
    strict_runtime_diagnostics: bool,
    gradient_probe_state,
    gradient_probe_handles,
    tensor_parallel_group,
    log_parameter_audit_details: bool,
) -> None:
    """Validate the first synchronized DSpark backward before stepping."""

    diagnostics = get_vllm_ascend_training_diagnostics()
    if not dist.is_initialized() or dist.get_rank() == 0:
        cache_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key.startswith(
                (
                    "dsa_",
                    "dspark_device_alias",
                    "moe_",
                    "bf16_routed_moe_",
                    "bf16_qnorm_",
                )
            )
        }
        logger.info(
            "DSpark first-backward runtime diagnostics: %s",
            cache_diagnostics,
        )
    if diagnostics.get("dsa_accumulated_cache_scatter_updates", 0) <= 0:
        message = (
            "DSpark context/query cache writes were not coalesced by the "
            "generic DSA replay hook; DSpark precomputes context KV through "
            "its own runtime path"
        )
        if strict_runtime_diagnostics:
            raise RuntimeError(message)
        logger.warning("%s; continuing with gradient audit", message)

    reuse_host_metadata = bool(
        getattr(
            model_engine.module,
            "reuse_dsa_backward_host_metadata",
            False,
        )
    )
    if reuse_host_metadata:
        registrations = int(
            diagnostics.get("dsa_backward_host_metadata_registrations", 0)
        )
        hits = int(diagnostics.get("dsa_backward_host_metadata_hits", 0))
        fallbacks = int(diagnostics.get("dsa_backward_host_metadata_fallbacks", 0))
        if registrations <= 0 or hits <= 0 or fallbacks > 0:
            message = (
                "DSpark DSA host metadata reuse is enabled but inactive: "
                f"registrations={registrations}, hits={hits}, "
                f"fallbacks={fallbacks}"
            )
            if strict_runtime_diagnostics:
                raise RuntimeError(message)
            logger.warning("%s; continuing with device metadata fallback", message)

    grouped = {}
    audit_details = []
    failures = []
    sampled_zero_parameters = []
    for name, parameter in model_engine.module.named_parameters():
        if not parameter.requires_grad:
            continue
        state = gradient_probe_state.get(name)
        calls = 0 if state is None else int(state["calls"])
        checked_calls = 0 if state is None else int(state["checked_calls"])
        finite = bool(
            state is not None and state["finite"] is not None and state["finite"].item()
        )
        max_abs = (
            None
            if state is None or checked_calls == 0 or state["max_abs"] is None
            else float(state["max_abs"].item())
        )
        status = checked_calls > 0 and finite and max_abs is not None and max_abs > 0
        reason = None
        if checked_calls == 0:
            reason = "missing"
        elif not finite:
            reason = "nonfinite"
        elif max_abs == 0.0:
            reason = "zero"

        # markov_w1 is vocabulary sharded. A rank can legitimately receive no
        # selected token IDs, but at least one rank in the TP group must have a
        # finite nonzero gradient.
        if ".markov_head.markov_w1.weight" in name:
            tp_status = torch.tensor(
                int(status),
                dtype=torch.int32,
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(
                    tp_status,
                    op=dist.ReduceOp.MAX,
                    group=tensor_parallel_group,
                )
            if bool(tp_status.item()):
                status = True
                reason = None

        group_name = dspark_parameter_audit_group(name)
        group_state = grouped.setdefault(
            group_name,
            {
                "parameters": 0,
                "nonzero": 0,
                "missing": 0,
                "zero": 0,
                "nonfinite": 0,
            },
        )
        group_state["parameters"] += 1
        if status:
            group_state["nonzero"] += 1
        else:
            group_state[reason] += 1
            if reason == "zero":
                sampled_zero_parameters.append(name)
                if strict_runtime_diagnostics:
                    failures.append((name, reason))
            else:
                failures.append((name, reason))
        audit_details.append(
            {
                "name": name,
                "group": group_name,
                "hook_calls": calls,
                "checked_calls": checked_calls,
                "finite": finite,
                "max_abs": max_abs,
                "status": "nonzero" if status else reason,
            }
        )
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("DSpark first-backward gradient coverage: %s", grouped)
        if log_parameter_audit_details:
            logger.info(
                "DSpark first-backward parameter gradients: %s",
                audit_details,
            )
    if sampled_zero_parameters and not strict_runtime_diagnostics:
        logger.warning(
            "DSpark first-backward audit observed a zero first local "
            "gradient contribution for %d parameters on rank %d. This is "
            "not treated as a disconnected graph because later anchor "
            "chunks can contribute nonzero gradients: %s",
            len(sampled_zero_parameters),
            dist.get_rank() if dist.is_initialized() else 0,
            sampled_zero_parameters,
        )
    local_failure = torch.tensor(
        int(bool(failures)),
        dtype=torch.int32,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(
            local_failure,
            op=dist.ReduceOp.MAX,
        )
    if bool(local_failure.item()):
        moe_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key.startswith(("moe_", "bf16_routed_moe_"))
        }
        raise RuntimeError(
            "DSpark first-backward gradient audit failed: "
            f"rank={dist.get_rank() if dist.is_initialized() else 0}, "
            f"invalid_parameters={failures}, "
            f"moe_diagnostics={moe_diagnostics}"
        )
    for handle in gradient_probe_handles:
        handle.remove()
    gradient_probe_handles.clear()


def run_dspark_training(
    *,
    model_engine,
    target_lm_head,
    dataloader,
    sampler,
    cfg,
    logger,
    tracker,
    device,
    parallel_topology,
    checkpoint_dir,
    initial_global_step=0,
    initial_global_examples=0,
    save_checkpoints=True,
    save_full_model_per_rank=True,
    save_individual_model_state=True,
) -> dict[str, int | float | bool]:
    """Train the imported serving-time DSpark block drafter."""

    optimizations = resolve_optimization_config(cfg, "dspark")
    gradient_accumulation_steps = int(cfg.deepspeed.gradient_accumulation_steps)
    num_anchors = int(getattr(cfg.train, "dspark_num_anchors", 1))
    configured_anchor_spacing = getattr(
        cfg.train,
        "dspark_anchor_min_spacing",
        None,
    )
    ce_weight = float(getattr(cfg.train, "dspark_ce_weight", 0.1))
    l1_weight = float(getattr(cfg.train, "dspark_l1_weight", 0.9))
    kl_weight = float(getattr(cfg.train, "dspark_kl_weight", 0.0))
    confidence_weight = float(getattr(cfg.train, "dspark_confidence_weight", 0.0))
    hidden_state_noise_std = float(
        getattr(cfg.train, "dspark_hidden_state_noise_std", 0.0)
    )
    logging_interval = int(cfg.logging.interval)
    if logging_interval <= 0:
        raise ValueError("logging.interval must be positive")
    extended_step_metrics = bool(getattr(cfg.logging, "extended_step_metrics", False))
    log_parameter_audit_details = bool(
        getattr(cfg.logging, "parameter_audit_details", False)
    )
    strict_runtime_diagnostics = bool(
        getattr(optimizations, "strict_runtime_diagnostics", False)
    )
    decay_gamma = getattr(cfg.train, "dspark_loss_decay_gamma", None)
    anchor_micro_batch_size = int(
        get_dspark_optimization(cfg, "anchor_micro_batch_size", 0)
    )
    prefix_accuracy_interval = int(
        getattr(cfg.logging, "prefix_accuracy_interval", logging_interval)
    )
    distribution_metrics_interval = int(
        getattr(cfg.logging, "distribution_metrics_interval", 0)
    )
    profile_step_phases = bool(getattr(cfg.logging, "profile_step_phases", False))
    phase_profile_interval = int(getattr(cfg.logging, "phase_profile_interval", 10))
    vocab_parallel_loss = bool(
        get_dspark_optimization(cfg, "vocab_parallel_loss", False)
    )
    reuse_combined_target_hidden_states = bool(
        get_dspark_optimization(
            cfg,
            "reuse_combined_target_hidden_states",
            False,
        )
    )
    requested_gradient_communication_coalescing = bool(
        get_dspark_optimization(
            cfg,
            "coalesce_gradient_communication",
            False,
        )
    )
    zero_config = getattr(cfg.deepspeed, "zero_optimization", None)
    zero_stage = int(getattr(zero_config, "stage", 0) or 0)
    if requested_gradient_communication_coalescing and zero_stage >= 2:
        raise ValueError(
            "optimizations.dspark.coalesce_gradient_communication uses "
            "DeepSpeed "
            "no_sync and therefore requires ZeRO stage 0 or 1"
        )
    coalesce_gradient_communication = (
        requested_gradient_communication_coalescing
        and callable(getattr(model_engine, "no_sync", None))
    )
    if (
        requested_gradient_communication_coalescing
        and not coalesce_gradient_communication
        and is_main_process()
    ):
        logger.warning(
            "DSpark gradient communication coalescing requested, but this "
            "DeepSpeed engine does not expose no_sync(); using legacy "
            "per-backward gradient reduction"
        )
    if is_main_process():
        logger.info(
            "DSpark gradient communication coalescing: requested=%s, "
            "active=%s, zero_stage=%d",
            requested_gradient_communication_coalescing,
            coalesce_gradient_communication,
            zero_stage,
        )
        logger.info(
            "DSpark hybrid RoPE: vLLM-forward/Torch-backward active=%s",
            bool(
                getattr(
                    model_engine.module,
                    "using_training_safe_rope",
                    False,
                )
            ),
        )
        logger.info(
            "DSpark DSA query replay: vLLM-forward/Torch-backward active=%s",
            bool(
                getattr(
                    model_engine.module,
                    "using_training_dsa_query_replay",
                    False,
                )
            ),
        )
        logger.info(
            "DSpark trainable internal router: model-level patch active=%s",
            bool(
                getattr(
                    model_engine.module,
                    "using_model_internal_router_training_forward",
                    False,
                )
            ),
        )
        logger.info(
            "DSpark TP-local vocabulary loss: active=%s",
            vocab_parallel_loss,
        )
        logger.info(
            "DSpark prefix accuracy interval: %d optimizer steps (0 disables)",
            prefix_accuracy_interval,
        )
        logger.info(
            "DSpark teacher-forced distribution metrics interval: %d "
            "optimizer steps (0 disables)",
            distribution_metrics_interval,
        )
        logger.info(
            "DSpark phase profiling: active=%s, interval=%d "
            "(the first step is always sampled)",
            profile_step_phases,
            phase_profile_interval,
        )
    initial_global_step = int(initial_global_step)
    if initial_global_step < 0:
        raise ValueError("initial_global_step must be non-negative")
    initial_global_examples = int(initial_global_examples)
    if initial_global_examples < 0:
        raise ValueError("initial_global_examples must be non-negative")
    max_optimizer_steps = getattr(cfg.train, "max_optimizer_steps", None)
    if max_optimizer_steps is not None:
        max_optimizer_steps = int(max_optimizer_steps)
        if max_optimizer_steps <= 0:
            raise ValueError("train.max_optimizer_steps must be positive")
    optimizer_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_optimizer_steps = int(cfg.train.epochs) * optimizer_steps_per_epoch
    if max_optimizer_steps is not None:
        total_optimizer_steps = min(
            total_optimizer_steps,
            max_optimizer_steps,
        )
    total_optimizer_steps += initial_global_step
    stop_global_step = (
        None
        if max_optimizer_steps is None
        else initial_global_step + max_optimizer_steps
    )
    if num_anchors <= 0:
        raise ValueError("train.dspark_num_anchors must be positive")
    if anchor_micro_batch_size < 0:
        raise ValueError(
            "optimizations.dspark.anchor_micro_batch_size must be non-negative"
        )
    if prefix_accuracy_interval < 0:
        raise ValueError("logging.prefix_accuracy_interval must be non-negative")
    if distribution_metrics_interval < 0:
        raise ValueError("logging.distribution_metrics_interval must be non-negative")
    if profile_step_phases and phase_profile_interval <= 0:
        raise ValueError(
            "logging.phase_profile_interval must be positive "
            "when phase profiling is enabled"
        )
    fp16_config = getattr(cfg.deepspeed, "fp16", None)
    if (
        reuse_combined_target_hidden_states
        and fp16_config is not None
        and bool(getattr(fp16_config, "enabled", False))
    ):
        raise ValueError(
            "optimizations.dspark.reuse_combined_target_hidden_states "
            "currently "
            "requires BF16 or FP32 because its gradient replay must not "
            "apply dynamic loss scaling twice"
        )
    if configured_anchor_spacing is not None and int(configured_anchor_spacing) <= 0:
        raise ValueError("train.dspark_anchor_min_spacing must be positive")
    if ce_weight < 0 or l1_weight < 0 or kl_weight < 0:
        raise ValueError(
            "train.dspark_ce_weight, train.dspark_l1_weight, and "
            "train.dspark_kl_weight must be non-negative"
        )
    if confidence_weight < 0:
        raise ValueError("train.dspark_confidence_weight must be non-negative")
    if hidden_state_noise_std < 0:
        raise ValueError("train.dspark_hidden_state_noise_std must be non-negative")
    if vocab_parallel_loss and not callable(
        getattr(target_lm_head, "local_logits", None)
    ):
        raise ValueError(
            "optimizations.dspark.vocab_parallel_loss requires "
            "optimizations.common.parallel_lm_head=true"
        )
    global_step = initial_global_step
    global_examples = initial_global_examples
    observed_update = False
    observed_nonzero_optimizer_lr = False
    first_gradient_audit_complete = False
    gradient_probe_control = {
        "enabled": False,
        "batch_captured": False,
    }
    gradient_probe_state = {}
    gradient_probe_handles = []
    for name, parameter in model_engine.module.named_parameters():
        if not parameter.requires_grad:
            continue
        state = {
            "calls": 0,
            "checked_calls": 0,
            "captured": False,
            "finite": None,
            "max_abs": None,
            "max_index": None,
        }
        gradient_probe_state[name] = state

        def record_gradient(gradient, *, probe_state=state):
            probe_state["calls"] += 1
            if not gradient_probe_control["enabled"] or probe_state["captured"]:
                return gradient
            finite, max_abs, max_index = capture_dspark_gradient_probe(gradient)
            probe_state["checked_calls"] += 1
            probe_state["captured"] = True
            probe_state["finite"] = finite
            probe_state["max_abs"] = max_abs
            probe_state["max_index"] = max_index
            return gradient

        gradient_probe_handles.append(parameter.register_hook(record_gradient))
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model_engine.module.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named_parameters:
        raise RuntimeError("DSpark model does not expose any trainable parameters")
    update_probe_name, update_probe_parameter = next(
        (
            (name, parameter)
            for name, parameter in trainable_named_parameters
            if ".main_proj.weight" in name
        ),
        next(
            (
                (name, parameter)
                for name, parameter in trainable_named_parameters
                if parameter.dtype == torch.float32
            ),
            trainable_named_parameters[0],
        ),
    )
    # A bounded probe validates that DeepSpeed's optimizer writes a parameter
    # at least once without cloning a large TP shard. A non-zero learning rate
    # does not guarantee a representable per-step update for BF16 parameters.
    update_probe_values = min(update_probe_parameter.numel(), 4096)
    update_probe_snapshot = (
        update_probe_parameter.detach().reshape(-1)[:update_probe_values].clone()
    )

    for epoch in range(int(cfg.train.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(dataloader)
        batch_index = 0
        batches_per_epoch = len(dataloader)
        while batch_index < batches_per_epoch:
            step_started = perf_counter()
            will_log_step = (global_step + 1) % logging_interval == 0
            profile_this_step = bool(
                profile_step_phases
                and is_main_process()
                and will_log_step
                and (
                    global_step + 1 == logging_interval
                    or (global_step + 1) % phase_profile_interval == 0
                )
            )
            model_engine.module.set_phase_profiling(profile_this_step)
            window_size = min(
                gradient_accumulation_steps,
                batches_per_epoch - batch_index,
            )
            accumulation_scale = gradient_accumulation_steps / window_size
            data_wait_seconds = 0.0
            step_metric_totals = torch.zeros(
                6,
                dtype=torch.float32,
                device=device,
            )
            block_size = int(model_engine.module.runtime_contract.block_size)
            step_prefix_correct = torch.zeros(
                block_size,
                dtype=torch.float32,
                device=device,
            )
            step_prefix_total = torch.zeros_like(step_prefix_correct)
            step_distribution_position_counts = torch.zeros_like(step_prefix_correct)
            step_distribution_l1_position_sums = torch.zeros_like(step_prefix_correct)
            step_top1_correct_counts = torch.zeros_like(step_prefix_correct)
            compute_prefix_accuracy = bool(
                prefix_accuracy_interval > 0
                and will_log_step
                and (
                    global_step + 1 == logging_interval
                    or (global_step + 1) % prefix_accuracy_interval == 0
                )
            )
            compute_distribution_metrics = bool(
                distribution_metrics_interval > 0
                and will_log_step
                and (
                    global_step + 1 == logging_interval
                    or (global_step + 1) % distribution_metrics_interval == 0
                )
            )
            step_requested_anchors = 0
            step_active_anchors = 0
            step_underfilled_anchor_batches = 0
            step_anchor_chunks = 0
            step_padding_anchor_chunks = 0
            step_combined_reuse_batches = 0
            step_combined_reuse_chunks = 0
            step_deferred_gradient_syncs = 0
            step_lazy_mmap_cache_hits = 0
            step_lazy_mmap_cache_misses = 0
            step_lazy_mmap_cache_evictions = 0
            step_optimizer_lr = None
            step_optimizer_probe_delta = 0.0
            for offset in range(window_size):
                # Fetch incrementally so the DataLoader can prepare later
                # accumulation micro-batches while this rank executes DSpark.
                # Fetching the entire window up front drains the small
                # prefetch queue, then leaves its worker idle for the full
                # device step.
                data_wait_started = perf_counter()
                dspark_runtime_trace(f"before loading batch={batch_index}")
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "DataLoader ended before its reported length while "
                        "building a DSpark accumulation window"
                    ) from error
                data_wait_seconds += perf_counter() - data_wait_started
                dspark_runtime_trace(f"after loading batch={batch_index}")
                step_lazy_mmap_cache_hits += int(batch.get("lazy_mmap_cache_hits", 0))
                step_lazy_mmap_cache_misses += int(
                    batch.get("lazy_mmap_cache_misses", 0)
                )
                step_lazy_mmap_cache_evictions += int(
                    batch.get("lazy_mmap_cache_evictions", 0)
                )
                is_boundary = offset == window_size - 1
                with model_engine.module.profile_phase("batch_to_device_and_noise"):
                    input_ids = batch["input_ids"].to(device)
                    raw_mask = batch["raw_mask"].to(device)
                    target_hidden_states = batch["hidden_states"].to(device)
                    if hidden_state_noise_std > 0:
                        target_hidden_states = target_hidden_states + (
                            torch.randn_like(target_hidden_states)
                            * hidden_state_noise_std
                        )
                    target_head_hidden_states = batch["target_head_hidden_states"].to(
                        device
                    )
                    positions = batch["positions"].to(device)
                with model_engine.module.profile_phase("prepare_inputs"):
                    targets, runtime_inputs = (
                        model_engine.module.prepare_training_inputs(
                            input_ids=input_ids,
                            raw_mask=raw_mask,
                            positions=positions,
                            num_anchors=num_anchors,
                        )
                    )
                    active_anchors = torch.nonzero(
                        targets.block_keep_mask,
                        as_tuple=False,
                    )
                expected_requests = int(input_ids.shape[0]) * num_anchors
                actual_requests = int(active_anchors.shape[0])
                if actual_requests <= 0:
                    raise RuntimeError(
                        "DSpark training batch has no valid anchors. Provide "
                        "supervised tokens or reduce train.dspark_block_size."
                    )
                with model_engine.module.profile_phase("host_metadata_prefetch"):
                    active_batch_indices = active_anchors[:, 0]
                    active_anchor_indices = active_anchors[:, 1]
                    active_host_metadata = (
                        torch.cat(
                            (
                                runtime_inputs.context_lengths[
                                    active_batch_indices,
                                    active_anchor_indices,
                                ]
                                .to(dtype=runtime_inputs.query_positions.dtype)
                                .unsqueeze(1),
                                runtime_inputs.query_positions[
                                    active_batch_indices,
                                    active_anchor_indices,
                                ],
                            ),
                            dim=1,
                        )
                        .detach()
                        .to(device="cpu")
                    )
                    active_context_lengths_cpu = active_host_metadata[:, 0].to(
                        dtype=torch.int32
                    )
                    active_query_positions_cpu = active_host_metadata[:, 1:]
                step_requested_anchors += expected_requests
                step_active_anchors += actual_requests
                step_underfilled_anchor_batches += int(
                    actual_requests < expected_requests
                )
                local_weight = build_dspark_loss_weights(
                    targets.eval_mask[active_anchors[:, 0], active_anchors[:, 1]],
                    loss_decay_gamma=decay_gamma,
                ).sum()
                global_weight = local_weight.detach().float().clone()
                if dist.is_initialized():
                    dist.all_reduce(
                        global_weight,
                        op=dist.ReduceOp.SUM,
                        group=parallel_topology.data_parallel_group,
                    )
                dp_scale = (
                    local_weight.detach().float()
                    * parallel_topology.data_parallel_size
                    / global_weight.clamp_min(1.0)
                )
                local_chunk_count = math.ceil(
                    actual_requests / (anchor_micro_batch_size or actual_requests)
                )
                dspark_runtime_trace(
                    "before DP chunk-count sync "
                    f"batch={batch_index} anchors={actual_requests} "
                    f"chunks={local_chunk_count}"
                )
                # DP=1 already knows its chunk count on the host. Avoid an
                # otherwise redundant NPU scalar allocation and blocking item().
                synchronized_chunk_count = local_chunk_count
                if dist.is_initialized() and parallel_topology.data_parallel_size > 1:
                    synchronized_chunk_count_tensor = torch.tensor(
                        local_chunk_count,
                        dtype=torch.int32,
                        device=device,
                    )
                    dist.all_reduce(
                        synchronized_chunk_count_tensor,
                        op=dist.ReduceOp.MAX,
                        group=parallel_topology.data_parallel_group,
                    )
                    synchronized_chunk_count = int(
                        synchronized_chunk_count_tensor.item()
                    )
                dspark_runtime_trace(
                    "after DP chunk-count sync "
                    f"batch={batch_index} synchronized_chunks="
                    f"{synchronized_chunk_count}"
                )
                chunk_size = anchor_micro_batch_size or actual_requests
                step_anchor_chunks += local_chunk_count
                step_padding_anchor_chunks += (
                    synchronized_chunk_count - local_chunk_count
                )
                reuse_combined_for_batch = (
                    reuse_combined_target_hidden_states and synchronized_chunk_count > 1
                )
                combined_target_graph = None
                reusable_combined_target_hidden_states = None
                replay_combined_gradients = False
                capture_gradient_probe_batch = not gradient_probe_control[
                    "batch_captured"
                ]
                if capture_gradient_probe_batch:
                    gradient_probe_control["enabled"] = True
                if reuse_combined_for_batch:
                    with model_engine.module.profile_phase("target_combine"):
                        combined_target_graph = (
                            model_engine.module.combine_target_hidden_states(
                                target_hidden_states.reshape(
                                    -1,
                                    target_hidden_states.shape[-1],
                                )
                            ).reshape(
                                target_hidden_states.shape[0],
                                target_hidden_states.shape[1],
                                -1,
                            )
                        )
                    replay_combined_gradients = bool(
                        combined_target_graph.requires_grad
                    )
                    reusable_combined_target_hidden_states = (
                        combined_target_graph.detach().requires_grad_(True)
                        if replay_combined_gradients
                        else combined_target_graph.detach()
                    )
                    step_combined_reuse_batches += 1
                    step_combined_reuse_chunks += synchronized_chunk_count
                for chunk_index in range(synchronized_chunk_count):
                    is_last_chunk = chunk_index == synchronized_chunk_count - 1
                    model_engine.set_gradient_accumulation_boundary(
                        is_boundary and is_last_chunk and not replay_combined_gradients
                    )
                    chunk_start = chunk_index * chunk_size
                    chunk_end = min(chunk_start + chunk_size, actual_requests)
                    is_padding_chunk = chunk_start >= actual_requests
                    chunk_anchors = (
                        active_anchors[:1]
                        if is_padding_chunk
                        else active_anchors[chunk_start:chunk_end]
                    )
                    chunk_context_lengths_cpu = (
                        active_context_lengths_cpu[:1]
                        if is_padding_chunk
                        else active_context_lengths_cpu[chunk_start:chunk_end]
                    )
                    chunk_query_positions_cpu = (
                        active_query_positions_cpu[:1]
                        if is_padding_chunk
                        else active_query_positions_cpu[chunk_start:chunk_end]
                    )
                    # vLLM synchronizes DP metadata inside its runtime forward.
                    # Padding replicas must therefore run a real request, not
                    # only a zero backward, so every DP rank enters the same
                    # sequence of runtime collectives.
                    detailed_runtime_trace = (
                        os.environ.get("SPECO_DSPARK_RUNTIME_TRACE_DETAIL", "0") == "1"
                    )
                    trace_runtime_forward = detailed_runtime_trace or chunk_index == 0
                    if trace_runtime_forward:
                        dspark_runtime_trace(
                            "before vLLM forward "
                            f"batch={batch_index} chunk={chunk_index}/"
                            f"{synchronized_chunk_count} padding={is_padding_chunk} "
                            f"anchors={int(chunk_anchors.shape[0])}"
                        )
                    output = model_engine(
                        input_ids=input_ids,
                        raw_mask=raw_mask,
                        target_hidden_states=target_hidden_states,
                        target_head_hidden_states=target_head_hidden_states,
                        positions=positions,
                        target_lm_head=target_lm_head,
                        num_anchors=num_anchors,
                        prepared_targets=targets,
                        prepared_runtime_inputs=runtime_inputs,
                        active_anchors=chunk_anchors,
                        active_context_lengths_cpu=(chunk_context_lengths_cpu),
                        active_query_positions_cpu=(chunk_query_positions_cpu),
                        combined_target_hidden_states=(
                            reusable_combined_target_hidden_states
                        ),
                        cross_entropy_weight=ce_weight,
                        distribution_l1_weight=l1_weight,
                        distribution_kl_weight=kl_weight,
                        confidence_weight=confidence_weight,
                        loss_decay_gamma=decay_gamma,
                        compute_prefix_accuracy=compute_prefix_accuracy,
                        compute_distribution_metrics=(compute_distribution_metrics),
                        vocab_parallel_loss=vocab_parallel_loss,
                        validate_vocab_parallel_projection=(strict_runtime_diagnostics),
                    )
                    if trace_runtime_forward:
                        dspark_runtime_trace(
                            "after vLLM forward "
                            f"batch={batch_index} chunk={chunk_index}/"
                            f"{synchronized_chunk_count}"
                        )
                    if not is_padding_chunk:
                        chunk_weight = output.loss.valid_weight.detach().float()
                        chunk_scale = chunk_weight / local_weight.clamp_min(1.0)
                        synchronize_backward = should_synchronize_dspark_backward(
                            is_accumulation_boundary=is_boundary,
                            is_last_chunk=is_last_chunk,
                            replay_combined_gradients=replay_combined_gradients,
                        )
                        with dspark_backward_sync_context(
                            model_engine,
                            synchronize=synchronize_backward,
                            coalesce_gradient_communication=(
                                coalesce_gradient_communication
                            ),
                        ):
                            backward_phase = (
                                "backward_sync"
                                if synchronize_backward
                                else "backward_local"
                            )
                            with model_engine.module.profile_phase(backward_phase):
                                model_engine.backward(
                                    output.loss.total
                                    * accumulation_scale
                                    * dp_scale
                                    * chunk_scale
                                )
                        step_deferred_gradient_syncs += int(
                            coalesce_gradient_communication and not synchronize_backward
                        )
                        if trace_runtime_forward:
                            dspark_runtime_trace(
                                "after backward "
                                f"batch={batch_index} chunk={chunk_index}/"
                                f"{synchronized_chunk_count}"
                            )
                        # Mirror the exact objective sent through backward.
                        # DeepSpeed averages gradients across DP, so each
                        # rank's metric contribution carries the same dp_scale
                        # before the global SUM below is divided by DP size.
                        metric_scale = chunk_scale.detach() * dp_scale.detach()
                        step_metric_totals.add_(
                            torch.stack(
                                (
                                    output.loss.total.detach() * metric_scale,
                                    output.loss.cross_entropy.detach() * metric_scale,
                                    output.loss.distribution_l1.detach() * metric_scale,
                                    output.loss.distribution_kl.detach() * metric_scale,
                                    output.loss.confidence.detach() * metric_scale,
                                    output.packed.eval_mask.sum().to(torch.float32),
                                )
                            )
                        )
                        if compute_prefix_accuracy:
                            if (
                                output.prefix_correct_counts is None
                                or output.prefix_total_counts is None
                            ):
                                raise RuntimeError(
                                    "DSpark prefix accuracy was requested but "
                                    "the model did not return prefix counts"
                                )
                            step_prefix_correct.add_(output.prefix_correct_counts)
                            step_prefix_total.add_(output.prefix_total_counts)
                        if compute_distribution_metrics:
                            distribution_fields = (
                                output.loss.distribution_position_counts,
                                output.loss.distribution_l1_position_sums,
                                output.loss.top1_correct_counts,
                            )
                            if any(value is None for value in distribution_fields):
                                raise RuntimeError(
                                    "DSpark distribution metrics were requested "
                                    "but the model did not return position stats"
                                )
                            step_distribution_position_counts.add_(
                                distribution_fields[0]
                            )
                            step_distribution_l1_position_sums.add_(
                                distribution_fields[1]
                            )
                            step_top1_correct_counts.add_(distribution_fields[2])
                    else:
                        synchronize_backward = should_synchronize_dspark_backward(
                            is_accumulation_boundary=is_boundary,
                            is_last_chunk=is_last_chunk,
                            replay_combined_gradients=replay_combined_gradients,
                        )
                        with dspark_backward_sync_context(
                            model_engine,
                            synchronize=synchronize_backward,
                            coalesce_gradient_communication=(
                                coalesce_gradient_communication
                            ),
                        ):
                            backward_phase = (
                                "backward_sync"
                                if synchronize_backward
                                else "backward_local"
                            )
                            with model_engine.module.profile_phase(backward_phase):
                                model_engine.backward(output.loss.total * 0.0)
                        step_deferred_gradient_syncs += int(
                            coalesce_gradient_communication and not synchronize_backward
                        )
                        if trace_runtime_forward:
                            dspark_runtime_trace(
                                "after padding backward "
                                f"batch={batch_index} chunk={chunk_index}/"
                                f"{synchronized_chunk_count}"
                            )

                if replay_combined_gradients:
                    model_engine.set_gradient_accumulation_boundary(is_boundary)
                    replay_loss = build_combined_target_replay_loss(
                        combined_target_graph,
                        reusable_combined_target_hidden_states,
                        gradient_accumulation_steps=(gradient_accumulation_steps),
                    )
                    with dspark_backward_sync_context(
                        model_engine,
                        synchronize=is_boundary,
                        coalesce_gradient_communication=(
                            coalesce_gradient_communication
                        ),
                    ):
                        replay_backward_phase = (
                            "combined_replay_backward_sync"
                            if is_boundary
                            else "combined_replay_backward_local"
                        )
                        with model_engine.module.profile_phase(replay_backward_phase):
                            model_engine.backward(replay_loss)
                    step_deferred_gradient_syncs += int(
                        coalesce_gradient_communication and not is_boundary
                    )
                    dspark_runtime_trace(
                        "after combined hidden-state gradient replay "
                        f"batch={batch_index}"
                    )
                    del replay_loss
                if reuse_combined_for_batch:
                    del combined_target_graph
                    del reusable_combined_target_hidden_states
                if capture_gradient_probe_batch:
                    gradient_probe_control["enabled"] = False
                    gradient_probe_control["batch_captured"] = True

                if is_boundary and not first_gradient_audit_complete:
                    audit_dspark_first_backward(
                        model_engine=model_engine,
                        device=device,
                        logger=logger,
                        strict_runtime_diagnostics=strict_runtime_diagnostics,
                        gradient_probe_state=gradient_probe_state,
                        gradient_probe_handles=gradient_probe_handles,
                        tensor_parallel_group=(parallel_topology.tensor_parallel_group),
                        log_parameter_audit_details=(log_parameter_audit_details),
                    )
                    first_gradient_audit_complete = True
                if is_boundary:
                    step_optimizer_lr = optimizer_learning_rates(model_engine)[0]
                    observed_nonzero_optimizer_lr = (
                        observed_nonzero_optimizer_lr or step_optimizer_lr > 0.0
                    )
                    should_read_update_probe = strict_runtime_diagnostics or (
                        step_optimizer_lr > 0.0 and not observed_update
                    )
                    if should_read_update_probe:
                        update_probe_before_step = (
                            update_probe_parameter.detach()
                            .reshape(-1)[:update_probe_values]
                            .clone()
                        )
                if is_boundary:
                    with model_engine.module.profile_phase("optimizer_step"):
                        model_engine.step()
                else:
                    model_engine.step()
                if is_boundary:
                    from verl_speco.ops.ascend_weight_cache import (
                        refresh_fp32_weight_caches,
                    )

                    refresh_fp32_weight_caches(model_engine.module)
                if is_boundary and should_read_update_probe:
                    update_probe_delta = float(
                        (
                            update_probe_parameter.detach().reshape(-1)[
                                :update_probe_values
                            ]
                            - update_probe_before_step
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    if not math.isfinite(update_probe_delta):
                        raise RuntimeError(
                            "DSpark optimizer update probe became non-finite: "
                            f"{update_probe_name}"
                        )
                    observed_update = observed_update or update_probe_delta > 0.0
                    step_optimizer_probe_delta = update_probe_delta
                global_examples += int(input_ids.shape[0])
                batch_index += 1

            global_step += 1
            if global_step % logging_interval == 0:
                # NPU work is asynchronous. Synchronize only at a logging
                # boundary so the reported wall time matches log timestamps.
                synchronize_device_for_timing(device)
                phase_profile_summary = model_engine.module.consume_phase_profile()
                elapsed = perf_counter() - step_started
                data_wait_elapsed = data_wait_seconds
                non_data_elapsed = max(elapsed - data_wait_elapsed, 0.0)
                global_loss_totals = step_metric_totals[:5].detach().clone()
                if dist.is_initialized() and parallel_topology.data_parallel_size > 1:
                    dist.all_reduce(
                        global_loss_totals,
                        op=dist.ReduceOp.SUM,
                        group=parallel_topology.data_parallel_group,
                    )
                loss_normalizer = float(
                    window_size * parallel_topology.data_parallel_size
                )
                (
                    step_loss,
                    step_cross_entropy,
                    step_distribution_l1,
                    step_distribution_kl,
                    step_confidence,
                ) = (global_loss_totals / loss_normalizer).to("cpu").tolist()
                step_valid_slots_value = float(step_metric_totals[5].detach().item())
                step_valid_slots = round(step_valid_slots_value)
                mean_loss = step_loss
                prefix_accuracy: list[float] = []
                if compute_prefix_accuracy:
                    prefix_values = torch.cat((step_prefix_correct, step_prefix_total))
                    if dist.is_initialized():
                        dist.all_reduce(
                            prefix_values,
                            op=dist.ReduceOp.SUM,
                            group=parallel_topology.data_parallel_group,
                        )
                    global_prefix_correct = prefix_values[:block_size]
                    global_prefix_total = prefix_values[block_size:]
                    prefix_accuracy = (
                        torch.where(
                            global_prefix_total > 0,
                            global_prefix_correct / global_prefix_total,
                            torch.zeros_like(global_prefix_correct),
                        )
                        .to("cpu")
                        .tolist()
                    )
                    prefix_accuracy_text = ", ".join(
                        f"t{position}={accuracy:.6f}"
                        for position, accuracy in enumerate(
                            prefix_accuracy,
                            start=1,
                        )
                    )
                else:
                    prefix_accuracy_text = "not_computed"
                distribution_l1_by_position: list[float] = []
                top1_agreement_by_position: list[float] = []
                expected_acceptance_by_position: list[float] = []
                expected_acceptance_overall = 0.0
                distribution_l1_text = "not_computed"
                top1_agreement_text = "not_computed"
                expected_acceptance_text = "not_computed"
                if compute_distribution_metrics:
                    distribution_values = torch.cat(
                        (
                            step_distribution_position_counts,
                            step_distribution_l1_position_sums,
                            step_top1_correct_counts,
                        )
                    )
                    if dist.is_initialized():
                        dist.all_reduce(
                            distribution_values,
                            op=dist.ReduceOp.SUM,
                            group=parallel_topology.data_parallel_group,
                        )
                    position_counts = distribution_values[:block_size]
                    l1_position_sums = distribution_values[block_size : 2 * block_size]
                    top1_correct_counts = distribution_values[2 * block_size :]
                    distribution_l1_tensor = torch.where(
                        position_counts > 0,
                        l1_position_sums / position_counts,
                        torch.zeros_like(l1_position_sums),
                    )
                    top1_agreement_tensor = torch.where(
                        position_counts > 0,
                        top1_correct_counts / position_counts,
                        torch.zeros_like(top1_correct_counts),
                    )
                    expected_acceptance_tensor = (
                        1.0 - 0.5 * distribution_l1_tensor
                    ).clamp(0.0, 1.0)
                    total_distribution_count = position_counts.sum()
                    if total_distribution_count.item() > 0:
                        expected_acceptance_overall = float(
                            (
                                1.0
                                - 0.5
                                * l1_position_sums.sum()
                                / total_distribution_count
                            )
                            .clamp(0.0, 1.0)
                            .item()
                        )
                    distribution_l1_by_position = distribution_l1_tensor.to(
                        "cpu"
                    ).tolist()
                    top1_agreement_by_position = top1_agreement_tensor.to(
                        "cpu"
                    ).tolist()
                    expected_acceptance_by_position = expected_acceptance_tensor.to(
                        "cpu"
                    ).tolist()
                    distribution_l1_text = ", ".join(
                        f"t{position}={value:.6f}"
                        for position, value in enumerate(
                            distribution_l1_by_position,
                            start=1,
                        )
                    )
                    top1_agreement_text = ", ".join(
                        f"t{position}={value:.6f}"
                        for position, value in enumerate(
                            top1_agreement_by_position,
                            start=1,
                        )
                    )
                    expected_acceptance_text = ", ".join(
                        f"t{position}={value:.6f}"
                        for position, value in enumerate(
                            expected_acceptance_by_position,
                            start=1,
                        )
                    )
                if is_main_process():
                    step_log_format = (
                        "DSpark train step %d/%d | epoch=%d | loss=%.6f | "
                        "ce=%.6f | l1=%.6f | kl=%.6f | confidence=%.6f | "
                        "valid_block_tokens=%d | anchors=%d/%d | "
                        "anchor_chunks=%d | padding_chunks=%d | "
                        "combined_reuse=%d batches/%d chunks | "
                        "deferred_grad_syncs=%d | "
                        "mmap_cache=%d hits/%d misses/%d evictions | "
                        "underfilled_micro_batches=%d | micro_batches=%d | "
                        "step_time=%.3fs | lr=%.3e"
                    )
                    step_log_args = (
                        global_step,
                        total_optimizer_steps,
                        epoch + 1,
                        mean_loss,
                        step_cross_entropy,
                        step_distribution_l1,
                        step_distribution_kl,
                        step_confidence,
                        step_valid_slots,
                        step_active_anchors,
                        step_requested_anchors,
                        step_anchor_chunks,
                        step_padding_anchor_chunks,
                        step_combined_reuse_batches,
                        step_combined_reuse_chunks,
                        step_deferred_gradient_syncs,
                        step_lazy_mmap_cache_hits,
                        step_lazy_mmap_cache_misses,
                        step_lazy_mmap_cache_evictions,
                        step_underfilled_anchor_batches,
                        window_size,
                        elapsed,
                        step_optimizer_lr if step_optimizer_lr is not None else 0.0,
                    )
                    if extended_step_metrics:
                        step_log_format += (
                            " | data_wait_time=%.3fs | non_data_time=%.3fs | "
                            "optimizer_probe_delta=%.3e"
                        )
                        step_log_args += (
                            data_wait_elapsed,
                            non_data_elapsed,
                            step_optimizer_probe_delta,
                        )
                    if compute_prefix_accuracy:
                        step_log_format += " | cumulative_target_accuracy=[%s]"
                        step_log_args += (prefix_accuracy_text,)
                    if compute_distribution_metrics:
                        step_log_format += (
                            " | teacher_forced_top1=[%s]"
                            " | teacher_forced_l1=[%s]"
                            " | expected_sampling_acceptance=[%s]"
                            " | expected_sampling_acceptance_overall=%.6f"
                        )
                        step_log_args += (
                            top1_agreement_text,
                            distribution_l1_text,
                            expected_acceptance_text,
                            expected_acceptance_overall,
                        )
                    logger.info(step_log_format, *step_log_args)
                    if phase_profile_summary:
                        host_profile_text = ", ".join(
                            f"{name}={float(values['host_seconds']):.3f}s/"
                            f"{int(values['calls'])}"
                            for name, values in phase_profile_summary.items()
                        )
                        device_profile_text = ", ".join(
                            f"{name}={float(values['device_seconds']):.3f}s"
                            for name, values in phase_profile_summary.items()
                        )
                        logger.info(
                            "DSpark step %d phase profile | host=[%s] | device=[%s]",
                            global_step,
                            host_profile_text,
                            device_profile_text,
                        )
                dspark_metrics = {
                    "train_loss": mean_loss,
                    "dspark_cross_entropy": step_cross_entropy,
                    "dspark_distribution_l1": step_distribution_l1,
                    "dspark_distribution_kl": step_distribution_kl,
                    "dspark_confidence": step_confidence,
                    "valid_block_tokens": step_valid_slots,
                    "dspark_active_anchors": step_active_anchors,
                    "dspark_requested_anchors": step_requested_anchors,
                    "dspark_anchor_chunks": step_anchor_chunks,
                    "dspark_padding_anchor_chunks": (step_padding_anchor_chunks),
                    "dspark_combined_reuse_batches": (step_combined_reuse_batches),
                    "dspark_combined_reuse_chunks": (step_combined_reuse_chunks),
                    "dspark_deferred_gradient_syncs": (step_deferred_gradient_syncs),
                    "lazy_mmap_cache_hits": step_lazy_mmap_cache_hits,
                    "lazy_mmap_cache_misses": step_lazy_mmap_cache_misses,
                    "lazy_mmap_cache_evictions": (step_lazy_mmap_cache_evictions),
                    "dspark_underfilled_anchor_batches": (
                        step_underfilled_anchor_batches
                    ),
                    "step_time": elapsed,
                    "data_wait_time": data_wait_elapsed,
                    "non_data_time": non_data_elapsed,
                    "optimizer_lr": (
                        step_optimizer_lr if step_optimizer_lr is not None else 0.0
                    ),
                    "optimizer_probe_delta": step_optimizer_probe_delta,
                }
                if compute_prefix_accuracy:
                    dspark_metrics.update(
                        {
                            f"dspark_prefix_accuracy_t{position}": accuracy
                            for position, accuracy in enumerate(
                                prefix_accuracy,
                                start=1,
                            )
                        }
                    )
                if compute_distribution_metrics:
                    for position, value in enumerate(
                        top1_agreement_by_position,
                        start=1,
                    ):
                        dspark_metrics[f"dspark_teacher_forced_top1_t{position}"] = (
                            value
                        )
                    for position, value in enumerate(
                        distribution_l1_by_position,
                        start=1,
                    ):
                        dspark_metrics[f"dspark_teacher_forced_l1_t{position}"] = value
                    for position, value in enumerate(
                        expected_acceptance_by_position,
                        start=1,
                    ):
                        dspark_metrics[
                            f"dspark_expected_sampling_acceptance_t{position}"
                        ] = value
                    dspark_metrics["dspark_expected_sampling_acceptance_overall"] = (
                        expected_acceptance_overall
                    )
                for phase_name, values in phase_profile_summary.items():
                    metric_prefix = f"dspark_phase_{phase_name}"
                    dspark_metrics[f"{metric_prefix}_host_seconds"] = float(
                        values["host_seconds"]
                    )
                    dspark_metrics[f"{metric_prefix}_device_seconds"] = float(
                        values["device_seconds"]
                    )
                    dspark_metrics[f"{metric_prefix}_calls"] = int(values["calls"])
                tracker.log_metrics(
                    dspark_metrics,
                    step=global_step,
                )
            if stop_global_step is not None and global_step >= stop_global_step:
                break

        if save_checkpoints:
            save_deepspeed_checkpoint(
                model_engine,
                checkpoint_dir,
                f"epoch_{epoch + 1}",
                logger,
                client_state={
                    # A co-training restart may consume a different immutable
                    # TrainingPlan, so DSpark resumes optimizer progress while
                    # beginning the new plan at its first local epoch.
                    "next_epoch": 0,
                    "global_step": global_step,
                    "global_examples_seen": global_examples,
                    "last_val_at_examples": 0,
                    "parallel_topology": parallel_topology.metadata(),
                    "training_method": "dspark",
                },
                save_full_model_per_rank=save_full_model_per_rank,
                save_individual_model_state=save_individual_model_state,
                parallel_topology=parallel_topology,
            )
        if stop_global_step is not None and global_step >= stop_global_step:
            break

    observed_update = observed_update or not torch.equal(
        update_probe_snapshot,
        update_probe_parameter.detach().reshape(-1)[:update_probe_values],
    )
    return {
        "global_step": global_step,
        "global_examples": global_examples - initial_global_examples,
        "observed_update": observed_update,
        "observed_nonzero_optimizer_lr": observed_nonzero_optimizer_lr,
    }
