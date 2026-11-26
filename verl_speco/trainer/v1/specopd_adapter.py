# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""V1 sync lifecycle and TransferQueue bridge for one-use SpecOPD groups."""

import os
from functools import partial

from verl_speco.specopd_config import spec_opd_enabled, specopd_training_config


def enabled(trainer):
    return spec_opd_enabled(specopd_training_config(getattr(trainer, "config", None)))


def driver(trainer):
    value = trainer.__dict__.get("_specopd_v1_driver")
    if value is None:
        from verl_speco.integration.specopd_driver import SpecOPDDriver

        value = SpecOPDDriver(trainer)
        trainer._specopd_v1_driver = value
    return value


def prepare_rollout(trainer, batch):
    from tensordict import NonTensorData

    value = driver(trainer)
    value.bootstrap()
    due = value.begin_step(trainer.global_steps)
    batch["_specopd_drafter_revision"] = NonTensorData(value.drafter_revision)
    batch["_specopd_skip_collection"] = NonTensorData(not due)
    return batch


def _native_trunk_log_prob(trainer, batch, trunk, *, calculate_entropy):
    import torch
    import transfer_queue as tq
    from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
    from verl.utils.seqlen_balancing import (
        calculate_workload,
        get_seqlen_balanced_partitions,
    )
    from verl.workers.utils.padding import response_from_nested

    from verl_speco.joint_scoring import JointScores

    keys = [batch.keys[i] for i, _ in trunk]
    subset = batch.select_keys(keys)
    subset.extra_info.update(
        calculate_entropy=calculate_entropy,
        compute_loss=False,
        temperature=trainer.config.actor_rollout_ref.rollout.temperature,
    )
    worker_group = trainer.actor_rollout_wg
    # V1 _balance_batch initializes the native dispatch map before old-logprob.
    dp_size = max(worker_group._dispatch_info["actor"]) + 1
    subset = upsample_batch_to_divisible_size(
        subset, dp_size, trainer.tokenizer.eos_token_id
    )
    padding_keys = subset.keys[len(keys) :]
    try:
        if dp_size > 1:
            lengths = torch.tensor(
                [tag["seq_len"] for tag in subset.tags], dtype=torch.int64
            )
            partitions = get_seqlen_balanced_partitions(
                calculate_workload(lengths),
                k_partitions=dp_size,
                equal_size=True,
            )
            subset.reorder([i for partition in partitions for i in partition])
        output = worker_group.compute_log_prob(subset)
        if (
            set(output.keys) != set(subset.keys)
            or output.partition_id != batch.partition_id
        ):
            raise RuntimeError(
                "native old-logprob did not return the requested TransferQueue keys"
            )
    finally:
        if padding_keys:
            tq.kv_clear(keys=padding_keys, partition_id=batch.partition_id)
    fields = ["log_probs", "response_mask"]
    if calculate_entropy:
        fields.append("entropy")
    data = tq.kv_batch_get(
        keys=keys, partition_id=batch.partition_id, select_fields=fields
    )
    mask = data["response_mask"]
    log_probs = response_from_nested(data["log_probs"], mask).unbind()
    entropies = (
        response_from_nested(data["entropy"], mask).unbind()
        if calculate_entropy
        else (None,) * len(trunk)
    )
    results = []
    for (_, request), log_prob, entropy in zip(
        trunk, log_probs, entropies, strict=True
    ):
        if log_prob.numel() != len(request.trunk_ids) - request.response_start:
            raise RuntimeError(
                "native old-logprob response length differs from the joint request"
            )
        results.append(JointScores(request.sample_id, log_prob, (), {}, entropy))
    return tuple(results)


def old_log_prob(trainer, batch, metrics):
    import transfer_queue as tq
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.workers.utils.padding import response_to_nested

    nested, padded = trainer._speco_v1_batch_data(batch)
    value = driver(trainer)
    if os.environ.get("VERL_SPECO_V1_NATIVE_TRUNK_OLD_LOGPROB") == "1":
        output = value.old_log_prob(
            padded, native_trunk=partial(_native_trunk_log_prob, trainer, batch)
        )
    else:
        output = value.old_log_prob(padded)
    log_probs = output["old_log_probs"].float()
    entropy = output["entropy"].float()
    nested["old_log_probs"] = response_to_nested(log_probs, nested["response_mask"])
    nested["entropy"] = response_to_nested(entropy, nested["response_mask"])
    result = tq.kv_batch_put(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=nested.select("old_log_probs", "entropy"),
    )
    config = trainer.config.actor_rollout_ref.actor
    metrics["actor/entropy"] = (
        agg_loss(
            loss_mat=entropy,
            loss_mask=padded.batch["response_mask"],
            loss_agg_mode=config.loss_agg_mode,
            loss_scale_factor=config.loss_scale_factor,
        )
        .detach()
        .item()
    )
    if trainer.config.actor_rollout_ref.rollout.get("calculate_log_probs", False):
        from verl.utils.debug.metrics import calculate_debug_metrics

        padded.batch["old_log_probs"] = log_probs
        metrics.update(calculate_debug_metrics(padded))
    metrics.update(value.metrics)
    return result
