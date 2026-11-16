# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""V1 sync lifecycle and TransferQueue bridge for one-use SpecOPD groups."""

from verl_speco.opd_method import spec_opd_enabled


def enabled(trainer):
    value = getattr(trainer, "config", None)
    for key in ("actor_rollout_ref", "rollout", "drafter", "training"):
        value = value.get(key, {}) if hasattr(value, "get") else getattr(value, key, {})
    return spec_opd_enabled(value)


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
    batch["_specopd_drafter_revision"] = NonTensorData(value.cycle.drafter_revision)
    batch["_specopd_skip_collection"] = NonTensorData(not due)
    return batch


def old_log_prob(trainer, batch, metrics):
    import transfer_queue as tq
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.workers.utils.padding import response_to_nested

    nested, padded = trainer._speco_v1_batch_data(batch)
    value = driver(trainer)
    output, _ = value.old_log_prob(padded)
    log_probs = output.batch["old_log_probs"].float()
    entropy = output.batch["entropys"].float()
    nested["old_log_probs"] = response_to_nested(log_probs, nested["response_mask"])
    nested["entropy"] = response_to_nested(entropy, nested["response_mask"])
    result = tq.kv_batch_put(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=nested.select("old_log_probs", "entropy"),
    )
    config = trainer.config.actor_rollout_ref.actor
    metrics["actor/entropy"] = agg_loss(
        loss_mat=entropy,
        loss_mask=padded.batch["response_mask"],
        loss_agg_mode=config.loss_agg_mode,
        loss_scale_factor=config.loss_scale_factor,
    ).detach().item()
    if trainer.config.actor_rollout_ref.rollout.get("calculate_log_probs", False):
        from verl.utils.debug.metrics import calculate_debug_metrics

        padded.batch["old_log_probs"] = log_probs
        metrics.update(calculate_debug_metrics(padded))
    metrics.update(value.metrics)
    return result
