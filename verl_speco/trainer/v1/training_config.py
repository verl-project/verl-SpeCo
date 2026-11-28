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

"""Configuration checks shared by the driver and V1 trainers."""


def validate_v1_training_config(config) -> None:
    drafter = config.actor_rollout_ref.rollout.get("drafter", {}) or {}
    training = drafter.get("training", {}) or {}
    if not bool(drafter.get("enable", False)) or not bool(
        drafter.get("enable_drafter_training", False)
    ):
        return
    if str(config.actor_rollout_ref.rollout.get("name", "")).lower() != "vllm":
        raise ValueError(
            "Phase-1 V1 online SPECO training supports rollout.name=vllm only"
        )
    from verl_speco.trainer.scheduler import DrafterScheduleConfig

    bubble = (
        DrafterScheduleConfig.from_mapping(training).execution_strategy.value
        == "rollout_idle_worker"
    )
    mode = str(config.trainer.v1.get("trainer_mode", "sync"))
    idle = (training.get("scheduler", {}) or {}).get("idle_worker", {}) or {}
    if bubble and idle.get("v1_head_preparation_enable", False):
        raise ValueError(
            "v1_head_preparation_enable is no longer supported; remove the "
            "override or set it to false. Quota heads are prepared by training preflight."
        )
    if bubble and mode != "sync":
        raise ValueError(
            "V1 Bubble Time supports only trainer_mode=sync; "
            "disable rollout_idle_worker for async training"
        )
    if bubble and str(training.get("mode", "online")).strip().lower() == "collect_only":
        raise ValueError(
            "V1 Bubble Time does not support training.mode=collect_only; "
            "use online training or disable rollout_idle_worker"
        )
    if bubble and not bool(training.get("save_full_drafter_checkpoint", True)):
        raise ValueError(
            "V1 Bubble Time requires save_full_drafter_checkpoint=true "
            "to restore drafter weights and state from a joint checkpoint"
        )
    local_updates = int(
        (config.trainer.v1.get(mode, {}) or {}).get("parameter_sync_step", 1)
    )
    if bubble and local_updates > 1:
        # Sync updates pi after each local batch.
        # A global-step head key would alias distinct policy snapshots.
        raise ValueError(
            f"V1 Bubble {mode} requires parameter_sync_step=1: multiple local "
            "actor updates would mix policy/head snapshots under one quota "
            "version"
        )
    if bool(training.get("collect_hidden_states_from_sgl", False)):
        raise ValueError(
            "verl V1 online SPECO training currently requires "
            "collect_hidden_states_from_sgl=false; use old-logprob collection"
        )
    if not bool(training.get("collect_hidden_states_from_old_logprob", False)):
        raise ValueError(
            "verl V1 online SPECO training requires "
            "training.collect_hidden_states_from_old_logprob=true; use --config-name=speco_v1_trainer or set the option explicitly"
        )
    rollout_correction = config.algorithm.get("rollout_correction", None)
    if rollout_correction and bool(rollout_correction.get("bypass_mode", False)):
        raise ValueError(
            "Phase-1 V1 online SPECO training is incompatible with "
            "algorithm.rollout_correction.bypass_mode=true because hidden "
            "states must be collected from actor old-logprob inference"
        )
    freeze = training.get("drafter_convergence_freeze", {}) or {}
    if (
        not bubble
        and freeze.get("enabled", False)
        and str(config.trainer.v1.get("trainer_mode", "sync"))
        not in {"sync", "colocate_async"}
    ):
        raise ValueError(
            "V1 marginal-utility freeze requires sync or colocate_async for fixed-target paired probes"
        )
