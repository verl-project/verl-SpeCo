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
"""Hydra entrypoint for SPECO training.

When speculative drafting is disabled, SPECO falls through to verl's native
``run_ppo`` so the actor -> rollout weight-sync path matches upstream verl
exactly.  SPECO's runtime bridge, weight-sync compat extension and trainer are
only imported when a drafter is enabled, keeping the no-drafter reward
distribution aligned with verl.
"""

import hydra


def _config_get(config, *path, default=None):
    """Read a nested config value from OmegaConf or plain namespaces."""

    node = config
    for key in path:
        if node is None:
            return default
        if hasattr(node, "get"):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


def should_bypass_speco(config) -> bool:
    """Return whether the run should skip SPECO and use verl's native path.

    A run always bypasses SPECO when the drafter is disabled for both rollout
    and training, so the actor -> rollout weight-sync path matches upstream
    verl exactly.  Requesting drafter training without enabling rollout is
    rejected so the bypass never hides an inconsistent configuration.
    """

    rollout_enabled = bool(
        _config_get(
            config,
            "actor_rollout_ref",
            "rollout",
            "drafter",
            "enable",
            default=False,
        )
    )
    training_enabled = bool(
        _config_get(
            config,
            "actor_rollout_ref",
            "rollout",
            "drafter",
            "enable_drafter_training",
            default=False,
        )
    )
    if training_enabled and not rollout_enabled:
        raise ValueError("enable_drafter_training=true requires drafter.enable=true")

    return not rollout_enabled and not training_enabled


def _strip_speco_overlay_for_native_run(config) -> None:
    """Drop SPECO-only config keys so verl's native ``run_ppo`` accepts it.

    The bypass path hands the config to verl directly. verl instantiates its
    ``RolloutConfig`` from ``actor_rollout_ref.rollout`` and rejects the
    speco_base ``drafter`` overlay; the top-level ``speco`` block is also
    SPECO-only. Removing both keeps the native run free of SPECO config.
    """

    from omegaconf import OmegaConf, open_dict

    with open_dict(config):
        rollout = _config_get(config, "actor_rollout_ref", "rollout")
        if OmegaConf.is_config(rollout) and "drafter" in rollout:
            del rollout["drafter"]
        if OmegaConf.is_config(config) and "speco" in config:
            del config["speco"]


def run(config) -> None:
    """Resolve SPECO/verl compatibility, device and the task-runner dispatch."""

    from verl.trainer import main_ppo
    from verl.utils.device import auto_set_device

    from verl_speco.integration.compat import check_compatible_verl

    check_compatible_verl()
    auto_set_device(config)
    migrate_legacy_reward_impl = getattr(main_ppo, "migrate_legacy_reward_impl", None)
    if migrate_legacy_reward_impl is not None:
        # Present in verl 0.8 and intentionally removed from the 0.9 legacy
        # runner.  Apply it only where upstream still defines the migration.
        config = migrate_legacy_reward_impl(config)

    if should_bypass_speco(config):
        # Native verl path: keep SPECO runtime/compat patches unloaded so the
        # actor -> rollout weight sync matches verl exactly.  verl selects its
        # own TaskRunner (legacy on 0.8, legacy or V1 on 0.9).
        _strip_speco_overlay_for_native_run(config)
        main_ppo.run_ppo(config)
        return

    import ray

    from verl_speco.integration.task_runner import SpecoTaskRunner

    main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SpecoTaskRunner))


@hydra.main(config_path="config", config_name="speco_trainer", version_base=None)
def main(config):
    run(config)


if __name__ == "__main__":
    main()
