"""Hydra entrypoint for SPECO training."""

import hydra


@hydra.main(config_path="config", config_name="speco_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    from verl_speco.integration.compat import check_compatible_verl
    from verl_speco.integration.task_runner import SpecoTaskRunner

    check_compatible_verl()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    import ray

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SpecoTaskRunner))


if __name__ == "__main__":
    main()
