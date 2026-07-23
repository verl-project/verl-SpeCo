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
