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
"""Exercise the installed V1-only runtime on a CPU runner."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("SPECO_REQUIRE_V1_ONLY") != "1", reason="requires pinned V1-only verl"
)


@pytest.mark.parametrize("name", ["speco_trainer", "speco_v1_trainer"])
def test_real_v1_sync_config_and_shared_hooks(name, monkeypatch):
    from hydra import compose, initialize_config_dir
    from verl.trainer import main_ppo
    from verl.trainer.ppo.v1 import PPOTrainerSync
    from verl_speco import main
    from verl_speco.integration import task_runner, verl_npu_vllm_compat
    from verl_speco.trainer.v1.factory import get_speco_v1_trainer_cls

    with initialize_config_dir(
        config_dir=str(Path(main.__file__).parent / "config"), version_base=None
    ):
        cfg = compose(
            config_name=name,
            overrides=[
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.drafter.enable=true",
                "actor_rollout_ref.rollout.drafter.enable_drafter_training=true",
                "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=true",
                "algorithm.adv_estimator=grpo",
            ],
        )

    reached = []

    def inspect_before_ray(config, task_runner_class):
        assert task_runner._VERL_TASK_RUNNER_API == "v1-only"
        assert not verl_npu_vllm_compat._needs_legacy_fused_moe_guard()
        assert callable(task_runner_class.remote)
        assert config.trainer.use_v1 and config.transfer_queue.enable
        cls = get_speco_v1_trainer_cls("sync")
        assert issubclass(cls, PPOTrainerSync)
        trainer = cls(config=config)
        assert trainer.trainer_mode == "sync"
        # The V1 mixin binds the actual legacy hook module on demand.
        worker_group = object()
        trainer.drafter_wg = worker_group
        assert trainer._require_speco_worker_group() is worker_group
        trainer._speco_validate_v1_training_config(config)
        trainer.close_dataloaders()
        reached.append(True)

    monkeypatch.setattr(main_ppo, "run_ppo", inspect_before_ray)
    main.run(cfg)
    assert reached == [True]
