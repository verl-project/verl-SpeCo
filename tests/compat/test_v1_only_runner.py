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
"""Exercise SpeCo's driver and Ray actor boundary without creating a cluster."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_speco import main
from verl_speco.integration import compat
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__path__ = []
    vars(module).update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent, child = name.rsplit(".", 1)
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


@pytest.fixture
def upstream(monkeypatch):
    for name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.utils"):
        _module(monkeypatch, name)
    _module(
        monkeypatch,
        "verl.trainer.ppo.utils",
        need_critic=Mock(),
        need_reference_policy=Mock(),
    )
    _module(monkeypatch, "verl.utils.config", validate_config=Mock())
    _module(monkeypatch, "verl.utils.device", auto_set_device=Mock())
    entry = _module(monkeypatch, "verl.trainer.main_ppo", TaskRunnerV1=object())
    monkeypatch.delitem(sys.modules, "verl.trainer.main_ppo_v0", raising=False)
    _module(monkeypatch, "ray", remote=Mock())
    monkeypatch.setattr(compat, "check_compatible_verl", Mock())
    return entry


def _load_runner(monkeypatch):
    name = "verl_speco.integration.task_runner"
    path = Path(compat.__file__).with_name("task_runner.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "rollout, training", [(False, False), (True, False), (True, True)]
)
def test_original_recipe_flags_reach_correct_runner(
    upstream, monkeypatch, rollout, training
):
    config = OmegaConf.create(
        {
            "trainer": {"use_v1": False, "v1": {"trainer_mode": "sync"}},
            "transfer_queue": {"enable": False},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {
                        "enable": rollout,
                        "enable_drafter_training": training,
                        "training": {"collect_hidden_states_from_old_logprob": True},
                    },
                }
            },
            "algorithm": {},
        }
    )
    monkeypatch.setattr(main, "configure_compile_cache_runtime_env", lambda _: {})
    monkeypatch.setattr(main, "configure_drafter_runtime_env", lambda _: {})
    monkeypatch.setattr(main, "drafter_diagnostic_env", dict)
    monkeypatch.setattr(main, "configure_phase_diagnostic_runtime_env", lambda _: None)
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.task_runner", raising=False
    )
    if rollout:
        adapter = _load_runner(monkeypatch)
        assert adapter._VERL_TASK_RUNNER_API == "v1-only"
        remote = sys.modules["ray"].remote
        remote.return_value.side_effect = lambda cls: cls
        delegated = Mock()
        monkeypatch.setattr(adapter.SpecoTaskRunner, "_run_with_v1_trainer", delegated)
        monkeypatch.setattr(
            adapter, "_install_vllm_import_compat_for_task_runner", Mock()
        )

    def run_ppo(config, task_runner_class):
        assert config.trainer.use_v1 and config.transfer_queue.enable
        if rollout:
            assert task_runner_class is adapter.SpecoTaskRunner
            task_runner_class().run(config)
            delegated.assert_called_once_with(config)
            SpecoV1Mixin._speco_validate_v1_training_config(config)
            assert SpecoV1Mixin._speco_online_enabled_from_config(config) == training
        else:
            assert task_runner_class is upstream.TaskRunnerV1
            assert "drafter" not in config.actor_rollout_ref.rollout
            assert "verl_speco.integration.task_runner" not in sys.modules

    upstream.run_ppo = run_ppo
    main.run(config)


@pytest.mark.parametrize("api", ["0.8", "0.9"])
def test_legacy_runner_still_uses_installed_base(upstream, monkeypatch, api):
    class LegacyRunner:
        pass

    if api == "0.8":
        del upstream.TaskRunnerV1
        upstream.TaskRunner = LegacyRunner
    else:
        _module(monkeypatch, "verl.trainer.main_ppo_v0", BaseTaskRunner=LegacyRunner)
    assert not compat.has_only_v1_runner(upstream)
    adapter = _load_runner(monkeypatch)
    assert adapter._VERL_TASK_RUNNER_API == api
    assert issubclass(adapter.SpecoTaskRunner, LegacyRunner)


@pytest.mark.parametrize("fail", [False, True])
def test_v1_lifecycle_closes_driver_resources(upstream, monkeypatch, fail):
    adapter = _load_runner(monkeypatch)
    config = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": "sync"}},
            "transfer_queue": {"enable": True},
            "actor_rollout_ref": {"rollout": {"drafter": {"enable": True}}},
        }
    )
    from verl_speco.integration import agent_loop_runtime
    from verl_speco.trainer.v1 import factory

    tq = _module(monkeypatch, "transfer_queue", init=Mock(), close=Mock())
    manager_cls = Mock()
    _module(monkeypatch, "verl.trainer.ppo.v1", AgentLoopManagerTQ=manager_cls)
    _module(monkeypatch, "verl.utils.import_utils", load_class_from_fqn=Mock())
    trainer = Mock()
    monkeypatch.setattr(
        factory, "get_speco_v1_trainer_cls", lambda _: lambda config: trainer
    )
    monkeypatch.setattr(agent_loop_runtime, "install_agent_loop_runtime_patch", Mock())
    monkeypatch.setattr(
        agent_loop_runtime, "install_v1_agent_loop_worker_bridge", Mock()
    )
    if fail:
        trainer.fit.side_effect = RuntimeError("fit failed")
        with pytest.raises(RuntimeError, match="fit failed"):
            adapter.SpecoTaskRunner()._run_with_v1_trainer(config)
    else:
        adapter.SpecoTaskRunner()._run_with_v1_trainer(config)
    trainer.init.assert_called_once_with()
    trainer.prepare_for_fit.assert_called_once_with(manager_cls.create.return_value)
    trainer.fit.assert_called_once_with(manager_cls.create.return_value)
    assert (
        "drafter"
        not in manager_cls.create.call_args.kwargs["config"].actor_rollout_ref.rollout
    )
    assert config.actor_rollout_ref.rollout.drafter.enable
    trainer.close_dataloaders.assert_called_once_with()
    trainer.logger.finish.assert_called_once_with(exit_code=int(fail))
    tq.init.assert_called_once_with(config.transfer_queue)
    tq.close.assert_called_once_with()


@pytest.mark.parametrize(
    "version, accepted",
    [
        ("0.10.0.dev", True),
        ("0.10.0.dev0+local", True),
        ("0.10.0rc1", True),
        ("0.10.0", True),
        ("0.10.2", True),
        ("0.11.0.dev0", True),
        ("0.11.0", True),
        ("0.12.0", True),
        ("1.0.0", True),
        ("0.9.1", False),
        ("invalid", False),
    ],
)
def test_v1_release_line_requires_matching_api(monkeypatch, version, accepted):
    monkeypatch.delenv(compat.ALLOW_UNSUPPORTED_ENV, raising=False)
    monkeypatch.setattr(compat, "_read_imported_verl_version", lambda: version)
    monkeypatch.setattr(compat, "_read_distribution_vcs_info", lambda: (None, "main"))
    check = Mock(return_value=())
    monkeypatch.setattr(compat, "_missing_required_api", check)
    assert compat.resolve_verl_compatibility().supported is accepted
    if accepted:
        check.assert_called_once_with(compat.V1_VERSION_RANGE)
        check.return_value = ("verl.trainer.ppo.v1.PPOTrainerSync",)
        assert not compat.resolve_verl_compatibility().supported
        with pytest.raises(RuntimeError, match="PPOTrainerSync"):
            compat.check_compatible_verl(strict=True)
