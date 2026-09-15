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
"""Contract for the SPECO entry-level native bypass.

When speculative drafting is disabled, ``verl_speco.main`` must fall through to
verl's native ``run_ppo`` so the actor -> rollout weight-sync path matches
upstream verl exactly.  SPECO's task runner, runtime bridge and weight-sync
compat extension stay unloaded.  ``SpecoTaskRunner.run`` additionally refuses a
drafter-disabled config so accidental reuse of the SPECO runner can never
change the no-drafter reward distribution.
"""

from __future__ import annotations

import pytest

omegaconf = pytest.importorskip("omegaconf", reason="bypass contract needs omegaconf")
OmegaConf = omegaconf.OmegaConf


def _config(*, enable=False, training=False) -> object:
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": enable,
                        "enable_drafter_training": training,
                    }
                }
            },
        }
    )


def test_should_bypass_true_when_drafter_disabled() -> None:
    from verl_speco.main import should_bypass_speco

    assert should_bypass_speco(_config()) is True


def test_should_bypass_false_when_drafter_rollout_enabled() -> None:
    from verl_speco.main import should_bypass_speco

    assert should_bypass_speco(_config(enable=True)) is False


def test_should_bypass_false_when_drafter_training_enabled() -> None:
    from verl_speco.main import should_bypass_speco

    assert should_bypass_speco(_config(enable=True, training=True)) is False


def test_should_bypass_rejects_training_without_rollout() -> None:
    from verl_speco.main import should_bypass_speco

    with pytest.raises(
        ValueError,
        match="enable_drafter_training=true requires drafter.enable=true",
    ):
        should_bypass_speco(_config(enable=False, training=True))


def test_should_bypass_defaults_true_when_speco_block_missing() -> None:
    from verl_speco.main import should_bypass_speco

    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"drafter": {"enable": False}}}}
    )
    assert should_bypass_speco(config) is True


def _patch_verl_entry(monkeypatch) -> list:
    import sys

    import verl.trainer.main_ppo as main_ppo
    import verl.utils.device as device
    import verl_speco.integration.compat as compat

    calls: list = []

    def fake_run_ppo(config, task_runner_class=None):
        calls.append(task_runner_class)

    monkeypatch.setattr(main_ppo, "run_ppo", fake_run_ppo)
    monkeypatch.setattr(device, "auto_set_device", lambda config: None)
    monkeypatch.setattr(compat, "check_compatible_verl", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_ppo, "migrate_legacy_reward_impl", None, raising=False
    )
    # Drop any cached SPECO task-runner module so the no-drafter dispatch can
    # prove the bypass never imports it.  monkeypatch restores the originals.
    for mod in list(sys.modules):
        if mod == "verl_speco.integration.task_runner" or mod.startswith(
            "verl_speco.integration.task_runner."
        ):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    return calls


def test_run_bypasses_to_native_verl_when_drafter_disabled(monkeypatch) -> None:
    pytest.importorskip("verl", reason="dispatch contract needs verl")
    pytest.importorskip("ray", reason="dispatch contract needs ray")
    import sys

    from verl_speco.main import run

    calls = _patch_verl_entry(monkeypatch)
    run(_config())

    # Native verl path: run_ppo is called without a SPECO task runner so verl
    # selects its own legacy or V1 TaskRunner.
    assert calls == [None]
    # The no-drafter bypass must never import the SPECO task runner, which
    # would pull in the runtime bridge, weight-sync compat extension and
    # forced no-async-scheduling configuration.
    assert "verl_speco.integration.task_runner" not in sys.modules


def test_run_uses_speco_task_runner_when_drafter_enabled(monkeypatch) -> None:
    pytest.importorskip("verl", reason="dispatch contract needs verl")
    pytest.importorskip("ray", reason="dispatch contract needs ray")

    from verl_speco.main import run

    calls = _patch_verl_entry(monkeypatch)
    run(_config(enable=True))

    assert len(calls) == 1
    task_runner_class = calls[0]
    assert task_runner_class is not None
    from verl_speco.integration.task_runner import SpecoTaskRunner

    assert (
        getattr(task_runner_class, "__ray_actor_class__", None) is SpecoTaskRunner
    )


def test_run_bypass_propagates_training_without_rollout_error(monkeypatch) -> None:
    pytest.importorskip("verl", reason="dispatch contract needs verl")
    pytest.importorskip("ray", reason="dispatch contract needs ray")

    from verl_speco.main import run

    _patch_verl_entry(monkeypatch)
    with pytest.raises(
        ValueError,
        match="enable_drafter_training=true requires drafter.enable=true",
    ):
        run(_config(enable=False, training=True))


def test_run_bypass_strips_speco_overlay_before_native_run(monkeypatch) -> None:
    pytest.importorskip("verl", reason="dispatch contract needs verl")
    pytest.importorskip("ray", reason="dispatch contract needs ray")
    from omegaconf import OmegaConf

    import verl.trainer.main_ppo as main_ppo
    import verl.utils.device as device
    import verl_speco.integration.compat as compat

    seen: list = []

    def fake_run_ppo(config, task_runner_class=None):
        seen.append(config)

    monkeypatch.setattr(main_ppo, "run_ppo", fake_run_ppo)
    monkeypatch.setattr(device, "auto_set_device", lambda config: None)
    monkeypatch.setattr(compat, "check_compatible_verl", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_ppo, "migrate_legacy_reward_impl", None, raising=False
    )

    from verl_speco.main import run

    config = OmegaConf.create(
        {
            "speco": {"enable": True},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "n": 1,
                    "drafter": {"enable": False, "enable_drafter_training": False},
                }
            },
        }
    )

    run(config)

    assert len(seen) == 1
    native_config = seen[0]
    # The speco_base drafter overlay is rejected by verl's RolloutConfig, so
    # the bypass must drop it (and the top-level speco block) before run_ppo.
    assert "drafter" not in native_config.actor_rollout_ref.rollout
    assert "speco" not in native_config
    # The strip must be surgical: native rollout keys stay intact.
    assert native_config.actor_rollout_ref.rollout.name == "vllm"
    assert native_config.actor_rollout_ref.rollout.n == 1
