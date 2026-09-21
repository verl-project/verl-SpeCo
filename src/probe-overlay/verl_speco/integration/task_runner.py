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
"""TaskRunner hook for the SPECO trainer."""

import json
import logging
import os
import socket
from pprint import pprint

import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import (
    need_critic,
    need_reference_policy,
)
from verl.utils.config import validate_config

try:
    from verl.trainer.main_ppo_v0 import BaseTaskRunner as _TaskRunnerBase
except ImportError:
    # verl 0.8 exposes the legacy runner directly from main_ppo.  Keep this
    # import isolated from the 0.9-only module so importing verl-SpeCo does not
    # require both release APIs to exist in the same environment.
    from verl.trainer.main_ppo import TaskRunner as _TaskRunnerBase

    _VERL_TASK_RUNNER_API = "0.8"
else:
    _VERL_TASK_RUNNER_API = "0.9"

logger = logging.getLogger(__name__)


def _serialize_drafter_config(config):
    try:
        drafter = OmegaConf.to_container(
            config.actor_rollout_ref.rollout.drafter, resolve=True
        )
    except Exception:  # noqa: BLE001
        return ""
    return json.dumps(drafter, sort_keys=True) if isinstance(drafter, dict) else ""


def _unwrap_ray_remote_actor_class(worker_cls):
    return getattr(worker_cls, "__ray_actor_class__", worker_cls)


def _remotify_like_worker_mapping_value(role_worker_cls, wrapped_cls):
    if hasattr(role_worker_cls, "__ray_actor_class__"):
        return ray.remote(wrapped_cls)
    return wrapped_cls


def _drafter_rollout_enabled(config) -> bool:
    try:
        drafter = config.actor_rollout_ref.rollout.get("drafter")
    except (AttributeError, TypeError):
        return False
    if drafter is None:
        return False
    if hasattr(drafter, "get"):
        return bool(drafter.get("enable", False))
    return bool(getattr(drafter, "enable", False))


def _rollout_name(config):
    try:
        return config.actor_rollout_ref.rollout.get("name")
    except (AttributeError, TypeError):
        return None


def _install_vllm_import_compat_for_task_runner(config) -> bool:
    if _rollout_name(config) != "vllm":
        return False
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )

    return install_verl_npu_vllm_import_compat()


class SpecoTaskRunner(_TaskRunnerBase):
    """External TaskRunner that swaps in SpecoRayPPOTrainer.

    The upstream runner moved from ``main_ppo.TaskRunner`` in verl 0.8 to
    ``main_ppo_v0.BaseTaskRunner`` in verl 0.9.  The shared SPECO hooks are the
    same, while dataset/model construction is selected below per release.
    """

    def add_actor_rollout_worker(self, config):
        worker_cls, ray_worker_group_cls = super().add_actor_rollout_worker(config)
        if _rollout_name(config) != "vllm":
            return worker_cls, ray_worker_group_cls

        from verl_speco.integration.verl_npu_vllm_compat import (
            VerlNPUVLLMImportCompatMixin,
        )

        raw_worker_cls = _unwrap_ray_remote_actor_class(worker_cls)
        if issubclass(raw_worker_cls, VerlNPUVLLMImportCompatMixin):
            return worker_cls, ray_worker_group_cls

        wrapped_cls = type(
            f"SpecoVLLMCompat{raw_worker_cls.__name__}",
            (VerlNPUVLLMImportCompatMixin, raw_worker_cls),
            {
                "__module__": __name__,
                "__doc__": raw_worker_cls.__doc__,
            },
        )
        for role, role_worker_cls in list(self.role_worker_mapping.items()):
            raw_role_worker_cls = _unwrap_ray_remote_actor_class(role_worker_cls)
            if role_worker_cls is worker_cls or raw_role_worker_cls is raw_worker_cls:
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(
                    role_worker_cls, wrapped_cls
                )
        logger.warning(
            "SPECO vLLM worker import compatibility enabled: %s", wrapped_cls.__name__
        )
        return _remotify_like_worker_mapping_value(
            worker_cls, wrapped_cls
        ), ray_worker_group_cls

    def add_speco_drafter_worker(self, config):
        """Return the external SPECO drafter worker class when online training is enabled."""
        from verl_speco.workers import SpecoWorker

        enable_drafter = bool(
            config.actor_rollout_ref.rollout.drafter.enable
            and config.actor_rollout_ref.rollout.drafter.enable_drafter_training
        )
        if not enable_drafter:
            return None
        return ray.remote(SpecoWorker)

    def _with_speco_rollout_publish_mixin(self, worker_cls, config):
        from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

        enable_drafter = bool(config.actor_rollout_ref.rollout.drafter.enable)
        raw_worker_cls = _unwrap_ray_remote_actor_class(worker_cls)
        if not enable_drafter or issubclass(raw_worker_cls, DraftWeightPublishMixin):
            return worker_cls

        wrapped_cls = type(
            f"Speco{raw_worker_cls.__name__}",
            (DraftWeightPublishMixin, raw_worker_cls),
            {
                "__module__": __name__,
                "__doc__": raw_worker_cls.__doc__,
                "_speco_sglang_drafter_config_env": _serialize_drafter_config(config),
            },
        )
        for role, role_worker_cls in list(self.role_worker_mapping.items()):
            raw_role_worker_cls = _unwrap_ray_remote_actor_class(role_worker_cls)
            if role_worker_cls is worker_cls or raw_role_worker_cls is raw_worker_cls:
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(
                    role_worker_cls, wrapped_cls
                )
        return _remotify_like_worker_mapping_value(worker_cls, wrapped_cls)

    def run(self, config):
        from verl_speco.integration.compat import check_compatible_verl

        check_compatible_verl()
        if _VERL_TASK_RUNNER_API == "0.9" and bool(config.trainer.get("use_v1", False)):
            raise RuntimeError(
                "verl-SpeCo extends the legacy RayPPOTrainer on release/v0.9.0; "
                "set trainer.use_v1=false. The V1 trainer does not expose the "
                "online drafter training and atomic weight-publish hooks yet."
            )
        if not _drafter_rollout_enabled(config):
            # The no-drafter path must reach verl through the entry-level bypass
            # in ``verl_speco.main`` so SPECO runtime/compat patches stay
            # unloaded.  Refuse here so accidental reuse of the SPECO runner can
            # never change the actor -> rollout weight-sync path.
            raise RuntimeError(
                "SpecoTaskRunner requires drafter.enable=true; "
                "use the native verl bypass path for a no-drafter run"
            )
        # Ray actors do not share imported modules. Install this in the task
        # runner process before LLMServerManager imports verl's vLLM adapter.
        _install_vllm_import_compat_for_task_runner(config)
        return self._run_with_speco_trainer(config)

    def _run_with_speco_trainer(self, config):
        from verl.utils.dataset.rl_dataset import collate_fn

        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        if _VERL_TASK_RUNNER_API == "0.9":
            from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
            from verl.utils.config import omega_conf_to_dataclass
            from verl.workers.config import HFModelConfig
        else:
            from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
            from verl.utils import hf_processor, hf_tokenizer
            from verl.utils.fs import copy_to_local

        print(f"SpecoTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        actor_rollout_cls = self._with_speco_rollout_publish_mixin(
            actor_rollout_cls, config
        )
        self.add_critic_worker(config)
        speco_worker_cls = self.add_speco_drafter_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        if _VERL_TASK_RUNNER_API == "0.9":
            model_config: HFModelConfig = omega_conf_to_dataclass(
                config.actor_rollout_ref.model
            )
            tokenizer = model_config.tokenizer
            processor = model_config.processor
        else:
            local_path = copy_to_local(
                config.actor_rollout_ref.model.path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(
                local_path,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = SpecoRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            speco_worker_cls=speco_worker_cls,
        )

        trainer.init_workers()
        trainer.fit()
