"""TaskRunner hook for the SPECO trainer."""

import os
import socket
import json
import logging
from contextlib import contextmanager, nullcontext
from pprint import pprint

import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

logger = logging.getLogger(__name__)


def _serialize_drafter_config(config):
    try:
        drafter = OmegaConf.to_container(config.actor_rollout_ref.rollout.drafter, resolve=True)
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


def _open_config_mapping(mapping):
    return open_dict(mapping) if OmegaConf.is_config(mapping) else nullcontext()


@contextmanager
def _prepare_no_drafter_upstream_config(config):
    rollout_config = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
    missing = object()
    drafter_config = missing
    no_async_scheduling = missing
    vllm_engine_kwargs = None
    if rollout_config is not None and hasattr(rollout_config, "__contains__") and "drafter" in rollout_config:
        drafter_config = rollout_config["drafter"]
        with _open_config_mapping(rollout_config):
            del rollout_config["drafter"]
    if rollout_config is not None and rollout_config.get("name") == "vllm":
        with _open_config_mapping(rollout_config):
            engine_kwargs = rollout_config.get("engine_kwargs")
            if engine_kwargs is None:
                engine_kwargs = {}
                rollout_config["engine_kwargs"] = engine_kwargs
            with _open_config_mapping(engine_kwargs):
                vllm_engine_kwargs = engine_kwargs.get("vllm")
                if vllm_engine_kwargs is None:
                    vllm_engine_kwargs = {}
                    engine_kwargs["vllm"] = vllm_engine_kwargs
                with _open_config_mapping(vllm_engine_kwargs):
                    no_async_scheduling = vllm_engine_kwargs.get("no-async-scheduling", missing)
                    vllm_engine_kwargs["no-async-scheduling"] = True
        logger.info("SPECO no-drafter baseline: forcing vLLM async scheduling off")
    try:
        yield
    finally:
        if drafter_config is not missing:
            with _open_config_mapping(rollout_config):
                rollout_config["drafter"] = drafter_config
        if vllm_engine_kwargs is not None:
            with _open_config_mapping(vllm_engine_kwargs):
                if no_async_scheduling is missing:
                    del vllm_engine_kwargs["no-async-scheduling"]
                else:
                    vllm_engine_kwargs["no-async-scheduling"] = no_async_scheduling


class SpecoTaskRunner(TaskRunner):
    """External TaskRunner that swaps in SpecoRayPPOTrainer.

    Adapted from verl v0.8.0
    ``verl/trainer/main_ppo.py::TaskRunner.run``.
    """

    def add_actor_rollout_worker(self, config):
        worker_cls, ray_worker_group_cls = super().add_actor_rollout_worker(config)
        if _drafter_rollout_enabled(config) or _rollout_name(config) != "vllm":
            return worker_cls, ray_worker_group_cls

        from verl_speco.integration.verl_npu_vllm_compat import VerlNPUVLLMImportCompatMixin

        raw_worker_cls = _unwrap_ray_remote_actor_class(worker_cls)
        if issubclass(raw_worker_cls, VerlNPUVLLMImportCompatMixin):
            return worker_cls, ray_worker_group_cls

        wrapped_cls = type(
            f"SpecoNoDrafter{raw_worker_cls.__name__}",
            (VerlNPUVLLMImportCompatMixin, raw_worker_cls),
            {
                "__module__": __name__,
                "__doc__": raw_worker_cls.__doc__,
            },
        )
        for role, role_worker_cls in list(self.role_worker_mapping.items()):
            raw_role_worker_cls = _unwrap_ray_remote_actor_class(role_worker_cls)
            if role_worker_cls is worker_cls or raw_role_worker_cls is raw_worker_cls:
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(role_worker_cls, wrapped_cls)
        logger.warning("SPECO no-drafter vLLM worker import compatibility enabled: %s", wrapped_cls.__name__)
        return _remotify_like_worker_mapping_value(worker_cls, wrapped_cls), ray_worker_group_cls

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

        enable_drafter = bool(
            config.actor_rollout_ref.rollout.drafter.enable
        )
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
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(role_worker_cls, wrapped_cls)
        return _remotify_like_worker_mapping_value(worker_cls, wrapped_cls)

    def run(self, config):
        # Preserve the upstream release/v0.8.0 execution path when SPECO is
        # disabled. This keeps the no-drafter baseline independent from the
        # custom TaskRunner and trainer integration.
        if not _drafter_rollout_enabled(config):
            with _prepare_no_drafter_upstream_config(config):
                return super().run(config)

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        print(f"SpecoTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        actor_rollout_cls = self._with_speco_rollout_publish_mixin(actor_rollout_cls, config)
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

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

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
