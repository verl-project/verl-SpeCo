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

"""Thin lifecycle integration for the verl V1 PPO trainers."""

from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
import time
from typing import Any, cast

logger = logging.getLogger(__name__)


def _plain_config(value):
    """Convert an OmegaConf/plain mapping to a JSON-safe value."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        if isinstance(value, dict) or hasattr(value, "items"):
            return {str(key): _plain_config(item) for key, item in value.items()}
        return value


def _unwrap_remote(cls):
    return getattr(cls, "__ray_actor_class__", cls)


def _remotify_like(original, cls):
    import ray

    return ray.remote(cls) if hasattr(original, "__ray_actor_class__") else cls


class SpecoV1Mixin:
    """Add SPECO worker/runtime wiring while preserving the V1 PPO loop.

    Phase 1 deliberately does not alter V1's ``KVBatchMeta`` pipeline.  The
    V1 loop owns sampling and policy updates; the shared SPECO scheduler,
    collector, feature store, and publication facade are delegated to the
    legacy adapter implementation.
    """

    speco_worker_cls = None
    global_steps: int
    _speco_prepared_for_fit = False
    _speco_prefit_reissue_consumed = False
    _speco_prefit_on_train_begin_consumed = False

    @staticmethod
    def _speco_v1_standalone_publish_worker_cls(drafter_config=None):
        """Return a rollout-side worker that can consume a draft IPC payload.

        ``separate_async`` gives its continuously-serving replicas their own
        ``CheckpointEngineWorker`` group.  Unlike a hybrid actor/rollout worker,
        that upstream worker has no draft-publish RPC, even though its embedded
        vLLM rollout already has the required IPC receiver.  Add the narrow
        publish facade before the standalone replicas are created.
        """
        serialized_drafter_config = json.dumps(
            _plain_config(drafter_config or {}), sort_keys=True
        )
        cached = getattr(
            SpecoV1Mixin, "_speco_v1_standalone_publish_worker_remote", None
        )
        if (
            cached is not None
            and getattr(
                SpecoV1Mixin, "_speco_v1_standalone_publish_worker_config", None
            )
            == serialized_drafter_config
        ):
            return cached

        import ray
        from verl.checkpoint_engine.base import CheckpointEngineWorker
        from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

        class SpecoV1StandalonePublishWorker(
            DraftWeightPublishMixin, CheckpointEngineWorker
        ):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rollout = self.server_adapter
                # Upstream RolloutConfig deliberately has no SPECO ``drafter``
                # extension.  Preserve it in the facade config so a standalone
                # publish RPC reaches the embedded rollout instead of no-oping.
                self.config = {
                    "rollout": {
                        "name": self.rollout_config.name,
                        "drafter": json.loads(type(self)._speco_drafter_config_env),
                    }
                }

        SpecoV1StandalonePublishWorker.__module__ = __name__
        SpecoV1StandalonePublishWorker._speco_drafter_config_env = (
            serialized_drafter_config
        )
        globals()[SpecoV1StandalonePublishWorker.__name__] = (
            SpecoV1StandalonePublishWorker
        )
        cached = ray.remote(SpecoV1StandalonePublishWorker)
        SpecoV1Mixin._speco_v1_standalone_publish_worker_remote = cached
        SpecoV1Mixin._speco_v1_standalone_publish_worker_config = (
            serialized_drafter_config
        )
        return cached

    def _speco_install_v1_standalone_publish_worker(self) -> None:
        mode = str(self.config.trainer.v1.get("trainer_mode", "sync")).lower()
        drafter = self.config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        if mode != "separate_async" or not bool(drafter.get("enable", False)):
            return

        from verl.workers.rollout.replica import RolloutMode, RolloutReplica

        if getattr(RolloutReplica, "_speco_v1_publish_worker_patched", False):
            return
        original = RolloutReplica.get_ray_class_with_init_args

        def get_ray_class_with_speco_publish(replica):
            init_args = original(replica)
            if getattr(replica, "rollout_mode", None) == RolloutMode.STANDALONE:
                init_args.cls = SpecoV1Mixin._speco_v1_standalone_publish_worker_cls(
                    drafter
                )
            return init_args

        RolloutReplica.get_ray_class_with_init_args = get_ray_class_with_speco_publish
        RolloutReplica._speco_v1_publish_worker_patched = True

    def _speco_init_state(self):
        from verl_speco.trainer.scheduler import DrafterRuntimeState, DrafterScheduler

        self.device_name = getattr(
            self, "device_name", self.config.trainer.get("device", "cuda")
        )
        self.drafter_wg = None
        self._drafter_scheduler = DrafterScheduler()
        self._drafter_runtime_state = DrafterRuntimeState()
        self._pending_drafter_publish_refs = None
        self._pending_drafter_checkpoint_refs = []
        self._pending_target_lm_head_sync = None
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_oldlogprob_candidate_samples = 0
        self._speco_last_oldlogprob_short_response_skipped = 0
        self._speco_last_oldlogprob_planned_samples = 0
        self._speco_last_oldlogprob_collected_samples = 0
        self._speco_last_oldlogprob_collected_rows = 0
        self._speco_last_oldlogprob_payload_mib = 0.0
        self._speco_last_oldlogprob_select_elapsed_sec = 0.0
        self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
        self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
        self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
        self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
        self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
        self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
        self._speco_last_oldlogprob_total_elapsed_sec = 0.0
        self._speco_last_collect_interval_matched = 0
        self._speco_last_collection_outcome = None
        self._speco_drafter_frozen = False
        self._speco_freeze_policy = None
        self._speco_freeze_policy_active = False
        self._speco_last_freeze_decision = None
        self._speco_freeze_branch_save_pending = None
        self._speco_freeze_branch_saved = False
        self._speco_last_checkpoint_saved_step = None
        self._speco_last_request_accept_len_records = []
        self._speco_pending_request_accept_len_payloads = []
        self._speco_last_request_accept_len_var = 0.0
        self._speco_v1_evidence_step = None

    def __getattr__(self, name):
        """Lazily expose the stable SPECO facade implemented by the legacy adapter.

        The V1 trainer owns the PPO loop, while the scheduler, feature-store,
        and worker RPC facade are shared with the legacy trainer.  Lazy binding
        avoids importing the legacy Ray trainer during V1 module discovery and
        keeps one implementation of the runtime protocol.
        """
        if (
            name.startswith("_speco")
            or name.startswith("speco_")
            or name.startswith("is_drafter_")
            or name
            in {
                "_ray_get_if_needed",
                "_first_non_null",
                "_require_speco_worker_group",
            }
        ):
            from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

            descriptor = inspect.getattr_static(SpecoRayPPOTrainer, name, None)
            if descriptor is not None:
                return descriptor.__get__(self, type(self))
        raise AttributeError(name)

    @staticmethod
    def _speco_flatten_checkpoint_results(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened: list[dict[str, Any]] = []
            for item in value:
                flattened.extend(SpecoV1Mixin._speco_flatten_checkpoint_results(item))
            return flattened
        return []

    def _speco_feature_store_checkpoint_configured(self) -> bool:
        training = (self.config.actor_rollout_ref.rollout.get("drafter", {}) or {}).get(
            "training", {}
        ) or {}
        feature_store = training.get("feature_store", None)
        return bool(feature_store and feature_store.get("path", None))

    def _speco_v1_checkpoint_directory(self, global_step: int | None = None) -> str:
        step = self.global_steps if global_step is None else global_step
        return os.path.join(
            str(self.config.trainer.default_local_dir), f"global_step_{int(step)}"
        )

    def _speco_v1_joint_checkpoint_manifest_path(
        self, global_step: int | None = None
    ) -> str:
        return os.path.join(
            self._speco_v1_checkpoint_directory(global_step), "speco_v1_manifest.json"
        )

    def _speco_finalize_v1_drafter_publish(self) -> dict[str, Any]:
        if not getattr(self, "_speco_v1_pending_training", False):
            return {}
        training_plan = getattr(self, "_speco_v1_training_plan", None)
        try:
            policy = getattr(self, "_speco_freeze_policy", None)
            run_probe = policy is not None and policy.probe_due(
                policy.drafter_version + 1
            )
            before_arm = None
            probe = None
            probe_error = None
            if run_probe:
                try:
                    before_arm = self._speco_run_freeze_probe_arm()
                except Exception:  # noqa: BLE001 - retain fail-open policy semantics
                    probe_error = "before_arm_failed"
                    logger.exception("SPECO V1 paired probe before-arm failed")
            # Keep the established V1 order: the parent completes its actor
            # weight sync before drafter SHM reload starts.  The publish RPC is
            # still awaited here, so the next V1 step cannot consume a partial
            # drafter revision.
            metrics = self._speco_publish_drafter_weights(
                True, training_plan, after_weight_update=True
            )
            waited = self._speco_wait_pending_drafter_publish()
            if (
                getattr(self, "_speco_colocate_admission_closed", False)
                and policy is not None
                and bool(metrics.get("drafter/published", False))
            ):
                # Label requests only after successful atomic commits on every
                # serving replica. No ordinary request can race this RPC.
                self._speco_rollout_admission_rpc(
                    "speco_set_serving_drafter_revision", policy.drafter_version + 1
                )
            metrics["drafter/publish_waited_after_weight_sync"] = int(waited)
            metrics["drafter/publish_safe_point_after_weight_sync"] = 1
            if run_probe and before_arm is not None and policy is not None:
                try:
                    probe = self._speco_build_freeze_probe_comparison(
                        before_arm,
                        self._speco_run_freeze_probe_arm(),
                        version_before=policy.drafter_version,
                    )
                except Exception:  # noqa: BLE001 - missing evidence keeps training
                    probe_error = "after_arm_failed"
                    logger.exception("SPECO V1 paired probe after-arm failed")
            if policy is not None:
                total_cost = float(getattr(self, "_speco_v1_train_cost_seconds", 0.0))
                total_cost += self._speco_freeze_publish_cost_seconds(metrics) or 0.0
                self._speco_freeze_observe_publish(
                    metrics, update_cost_seconds=total_cost, probe=probe
                )
                if run_probe:
                    self._speco_freeze_log_probe(probe, probe_error)
                metrics.update(self._speco_freeze_metrics())
            return metrics
        finally:
            # A failed publish raises after the legacy facade has restored the
            # previous payload.  Do not retry it from a later checkpoint hook.
            self._speco_v1_pending_training = False
            self._speco_v1_training_plan = None

    def _speco_freeze_probe_prompt_ids(self, chunk):
        """Encode the fixed V1 raw prompts with the serving tokenizer path."""
        cached = chunk.get("_speco_v1_probe_prompt_ids")
        if cached is not None:
            return cached
        if "input_ids" in chunk and "attention_mask" in chunk:
            prompts = [
                ids[mask.bool()].tolist()
                for ids, mask in zip(
                    chunk["input_ids"], chunk["attention_mask"], strict=True
                )
            ]
        else:
            # Current native V1 datasets defer tokenization to AgentLoop.
            # Use exactly its ContinuousToken builder and left-truncation,
            # rather than guessing a chat template or retokenizing each arm.
            from transformers import AutoConfig
            from verl.utils.tokenizer.continuous_token_wiring import (
                create_continuous_token_builder,
            )

            model_cfg = self.config.actor_rollout_ref.model
            hf_config = AutoConfig.from_pretrained(
                model_cfg.path,
                trust_remote_code=model_cfg.get("trust_remote_code", False),
            )
            builder = create_continuous_token_builder(
                self.tokenizer,
                hf_model_type=hf_config.model_type,
                chat_template_kwargs=dict(
                    self.config.data.get("apply_chat_template_kwargs", {})
                ),
                mm_processor_kwargs=dict(
                    self.config.data.get("mm_processor_kwargs", {})
                ),
                processor=self.processor,
            )
            limit = int(self.config.actor_rollout_ref.rollout.prompt_length)
            prompts = [
                builder.build_initial_tokens(list(messages))[-limit:]
                for messages in chunk["raw_prompt"]
            ]
        chunk["_speco_v1_probe_prompt_ids"] = prompts
        return prompts

    def _speco_run_freeze_probe_arm(self):
        """Probe the serving V1 engines without inserting samples into TQ.

        Both arms run at the post-target-sync boundary in sync mode. Fixed
        prompt order maps each pair to the same server. V1 collects features
        only during actor old-logprob inference, which these direct serving
        calls bypass. Probe tokens never enter PPO/rewards or the TQ buffer.
        """
        import ray
        from uuid import uuid4

        servers = list(self.llm_server_manager.server_handles)
        if not servers:
            raise RuntimeError("V1 paired probes require initialized rollout servers")
        max_tokens = int(self._speco_freeze_probe_config().get("max_new_tokens", 2048))
        rows, offset, wall_seconds = [], 0, 0.0
        for chunk in self._speco_freeze_probe_pool_data():
            prompts = self._speco_freeze_probe_prompt_ids(chunk)
            refs = []
            for index, prompt_ids in enumerate(prompts):
                pair_id = f"speco-freeze-probe-{offset + index:04d}"
                server = servers[(offset + index) % len(servers)]
                generate = (
                    server.speco_generate_probe
                    if getattr(self, "_speco_colocate_admission_closed", False)
                    else server.generate
                )
                refs.append(
                    generate.remote(
                        prompt_ids=prompt_ids,
                        sampling_params={
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_tokens": max_tokens,
                            "logprobs": False,
                        },
                        request_id=f"{pair_id}-{uuid4().hex}",
                    )
                )
            started = time.perf_counter()
            outputs = ray.get(refs)
            wall_seconds += time.perf_counter() - started
            for index, output in enumerate(outputs):
                fields = getattr(output, "extra_fields", {}) or {}
                accept = fields.get("_verl_request_mean_accept_len")
                if accept is None:
                    rounds = fields.get("spec_num_verify_steps", 0) or 0
                    accepted = fields.get("spec_num_accepted_tokens", 0) or 0
                    accept = (
                        1.0 + float(accepted) / float(rounds) if rounds > 0 else None
                    )
                seconds = fields.get("_speco_vllm_request_elapsed_sec")
                rows.append(
                    {
                        "uid": f"speco-freeze-probe-{offset + index:04d}",
                        "accept": accept,
                        "tokens": len(getattr(output, "token_ids", [])),
                        "seconds": float(seconds)
                        if seconds is not None and float(seconds) > 0
                        else None,
                    }
                )
            offset += len(outputs)
        if not rows:
            raise RuntimeError("V1 paired probe returned no rows")
        return rows, wall_seconds

    def _speco_snapshot_v1_feature_store_checkpoint(self) -> list[dict[str, Any]]:
        if not self._speco_feature_store_checkpoint_configured():
            return []
        results = self._ray_get_if_needed(
            self.speco_get_feature_store_checkpoint_state(self.global_steps)
        )
        flattened = self._speco_flatten_checkpoint_results(results)
        failures = [
            result
            for result in flattened
            if not bool(result.get("saved", False))
            and result.get("reason")
            not in {"not_in_training_group", "not_feature_store_leader"}
        ]
        if failures:
            raise RuntimeError(
                f"Feature-store checkpoint cursor save failed: {failures}"
            )
        saved = [result for result in flattened if bool(result.get("saved", False))]
        if not saved:
            raise RuntimeError(
                "Feature-store checkpoint cursor produced no saved state"
            )
        return saved

    def _speco_write_v1_joint_checkpoint_manifest(
        self,
        *,
        drafter_results: Any,
        feature_store: list[dict[str, Any]],
    ) -> str:
        checkpoint_dir = self._speco_v1_checkpoint_directory()
        os.makedirs(checkpoint_dir, exist_ok=True)
        manifest_path = self._speco_v1_joint_checkpoint_manifest_path()
        manifest = {
            "format": "speco_v1_joint_checkpoint",
            "version": 1,
            "global_step": int(self.global_steps),
            "trainer_mode": str(self.trainer_mode),
            # This is the revision actually committed to rollout, which can
            # legitimately lag global_step when a training plan does not publish.
            "drafter_version": getattr(self, "_speco_last_published_drafter_step", 0),
            "drafter_checkpoints": self._speco_flatten_checkpoint_results(
                drafter_results
            ),
            "feature_store": feature_store,
        }
        async_evidence = getattr(self, "_speco_v1_async_checkpoint_evidence", None)
        if async_evidence is not None:
            manifest["async_rollout_checkpoint"] = async_evidence
        fd, temporary_path = tempfile.mkstemp(
            prefix=".speco_v1_manifest.", suffix=".tmp", dir=checkpoint_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=True, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, manifest_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return manifest_path

    def _speco_restore_v1_feature_store_checkpoint(self) -> None:
        if not self._speco_feature_store_checkpoint_configured():
            return
        resume_step = self._speco_resume_global_step_hint()
        if resume_step is None:
            return
        manifest_path = self._speco_v1_joint_checkpoint_manifest_path(resume_step)
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                "V1 resume with Feature Store enabled requires a joint SPECO "
                f"checkpoint manifest: {manifest_path}"
            )
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid V1 SPECO checkpoint manifest: {manifest_path}"
            ) from exc
        if (
            manifest.get("format") != "speco_v1_joint_checkpoint"
            or int(manifest.get("version", 0)) != 1
            or int(manifest.get("global_step", -1)) != int(resume_step)
        ):
            raise RuntimeError(
                f"V1 SPECO checkpoint manifest does not match resumed step {resume_step}"
            )
        states = manifest.get("feature_store")
        if not isinstance(states, list) or not states:
            raise RuntimeError(
                "V1 SPECO checkpoint manifest must contain a Feature Store cursor"
            )
        # Writers have rank/PID-specific shard prefixes and can flush unequal
        # numbers of shards. Their local next indices are not part of the
        # shared manifest prefix that restore validates.
        unique_states = {
            json.dumps(
                {
                    key: value
                    for key, value in state["cursor"].items()
                    if key != "next_shard_index"
                },
                sort_keys=True,
            ): state
            for state in states
            if isinstance(state, dict) and isinstance(state.get("cursor"), dict)
        }
        if len(unique_states) != 1:
            raise RuntimeError(
                "V1 SPECO checkpoint manifest must contain one unique Feature Store cursor"
            )
        state = next(iter(unique_states.values()))
        results = self._ray_get_if_needed(
            self.speco_restore_feature_store_checkpoint_state(state)
        )
        flattened = self._speco_flatten_checkpoint_results(results)
        failures = [
            result
            for result in flattened
            if not bool(result.get("restored", False))
            and result.get("reason")
            not in {"not_in_training_group", "not_feature_store_leader"}
        ]
        if failures or not any(result.get("restored", False) for result in flattened):
            raise RuntimeError(
                "Feature-store checkpoint cursor restore failed: "
                f"{failures or flattened}"
            )
        logger.info("SPECO V1 restored Feature Store cursor from step=%s", resume_step)

    def attach_speco_worker_group(self, worker_group):
        """Bind V1 drafter workers to the shared SPECO adapter facade."""
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        return SpecoRayPPOTrainer.attach_speco_worker_group(self, worker_group)

    @staticmethod
    def _speco_online_enabled_from_config(config) -> bool:
        drafter = config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        return bool(
            drafter.get("enable", False)
            and drafter.get("enable_drafter_training", False)
        )

    def _speco_v1_async_rollout_enabled(self) -> bool:
        mode = str(self.config.trainer.v1.get("trainer_mode", "sync")).lower()
        drafter = self.config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        return mode in {"colocate_async", "separate_async"} and bool(
            drafter.get("enable", False)
        )

    @staticmethod
    def _speco_validate_v1_training_config(config) -> None:
        drafter = config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        training = drafter.get("training", {}) or {}
        if not bool(drafter.get("enable", False)) or not bool(
            drafter.get("enable_drafter_training", False)
        ):
            return
        if bool(training.get("collect_hidden_states_from_sgl", False)):
            raise ValueError(
                "verl V1 online SPECO training currently requires "
                "collect_hidden_states_from_sgl=false; use old-logprob collection"
            )
        if not bool(training.get("collect_hidden_states_from_old_logprob", False)):
            raise ValueError(
                "verl V1 online SPECO training requires "
                "training.collect_hidden_states_from_old_logprob=true"
            )
        rollout_correction = config.algorithm.get("rollout_correction", None)
        if rollout_correction and bool(rollout_correction.get("bypass_mode", False)):
            raise ValueError(
                "Phase-1 V1 online SPECO training is incompatible with "
                "algorithm.rollout_correction.bypass_mode=true because hidden "
                "states must be collected from actor old-logprob inference"
            )
        if str(config.actor_rollout_ref.rollout.get("name", "")).lower() != "vllm":
            raise ValueError(
                "Phase-1 V1 online SPECO training supports rollout.name=vllm only"
            )
        freeze = training.get("drafter_convergence_freeze", {}) or {}
        if freeze.get("enabled", False) and str(
            config.trainer.v1.get("trainer_mode", "sync")
        ) not in {"sync", "colocate_async"}:
            raise ValueError(
                "V1 marginal-utility freeze requires sync or colocate_async for fixed-target paired probes"
            )

    def _setup(self):
        from verl_speco.integration.vllm_runtime import (
            configure_vllm_runtime_from_config,
        )

        # Must precede PPOTrainerSeparateAsync's standalone LLMServerManager
        # construction, otherwise its CheckpointEngineWorker actors are already
        # fixed without the draft-publication facade.
        self._speco_install_v1_standalone_publish_worker()
        self._speco_validate_v1_training_config(self.config)
        self._speco_init_state()
        if self._speco_online_enabled_from_config(self.config):
            # Resolve a resumed draft checkpoint before V1 creates the rollout
            # server.  The server reads drafter.model_path during construction.
            self._speco_prepare_drafter_checkpoint_for_worker_init()

        configure_vllm_runtime_from_config(self.config)
        self._speco_drafter_config_for_worker_wrap = (
            self.config.actor_rollout_ref.rollout.get("drafter", None)
        )
        # SPECO translates this extension field into vLLM-native
        # ``engine_kwargs.vllm.speculative_config`` above.  Keep the source
        # field present as well: custom/third-party V1 launchers may read it,
        # and removing it here would make their rollout replicas silently lose
        # speculative decoding.
        result = super()._setup()
        if self._speco_online_enabled_from_config(self.config):
            # The actor/rollout workers and their placement group now exist, so
            # the drafter can safely share their bundles before the first
            # checkpoint-manager weight update in on_init_end().
            self._init_v1_speco_drafter_workers()
            self._speco_restore_v1_feature_store_checkpoint()
        self._speco_v1_state = {"drafter_version": None, "features_collected": 0}
        return result

    def _init_resource_pool_mgr(self):
        result = super()._init_resource_pool_mgr()
        self._wrap_v1_actor_worker()
        return result

    def _speco_drafter_publish_method(self, method_name: str):
        """Route both publish and rollback to the replicas serving rollouts.

        The hybrid actor worker is in trainer mode during this boundary.  Its
        vLLM server must not consume the draft IPC stream; doing so races the
        detached actor's own weight lifecycle.  The standalone rollout workers
        are the replicas that serve the next batch, so publish directly to them.
        """
        mode = str(self.config.trainer.v1.get("trainer_mode", "sync")).lower()
        if mode != "separate_async":
            from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

            return SpecoRayPPOTrainer._speco_drafter_publish_method(
                cast(Any, self), method_name
            )

        manager = getattr(self, "standalone_server_manager", None)
        replicas = manager.get_replicas() if manager is not None else []
        workers = [worker for replica in replicas for worker in replica.workers]
        if not workers:
            raise RuntimeError(
                "V1 separate_async drafter publish requires initialized standalone rollout workers"
            )

        from verl.single_controller.ray.base import RayClassWithInitArgs, RayWorkerGroup

        worker_group = RayWorkerGroup(
            worker_handles=workers,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=self._speco_v1_standalone_publish_worker_cls()
            ),
        )
        return getattr(worker_group, method_name)

    def _wrap_v1_actor_worker(self):
        rollout = self.config.actor_rollout_ref.rollout
        drafter_config = getattr(self, "_speco_drafter_config_for_worker_wrap", None)
        if str(rollout.get("name", "")).lower() != "vllm":
            return
        from verl.trainer.ppo.utils import Role

        from verl_speco.integration.rollout_publish import DraftWeightPublishMixin
        from verl_speco.integration.verl_npu_vllm_compat import (
            VerlNPUVLLMImportCompatMixin,
        )

        for role, worker_cls in list(self.role_worker_mapping.items()):
            if role not in {Role.ActorRollout, Role.ActorRolloutRef}:
                continue
            raw = _unwrap_remote(worker_cls)
            bases = []
            if not issubclass(raw, VerlNPUVLLMImportCompatMixin):
                bases.append(VerlNPUVLLMImportCompatMixin)
            if bool(
                (drafter_config or rollout.get("drafter", {})).get("enable", False)
            ) and not issubclass(raw, DraftWeightPublishMixin):
                bases.append(DraftWeightPublishMixin)
            if not bases:
                continue
            wrapped = type(
                f"SpecoV1{raw.__name__}",
                tuple(bases) + (raw,),
                {
                    "__module__": __name__,
                    "__doc__": raw.__doc__,
                    "_speco_drafter_config_env": json.dumps(
                        _plain_config(drafter_config or rollout.get("drafter", {})),
                        sort_keys=True,
                    ),
                },
            )
            self.role_worker_mapping[role] = _remotify_like(worker_cls, wrapped)
            logger.info("SPECO V1 wrapped actor worker: %s", wrapped.__name__)

    def on_init_end(self):
        result = super().on_init_end()
        online_drafter = bool(
            self.config.actor_rollout_ref.rollout.get("drafter", {}).get(
                "enable_drafter_training", False
            )
        )
        logger.info(
            "SPECO V1 trainer initialized: mode=%s, online_drafter=%s",
            self.trainer_mode,
            online_drafter,
        )
        return result

    def on_step_end(self):
        try:
            result = super().on_step_end()
            # Native colocate_async resumes the engine during target sync.
            # Keep ordinary requests behind admission until both probe arms
            # and the drafter publication have settled at that fixed target.
            publish_metrics = self._speco_finalize_v1_drafter_publish()
            if publish_metrics:
                self._pending_sync_metrics = {
                    **(getattr(self, "_pending_sync_metrics", None) or {}),
                    **publish_metrics,
                }
            return result
        finally:
            if getattr(self, "_speco_colocate_admission_closed", False):
                self._speco_rollout_admission_rpc("speco_resume_rollout_admission")
                self._speco_colocate_admission_closed = False

    def _save_checkpoint(self):
        if (
            getattr(self, "_speco_last_checkpoint_saved_step", None)
            == self.global_steps
        ):
            return None
        drafter_results = None
        if self._speco_online_enabled_from_config(self.config):
            self._speco_wait_pending_drafter_publish()
            drafter_results = self._speco_save_drafter_checkpoint(wait=True)
            self._speco_freeze_save_state()
        result = super()._save_checkpoint()
        if getattr(
            self, "_speco_colocate_admission_closed", False
        ) and self._speco_freeze_branch_config().get("enabled", False):
            queue_path = os.path.join(
                self._speco_v1_checkpoint_directory(), "transfer_queue"
            )
            if not os.path.isdir(queue_path) or not os.listdir(queue_path):
                raise RuntimeError(
                    "Async freeze branch is incomplete: TransferQueue snapshot is missing"
                )
            self._speco_v1_async_checkpoint_evidence = {
                "transfer_queue": queue_path,
                "checkpoint_step": int(self.global_steps),
                "inflight_prompts_reissued_on_resume": True,
            }
        if self._speco_online_enabled_from_config(self.config):
            feature_store = self._speco_snapshot_v1_feature_store_checkpoint()
            self._speco_write_v1_joint_checkpoint_manifest(
                drafter_results=drafter_results,
                feature_store=feature_store,
            )
        self._speco_last_checkpoint_saved_step = self.global_steps
        return result

    def on_sample_end(self):
        if not self._speco_colocate_probe_boundary_enabled():
            return super().on_sample_end()
        self._speco_rollout_admission_rpc("speco_close_rollout_admission")
        self._speco_colocate_admission_closed = True
        try:
            # Wait only for serving RPCs after abort, not for TQ agent loops:
            # those loops can be waiting to retry when cards are returned.
            self.checkpoint_manager.abort_replicas()
            self._speco_rollout_admission_rpc("speco_wait_rollout_idle")
            self.checkpoint_manager.sleep_replicas()
        except BaseException:
            self._speco_rollout_admission_rpc("speco_resume_rollout_admission")
            self._speco_colocate_admission_closed = False
            raise

    def _speco_colocate_probe_boundary_enabled(self):
        training = self._speco_drafter_training_config() or {}
        return str(
            self.config.trainer.v1.get("trainer_mode", "sync")
        ).lower() == "colocate_async" and bool(
            training.get("drafter_convergence_freeze", {}).get("enabled", False)
        )

    def _speco_rollout_admission_rpc(self, method, *args):
        import ray

        servers = list(self.llm_server_manager.server_handles)
        if not servers:
            raise RuntimeError(
                "V1 async probe boundary requires initialized rollout servers"
            )
        return ray.get([getattr(server, method).remote(*args) for server in servers])

    def prepare_step(self):
        """Avoid creating an unconsumed rollout at the async loop boundary.

        V1 calls ``prepare_step`` before sampling the current batch.  In
        ``colocate_async`` that method submits a batch which becomes
        unconsumed at the final loop boundary, because upstream ``fit``
        returns immediately after the step.  Its agent-loop request then
        races Ray's rollout-server teardown and surfaces as an
        ``ActorDiedError`` or vLLM ``EngineDeadError``.

        Sync mode must retain the parent behavior because it creates the batch
        consumed by the current step rather than maintaining an async prefetch
        buffer.  ``separate_async`` must also retain its parent behavior: its
        final ``prepare_step`` submits the batch that is consumed by the final
        update, then waits for it and switches hybrid workers back to trainer
        mode.  The V1 agent-loop drain in ``fit`` makes any post-final queued
        work safe before teardown.
        """
        mode = str(self.config.trainer.v1.get("trainer_mode", "sync")).lower()
        is_last_step = int(self.global_steps) >= int(self.total_training_steps)
        if mode == "colocate_async" and is_last_step:
            logger.info(
                "SPECO V1 skipping final async rollout prefetch at step=%s; "
                "the batch cannot be consumed before trainer teardown",
                self.global_steps,
            )
            return {}
        return super().prepare_step()

    @staticmethod
    def _speco_v1_drain_agent_loop(agent_loop_manager: Any) -> int:
        """Wait for all requests submitted before V1 destroys rollout actors."""
        workers = list(getattr(agent_loop_manager, "agent_loop_workers", None) or [])
        drain_refs = []
        for worker in workers:
            drain = getattr(worker, "speco_drain", None)
            remote = getattr(drain, "remote", None)
            if callable(remote):
                drain_refs.append(remote())
        if not drain_refs:
            return 0

        import ray

        ray.get(drain_refs)
        logger.info(
            "SPECO V1 drained %s agent-loop workers before rollout teardown",
            len(drain_refs),
        )
        return len(drain_refs)

    def _speco_v1_shutdown_dataloaders(self) -> int:
        """Stop StatefulDataLoader workers before the owning Ray task exits.

        verl V1 keeps the training iterator alive in ``train_dataloader_it``.
        If it is left for Ray process teardown, Ray kills the iterator's child
        processes and PyTorch's SIGCHLD handler reports a misleading
        ``DataLoader worker ... is killed by signal: Killed`` after the final
        training step.  StatefulDataLoader currently exposes worker shutdown on
        its iterator, so clean up every live iterator while the task is still
        in normal Python control flow.
        """
        holders = [(self, "train_dataloader_it")]
        for loader_name in ("train_dataloader", "val_dataloader"):
            loader = getattr(self, loader_name, None)
            if loader is not None:
                holders.append((loader, "_iterator"))

        shutdown_count = 0
        seen: set[int] = set()
        for owner, attribute in holders:
            iterator = getattr(owner, attribute, None)
            if iterator is None or id(iterator) in seen:
                continue
            seen.add(id(iterator))
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
                shutdown_count += 1

        # Drop all references after shutdown so a later destructor cannot race
        # Ray teardown or attempt to close the same multiprocessing iterator.
        for owner, attribute in holders:
            if hasattr(owner, attribute):
                setattr(owner, attribute, None)

        if shutdown_count:
            logger.info(
                "SPECO V1 shut down %s DataLoader iterator(s) before Ray teardown",
                shutdown_count,
            )
        return shutdown_count

    def _init_v1_speco_drafter_workers(self):
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.trainer.ppo.utils import Role
        from verl_speco.workers import SpecoWorker

        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self.role_worker_mapping
            else Role.ActorRollout
        )
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        worker_cls = self.speco_worker_cls or SpecoWorker
        remote_worker_cls = (
            worker_cls
            if hasattr(worker_cls, "__ray_actor_class__")
            else __import__("ray").remote(worker_cls)
        )
        drafter_cls = RayClassWithInitArgs(
            cls=remote_worker_cls,
            config=self.config.actor_rollout_ref,
            role="drafter",
            device_name=self.config.trainer.device,
        )
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=drafter_cls,
            name_prefix="speco_v1_drafter",
            device_name=self.config.trainer.device,
        )
        worker_group.init_model()
        self.attach_speco_worker_group(worker_group)

    def _speco_v1_batch_data(self, batch):
        import transfer_queue as tq
        from verl.protocol import DataProto
        from verl_speco.trainer.v1.batch_adapter import to_legacy_padded_batch

        fields = ["prompts", "responses", "input_ids", "response_mask", "position_ids"]
        if bool(
            self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False)
        ):
            fields.append("rollout_log_probs")
        if bool(
            self.config.actor_rollout_ref.rollout.get(
                "enable_rollout_routing_replay", False
            )
        ):
            fields.append("routed_experts")
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=fields
        )
        pad_token_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", 0)
        result = DataProto(batch=to_legacy_padded_batch(data, pad_token_id or 0))
        # V1 stores request metadata in TQ, not DataProto.non_tensor_batch.
        # The shared hard-sample and freeze policies require those exact cells.
        stats = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        cells = stats.get("extra_fields", [])
        if hasattr(cells, "tolist"):
            cells = cells.tolist()
        import numpy as np

        request_fields = [getattr(cell, "data", cell) for cell in cells]
        # Upstream balances V1 batches with synthetic zero-loss trajectories.
        # Their extra_fields are copied from a real trajectory, including its
        # request ID/counters. Keep those actor rows, but never count them as
        # serving requests or hard-sampling evidence.
        tags = getattr(batch, "tags", [])
        padding = [
            bool(tags[index].get("is_padding", False)) if index < len(tags) else False
            for index in range(len(request_fields))
        ]
        result.meta_info["speco_v1_padding_rows"] = sum(padding)
        for key in (
            "_verl_request_mean_accept_len",
            "_speco_vllm_request_id",
            "_speco_vllm_request_verify_rounds",
            "_speco_vllm_request_accepted_tokens",
            "_speco_vllm_request_completion_index",
            "_speco_vllm_request_elapsed_sec",
            "_speco_drafter_revision_min",
            "_speco_drafter_revision_max",
            "_speco_drafter_revision_stats",
        ):
            values = [
                cell.get(key) if isinstance(cell, dict) and not padding[index] else None
                for index, cell in enumerate(request_fields)
            ]
            if len(values) == len(result):
                result.non_tensor_batch[key] = np.asarray(values, dtype=object)
        return data, result

    def _speco_observe_async_revision_evidence(self, evidence, metrics):
        import numpy as np
        from verl.protocol import DataProto
        from verl_speco.integration.partial_rollout_stats import (
            group_serving_revision_evidence,
        )

        current = self._speco_freeze_policy.drafter_version
        groups, matched = group_serving_revision_evidence(
            evidence.non_tensor_batch, len(evidence), current
        )
        metrics["drafter/freeze_revision_matched_rows"] = matched
        metrics["drafter/freeze_revision_excluded_rows"] = len(evidence) - matched
        versions = sorted(groups) or [current]
        opportunity = bool(self._speco_should_train_drafter_this_step())
        try:
            for position, version in enumerate(versions):
                rows = groups.get(version, [])
                projection = DataProto(
                    non_tensor_batch={
                        "_speco_vllm_request_id": np.array(
                            [row["id"] for row in rows], dtype=object
                        ),
                        "_verl_request_mean_accept_len": np.array(
                            [row["mean"] for row in rows]
                        ),
                    },
                    meta_info={
                        "metrics": {
                            "response_length/mean": sum(row["tokens"] for row in rows)
                            / len(rows)
                            if rows
                            else 0.0
                        }
                    },
                )
                self._speco_populate_request_accept_len_records(projection)
                self._speco_observe_rollout_evidence(
                    projection,
                    version,
                    generation_seconds=sum(row["seconds"] for row in rows),
                    opportunity_this_step=opportunity and position == len(versions) - 1,
                )
        finally:
            # Hard sampling/PPO retain complete original trajectories; only
            # freeze windows use per-request, per-serving-revision projections.
            self._speco_populate_request_accept_len_records(evidence)

    def _compute_old_log_prob(self, batch, metrics):
        policy = getattr(self, "_speco_freeze_policy", None)
        if (
            policy is not None
            and getattr(self, "_speco_v1_evidence_step", None) != self.global_steps
        ):
            _, evidence = self._speco_v1_batch_data(batch)
            metrics["drafter/request_padding_rows"] = evidence.meta_info[
                "speco_v1_padding_rows"
            ]
            self._speco_populate_request_accept_len_records(evidence)
            if self._speco_colocate_probe_boundary_enabled():
                self._speco_observe_async_revision_evidence(evidence, metrics)
            else:
                self._speco_observe_rollout_evidence(evidence, policy.drafter_version)
            self._speco_v1_evidence_step = self.global_steps
            metrics.update(self._speco_freeze_metrics())
        if not self._speco_oldlogprob_collection_enabled():
            return super()._compute_old_log_prob(batch, metrics)

        import transfer_queue as tq
        import torch
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import (
            left_right_2_no_padding,
            no_padding_2_padding,
            response_to_nested,
        )
        from verl_speco.integration.oldlogprob_runtime import (
            OLD_LOGPROB_AUX_LAYER_IDS_KEY,
            OLD_LOGPROB_COLLECT_MASK_KEY,
            OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
            OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
            OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY,
            OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
            OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
            OLD_LOGPROB_OWNER_RANK_KEY,
            OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY,
        )

        nested_data, data_proto = self._speco_v1_batch_data(batch)
        collect_plan = self._speco_build_oldlogprob_collect_plan(data_proto)
        if collect_plan is None:
            return super()._compute_old_log_prob(batch, metrics)

        control = data_proto.to_tensordict()
        control = left_right_2_no_padding(control)
        control[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
        control[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan["hidden_positions"]
        control[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan[
            "hidden_position_mask"
        ]
        control[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
        control[OLD_LOGPROB_SOURCE_BATCH_INDICES_KEY] = torch.arange(
            len(data_proto), device=control["input_ids"].device
        )
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_AUX_LAYER_IDS_KEY,
            self._speco_oldlogprob_aux_layer_ids(),
        )
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
            self._speco_oldlogprob_hidden_capture_impl(),
        )
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
            self._speco_oldlogprob_hidden_layout(),
        )
        tu.assign_non_tensor_data(control, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True)
        actor_megatron_cfg = self.config.actor_rollout_ref.actor.get("megatron", {})
        tu.assign_non_tensor_data(
            control,
            "speco_oldlogprob_sp_disabled",
            not bool(actor_megatron_cfg.get("sequence_parallel", True)),
        )
        tu.assign_non_tensor(
            control,
            # V1 always computes entropy in this stage.  Preserve that contract
            # even though the legacy adapter permits disabling it.
            calculate_entropy=True,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        output = self._speco_compute_collection_log_prob(control)
        output_data = output
        collected = self._speco_collect_oldlogprob_features(
            data_proto, collect_plan, output_data
        )
        self._speco_v1_state["features_collected"] = int(
            self._speco_v1_state.get("features_collected", 0) + collected
        )
        collection_plan_data = collect_plan["collection_plan"]
        collection_outcome = getattr(self, "_speco_last_collection_outcome", None)
        metrics.update(
            {
                "drafter/oldlogprob_candidate_samples": int(
                    getattr(self, "_speco_last_oldlogprob_candidate_samples", 0)
                ),
                "drafter/oldlogprob_short_response_skipped": int(
                    getattr(self, "_speco_last_oldlogprob_short_response_skipped", 0)
                ),
                "drafter/oldlogprob_planned_samples": int(
                    getattr(self, "_speco_last_oldlogprob_planned_samples", 0)
                ),
                "drafter/oldlogprob_collected_samples": int(
                    getattr(self, "_speco_last_oldlogprob_collected_samples", 0)
                ),
                "drafter/oldlogprob_collected_rows": int(
                    getattr(self, "_speco_last_oldlogprob_collected_rows", 0)
                ),
            }
        )
        metrics.update(collection_plan_data.metrics())
        metrics.update(getattr(self, "_speco_last_hard_selection_metrics", {}) or {})
        if collection_outcome is not None:
            metrics.update(collection_outcome.metrics())
            worker_versions = [
                (
                    result.worker_id,
                    result.data_version,
                    result.buffer_version_before,
                    result.buffer_version_after,
                )
                for result in (collection_outcome.worker_results or [])
            ]
            logger.info(
                "SPECO V1 old-logprob collection step=%s collection_id=%s "
                "candidates=%s planned=%s collected=%s rows=%s reason=%s "
                "worker_versions=%s",
                collection_plan_data.source_global_step,
                collection_plan_data.collection_id,
                self._speco_last_oldlogprob_candidate_samples,
                self._speco_last_oldlogprob_planned_samples,
                self._speco_last_oldlogprob_collected_samples,
                self._speco_last_oldlogprob_collected_rows,
                collection_outcome.reason,
                worker_versions,
            )

        response_mask = data_proto.batch.get("response_mask")
        log_probs = no_padding_2_padding(output_data["log_probs"], control)
        entropy = output_data.get("entropy")
        entropy = (
            no_padding_2_padding(entropy, control)
            if entropy is not None
            else torch.zeros_like(log_probs)
        )
        nested_data["old_log_probs"] = response_to_nested(
            log_probs.float(), nested_data["response_mask"]
        )
        nested_data["entropy"] = response_to_nested(
            entropy.float(), nested_data["response_mask"]
        )
        updated_batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=nested_data.select("old_log_probs", "entropy"),
        )
        if bool(
            self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False)
        ):
            from verl.utils.debug.metrics import calculate_debug_metrics

            # The V1 batch is normally kept in TransferQueue, so construct the
            # same padded view expected by the upstream debug helper after the
            # new old-logprobs have been produced.
            data_proto.batch["old_log_probs"] = log_probs.float()
            metrics.update(calculate_debug_metrics(data_proto))
        from verl.trainer.ppo.core_algos import agg_loss

        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()
        return updated_batch

    def _speco_compute_collection_log_prob(self, control):
        """Keep separate-async's fixed proximal policy during collection.

        The collection forward bypasses the parent's KVBatchMeta adapter, but
        must not bypass PPOTrainerSeparateAsync's CPU policy snapshots. Share
        its snapshot slots so collected and non-collected local steps can be
        interleaved within the same parameter-sync cycle.
        """
        worker = self.actor_rollout_wg
        if self.config.trainer.v1.get("trainer_mode", "sync") != "separate_async":
            return worker.compute_log_prob(control)

        local_step = self.local_trigger_step
        if local_step == 0:
            worker.save_model_to_cpu(0)
            return worker.compute_log_prob(control)

        worker.save_model_to_cpu(local_step)
        try:
            worker.restore_model_from_cpu(0)
            return worker.compute_log_prob(control)
        finally:
            # Restore before releasing the only copy of the training weights;
            # if restoration fails, leave that snapshot available for recovery.
            worker.restore_model_from_cpu(local_step)
            worker.clear_cpu_model(local_step)

    def _speco_v1_spec_decode_sidecar_metrics(self) -> dict[str, float]:
        """Read cumulative EngineCore counters when RequestOutput has no stats."""

        from verl_speco.integration.vllm_runtime import (
            SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY,
            read_vllm_spec_decode_sidecar_totals,
        )

        run_dir = getattr(
            getattr(getattr(self, "config", None), "trainer", None),
            "default_local_dir",
            None,
        )
        actor_rollout_ref = getattr(self.config, "actor_rollout_ref", None)
        rollout = getattr(actor_rollout_ref, "rollout", {}) or {}
        engine_kwargs = rollout.get("engine_kwargs", {}) or {}
        vllm_kwargs = engine_kwargs.get("vllm", {}) or {}
        additional_config = vllm_kwargs.get("additional_config", {}) or {}
        directory = additional_config.get(SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY)
        if not directory and run_dir:
            # Compatibility fallback for launchers configured before per-run
            # sidecar directories were introduced.
            directory = os.path.join(os.fspath(run_dir), ".spec_decode_stats")
        current = read_vllm_spec_decode_sidecar_totals(directory)
        previous = self.__dict__.get(
            "_speco_vllm_spec_decode_sidecar_previous",
            {"drafts": 0.0, "accepted_tokens": 0.0, "draft_tokens": 0.0},
        )
        self._speco_vllm_spec_decode_sidecar_previous = current
        drafts = max(0.0, current["drafts"] - float(previous.get("drafts", 0.0)))
        accepted = max(
            0.0,
            current["accepted_tokens"] - float(previous.get("accepted_tokens", 0.0)),
        )
        if drafts <= 0.0:
            return {}
        return {
            "drafter/spec_decode/mean_acceptance_length": 1.0 + accepted / drafts,
        }

    def _speco_v1_spec_decode_metrics(self, batch: Any) -> dict[str, float]:
        """Aggregate SpeCo's vLLM acceptance counters for one V1 global step.

        V1's upstream metric collector only fetches ``extra_fields`` when the
        upstream MTP feature is enabled.  SpeCo speculative decoding is
        configured independently, so its counters would otherwise remain in
        TransferQueue and never reach the trainer logger.
        """
        try:
            import transfer_queue as tq
        except ImportError:
            return self._speco_v1_spec_decode_sidecar_metrics()

        try:
            stats_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = stats_data.pop("extra_fields", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SPECO V1 could not read speculative metrics: %s", exc)
            return self._speco_v1_spec_decode_sidecar_metrics()

        if hasattr(extra_fields, "tolist"):
            extra_fields = extra_fields.tolist()
        if not isinstance(extra_fields, list):
            return self._speco_v1_spec_decode_sidecar_metrics()

        total_drafts = 0.0
        total_accepted = 0.0
        tags = list(getattr(batch, "tags", None) or [])
        for index, fields in enumerate(extra_fields):
            if index < len(tags) and bool(tags[index].get("is_padding", False)):
                continue
            # TransferQueue returns object-backed values for ``extra_fields``
            # in the live V1 replay-buffer path.  They expose the user dict
            # through ``.data`` rather than being dict instances themselves.
            # The initial bridge only covered plain dicts, silently dropping
            # every real rollout counter.
            fields = getattr(fields, "data", fields)
            if not isinstance(fields, dict):
                continue
            try:
                # These are the native verl rollout fields.  In particular,
                # ToolAgentLoop knows to accumulate them across tool turns.
                drafts = float(fields.get("spec_num_verify_steps", 0.0) or 0.0)
                accepted = float(fields.get("spec_num_accepted_tokens", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            total_drafts += max(0.0, drafts)
            total_accepted += max(0.0, accepted)

        if total_drafts <= 0.0:
            return self._speco_v1_spec_decode_sidecar_metrics()
        return {
            "drafter/spec_decode/mean_acceptance_length": 1.0
            + total_accepted / total_drafts,
        }

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        """Preserve upstream V1 metrics and add SpeCo speculative decoding."""
        result = super()._compute_metrics(
            batch, metrics, timing_raw, global_steps, epoch
        )
        metrics.update(self._speco_v1_spec_decode_metrics(batch))
        if getattr(self, "_speco_freeze_policy", None) is not None:
            metrics.update(self._speco_freeze_metrics())
        if getattr(self, "_speco_ci_result", None) is not None:
            # Shared training/publication facades already record their own
            # outcomes; count only generation here to avoid double counting.
            self._speco_ci_result.record_metrics({"training/global_step": global_steps})
        return result

    def _update_actor(self, batch, metrics):
        if not self._speco_online_enabled_from_config(self.config):
            return super()._update_actor(batch, metrics)

        # A V1 global step can contain several local actor updates when
        # ``parameter_sync_step > 1``.  Those updates all share one global
        # weight version and one drafter publication boundary, so training the
        # drafter at every local update would perform multiple optimizer steps
        # for a single global step.  Let the preceding local updates collect
        # their features into the data buffer, then schedule exactly once after
        # the final actor update.
        parameter_sync_step = max(int(getattr(self, "parameter_sync_step", 1)), 1)
        local_trigger_step = int(getattr(self, "local_trigger_step", 0))
        if local_trigger_step < parameter_sync_step - 1:
            metrics["drafter/training_deferred_to_global_step_end"] = 1
            return super()._update_actor(batch, metrics)

        event = self._speco_on_before_actor_update()
        plan = event.training_plan
        metrics.update(event.metrics or {})
        result = super()._update_actor(batch, metrics)
        pending_sync = getattr(self, "_pending_target_lm_head_sync", None)
        if pending_sync is not None:
            metrics.update(self._speco_finish_target_lm_head_weight_sync(pending_sync))
            self._pending_target_lm_head_sync = None
        if plan is not None:
            if plan.launch:
                started = time.perf_counter()
                trained, train_metrics = self._speco_train_drafter(plan)
                self._speco_v1_train_cost_seconds = time.perf_counter() - started
            else:
                trained = False
                train_metrics = {
                    "drafter/trained": 0,
                    "drafter/train_successful_steps_max": 0,
                    "drafter/train_no_trainable_batch": int(
                        plan.reason == "no_trainable_batch"
                    ),
                    "drafter/train_activation_failed": 0,
                }
            metrics.update(train_metrics)
            self._speco_v1_pending_training = bool(trained)
            self._speco_v1_training_plan = plan
        if getattr(self, "_speco_freeze_policy", None) is not None:
            self._speco_maybe_save_freeze_branch_checkpoint()
            self._speco_flush_request_accept_len_payloads()
        return result

    def _speco_async_prefit_rollout_warmup_enabled(self) -> bool:
        v1_config = self.config.trainer.v1
        mode = str(v1_config.get("trainer_mode", "sync")).lower()
        return mode in {"colocate_async", "separate_async"} and bool(
            v1_config.get("pre_fit_rollout_warmup", True)
        )

    def _speco_run_async_prefit_rollout_warmup(self, agent_loop_manager) -> None:
        """Finish V1's initial rollout batch before the timed training loop.

        Upstream async trainers submit ``num_warmup_batches`` from
        ``on_train_begin`` but enter step 1 immediately.  The first step then
        absorbs vLLM/MRV2's first real decode and waits for the replay buffer.
        Run the same hooks at the upcoming global step and wait for one full
        training batch here; the batch remains in TQ and is consumed normally
        by step 1, so this changes timing boundaries rather than training data.
        """

        if not self._speco_async_prefit_rollout_warmup_enabled():
            return

        from verl.utils.skip import SkipManager

        self.agent_loop_manager = agent_loop_manager
        next_global_step = int(getattr(self, "global_steps", 0)) + 1
        SkipManager.init(self.config)
        SkipManager.set_step(next_global_step)

        original_global_steps = self.global_steps
        started = time.perf_counter()
        try:
            self.global_steps = next_global_step
            upstream = cast(Any, super())
            reissued = int(upstream._reissue_inflight_prompts() or 0)
            upstream.on_train_begin()
        finally:
            self.global_steps = original_global_steps

        mode = str(self.config.trainer.v1.get("trainer_mode", "sync")).lower()
        mode_config = self.config.trainer.v1.get(mode, {}) or {}
        skip_rollout = bool(self.config.skip.rollout_tq.get("enable", False))
        warmup_batches = (
            0 if skip_rollout else int(mode_config.get("num_warmup_batches", 0) or 0)
        )
        submitted = warmup_batches * int(self.config.data.train_batch_size)
        target_count = min(
            max(reissued + submitted, 0), int(self.config.data.train_batch_size)
        )
        if target_count > 0:
            self.replay_buffer.wait_for_sampleable(
                next_global_step, "train", target_count
            )

        # Upstream fit() invokes these hooks after incrementing global_steps.
        # They have already run for that exact step, so consume those calls once.
        self._speco_prefit_reissue_consumed = True
        self._speco_prefit_on_train_begin_consumed = True
        logger.info(
            "SPECO V1 async rollout pre-fit warmup completed: step=%s, "
            "reissued=%s, submitted=%s, sampleable_target=%s, elapsed=%.3fs",
            next_global_step,
            reissued,
            submitted,
            target_count,
            time.perf_counter() - started,
        )

    def prepare_for_fit(self, agent_loop_manager) -> None:
        """Run expensive V1 activation/warmup before entering trainer.fit()."""

        if bool(getattr(self, "_speco_prepared_for_fit", False)):
            return
        self._speco_init_ci_result()
        if self._speco_online_enabled_from_config(self.config):
            self._speco_activate_drafter_training_model_before_fit()
            self._speco_freeze_policy = self._speco_init_freeze_policy()
            self._speco_validate_freeze_branch_config()
            if self._speco_colocate_probe_boundary_enabled():
                if self._speco_freeze_policy is None:
                    raise RuntimeError("V1 async freeze requires an initialized policy")
                if self._speco_freeze_branch_config().get("enabled", False):
                    from verl.trainer.ppo.v1.trainer_base import _tq_supports_checkpoint

                    if not _tq_supports_checkpoint():
                        raise RuntimeError(
                            "V1 async freeze branch requires TransferQueue checkpoint/reissue support"
                        )
                self._speco_rollout_admission_rpc(
                    "speco_set_serving_drafter_revision",
                    self._speco_freeze_policy.drafter_version,
                )
        if self._speco_v1_async_rollout_enabled():
            self._speco_run_async_prefit_rollout_warmup(agent_loop_manager)
        self._speco_prepared_for_fit = True

    def _reissue_inflight_prompts(self, *args, **kwargs):
        if bool(getattr(self, "_speco_prefit_reissue_consumed", False)):
            self._speco_prefit_reissue_consumed = False
            return 0
        return super()._reissue_inflight_prompts(*args, **kwargs)

    def on_train_begin(self):
        if bool(getattr(self, "_speco_prefit_on_train_begin_consumed", False)):
            self._speco_prefit_on_train_begin_consumed = False
            logger.info(
                "SPECO V1 skipped duplicate in-fit warmup submission; "
                "pre-fit rollout batch is already sampleable"
            )
            return None
        return super().on_train_begin()

    def _speco_init_ci_result(self):
        if (
            os.getenv("SPECO_CI_RESULT_PATH")
            and getattr(self, "_speco_ci_result", None) is None
        ):
            from verl_speco.integration.ci_result import CIResult

            self._speco_ci_result = CIResult(
                self._speco_online_enabled_from_config(self.config)
            )

    def fit(self, agent_loop_manager):
        result_path = os.getenv("SPECO_CI_RESULT_PATH")
        if not result_path:
            return self._speco_fit_impl(agent_loop_manager)
        self._speco_init_ci_result()
        try:
            result = self._speco_fit_impl(agent_loop_manager)
            self._speco_ci_result.data["completed"] = True
            return result
        except BaseException as error:
            self._speco_ci_result.data["errors"].append(
                {"type": type(error).__name__, "message": str(error)}
            )
            raise
        finally:
            self._speco_ci_result.write(result_path)

    def _speco_fit_impl(self, agent_loop_manager):
        completed = False
        try:
            if not bool(getattr(self, "_speco_prepared_for_fit", False)):
                # Compatibility fallback for callers other than SpecoTaskRunner.
                self.prepare_for_fit(agent_loop_manager)
            result = super().fit(agent_loop_manager)
            completed = True
            return result
        finally:
            if self._speco_v1_async_rollout_enabled():
                # ``fit`` may return or raise while async agent-loop actors
                # still own vLLM requests or output handlers.  Drain them
                # before any rollout/DataLoader teardown in either path.
                cancelled = False
                if not completed:
                    import ray

                    refs = []
                    for worker in (
                        getattr(agent_loop_manager, "agent_loop_workers", ()) or ()
                    ):
                        cancel = getattr(worker, "speco_cancel_pending", None)
                        if cancel is not None:
                            refs.append(cancel.remote())
                    if refs:
                        ray.get(refs)
                        cancelled = True
                if not cancelled:
                    self._speco_v1_drain_agent_loop(agent_loop_manager)
            if self._speco_online_enabled_from_config(self.config):
                self._speco_wait_pending_drafter_publish()
                self._speco_wait_pending_drafter_checkpoint()
            self._speco_v1_shutdown_dataloaders()

    def speco_v1_status(self) -> dict[str, Any]:
        """Return diagnostics used by smoke tests and startup logging."""

        return {
            "trainer_mode": str(self.trainer_mode),
            "adapter": type(self).__name__,
            "features_collected": int(
                getattr(self, "_speco_v1_state", {}).get("features_collected", 0)
            ),
        }
