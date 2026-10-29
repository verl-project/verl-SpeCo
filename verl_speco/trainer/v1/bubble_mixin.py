# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""V1 lifecycle integration for resource-leased Bubble training."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle

logger = logging.getLogger(__name__)


class V1BubbleMixin:
    if TYPE_CHECKING:
        # SpecoV1Mixin supplies the dynamically bound shared trainer facade.
        def __getattr__(self, name: str) -> Any: ...

    def _speco_rollout_idle_worker_enabled(self) -> bool:
        from verl_speco.trainer.scheduler import DrafterScheduleConfig

        if getattr(self, "config", None) is None:
            return False
        actor = (
            self.config.get("actor_rollout_ref")
            if hasattr(self.config, "get")
            else getattr(self.config, "actor_rollout_ref", None)
        )
        if actor is None:
            return False
        drafter = actor.rollout.get("drafter", {}) or {}
        return bool(
            drafter.get("enable", False)
            and drafter.get("enable_drafter_training", False)
            and DrafterScheduleConfig.from_mapping(
                drafter.get("training", {})
            ).execution_strategy.value
            == "rollout_idle_worker"
        )

    def _speco_bubble_session(self) -> BubbleLifecycle:
        session = self.__dict__.get("_speco_bubble_lifecycle")
        if session is None:
            session = BubbleLifecycle(self)
            self._speco_bubble_lifecycle = session
        return session

    def _speco_bubble_roi_supported(self) -> bool:
        return self._speco_rollout_idle_worker_enabled()

    def _speco_emit_rollout_generation_started(self):
        from verl_speco.trainer.scheduler.schedule_types import (
            RolloutWorkerEvent,
            RolloutWorkerEventType,
        )

        if not self._speco_rollout_idle_worker_enabled():
            return {}
        scheduler = self._speco_get_drafter_scheduler()
        groups = scheduler.rollout_idle_replica_groups()
        if not groups:
            raise RuntimeError(
                "V1 Bubble generation requires registered physical replica groups"
            )
        metrics = {}
        for replica_rank, workers in groups.items():
            metrics.update(
                scheduler.on_worker_event(
                    RolloutWorkerEvent(
                        RolloutWorkerEventType.GENERATION_STARTED,
                        worker_id=workers[0],
                        replica_rank=replica_rank,
                    )
                )
            )
        return metrics

    def _speco_v1_separate_bubble(self) -> bool:
        return (
            self._speco_rollout_idle_worker_enabled()
            and str(self.config.trainer.v1.get("trainer_mode", "sync"))
            == "separate_async"
        )

    def _speco_v1_serving_managers(self):
        managers = [("hybrid", self.llm_server_manager)]
        if str(self.config.trainer.v1.get("trainer_mode", "sync")) == "separate_async":
            managers.append(("standalone", self.standalone_server_manager))
        return managers

    def _speco_v1_probe_servers(self):
        if self._speco_v1_separate_bubble():
            return list(self.standalone_server_manager.server_handles)
        return list(self.llm_server_manager.server_handles)

    def _speco_v1_publish_managers(self):
        managers = self._speco_v1_serving_managers()
        if (
            self._speco_v1_separate_bubble()
            and getattr(self.current_mode, "value", None) == "trainer"
        ):
            return [item for item in managers if item[0] == "standalone"]
        return managers

    def _speco_v1_active_checkpoint_managers(self):
        if not self._speco_v1_separate_bubble():
            return [self.checkpoint_manager]
        managers = [self.standalone_checkpoint_manager]
        if getattr(self.current_mode, "value", None) != "trainer":
            managers.insert(0, self.checkpoint_manager)
        return managers

    def _speco_v1_bubble_publish_method(self, method_name):
        import ray
        from verl.single_controller.ray.base import RayClassWithInitArgs, RayWorkerGroup
        from verl_speco.trainer.v1.bubble_workers import normalize_publish_ack

        normalizer = ray.remote(num_cpus=0)(normalize_publish_ack)
        endpoints = []
        expected = set()
        for name, manager in self._speco_v1_publish_managers():
            replicas = manager.get_replicas()
            if not replicas or any(not replica.workers for replica in replicas):
                raise RuntimeError(
                    "Bubble publication has an empty serving worker domain"
                )
            if name == "hybrid":
                operation = self._speco_actor_rollout_method(method_name)
                count = sum(len(replica.workers) for replica in replicas)
                endpoints.append((name, count, operation))
                expected.update(f"{name}:{rank}" for rank in range(count))
            else:
                for replica in replicas:
                    namespace = f"{name}:{replica.replica_rank}"
                    group = RayWorkerGroup(
                        worker_handles=replica.workers,
                        ray_cls_with_init=RayClassWithInitArgs(
                            cls=self._speco_v1_standalone_publish_worker_cls(),
                        ),
                    )
                    endpoints.append(
                        (namespace, len(replica.workers), getattr(group, method_name))
                    )
                    expected.update(
                        f"{namespace}:{rank}" for rank in range(len(replica.workers))
                    )
        self._speco_v1_publish_expected = expected

        def submit(*args, **kwargs):
            refs = []
            for name, count, operation in endpoints:
                responses = operation(*args, **kwargs)
                if not isinstance(responses, (list, tuple)) or len(responses) != count:
                    raise RuntimeError(
                        "Bubble publish RPC returned an incomplete worker domain"
                    )
                refs.extend(
                    normalizer.remote(response, name, rank)
                    for rank, response in enumerate(responses)
                )
            return refs

        return submit

    def _speco_validate_bubble_publish_acks(
        self, acknowledgements, source_version, *, staged=False
    ):
        if not self._speco_v1_separate_bubble():
            from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

            return SpecoRayPPOTrainer._speco_validate_bubble_publish_acks(
                self,
                acknowledgements,
                source_version,
                staged=staged,
            )
        acks = self._speco_get_drafter_scheduler()._flatten_publish_acknowledgements(
            acknowledgements
        )
        seen = set()
        for ack in acks:
            identity = str(ack.get("worker_id"))
            if (
                identity in seen
                or ack.get("staged_version" if staged else "published_version")
                != source_version
                or not ack.get("staged" if staged else "published")
                or (staged and ack.get("published"))
            ):
                raise RuntimeError("Invalid or duplicate separate Bubble publish ACK")
            seen.add(identity)
        if not seen or seen != self._speco_v1_publish_expected:
            raise RuntimeError(
                "Separate Bubble publish ACKs do not cover all serving workers"
            )

    def _speco_init_v1_standalone_bubble_workers(self):
        from verl_speco.trainer.v1.bubble_workers import create_standalone_bubble_group

        self.attach_speco_worker_group(create_standalone_bubble_group(self))

    def _speco_park_drafter_training_memory(self, boundary):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        if not self._speco_v1_separate_bubble():
            return SpecoRayPPOTrainer._speco_park_drafter_training_memory(
                self, boundary
            )
        # Standalone GPUs belong to serving outside the lease, including when
        # the shared configuration requests a hot model between training plans.
        started = time.perf_counter()
        results = self._ray_get_if_needed(
            self._require_speco_worker_group().park_drafter_training_model()
        )
        items = self._speco_flatten_checkpoint_results(results)
        scheduler = self._speco_get_drafter_scheduler()
        scheduler.record_memory_park_elapsed(
            max(
                (float(item.get("elapsed_sec", 0) or 0) for item in items),
                default=0,
            )
        )
        scheduler._idle_worker_hot_prewarmed_groups.clear()
        return {
            f"timing_s/drafter_memory_park_{boundary}": time.perf_counter() - started
        }

    def speco_get_drafter_training_resource_metadata(self):
        records = self._ray_get_if_needed(
            self._require_speco_worker_group().get_drafter_training_resource_metadata(),
        )
        if not self._speco_rollout_idle_worker_enabled():
            return records
        import ray
        from verl_speco.integration.resource_identity import map_training_replicas

        replicas = {
            replica.replica_rank: ray.get(
                [worker.speco_resource_identity.remote() for worker in replica.workers]
            )
            for replica in (
                self.standalone_server_manager
                if self._speco_v1_separate_bubble()
                else self.llm_server_manager
            ).get_replicas()
        }
        return map_training_replicas(records, replicas)

    def _speco_register_drafter_training_resource_metadata(self):
        from verl_speco.integration.rollout_idle_events import (
            configure_rollout_resource_groups,
        )
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        metrics = SpecoRayPPOTrainer._speco_register_drafter_training_resource_metadata(
            self
        )
        if self._speco_v1_separate_bubble() and not configure_rollout_resource_groups(
            self._speco_rollout_idle_event_bus_name(),
            self._speco_get_drafter_scheduler().rollout_idle_replica_groups(),
            strict_groups=True,
        ):
            raise RuntimeError(
                "Separate V1 Bubble requires isolated standalone resource leases"
            )
        return metrics

    def _speco_v1_admission(self, method: str) -> None:
        import ray

        refs = [
            getattr(server, method).remote()
            for _, manager in self._speco_v1_serving_managers()
            for server in manager.server_handles
        ]
        if not refs:
            raise RuntimeError("Bubble boundary has no serving replicas")
        ray.get(refs)

    @contextmanager
    def _speco_v1_quiescent_boundary(self, *, charge_cost=True):
        """Pause ordinary requests, but allow dedicated probe RPCs."""
        already_closed = bool(getattr(self, "_speco_colocate_admission_closed", False))
        self._speco_bubble_session().close("v1_publish")
        pause_started = time.perf_counter()
        pause_wall_started = time.time()
        if not already_closed:
            self._speco_v1_admission("speco_close_rollout_admission")
            self._speco_colocate_admission_closed = True
        completed = False
        try:
            managers = [self.checkpoint_manager]
            if self._speco_v1_separate_bubble():
                managers.append(self.standalone_checkpoint_manager)
            for manager in managers:
                # Abort partial requests so the idle wait cannot wait forever
                # on a paused request; TQ retains the upstream retry ownership.
                manager.abort_replicas()
            self._speco_v1_admission("speco_wait_rollout_idle")
            if charge_cost:
                self._speco_bubble_session().costs.record_interval(
                    f"pause:{self.global_steps}",
                    pause_wall_started,
                    time.time(),
                )
            yield
            completed = True
        finally:
            if completed and (
                not already_closed
                or getattr(self, "_speco_v1_training_admission_owned", False)
            ):
                resume_wall_started = time.time()
                self._speco_v1_admission("speco_resume_rollout_admission")
                self._speco_colocate_admission_closed = False
                self._speco_v1_training_admission_owned = False
                receipts = self._speco_bubble_session().receipts
                self._speco_v1_last_pause_seconds = time.perf_counter() - pause_started
                if receipts and charge_cost:
                    receipts[-1]["costs"].record_interval(
                        f"resume:{self.global_steps}",
                        resume_wall_started,
                        time.time(),
                    )

    def switch_to_rollout(self):
        if not self._speco_v1_separate_bubble():
            return super().switch_to_rollout()
        from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode

        # Keep hybrid replicas out of the balancer until the last acknowledged
        # drafter snapshot has been applied as well as the actor target.
        with self._speco_v1_quiescent_boundary():
            self.checkpoint_manager.update_weights(self.global_steps)
            payload = getattr(self, "_speco_last_published_drafter_payload", None)
            version = getattr(self, "_speco_last_published_drafter_step", None)
            if payload is not None:
                acknowledgements = self._ray_get_if_needed(
                    self._speco_actor_rollout_method("update_draft_weights")(
                        payload,
                        global_steps=version,
                    ),
                )
                expected = {
                    str(rank) for rank in range(self.actor_rollout_wg.world_size)
                }
                items = self._speco_get_drafter_scheduler()._flatten_publish_acknowledgements(
                    acknowledgements
                )
                if (
                    len(items) != len(expected)
                    or {
                        str(item.get("worker_id", item.get("worker_rank")))
                        for item in items
                    }
                    != expected
                    or any(
                        not item.get("published")
                        or item.get("published_version") != version
                        for item in items
                    )
                ):
                    raise RuntimeError(
                        "Hybrid rollout cannot rejoin with stale drafter weights"
                    )
            policy = getattr(self, "_speco_freeze_policy", None)
            if policy is not None:
                import ray

                ray.get(
                    [
                        server.speco_set_serving_drafter_revision.remote(
                            policy.drafter_version
                        )
                        for server in self.llm_server_manager.server_handles
                    ]
                )
            self.checkpoint_manager.resume_generation_replicas()
            self.current_mode = HybridEngineMode.ROLLOUT
            self.add_replicas_to_balancer()
        self._speco_bubble_session().start()

    def _speco_v1_cache_old_policy_head(self) -> None:
        if not self._speco_rollout_idle_worker_enabled():
            return
        # Called inside separate_async's pi_old save/restore context. Materialize
        # the export before pi_old is restored away, not at the next actor hook.
        wall_started = time.time()
        metrics, pending = self._speco_start_target_lm_head_weight_sync(None)
        if pending is not None:
            self._pending_target_lm_head_sync = pending
            self._ray_get_if_needed(pending.get("fetch_refs", pending.get("refs")))
            metrics.update(self._speco_poll_target_lm_head_prefetch())
            self._speco_bubble_session().costs.record_interval(
                f"head:{self.global_steps}",
                wall_started,
                time.time(),
            )
        self._speco_record_rollout_idle_metrics(metrics)

    def _speco_finish_target_lm_head_weight_sync(self, pending):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        result = SpecoRayPPOTrainer._speco_finish_target_lm_head_weight_sync(
            self, pending
        )
        if self._speco_rollout_idle_worker_enabled():
            self._speco_record_target_head_transfer_cost(pending)
        return result

    def _speco_record_target_head_transfer_cost(self, pending):
        origin = pending.get("sync_started", time.perf_counter())
        started = time.time() - max(0.0, time.perf_counter() - origin)
        version = pending.get(
            "target_version", pending.get("global_step", self.global_steps)
        )
        # Different transfers of one version incur additional work. Duplicate
        # synchronous/polled completions share the original dispatch identity.
        self._speco_bubble_session().costs.record_interval(
            f"head_transfer:{version}:{origin}",
            started,
            time.time(),
        )

    def _speco_dispatch_target_lm_head_payload(self, payload, **kwargs):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        metrics, pending = SpecoRayPPOTrainer._speco_dispatch_target_lm_head_payload(
            self,
            payload,
            **kwargs,
        )
        if pending is not None:
            pending["sync_started"] = kwargs["sync_started"]
        return metrics, pending

    def _speco_poll_target_lm_head_prefetch(self):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        # The polling state machine consumes dispatch ACKs directly, without
        # entering the synchronous finish hook. Keep one origin across stages.
        pending = self._pending_target_lm_head_sync
        result = SpecoRayPPOTrainer._speco_poll_target_lm_head_prefetch(self)
        if pending is not None:
            if self._pending_target_lm_head_sync is not None:
                self._pending_target_lm_head_sync.setdefault(
                    "sync_started", pending["sync_started"]
                )
            elif result.get("drafter/target_lm_head_synced"):
                self._speco_record_target_head_transfer_cost(pending)
        return result

    def _speco_activate_drafter_training_model_before_fit(self):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        started = time.time()
        result = SpecoRayPPOTrainer._speco_activate_drafter_training_model_before_fit(
            self
        )
        if self._speco_rollout_idle_worker_enabled():
            self._speco_bubble_session().costs.record_interval(
                "bootstrap", started, time.time()
            )
        return result

    def _speco_poll_pending_drafter_training(self):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        plan, outcome = SpecoRayPPOTrainer._speco_poll_pending_drafter_training(self)
        if (
            plan is not None
            and outcome is not None
            and self._speco_rollout_idle_worker_enabled()
        ):
            self._speco_bubble_session().costs.observe_training(plan, outcome)
        return plan, outcome

    def _speco_wait_pending_drafter_training(self):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        plan, outcome = SpecoRayPPOTrainer._speco_wait_pending_drafter_training(self)
        if (
            plan is not None
            and outcome is not None
            and self._speco_rollout_idle_worker_enabled()
        ):
            self._speco_bubble_session().costs.observe_training(plan, outcome)
        return plan, outcome

    def _speco_freeze_observe_publish(self, metrics, **kwargs):
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        if self.__dict__.get("_speco_v1_publish_transaction", False):
            # The common ACK path notifies before the after-arm exists. The
            # transaction supplies the sole version-clock observation below.
            self._speco_v1_publish_ack = bool(metrics.get("drafter/published"))
            return
        return SpecoRayPPOTrainer._speco_freeze_observe_publish(self, metrics, **kwargs)

    def _speco_v1_publish_bubble(self) -> dict[str, Any]:
        session = self._speco_bubble_session()
        metrics: dict[str, Any] = {}
        while session.pending:
            plan, outcome = session.pending[0]
            policy = getattr(self, "_speco_freeze_policy", None)
            version_before = (
                policy.drafter_version + len(session.receipts)
                if policy is not None
                else 0
            )
            run_probe = bool(
                policy is not None and policy.probe_due(version_before + 1)
            )
            before = comparison = None
            probe_seconds = 0.0
            self._speco_v1_publish_transaction = True
            self._speco_v1_publish_ack = False
            try:
                if run_probe:
                    started = time.perf_counter()
                    probe_wall_started = time.time()
                    try:
                        self._speco_v1_probe_expected_revision = version_before
                        before = self._speco_run_freeze_probe_arm()
                    except Exception:
                        metrics["bubble/probe_before_failed"] = 1
                    probe_seconds += time.perf_counter() - started
                    session.costs.record_interval(
                        f"probe_before:{plan.plan_id}", probe_wall_started, time.time()
                    )
                publish_wall_started = time.time()
                publication = self._speco_publish_drafter_weights(
                    outcome.trained,
                    plan,
                    after_weight_update=True,
                )
                self._speco_wait_pending_drafter_publish()
                session.costs.record_interval(
                    f"publish:{plan.plan_id}", publish_wall_started, time.time()
                )
                if not self._speco_v1_publish_ack and not (
                    policy is None and publication.get("drafter/published")
                ):
                    raise RuntimeError(
                        "Bubble publication did not complete its apply ACK"
                    )
                if policy is not None:
                    import ray

                    ray.get(
                        [
                            server.speco_set_serving_drafter_revision.remote(
                                version_before + 1
                            )
                            for _, manager in self._speco_v1_publish_managers()
                            for server in manager.server_handles
                        ]
                    )
                if run_probe and before is not None:
                    started = time.perf_counter()
                    probe_wall_started = time.time()
                    try:
                        self._speco_v1_probe_expected_revision = version_before + 1
                        comparison = self._speco_build_freeze_probe_comparison(
                            before,
                            self._speco_run_freeze_probe_arm(),
                            version_before=version_before,
                        )
                    except Exception:
                        metrics["bubble/probe_after_failed"] = 1
                    probe_seconds += time.perf_counter() - started
                    session.costs.record_interval(
                        f"probe_after:{plan.plan_id}", probe_wall_started, time.time()
                    )
                metrics.update(publication)
                metrics["bubble/quota_update_cost_seconds"] = session.costs.total
                metrics["bubble/quota_probe_seconds"] = probe_seconds
            finally:
                self._speco_v1_publish_transaction = False
                self._speco_v1_probe_expected_revision = None
            session.receipts.append({"costs": session.costs, "probe": comparison})
            session.published(plan.plan_id)
        return metrics

    def _speco_v1_finish_receipts(self):
        session = self._speco_bubble_session()
        metrics = {}
        while session.receipts:
            receipt = session.receipts[0]
            self._speco_freeze_observe_publish(
                {"drafter/published": 1},
                update_cost_seconds=receipt["costs"].total,
                probe=receipt["probe"],
            )
            decision = getattr(self, "_speco_last_freeze_decision", None)
            if decision is not None:
                if getattr(self, "_speco_freeze_policy_active", False):
                    self._speco_drafter_frozen = not bool(decision.should_train)
                self._speco_freeze_emit_transition(decision)
                self._speco_record_freeze_branch_request(decision)
            metrics["bubble/quota_update_cost_seconds"] = receipt["costs"].total
            session.receipts.pop(0)
        if getattr(self, "_speco_freeze_policy", None) is not None:
            metrics.update(self._speco_freeze_metrics())
        return metrics

    def _speco_v1_bubble_step_end(self, parent):
        boundary_started = time.perf_counter()
        session = self._speco_bubble_session()
        metrics = session.close("v1_target_sync")
        scheduler = self._speco_get_drafter_scheduler()
        final_debt = (
            self.global_steps >= self.total_training_steps
            and scheduler._training_quota_debt_steps > 0
        )
        topup_plan = None
        if not final_debt and scheduler._training_quota_debt_steps > 0:
            config = self._speco_drafter_schedule_config()
            if config.training_quota_allow_critical_path_fallback:
                # Preserve the existing bounded assistance/freshness policy;
                # only this explicit configuration may manufacture a pause.
                topup_plan = scheduler._plan_training_quota_topup(
                    self._speco_drafter_schedule_context(),
                    config,
                )
        if (
            not session.pending
            and not final_debt
            and topup_plan is None
            and getattr(self, "_speco_deferred_checkpoint_step", None) is None
            and not getattr(self, "_speco_colocate_admission_closed", False)
        ):
            # Native target synchronization already coordinates its own pool.
            # A plain actor update does not need a global Bubble probe drain.
            self._speco_wait_target_lm_head_prefetch()
            result = parent()
            metrics.update(scheduler.training_work_metrics())
            self._pending_sync_metrics = {
                **(getattr(self, "_pending_sync_metrics", None) or {}),
                **metrics,
            }
            if (
                self.global_steps < self.total_training_steps
                and self._speco_v1_separate_bubble()
            ):
                session.start()
            return result
        with self._speco_v1_quiescent_boundary():
            self._speco_wait_target_lm_head_prefetch()
            if topup_plan is not None:
                for manager in (
                    [self.checkpoint_manager, self.standalone_checkpoint_manager]
                    if self._speco_v1_separate_bubble()
                    else [self.checkpoint_manager]
                ):
                    manager.sleep_replicas()
                metrics.update(scheduler.prepare_training_execution(topup_plan))
                started = time.perf_counter()
                trained, train_metrics = self._speco_train_drafter(topup_plan)
                from types import SimpleNamespace

                session.enqueue(
                    topup_plan,
                    SimpleNamespace(
                        trained=trained,
                        raw_results=[],
                        elapsed_sec=time.perf_counter() - started,
                    ),
                )
                metrics.update(train_metrics)
            # Final quotas have no future rollout in which to repay their debt.
            while (
                self.global_steps >= self.total_training_steps
                and scheduler._training_quota_debt_steps > 0
            ):
                debt_before = scheduler._training_quota_debt_steps
                if not self._speco_drafter_schedule_config().training_quota_allow_critical_path_fallback:
                    raise RuntimeError(
                        "Final V1 Bubble quota remains incomplete; critical-path fallback is disabled"
                    )
                managers = [self.checkpoint_manager]
                if self._speco_v1_separate_bubble():
                    managers.append(self.standalone_checkpoint_manager)
                for manager in managers:
                    manager.sleep_replicas()
                event = self._speco_on_before_actor_update(force_quota_completion=True)
                plan = event.training_plan
                if plan is None or not plan.launch:
                    raise RuntimeError(
                        "Final V1 Bubble quota cannot be completed safely"
                    )
                started = time.perf_counter()
                trained, train_metrics = self._speco_train_drafter(plan)
                from types import SimpleNamespace

                session.enqueue(
                    plan,
                    SimpleNamespace(
                        trained=trained,
                        raw_results=[],
                        elapsed_sec=time.perf_counter() - started,
                    ),
                )
                metrics.update(train_metrics)
                if not trained or scheduler._training_quota_debt_steps >= debt_before:
                    raise RuntimeError("Final V1 Bubble quota remains incomplete")
            if topup_plan is not None or final_debt:
                metrics.update(self._speco_park_drafter_training_memory("v1_topup"))
            # The native mode owns actor target synchronization. Admission is
            # closed across both probe arms, even if native code resumes engines.
            result = parent()
            metrics.update(self._speco_v1_publish_bubble())
        metrics.update(self._speco_v1_finish_receipts())
        metrics["bubble/probe_pause_seconds"] = getattr(
            self, "_speco_v1_last_pause_seconds", 0.0
        )
        metrics["bubble/step_end_critical_path_seconds"] = (
            time.perf_counter() - boundary_started
        )
        metrics.update(scheduler.training_work_metrics())
        if getattr(self, "_speco_deferred_checkpoint_step", None) is not None:
            # Persist the acknowledged version and its ROI observation together.
            # The first boundary includes resume costs before committing policy.
            with self._speco_v1_quiescent_boundary(charge_cost=False):
                for manager in (
                    [self.checkpoint_manager, self.standalone_checkpoint_manager]
                    if self._speco_v1_separate_bubble()
                    else [self.checkpoint_manager]
                ):
                    manager.sleep_replicas()
                checkpoint_started = time.perf_counter()
                self._save_checkpoint()
                self._speco_v1_record_timing(
                    "drafter_v1_checkpoint", time.perf_counter() - checkpoint_started
                )
                # Wake the serving engines with the same already-synchronized
                # target before opening admission. No probe spans this update.
                parent()
        if getattr(self, "_speco_freeze_branch_save_pending", None) is not None:
            checkpoint_started = time.perf_counter()
            self._speco_maybe_save_freeze_branch_checkpoint()
            self._speco_v1_record_timing(
                "drafter_v1_checkpoint", time.perf_counter() - checkpoint_started
            )
        self._pending_sync_metrics = {
            **(getattr(self, "_pending_sync_metrics", None) or {}),
            **metrics,
        }
        if (
            self.global_steps < self.total_training_steps
            and self._speco_v1_separate_bubble()
        ):
            session.start()
        return result

    def _speco_v1_idle_option(self, name, default):
        training = self.config.actor_rollout_ref.rollout.drafter.get("training", {})
        return training.get("scheduler", {}).get("idle_worker", {}).get(name, default)

    def _speco_v1_boundary_diagnostic(self, event, boundary):
        if not self._speco_v1_idle_option("boundary_diagnostics_enable", False):
            return
        session = self._speco_bubble_session()
        logger.warning(
            "[BubbleTime] v1_boundary: step=%s event=%s boundary=%s "
            "wall_ts=%.6f event_thread_active=%s generation_open=%s",
            self.global_steps,
            event,
            boundary,
            time.time(),
            session.thread is not None,
            getattr(self, "_speco_rollout_idle_generation_open", False),
        )

    def _speco_v1_log_runtime_provenance(self):
        """Identify this driver's sources once, before creating rollout workers."""
        import hashlib
        import json
        import subprocess
        from pathlib import Path

        from verl_speco.trainer import bubble_lifecycle
        from verl_speco.trainer.v1 import speco_mixin

        sources = {}
        for filename in (__file__, bubble_lifecycle.__file__, speco_mixin.__file__):
            path = Path(filename).resolve()
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                digest = "unavailable"
            sources[str(path)] = digest
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            commit = (
                revision.stdout.strip() if revision.returncode == 0 else "unavailable"
            )
        except (OSError, subprocess.TimeoutExpired):
            commit = "unavailable"
        report = {
            "commit": commit,
            "source_sha256": sources,
            "mode": str(self.config.trainer.v1.get("trainer_mode", "sync")),
            "head_preparation_requested": bool(
                self._speco_v1_idle_option("v1_head_preparation_enable", False)
            ),
            "head_preparation_effective": False,
            "boundary_diagnostics_enable": bool(
                self._speco_v1_idle_option("boundary_diagnostics_enable", False)
            ),
        }
        # A single startup record must survive module logger filters. No RPC,
        # device read, or per-step subprocess is added by this diagnostic.
        print(
            "[BubbleTime] v1_runtime: " + json.dumps(report, sort_keys=True), flush=True
        )

    def _speco_v1_bubble_update_actor(self, batch, metrics, update):
        started = time.perf_counter()
        separate = self._speco_v1_separate_bubble()
        if not separate:
            metrics.update(self._speco_bubble_session().close("v1_before_actor"))
        # Export was fenced inside the old-policy collection context.
        if self._pending_target_lm_head_sync is not None:
            metrics.update(self._speco_poll_target_lm_head_prefetch())
        actor_started = time.perf_counter()
        self._speco_v1_record_timing("drafter_v1_before_actor", actor_started - started)
        self._speco_v1_boundary_diagnostic("actor_begin", "v1_update_actor")
        try:
            result = update(batch, metrics)
        except BaseException:
            self._speco_v1_boundary_diagnostic("actor_failed", "v1_update_actor")
            raise
        self._speco_v1_boundary_diagnostic("actor_end", "v1_update_actor")
        after_started = time.perf_counter()
        self._speco_v1_record_timing(
            "actor_update_without_drafter", after_started - actor_started
        )
        metrics.update(self._speco_pop_rollout_idle_metrics())
        self._speco_v1_record_timing(
            "drafter_v1_after_actor", time.perf_counter() - after_started
        )
        return result

    def _speco_v1_record_timing(self, name, seconds):
        if self.__dict__.get("_speco_v1_timing_step") != self.global_steps:
            self._speco_v1_timing_step = self.global_steps
            self._speco_v1_timings = {}
        timings = self._speco_v1_timings
        timings[name] = timings.get(name, 0.0) + max(0.0, float(seconds))

    def _speco_v1_timing_metrics(self):
        if self.__dict__.get("_speco_v1_timing_step") != self.global_steps:
            return {}
        timings = self._speco_v1_timings
        metrics = {f"timing_s/{name}": value for name, value in timings.items()}
        metrics["timing_s/drafter"] = sum(
            timings.get(name, 0.0)
            for name in (
                "drafter_v1_sample_end",
                "drafter_v1_before_actor",
                "drafter_v1_after_actor",
                "drafter_v1_publish_blocking",
            )
        )
        return metrics

    def step(self, metrics, timing_raw):
        if self._speco_rollout_idle_worker_enabled():
            if not self._speco_v1_separate_bubble():
                # Collection can be skipped while a quota remains outstanding.
                # Advance the worker's control step before opening events, but
                # leave the quota's data/head version pinned to its collection.
                self._speco_set_drafter_global_step()
            self._speco_bubble_session().start()
        return super().step(metrics, timing_raw)

    def on_sample_begin(self):
        result = super().on_sample_begin()
        if self._speco_rollout_idle_worker_enabled():
            self._speco_bubble_session().start()
        return result

    def on_validate_begin(self):
        if self._speco_rollout_idle_worker_enabled():
            self._speco_bubble_session().close("v1_validation")
            self._speco_validation_in_progress = True
        try:
            return super().on_validate_begin()
        except BaseException:
            self._speco_validation_in_progress = False
            raise

    def on_validate_end(self):
        try:
            return super().on_validate_end()
        finally:
            if self._speco_rollout_idle_worker_enabled():
                from verl_speco.integration.rollout_idle_events import (
                    drain_rollout_idle_events,
                )

                drain_rollout_idle_events(self._speco_rollout_idle_event_bus_name())
                self._speco_validation_in_progress = False
