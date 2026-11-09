# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""V1 lifecycle integration for resource-leased Bubble training."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from verl_speco.trainer.bubble_lifecycle import BUBBLE_RUNTIME_FIELDS, BubbleLifecycle

logger = logging.getLogger(__name__)


class _BubbleRuntimeField:
    """Keep legacy facade attribute access backed by one V1 session store."""

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def _storage(owner):
        instance = vars(owner)
        session = instance.get("_speco_bubble_lifecycle")
        return getattr(session, "runtime_state", instance)

    def __get__(self, owner, owner_type=None):
        if owner is None:
            return self
        try:
            return self._storage(owner)[self.name]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, owner, value):
        storage = self._storage(owner)
        storage[self.name] = value
        if storage is not vars(owner):
            vars(owner).pop(self.name, None)

    def __delete__(self, owner):
        storage = self._storage(owner)
        try:
            del storage[self.name]
        except KeyError:
            raise AttributeError(self.name) from None
        if storage is not vars(owner):
            vars(owner).pop(self.name, None)


class V1BubbleMixin:
    _speco_session_state_owner = True

    if TYPE_CHECKING:
        # SpecoV1Mixin supplies the dynamically bound shared trainer facade.
        def __getattr__(self, name: str) -> Any: ...

    def _speco_rollout_idle_worker_enabled(self) -> bool:
        from verl_speco.trainer.scheduler import DrafterScheduleConfig

        cached = self.__dict__.get("_speco_v1_bubble_enabled_cache")
        if cached is not None:
            return cached
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
        enabled = bool(
            drafter.get("enable", False)
            and drafter.get("enable_drafter_training", False)
            and DrafterScheduleConfig.from_mapping(
                drafter.get("training", {})
            ).execution_strategy.value
            == "rollout_idle_worker"
        )
        if (
            enabled
            and str(self.config.trainer.v1.get("trainer_mode", "sync")) != "sync"
        ):
            raise ValueError("V1 Bubble Time supports only trainer_mode=sync")
        return enabled

    def _speco_bubble_session(self) -> BubbleLifecycle:
        session = self.__dict__.get("_speco_bubble_lifecycle")
        if session is None:
            session = BubbleLifecycle(self)
            self._speco_bubble_lifecycle = session
        return session

    def _speco_bubble_clock(self):
        from verl_speco.trainer.bubble_lifecycle import CostClock

        session = self._speco_bubble_session()
        if not hasattr(session, "clock"):
            session.clock = CostClock()
        return session.clock

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

    def _speco_v1_probe_servers(self):
        return list(self.llm_server_manager.server_handles)

    def _speco_park_drafter_training_memory(self, boundary):
        if not self._speco_rollout_idle_worker_enabled():
            return {}
        # V1 sync returns leased GPUs to serving or PPO at these boundaries.
        # keep_hot_between_plans is only valid inside a resource-owned window;
        # it must not bypass parking across actor updates or rollout admission.
        # Preserve CPU optimizer/replay state and the fixed quota writer.
        session = self._speco_bubble_session()
        if getattr(session, "memory_parked", False):
            return {}
        started = time.perf_counter()
        results = self._ray_get_if_needed(
            self._require_speco_worker_group().park_drafter_training_model()
        )
        items = self._speco_flatten_checkpoint_results(results)
        session.memory_parked = True
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
            for replica in self.llm_server_manager.get_replicas()
        }
        return map_training_replicas(records, replicas)

    def _speco_v1_admission(self, method: str) -> None:
        import math

        import ray

        timeout = float(self._speco_v1_idle_option("admission_timeout_seconds", 300))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "Bubble admission_timeout_seconds must be finite and positive"
            )
        changes_state = method in (
            "speco_close_rollout_admission",
            "speco_resume_rollout_admission",
        )
        if changes_state:
            previous = self.__dict__.get("_speco_v1_admission_mutation")
            if previous is not None:
                previous_method, previous_refs = previous
                if previous_refs:
                    _, pending = ray.wait(
                        previous_refs, num_returns=len(previous_refs), timeout=timeout
                    )
                    if pending:
                        raise TimeoutError(
                            f"Previous Bubble admission {previous_method} is still in flight"
                        )
        refs: list[Any] = []
        if changes_state:
            self._speco_v1_admission_mutation = (method, refs)
        for server in self.llm_server_manager.server_handles:
            refs.append(getattr(server, method).remote())
        if not refs:
            raise RuntimeError("Bubble boundary has no serving replicas")
        ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        ready_set = set(ready)
        for index, ref in enumerate(refs):
            if ref in ready_set:
                try:
                    ray.get(ref)
                except Exception as exc:
                    raise RuntimeError(
                        f"Bubble admission {method} failed on serving replica index {index}"
                    ) from exc
        if pending:
            missing = [index for index, ref in enumerate(refs) if ref not in ready_set]
            raise TimeoutError(
                f"Bubble admission {method} timed out after {timeout:g}s "
                f"on serving replica indices {missing}"
            )
        if changes_state:
            self.__dict__.pop("_speco_v1_admission_mutation", None)

    @contextmanager
    def _speco_v1_checkpoint_replicas_asleep(self, *, record_restore=False):
        """Restore target weights before the enclosing admission boundary can open."""
        try:
            self.checkpoint_manager.sleep_replicas()
            yield
        finally:
            started = time.perf_counter()
            try:
                self.checkpoint_manager.update_weights(self.global_steps)
            except Exception as exc:
                raise RuntimeError(
                    "V1 Bubble checkpoint rollout weight restore failed; admission remains closed"
                ) from exc
            finally:
                if record_restore:
                    self._speco_v1_record_timing(
                        "checkpoint_rollout_restore", time.perf_counter() - started
                    )

    @contextmanager
    def _speco_v1_quiescent_boundary(self, *, charge_cost=True, timings=None):
        """Close admission and drain sync serving RPCs without pausing vLLM.

        abort_replicas() can leave vLLM >= 0.12 generation paused. The naive
        sync weight update does not resume it, and probes inside this boundary
        need a runnable engine. Sync has no concurrent rollout producer here;
        admitted requests finish before weights or probe state are changed.
        """
        already_closed = bool(getattr(self, "_speco_colocate_admission_closed", False))
        self._speco_bubble_session().close("v1_publish")
        pause_started = time.perf_counter()
        pause_wall_started = self._speco_bubble_clock().now()
        if not already_closed:
            self._speco_v1_admission("speco_close_rollout_admission")
            self._speco_colocate_admission_closed = True
        completed = False
        try:
            self._speco_v1_admission("speco_wait_rollout_idle")
            if timings is not None:
                timings["timing_s/drafter_v1_admission_close_and_drain"] = (
                    time.perf_counter() - pause_started
                )
            if charge_cost:
                self._speco_bubble_session().costs.record_interval(
                    f"pause:{self.global_steps}",
                    pause_wall_started,
                    self._speco_bubble_clock().now(),
                )
            yield
            completed = True
        finally:
            if completed and not already_closed:
                resume_wall_started = self._speco_bubble_clock().now()
                resume_started = time.perf_counter()
                self._speco_colocate_admission_closed = False
                self._speco_v1_admission("speco_resume_rollout_admission")
                if timings is not None:
                    timings["timing_s/drafter_v1_admission_resume"] = (
                        time.perf_counter() - resume_started
                    )
                receipts = self._speco_bubble_session().receipts
                self._speco_v1_last_pause_seconds = time.perf_counter() - pause_started
                if receipts and charge_cost:
                    receipts[-1]["costs"].record_interval(
                        f"resume:{self.global_steps}",
                        resume_wall_started,
                        self._speco_bubble_clock().now(),
                    )

    def _speco_v1_cache_old_policy_head(self) -> None:
        if not self._speco_rollout_idle_worker_enabled():
            return
        # Materialize the sync collection head before the next actor update.
        wall_started = self._speco_bubble_clock().now()
        metrics, pending = self._speco_start_target_lm_head_weight_sync(None)
        if pending is not None:
            self._pending_target_lm_head_sync = pending
            self._ray_get_if_needed(pending.get("fetch_refs", pending.get("refs")))
            metrics.update(self._speco_poll_target_lm_head_prefetch())
            self._speco_bubble_session().costs.record_interval(
                f"head:{self.global_steps}",
                wall_started,
                self._speco_bubble_clock().now(),
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
        started = self._speco_bubble_clock().from_monotonic(origin)
        version = pending.get(
            "target_version", pending.get("global_step", self.global_steps)
        )
        # Different transfers of one version incur additional work. Duplicate
        # synchronous/polled completions share the original dispatch identity.
        self._speco_bubble_session().costs.record_interval(
            f"head_transfer:{version}:{origin}",
            started,
            self._speco_bubble_clock().now(),
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

        if not self._speco_rollout_idle_worker_enabled():
            return SpecoRayPPOTrainer._speco_activate_drafter_training_model_before_fit(
                self
            )
        self._speco_bubble_session().memory_parked = False
        started = self._speco_bubble_clock().now()
        result = SpecoRayPPOTrainer._speco_activate_drafter_training_model_before_fit(
            self
        )
        if self._speco_rollout_idle_worker_enabled():
            self._speco_bubble_session().costs.record_interval(
                "bootstrap", started, self._speco_bubble_clock().now()
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
                applied = self.__dict__.get("_speco_v1_applied_publish")
                if applied is None:
                    if run_probe:
                        started = time.perf_counter()
                        probe_wall_started = self._speco_bubble_clock().now()
                        try:
                            self._speco_v1_probe_expected_revision = version_before
                            before = self._speco_run_freeze_probe_arm()
                        except Exception:
                            metrics["bubble/probe_before_failed"] = 1
                        probe_seconds += time.perf_counter() - started
                        session.costs.record_interval(
                            f"probe_before:{plan.plan_id}",
                            probe_wall_started,
                            self._speco_bubble_clock().now(),
                        )
                    publish_wall_started = self._speco_bubble_clock().now()
                    publication = self._speco_publish_drafter_weights(
                        outcome.trained,
                        plan,
                        after_weight_update=True,
                    )
                    wait_started = time.perf_counter()
                    self._speco_wait_pending_drafter_publish()
                    wait_key = "timing_s/drafter_v1_publish_pending_wait"
                    metrics[wait_key] = (
                        metrics.get(wait_key, 0.0) + time.perf_counter() - wait_started
                    )
                    session.costs.record_interval(
                        f"publish:{plan.plan_id}",
                        publish_wall_started,
                        self._speco_bubble_clock().now(),
                    )
                    if not self._speco_v1_publish_ack and not (
                        policy is None and publication.get("drafter/published")
                    ):
                        raise RuntimeError(
                            "Bubble publication did not complete its apply ACK"
                        )
                    self._speco_v1_applied_publish = (
                        plan.plan_id,
                        version_before,
                        before,
                        publication,
                        probe_seconds,
                    )
                else:
                    (
                        applied_plan,
                        applied_version,
                        before,
                        publication,
                        probe_seconds,
                    ) = applied
                    if (applied_plan, applied_version) != (
                        plan.plan_id,
                        version_before,
                    ):
                        raise RuntimeError(
                            "Bubble applied publication does not match the pending plan"
                        )
                if policy is not None:
                    import ray

                    revision_started = time.perf_counter()
                    refs: list[Any] | None = self.__dict__.get(
                        "_speco_v1_revision_refs"
                    )
                    if refs is None:
                        refs = []
                        self._speco_v1_revision_refs = refs
                        self._speco_v1_revision_submitted = False
                        for server in self.llm_server_manager.server_handles:
                            refs.append(
                                server.speco_set_serving_drafter_revision.remote(
                                    version_before + 1
                                )
                            )
                        self._speco_v1_revision_submitted = True
                    timeout = float(
                        self._speco_v1_idle_option("admission_timeout_seconds", 300)
                    )
                    if not refs:
                        self.__dict__.pop("_speco_v1_revision_refs", None)
                        raise RuntimeError(
                            "Bubble revision sync has no serving replicas"
                        )
                    _, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
                    if pending:
                        raise TimeoutError("Bubble serving revision sync timed out")
                    self.__dict__.pop("_speco_v1_revision_refs", None)
                    ray.get(refs)
                    if not self._speco_v1_revision_submitted:
                        raise RuntimeError(
                            "Bubble serving revision broadcast was incomplete"
                        )
                    revision_key = "timing_s/drafter_v1_publish_serving_revision"
                    metrics[revision_key] = (
                        metrics.get(revision_key, 0.0)
                        + time.perf_counter()
                        - revision_started
                    )
                if run_probe and before is not None:
                    started = time.perf_counter()
                    probe_wall_started = self._speco_bubble_clock().now()
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
                        f"probe_after:{plan.plan_id}",
                        probe_wall_started,
                        self._speco_bubble_clock().now(),
                    )
                metrics.update(publication)
                metrics["bubble/quota_update_cost_seconds"] = session.costs.total
                metrics["bubble/quota_probe_seconds"] = probe_seconds
            finally:
                self._speco_v1_publish_transaction = False
                self._speco_v1_probe_expected_revision = None
            session.receipts.append({"costs": session.costs, "probe": comparison})
            session.published(plan.plan_id)
            self.__dict__.pop("_speco_v1_applied_publish", None)
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
            # The policy has consumed this publish. Later hooks must not replay it.
            session.receipts.pop(0)
            decision = getattr(self, "_speco_last_freeze_decision", None)
            if decision is not None:
                if getattr(self, "_speco_freeze_policy_active", False):
                    self._speco_drafter_frozen = not bool(decision.should_train)
                self._speco_record_freeze_branch_request(decision)
                self._speco_freeze_emit_transition(decision)
            metrics["bubble/quota_update_cost_seconds"] = receipt["costs"].total
        if getattr(self, "_speco_freeze_policy", None) is not None:
            metrics.update(self._speco_freeze_metrics())
        return metrics

    def _speco_v1_prefetch_publish_snapshot(self) -> dict[str, Any]:
        """Serialize cached CPU weights while native target synchronization runs.

        Do not stage/apply rollout weights here: paired probes and publication
        ACKs still belong to the post-target-sync transaction. The shared
        publisher consumes these same refs instead of exporting a second time.
        """
        config = self._speco_drafter_schedule_config()
        if not config.idle_worker_early_publish_staging or not config.publish_async:
            return {}
        session = self._speco_bubble_session()
        scheduler = self._speco_get_drafter_scheduler()
        if (
            len(session.pending) != 1
            or scheduler._training_quota_debt_steps > 0
            or not getattr(self, "_speco_colocate_admission_closed", False)
            or getattr(self, "_pending_drafter_publish_refs", None)
            or getattr(self, "_speco_validation_in_progress", False)
            or getattr(self, "_speco_checkpoint_in_progress", False)
        ):
            return {}
        plan, outcome = session.pending[0]
        if (
            not outcome.trained
            or not plan.launch
            or not plan.publish_after_success
            or int(plan.source_global_step) != int(self.global_steps)
        ):
            return {}
        publish_plan = scheduler.plan_publish(
            global_step=self.global_steps,
            drafter_trained=outcome.trained,
            config=config,
            training_plan=plan,
        )
        if not publish_plan.publish:
            return {}
        prefetched = getattr(self, "_speco_publish_snapshot_prefetch", None)
        if prefetched is not None:
            if (
                int(prefetched["source_step"]) != int(plan.source_global_step)
                or prefetched.get("plan_id", plan.plan_id) != plan.plan_id
            ):
                raise RuntimeError(
                    "Bubble snapshot prefetch belongs to another publication"
                )
            return {}
        export = getattr(
            self._require_speco_worker_group(), "maybe_publish_async", None
        )
        if not callable(export):
            return {}
        started = time.perf_counter()
        # This RPC can only consume a cached CPU snapshot. On a cache miss the
        # original publisher exports at the safe point after native target sync.
        refs = export(source_step=int(plan.source_global_step))
        self._speco_publish_snapshot_prefetch = {
            "refs": refs,
            "source_step": int(plan.source_global_step),
            "plan_id": plan.plan_id,
        }
        return {
            "bubble/publish_snapshot_prefetch_submitted": 1,
            "timing_s/drafter_publish_snapshot_prefetch_submit": time.perf_counter()
            - started,
        }

    @contextmanager
    def _speco_v1_record_topup_cost(self, phase):
        session = self._speco_bubble_session()
        clock = self._speco_bubble_clock()
        started = clock.now()
        yield
        session.costs.record_interval(
            f"topup:{self.global_steps}:{phase}", started, clock.now()
        )

    def _speco_v1_bubble_step_end(self, parent):
        boundary_started = time.perf_counter()
        session = self._speco_bubble_session()
        metrics = session.close("v1_target_sync")
        if session.receipts:
            # Reconcile acknowledged publications before planning or taking
            # the fast path. The current boundary's receipts are handled below.
            metrics.update(self._speco_v1_finish_receipts())
        scheduler = self._speco_get_drafter_scheduler()
        final_debt = (
            self.global_steps >= self.total_training_steps
            and scheduler._training_quota_debt_steps > 0
        )
        topup_plan = None
        optimizer_bootstrap = False
        if not final_debt and scheduler._training_quota_debt_steps > 0:
            config = self._speco_drafter_schedule_config()
            if config.training_quota_allow_critical_path_fallback:
                optimizer_bootstrap = scheduler.needs_idle_optimizer_bootstrap(config)
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
            return result
        with self._speco_v1_quiescent_boundary(timings=metrics):
            started = time.perf_counter()
            self._speco_wait_target_lm_head_prefetch()
            metrics["timing_s/drafter_v1_target_head_prefetch_wait"] = (
                time.perf_counter() - started
            )
            if topup_plan is not None:
                started = time.perf_counter()
                with self._speco_v1_record_topup_cost("sleep"):
                    self.checkpoint_manager.sleep_replicas()
                metrics["timing_s/drafter_v1_topup_replica_sleep"] = (
                    time.perf_counter() - started
                )
                started = time.perf_counter()
                metrics.update(scheduler.prepare_training_execution(topup_plan))
                metrics["timing_s/drafter_v1_topup_prepare"] = (
                    time.perf_counter() - started
                )
                started = time.perf_counter()
                trained, train_metrics = self._speco_train_drafter(topup_plan)
                elapsed = time.perf_counter() - started
                if optimizer_bootstrap:
                    metrics["timing_s/drafter_v1_optimizer_bootstrap"] = elapsed
                from types import SimpleNamespace

                session.enqueue(
                    topup_plan,
                    SimpleNamespace(
                        trained=trained,
                        raw_results=[],
                        elapsed_sec=elapsed,
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
                with self._speco_v1_record_topup_cost(f"final_sleep:{debt_before}"):
                    self.checkpoint_manager.sleep_replicas()
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
                with self._speco_v1_record_topup_cost("park"):
                    metrics.update(self._speco_park_drafter_training_memory("v1_topup"))
            # The native mode owns actor target synchronization. Admission is
            # closed across both probe arms, even if native code resumes engines.
            metrics.update(self._speco_v1_prefetch_publish_snapshot())
            result = parent()
            started = time.perf_counter()
            metrics.update(self._speco_v1_publish_bubble())
            metrics["timing_s/drafter_v1_publish_total"] = time.perf_counter() - started
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
                with self._speco_v1_checkpoint_replicas_asleep(record_restore=True):
                    checkpoint_started = time.perf_counter()
                    self._save_checkpoint()
                    self._speco_v1_record_timing(
                        "drafter_v1_checkpoint",
                        time.perf_counter() - checkpoint_started,
                    )
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
            self._speco_bubble_clock().now(),
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


# Only the shared facade's declared fields are intercepted. Data descriptors
# keep raw instance-dictionary entries from shadowing a live Bubble session.
for _field_name in BUBBLE_RUNTIME_FIELDS:
    setattr(V1BubbleMixin, _field_name, _BubbleRuntimeField(_field_name))
