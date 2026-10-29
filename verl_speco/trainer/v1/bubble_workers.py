# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Build one distributed drafter group in existing standalone rollout pools."""

from __future__ import annotations


def create_standalone_bubble_group(trainer):
    import ray
    from verl.single_controller.ray.base import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )
    from verl_speco.workers import SpecoWorker

    replicas = trainer.standalone_server_manager.get_replicas()
    if not replicas:
        raise RuntimeError("Separate Bubble requires standalone rollout replicas")
    pools = [replica.resource_pool for replica in replicas]
    placements = [pg for pool in pools for pg in pool.get_placement_groups()]
    counts = [pg.bundle_count for pg in placements]
    if not counts or len(set(counts)) != 1:
        raise ValueError(
            "V1 Bubble requires uniform accelerator bundles per placement group"
        )
    pool = RayResourcePool(
        process_on_nodes=counts,
        use_gpu=True,
        max_colocate_count=max(source.max_colocate_count for source in pools),
        name_prefix="speco_standalone_bubble",
    )
    pool.pgs = placements

    class ReplicaOrderedWorkerGroup(RayWorkerGroup):
        def _init_with_resource_pool(
            self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None
        ):
            # Upstream sorts placement groups by IP. That can interleave two
            # multi-node replicas and silently build the wrong SP collective.
            self.resource_pool = resource_pool
            self._world_size = resource_pool.world_size
            self._get_master_addr_port(
                placements[0],
                bundle_index=0,
                master_port_range=self._ray_master_port_range,
            )
            rank = 0
            for pg_index, pg in enumerate(placements):
                for local_rank in range(pg.bundle_count):
                    self._create_worker(
                        rank,
                        pg_index,
                        pg,
                        local_rank,
                        resource_pool,
                        ray_cls_with_init,
                        worker_env,
                        detached,
                    )
                    rank += 1

    worker_cls = trainer.speco_worker_cls or SpecoWorker
    if not hasattr(worker_cls, "__ray_actor_class__"):
        worker_cls = ray.remote(worker_cls)
    init = RayClassWithInitArgs(
        cls=worker_cls,
        config=trainer.config.actor_rollout_ref,
        role="drafter",
        device_name=trainer.config.trainer.device,
        require_resource_identity=True,
        require_lease_offload=True,
    )
    # These are cooperative tenants in an already reserved placement group.
    # The GPU fraction is assigned by RayWorkerGroup; runtime leases govern
    # actual model residency. Do not request another dedicated CPU per tenant.
    init.update_options({"num_cpus": 0})
    group = ReplicaOrderedWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=init,
        name_prefix="speco_v1_standalone_drafter",
        device_name=trainer.config.trainer.device,
    )
    group.init_model()
    return group


def normalize_publish_ack(ack, namespace, expected_rank):
    if not isinstance(ack, dict) or str(
        ack.get("worker_id", ack.get("worker_rank", ack.get("rank")))
    ) != str(expected_rank):
        raise RuntimeError("V1 Bubble publish ACK came from an unexpected worker")
    return {**ack, "worker_id": f"{namespace}:{expected_rank}"}
