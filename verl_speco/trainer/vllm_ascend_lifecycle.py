# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Address-preserving state release behind the native drafter lifecycle hooks."""

import socket
from contextlib import contextmanager

import torch
import torch.distributed as dist

from verl_speco.trainer.ascend_state_parking import (
    AscendStateParking,
    available_host_memory,
)


class AscendTrainingLifecycle:
    def __init__(
        self,
        *,
        control_group=None,
        parking=None,
        parking_factory=None,
        gather=None,
        host_available=None,
        device_free=None,
        host=None,
        host_reserve_bytes=16 * 1024**3,
    ):
        self.control_group = control_group
        if gather is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Ascend state lifecycle requires initialized distributed workers"
                )
            if self.control_group is None:
                self.control_group = dist.new_group(backend="gloo")
            self.gather = self._gather
        else:
            self.gather = gather
        self.host_available = host_available or available_host_memory
        self.device_free = device_free or (lambda: torch.npu.mem_get_info()[0])
        self.host = host or socket.gethostname()
        self.host_reserve_bytes = host_reserve_bytes
        self.failed = False
        self.last_status = {"state": "new"}
        self.parking = (
            parking
            if parking is not None
            else self._settle("allocator_setup", parking_factory or AscendStateParking)
        )

    @property
    def state(self):
        return self.parking.state

    @property
    def coverage_inventory(self):
        return self.parking.coverage_inventory

    def _gather(self, value):
        results = [None] * dist.get_world_size(self.control_group)
        dist.all_gather_object(results, value, group=self.control_group)
        return results

    def _settle(self, phase, function):
        value, error = None, None
        try:
            value = function()
        except Exception as exception:  # noqa: BLE001 -- every rank settles before state transitions
            error = repr(exception)
        try:
            failures = self.gather(error)
        except BaseException:
            self.failed = True
            raise
        if any(failures):
            self.failed = True
            raise RuntimeError(f"Ascend state lifecycle {phase} failed: {failures}")
        return value

    @contextmanager
    def allocation_scope(self):
        if self.failed:
            raise RuntimeError("Ascend lifecycle failed; resume from a checkpoint")
        try:
            with self.parking.allocation_scope():
                yield
        except BaseException:
            self.failed = True
            raise

    def check_coverage(self, engine):
        return self.parking.check_coverage(engine)

    def park(self, engine):
        if self.failed:
            raise RuntimeError("Cannot park failed training state")
        if self.state == "parked":
            return self.last_status["managed_bytes"]
        if self.state != "active":
            raise RuntimeError("Training state is not active")
        # Every rank must finish device work before any rank can unmap storage.
        self._settle("quiesce", self.parking.synchronize)
        size = self._settle("coverage", lambda: self.parking.check_coverage(engine))
        budget = self._settle(
            "host_memory",
            lambda: {
                "host": self.host,
                "managed_bytes": size,
                "host_bytes": getattr(
                    self.parking, "host_snapshot_bytes", lambda value: value
                )(size),
                "available": self.host_available(),
                "disk": getattr(self.parking, "storage_budget", lambda value: None)(
                    size
                ),
            },
        )
        hosts, disks = {}, {}
        for record in self.gather(budget):
            entry = hosts.setdefault(
                record["host"], {"bytes": 0, "available": record["available"]}
            )
            entry["bytes"] += record["host_bytes"]
            entry["available"] = min(entry["available"], record["available"])
            disk = record.get("disk")
            if disk:
                entry = disks.setdefault(
                    disk["storage"], {"bytes": 0, "available": disk["available"]}
                )
                entry["bytes"] += disk["required"]
                entry["available"] = min(entry["available"], disk["available"])
        if any(
            d["bytes"] + self.host_reserve_bytes > d["available"]
            for d in disks.values()
        ):
            self.failed = True
            raise RuntimeError(
                f"Insufficient scratch disk capacity before parking: {disks}"
            )
        if any(
            h["bytes"] + self.host_reserve_bytes > h["available"]
            for h in hosts.values()
        ):
            self.failed = True
            raise RuntimeError(f"Insufficient host memory before parking: {hosts}")
        prepare = getattr(self.parking, "prepare_park", None)
        if prepare is not None:
            try:
                self._settle("snapshot_write", lambda: prepare(engine))
            except BaseException:
                self.parking.abort_prepared()
                raise
            size = self._settle("park", self.parking.commit_park)
        else:
            size = self._settle("park", lambda: self.parking.park(engine))
        free = self._settle("device_memory", self.device_free)
        self.last_status = {
            "state": self.state,
            "managed_bytes": size,
            "device_free_bytes": free,
            "allocation_inventory": self.coverage_inventory,
            "host_snapshot_bytes": budget["host_bytes"],
            "scratch_disk_bytes": budget["disk"]["required"] if budget["disk"] else 0,
        }
        return size

    def wake(self, engine):
        if self.failed:
            raise RuntimeError("Cannot wake failed training state")
        if self.state == "active":
            return
        if self.state != "parked":
            raise RuntimeError("Training state is not parked")

        def capacity():
            needed = self.last_status["managed_bytes"]
            free = self.device_free()
            if free < needed:
                raise RuntimeError(
                    f"Training needs {needed} bytes, only {free} are free; "
                    "release actor/rollout device memory before activation or checkpoint"
                )

        self._settle("wake_capacity", capacity)
        self._settle("wake", lambda: self.parking.wake(engine))
        self.last_status = {**self.last_status, "state": self.state}

    @contextmanager
    def active_operation(self, engine):
        """Exports/checkpoints can read state only while its storage is mapped."""
        was_parked = self.state == "parked"
        if was_parked:
            self.wake(engine)
        if self.failed or self.state != "active":
            raise RuntimeError("Training state cannot be read while unmapped or failed")
        try:
            yield
        finally:
            if was_parked and not self.failed:
                self.park(engine)
