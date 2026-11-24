# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU test support for standalone training workers."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest


@contextmanager
def gloo_process_group(rank, rendezvous, *, timeout, world_size=2):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@contextmanager
def gloo_workers(context, target, arguments, *, count=2, release=None):
    import socket
    import torch.distributed as dist

    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    started = []
    try:
        for rank in range(count):
            process = context.Process(target=target, args=arguments(rank, port))
            process.start()
            started.append(process)
        yield started
    finally:
        if release is not None:
            release.set()
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def join_workers(processes, timeout):
    for process in processes:
        process.join(timeout=timeout)
        assert process.exitcode == 0, f"{process.name} exited with {process.exitcode}"


def collect_gloo_workers(target, arguments, *, count=2, timeout=45, join_timeout=10):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    try:
        with gloo_workers(
            context,
            target,
            lambda rank, port: arguments(rank, port, results),
            count=count,
        ) as processes:
            rows = [results.get(timeout=timeout) for _ in processes]
            join_workers(processes, join_timeout)
        assert {row[0] for row in rows} == set(range(count))
        return sorted(rows, key=lambda row: row[0])
    finally:
        results.close()


def make_cpu_trainer(model, *, lr, distributed):
    import torch
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    def reduce_statistics(values):
        if distributed:
            torch.distributed.all_reduce(values)
            return values, torch.distributed.get_world_size()
        return values, 1

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model, trainer.runtime_device = model, torch.device("cpu")
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer.lr_scheduler = None
    trainer.optimizer_steps_total = trainer.training_steps = 0
    trainer._standalone_validate_grad_graph = True
    trainer.combine_standalone_metrics = False
    trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
    trainer.record_training_timing = lambda *args: None
    trainer._ulysses_group_context = nullcontext
    trainer._record_dflash_training_metrics = lambda losses: {}
    trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
    trainer._is_checkpoint_leader = lambda: False
    trainer._reduce_loss_statistics = reduce_statistics
    return trainer


def dspark_reference_backend(head):
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=head))
    backend.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "mode": "offline",
                    "feature_store": {"type": "ray_reference"},
                    "transfer_queue": {
                        "expected_feature": {"payload_kind": "vllm_reference"}
                    },
                }
            )
        )
    )
    return backend


class NativeTestEngine:
    def initialize(self, *args):
        return 0

    def register_memory(self, *args):
        return 0

    def unregister_memory(self, *args):
        return 0


class NativeCopyTestEngine(NativeTestEngine):
    def transfer_sync_read(self, segment, local, remote, size):
        import ctypes

        ctypes.memmove(local, remote, size)
        return 0
