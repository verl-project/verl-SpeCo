# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU/Gloo checks with real rank processes and two concurrent process groups."""

from __future__ import annotations

from datetime import timedelta
import sys
import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader, TQPrefetchIterator


class DistributedStore:
    def __init__(self, rank, mode):
        self.rank, self.mode = rank, mode

    def connect(self):
        pass

    def owner_ready(self):
        return True

    def get_many(self, entries):
        if self.mode == "failure" and self.rank == 1:
            raise ValueError("bad sample on rank 1")
        return [SimpleNamespace(input_ids=torch.tensor([self.rank]))]


def _worker(rank, rendezvous, mode):
    options = dist.ProcessGroupGloo._Options()
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=15), pg_options=options)
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=5), pg_options=options)
    commands = Queue()
    queued = []
    if mode != "empty":
        for index in range(5):
            queued.append({"kind": "batch", "plan_id": index,
                          "global_keys": [f"k{index}-0", f"k{index}-1"],
                          "assignments": [[{"key": f"k{index}-{r}", "tag": {"sequence_no": index}}]
                                          for r in range(2)]})
        queued.append({"kind": "stop"})
    def publish():
        for item in queued:
            commands.put(item)
    timer = None
    if mode == "delayed" and rank == 0:
        timer = threading.Timer(6, publish)
        timer.start()
    else:
        publish()
    loader = TQFeatureDataLoader(DistributedStore(rank, mode), batch_size=1,
                                 rank=rank, world_size=2, scheduled_commands=commands,
                                 process_group=group)
    source = TQPrefetchIterator(loader, 2, close_timeout=20)
    try:
        if mode == "failure":
            with pytest.raises(RuntimeError, match="prefetch failed"):
                next(source)
        elif mode == "full":
            with source._condition:
                assert source._condition.wait_for(lambda: len(source._batches) == 2, timeout=10)
        elif mode in {"drain", "delayed"}:
            for index, batch in enumerate(source):
                assert batch.plan_id == index
                assert batch.local_keys == [f"k{index}-{rank}"]
                # Represents model collectives on the training group, while
                # background metadata collectives run on their own group.
                value = torch.tensor(rank + 1)
                dist.all_reduce(value)
                assert value.item() == 3
            assert index == 4
        dist.barrier()
    finally:
        source.close()
        if timer is not None:
            timer.cancel()
            timer.join(timeout=2)
        dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.skipif(sys.platform == "win32", reason="Run real Gloo ranks on the Linux training platform")
@pytest.mark.parametrize("mode", ["drain", "full", "empty", "failure", "delayed"])
def test_two_rank_prefetch_lifecycle(mode):
    import tempfile
    from pathlib import Path
    from urllib.parse import unquote

    directory = tempfile.TemporaryDirectory(prefix="tq-prefetch-")
    rendezvous = unquote((Path(directory.name) / "rendezvous").resolve().as_uri())
    context = mp.spawn(_worker, args=(rendezvous, mode), nprocs=2, join=False)
    try:
        import time

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail("Two-rank prefetch did not finish within 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        directory.cleanup()
