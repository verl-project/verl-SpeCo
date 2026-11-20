# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Packed SP metadata preserves peer order, duplicates, dtype and empty ranks."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist

from verl_speco.integration import oldlogprob_runtime as runtime


@pytest.mark.parametrize("counts", [(3, 1), (0, 2), (0, 0), (1, 0, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_gather_matches_independent_peer_concatenation(
    monkeypatch, counts, dtype
):
    peers = []
    for rank, count in enumerate(counts):
        peers.append(
            {
                "rows": (torch.arange(count * 4).reshape(count, 4) + rank * 100).to(
                    dtype
                ),
                "batch_indices": torch.arange(count, dtype=torch.long) % 2,
                "row_indices": torch.arange(count, dtype=torch.long) % 2,
            }
        )
    group = object()
    calls = []
    maximum = max(counts)

    def reduce(tensor, *, op, group):
        calls.append("max")
        tensor.fill_(maximum)

    def gather(tensor, *, gather_list, dst, group):
        assert dst == 7
        calls.append("metadata" if tensor.dtype == torch.long else "rows")
        for target, peer, count in zip(gather_list, peers, counts, strict=True):
            target.zero_()
            if tensor.dtype == torch.long:
                assert tensor.shape == (maximum + 1, 2)
                target[0, 0] = count
                target[1 : count + 1, 0] = peer["batch_indices"]
                target[1 : count + 1, 1] = peer["row_indices"]
            else:
                target[:count] = peer["rows"]

    monkeypatch.setattr(dist, "all_reduce", reduce)
    monkeypatch.setattr(dist, "gather", gather)
    monkeypatch.setattr(dist, "get_global_rank", lambda actual, rank: 7)
    output = runtime._gather_cuda_sparse_sp_selected_to_source(
        {"sp_group": group, "sp_size": len(counts), "sp_rank": 0}, peers[0]
    )
    for result, key in zip(
        output, ["rows", "batch_indices", "row_indices"], strict=True
    ):
        assert torch.equal(result, torch.cat([peer[key] for peer in peers]))
    assert output[0].dtype == dtype
    assert calls == (["max", "metadata", "rows"] if maximum else ["max"])


@pytest.mark.parametrize("count,maximum", [(0, 3), (2, 3), (0, 0)])
def test_non_source_participates_without_materializing_gather_lists(
    monkeypatch, count, maximum
):
    calls = []

    def reduce(tensor, **kwargs):
        tensor.fill_(maximum)

    def gather(tensor, *, gather_list, dst, group):
        assert gather_list is None
        calls.append(tensor)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    monkeypatch.setattr(dist, "gather", gather)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: 4)
    selected = {
        "rows": torch.ones(count, 4),
        "batch_indices": torch.arange(count),
        "row_indices": torch.arange(count),
    }
    assert (
        runtime._gather_cuda_sparse_sp_selected_to_source(
            {"sp_group": object(), "sp_size": 2, "sp_rank": 1}, selected
        )
        is None
    )
    assert len(calls) == (2 if maximum else 0)
    if calls:
        assert calls[0][0, 0] == count
        assert torch.equal(calls[1][:count], selected["rows"])


def test_cuda_dispatch_uses_packed_path(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        runtime, "_gather_cuda_sparse_sp_selected_to_source", lambda *args: sentinel
    )
    selected = {"rows": SimpleNamespace(device=SimpleNamespace(type="cuda"))}
    assert runtime._gather_sparse_sp_selected_to_source({}, selected) is sentinel


def _compare_real_gloo_worker(rank, port):
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=30),
    )
    try:
        context = {"sp_group": dist.group.WORLD, "sp_size": 2, "sp_rank": rank}
        for dtype in (torch.float32, torch.bfloat16):
            for counts in ((0, 0), (0, 3), (5, 2), (3, 0)):
                count = counts[rank]
                selected = {
                    "rows": (torch.arange(count * 4).reshape(count, 4) + rank * 100).to(
                        dtype
                    ),
                    "batch_indices": torch.arange(count, dtype=torch.long) % 2,
                    "row_indices": torch.arange(count, dtype=torch.long) % 3,
                }
                # CPU dispatch retains the legacy four-gather implementation.
                expected = runtime._gather_sparse_sp_selected_to_source(
                    context, selected
                )
                actual = runtime._gather_cuda_sparse_sp_selected_to_source(
                    context, selected
                )
                if rank == 0:
                    for left, right in zip(expected, actual, strict=True):
                        assert torch.equal(left, right)
                else:
                    assert expected is actual is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo"
)
def test_real_two_rank_packed_gather_matches_legacy():
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_compare_real_gloo_worker, args=(port,), nprocs=2, join=True)
