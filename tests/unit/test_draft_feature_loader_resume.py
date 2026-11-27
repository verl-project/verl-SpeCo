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
"""Deterministic epoch/batch cursor for file-store resume."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.draft_dataset import (  # noqa: E402
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import DraftFeatureSample  # noqa: E402


class _Store:
    def __init__(self, num_samples: int = 10):
        self.keys = [f"k{i}" for i in range(num_samples)]

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0):
        keys = list(self.keys)
        if shuffle:
            random.Random(int(seed)).shuffle(keys)
        return iter(keys)

    def read(self, key: str) -> DraftFeatureSample:
        index = int(key[1:])
        return DraftFeatureSample(
            input_ids=torch.arange(index, index + 4, dtype=torch.long),
            loss_mask=torch.ones(4, dtype=torch.float32),
            hidden_states=torch.zeros(4, 8, dtype=torch.float32),
        )


def _ids(batch) -> list[int]:
    return [int(sample.input_ids[0].item()) for sample in batch]


def _config(**kwargs):
    defaults = {"batch_size": 2, "shuffle": False, "repeat": False}
    defaults.update(kwargs)
    return DraftFeatureDataLoaderConfig(**defaults)


def test_loader_skips_consumed_batches_after_resume() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1]
    assert _ids(next(iterator)) == [2, 3]
    assert loader.progress() == {"epoch": 0, "batch_index": 2}

    resumed = DraftFeatureDataLoader(
        _Store(), _config(start_epoch=0, skip_batches=2)
    )
    batches = [_ids(batch) for batch in resumed]
    assert batches == [[4, 5], [6, 7], [8, 9]]
    assert resumed.progress() == {"epoch": 0, "batch_index": 5}


def test_loader_resume_cursor_moves_to_next_epoch_when_fully_consumed() -> None:
    loader = DraftFeatureDataLoader(
        _Store(10), _config(batch_size=4, repeat=True)
    )
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1, 2, 3]
    assert _ids(next(iterator)) == [4, 5, 6, 7]
    assert _ids(next(iterator)) == [8, 9]
    assert loader.progress() == {"epoch": 0, "batch_index": 3}

    resumed = DraftFeatureDataLoader(
        _Store(10),
        _config(batch_size=4, repeat=True, start_epoch=0, skip_batches=3),
    )
    iterator = iter(resumed)
    assert _ids(next(iterator)) == [0, 1, 2, 3]
    assert resumed.progress() == {"epoch": 1, "batch_index": 1}


class _FlakyStore(_Store):
    def __init__(self, num_samples: int = 10, fail_key: str | None = None):
        super().__init__(num_samples)
        self.fail_key = fail_key

    def read(self, key: str):
        if key == self.fail_key:
            raise OSError("transient read failure")
        return super().read(key)


def test_loader_resume_rejects_read_errors_in_saved_epoch() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1]
    assert _ids(next(iterator)) == [2, 3]
    assert loader.progress() == {"epoch": 0, "batch_index": 2}

    # The saved prefix cannot be rebuilt because k2 is unreadable now; skipping
    # two batches would land on a different boundary.
    resumed = DraftFeatureDataLoader(
        _FlakyStore(fail_key="k2"), _config(start_epoch=0, skip_batches=2)
    )
    with pytest.raises(ValueError, match="unreadable"):
        next(iter(resumed))


def test_loader_resume_rejects_cursor_beyond_epoch() -> None:
    resumed = DraftFeatureDataLoader(
        _Store(10), _config(start_epoch=0, skip_batches=99)
    )
    with pytest.raises(ValueError, match="exceeds"):
        next(iter(resumed))


def test_loader_reports_epoch_read_errors() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    assert loader.epoch_read_errors() == 0
    list(loader)
    assert loader.epoch_read_errors() == 0


def test_loader_resume_synchronizes_read_errors_with_other_ranks(
    monkeypatch,
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(
        "verl_speco.trainer.draft_dataset._collective_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    max_calls = 0

    def fake_all_reduce(tensor, op=None):
        nonlocal max_calls
        # Length sync uses MIN and must keep the local count. MAX is used first
        # for the read/filter failure flag, then for the read-error count; make
        # only the latter look like another rank had unreadable samples.
        if op == dist.ReduceOp.MAX:
            max_calls += 1
            if max_calls == 2:
                tensor.fill_(3)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    loader = DraftFeatureDataLoader(
        _Store(10),
        _config(world_size=2, rank=0, start_epoch=0, skip_batches=2),
    )
    with pytest.raises(ValueError, match="unreadable on at least one rank"):
        next(iter(loader))


def test_loader_synchronizes_rank_failures_before_raising(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(
        "verl_speco.trainer.draft_dataset._collective_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op=None):
        # The first MAX is the read/filter failure flag; simulate another rank
        # having failed so this healthy rank must stop before batching.
        if op == dist.ReduceOp.MAX:
            tensor.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    loader = DraftFeatureDataLoader(
        _Store(10),
        _config(world_size=2, rank=0, start_epoch=0, skip_batches=2),
    )
    with pytest.raises(RuntimeError, match="failed on another rank"):
        next(iter(loader))


_GLOO_RESUME_DRIVER = r'''
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from verl_speco.trainer.draft_dataset import (
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import DraftFeatureSample


class Store:
    def __init__(self, fail_key=None):
        self.keys = [f"k{i}" for i in range(8)]
        self.fail_key = fail_key

    def iter_keys(self, *, shuffle=False, seed=0):
        return iter(list(self.keys))

    def read(self, key):
        if key == self.fail_key:
            raise OSError("injected read failure")
        index = int(key[1:])
        return DraftFeatureSample(
            input_ids=torch.arange(index, index + 4, dtype=torch.long),
            loss_mask=torch.ones(4, dtype=torch.float32),
            hidden_states=torch.zeros(4, 8, dtype=torch.float32),
        )


rank = int(sys.argv[1])
world_size = int(sys.argv[2])
init_file = sys.argv[3]
result_path = Path(sys.argv[4])
mode = sys.argv[5]
result = "init_failed"
try:
    dist.init_process_group(
        backend="gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    # k4 falls in rank 1's contiguous shard, so only rank 1 hits the failure.
    loader = DraftFeatureDataLoader(
        Store(fail_key="k4" if rank == 1 else None),
        DraftFeatureDataLoaderConfig(
            batch_size=2,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            repeat=False,
            start_epoch=0,
            skip_batches=2,
            on_error=mode,
        ),
    )
    try:
        next(iter(loader))
    except Exception as exc:
        result = f"error:{type(exc).__name__}:{exc}"
    else:
        result = "no_error"
except Exception as exc:
    result = f"error:{type(exc).__name__}:{exc}"
finally:
    result_path.write_text(result)
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
'''


def _run_gloo_resume_case(tmp_path: Path, mode: str) -> dict[str, str]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("gloo backend is required")

    script = tmp_path / "gloo_resume_driver.py"
    script.write_text(textwrap.dedent(_GLOO_RESUME_DRIVER).lstrip(), encoding="utf-8")
    init_file = tmp_path / "gloo_init"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(rank),
                "2",
                str(init_file),
                str(result_dir / f"rank{rank}.txt"),
                mode,
            ],
            env=env,
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 60
    timed_out = []
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out.append(process)
    for process in timed_out:
        process.kill()
        process.wait(timeout=5)
    assert not timed_out, f"gloo resume workers hung in mode={mode!r}"

    results = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(result_dir.glob("rank*.txt"))
    }
    assert set(results) == {"rank0.txt", "rank1.txt"}, results
    assert all(value.startswith("error:") for value in results.values()), results
    return results


def test_gloo_two_rank_resume_read_error_syncs(tmp_path: Path) -> None:
    results = _run_gloo_resume_case(tmp_path, "skip")
    assert "ValueError" in results["rank0.txt"]
    assert "ValueError" in results["rank1.txt"]
    assert "unreadable on at least one rank" in results["rank0.txt"]
    assert "unreadable on at least one rank" in results["rank1.txt"]


def test_gloo_two_rank_resume_failure_syncs(tmp_path: Path) -> None:
    results = _run_gloo_resume_case(tmp_path, "raise")
    assert "RuntimeError" in results["rank0.txt"]
    assert "RuntimeError" in results["rank1.txt"]
    assert "Failed to read feature sample" in results["rank1.txt"]
    assert (
        "failed on another rank" in results["rank0.txt"]
        or "failed on another rank" in results["rank1.txt"]
    )
