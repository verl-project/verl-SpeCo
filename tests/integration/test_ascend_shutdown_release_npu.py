# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Large, opt-in device handoff test; no target checkpoint or RL required."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECO_RUN_LARGE_NPU_TESTS") != "1",
    reason="Requires an exclusive NPU and about 42 GiB of host snapshot memory per rank",
)

PRESSURE_CHILD = """
import os, sys, time
from pathlib import Path
import torch
import torch_npu
rank, size, ready, release = sys.argv[1:]
torch.npu.set_device(int(rank))
hold = torch.empty(int(size), dtype=torch.uint8, device=f'npu:{rank}')
torch.npu.synchronize()
Path(ready).write_text('ready')
deadline = time.monotonic() + 120
while not Path(release).exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Parent did not release pressure process')
    time.sleep(0.05)
time.sleep(2)
del hold
torch.npu.empty_cache()
torch.npu.synchronize()
"""


def test_shutdown_waits_for_device_release_then_restores_large_pool(tmp_path):
    import torch_npu  # noqa: F401 -- register the NPU device

    from verl_speco.trainer.ascend_state_parking import AscendStateParking, state_probe
    from verl_speco.trainer.deepseek_v4_resident_probe import wake_parked_for_shutdown

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(rank)
    device = f"npu:{rank}"
    state_gib, pressure_gib = 42, 24
    parking = AscendStateParking()
    with parking.allocation_scope():
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        for index in range(state_gib):
            model.register_buffer(
                f"storage_{index}",
                torch.full((1024**3,), 7, dtype=torch.uint8, device=device),
            )
        engine = SimpleNamespace(
            model_engine=SimpleNamespace(
                module=model,
                optimizer=torch.optim.AdamW(model.parameters()),
                lr_scheduler=None,
                global_steps=0,
            ),
            teacher=torch.nn.Identity(),
        )
    before = state_probe(engine)
    managed = parking.park(engine)
    assert managed >= state_gib * 1024**3
    ready, release = tmp_path / "ready", tmp_path / "release"
    process = None
    with (tmp_path / "pressure.log").open("w") as log:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    PRESSURE_CHILD,
                    str(rank),
                    str(pressure_gib * 1024**3),
                    str(ready),
                    str(release),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 120
            while not ready.exists():
                if process.poll() is not None:
                    pytest.fail((tmp_path / "pressure.log").read_text())
                if time.monotonic() >= deadline:
                    raise TimeoutError("NPU pressure process did not become ready")
                time.sleep(0.05)
            free_before = torch.npu.mem_get_info()[0]
            assert free_before < managed, "The test did not create capacity pressure"
            release.write_text("release")
            evidence = wake_parked_for_shutdown(
                parking,
                engine,
                free_bytes=lambda: torch.npu.mem_get_info()[0],
            )
            assert evidence["observations"] > 1
            assert parking.state == "active" and state_probe(engine) == before
            assert process.wait(timeout=60) == 0
            report = {
                "rank": rank,
                "status": "passed",
                "managed_bytes": managed,
                "free_before": free_before,
                "release": evidence,
                "state_values_and_addresses_preserved": True,
            }
            root = Path(os.environ.get("SPECO_NPU_RELEASE_REPORT_DIR", str(tmp_path)))
            (root / f"rank_{rank}.json").write_text(json.dumps(report, indent=2))
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            wake_parked_for_shutdown(
                parking,
                engine,
                free_bytes=lambda: torch.npu.mem_get_info()[0],
            )
