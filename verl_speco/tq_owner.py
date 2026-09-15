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
"""Long-lived TransferQueue owner for standalone Producer/Consumer jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import threading
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from verl_speco.integration.transferqueue_bridge import (
    close_transfer_queue_owner,
    configure_transfer_queue,
    connect_ray_cluster,
    put_sample,
    start_transfer_queue_owner,
)
from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION


logger = logging.getLogger(__name__)


def install_signal_handlers(stop_event: threading.Event) -> None:
    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("TQ owner received signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def publish_owner_ready(run_id: str, schema_version: int) -> str:
    if not run_id:
        raise ValueError("transfer_queue.run_id must be set for standalone owner")
    key = f"control:v{int(schema_version)}:{run_id}:owner-ready"
    put_sample(
        key,
        {"marker": torch.tensor([1], dtype=torch.uint8)},
        tag={
            "record_type": "control",
            "status": "owner_ready",
            "schema_version": int(schema_version),
            "run_id": run_id,
        },
    )
    return key


def wait_until_stopped(stop_event: threading.Event) -> None:
    stop_event.wait()


def run_owner(config: Any, *, stop_event: threading.Event | None = None) -> int:
    training_cfg = config.actor_rollout_ref.rollout.drafter.training
    tq_cfg = OmegaConf.to_container(training_cfg.transfer_queue, resolve=True)
    if not isinstance(tq_cfg, dict):
        raise TypeError("transfer_queue configuration must resolve to a mapping")
    # Invoking the dedicated owner entrypoint is itself the request to enable
    # TQ. Keep speco_base.yaml disabled by default for ordinary training jobs,
    # and enable only this process's copied configuration.
    tq_cfg["enable"] = True
    if not configure_transfer_queue(tq_cfg):
        raise RuntimeError("Standalone TQ owner requires TransferQueue==0.1.10")
    ray_cfg = tq_cfg.get("ray", {})
    ray_address = ray_cfg.get("address")
    if not ray_address:
        raise ValueError(
            "transfer_queue.ray.address must point to a running Ray cluster"
        )
    namespace = ray_cfg.get("namespace")
    event = stop_event or threading.Event()
    if stop_event is None:
        install_signal_handlers(event)

    started = False
    try:
        connect_ray_cluster(str(ray_address), str(namespace) if namespace else None)
        start_transfer_queue_owner(tq_cfg)
        started = True
        ready_key = publish_owner_ready(
            str(tq_cfg.get("run_id") or ""),
            int(tq_cfg.get("schema_version", PROTOCOL_SCHEMA_VERSION)),
        )
        logger.info("TQ owner ready key=%s", ready_key)
        ready_file = os.environ.get("SPECO_TQ_OWNER_READY_FILE")
        if ready_file:
            Path(ready_file).touch()
        wait_until_stopped(event)
        return 0
    finally:
        if started:
            close_transfer_queue_owner()


@hydra.main(config_path="config", config_name="speco_base", version_base=None)
def main(config: Any) -> None:
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(run_owner(config))


if __name__ == "__main__":
    main()
