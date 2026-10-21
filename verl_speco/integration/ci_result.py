# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in CI evidence; independent of accelerator and Ray dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CIResult:
    def __init__(self, training: bool):
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "training": training,
            "completed": False,
            "optimizer_steps": 0,
            "publications": 0,
            "generation_steps": 0,
            "errors": [],
        }

    def record_workers(self, results):
        for result in results:
            if not isinstance(result, dict):
                continue
            self.data["errors"].extend(result.get("errors", []))
            if result.get("reason") == "activation_failed":
                self.data["errors"].append(
                    {"type": "ActivationError", "message": "activation_failed"}
                )

    def record_metrics(self, metrics):
        if not isinstance(metrics, dict):
            return
        if "training/global_step" in metrics:
            self.data["generation_steps"] += 1
        if metrics.get("drafter/trained", 0) > 0:
            self.data["optimizer_steps"] += int(
                metrics.get("drafter/train_successful_steps_max", 0)
            )
        self.data["publications"] += int(metrics.get("drafter/published", 0))
        for key, bad in (
            ("drafter/train_activation_failed", 1),
            ("drafter/train_worker_results_consistent", 0),
        ):
            if key in metrics and metrics[key] == bad:
                self.data["errors"].append(
                    {"type": "TrainingContractError", "message": key}
                )

    def write(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(target)


def validate_result(data, *, training):
    if data.get("schema_version") != 1 or data.get("training") is not training:
        raise ValueError("CI result schema or mode mismatch")
    if data.get("completed") is not True or data.get("errors") != []:
        raise ValueError(f"CI run incomplete or failed: {data.get('errors')}")
    required = (
        ("optimizer_steps", "publications") if training else ("generation_steps",)
    )
    for key in required:
        value = data.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"CI run has no successful {key}")
