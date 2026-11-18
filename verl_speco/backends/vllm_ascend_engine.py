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
"""A bounded adapter around the in-package DeepSeek V4 DSpark training function."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class Progress:
    global_step: int = 0
    global_examples: int = 0
    epoch: int = 0
    batch_cursor: int = 0


class RemainingBatches:
    """A sized view at optimizer boundaries, preserving the epoch's tail window."""

    def __init__(self, loader, start: int, count: int):
        if start < 0 or count <= 0 or start + count > len(loader):
            raise ValueError("Invalid optimizer-boundary batch cursor")
        self.loader, self.start, self.count = loader, start, count

    def __len__(self):
        return self.count

    def __iter__(self):
        import itertools

        return itertools.islice(iter(self.loader), self.start, self.start + self.count)


class VllmAscendDSparkEngine:
    """Delegate all numerical work to the DSpark numerical loop; retain resumable SpeCo progress.

    The numerical callable is injected so the adapter's lifecycle can be tested
    on CPU without pretending to test the Ascend serving model.
    """

    def __init__(
        self,
        *,
        model_engine,
        teacher,
        cfg,
        topology,
        device,
        tracker,
        logger,
        train_function,
        checkpoint_dir,
    ):
        required = {
            "model_engine",
            "target_lm_head",
            "dataloader",
            "sampler",
            "initial_global_step",
            "initial_global_examples",
            "save_checkpoints",
        }
        if not required.issubset(inspect.signature(train_function).parameters):
            raise RuntimeError(
                "DSpark run_dspark_training API has changed; review the adapter"
            )
        self.model_engine = model_engine
        self.teacher = teacher
        self.cfg = cfg
        self.topology = topology
        self.device = device
        self.tracker = tracker
        self.logger = logger
        self.train_function = train_function
        self.checkpoint_dir = str(checkpoint_dir)
        self.progress = Progress()
        self.failed = False

    def train_cycle(self, loader, sampler, *, max_steps: int) -> dict:
        if self.failed:
            raise RuntimeError("Training session failed; restore a recovery checkpoint")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        p = self.progress
        accumulation = int(self.cfg.deepspeed.gradient_accumulation_steps)
        if not 0 <= p.batch_cursor < len(loader) or p.batch_cursor % accumulation:
            raise ValueError("Resume cursor is not an optimizer boundary in this epoch")
        steps = min(max_steps, math.ceil((len(loader) - p.batch_cursor) / accumulation))
        batches = min(steps * accumulation, len(loader) - p.batch_cursor)
        sampler.set_epoch(p.epoch)
        cycle_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cycle_cfg.train.epochs = 1
        cycle_cfg.train.max_optimizer_steps = steps
        try:
            result = self.train_function(
                model_engine=self.model_engine,
                target_lm_head=self.teacher,
                dataloader=RemainingBatches(loader, p.batch_cursor, batches),
                sampler=None,  # sampler epoch is owned by the persistent cursor
                cfg=cycle_cfg,
                logger=self.logger,
                tracker=self.tracker,
                device=self.device,
                parallel_topology=self.topology,
                checkpoint_dir=self.checkpoint_dir,
                initial_global_step=p.global_step,
                initial_global_examples=p.global_examples,
                save_checkpoints=False,
            )
            if int(result["global_step"]) != p.global_step + steps:
                raise RuntimeError(
                    "DSpark training did not complete the planned optimizer steps"
                )
            engine_step = getattr(
                self.model_engine, "global_steps", result["global_step"]
            )
            if int(engine_step) != int(result["global_step"]):
                raise RuntimeError(
                    "DeepSpeed engine step differs from DSpark training progress"
                )
            if result.get("observed_nonzero_optimizer_lr") and not result.get(
                "observed_update"
            ):
                raise RuntimeError(
                    "Nonzero-LR training did not observe a parameter update"
                )
            p.global_step = int(result["global_step"])
            p.global_examples += int(result["global_examples"])
            p.batch_cursor += batches
            if p.batch_cursor == len(loader):
                p.epoch += 1
                p.batch_cursor = 0
            return dict(result)
        except BaseException:
            # A collective or optimizer may already have partially advanced.
            self.failed = True
            raise
