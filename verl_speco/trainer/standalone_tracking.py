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
"""Lightweight standalone tracking shared by Driver and Consumer."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

_CONSOLE_TRACKING_BACKEND = "console"


class StandaloneTokenWindow:
    """Aggregate successful CPU-prepared batches without per-step collectives."""

    _FIELDS = ("source_tokens", "window_tokens", "supervised_tokens", "padding_tokens", "samples")
    _NAMES = ("source", "window", "supervised", "padding", "samples")
    _WORK_FIELDS = ("window_train_input_positions", "window_statistics_input_positions")

    def __init__(self, interval: int):
        if type(interval) is not int or interval < 0:
            raise ValueError("token_metrics_interval_steps must be a non-negative integer")
        self.interval = interval
        self.totals = [0] * 9
        self.steps = 0
        self.started = 0.0
        self.checkpoint_seconds = 0.0
        self.run_started = 0.0
        self.run_checkpoint_seconds = 0.0
        self.run_totals = [0] * 9
        self.run_steps = 0

    def record_checkpoint(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid checkpoint duration")
        if self.interval:
            self.checkpoint_seconds += seconds
            self.run_checkpoint_seconds += seconds

    def update(self, counts, *, successful_steps, optimizer_step, max_steps, start_time, device, world_size):
        if not self.interval:
            return {}
        if self.steps == 0 and self.started == 0.0:
            self.started = start_time
        if self.run_steps == 0:
            self.run_started = start_time
        self.steps += 1
        self.run_steps += 1
        if isinstance(counts, dict):
            for index, field in enumerate(self._FIELDS):
                value = counts[field]
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid standalone token count: {field}")
                self.totals[index] += value
                self.run_totals[index] += value
            self.totals[5] += 1
            self.run_totals[5] += 1
            present = [field in counts for field in self._WORK_FIELDS]
            if any(present):
                if not all(present):
                    raise ValueError("Incomplete standalone window work counts")
                for index, field in enumerate(self._WORK_FIELDS, 6):
                    value = counts[field]
                    if type(value) is not int or value < 0:
                        raise ValueError(f"Invalid standalone window work count: {field}")
                    self.totals[index] += value
                    self.run_totals[index] += value
                self.totals[8] += 1
                self.run_totals[8] += 1
        if successful_steps % self.interval and not (max_steps > 0 and optimizer_step >= max_steps):
            return {}
        import torch
        import torch.distributed as dist

        totals = torch.tensor(self.totals, dtype=torch.int64, device=device)
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError("Token metrics require the matching distributed group")
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        values = totals.tolist()
        elapsed = max(time.perf_counter() - self.started, 0.0)
        timings = torch.tensor(
            [elapsed, max(elapsed - self.checkpoint_seconds, 0.0), self.checkpoint_seconds],
            dtype=torch.float32, device=device,
        )
        if world_size > 1:
            # Compare durations, never timestamps from different node clocks.
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        elapsed, excluding_checkpoints, checkpoint_seconds = timings.tolist()
        result = {}
        if values[5] == self.steps * world_size:
            result = {f"tokens/{key}": float(value)
                      for key, value in zip(self._NAMES, values[:5], strict=True)}
            if elapsed > 0:
                result["tokens/source_per_second"] = values[0] / elapsed
            if excluding_checkpoints > 0:
                result["tokens/source_per_second_excluding_checkpoints"] = values[0] / excluding_checkpoints
            result["perf/checkpoint_blocking_time"] = checkpoint_seconds
            if values[8] == self.steps * world_size:
                result.update({f"tokens/{field}": float(values[index])
                               for index, field in enumerate(self._WORK_FIELDS, 6)})
        else:
            logger.warning("Skipping incomplete standalone token metrics window")
        self.totals = [0] * 9
        self.steps = 0
        self.checkpoint_seconds = 0.0
        self.started = time.perf_counter()
        return result

    def finish(self, *, device, world_size):
        """Summarize the full measured training interval, including final saves."""
        if not self.interval or not self.run_steps:
            return {}
        import torch
        import torch.distributed as dist

        totals = torch.tensor(self.run_totals + [self.run_steps], dtype=torch.int64, device=device)
        elapsed = max(time.perf_counter() - self.run_started, 0.0)
        times = torch.tensor([elapsed, max(elapsed - self.run_checkpoint_seconds, 0.0),
                              self.run_checkpoint_seconds], dtype=torch.float32, device=device)
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError("Token metrics require the matching distributed group")
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
        values = totals.tolist()
        elapsed, excluding_checkpoints, checkpoint_seconds = times.tolist()
        result = {"perf/run_training_time": elapsed, "perf/run_checkpoint_blocking_time": checkpoint_seconds}
        if values[5] == values[9]:
            result.update({f"tokens/run_{key}": float(value)
                           for key, value in zip(self._NAMES, values[:5], strict=True)})
            if elapsed > 0:
                result["tokens/run_source_per_second"] = values[0] / elapsed
            if excluding_checkpoints > 0:
                result["tokens/run_source_per_second_excluding_checkpoints"] = values[0] / excluding_checkpoints
            if values[8] == values[9]:
                result.update({f"tokens/run_{field}": float(values[index])
                               for index, field in enumerate(self._WORK_FIELDS, 6)})
        else:
            logger.warning("Skipping incomplete standalone run token metrics")
        return result


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping, an object, or a None-shaped config section."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _standalone_tracking_backends(config: Any) -> list[str]:
    """Normalized, de-duplicated, non-console ``trainer.logger`` backends."""
    backends = _cfg_get(_cfg_get(config, "trainer"), "logger")
    if backends is None:
        return []
    if isinstance(backends, str):
        backends = [backends]
    unique = dict.fromkeys(
        backend
        for backend in (str(backend).strip().lower() for backend in backends)
        if backend and backend != _CONSOLE_TRACKING_BACKEND
    )
    return list(unique)


def _build_standalone_tracking(config: Any, *, rank: int) -> list[Any]:
    """Build one tracker per requested backend on rank 0 (e.g. TensorBoard, W&B).

    The console logger is handled by the training loop itself; console entries
    are dropped and unsupported ones are logged and ignored. Each backend is
    initialized independently, so one failing backend (e.g. W&B without
    credentials) never disables the others.
    """
    if rank != 0:
        return []
    requested = _standalone_tracking_backends(config)
    if not requested:
        return []
    try:
        from verl.utils.tracking import Tracking
    except Exception:
        logger.exception("[standalone rank=%s] tracking backend is unavailable", rank)
        return []
    supported = [
        backend for backend in requested if backend in Tracking.supported_backend
    ]
    unsupported = [
        backend for backend in requested if backend not in Tracking.supported_backend
    ]
    if unsupported:
        logger.warning(
            "[standalone rank=%s] ignoring unsupported tracking backends=%s",
            rank,
            unsupported,
        )
    trainer_cfg = _cfg_get(config, "trainer")
    project_name = str(_cfg_get(trainer_cfg, "project_name") or "verl_dspark_drafter")
    experiment_name = str(
        _cfg_get(trainer_cfg, "experiment_name") or "standalone_draft"
    )
    trackers: list[Any] = []
    drafter_cfg = _cfg_get(
        _cfg_get(_cfg_get(config, "actor_rollout_ref"), "rollout"), "drafter"
    )
    training_cfg = _cfg_get(drafter_cfg, "training")
    run_config = {
        name: _cfg_get(training_cfg, name)
        for name in (
            "lr",
            "batch_size_per_gpu",
            "max_steps",
            "lr_scheduler_type",
            "dspark_block_size",
            "dspark_num_anchors",
            "dspark_loss_mode",
            "dspark_ce_loss_alpha",
            "dspark_l1_loss_alpha",
            "dflash_block_size",
            "dflash_num_anchors",
            "dflash_loss_mode",
            "dflash2_block_size",
            "dflash2_num_anchors",
            "dflash2_loss_mode",
            "dflash2_selector_loss_weight",
            "eagle3_block_size",
        )
        if _cfg_get(training_cfg, name) is not None
    }
    algorithm = _cfg_get(drafter_cfg, "speculative_algorithm")
    if algorithm is not None:
        run_config["algorithm"] = str(algorithm)
    logger.info("[standalone rank=%s] requested tracking backends=%s", rank, requested)
    for backend in supported:
        try:
            trackers.append(
                Tracking(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    default_backend=[backend],
                    config={
                        "standalone_training": run_config,
                        "trainer": {
                            "project_name": project_name,
                            "experiment_name": experiment_name,
                        },
                    },
                )
            )
            logger.info(
                "[standalone rank=%s] tracking initialized backend=%s project=%s experiment=%s",
                rank,
                backend,
                project_name,
                experiment_name,
            )
        except Exception:
            logger.exception(
                "[standalone rank=%s] failed to initialize tracking backend=%s",
                rank,
                backend,
            )
    return trackers


def _log_standalone_tracking_metrics(
    trackers: list[Any], metrics: dict[str, float], *, step: int
) -> None:
    for tracking in trackers:
        try:
            tracking.log(data=dict(metrics), step=int(step))
        except Exception:
            logger.exception(
                "[standalone] failed to write tracking metrics at step=%s", step
            )


def _log_producer_tracking_metrics(
    trackers: list[Any],
    metrics: dict[str, float],
    *,
    published_total: int,
) -> None:
    """Producer uses its own axis on per-metric-step backends.

    W&B has a global step shared with optimizer logs, so avoid sending it an
    independent sample-count step. Training logging there remains unchanged.
    """
    for tracking in trackers:
        try:
            tracking.log(
                data=dict(metrics),
                step=int(published_total),
                backend=["swanlab", "tensorboard"],
            )
        except Exception:
            logger.exception(
                "Failed to log Producer window at published_total=%s", published_total
            )


def _select_standalone_tracking_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Upload core standalone metrics using each backend's available diagnostics."""
    prefix = next(
        (
            name
            for name in ("eagle3", "dflash", "dspark", "dflash2")
            if any(key.startswith(f"{name}/") for key in metrics)
        ),
        None,
    )
    if prefix is None:
        return dict(metrics)
    keys: tuple[str, ...] = (
        f"{prefix}/loss",
        f"{prefix}/top1_acc",
        "train/lr",
        "perf/step_time",
        "perf/loader_plan_wait_time",
        "perf/batch_ready_wait_time",
        "perf/loader_tq_get_time",
        "perf/loader_batch_prepare_time",
        "perf/batch_prepare_time",
        "perf/batch_finalize_time",
        "perf/commit_gate_wait_time",
        "perf/train_time",
        "tokens/source",
        "tokens/window",
        "tokens/supervised",
        "tokens/padding",
        "tokens/samples",
        "tokens/window_train_input_positions",
        "tokens/window_statistics_input_positions",
        "tokens/source_per_second",
        "tokens/source_per_second_excluding_checkpoints",
        "perf/checkpoint_blocking_time",
    )
    if prefix == "dspark":
        keys += ("dspark/ce_loss", "dspark/l1_loss")
    if prefix == "eagle3":
        keys += ("train/simulated_acc_len",)
    else:
        keys += (f"{prefix}/mean_acceptance_length",)
    if prefix == "dflash2":
        keys += ("dflash2/selector_loss", "dflash2/selector_accuracy")
    selected = {
        key: float(metrics[key])
        for key in keys
        if key in metrics and math.isfinite(float(metrics[key]))
    }
    wait_metric = "perf/batch_ready_wait_time"
    if all(
        key in selected for key in ("perf/step_time", wait_metric, "perf/train_time")
    ):
        # All three are rank-zero wall times for the same Consumer cycle.
        selected["perf/other_time"] = max(
            0.0,
            selected["perf/step_time"]
            - selected[wait_metric]
            - selected["perf/train_time"]
            - selected.get("perf/batch_prepare_time", 0.0)
            - selected.get("perf/batch_finalize_time", 0.0)
            - selected.get("perf/commit_gate_wait_time", 0.0),
        )
    return selected


def _finish_standalone_tracking(trackers: list[Any], *, rank: int) -> None:
    for tracking in trackers:
        try:
            finish = getattr(tracking, "finish", None)
            if callable(finish):
                finish()
                continue
            # verl release/v0.8.0 only finalizes backends in __del__().
            # Close them explicitly and remove successful entries so object
            # destruction cannot finalize the same backend a second time.
            backends = getattr(tracking, "logger", {})
            for name, backend in list(backends.items()):
                backend_finish = getattr(backend, "finish", None)
                if not callable(backend_finish):
                    continue
                try:
                    if name in {"wandb", "vemlp_wandb"}:
                        backend_finish(exit_code=0)
                    else:
                        backend_finish()
                except Exception:
                    logger.exception(
                        "[standalone rank=%s] failed to finalize tracking backend=%s",
                        rank,
                        name,
                    )
                else:
                    backends.pop(name, None)
        except Exception:
            logger.exception(
                "[standalone rank=%s] failed to finalize a tracking backend", rank
            )
