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
"""Persistent Ascend execution behind the existing SpeCo worker lifecycle.

Buffer transactions, scheduling, target-version preflight and publish snapshots
remain owned by SpecoWorker/DrafterBaseTrainer. Only numerical execution differs.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from verl_speco.backends.vllm_ascend_engine import Progress
from verl_speco.integration.deepseek_v4_online import (
    TARGET_OUTPUT_NAMES,
    apply_shared_token_weights,
    apply_target_output,
    execution_config,
    runtime_weight_name,
)
from verl_speco.integration.vllm_ascend_config import (
    execution_environment,
    load_vllm_ascend_config,
    training_fingerprint,
)
from verl_speco.integration.vllm_ascend_implementation import (
    implementation_identity_matches,
)
from verl_speco.trainer.checkpoint_io import (
    bounded_checkpoint_writes,
    checkpoint_io_slot,
)
from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
    to_engine_tensors,
)
from verl_speco.trainer.vllm_ascend_checkpoint import rank_rng_state, restore_rng_state
from verl_speco.trainer.vllm_ascend_mask import collate_features

logger = logging.getLogger(__name__)


class OnlineMetrics:
    def __init__(self):
        self.values = {}

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            number = float(value)
            if not torch.isfinite(torch.tensor(number)):
                raise ValueError(f"Non-finite online training metric: {key}")
            self.values[f"drafter/vllm_ascend/{key}"] = number


class VllmAscendOnlineMixin:
    """Override model execution while inheriting SpeCo buffer/publish contracts."""

    # Provided by DrafterBaseTrainer when the mixin is composed.
    config: Any

    def __init__(self, config, *, _allow_probe_engine_dp=False, **kwargs):
        training = config.rollout.drafter.training
        if training.get("publish_dtype") or training.get("publish_param_name_patterns"):
            raise ValueError(
                "Online DS4 publishes complete runtime shards with their native dtypes; publish_dtype and publish_param_name_patterns must be unset"
            )
        from verl_speco.integration.deepseek_v4_actor_capture import feature_source

        self._feature_source = feature_source(training)
        if training.get("use_data_buffer", False):
            raise ValueError(
                "DeepSeek V4 online training currently requires same-step data (use_data_buffer=false)"
            )
        if training.get("is_offload_param", False) or training.get(
            "is_offload_optimizer", False
        ):
            raise ValueError(
                "Online vllm_ascend retains its DeepSpeed engine; native FSDP offload flags are unsupported"
            )
        if int(config.rollout.get("pipeline_model_parallel_size", 1)) != 1 or (
            int(config.rollout.get("data_parallel_size", 1)) != 1
            and not _allow_probe_engine_dp
        ):
            raise ValueError(
                "Online vllm_ascend requires rollout PP=1 and engine DP=1; rollout replicas supply training DP"
            )
        if str(config.rollout.drafter.speculative_algorithm).upper() != "DSPARK":
            raise ValueError(
                "Online vllm_ascend currently supports DeepSeek V4 0731 DSpark"
            )
        self.execution = execution_config(config)
        self.cfg, self.source = load_vllm_ascend_config(self.execution)
        # Runtime-shard publication requires identical unquantized parameter
        # layouts on the training and serving sides.
        self.cfg.model.checkpoint_format = "bf16"
        self.cfg.model.vllm_quantization = None
        self.cfg.model.dequantize_mtp_for_training = False
        self.contract = DeepSeekV4Contract.from_model(config.model.path)
        if self.cfg.train.get("dspark_enable_confidence_head", False):
            raise ValueError("Online DS4 confidence-head export has not been validated")
        if self.cfg.train.get("segment_length") or self.cfg.train.get("max_len"):
            raise ValueError(
                "Online rollout windows are already bounded; disable backend segment_length and max_len"
            )
        expected_batch = int(self.cfg.deepspeed.train_micro_batch_size_per_gpu) * int(
            self.cfg.deepspeed.gradient_accumulation_steps
        )
        if int(training.batch_size_per_gpu) != expected_batch:
            raise ValueError(
                f"training.batch_size_per_gpu must equal backend microbatch * accumulation ({expected_batch})"
            )
        super().__init__(
            config=config,
            backend=SimpleNamespace(model_type="dspark", supports_ulysses_sp=False),
            **kwargs,
        )
        if _allow_probe_engine_dp and (
            self.training_group_world_size
            != int(config.rollout.tensor_model_parallel_size)
            or self.dp_group_world_size != int(config.rollout.data_parallel_size)
        ):
            raise ValueError(
                "Resident probe requires explicit matching TP/DP training groups"
            )
        from verl_speco.integration.ascend_state_config import state_parking_mode

        self._state_parking_mode = state_parking_mode(config)
        if (
            self._state_parking_mode == "camem"
            and not _allow_probe_engine_dp
            and not config.rollout.get("enable_sleep_mode", False)
        ):
            raise ValueError(
                "Same-card CaMem lifecycle requires rollout.enable_sleep_mode=true; "
                "actor/rollout must release device memory before drafter activation"
            )
        self._state_lifecycle = None
        self.ascend_engine = None
        self.online_metrics = OnlineMetrics()
        self._pending_output = None
        self._online_cursor = 0
        self._online_cursor_step = None
        self._online_failed = False
        self._numerical_rng = None
        self.work_dir = Path(training.vllm_ascend.work_dir)
        self.cfg.model.vllm_tensor_parallel_size = self.training_group_world_size
        self.cfg.model.vllm_data_parallel_size = self.dp_group_world_size
        self.identity = {
            "contract": self.contract.identity(),
            "source": self.source,
            "training": training_fingerprint(self.cfg),
            "optimizer_horizon": int(
                self.execution.actor_rollout_ref.rollout.drafter.training.max_steps
            ),
        }
        # All participating workers enter init_model together. Create the CPU
        # control group here, before per-plan preflight can reject a local rank.
        if self._state_parking_mode == "camem":
            self._get_state_lifecycle()

    def _resolve_drafter_fsdp_device_mesh(self):
        return None  # DeepSpeed owns TP/DP execution, not an FSDP wrapper.

    def collect_online_data(self, batch, hidden_states, target_logprobs=None):
        if target_logprobs is not None:
            raise ValueError("DS4 online capture expects raw pre-HC features")
        step = batch.get("global_step")
        if step is None or int(step) != self.current_rl_step:
            raise ValueError(
                "Online DS4 features must match the current RL target version"
            )
        ids = batch["input_ids"].reshape(-1)
        head = batch.get("target_head_hidden_states")
        if head is None:
            from verl_speco.integration.deepseek_v4_actor_capture import unpack_features
            from verl_speco.trainer.deepseek_v4_contract import LAYOUT

            if batch.get("hidden_states_layout") != LAYOUT:
                raise ValueError("Missing explicit DeepSeek V4 actor pre-HC layout")
            hidden_states, head = unpack_features(hidden_states, self.contract)
        rows = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        positions = batch["hidden_positions"].reshape(-1).cpu()
        if positions.dtype != torch.int64 or positions.numel() != rows:
            raise ValueError("Feature positions must be int64 and match captured rows")
        if bool(((positions < 0) | (positions >= ids.numel())).any()):
            raise ValueError(
                "Feature positions are outside the unpadded token sequence"
            )
        if rows > int(self.config.rollout.drafter.training.get("max_seq_len", 8192)):
            raise ValueError("Rollout feature rows exceed training.max_seq_len")
        mask = (positions >= batch["prompts"].numel()).float()
        sample = from_engine_tensors(
            {
                "input_ids": ids.cpu().index_select(0, positions),
                "positions": positions,
                "hidden_states": hidden_states.reshape(rows, -1),
                "target_head_hidden_states": head.reshape(rows, -1),
            },
            mask,
            self.contract,
            sample_id=str(len(self.collected_data)),
        )
        if not bool(sample.loss_mask.any()):
            return
        self.collected_data.append({"step": int(step), "feature": sample})
        self._mark_buffer_changed()

    def get_training_data_status(self, **kwargs):
        status = super().get_training_data_status(**kwargs)
        # Cross-step windows are unsafe with a freshly synchronized HC teacher.
        count = sum(
            int(item["step"]) == self.current_rl_step for item in self.collected_data
        )
        remaining = self.optimizer_steps_total < int(
            self.execution.actor_rollout_ref.rollout.drafter.training.max_steps
        )
        status.update(
            trainable_samples=count,
            trainable_batches=count // self.batch_size if remaining else 0,
            same_step_data_required=True,
            data_version=self.current_rl_step if count else None,
            oldest_sample_step=self.current_rl_step if count else None,
            newest_sample_step=self.current_rl_step if count else None,
            min_sample_step=self.current_rl_step,
            max_sample_step=self.current_rl_step,
        )
        return status

    def sync_target_output(self, payload, global_step):
        state = payload.get("ds4_target_output")
        if not isinstance(state, dict) or global_step is None:
            raise ValueError(
                "Online DS4 requires the complete versioned target output path"
            )
        if set(state) != set(TARGET_OUTPUT_NAMES) | {"embed_tokens.weight"}:
            raise ValueError(
                "Online DS4 actor sync requires HC, norm, LM head and shared embedding"
            )
        self._pending_output = state
        self._target_lm_head_weight_step = int(global_step)
        self._applied_target_output_step = None
        return {"accepted": True, "applied": False, "deferred": True}

    def get_target_lm_head_row_indices(self):
        return None  # HC distillation needs the complete vocabulary.

    def _get_state_lifecycle(self):
        lifecycle = getattr(self, "_state_lifecycle", None)
        if (
            lifecycle is None
            and getattr(self, "_state_parking_mode", "none") == "camem"
        ):
            from verl_speco.trainer.vllm_ascend_lifecycle import AscendTrainingLifecycle

            if (
                self.training_group_world_size * self.dp_group_world_size
                != dist.get_world_size()
            ):
                raise ValueError(
                    "CaMem lifecycle requires all drafter ranks in the training mesh"
                )
            settings = self.config.rollout.drafter.training.vllm_ascend
            storage = str(settings.get("state_parking_storage", "memory"))
            factory = None
            if storage == "file":
                from verl_speco.trainer.ascend_file_state import (
                    FileBackedAscendStateParking,
                )

                factory = lambda: FileBackedAscendStateParking(
                    self.work_dir / "state_parking",
                    chunk_bytes=int(settings.get("state_parking_chunk_mib", 128))
                    * 1024**2,
                )
            elif storage != "memory":
                raise ValueError("state_parking_storage must be memory or file")
            lifecycle = (
                AscendTrainingLifecycle(parking_factory=factory)
                if factory
                else AscendTrainingLifecycle()
            )
            self._state_lifecycle = lifecycle
        return lifecycle

    def _state_read_scope(self):
        if self._online_failed:
            raise RuntimeError(
                "Cannot read failed online Ascend state; resume from a checkpoint"
            )
        lifecycle = getattr(self, "_state_lifecycle", None)
        return (
            lifecycle.active_operation(self.ascend_engine)
            if lifecycle
            else nullcontext()
        )

    async def activate_training_model(self):
        if self._online_failed:
            raise RuntimeError("Online Ascend session failed; resume from a checkpoint")
        lifecycle = self._get_state_lifecycle()
        if self.ascend_engine is not None:
            if lifecycle:
                lifecycle.wake(self.ascend_engine)
            return True
        with lifecycle.allocation_scope() if lifecycle else nullcontext():
            return await self._initialize_training_model()

    async def _initialize_training_model(self):
        from verl_speco.backends.vllm_ascend_runtime import (
            build_engine,
            runtime_versions,
        )

        versions = runtime_versions(require=True)
        self.identity["runtime_versions"] = versions
        for name, value in self.execution.draft_engine.environment.items():
            os.environ.setdefault(name, str(value))
        for name in (
            "MTP_DS4_TP_SHARD_EH_PROJECTIONS",
            "MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS",
        ):
            if os.environ.get(name, "0").strip().lower() not in {"0", "false", "off"}:
                raise ValueError(f"Online runtime-shard publishing requires {name}=0")
            os.environ[name] = "0"
        self.identity["execution_environment"] = execution_environment(self.execution)
        from verl_speco.backends import vllm_ascend_parallel as parallel

        # Reuse SpeCo's existing groups. Rank order is replica-major, TP-minor.
        topology = parallel.DeepSpeedParallelTopology.__new__(
            parallel.DeepSpeedParallelTopology
        )
        topology.coordinates = parallel.parallel_coordinates(
            dist.get_rank(), dist.get_world_size(), self.training_group_world_size
        )
        topology.tensor_parallel_group = self.training_process_group
        topology.data_parallel_group = self.data_parallel_process_group
        self.cfg.model.vllm_tensor_parallel_size = self.training_group_world_size
        self.cfg.model.vllm_data_parallel_size = self.dp_group_world_size
        self.cfg.train.scheduler_total_steps = int(
            self.execution.actor_rollout_ref.rollout.drafter.training.max_steps
        )
        dataset = SimpleNamespace(
            segment_length=None,
            sequence_lengths=[
                int(self.config.rollout.drafter.training.get("max_seq_len", 8192))
            ],
        )
        self.ascend_engine = build_engine(
            cfg=self.cfg,
            topology=topology,
            loader=[None],
            dataset=dataset,
            device=self.runtime_device,
            tracker=self.online_metrics,
            logger=logger,
            checkpoint_dir=self.work_dir,
        )
        self.model = self.ascend_engine.model_engine
        self._applied_target_output_step = None
        import uuid

        run_id = [uuid.uuid4().hex if dist.get_rank() == 0 else None]
        lifecycle = getattr(self, "_state_lifecycle", None)
        if lifecycle:
            dist.broadcast_object_list(run_id, src=0, group=lifecycle.control_group)
        else:
            dist.broadcast_object_list(run_id, src=0)
        self.work_dir = self.work_dir / str(run_id[0])
        resume = self.execution.draft_engine.get("resume_from")
        if resume:
            path = Path(resume)
            loaded, state = self.model.load_checkpoint(str(path.parent), tag=path.name)
            if not loaded or not implementation_identity_matches(
                state.get("speco_online", {}).get("identity"), self.identity
            ):
                raise ValueError("Online Ascend recovery identity differs")
            saved = state["speco_online"]
            from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

            refresh_fp32_weight_caches(self.model.module)
            self.ascend_engine.progress = Progress(**saved["progress"])
            self.optimizer_steps_total = self.ascend_engine.progress.global_step
            if int(self.model.global_steps) != self.optimizer_steps_total:
                raise ValueError("Online Ascend optimizer progress was not restored")
            restore_rng_state(saved["rng_by_rank"][dist.get_rank()])
        self._numerical_rng = rank_rng_state()
        return True

    async def training_step(self, step, **kwargs):
        if self._online_failed:
            raise RuntimeError(
                "Online Ascend session is invalid after a failed collective"
            )
        if step != self._target_lm_head_weight_step or step != self.current_rl_step:
            raise ValueError("DS4 training target/feature versions differ")
        if self.optimizer_steps_total >= int(
            self.execution.actor_rollout_ref.rollout.drafter.training.max_steps
        ):
            return False
        await self.activate_training_model()
        lifecycle = getattr(self, "_state_lifecycle", None)
        with lifecycle.allocation_scope() if lifecycle else nullcontext():
            return self._training_step_active(step)

    def _training_step_active(self, step):
        engine = self.ascend_engine
        try:
            if getattr(self, "_applied_target_output_step", None) != step:
                apply_shared_token_weights(
                    engine.model_engine.module,
                    self._pending_output,
                    tp_rank=engine.topology.tensor_parallel_rank,
                    tp_size=engine.topology.tensor_parallel_size,
                )
                apply_target_output(
                    engine.teacher,
                    {name: self._pending_output[name] for name in TARGET_OUTPUT_NAMES},
                    tp_rank=engine.topology.tensor_parallel_rank,
                    tp_size=engine.topology.tensor_parallel_size,
                )
                self._applied_target_output_step = step
        except BaseException:
            self._online_failed = True
            raise
        items = [
            item["feature"] for item in self.collected_data if item["step"] == step
        ]
        if self._online_cursor_step != step:
            self._online_cursor, self._online_cursor_step = 0, step
        selected = items[self._online_cursor : self._online_cursor + self.batch_size]
        if len(selected) < self.batch_size:
            return False
        micro = int(self.cfg.deepspeed.train_micro_batch_size_per_gpu)
        loader = [
            collate_features(
                [
                    to_engine_tensors(sample, self.contract)
                    for sample in selected[start : start + micro]
                ]
            )
            for start in range(0, len(selected), micro)
        ]
        if getattr(self, "_numerical_rng", None) is not None:
            restore_rng_state(self._numerical_rng)
        try:
            engine.train_cycle(
                loader, SimpleNamespace(set_epoch=lambda epoch: None), max_steps=1
            )
            self.optimizer_steps_total = engine.progress.global_step
            self._online_cursor += self.batch_size
            self.training_steps += 1
            return True
        except BaseException:
            self._online_failed = True
            raise
        finally:
            self._numerical_rng = rank_rng_state()

    def get_training_metrics(self):
        return {
            **self.online_metrics.values,
            "drafter/optimizer_steps_total": float(self.optimizer_steps_total),
        }

    def reset_training_metrics(self):
        self.online_metrics.values.clear()

    async def cleanup_training(self, clear_data=True):
        # Moving vLLM parameters through FSDP offload would invalidate runtime
        # views. Retain the engine and optimizer, including across RL triggers.
        if clear_data:
            self.collected_data.clear()
            self.data_buffer.clear()
            self._mark_buffer_changed()
            self._online_cursor = 0
            self._online_cursor_step = None
        lifecycle = getattr(self, "_state_lifecycle", None)
        if self.model is not None and (not lifecycle or lifecycle.state == "active"):
            self.model.zero_grad()
        self._pending_output = None
        self._target_lm_head_weight_step = None
        if lifecycle and self.ascend_engine is not None and not self._online_failed:
            lifecycle.park(self.ascend_engine)
            logger.info("[speco ascend lifecycle] cleanup: %s", lifecycle.last_status)

    async def release_training_memory_after_activation(self):
        lifecycle = getattr(self, "_state_lifecycle", None)
        if lifecycle and self.ascend_engine is not None:
            lifecycle.park(self.ascend_engine)
            logger.info(
                "[speco ascend lifecycle] activation released: %s",
                lifecycle.last_status,
            )

    def _collective_errors(self, error):
        errors = [None] * dist.get_world_size()
        lifecycle = getattr(self, "_state_lifecycle", None)
        if lifecycle:
            errors = lifecycle.gather(error)
        else:
            dist.all_gather_object(errors, error)
        if any(errors):
            self._online_failed = True
            raise RuntimeError(f"Online Ascend collective phase failed: {errors}")

    def write_runtime_publish_shards(self, path):
        """Read mapped state; the existing worker snapshots before cleanup."""
        with self._state_read_scope():
            return self._write_runtime_publish_shards_active(path)

    def _write_runtime_publish_shards_active(self, path):
        """Collectively write DP0 runtime shards; these are not recovery checkpoints."""
        if self.ascend_engine is None:
            raise RuntimeError("Cannot publish an uninitialized Ascend trainer")
        topology = self.ascend_engine.topology
        path = Path(path)
        error = None
        try:
            if topology.data_parallel_rank == 0:
                path.mkdir(parents=True, exist_ok=True)
                with checkpoint_io_slot(path):
                    state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.module.model.named_parameters()
                        if "embed_tokens" not in name
                        and not name.startswith("lm_head.")
                    }
                    torch.save(
                        state,
                        path / f"model_states_rank_{topology.tensor_parallel_rank}.pt",
                    )
                    del state
        except Exception as exc:  # noqa: BLE001 -- propagate to every participant
            error = repr(exc)
        self._collective_errors(error)
        return path

    def get_model_state_dict(self):
        """Publish native runtime TP shards through SpeCo's existing IPC payload."""
        if self.ascend_engine is None:
            return None
        topology = self.ascend_engine.topology
        path = self.write_runtime_publish_shards(
            self.work_dir
            / f"publish_rl_{self.current_rl_step:08d}_optim_{self.optimizer_steps_total:08d}"
        )
        weights = None
        error = None
        if topology.global_rank == 0:
            try:
                weights = {}
                for rank in range(topology.tensor_parallel_size):
                    state = torch.load(
                        path / f"model_states_rank_{rank}.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    for name, tensor in state.items():
                        weights[
                            runtime_weight_name(
                                rank, topology.tensor_parallel_size, name
                            )
                        ] = tensor
            except Exception as exc:  # noqa: BLE001 -- propagate to every participant
                error = repr(exc)
        self._collective_errors(error)
        if topology.global_rank == 0:
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        # Other ranks need a truthy marker for the existing worker's snapshot
        # acknowledgement; only the global publish leader returns actual weights.
        return (
            weights
            if topology.global_rank == 0
            else {"_speco_non_owner": torch.empty(0)}
        )

    def save_checkpoint(
        self, step: int, wait: bool = True, *, defer_completion: bool = False
    ):
        with self._state_read_scope():
            return self._save_checkpoint_active(
                step, wait, defer_completion=defer_completion
            )

    def _save_checkpoint_active(
        self, step: int, wait: bool = True, *, defer_completion: bool = False
    ):
        # DeepSpeed collective save completes synchronously, including when the
        # common trainer interface allows deferred completion.
        global_step = step
        if self.ascend_engine is None:
            return {"saved": False, "reason": "not_initialized"}
        root = self.config.rollout.drafter.get("checkpoint_path")
        if not root:
            raise ValueError("Online Ascend checkpoint_path is required")
        tag = f"draft_step_{int(global_step)}"
        if self._online_failed:
            raise RuntimeError("Cannot checkpoint a failed online Ascend session")
        rng_by_rank = [None] * dist.get_world_size()
        lifecycle = getattr(self, "_state_lifecycle", None)
        if lifecycle:
            rng_by_rank = lifecycle.gather(self._numerical_rng)
        else:
            dist.all_gather_object(rng_by_rank, self._numerical_rng)
        state = {
            "identity": self.identity,
            "progress": asdict(self.ascend_engine.progress),
            "rng_by_rank": rng_by_rank,
        }
        with bounded_checkpoint_writes(self.model, root):
            if (
                self.model.save_checkpoint(
                    str(root), tag=tag, client_state={"speco_online": state}
                )
                is False
            ):
                raise RuntimeError("DeepSpeed online checkpoint save failed")
        error = None
        if dist.get_rank() == 0:
            try:
                target = Path(root) / tag / "speco_online.json"
                temporary = target.with_suffix(".json.incomplete")
                temporary.write_text(
                    json.dumps(
                        {"identity": self.identity, "global_step": int(global_step)}
                    )
                )
                os.replace(temporary, target)
            except Exception as exc:  # noqa: BLE001 -- propagate to every participant
                error = repr(exc)
        self._collective_errors(error)
        return {
            "saved": True,
            "completed": True,
            "checkpoint_path": str(Path(root) / tag),
        }

    def wait_checkpoint(self):
        return {"waited": False, "completed": True}


def build_online_trainer(config, **kwargs):
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    class VllmAscendOnlineTrainer(VllmAscendOnlineMixin, DrafterBaseTrainer):
        pass

    return VllmAscendOnlineTrainer(config, **kwargs)
