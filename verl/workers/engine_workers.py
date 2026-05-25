# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import functools
import logging
import os
import random
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    make_nd_compute_dispatch_fn,
    register,
)
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, is_npu_available, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine.fsdp.utils import (
    build_drafter_training_device_mesh,
    build_rollout_parallel_layout,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import diffusion_loss, ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DRAFTER_OWNER_ROUTE_MESH = "drafter_owner_route"


def _resolve_drafter_init_backend(device_name: str) -> str:
    device_name = str(device_name).lower()
    if device_name == "npu":
        return "cpu:gloo,npu:hccl"
    if device_name == "cuda":
        return "cpu:gloo,cuda:nccl"
    if device_name == "cpu":
        return "cpu:gloo"
    raise ValueError(f"Unsupported drafter device_name={device_name!r}")


@contextmanager
def _preserve_process_rng_state(device_name: str):
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_cpu_rng_state = torch.get_rng_state()
    torch_device_rng_state = None
    torch_device = None

    if str(device_name).lower() != "cpu":
        torch_device = get_torch_device()
        try:
            torch_device_rng_state = torch_device.get_rng_state()
        except (AttributeError, RuntimeError):
            torch_device_rng_state = None

    try:
        yield
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_cpu_rng_state)
        if torch_device is not None and torch_device_rng_state is not None:
            try:
                torch_device.set_rng_state(torch_device_rng_state)
            except (AttributeError, RuntimeError):
                logger.warning("Failed to restore %s RNG state after drafter training.", device_name)


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        # TODO(jhz): Switch to `set_expandable_segments` when the torch_npu library
        # supports `torch.npu.memory._set_allocator_settings`
        if is_npu_available:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            # for Diffusion models, FlopsCounter is not supported yet.
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.actor.strategy == "megatron" and self.config.actor.megatron.router_replay.mode != "disabled"
        )

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            actor_config.model_config.freeze_vision_tower = actor_config.freeze_vision_tower
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            elif model_config.get("model_type", "language_model") == "diffusion_model":
                self.loss_fn = partial(diffusion_loss, config=actor_config)
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh, full_config=self.config
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_lm_head_weight(self):
        if not self._is_actor or self.actor is None:
            return None

        per_tensor_param, _ = self.actor.engine.get_per_tensor_param(
            layered_summon=getattr(self, "layered_summon", False),
            base_sync_done=True,
        )
        selected_name = None
        selected_weight = None
        fallback_name = None
        fallback_weight = None

        for name, tensor in per_tensor_param:
            if not torch.is_tensor(tensor):
                continue
            name = str(name)
            if name == "model.embed_tokens.weight" or name.endswith(".embed_tokens.weight"):
                fallback_name = name
                fallback_weight = tensor
            if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
                selected_name = name
                selected_weight = tensor
                break

        if selected_weight is None:
            selected_name = fallback_name
            selected_weight = fallback_weight

        if self.rank != 0:
            return None
        if selected_weight is None:
            logger.warning("Unable to find actor lm_head.weight or tied model.embed_tokens.weight for drafter sync")
            return None

        weight = selected_weight.detach().cpu().to(torch.bfloat16).contiguous()
        logger.warning(
            "[actor lm_head export] name=%s shape=%s dtype=%s",
            selected_name,
            tuple(weight.shape),
            weight.dtype,
        )
        return {"name": selected_name, "weight": weight}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def update_draft_weights(self, weights: dict[str, torch.Tensor], global_steps: int = None):
        if not self.config.rollout.drafter.enable:
            return
        await self.rollout.update_draft_weights(weights, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_draft_weights_async(self, weights: dict[str, torch.Tensor], global_steps: int = None):
        if not self.config.rollout.drafter.enable:
            return
        await self.rollout.update_draft_weights(weights, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.
        """

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if self.config.rollout.checkpoint_engine.backend != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


class DrafterWorker(Worker):
    """Standalone drafter worker that runs in parallel with actor/rollout workers.

    The worker receives CPU rollout features from the PPO trainer and trains the
    drafter model periodically according to global RL steps.
    """

    def __init__(self, config: DictConfig, role: str = "drafter", device_name: Optional[str] = None, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        if device_name is None:
            raise ValueError("DrafterWorker requires an explicit device_name from the trainer initialization path")
        self.device_name = str(device_name).lower()
        self.trainer = None
        self.last_global_step = None
        self.last_trained_step = None
        self.training_process_group = None
        self.dp_process_group = None
        self.training_group_ranks = []
        self.training_group_world_size = 1
        self.dp_group_ranks = []
        self.dp_group_world_size = 1
        self.num_rollout_replicas = 1
        self.training_device_mesh = None
        self._process_group_initialized = False
        self._training_group_initialized = False

        self.rollout_tp = int(self.config.rollout.tensor_model_parallel_size)
        self.rollout_dp = int(self.config.rollout.data_parallel_size)
        self.infer_tp = self.rollout_tp * self.rollout_dp
        self.rollout_pp = int(self.config.rollout.pipeline_model_parallel_size)
        self.rollout_world_size = self.infer_tp * self.rollout_pp
        self.rollout_rank = self.rank % self.rollout_world_size
        self.replica_rank = self.rank // self.rollout_world_size
        self.local_infer_tp_rank = self.rollout_rank // self.rollout_pp
        self.local_infer_pp_rank = self.rollout_rank % self.rollout_pp
        self.local_drafter_sp_rank = None
        self.in_drafter_train_group = False
        self.is_drafter_group_leader = False
        self.global_publish_leader_rank = None
        self.is_global_publish_leader = False

        self.enable_drafter = bool(
            self.config.rollout.drafter.enable and self.config.rollout.drafter.enable_drafter_training
        )
        self.training_interval_steps = int(self.config.rollout.drafter.training.get("training_interval_steps", 1))
        self.publish_interval_steps = int(self.config.rollout.drafter.training.get("publish_interval_steps", 0))
        self.train_steps_per_trigger = int(self.config.rollout.drafter.training.get("step", 100))

    def _ensure_process_group_initialized(self):
        if not dist.is_initialized():
            initialize_global_process_group_ray(
                timeout_second=None,
                backend=_resolve_drafter_init_backend(self.device_name),
            )
        if not self._process_group_initialized:
            set_numa_affinity()
            self._process_group_initialized = True
        if dist.is_initialized() and dist.get_rank() != self.rank:
            raise RuntimeError(
                f"DrafterWorker rank mismatch: worker_rank={self.rank}, dist_rank={dist.get_rank()}"
            )

    def _ensure_training_group_initialized(self):
        if self._training_group_initialized:
            return

        self._ensure_process_group_initialized()
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        rollout_layout = build_rollout_parallel_layout(
            world_size=world_size,
            rollout_tp=self.rollout_tp,
            rollout_dp=self.rollout_dp,
            rollout_pp=self.config.rollout.pipeline_model_parallel_size,
        )
        self.num_rollout_replicas = rollout_layout.num_replicas

        self.global_publish_leader_rank = (
            rollout_layout.replica_training_ranks[0][0] if rollout_layout.replica_training_ranks else None
        )
        self.is_global_publish_leader = self.rank == self.global_publish_leader_rank
        self.training_device_mesh = build_drafter_training_device_mesh(self.device_name, rollout_layout)
        owner_route_rank = self.num_rollout_replicas
        owner_route_collect = False

        mesh_coordinate = self.training_device_mesh.get_coordinate()
        self.in_drafter_train_group = mesh_coordinate is not None
        if self.in_drafter_train_group:
            mesh_dp_rank, mesh_sp_rank = mesh_coordinate
            if mesh_dp_rank != self.replica_rank:
                raise ValueError(
                    "Drafter mesh dp coordinate does not match rollout replica rank: "
                    f"mesh_dp_rank={mesh_dp_rank}, rollout_replica_rank={self.replica_rank}, "
                    f"global_rank={self.rank}"
                )
            self.training_process_group = self.training_device_mesh["sp"].get_group()
            self.dp_process_group = self.training_device_mesh["dp"].get_group()
            self.training_group_ranks = list(rollout_layout.replica_training_ranks[mesh_dp_rank])
            self.dp_group_ranks = [
                rollout_layout.replica_training_ranks[replica_rank][mesh_sp_rank]
                for replica_rank in range(rollout_layout.num_replicas)
            ]
            self.training_group_world_size = self.training_device_mesh["sp"].size()
            self.dp_group_world_size = self.training_device_mesh["dp"].size()
            self.local_drafter_sp_rank = mesh_sp_rank
            self.is_drafter_group_leader = mesh_sp_rank == 0
            owner_route_rank = self.replica_rank
            owner_route_collect = self.is_drafter_group_leader

        self._register_dispatch_collect_info(
            mesh_name=DRAFTER_OWNER_ROUTE_MESH,
            dp_rank=owner_route_rank,
            is_collect=owner_route_collect,
        )
        self._training_group_initialized = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if not self.enable_drafter:
            return

        self._ensure_training_group_initialized()
        if not self.in_drafter_train_group:
            return

        from verl.workers.drafter.base_trainer import DrafterBaseTrainer
        from verl.workers.drafter.dflash_trainer_backend import DFlashTrainerBackend
        from verl.workers.drafter.eagle3_trainer_backend import Eagle3TrainerBackend
        from verl.workers.drafter.eagle_trainer_backend import EagleTrainerBackend

        algo = str(self.config.rollout.drafter.speculative_algorithm).upper()
        if algo == "EAGLE":
            trainer_backend = EagleTrainerBackend(self.config, self.config.model)
        elif algo == "EAGLE3":
            trainer_backend = Eagle3TrainerBackend(self.config, self.config.model)
        elif algo == "DFLASH":
            trainer_backend = DFlashTrainerBackend(self.config, self.config.model)
        else:
            raise ValueError(f"Unknown drafter algorithm: {self.config.rollout.drafter.speculative_algorithm}")

        # Build drafter around an explicit training mesh; process groups are now
        # derived views from the mesh for compatibility with APIs that still take groups.
        self.trainer = DrafterBaseTrainer(
            config=self.config,
            world_size=self.training_group_world_size,
            rollout_dp_rank=self.replica_rank,
            training_device_mesh=self.training_device_mesh,
            backend=trainer_backend,
        )

    def _store_rollout_sample(
        self,
        batch: dict,
        hidden_states: torch.Tensor,
        target_logprobs: Optional[torch.Tensor] = None,
    ):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return
        self.trainer.collect_online_data(batch, hidden_states, target_logprobs)

    @register(dispatch_mode=make_nd_compute_dispatch_fn(mesh_name=DRAFTER_OWNER_ROUTE_MESH))
    def collect_rollout_features(self, samples: list[dict]):
        if not samples:
            return
        for sample in samples:
            if not sample:
                continue
            batch = {
                "input_ids": sample["input_ids"],
                "prompts": sample["prompts"],
                "responses": sample["responses"],
            }
            for key in (
                "hidden_position_start",
                "hidden_position_end",
                "hidden_prefix_cache_rows",
                "hidden_window_start",
                "hidden_window_end",
                "target_logprobs_position_start",
                "target_logprobs_position_end",
            ):
                if key in sample:
                    batch[key] = sample[key]
            self._store_rollout_sample(
                batch=batch,
                hidden_states=sample["hidden_states"],
                target_logprobs=sample.get("target_logprobs"),
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_global_step(self, global_step: int):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return
        if global_step is None:
            return
        if self.last_global_step == global_step:
            return
        self.last_global_step = global_step
        self.trainer.clear_pending_publish_state_dict()
        self.trainer.increment_rl_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_target_lm_head_weight(self, payload: Optional[dict], global_step: Optional[int] = None):
        if not self.enable_drafter:
            return {"accepted": False, "applied": False, "reason": "disabled"}
        if not self.in_drafter_train_group or self.trainer is None:
            return {"accepted": False, "applied": False, "reason": "not_in_training_group"}
        if not payload:
            return {"accepted": False, "applied": False, "reason": "missing_payload"}

        weight = payload.get("weight")
        name = payload.get("name")
        result = self.trainer.sync_target_lm_head_weight(weight, global_step=global_step)
        if self.is_drafter_group_leader:
            logger.warning(
                "[drafter target lm_head sync] replica=%s source=%s global_step=%s result=%s",
                self.replica_rank,
                name,
                global_step,
                result,
            )
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def train_drafter(self):
        result = {
            "trained": False,
            "triggered": False,
            "successful_steps": 0,
            "attempted_steps": 0,
            "elapsed_sec": 0.0,
            "reason": "",
        }
        if not self.enable_drafter:
            result["reason"] = "disabled"
            return result
        if not self.in_drafter_train_group or self.trainer is None:
            result["reason"] = "not_in_training_group"
            return result
        if self.last_global_step is None:
            result["reason"] = "missing_global_step"
            return result
        if self.training_interval_steps <= 0:
            result["reason"] = "invalid_interval"
            return result
        if self.last_global_step % self.training_interval_steps != 0:
            result["reason"] = "interval_not_reached"
            return result

        with _preserve_process_rng_state(self.device_name):
            result["triggered"] = True
            start_ts = time.time()
            self.trainer.clear_pending_publish_state_dict()
            success = await self.trainer.activate_training_model()
            if not success:
                logger.error(
                    f"[DrafterWorker replica={self.replica_rank}] failed to activate trainer at step {self.last_global_step}"
                )
                self.trainer.clear_pending_publish_state_dict()
                await self.trainer.cleanup_training(clear_data=False)
                result["reason"] = "activation_failed"
                result["elapsed_sec"] = time.time() - start_ts
                return result

            try:
                for _ in range(self.train_steps_per_trigger):
                    result["attempted_steps"] += 1
                    step_ok = await self.trainer.training_step(self.last_global_step)
                    if step_ok:
                        result["successful_steps"] += 1
                if result["successful_steps"] > 0:
                    should_prepare_publish = (
                        self.publish_interval_steps <= 0
                        or self.last_global_step % self.publish_interval_steps == 0
                    )
                    if should_prepare_publish:
                        snapshot_ts = time.time()
                        cached = self.trainer.prepare_model_state_dict_for_publish(self.last_global_step)
                        result["publish_snapshot_cached"] = int(cached)
                        result["publish_snapshot_elapsed_sec"] = time.time() - snapshot_ts
                    else:
                        self.trainer.clear_pending_publish_state_dict()
                else:
                    self.trainer.clear_pending_publish_state_dict()
            finally:
                await self.trainer.cleanup_training(clear_data=result["successful_steps"] > 0)

            result["trained"] = result["successful_steps"] > 0
            result["reason"] = "trained" if result["trained"] else "no_trainable_batch"
            if result["trained"]:
                self.last_trained_step = self.last_global_step
            result["elapsed_sec"] = time.time() - start_ts
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def maybe_publish(self):
        if not self.enable_drafter or not self.in_drafter_train_group or self.trainer is None:
            return None
        if self.last_global_step is None:
            return None
        if self.training_interval_steps <= 0 or self.last_global_step % self.training_interval_steps != 0:
            return None
        if self.last_trained_step != self.last_global_step:
            return None
        has_snapshot, weights = self.trainer.pop_model_state_dict_for_publish(self.last_global_step)
        if not has_snapshot:
            logger.warning(
                "[DrafterWorker replica=%s rank=%s] Missing cached drafter publish snapshot at step %s; skip publish.",
                self.replica_rank,
                self.rank,
                self.last_global_step,
            )
            return None
        return weights if self.is_global_publish_leader else None
