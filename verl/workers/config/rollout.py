# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.model import MtpConfig

__all__ = [
    "SamplingConfig",
    "DiffusionSamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
    "DiffusionRolloutConfig",
    "CheckpointEngineConfig",
    "SkipConfig",
    "DrafterConfig",
    "DrafterRolloutConfig",
    "DrafterTrainingConfig",
]


@dataclass
class SkipConfig(BaseConfig):
    """
    Configuration for rollout skip: load/dump previously generated rollout data
    instead of computing new rollouts (e.g. for debugging or reuse).
    """

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    max_dump_step: int = 1
    action: str = "cache"  # cache | repeat | repeat_last

    def get(self, key: str, default=None):
        """Dict-like get for compatibility with code that uses skip.get('enable', False)."""
        return getattr(self, key, default)


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class DiffusionSamplingConfig(SamplingConfig):
    num_inference_steps: int = 40
    seed: int = 42
    extra_configs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DrafterRolloutConfig(BaseConfig):
    spec_steps: int = 3
    spec_topk: int = 1
    spec_verify_tokens: int = 3


@dataclass
class DrafterTrainingConfig(BaseConfig):
    collect_hidden_states_from_sgl: bool = False
    use_data_buffer: bool = False
    collect_interval_steps: int = 5
    collection_sample_rate: float = 1.0
    max_collect_samples_per_step_per_replica: Optional[int] = 16
    max_collect_tokens_per_step_per_replica: Optional[int] = 8192
    hidden_state_front_tokens_per_sample: Optional[int] = 2000
    hidden_state_max_tokens_per_sample: Optional[int] = None
    max_seq_len: int = 8192
    step: int = 10
    batch_size_per_gpu: int = 4
    training_interval_steps: int = 10
    publish_interval_steps: int = 0
    publish_async: bool = False
    publish_dtype: Optional[str] = None
    publish_param_name_patterns: Optional[list[str]] = None
    draft_update_weights_bucket_megabytes: Optional[int] = None
    draft_update_pause_generation: bool = True
    draft_update_flush_before: bool = True
    draft_update_flush_after: bool = True
    save_full_drafter_checkpoint: bool = False
    sample_last_n_steps: int = 20
    train_batches_per_cycle: int = 4
    lr: float = 1e-6
    lr_warmup_steps: int =1000
    min_lr_ratio: int = None
    warmup_style: str = "constant"
    use_logits: bool = False
    logits_topk: int = 128
    logits_loss_mode: str = "dense_tail"
    logits_sparse_min_intersection: int = 1
    logits_sparse_min_mass: Optional[float] = None
    logits_coverage_mask_min_ratio: Optional[float] = None
    logits_coverage_mask_require_top1: bool = False
    ttt_length: int = 1
    vocab_mapping_path: Optional[str] = None
    dflash_block_size: int = 16
    dflash_num_anchors: int = 512
    dflash_loss_decay_gamma: float = 7.0
    dflash_hidden_size: Optional[int] = None
    dflash_num_target_layers: int = 5
    dflash_num_hidden_layers: int = 1
    dflash_mask_token_id: Optional[int] = None
    dflash_target_layer_ids: Optional[list[int]] = None
    dflash_max_window: int = 512
    current_max_samples: int = 2000
    data_buffer_max_size: int = 1024
    hidden_state_clip_value: Optional[float] = 1.0e4

    def __post_init__(self):
        if self.collection_sample_rate < 0 or self.collection_sample_rate > 1:
            raise ValueError("`collection_sample_rate` must be in [0, 1].")
        if (
            self.max_collect_samples_per_step_per_replica is not None
            and self.max_collect_samples_per_step_per_replica < 0
        ):
            raise ValueError("`max_collect_samples_per_step_per_replica` must be non-negative or null.")
        if (
            self.max_collect_tokens_per_step_per_replica is not None
            and self.max_collect_tokens_per_step_per_replica < 0
        ):
            raise ValueError("`max_collect_tokens_per_step_per_replica` must be non-negative or null.")
        if self.hidden_state_max_tokens_per_sample is not None and self.hidden_state_max_tokens_per_sample < 0:
            raise ValueError("`hidden_state_max_tokens_per_sample` must be non-negative or null.")
        if self.hidden_state_front_tokens_per_sample is not None and self.hidden_state_front_tokens_per_sample < 0:
            raise ValueError("`hidden_state_front_tokens_per_sample` must be non-negative or null.")
        if (
            self.logits_coverage_mask_min_ratio is not None
            and (self.logits_coverage_mask_min_ratio < 0 or self.logits_coverage_mask_min_ratio > 1)
        ):
            raise ValueError("`logits_coverage_mask_min_ratio` must be in [0, 1] or null.")
        if self.logits_loss_mode not in {"dense_tail", "sparse_restricted"}:
            raise ValueError("`logits_loss_mode` must be either 'dense_tail' or 'sparse_restricted'.")
        if self.publish_interval_steps < 0:
            raise ValueError("`publish_interval_steps` must be non-negative.")
        if self.publish_dtype is not None and self.publish_dtype not in {"float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError("`publish_dtype` must be one of float32/fp32, float16/fp16, bfloat16/bf16, or null.")
        if self.draft_update_weights_bucket_megabytes is not None and self.draft_update_weights_bucket_megabytes <= 0:
            raise ValueError("`draft_update_weights_bucket_megabytes` must be positive or null.")
        if self.logits_sparse_min_intersection < 1:
            raise ValueError("`logits_sparse_min_intersection` must be positive.")
        if (
            self.logits_sparse_min_mass is not None
            and (self.logits_sparse_min_mass < 0 or self.logits_sparse_min_mass > 1)
        ):
            raise ValueError("`logits_sparse_min_mass` must be in [0, 1] or null.")
        if self.batch_size_per_gpu <= 0:
            raise ValueError("`batch_size_per_gpu` must be positive.")


@dataclass
class DrafterConfig(BaseConfig):

    enable: bool = False
    enable_drafter_training: bool = False
    speculative_algorithm: str = "EAGLE3"

    model_path: str ="/path/to/drafter/model"
    checkpoint_path: str ="/path/to/drafter/checkpoint"

    world_size: int = 16

    # rollout configuration for drafter
    rollout: DrafterRolloutConfig = field(default_factory=DrafterRolloutConfig)

    # training configuration for drafter
    training: DrafterTrainingConfig = field(default_factory=DrafterTrainingConfig)


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    # Fully qualified class name for custom AgentLoopManager (e.g., "mypackage.module.MyManager").
    # Security: This class will be dynamically imported via importlib. Only use trusted class paths.
    agent_loop_manager_class: Optional[str] = None


@dataclass
class TraceConfig(BaseConfig):
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None
    backend: Optional[str] = None
    token2text: bool = False
    max_samples_per_step_per_worker: Optional[int] = None

    def __post_init__(self):
        if self.max_samples_per_step_per_worker is not None and self.max_samples_per_step_per_worker < 0:
            raise ValueError("`max_samples_per_step_per_worker` must be a non-negative integer or null.")


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class PrometheusConfig(BaseConfig):
    """
    Configuration for Prometheus server
    """

    # whether enable prometheus on server mode rollout
    enable: bool = False
    # Port number that Prometheus listens on, default is 9090
    port: int = 9090
    # Path to Prometheus configuration file
    file: str = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    # Specify served_model_name to avoid displaying overly long model paths in Grafana
    served_model_name: Optional[str] = None


@dataclass
class CheckpointEngineConfig(BaseConfig):
    """
    Configuration for checkpoint engine to update weights from trainer to rollout
    """

    # Backend for checkpoint engine: naive, nccl, nixl, hccl
    backend: Optional[str] = "naive"
    # Bucket size in MB to transfer multiple weights at one time
    update_weights_bucket_megabytes: int = 2048
    # Additional keyword arguments for checkpoint engine
    engine_kwargs: dict = field(default_factory=dict)
    # If set, this Python module is imported on every worker process before the
    # backend is instantiated, allowing custom backends to register themselves
    # in CheckpointEngineRegistry.
    custom_backend_module: Optional[str] = None


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {
        "max_model_len",
        "load_format",
        "engine_kwargs",
        "prompt_length",
        "response_length",
        "expert_parallel_size",
        "moe_tensor_parallel_size",
    }

    name: Optional[str] = MISSING
    mode: str = "async"
    nnodes: int = 0
    n_gpus_per_node: int = 8

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1
    repetition_penalty: float = 1.0

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Extension point for custom configurations
    custom: Optional[dict] = None

    # Fully qualified class name for a custom CheckpointEngineManager. When set, the trainer
    # loads this class instead of the built-in CheckpointEngineManager.
    checkpoint_manager_class: Optional[str] = None

    # Checkpoint Engine config for update weights from trainer to rollout
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)

    # Rollout skip config (load/dump rollout data)
    skip: SkipConfig = field(default_factory=SkipConfig)

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    quantization: Optional[str] = None

    quantization_config_file: Optional[str] = None

    enable_rollout_routing_replay: bool = False

    enable_sleep_mode: bool = True

    mtp: MtpConfig = field(default_factory=MtpConfig)

    qat: Optional[dict] = None

    drafter: DrafterConfig = field(default_factory=DrafterConfig)

    def __post_init__(self):
        """Validate the rollout config"""
        # Deprecation warning for mode field - only async mode is supported
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )
        if self.mode != "async":
            warnings.warn(
                f"Unknown rollout mode '{self.mode}'. Only 'async' mode is supported. "
                "The 'mode' field is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.name != "trtllm" and self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size > 1:
            assert self.name == "trtllm", "moe_tensor_parallel_size is only supported for trtllm"

        if self.name == "trtllm":
            # If either expert_parallel_size or moe_tensor_parallel_size is at default 1,
            # convert to None so TensorRT-LLM treats it as unspecified.
            # When both unspecified: moe_ep_size=1, moe_tp_size=moe_world_size (no EP, all TP).
            # When only one set: the other is auto-derived from tensor_model_parallel_size.
            if self.expert_parallel_size is not None and self.expert_parallel_size == 1:
                self.expert_parallel_size = None
            if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size == 1:
                self.moe_tensor_parallel_size = None
            if self.expert_parallel_size is not None and self.moe_tensor_parallel_size is not None:
                assert self.moe_tensor_parallel_size * self.expert_parallel_size == self.tensor_model_parallel_size, (
                    "moe_tensor_parallel_size * expert_parallel_size must equal tensor_model_parallel_size "
                    f"(got {self.moe_tensor_parallel_size} * {self.expert_parallel_size} = "
                    f"{self.moe_tensor_parallel_size * self.expert_parallel_size}, "
                    f"tensor_model_parallel_size={self.tensor_model_parallel_size})"
                )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang" or self.name == "trtllm":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )


@dataclass
class DiffusionRolloutConfig(RolloutConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    val_kwargs: DiffusionSamplingConfig = field(default_factory=DiffusionSamplingConfig)

    # diffusion use
    height: int = 512

    width: int = 512

    num_inference_steps: int = 10

    extra_configs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate diffusion rollout config"""
        super().__post_init__()

        if self.pipeline_model_parallel_size > 1 and self.name == "vllm_omni":
            raise NotImplementedError(
                f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
            )
