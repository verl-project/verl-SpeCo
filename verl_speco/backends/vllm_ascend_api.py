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
"""Lazy numerical boundary for the in-repository DeepSeek V4 implementation."""

from verl_speco.backends.vllm_ascend_model import load_mtp_draft_model
from verl_speco.backends.vllm_ascend_options import (
    export_memory_chunking_environment,
    resolve_memory_chunking,
    resolve_optimization_config,
)
from verl_speco.backends.vllm_ascend_parallel import (
    audit_data_parallel_parameter_replicas,
)
from verl_speco.backends.vllm_ascend_scheduler import (
    build_deepspeed_scheduler_config,
    resolve_warmup_steps,
)
from verl_speco.models.target.deepseek_v4_head import load_target_lm_head
from verl_speco.trainer.deepseek_v4_metrics import set_seed
from verl_speco.trainer.deepseek_v4_training import run_dspark_training

__all__ = [
    "audit_data_parallel_parameter_replicas",
    "build_deepspeed_scheduler_config",
    "export_memory_chunking_environment",
    "load_mtp_draft_model",
    "load_target_lm_head",
    "resolve_memory_chunking",
    "resolve_optimization_config",
    "resolve_warmup_steps",
    "run_dspark_training",
    "set_seed",
]
