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
"""Deprecated source-helper API backed entirely by SpeCo's native modules.

Production code imports maintained modules directly. These aliases only keep
older tools that imported the former bridge from failing unnecessarily.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

from verl_speco.integration.vllm_ascend_implementation import implementation_fingerprint

_MODULES = {
    "offline_training.train_mtp": "verl_speco.backends.vllm_ascend_api",
    "training_common.modeling.ds4_dspark": "verl_speco.models.deepseek_v4.modeling_dspark",
    "training_common.utils.data": "verl_speco.data.deepseek_v4_sampler",
    "training_common.utils.parallel": "verl_speco.backends.vllm_ascend_parallel",
    "training_common.utils.optimization_config": "verl_speco.backends.vllm_ascend_options",
    "training_common.utils.ds_utils": "verl_speco.trainer.vllm_ascend_state",
    "loading_trained_mtp.loading_ds4": "verl_speco.models.deepseek_v4.export",
}


def source_fingerprint(root=None):
    return implementation_fingerprint()


def activate_source(root=None):
    warnings.warn(
        "External source activation is obsolete; SpeCo ships the DeepSeek V4 backend",
        DeprecationWarning,
        stacklevel=2,
    )
    return Path(__file__).parents[1]


def engine_module(name):
    try:
        module = _MODULES[name]
    except KeyError as error:
        raise ValueError(
            f"No native SpeCo compatibility module for {name!r}"
        ) from error
    return importlib.import_module(module)
