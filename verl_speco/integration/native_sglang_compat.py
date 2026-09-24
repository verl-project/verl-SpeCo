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
"""SGLang launch compatibility for SPECO's no-drafter native path."""

import logging
import os

try:
    from verl.trainer.main_ppo_v0 import BaseTaskRunner as _TaskRunnerBase
except ImportError:
    from verl.trainer.main_ppo import TaskRunner as _TaskRunnerBase

logger = logging.getLogger(__name__)


def install_native_sglang_compat(config) -> bool:
    """Install the bridge in the Ray TaskRunner process that launches servers."""

    del config
    from verl_speco.integration.sglang_runtime import (
        install_upstream_sglang_runtime_bridge,
    )

    installed = install_upstream_sglang_runtime_bridge(base_compat_only=True)
    if not installed:
        raise RuntimeError(
            "Failed to install SPECO's native SGLang launch compatibility "
            "inside the Ray TaskRunner process"
        )
    logger.warning(
        "SPECO native SGLang launch compatibility installed in pid=%s", os.getpid()
    )
    return True


class NativeSGLangCompatTaskRunner(_TaskRunnerBase):
    """Run native verl after installing only its SGLang launch compatibility."""

    def run(self, config):
        install_native_sglang_compat(config)
        return super().run(config)
