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
"""Environment switches shared by every verl SGLang patch.

Kept free of SGLang imports so runtime hooks that run before (or without) the main
``sglang_patch`` module -- e.g. the ROCm memory-capacity fallback in SGLang server actors --
honor the same controls.
"""

import os

DISABLE_SGLANG_PATCH_ENV = "VERL_DISABLE_SGLANG_PATCH"


def env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n", ""}:
        return False
    return default


def sglang_verl_patches_disabled() -> bool:
    """Whether ``VERL_DISABLE_SGLANG_PATCH`` turns off all verl SGLang patches."""
    return env_flag_enabled(DISABLE_SGLANG_PATCH_ENV, default=False)
