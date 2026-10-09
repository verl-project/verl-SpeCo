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
"""Hydra config package for SPECO."""

from __future__ import annotations

from typing import Any


def config_int(config: Any, key: str, default: int) -> int:
    """Resolve an integer config value, defaulting only when absent or ``None``.

    Unlike ``config.get(key, default) or default`` this keeps an explicit ``0``,
    which several knobs use to mean "fail on the first bad row".
    """
    value = config.get(key)
    return default if value is None else int(value)


__all__ = ["config_int"]
