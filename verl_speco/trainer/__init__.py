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
"""SPECO trainer adapters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

__all__ = ["SpecoRayPPOTrainer"]


def __getattr__(name: str) -> Any:
    """Load trainer adapters without importing Ray workers during package init."""
    if name == "SpecoRayPPOTrainer":
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        return SpecoRayPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
