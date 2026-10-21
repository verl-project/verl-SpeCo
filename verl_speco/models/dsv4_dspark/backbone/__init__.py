# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4-Flash sparse backbone (latent attention + MoE + mHC)."""

from __future__ import annotations

from .kernels import (
    get_kernel,
    register_kernel,
    set_active_backend,
    torch_kernel,
)
from .norm import RMSNorm, UnweightedRMSNorm

__all__ = [
    "RMSNorm",
    "UnweightedRMSNorm",
    "get_kernel",
    "register_kernel",
    "set_active_backend",
    "torch_kernel",
]
