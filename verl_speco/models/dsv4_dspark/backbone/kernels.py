# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device-agnostic op registry for the DSV4 backbone.

Every heavy op ships a pure-torch reference that is always registered and runs
on any device; accelerator implementations register under their device backend
name and are selected via ``set_active_backend``.
"""

from __future__ import annotations

from collections.abc import Callable

TORCH_BACKEND = "torch"

_REGISTRY: dict[tuple[str, str], Callable] = {}
_ACTIVE_BACKEND = TORCH_BACKEND


def register_kernel(op: str, backend: str, fn: Callable) -> None:
    _REGISTRY[(op, backend)] = fn


def torch_kernel(op: str) -> Callable[[Callable], Callable]:
    def _wrap(fn: Callable) -> Callable:
        register_kernel(op, TORCH_BACKEND, fn)
        return fn

    return _wrap


def set_active_backend(backend: str) -> str:
    global _ACTIVE_BACKEND
    prev, _ACTIVE_BACKEND = _ACTIVE_BACKEND, backend
    return prev


def get_kernel(op: str, backend: str | None = None) -> Callable:
    backend = backend or _ACTIVE_BACKEND
    fn = _REGISTRY.get((op, backend))
    if fn is not None:
        return fn
    fn = _REGISTRY.get((op, TORCH_BACKEND))
    if fn is None:
        raise KeyError(
            f"No implementation for op '{op}' (backend '{backend}' and no torch "
            "reference). Import the backbone modules so their torch references "
            "register, or register a kernel via register_kernel()."
        )
    return fn
