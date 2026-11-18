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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/internal_router.py.
from collections.abc import Callable, Iterable
from contextlib import ExitStack, contextmanager
from functools import wraps

import torch
from torch import nn


def find_internal_router_targets(root) -> tuple[object, ...]:
    """Find concrete gate-owning runners below a fused-MoE wrapper."""

    targets = []
    pending = [root]
    visited = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)

        is_internal = bool(getattr(candidate, "is_internal_router", False))
        owns_gate = hasattr(candidate, "gate") or hasattr(candidate, "_gate")
        if is_internal and owns_gate:
            targets.append(candidate)

        for attribute in ("runner", "_runner", "moe_runner"):
            child = getattr(candidate, attribute, None)
            if child is not None:
                pending.append(child)
        children = getattr(candidate, "children", None)
        if callable(children):
            pending.extend(children())
    return tuple(targets)


@contextmanager
def disable_runner_internal_router(experts):
    """Temporarily select a runner's external-router execution branch."""

    has_private_gate = hasattr(experts, "_gate")
    original_private_gate = getattr(experts, "_gate", None)
    has_public_gate = hasattr(experts, "gate")
    original_public_gate = getattr(experts, "gate", None)
    public_gate_replaced = False
    try:
        if has_public_gate:
            try:
                experts.gate = None
                public_gate_replaced = True
            except (AttributeError, TypeError):
                # Older runners expose gate as a read-only property backed by
                # _gate; the private assignment below handles that layout.
                pass
        if has_private_gate:
            experts._gate = None
        if bool(getattr(experts, "is_internal_router", False)):
            raise RuntimeError(
                "Failed to disable the runner's internal-router branch; "
                "unsupported gate storage layout"
            )
        yield
    finally:
        if public_gate_replaced:
            experts.gate = original_public_gate
        if has_private_gate:
            experts._gate = original_private_gate


def patch_model_internal_router_training_forward(
    moe_types: Iterable[type[nn.Module]],
    *,
    on_patched_call: Callable[[], None] | None = None,
) -> bool:
    """Route grad-enabled model forwards through trainable gate weights."""

    patched_any = False
    for moe_type in moe_types:
        if getattr(
            moe_type,
            "_speco_deepseek_v4_internal_router_training_patch",
            False,
        ):
            patched_any = True
            continue

        original_forward = moe_type.forward

        @wraps(original_forward)
        def training_forward(
            self,
            hidden_states,
            *args,
            __original_forward=original_forward,
            **kwargs,
        ):
            experts = self.experts
            if not torch.is_grad_enabled() or not bool(
                getattr(experts, "is_internal_router", False)
            ):
                return __original_forward(
                    self,
                    hidden_states,
                    *args,
                    **kwargs,
                )
            gate = getattr(self, "gate", None)
            if gate is None or not isinstance(
                getattr(gate, "weight", None),
                torch.Tensor,
            ):
                raise RuntimeError(
                    "DeepseekV4MoE internal router has no trainable gate.weight"
                )

            router_targets = find_internal_router_targets(experts)
            if not router_targets:
                raise RuntimeError(
                    "DeepseekV4MoE reports internal routing, but no concrete "
                    "gate-owning runner was found below self.experts"
                )

            # The model's external-router branch computes logits from the
            # model-owned gate while autograd is enabled. Disable every nested
            # runner that could otherwise overwrite those logits.
            with ExitStack() as stack:
                for router_target in router_targets:
                    stack.enter_context(disable_runner_internal_router(router_target))
                output = __original_forward(
                    self,
                    hidden_states,
                    *args,
                    **kwargs,
                )
            if on_patched_call is not None:
                on_patched_call()
            return output

        moe_type.forward = training_forward
        moe_type._speco_deepseek_v4_internal_router_training_patch = True
        patched_any = True
    return patched_any
