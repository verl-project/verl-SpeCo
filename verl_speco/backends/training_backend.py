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
"""One lazy execution boundary for Drafter forward/backward implementations.

``training.backend`` selects execution, while ``speculative_algorithm`` selects
the model/loss family. A missing backend retains the original PyTorch path.
"""

from __future__ import annotations

import importlib
from typing import Any

SUPPORTED_TRAINING_BACKENDS = ("pytorch", "vllm_ascend")


def _get(node: Any, name: str, default: Any = None) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    return getter(name, default) if callable(getter) else getattr(node, name, default)


def resolve_training_backend(config: Any) -> str:
    """Accept a full config or its actor_rollout_ref section without imports."""
    section = _get(config, "actor_rollout_ref", config)
    drafter = _get(_get(section, "rollout"), "drafter")
    raw = _get(_get(drafter, "training"), "backend", "pytorch")
    backend = str(raw).strip().lower()
    if backend not in SUPPORTED_TRAINING_BACKENDS:
        raise ValueError(
            f"Drafter training.backend must be pytorch or vllm_ascend, got {raw!r}"
        )
    return backend


def is_vllm_ascend_drafter(drafter: Any) -> bool:
    """Check a bare drafter section; absent backend preserves native behavior."""
    return resolve_training_backend({"rollout": {"drafter": drafter}}) == "vllm_ascend"


def actor_capture_enabled(drafter: Any) -> bool:
    """Opt-in DeepSeek V4 actor source; native backends retain their routing."""
    training = _get(drafter, "training", {})
    return (
        is_vllm_ascend_drafter(drafter)
        and bool(_get(drafter, "enable_drafter_training", False))
        and _get(training, "mode", "online") == "online"
        and bool(_get(training, "collect_hidden_states_from_old_logprob", False))
    )


def rollout_capture_enabled(drafter: Any) -> bool:
    """Direct capture in verl rollout was removed; retained for old callers."""
    return False


def require_pytorch_training(config: Any) -> None:
    """Prevent unsupported online/native paths from ignoring the new selector."""
    if resolve_training_backend(config) != "pytorch":
        raise ValueError(
            "The native model factory requires training.backend=pytorch. "
            "backend=vllm_ascend uses the dedicated SpecoWorker adapter for online "
            "training and verl_speco.draft_train for offline training. "
            "This native model factory cannot execute external online/TQ training."
        )


def run_standalone_training(config: Any) -> dict[str, Any]:
    """Select exactly one training implementation before constructing models."""
    backend = resolve_training_backend(config)
    if backend == "pytorch":
        native = importlib.import_module("verl_speco.trainer.draft_training_loop")
        native.log_resolved_config(config)
        return native.run_standalone_draft_training(config)
    runtime = importlib.import_module("verl_speco.trainer.vllm_ascend_training_loop")
    return runtime.run_vllm_ascend_training(config)


async def run_standalone_training_async(
    config: Any,
    *,
    scheduled_commands: Any | None = None,
    training_events: Any | None = None,
) -> dict[str, Any]:
    """Use the same selector in the existing Ray WorkerGroup entrypoint."""
    backend = resolve_training_backend(config)
    if backend == "pytorch":
        native = importlib.import_module("verl_speco.trainer.draft_training_loop")
        return await native._run_standalone_draft_training_async(
            config,
            scheduled_commands=scheduled_commands,
            training_events=training_events,
        )
    if scheduled_commands is not None or training_events is not None:
        raise ValueError(
            "backend=vllm_ascend does not yet support scheduled TransferQueue "
            "execution; use offline torch_shard input with draft_train_launcher"
        )
    # Keep NPU initialization on the worker thread, not an asyncio thread pool.
    runtime = importlib.import_module("verl_speco.trainer.vllm_ascend_training_loop")
    return runtime.run_vllm_ascend_training(config)
