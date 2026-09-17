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

"""Factory for SPECO's verl V1 trainer adapters.

The V1 trainer owns the PPO loop and TransferQueue lifecycle.  SPECO only
adds a thin mixin so upstream fixes to sampling, reward, advantage, and
checkpoint handling remain available without copying that loop.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Type


@lru_cache(maxsize=1)
def _trainer_types() -> dict[str, Type]:
    from .speco_mixin import SpecoV1Mixin

    try:
        from verl.trainer.ppo.v1 import (
            PPOTrainerColocateAsync,
            PPOTrainerSeparateAsync,
            PPOTrainerSync,
        )
    except ImportError as exc:  # pragma: no cover - exercised in dependency checks
        raise RuntimeError(
            "SPECO V1 support requires verl release/v0.9.0 with verl.trainer.ppo.v1"
        ) from exc

    class SpecoV1SyncTrainer(SpecoV1Mixin, PPOTrainerSync):
        pass

    class SpecoV1ColocateAsyncTrainer(SpecoV1Mixin, PPOTrainerColocateAsync):
        pass

    class SpecoV1SeparateAsyncTrainer(SpecoV1Mixin, PPOTrainerSeparateAsync):
        pass

    # Make the lazily-created classes importable for Ray/cloudpickle.  The
    # upstream V1 trainer itself is instantiated on the driver, but workers
    # may still serialize trainer metadata during checkpoint callbacks.
    for trainer_cls in (
        SpecoV1SyncTrainer,
        SpecoV1ColocateAsyncTrainer,
        SpecoV1SeparateAsyncTrainer,
    ):
        trainer_cls.__module__ = __name__
        trainer_cls.__qualname__ = trainer_cls.__name__
        globals()[trainer_cls.__name__] = trainer_cls

    return {
        "sync": SpecoV1SyncTrainer,
        "colocate_async": SpecoV1ColocateAsyncTrainer,
        "separate_async": SpecoV1SeparateAsyncTrainer,
    }


def get_speco_v1_trainer_cls(mode: str) -> Type:
    """Return the SPECO V1 adapter for ``trainer.v1.trainer_mode``."""

    normalized = str(mode or "sync").strip().lower()
    trainers = _trainer_types()
    try:
        return trainers[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(trainers))
        raise ValueError(
            f"Unknown SPECO V1 trainer mode {normalized!r}; available: {available}"
        ) from exc
