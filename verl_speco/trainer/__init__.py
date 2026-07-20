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
