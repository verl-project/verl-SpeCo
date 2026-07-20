"""SPECO worker adapters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verl_speco.workers.speco_worker import SpecoWorker

__all__ = ["SpecoWorker"]


def __getattr__(name: str) -> Any:
    """Load worker adapters without importing trainer modules during package init."""
    if name == "SpecoWorker":
        from verl_speco.workers.speco_worker import SpecoWorker

        return SpecoWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
