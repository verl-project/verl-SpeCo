"""
Modeling layer for Eagle-style drafters.
"""

# Import specific implementations for direct access
from .dflash import DFlashConfig, DFlashDraftModel
from .eagle.llama_eagle import LlamaForCausalLMEagle, LlamaForCausalLMEagle3
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel, AutoEagleDraftModel

# __all__ defines what is exported when someone does 'from modeling import *'
__all__ = [
    "LlamaForCausalLMEagle",
    "LlamaForCausalLMEagle3",
    "DFlashConfig",
    "DFlashDraftModel",
    "AutoDraftModelConfig",
    "AutoEagleDraftModel",
    "AutoEagle3DraftModel",
]