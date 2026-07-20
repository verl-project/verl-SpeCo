"""
Modeling layer for EAGLE3 and DFLASH drafters.
"""

# Import specific implementations for direct access
from .dflash import DFlashConfig, DFlashDraftModel
from .eagle.llama_eagle import LlamaForCausalLMEagle3
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .target import TargetHead

# __all__ defines what is exported when someone does 'from modeling import *'
__all__ = [
    "LlamaForCausalLMEagle3",
    "DFlashConfig",
    "DFlashDraftModel",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "TargetHead",
]
