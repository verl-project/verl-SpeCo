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
