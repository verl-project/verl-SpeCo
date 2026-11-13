# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4-Flash DSpark draft model package.

Exposes the config, the sparse-backbone draft model, and the released-weight
loader (key mapping + FP8/FP4/INT8/BF16 dequantization).
"""

from .configuration_dsv4_dspark import DSV4DSparkConfig
from .modeling_dsv4_dspark import DSV4DSparkDraftModel
from .weights import load_released_draft, map_released_key

__all__ = [
    "DSV4DSparkConfig",
    "DSV4DSparkDraftModel",
    "load_released_draft",
    "map_released_key",
]
