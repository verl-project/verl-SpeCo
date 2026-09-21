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
from __future__ import annotations

import json
import os

from verl_speco.models.dflash import DFlashConfig

# Keys the upstream z-lab DFlash2 checkpoints nest under ``dflash_config``
# instead of putting at the config top level. ``from_dflash2_pretrained``
# lifts them so the rest of this overlay can read them as plain attributes.
_NESTED_DFLASH_KEYS = (
    "block_size",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
    "mask_token_id",
    "target_layer_ids",
)


class DFlash2Config(DFlashConfig):
    """Configuration for the DFlash2 draft model.

    DFlash2 keeps the DFlash target-context backbone and block-diffusion drafting
    and adds two modules on top (see https://inco.ai/blog/dflash2/):

    - a two-tap dynamic depthwise convolution wrapped around every attention and
      MLP sublayer, which counteracts the accuracy decay DFlash shows toward the
      end of a block;
    - a candidate selector that keeps ``selector_top_k`` candidates per block
      position and traces one coherent path through them with a low-rank
      bilinear score over adjacent candidates.

    Defaults match the released ``z-lab/Qwen3.8-27B-DFlash2`` checkpoint.
    """

    model_type = "dflash2"

    def __init__(
        self,
        *args,
        block_size: int = 8,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        conv_kernel_size: int = 2,
        conv_group_size: int = 16,
        selector_rank: int = 256,
        selector_top_k: int = 16,
        selector_loss_weight: float = 1.0,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", None)
        super().__init__(*args, **kwargs)
        self.architectures = architectures or ["DFlash2DraftModel"]
        self.block_size = int(block_size)
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = float(loss_decay_gamma)
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_group_size = int(conv_group_size)
        self.selector_rank = int(selector_rank)
        self.selector_top_k = int(selector_top_k)
        self.selector_loss_weight = float(selector_loss_weight)

    def to_dict(self):
        """Serialize with the z-lab ``dflash_config`` block alongside the flat keys.

        vLLM's DFlash2 draft reads the convolution and selector knobs strictly
        from ``dflash_config``, so a checkpoint saved by the trainer has to carry
        that block to be servable as a rollout drafter. The flat keys stay for
        this overlay's own loaders; ``from_dflash2_pretrained`` lets a top-level
        value win over the nested one, so the two never disagree.
        """
        output = super().to_dict()
        nested = dict(output.get("dflash_config") or {})
        for key in _NESTED_DFLASH_KEYS:
            if output.get(key) is not None:
                nested[key] = output[key]
        if nested:
            output["dflash_config"] = nested
        return output

    @classmethod
    def from_dflash2_pretrained(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Upstream checkpoints keep the DFlash2-specific knobs in a nested
        # ``dflash_config`` block; a top-level value (from a checkpoint this
        # overlay wrote itself) always wins.
        nested = config.get("dflash_config") or {}
        for key in _NESTED_DFLASH_KEYS:
            if key not in config and key in nested:
                config[key] = nested[key]

        config["model_type"] = cls.model_type
        config["architectures"] = ["DFlash2DraftModel"]
        return cls.from_dict(config)
