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
"""Training-only parameters omitted from the pretrained drafter export."""

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)


class DrafterAuxiliaryState:
    """Keep the VAT head in the optimizer checkpoint's DCP sidecar."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        if not any(
            "vat_objective.accept_head" in name
            for name, _ in self.model.named_modules()
        ):
            return {}
        state = get_model_state_dict(
            self.model, options=StateDictOptions(full_state_dict=False)
        )
        return {
            name: value
            for name, value in state.items()
            if "vat_objective.accept_head." in name
        }

    def load_state_dict(self, state_dict):
        expected = self.state_dict()
        if expected.keys() != state_dict.keys():
            raise ValueError("VAT acceptance-head checkpoint parameters do not match")
        set_model_state_dict(
            self.model,
            state_dict,
            options=StateDictOptions(full_state_dict=False, strict=False),
        )
