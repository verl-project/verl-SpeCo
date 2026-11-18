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
"""DeepSeek V4 feature semantics carried by the existing SpeCo sample codec."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from verl_speco.trainer.feature_store import DraftFeatureSample

LAYOUT = "deepseek_v4_0731_dspark_pre_hc_v1"
EXTENSION = "ds4_draft"


@dataclass(frozen=True)
class DeepSeekV4Contract:
    hidden_size: int
    hc_mult: int
    vocab_size: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    config_fingerprint: str
    target_revision: str = "0731"

    @classmethod
    def from_model(cls, path: str | Path) -> DeepSeekV4Contract:
        """Read architecture only. Revision 0731 is a caller assertion, not a weight digest."""
        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("model_type") != "deepseek_v4":
            raise ValueError(
                "AIRDrafter DS4 integration requires model_type=deepseek_v4"
            )
        layers = tuple(int(i) for i in config.get("dspark_target_layer_ids", ()))
        if not layers or tuple(sorted(set(layers))) != layers:
            raise ValueError(
                "dspark_target_layer_ids must be nonempty, sorted and unique"
            )
        if min(layers) < 0 or max(layers) >= int(config["num_hidden_layers"]):
            raise ValueError("DSpark capture layers are outside the target model")
        values = [
            int(config[key])
            for key in ("hidden_size", "hc_mult", "vocab_size", "dspark_block_size")
        ]
        if min(values) <= 0:
            raise ValueError("DS4 dimensions must be positive")
        if config.get("dspark_bonus_anchor", False):
            raise ValueError("This integration requires dspark_bonus_anchor=false")
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        return cls(values[0], values[1], values[2], values[3], layers, fingerprint)

    def identity(self) -> dict:
        value = asdict(self)
        value["target_layer_ids"] = list(self.target_layer_ids)
        value["layout"] = LAYOUT
        value["target_offset"] = 2
        return value

    def validate(self, sample: DraftFeatureSample) -> None:
        sample.validate(strict=True)
        if sample.algorithm.upper() != "DSPARK":
            raise ValueError("DS4 training only accepts DSPARK samples")
        extension = sample.metadata.get(EXTENSION, {})
        if extension.get("contract") != self.identity():
            raise ValueError(
                "DS4 sample contract differs from the selected model/revision"
            )
        rows = sample.input_ids.numel()
        if rows < 3:
            raise ValueError("DS4 DSpark needs at least three token rows")
        if sample.input_ids.dtype != torch.int64 or sample.input_ids.ndim != 1:
            raise ValueError("input_ids must be a one-dimensional int64 tensor")
        if bool(((sample.input_ids < 0) | (sample.input_ids >= self.vocab_size)).any()):
            raise ValueError("Token ID is outside the model vocabulary")
        mask = sample.loss_mask
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("loss_mask must be binary, with 1 indicating supervision")
        positions = sample.position_ids
        if positions is None or positions.dtype != torch.int64 or positions.ndim != 1:
            raise ValueError(
                "DS4 requires explicit one-dimensional int64 absolute positions"
            )
        if int(positions[0]) < 0 or not torch.equal(positions[1:], positions[:-1] + 1):
            raise ValueError("DS4 positions must be nonnegative and contiguous")
        for name, tensor, width in (
            (
                "selected layers",
                sample.hidden_states,
                self.hidden_size * len(self.target_layer_ids),
            ),
            (
                "pre-HC teacher",
                extension.get("target_head_hidden_states"),
                self.hidden_size * self.hc_mult,
            ),
        ):
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != (rows, width):
                raise ValueError(f"Invalid {name} shape; expected {(rows, width)}")
            if not torch.is_floating_point(tensor) or not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{name} must contain finite floating-point states")


def from_engine_tensors(
    tensors: dict, loss_mask, contract: DeepSeekV4Contract, *, sample_id: str
) -> DraftFeatureSample:
    """Keep pre-HC explicit in a namespaced extension supported by SpeCo today.

    Metadata tensor trees already round-trip through TorchShardFeatureStore and
    the TQ codec. This avoids changing any native sample or protocol code.
    Masks use SpeCo semantics throughout: 1 is supervised, 0 is ignored.
    """
    ids = tensors["input_ids"].detach().cpu()
    if ids.ndim == 2 and ids.size(0) == 1:
        ids = ids.squeeze(0)
    positions = tensors["positions"].detach().cpu()
    if positions.ndim == 2 and positions.size(0) == 1:
        positions = positions.squeeze(0)
    loss_mask = torch.as_tensor(loss_mask).detach().cpu()
    if tuple(loss_mask.shape) != tuple(ids.shape) or not bool(
        ((loss_mask == 0) | (loss_mask == 1)).all()
    ):
        raise ValueError(
            "SpeCo loss_mask must be binary and match all input token rows"
        )
    values = {}
    for key in ("hidden_states", "target_head_hidden_states"):
        value = tensors[key].detach().cpu()
        if value.ndim == 3 and value.size(0) == 1:
            value = value.squeeze(0)
        values[key] = value.contiguous()
    sample = DraftFeatureSample(
        input_ids=ids.contiguous(),
        loss_mask=loss_mask.float().contiguous(),
        hidden_states=values["hidden_states"],
        algorithm="DSPARK",
        position_ids=positions.contiguous(),
        metadata={
            "sample_id": sample_id,
            "hidden_states_layout": LAYOUT,
            "target_revision": "0731",
            EXTENSION: {
                "contract": contract.identity(),
                "target_head_hidden_states": values["target_head_hidden_states"],
            },
        },
    )
    contract.validate(sample)
    return sample


def to_engine_tensors(sample: DraftFeatureSample, contract: DeepSeekV4Contract) -> dict:
    """Expose training tensors without changing SpeCo mask semantics."""
    contract.validate(sample)
    return {
        "input_ids": sample.input_ids,
        "hidden_states": sample.hidden_states,
        "positions": sample.position_ids,
        "loss_mask": sample.loss_mask,
        "target_head_hidden_states": sample.metadata[EXTENSION][
            "target_head_hidden_states"
        ],
    }
