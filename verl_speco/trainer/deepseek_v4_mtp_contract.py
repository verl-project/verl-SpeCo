# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""MTP consumes target pre-HC residuals, never DSpark auxiliary features."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from verl_speco.trainer.feature_store import DraftFeatureSample

LAYOUT = "deepseek_v4_mtp_pre_hc_v1"
EXTENSION = "ds4_mtp"


@dataclass(frozen=True)
class DeepSeekV4MTPContract:
    hidden_size: int
    hc_mult: int
    vocab_size: int
    num_mtp_layers: int
    config_fingerprint: str

    # The common capture builder uses this only as the speculation token count.
    block_size = 1

    @classmethod
    def from_model(cls, path):
        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("model_type") != "deepseek_v4":
            raise ValueError("MTP requires model_type=deepseek_v4")
        if config.get("dspark_block_size") or config.get("dspark_target_layer_ids"):
            raise ValueError(
                "DSpark checkpoints cannot initialize the MTP architecture"
            )
        count = int(config.get("num_nextn_predict_layers", 0))
        # Existing canonical native-weight export owns layer zero. Do not
        # silently drop additional heads until their mapping is supported.
        if count != 1:
            raise ValueError("This MTP adapter requires exactly one native MTP layer")
        dimensions = [int(config[k]) for k in ("hidden_size", "hc_mult", "vocab_size")]
        if min(dimensions) <= 0:
            raise ValueError("MTP dimensions must be positive")
        return cls(
            *dimensions,
            count,
            hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        )

    def identity(self):
        return {
            **asdict(self),
            "layout": LAYOUT,
            "token_offset": 1,
            "position_offset": 0,
        }

    def validate(self, sample):
        sample.validate(strict=True)
        if (
            sample.algorithm.upper() != "MTP"
            or sample.metadata.get(EXTENSION, {}).get("contract") != self.identity()
        ):
            raise ValueError("MTP sample algorithm/contract does not match this target")
        ids, positions, mask, hidden = (
            sample.input_ids,
            sample.position_ids,
            sample.loss_mask,
            sample.hidden_states,
        )
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.numel() < 2:
            raise ValueError("MTP requires at least two int64 token rows")
        if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise ValueError("MTP token ID is outside the vocabulary")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("MTP loss_mask must use SpeCo 1=supervised, 0=ignored")
        if positions is None or positions.dtype != torch.int64 or positions.ndim != 1:
            raise ValueError("MTP requires explicit int64 absolute positions")
        if int(positions[0]) < 0 or not torch.equal(positions[1:], positions[:-1] + 1):
            raise ValueError(
                "MTP absolute positions must be contiguous and nonnegative"
            )
        expected = (ids.numel(), self.hidden_size * self.hc_mult)
        if not torch.is_tensor(hidden) or tuple(hidden.shape) != expected:
            raise ValueError(f"MTP requires complete pre-HC states of shape {expected}")
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError("MTP pre-HC states must be finite floating-point values")


def from_engine_tensors(tensors, loss_mask, contract, *, sample_id):
    ids = tensors["input_ids"].detach().cpu().reshape(-1)
    positions = tensors["positions"].detach().cpu().reshape(-1)
    hidden = (
        tensors["hidden_states"].detach().cpu().reshape(ids.numel(), -1).contiguous()
    )
    sample = DraftFeatureSample(
        input_ids=ids,
        loss_mask=torch.as_tensor(loss_mask).detach().cpu().float().reshape(-1),
        hidden_states=hidden,
        algorithm="MTP",
        position_ids=positions,
        metadata={
            "sample_id": sample_id,
            "hidden_states_layout": LAYOUT,
            EXTENSION: {"contract": contract.identity()},
        },
    )
    contract.validate(sample)
    return sample
