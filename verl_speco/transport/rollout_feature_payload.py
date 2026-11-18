# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Typed feature payloads for vLLM's untyped utility/collective RPC replies."""

import json
from collections.abc import Mapping

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from verl_speco.transport.drafter_sample_protocol import (
    _decode_metadata_tree,
    _encode_metadata_tree,
)

_VERSION_KEY = "speco_feature_payload_version"


def _validate(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("Rollout feature payload must be a mapping")
    for name in ("input_ids", "positions", "hidden_states"):
        if not torch.is_tensor(payload.get(name)):
            raise ValueError(
                f"Rollout field {name} is not a Tensor; worker and receiver must use the typed feature codec"
            )
    head = payload.get("target_head_hidden_states")
    if head is not None and not torch.is_tensor(head):
        raise ValueError("Rollout pre-HC field is not a Tensor")
    return dict(payload)


def encode_feature_payload(payload):
    if payload is None:
        return None
    payload = _validate(payload)
    fields = {}
    manifest = _encode_metadata_tree(payload, fields, path="rollout")
    # Reuse SpeCo's tensor-tree manifest. Clone fields independently because
    # safetensors rejects shared storage; preserve all original dtypes.
    binary = save({name: value.clone() for name, value in fields.items()})
    return {_VERSION_KEY: 1, "manifest": json.dumps(manifest), "tensors": binary}


def decode_feature_payload(value):
    if not isinstance(value, Mapping):
        raise TypeError("Rollout feature response must be a mapping")
    if _VERSION_KEY not in value:
        # Preserve direct/in-process Tensor replies, never infer types from the
        # untyped vLLM [dtype, shape, buffer] wire descriptor.
        return _validate(value)
    if value[_VERSION_KEY] != 1:
        raise ValueError("Unsupported rollout feature payload version")
    binary = value.get("tensors")
    if not isinstance(binary, (bytes, bytearray, memoryview)):
        raise TypeError("Rollout feature tensors must use a binary payload")
    try:
        fields = load(bytes(binary))
    except SafetensorError as error:
        raise ValueError("Invalid rollout feature tensor payload") from error
    manifest = json.loads(value["manifest"])
    return _validate(_decode_metadata_tree(manifest, fields, path="rollout"))
