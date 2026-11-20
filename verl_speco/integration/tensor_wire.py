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
"""Lossless CPU tensor transport through Ray's NumPy object-store path."""

import math
import warnings

import numpy as np
import torch

_MARKER = "_speco_cpu_tensor_wire_v1"
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    )
}


def encode_cpu_tensors(tensors):
    """Return None before mutation for unsupported values; never quantize.

    Caller owns an independent, immutable CPU snapshot. NumPy views keep that
    storage alive until ray.put/serialization finishes. Consumers only read the
    resulting object-store-backed tensors during the awaited rollout update.
    """
    if (
        not isinstance(tensors, dict)
        or not tensors
        or not all(
            isinstance(name, str)
            and type(tensor) is torch.Tensor
            and tensor.device.type == "cpu"
            and tensor.layout == torch.strided
            and tensor.is_contiguous()
            and not tensor.is_conj()
            and not tensor.is_neg()
            and str(tensor.dtype).removeprefix("torch.") in _DTYPES
            for name, tensor in tensors.items()
        )
    ):
        return None
    return {
        _MARKER: {
            name: {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "data": tensor.detach().reshape(-1).view(torch.uint8).numpy(),
            }
            for name, tensor in tensors.items()
        }
    }


def decode_cpu_tensors(payload):
    if not isinstance(payload, dict) or _MARKER not in payload:
        return payload
    if (
        set(payload) != {_MARKER}
        or not isinstance(payload[_MARKER], dict)
        or not payload[_MARKER]
    ):
        raise ValueError("Invalid CPU tensor wire envelope")
    result = {}
    for name, entry in payload[_MARKER].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise TypeError("Invalid CPU tensor wire entry")
        shape, dtype, data = entry.get("shape"), entry.get("dtype"), entry.get("data")
        if (
            not isinstance(shape, (list, tuple))
            or not all(type(dim) is int and dim >= 0 for dim in shape)
            or not isinstance(dtype, str)
            or dtype not in _DTYPES
            or not isinstance(data, np.ndarray)
            or data.dtype != np.uint8
            or data.ndim != 1
            or not data.flags.c_contiguous
            or data.nbytes
            != math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
        ):
            raise ValueError(f"Invalid CPU tensor wire layout: {name}")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            result[name] = torch.from_numpy(data).view(_DTYPES[dtype]).reshape(shape)
    return result


def tensor_payload_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors.values())
