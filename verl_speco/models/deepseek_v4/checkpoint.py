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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/ds4_checkpoint_format.py.
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

_SUPPORTED_FORMATS = {"auto", "w8a8", "bf16"}
_FLOAT_DTYPES = {"BF16", "F32"}


def _is_mtp_tensor(name: str) -> bool:
    return name.startswith("mtp.")


def _is_weight_tensor(name: str, shape: tuple[int, ...]) -> bool:
    return len(shape) >= 2 and (
        name.endswith((".weight", "_weight")) or ".weight." in name
    )


def _is_scale_tensor(name: str) -> bool:
    return any(
        token in name
        for token in (
            "weight_scale",
            "_weight_scale",
            "fused_w1_scale",
            "fused_w2_scale",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


@dataclass(frozen=True)
class DS4CheckpointFormatReport:
    checkpoint_path: str
    requested_format: str
    resolved_format: str
    safetensors_files: int
    mtp_tensors: int
    mtp_weight_tensors: int
    dtype_counts: dict[str, int]
    int8_weight_tensors: tuple[str, ...]
    unsupported_weight_tensors: tuple[str, ...]
    scale_tensors: int
    quant_description_present: bool
    config_quantization_present: bool
    runtime_quantization: str | None
    requires_dequantization: bool

    @property
    def is_mixed_runtime_checkpoint(self) -> bool:
        return self.resolved_format == "bf16" and (
            self.quant_description_present or self.config_quantization_present
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_ds4_checkpoint_format(
    checkpoint_path: str | Path,
    requested_format: str = "auto",
) -> DS4CheckpointFormatReport:
    root = Path(checkpoint_path)
    requested = str(requested_format).strip().lower()
    if requested not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"checkpoint_format must be one of {sorted(_SUPPORTED_FORMATS)}, "
            f"got {requested_format!r}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint path must be a directory: {root}")

    safetensors_files = sorted(root.rglob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found under {root}")

    seen_keys: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    mtp_tensors = 0
    mtp_weight_tensors = 0
    mtp_weight_names: set[str] = set()
    int8_weight_tensors: list[str] = []
    unsupported_weight_tensors: list[str] = []
    scale_tensors = 0

    for tensor_path in safetensors_files:
        with safe_open(tensor_path, framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 -- safetensors/runtime key API is explicit
                if name in seen_keys:
                    raise ValueError(
                        f"Duplicate tensor key {name!r} across safetensors files"
                    )
                seen_keys.add(name)
                if not _is_mtp_tensor(name):
                    continue
                tensor_slice = source.get_slice(name)
                dtype = str(tensor_slice.get_dtype())
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                mtp_tensors += 1
                dtype_counts[dtype] += 1
                if _is_scale_tensor(name):
                    scale_tensors += 1
                if not _is_weight_tensor(name, shape):
                    continue
                mtp_weight_tensors += 1
                mtp_weight_names.add(name)
                if dtype == "I8":
                    int8_weight_tensors.append(name)
                elif dtype not in _FLOAT_DTYPES:
                    unsupported_weight_tensors.append(f"{name}:{dtype}")

    if mtp_tensors == 0:
        raise ValueError(f"Checkpoint contains no mtp.* tensors: {root}")
    if mtp_weight_tensors == 0:
        raise ValueError(f"Checkpoint contains no MTP matrix weights: {root}")
    if unsupported_weight_tensors:
        raise ValueError(
            "Unsupported MTP matrix weight dtypes; BF16 checkpoints may contain "
            "BF16/FP32 weights and W8A8 checkpoints may additionally contain INT8: "
            + ", ".join(unsupported_weight_tensors)
        )

    inferred = "w8a8" if int8_weight_tensors else "bf16"
    resolved = inferred if requested == "auto" else requested
    if resolved == "w8a8" and not int8_weight_tensors:
        raise ValueError(
            "checkpoint_format='w8a8' was requested but no INT8 MTP matrix "
            "weights were found"
        )
    if resolved == "bf16" and int8_weight_tensors:
        preview = ", ".join(int8_weight_tensors[:8])
        raise ValueError(
            "checkpoint_format='bf16' was requested but INT8 MTP matrix "
            f"weights were found: {preview}"
        )
    if resolved == "w8a8" and scale_tensors == 0:
        raise ValueError(
            "INT8 MTP weights were found without any MTP weight-scale tensors"
        )

    quant_description_path = root / "quant_model_description.json"
    quant_description = _read_json(quant_description_path)
    model_config = _read_json(root / "config.json")
    config_quantization = model_config.get("quantization_config")
    config_quantization_present = bool(config_quantization)
    quant_description_present = bool(quant_description)
    runtime_quantization = (
        "ascend"
        if resolved == "w8a8"
        or quant_description_present
        or config_quantization_present
        else None
    )

    if (
        resolved == "bf16"
        and config_quantization_present
        and not quant_description_present
    ):
        raise ValueError(
            "BF16 MTP weights were found with config.json quantization metadata "
            "but without quant_model_description.json. Remove stale global "
            "quantization metadata for a pure BF16 checkpoint or provide the "
            "per-tensor description for a mixed checkpoint."
        )
    if resolved == "bf16" and quant_description_present:
        missing_float_descriptions = [
            name for name in mtp_weight_names if name not in quant_description
        ]
        non_float_descriptions = [
            name
            for name in mtp_weight_names
            if name in quant_description and quant_description[name] != "FLOAT"
        ]
        if missing_float_descriptions:
            raise ValueError(
                "BF16 MTP matrix weights are missing from "
                "quant_model_description.json: "
                + ", ".join(sorted(missing_float_descriptions))
            )
        if non_float_descriptions:
            raise ValueError(
                "BF16 MTP tensors are not marked FLOAT in "
                "quant_model_description.json: "
                + ", ".join(sorted(non_float_descriptions))
            )

    return DS4CheckpointFormatReport(
        checkpoint_path=str(root.resolve()),
        requested_format=requested,
        resolved_format=resolved,
        safetensors_files=len(safetensors_files),
        mtp_tensors=mtp_tensors,
        mtp_weight_tensors=mtp_weight_tensors,
        dtype_counts=dict(sorted(dtype_counts.items())),
        int8_weight_tensors=tuple(sorted(int8_weight_tensors)),
        unsupported_weight_tensors=tuple(sorted(unsupported_weight_tensors)),
        scale_tensors=scale_tensors,
        quant_description_present=quant_description_present,
        config_quantization_present=config_quantization_present,
        runtime_quantization=runtime_quantization,
        requires_dequantization=resolved == "w8a8",
    )
