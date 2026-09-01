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
"""Algorithm-neutral TransferQueue codec for ``DraftFeatureSample``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from verl_speco.trainer.feature_store import DraftFeatureSample


PROTOCOL_SCHEMA_VERSION = 2
DRAFTER_TQ_PARTITION = "speco_drafter_features"
_MANIFEST_FIELD = "sample__manifest_json"
_REQUIRED_SAMPLE_FIELDS = ("input_ids", "loss_mask", "hidden_states")
_OPTIONAL_TENSOR_FIELDS = (
    "last_hidden_states",
    "target",
    "target_logprobs",
    "position_ids",
)


@dataclass(frozen=True)
class SampleMetadata:
    """Small control-plane envelope; training metadata belongs to the sample."""

    schema_version: int
    run_id: str
    sample_id: str
    sequence_no: int

    def validate(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported drafter protocol schema_version={self.schema_version}; "
                f"expected {PROTOCOL_SCHEMA_VERSION}"
            )
        if not self.run_id:
            raise ValueError("SampleMetadata.run_id must not be empty")
        if not self.sample_id:
            raise ValueError("SampleMetadata.sample_id must not be empty")
        if self.sequence_no < 0:
            raise ValueError("SampleMetadata.sequence_no must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SampleMetadata":
        try:
            meta = cls(
                schema_version=int(payload["schema_version"]),
                run_id=str(payload["run_id"]),
                sample_id=str(payload["sample_id"]),
                sequence_no=int(payload["sequence_no"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"ready tag missing required field {exc.args[0]!r}"
            ) from exc
        meta.validate()
        return meta


@dataclass(frozen=True)
class ExpectedFeatureConfig:
    """Consumer-side transport contract."""

    run_id: str
    schema_version: int = PROTOCOL_SCHEMA_VERSION


def make_sample_key(meta: SampleMetadata) -> str:
    meta.validate()
    return (
        f"drafter:v{meta.schema_version}:{meta.run_id}:"
        f"{meta.sequence_no:012d}:{meta.sample_id}"
    )


def make_ready_tag(meta: SampleMetadata) -> dict[str, Any]:
    meta.validate()
    return {"record_type": "sample", "status": "ready", **meta.to_dict()}


def parse_ready_tag(
    tag: Mapping[str, Any],
    *,
    run_id: str | None = None,
    schema_version: int = PROTOCOL_SCHEMA_VERSION,
) -> SampleMetadata | None:
    """Return one valid ready envelope, or ``None`` when it is not consumable."""

    if tag.get("record_type") != "sample" or tag.get("status") != "ready":
        return None
    try:
        meta = SampleMetadata.from_dict(tag)
    except (TypeError, ValueError):
        return None
    if meta.schema_version != int(schema_version):
        return None
    if run_id is not None and meta.run_id != str(run_id):
        return None
    return meta


def is_ready_sample_tag(
    tag: Mapping[str, Any],
    *,
    run_id: str,
    schema_version: int = PROTOCOL_SCHEMA_VERSION,
) -> bool:
    return (
        parse_ready_tag(tag, run_id=run_id, schema_version=schema_version) is not None
    )


def encode_sample(
    sample: DraftFeatureSample | Mapping[str, Any], meta: SampleMetadata
) -> dict[str, torch.Tensor]:
    """Losslessly encode one normalized feature sample into TQ tensor fields."""

    meta.validate()
    normalized = (
        sample
        if isinstance(sample, DraftFeatureSample)
        else DraftFeatureSample.from_dict(dict(sample), strict=True)
    )
    normalized.validate(strict=True)
    payload = normalized.to_dict()
    fields: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {
        "draft_feature_schema_version": int(normalized.schema_version),
        "algorithm": str(normalized.algorithm),
        "present_fields": [],
    }

    for name in ("input_ids", "loss_mask", *_OPTIONAL_TENSOR_FIELDS):
        value = payload.get(name)
        if value is None:
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"DraftFeatureSample.{name} must be a torch.Tensor")
        fields[f"sample__{name}"] = _cpu_contiguous(value)
        manifest["present_fields"].append(name)

    hidden = payload["hidden_states"]
    if torch.is_tensor(hidden):
        fields["sample__hidden_states"] = _cpu_contiguous(hidden)
        manifest["hidden_states_kind"] = "tensor"
    elif isinstance(hidden, (list, tuple)):
        hidden_fields: list[str] = []
        for index, value in enumerate(hidden):
            if not torch.is_tensor(value):
                raise TypeError(
                    f"DraftFeatureSample.hidden_states[{index}] must be a torch.Tensor"
                )
            field_name = f"sample__hidden_states__{index:06d}"
            fields[field_name] = _cpu_contiguous(value)
            hidden_fields.append(field_name)
        if not hidden_fields:
            raise ValueError("DraftFeatureSample.hidden_states list must not be empty")
        manifest["hidden_states_kind"] = "list"
        manifest["hidden_states_fields"] = hidden_fields
    else:
        raise TypeError(
            "DraftFeatureSample.hidden_states must be a tensor or tensor list"
        )
    manifest["present_fields"].append("hidden_states")
    manifest["metadata"] = _encode_metadata_tree(
        payload.get("metadata", {}), fields, path="metadata"
    )
    fields[_MANIFEST_FIELD] = _json_to_tensor(manifest)
    return fields


def decode_sample(
    key: str,
    tag: Mapping[str, Any],
    fields: Mapping[str, Any],
    expected_config: ExpectedFeatureConfig | Mapping[str, Any],
) -> DraftFeatureSample:
    """Validate one queue record and restore the complete training sample."""

    expected = (
        expected_config
        if isinstance(expected_config, ExpectedFeatureConfig)
        else ExpectedFeatureConfig(**dict(expected_config))
    )
    meta = parse_ready_tag(
        tag, run_id=expected.run_id, schema_version=expected.schema_version
    )
    if meta is None:
        raise ValueError(f"TQ sample {key!r} has an invalid or unexpected ready tag")
    expected_key = make_sample_key(meta)
    if key != expected_key:
        raise ValueError(
            f"TQ sample key mismatch: got {key!r}, expected {expected_key!r}"
        )
    if _MANIFEST_FIELD not in fields:
        raise ValueError(f"TQ sample {key!r} missing field {_MANIFEST_FIELD!r}")
    manifest = _tensor_to_json(fields[_MANIFEST_FIELD], name=_MANIFEST_FIELD)
    present = manifest.get("present_fields")
    if not isinstance(present, list):
        raise ValueError("sample manifest present_fields must be a list")
    missing = [name for name in _REQUIRED_SAMPLE_FIELDS if name not in present]
    if missing:
        raise ValueError(f"TQ sample {key!r} missing required sample fields: {missing}")

    try:
        sample_schema_version = int(manifest["draft_feature_schema_version"])
        algorithm = str(manifest["algorithm"])
    except KeyError as exc:
        raise ValueError(f"sample manifest missing field {exc.args[0]!r}") from exc
    payload: dict[str, Any] = {
        "schema_version": sample_schema_version,
        "algorithm": algorithm,
        "metadata": _decode_metadata_tree(
            manifest.get("metadata", {}), fields, path="metadata"
        ),
    }
    for name in ("input_ids", "loss_mask", *_OPTIONAL_TENSOR_FIELDS):
        if name in present:
            payload[name] = _require_tensor(fields, f"sample__{name}")
    payload["hidden_states"] = _decode_hidden_states(fields, manifest)
    return DraftFeatureSample.from_dict(payload, strict=True)


def make_eos_record(
    run_id: str, total_samples: int
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    if not run_id:
        raise ValueError("run_id must not be empty")
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    key = f"control:v{PROTOCOL_SCHEMA_VERSION}:{run_id}:eos"
    fields = {"marker": torch.tensor([1], dtype=torch.uint8)}
    tag = {
        "record_type": "control",
        "status": "eos",
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "run_id": run_id,
        "total_samples": int(total_samples),
    }
    return key, fields, tag


def _encode_metadata_tree(
    value: Any, fields: dict[str, torch.Tensor], *, path: str
) -> Any:
    if torch.is_tensor(value):
        field_name = f"sample__metadata_tensor__{len(fields):06d}"
        fields[field_name] = _cpu_contiguous(value)
        return {"__tq_tensor_ref__": field_name}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"DraftFeatureSample metadata key at {path} must be str"
                )
            encoded[key] = _encode_metadata_tree(item, fields, path=f"{path}.{key}")
        return {"__tq_mapping__": encoded}
    if isinstance(value, (list, tuple)):
        items = [
            _encode_metadata_tree(item, fields, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return {
            "__tq_sequence__": "tuple" if isinstance(value, tuple) else "list",
            "items": items,
        }
    raise TypeError(
        f"Unsupported DraftFeatureSample metadata value at {path}: {type(value).__name__}"
    )


def _decode_metadata_tree(value: Any, fields: Mapping[str, Any], *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid metadata manifest node at {path}")
    if "__tq_tensor_ref__" in value:
        return _require_tensor(fields, str(value["__tq_tensor_ref__"]))
    if "__tq_mapping__" in value:
        mapping = value["__tq_mapping__"]
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Invalid metadata mapping node at {path}")
        return {
            str(key): _decode_metadata_tree(item, fields, path=f"{path}.{key}")
            for key, item in mapping.items()
        }
    if "__tq_sequence__" in value:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid metadata sequence node at {path}")
        decoded = [
            _decode_metadata_tree(item, fields, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        ]
        if value["__tq_sequence__"] == "tuple":
            return tuple(decoded)
        if value["__tq_sequence__"] == "list":
            return decoded
    raise ValueError(f"Unknown metadata manifest node at {path}")


def _decode_hidden_states(
    fields: Mapping[str, Any], manifest: Mapping[str, Any]
) -> torch.Tensor | list[torch.Tensor]:
    kind = manifest.get("hidden_states_kind")
    if kind == "tensor":
        return _require_tensor(fields, "sample__hidden_states")
    if kind == "list":
        names = manifest.get("hidden_states_fields")
        if not isinstance(names, list) or not names:
            raise ValueError("sample manifest hidden_states_fields must be non-empty")
        return [_require_tensor(fields, str(name)) for name in names]
    raise ValueError(f"Unsupported hidden_states_kind={kind!r}")


def _json_to_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return torch.tensor(list(raw), dtype=torch.uint8)


def _tensor_to_json(value: Any, *, name: str) -> dict[str, Any]:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach().cpu().to(torch.uint8).reshape(-1)
    try:
        decoded = json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return decoded


def _require_tensor(fields: Mapping[str, Any], name: str) -> torch.Tensor:
    value = fields.get(name)
    if value is None or not torch.is_tensor(value):
        raise TypeError(f"TQ field {name!r} must be a torch.Tensor")
    return value.detach().cpu().contiguous()


def _cpu_contiguous(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected torch.Tensor, got {type(value)!r}")
    return value.detach().cpu().contiguous()


__all__ = [
    "DRAFTER_TQ_PARTITION",
    "PROTOCOL_SCHEMA_VERSION",
    "ExpectedFeatureConfig",
    "SampleMetadata",
    "decode_sample",
    "encode_sample",
    "is_ready_sample_tag",
    "make_eos_record",
    "make_ready_tag",
    "make_sample_key",
    "parse_ready_tag",
]
