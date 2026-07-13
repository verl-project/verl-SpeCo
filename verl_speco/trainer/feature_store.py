"""Feature store primitives for standalone SPECO draft training."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import torch


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.jsonl"
METADATA_NAME = "metadata.json"


@dataclass
class DraftFeatureSample:
    """A normalized draft-training sample.

    The tensor fields intentionally mirror the format used by TorchSpec-style
    independent draft training: ids, masks, hidden states and either
    last-hidden supervision or sparse target logprobs.
    """

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    hidden_states: torch.Tensor | list[torch.Tensor]
    algorithm: str = "EAGLE3"
    schema_version: int = SCHEMA_VERSION
    last_hidden_states: torch.Tensor | None = None
    target: torch.Tensor | None = None
    target_logprobs: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, strict: bool = True) -> "DraftFeatureSample":
        sample = cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            algorithm=str(payload.get("algorithm", payload.get("metadata", {}).get("algorithm", "EAGLE3"))),
            input_ids=payload["input_ids"],
            loss_mask=payload["loss_mask"],
            hidden_states=payload["hidden_states"],
            last_hidden_states=payload.get("last_hidden_states", payload.get("target")),
            target=payload.get("target"),
            target_logprobs=payload.get("target_logprobs"),
            position_ids=payload.get("position_ids"),
            metadata=dict(payload.get("metadata") or {}),
        )
        sample.validate(strict=strict)
        return sample

    def validate(self, *, strict: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION and strict:
            raise ValueError(f"Unsupported DraftFeatureSample schema_version={self.schema_version}")
        if not torch.is_tensor(self.input_ids):
            raise TypeError("DraftFeatureSample.input_ids must be a torch.Tensor")
        if not torch.is_tensor(self.loss_mask):
            raise TypeError("DraftFeatureSample.loss_mask must be a torch.Tensor")
        if not (torch.is_tensor(self.hidden_states) or isinstance(self.hidden_states, (list, tuple))):
            raise TypeError("DraftFeatureSample.hidden_states must be a tensor or tensor list")
        if self.input_ids.dim() > 1:
            self.input_ids = self.input_ids.reshape(-1)
        if self.loss_mask.dim() > 1:
            self.loss_mask = self.loss_mask.reshape(-1)
        if torch.is_tensor(self.position_ids) and self.position_ids.dim() > 1:
            self.position_ids = self.position_ids.reshape(-1)
        if torch.is_tensor(self.target_logprobs):
            while self.target_logprobs.dim() > 3 and self.target_logprobs.size(0) == 1:
                self.target_logprobs = self.target_logprobs.squeeze(0)
        if self.input_ids.size(0) != self.loss_mask.size(0) and strict:
            raise ValueError(
                "DraftFeatureSample input_ids/loss_mask length mismatch: "
                f"{self.input_ids.size(0)} vs {self.loss_mask.size(0)}"
            )
        if torch.is_tensor(self.position_ids) and self.position_ids.size(0) != self.input_ids.size(0) and strict:
            raise ValueError(
                "DraftFeatureSample input_ids/position_ids length mismatch: "
                f"{self.input_ids.size(0)} vs {self.position_ids.size(0)}"
            )
        if torch.is_tensor(self.hidden_states) and self.hidden_states.dim() == 3 and self.hidden_states.size(0) == 1:
            self.hidden_states = self.hidden_states.squeeze(0)
        if torch.is_tensor(self.last_hidden_states) and self.last_hidden_states.dim() == 3 and self.last_hidden_states.size(0) == 1:
            self.last_hidden_states = self.last_hidden_states.squeeze(0)
        if self.target_logprobs is not None and not torch.is_tensor(self.target_logprobs):
            raise TypeError("DraftFeatureSample.target_logprobs must be a tensor when provided")
        if torch.is_tensor(self.target_logprobs) and self.target_logprobs.dim() != 3 and strict:
            raise ValueError(
                "DraftFeatureSample.target_logprobs must have shape [rows, topk, 2], "
                f"got {tuple(self.target_logprobs.shape)}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate(strict=False)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "input_ids": self.input_ids.detach().cpu().contiguous(),
            "loss_mask": self.loss_mask.detach().cpu().float().contiguous(),
            "hidden_states": _cpu_tensor_tree(self.hidden_states),
            "metadata": dict(self.metadata),
        }
        if self.last_hidden_states is not None:
            payload["last_hidden_states"] = self.last_hidden_states.detach().cpu().contiguous()
        if self.target is not None:
            payload["target"] = self.target.detach().cpu().contiguous()
        if self.target_logprobs is not None:
            payload["target_logprobs"] = self.target_logprobs.detach().cpu().contiguous()
        if self.position_ids is not None:
            payload["position_ids"] = self.position_ids.detach().cpu().long().contiguous()
        return payload

    def to_training_item(self) -> dict[str, Any]:
        payload = self.to_dict()
        metadata = dict(payload.pop("metadata", {}) or {})
        item = {
            "input_ids": payload.pop("input_ids"),
            "loss_mask": payload.pop("loss_mask"),
            "hidden_states": payload.pop("hidden_states"),
            "step": int(metadata.get("global_step", metadata.get("step", 0)) or 0),
            "global_step": metadata.get("global_step"),
            "hidden_states_layout": metadata.get("hidden_states_layout"),
        }
        if "last_hidden_states" in payload:
            item["last_hidden_states"] = payload["last_hidden_states"]
        if "target" in payload and "last_hidden_states" not in item:
            item["last_hidden_states"] = payload["target"]
        if "target_logprobs" in payload:
            item["target_logprobs"] = payload["target_logprobs"]
        if "position_ids" in payload:
            item["position_ids"] = payload["position_ids"]
        for key, value in metadata.items():
            item.setdefault(key, value)
        _populate_verl_alignment_fields(item, metadata)
        return item


class DraftFeatureStore(Protocol):
    def write_many(self, samples: list[DraftFeatureSample | dict[str, Any]]) -> list[str]: ...

    def read(self, key: str) -> DraftFeatureSample: ...

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0) -> Iterator[str]: ...

    def get_metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _populate_verl_alignment_fields(item: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Restore online drafter alignment metadata for feature-store samples."""

    direct_fields = {
        "_verl_feature_start": "feature_start",
        "_verl_feature_end": "feature_end",
        "_verl_hidden_position_start": "hidden_position_start",
        "_verl_hidden_position_end": "hidden_position_end",
        "_verl_target_position_start": "target_logprobs_position_start",
        "_verl_target_position_end": "target_logprobs_position_end",
        "_verl_target_tensor_position_start": "target_logprobs_position_start",
        "_verl_target_tensor_position_end": "target_logprobs_position_end",
        "_verl_hidden_raw_target_position_start": "hidden_raw_target_logprobs_position_start",
        "_verl_hidden_raw_target_position_end": "hidden_raw_target_logprobs_position_end",
        "_verl_input_seq_length": "full_sequence_length",
    }
    for target_key, source_key in direct_fields.items():
        if target_key not in item and source_key in metadata:
            item[target_key] = metadata[source_key]

    if "_verl_hidden_positions" not in item and "hidden_positions" in metadata:
        item["_verl_hidden_positions"] = metadata["hidden_positions"]
    if "_verl_uses_hidden_positions" not in item:
        item["_verl_uses_hidden_positions"] = "hidden_positions" in metadata

    if "_verl_target_start" not in item and "target_logprobs" in item:
        item["_verl_target_start"] = 0
    if "_verl_target_end" not in item and torch.is_tensor(item.get("target_logprobs")):
        item["_verl_target_end"] = int(item["target_logprobs"].size(0))


class TorchShardFeatureStore:
    """Local ``torch.save`` shard store.

    This is the first-stage storage backend for P2. It favors simple,
    inspectable files over a long-running service.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_samples_per_shard: int = 1024,
        metadata: dict[str, Any] | None = None,
        strict_schema: bool = True,
        read_only: bool = False,
        shard_prefix: str = "shard",
    ):
        if path is None:
            raise ValueError("TorchShardFeatureStore requires a non-empty path")
        self.path = Path(path)
        self.max_samples_per_shard = max(int(max_samples_per_shard), 1)
        self.strict_schema = bool(strict_schema)
        self.read_only = bool(read_only)
        self.shard_prefix = str(shard_prefix or "shard")
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / MANIFEST_NAME
        self.metadata_path = self.path / METADATA_NAME
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "format": "torch_shard",
            "created_by": "verl_speco",
            "created_at": time.time(),
        }
        if metadata:
            self.metadata.update(metadata)
        self._manifest = self._load_manifest()
        self._pending: list[dict[str, Any]] = []
        self._next_shard_index = self._infer_next_shard_index()
        if not self.read_only:
            self._write_metadata()

    def write_many(self, samples: list[DraftFeatureSample | dict[str, Any]]) -> list[str]:
        if self.read_only:
            raise RuntimeError("Cannot write to a read-only TorchShardFeatureStore")
        keys: list[str] = []
        for sample_like in samples:
            sample = _coerce_sample(sample_like, strict=self.strict_schema)
            self._pending.append(sample.to_dict())
            keys.append(f"pending:{len(self._pending) - 1}")
            if len(self._pending) >= self.max_samples_per_shard:
                self.flush()
        return keys

    def flush(self) -> list[str]:
        if not self._pending:
            return []
        shard_name = f"{self.shard_prefix}_{self._next_shard_index:06d}.pt"
        shard_path = self.path / shard_name
        payload = {
            "samples": self._pending,
            "metadata": dict(self.metadata),
        }
        _atomic_torch_save(payload, shard_path)
        entry = {
            "path": shard_name,
            "num_samples": len(self._pending),
            "num_tokens": int(sum(_sample_token_count(sample) for sample in self._pending)),
            "min_global_step": _min_metadata_int(self._pending, "global_step"),
            "max_global_step": _max_metadata_int(self._pending, "global_step"),
        }
        with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        self._manifest.append(entry)
        self._pending = []
        self._next_shard_index += 1
        return [f"{shard_name}:{idx}" for idx in range(entry["num_samples"])]

    def flush_on_step(self, global_step: int | None, interval_steps: int) -> list[str]:
        """Flush pending samples on configured training-step boundaries."""
        interval_steps = int(interval_steps)
        if interval_steps <= 0 or global_step is None:
            return []
        if int(global_step) % interval_steps != 0:
            return []
        return self.flush()

    def read(self, key: str) -> DraftFeatureSample:
        shard_name, sample_index = _parse_key(key)
        shard = self._load_shard(shard_name)
        samples = shard.get("samples") or []
        sample = samples[int(sample_index)]
        return DraftFeatureSample.from_dict(sample, strict=self.strict_schema)

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0) -> Iterator[str]:
        self.flush()
        keys: list[str] = []
        for entry in self._load_manifest():
            shard_name = entry["path"]
            for idx in range(int(entry.get("num_samples", 0))):
                keys.append(f"{shard_name}:{idx}")
        if shuffle:
            random.Random(int(seed)).shuffle(keys)
        yield from keys

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open(encoding="utf-8") as metadata_file:
                    metadata.update(json.load(metadata_file))
            except (OSError, json.JSONDecodeError):
                pass
        metadata["num_shards"] = len(self._load_manifest())
        metadata["num_samples"] = sum(int(entry.get("num_samples", 0)) for entry in self._load_manifest())
        return metadata

    def close(self) -> None:
        if not self.read_only:
            self.flush()

    def _write_metadata(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=self.metadata_path.name,
            suffix=".tmp",
            dir=self.metadata_path.parent,
            delete=False,
        ) as metadata_file:
            tmp_name = metadata_file.name
            json.dump(self.metadata, metadata_file, ensure_ascii=True, indent=2, sort_keys=True)
        try:
            os.replace(tmp_name, self.metadata_path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            for line in manifest_file:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def _infer_next_shard_index(self) -> int:
        max_index = -1
        for entry in self._manifest:
            name = str(entry.get("path", ""))
            stem = Path(name).stem
            try:
                max_index = max(max_index, int(stem.split("_")[-1]))
            except (IndexError, ValueError):
                continue
        return max_index + 1

    def _load_shard(self, shard_name: str) -> dict[str, Any]:
        path = self.path / shard_name
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def build_feature_store_from_config(feature_store_cfg, *, read_only: bool = False) -> TorchShardFeatureStore:
    store_type = str(feature_store_cfg.get("type", "torch_shard") or "torch_shard")
    if store_type != "torch_shard":
        raise NotImplementedError(f"Unsupported draft feature store type: {store_type}")
    return TorchShardFeatureStore(
        feature_store_cfg.get("path"),
        max_samples_per_shard=int(feature_store_cfg.get("max_samples_per_shard", 1024)),
        strict_schema=bool(feature_store_cfg.get("strict_schema", True)),
        read_only=read_only,
    )


def _coerce_sample(sample_like: DraftFeatureSample | dict[str, Any], *, strict: bool) -> DraftFeatureSample:
    if isinstance(sample_like, DraftFeatureSample):
        sample_like.validate(strict=strict)
        return sample_like
    if isinstance(sample_like, dict):
        return DraftFeatureSample.from_dict(sample_like, strict=strict)
    raise TypeError(f"Unsupported draft feature sample type: {type(sample_like)!r}")


def _cpu_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, (list, tuple)):
        return [_cpu_tensor_tree(item) for item in value]
    return value


def _sample_token_count(sample: dict[str, Any]) -> int:
    loss_mask = sample.get("loss_mask")
    if torch.is_tensor(loss_mask):
        return int(loss_mask.detach().float().sum().item())
    input_ids = sample.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    return 0


def _metadata_int(sample: dict[str, Any], name: str) -> int | None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _min_metadata_int(samples: list[dict[str, Any]], name: str) -> int | None:
    values = [_metadata_int(sample, name) for sample in samples]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _max_metadata_int(samples: list[dict[str, Any]], name: str) -> int | None:
    values = [_metadata_int(sample, name) for sample in samples]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _parse_key(key: str) -> tuple[str, int]:
    if ":" not in key:
        raise ValueError(f"Invalid feature key {key!r}; expected 'shard.pt:index'")
    shard_name, index = key.rsplit(":", 1)
    return shard_name, int(index)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix=path.name, suffix=".tmp", dir=path.parent, delete=False) as tmp_file:
        tmp_name = tmp_file.name
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
