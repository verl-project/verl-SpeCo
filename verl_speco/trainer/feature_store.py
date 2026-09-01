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
"""Feature store primitives for standalone SPECO draft training."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, cast

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
    def from_dict(
        cls, payload: dict[str, Any], *, strict: bool = True
    ) -> "DraftFeatureSample":
        sample = cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            algorithm=str(
                payload.get(
                    "algorithm", payload.get("metadata", {}).get("algorithm", "EAGLE3")
                )
            ),
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
            raise ValueError(
                f"Unsupported DraftFeatureSample schema_version={self.schema_version}"
            )
        if not torch.is_tensor(self.input_ids):
            raise TypeError("DraftFeatureSample.input_ids must be a torch.Tensor")
        if not torch.is_tensor(self.loss_mask):
            raise TypeError("DraftFeatureSample.loss_mask must be a torch.Tensor")
        if not (
            torch.is_tensor(self.hidden_states)
            or isinstance(self.hidden_states, (list, tuple))
        ):
            raise TypeError(
                "DraftFeatureSample.hidden_states must be a tensor or tensor list"
            )
        if self.input_ids.dim() > 1:
            self.input_ids = self.input_ids.reshape(-1)
        if self.loss_mask.dim() > 1:
            self.loss_mask = self.loss_mask.reshape(-1)
        position_ids = self.position_ids
        if torch.is_tensor(position_ids):
            position_ids_tensor = cast(Any, position_ids)
            if position_ids_tensor.dim() > 1:
                self.position_ids = position_ids_tensor.reshape(-1)
        target_logprobs = self.target_logprobs
        if torch.is_tensor(target_logprobs):
            target_logprobs_tensor = cast(Any, target_logprobs)
            while (
                target_logprobs_tensor.dim() > 3 and target_logprobs_tensor.size(0) == 1
            ):
                target_logprobs_tensor = target_logprobs_tensor.squeeze(0)
            self.target_logprobs = target_logprobs_tensor
        if self.input_ids.size(0) != self.loss_mask.size(0) and strict:
            raise ValueError(
                "DraftFeatureSample input_ids/loss_mask length mismatch: "
                f"{self.input_ids.size(0)} vs {self.loss_mask.size(0)}"
            )
        if (
            torch.is_tensor(self.position_ids)
            and cast(Any, self.position_ids).size(0) != self.input_ids.size(0)
            and strict
        ):
            raise ValueError(
                "DraftFeatureSample input_ids/position_ids length mismatch: "
                f"{self.input_ids.size(0)} vs {cast(Any, self.position_ids).size(0)}"
            )
        hidden_states = self.hidden_states
        if torch.is_tensor(hidden_states):
            hidden_states_tensor = cast(Any, hidden_states)
            if hidden_states_tensor.dim() == 3 and hidden_states_tensor.size(0) == 1:
                self.hidden_states = hidden_states_tensor.squeeze(0)
        last_hidden_states = self.last_hidden_states
        if torch.is_tensor(last_hidden_states):
            last_hidden_states_tensor = cast(Any, last_hidden_states)
            if (
                last_hidden_states_tensor.dim() == 3
                and last_hidden_states_tensor.size(0) == 1
            ):
                self.last_hidden_states = last_hidden_states_tensor.squeeze(0)
        if self.target_logprobs is not None and not torch.is_tensor(
            self.target_logprobs
        ):
            raise TypeError(
                "DraftFeatureSample.target_logprobs must be a tensor when provided"
            )
        if (
            torch.is_tensor(self.target_logprobs)
            and cast(Any, self.target_logprobs).dim() != 3
            and strict
        ):
            raise ValueError(
                "DraftFeatureSample.target_logprobs must have shape [rows, topk, 2], "
                f"got {tuple(cast(Any, self.target_logprobs).shape)}"
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
            payload["last_hidden_states"] = (
                self.last_hidden_states.detach().cpu().contiguous()
            )
        if self.target is not None:
            payload["target"] = self.target.detach().cpu().contiguous()
        if self.target_logprobs is not None:
            payload["target_logprobs"] = (
                self.target_logprobs.detach().cpu().contiguous()
            )
        if self.position_ids is not None:
            payload["position_ids"] = (
                self.position_ids.detach().cpu().long().contiguous()
            )
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


@dataclass
class DraftReplaySample:
    """Compact token sample used to reconstruct target features offline."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    feature_positions: torch.Tensor
    draft_position_ids: torch.Tensor
    algorithm: str = "EAGLE3"
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, strict: bool = True
    ) -> "DraftReplaySample":
        sample = cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            algorithm=str(
                payload.get(
                    "algorithm", payload.get("metadata", {}).get("algorithm", "EAGLE3")
                )
            ),
            input_ids=payload["input_ids"],
            loss_mask=payload["loss_mask"],
            attention_mask=payload["attention_mask"],
            position_ids=payload["position_ids"],
            feature_positions=payload["feature_positions"],
            draft_position_ids=payload["draft_position_ids"],
            metadata=dict(payload.get("metadata") or {}),
        )
        sample.validate(strict=strict)
        return sample

    def validate(self, *, strict: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION and strict:
            raise ValueError(
                f"Unsupported DraftReplaySample schema_version={self.schema_version}"
            )
        tensor_fields = (
            "input_ids",
            "loss_mask",
            "attention_mask",
            "position_ids",
            "feature_positions",
            "draft_position_ids",
        )
        for name in tensor_fields:
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise TypeError(f"DraftReplaySample.{name} must be a torch.Tensor")
            if value.dim() > 1:
                setattr(self, name, value.reshape(-1))

        sequence_length = int(self.input_ids.numel())
        for name in ("loss_mask", "attention_mask", "position_ids"):
            value = cast(torch.Tensor, getattr(self, name))
            if strict and int(value.numel()) != sequence_length:
                raise ValueError(
                    f"DraftReplaySample input_ids/{name} length mismatch: "
                    f"{sequence_length} vs {int(value.numel())}"
                )
        if strict and int(self.feature_positions.numel()) <= 0:
            raise ValueError("DraftReplaySample.feature_positions must not be empty")
        if strict and int(self.draft_position_ids.numel()) != int(
            self.feature_positions.numel()
        ):
            raise ValueError(
                "DraftReplaySample feature_positions/draft_position_ids length mismatch: "
                f"{int(self.feature_positions.numel())} vs "
                f"{int(self.draft_position_ids.numel())}"
            )
        if int(self.feature_positions.numel()) > 0:
            positions = self.feature_positions.detach().cpu().long()
            if strict and (
                int(positions.min().item()) < 0
                or int(positions.max().item()) >= sequence_length
            ):
                raise ValueError(
                    "DraftReplaySample.feature_positions are outside input_ids: "
                    f"min={int(positions.min().item())} "
                    f"max={int(positions.max().item())} sequence_length={sequence_length}"
                )
            if strict and int(positions.numel()) > 1:
                deltas = positions[1:] - positions[:-1]
                if not bool((deltas == 1).all().item()):
                    raise ValueError(
                        "DraftReplaySample.feature_positions must be contiguous and increasing"
                    )

    def to_dict(self) -> dict[str, Any]:
        self.validate(strict=False)
        return {
            "schema_version": self.schema_version,
            "sample_type": "token_replay",
            "algorithm": self.algorithm,
            "input_ids": self.input_ids.detach().cpu().to(torch.int32).contiguous(),
            "loss_mask": self.loss_mask.detach().cpu().to(torch.float16).contiguous(),
            "attention_mask": self.attention_mask.detach().cpu().bool().contiguous(),
            "position_ids": self.position_ids.detach()
            .cpu()
            .to(torch.int32)
            .contiguous(),
            "feature_positions": self.feature_positions.detach()
            .cpu()
            .to(torch.int32)
            .contiguous(),
            "draft_position_ids": self.draft_position_ids.detach()
            .cpu()
            .to(torch.int32)
            .contiguous(),
            "metadata": dict(self.metadata),
        }


DraftStoredSample = DraftFeatureSample | DraftReplaySample


class DraftFeatureStore(Protocol):
    def write_many(
        self, samples: list[DraftStoredSample | dict[str, Any]]
    ) -> list[str]: ...

    def read(self, key: str) -> DraftStoredSample: ...

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0) -> Iterator[str]: ...

    def get_metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _populate_verl_alignment_fields(
    item: dict[str, Any], metadata: dict[str, Any]
) -> None:
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

    def write_many(
        self, samples: list[DraftStoredSample | dict[str, Any]]
    ) -> list[str]:
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
            "num_tokens": int(
                sum(_sample_token_count(sample) for sample in self._pending)
            ),
            "min_global_step": _min_metadata_int(self._pending, "global_step"),
            "max_global_step": _max_metadata_int(self._pending, "global_step"),
        }
        with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
            manifest_file.write(
                json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n"
            )
        self._manifest.append(entry)
        self._pending = []
        self._next_shard_index += 1
        num_samples = int(cast(Any, entry["num_samples"]))
        return [f"{shard_name}:{idx}" for idx in range(num_samples)]

    def flush_on_step(self, global_step: int | None, interval_steps: int) -> list[str]:
        """Flush pending samples on configured training-step boundaries."""
        interval_steps = int(interval_steps)
        if interval_steps <= 0 or global_step is None:
            return []
        if int(global_step) % interval_steps != 0:
            return []
        return self.flush()

    def read(self, key: str) -> DraftStoredSample:
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
        metadata["num_samples"] = sum(
            int(entry.get("num_samples", 0)) for entry in self._load_manifest()
        )
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
            json.dump(
                self.metadata,
                metadata_file,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
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


class TokenReplayFeatureStore(TorchShardFeatureStore):
    """Compact shard store containing tokens and replay alignment metadata."""

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
        replay_metadata = dict(metadata or {})
        replay_metadata["format"] = "token_replay"
        super().__init__(
            path,
            max_samples_per_shard=max_samples_per_shard,
            metadata=replay_metadata,
            strict_schema=strict_schema,
            read_only=read_only,
            shard_prefix=shard_prefix,
        )

    def write_many(
        self, samples: list[DraftStoredSample | dict[str, Any]]
    ) -> list[str]:
        if self.read_only:
            raise RuntimeError("Cannot write to a read-only TokenReplayFeatureStore")
        keys: list[str] = []
        for sample_like in samples:
            sample = _coerce_replay_sample(sample_like, strict=self.strict_schema)
            self._pending.append(sample.to_dict())
            keys.append(f"pending:{len(self._pending) - 1}")
            if len(self._pending) >= self.max_samples_per_shard:
                self.flush()
        return keys

    def read(self, key: str) -> DraftReplaySample:
        shard_name, sample_index = _parse_key(key)
        shard = self._load_shard(shard_name)
        samples = shard.get("samples") or []
        sample = samples[int(sample_index)]
        return DraftReplaySample.from_dict(sample, strict=self.strict_schema)


class JsonlTokenReplayFeatureStore:
    """Read token replay JSONL rows as compact replay samples.

    Supported row formats:
    - ``{"input_ids": [...], "loss_mask": [...]}``
    - ``{"conversations": [{"from": "human", "value": "..."}, ...]}``
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
        max_seq_len: int = 512,
        window_mode: str = "loss",
        tokenizer_path: str | os.PathLike[str] | None = None,
        trust_remote_code: bool = False,
        train_on: str = "last_assistant",
    ):
        if path is None:
            raise ValueError("JsonlTokenReplayFeatureStore requires a non-empty path")
        if not read_only:
            raise RuntimeError("JsonlTokenReplayFeatureStore is read-only")
        self.path = Path(path)
        self.max_samples_per_shard = max(int(max_samples_per_shard), 1)
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "format": "jsonl_token_replay",
            "created_by": "verl_speco",
            "created_at": time.time(),
        }
        if metadata:
            self.metadata.update(metadata)
        self.strict_schema = bool(strict_schema)
        self.read_only = bool(read_only)
        self.shard_prefix = str(shard_prefix or "shard")
        self.max_seq_len = int(max_seq_len or 0)
        self.window_mode = str(window_mode or "loss").strip().lower()
        self.tokenizer_path = os.fspath(tokenizer_path) if tokenizer_path else None
        self.trust_remote_code = bool(trust_remote_code)
        self.train_on = str(train_on or "last_assistant").strip().lower()
        self._tokenizer: Any | None = None
        if self.window_mode not in {"loss", "front", "full"}:
            raise ValueError(
                "feature_store.window_mode for jsonl_token_replay must be "
                "'loss', 'front' or 'full'"
            )
        if self.train_on != "last_assistant":
            raise ValueError(
                "feature_store.train_on for jsonl_token_replay currently supports "
                "only 'last_assistant'"
            )
        self._files = self._resolve_jsonl_files()
        self._line_offsets: dict[str, list[int]] = {}
        self._keys = self._build_keys()

    def write_many(
        self, samples: list[DraftStoredSample | dict[str, Any]]
    ) -> list[str]:
        raise RuntimeError("JsonlTokenReplayFeatureStore is read-only")

    def read(self, key: str) -> DraftReplaySample:
        file_name, row_index = _parse_key(key)
        file_path = self.path / file_name if self.path.is_dir() else self.path
        offsets = self._line_offsets.get(file_name)
        if offsets is None:
            self._keys = self._build_keys()
            offsets = self._line_offsets.get(file_name)
        if offsets is None or int(row_index) >= len(offsets):
            raise IndexError(f"JSONL row {row_index} not found in {file_path}")
        payload = _load_jsonl_offset(file_path, offsets[int(row_index)])
        return self._row_to_replay_sample(payload, file_name, int(row_index))

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0) -> Iterator[str]:
        keys = list(self._keys)
        if shuffle:
            random.Random(int(seed)).shuffle(keys)
        yield from keys

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "num_files": len(self._files),
                "num_samples": len(self._keys),
                "max_seq_len": self.max_seq_len,
                "window_mode": self.window_mode,
                "train_on": self.train_on,
                "tokenizer_path": self.tokenizer_path,
            }
        )
        return metadata

    def close(self) -> None:
        return

    def _resolve_jsonl_files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if self.path.is_dir():
            files = sorted(self.path.glob("*.jsonl"))
            if files:
                return files
        raise FileNotFoundError(f"No JSONL file found at {self.path}")

    def _build_keys(self) -> list[str]:
        keys: list[str] = []
        self._line_offsets = {}
        for file_path in self._files:
            file_key = (
                file_path.relative_to(self.path).as_posix()
                if self.path.is_dir()
                else file_path.name
            )
            offsets: list[int] = []
            with file_path.open("rb") as jsonl_file:
                while True:
                    offset = int(jsonl_file.tell())
                    line = jsonl_file.readline()
                    if not line:
                        break
                    if line.strip():
                        row_index = len(offsets)
                        offsets.append(offset)
                        keys.append(f"{file_key}:{row_index}")
            self._line_offsets[file_key] = offsets
        return keys

    def _row_to_replay_sample(
        self, payload: dict[str, Any], file_name: str, line_index: int
    ) -> DraftReplaySample:
        row_source = "jsonl_token_replay"
        if "input_ids" in payload or "loss_mask" in payload:
            input_ids = _json_list_tensor(payload, "input_ids", dtype=torch.long)
            loss_mask = _json_list_tensor(payload, "loss_mask", dtype=torch.float32)
        elif "conversations" in payload:
            input_ids, loss_mask = self._conversation_to_input_ids_and_loss_mask(
                payload
            )
            row_source = "jsonl_conversations"
        else:
            raise KeyError(
                "jsonl_token_replay sample must contain input_ids/loss_mask or "
                "conversations"
            )
        if int(input_ids.numel()) != int(loss_mask.numel()):
            raise ValueError(
                "jsonl_token_replay input_ids/loss_mask length mismatch: "
                f"{int(input_ids.numel())} vs {int(loss_mask.numel())}"
            )
        attention_mask = _optional_json_list_tensor(
            payload, "attention_mask", dtype=torch.bool
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = _optional_json_list_tensor(
            payload, "position_ids", dtype=torch.long
        )
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=0).sub(1).clamp_min(0)

        sequence_length = int(input_ids.numel())
        start, end = self._feature_window(loss_mask)
        feature_positions = torch.arange(start, end, dtype=torch.long)
        draft_position_ids = position_ids[start:end].long() + 1
        metadata = {
            "source": row_source,
            "jsonl_path": file_name,
            "jsonl_line": line_index,
            "sequence_length": end - start,
            "full_sequence_length": sequence_length,
            "feature_start": start,
            "feature_end": end,
            "loss_tokens": int(loss_mask[start:end].sum().item()),
        }
        for key in ("id", "hash", "primary_id", "finish_reason"):
            if key in payload:
                metadata[key] = payload[key]
        return DraftReplaySample(
            algorithm=str(payload.get("algorithm", "EAGLE3")),
            input_ids=input_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            feature_positions=feature_positions,
            draft_position_ids=draft_position_ids,
            metadata=metadata,
        )

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.tokenizer_path:
            raise ValueError(
                "feature_store.tokenizer_path is required when "
                "jsonl_token_replay reads conversations rows"
            )
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "jsonl_token_replay conversations rows require transformers"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )
        return self._tokenizer

    def _conversation_to_input_ids_and_loss_mask(
        self, payload: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversations = payload.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(
                "jsonl_token_replay conversations must be a non-empty list"
            )
        messages = [_conversation_item_to_message(item) for item in conversations]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(
                "jsonl_token_replay train_on=last_assistant requires the final "
                "conversation item to be assistant"
            )

        tokenizer = self._ensure_tokenizer()
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = _token_ids_to_list(prompt_ids)
        full_ids = _token_ids_to_list(full_ids)
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(
                "jsonl_token_replay conversations produced no assistant tokens"
            )
        loss_mask = [0.0] * len(prompt_ids) + [1.0] * (len(full_ids) - len(prompt_ids))
        return (
            torch.tensor(full_ids, dtype=torch.long).reshape(-1),
            torch.tensor(loss_mask, dtype=torch.float32).reshape(-1),
        )

    def _feature_window(self, loss_mask: torch.Tensor) -> tuple[int, int]:
        sequence_length = int(loss_mask.numel())
        if sequence_length <= 0:
            raise ValueError("jsonl_token_replay input_ids must not be empty")
        max_seq_len = self.max_seq_len if self.max_seq_len > 0 else sequence_length
        max_seq_len = min(max_seq_len, sequence_length)
        if self.window_mode == "full":
            return 0, sequence_length
        if self.window_mode == "front":
            return 0, max_seq_len

        active = torch.nonzero(loss_mask.float() > 0, as_tuple=False).reshape(-1)
        if int(active.numel()) <= 0:
            return 0, max_seq_len
        first_loss = int(active[0].item())
        start = max(first_loss - 1, 0)
        end = min(start + max_seq_len, sequence_length)
        if end <= start:
            end = min(start + 1, sequence_length)
        return start, end


class VllmSafetensorsFeatureStore(TorchShardFeatureStore):
    """Feature store for vLLM-extracted hidden states saved as safetensors.

    Each sample is stored in an individual ``.safetensors`` file with the
    non-tensor schema/metadata recorded in the shared manifest. This mirrors
    the vLLM/speculators hidden-state extraction flow while exposing the same
    ``DraftFeatureStore`` interface used by standalone training.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_samples_per_shard: int = 1024,
        metadata: dict[str, Any] | None = None,
        strict_schema: bool = True,
        read_only: bool = False,
        shard_prefix: str = "hs",
    ):
        safetensors_metadata = dict(metadata or {})
        safetensors_metadata["format"] = "vllm_safetensors"
        super().__init__(
            path,
            max_samples_per_shard=max_samples_per_shard,
            metadata=safetensors_metadata,
            strict_schema=strict_schema,
            read_only=read_only,
            shard_prefix=shard_prefix,
        )

    def write_many(
        self, samples: list[DraftStoredSample | dict[str, Any]]
    ) -> list[str]:
        if self.read_only:
            raise RuntimeError(
                "Cannot write to a read-only VllmSafetensorsFeatureStore"
            )
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError(
                "feature_store.type=vllm_safetensors requires safetensors"
            ) from exc

        keys: list[str] = []
        for sample_like in samples:
            sample = _coerce_sample(sample_like, strict=self.strict_schema)
            sample_index = self._infer_next_shard_index()
            file_name = f"{self.shard_prefix}_{sample_index:06d}.safetensors"
            file_path = self.path / file_name
            tensor_payload = self._sample_to_safetensors(sample)
            with tempfile.NamedTemporaryFile(
                prefix=file_path.name,
                suffix=".tmp",
                dir=file_path.parent,
                delete=False,
            ) as tmp_file:
                tmp_name = tmp_file.name
            try:
                save_file(tensor_payload, tmp_name)
                os.replace(tmp_name, file_path)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)

            entry = {
                "path": file_name,
                "num_samples": 1,
                "num_tokens": _sample_token_count(sample.to_dict()),
                "sample": self._sample_manifest(sample),
            }
            with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
                manifest_file.write(
                    json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n"
                )
            self._manifest.append(entry)
            self._next_shard_index = sample_index + 1
            keys.append(f"{file_name}:0")
        return keys

    def flush(self) -> list[str]:
        return []

    def read(self, key: str) -> DraftFeatureSample:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "feature_store.type=vllm_safetensors requires safetensors"
            ) from exc

        file_name, sample_index = _parse_key(key)
        if int(sample_index) != 0:
            raise ValueError(
                f"Invalid vllm_safetensors key {key!r}; expected sample index 0"
            )
        entries = {str(entry.get("path")): entry for entry in self._load_manifest()}
        entry = entries.get(file_name)
        if entry is None:
            raise KeyError(f"Missing vllm_safetensors manifest entry for {file_name}")
        tensors = load_file(str(self.path / file_name), device="cpu")
        manifest_sample = dict(entry.get("sample") or {})
        payload: dict[str, Any] = {
            "schema_version": int(
                manifest_sample.get("schema_version", SCHEMA_VERSION)
            ),
            "algorithm": manifest_sample.get("algorithm", "EAGLE3"),
            "input_ids": tensors["input_ids"],
            "loss_mask": tensors["loss_mask"],
            "hidden_states": tensors["hidden_states"],
            "metadata": dict(manifest_sample.get("metadata") or {}),
        }
        if "metadata.hidden_positions" in tensors:
            payload["metadata"]["hidden_positions"] = tensors[
                "metadata.hidden_positions"
            ].long()
        for optional_key in (
            "last_hidden_states",
            "target",
            "target_logprobs",
            "position_ids",
        ):
            if optional_key in tensors:
                payload[optional_key] = tensors[optional_key]
        return DraftFeatureSample.from_dict(payload, strict=self.strict_schema)

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0) -> Iterator[str]:
        keys = [f"{entry['path']}:0" for entry in self._load_manifest()]
        if shuffle:
            random.Random(int(seed)).shuffle(keys)
        yield from keys

    def _sample_to_safetensors(
        self, sample: DraftFeatureSample
    ) -> dict[str, torch.Tensor]:
        payload = sample.to_dict()
        hidden_states = payload["hidden_states"]
        if not torch.is_tensor(hidden_states):
            raise TypeError(
                "feature_store.type=vllm_safetensors requires tensor hidden_states"
            )
        tensors = {
            "input_ids": payload["input_ids"].long().contiguous(),
            "loss_mask": payload["loss_mask"].float().contiguous(),
            "hidden_states": hidden_states.contiguous(),
        }
        for optional_key in (
            "last_hidden_states",
            "target",
            "target_logprobs",
            "position_ids",
        ):
            value = payload.get(optional_key)
            if value is not None and torch.is_tensor(value):
                tensors[optional_key] = value.contiguous()
        metadata = payload.get("metadata") or {}
        hidden_positions = metadata.get("hidden_positions")
        if hidden_positions is not None and torch.is_tensor(hidden_positions):
            tensors["metadata.hidden_positions"] = hidden_positions.long().contiguous()
        return tensors

    def _sample_manifest(self, sample: DraftFeatureSample) -> dict[str, Any]:
        return {
            "schema_version": sample.schema_version,
            "algorithm": sample.algorithm,
            "metadata": _json_safe_metadata(sample.metadata),
        }


def build_feature_store_from_config(
    feature_store_cfg,
    *,
    read_only: bool = False,
    metadata: dict[str, Any] | None = None,
    shard_prefix: str = "shard",
    transfer_queue_cfg: Any | None = None,
) -> Any:
    store_type = (
        str(feature_store_cfg.get("type", "torch_shard") or "torch_shard")
        .strip()
        .lower()
    )
    if store_type == "tq":
        if not read_only:
            raise ValueError(
                "feature_store.type=tq is a read-only Consumer data source"
            )
        from verl_speco.trainer.tq_feature_store import TQFeatureStore

        tq_cfg = transfer_queue_cfg
        if tq_cfg is None:
            tq_cfg = feature_store_cfg.get("tq")
        if tq_cfg is None:
            raise ValueError(
                "feature_store.type=tq requires the sibling training.transfer_queue configuration"
            )
        return TQFeatureStore.from_config(tq_cfg)
    if store_type == "torch_shard":
        store_cls: type[TorchShardFeatureStore] = TorchShardFeatureStore
    elif store_type == "token_replay":
        store_cls = TokenReplayFeatureStore
    elif store_type in {"vllm_safetensors", "safetensors"}:
        store_cls = VllmSafetensorsFeatureStore
    elif store_type in {"jsonl_token_replay", "jsonl"}:
        return JsonlTokenReplayFeatureStore(
            feature_store_cfg.get("path"),
            max_samples_per_shard=int(
                feature_store_cfg.get("max_samples_per_shard", 1024)
            ),
            metadata=metadata,
            strict_schema=bool(feature_store_cfg.get("strict_schema", True)),
            read_only=read_only,
            shard_prefix=shard_prefix,
            max_seq_len=int(feature_store_cfg.get("max_seq_len", 512) or 0),
            window_mode=str(feature_store_cfg.get("window_mode", "loss") or "loss"),
            tokenizer_path=feature_store_cfg.get("tokenizer_path"),
            trust_remote_code=bool(feature_store_cfg.get("trust_remote_code", False)),
            train_on=str(
                feature_store_cfg.get("train_on", "last_assistant") or "last_assistant"
            ),
        )
    else:
        raise NotImplementedError(f"Unsupported draft feature store type: {store_type}")
    return store_cls(
        feature_store_cfg.get("path"),
        max_samples_per_shard=int(feature_store_cfg.get("max_samples_per_shard", 1024)),
        metadata=metadata,
        strict_schema=bool(feature_store_cfg.get("strict_schema", True)),
        read_only=read_only,
        shard_prefix=shard_prefix,
    )


def _coerce_sample(
    sample_like: DraftStoredSample | dict[str, Any], *, strict: bool
) -> DraftFeatureSample:
    if isinstance(sample_like, DraftFeatureSample):
        sample_like.validate(strict=strict)
        return sample_like
    if isinstance(sample_like, dict):
        return DraftFeatureSample.from_dict(sample_like, strict=strict)
    raise TypeError(f"Unsupported draft feature sample type: {type(sample_like)!r}")


def _coerce_replay_sample(
    sample_like: DraftStoredSample | dict[str, Any], *, strict: bool
) -> DraftReplaySample:
    if isinstance(sample_like, DraftReplaySample):
        sample_like.validate(strict=strict)
        return sample_like
    if isinstance(sample_like, dict):
        return DraftReplaySample.from_dict(sample_like, strict=strict)
    raise TypeError(f"Unsupported draft replay sample type: {type(sample_like)!r}")


def _cpu_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, (list, tuple)):
        return [_cpu_tensor_tree(item) for item in value]
    return value


def _sample_token_count(sample: dict[str, Any]) -> int:
    loss_mask = sample.get("loss_mask")
    if torch.is_tensor(loss_mask):
        return int(cast(Any, loss_mask).detach().float().sum().item())
    input_ids = sample.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(cast(Any, input_ids).numel())
    return 0


def _metadata_int(sample: dict[str, Any], name: str) -> int | None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _min_metadata_int(samples: list[dict[str, Any]], name: str) -> int | None:
    values: list[int | None] = [_metadata_int(sample, name) for sample in samples]
    int_values = [value for value in values if value is not None]
    return min(int_values) if int_values else None


def _max_metadata_int(samples: list[dict[str, Any]], name: str) -> int | None:
    values: list[int | None] = [_metadata_int(sample, name) for sample in samples]
    int_values = [value for value in values if value is not None]
    return max(int_values) if int_values else None


def _parse_key(key: str) -> tuple[str, int]:
    if ":" not in key:
        raise ValueError(f"Invalid feature key {key!r}; expected 'shard.pt:index'")
    shard_name, index = key.rsplit(":", 1)
    return shard_name, int(index)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=path.name, suffix=".tmp", dir=path.parent, delete=False
    ) as tmp_file:
        tmp_name = tmp_file.name
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _load_jsonl_offset(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as jsonl_file:
        jsonl_file.seek(int(offset))
        line = jsonl_file.readline().decode("utf-8").strip()
    if not line:
        raise ValueError(f"JSONL offset {offset} in {path} points to an empty line")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise TypeError(f"JSONL offset {offset} in {path} must contain a JSON object")
    return payload


def _json_list_tensor(
    payload: dict[str, Any], key: str, *, dtype: torch.dtype
) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"jsonl_token_replay sample missing required key {key!r}")
    value = payload[key]
    if not isinstance(value, list):
        raise TypeError(
            f"jsonl_token_replay {key} must be a JSON list, got {type(value).__name__}"
        )
    return torch.tensor(value, dtype=dtype).reshape(-1)


def _optional_json_list_tensor(
    payload: dict[str, Any], key: str, *, dtype: torch.dtype
) -> torch.Tensor | None:
    if key not in payload or payload[key] is None:
        return None
    return _json_list_tensor(payload, key, dtype=dtype)


def _normalize_conversation_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def _conversation_item_to_message(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        raise TypeError("jsonl_token_replay conversations entries must be JSON objects")
    role = _normalize_conversation_role(item.get("role", item.get("from")))
    content = item.get("content", item.get("value", ""))
    if role not in {"system", "user", "assistant"}:
        raise ValueError(
            f"Unsupported conversation role for jsonl_token_replay: {role!r}"
        )
    return {"role": role, "content": str(content)}


def _token_ids_to_list(value: Any) -> list[int]:
    if hasattr(value, "data") and isinstance(getattr(value, "data", None), dict):
        data = getattr(value, "data")
        if "input_ids" in data:
            value = data["input_ids"]
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "tokenizer.apply_chat_template must return token ids as a list, "
            f"tuple, tensor, or tolist()-compatible value; got {type(value).__name__}"
        )
    return [int(token_id) for token_id in value]


def _json_safe_metadata(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() <= 128:
            return value.detach().cpu().tolist()
        return {
            "__tensor__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
