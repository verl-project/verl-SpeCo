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
"""Streaming TransferQueue feature source for standalone drafter training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verl_speco.integration.transferqueue_bridge import (
    clear_samples,
    close_transfer_queue_client,
    configure_transfer_queue,
    connect_ray_cluster,
    connect_transfer_queue_client,
    get_samples,
    list_samples,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.transport.drafter_sample_protocol import (
    ExpectedFeatureConfig,
    decode_sample,
    parse_ready_tag,
)


@dataclass(frozen=True)
class ReadyEntry:
    """One discoverable sample record; payload tensors are not loaded yet."""

    key: str
    tag: dict[str, Any]


@dataclass(frozen=True)
class EosMetadata:
    """End-of-stream control record published by the Producer."""

    key: str
    run_id: str
    schema_version: int
    total_samples: int


class TQFeatureStore:
    """Thin Consumer adapter over the shared TransferQueue bridge.

    This intentionally does not implement the static ``iter_keys/read`` feature
    store protocol. TQ keys are added by the Producer and removed after a
    successful optimizer step, so discovery must happen for every global batch.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = _plain_dict(config)
        self.run_id = str(self.config.get("run_id") or "").strip()
        if not self.run_id:
            raise ValueError("transfer_queue.run_id is required for a TQ Consumer")
        self.schema_version = int(self.config.get("schema_version", 1))
        ray_cfg = _plain_dict(self.config.get("ray") or {})
        self.ray_address = str(ray_cfg.get("address") or "").strip()
        self.ray_namespace = str(ray_cfg.get("namespace") or "").strip() or None
        if not self.ray_address:
            raise ValueError("transfer_queue.ray.address is required for a TQ Consumer")
        self._connected = False
        # First version deliberately checks only run/protocol. Tensor
        # presence, lengths, shape and dtype self-consistency remain enforced by
        # decode_sample; model/tokenizer/layer identity checks stay disabled.
        self.expected_config = ExpectedFeatureConfig(
            run_id=self.run_id,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_config(cls, config: Any) -> "TQFeatureStore":
        return cls(_plain_dict(config))

    def connect(self) -> None:
        if self._connected:
            return
        if not configure_transfer_queue(self.config):
            raise RuntimeError(
                "TQ Consumer requires transfer_queue.enable=true and TransferQueue==0.1.10"
            )
        connect_ray_cluster(self.ray_address, self.ray_namespace)
        connect_transfer_queue_client()
        self._connected = True

    def list_ready(self, run_id: str | None = None) -> list[ReadyEntry]:
        self._require_connected()
        expected_run_id = str(run_id or self.run_id)
        ready: list[ReadyEntry] = []
        for key, raw_tag in list_samples().items():
            tag = dict(raw_tag)
            meta = parse_ready_tag(
                tag,
                run_id=expected_run_id,
                schema_version=self.schema_version,
            )
            if meta is None:
                continue
            ready.append(ReadyEntry(key=str(key), tag=tag))
        ready.sort(key=lambda entry: (int(entry.tag["sequence_no"]), entry.key))
        return ready

    def owner_ready(self) -> bool:
        """Whether the standalone Owner published this run's readiness marker."""

        self._require_connected()
        key = f"control:v{self.schema_version}:{self.run_id}:owner-ready"
        tag = list_samples().get(key)
        if not isinstance(tag, Mapping):
            return False
        return (
            tag.get("record_type") == "control"
            and tag.get("status") == "owner_ready"
            and str(tag.get("run_id") or "") == self.run_id
            and int(tag.get("schema_version", -1)) == self.schema_version
        )

    def get_many(self, entries: Sequence[ReadyEntry]) -> list[DraftFeatureSample]:
        self._require_connected()
        if not entries:
            return []
        records = get_samples([entry.key for entry in entries])
        if len(records) != len(entries):
            raise RuntimeError(
                f"TQ returned {len(records)} records for {len(entries)} requested entries"
            )
        samples: list[DraftFeatureSample] = []
        for entry, (key, fields) in zip(entries, records, strict=True):
            if key != entry.key:
                raise RuntimeError(
                    f"TQ batch result order mismatch: got key={key!r}, expected={entry.key!r}"
                )
            samples.append(
                decode_sample(
                    key=key,
                    tag=entry.tag,
                    fields=fields,
                    expected_config=self.expected_config,
                )
            )
        return samples

    def clear_many(self, keys: Sequence[str]) -> None:
        self._require_connected()
        clear_samples([str(key) for key in keys])

    def read_eos(self, run_id: str | None = None) -> EosMetadata | None:
        self._require_connected()
        expected_run_id = str(run_id or self.run_id)
        for key, raw_tag in list_samples().items():
            tag = dict(raw_tag)
            if tag.get("record_type") != "control" or tag.get("status") != "eos":
                continue
            if str(tag.get("run_id") or "") != expected_run_id:
                continue
            if int(tag.get("schema_version", -1)) != self.schema_version:
                continue
            return EosMetadata(
                key=str(key),
                run_id=expected_run_id,
                schema_version=self.schema_version,
                total_samples=int(tag.get("total_samples", 0)),
            )
        return None

    def close_local(self) -> None:
        if not self._connected:
            return
        close_transfer_queue_client()
        self._connected = False

    def close(self) -> None:
        """Compatibility with the standalone loop's existing cleanup path."""

        self.close_local()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "TQFeatureStore.connect() must be called before data access"
            )


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(value, DictConfig):
            converted = OmegaConf.to_container(value, resolve=True)
            if not isinstance(converted, dict):
                raise TypeError("Expected a mapping configuration")
            return dict(converted)
    except ImportError:  # pragma: no cover - the project depends on OmegaConf
        pass
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    raise TypeError(f"Expected a mapping configuration, got {type(value)!r}")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


__all__ = ["EosMetadata", "ReadyEntry", "TQFeatureStore"]
