# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Scheduler-assigned hidden references, independent of TransferQueue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.standalone_data import ReadyEntry
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ExpectedFeatureConfig,
)


class ReferenceFeatureStore:
    """Prepare hidden references without connecting to TransferQueue.

    Ready metadata is owned by the Driver's scheduler, not this rank-local adapter.
    """

    uses_transfer_queue = False

    def __init__(self, config: Mapping[str, Any]):
        self.config = _plain_dict(config)
        lifetime = self.config.get("reference_lifetime", "temporary")
        if lifetime not in {"temporary", "persistent"}:
            raise ValueError("Invalid hidden reference lifetime")
        self.release_hidden_states_after_use = lifetime == "temporary"
        self.run_id = str(self.config.get("run_id") or "").strip()
        if not self.run_id:
            raise ValueError("run_id is required for a reference Consumer")
        self.schema_version = int(
            self.config.get("schema_version", PROTOCOL_SCHEMA_VERSION)
        )
        ray_cfg = _plain_dict(self.config.get("ray") or {})
        self.ray_address = str(ray_cfg.get("address") or "").strip()
        self.ray_namespace = str(ray_cfg.get("namespace") or "").strip() or None
        self._connected = False
        expected = _plain_dict(self.config.get("expected_feature") or {})
        target_layer_ids = expected.get("target_layer_ids")
        self.expected_config = ExpectedFeatureConfig(
            run_id=self.run_id,
            schema_version=self.schema_version,
            algorithm=_optional_string(expected.get("algorithm")),
            target_model_path=_optional_string(expected.get("target_model_path")),
            target_revision=_optional_string(expected.get("target_revision")),
            tokenizer_fingerprint=_optional_string(
                expected.get("tokenizer_fingerprint")
            ),
            target_layer_ids=(
                tuple(int(value) for value in target_layer_ids)
                if target_layer_ids is not None
                else None
            ),
            hidden_states_layout=_optional_string(expected.get("hidden_states_layout")),
            hidden_dtype=_optional_string(expected.get("hidden_dtype")),
        )
        from verl_speco.producer.hidden_states_store import build_hidden_states_store

        self.hidden_states_store = build_hidden_states_store(
            self.config.get("hidden_states_store")
        )
        if lifetime == "persistent" and self.hidden_states_store.backend == "file":
            self.hidden_states_store.wait_for_writer = False

    @classmethod
    def from_config(cls, config: Any) -> ReferenceFeatureStore:
        return cls(_plain_dict(config))

    def connect(self) -> None:
        self._connected = True

    def owner_ready(self) -> bool:
        self._require_connected()
        return True

    def clear_many(self, keys: Sequence[str]) -> None:
        self._require_connected()
        # Only scheduler completion releases metadata; cleanup releases hidden references.

    def list_ready(self, run_id: str | None = None) -> list[ReadyEntry]:
        raise RuntimeError(
            "Hidden references must arrive through scheduler assignments"
        )

    def get_many(self, entries: Sequence[ReadyEntry]) -> list[DraftFeatureSample]:
        raise RuntimeError("Hidden references require CPU batch preparation")

    def close_local(self) -> None:
        self.hidden_states_store.close()
        self._connected = False

    def prepare_batch(self, entries: Sequence[ReadyEntry], **options) -> dict[str, Any]:
        """Materialize only scheduler-assigned file tags; never fetch TQ fields."""
        import torch

        from verl_speco.trainer.standalone_batch_prepare import (
            prepare_standalone_cpu_batch,
        )
        from verl_speco.trainer.target_feature_replay import FeatureContract
        from verl_speco.transport.drafter_sample_protocol import (
            SampleMetadata,
            make_sample_key,
        )

        self._require_connected()
        if not entries:
            raise ValueError("Cannot prepare an empty file batch")
        expected = self.expected_config
        for entry in entries:
            meta = SampleMetadata.from_dict(entry.tag)
            if meta.run_id != self.run_id or entry.key != make_sample_key(meta):
                raise ValueError("Hidden reference identity mismatch")
            contract = entry.tag["feature_contract"]
            for name, actual in (
                ("algorithm", contract.get("algorithm")),
                ("target_model_path", contract.get("target_model_id")),
                ("target_revision", contract.get("target_model_revision")),
                ("tokenizer_fingerprint", contract.get("tokenizer_fingerprint")),
                ("hidden_states_layout", contract.get("hidden_states_layout")),
                ("hidden_dtype", str(contract.get("dtype", "")).removeprefix("torch.")),
                ("target_layer_ids", tuple(contract.get("target_layer_ids", []))),
            ):
                wanted = getattr(expected, name)
                if wanted is not None and actual != wanted:
                    raise ValueError(
                        f"Unexpected file feature {name}: {actual!r} != {wanted!r}"
                    )
        values = dict(entries[0].tag["feature_contract"])
        values["dtype"] = getattr(torch, values["dtype"].removeprefix("torch."))
        return prepare_standalone_cpu_batch(
            [entry.tag for entry in entries],
            FeatureContract(**values),
            hidden_states_store=getattr(self, "hidden_states_store", None),
            read_timeout=float(self.config.get("io_timeout_seconds", 30)),
            **options,
        )

    def close(self) -> None:
        """Compatibility with the standalone loop's existing cleanup path."""

        self.close_local()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "ReferenceFeatureStore.connect() must be called before data access"
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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
