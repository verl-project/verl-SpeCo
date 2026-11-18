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
"""Import captured features into the SpeCo store and adapt them to DSpark."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from verl_speco.trainer.deepseek_v4_contract import (
    DeepSeekV4Contract,
    from_engine_tensors,
    to_engine_tensors,
)
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore
from verl_speco.trainer.vllm_ascend_mask import (
    build_segments,
    collate_features,
    estimate_anchor_capacity,
)


def import_capture(
    *,
    model: str,
    capture: str,
    masks: str,
    output: str,
    allow_ordinal: bool = False,
    algorithm: str = "DSPARK",
) -> dict:
    """Commit a new SpeCo store only after every input request validates.

    The supplied masks must describe the exact token sequence used for capture.
    Ordinal matching is explicit because it relies on the input JSONL order.
    Failed staging directories are kept for diagnosis and are never consumable
    under the requested output path.
    """
    sample_factory = from_engine_tensors
    contract_type = DeepSeekV4Contract
    if algorithm.upper() == "MTP":
        from verl_speco.trainer.deepseek_v4_mtp_contract import (
            DeepSeekV4MTPContract,
        )
        from verl_speco.trainer.deepseek_v4_mtp_contract import (
            from_engine_tensors as mtp_sample,
        )

        contract_type, sample_factory = DeepSeekV4MTPContract, mtp_sample
    elif algorithm.upper() != "DSPARK":
        raise ValueError("Capture import algorithm must be MTP or DSPARK")
    contract = contract_type.from_model(model)
    capture_root = Path(capture).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite feature store: {output_path}")
    rows = {}
    for line in Path(masks).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["sample_idx"])
        if key in rows:
            raise ValueError(f"Duplicate sample_idx: {key}")
        if not key or Path(key).name != key or key in {".", ".."}:
            raise ValueError(f"Unsafe sample_idx: {key!r}")
        if "loss_mask" not in row:
            raise ValueError(
                "Mask records require SpeCo loss_mask (1=supervised, 0=ignored)"
            )
        rows[key] = row["loss_mask"]
    if not rows:
        raise ValueError("Mask file contains no requests")
    matches = sum((capture_root / key).is_dir() for key in rows)
    if matches != len(rows):
        if matches or not allow_ordinal:
            raise ValueError(
                "Capture keys do not match all mask sample_idx values; ordinal mapping requires --allow-ordinal"
            )
        rows = {f"request_{i:06d}": mask for i, mask in enumerate(rows.values())}
        if not all((capture_root / key).is_dir() for key in rows):
            raise ValueError("Ordinal capture directories are incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.incomplete-", dir=output_path.parent
        )
    )
    store = TorchShardFeatureStore(
        staging,
        max_samples_per_shard=1,
        metadata={
            "ds4_contract": contract.identity(),
            **({"source": "mtp_capture_import"} if algorithm.upper() == "MTP" else {}),
        },
    )
    tokens = 0
    try:
        for key, mask in rows.items():
            tensors = {
                name: torch.load(
                    capture_root / key / f"step_0_{name}.cpu.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for name in (
                    (
                        "input_ids",
                        "positions",
                        "hidden_states",
                        "target_head_hidden_states",
                    )
                    if algorithm.upper() == "DSPARK"
                    else ("input_ids", "positions", "hidden_states")
                )
            }
            sample = sample_factory(tensors, mask, contract, sample_id=key)
            tokens += sample.input_ids.numel()
            store.write_many([sample])
    finally:
        store.close()
    # The store deliberately retains raw pre-HC states in metadata; the native
    # TorchShard and TQ tensor-tree codecs support this without modification.
    if output_path.exists():
        raise FileExistsError(output_path)
    os.rename(staging, output_path)
    return {
        "status": "passed",
        "samples": len(rows),
        "tokens": tokens,
        "output": str(output_path),
        "contract": contract.identity(),
    }


class StoreDataset(Dataset[dict[str, Any]]):
    """Read SpeCo samples with positive supervision and context-only overlap."""

    def __init__(self, path: str, contract: DeepSeekV4Contract, cfg):
        self.store = TorchShardFeatureStore(path, read_only=True)
        self.contract = contract
        if self.store.get_metadata().get("ds4_contract") != contract.identity():
            raise ValueError(
                "Feature store does not match the selected DS4 0731 contract"
            )
        self.keys = list(self.store.iter_keys())
        self.segment_length = cfg.train.get("segment_length")
        self.max_len = cfg.train.get("max_len")
        self.segment_overlap = int(cfg.train.get("segment_overlap", 0))
        self.segment_min_loss_tokens = int(cfg.train.get("segment_min_loss_tokens", 1))
        if self.max_len is not None and int(self.max_len) <= 0:
            raise ValueError("train.max_len must be positive")
        if self.segment_min_loss_tokens < 1:
            raise ValueError("At least one supervised token per segment is required")
        if self.segment_length is not None:
            self.segment_length = int(self.segment_length)
            if (
                self.max_len is not None
                or not 0 <= self.segment_overlap < self.segment_length
            ):
                raise ValueError(
                    "Use segment_length with 0 <= overlap < length, without max_len"
                )
            if self.segment_overlap < int(cfg.train.get("ttt_length", 1)):
                raise ValueError("segment_overlap must cover train.ttt_length")
        self.samples = []
        for key in self.keys:
            sample = self.store.read(key)
            if not isinstance(sample, DraftFeatureSample):
                raise TypeError("DSpark training requires materialized feature samples")
            contract.validate(sample)
            self.samples.append(
                {
                    "key": key,
                    "mask": sample.loss_mask.tolist(),
                }
            )
        # Keep overlap as context only, using SpeCo supervision semantics.
        self.segments = build_segments(self)
        self.sequence_lengths = [s.end - s.start for s in self.segments]
        self.source_ids = [s.sample_index for s in self.segments]
        self.segment_starts = [s.start for s in self.segments]
        if not self.segments:
            raise ValueError("No trainable segments in the feature store")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment = self.segments[index]
        sample = self.store.read(self.keys[segment.sample_index])
        values = to_engine_tensors(sample, self.contract)
        result = {
            key: value[segment.start : segment.end] for key, value in values.items()
        }
        mask = torch.zeros(segment.end - segment.start, dtype=torch.float32)
        start = segment.loss_start - segment.start
        end = segment.loss_end - segment.start
        mask[start:end] = result["loss_mask"][start:end]
        result["loss_mask"] = mask
        return result

    def anchor_capacities(self, *, num_anchors: int, spacing: int) -> list[int]:
        return [
            estimate_anchor_capacity(
                self.samples[s.sample_index]["mask"],
                s,
                num_anchors=num_anchors,
                min_anchor_spacing=spacing,
            )
            for s in self.segments
        ]


def build_loader(dataset: StoreDataset, cfg, *, dp_rank: int, dp_size: int):
    """Shard over DP only: all ranks in a TP group build identical batches."""
    from torch.utils.data import DataLoader, DistributedSampler

    from verl_speco.data import deepseek_v4_sampler as data

    if int(cfg.train.get("num_workers", 0)) != 0:
        raise ValueError(
            "The opt-in SpeCo store adapter currently requires train.num_workers=0"
        )
    batch_size = int(cfg.deepspeed.train_micro_batch_size_per_gpu)
    if batch_size <= 0 or not 0 <= dp_rank < dp_size:
        raise ValueError("Invalid DP topology or micro batch size")
    from verl_speco.backends import vllm_ascend_options as optimizations

    multiplier = int(
        os.environ.get(
            "MTP_LENGTH_BUCKET_MULTIPLIER", cfg.train.get("length_bucket_multiplier", 0)
        )
    )
    seed = int(cfg.debug.get("seed") or 0)
    aligned = bool(
        optimizations.get_dspark_optimization(cfg, "dp_align_length_buckets", False)
    )
    locality = bool(
        optimizations.get_optimization(cfg, "lazy_segment_source_locality", False)
    )
    sampler: (
        data.DistributedLengthBucketBatchSampler
        | data.DistributedSourceLocalBatchSampler
        | DistributedSampler
    )
    kwargs: dict[str, Any] = {}
    if multiplier > 0 and aligned:
        values = dataset.anchor_capacities(
            num_anchors=int(cfg.train.dspark_num_anchors),
            spacing=int(
                cfg.train.get("dspark_anchor_min_spacing")
                or dataset.contract.block_size
            ),
        )
        sampler = data.DistributedLengthBucketBatchSampler(
            values,
            batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
            bucket_size_multiplier=multiplier,
            align_replicas_by_length=True,
        )
        kwargs["batch_sampler"] = sampler
    elif locality and dataset.segment_length is not None:
        sampler = data.DistributedSourceLocalBatchSampler(
            dataset.source_ids,
            dataset.segment_starts,
            batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
        )
        kwargs["batch_sampler"] = sampler
    elif multiplier > 0:
        sampler = data.DistributedLengthBucketBatchSampler(
            dataset.sequence_lengths,
            batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
            bucket_size_multiplier=multiplier,
        )
        kwargs["batch_sampler"] = sampler
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=seed
        )
        kwargs.update(batch_size=batch_size, sampler=sampler)
    pad = dataset.segment_length if cfg.train.get("pad_segment_tails", False) else None
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset,
        collate_fn=partial(collate_features, pad_to_length=pad),
        num_workers=0,
        **kwargs,
    )
    if not len(loader):
        raise ValueError("No complete DP training batches")
    return sampler, loader


def main():
    parser = argparse.ArgumentParser(
        description="Import AIR DS4 0731 captures into a SpeCo feature store"
    )
    for name in ("model", "capture", "masks", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--allow-ordinal", action="store_true")
    parser.add_argument("--algorithm", choices=("DSPARK", "MTP"), default="DSPARK")
    args = vars(parser.parse_args())
    print(json.dumps(import_capture(**args), indent=2))


if __name__ == "__main__":
    main()
