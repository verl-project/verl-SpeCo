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
"""Non-destructive DS4 serving export using AIR's TP merge and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl_speco.integration.vllm_ascend_implementation import (
    compatible_implementation_fingerprint,
)
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract


def export_checkpoint(
    *,
    checkpoint: str,
    model: str,
    output: str,
    dp_rank: int = 0,
    source_root: str | None = None,
    format: str = "patched",
    max_shard_bytes: int = 2 * 1024**3,
) -> dict:
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    metadata = json.loads((checkpoint_path / "speco_ds4.json").read_text())
    if metadata["identity"]["contract"].get("layout") == "deepseek_v4_mtp_pre_hc_v1":
        from verl_speco.models.deepseek_v4.mtp_artifact import export_mtp_checkpoint

        return export_mtp_checkpoint(
            checkpoint=checkpoint,
            model=model,
            output=output,
            dp_rank=dp_rank,
            source_root=source_root,
            format=format,
            max_shard_bytes=max_shard_bytes,
        )
    contract = DeepSeekV4Contract.from_model(model)
    if metadata["identity"]["contract"] != contract.identity():
        raise ValueError("Export model configuration differs from training")
    if not compatible_implementation_fingerprint(
        metadata["identity"]["source_fingerprint"]
    ):
        raise ValueError("AIR exporter source differs from the training implementation")
    dp_size = int(metadata["topology"]["data_parallel_size"])
    if not 0 <= dp_rank < dp_size:
        raise ValueError("Export DP replica is outside the saved topology")
    from verl_speco.models.deepseek_v4 import export as exporter
    from verl_speco.models.deepseek_v4.export_validation import (
        validate_tp_export_replicas,
    )

    files = sorted(Path(model).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            "Target safetensors required to validate DSpark TP layout"
        )
    target_shapes = exporter._target_tensor_shapes(files)
    validate_tp_export_replicas(
        checkpoint_path, dp_rank=dp_rank, target_shapes=target_shapes
    )

    if format in {"draft-safetensors", "draft-pytorch"}:
        from verl_speco.models.deepseek_v4.artifact import save_draft_artifact

        weights = exporter._load_export_weights(
            checkpoint_path,
            "dspark",
            data_parallel_rank=dp_rank,
            target_shapes=target_shapes,
        )
        return save_draft_artifact(
            weights,
            output=output,
            config=json.loads((Path(model) / "config.json").read_text()),
            step=metadata["progress"]["global_step"],
            serialization="pytorch" if format == "draft-pytorch" else "safetensors",
            max_shard_bytes=max_shard_bytes,
        )
    if format != "patched":
        raise ValueError("Unknown draft export format")
    exporter.export_patched_model(
        str(checkpoint_path),
        Path(model),
        Path(output),
        layout="dspark",
        copy_mode="copy",
        data_parallel_rank=dp_rank,
    )
    exporter.verify_patched_model_weights(
        str(checkpoint_path), Path(output), layout="dspark", data_parallel_rank=dp_rank
    )
    report = {
        "status": "passed",
        "stage": "export",
        "output": str(Path(output).resolve()),
        "target_revision": "0731",
        "weights_verified": True,
        "inference_validated": False,
    }
    (Path(output) / "speco_export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Export and verify AIR DS4 0731 DSpark weights into a new serving checkpoint"
    )
    for name in ("checkpoint", "model", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--source-root",
        default=None,
        help="Deprecated; the native backend is packaged in SpeCo",
    )
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument(
        "--format",
        choices=("patched", "draft-safetensors", "draft-pytorch"),
        default="patched",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    print(json.dumps(export_checkpoint(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
