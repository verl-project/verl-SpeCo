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
# Adapted original artifact validation; no external checkout imports.
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from verl_speco.models.deepseek_v4.checkpoint import inspect_ds4_checkpoint_format


class TestDS4CheckpointFormat(unittest.TestCase):
    def _directory(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_detects_pure_bf16_checkpoint(self):
        root = self._directory()
        save_file(
            {
                "mtp.0.e_proj.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
                "mtp.0.hc_head_fn": torch.zeros(2, 3, dtype=torch.float32),
                "model.embed_tokens.weight": torch.zeros(
                    8,
                    3,
                    dtype=torch.bfloat16,
                ),
            },
            str(root / "model.safetensors"),
        )

        report = inspect_ds4_checkpoint_format(root)

        self.assertEqual(report.resolved_format, "bf16")
        self.assertIsNone(report.runtime_quantization)
        self.assertFalse(report.requires_dequantization)
        self.assertEqual(report.dtype_counts, {"BF16": 1, "F32": 1})

    def test_detects_w8a8_checkpoint(self):
        root = self._directory()
        save_file(
            {
                "mtp.0.e_proj.weight": torch.zeros(4, 3, dtype=torch.int8),
                "mtp.0.e_proj.weight_scale": torch.ones(4, 1),
                "mtp.0.hc_head_fn": torch.zeros(2, 3),
            },
            str(root / "model.safetensors"),
        )

        report = inspect_ds4_checkpoint_format(root)

        self.assertEqual(report.resolved_format, "w8a8")
        self.assertEqual(report.runtime_quantization, "ascend")
        self.assertTrue(report.requires_dequantization)
        self.assertEqual(report.scale_tensors, 1)

    def test_detects_bf16_mtp_in_quantized_model_package(self):
        root = self._directory()
        key = "mtp.0.e_proj.weight"
        save_file(
            {key: torch.zeros(4, 3, dtype=torch.bfloat16)},
            str(root / "model.safetensors"),
        )
        (root / "quant_model_description.json").write_text(
            json.dumps({key: "FLOAT", "model.layers.0.weight": "W8A8_DYNAMIC"}),
            encoding="utf-8",
        )
        (root / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "ascend"}}),
            encoding="utf-8",
        )

        report = inspect_ds4_checkpoint_format(root)

        self.assertEqual(report.resolved_format, "bf16")
        self.assertEqual(report.runtime_quantization, "ascend")
        self.assertTrue(report.is_mixed_runtime_checkpoint)
        self.assertFalse(report.requires_dequantization)

    def test_rejects_explicit_bf16_for_int8_weights(self):
        root = self._directory()
        save_file(
            {
                "mtp.0.e_proj.weight": torch.zeros(4, 3, dtype=torch.int8),
                "mtp.0.e_proj.weight_scale": torch.ones(4, 1),
            },
            str(root / "model.safetensors"),
        )

        with self.assertRaisesRegex(ValueError, "INT8 MTP matrix"):
            inspect_ds4_checkpoint_format(root, "bf16")

    def test_rejects_explicit_w8a8_for_bf16_weights(self):
        root = self._directory()
        save_file(
            {
                "mtp.0.e_proj.weight": torch.zeros(
                    4,
                    3,
                    dtype=torch.bfloat16,
                )
            },
            str(root / "model.safetensors"),
        )

        with self.assertRaisesRegex(ValueError, "no INT8 MTP matrix"):
            inspect_ds4_checkpoint_format(root, "w8a8")

    def test_rejects_stale_quant_description_for_bf16_weight(self):
        root = self._directory()
        key = "mtp.0.e_proj.weight"
        save_file(
            {key: torch.zeros(4, 3, dtype=torch.bfloat16)},
            str(root / "model.safetensors"),
        )
        (root / "quant_model_description.json").write_text(
            json.dumps({key: "W8A8_DYNAMIC"}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "not marked FLOAT"):
            inspect_ds4_checkpoint_format(root)

    def test_rejects_missing_mixed_description_for_bf16_weight(self):
        root = self._directory()
        key = "mtp.0.e_proj.weight"
        save_file(
            {key: torch.zeros(4, 3, dtype=torch.bfloat16)},
            str(root / "model.safetensors"),
        )
        (root / "quant_model_description.json").write_text(
            json.dumps({"model.layers.0.weight": "W8A8_DYNAMIC"}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "missing from"):
            inspect_ds4_checkpoint_format(root)

    def test_rejects_global_quantization_without_mixed_description(self):
        root = self._directory()
        save_file(
            {
                "mtp.0.e_proj.weight": torch.zeros(
                    4,
                    3,
                    dtype=torch.bfloat16,
                )
            },
            str(root / "model.safetensors"),
        )
        (root / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "ascend"}}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "without quant_model_description"):
            inspect_ds4_checkpoint_format(root)


if __name__ == "__main__":
    unittest.main()
