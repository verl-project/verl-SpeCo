# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fail-closed standalone CI result validation."""
import unittest

from ci.verify_dspark_standalone_checkpoint import changed_parameters, read_training_losses


class TestStandaloneResultValidation(unittest.TestCase):
    def test_losses_ignore_unrelated_logs_and_accept_ray_prefix(self):
        text = "unrelated avg_loss=nan\n(Worker pid=123) [standalone drafter metrics] step=5 avg_loss=0.1234 lr=1e-5\n"
        self.assertEqual(read_training_losses(text), [0.1234])

    def test_missing_or_nonfinite_losses_fail(self):
        for text in ("", "avg_loss=0.1", "[standalone drafter metrics] avg_loss=nan", "[standalone drafter metrics] avg_loss=inf"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_training_losses(text)

    def test_unchanged_checkpoint_fails(self):
        value = {"fc.weight": {"shape": [2, 4], "sha256": "initial"}}
        with self.assertRaises(ValueError):
            changed_parameters(value, value)

    def test_changed_weight_is_reported(self):
        before = {"fc.weight": {"shape": [2, 4], "sha256": "initial"}}
        after = {"fc.weight": {"shape": [2, 4], "sha256": "trained"}}
        self.assertEqual(changed_parameters(before, after), ["fc.weight"])

    def test_missing_weights_or_shape_changes_fail(self):
        before = {"fc.weight": {"shape": [2, 4], "sha256": "initial"}}
        for after in ({}, {"fc.weight": {"shape": [4, 2], "sha256": "trained"}}):
            with self.subTest(after=after), self.assertRaises(ValueError):
                changed_parameters(before, after)


if __name__ == "__main__":
    unittest.main()
