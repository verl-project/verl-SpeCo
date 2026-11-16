# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fail-closed standalone CI result validation."""
import unittest
import ast
from pathlib import Path
import types
from unittest.mock import patch

from ci.verify_dspark_standalone_checkpoint import changed_parameters, read_training_losses, validate_checkpoint_keys, resolve_checkpoint_layer_ids


class TestStandaloneResultValidation(unittest.TestCase):
    def setUp(self):
        # Execute the actual pure readers without importing the GPU trainer and
        # its optional dependencies into these CPU contract tests.
        path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"_source_target_layer_ids", "_source_vllm_aux_layer_ids"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        module = types.ModuleType("verl_speco.trainer.draft_training_loop")
        source = "from __future__ import annotations\n" + "\n".join(ast.unparse(node) for node in nodes)
        exec(compile(source, str(path), "exec"), module.__dict__)
        mock_import = patch.dict("sys.modules", {module.__name__: module})
        mock_import.start()
        self.addCleanup(mock_import.stop)

    def test_speculators_output_ids_are_not_shifted_twice(self):
        self.assertEqual(
            resolve_checkpoint_layer_ids({"aux_hidden_state_layer_ids": [2, 10, 18], "target_layer_ids": [2, 10, 18]}, "DSPARK"),
            ((1, 9, 17), (2, 10, 18)),
        )

    def test_speco_decoder_ids_and_exported_alias_agree(self):
        for config in (
            {"target_layer_ids": [1, 9, 17]},
            {"target_layer_ids": [1, 9, 17], "eagle_aux_hidden_state_layer_ids": [2, 10, 18]},
            {"dspark_config": {"target_layer_ids": [1, 9, 17]}},
        ):
            with self.subTest(config=config):
                self.assertEqual(resolve_checkpoint_layer_ids(config, "DSPARK"), ((1, 9, 17), (2, 10, 18)))

    def test_public_checkpoint_may_include_target_head(self):
        validate_checkpoint_keys({"fc.weight", "lm_head.weight"}, {"fc.weight"}, {"fc.weight"})

    def test_optional_head_does_not_hide_missing_or_unknown_weights(self):
        for saved in ({"lm_head.weight"}, {"fc.weight", "unknown.weight"}):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                validate_checkpoint_keys(saved, {"fc.weight"}, {"fc.weight"})

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
