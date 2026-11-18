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
# Adapted original numerical validation; no external source imports.
import unittest

from verl_speco.ops.deepseek_v4_moe import plan_budgeted_expert_groups


class BudgetedExpertGroupTests(unittest.TestCase):
    def test_auto_budget_packs_adjacent_experts(self):
        groups, effective_budget = plan_budgeted_expert_groups(
            [8, 0, 2, 3, 7, 1],
            "auto",
            4,
        )
        self.assertEqual(effective_budget, 8)
        self.assertEqual(
            groups,
            [
                (0, 1, 0, 8),
                (2, 4, 8, 13),
                (4, 6, 13, 21),
            ],
        )

    def test_budget_resolves_up_to_busiest_expert(self):
        groups, effective_budget = plan_budgeted_expert_groups(
            [3, 11, 2],
            4,
            16,
        )
        self.assertEqual(effective_budget, 11)
        self.assertEqual(
            groups,
            [(0, 1, 0, 3), (1, 2, 3, 14), (2, 3, 14, 16)],
        )

    def test_expert_limit_counts_zero_experts_in_weight_slice(self):
        groups, effective_budget = plan_budgeted_expert_groups(
            [2, 0, 0, 2],
            8,
            3,
        )
        self.assertEqual(effective_budget, 8)
        self.assertEqual(groups, [(0, 1, 0, 2), (3, 4, 2, 4)])

    def test_empty_routes_produce_no_groups(self):
        groups, effective_budget = plan_budgeted_expert_groups(
            [0, 0],
            "auto",
            8,
        )
        self.assertEqual(groups, [])
        self.assertEqual(effective_budget, 1)


if __name__ == "__main__":
    unittest.main()
