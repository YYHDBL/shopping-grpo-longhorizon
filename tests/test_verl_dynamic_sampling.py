"""Unit tests for the project-side reward-group filter."""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    append_training_diagnostic,
    build_rollout_diagnostics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


class RewardGroupSelectionTest(unittest.TestCase):
    def test_training_diagnostics_append_public_rollouts_as_jsonl(self):
        rollouts = build_rollout_diagnostics(
            ["task-a", "task-a"],
            [
                {
                    "task_id": 7,
                    "actions": [
                        {"tool": "search", "parameters": {"query": "shoe"}}
                    ],
                },
                {"task_id": 7, "termination_reason": "max_steps"},
            ],
        )
        self.assertEqual([item["rollout_index"] for item in rollouts], [0, 1])
        self.assertEqual(rollouts[0]["actions"][0]["tool"], "search")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics" / "training.jsonl"
            append_training_diagnostic(
                path,
                "generation_batch",
                3,
                rollouts=rollouts,
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["event"], "generation_batch")
        self.assertEqual(record["global_step"], 3)
        self.assertEqual(record["rollouts"][1]["termination_reason"], "max_steps")

    def test_all_zero_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0, 0, 0, 0])
        self.assertEqual(indices, [])
        self.assertEqual(stats["dropped_uids"], ("a",))
        self.assertEqual(stats["all_zero_utility_group_count"], 1)
        self.assertEqual(stats["all_purchase_success_group_count"], 0)

    def test_all_one_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [1, 1, 1, 1],
            terminal_utilities=[1.0, 1.0, 1.0, 1.0],
            purchase_success=[True] * 4,
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["kept_group_count"], 0)
        self.assertEqual(stats["all_purchase_success_group_count"], 1)

    def test_fractional_reward_variance_is_kept(self):
        rewards = [2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(["a"] * 4, rewards)
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["kept_uids"], ("a",))

    def test_mixed_uids_preserve_trajectory_indices(self):
        uids = ["a", "b", "a", "b", "a", "b", "a", "b"]
        rewards = [0, 2 / 7, 0, 4 / 7, 0, 2 / 7, 0, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [1, 3, 5, 7])
        self.assertEqual(stats["kept_uids"], ("b",))
        self.assertEqual(stats["dropped_uids"], ("a",))

    def test_zero_and_varying_groups_keep_only_varying_group(self):
        uids = ["zero"] * 4 + ["signal"] * 4
        rewards = [0, 0, 0, 0, 2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [4, 5, 6, 7])
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["dropped_group_count"], 1)

    def test_tolerance_treats_tiny_roundoff_as_constant(self):
        indices, _ = select_reward_varying_groups(
            ["a"] * 4,
            [0.5, 0.5 + 1.0e-9, 0.5, 0.5],
            tolerance=1.0e-8,
        )
        self.assertEqual(indices, [])

    def test_varying_terminal_utility_is_kept_without_purchase_success(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.85, -0.65, -0.50, -0.35],
            terminal_utilities=[-0.85, -0.65, -0.50, -0.35],
            purchase_success=[False] * 4,
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])
        self.assertEqual(stats["no_purchase_success_group_count"], 1)

    def test_varying_group_with_purchase_success_is_kept(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.5, 0.55, -0.5, -0.5],
            terminal_utilities=[-0.5, 0.55, -0.5, -0.5],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])

    def test_sampling_invalid_member_drops_the_whole_group_with_reason(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 0.2, 0.0, 0.0],
            terminal_utilities=[0.0, 0.2, 0.0, 0.0],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False, True, False, False],
            sampling_invalid_reasons=[(), ("infrastructure_invalid",), (), ()],
        )

        self.assertEqual(indices, [])
        self.assertEqual(stats["groups"][0]["drop_reason"], "sampling_invalid")
        self.assertEqual(stats["sampling_invalid_group_count"], 1)
        self.assertEqual(
            stats["sampling_invalid_reason_counts"]["infrastructure_invalid"],
            1,
        )

    def test_shopping_extra_fields_are_reduced_to_filter_signals(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward": {
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                    },
                },
                {
                    "infrastructure_invalid": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                    },
                },
            ]
        )

        self.assertEqual(utility, [0.55, 0.0])
        self.assertEqual(success, [True, False])
        self.assertEqual(invalid, [False, True])
        self.assertEqual(reasons, [(), ("infrastructure_invalid",)])

    def test_unverifiable_reward_is_sampling_invalid_but_not_infrastructure(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward_unverifiable": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                    },
                }
            ]
        )
        self.assertEqual(utility, [0.0])
        self.assertEqual(success, [False])
        self.assertEqual(invalid, [True])
        self.assertEqual(reasons, [("reward_unverifiable",)])

    def test_missing_shopping_filter_signal_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shopping"):
            extract_shopping_group_signals([None])

    def test_shopping_metrics_are_aggregated_for_a0_and_a1(self):
        infos = [
            {
                "steps": 10,
                "done": True,
                "termination_reason": "environment_done",
                "infrastructure_invalid": False,
                "reward": {
                    "full": 1.0,
                    "strict": 1.0,
                    "native": 1.0,
                    "semantic": 1.7,
                    "total": 1.73,
                    "efficiency": 0.03,
                    "penalty_overlong": 0.0,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "terminal_utility": 1.73,
                    "purchase_success": 1.0,
                    "sampling_invalid": False,
                    "r_type": 1.0,
                    "r_att": 1.0,
                    "r_option": 1.0,
                    "r_price": 1.0,
                },
            },
            {
                "steps": 35,
                "done": False,
                "termination_reason": "max_steps",
                "infrastructure_invalid": False,
                "reward": {
                    "full": 0.0,
                    "strict": 0.0,
                    "native": 0.0,
                    "semantic": 0.0,
                    "total": -0.05,
                    "efficiency": 0.0,
                    "penalty_overlong": 0.05,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "terminal_utility": -0.05,
                    "purchase_success": 0.0,
                    "sampling_invalid": False,
                    "r_type": 0.0,
                    "r_att": 0.0,
                    "r_option": 0.0,
                    "r_price": 0.0,
                },
            },
        ]

        metrics = aggregate_shopping_metrics(infos)

        self.assertEqual(metrics["reward/full_mean"], 0.5)
        self.assertEqual(metrics["reward/shaped_min"], -0.05)
        self.assertEqual(metrics["reward/shaped_max"], 1.73)
        self.assertEqual(metrics["reward/purchase_success_rate"], 0.5)
        self.assertEqual(metrics["component/r_type_mean"], 0.5)
        self.assertEqual(metrics["trajectory/average_steps"], 22.5)
        self.assertEqual(metrics["trajectory/done_rate"], 0.5)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
