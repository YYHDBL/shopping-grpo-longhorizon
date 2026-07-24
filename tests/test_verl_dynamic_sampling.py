"""Unit tests for the project-side reward-group filter."""

import unittest

from shopping_grpo.verl_dynamic_sampling import select_reward_varying_groups


class RewardGroupSelectionTest(unittest.TestCase):
    def test_all_zero_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0, 0, 0, 0])
        self.assertEqual(indices, [])
        self.assertEqual(stats["dropped_uids"], ("a",))

    def test_all_one_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [1, 1, 1, 1])
        self.assertEqual(indices, [])
        self.assertEqual(stats["kept_group_count"], 0)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
