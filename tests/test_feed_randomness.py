import unittest

from shopping_grpo.feed.randomness import CommonRandomNumbers, sigmoid


class FeedRandomnessTest(unittest.TestCase):
    def test_draws_are_addressable_and_reproducible(self):
        first = CommonRandomNumbers(7, "episode-a")
        second = CommonRandomNumbers(7, "episode-a")
        self.assertEqual(first.uniform(3, "click", "p1"), second.uniform(3, "click", "p1"))
        self.assertNotEqual(first.uniform(3, "click", "p1"), first.uniform(3, "cart", "p1"))

    def test_branch_specific_calls_do_not_shift_other_channels(self):
        factual = CommonRandomNumbers(11, "episode")
        counterfactual = CommonRandomNumbers(11, "episode")
        _ = factual.uniform(2, "branch-only")
        self.assertEqual(
            factual.uniform(9, "refund", "p2"),
            counterfactual.uniform(9, "refund", "p2"),
        )

    def test_helpers_stay_in_contract(self):
        table = CommonRandomNumbers("seed")
        self.assertTrue(2 <= table.integer(2, 4, 0, "delay") <= 4)
        self.assertLess(sigmoid(-5), sigmoid(0))
        self.assertLess(sigmoid(0), sigmoid(5))


if __name__ == "__main__":
    unittest.main()
