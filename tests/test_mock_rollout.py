import unittest

from scripts.run_mock_shop_rollout import build_mock_actions


class MockRolloutTest(unittest.TestCase):
    def test_mock_actions_are_tool_calls(self):
        actions = build_mock_actions(query="乳胶枕")

        self.assertEqual(actions[0]["name"], "search_products")
        self.assertIn("parameters", actions[0])


if __name__ == "__main__":
    unittest.main()
