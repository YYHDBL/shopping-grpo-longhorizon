import unittest

from shopping_grpo.verl_shop_context import make_initial_state


class VerlShopContextTest(unittest.TestCase):
    def test_make_initial_state(self):
        state = make_initial_state(task_id=3)

        self.assertEqual(state["task_id"], 3)
        self.assertEqual(state["total_reward"], 0.0)
        self.assertEqual(state["num_tool_calls"], 0)
        self.assertFalse(state["done"])
        self.assertEqual(state["action_history"], [])


if __name__ == "__main__":
    unittest.main()
